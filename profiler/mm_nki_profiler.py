from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as ni
import numpy as np
import torch 
import neuronxcc.nki.isa as nisa
from torch_xla.core import xla_model as xm
import os, math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np
import argparse
from scipy.special import softmax

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "


''' command: 

rm -rf /var/tmp/neuron-compile-cache/
rm *ntff *neff *pb
python mm_nki_profiler.py 
neuron-profile capture -n MODULE_SyncTensorsGraph.8_13759701792771451712.neff -s profile_8.ntff --profile-nth-exec=2
neuron-profile view -n MODULE_SyncTensorsGraph.8_13759701792771451712.neff -s profile_8_exec_2.ntff 
'''


@nki.jit
def fused_three_mm_xuv_transpose_block(X, U, V, mixed_precision=True,
                                       TILES_IN_BLOCK_N=4, 
                                       TILES_IN_BLOCK_M=1, 
                                       TILES_IN_BLOCK_D=2):
 
  """NKI kernel to compute VT @ (UT @ XT)

  Args:
      X: [K, M]
      U: [K, N]
      V: [N, D]
  Returns:
      out_ref: [D, M]
  """
  

  kernel_dtype = X.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert X.dtype == U.dtype == V.dtype

  K, M = X.shape
  K, N = U.shape
  N, D = V.shape

  out_ref = nl.ndarray((D, M), dtype=V.dtype, buffer=nl.hbm)

  K_n_tiles, K_tile_size = K // 128, 128
  N_n_tiles, N_tile_size = N // 128, 128
  N_block_size = N_tile_size * TILES_IN_BLOCK_N
  N_n_blocks = N // N_block_size # Mapping between block and tile: N_n_blocks * TILES_IN_BLOCK_N = N_n_tiles
  
  M_tile_size = 512
  M_block_size = M_tile_size * TILES_IN_BLOCK_M
  if M_block_size < M_tile_size:
    M_tile_size = M_block_size
    TILES_IN_BLOCK_M = 1

  M_n_blocks = M // M_block_size

  D_tile_size = 128
  D_block_size = D_tile_size * TILES_IN_BLOCK_D
  D_n_blocks = D // D_block_size

  ########################################################

  ip_q = nl.arange(K_tile_size)[:, None]
  if_q_tile = nl.arange(M_tile_size)[None, :]
  if_q_block = nl.arange(M_block_size)[None, :]

  ip_v = nl.arange(N_tile_size)[:, None]
  if_v_block = nl.arange(D_block_size)[None, :]

  ip_k = nl.arange(K_tile_size)[:, None]
  if_k_tile = nl.arange(N_tile_size)[None, :]
  if_k_block = nl.arange(N_block_size)[None, :]
  
  ######################################################## UT @ XT
  for i_m_block in nl.affine_range(M_n_blocks):  
    xu_res_buf = nl.ndarray((par_dim(N_tile_size), N_n_tiles, M_block_size), dtype=kernel_dtype)

    x_cache = nl.ndarray((K_n_tiles, par_dim(K_tile_size), M_block_size), dtype=pe_in_dt)
    for i_k_tile in nl.affine_range(K_n_tiles):  
        x_cache[i_k_tile, ip_q, if_q_block] = nl.load(X[
            i_k_tile * K_tile_size + ip_q, i_m_block * M_block_size + if_q_block],
          dtype=pe_in_dt)
    

    for i_n_block in nl.affine_range(N_n_blocks):

      u_cache = nl.ndarray((K_n_tiles, par_dim(K_tile_size), N_block_size), dtype=pe_in_dt)
      for i_k_tile in nl.affine_range(K_n_tiles):
        u_cache[i_k_tile, ip_k, if_k_block] = nl.load(U[
            i_k_tile * K_tile_size + ip_k, i_n_block * N_block_size + if_k_block],
          dtype=pe_in_dt) 
      
      for ib_m_tile in nl.affine_range(TILES_IN_BLOCK_M):
        # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=K)
        for ib_n_tile in nl.affine_range(TILES_IN_BLOCK_N):  # indent = 4
          # Since the K^T tile is the RHS, the m_len dimension will be P in the result
          # PSUM buffer shape: [M_tile_size P, N_tile_size F]
          xu_psum = nl.zeros((par_dim(N_tile_size), M_tile_size), dtype=np.float32, buffer=nl.psum)

          # Tensor indices for accessing xu result in N_tile_size
          if_xu = nl.arange(M_tile_size)[None, :]
          ip_xu = nl.arange(N_tile_size)[:, None]

          # Loop over contraction dim of Step 1 matmul
          for i_k_tile in nl.affine_range(K_n_tiles):  # indent = 6
            ##############################################################
            # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
            ##############################################################
            xu_psum[ip_xu, if_xu] += nisa.nc_matmul(moving=x_cache[i_k_tile, ip_q, if_q_tile + ib_m_tile * M_tile_size],
                                                    stationary=u_cache[i_k_tile, ip_k, if_k_tile + ib_n_tile * N_tile_size])

          xu_res_buf[ip_xu, i_n_block * TILES_IN_BLOCK_N + ib_n_tile, if_xu + ib_m_tile * M_tile_size] = nl.copy(
              xu_psum[ip_xu, if_xu],
              dtype=kernel_dtype)



    ######################################################## VT @ (UT @ XT)
    for i_d_block in nl.affine_range(D_n_blocks):
      
      v_cache = nl.ndarray((par_dim(N_tile_size), N_n_tiles, D_block_size), dtype=pe_in_dt)
      for i_n_tile in nl.affine_range(N_n_tiles):
        v_cache[ip_v, i_n_tile, if_v_block] = nl.load(V[
          i_n_tile * N_tile_size + ip_v, i_d_block * D_block_size + if_v_block],
        dtype=pe_in_dt)

      for ib_d_tile in nl.affine_range(TILES_IN_BLOCK_D):
        final_res_buf = nl.ndarray((par_dim(D_tile_size), M_block_size), dtype=kernel_dtype)
        
        if_out = nl.arange(M_tile_size)[None, :]
        ip_out = nl.arange(D_tile_size)[:, None]
        for ib_m_tile in nl.affine_range(TILES_IN_BLOCK_M):
          # Result psum buffer has the hidden dim as P
          res_psum = nl.zeros((par_dim(D_tile_size), M_tile_size),dtype=np.float32, buffer=nl.psum)

          ip_scores_t = nl.arange(N_tile_size)[:, None]
          if_scores_t = nl.arange(M_tile_size)[None, :]

          for i_n_tile in nl.affine_range(N_n_tiles):
            ######################################################################
            # Step 3. matmul_1(moving=trans_v, stationary=xu_res_buf, contract=N=N)
            ######################################################################
            ip_v_t = nl.arange(N_tile_size)[:, None]
            if_v_t = nl.arange(D_tile_size)[None, :]
            res_psum[ip_out, if_out] += \
              nisa.nc_matmul(moving=xu_res_buf[ip_scores_t, i_n_tile, if_scores_t + ib_m_tile * M_tile_size],
                            stationary=v_cache[ip_v_t, i_n_tile, if_v_t + ib_d_tile * D_tile_size])

          final_res_buf[ip_out, ib_m_tile * M_tile_size + if_out] = nl.copy(res_psum[ip_out, if_out], dtype=kernel_dtype)

        nl.store(
          out_ref[
              i_d_block * D_block_size + ib_d_tile * D_tile_size + ip_out, i_m_block * M_block_size + if_q_block],
          value=final_res_buf)
  return out_ref



@nki.jit
def fused_three_mm_xuv_transpose_block2(X, U, V, mixed_precision=True,
                                       TILES_IN_BLOCK_N=4, 
                                       TILES_IN_BLOCK_M=1, 
                                       TILES_IN_BLOCK_D=2):
 
  """NKI kernel to compute VT @ (UT @ XT)

  Args:
      X: [K, M]
      U: [K, N]
      V: [N, D]
  Returns:
      out_ref: [D, M]
  """
  

  kernel_dtype = X.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert X.dtype == U.dtype == V.dtype

  K, M = X.shape
  K, N = U.shape
  N, D = V.shape

  out_ref = nl.ndarray((D, M), dtype=V.dtype, buffer=nl.hbm)

  K_n_tiles, K_tile_size = K // 128, 128
  N_n_tiles, N_tile_size = N // 128, 128
  N_block_size = N_tile_size * TILES_IN_BLOCK_N
  N_n_blocks = N // N_block_size # Mapping between block and tile: N_n_blocks * TILES_IN_BLOCK_N = N_n_tiles
  
  M_tile_size = 512
  M_block_size = M_tile_size * TILES_IN_BLOCK_M
  if M_block_size < M_tile_size:
    M_tile_size = M_block_size
    TILES_IN_BLOCK_M = 1

  M_n_blocks = M // M_block_size

  D_tile_size = 128
  D_block_size = D_tile_size * TILES_IN_BLOCK_D
  D_n_blocks = D // D_block_size

  ########################################################

  ip_q = nl.arange(K_tile_size)[:, None]
  if_q_tile = nl.arange(M_tile_size)[None, :]
  if_q_block = nl.arange(M_block_size)[None, :]

  ip_v = nl.arange(N_tile_size)[:, None]
  if_v_block = nl.arange(D_block_size)[None, :]

  ip_k = nl.arange(K_tile_size)[:, None]
  if_k_tile = nl.arange(N_tile_size)[None, :]
  if_k_block = nl.arange(N_block_size)[None, :]
  
  ######################################################## UT @ XT
  for i_m_block in nl.affine_range(M_n_blocks):  
    xu_res_buf = nl.ndarray((par_dim(N_tile_size), TILES_IN_BLOCK_N, M_block_size), dtype=kernel_dtype)

    x_cache = nl.ndarray((K_n_tiles, par_dim(K_tile_size), M_block_size), dtype=pe_in_dt)
    for i_k_tile in nl.affine_range(K_n_tiles):  
        x_cache[i_k_tile, ip_q, if_q_block] = nl.load(X[
            i_k_tile * K_tile_size + ip_q, i_m_block * M_block_size + if_q_block],
          dtype=pe_in_dt)
    

    for i_n_block in nl.affine_range(N_n_blocks):

      u_cache = nl.ndarray((K_n_tiles, par_dim(K_tile_size), N_block_size), dtype=pe_in_dt)
      for i_k_tile in nl.affine_range(K_n_tiles):
        u_cache[i_k_tile, ip_k, if_k_block] = nl.load(U[
            i_k_tile * K_tile_size + ip_k, i_n_block * N_block_size + if_k_block],
          dtype=pe_in_dt) 
      
      for ib_m_tile in nl.affine_range(TILES_IN_BLOCK_M):
        # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=K)
        for ib_n_tile in nl.affine_range(TILES_IN_BLOCK_N):  # indent = 4
          # Since the K^T tile is the RHS, the m_len dimension will be P in the result
          # PSUM buffer shape: [M_tile_size P, N_tile_size F]
          xu_psum = nl.zeros((par_dim(N_tile_size), M_tile_size), dtype=np.float32, buffer=nl.psum)

          # Tensor indices for accessing xu result in N_tile_size
          if_xu = nl.arange(M_tile_size)[None, :]
          ip_xu = nl.arange(N_tile_size)[:, None]

          # Loop over contraction dim of Step 1 matmul
          for i_k_tile in nl.affine_range(K_n_tiles):  # indent = 6
            ##############################################################
            # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
            ##############################################################
            xu_psum[ip_xu, if_xu] += nisa.nc_matmul(moving=x_cache[i_k_tile, ip_q, if_q_tile + ib_m_tile * M_tile_size],
                                                    stationary=u_cache[i_k_tile, ip_k, if_k_tile + ib_n_tile * N_tile_size])

          xu_res_buf[ip_xu, i_n_block * TILES_IN_BLOCK_N + ib_n_tile, if_xu + ib_m_tile * M_tile_size] = nl.copy(
              xu_psum[ip_xu, if_xu],
              dtype=kernel_dtype)



    ######################################################## VT @ (UT @ XT)
    for i_d_block in nl.affine_range(D_n_blocks):
      
      v_cache = nl.ndarray((par_dim(N_tile_size), N_n_tiles, D_block_size), dtype=pe_in_dt)
      for i_n_tile in nl.affine_range(N_n_tiles):
        v_cache[ip_v, i_n_tile, if_v_block] = nl.load(V[
          i_n_tile * N_tile_size + ip_v, i_d_block * D_block_size + if_v_block],
        dtype=pe_in_dt)

      for ib_d_tile in nl.affine_range(TILES_IN_BLOCK_D):
        final_res_buf = nl.ndarray((par_dim(D_tile_size), M_block_size), dtype=kernel_dtype)
        
        if_out = nl.arange(M_tile_size)[None, :]
        ip_out = nl.arange(D_tile_size)[:, None]
        for ib_m_tile in nl.affine_range(TILES_IN_BLOCK_M):
          # Result psum buffer has the hidden dim as P
          res_psum = nl.zeros((par_dim(D_tile_size), M_tile_size),dtype=np.float32, buffer=nl.psum)

          ip_scores_t = nl.arange(N_tile_size)[:, None]
          if_scores_t = nl.arange(M_tile_size)[None, :]

          for i_n_tile in nl.affine_range(N_n_tiles):
            ######################################################################
            # Step 3. matmul_1(moving=trans_v, stationary=xu_res_buf, contract=N=N)
            ######################################################################
            ip_v_t = nl.arange(N_tile_size)[:, None]
            if_v_t = nl.arange(D_tile_size)[None, :]
            res_psum[ip_out, if_out] += \
              nisa.nc_matmul(moving=xu_res_buf[ip_scores_t, i_n_tile, if_scores_t + ib_m_tile * M_tile_size],
                            stationary=v_cache[ip_v_t, i_n_tile, if_v_t + ib_d_tile * D_tile_size])

          final_res_buf[ip_out, ib_m_tile * M_tile_size + if_out] = nl.copy(res_psum[ip_out, if_out], dtype=kernel_dtype)

        nl.store(
          out_ref[
              i_d_block * D_block_size + ib_d_tile * D_tile_size + ip_out, i_m_block * M_block_size + if_q_block],
          value=final_res_buf)
  return out_ref



if __name__ == "__main__":

  
  M, K, N = 4096, 7168, 18432
  device = xm.xla_device()
  cpu = torch.device('cpu')

  A = torch.rand((M, K), dtype=torch.bfloat16, device=device)
  B = torch.rand((K, N), dtype=torch.bfloat16, device=device)
  
  compress_ratio = 0.8
  r = math.ceil((K * N * compress_ratio) // ((K + N) * 128)) * 128

  U = torch.rand((K, r), dtype=torch.bfloat16, device=device)
  V = torch.rand((r, N), dtype=torch.bfloat16, device=device)

  xm.mark_step()
  xm.wait_device_ops()

  output_torch1 = torch.matmul(torch.matmul(A, U), V) 
  output_torch1 = output_torch1.T
  
  xm.mark_step()
  xm.wait_device_ops()


  output_torch2 = torch.einsum('bm,mn,np->bp', A, U, V) 
  output_torch2 = output_torch2.T

  xm.mark_step()
  xm.wait_device_ops()
  

  output_nki = fused_three_mm_xuv_transpose_block(A.T, U, V)

  xm.mark_step()
  xm.wait_device_ops()
  



  output_torch = output_torch1.to(device=cpu)
  output_nki = output_nki.to(device=cpu)
  if torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2):
      print("NKI and Torch match")
  else:
      print("NKI and Torch differ")

  output_torch1 = output_torch1.to(device=cpu)
  output_torch2 = output_torch2.to(device=cpu)
  if torch.allclose(output_torch1, output_torch2, atol=1e-4, rtol=1e-2):
      print("Torch1 and Torch2 match")
  else:
      print("Torch1 and Torch2 differ")

 