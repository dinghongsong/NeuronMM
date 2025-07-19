import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np
import argparse
from scipy.special import softmax



@nki.jit
def XUV_matmul(x_ref, u_ref, v_ref, mixed_precisioin=True):
    # TODO: Fix tiles padding
    kernel_dtype = x_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precisioin else np.float32
    assert x_ref.dtype == u_ref.dtype == v_ref.dtype

   
    m, n = x_ref.shape
    n, k = u_ref.shape
    k, d = v_ref.shape

    
    ###################
    out_ref = nl.ndarray((m, d), dtype=x_ref.dtype, buffer=nl.shared_hbm)

    x_p_n_tiles, x_p_tile_size, x_f_n_tiles, x_f_tile_size = n // 128, 128, m // 128, 128
    u_p_n_tiles, u_p_tile_size, u_f_n_tiles, u_f_tile_size = n // 128, 128, k // 128, 128
    v_p_n_tiles, v_p_tile_size, v_f_n_tiles, v_f_tile_size = k // 128, 128, d // 512, 512

    ###################### load tensor to SBUF from HBM
    

    x_local = nl.ndarray((x_p_n_tiles, x_f_n_tiles, par_dim(x_p_tile_size), x_f_tile_size), dtype=pe_in_dt) 
    ip_x = nl.arange(x_p_tile_size)[:, None]
    if_x = nl.arange(x_f_tile_size)[None, :]
    for i_f_tile in nl.affine_range(x_f_n_tiles):
        for j_p_tile in nl.affine_range(x_p_n_tiles):
            x_local[j_p_tile, i_f_tile, ip_x, if_x] = nl.load_transpose2d(
                x_ref[i_f_tile * x_f_tile_size + nl.arange(x_f_tile_size)[:, None], 
                      j_p_tile * x_p_tile_size + nl.arange(x_p_tile_size)[None, :]],
                dtype=pe_in_dt
            ) 

    u_local = nl.ndarray((u_p_n_tiles, u_f_n_tiles, par_dim(u_p_tile_size), u_f_tile_size), dtype=pe_in_dt) 
    ip_u = nl.arange(u_p_tile_size)[:, None]
    if_u = nl.arange(u_f_tile_size)[None, :]
    for i_p_tile in nl.affine_range(u_p_n_tiles):
        for j_f_tile in nl.affine_range(u_f_n_tiles):
            u_local[i_p_tile, j_f_tile, ip_u, if_u] = nl.load(
                u_ref[i_p_tile * u_p_tile_size + nl.arange(u_p_tile_size)[:, None], 
                      j_f_tile * u_f_tile_size + nl.arange(u_f_tile_size)[None, :]],
                dtype=pe_in_dt
            )
    
    v_local = nl.ndarray((v_p_n_tiles, v_f_n_tiles, par_dim(v_p_tile_size), v_f_tile_size), dtype=pe_in_dt) 
    ip_v = nl.arange(v_p_tile_size)[:, None]
    if_v = nl.arange(v_f_tile_size)[None, :]
    for i_p_tile in nl.affine_range(v_p_n_tiles):
        for j_f_tile in nl.affine_range(v_f_n_tiles):
            v_local[i_p_tile, j_f_tile, ip_v, if_v] = nl.load(
                v_ref[i_p_tile * v_p_tile_size + nl.arange(v_p_tile_size)[:, None], 
                      j_f_tile * v_f_tile_size + nl.arange(v_f_tile_size)[None, :]],
                dtype=pe_in_dt
            )
    
    ##################### X @ U @ V

    for i_x_f_tile in nl.affine_range(x_f_n_tiles):

        xu_res_buf = nl.ndarray((x_f_tile_size, k), dtype=kernel_dtype)
        ip_xu = nl.arange(x_f_tile_size)[:, None]
        if_xu = nl.arange(u_f_tile_size)[None, :]


        for i_u_f_tile in nl.affine_range(u_f_n_tiles):

            xu_psum = nl.zeros((x_f_tile_size, u_f_tile_size), dtype=nl.float32, buffer=nl.psum)
            for i_x_p_tile in nl.affine_range(x_p_n_tiles):
                xu_psum[ip_xu, if_xu] += nisa.nc_matmul(stationary=x_local[i_x_p_tile, i_x_f_tile, ip_x, if_x],
                                                        moving=u_local[i_x_p_tile, i_u_f_tile, ip_u, if_u])
            xu_res_buf[ip_xu, i_u_f_tile * u_f_tile_size + if_xu] = nl.copy(xu_psum[ip_xu, if_xu], dtype=kernel_dtype)
                
        trans_xu_res_buf = nl.ndarray((par_dim(u_f_tile_size), u_f_n_tiles, x_f_tile_size), dtype=pe_in_dt) 
        ip_score_t = nl.arange(u_f_tile_size)[:, None]
        if_score_t = nl.arange(x_f_tile_size)[None, :]
        ip_scores = nl.arange(x_f_tile_size)[:, None]
        if_scores = nl.arange(u_f_tile_size)[None, :]
        for i_u_f_tile in nl.affine_range(u_f_n_tiles):
            trans_xu_res_buf[ip_score_t, i_u_f_tile, if_score_t] = nisa.nc_transpose(xu_res_buf[ip_scores, i_u_f_tile * u_f_tile_size + if_scores])
        

        xuv_res_buf = nl.ndarray((x_f_tile_size, d), dtype=kernel_dtype)
        ip_xuv = nl.arange(x_f_tile_size)[:, None]
        if_xuv = nl.arange(v_f_tile_size)[None, :]

        for i_v_f_tile in nl.affine_range(v_f_n_tiles):
            
            xuv_psum = nl.zeros((x_f_tile_size, v_f_tile_size), dtype=nl.float32, buffer=nl.psum)
            ip_out = nl.arange(x_f_tile_size)[:, None]
            if_out = nl.arange(v_f_tile_size)[None, :]
            ip_v=nl.arange(v_p_tile_size)[:, None]
            if_v=nl.arange(v_f_tile_size)[None, :]

            for i_u_f_tile in nl.affine_range(u_f_n_tiles): 
                xuv_psum[ip_out, if_out] += nisa.nc_matmul(stationary=trans_xu_res_buf[ip_score_t, i_u_f_tile, if_score_t],
                                                            moving=v_local[i_u_f_tile, i_v_f_tile, ip_v, if_v]) 
            xuv_res_buf[ip_xuv, i_v_f_tile * v_f_tile_size + if_xuv] = nl.copy(xuv_psum[ip_out, if_out], dtype=kernel_dtype)
        
        nl.store(out_ref[i_x_f_tile * x_f_tile_size + ip_out, nl.arange(d)[None, :]], value=xuv_res_buf)
      
    return out_ref


@nki.jit
def nki_matmul_tiled_basic(lhsT, rhs):
  """
  A basic NKI matrix multiplication kernel that uses tiling.

  This kernel can handle large matrices that satisfy specific size
  multiple requirements.
  - lhsT: K and M dimensions must be multiples of 128.
  - rhs: N dimension must be a multiple of 512.

  Args:
      lhsT: The left-hand side operand, which is the transpose of A,
            with shape [K, M].
      rhs: The right-hand side operand, B, with shape [K, N].
  Returns:
      result: The result matrix D, with shape [M, N].
  """
  # --- 1. SETUP PHASE: Define tile sizes and get matrix dimensions ---
  K, M = lhsT.shape
  K_rhs, N = rhs.shape
  assert K == K_rhs, "The contraction dimension K must match for LHS and RHS"

  # Define the size of a "Tile", the basic unit processed by the hardware.
  # This corresponds to the required size multiples for the dimensions.
  TILE_M = 128
  TILE_K = 128
  TILE_N = 128

  # Check if input dimensions meet the requirements
  assert M % TILE_M == 0, f"Dimension M({M}) must be a multiple of {TILE_M}"
  assert K % TILE_K == 0, f"Dimension K({K}) must be a multiple of {TILE_K}"
  assert N % TILE_N == 0, f"Dimension N({N}) must be a multiple of {TILE_N}"

  # Calculate the number of tiles in each dimension
  NUM_TILES_M = M // TILE_M
  NUM_TILES_K = K // TILE_K
  NUM_TILES_N = N // TILE_N

  # Define the final output tensor in the main memory (HBM)
#   result = nl.zeros((M, N), dtype=lhsT.dtype, buffer=nl.hbm)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.hbm)

  # --- 2. TILING LOOPS: M -> N -> K ---
  # Iterate over each tile of the output matrix
  for m_tile_idx in nl.affine_range(NUM_TILES_M):
    for n_tile_idx in nl.affine_range(NUM_TILES_N):

      # Create an accumulator for the current output tile (result[m, n]).
      # It resides in PSUM for efficient accumulation operations.
      result_tile_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

      # Reduction loop: Iterate over all tiles along the K dimension.
      for k_tile_idx in nl.sequential_range(NUM_TILES_K):
        
        # Calculate the offset for the current tile in the large matrix
        m_offset = m_tile_idx * TILE_M
        n_offset = n_tile_idx * TILE_N
        k_offset = k_tile_idx * TILE_K

        # a. Load one tile of LHS from HBM to SBUF
        lhs_tile_sbuf = nl.load(
            lhsT[k_offset : k_offset + TILE_K,
                 m_offset : m_offset + TILE_M]
        )

        # b. Load one tile of RHS from HBM to SBUF
        rhs_tile_sbuf = nl.load(
            rhs[k_offset : k_offset + TILE_K,
                n_offset : n_offset + TILE_N]
        )

        # c. Perform matmul on the currently loaded tiles and accumulate the result
        #    into the PSUM accumulator.
        result_tile_psum[...] += nl.matmul(lhs_tile_sbuf, rhs_tile_sbuf, transpose_x=True)

      # d. After the reduction loop over K finishes, the PSUM accumulator
      #    holds the final result for this tile.
      # Write this final tile result back to the correct location in HBM.
      nl.store(
          result[m_tile_idx * TILE_M : (m_tile_idx + 1) * TILE_M,
                 n_tile_idx * TILE_N : (n_tile_idx + 1) * TILE_N],
          result_tile_psum
      )

  return result



@nki.jit
def three_mm_unfused(lhsT, rhs_B, rhs_C):

    intermediate = nki_matmul_tiled_basic(lhsT, rhs_B)
    # intermediate =nisa.nc_transpose(intermediate)
    intermediate = intermediate.reshape(
        (intermediate.shape[1], intermediate.shape[0]))

    result = nki_matmul_tiled_basic(intermediate, rhs_C)

    return result



def cpu_golden_attn(x, u, v):
  return np.matmul(np.matmul(x, u), v)


if __name__ == "__main__":

   
    M, N, K, D = 4096, 1280, 4096, 1280  # X[M,N] @ U[N,P] @ V[P,D]

    dtype = np.float32
    x_tensor = np.random.randn(M, N).astype(dtype)
    u_tensor = np.random.randn(N, K).astype(dtype)
    v_tensor = np.random.randn(K, D).astype(dtype)
    output_nki = np.empty((M, D), dtype=dtype)

    output_golden = cpu_golden_attn(x_tensor, u_tensor, v_tensor)


    def benchmark_nki(nki_func, x_tensor, u_tensor, v_tensor):
        
        output_nki = nki_func(x_tensor, u_tensor, v_tensor)
        allclose = np.allclose(output_nki, output_golden, atol=1e-5, rtol=1e-3)
        print(f">>>> match CPU reference? {allclose}")
        # assert allclose, "Accuracy check fails!"

        benchmark_func = nki.benchmark(nki_func,
                                        warmup=10, iters = 100,
                                        save_neff_name='file.neff',
                                        save_trace_name='profile.ntff')
        benchmark_func(x_tensor, u_tensor, v_tensor)

        metrics = benchmark_func.benchmark_result.nc_latency
        print(">>>> benchmark results")
        print("latency.p50 = " + str(metrics.get_latency_percentile(50)))
        print("latency.p99 = " + str(metrics.get_latency_percentile(99)))
        p99 = metrics.get_latency_percentile(99)
        print("Latency: {:.2f} ms (P99)".format(p99 / 1000.0))



    print("Benchmarking XUV_matmul")
    benchmark_nki(XUV_matmul, x_tensor, u_tensor, v_tensor)
    print('-' * 50)
    print("Benchmarking three_mm_unfused")
    benchmark_nki(three_mm_unfused, x_tensor.T, u_tensor, v_tensor)
    



