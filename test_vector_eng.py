import torch 

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as ni
import numpy as np
import torch 
import time
import neuronxcc.nki.isa as nisa
from torch_xla.core import xla_model as xm
import os
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
  """NKI kernel to compute a 128x128x512 matrix multiplication operation

  Args:
      lhsT: an input tensor of shape [128,128], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
      rhs: an input tensor of shape [128,512], a right hand side argument of the
        matrix multiplication
  Returns:
      result: the resulting output tensor of shape [128,512]
  """
  result = nl.ndarray((128, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  i_lhsT_p, i_lhsT_f = nl.mgrid[0:128, 0:128]
  i_rhs_p, i_rhs_f = nl.mgrid[0:128, 0:512]
  i_out_p, i_out_f = nl.mgrid[0:128, 0:512]

 
  lhs_tile = nl.load(lhsT[i_lhsT_p, i_lhsT_f])
  rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])

  
  result_psum = nisa.nc_matmul(lhs_tile, rhs_tile)

 
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)
  nl.store(result[i_out_p, i_out_f], value=result_sbuf)

  return result


@nki.jit
def nki_vector_matmul(lhs, rhs):
    """NKI kernel to compute a 1280x2048x1 matrix multiplication operation
    
    Args:
        lhs: an input tensor of shape [1,2048], a left hand side argument of the
            matrix multiplication
        rhs: an input tensor of shape [1280,2048], a right hand side argument of the
            matrix multiplication
    Returns:
        result: the resulting output tensor of shape [1280,1]
    """
    
    
    i_lhs_p, i_lhs_f = nl.mgrid[0:1, 0:2048]
    i_rhs_p, i_rhs_f = nl.mgrid[0:1280, 0:2048]
    i_out_p, i_out_f = nl.mgrid[0:1280, 0:1]
    
    M, N = rhs.shape
    result = nl.ndarray((1, N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    i_rhs = nl.mgrid[0:128, 0:N]
    rhs_tiles = nl.ndarray((nl.par_dim(128), M // 128, N), dtype=rhs.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(M // 128):
                rhs_tiles[i_rhs.p, i, i_rhs.x] = nl.load(
                    rhs[ i * 128 + i_rhs.p, i_rhs.x]
                )
    
    
    
     
    lhs_tile = nl.load(lhs[i_lhs_p, i_lhs_f])
    rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])
    
    
    result_psum = nisa.nc_vec_matmul(rhs_tile, lhs_tile)
    
     
    result_sbuf = nl.copy(result_psum, dtype=result.dtype)
    nl.store(result[i_out_p, i_out_f], value=result_sbuf)
    
    return result

if __name__ == "__main__":
  K, M, N = 128, 128, 512
  device = xm.xla_device()
  cpu = torch.device('cpu')

#   A = torch.rand((M, K), dtype=torch.bfloat16, device=device)
#   B = torch.rand((K, N), dtype=torch.bfloat16, device=device)


  A = torch.rand((1, 2048), dtype=torch.bfloat16, device=device)
  B = torch.rand((1280, 2048), dtype=torch.bfloat16, device=device)
  
  output_nki = nki_vector_matmul(A, B)
#   for _ in range(100):
#     start_time = time.time()
#     output_nki = nki_matmul_basic_(A.T, B)
#     xm.mark_step()
#     xm.wait_device_ops()
#     end_time = time.time()
#     print(f"output_nki={output_nki}")
#     print("nki matmul time (ms): ", (end_time - start_time ) * 1000)
