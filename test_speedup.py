import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np
import argparse
from scipy.special import softmax
import copy
import gc
import logging
import time
import math
from typing import List, Optional, Tuple, Type
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.language import nc
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
import torch
##############################################
def load_tensor_block(input_tensor, ofs: Tuple[int, int], load_shape: Tuple[int, nl.par_dim, int]):
    """
    Load a 2D rectangle region from the input HBM tensor to SBUF.
    The location of the 2D region is offset by (ofs[0], ofs[1]) at its upper left corner.
    The size of the 2D region to load into SBUF is (block_size * par_size, free_size).
    Load the input HBM tensor by (par_size, free_size) tiles in parallel in the block dimension.
    Output SBUF tensor has a shape of (block_size, par_size, free_size).

    +------------------+
    |                  |
    |    +--------+    |  ← Starting at (ofs[0], ofs[1])
    |    |Tile 0  |    |
    |    |Tile 1  |    |  Each tile is (par_size * free_size)
    |    |  ...   |    |
    |    |Tile N-1|    |  N = block_size
    |    +--------+    |
    |                  |
    +------------------+

    Args:
        input_tensor: the input 2D HBM tensor
        ofs: location offsets in the 2D HBM tensor dimensions
        load_shape: (par_dim(par_size), block_sizeblock_size, free_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """
    assert len(ofs) == 2, f"'ofs' expects (ofs_0, ofs_1). Received {ofs}."
    assert len(load_shape) == 3, f"'load_shape' expects (block, par, free). Received {load_shape}."
    max_rows, max_cols = input_tensor.shape
    load_block_size, load_par_size, load_free_size = load_shape
    tile_index = nl.mgrid[0:load_par_size, 0:load_free_size]
    loaded_tensor = nl.zeros(
        (nl.par_dim(load_par_size), load_block_size, load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )
    for block_id in nl.affine_range(load_block_size):
        row_indices = ofs[0] + block_id * load_par_size + tile_index.p
        col_indices = ofs[1] + tile_index.x
        loaded_tensor[tile_index.p, block_id, tile_index.x] = nl.load(
            input_tensor[row_indices, col_indices], mask=(row_indices < max_rows) & (col_indices < max_cols)
        )
    return loaded_tensor

def load_tensor_block_T(input_tensor, ofs: Tuple[int, int], load_shape: Tuple[int, nl.par_dim, int]):
    """
    Load a 2D rectangle region from the input HBM tensor to SBUF.
    The location of the 2D region is offset by (ofs[0], ofs[1]) at its upper left corner.
    The size of the 2D region to load into SBUF is (block_size * par_size, free_size).
    Load the input HBM tensor by (par_size, free_size) tiles in parallel in the block dimension.
    Output SBUF tensor has a shape of (block_size, par_size, free_size).

    +------------------+
    |                  |
    |    +--------+    |  ← Starting at (ofs[0], ofs[1])
    |    |Tile 0  |    |
    |    |Tile 1  |    |  Each tile is (par_size * free_size)
    |    |  ...   |    |
    |    |Tile N-1|    |  N = block_size
    |    +--------+    |
    |                  |
    +------------------+

    Args:
        input_tensor: the input 2D HBM tensor
        ofs: location offsets in the 2D HBM tensor dimensions
        load_shape: (par_dim(par_size), block_sizeblock_size, free_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """
    assert len(ofs) == 2, f"'ofs' expects (ofs_0, ofs_1). Received {ofs}."
    assert len(load_shape) == 3, f"'load_shape' expects (block, par, free). Received {load_shape}."
    max_rows, max_cols = input_tensor.shape
    load_block_size, load_par_size, load_free_size = load_shape
    tile_index = nl.mgrid[0:load_par_size, 0:load_free_size]
    tile_index_T = nl.mgrid[0:load_free_size, 0:load_par_size]
    loaded_tensor = nl.zeros(
        (nl.par_dim(load_par_size), load_block_size, load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )
    for block_id in nl.affine_range(load_block_size):
        row_indices = ofs[0]  + tile_index_T.p
        col_indices = ofs[1] + tile_index_T.x + block_id * load_par_size
        loaded_tensor[tile_index.p, block_id, tile_index.x] = nl.load_transpose2d(
            input_tensor[row_indices, col_indices], mask=(row_indices < max_rows) & (col_indices < max_cols)
        )
    return loaded_tensor



def get_fused_mlp_up_T_params(S):
    """
    Get optimized parameters for fused_mlp_up_T based on sequence length S.
    
    Args:
        S: Sequence length
        
    Returns:
        dict: Parameter configuration for fused_mlp_up_T
    """
    if S == 1:
        # return {
        #     'M_tiles_in_block': 16, 
        #     'r_tiles_in_block': 4, 
        #     'K_tiles_in_block': 1, 
        #     'N_tiles_in_block': 16
        # }
        return {
            'M_tiles_in_block': 1, 
            'r_tiles_in_block': 1, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 8
        }
    elif S < 128:
        return {
            'M_tiles_in_block': 1, 
            'r_tiles_in_block': 1, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 8
        }
    elif 128 <= S < 512:
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 4, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 4
        }
    elif S == 512:
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 4, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 16
        }
    elif 512 < S < 1024:
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 4, 
            'K_tiles_in_block': 2, 
            'N_tiles_in_block': 1
        }
    elif S == 1024:
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 4, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 4
        }
    else:  # S > 1024
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 4, 
            'K_tiles_in_block': 2, 
            'N_tiles_in_block': 1
        }


def get_fused_three_mm_XTUV_params(S):
    """
    Get optimized parameters for fused_three_mm_XTUV based on sequence length S.
    
    Args:
        S: Sequence length
        
    Returns:
        dict: Parameter configuration for fused_three_mm_XTUV
    """
    if S == 1:
        # return {
        #     'M_tiles_in_block': 16, 
        #     'r_tiles_in_block': 4, 
        #     'K_tiles_in_block': 1, 
        #     'N_tiles_in_block': 4
        # }
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 2, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 4
        }
    elif S <= 128:
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 2, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 4
        }
    elif 128 < S < 256:
        # For range 128 < S < 256, use similar to S <= 128 but slightly optimized
        return {
            'M_tiles_in_block': 8, 
            'r_tiles_in_block': 2, 
            'K_tiles_in_block': 1, 
            'N_tiles_in_block': 4
        }
    elif S == 256:
        return {
            'M_tiles_in_block': 4, 
            'r_tiles_in_block': 2, 
            'K_tiles_in_block': 2, 
            'N_tiles_in_block': 4
        }
    elif 256 < S < 512:
        # For range 256 < S < 512, use similar to S == 256
        return {
            'M_tiles_in_block': 4, 
            'r_tiles_in_block': 2, 
            'K_tiles_in_block': 2, 
            'N_tiles_in_block': 4
        }
    elif S == 512:
        return {
            'M_tiles_in_block': 4, 
            'r_tiles_in_block': 4, 
            'K_tiles_in_block': 4, 
            'N_tiles_in_block': 4
        }
    elif 512 < S < 1024:
        return {
            'M_tiles_in_block': 4, 
            'r_tiles_in_block': 2, 
            'K_tiles_in_block': 4, 
            'N_tiles_in_block': 4
        }
    elif S == 1024:
        return {
            'M_tiles_in_block': 4, 
            'r_tiles_in_block': 4, 
            'K_tiles_in_block': 8, 
            'N_tiles_in_block': 1
        }
    else:  # S > 1024
        return {
            'M_tiles_in_block': 4, 
            'r_tiles_in_block': 2, 
            'K_tiles_in_block': 8, 
            'N_tiles_in_block': 4
        }


def svd_mlp_with_fused_kernel(x, u_up, v_up, u_gate, v_gate, u_down, v_down, 
                             up_T_params=None, XTUV_params=None):
    """
    Implements the SwiGLU block using the mocked fused kernel.
    Follows the logic:
    1. Projections are done with the fused kernel, returning transposed results.
    2. Element-wise operations happen on the transposed results.
    3. The down-projection requires converting the layout back and forth.
    4. The final result is transposed back to the standard layout.
    
    Args:
        x: Input tensor (S, H)
        u_up, v_up: Up projection matrices
        u_gate, v_gate: Gate projection matrices  
        u_down, v_down: Down projection matrices
        up_T_params: Custom parameters for fused_mlp_up_T (optional)
        XTUV_params: Custom parameters for fused_three_mm_XTUV (optional)
    """
    S = x.shape[0]  # Get sequence length
    
    # Use custom parameters if provided, otherwise use auto-selected parameters
    if up_T_params is None:
        up_T_params = get_fused_mlp_up_T_params(S)
    if XTUV_params is None:
        XTUV_params = get_fused_three_mm_XTUV_params(S)
    
    # Calculate 'gate' projection. SiLU is applied inside the kernel.
    # Input: x (S, H). Output: activated_gate_t (I, S)
    activated_gate_t = fused_mlp_up_T(
        x, u_gate, v_gate, u_up, v_up, **up_T_params
    )
    
    # --- Down Projection ---
    
    # Call the kernel for the down projection.
    # Input: tmp_std_layout (S, I). Output: result_t (H, S)
    result = fused_three_mm_XTUV(
        activated_gate_t, u_down, v_down, **XTUV_params
    )
    
    return result

# For mlp down projection
#Input: X^T, U, V
#Ouput: XUV
@nki.jit
def fused_three_mm_XTUV(
    X_ref,  # Shape: (M, K) - stored as transpose
    U_ref,  # Shape: (M, r)
    V_ref,  # Shape: (r, N)
    mixed_precision=True,
    r_tiles_in_block=8,
    K_tiles_in_block=8,
    N_tiles_in_block=2,
    M_tiles_in_block=4
):
    # Use X_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = X_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert X_ref.dtype == U_ref.dtype == V_ref.dtype

    # Shape checking - X@U@V computation
    M, K = X_ref.shape # M - 7168
    M_U, r = U_ref.shape
    r_V, N = V_ref.shape # N - 18432
    assert tuple(U_ref.shape) == (M_U, r), "Input shape mismatch!"
    assert tuple(V_ref.shape) == (r_V, N), "Input shape mismatch!"
    assert M == M_U, "M dimension must match between X and U!"
    assert r == r_V, "r dimension must match between U and V!"

    out_ref = nl.ndarray((K, N), dtype=V_ref.dtype, buffer=nl.hbm)


    # Tiling configuration

    M_tile_size = 128
    M_block_size = M_tile_size * M_tiles_in_block

    if M < M_block_size:
        M_tile_size = min(M, M_tile_size)
        M_tiles_in_block = 1
        M_block_size = M_tile_size * M_tiles_in_block
    
    M_n_blocks = (M + M_block_size - 1) // M_block_size

    r_tile_size = 128
    r_block_size = r_tile_size * r_tiles_in_block
    r_n_blocks = (r + r_block_size - 1) // r_block_size

    K_tile_size = 128
    K_block_size = int(K_tile_size * K_tiles_in_block)
    if K_block_size < K_tile_size:
        K_tile_size = K_block_size
        K_tiles_in_block = 1
    
    if K < K_block_size:
        K_tile_size = min(K, K_tile_size)
        K_tiles_in_block = 1
        K_block_size = K_tile_size * K_tiles_in_block    

    K_n_blocks = (K + K_block_size - 1) // K_block_size


    N_tile_size = 512
    N_block_size = N_tile_size * N_tiles_in_block

    if N < N_block_size:
        N_tile_size = min(N, N_tile_size)
        N_tiles_in_block = 1
        N_block_size = N_tile_size * N_tiles_in_block

    N_n_blocks = (N + N_block_size - 1) // N_block_size

    # Index patterns
    ip_X = nl.arange(M_tile_size)[:, None]
    if_X_tile = nl.arange(K_tile_size)[None, :]
    if_X_block = nl.arange(K_block_size)[None, :]

    ip_V = nl.arange(r_tile_size)[:, None]
    if_V_tile = nl.arange(N_tile_size)[None, :]
    if_V_block = nl.arange(N_block_size)[None, :]

    ip_U = nl.arange(M_tile_size)[:, None]
    if_U_tile = nl.arange(r_tile_size)[None, :]
    if_U_block = nl.arange(r_block_size)[None, :]

    # Main computation loops
    for i_K_block in nl.affine_range(K_n_blocks):  # Loop over K dimension blocks
        # Buffer for intermediate result XU (K x r)
        XU_result_buf = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )
        # Loop over r dimension blocks
        for i_M_block in nl.sequential_range(M_n_blocks):
            # X_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, K_block_size), dtype=pe_in_dt)
            X_cache = load_tensor_block(
                X_ref,
                (i_M_block * M_block_size, i_K_block * K_block_size),
                (M_tiles_in_block, M_tile_size, K_block_size),
            )
            for i_r_block in nl.affine_range(r_n_blocks):

                # U_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, r_block_size), dtype=pe_in_dt)
                U_cache = load_tensor_block(
                    U_ref,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                for ib_K_tile in nl.affine_range(K_tiles_in_block):
                    for ib_r_tile in nl.affine_range(r_tiles_in_block):
                        # PSUM buffer for X @ U
                        XU_psum = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        # Index patterns for result
                        if_XU = nl.arange(K_tile_size)[None, :]
                        ip_XU = nl.arange(r_tile_size)[:, None]

                        # Contract over M dimension
                        for ib_M_tile in nl.affine_range(M_tiles_in_block):
                            # Compute X^T @ U (since X is stored transposed)
                            XU_psum[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )

                        XU_result_buf[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum[
                            ip_XU, if_XU
                        ]

        # Loop over N dimension blocks for final result
        for i_N_block in nl.affine_range(N_n_blocks):

            # TODO: Create a final result buffer
            final_result_buf = nl.zeros((K_tiles_in_block, par_dim(K_tile_size), N_block_size), dtype=kernel_dtype)

            if_out = nl.arange(N_block_size)[None, :]
            ip_out = nl.arange(K_tile_size)[:, None]

            for i_r_block in nl.sequential_range(r_n_blocks):
                # Compute (XU) @ V for current blocks

                # V_cache = nl.ndarray((par_dim(r_tile_size), r_tiles_in_block, N_block_size), dtype=pe_in_dt)
                V_cache = load_tensor_block(
                    V_ref,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                for ib_K_tile in nl.affine_range(K_tiles_in_block):

                    for ib_N_tile in nl.affine_range(N_tiles_in_block):
                        # PSUM buffer for final result
                        XUV_psum = nl.zeros((par_dim(K_tile_size), N_tile_size), dtype=np.float32, buffer=nl.psum)

                        ip_XU_t = nl.arange(r_tile_size)[:, None]
                        if_XU_t = nl.arange(K_tile_size)[None, :]

                        ip_XUV = nl.arange(K_tile_size)[:, None]
                        if_XUV = nl.arange(N_tile_size)[None, :]
                        
                        # Contract over r dimension
                        for ib_r_tile in nl.affine_range(r_tiles_in_block):
                            # Compute XU @ V
                            ip_V_t = nl.arange(r_tile_size)[:, None]
                            if_V_t = nl.arange(N_tile_size)[None, :]

                            XUV_psum[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=V_cache[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                                stationary=XU_result_buf[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                            )

                        final_result_buf[ib_K_tile, ip_XUV, ib_N_tile * N_tile_size + if_XUV] += XUV_psum[
                            ip_XUV, if_XUV
                        ]
            for ib_K_tile in nl.affine_range(K_tiles_in_block):
                # Store the final result for the current N block
                nl.store(
                    out_ref[
                        i_K_block * K_block_size + ib_K_tile * K_tile_size + ip_out,
                        i_N_block * N_block_size + if_out,
                    ],
                    value=final_result_buf[ib_K_tile, ip_out, if_out],
                    mask=(i_K_block * K_block_size + ib_K_tile * K_tile_size + ip_out < K) & (i_N_block * N_block_size + if_out < N)
                )
    return out_ref


#Input: X, U, V
# Output: (XUV)^T
@nki.jit
def fused_mlp_up_T(
    X_ref,  # Shape: (K, M) 
    U_ref,  # Shape: (M, r)
    V_ref,  # Shape: (r, N)
    U_ref_1, # Shape: (M, r)
    V_ref_1, # Shape: (r, N)
    mixed_precision=True,
    r_tiles_in_block=8,
    K_tiles_in_block=2,
    N_tiles_in_block=4,
    M_tiles_in_block=4
):
    # Use X_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = X_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert X_ref.dtype == U_ref.dtype == V_ref.dtype

    # Shape checking - X@U@V computation
    K, M = X_ref.shape # M - 7168
    M_U, r = U_ref.shape
    r_V, N = V_ref.shape # N - 18432
    assert tuple(U_ref.shape) == (M_U, r), "Input shape mismatch!"
    assert tuple(V_ref.shape) == (r_V, N), "Input shape mismatch!"
    assert M == M_U, "M dimension must match between X and U!"
    assert r == r_V, "r dimension must match between U and V!"

    out_ref = nl.ndarray((N, K), dtype=V_ref.dtype, buffer=nl.hbm)


    # Tiling configuration

    M_tile_size = 128
    M_block_size = M_tile_size * M_tiles_in_block
    M_n_blocks = (M + M_block_size - 1) // M_block_size

    r_tile_size = 128
    r_block_size = r_tile_size * r_tiles_in_block
    r_n_blocks = (r + r_block_size - 1) // r_block_size

    K_tile_size = 512
    K_block_size = int(K_tile_size * K_tiles_in_block)
    if K_block_size < K_tile_size:
        K_tile_size = K_block_size
        K_tiles_in_block = 1
    
    if K < K_block_size:
        K_tile_size = min(K, K_tile_size)
        K_tiles_in_block = 1
        K_block_size = K_tile_size * K_tiles_in_block

    
    K_n_blocks = (K + K_block_size - 1) // K_block_size

    N_tile_size = 128
    N_block_size = N_tile_size * N_tiles_in_block
    N_n_blocks = (N + N_block_size - 1) // N_block_size

    # Index patterns
    ip_X = nl.arange(M_tile_size)[:, None]
    if_X_tile = nl.arange(K_tile_size)[None, :]
    if_X_block = nl.arange(K_block_size)[None, :]

    ip_V = nl.arange(r_tile_size)[:, None]
    if_V_tile = nl.arange(N_tile_size)[None, :]
    if_V_block = nl.arange(N_block_size)[None, :]

    ip_U = nl.arange(M_tile_size)[:, None]
    if_U_tile = nl.arange(r_tile_size)[None, :]
    if_U_block = nl.arange(r_block_size)[None, :]

    # Main computation loops
    for i_K_block in nl.affine_range(K_n_blocks):  # Loop over K dimension blocks
        # Buffer for intermediate result XU (K x r)
        XU_result_buf = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )

        XU_result_buf_1 = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )
        # Loop over r dimension blocks
        for i_M_block in nl.sequential_range(M_n_blocks):
            # X_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, K_block_size), dtype=pe_in_dt)
            X_cache = load_tensor_block_T(
                X_ref,
                (i_K_block * K_block_size, i_M_block * M_block_size),
                (M_tiles_in_block, M_tile_size, K_block_size),
            )
            for i_r_block in nl.affine_range(r_n_blocks):

                # U_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, r_block_size), dtype=pe_in_dt)
                U_cache = load_tensor_block(
                    U_ref,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                U_cache_1 = load_tensor_block(
                    U_ref_1,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                for ib_K_tile in nl.affine_range(K_tiles_in_block):
                    for ib_r_tile in nl.affine_range(r_tiles_in_block):
                        # PSUM buffer for X @ U
                        XU_psum = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        XU_psum_1 = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        # Index patterns for result
                        if_XU = nl.arange(K_tile_size)[None, :]
                        ip_XU = nl.arange(r_tile_size)[:, None]

                        # Contract over M dimension
                        for ib_M_tile in nl.affine_range(M_tiles_in_block):
                            # Compute X^T @ U (since X is stored transposed)
                            XU_psum[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )
                            
                            XU_psum_1[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache_1[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )

                        XU_result_buf[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum[
                            ip_XU, if_XU
                        ]

                        XU_result_buf_1[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum_1[
                            ip_XU, if_XU
                        ]

        # Loop over N dimension blocks for final result
        for i_N_block in nl.affine_range(N_n_blocks):

            # TODO: Create a final result buffer
            final_result_buf = nl.zeros((N_tiles_in_block, par_dim(N_tile_size), K_block_size), dtype=kernel_dtype)

            final_result_buf_1 = nl.zeros((N_tiles_in_block, par_dim(N_tile_size), K_block_size), dtype=kernel_dtype)

            if_out = nl.arange(K_block_size)[None, :]
            ip_out = nl.arange(N_tile_size)[:, None]

            for i_r_block in nl.sequential_range(r_n_blocks):
                # Compute (XU) @ V for current blocks

                # V_cache = nl.ndarray((par_dim(r_tile_size), r_tiles_in_block, N_block_size), dtype=pe_in_dt)
                V_cache = load_tensor_block(
                    V_ref,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                
                V_cache_1 = load_tensor_block(
                    V_ref_1,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                
                for ib_K_tile in nl.affine_range(K_tiles_in_block):

                    for ib_N_tile in nl.affine_range(N_tiles_in_block):
                        # PSUM buffer for final result
                        XUV_psum = nl.zeros((par_dim(N_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        XUV_psum_1 = nl.zeros((par_dim(N_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        ip_XU_t = nl.arange(r_tile_size)[:, None]
                        if_XU_t = nl.arange(K_tile_size)[None, :]

                        ip_XUV = nl.arange(N_tile_size)[:, None]
                        if_XUV = nl.arange(K_tile_size)[None, :]
                        
                        # Contract over r dimension
                        for ib_r_tile in nl.affine_range(r_tiles_in_block):
                            # Compute XU @ V
                            ip_V_t = nl.arange(r_tile_size)[:, None]
                            if_V_t = nl.arange(N_tile_size)[None, :]

                            XUV_psum[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=XU_result_buf[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                                stationary=V_cache[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                            )

                            XUV_psum_1[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=XU_result_buf_1[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                                stationary=V_cache_1[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                            )

                        final_result_buf[ib_N_tile, ip_XUV, ib_K_tile * K_tile_size + if_XUV] += XUV_psum[
                            ip_XUV, if_XUV
                        ]
                        final_result_buf_1[ib_N_tile, ip_XUV, ib_K_tile * K_tile_size + if_XUV] += XUV_psum_1[
                            ip_XUV, if_XUV
                        ]
            for ib_N_tile in nl.affine_range(N_tiles_in_block):
                # Store the final result for the current N block
                nl.store(
                    out_ref[
                        i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out,
                        i_K_block * K_block_size + if_out,
                    ],
                    value=nl.multiply(nl.silu(final_result_buf[ib_N_tile, ip_out, if_out]), final_result_buf_1[ib_N_tile, ip_out, if_out]),
                    mask=(i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out < N) &
                          (i_K_block * K_block_size + if_out < K)
                )
    return out_ref


# Invoke MLP kernel as the following
def nki_mm(x, up_v_proj, up_u_proj, 
             gate_v_proj, gate_u_proj,
             down_v_proj, down_u_proj):
    
    
    # Call the optimized kernel with auto-selected parameters
    result = svd_mlp_with_fused_kernel(
        x, up_v_proj.T, up_u_proj.T, 
        gate_v_proj.T, gate_u_proj.T,  
        down_v_proj.T, down_u_proj.T, 
    )
    return result




##############################################


@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=8,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.
 
  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
  Returns:
      result: the resulting output tensor of shape [M,N]
  """
 
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
 
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512
 
  if M < TILE_M:
    TILE_M = M
    TILES_IN_BLOCK_M = 1
  else:
    TILES_IN_BLOCK_M = min(TILES_IN_BLOCK_M, M // TILE_M)
  
  if N < TILE_N:
    TILE_N = N
    TILES_IN_BLOCK_N = 1
  else:
    TILES_IN_BLOCK_N = min(TILES_IN_BLOCK_N, N // TILE_N)
  
  if K < TILE_K:
    TILE_K = K
    TILES_IN_BLOCK_K = 1
  else:
    TILES_IN_BLOCK_K = min(TILES_IN_BLOCK_K, K // TILE_K)
 
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K
 
  NUM_BLOCK_M = (M + BLOCK_M - 1) // BLOCK_M
  NUM_BLOCK_N = (N + BLOCK_N - 1) // BLOCK_N
  NUM_BLOCK_K = (K + BLOCK_K - 1) // BLOCK_K
 
  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)
 
    # Blocking K dimension (the contraction dimension)
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.zeros((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                           dtype=rhs.dtype,
                           buffer=nl.sbuf)
 
      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        k_index = (k * TILES_IN_BLOCK_K + bk_r) * TILE_K
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[k_index + i_rhs.p, BLOCK_N * n + i_rhs.x], 
            mask=(k_index + i_rhs.p < K) & (BLOCK_N * n + i_rhs.x < N))
 
      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.zeros((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                              dtype=lhsT.dtype,
                              buffer=nl.sbuf)
        
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          k_index = (k * TILES_IN_BLOCK_K + bk_l) * TILE_K
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[k_index + i_lhsT.p, BLOCK_M * m + i_lhsT.x], 
              mask=(k_index + i_lhsT.p < K) & (BLOCK_M * m + i_lhsT.x < M))
 
        # Do matmul with all tiles in the blocks
        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
 
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              k_index = (k * TILES_IN_BLOCK_K + bk) * TILE_K
              res_tile[...] += nisa.nc_matmul(
                  lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])
 
            # Accumulate on corresponding SBUF tile
            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]
 
    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
        result_packed = nl.zeros((TILE_M, BLOCK_N),
                                 dtype=result_tiles.dtype,
                                 buffer=nl.sbuf)
 
        # coalesce result tiles for better DMA performance
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x], 
                 mask=((TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p < M) & 
                      (BLOCK_N * n + i_res_packed.x < N))
 
  return result


# @nki.jit
# def XUV_matmul(x_ref, u_ref, v_ref, mixed_precisioin=True):
    
#     kernel_dtype = x_ref.dtype
#     pe_in_dt = nl.bfloat16 if mixed_precisioin else np.float32
#     assert x_ref.dtype == u_ref.dtype == v_ref.dtype

   
#     m, n = x_ref.shape
#     n, p = u_ref.shape
#     p, d = v_ref.shape

    
#     ###################
#     out_ref = nl.ndarray((m, d), dtype=x_ref.dtype, buffer=nl.shared_hbm)

#     x_seq_n_tiles, x_seq_tile_size = m // 128, 128
#     u_seq_n_tiles, u_seq_tile_size = p // 128, 128
#     v_seq_n_tiles, v_seq_tile_size = p // 128, 128

#     ###################### load tensor to SBUF from HBM
    

#     x_local = nl.ndarray((x_seq_n_tiles, par_dim(n), x_seq_tile_size), dtype=pe_in_dt) 
#     ip_q = nl.arange(n)[:, None]
#     if_q = nl.arange(x_seq_tile_size)[None, :]
#     for i_x_seq_tile in nl.affine_range(x_seq_n_tiles):
#         x_local[i_x_seq_tile, ip_q, if_q] = nl.load_transpose2d(
#             x_ref[i_x_seq_tile * x_seq_tile_size + nl.arange(x_seq_tile_size)[:, None], nl.arange(n)[None, :]],
#             dtype=pe_in_dt
#         ) 

#     u_local = nl.ndarray((u_seq_n_tiles, par_dim(n), u_seq_tile_size), dtype=pe_in_dt)
#     ip_u = nl.arange(n)[:, None]
#     if_u = nl.arange(u_seq_tile_size)[None, :]
#     for i_u_seq_tile in nl.affine_range(u_seq_n_tiles):
#         u_local[i_u_seq_tile, ip_u, if_u] = nl.load(
#             u_ref[nl.arange(n)[:, None], i_u_seq_tile * u_seq_tile_size + nl.arange(u_seq_tile_size)[None, :]],
#             dtype=pe_in_dt
#         )
    
#     v_local = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d), dtype=pe_in_dt)
#     ip_v = nl.arange(v_seq_tile_size)[:, None]
#     if_v = nl.arange(d)[None, :]
#     for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):   
#         v_local[ip_v, i_v_seq_tile, if_v] = nl.load(v_ref[i_v_seq_tile * v_seq_tile_size + ip_v, if_v], dtype=pe_in_dt)
    
#     ###################### X @ U @ V

#     for i_x_seq_tile in nl.affine_range(x_seq_n_tiles):
#         xu_res_buf = nl.ndarray((par_dim(x_seq_tile_size), p), dtype=kernel_dtype)
#         ip_qk = nl.arange(x_seq_tile_size)[:, None]
#         if_qk = nl.arange(u_seq_tile_size)[None, :]

#         for i_u_seq_tile in nl.affine_range(u_seq_n_tiles):
#             xu_psum = nl.zeros((par_dim(x_seq_tile_size), u_seq_tile_size), dtype=nl.float32, buffer=nl.psum)
#             xu_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=u_local[i_u_seq_tile, ip_u, if_u], 
#                                                     stationary=x_local[i_x_seq_tile, ip_q, if_q])
#             xu_res_buf[ip_qk, i_u_seq_tile * u_seq_tile_size + if_qk] = nl.copy(xu_psum[ip_qk, if_qk], dtype=kernel_dtype)
              
#         trans_xu_res_buf = nl.ndarray((par_dim(u_seq_tile_size), u_seq_n_tiles, x_seq_tile_size), dtype=pe_in_dt) 
#         ip_score_t = nl.arange(u_seq_tile_size)[:, None]
#         if_score_t = nl.arange(x_seq_tile_size)[None, :]
#         ip_scores = nl.arange(x_seq_tile_size)[:, None]
#         if_scores = nl.arange(u_seq_tile_size)[None, :]
#         for i_u_seq_tile in nl.affine_range(u_seq_n_tiles):
#             trans_xu_res_buf[ip_score_t, i_u_seq_tile, if_score_t] = nisa.nc_transpose(xu_res_buf[ip_scores, i_u_seq_tile * u_seq_tile_size + if_scores])
        
#         res_psum = nl.zeros((par_dim(x_seq_tile_size), d), dtype=nl.float32, buffer=nl.psum)
#         ip_out = nl.arange(x_seq_tile_size)[:, None]
#         if_out = nl.arange(d)[None, :]
#         ip_v_t=nl.arange(u_seq_tile_size)[:, None]
#         if_v_t=nl.arange(d)[None, :]
#         for i_u_seq_tile in nl.affine_range(u_seq_n_tiles): 
#             res_psum[ip_out, if_out] += nisa.nc_matmul(stationary=trans_xu_res_buf[ip_score_t, i_u_seq_tile, if_score_t],
#                                                         moving=v_local[ip_v_t, i_u_seq_tile, if_v_t]) 

#         res_sbuf = nl.copy(res_psum[ip_out, if_out], dtype=kernel_dtype)
#         nl.store(out_ref[i_x_seq_tile * x_seq_tile_size + ip_out, if_out], value=res_sbuf)
      
#     return out_ref


# @nki.jit
# def nki_matmul_tiled_basic(lhsT, rhs):
#   """
#   A basic NKI matrix multiplication kernel that uses tiling.

#   This kernel can handle large matrices that satisfy specific size
#   multiple requirements.
#   - lhsT: K and M dimensions must be multiples of 128.
#   - rhs: N dimension must be a multiple of 512.

#   Args:
#       lhsT: The left-hand side operand, which is the transpose of A,
#             with shape [K, M].
#       rhs: The right-hand side operand, B, with shape [K, N].
#   Returns:
#       result: The result matrix D, with shape [M, N].
#   """
#   # --- 1. SETUP PHASE: Define tile sizes and get matrix dimensions ---
#   K, M = lhsT.shape
#   K_rhs, N = rhs.shape
#   assert K == K_rhs, "The contraction dimension K must match for LHS and RHS"

#   # Define the size of a "Tile", the basic unit processed by the hardware.
#   # This corresponds to the required size multiples for the dimensions.
#   TILE_M = 128
#   TILE_K = 128
#   TILE_N = 128

#   # Check if input dimensions meet the requirements
#   assert M % TILE_M == 0, f"Dimension M({M}) must be a multiple of {TILE_M}"
#   assert K % TILE_K == 0, f"Dimension K({K}) must be a multiple of {TILE_K}"
#   assert N % TILE_N == 0, f"Dimension N({N}) must be a multiple of {TILE_N}"

#   # Calculate the number of tiles in each dimension
#   NUM_TILES_M = M // TILE_M
#   NUM_TILES_K = K // TILE_K
#   NUM_TILES_N = N // TILE_N

#   # Define the final output tensor in the main memory (HBM)
# #   result = nl.zeros((M, N), dtype=lhsT.dtype, buffer=nl.hbm)
#   result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.hbm)

#   # --- 2. TILING LOOPS: M -> N -> K ---
#   # Iterate over each tile of the output matrix
#   for m_tile_idx in nl.affine_range(NUM_TILES_M):
#     for n_tile_idx in nl.affine_range(NUM_TILES_N):

#       # Create an accumulator for the current output tile (result[m, n]).
#       # It resides in PSUM for efficient accumulation operations.
#       result_tile_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

#       # Reduction loop: Iterate over all tiles along the K dimension.
#       for k_tile_idx in nl.sequential_range(NUM_TILES_K):
        
#         # Calculate the offset for the current tile in the large matrix
#         m_offset = m_tile_idx * TILE_M
#         n_offset = n_tile_idx * TILE_N
#         k_offset = k_tile_idx * TILE_K

#         # a. Load one tile of LHS from HBM to SBUF
#         lhs_tile_sbuf = nl.load(
#             lhsT[k_offset : k_offset + TILE_K,
#                  m_offset : m_offset + TILE_M]
#         )

#         # b. Load one tile of RHS from HBM to SBUF
#         rhs_tile_sbuf = nl.load(
#             rhs[k_offset : k_offset + TILE_K,
#                 n_offset : n_offset + TILE_N]
#         )

#         # c. Perform matmul on the currently loaded tiles and accumulate the result
#         #    into the PSUM accumulator.
#         result_tile_psum[...] += nl.matmul(lhs_tile_sbuf, rhs_tile_sbuf, transpose_x=True)

#       # d. After the reduction loop over K finishes, the PSUM accumulator
#       #    holds the final result for this tile.
#       # Write this final tile result back to the correct location in HBM.
#       nl.store(
#           result[m_tile_idx * TILE_M : (m_tile_idx + 1) * TILE_M,
#                  n_tile_idx * TILE_N : (n_tile_idx + 1) * TILE_N],
#           result_tile_psum
#       )

#   return result



# @nki.jit
# def three_mm_unfused(lhsT, rhs_B, rhs_C):

#     intermediate = nki_matmul_tiled_basic(lhsT, rhs_B)
#     # intermediate =nisa.nc_transpose(intermediate)
#     intermediate = intermediate.reshape(
#         (intermediate.shape[1], intermediate.shape[0]))

#     result = nki_matmul_tiled_basic(intermediate, rhs_C)

#     return result



# def cpu_golden_attn(x, u, v):
#   return np.matmul(np.matmul(x, u), v)
import torch.nn.functional as F
def nki_baseline(x, up_proj, 
             gate_proj, down_proj):
    up = nki_matmul_fully_optimized_(x.t(), up_proj.t())
    gate = nki_matmul_fully_optimized_(x.t(), gate_proj.t())
    act = F.silu(gate) * up
    output = nki_matmul_fully_optimized_(act.t() , down_proj.t())

    return output

if __name__ == "__main__":

   

    # dtype = np.float16
    # x = np.random.rand(128, 4096).astype(dtype)

    # up_proj = np.random.rand(14336, 4096).astype(dtype)
    # gate_proj = np.random.rand(14336, 4096).astype(dtype)
    # down_proj = np.random.rand(4096, 14336).astype(dtype)

    # up_v_proj = np.random.rand(2560, 4096).astype(dtype)
    # up_u_proj = np.random.rand(14336, 2560).astype(dtype) 
    # gate_v_proj = np.random.rand(2560, 4096).astype(dtype)
    # gate_u_proj = np.random.rand(14336, 2560).astype(dtype)
    # down_v_proj = np.random.rand(2560, 14336).astype(dtype)
    # down_u_proj = np.random.rand(4096, 2560).astype(dtype)

    import torch
    import torch_xla.core.xla_model as xm

    
    device = xm.xla_device()   # XLA/Neuron device
    cpu = torch.device("cpu")

    dtype = torch.bfloat16

    
    # x = torch.rand((1, 4096), dtype=dtype, device=device)

    # up_proj = torch.rand((14336, 4096), dtype=dtype, device=device)
    # gate_proj = torch.rand((14336, 4096), dtype=dtype, device=device)
    # down_proj = torch.rand((4096, 14336), dtype=dtype, device=device)

    # up_v_proj = torch.rand((2560, 4096), dtype=dtype, device=device)
    # up_u_proj = torch.rand((14336, 2560), dtype=dtype, device=device)
    # gate_v_proj = torch.rand((2560, 4096), dtype=dtype, device=device)
    # gate_u_proj = torch.rand((14336, 2560), dtype=dtype, device=device)
    # down_v_proj = torch.rand((2560, 14336), dtype=dtype, device=device)
    # down_u_proj = torch.rand((4096, 2560), dtype=dtype, device=device)


#     seeedup:  1.2368935489603696
# nki_mm time:  0.002605915069580078
# nki_baseline time:  0.0033142566680908203
# r1 shape:  torch.Size([1, 4096])
# r2 shape:  torch.Size([1, 4096])
# seeedup:  1.2718206770356817
# nki_mm time:  0.0026280879974365234
# nki_baseline time:  0.0032987594604492188
# r1 shape:  torch.Size([1, 4096])
# r2 shape:  torch.Size([1, 4096])
# seeedup:  1.2551936859294204


    x = torch.rand((128, 3072), dtype=dtype, device=device)

    up_proj = torch.rand((8192, 3072), dtype=dtype, device=device)
    gate_proj = torch.rand((8192, 3072), dtype=dtype, device=device)
    down_proj = torch.rand((3072, 8192), dtype=dtype, device=device)

    up_v_proj = torch.rand((1792, 3072), dtype=dtype, device=device)
    up_u_proj = torch.rand((8192, 1792), dtype=dtype, device=device)
    gate_v_proj = torch.rand((1792, 3072), dtype=dtype, device=device)
    gate_u_proj = torch.rand((8192, 1792), dtype=dtype, device=device)
    down_v_proj = torch.rand((1792, 8192), dtype=dtype, device=device)
    down_u_proj = torch.rand((3072, 1792), dtype=dtype, device=device)

    

    for _ in range(10):
        start = time.time()
        r1 = nki_mm(x, up_v_proj, up_u_proj, 
                gate_v_proj, gate_u_proj,
                down_v_proj, down_u_proj)
        dur_mm = time.time() - start
        
        
        start2 = time.time()
        r2 = nki_baseline(x, up_proj, 
                gate_proj, down_proj)
        dur_baseline = time.time() - start2
        print("nki_mm time: ", dur_mm)
        print("nki_baseline time: ", dur_baseline)
        print("r1 shape: ", r1.shape)
        print("r2 shape: ", r2.shape)
        print("seeedup: ", dur_baseline / dur_mm)
    
    



