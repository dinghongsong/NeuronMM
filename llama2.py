# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




"""
Author: Dinghong Song, Jierui Xu, Dong Li
Date: 2025-03-10
Description: AWS NKI Contest llama.py

Please run the following command.
command: python main.py --enable-nki --mode evaluate_all --seq-len 640 --fused-qkv --context-encoding-buckets 640 --token-generation-buckets 640
Total Score: 11.02777378315294
# python3 main.py --mode evaluate_all --enable-nki --seq-len 640
"""

# Benchmark completed and its result is as following
# {
#     "e2e_model": {
#         "latency_ms_p50": 1253.8976669311523,
#         "latency_ms_p90": 1294.7640180587769,
#         "latency_ms_p95": 1384.1175198554993,
#         "latency_ms_p99": 1486.0365700721738,
#         "latency_ms_p100": 1511.5163326263428,
#         "latency_ms_avg": 1274.1368889808655,
#         "throughput": 502.30081676067954
#     },
#     "context_encoding_model": {
#         "latency_ms_p50": 64.02003765106201,
#         "latency_ms_p90": 65.3770923614502,
#         "latency_ms_p95": 65.37882089614868,
#         "latency_ms_p99": 65.39204835891724,
#         "latency_ms_p100": 65.39535522460938,
#         "latency_ms_avg": 64.28561210632324,
#         "throughput": 7980.012683888569
#     },
#     "token_generation_model": {
#         "latency_ms_p50": 7.59434700012207,
#         "latency_ms_p90": 8.08563232421875,
#         "latency_ms_p95": 8.54349136352539,
#         "latency_ms_p99": 8.585317134857178,
#         "latency_ms_p100": 9.05466079711914,
#         "latency_ms_avg": 7.688724143164499,
#         "throughput": 131.09281711356402
#     }
# }

# Final Score: 2.2582588332596973
#         Accuracy: True
#         Latency: 1486.0365700721738
#         Throughput: 502.30081676067954
#         NKI FLOPs Ratio: 0.9985456010433191
# Total Score: 10.79218183196161

import copy
import gc
import logging
import math
from typing import List, Optional, Tuple, Type
from neuronx_distributed.parallel_layers import utils 
import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    LinearWithAsyncCommunication,
)
from neuronx_distributed.parallel_layers.utils import (
cast_if_autocast_enabled,
verify_casted_dtype,)

from torch.distributed import ProcessGroup
from typing import (
    Optional, Tuple, Union, Any, Callable, Dict, Type, cast
)
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_dim,
    gather_from_tensor_model_parallel_region,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.quantization.quantization_config import QuantizationType, QuantizedDtype
from neuronx_distributed.quantization.quantization_layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    QuantizedColumnParallel,
    QuantizedRowParallel,
)
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel,
    quant_mlp_fused_add_isa_kernel,
    quant_mlp_isa_kernel,
)
from neuronxcc.nki._private_kernels.rmsnorm import rmsnorm_quant_isa_kernel
# from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch import nn, ones
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from neuronxcc.nki.language import par_dim
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
    GQA, GroupQueryAttention_O, GroupQueryAttention_QKV 
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    transpose_parallel_linear_layer,
)

# from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group
import neuronxcc.nki.isa as nisa
from torch_neuronx.xla_impl.ops import RmsNorm
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.compiler as ncc


from neuronx_distributed_inference.modules.attention.utils import (    apply_rotary_pos_emb,
    distributed_softmax,
    manual_softmax,
    move_heads_front,
    repeat_kv)
# from neuronx_distributed_inference.modules.attention.attention_base import (NeuronAttentionBase, FlashAttentionStrategy, _flash_fwd_call_bir )
from neuronx_distributed_inference.modules.attention.attention_base import (NeuronAttentionBase, FlashAttentionStrategy )

from torch import Tensor, nn

_LLAMA_MODULE_MAP = {}

logger = logging.getLogger("Neuron")

def update_base_addr(base_addr: int, tensor, advance: bool) -> int:
    if tensor.ndim == 2:
        # pardim, fdim
        buf_size = tensor.shape[1] * tensor.itemsize
    elif tensor.ndim == 3:
        # block_dim, pardim, fdim
        buf_size = tensor.shape[0] * tensor.shape[2] * tensor.itemsize
    else:
        raise NotImplementedError(
            f"Buffer size for tensor shape {tensor.shape} is unknown"
        )
    if advance:
        next_base_addr = base_addr + buf_size
        print(f"Allocate buf @ {base_addr} -> {next_base_addr}")
    else:
        next_base_addr = base_addr - buf_size
        print(f"Restore buf @ {base_addr} -> {next_base_addr}")
    return next_base_addr



def allocate_nki_matmul(total_input, weight):
    B, M, K = total_input.shape
    N, K_ = weight.shape

    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax

    TILES_IN_BLOCK_K=8
     
    if M < 256:
        TILES_IN_BLOCK_M=1
    elif M < 512:
        TILES_IN_BLOCK_M=2
    elif M < 1024:
        TILES_IN_BLOCK_M=4
    else:
        TILES_IN_BLOCK_M=8
    # TILES_IN_BLOCK_M=1

    if  N == 256:
        output = nki_matmul_hoist_load_(total_input.transpose(1, 2), weight.t())
    else:
        if N == 512:
            TILES_IN_BLOCK_N=1
        elif N >= 1024:
            TILES_IN_BLOCK_N=2
        # elif N == 1536:
        #     TILES_IN_BLOCK_N=3
        else:
            TILES_IN_BLOCK_N=1


        if K == 2048 and N == 2048:
            TILES_IN_BLOCK_N=4 
            TILES_IN_BLOCK_K=16
        elif K == 1024 and N == 2048:
            TILES_IN_BLOCK_N=4
            TILES_IN_BLOCK_K=8
        elif K == 2048 and N == 4096:
            TILES_IN_BLOCK_N=4
            TILES_IN_BLOCK_K=16
        elif K == 4096 and N == 2048:
            TILES_IN_BLOCK_N=4
            TILES_IN_BLOCK_K=16
        else:
            TILES_IN_BLOCK_K=1
        

        BLOCK_M = TILE_M * TILES_IN_BLOCK_M
        BLOCK_N = TILE_N * TILES_IN_BLOCK_N
        BLOCK_K = TILE_K * TILES_IN_BLOCK_K
        
        padding = False
        if M % BLOCK_M:
            res = BLOCK_M - (M % BLOCK_M)
            ones_matrix = torch.ones((B, res, K ), dtype=total_input.dtype, device=total_input.device)
            total_input = torch.cat((total_input, ones_matrix), axis=1)
            padding = True
        
        if N % BLOCK_N:
            res = BLOCK_N - (N % BLOCK_N)
            ones_matrix = torch.ones((res, K ), dtype=total_input.dtype, device=total_input.device)
            weight =  torch.cat((weight, ones_matrix), axis=0)
            padding = True

        
        
        # output = torch.empty((B, M, N), dtype=total_input.dtype, device=total_input.device)
        # for b in range(B):
        #     hidden_b = total_input[b]
        #     output_i = nki_matmul_fully_optimized_copy_(hidden_b.transpose(0, 1), weight.t(),
        #                                                 TILES_IN_BLOCK_M,TILES_IN_BLOCK_N,TILES_IN_BLOCK_K,
        #                                             )
        #     if padding:
        #         output_i = output_i[:M, :N]
            
        #     output[b,:,:] = output_i

        ##########################################

        lhs = total_input.view(-1, K)
        # print("=="*100)
        # print(f"lhs shape after reshape: {lhs.shape}") 
        # print(f"weight shape: {weight.t().shape}")
#         lhs shape after reshape: torch.Size([1024, 2048])
# weight shape: torch.Size([2048, 512])
# weight shape: torch.Size([2048, 1024])
        output = nki_matmul_fully_optimized_copy_(lhs.transpose(0, 1), weight.t(),
                                                        TILES_IN_BLOCK_M,
                                                        TILES_IN_BLOCK_N,
                                                        TILES_IN_BLOCK_K,
                                                    )
        if padding:
            output = output[:M, :N]
        output = output.reshape(B, M, N)

        ##########################################
    return output



@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_fused_rms_norm_qkv(hidden, gamma, weights, hidden_buffer_degree, eps):
    """
    Allocated kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        weights (_type_): Fused QKV linear weights, assumed to be eltwise-multiplied with RMS norm weight vector (gamma)
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    _dim, head_dim = weights.shape

    assert dim <= 8192 and dim & 128 == 0, "Unsupported hidden dimension"
    assert _dim == dim, "Reduction dimension must match"
    # assert head_dim <= 512, "Head dimension must be 512 or less"

    norm_dtype = nl.float32
    # norm_dtype = hidden.dtype

    out_tensor = nl.ndarray(
        (batch, seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm
    )

    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    ix, iy = nl.mgrid[0:pmax, 0:dim]
    i_lhs = nl.mgrid[0:pmax, 0:pmax]
    i_rhs = nl.mgrid[0:pmax, 0:fmax]
    i_res = nl.mgrid[0:pmax, 0:fmax]
    M = math.ceil(dim / pmax)
    N = math.ceil(head_dim / fmax)
    NUM_TRANSP_TILES = math.ceil(dim / fmax)
    NUM_TILES = math.ceil(seqlen / pmax)
    TILES_INT = math.ceil(NUM_TILES / hidden_buffer_degree)
    scale = 1 / dim
    sbuf_base_addr = 0
    
    gamma_bcast = nl.ndarray(
        (par_dim(pmax), dim),
        dtype=gamma.dtype,
        buffer=nl.sbuf,
    )
    i_f = nl.arange(dim)[None, :]
    i_w = nl.arange(1)[:, None] 
    # FIXME: Change to more efficient NKI broadcast APIs when they become available.
    for i in nl.affine_range(pmax):
        gamma_bcast[i, i_f] = nl.load(
            gamma.reshape((1, dim))[i_w, i_f], dtype=gamma.dtype
        )

    iden_x, iden_y = nl.mgrid[0:pmax, 0:128]

    identity_a = nl.shared_constant(
        np.identity(n=128, dtype=np.int8), dtype=hidden.dtype
    )
    identity_tensor = nl.ndarray(
        (par_dim(pmax), 128),
        dtype=weights.dtype,
        buffer=nl.sbuf,
    )

    # sbuf_base_addr = update_base_addr(sbuf_base_addr, identity_tensor, True)
    identity_tensor[iden_x, iden_y] = nl.load(identity_a, dtype=weights.dtype)
    bias_placeholder = nl.ndarray(
        (par_dim(pmax), 1),
        dtype=np.float32,
        buffer=nl.sbuf,
    )
    # sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, True)
    bias_placeholder[...] = 0

    for b in nl.affine_range(batch):
        weights_buffer = nl.ndarray(
            (M, par_dim(pmax), head_dim),
            dtype=weights.dtype,
            buffer=nl.sbuf
            
        )
        # sbuf_base_addr = update_base_addr(sbuf_base_addr, weights_buffer, True)
        # Preload the entire weights tensor. everything fits in SBUF for LLaMA 3.1 70B
        for m in nl.affine_range(M):
            # weights_buffer[m, i_rhs.p, i_rhs.x] = nl.load(
            #     weights[m * pmax + i_rhs.p, i_rhs.x],
            #     mask=(m * pmax + i_rhs.p < dim) & (i_rhs.x < head_dim),
            # )

            i_weights = nl.mgrid[0:pmax, 0: N * fmax]
            weights_buffer[m, i_weights.p, i_weights.x] = nl.load(weights[m * pmax + i_weights.p, i_weights.x],
                                                    mask=(m*pmax+i_weights.p<dim) & (i_weights.x<head_dim))
    
        for i in nl.affine_range(TILES_INT):
            # Double buffer the input tensor
            in_bufs = nl.ndarray(
                (hidden_buffer_degree, par_dim(pmax), dim),
                dtype=hidden.dtype,
                buffer=nl.sbuf,
            )
            # sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, True)
            for i_interleave_grp in nl.affine_range(hidden_buffer_degree):
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[
                        b, (hidden_buffer_degree * i + i_interleave_grp) * pmax + ix, iy
                    ],
                    mask=(hidden_buffer_degree * i + i_interleave_grp) * pmax + ix
                    < seqlen,
                )
                act = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=norm_dtype,
                    buffer=nl.sbuf,
                )
                # sbuf_base_addr = update_base_addr(sbuf_base_addr, act, True)

                # Write the RMS and RMS Reciprocal tensors back out here, in-place
                square_sum = nl.ndarray(
                    (par_dim(pmax), 1),
                    dtype=norm_dtype,
                    buffer=nl.sbuf,
                )
                # sbuf_base_addr = update_base_addr(sbuf_base_addr, square_sum, True)

                # Write the output of RMS and RMS^T (in-place) out to here
                out_tile = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=weights.dtype,
                    buffer=nl.sbuf,
                )
                

                # Store the final output tiles to here before sending back to DRAM
                output_sbuf = nl.ndarray(
                    (par_dim(pmax), fmax),
                    dtype=weights.dtype,
                    buffer=nl.sbuf,
                )
                # sbuf_base_addr = update_base_addr(sbuf_base_addr, output_sbuf, True)



                act[...] = nisa.activation_reduce(
                    op=nl.square,
                    data=in_bufs[i_interleave_grp],
                    reduce_op=np.add,
                    reduce_res=square_sum[...],
                    bias=bias_placeholder[...],
                )
                square_sum[...] = nisa.tensor_scalar(
                    square_sum[...], np.multiply, scale, op1=np.add, operand1=eps
                )
                square_sum[...] = nisa.activation(
                    op=nl.rsqrt, data=square_sum[...], bias=bias_placeholder[...]
                )

                # all PE array ops must output to FP32 on trn1 but must match input dtype in trn2
                if nisa.get_nc_version() == nisa.nc_version.gen3:
                    transpose_res_psum = nl.ndarray(
                        (NUM_TRANSP_TILES, par_dim(pmax), 4 * pmax),
                        dtype=weights.dtype,
                        buffer=nl.psum,
                    )  # FIXME: perf is better when all tiles are on bank 0?
                else:
                    transpose_res_psum = nl.ndarray(
                        (NUM_TRANSP_TILES, par_dim(pmax), 4 * pmax),
                        dtype=np.float32,
                        buffer=nl.psum,
                    )  # FIXME: perf is better when all tiles are on bank 0?

                for m in nl.affine_range(NUM_TRANSP_TILES):
                    # Perform (hidden .* RMS Reciprocal)^T in tiles of fmax (512)
                    out_tile[i_rhs.p, m * fmax + i_rhs.x] = nl.multiply(
                        in_bufs[i_interleave_grp, i_rhs.p, m * fmax + i_rhs.x],
                        square_sum[...],
                        dtype=weights.dtype,
                    )
                    
                    out_tile[i_rhs.p, m * fmax + i_rhs.x] = nl.multiply(
                        out_tile[i_rhs.p, m * fmax + i_rhs.x],
                        gamma_bcast[i_rhs.p, m * fmax + i_rhs.x],
                        dtype=weights.dtype,
                    )

                    TILES_IN_BLOCK = fmax // pmax
                    for j in nl.affine_range(TILES_IN_BLOCK):
                        transpose_res_psum[m, i_lhs.p, j * pmax + i_lhs.x] = (
                            nisa.nc_matmul(
                                out_tile[
                                    i_lhs.p, (m * TILES_IN_BLOCK + j) * pmax + i_lhs.x
                                ],
                                identity_tensor[...],
                                is_transpose=True,
                            )
                        )
                    out_tile[i_rhs.p, m * TILES_IN_BLOCK * pmax + i_rhs.x] = nl.copy(
                        transpose_res_psum[m], dtype=hidden.dtype
                    )

                # perform (RMSNorm(hidden)^T)^T @ wQKV  
                for n in nl.sequential_range(N):
                  
                  res_psum = nl.ndarray(
                      (par_dim(pmax), fmax), dtype=nl.float32, buffer=nl.psum
                  )
                  
                  for m in nl.affine_range(M):
                      res_psum += nisa.nc_matmul(
                          out_tile[i_lhs.p, m * pmax + i_lhs.x],
                          weights_buffer[m, i_rhs.p, n * fmax + i_rhs.x],
                      )

                  output_sbuf[...] = nl.copy(res_psum, dtype=out_tensor.dtype)
                  nl.store(
                      out_tensor[
                          b,
                          (hidden_buffer_degree * i + i_interleave_grp) * pmax + i_res.p,
                          n * fmax + i_res.x,
                      ],
                      value=output_sbuf,
                      mask=(
                          (hidden_buffer_degree * i + i_interleave_grp) * pmax + i_res.p
                          < seqlen
                      )
                      & (n * fmax + i_res.x < head_dim),
                  )

    return out_tensor



@nki.jit
def nki_fused_rms_norm_qkv_single_token(hidden, gamma, weights, eps):
    """
    Allocated kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        weights (_type_): Fused QKV linear weights, assumed to be eltwise-multiplied with RMS norm weight vector (gamma)
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.


    B, S, H = hidden.shape
    L, H = weights.shape
    output_tensor = nl.ndarray((B, S, L), dtype=hidden.dtype, buffer=nl.shared_hbm)
    gamma_bcast = nl.load(gamma.reshape((1, H)))
    scale = 1 / H
    TILE_P = nl.tile_size.pmax 
    for b in nl.affine_range(B):

        in_bufs = nl.load(hidden[b])
        square_sum = nl.ndarray(
                        (1, 1),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
        act = nisa.activation_reduce(
                        op=nl.square,
                        data=in_bufs,
                        reduce_op=np.add,
                        reduce_res=square_sum,

                    )
        square_sum[...] = nisa.tensor_scalar(
                        square_sum[...], np.multiply, scale, op1=np.add, operand1=eps
                    )
        square_sum[...] = nisa.activation(
            op=nl.rsqrt, data=square_sum[...]
        )
        
        output_tile = nl.multiply(
                        in_bufs,
                        square_sum[...],
                        dtype=hidden.dtype,
                    )

        # Multiply with the RMSNorm weight
        output_tile[...] = nl.multiply(
            output_tile[...], gamma_bcast[...]
        )


        output_tile = output_tile.broadcast_to((TILE_P, H))
        
        for i in nl.affine_range(L // TILE_P):
            weight_tile = nl.load(weights[i * TILE_P: (i + 1) * TILE_P, :])
            dot_tensor = nisa.tensor_tensor(weight_tile, output_tile, op=nl.multiply)

            reduce_tensor = nisa.tensor_reduce(nl.add, dot_tensor, axis=[1])
    
            nl.store(output_tensor[b, 0, i * TILE_P: (i + 1) * TILE_P],
                    value=reduce_tensor)


   
    return output_tensor




@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, eps):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

    # Make sure shapes match
    assert a_tensor.shape[2] == g_tensor.shape[0]

    # Generate tensor indices to index input tensor
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[2])[None, :]

    num_rows = a_tensor.shape[1]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently

    for b in nl.affine_range(a_tensor.shape[0]):
        for i in nl.affine_range(math.ceil(a_tensor.shape[1]/128)):
            # Load input data from external memory to on-chip memory
            a_tile = nl.zeros([128, a_tensor.shape[2]], a_tensor.dtype)
            # a_tile = nl.ndarray([128, a_tensor.shape[2]], a_tensor.dtype)
            a_tile[...] = nl.load(a_tensor[b, i * 128 + ix, iy], mask=(i * 128 + ix < num_rows))

            # Compute element-wise square of a_tensor
            in_square = nl.square(a_tile)

            # Calculate sum of squared elements, along last dimension
            square_sum = nl.sum(in_square, axis=[1])

            # Scale and get a reciprocal
            mean = square_sum / a_tensor.shape[2]

            # Take square root of mean and then reciprocal with
            # rsqrt API (one ISA instruction)
            rms_reciprocal = nl.rsqrt(mean + eps)

            # Scale the input tensor
            out_tile = nl.multiply(a_tile, rms_reciprocal)

            # Broadcast weight along first axis to match tensor shape
            # num_rows_active = min(num_rows - i * 128, 128)
            g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

            # Multiply with the RMSNorm weight
            out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

            # store the addition results back to external memory (out_tensor)
            nl.store(out_tensor[b, i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows))

    return out_tensor


@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
    B, K, M = lhsT.shape
    K_, N = rhs.shape
    result = nl.ndarray((B, M, N),dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax

    i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for b in nl.affine_range(B):
        for m in nl.affine_range(M // TILE_M):
            lhsT_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_M),
                                    dtype=lhsT.dtype,
                                    buffer=nl.sbuf)
            
            for k in nl.affine_range(K // TILE_K):
                lhsT_tiles[k, i_lhsT.p, i_lhsT.x] = nl.load(lhsT[b, k * TILE_K + i_lhsT.p,
                                                                m * TILE_M + i_lhsT.x])
            
            if N % TILE_N:
                i_rhs = nl.mgrid[0:TILE_K, 0:N]
                i_res = nl.mgrid[0:TILE_M, 0:N]
                rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), N),
                                    dtype=rhs.dtype,
                                    buffer=nl.sbuf)
                
                for k in nl.affine_range(K // TILE_K):
                    rhs_tiles[k, i_rhs.p, i_rhs.x] = nl.load(rhs[k * TILE_K + i_rhs.p,
                                                                i_rhs.x])
                
                res_psum = nl.zeros((TILE_M, N), nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    res_psum[...] += nisa.nc_matmul(lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                                            rhs_tiles[k, i_rhs.p, i_rhs.x]
                                            )
                res_sbuf = nl.copy(res_psum, dtype=result.dtype)
                nl.store(result[b, m * TILE_M + i_res.p, i_res.x], value=res_sbuf)

            else:
                for n in nl.affine_range(N // TILE_N):
                    rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                                        dtype=rhs.dtype,
                                        buffer=nl.sbuf)
                    
                    for k in nl.affine_range(K // TILE_K):
                        rhs_tiles[k, i_rhs.p, i_rhs.x] = nl.load(rhs[k * TILE_K + i_rhs.p,
                                                                    n * TILE_N + i_rhs.x])
                    
                    res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        res_psum[...] += nisa.nc_matmul(lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                                                rhs_tiles[k, i_rhs.p, i_rhs.x]
                                                )
                    res_sbuf = nl.copy(res_psum, dtype=result.dtype)
                    nl.store(result[b, m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sbuf)
            
        if M % TILE_M:
            m = M // TILE_M
            mod_m = M % TILE_M
            if N // TILE_N: 
                lhsT_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), mod_m),
                                        dtype=lhsT.dtype,
                                        buffer=nl.sbuf)
                
                for k in nl.affine_range(K // TILE_K):
                    lhsT_tiles[k, 0:TILE_K, :] = nl.load(lhsT[b,  k * TILE_K: (k + 1) * TILE_K,
                                                                    m * TILE_M:])
                for n in nl.affine_range(N // TILE_N):
                    rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                                        dtype=rhs.dtype,
                                        buffer=nl.sbuf)
                    
                    for k in nl.affine_range(K // TILE_K):
                        rhs_tiles[k, i_rhs.p, i_rhs.x] = nl.load(rhs[k * TILE_K + i_rhs.p,
                                                                    n * TILE_N + i_rhs.x])
                    
                    res_psum = nl.zeros((mod_m, TILE_N), nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        res_psum[...] += nisa.nc_matmul(lhsT_tiles[k, 0:TILE_K, :],
                                                rhs_tiles[k, i_rhs.p, i_rhs.x])
                    res_sbuf = nl.copy(res_psum, dtype=result.dtype)
                    nl.store(result[b, m * TILE_M:, n * TILE_N: (n + 1) * TILE_N], value=res_sbuf)
            
            else:
                n = N // TILE_N
                mod_n = N % TILE_N

                lhsT_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), mod_m), dtype=lhsT.dtype, buffer=nl.sbuf)
                for k in nl.affine_range(K // TILE_K):
                    lhsT_tiles[k, 0:TILE_K, :] = nl.load(lhsT[b,  k * TILE_K: (k + 1) * TILE_K,
                                                                    m * TILE_M:])
                    
                rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), mod_n), dtype=rhs.dtype, buffer=nl.sbuf)
                for k in nl.affine_range(K // TILE_K):
                    rhs_tiles[k, 0:TILE_K, :] = nl.load(rhs[k * TILE_K: (k + 1) * TILE_K, 
                                                                    n * TILE_N:])
                
                res_psum = nl.zeros((mod_m, mod_n), nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    res_psum[...] += nisa.nc_matmul(lhsT_tiles[k, 0:TILE_K, :], rhs_tiles[k,  0:TILE_K, :])
                res_sbuf = nl.copy(res_psum, dtype=result.dtype)
                
                nl.store(result[b, m * TILE_M:, n * TILE_N: ], value=res_sbuf)

    return result


@nki.jit
def nki_sinle_token_matmul_(total_input, weight):
  B, S, H = total_input.shape
  L, H = weight.shape
  output_tensor = nl.ndarray((B, S, L), dtype=total_input.dtype, buffer=nl.shared_hbm)
  
  TILE_P = nl.tile_size.pmax
  for b in nl.affine_range(B):
    input_bcast = nl.load(total_input[b, :, :]).broadcast_to((TILE_P, H))
    
    for i in nl.affine_range(L // TILE_P):
      weight_tile = nl.load(weight[i * TILE_P: (i + 1) * TILE_P, :])
    #   dot_tensor = input_bcast * weight_tile
      dot_tensor = nisa.tensor_tensor(input_bcast, weight_tile, op=nl.multiply)
      reduce_tensor = nisa.tensor_reduce(nl.add, dot_tensor, axis=[1])

      nl.store(output_tensor[b, 0, i * TILE_P: (i + 1) * TILE_P],
               value=reduce_tensor)
  
  return output_tensor






@nki.jit
def nki_matmul_fully_optimized_copy_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=8,
    TILES_IN_BLOCK_N=1,
    TILES_IN_BLOCK_K=1,
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

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    # the size has to be multiple of block size
   
    
    # print("="*100)
    # print("M:", M)
    # print("N:", N)
    # print("BLOCK_M:", BLOCK_M)
    # print("BLOCK_N:", BLOCK_N)
    # print("BLOCK_K:", BLOCK_K)
    # print("K:", K)
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    # Blocking N dimension (the RHS free dimension)
    for n in nl.affine_range(NUM_BLOCK_N):
        result_tiles = nl.zeros(
            (NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, nl.par_dim(TILE_M), TILE_N),
            dtype=lhsT.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)

            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
                rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                    rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p, BLOCK_N * n + i_rhs.x]
                )

            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
                # Loading tiles from lhsT
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                lhsT_tiles = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf
                )
                for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p, BLOCK_M * m + i_lhsT.x]
                    )

                # Do matmul with all tiles in the blocks
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
                i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile[...] += nisa.nc_matmul(
                                lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                                rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x],
                            )

                        # Accumulate on corresponding SBUF tile
                        result_tiles[m, bm, bn, i_res_mm.p, i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

        # Copying the result from SBUF to HBM
        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray((TILE_K, BLOCK_N), dtype=result_tiles.dtype, buffer=nl.sbuf)

                # coalesce result tiles for better DMA performance
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p, bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn, i_res.p, i_res.x])
                nl.store(
                    result[(TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p, BLOCK_N * n + i_res_packed.x],
                    value=result_packed[i_res_packed.p, i_res_packed.x],
                )

    return result

@nki.jit
def nki_rmsnorm(hidden, gamma, hidden_buffer_degree=1, eps=1e-6):
    """
    Allocated kernel that computes RMSNorm(hidden). This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    assert (
        dim == gamma.shape[0]
    ), f"gamma {gamma.shape} does not match with hidden {hidden.shape}"
    # assert gamma.dtype == hidden.dtype, "Gamma must match hidden dtype"
    assert dim <= 8192 and dim % 128 == 0, "Unsupported hidden dimension"

    norm_dtype = nl.float32
    # norm_dtype = hidden.dtype

    out_tensor = nl.ndarray(
        (batch, seqlen, dim), dtype=hidden.dtype, buffer=nl.shared_hbm
    )

    pmax = nl.tile_size.pmax  # 128
    i_p = nl.arange(pmax)[:, None]
    i_f = nl.arange(dim)[None, :]
    i_w = nl.arange(1)[:, None]
    NUM_TILES = math.ceil(seqlen / pmax)
    TILES_INT = math.ceil(NUM_TILES / hidden_buffer_degree)
    scale = 1 / dim

    # Allocate broadcasted gamma buffer
    gamma_bcast = nl.load(
            gamma.reshape((1, dim))
        ).broadcast_to((pmax, dim))
    
    # nl.ndarray(
    #     (par_dim(pmax), dim),
    #     dtype=gamma.dtype,
    #     buffer=nl.sbuf
    # )
  
    # for i in nl.affine_range(pmax):
    #     gamma_bcast[i, i_f] = nl.load(
    #         gamma.reshape((1, dim))[i_w, i_f], dtype=gamma.dtype
    #     )

    for b in nl.affine_range(batch):
        for i in nl.affine_range(TILES_INT):
            # Buffer the input tensor
            in_bufs = nl.zeros(
                (hidden_buffer_degree, par_dim(pmax), dim),
                dtype=hidden.dtype,
                buffer=nl.sbuf
            )
           
            for i_interleave_grp in nl.affine_range(hidden_buffer_degree):
                seq_pos = (hidden_buffer_degree * i + i_interleave_grp) * pmax
                mask = seq_pos + i_p < seqlen
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[b, seq_pos + i_p, i_f],
                    mask=mask,
                )

                # Write the RMS and RMS Reciprocal tensors back out here, in-place
                square_sum = nl.ndarray(
                    (par_dim(pmax), 1),
                    dtype=norm_dtype,
                    buffer=nl.sbuf,
                )

                ############################################################
                act = nisa.activation_reduce(
                    op=nl.square,
                    data=in_bufs[i_interleave_grp],
                    reduce_op=np.add,
                    reduce_res=square_sum[...],

                )

                square_sum[...] = nisa.tensor_scalar(
                    square_sum[...], np.multiply, scale, op1=np.add, operand1=eps
                )
                square_sum[...] = nisa.activation(
                    op=nl.rsqrt, data=square_sum[...]
                )

                ########################################
                # # Compute element-wise square of a_tensor
                # in_square = nl.square(in_bufs[i_interleave_grp])

                # # Calculate sum of squared elements, along last dimension
                # square_sum = nl.sum(in_square, axis=[1])

                # # Scale and get a reciprocal
                # mean = square_sum / hidden.shape[2]

                # # Take square root of mean and then reciprocal with
                # # rsqrt API (one ISA instruction)
                # square_sum[...]= nl.rsqrt(mean + eps)
                ########################################



                # # Apply normalization
               
                output_tile = nl.multiply(
                    in_bufs[i_interleave_grp],
                    square_sum[...],
                    dtype=hidden.dtype,
                )

                # Multiply with the RMSNorm weight
                output_tile[...] = nl.multiply(
                    output_tile[...], gamma_bcast[...], mask=mask
                )

                # Store result
                nl.store(
                    out_tensor[b, seq_pos + i_p, i_f],
                    value=output_tile,
                    mask=mask,
                )
  
    return out_tensor



@nki.jit
def nki_rmsnorm_single_token(hidden, gamma, hidden_buffer_degree=1, eps=1e-6):
    """
    Allocated kernel that computes RMSNorm(hidden). This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    assert (
        dim == gamma.shape[0]
    ), f"gamma {gamma.shape} does not match with hidden {hidden.shape}"
    # assert gamma.dtype == hidden.dtype, "Gamma must match hidden dtype"
    assert dim <= 8192 and dim % 128 == 0, "Unsupported hidden dimension"

    norm_dtype = nl.float32
    # norm_dtype = hidden.dtype

    out_tensor = nl.ndarray(
        (batch, seqlen, dim), dtype=hidden.dtype, buffer=nl.shared_hbm
    )

    scale = 1 / dim

    # Allocate broadcasted gamma buffer
    gamma_bcast = nl.load(gamma.reshape((1, dim)))
    for b in nl.affine_range(batch):
        in_bufs = nl.load(hidden[b])
        square_sum = nl.ndarray(
                        (1, 1),
                        dtype=norm_dtype,
                        buffer=nl.sbuf,
                    )
        act = nisa.activation_reduce(
                        op=nl.square,
                        data=in_bufs,
                        reduce_op=np.add,
                        reduce_res=square_sum,

                    )
        square_sum[...] = nisa.tensor_scalar(
                        square_sum[...], np.multiply, scale, op1=np.add, operand1=eps
                    )
        square_sum[...] = nisa.activation(
            op=nl.rsqrt, data=square_sum[...]
        )
        
        output_tile = nl.multiply(
                        in_bufs,
                        square_sum[...],
                        dtype=hidden.dtype,
                    )

        # Multiply with the RMSNorm weight
        output_tile[...] = nl.multiply(
            output_tile[...], gamma_bcast[...]
        )

        # Store result
        nl.store(
            out_tensor[b],
            value=output_tile,
        )

    return out_tensor




def nki_fuse_up_gate_matmul(total_input, gate_w, up_w):
    
    split_idx = up_w.shape[0]
    weight=torch.cat((gate_w, up_w), dim=0)
    
    if total_input.shape[1] == 1:
        output = torch.einsum('...m,mn->...n', total_input, weight.t())
    else:
        output = allocate_nki_matmul(total_input, weight)
       
    gate_proj_output, up_proj_output = torch.split(output, split_idx, dim=2)
    return gate_proj_output, up_proj_output


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, nki_enabled=False):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps
        self.nki_enabled = nki_enabled

    def forward(self, hidden_states):
        
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        result = RmsNorm.apply(
            hidden_states, self.weight, self.variance_epsilon, len(hidden_states.shape) - 1
        )
        return result.to(original_dtype)


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.model_parallel_is_initialized() else LlamaRMSNorm


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


def _register_module(key: str, cls: Type[nn.Module]):
    _LLAMA_MODULE_MAP[key] = cls


def register_module(key: str):
    """
    Register a module for use in NeuronLlama.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronLlamaAttention")
        class NeuronLlamaAttention(nn.Module):
            ...
    """

    def inner(cls: Type[nn.Module]):
        _register_module(key, cls)
        return cls

    return inner


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    for l in range(cfg.num_hidden_layers):  # noqa: E741
        llama_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = torch.cat(
            [
                llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"],
            ],
        )
        del llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"]

    gc.collect()

    return llama_state_dict

################ Config #######################

class NeuronConfigNKI(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nki_enabled = kwargs.pop("enable_nki", False)


class LlamaInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfigNKI

################ Nki Class #######################


class NkiLinearWithAsyncCommunication(LinearWithAsyncCommunication):
    """Linear layer execution with asynchronous communication."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        sequence_dimension: Optional[int] = 0,
        save_for_backward: bool = True,
        process_group: Optional[ProcessGroup] = None,
        reduce_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ctx.use_bias = bias is not None and weight.requires_grad
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel_enabled = sequence_parallel_enabled
        ctx.sequence_dimension = sequence_dimension
        ctx.compute_weight_gradient = weight.requires_grad
        if process_group is None:
            process_group = get_tensor_model_parallel_group(as_list=True)
        ctx.process_group = process_group
        ctx.reduce_dtype = reduce_dtype

        if ctx.sequence_parallel_enabled:
            assert (
                ctx.sequence_dimension is not None
            ), "Found `sequence_parallel_enabled` set to True, but `sequence_dimension` was None, and this occured in an unexpected area"

        if save_for_backward:
            if ctx.compute_weight_gradient:
                ctx.save_for_backward(input, weight)
            else:
                ctx.save_for_backward(weight)

        if ctx.sequence_parallel_enabled:
            # `input` is supposed to be 3D and the optimal order of dimension is [sequence, batch, hidden]
            # If not SBH, the necessary transposes will be added
            total_input = _gather_along_dim(input, ctx.sequence_dimension, process_group=ctx.process_group)
        else:
            total_input = input
        
        if total_input.shape[1] == 1:
            # output = nki_sinle_token_matmul_(total_input, weight)
            # output = nki_sinle_token_matmul_2(total_input.transpose(1, 2), weight.t())
            output = torch.einsum('...m,mn->...n', total_input, weight.t())

        else:
            output = allocate_nki_matmul(total_input, weight)

        if bias is not None:
            output = output + bias
        return output


class NkiColumnParallelLinear(ColumnParallelLinear):

    def forward(self, input: torch.Tensor, *_: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input, process_group=self.tensor_parallel_group)

        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel, # torch.Size([1, 32, 2048])
            weight=self.weight, # torch.Size([2048, 2048])
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=NkiLinearWithAsyncCommunication,
            process_group=self.tensor_parallel_group,
            # reduce_dtype = self.reduce_dtype,
        )


        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel, process_group=self.tensor_parallel_group)
            if self.pad and self.pad_size > 0:
                output = torch.narrow(output, -1, 0, self.output_size - self.pad_size)
        else:
            output = output_parallel
        if self.skip_bias_add:
            return output, self.bias
        output = (output + self.bias) if self.bias is not None else output
        return output


class NkiRowParallelLinear(RowParallelLinear):

    def forward(self, input_: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            if self.pad and self.pad_size > 0:
                input_ = torch.nn.functional.pad(input_, (0, self.pad_size))
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_, process_group=self.tensor_parallel_group)

        # Matrix multiply.
        output_ = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=NkiLinearWithAsyncCommunication,
            process_group=self.tensor_parallel_group,
            # reduce_dtype = self.reduce_dtype,
        )

        if self.reduce_output:
            # All-reduce across all the partitions.
            original_dtype = output_.dtype

            output_ = output_.to(self.reduce_dtype)

            if self.sequence_parallel_enabled:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_, self.sequence_dimension, process_group=self.tensor_parallel_group,
                )
            else:
                output_ = reduce_from_tensor_model_parallel_region(
                    output_, process_group=self.tensor_parallel_group,
                )

            output_ = output_.to(original_dtype)

        if self.skip_bias_add:
            return output_, self.bias
        output = (output_ + self.bias) if self.bias is not None else output_
        return output


class NkiGroupQueryAttention_QKV(GroupQueryAttention_QKV):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
        gather_output: bool = True,
        fused_qkv: bool = False,
        clip_qkv: Optional[float] = None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        rms_norm_eps: float = None,
        qkv_kernel_enabled: bool = False,
        logical_neuron_cores: int = 1,
    ):
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            tp_degree=tp_degree,
            dtype=dtype,
            bias=bias,
            desired_sharding_strategy=desired_sharding_strategy,
            gather_output = gather_output,
            fused_qkv = fused_qkv,
            clip_qkv = clip_qkv,
            sequence_parallel_enabled = sequence_parallel_enabled,
            sequence_dimension = sequence_dimension,
            tensor_model_parallel_group = tensor_model_parallel_group,
            rms_norm_eps = rms_norm_eps,
            qkv_kernel_enabled = qkv_kernel_enabled,
            # logical_neuron_cores = logical_neuron_cores,
        )
        

        if self.tensor_model_parallel_group is not None:
            if self.fused_qkv:
                self.Wqkv = NkiColumnParallelLinear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                )
                if self.qkv_kernel_enabled:
                    # we need to transpose the weights on the CPU side to avoid
                    # needing to transpose on the device when using QKV kernel
                    self.Wqkv.weight = transpose_parallel_linear_layer(self.Wqkv.weight)

                # Set heads info as weight parameter attributes to be used in weights sharding
                setattr(self.Wqkv.weight, "fused_qkv", True)
                setattr(self.Wqkv.weight, "num_attention_heads", self.num_attention_heads)
                setattr(self.Wqkv.weight, "num_key_value_heads", self.num_key_value_heads)
                setattr(self.Wqkv.weight, "head_dim", self.head_dim)

            else:
                self.q_proj = NkiColumnParallelLinear(
                    self.hidden_size,
                    self.num_attention_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                )
                self.k_proj = NkiColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                )
                self.v_proj = NkiColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                )
        else:
            if self.fused_qkv:
                self.Wqkv = nn.Linear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                )
            else:
                self.q_proj = nn.Linear(
                    self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.bias
                )
                self.k_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )
                self.v_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )



    def _kernel_qkv_forward(self, hidden_states, fused_rmsnorm, rmsnorm):
        logger.debug(
            f"QKV kernel: fused_rmsnorm={fused_rmsnorm} logical_neuron_cores={self.logical_neuron_cores}"
        )
        bs, seqlen, h = hidden_states.shape

        h2, fused_qkv_size = self.Wqkv.weight.shape
        logger.debug(
            f"fused QKV projection weight - shape: {self.Wqkv.weight.shape}, dtype: {self.Wqkv.weight.dtype}"
        )

        # shape checks
        assert (
            fused_qkv_size
            == (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim
            // self.tp_degree
        )
        assert h == h2
        
    
        QKV = nki_fused_rms_norm_qkv(hidden=hidden_states, 
                                gamma=rmsnorm.weight.unsqueeze(0),
                                weights=self.Wqkv.weight,
                                hidden_buffer_degree=1,
                                eps=self.rms_norm_eps)
        
        return self._split_fused_qkv(QKV)


class NkiGroupQueryAttention_O(GroupQueryAttention_O):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
        input_is_parallel: bool = False,
        layer_name: str = "o_proj",
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        rpl_reduce_dtype: torch.dtype = None,
    ):
        super().__init__(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        tp_degree=tp_degree,
        dtype=dtype,
        bias=bias,
        desired_sharding_strategy=desired_sharding_strategy,
        input_is_parallel=input_is_parallel,
        layer_name=layer_name,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sequence_dimension=sequence_dimension,
        tensor_model_parallel_group=tensor_model_parallel_group,
        rpl_reduce_dtype=rpl_reduce_dtype,
        )


        if self.tensor_model_parallel_group is not None:
            self.o_proj = NkiRowParallelLinear(  
                self.num_attention_heads * self.head_dim,
                self.hidden_size,
                bias=self.bias,
                input_is_parallel=self.input_is_parallel,
                dtype=self.dtype,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_dimension=sequence_dimension,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
                reduce_dtype=rpl_reduce_dtype,
            )
        else:
            self.o_proj = nn.Linear(
                self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.bias
            )



#########################  Neuron ######################### 

##################################################

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


###################################################

class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
        # self.quantized_kernel_lower_bound = self.neuron_config.quantized_kernel_lower_bound
        self.logical_neuron_cores = self.neuron_config.logical_neuron_cores
        mlp_bias = getattr(config, "mlp_bias", False)
        if parallel_state.model_parallel_is_initialized():
            if self.quantized_mlp_kernel_enabled:
                # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")

                quantization_type = QuantizationType(self.neuron_config.quantization_type)
                quantized_dtype = QuantizedDtype.F8E4M3
                self.gate_proj = QuantizedColumnParallel(
                    input_size=self.hidden_size,
                    output_size=self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    sequence_parallel_enabled=False,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    quantization_type=quantization_type,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.up_proj = QuantizedColumnParallel(
                    input_size=self.hidden_size,
                    output_size=self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    sequence_parallel_enabled=False,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    quantization_type=quantization_type,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = QuantizedRowParallel(
                    input_size=self.intermediate_size,
                    output_size=self.hidden_size,
                    bias=mlp_bias,
                    quantization_type=quantization_type,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    sequence_parallel_enabled=False,
                    quantization_per_channel_axis=0,
                    tensor_model_parallel_group=get_tp_group(config),
                )

            else:
                self.gate_proj = NkiColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.up_proj = NkiColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = NkiRowParallelLinear(
                    self.intermediate_size,
                    self.hidden_size,
                    bias=mlp_bias,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=self.sequence_parallel_enabled,
                    sequence_dimension=self.sequence_dimension,
                    tensor_model_parallel_group=get_tp_group(config),
                    reduce_dtype=config.neuron_config.rpl_reduce_dtype,
                )


                    # Transpose the weights to the layout expected by kernels
                
                # self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                # self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                # self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)
        
            if self.mlp_kernel_enabled:
                if self.quantized_mlp_kernel_enabled:
                    preprocess_quantized_linear_layer(self.gate_proj)
                    preprocess_quantized_linear_layer(self.up_proj)
                    preprocess_quantized_linear_layer(self.down_proj)

                else:
                    # Transpose the weights to the layout expected by kernels
                    self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                    self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                    self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def _kernel_enabled_quantized_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        grid = (vnc(self.logical_neuron_cores),)
        fused_residual = residual is not None
        logger.debug(
            f"MLP: quantized kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        )

        # Can't do residual add in the kernel if SP is enabled
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "Quantized MLP cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(quant_mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(quant_mlp_isa_kernel)

        # Handle SP RMSnorm
        x_orig_dtype = x.dtype
        if self.sequence_parallel_enabled:
            # This RMSNormQuant kernel will do quantization inside, so we pass the
            # lower_bound for clipping.
            # If we don't use this kernel, the MLP kernel below will do the
            # quantization, so we also pass lower_bound to that kernel.
            if self.rmsnorm_quantize_kernel_enabled:
                logger.debug(
                    "Running Quantized MLP kernel with sequence-parallel RMSnorm-Quantize kernel!"
                )
                _rmsnorm_quant_fwd_call = nki_jit()(rmsnorm_quant_isa_kernel)
                quant_rmsnorm_out = torch.zeros(
                    size=(
                        x.shape[0],  # batch size
                        x.shape[1],  # sequence length
                        x.shape[2] + 4,  # hidden size + 4 bytes for packing fp32 scale
                    ),
                    dtype=torch.int8,
                    device=x.device,
                )
                ln_w = rmsnorm.weight.unsqueeze(0)
                lower_bound = self.quantized_kernel_lower_bound
                _rmsnorm_quant_fwd_call[grid](
                    x, ln_w, lower_bound, quant_rmsnorm_out, kernel_name="QuantOnly"
                )
                x = gather_from_sequence_parallel_region(
                    quant_rmsnorm_out,
                    self.sequence_dimension,
                    process_group=get_tp_group(self.config),
                )

            else:
                logger.debug(
                    "Running Quantized MLP kernel with external (native compiler) sequence-parallel RMSnorm!"
                )
                x = gather_from_sequence_parallel_region(
                    x, self.sequence_dimension, process_group=get_tp_group(self.config)
                )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x_orig_dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        gate_w_scale = self.gate_proj.weight_scale
        up_w = self.up_proj.weight.data
        up_w_scale = self.up_proj.weight_scale
        down_w = self.down_proj.weight.data
        down_w_scale = self.down_proj.weight_scale
        lower_bound = self.quantized_kernel_lower_bound

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                lower_bound,
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                lower_bound,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(output_tensor)

        logger.debug(f"Quantized MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _kernel_enabled_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        fused_residual = residual is not None
        logger.debug(
            f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        )

        # Choose which kernel to call
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "MLP kernel cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x.dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        grid = (vnc(self.logical_neuron_cores),)

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w  # rmsnorm
                gate_w,  # gate_w
                up_w,  # up_w
                down_w,  # down_w
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,
                up_w,
                down_w,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(
                output_tensor, process_group=get_tp_group(self.config)
            )

        logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, rmsnorm, adapter_ids=None):
        # logger.debug("MLP: native compiler")
        # # all-gather is done here instead of CPL layers to
        # # avoid 2 all-gathers from up and gate projections
        # if self.sequence_parallel_enabled:
        #     x = gather_from_sequence_parallel_region(
        #         x, self.sequence_dimension, process_group=get_tp_group(self.config)
        #     )

        # gate_proj_output = (
        #     self.gate_proj(x, rmsnorm)
        #     if not is_lora_module(self.gate_proj)
        #     else self.gate_proj(x, adapter_ids)
        # )

        # up_proj_output = (
        #     self.up_proj(x, rmsnorm) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        # )

        # down_proj_input = self.act_fn(gate_proj_output) * up_proj_output

        # output = (
        #     self.down_proj(down_proj_input)
        #     if not is_lora_module(self.up_proj)
        #     else self.down_proj(down_proj_input, adapter_ids)
        # )

        #################
        
        


        gate_proj_output, up_proj_output =nki_fuse_up_gate_matmul(x, self.gate_proj.weight, self.up_proj.weight)
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
       
        output = (
            self.down_proj(down_proj_input)
            if not is_lora_module(self.up_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )
        
        logger.debug(f"MLP output shape {output.shape}")
        return output
    
    #############################################

        # logger.info("-"*30 + " nki_matmul_fully_optimized in MLP " + "-"*30)
        # b, s, h = x.shape
        # if s != 1:
        #     x = x.view(-1, h)

        #     up = nki_matmul_fully_optimized_(x.t(), self.up_proj.weight.t())
        #     gate = nki_matmul_fully_optimized_(x.t(), self.gate_proj.weight.t())
        #     act = self.act_fn(gate) * up
        #     output = nki_matmul_fully_optimized_(act.t() , self.down_proj.weight.t())
        # else:
        #     gate_proj_output = (
        #     self.gate_proj(x, rmsnorm)
        #     if not is_lora_module(self.gate_proj)
        #     else self.gate_proj(x, adapter_ids)
        # )

        #     up_proj_output = (
        #         self.up_proj(x, rmsnorm) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        #     )

        #     down_proj_input = self.act_fn(gate_proj_output) * up_proj_output

        #     output = (
        #         self.down_proj(down_proj_input)
        #         if not is_lora_module(self.up_proj)
        #         else self.down_proj(down_proj_input, adapter_ids)
        #     )


        # return output

    

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """
        if self.mlp_kernel_enabled:
            fused_rmsnorm = not self.sequence_parallel_enabled
            # Quantized MLP kernel
            if self.quantized_mlp_kernel_enabled:
                return self._kernel_enabled_quantized_mlp(
                    x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
                )
            # MLP kernel
            return self._kernel_enabled_mlp(
                x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
            )
        else:
            # No kernel
            return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)

class NeuronLlamaAttentionBase(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(config=config, tensor_model_parallel_group=tensor_model_parallel_group,
                         hidden_size=config.hidden_size, num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads)

        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.neuron_config.padding_side
        self.torch_dtype = config.neuron_config.torch_dtype
        self.is_medusa = config.neuron_config.is_medusa
        self.flash_decoding_enabled = config.neuron_config.flash_decoding_enabled
        self.num_cores_per_group = config.num_cores_per_group
        self.bias = getattr(config, "attention_bias", False)
        self.rpl_reduce_dtype = config.neuron_config.rpl_reduce_dtype
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.rms_norm_eps = config.rms_norm_eps
        self.logical_nc_config = 1

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = self.config.neuron_config.tp_degree
        else:
            self.tp_degree = 1

        self.fused_qkv = config.neuron_config.fused_qkv
        self.clip_qkv = None

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        logger.debug(
            f"Hello from NeuronLlamaAttention init! Is SP enabled? {self.sequence_parallel_enabled}. Dim? {self.sequence_dimension}"
        )

        self.init_gqa_properties()

        self.init_rope()


    def init_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

        self.qkv_proj = NkiGroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.o_bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            logical_neuron_cores=self.neuron_config.logical_neuron_cores,
        )
        self.o_proj = NkiGroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.o_bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(
            self.qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.logical_neuron_cores = self.neuron_config.logical_neuron_cores


    def init_rope(self):  # rope: Rotary Position Embedding
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
            # TODO(yihsian): Check if we can just use our own implementation
            if self.is_medusa:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            rope_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    factor=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(self.config)


@nki.jit
def fused_self_attn_for_SD_small_head_size(Q_ref, K_ref, V_ref, orig_seqlen, use_causal_mask=False,
                                           mixed_precision=True, att_mask = None):
  """
  Fused self attention kernel for small head dimension Stable Diffusion workload, 
  simplified for this tutorial. 
  
  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask 
  application. Does not include QKV projection, output projection, dropout, 
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the 
  d_head is smaller or equal to 128. Assertion is thrown if `d_head` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (seq_q, d_head)
   - k_ptr: shape   (seq_k, d_head)
   - v_ptr: shape   (seq_v, d_head)
   - out_ptr: shape (seq_q, d_head)
   - We use seq_q and seq_k and seq_v just for clarity, this kernel requires 
   seq_q == seq_k == seq_v

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_precision is True, then all Tensor Engine operation will be performed in
   bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
   will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  b_i = nl.program_id(0)
  h_i = nl.program_id(1)
  
  q_ref = Q_ref[b_i, h_i]
  k_ref = K_ref[b_i, h_i]
  v_ref = V_ref[b_i, h_i]
    
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  seqlen, d_head = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (seqlen,d_head), \
  f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  B_sz = Q_ref.shape[0]
  H_sz = Q_ref.shape[1]
  assert B_sz == K_ref.shape[0] == V_ref.shape[0], 'Batch size mismatch!'
  assert H_sz == K_ref.shape[1] == V_ref.shape[1], 'Head size mismatch!'

  if att_mask is not None:
    assert att_mask.shape == (seqlen, seqlen), 'Attention mask shape mismatch!'
  
  out_ref = nl.ndarray((B_sz, H_sz, seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 1.0 / np.sqrt(d_head)

  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the dimension of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    ip_v = nl.arange(v_seq_tile_size)[:, None]
    if_v = nl.arange(d_head_tile_size)[None, :]
    trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
      v_ref[i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
      dtype=pe_in_dt)

  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
      q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]
      ],
      dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
      k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      ##############################################################
      # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
      ##############################################################
      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],
                                              stationary=q_local[i_q_seq_tile, ip_q, if_q])

      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      # if use_causal_mask:
      #   # Magic number -9984.0 to replace -inf similar to what neuronx-cc uses
      #   qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
      #     pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
      #     on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
      # else:
      #   # Simply send psum result back to sbuf
      #   qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
      #                                                                         dtype=kernel_dtype)
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what neuronx-cc uses
        # qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
        #   pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
        #   on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
        pred = ((i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk) & (i_q_seq_tile * q_seq_tile_size + ip_qk < orig_seqlen) & (i_k_seq_tile * k_seq_tile_size + if_qk < orig_seqlen)) 
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.where(condition=pred, x=qk_psum[ip_qk, if_qk], y=-3.38953139e+38)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)


      
    #   if att_mask is not None:
    #     mask_sbuf = nl.load(att_mask[i_q_seq_tile * q_seq_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk], dtype=att_mask.dtype)
    #     qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.where(mask_sbuf, qk_psum[ip_qk, if_qk], -3.38953139e+38, dtype=kernel_dtype)
    #   else:
    #     qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

      # if att_mask is not None:
      #   print(f"({i_q_seq_tile, i_k_seq_tile})")
      #   nl.device_print("", att_mask[i_q_seq_tile * q_seq_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk])
      
      ###################################
      # Step 4. Softmax
      ###################################
      neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.tensor_reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    # Buffer for transposed softmax results (FP32 in PSUM)
    trans_softmax_res = nl.ndarray(
      (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    # Result psum buffer has the hidden dim as P
    attn_res_psum = nl.zeros((par_dim(q_seq_tile_size), d_head_tile_size),
                             dtype=np.float32, buffer=nl.psum)

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    # Loop over matmul_1 contraction
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ###################################
      # Step 5. transpose(softmax_res)
      ###################################
      ip_scores = nl.arange(q_seq_tile_size)[:, None]
      if_scores = nl.arange(k_seq_tile_size)[None, :]

      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nl.arange(q_seq_tile_size)[:, None]
    if_out = nl.arange(d_head_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ######################################################################
      # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
      ######################################################################
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += \
        nisa.nc_matmul(stationary=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       moving=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * (sum_divisor[ip_sum_res, if_sum_res])

    nl.store(
      out_ref[b_i, h_i, i_q_seq_tile * q_seq_tile_size + ip_out, if_out],
      value=attn_res_div)

  return out_ref


@nki.jit
def fused_self_attn_for_SD_small_head_size_wo_spmd(Q_ref, K_ref, V_ref, use_causal_mask=False,
                                           mixed_precision=True, att_mask = None):
  """
  Fused self attention kernel for small head dimension Stable Diffusion workload, 
  simplified for this tutorial. 
  
  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask 
  application. Does not include QKV projection, output projection, dropout, 
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the 
  d_head is smaller or equal to 128. Assertion is thrown if `d_head` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (seq_q, d_head)
   - k_ptr: shape   (seq_k, d_head)
   - v_ptr: shape   (seq_v, d_head)
   - out_ptr: shape (seq_q, d_head)
   - We use seq_q and seq_k and seq_v just for clarity, this kernel requires 
   seq_q == seq_k == seq_v

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_precision is True, then all Tensor Engine operation will be performed in
   bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
   will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
#   b_i = nl.program_id(0)
#   h_i = nl.program_id(1)
  
    
  kernel_dtype = Q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
#   assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  B_sz, H_sz, seqlen, d_head = Q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
#   assert tuple(q_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
#   assert tuple(k_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
#   assert tuple(v_ref.shape) == (seqlen,d_head), \
#   f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  assert B_sz == K_ref.shape[0] == V_ref.shape[0], 'Batch size mismatch!'
  assert H_sz == K_ref.shape[1] == V_ref.shape[1], 'Head size mismatch!'

  if att_mask is not None:
    assert att_mask.shape == (seqlen, seqlen), 'Attention mask shape mismatch!'
  
  out_ref = nl.ndarray((B_sz, H_sz, seqlen, d_head), dtype=Q_ref.dtype, buffer=nl.shared_hbm)

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 1.0 / np.sqrt(d_head)

  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the dimension of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  
  for b_i in nl.affine_range(B_sz):
    for h_i in nl.affine_range(H_sz):
        q_ref = Q_ref[b_i, h_i]
        k_ref = K_ref[b_i, h_i]
        v_ref = V_ref[b_i, h_i]
      ###################################
        # Step 1. transpose(tensor_v)
        ###################################
        # Buffer for v matrix transposed
        # Pre-fetch and keep it in SBUF throughout different softmax tiles
        trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ip_v = nl.arange(v_seq_tile_size)[:, None]
            if_v = nl.arange(d_head_tile_size)[None, :]
            trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
            v_ref[i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
            dtype=pe_in_dt)

        q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
        ip_q = nl.arange(d_head_tile_size)[:, None]
        if_q = nl.arange(q_seq_tile_size)[None, :]
        for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
            q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
            q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
                    nl.arange(d_head_tile_size)[None, :]
            ],
            dtype=pe_in_dt) * softmax_scale

        k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
        ip_k = nl.arange(d_head_tile_size)[:, None]
        if_k = nl.arange(k_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
                    nl.arange(d_head_tile_size)[None, :]],
            dtype=pe_in_dt)

        for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
            # A SBUF buffer for an independent softmax tile
            qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

            neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
            ip_max = nl.arange(q_seq_tile_size)[:, None]
            if_max = nl.arange(k_seq_n_tiles)[None, :]

            # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
            for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

                # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
                # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
                qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                                    dtype=np.float32, buffer=nl.psum)

                # Tensor indices for accessing qk result in k_seq_tile_size
                ip_qk = nl.arange(q_seq_tile_size)[:, None]
                if_qk = nl.arange(k_seq_tile_size)[None, :]

                ##############################################################
                # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
                ##############################################################
                qk_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],
                                                        stationary=q_local[i_q_seq_tile, ip_q, if_q])

                ###################################
                # Step 3. Apply optional causal mask
                ###################################
                # if use_causal_mask:
                #   # Magic number -9984.0 to replace -inf similar to what neuronx-cc uses
                #   qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
                #     pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
                #     on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
                # else:
                #   # Simply send psum result back to sbuf
                #   qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                #                                                                         dtype=kernel_dtype)
                
                
                if att_mask is not None:
                    mask_sbuf = nl.load(att_mask[i_q_seq_tile * q_seq_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk], dtype=att_mask.dtype)
                    qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.where(mask_sbuf, qk_psum[ip_qk, if_qk], -3.38953139e+38, dtype=kernel_dtype)
                else:
                    qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

                # if att_mask is not None:
                #   print(f"({i_q_seq_tile, i_k_seq_tile})")
                #   nl.device_print("", att_mask[i_q_seq_tile * q_seq_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk])
                
                ###################################
                # Step 4. Softmax
                ###################################
                neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                    np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
                    axis=(1,), dtype=kernel_dtype, negate=True)

            neg_max_res_final = nisa.tensor_reduce(
            np.min, data=neg_max_res[ip_max, if_max],
            axis=(1,), dtype=kernel_dtype, negate=False)

            ip_softmax = nl.arange(q_seq_tile_size)[:, None]
            if_softmax = nl.arange(seqlen)[None, :]
            ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
            if_sum_res = nl.arange(d_head_tile_size)[None, :]

            softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
            sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

            # Simply use a large tile of seq_len in size since this is a "blocking" instruction
            # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
            exp_res = nisa.activation(np.exp,
                                    data=qk_res_buf[ip_softmax, if_softmax],
                                    bias=neg_max_res_final, scale=1.0)

            sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
                                dtype=kernel_dtype)
            softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

            sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
            sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

            # Buffer for transposed softmax results (FP32 in PSUM)
            trans_softmax_res = nl.ndarray(
            (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt)

            # Result psum buffer has the hidden dim as P
            attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                                    dtype=np.float32, buffer=nl.psum)

            ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
            if_scores_t = nl.arange(q_seq_tile_size)[None, :]
            # Loop over matmul_1 contraction
            for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
                ###################################
                # Step 5. transpose(softmax_res)
                ###################################
                ip_scores = nl.arange(q_seq_tile_size)[:, None]
                if_scores = nl.arange(k_seq_tile_size)[None, :]

                trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
                    softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

            ip_out = nl.arange(d_head_tile_size)[:, None]
            if_out = nl.arange(q_seq_tile_size)[None, :]
            for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ######################################################################
            # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
            ######################################################################
                ip_v_t = nl.arange(k_seq_tile_size)[:, None]
                if_v_t = nl.arange(d_head_tile_size)[None, :]
                attn_res_psum[ip_out, if_out] += \
                    nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                                stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

            attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

            attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])

            nl.store(
            out_ref[b_i, h_i ,i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
            value=attn_res_div)
                   

  return out_ref




def fused_att_interface(Q, K, V, use_causal_mask=True, mixed_precision=True, att_mask=None):
  B, H, S, D = Q.shape
  assert K.shape == V.shape == (B, H, S, D), 'Input shape mismatch!'
  if att_mask is not None:
    assert att_mask.shape == (S, S), 'Attention mask shape mismatch!'
#   if S < 128:
#     softmax_scale = 1.0 / math.sqrt(D)
#     q_scaled = Q * softmax_scale
#     raw_score = torch.matmul(q_scaled, K.transpose(3, 2))
#     if att_mask is not None:
#       raw_score = torch.where(att_mask, raw_score, -9984.0)
    
#     norm_score = torch.nn.functional.softmax(raw_score, dim=-1)

#     return torch.matmul(norm_score, V)

  S_padded = ((S + 127) // 128) * 128  
  padding_needed = S_padded - S  
  Q = torch.nn.functional.pad(Q, (0, 0, 0, padding_needed))
  K = torch.nn.functional.pad(K, (0, 0, 0, padding_needed))
  V = torch.nn.functional.pad(V, (0, 0, 0, padding_needed))
#   pad_mask = torch.zeros((S_padded, S_padded), dtype=torch.bool).to(device=Q.device)
#   if att_mask is not None:
#     pad_mask[:S, :S] = att_mask
#   else:
#     pad_mask[:S, :S] = True
  # TODO: Handle the case where S_padded > 2**14, the following code is incorrect at the softmax step
  if S_padded > 2**14:
    chunk_size = 2**14
    Q_chunks = torch.split(Q, chunk_size, dim=2)
    K_chunks = torch.split(K, chunk_size, dim=2)
    V_chunks = torch.split(V, chunk_size, dim=2)
    processed_chunks = []
    for q_chunk, k_chunk, v_chunk in zip(Q_chunks, K_chunks, V_chunks):
      processed_chunk = fused_self_attn_for_SD_small_head_size[B, H](q_chunk, k_chunk, v_chunk, use_causal_mask, mixed_precision, att_mask)
      processed_chunks.append(processed_chunk)
    return torch.cat(processed_chunks, dim=2)[:, :, :S, :]
  else:
    return fused_self_attn_for_SD_small_head_size[B, H](Q, K, V, S, use_causal_mask, mixed_precision)[:, :, :S, :]



@register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(NeuronLlamaAttentionBase):
    
    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len, attention_mask)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            logger.debug(f"ATTN kernel: logical_neuron_cores={self.logical_neuron_cores}")
            # if we are using left padding, then the bzs needs be 1 (otherwise we get wrong result
            # because flash attention does not use attention_mask). In practice, we use right
            # padding so this is unlikely to cause issues
            assert self.padding_side == "right" or bsz == 1

            # original shape of q, k, v is BHSD, and expected output is also BHSD.
            logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # make sure to cast inputs to torch_dtype (this is needed because the downcast to bf16
            # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.

            # original Q shape: batch, num_heads, seqlen, d_head
            Q = (
                Q.permute(0, 1, 3, 2)  # after permute: batch, num_heads, d_head, seqlen
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.torch_dtype)
            )
            Q = Q / math.sqrt(self.head_dim)
            K_active = (
                K_active.permute(0, 1, 3, 2)
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.torch_dtype)
            )
            V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(
                self.torch_dtype
            )
            # shape: (B*H)DS
            attn_output = torch.zeros(
                bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
            )

            logger.debug("Input parameter shapes")
            logger.debug(f"Q input shape {Q.shape}")
            logger.debug(f"K input shape {K_active.shape}")
            logger.debug(f"V input shape {V_active.shape}")
            logger.debug(f"Attn output shape {attn_output.shape}")

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (vnc(self.logical_neuron_cores),)

                # _flash_fwd_call_bir[grid](
                #     Q,
                #     K_active,
                #     V_active,
                #     1.0,
                #     attn_output,
                #     kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                # )
                pass
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                # _flash_fwd_call_bir(
                #     Q,
                #     K_active,
                #     V_active,
                #     1.0,
                #     attn_output,
                #     kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                # )
                pass
            else:
                raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

            # shape: BHDS
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            # active_scores = self.scaled_qk(Q, K_active, attention_mask)
            # active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
            #     Q.dtype
            # )
            # attn_output = torch.matmul(active_scores, V_active)
            attn_output = fused_att_interface(Q, K_active, V_active)#, att_mask=attention_mask.view(q_len, q_len))
        return attn_output, flash_attn_strategy


    def compute_for_token_gen(
        self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask
    ) -> Tensor:
        """attention computation at token generation phase"""
        is_speculation = position_ids.shape[-1] > 1

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
        if is_speculation:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores, is_speculation)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output



# TODO: Modularize RotaryEmbedding. See how HF transformers does it in 4.43.
class Llama3RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.43 impl
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L78
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/modeling_rope_utils.py#L345

    This implementation ensures inv_freq is calculated and stored in fp32.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=500000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()
        self.dim = dim # 64
        self.max_position_embeddings = max_position_embeddings # 131072 = 128 k
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings # 8192
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size] e.g.[1,8,32,64] x doesn't involve computation #TODO: sdh when position_len > self.old_context_len ?
        if self.inv_freq is None:
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim) 
            )

            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / self.factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                        self.high_freq_factor - self.low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

        inv_freq_expanded = (  #[1, 32, 1]
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float() #[1, 1, 32]
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) #[1,32,32]
            emb = torch.cat((freqs, freqs), dim=-1)#[1,32,64]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _LLAMA_MODULE_MAP[config.neuron_config.attn_cls](
            config=config, tensor_model_parallel_group=get_tp_group(config)
        )
        self.mlp = NeuronLlamaMLP(config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = get_rmsnorm_cls()(
                config.hidden_size,
                eps=config.rms_norm_eps,
                nki_enabled=config.neuron_config.nki_enabled,
            )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
            nki_enabled=config.neuron_config.nki_enabled,
        )
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = config.neuron_config.rmsnorm_quantize_kernel_enabled
        self.mlp_kernel_fuse_residual_add = config.neuron_config.mlp_kernel_fuse_residual_add
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            # import time
            # s = time.perf_counter()
            hidden_states = self.input_layernorm(hidden_states)
            # print("rms norm time: ", (time.perf_counter() - s) * 1000)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            **kwargs,
        )

        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert (
                not self.sequence_parallel_enabled
            ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            # First residual add handled in the MLP kernel
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
                adapter_ids=adapter_ids,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            # RMSNorm (fused with QKV kernel when SP is disabled)
            if not self.mlp_kernel_enabled or self.sequence_parallel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                adapter_ids=adapter_ids,
            )

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        return outputs

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class NeuronLlamaModel(NeuronBaseModel):
    """
    The neuron version of the LlamaModel
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            self.lm_head = NkiColumnParallelLinear(
                config.hidden_size, # 2048
                config.vocab_size,  #128256
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        # In the target fp8 checkpoint, the 1st and last
        # layers are not using fp8.
        updated_configs = []
        for i in range(config.num_hidden_layers):
            # TODO: Remove hardcoded code to have non-quantized MLPs for first and last decoder block
            if i == 0 or i == config.num_hidden_layers - 1:
                non_quant_config = copy.deepcopy(config)
                non_quant_config.neuron_config.quantized_mlp_kernel_enabled = False
                updated_configs.append(non_quant_config)
            else:
                updated_configs.append(config)
        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(conf) for conf in updated_configs])
        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps, nki_enabled=config.neuron_config.nki_enabled)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            self.fc = NkiColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.medusa_speculation_length = config.neuron_config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = NkiColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(
                        config.hidden_size,
                        config.vocab_size,
                        gather_output=not self.on_device_sampling,
                        bias=False,
                    ),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)

class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path):
        return LlamaForCausalLM.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
    
    
    def get_compiler_args(self) -> str:
        import sys
        print(f"[DEBUG] get_compiler_args called on {self.__class__.__name__}", flush=True, file=sys.stderr)

        return (
            "--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={self.neuron_config.cc_pipeline_tiling_factor}'"
            f" --lnc={self.neuron_config.logical_nc_config}"
            " -O1"
        )