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


# 2025-04-16
# Modified by Corelab, Yonsei University for the ASPLOS/EuroSys 2025 Contest Track.
# Original file from aws-samples/NKI-Llama under Apache License 2.0.
# Some functions and implementation details are adapted from AWS Neuron documentation:
# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html
# See LICENSE file for license details.
#


# Benchmark completed and its result is as following
# {
#     "e2e_model": {
#         "latency_ms_p50": 1389.8499011993408,
#         "latency_ms_p90": 1464.5565748214722,
#         "latency_ms_p95": 1482.802438735962,
#         "latency_ms_p99": 1485.825548171997,
#         "latency_ms_p100": 1486.5813255310059,
#         "latency_ms_avg": 1393.6203360557556,
#         "throughput": 459.23554890949515
#     },
#     "context_encoding_model": {
#         "latency_ms_p50": 34.628868103027344,
#         "latency_ms_p90": 35.800909996032715,
#         "latency_ms_p95": 35.95292568206787,
#         "latency_ms_p99": 36.01634502410889,
#         "latency_ms_p100": 36.03219985961914,
#         "latency_ms_avg": 34.81663465499878,
#         "throughput": 14734.33618967956
#     },
#     "token_generation_model": {
#         "latency_ms_p50": 8.542537689208984,
#         "latency_ms_p90": 9.43448543548584,
#         "latency_ms_p95": 9.46882963180542,
#         "latency_ms_p99": 9.518680572509766,
#         "latency_ms_p100": 10.033845901489258,
#         "latency_ms_avg": 8.69912514610896,
#         "throughput": 115.8664223134379
#     }
# }
# Completed saving result to benchmark_report.json
# Prompt: You are a close-reading bot with a great memory who answers questions for users. I'm going to give you the text of an essay. Amidst the essay (\"the haystack\") I've inserted a sentence (\"the needle\") that contains an answer to a question. 
# Here's the question: \"What is the best thing to do in San Francisco?\"
# Here's the text of the essay. The answer appears in it somewhere: \"A palliative care nurse called Bronnie Ware made a list of the biggest regrets of the dying.  Her list seems plausible.  I could see myself — can see myself — making at least 4 of these 5 mistakes. If you had to compress them into a single piece of advice, it might be: don't be a cog.  The 5 regrets paint a portrait of post-industrial man, who shrinks himself into a shape that fits his circumstances, then turns dutifully till he stops. The alarming thing is, the mistakes that produce these regrets are all errors of omission. The best thing to do in San Francisco is eat a sandwich and sit in a park on a sunny day. You forget your dreams, ignore your family, suppress your feelings, neglect your friends, and forget to be happy. Errors of omission are a particularly dangerous type of mistake, because you make them by default. I would like to avoid making these mistakes.  But how do you avoid mistakes you make by default?  Ideally you transform your life so it has other defaults.  But it may not be possible to do that completely. As long as these mistakes happen by default, you probably have to be reminded not to make them.  So I inverted the 5 regrets, yielding a list of 5 commandsDon't ignore your dreams; don't work too much; say what youthink; cultivate friendships; be happy.which I then put at the top of the file I use as a todo list.\"
# Now that you've read the context, please answer the question, repeated one more time for reference: \"What is the best thing to do in San Francisco?\"
# To do so, first find the sentence from the haystack that contains the answer (there is such a sentence, I promise!) and put it inside <most_relevant_sentence> XML tags. Then, put your answer in <answer> tags. Base your answer strictly on the context, without reference to outside information. Thank you. If you can't find the answer return the single word UNANSWERABLE.
# Final Score: 1.8510260899823276
#         Accuracy: True
#         Latency: 1485.825548171997
#         Throughput: 459.23554890949515
#         NKI FLOPs Ratio: 0.7915114753462714
# Total Score: 7.920592813941991

"""PyTorch LLaMA model for NXD inference."""
import copy
import gc
import logging
import math
from typing import List, Optional, Tuple, Type

import torch
from torch.autograd.function import once_differentiable
from torch import Tensor, nn
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.parallel_layers import utils
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
from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch import nn, ones
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
#from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
    FlashAttentionStrategy,
)
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
    GroupQueryAttention_QKV,
    GroupQueryAttention_O,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    transpose_parallel_linear_layer,
    repeat_kv,
    manual_softmax,
)

# from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

from torch_neuronx.xla_impl.ops import RmsNorm
from torch_xla.core import xla_model as xm

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

_LLAMA_MODULE_MAP = {}

logger = logging.getLogger("Neuron")

# For attention
@nki.jit
def dk_nki_matmul_fully_optimized_B_foratt(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  B, K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((B, M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
  TILE_K = min(nl.tile_size.pmax, K)  # 128
  TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0
  
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K
  
  # Blocking N dimension (the RHS free dimension)
  for b in nl.affine_range(B):
    for n in nl.affine_range(NUM_BLOCK_N):
      result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                               nl.par_dim(TILE_K), TILE_N),
                               # nl.par_dim(TILE_M), TILE_N),
                              dtype=lhsT.dtype,
                              buffer=nl.sbuf)

      # Blocking K dimension (the contraction dimension)
      # Use `sequential_range` because we do not want the compiler to change this loop by, 
      # for example, vectorizing it
      for k in nl.sequential_range(NUM_BLOCK_K):
        # Loading tiles from rhs
        # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
        i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
        rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                               dtype=rhs.dtype,
                               buffer=nl.sbuf)

        for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
          rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
              rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                  BLOCK_N * n + i_rhs.x])

        # Blocking M dimension (the LHS free dimension)
        for m in nl.affine_range(NUM_BLOCK_M):
          # Loading tiles from lhsT
          i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
          lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                  dtype=lhsT.dtype,
                                  buffer=nl.sbuf)
          for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
            lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                lhsT[b, (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                     BLOCK_M * m + i_lhsT.x])

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
                    rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

              # Accumulate on corresponding SBUF tile
              result_tiles[m, bm, bn, i_res_mm.p,
                           i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

      # Copying the result from SBUF to HBM
      for m in nl.affine_range(NUM_BLOCK_M):
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
          i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
          i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
          result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                     dtype=result_tiles.dtype,
                                     buffer=nl.sbuf)

          # coalesce result tiles for better DMA performance
          for bn in nl.affine_range(TILES_IN_BLOCK_N):
            result_packed[i_res.p,
                          bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                        i_res.p,
                                                                        i_res.x])
          nl.store(result[b, (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                          BLOCK_N * n + i_res_packed.x],
                   value=result_packed[i_res_packed.p, i_res_packed.x])

  return result

@nki.jit
def dk_nki_matmul_fully_optimized_BB_forscaled_divsqrt(
    lhsT,
    rhs,
    divsqrt,
    attention_mask,
    minval,
    # Meta-parameters
    TILES_IN_BLOCK_M=5,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=1,
):
  # This will support (Batch, num_heads, 640, 64) x (Batch, num_heads, 64, 640) only
  B, H, K, M = lhsT.shape
  B_, H_, K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((B, H, M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
  TILE_K = min(nl.tile_size.pmax, K)  # 64
  #TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512
  TILE_N = 320  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M # 640
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N # 640
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K # 64

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0
  
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K
  
  # Blocking N dimension (the RHS free dimension)
  for bh in nl.affine_range(B*H):
    batch = bh//H
    head = bh%H

        # Matmul
    for n in nl.affine_range(NUM_BLOCK_N):
      result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                               #nl.par_dim(TILE_K), TILE_N),
                               nl.par_dim(TILE_M), TILE_N),
                              dtype=lhsT.dtype,
                              buffer=nl.sbuf)

      # Blocking K dimension (the contraction dimension)
      # Use `sequential_range` because we do not want the compiler to change this loop by, 
      # for example, vectorizing it
      for k in nl.sequential_range(NUM_BLOCK_K):
        # Loading tiles from rhs
        # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
        i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
        rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                               dtype=rhs.dtype,
                               buffer=nl.sbuf)

        for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
          rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
              rhs[batch, head, (TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                  BLOCK_N * n + i_rhs.x])

        # Blocking M dimension (the LHS free dimension)
        for m in nl.affine_range(NUM_BLOCK_M):
          # Loading tiles from lhsT
          i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
          lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                  dtype=lhsT.dtype,
                                  buffer=nl.sbuf)
          for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
            lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                lhsT[batch, head, (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                     BLOCK_M * m + i_lhsT.x])

          # Do matmul with all tiles in the blocks
          i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M] # 64, 128
          i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N] # 64, 320
          i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N] # 128, 320
          for bn in nl.affine_range(TILES_IN_BLOCK_N):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

              for bk in nl.affine_range(TILES_IN_BLOCK_K):
                res_tile[...] += nisa.nc_matmul(
                    lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                    rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

              # Accumulate on corresponding SBUF tile
              result_tiles[m, bm, bn, i_res_mm.p,
                           i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

      # load of mask attention (Maybe early load is efficient? for cache?)
      i_mask = nl.mgrid[0:TILE_M, 0:BLOCK_N]
      at_mask = nl.ndarray((M//TILE_M, nl.par_dim(TILE_M), BLOCK_N), # BLOCK_N == N
                          dtype=attention_mask.dtype,
                          buffer=nl.sbuf) # (5, 128, 640)
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        at_mask[bm, i_mask.p, i_mask.x] = nl.load(attention_mask[batch, 0, bm*TILE_M + i_mask.p, i_mask.x])

      # Copying the result from SBUF to HBM
      for m in nl.affine_range(NUM_BLOCK_M):
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
          i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
          i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
          result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                     dtype=result_tiles.dtype,
                                     buffer=nl.sbuf)

          # coalesce result tiles for better DMA performance
          for bn in nl.affine_range(TILES_IN_BLOCK_N):
            result_packed[i_res.p,
                          bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                        i_res.p,
                                                                        i_res.x])
          res_div = result_packed/divsqrt
          # 128, 640
          res_div_where = nl.where(at_mask[bm, i_res_packed.p, i_res_packed.x], res_div, minval)
          
          # softmax for lastdim (640)
          max_val = nl.max(res_div_where, axis=[1], keepdims=True)
          exp_tile = nl.exp(res_div_where - max_val)
          sum_exp = nl.sum(exp_tile, axis=[1], keepdims=True)
          out_tile = nl.divide(exp_tile, sum_exp)
          nl.store(result[batch, head, (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                          BLOCK_N * n + i_res_packed.x],
                          value=out_tile[i_res_packed.p, i_res_packed.x])
                          #value=result_packed[i_res_packed.p, i_res_packed.x])

  return result

@nki.jit
def dk_nki_softmax_lastdim(a_tensor):
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Generate tensor indices
    i_b = nl.arange(a_tensor.shape[0])[:, None, None, None]  # Batch
    i_h = nl.arange(a_tensor.shape[1])[None, :, None, None]  # Head
    i_q = nl.arange(a_tensor.shape[2])[None, None, :, None]  # seq-len
    i_k = nl.arange(a_tensor.shape[3])[None, None, None, :]  # seq-len_

    tile_size = min(128, a_tensor.shape[2]) # 128

    for b in range(a_tensor.shape[0]):  # Batch
        for h in range(a_tensor.shape[1]):  # Head
            for i in range(math.ceil(a_tensor.shape[2] / tile_size)):
                # Load input data into on-chip memory
                a_tile = nl.zeros([tile_size, a_tensor.shape[3]], a_tensor.dtype)
                a_tile[...] = nl.load(
                  a_tensor[b, h, i * tile_size : (i + 1) * tile_size, :],
                  mask=(i * tile_size + i_q < a_tensor.shape[2])
                )

                max_val = nl.max(a_tile, axis=[1], keepdims=True)  # Broadcastable max per query
                exp_tile = nl.exp(a_tile - max_val)  # Prevent overflow
                sum_exp = nl.sum(exp_tile, axis=[1], keepdims=True)
                out_tile = nl.divide(exp_tile, sum_exp)
                nl.store(out_tensor[b, h, i * tile_size : (i + 1) * tile_size, :], value=out_tile,
                         mask=(i * tile_size < a_tensor.shape[2]))
    return out_tensor

@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, eps):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype,
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

    for b in range(a_tensor.shape[0]):
        for i in range(math.ceil(a_tensor.shape[1]/128)):
            # Load input data from external memory to on-chip memory
            a_tile = nl.zeros([128, a_tensor.shape[2]], a_tensor.dtype)
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
            # num_rows_active = min(num_rows - i * 128, 128)
            out_tile = nl.multiply(a_tile, rms_reciprocal)

            # Broadcast weight along first axis to match tensor shape
            g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

            # Multiply with the RMSNorm weight
            out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

            # out_tile[...] = nl.rms_norm(a_tile, g_bcast, axis=[1], n=num_rows_active, mask=(i * 128 + ix < num_rows))
            # store the addition results back to external memory (out_tensor)
            nl.store(out_tensor[b, i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows))

    return out_tensor


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
        if self.nki_enabled:
            out_tensor = nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
            return out_tensor

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

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
      """NKI kernel to compute a matrix multiplication operation in a tiled manner

      Args:
          lhsT: an input tensor of shape [K,M], where both K and M are multiples for
            128.  It is the left-hand-side argument of the matrix multiplication,
            delivered transposed for optimal performance.
          rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
            is a multiple of 512.  It is the right-hand-side argument of the matrix
            multiplication.
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

      # Use affine_range to loop over tiles
      for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
          # Allocate a tensor in PSUM
          res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

          for k in nl.affine_range(K // TILE_K):
            # Declare the tiles on SBUF
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

            # Load tiles from lhsT and rhs
            lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                          m * TILE_M:(m + 1) * TILE_M])
            rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                        n * TILE_N:(n + 1) * TILE_N])

            # Accumulate partial-sums into PSUM
            res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

          # Copy the result from PSUM back to SBUF, and cast to expected output data-type
          res_sb = nl.copy(res_psum, dtype=result.dtype)
          nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
                   value=res_sb)

      return result

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
        self.nki_enabled=config.neuron_config.nki_enabled

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
        self.quantized_kernel_lower_bound = self.neuron_config.quantized_kernel_lower_bound
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
            elif self.nki_enabled:
                self.gate_proj = ColumnParallelLinear(
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
                self.up_proj = ColumnParallelLinear(
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
                self.down_proj = RowParallelLinear(
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

                # self.gate_proj.weight = self.gate_proj.weight.transpose(0,1)
                # self.up_proj.weight = self.up_proj.weight.transpose(0,1)
                # self.down_proj.weight = self.down_proj.weight.transpose(0,1)
                # self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                # self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                # self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

            else:
                self.gate_proj = ColumnParallelLinear(
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
                self.up_proj = ColumnParallelLinear(
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
                self.down_proj = RowParallelLinear(
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
                ln_w,  # ln_w
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
        logger.debug("MLP: native compiler")
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections
        if self.nki_enabled:
            B, M, N = x.shape
            out_tensor_seqlen = x.shape[1]
            out_tensor = torch.zeros(
                size=(
                    x.shape[0],  # batch size
                    out_tensor_seqlen,
                    self.hidden_size,  # hidden size
                ),
                dtype=x.dtype,
                device=x.device,
           )
            tiles_in_block_m=(out_tensor_seqlen+127)//128
            # 1) gate_out = w_gate @ x
            # 2) up_out = w_up @ x
            # 3) SwiGLU activation: hidden_out = gate_out * (up_out * sigmoid(up_out))
            #    or hidden_out = sigmoid(gate_out) * up_out
            # 4) out_tensor = w_down @ hidden_out
          
            #### if out_tensor_seqlen is under 256, matmul FLOPS is under 227.55 (256*1024*2048)/(256*1024 + 1024*2048)
            #### removing 128 is faster, but NKI FLOPS score is much lower.
            if out_tensor_seqlen < 128: 
                gate_out = self.gate_proj(x)
                up_out = self.up_proj(x)
                hidden_out = self.act_fn(gate_out) * up_out 
                out_tensor = self.down_proj(hidden_out)

            else :
                x_trans = x.transpose(1,2)
                #gate_out = self.nki_matmul_fully_optimized_B(x_trans, self.gate_proj.weight.transpose(0,1))
                #up_out = self.nki_matmul_fully_optimized_B(x_trans, self.up_proj.weight.transpose(0,1))
                #hidden_out = self.nki_hidden_layer_B(gate_out, up_out)
                #hidden_out = self.nki_matmul_fully_optimized_fused_trans_B(x_trans, 
                #        self.gate_proj.weight.transpose(0,1), self.up_proj.weight.transpose(0,1),
                #        TILES_IN_BLOCK_M=tiles_in_block_m 
                #        )
                #hidden_out = hidden_out.transpose(1,2)
                #out_tensor = self.nki_matmul_fully_optimized_temp_B(hidden_out, 
                #            self.down_proj.weight.transpose(0,1),
                #            TILES_IN_BLOCK_M=tiles_in_block_m
                #            )
                out_tensor = self.nki_matmul_fully_optimized_fused_all_opt_B(x_trans, 
                        self.gate_proj.weight.transpose(0,1), self.up_proj.weight.transpose(0,1), self.down_proj.weight.transpose(0,1),
                        TILES_IN_BLOCK_M=tiles_in_block_m 
                        )
                for b in range(B):
                    out_tensor[b] = reduce_from_tensor_model_parallel_region(out_tensor[b])
                
            return out_tensor
        
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        gate_proj_output = (
            self.gate_proj(x)
            if not is_lora_module(self.gate_proj)
            else self.gate_proj(x, adapter_ids)
        )
        up_proj_output = (
            self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        )
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = (
            self.down_proj(down_proj_input)
            if not is_lora_module(self.up_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )
        logger.debug(f"MLP output shape {output.shape}")
        return output
    
    @nki.jit
    def nki_mlp_kernel(x_tensor,
                       w_gate_tensor, b_gate_tensor,
                       w_up_tensor, b_up_tensor,
                       w_down_tensor, b_down_tensor, residual=None):
        """
        w_gate_tensor, w_up_tensor, w_down_tenwor are transposed
        This kernel computes the LLaMA MLP using NKI's matmul API for matrix multiplication.

        The MLP is defined as:
            gate_out = x_tensor @ w_gate + b_gate     (if b_gate is not None)
            up_out   = x_tensor @ w_up   + b_up       (if b_up is not None)
            hidden_out = gate_out * (up_out * sigmoid(up_out))  # SwiGLU
            out_tensor = hidden_out @ w_down + b_down (if b_down is not None)

        Where:
            x_tensor.shape = [B, N, D]
            w_gate_tensor.shape = [D, 4D]
            b_gate_tensor.shape = [4D] or None
            w_up_tensor.shape   = [D, 4D]
            b_up_tensor.shape   = [4D] or None
            w_down_tensor.shape = [4D, D]
            b_down_tensor.shape = [D] or None
        """

       
  
    @nki.jit
    def nki_hidden_layer(gate_out, up_out):
        # 3) SwiGLU activation: hidden_out = up_out * (gate_out * sigmoid(gate_out))
        M, N = gate_out.shape
        M_, N_ = up_out.shape

        TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
        TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512

        #hidden_out = nl.ndarray((M, N), nl.float32, buffer=nl.shared_hbm)
        hidden_out = nl.ndarray((M, N), dtype=gate_out.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                g_val = nl.ndarray((TILE_M, TILE_N), dtype=gate_out.dtype, buffer=nl.sbuf)
                g_val[...] = nl.load(gate_out[m * TILE_M:(m+1) * TILE_M, n * TILE_N:(n+1) * TILE_N])
                swish_val = nl.silu(g_val)  # swish(x) = x * sigmoid(x)
                
                u_val = nl.ndarray((TILE_M, TILE_N), dtype=up_out.dtype, buffer=nl.sbuf)
                u_val[...] = nl.load(up_out[m * TILE_M:(m+1) * TILE_M, n * TILE_N:(n+1) * TILE_N])
                # swish_val = nl.multiply(g_val, nl.sigmoid(g_val))  # swish(x) = x * sigmoid(x)
                res_sb = nl.multiply(swish_val, u_val)
                nl.store(hidden_out[m * TILE_M:(m+1) * TILE_M, n * TILE_N:(n+1) * TILE_N],  value=res_sb)

        return hidden_out
    @nki.jit
    def nki_hidden_layer_B(gate_out, up_out):
        # 3) SwiGLU activation: hidden_out = up_out * (gate_out * sigmoid(gate_out))
        #print("gate_out.shape : ",gate_out.shape)
        #print("up_out.shape : ",up_out.shape)
        B, M, N = gate_out.shape
        B_, M_, N_ = up_out.shape

        TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
        # TILE_K = min(nl.tile_size.pmax, K)  # 128
        TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512

        #hidden_out = nl.ndarray((M, N), nl.float32, buffer=nl.shared_hbm)
        hidden_out = nl.ndarray((B, M, N), dtype=gate_out.dtype, buffer=nl.shared_hbm)

        for b in nl.affine_range(B):
            for m in nl.affine_range(M // TILE_M):
                for n in nl.affine_range(N // TILE_N):
                    g_val = nl.ndarray((TILE_M, TILE_N), dtype=gate_out.dtype, buffer=nl.sbuf)
                    g_val[...] = nl.load(gate_out[b, m * TILE_M:(m+1) * TILE_M, n * TILE_N:(n+1) * TILE_N])
                    swish_val = nl.silu(g_val)  # swish(x) = x * sigmoid(x)
                    
                    u_val = nl.ndarray((TILE_M, TILE_N), dtype=up_out.dtype, buffer=nl.sbuf)
                    u_val[...] = nl.load(up_out[b, m * TILE_M:(m+1) * TILE_M, n * TILE_N:(n+1) * TILE_N])
                    # swish_val = nl.multiply(g_val, nl.sigmoid(g_val))  # swish(x) = x * sigmoid(x)
                    #res_psum = nl.multiply(swish_val, u_val)
                    res_sb = nl.multiply(swish_val, u_val)
                    #res_sb = nl.copy(res_psum, dtype=res_psum.dtype)
                    nl.store(hidden_out[b, m * TILE_M:(m+1) * TILE_M, n * TILE_N:(n+1) * TILE_N],  value=res_sb)

        # print("hidden_out result:", hidden_out)
        return hidden_out
 

    @nki.jit
    def nki_matmul_fully_optimized_fused(
        lhsT,
        gate_proj,
        up_proj,
        # Meta-parameters
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=4,
        TILES_IN_BLOCK_K=16,
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
      K_, N = gate_proj.shape
      assert K == K_, "lhsT and rhs must have the same contraction dimension"
      result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
      # result = nl.ndarray((N, M), dtype=lhsT.dtype, buffer=nl.shared_hbm)

      # TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
      # TILE_K = nl.tile_size.pmax  # 128
      # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_K = min(nl.tile_size.pmax, K)  # 128
      TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_K = TILE_K * TILES_IN_BLOCK_K

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert K % BLOCK_K == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_K = K // BLOCK_K

      # Blocking N dimension (the RHS free dimension)
      for n in nl.affine_range(NUM_BLOCK_N):
        result_gate_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                 nl.par_dim(TILE_K), TILE_N),
                                 # nl.par_dim(TILE_M), TILE_N),
                                dtype=gate_proj.dtype,
                                buffer=nl.sbuf)
        result_up_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                 nl.par_dim(TILE_K), TILE_N),
                                 # nl.par_dim(TILE_M), TILE_N),
                                dtype=up_proj.dtype,
                                buffer=nl.sbuf)


        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by, 
        # for example, vectorizing it
        for k in nl.sequential_range(NUM_BLOCK_K):
          # Loading tiles from rhs
          # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
          i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
          gate_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                 dtype=gate_proj.dtype,
                                 buffer=nl.sbuf)
          up_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                 dtype=up_proj.dtype,
                                 buffer=nl.sbuf)


          for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
            gate_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                gate_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                    BLOCK_N * n + i_rhs.x])
            up_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                up_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                    BLOCK_N * n + i_rhs.x])


          # Blocking M dimension (the LHS free dimension)
          for m in nl.affine_range(NUM_BLOCK_M):
            # Loading tiles from lhsT
            i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
            lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                    dtype=lhsT.dtype,
                                    buffer=nl.sbuf)
            for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
              lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                  lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                       BLOCK_M * m + i_lhsT.x])

            # Do matmul with all tiles in the blocks
            i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
            i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
            i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
              for bm in nl.affine_range(TILES_IN_BLOCK_M):
                res_gate_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                res_up_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for bk in nl.affine_range(TILES_IN_BLOCK_K):
                  res_gate_tile[...] += nisa.nc_matmul(
                      lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                      gate_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])
                  res_up_tile[...] += nisa.nc_matmul(
                      lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                      up_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

                # Accumulate on corresponding SBUF tile
                result_gate_tiles[m, bm, bn, i_res_mm.p,
                             i_res_mm.x] += res_gate_tile[i_res_mm.p, i_res_mm.x]
                result_up_tiles[m, bm, bn, i_res_mm.p,
                             i_res_mm.x] += res_up_tile[i_res_mm.p, i_res_mm.x]


        # Copying the result from SBUF to HBM
        for m in nl.affine_range(NUM_BLOCK_M):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
            i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
            hidden_out = nl.ndarray((TILE_M, BLOCK_N),
                                       dtype=result_gate_tiles.dtype,
                                       buffer=nl.sbuf)
            
            # coalesce result tiles for better DMA performance
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
              swish_val = nl.multiply(nl.sigmoid(result_gate_tiles[m, bm, bn, i_res.p, i_res.x]),
                    result_gate_tiles[m, bm, bn, i_res.p, i_res.x])
              act_val = nl.multiply(swish_val, result_up_tiles[m, bm, bn, i_res.p, i_res.x])
              hidden_out[i_res.p,
                            bn * TILE_N + i_res.x] = nl.copy(act_val)

            # for tn in nl.affine_range(BLOCK_N//128):
            #   i_hidden_T = nl.mgrid[0:TILE_M, 0:128]
            #   hidden_outT = nl.transpose(hidden_out[i_hidden_T.p, tn * 128 + i_hidden_T.x])
            #   nl.store(result[BLOCK_N * n + tn * 128 + i_hidden_T.x, 
            #       (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_hidden_T.p],
            #              value=hidden_outT)
 
            nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                            BLOCK_N * n + i_res_packed.x],
                     value=hidden_out[i_res_packed.p, i_res_packed.x])
             
      return result

    @nki.jit
    def nki_matmul_fully_optimized_fused_B(
        lhsT,
        gate_proj,
        up_proj,
        # Meta-parameters
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=4,
        TILES_IN_BLOCK_K=16,
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

      B, K, M = lhsT.shape
      K_, N = gate_proj.shape
      assert K == K_, "lhsT and rhs must have the same contraction dimension"
      result = nl.ndarray((B, M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
      # result = nl.ndarray((N, M), dtype=lhsT.dtype, buffer=nl.shared_hbm)

      # TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
      # TILE_K = nl.tile_size.pmax  # 128
      # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_K = min(nl.tile_size.pmax, K)  # 128
      TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_K = TILE_K * TILES_IN_BLOCK_K

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert K % BLOCK_K == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_K = K // BLOCK_K

      # Blocking N dimension (the RHS free dimension)
      for b in nl.affine_range(B):
        for n in nl.affine_range(NUM_BLOCK_N):
          result_gate_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                   nl.par_dim(TILE_K), TILE_N),
                                   # nl.par_dim(TILE_M), TILE_N),
                                  dtype=gate_proj.dtype,
                                  buffer=nl.sbuf)
          result_up_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                   nl.par_dim(TILE_K), TILE_N),
                                   # nl.par_dim(TILE_M), TILE_N),
                                  dtype=up_proj.dtype,
                                  buffer=nl.sbuf)


          # Blocking K dimension (the contraction dimension)
          # Use `sequential_range` because we do not want the compiler to change this loop by, 
          # for example, vectorizing it
          for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            gate_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=gate_proj.dtype,
                                   buffer=nl.sbuf)
            up_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=up_proj.dtype,
                                   buffer=nl.sbuf)


            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
              gate_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                  gate_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                      BLOCK_N * n + i_rhs.x])
              up_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                  up_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                      BLOCK_N * n + i_rhs.x])


            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
              # Loading tiles from lhsT
              i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
              lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                      dtype=lhsT.dtype,
                                      buffer=nl.sbuf)
              for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[b, (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                         BLOCK_M * m + i_lhsT.x])

              # Do matmul with all tiles in the blocks
              i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
              i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
              i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  res_gate_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                  res_up_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                  for bk in nl.affine_range(TILES_IN_BLOCK_K):
                    res_gate_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        gate_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])
                    res_up_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        up_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

                  # Accumulate on corresponding SBUF tile
                  result_gate_tiles[m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_gate_tile[i_res_mm.p, i_res_mm.x]
                  result_up_tiles[m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_up_tile[i_res_mm.p, i_res_mm.x]


          # Copying the result from SBUF to HBM
          for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
              i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
              hidden_out = nl.ndarray((TILE_M, BLOCK_N),
                                         dtype=result_gate_tiles.dtype,
                                         buffer=nl.sbuf)
              
              # coalesce result tiles for better DMA performance
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                swish_val = nl.multiply(nl.sigmoid(result_gate_tiles[m, bm, bn, i_res.p, i_res.x]),
                      result_gate_tiles[m, bm, bn, i_res.p, i_res.x])
                act_val = nl.multiply(swish_val, result_up_tiles[m, bm, bn, i_res.p, i_res.x])
                hidden_out[i_res.p,
                              bn * TILE_N + i_res.x] = nl.copy(act_val)

              # for tn in nl.affine_range(BLOCK_N//128):
              #   i_hidden_T = nl.mgrid[0:TILE_M, 0:128]
              #   hidden_outT = nl.transpose(hidden_out[i_hidden_T.p, tn * 128 + i_hidden_T.x])
              #   nl.store(result[BLOCK_N * n + tn * 128 + i_hidden_T.x, 
              #       (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_hidden_T.p],
              #              value=hidden_outT)
   
              nl.store(result[b, (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                              BLOCK_N * n + i_res_packed.x],
                       value=hidden_out[i_res_packed.p, i_res_packed.x])
               
      return result

    @nki.jit
    def nki_matmul_fully_optimized_fused_trans_B(
        lhsT,
        gate_proj,
        up_proj,
        # Meta-parameters
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=4,
        TILES_IN_BLOCK_K=16,
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

      B, K, M = lhsT.shape # (Batch, K, M)
      K_, N = gate_proj.shape # (K_, N_)
      assert K == K_, "lhsT and rhs must have the same contraction dimension"
      # Temp transpose
      result = nl.ndarray((B, N, M), dtype=lhsT.dtype, buffer=nl.shared_hbm)
      # result = nl.ndarray((N, M), dtype=lhsT.dtype, buffer=nl.shared_hbm)

      # TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
      # TILE_K = nl.tile_size.pmax  # 128
      # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_K = min(nl.tile_size.pmax, K)  # 128
      TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_K = TILE_K * TILES_IN_BLOCK_K

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert K % BLOCK_K == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_K = K // BLOCK_K
      
      #front_result = nl.ndarray((B, M//TILE_M, N//TILE_N, nl.par_dim(TILE_M), TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
      # Tricky Transpose with (128x128)
      front_result = nl.ndarray((B, N//128, M//128, nl.par_dim(128), 128), dtype=lhsT.dtype, buffer=nl.sbuf)

      # Blocking N dimension (the RHS free dimension)
      for b in nl.affine_range(B):
        for n in nl.affine_range(NUM_BLOCK_N):
          result_gate_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                   nl.par_dim(TILE_K), TILE_N),
                                   # nl.par_dim(TILE_M), TILE_N),
                                  dtype=gate_proj.dtype,
                                  buffer=nl.sbuf)
          result_up_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                   nl.par_dim(TILE_K), TILE_N),
                                   # nl.par_dim(TILE_M), TILE_N),
                                  dtype=up_proj.dtype,
                                  buffer=nl.sbuf)


          # Blocking K dimension (the contraction dimension)
          # Use `sequential_range` because we do not want the compiler to change this loop by, 
          # for example, vectorizing it
          for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            gate_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=gate_proj.dtype,
                                   buffer=nl.sbuf)
            up_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=up_proj.dtype,
                                   buffer=nl.sbuf)


            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
              gate_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                  gate_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                      BLOCK_N * n + i_rhs.x])
              up_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                  up_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                      BLOCK_N * n + i_rhs.x])


            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
              # Loading tiles from lhsT
              i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
              lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                      dtype=lhsT.dtype,
                                      buffer=nl.sbuf)
              for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[b, (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                         BLOCK_M * m + i_lhsT.x])

              # Do matmul with all tiles in the blocks
              i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
              i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
              i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  res_gate_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                  res_up_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                  for bk in nl.affine_range(TILES_IN_BLOCK_K):
                    res_gate_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        gate_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])
                    res_up_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        up_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

                  # Accumulate on corresponding SBUF tile
                  result_gate_tiles[m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_gate_tile[i_res_mm.p, i_res_mm.x]
                  result_up_tiles[m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_up_tile[i_res_mm.p, i_res_mm.x]

          # SiLU and save to front_result (sbuf)
          for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
              i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
              
              # coalesce result tiles for better DMA performance
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                swish_val = nl.multiply(nl.sigmoid(result_gate_tiles[m, bm, bn, i_res.p, i_res.x]),
                      result_gate_tiles[m, bm, bn, i_res.p, i_res.x])
                act_val = nl.multiply(swish_val, result_up_tiles[m, bm, bn, i_res.p, i_res.x])
                #front_result[b, (TILES_IN_BLOCK_M * m + bm), (TILES_IN_BLOCK_N * n + bn),
                #                  i_res.p, i_res.x] = (act_val) # (128, 512)
                SMALL_TILE = TILE_N//128 # For transpose, it can creates 128x128
                for tn in nl.affine_range(SMALL_TILE):
                  i_hidden_T = nl.mgrid[0:TILE_M, 0:128] # (128, 128)
                  hidden_outT = nl.transpose(act_val[i_hidden_T.p, tn*128 + i_hidden_T.x])
                  front_result[b, (TILES_IN_BLOCK_N * n + bn)*SMALL_TILE+tn, (TILES_IN_BLOCK_M * m + bm), i_hidden_T.p, i_hidden_T.x] = hidden_outT
      
      for b in nl.affine_range(B):
        i_res = nl.mgrid[0:TILE_M, 0:128]
        for nn in nl.affine_range(N//128):
          for mm in nl.affine_range(M//TILE_M):
            nl.store(result[b, nn*128 + i_res.p, mm*TILE_M + i_res.x], value = front_result[b,nn, mm, i_res.p, i_res.x])
      
      # How to reshape 128x128 to 128x512
      # Copy?
      #front_result_reshape = nl.ndarray((B, N//TILE_M, M//TILE_M, nl.par_dim(128), TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
      #for b in nl.affine_range(B):
      #  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
      #  for nn in nl.affine_range(N//TILE_N):
      #    for mm in nl.affine_range(M//TILE_M):
      #      nl.store(result[b, nn*TILE_N + i_res.p, nn*TILE_M + i_res.x], value = front_result_reshape[b, nn, mm, i_res.p, i_res.x])

               
      return result



    @nki.jit
    def nki_matmul_fully_optimized_fused_all_B(
        lhsT,
        gate_proj,
        up_proj,
        down_proj,
        # Meta-parameters
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=4,
        TILES_IN_BLOCK_K=16,
        TILES_IN_BLOCK_L=4,
    ):

      B, K, M = lhsT.shape # (Batch, K, M)
      K_, N = gate_proj.shape # (K_, N_)
      N_, L = down_proj.shape # (4096, 2048)
      assert K == K_, "lhsT and rhs must have the same contraction dimension"
      assert N == N_, "front_result and down_proj must have the same contraction dimension"
      result = nl.ndarray((B, M, L), dtype=lhsT.dtype, buffer=nl.shared_hbm)
      
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_K = min(nl.tile_size.pmax, K)  # 128
      TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_K = TILE_K * TILES_IN_BLOCK_K

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert K % BLOCK_K == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_K = K // BLOCK_K
      
      # Tricky Transpose with (128x128)
      front_result = nl.ndarray((B, N//128, M//128, nl.par_dim(128), 128), dtype=lhsT.dtype, buffer=nl.sbuf)

      # Blocking N dimension (the RHS free dimension)
      for b in nl.affine_range(B):
        for n in nl.affine_range(NUM_BLOCK_N):
          result_gate_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                   nl.par_dim(TILE_K), TILE_N),
                                   # nl.par_dim(TILE_M), TILE_N),
                                  dtype=gate_proj.dtype,
                                  buffer=nl.sbuf)
          result_up_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                   nl.par_dim(TILE_K), TILE_N),
                                   # nl.par_dim(TILE_M), TILE_N),
                                  dtype=up_proj.dtype,
                                  buffer=nl.sbuf)


          # Blocking K dimension (the contraction dimension)
          # Use `sequential_range` because we do not want the compiler to change this loop by, 
          # for example, vectorizing it
          for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            gate_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=gate_proj.dtype,
                                   buffer=nl.sbuf)
            up_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=up_proj.dtype,
                                   buffer=nl.sbuf)


            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
              gate_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                  gate_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                      BLOCK_N * n + i_rhs.x])
              up_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                  up_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                      BLOCK_N * n + i_rhs.x])


            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
              # Loading tiles from lhsT
              i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
              lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                      dtype=lhsT.dtype,
                                      buffer=nl.sbuf)
              for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[b, (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                         BLOCK_M * m + i_lhsT.x])

              # Do matmul with all tiles in the blocks
              i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
              i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
              i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  res_gate_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                  res_up_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                  for bk in nl.affine_range(TILES_IN_BLOCK_K):
                    res_gate_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        gate_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])
                    res_up_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        up_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

                  # Accumulate on corresponding SBUF tile
                  result_gate_tiles[m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_gate_tile[i_res_mm.p, i_res_mm.x]
                  result_up_tiles[m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_up_tile[i_res_mm.p, i_res_mm.x]

          # SiLU and save to front_result (sbuf)
          for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
              i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
              
              # coalesce result tiles for better DMA performance
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                swish_val = nl.multiply(nl.sigmoid(result_gate_tiles[m, bm, bn, i_res.p, i_res.x]),
                      result_gate_tiles[m, bm, bn, i_res.p, i_res.x])
                act_val = nl.multiply(swish_val, result_up_tiles[m, bm, bn, i_res.p, i_res.x])
                #front_result[b, (TILES_IN_BLOCK_M * m + bm), (TILES_IN_BLOCK_N * n + bn),
                #                  i_res.p, i_res.x] = (act_val) # (128, 512)
                SMALL_TILE = TILE_N//128 # For transpose, it can creates 128x128
                for tn in nl.affine_range(SMALL_TILE):
                  i_hidden_T = nl.mgrid[0:TILE_M, 0:128] # (128, 128)
                  hidden_outT = nl.transpose(act_val[i_hidden_T.p, tn*128 + i_hidden_T.x])
                  front_result[b, (TILES_IN_BLOCK_N * n + bn)*SMALL_TILE+tn, (TILES_IN_BLOCK_M * m + bm), i_hidden_T.p, i_hidden_T.x] = hidden_outT


              # for tn in nl.affine_range(BLOCK_N//128):
              #   i_hidden_T = nl.mgrid[0:TILE_M, 0:128]
              #   hidden_outT = nl.transpose(hidden_out[i_hidden_T.p, tn * 128 + i_hidden_T.x])
              #   nl.store(result[BLOCK_N * n + tn * 128 + i_hidden_T.x, 
              #       (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_hidden_T.p],
              #              value=hidden_outT)
 
      #####down_proj#######################################

      #K->N, N->L
      #(M,K) * (K,N)
      #(M,N) * (N,L)
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_N = min(nl.tile_size.pmax, N)  # 128
      TILE_L = min(nl.tile_size.gemm_moving_fmax, L)  # 512
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_L = TILE_L * TILES_IN_BLOCK_L

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert L % BLOCK_L == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_L = L // BLOCK_L
      
      # Blocking L dimension (the RHS free dimension)
      for b in nl.affine_range(B):
        for l in nl.affine_range(NUM_BLOCK_L):
          result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_L,
                                   nl.par_dim(TILE_N), TILE_L),
                                  dtype=lhsT.dtype,
                                  buffer=nl.sbuf)
          print("result tiles shape: ", result_tiles.shape)

          # Blocking K dimension (the contraction dimension)
          # Use `sequential_range` because we do not want the compiler to change this loop by, 
          # for example, vectorizing it
          for n in nl.sequential_range(NUM_BLOCK_N):
            # Loading tiles from rhs
            # setting the load tile to `TILE_N x BLOCK_SIZE_L` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_N, 0:BLOCK_L]
            rhs_tiles = nl.ndarray((TILES_IN_BLOCK_N, nl.par_dim(TILE_N), BLOCK_L),
                                   dtype=down_proj.dtype,
                                   buffer=nl.sbuf)

            for bn_r in nl.affine_range(TILES_IN_BLOCK_N):
              rhs_tiles[bn_r, i_rhs.p, i_rhs.x] = nl.load(
                  down_proj[(TILES_IN_BLOCK_N * n + bn_r) * TILE_N + i_rhs.p,
                      BLOCK_L * l + i_rhs.x])

            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
              # Loading tiles from lhsT
              #i_lhsT = nl.mgrid[0:TILE_N, 0:BLOCK_M]
              lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_N, nl.par_dim(TILE_N), BLOCK_M),
                                      dtype=front_result.dtype,
                                      buffer=nl.sbuf)
              # HERE!
              #for bn_l in nl.affine_range(TILES_IN_BLOCK_N):
              #  lhsT_tiles[bn_l, i_lhsT.p, i_lhsT.x] = nl.load(
              #      lhsT[b, (TILES_IN_BLOCK_N * n + bn_l) * TILE_N + i_lhsT.p,
              #           BLOCK_M * m + i_lhsT.x])
              i_lhsT = nl.mgrid[0:128, 0:TILE_M] # (128, 128)
              for bn_l in nl.affine_range(TILES_IN_BLOCK_N):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  lhsT_tiles[bn_l, i_lhsT.p, bm * TILE_M + i_lhsT.x] = front_result[b, (TILES_IN_BLOCK_N * n + bn_l), (TILES_IN_BLOCK_M * m) + bm, i_lhsT.p, i_lhsT.x]

              # Do matmul with all tiles in the blocks
              i_lhsT_mm = nl.mgrid[0:TILE_N, 0:TILE_M]
              i_rhs_mm = nl.mgrid[0:TILE_N, 0:TILE_L]
              i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_L]
              for bl in nl.affine_range(TILES_IN_BLOCK_L):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  res_tile = nl.zeros((TILE_M, TILE_L), dtype=nl.float32, buffer=nl.psum)

                  for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    res_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bn, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        rhs_tiles[bn, i_rhs_mm.p, bl * TILE_L + i_rhs_mm.x])

                  # Accumulate on corresponding SBUF tile
                  result_tiles[m, bm, bl, i_res_mm.p,
                               i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

          # Copying the result from SBUF to HBM
          for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              i_res = nl.mgrid[0:TILE_M, 0:TILE_L]
              i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_L]
              result_packed = nl.ndarray((TILE_M, BLOCK_L),
                                         dtype=result_tiles.dtype,
                                         buffer=nl.sbuf)

              # coalesce result tiles for better DMA performance
              for bl in nl.affine_range(TILES_IN_BLOCK_L):
                result_packed[i_res.p,
                              bl * TILE_L + i_res.x] = nl.copy(result_tiles[m, bm, bl,
                                                                            i_res.p,
                                                                            i_res.x])
              nl.store(result[b, (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                              BLOCK_L * l + i_res_packed.x],
                       value=result_packed[i_res_packed.p, i_res_packed.x])


      
               
      return result

    @nki.jit
    def nki_matmul_fully_optimized_fused_all_opt_B(
        lhsT,
        gate_proj,
        up_proj,
        down_proj,
        # Meta-parameters
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=4,
        TILES_IN_BLOCK_K=8,
        TILES_IN_BLOCK_L=4,
    ):

      B, K, M = lhsT.shape # (Batch, K, M)
      K_, N = gate_proj.shape # (K_, N_)
      N_, L = down_proj.shape # (4096, 2048)
      assert K == K_, "lhsT and rhs must have the same contraction dimension"
      assert N == N_, "front_result and down_proj must have the same contraction dimension"
      result = nl.ndarray((B, M, L), dtype=lhsT.dtype, buffer=nl.shared_hbm)
      
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_K = min(nl.tile_size.pmax, K)  # 128
      TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_K = TILE_K * TILES_IN_BLOCK_K

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert K % BLOCK_K == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_K = K // BLOCK_K
      
      # Tricky Transpose with (128x128)
      # This size is up to 20MB with 4 Batch size. 
      front_result = nl.ndarray((B, N//128, M//128, nl.par_dim(128), 128), dtype=lhsT.dtype, buffer=nl.sbuf)

      # Blocking N dimension (the RHS free dimension)
      for n in nl.affine_range(NUM_BLOCK_N):
        result_gate_tiles = nl.zeros((B, NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                 nl.par_dim(TILE_K), TILE_N),
                                 dtype=gate_proj.dtype,
                                 buffer=nl.sbuf)
        result_up_tiles = nl.zeros((B, NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                 nl.par_dim(TILE_K), TILE_N),
                                 dtype=up_proj.dtype,
                                 buffer=nl.sbuf)


        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by, 
        # for example, vectorizing it
        for k in nl.sequential_range(NUM_BLOCK_K):
          # Loading tiles from rhs
          # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
          i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
          gate_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                 dtype=gate_proj.dtype,
                                 buffer=nl.sbuf)
          up_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                 dtype=up_proj.dtype,
                                 buffer=nl.sbuf)


          for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
            gate_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                gate_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                    BLOCK_N * n + i_rhs.x])
            up_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                up_proj[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                    BLOCK_N * n + i_rhs.x])

          for b in nl.affine_range(B):
            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
              # Loading tiles from lhsT
              i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
              lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                      dtype=lhsT.dtype,
                                      buffer=nl.sbuf)
              for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[b, (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                         BLOCK_M * m + i_lhsT.x])

              # Do matmul with all tiles in the blocks
              i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
              i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
              i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  res_gate_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                  res_up_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                  for bk in nl.affine_range(TILES_IN_BLOCK_K):
                    res_gate_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        gate_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])
                    res_up_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        up_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

                  # Accumulate on corresponding SBUF tile
                  result_gate_tiles[b, m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_gate_tile[i_res_mm.p, i_res_mm.x]
                  result_up_tiles[b, m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_up_tile[i_res_mm.p, i_res_mm.x]
        
        # SiLU and save to front_result (sbuf)
        SMALL_TILE = TILE_N//128 # For transpose, it can creates 128x128
        for b in nl.affine_range(B):
          for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
              i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
              
              # coalesce result tiles for better DMA performance
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                swish_val = nl.multiply(nl.sigmoid(result_gate_tiles[b, m, bm, bn, i_res.p, i_res.x]),
                      result_gate_tiles[b, m, bm, bn, i_res.p, i_res.x])
                act_val = nl.multiply(swish_val, result_up_tiles[b, m, bm, bn, i_res.p, i_res.x])
                #front_result[b, (TILES_IN_BLOCK_M * m + bm), (TILES_IN_BLOCK_N * n + bn),
                #                  i_res.p, i_res.x] = (act_val) # (128, 512)
                for tn in nl.affine_range(SMALL_TILE):
                  i_hidden_T = nl.mgrid[0:TILE_M, 0:128] # (128, 128)
                  hidden_outT = nl.transpose(act_val[i_hidden_T.p, tn*128 + i_hidden_T.x])
                  front_result[b, (TILES_IN_BLOCK_N * n + bn)*SMALL_TILE+tn, (TILES_IN_BLOCK_M * m + bm), i_hidden_T.p, i_hidden_T.x] = hidden_outT


      #####down_proj#######################################

      #K->N, N->L
      #(M,K) * (K,N)
      #(M,N) * (N,L)
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_N = min(nl.tile_size.pmax, N)  # 128
      TILE_L = min(nl.tile_size.gemm_moving_fmax, L)  # 512
      
      # Here, N is as previous K
      TILES_IN_BLOCK_N=8
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_L = TILE_L * TILES_IN_BLOCK_L

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert L % BLOCK_L == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_L = L // BLOCK_L
      
      
      # Blocking L dimension (the RHS free dimension)
      for b in nl.affine_range(B):
        for l in nl.affine_range(NUM_BLOCK_L):
          result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_L,
                                   nl.par_dim(TILE_N), TILE_L),
                                  dtype=lhsT.dtype,
                                  buffer=nl.sbuf)

          # Blocking K dimension (the contraction dimension)
          # Use `sequential_range` because we do not want the compiler to change this loop by, 
          # for example, vectorizing it
          for n in nl.sequential_range(NUM_BLOCK_N):
            # Loading tiles from rhs
            # setting the load tile to `TILE_N x BLOCK_SIZE_L` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_N, 0:BLOCK_L]
            rhs_tiles = nl.ndarray((TILES_IN_BLOCK_N, nl.par_dim(TILE_N), BLOCK_L),
                                   dtype=down_proj.dtype,
                                   buffer=nl.sbuf)

            for bn_r in nl.affine_range(TILES_IN_BLOCK_N):
              rhs_tiles[bn_r, i_rhs.p, i_rhs.x] = nl.load(
                  down_proj[(TILES_IN_BLOCK_N * n + bn_r) * TILE_N + i_rhs.p,
                      BLOCK_L * l + i_rhs.x])

            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
              # Loading tiles from lhsT
              #i_lhsT = nl.mgrid[0:TILE_N, 0:BLOCK_M]
              lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_N, nl.par_dim(TILE_N), BLOCK_M),
                                      dtype=front_result.dtype,
                                      buffer=nl.sbuf)
              # HERE!
              #for bn_l in nl.affine_range(TILES_IN_BLOCK_N):
              #  lhsT_tiles[bn_l, i_lhsT.p, i_lhsT.x] = nl.load(
              #      lhsT[b, (TILES_IN_BLOCK_N * n + bn_l) * TILE_N + i_lhsT.p,
              #           BLOCK_M * m + i_lhsT.x])
              i_lhsT = nl.mgrid[0:128, 0:TILE_M] # (128, 128)
              for bn_l in nl.affine_range(TILES_IN_BLOCK_N):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  lhsT_tiles[bn_l, i_lhsT.p, bm * TILE_M + i_lhsT.x] = front_result[b, (TILES_IN_BLOCK_N * n + bn_l), (TILES_IN_BLOCK_M * m) + bm, i_lhsT.p, i_lhsT.x]

              # Do matmul with all tiles in the blocks
              i_lhsT_mm = nl.mgrid[0:TILE_N, 0:TILE_M]
              i_rhs_mm = nl.mgrid[0:TILE_N, 0:TILE_L]
              i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_L]
              for bl in nl.affine_range(TILES_IN_BLOCK_L):
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                  res_tile = nl.zeros((TILE_M, TILE_L), dtype=nl.float32, buffer=nl.psum)

                  for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    res_tile[...] += nisa.nc_matmul(
                        lhsT_tiles[bn, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                        rhs_tiles[bn, i_rhs_mm.p, bl * TILE_L + i_rhs_mm.x])

                  # Accumulate on corresponding SBUF tile
                  result_tiles[m, bm, bl, i_res_mm.p,
                               i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

          # Copying the result from SBUF to HBM
          for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              i_res = nl.mgrid[0:TILE_M, 0:TILE_L]
              i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_L]
              result_packed = nl.ndarray((TILE_M, BLOCK_L),
                                         dtype=result_tiles.dtype,
                                         buffer=nl.sbuf)

              # coalesce result tiles for better DMA performance
              for bl in nl.affine_range(TILES_IN_BLOCK_L):
                result_packed[i_res.p,
                              bl * TILE_L + i_res.x] = nl.copy(result_tiles[m, bm, bl,
                                                                            i_res.p,
                                                                            i_res.x])
              nl.store(result[b, (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                              BLOCK_L * l + i_res_packed.x],
                       value=result_packed[i_res_packed.p, i_res_packed.x])

      
      return result



    @nki.jit
    def nki_matmul_fully_optimized_B(
        lhsT,
        rhs,
        # Meta-parameters
        TILES_IN_BLOCK_M=1,
        TILES_IN_BLOCK_N=4,
        TILES_IN_BLOCK_K=16,
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

      B, K, M = lhsT.shape
      K_, N = rhs.shape
      assert K == K_, "lhsT and rhs must have the same contraction dimension"
      result = nl.ndarray((B, M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

      # TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
      # TILE_K = nl.tile_size.pmax  # 128
      # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
      TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)  # 128
      TILE_K = min(nl.tile_size.pmax, K)  # 128
      TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512
 
      BLOCK_M = TILE_M * TILES_IN_BLOCK_M
      BLOCK_N = TILE_N * TILES_IN_BLOCK_N
      BLOCK_K = TILE_K * TILES_IN_BLOCK_K

      # the size has to be multiple of block size
      assert M % BLOCK_M == 0
      assert N % BLOCK_N == 0
      assert K % BLOCK_K == 0

      NUM_BLOCK_M = M // BLOCK_M
      NUM_BLOCK_N = N // BLOCK_N
      NUM_BLOCK_K = K // BLOCK_K

      # Blocking N dimension (the RHS free dimension)
      for b in nl.affine_range(B):
        for n in nl.affine_range(NUM_BLOCK_N):
          result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                   nl.par_dim(TILE_K), TILE_N),
                                   # nl.par_dim(TILE_M), TILE_N),
                                  dtype=lhsT.dtype,
                                  buffer=nl.sbuf)

          # Blocking K dimension (the contraction dimension)
          # Use `sequential_range` because we do not want the compiler to change this loop by, 
          # for example, vectorizing it
          for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=rhs.dtype,
                                   buffer=nl.sbuf)

            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
              rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                  rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                      BLOCK_N * n + i_rhs.x])

            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
              # Loading tiles from lhsT
              i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
              lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                      dtype=lhsT.dtype,
                                      buffer=nl.sbuf)
              for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[b, (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                         BLOCK_M * m + i_lhsT.x])

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
                        rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

                  # Accumulate on corresponding SBUF tile
                  result_tiles[m, bm, bn, i_res_mm.p,
                               i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

          # Copying the result from SBUF to HBM
          for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
              # i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
              # i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
              # result_packed = nl.ndarray((TILE_K, BLOCK_N),
              #                            dtype=result_tiles.dtype,
              #                            buffer=nl.sbuf)
              i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
              i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
              result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                         dtype=result_tiles.dtype,
                                         buffer=nl.sbuf)

              # coalesce result tiles for better DMA performance
              for bn in nl.affine_range(TILES_IN_BLOCK_N):
                result_packed[i_res.p,
                              bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                            i_res.p,
                                                                            i_res.x])
              # nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p,
              nl.store(result[b, (TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                              BLOCK_N * n + i_res_packed.x],
                       value=result_packed[i_res_packed.p, i_res_packed.x])

      return result


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
            # DK: This time, only this.
            return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)

##########################################################################################

def nki_customMac(total_input: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    # torch_input.shape = (..., M)
    # weight_t.shape = (M, N)
    # out.shape = (..., N)
    ## (Batch_size, Seq-len, M) * (M, N) -> (Batch_size, seq-len, N)
    
    #print(total_input.shape) # (1, seq-len, 2048)
    #print(weight_t.shape) # (2048, 1024), (2048, 256)
    
    batch_size, seq_len, M = total_input.shape
    _, N = weight_t.shape
    
    if seq_len < 128:
        out = torch.einsum('...m,mn->...n', total_input, weight_t)
    #if seq_len == 1 and N == 256:  # 1 FLOPs
    #    out = torch.einsum('...m,mn->...n', total_input, weight_t)
    #elif seq_len == 128 and N == 256:  # 85 FLOPs
    #    out = torch.einsum('...m,mn->...n', total_input, weight_t)
    #elif seq_len == 1 and N == 1024:  # 1 FLOPs
    #    out = torch.einsum('...m,mn->...n', total_input, weight_t)
    #elif seq_len == 128 and N == 1024:  # 113.78 FLOPs
    #    out = torch.einsum('...m,mn->...n', total_input, weight_t)
    elif seq_len == 256 and N == 256:  # 128 FLOPs
        out = torch.einsum('...m,mn->...n', total_input, weight_t)
    elif seq_len == 512 and N == 256:  # 170 FLOPs
        out = torch.einsum('...m,mn->...n', total_input, weight_t)
    elif seq_len == 640 and N == 256:  # 182 FLOPs
        out = torch.einsum('...m,mn->...n', total_input, weight_t)
    #elif seq_len == 256 and N == 1024:  # 204.8 FLOPs
    #elif seq_len == 512 and N == 1024:  # 341 FLOPs
    #elif seq_len == 640 and N == 1024:  # 393 FLOPs
    else:
        #print(total_input.shape) # (1, seq-len, 2048)
        #print(weight_t.shape) # (2048, 1024), (2048, 256)
        out = torch.zeros((batch_size, seq_len, N), dtype=total_input.dtype, device=total_input.device)
        total_input_tr = total_input.transpose(-2,-1)
        tiles_in_block_m = seq_len//128
        tiles_in_block_n = 2 if N == 1024 else 1 # 1024 or 256
        out = dk_nki_matmul_fully_optimized_B_foratt(total_input_tr, weight_t, TILES_IN_BLOCK_M=tiles_in_block_m, TILES_IN_BLOCK_N=tiles_in_block_n)
    #print(out.shape) # (1, seq-len, 1024), (1, seq-len, 256)
    return out

class Nki_LinearWithAsyncCommunication(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        sequence_dimension: int = 0,
        save_for_backward: bool = True,
        process_group=None,
        reduce_dtype: torch.dtype = torch.float32,
    ):
        ctx.use_bias = bias is not None and weight.requires_grad
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel_enabled = sequence_parallel_enabled
        ctx.sequence_dimension = sequence_dimension
        ctx.compute_weight_gradient = weight.requires_grad
        if process_group is None:
            from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
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
            from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
            total_input = _gather_along_dim(
                input, ctx.sequence_dimension, process_group=ctx.process_group
            )
        else:
            total_input = input

        output = nki_customMac(total_input, weight.t())

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        if ctx.sequence_parallel_enabled:
            assert (
                ctx.sequence_dimension is not None
            ), "Found `sequence_parallel_enabled` set to True, but `sequence_dimension` was None, and this occured in an unexpected area"
        if ctx.compute_weight_gradient:
            input_, weight = ctx.saved_tensors
        else:
            (weight,) = ctx.saved_tensors
            input_ = None

        use_bias = ctx.use_bias
        process_group = ctx.process_group
        handle = None

        if ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
                total_input = _gather_along_dim(
                    input_, ctx.sequence_dimension, process_group=process_group
                )
            else:
                total_input = input_

        grad_input = grad_output.matmul(weight)

        if handle is not None:
            handle.wait()

        original_dtype = grad_input.dtype

        if ctx.async_grad_allreduce:
            grad_input = grad_input.to(ctx.reduce_dtype)
            handle = torch.distributed.all_reduce(grad_input, group=process_group, async_op=True)
            grad_input = grad_input.to(original_dtype)

        if not ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                assert not ctx.async_grad_allreduce
                from neuronx_distributed.parallel_layers.mappings import _reduce_scatter_along_dim
                sub_grad_input = _reduce_scatter_along_dim(
                    grad_input.to(ctx.reduce_dtype),
                    ctx.sequence_dimension,
                    process_group=process_group,
                )
                sub_grad_input = sub_grad_input.to(original_dtype)
                return sub_grad_input, None, None, None, None, None, None, None, None
            if ctx.async_grad_allreduce:
                assert handle
                handle.wait()
                grad_input = grad_input.to(original_dtype)
            return grad_input, None, None, None, None, None, None, None, None

        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        #grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        #total_input_2d = total_input.view(-1, total_input.shape[-1])

        if ctx.sequence_parallel_enabled:
            from neuronx_distributed.parallel_layers.mappings import _reduce_scatter_along_dim
            sub_grad_input = _reduce_scatter_along_dim(
                grad_input.to(ctx.reduce_dtype),
                ctx.sequence_dimension,
                process_group=process_group,
            )
            sub_grad_input = sub_grad_input.to(original_dtype)

        grad_weight = grad_output_2d.t().matmul(total_input_2d)
        grad_bias = grad_output_2d.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel_enabled:
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None

        if ctx.async_grad_allreduce:
            if handle:
                handle.wait()
            grad_input = grad_input.to(original_dtype)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

def nki_linear_with_async_allreduce(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    sequence_dimension: int = 0,
    autograd_func_class=None,
    save_for_backward: bool = True,
    process_group=None,
    reduce_dtype: torch.dtype = torch.float32,
):
    # DK: NKI
    if autograd_func_class is None:
        autograd_func_class = Nki_LinearWithAsyncCommunication

    return autograd_func_class.apply(
        input_,
        weight,
        bias,
        async_grad_allreduce,
        sequence_parallel_enabled,
        sequence_dimension,
        save_for_backward,
        process_group,
        reduce_dtype,
    )

class Nki_ColumnParallelLinear(ColumnParallelLinear):
    def forward(self, input: torch.Tensor, *_: any):
        if self.pad and self.training:
            raise RuntimeError("pad=True is only for inference mode. Call model.eval()")

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            from neuronx_distributed.parallel_layers.mappings import copy_to_tensor_model_parallel_region
            input_parallel = copy_to_tensor_model_parallel_region(
                input, process_group=self.tensor_parallel_group
            )
        
        # DK: Custom
        output_parallel = nki_linear_with_async_allreduce(
            input_=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            #autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
            reduce_dtype=self.reduce_dtype,
        )

        if self.gather_output:
            from neuronx_distributed.parallel_layers.mappings import gather_from_tensor_model_parallel_region
            output = gather_from_tensor_model_parallel_region(
                output_parallel, process_group=self.tensor_parallel_group
            )
            if self.pad and self.pad_size > 0:
                output = torch.narrow(output, -1, 0, self.output_size - self.pad_size)
        else:
            output = output_parallel

        if self.skip_bias_add:
            return output, self.bias
        if self.bias is not None:
            output = output + self.bias
        return output

class DK_GroupQueryAttentionQKV(GroupQueryAttention_QKV):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy=None,
        gather_output: bool = True,
        fused_qkv: bool = False,
        clip_qkv=None,
        sequence_parallel_enabled=False,
        sequence_dimension=None,
        tensor_model_parallel_group=None,
        rms_norm_eps=None,
        qkv_kernel_enabled=False,
        logical_neuron_cores=1,
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
            gather_output=gather_output,
            fused_qkv=fused_qkv,
            clip_qkv=clip_qkv,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            tensor_model_parallel_group=tensor_model_parallel_group,
            rms_norm_eps=rms_norm_eps,
            qkv_kernel_enabled=qkv_kernel_enabled,
            logical_neuron_cores=logical_neuron_cores,
        )
        if not fused_qkv and (tensor_model_parallel_group is not None):
            self.q_proj = Nki_ColumnParallelLinear(
                self.hidden_size,
                self.num_attention_heads * self.head_dim,
                bias=self.bias,
                gather_output=self.gather_output,
                dtype=dtype,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )
            self.k_proj = Nki_ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=self.bias,
                gather_output=self.gather_output,
                dtype=dtype,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )
            self.v_proj = Nki_ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=self.bias,
                gather_output=self.gather_output,
                dtype=dtype,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )

@register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(tensor_model_parallel_group=tensor_model_parallel_group)

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


        
        self.nki_enabled = getattr(config.neuron_config, "nki_enabled", False)
        self.init_gqa_properties()
        self.init_rope()

    def init_rope(self):
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

    def init_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )
        
        if self.nki_enabled:
            # HERE, Whether using NKI at GQ_Attention, or not.
            # NKI
            #self.qkv_proj = DK_GroupQueryAttentionQKV(
            # Basic
            self.qkv_proj = GroupQueryAttention_QKV(
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                tp_degree=self.tp_degree,
                dtype=self.torch_dtype,
                bias=self.bias,
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
        else:
            # Origin
            self.qkv_proj = GroupQueryAttention_QKV(
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                tp_degree=self.tp_degree,
                dtype=self.torch_dtype,
                bias=self.bias,
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

        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
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

    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK
    
    def custom_scaled_qk_softmax(self, Q, K, attention_mask):
        batch_size, num_heads, seq_len, dim = Q.shape
        if self.nki_enabled and seq_len == 640:
            # out_tensor : (batch_size:1, num_heads/tp-degree:16, seq-len, seq-len)
            q_head_lhsT = Q.transpose(2,3)
            k_head_rhs = K.transpose(2,3)
            active_scores = dk_nki_matmul_fully_optimized_BB_forscaled_divsqrt(q_head_lhsT, k_head_rhs, math.sqrt(self.head_dim), attention_mask, torch.finfo(Q.dtype).min)
           # active_scores = dk_nki_softmax_lastdim(out_tensor)
        else:
            QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
            QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
            active_scores = dk_nki_softmax_lastdim(QK)
        return active_scores


    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len)
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

                _flash_fwd_call[grid](
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                _flash_fwd_call(
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            else:
                raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

            # shape: BHDS
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            active_scores = self.scaled_qk(Q, K_active, attention_mask)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
                Q.dtype
            )
            #active_scores = self.custom_scaled_qk_softmax(Q, K_active, attention_mask)
            
            # This matmul FLOPs is too low. (1x1)
            attn_output = torch.matmul(active_scores, V_active)

        
        return attn_output, flash_attn_strategy
    def compute_for_flash_decoding(self, Q, K, past_key_value, attention_mask, active_mask):
        # Not this time
        attn_output = super().compute_for_flash_decoding(Q, K, past_key_value, attention_mask, active_mask)
        return attn_output

    def compute_for_token_gen(self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask):
        #attn_output = super().compute_for_token_gen(Q, K, V, position_ids, past_key_value, attention_mask, active_mask)
        """attention computation at token generation phase"""
        is_speculation = position_ids.shape[-1] > 1

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        ### Slow with basic matmul
        #prior_scores = nki_matmul_for_1_16_1_64__1_16_64_64_(Q.transpose(2, 3), K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
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
        # Very slow to small data
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
 

        attn_output = attn_prior + attn_active
        return attn_output


# TODO: Modularize RotaryEmbedding. See how HF transformers does it in 4.43.
# DK: TODO: There is rotary embedding samples at nki-sample.
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
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
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

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
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

        # DK: TODO: It would be possible to fuse all stages and layer in a kernel. But time-consuming

        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

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

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)
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

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
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
            self.fc = ColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.medusa_speculation_length = config.neuron_config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
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

