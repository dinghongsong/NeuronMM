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
"""PyTorch LLaMA model for NXD inference."""
import copy
import gc
import logging
import math
from neuronx_distributed.parallel_layers import utils 
from neuronx_distributed.parallel_layers.layers import LinearWithAsyncCommunication, BaseParallelLinear, _initialize_parameter_cpu
from neuronx_distributed.parallel_layers.utils import (
    EmbeddingUtility,
    cast_if_autocast_enabled,
    divide,
    get_padding_length,
    is_torch_version_greater_than_2,
    set_tensor_model_parallel_attributes,
    verify_casted_dtype,
)
from typing import (
    Optional, Tuple, Union, Any, Callable, Dict, Type, cast
    
)

from torch.nn.parameter import Parameter

import copy
from torch.distributed import ProcessGroup
import gc
import logging
import math
import numpy as np
from typing import List, Optional, Tuple, Type


import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki.language import par_dim


import torch
from typing import List, Optional, Tuple, Type
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from neuronxcc.nki.language import par_dim


import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    get_aot_mode
)
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_first_dim,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    # reduce_scatter_to_sequence_parallel_region_tiled,
    _gather_along_dim,
    _reduce_scatter_along_dim,
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_input_channels_to_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.utils import cpu_mode
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel,
    quant_mlp_fused_add_isa_kernel,
    quant_mlp_isa_kernel,
)
from neuronxcc.nki._private_kernels.rmsnorm import rmsnorm_quant_isa_kernel
# from neuronxcc.nki.compiler.backends.neuron.dimensions import CCPipeline  # noqa: N813
from neuronxcc.nki.language import nc
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    transpose_parallel_linear_layer,
)


from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
# from neuronx_distributed_inference.modules.eagle.utils import tiled_all_gather_matmul
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group


from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_first_dim,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    # reduce_scatter_to_sequence_parallel_region_tiled,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.utils import cpu_mode
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel,
    quant_mlp_fused_add_isa_kernel,
    quant_mlp_isa_kernel,
)
from neuronxcc.nki._private_kernels.rmsnorm import rmsnorm_quant_isa_kernel
from neuronxcc.nki.compiler.backends.neuron.dimensions import CCPipeline  # noqa: N813
from neuronxcc.nki.language import nc
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from models.config import InferenceConfig, NeuronConfig  # noqa: E402
from models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from modules.attention.attention_base import NeuronAttentionBase
from modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
)
from modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    transpose_parallel_linear_layer,
)
from modules.custom_calls import CustomRMSNorm
from modules.eagle.utils import tiled_all_gather_matmul
from modules.flashdecode.utils import calculate_num_cores_per_group
from modules.lora_serving.lora_module import is_lora_module
from utils.distributed import get_tp_group

from models.nki_kernels import XUV_matmul

_LLAMA_MODULE_MAP = {}

logger = logging.getLogger("Neuron")



def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


# Get the modules_to_not_convert from the neuron configs
def get_modules_to_not_convert(neuron_config: NeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def get_updated_configs(config: InferenceConfig):
    """
    Generate a list of configurations for each hidden layer in a Llama model.

    This function creates a list of InferenceConfig objects, one for each layer. It
    modifies the configurations for certain layers based on which modules should not
    be converted to quantized format. The function uses get_modules_to_not_convert()
    to determine which modules should not be converted.

    Args:
    config (InferenceConfig): The inference configuration for the model.

    Returns:
    list[InferenceConfig]: A list of InferenceConfig objects, one for each layer in the model.
                           Each config may be either the original config or a modified version
                           with "quantized_mlp_kernel_enabled" as False for that specific layer.
    """
    updated_configs = []
    modules_to_not_convert = get_modules_to_not_convert(config.neuron_config)
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for i in range(config.num_hidden_layers):
        # If any of the MLP modules for this layer are in modules_to_not_convert
        module_pattern = f"layers.{i}.mlp"
        if any(module_pattern in module for module in modules_to_not_convert):
            non_quant_config = copy.deepcopy(config)
            non_quant_config.neuron_config.quantized_mlp_kernel_enabled = False
            non_quant_config.neuron_config.activation_quantization_type = None
            non_quant_config.neuron_config.quantize_clamp_bound = float("inf")
            updated_configs.append(non_quant_config)
        else:
            updated_configs.append(config)
    return updated_configs


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


def _helper_concat_and_delete_qkv(llama_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        llama_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    llama_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(llama_state_dict, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(llama_state_dict, l, "scale")

    gc.collect()

    return llama_state_dict


class WeightGatheredColumnParallel(ColumnParallelLinear):
    """
    A specialized column-parallel linear layer that implements weight gathering optimization
    for efficient processing of long sequences in transformer models during eagle speculation.

    This layer provides two forward paths:
    1. Standard column-parallel forward (inherited from parent)
    2. Weight-gathered forward for long sequences
    """

    def forward_wg(self, input: torch, weight_gather: bool = False, hidden_size_threshold_for_cc_tiling: int = 16384):
        """
        Performs the forward pass with optional weight gathering optimization.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len/TP, 2*hidden_size)
            weight_gather (bool): Whether to use weight gathering optimization.
                                Typically True for sequences >= 32K

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If skip_bias_add is False: Output tensor of shape (batch_size, seq_len, hidden_size)
                - If skip_bias_add is True: Tuple of (output tensor, bias)
        """
        if weight_gather:
            use_collective_einsum = (input.shape[-1] >= hidden_size_threshold_for_cc_tiling)
            if use_collective_einsum:
                output = tiled_all_gather_matmul(self.weight, input, tp_degree=self.tensor_parallel_group.size(), tile_size=hidden_size_threshold_for_cc_tiling // 2)
            else:
                weight = _gather_along_first_dim(self.weight, process_group=self.tensor_parallel_group)
                output = self._forward_impl(
                    input=input,
                    weight=weight,
                    bias=None,
                    async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                    sequence_parallel_enabled=self.sequence_parallel_enabled,
                    sequence_dimension=self.sequence_dimension,
                    autograd_func_class=self.autograd_func_class,
                    process_group=self.tensor_parallel_group
                )

            if self.skip_bias_add:
                return output, self.bias

            output = (output + self.bias) if self.bias is not None else output
            return output
        else:
            return self.forward(input)


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
        return NeuronConfig

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



# def get_fused_mlp_up_T_params(S):
    
#     if S == 1:
#         # return {
#         #     'M_tiles_in_block': 16, 
#         #     'r_tiles_in_block': 4, 
#         #     'K_tiles_in_block': 1, 
#         #     'N_tiles_in_block': 16
#         # }
#         return {
#             'M_tiles_in_block': 8, 
#             'r_tiles_in_block': 2, 
#             'K_tiles_in_block': 1, 
#             'N_tiles_in_block': 4
#         }
#         # return {
#         #     'M_tiles_in_block': 1, 
#         #     'r_tiles_in_block': 1, 
#         #     'K_tiles_in_block': 1, 
#         #     'N_tiles_in_block': 8
#         # }
#     # elif 1< S < 128:
#     #     return {
#     #         'M_tiles_in_block': 1, 
#     #         'r_tiles_in_block': 1, 
#     #         'K_tiles_in_block': 1, 
#     #         'N_tiles_in_block': 8
#     #     }
#     elif 128 == S: 
#         # return {
#         #     'M_tiles_in_block': 8, 
#         #     'r_tiles_in_block': 4, 
#         #     'K_tiles_in_block': 1, 
#         #     'N_tiles_in_block': 4
#         # }
#         # return {
#         #     'M_tiles_in_block': 16, 
#         #     'r_tiles_in_block': 4, 
#         #     'K_tiles_in_block': 1, 
#         #     'N_tiles_in_block': 16
#         # }
#         return {
#             'M_tiles_in_block': 8, 
#             'r_tiles_in_block': 2, 
#             'K_tiles_in_block': 1, 
#             'N_tiles_in_block': 4
#         }
#     # elif S == 512:
#     #     return {
#     #         'M_tiles_in_block': 8, 
#     #         'r_tiles_in_block': 4, 
#     #         'K_tiles_in_block': 1, 
#     #         'N_tiles_in_block': 16
#     #     }
#     # elif 512 < S < 1024:
#     #     return {
#     #         'M_tiles_in_block': 8, 
#     #         'r_tiles_in_block': 4, 
#     #         'K_tiles_in_block': 2, 
#     #         'N_tiles_in_block': 1
#     #     }
#     # elif S == 1024:
#     #     return {
#     #         'M_tiles_in_block': 8, 
#     #         'r_tiles_in_block': 4, 
#     #         'K_tiles_in_block': 1, 
#     #         'N_tiles_in_block': 4
#     #     }
#     # else:  # S > 1024
#     #     return {
#     #         'M_tiles_in_block': 8, 
#     #         'r_tiles_in_block': 4, 
#     #         'K_tiles_in_block': 2, 
#     #         'N_tiles_in_block': 1
#     #     }


# def get_fused_three_mm_XTUV_params(S):

#     if S == 1:
#         return {
#             'M_tiles_in_block': 16, 
#             'r_tiles_in_block': 4, 
#             'K_tiles_in_block': 1, 
#             'N_tiles_in_block': 4
#         }
#         # return {
#         #     'M_tiles_in_block': 8, 
#         #     'r_tiles_in_block': 8, 
#         #     'K_tiles_in_block': 4, 
#         #     'N_tiles_in_block': 4
#         # }
#     elif  S == 128:
#         return {
#             'M_tiles_in_block': 8, 
#             'r_tiles_in_block': 8, 
#             'K_tiles_in_block': 4, 
#             'N_tiles_in_block': 4
#         }
#     # elif 128 < S < 256:
#     #     # For range 128 < S < 256, use similar to S <= 128 but slightly optimized
#     #     return {
#     #         'M_tiles_in_block': 8, 
#     #         'r_tiles_in_block': 2, 
#     #         'K_tiles_in_block': 1, 
#     #         'N_tiles_in_block': 4
#     #     }
#     # elif S == 256:
#     #     return {
#     #         'M_tiles_in_block': 4, 
#     #         'r_tiles_in_block': 2, 
#     #         'K_tiles_in_block': 2, 
#     #         'N_tiles_in_block': 4
#     #     }
#     # elif 256 < S < 512:
#     #     # For range 256 < S < 512, use similar to S == 256
#     #     return {
#     #         'M_tiles_in_block': 4, 
#     #         'r_tiles_in_block': 2, 
#     #         'K_tiles_in_block': 2, 
#     #         'N_tiles_in_block': 4
#     #     }
#     # elif S == 512:
#     #     return {
#     #         'M_tiles_in_block': 4, 
#     #         'r_tiles_in_block': 4, 
#     #         'K_tiles_in_block': 4, 
#     #         'N_tiles_in_block': 4
#     #     }
#     # elif 512 < S < 1024:
#     #     return {
#     #         'M_tiles_in_block': 4, 
#     #         'r_tiles_in_block': 2, 
#     #         'K_tiles_in_block': 4, 
#     #         'N_tiles_in_block': 4
#     #     }
#     # elif S == 1024:
#     #     return {
#     #         'M_tiles_in_block': 4, 
#     #         'r_tiles_in_block': 4, 
#     #         'K_tiles_in_block': 8, 
#     #         'N_tiles_in_block': 1
#     #     }
#     # else:  # S > 1024
#     #     return {
#     #         'M_tiles_in_block': 4, 
#     #         'r_tiles_in_block': 2, 
#     #         'K_tiles_in_block': 8, 
#     #         'N_tiles_in_block': 4
#     #     }



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


# # Invoke MLP kernel as the following
# def nki_mm(x, up_v_proj, up_u_proj, 
#              gate_v_proj, gate_u_proj,
#              down_v_proj, down_u_proj):
    
    
#     # Call the optimized kernel with auto-selected parameters
#     result = svd_mlp_with_fused_kernel(
#         x, up_v_proj.T, up_u_proj.T, 
#         gate_v_proj.T, gate_u_proj.T,  
#         down_v_proj.T, down_u_proj.T, 
#     )
#     return result




##############################################


class SVD_LlamaMLP(nn.Module):
    def __init__(self, config, compress_ratio=0.8):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        ######### original
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        # low_rank = math.ceil(self.intermediate_size * self.hidden_size * compress_ratio / ((self.intermediate_size + self.hidden_size) * 128)) * 128
        low_rank = round(self.intermediate_size * self.hidden_size * compress_ratio / ((self.intermediate_size + self.hidden_size) * 128)) * 128

        self.gate_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.gate_u_proj = nn.Linear(low_rank, self.intermediate_size, bias=False)

        self.up_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.up_u_proj = nn.Linear(low_rank, self.intermediate_size, bias=False)

        self.down_v_proj = nn.Linear(self.intermediate_size, low_rank, bias=False)
        self.down_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        

    def forward(self, x):

        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
    

# ########### original MLP_SVD
# class NeuronLlamaMLP_SVD(nn.Module):
#     """
#     This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
#     """

#     def __init__(self, config: InferenceConfig):
#         super().__init__()
#         self.config = config
#         self.neuron_config = config.neuron_config
#         self.tp_degree = config.neuron_config.tp_degree
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.act_fn = ACT2FN[config.hidden_act]

#         self.sequence_parallel_enabled = getattr(
#             self.neuron_config, "sequence_parallel_enabled", False
#         )
#         self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
#         self.rms_norm_eps = config.rms_norm_eps
#         self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
#         self.fused_rmsnorm_skip_gamma = self.config.neuron_config.fused_rmsnorm_skip_gamma
#         self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
#         self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
#         self.quantize_clamp_bound = self.neuron_config.quantize_clamp_bound
#         self.logical_nc_config = self.neuron_config.logical_nc_config
#         self.activation_quantization_type = self.neuron_config.activation_quantization_type
#         mlp_bias = getattr(config, "mlp_bias", False)

#         ############################################ SVD-Flash
#         # self.low_rank = int(self.intermediate_size * self.hidden_size * self.config.metadata["compress_ratio"] / (self.intermediate_size + self.hidden_size))
#         # self.low_rank = math.ceil(self.intermediate_size * self.hidden_size * self.config.metadata["compress_ratio"] / ((self.intermediate_size + self.hidden_size) * 128)) * 128
#         self.low_rank = round(self.intermediate_size * self.hidden_size * self.config.metadata["compress_ratio"] / ((self.intermediate_size + self.hidden_size) * 128)) * 128

#         ############################################
#         if self.neuron_config.quantized_mlp_kernel_enabled and self.quantize_clamp_bound == float(
#             "inf"
#         ):
#             logging.warning(
#                 "quantize_clamp_bound is not specified in NeuronConfig. We will use the default value of 1200 for llama models in quantized kernels."
#             )
#             self.quantize_clamp_bound = 1200.0
#         if parallel_state.model_parallel_is_initialized():
#             if self.neuron_config.quantized_mlp_kernel_enabled:
#                 # # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
#                 tp_degree = self.neuron_config.tp_degree
#                 self.intermediate_size += (
#                     get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
#                 )
#                 logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")
            
#             ############################################ original
#             # self.gate_proj = ColumnParallelLinear(
#             #     self.hidden_size,
#             #     self.intermediate_size,
#             #     bias=mlp_bias,
#             #     gather_output=False,
#             #     dtype=config.neuron_config.torch_dtype,
#             #     pad=True,
#             #     sequence_parallel_enabled=False,
#             #     sequence_dimension=None,
#             #     tensor_model_parallel_group=get_tp_group(config),
#             # )
#             # self.up_proj = ColumnParallelLinear(
#             #     self.hidden_size,
#             #     self.intermediate_size,
#             #     bias=mlp_bias,
#             #     gather_output=False,
#             #     dtype=config.neuron_config.torch_dtype,
#             #     pad=True,
#             #     sequence_parallel_enabled=False,
#             #     sequence_dimension=None,
#             #     tensor_model_parallel_group=get_tp_group(config),
#             # )
#             # self.down_proj = RowParallelLinear(
#             #     self.intermediate_size,
#             #     self.hidden_size,
#             #     bias=mlp_bias,
#             #     input_is_parallel=True,
#             #     dtype=config.neuron_config.torch_dtype,
#             #     pad=True,
#             #     sequence_parallel_enabled=self.sequence_parallel_enabled,
#             #     sequence_dimension=self.sequence_dimension,
#             #     tensor_model_parallel_group=get_tp_group(config),
#             #     reduce_dtype=config.neuron_config.rpl_reduce_dtype,
#             # )
#             ############################################

#             ############################################ SVD-Flash
#             self.gate_v_proj = ColumnParallelLinear(
#                 self.hidden_size,
#                 self.low_rank,
#                 bias=mlp_bias,
#                 gather_output=False,
#                 dtype=config.neuron_config.torch_dtype,
#                 pad=True,
#                 sequence_parallel_enabled=False,
#                 sequence_dimension=None,
#                 tensor_model_parallel_group=get_tp_group(config),
#             )
            
#             self.gate_u_proj = ColumnParallelLinear(
#                 self.low_rank,
#                 self.intermediate_size,
#                 bias=mlp_bias,
#                 gather_output=False,
#                 dtype=config.neuron_config.torch_dtype,
#                 pad=True,
#                 sequence_parallel_enabled=False,
#                 sequence_dimension=None,
#                 tensor_model_parallel_group=get_tp_group(config),
#             )
            
#             self.up_v_proj = ColumnParallelLinear(
#                 self.hidden_size,
#                 self.low_rank,
#                 bias=mlp_bias,
#                 gather_output=False,
#                 dtype=config.neuron_config.torch_dtype,
#                 pad=True,
#                 sequence_parallel_enabled=False,
#                 sequence_dimension=None,
#                 tensor_model_parallel_group=get_tp_group(config),
#             )

#             self.up_u_proj = ColumnParallelLinear(
#                 self.low_rank,
#                 self.intermediate_size,
#                 bias=mlp_bias,
#                 gather_output=False,
#                 dtype=config.neuron_config.torch_dtype,
#                 pad=True,
#                 sequence_parallel_enabled=False,
#                 sequence_dimension=None,
#                 tensor_model_parallel_group=get_tp_group(config),
#             )

#             self.down_v_proj = RowParallelLinear(
#                 self.intermediate_size,
#                 self.low_rank,
#                 bias=mlp_bias,
#                 input_is_parallel=True,
#                 dtype=config.neuron_config.torch_dtype,
#                 pad=True,
#                 sequence_parallel_enabled=self.sequence_parallel_enabled,
#                 sequence_dimension=self.sequence_dimension,
#                 tensor_model_parallel_group=get_tp_group(config),
#                 reduce_dtype=config.neuron_config.rpl_reduce_dtype,
#             )

#             self.down_u_proj = RowParallelLinear(
#                 self.low_rank,
#                 self.hidden_size,
#                 bias=mlp_bias,
#                 input_is_parallel=True,
#                 dtype=config.neuron_config.torch_dtype,
#                 pad=True,
#                 sequence_parallel_enabled=self.sequence_parallel_enabled,
#                 sequence_dimension=self.sequence_dimension,
#                 tensor_model_parallel_group=get_tp_group(config),
#                 reduce_dtype=config.neuron_config.rpl_reduce_dtype,
#             )
#             ############################################
            
#             if self.mlp_kernel_enabled:
#                 if self.neuron_config.quantized_mlp_kernel_enabled:
#                     setattr(
#                         self.gate_proj,
#                         "post_create_quantized_module_hook",
#                         preprocess_quantized_linear_layer,
#                     )
#                     setattr(
#                         self.up_proj,
#                         "post_create_quantized_module_hook",
#                         preprocess_quantized_linear_layer,
#                     )
#                     setattr(
#                         self.down_proj,
#                         "post_create_quantized_module_hook",
#                         preprocess_quantized_linear_layer,
#                     )
#                 else:
#                     # Transpose the weights to the layout expected by kernels
#                     self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
#                     self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
#                     self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

#         else:
#             self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
#             self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
#             self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

#     def _kernel_enabled_quantized_mlp(self, x, rmsnorm, residual, adapter_ids):
#         full_seqlen = x.shape[1] * (self.config.neuron_config.tp_degree if self.sequence_parallel_enabled else 1)
#         if full_seqlen <= self.neuron_config.seq_len_threshold_for_cc_tiling:  # Keep regular grid for TKG.
#             grid = (nc(self.logical_nc_config),)
#         else:  # Add CC pipelining dim for CTE kernel grid
#             grid = (CCPipeline(self.neuron_config.cc_pipeline_tiling_factor) * nc(self.logical_nc_config),)
#         fused_residual = residual is not None
#         fused_rmsnorm = rmsnorm is not None
#         logger.debug(
#             f"MLP: quantized kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_nc_config={self.logical_nc_config}"
#         )

#         # Can't do residual add in the kernel if SP is enabled
#         if fused_residual:
#             assert (
#                 not self.sequence_parallel_enabled
#             ), "Quantized MLP cannot have both fused residual add and sequence parallel RMSnorm!"
#             # Using fused residual add
#             _mlp_fwd_call = nki_jit()(quant_mlp_fused_add_isa_kernel)
#         else:
#             _mlp_fwd_call = nki_jit()(quant_mlp_isa_kernel)

#         if fused_rmsnorm:
#             ln_w = rmsnorm.weight.unsqueeze(0)
#         else:
#             ln_w = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device)

#         # Handle SP RMSnorm
#         x_orig_dtype = x.dtype
#         if self.sequence_parallel_enabled:
#             # This RMSNormQuant kernel will do quantization inside, so we pass the
#             # clamp_bound for clipping.
#             # If we don't use this kernel, the MLP kernel below will do the
#             # quantization, so we also pass clamp_bound to that kernel.
#             if self.rmsnorm_quantize_kernel_enabled:
#                 logger.debug(
#                     "Running Quantized MLP kernel with sequence-parallel RMSnorm-Quantize kernel!"
#                 )
#                 _rmsnorm_quant_fwd_call = nki_jit()(rmsnorm_quant_isa_kernel)
#                 quant_rmsnorm_out = torch.zeros(
#                     size=(
#                         x.shape[0],  # batch size
#                         x.shape[1],  # sequence length
#                         x.shape[2] + 4,  # hidden size + 4 bytes for packing fp32 scale
#                     ),
#                     dtype=torch.int8,
#                     device=x.device,
#                 )
#                 clamp_bound = self.quantize_clamp_bound
#                 _rmsnorm_quant_fwd_call[grid](
#                     x, ln_w, clamp_bound, quant_rmsnorm_out, kernel_name="QuantOnly"
#                 )
#                 x = gather_from_sequence_parallel_region(
#                     quant_rmsnorm_out,
#                     self.sequence_dimension,
#                     process_group=get_tp_group(self.config),
#                     tile_cc=self.neuron_config.tile_cc,
#                 )

#             else:
#                 logger.debug(
#                     "Running Quantized MLP kernel with external (native compiler) sequence-parallel RMSnorm!"
#                 )
#                 x = gather_from_sequence_parallel_region(
#                     x, self.sequence_dimension, process_group=get_tp_group(self.config), tile_cc=self.neuron_config.tile_cc
#                 )

#         # Build output tensor
#         output_tensor_seqlen = x.shape[1]
#         output_tensor = torch.zeros(
#             size=(
#                 x.shape[0],  # batch size
#                 output_tensor_seqlen,
#                 self.hidden_size,  # hidden size
#             ),
#             dtype=x_orig_dtype,
#             device=x.device,
#         )

#         # Grab weights
#         # all weights of the layers are stored in (out, in) shape
#         # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
#         gate_w = self.gate_proj.weight.data
#         gate_w_scale = self.gate_proj.scale
#         up_w = self.up_proj.weight.data
#         up_w_scale = self.up_proj.scale
#         down_w = self.down_proj.weight.data
#         down_w_scale = self.down_proj.scale
#         clamp_bound = self.quantize_clamp_bound

#         if fused_residual:
#             residual_output_tensor = torch.zeros(
#                 size=(
#                     x.shape[0],  # batch size
#                     output_tensor_seqlen,
#                     self.hidden_size,  # hidden size
#                 ),
#                 dtype=x.dtype,
#                 device=x.device,
#             )

#             _mlp_fwd_call[grid](
#                 x,  # attn_output
#                 residual,  # hidden
#                 ln_w,  # ln_w
#                 gate_w,  # gate_w
#                 gate_w_scale,
#                 up_w,  # up_w
#                 up_w_scale,
#                 down_w,  # down_w
#                 down_w_scale,
#                 clamp_bound,
#                 output_tensor,  # out
#                 add_out=residual_output_tensor,
#                 fused_rmsnorm=fused_rmsnorm,
#                 eps=self.rms_norm_eps,
#                 kernel_name="MLP",
#                 store_add=True,
#             )
#             residual = residual_output_tensor
#         else:
#             _mlp_fwd_call[grid](
#                 x,  # hidden
#                 # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
#                 ln_w,
#                 gate_w,  # gate_w
#                 gate_w_scale,
#                 up_w,  # up_w
#                 up_w_scale,
#                 down_w,  # down_w
#                 down_w_scale,
#                 clamp_bound,
#                 output_tensor,  # out
#                 # Run RMSNorm inside the kernel if NOT using SP rmsnorm
#                 fused_rmsnorm=fused_rmsnorm,
#                 eps=self.rms_norm_eps,
#                 kernel_name="MLP",
#             )
#             residual = None

#         # All-reduce or reduce-scatter, depending on whether SP is enabled
#         if self.sequence_parallel_enabled:
#             if self.neuron_config.tile_cc:
#                 output_tensor = reduce_scatter_to_sequence_parallel_region_tiled(
#                     output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
#                 )
#             else:
#                 output_tensor = reduce_scatter_to_sequence_parallel_region(
#                     output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
#                 )
#         else:
#             output_tensor = reduce_from_tensor_model_parallel_region(output_tensor)

#         logger.debug(f"Quantized MLP output shape {output_tensor.shape}")
#         return (output_tensor, residual)

#     def _kernel_enabled_mlp(self, x, rmsnorm, residual, adapter_ids):
#         fused_residual = residual is not None
#         fused_rmsnorm = rmsnorm is not None
#         logger.debug(
#             f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, skip_gamma={self.fused_rmsnorm_skip_gamma}, logical_nc_config={self.logical_nc_config}"
#         )

#         # Choose which kernel to call
#         if fused_residual:
#             assert (
#                 not self.sequence_parallel_enabled
#             ), "MLP kernel cannot have both fused residual add and sequence parallel RMSnorm!"
#             # Using fused residual add
#             _mlp_fwd_call = nki_jit()(mlp_fused_add_isa_kernel)
#         else:
#             _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

#         if self.sequence_parallel_enabled:
#             x = gather_from_sequence_parallel_region(
#                 x, self.sequence_dimension, process_group=get_tp_group(self.config), tile_cc=self.neuron_config.tile_cc
#             )

#         # Build output tensor
#         output_tensor_seqlen = x.shape[1]
#         output_tensor = torch.zeros(
#             size=(
#                 x.shape[0],  # batch size
#                 output_tensor_seqlen,
#                 self.hidden_size,  # hidden size
#             ),
#             dtype=x.dtype,
#             device=x.device,
#         )

#         # Grab weights
#         # all weights of the layers are stored in (out, in) shape
#         # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
#         if fused_rmsnorm:
#             ln_w = rmsnorm.weight.unsqueeze(0)
#         else:
#             ln_w = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device)
#         gate_w = self.gate_proj.weight.data
#         up_w = self.up_proj.weight.data
#         down_w = self.down_proj.weight.data

#         if output_tensor_seqlen <= self.neuron_config.seq_len_threshold_for_cc_tiling:  # Keep regular grid for TKG. Messes up the MLP impl
#             grid = (nc(self.logical_nc_config),)
#         else:  # Add CC pipelining dim for CTE kernel grid
#             grid = (CCPipeline(self.neuron_config.cc_pipeline_tiling_factor) * nc(self.logical_nc_config),)

#         if fused_residual:
#             residual_output_tensor = torch.zeros(
#                 size=(
#                     x.shape[0],  # batch size
#                     output_tensor_seqlen,
#                     self.hidden_size,  # hidden size
#                 ),
#                 dtype=x.dtype,
#                 device=x.device,
#             )

#             _mlp_fwd_call[grid](
#                 x,  # attn_output
#                 residual,  # hidden
#                 ln_w,  # ln_w
#                 gate_w,  # gate_w
#                 up_w,  # up_w
#                 down_w,  # down_w
#                 output_tensor,  # out
#                 kernel_name="MLP",
#                 add_out=residual_output_tensor,
#                 fused_rmsnorm=fused_rmsnorm,
#                 skip_gamma=self.fused_rmsnorm_skip_gamma,
#                 eps=self.rms_norm_eps,
#                 store_add=True,
#             )
#             residual = residual_output_tensor
#         else:
#             _mlp_fwd_call[grid](
#                 x,  # hidden
#                 # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
#                 ln_w,
#                 gate_w,
#                 up_w,
#                 down_w,
#                 output_tensor,  # out
#                 kernel_name="MLP",
#                 # Run RMSNorm inside the kernel if NOT using SP rmsnorm
#                 fused_rmsnorm=fused_rmsnorm,
#                 skip_gamma=self.fused_rmsnorm_skip_gamma,
#                 eps=self.rms_norm_eps,
#             )
#             residual = None

#         # All-reduce or reduce-scatter, depending on whether SP is enabled
#         if self.sequence_parallel_enabled:
#             if self.neuron_config.tile_cc:
#                 output_tensor = reduce_scatter_to_sequence_parallel_region_tiled(
#                     output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
#                 )
#             else:
#                 output_tensor = reduce_scatter_to_sequence_parallel_region(
#                     output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
#                 )
#         else:
#             output_tensor = reduce_from_tensor_model_parallel_region(
#                 output_tensor, process_group=get_tp_group(self.config)
#             )

#         logger.debug(f"MLP output shape {output_tensor.shape}")
#         return (output_tensor, residual)

#     def _native_mlp(self, x, adapter_ids=None):
#         logger.debug("MLP: native compiler")
#         # all-gather is done here instead of CPL layers to
#         # avoid 2 all-gathers from up and gate projections
#         if self.sequence_parallel_enabled:
#             x = gather_from_sequence_parallel_region(
#                 x, self.sequence_dimension, process_group=get_tp_group(self.config)
#             )
#         gate_proj_output = (
#             self.gate_proj(x)
#             if not is_lora_module(self.gate_proj)
#             else self.gate_proj(x, adapter_ids)
#         )

#         up_proj_output = (
#             self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
#         )

#         down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
#         output = (
#             self.down_proj(down_proj_input)
#             if not is_lora_module(self.down_proj)
#             else self.down_proj(down_proj_input, adapter_ids)
#         )
#         logger.debug(f"MLP output shape {output.shape}")
#         return output

#     def _neuron_mm(self, x):

#         logger.info("-"*30 + " neuron_mm mlp " + "-"*30)

#         # up = self.up_u_proj(self.up_v_proj(x))
#         # gate = self.gate_u_proj(self.gate_v_proj(x))
#         # return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        
#         b, s, h = x.shape
#         x = x.view(-1, h)
#         # return nki_mm(x, self.up_v_proj.weight, self.up_u_proj.weight, 
#         #      self.gate_v_proj.weight, self.gate_u_proj.weight,
#         #      self.down_v_proj.weight, self.down_u_proj.weight)
    

#         result = svd_mlp_with_fused_kernel(
#             x, self.up_v_proj.weight.t(), self.up_u_proj.weight.t(), 
#             self.gate_v_proj.weight.t(), self.gate_u_proj.weight.t(),  
#             self.down_v_proj.weight.t(), self.down_u_proj.weight.t())
#         return result
    
#     def _svd_baseline(self, x):
#         logger.info("-"*30 + " svd baseline mlp " + "-"*30)
#         b, s, h = x.shape
#         x = x.view(-1, h)
#         up_v = nki_matmul_fully_optimized_(x.t(), self.up_v_proj.weight.t())
#         up = nki_matmul_fully_optimized_(up_v.t(), self.up_u_proj.weight.t())
#         gate_v = nki_matmul_fully_optimized_(x.t(), self.gate_v_proj.weight.t())
#         gate = nki_matmul_fully_optimized_(gate_v.t(), self.gate_u_proj.weight.t())
#         act = self.act_fn(gate) * up
#         output_v = nki_matmul_fully_optimized_(act.t() , self.down_v_proj.weight.t())
#         output = nki_matmul_fully_optimized_(output_v.t() , self.down_u_proj.weight.t())
#         return output

    
#     def _svd_flash_mlp(self, x):

#         logger.info("-"*30 + " svd-flash mlp " + "-"*30)
#         b, s, h = x.shape
#         return XUV_matmul(x.view(-1, h), self.up_v_proj.weight, self.up_u_proj.weight)  # TODO: Fix tiles padding


#     def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
#         """
#         If residual is passed in, will fuse its add into the MLP kernel
#         If rmsnorm is passed in, will fuse the rmsnorm into the MLP kernel

#         Returns a tuple of (output, residual), where residual is the output of the residual add
#         """

#         if self.mlp_kernel_enabled:
#             # Quantized MLP kernel
#             if self.quantized_mlp_kernel_enabled:
#                 return self._kernel_enabled_quantized_mlp(
#                     x, rmsnorm, residual, adapter_ids=adapter_ids
#                 )
#             # MLP kernel
#             return self._kernel_enabled_mlp(x, rmsnorm, residual, adapter_ids=adapter_ids)
#         else:
#             # No kernel
#             assert rmsnorm is None and residual is None
#             # return (self._native_mlp(x, adapter_ids=adapter_ids), None)

#             return (self._neuron_mm(x), None)
#             # return (self._svd_baseline(x), None)

################################################




def linear_with_async_allreduce(
    input: torch.Tensor,
    weight0: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    sequence_dimension: Optional[int] = 0,
    autograd_func_class: Type[torch.autograd.Function] = LinearWithAsyncCommunication,
    save_for_backward: bool = True,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    args = cast_if_autocast_enabled(
        input,
        weight0,
        weight,
        bias,
        async_grad_allreduce,
        sequence_parallel_enabled,
        sequence_dimension,
        save_for_backward,
        process_group,
    )
    verify_casted_dtype(args)
    with torch.cuda.amp.autocast(enabled=False):
        return autograd_func_class.apply(*args)



def _initialize_affine_weight_neuron(
    weight: torch.Tensor,
    init_method: Callable[[torch.Tensor], None],
    partition_dim: int,
    num_partitions:int,
    stride: int = 1,
) -> None:
    """Initialize affine weight for model parallel on Neuron device.

    Args:
        weight (Parameter):
        init_method (Callable[[Tensor], None]): Taking a Tensor and initialize its elements.
        partition_dim (int): Dimension to apply partition.
    """

    set_tensor_model_parallel_attributes(
        tensor=weight,
        is_parallel=True,
        dim=partition_dim,
        stride=stride,
        num_partitions=num_partitions,
    )

    with get_xla_rng_tracker().fork():
        init_method(weight)



class SVDColumnParallelLinear(BaseParallelLinear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p]. Here A is the weight matrix,
    X is the input.

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be batch,sequence and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all Neuron devices, otherwise, every Neuron device will have its output
                       which is Y_i = XA_i
        dtype: dtype of the weights
        device: Device on which the weights should be initialized.
    """

    def __init__(
        self,
        input_size: int,
        rank_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        stride: int = 1,
        init_method: Optional[Callable[[Any], torch.Tensor]] = None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        keep_master_weight: bool = False,
        skip_bias_add: bool = False,
        pad: bool = False,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ):
     
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.rank_size = rank_size
        self.output_size = output_size
        self.add_bias = bias
        self.gather_output = gather_output
        self.arg_init_method = init_method

        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else cast(ProcessGroup, get_tensor_model_parallel_group())

        world_size = torch.distributed.get_world_size(group=self.tensor_parallel_group)
        self.pad = pad
        if self.pad:
            self.pad_size = get_padding_length(self.output_size, world_size)
            self.output_size = self.output_size + self.pad_size

        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = utils.divide(self.output_size, world_size)
        self.dtype = dtype
        self.device = device
        self.stride = stride
        self.keep_master_weight = keep_master_weight
        self.skip_bias_add = skip_bias_add
        self.bias_shape: Optional[Tuple[int]]

        self.initialize_weight_and_bias()

        self.async_tensor_model_parallel_allreduce = not sequence_parallel_enabled and world_size > 1
        if sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(f"`sequence_parallel_enabled` is set to `True`, but got world_size of {world_size}")

            if sequence_dimension is None:
                warnings.warn(
                    "`sequence_parallel_enabled` is set to `True`, but got `sequence_dimension` as `None`. Defaulting `sequence_dimension` to 0."
                )
                sequence_dimension = 0

        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.sequence_dimension = sequence_dimension

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_async_allreduce

    def set_weight_and_bias_config(self) -> None:
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # self.weight_shape = (self.output_size_per_partition, self.input_size)

        self.weight_shape = (self.output_size_per_partition, self.rank_size)
        self.weight_partition_dim = 0

        if self.add_bias:
            bias_size = self.output_size if self.gather_output else self.output_size_per_partition
            self.bias_shape = (bias_size,)
        else:
            self.bias_shape = None

    def initialize_weight_and_bias(self):
        self.set_weight_and_bias_config()
        init_device = self.device

        # Get torch init device if device is not explicitly mentioned
        self.weight_u = Parameter(torch.empty((self.rank_size, self.input_size), device=init_device, dtype=self.dtype))
        self.weight_v = Parameter(torch.empty(*self.weight_shape, device=init_device, dtype=self.dtype))
        # Mark the true device after weight initialization
        self.device = self.weight_v.device

        # Initialize weight.
        if self.device.type == "cpu":
            self.init_weight_cpu()
        elif self.device.type == "meta":
            utils.set_tensor_model_parallel_attributes(
                tensor=self.weight_v, is_parallel=True, dim=self.weight_partition_dim,
                stride=self.stride, num_partitions=self.tensor_parallel_group.size(),
            )
        else:
            _initialize_affine_weight_neuron(
                self.weight_v, self._init_weight, partition_dim=self.weight_partition_dim,
                num_partitions=self.tensor_parallel_group.size(),
                stride=self.stride
            )
        if self.add_bias:
            assert self.bias_shape
            if self.device is None or self.device.type == "cpu":
                self.bias = Parameter(torch.empty(*self.bias_shape, dtype=self.dtype))
            else:
                self.bias = Parameter(torch.empty(*self.bias_shape, device=self.device, dtype=self.dtype))
            if self.bias.device != torch.device("meta"):
                self._init_bias()

            if not self.gather_output:
                set_tensor_model_parallel_attributes(
                    self.bias, True, 0, stride=self.stride, num_partitions=self.tensor_parallel_group.size(),
                )
        else:
            self.register_parameter("bias", None)

    def init_weight_cpu(self) -> None:
        self.master_weight = _initialize_parameter_cpu(
            param=self.weight_v,
            partition_dim=self.weight_partition_dim,
            num_partitions=self.tensor_parallel_group.size(),
            init_method=self._init_weight,
            param_dtype=self.dtype,
            stride=self.stride,
            return_master_param=self.keep_master_weight,
        )

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
            input=input_parallel,
            weight0=self.weight_u,
            weight=self.weight_v,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=NkiLinearWithAsyncCommunication,
            process_group=self.tensor_parallel_group
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

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        if not self.pad or self.pad_size == 0:
            return
        if self.output_size != model_state_dict[prefix].shape[0] + self.pad_size:
            size = model_state_dict[prefix].shape[0]
            raise RuntimeError(f"State dict {prefix} is of an unexpected size {size} expected {size - self.pad_size}")
        model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, 0, 0, self.pad_size))


class NkiLinearWithAsyncCommunication(LinearWithAsyncCommunication):
    """Linear layer execution with asynchronous communication."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight0: torch.Tensor,
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

        
        ####################### old
        total_input = input
        # if total_input.shape[1] == 1:
        #     # output = nki_sinle_token_matmul_(total_input, weight)
        #     # output = nki_sinle_token_matmul_2(total_input.transpose(1, 2), weight.t())
        #     # output = torch.einsum('...m,mn->...n', total_input, weight.t())
        #     output = torch.einsum('...m,mn,nk->...k', total_input, weight0.t(), weight.t())

        # else:
        #     B, M, K = total_input.shape
        #     total_input = total_input.view(-1, K)
        #     # print("total_input.shape: ", total_input.shape)
        #     # print("weight0.t().shape: ", weight0.t().shape)
        #     # print("weight.t().shape: ", weight.t().shape)
         
            
        #     output = fused_three_mm_XTUV(total_input.t(), weight0.t(), weight.t())
        #     output = output.reshape(B, M, -1)
        #######################
        
        # B, M, K = total_input.shape
        # total_input = total_input.view(-1, K)
        # print("total_input.shape: ", total_input.shape)
        # print("weight0.t().shape: ", weight0.t().shape)
        # print("weight.t().shape: ", weight.t().shape)
        
        output = fused_three_mm_XTUV(total_input, weight0.t(), weight.t())
        # output = output.reshape(B, M, -1)
        #######################
    
        # print("output.shape: ", output.shape)
        if bias is not None:
            output = output + bias
        return output



class SVDRowParallelLinear(BaseParallelLinear):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    .. note::
        Input(X) is supposed to be three dimensional and each dimension
        is expected to be batch, sequence, and hidden feature, respectively.
        A is the weight matrix.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the Neuron devices and we do not split
                           again.
        dtype: dtype of the weights
        device: Device on which the weights should be initialized.
    """

    def __init__(
        self,
        input_size: int,
        rank_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        stride: int = 1,
        init_method: Optional[Callable[..., Any]] = None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        keep_master_weight: bool = False,
        skip_bias_add: bool = False,
        pad: bool = False,
        reduce_output: bool = True,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        reduce_dtype: torch.dtype = None,
    ):
        # super().__init__(
        # input_size=input_size,
        # output_size=output_size,
        # bias=bias,
        # input_is_parallel=input_is_parallel,
        # dtype=dtype,
        # device=device,
        # stride=stride,
        # init_method=init_method,
        # sequence_parallel_enabled=sequence_parallel_enabled,
        # sequence_dimension= sequence_dimension,
        # keep_master_weight=keep_master_weight,
        # skip_bias_add=skip_bias_add,
        # pad=pad,
        # reduce_output=reduce_output,
        # tensor_model_parallel_group=tensor_model_parallel_group,
        # reduce_dtype=reduce_dtype)
        
        super().__init__()
        # Keep input parameters
        self.input_size = input_size
        self.rank_size = rank_size
        self.output_size = output_size
        self.add_bias = bias
        self.input_is_parallel = input_is_parallel
        self.pad = pad
        self.reduce_output = reduce_output
        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else cast(ProcessGroup, get_tensor_model_parallel_group())

        if reduce_dtype is None:
            reduce_dtype = dtype
            
        # hardware_type = hardware(get_platform_target())
        # # Updating the reduction dtype to be FLOAT32 to reduce errors due to precision
        # if os.getenv("XLA_DOWNCAST_BF16") == "1" and hardware_type == hardware.TRN2:
        #     if reduce_dtype==torch.float32:
        #         reduce_dtype=torch.float64

        self.reduce_dtype = reduce_dtype

        world_size = self.tensor_parallel_group.size()
        if self.pad:
            self.pad_size = get_padding_length(self.input_size, world_size)
            self.input_size = self.input_size + self.pad_size
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = utils.divide(self.input_size, world_size)
        self.arg_init_method = init_method
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")

        if self.sequence_parallel_enabled and sequence_dimension is None:
            warnings.warn(
                "`sequence_parallel_enabled` is set to `True`, but got `sequence_dimension` as `None`. Defaulting `sequence_dimension` to 0."
            )
            sequence_dimension = 0

        self.sequence_dimension: int = sequence_dimension  # type: ignore
        self.dtype = dtype
        self.device = device
        self.stride = stride
        self.keep_master_weight = keep_master_weight
        self.skip_bias_add = skip_bias_add
        self.bias_shape: Optional[Tuple[int]]
        self.initialize_weight_and_bias()

        self._forward_impl = linear_with_async_allreduce

    def set_weight_and_bias_config(self) -> None:
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight_shape = (self.rank_size, self.input_size_per_partition)
        self.weight_partition_dim = 1

        if self.add_bias:
            self.bias_shape = (self.output_size,)
        else:
            self.bias_shape = None

    def initialize_weight_and_bias(self) -> None:
        self.set_weight_and_bias_config()
        init_device = self.device
        
        self.weight_v = Parameter(torch.empty((self.output_size, self.rank_size), device=init_device, dtype=self.dtype))

        # Get torch init device if device is not explicitly mentioned
        self.weight_u = Parameter(torch.empty(*self.weight_shape, device=init_device, dtype=self.dtype))
        self.device = self.weight_u.device

        # Initialize weight.
        if self.device.type == "cpu":
            self.init_weight_cpu()
        elif self.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight_u, is_parallel=True, dim=self.weight_partition_dim,
                stride=self.stride, num_partitions=self.tensor_parallel_group.size(),
            )
        else:
            _initialize_affine_weight_neuron(
                self.weight_u, self._init_weight, partition_dim=self.weight_partition_dim,
                num_partitions=self.tensor_parallel_group.size(),
                stride=self.stride,
            )

        if self.add_bias:
            assert self.bias_shape
            if self.device is None or self.device.type == "cpu":
                self.bias = Parameter(torch.empty(*self.bias_shape, dtype=self.dtype))
            else:
                self.bias = Parameter(torch.empty(*self.bias_shape, device=self.device, dtype=self.dtype))
            if self.bias.device != torch.device("meta"):
                self._init_bias()
            setattr(self.bias, "sequence_parallel_enabled", self.sequence_parallel_enabled)
        else:
            self.register_parameter("bias", None)

    def init_weight_cpu(self) -> None:
        self.master_weight = _initialize_parameter_cpu(
            param=self.weight_u,
            partition_dim=self.weight_partition_dim,
            num_partitions=self.tensor_parallel_group.size(),
            init_method=self._init_weight,
            param_dtype=self.dtype,
            stride=self.stride,
            return_master_param=self.keep_master_weight,
        )

    def _init_bias(self) -> None:
        bound = 1 / math.sqrt(self.input_size_per_partition) if self.input_size_per_partition > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

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
            weight0=self.weight_u,
            weight=self.weight_v,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=NkiLinearWithAsyncCommunication,
            process_group=self.tensor_parallel_group,
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

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> None:
        if not self.pad or self.pad_size == 0:
            return
        if self.input_size != model_state_dict[prefix].shape[1] + self.pad_size:
            size = model_state_dict[prefix].shape[1]
            raise RuntimeError(f"State dict {prefix} is of an unexpected size {size} expected {size - self.pad_size}")
        model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, self.pad_size))





class NeuronLlamaMLP_SVD(nn.Module):
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
        # self.fused_rmsnorm_skip_gamma = self.config.neuron_config.fused_rmsnorm_skip_gamma
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
        # self.quantize_clamp_bound = self.neuron_config.quantize_clamp_bound
        # self.logical_nc_config = self.neuron_config.logical_nc_config
        # self.activation_quantization_type = self.neuron_config.activation_quantization_type
        mlp_bias = getattr(config, "mlp_bias", False)

        ############################################ SVD-Flash
        # self.low_rank = int(self.intermediate_size * self.hidden_size * self.config.metadata["compress_ratio"] / (self.intermediate_size + self.hidden_size))
        # self.low_rank = math.ceil(self.intermediate_size * self.hidden_size * self.config.metadata["compress_ratio"] / ((self.intermediate_size + self.hidden_size) * 128)) * 128
        # self.low_rank = round(self.intermediate_size * self.hidden_size * 0.8 / ((self.intermediate_size + self.hidden_size) * 128)) * 128
        # self.low_rank = int(self.intermediate_size * self.hidden_size * 0.8 / ((self.intermediate_size + self.hidden_size))) 
        self.low_rank = round(self.intermediate_size * self.hidden_size * self.config.metadata["compress_ratio"] / ((self.intermediate_size + self.hidden_size) * 128)) * 128

        ############################################
        if self.neuron_config.quantized_mlp_kernel_enabled and self.quantize_clamp_bound == float(
            "inf"
        ):
            logging.warning(
                "quantize_clamp_bound is not specified in NeuronConfig. We will use the default value of 1200 for llama models in quantized kernels."
            )
            self.quantize_clamp_bound = 1200.0
        if parallel_state.model_parallel_is_initialized():
            if self.neuron_config.quantized_mlp_kernel_enabled:
                # # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")
            
            
            
            ############################################ original
            # self.gate_proj = ColumnParallelLinear(
            #     self.hidden_size,
            #     self.intermediate_size,
            #     bias=mlp_bias,
            #     gather_output=False,
            #     dtype=config.neuron_config.torch_dtype,
            #     pad=True,
            #     sequence_parallel_enabled=False,
            #     sequence_dimension=None,
            #     tensor_model_parallel_group=get_tp_group(config),
            # )
            # self.up_proj = ColumnParallelLinear(
            #     self.hidden_size,
            #     self.intermediate_size,
            #     bias=mlp_bias,
            #     gather_output=False,
            #     dtype=config.neuron_config.torch_dtype,
            #     pad=True,
            #     sequence_parallel_enabled=False,
            #     sequence_dimension=None,
            #     tensor_model_parallel_group=get_tp_group(config),
            # )
            # self.down_proj = RowParallelLinear(
            #     self.intermediate_size,
            #     self.hidden_size,
            #     bias=mlp_bias,
            #     input_is_parallel=True,
            #     dtype=config.neuron_config.torch_dtype,
            #     pad=True,
            #     sequence_parallel_enabled=self.sequence_parallel_enabled,
            #     sequence_dimension=self.sequence_dimension,
            #     tensor_model_parallel_group=get_tp_group(config),
            #     reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            # )
            ############################################
            
            
             
            
            self.gate_proj = SVDColumnParallelLinear(
                self.hidden_size,
                self.low_rank,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            
            self.up_proj = SVDColumnParallelLinear(
                self.hidden_size,
                self.low_rank,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            
            self.down_proj = SVDRowParallelLinear(
                self.intermediate_size,
                self.low_rank,
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
                    
            ############################################
            
            # if self.tp_degree >= 1:
                
            #     self.gate_proj = SVDColumnParallelLinear(
            #         self.hidden_size,
            #         self.low_rank,
            #         self.intermediate_size,
            #         bias=mlp_bias,
            #         gather_output=False,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=False,
            #         sequence_dimension=None,
            #         tensor_model_parallel_group=get_tp_group(config),
            #     )
                
            #     self.up_proj = SVDColumnParallelLinear(
            #         self.hidden_size,
            #         self.low_rank,
            #         self.intermediate_size,
            #         bias=mlp_bias,
            #         gather_output=False,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=False,
            #         sequence_dimension=None,
            #         tensor_model_parallel_group=get_tp_group(config),
            #     )
                
            #     self.down_proj = SVDRowParallelLinear(
            #         self.intermediate_size,
            #         self.low_rank,
            #         self.hidden_size,
            #         bias=mlp_bias,
            #         input_is_parallel=True,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=self.sequence_parallel_enabled,
            #         sequence_dimension=self.sequence_dimension,
            #         tensor_model_parallel_group=get_tp_group(config),
            #         reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            #     )
                    
            
            # else:
            #     ############################################ SVD-Flash
            #     self.gate_v_proj = ColumnParallelLinear(
            #         self.hidden_size,
            #         self.low_rank,
            #         bias=mlp_bias,
            #         gather_output=False,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=False,
            #         sequence_dimension=None,
            #         tensor_model_parallel_group=get_tp_group(config),
            #     )
                
            #     self.gate_u_proj = ColumnParallelLinear(
            #         self.low_rank,
            #         self.intermediate_size,
            #         bias=mlp_bias,
            #         gather_output=False,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=False,
            #         sequence_dimension=None,
            #         tensor_model_parallel_group=get_tp_group(config),
            #     )
                
            #     self.up_v_proj = ColumnParallelLinear(
            #         self.hidden_size,
            #         self.low_rank,
            #         bias=mlp_bias,
            #         gather_output=False,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=False,
            #         sequence_dimension=None,
            #         tensor_model_parallel_group=get_tp_group(config),
            #     )

            #     self.up_u_proj = ColumnParallelLinear(
            #         self.low_rank,
            #         self.intermediate_size,
            #         bias=mlp_bias,
            #         gather_output=False,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=False,
            #         sequence_dimension=None,
            #         tensor_model_parallel_group=get_tp_group(config),
            #     )

            #     self.down_v_proj = RowParallelLinear(
            #         self.intermediate_size,
            #         self.low_rank,
            #         bias=mlp_bias,
            #         input_is_parallel=True,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=self.sequence_parallel_enabled,
            #         sequence_dimension=self.sequence_dimension,
            #         tensor_model_parallel_group=get_tp_group(config),
            #         reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            #     )

            #     self.down_u_proj = RowParallelLinear(
            #         self.low_rank,
            #         self.hidden_size,
            #         bias=mlp_bias,
            #         input_is_parallel=True,
            #         dtype=config.neuron_config.torch_dtype,
            #         pad=True,
            #         sequence_parallel_enabled=self.sequence_parallel_enabled,
            #         sequence_dimension=self.sequence_dimension,
            #         tensor_model_parallel_group=get_tp_group(config),
            #         reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            #     )
            #     ############################################
                
            # if self.mlp_kernel_enabled:
            #     if self.neuron_config.quantized_mlp_kernel_enabled:
            #         setattr(
            #             self.gate_proj,
            #             "post_create_quantized_module_hook",
            #             preprocess_quantized_linear_layer,
            #         )
            #         setattr(
            #             self.up_proj,
            #             "post_create_quantized_module_hook",
            #             preprocess_quantized_linear_layer,
            #         )
            #         setattr(
            #             self.down_proj,
            #             "post_create_quantized_module_hook",
            #             preprocess_quantized_linear_layer,
            #         )
            #     else:
            #         # Transpose the weights to the layout expected by kernels
            #         self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
            #         self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
            #         self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    

    def _native_mlp(self, x, adapter_ids=None):
        logger.debug("MLP: native compiler")
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections
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
            if not is_lora_module(self.down_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )
        logger.debug(f"MLP output shape {output.shape}")
        return output

    def _neuron_mm(self, x):

        # logger.info("-"*30 + " neuron_mm mlp " + "-"*30)
        print("-"*30 + " neuron_mm mlp " + "-"*30)



        S = x.shape[0]  # Get sequence length        
        
        up_T_params = get_fused_mlp_up_T_params(S)
        
        # Calculate 'gate' projection. SiLU is applied inside the kernel.
        # Input: x (S, H). Output: activated_gate_t (I, S)
        b, s, h = x.shape
        x = x.view(-1, h)
        

        activated_gate_t = fused_mlp_up_T(
            x,  self.gate_proj.weight_u.t(), self.gate_proj.weight_v.t(), self.up_proj.weight_u.t(), self.up_proj.weight_v.t(), **up_T_params
        )
        
        output = self.down_proj(activated_gate_t)
        output = output.reshape(b, s, h)
        
        
        
        # # --- Down Projection ---
        
        # output = fused_three_mm_XTUV(
        #     activated_gate_t, self.down_proj.weight_u.t(), self.down_proj.weight_v.t(), **XTUV_params
        # )
        # output = output.reshape(b, s, h)
            

        return output

    
    def _svd_baseline(self, x):
        logger.info("-"*30 + " svd baseline mlp " + "-"*30)
        b, s, h = x.shape
        x = x.view(-1, h)
        up_v = nki_matmul_fully_optimized_(x.t(), self.up_v_proj.weight.t())
        up = nki_matmul_fully_optimized_(up_v.t(), self.up_u_proj.weight.t())
        gate_v = nki_matmul_fully_optimized_(x.t(), self.gate_v_proj.weight.t())
        gate = nki_matmul_fully_optimized_(gate_v.t(), self.gate_u_proj.weight.t())
        act = self.act_fn(gate) * up
        output_v = nki_matmul_fully_optimized_(act.t() , self.down_v_proj.weight.t())
        output = nki_matmul_fully_optimized_(output_v.t() , self.down_u_proj.weight.t())
        return output



    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel
        If rmsnorm is passed in, will fuse the rmsnorm into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """

        if self.mlp_kernel_enabled:
            # Quantized MLP kernel
            if self.quantized_mlp_kernel_enabled:
                return self._kernel_enabled_quantized_mlp(
                    x, rmsnorm, residual, adapter_ids=adapter_ids
                )
            # MLP kernel
            return self._kernel_enabled_mlp(x, rmsnorm, residual, adapter_ids=adapter_ids)
        else:
            # No kernel
            # assert rmsnorm is None and residual is None
            # return (self._native_mlp(x, adapter_ids=adapter_ids), None)

            return (self._neuron_mm(x), residual)
            # return (self._svd_baseline(x), None)


#################################################

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
        self.fused_rmsnorm_skip_gamma = self.config.neuron_config.fused_rmsnorm_skip_gamma
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
        self.quantize_clamp_bound = self.neuron_config.quantize_clamp_bound
        self.logical_nc_config = self.neuron_config.logical_nc_config
        self.activation_quantization_type = self.neuron_config.activation_quantization_type
        mlp_bias = getattr(config, "mlp_bias", False)

        if self.neuron_config.quantized_mlp_kernel_enabled and self.quantize_clamp_bound == float(
            "inf"
        ):
            logging.warning(
                "quantize_clamp_bound is not specified in NeuronConfig. We will use the default value of 1200 for llama models in quantized kernels."
            )
            self.quantize_clamp_bound = 1200.0
        if parallel_state.model_parallel_is_initialized():
            if self.neuron_config.quantized_mlp_kernel_enabled:
                # # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")
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
                if self.neuron_config.quantized_mlp_kernel_enabled:
                    setattr(
                        self.gate_proj,
                        "post_create_quantized_module_hook",
                        preprocess_quantized_linear_layer,
                    )
                    setattr(
                        self.up_proj,
                        "post_create_quantized_module_hook",
                        preprocess_quantized_linear_layer,
                    )
                    setattr(
                        self.down_proj,
                        "post_create_quantized_module_hook",
                        preprocess_quantized_linear_layer,
                    )
                else:
                    # Transpose the weights to the layout expected by kernels
                    self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                    self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                    self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def _kernel_enabled_quantized_mlp(self, x, rmsnorm, residual, adapter_ids):
        full_seqlen = x.shape[1] * (self.config.neuron_config.tp_degree if self.sequence_parallel_enabled else 1)
        if full_seqlen <= self.neuron_config.seq_len_threshold_for_cc_tiling:  # Keep regular grid for TKG.
            grid = (nc(self.logical_nc_config),)
        else:  # Add CC pipelining dim for CTE kernel grid
            grid = (CCPipeline(self.neuron_config.cc_pipeline_tiling_factor) * nc(self.logical_nc_config),)
        fused_residual = residual is not None
        fused_rmsnorm = rmsnorm is not None
        logger.debug(
            f"MLP: quantized kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_nc_config={self.logical_nc_config}"
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

        if fused_rmsnorm:
            ln_w = rmsnorm.weight.unsqueeze(0)
        else:
            ln_w = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device)

        # Handle SP RMSnorm
        x_orig_dtype = x.dtype
        if self.sequence_parallel_enabled:
            # This RMSNormQuant kernel will do quantization inside, so we pass the
            # clamp_bound for clipping.
            # If we don't use this kernel, the MLP kernel below will do the
            # quantization, so we also pass clamp_bound to that kernel.
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
                clamp_bound = self.quantize_clamp_bound
                _rmsnorm_quant_fwd_call[grid](
                    x, ln_w, clamp_bound, quant_rmsnorm_out, kernel_name="QuantOnly"
                )
                x = gather_from_sequence_parallel_region(
                    quant_rmsnorm_out,
                    self.sequence_dimension,
                    process_group=get_tp_group(self.config),
                    tile_cc=self.neuron_config.tile_cc,
                )

            else:
                logger.debug(
                    "Running Quantized MLP kernel with external (native compiler) sequence-parallel RMSnorm!"
                )
                x = gather_from_sequence_parallel_region(
                    x, self.sequence_dimension, process_group=get_tp_group(self.config), tile_cc=self.neuron_config.tile_cc
                )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
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
        gate_w = self.gate_proj.weight.data
        gate_w_scale = self.gate_proj.scale
        up_w = self.up_proj.weight.data
        up_w_scale = self.up_proj.scale
        down_w = self.down_proj.weight.data
        down_w_scale = self.down_proj.scale
        clamp_bound = self.quantize_clamp_bound

        if fused_residual:
            residual_output_tensor = torch.zeros(
                size=(
                    x.shape[0],  # batch size
                    output_tensor_seqlen,
                    self.hidden_size,  # hidden size
                ),
                dtype=x.dtype,
                device=x.device,
            )

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
                clamp_bound,
                output_tensor,  # out
                add_out=residual_output_tensor,
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            residual = residual_output_tensor
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
                clamp_bound,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            if self.neuron_config.tile_cc:
                output_tensor = reduce_scatter_to_sequence_parallel_region_tiled(
                    output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
                )
            else:
                output_tensor = reduce_scatter_to_sequence_parallel_region(
                    output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
                )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(output_tensor)

        logger.debug(f"Quantized MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _kernel_enabled_mlp(self, x, rmsnorm, residual, adapter_ids):
        fused_residual = residual is not None
        fused_rmsnorm = rmsnorm is not None
        logger.debug(
            f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, skip_gamma={self.fused_rmsnorm_skip_gamma}, logical_nc_config={self.logical_nc_config}"
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
                x, self.sequence_dimension, process_group=get_tp_group(self.config), tile_cc=self.neuron_config.tile_cc
            )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
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
        if fused_rmsnorm:
            ln_w = rmsnorm.weight.unsqueeze(0)
        else:
            ln_w = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device)
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        if output_tensor_seqlen <= self.neuron_config.seq_len_threshold_for_cc_tiling:  # Keep regular grid for TKG. Messes up the MLP impl
            grid = (nc(self.logical_nc_config),)
        else:  # Add CC pipelining dim for CTE kernel grid
            grid = (CCPipeline(self.neuron_config.cc_pipeline_tiling_factor) * nc(self.logical_nc_config),)

        if fused_residual:
            residual_output_tensor = torch.zeros(
                size=(
                    x.shape[0],  # batch size
                    output_tensor_seqlen,
                    self.hidden_size,  # hidden size
                ),
                dtype=x.dtype,
                device=x.device,
            )

            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                up_w,  # up_w
                down_w,  # down_w
                output_tensor,  # out
                kernel_name="MLP",
                add_out=residual_output_tensor,
                fused_rmsnorm=fused_rmsnorm,
                skip_gamma=self.fused_rmsnorm_skip_gamma,
                eps=self.rms_norm_eps,
                store_add=True,
            )
            residual = residual_output_tensor
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,
                up_w,
                down_w,
                output_tensor,  # out
                kernel_name="MLP",
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                skip_gamma=self.fused_rmsnorm_skip_gamma,
                eps=self.rms_norm_eps,
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            if self.neuron_config.tile_cc:
                output_tensor = reduce_scatter_to_sequence_parallel_region_tiled(
                    output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
                )
            else:
                output_tensor = reduce_scatter_to_sequence_parallel_region(
                    output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config),
                )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(
                output_tensor, process_group=get_tp_group(self.config)
            )

        logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, adapter_ids=None):
        # logger.debug("MLP: native compiler")
        # # all-gather is done here instead of CPL layers to
        # # avoid 2 all-gathers from up and gate projections
        # if self.sequence_parallel_enabled:
        #     x = gather_from_sequence_parallel_region(
        #         x, self.sequence_dimension, process_group=get_tp_group(self.config)
        #     )
        # gate_proj_output = (
        #     self.gate_proj(x)
        #     if not is_lora_module(self.gate_proj)
        #     else self.gate_proj(x, adapter_ids)
        # )

        # up_proj_output = (
        #     self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        # )

        # down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        # output = (
        #     self.down_proj(down_proj_input)
        #     if not is_lora_module(self.down_proj)
        #     else self.down_proj(down_proj_input, adapter_ids)
        # )
        # logger.debug(f"MLP output shape {output.shape}")

        #############################################

        logger.info("-"*30 + " nki_matmul_fully_optimized in MLP " + "-"*30)
        b, s, h = x.shape
        x = x.view(-1, h)

        up = nki_matmul_fully_optimized_(x.t(), self.up_proj.weight.t())
        gate = nki_matmul_fully_optimized_(x.t(), self.gate_proj.weight.t())
        act = self.act_fn(gate) * up
        output = nki_matmul_fully_optimized_(act.t() , self.down_proj.weight.t())

        return output

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel
        If rmsnorm is passed in, will fuse the rmsnorm into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """

        if self.mlp_kernel_enabled:
            # Quantized MLP kernel
            if self.quantized_mlp_kernel_enabled:
                return self._kernel_enabled_quantized_mlp(
                    x, rmsnorm, residual, adapter_ids=adapter_ids
                )
            # MLP kernel
            return self._kernel_enabled_mlp(x, rmsnorm, residual, adapter_ids=adapter_ids)
        else:
            # No kernel
            assert rmsnorm is None and residual is None
            return (self._native_mlp(x, adapter_ids=adapter_ids), None)


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
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rotary_emb=self.get_rope(config=config),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rms_norm_eps=config.rms_norm_eps,
            attention_chunk_size=getattr(config, "attention_chunk_size", None),
        )

    def get_rope(self, config: InferenceConfig):
        if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
            # TODO: Check if we can just use our own implementation
            if config.neuron_config.is_medusa:
                rotary_emb = LlamaRotaryEmbedding(config)
            else:
                rotary_emb = RotaryEmbedding(
                    getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                )
        else:
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                rotary_emb = Llama3RotaryEmbedding(
                    dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                    factor=config.rope_scaling["factor"],
                    low_freq_factor=config.rope_scaling["low_freq_factor"],
                    high_freq_factor=config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                rotary_emb = LlamaRotaryEmbedding(config)

        return rotary_emb


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
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
        inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))

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
        return torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = self.get_inv_freqs(x.device)

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
        
        ####################
        self.config = config
        if self.config.metadata is not None and self.config.metadata["svd_llama"] is True:
            self.mlp = NeuronLlamaMLP_SVD(config)
        else:
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
            )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = config.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = config.neuron_config.rmsnorm_quantize_kernel_enabled
        self.mlp_kernel_fuse_residual_add = config.neuron_config.mlp_kernel_fuse_residual_add
        self.qkv_kernel_fuse_residual_add = config.neuron_config.qkv_kernel_fuse_residual_add
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.is_prefill_stage = config.neuron_config.is_prefill_stage
        self.config = config

        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.mlp_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,  # residual from previous layer used by QKV
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        entry_hidden_states = hidden_states

        qkv_fused_rmsnorm = None
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # produced another residual used by MLP
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=qkv_fused_rmsnorm,
            rotary_position_ids=rotary_position_ids,
            residual=residual,
            **kwargs,
        )

        if attn_output.residual is None:
            residual = entry_hidden_states  # input to attention
        else:
            # residual will only be returned by attn/qkv if fuse add qkv kernel is enabled
            assert self.qkv_kernel_fuse_residual_add, \
                "residual add before qkv should be computed in the previous layer, \
                 unless qkv_kernel_fuse_residual_add is specified"
            assert (
                not self.sequence_parallel_enabled
            ), "qkv_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            assert (
                self.qkv_kernel_enabled
            ), "qkv_kernel_fuse_residual_add should be used with qkv_kernel_enabled"
            residual = attn_output.residual

        hidden_states = attn_output.hidden_states
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

            if self.mlp_kernel_enabled and self.mlp_kernel_fused_rmsnorm:
                mlp_fused_rmsnorm = self.post_attention_layernorm
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                mlp_fused_rmsnorm = None

            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=mlp_fused_rmsnorm,
                adapter_ids=adapter_ids,
            )

        # if fuse residual add with qkv, we leave this add to the next layer's QKV
        # unless it is the last layer in which case we add it here
        if not self.qkv_kernel_fuse_residual_add:
            hidden_states = residual + hidden_states
            residual = None  # set to None to prevent it from being used again

        # also return residual for QKV in the next layer
        outputs = (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, residual)
        # # #################
        # print(attn_output.present_key_value[0].cpu())
        # from torch import Tensor
        # from typing import Tuple
        # B, H, T, D = 1, 8, 128, 128   # 举例
        # k1 = torch.randn(B, H, T, D)
        # v1 = torch.randn(B, H, T, D)
        # kv: Tuple[Tensor, Tensor] = (k1, v1)
        
        # print(tuple(map(repr, kv)))

        # # present_key_value_obj = (repr(present_key_value[0]), repr(present_key_value[1])) 
        # cos_cache = torch.randn(B, T, D)
        # sin_cache = torch.randn(B, T, D)
        
        # ##########################
        
        
        return outputs


    def forward_new(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,  # residual from previous layer used by QKV
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        entry_hidden_states = hidden_states

        qkv_fused_rmsnorm = None
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # # Self Attention
        # # produced another residual used by MLP
        # attn_output = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     adapter_ids=adapter_ids,
        #     rmsnorm=qkv_fused_rmsnorm,
        #     rotary_position_ids=rotary_position_ids,
        #     residual=residual,
        #     **kwargs,
        # )

        # if attn_output.residual is None:
        #     residual = entry_hidden_states  # input to attention
        # else:
        #     # residual will only be returned by attn/qkv if fuse add qkv kernel is enabled
        #     assert self.qkv_kernel_fuse_residual_add, \
        #         "residual add before qkv should be computed in the previous layer, \
        #          unless qkv_kernel_fuse_residual_add is specified"
        #     assert (
        #         not self.sequence_parallel_enabled
        #     ), "qkv_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
        #     assert (
        #         self.qkv_kernel_enabled
        #     ), "qkv_kernel_fuse_residual_add should be used with qkv_kernel_enabled"
        #     residual = attn_output.residual

        # hidden_states = attn_output.hidden_states
        residual = entry_hidden_states 
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

            if self.mlp_kernel_enabled and self.mlp_kernel_fused_rmsnorm:
                mlp_fused_rmsnorm = self.post_attention_layernorm
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                mlp_fused_rmsnorm = None

            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=mlp_fused_rmsnorm,
                adapter_ids=adapter_ids,
            )

        # if fuse residual add with qkv, we leave this add to the next layer's QKV
        # unless it is the last layer in which case we add it here
        if not self.qkv_kernel_fuse_residual_add:
            hidden_states = residual + hidden_states
            residual = None  # set to None to prevent it from being used again

        # also return residual for QKV in the next layer
        # outputs = (hidden_states, None, None, None, residual)
        B, H, T, D = 1, 8, 128, 128   # 举例
        present_key_value = (torch.randn(B, H, T, D), torch.randn(B, H, T, D))
        # cos_cache = torch.randn(B, T, D)
        # sin_cache = torch.randn(B, T, D)
        cos_cache = None
        sin_cache = None
        outputs = (hidden_states, (None,None), cos_cache, sin_cache, residual)

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
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tile_cc=self.neuron_config.tile_cc,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
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

        updated_configs = get_updated_configs(config)

        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(conf) for conf in updated_configs])

        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            # replicate fc weights since activations are sequence sharded
            self.fc = WeightGatheredColumnParallel(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True, sequence_dimension=1
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

        self.attention_chunk_size = getattr(config, "attention_chunk_size", None)


# class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
#     """
#     This class extends LlamaForCausalLM create traceable
#     blocks for Neuron.

#     Args:
#         LlamaForCausalLM (_type_): _description_
#     """

#     _model_cls = NeuronLlamaModel

#     @staticmethod
#     def load_hf_model(model_path, **kwargs):
#         return LlamaForCausalLM.from_pretrained(model_path, **kwargs)

#     @staticmethod
#     def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
#         """This function should be over-ridden in child classes as needed"""

#         neuron_config = config.neuron_config
#         # to facilitate rank usage in attention
#         num_layers = config.num_hidden_layers
#         tp_degree = neuron_config.tp_degree
#         for i in range(num_layers):
#             state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
#                 0, tp_degree // neuron_config.cp_degree, dtype=torch.int32
#             )
#             state_dict[f"layers.{i}.self_attn.global_rank.rank"] = torch.arange(
#                 0, neuron_config.world_size, dtype=torch.int32
#             )

#             """
#             for every layer do the following transformations
#             gate_w_prime = (gate_w.T * gamma).T
#             up_w_prime = (up_w.T * gamma).T
#             """
#             if (
#                 neuron_config.fused_rmsnorm_skip_gamma
#                 and not neuron_config.sequence_parallel_enabled
#             ):
#                 if neuron_config.mlp_kernel_enabled:
#                     # MLP
#                     state_dict[f"layers.{i}.mlp.gate_proj.weight"] = state_dict[
#                         f"layers.{i}.mlp.gate_proj.weight"
#                     ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
#                     state_dict[f"layers.{i}.mlp.up_proj.weight"] = state_dict[
#                         f"layers.{i}.mlp.up_proj.weight"
#                     ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)

#                 if neuron_config.qkv_kernel_enabled:
#                     # QKV
#                     state_dict[f"layers.{i}.self_attn.q_proj.weight"] = state_dict[
#                         f"layers.{i}.self_attn.q_proj.weight"
#                     ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
#                     state_dict[f"layers.{i}.self_attn.k_proj.weight"] = state_dict[
#                         f"layers.{i}.self_attn.k_proj.weight"
#                     ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
#                     state_dict[f"layers.{i}.self_attn.v_proj.weight"] = state_dict[
#                         f"layers.{i}.self_attn.v_proj.weight"
#                     ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)

#         if neuron_config.fused_qkv:
#             state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

#         if neuron_config.vocab_parallel:
#             # TODO: this hack can be removed after replication_id is ready to use
#             state_dict["embed_tokens.rank_util.rank"] = torch.arange(
#                 0, neuron_config.local_ranks_size, dtype=torch.int32
#             )

#         # to facilitate rank usage in base model
#         state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
#         # print(state_dict.keys())
#         return state_dict

#     @staticmethod
#     def update_state_dict_for_tied_weights(state_dict):
#         # print(self.config.metadata)
#         print(list(state_dict.keys()))
#         state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
#         # state_dict["embed_tokens.weight"] = state_dict["lm_head.weight"].clone()

#     @classmethod
#     def get_config_cls(cls):
#         return LlamaInferenceConfig



def add_state_dict_for_mlp(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    tp_size = cfg.neuron_config.tp_degree
    for l in range(cfg.num_hidden_layers):  # noqa: E741
        # dummpy_concat = torch.cat(
        #     [
        #         llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"],
        #         llama_state_dict[f"layers.{l}.mlp.up_proj.weight"],
        #     ],
        # )
        # hidden = llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"].shape[0]
        # per_tp_output = hidden // tp_size
        # output = torch.zeros_like(dummpy_concat)
        # for i in range(tp_size):
        #     output[2 * i * per_tp_output : (2 * i + 1) * per_tp_output,:] = llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"][i * per_tp_output : (i + 1) * per_tp_output,:]
        #     output[(2 * i + 1) * per_tp_output : (2 * i + 2) * per_tp_output,:] = llama_state_dict[f"layers.{l}.mlp.up_proj.weight"][i * per_tp_output : (i + 1) * per_tp_output,:]
        # llama_state_dict[f"layers.{l}.mlp.gateup.weight"] = output
        # del llama_state_dict[f"layers.{l}.mlp.gate_proj.weight"]
        # del llama_state_dict[f"layers.{l}.mlp.up_proj.weight"]
        # del dummpy_concat
        
        llama_state_dict[f"layers.{l}.mlp.gate_proj.weight_u"] = llama_state_dict[f"layers.{l}.mlp.gate_v_proj.weight"]
        llama_state_dict[f"layers.{l}.mlp.gate_proj.weight_v"] = llama_state_dict[f"layers.{l}.mlp.gate_u_proj.weight"]
        llama_state_dict[f"layers.{l}.mlp.up_proj.weight_u"] = llama_state_dict[f"layers.{l}.mlp.up_v_proj.weight"]
        llama_state_dict[f"layers.{l}.mlp.up_proj.weight_v"] = llama_state_dict[f"layers.{l}.mlp.up_u_proj.weight"]
        llama_state_dict[f"layers.{l}.mlp.down_proj.weight_u"] = llama_state_dict[f"layers.{l}.mlp.down_v_proj.weight"]
        llama_state_dict[f"layers.{l}.mlp.down_proj.weight_v"] = llama_state_dict[f"layers.{l}.mlp.down_u_proj.weight"]
        
        del  llama_state_dict[f"layers.{l}.mlp.gate_v_proj.weight"]
        del  llama_state_dict[f"layers.{l}.mlp.gate_u_proj.weight"]
        del  llama_state_dict[f"layers.{l}.mlp.up_v_proj.weight"]
        del  llama_state_dict[f"layers.{l}.mlp.up_u_proj.weight"]
        del  llama_state_dict[f"layers.{l}.mlp.down_v_proj.weight"]
        del  llama_state_dict[f"layers.{l}.mlp.down_u_proj.weight"]

    gc.collect()
    print(llama_state_dict.keys())
    return llama_state_dict

class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return LlamaForCausalLM.from_pretrained(model_path, **kwargs)

    # @staticmethod
    # def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
    #     """This function should be over-ridden in child classes as needed"""

    #     neuron_config = config.neuron_config
    #     # to facilitate rank usage in attention
    #     num_layers = config.num_hidden_layers
    #     tp_degree = neuron_config.tp_degree
    #     for i in range(num_layers):
    #         state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
    #             0, tp_degree // neuron_config.cp_degree, dtype=torch.int32
    #         )
    #         state_dict[f"layers.{i}.self_attn.global_rank.rank"] = torch.arange(
    #             0, neuron_config.world_size, dtype=torch.int32
    #         )

    #         """
    #         for every layer do the following transformations
    #         gate_w_prime = (gate_w.T * gamma).T
    #         up_w_prime = (up_w.T * gamma).T
    #         """
    #         if (
    #             neuron_config.fused_rmsnorm_skip_gamma
    #             and not neuron_config.sequence_parallel_enabled
    #         ):
    #             if neuron_config.mlp_kernel_enabled:
    #                 # MLP
    #                 state_dict[f"layers.{i}.mlp.gate_proj.weight"] = state_dict[
    #                     f"layers.{i}.mlp.gate_proj.weight"
    #                 ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
    #                 state_dict[f"layers.{i}.mlp.up_proj.weight"] = state_dict[
    #                     f"layers.{i}.mlp.up_proj.weight"
    #                 ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)

    #             if neuron_config.qkv_kernel_enabled:
    #                 # QKV
    #                 state_dict[f"layers.{i}.self_attn.q_proj.weight"] = state_dict[
    #                     f"layers.{i}.self_attn.q_proj.weight"
    #                 ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
    #                 state_dict[f"layers.{i}.self_attn.k_proj.weight"] = state_dict[
    #                     f"layers.{i}.self_attn.k_proj.weight"
    #                 ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
    #                 state_dict[f"layers.{i}.self_attn.v_proj.weight"] = state_dict[
    #                     f"layers.{i}.self_attn.v_proj.weight"
    #                 ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)

    #     if neuron_config.fused_qkv:
    #         state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

    #     if neuron_config.vocab_parallel:
    #         # TODO: this hack can be removed after replication_id is ready to use
    #         state_dict["embed_tokens.rank_util.rank"] = torch.arange(
    #             0, neuron_config.local_ranks_size, dtype=torch.int32
    #         )

    #     # to facilitate rank usage in base model
    #     state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
    #     # print(state_dict.keys())
    #     return state_dict
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)
            
        if config.metadata is not None and config.metadata["svd_llama"] is True:
            state_dict = add_state_dict_for_mlp(state_dict, config)

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
        # print(self.config.metadata)
        print(list(state_dict.keys()))
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        # state_dict["embed_tokens.weight"] = state_dict["lm_head.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
  