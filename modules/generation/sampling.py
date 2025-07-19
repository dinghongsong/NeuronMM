import logging
from typing import Any, Dict, Union

import torch
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.operators.topk import topk as nxd_topk
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.utils.utils import hardware
from neuronxcc.nki._private_kernels.cumsum import cumsum as nki_cumsum
from torch_neuronx.utils import get_platform_target
from torch_neuronx.xla_impl.ops import nki_jit, xla_hlo_call

from models.config import NeuronConfig, OnDeviceSamplingConfig

logger = logging.getLogger("Neuron")


def mask_padded_logits(logits, rank_id, world_size, pad_size=None):
    if pad_size is None or pad_size == 0:
        return logits

    # invalid if rank_id == tp_degree - 1
    last_rank_mask = torch.eq(
        torch.full(logits.shape, world_size - 1, device=logits.device, dtype=torch.int32),
        rank_id.broadcast_to(logits.shape),
    )
    #   and index >= logits.shape[-1] - pad
    on_pad_mask = torch.ge(
        torch.arange(logits.shape[-1], device=logits.device, dtype=torch.int32).broadcast_to(
            logits.shape
        ),
        torch.full(
            logits.shape, logits.shape[-1] - pad_size, device=logits.device, dtype=torch.int32
        ),
    )
    invalid_mask = last_rank_mask * on_pad_mask
    logits = torch.where(
        invalid_mask, torch.full_like(logits, torch.finfo(logits.dtype).min), logits
    )

    return logits


def cumsum(tensor_in, dim, on_cpu: False):
    if on_cpu:
        logger.debug("On CPU, using torch cumsum")
        return torch.cumsum(tensor_in, dim=dim)
    init_shape_len = len(tensor_in.shape)
    cumsum_dim = dim % init_shape_len
    last_dim = init_shape_len - 1
    is_transposed = False
    if cumsum_dim != last_dim:
        tensor_in = torch.transpose(tensor_in, cumsum_dim, last_dim)
        is_transposed = True
    init_shape = tensor_in.shape
    cumsum_len = init_shape[last_dim]
    # Prioritize nki kernel for float dtype, then matmul cumsum if not input is not float
    if torch.is_floating_point(tensor_in):
        logger.debug("Using NKI cumsum")
        tensor_in = tensor_in.view(-1, cumsum_len)
        nki_cumsum_func = nki_jit()(nki_cumsum)
        output = torch.zeros_like(tensor_in, device=tensor_in.device, dtype=tensor_in.dtype)
        nki_cumsum_func(tensor_in, output, axis=1)
        output = output.view(init_shape)
        if is_transposed:
            output = torch.transpose(output, cumsum_dim, last_dim)
        return output
    else:
        logger.debug("Using matmul cumsum")
        triu = torch.triu(
            torch.ones(
                cumsum_len,
                cumsum_len,
                dtype=tensor_in.dtype,
                device=tensor_in.device,
            )
        )
        output = tensor_in @ triu
        if is_transposed:
            output = torch.transpose(output, cumsum_dim, last_dim)
        return output


@xla_hlo_call
def rand_like(tensor):
    dtype = tensor.dtype
    shape = tensor.sizes
    minimum = dtype.Constant(constant_value=0)
    maximum = dtype.Constant(constant_value=1)
    return dtype[shape].Rng(minimum, maximum, distribution=1)  # Uniform distribution


def validate_sampling_params(
    params: torch.Tensor, on_device_sampling_config: Union[Dict[str, Any], OnDeviceSamplingConfig]
) -> None:
    """
    Validates sampling parameters for language models.

    Args:
    params (torch.Tensor): Tensor of shape (batch_size, 3) containing sampling parameters
                           in the order: top-k, top-p, temperature.
    on_device_sampling_config

    Raises:
    ValueError: If any of the parameters are invalid.
    """
    if params.shape[1] != 3:
        raise ValueError(f"Expected tensor of shape (batch_size, 3), but got {params.shape}")

    # autocast params tensor to float32
    params = params.to(torch.float32)

    # Unpack parameters
    top_k, top_p, temperature = params[:, 0], params[:, 1], params[:, 2]

    if isinstance(on_device_sampling_config, OnDeviceSamplingConfig):
        global_top_k = on_device_sampling_config.global_topk
    else:
        global_top_k = on_device_sampling_config["global_topk"]

    # Validate top-k value range
    valid_top_k = (top_k == -1) | ((top_k > 0) & (top_k <= global_top_k))
    if not torch.all(valid_top_k):
        raise ValueError(
            f"Invalid top-k values found. top-k must be -1 or greater than 0 but less than or equal to {global_top_k=}. Found {top_k=}."
        )

    # checks if top-k values can be represented as integers
    if not torch.equal(top_k, top_k.floor()):
        raise ValueError(
            f"Invalid top-k values found. top-k values should be able to be represented as integer values, but found decimal parts. Found {top_k=}."
        )

    # Validate top-p
    valid_top_p = (top_p > 0.0) & (top_p <= 1.0)
    if not torch.all(valid_top_p):
        raise ValueError(
            f"Invalid top-p values found. top-p must be in the range (0.0, 1.0]. Found {top_p=}."
        )

    # Validate temperature
    valid_temp = temperature > 0.0
    if not torch.all(valid_temp):
        raise ValueError(
            f"Invalid temperature values found. Temperature must be strictly greater than 0.0. Found {temperature=}."
        )


def prepare_sampling_params(batch_size, top_k=[1], top_p=[1.0], temperature=[1.0]):
    top_k = prepare_tensor(top_k)
    top_p = prepare_tensor(top_p)
    temperature = prepare_tensor(temperature)

    assert (
        top_k.shape[0] == top_p.shape[0] == temperature.shape[0]
    ), f"sampling params shapes don't match. \
        Got top_k shape: {top_k.shape}, top_p shape: {top_p.shape}, temperature shape: {temperature.shape}"

    if top_k.shape[0] == 1:
        top_k = top_k.broadcast_to(batch_size)
        top_p = top_p.broadcast_to(batch_size)
        temperature = temperature.broadcast_to(batch_size)
    stacked = torch.stack([top_k, top_p, temperature], dim=1)
    return stacked


def prepare_tensor(val: Union[torch.Tensor, list, float]):
    if not torch.is_tensor(val):
        if not isinstance(val, list):
            val = [val]
        val = torch.tensor(val)
    return val


class Sampler(torch.nn.Module):
    """
    Use this to implement sampling techniques

    """

    def __init__(self, neuron_config: NeuronConfig, do_sample=None):
        super().__init__()
        self.on_device_sampling = neuron_config.on_device_sampling_config is not None

        assert self.on_device_sampling, "on device configs is not initialized"

        if hasattr(neuron_config, "is_medusa"):
            self.is_medusa = neuron_config.is_medusa
        else:
            self.is_medusa = False

        self.neuron_config = neuron_config
        self.do_sample = (
            do_sample
            if do_sample is not None
            else neuron_config.on_device_sampling_config.do_sample
        )
        self.dynamic = neuron_config.on_device_sampling_config.dynamic
        self.deterministic = neuron_config.on_device_sampling_config.deterministic
        self.global_topk = neuron_config.on_device_sampling_config.global_topk
        self.IGNORED_LOGITS_VALUE = (
            -3000
        )  # large negative values will be transformed to ~0 in softmax, this is to ignore tokens that are beyond topk range
        if not self.neuron_config.on_cpu:
            if (
                hasattr(self.neuron_config, "use_draft_group")
                and self.neuron_config.use_draft_group
            ):
                self.process_group = parallel_state.get_speculative_draft_group(as_list=False)
            else:
                self.process_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.process_group = None

    def _soft_max(self, logits, dim):
        return torch.nn.functional.softmax(input=logits, dim=dim)

    def _get_top_k_num_stages(self):
        hardware_type = hardware(get_platform_target())
        if (
            hardware_type == hardware.TRN2
            and self.neuron_config.tp_degree == self.neuron_config.world_size == 64
            and self.neuron_config.logical_nc_config == 2
        ):
            return 3
        elif (
            hardware_type == hardware.TRN1
            and self.neuron_config.tp_degree == self.neuron_config.world_size == 32
        ):
            return 2
        else:
            return 1

    def _top_k_masked(self, logits, top_k, dim, rank_id):
        if self.global_topk > 0:
            if self.neuron_config.on_cpu:
                sorted_logits, indeces = torch.topk(input=logits, k=self.global_topk, dim=dim)
            else:
                sorted_logits, indeces = nxd_topk(
                    tensor=logits,
                    k=self.global_topk,
                    dim=dim,
                    gather_dim=dim,
                    process_group=self.process_group,
                    stages=self._get_top_k_num_stages(),
                    rank_id=rank_id,
                )
        else:
            sorted_logits, indeces = torch.sort(input=logits, dim=dim, descending=True)

        vocab_size = sorted_logits.shape[-1]
        mask = torch.arange(vocab_size, device=logits.device)
        mask = mask.broadcast_to(*sorted_logits.shape)

        mask = torch.greater_equal(mask, top_k)
        sorted_logits = sorted_logits.masked_fill_(mask, self.IGNORED_LOGITS_VALUE)
        return sorted_logits, indeces

    def _top_p(self, top_k_logits_values, probs_cumsum, top_p, dim):
        top_p_mask = torch.greater(probs_cumsum, top_p)
        top_k_logits_values = top_k_logits_values.masked_fill_(
            top_p_mask, self.IGNORED_LOGITS_VALUE
        )
        probs_soft_max = self._soft_max(top_k_logits_values, dim)  # custom call
        probs_cumsum = cumsum(tensor_in=probs_soft_max, dim=dim, on_cpu=self.neuron_config.on_cpu)
        return probs_cumsum

    def _rand_selector(self, probs_cumsum, num_samples=1):
        if probs_cumsum.dim() == 3:
            # Eagle TKG could pass probs_cumsum of shape (batch_size, speculation_len, global_k)

            shape = (probs_cumsum.shape[0], 1, num_samples)
        else:
            shape = (probs_cumsum.shape[0], num_samples)

        if self.deterministic:
            rand_selector = torch.full(shape, 0.5, device=probs_cumsum.device)
        else:
            zeros = torch.zeros(shape, device=probs_cumsum.device, dtype=probs_cumsum.dtype)
            rand_selector = (
                torch.rand_like(zeros) if self.neuron_config.on_cpu else rand_like(zeros)
            )
        return rand_selector

    def _multinomial(self, probs, dim, num_samples=1):
        probs_cumsum = cumsum(tensor_in=probs, dim=dim, on_cpu=self.neuron_config.on_cpu)
        rand_selector = self._rand_selector(probs_cumsum, num_samples)
        greater_than_rand = torch.greater(rand_selector, probs_cumsum)
        counts = torch.sum(greater_than_rand, dim=dim).unsqueeze(dim)
        return counts

    def _argmax_sample(self, token_logits, return_values, dim):
        if self.neuron_config.on_cpu:
            return torch.argmax(token_logits, dim=dim)
        else:
            # distributed argmax
            tokens = nxd_argmax(
                tensor=token_logits,
                dim=dim,
                gather_dim=dim,
                keepdim=False,
                process_group=self.process_group,
            )
            values = torch.ones(tokens.shape, dtype=token_logits.dtype, device=tokens.device)
            if return_values:
                return tokens, values
            return tokens

    def _multinomial_sample(self, token_logits, sampling_params, return_values, dim, rank_id):
        batch_size = token_logits.shape[0]
        top_k = sampling_params[:, 0]
        top_p = sampling_params[:, 1]
        temperature = sampling_params[:, 2]
        if token_logits.dim() == 3:
            # Eagle TKG could pass token_logits of shape (batch_size, speculation_len, vocab_size)
            sampling_param_shape = (batch_size, 1, 1)
        else:
            sampling_param_shape = (batch_size, 1)
        top_k = top_k.reshape(*sampling_param_shape)
        top_p = top_p.reshape(*sampling_param_shape)
        temperature = temperature.reshape(*sampling_param_shape)

        top_k_logits_values, top_k_logits_indices = self._top_k_masked(
            token_logits, top_k, dim, rank_id
        )
        if self.is_medusa:
            return top_k_logits_indices

        if self.dynamic or torch.any(temperature != 1.0):
            top_k_logits_values = torch.divide(top_k_logits_values, temperature)

        if self.dynamic or torch.any(top_p < 1.0):  # apply top_p sampling
            probs_soft_max = self._soft_max(top_k_logits_values, dim)
            probs_cumsum = cumsum(
                tensor_in=probs_soft_max, dim=dim, on_cpu=self.neuron_config.on_cpu
            )
            top_p = torch.max(torch.min(probs_cumsum), top_p)
            top_p_mask = torch.greater(probs_cumsum, top_p).index_fill_(
                dim, torch.tensor([0], device=top_p.device), False
            )  # need to keep at least one token
            top_k_logits_values = top_k_logits_values.masked_fill_(
                top_p_mask, self.IGNORED_LOGITS_VALUE
            )

        probs_soft_max = self._soft_max(top_k_logits_values, dim)  # custom call
        if return_values:
            return top_k_logits_indices, probs_soft_max

        counts = self._multinomial(probs_soft_max, dim)
        return torch.gather(input=top_k_logits_indices, dim=dim, index=counts).flatten()

    def forward(self, token_logits, sampling_params, return_values=False, rank_id=None):
        """
        forward to perform topk, topp, temperature and multinomial sampling.

        Inputs:
            token_logits: tensor whose first dimension is Batch Size
                and whose final dimension is Vocabulary Size
            sampling_params: a 2D tensor of size (Batch Size, 3)
            containing the following sampling params:
                * top_k: value to use for top_k sampling
                * top_p: value to use for top_p sampling
                * temperature: value to use for temperature sampling

        Output:
            Tensor containing 1 sampled token id per batch size.
            Output size is (Batch Size,)

        Note: Using torch.multinomial on device causes trace to hang.
        This is because torch.multinomial performs a number of distribution
        validation steps, which is content dependent. Hence we implement multinomial
        distribution here instead.
        """
        dim = len(token_logits.shape) - 1  # vocab_size dimension

        if self.do_sample or self.dynamic or self.is_medusa:
            return self._multinomial_sample(
                token_logits, sampling_params, return_values, dim, rank_id
            )
        else:
            return self._argmax_sample(token_logits, return_values, dim)
