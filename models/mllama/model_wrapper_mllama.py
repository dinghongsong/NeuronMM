import torch

from models.mllama.utils import (
    NUM_IMAGE_PER_PROMPT,
    get_image_tensors,
)
from models.model_wrapper import DecoderModelInstance, ModelWrapper
from modules.generation.sampling import prepare_sampling_params
from modules.padding import pad_tensor


class ModelWrapperMllama(ModelWrapper):
    """
    A class that wraps the Llama Multimodal model for context encoding, speculation, and token generation tasks.
    This class overrides input_generator() to provide additional pixel_values, aspect_ratios in the sample inputs for tracing.
    It removes inputs related to medusa and Lora since Llama MM does not support them.
    """

    def input_generator(
        self,
    ):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )

            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)

            # Get the count of sampling params currently supported.
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32
            )

            # Default to 1x1 aspect ratios (because the aspect ratio values are used in computations)
            pixel_values, aspect_ratios, num_chunks, has_image = get_image_tensors(
                self.config, [[]] * self.neuron_config.batch_size, (n_active_tokens > 1)
            )

            vision_mask = torch.zeros(
                (self.neuron_config.batch_size, NUM_IMAGE_PER_PROMPT, 2),
                dtype=torch.int32,
            )

            inputs.append(
                (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    pixel_values,
                    aspect_ratios,
                    vision_mask,
                    num_chunks,
                    has_image,
                )
            )

        return inputs

    def get_model_instance(self):
        return MMDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )

    def _forward_with_pad(self, *args):
        seq_ids = args[3]
        if len(args) > 4:
            args_need_padding = args[4:10]
        else:
            args_need_padding = None

        padded_args = []
        # pad input_ids, attn_mask and position_ids
        for arg in args[0:3]:
            padded_shape = list(arg.shape)
            padded_shape[0] = self.neuron_config.batch_size
            padded_tensor, _ = pad_tensor(arg, padded_shape)
            padded_args.append(padded_tensor)

        # need to handle seq_ids separately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,
        # 0]. then the kv cache of padded input could be written into the first cache line, so we need to pad as [0,
        # 2, 1, 3] instead

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list
            + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        reorder_seq_ids = not self.is_prefix_caching
        padded_seq_ids, indices = torch.sort(padded_seq_ids) if reorder_seq_ids else (padded_seq_ids, None)
        padded_args.append(padded_seq_ids)

        if args_need_padding is not None:
            for arg in args_need_padding:
                padded_shape = list(arg.shape)
                padded_shape[0] = self.neuron_config.batch_size
                padded_tensor, _ = pad_tensor(arg, padded_shape)
                padded_tensor = torch.index_select(padded_tensor, 0, indices)
                padded_args.append(padded_tensor)

        outputs = self._forward(*padded_args)
        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        if self.is_neuron():
            logits = outputs
            return logits[: seq_ids.shape[0]]
        else:
            logits, *kv_cache = outputs
            return [logits[: seq_ids.shape[0]], *kv_cache]


class MMDecoderModelInstance(DecoderModelInstance):
    def enable_or_disable_sp(self, parent_module, sp_enabled, name_prefix="model"):
        for name, module in parent_module.named_children():
            module_full_name = f"{name_prefix}.{name}"
            sp_modify_allowed = not getattr(module, "dont_modify_sequence_parallel_enabled", False)
            if sp_modify_allowed and getattr(module, "sequence_dimension", None) is not None:
                setattr(module, "sequence_parallel_enabled", sp_enabled)
            # Recurse
            if sp_modify_allowed:
                self.enable_or_disable_sp(module, sp_enabled, module_full_name)

    def get(self, bucket_rank, **kwargs):
        """
        Override DecoderModelInstance.get() to add vision_tokens and vision_key_values aliasing, and for disabling SP at lower buckets.
        """

        self.module, self.input_output_aliases = super().get(bucket_rank, **kwargs)

        # TODO: This is a hack for disabling sequence parallel at low sequence lengths. Replace this with a more permanent solution in NxDI.
        if self.neuron_config.sequence_parallel_enabled:
            # Enable SP at seq len >= 1k, otherwise disable
            seq_len = self.module.n_positions
            sp_enabled = bool(seq_len >= 1024)
            print(f"Setting sp_enabled={sp_enabled} in model for seq len {seq_len}", flush=True)
            self.enable_or_disable_sp(self.module, sp_enabled=sp_enabled)

        if self.module.kv_mgr is not None:
            past_key_values = self.module.kv_mgr.past_key_values
            vision_key_values = (
                self.module.kv_mgr.vision_key_values if not self.neuron_config.skip_vision else []
            )
        else:
            past_key_values = self.module.past_key_values
            vision_key_values = []

        num_output_from_trace = 1 + len(past_key_values)

        for i in range(len(vision_key_values)):
            self.input_output_aliases[vision_key_values[i]] = num_output_from_trace
            num_output_from_trace += 1

        return self.module, self.input_output_aliases
