## Setup Steps

1. Launch a Tranium instance using [AWS EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#LaunchInstances:) with the following settings:  
   i. **Name and tags**: SVD-Flash  
   ii. **Amazon Machine Image**: Deep Learning AMI Neuron (Ubuntu 22.04)  
   iii. **Instance type**: trn1.2xlarge  
   iv. **Key pair (login)**: create a new key pair  
   v. **Metadata version [under “Advanced details”]**: V2 only (otherwise, you will encounter a not authorized error)  
   vi. When connecting to these instances via SSH, use the username of *ubuntu*.

2. Activate the Neuron virtual environment to run inference by running  
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate

3. Download `Llama-3.2-1B` from Hugging face
    ``` 
    mkdir models

    huggingface-cli download --token  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 meta-llama/Llama-3.2-1B --local-dir ./models/llama-3.2-1b

    cd /home/ubuntu/models/llama-3.2-1b

    mv model.safetensors  model_ori.safetensors

4. Download the weights after SVD and post-training processing
   ```
   wget "https://huggingface.co/SVD-Flash/llama-3.2-1b_0.8_svd/resolve/main/llama-3.2-1b_svd_0.8_weights.safetensors?download=true" \
     -O model.safetensors   

5. Testing Example (Without Tensor Parallelism): Llama inference with logit matching accuracy check using custom error tolerances
   ```
   inference_demo \
    --model-type llama \
    --task-type causal-lm \
    run \
    --model-path /home/ubuntu/models/llama-3.2-1b \
    --compiled-model-path /home/ubuntu/traced_model/llama-3.2-1b \
    --torch-dtype bfloat16 \
    --batch-size 2 \
    --max-context-length 32 \
    --seq-len 64 \
    --check-accuracy-mode logit-matching \
    --divergence-difference-tol 0.005 \
    --tol-map "{5: (1e-5, 0.02)}" \
    --enable-bucketing \
    --top-k 1 \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is" \
    --compress-ratio 0.8

6. The following modifications are needed for `LlamaAttention` and `LlamaMLP` in the `transformers.models.llama.modeling_llama.py` module:
```

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, compress_ratio=0.8):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        ########### original
        # self.q_proj = nn.Linear(
        #     config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.k_proj = nn.Linear(
        #     config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.v_proj = nn.Linear(
        #     config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.o_proj = nn.Linear(
        #     config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        # )
        
        ########### low rank decomposition
        low_rank_qo = compress_ratio * (config.hidden_size * config.num_attention_heads * self.head_dim) // (config.hidden_size + config.num_attention_heads * self.head_dim)
        low_rank_kv = compress_ratio * (config.hidden_size * config.num_key_value_heads * self.head_dim) // (config.hidden_size + config.num_key_value_heads * self.head_dim)
        low_rank_qo = int(low_rank_qo)
        low_rank_kv = int(low_rank_kv)

        self.q_v_proj = nn.Linear(config.hidden_size, low_rank_qo, bias=config.attention_bias)
        self.q_u_proj = nn.Linear(low_rank_qo, config.num_attention_heads * self.head_dim, bias=config.attention_bias)

        self.k_v_proj = nn.Linear(config.hidden_size, low_rank_kv, bias=config.attention_bias)
        self.k_u_proj = nn.Linear(low_rank_kv, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        self.v_v_proj = nn.Linear(config.hidden_size, low_rank_kv, bias=config.attention_bias)
        self.v_u_proj = nn.Linear(low_rank_kv, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        
        self.o_v_proj = nn.Linear(config.num_attention_heads * self.head_dim, low_rank_qo, bias=config.attention_bias)
        self.o_u_proj = nn.Linear(low_rank_qo, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


        query_states = self.q_u_proj(self.q_v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_states = self.k_u_proj(self.k_v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        value_states =  self.v_u_proj(self.v_v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))
        return attn_output, attn_weights



class LlamaMLP(nn.Module):
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

        low_rank = int(self.intermediate_size * self.hidden_size * compress_ratio / (self.intermediate_size + self.hidden_size))

        self.gate_v_proj = nn.Linear(self.hidden_size, low_rank, bias=config.mlp_bias)
        self.gate_u_proj = nn.Linear(low_rank, self.intermediate_size, bias=config.mlp_bias)

        self.up_v_proj = nn.Linear(self.hidden_size, low_rank, bias=config.mlp_bias)
        self.up_u_proj = nn.Linear(low_rank, self.intermediate_size, bias=config.mlp_bias)

        self.down_v_proj = nn.Linear(self.intermediate_size, low_rank, bias=config.mlp_bias)
        self.down_u_proj = nn.Linear(low_rank, self.hidden_size, bias=config.mlp_bias)
        

    def forward(self, x):

        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
    
