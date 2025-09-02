import torch
import json, os
from tqdm import tqdm
import torch.nn as nn
import math
from safetensors.torch import save_file, load_file, save_model
import shutil
# from component.svd_llama import SVD_LlamaMLP
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from models.config import NeuronConfig, OnDeviceSamplingConfig
from models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM

# model_path = "/home/ubuntu/model_hf/Llama-3.1-8B-Instruct/"
# traced_model_path = "/home/ubuntu/traced_model/Llama-3.1-8B-Instruct/"

# model_path = "/home/ubuntu/model_hf/Llama-3.1-8B/"
# traced_model_path = "/home/ubuntu/traced_model/Llama-3.1-8B/"
# quantized_model_path = "/home/ubuntu/model_hf/Llama-3.1-8B-quantized"



# model_path = "/home/ubuntu/model_hf/Llama-2-7B/"
# traced_model_path = "/home/ubuntu/traced_model/Llama-2-7B/"


# model_path = "/home/ubuntu/model_hf/Llama-3.1-8B-FP8/"
# traced_model_path = "/home/ubuntu/traced_model/Llama-3.1-8B-FP8/"
# quantized_model_path = "/home/ubuntu/model_hf/Llama-3.1-8B-FP8-quantized"

# model_path = "/home/ubuntu/model_hf/Llama-3.2-1B/"
# traced_model_path = "/home/ubuntu/traced_model/Llama-3.2-1B/"

model_path = "/home/ubuntu/models/llama-3.2-1b/"
traced_model_path = "/home/ubuntu/traced_model/llama-3.2-1b/"

# quantized_model_path = "/home/ubuntu/model_hf/Llama-3.2-1B-quantized"

torch.manual_seed(0)
ratio=0.8
def get_model_from_huggingface(model_id):
        # if "opt" in model_id or "mistral" in model_id:
    #     tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    # else:
    #     tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    model = model.eval()
    return model, tokenizer

def get_submodule(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_submodule(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

from transformers import LlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, AutoModelForCausalLM


# class SVDLlamaConfig(LlamaConfig):
#     model_type = "svd_llama"

# class SVDLlamaForCausalLM(LlamaForCausalLM):
#     config_class = SVDLlamaConfig
#     def __init__(self, config):
#         super().__init__(config)
#         for i, layer in enumerate(self.model.layers):
#             svd_mlp = SVD_LlamaMLP(config=config, compress_ratio=0.8)  
#             layer.mlp = svd_mlp

# AutoConfig.register("svd_llama", SVDLlamaConfig)
# AutoModelForCausalLM.register(SVDLlamaConfig, SVDLlamaForCausalLM)


def svd_flash():

    # model = SVDLlamaForCausalLM.from_pretrained(model_path + "svd_llama")
    # tokenizer = AutoTokenizer.from_pretrained(model_path + "svd_llama")

    model, tokenizer = get_model_from_huggingface(model_path)
    print('-' * 90)
    print(model)
    
    os.makedirs(model_path + "svd_llama", exist_ok=True)
    if 'opt' in model_path:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition ...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = get_submodule(layer)
        subset = {
            k: v for k, v in subset.items()
            if "self_attn" not in k
        }
        #### Replace MLP ####
        if "Llama" in model_path or "vicuna" in model_path:
            svd_mlp = SVD_LlamaMLP(config=model.config, compress_ratio=ratio)
        elif "mistral" in model_path:
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_path:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        
        for name in subset:
            W = subset[name].weight.data.float()
            dtype = W.dtype
            
            U, S, VT = torch.linalg.svd(W, full_matrices=False)
            num_s_after_trunc = math.ceil(W.shape[0] * W.shape[1] * ratio / ((W.shape[0] + W.shape[1]) * 128)) * 128
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = VT[:num_s_after_trunc, :]
            truc_sigma = torch.diag(truc_s)
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_path:
                if "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()
    
    state_dict = model.state_dict() 
    print(state_dict)
    # save_file(state_dict, "model.safetensors")
    save_model(model, "model.safetensors")


    # model.config.model_type = "svd_llama"
    # model.save_pretrained(model_path + "svd_llama")
    # tokenizer.save_pretrained(model_path + "svd_llama")
    print(model)

def run_llama():
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id,
    }
    generation_config.update(**generation_config_kwargs)

    neuron_config = NeuronConfig(
        tp_degree=2,
        batch_size=1,
        max_context_length=32,
        seq_len=64,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=True,
        flash_decoding_enabled=False,
        # ################
        # quantized=True,
        # quantized_checkpoints_path=quantized_model_path,
        # quantization_dtype="int8",
        # quantization_type="per_tensor_symmetric"
    )

    
    
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
        
    #################
    # Quantize the model and save it to `quantized_checkpoints_path`.
    # NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)
    #################

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronLlamaForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # # Generate outputs.
    # print("\nGenerating outputs...")
    # prompts = ["I believe the meaning of life is", "The color of the sky is"]
    # sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    # print(f"Prompts: {prompts}")
    # inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    # generation_model = HuggingFaceGenerationAdapter(model)
    # outputs = generation_model.generate(
    #     inputs.input_ids,
    #     generation_config=generation_config,
    #     attention_mask=inputs.attention_mask,
    #     max_length=model.config.neuron_config.max_length,
    #     sampling_params=sampling_params,
    # )
    # output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print("Generated outputs:")
    # for i, output_token in enumerate(output_tokens):
    #     print(f"Output {i}: {output_token}")

    
    print('-' * 90)
    print("model: ", model_path)
    print("benchmark_sampling: ")
    report = benchmark_sampling(model, None, generation_config, benchmark_report_path=None)
    with open("/home/ubuntu/neuronx-distributed-inference/output.log", "a") as f:
        print('-' * 90, file=f)
        print("model: ", model_path, file=f)
        print(json.dumps(report, indent=4), file=f)

    return report



def run_svd_llama():

    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id,
    }
    generation_config.update(**generation_config_kwargs)

    neuron_config = NeuronConfig(
        tp_degree=1,
        batch_size=1,
        max_context_length=32,
        seq_len=64,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=True,
        flash_decoding_enabled=False,
        # ################
        # quantized=True,
        # quantized_checkpoints_path=quantized_model_path,
        # quantization_dtype="int8",
        # quantization_type="per_tensor_symmetric"
    )

    
    
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
        metadata={"svd_llama": True,
                  "compress_ratio": 0.8}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
        
    #################
    # Quantize the model and save it to `quantized_checkpoints_path`.
    # NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)
    #################

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path , config)
    model.compile(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronLlamaForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["I believe the meaning of life is", "The color of the sky is"]
    sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    
    print('-' * 90)
    print("model: ", model_path)
    print("benchmark_sampling: ")
    report = benchmark_sampling(model, None, generation_config, benchmark_report_path=None)
    with open("/home/ubuntu/neuronx-distributed-inference/output.log", "a") as f:
        print('-' * 90, file=f)
        print("model: ", model_path, file=f)
        print(json.dumps(report, indent=4), file=f)

    return report


def run_svd_llama_ori():

    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id,
    }
    generation_config.update(**generation_config_kwargs)

    neuron_config = NeuronConfig(
        tp_degree=1,
        batch_size=1,
        max_context_length=32,
        seq_len=64,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=True,
        flash_decoding_enabled=False,
        # ################
        # quantized=True,
        # quantized_checkpoints_path=quantized_model_path,
        # quantization_dtype="int8",
        # quantization_type="per_tensor_symmetric"
    )

    
    
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
        metadata={"svd_llama": True,
                  "compress_ratio": 0.8}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path + "svd_llama", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
        
    #################
    # Quantize the model and save it to `quantized_checkpoints_path`.
    # NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)
    #################

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path + "svd_llama", config)
    model.compile(traced_model_path + "svd_llama")
    tokenizer.save_pretrained(traced_model_path + "svd_llama")

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronLlamaForCausalLM(traced_model_path + "svd_llama")
    model.load(traced_model_path + "svd_llama")
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path + "svd_llama")

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["I believe the meaning of life is", "The color of the sky is"]
    sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    
    print('-' * 90)
    print("model: ", model_path)
    print("benchmark_sampling: ")
    report = benchmark_sampling(model, None, generation_config, benchmark_report_path=None)
    with open("/home/ubuntu/neuronx-distributed-inference/output.log", "a") as f:
        print('-' * 90, file=f)
        print("model: ", model_path, file=f)
        print(json.dumps(report, indent=4), file=f)

    return report



def run_svd_llama2():

    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id,
    }
    generation_config.update(**generation_config_kwargs)

    neuron_config = NeuronConfig(
        tp_degree=1,
        batch_size=1,
        max_context_length=32,
        seq_len=64,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=True,
        flash_decoding_enabled=False,
        # ################
        # quantized=True,
        # quantized_checkpoints_path=quantized_model_path,
        # quantization_dtype="int8",
        # quantization_type="per_tensor_symmetric"
    )

    
    
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
        metadata={"svd_llama": True,
                  "compress_ratio": 0.8}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
        
    #################
    # Quantize the model and save it to `quantized_checkpoints_path`.
    # NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)
    #################

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path , config)
    model.compile(traced_model_path)
    # tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    # print("\nLoading model from compiled checkpoint...")
    # model = NeuronLlamaForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["I believe the meaning of life is", "The color of the sky is"]
    sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    
    print('-' * 90)
    print("model: ", model_path)
    print("benchmark_sampling: ")
    report = benchmark_sampling(model, None, generation_config, benchmark_report_path=None)
    with open("/home/ubuntu/neuronx-distributed-inference/output.log", "a") as f:
        print('-' * 90, file=f)
        print("model: ", model_path, file=f)
        print(json.dumps(report, indent=4), file=f)

    return report

if __name__ == "__main__":
    # run_llama()
    # svd_flash()
    run_svd_llama2()
