"""Driver Script for the ASPLOS OPTNKI Contest"""
import argparse
import ast
import base64
import copy
import json
import os
import time
import torch
import shutil
from tqdm import tqdm
from torch_neuronx.pyhlo.hlo_pb2 import HloModuleProto
from torch_neuronx.testing.validation import logit_validation
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig, to_torch_dtype
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.random import set_random_seed

from neuronx_distributed_inference.utils.benchmark import create_submodule_latency_collectors, register_latency_collectors, generate_report, Benchmark
import torch.nn as nn
from safetensors.torch import save_file, load_file, save_model

# Load the baseline model 
from neuronx_distributed_inference.models.llama import modeling_llama as baseline_llama

# Load the model for ASPLOS contest
from llama3 import NeuronLlamaForCausalLM
import importlib
from test import *
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
from llama_svd import NeuronLlamaForCausalLM, SVD_LlamaMLP

BENCHMARK_REPORT_FILENAME = "benchmark_report.json"
set_random_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()

    # ASPLOS contest specific
    parser.add_argument("--mode", choices=["evaluate_single", "evaluate_all", "validate", "generate"])
    parser.add_argument("--llama", type=str, default="llama")
    parser.add_argument("--enable-nki", action="store_true")
    parser.add_argument("--base-latency", type=float, default=526.15)
    parser.add_argument("--base-throughput", type=float, default=134.61)

    # Model path
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/models/llama-3.2-1b/")
    parser.add_argument("--compiled-model-path", type=str,
                        default="/home/ubuntu/traced_model/llama-3.2-1b/")

    # Evaluation
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--divergence-difference-tol", type=float, default=0.001)
    parser.add_argument("--tol-map", type=str)
    parser.add_argument("--num-tokens-to-check", type=int)

    # Generation
    parser.add_argument("--prompt", dest="prompts", action="append")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--global-topk", type=int)
    parser.add_argument("--do-sample", type=bool, default=True)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--pad-token-id", type=int, default=2)

    # Basic config
    parser.add_argument("--torch-dtype", type=to_torch_dtype, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--padding-side", type=str)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-active-tokens", type=int)
    parser.add_argument("--n-positions", type=int)
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    parser.add_argument("--output-logits", action="store_true")
    parser.add_argument("--vocab-parallel", action="store_true")

    # Attention
    parser.add_argument("--fused-qkv", action="store_true")
    parser.add_argument("--sequence-parallel-enabled", action="store_true")
    parser.add_argument("--flash-decoding-enabled", action="store_true")

    # On device sampling
    parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    parser.add_argument("--enable-bucketing", type=bool, default=True)
    parser.add_argument("--bucket-n-active-tokens", action="store_true")
    parser.add_argument("--context-encoding-buckets", nargs="+", type=int)
    parser.add_argument("--token-generation-buckets", nargs="+", type=int)

    # Parallelism
    parser.add_argument("--tp-degree", type=int, default=1)

    # Kernels
    parser.add_argument("--qkv-kernel-enabled", action="store_true")
    parser.add_argument("--attn-kernel-enabled", action="store_true")
    parser.add_argument("--mlp-kernel-enabled", action="store_true")
    parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    parser.add_argument("--quantized-kernel-lower-bound", type=float, default=1200.0)
    parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")

    return parser.parse_args()


def validate_file_exists(path):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise argparse.ArgumentError("Path must exist and be a file")
    return path


def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def prepare_inference(model_cls, args):
    # Initialize configs.
    print("Loading configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    if args.on_device_sampling:
        config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)

    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)

    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(args.model_path)
    )

    model = model_cls(args.model_path, config)

    # Compile and save model.
    compiling_start_time = time.monotonic()
    print("\nCompiling and saving model...")
    model.compile(args.compiled_model_path, debug=False)

    compiling_end_time = time.monotonic()
    total_compiling_time = compiling_end_time - compiling_start_time
    print(f"Compiling and tracing time: {total_compiling_time} seconds")

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(args.compiled_model_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - compiling_end_time
    print(f"Total model loading time: {model_loading_time} seconds")

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)
    neuron_config.pad_token_id = tokenizer.pad_token_id

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = [
        "do_sample",
        "top_k",
        "pad_token_id",
        "dynamic",
        "top_p",
        "temperature",
    ]
    generation_config_kwargs = {
        k: getattr(args, k) for k in generation_config_args if getattr(args, k) is not None
    }
    generation_config.update(**generation_config_kwargs)

    return model, tokenizer, generation_config


def prepare_inference_svd_model(model_cls, args, svd=False):
    
    if svd is True:
        args.model_path = args.model_path + "/svd_llama"
        
    # Initialize configs.
    print("Loading configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    if args.on_device_sampling:
        config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)

    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)

    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(args.model_path)
    )

    model = model_cls(args.model_path, config)

    # Compile and save model.
    compiling_start_time = time.monotonic()
    print("\nCompiling and saving model...")
    model.compile(args.compiled_model_path, debug=False)

    compiling_end_time = time.monotonic()
    total_compiling_time = compiling_end_time - compiling_start_time
    print(f"Compiling and tracing time: {total_compiling_time} seconds")

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(args.compiled_model_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - compiling_end_time
    print(f"Total model loading time: {model_loading_time} seconds")

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)
    neuron_config.pad_token_id = tokenizer.pad_token_id

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = [
        "do_sample",
        "top_k",
        "pad_token_id",
        "dynamic",
        "top_p",
        "temperature",
    ]
    generation_config_kwargs = {
        k: getattr(args, k) for k in generation_config_args if getattr(args, k) is not None
    }
    generation_config.update(**generation_config_kwargs)

    return model, tokenizer, generation_config


def generate_submodule_reports(latency_collectors, neuron_config, num_runs):
    reports = {}
    for key, collector in latency_collectors.items():
        tokens_len = neuron_config.max_length
        if key == "context_encoding_model":
            tokens_len = neuron_config.seq_len - neuron_config.max_new_tokens
        elif key == "token_generation_model":
            tokens_len = neuron_config.max_new_tokens
        reports[key] = generate_report(
            collector.latency_list, tokens_len, neuron_config.max_batch_size, num_runs
        )
    return reports


def benchmark_sampling(model, tokenizer, generation_config, prompts):
    neuron_config = model.neuron_config

    sampling_params = prepare_sampling_params(
        batch_size=neuron_config.batch_size,
        top_k=generation_config.top_k
        if isinstance(generation_config.top_k, list)
        else [generation_config.top_k],
        top_p=generation_config.top_p
        if isinstance(generation_config.top_p, list)
        else [generation_config.top_p],
        temperature=generation_config.temperature
        if isinstance(generation_config.temperature, list)
        else [generation_config.temperature],
    )

    report = {}

    # on_device_sampling flow does not support min_new_tokens
    # to override eos_tokens so we remove EOS tokens to ensure
    # token generation happens.
    modified_generation_config = copy.deepcopy(generation_config)
    if model.on_device_sampling:
        modified_generation_config.eos_token_id = []
    
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    neuron_config.max_new_tokens = neuron_config.seq_len - input_ids.shape[1]

    input_param = {
        "input_ids": input_ids,
        "generation_config": modified_generation_config,
        "attention_mask": attention_mask,
        "min_new_tokens": neuron_config.max_new_tokens,
        "max_new_tokens": neuron_config.max_new_tokens,
        "top_k": 1,
        "do_sample": not neuron_config.enable_fused_speculation,
        "sampling_params": sampling_params,
        "max_length": neuron_config.max_length
        if neuron_config.max_new_tokens is None
        else None,
    }

    latency_collectors = create_submodule_latency_collectors(model)

    def post_warmup_func():
        register_latency_collectors(latency_collectors, model)

    # Register latency collectors after warm-up to avoid recording warm-up metrics.
    generation_model = HuggingFaceGenerationAdapter(model)
    e2e_benchmark = Benchmark(
        generation_model.generate,
        input_param,
        preprocess_func=model.reset,
        post_warmup_func=post_warmup_func,
    )
    e2e_benchmark.run()
    report["e2e_model"] = generate_report(
        e2e_benchmark.latency_list,
        neuron_config.max_length,
        neuron_config.max_batch_size,
        n_runs=e2e_benchmark.num_runs,
    )
        
    report.update(
        generate_submodule_reports(
            latency_collectors, neuron_config, e2e_benchmark.num_runs
        )
    )
    
    model.reset()

    print("Benchmark completed and its result is as following")
    print(json.dumps(report, indent=4))
    with open(BENCHMARK_REPORT_FILENAME, "w") as f:
        json.dump(report, f)
    print("Completed saving result to " + BENCHMARK_REPORT_FILENAME)

    return report


def check_accuracy_logits(base_model, base_generation_config, neuron_model, tokenizer, generation_config, prompts, divergence_difference_tol, tol_map, num_tokens_to_check):
    assert (prompts is not None)

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    initial_input_ids = inputs.input_ids
    initial_attention_mask = inputs.attention_mask
    seq_len = neuron_model.config.neuron_config.seq_len

    neuron_model.config.neuron_config.max_new_tokens = seq_len - initial_input_ids.shape[1]

    # model = neuron_model.load_hf_model(neuron_model.model_path)
    model = HuggingFaceGenerationAdapter(base_model)
    new_tokens = neuron_model.config.neuron_config.max_new_tokens
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=initial_input_ids,
            attention_mask=initial_attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=base_generation_config,
        )
    expected_logits = torch.stack(outputs.scores)

    if num_tokens_to_check is not None:
        print(f"Validating logits for first {num_tokens_to_check} tokens")
        expected_logits = expected_logits[:num_tokens_to_check, :, :]

    expected_token_ids = expected_logits.argmax(dim=2).T
    expected_tokens = tokenizer.batch_decode(
        expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Expected Output: ", expected_tokens, expected_token_ids)
    print("Expected Logits Shape: ", expected_logits.shape)

    model = HuggingFaceGenerationAdapter(neuron_model)
    expected_attention_mask = torch.ones(
        (
            initial_attention_mask.shape[0],
            expected_token_ids.shape[1],
        ),
        dtype=torch.int32,
    )
    extrapolated_attention_mask = torch.cat(
        (initial_attention_mask, expected_attention_mask), dim=1
    )

    def generate_fn(input_ids):
        input_length = input_ids.shape[1]
        attention_mask = extrapolated_attention_mask[:, :input_length]
        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=seq_len - input_length,
                min_new_tokens=seq_len - input_length,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                generation_config=generation_config,
            )

        actual_logits = torch.stack(model_outputs.scores)
        actual_token_ids = actual_logits.argmax(dim=2).T
        actual_tokens = tokenizer.batch_decode(
            actual_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Actual Output: ", actual_tokens, actual_token_ids)
        print("Actual Logits Shape: ", actual_logits.shape)
        return torch.stack(model_outputs.scores)

    passed, _, status_msg = logit_validation(
        input_ids=initial_input_ids,
        generate_fn=generate_fn,
        expected_logits=expected_logits,
        tol_map=tol_map,
        divergence_difference_tol=divergence_difference_tol,
    )
    print("STATUS MSG", status_msg)
    assert passed, status_msg

    print("Passed logits validation")


def run_generation(model, tokenizer, prompts, generation_config):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")

    _, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        generation_config=generation_config,
        max_length=model.neuron_config.max_length,
    )

    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


def run_accuracy_check(
    base_model,
    base_generation_config,
    model,
    tokenizer,
    generation_config,
    prompt,
    divergence_difference_tol,
    tol_map,
    num_tokens_to_check=None,
):
    if tol_map:
        tol_map = ast.literal_eval(tol_map)

    try:
        check_accuracy_logits(
            base_model=base_model,
            base_generation_config=base_generation_config,
            neuron_model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            prompts=prompt,
            divergence_difference_tol=divergence_difference_tol,
            tol_map=tol_map,
            num_tokens_to_check=num_tokens_to_check,
        )
    except AssertionError:
        return False

    return True


def count_nki_flop_ratio(
    hlo_path_context_enc="/tmp/nxd_model/context_encoding_model/_tp0_bk0/model/graph.hlo",
    hlo_path_token_gen="/tmp/nxd_model/token_generation_model/_tp0_bk0/model/graph.hlo"
):
    hlo_macs = 0
    nki_macs = 0

    def parse_hlo_file(hlo_file_path):
        with open(hlo_file_path, 'rb') as f:
            hlo_data = f.read()

        hlo_proto = HloModuleProto()
        hlo_proto.ParseFromString(hlo_data)
        return hlo_proto

    def count_mac(hlo_proto):
        nki_mac = 0
        hlo_mac = 0

        for computation in hlo_proto.computations:
            instruction_map = {instr.id: instr for instr in computation.instructions}

            for instruction in computation.instructions:
                # Finding NKI ops
                if instruction.opcode == "custom-call":
                    if instruction.custom_call_target == 'AwsNeuronCustomNativeKernel':
                        try:
                            backend_config = instruction.backend_config
                            config = json.loads(base64.b64decode(backend_config))
                            mac_count = int(config['mac_count'])
                        except Exception:
                            mac_count = 0

                        nki_mac += mac_count
                        hlo_mac += mac_count
                elif instruction.opcode == "dot":
                    # Get dot dimension numbers
                    dnums = instruction.dot_dimension_numbers

                    # Get shapes of operands using operand_ids
                    lhs_shape = instruction_map[instruction.operand_ids[0]].shape
                    rhs_shape = instruction_map[instruction.operand_ids[1]].shape

                    # Initialize counters
                    lhs_batch = 1
                    lhs_contracting_size = 1
                    lhs_non_contracting_size = 1
                    rhs_non_contracting_size = 1

                    # Process LHS shape
                    for i in range(len(lhs_shape.dimensions)):
                        if i in dnums.lhs_contracting_dimensions:
                            lhs_contracting_size *= lhs_shape.dimensions[i]
                        elif i in dnums.lhs_batch_dimensions:
                            lhs_batch *= lhs_shape.dimensions[i]
                        else:
                            lhs_non_contracting_size *= lhs_shape.dimensions[i]

                    # Process RHS shape
                    for i in range(len(rhs_shape.dimensions)):
                        if i not in dnums.rhs_contracting_dimensions and \
                           i not in dnums.rhs_batch_dimensions:
                            rhs_non_contracting_size *= rhs_shape.dimensions[i]

                    mac_count = (lhs_batch * lhs_non_contracting_size *
                                 lhs_contracting_size * rhs_non_contracting_size)
                    hlo_mac += mac_count

        return hlo_mac, nki_mac

    hlo_proto_context_enc = parse_hlo_file(hlo_path_context_enc)
    hlo_proto_token_gen = parse_hlo_file(hlo_path_token_gen)
    hlo_mac_context_enc, nki_mac_context_enc = count_mac(hlo_proto_context_enc)
    hlo_mac_token_gen, nki_mac_token_gen = count_mac(hlo_proto_token_gen)

    # FIXME: Need to consider token gen get executed more
    hlo_macs = hlo_mac_context_enc + hlo_mac_token_gen
    nki_macs = nki_mac_context_enc + nki_mac_token_gen

    if hlo_macs == 0:
        assert nki_macs == 0
        nki_flop_ratio = 0
    else:
        nki_flop_ratio = nki_macs / hlo_macs

    return nki_flop_ratio


def calculate_score(base_latency, base_throughput, accuracy, latency, throughput, nki_flop_ratio):
    

    increased_throughput = throughput / base_throughput
    reduced_latency = base_latency / latency

    final_score = accuracy * reduced_latency * increased_throughput * (1 + nki_flop_ratio)

    print ('In this final score of ', final_score, ' the contestant got a breakdown as follows.')
    print ('accuracy: ', accuracy)
    print ('reduced_latency: ', reduced_latency)
    print ('increased throughput: ',  increased_throughput)
    print ('nki flop ratio: ', nki_flop_ratio)
    
    return final_score



def get_submodule(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_submodule(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def svd_flash(args):

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model = model.eval()
    args.compress_ratio = 0.8
    print('-' * 90)
    print("model: ", args.model_path)
    # print(model)
    
    save_dir = args.model_path + "/svd_llama"
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(args.model_path):
        if filename.endswith(".json"):
            src_path = os.path.join(args.model_path, filename)
            dst_path = os.path.join(save_dir, filename)
            shutil.copy2(src_path, dst_path)

    if 'opt' in args.model_path:
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
        if "Llama" in args.model_path or "llama" in args.model_path or "vicuna" in args.model_path:
            svd_mlp = SVD_LlamaMLP(config=model.config, compress_ratio=args.compress_ratio)
        elif "mistral" in model_path:
            svd_mlp = SVD_MistralMLP(config=model.config, compress_ratio=args.compress_ratio)
        elif 'opt' in model_path:
            svd_decoder = SVDOPTDecoderLayer(model.config, compress_ratio=args.compress_ratio)
        
        for name in subset:
            W = subset[name].weight.data.float()
            dtype = W.dtype
            
            U, S, VT = torch.linalg.svd(W, full_matrices=False)
            num_s_after_trunc =round(W.shape[0] * W.shape[1] * args.compress_ratio / ((W.shape[0] + W.shape[1]) * 128)) * 128
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = VT[:num_s_after_trunc, :]
            truc_sigma = torch.diag(truc_s)
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in args.model_path:
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

    if "Llama-3.1-8B" not in args.model_path and 'lm_head.weight' in state_dict:
        del state_dict['lm_head.weight']  # tie_word_embeddings
    save_file(state_dict, os.path.join(save_dir,"model.safetensors"))
    
   



def main():
    args = parse_args()
    if not args.prompts:
        args.prompts = ["I believe the meaning of life is"]
    args.batch_size = len(args.prompts)
    args.max_length = args.seq_len
    args.tol_map = "{None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}"
    
    # svd_flash(args)
    
    # model, tokenizer, generation_config = prepare_inference(NeuronLlamaForCausalLM, args)
    model, tokenizer, generation_config = prepare_inference_svd_model(NeuronLlamaForCausalLM, args, svd=True)
    



    

    prompts = parse_prompts("prompts.txt")
    prompt_data = parse_prompt_data("prompt_data.txt")
    assert len(prompts) == len(prompt_data)

    total_score = 0

    # Iterate through the prompts
    for i, prompt in enumerate(prompts):
        data = prompt_data[i]
        base_latency = float(data[3])
        base_throughput = float(data[4])
        accuracy = 1
       

        report = benchmark_sampling(model, tokenizer, generation_config, [prompt])

        latency = report["e2e_model"]["latency_ms_p99"]
        throughput = report["e2e_model"]["throughput"]

        nki_flop_ratio = count_nki_flop_ratio()

        score = calculate_score(base_latency, base_throughput, accuracy, latency, throughput, nki_flop_ratio)
        print(
            f"Prompt: {prompt}\n"
            f"Final Score: {score}\n"
            f"\tAccuracy: {accuracy}\n"
            f"\tLatency: {latency}\n"
            f"\tThroughput: {throughput}\n"
            f"\tNKI FLOPs Ratio: {nki_flop_ratio}"
        )
        total_score += score

    print(f"\nTotal Score: {total_score}\n")

    


if __name__ == "__main__":
    main()
