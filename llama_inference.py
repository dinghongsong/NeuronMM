import argparse
import ast
import copy
import json
import os
import time
import warnings
from enum import Enum
from functools import partial
from typing import Type
import shutil
from tqdm import tqdm
import torch
from neuronx_distributed.quantization.quantization_config import (
    ActivationQuantizationType,
    QuantizationType,
)
import torch.nn as nn
from safetensors.torch import save_file, load_file, save_model

from transformers import AutoTokenizer, GenerationConfig
import math
from models.application_base import NeuronApplicationBase
from models.config import (
    FusedSpecNeuronConfig,
    OnDeviceSamplingConfig,
    ChunkedPrefillConfig,
    to_torch_dtype,
)
from models.dbrx.modeling_dbrx import NeuronDbrxForCausalLM
from models.llama.modeling_llama import NeuronLlamaForCausalLM, SVD_LlamaMLP
from models.mixtral.modeling_mixtral import NeuronMixtralForCausalLM
from modules.lora_serving import LoraServingConfig
from utils.accuracy import (
    check_accuracy,
    check_accuracy_logits,
    get_generate_outputs,
)
from utils import argparse_utils
from utils.benchmark import benchmark_sampling
from utils.debug_utils import capture_model_inputs
from utils.distributed import get_init_rank, get_init_world_size
from utils.exceptions import LogitMatchingValidationError
from utils.hf_adapter import load_pretrained_config
from utils.random import set_random_seed
from utils.constants import BENCHMARK_REPORT_PATH
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM

set_random_seed(0)


MODEL_TYPES = {
    "llama": {"causal-lm": NeuronLlamaForCausalLM},
    "mixtral": {"causal-lm": NeuronMixtralForCausalLM},
    "dbrx": {"causal-lm": NeuronDbrxForCausalLM},
}


class CheckAccuracyMode(Enum):
    SKIP_ACCURACY_CHECK = "skip-accuracy-check"
    TOKEN_MATCHING = "token-matching"
    LOGIT_MATCHING = "logit-matching"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=MODEL_TYPES.keys(), required=True)
    parser.add_argument("--task-type", type=str, required=True)
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run")
    setup_run_parser(run_parser)

    args = parser.parse_args()

    # Handle deprecated "--logical-neuron-cores" argument.
    if args.logical_neuron_cores is not None:
        warning_message = (
            "'--logical-neuron-cores' is deprecated and no longer needed. "
            "By default, NxD Inference now chooses the correct LNC based on instance type. "
            "To set LNC manually, use the '--logical-nc-config' argument. "
            "In a future release, the '--logical-neuron-cores' argument will be removed."
        )
        warnings.warn(warning_message, category=UserWarning)
        args.logical_nc_config = args.logical_neuron_cores
        del args.logical_neuron_cores

    return args


def setup_run_parser(run_parser: argparse.ArgumentParser):
    run_parser.add_argument("--model-path", type=str, required=True)
    run_parser.add_argument("--compiled-model-path", type=str, required=True)

    # SVD-Flash
    run_parser.add_argument("--compress-ratio", type=float, default=0.8)

    # Evaluation
    run_parser.add_argument("--benchmark", action="store_true")
    run_parser.add_argument(
        "--check-accuracy-mode",
        type=CheckAccuracyMode,
        choices=list(CheckAccuracyMode),
        default=CheckAccuracyMode.SKIP_ACCURACY_CHECK,
    )
    run_parser.add_argument("--expected-outputs-path", type=validate_file_exists)
    run_parser.add_argument("--divergence-difference-tol", type=float, default=0.001)
    run_parser.add_argument("--tol-map", type=str)
    run_parser.add_argument("--num-tokens-to-check", type=int)

    # Generation
    run_parser.add_argument("--prompt", dest="prompts", type=str, action="append", required=True)
    run_parser.add_argument("--top-k", type=int, default=1)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--temperature", type=float, default=1.0)
    run_parser.add_argument("--global-topk", type=int)
    run_parser.add_argument("--do-sample", action="store_true", default=False)
    run_parser.add_argument("--dynamic", action="store_true", default=False)
    run_parser.add_argument("--pad-token-id", type=int, default=0)

    # Basic config
    run_parser.add_argument("--torch-dtype", type=to_torch_dtype)
    run_parser.add_argument("--batch-size", type=int)
    run_parser.add_argument("--padding-side", type=str)
    run_parser.add_argument("--allow-input-truncation", action="store_true")
    run_parser.add_argument("--seq-len", type=int)
    run_parser.add_argument("--n-active-tokens", type=int)
    run_parser.add_argument("--n-positions", type=int)
    run_parser.add_argument("--max-context-length", type=int)
    run_parser.add_argument("--max-new-tokens", type=int)
    run_parser.add_argument("--max-length", type=int)
    run_parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    run_parser.add_argument("--attention-dtype", type=to_torch_dtype)
    run_parser.add_argument("--output-logits", action="store_true")
    run_parser.add_argument("--vocab-parallel", action="store_true")
    run_parser.add_argument("--layer-boundary-markers", action="store_true", default=False)

    # Attention
    run_parser.add_argument("--fused-qkv", action="store_true")
    run_parser.add_argument("--sequence-parallel-enabled", action="store_true")
    run_parser.add_argument("--weight-gather-seq-len-threshold", type=int)
    run_parser.add_argument("--flash-decoding-enabled", action="store_true")

    # Continuous batching
    run_parser.add_argument("--ctx-batch-size", type=int)
    run_parser.add_argument("--tkg-batch-size", type=int)
    run_parser.add_argument("--max-batch-size", type=int)
    run_parser.add_argument("--is-continuous-batching", action="store_true")

    # KV cache
    run_parser.add_argument("--kv-cache-batch-size", type=int)
    run_parser.add_argument("--kv-cache-padding-size", type=int)
    run_parser.add_argument("--disable-kv-cache-tiling", action="store_true")

    # On device sampling
    run_parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    run_parser.add_argument("--enable-bucketing", action="store_true")
    run_parser.add_argument("--bucket-n-active-tokens", action="store_true")
    run_parser.add_argument("--context-encoding-buckets", nargs="+", type=int)
    run_parser.add_argument("--prefix-buckets", nargs="+", type=int)
    run_parser.add_argument("--token-generation-buckets", nargs="+", type=int)

    # Quantization
    run_parser.add_argument("--quantized", action="store_true")
    run_parser.add_argument("--quantized-checkpoints-path", type=str)
    run_parser.add_argument(
        "--quantization-type", type=str, choices=[t.value for t in QuantizationType]
    )
    run_parser.add_argument("--kv-cache-quant", action="store_true")
    run_parser.add_argument("--quantization-dtype", type=str)
    run_parser.add_argument(
        "--modules-to-not-convert-file",
        type=get_modules_to_not_convert_json,
        dest="modules_to_not_convert_lists",
    )

    # MoE
    run_parser.add_argument("--capacity-factor", type=float)
    run_parser.add_argument("--early-expert-affinity-modulation", action="store_true")
    run_parser.add_argument("--disable-normalize-top-k-affinities", action="store_true")
    run_parser.add_argument("--fused-shared-experts", action="store_true")

    # Router Config
    run_parser.add_argument("--router-act-fn", type=str)
    run_parser.add_argument("--router-dtype", type=str)

    # Speculative decoding
    run_parser.add_argument("--draft-model-path", type=str)
    run_parser.add_argument("--draft-model-tp-degree", type=int, default=None)
    run_parser.add_argument("--compiled-draft-model-path", type=str)
    run_parser.add_argument("--enable-fused-speculation", action="store_true", default=False)
    run_parser.add_argument("--enable-eagle-speculation", action="store_true", default=False)
    run_parser.add_argument("--enable-eagle-draft-input-norm", action="store_true", default=False)

    run_parser.add_argument("--speculation-length", type=int)
    run_parser.add_argument("--spec-batch-size", type=int)

    # Medusa decoding
    run_parser.add_argument("--is-medusa", action="store_true")
    run_parser.add_argument("--medusa-speculation-length", type=int)
    run_parser.add_argument("--num-medusa-heads", type=int)
    run_parser.add_argument("--medusa-tree-json", type=load_json_file, dest="medusa_tree")

    # Token Tree
    run_parser.add_argument("--token-tree-json", type=load_json_file, dest="token_tree_config")

    # Parallelism
    run_parser.add_argument("--tp-degree", type=int)
    run_parser.add_argument("--cp-degree", type=int)
    run_parser.add_argument("--pp-degree", type=int)
    run_parser.add_argument("--ep-degree", type=int)
    run_parser.add_argument("--world-size", type=int)
    run_parser.add_argument("--start_rank_id", type=int)
    run_parser.add_argument("--local_ranks_size", type=int)
    run_parser.add_argument(
        "--enable-torch-dist",
        action="store_true",
        help="Use torch.distributed (gloo) backend when running multi-node examples. "
        "This is useful for ensuring processes on different nodes are in sync",
    )
    run_parser.add_argument(
        "--save-sharded-checkpoint",
        action="store_true",
        help="Save sharded checkpoints to disk when compiling NxDI model. "
        "When loading NxDI model, sharded checkpoints will be loaded from the compiled model path.",
    )
    run_parser.add_argument(
        "--skip-sharding",
        action="store_true",
        help="Skip sharding checkpoints when compiling NxDI model. "
    )

    # PA and CF
    run_parser.add_argument(
        "--enable-block-kv-layout", dest="is_block_kv_layout", action="store_true"
    )
    run_parser.add_argument("--pa-num-blocks", type=int)
    run_parser.add_argument("--pa-block-size", type=int)
    run_parser.add_argument(
        "--enable-chunked-prefill", dest="is_chunked_prefill", action="store_true"
    )
    run_parser.add_argument(
        "--enable-prefix-caching", dest="is_prefix_caching", action="store_true"
    )
    run_parser.add_argument("--max-num-seqs", type=int)

    # Async
    run_parser.add_argument("--async-mode", action="store_true")

    # Lora
    run_parser.add_argument("--enable-lora", action="store_true")
    run_parser.add_argument("--max-loras", type=int, default=1)
    run_parser.add_argument("--max-lora-rank", type=int, default=16)
    run_parser.add_argument("--target-modules", nargs="+")
    run_parser.add_argument("--max-loras-on-cpu", type=int)
    run_parser.add_argument("--lora-ckpt-path", dest="lora_ckpt_paths", type=str, action="append")
    run_parser.add_argument("--adapter-id", dest="adapter_ids", type=str, action="append")

    # Kernels
    run_parser.add_argument("--qkv-kernel-enabled", action="store_true")
    run_parser.add_argument("--qkv-kernel-nbsd-layout", action="store_true")
    run_parser.add_argument("--attn-kernel-enabled", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--mlp-kernel-enabled", action="store_true")
    run_parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    run_parser.add_argument("--fused-rmsnorm-skip-gamma", action="store_true")
    run_parser.add_argument(
        "--activation-quantization-type",
        type=str,
        choices=[e.value for e in ActivationQuantizationType],
    )
    run_parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    run_parser.add_argument("--quantize-clamp-bound", type=float, default=float("inf"))
    run_parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")
    run_parser.add_argument("--qkv-kernel-fuse-residual-add", action="store_true")
    run_parser.add_argument("--attn-tkg-nki-kernel-enabled", action="store_true")
    run_parser.add_argument("--attn-tkg-builtin-kernel-enabled", action="store_true")
    run_parser.add_argument("--attn-block-tkg-nki-kernel-enabled", action="store_true")
    run_parser.add_argument("--attn-block-tkg-nki-kernel-cache-update", action="store_true")
    run_parser.add_argument("--k-cache-transposed", action="store_true")

    # Logical NeuronCore Configuration (LNC)
    lnc_group = run_parser.add_mutually_exclusive_group()
    lnc_group.add_argument(
        "--logical-neuron-cores", type=int
    )  # Deprecated. Use --logical-nc-config.
    lnc_group.add_argument("--logical-nc-config", type=int)

    # Compiler Args
    run_parser.add_argument("--cc-pipeline-tiling-factor", type=int, default=2)
    run_parser.add_argument("--enable-spill-reload-dge", action="store_true")

    # CPU
    run_parser.add_argument("--on-cpu", action="store_true")

    # Report generation
    run_parser.add_argument(
        "--benchmark-report-path",
        type=str,
        default=BENCHMARK_REPORT_PATH,
        help="File path to save benchmark report."
    )

    # Debugging
    run_parser.add_argument(
        "--capture-indices",
        nargs="+",
        action=argparse_utils.StringOrIntegers,
        default=None,
        help=f"Specify '{argparse_utils.AUTO}' when using check accuracy mode with {CheckAccuracyMode.LOGIT_MATCHING} for inferrring capture indices when the test fails and use the indices to capture inputs. Otherwise, provide any number of integer values for capturing inputs at those indices.")
    run_parser.add_argument("--input-capture-save-dir", type=str, default=None)

    run_parser.add_argument("--cast-type", choices=["config", "as-declared"], default="config",
                            help="If set to 'config', all parameters will be casted to neuron_config.torch_dtype. "
                            "If set to 'as-declared', casting will be done based on the dtype set for each parameter")

    # Optional demo arguments
    run_parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="skip model warmup.",
    )

    run_parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="skip model compilation. If this option is set, then compiled model must be "
        "present at path specified by --compiled-model-path argument",
    )
    run_parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only perform model compilation.",
    )
    run_parser.add_argument(
        "--compile-dry-run",
        action="store_true",
        help="Perform a compilation dry run (minimal model trace)",
    )
    run_parser.add_argument(
        "--hlo-debug",
        action="store_true",
        help="Adds metadata into the generated HLO. This metadata maps the HLO "
        "operators to the corresponding lines in the PyTorch code",
    )
    run_parser.add_argument(
        "--apply-seq-ids-mask",
        action='store_true',
        help="Avoid KV cache update on inactive (padded) seq_ids"
    )
    run_parser.add_argument(
        "--input-start-offsets",
        nargs="+",
        default=None,
        type=int,
        help="Shift the input right by an offset. There can be multiple offsets, each per sequence."
        "If only 1 value is provided, all sequences will be shifted by this amount. "
        "This flag can be used to test chunked attention."
    )


def validate_file_exists(path):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise argparse.ArgumentError("Path must exist and be a file")
    return path


def load_json_file(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_submodule(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_submodule(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_modules_to_not_convert_json(json_path):
    modules_to_not_convert, draft_model_modules_to_not_convert = None, None
    assert os.path.exists(json_path), f"File not found: {json_path}"
    data = load_json_file(json_path)
    if "model" in data:
        modules_to_not_convert = data["model"]["modules_to_not_convert"]
    elif "modules_to_not_convert" in data:
        modules_to_not_convert = data["modules_to_not_convert"]
    # Handle draft model modules if they exist
    if "draft_model" in data:
        draft_model_modules_to_not_convert = data["draft_model"]["modules_to_not_convert"]
    return modules_to_not_convert, draft_model_modules_to_not_convert


def create_neuron_config(model_cls, args):
    # Initialize configs.
    print("-" * 100)
    print("Loading configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    if args.on_device_sampling:
        config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)
    if args.is_chunked_prefill:
        max_num_seqs = config_kwargs.pop("max_num_seqs", 0)
        config_kwargs["chunked_prefill_config"] = ChunkedPrefillConfig(
            max_num_seqs=max_num_seqs,
        )

    if (args.quantized and args.quantization_dtype == "f8e4m3") or args.kv_cache_quant:
        os.environ["XLA_HANDLE_SPECIAL_SCALAR"] = "1"
        os.environ["UNSAFE_FP8FNCAST"] = "1"

    if args.modules_to_not_convert_lists:
        modules_to_not_convert, draft_modules = args.modules_to_not_convert_lists
        if modules_to_not_convert is not None:
            config_kwargs["modules_to_not_convert"] = modules_to_not_convert
        if draft_modules is not None:
            config_kwargs["draft_model_modules_to_not_convert"] = draft_modules

    adapter_ids = None
    if args.enable_lora:
        config_kwargs["lora_config"] = LoraServingConfig(
            max_loras=args.max_loras,
            max_lora_rank=args.max_lora_rank,
            target_modules=args.target_modules,
            max_loras_on_cpu=args.max_loras_on_cpu,
            lora_ckpt_paths=args.lora_ckpt_paths,
        )
        adapter_ids = args.adapter_ids
    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)
    return adapter_ids, neuron_config

def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def run_generation(
    model,
    tokenizer,
    prompts,
    generation_config,
    draft_model=None,
    adapter_ids=None,
    input_capture_hook=None,
    input_start_offsets=None,
):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")
    if len(prompts) == 1 and model.config.neuron_config.batch_size > 1:
        prompts = prompts * model.config.neuron_config.batch_size

    _, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        draft_model=draft_model,
        generation_config=generation_config,
        adapter_ids=adapter_ids,
        max_length=model.neuron_config.max_length,
        input_capture_hook=input_capture_hook,
        input_start_offsets=input_start_offsets
    )

    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


def run_accuracy_check(
    model,
    tokenizer,
    generation_config,
    prompt,
    check_accuracy_mode,
    divergence_difference_tol,
    tol_map,
    num_tokens_to_check=None,
    draft_model=None,
    expected_outputs_path=None,
    input_start_offsets=None,
):
    if model.neuron_config.is_medusa:
        # Medusa doesn't use greedy sampling, so check accuracy doesn't work.
        assert (
            check_accuracy_mode == CheckAccuracyMode.SKIP_ACCURACY_CHECK
        ), "Accuracy checking not supported for Medusa"
    if input_start_offsets:
        assert all(offset < model.config.neuron_config.max_context_length for offset in input_start_offsets), "Input offset has to be less than max context length"
    if check_accuracy_mode == CheckAccuracyMode.SKIP_ACCURACY_CHECK:
        print("\nSkipping accuracy check")
        return

    expected_outputs = None
    if expected_outputs_path is not None:
        expected_outputs = torch.load(expected_outputs_path)

    if check_accuracy_mode == CheckAccuracyMode.TOKEN_MATCHING:
        print("\nChecking accuracy by token matching")
        check_accuracy(
            model,
            tokenizer,
            generation_config,
            prompt=prompt,
            draft_model=draft_model,
            expected_token_ids=expected_outputs,
            num_tokens_to_check=num_tokens_to_check,
            input_start_offsets=input_start_offsets,
        )
    elif check_accuracy_mode == CheckAccuracyMode.LOGIT_MATCHING:
        assert draft_model is None, "Logit matching not supported for speculation"
        print("\nChecking accuracy by logit matching")

        expected_logits = None
        if expected_outputs is not None:
            expected_logits = torch.stack(expected_outputs.scores)

        if tol_map:
            tol_map = ast.literal_eval(tol_map)

        check_accuracy_logits(
            model,
            tokenizer,
            generation_config,
            prompt=prompt,
            expected_logits=expected_logits,
            divergence_difference_tol=divergence_difference_tol,
            tol_map=tol_map,
            num_tokens_to_check=num_tokens_to_check,
            input_start_offsets=input_start_offsets,
        )
    else:
        raise ValueError(f"Unsupported check accuracy mode: {check_accuracy_mode}")




def run_inference(model_cls: Type[NeuronApplicationBase], args, svd=False):

    if svd is True:
        args.model_path = args.model_path + "/svd_llama"
    
    ############################################ Configure generation config
    
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
    remaining_kwargs = generation_config.update(**generation_config_kwargs)
    for k, v in remaining_kwargs.items():
        generation_config.__dict__[k] = v

    ############################################  config

    adapter_ids, neuron_config = create_neuron_config(model_cls, args)
    config = model_cls.get_config_cls()( # LlamaInferenceConfig -> InferenceConfig
        neuron_config, load_config=load_pretrained_config(args.model_path),
        metadata={"svd_llama": svd,
                  "compress_ratio": args.compress_ratio}
    )

    ############################################  compile

    model = model_cls(args.model_path, config)
    model.compile(args.compiled_model_path, debug=args.hlo_debug, dry_run=args.compile_dry_run)
    if not args.on_cpu:
        print("\nLoading model to Neuron...")
        model.load(args.compiled_model_path)
    else:
        print("\nLoading model to CPU...")
        model.to_cpu()
    
    
    ############################################  Generate outputs

    # # Load tokenizer.
    # tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)

    # run_generation(
    #     model,
    #     tokenizer,
    #     args.prompts,
    #     generation_config,
    #     draft_model=None,
    #     adapter_ids=adapter_ids,
    #     input_capture_hook=None,
    #     input_start_offsets=args.input_start_offsets,
    # )

    ############################################  benchmark

    print('-' * 90)
    print("model: ", args.model_path)
    report = benchmark_sampling(model, None, generation_config, benchmark_report_path=None)
    with open("/home/ubuntu/SVD-Flash/output.log", "a") as f:
        print('-' * 90, file=f)
        print("model: ", args.model_path, file=f)
        print(json.dumps(report, indent=4), file=f)
    return report


def svd_flash(args):

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model = model.eval()

    print('-' * 90)
    print(model)
    
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
            num_s_after_trunc = math.ceil(W.shape[0] * W.shape[1] * args.compress_ratio / ((W.shape[0] + W.shape[1]) * 128)) * 128
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
    if 'lm_head.weight' in state_dict:
        del state_dict['lm_head.weight']
    save_file(state_dict, os.path.join(save_dir,"model.safetensors"))





if __name__ == "__main__":
    args = parse_args()
    report_wo_svd = run_inference(NeuronLlamaForCausalLM, args, svd=False)
    svd_flash(args)
    report_svd = run_inference(NeuronLlamaForCausalLM, args, svd=True)
    
    print("e2e_model time wo svd: ", report_wo_svd["e2e_model"]["latency_ms_avg"])
    print("e2e_model time with svd: ", report_svd["e2e_model"]["latency_ms_avg"])
    print("E2E Speedup: ", report_wo_svd["e2e_model"]["latency_ms_avg"] / report_svd["e2e_model"]["latency_ms_avg"])
    with open("/home/ubuntu/SVD-Flash/output.log", "a") as f:
        print("e2e_model time wo svd: ", report_wo_svd["e2e_model"]["latency_ms_avg"], file=f)
        print("e2e_model time with svd: ", report_svd["e2e_model"]["latency_ms_avg"], file=f)
        print("E2E Speedup: ", report_wo_svd["e2e_model"]["latency_ms_avg"] / report_svd["e2e_model"]["latency_ms_avg"], file=f)
    
