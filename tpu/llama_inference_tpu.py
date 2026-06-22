"""
End-to-end (MLP-path) smoke test for the TPU/Pallas SVD-SwiGLU MLP.

This is the TPU counterpart of ``NeuronMM/llama_inference.py``: it loads the
SVD-factored MLP weights produced by that script's ``svd_flash()`` conversion
and runs every layer's MLP through the Pallas kernel (``svd_swiglu_mlp``),
checking the result against the plain decomposed-linear reference
(``SVD_LlamaMLP.forward``) and timing it.

Scope: this exercises the **MLP path only** -- attention, norms, embeddings, and
token generation are out of scope (the kernel work lives entirely in the MLP).
It's the "does the ported kernel run end-to-end on real weights" check, not a
full generation harness.

Weights
-------
- With ``--weights-path model.safetensors`` it reads the keys
  ``model.layers.{i}.mlp.{gate,up,down}_{u,v}_proj.weight`` saved by
  ``llama_inference.py::svd_flash`` (and matching ``SVD_LlamaMLP``).
- Without a path it synthesizes random weights of the right shapes so the
  pipeline runs anywhere (used for the CPU smoke test).

Run (CPU, no TPU -- uses Pallas interpret mode):
    python NeuronMM/tpu/llama_inference_tpu.py --interpret --layers 4

Run (TPU host):
    python NeuronMM/tpu/llama_inference_tpu.py --weights-path .../model.safetensors
"""

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from svd_mlp_tpu import svd_swiglu_mlp, svd_swiglu_mlp_ref


# ---------------------------------------------------------------------------
# Config (mirrors SVD_LlamaMLP's rank formula)
# ---------------------------------------------------------------------------
@dataclass
class LlamaMLPConfig:
    hidden_size: int = 2048          # Llama-3.2-1B
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    compress_ratio: float = 0.8
    tile: int = 128

    @property
    def low_rank(self) -> int:
        I, H, r = self.intermediate_size, self.hidden_size, self.compress_ratio
        # round(I*H*ratio / ((I+H)*tile)) * tile   -- identical to SVD_LlamaMLP
        return round(I * H * r / ((I + H) * self.tile)) * self.tile


# One layer's six SVD factor matrices (jnp arrays), in the kernel's convention:
#   gate/up: U (I, r), V (r, H)        down: U (H, r), V (r, I)
@dataclass
class MLPWeights:
    U_gate: jnp.ndarray
    V_gate: jnp.ndarray
    U_up: jnp.ndarray
    V_up: jnp.ndarray
    U_down: jnp.ndarray
    V_down: jnp.ndarray


# ---------------------------------------------------------------------------
# Weight sources
# ---------------------------------------------------------------------------
def random_weights(cfg: LlamaMLPConfig, dtype, seed=0):
    H, I, r = cfg.hidden_size, cfg.intermediate_size, cfg.low_rank
    s = 1.0 / (H ** 0.5)
    layers = []
    key = jax.random.PRNGKey(seed)
    for _ in range(cfg.num_hidden_layers):
        key, *ks = jax.random.split(key, 7)
        def rnd(k, shape):
            return (jax.random.normal(k, shape, jnp.float32) * s).astype(dtype)
        layers.append(MLPWeights(
            U_gate=rnd(ks[0], (I, r)), V_gate=rnd(ks[1], (r, H)),
            U_up=rnd(ks[2], (I, r)),   V_up=rnd(ks[3], (r, H)),
            U_down=rnd(ks[4], (H, r)), V_down=rnd(ks[5], (r, I)),
        ))
    return layers


def load_weights(path, cfg: LlamaMLPConfig, dtype):
    """Load SVD MLP weights saved by llama_inference.py::svd_flash.

    Parses the safetensors file directly (mmap, no full-file load) and upcasts
    via ml_dtypes -- this handles the checkpoint's bf16 tensors without needing
    torch or the safetensors library (NumPy alone has no bf16 dtype).
    """
    import json
    import mmap
    import struct
    import numpy as np
    import ml_dtypes

    st_dtype = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
                "BF16": ml_dtypes.bfloat16}

    layers = []
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

        def g(key):
            meta = header[key]
            lo, hi = meta["data_offsets"]
            arr = np.frombuffer(mm[base + lo: base + hi],
                                dtype=st_dtype[meta["dtype"]]).reshape(meta["shape"])
            return jnp.asarray(arr.astype(np.float32)).astype(dtype)

        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}.mlp"
            layers.append(MLPWeights(
                U_gate=g(f"{p}.gate_u_proj.weight"), V_gate=g(f"{p}.gate_v_proj.weight"),
                U_up=g(f"{p}.up_u_proj.weight"),     V_up=g(f"{p}.up_v_proj.weight"),
                U_down=g(f"{p}.down_u_proj.weight"), V_down=g(f"{p}.down_v_proj.weight"),
            ))
    return layers


# ---------------------------------------------------------------------------
# MLP layer evaluation
# ---------------------------------------------------------------------------
def mlp_kernel(x, w: MLPWeights, *, interpret, BM, BN, BK=None):
    return svd_swiglu_mlp(
        x, w.U_gate, w.V_gate, w.U_up, w.V_up, w.U_down, w.V_down,
        BM=BM, BN=BN, BK=BK, interpret=interpret,
    )


def mlp_reference(x, w: MLPWeights):
    # exactly SVD_LlamaMLP.forward: down_u(down_v( silu(gate_u(gate_v x)) * up_u(up_v x) ))
    return svd_swiglu_mlp_ref(x, w.U_gate, w.V_gate, w.U_up, w.V_up, w.U_down, w.V_down)


def run_stack(x, layers, *, use_kernel, interpret, BM, BN):
    """Chain every layer's MLP (for TIMING only).

    Note: real transformers interleave attention + norms + residuals between MLPs;
    chaining MLPs directly is fine as a throughput proxy but is NOT numerically
    meaningful (activations can explode), so we don't use it for correctness.
    """
    h = x
    for w in layers:
        h = mlp_kernel(h, w, interpret=interpret, BM=BM, BN=BN) if use_kernel \
            else mlp_reference(h, w)
    return h


def check_each_layer(x, layers, *, interpret, BM, BN, BK=None):
    """Correctness: run every layer's real weights through kernel vs reference
    on the SAME input (independently), and report the worst relative error."""
    worst = 0.0
    for i, w in enumerate(layers):
        out_k = jax.block_until_ready(mlp_kernel(x, w, interpret=interpret, BM=BM, BN=BN, BK=BK))
        out_r = jax.block_until_ready(mlp_reference(x, w))
        a, b = out_k.astype(jnp.float32), out_r.astype(jnp.float32)
        rel = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-9))
        worst = max(worst, rel)
    return worst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-path", type=str, default=None,
                    help="safetensors from svd_flash; if omitted, random weights are used")
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--intermediate-size", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--compress-ratio", type=float, default=0.8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=512)
    ap.add_argument("--bk", type=int, default=None, help="contraction block; set (e.g. 512) to stream U — required on TPU v4")
    ap.add_argument("--interpret", action="store_true",
                    help="run Pallas kernels on CPU (no TPU). Auto-on when backend != tpu.")
    args = ap.parse_args()

    on_tpu = jax.default_backend() == "tpu"
    interpret = args.interpret or not on_tpu
    dtype = jnp.bfloat16 if on_tpu else jnp.float32

    cfg = LlamaMLPConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        compress_ratio=args.compress_ratio,
    )
    print(f"backend={jax.default_backend()}  interpret={interpret}  dtype={dtype.__name__}")
    print(f"config: H={cfg.hidden_size} I={cfg.intermediate_size} layers={cfg.num_hidden_layers} "
          f"ratio={cfg.compress_ratio} -> low_rank={cfg.low_rank}")

    if args.weights_path:
        print(f"loading SVD weights from {args.weights_path}")
        layers = load_weights(args.weights_path, cfg, dtype)
    else:
        print("synthesizing random SVD weights (no --weights-path given)")
        layers = random_weights(cfg, dtype)

    x = (jax.random.normal(jax.random.PRNGKey(123), (args.seq_len, cfg.hidden_size), jnp.float32)
         ).astype(dtype)

    # 1) correctness: every layer's real weights, kernel vs reference, independently
    rel = check_each_layer(x, layers, interpret=interpret, BM=args.bm, BN=args.bn, BK=args.bk)
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-3
    status = "PASS" if rel < tol else "FAIL"
    print(f"\n[correctness] worst rel err over {cfg.num_hidden_layers} layers "
          f"(kernel vs reference) = {rel:.3e}  -> {status}")

    # 2) timing (only meaningful on TPU)
    if on_tpu:
        f = jax.jit(lambda x: run_stack(x, layers, use_kernel=True,
                                        interpret=False, BM=args.bm, BN=args.bn))
        for _ in range(3):
            jax.block_until_ready(f(x))
        t0 = time.perf_counter()
        for _ in range(20):
            o = f(x)
        jax.block_until_ready(o)
        print(f"[timing] {(time.perf_counter()-t0)/20*1e3:.3f} ms / MLP-stack pass")
    else:
        print("[timing] skipped (CPU/interpret timings are not representative)")

    print("\nDone.")


if __name__ == "__main__":
    main()
