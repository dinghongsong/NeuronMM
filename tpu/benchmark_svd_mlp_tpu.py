"""
Benchmark: SVD-compressed SwiGLU MLP (Pallas) vs the uncompressed dense MLP,
the TPU analog of the speedup test in ``NeuronMM/test_speedup.py``.

Meant to run on a TPU host:

    python NeuronMM/tpu/benchmark_svd_mlp_tpu.py

On CPU the timings are not meaningful (and Pallas runs via the slow interpret
path); use it there only to confirm the code executes end-to-end. The script
auto-detects: real kernels on TPU, interpret mode otherwise.
"""

import time

import jax
import jax.numpy as jnp

from svd_mlp_tpu import svd_swiglu_mlp, svd_swiglu_mlp_ref, dense_swiglu_mlp_ref
from block_size_model import TPU_V4, pick_block_sizes


ON_TPU = jax.default_backend() == "tpu"
DTYPE = jnp.bfloat16 if ON_TPU else jnp.float32


def _rand(key, shape, scale):
    return (jax.random.normal(key, shape, dtype=jnp.float32) * scale).astype(DTYPE)


def bench(fn, *args, iters=50, warmup=10):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main():
    # llama3-3b MLP shape (compress ratio ~0.8 -> r=1792), prefill of 128 tokens
    S, H, I, r, r_d = 128, 3072, 8192, 1792, 1792
    ks = jax.random.split(jax.random.PRNGKey(0), 10)
    s = 1.0 / (H ** 0.5)

    x = _rand(ks[0], (S, H), 1.0)
    U_gate, V_gate = _rand(ks[1], (I, r), s), _rand(ks[2], (r, H), s)
    U_up, V_up = _rand(ks[3], (I, r), s), _rand(ks[4], (r, H), s)
    U_down, V_down = _rand(ks[5], (H, r_d), s), _rand(ks[6], (r_d, I), s)

    # dense weights of the same nominal shape (the uncompressed baseline)
    W_gate = _rand(ks[7], (I, H), s)
    W_up = _rand(ks[8], (I, H), s)
    W_down = _rand(ks[9], (H, I), s)

    # On TPU the fused path (U held whole) overflows VMEM at this shape
    # (~30MB > 16MiB v4); ask the analytical model for the K-block size so the
    # kernel fits. interpret/CPU has no VMEM limit, so keep the fused path there.
    BK = None
    if ON_TPU:
        bks = [pick_block_sizes(S, H, r,   I, TPU_V4, kernel="upgate").BK,   # up/gate
               pick_block_sizes(S, I, r_d, H, TPU_V4, kernel="xuv").BK]      # down
        BK = min([b for b in bks if b], default=None)
        print(f"  [TPU] fused path exceeds VMEM at this shape; K-blocking with BK={BK}")

    dense = jax.jit(lambda x: dense_swiglu_mlp_ref(x, W_gate, W_up, W_down))
    svd_pallas = jax.jit(lambda x: svd_swiglu_mlp(
        x, U_gate, V_gate, U_up, V_up, U_down, V_down, BK=BK, interpret=not ON_TPU))
    # XLA's own fusion of the SAME SVD math -- the real bar to beat on TPU.
    svd_xla = jax.jit(lambda x: svd_swiglu_mlp_ref(
        x, U_gate, V_gate, U_up, V_up, U_down, V_down))

    print(f"backend={jax.default_backend()}  dtype={DTYPE.__name__}  "
          f"shape S={S} H={H} I={I} r={r}")
    t_dense = bench(dense, x)
    t_xla = bench(svd_xla, x)
    t_pallas = bench(svd_pallas, x)
    print(f"  dense MLP (no SVD)   : {t_dense:8.4f} ms/iter")
    print(f"  svd  MLP  (XLA)      : {t_xla:8.4f} ms/iter   ({t_dense/t_xla:5.2f}x vs dense)")
    print(f"  svd  MLP  (Pallas)   : {t_pallas:8.4f} ms/iter   ({t_dense/t_pallas:5.2f}x vs dense)")
    print(f"  --> Pallas vs XLA    : {t_xla / t_pallas:6.3f}x   "
          f"({'Pallas wins' if t_pallas < t_xla else 'XLA already wins -- kernel not worth it here'})")
    if not ON_TPU:
        print("  (CPU/interpret timings are NOT representative -- run on TPU.)")


if __name__ == "__main__":
    main()
