"""
Correctness checks for the TPU/Pallas SVD-SwiGLU MLP.

Dual-mode, auto-detecting the backend (same detection as
``benchmark_svd_mlp_tpu.py``):

  * No TPU present  -> Pallas ``interpret=True``: the kernel logic runs on CPU,
    so the math is validated without any TPU. This is the laptop/CI path.
  * TPU present     -> ``interpret=False``: the REAL Mosaic kernels compile and
    run on the TPU, and their output is asserted against the plain-JAX
    reference. This is the path that was previously missing -- the kernels had
    only ever been validated in interpret mode.

Run the same command in both places:

    python NeuronMM/tpu/test_svd_mlp_tpu.py

On TPU we keep fp32 inputs and force ``jax_default_matmul_precision=highest`` so
the comparison isolates kernel/tiling correctness from MXU rounding; the bf16
*performance* path is exercised separately by the benchmark. A few shapes have a
rank ``r`` that is not a 128-multiple (to test host-side padding); Mosaic can't
tile those, so they stay interpret-only and SKIP on TPU with a printed reason.
"""

import jax
import jax.numpy as jnp

from svd_mlp_tpu import (
    svd_swiglu_mlp,
    svd_swiglu_mlp_ref,
    dense_swiglu_mlp_ref,
)
from block_size_model import TPU_V4, pick_block_sizes

# --- backend detection: real kernels on TPU, interpret everywhere else --------
ON_TPU = jax.default_backend() == "tpu"
INTERPRET = not ON_TPU
if ON_TPU:
    # fp32 matmul (3-pass) so the kernel-vs-reference diff reflects the algorithm,
    # not the MXU's default bf16 inputs. Both the kernels and the jnp reference
    # honor this global setting.
    jax.config.update("jax_default_matmul_precision", "highest")

# fp32 keeps the reference comparison tight on both paths.
DTYPE = jnp.float32
# A real algorithmic bug shows up as an order-1 (or larger) discrepancy; these
# tolerances absorb residual fp32-MXU rounding on TPU while still catching that.
ATOL = 2e-2 if ON_TPU else 1e-3
RTOL = 2e-2 if ON_TPU else 1e-3

MODE = f"TPU/real-kernels (matmul=highest)" if ON_TPU else "CPU/interpret"
print(f"backend={jax.default_backend()}  mode={MODE}  "
      f"atol={ATOL:g} rtol={RTOL:g}\n")


def _rand(key, shape, scale=1.0):
    return (jax.random.normal(key, shape, dtype=jnp.float32) * scale).astype(DTYPE)


def make_inputs(S, H, I, r, r_d, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 7)
    # scale weights down so activations stay in a sane range for the fp32 compare
    s = 1.0 / (H ** 0.5)
    x = _rand(ks[0], (S, H))
    U_gate = _rand(ks[1], (I, r), s)
    V_gate = _rand(ks[2], (r, H), s)
    U_up = _rand(ks[3], (I, r), s)
    V_up = _rand(ks[4], (r, H), s)
    U_down = _rand(ks[5], (H, r_d), s)
    V_down = _rand(ks[6], (r_d, I), s)
    return x, U_gate, V_gate, U_up, V_up, U_down, V_down


def _report(tag, ref, out):
    """Shared assert + print so every case prints its actual error."""
    assert out.shape == ref.shape, (out.shape, ref.shape)
    err = float(jnp.max(jnp.abs(out - ref)))
    rel = err / float(jnp.max(jnp.abs(ref)) + 1e-9)
    ok = bool(jnp.allclose(out, ref, atol=ATOL, rtol=RTOL))
    print(f"[{tag}]  max|err|={err:.2e}  rel={rel:.2e}  -> {'PASS' if ok else 'FAIL'}")
    assert ok, f"{tag}: Pallas output diverged from reference"


def _tpu_bk(M, K, r, N, kernel):
    """fp32 K-block size that fits TPU v4 VMEM for one projection, via the
    analytical model. ``or 512`` guards the (not-expected-here) case where the
    model reports the fused path fits -- we still want to exercise K-blocking."""
    return pick_block_sizes(M, K, r, N, TPU_V4, dtype_bytes=4, kernel=kernel).BK or 512


def test_matches_reference(shape_name, S, H, I, r, r_d):
    """Fused path (BK=None): U held whole, rank strip cached in VMEM."""
    x, U_gate, V_gate, U_up, V_up, U_down, V_down = make_inputs(S, H, I, r, r_d)
    ref = svd_swiglu_mlp_ref(x, U_gate, V_gate, U_up, V_up, U_down, V_down)
    out = svd_swiglu_mlp(
        x, U_gate, V_gate, U_up, V_up, U_down, V_down,
        BM=128, BN=512, interpret=INTERPRET,
    )
    _report(f"fused:{shape_name} S={S} H={H} I={I} r={r}", ref, out)


def test_k_blocked_matches_reference(shape_name, S, H, I, r, r_d, BK):
    """The K-blocked path (BK set): streams U in BK chunks and accumulates the
    rank strip. This is the path that actually runs on TPU at large shapes,
    where holding U whole overflows VMEM."""
    x, U_gate, V_gate, U_up, V_up, U_down, V_down = make_inputs(S, H, I, r, r_d)
    ref = svd_swiglu_mlp_ref(x, U_gate, V_gate, U_up, V_up, U_down, V_down)
    out = svd_swiglu_mlp(
        x, U_gate, V_gate, U_up, V_up, U_down, V_down,
        BM=128, BN=512, BK=BK, interpret=INTERPRET,
    )
    _report(f"k-blocked:{shape_name} BK={BK} (down K={I}%BK={I % BK} exercises K-pad)",
            ref, out)


def test_unaligned_shapes():
    """S, I, H not multiples of the block sizes -> exercises host-side padding.
    r=96 is not a 128-multiple, so Mosaic can't tile it: interpret-only."""
    if ON_TPU:
        print("[unaligned] SKIP on TPU (r=96 not a 128-multiple -> Mosaic tiling); "
              "interpret-only padding check")
        return
    S, H, I, r, r_d = 100, 384, 1536, 96, 80
    x, U_gate, V_gate, U_up, V_up, U_down, V_down = make_inputs(S, H, I, r, r_d, seed=1)
    ref = svd_swiglu_mlp_ref(x, U_gate, V_gate, U_up, V_up, U_down, V_down)
    out = svd_swiglu_mlp(
        x, U_gate, V_gate, U_up, V_up, U_down, V_down,
        BM=64, BN=256, interpret=INTERPRET,
    )
    _report(f"unaligned S={S} H={H} I={I}", ref, out)


def test_compression_is_an_approximation():
    """Sanity: the SVD path reproduces a dense MLP built from W = U @ V.
    r=128 (a 128-multiple) so this runs on TPU too."""
    S, H, I, r, r_d = 128, 256, 512, 128, 128
    x, U_gate, V_gate, U_up, V_up, U_down, V_down = make_inputs(S, H, I, r, r_d, seed=2)
    # Build the *exact* dense weights these factors represent, then check the
    # compressed path reproduces the dense MLP on those weights.
    W_gate = U_gate @ V_gate
    W_up = U_up @ V_up
    W_down = U_down @ V_down
    dense = dense_swiglu_mlp_ref(x, W_gate, W_up, W_down)
    out = svd_swiglu_mlp(
        x, U_gate, V_gate, U_up, V_up, U_down, V_down,
        BM=128, BN=256, interpret=INTERPRET,
    )
    _report("vs-dense reconstructs W=U@V dense MLP", dense, out)


if __name__ == "__main__":
    # llama3-3b MLP shape from test_speedup.py (compress ratio ~0.8 -> r=1792).
    LL = dict(S=128, H=3072, I=8192, r=1792, r_d=1792)

    # ---- fused path (U held whole) ------------------------------------------
    # Small shape fits VMEM in fp32, so it runs the real fused kernel on TPU.
    test_matches_reference("small", S=256, H=512, I=2048, r=256, r_d=256)
    # llama3-3b held-whole overflows TPU v4 VMEM (~21MB U alone > 13MB budget);
    # run the fused kernel only in interpret mode -- the shape is covered on TPU
    # by the K-blocked case below (which is what production uses).
    if INTERPRET:
        test_matches_reference("llama3-3b", **LL)
    else:
        print("[fused:llama3-3b] SKIP on TPU (U whole overflows VMEM); "
              "covered by k-blocked below")

    # ---- K-blocked path (U streamed) ----------------------------------------
    # The path that runs on real TPU at large shapes. On TPU pick BK from the
    # analytical model (fp32) so the strip kernel fits VMEM; on CPU/interpret
    # there's no VMEM limit, so use 1536 (matches the bf16 model pick on v4).
    if ON_TPU:
        bk_ll = min(_tpu_bk(LL["S"], LL["H"], LL["r"], LL["I"], "upgate"),   # up/gate
                    _tpu_bk(LL["S"], LL["I"], LL["r_d"], LL["H"], "xuv"))    # down
        print(f"  [TPU] K-blocking llama3-3b with model-picked BK={bk_ll} (fp32)")
    else:
        bk_ll = 1536
    test_k_blocked_matches_reference("small", S=256, H=512, I=2048, r=256, r_d=256, BK=256)
    test_k_blocked_matches_reference("llama3-3b", BK=bk_ll, **LL)

    # ---- padding / sanity (interpret-only where r is not a 128-multiple) -----
    test_unaligned_shapes()
    test_compression_is_an_approximation()

    where = "real TPU kernels" if ON_TPU else "CPU interpret mode"
    print(f"\nAll checks passed ({where}).")
