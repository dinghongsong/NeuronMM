"""
colab_svd_mlp.py — self-contained, Colab-ready SVD-compressed SwiGLU MLP on TPU.

A single-file port of NeuronMM/SVD-Flash (Song et al., arXiv:2510.25977,
Algorithm 2/3) tuned to run on a **free Colab/Kaggle TPU** — which is *not* a v4.
Colab currently hands out a single-chip **TPU v5e-1** (sometimes a **v2-8**);
Kaggle gives a **v3-8**. The existing `svd_mlp_tpu.py` hardcodes the v4 spec, so
the block sizes are wrong on those cores. This file fixes that with three things:

  1. AUTO-SPEC      — reads `jax.devices()[0].device_kind` and builds the
                      analytical block-size model's spec for whatever core you got.
  2. AUTO-FALLBACK  — tries the fast fused path first; if Mosaic overflows VMEM it
                      automatically streams the weights (K-blocking) and shrinks the
                      tile until it compiles. You never have to hand-pick `BK`.
  3. ONE FILE       — kernels + block model + correctness + benchmark, no local
                      imports. Upload just this file, or paste it into one cell.

Run (identical command on a laptop or a TPU; it auto-detects the backend):

    python colab_svd_mlp.py            # correctness, then benchmark if on TPU
    python colab_svd_mlp.py --bench    # benchmark only
    python colab_svd_mlp.py --shape llama-3b
    python colab_svd_mlp.py --check    # correctness only (runs on CPU too)

On CPU there is no TPU, so Pallas runs in `interpret=True` (the kernel *logic* is
validated, timings are meaningless). On a TPU host the real Mosaic kernels compile.

The math (per MLP weight  W ≈ U @ V,  U:(out,r) V:(r,in), rank r ≪ out,in):

    gate = (x @ V_gate.T) @ U_gate.T ;  up = (x @ V_up.T) @ U_up.T
    h    = silu(gate) * up
    out  = (h @ V_down.T) @ U_down.T

Each projection's rank strip `(BM, r)` is cached in on-chip VMEM and reused across
output-column blocks (the paper's on-chip caching), so it never spills to HBM.
"""

import argparse
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# ===========================================================================
# Hardware spec — auto-detected from the live TPU (no hardcoded v4)
# ===========================================================================
class TPUSpec:
    """Per-core spec used by the block-size model.

    `vmem_bytes` and `ridge` (roofline ridge point, peak-bf16-FLOPs / HBM-bytes/s)
    are the only hardware numbers the model needs. VMEM per generation is only
    approximately documented, so it is treated as a *budget hint*: the runtime
    autotuner (see `svd_swiglu_mlp_auto`) shrinks tiles until the kernel actually
    fits, so an imperfect VMEM guess costs at most one wasted compile, never a
    wrong/over-allocating run.
    """

    def __init__(self, name, vmem_bytes, peak_tf, peak_gb, tile_m=128, tile_n=512,
                 headroom=0.75, bm_cap=512):
        self.name = name
        self.vmem_bytes = vmem_bytes
        self.peak_tf = peak_tf             # peak bf16 compute, TFLOP/s (roofline)
        self.peak_gb = peak_gb             # peak HBM bandwidth, GB/s   (roofline)
        self.ridge = (peak_tf * 1e12) / (peak_gb * 1e9)   # FLOPs/byte (ridge point)
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.headroom = headroom
        self.bm_cap = bm_cap               # largest row block the ladder tries —
        #   keeps the matrix unit fed. Bigger MXU wants bigger BM: v5e=512, v6e=1024.

    def __repr__(self):
        return (f"TPUSpec({self.name}, VMEM~{self.vmem_bytes/2**20:.0f}MiB, "
                f"peak {self.peak_tf:.0f}TF/s {self.peak_gb:.0f}GB/s, "
                f"ridge~{self.ridge:.0f} FLOPs/byte)")


# Best-effort per-core specs keyed by a substring of `device_kind`. This is the
# SINGLE per-architecture table for the whole project — the benchmark suite reads
# `spec.peak_tf` / `spec.peak_gb` from here instead of re-deriving roofline peaks.
#   peak_tf = peak bf16 compute (TFLOP/s);  peak_gb = peak HBM bandwidth (GB/s).
#   ridge   = peak_tf / peak_gb (FLOPs/byte), computed by TPUSpec.
#   vmem    = approximate usable VMEM (conservative; the autotuner corrects it).
# Sources: Google Cloud TPU system-architecture docs; Jouppi et al. (v4,
# arXiv:2304.01433); the JAX "How to Scale Your Model" book.
_MiB = 2 ** 20
_KNOWN_SPECS = [
    # (match substring, name,    usable-vmem,  peak_tf, peak_gb, bm_cap)
    ("v2",       "TPU v2",   16 * _MiB,   46.0,    700.0,  512),   # ridge ~66
    ("v3",       "TPU v3",   16 * _MiB,  123.0,    900.0,  512),   # ~137
    ("v4",       "TPU v4",   16 * _MiB,  275.0,   1200.0,  512),   # ~229
    # v5e VMEM calibrated from real runs: holding a whole U (~29 MB at llama-3b)
    # overflowed and BK=1536 was the largest streaming chunk that fit -> usable
    # budget ~15 MB (≈20 MiB before headroom). Mosaic double-buffers every operand,
    # so the effective scratchpad is far below the nominal VMEM. BM=512 measured best.
    ("v5 lite",  "TPU v5e",  20 * _MiB,  197.0,    819.0,  512),   # ~241
    ("v5e",      "TPU v5e",  20 * _MiB,  197.0,    819.0,  512),
    # v6e (Trillium): usable VMEM is ~32 MiB like other gens (the 128 MiB nominal is
    # shared CMEM, NOT the Pallas scratchpad — the fused 29 MB-U path OOMs here too).
    # Its larger matrix unit wants a BIGGER row block: measured BM=1024 -> 1.08x XLA
    # at S=2048 (vs 0.74x at BM=512), so cap BM at 1024 on v6e.
    ("v6 lite",  "TPU v6e",  32 * _MiB,  918.0,   1640.0, 1024),
    ("v6e",      "TPU v6e",  32 * _MiB,  918.0,   1640.0, 1024),
    ("v5",       "TPU v5p",  64 * _MiB,  459.0,   2765.0,  512),   # ~166 (after v5e)
]
# Safe fallback if the kind string is unrecognized: small VMEM forces the
# always-correct K-blocked path; mid-range peaks keep the BM heuristic sane.
_DEFAULT_SPEC = TPUSpec("TPU (unknown)", 16 * _MiB, 197.0, 819.0)


def detect_spec(verbose=True):
    """Build a `TPUSpec` for the live accelerator (or the safe default on CPU)."""
    if jax.default_backend() != "tpu":
        if verbose:
            print("  [spec] no TPU — using conservative default (interpret mode)")
        return _DEFAULT_SPEC
    kind = jax.devices()[0].device_kind
    low = kind.lower()
    for sub, name, vmem, peak_tf, peak_gb, bm_cap in _KNOWN_SPECS:
        if sub in low:
            spec = TPUSpec(name, vmem, peak_tf, peak_gb, bm_cap=bm_cap)
            if verbose:
                print(f"  [spec] device_kind={kind!r} -> {spec}")
            return spec
    if verbose:
        print(f"  [spec] unrecognized device_kind={kind!r} -> {_DEFAULT_SPEC}")
    return _DEFAULT_SPEC


# ===========================================================================
# Analytical block-size model (paper Sec. 4.2.1, Eq. 10 & 11) — first guess
# ===========================================================================
def _ceil_div(a, b):
    return (a + b - 1) // b


def _ceil_mult(x, m):
    return int(_ceil_div(int(x + 0.999999), m) * m)


def _arith_intensity(r, BM, s):
    """Eq. 10: FLOPs per HBM byte for the X·U·V chain."""
    return (2.0 * r) / ((1.0 + r / BM) * s)


def _min_bm_compute_bound(r, s, ridge, tile_m):
    """Smallest BM whose arithmetic intensity reaches the ridge (or None if the
    rank is too small to ever saturate the MXU — a memory-bound signal)."""
    if (2.0 * r) / s <= ridge:
        return None
    bm = r / ((2.0 * r) / (s * ridge) - 1.0)
    return _ceil_mult(bm, tile_m)


def _peak_vmem(BM, BN, K, r, s, kernel, BK):
    """Approx peak VMEM (bytes) for this repo's kernels at a given tiling."""
    if BK is None:                                   # fused: U held whole
        per_set = BM * r * s + K * r * s + 2 * (r * BN * s) + BM * BN * 4
        sets = 2 if kernel == "upgate" else 1
        return BM * K * s + BM * BN * s + sets * per_set
    strip = 2 * (BM * BK * s + BK * r * s) + BM * r * 4 + BM * r * s
    sets = 2 if kernel == "upgate" else 1
    expand = sets * BM * r * s + sets * 2 * (r * BN * s) + BM * BN * s + sets * BM * BN * 4
    return max(strip, expand)


def plan_blocks(M, K, r, N, spec, s, kernel):
    """Return (BM, BN, BK, info) for one projection X(M,K)·U(K,r)·V(r,N).

    Strategy: pick the smallest compute-bound BM, then the widest BN (and, if the
    fused path overflows, the largest BK) that fits the VMEM budget."""
    budget = int(spec.vmem_bytes * spec.headroom)
    bm = _min_bm_compute_bound(r, s, spec.ridge, spec.tile_m)
    mem_bound = bm is None
    if bm is None:
        bm = spec.tile_m
    BM = min(_ceil_mult(bm, spec.tile_m), _ceil_mult(M, spec.tile_m)) if M >= spec.tile_m else M

    def widest_bn(bk):
        best = 0
        cand = spec.tile_n
        while cand <= max(N, spec.tile_n):
            bn = min(cand, N)
            if _peak_vmem(BM, bn, K, r, s, kernel, bk) <= budget:
                best = bn
            else:
                break
            cand += spec.tile_n
        return best

    ai = _arith_intensity(r, BM, s)
    info = dict(ai=ai, compute_bound=ai >= spec.ridge, mem_bound=mem_bound)

    bn = widest_bn(None)                              # 1) prefer fused (U whole)
    if bn > 0:
        info["path"] = "fused"
        return BM, bn, None, info

    bk = (K // spec.tile_m) * spec.tile_m or spec.tile_m   # 2) K-block
    while bk >= spec.tile_m:
        if _peak_vmem(BM, spec.tile_n, K, r, s, kernel, bk) <= budget:
            break
        bk -= spec.tile_m
    bn = widest_bn(bk) or spec.tile_n
    info["path"] = "k-blocked"
    return BM, bn, bk, info


# ===========================================================================
# Pallas plumbing
# ===========================================================================
def _pad_to(x, axis, multiple):
    n = x.shape[axis]
    pad = _ceil_mult(n, multiple) - n
    if pad == 0:
        return x
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, pad)
    return jnp.pad(x, widths)


# `CompilerParams` (current) vs `TPUCompilerParams` (older JAX) vs neither.
_CP = getattr(pltpu, "CompilerParams", None) or getattr(pltpu, "TPUCompilerParams", None)


def _cp(dimension_semantics):
    """Pipelining hint kwargs, feature-detected so it is a no-op on old/odd JAX."""
    if _CP is None:
        return {}
    try:
        return {"compiler_params": _CP(dimension_semantics=dimension_semantics)}
    except Exception:
        return {}


# --- fused XUV (down projection): O = (X @ U) @ V --------------------------
def _xuv_kernel(x_ref, U_ref, V_ref, o_ref, xu_strip):
    @pl.when(pl.program_id(1) == 0)
    def _():
        xu = jnp.dot(x_ref[...], U_ref[...], preferred_element_type=jnp.float32)
        xu_strip[...] = xu.astype(xu_strip.dtype)          # cache rank strip on-chip
    o = jnp.dot(xu_strip[...], V_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = o.astype(o_ref.dtype)


def _xuv(X, U, V, BM, BN, interpret):
    M, K = X.shape
    _, r = U.shape
    N = V.shape[1]
    return pl.pallas_call(
        _xuv_kernel,
        grid=(M // BM, N // BN),
        in_specs=[
            pl.BlockSpec((BM, K), lambda m, n: (m, 0)),
            pl.BlockSpec((K, r), lambda m, n: (0, 0)),
            pl.BlockSpec((r, BN), lambda m, n: (0, n)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda m, n: (m, n)),
        out_shape=jax.ShapeDtypeStruct((M, N), X.dtype),
        scratch_shapes=[pltpu.VMEM((BM, r), X.dtype)],
        interpret=interpret, name="svd_xuv",
        **_cp(("parallel", "arbitrary")),
    )(X, U, V)


# --- fused Up+Gate: O = silu((X@Ug)@Vg) * ((X@Uu)@Vu) ----------------------
def _upgate_kernel(x_ref, Ug_ref, Vg_ref, Uu_ref, Vu_ref, o_ref, g_strip, u_strip):
    @pl.when(pl.program_id(1) == 0)
    def _():
        g_strip[...] = jnp.dot(x_ref[...], Ug_ref[...],
                               preferred_element_type=jnp.float32).astype(g_strip.dtype)
        u_strip[...] = jnp.dot(x_ref[...], Uu_ref[...],
                               preferred_element_type=jnp.float32).astype(u_strip.dtype)
    gate = jnp.dot(g_strip[...], Vg_ref[...], preferred_element_type=jnp.float32)
    up = jnp.dot(u_strip[...], Vu_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = (jax.nn.silu(gate) * up).astype(o_ref.dtype)


def _upgate(X, Ug, Vg, Uu, Vu, BM, BN, interpret):
    M, K = X.shape
    _, r = Ug.shape
    N = Vg.shape[1]
    sx = pl.BlockSpec((BM, K), lambda m, n: (m, 0))
    sU = pl.BlockSpec((K, r), lambda m, n: (0, 0))
    sV = pl.BlockSpec((r, BN), lambda m, n: (0, n))
    return pl.pallas_call(
        _upgate_kernel,
        grid=(M // BM, N // BN),
        in_specs=[sx, sU, sV, sU, sV],
        out_specs=pl.BlockSpec((BM, BN), lambda m, n: (m, n)),
        out_shape=jax.ShapeDtypeStruct((M, N), X.dtype),
        scratch_shapes=[pltpu.VMEM((BM, r), X.dtype), pltpu.VMEM((BM, r), X.dtype)],
        interpret=interpret, name="svd_upgate",
        **_cp(("parallel", "arbitrary")),
    )(X, Ug, Vg, Uu, Vu)


# --- K-blocked path: stream U in BK chunks, accumulate the rank strip -------
def _xu_strip_kernel(x_ref, U_ref, o_ref, acc):
    @pl.when(pl.program_id(1) == 0)
    def _():
        acc[...] = jnp.zeros_like(acc)
    acc[...] += jnp.dot(x_ref[...], U_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = acc[...].astype(o_ref.dtype)


def _xu_strip(X, U, BM, BK, interpret):
    M, K = X.shape
    r = U.shape[1]
    return pl.pallas_call(
        _xu_strip_kernel,
        grid=(M // BM, K // BK),
        in_specs=[
            pl.BlockSpec((BM, BK), lambda m, k: (m, k)),
            pl.BlockSpec((BK, r), lambda m, k: (k, 0)),
        ],
        out_specs=pl.BlockSpec((BM, r), lambda m, k: (m, 0)),
        out_shape=jax.ShapeDtypeStruct((M, r), X.dtype),
        scratch_shapes=[pltpu.VMEM((BM, r), jnp.float32)],
        interpret=interpret, name="svd_xu_strip",
        **_cp(("parallel", "arbitrary")),
    )(X, U)


def _expand_kernel(a_ref, V_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], V_ref[...],
                         preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _expand(A, V, BM, BN, interpret):
    M, r = A.shape
    N = V.shape[1]
    return pl.pallas_call(
        _expand_kernel,
        grid=(M // BM, N // BN),
        in_specs=[pl.BlockSpec((BM, r), lambda m, n: (m, 0)),
                  pl.BlockSpec((r, BN), lambda m, n: (0, n))],
        out_specs=pl.BlockSpec((BM, BN), lambda m, n: (m, n)),
        out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
        interpret=interpret, name="svd_expand",
        **_cp(("parallel", "arbitrary")),
    )(A, V)


def _expand_swiglu_kernel(g_ref, Vg_ref, u_ref, Vu_ref, o_ref):
    gate = jnp.dot(g_ref[...], Vg_ref[...], preferred_element_type=jnp.float32)
    up = jnp.dot(u_ref[...], Vu_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = (jax.nn.silu(gate) * up).astype(o_ref.dtype)


def _expand_swiglu(g, Vg, u, Vu, BM, BN, interpret):
    M, r = g.shape
    N = Vg.shape[1]
    sm = pl.BlockSpec((BM, r), lambda m, n: (m, 0))
    sn = pl.BlockSpec((r, BN), lambda m, n: (0, n))
    return pl.pallas_call(
        _expand_swiglu_kernel,
        grid=(M // BM, N // BN),
        in_specs=[sm, sn, sm, sn],
        out_specs=pl.BlockSpec((BM, BN), lambda m, n: (m, n)),
        out_shape=jax.ShapeDtypeStruct((M, N), g.dtype),
        interpret=interpret, name="svd_expand_swiglu",
        **_cp(("parallel", "arbitrary")),
    )(g, Vg, u, Vu)


# ===========================================================================
# Public kernel API
# ===========================================================================
def svd_swiglu_mlp(x, U_gate, V_gate, U_up, V_up, U_down, V_down,
                   *, BM=128, BN=512, BK=None, interpret=False):
    """SVD-compressed SwiGLU MLP (paper Algorithm 2). BK=None holds each U whole in
    VMEM (fastest where it fits); set BK to stream U in contraction chunks."""
    S, H = x.shape
    I, r = U_gate.shape
    H_d, r_d = U_down.shape
    assert V_gate.shape == (r, H) and V_up.shape == (r, H) and U_up.shape == (I, r)
    assert V_down.shape == (r_d, I) and H_d == H

    x_p = _pad_to(x, 0, BM)
    if BK is None:
        h_p = _upgate(x_p, V_gate.T, _pad_to(U_gate.T, 1, BN),
                      V_up.T, _pad_to(U_up.T, 1, BN), BM, BN, interpret)
        out_p = _xuv(h_p, _pad_to(V_down.T, 0, BN), _pad_to(U_down.T, 1, BN),
                     BM, BN, interpret)
        return out_p[:S, :H]

    # K-blocked: up/gate (K=H), then down (K=I)
    xk = _pad_to(x_p, 1, BK)
    g = _xu_strip(xk, _pad_to(V_gate.T, 0, BK), BM, BK, interpret)
    u = _xu_strip(xk, _pad_to(V_up.T, 0, BK), BM, BK, interpret)
    h_p = _expand_swiglu(g, _pad_to(U_gate.T, 1, BN), u, _pad_to(U_up.T, 1, BN),
                         BM, BN, interpret)
    hk = _pad_to(h_p, 1, BK)
    d = _xu_strip(hk, _pad_to(V_down.T, 0, BK), BM, BK, interpret)
    out_p = _expand(d, _pad_to(U_down.T, 1, BN), BM, BN, interpret)
    return out_p[:S, :H]


def _looks_like_vmem_error(e):
    """True only for VMEM-*capacity* errors, which a smaller tile can fix. Hard
    Mosaic compile errors (e.g. 'Bad lhs type' from a precision/dtype mismatch)
    must NOT match — retrying with smaller blocks can't help and only thrashes."""
    s = f"{type(e).__name__}: {e}".lower()
    if "bad lhs type" in s or "bad rhs type" in s:
        return False
    return any(k in s for k in ("vmem", "resource_exhausted", "out of memory",
                                "not enough memory"))


def _ladder(S, K_max, spec, try_fused):
    """Configs to try, best→safest. **Large BM first.** A 128×128 matrix unit is
    starved by a small BM: measured on v5e (llama-3b, S=2048) the analytical
    model's min-compute-bound BM=128 hits ~53% utilization (0.55× XLA), whereas
    BM=512 hits ~94% (0.97× XLA). BM is capped at 512 — 640+ measured *slower*
    (VMEM pressure forces worse tiling). A modest BK frees VMEM for the bigger BM
    (the dominant lever); BN=256 is tried before 512 to stay off the VMEM edge.
    The caller's runtime VMEM fallback catches anything that still overflows."""
    tm = spec.tile_m
    cap = getattr(spec, "bm_cap", 512)
    S_r = _ceil_mult(S, tm)
    bms = [bm for bm in (1024, 768, 512, 384, 256, 128) if bm <= S_r and bm <= cap] or [min(S, tm)]
    bns = [256, 512]
    bks = [bk for bk in (512, 896, 256, 1280, 1792, 3072) if bk <= K_max]
    out, seen = [], set()
    def add(c):
        if c not in seen:
            seen.add(c)
            out.append(c)
    # K-blocked LARGE-BM first — it is the usual winner because a big row block
    # keeps the matrix unit fed. The fused (U-whole) path is only tried as a
    # FALLBACK: when it fits at all it is often only at a small BM (VMEM), which
    # starves the MXU and loses to k-blocked-large-BM (measured on llama-1b/v6e).
    for bm in bms:
        for bk in bks:
            for bn in bns:
                add((bm, bn, bk))
    if try_fused:
        for bm in bms:
            for bn in bns:
                add((bm, bn, None))
    return out


def svd_swiglu_mlp_auto(x, U_gate, V_gate, U_up, V_up, U_down, V_down,
                        *, spec=None, interpret=False, verbose=True):
    """Pick a tiling from the analytical model for the detected TPU, then run —
    automatically falling back to K-blocking / smaller tiles if Mosaic reports a
    VMEM overflow. Returns (output, chosen_config_dict)."""
    if spec is None:
        spec = detect_spec(verbose=verbose)
    S, H = x.shape
    I, r = U_gate.shape
    _, r_d = U_down.shape
    s = jnp.dtype(x.dtype).itemsize

    # Diagnostics only: the analytical model's arithmetic-intensity read. The paper
    # model under-sizes BM (it picks the *min* compute-bound BM), which starves the
    # 128×128 matrix unit; the ladder overrides it toward large BM. Measured on v5e
    # (llama-3b, S=2048): model's BM=128 -> 0.55× XLA; BM=512 -> 0.97× XLA.
    _, _, _, i1 = plan_blocks(S, H, r, I, spec, s, "upgate")
    K_max = max(H, I)
    # Try the fused (U-whole) path first only when it actually fits the budget.
    try_fused = (max(H, I) * max(r, r_d) * s) <= spec.vmem_bytes * spec.headroom
    if verbose:
        print(f"  [plan] AI(up/gate)={i1['ai']:.0f} "
              f"{'compute-bound' if i1['compute_bound'] else 'MEM-bound'}; "
              f"autotuner prefers large BM (cap 512) for MXU occupancy")

    last = None
    for (bm, bn, bk) in _ladder(S, K_max, spec, try_fused):
        try:
            out = svd_swiglu_mlp(x, U_gate, V_gate, U_up, V_up, U_down, V_down,
                                 BM=bm, BN=bn, BK=bk, interpret=interpret)
            jax.block_until_ready(out)          # force compile+exec to surface VMEM errors
            cfg = dict(BM=bm, BN=bn, BK=bk,
                       path="fused" if bk is None else "k-blocked")
            if verbose:
                print(f"  [auto] using {cfg}")
            return out, cfg
        except Exception as e:                  # noqa: BLE001 — autotune retry
            if not _looks_like_vmem_error(e):
                raise
            last = e
            if verbose:
                print(f"  [auto] BM={bm} BN={bn} BK={bk} overflowed "
                      f"({type(e).__name__}); shrinking…")
    raise RuntimeError(
        f"No tiling fit VMEM on {spec.name}. Last error: {last}")


# ===========================================================================
# Plain-JAX references (ground truth + dense baseline)
# ===========================================================================
def svd_swiglu_mlp_ref(x, U_gate, V_gate, U_up, V_up, U_down, V_down):
    gate = (x @ V_gate.T) @ U_gate.T
    up = (x @ V_up.T) @ U_up.T
    h = jax.nn.silu(gate) * up
    return (h @ V_down.T) @ U_down.T


def dense_swiglu_mlp_ref(x, W_gate, W_up, W_down):
    gate = x @ W_gate.T
    up = x @ W_up.T
    h = jax.nn.silu(gate) * up
    return h @ W_down.T


# ===========================================================================
# Shapes, inputs, correctness, benchmark
# ===========================================================================
SHAPES = {
    "small":    dict(S=256, H=512,  I=2048,  r=256,  r_d=256),
    "llama-1b": dict(S=128, H=2048, I=8192,  r=1280, r_d=1280),   # Llama-3.2-1B MLP
    "llama-3b": dict(S=128, H=3072, I=8192,  r=1792, r_d=1792),   # Llama-3.2-3B MLP
    "llama-8b": dict(S=128, H=4096, I=14336, r=2560, r_d=2560),   # Llama-3-8B MLP
    "deepseek-v3": dict(S=128, H=7168, I=18432, r=4096, r_d=4096), # DeepSeek-V3 MLP (paper Sec 5.3)
}

ON_TPU = jax.default_backend() == "tpu"


def _make_factors(S, H, I, r, r_d, dtype, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 10)
    sc = 1.0 / (H ** 0.5)
    rnd = lambda k, shp, s=1.0: (jax.random.normal(k, shp, jnp.float32) * s).astype(dtype)
    x = rnd(ks[0], (S, H))
    f = (rnd(ks[1], (I, r), sc), rnd(ks[2], (r, H), sc),     # gate U,V
         rnd(ks[3], (I, r), sc), rnd(ks[4], (r, H), sc),     # up   U,V
         rnd(ks[5], (H, r_d), sc), rnd(ks[6], (r_d, I), sc)) # down U,V
    dense = (rnd(ks[7], (I, H), sc), rnd(ks[8], (I, H), sc), rnd(ks[9], (H, I), sc))
    return x, f, dense


def run_correctness(shape="llama-3b"):
    """Assert the Pallas output matches the plain-JAX reference (fp32, tight)."""
    print(f"\n=== correctness ({'real TPU kernels' if ON_TPU else 'CPU interpret'}) ===")
    # fp32 'highest' isolates tiling correctness from MXU rounding for the compare,
    # but it is a GLOBAL flag — restore it in `finally` so a later bf16 benchmark in
    # the same session isn't asked for fp32-precision matmuls on bf16 ('Bad lhs type').
    if ON_TPU:
        jax.config.update("jax_default_matmul_precision", "highest")
    try:
        spec = detect_spec()
        atol = rtol = 2e-2 if ON_TPU else 1e-3
        cases = ["small", shape] if shape != "small" else ["small"]
        for name in cases:
            d = SHAPES[name]
            x, f, _ = _make_factors(**d, dtype=jnp.float32)
            ref = svd_swiglu_mlp_ref(x, *f)
            out, cfg = svd_swiglu_mlp_auto(x, *f, spec=spec, interpret=not ON_TPU,
                                           verbose=False)
            err = float(jnp.max(jnp.abs(out - ref)))
            rel = err / (float(jnp.max(jnp.abs(ref))) + 1e-9)
            ok = bool(jnp.allclose(out, ref, atol=atol, rtol=rtol))
            print(f"  [{name:9s}] {cfg['path']:9s} BN={cfg['BN']} BK={cfg['BK']}  "
                  f"max|err|={err:.2e} rel={rel:.2e} -> {'PASS' if ok else 'FAIL'}")
            assert ok, f"{name}: Pallas diverged from reference"
        print("  all correctness checks passed.")
    finally:
        if ON_TPU:
            jax.config.update("jax_default_matmul_precision", "default")


def _bench(fn, *args, iters=50, warmup=10):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def run_benchmark(shape="llama-3b"):
    """Dense vs XLA-fused-SVD vs Pallas-SVD. The Pallas-vs-XLA ratio is the result."""
    d = SHAPES[shape]
    dtype = jnp.bfloat16 if ON_TPU else jnp.float32
    # bf16 wants the fast (default) MXU precision; force it in case a prior
    # correctness run left the global flag on 'highest' (-> 'Bad lhs type' on bf16).
    if ON_TPU:
        jax.config.update("jax_default_matmul_precision", "default")
    spec = detect_spec()
    x, f, (W_gate, W_up, W_down) = _make_factors(**d, dtype=dtype)

    print(f"\n=== benchmark: {shape}  S={d['S']} H={d['H']} I={d['I']} r={d['r']}  "
          f"dtype={dtype.__name__}  backend={jax.default_backend()} ===")

    # Resolve a working Pallas config once (eager autotune), then jit it.
    _, cfg = svd_swiglu_mlp_auto(x, *f, spec=spec, interpret=not ON_TPU)
    dense = jax.jit(lambda x: dense_swiglu_mlp_ref(x, W_gate, W_up, W_down))
    svd_xla = jax.jit(lambda x: svd_swiglu_mlp_ref(x, *f))
    svd_pallas = jax.jit(lambda x: svd_swiglu_mlp(
        x, *f, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], interpret=not ON_TPU))

    it = 50 if ON_TPU else 3
    t_dense = _bench(dense, x, iters=it)
    t_xla = _bench(svd_xla, x, iters=it)
    t_pallas = _bench(svd_pallas, x, iters=it)
    print(f"  dense MLP (no SVD)   : {t_dense:8.4f} ms/iter")
    print(f"  svd  MLP  (XLA)      : {t_xla:8.4f} ms/iter   ({t_dense/t_xla:5.2f}x vs dense)")
    print(f"  svd  MLP  (Pallas)   : {t_pallas:8.4f} ms/iter   ({t_dense/t_pallas:5.2f}x vs dense)")
    verdict = "Pallas wins" if t_pallas < t_xla else "XLA already wins — kernel not worth it here"
    print(f"  --> Pallas vs XLA    : {t_xla/t_pallas:6.3f}x   ({verdict})")
    if not ON_TPU:
        print("  (CPU/interpret timings are NOT representative — run on a TPU.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="llama-3b", choices=list(SHAPES),
                    help="MLP shape to use (default: llama-3b)")
    ap.add_argument("--check", action="store_true", help="correctness only")
    ap.add_argument("--bench", action="store_true", help="benchmark only")
    args = ap.parse_args()

    print(f"jax {jax.__version__}  backend={jax.default_backend()}  "
          f"devices={jax.device_count()}")
    do_check = args.check or not args.bench
    do_bench = args.bench or (not args.check and ON_TPU)
    if do_check:
        run_correctness(args.shape)
    if do_bench:
        run_benchmark(args.shape)
    elif not args.check and not ON_TPU:
        print("\n(no TPU: skipped benchmark — run on a Colab/Kaggle TPU for timings.)")


if __name__ == "__main__":
    main()
