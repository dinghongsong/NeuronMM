"""
TPU / Pallas implementation of NeuronMLP (NeuronMM / SVD-Flash):
SVD-compressed SwiGLU MLP for LLM inference, ported from the AWS-Trainium NKI
kernels to TPU.

Faithful to the paper "NeuronMLP: Efficient LLM Inference via Singular Value
Decomposition Compression and Tiling on AWS Trainium" (Song et al., arXiv
2510.25977), in particular Algorithm 2 (FusedMLP), Algorithm 3 (UpGateProjection),
and the TrainiumFusion techniques in Sec. 4.2.

The algorithm
-------------
Each MLP weight ``W (out, in)`` is approximated offline as ``W ~= U @ V`` with
``U: (out, r)``, ``V: (r, in)`` (block-aligned SVD, paper Eq. 5). A linear layer
``y = x @ W.T`` then becomes the three-matrix chain the paper calls ``X U V``:

    y = x @ W.T = x @ (U @ V).T = (x @ V.T) @ U.T
        x:(S,in) --[contract in]--> (x @ V.T):(S,r) --[contract r]--> (S,out)

The intermediate ``(S, r)`` is the *rank strip*. A naive two-kernel split would
write it to HBM and read it back; for low compression ratios r is large, so that
traffic dominates. The paper's three techniques avoid it:

  1. Caching (Sec 4.2.1): compute a row strip of the rank intermediate ``(B_M, r)``
     once and keep it in on-chip SRAM, reusing it for every output-column block.
  2. Implicit transposition: a Trainium-only trick to satisfy the systolic
     array's "stationary must be transposed" rule -- NOT needed on TPU, where
     jnp.dot / Mosaic handle operand layout. (One thing that gets *simpler*.)
  3. Blocking: tile rows (B_M) and output columns (B_N).

This file realizes technique (1) directly: a single Pallas kernel per projection
computes the rank strip into a persistent VMEM scratch (the SBUF-cache analog)
guarded by ``pl.when(col_block == 0)``, then reuses it for all output-column
blocks of the same row strip. The strip never touches HBM.

Trainium -> TPU mapping
-----------------------
    SBUF (24 MB, 128 partitions)              ->  VMEM (incl. the scratch cache)
    PSUM (2 MB) + nc_matmul + reduction loop  ->  jnp.dot (Mosaic drives the MXU)
    capacity-aware SBUF caching (Sec 4.2.1)   ->  persistent VMEM scratch + pl.when
    implicit transposition (Sec 4.2.1)        ->  unnecessary (jnp.dot any layout)
    SiLU/multiply on Scalar+Vector engines    ->  jax.nn.silu / *  (VPU)
    bf16 multiply / fp32 accumulate           ->  preferred_element_type=jnp.float32
    block-size model Eq. 10/11 + roofline     ->  BM / BN here (re-tune for VMEM)

CPU testing: pass ``interpret=True`` to run the kernel logic on CPU (no TPU).
On a TPU host the default ``interpret=False`` lowers to real TPU kernels.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ceil_div(a, b):
    return (a + b - 1) // b


def _pad_to(x, axis, multiple):
    n = x.shape[axis]
    pad = _ceil_div(n, multiple) * multiple - n
    if pad == 0:
        return x
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, pad)
    return jnp.pad(x, widths)


# ---------------------------------------------------------------------------
# Faithful fused XUV kernel  (paper "XUV_Kernel" / down-projection, Alg. 2)
#   O = (X @ U) @ V        X:(M,K)  U:(K,r)  V:(r,N)  ->  O:(M,N)
# Grid (M/BM, N/BN). The rank strip XU:(BM, r) is computed once per row block
# (when the column-block index == 0) into a persistent VMEM scratch -- the
# paper's on-chip caching -- and reused for every column block. Never to HBM.
# ---------------------------------------------------------------------------
def _xuv_kernel(x_ref, U_ref, V_ref, o_ref, xu_strip):
    @pl.when(pl.program_id(1) == 0)
    def _compute_strip():
        xu = jnp.dot(x_ref[...], U_ref[...], preferred_element_type=jnp.float32)
        xu_strip[...] = xu.astype(xu_strip.dtype)            # cache (BM, r) on-chip

    o = jnp.dot(xu_strip[...], V_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = o.astype(o_ref.dtype)


def _xuv(X, U, V, BM, BN, interpret):
    M, K = X.shape
    K_u, r = U.shape
    r_v, N = V.shape
    assert K == K_u and r == r_v
    gs = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(M // BM, N // BN),
        in_specs=[
            pl.BlockSpec(index_map=lambda m, n: (m, 0), block_shape=(BM, K)),  # X row block
            pl.BlockSpec(index_map=lambda m, n: (0, 0), block_shape=(K, r)),   # U (whole)
            pl.BlockSpec(index_map=lambda m, n: (0, n), block_shape=(r, BN)),  # V col block
        ],
        out_specs=pl.BlockSpec(index_map=lambda m, n: (m, n), block_shape=(BM, BN)),
        scratch_shapes=[pltpu.VMEM((BM, r), X.dtype)],
    )
    return pl.pallas_call(
        _xuv_kernel, grid_spec=gs,
        out_shape=jax.ShapeDtypeStruct((M, N), X.dtype),
        interpret=interpret, name="svd_xuv",
    )(X, U, V)


# ---------------------------------------------------------------------------
# Faithful fused Up+Gate kernel  (paper Algorithm 3, UpGateProjection)
#   O = silu((X @ Ug) @ Vg) * ((X @ Uu) @ Vu)
#   X:(M,K)  Ug,Uu:(K,r)  Vg,Vu:(r,N)  ->  O:(M,N)
# Two rank strips (gate, up) cached on-chip; SwiGLU fused before the store, so
# the gate/up activations (M,N) and the rank strips never touch HBM.
# ---------------------------------------------------------------------------
def _upgate_kernel(x_ref, Ug_ref, Vg_ref, Uu_ref, Vu_ref, o_ref, g_strip, u_strip):
    @pl.when(pl.program_id(1) == 0)
    def _compute_strips():
        g = jnp.dot(x_ref[...], Ug_ref[...], preferred_element_type=jnp.float32)
        u = jnp.dot(x_ref[...], Uu_ref[...], preferred_element_type=jnp.float32)
        g_strip[...] = g.astype(g_strip.dtype)
        u_strip[...] = u.astype(u_strip.dtype)

    gate = jnp.dot(g_strip[...], Vg_ref[...], preferred_element_type=jnp.float32)
    up = jnp.dot(u_strip[...], Vu_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = (jax.nn.silu(gate) * up).astype(o_ref.dtype)


def _upgate(X, Ug, Vg, Uu, Vu, BM, BN, interpret):
    M, K = X.shape
    _, r = Ug.shape
    _, N = Vg.shape
    spec_x = pl.BlockSpec(index_map=lambda m, n: (m, 0), block_shape=(BM, K))
    spec_U = pl.BlockSpec(index_map=lambda m, n: (0, 0), block_shape=(K, r))
    spec_V = pl.BlockSpec(index_map=lambda m, n: (0, n), block_shape=(r, BN))
    gs = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(M // BM, N // BN),
        in_specs=[spec_x, spec_U, spec_V, spec_U, spec_V],
        out_specs=pl.BlockSpec(index_map=lambda m, n: (m, n), block_shape=(BM, BN)),
        scratch_shapes=[pltpu.VMEM((BM, r), X.dtype), pltpu.VMEM((BM, r), X.dtype)],
    )
    return pl.pallas_call(
        _upgate_kernel, grid_spec=gs,
        out_shape=jax.ShapeDtypeStruct((M, N), X.dtype),
        interpret=interpret, name="svd_upgate",
    )(X, Ug, Vg, Uu, Vu)


# ===========================================================================
# K-BLOCKED PATH (for when whole U exceeds VMEM -- required on TPU v4, 16 MiB)
# ---------------------------------------------------------------------------
# When U (K, r) does not fit in VMEM, stream it in BK-sized chunks of the
# contraction dim K (paper Alg. 3, the inner k-loop, lines 7-11): the rank strip
# XU (M, r) is accumulated across K-chunks in a VMEM scratch, then a second
# kernel expands it (XU @ V) and fuses SwiGLU. The strip is small (M, r), so the
# split costs only a tiny HBM round-trip -- far cheaper than spilling the big
# gate/up activations. With BK >= K this is identical to the fused path above.
# ===========================================================================
def _xu_strip_kernel(x_ref, U_ref, o_ref, acc):
    @pl.when(pl.program_id(1) == 0)
    def _zero():
        acc[...] = jnp.zeros_like(acc)
    acc[...] += jnp.dot(x_ref[...], U_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = acc[...].astype(o_ref.dtype)            # last k-step = final XU


def _xu_strip(X, U, BM, BK, interpret):
    """XU = X @ U  with the contraction K streamed in BK chunks.  -> (M, r)."""
    M, K = X.shape
    K_u, r = U.shape
    assert K == K_u
    gs = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(M // BM, K // BK),
        in_specs=[
            pl.BlockSpec(index_map=lambda m, k: (m, k), block_shape=(BM, BK)),
            pl.BlockSpec(index_map=lambda m, k: (k, 0), block_shape=(BK, r)),
        ],
        out_specs=pl.BlockSpec(index_map=lambda m, k: (m, 0), block_shape=(BM, r)),
        scratch_shapes=[pltpu.VMEM((BM, r), jnp.float32)],
    )
    return pl.pallas_call(
        _xu_strip_kernel, grid_spec=gs,
        out_shape=jax.ShapeDtypeStruct((M, r), X.dtype),
        interpret=interpret, name="svd_xu_strip",
    )(X, U)


def _expand_kernel(a_ref, V_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], V_ref[...],
                         preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _expand(A, V, BM, BN, interpret):
    """O = A @ V  (contraction = small rank, kept whole).  -> (M, N)."""
    M, r = A.shape
    return pl.pallas_call(
        _expand_kernel,
        grid=(M // BM, V.shape[1] // BN),
        in_specs=[
            pl.BlockSpec(index_map=lambda m, n: (m, 0), block_shape=(BM, r)),
            pl.BlockSpec(index_map=lambda m, n: (0, n), block_shape=(r, BN)),
        ],
        out_specs=pl.BlockSpec(index_map=lambda m, n: (m, n), block_shape=(BM, BN)),
        out_shape=jax.ShapeDtypeStruct((M, V.shape[1]), A.dtype),
        interpret=interpret, name="svd_expand",
    )(A, V)


def _expand_swiglu_kernel(g_ref, Vg_ref, u_ref, Vu_ref, o_ref):
    gate = jnp.dot(g_ref[...], Vg_ref[...], preferred_element_type=jnp.float32)
    up = jnp.dot(u_ref[...], Vu_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = (jax.nn.silu(gate) * up).astype(o_ref.dtype)


def _expand_swiglu(g, Vg, u, Vu, BM, BN, interpret):
    """silu(g @ Vg) * (u @ Vu)  (contraction = small rank).  -> (M, N)."""
    M, r = g.shape
    N = Vg.shape[1]
    sm = pl.BlockSpec(index_map=lambda m, n: (m, 0), block_shape=(BM, r))
    sn = pl.BlockSpec(index_map=lambda m, n: (0, n), block_shape=(r, BN))
    return pl.pallas_call(
        _expand_swiglu_kernel,
        grid=(M // BM, N // BN),
        in_specs=[sm, sn, sm, sn],
        out_specs=pl.BlockSpec(index_map=lambda m, n: (m, n), block_shape=(BM, BN)),
        out_shape=jax.ShapeDtypeStruct((M, N), g.dtype),
        interpret=interpret, name="svd_expand_swiglu",
    )(g, Vg, u, Vu)


# ---------------------------------------------------------------------------
# Public API: SVD-compressed SwiGLU MLP  (paper Algorithm 2, FusedMLP)
# ---------------------------------------------------------------------------
def svd_swiglu_mlp(
    x,                      # (S, H)
    U_gate, V_gate,         # gate: U (I, r),   V (r, H)
    U_up, V_up,             # up:   U (I, r),   V (r, H)
    U_down, V_down,         # down: U (H, r_d), V (r_d, I)
    *,
    BM=128,                 # row (sequence) block   -- paper's B_M (Eq. 10/11)
    BN=512,                 # output-column block    -- paper's B_N
    BK=None,                # contraction block. None => hold U whole (fits big VMEM);
                            # set it (e.g. 512) to stream U -- REQUIRED on TPU v4 (16 MiB).
    interpret=False,        # True => run on CPU (no TPU)
):
    """SVD-compressed SwiGLU MLP (paper Algorithm 2): UpGate then Down projection.

    Each weight is given low-rank factored, ``W ~= U @ V``; we express each
    projection as ``y = x @ V.T @ U.T`` (the paper's "X U V" chain). With BK=None
    the rank strip is held in VMEM and U whole (best, but needs VMEM >= K*r*s).
    With BK set, U is streamed in BK chunks of the contraction dim and the rank
    strip accumulated -- the only viable path when U exceeds VMEM (TPU v4).
    """
    S, H = x.shape
    I, r = U_gate.shape
    H_d, r_d = U_down.shape
    assert V_gate.shape == (r, H) and V_up.shape == (r, H) and U_up.shape == (I, r)
    assert V_down.shape == (r_d, I) and H_d == H

    x_p = _pad_to(x, 0, BM)                                 # (S_pad, H)

    if BK is None:
        # ---- Fused path: U held whole, rank strip cached in VMEM (Alg. 3) ----
        h_p = _upgate(
            x_p,
            V_gate.T, _pad_to(U_gate.T, 1, BN),
            V_up.T,   _pad_to(U_up.T, 1, BN),
            BM, BN, interpret,
        )
        out_p = _xuv(
            h_p,
            _pad_to(V_down.T, 0, BN),
            _pad_to(U_down.T, 1, BN),
            BM, BN, interpret,
        )
        return out_p[:S, :H]

    # ---- K-blocked path: stream U in BK chunks; accumulate the rank strip ----
    # Up + Gate.  contraction K = H; rank-strip then SwiGLU-expand over I.
    xk = _pad_to(x_p, 1, BK)                                # (S_pad, H_pk)
    g_strip = _xu_strip(xk, _pad_to(V_gate.T, 0, BK), BM, BK, interpret)   # (S_pad, r)
    u_strip = _xu_strip(xk, _pad_to(V_up.T, 0, BK),  BM, BK, interpret)
    h_p = _expand_swiglu(
        g_strip, _pad_to(U_gate.T, 1, BN),
        u_strip, _pad_to(U_up.T, 1, BN),
        BM, BN, interpret,
    )                                                       # (S_pad, I_pad)

    # Down.  contraction K = I; rank-strip then plain expand over H.
    hk = _pad_to(h_p, 1, BK)                                # (S_pad, I_pk)
    d_strip = _xu_strip(hk, _pad_to(V_down.T, 0, BK), BM, BK, interpret)   # (S_pad, r_d)
    out_p = _expand(d_strip, _pad_to(U_down.T, 1, BN), BM, BN, interpret)  # (S_pad, H_pad)
    return out_p[:S, :H]


# ---------------------------------------------------------------------------
# Plain-JAX reference (ground truth) and dense baseline (speedup reference)
# ---------------------------------------------------------------------------
def _svd_proj_ref(x, U, V):
    return (x @ V.T) @ U.T                                  # y = x @ W.T, W ~= U@V


def svd_swiglu_mlp_ref(x, U_gate, V_gate, U_up, V_up, U_down, V_down):
    gate = _svd_proj_ref(x, U_gate, V_gate)
    up = _svd_proj_ref(x, U_up, V_up)
    h = jax.nn.silu(gate) * up
    return _svd_proj_ref(h, U_down, V_down)


def dense_swiglu_mlp_ref(x, W_gate, W_up, W_down):
    """Uncompressed SwiGLU MLP -- the speedup baseline (paper's NKI XW)."""
    gate = x @ W_gate.T
    up = x @ W_up.T
    h = jax.nn.silu(gate) * up
    return h @ W_down.T
