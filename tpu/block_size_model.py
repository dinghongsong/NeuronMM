"""
Analytical block-size selector for the TPU SVD-MLP kernels.

Ports the NeuronMLP paper's performance model (Sec. 4.2.1, Eq. 10 & 11) to TPU.
Instead of brute-forcing BM/BN on hardware (what ``tuning_mlp_up.py`` does), this
computes them from two formulas plus the roofline ridge point:

  Eq. 10  Arithmetic Intensity  AI(BM) = 2r / ((1 + r/BM) * s)        [FLOPs/byte]
  Eq. 11  Peak on-chip SRAM     (BM*r + (BM + Br)*max(BK, BN)) * s    [bytes]

Logic:
  1. A kernel is compute-bound once AI exceeds the hardware ridge point. Solve
     Eq. 10 = ridge for the smallest BM that saturates the matmul unit.
  2. Grow BM (raises AI / reuse) and BN as large as Eq. 11 allows within VMEM.

  >>> IMPORTANT <<<  Two numbers are hardware-specific and MUST be set to the
  target TPU's real spec (left as required inputs, not guessed here):
     - TPUSpec.vmem_bytes            : usable on-chip VMEM per core
     - TPUSpec.bf16_ridge_flops_byte : roofline ridge point for bf16
  The paper's Trainium values (24 MB SBUF, 222 FLOPs/byte) are included ONLY as
  a reference point in PLACEHOLDER_TRAINIUM below -- replace before trusting the
  output for a TPU.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Hardware spec (fill in for the target TPU)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TPUSpec:
    name: str
    vmem_bytes: int                 # usable VMEM per core (bytes)
    bf16_ridge_flops_byte: float    # roofline ridge point for bf16 (FLOPs/byte)
    tile_m: int = 128               # MXU partition tile (rows)
    tile_n: int = 512               # output free-dim tile (moving operand width)
    vmem_headroom: float = 0.80     # fraction of VMEM we allow the kernel to use


# Reference point ONLY -- these are Trainium's numbers from the paper, NOT a TPU.
PLACEHOLDER_TRAINIUM = TPUSpec(
    name="PLACEHOLDER(Trainium-from-paper)",
    vmem_bytes=24 * 1024 * 1024,
    bf16_ridge_flops_byte=222.0,
)

# TPU v4 (the target). Sources: Jouppi et al. arXiv:2304.01433; Google Cloud TPU
# v4 docs; jax-ml scaling-book.
#   - VMEM = 16 MiB per TensorCore (the Pallas scratchpad; there is also 128 MiB
#     shared CMEM, not used here).
#   - peak bf16 = 275 TFLOPS/chip,  HBM ~1.2 TB/s  ->  ridge = 275e12/1.2e12 ~= 224.
TPU_V4 = TPUSpec(
    name="TPU v4",
    vmem_bytes=16 * 1024 * 1024,
    bf16_ridge_flops_byte=275e12 / 1.2e12,   # ~= 229 FLOPs/byte
)


# ---------------------------------------------------------------------------
# The two model equations
# ---------------------------------------------------------------------------
def arithmetic_intensity(r: int, BM: int, s: int) -> float:
    """Eq. 10: FLOPs per byte of HBM traffic for the XUV chain."""
    return (2.0 * r) / ((1.0 + r / BM) * s)


def _ceil_mult(x: float, m: int) -> int:
    import math
    return int(math.ceil(x / m) * m)


def min_bm_compute_bound(r: int, s: int, ridge: float, tile_m: int) -> Optional[int]:
    """Smallest BM (rounded up to tile_m) whose AI reaches the ridge point.

    Returns None if the chain is too skinny to ever be compute-bound
    (max achievable AI = 2r/s < ridge), which itself is a useful signal.
    """
    max_ai = (2.0 * r) / s                      # BM -> infinity
    if max_ai <= ridge:
        return None                             # cannot saturate the MXU at any BM
    # AI(BM) >= ridge  =>  BM >= r / (2r/(s*ridge) - 1)
    bm = r / ((2.0 * r) / (s * ridge) - 1.0)
    return _ceil_mult(bm, tile_m)


def estimate_peak_vmem_bytes(BM, BN, K, r, s, kernel="xuv", BK=None) -> int:
    """Peak VMEM for this repo's kernels.

    BK is None  -> FUSED kernel: U held whole (K,r), rank strip cached, V streamed.
    BK is set   -> K-BLOCKED path (two kernels): a strip kernel that streams U in
                   BK chunks, then an expand kernel. Peak = max of the two stages.
    """
    if BK is None:
        x_block = BM * K * s
        u_whole = K * r * s
        strip = BM * r * s
        v_block = 2 * (r * BN * s)              # double-buffered stream
        out_block = BM * BN * s
        out_accum_fp32 = BM * BN * 4
        per_set = strip + u_whole + v_block + out_accum_fp32
        sets = 2 if kernel == "upgate" else 1
        return x_block + out_block + sets * per_set

    # K-blocked: strip kernel (run once per projection; gate/up run sequentially)
    strip_peak = (2 * (BM * BK * s + BK * r * s)  # double-buffered x & U chunks
                  + BM * r * 4                     # fp32 accumulator
                  + BM * r * s)                    # strip output
    # expand kernel: rank strip(s) + streamed V block(s) + out + fp32 accum
    sets = 2 if kernel == "upgate" else 1
    expand_peak = (sets * BM * r * s
                   + sets * 2 * (r * BN * s)
                   + BM * BN * s
                   + sets * BM * BN * 4)
    return max(strip_peak, expand_peak)


@dataclass
class BlockChoice:
    BM: int
    BN: int
    BK: Optional[int]                 # None => fused (U whole); int => K-blocked
    arithmetic_intensity: float
    compute_bound: bool
    peak_vmem_bytes: int
    peak_vmem_mb: float
    path: str                         # "fused" or "k-blocked"
    note: str = ""


def pick_block_sizes(M, K, r, N, spec: TPUSpec, dtype_bytes=2, kernel="xuv") -> BlockChoice:
    """Choose (BM, BN) for one projection of shape  X(M,K) @ U(K,r) @ V(r,N).

    Strategy (paper Sec. 4.2.1): find the min BM that is compute-bound, then grow
    BM (higher AI) and BN as far as the VMEM budget allows.
    """
    s = dtype_bytes
    budget = int(spec.vmem_bytes * spec.vmem_headroom)

    bm_min = min_bm_compute_bound(r, s, spec.bf16_ridge_flops_byte, spec.tile_m)
    note = ""
    if bm_min is None:
        bm_min = spec.tile_m
        note = (f"rank r={r} too small to be compute-bound at bf16 "
                f"(max AI={2*r/s:.0f} < ridge={spec.bf16_ridge_flops_byte:.0f}); "
                f"kernel will be memory-bound — consider a higher compression ratio.")
    bm_min = min(bm_min, _ceil_mult(M, spec.tile_m))

    BM = min(bm_min, _ceil_mult(M, spec.tile_m)) if M >= spec.tile_m else M

    def max_bn(bk):
        bn = 0
        cand = spec.tile_n
        while cand <= max(N, spec.tile_n):
            if estimate_peak_vmem_bytes(BM, min(cand, N), K, r, s, kernel, BK=bk) <= budget:
                bn = min(cand, N)
            else:
                break
            cand += spec.tile_n
        return bn

    ai = arithmetic_intensity(r, BM, s)

    # 1) Prefer the FUSED kernel (U whole) -- best when it fits.
    bn = max_bn(None)
    if bn > 0:
        peak = estimate_peak_vmem_bytes(BM, bn, K, r, s, kernel, BK=None)
        return BlockChoice(BM=BM, BN=bn, BK=None, arithmetic_intensity=ai,
                           compute_bound=ai >= spec.bf16_ridge_flops_byte,
                           peak_vmem_bytes=peak, peak_vmem_mb=peak/1024/1024,
                           path="fused", note=note)

    # 2) Fall back to K-blocking: largest BK (mult of tile_m, <= K) whose strip
    #    kernel fits, then the widest BN for the expand kernel.
    bk = (K // spec.tile_m) * spec.tile_m or spec.tile_m
    while bk >= spec.tile_m:
        if estimate_peak_vmem_bytes(BM, spec.tile_n, K, r, s, kernel, BK=bk) <= budget:
            break
        bk -= spec.tile_m
    bn = max_bn(bk)
    if bn == 0:
        raise ValueError(
            f"Even K-blocked, no (BM,BN,BK) fits VMEM {budget/1e6:.1f}MB for "
            f"K={K}, r={r}. Reduce BM or raise headroom.")
    peak = estimate_peak_vmem_bytes(BM, bn, K, r, s, kernel, BK=bk)
    extra = (f" U whole = {K*r*s/1024/1024:.0f}MB > VMEM budget, so streaming U in "
             f"BK={bk} chunks.")
    return BlockChoice(BM=BM, BN=bn, BK=bk, arithmetic_intensity=ai,
                       compute_bound=ai >= spec.bf16_ridge_flops_byte,
                       peak_vmem_bytes=peak, peak_vmem_mb=peak/1024/1024,
                       path="k-blocked", note=(note + extra).strip())


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    spec = TPU_V4                   # the target (16 MiB VMEM, ridge ~229)
    print(f"Using spec: {spec.name}  "
          f"VMEM={spec.vmem_bytes/1024/1024:.0f}MiB  "
          f"ridge={spec.bf16_ridge_flops_byte:.0f} FLOPs/byte\n")

    # DeepSeek-V3 up_projection from the paper (Sec. 5.3): M=4096, K=7168, r=4096, N=18432
    shapes = [
        ("deepseek-up  (paper)", 4096, 7168, 4096, 18432, "upgate"),
        ("llama3-1b  up/gate",   128,  2048, 1280, 8192,  "upgate"),
        ("llama3-1b  down",      128,  8192, 1280, 2048,  "xuv"),
        ("llama3-8b  up/gate",   512,  4096, 2560, 14336, "upgate"),
    ]
    for name, M, K, r, N, kind in shapes:
        try:
            c = pick_block_sizes(M, K, r, N, spec, dtype_bytes=2, kernel=kind)
        except ValueError:
            print(f"{name:22s}  needs r-blocking too (rank r={r} too large for "
                  f"{spec.vmem_bytes//1024//1024}MiB VMEM at BN>=512) — kernel extension")
            continue
        bk = "whole" if c.BK is None else str(c.BK)
        print(f"{name:22s}  BM={c.BM:4d} BN={c.BN:5d} BK={bk:5s}  [{c.path:9s}]  "
              f"AI={c.arithmetic_intensity:6.0f} "
              f"{'compute-bound' if c.compute_bound else 'MEM-bound '}  "
              f"peak={c.peak_vmem_mb:4.1f}MB")
        if c.note:
            print(f"{'':22s}  note: {c.note}")
