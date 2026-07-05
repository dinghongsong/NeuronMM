"""
benchmarks.py — architecture-aware benchmark suite for the TPU SVD-MLP kernel.

Consolidates the per-stage experiment scripts (profile / tiling sweep / autotuner
check / decode / fused-vs-K-blocked / end-to-end layer / multi-shape / paper
table) into ONE CLI. Every experiment auto-detects the live accelerator
(v5e / v6e / v5p / v4 / ...) and pulls the roofline peak + tile caps from
`colab_svd_mlp.detect_spec()`, so the same command reports correct %-of-roofline
numbers on any chip — that is the whole point of the suite.

    python benchmarks.py <experiment> [--shape S ...] [--seq N ...]

Experiments:
    profile     whole-path roofline vs S + per-kernel breakdown      (stage 0)
    tiling      BM/BN/BK sweep at one S; best config + correctness    (stage 1)
    autotune    confirm the default autotuner's picks across S        (stage A)
    decode      decode regime S=1..32, bandwidth/dispatch analysis    (stage B)
    fused       fused (2-launch) vs K-blocked prefill                 (stage C)
    layer       full transformer layer (attention + SVD-MLP)          (stage D)
    table       paper main table: best-tuned Pallas vs XLA vs dense
    multishape  default autotuner generalization across all shapes
    all         the standard reporting set: table + decode + layer

On CPU there is no TPU, so Pallas runs in interpret mode (logic only — timings are
meaningless and iteration counts are dropped to 1). Run on a Colab/Kaggle TPU host
for real numbers. All paths pin matmul precision identically (bf16 in / fp32 acc).
"""

import argparse
import itertools
import math
import time

import jax
import jax.numpy as jnp

import colab_svd_mlp as m

jax.config.update("jax_default_matmul_precision", "default")  # identical for ALL paths

ON_TPU = jax.default_backend() == "tpu"
INTERP = not ON_TPU
SB = 2                                    # bf16 bytes
SPEC = m.detect_spec(verbose=False)
PEAK_TF, PEAK_GB = SPEC.peak_tf, SPEC.peak_gb     # single source of roofline truth
CAP = SPEC.bm_cap

# Per-shape transformer metadata (layers, attention heads) for the end-to-end layer
# experiment; the MLP shape (H, I, r, r_d) itself comes from colab_svd_mlp.SHAPES.
LAYER_META = {"llama-1b": (16, 32), "llama-3b": (28, 24), "llama-8b": (32, 32)}


# ===========================================================================
# Shared helpers
# ===========================================================================
def device_label():
    return jax.devices()[0].device_kind if ON_TPU else "cpu (interpret)"


def shape_dims(name):
    d = m.SHAPES[name]
    return d["H"], d["I"], d["r"], d["r_d"]


def flops(S, H, I, r, r_d):
    """Exact SVD-MLP FLOPs: (x·Vᵀ)·Uᵀ for gate+up (rank r) and down (rank r_d)."""
    return 2 * (2 * S * H * r + 2 * S * r * I) + (2 * S * I * r_d + 2 * S * r_d * H)


def cm(x, mult):
    return int(math.ceil(x / mult) * mult)


def bench_s(fn, *args, iters=50, warmup=12, serial=False):
    """Seconds/call. serial=True times per-call (latency); else async-dispatch
    (throughput). Iteration counts collapse to 1 in interpret mode."""
    if INTERP:
        iters, warmup = 1, 0
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    if serial:
        t = time.perf_counter()
        for _ in range(iters):
            jax.block_until_ready(fn(*args))
        return (time.perf_counter() - t) / iters
    t = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / iters


def bench_ms(fn, *args, **kw):
    return bench_s(fn, *args, **kw) * 1e3


def util_pct(ms, S, H, I, r, r_d):
    """Achieved % of the compute roofline (FLOPs are exact -> faithful MXU util)."""
    return 100 * flops(S, H, I, r, r_d) / (ms * 1e-3) / 1e12 / PEAK_TF


def rel_to_fp32(out, ref):
    return float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)) /
                 (jnp.max(jnp.abs(ref)) + 1e-9))


def correctness(H, I, r, r_d, BM, BN, BK, S=512):
    """rel error of a bf16 Pallas config vs an fp32 reference at sequence S."""
    xb, fb, _ = m._make_factors(S=S, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
    ref = m.svd_swiglu_mlp_ref(xb.astype(jnp.float32), *[t.astype(jnp.float32) for t in fb])
    out = m.svd_swiglu_mlp(xb, *fb, BM=BM, BN=BN, BK=BK, interpret=INTERP)
    return rel_to_fp32(out, ref)


def header():
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  device={device_label()}")
    print(f"spec: {SPEC}  bm_cap={CAP}  peak={PEAK_TF:.0f}TF/s {PEAK_GB:.0f}GB/s\n")


def jx(fn):
    return jax.jit(fn)


# ===========================================================================
# stage 0 — whole-path roofline vs S, then per-kernel breakdown
# ===========================================================================
def cmd_profile(args):
    name = (args.shape or ["llama-3b"])[0]
    H, I, r, r_d = shape_dims(name)
    Svals = args.seq or [128, 512, 2048, 4096]

    # Autotune the tiling ONCE (VMEM peak depends on BM, not S, so reuse it).
    x0, f, (Wg, Wu, Wd) = m._make_factors(S=128, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
    Ug, Vg, Uu, Vu, Ud, Vd = f
    _, cfg = m.svd_swiglu_mlp_auto(x0, *f, spec=SPEC, interpret=INTERP, verbose=False)
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    BKb = BK if BK is not None else 1024
    svd_w = sum(t.size for t in f) * SB
    print(f"### profile {name}  H={H} I={I} r={r}  cfg={cfg} ###\n")

    print("=== whole-path vs S (ms; Pallas achieved TF/s & GB/s) ===")
    print(f"{'S':>5} {'dense':>8} {'XLA':>8} {'Pallas':>8} {'P/XLA':>7} {'P/dns':>7}"
          f" | {'P TF/s':>7} {'%pk':>5} {'P GB/s':>7} {'%pk':>5}")
    for S in Svals:
        x = jax.random.normal(jax.random.PRNGKey(S), (S, H), jnp.float32).astype(jnp.bfloat16)
        td = bench_ms(jx(lambda x: m.dense_swiglu_mlp_ref(x, Wg, Wu, Wd)), x, iters=200, warmup=30)
        tx = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x, iters=200, warmup=30)
        tp = bench_ms(jx(lambda x: m.svd_swiglu_mlp(x, *f, BM=BM, BN=BN, BK=BK, interpret=INTERP)),
                      x, iters=200, warmup=30)
        tf = flops(S, H, I, r, r_d) / (tp * 1e-3) / 1e12
        gb = (svd_w + 2 * S * H * SB) / (tp * 1e-3) / 1e9
        print(f"{S:>5} {td:>8.4f} {tx:>8.4f} {tp:>8.4f} {tx/tp:>7.3f} {td/tp:>7.3f}"
              f" | {tf:>7.1f} {100*tf/PEAK_TF:>4.1f}% {gb:>7.1f} {100*gb/PEAK_GB:>4.1f}%")

    # ---- per-kernel breakdown at the worst (largest) S -------------------------
    S = max(Svals)
    print(f"\n=== per-kernel breakdown @ S={S} (TF/s exact; GB/s approx) ===")
    x = jax.random.normal(jax.random.PRNGKey(7), (S, H), jnp.float32).astype(jnp.bfloat16)
    xk = m._pad_to(m._pad_to(x, 0, BM), 1, BKb)
    VgT, VuT = m._pad_to(Vg.T, 0, BKb), m._pad_to(Vu.T, 0, BKb)
    UgT, UuT = m._pad_to(Ug.T, 1, BN), m._pad_to(Uu.T, 1, BN)
    VdT, UdT = m._pad_to(Vd.T, 0, BKb), m._pad_to(Ud.T, 1, BN)
    k_strip = jx(lambda a, b: m._xu_strip(a, b, BM, BKb, INTERP))
    k_exsg = jx(lambda g, Ug, u, Uu: m._expand_swiglu(g, Ug, u, Uu, BM, BN, INTERP))
    k_ex = jx(lambda dd, Ud: m._expand(dd, Ud, BM, BN, INTERP))
    g = k_strip(xk, VgT); u = k_strip(xk, VuT)
    hp = k_exsg(g, UgT, u, UuT); hk = m._pad_to(hp, 1, BKb)
    dd = k_strip(hk, VdT)
    by = lambda *a: sum(t.size for t in a) * SB

    def row(tag, ms, fl, byts):
        tf, gb = fl / (ms * 1e-3) / 1e12, byts / (ms * 1e-3) / 1e9
        print(f"  {tag:18s}{ms:8.4f} ms  {tf:6.1f} TF/s ({100*tf/PEAK_TF:4.1f}%)"
              f"  {gb:6.1f} GB/s ({100*gb/PEAK_GB:4.1f}%)")
        return ms

    t1 = row("1 strip gate", bench_ms(k_strip, xk, VgT, iters=200, warmup=30), 2*xk.shape[0]*xk.shape[1]*r, by(xk, VgT, g))
    t2 = row("2 strip up", bench_ms(k_strip, xk, VuT, iters=200, warmup=30), 2*xk.shape[0]*xk.shape[1]*r, by(xk, VuT, u))
    t3 = row("3 expand+swiglu", bench_ms(k_exsg, g, UgT, u, UuT, iters=200, warmup=30), 2*(2*g.shape[0]*r*UgT.shape[1]), by(g, UgT, u, UuT, hp))
    t4 = row("4 strip down", bench_ms(k_strip, hk, VdT, iters=200, warmup=30), 2*hk.shape[0]*hk.shape[1]*r_d, by(hk, VdT, dd))
    t5 = row("5 expand down", bench_ms(k_ex, dd, UdT, iters=200, warmup=30), 2*dd.shape[0]*r_d*UdT.shape[1], by(dd, UdT))
    whole = bench_ms(jx(lambda x: m.svd_swiglu_mlp(x, *f, BM=BM, BN=BN, BK=BK, interpret=INTERP)),
                     x, iters=200, warmup=30)
    print(f"\n  sum of 5 kernels = {t1+t2+t3+t4+t5:.4f} ms   vs whole-path jit = {whole:.4f} ms"
          f"  (gap = isolated-dispatch overhead; the fused path is the truth)")


# ===========================================================================
# stage 1 — BM/BN/BK tiling sweep: tuning vs structural? + best-config S-sweep
# ===========================================================================
def cmd_tiling(args):
    name = (args.shape or ["llama-3b"])[0]
    H, I, r, r_d = shape_dims(name)
    S = (args.seq or [2048])[0]
    BMs = [bm for bm in (256, 512, 1024) if bm <= CAP] or [128]
    BNs, BKs = [256, 512], [512, 1024, 2048, 3072]
    if INTERP:
        BMs, BNs, BKs = [128], [512], [1792]

    x, f, _ = m._make_factors(S=S, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
    xla = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x, iters=100, warmup=20)
    print(f"### tiling {name}  S={S} ###")
    print(f"XLA = {xla:.4f} ms  ({util_pct(xla, S, H, I, r, r_d):.0f}% roofline)  <- bar to beat\n")

    res = []
    for BM, BN, BK in itertools.product(BMs, BNs, BKs):
        try:
            t = bench_ms(jx(lambda x, a=BM, b=BN, c=BK: m.svd_swiglu_mlp(
                x, *f, BM=a, BN=b, BK=c, interpret=INTERP)), x, iters=100, warmup=20)
            res.append((t, BM, BN, BK))
            flag = "  <-- BEATS XLA" if t < xla else ""
            print(f"  BM{BM:4d} BN{BN:4d} BK{BK:4d}: {t:7.4f} ms  {util_pct(t, S, H, I, r, r_d):3.0f}%"
                  f"  ({xla/t:.2f}x XLA){flag}")
        except Exception:                       # noqa: BLE001 — VMEM overflow on this tile
            print(f"  BM{BM:4d} BN{BN:4d} BK{BK:4d}: SKIP (VMEM)")
    if not res:
        print("  (no config fit VMEM)"); return
    res.sort()
    print("\n--- top 3 ---")
    for t, BM, BN, BK in res[:3]:
        print(f"  BM{BM} BN{BN} BK{BK}: {t:.4f} ms  {util_pct(t, S, H, I, r, r_d):.0f}%  ({xla/t:.2f}x XLA)")

    # correctness of the winner, then the best config across S
    _, bBM, bBN, bBK = res[0]
    rel = correctness(H, I, r, r_d, bBM, bBN, bBK, S=min(512, cm(S, 128)))
    print(f"\n  winner BM{bBM} BN{bBN} BK{bBK} correctness: rel={rel:.2e} -> "
          f"{'PASS' if rel < 3e-2 else 'FAIL'}")
    print(f"\n=== best config across S (BM={bBM} BN={bBN} BK={bBK}) ===")
    print(f"{'S':>5} {'XLA':>9} {'Pallas':>9} {'P/XLA':>7} {'P%pk':>6}")
    for Sv in (args.seq2 or [512, 1024, 2048]):
        x = jax.random.normal(jax.random.PRNGKey(Sv), (Sv, H), jnp.float32).astype(jnp.bfloat16)
        bm = min(bBM, cm(Sv, 128))
        tx = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x, iters=100, warmup=20)
        try:
            tp = bench_ms(jx(lambda x: m.svd_swiglu_mlp(x, *f, BM=bm, BN=bBN, BK=bBK, interpret=INTERP)),
                          x, iters=100, warmup=20)
            print(f"{Sv:>5} {tx:>9.4f} {tp:>9.4f} {tx/tp:>7.3f} {util_pct(tp, Sv, H, I, r, r_d):>5.0f}%")
        except Exception:
            print(f"{Sv:>5} {tx:>9.4f}  SKIP (VMEM)")


# ===========================================================================
# stage A — confirm the DEFAULT autotuner picks good configs end-to-end
# ===========================================================================
def cmd_autotune(args):
    name = (args.shape or ["llama-3b"])[0]
    H, I, r, r_d = shape_dims(name)
    print(f"### autotune (default ladder) {name} ###")
    print(f"{'S':>5} {'picked (BM,BN,BK)':>22} {'XLA':>8} {'Pallas':>8} {'P/XLA':>7} {'%pk':>6}")
    for S in (args.seq or [512, 2048, 4096]):
        x, f, _ = m._make_factors(S=S, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
        _, cfg = m.svd_swiglu_mlp_auto(x, *f, spec=SPEC, interpret=INTERP, verbose=False)
        tx = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x, iters=100, warmup=20)
        tp = bench_ms(jx(lambda x: m.svd_swiglu_mlp(
            x, *f, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], interpret=INTERP)), x, iters=100, warmup=20)
        tag = f"({cfg['BM']},{cfg['BN']},{cfg['BK']})"
        print(f"{S:>5} {tag:>22} {tx:>8.4f} {tp:>8.4f} {tx/tp:>7.3f} {util_pct(tp, S, H, I, r, r_d):>5.0f}%")


# ===========================================================================
# stage B — decode regime (S=1..32): bandwidth- & dispatch-bound analysis
# ===========================================================================
def cmd_decode(args):
    name = (args.shape or ["llama-3b"])[0]
    H, I, r, r_d = shape_dims(name)
    VMEM_BUDGET = int(SPEC.vmem_bytes * 0.80)

    # weights are S-independent: build + device_put ONCE, pre-cast to bf16
    _, f, (Wg, Wu, Wd) = m._make_factors(S=8, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
    f = tuple(jax.device_put(t) for t in f)
    Wg, Wu, Wd = jax.device_put(Wg), jax.device_put(Wu), jax.device_put(Wd)
    jax.block_until_ready(list(f) + [Wg, Wu, Wd])
    SVD_WB = sum(t.size for t in f) * SB
    DENSE_WB = (Wg.size + Wu.size + Wd.size) * SB
    FLOOR_US = SVD_WB / (PEAK_GB * 1e9) * 1e6
    print(f"### decode {name}  SVD weight stream={SVD_WB/1e6:.1f}MB -> floor={FLOOR_US:.1f}us "
          f"@ {PEAK_GB:.0f}GB/s (dense {DENSE_WB/1e6:.1f}MB) ###\n")

    def vmem_ok(BM, BN, BK):
        up = m._peak_vmem(BM, BN, H, r, SB, "upgate", BK)
        dn = m._peak_vmem(BM, BN, I, r_d, SB, "xuv", BK)
        return max(up, dn) <= VMEM_BUDGET, max(up, dn) / 2**20

    def median_us(fn, x):                 # per-call serialized latency (headline)
        reps = sorted(bench_s(fn, x, iters=300, warmup=25, serial=True) * 1e6 for _ in range(5))
        return reps[len(reps)//2]

    def async_us(fn, x):
        return bench_s(fn, x, iters=300, warmup=25) * 1e6

    def correct(BM, BN, BK, x, S):
        ref = m.svd_swiglu_mlp_ref(x.astype(jnp.float32), *[t.astype(jnp.float32) for t in f])
        out = m.svd_swiglu_mlp(x, *f, BM=BM, BN=BN, BK=BK, interpret=INTERP)
        return rel_to_fp32(out[:S], ref) < 3e-2

    # Pallas decode candidates (small BM). Pf = fused 2-launch (vs 5) to test
    # whether reducing kernel-launch count helps (it does not — overhead is per-call).
    CANDS = [("P.BM1", 1, 512, 512), ("P.BM8", 8, 512, 512), ("P.BM16", 16, 512, 512),
             ("Pf.BM8.fused", 8, 512, None)]
    for S in (args.seq or [1, 2, 4, 8, 16, 32]):
        x = jax.device_put(jax.random.normal(jax.random.PRNGKey(1000 + S), (S, H), jnp.float32).astype(jnp.bfloat16))
        xla = jx(lambda x: m.svd_swiglu_mlp_ref(x, *f))
        den = jx(lambda x: m.dense_swiglu_mlp_ref(x, Wg, Wu, Wd))
        a_xla, b_xla, a_den = median_us(xla, x), async_us(xla, x), median_us(den, x)
        gbx = SVD_WB / (a_xla * 1e-6) / 1e9
        print(f"=== S={S} ===  floor={FLOOR_US:.0f}us")
        print(f"  {'cand':12s} {'lat_us':>8} {'us/tok':>7} {'GB/s':>6} {'%bw':>5} "
              f"{'vXLA':>6} {'vDense':>7} {'dispatch_gap':>12}  ok")
        print(f"  {'XLA':12s} {a_xla:8.1f} {a_xla/S:7.1f} {gbx:6.0f} {100*gbx/PEAK_GB:4.0f}% "
              f"{'1.00':>6} {a_den/a_xla:7.2f} {a_xla-b_xla:11.1f}u")
        print(f"  {'dense':12s} {a_den:8.1f} {a_den/S:7.1f} {DENSE_WB/(a_den*1e-6)/1e9:6.0f} "
              f"{100*DENSE_WB/(a_den*1e-6)/1e9/PEAK_GB:4.0f}% {a_xla/a_den:6.2f}")
        cands = CANDS + ([("C0.BM512", 512, 512, 512)] if S in (1, 8) else [])
        for nm, BM, BN, BK in cands:
            if BM % 8 != 0 and BM != S:
                print(f"  {nm:12s} SKIP (BM={BM} invalid Pallas tile at S={S})"); continue
            ok_v, mb = vmem_ok(BM, BN, BK)
            if not ok_v:
                print(f"  {nm:12s} SKIP VMEM {mb:.1f}MB > budget"); continue
            try:
                if not correct(BM, BN, BK, x, S):
                    print(f"  {nm:12s} FAIL correctness"); continue
                fn = jx(lambda x, a=BM, b=BN, c=BK: m.svd_swiglu_mlp(x, *f, BM=a, BN=b, BK=c, interpret=INTERP))
                aP, bP = median_us(fn, x), async_us(fn, x)
            except Exception as e:               # noqa: BLE001 — actual VMEM OOM (predictor under-estimated)
                print(f"  {nm:12s} SKIP ({type(e).__name__})"); continue
            gbp = SVD_WB / (aP * 1e-6) / 1e9
            print(f"  {nm:12s} {aP:8.1f} {aP/S:7.1f} {gbp:6.0f} {100*gbp/PEAK_GB:4.0f}% "
                  f"{a_xla/aP:6.2f} {a_den/aP:7.2f} {aP-bP:11.1f}u")
        print()


# ===========================================================================
# stage C — fused (2-launch, U held whole) vs K-blocked prefill
# ===========================================================================
def cmd_fused(args):
    name = (args.shape or ["llama-3b"])[0]
    H, I, r, r_d = shape_dims(name)
    print(f"### fused-vs-kblocked {name}  vmem_budget={SPEC.vmem_bytes*SPEC.headroom/2**20:.0f}MB ###\n")

    _, f, (Wg, Wu, Wd) = m._make_factors(S=8, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
    x0, f0, _ = m._make_factors(S=2048, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
    _, cfg = m.svd_swiglu_mlp_auto(x0, *f0, spec=SPEC, interpret=INTERP, verbose=False)
    print(f"autotuner picks: {cfg}\n")

    print("=== correctness: fused (BK=None) vs fp32 ref @ S=512 ===")
    for BM, BN in [(min(512, CAP), 512), (min(512, CAP), 1024)]:
        try:
            rel = correctness(H, I, r, r_d, BM, BN, None, S=512)
            print(f"  fused BM={BM} BN={BN}: rel={rel:.2e} -> {'PASS' if rel < 3e-2 else 'FAIL'}")
        except Exception as e:                  # noqa: BLE001
            print(f"  fused BM={BM} BN={BN}: ERR {type(e).__name__}")

    FUSED = [(min(512, CAP), 512), (min(512, CAP), 1024)]   # fused candidates
    KBLK = (256, 512)                                       # (BN, BK) for K-blocked, large BM
    print(f"\n{'S':>5} {'dense':>8} {'XLA':>8} {'fused':>8} {'Kblk':>8} | "
          f"{'F/XLA':>6} {'K/XLA':>6} {'F%pk':>5} {'X%pk':>5}  {'fused cfg':>13}")
    for S in (args.seq or [128, 512, 2048, 4096]):
        x = jax.random.normal(jax.random.PRNGKey(S), (S, H), jnp.float32).astype(jnp.bfloat16)
        td = bench_ms(jx(lambda x: m.dense_swiglu_mlp_ref(x, Wg, Wu, Wd)), x)
        tx = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x)
        best = (float("inf"), "")
        for BM, BN in FUSED:
            bmu = min(BM, cm(S, 128))
            try:
                t = bench_ms(jx(lambda x, a=bmu, b=BN: m.svd_swiglu_mlp(x, *f, BM=a, BN=b, BK=None, interpret=INTERP)), x)
                if t < best[0]:
                    best = (t, f"BM{bmu}BN{BN}")
            except Exception:
                pass
        tf, fcfg = best
        bm = min(CAP, cm(S, 128))
        try:
            tk = bench_ms(jx(lambda x: m.svd_swiglu_mlp(x, *f, BM=bm, BN=KBLK[0], BK=KBLK[1], interpret=INTERP)), x)
        except Exception:
            tk = float("nan")
        fpk = util_pct(tf, S, H, I, r, r_d) if tf == tf and tf != float("inf") else 0
        print(f"{S:>5} {td:>8.4f} {tx:>8.4f} {tf:>8.4f} {tk:>8.4f} | "
              f"{tx/tf if tf else 0:>6.3f} {tx/tk:>6.3f} {fpk:>4.0f}% "
              f"{util_pct(tx, S, H, I, r, r_d):>4.0f}%  {fcfg:>13}")


# ===========================================================================
# stage D — full transformer LAYER (attention + SVD-MLP), MLP win diluted
# ===========================================================================
def _attn(x, Wq, Wk, Wv, Wo, heads):
    S, H = x.shape
    dh = H // heads
    q = (x @ Wq).reshape(S, heads, dh).transpose(1, 0, 2)
    k = (x @ Wk).reshape(S, heads, dh).transpose(1, 0, 2)
    v = (x @ Wv).reshape(S, heads, dh).transpose(1, 0, 2)
    a = jax.nn.softmax((q @ k.transpose(0, 2, 1)) / dh**0.5, axis=-1) @ v
    return a.transpose(1, 0, 2).reshape(S, H) @ Wo


def cmd_layer(args):
    shapes = args.shape or ["llama-1b", "llama-3b", "llama-8b"]
    for name in shapes:
        H, I, r, r_d = shape_dims(name)
        L, heads = LAYER_META.get(name, (28, max(1, H // 128)))
        _, f, (Wg, Wu, Wd) = m._make_factors(S=8, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
        kk = jax.random.split(jax.random.PRNGKey(1), 4)
        sc = 1.0 / H**0.5
        Wq, Wk, Wv, Wo = [(jax.random.normal(kk[i], (H, H), jnp.float32)*sc).astype(jnp.bfloat16) for i in range(4)]
        print(f"\n### layer {name}  H={H} I={I} r={r}  L={L} layers, {heads} heads ###")
        print(f"{'S':>5} {'attn':>7} {'mlp:dns':>8} {'mlp:XLA':>8} {'mlp:ours':>9} | "
              f"{'layer P/XLA':>11} {'layer P/dns':>11} {'model ours ms':>14}")
        for S in (args.seq or [512, 1024]):
            x = jax.random.normal(jax.random.PRNGKey(S), (S, H), jnp.float32).astype(jnp.bfloat16)
            try:
                _, cfg = m.svd_swiglu_mlp_auto(x, *f, spec=SPEC, interpret=INTERP, verbose=False)
                a_t = bench_ms(jx(lambda x: _attn(x, Wq, Wk, Wv, Wo, heads)), x)
                md = bench_ms(jx(lambda x: m.dense_swiglu_mlp_ref(x, Wg, Wu, Wd)), x)
                mx = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x)
                mo = bench_ms(jx(lambda x: m.svd_swiglu_mlp(
                    x, *f, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], interpret=INTERP)), x)
            except Exception as e:               # noqa: BLE001 — VMEM OOM at largest shape
                print(f"{S:>5}  SKIP ({type(e).__name__})"); continue
            lo, lx, ld = a_t + mo, a_t + mx, a_t + md
            print(f"{S:>5} {a_t:>7.4f} {md:>8.4f} {mx:>8.4f} {mo:>9.4f} | "
                  f"{lx/lo:>11.3f} {ld/lo:>11.3f} {L*lo:>14.3f}")


# ===========================================================================
# paper main table — BEST-tuned config per (shape, S)
# ===========================================================================
def _table_cfgs(S):
    Sr = cm(S, 128)
    if S <= 8:                                   # decode: small BM
        return [(8, 512, 512), (16, 512, 512)]
    bms = [bm for bm in (1024, 512) if bm <= Sr and bm <= CAP] or [min(Sr, CAP)]
    return [(bm, bn, bk) for bm in bms for bn in (256, 512) for bk in (512, 1024)]


def cmd_table(args):
    shapes = args.shape or ["llama-1b", "llama-3b", "llama-8b"]
    for name in shapes:
        H, I, r, r_d = shape_dims(name)
        _, f, (Wg, Wu, Wd) = m._make_factors(S=8, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
        print(f"\n### table {name}  H={H} I={I} r={r} ###")
        print(f"{'S':>5} {'dense':>8} {'XLA':>8} {'bestP':>8} {'cfg':>16} | "
              f"{'P/XLA':>6} {'P/dns':>6} {'P%pk':>5} {'X%pk':>5}")
        for S in (args.seq or [1, 512, 1024, 2048]):
            x = jax.random.normal(jax.random.PRNGKey(S), (S, H), jnp.float32).astype(jnp.bfloat16)
            td = bench_ms(jx(lambda x: m.dense_swiglu_mlp_ref(x, Wg, Wu, Wd)), x)
            tx = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x)
            best = (float("inf"), None)
            for BM, BN, BK in _table_cfgs(S):
                try:
                    t = bench_ms(jx(lambda x, a=BM, b=BN, c=BK: m.svd_swiglu_mlp(
                        x, *f, BM=a, BN=b, BK=c, interpret=INTERP)), x)
                    if t < best[0]:
                        best = (t, (BM, BN, BK))
                except Exception:
                    pass
            tp, bc = best
            if bc is None:
                print(f"{S:>5} {td:>8.4f} {tx:>8.4f}  (all configs OOM)"); continue
            tag = f"BM{bc[0]}BN{bc[1]}BK{bc[2]}"
            print(f"{S:>5} {td:>8.4f} {tx:>8.4f} {tp:>8.4f} {tag:>16} | "
                  f"{tx/tp:>6.3f} {td/tp:>6.3f} {util_pct(tp, S, H, I, r, r_d):>4.0f}% "
                  f"{util_pct(tx, S, H, I, r, r_d):>4.0f}%")


# ===========================================================================
# multi-shape — DEFAULT autotuner generalization across shapes
# ===========================================================================
def cmd_multishape(args):
    shapes = args.shape or ["llama-1b", "llama-3b", "llama-8b"]
    for name in shapes:
        H, I, r, r_d = shape_dims(name)
        _, f, (Wg, Wu, Wd) = m._make_factors(S=8, H=H, I=I, r=r, r_d=r_d, dtype=jnp.bfloat16)
        print(f"\n### multishape {name}  H={H} I={I} r={r} ###")
        print(f"{'S':>5} {'dense':>8} {'XLA':>8} {'Pallas':>8} {'cfg':>16} | "
              f"{'P/XLA':>6} {'P/dns':>6} {'P%pk':>5} {'X%pk':>5}")
        for S in (args.seq or [1, 512, 1024, 2048]):
            x = jax.random.normal(jax.random.PRNGKey(S), (S, H), jnp.float32).astype(jnp.bfloat16)
            try:
                _, cfg = m.svd_swiglu_mlp_auto(x, *f, spec=SPEC, interpret=INTERP, verbose=False)
            except Exception as e:              # noqa: BLE001
                print(f"{S:>5}  autotune FAIL: {type(e).__name__}"); continue
            td = bench_ms(jx(lambda x: m.dense_swiglu_mlp_ref(x, Wg, Wu, Wd)), x)
            tx = bench_ms(jx(lambda x: m.svd_swiglu_mlp_ref(x, *f)), x)
            tp = bench_ms(jx(lambda x: m.svd_swiglu_mlp(
                x, *f, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], interpret=INTERP)), x)
            tag = f"BM{cfg['BM']}BN{cfg['BN']}BK{cfg['BK']}"
            print(f"{S:>5} {td:>8.4f} {tx:>8.4f} {tp:>8.4f} {tag:>16} | "
                  f"{tx/tp:>6.3f} {td/tp:>6.3f} {util_pct(tp, S, H, I, r, r_d):>4.0f}% "
                  f"{util_pct(tx, S, H, I, r, r_d):>4.0f}%")


# ===========================================================================
# CLI
# ===========================================================================
EXPERIMENTS = {
    "profile": cmd_profile, "tiling": cmd_tiling, "autotune": cmd_autotune,
    "decode": cmd_decode, "fused": cmd_fused, "layer": cmd_layer,
    "table": cmd_table, "multishape": cmd_multishape,
}


def cmd_all(args):
    for nm in ("table", "decode", "layer"):
        print("\n" + "=" * 78 + f"\n  {nm.upper()}\n" + "=" * 78)
        EXPERIMENTS[nm](args)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--shape", nargs="+", default=None, choices=list(m.SHAPES),
                    help="MLP shape(s); default depends on the experiment")
    ap.add_argument("--seq", nargs="+", type=int, default=None,
                    help="sequence length(s) to sweep; default depends on the experiment")
    ap.add_argument("--seq2", nargs="+", type=int, default=None,
                    help="(tiling) sequence lengths for the best-config S-sweep")
    args = ap.parse_args()
    header()
    if INTERP:
        print("  (no TPU: interpret mode — logic only, timings are NOT representative)\n")
    (cmd_all if args.experiment == "all" else EXPERIMENTS[args.experiment])(args)


if __name__ == "__main__":
    main()
