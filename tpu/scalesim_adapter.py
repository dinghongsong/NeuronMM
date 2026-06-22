"""
SCALE-Sim adapter: offline cycle / memory-traffic estimates for the SVD MLP on a
TPU-v4-like systolic array, WITHOUT a TPU.

SCALE-Sim (https://github.com/scalesim-project/SCALE-Sim) models a configurable
systolic array + on-chip SRAM and reports, per GEMM: cycles, array utilization
(~MFU), and DRAM (HBM) accesses. We map each MLP matmul to a GEMM and compare:

    dense MLP  (3 GEMMs: gate, up, down)           -- paper's "NKI XW"
    svd   MLP  (6 GEMMs: {gate,up,down}_{v,u})      -- paper's "NKI XUV"

so this reproduces, offline, the paper's *SVD-alone* effect (compute + traffic
reduction; ~1.54x in the paper).

IMPORTANT -- what SCALE-Sim does NOT capture:
  * It models each GEMM independently, so it does NOT see kernel FUSION (the
    on-chip rank-strip caching that avoids re-materializing intermediates). The
    paper's *TrainiumFusion* gain (the other ~1.35x) is exactly the part our
    Pallas kernel adds and SCALE-Sim cannot show -- it would, if anything,
    over-count SVD's DRAM traffic (it materializes every intermediate).
  * No bf16 numerics, no VPU (SiLU), flat SRAM model (not VMEM/CMEM split), and a
    simplified DRAM model. Treat outputs as first-order estimates, not truth.

So: SCALE-Sim is a good sanity check on array utilization and the SVD traffic
reduction; it is NOT a substitute for measuring the fused kernel on a real TPU.

Array = 128x128 (TPU v4 MXU). SRAM sized to v4 VMEM (16 MiB). Dataflow = ws.
"""

import csv
import os
import tempfile

# llama-3.2-1b MLP shape (matches llama_inference_tpu.py defaults)
S, H, I, R = 128, 2048, 8192, 1280     # seq, hidden, intermediate, svd rank


def write_config(path, vmem_kb=16 * 1024, array=128, dataflow="ws"):
    # split VMEM across SCALE-Sim's three SRAM buffers (sum = VMEM)
    ifmap = vmem_kb * 6 // 16
    filt = vmem_kb * 6 // 16
    ofmap = vmem_kb - ifmap - filt
    # canonical SCALE-Sim v2 (2.0.2) config -- no layout/sparsity sections
    with open(path, "w") as f:
        f.write(
            "[general]\n"
            "run_name = neuronmm_v4\n\n"
            "[architecture_presets]\n"
            f"ArrayHeight:    {array}\n"
            f"ArrayWidth:     {array}\n"
            f"IfmapSramSzkB:  {ifmap}\n"
            f"FilterSramSzkB: {filt}\n"
            f"OfmapSramSzkB:  {ofmap}\n"
            "IfmapOffset:    0\n"
            "FilterOffset:   10000000\n"
            "OfmapOffset:    20000000\n"
            f"Dataflow:       {dataflow}\n"
            "Bandwidth:      100\n"
            "MemoryBanks:    1\n\n"
            "[run_presets]\n"
            "InterfaceBandwidth: CALC\n"
        )


def write_topology(path, gemms):
    # GEMM topology: header row, then "name, M, N, K," (trailing comma required)
    with open(path, "w") as f:
        f.write("Layer,M,N,K,\n")
        for name, M, N, K in gemms:
            f.write(f"{name},{M},{N},{K},\n")


def dense_gemms(S, H, I, R):
    # y = x @ W.T  ->  GEMM (M=S, N=out, K=in)
    return [
        ("gate", S, I, H),
        ("up",   S, I, H),
        ("down", S, H, I),
    ]


def svd_gemms(S, H, I, R):
    # each projection = two GEMMs through the rank R
    return [
        ("gate_v", S, R, H), ("gate_u", S, I, R),
        ("up_v",   S, R, H), ("up_u",   S, I, R),
        ("down_v", S, R, I), ("down_u", S, H, R),
    ]


def run(cfg, topo, outdir):
    from scalesim.scale_sim import scalesim
    s = scalesim(save_disk_space=True, verbose=False,
                 config=cfg, topology=topo, input_type_gemm=True)
    s.run_scale(top_path=outdir)
    return os.path.join(outdir, "neuronmm_v4")


def parse_reports(run_dir):
    cycles = 0
    utils = []
    comp = os.path.join(run_dir, "COMPUTE_REPORT.csv")
    with open(comp) as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            cycles += int(float(row["Total Cycles"]))
            utils.append(float(row["Compute Util %"]))
    dram = 0
    det = os.path.join(run_dir, "DETAILED_ACCESS_REPORT.csv")
    with open(det) as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            for key, v in row.items():
                if "DRAM" in key and ("Reads" in key or "Writes" in key):
                    dram += int(float(v))
    return cycles, (sum(utils) / len(utils) if utils else 0.0), dram


def main():
    import argparse
    ap = argparse.ArgumentParser(description="SCALE-Sim dense-vs-SVD MLP estimate")
    ap.add_argument("--seq", type=int, default=S, help="sequence length (M)")
    ap.add_argument("--hidden", type=int, default=H)
    ap.add_argument("--intermediate", type=int, default=I)
    ap.add_argument("--rank", type=int, default=None, help="SVD rank (overrides --ratio)")
    ap.add_argument("--ratio", type=float, default=0.8,
                    help="compress ratio -> rank = round(I*H*ratio/((I+H)*128))*128")
    ap.add_argument("--array", type=int, default=128, help="systolic array dim (v4 MXU=128)")
    ap.add_argument("--vmem-mib", type=int, default=16, help="on-chip SRAM in MiB (v4=16)")
    ap.add_argument("--dataflow", choices=["ws", "os", "is"], default="ws")
    args = ap.parse_args()

    s, h, i = args.seq, args.hidden, args.intermediate
    r = args.rank or round(i * h * args.ratio / ((i + h) * 128)) * 128

    work = tempfile.mkdtemp(prefix="scalesim_")
    cfg = os.path.join(work, "arch.cfg")
    write_config(cfg, vmem_kb=args.vmem_mib * 1024, array=args.array, dataflow=args.dataflow)

    results = {}
    for label, gemms in [("dense", dense_gemms(s, h, i, r)),
                         ("svd", svd_gemms(s, h, i, r))]:
        topo = os.path.join(work, f"{label}.csv")
        write_topology(topo, gemms)
        run_dir = run(cfg, topo, os.path.join(work, label))
        results[label] = parse_reports(run_dir)

    print(f"\narray {args.array}x{args.array}, SRAM={args.vmem_mib}MiB, {args.dataflow}.  "
          f"MLP  S={s} H={h} I={i} r={r}\n")
    print(f"{'':6s} {'cycles':>12s} {'avg array util %':>18s} {'DRAM accesses':>16s}")
    for label in ("dense", "svd"):
        cyc, util, dram = results[label]
        print(f"{label:6s} {cyc:12d} {util:18.1f} {dram:16d}")
    dc, _, dd = results["dense"]
    sc, _, sd = results["svd"]
    print(f"\nSVD-alone (per-GEMM, no fusion):  "
          f"{dc/sc:.2f}x fewer cycles,  {dd/sd:.2f}x less DRAM traffic")
    print("Fusion (rank-strip caching) is ON TOP of this and needs the Pallas "
          "kernel on real TPU — SCALE-Sim can't model it.")


if __name__ == "__main__":
    main()
