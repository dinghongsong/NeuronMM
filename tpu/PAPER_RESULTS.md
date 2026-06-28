# Extending NeuronMM (SVD-Flash) to the TPU — Results

Draft results for porting the NeuronMM / SVD-Flash SVD-compressed SwiGLU-MLP kernel
(Song et al., *NeuronMLP*, arXiv:2510.25977) from AWS Trainium (NKI) to Google TPU
via JAX/Pallas. All numbers are **measured on real Cloud TPUs**, bf16 inputs with
fp32 accumulation, matmul precision pinned identically across all paths.

## 1. Setup

| Item | Value |
|---|---|
| Hardware | **TPU v5e** (`v5litepod-1`, europe-west4) and **TPU v6e / Trillium** (`v6e-1`, asia-northeast1) |
| Models (MLP shapes) | Llama-3.2-1B (H=2048, I=8192), Llama-3.2-3B (H=3072, I=8192), Llama-3-8B (H=4096, I=14336) |
| Compression | block-aligned SVD, **compress ratio 0.8** → ranks r = 1280 / 1792 / 2560 (paper Eq. 5) |
| Sequence lengths | S = 1 (decode), 512, 1024, 2048 (prefill) |
| Baselines | **XLA-SVD** (XLA compiling the same `(x·Vᵀ)·Uᵀ` SVD math) and **dense** (uncompressed SwiGLU) |
| Metric | per-call latency (ms); speedup vs XLA-SVD and vs dense; achieved % of the bf16 compute roofline (v5e 197 TFLOP/s, v6e 918 TFLOP/s) |
| Kernel config | best of a per-(shape,S) tile sweep; correctness gated to rel ≤ 3e-2 vs an fp32 reference |

The TPU baseline is fundamentally different from the paper's Trainium baseline:
**XLA already compiles the SVD math** (it materializes the rank intermediate, same as
our K-blocked kernel). So on TPU the bar is not an un-fused framework path — it is a
production compiler. This reframes the question from "does fusion help?" to "can a
hand-written kernel beat XLA?"

## 2. Main result — MLP kernel vs XLA vs dense

**Speedup of the Pallas SVD-MLP over XLA-SVD (>1.0 = we win). Best-tuned config.**

### TPU v5e (mature XLA backend)
| shape | S=1 | S=512 | S=1024 | S=2048 | XLA % of roofline |
|---|---|---|---|---|---|
| llama-1b | 0.99× | 0.98× | 0.96× | 0.97× | 90–96% |
| llama-3b | 1.01× | 0.98× | 0.96× | 0.98× | 92–97% |
| llama-8b | 1.03× | 0.99× | 1.00× | 1.00× | 95–97% |

### TPU v6e / Trillium (newer XLA backend)
| shape | S=1 | S=512 | S=1024 | S=2048 | XLA % of roofline |
|---|---|---|---|---|---|
| llama-1b | 0.99× | **1.02×** | 0.91× | **1.05×** | 50–74% |
| llama-3b | **1.06×** | 0.87× | 0.99× | **1.08×** | 60–75% |
| llama-8b | 1.01× | 0.95× | **1.17×** | **1.05×** | 57–82% |

**Headline finding — the hand-written kernel's advantage is compiler-maturity-dependent:**
- On **v5e**, XLA is at **90–97% of the compute roofline** — essentially optimal — and our
  kernel **ties** it (0.96–1.03×). There is no slack to claim.
- On **v6e**, XLA's newer codegen reaches only **50–82% of roofline**, and a v6e-retuned
  kernel **beats it by up to 1.17×** at large S (and 1.05–1.08× consistently at S=2048).

In every case the SVD kernel beats **dense** by **1.13–1.50×** (the compression's
fixed FLOP/byte saving, which both XLA and Pallas realize) — i.e. the NeuronMM
compression benefit transfers to TPU.

## 3. Mechanism — the roofline tells the whole story

Achieved TFLOP/s is exact (FLOPs are unambiguous), so it is a faithful matrix-unit
utilization measure. At S=2048, llama-3b:

| | v5e | v6e |
|---|---|---|
| XLA-SVD utilization | **97%** of peak | **75%** of peak |
| our kernel utilization | 95% | **81%** |
| our kernel vs XLA | 0.98× (tie) | **1.08× (win)** |

The kernel's lever is the **row-block size BM** (keeps the 128×128 matrix unit fed):
the paper's analytical block model picks the *minimum* compute-bound BM=128, which
starves the unit (≈53% utilization on v5e). Sweeping BM is decisive — and the optimal
BM **scales with the matrix unit**: **v5e wants BM=512, v6e wants BM=1024** (its larger
MXU needs a bigger row block). This is a clean ablation (Fig.: utilization vs BM).

## 4. Decode regime (S=1)

Single-token decode is **bandwidth- and dispatch-bound**, not compute-bound: latency is
~flat in S (≈200 µs/call on v6e for 1–32 tokens), dominated by a ~90 µs host-dispatch
overhead **shared by XLA and Pallas**. Consequences (measured):
- With decode-appropriate small BM (8/16), Pallas **ties** XLA at S=1 (0.99–1.06×).
- The fused single-launch kernel is **impossible** at these shapes on *any* current TPU
  (usable VMEM ≈ 16–32 MB; the 29 MB weights never fit whole — K-blocking is mandatory).
- Reducing kernel-launch count does **not** help: a fused 2-launch path ties the
  5-launch path (the overhead is per-*call*, not per-*kernel*).

This is a useful negative result: the paper's decode latency win came from a weak
Trainium dispatch path; TPU's compiled executable does not expose that slack.

## 5. End-to-end — full transformer layer (attention + MLP)

A layer is attention (not compressed; identical across variants) + the SVD-MLP. With
attention ≈ **40–47% of layer latency**, the MLP speedup is diluted ~half:

| chip / shape / S | MLP vs XLA | **layer vs XLA** | layer vs dense |
|---|---|---|---|
| v5e llama-3b, S=1024 | 0.96× | 0.98× | **1.12×** |
| v6e llama-8b, S=1024 | **1.17×** | **1.10×** | **1.19×** |
| v6e llama-3b, S=1024 | 0.99× | ~1.00× | 1.09× |

So end-to-end: on v5e the layer **ties** XLA but is **1.12× faster than a dense-MLP layer**
(the compression benefit); on v6e, where the MLP wins, the layer **wins ~1.10×** over
XLA-SVD and **1.19×** over dense at the favorable shapes.

## 6. Honest scope & limitations

- The win over XLA is **specific to v6e** (immature codegen) and **large S** (≥1024–2048,
  multiple row blocks); on v5e it is a **tie**, and at S=512/1024 on v6e it is mixed (a
  single-row-block dip at S=1024 with BM=1024). We do **not** claim a blanket "Pallas > XLA."
- **DeepSeek-V3** (H=7168, I=18432, r=4096) shape is wired in but not yet swept (heavy);
  a dedicated run is future work for full paper-faithfulness.
- A **full-model generation** loop (KV cache, sampling) is approximated here by the
  per-layer attention+MLP decomposition; integrating into a real decode loop is future work.
- TPU **v4** is retired (unavailable in any region we probed) and **v5p** requires a quota
  grant; the cross-hardware story is v5e vs v6e.

## 7. Takeaway for the paper

Porting SVD-Flash to TPU shows the **compression benefit transfers** (1.13–1.50× MLP,
~1.12–1.19× per layer over dense, via both XLA and our kernel). The novel TPU-specific
result is that **hand-written kernels earn their keep as a function of compiler maturity**:
they **tie** a mature backend (v5e, XLA at the roofline) and **beat** a new-silicon backend
(v6e, XLA at 50–82% of roofline) by up to **1.17×**. The kernel's portability lever is a
single hardware-scaled parameter — the row-block size BM (512 on v5e, 1024 on v6e) — which
the paper's analytical model under-sizes.
