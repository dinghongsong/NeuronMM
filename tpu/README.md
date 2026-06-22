# NeuronMM on TPU (Pallas)

TPU/Pallas port of the SVD-compressed SwiGLU MLP from the Trainium NKI kernels
(`NeuronMM/test_speedup.py`). Same algorithm, expressed against the TPU memory
hierarchy via [Pallas](https://docs.jax.dev/en/latest/pallas/index.html).

## What's here

| File | Purpose |
|---|---|
| `svd_mlp_tpu.py` | The fused kernels + public `svd_swiglu_mlp(...)` and a plain-JAX reference. |
| `test_svd_mlp_tpu.py` | Correctness vs the reference (runs on CPU via interpret mode). |
| `benchmark_svd_mlp_tpu.py` | SVD vs dense MLP timing (meaningful only on a real TPU). |
| `llama_inference_tpu.py` | End-to-end MLP-stack driver: loads `svd_flash` weights (or random), runs every layer's MLP through the kernel, checks vs the decomposed-linear reference. TPU counterpart of `../llama_inference.py`. |
| `block_size_model.py` | Analytical `BM`/`BN` selector — paper Eq. 10/11 + roofline, ported to TPU (fill in the target TPU's VMEM + ridge point). Replaces the brute-force `../tuning_mlp_up.py`. |
| `COLAB.md` | Copy-paste cells to run correctness + the dense/XLA/Pallas benchmark on a free Colab/Kaggle TPU. |
| `scalesim_adapter.py` | Offline cycle / HBM-traffic estimate (dense vs SVD) on a v4-like 128×128 systolic array via SCALE-Sim — `pip install scalesim==2.0.2`. Cross-checks the block model without a TPU; can't see kernel fusion. |

## The math

Every MLP weight `W (out, in)` is stored SVD-factored as `W ≈ U @ V`
(`U: (out, r)`, `V: (r, in)`, rank `r ≪ out, in`). A linear layer becomes a
chain of two skinny matmuls:

```
y = x @ W.T = (x @ V.T) @ U.T      # reduce to rank r, then expand back to out
```

For the SwiGLU MLP:

```
gate = (x @ V_gate.T) @ U_gate.T          # (S, I)
up   = (x @ V_up.T)   @ U_up.T            # (S, I)
h    = silu(gate) * up                    # (S, I)
out  = (h @ V_down.T) @ U_down.T          # (S, H)
```

## What is fused, and why (paper Sec. 4.2.1, Algorithm 3)

Both matmuls of each projection are fused into **one** Pallas kernel. The rank
strip `(B_M, r)` (`= x @ V.T`) is computed once per row block into a **persistent
VMEM scratch**, guarded by `pl.when(col_block == 0)`, then reused for every
output-column block — the direct port of the paper's **on-chip caching**. The
rank strip and the gate/up activations never touch HBM; only the final `(S, I)`
(and `(S, H)`) results are written back. SwiGLU `silu(gate) * up` is fused before
the store (the paper's Scalar+Vector-engine step → the TPU VPU).

The paper's other two techniques map as: **blocking** → the `BM`/`BN` grid;
**implicit transposition** → *unnecessary on TPU*, since `jnp.dot`/Mosaic handle
operand layout (no systolic "stationary-must-be-transposed" rule to dodge).

## Trainium → TPU mapping

| Trainium / NKI | TPU / Pallas |
|---|---|
| `nl.load` / `nl.store` + manual tile loops | `pl.BlockSpec` + `grid` |
| SBUF (24 MB) scratch | VMEM |
| capacity-aware SBUF caching of the rank strip | persistent VMEM scratch + `pl.when(col==0)` |
| `nisa.nc_matmul` into PSUM (2 MB) + K-loop | `jnp.dot` (Mosaic drives the MXU) |
| `nc_transpose` / implicit transposition | gone — `jnp.dot` takes any layout |
| `nl.silu` / `nl.multiply` | `jax.nn.silu` / `*` (VPU) |
| bf16 multiply / fp32 accumulate | `preferred_element_type=jnp.float32` |
| `get_*_params` / `tuning_mlp_up.py` (brute force) | `BM` / `BN` — re-derive via paper Eq. 10/11 for VMEM |

## Running

Correctness on a CPU box (no TPU) — uses Pallas `interpret=True`:

```bash
pip install "jax[cpu]"
cd NeuronMM/tpu
python test_svd_mlp_tpu.py
```

On a TPU host (Cloud TPU VM / Colab / Kaggle), `svd_swiglu_mlp(..., interpret=False)`
(the default) lowers to real TPU kernels:

```bash
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python benchmark_svd_mlp_tpu.py
```

## Status / next steps

- [x] Fused kernels (Algorithm 2/3) with VMEM rank-strip caching; correctness
      verified vs reference (incl. unaligned shapes).
- [x] End-to-end MLP driver that consumes `svd_flash`-format weights
      (`llama_inference_tpu.py`), verified on CPU.
- [x] Validated on **real** weights: all 16 layers of `Macro2017/llama-3.2-1b_0.8_svd`
      (bf16, loaded via `ml_dtypes` — no torch) match the reference, rel err 0.
- [x] Target = **TPU v4** (16 MiB VMEM, ridge ~229 FLOPs/byte) wired into
      `block_size_model.py` (`TPU_V4`).
- [x] **K-blocking implemented** (`BK` arg): streams `U` in contraction chunks when
      it exceeds VMEM — required on v4 even for the 1B model. Verified on synthetic
      and real weights (all 16 layers, rel err ~7e-7).
- [x] Offline cycle/traffic estimate via SCALE-Sim: SVD-alone ~1.28× fewer cycles,
      ~1.27× less DRAM at S=128; 20% array util corroborates the MEM-bound verdict.
- [ ] Run on real TPU v4; confirm Pallas beats plain-XLA `x @ V.T @ U.T`.
- [ ] r-blocking (block the rank dim) for larger models (8B / DeepSeek) whose rank
      strip + V blocks exceed 16 MiB VMEM even when K-blocked.
- [ ] Wire into a JAX/Flax Llama (swap `SVD_LlamaMLP.forward` for the kernel) for
      full generation, plus tensor-parallel sharding (paper Fig. 4).
```
