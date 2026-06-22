# Running the TPU SVD-MLP on a free Colab/Kaggle TPU

Goal: get the **first real TPU number** — does the Pallas kernel beat (a) the dense
MLP and (b) XLA's own fusion of the same SVD math — plus the analytical block sizes.

## 0. Get a TPU runtime
- **Colab**: Runtime → Change runtime type → Hardware accelerator → **TPU** → Save.
- **Kaggle**: Notebook → Settings → Accelerator → **TPU VM**.

## 1. Install JAX with TPU support

```python
# Colab TPU VM already has jax+libtpu, but pin a known-good pair if needed:
!pip -q install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
import jax
print("devices:", jax.devices())          # expect TpuDevice(...) entries
assert jax.default_backend() == "tpu", "Not on a TPU runtime — switch the accelerator."
```

## 2. Get the code

Easiest: push the `NeuronMM/tpu/` folder to a repo and clone it, **or** upload the
six files (`svd_mlp_tpu.py`, `test_svd_mlp_tpu.py`, `benchmark_svd_mlp_tpu.py`,
`block_size_model.py`, `scalesim_adapter.py`, `llama_inference_tpu.py`) via the
Colab file browser.

> NOTE: as of now `tpu/` is **untracked** locally and `origin` is the upstream
> `dinghongsong/NeuronMM`, so a fresh clone has **no** `tpu/` folder. Either
> `git add tpu/ && commit && push` to your own fork first (then Option A), or
> just upload the six files (Option B).

```python
# Option A — clone (ONLY after you've committed+pushed tpu/ to your fork):
# !git clone https://github.com/<you>/NeuronMM.git && cd NeuronMM/tpu
# Option B — upload, then:
import os; os.chdir("/content")            # or wherever the 5 files landed
```

## 3. Correctness on real TPU kernels (interpret OFF)

```python
# the test file forces CPU; on TPU just call the kernel directly with interpret=False
import jax, jax.numpy as jnp
from svd_mlp_tpu import svd_swiglu_mlp, svd_swiglu_mlp_ref

S,H,I,r = 128, 3072, 8192, 1792
k = jax.random.split(jax.random.PRNGKey(0), 7); sc = 1/ H**0.5
mk = lambda key,shp: (jax.random.normal(key,shp,jnp.float32)*sc).astype(jnp.bfloat16)
x   = mk(k[0],(S,H))
Ug,Vg = mk(k[1],(I,r)), mk(k[2],(r,H))
Uu,Vu = mk(k[3],(I,r)), mk(k[4],(r,H))
Ud,Vd = mk(k[5],(H,r)), mk(k[6],(r,I))
# NOTE: BK=1536 is REQUIRED here. The fused default (BK=None) holds each U whole
# in VMEM (~30MB peak) and overflows the v4's ~13MB budget at this shape; the
# K-blocked path streams U. The number comes from block_size_model.pick_block_sizes.
out = svd_swiglu_mlp(x, Ug,Vg, Uu,Vu, Ud,Vd, BK=1536, interpret=False)   # REAL TPU kernel
ref = svd_swiglu_mlp_ref(x, Ug,Vg, Uu,Vu, Ud,Vd)
print("rel err:", float(jnp.max(jnp.abs(out.astype(jnp.float32)-ref.astype(jnp.float32)))
                        / jnp.max(jnp.abs(ref.astype(jnp.float32)))))   # expect ~1e-2 (bf16)
```

## 4. The benchmark — the number you actually want

```python
!python benchmark_svd_mlp_tpu.py
```
The benchmark now auto-detects the VMEM overflow at this shape and K-blocks the
Pallas path (it prints the chosen `BK`); no edit needed. Prints, on TPU:
```
dense MLP (no SVD)   : ... ms/iter
svd  MLP  (XLA)      : ... ms/iter   (..x vs dense)
svd  MLP  (Pallas)   : ... ms/iter   (..x vs dense)
--> Pallas vs XLA    : ..x   (Pallas wins / XLA already wins ...)
```
The **`Pallas vs XLA`** line is the key result: it tells you whether the hand-written
kernel is worth keeping for this shape, or whether XLA's fusion is already good enough.

## 5. Analytical block sizes — already has a TPU v4 spec

`block_size_model.py` ships a real `TPU_V4` spec (16 MiB VMEM, ridge ≈229) — no
edit needed for v4. If your Colab TPU is a different gen (v2/v3/v5e), add a
`TPUSpec` with that core's usable VMEM and bf16 ridge point and pass it as `spec`:

```python
from block_size_model import TPU_V4, TPUSpec, pick_block_sizes
spec = TPU_V4                          # or your own TPUSpec(...) for v2/v3/v5e
print(pick_block_sizes(M=128, K=3072, r=1792, N=8192, spec=spec, kernel="upgate"))
# -> k-blocked, BK=1536 on v4 (this is where the BK in steps 3-4 comes from)
```
Then pass the returned `BM`/`BN` to `svd_swiglu_mlp(..., BM=, BN=)` and re-run the
benchmark to confirm the model's choice.

## 6. End-to-end MLP stack (optional)

```python
!python llama_inference_tpu.py --layers 16          # random weights, real kernels
# or with converted weights from llama_inference.py::svd_flash:
# !python llama_inference_tpu.py --weights-path /content/model.safetensors --layers 16
```

## What to report back
- The three benchmark timings + the `Pallas vs XLA` ratio (per shape you care about).
- Which shapes say `needs K-blocking` from `block_size_model.py` (those need the
  contraction-dim blocking TODO before they'll run at full size).
```
