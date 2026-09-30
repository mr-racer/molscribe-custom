# MolScribe (custom fork)

Picture of a molecule in → SMILES / molfile out.

A fork of [thomas0809/MolScribe](https://github.com/thomas0809/MolScribe) that uses **the same model and the same
public checkpoint**, so predictions stay comparable with the original, but runs 3–5× faster and reads far more
of the abbreviations that chemists actually draw (`OTBS`, `NHBoc`, `CO2Et`, `PPh3`, metal ligands, counter-ions —
9 947 entries instead of 89).

This page is everything needed to put it into a service. The detailed comparison with upstream, the benchmark
tables and the internals are in [docs/fork-details.md](docs/fork-details.md); the original project's README is
[README_upstream.md](README_upstream.md).

Reaction schemes (image → reactants / conditions / products) are a different model:
[rxnscribe-custom](https://gitlab.odanchem.org/odanchem/rxnscribe-custom), which plugs into a MolScribe that is
already running — see [Adding RxnScribe](#adding-rxnscribe) below.

---

## 1. Install

Python ≥ 3.7, PyTorch ≥ 1.11 (tested up to torch 2.13 / CUDA 13). A GPU is optional but ~10× faster.

```bash
pip install git+https://gitlab.odanchem.org/odanchem/molscribe-custom.git@main      # inference only
pip install "molscribe[serve] @ git+https://gitlab.odanchem.org/odanchem/molscribe-custom.git@main"
```

Inference pulls in only `torch`, `timm`, `rdkit`, `opencv`, `numpy`. Extras: `[serve]` (FastAPI router),
`[draw]` (`draw_prediction`), `[train]` (training and evaluation scripts).

The repository is private — pip needs credentials in the URL (`git+https://oauth2:<token>@gitlab.odanchem.org/...`)
or a deploy key.

### Checkpoint

One file, ~450 MB, from the upstream authors (unchanged — this fork does not retrain the model):

```bash
mkdir -p ckpts
wget -P ckpts https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth
```

Nothing else is downloaded at runtime. Put the file on a volume and pass its path; there is no hidden cache,
so the container works with no network.

### Smoke test

```bash
python predict.py --model_path ckpts/swin_base_char_aux_1m680k.pth --image_path assets/example.png
```

(`predict.py` hard-codes `cuda`; on a machine without a GPU use the three lines of §2 with
`device=torch.device("cpu")`.)

---

## 2. Use it

```python
import cv2, torch
from molscribe import MolScribe

model = MolScribe("ckpts/swin_base_char_aux_1m680k.pth", device=torch.device("cuda"), precision="tf32")

img = cv2.cvtColor(cv2.imread("mol.png"), cv2.COLOR_BGR2RGB)     # RGB numpy array
model.predict_image(img)                    # {'smiles': ..., 'molfile': ...}
model.predict_images(images, batch_size=64) # list of the same dicts, in order
```

Loading the model takes a few seconds; do it **once** at start-up and keep the object. `predict_images` takes a
lock around the GPU work, so several threads (e.g. FastAPI's thread pool) may call it concurrently — they queue.

Optional per-call flags: `return_confidence=True`, `return_atoms_bonds=True` (atom coordinates and the bond list).

---

## 3. The two knobs that matter: `precision` and `batch_size`

Everything else has a sane default. Pick a mode:

### Latency mode — one image per request

A worker that answers requests one at a time (search, an interactive UI):

```python
model = MolScribe(ckpt, device=torch.device("cuda"), precision="fp32")
model.predict_image(img)      # ≈ 47 ms, ≈ 0.5 GB VRAM
```

Use **`fp32` or `tf32`** here. At `batch_size=1` fp16 is *slower* (69 ms): the casting costs more than it saves.
`tf32` produces the same predictions as `fp32` on the benchmark and speeds up the larger matmuls, so it is an
equally safe default; at batch 1 the two are within noise of each other.

### Throughput mode — a pile of images

Processing a corpus, a batch job, a queue that can be drained in groups:

```python
model = MolScribe(ckpt, device=torch.device("cuda"), precision="fp16")
model.predict_images(images, batch_size=64)   # ≈ 11 ms/image, ≈ 89 images/s
```

Use **`fp16`** and the largest `batch_size` the VRAM budget allows (see below). `bf16` is equivalent; both differ
from fp32 on about 1 % of images with no measurable accuracy change.

| On an RTX A6000 | fp32 | tf32 | fp16 |
|---|---|---|---|
| Latency, one image per call | **47 ms** | ≈ fp32 (not measured separately) | 69 ms |
| Throughput, `batch_size=64` | 18 ms/img · 56 img/s | 13 ms/img · 75 img/s | **11 ms/img · 89 img/s** |

Beyond `batch_size≈64` the gain is small (56 → 57 img/s from 32 to 64 in fp32). Spare VRAM is better spent on a
second process than on a bigger batch.

On CPU everything still works (`device=torch.device("cpu")`, `precision="fp32"`), at roughly upstream's speed —
the CPU is compute-bound, so the fork's optimisations do not help there.

---

## 4. VRAM

Peak allocated by PyTorch; add ≈0.4 GB for the CUDA context itself.

| `batch_size` | fp32 / tf32 | fp16 / bf16 |
|---|---|---|
| 1 | 0.5 GB | 0.5 GB |
| 32 | 4.2 GB | 2.9 GB |
| 64 | 8.0 GB | 5.4 GB |

Rule of thumb: **peak ≈ 0.5 GB + `batch_size` × 0.12 GB (fp32) or × 0.08 GB (fp16)**. For a budget of `V` GB:

```
batch_size ≈ (V − 1) / 0.12     # fp32 / tf32
batch_size ≈ (V − 1) / 0.08     # fp16 / bf16
```

So 4 GB → 25 (fp32) or 37 (fp16); 12 GB → 90 or 130.

To make the budget a hard limit instead of an estimate:

```python
torch.cuda.set_per_process_memory_fraction(4 * 2**30 / torch.cuda.get_device_properties(0).total_memory)  # 4 GB
```

PyTorch then raises `torch.cuda.OutOfMemoryError` instead of eating the whole card. Catch it and halve
`batch_size`.

**Do not run 3–4 one-model containers to get throughput.** One process at `batch_size=64, precision="fp16"` does
≈89 img/s — more than four upstream workers at `batch_size=1` put together, on a quarter of the VRAM.

---

## 5. Serving it over HTTP

`molscribe/remote.py` turns a loaded model into a FastAPI route (or a Celery task). It is not imported by
`molscribe` itself, so it costs nothing when unused.

```python
import torch
from fastapi import FastAPI
from molscribe import MolScribe
from molscribe.remote import make_router

model = MolScribe("/models/swin_base_char_aux_1m680k.pth", device=torch.device("cuda"), precision="fp16")
app = FastAPI()
app.include_router(make_router(model))          # POST /molscribe/predict_batch
```

Request `{"images": [<base64 PNG>, ...], "batch_size": 32}` → `{"predictions": [...]}`, the same dicts
`predict_images` returns. Images travel as lossless PNG, so results are identical to an in-process call.

Celery instead of HTTP: `register_celery_task(celery_app, lambda: model)` → task `molscribe.predict_batch`.

Minimal Dockerfile:

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime
RUN pip install "molscribe[serve] @ git+https://oauth2:$TOKEN@gitlab.odanchem.org/odanchem/molscribe-custom.git@main" \
                fastapi uvicorn
COPY weights/swin_base_char_aux_1m680k.pth /models/
```

---

## 6. Adding RxnScribe

[RxnScribe](https://gitlab.odanchem.org/odanchem/rxnscribe-custom) parses whole reaction schemes and needs a
molecule recogniser for the crops it cuts out — i.e. exactly this MolScribe. It accepts any object with
`predict_images`, so an existing deployment does not have to change: either mount `make_router(model)` as above
and point RxnScribe's HTTP client at it, or, in a single container, hand it the live model object:

```python
from rxnscribe.serving import attach
attach(app, molscribe=model, ckpt="/models/pix2seq_reaction_full.ckpt", device=model.device, precision="fp16")
```

Details and the three deployment layouts: [rxnscribe-custom/README.md](https://gitlab.odanchem.org/odanchem/rxnscribe-custom).

---

## 7. When something goes wrong

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.OutOfMemoryError` | halve `batch_size`; see §4 for the budget formula |
| Slower than the numbers above at `batch_size=1` | `precision="fp16"` — switch to `tf32` |
| First call much slower than the rest | CUDA graph capture and cuDNN autotuning; warm up with one dummy image at start-up |
| Results change slightly when `batch_size` changes | cuDNN picks different convolution algorithms per shape. Predictions themselves are batch-invariant in this fork (they were not upstream); keep `batch_size` fixed if bit-exactness matters |
| A drawn abbreviation comes out as `*` | it is not in the dictionary. Adding entries: [docs/fork-details.md](docs/fork-details.md#the-label-dictionary) |
| Runs on CPU although a GPU is present | `device` defaults to CPU — pass `torch.device("cuda")` explicitly |

---

## Branches

* `main` — inference, serving, the label dictionary. This is what a service should install.
* `feature/metal-ocsr` — work in progress: fine-tuning on organometallics (dative bonds, ligands). Not for
  production yet.
