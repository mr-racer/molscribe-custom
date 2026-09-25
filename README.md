# MolScribe (custom fork)

Optical chemical structure recognition (image → SMILES / molfile), forked from
[thomas0809/MolScribe](https://github.com/thomas0809/MolScribe). Same model, same checkpoints
(`swin_base_char_aux_1m680k.pth`), different inference code and a much larger label dictionary.
The original README is kept in [README_upstream.md](README_upstream.md).

## What is different from upstream

| Area | Upstream | This fork |
|---|---|---|
| Decoding | Greedy search that shrinks the batch, grows the KV cache with `torch.cat`, and syncs with the host every step | Fixed batch, preallocated KV cache, host sync every 16 steps, one decoding step captured as a **CUDA graph** and replayed |
| Batch correctness | Positional encoding was indexed by the row in the batch → the prediction for an image **depended on its neighbours** (70 of 1029 benchmark images changed with batch size) | Every row gets the same encoding as `batch_size=1`; results are batch-invariant |
| Precision | fp32 only | `fp32`, `tf32`, `fp16`, `bf16` (autocast); activation checkpointing disabled at inference |
| Preprocessing | albumentations, single-threaded (≈9 ms/image, 36 % of a batched run) | OpenCV only, bitwise-identical output, prepared in threads while the GPU works; uint8 goes to the GPU, normalization happens there |
| Dependencies | OpenNMT-py 2.2 (needs torchtext), albumentations 1.1, scikit-image, matplotlib, pandas | `torch`, `timm==0.4.12`, `rdkit`, `opencv`, `numpy` (1.x or 2.x). Works on torch 2.13 / CUDA 13 |
| Label dictionary | 89 abbreviations, several wrong (`BOC`→boron, `SO2NH2`→nitro, `cPr`→n-propyl) | **9 947 entries**: protecting groups, alkyl/aryl/heteroaryl, acyl, fluoroalkyl, S/P/B/Si/Sn groups, amines, PEG/linkers, counter-ions, common reagents; every entry validated with RDKit and reviewed; all-caps spellings, reversed spellings (`EtO2C`/`CO2Et`), formulas with R (`OR`, `NR2`) |
| Organometallics | Charge-separated bonds, ligands lost | Dative bonds `L->M`, ligand labels (CO, PPh3, IMes, PEt3 …), counter-ions (BF4⁻, PF6⁻, BArF⁻…); a label's bond count selects the entry (`CO` = ligand with 1 bond, carbonyl with 2, molecule with 0) |
| Placeholders | fixed list `R, R1…R12, X, Y, Z…` | pattern: `Rα`, `R1'`, `Ar'`, `EWG`, `LG`, `PG`, `Nu`, `?` … → `*` |
| In-line groups | not supported | `-CONH-`, `-NHCO-`, `-CO2-`, `-SO2-`, `-OCH2O-`, `-Gly-`: left neighbour → first atom, right neighbour → second |

Accuracy is unchanged or slightly better (ACS + MolRecBench-Wild, 500 images, exact match with stereo:
63.6 % → 63.8 %; 66.9 % → 68.0 % on images whose ground truth has no unresolved labels).

## Speed (RTX A6000, 1 029 real images, full `predict_images` incl. RDKit)

| Mode | Upstream | This fork | Speed-up |
|---|---|---|---|
| **Latency** — one image per call (`batch_size=1`, fp32) | 241 ms | **47 ms** | ×5.2 |
| Latency, metal complexes (larger molecules) | 408 ms | **68 ms** | ×6.0 |
| **Throughput** — batched, fp32 | 46 ms/img, 22 img/s (bs 32) | 18 ms/img, 56 img/s (bs 64) | ×2.6 |
| Throughput, tf32, bs 64 | — | 13 ms/img, 75 img/s | ×3.5 |
| Throughput, **fp16**, bs 64 | — | **11 ms/img, 89 img/s** | ×4.1 |

Where the time went: preprocessing 8.9 → 0.3 ms/img, decoder ≈30 → 5 ms/img (CUDA graphs), encoder 13 → 5 ms/img (fp16).
On CPU the two versions are at parity (the CPU is compute-bound, not launch-bound).
Reproduce with `benchmark/` (`prepare_data.py`, `run_benchmark.py`, `evaluate.py`, `profile_stages.py`).

## Usage

```python
import cv2, torch
from molscribe import MolScribe

model = MolScribe("ckpts/swin_base_char_aux_1m680k.pth", device=torch.device("cuda"), precision="fp16")

img = cv2.cvtColor(cv2.imread("mol.png"), cv2.COLOR_BGR2RGB)
out = model.predict_image(img)                        # {'smiles': ..., 'molfile': ...}
outs = model.predict_images(list_of_images, batch_size=64)
```

`predict_image` / `predict_images` accept the same options as upstream (`return_atoms_bonds`, `return_confidence`).

### Constructor options

| Option | Default | Meaning |
|---|---|---|
| `device` | `cpu` | `torch.device("cuda")` for GPU |
| `precision` | `fp32` | `tf32` (TF32 matmuls, results identical to fp32 on the benchmark), `fp16`, `bf16` (autocast; ~1 % of images differ from fp32, accuracy unchanged) |
| `fast_decoding` | `True` | static-batch decoder. `False` = upstream decoder (only for A/B checks) |
| `cuda_graph` | `True` | capture the decoding step as a CUDA graph (needs CUDA + `fast_decoding`) |
| `preprocess_threads` | `min(8, cpus)` | threads that prepare the next batch while the current one runs; `0` disables |
| `num_workers` | `1` | processes for the RDKit postprocessing (`1` = in-process) |

## Choosing a mode

**Latency mode** — many independent requests of one image each (typical worker fed from a queue):

```python
model = MolScribe(ckpt, device=torch.device("cuda"), precision="tf32")
model.predict_image(img)          # ≈ 47 ms, ≈ 0.5 GB of VRAM
```
Use `fp32` or `tf32`. At `batch_size=1` fp16 is *slower* (69 ms): casting costs more than it saves.

**Throughput mode** — a pile of images to process as fast as possible:

```python
model = MolScribe(ckpt, device=torch.device("cuda"), precision="fp16")
model.predict_images(images, batch_size=64)   # ≈ 11 ms/img
```
Keep the batch size fixed: one CUDA graph is captured per distinct batch size (and one more for the last, smaller
batch), each holding its own buffers. Two or three distinct sizes are fine, fifty are not.

Both modes can live in one process: one `MolScribe` object, `predict_image` for single requests and
`predict_images(..., batch_size=N)` for bulk jobs. Because results are batch-invariant, collecting single requests
into a batch (dynamic batching) changes nothing but speed.

## VRAM

Peak memory allocated by PyTorch (add ≈0.4 GB for the CUDA context that `nvidia-smi` shows on top):

| `batch_size` | fp32 / tf32 | fp16 / bf16 |
|---|---|---|
| 1 | 0.5 GB | 0.5 GB |
| 32 | 4.2 GB | 2.9 GB |
| 64 | 8.0 GB | 5.4 GB |

Rule of thumb: **peak ≈ 0.5 GB + batch_size × 0.12 GB (fp32) or × 0.08 GB (fp16)**. So for a budget of `V` GB:

```
batch_size ≈ (V − 1) / 0.12   # fp32/tf32
batch_size ≈ (V − 1) / 0.08   # fp16/bf16
```
e.g. 4 GB → 25 (fp32) or 37 (fp16); 12 GB → 90 (fp32) or 130 (fp16). Beyond ~64 the throughput gain is small
(56 → 57 img/s from 32 to 64 in fp32), so a bigger budget is better spent on a second process than on a bigger batch.

To enforce the budget rather than estimate it:

```python
torch.cuda.set_per_process_memory_fraction(4 / torch.cuda.get_device_properties(0).total_memory * 2**30)  # 4 GB
```
PyTorch then raises `OutOfMemoryError` instead of overshooting; catch it and halve `batch_size`.

Running 3–4 containers with one model each is no longer needed: one process with `batch_size=64, precision="fp16"`
gives ≈89 img/s, more than four upstream workers at `batch_size=1` (4 × 4 img/s).

## The label dictionary

Labels drawn on atoms (`OTBS`, `NHBoc`, `CO2Et`, `PPh3` …) are expanded by `molscribe/constants.py`:

* hand-written entries at the top of `EXTRA_SUBSTITUTIONS`, the generated block between the
  `GENERATED ABBREVIATIONS` markers;
* fragment SMILES: **attachment atom first**, with radical electrons = number of bonds
  (`[O][Si](C)(C)C(C)(C)C` for OTBS, `[C](=O)OC` for CO2Me, `[NH]C(=O)OC(C)(C)C` for NHBoc);
  the third argument of `_extra` is the bond count when it is not 1 (`_extra(['CO'], "[C]=O", 2)`,
  `_extra(['BF4'], "F[B-](F)(F)F", 0)`);
* hydrogen counts on heteroatoms are recomputed from the final valence, so one entry serves `NHBoc` as a terminal
  group and `NBoc` inside a ring;
* ligands bonded only to metals come from `LIGAND_SMILES` (dative bond);
* all-caps variants (`BOC`, `NHBOC`) are generated automatically.

To add entries: append `_extra([...], "...")` lines, then run
`python benchmark/abbrev_check.py` (every key must expand to a valid molecule) and
`python benchmark/test_postprocess.py`. Bulk additions go through `benchmark/abbrev_merge.py`
(JSON in, validated block out).

## Files

```
molscribe/interface.py            MolScribe class, batching, precision, preprocessing threads
molscribe/inference/static_greedy.py   static-batch decoder + CUDA graph capture
molscribe/transforms.py           OpenCV preprocessing (== albumentations pipeline, bitwise)
molscribe/transformer/onmt_modules.py  the few OpenNMT modules the decoder uses (vendored, MIT)
molscribe/chemistry.py            graph -> SMILES, label expansion, dative bonds
molscribe/constants.py            label dictionary, R-group pattern, ligands, counter-ions
benchmark/                        data preparation, timing, accuracy, equivalence checks, profiler
```

Training code and scripts are unchanged from upstream.
