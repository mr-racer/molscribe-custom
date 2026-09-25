# Metal-OCSR, stage 1: synthetic-only fine-tuning of MolScribe on organometallics

Date: 2026-09-25. Branch: `feature/metal-ocsr` (main is not touched).

## 1. Goal and scope

Teach the MolScribe fork to recognise organometallic / coordination complexes from images, and find out
whether the approach works *in principle* on the MolScribe architecture.

Stage 1 (this spec): fine-tune on **synthetic images only** and measure on held-out real images from papers.
Stage 2 (later, separate spec): fine-tune on real images.

Out of scope for stage 1: real-image training data, style transfer, stereo at the metal centre,
counter-ions in the output, architecture changes.

### Baseline (current checkpoint `swin_base_char_aux_1m680k.pth`, GPU bench of 2026-09-25)

| Set | n | exact match (no stereo) |
|---|---|---|
| MRBW metal (`metal.csv`) | 529 | 12.5 % |
| MRBW metal, clean subset | 183 | 22 % |
| general (ACS + MRBW organic) | 500 | 70.4 % |

## 2. Infrastructure

Everything runs on `oleg@10.2.0.181` (RTX A6000 48 GB). During training the GPU is fully ours.
The user launches long jobs in **tmux**; the implementation delivers ready-to-paste commands.

- **Python env:** conda env `como` (torch 2.13+cu130, rdkit 2026.03 with Cairo, albumentations 2.0.8,
  SmilesPE, opencv) plus a venv created with `--system-site-packages` that adds only what is missing:
  `timm==0.4.12`, `mlflow-skinny==3.16.0`, `metal2d` (editable, `--no-deps`). Set `NO_ALBUMENTATIONS_UPDATE=1`.
  pip goes through the proxy `http://193.233.4.1:3128`.
- **MLflow:** existing server `http://localhost:5000` (sqlite backend `/mnt/hard1/ivans_data/mlflow/mlflow.db`),
  experiment `molscribe-metal-ocsr`.
- **Code:** development on branch `feature/metal-ocsr` of this repo; the server gets the branch as a git worktree
  under the project folder. New pipeline code lives in a top-level `metal_ocsr/` package
  (`scripts/` is git-ignored in this repo). Changes to `molscribe/` are kept minimal (section 6).

### Project folder `/mnt/hard1/ivans_data/metal_ocsr/`

```
metal_ocsr/
├── README.md                  structure, provenance, how to reproduce
├── env/venv/                  venv over conda `como`; env/requirements-extra.txt
├── code/
│   ├── molscribe/             git worktree, branch feature/metal-ocsr
│   └── metal2d/               clone of levakrasnovs/metal2d, pinned commit
├── data/
│   ├── raw/                   downloads as-is, never edited (+ SOURCES.md: URL, licence, date)
│   │   ├── krasnov/           MetalLipoDB, MetalCytoToxDB, IrLumDB, IrCytoToxDB (Zenodo, CC-BY-4.0)
│   │   ├── xyz2mol_tm/        CSD-derived dative SMILES
│   │   ├── tmqm/
│   │   ├── db_catalysts/      read-only export of reactions.reaction_catalyst
│   │   └── pubchem_organic/   organic replay SMILES sampled from /mnt/hard1/pubchem_db
│   ├── pool/                  normalised SMILES pool, leakage report, train/val split
│   ├── synth/                 rendered images (sharded), train_metal.csv, val_metal.csv, qa/
│   ├── organic/               train_organic.csv, val_uspto_1k.csv
│   └── test/                  t1_lebedev_metal/, t2_mrbw_metal/, t3_general/
├── ckpts/base/                symlink to swin_base_char_aux_1m680k.pth
├── runs/<run>/                train.py output: per-epoch checkpoints, logs, args
├── eval/<run>/epoch_XX/       per-set predictions CSV + metrics.json
├── scripts/                   tmux-ready shell wrappers (build, render, train, eval)
└── scratch/                   one-off probes
```

## 3. SMILES pool

| Source | ~unique | Role |
|---|---|---|
| xyz2mol_tm `csd_smiles.csv` ("fixed" dative column) + tmQM | ~220k | bulk diversity (CSD use approved by the user) |
| MetalLipoDB `smiles_complex` | 4.7k | paper-style complexes, 546 multinuclear |
| `reactions.reaction_catalyst` (read-only) | ~85 bonded + reconnected | in-domain; oversampled ×20 |

Catalyst reconnection only where unambiguous: `[Pd]` + neutral phosphines → P–Pd; `[M+n]` + halide /
acetate / triflate anions → covalent M–X. Ambiguous records (dba, Pd/C, bare salts such as CuI) are dropped.

Filters, in order: RDKit parses → metal2d depicts without exception → label sequence ≤ 480 `chartok_coords`
tokens (≈ ≤ 100 heavy atoms) → dedup on the normalised key.

**Leakage guard:** every pool entry whose metal-disconnected, charge-stripped, stereo-free key matches a test GT
(T1, T2) is removed and listed in `pool/leakage_removed.csv`.

Stage-1 training subset: ~100k structures, stratified by metal (sqrt weighting, floor per metal) so Ru/Cu do
not dominate. Validation: 2k structures held out **by structure**.

## 4. Labels ("as drawn")

The label graph is what the image shows:

- Dative bond → **single** bond. Charges that exist only because of the dative convention are removed
  (`[Cl-]->[Pt+2]` → `Cl–Pt`, `[c-]->[Ir+3]` → `c–Ir`). Neutral donors stay neutral (pyridine N with 4 bonds);
  the fork's `_coordination_bonds_to_dative` turns these back into dative bonds at inference.
- **Overall complex charge** (Fe⁺, Mn⁺ in the test set) is drawn on the metal with p = 0.5 and then appears in the
  label; otherwise the metal is neutral.
- **η-groups** (Cp, Cp*, arenes, allyl) → a centroid pseudo-atom `[Ct]` at the ring centre, single bond M–Ct;
  the M–C ring bonds are removed from the label graph. `[Ct]` is built from characters already in
  `vocab_chars.json`; no vocabulary or embedding change.
- **Abbreviations:** with p ≈ 0.3 per group, `CO`, `Me`, `Et`, `iPr`, `tBu`, `Ph`, `PPh3`, `Cp*` are condensed
  into a labelled pseudo-atom (only labels the fork already expands, see `constants.py`).
- No stereo at the metal: a wedge / hash drawn on an M–L bond is labelled as a wedge edge (types 5/6, as drawn,
  so the visual signal stays consistent with organic data), and post-processing strips any stereo that lands on a
  metal atom.

## 5. Rendering and augmentation

Offline, on the server CPUs, **2 renders per SMILES** with independent random parameters (~200k PNG).
Coordinates come from `metal2d.depict()`; drawing uses the user's standard RDKit style
(`NMRArena-release/scratch_si/render_structure_panels.py`: `MolDraw2DCairo`, `bondLineWidth=2`,
`padding=0.10`, `minFontSize=9`, `maxFontSize=-1`, default palette). Atom pixel positions are read from the
drawer (`GetDrawCoords`), so labels are exact. `[Ct]` is drawn as an invisible atom (empty label).

**Layer 1: geometry** (applied to 2D coordinates before drawing, so text stays upright)

| Augmentation | p |
|---|---|
| rotate layout 0–360° | 0.7 |
| mirror | 0.3 |
| perspective squash of η-rings (sandwich look) | 0.5 per η-ring |
| wedge / hash on 1–2 M–L bonds (piano-stool legs) | 0.3 |
| CoordGen layout instead of metal2d | 0.15 |
| bond length jitter 15–45 px | always |

**Layer 2: style around standard RDKit**: 50 % exactly the standard style; otherwise `bondLineWidth` 1–3,
font size, `multipleBondOffset`, font face (DejaVu Sans / Liberation Sans / Liberation Serif),
black-and-white palette (0.4), explicit CH₃ (0.15), coloured substructure highlight (0.1),
**aromatic circles** drawn by us inside arene / Cp rings from ring pixel coordinates (0.3).

**Layer 3: crop-from-paper noise**, always placed **outside** the molecule's convex hull with a margin, so the
label stays correct:

| Noise | p |
|---|---|
| compound label near the structure: `3a`, `II`, `12b`, `Ru1`, `(±)-7` (regular / bold) | 0.5 |
| intrusions clipped by the box edge: fragments of other molecules, arrows, `+`, `(iv)`, `t-Bu`, `K2CO3`, `2 NaCl` | 0.4 |
| charge brackets `[ ]2+` and counter-ion text `(PF6)2` (label unchanged: model learns to ignore) | 0.15 |
| loose / tight box, table-frame fragments | 0.05 |

**Layer 4: load-time degradation** (albumentations, keypoint-aware), metal profile:
existing MolScribe transforms (CropWhite, CropAndPad, PadWhite, Downscale 0.2–0.5, Blur, GaussNoise,
salt-and-pepper) **with image rotation limited to ±5°** instead of ±90°, plus JPEG q30–90 (0.3),
binarisation (0.1), erode/dilate (0.2), background tint (0.1). The organic replay keeps the upstream profile.

QA before training: a sheet of 100 random samples with atom points and label SMILES overlaid.

## 6. Training

Start from `swin_base_char_aux_1m680k.pth`; same architecture flags as upstream (`swin_base`, `transformer`,
`chartok_coords,edges`, `coord_bins 64`, `sep_xy`, `input_size 384`, `vocab_chars.json`).

Data per epoch, 1:1: `aux_file` = ~200k metal PNGs with exact coordinates; `train_file` = 200k organic SMILES
from the PubChem slice, rendered on the fly by Indigo (`--dynamic_indigo --mol_augment --include_condensed`).

| | upstream (from scratch) | stage-1 fine-tune |
|---|---|---|
| encoder / decoder LR | 4e-4 / 4e-4 | 5e-5 / 1e-4 |
| warmup, schedule | 0.02, cosine | 0.05, cosine |
| epochs | 30 | 6 |
| batch | 256 | 128 effective (as large as fits in 48 GB, no accumulation if possible) |
| other | fp16, grad checkpointing, label smoothing 0.1 | same |

Checkpoint selection uses **only** synthetic metal val (2k) and organic val (1k from USPTO real,
`molscribe_bench/raw/real.zip`); test sets are never used for selection.

Code changes in `molscribe/` (minimal):
1. `dataset.py`: flag for aux coordinates already normalised to the image (current `--coords_file aux_file`
   re-normalises to the molecule box); per-dataset augmentation profile (metal vs upstream).
2. `chemistry.py`: `[Ct]` post-processing: find the predicted ring whose polygon contains the centroid and replace
   `[Ct]` with M–C bonds to all its atoms.
3. `train.py`: new flags; MLflow logging (params, per-step loss/LR, per-epoch val metrics, artifacts).

Runs: **E0** baseline (no training); **E1** main run, test metrics after every epoch;
**E1-control** (optional, if the gain is ambiguous): same schedule, organic data only.

## 7. Evaluation

| Set | Content | GT |
|---|---|---|
| T1 | lebedev_bench `publication_metal`, 21 images (3 PMC papers; η-rich) | annotated by Claude, reviewed by the user |
| T2 | MRBW metal, 529 (clean 183) | existing |
| T3 | general, 500 | existing |
| T4 | synthetic metal val, 2k | generator |

T1 GT: RDKit form (dative bonds, η as M–C to every ring atom, as in xyz2mol_tm), main complex only
(no arrows, reagents, `+ 2NaCl`, compound numbers or counter-ions). A review sheet (original | GT drawn by
metal2d | SMILES) is approved by the user before T1 is used.

Metrics (one script for E0 and every epoch):
- **EM-strict**: canonical SMILES equality after normalisation: all metal bonds single, formal charges on the
  metal and its neighbours zeroed, stereo removed, predicted `[Ct]` expanded.
- **EM-ligand**: metal disconnected; compare multisets of ligand SMILES + metal symbols.
- valid rate; breakdown per metal and η / non-η.
- T3 keeps the existing bench metrics (`em_nostereo`, `em_stereo`) for comparability with 70.4 %.

MLflow per run: params (all args, git commit, dataset sizes and hashes, per-metal composition),
metrics (loss/LR per step; T4, organic val, T1–T3 per epoch), artifacts (QA sheet, data manifest, per-image
prediction CSVs, failure galleries for T1/T2). Checkpoints stay on disk; MLflow stores their paths.

## 8. Decision rules after E1

- T4 ≥ 90 %, T1/T2 improve, T3 drops ≤ 1 pp → the approach works; go to stage 2 (real data).
- T4 high, T1/T2 barely move → image-domain gap; next: style, augmentation, real data.
- T4 low → label / architecture problem (sequence length, `[Ct]`, M–L bonds); fix before spending GPU time.

## 9. Risks

- metal2d is beta (0.3.1); polynuclear layouts may fail → fall back to CoordGen, count failures in the manifest.
- `[Ct]` ring assignment fails when the predicted ring is incomplete → count as a miss; no heuristics in stage 1.
- CSD-derived SMILES may encode η-bonds inconsistently → normalise through RDKit (`DativeBondsToHaptic` /
  `HapticBondsToDative`) and keep only entries that round-trip.
- Label convention mismatch between T2 gold and our labels → the normalised metrics above; report EM-ligand too.
