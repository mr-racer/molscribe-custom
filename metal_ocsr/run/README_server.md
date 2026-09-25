# metal_ocsr — stage 1: synthetic-only fine-tuning of MolScribe on organometallics

Server folder `/mnt/hard1/ivans_data/metal_ocsr` on 10.2.0.181. Design:
`code/molscribe/docs/superpowers/specs/2026-09-25-metal-ocsr-design.md`
(branch `feature/metal-ocsr` of github.com/mr-racer/molscribe-custom; `main` is not touched).

## Layout

```
metal_ocsr/
├── README.md                     this file (copy of code/molscribe/metal_ocsr/run/README_server.md)
├── env/
│   ├── venv/                     venv over conda env `como` (--system-site-packages)
│   ├── requirements-extra.txt    what the venv adds on top of como
│   └── constraints.txt           pins numpy/opencv/torch of como for pip
├── code/
│   ├── molscribe/                git worktree, branch feature/metal-ocsr
│   └── metal2d/                  levakrasnovs/metal2d @ 5a92226 (on sys.path via venv .pth file)
├── ckpts/base/                   swin_base_char_aux_1m680k.pth -> molscribe2_bench checkpoint
├── data/
│   ├── raw/                      downloads, never edited
│   │   ├── krasnov/              MetalLipoDB, MetalCytoToxDB, IrLumDB, IrCytoToxDB (Zenodo, CC-BY-4.0)
│   │   ├── xyz2mol_tm/           csd_smiles.csv etc. (github jensengroup/xyz2mol_tm, CSD-derived; COMMIT)
│   │   ├── tmqm/                 tmQM_y.csv (github uiocompcat/tmQM)
│   │   ├── db_catalysts/         reaction_catalyst_smiles.csv (export of reactions.reaction_catalyst,
│   │   │                         read-only) and catalysts_reconnected.csv
│   │   ├── lebedev_images/       the 21 publication_metal images of lebedev_bench
│   │   └── uspto_real/real.zip   MolScribe real benchmark images (USPTO subset -> organic validation)
│   ├── pool/                     candidates.csv (all sources, keys, drop reasons), leakage_removed.csv,
│   │                             train.csv (100k structures), val.csv (2k), manifest.json
│   ├── synth/                    rendered training data
│   │   ├── images/{train,val}/NNN/<pool_id>_<k>.png
│   │   ├── train_metal.csv, val_metal.csv   image_id, file_path, SMILES (label), node_coords, edges, gold ...
│   │   ├── {train,val}_manifest.json        counts, skip reasons, timing
│   │   └── qa/                   sample sheets with label atoms overlaid
│   ├── organic/                  train_organic.csv (200k PubChem SMILES, rendered on the fly),
│   │                             val_uspto_1k.csv + uspto_val/ (checkpoint selection only)
│   ├── test/                     never used for training or checkpoint selection
│   │   ├── t1_lebedev_metal/     gt.csv (re-annotated, 17 scored + 4 Markush excluded), images/, review_*.png
│   │   ├── t2_mrbw_metal/        MolRecBench-Wild metal subset, 529
│   │   └── t3_general/           ACS + MRBW organic, 500
│   └── smoke/                    tiny subsets for the smoke test
├── runs/<run>/                   train.py output: checkpoints per epoch, train.log, mlflow_run_id.txt, best_valid.json
├── eval/<run>/epoch_XX/          metal_ocsr.eval_metal output: <set>_pred.csv, metrics.json, failure galleries
├── logs/                         data-building logs
└── scratch/                      one-off probes
```

## Conventions

* Source form: RDKit dative SMILES (xyz2mol_tm / MetalLipoDB). Drawn form (training labels): plain metal-ligand
  bonds, no charges that only come from the dative convention, eta rings as a centroid pseudo-atom `[Ct]`.
  Post-processing (`molscribe/metal.py`) expands `[Ct]` back to dative bonds and writes Cp as `[cH-]`.
* Metrics: `em_strict` = `molscribe.metal.metal_key` equality (metal bonds single, charges cleared, eta rings
  bond-order free); `em_ligand` = same with the metal disconnected; `em_bench` = the earlier benchmark metric.
* Leakage guard: pool structures whose ligand key equals a T1/T2 gold were removed (`data/pool/leakage_removed.csv`).

## Reproduce

```bash
source code/molscribe/metal_ocsr/run/env.sh && cd $ROOT
$PY -m metal_ocsr.catalysts --from_csv scratch/catalyst_metal_classes.csv --out data/raw/db_catalysts
$PY -m metal_ocsr.organic --root $ROOT --pubchem /mnt/hard1/pubchem_db/pubchem_slice/pubchem_compounds.csv \
    --n 200000 --uspto_zip data/raw/uspto_real/real.zip --n_val 1000
$PY -m metal_ocsr.testsets.t1_lebedev_gt --images data/raw/lebedev_images --out data/test/t1_lebedev_metal
$PY -m metal_ocsr.pool --root $ROOT --n_total 102000 --n_val 2000 --workers 96
$PY -m metal_ocsr.make_synth --root $ROOT --split val --workers 96
$PY -m metal_ocsr.make_synth --root $ROOT --split train --workers 120
```

## Runs

| run | what | command |
|---|---|---|
| E0-baseline | base checkpoint, no training | `$PY -m metal_ocsr.eval_metal --root $ROOT --ckpt $BASE_CKPT --out eval/E0-baseline/epoch_00 --mlflow_run_name E0-baseline` |
| smoke | 20 steps, pipeline check | `bash code/molscribe/metal_ocsr/run/smoke_train.sh` |
| E1 | 6 epochs, metal synth + organic replay | `bash code/molscribe/metal_ocsr/run/train_E1.sh` (in tmux), then `bash code/molscribe/metal_ocsr/run/eval_run.sh E1` |

MLflow: http://localhost:5000, experiment `molscribe-metal-ocsr` (smoke runs: `molscribe-metal-ocsr-smoke`).
