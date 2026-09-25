"""Evaluate a checkpoint on the stage-1 test sets and log the metrics to MLflow.

usage:
  python -m metal_ocsr.eval_metal --root /mnt/hard1/ivans_data/metal_ocsr --ckpt <.pth> --out eval/<run>/epoch_XX \
      [--sets t1_lebedev_metal,t2_mrbw_metal,t3_general,t4_synth_val] [--mlflow_run_id ID | --mlflow_run_name NAME] \
      [--step N] [--batch_size 32] [--device cuda]

sets    t1_lebedev_metal, t2_mrbw_metal, t3_general: root/data/test/<set>/gt.csv (id, source, image, gold, include)
        t4_synth_val: root/data/synth/val_metal.csv (gold = source SMILES of the rendered structure)
metrics metal sets:  em_bench   the metal metric of benchmark/evaluate.py (continuity with earlier numbers)
                     em_strict  molscribe.metal.metal_key: also tolerant to eta-ring drawing (circle / Kekule / Cp-)
                     em_ligand  metal disconnected: ligands and metals right, connectivity to the metal ignored
                     valid, rdkit_valid, and the same per subset (clean / eta / tags / primary metal)
        general set: em_stereo, em_nostereo, valid (benchmark/evaluate.py canon_general)
outputs <out>/<set>_pred.csv, <out>/metrics.json, <out>/<set>_failures.png (T1 and first failures of T2)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from molscribe.metal import metal_key, replace_rgroups

RDLogger.DisableLog('rdApp.*')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'benchmark'))
from evaluate import canon_general, canon_metal  # noqa: E402

METAL_SETS = {'t1_lebedev_metal', 't2_mrbw_metal', 't4_synth_val'}


def load_set(root, name):
    if name == 't4_synth_val':
        df = pd.read_csv(os.path.join(root, 'data', 'synth', 'val_metal.csv'))
        return pd.DataFrame(dict(id=df.image_id, source='synth', image=df.file_path, gold=df.gold, include=1,
                                 tags=np.where(df.has_eta, 'eta', ''), primary_metal=df.primary_metal)), \
            os.path.join(root, 'data')
    base = os.path.join(root, 'data', 'test', name)
    df = pd.read_csv(os.path.join(base, 'gt.csv'), keep_default_na=False)
    if 'tags' not in df.columns:
        df['tags'] = ''
    return df, base


def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def score_rows(df, preds, metal):
    rows = []
    for (_, r), p in zip(df.iterrows(), preds):
        res = dict(id=r.id, source=r.source, gold=r.gold, pred=p, tags=r.get('tags', ''),
                   primary_metal=r.get('primary_metal', ''))
        if metal:
            g_b, p_b = canon_metal(r.gold, False), canon_metal(p, False)
            g_s, p_s = metal_key(r.gold), metal_key(p)
            g_l, p_l = metal_key(r.gold, ligands_only=True), metal_key(p, ligands_only=True)
            res.update(em_bench=g_b is not None and g_b == p_b, em_strict=g_s is not None and g_s == p_s,
                       em_ligand=g_l is not None and g_l == p_l, valid=p_s is not None,
                       clean='*' not in replace_rgroups(r.gold))
        else:
            res.update(em_stereo=canon_general(r.gold, True) is not None and canon_general(r.gold, True) == canon_general(p, True),
                       em_nostereo=canon_general(r.gold, False) is not None and canon_general(r.gold, False) == canon_general(p, False),
                       valid=canon_general(p, False) is not None)
        res['rdkit_valid'] = bool(p) and Chem.MolFromSmiles(p) is not None
        rows.append(res)
    return pd.DataFrame(rows)


def summarize(det, metal):
    keys = ['em_bench', 'em_strict', 'em_ligand', 'valid', 'rdkit_valid'] if metal else \
        ['em_stereo', 'em_nostereo', 'valid', 'rdkit_valid']
    groups = {'all': det}
    if metal:
        groups['clean'] = det[det.clean]
        for tag in sorted({t for ts in det.tags for t in str(ts).split(',') if t}):
            groups[f'tag_{tag}'] = det[det.tags.astype(str).str.contains(tag)]
        groups['no_eta'] = det[~det.tags.astype(str).str.contains('eta')]
        if det.primary_metal.astype(str).str.len().gt(0).any():
            for m, g in det.groupby('primary_metal'):
                if len(g) >= 20:
                    groups[f'metal_{m}'] = g
    for src, g in det.groupby('source'):
        groups[f'src_{src}'] = g
    return {name: dict(n=len(g), **{k: round(float(g[k].mean()), 4) for k in keys}) for name, g in groups.items()
            if len(g)}


def failure_gallery(df, det, base, path, limit=24):
    fails = det[~det.em_strict] if 'em_strict' in det else det[~det.em_nostereo]
    tiles = []
    for _, r in fails.head(limit).iterrows():
        img = cv2.imdecode(np.fromfile(os.path.join(base, df.set_index('id').loc[r.id, 'image']), dtype=np.uint8), 1)
        h, w = img.shape[:2]
        k = 360 / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        tile = np.full((440, 380, 3), 255, np.uint8)
        tile[:img.shape[0], :img.shape[1]] = img
        for i, (label, s) in enumerate((('gold', r.gold), ('pred', r.pred))):
            cv2.putText(tile, f'{label}: {str(s)[:58]}', (3, 380 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (0, 120, 0) if label == 'gold' else (0, 0, 200), 1)
            cv2.putText(tile, str(s)[58:116], (3, 389 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90, 90, 90), 1)
        tiles.append(tile)
    if not tiles:
        return None
    while len(tiles) % 4:
        tiles.append(np.full((440, 380, 3), 255, np.uint8))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sets', default='t1_lebedev_metal,t2_mrbw_metal,t3_general,t4_synth_val')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--mlflow_uri', default='http://localhost:5000')
    ap.add_argument('--mlflow_experiment', default='molscribe-metal-ocsr')
    ap.add_argument('--mlflow_run_id', default=None, help='append to an existing run (e.g. the training run)')
    ap.add_argument('--mlflow_run_name', default=None, help='new run (e.g. E0-baseline)')
    ap.add_argument('--step', type=int, default=0, help='MLflow step, the epoch number of the checkpoint')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from molscribe import MolScribe
    model = MolScribe(args.ckpt, device=torch.device(args.device), num_workers=8)

    report, artifacts = {}, []
    for name in args.sets.split(','):
        df, base = load_set(args.root, name)
        df = df[df.include.astype(int) == 1].reset_index(drop=True)
        if args.limit:
            df = df.head(args.limit)
        t0 = time.time()
        images = [load_image(os.path.join(base, p)) for p in df.image]
        preds = [o['smiles'] for o in model.predict_images(images, batch_size=args.batch_size)]
        metal = name in METAL_SETS
        det = score_rows(df, preds, metal)
        det.to_csv(os.path.join(args.out, f'{name}_pred.csv'), index=False)
        report[name] = summarize(det, metal)
        report[name]['all']['seconds'] = round(time.time() - t0, 1)
        artifacts.append(os.path.join(args.out, f'{name}_pred.csv'))
        if name in ('t1_lebedev_metal', 't2_mrbw_metal'):
            g = failure_gallery(df, det, base, os.path.join(args.out, f'{name}_failures.png'))
            if g:
                artifacts.append(g)
        m = report[name]['all']
        print(name, json.dumps(m))
    report['ckpt'] = args.ckpt
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(report, f, indent=1)

    if args.mlflow_run_id or args.mlflow_run_name:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_id=args.mlflow_run_id, run_name=None if args.mlflow_run_id else args.mlflow_run_name):
            flat = {f'test/{s}/{g}/{k}': v for s, groups in report.items() if isinstance(groups, dict)
                    for g, vals in groups.items() for k, v in vals.items() if k != 'n' and isinstance(v, (int, float))}
            mlflow.log_metrics(flat, step=args.step)
            for a in artifacts + [os.path.join(args.out, 'metrics.json')]:
                mlflow.log_artifact(a, artifact_path=f'eval/epoch_{args.step:02d}')
            if args.mlflow_run_name:
                mlflow.log_params(dict(ckpt=args.ckpt, sets=args.sets))


if __name__ == '__main__':
    main()
