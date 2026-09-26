"""Render the pool into MolScribe aux training data.

usage:
  python -m metal_ocsr.make_synth --root /mnt/hard1/ivans_data/metal_ocsr --split train --workers 64
  python -m metal_ocsr.make_synth --root ... --split val --workers 64

reads   root/data/pool/{split}.csv (column renders = images per structure)
writes  root/data/synth/images/{split}/{shard}/{pool_id}_{k}.png
        root/data/synth/{split}_metal.csv   image_id, file_path (relative to root/data), SMILES (label),
                                            node_coords, edges, gold (source SMILES) + metadata
        root/data/synth/{split}_manifest.json   counts, skip reasons, timing
        root/data/synth/qa/{split}_qa_XX.png    random samples with the label atoms overlaid
The metal2d layout is computed once per structure and reused by its renders; every render has its own seed.
"""
import argparse
import json
import os
import signal
import time
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from rdkit import RDLogger

from . import label as L
from .render import depict, render

RDLogger.DisableLog('rdApp.*')
TIMEOUT_S = 30
ARGS = {}


def _alarm(signum, frame):
    raise TimeoutError()


def _init(root, split, seed, out_name='synth'):
    RDLogger.DisableLog('rdApp.*')
    import logging
    logging.getLogger('metal2d').setLevel(logging.ERROR)
    signal.signal(signal.SIGALRM, _alarm)
    ARGS.update(root=root, split=split, seed=seed, out_name=out_name)


def work(row):
    """Render every image of one pool structure. Returns (rows, reasons, seconds)."""
    t0 = time.time()
    rows, reasons = [], Counter()
    split, data = ARGS['split'], os.path.join(ARGS['root'], 'data')
    shard = f'{int(row["pool_id"][1:]) // 1000:03d}'
    rel_dir = os.path.join(ARGS['out_name'], 'images', split, shard)
    os.makedirs(os.path.join(data, rel_dir), exist_ok=True)
    signal.alarm(TIMEOUT_S)
    try:
        src = L.load_source(row['smiles'])
        try:
            layout = depict(src)
        except L.Skip:
            layout = None  # every render falls back to CoordGen
        for k in range(int(row['renders'])):
            rng = np.random.default_rng([ARGS['seed'], int(row['pool_id'][1:]), k, int(split == 'val')])
            try:
                s = render(src, rng, depicted=layout)
            except L.Skip as e:
                reasons[str(e)] += 1
                continue
            except Exception as e:
                reasons['error:' + type(e).__name__] += 1
                continue
            image_id = f'{row["pool_id"]}_{k}'
            rel = os.path.join(rel_dir, image_id + '.png')
            cv2.imwrite(os.path.join(data, rel), s['image'])
            rows.append(dict(image_id=image_id, file_path=rel, SMILES=s['smiles'],
                             node_coords=json.dumps(s['node_coords']), edges=json.dumps(s['edges']),
                             gold=s['gold'] or row['canonical'], pool_id=row['pool_id'], source=row['source'],
                             primary_metal=row['primary_metal'], has_eta=row['has_eta'], **s['meta']))
            reasons['ok'] += 1
    except TimeoutError:
        reasons['timeout'] += 1
    except L.Skip as e:
        reasons[str(e)] += int(row['renders'])
    finally:
        signal.alarm(0)
    return rows, reasons, time.time() - t0


def qa_sheets(df, data, out_dir, split, n=60, per_sheet=20, seed=0):
    """Random samples with the label atom positions (red) and the label SMILES, for a visual check before training."""
    os.makedirs(out_dir, exist_ok=True)
    sample = df.sample(min(n, len(df)), random_state=seed)
    tiles = []
    for _, r in sample.iterrows():
        img = cv2.imread(os.path.join(data, r.file_path))
        H, W = img.shape[:2]
        for x, y in json.loads(r.node_coords):
            cv2.circle(img, (int(x * W), int(y * H)), max(2, int(min(H, W) / 150)), (0, 0, 255), -1)
        sc = 360 / max(H, W)
        img = cv2.resize(img, (max(1, int(W * sc)), max(1, int(H * sc))))
        tile = np.full((420, 380, 3), 255, np.uint8)
        tile[:img.shape[0], :img.shape[1]] = img
        text = r.SMILES
        for line in range(2):
            cv2.putText(tile, text[line * 62:(line + 1) * 62], (3, 385 + 16 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (160, 0, 0), 1)
        tiles.append(tile)
    for s in range(0, len(tiles), per_sheet):
        chunk = tiles[s:s + per_sheet]
        while len(chunk) % 5:
            chunk.append(np.full((420, 380, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(chunk[i:i + 5]) for i in range(0, len(chunk), 5)])
        cv2.imwrite(os.path.join(out_dir, f'{split}_qa_{s // per_sheet:02d}.png'), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', required=True, choices=['train', 'val'])
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='first N structures only (smoke runs)')
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--out_name', default='synth', help='output folder under root/data (E1: synth, E2: synth_e2)')
    args = ap.parse_args()
    data = os.path.join(args.root, 'data')
    pool = pd.read_csv(os.path.join(data, 'pool', f'{args.split}.csv'))
    if args.limit:
        pool = pool.head(args.limit)
    t0 = time.time()
    rows, reasons, secs = [], Counter(), []
    # apply_async + get(timeout) instead of imap: a native crash (segfault inside RDKit layout code) kills the worker
    # and loses its task; imap would then wait forever, here the task times out and the pool replaces the worker
    with Pool(args.workers, initializer=_init, initargs=(args.root, args.split, args.seed, args.out_name), maxtasksperchild=500) as p:
        pending = [(rec, p.apply_async(work, (rec,))) for rec in pool.to_dict('records')]
        for i, (rec, res) in enumerate(pending):
            try:
                r, why, s = res.get(timeout=TIMEOUT_S * 4)
            except Exception:
                r, why, s = [], Counter(crash=int(rec['renders'])), 0.0
            rows.extend(r)
            reasons.update(why)
            secs.append(s)
            if (i + 1) % 5000 == 0:
                print(f'{i + 1}/{len(pool)} structures, {len(rows)} images, {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows).sort_values('image_id').reset_index(drop=True)
    out_csv = os.path.join(data, args.out_name, f'{args.split}_metal.csv')
    df.to_csv(out_csv, index=False)
    manifest = dict(split=args.split, structures=len(pool), images=len(df), reasons=reasons,
                    seconds=round(time.time() - t0), per_structure_s=dict(mean=float(np.mean(secs)),
                                                                          p99=float(np.percentile(secs, 99))),
                    sources=Counter(df.source), metals=Counter(df.primary_metal).most_common(),
                    eta_images=int(df.n_eta.gt(0).sum()), standard_style=float(df.standard_style.mean()),
                    coordgen=float(df.coordgen.mean()), mean_atoms=float(df.n_atoms.mean()),
                    rgroup_images=float(df.n_rgroup.gt(0).mean()), dative_images=float(df.n_dative.gt(0).mean()),
                    arrow_images=float(df.arrows.mean()))
    with open(os.path.join(data, args.out_name, f'{args.split}_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, default=int)
    qa_sheets(df, data, os.path.join(data, args.out_name, 'qa'), args.split)
    print(json.dumps({k: manifest[k] for k in ('structures', 'images', 'reasons', 'seconds')}, default=int))


if __name__ == '__main__':
    main()
