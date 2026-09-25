"""Stage-1 SMILES pool: all sources -> normalised, de-duplicated, leakage-free, stratified train/val split.

usage:
  python -m metal_ocsr.pool --root /mnt/hard1/ivans_data/metal_ocsr --n_total 102000 --n_val 2000 --workers 64

inputs  (root/data/raw)   xyz2mol_tm/csd_smiles.csv (smiles_csd_api_fixed), tmqm/tmQM_y.csv (SMILES),
                          krasnov/MetalLipoDB.csv (smiles_complex), db_catalysts/catalysts_reconnected.csv (smiles)
        (root/data/test)  */gt.csv, column gold: test ground truth for the leakage guard
outputs (root/data/pool)  candidates.csv  every parsed candidate with keys (or the reason it was dropped)
                          leakage_removed.csv, train.csv, val.csv, manifest.json
Keys (molscribe.metal.metal_key): key_strict de-duplicates; key_ligand (metal disconnected, charges and stereo
removed) is the leakage key, deliberately looser than exact match so near-duplicates of test structures go too.
"""
import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from molscribe.metal import is_metal, metal_key
from . import label as L

RDLogger.DisableLog('rdApp.*')

SOURCE_PRIORITY = ['catalysts', 'metallipodb', 'csd', 'tmqm']  # kept on duplicates, in this order
MIN_HEAVY, MAX_HEAVY = 4, 100
CATALYST_RENDERS, DEFAULT_RENDERS = 40, 2  # catalysts oversampled x20


def load_sources(raw):
    frames = []
    csd = pd.read_csv(os.path.join(raw, 'xyz2mol_tm', 'csd_smiles.csv'), usecols=['IDs', 'smiles_csd_api_fixed'])
    frames.append(pd.DataFrame(dict(source='csd', source_id=csd.IDs, smiles=csd.smiles_csd_api_fixed)))
    tmqm = pd.read_csv(os.path.join(raw, 'tmqm', 'tmQM_y.csv'), sep=';', usecols=['CSD_code', 'SMILES'])
    frames.append(pd.DataFrame(dict(source='tmqm', source_id=tmqm.CSD_code, smiles=tmqm.SMILES)))
    lipo = pd.read_csv(os.path.join(raw, 'krasnov', 'MetalLipoDB.csv'), keep_default_na=False, na_values=[''])
    lipo = lipo.drop_duplicates('smiles_complex')
    frames.append(pd.DataFrame(dict(source='metallipodb', source_id=lipo.record_id, smiles=lipo.smiles_complex,
                                    doi=lipo.doi, article_label=lipo.abbreviation_in_the_article)))
    cat_path = os.path.join(raw, 'db_catalysts', 'catalysts_reconnected.csv')
    if os.path.exists(cat_path):
        cat = pd.read_csv(cat_path)
        frames.append(pd.DataFrame(dict(source='catalysts', source_id=cat.source_smiles, smiles=cat.smiles)))
    df = pd.concat(frames, ignore_index=True)
    return df[df.smiles.notna() & (df.smiles.astype(str).str.len() > 0)].reset_index(drop=True)


def describe(smiles):
    """Parse one candidate; returns a dict of features or the drop reason."""
    try:
        src = L.load_source(smiles)
    except L.Skip as e:
        return dict(reason=str(e))
    n = src.GetNumHeavyAtoms()
    if not MIN_HEAVY <= n <= MAX_HEAVY:
        return dict(reason='size')
    metals = Counter(a.GetSymbol() for a in src.GetAtoms() if is_metal(a))
    canon = Chem.MolToSmiles(src)
    key_strict, key_ligand = metal_key(canon), metal_key(canon, ligands_only=True)
    if key_strict is None or key_ligand is None:
        return dict(reason='key')
    return dict(reason='', canonical=canon, n_heavy=n, metals='-'.join(sorted(metals)),
                primary_metal=max(metals, key=lambda s: (metals[s], Chem.Atom(s).GetAtomicNum())),
                has_eta=bool(L.eta_rings(src)), multinuclear=sum(metals.values()) > 1,
                key_strict=key_strict, key_ligand=key_ligand)


def test_keys(test_dir):
    keys = {}
    for path in glob.glob(os.path.join(test_dir, '*', 'gt.csv')):
        gt = pd.read_csv(path)
        for gold in gt.gold.dropna():
            k = metal_key(gold, ligands_only=True)
            if k:
                keys[k] = os.path.basename(os.path.dirname(path))
    return keys


def stratified(df, n, rng, floor=300):
    """n rows with every primary metal represented: a floor per metal, the rest shared by sqrt(size)."""
    groups = {m: g for m, g in df.groupby('primary_metal')}
    if n >= len(df):
        return df
    quota = {m: min(len(g), floor) for m, g in groups.items()}
    left = n - sum(quota.values())
    if left < 0:
        raise ValueError('floor too high for n')
    names = sorted(groups)
    w = np.array([np.sqrt(len(groups[m])) for m in names])
    for _ in range(10):  # redistribute what small metals cannot take
        room = np.array([len(groups[m]) - quota[m] for m in names])
        if left <= 0 or room.sum() == 0:
            break
        share = np.floor(w * (room > 0) / (w * (room > 0)).sum() * left).astype(int)
        share = np.minimum(share, room)
        if share.sum() == 0:
            share[np.argmax(room)] = min(left, room.max())
        for m, s in zip(names, share):
            quota[m] += int(s)
        left = n - sum(quota.values())
    parts = [g.sample(quota[m], random_state=int(rng.integers(1 << 31))) for m, g in groups.items()]
    return pd.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--n_total', type=int, default=102000, help='structures in train + val')
    ap.add_argument('--n_val', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--seed', type=int, default=20260925)
    args = ap.parse_args()
    raw, out = os.path.join(args.root, 'data', 'raw'), os.path.join(args.root, 'data', 'pool')
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_sources(raw)
    with Pool(args.workers) as p:
        feats = p.map(describe, df.smiles.tolist(), chunksize=256)
    df = pd.concat([df, pd.DataFrame(feats)], axis=1)
    df.to_csv(os.path.join(out, 'candidates.csv'), index=False)
    manifest = dict(candidates=Counter(df.source), dropped=Counter(df.reason[df.reason != '']))

    ok = df[df.reason == ''].copy()
    ok['priority'] = ok.source.map(SOURCE_PRIORITY.index)
    ok = ok.sort_values('priority').drop_duplicates('key_strict')
    tkeys = test_keys(os.path.join(args.root, 'data', 'test'))
    leak = ok.key_ligand.isin(tkeys)
    leaked = ok[leak].assign(test_set=ok.key_ligand[leak].map(tkeys))
    leaked.to_csv(os.path.join(out, 'leakage_removed.csv'), index=False)
    ok = ok[~leak]
    manifest.update(unique=Counter(ok.source), leakage_removed=Counter(leaked.test_set), test_keys=len(tkeys))

    fixed = ok[ok.source.isin(['catalysts', 'metallipodb'])]
    bulk = stratified(ok[ok.source.isin(['csd', 'tmqm'])], max(0, args.n_total - len(fixed)), rng)
    chosen = pd.concat([fixed, bulk]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    chosen['renders'] = np.where(chosen.source == 'catalysts', CATALYST_RENDERS, DEFAULT_RENDERS)
    val_pool = chosen[chosen.source != 'catalysts']
    val = stratified(val_pool, args.n_val, rng, floor=5)
    train = chosen.drop(val.index)
    val = val.assign(renders=1)
    cols = ['source', 'source_id', 'smiles', 'canonical', 'metals', 'primary_metal', 'n_heavy', 'has_eta',
            'multinuclear', 'renders', 'key_strict', 'key_ligand', 'doi', 'article_label']
    for name, part in (('train', train), ('val', val)):
        part = part.reset_index(drop=True)
        part.insert(0, 'pool_id', [f'{name[0]}{i:06d}' for i in range(len(part))])
        part[['pool_id'] + cols].to_csv(os.path.join(out, f'{name}.csv'), index=False)
        manifest[name] = dict(structures=len(part), renders=int(part.renders.sum()), sources=Counter(part.source),
                              eta=int(part.has_eta.sum()), multinuclear=int(part.multinuclear.sum()),
                              metals=Counter(part.primary_metal).most_common())
    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, default=int)
    print(json.dumps({k: manifest[k] for k in ('unique', 'leakage_removed', 'dropped')}, default=int))
    print('train', manifest['train']['structures'], 'val', manifest['val']['structures'])


if __name__ == '__main__':
    main()
