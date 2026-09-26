"""Organic data: replay SMILES for on-the-fly Indigo rendering, and a real organic validation set.

usage:
  python -m metal_ocsr.organic --root /mnt/hard1/ivans_data/metal_ocsr \
      --pubchem /mnt/hard1/pubchem_db/pubchem_slice/pubchem_compounds.csv --n 200000 \
      --uspto_zip /mnt/hard1/ivans_data/metal_ocsr/data/raw/uspto_real/real.zip --n_val 1000

writes  root/data/organic/train_organic.csv   SMILES (rendered by MolScribe's dynamic Indigo pipeline)
        root/data/organic/val_uspto_1k.csv    image_id, file_path (relative to root/data), SMILES
        root/data/organic/uspto_val/*.png
The replay keeps the organic skills of the pretrained model while it learns metals; the USPTO subset is only used to
select checkpoints (the organic test set is the separate general-500 benchmark).
"""
import argparse
import os
import random
import zipfile
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem, RDLogger

from molscribe.constants import METALS

RDLogger.DisableLog('rdApp.*')


def keep(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not 6 <= mol.GetNumHeavyAtoms() <= 70:
        return None
    if any(a.GetSymbol() in METALS for a in mol.GetAtoms()) or len(Chem.GetMolFrags(mol)) > 2:
        return None
    return Chem.MolToSmiles(mol)


def sample_lines(path, k, seed):
    """Reservoir sample of k data lines (the file has ~34M lines)."""
    rng = random.Random(seed)
    out = []
    with open(path, encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            if i < k:
                out.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    out[j] = line
    return [line.rstrip('\n').split(',', 1)[1] for line in out if ',' in line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--pubchem', required=True)
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--uspto_zip', default=None)
    ap.add_argument('--n_val', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--out_name', default='train_organic.csv')
    ap.add_argument('--skip_val', action='store_true')
    args = ap.parse_args()
    out = os.path.join(args.root, 'data', 'organic')
    os.makedirs(os.path.join(out, 'uspto_val'), exist_ok=True)

    candidates = sample_lines(args.pubchem, int(args.n * 1.3), args.seed)
    with Pool(args.workers) as p:
        smiles = [s for s in p.map(keep, candidates, chunksize=512) if s]
    smiles = list(dict.fromkeys(smiles))[:args.n]
    pd.DataFrame(dict(SMILES=smiles)).to_csv(os.path.join(out, args.out_name), index=False)
    print('organic replay', len(smiles))
    if args.skip_val:
        return

    with zipfile.ZipFile(args.uspto_zip) as z:
        df = pd.read_csv(z.open('real/USPTO.csv'))
        df = df.sample(args.n_val, random_state=args.seed).reset_index(drop=True)
        paths = []
        for fp in df.file_path:
            name = os.path.basename(fp)
            with open(os.path.join(out, 'uspto_val', name), 'wb') as f:
                f.write(z.read(fp))
            paths.append(os.path.join('organic', 'uspto_val', name))
    df['file_path'] = paths
    df[['image_id', 'file_path', 'SMILES']].to_csv(os.path.join(out, 'val_uspto_1k.csv'), index=False)
    print('uspto val', len(df))


if __name__ == '__main__':
    main()
