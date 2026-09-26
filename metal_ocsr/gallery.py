"""Side-by-side galleries: original image | gold | base model | fine-tuned model, structures drawn with metal2d.

usage:
  python -m metal_ocsr.gallery --root /mnt/hard1/ivans_data/metal_ocsr --set t1_lebedev_metal \
      --ckpt runs/E1/swin_base_transformer_ep5.pth --out eval/E1/galleries

Predictions go through the current post-processing. When it still returns '<invalid>', the panel shows the raw graph
the model predicted (its own atom positions, symbols and bond orders), so a panel always shows what the model produced.
Rows are sorted into sheets by outcome: e1_correct, improved (base wrong -> fine-tuned right), regressed, wrong;
index.csv lists every row with both predictions and flags.
"""
import argparse
import os
import signal
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from molscribe.metal import metal_key, replace_rgroups

RDLogger.DisableLog('rdApp.*')
PANEL = 300
TMP = tempfile.mkdtemp(prefix='gallery_')


def _alarm(signum, frame):
    raise TimeoutError()


def _init():
    RDLogger.DisableLog('rdApp.*')
    import logging
    logging.getLogger('metal2d').setLevel(logging.ERROR)
    signal.signal(signal.SIGALRM, _alarm)


def _blank(text, colour=(120, 120, 120)):
    img = np.full((PANEL, PANEL, 3), 255, np.uint8)
    for i in range(0, min(len(text), 5 * 40), 40):
        cv2.putText(img, text[i:i + 40], (6, 140 + i // 40 * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)
    return img


def _draw_metal2d(smiles):
    import metal2d
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
    path = os.path.join(TMP, f'{os.getpid()}.png')
    metal2d.draw(metal2d.depict(mol), path, size=(PANEL, PANEL))
    return cv2.imread(path)


def _draw_graph(atoms, bonds):
    """Raw predicted graph: the model's own atom positions, symbols as labels, bond orders as predicted."""
    if not atoms:
        return None
    from rdkit.Geometry import Point3D
    mol = Chem.RWMol()
    for a in atoms:
        sym = a['atom_symbol'].strip('[]')
        atom = Chem.Atom(6) if sym in ('C', 'c') else Chem.Atom(0)
        atom.SetNoImplicit(True)
        idx = mol.AddAtom(atom)
        if sym not in ('C', 'c'):
            mol.GetAtomWithIdx(idx).SetProp('atomLabel', sym)
    types = {'single': Chem.BondType.SINGLE, 'double': Chem.BondType.DOUBLE, 'triple': Chem.BondType.TRIPLE,
             'aromatic': Chem.BondType.AROMATIC}
    for b in bonds:
        i, j = b['endpoint_atoms']
        mol.AddBond(int(i), int(j), types.get(b['bond_type'], Chem.BondType.SINGLE))
        if 'wedge' in b['bond_type']:
            mol.GetBondBetweenAtoms(int(i), int(j)).SetBondDir(
                Chem.BondDir.BEGINDASH if 'dash' in b['bond_type'] else Chem.BondDir.BEGINWEDGE)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for k, a in enumerate(atoms):
        conf.SetAtomPosition(k, Point3D(float(a['x']), -float(a['y']), 0.0))
    mol.AddConformer(conf)
    m = mol.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(m)
    d = rdMolDraw2D.MolDraw2DCairo(PANEL, PANEL)
    d.drawOptions().prepareMolsBeforeDrawing = False
    d.DrawMolecule(m)
    d.FinishDrawing()
    return cv2.imdecode(np.frombuffer(d.GetDrawingText(), np.uint8), cv2.IMREAD_COLOR)


def render(job):
    """job = (smiles, atoms, bonds) -> (panel image, how it was drawn)."""
    smiles, atoms, bonds = job
    signal.alarm(30)
    try:
        if isinstance(smiles, str) and smiles and smiles != '<invalid>':
            img = _draw_metal2d(replace_rgroups(smiles))
            if img is not None:
                return img, 'smiles'
        if atoms:
            img = _draw_graph(atoms, bonds)
            if img is not None:
                return img, 'raw graph (post-processing failed)'
        return _blank(f'cannot draw: {smiles}'), 'none'
    except Exception as e:
        return _blank(f'{type(e).__name__}: {smiles}'), 'none'
    finally:
        signal.alarm(0)


def render_all(jobs, workers):
    out = [None] * len(jobs)
    with Pool(workers, initializer=_init, maxtasksperchild=200) as p:
        pending = [p.apply_async(render, (j,)) for j in jobs]
        for i, r in enumerate(pending):
            try:
                out[i] = r.get(timeout=120)
            except Exception:
                out[i] = (_blank(f'render crashed: {jobs[i][0]}'), 'none')
    return out


def panel(img, title, ok=None, note=''):
    h, w = img.shape[:2]
    k = min((PANEL - 4) / w, (PANEL - 30) / h)
    img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    out = np.full((PANEL + 40, PANEL, 3), 255, np.uint8)
    out[26:26 + img.shape[0], 2:2 + img.shape[1]] = img
    colour = (0, 0, 0) if ok is None else ((0, 140, 0) if ok else (0, 0, 210))
    mark = '' if ok is None else (' OK' if ok else ' WRONG')
    cv2.putText(out, title + mark, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    if note:
        cv2.putText(out, note[:48], (4, PANEL + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
    cv2.rectangle(out, (0, 0), (PANEL - 1, PANEL + 39), (200, 200, 200), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--set', required=True)
    ap.add_argument('--ckpt', required=True, help='fine-tuned checkpoint (relative to root or absolute)')
    ap.add_argument('--base_ckpt', default='ckpts/base/swin_base_char_aux_1m680k.pth')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows_per_sheet', type=int, default=8)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    root = args.root
    base_dir = os.path.join(root, 'data', 'test', args.set)
    gt = pd.read_csv(os.path.join(base_dir, 'gt.csv'), keep_default_na=False)
    out = os.path.join(root, args.out, args.set)
    os.makedirs(out, exist_ok=True)

    import torch
    from molscribe import MolScribe
    images = [cv2.cvtColor(cv2.imdecode(np.fromfile(os.path.join(base_dir, p), dtype=np.uint8), 1), cv2.COLOR_BGR2RGB)
              for p in gt.image]
    preds = {}
    for name, ck in (('base', args.base_ckpt), ('ft', args.ckpt)):
        model = MolScribe(ck if os.path.isabs(ck) else os.path.join(root, ck), device=torch.device('cuda'))
        preds[name] = model.predict_images(images, return_atoms_bonds=True, batch_size=32)
        del model
        torch.cuda.empty_cache()

    gold_ok = [bool(g) for g in gt.gold]
    jobs = [(g, None, None) for g in gt.gold]
    for name in ('base', 'ft'):
        jobs += [(p.get('smiles', ''), p.get('atoms'), p.get('bonds')) for p in preds[name]]
    drawn = render_all(jobs, args.workers)
    n = len(gt)
    gold_img, base_img, ft_img = drawn[:n], drawn[n:2 * n], drawn[2 * n:]

    rows = []
    for i, r in gt.iterrows():
        rec = dict(id=r.id, gold=r.gold, include=r.get('include', 1), tags=r.get('tags', ''))
        for name in ('base', 'ft'):
            p = preds[name][i].get('smiles', '')
            rec[f'pred_{name}'] = p
            rec[f'strict_{name}'] = gold_ok[i] and metal_key(p) is not None and metal_key(p) == metal_key(r.gold)
            rec[f'ligand_{name}'] = gold_ok[i] and metal_key(p, ligands_only=True) == metal_key(r.gold, ligands_only=True)
        rec['group'] = ('no_gold' if not gold_ok[i] else
                        'improved' if rec['strict_ft'] and not rec['strict_base'] else
                        'regressed' if rec['strict_base'] and not rec['strict_ft'] else
                        'e1_correct' if rec['strict_ft'] else 'wrong')
        rows.append(rec)
    index = pd.DataFrame(rows)
    index.to_csv(os.path.join(out, 'index.csv'), index=False)

    def row_image(i):
        rec = rows[i]
        orig = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        lig = lambda k: 'ligands OK' if rec[f'ligand_{k}'] else ''
        return np.hstack([
            panel(orig, rec['id'][:30]),
            panel(gold_img[i][0] if gold_ok[i] else _blank('no gold (Markush / excluded)'), 'gold'),
            panel(base_img[i][0], 'base (E0)', rec['strict_base'] if gold_ok[i] else None,
                  f"{lig('base')} {base_img[i][1] if base_img[i][1] != 'smiles' else ''}"),
            panel(ft_img[i][0], 'fine-tuned (E1)', rec['strict_ft'] if gold_ok[i] else None,
                  f"{lig('ft')} {ft_img[i][1] if ft_img[i][1] != 'smiles' else ''}"),
        ])

    for group in ['improved', 'e1_correct', 'regressed', 'wrong', 'no_gold']:
        idx = [i for i, r in enumerate(rows) if r['group'] == group]
        for s in range(0, len(idx), args.rows_per_sheet):
            sheet = np.vstack([row_image(i) for i in idx[s:s + args.rows_per_sheet]])
            cv2.imwrite(os.path.join(out, f'{group}_{s // args.rows_per_sheet:02d}.png'), sheet)
    summary = index[index.group != 'no_gold']
    print(args.set, 'n', len(summary), 'base', round(summary.strict_base.mean(), 4), 'ft', round(summary.strict_ft.mean(), 4),
          dict(index.group.value_counts()))


if __name__ == '__main__':
    main()
