"""Ground truth for T1: the 21 `publication_metal` images of lebedev_bench (3 PMC papers).

The dataset's own target_smiles are PubChem-style placeholders (disconnected, trindane written as benzene, carbonyls
bound through O), so every image was re-annotated from the picture. Gold is the RDKit dative form (xyz2mol_tm /
MetalLipoDB convention): donor->metal bonds, anionic ligands charged, eta rings as dative bonds from every ring atom.
Only the drawn complex counts: arrows, reagents ("+ 2NaCl"), compound numbers and stoichiometric factors do not.

Four images (0222-0225) draw substituents with a variable ring position and "●" end groups (Markush); they have no
single structure and are excluded from the metrics (include=0).

usage: python -m metal_ocsr.testsets.t1_lebedev_gt --images <lebedev_bench/dataset/images> --out <data/test/t1_lebedev_metal>
"""
import argparse
import glob
import os
import shutil

import pandas as pd
from rdkit import Chem

CO = '[C-]#[O+]'
TRINDANE = 'C1CC2=C3CCCC3=C4CCCC4=C2C1'
CP = '[cH-]1cccc1'
DTBPY = 'CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1'


def _donors(frag, donors):
    if donors == 'ring6':
        return next(r for r in frag.GetRingInfo().AtomRings() if len(r) == 6)
    if donors == 'ring5':
        return next(r for r in frag.GetRingInfo().AtomRings() if len(r) == 5)
    if donors == 'aromatic_n':
        return tuple(a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() == 'N' and a.GetIsAromatic())
    if donors == 'ring6_no_exo':
        ring = next(r for r in frag.GetRingInfo().AtomRings() if len(r) == 6)
        exo = {i for i in ring for b in frag.GetAtomWithIdx(i).GetBonds()
               if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtomIdx(i) not in ring}
        return tuple(i for i in ring if i not in exo)
    return donors


def build(metal, ligands):
    """metal SMILES + (ligand SMILES, donors) pairs; every donor atom gets a dative bond to the metal (atom 0)."""
    mol = Chem.RWMol(Chem.MolFromSmiles(metal))
    for smi, donors in ligands:
        frag = Chem.MolFromSmiles(smi)
        off = mol.GetNumAtoms()
        mol = Chem.RWMol(Chem.CombineMols(mol, frag))
        for i in _donors(frag, donors):
            mol.AddBond(off + i, 0, Chem.BondType.DATIVE)
    out = mol.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out)


def mapped(smiles, links):
    """Atom-mapped SMILES; links = (donor map, acceptor map) become dative bonds; maps are cleared."""
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    for d, a in links:
        mol.AddBond(idx[d], idx[a], Chem.BondType.DATIVE)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    out = mol.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out)


def ferrocene_unit(off, ring_smiles):
    """Fe(II) (map off) + plain Cp (maps off+1..5) + a substituted ring whose five ring atoms carry maps
    off+11..off+15 in `ring_smiles` (a format string with {a}..{e} for those maps). Returns (smiles, links)."""
    fe = f'[Fe+2:{off}]'
    cp = ''.join([f'[cH-:{off + 1}]1'] + [f'[cH:{off + k}]' for k in range(2, 6)] + ['1'])
    ring = ring_smiles.format(**{c: off + 11 + k for k, c in enumerate('abcde')})
    links = [(off + k, off) for k in range(1, 6)] + [(off + 11 + k, off) for k in range(5)]
    return '.'.join([fe, cp, ring]), links


def gt_0229():
    """[Pd(C,N-ferrocenyl SAMP-hydrazone)(mu-NO2-kN:kO)]2 (compound 30)."""
    parts, links = [], []
    for off, pd in ((100, 130), (200, 230)):
        ring = ('[c:{a}]1([Pd+:%d])[cH-:{b}][cH:{c}][cH:{d}][c:{e}]1/C(C)=[N:%d]/N1CCCC1COC' % (pd, off + 26))
        smi, lk = ferrocene_unit(off, ring)
        parts.append(smi)
        links += lk + [(off + 26, pd)]
    parts += ['O=[N:40][O-:41]', 'O=[N:42][O-:43]']
    links += [(40, 130), (41, 230), (42, 230), (43, 130)]
    return mapped('.'.join(parts), links)


def gt_0231():
    """trans-PdCl2(ferrocenyl-CH=N-CH(CH3)CH2OH)2 (compound 31)."""
    parts, links = ['[Pd+2:50].[Cl-:51].[Cl-:52]'], [(51, 50), (52, 50)]
    for off in (100, 200):
        smi, lk = ferrocene_unit(off, '[cH-:{a}]1[cH:{b}][cH:{c}][cH:{d}][c:{e}]1/C=[N:%d]/C(C)CO' % (off + 26))
        parts.append(smi)
        links += lk + [(off + 26, 50)]
    return mapped('.'.join(parts), links)


def gt_0227():
    """Pt(dtbpy)(1-(2-pyridylmethyl)-4-phenyl-1,2,3-triazol-5-yl)2; the image writes the second triazolyl as [ ]2."""
    triazolyl = '[c-]1c(-c2ccccc2)nnn1Cc1ccccn1'
    return build('[Pt+2]', [(DTBPY, 'aromatic_n'), (triazolyl, (0,)), (triazolyl, (0,))])


GT = {
    '0211': (build('[Cr]', [(TRINDANE, 'ring6')] + [(CO, (0,))] * 3), 'eta6'),
    '0212': (build('[Mo]', [(TRINDANE, 'ring6')] + [(CO, (0,))] * 3), 'eta6'),
    '0213': (build('[Fe+2]', [(TRINDANE, 'ring6'), (CP, 'ring5')]), 'eta6,eta5,charge'),
    '0214': (build('[Mn+]', [(TRINDANE, 'ring6')] + [(CO, (0,))] * 3), 'eta6,charge'),
    '0215': (build('[Fe+2]', [('Cc1c(C)c(C)c(C)c(C)c1C', 'ring6'), (CP, 'ring5')]), 'eta6,eta5,charge'),
    # 48: eta5-(6-methylene-pentamethylcyclohexadienyl), drawn with a partial circle
    '0216': (build('[Fe+2]', [('C=C1C(C)=C(C)[C-](C)C(C)=C1C', 'ring6_no_exo'), (CP, 'ring5')]), 'eta5_on_6ring,eta5'),
    '0217': (build('[Fe+2]', [('CCc1c(C)c(C)c(C)c(C)c1C', 'ring6'), (CP, 'ring5')]), 'eta6,eta5,charge'),
    '0218': (build('[Fe+2]', [('CCc1c(CC)c(CC)c(CC)c(CC)c1CC', 'ring6'), (CP, 'ring5')]), 'eta6,eta5,charge'),
    '0219': (build('[Mn+]', [('Cc1c(C)c(C)c(C)c(C)c1C', 'ring6')] + [(CO, (0,))] * 3), 'eta6,charge'),
    '0220': (build('[Fe+2]', [(TRINDANE, 'ring6'), (CP, 'ring5')]), 'eta6,eta5,charge'),
    '0221': (build('[Mn+]', [(TRINDANE, 'ring6')] + [(CO, (0,))] * 3), 'eta6,charge'),
    '0222': ('', 'markush'),
    '0223': ('', 'markush'),
    '0224': ('', 'markush'),
    '0225': ('', 'markush'),
    '0226': (build('[Pt+2]', [(DTBPY, 'aromatic_n'), ('[C-]#Cc1ccccc1', (0,)), ('[C-]#Cc1ccccc1', (0,))]), 'sigma'),
    '0227': (gt_0227(), 'sigma,bracket_repeat'),
    '0228': (build('[Fe+2]', [('CC(=O)c1cc[cH-]c1', 'ring5'), (CP, 'ring5')]), 'eta5'),
    '0229': (gt_0229(), 'eta5,sigma,multinuclear'),
    '0230': (build('[Fe+2]', [('OCC(C)/N=C/c1cc[cH-]c1', 'ring5'), (CP, 'ring5')]), 'eta5'),
    '0231': (gt_0231(), 'eta5,sigma,multinuclear'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='lebedev_bench/dataset/images')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    rows = []
    for key, (gold, tags) in sorted(GT.items()):
        src = glob.glob(os.path.join(args.images, f'{key}_publication_metal_*'))[0]
        name = os.path.basename(src)
        shutil.copy(src, os.path.join(args.out, 'images', name))
        rows.append(dict(id=name, source='lebedev', image=f'images/{name}', gold=gold, tags=tags,
                         include=int('markush' not in tags)))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'gt.csv'), index=False)
    print(len(rows), 'rows,', sum(r['include'] for r in rows), 'scored')


if __name__ == '__main__':
    main()
