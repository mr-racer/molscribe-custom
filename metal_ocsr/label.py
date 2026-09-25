"""Source (dative) SMILES -> drawn-form label molecule, and MolScribe label outputs.

The label graph is what the picture shows (see molscribe/metal.py for the two conventions):
  * metal-ligand dative bonds become plain single bonds; an anionic donor loses the charge that only exists because
    of the dative convention ([Cl-]->[Pt+2] becomes Cl-Pt); a cationic complex may show its charge on the metal;
  * every ring whose atoms are all bonded to one metal (eta ring) becomes a centroid pseudo-atom "Ct" in the ring
    centre bonded to the metal; the ring loses its charges (Cp- is drawn neutral);
  * selected groups are condensed into text labels (CO, PPh3, Me, tBu ...), tokens the fork expands again.
Hydrogen counts are frozen from the source molecule, so none of these edits changes an atom's H count.

Atoms of the working molecule ("draft", an RWMol with a 2D conformer) carry:
  kind   'atom' | 'ct' | 'abbr'
  src    index in the source molecule (-1 for added centroids)
  token  label token for pseudo-atoms ('Ct', 'CO', 'Me' ...)
  atomLabel  text RDKit draws for pseudo-atoms ('' for Ct)
Bonds carry the int property 'draw': the Kekule bond order used for drawing (labels keep aromatic bonds).
"""
import re
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from molscribe.constants import LIGAND_SMILES
from molscribe.metal import CENTROID, _organic_rings, is_metal

DATIVE_TYPES = {Chem.BondType.DATIVE, Chem.BondType.DATIVEONE, Chem.BondType.DATIVEL, Chem.BondType.DATIVER,
                Chem.BondType.ZERO}
MAX_LABEL_TOKENS = 480  # FORMAT_INFO['chartok_coords']['max_len']


class Skip(Exception):
    """The structure cannot be turned into a training sample; the message is the reason (counted in the manifest)."""


# --------------------------------------------------------------------------------------------------------------------
#  source
# --------------------------------------------------------------------------------------------------------------------
def load_source(smiles):
    """Largest metal-containing fragment, sanitized, without isotopes and without stereo at metals."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Skip('parse')
    # a metal without neighbours (PubChem-style "[Fe].C1=CC=C[CH]1...") carries no drawable coordination
    frags = [f for f in Chem.GetMolFrags(mol, asMols=True)
             if any(is_metal(a) and a.GetDegree() > 0 for a in f.GetAtoms())]
    if not frags:
        raise Skip('no_bonded_metal')
    mol = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    for a in mol.GetAtoms():
        a.SetIsotope(0)
        if is_metal(a):
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return mol


def eta_rings(mol):
    """(metal, ring) for every ring all of whose atoms are bonded to that metal."""
    rings = _organic_rings(mol)
    out = []
    for a in mol.GetAtoms():
        if is_metal(a):
            nb = {n.GetIdx() for n in a.GetNeighbors()}
            out.extend((a.GetIdx(), r) for r in rings if set(r) <= nb)
    return out


# --------------------------------------------------------------------------------------------------------------------
#  draft
# --------------------------------------------------------------------------------------------------------------------
def _set_draw(bond, order):
    bond.SetIntProp('draw', int(order))


def _readd(rw, begin, end, btype, draw):
    """(Re)create a bond with a given begin atom; RDKit draws wedges from the begin atom."""
    if rw.GetBondBetweenAtoms(begin, end) is not None:
        rw.RemoveBond(begin, end)
    rw.AddBond(begin, end, btype)
    b = rw.GetBondBetweenAtoms(begin, end)
    _set_draw(b, draw)
    return b


def make_draft(src, depicted, show_charge):
    """Drawn-form working molecule. `depicted` is `src` with a 2D conformer (same atom order).

    Returns (draft, centroids) where centroids is a list of (ct index, metal index, ring atoms).
    """
    if depicted.GetNumAtoms() != src.GetNumAtoms() or depicted.GetNumConformers() == 0:
        raise Skip('depiction')
    kek = Chem.Mol(src)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        raise Skip('kekulize')

    rw = Chem.RWMol(depicted)
    hs = [a.GetTotalNumHs() for a in src.GetAtoms()]
    net = sum(a.GetFormalCharge() for a in src.GetAtoms())
    for a in rw.GetAtoms():
        a.SetIntProp('src', a.GetIdx())
        a.SetProp('kind', 'atom')
    for b in rw.GetBonds():
        kb = kek.GetBondBetweenAtoms(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        _set_draw(b, 1 if kb.GetBondType() in DATIVE_TYPES else kb.GetBondTypeAsDouble())

    etas = eta_rings(src)
    if max(Counter(r for _, r in etas).values(), default=0) > 1:
        raise Skip('ring_on_two_metals')
    conf = rw.GetConformer()
    eta_atoms, centroids = set(), []
    for m, ring in etas:
        for i in ring:
            rw.RemoveBond(i, m)
        eta_atoms.update(ring)
        xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in ring]).mean(axis=0)
        dummy = Chem.Atom(0)
        dummy.SetNoImplicit(True)
        ct = rw.AddAtom(dummy)
        conf = rw.GetConformer()
        conf.SetAtomPosition(ct, Point3D(float(xy[0]), float(xy[1]), 0.0))
        a = rw.GetAtomWithIdx(ct)
        a.SetIntProp('src', -1)
        a.SetProp('kind', 'ct')
        a.SetProp('token', CENTROID)
        a.SetProp('atomLabel', '')
        _readd(rw, m, ct, Chem.BondType.SINGLE, 1)
        centroids.append((ct, m, tuple(ring)))

    # remaining metal bonds: dative -> single with the metal as begin atom, anionic donors neutralised
    for b in list(rw.GetBonds()):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ai, aj = rw.GetAtomWithIdx(i), rw.GetAtomWithIdx(j)
        if not (is_metal(ai) or is_metal(aj)) or aj.GetProp('kind') == 'ct' or ai.GetProp('kind') == 'ct':
            continue
        btype, draw = b.GetBondType(), b.GetIntProp('draw')
        if is_metal(ai) and is_metal(aj):
            if btype in DATIVE_TYPES:
                _readd(rw, i, j, Chem.BondType.SINGLE, 1)
            continue
        m, x = (i, j) if is_metal(ai) else (j, i)
        if btype in DATIVE_TYPES:
            _readd(rw, m, x, Chem.BondType.SINGLE, 1)
            donor = rw.GetAtomWithIdx(x)
            if donor.GetFormalCharge() < 0:
                donor.SetFormalCharge(donor.GetFormalCharge() + 1)
        elif i != m:
            _readd(rw, m, x, btype, draw)

    metals = [a for a in rw.GetAtoms() if is_metal(a)]
    for a in rw.GetAtoms():
        if a.GetIdx() in eta_atoms:
            a.SetFormalCharge(0)
        # a drawn carbonyl is M-C#O without charges; the dative form leaves [O+] behind
        if a.GetSymbol() == 'O' and a.GetFormalCharge() == 1 and any(
                b.GetBondType() == Chem.BondType.TRIPLE and
                any(is_metal(n) for n in b.GetOtherAtom(a).GetNeighbors()) for b in a.GetBonds()):
            a.SetFormalCharge(0)
        if a.GetProp('kind') == 'atom':
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(hs[a.GetIntProp('src')])
            a.SetNumRadicalElectrons(0)
    for a in metals:
        a.SetFormalCharge(0)
    if show_charge and len(metals) == 1:
        residual = sum(a.GetFormalCharge() for a in rw.GetAtoms())
        if net - residual > 0:
            metals[0].SetFormalCharge(net - residual)
    return rw, centroids


# --------------------------------------------------------------------------------------------------------------------
#  abbreviations
# --------------------------------------------------------------------------------------------------------------------
# token, SMARTS, indices of the group atoms, index of the anchor (becomes the label atom), display variants
ORGANIC_GROUPS = [
    ('tBu', '[CX4;!R]([CH3])([CH3])([CH3])-[*]', (0, 1, 2, 3), 0, ['tBu', 't-Bu', '<sup>t</sup>Bu']),
    ('iPr', '[CH1;X4;!R]([CH3])([CH3])-[*]', (0, 1, 2), 0, ['iPr', 'i-Pr', '<sup>i</sup>Pr']),
    ('CF3', '[CX4;!R](F)(F)(F)-[*]', (0, 1, 2, 3), 0, ['CF<sub>3</sub>']),
    ('Ph', '[*]-[c;R1]1[cH;R1][cH;R1][cH;R1][cH;R1][cH;R1]1', (1, 2, 3, 4, 5, 6), 1, ['Ph']),
    ('OMe', '[OX2;!R]([CH3])-[#6]', (0, 1), 0, ['OMe']),
    ('Et', '[CH2;X4;!R]([CH3])-[*]', (0, 1), 0, ['Et']),
    ('Me', '[CH3;X4]-[*]', (0,), 0, ['Me']),
]
ORGANIC_PATTERNS = [(tok, Chem.MolFromSmarts(s), g, a, d) for tok, s, g, a, d in ORGANIC_GROUPS]

LIGAND_DISPLAY = {'CO': ['CO'], 'PPh3': ['PPh<sub>3</sub>'], 'PCy3': ['PCy<sub>3</sub>'], 'PMe3': ['PMe<sub>3</sub>'],
                  'PEt3': ['PEt<sub>3</sub>'], 'PtBu3': ['P<sup>t</sup>Bu<sub>3</sub>'],
                  'PiPr3': ['P<sup>i</sup>Pr<sub>3</sub>'], 'MeCN': ['MeCN', 'NCMe'], 'py': ['py'],
                  'DMSO': ['DMSO'], 'THF': ['THF']}
LIGAND_P = {'CO': 0.7}  # everything else in LIGAND_DISPLAY uses p_ligand


def _ligand_table():
    table = {}
    for tok in LIGAND_DISPLAY:
        lig = Chem.MolFromSmiles(LIGAND_SMILES[tok])
        table[Chem.MolToSmiles(lig)] = (tok, lig.GetAtomWithIdx(0).GetSymbol())
    return table


LIGANDS = _ligand_table()


def _ligand_candidates(src):
    """(token, donor atom, fragment atoms) for whole ligands that match a LIGAND_SMILES entry."""
    rw = Chem.RWMol(src)
    donors = {}
    for b in list(rw.GetBonds()):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if is_metal(rw.GetAtomWithIdx(i)) != is_metal(rw.GetAtomWithIdx(j)):
            x = j if is_metal(rw.GetAtomWithIdx(i)) else i
            donors[x] = donors.get(x, 0) + 1
            rw.RemoveBond(i, j)
    out = []
    frags = Chem.GetMolFrags(rw, sanitizeFrags=False)
    for frag in frags:
        frag_donors = [x for x in frag if x in donors]
        if len(frag_donors) != 1 or donors[frag_donors[0]] != 1:
            continue
        smi = Chem.MolFragmentToSmiles(src, atomsToUse=list(frag))
        key = Chem.MolFromSmiles(smi)
        if key is None:
            continue
        hit = LIGANDS.get(Chem.MolToSmiles(key))
        if hit and src.GetAtomWithIdx(frag_donors[0]).GetSymbol() == hit[1]:
            out.append((hit[0], frag_donors[0], tuple(frag)))
    return out


def choose_abbreviations(src, draft, centroids, rng, rate_organic, p_ligand):
    """Pick non-overlapping groups to condense: list of (token, anchor, atoms, display)."""
    blocked = {i for _, m, ring in centroids for i in ring}
    blocked |= {a.GetIdx() for a in src.GetAtoms() if is_metal(a)}
    chosen, used = [], set()
    for tok, donor, atoms in _ligand_candidates(src):
        if set(atoms) & (used | blocked) or rng.random() >= LIGAND_P.get(tok, p_ligand):
            continue
        chosen.append((tok, donor, atoms, str(rng.choice(LIGAND_DISPLAY[tok]))))
        used.update(atoms)
    for tok, patt, group, anchor, display in ORGANIC_PATTERNS:
        for match in src.GetSubstructMatches(patt):
            atoms = tuple(match[k] for k in group)
            if set(atoms) & (used | blocked) or rng.random() >= rate_organic:
                continue
            chosen.append((tok, match[anchor], atoms, str(rng.choice(display))))
            used.update(atoms)
    return chosen


def condense(draft, chosen):
    """Turn each chosen group into one labelled pseudo-atom at the anchor position; returns atoms to delete."""
    delete = []
    for tok, anchor, atoms, display in chosen:
        a = draft.GetAtomWithIdx(anchor)
        a.SetAtomicNum(0)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetIsAromatic(False)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        a.SetProp('kind', 'abbr')
        a.SetProp('token', tok)
        a.SetProp('atomLabel', display)
        inside = set(atoms)
        for b in a.GetBonds():
            if b.GetOtherAtomIdx(anchor) not in inside:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
                _set_draw(b, 1)
        delete.extend(i for i in atoms if i != anchor)
    return delete


def delete_atoms(draft, atoms):
    for i in sorted(set(atoms), reverse=True):
        draft.RemoveAtom(i)


# --------------------------------------------------------------------------------------------------------------------
#  label outputs
# --------------------------------------------------------------------------------------------------------------------
_PSEUDO = re.compile(r'\[(\d+)\*\]')


def edge_type(bond):
    d = bond.GetBondDir()
    if d == Chem.BondDir.BEGINWEDGE:
        return 5
    if d == Chem.BondDir.BEGINDASH:
        return 6
    if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
        return 4
    return {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 1)


def label_outputs(draft):
    """(label SMILES, atom order, edges) in MolScribe's aux format.

    The SMILES lists atoms in `order` (draft indices); pseudo-atoms are written as [token]. Edges are
    [u, v, t] with positions in that order: t = 1/2/3 single/double/triple, 4 aromatic, 5/6 wedge/hash starting at u.
    """
    m = Chem.RWMol(draft)
    tokens = {}
    for a in m.GetAtoms():
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if a.GetAtomicNum() == 0:
            iso = 500 + len(tokens)
            tokens[iso] = a.GetProp('token')
            a.SetIsotope(iso)
    for b in m.GetBonds():
        b.SetStereo(Chem.BondStereo.STEREONONE)
        b.SetBondDir(Chem.BondDir.NONE)
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    smiles = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
    order = [int(i) for i in m.GetProp('_smilesAtomOutputOrder').strip('[]').split(',') if i != '']
    smiles = _PSEUDO.sub(lambda g: f'[{tokens[int(g.group(1))]}]', smiles)
    pos = {atom: k for k, atom in enumerate(order)}
    edges = []
    for b in draft.GetBonds():
        u, v = pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()]
        edges.append([u, v, edge_type(b)])
    return smiles, order, edges


def check_label(smiles, n_atoms):
    """The MolScribe tokenizer must see exactly one atom token per atom, and the sequence must fit."""
    from SmilesPE.pretokenizer import atomwise_tokenizer
    tokens = atomwise_tokenizer(smiles)
    atoms = sum(1 for t in tokens if t.isalpha() or t.startswith('[') or t == '*')
    if atoms != n_atoms:
        raise Skip('token_mismatch')
    if len(smiles) + 2 * n_atoms + 2 > MAX_LABEL_TOKENS:
        raise Skip('too_long')
