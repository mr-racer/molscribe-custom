"""Organometallic helpers for post-processing and evaluation.

Two structure conventions meet here:

source form  RDKit dative SMILES as in xyz2mol_tm / MetalLipoDB: donor->metal bonds, ligand charges written out
             ([Cl-]->[Pt+2]), an eta-bonded ring as dative bonds from every ring atom to the metal.
drawn form   what a figure shows and what MolScribe predicts: plain metal-ligand lines, no charges that only exist
             because of the dative convention, and an eta-bonded ring drawn as one line from the metal to a
             centroid pseudo-atom "Ct" placed inside the ring.

`expand_centroids` and `fix_cyclopentadienyl` turn a predicted drawn-form graph into the source form;
`metal_key` normalises either form so that predictions and ground truth can be compared.
"""
import re

from rdkit import Chem

from .constants import METALS

CENTROID = 'Ct'
MAX_RING = 8


def is_metal(atom):
    return atom.GetSymbol() in METALS


def is_centroid(atom):
    if atom.GetAtomicNum() != 0:
        return False
    alias = Chem.GetAtomAlias(atom) or (atom.GetProp('molFileAlias') if atom.HasProp('molFileAlias') else '')
    return alias == CENTROID


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for k in range(n):
        x1, y1 = polygon[k]
        x2, y2 = polygon[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _organic_rings(mol):
    """Rings (3-8 atoms) of the molecule with all bonds to metals removed, in ring order.

    Ring perception on the full graph is useless for eta complexes: a metal bonded to every atom of a Cp ring closes
    many 3-membered M-C-C cycles and the Cp ring itself can drop out of the ring basis.
    """
    rw = Chem.RWMol(mol)
    for b in list(rw.GetBonds()):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if is_metal(rw.GetAtomWithIdx(i)) or is_metal(rw.GetAtomWithIdx(j)):
            rw.RemoveBond(i, j)
    rw.UpdatePropertyCache(strict=False)
    rings = Chem.GetSymmSSSR(rw)
    return [tuple(r) for r in rings
            if 3 <= len(r) <= MAX_RING and not any(is_centroid(rw.GetAtomWithIdx(i)) for i in r)]


def expand_centroids(mol, coords):
    """Replace every Ct pseudo-atom bonded to a metal by dative bonds from the atoms of the ring that contains it.

    mol: molecule built from a predicted graph, atom i drawn at coords[i]. The ring is found geometrically: the
    smallest ring of the prediction (3-8 atoms, no metal) whose polygon contains the Ct position. A Ct outside every
    ring is left in place, so the prediction stays visibly wrong instead of being guessed.
    """
    centroids = [a.GetIdx() for a in mol.GetAtoms() if is_centroid(a)]
    if not centroids:
        return mol
    rings = _organic_rings(mol)
    mol = Chem.RWMol(mol)
    expanded = []
    for c in centroids:
        metals = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if is_metal(n)]
        inside = [ring for ring in rings if _point_in_polygon(coords[c], [coords[i] for i in ring])]
        if not metals or not inside:
            continue
        ring = min(inside, key=len)
        for m in metals:
            for i in ring:
                if mol.GetBondBetweenAtoms(i, m) is None:
                    mol.AddBond(i, m, Chem.BondType.DATIVE)
        _freeze_ring_hydrogens(mol, ring)
        expanded.append(c)
    for c in sorted(expanded, reverse=True):
        mol.RemoveAtom(c)
    return mol.GetMol()


def _freeze_ring_hydrogens(mol, ring):
    """Write explicit H counts on the carbons of an eta-bonded ring.

    RDKit cannot derive implicit hydrogens of an aromatic carbon that also carries a dative bond (the eta-Cp ring then
    fails to kekulize), so the count is set from the organic bonds only: aromatic C gets one H unless substituted,
    other C gets 4 minus its bond orders.
    """
    for i in ring:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() != 'C':
            continue
        organic = [b for b in atom.GetBonds() if not is_metal(b.GetOtherAtom(atom))]
        if atom.GetIsAromatic():
            substituents = sum(1 for b in organic if b.GetOtherAtomIdx(i) not in ring)
            h = max(0, 1 - substituents)
        else:
            h = max(0, 4 - int(sum(b.GetBondTypeAsDouble() for b in organic)))
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(h)


def _haptic_rings(mol, min_bonded=3):
    """(metal index, ring atoms) for rings of which at least `min_bonded` atoms are bonded to the same metal."""
    rings = _organic_rings(mol)
    out = []
    for atom in mol.GetAtoms():
        if not is_metal(atom):
            continue
        neighbours = {n.GetIdx() for n in atom.GetNeighbors()}
        for ring in rings:
            if len(neighbours & set(ring)) >= min_bonded:
                out.append((atom.GetIdx(), ring))
    return out


def fix_cyclopentadienyl(mol):
    """Write eta5-C5 rings the way the dative convention does: one [cH-] in the ring and +1 on the metal.

    A drawn Cp ring is either aromatic (circle) or has two double bonds and one sp3-looking carbon. As predicted it is
    neutral, which leaves an aromatic ring that cannot be kekulized or a CH2 that is not there. The fix is applied only
    when it produces a molecule RDKit can sanitize.
    """
    original = mol
    mol = Chem.RWMol(mol)
    changed = False
    for metal, ring in _haptic_rings(mol, min_bonded=5):
        if len(ring) != 5 or any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' or
                                 mol.GetAtomWithIdx(i).GetFormalCharge() != 0 for i in ring):
            continue
        ring_set = set(ring)
        in_ring = [mol.GetBondBetweenAtoms(i, j) for i in ring for j in ring if i < j
                   and mol.GetBondBetweenAtoms(i, j) is not None]
        if all(b.GetIsAromatic() or b.GetBondType() == Chem.BondType.AROMATIC for b in in_ring):
            for i in ring:
                mol.GetAtomWithIdx(i).SetIsAromatic(True)
            for b in in_ring:
                b.SetIsAromatic(True)
            target = min(ring, key=lambda i: mol.GetAtomWithIdx(i).GetDegree())
        else:
            no_double = [i for i in ring if not any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtomIdx(i) in ring_set
                for b in mol.GetAtomWithIdx(i).GetBonds())]
            if len(no_double) != 1:
                continue
            target = no_double[0]
            # the sp3-looking carbon was drawn as CH2 by valence; as the carbanion it is CH
            t = mol.GetAtomWithIdx(target)
            t.SetNoImplicit(True)
            t.SetNumExplicitHs(max(0, t.GetTotalNumHs() - 1))
        mol.GetAtomWithIdx(target).SetFormalCharge(-1)
        m = mol.GetAtomWithIdx(metal)
        m.SetFormalCharge(m.GetFormalCharge() + 1)
        changed = True
    if not changed:
        return original
    trial = mol.GetMol()
    try:
        Chem.SanitizeMol(Chem.Mol(trial))
    except Exception:
        return original
    return trial


# --------------------------------------------------------------------------------------------------------------------
#  evaluation keys
# --------------------------------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"(\[[^\]]+\]|Br|Cl|.)")


def replace_rgroups(smiles):
    """R1..Rn -> [n*], any other bracket token RDKit cannot read -> * (same rule as benchmark/evaluate.py)."""
    out = []
    for tok in _TOKEN.findall(smiles):
        if tok.startswith("[") and tok.endswith("]"):
            sym = tok[1:-1]
            if sym[:1] == "R" and sym[1:].isdigit():
                tok = f"[{sym[1:]}*]"
            elif Chem.AtomFromSmiles(tok) is None:
                tok = "*"
        out.append(tok)
    return "".join(out)


def metal_key(smiles, stereo=False, ligands_only=False):
    """Canonical key that ignores how metal-ligand bonds are written.

    Bonds to metals are removed; all formal charges are cleared (the dative convention moves charges onto ligand atoms
    that no figure shows: [C-]#[O+] vs C#O, [N+]=[C-] carbenes); hydrogens of non-metal atoms are recomputed from
    valence. Atoms of eta-bonded rings (>= 3 ring atoms bonded to one metal)
    additionally lose H counts, aromaticity and ring bond orders, because a drawn Cp / arene can be written as a
    circle, as alternating bonds or as a charged ring. Then the metal bonds are put back as single bonds, unless
    `ligands_only`, which compares the disconnected ligands and metals only.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(replace_rgroups(smiles), sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.RemoveHs(mol, sanitize=False)
        rw = Chem.RWMol(mol)
        eta = set()
        for _, ring in _haptic_rings(rw):
            eta.update(ring)
        metal_bonds = []
        for b in list(rw.GetBonds()):
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if is_metal(rw.GetAtomWithIdx(i)) or is_metal(rw.GetAtomWithIdx(j)):
                metal_bonds.append((i, j))
        for i, j in metal_bonds:
            rw.RemoveBond(i, j)
        for b in rw.GetBonds():
            if b.GetBeginAtomIdx() in eta and b.GetEndAtomIdx() in eta:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
        for a in rw.GetAtoms():
            a.SetNumRadicalElectrons(0)
            idx = a.GetIdx()
            a.SetFormalCharge(0)
            if is_metal(a) or a.GetAtomicNum() == 0 or idx in eta:
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
                a.SetIsAromatic(False)
            else:
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            rw.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(rw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                             catchErrors=True)
        if not ligands_only:
            for i, j in metal_bonds:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
            rw.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(rw, isomericSmiles=stereo)
    except Exception:
        return None
