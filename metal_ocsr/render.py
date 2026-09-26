"""Render one training sample: image + MolScribe label (SMILES, normalised atom coordinates, edges).

Pipeline (spec section 5):
  layout      metal2d.depict (CoordGen with p=0.15)
  draft       drawn-form label molecule (label.make_draft), centroids for eta rings
  geometry    perspective squash of eta rings, mirror, rotation: on 2D coordinates, so text stays upright
  wedges      organic stereo wedges from the source chirality; optional wedge/hash "legs" on metal-ligand bonds
  labels      condensed groups (CO, PPh3, Me, tBu ...)
  drawing     the user's standard RDKit style (MolDraw2DCairo, bondLineWidth 2, padding 0.10, minFontSize 9) in half
              of the samples, a varied style otherwise; aromatic circles drawn over chosen rings
  context     compound numbers, intrusions, charge brackets, frame lines (noise.add_context)
Load-time degradation (rotation +-5 deg, blur, JPEG ...) is applied by the training dataset, not here.
"""
import math

import cv2
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

from molscribe.metal import _organic_rings, is_metal
from . import label as L
from .noise import FONTS, add_context

STANDARD = dict(lw=2.0, padding=0.10, min_font=9, base_font=0.6, mbo=0.15, font=None, bw=False,
                explicit_methyl=False, highlight=False)
FONT_FILES = [f for f in FONTS['regular'] if 'Mono' not in f]


def sample_params(rng):
    p = dict(STANDARD)
    p['standard'] = bool(rng.random() < 0.5)
    if not p['standard']:
        p.update(lw=float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])),
                 min_font=int(rng.integers(7, 14)),
                 base_font=float(rng.uniform(0.45, 0.8)),
                 mbo=float(rng.uniform(0.1, 0.2)),
                 font=(str(rng.choice(FONT_FILES)) if FONT_FILES and rng.random() < 0.7 else None),
                 bw=bool(rng.random() < 0.4),
                 explicit_methyl=bool(rng.random() < 0.15),
                 highlight=bool(rng.random() < 0.1))
    p.update(bond_px=float(rng.uniform(15, 45)),
             coordgen=bool(rng.random() < 0.15),
             squash=0.5, mirror=bool(rng.random() < 0.3),
             rotate=(float(rng.uniform(0, 2 * math.pi)) if rng.random() < 0.7 else 0.0),
             metal_legs=bool(rng.random() < 0.3),
             circles=bool(rng.random() < 0.3),
             show_charge=bool(rng.random() < 0.5),
             rate_organic=float(rng.uniform(0.0, 0.7)), p_ligand=0.5,
             # 89 % of the structures have an R site, so p=0.34 gives ~30 % R-labelled images
             rgroup=bool(rng.random() < 0.34),
             arrows=bool(rng.random() < 0.1),
             metal_label=str(rng.choice(['plain', 'oxidation', 'bracket'], p=[0.77, 0.15, 0.08])),
             context=dict(label=0.5, intrusion=0.4, brackets=0.15, frame=0.05))
    return p


ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI']


# --------------------------------------------------------------------------------------------------------------------
#  layout and geometry
# --------------------------------------------------------------------------------------------------------------------
def depict(src, coordgen=False):
    if not coordgen:
        import metal2d
        mol = metal2d.depict(Chem.Mol(src))
    else:
        mol = Chem.Mol(src)
        rdCoordGen.AddCoords(mol)
    if mol.GetNumConformers() == 0:
        raise L.Skip('depiction')
    xy = mol.GetConformer().GetPositions()
    if not np.isfinite(xy).all():
        raise L.Skip('depiction')
    return mol


def _positions(mol):
    return np.array(mol.GetConformer().GetPositions()[:, :2], dtype=float)


def _set_positions(mol, xy):
    conf = mol.GetConformer()
    for i, (x, y) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))


def _subtree(mol, start, stop):
    """Atoms reachable from `start` without entering `stop`; None if the walk reaches a metal."""
    seen, todo = {start}, [start]
    while todo:
        a = mol.GetAtomWithIdx(todo.pop())
        for n in a.GetNeighbors():
            j = n.GetIdx()
            if j in seen or j in stop:
                continue
            if is_metal(n) or n.GetProp('kind') == 'ct':
                return None
            seen.add(j)
            todo.append(j)
    return seen


def geometry(draft, centroids, rng, p):
    xy = _positions(draft)
    ring_count = {}
    for r in _organic_rings(draft):
        for i in r:
            ring_count[i] = ring_count.get(i, 0) + 1
    for ct, m, ring in centroids:
        if rng.random() >= p['squash'] or any(ring_count.get(i, 0) > 1 for i in ring):
            continue
        axis = xy[ct] - xy[m]
        if np.linalg.norm(axis) < 1e-6:
            continue
        axis /= np.linalg.norm(axis)
        s = rng.uniform(0.3, 0.5)
        moves = {}
        for i in ring:
            along = (xy[i] - xy[ct]) @ axis
            shift = -(1 - s) * along * axis
            sub = _subtree(draft, i, set(ring) | {m, ct})
            if sub is None:
                moves = None
                break
            for j in sub:
                moves[j] = shift
        if moves:
            for j, shift in moves.items():
                xy[j] += shift
    if p['mirror']:
        xy[:, 0] = -xy[:, 0]
    if p['rotate']:
        c, s = math.cos(p['rotate']), math.sin(p['rotate'])
        centre = xy.mean(axis=0)
        xy = (xy - centre) @ np.array([[c, s], [-s, c]]) + centre
    _set_positions(draft, xy)


def wedges(src, draft, rng, p):
    """Organic stereo wedges consistent with the final coordinates, plus optional metal-ligand legs."""
    n = src.GetNumAtoms()
    m = Chem.Mol(src)
    conf = Chem.Conformer(n)
    xy = _positions(draft)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(float(xy[i][0]), float(xy[i][1]), 0.0))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    try:
        Chem.WedgeMolBonds(m, m.GetConformer())
    except Exception:
        return
    for b in m.GetBonds():
        d = b.GetBondDir()
        if d not in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if is_metal(m.GetAtomWithIdx(i)) or is_metal(m.GetAtomWithIdx(j)):
            continue
        rb = draft.GetBondBetweenAtoms(i, j)
        if rb is None or rb.GetBondType() != Chem.BondType.SINGLE:
            continue
        if rb.GetBeginAtomIdx() != i:
            rb = L._readd(draft, i, j, Chem.BondType.SINGLE, 1)
        rb.SetBondDir(d)
    if p['metal_legs']:
        legs = [b for b in draft.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE
                and is_metal(b.GetBeginAtom()) and not is_metal(b.GetEndAtom())
                and b.GetEndAtom().GetProp('kind') != 'ct' and b.GetBondDir() == Chem.BondDir.NONE
                and not b.HasProp('dative')]
        if len(legs) >= 3:
            for b in rng.choice(legs, size=min(len(legs), int(rng.integers(1, 3))), replace=False):
                b.SetBondDir(Chem.BondDir.BEGINWEDGE if rng.random() < 0.5 else Chem.BondDir.BEGINDASH)


def mark_circles(draft, centroids, rng, p):
    """Rings that get an aromatic circle: every eta ring and each other aromatic 5/6-ring with p=0.5."""
    if not p['circles']:
        return
    eta = {tuple(sorted(r)) for _, _, r in centroids}
    rings = []
    for r in _organic_rings(draft):
        key = tuple(sorted(r))
        aromatic = all(draft.GetAtomWithIdx(i).GetIsAromatic() for i in r) and len(r) in (5, 6)
        if key in eta or (aromatic and rng.random() < 0.5):
            rings.append(r)
    for k, r in enumerate(rings):
        for i in r:
            a = draft.GetAtomWithIdx(i)
            prev = a.GetProp('circles') if a.HasProp('circles') else ''
            a.SetProp('circles', f'{prev},{k}:{len(r)}' if prev else f'{k}:{len(r)}')


def circle_rings(mol):
    """Rings marked by mark_circles that survived the condensation, as lists of atom indices."""
    groups = {}
    for a in mol.GetAtoms():
        if a.HasProp('circles'):
            for tag in a.GetProp('circles').split(','):
                groups.setdefault(tag, []).append(a.GetIdx())
    return [atoms for tag, atoms in groups.items() if len(atoms) == int(tag.split(':')[1])]


# --------------------------------------------------------------------------------------------------------------------
#  drawing
# --------------------------------------------------------------------------------------------------------------------
def drawing_mol(final, circled, p=None, rng=None):
    """Copy for RDKit: Kekule bond orders, circled rings as single bonds, no aromatic flags; optionally dative bonds
    as arrows (donor -> metal) and metals written with an oxidation state (Co^III) or in brackets ([Fe])."""
    dm = Chem.RWMol(final)
    if p and p.get('arrows'):
        for b in [b for b in dm.GetBonds() if b.HasProp('dative')]:
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            metal, donor = (i, j) if is_metal(dm.GetAtomWithIdx(i)) else (j, i)
            dm.RemoveBond(i, j)
            dm.AddBond(donor, metal, Chem.BondType.DATIVE)
    if p and p.get('metal_label', 'plain') != 'plain':
        for a in dm.GetAtoms():
            if is_metal(a) and a.GetFormalCharge() == 0 and not a.HasProp('atomLabel'):
                a.SetProp('atomLabel', f'{a.GetSymbol()}<sup>{ROMAN[int(rng.integers(0, 4))]}</sup>'
                          if p['metal_label'] == 'oxidation' else f'[{a.GetSymbol()}]')
    in_circle = {frozenset((i, j)) for r in circled for i in r for j in r}
    for b in dm.GetBonds():
        if b.GetIsAromatic() or b.GetBondType() == Chem.BondType.AROMATIC:
            key = frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            b.SetBondType(Chem.BondType.SINGLE if key in in_circle else
                          {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                           3: Chem.BondType.TRIPLE}[b.GetIntProp('draw')])
        b.SetIsAromatic(False)
    for a in dm.GetAtoms():
        a.SetIsAromatic(False)
    dm.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(dm)
    return dm.GetMol()


def draw(final, circled, rng, p):
    dm = drawing_mol(final, circled, p, rng)
    n = dm.GetNumAtoms()
    xy = _positions(dm)
    lengths = [np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()]) for b in dm.GetBonds()]
    blen = float(np.median([x for x in lengths if x > 1e-3])) if lengths else 1.5
    extent = (xy.max(axis=0) - xy.min(axis=0)) / blen * p['bond_px']
    margin = 3 * p['bond_px'] + 20
    W, H = int(extent[0] + 2 * margin), int(extent[1] + 2 * margin)
    d2d = rdMolDraw2D.MolDraw2DCairo(max(W, 64), max(H, 64))
    o = d2d.drawOptions()
    o.prepareMolsBeforeDrawing = False
    o.fixedBondLength = p['bond_px']
    o.padding = p['padding']
    o.bondLineWidth = p['lw']
    o.scaleBondWidth = False
    o.minFontSize = p['min_font']
    o.maxFontSize = -1
    o.baseFontSize = p['base_font']
    o.multipleBondOffset = p['mbo']
    o.addStereoAnnotation = False
    o.explicitMethyl = p['explicit_methyl']
    if p['font']:
        o.fontFile = p['font']
    if p['bw']:
        o.useBWAtomPalette()
    kwargs = {}
    if p['highlight'] and dm.GetNumBonds():
        o.continuousHighlight = False
        colour = [(0.85, 0.1, 0.1), (0.1, 0.3, 0.85), (0.1, 0.6, 0.2)][int(rng.integers(0, 3))]
        bonds = [int(i) for i in rng.choice(dm.GetNumBonds(), size=max(1, dm.GetNumBonds() // 5), replace=False)]
        kwargs = dict(highlightAtoms=[], highlightBonds=bonds, highlightBondColors={b: colour for b in bonds})
    d2d.DrawMolecule(dm, **kwargs)
    d2d.FinishDrawing()
    img = cv2.imdecode(np.frombuffer(d2d.GetDrawingText(), np.uint8), cv2.IMREAD_COLOR)
    pix = np.array([[d2d.GetDrawCoords(i).x, d2d.GetDrawCoords(i).y] for i in range(n)])
    # aromatic circles
    lw = max(1, int(round(p['lw'])))
    for ring in circled:
        pts = pix[ring]
        centre = pts.mean(axis=0)
        radius = 0.58 * float(np.mean(np.linalg.norm(pts - centre, axis=1)))
        cv2.circle(img, (int(round(centre[0])), int(round(centre[1]))), int(round(radius)), (0, 0, 0), lw,
                   cv2.LINE_AA)
    # R placeholders drawn as spheres (the MRBW-style ball)
    for a in dm.GetAtoms():
        if a.HasProp('ball'):
            centre = (int(round(pix[a.GetIdx()][0])), int(round(pix[a.GetIdx()][1])))
            radius = max(3, int(round(0.3 * p['bond_px'])))
            shade = int(rng.integers(100, 210))
            cv2.circle(img, centre, radius, (shade, shade, shade), -1, cv2.LINE_AA)
            cv2.circle(img, centre, radius, (40, 40, 40), 1, cv2.LINE_AA)
    return _crop(img, pix)


def check_overlap(pix, bond_px, min_ratio=0.35):
    """Reject layouts with atoms drawn on top of each other (knotted macrocycles, failed cluster layouts): the
    model would be trained to see one atom where the label says two."""
    if len(pix) < 2:
        return
    d = np.linalg.norm(pix[:, None, :] - pix[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if d.min() < min_ratio * bond_px:
        raise L.Skip('atom_overlap')


def _crop(img, pix, pad=4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = np.argwhere(gray < 245)
    if len(dark) == 0:
        raise L.Skip('empty_drawing')
    y0, x0 = np.maximum(dark.min(axis=0) - pad, 0)
    y1, x1 = np.minimum(dark.max(axis=0) + pad + 1, gray.shape)
    return img[y0:y1, x0:x1], pix - np.array([x0, y0])


# --------------------------------------------------------------------------------------------------------------------
#  sample
# --------------------------------------------------------------------------------------------------------------------
def render(src, rng, depicted=None, p=None):
    """src: source molecule (label.load_source). depicted: cached metal2d layout of src (optional).

    Returns dict(image=gray uint8, smiles=label SMILES, node_coords=[[x, y]...] in [0, 1], edges=[[u, v, t]...],
    meta=...). Raises label.Skip with a reason when the structure cannot be rendered.
    """
    p = p or sample_params(rng)
    if p['coordgen'] or depicted is None:
        depicted = depict(src, coordgen=p['coordgen'])
    draft, centroids = L.make_draft(src, depicted, p['show_charge'])
    chosen = L.choose_abbreviations(src, draft, centroids, rng, p['rate_organic'], p['p_ligand'])
    replacements, additions = [], []
    if p['rgroup']:
        blocked = {i for _, _, r in centroids for i in r} | {i for _, _, atoms, _ in chosen for i in atoms}
        blocked |= {a.GetIdx() for a in src.GetAtoms() if is_metal(a)}
        replacements, additions = L.choose_rgroups(src, draft, blocked, rng)
        if additions:
            xy0 = _positions(depicted)
            blen = float(np.median([np.linalg.norm(xy0[b.GetBeginAtomIdx()] - xy0[b.GetEndAtomIdx()])
                                    for b in depicted.GetBonds()])) if depicted.GetNumBonds() else 1.5
            L.add_rgroups(draft, additions, blen)
    gold = L.gold_with_rgroups(src, replacements, additions) if replacements or additions else None
    chosen = chosen + replacements
    geometry(draft, centroids, rng, p)
    wedges(src, draft, rng, p)
    deleted = set()
    for _, _, atoms, _ in chosen:
        deleted.update(atoms)
    mark_circles(draft, [(ct, m, r) for ct, m, r in centroids if not set(r) & deleted], rng, p)
    L.delete_atoms(draft, L.condense(draft, chosen))
    final = draft.GetMol()

    smiles, order, edges = L.label_outputs(final)
    L.check_label(smiles, final.GetNumAtoms())
    img, pix = draw(final, circle_rings(final), rng, p)
    # the centroid sits inside its (possibly squashed) ring by design
    check_overlap(pix[[a.GetIdx() for a in final.GetAtoms() if a.GetProp('kind') != 'ct']], p['bond_px'])
    img, pix = add_context(img, pix, rng, p['bond_px'], p['context'])
    H, W = img.shape
    coords = pix[order] / np.array([W, H])
    if not ((coords >= 0) & (coords <= 1)).all():
        raise L.Skip('coords_outside')
    meta = dict(n_atoms=final.GetNumAtoms(), n_eta=len(centroids), n_abbr=len(chosen) - len(replacements),
                n_rgroup=len(replacements) + len(additions), arrows=p['arrows'], metal_label=p['metal_label'],
                n_dative=sum(1 for e in edges if e[2] == 7),
                standard_style=p['standard'], coordgen=p['coordgen'], circles=len(circle_rings(final)))
    return dict(image=img, smiles=smiles, node_coords=np.round(coords, 5).tolist(), edges=edges, meta=meta, gold=gold)
