"""Metal catalysts from the reaction database (reactions.reaction_catalyst, read-only) as dative SMILES.

Most records are PubChem-style: the metal is a separate ion or atom ([Pd].PPh3.PPh3..., [Cu+].[I-]), so the
coordination is missing. It is rebuilt only where it is unambiguous:
  neutral phosphines, nitriles           -> dative bond to the metal
  mono-atomic anions ([Cl-], [O-2])      -> covalent bond of order |charge| (Cl-Pd, Mn=O)
  anions with charged O/N/S (OAc, OTf)   -> covalent bond from every charged atom
  acac-type enolates                     -> O,O-chelate
The metal's positive charge must be used up exactly. Anything else (dba, cod, arenes, bare metals, several
separate metal ions) is dropped with a reason.

usage:
  python -m metal_ocsr.catalysts --env /mnt/hard1/rxn_etl/.env --out data/raw/db_catalysts        (live DB)
  python -m metal_ocsr.catalysts --from_csv scratch/catalyst_metal_classes.csv --out data/raw/db_catalysts
"""
import argparse
import collections
import os

from rdkit import Chem, RDLogger

from molscribe.metal import is_metal

RDLogger.DisableLog('rdApp.*')

ALKALI = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba'}
DONOR_VALENCE = {'N': 3, 'P': 3, 'As': 3, 'S': 2, 'Se': 2}
PHOSPHINE = Chem.MolFromSmarts('[PX3;H0;+0]')
NITRILE = Chem.MolFromSmarts('[NX1;+0]#[C]')
ACAC = Chem.MolFromSmarts('[O-]-[#6]=[#6]-[#6]=O')
SIGMA_CP = Chem.MolFromSmarts('[!#1;!#6;!#7;!#8;!#9;!#15;!#16;!#17;!#35;!#53]-[#6;R]1-[#6]=[#6]-[#6]=[#6]-1')


class Drop(Exception):
    pass


def complex_metal(atom):
    return is_metal(atom) and atom.GetSymbol() not in ALKALI


def _fix_bonded(mol):
    """Bonded records: turn neutral donor-metal bonds into dative bonds and drop PubChem's artefact H on donors."""
    rw = Chem.RWMol(mol)
    for b in list(rw.GetBonds()):
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if complex_metal(a) == complex_metal(c):
            continue
        m, x = (a, c) if complex_metal(a) else (c, a)
        if b.GetBondType() != Chem.BondType.SINGLE or x.GetFormalCharge() != 0:
            continue
        organic = [bb for bb in x.GetBonds() if not is_metal(bb.GetOtherAtom(x))]
        order = sum(bb.GetBondTypeAsDouble() for bb in organic)
        sym = x.GetSymbol()
        dative = (sym in DONOR_VALENCE and order >= DONOR_VALENCE[sym]) or (sym == 'O' and order >= 2) or \
                 (sym in ('N', 'P', 'As') and x.GetIsAromatic())
        if not dative:
            continue
        mi, xi = m.GetIdx(), x.GetIdx()
        rw.RemoveBond(mi, xi)
        rw.AddBond(xi, mi, Chem.BondType.DATIVE)
        x = rw.GetAtomWithIdx(xi)
        if sym in DONOR_VALENCE and not x.GetIsAromatic():
            x.SetNoImplicit(True)
            x.SetNumExplicitHs(max(0, DONOR_VALENCE[sym] - int(order)))
    return rw


def _connect(mol):
    """Attach the separate ligand fragments to the fragment that holds the single metal (an ion such as [Pd+2] or an
    already bonded core such as Cl[Pd]Cl)."""
    frags = Chem.GetMolFrags(mol)
    metals = [a.GetIdx() for a in mol.GetAtoms() if complex_metal(a)]
    if len(frags) == 1:
        if mol.GetNumAtoms() == 1:
            raise Drop('bare_metal')
        return Chem.RWMol(mol)
    if len(metals) != 1:
        raise Drop('metal_count')
    m = metals[0]
    core = next(f for f in frags if m in f)
    rw = Chem.RWMol(mol)
    charge = rw.GetAtomWithIdx(m).GetFormalCharge()
    ligands = [f for f in frags if f != core]
    for frag in ligands:
        sub = set(frag)
        charged = [i for i in frag if rw.GetAtomWithIdx(i).GetFormalCharge() < 0]
        if len(frag) == 1 and charged:
            x = rw.GetAtomWithIdx(frag[0])
            k = -x.GetFormalCharge()
            rw.AddBond(frag[0], m, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[k])
            x.SetFormalCharge(0)
            x.SetNumExplicitHs(0)
            x.SetNoImplicit(True)
            charge -= k
            continue
        acac = [match for match in mol.GetSubstructMatches(ACAC) if set(match) <= sub]
        if acac:
            o_minus, o_carbonyl = acac[0][0], acac[0][4]
            rw.AddBond(o_minus, m, Chem.BondType.SINGLE)
            rw.GetAtomWithIdx(o_minus).SetFormalCharge(0)
            rw.AddBond(o_carbonyl, m, Chem.BondType.DATIVE)
            charge -= 1
            continue
        if charged:
            if any(rw.GetAtomWithIdx(i).GetSymbol() not in ('O', 'N', 'S') for i in charged):
                raise Drop('anion_type')
            for i in charged:
                rw.AddBond(i, m, Chem.BondType.SINGLE)
                rw.GetAtomWithIdx(i).SetFormalCharge(0)
                charge -= 1
            continue
        donors = [match[0] for patt in (PHOSPHINE, NITRILE) for match in mol.GetSubstructMatches(patt)
                  if match[0] in sub]
        if len(donors) != 1 or any(rw.GetAtomWithIdx(i).GetFormalCharge() > 0 for i in frag):
            raise Drop('ambiguous_ligand')
        rw.AddBond(donors[0], m, Chem.BondType.DATIVE)
    if charge != 0:
        raise Drop('unbalanced_charge')
    rw.GetAtomWithIdx(m).SetFormalCharge(0)
    return rw


def reconnect(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Drop('parse')
    metals = [a for a in mol.GetAtoms() if complex_metal(a)]
    if not metals:
        raise Drop('no_metal')
    if mol.HasSubstructMatch(SIGMA_CP):
        raise Drop('sigma_cp')  # PubChem writes metallocenes with a sigma M-C bond; the figure is a sandwich
    if any(a.GetDegree() > 0 for a in metals):
        mol = _fix_bonded(mol).GetMol()
        mol.UpdatePropertyCache(strict=False)
    out = _connect(mol).GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        raise Drop('sanitize')
    return Chem.MolToSmiles(out)


def export(env_path):
    import pandas as pd
    import psycopg2
    from dotenv import dotenv_values
    env = dotenv_values(env_path)
    conn = psycopg2.connect(host='localhost', port=env['POSTGRES_PORT'], dbname=env['POSTGRES_DB'],
                            user=env['POSTGRES_USER'], password=env['POSTGRES_PASSWORD'])
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute('select smiles, count(*) from reactions.reaction_catalyst where smiles is not null group by smiles')
        rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=['smiles', 'n_rows'])


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--env', help='.env with POSTGRES_* of the rxn_etl database (read-only export)')
    src.add_argument('--from_csv', help='an earlier export with columns smiles and n / n_rows')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.env:
        df = export(args.env)
    else:
        df = pd.read_csv(args.from_csv).rename(columns={'n': 'n_rows'})[['smiles', 'n_rows']]
    df.to_csv(os.path.join(args.out, 'reaction_catalyst_smiles.csv'), index=False)
    reasons = collections.Counter()
    out = []
    for smi, n in zip(df.smiles, df.n_rows):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None or not any(complex_metal(a) for a in mol.GetAtoms()):
            continue
        try:
            out.append(dict(source_smiles=smi, smiles=reconnect(smi), n_rows=n))
            reasons['ok'] += 1
        except Drop as e:
            reasons[str(e)] += 1
    pd.DataFrame(out).to_csv(os.path.join(args.out, 'catalysts_reconnected.csv'), index=False)
    print(dict(reasons))


if __name__ == '__main__':
    main()
