"""Build the evaluation sets used by benchmark/run_benchmark.py.

general.csv : ACS (all 331 images, MolScribe paper) + a seeded sample of SMILES-eligible, metal-free
              MolRecBench-Wild records (gold = expanded canonical SMILES from the benchmark's own converter).
metal.csv   : every MolRecBench-Wild record that contains a metal atom (gold = SMILES with dative bonds kept as
              dative; compared with a convention-agnostic metric, see evaluate.py).

usage:
  python benchmark/prepare_data.py --raw <dir with real/acs, mrbw-*.parquet, MolRecBench-Wild-repo> --out <dir>
"""
import argparse
import csv
import io
import os
import random
import re
import shutil
import sys

import pyarrow.parquet as pq
from PIL import Image
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U",
}
# MolRecBench-Wild bond codes that can be expressed with plain RDKit bonds
DATIVE_CODES = {11, 21}
SIMPLE_CODES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
                4: Chem.BondType.AROMATIC, 5: Chem.BondType.SINGLE, 6: Chem.BondType.SINGLE,
                15: Chem.BondType.SINGLE, 16: Chem.BondType.SINGLE, 17: Chem.BondType.SINGLE,
                18: Chem.BondType.DOUBLE, 19: Chem.BondType.DOUBLE, 20: Chem.BondType.TRIPLE}


def strip(symbol):
    return re.sub(r"[\[\]]", "", str(symbol))


def load_records(raw):
    cols = ["image", "id", "evaluation_subset", "hardcase_label", "symbols", "charges", "radicals", "valences",
            "isotopes", "attach_points", "coords", "bonds", "brackets"]
    rows = []
    for i in range(4):
        rows += pq.read_table(os.path.join(raw, f"mrbw-{i}.parquet"), columns=cols).to_pylist()
    return rows


def save_image(record, path):
    img = Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB")
    img.save(path)


def metal_gold_smiles(record, expand):
    """Graph -> SMILES keeping dative bonds; superatoms expanded with the benchmark's own dictionary."""
    rw = Chem.RWMol()
    labels = {}
    for i, s in enumerate(record["symbols"]):
        sym = strip(s)
        atom = Chem.AtomFromSmiles(f"[{sym}]") if sym not in ("R", "X", "Y", "Z") else None
        if atom is None:  # superatom: dummy atom tagged by isotope, replaced by its [label] after writing
            atom = Chem.Atom(0)
            atom.SetIsotope(500 + i)
            labels[i] = sym
        charge = (record["charges"] or [None] * len(record["symbols"]))[i]
        if charge:
            atom.SetFormalCharge(int(charge))
        atom.SetNoImplicit(False)
        rw.AddAtom(atom)
    for a, b, code in record["bonds"]:
        if code in DATIVE_CODES:
            donor, acceptor = (a, b) if strip(record["symbols"][b]) in METALS else (b, a)
            rw.AddBond(donor, acceptor, Chem.BondType.DATIVE)
        elif code in SIMPLE_CODES:
            rw.AddBond(a, b, SIMPLE_CODES[code])
        else:  # any / wavy / hydrogen bond: drawn line, keep connectivity
            rw.AddBond(a, b, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    smiles = re.sub(r"\[(\d+)\*\]", lambda m: f"[{labels[int(m.group(1)) - 500]}]", smiles)
    expanded = expand(smiles)
    return expanded or smiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_mrbw", type=int, default=169)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.raw, "MolRecBench-Wild-repo"))
    from evaluate.smiles_metric import filter_reasons, normalize_smiles
    from evaluate.utils import carbon_to_smiles, replace_superatom_with_mol

    def expand(smiles):
        try:
            expanded, _ = replace_superatom_with_mol(smiles, report_missing_abbr=False)
            return expanded
        except Exception:
            return None

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    general = []
    with open(os.path.join(args.raw, "real", "acs.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = os.path.join(img_dir, f"acs_{row['image_id']}.png")
            shutil.copy(os.path.join(args.raw, row["file_path"]), path)
            general.append(dict(id=f"acs_{row['image_id']}", source="ACS", image=path, gold=row["SMILES"]))

    records = load_records(args.raw)
    rng = random.Random(args.seed)
    eligible, metal = [], []
    for r in records:
        has_metal = any(strip(s) in METALS for s in r["symbols"])
        if has_metal:
            metal.append(r)
        elif not filter_reasons(r):
            eligible.append(r)
    rng.shuffle(eligible)
    picked = 0
    for r in eligible:
        if picked >= args.n_mrbw:
            break
        try:
            _, canonical, _, ok_expand, ok_canon, _ = normalize_smiles(carbon_to_smiles(r), {}, ignore_cistrans=False)
            expanded, _ = replace_superatom_with_mol(carbon_to_smiles(r), report_missing_abbr=False)
        except Exception:
            continue
        if not (ok_expand and ok_canon and expanded):
            continue
        path = os.path.join(img_dir, f"mrbw_{picked:04d}.png")
        save_image(r, path)
        general.append(dict(id=r["id"], source=f"MRBW-{r['evaluation_subset']}", image=path, gold=expanded))
        picked += 1

    metal_rows = []
    for i, r in enumerate(metal):
        try:
            gold = metal_gold_smiles(r, expand)
        except Exception as e:
            print("skip metal record", r["id"], type(e).__name__, e)
            continue
        path = os.path.join(img_dir, f"metal_{i:04d}.png")
        save_image(r, path)
        metal_rows.append(dict(id=r["id"], source=f"MRBW-{r['evaluation_subset']}", image=path, gold=gold))

    for name, rows in (("general.csv", general), ("metal.csv", metal_rows)):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "source", "image", "gold"])
            w.writeheader()
            for row in rows:
                row = dict(row, image=os.path.relpath(row["image"], args.out).replace("\\", "/"))
                w.writerow(row)
        print(name, len(rows))


if __name__ == "__main__":
    main()
