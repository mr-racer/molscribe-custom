"""Score benchmark predictions.

general set: exact match of RDKit canonical SMILES, with stereo (chirality + cis/trans) and without stereo.
             R-groups R1..Rn -> [n*], any other unparsable bracket token -> * (as molscribe/evaluate.py).
metal set:   the ground truth draws metal-ligand bonds as plain lines, while predictions may use dative bonds or
             charge-separated (PubChem-style) bonds. Both sides are normalized before comparison: bonds to metals
             become single bonds, formal charges on metals and on atoms bonded to metals are cleared, hydrogens are
             recomputed from valence and aromaticity is perceived on the metal-free ligands.

usage: python benchmark/evaluate.py --data <data dir> --pred <run dir> [--pred <run dir> ...] --out metrics.json
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U",
}
TOKEN = re.compile(r"(\[[^\]]+\]|Br|Cl|.)")


def replace_rgroups(smiles):
    out = []
    for tok in TOKEN.findall(smiles):
        if tok.startswith("[") and tok.endswith("]"):
            sym = tok[1:-1]
            if sym[:1] == "R" and sym[1:].isdigit():
                tok = f"[{sym[1:]}*]"
            elif Chem.AtomFromSmiles(tok) is None:
                tok = "*"
        out.append(tok)
    return "".join(out)


def canon_general(smiles, stereo):
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(replace_rgroups(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=stereo)
    except Exception:
        return None


def canon_metal(smiles, stereo):
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(replace_rgroups(smiles), sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.RemoveHs(mol, sanitize=False)
        rw = Chem.RWMol(mol)
        metal_bonds, touched = [], set()
        for b in list(rw.GetBonds()):
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if rw.GetAtomWithIdx(i).GetSymbol() in METALS or rw.GetAtomWithIdx(j).GetSymbol() in METALS:
                metal_bonds.append((i, j))
                touched.update((i, j))
        for i, j in metal_bonds:
            rw.RemoveBond(i, j)
        for a in rw.GetAtoms():
            a.SetNumRadicalElectrons(0)
            if a.GetIdx() in touched or a.GetSymbol() in METALS:
                a.SetFormalCharge(0)
            if a.GetSymbol() in METALS or a.GetAtomicNum() == 0:
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
            else:
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            rw.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(rw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                             catchErrors=True)
        for i, j in metal_bonds:
            rw.AddBond(i, j, Chem.BondType.SINGLE)
        rw.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(rw, isomericSmiles=stereo)
    except Exception:
        return None


def score(gold_rows, preds, kind):
    canon = canon_general if kind == "general" else canon_metal
    per_source = defaultdict(lambda: defaultdict(int))
    details = []
    for r in gold_rows:
        p = preds.get(r["id"], "")
        res = {}
        for stereo in (True, False):
            g, q = canon(r["gold"], stereo), canon(p, stereo)
            res["stereo" if stereo else "nostereo"] = g is not None and q is not None and g == q
        res["valid"] = canon(p, False) is not None
        # strict: the prediction as-is passes full RDKit sanitization (usable downstream without special handling)
        res["rdkit_valid"] = bool(p) and Chem.MolFromSmiles(p) is not None
        res["gold_ok"] = canon(r["gold"], False) is not None
        # "clean": the ground truth has no unexpanded label / wildcard left
        clean = "*" not in replace_rgroups(r["gold"])
        for src in ("all", r["source"].split("-")[0]) + (("clean",) if clean else ()):
            d = per_source[src]
            d["n"] += 1
            for k in ("stereo", "nostereo", "valid", "rdkit_valid", "gold_ok"):
                d[k] += int(res[k])
        details.append(dict(id=r["id"], source=r["source"], **res))
    summary = {src: dict(n=d["n"], em_stereo=round(d["stereo"] / d["n"], 4),
                         em_nostereo=round(d["nostereo"] / d["n"], 4), valid=round(d["valid"] / d["n"], 4),
                         rdkit_valid=round(d["rdkit_valid"] / d["n"], 4),
                         gold_parsable=round(d["gold_ok"] / d["n"], 4))
               for src, d in per_source.items()}
    return summary, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expand_gold", default=None,
                    help="molscribe checkout whose abbreviation dictionary expands labels left in the metal ground "
                         "truth (CO, PPh3, Dipp ...); adds a '+labels' variant of the metal metrics")
    args = ap.parse_args()

    gold = {}
    for kind in ("general", "metal"):
        with open(os.path.join(args.data, f"{kind}.csv"), newline="", encoding="utf-8") as f:
            gold[kind] = list(csv.DictReader(f))
    gold_expanded = None
    if args.expand_gold:
        import sys
        sys.path.insert(0, os.path.abspath(args.expand_gold))
        from molscribe.chemistry import _postprocess_smiles
        gold_expanded = []
        for r in gold["metal"]:
            expanded, _, ok = _postprocess_smiles(r["gold"])
            gold_expanded.append(dict(r, gold=expanded if ok else r["gold"]))

    report = {}
    for run_dir in args.pred:
        run = os.path.basename(os.path.normpath(run_dir))
        for path in sorted(p for p in glob.glob(os.path.join(run_dir, "*_bs*.csv")) if not p.endswith("_details.csv")):
            name = os.path.basename(path)[:-4]
            kind = name.split("_")[0]
            with open(path, newline="", encoding="utf-8") as f:
                preds = {r["id"]: r["pred"] for r in csv.DictReader(f)}
            rows = [r for r in gold[kind] if r["id"] in preds]
            summary, details = score(rows, preds, kind)
            timing_path = path[:-4] + "_timing.json"
            timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
            report[f"{run}/{name}"] = dict(metrics=summary, timing=timing)
            with open(path[:-4] + "_details.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
                w.writeheader()
                w.writerows(details)
            m = summary["all"]
            print(f"{run:28s} {name:16s} n={m['n']:4d}  EM stereo={m['em_stereo']:.3f}  "
                  f"EM no-stereo={m['em_nostereo']:.3f}  valid={m['valid']:.3f}  rdkit_valid={m['rdkit_valid']:.3f}  "
                  f"{timing.get('ms_per_image', float('nan'))} ms/img")
            if kind == "metal" and gold_expanded is not None:
                rows_x = [r for r in gold_expanded if r["id"] in preds]
                summary_x, _ = score(rows_x, preds, kind)
                report[f"{run}/{name}+labels"] = dict(metrics=summary_x, timing=timing)
                m = summary_x["all"]
                print(f"{run:28s} {name + '+labels':16s} n={m['n']:4d}  EM stereo={m['em_stereo']:.3f}  "
                      f"EM no-stereo={m['em_nostereo']:.3f}")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
