"""Merge reviewed abbreviation items into molscribe/constants.py (between the GENERATED markers).

usage: python benchmark/abbrev_merge.py accepted.json [--dry-run]

Input: {"accepted": [{"abbrevs": [...], "smiles": "...", "name": "...", "n_attach": 1, "category": "...",
                      "note": "..."}, ...]}
Every item is re-validated with RDKit here (parse, sanitize, radical count vs n_attach, attach-a-methyl probe);
keys that already exist with the same n_attach are dropped; keys colliding with element symbols are kept only if
the item carries element_collision_ok. The generated block is grouped by category and sorted by first key.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CONSTANTS = os.path.join(ROOT, "molscribe", "constants.py")
BEGIN = "    # --- BEGIN GENERATED ABBREVIATIONS (benchmark/abbrev_merge.py) ---\n"
END = "    # --- END GENERATED ABBREVIATIONS ---\n"


def validate(item):
    """Return an error string or None."""
    smiles, n_attach = item["smiles"], int(item.get("n_attach", 1))
    if "*" in smiles or not smiles:
        return "empty or wildcard SMILES"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "does not parse"
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return f"does not sanitize: {e}"
    radicals = [(a.GetIdx(), a.GetNumRadicalElectrons()) for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    total = sum(r for _, r in radicals)
    atom0 = mol.GetAtomWithIdx(0)
    if n_attach == 0:
        if total and not item.get("stable_radical"):
            return "radicals in a 0-attach entry"
        return None
    if total == 0 and atom0.GetSymbol() in ("S", "P", "Se", "Te") and n_attach == 2:
        radicals = [(0, 2)]
    elif total != n_attach or radicals[0][0] != 0:
        return f"radical electrons {radicals} do not match n_attach={n_attach} on atom 0"
    rw = Chem.RWMol(mol)
    for idx, r in radicals:
        for _ in range(r):
            c = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(idx, c, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(idx).SetNumRadicalElectrons(0)
        if rw.GetAtomWithIdx(idx).GetSymbol() != "C":
            rw.GetAtomWithIdx(idx).SetNumExplicitHs(0)
            rw.GetAtomWithIdx(idx).SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rw)
    except Exception as e:
        return f"probe with methyl fails: {e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from molscribe.constants import ABBREVIATIONS_BY_ATTACH, is_rgroup

    data = json.load(open(args.path, encoding="utf-8"))
    items = data["accepted"] if "accepted" in data else data["items"]
    by_cat, dropped, seen = defaultdict(list), [], set()
    for it in items:
        n_attach = int(it.get("n_attach", 1))
        err = validate(it)
        if err:
            dropped.append((it["abbrevs"], err))
            continue
        keys = []
        for k in it["abbrevs"]:
            if not isinstance(k, str) or not k or " " in k or '"' in k or "\\" in k:
                continue
            if (k, n_attach) in ABBREVIATIONS_BY_ATTACH or (k, n_attach) in seen:
                continue
            if Chem.AtomFromSmiles(f"[{k}]") is not None and not it.get("element_collision_ok"):
                dropped.append(([k], "collides with an element symbol"))
                continue
            if is_rgroup(k):
                dropped.append(([k], "looks like an R-group placeholder"))
                continue
            keys.append(k)
            seen.add((k, n_attach))
        if keys:
            by_cat[it.get("category") or "misc"].append((keys, it["smiles"], n_attach, it.get("name", "")))

    lines = [BEGIN]
    for cat in sorted(by_cat):
        lines.append(f"    # {cat}\n")
        for keys, smiles, n_attach, name in sorted(by_cat[cat], key=lambda t: t[0][0].lower()):
            comment = f"  # {name}" if name else ""
            extra = f", {n_attach}" if n_attach != 1 else ""
            lines.append(f"    _extra({json.dumps(keys, ensure_ascii=False)}, {json.dumps(smiles)}{extra}),{comment}\n")
    lines.append(END)
    block = "".join(lines).replace('["', "['").replace('"]', "']").replace('", "', "', '")

    src = open(CONSTANTS, encoding="utf-8").read()
    start, end = src.index(BEGIN), src.index(END) + len(END)
    new_src = src[:start] + block + src[end:]
    n_items = sum(len(v) for v in by_cat.values())
    n_keys = sum(len(k) for v in by_cat.values() for k, *_ in v)
    print(f"{n_items} items / {n_keys} new keys in {len(by_cat)} categories; dropped {len(dropped)}")
    for keys, why in dropped[:40]:
        print("   dropped", keys, "-", why)
    if not args.dry_run:
        open(CONSTANTS, "w", encoding="utf-8", newline="\n").write(new_src)
        print("written to", CONSTANTS)


if __name__ == "__main__":
    main()
