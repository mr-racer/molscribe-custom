"""Expand every dictionary key through the real graph->SMILES path and report the ones that do not give a valid
molecule. 1-bond keys are attached to a benzene carbon, 2-bond keys are placed inside a saturated ring, 0-bond keys
stand alone.

usage: python benchmark/abbrev_check.py [--show N]
"""
import argparse
import os
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molscribe.chemistry import _convert_graph_to_smiles  # noqa: E402
from molscribe.constants import ABBREVIATIONS_BY_ATTACH  # noqa: E402


def expand(label, n_attach):
    if n_attach == 1:
        symbols = ["c"] * 6 + [f"[{label}]"]
        bonds = [(i, (i + 1) % 6, 4) for i in range(6)] + [(0, 6, 1)]
    elif n_attach == 2:
        symbols = ["C"] * 5 + [f"[{label}]"]
        bonds = [(i, (i + 1) % 6, 1) for i in range(6)]
    else:
        symbols, bonds = [f"[{label}]"], []
    n = len(symbols)
    edges = [[0] * n for _ in range(n)]
    for i, j, order in bonds:
        edges[i][j] = edges[j][i] = order
    coords = [[i / max(n - 1, 1), 0.5] for i in range(n)]
    smiles, _, ok = _convert_graph_to_smiles(coords, symbols, edges)
    mol = Chem.MolFromSmiles(smiles) if ok and smiles else None
    return smiles, mol is not None and "*" not in smiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=50)
    args = ap.parse_args()
    bad = []
    for (label, n_attach), sub in sorted(ABBREVIATIONS_BY_ATTACH.items()):
        smiles, ok = expand(label, n_attach)
        if not ok:
            bad.append((label, n_attach, sub.smiles, smiles))
    print(f"{len(ABBREVIATIONS_BY_ATTACH) - len(bad)}/{len(ABBREVIATIONS_BY_ATTACH)} keys expand to a valid molecule")
    for label, n_attach, frag, smiles in bad[:args.show]:
        print(f"  FAIL {label!r} (n_attach={n_attach}) fragment={frag} -> {smiles}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
