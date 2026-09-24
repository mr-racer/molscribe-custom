"""Unit checks for the graph -> SMILES postprocessing (no model needed).

usage: python benchmark/test_postprocess.py
Each case builds a small atom/bond graph the way the model would output it and checks the expanded SMILES.
"""
import os
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molscribe.chemistry import _convert_graph_to_smiles  # noqa: E402


def canon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def graph(symbols, bonds, coords=None):
    """bonds: list of (i, j, order); coords default to a horizontal line so 'left' and 'right' are well defined."""
    n = len(symbols)
    edges = [[0] * n for _ in range(n)]
    for i, j, order in bonds:
        edges[i][j] = edges[j][i] = order
    if coords is None:
        coords = [[i / max(n - 1, 1), 0.5] for i in range(n)]
    smiles, _, ok = _convert_graph_to_smiles(coords, symbols, edges)
    return smiles


CASES = [
    # label on a benzene ring (single attachment)
    ("phenyl-OTBS", graph(["c"] * 6 + ["[OTBS]"], [(i, (i + 1) % 6, 4) for i in range(6)] + [(0, 6, 1)]),
     "CC(C)(C)[Si](C)(C)Oc1ccccc1"),
    ("uppercase BOC", graph(["N", "[BOC]"], [(0, 1, 1)]), "CC(C)(C)OC(=O)N"),
    # in-line two-attachment groups, read left to right
    ("C-CONH-C", graph(["C", "[CONH]", "C"], [(0, 1, 1), (1, 2, 1)]), "CNC(C)=O"),
    ("C-NHCO-C", graph(["C", "[NHCO]", "C"], [(0, 1, 1), (1, 2, 1)]), "CNC(C)=O"),
    ("C-CO2-C", graph(["C", "[CO2]", "C"], [(0, 1, 1), (1, 2, 1)]), "COC(C)=O"),
    ("reversed geometry: C right of NHCO", graph(["C", "[NHCO]", "C"], [(0, 1, 1), (1, 2, 1)],
                                                coords=[[1.0, 0.5], [0.5, 0.5], [0.0, 0.5]]), "CNC(C)=O"),
    ("CO with two bonds is a carbonyl", graph(["C", "[CO]", "C"], [(0, 1, 1), (1, 2, 1)]), "CC(C)=O"),
    ("CO on a metal is a ligand", graph(["[Mn+]", "[CO]"], [(0, 1, 1)]), "[O+]#[C-]->[Mn+]"),
    ("SO2 in-line", graph(["C", "[SO2]", "C"], [(0, 1, 1), (1, 2, 1)]), "CS(C)(=O)=O"),
    ("OCH2O bridge", graph(["C", "[OCH2O]", "C"], [(0, 1, 1), (1, 2, 1)]), "COCOC"),
    # R-group placeholders stay wildcards
    ("R-alpha", graph(["C", "[Rα]"], [(0, 1, 1)]), "*C"),
    ("EWG", graph(["C", "[EWG]"], [(0, 1, 1)]), "*C"),
    ("R1 prime", graph(["C", "[R1']"], [(0, 1, 1)]), "*C"),
    ("Ar prime", graph(["C", "[Ar']"], [(0, 1, 1)]), "*C"),
    ("OR expands to O-*", graph(["C", "[OR]"], [(0, 1, 1)]), "*OC"),
    ("NR2", graph(["C", "[NR2]"], [(0, 1, 1)]), "*N(*)C"),
    # hydrogen isotopes
    ("D", graph(["C", "[D]"], [(0, 1, 1)]), "[2H]C"),
    ("CD3", graph(["C", "[CD3]"], [(0, 1, 1)]), "[2H]C([2H])([2H])C"),
    # previously fixed behaviour must stay
    ("NMe in ring", graph(["C"] * 5 + ["[NMe]"], [(i, (i + 1) % 6, 1) for i in range(6)]), "CN1CCCCC1"),
    ("SO2NH2 not nitro", graph(["c"] * 6 + ["[SO2NH2]"], [(i, (i + 1) % 6, 4) for i in range(6)] + [(0, 6, 1)]),
     "NS(=O)(=O)c1ccccc1"),
]


def main():
    failed = 0
    for name, got, expected in CASES:
        ok = canon(got) is not None and canon(got) == canon(expected)
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:40s} {got}" + ("" if ok else f"   expected {expected}"))
    print(f"{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
