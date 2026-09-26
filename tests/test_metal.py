"""python -m unittest discover -s tests -v"""
import unittest

from rdkit import Chem

from molscribe.metal import expand_centroids, fix_cyclopentadienyl, metal_key


def graph_mol(symbols, bonds):
    """Molecule as _convert_graph_to_smiles builds it: '*' atoms carry the label as alias."""
    mol = Chem.RWMol()
    for s in symbols:
        if s == 'Ct':
            atom = Chem.Atom(0)
            Chem.SetAtomAlias(atom, s)
            atom.SetProp('molFileAlias', s)
        else:
            atom = Chem.AtomFromSmiles(s if s.startswith('[') else f'[{s}]' if len(s) > 1 else s)
        mol.AddAtom(atom)
    for i, j, t in bonds:
        mol.AddBond(i, j, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                           4: Chem.BondType.AROMATIC}[t])
    return mol


def sandwich(ring_smiles, bond_type, metal):
    """Two copies of a 5-ring bonded to one metal through every ring atom, as a SMILES string."""
    ring = Chem.MolFromSmiles(ring_smiles)
    mol = Chem.RWMol(Chem.CombineMols(Chem.CombineMols(ring, ring), Chem.MolFromSmiles(metal)))
    m = mol.GetNumAtoms() - 1
    for i in range(10):
        mol.AddBond(i, m, bond_type)
    return Chem.MolToSmiles(mol)


class MetalKeyTest(unittest.TestCase):

    def assertSameKey(self, a, b, **kw):
        ka, kb = metal_key(a, **kw), metal_key(b, **kw)
        self.assertIsNotNone(ka, a)
        self.assertEqual(ka, kb, f'\n{a}\n{b}')

    def test_cisplatin_dative_vs_drawn(self):
        self.assertSameKey('N->[Pt+2](<-N)(<-[Cl-])<-[Cl-]', 'Cl[Pt](Cl)(N)N')

    def test_pyridine_chelate(self):
        self.assertSameKey('[Cl-]->[Pt+2]1(<-[Cl-])<-[n]2ccccc2-c2cccc[n]->12',
                           'Cl[Pt]1(Cl)[n]2ccccc2-c2cccc[n]12')

    def test_carbonyls(self):
        # the arrow starts at the preceding atom: [O+]#[C-]->[Cr] is carbon-bound
        self.assertSameKey('[O+]#[C-]->[Cr](<-[C-]#[O+])<-[C-]#[O+]', 'O#C[Cr](C#O)C#O')

    def test_ferrocene_dative_vs_kekule_drawing(self):
        self.assertSameKey(sandwich('[cH-]1cccc1', Chem.BondType.DATIVE, '[Fe+2]'),
                           sandwich('C1=CC=CC1', Chem.BondType.SINGLE, '[Fe]'))

    def test_ligand_key_ignores_connectivity(self):
        a = 'N->[Pt+2](<-N)(<-[Cl-])<-[Cl-]'
        b = 'Cl[Pt]N.N.Cl'
        self.assertNotEqual(metal_key(a), metal_key(b))
        self.assertEqual(metal_key(a, ligands_only=True), metal_key(b, ligands_only=True))


class CentroidTest(unittest.TestCase):

    def cp_iron(self, aromatic):
        # Cp ring (atoms 0-4, pentagon), centroid 5 inside it, Fe 6 below, Cl 7 on Fe
        symbols = ['C', 'C', 'C', 'C', 'C', 'Ct', 'Fe', 'Cl']
        if aromatic:
            symbols[:5] = ['c'] * 5
            ring = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 0, 4)]
        else:
            ring = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 0, 1)]
        bonds = ring + [(5, 6, 1), (6, 7, 1)]
        coords = [(0.50, 0.10), (0.62, 0.18), (0.58, 0.30), (0.42, 0.30), (0.38, 0.18),
                  (0.50, 0.22), (0.50, 0.60), (0.80, 0.60)]
        return graph_mol(symbols, bonds), coords

    def test_centroid_becomes_five_dative_bonds(self):
        mol, coords = self.cp_iron(aromatic=False)
        out = expand_centroids(mol, coords)
        fe = [a for a in out.GetAtoms() if a.GetSymbol() == 'Fe'][0]
        self.assertEqual(out.GetNumAtoms(), 7)
        self.assertEqual(sum(1 for n in fe.GetNeighbors() if n.GetSymbol() == 'C'), 5)

    def test_centroid_outside_ring_is_kept(self):
        mol, coords = self.cp_iron(aromatic=False)
        coords[5] = (0.9, 0.9)
        out = expand_centroids(mol, coords)
        self.assertEqual(out.GetNumAtoms(), 8)

    def test_cp_fix_gives_sanitizable_cyclopentadienyl(self):
        for aromatic in (False, True):
            with self.subTest(aromatic=aromatic):
                self._check_cp_fix(aromatic)

    def _check_cp_fix(self, aromatic):
        mol, coords = self.cp_iron(aromatic)
        m = Chem.Mol(fix_cyclopentadienyl(expand_centroids(mol, coords)))
        Chem.SanitizeMol(m)
        ring_c = [a for a in m.GetAtoms() if a.GetSymbol() == 'C']
        self.assertEqual([a.GetFormalCharge() for a in ring_c].count(-1), 1, Chem.MolToSmiles(m))
        self.assertEqual(sum(a.GetTotalNumHs() for a in ring_c), 5, Chem.MolToSmiles(m))


def decoder_graph(symbols, bonds, coords=None):
    n = len(symbols)
    edges = [[0] * n for _ in range(n)]
    for i, j, t in bonds:
        edges[i][j] = edges[j][i] = t
    return coords or [(0.1 * i, 0.1 * (i % 3)) for i in range(n)], symbols, edges


class GraphToSmilesTest(unittest.TestCase):

    def test_aromatic_chelating_pyridines(self):
        """The fine-tuned model predicts aromatic ring edges (type 4) for a metal-bound pyridine; the n then has
        three bonds and could not be kekulized before the M-N bond became dative -> every such complex was invalid."""
        from molscribe.chemistry import _convert_graph_to_smiles
        # 2,2'-bipyridine PtCl2: ring A n0 c1 c2 c3 c4 c5, ring B n6 c7 c8 c9 c10 c11, c5-c11 link, Pt12, Cl13, Cl14
        symbols = ['n', 'c', 'c', 'c', 'c', 'c', 'n', 'c', 'c', 'c', 'c', 'c', '[Pt]', 'Cl', 'Cl']
        ring_a = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 0, 4)]
        ring_b = [(6, 7, 4), (7, 8, 4), (8, 9, 4), (9, 10, 4), (10, 11, 4), (11, 6, 4)]
        bonds = ring_a + ring_b + [(5, 11, 1), (0, 12, 1), (6, 12, 1), (12, 13, 1), (12, 14, 1)]
        smiles, _, ok = _convert_graph_to_smiles(*decoder_graph(symbols, bonds))
        self.assertTrue(ok, smiles)
        self.assertEqual(metal_key(smiles), metal_key('[Cl-]->[Pt+2]1(<-[Cl-])<-[n]2ccccc2-c2cccc[n]->12'), smiles)

    def test_aromatic_cp_through_centroid(self):
        """Circle-drawn Cp: aromatic neutral c1cccc1 + Ct; the molblock step used to kekulize it and fail."""
        from molscribe.chemistry import _convert_graph_to_smiles
        symbols = ['c', 'c', 'c', 'c', 'c', '[Ct]', '[Fe]', 'Cl']
        bonds = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 0, 4), (5, 6, 1), (6, 7, 1)]
        coords = [(0.50, 0.10), (0.62, 0.18), (0.58, 0.30), (0.42, 0.30), (0.38, 0.18),
                  (0.50, 0.22), (0.50, 0.60), (0.80, 0.60)]
        smiles, _, ok = _convert_graph_to_smiles(*decoder_graph(symbols, bonds, coords))
        self.assertTrue(ok, smiles)
        self.assertIsNotNone(Chem.MolFromSmiles(smiles), smiles)

    def test_cp_iron_dicarbonyl_chloride_from_decoder_graph(self):
        from molscribe.chemistry import _convert_graph_to_smiles
        symbols = ['C', 'C', 'C', 'C', 'C', '[Ct]', '[Fe]', '[CO]', '[CO]', 'Cl']
        coords = [(0.50, 0.10), (0.62, 0.18), (0.58, 0.30), (0.42, 0.30), (0.38, 0.18),
                  (0.50, 0.22), (0.50, 0.60), (0.30, 0.80), (0.70, 0.80), (0.80, 0.55)]
        n = len(symbols)
        edges = [[0] * n for _ in range(n)]
        for i, j, t in [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 0, 1),
                        (5, 6, 1), (6, 7, 1), (6, 8, 1), (6, 9, 1)]:
            edges[i][j] = edges[j][i] = t
        smiles, _, ok = _convert_graph_to_smiles(coords, symbols, edges)
        self.assertTrue(ok)
        self.assertIsNotNone(Chem.MolFromSmiles(smiles), smiles)
        gold = Chem.RWMol(Chem.MolFromSmiles('[cH-]1cccc1.[Fe+]Cl.[C-]#[O+].[C-]#[O+]'))
        fe = [a.GetIdx() for a in gold.GetAtoms() if a.GetSymbol() == 'Fe'][0]
        for i in list(range(5)) + [a.GetIdx() for a in gold.GetAtoms()
                                   if a.GetSymbol() == 'C' and a.GetFormalCharge() == -1 and not a.IsInRing()]:
            gold.AddBond(i, fe, Chem.BondType.DATIVE)
        gold = Chem.MolToSmiles(gold)
        self.assertIsNotNone(metal_key(gold), gold)
        self.assertEqual(metal_key(smiles), metal_key(gold), f'\n{smiles}\n{gold}')


if __name__ == '__main__':
    unittest.main()
