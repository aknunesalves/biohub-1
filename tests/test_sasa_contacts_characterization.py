"""
Testes de caracterização dos comandos `sasa` e `contacts`.

Os arquivos em tests/data/characterization/ foram gerados pela implementação
original (força bruta) a partir de debug/00-fetchpdb/1TUP.pdb. As versões
otimizadas precisam reproduzir essas saídas exatamente.

Execução (a partir da raiz do repositório):
    python -m unittest discover -s tests -v
"""

import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIOHUB = os.path.join(REPO_ROOT, "biohub.py")
PDB_1TUP = os.path.join(REPO_ROOT, "debug", "00-fetchpdb", "1TUP.pdb")
EXPECTED_DIR = os.path.join(REPO_ROOT, "tests", "data", "characterization")

sys.path.insert(0, REPO_ROOT)
import biohub  # noqa: E402


def run_biohub(*args, cwd):
    """Executa a CLI como o usuário faria e falha o teste se o processo falhar."""
    result = subprocess.run(
        [sys.executable, BIOHUB, *args],
        cwd=cwd, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(f"biohub.py {' '.join(args)} falhou:\n{result.stderr}")
    return result


def read_lines(path):
    # Modo texto normaliza CRLF/LF, que variam entre sistemas operacionais.
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


class CharacterizationTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def assertSameOutput(self, produced_name, expected_name):
        produced = read_lines(os.path.join(self.tmp, produced_name))
        expected = read_lines(os.path.join(EXPECTED_DIR, expected_name))
        self.assertEqual(len(produced), len(expected), f"{produced_name}: número de linhas diferente")
        for line_no, (got, want) in enumerate(zip(produced, expected), start=1):
            self.assertEqual(got, want, f"{produced_name}: linha {line_no} diferente")


class TestSasaCharacterization(CharacterizationTestCase):
    def test_sasa_default_points_matches_original(self):
        run_biohub("sasa", PDB_1TUP, "-o", "sasa.csv", cwd=self.tmp)
        self.assertSameOutput("sasa.csv", "sasa_200.csv")

    def test_sasa_custom_points_and_annotated_pdb_match_original(self):
        run_biohub("sasa", PDB_1TUP, "--num-points", "50", "-o", "sasa.csv",
                   "--write-pdb", "sasa.pdb", cwd=self.tmp)
        self.assertSameOutput("sasa.csv", "sasa_50.csv")
        self.assertSameOutput("sasa.pdb", "sasa_50.pdb")


class TestContactsCharacterization(CharacterizationTestCase):
    def test_contacts_default_threshold_matches_original(self):
        run_biohub("contacts", PDB_1TUP, "-o", "contacts.csv", cwd=self.tmp)
        self.assertSameOutput("contacts.csv", "contacts_8A.csv")

    def test_contacts_custom_threshold_matches_original(self):
        run_biohub("contacts", PDB_1TUP, "-t", "4.5", "-o", "contacts.csv", cwd=self.tmp)
        self.assertSameOutput("contacts.csv", "contacts_4.5A.csv")


class TestSasaEdgeCases(unittest.TestCase):
    def _atom(self, x, y, z, element="C"):
        return {"x": x, "y": y, "z": z, "element": element}

    def test_empty_atom_list(self):
        self.assertEqual(biohub.compute_sasa_per_atom([], 50, 1.4), [])

    def test_non_positive_num_points_returns_zero(self):
        atoms = [self._atom(0, 0, 0), self._atom(1, 0, 0)]
        self.assertEqual(biohub.compute_sasa_per_atom(atoms, 0, 1.4), [0, 0])

    def test_isolated_atom_is_fully_exposed(self):
        # Átomo isolado: SASA = área da esfera estendida (raio VdW + sonda).
        sasa = biohub.compute_sasa_per_atom([self._atom(0, 0, 0)], 100, 1.4)[0]
        expected = 4.0 * 3.141592653589793 * (biohub.VDW_RADII["C"] + 1.4) ** 2
        self.assertAlmostEqual(sasa, expected, places=9)

    def test_distant_atoms_do_not_occlude_each_other(self):
        far = [self._atom(0, 0, 0), self._atom(100, 0, 0)]
        alone = biohub.compute_sasa_per_atom([self._atom(0, 0, 0)], 100, 1.4)[0]
        self.assertEqual(biohub.compute_sasa_per_atom(far, 100, 1.4), [alone, alone])


class TestContactsEdgeCases(unittest.TestCase):
    def test_sequence_neighbors_are_ignored(self):
        residue_atoms = {1: [(0.0, 0.0, 0.0)], 2: [(1.0, 0.0, 0.0)]}
        self.assertEqual(biohub.find_residue_contacts(residue_atoms, 8.0), [])

    def test_contact_exactly_at_threshold_is_included(self):
        residue_atoms = {1: [(0.0, 0.0, 0.0)], 5: [(8.0, 0.0, 0.0)]}
        self.assertEqual(biohub.find_residue_contacts(residue_atoms, 8.0), [(1, 5, 8.0)])

    def test_distant_residues_are_not_contacts(self):
        residue_atoms = {1: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 5: [(50.0, 0.0, 0.0)]}
        self.assertEqual(biohub.find_residue_contacts(residue_atoms, 8.0), [])


if __name__ == "__main__":
    unittest.main()
