"""
Testes do comando `physchem` contra valores de referência.

Referências:
- ExPASy ProtParam para o 1TUP (debug/02-physchem/1UTP_ProParam_Predicion.csv):
  pI 8.34, índice de instabilidade 73.97, meia-vida em E. coli >10 horas.
- Tabela de meia-vida (E. coli, in vivo) da documentação do ProtParam:
  https://web.expasy.org/protparam/protparam-doc.html
- Demais valores de pI e índice de instabilidade calculados com Biopython
  (Bio.SeqUtils.ProtParam), que implementa os mesmos métodos do ProtParam.

Execução (a partir da raiz do repositório):
    python -m unittest discover -s tests -v
"""

import csv
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIOHUB = os.path.join(REPO_ROOT, "biohub.py")
PDB_1TUP = os.path.join(REPO_ROOT, "debug", "00-fetchpdb", "1TUP.pdb")

sys.path.insert(0, REPO_ROOT)
import biohub  # noqa: E402


def physchem(sequence):
    """Executa `biohub.py physchem` e devolve o CSV como dicionário Propriedade -> Valor."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "physchem.csv")
        result = subprocess.run(
            [sys.executable, BIOHUB, "physchem", sequence, "-o", out],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise AssertionError(f"physchem falhou:\n{result.stderr}")
        with open(out, newline="", encoding="utf-8") as f:
            return {row[0]: row[1] for row in list(csv.reader(f))[1:]}


def composition(sequence):
    return {aa: sequence.count(aa) for aa in biohub.MOLECULAR_WEIGHT}


class TestPhyschem1TUPAgainstProtParam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequence = biohub.get_sequence_from_pdb(PDB_1TUP)
        cls.result = physchem(cls.sequence)

    def test_isoelectric_point(self):
        self.assertEqual(self.result["Ponto Isoelétrico (pI)"], "8.34")

    def test_instability_index(self):
        self.assertEqual(self.result["Índice de Instabilidade"], "73.97 (Instável)")

    def test_half_life(self):
        self.assertEqual(self.result["Meia-Vida (E. coli, in vivo)"], ">10 horas")

    def test_unchanged_properties(self):
        # Propriedades que já concordavam com o ProtParam continuam iguais.
        self.assertEqual(self.result["Comprimento"], "196")
        self.assertEqual(self.result["Peso Molecular (Da)"], "22003.86")
        self.assertEqual(self.result["GRAVY (Hidropaticidade)"], "-0.503")
        self.assertEqual(self.result["Índice Alifático"], "65.56")
        self.assertEqual(self.result["Total de Resíduos Ácidos (Asp+Glu)"], "19")


class TestHalfLifeTable(unittest.TestCase):
    # Coluna "E. coli" da tabela da documentação do ProtParam ("?" para Pro).
    PROTPARAM_E_COLI = {
        'A': '>10 horas', 'R': '2 min', 'N': '>10 horas', 'D': '>10 horas', 'C': '>10 horas',
        'Q': '>10 horas', 'E': '>10 horas', 'G': '>10 horas', 'H': '>10 horas', 'I': '>10 horas',
        'L': '2 min', 'K': '2 min', 'M': '>10 horas', 'F': '2 min', 'P': 'Desconhecido',
        'S': '>10 horas', 'T': '>10 horas', 'W': '2 min', 'Y': '2 min', 'V': '>10 horas',
    }

    def test_every_n_terminal_residue(self):
        for aa, expected in self.PROTPARAM_E_COLI.items():
            with self.subTest(n_terminal=aa):
                result = physchem(aa + "GAGA")
                self.assertEqual(result["Meia-Vida (E. coli, in vivo)"], expected)


class TestIsoelectricPoint(unittest.TestCase):
    BIOPYTHON_REFERENCE = {
        # sequência: pI (Biopython)
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ": 9.9880,  # N-terminal Met (pKa específico)
        "SHHHHHHGKD": 6.9920,                         # N-terminal Ser, C-terminal Asp
        "AWGYRCKLVNPE": 8.2414,                       # N-terminal Ala, C-terminal Glu
        "TTTKRRRHD": 11.7113,                         # proteína muito básica
    }

    def test_matches_biopython(self):
        for sequence, expected in self.BIOPYTHON_REFERENCE.items():
            with self.subTest(sequence=sequence):
                pi = biohub.calculate_isoelectric_point(sequence, composition(sequence))
                self.assertAlmostEqual(pi, expected, delta=0.001)

    def test_net_charge_is_zero_at_pi(self):
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        comp = composition(sequence)
        pi = biohub.calculate_isoelectric_point(sequence, comp)
        self.assertAlmostEqual(biohub.net_charge_at_ph(pi, sequence, comp), 0.0, places=3)

    def test_very_acidic_and_very_basic_proteins(self):
        # pI fora da faixa 4-12 continua sendo encontrado (busca entre pH 0 e 14).
        acidic = "D" * 30 + "E" * 30
        basic = "K" * 30 + "R" * 30
        self.assertLess(biohub.calculate_isoelectric_point(acidic, composition(acidic)), 4.0)
        self.assertGreater(biohub.calculate_isoelectric_point(basic, composition(basic)), 12.0)


class TestInstabilityIndex(unittest.TestCase):
    BIOPYTHON_REFERENCE = {
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ": "53.55",
        "SHHHHHHGKD": "-9.86",
        "AWGYRCKLVNPE": "-15.63",
        "TTTKRRRHD": "193.80",
    }

    def test_matches_biopython(self):
        for sequence, expected in self.BIOPYTHON_REFERENCE.items():
            with self.subTest(sequence=sequence):
                value = physchem(sequence)["Índice de Instabilidade"].split()[0]
                self.assertEqual(value, expected)

    def test_previously_wrong_dipeptide_weights(self):
        # Células da matriz DIWV que divergiam de Guruprasad et al. (1990).
        expected = {('C', 'P'): 20.26, ('C', 'R'): 1.0, ('H', 'K'): 24.68, ('N', 'K'): 24.68,
                    ('P', 'D'): -6.54, ('S', 'S'): 20.26, ('F', 'Y'): 33.601, ('K', 'Q'): 24.64}
        for (first, second), weight in expected.items():
            with self.subTest(dipeptide=first + second):
                self.assertEqual(biohub.DIWV[first][second], weight)


if __name__ == "__main__":
    unittest.main()
