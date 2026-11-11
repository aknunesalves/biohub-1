"""
Testes de robustez da CLI: códigos de saída, PDBs com múltiplos modelos e
conformações alternativas (altLoc), validação de sequência e timeout de download.

Execução (a partir da raiz do repositório):
    python -m unittest discover -s tests -v
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIOHUB = os.path.join(REPO_ROOT, "biohub.py")
PDB_1TUP = os.path.join(REPO_ROOT, "debug", "00-fetchpdb", "1TUP.pdb")
EXPECTED_DIR = os.path.join(REPO_ROOT, "tests", "data", "characterization")

sys.path.insert(0, REPO_ROOT)
import biohub  # noqa: E402


def run_biohub(*args, cwd):
    return subprocess.run([sys.executable, BIOHUB, *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def atom_lines_1tup(max_residues):
    """Linhas ATOM dos primeiros `max_residues` resíduos da cadeia A do 1TUP."""
    lines, residues = [], []
    with open(PDB_1TUP, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                res = line[22:27]
                if res not in residues:
                    if len(residues) == max_residues:
                        break
                    residues.append(res)
                lines.append(line.rstrip("\r\n"))
    return lines


def shift_x(line, delta):
    x = float(line[30:38]) + delta
    return f"{line[:30]}{x:8.3f}{line[38:]}"


def with_altloc(line, altloc):
    return line[:16] + altloc + line[17:]


def write_pdb(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\nEND\n")


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestExitCodes(TempDirTestCase):
    def assertFails(self, *args, message):
        result = run_biohub(*args, cwd=self.tmp)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(message, result.stderr)

    def test_missing_pdb_file_returns_1(self):
        for command in ("fasta", "contacts", "hydrophoby", "sasa"):
            with self.subTest(command=command):
                self.assertFails(command, "nao_existe.pdb", message="Arquivo não encontrado")

    def test_missing_csv_file_returns_1(self):
        self.assertFails("csv2fasta", "nao_existe.csv", message="Arquivo CSV não encontrado")

    def test_invalid_pdb_id_returns_1(self):
        # Rejeitado antes de qualquer acesso à rede.
        self.assertFails("fetchpdb", "ABC", message="inválido")

    def test_empty_sequence_returns_1(self):
        self.assertFails("physchem", "", message="vazia")

    def test_invalid_sequence_characters_return_1(self):
        result = run_biohub("physchem", "MKX*TB", cwd=self.tmp)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("caracteres inválidos: *, B, X", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_successful_commands_return_0(self):
        self.assertEqual(run_biohub("fasta", PDB_1TUP, cwd=self.tmp).returncode, 0)
        self.assertEqual(run_biohub("physchem", "mktayiakqr", cwd=self.tmp).returncode, 0)


class TestCleanPdbOutputsUnchanged(TempDirTestCase):
    """Arquivos sem MODEL e sem altLoc continuam gerando exatamente a mesma saída."""

    def assertSameAsExpected(self, produced, expected):
        self.assertEqual(read_lines(self.path(produced)), read_lines(os.path.join(EXPECTED_DIR, expected)))

    def test_fasta(self):
        result = run_biohub("fasta", PDB_1TUP, cwd=self.tmp)
        self.assertEqual(result.stdout.splitlines(), read_lines(os.path.join(EXPECTED_DIR, "fasta_stdout.txt")))

    def test_hydrophoby(self):
        run_biohub("hydrophoby", PDB_1TUP, "-o", "h.csv", "--write-pdb", "h.pdb", cwd=self.tmp)
        self.assertSameAsExpected("h.csv", "hydrophoby.csv")
        self.assertSameAsExpected("h.pdb", "hydrophoby.pdb")

    def test_contacts(self):
        run_biohub("contacts", PDB_1TUP, "-t", "4.5", "-o", "c.csv", cwd=self.tmp)
        self.assertSameAsExpected("c.csv", "contacts_4.5A.csv")


class TestStructureVariants(TempDirTestCase):
    """Variantes de um mesmo trecho do 1TUP que devem produzir o mesmo resultado do arquivo limpo."""

    COMMANDS = {
        "fasta": lambda pdb: ("fasta", pdb, "-o", "out.txt"),
        "contacts": lambda pdb: ("contacts", pdb, "-o", "out.txt"),
        "hydrophoby": lambda pdb: ("hydrophoby", pdb, "-o", "out.txt"),
        "sasa": lambda pdb: ("sasa", pdb, "--num-points", "30", "-o", "out.txt"),
    }

    def setUp(self):
        super().setUp()
        self.base = atom_lines_1tup(max_residues=15)
        write_pdb(self.path("clean.pdb"), self.base)

    def output_of(self, pdb_name, command):
        out_dir = tempfile.mkdtemp(dir=self.tmp)
        result = run_biohub(*self.COMMANDS[command](self.path(pdb_name)), cwd=out_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        return read_lines(os.path.join(out_dir, "out.txt")), result.stderr

    def assertSameAsClean(self, pdb_name):
        for command in self.COMMANDS:
            with self.subTest(command=command):
                expected, _ = self.output_of("clean.pdb", command)
                produced, _ = self.output_of(pdb_name, command)
                if command == "fasta":  # o cabeçalho FASTA traz o nome do arquivo
                    expected, produced = expected[1:], produced[1:]
                self.assertEqual(produced, expected)

    def test_multiple_models_use_only_the_first(self):
        # Como em arquivos de RMN: mesmos átomos (mesmos números) em outra conformação.
        model_2 = [shift_x(line, 0.7) for line in self.base]
        lines = ["MODEL        1", *self.base, "ENDMDL", "MODEL        2", *model_2, "ENDMDL"]
        write_pdb(self.path("multi.pdb"), lines)
        self.assertSameAsClean("multi.pdb")
        _, stderr = self.output_of("multi.pdb", "sasa")
        self.assertIn("múltiplos modelos", stderr)

    def test_alternate_locations_keep_only_the_first(self):
        # Resíduos 3 a 6 com conformações A (coordenadas originais) e B (deslocada).
        residues_with_altloc = sorted({line[22:27] for line in self.base})[2:6]
        lines, serial_b = [], 90000
        for line in self.base:
            if line[22:27] in residues_with_altloc:
                lines.append(with_altloc(line, "A"))
                serial_b += 1
                lines.append(with_altloc(f"{line[:6]}{serial_b:5d}{line[11:]}", "B")[:30]
                             + f"{float(line[30:38]) + 1.5:8.3f}" + line[38:])
            else:
                lines.append(line)
        write_pdb(self.path("altloc.pdb"), lines)
        self.assertSameAsClean("altloc.pdb")


class TestFetchPdbDownload(TempDirTestCase):
    def setUp(self):
        super().setUp()
        biohub._error_count = 0
        self.args = argparse.Namespace(pdb_id="1tup", output=self.path("1TUP.pdb"),
                                       chains=None, protein_only=False)

    def fetch(self):
        with contextlib.redirect_stderr(io.StringIO()):
            biohub.handle_fetch_pdb(self.args)

    def test_download_uses_timeout(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"HEADER    TEST\nEND\n"
        with mock.patch.object(biohub.urllib.request, "urlopen", return_value=response) as urlopen:
            self.fetch()
        self.assertEqual(urlopen.call_args.kwargs.get("timeout"), biohub.DOWNLOAD_TIMEOUT_SECONDS)
        self.assertEqual(biohub._error_count, 0)
        self.assertTrue(os.path.exists(self.args.output))

    def test_timeout_is_reported_as_error(self):
        with mock.patch.object(biohub.urllib.request, "urlopen", side_effect=TimeoutError("timed out")):
            self.fetch()
        self.assertEqual(biohub._error_count, 1)

    def test_http_error_is_reported_as_error(self):
        error = urllib.error.HTTPError("https://files.rcsb.org/download/1TUP.pdb", 404, "Not Found", {}, None)
        with mock.patch.object(biohub.urllib.request, "urlopen", side_effect=error):
            self.fetch()
        self.assertEqual(biohub._error_count, 1)


class TestReportError(unittest.TestCase):
    def test_main_exits_with_1_after_reported_error(self):
        biohub._error_count = 0
        with mock.patch.object(sys, "argv", ["biohub.py", "fasta", "nao_existe.pdb"]), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                biohub.main()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
