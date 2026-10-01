"""Testes da CLI `--apenas` (execução parcial por bloco).

Cobre a função pura `flags_para_apenas` (quais RUN_* ligam/desligam), o sufixo de pasta
(`sufixo_apenas`, nunca `_completa`) e o parse do argparse (choices e múltiplos blocos),
sem executar o experimento.

Execução: .venv/bin/python -m unittest tests.test_apenas_flag
"""
import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402

TODAS = [f for flags in dm.BLOCOS_APENAS.values() for f in flags]


class TestFlagsParaApenas(unittest.TestCase):
    def test_todas_as_flags_existem_no_modulo(self):
        for flag in TODAS:
            self.assertTrue(hasattr(dm, flag), flag)
            self.assertIsInstance(getattr(dm, flag), bool, flag)

    def test_bloco3_sozinho(self):
        flags = dm.flags_para_apenas(["bloco3"])
        self.assertEqual(set(flags), set(TODAS))
        self.assertTrue(flags["RUN_HOMEM_MULHER"])
        for flag in TODAS:
            if flag != "RUN_HOMEM_MULHER":
                self.assertFalse(flags[flag], flag)

    def test_oraculo_nao_liga_llm(self):
        flags = dm.flags_para_apenas(["oraculo"])
        self.assertTrue(flags["RUN_ORACLE_VALIDATION"])
        self.assertFalse(flags["RUN_PHASES_ABC"])
        self.assertFalse(flags["RUN_PHASE_E"])
        self.assertFalse(flags["RUN_PROBLEM_MEIALUA"])
        self.assertFalse(flags["RUN_HOMEM_MULHER"])

    def test_combinacao_de_blocos(self):
        flags = dm.flags_para_apenas(["bloco3", "bloco2"])
        self.assertTrue(flags["RUN_PHASE_E"])
        self.assertTrue(flags["RUN_PROBLEM_MEIALUA"])
        self.assertTrue(flags["RUN_HOMEM_MULHER"])
        self.assertFalse(flags["RUN_PHASES_ABC"])
        self.assertFalse(flags["RUN_R3R4_EXPERIMENT"])
        self.assertFalse(flags["RUN_ORACLE_VALIDATION"])

    def test_todos_os_blocos_equivale_a_tudo_ligado(self):
        flags = dm.flags_para_apenas(list(dm.BLOCOS_APENAS))
        self.assertTrue(all(flags.values()))

    def test_bloco_desconhecido_ou_vazio(self):
        with self.assertRaises(ValueError):
            dm.flags_para_apenas(["bloco9"])
        with self.assertRaises(ValueError):
            dm.flags_para_apenas([])

    def test_nao_toca_subflags(self):
        # Sub-flags gateadas pelo bloco-pai não entram no dicionário.
        flags = dm.flags_para_apenas(["bloco1"])
        for sub in ("RUN_CLASS_ORDER_BIAS", "RUN_DILUTION", "RUN_HM_CLASS_NAMES_AB",
                    "RUN_ORACLE_MEIALUA", "RUN_CLASSICAL_BASELINES"):
            self.assertNotIn(sub, flags)


class TestSufixoApenas(unittest.TestCase):
    def test_um_bloco(self):
        self.assertEqual(dm.sufixo_apenas(["bloco3"]), "apenas-bloco3")

    def test_ordem_canonica_e_sem_completa(self):
        s = dm.sufixo_apenas(["bloco3", "oraculo", "bloco1"])
        self.assertEqual(s, "apenas-bloco1+bloco3+oraculo")
        self.assertNotIn("completa", s)


class TestArgparse(unittest.TestCase):
    def _parser(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--apenas", nargs="+", choices=list(dm.BLOCOS_APENAS), default=None)
        return p

    def test_aceita_multiplos(self):
        args = self._parser().parse_args(["--apenas", "bloco2", "bloco3"])
        self.assertEqual(args.apenas, ["bloco2", "bloco3"])

    def test_rejeita_invalido(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["--apenas", "bloco9"])


if __name__ == "__main__":
    unittest.main()
