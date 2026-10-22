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
        self.assertTrue(flags["RUN_PROBLEMA_G"])
        for flag in TODAS:
            if flag != "RUN_PROBLEMA_G":
                self.assertFalse(flags[flag], flag)

    def test_bloco1_liga_problema_d_e_nao_o_problema_f(self):
        # Meia-lua: Problema D (Bloco 1, Passos 2 a 5 como o A, sem split) é
        # RUN_PROBLEMA_D; o Problema F (aprendiz, split 70/30) fica no bloco2.
        flags = dm.flags_para_apenas(["bloco1"])
        self.assertTrue(flags["RUN_PROBLEMAS_ABC"])
        self.assertTrue(flags["RUN_A_PASSO4_R3R4"])
        self.assertTrue(flags["RUN_PROBLEMA_D"])
        self.assertFalse(flags["RUN_PROBLEMA_F"])
        self.assertFalse(flags["RUN_PROBLEMA_E"])
        self.assertFalse(flags["RUN_PROBLEMA_G"])
        self.assertFalse(flags["RUN_ORACULO"])

    def test_oraculo_nao_liga_llm(self):
        flags = dm.flags_para_apenas(["oraculo"])
        self.assertTrue(flags["RUN_ORACULO"])
        self.assertFalse(flags["RUN_PROBLEMAS_ABC"])
        self.assertFalse(flags["RUN_PROBLEMA_E"])
        self.assertFalse(flags["RUN_PROBLEMA_F"])
        self.assertFalse(flags["RUN_PROBLEMA_D"])
        self.assertFalse(flags["RUN_PROBLEMA_G"])

    def test_combinacao_de_blocos(self):
        flags = dm.flags_para_apenas(["bloco3", "bloco2"])
        self.assertTrue(flags["RUN_PROBLEMA_E"])
        self.assertTrue(flags["RUN_PROBLEMA_F"])
        self.assertTrue(flags["RUN_PROBLEMA_G"])
        self.assertFalse(flags["RUN_PROBLEMAS_ABC"])
        self.assertFalse(flags["RUN_A_PASSO4_R3R4"])
        self.assertFalse(flags["RUN_PROBLEMA_D"])
        self.assertFalse(flags["RUN_ORACULO"])

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
        for sub in ("RUN_VIES_ORDEM_CLASSES", "RUN_E_PASSO4_DILUICAO", "RUN_PROBLEMA_G_CLASSES_AB",
                    "RUN_D_PASSO5_TETO", "RUN_E_PASSO6_BASELINES"):
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


class TestOraculoAcumulaSementes(unittest.TestCase):
    """13/10/2026: o acumulador do oráculo era sobrescrito a cada seed (só a última sobrevivia, em dobro).
    A última execução `--apenas oraculo` deve ter 3 seeds × 3 problemas × 2 algoritmos = 18 linhas."""

    def test_csv_do_oraculo_tem_as_tres_sementes(self):
        import glob
        import pandas as pd
        pastas = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "resultados", "execucao_*_apenas-oraculo")))
        if not pastas:
            self.skipTest("nenhuma execução --apenas oraculo em resultados/")
        csvs = glob.glob(os.path.join(pastas[-1], "bloco1_oraculo_passos_1_a_3_*.csv"))
        if not csvs:
            self.skipTest("execução --apenas oraculo sem CSV do oráculo")
        df = pd.read_csv(csvs[0])
        if "Oráculo sobre B" in set(df["problem"]):
            self.skipTest("execução --apenas oraculo anterior a 14/10/2026 (oráculo sobre B)")
        # 14/10/2026: os três oráculos rodam sobre as gaussianas do E (E3 = Problema 1, E1, E2)
        self.assertEqual(set(df["problem"]), {"Oráculo sobre E3", "Oráculo sobre E1", "Oráculo sobre E2"})
        self.assertEqual(sorted(df["random_seed"].unique().tolist()), sorted(dm.RANDOM_SEEDS))
        self.assertEqual(len(df), len(dm.RANDOM_SEEDS) * 3 * 2)
        self.assertFalse(df.duplicated(["random_seed", "problem", "algorithm"]).any())
