"""Reestimação offline (`--reestimar-de`, 15/10/2026): rótulos do LLM reaproveitados de outra execução."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dissertacao_mestrado as dm  # noqa: E402


class TestNomePasta(unittest.TestCase):
    def test_sufixo_reestimada(self):
        self.assertEqual(dm.nome_pasta_execucao("T", None, False, reestimada=True), "execucao_T_completa_reestimada")
        self.assertEqual(dm.nome_pasta_execucao("T", None, False), "execucao_T_completa")
        self.assertEqual(dm.nome_pasta_execucao("T", None, True, reestimada=True), "execucao_T_smoke_reestimada")


class TestColetaReproduzida(unittest.TestCase):
    def setUp(self):
        self._orig = dm.collect_llm_decisions
        self._model = dm.MODEL_NAME
        self._inter = list(dm.LLM_INTERACTIONS)
        dm.MODEL_NAME = "m/teste"
        self.X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        self.tmp = tempfile.mkdtemp()
        tag0 = "bloco1|problema=ABC|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42"
        tag4 = "bloco1|problema=ABC|passo=4|regime=exemplos_da_metrica|k=4|rep=0|seed=42"
        ex_orig = [(5.0, 5.0, "A"), (-5.0, -5.0, "B")]
        regs = []
        for (x1, x2), lab in zip(self.X, ["A", "B", "A"]):          # zero-shot: os 3 pontos
            regs.append({"point": {"x1": x1, "x2": x2}, "parsed_label": lab, "malformed": False, "model": "m/teste",
                         "tag": tag0, "prompt": dm._montar_prompt(x1, x2, "A", "B", None, "x1", "x2", None, "default")})
        for (x1, x2), lab in zip(self.X[:2], ["B", "B"]):          # Passo 4: só 2 pontos (o 3º era exemplo)
            regs.append({"point": {"x1": x1, "x2": x2}, "parsed_label": lab, "malformed": False, "model": "m/teste",
                         "tag": tag4, "prompt": dm._montar_prompt(x1, x2, "A", "B", ex_orig, "x1", "x2", None, "default")})
        with open(os.path.join(self.tmp, "llm_interactions.json"), "w") as fh:
            json.dump(regs, fh)
        self.tag4, self.ex_orig = tag4, ex_orig
        dm._instalar_reestimacao_offline(self.tmp)

    def tearDown(self):
        dm.collect_llm_decisions = self._orig
        dm.MODEL_NAME = self._model
        dm.LLM_INTERACTIONS[:] = self._inter
        dm._REESTIMACAO_DE = None
        dm._REESTIMACAO_RELATORIO = None

    def test_prompt_identico_e_exato(self):
        y, mal = dm.collect_llm_decisions(self.X[:2], "A", "B", examples=self.ex_orig, verbose=False, tag=self.tag4)
        np.testing.assert_array_equal(y, [1, 1])
        self.assertEqual(mal, 0)

    def test_prompt_diferente_cai_no_mesmo_ponto_e_depois_no_zero_shot(self):
        ex_novo = [(5.0, 5.0, "B"), (-5.0, -5.0, "A")]      # rótulos dos exemplos mudaram → prompt diferente
        y, _ = dm.collect_llm_decisions(self.X, "A", "B", examples=ex_novo, verbose=False, tag=self.tag4)
        np.testing.assert_array_equal(y, [1, 1, 0])          # 2 pelo mesmo ponto/tag, o 3º pelo zero-shot ("A")
        pasta = tempfile.mkdtemp()
        dm._REESTIMACAO_RELATORIO(pasta)
        self.assertTrue(os.path.exists(os.path.join(pasta, "reestimacao_lacunas.csv")))
        self.assertTrue(os.path.exists(os.path.join(pasta, "reestimacao_origem.txt")))
        import pandas as pd
        df = pd.read_csv(os.path.join(pasta, "reestimacao_lacunas.csv"))
        self.assertEqual(int(df["n_aproximados"].sum()), 2)
        self.assertEqual(int(df["n_aproximados_zero_shot"].sum()), 1)

    def test_ponto_desconhecido_e_erro(self):
        with self.assertRaises(RuntimeError):
            dm.collect_llm_decisions(np.array([[9.0, 9.0]]), "A", "B", verbose=False, tag="bloco1|problema=ABC|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42")

    def test_lote_usa_a_coleta_reproduzida(self):
        res = dm.collect_llm_decisions_lote([
            {"X": self.X[:2], "nome_classe_0": "A", "nome_classe_1": "B", "examples": self.ex_orig, "verbose": False, "tag": self.tag4},
            {"X": self.X, "nome_classe_0": "A", "nome_classe_1": "B", "verbose": False,
             "tag": "bloco1|problema=ABC|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42"},
        ])
        np.testing.assert_array_equal(res[0][0], [1, 1])
        np.testing.assert_array_equal(res[1][0], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
