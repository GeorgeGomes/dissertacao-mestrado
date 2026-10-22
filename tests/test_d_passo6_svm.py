"""Problema D, Passo 6 (16/10/2026): números do SVM RBF da figura bloco1_graphic_14 e a tabela 19."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classical_baselines import d_passo6_svm_medidas  # noqa: E402
from data_problems import create_problem_d_meialua  # noqa: E402
import gerar_tabelas as gt  # noqa: E402


class TestMedidas(unittest.TestCase):
    def setUp(self):
        self.X, self.y = create_problem_d_meialua(n_samples=150, random_state=42)[:2]

    def test_svm_no_rotulo_real_separa_a_meia_lua(self):
        med = d_passo6_svm_medidas(self.X, self.y, self.y)
        self.assertGreater(med["svm_real_accuracy_vs_true"], 0.95)
        self.assertEqual(med["svm_real_n_errors"], int(round((1 - med["svm_real_accuracy_vs_true"]) * 150)))
        self.assertEqual(med["svm_llm_fidelity_vs_llm"], med["svm_real_accuracy_vs_true"])   # mesmos rótulos → mesmo SVM
        self.assertEqual(med["collapsed"], 0)

    def test_colapso_deixa_o_svm_do_llm_em_nan(self):
        y_llm = np.zeros_like(self.y)
        med = d_passo6_svm_medidas(self.X, self.y, y_llm, metric={"n_features": 2, "fidelity_perc_vs_llm": 1.0, "accuracy_perc_vs_true": 0.5})
        self.assertEqual(med["collapsed"], 1)
        self.assertTrue(np.isnan(med["svm_llm_fidelity_vs_llm"]))
        self.assertFalse(np.isnan(med["svm_real_accuracy_vs_true"]))
        self.assertEqual(med["metric_n_features"], 2)

    def test_rotulos_ruidosos_do_llm(self):
        rs = np.random.RandomState(0)
        y_llm = self.y.copy(); flip = rs.choice(150, 20, replace=False); y_llm[flip] = 1 - y_llm[flip]
        med = d_passo6_svm_medidas(self.X, self.y, y_llm)
        self.assertAlmostEqual(med["llm_accuracy_vs_true"], 130 / 150)
        self.assertGreater(med["svm_llm_fidelity_vs_llm"], 0.7)
        self.assertLess(med["svm_llm_accuracy_vs_true"], 1.0)


class TestTabela19(unittest.TestCase):
    def test_tabela_por_modelo_com_media_e_colapso(self):
        rows = []
        for seed, col in ((42, 0), (123, 1), (7, 0)):
            rows.append({"provider": "openrouter", "model": "m/x", "random_seed": seed, "n_pontos": 150,
                         "n_classe_0_llm": 150 if col else 80, "n_classe_1_llm": 0 if col else 70, "collapsed": col,
                         "llm_accuracy_vs_true": 0.5 if col else 0.8, "svm_real_accuracy_vs_true": 0.98, "svm_real_n_errors": 3,
                         "svm_llm_fidelity_vs_llm": np.nan if col else 0.9, "svm_llm_accuracy_vs_true": np.nan if col else 0.85,
                         "svm_llm_n_errors_vs_true": np.nan if col else 22, "metric_n_features": np.nan if col else 4,
                         "metric_fidelity_vs_llm": np.nan if col else 0.92, "metric_accuracy_vs_true": np.nan if col else 0.82})
        t = gt.tab_d_passo6_svm({"d_passo6_svm": pd.DataFrame(rows)}, "m/x")
        self.assertIsNotNone(t)
        self.assertEqual(t.nn, 19)
        self.assertEqual(len(t.rows), 4)
        self.assertEqual([r[0] for r in t.rows], ["7", "42", "123", "média"])   # ordenada por semente
        self.assertIn("colapso", t.rows[2][1])
        self.assertEqual(t.rows[2][5], "—")
        self.assertEqual(t.rows[-1][0], "média")
        self.assertIn("1 semente(s)", t.rodape)
        self.assertIn("bloco1_table_19_D_passo6_svm", t.nome_png())
        self.assertIsNone(gt.tab_d_passo6_svm({"d_passo6_svm": pd.DataFrame(rows)}, "outro"))
        self.assertIsNone(gt.tab_d_passo6_svm({}, "m/x"))


if __name__ == "__main__":
    unittest.main()
