"""Testes das funções estatísticas de ``relatorios.py``.

Prioridade: ``bootstrap_ci`` — é o número que vira "IC 95%" impresso na
dissertação. Contratos verificados:

1. Dados constantes → IC degenerado no próprio valor (média = lo = hi).
2. Menos de 2 pontos → devolve (média, média, média) sem quebrar.
3. Ordem: lo ≤ média ≤ hi em amostra real.
4. Determinismo: duas chamadas com os mesmos dados devolvem o MESMO IC
   (seed interna fixa — reprodutibilidade bit-a-bit do IC relatado).
5. Sanidade estatística: para N(0,1) com n=200, a meia-largura do IC 95%
   fica próxima de 1.96/√200 ≈ 0.139 (aproximação normal).
6. Consistência: mais dados → IC mais estreito.

Execução: .venv/bin/python -m unittest tests.test_relatorios
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

from relatorios import bootstrap_ci, summarize_cross_linearity, PROBLEMA_LINEAR_R3R4  # noqa: E402


class TestBootstrapCI(unittest.TestCase):
    def test_dados_constantes_ic_degenerado(self):
        mean, lo, hi = bootstrap_ci(np.full(30, 0.75))
        self.assertEqual((mean, lo, hi), (0.75, 0.75, 0.75))

    def test_menos_de_dois_pontos_nao_quebra(self):
        mean, lo, hi = bootstrap_ci(np.array([0.4]))
        self.assertEqual((mean, lo, hi), (0.4, 0.4, 0.4))

    def test_ordem_lo_media_hi(self):
        rng = np.random.RandomState(7)
        data = rng.normal(0.8, 0.1, size=50)
        mean, lo, hi = bootstrap_ci(data)
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)
        self.assertAlmostEqual(mean, float(np.mean(data)), places=12)

    def test_deterministico_entre_chamadas(self):
        rng = np.random.RandomState(11)
        data = rng.normal(0.5, 0.2, size=40)
        self.assertEqual(bootstrap_ci(data), bootstrap_ci(data))

    def test_meia_largura_compativel_com_aproximacao_normal(self):
        # N(0,1), n=200: IC 95% da média ≈ ±1.96/√200 ≈ ±0.1386. O bootstrap
        # percentil deve chegar perto (tolerância folgada p/ ruído de amostra).
        rng = np.random.RandomState(42)
        data = rng.normal(0.0, 1.0, size=200)
        mean, lo, hi = bootstrap_ci(data)
        meia_largura = (hi - lo) / 2
        self.assertGreater(meia_largura, 0.10)
        self.assertLess(meia_largura, 0.18)

    def test_mais_dados_estreitam_o_ic(self):
        rng = np.random.RandomState(3)
        base = rng.normal(0.0, 1.0, size=400)
        _, lo_p, hi_p = bootstrap_ci(base[:50])
        _, lo_g, hi_g = bootstrap_ci(base)
        self.assertLess(hi_g - lo_g, hi_p - lo_p)


class TestSummarizeCrossLinearity(unittest.TestCase):
    """O CSV cruzado deve ter R2/R3/R4 do Problema A (perceptron E nnls) com o alias."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inclui_r2_r3_r4_dois_algoritmos_e_rotulo_problema_a(self):
        r2 = [{'accuracy': .80, 'accuracy_nnls': .81, 'seed': 42}]
        r3 = [{'accuracy': .85, 'accuracy_nnls': .86, 'seed': 42}]
        r4 = [{'accuracy': .84, 'accuracy_nnls': .88, 'seed': 42}]
        f = summarize_cross_linearity(r3, [], self.tmp, results_abc_r4=r4,
                                      model_name="gpt-4o-mini", results_abc_r2=r2)
        self.assertEqual(os.path.basename(f), "final_cross_linearity__gpt4mini.csv")
        df = pd.read_csv(f)
        self.assertEqual(set(df['problem']), {PROBLEMA_LINEAR_R3R4})
        self.assertEqual(PROBLEMA_LINEAR_R3R4, "A_linear")
        self.assertEqual(sorted(df['n_features'].unique()), [2, 3, 4])
        self.assertEqual(set(df['algorithm']), {"perceptron", "nnls"})
        self.assertEqual(len(df), 6)

    def test_r2_sem_nnls_gera_so_perceptron(self):
        r2 = [{'accuracy': .80, 'accuracy_nnls': None, 'seed': 7}]
        r3 = [{'accuracy': .85, 'seed': 7}]
        f = summarize_cross_linearity(r3, [], self.tmp, model_name="google/gemini-2.5-flash-lite",
                                      results_abc_r2=r2)
        self.assertTrue(f.endswith("final_cross_linearity__flashlite.csv"))
        df = pd.read_csv(f)
        self.assertEqual(set(df['algorithm']), {"perceptron"})
        self.assertEqual(sorted(df['n_features'].unique()), [2, 3])


if __name__ == "__main__":
    unittest.main()
