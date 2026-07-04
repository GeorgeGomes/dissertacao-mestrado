"""Teste de regressão DOURADO da cadeia numérica, sem chamar a API.

Reconstrói os rótulos da Fase A (seed 42, config default) a partir do
`llm_interactions_parte001.json` da última execução completa e verifica que a
cadeia determinística dados -> centróides -> NNLS reproduz EXATAMENTE o W e a
fidelidade registrados no `bloco1_algorithm_comparison_*.csv` daquela execução.

Serve de rede de segurança para refatorações (ex.: extração de módulos): se a
movimentação de código alterar qualquer coisa na cadeia numérica, este teste
quebra — sem gastar um centavo de API.

Pula silenciosamente se a pasta da execução de referência não existir
(ex.: clone limpo do repositório sem os artefatos de execução).
"""
import json
import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Execução completa de referência (1 modelo, 3 seeds). Valores dourados abaixo
# foram extraídos de bloco1_algorithm_comparison_20260701_030559.csv (seed 42).
EXEC_REF = os.path.join(BASE_DIR, "execucao_2026-06-30_23-46-58")
INTERACOES = os.path.join(EXEC_REF, "llm_interactions_parte001.json")

W_NNLS_GOLDEN = np.array([0.035725, 0.141970])   # w_0, w_1 (CSV, 6 casas)
FID_NNLS_GOLDEN = 0.820000                        # fidelity_problem_a (CSV)
FID_PERC_STORED = 0.766667                        # fidelidade do W-perceptron armazenado
W_PERC_STORED = np.array([0.000232, 0.002469])    # bloco1_phases_abc (seed 42, default)


@unittest.skipUnless(os.path.exists(INTERACOES), "execução de referência ausente")
class TestRegressaoCadeiaNumerica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_problems import create_problem_a
        from metrics import compute_centroids

        with open(INTERACOES) as fh:
            recs = json.load(fh)[:150]  # os 150 primeiros = Fase A seed 42
        cls.X_a, _ = create_problem_a(150, 42)
        lab = {
            (round(r["point"]["x1"], 9), round(r["point"]["x2"], 9)): r["parsed_label"]
            for r in recs
        }
        cls.y_llm = np.array(
            [0 if lab[(round(x1, 9), round(x2, 9))] == "A" else 1 for x1, x2 in cls.X_a]
        )
        cls.centroids = compute_centroids(cls.X_a, cls.y_llm)

    def test_registros_cobrem_o_problema_a(self):
        # Garante que a reconstrução mapeou todos os 150 pontos (sem KeyError já é
        # meio caminho; aqui fixamos o balanço de classes observado na execução).
        self.assertEqual(len(self.y_llm), 150)
        self.assertEqual(set(np.unique(self.y_llm)), {0, 1})

    def test_nnls_reproduz_w_e_fidelidade_do_csv(self):
        from least_squares_inverse import train_least_squares_inverse
        from metrics import predict_with_metric

        w, _ = train_least_squares_inverse(self.X_a, self.y_llm, self.centroids)
        np.testing.assert_allclose(w, W_NNLS_GOLDEN, atol=5e-7)
        fid = (predict_with_metric(self.X_a, self.centroids, w) == self.y_llm).mean()
        self.assertAlmostEqual(fid, FID_NNLS_GOLDEN, places=6)

    def test_fidelidade_do_w_perceptron_armazenado(self):
        from metrics import predict_with_metric

        fid = (
            predict_with_metric(self.X_a, self.centroids, W_PERC_STORED) == self.y_llm
        ).mean()
        self.assertAlmostEqual(fid, FID_PERC_STORED, places=6)

    def test_perceptron_atual_reproduz_direcao_do_w(self):
        # O algoritmo do perceptron pode evoluir (ex.: fix da busca em γ de 03/07);
        # o contrato de regressão é a DIREÇÃO do W (cosseno ~1 com o armazenado),
        # não igualdade bit-a-bit de γ.
        from relaxed_perceptron import train_relaxed_perceptron

        w, _ = train_relaxed_perceptron(
            self.X_a, self.y_llm, self.centroids,
            eta=0.001, C=1.0, delta_gamma=0.05, max_epochs=50, tol=1e-4,
            use_best_effort=True,
        )
        cos = float(
            np.dot(w, W_PERC_STORED)
            / (np.linalg.norm(w) * np.linalg.norm(W_PERC_STORED) + 1e-12)
        )
        self.assertGreater(cos, 0.9999)


if __name__ == "__main__":
    unittest.main()
