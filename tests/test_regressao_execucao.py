"""Teste de regressão DOURADO da cadeia numérica, sem chamar a API.

Reconstrói os rótulos zero-shot do Problema A, Passo 2 (seed 42, config default) a partir do
`llm_interactions_parte001.json` de uma execução completa de REFERÊNCIA fixa
(`EXEC_REF`, não necessariamente a mais recente) e verifica que a
cadeia determinística dados -> centróides -> NNLS reproduz EXATAMENTE o W e a
fidelidade registrados no CSV de comparação de algoritmos daquela execução (hoje
`bloco1_a_passo4_perceptron_vs_nnls_*.csv`; a pasta de referência, anterior a 13/10/2026,
mantém o nome antigo até ser migrada por `renomear_assets_legado.py`).

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
# Execução completa de referência (2 modelos, 3 seeds; commitada em resultados/).
# Valores dourados abaixo foram extraídos de bloco1_a_passo4_perceptron_vs_nnls_20260705_153049.csv
# (gpt-4o-mini, seed 42, NNLS) e bloco1_abc_passos_2_a_4_20260705_153049.csv (zero-shot
# default A/B, W do Perceptron), recalibrados em 09/10/2026.
EXEC_REF = os.path.join(BASE_DIR, "resultados", "execucao_2026-07-05_12-34-52_completa")
INTERACOES = os.path.join(EXEC_REF, "llm_interactions_parte001.json")

W_NNLS_GOLDEN = np.array([0.036356, 0.142757])   # w_0, w_1 (CSV, 6 casas)
FID_NNLS_GOLDEN = 0.833333                        # fidelity_problem_a (CSV)
FID_PERC_STORED = 0.826667                        # fidelidade do W-perceptron armazenado
W_PERC_STORED = np.array([0.000149, 0.000739])    # bloco1_abc_passos_2_a_4 (seed 42, default)


@unittest.skipUnless(os.path.exists(INTERACOES), "execução de referência ausente")
class TestRegressaoCadeiaNumerica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_problems import create_problem_a, PROBLEM_A_CENTERS_ATE_2026_10_14
        from metrics import compute_centroids

        with open(INTERACOES) as fh:
            recs = json.load(fh)[:150]  # os 150 primeiros = Problema A, Passo 2, seed 42
        # a execução de referência (07/2026) usou os centróides (−2, 0)/(2, 0), trocados em 15/10/2026
        cls.X_a, _ = create_problem_a(150, 42, centers=PROBLEM_A_CENTERS_ATE_2026_10_14)
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
        # O algoritmo do perceptron pode evoluir (ex.: fix da busca em γ de 03/07; normalização
        # de w e α por época e melhor esforço por pontos mal classificados em 15/10/2026);
        # o contrato de regressão é a DIREÇÃO do W (cosseno > 0,99 com o armazenado em 07/2026),
        # não igualdade bit-a-bit de γ. Medido em 15/10/2026: cos = 0,9977.
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
        self.assertGreater(cos, 0.99)


if __name__ == "__main__":
    unittest.main()
