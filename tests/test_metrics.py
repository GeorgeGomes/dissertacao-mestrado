"""Testes de unidade das funções numéricas centrais.

Cobre a distância métrica (`d_W`), os centróides (`compute_centroids`), a classificação
por vizinho mais próximo sob d_W (`predict_with_metric`), a augmentação de features
(`augment_features`) e as métricas de concordância (`compute_consistency_metrics`).
São funções puras — fáceis de testar e críticas para a corretude dos resultados.

Execução:
    cd src && ../.venv/bin/python -m pytest ../tests/test_metrics.py -q
"""
import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402


class TestDistanciaMetrica(unittest.TestCase):
    def test_d_W_e_distancia_quadrada_ponderada(self):
        x = np.array([1.0, 2.0])
        c = np.array([0.0, 0.0])
        w = np.array([1.0, 1.0])
        # Euclidiana ao quadrado: 1 + 4 = 5 (d_W retorna o quadrado, não a raiz).
        self.assertAlmostEqual(dm.d_W(x, c, w), 5.0)

    def test_d_W_ponderacao_anisotropica(self):
        x = np.array([1.0, 1.0])
        c = np.array([0.0, 0.0])
        # Peso 5x em x2 deve dominar a distância.
        self.assertAlmostEqual(dm.d_W(x, c, np.array([0.3, 1.5])), 0.3 + 1.5)

    def test_d_W_zero_no_proprio_centroide(self):
        x = np.array([2.0, -3.0])
        self.assertEqual(dm.d_W(x, x, np.array([1.0, 1.0])), 0.0)


class TestCentroides(unittest.TestCase):
    def test_compute_centroids_media_por_classe(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0],   # classe 0 → média (1, 0)
                      [0.0, 4.0], [0.0, 6.0]])   # classe 1 → média (0, 5)
        y = np.array([0, 0, 1, 1])
        c = dm.compute_centroids(X, y)
        np.testing.assert_allclose(c[0], [1.0, 0.0])
        np.testing.assert_allclose(c[1], [0.0, 5.0])


class TestPredicaoComMetrica(unittest.TestCase):
    def test_vizinho_mais_proximo_isotropico(self):
        centroids = np.array([[0.0, 0.0], [0.0, 4.0]])
        X = np.array([[0.0, 0.5], [0.0, 3.5]])
        w = np.array([1.0, 1.0])
        np.testing.assert_array_equal(dm.predict_with_metric(X, centroids, w), [0, 1])

    def test_peso_zero_ignora_eixo(self):
        # Com w=[1,0], a coordenada x2 é ignorada: ambos os pontos colapsam em x1.
        centroids = np.array([[0.0, 0.0], [3.0, 100.0]])
        X = np.array([[0.4, 99.0], [2.6, 1.0]])
        np.testing.assert_array_equal(
            dm.predict_with_metric(X, centroids, np.array([1.0, 0.0])), [0, 1])


class TestAugmentFeatures(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[2.0, 3.0], [-1.0, 4.0]])

    def test_n2_inalterado(self):
        np.testing.assert_array_equal(dm.augment_features(self.X, 2), self.X)

    def test_n3_adiciona_produto(self):
        out = dm.augment_features(self.X, 3)
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_allclose(out[:, 2], [2.0 * 3.0, -1.0 * 4.0])  # x1·x2

    def test_n4_adiciona_quadrados(self):
        out = dm.augment_features(self.X, 4)
        self.assertEqual(out.shape, (2, 4))
        np.testing.assert_allclose(out[:, 2], [4.0, 1.0])    # x1²
        np.testing.assert_allclose(out[:, 3], [9.0, 16.0])   # x2²

    def test_n_features_invalido_levanta(self):
        with self.assertRaises(ValueError):
            dm.augment_features(self.X, 5)


class TestConsistencyMetrics(unittest.TestCase):
    def test_concordancia_perfeita(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        m = dm.compute_consistency_metrics(y, y.copy())
        self.assertAlmostEqual(m.accuracy, 1.0)
        self.assertAlmostEqual(m.cohen_kappa, 1.0)
        self.assertAlmostEqual(m.f1_score, 1.0)
        self.assertEqual(m.n_disagreements, 0)

    def test_discordancia_total_balanceada_da_kappa_negativo(self):
        y_llm = np.array([0, 0, 1, 1])
        y_metric = np.array([1, 1, 0, 0])
        m = dm.compute_consistency_metrics(y_llm, y_metric)
        self.assertAlmostEqual(m.accuracy, 0.0)
        self.assertLess(m.cohen_kappa, 0.0)  # pior que o acaso

    def test_kappa_e_simetrico(self):
        a = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        b = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        m1 = dm.compute_consistency_metrics(a, b)
        m2 = dm.compute_consistency_metrics(b, a)
        self.assertAlmostEqual(m1.cohen_kappa, m2.cohen_kappa)
        self.assertAlmostEqual(m1.accuracy, m2.accuracy)

    def test_contagem_de_acordos(self):
        y_llm = np.array([0, 1, 0, 1])
        y_metric = np.array([0, 1, 1, 1])  # 1 discordância (índice 2)
        m = dm.compute_consistency_metrics(y_llm, y_metric)
        self.assertEqual(m.n_disagreements, 1)
        self.assertEqual(m.n_agreements, 3)
        np.testing.assert_array_equal(m.disagreement_indices, [2])


if __name__ == "__main__":
    unittest.main()
