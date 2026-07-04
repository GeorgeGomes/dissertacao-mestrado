"""Testes da geração de dados (`data_problems`).

Verifica shape/rótulos dos geradores sintéticos, reprodutibilidade por semente,
a anisotropia do gerador da Oracle Validation e a carga da base real peso×altura.
"""
import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import data_problems as dp  # noqa: E402

CSV_REAL = os.path.join(
    os.path.dirname(__file__), "..", "dados_reais", "homem_mulher", "peso_altura.csv"
)


class TestGeradoresSinteticos(unittest.TestCase):
    def test_shapes_e_rotulos_binarios(self):
        casos = [
            (dp.create_problem_a, 150),
            (dp.create_problem_b, 100),
            (dp.create_problem_c, 100),
            (dp.create_problem_e_expert, 150),
            (dp.create_problem_d_meialua, 150),
        ]
        for gerador, n in casos:
            X, y = gerador(n_samples=n, random_state=7)
            self.assertEqual(X.shape, (n, 2), gerador.__name__)
            self.assertEqual(set(np.unique(y).tolist()), {0, 1}, gerador.__name__)

    def test_reprodutibilidade_mesma_semente(self):
        X1, y1 = dp.create_problem_a(n_samples=120, random_state=42)
        X2, y2 = dp.create_problem_a(n_samples=120, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_sementes_diferentes_geram_dados_diferentes(self):
        X1, _ = dp.create_problem_a(n_samples=120, random_state=42)
        X2, _ = dp.create_problem_a(n_samples=120, random_state=123)
        self.assertFalse(np.array_equal(X1, X2))


class TestAnisotropico(unittest.TestCase):
    def test_variancia_por_dimensao_reflete_std_per_dim(self):
        centers = [(-1.5, 1.0), (1.5, -1.0)]
        # x1 bem apertado (0.1), x2 bem espalhado (2.0)
        X, y = dp.create_anisotropic_problem(
            n_samples=400, centers=centers, std_per_dim=[0.1, 2.0], random_state=0
        )
        self.assertEqual(X.shape, (400, 2))
        for c in (0, 1):
            std_x1 = X[y == c, 0].std()
            std_x2 = X[y == c, 1].std()
            self.assertLess(std_x1, std_x2)          # anisotropia na direção certa
        # Centróides aproximadamente respeitados (tolerância ampla por causa da variância).
        np.testing.assert_allclose(X[y == 0].mean(axis=0), centers[0], atol=0.6)
        np.testing.assert_allclose(X[y == 1].mean(axis=0), centers[1], atol=0.6)


class TestConstantesGeometria(unittest.TestCase):
    def test_centros_tem_duas_classes_2d(self):
        for centros in (dp.PROBLEM_A_CENTERS, dp.PROBLEM_B_CENTERS,
                        dp.PROBLEM_C_CENTERS, dp.PROBLEM_E_CENTERS):
            self.assertEqual(len(centros), 2)
            self.assertTrue(all(len(c) == 2 for c in centros))

    def test_desvios_positivos(self):
        for std in (dp.PROBLEM_A_STD, dp.PROBLEM_B_STD, dp.PROBLEM_C_STD, dp.PROBLEM_E_STD):
            self.assertGreater(std, 0)


class TestBaseReal(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(CSV_REAL), "CSV peso×altura ausente")
    def test_carrega_peso_altura(self):
        X, y = dp.create_problem_homem_mulher(csv_path=CSV_REAL)
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.shape[1], 2)          # (peso, altura)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(set(np.unique(y).tolist()), {0, 1})
        self.assertEqual(X.dtype, float)


if __name__ == "__main__":
    unittest.main()
