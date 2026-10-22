"""Testes da geração de dados (`data_problems`).

Verifica shape/rótulos dos geradores sintéticos, reprodutibilidade por semente,
e a carga da base real peso×altura.
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
CSV_REAL_KG_M = os.path.join(
    os.path.dirname(__file__), "..", "dados_reais", "homem_mulher", "peso_altura_kg_m.csv"
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

    @unittest.skipUnless(os.path.exists(CSV_REAL_KG_M), "CSV peso×altura em kg/m ausente")
    def test_kg_m_alinhado_com_a_base_normalizada(self):
        """G2: mesma base de G1 em unidades físicas, mesma ordem, mesmo rótulo, mapa linear."""
        X, y = dp.create_problem_homem_mulher(csv_path=CSV_REAL)
        X_fis, y_fis = dp.create_problem_homem_mulher_kg_m(csv_path=CSV_REAL_KG_M)
        self.assertEqual(X_fis.shape, X.shape)
        np.testing.assert_array_equal(y_fis, y)
        # Regra de três do orientador: -1 → 45 kg / 1,50 m e +1 → 110 kg / 1,95 m
        kg = 45.0 + (X[:, 0] + 1.0) * 65.0 / 2.0
        m = 1.50 + (X[:, 1] + 1.0) * 0.45 / 2.0
        np.testing.assert_allclose(X_fis[:, 0], kg, atol=0.051)   # 1 casa decimal
        np.testing.assert_allclose(X_fis[:, 1], m, atol=0.0051)   # 2 casas decimais
        # Classe 1 (Homem) é mais pesada e mais alta em média
        self.assertGreater(X_fis[y == 1, 0].mean(), X_fis[y == 0, 0].mean())
        self.assertGreater(X_fis[y == 1, 1].mean(), X_fis[y == 0, 1].mean())


if __name__ == "__main__":
    unittest.main()
