"""Perceptron Estruturado, mudanças de 15/10/2026 (B, C, D).

B: w e α são divididos por ||w|| ao fim de cada época (a restrição é homogênea em w; sem isso o w
   encolhia até ~10⁻³ e qualquer α > 0 satisfazia a margem relaxada, "γ viável" degenerado).
C: melhor esforço = época com MENOS pontos mal classificados (margem bruta < 0), não menos violações
   relaxadas.
D: a margem bruta mínima/||w|| da solução devolvida fica em `margem_minima_` e em
   `ULTIMA_MARGEM_MINIMA`; o runner grava a coluna `margem_minima`; as tabelas marcam † se < 0.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import relaxed_perceptron as rp  # noqa: E402
from relaxed_perceptron import train_relaxed_perceptron  # noqa: E402
from least_squares_inverse import train_least_squares_inverse  # noqa: E402
from metrics import compute_centroids, predict_with_metric  # noqa: E402
from protocolo import PERCEPTRON_PARAMS  # noqa: E402


def _fid(X, y, w):
    return float(np.mean(predict_with_metric(X, compute_centroids(X, y), w) == y))


def _nao_separavel(seed=42, n=150):
    """Duas gaussianas sobrepostas rotuladas por um critério que depende SÓ de x2 e com ruído:
    reproduz o padrão que fazia o Perceptron antigo devolver ||W|| ≈ 10⁻³."""
    rs = np.random.RandomState(seed)
    X = np.vstack([rs.normal([-2, -0.5], 1.2, (n // 2, 2)), rs.normal([2, 0.5], 1.2, (n - n // 2, 2))])
    y = (X[:, 1] + 0.3 * rs.normal(size=n) > 0.4).astype(int)
    return X, y


class TestNormalizacaoPorEpoca(unittest.TestCase):
    def test_w_devolvido_tem_norma_unitaria_e_nao_degenera(self):
        X, y = _nao_separavel()
        w, gamma = train_relaxed_perceptron(X, y, compute_centroids(X, y), **PERCEPTRON_PARAMS,
                                            verbose=False, use_best_effort=True)
        self.assertAlmostEqual(float(np.linalg.norm(w)), 1.0, places=6)
        self.assertGreater(w.max(), 0.1, "antes da correção o w evaporava (||W|| ~ 1e-3)")

    def test_fidelidade_nao_abaixo_do_nnls_nem_da_euclidiana(self):
        X, y = _nao_separavel()
        c = compute_centroids(X, y)
        w, _ = train_relaxed_perceptron(X, y, c, **PERCEPTRON_PARAMS, verbose=False, use_best_effort=True)
        w_n = train_least_squares_inverse(X, y, c)
        w_n = np.asarray(w_n[0] if isinstance(w_n, tuple) else w_n).ravel()
        self.assertGreaterEqual(_fid(X, y, w) + 0.02, _fid(X, y, w_n))
        self.assertGreaterEqual(_fid(X, y, w), _fid(X, y, np.array([1.0, 1.0])))


class TestSeparavelContinuaIgual(unittest.TestCase):
    def test_oraculo_separavel_recupera_direcao_com_gamma_positivo(self):
        rs = np.random.RandomState(7)
        X = np.vstack([rs.normal([-1.5, 1.0], 0.8, (75, 2)), rs.normal([1.5, -1.0], 0.8, (75, 2))])
        w_true = np.array([0.3, 1.5])
        c = np.array([[-1.5, 1.0], [1.5, -1.0]])
        y = predict_with_metric(X, c, w_true)
        w, gamma = train_relaxed_perceptron(X, y, compute_centroids(X, y), **PERCEPTRON_PARAMS,
                                            verbose=False, use_best_effort=True)
        self.assertGreater(gamma, 0.0)
        self.assertEqual(_fid(X, y, w), 1.0)
        cos = float(w @ w_true / (np.linalg.norm(w) * np.linalg.norm(w_true)))
        self.assertGreater(cos, 0.98)
        self.assertGreaterEqual(rp.ULTIMA_MARGEM_MINIMA, 0.0)


class TestMargemMinima(unittest.TestCase):
    def test_margem_minima_negativa_quando_nao_separa(self):
        X, y = _nao_separavel()
        w, gamma = train_relaxed_perceptron(X, y, compute_centroids(X, y), **PERCEPTRON_PARAMS,
                                            verbose=False, use_best_effort=True)
        self.assertLess(rp.ULTIMA_MARGEM_MINIMA, 0.0)
        self.assertLess(_fid(X, y, w), 1.0)

    def test_margem_minima_coerente_com_a_fidelidade(self):
        X, y = _nao_separavel(seed=123)
        w, _ = train_relaxed_perceptron(X, y, compute_centroids(X, y), **PERCEPTRON_PARAMS,
                                        verbose=False, use_best_effort=True)
        self.assertEqual(_fid(X, y, w) == 1.0, rp.ULTIMA_MARGEM_MINIMA >= 0.0)


class TestTabelasMarcamMargemNegativa(unittest.TestCase):
    def test_marca_gamma_usa_margem_minima(self):
        import gerar_tabelas_bloco1 as b1
        self.assertEqual(b1.marca_gamma({"gamma_min": 0.05, "margem_minima_min": -0.2}), "†")
        self.assertEqual(b1.marca_gamma({"gamma_min": 0.05, "margem_minima_min": 0.1}), "")
        self.assertEqual(b1.marca_gamma({"gamma_min": 0.0, "margem_minima_min": float("nan")}), "†")
        self.assertIn("margem bruta", b1.NOTA_GAMMA)


if __name__ == "__main__":
    unittest.main()
