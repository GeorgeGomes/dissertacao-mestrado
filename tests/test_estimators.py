"""Testes das funções de conveniência de estimação de W.

`train_relaxed_perceptron` (em relaxed_perceptron.py) e `train_least_squares_inverse`
(em least_squares_inverse.py) apenas instanciam a classe do algoritmo e rodam o `fit`.
Estes testes verificam o CONTRATO do wrapper (shapes, w >= 0, forma do retorno,
histórico opcional) e uma sanidade de que reproduzem rótulos linearmente separáveis.
"""
import os
import sys
import unittest
import warnings

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

from relaxed_perceptron import RelaxedPerceptron, train_relaxed_perceptron  # noqa: E402
from least_squares_inverse import train_least_squares_inverse  # noqa: E402
from metrics import predict_with_metric  # noqa: E402

# Dados separáveis por centróide mais próximo (rótulos = vizinho mais próximo euclidiano).
CENTROIDS = np.array([[0.0, 0.0], [0.0, 4.0]])
X = np.array([[-1, 0], [1, 0], [-1, 4], [1, 4], [0, 0.5], [0, 3.5]], dtype=float)
Y = np.array([0, 0, 1, 1, 0, 1])


class TestTrainRelaxedPerceptron(unittest.TestCase):
    def test_contrato_e_projecao_nao_negativa(self):
        w, gamma = train_relaxed_perceptron(X, Y, CENTROIDS, verbose=False)
        self.assertEqual(np.asarray(w).shape, (2,))
        self.assertTrue((np.asarray(w) >= 0).all())   # projeção max(0, w)
        self.assertIsInstance(float(gamma), float)

    def test_return_history_devolve_tripla_com_lista(self):
        out = train_relaxed_perceptron(X, Y, CENTROIDS, verbose=False, return_history=True)
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[2], list)          # gamma_history

    def test_reproduz_rotulos_separaveis(self):
        w, _ = train_relaxed_perceptron(X, Y, CENTROIDS, verbose=False)
        np.testing.assert_array_equal(predict_with_metric(X, CENTROIDS, w), Y)

    def test_reprodutivel_independente_do_rng_global(self):
        # Regressão: o embaralhamento usa um RNG LOCAL semeado, então "sujar" o RNG
        # global do NumPy entre as duas chamadas NÃO pode mudar o W* aprendido.
        # (Antes da correção, chamadas consecutivas divergiam.)
        # Usa rótulos anisotrópicos separáveis para o perceptron aprender um W não-trivial.
        c = np.array([[-1.5, 1.0], [1.5, -1.0]])
        rng = np.random.RandomState(0)
        Xa = rng.randn(120, 2) * 0.6
        Xa[:60] += c[0]
        Xa[60:] += c[1]
        ya = predict_with_metric(Xa, c, np.array([0.3, 1.5]))

        np.random.seed(999)
        w1, _ = train_relaxed_perceptron(Xa, ya, c, verbose=False)
        for _ in range(37):        # avança o estado do RNG global de propósito
            np.random.rand()
        w2, _ = train_relaxed_perceptron(Xa, ya, c, verbose=False)
        np.testing.assert_array_equal(w1, w2)


class TestTrainLeastSquaresInverse(unittest.TestCase):
    def test_contrato_gamma_zero_e_projecao(self):
        w, gamma = train_least_squares_inverse(X, Y, CENTROIDS, verbose=False)
        self.assertEqual(np.asarray(w).shape, (2,))
        self.assertTrue((np.asarray(w) >= 0).all())   # NNLS: w >= 0 por construção
        self.assertEqual(gamma, 0.0)                  # sem margem explícita

    def test_reproduz_rotulos_separaveis(self):
        w, _ = train_least_squares_inverse(X, Y, CENTROIDS, verbose=False)
        np.testing.assert_array_equal(predict_with_metric(X, CENTROIDS, w), Y)

    def test_degenerado_emite_warning_com_w_nulo(self):
        # Centróides sobrepostos: nenhuma métrica diagonal separa — o NNLS devolve
        # W nulo, e isso NÃO pode passar em silêncio (warning obrigatório).
        c_iguais = np.array([[0.0, 0.0], [0.0, 0.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            w, _ = train_least_squares_inverse(X, Y, c_iguais, verbose=False)
        self.assertFalse(np.any(np.asarray(w) > 0))
        self.assertTrue(
            any("NNLS degenerado" in str(c.message) for c in caught),
            "solução degenerada do NNLS deve emitir RuntimeWarning",
        )

    def test_separavel_nao_emite_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_least_squares_inverse(X, Y, CENTROIDS, verbose=False)
        self.assertFalse(
            any("NNLS degenerado" in str(c.message) for c in caught),
            "caso separável não deve emitir warning de degenerescência",
        )


class TestBuscaBinariaGamma(unittest.TestCase):
    def test_teto_sem_convergir_emite_warning(self):
        # Regressão: sair por max_iterations sem fechar |γ_hi−γ_lo| ≤ tol era
        # silencioso — o γ retornado é a melhor margem viável testada, mas pode
        # ser artefato do orçamento (max_iterations × delta_gamma), não o máximo.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RelaxedPerceptron(max_iterations=1).fit(X, Y, CENTROIDS)
        self.assertTrue(
            any("max_iterations" in str(c.message) for c in caught),
            "teto atingido sem convergência deve emitir warning",
        )

    def test_convergencia_nao_emite_warning_de_teto(self):
        # Com delta_gamma na escala do γ* deste dataset (~12, em unidades de
        # distância ao quadrado), a busca forma o bracket e converge em ~30 iters.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = RelaxedPerceptron(delta_gamma=1.0)
            m.fit(X, Y, CENTROIDS)
        self.assertFalse(
            any("max_iterations" in str(c.message) for c in caught),
            "busca convergida não deve emitir warning de teto",
        )

    def test_bracket_gamma_hi_nunca_afrouxa(self):
        # Regressão: o candidato γ_lo + delta podia ultrapassar um γ_hi já provado
        # inviável e, ao falhar, AUMENTAR o teto (bracket não-monótono). O teto
        # γ_hi deve ser não-crescente ao longo da busca.
        m = RelaxedPerceptron(delta_gamma=1.0)
        m.fit(X, Y, CENTROIDS)
        his = [h["gamma_hi"] for h in m.gamma_history if h["gamma_hi"] is not None]
        self.assertTrue(len(his) > 0, "a busca deve encontrar um γ inviável neste dataset")
        self.assertTrue(
            all(b <= a for a, b in zip(his, his[1:])),
            f"γ_hi deve ser não-crescente, obtido: {his}",
        )


class TestValidacaoHiperparametros(unittest.TestCase):
    def test_delta_gamma_nao_positivo_levanta(self):
        # δ ≤ 0 faria a busca em γ nunca progredir — deve falhar na construção.
        for delta in (0.0, -0.05):
            with self.assertRaises(ValueError):
                RelaxedPerceptron(delta_gamma=delta)

    def test_delta_gamma_positivo_constroi(self):
        RelaxedPerceptron(delta_gamma=0.05)  # não deve levantar


if __name__ == "__main__":
    unittest.main()
