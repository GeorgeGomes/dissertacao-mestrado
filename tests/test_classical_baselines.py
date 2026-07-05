"""Testes dos baselines clássicos (k-NN, Regressão Logística, SVM).

Esses classificadores respondem, na dissertação, à pergunta "o LLM faz algo que
um classificador trivial não faria?" — então precisam estar corretos por si.
Contratos verificados:

1. Dataset trivialmente separável → os 3 baselines acertam 100% (vs expert e
   vs GT) e o dicionário de métricas tem as chaves esperadas.
2. k do k-NN: automático = mín(5, n_treino) forçado a ÍMPAR (desempate binário).
3. Treino degenerado (classe única ou < 2 pontos) → ``run`` devolve {} sem
   levantar exceção (contrato usado pelo runner para pular a célula).

Execução: .venv/bin/python -m unittest tests.test_classical_baselines
"""
import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

from classical_baselines import (  # noqa: E402
    ClassicalBaselineRunner,
    KNNBaseline,
)

MEDIDAS = ("accuracy_vs_expert", "kappa_vs_expert", "f1_vs_expert", "accuracy_vs_gt")


def _dataset_separavel(n_por_classe=20, seed=0):
    """Duas nuvens bem afastadas: (-5,0) e (+5,0) — qualquer método acerta tudo."""
    rng = np.random.RandomState(seed)
    X0 = rng.normal(loc=[-5.0, 0.0], scale=0.3, size=(n_por_classe, 2))
    X1 = rng.normal(loc=[+5.0, 0.0], scale=0.3, size=(n_por_classe, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_por_classe + [1] * n_por_classe)
    return X, y


class TestClassicalBaselines(unittest.TestCase):
    def test_dataset_separavel_todos_acertam_tudo(self):
        X_train, y_train = _dataset_separavel(seed=0)
        X_test, y_test = _dataset_separavel(seed=1)
        results = ClassicalBaselineRunner().run(
            X_train, y_train, X_test,
            y_test_expert=y_test, y_test_gt=y_test, n_shot=len(X_train),
        )
        self.assertEqual(len(results), 3)  # k-NN, LR, SVM
        for nome, metricas in results.items():
            for chave in MEDIDAS:
                self.assertIn(chave, metricas, f"{nome} sem {chave}")
            self.assertEqual(metricas["accuracy_vs_expert"], 1.0, nome)
            self.assertEqual(metricas["accuracy_vs_gt"], 1.0, nome)
            self.assertEqual(metricas["kappa_vs_expert"], 1.0, nome)

    def test_knn_k_automatico_e_impar(self):
        X, y = _dataset_separavel(n_por_classe=10)
        knn = KNNBaseline().fit(X, y)
        self.assertEqual(knn.k_, 5)  # mín(5, 20) = 5, já ímpar

        # Com 4 pontos de treino: mín(5, 4) = 4 → par → forçado a 3.
        knn4 = KNNBaseline().fit(X[:4], np.array([0, 0, 1, 1]))
        self.assertEqual(knn4.k_, 3)
        self.assertEqual(knn4.name, "k-NN (k=3)")

    def test_treino_degenerado_devolve_vazio(self):
        X, y = _dataset_separavel()
        runner = ClassicalBaselineRunner()
        # Classe única no treino
        self.assertEqual(runner.run(X[:10], np.zeros(10, dtype=int), X, y, y, 10), {})
        # Menos de 2 pontos
        self.assertEqual(runner.run(X[:1], y[:1], X, y, y, 1), {})


if __name__ == "__main__":
    unittest.main()
