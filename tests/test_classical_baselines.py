"""Testes dos baselines clássicos (k-NN, Regressão Logística, SVM).

Esses classificadores respondem, na dissertação, à pergunta "o LLM faz algo que
um classificador trivial não faria?" — então precisam estar corretos por si.
Contratos verificados:

1. Dataset trivialmente separável → os 3 baselines acertam 100% (vs especialista e
   vs GT) e o dicionário de métricas tem as chaves esperadas.
2. k do k-NN: automático = mín(5, n_treino) forçado a ÍMPAR (desempate binário).
3. Treino degenerado (classe única ou < 2 pontos) → ``run`` devolve {} sem
   levantar exceção (contrato usado pelo runner para pular a célula).
4. ``f1_vs_expert`` é a F1-Macro (seção IV.3 do orientador), com valor conhecido.

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
            y_test_expert=y_test, y_test_gt=y_test,
        )
        self.assertEqual(len(results), 3)  # k-NN, LR, SVM
        for nome, metricas in results.items():
            for chave in MEDIDAS:
                self.assertIn(chave, metricas, f"{nome} sem {chave}")
            self.assertEqual(metricas["accuracy_vs_expert"], 1.0, nome)
            self.assertEqual(metricas["accuracy_vs_gt"], 1.0, nome)
            self.assertEqual(metricas["kappa_vs_expert"], 1.0, nome)

    def test_f1_vs_expert_e_macro(self):
        """``f1_vs_expert`` é a F1-Macro da seção IV.3 (média das F1 por classe), não
        a weighted nem a binária. Força o k-NN (k=1) a reproduzir uma predição conhecida
        e confere o valor contra o cálculo manual."""
        # Treino: 1 ponto por classe; teste: pontos cuja predição por k-NN (k=1) é conhecida.
        X_train = np.array([[-5.0, 0.0], [5.0, 0.0]])
        y_train = np.array([0, 1])
        # Predições pelo vizinho mais próximo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        X_test = np.array([[-5.0, 0.0]] * 9 + [[5.0, 0.0]])
        # Referência (especialista): [0]*8 + [1, 1] → classe 1 metade errada.
        y_expert = np.array([0] * 8 + [1, 1])
        results = ClassicalBaselineRunner().run(X_train, y_train, X_test, y_expert, y_expert)
        knn = next(m for nome, m in results.items() if nome.startswith("k-NN"))
        # Classe 0: VP=8, FP=1, FN=0 → F1 = 16/17. Classe 1: VP=1, FP=0, FN=1 → F1 = 2/3.
        f1_macro = (16 / 17 + 2 / 3) / 2
        self.assertAlmostEqual(knn["f1_vs_expert"], f1_macro, places=6)
        # Distingue de weighted ((8/10)·16/17 + (2/10)·2/3) e de binary (2/3).
        f1_weighted = 0.8 * 16 / 17 + 0.2 * 2 / 3
        self.assertNotAlmostEqual(knn["f1_vs_expert"], f1_weighted, places=3)
        self.assertNotAlmostEqual(knn["f1_vs_expert"], 2 / 3, places=3)
        self.assertAlmostEqual(knn["accuracy_vs_expert"], 0.9)

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
        self.assertEqual(runner.run(X[:10], np.zeros(10, dtype=int), X, y, y), {})
        # Menos de 2 pontos
        self.assertEqual(runner.run(X[:1], y[:1], X, y, y), {})


if __name__ == "__main__":
    unittest.main()
