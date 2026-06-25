"""Baselines clássicos para comparação com o LLM na Fase D.

Treina classificadores supervisionados nos mesmos exemplos few-shot fornecidos ao LLM
e avalia no mesmo conjunto de teste. Responde à pergunta: "O LLM faz algo que um
classificador trivial não faria com os mesmos dados?"

Classificadores:
    - KNNBaseline: k-Nearest Neighbors (k ajustado ao n_shot)
    - LogisticRegressionBaseline: Regressão Logística regularizada
    - SVMBaseline: SVM com kernel RBF
"""

import numpy as np
from typing import Dict, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


class KNNBaseline:
    """k-Nearest Neighbors com k ajustado ao número de exemplos disponíveis."""

    def __init__(self, n_neighbors: Optional[int] = None, verbose: bool = False):
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.clf_ = None
        self.k_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNBaseline":
        """Treina o k-NN. Ajusta k automaticamente se não especificado."""
        if self.n_neighbors is None:
            k = min(5, len(X_train))
            if k % 2 == 0:
                k = max(1, k - 1)
        else:
            k = min(self.n_neighbors, len(X_train))

        self.k_ = k
        self.clf_ = KNeighborsClassifier(n_neighbors=k)
        self.clf_.fit(X_train, y_train)

        if self.verbose:
            print(f"    [k-NN] k={k}, treino={len(X_train)} pontos")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.clf_.predict(X_test)

    @property
    def name(self) -> str:
        return f"k-NN (k={self.k_})"


class LogisticRegressionBaseline:
    """Regressão Logística regularizada (L2)."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000, verbose: bool = False):
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.clf_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
        self.clf_ = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.clf_.fit(X_train, y_train)

        if self.verbose:
            print(f"    [Logistic Regression] treino={len(X_train)} pontos, "
                  f"coefs={self.clf_.coef_[0]}")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.clf_.predict(X_test)

    @property
    def name(self) -> str:
        return "Logistic Regression"


class SVMBaseline:
    """SVM com kernel RBF."""

    def __init__(self, kernel: str = 'rbf', random_state: int = 42, verbose: bool = False):
        self.kernel = kernel
        self.random_state = random_state
        self.verbose = verbose
        self.clf_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVMBaseline":
        self.clf_ = SVC(kernel=self.kernel, random_state=self.random_state)
        self.clf_.fit(X_train, y_train)

        if self.verbose:
            print(f"    [SVM ({self.kernel.upper()})] treino={len(X_train)} pontos, "
                  f"support_vectors={self.clf_.n_support_}")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.clf_.predict(X_test)

    @property
    def name(self) -> str:
        return f"SVM ({self.kernel.upper()})"


class ClassicalBaselineRunner:
    """Orquestra a execução de todos os baselines clássicos e coleta métricas."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test_expert: np.ndarray,
        y_test_gt: np.ndarray,
        n_shot: int,
    ) -> Dict[str, Dict[str, float]]:
        """Treina todos os baselines e retorna métricas comparativas.

        Args:
            X_train: Features dos exemplos few-shot (n_shot, 2)
            y_train: Rótulos binários (0/1) dos exemplos (do perito)
            X_test: Features do conjunto de teste
            y_test_expert: Rótulos do perito para o teste
            y_test_gt: Rótulos ground truth para o teste
            n_shot: Número de exemplos

        Returns:
            Dict[nome_classificador, Dict[métrica, valor]]
        """
        if len(X_train) < 2 or len(np.unique(y_train)) < 2:
            return {}

        baselines = [
            KNNBaseline(verbose=self.verbose),
            LogisticRegressionBaseline(verbose=self.verbose),
            SVMBaseline(verbose=self.verbose),
        ]

        results = {}
        for baseline in baselines:
            try:
                baseline.fit(X_train, y_train)
                y_pred = baseline.predict(X_test)

                results[baseline.name] = {
                    'accuracy_vs_expert': accuracy_score(y_test_expert, y_pred),
                    'kappa_vs_expert': cohen_kappa_score(y_test_expert, y_pred),
                    'f1_vs_expert': f1_score(y_test_expert, y_pred, average='weighted', zero_division=0),
                    'accuracy_vs_gt': accuracy_score(y_test_gt, y_pred),
                }

                if self.verbose:
                    m = results[baseline.name]
                    print(f"    {baseline.name}: acc_expert={m['accuracy_vs_expert']:.1%}, "
                          f"kappa={m['kappa_vs_expert']:.3f}, f1={m['f1_vs_expert']:.3f}, "
                          f"acc_gt={m['accuracy_vs_gt']:.1%}")

            except Exception as e:
                if self.verbose:
                    print(f"    {baseline.name}: ERRO — {e}")

        return results
