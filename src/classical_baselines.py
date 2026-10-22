"""Baselines clássicos para comparação com o LLM no Problema E, Passo 6 (Bloco 2).

Treina classificadores supervisionados nos mesmos exemplos few-shot fornecidos ao LLM
e avalia no mesmo conjunto de teste. Responde à pergunta: "O LLM faz algo que um
classificador trivial não faria com os mesmos dados?"

Classificadores:
    - KNNBaseline: k-Nearest Neighbors (k ajustado ao k_exemplos)
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
    """SVM com kernel RBF.

    Nota: em ``sklearn.svm.SVC`` o ``random_state`` só afeta a estimativa de probabilidades
    (``probability=True``, não usada aqui); com ``probability=False`` o ajuste é
    determinístico e o parâmetro é mantido só por uniformidade com os demais baselines.
    """

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
    ) -> Dict[str, Dict[str, float]]:
        """Treina todos os baselines e retorna métricas comparativas.

        Métricas (seção IV.3 da metodologia do orientador), com a REFERÊNCIA em
        ``y_test_expert``: acurácia (fração de concordância), Cohen's kappa e
        F1-Macro (``average='macro'``, média das F1 por classe), mais a acurácia
        contra o rótulo real (``y_test_gt``).

        Sobre os nomes das chaves: a referência é o especialista do Problema E (Bloco 2),
        daí ``*_vs_expert``. Este runner é usado SÓ no Bloco 2 (baselines clássicos
        do Passo 6 do Problema E); o reuso nos Problemas B e C foi removido.

        Args:
            X_train: Atributos dos exemplos few-shot (k_exemplos, 2)
            y_train: Rótulos binários (0/1) dos exemplos (da referência)
            X_test: Atributos do conjunto de teste
            y_test_expert: Rótulos da referência (especialista do Problema E)
            y_test_gt: Rótulos reais para o teste

        Returns:
            Dict[nome_classificador, Dict[métrica, valor]] com as chaves
            ``accuracy_vs_expert``, ``kappa_vs_expert``, ``f1_vs_expert``,
            ``accuracy_vs_gt``.
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
                    'f1_vs_expert': f1_score(y_test_expert, y_pred, average='macro', zero_division=0),
                    'accuracy_vs_gt': accuracy_score(y_test_gt, y_pred),
                }

                if self.verbose:
                    m = results[baseline.name]
                    print(f"    {baseline.name}: acc_expert={m['accuracy_vs_expert']:.1%}, "
                          f"kappa={m['kappa_vs_expert']:.3f}, f1_macro={m['f1_vs_expert']:.3f}, "
                          f"acc_gt={m['accuracy_vs_gt']:.1%}")

            except Exception as e:
                if self.verbose:
                    print(f"    {baseline.name}: ERRO — {e}")

        return results


def d_passo6_svm_medidas(X, y_true, y_llm, metric: Optional[dict] = None) -> Dict[str, float]:
    """Números da figura `bloco1_graphic_14` (Problema D, Passo 6), 16/10/2026.

    Dois SVM RBF (``gamma='scale'``, ``C=1``, os mesmos da figura), ajustados e avaliados nos MESMOS
    150 pontos: (a) treinado no rótulo real, acurácia vs rótulo real (a régua não linear de referência);
    (b) treinado nos rótulos zero-shot do LLM, fidelidade (vs LLM) e acurácia vs rótulo real. Se o LLM
    colapsou (regra `colapsou`), o SVM (b) não é ajustável e as medidas ficam NaN. ``metric`` (dict do
    Passo 4 com `n_features`, `fidelity_perc_vs_llm`, `accuracy_perc_vs_true`) entra só para a tabela.
    """
    from protocolo import colapsou
    X = np.asarray(X, dtype=float); y_true = np.asarray(y_true).astype(int); y_llm = np.asarray(y_llm).astype(int)
    nan = float("nan")
    out = {
        "n_pontos": int(len(y_true)),
        "n_classe_0_llm": int(np.sum(y_llm == 0)), "n_classe_1_llm": int(np.sum(y_llm == 1)),
        "collapsed": int(colapsou(y_llm)),
        "llm_accuracy_vs_true": float(np.mean(y_llm == y_true)),
        "svm_real_accuracy_vs_true": nan, "svm_real_n_errors": nan,
        "svm_llm_fidelity_vs_llm": nan, "svm_llm_accuracy_vs_true": nan, "svm_llm_n_errors_vs_true": nan,
        "metric_n_features": (int(metric["n_features"]) if metric else nan),
        "metric_fidelity_vs_llm": (float(metric["fidelity_perc_vs_llm"]) if metric else nan),
        "metric_accuracy_vs_true": (float(metric["accuracy_perc_vs_true"]) if metric else nan),
    }
    if len(np.unique(y_true)) == 2:
        pred = SVC(kernel="rbf", gamma="scale", C=1.0).fit(X, y_true).predict(X)
        out["svm_real_accuracy_vs_true"] = float(np.mean(pred == y_true))
        out["svm_real_n_errors"] = int(np.sum(pred != y_true))
    if not out["collapsed"]:
        pred = SVC(kernel="rbf", gamma="scale", C=1.0).fit(X, y_llm).predict(X)
        out["svm_llm_fidelity_vs_llm"] = float(np.mean(pred == y_llm))
        out["svm_llm_accuracy_vs_true"] = float(np.mean(pred == y_true))
        out["svm_llm_n_errors_vs_true"] = int(np.sum(pred != y_true))
    return out
