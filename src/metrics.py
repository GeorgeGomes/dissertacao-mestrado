"""Funções numéricas centrais: distância métrica, centróides, augmentação de
features, classificação por vizinho mais próximo e métricas de concordância.

Extraído de ``dissertacao_mestrado.py`` para isolar a geometria/métrica do runner e
permitir testes de unidade independentes (ver ``tests/test_metrics.py``). São funções
puras — dependem apenas de numpy e scikit-learn — sem efeitos colaterais, então podem
ser importadas sem disparar a lógica de execução do experimento.

Este módulo NÃO depende de ``dissertacao_mestrado`` (evita import circular): o runner
principal importa daqui, nunca o contrário.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, cohen_kappa_score,
    f1_score, precision_score, recall_score
)


# =============================================================================
# DISTÂNCIA MÉTRICA E GEOMETRIA
# =============================================================================

def d_W(x: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
    """Distância de Mahalanobis com matriz diagonal W.

    Justificativa: a restrição diagonal reduz a complexidade de O(d²) para O(d),
    tornando o aprendizado tratável sem perda crítica de poder discriminativo em 2D.

    Nota: retorna a distância AO QUADRADO (sem a raiz) — suficiente e mais barato
    para classificação por argmin, que é monotônica na raiz.
    """
    return np.sum(w * (x - c)**2)


def compute_centroids(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula os centróides de cada classe a partir dos rótulos do LLM.

    Os centróides são usados como âncoras para a métrica de Mahalanobis:
    um ponto é classificado na classe cujo centróide está mais próximo sob d_W.
    """
    classes = np.unique(y)
    centroids = np.array([X[y == c].mean(axis=0) for c in classes])
    return centroids


def augment_to_r3(X: np.ndarray) -> np.ndarray:
    """Adiciona feature de interação x3 = x1 * x2, projetando dados de R2 para R3.

    Usada no experimento de não-linearidade implícita: o LLM classifica apenas
    com (x1, x2), mas nos bastidores adicionamos x3 = x1·x2 e tentamos aprender
    uma métrica diagonal com 3 pesos. Se a fidelidade com 3 pesos superar a de
    2 pesos, há evidência de que o LLM adota implicitamente um critério não-linear.
    O vetor W terá 3 componentes; projetado de volta para R2, a fronteira de
    decisão corresponde a uma hipérbole.
    """
    x3 = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, x3])


def augment_to_r4(X: np.ndarray) -> np.ndarray:
    """Adiciona features quadráticas x1², x2² para projetar dados em R4.

    Justificativa (e-mail orientador 19:15): para o estudo de caso peso×altura,
    o classificador ótimo bayesiano é
        f(x1, x2) = x2² - x2 + x1² - x1 + cte
    — fronteira elíptica. Com 4 features (x1, x2, x1², x2²), a métrica diagonal
    consegue representar exatamente essa fronteira como combinação linear,
    enquanto com 3 features (x1·x2) só representa hipérbole.
    """
    x_sq = X ** 2
    return np.hstack([X, x_sq])


def augment_features(X: np.ndarray, n_features: int) -> np.ndarray:
    """Aumenta X para o número de features pedido (2, 3 ou 4).

    - 2: (x1, x2) — original
    - 3: (x1, x2, x1·x2) — hipérbole (augment_to_r3)
    - 4: (x1, x2, x1², x2²) — elipse (augment_to_r4)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < 2:
        raise ValueError(f"X precisa ter ao menos 2 colunas; tem {X.shape[1]}")

    X2 = X[:, :2]
    if n_features == 2:
        return X2
    if n_features == 3:
        return augment_to_r3(X2)
    if n_features == 4:
        return augment_to_r4(X2)
    raise ValueError(f"n_features deve ser 2, 3 ou 4 — recebido {n_features}")


def predict_with_metric(X: np.ndarray, centroids: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Prediz a classe de cada ponto usando a métrica estimada (vizinho mais próximo sob d_W)."""
    predictions = []
    for xi in X:
        distances = [d_W(xi, c, w) for c in centroids]
        predictions.append(np.argmin(distances))
    return np.array(predictions)


def compute_metric_confidence(
    X: np.ndarray,
    centroids: np.ndarray,
    w: np.ndarray,
    y_pred: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula pontuações de confiança baseadas em margem usando a fórmula do orientador.

    Definição de margem (orientador): margem = d_W(x, centróide_errado) - d_W(x, centróide_previsto)

    No caso de 2 classes (centróides[0] e centróides[1]):
        - Se classe predita == 0: d_pred = d0, d_errado = d1, logo margem = d1 - d0
        - Se classe predita == 1: d_pred = d1, d_errado = d0, logo margem = d0 - d1
        - Isso sempre equivale a |d1 - d0|, confirmando equivalência com a diferença absoluta de distâncias.

    margem >= 0 é garantida por construção: a classe predita sempre tem a menor
    (ou igual) distância ao seu centróide, logo d_errado >= d_pred por definição.
    """
    n_samples = X.shape[0]
    confidences = np.zeros(n_samples)
    predictions = np.zeros(n_samples, dtype=int)

    for i, xi in enumerate(X):
        d0 = d_W(xi, centroids[0], w)
        d1 = d_W(xi, centroids[1], w)

        if d0 <= d1:
            predictions[i] = 0
            d_pred = d0
            d_wrong = d1
        else:
            predictions[i] = 1
            d_pred = d1
            d_wrong = d0
        # Fórmula do orientador: margem = d_W(x, centróide_errado) - d_W(x, centróide_previsto)
        # No caso de 2 classes, isso sempre equivale a |d1 - d0|: uma distância é d_pred e a outra d_wrong.
        # margem >= 0 é garantida: a classe predita sempre tem a menor (ou igual) distância.
        margin = d_wrong - d_pred
        assert margin >= 0, "Margem deve ser não-negativa por construção"
        confidences[i] = margin

    if y_pred is not None:
        assert np.array_equal(predictions, y_pred), "Discordância de predição no cálculo de confiança"

    return confidences, predictions


# =============================================================================
# MÉTRICAS DE CONCORDÂNCIA (LLM × métrica estimada)
# =============================================================================

@dataclass
class ConsistencyMetrics:
    """Armazena métricas detalhadas de consistência entre predições do LLM e da métrica estimada."""
    accuracy: float
    cohen_kappa: float
    f1_score: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray
    n_agreements: int
    n_disagreements: int
    disagreement_indices: np.ndarray

    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.1%} | "
            f"Kappa: {self.cohen_kappa:.3f} | "
            f"F1: {self.f1_score:.3f}"
        )


def compute_consistency_metrics(
    y_llm: np.ndarray,
    y_metric: np.ndarray
) -> ConsistencyMetrics:
    """Calcula métricas detalhadas de consistência entre predições do LLM e da métrica estimada.

    Nota sobre concordância simétrica:
    - accuracy e kappa são métricas SIMÉTRICAS — o resultado é idêntico independentemente
      de qual vetor é tratado como "referência". Kappa é a métrica principal para H1.
    - F1, precision e recall NÃO são simétricas por construção; para preservar a simetria,
      calcula-se a média das duas direções (LLM→métrica e métrica→LLM), evitando que uma
      das partes seja arbitrariamente elevada a "ground truth".
    - confusion_matrix usa y_metric como referência (linhas = rótulos da métrica).
    """
    accuracy = accuracy_score(y_metric, y_llm)
    kappa = cohen_kappa_score(y_metric, y_llm)

    # F1/precision/recall simétricos: média das duas direções possíveis
    f1 = (
        f1_score(y_metric, y_llm, average='weighted', zero_division=0) +
        f1_score(y_llm, y_metric, average='weighted', zero_division=0)
    ) / 2
    precision = (
        precision_score(y_metric, y_llm, average='weighted', zero_division=0) +
        precision_score(y_llm, y_metric, average='weighted', zero_division=0)
    ) / 2
    recall = (
        recall_score(y_metric, y_llm, average='weighted', zero_division=0) +
        recall_score(y_llm, y_metric, average='weighted', zero_division=0)
    ) / 2
    cm = confusion_matrix(y_metric, y_llm)

    disagreements = y_llm != y_metric
    n_disagreements = np.sum(disagreements)
    n_agreements = len(y_llm) - n_disagreements
    disagreement_indices = np.where(disagreements)[0]

    return ConsistencyMetrics(
        accuracy=accuracy,
        cohen_kappa=kappa,
        f1_score=f1,
        precision=precision,
        recall=recall,
        confusion_matrix=cm,
        n_agreements=n_agreements,
        n_disagreements=n_disagreements,
        disagreement_indices=disagreement_indices
    )
