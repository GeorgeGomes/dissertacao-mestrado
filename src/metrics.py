"""Funções numéricas centrais: distância métrica, centróides, augmentação de
atributos, classificação por centróide mais próximo e métricas de avaliação
(concordância entre dois vetores de rótulos: acurácia/fidelidade/consistência,
Cohen's kappa e F1-Macro, conforme a seção IV.3 da metodologia do orientador).

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
    accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
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

    Justificativa (orientador): para o estudo de caso peso×altura,
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
# MÉTRICAS DE AVALIAÇÃO (LLM × métrica estimada), seção IV.3 do orientador
# =============================================================================

@dataclass
class ConsistencyMetrics:
    """Métricas de avaliação entre dois vetores de rótulos (LLM e métrica estimada).

    Campos de métrica (seção IV.3 da metodologia):
        accuracy:    fração de concordância (1/n) Σ 1[y_metric_i = y_llm_i]. O chamador
                     dá o nome: Fidelidade (Problemas A e D), Consistência (B e C) ou
                     Acurácia (quando o segundo vetor é o ground truth ou o especialista).
        cohen_kappa: Cohen's kappa, k = (P0 − Pe)/(1 − Pe).
        f1_score:    F1-Macro (média das F1 por classe).

    Campos auxiliares (contagens, não são métricas de avaliação):
        confusion_matrix, n_agreements, n_disagreements, disagreement_indices.
    """
    accuracy: float
    cohen_kappa: float
    f1_score: float
    confusion_matrix: np.ndarray
    n_agreements: int
    n_disagreements: int
    disagreement_indices: np.ndarray


def compute_consistency_metrics(
    y_llm: np.ndarray,
    y_metric: np.ndarray
) -> ConsistencyMetrics:
    """Calcula as métricas de avaliação da seção IV.3 entre os rótulos do LLM e os
    rótulos da métrica estimada (ou de outra referência: ground truth, especialista).

    - ``accuracy``: concordância (1/n) Σ 1[y_metric_i = y_llm_i]. É a mesma fórmula
      que o PDF chama de Fidelidade (Problemas A e D: a métrica reproduz o LLM),
      Consistência (Problemas B e C: w transferido do A, centróides locais) ou
      Acurácia (referência = ground truth ou especialista); quem decide o nome é o
      chamador, conforme o que passou em ``y_metric``.
    - ``cohen_kappa``: k = (P0 − Pe)/(1 − Pe), com P0 = accuracy e
      Pe = Σ_k p_k p'_k, onde p_k é a marginal da métrica (``y_metric``) e p'_k a
      marginal do LLM (``y_llm``). Faixas de Landis e Koch (1977). Simétrico.
    - ``f1_score``: F1-Macro. Para cada classe k, P_k = VP/(VP+FP), R_k = VP/(VP+FN),
      F1_k = 2 P_k R_k/(P_k + R_k); o valor é a média das F1_k entre as classes.
      ``y_metric`` é tratado como referência e ``y_llm`` como predição. Como
      F1_k = 2VP/(2VP + FP + FN) é simétrica em FP e FN, trocar os papéis dos dois
      vetores não altera a F1-Macro: uma única chamada basta, sem simetrização.
      ``zero_division=0`` cobre classe ausente em um dos vetores.
    - ``confusion_matrix`` usa ``y_metric`` como referência (linhas = métrica).
    """
    y_llm = np.asarray(y_llm)
    y_metric = np.asarray(y_metric)

    accuracy = accuracy_score(y_metric, y_llm)
    kappa = cohen_kappa_score(y_metric, y_llm)
    f1 = f1_score(y_metric, y_llm, average='macro', zero_division=0)
    cm = confusion_matrix(y_metric, y_llm)

    disagreements = y_llm != y_metric
    n_disagreements = int(np.sum(disagreements))
    n_agreements = len(y_llm) - n_disagreements
    disagreement_indices = np.where(disagreements)[0]

    return ConsistencyMetrics(
        accuracy=accuracy,
        cohen_kappa=kappa,
        f1_score=f1,
        confusion_matrix=cm,
        n_agreements=n_agreements,
        n_disagreements=n_disagreements,
        disagreement_indices=disagreement_indices
    )
