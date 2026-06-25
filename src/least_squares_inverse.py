"""Algoritmo de otimização inversa via Mínimos Quadrados (NNLS).

Encontra W que minimiza o erro quadrático nas margens observadas.
Resolve: min_w ||Aw - b||^2 sujeito a w >= 0.

A formulação é inspirada em Schultz & Joachims (2003), que propõem restrições de margem
para aprendizado de métrica: d_W(xi, ck) - d_W(xi, cl) ≥ 1, com minimização de ||w||.
Aqui relaxamos para mínimos quadrados (margem-alvo, não margem-mínima), usando NNLS
para garantir w ≥ 0 (condição de semidefinida positiva da matriz diagonal).

Referências:
    [1] Schultz, M. & Joachims, T. (2003).
        "Learning a distance metric from relative comparisons."
        → Formulação de margem unitária para metric learning (Eq. 10).
    [2] Ahuja, R.K. & Orlin, J.B. (2001).
        "Inverse Optimization." Operations Research, 49(5), pp. 771-783.
        → Framework teórico de otimização inversa: dado solução observada, inferir função objetivo.
    [3] Xing, E.P., Ng, A.Y., Jordan, M.I. & Russell, S. (2002).
        "Distance metric learning with application to clustering with side-information."
        → Formulação original de aprendizado de métrica de Mahalanobis com w ≥ 0.
"""

import numpy as np
from typing import Tuple


class LeastSquaresInverse:
    """Aprende uma métrica de Mahalanobis diagonal via Non-Negative Least Squares.

    A restrição w ≥ 0 é satisfeita nativamente pelo algoritmo NNLS (scipy),
    sem necessidade de projeção post-hoc. Isso garante que a matriz diagonal
    seja semidefinida positiva (Xing et al., 2002).
    """

    def __init__(self, target_margin: float = 1.0, verbose: bool = False):
        """
        Args:
            target_margin: Margem-alvo para as restrições de distância.
                O valor define a ESCALA de w, não a geometria da solução:
                multiplicar target_margin por k equivale a multiplicar w por k.
                Usa-se 1.0 por convenção (Schultz & Joachims, 2003, Eq. 10),
                pois a razão entre componentes de w (que determina a fronteira
                de decisão) é invariante à escala da margem.
            verbose: Se True, imprime detalhes do treinamento.
        """
        self.target_margin = target_margin
        self.verbose = verbose

        # Resultados após fit()
        self.w_ = None
        self.residual_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centroids: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Treina via NNLS e retorna (w, gamma=0.0).

        Constrói sistema linear: para cada ponto xi com classe yi,
            Σ_j w_j · [(xi_j - ck_j)² - (xi_j - cl_j)²] ≈ target_margin
        onde cl é o centróide correto e ck é o centróide rival mais próximo.

        A linha i da matriz A é:  (xi - ck)² - (xi - cl)²
        O vetor b tem todas entradas = target_margin.

        NNLS garante w ≥ 0 nativamente (sem clamping post-hoc).
        """
        from scipy.optimize import nnls

        m, d = X.shape
        n_classes = len(centroids)

        # Constrói sistema linear: para cada ponto, margem = d(x, wrong) - d(x, correct) > 0
        A = []
        b = []
        for i in range(m):
            xi = X[i]
            yi = int(y[i])
            correct = centroids[yi]

            # Encontra centróide errado mais próximo (distância Euclidiana não-ponderada)
            dists = [np.sum((xi - centroids[k]) ** 2) for k in range(n_classes)]
            dists[yi] = np.inf
            wrong_idx = np.argmin(dists)
            wrong = centroids[wrong_idx]

            # Linha da matriz A: coeficientes de w na expressão da margem
            # margem(xi) = d_W(xi, wrong) - d_W(xi, correct)
            #             = Σ_j w_j · [(xi_j - wrong_j)² - (xi_j - correct_j)²]
            row = (xi - wrong) ** 2 - (xi - correct) ** 2
            A.append(row)
            b.append(self.target_margin)

        A = np.array(A)
        b = np.array(b)

        # Non-negative least squares: min ||Aw - b||² s.t. w ≥ 0
        # NNLS (Lawson & Hanson, 1974) garante w ≥ 0 nativamente —
        # não é necessário clamping post-hoc, que violaria a solução ótima.
        w_learned, residual = nnls(A, b)

        if self.verbose:
            print(f"    [Mínimos Quadrados] W encontrado: [{', '.join(f'{wi:.4f}' for wi in w_learned)}]")
            print(f"    Resíduo: {residual:.4f}")
            n_zero = np.sum(w_learned == 0.0)
            if n_zero > 0:
                print(f"    ⚠ {n_zero} componente(s) de w = 0 (feature irrelevante para a métrica)")

        self.w_ = w_learned
        self.residual_ = residual
        return w_learned, 0.0  # gamma=0 (sem margem explícita garantida)
