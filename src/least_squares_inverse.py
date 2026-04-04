"""Algoritmo de otimização inversa via Mínimos Quadrados (NNLS).

Encontra W que minimiza o erro quadrático nas margens observadas.
Resolve: min_w ||Aw - b||^2 sujeito a w >= 0.

Referência: abordagem inspirada em Ahuja & Orlin (otimização inversa clássica).
"""

import numpy as np
from typing import Tuple


class LeastSquaresInverse:
    """Aprende uma métrica de Mahalanobis diagonal via Non-Negative Least Squares."""

    def __init__(self, verbose: bool = False):
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
        """Treina via NNLS e retorna (w, gamma=0.0)."""
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

            # Encontra centróide errado mais próximo
            dists = [np.sum((xi - centroids[k]) ** 2) for k in range(n_classes)]
            dists[yi] = np.inf
            wrong_idx = np.argmin(dists)
            wrong = centroids[wrong_idx]

            # margem = sum(w * [(xi - wrong)^2 - (xi - correct)^2])
            row = (xi - wrong) ** 2 - (xi - correct) ** 2
            A.append(row)
            b.append(1.0)  # Queremos margem unitária

        A = np.array(A)
        b = np.array(b)

        # Non-negative least squares (garante w >= 0)
        w_learned, residual = nnls(A, b)

        # Evita pesos zero
        w_learned = np.maximum(w_learned, 1e-6)

        if self.verbose:
            print(f"    [Mínimos Quadrados] W encontrado: [{', '.join(f'{wi:.4f}' for wi in w_learned)}]")
            print(f"    Resíduo: {residual:.4f}")

        self.w_ = w_learned
        self.residual_ = residual
        return w_learned, 0.0  # gamma=0 (sem margem explícita)
