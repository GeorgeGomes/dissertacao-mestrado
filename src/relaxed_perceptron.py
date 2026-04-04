"""Perceptron Estruturado com Relaxação de Margem para aprendizado de métrica de Mahalanobis diagonal.

Algoritmo: busca binária no gamma ótimo tal que todos os pontos satisfaçam
d_W(xi, centróide_errado) - d_W(xi, centróide_correto) >= gamma * ||w||.

Referência: Perceptron Estruturado com margem (indicado pelo orientador na Reunião 1).
"""

import numpy as np
import warnings
from typing import Tuple


class RelaxedPerceptron:
    """Aprende uma métrica de Mahalanobis diagonal via Perceptron Estruturado com busca binária em gamma."""

    def __init__(
        self,
        eta: float = 0.1,
        C: float = 1.0,
        gamma_init: float = 0.1,
        delta_gamma: float = 0.1,
        max_epochs: int = 100,
        tol: float = 1e-5,
        use_best_effort: bool = False,
        max_iterations: int = 50,
        verbose: bool = False,
    ):
        self.eta = eta
        self.C = C
        self.gamma_init = gamma_init
        self.delta_gamma = delta_gamma
        self.max_epochs = max_epochs
        self.tol = tol
        self.use_best_effort = use_best_effort
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Resultados após fit()
        self.w_ = None
        self.gamma_ = None

    @staticmethod
    def d_W(x: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
        """Distância de Mahalanobis com matriz diagonal W."""
        return np.sum(w * (x - c) ** 2)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centroids: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Treina o Perceptron Estruturado e retorna (w, gamma)."""
        m, d = X.shape
        lambda_ = 1.0 / self.C

        w = np.ones(d)
        alpha = np.zeros(m)
        gamma = self.gamma_init

        gamma_lo = 0.0
        gamma_hi = np.inf

        w_star = w.copy()
        alpha_star = alpha.copy()

        w_best_effort = w.copy()
        min_global_violations = float("inf")
        found_feasible_solution = False

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if self.verbose:
                print(f"    Testando gamma = {gamma:.6f}")

            final_violations = -1

            for epoch in range(self.max_epochs):
                violations = 0
                indices = np.random.permutation(m)

                for i in indices:
                    xi, yi = X[i], int(y[i])

                    distances = np.array([self.d_W(xi, c, w) for c in centroids])

                    l = yi
                    distances_temp = distances.copy()
                    distances_temp[l] = np.inf
                    k = np.argmin(distances_temp)

                    norm_w = np.linalg.norm(w)
                    if norm_w < 1e-9:
                        norm_w = 1e-9

                    margin = (
                        self.d_W(xi, centroids[k], w)
                        - self.d_W(xi, centroids[l], w)
                        + lambda_ * alpha[i]
                    )

                    if margin < gamma * norm_w:
                        violations += 1

                        g = (xi - centroids[l]) ** 2 - (xi - centroids[k]) ** 2

                        factor = 1 - self.eta * gamma / norm_w
                        factor = np.clip(factor, 0.0, 2.0)

                        w = w * factor - self.eta * g
                        w = np.maximum(0, w)

                        alpha = alpha * factor
                        alpha[i] = alpha[i] + self.eta
                        alpha = np.clip(alpha, 0, 1e6)

                final_violations = violations

                if violations < min_global_violations:
                    min_global_violations = violations
                    w_best_effort = w.copy()

                if violations == 0:
                    if self.verbose:
                        print(f"      Convergiu na época {epoch + 1}")
                    break

            if final_violations == 0:
                found_feasible_solution = True
                gamma_lo = gamma
                w_star = w.copy()
                alpha_star = alpha.copy()
                gamma = gamma + self.delta_gamma
            else:
                gamma_hi = gamma

                if not np.isinf(gamma_hi):
                    gamma = (gamma_lo + gamma_hi) / 2
                    w = w_star.copy()
                    alpha = alpha_star.copy()
                else:
                    break

            if abs(gamma_hi - gamma_lo) <= self.tol:
                break

        if found_feasible_solution:
            self.w_ = w_star
            self.gamma_ = gamma_lo
            return w_star, gamma_lo

        if self.use_best_effort:
            msg = f"Perceptron: separação perfeita não encontrada. Retornando melhor esforço (violações: {min_global_violations}). Métrica pode ser imprecisa."
            if self.verbose:
                print(f"    ⚠️ {msg}")
            else:
                warnings.warn(f"\n  ⚠️ {msg}")
            self.w_ = w_best_effort
            self.gamma_ = 0.0
            return w_best_effort, 0.0
        else:
            msg = f"Perceptron: separação perfeita não encontrada. Retornando pesos iniciais (equivalente à distância Euclidiana). Verifique a separabilidade dos dados do LLM."
            if self.verbose:
                print(f"    ❌ {msg}")
            else:
                warnings.warn(f"\n  ❌ {msg}")
            self.w_ = w_star
            self.gamma_ = 0.0
            return w_star, 0.0
