"""Perceptron Estruturado com Relaxação de Margem para aprendizado de métrica de Mahalanobis diagonal.

Formulação do problema (Coelho, Borges & Fonseca Neto, CILAMCE 2017, Eq. 28):

    Max γ
    Sujeito a: d_W(xi, ck) - d_W(xi, cl) + λ·αi ≥ γ·||w||,  i = 1, ..., m
               w ≥ 0

Onde:
    - d_W(xi, c) = Σ_j w_j·(x_ij - c_j)²   é a distância de Mahalanobis com matriz diagonal W
      (caso particular da Eq. 1 de Xing et al., 2002, restrito à matriz diagonal;
       equivalente à distância Euclidiana normalizada, cf. Seção 3.1 do CILAMCE 2017)
    - cl é o centróide correto do ponto xi (classe verdadeira yi)
    - ck é o centróide rival mais próximo: k = argmin_{j≠yi} d_W(xi, cj)
    - γ é a margem geométrica a ser maximizada (análogo ao SVM, cf. Schultz & Joachims, 2003, Eq. 10)
    - α é o vetor de variáveis duais associado às restrições, com penalização λ = 1/C
      (flexibilização de margem análoga às variáveis de folga do SVM;
       cf. Villela et al., 2016 para a equivalência com penalização quadrática)
    - w ≥ 0 garante que a matriz diagonal seja semidefinida positiva,
      condição necessária para d_W ser uma métrica válida (Xing et al., 2002)

A restrição compara distâncias ao centróide — quantidade linear O(m) de restrições,
em contraste com O(m³) restrições de triplas em Schultz & Joachims (2003).
Esta redução é a contribuição principal da formulação de Coelho et al.

Referências:
    [1] Coelho, M.A.N., Borges, C.C.H. & Fonseca Neto, R. (2017).
        "Uso de Predição Estruturada para o Aprendizado de Métrica." CILAMCE 2017.
        → Formulação principal (Eq. 28-29) e algoritmo de relaxação (Seção 5.2).
    [2] Coelho, M.A.N., Fonseca Neto, R. & Borges, C.C.H. (2012).
        "Perceptron models for online structured prediction." IDEAL 2012. Springer, pp. 320-327.
        → Perceptron Estruturado original com margem (Eq. 25).
    [3] Coelho, M.A.N., Borges, C.C.H. & Fonseca Neto, R. (2016).
        "A dual method for solving the nonlinear structured prediction problem."
        Pattern Recognition Letters, 75, pp. 55-62.
        → Método dual e extensão kernel do perceptron estruturado.
    [4] Schultz, M. & Joachims, T. (2003).
        "Learning a distance metric from relative comparisons."
        → Formulação SVM de margem para metric learning (Eq. 10); base teórica da maximização de margem.
    [5] Taskar, B., Chatalbashev, V., Koller, D. & Guestrin, C. (2005).
        "Learning structured prediction models: a large margin approach."
        → Framework de predição estruturada com margem (Eq. 23 no CILAMCE 2017).
    [6] Xing, E.P., Ng, A.Y., Jordan, M.I. & Russell, S. (2002).
        "Distance metric learning with application to clustering with side-information."
        → Formulação original do problema de aprendizado de métrica de Mahalanobis.
    [7] Villela et al. (2016).
        → Equivalência entre penalização quadrática de folga e variáveis duais α.
"""

import numpy as np
import warnings
from typing import Tuple


class RelaxedPerceptron:
    """Aprende uma métrica de Mahalanobis diagonal via Perceptron Estruturado com busca binária em gamma.

    O algoritmo opera em dois níveis:
      - Loop externo: busca binária no gamma ótimo (margem máxima viável).
        Incrementa gamma enquanto o perceptron encontra solução viável;
        quando falha, faz bisseção entre último gamma viável e inviável.
        (CILAMCE 2017, Seção 5.2, Eq. 31)
      - Loop interno: perceptron estruturado com relaxação.
        Para cada gamma fixo, itera sobre os pontos corrigindo w quando
        a restrição de margem é violada. (CILAMCE 2017, Eq. 29)
    """

    def __init__(
        self,
        eta: float = 0.001,
        C: float = 1.0,
        gamma_init: float = 0.1,
        delta_gamma: float = 0.1,
        max_epochs: int = 100,
        tol: float = 1e-5,
        use_best_effort: bool = False,
        max_iterations: int = 50,
        verbose: bool = False,
    ):
        # Taxa de aprendizado (comprimento do passo), cf. CILAMCE 2017 Seção 6:
        # "η = 0.001 e a constante de penalização C variou de 1 até 0.1"
        # (Coelho, Borges & Fonseca Neto, CILAMCE 2017, p. 16)
        self.eta = eta
        # Constante de penalização para flexibilização de margem (λ = 1/C).
        # Controla o trade-off entre maximizar margem e permitir violações.
        # Análogo ao parâmetro C do SVM (Schultz & Joachims, 2003, Eq. 10).
        self.C = C
        # Margem inicial para o loop externo de busca binária (CILAMCE 2017, Eq. 31)
        self.gamma_init = gamma_init
        # Incremento de gamma quando solução viável é encontrada (CILAMCE 2017, Eq. 31: γ(t+1) = γ(t) + δ)
        self.delta_gamma = delta_gamma
        # Máximo de épocas do perceptron interno para cada gamma candidato
        self.max_epochs = max_epochs
        # Tolerância para convergência da busca binária: |gamma_hi - gamma_lo| ≤ tol
        self.tol = tol
        # Se True, retorna melhor solução parcial quando separação perfeita é impossível
        self.use_best_effort = use_best_effort
        # Máximo de iterações do loop externo (busca binária em gamma)
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Resultados após fit()
        self.w_ = None
        self.gamma_ = None
        # Histórico da busca binária em γ (item b da reunião 30/04/2026, ~520s):
        # lista de dicts {iter, gamma, gamma_lo, gamma_hi, violations, viable}.
        # Preenchida em cada iteração do loop externo para diagnóstico de convergência.
        self.gamma_history = []

    @staticmethod
    def d_W(x: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
        """Distância de Mahalanobis com matriz diagonal W.

        d_W(x, c) = Σ_j w_j · (x_j - c_j)²

        Caso particular de d_A(x, c) = (x-c)^T A (x-c) com A = diag(w).
        Quando w = [1,...,1], reduz-se à distância Euclidiana.
        Quando w = 1/σ², reduz-se à distância Euclidiana normalizada.
        (Xing et al., 2002; CILAMCE 2017, Seção 3.1)
        """
        return np.sum(w * (x - c) ** 2)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centroids: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Treina o Perceptron Estruturado com Relaxação de Margem.

        Resolve o problema de maximização de margem (CILAMCE 2017, Eq. 28):
            Max γ  s.t.  d_W(xi,ck) - d_W(xi,cl) + λ·αi ≥ γ·||w||  ∀i, w ≥ 0

        Parâmetros:
            X: matriz (m, d) de pontos de treino
            y: vetor (m,) de rótulos (0 ou 1) — atribuídos pelo especialista ou LLM
            centroids: matriz (n_classes, d) de centróides das classes

        Retorna:
            (w, gamma): vetor de pesos da métrica e margem ótima encontrada
        """
        m, d = X.shape

        # λ = 1/C: constante de penalização das variáveis duais α.
        # Quanto maior C, menor λ, menos penalização → margem mais rígida.
        # Quanto menor C, maior λ, mais penalização → permite mais violações (margem flexível).
        # (Análogo ao SVM: Schultz & Joachims, 2003, Eq. 10; CILAMCE 2017, Eq. 28)
        lambda_ = 1.0 / self.C

        # Inicialização: w = [1,...,1] equivale à métrica Euclidiana (sem preferência dimensional)
        w = np.ones(d)
        # α: variáveis duais / multiplicadores de Lagrange associados às restrições de margem.
        # Inicializadas em zero (nenhuma violação acumulada).
        # (CILAMCE 2017, Eq. 28: "α representa o vetor de variáveis duais")
        alpha = np.zeros(m)
        gamma = self.gamma_init

        # --- Busca binária em gamma (CILAMCE 2017, Seção 5.2, Eq. 31) ---
        # gamma_lo: maior gamma para o qual existe solução viável (todas restrições satisfeitas)
        # gamma_hi: menor gamma para o qual NÃO existe solução viável
        # O ótimo está no intervalo [gamma_lo, gamma_hi].
        gamma_lo = 0.0
        gamma_hi = np.inf

        # w_star/alpha_star: melhor solução viável encontrada (associada a gamma_lo)
        w_star = w.copy()
        alpha_star = alpha.copy()

        # best_effort: solução com menor número de violações (usada como fallback
        # quando separação perfeita é impossível, e.g., dados não linearmente separáveis)
        w_best_effort = w.copy()
        min_global_violations = float("inf")
        found_feasible_solution = False

        iteration = 0
        # Limpa histórico de chamadas anteriores (caso fit() seja chamado múltiplas vezes)
        self.gamma_history = []

        # --- Loop externo: busca binária em gamma ---
        # Estratégia (CILAMCE 2017, Seção 5.2):
        #   1. Começa com gamma_init e incrementa por delta_gamma enquanto viável
        #   2. Quando inviável, faz bisseção: gamma = (gamma_lo + gamma_hi) / 2
        #   3. Converge quando |gamma_hi - gamma_lo| ≤ tol
        while iteration < self.max_iterations:
            iteration += 1

            if self.verbose:
                hi_str = f"{gamma_hi:.6f}" if not np.isinf(gamma_hi) else "∞"
                print(f"    [iter {iteration}] γ={gamma:.6f} (γ_lo={gamma_lo:.6f}, γ_hi={hi_str})")

            final_violations = -1

            # --- Loop interno: perceptron estruturado para gamma fixo ---
            # Para o gamma candidato atual, tenta encontrar w tal que todas
            # as restrições de margem sejam satisfeitas (zero violações).
            for epoch in range(self.max_epochs):
                violations = 0
                # Permutação aleatória dos índices (processamento online/estocástico,
                # análogo ao K-means online, CILAMCE 2017, Seção 4.1)
                indices = np.random.permutation(m)

                for i in indices:
                    xi, yi = X[i], int(y[i])

                    # Calcula distância de xi a todos os centróides usando métrica atual w
                    distances = np.array([self.d_W(xi, c, w) for c in centroids])

                    # l = centróide correto (classe verdadeira de xi)
                    # k = centróide rival mais próximo: k = argmin_{j≠l} d_W(xi, cj)
                    # (CILAMCE 2017, Seção 5.2: "o melhor centróide candidato ck onde k = argmin_{j≠l}")
                    l = yi
                    distances_temp = distances.copy()
                    distances_temp[l] = np.inf
                    k = np.argmin(distances_temp)

                    # ||w||₂ para normalização da margem (margem geométrica, não funcional)
                    norm_w = np.linalg.norm(w)
                    if norm_w < 1e-9:
                        norm_w = 1e-9

                    # Verifica restrição de margem (CILAMCE 2017, Eq. 28):
                    #   d_W(xi, ck) - d_W(xi, cl) + λ·αi ≥ γ·||w||
                    #
                    # Interpretação: a distância ao centróide rival (ck) deve ser
                    # suficientemente maior que a distância ao centróide correto (cl),
                    # com folga proporcional à margem γ normalizada por ||w||.
                    # O termo λ·αi relaxa a restrição para pontos difíceis (variáveis de folga).
                    margin = (
                        self.d_W(xi, centroids[k], w)
                        - self.d_W(xi, centroids[l], w)
                        + lambda_ * alpha[i]
                    )

                    if margin < gamma * norm_w:
                        # Restrição violada → atualizar w e α
                        violations += 1

                        # Gradiente da margem em relação a w (CILAMCE 2017, Eq. 29):
                        #   g = ∂/∂w [d_W(xi,cl) - d_W(xi,ck)]
                        #     = (xi - cl)² - (xi - ck)²
                        #
                        # Nota: g aponta na direção que DIMINUI a margem,
                        # por isso subtraímos (w ← w - η·g) para AUMENTÁ-la.
                        g = (xi - centroids[l]) ** 2 - (xi - centroids[k]) ** 2

                        # Regra de correção completa (CILAMCE 2017, Eq. 29):
                        #   w ← w·(1 - η·γ/||w||) - η·g
                        #   α ← α·(1 - η·γ/||w||)
                        #   αi ← αi + η
                        #
                        # O fator (1 - η·γ/||w||) realiza regularização implícita:
                        # encolhe w proporcionalmente a γ/||w||, o que penaliza ||w|| grande
                        # e favorece soluções com margem geométrica (γ/||w||) significativa.
                        # Análogo ao weight decay em SGD com regularização L2.
                        factor = 1 - self.eta * gamma / norm_w
                        factor = np.clip(factor, 0.0, 2.0)

                        w = w * factor - self.eta * g
                        # Projeção em w ≥ 0: garante semidefinida positiva da matriz diagonal
                        # (Xing et al., 2002: "o vetor de parâmetros deve possuir componentes não negativos")
                        w = np.maximum(0, w)

                        # Atualização das variáveis duais α (CILAMCE 2017, Eq. 29):
                        # - α global é encolhido pelo mesmo fator (regularização conjunta)
                        # - αi é incrementado por η (acumula "custo" de violação do ponto i)
                        alpha = alpha * factor
                        alpha[i] = alpha[i] + self.eta
                        alpha = np.clip(alpha, 0, 1e6)

                final_violations = violations

                # Rastreia melhor solução parcial (menor número de violações)
                # para uso como fallback em problemas não separáveis
                if violations < min_global_violations:
                    min_global_violations = violations
                    w_best_effort = w.copy()

                # Se zero violações: gamma atual é viável, podemos tentar margem maior
                if violations == 0:
                    if self.verbose:
                        print(f"      Convergiu na época {epoch + 1}")
                    break

            # --- Registra entrada no histórico de γ (diagnóstico item b) ---
            viable_now = (final_violations == 0)
            self.gamma_history.append({
                "iter": iteration,
                "gamma": float(gamma),
                "gamma_lo": float(gamma_lo),
                "gamma_hi": float(gamma_hi) if not np.isinf(gamma_hi) else None,
                "violations": int(final_violations),
                "viable": bool(viable_now),
            })

            # --- Atualização da busca binária (CILAMCE 2017, Seção 5.2) ---
            if viable_now:
                # Gamma viável: salva solução e tenta margem maior
                # (CILAMCE 2017, Eq. 31: γ(t+1) = γ(t) + δ)
                found_feasible_solution = True
                gamma_lo = gamma
                w_star = w.copy()
                alpha_star = alpha.copy()
                gamma = gamma + self.delta_gamma
            else:
                # Gamma inviável: marca como limite superior e faz bisseção
                gamma_hi = gamma

                if not np.isinf(gamma_hi):
                    # Bisseção: γ = (γ_lo + γ_hi) / 2
                    # Restaura w e α da última solução viável para reiniciar a busca
                    gamma = (gamma_lo + gamma_hi) / 2
                    w = w_star.copy()
                    alpha = alpha_star.copy()
                else:
                    # Se gamma_hi = ∞ (primeira tentativa já falhou), não há como fazer bisseção
                    break

            # Critério de parada da busca binária
            if abs(gamma_hi - gamma_lo) <= self.tol:
                break

        # --- Retorno dos resultados ---
        if found_feasible_solution:
            # Retorna solução ótima: w* com margem máxima gamma_lo
            self.w_ = w_star
            self.gamma_ = gamma_lo
            return w_star, gamma_lo

        # Fallback para problemas não linearmente separáveis sob métrica diagonal
        if self.use_best_effort:
            # Retorna w com menor número de violações encontrado durante o treinamento.
            # gamma = 0.0 indica que nenhuma margem positiva foi garantida.
            msg = f"Perceptron: separação perfeita não encontrada. Retornando melhor esforço (violações: {min_global_violations}). Métrica pode ser imprecisa."
            if self.verbose:
                print(f"    ⚠️ {msg}")
            else:
                warnings.warn(f"\n  ⚠️ {msg}")
            self.w_ = w_best_effort
            self.gamma_ = 0.0
            return w_best_effort, 0.0
        else:
            # Retorna pesos iniciais (Euclidiana) — sinaliza falha total do aprendizado
            msg = f"Perceptron: separação perfeita não encontrada. Retornando pesos iniciais (equivalente à distância Euclidiana). Verifique a separabilidade dos dados do LLM."
            if self.verbose:
                print(f"    ❌ {msg}")
            else:
                warnings.warn(f"\n  ❌ {msg}")
            self.w_ = w_star
            self.gamma_ = 0.0
            return w_star, 0.0
