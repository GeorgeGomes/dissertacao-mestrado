"""Geração dos conjuntos de dados sintéticos e carga da base real.

Reúne a geometria dos problemas (centróides e desvios) e os geradores usados nos três
blocos: A/B/C (gaussianos lineares), E (perito linear), D/F (meia-lua não-linear),
peso×altura (base real) e o gerador anisotrópico da Oracle Validation.

Extraído de ``dissertacao_mestrado.py`` para isolar a geração de dados do runner e
permitir testes de shape/seed/reprodutibilidade (ver ``tests/test_data_problems.py``).
Depende apenas de numpy/pandas/scikit-learn — não do runner (evita import circular).
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons


# ── Geometria dos problemas (centróides e desvio-padrão) ────────────────────
# Parâmetros do Problema A (linha de base, isotrópico — classes bem separadas horizontalmente)
PROBLEM_A_CENTERS = [(-2.0, 0.0), (2.0, 0.0)]
PROBLEM_A_STD = 1.2

# Parâmetros do Problema B (rotação HORÁRIA AGRESSIVA — centróides deslocados em ±1.5)
# Simétrico ao Problema C (anti-horária) e proposital para evitar transferência trivial.
PROBLEM_B_CENTERS = [(-2.0, 1.5), (2.0, -1.5)]
PROBLEM_B_STD = 1.2

# Parâmetros do Problema C (rotação ANTI-HORÁRIA AGRESSIVA em relação ao A:
# classe 0 desce, classe 1 sobe — orientação oposta ao Problema B, mesma magnitude)
PROBLEM_C_CENTERS = [(-2.0, -1.5), (2.0, 1.5)]
PROBLEM_C_STD = 1.2

# Parâmetros do Problema E (Bloco 2 — perito linear, antigo "Problema D" Fase D antiga)
# Geometria propositalmente distinta para que a métrica do perito não seja trivial.
PROBLEM_E_CENTERS = [(-1.5, 1.0), (1.5, -1.0)]
PROBLEM_E_STD = 1.3


# ── Geradores ───────────────────────────────────────────────────────────────

def create_problem_a(n_samples: int = 150, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema A para aprendizado da métrica.

    Problema A serve como base de treino: o LLM classifica estes pontos em zero-shot
    e a métrica W é estimada a partir dessas decisões.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_A_CENTERS,
        cluster_std=PROBLEM_A_STD,
        random_state=random_state
    )
    return X, y


def create_problem_b(n_samples: int = 100, random_state: int = 43) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema B (Bloco 1 — rotação HORÁRIA AGRESSIVA).

    Centróides em (-2, +1.5) e (2, -1.5), pareados simetricamente com C (±1.5,
    direção oposta). Decisão 14/05/2026: magnitude ±1.5 testa transferência de W
    em condições severas, mais agressiva que a versão anterior ±0.8.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_B_CENTERS,
        cluster_std=PROBLEM_B_STD,
        random_state=random_state
    )
    return X, y


def create_problem_c(n_samples: int = 100, random_state: int = 44) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema C para teste de consistência adicional.

    Problema C aplica rotação anti-horária dos centróides (classe 0 para baixo,
    classe 1 para cima), de orientação oposta ao Problema B (que é horária).
    Garante que B e C sejam visualmente distintos para testar a generalização da
    métrica em duas direções geométricas diferentes.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_C_CENTERS,
        cluster_std=PROBLEM_C_STD,
        random_state=random_state
    )
    return X, y


def create_problem_e_expert(n_samples: int = 150, random_state: int = 45) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera o conjunto de dados do Problema E (Bloco 2 — perito linear, antigo "Problema D"
    da Fase D antiga; agora Fase E quando o LLM atua como aprendiz via in-context learning).
    Usa geometria propositalmente distinta para que o LLM não possa aprender a métrica do
    perito por intuição simples — é necessário capturar o peso anisotrópico w2 >> w1.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_E_CENTERS,
        cluster_std=PROBLEM_E_STD,
        random_state=random_state
    )
    return X, y


def create_problem_d_meialua(n_samples: int = 150, random_state: int = 46) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o Problema E — meia-lua (não-linear).

    Caso canônico de fronteira não-linear, gerado por ``sklearn.datasets.make_moons``.
    Justificativa (reunião 30/04/2026, ~2558s): o R3 atual sobre A/B/C lineares
    não melhora ao adicionar x3=x1·x2. Aqui sim — em problema genuinamente
    não-linear espera-se que features quadráticas tragam ganho mensurável.
    """
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=random_state)
    return X, y


def create_problem_homem_mulher(csv_path: str = "dados_reais/homem_mulher/peso_altura.csv") -> Tuple[np.ndarray, np.ndarray]:
    """Carrega a base real peso × altura (homem/mulher) enviada pelo orientador.

    Dataset com 100 amostras normalizadas (~[-1, 1]), balanceado 50/50.
    Classificador ótimo bayesiano é uma elipse (fronteira quadrática):
        f(x1, x2) = x2^2 - x2 + x1^2 - x1 + cte
    Por isso a Fase A com 4 features (x1, x2, x1², x2²) deve recuperar melhor
    o critério do que a versão linear de 2 features.

    Args:
        csv_path: caminho relativo ao diretório de trabalho para o CSV consolidado.

    Returns:
        (X, y) onde X tem shape (n, 2) com colunas (peso, altura) e y em {0, 1}.
    """
    df = pd.read_csv(csv_path)
    X = df[["peso", "altura"]].values.astype(float)
    y = df["classe"].values.astype(int)
    return X, y


def create_anisotropic_problem(
    n_samples: int,
    centers: List[Tuple[float, float]],
    std_per_dim: List[float],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados com variância diferente por dimensão (anisotrópico).

    Gera clusters isotrópicos unitários centrados na origem e depois escala
    cada dimensão pelo desvio padrão desejado e translada para os centróides.
    """
    centers_arr = np.array(centers)
    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[0, 0]] * len(centers),
        cluster_std=1.0,
        random_state=random_state,
    )
    for c in range(len(centers)):
        mask = y == c
        X[mask, 0] = X[mask, 0] * std_per_dim[0] + centers_arr[c, 0]
        X[mask, 1] = X[mask, 1] * std_per_dim[1] + centers_arr[c, 1]
    return X, y
