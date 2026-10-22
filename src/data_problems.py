"""Geração dos conjuntos de dados sintéticos e carga da base real.

Reúne a geometria dos problemas (centróides e desvios) e os geradores usados nos três
blocos: A/B/C (gaussianos lineares), E (especialista linear), D/F (meia-lua não-linear)
e peso×altura (base real). A validação do oráculo usa os dados de E (Oráculo sobre B,
E1 e E2, todos com a geometria do Problema E).

Extraído de ``dissertacao_mestrado.py`` para isolar a geração de dados do runner e
permitir testes de shape/seed/reprodutibilidade (ver ``tests/test_data_problems.py``).
Depende apenas de numpy/pandas/scikit-learn — não do runner (evita import circular).
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons


# ── Geometria dos problemas (centróides e desvio-padrão) ────────────────────
# Parâmetros do Problema A (linha de base). Desde 15/10/2026 os centróides diferem nas DUAS
# coordenadas: com (−2, 0) e (2, 0) (até 14/10/2026) a diferença de distâncias do centróide mais
# próximo não dependia de w2 e a métrica estimada no A deixava w2 livre (instabilidade de H1 em B
# entre as completas de 13/10 e 14/10/2026). Mesma separação em x1 e mesmo σ.
PROBLEM_A_CENTERS = [(-2.0, -0.5), (2.0, 0.5)]
PROBLEM_A_CENTERS_ATE_2026_10_14 = [(-2.0, 0.0), (2.0, 0.0)]   # só para ler execuções antigas e testes de regressão
PROBLEM_A_STD = 1.2

# Parâmetros do Problema B (rotação HORÁRIA em relação ao A: classe 0 sobe, classe 1 desce).
# Coordenadas FIXAS (iguais às de todas as execuções), para B e C continuarem comparáveis entre
# execuções e com a mesma separação entre si; desde 15/10/2026 o A está inclinado em +0,5, logo B
# gira 2,0 no sentido horário e C 1,0 no anti-horário em relação ao A (antes: ±1,5 simétricos).
PROBLEM_B_CENTERS = [(-2.0, 1.5), (2.0, -1.5)]
PROBLEM_B_STD = 1.2

# Parâmetros do Problema C (rotação ANTI-HORÁRIA AGRESSIVA em relação ao A:
# classe 0 desce, classe 1 sobe — orientação oposta ao Problema B, mesma magnitude)
PROBLEM_C_CENTERS = [(-2.0, -1.5), (2.0, 1.5)]
PROBLEM_C_STD = 1.2

# Parâmetros do Problema E (Bloco 2 — especialista linear; chamava-se "Problema D" até
# 14/05/2026, quando a meia-lua passou a ser o Problema D; o arquivo problema_D.csv só foi
# renomeado para problema_E.csv em 01/10/2026 — ver renomear_assets_legado.py)
# Geometria propositalmente distinta para que a métrica do especialista não seja trivial.
PROBLEM_E_CENTERS = [(-1.5, 1.0), (1.5, -1.0)]
PROBLEM_E_STD = 1.3


# ── Geradores ───────────────────────────────────────────────────────────────

def create_problem_a(n_samples: int = 150, random_state: int = 42,
                     centers=None) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema A para aprendizado da métrica.

    Problema A serve como base de treino: o LLM classifica estes pontos em zero-shot
    e a métrica W é estimada a partir dessas decisões. ``centers`` (opcional) permite
    reproduzir execuções anteriores a 15/10/2026 com `PROBLEM_A_CENTERS_ATE_2026_10_14`.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_A_CENTERS if centers is None else centers,
        cluster_std=PROBLEM_A_STD,
        random_state=random_state
    )
    return X, y


def create_problem_b(n_samples: int = 100, random_state: int = 43) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema B (Bloco 1 — rotação HORÁRIA em relação ao A).

    Centróides FIXOS em (-2, +1.5) e (2, -1.5) (classe 0 sobe, classe 1 desce), iguais em
    todas as execuções. Como desde 15/10/2026 o A está inclinado em +0,5 (centróides
    (-2, -0.5) e (2, 0.5)), B gira 2,0 no sentido horário em relação ao A (até 14/10/2026,
    com o A horizontal, eram ±1,5 simétricos com C). Decisão 14/05/2026: magnitude maior
    que a versão anterior (±0.8) para testar a transferência de W em condições severas.
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

    Centróides FIXOS em (-2, -1.5) e (2, +1.5) (classe 0 para baixo, classe 1 para cima),
    de orientação oposta ao Problema B (que é horária). Com o A inclinado em +0,5 (desde
    15/10/2026), C gira 1,0 no sentido anti-horário em relação ao A, enquanto B gira 2,0 no
    horário (até 14/10/2026 eram ±1,5 simétricos). Garante que B e C sejam visualmente
    distintos para testar a generalização da métrica em duas direções geométricas.
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
    Gera o conjunto de dados do Problema E (Bloco 2 — especialista linear; até 14/05/2026
    era o "Problema D"; o CSV problem_D.csv → problem_E.csv em 01/10/2026 e problema_E.csv desde 13/10/2026). Usado no Problema E, em que o LLM atua
    como aprendiz via in-context learning.
    Usa geometria propositalmente distinta para que o LLM não possa aprender a métrica do
    especialista por intuição simples — é necessário capturar o peso anisotrópico w2 >> w1.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_E_CENTERS,
        cluster_std=PROBLEM_E_STD,
        random_state=random_state
    )
    return X, y


def create_problem_d_meialua(n_samples: int = 150, random_state: int = 46) -> Tuple[np.ndarray, np.ndarray]:
    """Gera a meia-lua (não-linear): Problema D no Bloco 1 (LLM como especialista) e
    Problema F no Bloco 2 (LLM como aprendiz).

    Caso canônico de fronteira não-linear, gerado por ``sklearn.datasets.make_moons``.
    Justificativa (reunião 30/04/2026): o R³ sobre A/B/C lineares
    não melhora ao adicionar x3=x1·x2. Aqui sim — em problema genuinamente
    não-linear espera-se que features quadráticas tragam ganho mensurável.
    """
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=random_state)
    return X, y


def create_problem_homem_mulher_kg_m(
    csv_path: str = "dados_reais/homem_mulher/peso_altura_kg_m.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega a base peso × altura em unidades físicas (subproblema G2).

    ``peso_altura_kg_m.csv`` é a mesma base de ``peso_altura.csv`` (mesma ordem de
    linhas, mesmo rótulo) convertida por regra de três (e-mail do orientador de
    28/08/2026): peso 45–110 kg, altura 1,50–1,95 m. Serve APENAS para o texto do
    prompt do G2; a métrica continua estimada nos valores normalizados.

    Returns:
        (X_fis, y) com X_fis (n, 2) = (peso_kg, altura_m) e y em {0, 1}, 1 = Homem.
    """
    df = pd.read_csv(csv_path)
    X = df[["peso_kg", "altura_m"]].values.astype(float)
    y = df["classe"].values.astype(int)
    return X, y


def create_problem_homem_mulher(csv_path: str = "dados_reais/homem_mulher/peso_altura.csv") -> Tuple[np.ndarray, np.ndarray]:
    """Carrega a base real peso × altura (homem/mulher) enviada pelo orientador.

    Dataset com 100 amostras normalizadas (~[-1, 1]), balanceado 50/50.
    Classificador ótimo bayesiano é uma elipse (fronteira quadrática):
        f(x1, x2) = x2^2 - x2 + x1^2 - x1 + cte
    Por isso a métrica com 4 atributos (x1, x2, x1², x2²) deve recuperar melhor
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
