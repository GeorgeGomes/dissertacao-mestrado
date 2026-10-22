"""Constantes do protocolo experimental compartilhadas entre módulos.

Vivem aqui (e não no bloco de configuração do runner) as constantes do protocolo
que ``plots.py``/``relatorios.py`` também consomem (``EXPERT_W``, ``EXAMPLE_STRATEGIES``,
``PERCEPTRON_PARAMS``) — movê-las evita import circular — e, por coesão, o par
``EXPERT_CENTROIDS`` do especialista. O runner as reexporta, então
``dissertacao_mestrado.EXPERT_W`` continua válido. As demais constantes de
configuração permanecem no topo do runner.
"""
import numpy as np


# Métrica do especialista PRINCIPAL (subproblema E1, antigo "aniso_x2") do Bloco 2 / Problema E:
# anisotrópica, pondera x2 muito mais do que x1, para que o LLM não possa se basear só em x1.
# Os demais especialistas (E2 = antigo aniso_x1, E3 = antigo euclidean) estão em EXPERT_CONFIGS, no runner.
EXPERT_W = np.array([0.3, 1.5])  # Weights x2 heavily


EXPERT_CENTROIDS = np.array([[-1.5, 1.0], [1.5, -1.0]])


# Estratégias de dificuldade de exemplos para o Problema E (Passo 3)
EXAMPLE_STRATEGIES = ["easy", "hard", "mixed", "random"]


# Hiperparâmetros do Perceptron Estruturado usados em TODAS as estimações de W do
# protocolo (Coelho, Borges & Fonseca Neto, CILAMCE 2017, p. 16: η=0.001, C∈[0.1, 1]).
# Ponto ÚNICO de configuração — o runner e print_a_passo4_sensibilidade_perceptron
# (que varia eta/C/delta_gamma localmente de propósito) leem daqui.
PERCEPTRON_PARAMS = {
    "eta": 0.001,
    "C": 1.0,
    "delta_gamma": 0.05,
    "max_epochs": 50,
    "tol": 1e-4,
}


# Piso da classe minoritária para o LLM NÃO ser considerado colapsado (09/10/2026).
# Até 08/10 "colapso" era só classe única; um rótulo 149/1 (visto no Problema D com o
# gpt-4o-mini) era tratado como válido, e o centróide de uma classe com 1 ponto gera uma
# métrica artefato que entrava nas médias. Com menos de MIN_PONTOS_CLASSE_MINORITARIA
# pontos na classe menos frequente, a coleta é marcada `collapsed=1` (medidas NaN, fora
# das médias e figuras) em todos os problemas (A, B/C, D, E, F e G).
MIN_PONTOS_CLASSE_MINORITARIA = 5


def colapsou(y_llm) -> bool:
    """Regra única de colapso do LLM (I13, 09/10/2026): classe única OU classe
    minoritária com menos de MIN_PONTOS_CLASSE_MINORITARIA pontos.

    Uma coleta colapsada não gera métrica (centróide de 1 ponto é artefato): vira
    `collapsed=1` com medidas NaN em todos os problemas e passos. Vive aqui para que
    o runner, ``plots.py`` e as tabelas apliquem o MESMO critério (13/10/2026: antes a
    figura ``bloco1_graphic_14`` e os painéis de fechamento só reconheciam classe única).
    """
    y = np.asarray(y_llm)
    if len(y) == 0:
        return True
    n1 = int(np.sum(y == 1))
    return min(n1, len(y) - n1) < MIN_PONTOS_CLASSE_MINORITARIA
