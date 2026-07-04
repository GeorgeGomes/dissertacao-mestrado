"""Constantes do protocolo experimental compartilhadas entre módulos.

Vivem aqui (e não no bloco de configuração do runner) apenas as constantes que
``plots.py``/``relatorios.py`` também consomem — movê-las evita import circular.
O runner as reexporta, então ``dissertacao_mestrado.EXPERT_W`` continua válido.
As demais constantes de configuração permanecem no topo do runner.
"""
import numpy as np


# Métrica do perito para o Bloco 2 / Fase E
# O perito usa uma métrica ANISOTRÓPICA que pondera x2 muito mais do que x1.
# Isso torna a classificação não trivial: o LLM não pode se basear apenas em x1.
EXPERT_W = np.array([0.3, 1.5])  # Weights x2 heavily


EXPERT_CENTROIDS = np.array([[-1.5, 1.0], [1.5, -1.0]])


# Estratégias de dificuldade de exemplos para a Fase E
EXAMPLE_STRATEGIES = ["easy", "hard", "mixed", "random"]
