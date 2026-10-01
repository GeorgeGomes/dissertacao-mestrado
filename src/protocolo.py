"""Constantes do protocolo experimental compartilhadas entre módulos.

Vivem aqui (e não no bloco de configuração do runner) as constantes do protocolo
que ``plots.py``/``relatorios.py`` também consomem (``EXPERT_W``, ``EXAMPLE_STRATEGIES``,
``PERCEPTRON_PARAMS``) — movê-las evita import circular — e, por coesão, o par
``EXPERT_CENTROIDS`` do perito. O runner as reexporta, então
``dissertacao_mestrado.EXPERT_W`` continua válido. As demais constantes de
configuração permanecem no topo do runner.
"""
import numpy as np


# Métrica do perito PRINCIPAL ("aniso_x2") do Bloco 2 / Fase E: anisotrópica,
# pondera x2 muito mais do que x1, para que o LLM não possa se basear só em x1.
# Os demais peritos (aniso_x1, euclidean) estão em EXPERT_CONFIGS, no runner.
EXPERT_W = np.array([0.3, 1.5])  # Weights x2 heavily


EXPERT_CENTROIDS = np.array([[-1.5, 1.0], [1.5, -1.0]])


# Estratégias de dificuldade de exemplos para a Fase E
EXAMPLE_STRATEGIES = ["easy", "hard", "mixed", "random"]


# Hiperparâmetros do Perceptron Estruturado usados em TODAS as estimações de W do
# protocolo (Coelho, Borges & Fonseca Neto, CILAMCE 2017, p. 16: η=0.001, C∈[0.1, 1]).
# Ponto ÚNICO de configuração — o runner e print_hyperparameter_sensitivity
# (que varia eta/C/delta_gamma localmente de propósito) leem daqui.
PERCEPTRON_PARAMS = {
    "eta": 0.001,
    "C": 1.0,
    "delta_gamma": 0.05,
    "max_epochs": 50,
    "tol": 1e-4,
}
