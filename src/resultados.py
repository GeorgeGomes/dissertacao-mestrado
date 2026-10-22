"""Dataclasses de resultado dos experimentos (Blocos 1 e 2; o Bloco 3 e os
Problemas F e G, bloco23_*, usam dicts).

Extraídas de ``dissertacao_mestrado.py`` (etapa 1 da modularização, 2026) para que
``plots.py`` e ``relatorios.py`` possam tipar seus parâmetros sem importar o
runner (evita import circular). O runner reexporta os nomes, então
``dissertacao_mestrado.ResultadoABC`` continua válido para os testes.
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from protocolo import EXPERT_W


@dataclass
class LearnedMetric:
    """Armazena a métrica estimada/inferida no Problema A via otimização inversa.

    Nota terminológica: usamos 'estimada' ou 'inferida' (não 'aprendida') porque
    a métrica é obtida por otimização inversa a partir de decisões observadas,
    não por aprendizado supervisionado direto.
    """
    w: np.ndarray
    centroids: np.ndarray
    gamma: float
    source_problem: str
    # 15/10/2026 (D): margem bruta mínima/||w|| sob a solução devolvida (NaN no NNLS); < 0 = não separa os rótulos
    margem_minima: float = float("nan")


@dataclass
class ResultadoABC:
    """Armazena os resultados de um experimento completo (Problema A, Passos 2 e 4; Problemas B e C, Passos 2 a 4)."""
    provider: str
    model_name: str
    temperature: float
    random_seed: int
    k_passo4_bc: int                 # k dos exemplos rotulados pela métrica em B e C (Passo 4); 0 = zero-shot (Passo 2)
    nomes_classes: Tuple[str, str]
    repeticao: int
    # Métricas do Problema A (Passo 4)
    fidelidade_problema_a: float
    acuracia_llm_vs_gt_problema_a: float
    # Métricas do Problema B (Passo 3)
    consistencia_problema_b: float
    kappa_problema_b: float
    f1_problema_b: float
    acuracia_llm_vs_gt_problema_b: float
    acuracia_metrica_vs_gt_problema_b: float
    # Métricas do Problema C (Passo 3)
    consistencia_problema_c: float
    kappa_problema_c: float
    f1_problema_c: float
    acuracia_llm_vs_gt_problema_c: float
    acuracia_metrica_vs_gt_problema_c: float
    # Ŵ_LLM estimada (Perceptron) e γ ótimo. O campo mantém o nome w_aprendido
    # por compatibilidade com os CSVs existentes (terminologia atual: 'estimada').
    w_aprendido: np.ndarray
    gamma_otimo: float
    # Distribuição das classes (número de pontos por classe)
    n_classe_0_problema_a: int
    n_classe_1_problema_a: int
    n_classe_0_problema_b: int
    n_classe_1_problema_b: int
    n_classe_0_problema_c: int
    n_classe_1_problema_c: int
    # Informações detalhadas sobre discordâncias LLM vs. métrica
    n_disagreements_b: int = 0
    n_disagreements_c: int = 0
    # Rastreamento de respostas malformadas do LLM
    n_malformed_responses: int = 0
    # (A consistência euclidiana e a `diagonal_limitation_flag` foram removidas em
    # 09/10/2026, M15: não são métricas da seção IV.3 do orientador.)
    w_ratio: float = 0.0
    # Direção de W (vetor unitário) — invariante à escala, captura a geometria real
    w_direction: np.ndarray = None
    # Similaridade cosseno entre W do Perceptron e do NNLS (robustez ao método)
    w_cosine_sim_nnls: float = 0.0
    # 15/10/2026 (D): margem bruta mínima/||w|| do Perceptron; < 0 = Ŵ não separa os rótulos do LLM (†); NaN se não medida
    margem_minima: float = float("nan")
    collapsed: int = 0     # 1 = o LLM colapsou no Passo 2 do A (classe única ou minoritária < 5): métricas NaN; B/C ficam em collapsed_b/c (14/10/2026)
    collapsed_b: int = 0   # colapso do LLM no Problema B (consistência B NaN)
    collapsed_c: int = 0   # colapso do LLM no Problema C (consistência C NaN)
    # ABLAÇÃO (I10, só CSV e log): consistência/κ com W_A e os centróides TRANSFERIDOS
    # do Problema A, em vez dos centróides locais usados nas colunas principais.
    consistencia_centroides_a_problema_b: float = float("nan")
    consistencia_centroides_a_problema_c: float = float("nan")
    kappa_centroides_a_problema_b: float = float("nan")
    kappa_centroides_a_problema_c: float = float("nan")
    feature_names: Tuple[str, str] = ("x1", "x2")
    prompt_variant: str = "default"


@dataclass
class ResultadoAprendiz:
    """
    Armazena os resultados do Problema E, Passos 2 a 5 (LLM como Aprendiz: zero-shot,
    estratégias, diluição e ordem dos exemplos).

    Linhas ZERO-SHOT (``k == 0``): o prompt não depende da estratégia nem do
    especialista, então o runner coleta UMA vez por (modelo, seed) e reaproveita os
    rótulos (I16) em cada combinação estratégia × especialista. No CSV
    ``bloco2_e_passos_2_3`` isso aparece como 4 estratégias × 3 especialistas = 12 linhas
    zero-shot por (modelo, seed), com as métricas recalculadas contra cada
    especialista, mas com os MESMOS rótulos do LLM e o MESMO ``n_malformed_responses``
    da coleta única. Decisão (09/10/2026): o valor é mantido em todas as linhas, por
    descrever a procedência dos rótulos daquela linha; ao somar malformadas, contar o
    zero-shot uma vez por (modelo, seed) (ou usar a auditoria offline, fonte única).
    """
    provider: str
    model_name: str
    temperature: float
    random_seed: int
    k: int                      # nº de exemplos no prompt (até 13/10/2026: n_shot; a coluna de CSV já era `k`)
    # Estratégia de seleção: "easy"/"hard"/"mixed"/"random" (Problema E, Passo 3), ou
    # "mixed_order_<ordem>" (viés de ordem) ou "dilution_<N>hard_<M>easy" (diluição)
    example_strategy: str
    nomes_classes: Tuple[str, str]
    repeticao: int
    # Métricas principais: LLM vs. Especialista
    accuracy_llm_vs_expert: float
    kappa_llm_vs_expert: float
    f1_llm_vs_expert: float
    # Métricas adicionais
    accuracy_expert_vs_gt: float  # Quão boa é a própria classificação do especialista?
    accuracy_llm_vs_gt: float  # Acurácia do LLM vs. rótulos verdadeiros
    # Distribuição das classes (número de pontos por classe)
    n_classe_0_expert: int
    n_classe_1_expert: int
    n_classe_0_llm: int
    n_classe_1_llm: int
    # Informações de discordância
    n_disagreements: int
    n_total_test: int
    # Respostas malformadas da coleta que gerou os rótulos desta linha (no zero-shot,
    # a coleta única por (modelo, seed) repetida nas 12 linhas; ver docstring)
    n_malformed_responses: int = 0
    # Informações da métrica do especialista (para referência nos resultados)
    expert_w: np.ndarray = field(default_factory=lambda: EXPERT_W.copy())
    expert_name: str = "E1"  # subproblema E1 = especialista W=[0,3; 1,5] (antigo aniso_x2)
