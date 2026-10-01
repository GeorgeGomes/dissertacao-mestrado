"""Dataclasses de resultado dos experimentos (Blocos 1 e 2; o Bloco 3 e os pipelines
externos bloco23_* usam dicts).

Extraídas de ``dissertacao_mestrado.py`` (Fase 1 da modularização) para que
``plots.py`` e ``relatorios.py`` possam tipar seus parâmetros sem importar o
runner (evita import circular). O runner reexporta os nomes, então
``dissertacao_mestrado.ResultadoExperimento`` continua válido para os testes.
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


@dataclass
class ResultadoExperimento:
    """Armazena os resultados de um experimento completo (Fase A + Fase B + Fase C)."""
    provider: str
    model_name: str
    temperature: float
    random_seed: int
    n_shot: int
    nomes_classes: Tuple[str, str]
    repeticao: int
    # Métricas da Fase A
    fidelidade_problema_a: float
    acuracia_llm_vs_gt_problema_a: float
    # Métricas da Fase B
    consistencia_problema_b: float
    kappa_problema_b: float
    f1_problema_b: float
    acuracia_llm_vs_gt_problema_b: float
    acuracia_metrica_vs_gt_problema_b: float
    # Métricas da Fase C
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
    # Consistência da linha de base euclidiana (para verificação de limitação da métrica diagonal)
    consistencia_euclidiana_problema_b: float = 0.0
    consistencia_euclidiana_problema_c: float = 0.0
    diagonal_limitation_flag: int = 0
    w_ratio: float = 0.0
    # Direção de W (vetor unitário) — invariante à escala, captura a geometria real
    w_direction: np.ndarray = None
    # Similaridade cosseno entre W do Perceptron e do NNLS (robustez ao método)
    w_cosine_sim_nnls: float = 0.0
    feature_names: Tuple[str, str] = ("x1", "x2")
    prompt_variant: str = "default"


@dataclass
class ResultadoPhaseEExperimento:
    """
    Armazena os resultados do experimento da Fase E (LLM como Aprendiz).
    """
    provider: str
    model_name: str
    temperature: float
    random_seed: int
    n_shot: int
    # Estratégia de seleção: "easy"/"hard"/"mixed"/"random" (Fase E), ou
    # "mixed_order_<ordem>" (viés de ordem) ou "dilution_<N>hard_<M>easy" (diluição)
    example_strategy: str
    nomes_classes: Tuple[str, str]
    repeticao: int
    # Métricas principais: LLM vs. Perito
    accuracy_llm_vs_expert: float
    kappa_llm_vs_expert: float
    f1_llm_vs_expert: float
    # Métricas adicionais
    accuracy_expert_vs_gt: float  # Quão boa é a própria classificação do perito?
    accuracy_llm_vs_gt: float  # Acurácia do LLM vs. rótulos verdadeiros
    # Distribuição das classes (número de pontos por classe)
    n_classe_0_expert: int
    n_classe_1_expert: int
    n_classe_0_llm: int
    n_classe_1_llm: int
    # Informações de discordância
    n_disagreements: int
    n_total_test: int
    # Respostas malformadas
    n_malformed_responses: int = 0
    # Informações da métrica do perito (para referência nos resultados)
    expert_w: np.ndarray = field(default_factory=lambda: EXPERT_W.copy())
    expert_name: str = "aniso_x2"
