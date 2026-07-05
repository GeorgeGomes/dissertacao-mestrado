import os
from dotenv import load_dotenv
load_dotenv()


"""
EXPLICANDO DECISÕES DE LLMs VIA OTIMIZAÇÃO INVERSA
EXPERIMENTO ORGANIZADO EM 3 BLOCOS AUTO-CONTIDOS

=====================================================
CONTEXTO E MOTIVAÇÃO
=====================================================

A questão central da dissertação é: um LLM possui um processo decisório ESTÁVEL
e APRENDÍVEL? Quando classifica pontos, o LLM utiliza algum critério implícito.
Esse critério pode ser capturado matematicamente via otimização inversa.
Se for estável, aplicá-lo a outros problemas deve produzir classificações
concordantes com as que o LLM faria por conta própria.

Algoritmos de aprendizado (otimização inversa):
  - Perceptron Estruturado com Relaxação de Margem (Coelho, Borges &
    Fonseca Neto, CILAMCE 2017; eta=0.001, C in [0.1, 1.0]).
  - NNLS (Lawson-Hanson 1974, scipy.optimize.nnls; sem hiperparâmetros).

=====================================================
ORGANIZAÇÃO EM 3 BLOCOS AUTO-CONTIDOS (refactor 14/05/2026)
=====================================================

BLOCO 1 — LLM como FONTE (otimização inversa)
  Problemas:
    A — linear sintético (centroides ±2.0 horizontal, std=1.2)
    B — linear, rotação HORARIA AGRESSIVA (centroides ±1.5)
    C — linear, rotação ANTI-HORARIA AGRESSIVA (centroides ±1.5,
        mesma magnitude de B, direção oposta — pareados)
    D — meia-lua sintética (sklearn.make_moons, noise=0.15) [não-linear]
  Experimentos: Oracle Validation (sanity), Fase A (W via Perc+NNLS),
    Fases B/C (transferência), R2 -> R3 -> R4 em todos os 4 problemas,
    vieses de classe, variantes de prompt, nomes semânticos.

BLOCO 2 — LLM como APRENDIZ (perito externo via in-context learning)
  Problemas:
    E — antigo "Problema D" linear (centroides ±1.5/±1.0, std=1.3),
        perito com W_expert=[0.3, 1.5] (x2 dominante).
    F — meia-lua sintética com perito = ground truth do make_moons.
  Experimentos: Fase E no Problema E (estratégias easy/hard/mixed/random,
    múltiplos experts, diluição, recency bias), Fase E no Problema F,
    baselines clássicos (k-NN, LR, SVM), LLM vs Perceptron baseline.

BLOCO 3 — Estudo de caso REAL (peso × altura)
  Problema: base do orientador (~100 amostras, peso vs altura normalizado),
    classificador ótimo bayesiano com fronteira ELIPTICA:
      f(x1,x2) = x2^2 - x2 + x1^2 - x1 + cte
  Experimentos: Fase A R2/R3/R4 (R4 = elipse, recupera classif. ótimo),
    acurácia vs rótulo REAL (atende e-mail orientador 22:04, ponto 4),
    paradoxo do overfitting, Fase E, LLM vs Perceptron, "quando o LLM ganha?".

=====================================================
NOMENCLATURA DAS FASES EXPERIMENTAIS
=====================================================

Fase A — LLM rotula em zero-shot no Problema A (linear). Aprende-se W via
         Perceptron Estruturado + NNLS (otim. inversa). Bloco 1.
Fase B — Aplicação de W em Problema B (rotação horária ±1.5). Bloco 1.
Fase C — Aplicação de W em Problema C (rotação anti-horária ±1.5). Bloco 1.
Fase D — Aplicação em Problema D (meia-lua não-linear) com augmentação
         R2/R3/R4. Detecta não-linearidade implícita no critério do LLM.
         Bloco 1, realizada via run_external_problem_pipeline(problem_name="meia_lua_...").
Fase E — LLM como APRENDIZ (in-context learning). Bloco 2. Um perito
         externo (Problema E linear com W_expert=[0.3, 1.5] ou Problema F
         meia-lua com ground truth) rotula os dados; o LLM tenta reproduzir
         o critério via exemplos few-shot. Função: phase_e_llm_as_learner.

=====================================================
PROBLEMA DIRETO vs. PROBLEMA INVERSO
=====================================================

PROBLEMA DIRETO: dado classificador conhecido, classificar novos pontos.
  -> LLM faz isso naturalmente (zero/few-shot).

PROBLEMA INVERSO: dadas as classificações do LLM, INFERIR o critério W.
  -> Aprendemos W via Perceptron Estruturado E NNLS em paralelo (cosseno
     entre W's > 0.97; robustez ao método de estimação).
  -> Aplicamos W a outros problemas para ver se o LLM se repete.

Mahalanobis DIAGONAL: O(d) parâmetros vs O(d^2) na cheia. Simplificação
  deliberada; próximo passo é evoluir para forma SDP parametrizada mais
  geral (orientador, Reunião 1).

=====================================================
HIPÓTESES DO EXPERIMENTO
=====================================================

H1 (Consistência) — Bloco 1: o LLM mantém critério decisório implícito em
  problemas distintos. W estimada em A prevê classificações em B/C com
  Kappa > 0.5 (Landis & Koch 1977).

H2 (Few-shot amplifica) — Bloco 1: exemplos rotulados pela métrica
  aumentam a concordância LLM-métrica.

H3 (LLM como aprendiz) — Bloco 2: via in-context learning, o LLM reproduz
  o critério de um perito externo. Concordância cresce com n_shot.

H4 (Exemplos difíceis) — Bloco 2 [REFUTADA com significância]: hipótese
  a priori "hard > easy". Refutada: easy > hard em n_shot pequeno
  (Cohen d ~ 1.5); efeito desaparece em n_shot >= 20.

H5 (Estabilidade) — Transversal: comportamento reproduzível entre 3 seeds
  x 3 repetições = 9 observações por configuração.

=====================================================
DESIGN DE VARIABILIDADE
=====================================================

- N_REPETICOES=3: variabilidade do LLM na MESMA base.
- RANDOM_SEEDS=[42, 123, 7]: BASES distintas.
- NOMES_CLASSES variados: detecta viés semântico (Bloco 1).
- Temperatura 0.0 REDUZ mas não elimina estocasticidade. Motivos:
    batching/arredondamento, hardware heterogêneo, atualizações silenciosas
    do provider, top-k residual, softmax paralelo não-associativo.

=====================================================
RECURSOS PRINCIPAIS (mapeados aos blocos)
=====================================================
- BLOCO 1: Fase A (otim. inversa), Fases B/C (transferência), Oracle
  Validation, R2 -> R3 -> R4 em lineares + meia-lua, vieses, prompts.
- BLOCO 2: Fase E no perito linear (estratégias, múltiplos experts,
  diluição, recency bias), Fase E na meia-lua, baselines clássicos,
  LLM vs Perceptron baseline.
- BLOCO 3: peso x altura (Fase A R2/R3/R4, paradoxo do overfitting,
  Fase E, LLM vs Perceptron).
- Comum: bootstrap CI 10k, Wilcoxon pareado, Cohen's d, concorrência
  asyncio.Semaphore(10), parser 7 camadas + fallback MD5, 3 seeds x 3 reps,
  30+ PNGs e 11 CSVs por execução nomeados com prefixos bloco{1,2,3}_,
  bloco23_, final_.

PROVEDORES: OpenAI e OpenRouter (ativos); Anthropic/Gemini direto (SDK integrado, inativo).
"""

import sys
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Optional, Dict
import warnings

# matplotlib/seaborn agora vivem em plots.py (que fixa o backend 'Agg');
# json/re/dataclasses migraram junto com execucao_io.py e resultados.py.


from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from classical_baselines import ClassicalBaselineRunner

import anthropic
import time
import asyncio
import traceback

from relaxed_perceptron import train_relaxed_perceptron
from least_squares_inverse import train_least_squares_inverse
from llm_parser import parse_llm_response
from metrics import (
    # d_W é reexportado aqui para os testes (dm.d_W em test_metrics.py).
    ConsistencyMetrics, compute_consistency_metrics,
    d_W, compute_centroids, augment_to_r3, augment_to_r4, augment_features,
    predict_with_metric, compute_metric_confidence,
)
from llm_client import PROVIDER_CONFIG, get_client, get_async_client, get_extra_body
from data_problems import (
    PROBLEM_A_CENTERS, PROBLEM_B_CENTERS, PROBLEM_C_CENTERS,
    create_problem_a, create_problem_b, create_problem_c, create_problem_e_expert,
    create_problem_d_meialua, create_problem_homem_mulher, create_anisotropic_problem,
)
from plots import (  # reexport: extraído na Fase 1 da modularização
    plot_algorithm_comparison, plot_class_names_effect, plot_class_order_bias,
    plot_classical_baselines_comparison, plot_confusion_matrices_detailed, plot_consistency_comparison_extended, plot_dataset_overview,
    plot_dilution_experiment, plot_example_order_bias, plot_experiment_summary_dashboard, plot_external_decision_boundary,
    plot_external_features_comparison, plot_external_learning_curve, plot_feature_names_effect, plot_gamma_convergence,
    plot_hits_and_errors, plot_llm_labels_per_problem, plot_margin_analysis_detailed, plot_meialua_svm_vs_llm,
    plot_metric_errors_phase_a, plot_model_comparison, plot_oracle_meialua, plot_oracle_transfer,
    plot_oracle_w_recovery, plot_phase_e_example_locations, plot_phase_e_learning_curve, plot_phase_e_llm_vs_perceptron,
    plot_phase_e_strategy_comparison, plot_problem_overview, plot_prompt_variant_comparison, plot_r3_comparison,
    plot_seed_comparison, plot_w_comparison_algorithms, plot_w_distribution, visualize_all_problems,
    visualize_problem_e_with_expert,
)
from relatorios import (  # reexport: extraído na Fase 1 da modularização
    print_error_analysis_by_region, print_example_order_analysis,
    print_final_analysis, print_hyperparameter_sensitivity, print_phase_e_analysis, print_section,
    print_statistical_summary, summarize_cross_linearity,
)
from execucao_io import (  # reexport: extraído na Fase 1 da modularização
    # MODEL_ALIAS e _model_slug são reexportados aqui para os testes (dm.*).
    LLM_INTERACTIONS, LOG_CHUNK_LIMIT_BYTES, MODEL_ALIAS, Tee,
    _model_alias, _model_slug, checkpoint_interactions,
    llm_asset, salvar_json_em_chunks, salvar_log_em_chunks,
)
from resultados import (  # reexport: extraído na Fase 1 da modularização
    LearnedMetric, ResultadoExperimento, ResultadoPhaseEExperimento,
)
from protocolo import (  # reexport: extraído na Fase 1 da modularização
    EXAMPLE_STRATEGIES, EXPERT_CENTROIDS, EXPERT_W,
)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Raiz do projeto (pasta acima de src/). Ancora dados reais e pastas de execução
# ao repositório, e NÃO ao diretório de trabalho — assim o script produz os mesmos
# caminhos independentemente de onde é invocado (evita FileNotFoundError no dataset
# real e outputs espalhados quando rodado de dentro de src/).
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════════
# SELEÇÃO DE MODELOS - ADICIONE OS MODELOS QUE DESEJA COMPARAR
# ═══════════════════════════════════════════════════════════════════════════

# Formato: (provider, model, temperature, scope)
#   scope "full" — grid completo (vieses, variantes, diluição, ordem, múltiplos peritos)
#                  + seeds extras do pipeline central (EXTRA_SEEDS_CORE).
#   scope "core" — apenas o pipeline central: Fases A-C + Fase E (perito principal)
#                  + pipelines externos (peso×altura e meia-lua). Comparação entre
#                  modelos sem multiplicar o custo do grid auxiliar.
MODELS_TO_TEST = [
    # Protocolo: 2 modelos, ambos com scope="full" — o grid COMPLETO de
    # experimentos roda nos 2, tratados de forma IGUAL: cada um gera o conjunto
    # completo de gráficos na raiz da execução, com seu alias (MODEL_ALIAS) no
    # nome do asset. CSVs consolidam ambos (colunas provider/model).
    ("openai", "gpt-4o-mini", 0.0, "full"),
    ("openrouter", "google/gemini-2.5-flash-lite", 0.0, "full"),
]

# PROVIDER_CONFIG, get_client e get_async_client → llm_client.py (importados no topo).
# Ver tests/test_llm_client.py.


# Cliente global — será definido para cada modelo durante os experimentos
client = None
async_client = None  # Cliente assíncrono para chamadas concorrentes
MODEL_NAME = None
CURRENT_PROVIDER = None
# Temperatura 0.0: REDUZ mas NÃO ELIMINA estocasticidade do LLM.
# Mesmo com temp=0, variabilidade pode ocorrer por: batching em GPU (arredondamentos float16),
# hardware heterogêneo entre requests, atualizações silenciosas do modelo pelo provider,
# e paralelismo não-determinístico em operações de ponto flutuante (softmax não-associativa).
# Por isso o experimento usa múltiplas seeds (RANDOM_SEEDS) e repetições (N_REPETICOES).
CURRENT_TEMPERATURE = 0.0


# Bloco 1 (A) e Bloco 2 (E) usam mais amostras — envolvem aprendizado.
# Problemas B e C (testes de consistência) precisam só de 100 pontos para Kappa/F1 confiáveis.
N_SAMPLES_PROBLEM_A = 150  # Bloco 1: treino da métrica via otim. inversa
N_SAMPLES_PROBLEM_B = 100  # Bloco 1: teste de consistência (rotação horária ±1.5)
N_SAMPLES_PROBLEM_C = 100  # Bloco 1: teste de consistência (rotação anti-horária ±1.5)
N_SAMPLES_PROBLEM_E = 150  # Bloco 2: perito linear (Fase E)


# Meia-lua (Bloco 1 = Problema D; Bloco 2 = Problema F) usa N_SAMPLES_PROBLEM_A=150 também
FEW_SHOT_SIZES = [0, 4, 10, 20, 40]  # Tamanhos few-shot para Fases B e C (pares: balanço exato de classes no prompt)
FEW_SHOT_SIZES_PHASE_E = [0, 4, 10, 20, 40]  # versão completa (pares: balanço exato)
N_REPETICOES = 3

# Pontos de ancoragem dos experimentos de viés (inversão de nomes de classe e
# variantes de prompt): mesmos n_shot para que os efeitos sejam comparáveis entre si.
BIAS_N_SHOTS = [0, 10]

# Grades dos experimentos auxiliares (reduzidas no --rapido):
DILUTION_EASY_ADDITIONS = [0, 2, 4, 10, 16, 20]  # N easy adicionados na diluição
EXAMPLE_ORDER_N_SHOTS = [4, 10, 20]              # n_shots do viés de ordem

# Múltiplas sementes aleatórias para garantir robustez dos resultados
RANDOM_SEEDS = [42, 123, 7]

# Sementes EXTRAS do pipeline central (apenas o modelo principal): elevariam o n de
# bases distintas para 6 nas Fases A-C e na Fase E (perito principal), habilitando
# testes pareados por seed (Wilcoxon exige n>=5-6 para p<0,05).
# DESATIVADAS (decisão 02/07/2026): protocolo uniforme de 3 seeds para tudo, em
# todos os modelos. Para reativar as 6 seeds no central: [2025, 314, 611].
EXTRA_SEEDS_CORE = []


def reps_para(n_shot: int, sorteio_estocastico: bool = True) -> int:
    """Regra única de repetição do protocolo.

    Repetições existem para variar o SORTEIO dos exemplos few-shot (variância de
    amostragem). Onde não há sorteio — zero-shot, ou seleção determinística por
    margem (Fases B/C, diluição) — roda-se 1 coleta; o não-determinismo por consulta
    do LLM (flip a T=0) é quantificado pela auditoria offline
    (src/audit_interactions.py), com n ordens de magnitude maior que N_REPETICOES.
    """
    if n_shot == 0 or not sorteio_estocastico:
        return 1
    return N_REPETICOES











# Controle de limite de requisições (rate limit)
MAX_RETRIES = 5
INITIAL_BACKOFF = 10

# Concorrência máxima para chamadas async à API (ajustar conforme tier do OpenAI)
# Tier 1: ~8, Tier 2+: 15-20
MAX_CONCURRENCY = 10

# Máximo de retentativas para respostas malformadas do LLM
MAX_FORMAT_RETRIES = 5

# A geometria dos problemas (PROBLEM_*_CENTERS / PROBLEM_*_STD) e os geradores
# create_problem_* foram extraídos para data_problems.py (importados no topo).
# Ver tests/test_data_problems.py.



# Hiperparâmetros do Perceptron Estruturado usados em TODAS as estimações de W do
# protocolo (Coelho, Borges & Fonseca Neto, CILAMCE 2017, p. 16: η=0.001, C∈[0.1, 1]).
# Ponto ÚNICO de configuração — mudar aqui alcança os 8 call sites de uma vez.
# A sensibilidade a eta/C/delta_gamma é explorada à parte em
# print_hyperparameter_sensitivity, que varia os valores localmente de propósito.
PERCEPTRON_PARAMS = {
    "eta": 0.001,
    "C": 1.0,
    "delta_gamma": 0.05,
    "max_epochs": 50,
    "tol": 1e-4,
}

# Coletor global para diagnóstico da busca binária em γ no Perceptron Estruturado
# (item b da reunião 30/04/2026, ~520s — orientador pediu para verificar se gamma
# converge crescentemente). Preenchido nas chamadas estratégicas a
# train_relaxed_perceptron(..., return_history=True) e plotado em
# plot_gamma_convergence -> final_10_gamma_convergence.png.
PERCEPTRON_GAMMA_DIAGNOSTICS: List[dict] = []

# Nomes de classe distintos para testar viés semântico do LLM (conjunto reduzido)
NOMES_CLASSES = [
    ("A", "B"),
    ("0", "1"),
    ("Positivo", "Negativo"),
    ("Azul", "Vermelho"),
]

# Nomes de classes com ordem invertida para testar viés de posição
NOMES_CLASSES_INVERTIDAS = [
    ("B", "A"),
    ("1", "0"),
    ("Negativo", "Positivo"),
    ("Vermelho", "Azul"),
]

# Nomes semânticos para features — testar viés semântico nas variáveis
NOMES_FEATURES = [
    ("x1", "x2"),                    # Original (neutro)
    ("altura", "peso"),              # Semântico
    ("feature_1", "feature_2"),      # Técnico
]

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DE EXPERTS MÚLTIPLOS (Fase E)
# ═══════════════════════════════════════════════════════════════════════════

EXPERT_CONFIGS = [
    {"name": "aniso_x2",  "w": np.array([0.3, 1.5]),  "desc": "x2 dominante (original)"},
    {"name": "aniso_x1",  "w": np.array([1.5, 0.3]),  "desc": "x1 dominante (invertido)"},
    {"name": "euclidean", "w": np.array([1.0, 1.0]),  "desc": "pesos iguais (Euclidiana)"},
]

# Configurações de problemas anisotrópicos para validação do oráculo
# W verdadeiro = [1/σ_x1², 1/σ_x2²] (Bayes ótimo para Gaussianas diagonais)
ORACLE_ANISO_CONFIGS = [
    {"name": "x2_dom", "centers": [(-2.0, 0.5), (2.0, -0.5)], "std": [0.5, 2.0],
     "desc": "x1 preciso, x2 ruidoso"},
    {"name": "x1_dom", "centers": [(-0.5, 2.0), (0.5, -2.0)], "std": [2.0, 0.5],
     "desc": "x1 ruidoso, x2 preciso"},
    {"name": "forte_aniso", "centers": [(-1.5, 1.0), (1.5, -1.0)], "std": [0.3, 1.5],
     "desc": "anisotropia forte"},
]

# ═══════════════════════════════════════════════════════════════════════════
# FLAGS DE EXECUÇÃO — controla quais experimentos rodar
# ═══════════════════════════════════════════════════════════════════════════

RUN_PHASES_ABC = True              # Bloco 1: Fases A (treino W), B/C (transferência)
RUN_PHASE_E = True                 # Bloco 2: Fase E — LLM como aprendiz (in-context learning)
RUN_CLASS_ORDER_BIAS = True        # Teste de inversão de ordem das classes
RUN_FEATURE_NAMES = True           # Teste de nomes semânticos nas features
RUN_DILUTION = True                # Experimento de diluição
RUN_R3R4_EXPERIMENT = True           # Augmentação R3 (x1·x2) e R4 (x1², x2²) p/ detectar não-linearidade implícita
RUN_MULTIPLE_EXPERTS = True        # Múltiplas configs de expert na Fase E
RUN_ALGORITHM_COMPARISON = True    # Comparação Perceptron × NNLS (robustez ao método)
RUN_ORACLE_VALIDATION = True       # Validação: algoritmos recuperam W conhecido?
RUN_EXAMPLE_ORDER_BIAS = True      # Teste de viés de ordem dos exemplos few-shot
RUN_PROMPT_VARIANTS = True         # Teste de múltiplas variantes de prompt
RUN_CLASSICAL_BASELINES = True     # Comparação com baselines clássicos (k-NN, LR, SVM)
RUN_PROBLEM_MEIALUA = True               # Meia-lua: Problema D (Bloco 1) e Problema F (Bloco 2) — não-linearidade explícita p/ R3/R4
RUN_HOMEM_MULHER = True            # Estudo de caso real: peso × altura (homem/mulher), elipse
RUN_HM_CLASS_NAMES_AB = True       # Item 17 (reunião 20/05): repetir peso×altura com classes "A"/"B" (sem prior semântico)

# Item 17 (reunião 20/05 + e-mail): nomes de classe testados no peso×altura.
# Isola o prior semântico nos NOMES DAS CLASSES — "Homem"/"Mulher" carregam
# significado; "A"/"B" são neutros. Complementa o teste de nomes de FEATURE
# (x1/x2 vs peso/altura). Cada par extra multiplica o custo de API do bloco.
HM_CLASS_NAME_VARIANTS = [
    ("Homem", "Mulher", "homem_mulher"),      # semântico (original)
    ("A", "B", "homem_mulher_classesAB"),     # neutro (item 17)
]

# Estratégias de ordenação dos exemplos few-shot
EXAMPLE_ORDERINGS = ["class0_first", "class1_first", "shuffled", "alternating"]

# Variantes de prompt para teste de sensibilidade
PROMPT_VARIANTS = {
    "default": {
        "system_message": "You are a classifier. Respond only with the class label.",
        "max_tokens": 50,
        "description": "Template padrão (baseline)",
    },
    "geometric": {
        "system_message": "You are a geometric classifier analyzing point positions in metric space. Respond only with the class label.",
        "max_tokens": 50,
        "description": "Contexto geométrico/espacial explícito",
    },
    "cot": {
        "system_message": "You are a classifier. Think step by step, then give your final answer on the last line as just the class label.",
        "max_tokens": 300,
        "description": "Chain-of-thought antes da resposta",
    },
    "tabular": {
        "system_message": "You are a classifier. Respond only with the class label.",
        "max_tokens": 50,
        "description": "Formato tabular para coordenadas",
    },
}

# =============================================================================
# ESTRUTURAS DE DADOS — extraídas para `resultados.py` (importadas no topo).
# `ConsistencyMetrics` e `compute_consistency_metrics` vivem em `metrics.py`.
# =============================================================================


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================





# compute_consistency_metrics → metrics.py (importado no topo).


# =============================================================================
# GERAÇÃO DE CONJUNTOS DE DADOS
# =============================================================================

# create_problem_a/b/c/e_expert/d_meialua/homem_mulher e create_anisotropic_problem
# → data_problems.py (importados no topo). Ver tests/test_data_problems.py.








# =============================================================================
# INTERAÇÃO COM O LLM (COM RETENTATIVA DE FORMATO)
# =============================================================================

def build_prompt_zero_shot(x1: float, x2: float,
                           nome_classe_0: str, nome_classe_1: str,
                           nome_feature_0: str = "x1", nome_feature_1: str = "x2",
                           extra_features: Optional[List[Tuple[str, float]]] = None) -> str:
    """Constrói o prompt ZERO-SHOT (sem exemplos fornecidos ao LLM).

    Args:
        extra_features: Lista de (nome, valor) para features adicionais.
    """
    n_dim = 2 + (len(extra_features) if extra_features else 0)
    dim_label = f"{n_dim}D"

    features_text = f"{nome_feature_0} = {x1:.4f}\n{nome_feature_1} = {x2:.4f}"
    if extra_features:
        for fname, fval in extra_features:
            features_text += f"\n{fname} = {fval:.4f}"

    return f"""You are a binary classifier for {dim_label} points.

Classify the given point as Class {nome_classe_0} or Class {nome_classe_1}.
Answer ONLY with "{nome_classe_0}" or "{nome_classe_1}", nothing else.

Point to classify:
{features_text}

Your classification:"""


def build_prompt_few_shot(x1: float, x2: float,
                          examples: List,
                          nome_classe_0: str, nome_classe_1: str,
                          nome_feature_0: str = "x1", nome_feature_1: str = "x2",
                          extra_features: Optional[List[Tuple[str, float]]] = None) -> str:
    """Constrói o prompt FEW-SHOT com exemplos rotulados (origem varia por fase).

    Origem dos rótulos depende do chamador:
      - Fases B/C: rotulados pela métrica Ŵ_LLM (ancoragem em H2).
      - Fase E: rotulados pelo perito externo (W_expert ou ground truth da meia-lua).
      - Pipeline externo: rotulados pela melhor métrica identificada em Fase A 2/3/4 feat.
    Examples pode conter tuplas de 3 (x1, x2, label) ou 4+ elementos para features extras.
    """
    n_dim = 2 + (len(extra_features) if extra_features else 0)
    dim_label = f"{n_dim}D"

    examples_lines = []
    for ex in examples:
        line = f"  {nome_feature_0} = {ex[0]:.4f}, {nome_feature_1} = {ex[1]:.4f}"
        if len(ex) > 3:
            # Extra features no exemplo
            for i in range(3, len(ex) - 1):
                line += f", x{i} = {ex[i]:.4f}"
            line += f" -> {ex[-1]}"
        else:
            line += f" -> {ex[2]}"
        examples_lines.append(line)
    examples_text = "\n".join(examples_lines)

    features_text = f"{nome_feature_0} = {x1:.4f}\n{nome_feature_1} = {x2:.4f}"
    if extra_features:
        for fname, fval in extra_features:
            features_text += f"\n{fname} = {fval:.4f}"

    return f"""You are a binary classifier for {dim_label} points.

Learn the classification pattern from the examples below, then classify the new point.

Examples:
{examples_text}

Answer ONLY with "{nome_classe_0}" or "{nome_classe_1}", nothing else.

Point to classify:
{features_text}

Your classification:"""


def build_prompt_zero_shot_variant(
    variant: str, x1: float, x2: float,
    nome_classe_0: str, nome_classe_1: str,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None
) -> str:
    """Constrói prompt zero-shot usando a variante especificada."""
    if variant == "default":
        return build_prompt_zero_shot(x1, x2, nome_classe_0, nome_classe_1,
                                      nome_feature_0, nome_feature_1, extra_features)

    n_dim = 2 + (len(extra_features) if extra_features else 0)
    dim_label = f"{n_dim}D"

    features_text = f"{nome_feature_0} = {x1:.4f}\n{nome_feature_1} = {x2:.4f}"
    if extra_features:
        for fname, fval in extra_features:
            features_text += f"\n{fname} = {fval:.4f}"

    if variant == "geometric":
        return f"""You are a binary classifier for {dim_label} points in Euclidean space.

Points exist in a {dim_label} metric space. Classify the given point based on spatial proximity to cluster centers.

Classify as Class {nome_classe_0} or Class {nome_classe_1}.
Answer ONLY with "{nome_classe_0}" or "{nome_classe_1}", nothing else.

Point to classify:
{features_text}

Your classification:"""

    elif variant == "cot":
        return f"""You are a binary classifier for {dim_label} points.

Classify the given point as Class {nome_classe_0} or Class {nome_classe_1}.

Point to classify:
{features_text}

Think step by step about which class this point belongs to, then state your final classification ("{nome_classe_0}" or "{nome_classe_1}") on the last line."""

    elif variant == "tabular":
        table_rows = f"| {nome_feature_0} | {x1:.4f} |\n| {nome_feature_1} | {x2:.4f} |"
        if extra_features:
            for fname, fval in extra_features:
                table_rows += f"\n| {fname} | {fval:.4f} |"
        return f"""You are a binary classifier for {dim_label} points.

Classify the given point as Class {nome_classe_0} or Class {nome_classe_1}.
Answer ONLY with "{nome_classe_0}" or "{nome_classe_1}", nothing else.

Point to classify:
| Feature | Value |
|---------|-------|
{table_rows}

Your classification:"""

    else:
        raise ValueError(f"Variante de prompt desconhecida: {variant}")


def build_prompt_few_shot_variant(
    variant: str, x1: float, x2: float,
    examples: List,
    nome_classe_0: str, nome_classe_1: str,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None
) -> str:
    """Constrói prompt few-shot usando a variante especificada."""
    if variant == "default":
        return build_prompt_few_shot(x1, x2, examples, nome_classe_0, nome_classe_1,
                                      nome_feature_0, nome_feature_1, extra_features)

    n_dim = 2 + (len(extra_features) if extra_features else 0)
    dim_label = f"{n_dim}D"

    # Monta linhas de exemplos
    examples_lines = []
    for ex in examples:
        line = f"  {nome_feature_0} = {ex[0]:.4f}, {nome_feature_1} = {ex[1]:.4f}"
        if len(ex) > 3:
            for i in range(3, len(ex) - 1):
                line += f", x{i} = {ex[i]:.4f}"
            line += f" -> {ex[-1]}"
        else:
            line += f" -> {ex[2]}"
        examples_lines.append(line)
    examples_text = "\n".join(examples_lines)

    features_text = f"{nome_feature_0} = {x1:.4f}\n{nome_feature_1} = {x2:.4f}"
    if extra_features:
        for fname, fval in extra_features:
            features_text += f"\n{fname} = {fval:.4f}"

    if variant == "geometric":
        return f"""You are a binary classifier for {dim_label} points in Euclidean space.

Points exist in a {dim_label} metric space. Learn the classification pattern from the examples below based on spatial proximity, then classify the new point.

Examples:
{examples_text}

Answer ONLY with "{nome_classe_0}" or "{nome_classe_1}", nothing else.

Point to classify:
{features_text}

Your classification:"""

    elif variant == "cot":
        return f"""You are a binary classifier for {dim_label} points.

Learn the classification pattern from the examples below, then classify the new point.

Examples:
{examples_text}

Point to classify:
{features_text}

Think step by step about which class this point belongs to based on the examples, then state your final classification ("{nome_classe_0}" or "{nome_classe_1}") on the last line."""

    elif variant == "tabular":
        # Exemplos em formato tabular
        table_examples = []
        for ex in examples:
            row = f"| {ex[0]:.4f} | {ex[1]:.4f}"
            if len(ex) > 3:
                for i in range(3, len(ex) - 1):
                    row += f" | {ex[i]:.4f}"
                row += f" | {ex[-1]} |"
            else:
                row += f" | {ex[2]} |"
            table_examples.append(row)

        header = f"| {nome_feature_0} | {nome_feature_1}"
        separator = "|---------|---------|"
        if extra_features:
            for fname, _ in extra_features:
                header += f" | {fname}"
                separator += "---------|"
        header += " | Class |"
        separator += "-------|"

        table_text = header + "\n" + separator + "\n" + "\n".join(table_examples)

        point_row = f"| {nome_feature_0} | {x1:.4f} |\n| {nome_feature_1} | {x2:.4f} |"
        if extra_features:
            for fname, fval in extra_features:
                point_row += f"\n| {fname} | {fval:.4f} |"

        return f"""You are a binary classifier for {dim_label} points.

Learn the classification pattern from the examples below, then classify the new point.

Examples:
{table_text}

Answer ONLY with "{nome_classe_0}" or "{nome_classe_1}", nothing else.

Point to classify:
| Feature | Value |
|---------|-------|
{point_row}

Your classification:"""

    else:
        raise ValueError(f"Variante de prompt desconhecida: {variant}")


def llm_classify_point_anthropic(client: anthropic.Anthropic, model_name: str,
                                  prompt: str, temperature: float,
                                  system_message: str = "You are a classifier. Respond only with the class label.",
                                  max_tokens: int = 50) -> str:
    """Classifica um ponto usando a API da Anthropic (Claude)."""
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # Guarda contra resposta sem blocos de conteúdo (ou texto None): devolve
    # string vazia (tratada como malformada, com retry) em vez de estourar.
    if not getattr(response, "content", None):
        return ""
    return (getattr(response.content[0], "text", None) or "").strip()


async def async_llm_classify_point_openai(async_client, model_name: str, prompt: str,
                                           temperature: float,
                                           system_message: str = "You are a classifier. Respond only with the class label.") -> Tuple[str, dict]:
    """Chama a API compatível com OpenAI (OpenAI, Gemini ou OpenRouter) de forma assíncrona.

    Envolve a chamada em asyncio.wait_for(timeout=90s) como segunda barreira contra
    travamentos: se a requisição não responder em 90s mesmo após retries do SDK,
    levanta TimeoutError em vez de bloquear o semaphore indefinidamente.

    Retorna (texto, meta): meta traz o modelo RESOLVIDO pela API (``response.model``,
    o snapshot datado — reprodutibilidade) e, no OpenRouter, o provedor de inferência
    que serviu a requisição (``response.provider``).
    """
    async def _do_call():
        response = await async_client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            extra_body=get_extra_body(CURRENT_PROVIDER, model_name),
        )
        meta = {
            "model_resolved": getattr(response, "model", None),
            "inference_provider": getattr(response, "provider", None),
        }
        # OpenRouter pode devolver HTTP 200 com choices vazio/None (erro embutido
        # no corpo) — devolve string vazia (tratada como malformada, com retry)
        # em vez de estourar 'NoneType' object is not subscriptable.
        if not getattr(response, "choices", None):
            return "", meta
        content = response.choices[0].message.content
        return (content or "").strip(), meta

    try:
        return await asyncio.wait_for(_do_call(), timeout=90.0)
    except asyncio.TimeoutError:
        # Devolve string vazia — será tratada como malformada e cai no fallback MD5
        return "", {}


# `parse_llm_response` foi extraído para o módulo `llm_parser.py` (importado no topo).
# A lógica é idêntica; isolá-la permite testes de unidade independentes (tests/test_parser.py).


# ── Freio GLOBAL de rate limit ────────────────────────────────────────────────
# O backoff por chamada não basta: com MAX_CONCURRENCY workers, o que recebe 429
# dorme, mas os demais continuam disparando e realimentam o rate limit do
# provedor pinado (allow_fallbacks=False impede o OpenRouter de desviar — por
# design, para reprodutibilidade). O freio faz TODOS os workers segurarem as
# próximas chamadas até a janela de cooldown passar. Timestamps em
# time.monotonic() (independe do event loop; testável fora de asyncio).
_RATE_LIMIT_GATE = {"until": 0.0}
# Sinais de CAPACIDADE do provedor acionam o freio global; timeouts/5xx avulsos
# continuam sendo tratados só pelo backoff da própria chamada.
_TOKENS_CAPACIDADE = ("429", "rate_limit", "overloaded", "temporarily")


def _acionar_rate_limit_global(segundos: float) -> None:
    """Estende a janela global de cooldown (nunca a encurta)."""
    ate = time.monotonic() + segundos
    if ate > _RATE_LIMIT_GATE["until"]:
        _RATE_LIMIT_GATE["until"] = ate


async def _respeitar_rate_limit_global() -> None:
    """Aguarda a janela global de cooldown antes de emitir uma chamada.

    Loop (e não um sleep único) porque outro worker pode ESTENDER a janela
    enquanto este dorme.
    """
    while True:
        falta = _RATE_LIMIT_GATE["until"] - time.monotonic()
        if falta <= 0:
            return
        await asyncio.sleep(falta)


async def async_llm_classify_point(
    x1: float, x2: float,
    nome_classe_0: str, nome_classe_1: str,
    examples: Optional[List] = None,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None,
    prompt_variant: str = "default"
) -> Tuple[str, str, bool]:
    """Classifica um ponto via LLM (retries + fallback MD5). Retorna (label_parsed, raw_response, was_malformed)."""
    # Seleciona builder de prompt conforme a variante
    if examples is None or len(examples) == 0:
        prompt = build_prompt_zero_shot_variant(prompt_variant, x1, x2, nome_classe_0, nome_classe_1,
                                                nome_feature_0, nome_feature_1, extra_features)
    else:
        prompt = build_prompt_few_shot_variant(prompt_variant, x1, x2, examples, nome_classe_0, nome_classe_1,
                                               nome_feature_0, nome_feature_1, extra_features)

    # Busca system_message e max_tokens da variante
    variant_config = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["default"])
    system_msg = variant_config["system_message"]
    max_tok = variant_config["max_tokens"]

    all_responses = []
    all_prompts = []
    all_metas = []

    for format_attempt in range(MAX_FORMAT_RETRIES):
        for rate_attempt in range(MAX_RETRIES):
            await _respeitar_rate_limit_global()
            try:
                config = PROVIDER_CONFIG[CURRENT_PROVIDER]

                if config["client_type"] == "anthropic":
                    # Anthropic não tem async_client neste script; fallback sync em thread
                    label = await asyncio.get_event_loop().run_in_executor(
                        None, llm_classify_point_anthropic, client, MODEL_NAME, prompt,
                        CURRENT_TEMPERATURE, system_msg, max_tok
                    )
                    call_meta = {}
                else:
                    label, call_meta = await async_llm_classify_point_openai(async_client, MODEL_NAME, prompt,
                                                                             CURRENT_TEMPERATURE, system_msg)

                break
            except Exception as e:
                error_str = str(e).lower()
                # Retentáveis: rate limit, 400 transitório do endpoint compatível,
                # E transientes de infraestrutura (5xx, timeout, conexão) —
                # relevantes com provedores via OpenRouter.
                retentavel = any(tok in error_str for tok in (
                    "429", "rate_limit", "overloaded", "400", "could not parse",
                    "500", "502", "503", "timeout", "connection", "temporarily",
                ))
                if retentavel:
                    # Jitter de ±25%: sem ele, todos os workers que receberam 429
                    # no mesmo instante acordam juntos e re-disparam em rajada,
                    # re-acionando o rate limit. Só afeta TIMING de retry, nunca
                    # os dados — não compromete a reprodutibilidade do experimento.
                    wait_time = INITIAL_BACKOFF * (2 ** rate_attempt) * (0.75 + 0.5 * random.random())
                    if any(tok in error_str for tok in _TOKENS_CAPACIDADE):
                        # Sinal de capacidade do provedor: TODOS os workers seguram
                        # as próximas chamadas até a janela passar (freio global).
                        _acionar_rate_limit_global(wait_time)
                    print(f"  ⏳ Erro retentável ({str(e)[:80]}). Aguardando {wait_time:.0f}s (tentativa {rate_attempt + 1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        else:
            raise Exception(f"Máximo de tentativas ({MAX_RETRIES}) excedido por limite de requisições")

        all_responses.append(label)
        all_prompts.append(prompt)
        all_metas.append(call_meta)
        parsed_label, is_valid = parse_llm_response(label, nome_classe_0, nome_classe_1)

        if is_valid:
            raw = all_responses[0] if all_responses else ""
            LLM_INTERACTIONS.append({
                "point": {"x1": x1, "x2": x2},
                "prompt": all_prompts[0],
                "raw_response": raw,
                "parsed_label": parsed_label,
                "model": MODEL_NAME,
                "model_resolved": call_meta.get("model_resolved"),
                "inference_provider": call_meta.get("inference_provider"),
                "provider": CURRENT_PROVIDER,
                "temperature": CURRENT_TEMPERATURE,
                "format_retries": format_attempt,
                "malformed": False,
            })
            return parsed_label, raw, False

        if format_attempt < MAX_FORMAT_RETRIES - 1:
            if examples is None or len(examples) == 0:
                prompt = f"""You are a binary classifier for 2D points.

Classify the given point as EXACTLY one of these two classes: "{nome_classe_0}" or "{nome_classe_1}".

IMPORTANT: Your response must be EXACTLY "{nome_classe_0}" or "{nome_classe_1}" with no other text.

Point to classify:
x1 = {x1:.4f}
x2 = {x2:.4f}

Your classification (respond with ONLY the class name):"""
            else:
                examples_text = "\n".join([
                    f"  x1 = {ex[0]:.4f}, x2 = {ex[1]:.4f} -> {ex[2]}"
                    for ex in examples
                ])
                prompt = f"""You are a binary classifier for 2D points.

Learn the classification pattern from the examples below, then classify the new point.

Examples:
{examples_text}

IMPORTANT: Your response must be EXACTLY "{nome_classe_0}" or "{nome_classe_1}" with no other text.

Point to classify:
x1 = {x1:.4f}
x2 = {x2:.4f}

Your classification (respond with ONLY the class name):"""

    # Fallback: classe aleatória uniforme via hash (sem viés geométrico, reproduzível)
    import hashlib
    hash_seed = hashlib.md5(f"{x1:.6f}_{x2:.6f}".encode()).hexdigest()
    fallback_bit = int(hash_seed, 16) % 2
    fallback = nome_classe_0 if fallback_bit == 0 else nome_classe_1
    raw = all_responses[0] if all_responses else ""

    first_meta = all_metas[0] if all_metas else {}
    LLM_INTERACTIONS.append({
        "point": {"x1": x1, "x2": x2},
        "prompt": all_prompts[0] if all_prompts else "",
        "raw_response": raw,
        "all_responses": all_responses,
        "parsed_label": fallback,
        "model": MODEL_NAME,
        "model_resolved": first_meta.get("model_resolved"),
        "inference_provider": first_meta.get("inference_provider"),
        "provider": CURRENT_PROVIDER,
        "temperature": CURRENT_TEMPERATURE,
        "format_retries": MAX_FORMAT_RETRIES,
        "malformed": True,
    })

    warnings.warn(
        f"\n  ⚠️ RESPOSTA MALFORMADA após {MAX_FORMAT_RETRIES} tentativas para o ponto ({x1:.4f}, {x2:.4f}).\n"
        f"     Respostas recebidas: {all_responses}\n"
        f"     Esperado: '{nome_classe_0}' ou '{nome_classe_1}'\n"
        f"     Fallback aleatório (hash-based): '{fallback}'\n"
    )

    return fallback, raw, True


async def async_collect_llm_decisions(
    X: np.ndarray,
    nome_classe_0: str,
    nome_classe_1: str,
    examples: Optional[List] = None,
    verbose: bool = True,
    label_prefix: str = "",
    nome_feature_0: str = "x1",
    nome_feature_1: str = "x2",
    extra_features_matrix: Optional[np.ndarray] = None,
    extra_feature_names: Optional[List[str]] = None,
    prompt_variant: str = "default"
) -> Tuple[np.ndarray, int]:
    """Versão assíncrona de collect_llm_decisions com chamadas concorrentes à API."""
    n_examples = 0 if examples is None else len(examples)
    shot_type = "zero-shot" if n_examples == 0 else f"{n_examples}-shot"
    n_total = len(X)

    if verbose:
        print(f"  {label_prefix}Coletando decisões do LLM ({shot_type}) — {n_total} pontos [async, concurrency={MAX_CONCURRENCY}]...")
        sys.stdout.flush()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    results = [None] * n_total  # (label, raw, was_malformed)
    completed = [0]
    malformed_count = [0]
    t_start = time.time()
    progress_lock = asyncio.Lock()
    progress_step = 10 if n_total > 30 else 1

    async def classify_one(i, x1, x2):
        extra_feats = None
        if extra_features_matrix is not None and extra_feature_names is not None:
            extra_feats = list(zip(extra_feature_names, extra_features_matrix[i]))

        try:
            async with semaphore:
                label, raw, was_malformed = await async_llm_classify_point(
                    x1, x2, nome_classe_0, nome_classe_1,
                    examples, nome_feature_0, nome_feature_1, extra_feats,
                    prompt_variant=prompt_variant
                )
        except Exception as exc:
            # Falha DEFINITIVA (esgotou retries ou erro não-retentável): não
            # derruba a execução inteira (gather é fail-fast) — registra como
            # malformada com o fallback MD5 e segue; a auditoria offline acusa.
            import hashlib
            hs = hashlib.md5(f"{x1:.6f}_{x2:.6f}".encode()).hexdigest()
            label = nome_classe_0 if int(hs, 16) % 2 == 0 else nome_classe_1
            raw, was_malformed = "", True
            LLM_INTERACTIONS.append({
                "point": {"x1": x1, "x2": x2},
                "prompt": "",
                "raw_response": "",
                "parsed_label": label,
                "model": MODEL_NAME,
                "model_resolved": None,
                "inference_provider": None,
                "provider": CURRENT_PROVIDER,
                "temperature": CURRENT_TEMPERATURE,
                "format_retries": MAX_FORMAT_RETRIES,
                "malformed": True,
                "error": str(exc)[:300],
            })
            print(f"    ⚠ Falha definitiva no ponto ({x1:.3f}, {x2:.3f}): "
                  f"{str(exc)[:120]} — fallback aplicado, execução continua.")

        results[i] = (label, raw, was_malformed)
        if was_malformed:
            malformed_count[0] += 1

        if verbose:
            async with progress_lock:
                completed[0] += 1
                done = completed[0]
                if done % progress_step == 0 or done == n_total:
                    elapsed = time.time() - t_start
                    avg = elapsed / done
                    remaining = avg * (n_total - done)
                    eta_str = f"{remaining:.0f}s" if remaining >= 1 else "<1s"
                    w = len(str(n_total))
                    print(
                        f"    [{done:>{w}}/{n_total}] "
                        f"{elapsed:.1f}s | ETA: {eta_str}"
                    )
                    sys.stdout.flush()

    tasks = [classify_one(i, x1, x2) for i, (x1, x2) in enumerate(X)]
    await asyncio.gather(*tasks)

    labels = [results[i][0] for i in range(n_total)]
    y_llm = np.array([0 if l == nome_classe_0 else 1 for l in labels])

    if verbose:
        total_elapsed = time.time() - t_start
        n_0 = np.sum(y_llm == 0)
        n_1 = np.sum(y_llm == 1)
        print(f"    ✓ Concluído em {total_elapsed:.1f}s — "
              f"Classe {nome_classe_0}: {n_0} ({100*n_0/n_total:.1f}%) | "
              f"Classe {nome_classe_1}: {n_1} ({100*n_1/n_total:.1f}%)")
        if malformed_count[0] > 0:
            print(f"    ⚠️ Respostas malformadas (fallback usado): {malformed_count[0]}/{n_total}")
        sys.stdout.flush()

    return y_llm, malformed_count[0]


def collect_llm_decisions(
    X: np.ndarray,
    nome_classe_0: str,
    nome_classe_1: str,
    examples: Optional[List] = None,
    verbose: bool = True,
    label_prefix: str = "",
    nome_feature_0: str = "x1",
    nome_feature_1: str = "x2",
    extra_features_matrix: Optional[np.ndarray] = None,
    extra_feature_names: Optional[List[str]] = None,
    prompt_variant: str = "default"
) -> Tuple[np.ndarray, int]:
    """Coleta as decisões do LLM para todos os pontos do conjunto de dados.

    Wrapper síncrono que delega para async_collect_llm_decisions com chamadas concorrentes.
    """
    return asyncio.run(async_collect_llm_decisions(
        X, nome_classe_0, nome_classe_1,
        examples=examples, verbose=verbose, label_prefix=label_prefix,
        nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
        extra_features_matrix=extra_features_matrix,
        extra_feature_names=extra_feature_names,
        prompt_variant=prompt_variant
    ))


# =============================================================================
# APRENDIZADO DE MÉTRICA (OTIMIZAÇÃO INVERSA: PERCEPTRON ESTRUTURADO + NNLS)
# Inclui também augment_to_r3/r4 e augment_features p/ R2/R3/R4
# =============================================================================

# d_W e compute_centroids → metrics.py (importados no topo).


# train_relaxed_perceptron → relaxed_perceptron.py e train_least_squares_inverse →
# least_squares_inverse.py (funções de conveniência junto de suas classes; importadas no topo).


# augment_to_r3, augment_to_r4, augment_features, predict_with_metric e
# compute_metric_confidence → metrics.py (importados no topo). Ver tests/test_metrics.py.


# =============================================================================
# FASE A: APRENDIZADO DA MÉTRICA NO PROBLEMA A (APENAS ZERO-SHOT)
# =============================================================================

def phase_a_learn_metric(
    X_train_a: np.ndarray,
    y_true_train_a: np.ndarray,
    nome_classe_0: str,
    nome_classe_1: str,
    verbose: bool = True,
    prompt_variant: str = "default"
) -> Tuple[LearnedMetric, LearnedMetric, np.ndarray, float, float, float, int]:
    """FASE A: Aprende a métrica W a partir das decisões do LLM no Problema A.

    Retorna:
        learned_metric: LearnedMetric via Perceptron Estruturado (algoritmo principal)
        learned_metric_nnls: LearnedMetric via NNLS (validação cruzada do método)
        y_llm_train_a: decisões do LLM no Problema A
        fidelity: fidelidade do Perceptron (métrica vs. LLM)
        fidelity_nnls: fidelidade do NNLS (métrica vs. LLM)
        llm_accuracy: acurácia do LLM vs. ground truth
        n_malformed: número de respostas malformadas
    """
    variant_label = f" [prompt: {prompt_variant}]" if prompt_variant != "default" else ""
    if verbose:
        print("\n  ══════════════════════════════════════════════════════════════")
        print(f"  FASE A: ESTIMANDO A MÉTRICA W DO PROBLEMA A (ZERO-SHOT){variant_label}")
        print("  ══════════════════════════════════════════════════════════════")
        print("  Objetivo: capturar o processo decisório INTRÍNSECO do LLM.")
        print("  O LLM classifica pontos sem exemplos (zero-shot), e a métrica")
        print("  Ŵ_LLM é inferida a partir dessas decisões via otimização inversa.")

    if verbose:
        print(f"\n  Passo 1: Coletando decisões do LLM no Problema A (zero-shot)...")
        print(f"    Classes usadas: '{nome_classe_0}' e '{nome_classe_1}'")
        print(f"    Pontos de treino: {len(X_train_a)}")

    y_llm_train_a, n_malformed = collect_llm_decisions(
        X_train_a, nome_classe_0, nome_classe_1,
        examples=None, verbose=verbose,
        label_prefix=f"[Problema A{variant_label}] ",
        prompt_variant=prompt_variant
    )

    if len(np.unique(y_llm_train_a)) < 2:
        print("  ⚠ AVISO: O LLM classificou tudo como uma única classe! Verifique o modelo e os prompts.")
        return None, None, y_llm_train_a, 0.5, 0.5, accuracy_score(y_true_train_a, y_llm_train_a), n_malformed

    centroids_a = compute_centroids(X_train_a, y_llm_train_a)

    if verbose:
        print(f"\n  Passo 2: Calculando centróides a partir das decisões do LLM...")
        print(f"    Centróide da Classe 0: ({centroids_a[0, 0]:.3f}, {centroids_a[0, 1]:.3f})")
        print(f"    Centróide da Classe 1: ({centroids_a[1, 0]:.3f}, {centroids_a[1, 1]:.3f})")

    # --- Algoritmo 1: Perceptron Estruturado com Relaxação de Margem ---
    if verbose:
        print("\n  Passo 3a: Estimando a métrica W via Perceptron Estruturado com Relaxação de Margem...")
        print("    (Busca binária no gamma ótimo para maximizar a margem da fronteira de decisão)")

    w_learned, gamma_optimal, _gamma_hist = train_relaxed_perceptron(
        X_train_a, y_llm_train_a, centroids_a,
        **PERCEPTRON_PARAMS,
        verbose=verbose,
        use_best_effort=True,  # retorna melhor W parcial quando separação perfeita é impossível
        return_history=True,   # captura evolução de γ para diagnóstico (item b reunião 30/04)
    )
    # Anota diagnóstico de γ para esta chamada (Fase A no Problema A);
    # "model" permite filtrar o final_10 por modelo na seção de visualizações.
    PERCEPTRON_GAMMA_DIAGNOSTICS.append({
        "label": "Fase A — Problema A",
        "model": MODEL_NAME,
        "gamma_final": float(gamma_optimal),
        "history": _gamma_hist,
    })

    if verbose:
        w_norm_val = np.linalg.norm(w_learned)
        w_unit = w_learned / w_norm_val if w_norm_val > 0 else w_learned
        print(f"    >>> [Perceptron] Ŵ_LLM estimado (bruto): [{w_learned[0]:.4f}, {w_learned[1]:.4f}]")
        print(f"    >>> [Perceptron] Ŵ_LLM estimado (unitário): [{w_unit[0]:.4f}, {w_unit[1]:.4f}]")
        w_ratio = w_learned[0] / w_learned[1] if w_learned[1] != 0 else float('inf')
        print(f"    >>> [Perceptron] Razão w1/w2: {w_ratio:.4f}  ← determina a geometria da fronteira")
        print(f"    >>> [Perceptron] Gamma ótimo encontrado: {gamma_optimal:.4f}")
        # Nota: a magnitude de W depende da escala do target_margin e gamma;
        # apenas a direção (razão w1/w2) é invariante e comparável entre seeds.

    # --- Algoritmo 2: Mínimos Quadrados (NNLS) — validação cruzada do método ---
    if verbose:
        print("\n  Passo 3b: Estimando a métrica W via Mínimos Quadrados (NNLS)...")
        print("    (Validação cruzada: se ambos convergem para W similar, resultado é robusto ao método)")

    w_nnls, _ = train_least_squares_inverse(
        X_train_a, y_llm_train_a, centroids_a, verbose=verbose
    )

    if verbose:
        w_nnls_norm = w_nnls / np.sum(w_nnls) if np.sum(w_nnls) > 0 else w_nnls
        print(f"    >>> [NNLS] Ŵ_LLM estimado (bruto): [{w_nnls[0]:.4f}, {w_nnls[1]:.4f}]")
        print(f"    >>> [NNLS] Ŵ_LLM estimado (normalizado): [{w_nnls_norm[0]:.3f}, {w_nnls_norm[1]:.3f}]")
        w_nnls_ratio = w_nnls[0] / w_nnls[1] if w_nnls[1] != 0 else float('inf')
        print(f"    >>> [NNLS] Razão w1/w2: {w_nnls_ratio:.4f}")
        # Comparação entre algoritmos
        cos_sim = np.dot(w_learned, w_nnls) / (np.linalg.norm(w_learned) * np.linalg.norm(w_nnls) + 1e-9)
        print(f"    >>> Similaridade cosseno (Perceptron vs NNLS): {cos_sim:.4f}")

    # --- Fidelidade dos 2 algoritmos ---
    y_pred_metric_a = predict_with_metric(X_train_a, centroids_a, w_learned)
    fidelity = accuracy_score(y_llm_train_a, y_pred_metric_a)

    y_pred_nnls_a = predict_with_metric(X_train_a, centroids_a, w_nnls)
    fidelity_nnls = accuracy_score(y_llm_train_a, y_pred_nnls_a)

    llm_accuracy = accuracy_score(y_true_train_a, y_llm_train_a)

    if verbose:
        print(f"\n  Passo 4: Verificando fidelidade das métricas estimadas...")
        print(f"    [Perceptron] Fidelidade (métrica vs. LLM no Problema A): {fidelity:.1%}")
        print(f"    [NNLS]       Fidelidade (métrica vs. LLM no Problema A): {fidelity_nnls:.1%}")
        print(f"    Acurácia do LLM vs. ground truth: {llm_accuracy:.1%}")
        fids = [fidelity, fidelity_nnls]
        if all(f >= 0.9 for f in fids):
            print(f"    Ambas as métricas estimadas reproduzem bem as decisões do LLM.")
        elif any(f >= 0.9 for f in fids):
            print(f"    ⚠ Fidelidade divergente entre algoritmos — investigar.")
        else:
            print(f"    ⚠ Fidelidade baixa em ambos — a métrica pode não representar bem o LLM.")

    learned_metric = LearnedMetric(
        w=w_learned, centroids=centroids_a,
        gamma=gamma_optimal, source_problem="Problem_A"
    )

    learned_metric_nnls = LearnedMetric(
        w=w_nnls, centroids=centroids_a,
        gamma=0.0, source_problem="Problem_A"
    )

    return learned_metric, learned_metric_nnls, y_llm_train_a, fidelity, fidelity_nnls, llm_accuracy, n_malformed


# =============================================================================
# FASES B/C: TESTE DE CONSISTÊNCIA EM NOVOS PROBLEMAS
# =============================================================================

def select_confident_examples(
    X: np.ndarray,
    centroids: np.ndarray,
    w: np.ndarray,
    n_examples: int,
    nome_classe_0: str,
    nome_classe_1: str,
    verbose: bool = True
) -> Tuple[List[Tuple[float, float, str]], np.ndarray, np.ndarray, np.ndarray]:
    """Seleciona exemplos few-shot por Aprendizado Ativo (maior margem por classe).

    Justificativa: exemplos com maior margem são os mais representativos de cada classe —
    estão longe da fronteira, são os casos mais claros e ancoramos bem o LLM ao padrão da métrica.

    ``n_examples`` é o TOTAL de exemplos, honrado exatamente mesmo quando ímpar:
    a classe 0 recebe o exemplo excedente (⌈n/2⌉ vs ⌊n/2⌋), determinístico.
    """
    confidences, y_pred_metric = compute_metric_confidence(X, centroids, w)

    n_per_class_0 = n_examples - n_examples // 2
    n_per_class_1 = n_examples // 2

    idx_class_0 = np.where(y_pred_metric == 0)[0]
    idx_class_1 = np.where(y_pred_metric == 1)[0]

    conf_class_0 = confidences[idx_class_0]
    conf_class_1 = confidences[idx_class_1]

    sorted_idx_0 = idx_class_0[np.argsort(-conf_class_0)]
    sorted_idx_1 = idx_class_1[np.argsort(-conf_class_1)]

    selected_idx_0 = sorted_idx_0[:n_per_class_0]
    selected_idx_1 = sorted_idx_1[:n_per_class_1]

    if verbose:
        if len(selected_idx_0) > 0:
            print(f"      Classe 0: {len(selected_idx_0)} pontos selecionados, "
                  f"faixa de margem [{confidences[selected_idx_0].min():.3f}, "
                  f"{confidences[selected_idx_0].max():.3f}]")
        if len(selected_idx_1) > 0:
            print(f"      Classe 1: {len(selected_idx_1)} pontos selecionados, "
                  f"faixa de margem [{confidences[selected_idx_1].min():.3f}, "
                  f"{confidences[selected_idx_1].max():.3f}]")

    examples = []
    for i in selected_idx_0:
        examples.append((X[i, 0], X[i, 1], nome_classe_0))
    for i in selected_idx_1:
        examples.append((X[i, 0], X[i, 1], nome_classe_1))

    selected_indices = np.concatenate([selected_idx_0, selected_idx_1])
    return examples, y_pred_metric, confidences, selected_indices


def phase_consistency_test(
    X: np.ndarray,
    y_true: np.ndarray,
    learned_metric: LearnedMetric,
    nome_classe_0: str,
    nome_classe_1: str,
    n_shot: int,
    problem_name: str,
    verbose: bool = True,
    prompt_variant: str = "default"
) -> Tuple[ConsistencyMetrics, np.ndarray, np.ndarray, float, float, int, float, Dict]:
    """Teste de consistência genérico em um novo problema usando a métrica estimada no Problema A.

    Apenas o vetor W é transferido do Problema A. Os centróides são recalculados localmente
    a partir dos rótulos que o LLM atribui neste problema, isolando o efeito dos pesos da
    métrica do efeito da posição dos centróides entre problemas com distribuições diferentes.

    Garante que os exemplos few-shot (quando usados) são excluídos do conjunto de avaliação,
    evitando vazamento de dados (data leakage) entre seleção de exemplos e avaliação.

    Quando n_shot > 0, também treina baselines clássicos (k-NN, LR, SVM) nos mesmos exemplos
    few-shot e avalia a consistência deles com a métrica, para comparação com o LLM.

    Retorna:
        consistency_metrics: métricas de consistência LLM vs. métrica
        y_llm: decisões do LLM no conjunto de teste
        y_metric_test: predições da métrica no conjunto de teste
        llm_accuracy: acurácia LLM vs. ground truth
        metric_accuracy: acurácia métrica vs. ground truth
        n_malformed: respostas malformadas do LLM
        consistency_euclidean: consistência da baseline Euclidiana
        baselines_consistency: dict {nome_clf: {accuracy_vs_expert, kappa_vs_expert, ...}} (vazio se zero-shot)
    """
    if verbose:
        print(f"\n  ══════════════════════════════════════════════════════════════")
        print(f"  {problem_name}: TESTE DE CONSISTÊNCIA")
        print(f"  ══════════════════════════════════════════════════════════════")
        print(f"  Aplicando a métrica W estimada no Problema A a este novo problema.")
        print(f"  Objetivo: verificar se o LLM concorda com a métrica em dados não vistos.")

    examples = None
    selected_indices = np.array([], dtype=int)

    if n_shot > 0:
        if verbose:
            print(f"\n  Selecionando {n_shot} exemplos few-shot (Aprendizado Ativo, rotulados pela métrica)...")

        examples, _, _, selected_indices = select_confident_examples(
            X, learned_metric.centroids, learned_metric.w,
            n_shot, nome_classe_0, nome_classe_1, verbose=verbose
        )

    # Correção de vazamento: exclui exemplos few-shot da avaliação.
    # No modo zero-shot, selected_indices é vazio e test_mask mantém todos os pontos.
    test_mask = np.ones(len(X), dtype=bool)
    test_mask[selected_indices] = False
    X_test = X[test_mask]
    y_true_test = y_true[test_mask]

    if verbose:
        print(f"\n  Coletando decisões do LLM para {len(X_test)} pontos de teste...")

    y_llm, n_malformed = collect_llm_decisions(
        X_test, nome_classe_0, nome_classe_1,
        examples=examples, verbose=verbose,
        label_prefix=f"[{problem_name}] ",
        prompt_variant=prompt_variant
    )

    # --- Centróides locais vs. transferidos ---
    # DECISÃO DE DESIGN: apenas W é transferido do Problema A; centróides são
    # recalculados localmente a partir dos rótulos do LLM neste problema.
    #
    # JUSTIFICATIVA: Os Problemas A, B e C têm distribuições com centróides
    # geométricos em posições DIFERENTES (por design). Transferir centróides do
    # Problema A mediria erro de localização (centróides errados), não inconsistência
    # de W. Recalcular centróides isola o efeito dos PESOS da métrica (o que queremos
    # testar) do efeito da POSIÇÃO dos centróides entre distribuições diferentes.
    #
    # TESTE ABLATIVO: também calculamos consistência com centróides transferidos
    # (do Problema A) para demonstrar que a diferença existe e justificar a decisão.
    unique_classes = np.unique(y_llm)
    if len(unique_classes) < 2:
        if verbose:
            print(f"    AVISO: LLM atribuiu todos os pontos à classe {unique_classes[0]}. "
                  f"Usando centróides do Problema A como fallback.")
        local_centroids = learned_metric.centroids
    else:
        local_centroids = compute_centroids(X_test, y_llm)

    # Predições da métrica usando W do Problema A + centróides locais (abordagem principal)
    confidences_test, y_metric_test = compute_metric_confidence(
        X_test, local_centroids, learned_metric.w
    )

    # Teste ablativo: predições com centróides TRANSFERIDOS do Problema A
    # Se consistência_transferida << consistência_local, confirma que recalcular é necessário.
    y_metric_transferred = predict_with_metric(
        X_test, learned_metric.centroids, learned_metric.w
    )
    consistency_transferred = compute_consistency_metrics(y_llm, y_metric_transferred)

    if verbose:
        n_0 = np.sum(y_metric_test == 0)
        n_1 = np.sum(y_metric_test == 1)
        print(f"    Predições da métrica (W_A + centróides locais): Classe 0={n_0}, Classe 1={n_1}")
        print(f"    [Ablativo] Consistência com centróides LOCAIS:       "
              f"{compute_consistency_metrics(y_llm, y_metric_test).accuracy:.1%}")
        print(f"    [Ablativo] Consistência com centróides TRANSFERIDOS: "
              f"{consistency_transferred.accuracy:.1%}")
        delta = compute_consistency_metrics(y_llm, y_metric_test).accuracy - consistency_transferred.accuracy
        if delta > 0.05:
            print(f"    → Centróides locais melhoram em {delta:.1%} — recálculo justificado.")
        elif delta > -0.05:
            print(f"    → Diferença marginal ({delta:+.1%}) — ambas abordagens comparáveis.")
        else:
            print(f"    → ⚠ Centróides transferidos superiores ({delta:+.1%}) — investigar.")

    consistency_metrics = compute_consistency_metrics(y_llm, y_metric_test)
    llm_accuracy = accuracy_score(y_true_test, y_llm)
    metric_accuracy = accuracy_score(y_true_test, y_metric_test)

    # Linha de base Euclidiana: pesos uniformes para verificar limitação da métrica diagonal.
    # Se a métrica estimada não superar a Euclidiana em >1%, sinalizamos limitação diagonal.
    w_euclidean = np.ones_like(learned_metric.w)
    y_pred_euclidean = predict_with_metric(X_test, local_centroids, w_euclidean)
    euclidean_metrics = compute_consistency_metrics(y_llm, y_pred_euclidean)
    consistency_euclidean = euclidean_metrics.accuracy

    # --- Baselines clássicos (quando few-shot: mesmos exemplos, mesma avaliação) ---
    # Responde: "A consistência do LLM é melhor que a de um classificador trivial
    # treinado nos mesmos N exemplos rotulados pela métrica?"
    baselines_consistency = {}
    if n_shot > 0 and len(selected_indices) >= 2:
        X_train_bl = X[selected_indices]
        # Rótulos dos exemplos few-shot: vêm da métrica (mesmos que o LLM recebeu)
        y_train_bl = predict_with_metric(X[selected_indices], learned_metric.centroids, learned_metric.w)

        if len(np.unique(y_train_bl)) >= 2:
            runner = ClassicalBaselineRunner(verbose=verbose)
            # Avalia consistência: baselines vs. métrica (y_metric_test)
            # e baselines vs. ground truth (y_true_test)
            baseline_results = runner.run(
                X_train_bl, y_train_bl,
                X_test, y_metric_test, y_true_test,
                n_shot=n_shot
            )
            for clf_name, metrics in baseline_results.items():
                baselines_consistency[clf_name] = metrics

            if verbose and baselines_consistency:
                print(f"\n  BASELINES CLÁSSICOS ({problem_name}, {n_shot}-shot):")
                print(f"    {'Classificador':<25} {'Cons. vs Métrica':>16} {'Kappa':>8}")
                print(f"    {'─'*25} {'─'*16} {'─'*8}")
                for clf_name, m in baselines_consistency.items():
                    print(f"    {clf_name:<25} {m['accuracy_vs_expert']:>15.1%} {m['kappa_vs_expert']:>8.3f}")
                print(f"    {'LLM':<25} {consistency_metrics.accuracy:>15.1%} {consistency_metrics.cohen_kappa:>8.3f}")

    if verbose:
        print(f"\n  RESULTADOS: {problem_name}")
        print(f"    CONSISTÊNCIA (LLM vs. Métrica W_A + centróides locais): {consistency_metrics.accuracy:.1%}")
        print(f"    Kappa de Cohen: {consistency_metrics.cohen_kappa:.3f}")
        print(f"    F1-Score: {consistency_metrics.f1_score:.3f}")
        print(f"    Concordâncias: {consistency_metrics.n_agreements} | Discordâncias: {consistency_metrics.n_disagreements}")
        print(f"    Consistência da linha de base Euclidiana: {consistency_euclidean:.1%}")
        print(f"    Conjunto de teste: {len(X_test)} pontos (excluídos {len(selected_indices)} exemplos few-shot)")
        if consistency_metrics.accuracy >= 0.85:
            print(f"    O LLM MANTÉM consistência elevada com a métrica (W_A + centróides locais).")
        elif consistency_metrics.accuracy >= 0.7:
            print(f"    O LLM mantém consistência PARCIAL com a métrica (W_A + centróides locais).")
        else:
            print(f"    ⚠ O LLM NÃO mantém consistência com a métrica (W_A + centróides locais).")

    return consistency_metrics, y_llm, y_metric_test, llm_accuracy, metric_accuracy, n_malformed, consistency_euclidean, baselines_consistency, X_test


# =============================================================================
# FASE E: LLM COMO APRENDIZ
# =============================================================================

def expert_classify(
    X: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray
) -> np.ndarray:
    """
    Classifica pontos usando a métrica conhecida do perito.
    O perito atua como "verdade absoluta" (rótulo de referência) para a Fase E:
    o objetivo do LLM é aprender a reproduzir essa classificação a partir de exemplos.
    """
    return predict_with_metric(X, expert_centroids, expert_w)


# =============================================================================
# VALIDAÇÃO DO ORÁCULO: ALGORITMOS RECUPERAM W CONHECIDO?
# =============================================================================

def run_oracle_validation(
    X_a: np.ndarray, y_a: np.ndarray,
    X_b: np.ndarray, y_b: np.ndarray,
    X_c: np.ndarray, y_c: np.ndarray,
    X_e: np.ndarray, y_e: np.ndarray,
    expert_configs: List[dict],
    random_seed: int,
    verbose: bool = True,
) -> List[dict]:
    """Valida se os algoritmos de otimização inversa recuperam um W conhecido.

    Testa cada problema com seus centroides verdadeiros (conhecidos porque os
    dados são sintéticos). Para Problemas A-C o W verdadeiro é [1,1] (euclidiano,
    make_blobs gera com variância igual). Para Problema E (perito linear), testa cada expert_config.

    Usa os centroides GERADORES (não compute_centroids) para eliminar a
    ambiguidade W/centróides e testar puramente a capacidade de recuperação.
    """
    results = []

    # Configurações de cada problema com seus parâmetros conhecidos
    problems_abc = [
        {
            "name": "Problema A", "X": X_a, "y_gt": y_a,
            "true_centroids": np.array(PROBLEM_A_CENTERS),
            "true_w": np.array([1.0, 1.0]),
        },
        {
            "name": "Problema B", "X": X_b, "y_gt": y_b,
            "true_centroids": np.array(PROBLEM_B_CENTERS),
            "true_w": np.array([1.0, 1.0]),
        },
        {
            "name": "Problema C", "X": X_c, "y_gt": y_c,
            "true_centroids": np.array(PROBLEM_C_CENTERS),
            "true_w": np.array([1.0, 1.0]),
        },
    ]

    if verbose:
        print(f"\n  ═══ Parte 1: Problemas A, B, C (W verdadeiro = [1.0, 1.0]) ═══")

    for prob in problems_abc:
        X = prob["X"]
        y_gt = prob["y_gt"]
        true_centroids = prob["true_centroids"]
        true_w = prob["true_w"]
        prob_name = prob["name"]

        if verbose:
            print(f"\n  ── {prob_name}: centroides={true_centroids.tolist()}, W_verdadeiro={true_w}")

        n_classes = len(np.unique(y_gt))
        if n_classes < 2:
            if verbose:
                print(f"    ⚠ Apenas 1 classe no ground truth — pulando")
            continue

        # Testar cada algoritmo com centroides verdadeiros
        for algo_name, train_fn in _oracle_algorithms(X, y_gt, true_centroids):
            w_recovered, gamma = train_fn()
            _append_oracle_result(
                results, X, y_gt, true_centroids, true_w, w_recovered,
                random_seed, prob_name, "euclidean_gt", algo_name, verbose,
            )

    # Problema E (perito linear, Bloco 2) com cada expert config
    if verbose:
        print(f"\n  ═══ Parte 2: Problema E com experts (centroides do expert) ═══")

    for expert_cfg in expert_configs:
        expert_w = expert_cfg["w"]
        expert_name = expert_cfg["name"]
        expert_desc = expert_cfg["desc"]
        expert_centroids = EXPERT_CENTROIDS

        # Rótulos gerados pelo expert (oráculo determinístico)
        y_oracle_e = expert_classify(X_e, expert_w, expert_centroids)

        n_classes = len(np.unique(y_oracle_e))
        if n_classes < 2:
            if verbose:
                print(f"    ⚠ Expert {expert_name} gerou apenas 1 classe — pulando")
            continue

        if verbose:
            print(f"\n  ── Problema E / Expert {expert_name}: W={expert_w} ({expert_desc})")

        for algo_name, train_fn in _oracle_algorithms(X_e, y_oracle_e, expert_centroids):
            w_recovered, gamma = train_fn()
            _append_oracle_result(
                results, X_e, y_oracle_e, expert_centroids, expert_w, w_recovered,
                random_seed, "Problema E", expert_name, algo_name, verbose,
            )

    # Parte 3: Problemas com variância anisotrópica (W ≠ [1,1])
    if verbose:
        print(f"\n  ═══ Parte 3: Problemas anisotrópicos (W ≠ [1,1]) ═══")

    for aniso_cfg in ORACLE_ANISO_CONFIGS:
        std = np.array(aniso_cfg["std"])
        true_w = 1.0 / (std ** 2)  # Classificador de Bayes ótimo
        true_centroids = np.array(aniso_cfg["centers"])
        aniso_name = aniso_cfg["name"]

        X_aniso, y_aniso = create_anisotropic_problem(
            150, aniso_cfg["centers"], aniso_cfg["std"], random_seed,
        )

        n_classes = len(np.unique(y_aniso))
        if n_classes < 2:
            if verbose:
                print(f"    ⚠ Problema anisotrópico {aniso_name} gerou apenas 1 classe — pulando")
            continue

        if verbose:
            print(f"\n  ── Aniso {aniso_name}: std={std}, W_verdadeiro=[{true_w[0]:.3f}, {true_w[1]:.3f}] ({aniso_cfg['desc']})")

        for algo_name, train_fn in _oracle_algorithms(X_aniso, y_aniso, true_centroids):
            w_recovered, gamma = train_fn()
            _append_oracle_result(
                results, X_aniso, y_aniso, true_centroids, true_w, w_recovered,
                random_seed, f"Aniso_{aniso_name}", aniso_name, algo_name, verbose,
            )

    return results


def run_oracle_meialua(
    X_ml: np.ndarray,
    y_ml: np.ndarray,
    random_seed: int,
    n_features_list: Tuple[int, ...] = (2, 3, 4),
    verbose: bool = True,
) -> List[dict]:
    """Oracle de APROXIMAÇÃO para a meia-lua (item 5, reunião 20/05).

    A meia-lua é genuinamente não-linear: **não existe** um W diagonal que a
    *gere* (por isso não faz sentido gerar a meia-lua a partir de um W). Mas o
    orientador observou que pode existir uma **mudança de métrica que a aproxima**
    (~391-502s: "você pode ter um W que aproxima... uma mudança de métrica que
    aproxima um pouco da meia-lua").

    Aqui partimos dos rótulos VERDADEIROS (ground truth da meia-lua) e aprendemos
    W em espaço aumentado (2, 3, 4 features) via Perceptron e NNLS, medindo a
    fidelidade da métrica resultante vs ground truth. Espera-se que a fidelidade
    **cresça** ao adicionar x1²/x2² (R4), confirmando que uma métrica diagonal em
    espaço aumentado aproxima a fronteira não-linear — análogo ao Oracle linear,
    mas para o caso não-linear.
    """
    results = []
    if len(np.unique(y_ml)) < 2:
        return results

    for n_feat in n_features_list:
        X_aug = augment_features(X_ml, n_feat)
        centroids = compute_centroids(X_aug, y_ml)

        w_perc, gamma_perc = train_relaxed_perceptron(
            X_aug, y_ml, centroids,
            **PERCEPTRON_PARAMS,
            verbose=False,
            use_best_effort=True,
        )
        w_nnls, _ = train_least_squares_inverse(X_aug, y_ml, centroids, verbose=False)

        for algo_name, w_rec in (("perceptron", w_perc), ("nnls", w_nnls)):
            y_pred = predict_with_metric(X_aug, centroids, w_rec)
            fidelity = accuracy_score(y_ml, y_pred)
            n_err = int(np.sum(y_pred != y_ml))
            results.append({
                'random_seed': random_seed,
                'n_features': n_feat,
                'algorithm': algo_name,
                'fidelity_vs_true': fidelity,
                'n_errors': n_err,
                'n_samples': len(y_ml),
                'w_recovered_0': float(w_rec[0]),
                'w_recovered_1': float(w_rec[1]),
            })
            if verbose:
                print(
                    f"    [meia-lua oracle n_feat={n_feat} {algo_name.upper():10s}] "
                    f"fidelidade vs GT={fidelity:.1%} ({n_err}/{len(y_ml)} erros) | "
                    f"W=[{', '.join(f'{wi:.3f}' for wi in w_rec)}]"
                )
    return results




def _oracle_algorithms(X, y, centroids):
    """Retorna iterador de (nome, funcao_treino) para os algoritmos de otimização inversa."""
    yield "perceptron", lambda: train_relaxed_perceptron(
        X, y, centroids,
        **PERCEPTRON_PARAMS,
        verbose=False,
        use_best_effort=True,
    )
    yield "nnls", lambda: train_least_squares_inverse(
        X, y, centroids, verbose=False,
    )


def _append_oracle_result(
    results, X, y_true, centroids, true_w, w_recovered,
    random_seed, problem_name, expert_name, algo_name, verbose,
):
    """Calcula métricas e adiciona resultado do oráculo à lista."""
    # Fidelidade: métrica recuperada vs rótulos verdadeiros
    y_pred = predict_with_metric(X, centroids, w_recovered)
    fidelity = accuracy_score(y_true, y_pred)

    # Normalizar para comparação de razões
    w_sum = np.sum(w_recovered)
    w_norm_recovered = w_recovered / w_sum if w_sum > 0 else w_recovered

    # Similaridade de cosseno
    norm_t = np.linalg.norm(true_w)
    norm_r = np.linalg.norm(w_recovered)
    cosine_sim = np.dot(true_w, w_recovered) / (norm_t * norm_r) if (norm_t > 0 and norm_r > 0) else 0.0

    # Razões w1/w2
    true_ratio = true_w[0] / true_w[1] if true_w[1] > 0 else float('inf')
    recovered_ratio = w_recovered[0] / w_recovered[1] if w_recovered[1] > 1e-10 else float('inf')

    if verbose:
        print(f"    [{algo_name.upper():10s}] W_rec=[{w_recovered[0]:.4f}, {w_recovered[1]:.4f}] | "
              f"Fidelidade={fidelity:.1%} | Cosseno={cosine_sim:.4f} | "
              f"Razão w1/w2: verdadeira={true_ratio:.3f} recuperada={recovered_ratio:.3f}")

    results.append({
        'random_seed': random_seed,
        'problem': problem_name,
        'expert_name': expert_name,
        'true_w_0': true_w[0],
        'true_w_1': true_w[1],
        'true_w_ratio': true_ratio,
        'algorithm': algo_name,
        'recovered_w_0': w_recovered[0],
        'recovered_w_1': w_recovered[1],
        'recovered_w_norm_0': w_norm_recovered[0],
        'recovered_w_norm_1': w_norm_recovered[1],
        'recovered_w_ratio': recovered_ratio,
        'cosine_similarity': cosine_sim,
        'fidelity': fidelity,
    })


def select_examples_by_strategy(
    X: np.ndarray,
    y_expert: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    n_examples: int,
    strategy: str,
    nome_classe_0: str,
    nome_classe_1: str,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[List[Tuple[float, float, str]], np.ndarray]:
    """
    Seleciona exemplos few-shot usando estratégias de dificuldade.

    Estratégias:
        "easy":   Pontos com MAIOR margem (longe da fronteira de decisão)
                  → Exemplos mais óbvios e claros; facilitam a identificação do padrão global
        "hard":   Pontos com MENOR margem (próximos à fronteira de decisão)
                  → Exemplos mais ambíguos; testam se o LLM capta a fronteira com precisão
        "mixed":  Metade fáceis + metade difíceis
                  → Representação balanceada; hipoteticamente a melhor estratégia
        "random": Seleção aleatória (linha de base)
                  → Sem critério estratégico; referência comparativa

    Args:
        X: Matriz de características dos pontos
        y_expert: Rótulos do perito para todos os pontos
        expert_w: Pesos da métrica do perito
        expert_centroids: Centróides do perito
        n_examples: Número total de exemplos a selecionar (honrado exatamente,
            inclusive ímpar: a classe 0 recebe o exemplo excedente — ⌈n/2⌉ vs ⌊n/2⌋).
        strategy: Uma das estratégias: "easy", "hard", "mixed", "random"
        nome_classe_0: Nome da classe 0
        nome_classe_1: Nome da classe 1
        random_state: Semente aleatória para reprodutibilidade
        verbose: Se deve imprimir detalhes da seleção

    Returns:
        Tupla de (lista_de_exemplos, índices_selecionados)
    """
    rng = np.random.RandomState(random_state)

    # Calcula as margens (confiança) de todos os pontos sob a métrica do perito
    confidences, _ = compute_metric_confidence(X, expert_centroids, expert_w)

    # Honra n_examples EXATO mesmo quando ímpar: a classe 0 recebe o exemplo
    # excedente (⌈n/2⌉ vs ⌊n/2⌋) — determinístico, para o "5-shot" ter 5 exemplos
    # de fato no prompt (antes, 5//2 por classe truncava para 4).
    n_per_class_0 = n_examples - n_examples // 2
    n_per_class_1 = n_examples // 2

    idx_class_0 = np.where(y_expert == 0)[0]
    idx_class_1 = np.where(y_expert == 1)[0]

    # Aviso de desbalanceamento: alerta se uma classe tiver menos pontos do que o solicitado
    if len(idx_class_0) < n_per_class_0:
        print(f"  AVISO: Classe 0 tem apenas {len(idx_class_0)} pontos disponíveis, "
              f"mas {n_per_class_0} foram solicitados. A seleção será truncada.")
    if len(idx_class_1) < n_per_class_1:
        print(f"  AVISO: Classe 1 tem apenas {len(idx_class_1)} pontos disponíveis, "
              f"mas {n_per_class_1} foram solicitados. A seleção será truncada.")

    conf_class_0 = confidences[idx_class_0]
    conf_class_1 = confidences[idx_class_1]

    if strategy == "easy":
        # Seleciona pontos com MAIOR margem (longe da fronteira, mais confiantes)
        # Justificativa: exemplos "fáceis" são os mais representativos de cada classe
        sorted_0 = idx_class_0[np.argsort(-conf_class_0)]  # Ordem decrescente (maior margem primeiro)
        sorted_1 = idx_class_1[np.argsort(-conf_class_1)]
        selected_0 = sorted_0[:n_per_class_0]
        selected_1 = sorted_1[:n_per_class_1]

    elif strategy == "hard":
        # Seleciona pontos com MENOR margem (próximos à fronteira, mais ambíguos)
        # Justificativa: exemplos "difíceis" testam se o LLM capta a fronteira com precisão
        sorted_0 = idx_class_0[np.argsort(conf_class_0)]  # Ordem crescente (menor margem primeiro)
        sorted_1 = idx_class_1[np.argsort(conf_class_1)]
        selected_0 = sorted_0[:n_per_class_0]
        selected_1 = sorted_1[:n_per_class_1]

    elif strategy == "mixed":
        # Metade fáceis (maior margem) + metade difíceis (menor margem) por classe
        # Justificativa: cobertura balanceada da fronteira e das regiões centrais de cada classe
        sorted_0_desc = idx_class_0[np.argsort(-conf_class_0)]
        sorted_0_asc = idx_class_0[np.argsort(conf_class_0)]
        sorted_1_desc = idx_class_1[np.argsort(-conf_class_1)]
        sorted_1_asc = idx_class_1[np.argsort(conf_class_1)]

        n_easy_0 = n_per_class_0 // 2
        n_hard_0 = n_per_class_0 - n_easy_0
        n_easy_1 = n_per_class_1 // 2
        n_hard_1 = n_per_class_1 - n_easy_1

        easy_0 = sorted_0_desc[:n_easy_0]
        hard_0 = sorted_0_asc[:n_hard_0]
        easy_1 = sorted_1_desc[:n_easy_1]
        hard_1 = sorted_1_asc[:n_hard_1]

        # Remove duplicatas preservando a ordem de inserção (easy primeiro, depois hard).
        # np.unique não é usado aqui pois ordena numericamente e destruiria o balanço easy/hard.
        def _unique_ordered(arr: np.ndarray) -> np.ndarray:
            seen = set()
            return np.array([x for x in arr if not (x in seen or seen.add(x))], dtype=arr.dtype)

        selected_0 = _unique_ordered(np.concatenate([easy_0, hard_0]))[:n_per_class_0]
        selected_1 = _unique_ordered(np.concatenate([easy_1, hard_1]))[:n_per_class_1]

    elif strategy == "random":
        # Seleção aleatória (linha de base sem critério estratégico)
        selected_0 = rng.choice(idx_class_0, size=min(n_per_class_0, len(idx_class_0)), replace=False)
        selected_1 = rng.choice(idx_class_1, size=min(n_per_class_1, len(idx_class_1)), replace=False)

    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    selected_indices = np.concatenate([selected_0, selected_1])

    if verbose:
        print(f"      Estratégia: {strategy.upper()}")
        if len(selected_0) > 0:
            margins_0 = confidences[selected_0]
            print(f"      Classe 0: {len(selected_0)} exemplos, "
                  f"faixa de margem [{margins_0.min():.3f}, {margins_0.max():.3f}], "
                  f"média={margins_0.mean():.3f}")
        if len(selected_1) > 0:
            margins_1 = confidences[selected_1]
            print(f"      Classe 1: {len(selected_1)} exemplos, "
                  f"faixa de margem [{margins_1.min():.3f}, {margins_1.max():.3f}], "
                  f"média={margins_1.mean():.3f}")

    # Monta a lista de exemplos com os rótulos do PERITO (não do ground truth)
    examples = []
    for i in selected_0:
        examples.append((X[i, 0], X[i, 1], nome_classe_0))
    for i in selected_1:
        examples.append((X[i, 0], X[i, 1], nome_classe_1))

    return examples, selected_indices


def select_examples_dilution(
    X: np.ndarray,
    y_expert: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    n_hard_fixed: int,
    n_easy_added: int,
    nome_classe_0: str,
    nome_classe_1: str,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[List[Tuple[float, float, str]], np.ndarray]:
    """Seleciona exemplos para o experimento de diluição: N hard fixos + M easy adicionais.

    Primeiro seleciona os top-N pontos hard (menor margem), depois adiciona
    M pontos easy (maior margem), excluindo os hard já selecionados.
    """
    confidences, _ = compute_metric_confidence(X, expert_centroids, expert_w)

    n_hard_per_class = n_hard_fixed // 2
    n_easy_per_class = n_easy_added // 2

    idx_class_0 = np.where(y_expert == 0)[0]
    idx_class_1 = np.where(y_expert == 1)[0]

    conf_0 = confidences[idx_class_0]
    conf_1 = confidences[idx_class_1]

    # Selecionar hard (menor margem)
    sorted_0_asc = idx_class_0[np.argsort(conf_0)]
    sorted_1_asc = idx_class_1[np.argsort(conf_1)]
    hard_0 = sorted_0_asc[:n_hard_per_class]
    hard_1 = sorted_1_asc[:n_hard_per_class]
    hard_set = set(hard_0.tolist() + hard_1.tolist())

    # Selecionar easy (maior margem), excluindo hard
    available_0 = np.array([i for i in idx_class_0 if i not in hard_set])
    available_1 = np.array([i for i in idx_class_1 if i not in hard_set])
    if len(available_0) > 0:
        sorted_avail_0_desc = available_0[np.argsort(-confidences[available_0])]
        easy_0 = sorted_avail_0_desc[:n_easy_per_class]
    else:
        easy_0 = np.array([], dtype=int)
    if len(available_1) > 0:
        sorted_avail_1_desc = available_1[np.argsort(-confidences[available_1])]
        easy_1 = sorted_avail_1_desc[:n_easy_per_class]
    else:
        easy_1 = np.array([], dtype=int)

    selected_indices = np.concatenate([hard_0, hard_1, easy_0, easy_1]).astype(int)

    examples = []
    for i in selected_indices:
        label = nome_classe_0 if y_expert[i] == 0 else nome_classe_1
        examples.append((X[i, 0], X[i, 1], label))

    if verbose:
        print(f"      Diluição: {len(hard_0)+len(hard_1)} hard + {len(easy_0)+len(easy_1)} easy = {len(selected_indices)} total")

    return examples, selected_indices


def reorder_examples(
    examples: List[Tuple],
    ordering: str,
    nome_classe_0: str,
    nome_classe_1: str,
    random_state: int = 42
) -> List[Tuple]:
    """Reordena exemplos few-shot para testar viés de recência (recency bias).

    Estratégias de ordenação:
        "class0_first":  Todos da classe 0, depois classe 1 (padrão atual)
        "class1_first":  Todos da classe 1, depois classe 0 (invertido)
        "shuffled":      Ordem aleatória (elimina viés posicional)
        "alternating":   Alternado (classe 0, classe 1, classe 0, ...)

    Args:
        examples: Lista de tuplas (x1, x2, label) ou (x1, x2, ..., label)
        ordering: Estratégia de ordenação
        nome_classe_0: Nome da classe 0
        nome_classe_1: Nome da classe 1
        random_state: Semente para reprodutibilidade do shuffle

    Returns:
        Lista de exemplos reordenada
    """
    if ordering == "class0_first":
        # Ordem padrão: classe 0 primeiro, depois classe 1
        ex_0 = [e for e in examples if e[-1] == nome_classe_0]
        ex_1 = [e for e in examples if e[-1] == nome_classe_1]
        return ex_0 + ex_1

    elif ordering == "class1_first":
        # Invertido: classe 1 primeiro, depois classe 0
        ex_0 = [e for e in examples if e[-1] == nome_classe_0]
        ex_1 = [e for e in examples if e[-1] == nome_classe_1]
        return ex_1 + ex_0

    elif ordering == "shuffled":
        # Aleatório: elimina qualquer viés posicional
        rng = np.random.RandomState(random_state)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        return shuffled

    elif ordering == "alternating":
        # Alternado: classe 0, classe 1, classe 0, classe 1, ...
        ex_0 = [e for e in examples if e[-1] == nome_classe_0]
        ex_1 = [e for e in examples if e[-1] == nome_classe_1]
        result = []
        i0, i1 = 0, 0
        while i0 < len(ex_0) or i1 < len(ex_1):
            if i0 < len(ex_0):
                result.append(ex_0[i0])
                i0 += 1
            if i1 < len(ex_1):
                result.append(ex_1[i1])
                i1 += 1
        return result

    else:
        raise ValueError(f"Ordenação desconhecida: {ordering}")


def phase_e_llm_as_learner(
    X_e: np.ndarray,
    y_gt_e: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    n_shot: int,
    strategy: str,
    nome_classe_0: str,
    nome_classe_1: str,
    provider: str,
    model_name: str,
    temperature: float,
    random_seed: int,
    repeticao: int,
    verbose: bool = True
) -> ResultadoPhaseEExperimento:
    """
    Fase E — LLM como Aprendiz.

    O perito (com métrica W_expert conhecida) rotula os dados.
    O LLM recebe exemplos few-shot do perito e tenta
    aprender o padrão de classificação do perito.

    Testa se o LLM consegue MELHORAR seu alinhamento com o perito
    conforme mais exemplos são fornecidos, e como a dificuldade dos exemplos
    afeta a curva de aprendizado.
    """
    shot_label = "zero-shot" if n_shot == 0 else f"{n_shot}-shot"

    if verbose:
        print(f"\n  ══════════════════════════════════════════════════════════════")
        print(f"  FASE E: LLM COMO APRENDIZ ({shot_label}, estratégia={strategy})")
        print(f"  Métrica do Perito W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
        print(f"  ══════════════════════════════════════════════════════════════")
        print(f"  O perito rotula os dados com sua métrica conhecida.")
        print(f"  O LLM recebe exemplos do perito e deve APRENDER o padrão de classificação.")

    # Passo 1: Perito classifica todos os pontos usando sua métrica W_expert conhecida
    y_expert = expert_classify(X_e, expert_w, expert_centroids)
    expert_accuracy = accuracy_score(y_gt_e, y_expert)

    if verbose:
        n_0 = np.sum(y_expert == 0)
        n_1 = np.sum(y_expert == 1)
        print(f"\n  Passo 1: Classificação pelo Perito (métrica W_expert conhecida)")
        print(f"    Rótulos do perito: Classe 0={n_0}, Classe 1={n_1}")
        print(f"    Acurácia do perito vs. ground truth: {expert_accuracy:.1%}")

    # Passo 2: Seleciona exemplos few-shot conforme a estratégia escolhida (se n_shot > 0)
    examples = None
    example_indices = np.array([], dtype=int)

    if n_shot > 0:
        if verbose:
            print(f"\n  Passo 2: Selecionando {n_shot} exemplos (estratégia={strategy})...")
            estrategia_descricao = {
                "easy": "pontos LONGE da fronteira (mais fáceis)",
                "hard": "pontos PRÓXIMOS à fronteira (mais difíceis)",
                "mixed": "metade fáceis + metade difíceis (balanceado)",
                "random": "seleção aleatória (linha de base)"
            }
            print(f"    Estratégia: {strategy.upper()} — {estrategia_descricao.get(strategy, '')}")

        examples, example_indices = select_examples_by_strategy(
            X_e, y_expert, expert_w, expert_centroids,
            n_examples=n_shot,
            strategy=strategy,
            nome_classe_0=nome_classe_0,
            nome_classe_1=nome_classe_1,
            random_state=random_seed + repeticao,
            verbose=verbose
        )

    # Passo 3: Define o conjunto de teste — todos os pontos NÃO usados como exemplos
    # Isso garante avaliação honesta: o LLM não é testado nos pontos que recebeu como exemplos
    all_indices = np.arange(len(X_e))
    test_mask = np.ones(len(X_e), dtype=bool)
    if len(example_indices) > 0:
        test_mask[example_indices] = False
    test_indices = all_indices[test_mask]

    X_test = X_e[test_indices]
    y_expert_test = y_expert[test_indices]
    y_gt_test = y_gt_e[test_indices]

    if verbose:
        print(f"\n  Passo 3: LLM classificando {len(X_test)} pontos de teste ({shot_label})...")
        if n_shot > 0:
            print(f"    O LLM receberá {n_shot} exemplos rotulados pelo perito como contexto.")

    # Passo 4: LLM classifica os pontos de teste usando os exemplos como contexto few-shot
    y_llm_test, n_malformed = collect_llm_decisions(
        X_test, nome_classe_0, nome_classe_1,
        examples=examples, verbose=verbose,
        label_prefix="[Fase E] "
    )

    # Passo 5: Computa métricas de alinhamento (LLM vs. Perito) e acurácia vs. ground truth
    consistency = compute_consistency_metrics(y_llm_test, y_expert_test)
    llm_accuracy_vs_gt = accuracy_score(y_gt_test, y_llm_test)

    if verbose:
        print(f"\n  ═══════════════════════════════════════════════════════════")
        print(f"  RESULTADOS: FASE E ({shot_label}, estratégia={strategy})")
        print(f"  ═══════════════════════════════════════════════════════════")
        print(f"    Concordância LLM vs. Perito:  {consistency.accuracy:.1%}")
        print(f"    Kappa de Cohen:               {consistency.cohen_kappa:.3f}")
        print(f"    F1-Score:                     {consistency.f1_score:.3f}")
        print(f"    ")
        print(f"    Acurácia do LLM vs. GT:       {llm_accuracy_vs_gt:.1%}")
        print(f"    Acurácia do Perito vs. GT:    {expert_accuracy:.1%}")
        print(f"    Concordâncias: {consistency.n_agreements} | Discordâncias: {consistency.n_disagreements}")

        if consistency.accuracy > 0.85:
            print(f"    → O LLM APRENDEU com sucesso o padrão de classificação do perito!")
        elif consistency.accuracy > 0.7:
            print(f"    → O LLM tem ALINHAMENTO PARCIAL com o perito.")
        else:
            print(f"    → O LLM NÃO conseguiu aprender o padrão do perito com {n_shot} exemplos.")

    return ResultadoPhaseEExperimento(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        random_seed=random_seed,
        n_shot=n_shot,
        example_strategy=strategy,
        nomes_classes=(nome_classe_0, nome_classe_1),
        repeticao=repeticao,
        accuracy_llm_vs_expert=consistency.accuracy,
        kappa_llm_vs_expert=consistency.cohen_kappa,
        f1_llm_vs_expert=consistency.f1_score,
        accuracy_expert_vs_gt=expert_accuracy,
        accuracy_llm_vs_gt=llm_accuracy_vs_gt,
        n_classe_0_expert=int(np.sum(y_expert == 0)),
        n_classe_1_expert=int(np.sum(y_expert == 1)),
        n_classe_0_llm=int(np.sum(y_llm_test == 0)),
        n_classe_1_llm=int(np.sum(y_llm_test == 1)),
        n_disagreements=consistency.n_disagreements,
        n_total_test=len(X_test),
        n_malformed_responses=n_malformed,
        expert_w=expert_w.copy(),
    )


# =============================================================================
# EXPERIMENTO COMPLETO (FASES A-C)
# =============================================================================

def run_complete_experiment(
    X_train_a: np.ndarray,
    y_true_train_a: np.ndarray,
    X_b: np.ndarray,
    y_true_b: np.ndarray,
    X_c: np.ndarray,
    y_true_c: np.ndarray,
    n_shot: int,
    nome_classe_0: str,
    nome_classe_1: str,
    repeticao: int,
    provider: str,
    model_name: str,
    temperature: float,
    random_seed: int,
    learned_metric_cache: Optional[LearnedMetric] = None,
    learned_metric_nnls_cache: Optional[LearnedMetric] = None,
    y_llm_train_a_cache: Optional[np.ndarray] = None,
    fidelity_cache: Optional[float] = None,
    fidelity_nnls_cache: Optional[float] = None,
    llm_accuracy_a_cache: Optional[float] = None,
    n_malformed_a_cache: Optional[int] = None,
    verbose: bool = True,
    prompt_variant: str = "default"
) -> Tuple[ResultadoExperimento, LearnedMetric, np.ndarray, float, float, int]:
    """Executa o experimento completo: Fase A + Fase B + Fase C.

    Utiliza cache da Fase A para evitar chamadas redundantes à API do LLM —
    uma vez que W é aprendido, ele é reutilizado em todos os tamanhos de n_shot
    e repetições com os mesmos nomes de classe e semente.
    """
    shot_label = "zero-shot" if n_shot == 0 else f"{n_shot}-shot"

    if verbose:
        print_section(
            f"EXPERIMENT: Phase B/C={shot_label} | Classes: {nome_classe_0}/{nome_classe_1} | Rep: {repeticao+1} | Seed: {random_seed}",
            "─"
        )

    total_malformed = 0

    if learned_metric_cache is not None:
        if verbose:
            print("\n  [Usando resultados da Fase A do cache — evitando chamadas redundantes à API]")
        learned_metric = learned_metric_cache
        learned_metric_nnls = learned_metric_nnls_cache
        y_llm_train_a = y_llm_train_a_cache
        fidelity_a = fidelity_cache
        fidelity_nnls_a = fidelity_nnls_cache
        llm_accuracy_a = llm_accuracy_a_cache
        total_malformed += n_malformed_a_cache if n_malformed_a_cache else 0
    else:
        if verbose:
            print("\n  [Iniciando Fase A: Estimando a Métrica W a partir das decisões do LLM...]")
        result_a = phase_a_learn_metric(
            X_train_a, y_true_train_a,
            nome_classe_0, nome_classe_1, verbose=verbose,
            prompt_variant=prompt_variant
        )
        learned_metric, learned_metric_nnls, y_llm_train_a, fidelity_a, fidelity_nnls_a, llm_accuracy_a, n_malformed_a = result_a
        total_malformed += n_malformed_a

    if learned_metric is None:
        return (
            ResultadoExperimento(
                provider=provider, model_name=model_name, temperature=temperature,
                random_seed=random_seed, n_shot=n_shot,
                nomes_classes=(nome_classe_0, nome_classe_1), repeticao=repeticao,
                fidelidade_problema_a=0.5, acuracia_llm_vs_gt_problema_a=llm_accuracy_a,
                consistencia_problema_b=0.5, kappa_problema_b=0.0, f1_problema_b=0.5,
                acuracia_llm_vs_gt_problema_b=0.5, acuracia_metrica_vs_gt_problema_b=0.5,
                consistencia_problema_c=0.5, kappa_problema_c=0.0, f1_problema_c=0.5,
                acuracia_llm_vs_gt_problema_c=0.5, acuracia_metrica_vs_gt_problema_c=0.5,
                w_aprendido=np.ones(2), gamma_otimo=0.0,
                n_classe_0_problema_a=np.sum(y_llm_train_a == 0),
                n_classe_1_problema_a=np.sum(y_llm_train_a == 1),
                n_classe_0_problema_b=0, n_classe_1_problema_b=0,
                n_classe_0_problema_c=0, n_classe_1_problema_c=0,
                n_disagreements_b=0, n_disagreements_c=0,
                n_malformed_responses=total_malformed,
                consistencia_euclidiana_problema_b=0.0,
                consistencia_euclidiana_problema_c=0.0,
                diagonal_limitation_flag=0,
                w_direction=np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                w_cosine_sim_nnls=0.0,
                prompt_variant=prompt_variant,
            ),
            None, y_llm_train_a, 0.5, llm_accuracy_a, total_malformed, {}
        )

    # --- Fases B/C com Perceptron (algoritmo principal) ---
    metrics_b, y_llm_b, y_pred_metric_b, llm_accuracy_b, metric_accuracy_b, n_malformed_b, euclidean_b, baselines_b, X_test_b = phase_consistency_test(
        X_b, y_true_b, learned_metric,
        nome_classe_0, nome_classe_1, n_shot, "FASE B (Problema B)", verbose=verbose,
        prompt_variant=prompt_variant
    )
    total_malformed += n_malformed_b

    metrics_c, y_llm_c, y_pred_metric_c, llm_accuracy_c, metric_accuracy_c, n_malformed_c, euclidean_c, baselines_c, X_test_c = phase_consistency_test(
        X_c, y_true_c, learned_metric,
        nome_classe_0, nome_classe_1, n_shot, "FASE C (Problema C)", verbose=verbose,
        prompt_variant=prompt_variant
    )
    total_malformed += n_malformed_c

    # --- Fases B/C com NNLS (validação cruzada do método) ---
    # Usa os mesmos y_llm já coletados (não faz novas chamadas à API).
    # Apenas compara predições da métrica NNLS com as decisões do LLM.
    nnls_detail = {}
    if learned_metric_nnls is not None:
        # Problema B: predições NNLS vs. LLM (usando y_llm_b já coletado acima)
        # Usa X_test_b (filtrado, sem exemplos few-shot) para alinhar com y_llm_b
        local_centroids_b = compute_centroids(X_test_b, y_llm_b) if len(np.unique(y_llm_b)) >= 2 else learned_metric_nnls.centroids
        y_pred_nnls_b = predict_with_metric(X_test_b, local_centroids_b, learned_metric_nnls.w)
        metrics_nnls_b = compute_consistency_metrics(y_llm_b, y_pred_nnls_b)

        # Problema C: predições NNLS vs. LLM (usando y_llm_c já coletado acima)
        local_centroids_c = compute_centroids(X_test_c, y_llm_c) if len(np.unique(y_llm_c)) >= 2 else learned_metric_nnls.centroids
        y_pred_nnls_c = predict_with_metric(X_test_c, local_centroids_c, learned_metric_nnls.w)
        metrics_nnls_c = compute_consistency_metrics(y_llm_c, y_pred_nnls_c)

        if verbose:
            print(f"\n  --- Comparação Perceptron vs NNLS (Fases B/C) ---")
            print(f"    [Perceptron] Consistência B: {metrics_b.accuracy:.1%} | Kappa B: {metrics_b.cohen_kappa:.3f}")
            print(f"    [NNLS]       Consistência B: {metrics_nnls_b.accuracy:.1%} | Kappa B: {metrics_nnls_b.cohen_kappa:.3f}")
            print(f"    [Perceptron] Consistência C: {metrics_c.accuracy:.1%} | Kappa C: {metrics_c.cohen_kappa:.3f}")
            print(f"    [NNLS]       Consistência C: {metrics_nnls_c.accuracy:.1%} | Kappa C: {metrics_nnls_c.cohen_kappa:.3f}")

        nnls_detail = {
            'metrics_nnls_b': metrics_nnls_b, 'metrics_nnls_c': metrics_nnls_c,
            'y_pred_nnls_b': y_pred_nnls_b, 'y_pred_nnls_c': y_pred_nnls_c,
            'fidelity_nnls_a': fidelity_nnls_a,
            'w_nnls': learned_metric_nnls.w,
        }

    # Sinalização de limitação diagonal: verifica se a métrica estimada supera a Euclidiana.
    # Se a melhoria for <= 1%, a restrição diagonal pode ser o gargalo — métricas não-diagonais
    # (ex.: Mahalanobis completo) poderiam ser mais expressivas neste caso.
    if metrics_b.accuracy <= euclidean_b + 0.01:
        diagonal_limitation_flag = 1
    else:
        diagonal_limitation_flag = 0

    # Direção de W (vetor unitário) — invariante à escala da margem.
    # Comparar direções entre seeds é o teste correto de H5 (estabilidade),
    # pois a magnitude de W depende da escala do target_margin e do gamma,
    # mas a razão entre componentes (que determina a fronteira) é invariante.
    w_vec = learned_metric.w
    w_norm_val = np.linalg.norm(w_vec)
    w_dir = w_vec / w_norm_val if w_norm_val > 0 else w_vec

    # Similaridade cosseno com NNLS (robustez ao método de estimação)
    cos_sim_nnls = 0.0
    if learned_metric_nnls is not None:
        w_nnls_vec = learned_metric_nnls.w
        denom = np.linalg.norm(w_vec) * np.linalg.norm(w_nnls_vec)
        cos_sim_nnls = float(np.dot(w_vec, w_nnls_vec) / (denom + 1e-9))

    return (
        ResultadoExperimento(
            provider=provider, model_name=model_name, temperature=temperature,
            random_seed=random_seed, n_shot=n_shot,
            nomes_classes=(nome_classe_0, nome_classe_1), repeticao=repeticao,
            fidelidade_problema_a=fidelity_a,
            acuracia_llm_vs_gt_problema_a=llm_accuracy_a,
            consistencia_problema_b=metrics_b.accuracy,
            kappa_problema_b=metrics_b.cohen_kappa,
            f1_problema_b=metrics_b.f1_score,
            acuracia_llm_vs_gt_problema_b=llm_accuracy_b,
            acuracia_metrica_vs_gt_problema_b=metric_accuracy_b,
            consistencia_problema_c=metrics_c.accuracy,
            kappa_problema_c=metrics_c.cohen_kappa,
            f1_problema_c=metrics_c.f1_score,
            acuracia_llm_vs_gt_problema_c=llm_accuracy_c,
            acuracia_metrica_vs_gt_problema_c=metric_accuracy_c,
            w_aprendido=learned_metric.w,
            gamma_otimo=learned_metric.gamma,
            n_classe_0_problema_a=np.sum(y_llm_train_a == 0),
            n_classe_1_problema_a=np.sum(y_llm_train_a == 1),
            n_classe_0_problema_b=np.sum(y_llm_b == 0),
            n_classe_1_problema_b=np.sum(y_llm_b == 1),
            n_classe_0_problema_c=np.sum(y_llm_c == 0),
            n_classe_1_problema_c=np.sum(y_llm_c == 1),
            n_disagreements_b=metrics_b.n_disagreements,
            n_disagreements_c=metrics_c.n_disagreements,
            n_malformed_responses=total_malformed,
            consistencia_euclidiana_problema_b=euclidean_b,
            consistencia_euclidiana_problema_c=euclidean_c,
            diagonal_limitation_flag=diagonal_limitation_flag,
            w_direction=w_dir,
            w_cosine_sim_nnls=cos_sim_nnls,
            prompt_variant=prompt_variant,
        ),
        learned_metric, y_llm_train_a, fidelity_a, llm_accuracy_a, total_malformed,
        {
            'y_llm_b': y_llm_b, 'y_llm_c': y_llm_c,
            'y_metric_b': y_pred_metric_b, 'y_metric_c': y_pred_metric_c,
            'metrics_b': metrics_b, 'metrics_c': metrics_c,
            'baselines_b': baselines_b, 'baselines_c': baselines_c,
            'learned_metric_nnls': learned_metric_nnls,
            **nnls_detail,
        }
    )


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def _run_phase_abc_experiment(X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                              n_shot, nome_0, nome_1, rep, provider, model_name,
                              temperature, seed, seed_idx, phase_a_cache, verbose,
                              prompt_variant="default"):
    """Helper para executar um experimento A-C com cache."""
    # Inclui provider/model_name na chave para blindar contra contaminação entre
    # modelos caso o phase_a_cache passe a ser compartilhado entre iterações de modelo
    # (hoje ele é reinicializado por modelo+seed, mas a chave não deve depender disso).
    cache_key = (provider, model_name, seed, nome_0, nome_1, prompt_variant)
    if cache_key in phase_a_cache:
        cached = phase_a_cache[cache_key]
        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail = run_complete_experiment(
            X_train_a.copy(), y_train_a.copy(),
            X_b.copy(), y_b.copy(), X_c.copy(), y_c.copy(),
            n_shot=n_shot, nome_classe_0=nome_0, nome_classe_1=nome_1,
            repeticao=rep, provider=provider, model_name=model_name,
            temperature=temperature, random_seed=seed,
            learned_metric_cache=cached['metric'],
            learned_metric_nnls_cache=cached.get('metric_nnls'),
            y_llm_train_a_cache=cached['y_llm'],
            fidelity_cache=cached['fidelity'],
            fidelity_nnls_cache=cached.get('fidelity_nnls'),
            llm_accuracy_a_cache=cached['llm_acc'],
            n_malformed_a_cache=cached['n_malformed'],
            verbose=verbose,
            prompt_variant=prompt_variant
        )
    else:
        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail = run_complete_experiment(
            X_train_a.copy(), y_train_a.copy(),
            X_b.copy(), y_b.copy(), X_c.copy(), y_c.copy(),
            n_shot=n_shot, nome_classe_0=nome_0, nome_classe_1=nome_1,
            repeticao=rep, provider=provider, model_name=model_name,
            temperature=temperature, random_seed=seed,
            verbose=verbose,
            prompt_variant=prompt_variant
        )
        if learned_metric is not None:
            phase_a_cache[cache_key] = {
                'metric': learned_metric,
                'metric_nnls': detail.get('learned_metric_nnls'),
                'y_llm': y_llm_train_a,
                'fidelity': fidelity,
                'fidelity_nnls': detail.get('fidelity_nnls_a'),
                'llm_acc': llm_acc_a,
                'n_malformed': n_malformed,
            }
    return result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail


# =============================================================================
# PIPELINE PARA PROBLEMAS EXTERNOS (Bloco 2/3 — peso×altura e meia-lua)
#
# Implementa os itens da reunião 30/04/2026 (e-mails 19:15 e 22:04):
#   - Item 2: base peso×altura como problema central
#   - Item 3: variantes de prompt (x1,x2 vs peso,altura)
#   - Item 4: Fase A com n_features ∈ {2, 3, 4} para detectar não-linearidade
#   - Item 5: acurácia da métrica vs rótulo ORIGINAL (atende e-mail 22:04 ponto 4)
#   - Item 6: Fase E com a melhor métrica (exemplos no mesmo n_feat)
#   - Item 7: visualização ponto-a-ponto das rotulações (e-mail 22:06)
#   - Itens 8-11: mesma pipeline aplicada à meia-lua (Parte 3 do plano)
# =============================================================================

def phase_a_multifeature(
    X: np.ndarray,
    y_true: np.ndarray,
    n_features: int,
    nome_classe_0: str,
    nome_classe_1: str,
    nome_feature_0: str = "x1",
    nome_feature_1: str = "x2",
    prompt_variant: str = "default",
    y_llm_cached: Optional[np.ndarray] = None,
    verbose: bool = False,
    label_prefix: str = "",
) -> Optional[dict]:
    """Fase A generalizada para 2, 3 ou 4 features.

    O LLM SEMPRE vê apenas 2 features (x1, x2) — as features adicionais
    (x1·x2 ou x1², x2²) são aplicadas APENAS na construção da métrica W.
    Isso é proposital: queremos saber se uma métrica diagonal em espaço
    aumentado captura a não-linearidade no critério decisional do LLM.

    Retorna dict com Perceptron + NNLS, fidelidade (vs LLM) e
    acurácia (vs rótulos verdadeiros).
    """
    if y_llm_cached is None:
        y_llm, n_malformed = collect_llm_decisions(
            X[:, :2], nome_classe_0, nome_classe_1,
            examples=None, verbose=verbose, label_prefix=label_prefix,
            nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
            prompt_variant=prompt_variant,
        )
    else:
        y_llm = y_llm_cached
        n_malformed = 0

    if len(np.unique(y_llm)) < 2:
        return None

    X_aug = augment_features(X, n_features)
    centroids = compute_centroids(X_aug, y_llm)

    w_perc, gamma = train_relaxed_perceptron(
        X_aug, y_llm, centroids,
        **PERCEPTRON_PARAMS,
        verbose=False,
        use_best_effort=True,
    )
    w_nnls, _ = train_least_squares_inverse(X_aug, y_llm, centroids, verbose=False)

    y_metric_perc = predict_with_metric(X_aug, centroids, w_perc)
    y_metric_nnls = predict_with_metric(X_aug, centroids, w_nnls)

    return {
        'n_features': n_features,
        'prompt_variant': prompt_variant,
        'feature_names': (nome_feature_0, nome_feature_1),
        'X_aug': X_aug,
        'y_llm': y_llm,
        'y_true': y_true,
        'centroids': centroids,
        'w_perc': w_perc,
        'w_nnls': w_nnls,
        'gamma_perc': gamma,
        'y_metric_perc': y_metric_perc,
        'y_metric_nnls': y_metric_nnls,
        'fidelity_perc_vs_llm': accuracy_score(y_llm, y_metric_perc),
        'fidelity_nnls_vs_llm': accuracy_score(y_llm, y_metric_nnls),
        'accuracy_perc_vs_true': accuracy_score(y_true, y_metric_perc),
        'accuracy_nnls_vs_true': accuracy_score(y_true, y_metric_nnls),
        'llm_accuracy_vs_true': accuracy_score(y_true, y_llm),
        # Item 15 (reunião 20/05): contagem absoluta de erros (problema pequeno).
        'n_samples': len(y_true),
        'n_errors_perc_vs_true': int(np.sum(y_metric_perc != y_true)),
        'n_errors_nnls_vs_true': int(np.sum(y_metric_nnls != y_true)),
        'n_errors_llm_vs_true': int(np.sum(y_llm != y_true)),
        'n_malformed': n_malformed,
    }




def run_external_problem_pipeline(
    problem_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    nome_classe_0: str,
    nome_classe_1: str,
    feature_variants: List[Tuple[str, str]],
    seeds: List[int],
    n_shots_phase_e: List[int],
    pasta_execucao: str,
    n_train_ratio: float = 0.7,
    verbose: bool = True,
) -> dict:
    """Pipeline completo para um problema externo (peso×altura, meia-lua).

    Para cada (seed, feature_variant):
      1. Split treino/teste (n_train_ratio).
      2. Fase A com n_features ∈ {2, 3, 4}, calculando fidelidade (vs LLM)
         e acurácia (vs rótulos reais).
      3. Identifica a melhor configuração (maior acurácia vs real).
      4. Fase E: LLM aprende com exemplos rotulados pela melhor métrica.

    Retorna dict com:
      - 'phase_a_results': lista de resultados da Fase A (todas configs)
      - 'phase_e_results': lista de resultados da Fase E (somente best)
      - 'llm_label_maps': dict {(problem_name, seed, feat_0, feat_1, kind):
        {'X', 'y_llm', 'y_true', ...}} — o problem_name na chave evita colisão
        entre problemas que usam os mesmos nomes de feature (ex.: meia-lua e
        peso×altura ambos com x1/x2)
                          para a visualização ponto-a-ponto
    """
    all_phase_a = []
    all_phase_e = []
    llm_label_maps = {}

    if verbose:
        print_section(f"PIPELINE: {problem_name}", "═")
        print(f"  Variantes de feature: {feature_variants}")
        print(f"  Sementes: {seeds}")
        print(f"  N amostras total: {len(X)} | Split treino/teste: {n_train_ratio:.0%}/{1-n_train_ratio:.0%}")

    for seed in seeds:
        # RNG LOCAL por seed (não o estado global): blinda o split e o sorteio
        # de exemplos contra qualquer consumo de aleatoriedade global inserido
        # entre este ponto e os usos abaixo.
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        n_train = int(n_train_ratio * len(X))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        X_train, y_train = X[train_idx], y_true[train_idx]
        X_test, y_test = X[test_idx], y_true[test_idx]

        if verbose:
            print(f"\n  ── Seed {seed} ── n_train={len(X_train)}, n_test={len(X_test)}")

        for feat_0, feat_1 in feature_variants:
            if verbose:
                print(f"\n    Variante de feature: ({feat_0}, {feat_1})")

            # Coleta LLM uma única vez (n_features=2 visível para o LLM)
            t_collect = time.time()
            print(f"    [{problem_name} seed={seed} {feat_0}/{feat_1}] Coletando LLM Fase A (zero-shot, n={len(X_train)})...", flush=True)
            y_llm_train, n_mal = collect_llm_decisions(
                X_train, nome_classe_0, nome_classe_1,
                examples=None, verbose=False,
                nome_feature_0=feat_0, nome_feature_1=feat_1,
                label_prefix=f"[{problem_name} seed={seed} {feat_0}/{feat_1}] ",
            )
            print(f"    [{problem_name} seed={seed} {feat_0}/{feat_1}] Fase A coletada em {time.time()-t_collect:.1f}s (malformadas={n_mal})", flush=True)

            llm_label_maps[(problem_name, seed, feat_0, feat_1, 'train')] = {
                'X': X_train.copy(), 'y_llm': y_llm_train.copy(),
                'y_true': y_train.copy(),
            }

            # Fase A com 2 / 3 / 4 features (reutiliza y_llm_train)
            results_by_nfeat = []
            for n_feat in [2, 3, 4]:
                r = phase_a_multifeature(
                    X_train, y_train, n_feat,
                    nome_classe_0, nome_classe_1,
                    feat_0, feat_1,
                    prompt_variant="default",
                    y_llm_cached=y_llm_train,
                    verbose=False,
                    label_prefix=f"[{problem_name} seed={seed} {n_feat}feat] ",
                )
                if r is None:
                    if verbose:
                        print(f"      n_features={n_feat}: classe única no LLM, pulando.")
                    continue
                r['seed'] = seed
                r['problem_name'] = problem_name
                results_by_nfeat.append(r)
                all_phase_a.append(r)

                if verbose:
                    print(
                        f"      n_features={n_feat}: fid_perc={r['fidelity_perc_vs_llm']:.1%} | "
                        f"acc_perc_real={r['accuracy_perc_vs_true']:.1%} | "
                        f"fid_nnls={r['fidelity_nnls_vs_llm']:.1%} | "
                        f"acc_nnls_real={r['accuracy_nnls_vs_true']:.1%} | "
                        f"llm_acc_real={r['llm_accuracy_vs_true']:.1%}"
                    )

            if not results_by_nfeat:
                if verbose:
                    print("      ⚠ Nenhuma configuração rodou — pulando Fase E.")
                continue

            # Δ de ganho 2→3 e 2→4 (e-mail orientador 22:04 ponto 3: "Verificar se houve ganho")
            if verbose:
                fid_by_nfeat = {r['n_features']: r['fidelity_perc_vs_llm'] for r in results_by_nfeat}
                acc_by_nfeat = {r['n_features']: r['accuracy_perc_vs_true'] for r in results_by_nfeat}
                if 2 in fid_by_nfeat:
                    base_fid = fid_by_nfeat[2]
                    base_acc = acc_by_nfeat[2]
                    print(f"    Ganhos vs baseline (2 features) — seed={seed} | {feat_0}/{feat_1}:")
                    for nf in (3, 4):
                        if nf in fid_by_nfeat:
                            d_fid = fid_by_nfeat[nf] - base_fid
                            d_acc = acc_by_nfeat[nf] - base_acc
                            marker = (
                                "✓ GANHO" if d_acc > 0.01
                                else ("≈ neutro" if abs(d_acc) <= 0.01 else "✗ perda")
                            )
                            print(
                                f"      2→{nf} features: Δ fidelidade={d_fid:+.1%} | "
                                f"Δ acurácia_real={d_acc:+.1%} ({marker})"
                            )

            # Seleciona a MELHOR configuração pela acurácia vs rótulo real (Perceptron)
            best = max(results_by_nfeat, key=lambda r: r['accuracy_perc_vs_true'])
            if verbose:
                print(
                    f"    ★ Melhor: n_features={best['n_features']} "
                    f"(acurácia vs real = {best['accuracy_perc_vs_true']:.1%})"
                )

            # Fase E: LLM aprende com exemplos rotulados pela MELHOR métrica.
            # Os exemplos têm `best['n_features']` features no prompt.
            X_test_aug = augment_features(X_test, best['n_features'])
            y_metric_test = predict_with_metric(X_test_aug, best['centroids'], best['w_perc'])

            extra_features_names_test = None
            if best['n_features'] >= 3:
                extra_names = []
                if best['n_features'] == 3:
                    extra_names = [f"{feat_0}*{feat_1}"]
                elif best['n_features'] == 4:
                    extra_names = [f"{feat_0}²", f"{feat_1}²"]
                extra_features_names_test = extra_names

            X_train_aug = augment_features(X_train, best['n_features'])
            y_metric_train = predict_with_metric(X_train_aug, best['centroids'], best['w_perc'])

            for n_shot in n_shots_phase_e:
                # Item 10 (reunião 20/05): repetir a coleta N_REPETICOES vezes para
                # tirar média e descartar anomalias de execução única (ex.: o salto
                # 89% em n=10 → 44% no baseline em n=20 que o orientador estranhou).
                # Zero-shot não tem exemplos (prompt fixo, temp=0) → 1 repetição basta.
                n_reps_eff = 1 if n_shot == 0 else N_REPETICOES
                for rep in range(n_reps_eff):
                    if n_shot == 0:
                        examples_for_prompt = None
                        ex_indices = None
                    else:
                        # Escolhe exemplos balanceados de classes diferentes. Cada
                        # repetição sorteia um conjunto diferente (o rng local da
                        # seed avança), produzindo a variabilidade mediada nos gráficos.
                        n_shot_actual = min(n_shot, len(X_train))
                        ex_indices = rng.choice(len(X_train), size=n_shot_actual, replace=False)
                        examples_for_prompt = []
                        for ei in ex_indices:
                            row = [X_train[ei, 0], X_train[ei, 1]]
                            if best['n_features'] >= 3:
                                row.extend(X_train_aug[ei, 2:].tolist())
                            row.append(nome_classe_0 if y_metric_train[ei] == 0 else nome_classe_1)
                            examples_for_prompt.append(tuple(row))

                    extra_matrix_test = X_test_aug[:, 2:] if best['n_features'] >= 3 else None

                    t_d = time.time()
                    print(f"    [{problem_name} D seed={seed} {feat_0}/{feat_1}] n_shot={n_shot} rep={rep+1}/{n_reps_eff} | best n_feat={best['n_features']} | coletando LLM (n_test={len(X_test)})...", flush=True)
                    y_llm_test, n_mal_d = collect_llm_decisions(
                        X_test, nome_classe_0, nome_classe_1,
                        examples=examples_for_prompt, verbose=False,
                        nome_feature_0=feat_0, nome_feature_1=feat_1,
                        extra_features_matrix=extra_matrix_test,
                        extra_feature_names=extra_features_names_test,
                        label_prefix=f"[{problem_name} D seed={seed} {feat_0}/{feat_1} n={n_shot} r{rep+1}] ",
                    )
                    print(f"    [{problem_name} D seed={seed} {feat_0}/{feat_1}] n_shot={n_shot} rep={rep+1}/{n_reps_eff} coletado em {time.time()-t_d:.1f}s (malformadas={n_mal_d})", flush=True)

                    # Guarda o mapa ponto-a-ponto só da 1ª repetição (visualização)
                    if rep == 0:
                        llm_label_maps[(problem_name, seed, feat_0, feat_1, n_shot)] = {
                            'X': X_test.copy(), 'y_llm': y_llm_test.copy(),
                            'y_true': y_test.copy(), 'y_metric': y_metric_test.copy(),
                        }

                    acc_llm_vs_metric = accuracy_score(y_metric_test, y_llm_test)
                    acc_llm_vs_true = accuracy_score(y_test, y_llm_test)
                    kappa = cohen_kappa_score(y_metric_test, y_llm_test)
                    f1 = f1_score(y_metric_test, y_llm_test, zero_division=0)
                    # Item 15 (reunião 20/05): nº absoluto de amostras erradas — útil
                    # em problemas pequenos (peso×altura tem ~30 pontos de teste).
                    n_test_e = len(y_test)
                    n_err_llm_vs_true = int(np.sum(y_llm_test != y_test))
                    n_err_llm_vs_metric = int(np.sum(y_llm_test != y_metric_test))
                    # Comparação com Perceptron treinado sobre os MESMOS exemplos do expert
                    acc_perc_baseline = None
                    if examples_for_prompt is not None and n_shot >= 4:
                        try:
                            # Reconstroi (X_examples, y_examples) a partir dos índices few-shot
                            x_ex = X_train[ex_indices]
                            y_ex = y_metric_train[ex_indices]
                            if len(np.unique(y_ex)) >= 2:
                                x_ex_aug = augment_features(x_ex, best['n_features'])
                                c_ex = compute_centroids(x_ex_aug, y_ex)
                                w_baseline, _ = train_relaxed_perceptron(
                                    x_ex_aug, y_ex, c_ex,
                                    **PERCEPTRON_PARAMS,
                                    verbose=False,
                                    use_best_effort=True,
                                )
                                y_baseline = predict_with_metric(X_test_aug, c_ex, w_baseline)
                                acc_perc_baseline = accuracy_score(y_test, y_baseline)
                        except Exception as e:
                            # Caminho de DADOS (não cosmético): a acurácia do baseline-perceptron
                            # entra no CSV externo. Falha silenciosa viraria None sem rastro —
                            # registra o motivo para auditabilidade.
                            acc_perc_baseline = None
                            print(f"    ⚠ Baseline-perceptron (Fase E externa) falhou "
                                  f"[{problem_name}, n_shot={n_shot}]: {e}")

                    all_phase_e.append({
                        'problem_name': problem_name,
                        'seed': seed,
                        'rep': rep,
                        'feature_names': (feat_0, feat_1),
                        'n_features': best['n_features'],
                        'n_shot': n_shot,
                        'accuracy_llm_vs_metric': acc_llm_vs_metric,
                        'accuracy_llm_vs_true': acc_llm_vs_true,
                        'kappa_llm_vs_metric': kappa,
                        'f1_llm_vs_metric': f1,
                        'accuracy_perceptron_baseline_vs_true': acc_perc_baseline,
                        'n_test': n_test_e,
                        'n_errors_llm_vs_true': n_err_llm_vs_true,
                        'n_errors_llm_vs_metric': n_err_llm_vs_metric,
                        'n_malformed': n_mal_d,
                    })

                    if verbose:
                        baseline_str = f" | perc_baseline={acc_perc_baseline:.1%}" if acc_perc_baseline is not None else ""
                        print(
                            f"      D n_shot={n_shot} rep={rep+1}: LLM_vs_metric={acc_llm_vs_metric:.1%} | "
                            f"LLM_vs_real={acc_llm_vs_true:.1%} | erros_vs_real={n_err_llm_vs_true}/{n_test_e}{baseline_str}"
                        )

    return {
        'phase_a_results': all_phase_a,
        'phase_e_results': all_phase_e,
        'llm_label_maps': llm_label_maps,
    }








# =============================================================================
# VISUALIZAÇÕES DOS PROBLEMAS EXTERNOS (peso × altura, meia-lua)
# Os 4 plots abaixo respondem aos pedidos do e-mail 22:04 (pontos 3-5):
#   - Curva de aprendizado da Fase E com a melhor métrica
#   - Comparação 2/3/4 features (resposta visual a "houve ganho?")
#   - Fronteira de decisão por n_features (projetada em R2)
#   - LLM vs Perceptron baseline (reunião ~2110s)
# =============================================================================









def main():
    global client, async_client, MODEL_NAME, CURRENT_PROVIDER, CURRENT_TEMPERATURE
    global FEW_SHOT_SIZES, FEW_SHOT_SIZES_PHASE_E, N_REPETICOES, RANDOM_SEEDS
    global EXTRA_SEEDS_CORE, MODELS_TO_TEST, BIAS_N_SHOTS
    global DILUTION_EASY_ADDITIONS, EXAMPLE_ORDER_N_SHOTS

    # ─────────────────────────────────────────────────────────────────────
    # ARGUMENTOS DE LINHA DE COMANDO
    # --rapido (ou --smoke): execução curta de teste — few-shot [0, 4],
    # 1 repetição e apenas a seed 42. Sem a flag, roda o experimento completo.
    # ─────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Experimento de consistência decisional de LLMs via otimização inversa."
    )
    parser.add_argument(
        "--rapido", "--smoke", dest="rapido", action="store_true",
        help="Smoke test: TODOS os experimentos rodam UMA vez cada (inclusive os "
             "auxiliares, para qualquer modelo), com 1 seed (42), 1 repetição e "
             "few-shot reduzido a [0, 4]. Sem esta flag, roda o experimento completo.",
    )
    parser.add_argument(
        "--modelo", dest="modelo", default=None,
        help="Filtra MODELS_TO_TEST por substring do nome do modelo (ex.: --modelo gemini). "
             "Útil para smoke test isolado de um modelo novo: "
             "python src/dissertacao_mestrado.py --rapido --modelo gemini",
    )
    args = parser.parse_args()

    if args.modelo:
        MODELS_TO_TEST = [
            m for m in MODELS_TO_TEST if args.modelo.lower() in m[1].lower()
        ]
        if not MODELS_TO_TEST:
            raise SystemExit(
                f"--modelo '{args.modelo}' não casa com nenhum modelo em MODELS_TO_TEST"
            )
        print(f"  --modelo '{args.modelo}': rodando apenas "
              f"{', '.join(m[1] for m in MODELS_TO_TEST)}")

    modo_rapido = args.rapido
    if modo_rapido:
        # Regra do smoke test: TODOS os experimentos executam (cobertura completa
        # de código/prompts), cada um UMA vez — 1 seed, 1 repetição, e onde há
        # few-shot roda apenas o zero-shot + UM few-shot.
        FEW_SHOT_SIZES = [0, 4]
        FEW_SHOT_SIZES_PHASE_E = [0, 4]
        N_REPETICOES = 1
        RANDOM_SEEDS = [42]
        EXTRA_SEEDS_CORE = []
        BIAS_N_SHOTS = [0, 4]              # zero-shot + um few-shot
        DILUTION_EASY_ADDITIONS = [0, 4]   # 2 pontos da curva de diluição
        EXAMPLE_ORDER_N_SHOTS = [5]        # um few-shot no viés de ordem

    # Guard: valida as chaves de API de TODOS os modelos selecionados ANTES de
    # iniciar qualquer coleta — falha na hora zero, não no meio da execução paga.
    _chaves_faltando = []
    for _prov, _mod, _tmp, _scp in MODELS_TO_TEST:
        _env_var = PROVIDER_CONFIG[_prov]["api_key_env"]
        if not os.getenv(_env_var):
            _chaves_faltando.append(f"{_prov}/{_mod} → defina {_env_var}")
    if _chaves_faltando:
        raise SystemExit(
            "ERRO: chave(s) de API ausente(s) no ambiente/.env:\n  - "
            + "\n  - ".join(_chaves_faltando)
            + "\nDefina a(s) chave(s) no .env ou restrinja com --modelo <substring>."
        )

    # ─────────────────────────────────────────────────────────────────────
    # CRIAÇÃO DA PASTA DE EXECUÇÃO E INÍCIO DO LOG
    # ─────────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Sufixo no nome da pasta distingue smoke (--rapido) de execução completa:
    # ferramentas/skills que precisam de números finais filtram por "_completa".
    tipo_execucao = "smoke" if modo_rapido else "completa"
    pasta_execucao = str(BASE_DIR / f"execucao_{timestamp}_{tipo_execucao}")
    os.makedirs(pasta_execucao, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" EXPERIMENTO INICIADO: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f" Pasta de saída criada: {pasta_execucao}/")
    print(f" Todos os arquivos (imagens, CSVs e log) serão salvos nesta pasta.")
    print(f"{'='*70}\n")

    # stdout E stderr no MESMO buffer (ordem de chegada): warnings dos
    # estimadores (warnings.warn -> stderr) e tracebacks precisam constar no
    # log_execucao.txt persistido — sem isso o log diria "zero avisos" falsamente.
    tee = Tee()
    tee_err = Tee(sys.stderr, tee._buffer)
    sys.stdout = tee
    sys.stderr = tee_err

    print_section("EXPERIMENTO: CONSISTÊNCIA DECISIONAL DE LLMs VIA OTIMIZAÇÃO INVERSA (v5.0)", "=")
    if modo_rapido:
        print_section(
            "⚡ MODO RÁPIDO ATIVO (--rapido) — smoke test de cobertura completa\n"
            "   TODOS os experimentos, 1x cada | 1 seed (42) | 1 repetição | few-shot [0, 4]\n"
            "   (NÃO usar para resultados finais)",
            "="
        )
    print("  Organizado em 3 BLOCOS auto-contidos:")
    print("    BLOCO 1 — LLM como FONTE (otim. inversa em A/B/C lineares + D meia-lua)")
    print("    BLOCO 2 — LLM como APRENDIZ (Fase E no perito linear E + meia-lua F)")
    print("    BLOCO 3 — Estudo de caso REAL (peso × altura, elipse)")
    print(f"  Pasta de execução: {pasta_execucao}/")
    print(f"  Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────────────────────────────
    # FLAGS DE EXECUÇÃO ATIVAS
    # ─────────────────────────────────────────────────────────────────────
    flags_active = []
    if RUN_PHASES_ABC: flags_active.append("Fases A-C")
    if RUN_PHASE_E: flags_active.append("Fase E")
    if RUN_CLASS_ORDER_BIAS: flags_active.append("Viés de Ordem")
    if RUN_FEATURE_NAMES: flags_active.append("Nomes de Features")
    if RUN_DILUTION: flags_active.append("Diluição")
    if RUN_R3R4_EXPERIMENT: flags_active.append("Não-linearidade Implícita (R3/R4)")
    if RUN_MULTIPLE_EXPERTS: flags_active.append("Múltiplos Experts")
    if RUN_ALGORITHM_COMPARISON: flags_active.append("Comparação Algoritmos")
    if RUN_ORACLE_VALIDATION: flags_active.append("Validação Oráculo")
    if RUN_EXAMPLE_ORDER_BIAS: flags_active.append("Viés Ordem Exemplos")
    if RUN_PROMPT_VARIANTS: flags_active.append("Variantes de Prompt")
    if RUN_CLASSICAL_BASELINES: flags_active.append("Baselines Clássicos")
    if RUN_HOMEM_MULHER: flags_active.append("Peso×Altura (real, elipse)")
    if RUN_PROBLEM_MEIALUA: flags_active.append("Meia-lua (não-linear sintético)")

    models_str = '\n    '.join([f"- {p}/{m} (temp={t}, scope={s})" for p, m, t, s in MODELS_TO_TEST])

    print(f"""
    ═══════════════════════════════════════════════════
    MODELS TO TEST ({len(MODELS_TO_TEST)}):
    {models_str}
    ═══════════════════════════════════════════════════

    FLAGS ATIVAS: {', '.join(flags_active)}

    PHASES A-C:
      Few-shot sizes: {FEW_SHOT_SIZES}
      Class name variations: {len(NOMES_CLASSES)}
      Repetições (regra única): 1 coleta onde a seleção é determinística
        (zero-shot; margem em B/C; diluição; ordenações); {N_REPETICOES} onde há
        sorteio de exemplos (Fase E, externos few-shot). Flip de T=0 → auditoria.
      Seeds: {RANDOM_SEEDS} (+ extras no pipeline central do modelo full: {EXTRA_SEEDS_CORE})
      Vieses ancorados em n_shot = {BIAS_N_SHOTS}

    PHASE D:
      Few-shot sizes: {FEW_SHOT_SIZES_PHASE_E}
      Example strategies: {EXAMPLE_STRATEGIES}
      Expert configs: {[c['name'] for c in EXPERT_CONFIGS] if RUN_MULTIPLE_EXPERTS else ['aniso_x2 (original)']}
      Problem D samples: {N_SAMPLES_PROBLEM_E}
    ═══════════════════════════════════════════════════
    """)

    all_results_abc = []
    all_results_e = []
    all_results_dilution = []
    all_results_abc_alternative = []  # Para comparação de algoritmos (NNLS)
    all_results_oracle = []  # Para validação do oráculo
    all_results_oracle_meialua = []  # Item 5: oracle de aproximação da meia-lua
    all_results_example_order = []  # Para viés de ordem dos exemplos
    all_results_baselines = []  # Para baselines clássicos (k-NN, LR, SVM)
    results_r3_2feat = []  # Fidelidade da métrica R2 (2 pesos) sobre rótulos do LLM
    results_r3_3feat = []  # Fidelidade da métrica R3 (3 pesos, x3=x1*x2) sobre os mesmos rótulos
    results_r3_4feat = []  # Fidelidade da métrica R4 (4 pesos, +x1², x2²) — sanity check elipse em A/B/C
    # Cache de dados da Fase A para gráficos de erros
    phase_a_data_for_plots = {}
    # Dados detalhados por seed para visualizações abrangentes
    seed_detailed_data = {}

    for model_idx, (provider, model_name, temperature, model_scope) in enumerate(MODELS_TO_TEST):
        client = get_client(provider)
        async_client = get_async_client(provider)
        MODEL_NAME = model_name
        CURRENT_PROVIDER = provider
        CURRENT_TEMPERATURE = temperature

        # Seeds do modelo: as extras do pipeline central só para o modelo "full".
        seeds_do_modelo = RANDOM_SEEDS + (EXTRA_SEEDS_CORE if model_scope == "full" else [])

        for seed_idx, seed in enumerate(seeds_do_modelo):
            # Seeds extras rodam SÓ o pipeline central (Fases A-C + Fase E perito
            # principal); experimentos auxiliares ficam nas 3 RANDOM_SEEDS do
            # modelo "full".
            is_seed_extra = seed_idx >= len(RANDOM_SEEDS)
            # No --rapido, TODOS os modelos rodam todos os experimentos (smoke test
            # de cobertura completa); na execução normal, só o modelo "full".
            run_aux = (model_scope == "full" or modo_rapido) and not is_seed_extra

            # Checkpoint das interações coletadas até aqui (crash a qualquer
            # momento perde no máximo uma iteração de seed, não a execução inteira)
            checkpoint_interactions(pasta_execucao)

            print_section(
                f"BLOCO 1 — LLM como FONTE | MODEL {model_idx + 1}/{len(MODELS_TO_TEST)}: "
                f"{provider}/{model_name} (temp={temperature}, scope={model_scope}) | "
                f"SEED {seed_idx + 1}/{len(seeds_do_modelo)}: {seed}"
                + (" [extra: só pipeline central]" if is_seed_extra else ""),
                "═"
            )

            np.random.seed(seed)

            # ─── Passo 1: Geração dos dados sintéticos ─────────────────────────
            print(f"\n[PASSO 1] Gerando conjuntos de dados sintéticos com semente aleatória {seed}...", flush=True)

            X_a, y_a = create_problem_a(N_SAMPLES_PROBLEM_A, seed)
            X_b, y_b = create_problem_b(N_SAMPLES_PROBLEM_B, seed + 1)
            X_c, y_c = create_problem_c(N_SAMPLES_PROBLEM_C, seed + 2)
            X_e, y_e = create_problem_e_expert(N_SAMPLES_PROBLEM_E, seed + 3)

            print(f"  Dados gerados: A={len(X_a)}, B={len(X_b)}, C={len(X_c)}, D={len(X_e)}")

            # Salva dados sintéticos na pasta de execução
            seed_data_dir = os.path.join(pasta_execucao, f"dados_sinteticos_seed{seed}")
            os.makedirs(seed_data_dir, exist_ok=True)
            for name, X_data, y_data in [("A", X_a, y_a), ("B", X_b, y_b), ("C", X_c, y_c), ("D", X_e, y_e)]:
                df = pd.DataFrame({"x1": X_data[:, 0], "x2": X_data[:, 1], "y": y_data})
                df.to_csv(os.path.join(seed_data_dir, f"problem_{name}.csv"), index=False)
            print(f"  Dados salvos em: {seed_data_dir}/")

            if seed_idx == 0 and model_idx == 0:
                print(f"\n[PASSO 2] Gerando visualizações iniciais...")

                visualize_all_problems(X_a, y_a, X_b, y_b, X_c, y_c,
                                       filename=os.path.join(pasta_execucao, "bloco1_01_problemas_lineares.png"))
                print(f"  Imagem salva: bloco1_01_problemas_lineares.png")

                y_expert_e_viz = expert_classify(X_e, EXPERT_W, EXPERT_CENTROIDS)
                visualize_problem_e_with_expert(
                    X_e, y_e, y_expert_e_viz, EXPERT_W, EXPERT_CENTROIDS,
                    filename=os.path.join(pasta_execucao, "bloco2_01_problema_e_expert.png")
                )
                print(f"  Imagem salva: bloco2_01_problema_e_expert.png")

                plot_phase_e_example_locations(
                    X_e, y_expert_e_viz, EXPERT_W, EXPERT_CENTROIDS,
                    n_examples=10, random_state=seed,
                    filename=os.path.join(pasta_execucao, "bloco2_03_phase_e_strategies.png")
                )
                print(f"  Imagem salva: bloco2_03_phase_e_strategies.png")

            X_train_a, y_train_a = X_a, y_a

            # ═══════════════════════════════════════════════════════════════
            # FASES A-C (Experimento original de consistência)
            # ═══════════════════════════════════════════════════════════════

            phase_a_cache = {}

            if RUN_PHASES_ABC:

                print(f"\n[PASSO 3] Iniciando Fases A-C...")
                print(f"  Modelo: {provider}/{model_name} | Seeds: {seed} | Few-shot: {FEW_SHOT_SIZES}")

                print(f"\n>>> Fases A-C: Experimento Principal", flush=True)
                for n_shot in FEW_SHOT_SIZES:
                    # Seleção de exemplos das Fases B/C é DETERMINÍSTICA (maior margem)
                    # → repetição não sorteia nada; roda-se 1 coleta (ver reps_para).
                    n_reps_abc = reps_para(n_shot, sorteio_estocastico=False)
                    for rep in range(n_reps_abc):
                        print(f"  [ABC] n_shot={n_shot}, rep={rep+1}/{n_reps_abc}, seed={seed}", flush=True)
                        verbose = (rep == 0 and n_shot == FEW_SHOT_SIZES[0] and seed_idx == 0)
                        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail_bc = \
                            _run_phase_abc_experiment(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                n_shot, "A", "B", rep, provider, model_name,
                                temperature, seed, seed_idx, phase_a_cache, verbose
                            )
                        all_results_abc.append(result)

                        # Guarda dados para plots de erros da Fase A e visualizações detalhadas
                        # (TODOS os modelos alimentam as visualizações por seed —
                        # dicts chaveados por (provider, model_name, seed))
                        if n_shot == 0 and rep == 0 and learned_metric is not None:
                            cache_key_ab = (provider, model_name, seed, "A", "B", "default")
                            if cache_key_ab in phase_a_cache:
                                cached = phase_a_cache[cache_key_ab]
                                y_metric_a = predict_with_metric(X_train_a, cached['metric'].centroids, cached['metric'].w)
                                phase_a_data_for_plots[(provider, model_name, seed)] = {
                                    'X': X_train_a, 'y_llm': cached['y_llm'],
                                    'y_metric': y_metric_a,
                                    'w': cached['metric'].w,
                                    'centroids': cached['metric'].centroids,
                                }
                                seed_detailed_data[(provider, model_name, seed)] = {
                                    'X_a': X_train_a, 'y_gt_a': y_train_a,
                                    'X_b': X_b, 'y_gt_b': y_b,
                                    'X_c': X_c, 'y_gt_c': y_c,
                                    'X_e': X_e, 'y_gt_e': y_e,
                                    'y_llm_a': cached['y_llm'],
                                    'y_metric_a': y_metric_a,
                                    'learned_metric': learned_metric,
                                    'y_llm_b': detail_bc.get('y_llm_b'),
                                    'y_llm_c': detail_bc.get('y_llm_c'),
                                    'y_metric_b': detail_bc.get('y_metric_b'),
                                    'y_metric_c': detail_bc.get('y_metric_c'),
                                    'metrics_b': detail_bc.get('metrics_b'),
                                    'metrics_c': detail_bc.get('metrics_c'),
                                }

                print(f"  ✓ Experimento principal concluído ({len(FEW_SHOT_SIZES)} n_shots × 1 coleta)", flush=True)

                # Variações de nomes de classe — ancoradas em BIAS_N_SHOTS ({0, 10})
                # para efeitos comparáveis com as demais análises de viés.
                if run_aux:
                    print(f"\n>>> Fases A-C: Variações de nomes de classe (n_shot={BIAS_N_SHOTS})", flush=True)
                    for nome_0, nome_1 in NOMES_CLASSES[1:]:
                        for n_shot_bias in BIAS_N_SHOTS:
                            print(f"  [ABC-Nomes] classes=({nome_0},{nome_1}), n_shot={n_shot_bias}, seed={seed}", flush=True)
                            verbose = (n_shot_bias == BIAS_N_SHOTS[0] and seed_idx == 0)
                            result, *_ = _run_phase_abc_experiment(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                n_shot_bias, nome_0, nome_1, 0, provider, model_name,
                                temperature, seed, seed_idx, phase_a_cache, verbose
                            )
                            all_results_abc.append(result)

                    print(f"  ✓ Variações de nomes de classe concluídas", flush=True)

                # ═══════════════════════════════════════════════════════
                # TESTE DE INVERSÃO DE ORDEM DAS CLASSES
                # ═══════════════════════════════════════════════════════
                if RUN_CLASS_ORDER_BIAS and run_aux:
                    print(f"\n>>> Fases A-C: Teste de inversão de ordem das classes", flush=True)
                    for nome_0, nome_1 in NOMES_CLASSES_INVERTIDAS:
                        for n_shot_bias in BIAS_N_SHOTS:
                            print(f"  [ABC-Inversão] classes=({nome_0},{nome_1}), n_shot={n_shot_bias}, seed={seed}", flush=True)
                            verbose = (n_shot_bias == BIAS_N_SHOTS[0] and seed_idx == 0)
                            result, *_ = _run_phase_abc_experiment(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                n_shot_bias, nome_0, nome_1, 0, provider, model_name,
                                temperature, seed, seed_idx, phase_a_cache, verbose
                            )
                            all_results_abc.append(result)
                    print(f"  ✓ Teste de inversão de ordem das classes concluído", flush=True)

                # ═══════════════════════════════════════════════════════
                # TESTE DE VARIANTES DE PROMPT (Sensibilidade ao prompt)
                # ═══════════════════════════════════════════════════════
                if RUN_PROMPT_VARIANTS and run_aux:
                    print(f"\n>>> Fases A-C: Teste de variantes de prompt", flush=True)
                    for variant_name in PROMPT_VARIANTS:
                        if variant_name == "default":
                            continue  # Já testado no loop principal
                        for n_shot_bias in BIAS_N_SHOTS:
                            print(f"  [ABC-Prompt] variant={variant_name}, n_shot={n_shot_bias}, seed={seed}", flush=True)
                            verbose = (n_shot_bias == BIAS_N_SHOTS[0] and seed_idx == 0)
                            result, *_ = _run_phase_abc_experiment(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                n_shot_bias, "A", "B", 0, provider, model_name,
                                temperature, seed, seed_idx, phase_a_cache, verbose,
                                prompt_variant=variant_name
                            )
                            all_results_abc.append(result)
                    print(f"  ✓ Teste de variantes de prompt concluído", flush=True)

                # ═══════════════════════════════════════════════════════
                # TESTE DE NOMES SEMÂNTICOS NAS FEATURES
                # ═══════════════════════════════════════════════════════
                if RUN_FEATURE_NAMES and run_aux:
                    print(f"\n>>> Fases A-C: Teste de nomes semânticos nas features", flush=True)
                    for feat_0, feat_1 in NOMES_FEATURES[1:]:  # Pula o neutro (já testado)
                        print(f"    Features: {feat_0}/{feat_1}")
                        # Classe A/B, zero-shot; roda em TODAS as RANDOM_SEEDS
                        # (antes: apenas seed_idx==0 — célula fraca do grid, corrigida).
                        if run_aux:
                            # Coleta decisões com nomes de features alterados
                            y_llm_feat, n_malf = collect_llm_decisions(
                                X_train_a, "A", "B",
                                examples=None, verbose=True,
                                label_prefix=f"[Features {feat_0}/{feat_1}] ",
                                nome_feature_0=feat_0, nome_feature_1=feat_1
                            )
                            llm_acc_feat = accuracy_score(y_train_a, y_llm_feat)
                            if len(np.unique(y_llm_feat)) >= 2:
                                centroids_feat = compute_centroids(X_train_a, y_llm_feat)

                                # Perceptron
                                w_feat, gamma_feat = train_relaxed_perceptron(
                                    X_train_a, y_llm_feat, centroids_feat,
                                    **PERCEPTRON_PARAMS,
                                    verbose=False,
                                    use_best_effort=True
                                )
                                y_metric_feat = predict_with_metric(X_train_a, centroids_feat, w_feat)
                                fid_feat = accuracy_score(y_llm_feat, y_metric_feat)

                                # NNLS (validação cruzada)
                                w_feat_nnls, _ = train_least_squares_inverse(
                                    X_train_a, y_llm_feat, centroids_feat, verbose=False
                                )
                                y_metric_feat_nnls = predict_with_metric(X_train_a, centroids_feat, w_feat_nnls)
                                fid_feat_nnls = accuracy_score(y_llm_feat, y_metric_feat_nnls)

                                # Testa consistência nos problemas B e C
                                y_llm_b_feat, n_malf_b = collect_llm_decisions(
                                    X_b, "A", "B", examples=None, verbose=False,
                                    label_prefix=f"[Features {feat_0}/{feat_1} B] ",
                                    nome_feature_0=feat_0, nome_feature_1=feat_1
                                )
                                # Perceptron B/C
                                y_metric_b_feat = predict_with_metric(
                                    X_b, centroids_feat, w_feat)
                                cons_b_feat = compute_consistency_metrics(y_llm_b_feat, y_metric_b_feat)

                                y_llm_c_feat, n_malf_c = collect_llm_decisions(
                                    X_c, "A", "B", examples=None, verbose=False,
                                    label_prefix=f"[Features {feat_0}/{feat_1} C] ",
                                    nome_feature_0=feat_0, nome_feature_1=feat_1
                                )
                                y_metric_c_feat = predict_with_metric(
                                    X_c, centroids_feat, w_feat)
                                cons_c_feat = compute_consistency_metrics(y_llm_c_feat, y_metric_c_feat)

                                # NNLS B/C
                                y_metric_b_feat_nnls = predict_with_metric(X_b, centroids_feat, w_feat_nnls)
                                cons_b_feat_nnls = compute_consistency_metrics(y_llm_b_feat, y_metric_b_feat_nnls)
                                y_metric_c_feat_nnls = predict_with_metric(X_c, centroids_feat, w_feat_nnls)
                                cons_c_feat_nnls = compute_consistency_metrics(y_llm_c_feat, y_metric_c_feat_nnls)

                                cos_sim_feat = np.dot(w_feat, w_feat_nnls) / (np.linalg.norm(w_feat) * np.linalg.norm(w_feat_nnls) + 1e-9)
                                print(f"    [Perceptron] Ŵ_LLM com features {feat_0}/{feat_1}: [{w_feat[0]:.4f}, {w_feat[1]:.4f}]")
                                print(f"    [NNLS]       Ŵ_LLM com features {feat_0}/{feat_1}: [{w_feat_nnls[0]:.4f}, {w_feat_nnls[1]:.4f}]")
                                print(f"    Similaridade cosseno (Perceptron vs NNLS): {cos_sim_feat:.4f}")
                                print(f"    [Perceptron] Fidelidade: {fid_feat:.1%}, Cons B: {cons_b_feat.accuracy:.1%}, Cons C: {cons_c_feat.accuracy:.1%}")
                                print(f"    [NNLS]       Fidelidade: {fid_feat_nnls:.1%}, Cons B: {cons_b_feat_nnls.accuracy:.1%}, Cons C: {cons_c_feat_nnls.accuracy:.1%}")

                                # Armazena resultado completo
                                result_feat = ResultadoExperimento(
                                    provider=provider, model_name=model_name,
                                    temperature=temperature, random_seed=seed,
                                    n_shot=0, nomes_classes=("A", "B"), repeticao=0,
                                    fidelidade_problema_a=fid_feat,
                                    acuracia_llm_vs_gt_problema_a=llm_acc_feat,
                                    consistencia_problema_b=cons_b_feat.accuracy,
                                    kappa_problema_b=cons_b_feat.cohen_kappa,
                                    f1_problema_b=cons_b_feat.f1_score,
                                    acuracia_llm_vs_gt_problema_b=accuracy_score(y_b, y_llm_b_feat),
                                    acuracia_metrica_vs_gt_problema_b=accuracy_score(y_b, y_metric_b_feat),
                                    consistencia_problema_c=cons_c_feat.accuracy,
                                    kappa_problema_c=cons_c_feat.cohen_kappa,
                                    f1_problema_c=cons_c_feat.f1_score,
                                    acuracia_llm_vs_gt_problema_c=accuracy_score(y_c, y_llm_c_feat),
                                    acuracia_metrica_vs_gt_problema_c=accuracy_score(y_c, y_metric_c_feat),
                                    w_aprendido=w_feat, gamma_otimo=gamma_feat,
                                    n_classe_0_problema_a=int(np.sum(y_llm_feat == 0)),
                                    n_classe_1_problema_a=int(np.sum(y_llm_feat == 1)),
                                    n_classe_0_problema_b=int(np.sum(y_llm_b_feat == 0)),
                                    n_classe_1_problema_b=int(np.sum(y_llm_b_feat == 1)),
                                    n_classe_0_problema_c=int(np.sum(y_llm_c_feat == 0)),
                                    n_classe_1_problema_c=int(np.sum(y_llm_c_feat == 1)),
                                    n_disagreements_b=cons_b_feat.n_disagreements,
                                    n_disagreements_c=cons_c_feat.n_disagreements,
                                    n_malformed_responses=n_malf + n_malf_b + n_malf_c,
                                    w_direction=w_feat / np.linalg.norm(w_feat) if np.linalg.norm(w_feat) > 0 else w_feat,
                                    w_cosine_sim_nnls=float(cos_sim_feat),
                                    feature_names=(feat_0, feat_1),
                                )
                                all_results_abc.append(result_feat)
                            else:
                                print(f"    ⚠ Classe única — pulando {feat_0}/{feat_1}")
                    print(f"  ✓ Teste de nomes de features concluído", flush=True)

                # ═══════════════════════════════════════════════════════
                # COMPARAÇÃO DE ALGORITMOS
                # ═══════════════════════════════════════════════════════
                if RUN_ALGORITHM_COMPARISON and run_aux:
                    print(f"\n>>> Comparação de Algoritmos: Perceptron vs NNLS", flush=True)
                    cache_key_ab = (provider, model_name, seed, "A", "B", "default")
                    if cache_key_ab in phase_a_cache:
                        cached = phase_a_cache[cache_key_ab]
                        y_llm_for_alt = cached['y_llm']
                        if len(np.unique(y_llm_for_alt)) >= 2:
                            centroids_alt = compute_centroids(X_train_a, y_llm_for_alt)
                            w_alt, gamma_alt = train_least_squares_inverse(
                                X_train_a, y_llm_for_alt, centroids_alt, verbose=True
                            )
                            y_metric_alt = predict_with_metric(X_train_a, centroids_alt, w_alt)
                            fid_alt = accuracy_score(y_llm_for_alt, y_metric_alt)

                            # Cria LearnedMetric com W do NNLS para rodar Fases B e C
                            learned_metric_alt = LearnedMetric(
                                w=w_alt, centroids=centroids_alt,
                                gamma=gamma_alt, source_problem="Problem_A"
                            )

                            # Fase B com W do NNLS
                            print(f"    [NNLS] Rodando Fase B (Problema B)...", flush=True)
                            metrics_b_alt, y_llm_b_alt, y_pred_b_alt, llm_acc_b_alt, metric_acc_b_alt, n_malf_b_alt, euc_b_alt, _bl_b_alt, _Xt_b_alt = \
                                phase_consistency_test(
                                    X_b, y_b, learned_metric_alt,
                                    "A", "B", 0, "FASE B — NNLS", verbose=(seed_idx == 0)
                                )

                            # Fase C com W do NNLS
                            print(f"    [NNLS] Rodando Fase C (Problema C)...", flush=True)
                            metrics_c_alt, y_llm_c_alt, y_pred_c_alt, llm_acc_c_alt, metric_acc_c_alt, n_malf_c_alt, euc_c_alt, _bl_c_alt, _Xt_c_alt = \
                                phase_consistency_test(
                                    X_c, y_c, learned_metric_alt,
                                    "A", "B", 0, "FASE C — NNLS", verbose=(seed_idx == 0)
                                )

                            result_alt = ResultadoExperimento(
                                provider=provider, model_name=model_name, temperature=temperature,
                                random_seed=seed, n_shot=0,
                                nomes_classes=("A", "B"), repeticao=0,
                                fidelidade_problema_a=fid_alt,
                                acuracia_llm_vs_gt_problema_a=cached['llm_acc'],
                                consistencia_problema_b=metrics_b_alt.accuracy,
                                kappa_problema_b=metrics_b_alt.cohen_kappa,
                                f1_problema_b=metrics_b_alt.f1_score,
                                acuracia_llm_vs_gt_problema_b=llm_acc_b_alt,
                                acuracia_metrica_vs_gt_problema_b=metric_acc_b_alt,
                                consistencia_problema_c=metrics_c_alt.accuracy,
                                kappa_problema_c=metrics_c_alt.cohen_kappa,
                                f1_problema_c=metrics_c_alt.f1_score,
                                acuracia_llm_vs_gt_problema_c=llm_acc_c_alt,
                                acuracia_metrica_vs_gt_problema_c=metric_acc_c_alt,
                                w_aprendido=w_alt, gamma_otimo=gamma_alt,
                                n_classe_0_problema_a=int(np.sum(y_llm_for_alt == 0)),
                                n_classe_1_problema_a=int(np.sum(y_llm_for_alt == 1)),
                                n_classe_0_problema_b=int(np.sum(y_llm_b_alt == 0)),
                                n_classe_1_problema_b=int(np.sum(y_llm_b_alt == 1)),
                                n_classe_0_problema_c=int(np.sum(y_llm_c_alt == 0)),
                                n_classe_1_problema_c=int(np.sum(y_llm_c_alt == 1)),
                                n_disagreements_b=metrics_b_alt.n_disagreements,
                                n_disagreements_c=metrics_c_alt.n_disagreements,
                                n_malformed_responses=n_malf_b_alt + n_malf_c_alt,
                                consistencia_euclidiana_problema_b=euc_b_alt,
                                consistencia_euclidiana_problema_c=euc_c_alt,
                                diagonal_limitation_flag=1 if metrics_b_alt.accuracy <= euc_b_alt + 0.01 else 0,
                                w_direction=w_alt / np.linalg.norm(w_alt) if np.linalg.norm(w_alt) > 0 else w_alt,
                                w_cosine_sim_nnls=0.0,  # Este JÁ é o NNLS, não há segundo algoritmo
                            )
                            all_results_abc_alternative.append(result_alt)

                            # Guarda W do NNLS para visualizações
                            if (provider, model_name, seed) in seed_detailed_data:
                                seed_detailed_data[(provider, model_name, seed)]['w_nnls'] = w_alt
                                seed_detailed_data[(provider, model_name, seed)]['centroids_nnls'] = centroids_alt

                            print(f"    --- Comparação Fase A (Fidelidade) ---", flush=True)
                            print(f"    Perceptron: W=[{cached['metric'].w[0]:.4f}, {cached['metric'].w[1]:.4f}], Fidelidade={cached['fidelity']:.1%}")
                            print(f"    NNLS:       W=[{w_alt[0]:.4f}, {w_alt[1]:.4f}], Fidelidade={fid_alt:.1%}")
                            print(f"    --- Comparação Fase B (Consistência) ---", flush=True)
                            perc_b = [r for r in all_results_abc
                                      if r.random_seed == seed and r.n_shot == 0
                                      and r.nomes_classes == ("A", "B") and r.repeticao == 0]
                            if perc_b:
                                print(f"    Perceptron: Consistência B={perc_b[0].consistencia_problema_b:.1%}, Kappa={perc_b[0].kappa_problema_b:.3f}")
                            print(f"    NNLS:       Consistência B={metrics_b_alt.accuracy:.1%}, Kappa={metrics_b_alt.cohen_kappa:.3f}")
                            print(f"    --- Comparação Fase C (Consistência) ---", flush=True)
                            if perc_b:
                                print(f"    Perceptron: Consistência C={perc_b[0].consistencia_problema_c:.1%}, Kappa={perc_b[0].kappa_problema_c:.3f}")
                            print(f"    NNLS:       Consistência C={metrics_c_alt.accuracy:.1%}, Kappa={metrics_c_alt.cohen_kappa:.3f}")
                    print(f"  ✓ Comparação de algoritmos concluída", flush=True)

            # ═══════════════════════════════════════════════════════════════
            # VALIDAÇÃO DO ORÁCULO: ALGORITMOS RECUPERAM W CONHECIDO?
            # ═══════════════════════════════════════════════════════════════

            # Oráculo é INDEPENDENTE do modelo (0 chamadas LLM) — roda uma vez, no
            # primeiro modelo, para todas as seeds (inclusive extras: de graça).
            if RUN_ORACLE_VALIDATION and model_idx == 0:
                print(f"\n>>> Validação do Oráculo: Algoritmos recuperam W conhecido?", flush=True)
                oracle_results = run_oracle_validation(
                    X_a, y_a, X_b, y_b, X_c, y_c, X_e, y_e,
                    EXPERT_CONFIGS,
                    random_seed=seed,
                    verbose=(seed_idx == 0),
                )
                all_results_oracle.extend(oracle_results)
                print(f"  ✓ Validação do oráculo concluída para seed={seed}", flush=True)

                # Item 5 (reunião 20/05): oracle de APROXIMAÇÃO da meia-lua —
                # mostra que existe W (em espaço aumentado) que aproxima a fronteira
                # não-linear, com fidelidade crescente em 2→3→4 features.
                if RUN_PROBLEM_MEIALUA:
                    X_ml_o, y_ml_o = create_problem_d_meialua(
                        n_samples=N_SAMPLES_PROBLEM_A, random_state=seed,
                    )
                    ml_oracle = run_oracle_meialua(
                        X_ml_o, y_ml_o, random_seed=seed, verbose=(seed_idx == 0),
                    )
                    all_results_oracle_meialua.extend(ml_oracle)

            # ═══════════════════════════════════════════════════════════════
            # FASE E: LLM COMO APRENDIZ
            # ═══════════════════════════════════════════════════════════════

            if RUN_PHASE_E:
                # Múltiplos peritos: só no grid completo (modelo "full", seeds base).
                # Modelos "core" e seeds extras rodam apenas o perito principal.
                usa_multiplos_experts = RUN_MULTIPLE_EXPERTS and run_aux
                expert_configs_to_run = EXPERT_CONFIGS if usa_multiplos_experts else [EXPERT_CONFIGS[0]]

                for expert_cfg in expert_configs_to_run:
                    expert_w = expert_cfg["w"]
                    expert_name = expert_cfg["name"]
                    expert_centroids = EXPERT_CENTROIDS

                    print(f"\n[PASSO 4] Fase E — Expert: {expert_name} ({expert_cfg['desc']})")
                    print(f"  W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
                    print_section(f"BLOCO 2 — LLM como APRENDIZ | FASE E: Expert {expert_name}", "═")

                    total_e_combos = len(FEW_SHOT_SIZES_PHASE_E) * len(EXAMPLE_STRATEGIES)
                    combo_count = 0
                    for n_shot_e in FEW_SHOT_SIZES_PHASE_E:
                        for strategy in EXAMPLE_STRATEGIES:
                            combo_count += 1
                            print(f"  [Fase E] Expert={expert_name}, n_shot={n_shot_e}, strategy={strategy} ({combo_count}/{total_e_combos}), seed={seed}", flush=True)
                            if n_shot_e == 0 and strategy != EXAMPLE_STRATEGIES[0]:
                                base_results = all_results_e[-reps_para(0):]
                                for rep, prev_result in enumerate(base_results):
                                    dup_result = ResultadoPhaseEExperimento(
                                        provider=prev_result.provider,
                                        model_name=prev_result.model_name,
                                        temperature=prev_result.temperature,
                                        random_seed=prev_result.random_seed,
                                        n_shot=0,
                                        example_strategy=strategy,
                                        nomes_classes=prev_result.nomes_classes,
                                        repeticao=rep,
                                        accuracy_llm_vs_expert=prev_result.accuracy_llm_vs_expert,
                                        kappa_llm_vs_expert=prev_result.kappa_llm_vs_expert,
                                        f1_llm_vs_expert=prev_result.f1_llm_vs_expert,
                                        accuracy_expert_vs_gt=prev_result.accuracy_expert_vs_gt,
                                        accuracy_llm_vs_gt=prev_result.accuracy_llm_vs_gt,
                                        n_classe_0_expert=prev_result.n_classe_0_expert,
                                        n_classe_1_expert=prev_result.n_classe_1_expert,
                                        n_classe_0_llm=prev_result.n_classe_0_llm,
                                        n_classe_1_llm=prev_result.n_classe_1_llm,
                                        n_disagreements=prev_result.n_disagreements,
                                        n_total_test=prev_result.n_total_test,
                                        n_malformed_responses=prev_result.n_malformed_responses,
                                        expert_w=prev_result.expert_w.copy(),
                                        expert_name=expert_name,
                                    )
                                    all_results_e.append(dup_result)
                                continue

                            # Zero-shot: 1 coleta (não há exemplos a sortear). Few-shot:
                            # só a estratégia "random" consome o sorteio (random_state=
                            # seed+rep); easy/hard/mixed são determinísticas por margem —
                            # repetir recoletaria o MESMO prompt (o flip de T=0 já é
                            # quantificado pela auditoria offline). Regra única: reps_para.
                            reps_e = reps_para(n_shot_e, sorteio_estocastico=(strategy == "random"))
                            for rep in range(reps_e):
                                is_verbose = (
                                    rep == 0 and seed_idx == 0 and
                                    (n_shot_e in [0, FEW_SHOT_SIZES_PHASE_E[-1]])
                                )

                                result_e = phase_e_llm_as_learner(
                                    X_e.copy(), y_e.copy(),
                                    expert_w=expert_w,
                                    expert_centroids=expert_centroids,
                                    n_shot=n_shot_e,
                                    strategy=strategy,
                                    nome_classe_0="A",
                                    nome_classe_1="B",
                                    provider=provider,
                                    model_name=model_name,
                                    temperature=temperature,
                                    random_seed=seed,
                                    repeticao=rep,
                                    verbose=is_verbose
                                )
                                result_e.expert_name = expert_name
                                all_results_e.append(result_e)

                print(f"  ✓ Fase E concluída para seed={seed}", flush=True)

                # ═══════════════════════════════════════════════════════
                # BASELINES CLÁSSICOS (k-NN, LR, SVM)
                # ═══════════════════════════════════════════════════════
                # Baselines clássicos são INDEPENDENTES do LLM (treinam nos exemplos
                # rotulados pelo perito) — rodar uma vez, no primeiro modelo, evita
                # linhas duplicadas nos CSVs/plots com múltiplos modelos.
                if RUN_CLASSICAL_BASELINES and model_idx == 0:
                    print(f"\n>>> Baselines clássicos na Fase E", flush=True)
                    for expert_cfg_bl in (EXPERT_CONFIGS if RUN_MULTIPLE_EXPERTS else [EXPERT_CONFIGS[0]]):
                        expert_w_bl = expert_cfg_bl["w"]
                        expert_name_bl = expert_cfg_bl["name"]
                        y_expert_bl = expert_classify(X_e, expert_w_bl, EXPERT_CENTROIDS)
                        expert_acc_bl = accuracy_score(y_e, y_expert_bl)

                        for n_shot_bl in FEW_SHOT_SIZES_PHASE_E:
                            if n_shot_bl == 0:
                                continue  # Baselines precisam de dados de treino
                            for strategy_bl in EXAMPLE_STRATEGIES:
                                examples_bl, indices_bl = select_examples_by_strategy(
                                    X_e, y_expert_bl, expert_w_bl, EXPERT_CENTROIDS,
                                    n_examples=n_shot_bl, strategy=strategy_bl,
                                    nome_classe_0="A", nome_classe_1="B",
                                    random_state=seed, verbose=False
                                )

                                # Monta X_train e y_train a partir dos exemplos
                                X_train_bl = X_e[indices_bl]
                                y_train_bl = y_expert_bl[indices_bl]

                                # Monta conjunto de teste (exclui exemplos)
                                test_mask_bl = np.ones(len(X_e), dtype=bool)
                                test_mask_bl[indices_bl] = False
                                X_test_bl = X_e[test_mask_bl]
                                y_expert_test_bl = y_expert_bl[test_mask_bl]
                                y_gt_test_bl = y_e[test_mask_bl]

                                is_verbose_bl = (seed_idx == 0 and n_shot_bl == FEW_SHOT_SIZES_PHASE_E[-1]
                                                 and strategy_bl == "mixed")
                                if is_verbose_bl:
                                    print(f"  [Baselines] expert={expert_name_bl}, n_shot={n_shot_bl}, "
                                          f"strategy={strategy_bl}, seed={seed}", flush=True)

                                runner = ClassicalBaselineRunner(verbose=is_verbose_bl)
                                baseline_results = runner.run(
                                    X_train_bl, y_train_bl,
                                    X_test_bl, y_expert_test_bl, y_gt_test_bl,
                                    n_shot=n_shot_bl
                                )

                                for clf_name, metrics in baseline_results.items():
                                    all_results_baselines.append({
                                        'provider': 'classical',
                                        'model': clf_name,
                                        'random_seed': seed,
                                        'n_shot': n_shot_bl,
                                        'example_strategy': strategy_bl,
                                        'expert_name': expert_name_bl,
                                        'accuracy_vs_expert': metrics['accuracy_vs_expert'],
                                        'kappa_vs_expert': metrics['kappa_vs_expert'],
                                        'f1_vs_expert': metrics['f1_vs_expert'],
                                        'accuracy_vs_gt': metrics['accuracy_vs_gt'],
                                        'accuracy_expert_vs_gt': expert_acc_bl,
                                        'n_total_test': len(X_test_bl),
                                    })

                    print(f"  ✓ Baselines clássicos concluídos para seed={seed}", flush=True)

                # ═══════════════════════════════════════════════════════
                # EXPERIMENTO DE DILUIÇÃO
                # ═══════════════════════════════════════════════════════
                if RUN_DILUTION and run_aux:
                    print(f"\n>>> Experimento de Diluição: 3 hard fixos + N easy progressivos", flush=True)
                    y_expert_dilution = expert_classify(X_e, EXPERT_W, EXPERT_CENTROIDS)
                    n_hard_fixed = 4  # 2 por classe (arredondado para par)
                    easy_additions = DILUTION_EASY_ADDITIONS  # N easy adicionados

                    for dil_idx, n_easy in enumerate(easy_additions):
                        n_total = n_hard_fixed + n_easy
                        print(f"  [Diluição] {n_hard_fixed} hard + {n_easy} easy = {n_total} total ({dil_idx+1}/{len(easy_additions)})", flush=True)

                        examples_dil, selected_dil = select_examples_dilution(
                            X_e, y_expert_dilution, EXPERT_W, EXPERT_CENTROIDS,
                            n_hard_fixed=n_hard_fixed, n_easy_added=n_easy,
                            nome_classe_0="A", nome_classe_1="B",
                            random_state=seed, verbose=True
                        )

                        # Monta conjunto de teste
                        test_mask = np.ones(len(X_e), dtype=bool)
                        test_mask[selected_dil] = False
                        X_test_dil = X_e[test_mask]
                        y_expert_test_dil = y_expert_dilution[test_mask]
                        y_gt_test_dil = y_e[test_mask]

                        # Seleção da diluição é determinística (margem) → 1 coleta.
                        for rep in range(reps_para(n_total, sorteio_estocastico=False)):
                            y_llm_dil, n_malf_dil = collect_llm_decisions(
                                X_test_dil, "A", "B",
                                examples=examples_dil if n_total > 0 else None,
                                verbose=(rep == 0 and seed_idx == 0),
                                label_prefix=f"[Diluição {n_total}ex] "
                            )
                            cons_dil = compute_consistency_metrics(y_llm_dil, y_expert_test_dil)
                            llm_acc_dil = accuracy_score(y_gt_test_dil, y_llm_dil)
                            expert_acc_dil = accuracy_score(y_gt_test_dil, y_expert_test_dil)

                            result_dil = ResultadoPhaseEExperimento(
                                provider=provider, model_name=model_name,
                                temperature=temperature, random_seed=seed,
                                n_shot=n_total,
                                example_strategy=f"dilution_{n_hard_fixed}hard_{n_easy}easy",
                                nomes_classes=("A", "B"), repeticao=rep,
                                accuracy_llm_vs_expert=cons_dil.accuracy,
                                kappa_llm_vs_expert=cons_dil.cohen_kappa,
                                f1_llm_vs_expert=cons_dil.f1_score,
                                accuracy_expert_vs_gt=expert_acc_dil,
                                accuracy_llm_vs_gt=llm_acc_dil,
                                n_classe_0_expert=int(np.sum(y_expert_test_dil == 0)),
                                n_classe_1_expert=int(np.sum(y_expert_test_dil == 1)),
                                n_classe_0_llm=int(np.sum(y_llm_dil == 0)),
                                n_classe_1_llm=int(np.sum(y_llm_dil == 1)),
                                n_disagreements=cons_dil.n_disagreements,
                                n_total_test=len(X_test_dil),
                                n_malformed_responses=n_malf_dil,
                                expert_w=EXPERT_W.copy(),
                                expert_name="dilution",
                            )
                            all_results_dilution.append(result_dil)
                    print(f"  ✓ Experimento de diluição concluído para seed={seed}", flush=True)

            # ═══════════════════════════════════════════════════════════════
            # VIÉS DE ORDEM DOS EXEMPLOS FEW-SHOT (Recency Bias)
            # ═══════════════════════════════════════════════════════════════

            if RUN_EXAMPLE_ORDER_BIAS and RUN_PHASE_E and run_aux:
                print(f"\n>>> Experimento de Viés de Ordem dos Exemplos Few-Shot", flush=True)
                y_expert_order = expert_classify(X_e, EXPERT_W, EXPERT_CENTROIDS)
                expert_acc_order = accuracy_score(y_e, y_expert_order)

                # Usa n_shot=10 e estratégia "mixed" como configuração fixa
                # para isolar o efeito da ordenação
                ORDER_TEST_N_SHOTS = EXAMPLE_ORDER_N_SHOTS

                for n_shot_order in ORDER_TEST_N_SHOTS:
                    # Seleciona exemplos uma vez (estratégia mixed)
                    base_examples, base_indices = select_examples_by_strategy(
                        X_e, y_expert_order, EXPERT_W, EXPERT_CENTROIDS,
                        n_examples=n_shot_order,
                        strategy="mixed",
                        nome_classe_0="A", nome_classe_1="B",
                        random_state=seed,
                        verbose=False
                    )

                    # Monta conjunto de teste
                    test_mask_order = np.ones(len(X_e), dtype=bool)
                    test_mask_order[base_indices] = False
                    X_test_order = X_e[test_mask_order]
                    y_expert_test_order = y_expert_order[test_mask_order]
                    y_gt_test_order = y_e[test_mask_order]

                    for ordering in EXAMPLE_ORDERINGS:
                        reordered = reorder_examples(
                            base_examples, ordering,
                            nome_classe_0="A", nome_classe_1="B",
                            random_state=seed
                        )
                        print(f"  [Ordem] n_shot={n_shot_order}, ordering={ordering}, seed={seed}", flush=True)

                        # Compara ORDENAÇÕES FIXAS do mesmo conjunto de exemplos —
                        # não há sorteio por repetição → 1 coleta por ordenação.
                        n_reps_order = reps_para(n_shot_order, sorteio_estocastico=False)
                        for rep in range(n_reps_order):
                            t_rep_start = time.time()
                            print(f"    [Ordem {ordering}] rep {rep+1}/{n_reps_order} (n_test={len(X_test_order)})...", flush=True)
                            y_llm_order, n_malf_order = collect_llm_decisions(
                                X_test_order, "A", "B",
                                examples=reordered,
                                verbose=(rep == 0 and seed_idx == 0 and ordering == EXAMPLE_ORDERINGS[0] and n_shot_order == ORDER_TEST_N_SHOTS[0]),
                                label_prefix=f"[Ordem {ordering}] "
                            )
                            cons_order = compute_consistency_metrics(y_llm_order, y_expert_test_order)
                            llm_acc_order = accuracy_score(y_gt_test_order, y_llm_order)
                            print(
                                f"    [Ordem {ordering}] rep {rep+1}/{n_reps_order} concluída em "
                                f"{time.time()-t_rep_start:.1f}s — acc={cons_order.accuracy:.1%}, kappa={cons_order.cohen_kappa:.3f}, malf={n_malf_order}",
                                flush=True,
                            )

                            result_order = ResultadoPhaseEExperimento(
                                provider=provider, model_name=model_name,
                                temperature=temperature, random_seed=seed,
                                n_shot=n_shot_order,
                                example_strategy=f"mixed_order_{ordering}",
                                nomes_classes=("A", "B"), repeticao=rep,
                                accuracy_llm_vs_expert=cons_order.accuracy,
                                kappa_llm_vs_expert=cons_order.cohen_kappa,
                                f1_llm_vs_expert=cons_order.f1_score,
                                accuracy_expert_vs_gt=expert_acc_order,
                                accuracy_llm_vs_gt=llm_acc_order,
                                n_classe_0_expert=int(np.sum(y_expert_test_order == 0)),
                                n_classe_1_expert=int(np.sum(y_expert_test_order == 1)),
                                n_classe_0_llm=int(np.sum(y_llm_order == 0)),
                                n_classe_1_llm=int(np.sum(y_llm_order == 1)),
                                n_disagreements=cons_order.n_disagreements,
                                n_total_test=len(X_test_order),
                                n_malformed_responses=n_malf_order,
                                expert_w=EXPERT_W.copy(),
                                expert_name=f"order_{ordering}",
                            )
                            all_results_example_order.append(result_order)

                print(f"  ✓ Experimento de ordem dos exemplos concluído para seed={seed}", flush=True)

            # ═══════════════════════════════════════════════════════════════
            # NÃO-LINEARIDADE IMPLÍCITA: MÉTRICA R3 SOBRE RÓTULOS R2
            # O LLM classifica apenas com (x1, x2). Nos bastidores, adicionamos
            # x3 = x1*x2 e aprendemos métrica com 3 pesos. Se a fidelidade R3
            # supera R2, o LLM adota implicitamente um critério não-linear.
            # ═══════════════════════════════════════════════════════════════

            if RUN_R3R4_EXPERIMENT:
                print(f"\n>>> Projeção R3: x3 = x1 * x2 (seed={seed})", flush=True)

                # Reutiliza classificações do LLM da Fase A (2 features)
                # O LLM NÃO sabe da existência de x3 — queremos verificar se
                # implicitamente ele adota não-linearidade no processo de classificação
                cache_key_r3 = (provider, model_name, seed, "A", "B", "default")
                if cache_key_r3 not in phase_a_cache:
                    print("  ⚠ Cache da Fase A não disponível para esta seed, pulando R3")
                else:
                    cached_r3 = phase_a_cache[cache_key_r3]
                    y_llm_r2 = cached_r3['y_llm']
                    fid_r2 = cached_r3['fidelity']
                    fid_r2_nnls = cached_r3.get('fidelity_nnls')

                    # Aumenta X para R3 (x3 = x1 * x2)
                    X_a_r3 = augment_to_r3(X_train_a)
                    print(f"  X_a original: {X_train_a.shape} → X_a_r3: {X_a_r3.shape}")

                    # Salva dados R3
                    df_r3 = pd.DataFrame({"x1": X_a_r3[:, 0], "x2": X_a_r3[:, 1], "x3": X_a_r3[:, 2], "y": y_train_a})
                    seed_data_dir = os.path.join(pasta_execucao, f"dados_sinteticos_seed{seed}")
                    os.makedirs(seed_data_dir, exist_ok=True)
                    df_r3.to_csv(os.path.join(seed_data_dir, "problem_A_r3.csv"), index=False)
                    print(f"  Dados R3 salvos em: {seed_data_dir}/problem_A_r3.csv")

                    if len(np.unique(y_llm_r2)) >= 2:
                        # Centroides em R3 usando rótulos do LLM (que viu apenas 2 features)
                        centroids_r3 = compute_centroids(X_a_r3, y_llm_r2)

                        # Perceptron — aprende W com 3 pesos sobre os MESMOS rótulos do LLM
                        w_r3, gamma_r3 = train_relaxed_perceptron(
                            X_a_r3, y_llm_r2, centroids_r3,
                            **PERCEPTRON_PARAMS,
                            verbose=(seed_idx == 0),
                            use_best_effort=True
                        )
                        y_metric_r3 = predict_with_metric(X_a_r3, centroids_r3, w_r3)
                        fid_r3 = accuracy_score(y_llm_r2, y_metric_r3)

                        # NNLS
                        w_r3_nnls, _ = train_least_squares_inverse(
                            X_a_r3, y_llm_r2, centroids_r3, verbose=(seed_idx == 0)
                        )
                        y_metric_r3_nnls = predict_with_metric(X_a_r3, centroids_r3, w_r3_nnls)
                        fid_r3_nnls = accuracy_score(y_llm_r2, y_metric_r3_nnls)

                        cos_sim_r3 = np.dot(w_r3, w_r3_nnls) / (np.linalg.norm(w_r3) * np.linalg.norm(w_r3_nnls) + 1e-9)
                        print(f"  [Perceptron] Ŵ_LLM (3 pesos): [{w_r3[0]:.4f}, {w_r3[1]:.4f}, {w_r3[2]:.4f}]")
                        print(f"  [NNLS]       Ŵ_LLM (3 pesos): [{w_r3_nnls[0]:.4f}, {w_r3_nnls[1]:.4f}, {w_r3_nnls[2]:.4f}]")
                        print(f"  Similaridade cosseno (Perceptron vs NNLS): {cos_sim_r3:.4f}")
                        print(f"  --- Comparação de fidelidade (mesmos rótulos do LLM) ---")
                        print(f"  [Perceptron] Fidelidade R2 (2 pesos): {fid_r2:.1%}  →  R3 (3 pesos): {fid_r3:.1%}  (Δ = {fid_r3 - fid_r2:+.1%})")
                        if fid_r2_nnls is not None:
                            print(f"  [NNLS]       Fidelidade R2 (2 pesos): {fid_r2_nnls:.1%}  →  R3 (3 pesos): {fid_r3_nnls:.1%}  (Δ = {fid_r3_nnls - fid_r2_nnls:+.1%})")
                        if fid_r3 > fid_r2:
                            print(f"  ► Evidência de não-linearidade implícita: métrica R3 se ajusta melhor")
                        else:
                            print(f"  ► Sem evidência de não-linearidade: métrica R2 já é suficiente")

                        results_r3_3feat.append({
                            'accuracy': fid_r3, 'w': w_r3, 'seed': seed,
                            'accuracy_nnls': fid_r3_nnls, 'w_nnls': w_r3_nnls,
                            'provider': provider, 'model': model_name,
                        })
                        results_r3_2feat.append({'accuracy': fid_r2, 'seed': seed,
                                                 'provider': provider, 'model': model_name})

                        # ─────────────────────────────────────────────────────────────
                        # R4: x3 = x1², x4 = x2² (elipse) — sanity check em problemas
                        # lineares; orientador (~2740s) espera NENHUM ganho aqui.
                        # ─────────────────────────────────────────────────────────────
                        X_a_r4 = augment_to_r4(X_train_a)
                        centroids_r4 = compute_centroids(X_a_r4, y_llm_r2)

                        df_r4 = pd.DataFrame({
                            "x1": X_a_r4[:, 0], "x2": X_a_r4[:, 1],
                            "x1_sq": X_a_r4[:, 2], "x2_sq": X_a_r4[:, 3],
                            "y": y_train_a,
                        })
                        df_r4.to_csv(os.path.join(seed_data_dir, "problem_A_r4.csv"), index=False)

                        w_r4, _ = train_relaxed_perceptron(
                            X_a_r4, y_llm_r2, centroids_r4,
                            **PERCEPTRON_PARAMS,
                            verbose=False,
                            use_best_effort=True,
                        )
                        y_metric_r4 = predict_with_metric(X_a_r4, centroids_r4, w_r4)
                        fid_r4 = accuracy_score(y_llm_r2, y_metric_r4)

                        w_r4_nnls, _ = train_least_squares_inverse(
                            X_a_r4, y_llm_r2, centroids_r4, verbose=False,
                        )
                        y_metric_r4_nnls = predict_with_metric(X_a_r4, centroids_r4, w_r4_nnls)
                        fid_r4_nnls = accuracy_score(y_llm_r2, y_metric_r4_nnls)

                        print(f"  [Perceptron] Ŵ_LLM (4 pesos): [{w_r4[0]:.4f}, {w_r4[1]:.4f}, {w_r4[2]:.4f}, {w_r4[3]:.4f}]")
                        print(f"  [NNLS]       Ŵ_LLM (4 pesos): [{w_r4_nnls[0]:.4f}, {w_r4_nnls[1]:.4f}, {w_r4_nnls[2]:.4f}, {w_r4_nnls[3]:.4f}]")
                        print(f"  [Perceptron] Fidelidade R3 (3 pesos): {fid_r3:.1%}  →  R4 (4 pesos): {fid_r4:.1%}  (Δ = {fid_r4 - fid_r3:+.1%})")
                        if fid_r2_nnls is not None:
                            print(f"  [NNLS]       Fidelidade R3 (3 pesos): {fid_r3_nnls:.1%}  →  R4 (4 pesos): {fid_r4_nnls:.1%}  (Δ = {fid_r4_nnls - fid_r3_nnls:+.1%})")
                        if fid_r4 > fid_r3:
                            print(f"  ► R4 (elipse) supera R3 — sinal de não-linearidade quadrática isotrópica")
                        else:
                            print(f"  ► R4 (elipse) NÃO supera R3 — problema linear, conforme esperado em A/B/C")

                        results_r3_4feat.append({
                            'accuracy': fid_r4, 'w': w_r4, 'seed': seed,
                            'accuracy_nnls': fid_r4_nnls, 'w_nnls': w_r4_nnls,
                            'provider': provider, 'model': model_name,
                        })

                    print(f"  ✓ Experimento R2 vs R3 vs R4 concluído (seed={seed})", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PIPELINE DE PROBLEMAS EXTERNOS NÃO-LINEARES
    # (Parte 2 do plano: peso × altura | Parte 3 do plano: meia-lua)
    # ═══════════════════════════════════════════════════════════════════

    external_phase_a_results: List[dict] = []
    external_phase_e_results: List[dict] = []
    external_llm_label_maps: dict = {}

    if RUN_HOMEM_MULHER:
        try:
            print_section("BLOCO 3 — ESTUDO DE CASO REAL: PESO × ALTURA (homem/mulher)", "═")
            X_hm, y_hm = create_problem_homem_mulher(
                str(BASE_DIR / "dados_reais/homem_mulher/peso_altura.csv")
            )
            print(f"  Base: shape={X_hm.shape} | classes={dict(zip(*np.unique(y_hm, return_counts=True)))}")
            print(f"  Classificador ótimo: elipse x2² - x2 + x1² - x1 + cte (e-mail orientador 19:15)")

            # Visualização do problema antes de qualquer coleta LLM
            def _elipse_otima(x1, x2):
                # Coeficientes do e-mail; constante calibrada por mediana da decisão real
                z = x2**2 - x2 + x1**2 - x1
                return z - np.median(z[y_hm == 1])  # subtrai mediana da classe 1
            plot_problem_overview(
                X_hm, y_hm,
                title="Peso × Altura (homem/mulher) — base real do orientador",
                feature_names=("peso", "altura"),
                optimal_boundary_fn=_elipse_otima,
                filename=os.path.join(pasta_execucao, "bloco3_01_peso_altura_overview.png"),
            )
            print(f"  Gráfico salvo: bloco3_01_peso_altura_overview.png")

            # Item 17 (reunião 20/05): roda o peso×altura uma vez por par de nomes
            # de classe. Com "Homem"/"Mulher" o LLM pode usar prior semântico; com
            # "A"/"B" ele fica cego ao significado. Comparar as duas isola o efeito
            # do nome da classe (complementa o teste de nomes de feature x1/x2).
            class_name_variants = (
                HM_CLASS_NAME_VARIANTS if RUN_HM_CLASS_NAMES_AB
                else [("Homem", "Mulher", "homem_mulher")]
            )
            # Comparação entre modelos ("central + caso real"): TODOS os modelos de
            # MODELS_TO_TEST rodam o caso real, em pé de igualdade. Os PNGs de todos
            # vão para a raiz da execução — o alias do modelo no nome do asset
            # (llm_asset) evita sobrescrita entre modelos.
            for ext_provider, ext_model, ext_temp, ext_scope in MODELS_TO_TEST:
                checkpoint_interactions(pasta_execucao)
                client = get_client(ext_provider)
                async_client = get_async_client(ext_provider)
                MODEL_NAME = ext_model
                CURRENT_PROVIDER = ext_provider
                CURRENT_TEMPERATURE = ext_temp
                # Escopo "full" (ou --rapido) roda também a variante A/B de nomes
                # de classe; "core" roda só a semântica (Homem/Mulher).
                variants_do_modelo = (
                    class_name_variants if (ext_scope == "full" or modo_rapido)
                    else class_name_variants[:1]
                )

                for nc0, nc1, pname in variants_do_modelo:
                    print_section(
                        f"  → {ext_provider}/{ext_model} | nomes de classe: {nc0}/{nc1} ({pname})", "─")
                    hm_pipeline = run_external_problem_pipeline(
                        problem_name=pname,
                        X=X_hm, y_true=y_hm,
                        nome_classe_0=nc0, nome_classe_1=nc1,
                        feature_variants=[("x1", "x2"), ("peso", "altura")],
                        seeds=RANDOM_SEEDS,
                        n_shots_phase_e=FEW_SHOT_SIZES_PHASE_E,
                        pasta_execucao=pasta_execucao,
                        n_train_ratio=0.7,
                        verbose=True,
                    )
                    for _r in hm_pipeline['phase_a_results'] + hm_pipeline['phase_e_results']:
                        _r['provider'] = ext_provider
                        _r['model'] = ext_model
                    external_phase_a_results.extend(hm_pipeline['phase_a_results'])
                    external_phase_e_results.extend(hm_pipeline['phase_e_results'])
                    # O scatter ponto-a-ponto (final_08) é gerado para TODOS os
                    # modelos: a chave ganha o modelo na frente. Só a variante
                    # semântica (Homem/Mulher) alimenta a visualização — a A/B
                    # fica nos CSVs.
                    if pname == "homem_mulher":
                        external_llm_label_maps.update({
                            (ext_model, *k): v
                            for k, v in hm_pipeline['llm_label_maps'].items()
                        })
            print("  ✓ Pipeline peso × altura concluído (todos os modelos).", flush=True)
        except FileNotFoundError as exc:
            print(f"  ⚠ Base peso × altura não encontrada: {exc}")
        except Exception as exc:
            print(f"  ⚠ Erro no pipeline peso × altura: {exc}")
            traceback.print_exc()

    if RUN_PROBLEM_MEIALUA:
        try:
            print_section("BLOCO 1/2 — PROBLEMA D/F: MEIA-LUA (não-linear sintético)", "═")
            # Todos os modelos rodam a meia-lua (barato, e testa se o colapso
            # zero-shot se repete entre modelos). PNGs de todos na raiz, com o
            # alias do modelo no nome.
            for _ml_idx, (ext_provider, ext_model, ext_temp, ext_scope) in enumerate(MODELS_TO_TEST):
                checkpoint_interactions(pasta_execucao)
                client = get_client(ext_provider)
                async_client = get_async_client(ext_provider)
                MODEL_NAME = ext_model
                CURRENT_PROVIDER = ext_provider
                CURRENT_TEMPERATURE = ext_temp
                print_section(f"  → Meia-lua com {ext_provider}/{ext_model}", "─")

                for seed in RANDOM_SEEDS:
                    # Reseta o RNG global por semente (consistente com os demais loops):
                    # garante que qualquer estocasticidade que dependa do estado global
                    # do NumPy seja reprodutível por semente neste bloco.
                    np.random.seed(seed)
                    X_ml, y_ml = create_problem_d_meialua(n_samples=N_SAMPLES_PROBLEM_A, random_state=seed)
                    print(f"  Meia-lua seed={seed}: shape={X_ml.shape}")

                    # Visualização do problema (dado sintético, independe do
                    # modelo — 1 cópia basta: só no primeiro modelo do loop)
                    if _ml_idx == 0:
                        plot_problem_overview(
                            X_ml, y_ml,
                            title=f"Problema E — Meia-lua (sklearn.make_moons, seed={seed})",
                            feature_names=("x1", "x2"),
                            optimal_boundary_fn=None,
                            filename=os.path.join(pasta_execucao, f"bloco1_02_problema_d_meialua_seed{seed}_overview.png"),
                        )
                        print(f"  Gráfico salvo: bloco1_02_problema_d_meialua_seed{seed}_overview.png")

                    ml_pipeline = run_external_problem_pipeline(
                        problem_name=f"meia_lua_seed{seed}",
                        X=X_ml, y_true=y_ml,
                        nome_classe_0="A", nome_classe_1="B",
                        feature_variants=[("x1", "x2")],
                        seeds=[seed],
                        n_shots_phase_e=FEW_SHOT_SIZES_PHASE_E,
                        pasta_execucao=pasta_execucao,
                        n_train_ratio=0.7,
                        verbose=True,
                    )
                    for _r in ml_pipeline['phase_a_results'] + ml_pipeline['phase_e_results']:
                        _r['provider'] = ext_provider
                        _r['model'] = ext_model
                    external_phase_a_results.extend(ml_pipeline['phase_a_results'])
                    external_phase_e_results.extend(ml_pipeline['phase_e_results'])
                    external_llm_label_maps.update({
                        (ext_model, *k): v
                        for k, v in ml_pipeline['llm_label_maps'].items()
                    })

                    # Item 7 (reunião 20/05): superfície SVM gaussiano (RBF) vs LLM.
                    # Usa os rótulos zero-shot do LLM (chave 'train') e a melhor
                    # métrica diagonal aprendida para a fronteira tracejada.
                    try:
                        lbl = ml_pipeline['llm_label_maps'].get(
                            (f"meia_lua_seed{seed}", seed, 'x1', 'x2', 'train'))
                        pa = ml_pipeline['phase_a_results']
                        best_metric = max(pa, key=lambda r: r['accuracy_perc_vs_true']) if pa else None
                        if lbl is not None:
                            svm_saved = plot_meialua_svm_vs_llm(
                                lbl['X'], lbl['y_true'], lbl['y_llm'],
                                metric=best_metric, seed=seed,
                                filename=llm_asset(
                                    pasta_execucao,
                                    f"bloco23_external_svm_meialua_seed{seed}.png",
                                    ext_model,
                                ),
                            )
                            if svm_saved:
                                print(f"  Gráfico salvo: bloco23_external_svm_meialua_seed{seed}__{_model_alias(ext_model)}.png")
                            else:
                                print(f"  ⚠ Plot SVM meia-lua seed={seed} não gerado (classe única no ground truth).")
                    except Exception as exc_svm:
                        print(f"  ⚠ Falha no plot SVM meia-lua seed={seed}: {exc_svm}")
            print("  ✓ Pipeline meia-lua concluído (todos os modelos).", flush=True)
        except Exception as exc:
            print(f"  ⚠ Erro no pipeline meia-lua: {exc}")
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZAÇÕES FINAIS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n[PASSO 5] Gerando visualizações finais...", flush=True)
    print_section("BLOCO 1 — VISUALIZAÇÕES (Fases A/B/C lineares + meia-lua)", "═")

    # Cada modelo de MODELS_TO_TEST gera o conjunto COMPLETO de visualizações
    # na raiz da execução, com seu alias (MODEL_ALIAS) no nome do asset; os
    # CSVs guardam todos os modelos e o final_09_model_comparison.png compara todos.
    if all_results_abc and len(MODELS_TO_TEST) > 1:
        plot_model_comparison(all_results_abc, filename=os.path.join(pasta_execucao, "final_09_model_comparison.png"))
        print(f"  Gráfico salvo: final_09_model_comparison.png")

    # Validação do oráculo (independe de modelo — nenhuma chamada de LLM)
    if all_results_oracle:
        plot_oracle_w_recovery(all_results_oracle,
            filename=os.path.join(pasta_execucao, "bloco1_03_oracle_w_recovery.png"))
        print(f"  Gráfico salvo: bloco1_03_oracle_w_recovery.png")
        plot_oracle_transfer(all_results_oracle,
            filename=os.path.join(pasta_execucao, "bloco1_04_oracle_transfer.png"))
        print(f"  Gráfico salvo: bloco1_04_oracle_transfer.png")

    # Item 5: oracle de aproximação da meia-lua (fidelidade vs GT por n_features)
    if all_results_oracle_meialua:
        plot_oracle_meialua(all_results_oracle_meialua,
            filename=os.path.join(pasta_execucao, "bloco1_04b_oracle_meialua.png"))
        print(f"  Gráfico salvo: bloco1_04b_oracle_meialua.png")

    for _viz_provider, _viz_model, _viz_temp, _viz_scope in MODELS_TO_TEST:
        abc_m = [r for r in all_results_abc
                 if r.provider == _viz_provider and r.model_name == _viz_model]
        e_m = [r for r in all_results_e
               if r.provider == _viz_provider and r.model_name == _viz_model]
        # Baselines clássicos independem do LLM (lista única, treinada 1×) —
        # entram na comparação de todos os modelos (são dicts com
        # provider='classical' e model=<nome do classificador>).
        baselines_m = all_results_baselines
        dilution_m = [r for r in all_results_dilution
                      if r.provider == _viz_provider and r.model_name == _viz_model]
        order_m = [r for r in all_results_example_order
                   if r.provider == _viz_provider and r.model_name == _viz_model]
        alternative_m = [r for r in all_results_abc_alternative
                         if r.provider == _viz_provider and r.model_name == _viz_model]
        phase_a_plots_m = {s: d for (p, m, s), d in phase_a_data_for_plots.items()
                           if (p, m) == (_viz_provider, _viz_model)}
        seed_detail_m = {s: d for (p, m, s), d in seed_detailed_data.items()
                         if (p, m) == (_viz_provider, _viz_model)}
        if not (abc_m or e_m):
            continue  # modelo sem dados nesta execução (ex.: --rapido --modelo X)

        print_section(
            f"VISUALIZAÇÕES — {_viz_provider}/{_viz_model} "
            f"(alias: {_model_alias(_viz_model)})", "═")

        if abc_m:
            fname = llm_asset(pasta_execucao, "bloco1_09_consistency_extended.png", _viz_model)
            plot_consistency_comparison_extended(abc_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")
            fname = llm_asset(pasta_execucao, "bloco1_14a_class_names_effect.png", _viz_model)
            plot_class_names_effect(abc_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            if len(RANDOM_SEEDS) > 1:
                fname = llm_asset(pasta_execucao, "bloco1_08_seed_comparison.png", _viz_model)
                plot_seed_comparison(abc_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Distribuição de W
            fname = llm_asset(pasta_execucao, "bloco1_06_w_distribution.png", _viz_model)
            plot_w_distribution(abc_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Erros da métrica na Fase A
            for seed_key, data in phase_a_plots_m.items():
                fname = llm_asset(pasta_execucao, f"bloco1_05_fase_a_errors_seed{seed_key}.png", _viz_model)
                plot_metric_errors_phase_a(
                    data['X'], data['y_llm'], data['y_metric'],
                    data['w'], data['centroids'], filename=fname
                )
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Análise quantitativa de erros por região (complementa bloco1_05)
            if phase_a_plots_m:
                print_error_analysis_by_region(phase_a_plots_m)

            # Análise de sensibilidade dos hiperparâmetros do Perceptron
            if phase_a_plots_m:
                print_hyperparameter_sensitivity(phase_a_plots_m)

            # Viés de ordem das classes
            if RUN_CLASS_ORDER_BIAS:
                fname = llm_asset(pasta_execucao, "bloco1_12_class_order_bias.png", _viz_model)
                plot_class_order_bias(abc_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Efeito de nomes de features
            if RUN_FEATURE_NAMES:
                fname = llm_asset(pasta_execucao, "bloco1_14b_feature_names_effect.png", _viz_model)
                plot_feature_names_effect(abc_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Variantes de prompt
            if RUN_PROMPT_VARIANTS:
                fname = llm_asset(pasta_execucao, "bloco1_13_prompt_variants.png", _viz_model)
                plot_prompt_variant_comparison(abc_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

        print_section("BLOCO 2 — VISUALIZAÇÕES (Fase E, LLM como aprendiz)", "═")

        if e_m:
            fname = llm_asset(pasta_execucao, "bloco2_04_phase_e_learning_curve.png", _viz_model)
            plot_phase_e_learning_curve(e_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")
            fname = llm_asset(pasta_execucao, "bloco2_05_phase_e_strategy_comparison.png", _viz_model)
            plot_phase_e_strategy_comparison(e_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

        # Baselines clássicos
        if baselines_m and e_m:
            fname = llm_asset(pasta_execucao, "bloco2_09_classical_baselines.png", _viz_model)
            plot_classical_baselines_comparison(e_m, baselines_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

        # Experimento de diluição (CSV guarda todos os modelos)
        if dilution_m:
            fname = llm_asset(pasta_execucao, "bloco2_06_dilution.png", _viz_model)
            plot_dilution_experiment(dilution_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

        # Viés de ordem dos exemplos few-shot
        if order_m:
            fname = llm_asset(pasta_execucao, "bloco2_07_example_order.png", _viz_model)
            plot_example_order_bias(order_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Análise quantitativa do viés de ordem (recency bias)
            print_example_order_analysis(order_m)

        # Não-linearidade implícita: fidelidade R2 (2 pesos) vs R3 (3 pesos)
        # (o CSV r3r4 consolida todos os modelos)
        r3_2feat_viz = [r for r in results_r3_2feat if r.get('model') in (None, _viz_model)]
        r3_3feat_viz = [r for r in results_r3_3feat if r.get('model') in (None, _viz_model)]
        if r3_2feat_viz and r3_3feat_viz:
            fname = llm_asset(pasta_execucao, "bloco1_10_r3r4_comparison.png", _viz_model)
            plot_r3_comparison(r3_2feat_viz, r3_3feat_viz, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

        # Comparação de algoritmos — Perceptron × NNLS
        if alternative_m and abc_m:
            perc_results = [r for r in abc_m
                           if r.n_shot == 0 and r.nomes_classes == ("A", "B")]
            if perc_results:
                fname = llm_asset(pasta_execucao, "bloco1_07_algorithm_comparison.png", _viz_model)
                plot_algorithm_comparison(perc_results, alternative_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

        # Diagnóstico da busca binária em γ (item b reunião 30/04/2026, ~520s)
        gamma_diag_m = [d for d in PERCEPTRON_GAMMA_DIAGNOSTICS
                        if d.get("model") in (None, _viz_model)]
        if gamma_diag_m:
            plot_gamma_convergence(
                gamma_diag_m,
                filename=llm_asset(pasta_execucao, "final_10_gamma_convergence.png", _viz_model),
            )

        # ═══════════════════════════════════════════════════════════════
        # VISUALIZAÇÕES DETALHADAS POR SEED (deste modelo)
        # ═══════════════════════════════════════════════════════════════

        print_section("FECHAMENTO — VISUALIZAÇÕES DETALHADAS POR SEED", "═")
        for seed_key, sdata in seed_detail_m.items():
            print(f"\n  Gerando visualizações detalhadas para seed {seed_key} ({_model_alias(_viz_model)})...")

            fname = llm_asset(pasta_execucao, f"final_04_dataset_overview_seed{seed_key}.png", _viz_model)
            plot_dataset_overview(sdata, seed_key, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            fname = llm_asset(pasta_execucao, f"final_05_hits_errors_seed{seed_key}.png", _viz_model)
            plot_hits_and_errors(sdata, seed_key, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            fname = llm_asset(pasta_execucao, f"final_06_w_algorithms_seed{seed_key}.png", _viz_model)
            plot_w_comparison_algorithms(sdata, seed_key, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            fname = llm_asset(pasta_execucao, f"final_02_confusion_matrices_seed{seed_key}.png", _viz_model)
            plot_confusion_matrices_detailed(sdata, seed_key, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            fname = llm_asset(pasta_execucao, f"final_07_margin_analysis_seed{seed_key}.png", _viz_model)
            plot_margin_analysis_detailed(sdata, seed_key, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

            fname = llm_asset(pasta_execucao, f"final_03_dashboard_seed{seed_key}.png", _viz_model)
            plot_experiment_summary_dashboard(sdata, seed_key, abc_m, e_m, filename=fname)
            print(f"  Gráfico salvo: {os.path.basename(fname)}")

        # ═══════════════════════════════════════════════════════════════
        # ANALYSIS (deste modelo; CSVs guardam todos os modelos)
        # ═══════════════════════════════════════════════════════════════

        if abc_m:
            print_final_analysis(abc_m)
        if e_m:
            print_phase_e_analysis(e_m)

        # Sumário estatístico com Bootstrap CI, Wilcoxon e Cohen's d
        print_statistical_summary(abc_m, e_m)

    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n[PASSO 6] Salvando resultados em CSV...", flush=True)
    timestamp_csv = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_results_abc:
        df_abc = pd.DataFrame([
            {
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'n_shot_phase_bc': r.n_shot,
                'class_0': r.nomes_classes[0], 'class_1': r.nomes_classes[1],
                'feature_0': r.feature_names[0], 'feature_1': r.feature_names[1],
                'prompt_variant': r.prompt_variant,
                'repetition': r.repeticao, 'fidelity_problem_a': r.fidelidade_problema_a,
                'consistency_problem_b': r.consistencia_problema_b,
                'kappa_problem_b': r.kappa_problema_b, 'f1_problem_b': r.f1_problema_b,
                'consistency_problem_c': r.consistencia_problema_c,
                'kappa_problem_c': r.kappa_problema_c, 'f1_problem_c': r.f1_problema_c,
                'llm_accuracy_problem_a': r.acuracia_llm_vs_gt_problema_a,
                'llm_accuracy_problem_b': r.acuracia_llm_vs_gt_problema_b,
                'llm_accuracy_problem_c': r.acuracia_llm_vs_gt_problema_c,
                'metric_accuracy_problem_b': r.acuracia_metrica_vs_gt_problema_b,
                'metric_accuracy_problem_c': r.acuracia_metrica_vs_gt_problema_c,
                'w_0': r.w_aprendido[0], 'w_1': r.w_aprendido[1],
                'w_ratio': r.w_aprendido[0] / r.w_aprendido[1] if r.w_aprendido[1] != 0 else float('inf'),
                'w_direction_0': r.w_direction[0] if r.w_direction is not None else 0.0,
                'w_direction_1': r.w_direction[1] if r.w_direction is not None else 0.0,
                'w_cosine_sim_nnls': r.w_cosine_sim_nnls,
                'gamma': r.gamma_otimo,
                'n_disagreements_b': r.n_disagreements_b, 'n_disagreements_c': r.n_disagreements_c,
                'n_malformed_responses': r.n_malformed_responses,
                'euclidean_consistency_b': r.consistencia_euclidiana_problema_b,
                'euclidean_consistency_c': r.consistencia_euclidiana_problema_c,
                'diagonal_limitation_flag': r.diagonal_limitation_flag,
            }
            for r in all_results_abc
        ])
        filename_abc = os.path.join(pasta_execucao, f"bloco1_phases_abc_{timestamp_csv}.csv")
        df_abc.to_csv(filename_abc, index=False)
        print(f"  Resultados das Fases A-C salvos em: {filename_abc}")

    if all_results_e:
        df_d = pd.DataFrame([
            {
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'n_shot': r.n_shot,
                'example_strategy': r.example_strategy,
                'expert_name': r.expert_name,
                'class_0': r.nomes_classes[0], 'class_1': r.nomes_classes[1],
                'repetition': r.repeticao,
                'accuracy_llm_vs_expert': r.accuracy_llm_vs_expert,
                'kappa_llm_vs_expert': r.kappa_llm_vs_expert,
                'f1_llm_vs_expert': r.f1_llm_vs_expert,
                'accuracy_expert_vs_gt': r.accuracy_expert_vs_gt,
                'accuracy_llm_vs_gt': r.accuracy_llm_vs_gt,
                'n_class_0_expert': r.n_classe_0_expert,
                'n_class_1_expert': r.n_classe_1_expert,
                'n_class_0_llm': r.n_classe_0_llm,
                'n_class_1_llm': r.n_classe_1_llm,
                'n_disagreements': r.n_disagreements,
                'n_total_test': r.n_total_test,
                'n_malformed_responses': r.n_malformed_responses,
                'expert_w_0': r.expert_w[0], 'expert_w_1': r.expert_w[1],
            }
            for r in all_results_e
        ])
        filename_e = os.path.join(pasta_execucao, f"bloco2_phase_e_{timestamp_csv}.csv")
        df_d.to_csv(filename_e, index=False)
        print(f"  Resultados da Fase E salvos em: {filename_e}")

    if all_results_abc_alternative:
        def _algo_row(r, algo_name):
            return {
                'algorithm': algo_name,
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'n_shot_phase_bc': r.n_shot,
                'class_0': r.nomes_classes[0], 'class_1': r.nomes_classes[1],
                'repetition': r.repeticao, 'fidelity_problem_a': r.fidelidade_problema_a,
                'consistency_problem_b': r.consistencia_problema_b,
                'kappa_problem_b': r.kappa_problema_b, 'f1_problem_b': r.f1_problema_b,
                'consistency_problem_c': r.consistencia_problema_c,
                'kappa_problem_c': r.kappa_problema_c, 'f1_problem_c': r.f1_problema_c,
                'w_0': r.w_aprendido[0], 'w_1': r.w_aprendido[1],
                'w_ratio': r.w_aprendido[0] / r.w_aprendido[1] if r.w_aprendido[1] != 0 else float('inf'),
                'w_direction_0': r.w_direction[0] if r.w_direction is not None else 0.0,
                'w_direction_1': r.w_direction[1] if r.w_direction is not None else 0.0,
                'gamma': r.gamma_otimo,
                'n_disagreements_b': r.n_disagreements_b, 'n_disagreements_c': r.n_disagreements_c,
            }
        alt_rows = [_algo_row(r, 'NNLS') for r in all_results_abc_alternative]
        df_alt = pd.DataFrame(alt_rows)
        filename_alt = os.path.join(pasta_execucao, f"bloco1_algorithm_comparison_{timestamp_csv}.csv")
        df_alt.to_csv(filename_alt, index=False)
        print(f"  Resultados comparação de algoritmos (NNLS) salvos em: {filename_alt}")

    if all_results_dilution:
        df_dilution = pd.DataFrame([
            {
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'n_shot': r.n_shot,
                'example_strategy': r.example_strategy,
                'repetition': r.repeticao,
                'accuracy_llm_vs_expert': r.accuracy_llm_vs_expert,
                'kappa_llm_vs_expert': r.kappa_llm_vs_expert,
                'f1_llm_vs_expert': r.f1_llm_vs_expert,
            }
            for r in all_results_dilution
        ])
        filename_dil = os.path.join(pasta_execucao, f"bloco2_dilution_{timestamp_csv}.csv")
        df_dilution.to_csv(filename_dil, index=False)
        print(f"  Resultados da Diluição salvos em: {filename_dil}")

    if results_r3_2feat or results_r3_3feat or results_r3_4feat:
        r3_rows = []
        for r in results_r3_3feat:
            w = r['w']
            w_nnls = r.get('w_nnls', np.array([np.nan]*3))
            r3_rows.append({
                'provider': r.get('provider'), 'model': r.get('model'),
                'seed': r['seed'], 'n_features': 3, 'algorithm': 'perceptron',
                'fidelidade': r['accuracy'],
                'w0': w[0], 'w1': w[1],
                'w2': w[2] if len(w) > 2 else np.nan,
                'w3': np.nan,
            })
            r3_rows.append({
                'provider': r.get('provider'), 'model': r.get('model'),
                'seed': r['seed'], 'n_features': 3, 'algorithm': 'nnls',
                'fidelidade': r.get('accuracy_nnls', np.nan),
                'w0': w_nnls[0], 'w1': w_nnls[1],
                'w2': w_nnls[2] if len(w_nnls) > 2 else np.nan,
                'w3': np.nan,
            })
        for r in results_r3_4feat:
            w = r['w']
            w_nnls = r.get('w_nnls', np.array([np.nan]*4))
            r3_rows.append({
                'provider': r.get('provider'), 'model': r.get('model'),
                'seed': r['seed'], 'n_features': 4, 'algorithm': 'perceptron',
                'fidelidade': r['accuracy'],
                'w0': w[0], 'w1': w[1],
                'w2': w[2] if len(w) > 2 else np.nan,
                'w3': w[3] if len(w) > 3 else np.nan,
            })
            r3_rows.append({
                'provider': r.get('provider'), 'model': r.get('model'),
                'seed': r['seed'], 'n_features': 4, 'algorithm': 'nnls',
                'fidelidade': r.get('accuracy_nnls', np.nan),
                'w0': w_nnls[0], 'w1': w_nnls[1],
                'w2': w_nnls[2] if len(w_nnls) > 2 else np.nan,
                'w3': w_nnls[3] if len(w_nnls) > 3 else np.nan,
            })
        for r in results_r3_2feat:
            r3_rows.append({
                'provider': r.get('provider'), 'model': r.get('model'),
                'seed': r['seed'], 'n_features': 2, 'algorithm': 'llm_2feat',
                'fidelidade': r['accuracy'],
                'w0': np.nan, 'w1': np.nan, 'w2': np.nan, 'w3': np.nan,
            })
        df_r3_csv = pd.DataFrame(r3_rows)
        filename_r3 = os.path.join(pasta_execucao, f"bloco1_r3r4_comparison_{timestamp_csv}.csv")
        df_r3_csv.to_csv(filename_r3, index=False)
        print(f"  Resultados R2 vs R3 vs R4 salvos em: {filename_r3}")

    if all_results_example_order:
        df_order = pd.DataFrame([
            {
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'n_shot': r.n_shot,
                'example_ordering': r.example_strategy.replace("mixed_order_", ""),
                'repetition': r.repeticao,
                'accuracy_llm_vs_expert': r.accuracy_llm_vs_expert,
                'kappa_llm_vs_expert': r.kappa_llm_vs_expert,
                'f1_llm_vs_expert': r.f1_llm_vs_expert,
                'accuracy_expert_vs_gt': r.accuracy_expert_vs_gt,
                'accuracy_llm_vs_gt': r.accuracy_llm_vs_gt,
                'n_disagreements': r.n_disagreements,
                'n_total_test': r.n_total_test,
                'n_malformed_responses': r.n_malformed_responses,
            }
            for r in all_results_example_order
        ])
        filename_order = os.path.join(pasta_execucao, f"bloco2_example_order_{timestamp_csv}.csv")
        df_order.to_csv(filename_order, index=False)
        print(f"  Resultados do Viés de Ordem salvos em: {filename_order}")

    if all_results_baselines:
        df_baselines = pd.DataFrame(all_results_baselines)
        filename_bl = os.path.join(pasta_execucao, f"bloco2_classical_baselines_{timestamp_csv}.csv")
        df_baselines.to_csv(filename_bl, index=False)
        print(f"  Resultados dos Baselines Clássicos salvos em: {filename_bl}")

    if all_results_oracle:
        df_oracle = pd.DataFrame(all_results_oracle)
        filename_oracle = os.path.join(pasta_execucao, f"bloco1_oracle_validation_{timestamp_csv}.csv")
        df_oracle.to_csv(filename_oracle, index=False)
        print(f"  Resultados da Validação do Oráculo salvos em: {filename_oracle}")

    if all_results_oracle_meialua:  # Item 5: oracle de aproximação da meia-lua
        df_oracle_ml = pd.DataFrame(all_results_oracle_meialua)
        filename_oracle_ml = os.path.join(pasta_execucao, f"bloco1_oracle_meialua_{timestamp_csv}.csv")
        df_oracle_ml.to_csv(filename_oracle_ml, index=False)
        print(f"  Oracle meia-lua (aproximação) salvo em: {filename_oracle_ml}")

    # ─── Problemas externos não-lineares (peso×altura, meia-lua) ──────────
    # CSVs guardam TODOS os modelos; plots/sínteses são gerados POR MODELO
    # (alias no nome do asset). _primary_model resta só como fallback de
    # preenchimento para linhas legadas sem coluna model.
    _primary_model = MODELS_TO_TEST[0][1]

    if external_phase_a_results:
        rows_a = []

        def _w_comp(w, i):
            """Componente i do vetor W (ou None se ausente)."""
            try:
                return float(w[i]) if w is not None and len(w) > i else None
            except (TypeError, IndexError):
                return None

        for r in external_phase_a_results:
            wp = r.get('w_perc')
            wn = r.get('w_nnls')
            wp0, wp1 = _w_comp(wp, 0), _w_comp(wp, 1)
            row = {
                'provider': r.get('provider'),
                'model': r.get('model'),
                'problem_name': r.get('problem_name'),
                'seed': r.get('seed'),
                'n_features': r.get('n_features'),
                'feature_names': '/'.join(r.get('feature_names', ('', ''))),
                'prompt_variant': r.get('prompt_variant'),
                'fidelity_perc_vs_llm': r.get('fidelity_perc_vs_llm'),
                'fidelity_nnls_vs_llm': r.get('fidelity_nnls_vs_llm'),
                'accuracy_perc_vs_true': r.get('accuracy_perc_vs_true'),
                'accuracy_nnls_vs_true': r.get('accuracy_nnls_vs_true'),
                'llm_accuracy_vs_true': r.get('llm_accuracy_vs_true'),
                # Item 15: contagem absoluta de erros (peso×altura é pequeno)
                'n_samples': r.get('n_samples'),
                'n_errors_perc_vs_true': r.get('n_errors_perc_vs_true'),
                'n_errors_nnls_vs_true': r.get('n_errors_nnls_vs_true'),
                'n_errors_llm_vs_true': r.get('n_errors_llm_vs_true'),
                # Item 3 (reunião 20/05): exibir o W APRENDIDO também no Bloco 3.
                # w_perc_* / w_nnls_* (até 4 componentes) + razão w0/w1 do Perceptron.
                'w_perc_0': wp0,
                'w_perc_1': wp1,
                'w_perc_2': _w_comp(wp, 2),
                'w_perc_3': _w_comp(wp, 3),
                'w_nnls_0': _w_comp(wn, 0),
                'w_nnls_1': _w_comp(wn, 1),
                'w_nnls_2': _w_comp(wn, 2),
                'w_nnls_3': _w_comp(wn, 3),
                'w_perc_ratio': (wp0 / wp1) if (wp0 is not None and wp1) else None,
                'gamma_perc': r.get('gamma_perc'),
                'n_malformed': r.get('n_malformed'),
            }
            rows_a.append(row)
        df_ext_a = pd.DataFrame(rows_a)
        fname_ext_a = os.path.join(pasta_execucao, f"bloco23_external_phase_a_{timestamp_csv}.csv")
        df_ext_a.to_csv(fname_ext_a, index=False)
        print(f"  Resultados Fase A externos (peso×altura + meia-lua) salvos em: {fname_ext_a}")

    if external_phase_e_results:
        df_ext_d = pd.DataFrame(external_phase_e_results)
        # converte tupla feature_names para string
        df_ext_d['feature_names'] = df_ext_d['feature_names'].apply(lambda t: '/'.join(t) if isinstance(t, tuple) else t)
        fname_ext_e = os.path.join(pasta_execucao, f"bloco23_external_phase_e_{timestamp_csv}.csv")
        df_ext_d.to_csv(fname_ext_e, index=False)
        print(f"  Resultados Fase E externos salvos em: {fname_ext_e}")

    # Tabela cruzada linear × não-linear — uma por modelo (alias no nome)
    for _ext_provider, _ext_model, _ext_temp, _ext_scope in MODELS_TO_TEST:
        r3_3feat_m = [r for r in results_r3_3feat if r.get('model') in (None, _ext_model)]
        r3_4feat_m = [r for r in results_r3_4feat if r.get('model') in (None, _ext_model)]
        ext_a_m = [
            r for r in external_phase_a_results
            if r.get('provider') in (None, _ext_provider)
            and r.get('model') in (None, _ext_model)
        ]
        if not (r3_3feat_m or ext_a_m):
            continue
        cross_fname = summarize_cross_linearity(
            results_abc_r3=r3_3feat_m,
            external_results=ext_a_m,
            pasta_execucao=pasta_execucao,
            results_abc_r4=r3_4feat_m if r3_4feat_m else None,
            model_name=_ext_model,
        )
        if cross_fname:
            print(f"  Comparação cruzada linear×não-linear salva em: {cross_fname}")

    # Visualização ponto-a-ponto das rotulações do LLM (item G, e-mail 22:06)
    # A chave carrega o modelo e o problema — um scatter por
    # (modelo, problema, seed, features, kind).
    if external_llm_label_maps:
        print(f"\n  Gerando scatter ponto-a-ponto das rotulações do LLM...")
        for key, data in external_llm_label_maps.items():
            model_lbl, prob_name, seed_val, feat_0, feat_1, kind = key
            kind_label = f"n_shot={kind}" if isinstance(kind, int) else str(kind)
            # Nome do problema no arquivo desambigua problemas com os mesmos
            # nomes de feature; o sufixo _seed{N} do meia_lua_seed{N} é
            # removido porque o seed já aparece como componente próprio.
            prob_slug = prob_name.replace(f"_seed{seed_val}", "")
            fname_lbl = llm_asset(
                pasta_execucao,
                f"final_08_llm_labels_{prob_slug}_seed{seed_val}_{feat_0}_{feat_1}_{kind}.png",
                model_lbl,
            )
            try:
                plot_llm_labels_per_problem(
                    X=data['X'], y_llm=data['y_llm'], y_true=data.get('y_true'),
                    title=(f"Rotulação LLM ({_model_alias(model_lbl)}) — {prob_slug} | "
                           f"seed={seed_val} | {feat_0}/{feat_1} | {kind_label}"),
                    feature_names=(feat_0, feat_1),
                    filename=fname_lbl,
                )
            except Exception as exc:
                print(f"    ⚠ Falha em {fname_lbl}: {exc}")
        print(f"  ✓ Scatter ponto-a-ponto gerados em {pasta_execucao}/final_08_llm_labels_*.png")

    # ─── Plots adicionais para apresentação (problemas externos) ──────────
    # Gerados POR MODELO, com o alias no nome do asset.
    for _ext_provider, _ext_model, _ext_temp, _ext_scope in MODELS_TO_TEST:
        ext_e_m = [
            r for r in external_phase_e_results
            if r.get('provider') in (None, _ext_provider)
            and r.get('model') in (None, _ext_model)
        ]
        ext_a_m = [
            r for r in external_phase_a_results
            if r.get('provider') in (None, _ext_provider)
            and r.get('model') in (None, _ext_model)
        ]
        if ext_e_m:
            try:
                plot_external_learning_curve(
                    ext_e_m,
                    filename=llm_asset(pasta_execucao, "bloco23_external_learning_curve.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_external_learning_curve__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_external_learning_curve ({_ext_model}): {exc}")

            try:
                plot_phase_e_llm_vs_perceptron(
                    ext_e_m,
                    filename=llm_asset(pasta_execucao, "bloco23_external_llm_vs_perceptron.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_external_llm_vs_perceptron__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_external_llm_vs_perceptron ({_ext_model}): {exc}")

        if ext_a_m:
            try:
                plot_external_features_comparison(
                    ext_a_m,
                    filename=llm_asset(pasta_execucao, "bloco23_external_features_comparison.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_external_features_comparison__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_external_features_comparison ({_ext_model}): {exc}")

            try:
                plot_external_decision_boundary(
                    ext_a_m,
                    filename=llm_asset(pasta_execucao, "bloco23_external_decision_boundary.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_external_decision_boundary__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_external_decision_boundary ({_ext_model}): {exc}")

    if external_phase_a_results:
        # ─── Resumo consolidado no log (auxilia roteiro da apresentação) ──────
        print_section("BLOCO 2/3 — RESUMO CONSOLIDADO: Pipeline externos (Fase A)", "═")
        df_ext = pd.DataFrame(external_phase_a_results)
        if 'model' not in df_ext.columns:
            df_ext['model'] = _primary_model
        df_ext['model'] = df_ext['model'].fillna(_primary_model)
        df_ext['variant'] = df_ext['feature_names'].apply(
            lambda t: '/'.join(t) if isinstance(t, tuple) else str(t)
        )
        # Agrupa pelo nome-base do problema: meia_lua_seed{N} → meia_lua, para
        # que a média±desvio agregue os 3 seeds (cada seed da meia-lua é um
        # pipeline próprio porque o dataset sintético é regenerado por seed).
        df_ext['problem_base'] = df_ext['problem_name'].str.replace(
            r'_seed\d+$', '', regex=True
        )

        def _fmt_mean_std(serie) -> str:
            # ± só faz sentido com 2+ valores; com n=1 o std amostral é NaN.
            if len(serie) > 1:
                return f"{serie.mean():.1%}±{serie.std():.1%}"
            return f"{serie.mean():.1%}"

        for model_sum in sorted(df_ext['model'].unique()):
            df_m = df_ext[df_ext['model'] == model_sum]
            print(f"\n  ═══ Modelo: {model_sum} ═══")
            for problem in sorted(df_m['problem_base'].unique()):
                sub = df_m[df_m['problem_base'] == problem]
                for variant in sorted(sub['variant'].unique()):
                    sv = sub[sub['variant'] == variant]
                    n_seeds = sv['seed'].nunique() if 'seed' in sv.columns else len(sv)
                    print(f"\n  {problem} | {variant} (seeds agregados: {n_seeds}):")
                    for nf in sorted(sv['n_features'].unique()):
                        ss = sv[sv['n_features'] == nf]
                        print(
                            f"    n_features={nf}: "
                            f"fid_perc={_fmt_mean_std(ss['fidelity_perc_vs_llm'])} | "
                            f"acc_perc_real={_fmt_mean_std(ss['accuracy_perc_vs_true'])} | "
                            f"llm_real={_fmt_mean_std(ss['llm_accuracy_vs_true'])}"
                        )
        best_overall = df_ext.loc[df_ext['accuracy_perc_vs_true'].idxmax()]
        print(
            f"\n  ★ Melhor configuração geral: {best_overall.get('model', '')} | "
            f"{best_overall['problem_name']} | "
            f"{best_overall['variant']} | n_features={best_overall['n_features']} | "
            f"acc_real={best_overall['accuracy_perc_vs_true']:.1%}"
        )

    if external_phase_e_results:
        print_section("BLOCO 2/3 — RESUMO CONSOLIDADO: Fase E externa (LLM vs Perceptron)", "═")
        df_d = pd.DataFrame(external_phase_e_results)
        if 'model' not in df_d.columns:
            df_d['model'] = _primary_model
        df_d['model'] = df_d['model'].fillna(_primary_model)
        df_d['variant'] = df_d['feature_names'].apply(
            lambda t: '/'.join(t) if isinstance(t, tuple) else str(t)
        )
        for problem in sorted(df_d['problem_name'].unique()):
            sub_all = df_d[df_d['problem_name'] == problem]
            for model_sum in sorted(sub_all['model'].unique()):
                sub = sub_all[sub_all['model'] == model_sum]
                for variant in sorted(sub['variant'].unique()):
                    sv = sub[sub['variant'] == variant]
                    print(f"\n  {problem} | {model_sum} | {variant}:")
                    for nshot in sorted(sv['n_shot'].unique()):
                        ss = sv[sv['n_shot'] == nshot]
                        llm_real = ss['accuracy_llm_vs_true'].mean()
                        pb = ss['accuracy_perceptron_baseline_vs_true'].dropna()
                        if len(pb):
                            delta = llm_real - pb.mean()
                            marker = "LLM>Perceptron" if delta > 0.01 else (
                                "Perceptron>LLM" if delta < -0.01 else "≈ empate"
                            )
                            print(
                                f"    n_shot={nshot}: llm_real={llm_real:.1%} | "
                                f"perceptron_real={pb.mean():.1%} | Δ={delta:+.1%} ({marker})"
                            )
                        else:
                            print(f"    n_shot={nshot}: llm_real={llm_real:.1%}")

    # ─── Salvar interações com a LLM em JSON ──────────
    interactions_paths = salvar_json_em_chunks(
        LLM_INTERACTIONS, pasta_execucao, nome_base="llm_interactions"
    )
    # Salvamento oficial concluído → o checkpoint intermediário é redundante
    _ckpt = os.path.join(pasta_execucao, "llm_interactions_checkpoint.json")
    if os.path.exists(_ckpt):
        os.remove(_ckpt)
    if len(interactions_paths) == 1:
        print(f"  Interações LLM salvas em: {interactions_paths[0]} "
              f"({len(LLM_INTERACTIONS)} chamadas)")
    else:
        print(f"  Interações LLM ({len(LLM_INTERACTIONS)} chamadas) excederam "
              f"{LOG_CHUNK_LIMIT_BYTES // (1024*1024)} MiB — "
              f"fatiadas em {len(interactions_paths)} arquivos JSON válidos:")
        for p in interactions_paths:
            print(f"    - {p}")

    # ─── Auditoria do fallback do parser (fonte da verdade: LLM_INTERACTIONS) ──────
    # Cada resposta não parseada cai no fallback determinístico por hash e ENTRA nas
    # métricas (κ/consistência). Um resumo explícito da taxa por run mantém isso
    # auditável de relance — além da coluna `n_malformed_responses` já salva nos CSVs.
    n_calls_total = len(LLM_INTERACTIONS)
    n_fallback = sum(1 for it in LLM_INTERACTIONS if it.get("malformed"))
    taxa_fallback = (100.0 * n_fallback / n_calls_total) if n_calls_total else 0.0
    print(f"\n  ── Auditoria do parser ──")
    print(f"     Chamadas ao LLM: {n_calls_total} | Fallback (hash) acionado: "
          f"{n_fallback} ({taxa_fallback:.2f}%)")
    if n_fallback == 0:
        print(f"     ✓ Nenhum fallback: métricas (κ/consistência) não contêm rótulos pseudo-aleatórios.")
    elif taxa_fallback <= 5.0:
        print(f"     ✓ Taxa ≤ 5%: impacto do fallback nas métricas é marginal.")
    else:
        print(f"     ⚠️ Taxa > 5%: rótulos de fallback podem poluir κ/consistência — "
              f"reportar métricas com/sem fallback.")

    # ─── Passo 7: Salvar log TXT (fatiado em partes se passar de 10 MiB) ──────────
    print(f"\n[PASSO 7] Salvando log completo da execução...")
    sys.stdout = tee._stream
    sys.stderr = tee_err._stream
    log_paths = salvar_log_em_chunks(tee.getvalue(), pasta_execucao, nome_base="log_execucao")
    if len(log_paths) == 1:
        log_path = log_paths[0]
    else:
        log_path = f"{len(log_paths)} partes: " + ", ".join(os.path.basename(p) for p in log_paths)
        print(f"  Log excedeu {LOG_CHUNK_LIMIT_BYTES // (1024*1024)} MiB — "
              f"fatiado em {len(log_paths)} arquivos:")
        for p in log_paths:
            print(f"    - {p}")

    n_arquivos = len(os.listdir(pasta_execucao))
    print(f"\n{'='*70}")
    print(f" EXPERIMENTO CONCLUÍDO!")
    print(f" Pasta de saída: {pasta_execucao}/")
    print(f" Total de arquivos gerados: {n_arquivos}")
    print(f"   - Imagens PNG: gráficos de todas as fases")
    print(f"   - CSVs: resultados numéricos detalhados")
    print(f"   - Log TXT: {log_path}")
    print(f"{'='*70}\n")

    return all_results_abc, all_results_e


if __name__ == "__main__":
    results_abc, results_e = main()