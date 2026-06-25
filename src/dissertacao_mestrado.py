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

PROVEDORES: OpenAI (ativo), Claude/Anthropic (SDK integrado).
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend não-interativo: salva em arquivo, não abre janela
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass, field
import warnings
import json


# =============================================================================
# CAPTURA DE LOG (TEE): EXIBE NO TERMINAL E ARMAZENA EM BUFFER PARA O TXT
# =============================================================================

class Tee:
    """Duplica o output: exibe no terminal E armazena em buffer para salvar no log TXT."""
    def __init__(self):
        self._stdout = sys.stdout
        self._buffer = []

    def write(self, msg):
        self._stdout.write(msg)
        self._buffer.append(msg)

    def flush(self):
        self._stdout.flush()

    def getvalue(self):
        return ''.join(self._buffer)

from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (
    accuracy_score, confusion_matrix, cohen_kappa_score,
    f1_score, precision_score, recall_score
)
from classical_baselines import ClassicalBaselineRunner
import seaborn as sns

from openai import OpenAI, AsyncOpenAI
import anthropic
import time
import asyncio
import traceback

from relaxed_perceptron import RelaxedPerceptron
from least_squares_inverse import LeastSquaresInverse

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════════
# SELEÇÃO DE MODELOS - ADICIONE OS MODELOS QUE DESEJA COMPARAR
# ═══════════════════════════════════════════════════════════════════════════

MODELS_TO_TEST = [
    ("openai", "gpt-4o-mini", 0.0),                  # OpenAI GPT-4o-mini
    # ("anthropic", "claude-sonnet-4-5", 0.0),           # Claude Sonnet 4.5
    # ("gemini", "gemini-2.0-flash", 0.0),             # Google Gemini
]

# URLs base e variáveis de ambiente com chaves de API por provedor
PROVIDER_CONFIG = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "client_type": "openai",
    },
    "anthropic": {
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
}


def get_client(provider: str) -> Union[OpenAI, anthropic.Anthropic]:
    """Cria o cliente de API para o provedor especificado (OpenAI, Anthropic ou Gemini)."""
    config = PROVIDER_CONFIG[provider]

    if config["client_type"] == "anthropic":
        return anthropic.Anthropic(api_key=os.getenv(config["api_key_env"]))
    else:
        if config["base_url"]:
            return OpenAI(
                api_key=os.getenv(config["api_key_env"]),
                base_url=config["base_url"]
            )
        else:
            return OpenAI(api_key=os.getenv(config["api_key_env"]))


def get_async_client(provider: str) -> Optional[AsyncOpenAI]:
    """Cria o cliente assíncrono para chamadas concorrentes (apenas OpenAI/Gemini).

    Timeout de 60s previne que uma única requisição travada bloqueie o semaphore
    indefinidamente (causa observada: terminal parou silenciosamente em run anterior).
    """
    config = PROVIDER_CONFIG[provider]
    if config["client_type"] == "anthropic":
        return None
    if config["base_url"]:
        return AsyncOpenAI(
            api_key=os.getenv(config["api_key_env"]),
            base_url=config["base_url"],
            timeout=60.0,
            max_retries=2,
        )
    else:
        return AsyncOpenAI(
            api_key=os.getenv(config["api_key_env"]),
            timeout=60.0,
            max_retries=2,
        )


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

# Log de todas as interações com a LLM (prompt, resposta bruta, parsing)
LLM_INTERACTIONS = []

# Bloco 1 (A) e Bloco 2 (E) usam mais amostras — envolvem aprendizado.
# Problemas B e C (testes de consistência) precisam só de 100 pontos para Kappa/F1 confiáveis.
N_SAMPLES_PROBLEM_A = 150  # Bloco 1: treino da métrica via otim. inversa
N_SAMPLES_PROBLEM_B = 100  # Bloco 1: teste de consistência (rotação horária ±1.5)
N_SAMPLES_PROBLEM_C = 100  # Bloco 1: teste de consistência (rotação anti-horária ±1.5)
N_SAMPLES_PROBLEM_E = 150  # Bloco 2: perito linear (Fase E)
# Meia-lua (Bloco 1 = Problema D; Bloco 2 = Problema F) usa N_SAMPLES_PROBLEM_A=150 também
# FEW_SHOT_SIZES = [0, 5]  # Tamanhos few-shot para Fases B e C
FEW_SHOT_SIZES = [0, 5, 10, 20, 40]  # Tamanhos few-shot para Fases B e C
# FEW_SHOT_SIZES_PHASE_E = [0, 5]  # versão curta (smoke-test, alinhada com FEW_SHOT_SIZES)
FEW_SHOT_SIZES_PHASE_E = [0, 5, 10, 20, 40]  # versão completa
# N_REPETICOES = 1
N_REPETICOES = 3

# Múltiplas sementes aleatórias para garantir robustez dos resultados
# RANDOM_SEEDS = [42]
RANDOM_SEEDS = [42, 123, 7]
# RANDOM_SEEDS = [42, 123, 7, 256, 999]

# Controle de limite de requisições (rate limit)
MAX_RETRIES = 5
INITIAL_BACKOFF = 10

# Concorrência máxima para chamadas async à API (ajustar conforme tier do OpenAI)
# Tier 1: ~8, Tier 2+: 15-20
MAX_CONCURRENCY = 10

# Máximo de retentativas para respostas malformadas do LLM
MAX_FORMAT_RETRIES = 5

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

# Métrica do perito para o Bloco 2 / Fase E
# O perito usa uma métrica ANISOTRÓPICA que pondera x2 muito mais do que x1.
# Isso torna a classificação não trivial: o LLM não pode se basear apenas em x1.
EXPERT_W = np.array([0.3, 1.5])  # Weights x2 heavily
EXPERT_CENTROIDS = np.array([[-1.5, 1.0], [1.5, -1.0]])

# Estratégias de dificuldade de exemplos para a Fase E
EXAMPLE_STRATEGIES = ["easy", "hard", "mixed", "random"]

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
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class LearnedMetric:
    """Armazena a métrica estimada/inferida no Problema A via otimização inversa.

    Nota terminológica: usamos 'estimada' ou 'inferida' (não 'aprendida') porque
    a métrica é obtida por otimização inversa a partir de decisões observadas,
    não por aprendizado supervisionado direto. O campo w_aprendido mantém o nome
    por compatibilidade com CSVs existentes.
    """
    w: np.ndarray
    centroids: np.ndarray
    gamma: float
    source_problem: str


@dataclass
class ConsistencyMetrics:
    """Armazena métricas detalhadas de consistência entre predições do LLM e da métrica estimada."""
    accuracy: float
    cohen_kappa: float
    f1_score: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray
    n_agreements: int
    n_disagreements: int
    disagreement_indices: np.ndarray

    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.1%} | "
            f"Kappa: {self.cohen_kappa:.3f} | "
            f"F1: {self.f1_score:.3f}"
        )


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
    # Parâmetros aprendidos pela métrica
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
    example_strategy: str  # estratégia de seleção: "easy", "hard", "mixed" ou "random"
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


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def print_section(title: str, char: str = "="):
    """Imprime um cabeçalho de seção formatado."""
    line = char * 70
    print(f"\n{line}")
    print(f" {title}")
    print(f"{line}\n")


def print_box(text: str):
    """Imprime texto em uma caixa formatada."""
    lines = text.strip().split('\n')
    max_len = min(max(len(line) for line in lines), 90)
    print("┌" + "─" * (max_len + 2) + "┐")
    for line in lines:
        if len(line) > max_len:
            line = line[:max_len-3] + "..."
        print(f"│ {line:<{max_len}} │")
    print("└" + "─" * (max_len + 2) + "┘")


def compute_consistency_metrics(
    y_llm: np.ndarray,
    y_metric: np.ndarray
) -> ConsistencyMetrics:
    """Calcula métricas detalhadas de consistência entre predições do LLM e da métrica estimada.

    Nota sobre concordância simétrica:
    - accuracy e kappa são métricas SIMÉTRICAS — o resultado é idêntico independentemente
      de qual vetor é tratado como "referência". Kappa é a métrica principal para H1.
    - F1, precision e recall NÃO são simétricas por construção; para preservar a simetria,
      calcula-se a média das duas direções (LLM→métrica e métrica→LLM), evitando que uma
      das partes seja arbitrariamente elevada a "ground truth".
    - confusion_matrix usa y_metric como referência (linhas = rótulos da métrica).
    """
    accuracy = accuracy_score(y_metric, y_llm)
    kappa = cohen_kappa_score(y_metric, y_llm)

    # F1/precision/recall simétricos: média das duas direções possíveis
    f1 = (
        f1_score(y_metric, y_llm, average='weighted', zero_division=0) +
        f1_score(y_llm, y_metric, average='weighted', zero_division=0)
    ) / 2
    precision = (
        precision_score(y_metric, y_llm, average='weighted', zero_division=0) +
        precision_score(y_llm, y_metric, average='weighted', zero_division=0)
    ) / 2
    recall = (
        recall_score(y_metric, y_llm, average='weighted', zero_division=0) +
        recall_score(y_llm, y_metric, average='weighted', zero_division=0)
    ) / 2
    cm = confusion_matrix(y_metric, y_llm)

    disagreements = y_llm != y_metric
    n_disagreements = np.sum(disagreements)
    n_agreements = len(y_llm) - n_disagreements
    disagreement_indices = np.where(disagreements)[0]

    return ConsistencyMetrics(
        accuracy=accuracy,
        cohen_kappa=kappa,
        f1_score=f1,
        precision=precision,
        recall=recall,
        confusion_matrix=cm,
        n_agreements=n_agreements,
        n_disagreements=n_disagreements,
        disagreement_indices=disagreement_indices
    )


# =============================================================================
# GERAÇÃO DE CONJUNTOS DE DADOS
# =============================================================================

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


def _save_panels_individually(panels, combined_filename, dpi=150):
    """Salva cada painel de uma figura composta como imagem individual.

    Args:
        panels: lista de (sufixo, draw_func, figsize) onde:
            - sufixo: string adicionada ao nome do arquivo (ex: "problema_a")
            - draw_func: callable(ax) que desenha em um único eixo
            - figsize: (largura, altura) da figura individual
        combined_filename: caminho completo do arquivo combinado (ex: "pasta/01_all.png")
        dpi: resolução das imagens individuais
    """
    if not combined_filename:
        return
    base, ext = os.path.splitext(combined_filename)
    for suffix, draw_func, figsize in panels:
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=figsize)
        draw_func(ax_ind)
        fig_ind.tight_layout()
        fig_ind.savefig(f"{base}_{suffix}{ext}", dpi=dpi, bbox_inches='tight')
        plt.close(fig_ind)


def visualize_all_problems(
    X_a: np.ndarray, y_a: np.ndarray,
    X_b: np.ndarray, y_b: np.ndarray,
    X_c: np.ndarray, y_c: np.ndarray,
    filename: str = None
):
    """Visualiza os três problemas sintéticos lado a lado para inspeção visual da geometria."""
    problems = [
        (X_a, y_a, PROBLEM_A_CENTERS, PROBLEM_A_STD, "PROBLEMA A\n(Aprendizado da Métrica)"),
        (X_b, y_b, PROBLEM_B_CENTERS, PROBLEM_B_STD, "PROBLEMA B\n(Teste de Consistência 1)"),
        (X_c, y_c, PROBLEM_C_CENTERS, PROBLEM_C_STD, "PROBLEMA C\n(Teste de Consistência 2)")
    ]

    def _draw_problem(ax, X, y, centers, std, title):
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*centers[0], marker='*', s=300, c='blue',
                   edgecolor='black', linewidth=2, label='Centro 0', zorder=5)
        ax.scatter(*centers[1], marker='*', s=300, c='red',
                   edgecolor='black', linewidth=2, label='Centro 1', zorder=5)
        ax.set_title(f"{title}\nCentros: {centers}, σ={std}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5, 5)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (X, y, centers, std, title) in zip(axes, problems):
        _draw_problem(ax, X, y, centers, std, title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["problema_a", "problema_b", "problema_c"]
    panels = [
        (suffixes[i], lambda ax, p=problems[i]: _draw_problem(ax, *p), (7, 5))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def visualize_problem_e_with_expert(
    X_e: np.ndarray, y_gt_e: np.ndarray,
    y_expert_e: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    filename: str = None
):
    """
    Visualiza o Problema E (perito linear do Bloco 2) com a fronteira de decisão do perito.
    Compara os rótulos verdadeiros (ground truth) com os rótulos do perito lado a lado.
    """
    # Precomputa grid para reutilizar nos painéis
    x_min, x_max = -6, 6
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, expert_centroids, expert_w)
    Z = Z.reshape(xx.shape)
    confidences, _ = compute_metric_confidence(grid_points, expert_centroids, expert_w)
    conf_grid = confidences.reshape(xx.shape)

    def _draw_ground_truth(ax):
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_gt_e, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*PROBLEM_E_CENTERS[0], marker='*', s=300, c='blue',
                   edgecolor='black', linewidth=2, label='Centro Verdadeiro 0', zorder=5)
        ax.scatter(*PROBLEM_E_CENTERS[1], marker='*', s=300, c='red',
                   edgecolor='black', linewidth=2, label='Centro Verdadeiro 1', zorder=5)
        ax.set_title("Problema E: Rótulos Verdadeiros (Ground Truth)", fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    def _draw_expert_boundary(ax):
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_expert_e, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*expert_centroids[0], marker='X', s=300, c='blue',
                   edgecolor='k', linewidth=2, label='Centróide do Perito 0', zorder=5)
        ax.scatter(*expert_centroids[1], marker='X', s=300, c='red',
                   edgecolor='k', linewidth=2, label='Centróide do Perito 1', zorder=5)
        ax.set_title(f"Rótulos do Perito (W=[{expert_w[0]:.1f}, {expert_w[1]:.1f}])",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    def _draw_margin_heatmap(ax):
        im = ax.contourf(xx, yy, conf_grid, levels=20, cmap="RdYlGn")
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        plt.colorbar(im, ax=ax, label="Margem (confiança)")
        ax.scatter(X_e[:, 0], X_e[:, 1], c='black', alpha=0.3, s=20)
        ax.set_title("Margem do Perito (Distância à Fronteira)", fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_ground_truth(axes[0])
    _draw_expert_boundary(axes[1])
    _draw_margin_heatmap(axes[2])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("ground_truth", _draw_ground_truth, (7, 5)),
        ("fronteira_perito", _draw_expert_boundary, (7, 5)),
        ("margem_perito", _draw_margin_heatmap, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


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


def llm_classify_point_openai(client: OpenAI, model_name: str, prompt: str,
                               temperature: float,
                               system_message: str = "You are a classifier. Respond only with the class label.") -> str:
    """Classifica um ponto usando API compatível com OpenAI (OpenAI ou Gemini)."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


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
    return response.content[0].text.strip()


async def async_llm_classify_point_openai(async_client, model_name: str, prompt: str,
                                           temperature: float,
                                           system_message: str = "You are a classifier. Respond only with the class label.") -> str:
    """Versão assíncrona de llm_classify_point_openai para chamadas concorrentes.

    Envolve a chamada em asyncio.wait_for(timeout=90s) como segunda barreira contra
    travamentos: se a requisição não responder em 90s mesmo após retries do SDK,
    levanta TimeoutError em vez de bloquear o semaphore indefinidamente.
    """
    async def _do_call():
        response = await async_client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    try:
        return await asyncio.wait_for(_do_call(), timeout=90.0)
    except asyncio.TimeoutError:
        # Devolve string vazia — será tratada como malformada e cai no fallback MD5
        return ""


def parse_llm_response(
    label: str,
    nome_classe_0: str,
    nome_classe_1: str
) -> Tuple[Optional[str], bool]:
    """Interpreta a resposta do LLM e verifica se é válida (corresponde a uma das classes).

    Tentativas em ordem crescente de permissividade:
      0. Extração da última linha e marcadores CoT ("Final answer:", etc.)
      1. Match exato (case-insensitive)
      2. Match exato após remoção de aspas e pontuação marginal
      3. Classe numérica: "0"/"1" quando os nomes são dígitos
      4. Word-boundary regex (evita falsos positivos em substrings)
      5. Contains exclusivo (apenas uma das classes aparece no texto)
      6. Padrões comuns de resposta ("Class X", "Answer is X", ...)
      7. Starts-with (a resposta começa com o nome da classe)
    """
    label_clean = label.strip()
    label_upper = label_clean.upper()
    nome_0_upper = nome_classe_0.upper()
    nome_1_upper = nome_classe_1.upper()

    # 0. Extração CoT: tenta marcadores de resposta final e última linha
    # Necessário para variante chain-of-thought que produz raciocínio antes da resposta
    if '\n' in label_clean:
        # Tenta extrair após marcadores comuns de resposta final
        cot_markers = [
            r'(?:final\s+answer|final\s+classification|my\s+(?:final\s+)?answer|therefore|classification)\s*[:=]\s*',
        ]
        for marker_pattern in cot_markers:
            match = re.search(marker_pattern + r'(.+)', label_clean, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().strip('"\'.,!? ')
                extracted_upper = extracted.upper()
                if extracted_upper == nome_0_upper:
                    return nome_classe_0, True
                if extracted_upper == nome_1_upper:
                    return nome_classe_1, True

        # Tenta a última linha não-vazia como resposta
        last_line = [l.strip() for l in label_clean.split('\n') if l.strip()][-1]
        last_line_clean = re.sub(r'^[\s\'\"]+|[\s\'\".,!?]+$', '', last_line)
        last_upper = last_line_clean.upper()
        if last_upper == nome_0_upper:
            return nome_classe_0, True
        if last_upper == nome_1_upper:
            return nome_classe_1, True

    # 1. Match exato
    if label_upper == nome_0_upper:
        return nome_classe_0, True
    if label_upper == nome_1_upper:
        return nome_classe_1, True

    # 2. Remove aspas/pontuação marginal e tenta novamente
    stripped = re.sub(r'^[\s\'\"]+|[\s\'\".,!?]+$', '', label_clean)
    stripped_upper = stripped.upper()
    if stripped_upper == nome_0_upper:
        return nome_classe_0, True
    if stripped_upper == nome_1_upper:
        return nome_classe_1, True

    # 3. Classes numéricas: se o nome for "0"/"1", aceita variações como " 0 " ou "Class 0"
    if nome_0_upper.isdigit() and nome_1_upper.isdigit():
        nums_found = re.findall(r'\b\d+\b', label_upper)
        hits_0 = nums_found.count(nome_0_upper)
        hits_1 = nums_found.count(nome_1_upper)
        if hits_0 > hits_1:
            return nome_classe_0, True
        if hits_1 > hits_0:
            return nome_classe_1, True

    # 4. Word-boundary regex (mais preciso que contains para nomes curtos como "A"/"B")
    pat_0 = re.compile(r'\b' + re.escape(nome_0_upper) + r'\b')
    pat_1 = re.compile(r'\b' + re.escape(nome_1_upper) + r'\b')
    wb_0 = bool(pat_0.search(label_upper))
    wb_1 = bool(pat_1.search(label_upper))
    if wb_0 and not wb_1:
        return nome_classe_0, True
    if wb_1 and not wb_0:
        return nome_classe_1, True

    # 5. Contains exclusivo (apenas uma das classes aparece em qualquer posição)
    contains_0 = nome_0_upper in label_upper
    contains_1 = nome_1_upper in label_upper
    if contains_0 and not contains_1:
        return nome_classe_0, True
    if contains_1 and not contains_0:
        return nome_classe_1, True

    # 6. Padrões comuns de resposta em linguagem natural
    common_patterns = [
        f"CLASS {nome_0_upper}", f"CLASS {nome_1_upper}",
        f"CLASSE {nome_0_upper}", f"CLASSE {nome_1_upper}",
        f"ANSWER IS {nome_0_upper}", f"ANSWER IS {nome_1_upper}",
        f"ANSWER: {nome_0_upper}", f"ANSWER: {nome_1_upper}",
        f"CLASSIFICATION: {nome_0_upper}", f"CLASSIFICATION: {nome_1_upper}",
        f"CLASSIFIED AS {nome_0_upper}", f"CLASSIFIED AS {nome_1_upper}",
        f"BELONGS TO {nome_0_upper}", f"BELONGS TO {nome_1_upper}",
        f"IS {nome_0_upper}", f"IS {nome_1_upper}",
    ]
    for pattern in common_patterns:
        if pattern in label_upper:
            return (nome_classe_0 if nome_0_upper in pattern else nome_classe_1), True

    # 7. Starts-with (a resposta começa diretamente com o nome da classe)
    if label_upper.startswith(nome_0_upper):
        return nome_classe_0, True
    if label_upper.startswith(nome_1_upper):
        return nome_classe_1, True

    return None, False


def llm_classify_point(
    x1: float, x2: float,
    nome_classe_0: str, nome_classe_1: str,
    examples: Optional[List] = None,
    malformed_counter: Optional[List] = None,
    raw_out: Optional[List] = None,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None
) -> str:
    """Classifica um único ponto usando o LLM, com lógica de retentativa para respostas malformadas.

    raw_out: lista de 1 elemento passada por referência; ao retornar, raw_out[0] conterá
             a resposta bruta original do LLM (antes do parsing), útil para logging.
    extra_features: lista de (nome, valor) para features adicionais.
    """
    global client, MODEL_NAME, CURRENT_PROVIDER, CURRENT_TEMPERATURE

    if examples is None or len(examples) == 0:
        prompt = build_prompt_zero_shot(x1, x2, nome_classe_0, nome_classe_1,
                                        nome_feature_0, nome_feature_1, extra_features)
    else:
        prompt = build_prompt_few_shot(x1, x2, examples, nome_classe_0, nome_classe_1,
                                       nome_feature_0, nome_feature_1, extra_features)

    all_responses = []

    for format_attempt in range(MAX_FORMAT_RETRIES):
        for rate_attempt in range(MAX_RETRIES):
            try:
                config = PROVIDER_CONFIG[CURRENT_PROVIDER]

                if config["client_type"] == "anthropic":
                    label = llm_classify_point_anthropic(client, MODEL_NAME, prompt, CURRENT_TEMPERATURE)
                else:
                    label = llm_classify_point_openai(client, MODEL_NAME, prompt, CURRENT_TEMPERATURE)

                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower() or "overloaded" in error_str.lower():
                    wait_time = INITIAL_BACKOFF * (2 ** rate_attempt)
                    print(f"  ⏳ Limite de requisições atingido. Aguardando {wait_time}s (tentativa {rate_attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                else:
                    raise e
        else:
            raise Exception(f"Máximo de tentativas ({MAX_RETRIES}) excedido por limite de requisições")

        all_responses.append(label)
        # Salva a primeira resposta bruta para quem chamar com raw_out
        if raw_out is not None and len(raw_out) == 0:
            raw_out.append(label)

        parsed_label, is_valid = parse_llm_response(label, nome_classe_0, nome_classe_1)

        if is_valid:
            return parsed_label

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

    if malformed_counter is not None:
        malformed_counter[0] += 1
    if raw_out is not None and len(raw_out) == 0:
        raw_out.append(all_responses[0] if all_responses else "")

    # Fallback para respostas malformadas: atribui classe aleatória uniforme (50/50),
    # usando hash das coordenadas como seed para reprodutibilidade entre execuções.
    #
    # NOTA IMPORTANTE: o método anterior (coord_bits % 2) introduzia viés geométrico
    # sistemático — a paridade da soma de coordenadas correlaciona com a posição espacial,
    # contaminando a métrica estimada. O hash distribui uniformemente sem padrão espacial.
    #
    # Se a taxa de fallback for significativa (>5%), os resultados desta seed/config
    # devem ser interpretados com cautela — o ruído do fallback pode dominar.
    import hashlib
    hash_seed = hashlib.md5(f"{x1:.6f}_{x2:.6f}".encode()).hexdigest()
    fallback_bit = int(hash_seed, 16) % 2
    fallback = nome_classe_0 if fallback_bit == 0 else nome_classe_1

    warnings.warn(
        f"\n  ⚠️ RESPOSTA MALFORMADA após {MAX_FORMAT_RETRIES} tentativas para o ponto ({x1:.4f}, {x2:.4f}).\n"
        f"     Respostas recebidas: {all_responses}\n"
        f"     Esperado: '{nome_classe_0}' ou '{nome_classe_1}'\n"
        f"     Fallback aleatório (hash-based): '{fallback}'\n"
    )

    return fallback


async def async_llm_classify_point(
    x1: float, x2: float,
    nome_classe_0: str, nome_classe_1: str,
    examples: Optional[List] = None,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None,
    prompt_variant: str = "default"
) -> Tuple[str, str, bool]:
    """Versão assíncrona de llm_classify_point. Retorna (label_parsed, raw_response, was_malformed)."""
    global async_client, MODEL_NAME, CURRENT_PROVIDER, CURRENT_TEMPERATURE

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

    for format_attempt in range(MAX_FORMAT_RETRIES):
        for rate_attempt in range(MAX_RETRIES):
            try:
                config = PROVIDER_CONFIG[CURRENT_PROVIDER]

                if config["client_type"] == "anthropic":
                    # Anthropic não tem async_client neste script; fallback sync em thread
                    label = await asyncio.get_event_loop().run_in_executor(
                        None, llm_classify_point_anthropic, client, MODEL_NAME, prompt,
                        CURRENT_TEMPERATURE, system_msg, max_tok
                    )
                else:
                    label = await async_llm_classify_point_openai(async_client, MODEL_NAME, prompt,
                                                                   CURRENT_TEMPERATURE, system_msg)

                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower() or "overloaded" in error_str.lower() or "400" in error_str or "could not parse" in error_str.lower():
                    wait_time = INITIAL_BACKOFF * (2 ** rate_attempt)
                    print(f"  ⏳ Erro retentável ({error_str[:80]}). Aguardando {wait_time}s (tentativa {rate_attempt + 1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        else:
            raise Exception(f"Máximo de tentativas ({MAX_RETRIES}) excedido por limite de requisições")

        all_responses.append(label)
        all_prompts.append(prompt)
        parsed_label, is_valid = parse_llm_response(label, nome_classe_0, nome_classe_1)

        if is_valid:
            raw = all_responses[0] if all_responses else ""
            LLM_INTERACTIONS.append({
                "point": {"x1": x1, "x2": x2},
                "prompt": all_prompts[0],
                "raw_response": raw,
                "parsed_label": parsed_label,
                "model": MODEL_NAME,
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

    LLM_INTERACTIONS.append({
        "point": {"x1": x1, "x2": x2},
        "prompt": all_prompts[0] if all_prompts else "",
        "raw_response": raw,
        "all_responses": all_responses,
        "parsed_label": fallback,
        "model": MODEL_NAME,
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

        async with semaphore:
            label, raw, was_malformed = await async_llm_classify_point(
                x1, x2, nome_classe_0, nome_classe_1,
                examples, nome_feature_0, nome_feature_1, extra_feats,
                prompt_variant=prompt_variant
            )

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

def d_W(x: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
    """Distância de Mahalanobis com matriz diagonal W.

    Justificativa: a restrição diagonal reduz a complexidade de O(d²) para O(d),
    tornando o aprendizado tratável sem perda crítica de poder discriminativo em 2D.
    """
    return np.sum(w * (x - c)**2)


def compute_centroids(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula os centróides de cada classe a partir dos rótulos do LLM.

    Os centróides são usados como âncoras para a métrica de Mahalanobis:
    um ponto é classificado na classe cujo centróide está mais próximo sob d_W.
    """
    classes = np.unique(y)
    centroids = np.array([X[y == c].mean(axis=0) for c in classes])
    return centroids


def train_relaxed_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    centroids: np.ndarray,
    eta: float = 0.001,
    C: float = 1.0,
    gamma_init: float = 0.1,
    delta_gamma: float = 0.1,
    max_epochs: int = 100,
    tol: float = 1e-5,
    verbose: bool = False,
    use_best_effort: bool = False,
    return_history: bool = False,
):
    """Wrapper que delega para RelaxedPerceptron.

    Defaults seguem Coelho, Borges & Fonseca Neto (CILAMCE 2017, Seção 6, p. 16):
    "a taxa de aprendizado η = 0.001 e a constante C variou de 1 até 0.1".

    Se return_history=True, retorna (w, gamma, gamma_history) onde gamma_history
    é a lista de dicts capturada durante a busca binária em γ (item b da reunião
    30/04/2026). Caso contrário, retorna apenas (w, gamma) — compat. com chamadas existentes.
    """
    model = RelaxedPerceptron(
        eta=eta, C=C, gamma_init=gamma_init, delta_gamma=delta_gamma,
        max_epochs=max_epochs, tol=tol, use_best_effort=use_best_effort,
        verbose=verbose,
    )
    w, gamma = model.fit(X, y, centroids)
    if return_history:
        return w, gamma, model.gamma_history
    return w, gamma


def train_least_squares_inverse(
    X: np.ndarray,
    y: np.ndarray,
    centroids: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """Wrapper que delega para LeastSquaresInverse."""
    model = LeastSquaresInverse(verbose=verbose)
    return model.fit(X, y, centroids)


def augment_to_r3(X: np.ndarray) -> np.ndarray:
    """Adiciona feature de interação x3 = x1 * x2, projetando dados de R2 para R3.

    Usada no experimento de não-linearidade implícita: o LLM classifica apenas
    com (x1, x2), mas nos bastidores adicionamos x3 = x1·x2 e tentamos aprender
    uma métrica diagonal com 3 pesos. Se a fidelidade com 3 pesos superar a de
    2 pesos, há evidência de que o LLM adota implicitamente um critério não-linear.
    O vetor W terá 3 componentes; projetado de volta para R2, a fronteira de
    decisão corresponde a uma hipérbole.
    """
    x3 = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, x3])


def augment_to_r4(X: np.ndarray) -> np.ndarray:
    """Adiciona features quadráticas x1², x2² para projetar dados em R4.

    Justificativa (e-mail orientador 19:15): para o estudo de caso peso×altura,
    o classificador ótimo bayesiano é
        f(x1, x2) = x2² - x2 + x1² - x1 + cte
    — fronteira elíptica. Com 4 features (x1, x2, x1², x2²), a métrica diagonal
    consegue representar exatamente essa fronteira como combinação linear,
    enquanto com 3 features (x1·x2) só representa hipérbole.
    """
    x_sq = X ** 2
    return np.hstack([X, x_sq])


def augment_features(X: np.ndarray, n_features: int) -> np.ndarray:
    """Aumenta X para o número de features pedido (2, 3 ou 4).

    - 2: (x1, x2) — original
    - 3: (x1, x2, x1·x2) — hipérbole (augment_to_r3)
    - 4: (x1, x2, x1², x2²) — elipse (augment_to_r4)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < 2:
        raise ValueError(f"X precisa ter ao menos 2 colunas; tem {X.shape[1]}")

    X2 = X[:, :2]
    if n_features == 2:
        return X2
    if n_features == 3:
        return augment_to_r3(X2)
    if n_features == 4:
        return augment_to_r4(X2)
    raise ValueError(f"n_features deve ser 2, 3 ou 4 — recebido {n_features}")


def predict_with_metric(X: np.ndarray, centroids: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Prediz a classe de cada ponto usando a métrica estimada (vizinho mais próximo sob d_W)."""
    predictions = []
    for xi in X:
        distances = [d_W(xi, c, w) for c in centroids]
        predictions.append(np.argmin(distances))
    return np.array(predictions)


def compute_metric_confidence(
    X: np.ndarray,
    centroids: np.ndarray,
    w: np.ndarray,
    y_pred: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula pontuações de confiança baseadas em margem usando a fórmula do orientador.

    Definição de margem (orientador): margem = d_W(x, centróide_errado) - d_W(x, centróide_previsto)

    No caso de 2 classes (centróides[0] e centróides[1]):
        - Se classe predita == 0: d_pred = d0, d_errado = d1, logo margem = d1 - d0
        - Se classe predita == 1: d_pred = d1, d_errado = d0, logo margem = d0 - d1
        - Isso sempre equivale a |d1 - d0|, confirmando equivalência com a diferença absoluta de distâncias.

    margem >= 0 é garantida por construção: a classe predita sempre tem a menor
    (ou igual) distância ao seu centróide, logo d_errado >= d_pred por definição.
    """
    n_samples = X.shape[0]
    confidences = np.zeros(n_samples)
    predictions = np.zeros(n_samples, dtype=int)

    for i, xi in enumerate(X):
        d0 = d_W(xi, centroids[0], w)
        d1 = d_W(xi, centroids[1], w)

        if d0 <= d1:
            predictions[i] = 0
            d_pred = d0
            d_wrong = d1
        else:
            predictions[i] = 1
            d_pred = d1
            d_wrong = d0
        # Fórmula do orientador: margem = d_W(x, centróide_errado) - d_W(x, centróide_previsto)
        # No caso de 2 classes, isso sempre equivale a |d1 - d0|: uma distância é d_pred e a outra d_wrong.
        # margem >= 0 é garantida: a classe predita sempre tem a menor (ou igual) distância.
        margin = d_wrong - d_pred
        assert margin >= 0, "Margem deve ser não-negativa por construção"
        confidences[i] = margin

    if y_pred is not None:
        assert np.array_equal(predictions, y_pred), "Discordância de predição no cálculo de confiança"

    return confidences, predictions


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
        eta=0.001, C=1.0, delta_gamma=0.05,  # Coelho et al. CILAMCE 2017, p. 16
        max_epochs=50, tol=1e-4, verbose=verbose,
        use_best_effort=True,  # retorna melhor W parcial quando separação perfeita é impossível
        return_history=True,   # captura evolução de γ para diagnóstico (item b reunião 30/04)
    )
    # Anota diagnóstico de γ para esta chamada (Fase A no Problema A)
    PERCEPTRON_GAMMA_DIAGNOSTICS.append({
        "label": "Fase A — Problema A",
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
    n_per_class: int,
    nome_classe_0: str,
    nome_classe_1: str,
    verbose: bool = True
) -> Tuple[List[Tuple[float, float, str]], np.ndarray, np.ndarray, np.ndarray]:
    """Seleciona exemplos few-shot por Aprendizado Ativo (maior margem por classe).

    Justificativa: exemplos com maior margem são os mais representativos de cada classe —
    estão longe da fronteira, são os casos mais claros e ancoramos bem o LLM ao padrão da métrica.
    """
    confidences, y_pred_metric = compute_metric_confidence(X, centroids, w)

    idx_class_0 = np.where(y_pred_metric == 0)[0]
    idx_class_1 = np.where(y_pred_metric == 1)[0]

    conf_class_0 = confidences[idx_class_0]
    conf_class_1 = confidences[idx_class_1]

    sorted_idx_0 = idx_class_0[np.argsort(-conf_class_0)]
    sorted_idx_1 = idx_class_1[np.argsort(-conf_class_1)]

    selected_idx_0 = sorted_idx_0[:n_per_class]
    selected_idx_1 = sorted_idx_1[:n_per_class]

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

        n_per_class = n_shot // 2
        examples, _, _, selected_indices = select_confident_examples(
            X, learned_metric.centroids, learned_metric.w,
            n_per_class, nome_classe_0, nome_classe_1, verbose=verbose
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

    true_centroids_d = np.array(PROBLEM_E_CENTERS)

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


def _oracle_algorithms(X, y, centroids):
    """Retorna iterador de (nome, funcao_treino) para os algoritmos de otimização inversa."""
    yield "perceptron", lambda: train_relaxed_perceptron(
        X, y, centroids,
        eta=0.001, C=1.0, delta_gamma=0.05,  # Coelho et al. CILAMCE 2017, p. 16
        max_epochs=50, tol=1e-4, verbose=False,
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
    w_norm_true = true_w / np.sum(true_w)

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
        n_examples: Número total de exemplos a selecionar
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

    n_per_class = n_examples // 2

    idx_class_0 = np.where(y_expert == 0)[0]
    idx_class_1 = np.where(y_expert == 1)[0]

    # Aviso de desbalanceamento: alerta se uma classe tiver menos pontos do que o solicitado
    if len(idx_class_0) < n_per_class:
        print(f"  AVISO: Classe 0 tem apenas {len(idx_class_0)} pontos disponíveis, "
              f"mas {n_per_class} foram solicitados por classe. A seleção será truncada.")
    if len(idx_class_1) < n_per_class:
        print(f"  AVISO: Classe 1 tem apenas {len(idx_class_1)} pontos disponíveis, "
              f"mas {n_per_class} foram solicitados por classe. A seleção será truncada.")

    conf_class_0 = confidences[idx_class_0]
    conf_class_1 = confidences[idx_class_1]

    if strategy == "easy":
        # Seleciona pontos com MAIOR margem (longe da fronteira, mais confiantes)
        # Justificativa: exemplos "fáceis" são os mais representativos de cada classe
        sorted_0 = idx_class_0[np.argsort(-conf_class_0)]  # Ordem decrescente (maior margem primeiro)
        sorted_1 = idx_class_1[np.argsort(-conf_class_1)]
        selected_0 = sorted_0[:n_per_class]
        selected_1 = sorted_1[:n_per_class]

    elif strategy == "hard":
        # Seleciona pontos com MENOR margem (próximos à fronteira, mais ambíguos)
        # Justificativa: exemplos "difíceis" testam se o LLM capta a fronteira com precisão
        sorted_0 = idx_class_0[np.argsort(conf_class_0)]  # Ordem crescente (menor margem primeiro)
        sorted_1 = idx_class_1[np.argsort(conf_class_1)]
        selected_0 = sorted_0[:n_per_class]
        selected_1 = sorted_1[:n_per_class]

    elif strategy == "mixed":
        # Metade fáceis (maior margem) + metade difíceis (menor margem) por classe
        # Justificativa: cobertura balanceada da fronteira e das regiões centrais de cada classe
        n_easy = n_per_class // 2
        n_hard = n_per_class - n_easy

        sorted_0_desc = idx_class_0[np.argsort(-conf_class_0)]
        sorted_0_asc = idx_class_0[np.argsort(conf_class_0)]
        sorted_1_desc = idx_class_1[np.argsort(-conf_class_1)]
        sorted_1_asc = idx_class_1[np.argsort(conf_class_1)]

        easy_0 = sorted_0_desc[:n_easy]
        hard_0 = sorted_0_asc[:n_hard]
        easy_1 = sorted_1_desc[:n_easy]
        hard_1 = sorted_1_asc[:n_hard]

        # Remove duplicatas preservando a ordem de inserção (easy primeiro, depois hard).
        # np.unique não é usado aqui pois ordena numericamente e destruiria o balanço easy/hard.
        def _unique_ordered(arr: np.ndarray) -> np.ndarray:
            seen = set()
            return np.array([x for x in arr if not (x in seen or seen.add(x))], dtype=arr.dtype)

        selected_0 = _unique_ordered(np.concatenate([easy_0, hard_0]))[:n_per_class]
        selected_1 = _unique_ordered(np.concatenate([easy_1, hard_1]))[:n_per_class]

    elif strategy == "random":
        # Seleção aleatória (linha de base sem critério estratégico)
        selected_0 = rng.choice(idx_class_0, size=min(n_per_class, len(idx_class_0)), replace=False)
        selected_1 = rng.choice(idx_class_1, size=min(n_per_class, len(idx_class_1)), replace=False)

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
# VISUALIZAÇÕES (FASES A-C)
# =============================================================================

def plot_consistency_comparison_extended(resultados: List[ResultadoExperimento], filename: str = None):
    """Gráfico comparativo estendido incluindo Problema C e métricas adicionais (Kappa, F1)."""
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (t={r.temperature})",
            'n_shot': r.n_shot,
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'f1_b': r.f1_problema_b,
            'f1_c': r.f1_problema_c,
            'fidelidade': r.fidelidade_problema_a,
        }
        for r in resultados
    ])

    df_cons = df.groupby('n_shot').agg({
        'consistencia_b': ['mean', 'std'], 'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_cons.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']
    x = np.arange(len(df_cons))
    width = 0.35

    df_kappa = df.groupby('n_shot').agg({'kappa_b': ['mean', 'std'], 'kappa_c': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']

    df_f1 = df.groupby('n_shot').agg({'f1_b': ['mean', 'std'], 'f1_c': ['mean', 'std']}).reset_index()
    df_f1.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']

    def _draw_consistency(ax):
        ax.bar(x - width/2, df_cons['b_mean'], width, yerr=df_cons['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_cons['c_mean'], width, yerr=df_cons['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_cons['n_shot']])
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência: Problema B vs C', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_kappa(ax):
        ax.bar(x - width/2, df_kappa['b_mean'], width, yerr=df_kappa['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_kappa['c_mean'], width, yerr=df_kappa['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_kappa['n_shot']])
        ax.set_ylabel("Kappa de Cohen")
        ax.set_title("Kappa de Cohen: B vs C", fontweight='bold')
        ax.legend(); ax.set_ylim(-0.1, 1.1)

    def _draw_f1(ax):
        ax.bar(x - width/2, df_f1['b_mean'], width, yerr=df_f1['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_f1['c_mean'], width, yerr=df_f1['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_f1['n_shot']])
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score: B vs C', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_boxplot(ax):
        from matplotlib.patches import Patch
        metrics_data = []
        for col, label in [('consistencia_b', 'Acu B'), ('consistencia_c', 'Acu C'),
                           ('kappa_b', 'Kappa B'), ('kappa_c', 'Kappa C'),
                           ('f1_b', 'F1 B'), ('f1_c', 'F1 C')]:
            for val in df[col]:
                metrics_data.append({'Metric': label, 'Value': val})
        df_metrics = pd.DataFrame(metrics_data)
        colors_bp = ['steelblue', 'coral'] * 3
        positions = [0, 0.6, 1.5, 2.1, 3.0, 3.6]
        for i, (metric, color) in enumerate(zip(['Acu B', 'Acu C', 'Kappa B', 'Kappa C', 'F1 B', 'F1 C'], colors_bp)):
            data = df_metrics[df_metrics['Metric'] == metric]['Value']
            bp = ax.boxplot([data], positions=[positions[i]], widths=0.4, patch_artist=True)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)
        ax.set_xticks([0.3, 1.8, 3.3])
        ax.set_xticklabels(['Acurácia', "Kappa de Cohen", 'F1-Score'])
        ax.set_ylabel('Pontuação')
        ax.set_title('Distribuição de Todas as Métricas', fontweight='bold')
        legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Problema B'),
                           Patch(facecolor='coral', alpha=0.7, label='Problema C')]
        ax.legend(handles=legend_elements, loc='lower right')

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_consistency(axes[0, 0])
    _draw_kappa(axes[0, 1])
    _draw_f1(axes[1, 0])
    _draw_boxplot(axes[1, 1])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_consistency, (7, 5)),
        ("kappa", _draw_kappa, (7, 5)),
        ("f1", _draw_f1, (7, 5)),
        ("distribuicao", _draw_boxplot, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_class_names_effect(resultados: List[ResultadoExperimento], filename: str = None):
    """Analisa o efeito dos nomes das classes na consistência do LLM com a métrica."""
    df = pd.DataFrame([
        {
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
        }
        for r in resultados
    ])

    df_names = df.groupby('nomes').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std']
    }).reset_index()
    df_names.columns = ['nomes', 'b_mean', 'b_std', 'c_mean', 'c_std']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_names))
    width = 0.35

    ax.bar(x - width/2, df_names['b_mean'], width, yerr=df_names['b_std'],
           label='Problema B', color='steelblue', capsize=3, edgecolor='black')
    ax.bar(x + width/2, df_names['c_mean'], width, yerr=df_names['c_std'],
           label='Problema C', color='coral', capsize=3, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df_names['nomes'], rotation=45, ha='right')
    ax.set_ylabel('Consistência Média')
    ax.set_title('Efeito dos Nomes das Classes na Consistência', fontsize=12, fontweight='bold')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara modelos se múltiplos foram testados no experimento."""
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (t={r.temperature})",
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'fidelidade': r.fidelidade_problema_a,
        }
        for r in resultados
    ])
    models = df['model'].unique()
    if len(models) < 2:
        print("  Pulando comparação entre modelos (apenas 1 modelo testado)")
        return

    df_models = df.groupby('model').agg({
        'consistencia_b': 'mean', 'consistencia_c': 'mean',
        'kappa_b': 'mean', 'kappa_c': 'mean', 'fidelidade': 'mean',
    }).reset_index().sort_values('consistencia_b', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    width = 0.35

    ax = axes[0]
    for i, (_, row) in enumerate(df_models.iterrows()):
        ax.bar(i - width/2, row['consistencia_b'], width, color=colors[i], edgecolor='black', alpha=0.8)
        ax.bar(i + width/2, row['consistencia_c'], width, color=colors[i], edgecolor='black', alpha=0.5, hatch='//')
    ax.set_xticks(range(len(df_models)))
    ax.set_xticklabels([m.split('/')[-1] for m in df_models['model']], rotation=45, ha='right')
    ax.set_ylabel('Consistência')
    ax.set_title('COMPARAÇÃO DE MODELOS: Consistência', fontweight='bold')
    ax.set_ylim(0, 1.1)

    ax = axes[1]
    metrics = ['consistencia_b', 'consistencia_c', 'kappa_b', 'kappa_c', 'fidelidade']
    labels = ['Consist. B', 'Consist. C', 'Kappa B', 'Kappa C', 'Fidelidade']
    x_metrics = np.arange(len(metrics))
    w = 0.8 / len(models)
    for i, (_, row) in enumerate(df_models.iterrows()):
        values = [row[m] for m in metrics]
        x_pos = x_metrics + i * w - (len(models) - 1) * w / 2
        ax.bar(x_pos, values, w * 0.9, color=colors[i], edgecolor='black', alpha=0.8,
               label=row['model'].split('/')[-1])
    ax.set_xticks(x_metrics)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Pontuação')
    ax.set_title('COMPARAÇÃO DE MODELOS: Todas as Métricas', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_seed_comparison(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara os resultados entre diferentes sementes aleatórias para avaliar robustez."""
    df = pd.DataFrame([
        {
            'seed': r.random_seed,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
        }
        for r in resultados
    ])
    seeds = sorted(df['seed'].unique())
    if len(seeds) < 2:
        print("  Pulando comparação entre sementes (apenas 1 semente testada)")
        return

    df_seeds = df.groupby('seed').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_seeds.columns = ['seed', 'b_mean', 'b_std', 'c_mean', 'c_std']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, df_seeds['b_mean'], width, yerr=df_seeds['b_std'],
           label='Problema B', color='steelblue', capsize=5, edgecolor='black')
    ax.bar(x + width/2, df_seeds['c_mean'], width, yerr=df_seeds['c_std'],
           label='Problema C', color='coral', capsize=5, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Semente {s}' for s in seeds])
    ax.set_ylabel('Consistência')
    ax.set_title('CONSISTÊNCIA POR SEMENTE ALEATÓRIA', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# VISUALIZAÇÕES DA FASE E (LLM como aprendiz)
# =============================================================================

def plot_phase_e_learning_curve(
    results_e: List[ResultadoPhaseEExperimento],
    filename: str = None
):
    """
    Curva de aprendizado mostrando como o LLM melhora com mais exemplos,
    discriminada por estratégia de seleção de exemplos.

    Esta é a VISUALIZAÇÃO PRINCIPAL da Fase E:
    - Eixo X: número de exemplos few-shot
    - Eixo Y: concordância LLM vs. Perito
    - Linhas: uma por estratégia (easy, hard, mixed, random)
    """
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name}",
            'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
            'f1': r.f1_llm_vs_expert,
        }
        for r in results_e
    ])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    strategy_colors = {
        'easy': '#2ecc71',    # Green
        'hard': '#e74c3c',    # Red
        'mixed': '#3498db',   # Blue
        'random': '#95a5a6',  # Gray
    }
    strategy_markers = {
        'easy': 'o',
        'hard': 's',
        'mixed': 'D',
        'random': '^',
    }

    metrics_config = [
        ('accuracy', 'Concordância LLM vs. Perito (Acurácia)'),
        ('kappa', "Kappa de Cohen"),
        ('f1', 'F1-Score'),
    ]

    def _draw_metric(ax, metric_col, metric_name):
        for strategy in EXAMPLE_STRATEGIES:
            df_strat = df[df['strategy'] == strategy]
            if len(df_strat) == 0:
                continue
            df_grouped = df_strat.groupby('n_shot').agg({
                metric_col: ['mean', 'std']
            }).reset_index()
            df_grouped.columns = ['n_shot', 'mean', 'std']
            df_grouped = df_grouped.sort_values('n_shot')
            ax.errorbar(
                df_grouped['n_shot'], df_grouped['mean'],
                yerr=df_grouped['std'],
                marker=strategy_markers[strategy],
                color=strategy_colors[strategy],
                label=f'{strategy.capitalize()}',
                linewidth=2, markersize=8, capsize=4
            )
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance aleatória')
        ax.set_xlabel('Número de Exemplos Few-Shot', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'Fase E: {metric_name}\nvs Número de Exemplos', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (metric_col, metric_name) in enumerate(metrics_config):
        _draw_metric(axes[i], metric_col, metric_name)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, mc=metrics_config[i]: _draw_metric(ax, mc[0], mc[1]), (7, 5))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_phase_e_strategy_comparison(
    results_e: List[ResultadoPhaseEExperimento],
    filename: str = None
):
    """
    Compara estratégias em cada nível de n_shot usando gráficos de barras agrupadas.
    Permite verificar qual estratégia de seleção de exemplos é mais eficaz para cada quantidade de shots.
    """
    df = pd.DataFrame([
        {
            'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
        }
        for r in results_e
    ])

    # Filtra apenas os resultados few-shot (exclui zero-shot, que é igual para todas as estratégias)
    df_fs = df[df['n_shot'] > 0]

    if len(df_fs) == 0:
        print("  Sem resultados few-shot para comparar estratégias.")
        return

    n_shots = sorted(df_fs['n_shot'].unique())

    strategy_colors = {
        'easy': '#2ecc71',
        'hard': '#e74c3c',
        'mixed': '#3498db',
        'random': '#95a5a6',
    }

    def _draw_n_shot(ax, n):
        df_n = df_fs[df_fs['n_shot'] == n]
        df_strat = df_n.groupby('strategy').agg({
            'accuracy': ['mean', 'std']
        }).reset_index()
        df_strat.columns = ['strategy', 'mean', 'std']
        bars = ax.bar(
            range(len(df_strat)), df_strat['mean'], yerr=df_strat['std'],
            color=[strategy_colors.get(s, 'gray') for s in df_strat['strategy']],
            edgecolor='black', capsize=5
        )
        ax.set_xticks(range(len(df_strat)))
        ax.set_xticklabels([s.capitalize() for s in df_strat['strategy']], fontsize=10)
        ax.set_ylabel('Concordância LLM vs. Perito')
        ax.set_title(f'{n} Exemplos Few-Shot', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        for bar, mean_val in zip(bars, df_strat['mean']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean_val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Figura combinada
    fig, axes = plt.subplots(1, len(n_shots), figsize=(5 * len(n_shots), 5), squeeze=False)
    axes = axes[0]
    for ax_idx, n in enumerate(n_shots):
        _draw_n_shot(axes[ax_idx], n)
    plt.suptitle('Fase E: Comparação de Estratégias por Número de Exemplos',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        (f"{n}shot", lambda ax, n_val=n: _draw_n_shot(ax, n_val), (6, 5))
        for n in n_shots
    ]
    _save_panels_individually(panels, filename)


def plot_phase_e_example_locations(
    X_e: np.ndarray,
    y_expert: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    n_examples: int = 10,
    random_state: int = 42,
    filename: str = None
):
    """
    Visualiza ONDE cada estratégia seleciona os exemplos.
    Exibe seleções easy/hard/mixed/random sobre a fronteira de decisão do perito.
    Fundamental para entender intuitivamente o que cada estratégia oferece ao LLM.
    """
    strategies = ["easy", "hard", "mixed", "random"]

    x_min, x_max = -6, 6
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, expert_centroids, expert_w)
    Z = Z.reshape(xx.shape)
    confidences_all, _ = compute_metric_confidence(X_e, expert_centroids, expert_w)

    def _draw_strategy(ax, strategy):
        ax.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=1.5, levels=[0.5])
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_expert, cmap="coolwarm",
                   alpha=0.2, edgecolor="gray", s=30)
        _, selected_indices = select_examples_by_strategy(
            X_e, y_expert, expert_w, expert_centroids,
            n_examples=n_examples, strategy=strategy,
            nome_classe_0="A", nome_classe_1="B",
            random_state=random_state, verbose=False
        )
        ax.scatter(X_e[selected_indices, 0], X_e[selected_indices, 1],
                   c=y_expert[selected_indices], cmap="coolwarm",
                   edgecolor="black", s=200, linewidth=2, marker='*',
                   zorder=5, label=f'Selecionados ({len(selected_indices)})')
        sel_margins = confidences_all[selected_indices]
        ax.set_title(f'Estratégia: {strategy.upper()}\n'
                     f'Margem média: {sel_margins.mean():.3f} '
                     f'[{sel_margins.min():.3f}, {sel_margins.max():.3f}]',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, strategy in zip(axes.flatten(), strategies):
        _draw_strategy(ax, strategy)
    plt.suptitle(f'Fase E: Estratégias de Seleção de Exemplos ({n_examples} exemplos)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        (s, lambda ax, strat=s: _draw_strategy(ax, strat), (7, 5))
        for s in strategies
    ]
    _save_panels_individually(panels, filename)


# =============================================================================
# GRÁFICOS DE ANÁLISE ESTENDIDA
# =============================================================================

def plot_w_distribution(resultados: List[ResultadoExperimento], filename: str = None):
    """Distribuição do vetor W estimado entre sementes e repetições (pedido do orientador)."""
    data = []
    for r in resultados:
        if r.n_shot == 0:  # W é aprendido apenas no zero-shot (Fase A)
            data.append({
                'seed': r.random_seed,
                'w0': r.w_aprendido[0],
                'w1': r.w_aprendido[1],
                'ratio': r.w_aprendido[0] / r.w_aprendido[1] if r.w_aprendido[1] != 0 else float('inf'),
                'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
                'rep': r.repeticao,
            })

    if not data:
        print("  Sem dados de W para plotar distribuição.")
        return

    df = pd.DataFrame(data)
    seeds = sorted(df['seed'].unique())
    colors_seed = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    seeds_str = [str(s) for s in seeds]
    w0_by_seed = [df[df['seed'] == s]['w0'].values for s in seeds]
    w1_by_seed = [df[df['seed'] == s]['w1'].values for s in seeds]
    x_seeds = np.arange(len(seeds))
    width = 0.35
    df_ratio = df.groupby('seed').agg({'ratio': ['mean', 'std']}).reset_index()
    df_ratio.columns = ['seed', 'mean', 'std']

    def _draw_w_scatter(ax):
        for i, seed in enumerate(seeds):
            subset = df[df['seed'] == seed]
            ax.scatter(subset['w0'], subset['w1'], c=[colors_seed[i]], s=100,
                       edgecolor='black', label=f'Semente {seed}', zorder=3)
        ax.set_xlabel('$w_1$ (peso da dimensão $x_1$)')
        ax.set_ylabel('$w_2$ (peso da dimensão $x_2$)')
        ax.set_title('Ŵ_LLM Estimado por Semente', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    def _draw_w_boxplot(ax):
        from matplotlib.patches import Patch
        bp1 = ax.boxplot(w0_by_seed, positions=x_seeds - width/2, widths=width*0.8, patch_artist=True)
        bp2 = ax.boxplot(w1_by_seed, positions=x_seeds + width/2, widths=width*0.8, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('steelblue'); patch.set_alpha(0.7)
        for patch in bp2['boxes']:
            patch.set_facecolor('coral'); patch.set_alpha(0.7)
        ax.set_xticks(x_seeds); ax.set_xticklabels(seeds_str)
        ax.set_xlabel('Semente Aleatória'); ax.set_ylabel('Valor do Peso')
        ax.set_title('Distribuição de $w_1$ e $w_2$ por Semente', fontweight='bold')
        ax.legend(handles=[Patch(facecolor='steelblue', alpha=0.7, label='$w_1$'),
                           Patch(facecolor='coral', alpha=0.7, label='$w_2$')])

    def _draw_w_ratio(ax):
        ax.bar(range(len(df_ratio)), df_ratio['mean'], yerr=df_ratio['std'],
               color='mediumpurple', edgecolor='black', capsize=5, alpha=0.8)
        ax.set_xticks(range(len(df_ratio)))
        ax.set_xticklabels([str(int(s)) for s in df_ratio['seed']])
        ax.set_xlabel('Semente Aleatória'); ax.set_ylabel('Razão $w_1/w_2$')
        ax.set_title('Razão $w_1/w_2$ por Semente', fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Euclidiana (razão=1)')
        ax.legend()

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_w_scatter(axes[0])
    _draw_w_boxplot(axes[1])
    _draw_w_ratio(axes[2])
    plt.suptitle('Distribuição do Vetor Ŵ_LLM Estimado entre Sementes e Repetições',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("scatter", _draw_w_scatter, (7, 5)),
        ("boxplot", _draw_w_boxplot, (7, 5)),
        ("ratio", _draw_w_ratio, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_metric_errors_phase_a(
    X: np.ndarray, y_llm: np.ndarray, y_metric: np.ndarray,
    w: np.ndarray, centroids: np.ndarray, filename: str = None
):
    """Visualiza onde a métrica erra vs. a rotulação do LLM na Fase A (pedido do orientador)."""
    agreements = y_llm == y_metric
    disagreements = ~agreements
    fidelity = np.mean(agreements)

    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, centroids, w)
    Z = Z.reshape(xx.shape)

    confidences, _ = compute_metric_confidence(X, centroids, w)
    bins = np.linspace(0, confidences.max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    error_rates = []
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        error_rates.append(np.mean(disagreements[mask]) if np.sum(mask) > 0 else 0)

    def _draw_error_map(ax):
        ax.contourf(xx, yy, Z, alpha=0.15, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        ax.scatter(X[agreements, 0], X[agreements, 1], c=y_llm[agreements],
                   cmap="coolwarm", alpha=0.5, edgecolor="gray", s=40, label=f'Concordam ({np.sum(agreements)})')
        ax.scatter(X[disagreements, 0], X[disagreements, 1],
                   c='yellow', edgecolor="red", s=150, linewidth=2, marker='X',
                   label=f'Discordam ({np.sum(disagreements)})', zorder=5)
        ax.scatter(*centroids[0], marker='D', s=200, c='blue', edgecolor='k', linewidth=2, zorder=6)
        ax.scatter(*centroids[1], marker='D', s=200, c='red', edgecolor='k', linewidth=2, zorder=6)
        ax.set_title(f'Fase A: Ŵ_LLM vs. Rotulação do LLM\nFidelidade: {fidelity:.1%}', fontweight='bold')
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    def _draw_error_by_margin(ax):
        colors_bar = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.1 else '#2ecc71' for r in error_rates]
        ax.bar(range(len(error_rates)), error_rates, color=colors_bar, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(error_rates)))
        ax.set_xticklabels([f'{b:.1f}' for b in bin_centers], rotation=45)
        ax.set_xlabel('Margem (distância à fronteira)')
        ax.set_ylabel('Taxa de Erro')
        ax.set_title('Taxa de Discordância por Faixa de Margem', fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _draw_error_map(axes[0])
    _draw_error_by_margin(axes[1])
    plt.suptitle('Análise de Erros da Métrica Ŵ_LLM na Fase A', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("mapa_erros", _draw_error_map, (7, 6)),
        ("erros_por_margem", _draw_error_by_margin, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_class_order_bias(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara resultados entre classes originais e invertidas para detectar viés de posição."""
    data = []
    for r in resultados:
        nome_pair = f"{r.nomes_classes[0]}/{r.nomes_classes[1]}"
        data.append({
            'pair': nome_pair,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'fidelidade': r.fidelidade_problema_a,
        })

    if not data:
        return

    df = pd.DataFrame(data)
    df_grouped = df.groupby('pair').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std'],
        'fidelidade': ['mean', 'std'],
    }).reset_index()
    df_grouped.columns = ['pair', 'b_mean', 'b_std', 'c_mean', 'c_std', 'fid_mean', 'fid_std']

    x = np.arange(len(df_grouped))
    width = 0.35

    def _draw_consistency_bias(ax):
        ax.bar(x - width/2, df_grouped['b_mean'], width, yerr=df_grouped['b_std'],
               label='Problema B', color='steelblue', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped['c_mean'], width, yerr=df_grouped['c_std'],
               label='Problema C', color='coral', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['pair'], rotation=45, ha='right')
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência por Par de Classes\n(inclui pares invertidos)', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_fidelity_bias(ax):
        ax.bar(x, df_grouped['fid_mean'], yerr=df_grouped['fid_std'],
               color='mediumpurple', capsize=3, edgecolor='black', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['pair'], rotation=45, ha='right')
        ax.set_ylabel('Fidelidade Fase A')
        ax.set_title('Fidelidade da Métrica por Par de Classes', fontweight='bold')
        ax.set_ylim(0, 1.1)

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _draw_consistency_bias(axes[0])
    _draw_fidelity_bias(axes[1])
    plt.suptitle('Análise de Viés de Ordem/Posição das Classes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_consistency_bias, (7, 6)),
        ("fidelidade", _draw_fidelity_bias, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_feature_names_effect(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara métricas e pesos aprendidos entre diferentes nomes de features."""
    # Filtra resultados que têm feature_names não-padrão ou padrão para comparação
    feat_results = [r for r in resultados if r.feature_names != ("x1", "x2")
                    or (r.n_shot == 0 and r.nomes_classes == ("A", "B") and r.prompt_variant == "default")]

    if len(feat_results) < 2:
        return

    data = []
    for r in feat_results:
        label = f"{r.feature_names[0]}/{r.feature_names[1]}"
        data.append({
            'features': label,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'fidelidade': r.fidelidade_problema_a,
            'w0': r.w_aprendido[0] if r.w_aprendido is not None else np.nan,
            'w1': r.w_aprendido[1] if r.w_aprendido is not None else np.nan,
        })

    if not data:
        return

    df = pd.DataFrame(data)
    df_grouped = df.groupby('features').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std'],
        'fidelidade': ['mean', 'std'],
        'w0': ['mean', 'std'],
        'w1': ['mean', 'std'],
    }).reset_index()
    df_grouped.columns = ['features', 'b_mean', 'b_std', 'c_mean', 'c_std',
                          'fid_mean', 'fid_std', 'w0_mean', 'w0_std', 'w1_mean', 'w1_std']

    x = np.arange(len(df_grouped))
    width = 0.35

    def _draw_feat_consistency(ax):
        ax.bar(x - width/2, df_grouped['b_mean'], width, yerr=df_grouped['b_std'],
               label='Problema B', color='steelblue', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped['c_mean'], width, yerr=df_grouped['c_std'],
               label='Problema C', color='coral', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência por Nome de Feature', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_feat_fidelity(ax):
        ax.bar(x, df_grouped['fid_mean'], yerr=df_grouped['fid_std'],
               color='mediumpurple', capsize=3, edgecolor='black', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Fidelidade Fase A')
        ax.set_title('Fidelidade da Métrica por Nome de Feature', fontweight='bold')
        ax.set_ylim(0, 1.1)

    def _draw_feat_weights(ax):
        ax.bar(x - width/2, df_grouped['w0_mean'], width, yerr=df_grouped['w0_std'],
               label='w[0]', color='forestgreen', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped['w1_mean'], width, yerr=df_grouped['w1_std'],
               label='w[1]', color='darkorange', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Peso W')
        ax.set_title('Pesos Aprendidos por Nome de Feature', fontweight='bold')
        ax.legend()

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _draw_feat_consistency(axes[0])
    _draw_feat_fidelity(axes[1])
    _draw_feat_weights(axes[2])
    plt.suptitle('Efeito dos Nomes Semânticos de Features na Métrica Estimada',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_feat_consistency, (7, 6)),
        ("fidelidade", _draw_feat_fidelity, (7, 6)),
        ("pesos", _draw_feat_weights, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_classical_baselines_comparison(
    results_e: List[ResultadoPhaseEExperimento],
    results_baselines: List[Dict],
    filename: str = None
):
    """Compara LLM vs. baselines clássicos (k-NN, LR, SVM) na Fase E.

    Mostra curvas de aprendizado do LLM (mixed strategy) junto com os baselines treinados
    nos mesmos exemplos, respondendo: o LLM faz algo que um classificador trivial não faz?
    """
    if not results_e or not results_baselines:
        return

    # Filtra LLM: apenas estratégia mixed (a mais balanceada) para comparação justa
    df_llm = pd.DataFrame([{
        'n_shot': r.n_shot,
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
        'source': 'LLM (few-shot)',
    } for r in results_e if r.example_strategy == 'mixed'])

    df_bl = pd.DataFrame(results_baselines)
    df_bl = df_bl[df_bl['example_strategy'] == 'mixed']

    if df_llm.empty or df_bl.empty:
        return

    clf_colors = {
        'LLM (few-shot)': '#3498db',
        'k-NN': '#e74c3c',
        'Logistic Regression': '#2ecc71',
        'SVM (RBF)': '#9b59b6',
    }
    clf_markers = {
        'LLM (few-shot)': 'D',
        'k-NN': 'o',
        'Logistic Regression': 's',
        'SVM (RBF)': '^',
    }
    clf_names = df_bl['model'].unique()

    metrics_config = [
        ('accuracy', 'accuracy_vs_expert', 'Concordância vs. Perito'),
        ('kappa', 'kappa_vs_expert', 'Kappa de Cohen'),
        ('f1', 'f1_vs_expert', 'F1-Score'),
    ]

    def _draw_baseline_metric(ax, llm_col, bl_col, title):
        df_llm_grouped = df_llm.groupby('n_shot').agg({
            llm_col: ['mean', 'std']
        }).reset_index()
        df_llm_grouped.columns = ['n_shot', 'mean', 'std']
        df_llm_grouped = df_llm_grouped.sort_values('n_shot')
        ax.errorbar(
            df_llm_grouped['n_shot'], df_llm_grouped['mean'],
            yerr=df_llm_grouped['std'],
            marker=clf_markers['LLM (few-shot)'],
            color=clf_colors['LLM (few-shot)'],
            label='LLM (few-shot)',
            linewidth=2.5, markersize=9, capsize=4
        )
        for clf_name in clf_names:
            df_clf = df_bl[df_bl['model'] == clf_name]
            df_clf_grouped = df_clf.groupby('n_shot').agg({
                bl_col: ['mean', 'std']
            }).reset_index()
            df_clf_grouped.columns = ['n_shot', 'mean', 'std']
            df_clf_grouped = df_clf_grouped.sort_values('n_shot')
            color_key = clf_name
            for k in clf_colors:
                if k in clf_name or clf_name.startswith(k.split(' ')[0]):
                    color_key = k
                    break
            ax.errorbar(
                df_clf_grouped['n_shot'], df_clf_grouped['mean'],
                yerr=df_clf_grouped['std'],
                marker=clf_markers.get(color_key, 'x'),
                color=clf_colors.get(color_key, 'gray'),
                label=clf_name,
                linewidth=1.5, markersize=7, capsize=3,
                linestyle='--', alpha=0.8
            )
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.4, label='Chance')
        ax.set_xlabel('Número de Exemplos', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (llm_col, bl_col, title) in enumerate(metrics_config):
        _draw_baseline_metric(axes[i], llm_col, bl_col, title)
    plt.suptitle('LLM vs. Baselines Clássicos (mesmos exemplos few-shot, estratégia mixed)\n'
                 'O LLM faz algo diferente de um classificador trivial?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, mc=metrics_config[i]: _draw_baseline_metric(ax, *mc), (7, 6))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_prompt_variant_comparison(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara métricas e W aprendidos entre diferentes variantes de prompt.

    Testa se o prompt confunde a medição de consistência do LLM.
    """
    # Filtra resultados relevantes: classes A/B, n_shot=0, com variantes
    relevant = [r for r in resultados
                if r.nomes_classes == ("A", "B") and r.n_shot == 0]
    if not relevant:
        return

    df = pd.DataFrame([{
        'variant': r.prompt_variant,
        'fidelidade': r.fidelidade_problema_a,
        'consistencia_b': r.consistencia_problema_b,
        'consistencia_c': r.consistencia_problema_c,
        'kappa_b': r.kappa_problema_b,
        'kappa_c': r.kappa_problema_c,
        'w_0': r.w_aprendido[0],
        'w_1': r.w_aprendido[1],
    } for r in relevant])

    variants = sorted(df['variant'].unique())
    if len(variants) < 2:
        return

    colors = {
        'default': '#3498db',
        'geometric': '#2ecc71',
        'cot': '#e74c3c',
        'tabular': '#9b59b6',
    }
    labels_map = {
        'default': 'Default',
        'geometric': 'Geométrico',
        'cot': 'Chain-of-Thought',
        'tabular': 'Tabular',
    }

    x = np.arange(len(variants))
    width = 0.35
    df_grouped_pv = df.groupby('variant').agg({
        'consistencia_b': ['mean', 'std'], 'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_grouped_pv.columns = ['variant', 'b_mean', 'b_std', 'c_mean', 'c_std']
    df_grouped_pv = df_grouped_pv.set_index('variant').loc[variants].reset_index()

    df_fid = df.groupby('variant').agg({'fidelidade': ['mean', 'std']}).reset_index()
    df_fid.columns = ['variant', 'fid_mean', 'fid_std']
    df_fid = df_fid.set_index('variant').loc[variants].reset_index()
    bar_colors = [colors.get(v, 'gray') for v in variants]

    def _draw_pv_consistency(ax):
        ax.bar(x - width/2, df_grouped_pv['b_mean'], width, yerr=df_grouped_pv['b_std'],
               label='Problema B', color='steelblue', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped_pv['c_mean'], width, yerr=df_grouped_pv['c_std'],
               label='Problema C', color='coral', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels([labels_map.get(v, v) for v in variants], rotation=30, ha='right')
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência por Variante de Prompt', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_pv_fidelity(ax):
        ax.bar(x, df_fid['fid_mean'], yerr=df_fid['fid_std'],
               color=bar_colors, capsize=3, edgecolor='black', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([labels_map.get(v, v) for v in variants], rotation=30, ha='right')
        ax.set_ylabel('Fidelidade Fase A')
        ax.set_title('Fidelidade da Métrica por Variante', fontweight='bold')
        ax.set_ylim(0, 1.1)

    def _draw_pv_w_scatter(ax):
        for variant in variants:
            df_v = df[df['variant'] == variant]
            ax.scatter(df_v['w_0'], df_v['w_1'], c=colors.get(variant, 'gray'),
                       label=labels_map.get(variant, variant),
                       s=80, edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.set_xlabel('w[0] (peso x1)'); ax.set_ylabel('w[1] (peso x2)')
        ax.set_title('Métrica W Estimada por Variante\n(estabilidade entre prompts)', fontweight='bold')
        ax.legend(fontsize=8); ax.set_aspect('equal', adjustable='datalim'); ax.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _draw_pv_consistency(axes[0])
    _draw_pv_fidelity(axes[1])
    _draw_pv_w_scatter(axes[2])
    plt.suptitle('Sensibilidade ao Prompt: Mesmos dados, diferentes templates\n'
                 '(Se W muda, o prompt confunde a medição)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_pv_consistency, (7, 6)),
        ("fidelidade", _draw_pv_fidelity, (7, 6)),
        ("w_scatter", _draw_pv_w_scatter, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_example_order_bias(results_order: List[ResultadoPhaseEExperimento], filename: str = None):
    """Compara métricas entre diferentes ordenações dos exemplos few-shot.

    Detecta viés de recência: se a ordem dos exemplos afeta a performance do LLM.
    """
    if not results_order:
        return

    df = pd.DataFrame([{
        'n_shot': r.n_shot,
        'ordering': r.example_strategy.replace("mixed_order_", ""),
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
    } for r in results_order])

    n_shots = sorted(df['n_shot'].unique())
    orderings = df['ordering'].unique()

    order_colors = {
        'class0_first': '#3498db',
        'class1_first': '#e74c3c',
        'shuffled': '#2ecc71',
        'alternating': '#9b59b6',
    }
    order_labels = {
        'class0_first': 'Classe 0 primeiro',
        'class1_first': 'Classe 1 primeiro',
        'shuffled': 'Aleatório',
        'alternating': 'Alternado',
    }

    metrics_order = [
        ('accuracy', 'Concordância LLM vs. Perito'),
        ('kappa', 'Kappa de Cohen'),
        ('f1', 'F1-Score'),
    ]

    def _draw_order_metric(ax, metric, metric_name, show_legend=False):
        x = np.arange(len(n_shots))
        width = 0.18
        n_orderings = len(orderings)
        for i, ordering in enumerate(orderings):
            df_ord = df[df['ordering'] == ordering]
            means, stds = [], []
            for ns in n_shots:
                vals = df_ord[df_ord['n_shot'] == ns][metric]
                means.append(vals.mean()); stds.append(vals.std())
            offset = (i - n_orderings / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=order_labels.get(ordering, ordering),
                   color=order_colors.get(ordering, f'C{i}'),
                   capsize=3, edgecolor='black', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([f'{ns}-shot' for ns in n_shots])
        ax.set_ylabel(metric_name); ax.set_title(metric_name, fontweight='bold')
        if metric in ('accuracy', 'f1'):
            ax.set_ylim(0, 1.1)
        elif metric == 'kappa':
            ax.set_ylim(-0.1, 1.1)
        if show_legend:
            ax.legend(fontsize=8)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, (metric, metric_name) in enumerate(metrics_order):
        _draw_order_metric(axes[ax_idx], metric, metric_name, show_legend=(ax_idx == 0))
    plt.suptitle('Viés de Ordem dos Exemplos Few-Shot (Recency Bias)\n'
                 'Mesmos exemplos (mixed), diferentes ordenações',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, m=metrics_order[i]: _draw_order_metric(ax, m[0], m[1], show_legend=True), (7, 6))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_dilution_experiment(results_dilution: List[ResultadoPhaseEExperimento], filename: str = None):
    """Gráfico do experimento de diluição: 3 hard fixos + N easy progressivos."""
    if not results_dilution:
        return

    df = pd.DataFrame([{
        'n_shot': r.n_shot,
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'strategy': r.example_strategy,
    } for r in results_dilution])

    df_acc = df.groupby('n_shot').agg({'accuracy': ['mean', 'std']}).reset_index()
    df_acc.columns = ['n_shot', 'mean', 'std']
    df_acc = df_acc.sort_values('n_shot')
    ref_3hard = df[df['n_shot'] == 3]

    df_kappa = df.groupby('n_shot').agg({'kappa': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['n_shot', 'mean', 'std']
    df_kappa = df_kappa.sort_values('n_shot')

    def _draw_dilution_accuracy(ax):
        ax.errorbar(df_acc['n_shot'], df_acc['mean'], yerr=df_acc['std'],
                    marker='o', linewidth=2, markersize=8, capsize=4, color='#e74c3c')
        if len(ref_3hard) > 0:
            ax.axhline(y=ref_3hard['accuracy'].mean(), color='gray', linestyle='--',
                       alpha=0.7, label=f'3 hard puros ({ref_3hard["accuracy"].mean():.1%})')
        ax.set_xlabel('Total de Exemplos (3 hard + N easy)')
        ax.set_ylabel('Concordância LLM vs. Perito')
        ax.set_title('Experimento de Diluição: Acurácia', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    def _draw_dilution_kappa(ax):
        ax.errorbar(df_kappa['n_shot'], df_kappa['mean'], yerr=df_kappa['std'],
                    marker='s', linewidth=2, markersize=8, capsize=4, color='#3498db')
        ax.set_xlabel('Total de Exemplos (3 hard + N easy)')
        ax.set_ylabel('Kappa de Cohen')
        ax.set_title('Experimento de Diluição: Kappa', fontweight='bold')
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_dilution_accuracy(axes[0])
    _draw_dilution_kappa(axes[1])
    plt.suptitle('Experimento de Diluição: Hard Fixos + Easy Progressivos', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("accuracy", _draw_dilution_accuracy, (7, 5)),
        ("kappa", _draw_dilution_kappa, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_r3_comparison(results_r2: List, results_r3: List, filename: str = None):
    """Compara fidelidade com 2 pesos vs 3 pesos (projeção R3) usando as mesmas
    classificações do LLM (que viu apenas x1, x2). Se a fidelidade R3 > R2,
    há evidência de não-linearidade implícita no processo decisório do LLM."""
    if not results_r2 or not results_r3:
        return

    r2_means = np.mean([r['accuracy'] for r in results_r2])
    r2_stds = np.std([r['accuracy'] for r in results_r2])
    r3_means = np.mean([r['accuracy'] for r in results_r3])
    r3_stds = np.std([r['accuracy'] for r in results_r3])
    has_nnls = any('accuracy_nnls' in r for r in results_r3)

    algo_data = [('Perceptron', [r['accuracy'] for r in results_r3], 'steelblue')]
    if has_nnls:
        algo_data.append(('NNLS', [r.get('accuracy_nnls', np.nan) for r in results_r3], 'coral'))

    def _draw_r3_comparison(ax):
        labels_r3 = ['2 pesos\n($w_1$, $w_2$)', '3 pesos\n($w_1$, $w_2$, $w_3$)\n$x_3 = x_1 \\cdot x_2$']
        means = [r2_means, r3_means]
        stds = [r2_stds, r3_stds]
        bars = ax.bar(labels_r3, means, yerr=stds, color=['steelblue', 'coral'],
                      edgecolor='black', capsize=5, alpha=0.8)
        ax.set_ylabel('Fidelidade (métrica vs. LLM)')
        ax.set_title('Perceptron: Métrica Linear vs. Quadrática', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        delta = means[1] - means[0]
        ax.text(0.5, 0.05, f'$\\Delta$ = {delta:+.1%}', ha='center', va='bottom',
                transform=ax.transAxes, fontsize=10, style='italic',
                color='green' if delta > 0 else 'gray')

    def _draw_r3_algo(ax):
        x_pos = np.arange(len(algo_data))
        for i, (name, accs, color) in enumerate(algo_data):
            accs_clean = [a for a in accs if not np.isnan(a)]
            m = np.mean(accs_clean) if accs_clean else 0
            s = np.std(accs_clean) if len(accs_clean) > 1 else 0
            ax.bar(i, m, yerr=s, color=color, edgecolor='black', capsize=5, alpha=0.8)
            ax.text(i, m + 0.02, f'{m:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos); ax.set_xticklabels([d[0] for d in algo_data])
        ax.set_ylabel('Fidelidade R3 (3 pesos)')
        ax.set_title('R3: Fidelidade por Algoritmo', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_r3_comparison(axes[0])
    _draw_r3_algo(axes[1])
    plt.suptitle('Não-linearidade Implícita: Fidelidade com 2 vs 3 pesos\n'
                 '(LLM viu apenas $x_1$, $x_2$ — $x_3 = x_1 \\cdot x_2$ adicionado nos bastidores)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("linear_vs_quadratica", _draw_r3_comparison, (7, 5)),
        ("algoritmos_r3", _draw_r3_algo, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_algorithm_comparison(results_perceptron: List, results_alternative: List,
                              filename: str = None):
    """Compara W aprendido por Perceptron vs NNLS em todas as fases (A, B, C)."""
    if not results_perceptron or not results_alternative:
        return

    bar_width = 0.25
    w_perc = np.array([[r.w_aprendido[0], r.w_aprendido[1]] for r in results_perceptron])
    w_alt = np.array([[r.w_aprendido[0], r.w_aprendido[1]] for r in results_alternative])
    fid_perc = [r.fidelidade_problema_a for r in results_perceptron]
    fid_alt = [r.fidelidade_problema_a for r in results_alternative]

    algo_list = [
        ('Perceptron', results_perceptron, 'steelblue'),
        ('NNLS', results_alternative, 'coral'),
    ]

    def _draw_algo_w_scatter(ax):
        ax.scatter(w_perc[:, 0], w_perc[:, 1], c='steelblue', s=100, edgecolor='k', label='Perceptron', zorder=3)
        ax.scatter(w_alt[:, 0], w_alt[:, 1], c='coral', s=100, edgecolor='k', label='NNLS', zorder=3)
        ax.set_xlabel('$w_1$'); ax.set_ylabel('$w_2$')
        ax.set_title('Ŵ_LLM: Perceptron vs. NNLS', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _draw_algo_fidelity(ax):
        box_data = [fid_perc, fid_alt]
        box_labels = ['Perceptron', 'NNLS']
        box_colors = ['steelblue', 'coral']
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for i, color in enumerate(box_colors):
            bp['boxes'][i].set_facecolor(color); bp['boxes'][i].set_alpha(0.7)
        ax.set_ylabel('Fidelidade')
        ax.set_title('Fase A: Fidelidade (Métrica vs. LLM)', fontweight='bold')
        ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')

    def _draw_algo_phase(ax, phase_letter, cons_attr, kappa_attr):
        x_pos = np.array([0, 1])
        n_algos_local = len(algo_list)
        for j, (name, res, color) in enumerate(algo_list):
            cons_vals = [getattr(r, cons_attr) for r in res]
            kappa_vals = [getattr(r, kappa_attr) for r in res]
            means = [np.mean(cons_vals), np.mean(kappa_vals)]
            stds = [np.std(cons_vals) if len(cons_vals) > 1 else 0,
                    np.std(kappa_vals) if len(kappa_vals) > 1 else 0]
            offset = (j - (n_algos_local - 1) / 2) * bar_width
            ax.bar(x_pos + offset, means, bar_width, yerr=stds,
                   color=color, alpha=0.7, edgecolor='k', capsize=4, label=name)
        ax.set_xticks(x_pos); ax.set_xticklabels(['Consistência', 'Kappa'])
        ax.set_title(f'Fase {phase_letter}: Consistência no Problema {phase_letter}', fontweight='bold')
        ax.set_ylim(-0.1, 1.1); ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=8)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_algo_w_scatter(axes[0, 0])
    _draw_algo_fidelity(axes[0, 1])
    _draw_algo_phase(axes[1, 0], 'B', 'consistencia_problema_b', 'kappa_problema_b')
    _draw_algo_phase(axes[1, 1], 'C', 'consistencia_problema_c', 'kappa_problema_c')
    plt.suptitle('Comparação de Algoritmos de Otimização Inversa\n(Fases A, B e C)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("w_scatter", _draw_algo_w_scatter, (7, 6)),
        ("fidelidade", _draw_algo_fidelity, (7, 6)),
        ("fase_b", lambda ax: _draw_algo_phase(ax, 'B', 'consistencia_problema_b', 'kappa_problema_b'), (7, 6)),
        ("fase_c", lambda ax: _draw_algo_phase(ax, 'C', 'consistencia_problema_c', 'kappa_problema_c'), (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_gamma_convergence(diagnostics: List[dict], filename: str = None):
    """Diagnóstico da busca binária em γ no Perceptron Estruturado.

    Item b da reunião 30/04/2026 (~520s): orientador pediu para verificar se
    γ converge crescentemente entre execuções (não fica estagnado). Cada entrada
    de `diagnostics` é {"label": str, "gamma_final": float, "history": [dicts]}.
    Cada dict do history: {iter, gamma, gamma_lo, gamma_hi, violations, viable}.

    Painel 1: evolução de γ por iteração (curva por execução).
    Painel 2: γ_lo (limite inferior viável) por iteração — deve ser monotônico.
    """
    if not diagnostics:
        print("  ⚠ Sem histórico de γ para plotar (nenhuma chamada com return_history=True).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    for idx, entry in enumerate(diagnostics):
        hist = entry.get("history") or []
        if not hist:
            continue
        iters = [h["iter"] for h in hist]
        gammas = [h["gamma"] for h in hist]
        gamma_los = [h["gamma_lo"] for h in hist]
        viable = [h["viable"] for h in hist]

        color = cmap(idx % 10)
        label = f"{entry.get('label', f'exec_{idx}')} (γ*={entry.get('gamma_final', float('nan')):.3f})"

        # Painel 1: γ testado por iteração — marcadores diferentes para viable/inviable
        axes[0].plot(iters, gammas, "-", color=color, alpha=0.5, linewidth=1.0)
        v_iters = [i for i, v in zip(iters, viable) if v]
        v_gammas = [g for g, v in zip(gammas, viable) if v]
        nv_iters = [i for i, v in zip(iters, viable) if not v]
        nv_gammas = [g for g, v in zip(gammas, viable) if not v]
        axes[0].scatter(v_iters, v_gammas, s=40, marker="o", color=color, label=label,
                        edgecolor="black", linewidth=0.4)
        axes[0].scatter(nv_iters, nv_gammas, s=40, marker="x", color=color)

        # Painel 2: γ_lo (melhor margem viável até o momento) — DEVE ser monotônico crescente
        axes[1].plot(iters, gamma_los, "-o", color=color, alpha=0.8, markersize=4,
                     label=label)

    axes[0].set_xlabel("Iteração da busca binária")
    axes[0].set_ylabel("γ testado")
    axes[0].set_title("γ candidato por iteração\n(○ = viável, × = inviável)", fontsize=10, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")

    axes[1].set_xlabel("Iteração da busca binária")
    axes[1].set_ylabel("γ_lo (melhor margem viável)")
    axes[1].set_title("γ_lo monotônico (orientador, item b ~520s)", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")

    plt.suptitle("Convergência da busca binária em γ — Perceptron Estruturado (CILAMCE 2017, Eq. 31)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Imagem salva: {os.path.basename(filename)}")
    plt.close(fig)

    # Verificação textual: γ_lo deve ser monotônico não-decrescente em cada execução
    print("\n  Diagnóstico da busca binária em γ:")
    for entry in diagnostics:
        hist = entry.get("history") or []
        if not hist:
            continue
        gamma_los = [h["gamma_lo"] for h in hist]
        monotone = all(gamma_los[i] <= gamma_los[i + 1] for i in range(len(gamma_los) - 1))
        status = "✓ monotônico" if monotone else "⚠ NÃO monotônico"
        print(f"    {entry.get('label', '?'):40s}  iters={len(hist):3d}  γ*={entry.get('gamma_final', float('nan')):.4f}  {status}")


def plot_oracle_w_recovery(oracle_results: List[dict], filename: str = None):
    """Visualiza a recuperação de W conhecido pelos algoritmos (validação do oráculo)."""
    if not oracle_results:
        return

    from matplotlib.lines import Line2D

    colors_algo = {'perceptron': 'steelblue', 'nnls': 'coral'}
    labels = list(dict.fromkeys(f"{r['problem']}\n{r['expert_name']}" for r in oracle_results))
    label_results = {}
    for r in oracle_results:
        key = f"{r['problem']}\n{r['expert_name']}"
        algo = r['algorithm']
        label_results.setdefault(key, {})[algo] = r
    n_labels = len(labels)
    x_base = np.arange(n_labels)

    def _draw_w_scatter(ax):
        for r in oracle_results:
            cl = colors_algo[r['algorithm']]
            true_norm = np.array([r['true_w_0'], r['true_w_1']])
            true_norm = true_norm / np.sum(true_norm)
            ax.scatter(true_norm[0], r['recovered_w_norm_0'], c=cl, s=80,
                       edgecolor='k', linewidth=0.5, zorder=3, alpha=0.7)
            ax.scatter(true_norm[1], r['recovered_w_norm_1'], c=cl, s=80,
                       edgecolor='k', linewidth=0.5, zorder=3, alpha=0.7)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Recuperação perfeita')
        ax.set_xlabel('$w$ verdadeiro (normalizado)')
        ax.set_ylabel('$w$ recuperado (normalizado)')
        ax.set_title('W Normalizado: Verdadeiro vs Recuperado', fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='PERCEPTRON'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', markersize=8, label='NNLS'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    def _draw_ratio(ax):
        bar_width = 0.2
        max_ratio_for_plot = 10.0
        for i, label in enumerate(labels):
            algos = label_results[label]
            first = list(algos.values())[0]
            true_ratio = min(first['true_w_ratio'], max_ratio_for_plot)
            ax.bar(x_base[i] - bar_width, true_ratio, bar_width, color='gray',
                   edgecolor='k', alpha=0.7, label='Verdadeiro' if i == 0 else '')
            for j, algo in enumerate(['perceptron', 'nnls']):
                if algo in algos:
                    ratio = algos[algo]['recovered_w_ratio']
                    if ratio != float('inf'):
                        ratio_plot = min(ratio, max_ratio_for_plot)
                        ax.bar(x_base[i] + bar_width * j, ratio_plot, bar_width,
                               color=colors_algo[algo], edgecolor='k', alpha=0.7,
                               label=algo.upper() if i == 0 else '')
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Razão $w_1 / w_2$')
        ax.set_title('Razão $w_1/w_2$: Verdadeira vs Recuperada', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_cosine(ax):
        bar_width = 0.3
        for i, label in enumerate(labels):
            algos = label_results[label]
            for j, algo in enumerate(['perceptron', 'nnls']):
                if algo in algos:
                    ax.bar(x_base[i] + bar_width * (j - 0.5), algos[algo]['cosine_similarity'],
                           bar_width, color=colors_algo[algo], edgecolor='k', alpha=0.7,
                           label=algo.upper() if i == 0 else '')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfeito (1.0)')
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Similaridade de Cosseno')
        ax.set_title('Cosseno entre W Verdadeiro e Recuperado', fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_fidelity(ax):
        bar_width = 0.3
        for i, label in enumerate(labels):
            algos = label_results[label]
            for j, algo in enumerate(['perceptron', 'nnls']):
                if algo in algos:
                    ax.bar(x_base[i] + bar_width * (j - 0.5), algos[algo]['fidelity'] * 100,
                           bar_width, color=colors_algo[algo], edgecolor='k', alpha=0.7,
                           label=algo.upper() if i == 0 else '')
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Meta 95%')
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Fidelidade (%)')
        ax.set_title('Fidelidade: Rótulos Verdadeiros vs Métrica Recuperada', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_w_scatter(axes[0, 0])
    _draw_ratio(axes[0, 1])
    _draw_cosine(axes[1, 0])
    _draw_fidelity(axes[1, 1])
    plt.suptitle('Validação do Oráculo: Recuperação de W Conhecido\n(Centroides verdadeiros, sem LLM)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("w_scatter", _draw_w_scatter, (7, 7)),
        ("ratio", _draw_ratio, (8, 6)),
        ("cosseno", _draw_cosine, (8, 6)),
        ("fidelidade", _draw_fidelity, (8, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_oracle_transfer(oracle_results: List[dict], filename: str = None):
    """Visualiza a fidelidade por problema e algoritmo (todos os problemas lado a lado)."""
    if not oracle_results:
        return

    colors_algo = {'perceptron': 'steelblue', 'nnls': 'coral'}
    expert_names = list(dict.fromkeys(r['expert_name'] for r in oracle_results))

    def _draw_expert(ax, ename):
        subset = [r for r in oracle_results if r['expert_name'] == ename]
        probs_for_expert = list(dict.fromkeys(r['problem'] for r in subset))
        true_w_str = f"[{subset[0]['true_w_0']}, {subset[0]['true_w_1']}]"
        bar_width = 0.25
        x_pos = np.arange(len(probs_for_expert))
        for j, algo in enumerate(['perceptron', 'nnls']):
            values = []
            for prob in probs_for_expert:
                r_match = [r for r in subset if r['algorithm'] == algo and r['problem'] == prob]
                values.append(r_match[0]['fidelity'] * 100 if r_match else 0)
            bars = ax.bar(x_pos + bar_width * (j - 1), values, bar_width,
                          color=colors_algo[algo], edgecolor='k', alpha=0.7,
                          label=algo.upper())
            for k, v in enumerate(values):
                ax.text(x_pos[k] + bar_width * (j - 1), v + 1, f'{v:.1f}%',
                        ha='center', va='bottom', fontsize=7)
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90%')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('Problema ', '') for p in probs_for_expert], fontsize=9)
        ax.set_xlabel('Problema')
        ax.set_ylabel('Fidelidade (%)')
        ax.set_title(f'{ename}\nW = {true_w_str}', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, len(expert_names), figsize=(5 * len(expert_names), 5), squeeze=False)
    for idx, ename in enumerate(expert_names):
        _draw_expert(axes[0, idx], ename)
    plt.suptitle('Validação do Oráculo: Fidelidade por Problema\n(Centroides verdadeiros, sem LLM)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por expert
    if filename:
        base, ext = os.path.splitext(filename)
        for ename in expert_names:
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(6, 5))
            _draw_expert(ax_ind, ename)
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_{ename}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_dataset_overview(data: dict, seed: int, filename: str = None):
    """Visão completa dos 4 datasets: ground truth vs classificação LLM."""
    colors_gt = {0: '#3498db', 1: '#e74c3c'}
    colors_llm = {0: '#2980b9', 1: '#c0392b'}

    problems = [
        ('A', data['X_a'], data['y_gt_a'], data.get('y_llm_a')),
        ('B', data['X_b'], data['y_gt_b'], data.get('y_llm_b')),
        ('C', data['X_c'], data['y_gt_c'], data.get('y_llm_c')),
        ('D', data['X_e'], data['y_gt_e'], None),
    ]

    def _draw_problem(axes_pair, name, X, y_gt, y_llm):
        ax_gt, ax_llm = axes_pair
        for c in [0, 1]:
            mask = y_gt == c
            ax_gt.scatter(X[mask, 0], X[mask, 1], c=colors_gt[c], s=15, alpha=0.6, label=f'Classe {c}')
        centroid_0 = X[y_gt == 0].mean(axis=0)
        centroid_1 = X[y_gt == 1].mean(axis=0)
        ax_gt.scatter(*centroid_0, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
        ax_gt.scatter(*centroid_1, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
        ax_gt.set_title(f'Problema {name} — Ground Truth (n={len(X)})', fontweight='bold', fontsize=10)
        ax_gt.legend(fontsize=8)
        ax_gt.grid(True, alpha=0.3)

        if y_llm is not None:
            for c in [0, 1]:
                mask = y_llm == c
                ax_llm.scatter(X[mask, 0], X[mask, 1], c=colors_llm[c], s=15, alpha=0.6, label=f'Classe {c}')
            c0_llm = X[y_llm == 0].mean(axis=0) if np.any(y_llm == 0) else centroid_0
            c1_llm = X[y_llm == 1].mean(axis=0) if np.any(y_llm == 1) else centroid_1
            ax_llm.scatter(*c0_llm, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
            ax_llm.scatter(*c1_llm, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
            n0, n1 = np.sum(y_llm == 0), np.sum(y_llm == 1)
            ax_llm.set_title(f'Problema {name} — LLM (C0={n0}, C1={n1})', fontweight='bold', fontsize=10)
            ax_llm.legend(fontsize=8)
        else:
            ax_llm.text(0.5, 0.5, 'Sem dados LLM', ha='center', va='center', transform=ax_llm.transAxes, fontsize=12, color='gray')
            ax_llm.set_title(f'Problema {name} — LLM', fontweight='bold', fontsize=10)
        ax_llm.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for col, (name, X, y_gt, y_llm) in enumerate(problems):
        _draw_problem((axes[0, col], axes[1, col]), name, X, y_gt, y_llm)
    fig.suptitle(f'Visão Geral dos Datasets — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por problema (GT + LLM)
    if filename:
        base, ext = os.path.splitext(filename)
        for name, X, y_gt, y_llm in problems:
            fig_ind, axes_ind = plt.subplots(2, 1, figsize=(6, 10))
            _draw_problem((axes_ind[0], axes_ind[1]), name, X, y_gt, y_llm)
            fig_ind.suptitle(f'Problema {name} — Seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_problema_{name.lower()}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_hits_and_errors(data: dict, seed: int, filename: str = None):
    """Acertos e erros da métrica vs LLM para Problemas A, B, C."""
    problems = [
        ('A', data['X_a'], data.get('y_llm_a'), data.get('y_metric_a')),
        ('B', data['X_b'], data.get('y_llm_b'), data.get('y_metric_b')),
        ('C', data['X_c'], data.get('y_llm_c'), data.get('y_metric_c')),
    ]
    learned_metric = data.get('learned_metric')

    def _draw_problem_row(axes_row, name, X, y_llm, y_metric):
        if y_llm is None or y_metric is None:
            for col in range(3):
                axes_row[col].text(0.5, 0.5, 'Sem dados', ha='center', va='center',
                                   transform=axes_row[col].transAxes, fontsize=12, color='gray')
                axes_row[col].set_title(f'Problema {name}')
            return

        n_min = min(len(y_llm), len(y_metric), len(X))
        X_plot = X[:n_min]
        y_llm_plot = y_llm[:n_min]
        y_metric_plot = y_metric[:n_min]
        hits = y_llm_plot == y_metric_plot
        errors = ~hits

        ax1 = axes_row[0]
        ax1.scatter(X_plot[hits, 0], X_plot[hits, 1], c='#27ae60', s=15, alpha=0.5, label=f'Acerto ({hits.sum()})')
        ax1.scatter(X_plot[errors, 0], X_plot[errors, 1], c='#e74c3c', s=30, alpha=0.8, marker='x', label=f'Erro ({errors.sum()})')
        acc = hits.sum() / len(hits) * 100
        ax1.set_title(f'Problema {name}: Acertos/Erros ({acc:.1f}%)', fontweight='bold', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes_row[1]
        if learned_metric is not None:
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = predict_with_metric(grid_points, learned_metric.centroids, learned_metric.w)
            Z = Z.reshape(xx.shape)
            ax2.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu')
            ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        for c in [0, 1]:
            mask = y_llm_plot == c
            ax2.scatter(X_plot[mask, 0], X_plot[mask, 1], s=15, alpha=0.5, label=f'LLM Classe {c}')
        ax2.set_title(f'Problema {name}: Fronteira W_A', fontweight='bold', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = axes_row[2]
        if learned_metric is not None:
            confidences, _ = compute_metric_confidence(X_plot, learned_metric.centroids, learned_metric.w)
            sc = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=confidences, cmap='viridis', s=15, alpha=0.6)
            ax3.scatter(X_plot[errors, 0], X_plot[errors, 1], c='red', s=50, marker='x', linewidths=2, label=f'Erros ({errors.sum()})')
            plt.colorbar(sc, ax=ax3, label='Margem')
        ax3.set_title(f'Problema {name}: Confiança + Erros', fontweight='bold', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    for row, (name, X, y_llm, y_metric) in enumerate(problems):
        _draw_problem_row(axes[row], name, X, y_llm, y_metric)
    fig.suptitle(f'Acertos e Erros: Métrica vs LLM — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por problema (1×3: acertos, fronteira, confiança)
    if filename:
        base, ext = os.path.splitext(filename)
        for name, X, y_llm, y_metric in problems:
            fig_ind, axes_ind = plt.subplots(1, 3, figsize=(18, 5))
            _draw_problem_row(axes_ind, name, X, y_llm, y_metric)
            fig_ind.suptitle(f'Acertos e Erros: Problema {name} — Seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_problema_{name.lower()}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_w_comparison_algorithms(data: dict, seed: int, filename: str = None):
    """Comparação visual dos W aprendidos: Perceptron vs NNLS."""
    learned_metric = data.get('learned_metric')
    w_nnls = data.get('w_nnls')
    X = data['X_a']
    y_llm = data.get('y_llm_a')

    if learned_metric is None or y_llm is None:
        return

    w_perc = learned_metric.w
    centroids_perc = learned_metric.centroids
    centroids_nnls = data.get('centroids_nnls', centroids_perc)

    has_nnls = w_nnls is not None

    # Helper para plotar fronteira
    def plot_boundary(ax, X, y, w, centroids, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_with_metric(grid_points, centroids, w)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.15, cmap='RdBu')
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        for c in [0, 1]:
            mask = y == c
            ax.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.5, label=f'Classe {c}')
        ax.scatter(*centroids[0], c='black', marker='X', s=150, zorder=5, edgecolors='white', linewidths=1.5)
        ax.scatter(*centroids[1], c='black', marker='X', s=150, zorder=5, edgecolors='white', linewidths=1.5)
        w_norm = w / np.sum(w) if np.sum(w) > 0 else w
        ax.set_title(f'{title}\nW=[{w[0]:.3f}, {w[1]:.3f}] norm=[{w_norm[0]:.2f}, {w_norm[1]:.2f}]',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_bar_chart(ax_bar):
        x_pos = np.arange(2)
        algos = [('Perceptron', w_perc, 'steelblue')]
        if has_nnls:
            algos.append(('NNLS', w_nnls, 'coral'))
        n_algos = len(algos)
        width = 0.8 / n_algos
        for j, (name, w, color) in enumerate(algos):
            offset = (j - (n_algos - 1) / 2) * width
            bars = ax_bar.bar(x_pos + offset, w, width, label=name, color=color, alpha=0.8, edgecolor='k')
            for bar in bars:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(['w1 (x1)', 'w2 (x2)'])
        ax_bar.set_ylabel('Peso')
        ratio_strs = []
        for name, w, _ in algos:
            ratio = w[0] / w[1] if w[1] > 0 else float('inf')
            ratio_strs.append(f'{name}={ratio:.2f}')
        ax_bar.set_title(f'Comparação de Pesos\nRatio w1/w2: {", ".join(ratio_strs)}',
                         fontweight='bold', fontsize=10)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')

    # Figura combinada: boundary plots + bar chart
    idx = 0
    n_boundary = 1 + int(has_nnls)
    n_cols = n_boundary + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    plot_boundary(axes[idx], X, y_llm, w_perc, centroids_perc, 'Perceptron Estruturado')
    idx += 1
    if has_nnls:
        plot_boundary(axes[idx], X, y_llm, w_nnls, centroids_nnls, 'NNLS (Mín. Quadrados)')
        idx += 1
    _draw_bar_chart(axes[idx])

    fig.suptitle(f'Comparação de Algoritmos de Otimização Inversa — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    if filename:
        base, ext = os.path.splitext(filename)
        # Fronteira Perceptron
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
        plot_boundary(ax_ind, X, y_llm, w_perc, centroids_perc, 'Perceptron Estruturado')
        fig_ind.tight_layout()
        fig_ind.savefig(f"{base}_perceptron{ext}", dpi=150, bbox_inches='tight')
        plt.close(fig_ind)
        if has_nnls:
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
            plot_boundary(ax_ind, X, y_llm, w_nnls, centroids_nnls, 'NNLS (Mín. Quadrados)')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_nnls{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)
        # Bar chart
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
        _draw_bar_chart(ax_ind)
        fig_ind.tight_layout()
        fig_ind.savefig(f"{base}_barras{ext}", dpi=150, bbox_inches='tight')
        plt.close(fig_ind)


def plot_confusion_matrices_detailed(data: dict, seed: int, filename: str = None):
    """Matrizes de confusão: LLM vs Métrica e LLM vs Ground Truth para cada problema."""
    problems = [
        ('A', data.get('y_llm_a'), data.get('y_metric_a'), data['y_gt_a']),
        ('B', data.get('y_llm_b'), data.get('y_metric_b'), data['y_gt_b']),
        ('C', data.get('y_llm_c'), data.get('y_metric_c'), data['y_gt_c']),
    ]

    def _draw_problem_col(axes_col, name, y_llm, y_metric, y_gt):
        ax1, ax2 = axes_col
        if y_llm is not None and y_metric is not None:
            n_min = min(len(y_llm), len(y_metric))
            cm = np.zeros((2, 2), dtype=int)
            for i in range(n_min):
                cm[int(y_metric[i]), int(y_llm[i])] += 1
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
            ax1.set_xlabel('LLM')
            ax1.set_ylabel('Métrica')
            acc = np.trace(cm) / cm.sum() * 100
            ax1.set_title(f'Problema {name}: LLM vs Métrica\n({acc:.1f}% concordância)', fontweight='bold', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax1.transAxes, color='gray')
            ax1.set_title(f'Problema {name}: LLM vs Métrica')

        if y_llm is not None:
            n_min = min(len(y_llm), len(y_gt))
            cm2 = np.zeros((2, 2), dtype=int)
            for i in range(n_min):
                cm2[int(y_gt[i]), int(y_llm[i])] += 1
            sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                        xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
            ax2.set_xlabel('LLM')
            ax2.set_ylabel('Ground Truth')
            acc2 = np.trace(cm2) / cm2.sum() * 100
            ax2.set_title(f'Problema {name}: LLM vs GT\n({acc2:.1f}% acurácia)', fontweight='bold', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax2.transAxes, color='gray')
            ax2.set_title(f'Problema {name}: LLM vs GT')

    # Figura combinada
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, (name, y_llm, y_metric, y_gt) in enumerate(problems):
        _draw_problem_col((axes[0, col], axes[1, col]), name, y_llm, y_metric, y_gt)
    fig.suptitle(f'Matrizes de Confusão — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por problema (2×1: LLM vs Métrica + LLM vs GT)
    if filename:
        base, ext = os.path.splitext(filename)
        for name, y_llm, y_metric, y_gt in problems:
            fig_ind, axes_ind = plt.subplots(2, 1, figsize=(6, 10))
            _draw_problem_col((axes_ind[0], axes_ind[1]), name, y_llm, y_metric, y_gt)
            fig_ind.suptitle(f'Matrizes de Confusão: Problema {name} — Seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_problema_{name.lower()}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_margin_analysis_detailed(data: dict, seed: int, filename: str = None):
    """Análise detalhada de margens/confiança da métrica estimada."""
    learned_metric = data.get('learned_metric')
    if learned_metric is None:
        return

    X_a = data['X_a']
    y_llm_a = data.get('y_llm_a')
    y_metric_a = data.get('y_metric_a')
    conf_a, _ = compute_metric_confidence(X_a, learned_metric.centroids, learned_metric.w)

    def _draw_histograma(ax):
        ax.hist(conf_a, bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.median(conf_a), color='red', linestyle='--', label=f'Mediana: {np.median(conf_a):.2f}')
        ax.axvline(np.mean(conf_a), color='orange', linestyle='--', label=f'Média: {np.mean(conf_a):.2f}')
        ax.set_xlabel('Margem')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Margens — Problema A', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_taxa_erro(ax):
        if y_llm_a is not None and y_metric_a is not None:
            n_min = min(len(y_llm_a), len(y_metric_a), len(conf_a))
            errors = y_llm_a[:n_min] != y_metric_a[:n_min]
            conf_plot = conf_a[:n_min]
            bins = np.linspace(0, conf_plot.max(), 8)
            bin_indices = np.digitize(conf_plot, bins)
            error_rates = []
            bin_centers = []
            bin_counts = []
            for b in range(1, len(bins)):
                mask = bin_indices == b
                if mask.sum() > 0:
                    error_rates.append(errors[mask].mean() * 100)
                    bin_centers.append((bins[b-1] + bins[b]) / 2)
                    bin_counts.append(mask.sum())
            bars = ax.bar(bin_centers, error_rates, width=(bins[1]-bins[0])*0.8, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
            for bar, count in zip(bars, bin_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Margem')
        ax.set_ylabel('Taxa de Erro (%)')
        ax.set_title('Taxa de Erro por Faixa de Margem', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _draw_mapa_confianca(ax):
        sc = ax.scatter(X_a[:, 0], X_a[:, 1], c=conf_a, cmap='viridis', s=20, alpha=0.7)
        ax.scatter(*learned_metric.centroids[0], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2)
        ax.scatter(*learned_metric.centroids[1], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2)
        plt.colorbar(sc, ax=ax, label='Margem')
        ax.set_title('Mapa de Confiança — Problema A', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _draw_violin(ax):
        all_margins = [conf_a]
        labels = ['A']
        for pname, X_p in [('B', data['X_b']), ('C', data['X_c'])]:
            conf_p, _ = compute_metric_confidence(X_p, learned_metric.centroids, learned_metric.w)
            all_margins.append(conf_p)
            labels.append(pname)
        parts = ax.violinplot(all_margins, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f'Problema {l}' for l in labels])
        ax.set_ylabel('Margem')
        ax.set_title('Distribuição de Margens por Problema', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _draw_histograma(axes[0, 0])
    _draw_taxa_erro(axes[0, 1])
    _draw_mapa_confianca(axes[1, 0])
    _draw_violin(axes[1, 1])
    fig.suptitle(f'Análise de Margens — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("histograma", _draw_histograma, (7, 6)),
        ("taxa_erro", _draw_taxa_erro, (7, 6)),
        ("mapa_confianca", _draw_mapa_confianca, (7, 6)),
        ("violin", _draw_violin, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_experiment_summary_dashboard(data: dict, seed: int,
                                      results_abc: list, results_e: list,
                                      filename: str = None):
    """Dashboard completo do experimento para uma seed."""
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    learned_metric = data.get('learned_metric')
    colors = {0: '#3498db', 1: '#e74c3c'}

    # ─── BLOCO 1 (topo): 4 datasets com ground truth ───
    for col, (name, X, y_gt) in enumerate([
        ('A', data['X_a'], data['y_gt_a']),
        ('B', data['X_b'], data['y_gt_b']),
        ('C', data['X_c'], data['y_gt_c']),
        ('D', data['X_e'], data['y_gt_e']),
    ]):
        ax = fig.add_subplot(gs[0, col])
        for c in [0, 1]:
            mask = y_gt == c
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=10, alpha=0.5)
        ax.set_title(f'Prob. {name} (n={len(X)})', fontweight='bold', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    # ─── BLOCO 2: Fase A — acertos/erros + fronteira ───
    y_llm_a = data.get('y_llm_a')
    y_metric_a = data.get('y_metric_a')
    X_a = data['X_a']

    ax_a1 = fig.add_subplot(gs[1, 0])
    if y_llm_a is not None and y_metric_a is not None:
        hits = y_llm_a == y_metric_a
        ax_a1.scatter(X_a[hits, 0], X_a[hits, 1], c='#27ae60', s=10, alpha=0.5, label=f'OK ({hits.sum()})')
        ax_a1.scatter(X_a[~hits, 0], X_a[~hits, 1], c='#e74c3c', s=25, marker='x', label=f'Erro ({(~hits).sum()})')
        ax_a1.set_title(f'Fase A: Fidelidade {hits.mean():.1%}', fontweight='bold', fontsize=9)
        ax_a1.legend(fontsize=7)
    ax_a1.grid(True, alpha=0.2)

    ax_a2 = fig.add_subplot(gs[1, 1])
    if learned_metric is not None:
        x_min, x_max = X_a[:, 0].min() - 1, X_a[:, 0].max() + 1
        y_min, y_max = X_a[:, 1].min() - 1, X_a[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
        Z = predict_with_metric(np.c_[xx.ravel(), yy.ravel()], learned_metric.centroids, learned_metric.w).reshape(xx.shape)
        ax_a2.contourf(xx, yy, Z, alpha=0.15, cmap='RdBu')
        ax_a2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
    if y_llm_a is not None:
        for c in [0, 1]:
            ax_a2.scatter(X_a[y_llm_a == c, 0], X_a[y_llm_a == c, 1], c=colors[c], s=10, alpha=0.4)
    w_str = f'W=[{learned_metric.w[0]:.3f}, {learned_metric.w[1]:.3f}]' if learned_metric else ''
    ax_a2.set_title(f'Fase A: Fronteira {w_str}', fontweight='bold', fontsize=9)
    ax_a2.grid(True, alpha=0.2)

    # Margens Fase A
    ax_a3 = fig.add_subplot(gs[1, 2])
    if learned_metric is not None:
        conf_a, _ = compute_metric_confidence(X_a, learned_metric.centroids, learned_metric.w)
        ax_a3.hist(conf_a, bins=25, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax_a3.axvline(np.median(conf_a), color='red', linestyle='--', linewidth=1, label=f'Med={np.median(conf_a):.2f}')
        ax_a3.set_title('Margens Problema A', fontweight='bold', fontsize=9)
        ax_a3.legend(fontsize=7)
    ax_a3.grid(True, alpha=0.2)

    # W dos algoritmos
    ax_w = fig.add_subplot(gs[1, 3])
    if learned_metric is not None:
        algs = ['Perceptron']
        w_vals = [learned_metric.w]
        if data.get('w_nnls') is not None:
            algs.append('NNLS')
            w_vals.append(data['w_nnls'])
        x_pos = np.arange(2)
        width = 0.8 / len(algs)
        for i, (alg, w) in enumerate(zip(algs, w_vals)):
            offset = (i - (len(algs)-1)/2) * width
            ax_w.bar(x_pos + offset, w, width * 0.9, label=f'{alg}', alpha=0.8)
        ax_w.set_xticks(x_pos)
        ax_w.set_xticklabels(['w₁', 'w₂'])
        ax_w.set_title('Pesos W', fontweight='bold', fontsize=9)
        ax_w.legend(fontsize=7)
    ax_w.grid(True, alpha=0.2, axis='y')

    # ─── BLOCO 3: Fases B/C ───
    for col_offset, (name, X, y_llm, y_metric) in enumerate([
        ('B', data['X_b'], data.get('y_llm_b'), data.get('y_metric_b')),
        ('C', data['X_c'], data.get('y_llm_c'), data.get('y_metric_c')),
    ]):
        ax = fig.add_subplot(gs[2, col_offset])
        if y_llm is not None and y_metric is not None:
            n_min = min(len(y_llm), len(y_metric), len(X))
            hits = y_llm[:n_min] == y_metric[:n_min]
            ax.scatter(X[:n_min][hits, 0], X[:n_min][hits, 1], c='#27ae60', s=10, alpha=0.5)
            ax.scatter(X[:n_min][~hits, 0], X[:n_min][~hits, 1], c='#e74c3c', s=25, marker='x')
            ax.set_title(f'Fase {name}: Consistência {hits.mean():.1%}', fontweight='bold', fontsize=9)
        else:
            ax.set_title(f'Fase {name}: Sem dados', fontsize=9)
        ax.grid(True, alpha=0.2)

    # Barras de métricas B/C
    ax_bars = fig.add_subplot(gs[2, 2])
    seed_results = [r for r in results_abc if r.random_seed == seed and r.n_shot == 0
                    and r.nomes_classes == ("A", "B") and r.repeticao == 0]
    if seed_results:
        r = seed_results[0]
        metrics = ['Consist. B', 'Kappa B', 'F1 B', 'Consist. C', 'Kappa C', 'F1 C']
        values = [r.consistencia_problema_b, r.kappa_problema_b, r.f1_problema_b,
                  r.consistencia_problema_c, r.kappa_problema_c, r.f1_problema_c]
        bar_colors = ['#3498db']*3 + ['#e67e22']*3
        ax_bars.barh(metrics, values, color=bar_colors, alpha=0.8)
        for i, v in enumerate(values):
            ax_bars.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
        ax_bars.set_xlim(0, 1.15)
        ax_bars.set_title('Métricas Zero-Shot', fontweight='bold', fontsize=9)
    ax_bars.grid(True, alpha=0.2, axis='x')

    # Fidelidade por n_shot
    ax_fid = fig.add_subplot(gs[2, 3])
    seed_abc = [r for r in results_abc if r.random_seed == seed and r.nomes_classes == ("A", "B")]
    if seed_abc:
        for metric_name, getter, color in [
            ('Consist. B', lambda r: r.consistencia_problema_b, '#3498db'),
            ('Consist. C', lambda r: r.consistencia_problema_c, '#e67e22'),
        ]:
            by_nshot = {}
            for r in seed_abc:
                by_nshot.setdefault(r.n_shot, []).append(getter(r))
            nshots = sorted(by_nshot.keys())
            means = [np.mean(by_nshot[n]) for n in nshots]
            ax_fid.plot(nshots, means, 'o-', label=metric_name, color=color)
        ax_fid.set_xlabel('n_shot')
        ax_fid.set_title('Consistência vs n_shot', fontweight='bold', fontsize=9)
        ax_fid.legend(fontsize=7)
    ax_fid.grid(True, alpha=0.2)

    # ─── BLOCO 4: Fase E ───
    seed_d = [r for r in results_e if r.random_seed == seed]
    ax_d1 = fig.add_subplot(gs[3, 0:2])
    if seed_d:
        for strategy in ['easy', 'hard', 'mixed', 'random']:
            strat_results = [r for r in seed_d if r.example_strategy == strategy]
            if strat_results:
                by_nshot = {}
                for r in strat_results:
                    by_nshot.setdefault(r.n_shot, []).append(r.accuracy_llm_vs_expert)
                nshots = sorted(by_nshot.keys())
                means = [np.mean(by_nshot[n]) for n in nshots]
                ax_d1.plot(nshots, means, 'o-', label=strategy)
        ax_d1.set_xlabel('n_shot')
        ax_d1.set_ylabel('Acurácia LLM vs Expert')
        ax_d1.set_title('Fase E: Learning Curve por Estratégia', fontweight='bold', fontsize=9)
        ax_d1.legend(fontsize=8)
    else:
        ax_d1.text(0.5, 0.5, 'Fase E não executada', ha='center', va='center', transform=ax_d1.transAxes, color='gray')
    ax_d1.grid(True, alpha=0.2)

    # Confusion matrix A (LLM vs Métrica)
    ax_cm = fig.add_subplot(gs[3, 2])
    if y_llm_a is not None and y_metric_a is not None:
        cm = np.zeros((2, 2), dtype=int)
        for i in range(len(y_llm_a)):
            cm[int(y_metric_a[i]), int(y_llm_a[i])] += 1
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
        ax_cm.set_xlabel('LLM')
        ax_cm.set_ylabel('Métrica')
        ax_cm.set_title('Conf. Matrix A: LLM vs Métrica', fontweight='bold', fontsize=9)

    # Info textual
    ax_info = fig.add_subplot(gs[3, 3])
    ax_info.axis('off')
    info_lines = [f'Seed: {seed}']
    if learned_metric:
        w = learned_metric.w
        w_norm = w / np.sum(w) if np.sum(w) > 0 else w
        info_lines.extend([
            f'W Perceptron: [{w[0]:.4f}, {w[1]:.4f}]',
            f'W norm: [{w_norm[0]:.3f}, {w_norm[1]:.3f}]',
            f'Gamma: {learned_metric.gamma:.4f}',
        ])
    if data.get('w_nnls') is not None:
        wn = data['w_nnls']
        info_lines.append(f'W NNLS: [{wn[0]:.4f}, {wn[1]:.4f}]')
    if seed_results:
        r = seed_results[0]
        info_lines.extend([
            f'',
            f'Fidelidade A: {r.fidelidade_problema_a:.1%}',
            f'Consistência B: {r.consistencia_problema_b:.1%}',
            f'Consistência C: {r.consistencia_problema_c:.1%}',
            f'Kappa B: {r.kappa_problema_b:.3f}',
            f'Kappa C: {r.kappa_problema_c:.3f}',
        ])
    ax_info.text(0.05, 0.95, '\n'.join(info_lines), transform=ax_info.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_info.set_title('Resumo', fontweight='bold', fontsize=9)

    fig.suptitle(f'Dashboard Completo do Experimento — Seed {seed}', fontsize=16, fontweight='bold')
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def print_error_analysis_by_region(phase_a_data: dict):
    """Análise quantitativa de erros por região (distância à fronteira).

    Responde: onde o LLM erra? Perto da fronteira (margem baixa) ou longe?
    Complementa o gráfico 11 com métricas numéricas no log.
    """
    print_section("ANÁLISE DE ERROS POR REGIÃO (DISTÂNCIA À FRONTEIRA)", "═")

    for seed_key, data in sorted(phase_a_data.items()):
        X = data['X']
        y_llm = data['y_llm']
        y_metric = data['y_metric']
        w = data['w']
        centroids = data['centroids']

        agreements = y_llm == y_metric
        disagreements = ~agreements
        n_total = len(X)
        n_errors = np.sum(disagreements)

        if n_errors == 0:
            print(f"\n  Seed {seed_key}: Fidelidade perfeita (0 discordâncias)")
            continue

        confidences, _ = compute_metric_confidence(X, centroids, w)

        # Dividir em 3 regiões: baixa margem (< p33), média, alta (> p66)
        p33 = np.percentile(confidences, 33)
        p66 = np.percentile(confidences, 66)

        regions = {
            'Baixa margem (fronteira)': confidences < p33,
            'Margem média': (confidences >= p33) & (confidences < p66),
            'Alta margem (longe)': confidences >= p66,
        }

        print(f"\n  Seed {seed_key}: {n_errors}/{n_total} discordâncias ({n_errors/n_total:.1%})")
        print(f"  Margem: min={confidences.min():.3f}, mediana={np.median(confidences):.3f}, "
              f"max={confidences.max():.3f}")
        print(f"\n  {'Região':<28} {'N pontos':>10} {'Erros':>7} {'Taxa erro':>10} {'% dos erros':>12}")
        print(f"  {'─'*28} {'─'*10} {'─'*7} {'─'*10} {'─'*12}")

        for region_name, mask in regions.items():
            n_region = np.sum(mask)
            n_err_region = np.sum(disagreements[mask])
            rate = n_err_region / n_region if n_region > 0 else 0
            pct_errors = n_err_region / n_errors if n_errors > 0 else 0
            print(f"  {region_name:<28} {n_region:>10} {n_err_region:>7} {rate:>10.1%} {pct_errors:>12.1%}")

        # Correlação margem × acerto
        from scipy import stats as sp_stats
        corr, p_val = sp_stats.pointbiserialr(agreements.astype(int), confidences)
        print(f"\n  Correlação ponto-bisserial (acerto × margem): r={corr:.3f}, p={p_val:.4f}")
        if corr > 0 and p_val < 0.05:
            print(f"  → Confirmado: pontos com maior margem têm mais acertos (esperado)")
        elif p_val >= 0.05:
            print(f"  → Correlação NÃO significativa — erros não se concentram na fronteira")

        # Margem média dos pontos corretos vs errados
        margin_correct = confidences[agreements]
        margin_errors = confidences[disagreements]
        print(f"\n  Margem média dos acertos:  {np.mean(margin_correct):.3f} (±{np.std(margin_correct):.3f})")
        print(f"  Margem média dos erros:    {np.mean(margin_errors):.3f} (±{np.std(margin_errors):.3f})")

        if len(margin_errors) >= 2 and len(margin_correct) >= 2:
            stat, p_mw = sp_stats.mannwhitneyu(margin_correct, margin_errors, alternative='greater')
            print(f"  Mann-Whitney U (acertos > erros): U={stat:.0f}, p={p_mw:.4f}")
            if p_mw < 0.05:
                print(f"  → Significativo: erros têm margem menor que acertos")
            else:
                print(f"  → NÃO significativo: erros não se concentram em margens baixas")

    print()


def print_hyperparameter_sensitivity(phase_a_data: dict):
    """Análise de sensibilidade dos hiperparâmetros do Perceptron Estruturado.

    Re-executa o Perceptron com diferentes combinações de (eta, C) sobre os
    mesmos dados da Fase A (sem chamadas adicionais à API). Reporta se a
    direção de W é estável, o que indicaria robustez à escolha de hiperparâmetros.

    Custo computacional: ~O(n_configs × n_seeds × custo_perceptron), sem API calls.
    """
    print_section("SENSIBILIDADE DOS HIPERPARÂMETROS DO PERCEPTRON", "═")
    print(f"\n  Re-execução do Perceptron com diferentes (eta, C) sobre os mesmos")
    print(f"  dados da Fase A. Nenhuma chamada adicional à API é feita.")
    print(f"  Se a direção de W (cosseno) é estável, o resultado é robusto.\n")

    # Configurações a testar (inclui a configuração padrão para referência)
    # Faixa de C inclui [0.1, 1.0] conforme Coelho et al. CILAMCE 2017, p. 16
    ETA_VALUES = [0.0001, 0.001, 0.01, 0.1]
    C_VALUES = [0.1, 1.0, 10.0]
    DELTA_GAMMA_VALUES = [0.01, 0.05, 0.1]

    # Configuração padrão usada no experimento (Coelho et al. CILAMCE 2017)
    DEFAULT_ETA = 0.001
    DEFAULT_C = 1.0
    DEFAULT_DELTA = 0.05

    for seed_key, data in sorted(phase_a_data.items()):
        X = data['X']
        y_llm = data['y_llm']
        centroids = data['centroids']
        w_default = data['w']
        w_default_norm = np.linalg.norm(w_default)
        w_default_dir = w_default / w_default_norm if w_default_norm > 0 else w_default

        print(f"  Seed {seed_key}: W padrão (eta={DEFAULT_ETA}, C={DEFAULT_C}) = "
              f"[{w_default[0]:.4f}, {w_default[1]:.4f}]")

        # --- Sensibilidade a eta (C fixo) ---
        print(f"\n  Variando eta (C={DEFAULT_C}, delta_gamma={DEFAULT_DELTA}):")
        print(f"    {'eta':>6} {'W bruto':>22} {'W unitário':>22} {'cos(default)':>14} {'Fidelidade':>12}")
        print(f"    {'─'*6} {'─'*22} {'─'*22} {'─'*14} {'─'*12}")

        cos_sims_eta = []
        for eta in ETA_VALUES:
            w_test, _ = train_relaxed_perceptron(
                X, y_llm, centroids,
                eta=eta, C=DEFAULT_C, delta_gamma=DEFAULT_DELTA,
                max_epochs=50, tol=1e-4, verbose=False, use_best_effort=True
            )
            w_norm = np.linalg.norm(w_test)
            w_dir = w_test / w_norm if w_norm > 0 else w_test
            cos_sim = float(np.dot(w_default_dir, w_dir)) if w_norm > 0 else 0.0
            cos_sims_eta.append(cos_sim)
            y_pred = predict_with_metric(X, centroids, w_test)
            fid = np.mean(y_pred == y_llm)
            marker = " ← padrão" if eta == DEFAULT_ETA else ""
            print(f"    {eta:>6.2f} [{w_test[0]:>8.4f}, {w_test[1]:>8.4f}] "
                  f"[{w_dir[0]:>8.4f}, {w_dir[1]:>8.4f}] "
                  f"{cos_sim:>14.4f} {fid:>12.1%}{marker}")

        # --- Sensibilidade a C (eta fixo) ---
        print(f"\n  Variando C (eta={DEFAULT_ETA}, delta_gamma={DEFAULT_DELTA}):")
        print(f"    {'C':>6} {'W bruto':>22} {'W unitário':>22} {'cos(default)':>14} {'Fidelidade':>12}")
        print(f"    {'─'*6} {'─'*22} {'─'*22} {'─'*14} {'─'*12}")

        cos_sims_c = []
        for C in C_VALUES:
            w_test, _ = train_relaxed_perceptron(
                X, y_llm, centroids,
                eta=DEFAULT_ETA, C=C, delta_gamma=DEFAULT_DELTA,
                max_epochs=50, tol=1e-4, verbose=False, use_best_effort=True
            )
            w_norm = np.linalg.norm(w_test)
            w_dir = w_test / w_norm if w_norm > 0 else w_test
            cos_sim = float(np.dot(w_default_dir, w_dir)) if w_norm > 0 else 0.0
            cos_sims_c.append(cos_sim)
            y_pred = predict_with_metric(X, centroids, w_test)
            fid = np.mean(y_pred == y_llm)
            marker = " ← padrão" if C == DEFAULT_C else ""
            print(f"    {C:>6.1f} [{w_test[0]:>8.4f}, {w_test[1]:>8.4f}] "
                  f"[{w_dir[0]:>8.4f}, {w_dir[1]:>8.4f}] "
                  f"{cos_sim:>14.4f} {fid:>12.1%}{marker}")

        # --- Sensibilidade a delta_gamma (eta, C fixos) ---
        print(f"\n  Variando delta_gamma (eta={DEFAULT_ETA}, C={DEFAULT_C}):")
        print(f"    {'δγ':>6} {'W bruto':>22} {'W unitário':>22} {'cos(default)':>14} {'Fidelidade':>12}")
        print(f"    {'─'*6} {'─'*22} {'─'*22} {'─'*14} {'─'*12}")

        cos_sims_dg = []
        for dg in DELTA_GAMMA_VALUES:
            w_test, _ = train_relaxed_perceptron(
                X, y_llm, centroids,
                eta=DEFAULT_ETA, C=DEFAULT_C, delta_gamma=dg,
                max_epochs=50, tol=1e-4, verbose=False, use_best_effort=True
            )
            w_norm = np.linalg.norm(w_test)
            w_dir = w_test / w_norm if w_norm > 0 else w_test
            cos_sim = float(np.dot(w_default_dir, w_dir)) if w_norm > 0 else 0.0
            cos_sims_dg.append(cos_sim)
            y_pred = predict_with_metric(X, centroids, w_test)
            fid = np.mean(y_pred == y_llm)
            marker = " ← padrão" if dg == DEFAULT_DELTA else ""
            print(f"    {dg:>6.2f} [{w_test[0]:>8.4f}, {w_test[1]:>8.4f}] "
                  f"[{w_dir[0]:>8.4f}, {w_dir[1]:>8.4f}] "
                  f"{cos_sim:>14.4f} {fid:>12.1%}{marker}")

        # --- Veredicto ---
        all_cos = cos_sims_eta + cos_sims_c + cos_sims_dg
        min_cos = min(all_cos) if all_cos else 0
        mean_cos = np.mean(all_cos) if all_cos else 0
        n_configs = len(all_cos)

        print(f"\n  Resumo seed {seed_key}: {n_configs} configurações testadas")
        print(f"    Cosseno mínimo com padrão: {min_cos:.4f}")
        print(f"    Cosseno médio com padrão:  {mean_cos:.4f}")

        if min_cos > 0.95:
            print(f"    → W ROBUSTO aos hiperparâmetros (cos mín > 0.95)")
        elif min_cos > 0.80:
            print(f"    → W MODERADAMENTE sensível (0.80 < cos mín < 0.95)")
        else:
            print(f"    → W SENSÍVEL aos hiperparâmetros (cos mín < 0.80) — cautela nas conclusões")

        print()

    print(f"  Nota: apenas a DIREÇÃO de W importa (escala é arbitrária).")
    print(f"  Cosseno > 0.95 entre configs = hiperparâmetros não afetam a fronteira.\n")


def print_example_order_analysis(results_order: List[ResultadoPhaseEExperimento]):
    """Análise quantitativa do viés de ordem dos exemplos few-shot (recency bias).

    Testa se a ordenação dos exemplos afeta significativamente a performance do LLM.
    Recency bias: LLMs tendem a dar mais peso aos últimos exemplos do prompt.
    """
    from scipy import stats as sp_stats

    print_section("ANÁLISE DE VIÉS DE ORDEM DOS EXEMPLOS (RECENCY BIAS)", "═")

    df = pd.DataFrame([{
        'seed': r.random_seed, 'rep': r.repeticao,
        'n_shot': r.n_shot,
        'ordering': r.example_strategy.replace("mixed_order_", ""),
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
    } for r in results_order])

    orderings = sorted(df['ordering'].unique())
    n_shots = sorted(df['n_shot'].unique())

    # Tabela resumo
    print(f"\n  {'Ordenação':<18} {'n_shot':>6} {'Acc média':>10} {'±std':>8} {'Kappa':>8} {'n':>4}")
    print(f"  {'─'*18} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*4}")

    for ns in n_shots:
        for ordering in orderings:
            subset = df[(df['ordering'] == ordering) & (df['n_shot'] == ns)]
            if len(subset) > 0:
                print(f"  {ordering:<18} {ns:>6} {subset['accuracy'].mean():>10.1%} "
                      f"{subset['accuracy'].std():>8.1%} {subset['kappa'].mean():>8.3f} "
                      f"{len(subset):>4}")
        print()

    # Teste estatístico por n_shot: Kruskal-Wallis (não-paramétrico, >2 grupos)
    print(f"  TESTES DE SIGNIFICÂNCIA:")
    print(f"  {'─'*60}")

    for ns in n_shots:
        groups = []
        group_names = []
        for ordering in orderings:
            vals = df[(df['ordering'] == ordering) & (df['n_shot'] == ns)]['accuracy'].values
            if len(vals) >= 2:
                groups.append(vals)
                group_names.append(ordering)

        if len(groups) >= 2:
            # Kruskal-Wallis: H0 = todas as ordenações têm mesma distribuição
            stat, p_val = sp_stats.kruskal(*groups)
            print(f"\n  {ns}-shot: Kruskal-Wallis H={stat:.3f}, p={p_val:.4f}")

            if p_val < 0.05:
                print(f"  → SIGNIFICATIVO: a ordem dos exemplos AFETA a performance")
                # Identificar qual ordenação é melhor/pior
                means = {name: np.mean(g) for name, g in zip(group_names, groups)}
                best = max(means, key=means.get)
                worst = min(means, key=means.get)
                diff = means[best] - means[worst]
                print(f"    Melhor: {best} ({means[best]:.1%}), Pior: {worst} ({means[worst]:.1%}), Δ={diff:+.1%}")

                # Teste pareado: class0_first vs class1_first (recency bias direto)
                if 'class0_first' in group_names and 'class1_first' in group_names:
                    c0 = df[(df['ordering'] == 'class0_first') & (df['n_shot'] == ns)]
                    c1 = df[(df['ordering'] == 'class1_first') & (df['n_shot'] == ns)]
                    # Parear por seed
                    c0_by_seed = c0.groupby('seed')['accuracy'].mean()
                    c1_by_seed = c1.groupby('seed')['accuracy'].mean()
                    common = c0_by_seed.index.intersection(c1_by_seed.index)
                    if len(common) >= 3:
                        diff_paired = c0_by_seed.loc[common].values - c1_by_seed.loc[common].values
                        mean_d, lo_d, hi_d = bootstrap_ci(diff_paired)
                        print(f"    Classe0_first - Classe1_first: Δ={mean_d:+.3f} "
                              f"CI95%=[{lo_d:+.3f}, {hi_d:+.3f}]")
                        if lo_d > 0:
                            print(f"    → Recency bias: última classe vista (classe 1) é FAVORECIDA")
                        elif hi_d < 0:
                            print(f"    → Recency bias: última classe vista (classe 0) é FAVORECIDA")
            else:
                print(f"  → NÃO significativo: a ordem dos exemplos NÃO afeta a performance")

    # Variabilidade geral por ordering (colapsando n_shots)
    overall = df.groupby('ordering')['accuracy'].agg(['mean', 'std'])
    max_range = overall['mean'].max() - overall['mean'].min()
    print(f"\n  Variação total entre ordenações: {max_range:.1%}")
    if max_range < 0.03:
        print(f"  → Efeito de ordem NEGLIGÍVEL (< 3 p.p.)")
    elif max_range < 0.10:
        print(f"  → Efeito de ordem MODERADO (3-10 p.p.) — reportar como limitação")
    else:
        print(f"  → Efeito de ordem GRANDE (> 10 p.p.) — recency bias significativo")

    print()


def print_phase_e_analysis(results_e: List[ResultadoPhaseEExperimento]):
    """Imprime análise detalhada dos resultados da Fase E."""
    print_section("ANÁLISE DA FASE E: LLM COMO APRENDIZ", "═")

    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (temp={r.temperature})",
            'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
            'f1': r.f1_llm_vs_expert,
            'acc_vs_gt': r.accuracy_llm_vs_gt,
            'expert_vs_gt': r.accuracy_expert_vs_gt,
            'n_disagreements': r.n_disagreements,
            'n_malformed': r.n_malformed_responses,
        }
        for r in results_e
    ])

    models = df['model'].unique()

    for model in models:
        model_df = df[df['model'] == model]
        print(f"\n{'═' * 70}")
        print(f" MODELO: {model}")
        print(f"{'═' * 70}")

        print(f"\n  Métrica do Perito W = [{EXPERT_W[0]:.2f}, {EXPERT_W[1]:.2f}]")
        print(f"  Acurácia do Perito vs. GT: {model_df['expert_vs_gt'].mean():.1%}")

        # 1. Curva de aprendizado por estratégia
        print(f"\n  1. CURVA DE APRENDIZADO (Concordância LLM vs. Perito):")
        print("  " + "-" * 65)
        header = f"  {'n_shot':>6}"
        for strat in EXAMPLE_STRATEGIES:
            header += f" | {strat:>12}"
        print(header)
        print("  " + "-" * 65)

        for n in sorted(model_df['n_shot'].unique()):
            line = f"  {n:>6}"
            for strat in EXAMPLE_STRATEGIES:
                subset = model_df[(model_df['n_shot'] == n) & (model_df['strategy'] == strat)]
                if len(subset) > 0:
                    mean = subset['accuracy'].mean()
                    std = subset['accuracy'].std()
                    line += f" | {mean:.1%}±{std:.1%}"
                else:
                    line += f" | {'N/D':>12}"
            print(line)

        # 2. Melhor estratégia por n_shot
        print(f"\n  2. MELHOR ESTRATÉGIA POR N_SHOT:")
        print("  " + "-" * 50)
        for n in sorted(model_df['n_shot'].unique()):
            if n == 0:
                continue
            subset = model_df[model_df['n_shot'] == n]
            best = subset.groupby('strategy')['accuracy'].mean().idxmax()
            best_val = subset.groupby('strategy')['accuracy'].mean().max()
            print(f"     {n:>3}-shot: {best.upper():>8} ({best_val:.1%})")

        # 3. Melhoria do zero-shot para o melhor few-shot
        print(f"\n  3. MELHORIA DO ZERO-SHOT PARA O MELHOR FEW-SHOT:")
        print("  " + "-" * 50)
        zero_shot_acc = model_df[model_df['n_shot'] == 0]['accuracy'].mean()
        print(f"     Linha de base zero-shot: {zero_shot_acc:.1%}")

        for strat in EXAMPLE_STRATEGIES:
            strat_df = model_df[model_df['strategy'] == strat]
            if len(strat_df) == 0:
                continue
            best_n = strat_df.groupby('n_shot')['accuracy'].mean().idxmax()
            best_acc = strat_df.groupby('n_shot')['accuracy'].mean().max()
            improvement = best_acc - zero_shot_acc
            print(f"     {strat.upper():>8}: melhor={best_acc:.1%} em {best_n}-shot "
                  f"(Δ={improvement:+.1%})")

        # 4. Respostas malformadas
        total_malformed = model_df['n_malformed'].sum()
        if total_malformed > 0:
            print(f"\n  ⚠️ Total de respostas malformadas: {total_malformed}")

    # Conclusão geral do experimento
    overall_zero = df[df['n_shot'] == 0]['accuracy'].mean()
    overall_best = df.groupby(['n_shot', 'strategy'])['accuracy'].mean().max()
    best_config = df.groupby(['n_shot', 'strategy'])['accuracy'].mean().idxmax()

    print_box(f"""
CONCLUSÃO DA FASE E: LLM COMO APRENDIZ

Métrica do perito: W = [{EXPERT_W[0]:.2f}, {EXPERT_W[1]:.2f}]
(Pondera a dimensão x2 {EXPERT_W[1]/EXPERT_W[0]:.1f}x mais do que x1)

Linha de base zero-shot: {overall_zero:.1%}
Melhor resultado few-shot: {overall_best:.1%} (estratégia {best_config[1]}, {best_config[0]}-shot)
Melhoria geral: {overall_best - overall_zero:+.1%}

{'✓ O LLM CONSEGUE aprender com exemplos do perito — o desempenho melhora com mais exemplos.' if overall_best - overall_zero > 0.1 else
 '~ O LLM mostra aprendizado MODERADO com exemplos do perito.' if overall_best - overall_zero > 0.05 else
 '✗ O LLM NÃO melhora significativamente com os exemplos do perito.'}
"""
    )

    # --- Análise data-driven de H4 (easy vs hard) ---
    # Compara acurácia média de cada estratégia (exceto zero-shot)
    df_fewshot = df[df['n_shot'] > 0]
    if len(df_fewshot) > 0:
        strat_means = df_fewshot.groupby('strategy')['accuracy'].mean()
        easy_mean = strat_means.get('easy', None)
        hard_mean = strat_means.get('hard', None)
        random_mean = strat_means.get('random', None)

        print_box(f"""
AVALIAÇÃO DA HIPÓTESE H4: "Exemplos hard são mais informativos que easy"

Acurácia média por estratégia (todos os n_shot > 0):
{chr(10).join(f'  {s.upper():>8}: {v:.1%}' for s, v in sorted(strat_means.items(), key=lambda x: -x[1]))}

{'RESULTADO: H4 REFUTADA.' if easy_mean is not None and hard_mean is not None and easy_mean > hard_mean else 'RESULTADO: H4 sustentável.' if easy_mean is not None and hard_mean is not None and hard_mean > easy_mean else 'RESULTADO: Dados insuficientes para avaliar H4.'}
{f'Exemplos easy ({easy_mean:.1%}) superam hard ({hard_mean:.1%}) consistentemente.' if easy_mean is not None and hard_mean is not None and easy_mean > hard_mean else ''}
{f'Exemplos hard ({hard_mean:.1%}) performam abaixo de random ({random_mean:.1%}).' if hard_mean is not None and random_mean is not None and hard_mean < random_mean else ''}

Interpretação: exemplos próximos à fronteira de decisão são AMBÍGUOS para o
LLM — não fornecem padrões claros de cada classe. Exemplos fáceis funcionam
como "âncoras" que definem bem as regiões de cada classe, permitindo ao LLM
interpolar para os pontos intermediários. Este é um achado relevante sobre
o mecanismo de aprendizado in-context dos LLMs: eles se beneficiam mais de
exemplos prototípicos (alta margem) do que de exemplos fronteiriços.

Nota: esta hipótese foi formulada a priori e refutada pelos dados.
A refutação é reportada como resultado genuíno, não como falha do método.
"""
        )


def print_final_analysis(resultados: List[ResultadoExperimento]):
    """Imprime a análise final dos resultados das Fases A-C."""
    print_section("ANÁLISE FINAL: FASES A-C", "═")

    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (temp={r.temperature})",
            'seed': r.random_seed,
            'n_shot': r.n_shot,
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'f1_b': r.f1_problema_b,
            'f1_c': r.f1_problema_c,
            'fidelidade': r.fidelidade_problema_a,
            'n_disagreements_b': r.n_disagreements_b,
            'n_disagreements_c': r.n_disagreements_c,
            'n_malformed': r.n_malformed_responses,
        }
        for r in resultados
    ])

    models = df['model'].unique()
    seeds = sorted(df['seed'].unique())

    for model in models:
        model_df = df[df['model'] == model]
        print(f"\n{'═' * 70}")
        print(f" MODELO: {model}")
        print(f"{'═' * 70}")

        print("\n  1. CONSISTÊNCIA POR NÚMERO DE EXEMPLOS:")
        print(f"     {'n_shot':>6} | {'Consist B':>10} | {'Consist C':>10} | {'Kappa B':>8} | {'Kappa C':>8}")
        print("     " + "-" * 55)
        for n in sorted(model_df['n_shot'].unique()):
            subset = model_df[model_df['n_shot'] == n]
            c_b = f"{subset['consistencia_b'].mean():.1%}±{subset['consistencia_b'].std():.1%}"
            c_c = f"{subset['consistencia_c'].mean():.1%}±{subset['consistencia_c'].std():.1%}"
            k_b = f"{subset['kappa_b'].mean():.3f}"
            k_c = f"{subset['kappa_c'].mean():.3f}"
            print(f"     {n:>6} | {c_b:>10} | {c_c:>10} | {k_b:>8} | {k_c:>8}")

        print("\n  2. CONSISTÊNCIA POR NOMES DE CLASSE:")
        for nome in model_df['nomes'].unique():
            subset = model_df[model_df['nomes'] == nome]
            print(f"     {nome:20s}: B={subset['consistencia_b'].mean():.1%}, C={subset['consistencia_c'].mean():.1%}")

    mean_b = df['consistencia_b'].mean()
    mean_c = df['consistencia_c'].mean()

    print_box(f"""
RESUMO DAS FASES A-C

Consistência Média Problema B: {mean_b:.1%} ± {df['consistencia_b'].std():.1%}
Consistência Média Problema C: {mean_c:.1%} ± {df['consistencia_c'].std():.1%}

{'✓ Os LLMs MANTÊM alta consistência em ambos os problemas.' if min(mean_b, mean_c) > 0.85 else
 '~ Consistência MODERADA entre os problemas.' if min(mean_b, mean_c) > 0.7 else
 '✗ Os LLMs NÃO mantêm consistência.'}
""")


# =============================================================================
# SIGNIFICÂNCIA ESTATÍSTICA
# =============================================================================

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calcula intervalo de confiança via bootstrap.

    Retorna (média, limite_inferior, limite_superior).
    Usa o método percentil (Efron & Tibshirani, 1993).
    """
    if len(data) < 2:
        return float(np.mean(data)), float(np.mean(data)), float(np.mean(data))
    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(data)), float(lo), float(hi)


def print_statistical_summary(
    all_results_abc: List[ResultadoExperimento],
    all_results_e: List[ResultadoPhaseEExperimento],
):
    """Imprime sumário estatístico com bootstrap CI e testes de significância.

    Endereça a crítica de ausência de testes estatísticos formais:
    - Bootstrap 95% CI para todas as métricas principais
    - Wilcoxon signed-rank test para comparações pareadas
    - Tamanho de efeito (Cohen's d) quando aplicável
    """
    from scipy import stats

    print_section("SUMÁRIO ESTATÍSTICO (Bootstrap 95% CI + Testes de Significância)", "═")

    # --- Fases A-C ---
    if all_results_abc:
        print(f"\n  ════════════════════════════════════════════════")
        print(f"  FASES A-C: INTERVALOS DE CONFIANÇA (Bootstrap)")
        print(f"  ════════════════════════════════════════════════")

        # Filtrar apenas classes A/B, prompt default para métricas limpas
        df = pd.DataFrame([{
            'seed': r.random_seed, 'n_shot': r.n_shot,
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'prompt_variant': r.prompt_variant,
            'fidelidade': r.fidelidade_problema_a,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'f1_b': r.f1_problema_b,
            'f1_c': r.f1_problema_c,
        } for r in all_results_abc])

        # Métricas por n_shot (classes A/B, default prompt)
        df_default = df[(df['nomes'] == 'A/B') & (df['prompt_variant'] == 'default')]
        if len(df_default) > 0:
            print(f"\n  Configuração base: classes A/B, prompt default")
            print(f"  n = {len(df_default)} observações ({len(df_default['seed'].unique())} seeds)")
            print(f"\n  {'Métrica':<25} {'n_shot':>6} {'Média':>8} {'95% CI':>20} {'n':>4}")
            print(f"  {'─'*25} {'─'*6} {'─'*8} {'─'*20} {'─'*4}")

            for n_shot in sorted(df_default['n_shot'].unique()):
                subset = df_default[df_default['n_shot'] == n_shot]
                print(f"  --- Perceptron ---")
                for col, label in [
                    ('fidelidade', 'Fidelidade A (Perc)'),
                    ('consistencia_b', 'Consistência B (Perc)'),
                    ('consistencia_c', 'Consistência C (Perc)'),
                    ('kappa_b', 'Kappa B (Perc)'),
                    ('kappa_c', 'Kappa C (Perc)'),
                ]:
                    values = subset[col].values
                    mean, lo, hi = bootstrap_ci(values)
                    ci_str = f"[{lo:.3f}, {hi:.3f}]"
                    print(f"  {label:<25} {n_shot:>6} {mean:>8.3f} {ci_str:>20} {len(values):>4}")
                print()

        # Teste: consistência zero-shot vs. 10-shot (pareado por seed)
        df_0 = df_default[df_default['n_shot'] == 0].groupby('seed')[['consistencia_b', 'consistencia_c']].mean()
        df_10 = df_default[df_default['n_shot'] == 10].groupby('seed')[['consistencia_b', 'consistencia_c']].mean() if 10 in df_default['n_shot'].values else None

        if df_10 is not None:
            common_seeds = df_0.index.intersection(df_10.index)
            if len(common_seeds) >= 3:
                print(f"\n  TESTE DE SIGNIFICÂNCIA: Zero-shot vs. 10-shot (pareado por seed)")
                print(f"  {'─'*60}")
                for col, label in [('consistencia_b', 'Consistência B'), ('consistencia_c', 'Consistência C')]:
                    vals_0 = df_0.loc[common_seeds, col].values
                    vals_10 = df_10.loc[common_seeds, col].values
                    diff = vals_10 - vals_0

                    mean_diff, lo_diff, hi_diff = bootstrap_ci(diff)

                    # Wilcoxon signed-rank (não-paramétrico, pareado)
                    if len(common_seeds) >= 5 and not np.all(diff == 0):
                        stat, p_value = stats.wilcoxon(vals_0, vals_10)
                        p_str = f"p={p_value:.4f}" if p_value >= 0.001 else f"p<0.001"
                    else:
                        p_str = f"n<5, teste não aplicável"

                    # Cohen's d
                    pooled_std = np.sqrt((np.std(vals_0)**2 + np.std(vals_10)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                    print(f"  {label}: Δ={mean_diff:+.3f} CI95%=[{lo_diff:+.3f}, {hi_diff:+.3f}]  "
                          f"{p_str}  d={cohens_d:.2f}")

                print(f"\n  Interpretação Cohen's d: |d|<0.2 negligível, 0.2-0.5 pequeno, 0.5-0.8 médio, >0.8 grande")

        # Teste: consistência métrica estimada vs. Euclidiana
        euc_data = [(r.consistencia_problema_b, r.consistencia_euclidiana_problema_b)
                     for r in all_results_abc
                     if r.nomes_classes == ("A", "B") and r.prompt_variant == "default" and r.n_shot == 0]
        if len(euc_data) >= 3:
            metric_vals = np.array([x[0] for x in euc_data])
            euc_vals = np.array([x[1] for x in euc_data])
            diff_euc = metric_vals - euc_vals
            mean_d, lo_d, hi_d = bootstrap_ci(diff_euc)

            print(f"\n  TESTE: Métrica estimada vs. Euclidiana (zero-shot, Problema B)")
            print(f"  {'─'*60}")
            print(f"  Δ(métrica - euclidiana) = {mean_d:+.3f} CI95%=[{lo_d:+.3f}, {hi_d:+.3f}]")
            if lo_d > 0:
                print(f"  → Métrica estimada SIGNIFICATIVAMENTE superior à Euclidiana")
            elif hi_d < 0:
                print(f"  → Euclidiana SIGNIFICATIVAMENTE superior — limitação diagonal?")
            else:
                print(f"  → Diferença NÃO significativa (CI inclui zero)")

        # --- Estabilidade de W entre seeds (H5) ---
        # A magnitude de W é arbitrária (depende da escala da margem-alvo).
        # O que importa é a DIREÇÃO: w_ratio = w1/w2 ou o ângulo do vetor unitário.
        # Se w_ratio é estável entre seeds (mesmo n_shot), H5 é sustentável.
        print(f"\n  ════════════════════════════════════════════════")
        print(f"  ESTABILIDADE DE W ENTRE SEEDS (H5)")
        print(f"  ════════════════════════════════════════════════")
        print(f"\n  IMPORTANTE: Valores absolutos de W são irrelevantes para a fronteira")
        print(f"  de decisão — apenas a DIREÇÃO (razão w1/w2) determina a geometria.")
        print(f"  Multiplicar W por k>0 não altera classificações.\n")

        for n_shot_val in sorted(df_default['n_shot'].unique()):
            subset_results = [r for r in all_results_abc
                              if r.nomes_classes == ("A", "B")
                              and r.prompt_variant == "default"
                              and r.n_shot == n_shot_val
                              and r.w_aprendido is not None
                              and np.linalg.norm(r.w_aprendido) > 0]

            if len(subset_results) < 2:
                continue

            # Coletar direções (vetores unitários) e razões
            directions = []
            ratios = []
            w_raw = []
            for r in subset_results:
                w = r.w_aprendido
                w_raw.append(w.copy())
                norm = np.linalg.norm(w)
                if norm > 0:
                    directions.append(w / norm)
                ratio = w[0] / w[1] if w[1] != 0 else float('inf')
                ratios.append(ratio)

            ratios_finite = [r for r in ratios if np.isfinite(r)]

            print(f"  n_shot={n_shot_val} ({len(subset_results)} observações, "
                  f"{len(set(r.random_seed for r in subset_results))} seeds):")

            # Tabela: seed | W bruto | W normalizado | w1/w2 | cos(NNLS)
            print(f"    {'Seed':>6} {'Rep':>4} {'W bruto':>20} {'W unitário':>20} {'w1/w2':>8} {'cos(NNLS)':>10}")
            print(f"    {'─'*6} {'─'*4} {'─'*20} {'─'*20} {'─'*8} {'─'*10}")
            for r in subset_results:
                w = r.w_aprendido
                norm = np.linalg.norm(w)
                w_unit = w / norm if norm > 0 else w
                ratio = w[0] / w[1] if w[1] != 0 else float('inf')
                cos_nnls = r.w_cosine_sim_nnls
                print(f"    {r.random_seed:>6} {r.repeticao:>4} "
                      f"[{w[0]:>7.3f}, {w[1]:>7.3f}] "
                      f"[{w_unit[0]:>7.3f}, {w_unit[1]:>7.3f}] "
                      f"{ratio:>8.3f} "
                      f"{cos_nnls:>10.4f}")

            # Similaridade cosseno entre todos os pares de W (entre seeds)
            if len(directions) >= 2:
                cos_sims = []
                for i in range(len(directions)):
                    for j in range(i + 1, len(directions)):
                        cos_sims.append(float(np.dot(directions[i], directions[j])))
                mean_cos, lo_cos, hi_cos = bootstrap_ci(np.array(cos_sims)) if len(cos_sims) >= 3 else (np.mean(cos_sims), np.min(cos_sims), np.max(cos_sims))
                print(f"\n    Similaridade cosseno entre seeds (pares): "
                      f"média={mean_cos:.4f}, range=[{lo_cos:.4f}, {hi_cos:.4f}]")
                if mean_cos > 0.95:
                    print(f"    → W ESTÁVEL entre seeds (cos > 0.95) — H5 sustentável")
                elif mean_cos > 0.80:
                    print(f"    → W MODERADAMENTE estável (0.80 < cos < 0.95)")
                else:
                    print(f"    → W INSTÁVEL entre seeds (cos < 0.80) — H5 em risco")

            # Bootstrap CI da razão w1/w2
            if len(ratios_finite) >= 2:
                mean_r, lo_r, hi_r = bootstrap_ci(np.array(ratios_finite)) if len(ratios_finite) >= 3 else (np.mean(ratios_finite), np.min(ratios_finite), np.max(ratios_finite))
                print(f"    Razão w1/w2: média={mean_r:.4f} CI95%=[{lo_r:.4f}, {hi_r:.4f}]")
                cv = np.std(ratios_finite) / np.mean(ratios_finite) if np.mean(ratios_finite) != 0 else float('inf')
                print(f"    Coeficiente de variação (CV): {cv:.2%}")
                if cv < 0.15:
                    print(f"    → Razão w1/w2 ESTÁVEL (CV < 15%)")
                elif cv < 0.30:
                    print(f"    → Razão w1/w2 MODERADAMENTE estável (15% < CV < 30%)")
                else:
                    print(f"    → Razão w1/w2 INSTÁVEL (CV > 30%) — métrica pode não ser captável")

            # Similaridade cosseno média entre algoritmos (robustez ao método)
            cos_nnls_vals = [r.w_cosine_sim_nnls for r in subset_results if r.w_cosine_sim_nnls > 0]
            if cos_nnls_vals:
                print(f"    Cosseno Perceptron-NNLS: média={np.mean(cos_nnls_vals):.4f} "
                      f"(min={np.min(cos_nnls_vals):.4f}, max={np.max(cos_nnls_vals):.4f})")

            print()

    # --- Fase E ---
    if all_results_e:
        print(f"\n  ════════════════════════════════════════════════")
        print(f"  FASE E: INTERVALOS DE CONFIANÇA (Bootstrap)")
        print(f"  ════════════════════════════════════════════════")

        df_d = pd.DataFrame([{
            'seed': r.random_seed, 'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'expert': r.expert_name,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
        } for r in all_results_e])

        # CI por (n_shot, strategy) para expert principal
        df_d_main = df_d[df_d['expert'] == df_d['expert'].iloc[0]] if len(df_d) > 0 else df_d
        if len(df_d_main) > 0:
            print(f"\n  Expert: {df_d_main['expert'].iloc[0]}")
            print(f"\n  {'Estratégia':<12} {'n_shot':>6} {'Acc Média':>10} {'95% CI':>20} {'n':>4}")
            print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*20} {'─'*4}")

            for strategy in sorted(df_d_main['strategy'].unique()):
                for n_shot in sorted(df_d_main['n_shot'].unique()):
                    subset = df_d_main[(df_d_main['strategy'] == strategy) & (df_d_main['n_shot'] == n_shot)]
                    if len(subset) > 0:
                        values = subset['accuracy'].values
                        mean, lo, hi = bootstrap_ci(values)
                        ci_str = f"[{lo:.3f}, {hi:.3f}]"
                        print(f"  {strategy:<12} {n_shot:>6} {mean:>10.3f} {ci_str:>20} {len(values):>4}")
                print()

        # Teste: easy vs. hard (pareado por seed, n_shot=10)
        for n_test in [5, 10, 20]:
            df_easy = df_d_main[(df_d_main['strategy'] == 'easy') & (df_d_main['n_shot'] == n_test)]
            df_hard = df_d_main[(df_d_main['strategy'] == 'hard') & (df_d_main['n_shot'] == n_test)]
            if len(df_easy) >= 3 and len(df_hard) >= 3:
                easy_by_seed = df_easy.groupby('seed')['accuracy'].mean()
                hard_by_seed = df_hard.groupby('seed')['accuracy'].mean()
                common = easy_by_seed.index.intersection(hard_by_seed.index)
                if len(common) >= 3:
                    diff_eh = easy_by_seed.loc[common].values - hard_by_seed.loc[common].values
                    mean_d, lo_d, hi_d = bootstrap_ci(diff_eh)
                    print(f"  TESTE easy vs. hard ({n_test}-shot): "
                          f"Δ={mean_d:+.3f} CI95%=[{lo_d:+.3f}, {hi_d:+.3f}]"
                          f" {'(sig.)' if lo_d > 0 or hi_d < 0 else '(n.s.)'}")

    print(f"\n  Nota: Bootstrap com 10.000 reamostras, seed fixa=42.")
    print(f"  Wilcoxon signed-rank test usado para comparações pareadas (não-paramétrico).")
    print(f"  Significância estatística avaliada pelo CI 95% não incluir zero.\n")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def _run_phase_abc_experiment(X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                              n_shot, nome_0, nome_1, rep, provider, model_name,
                              temperature, seed, seed_idx, phase_a_cache, verbose,
                              prompt_variant="default"):
    """Helper para executar um experimento A-C com cache."""
    cache_key = (seed, nome_0, nome_1, prompt_variant)
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
        eta=0.001, C=1.0, delta_gamma=0.05,  # Coelho et al. CILAMCE 2017, p. 16
        max_epochs=50, tol=1e-4, verbose=False,
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
      - 'llm_label_maps': dict {(seed, variant): {'X_train', 'y_llm', 'y_true'}}
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
        np.random.seed(seed)
        idx = np.random.permutation(len(X))
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

            llm_label_maps[(seed, feat_0, feat_1, 'train')] = {
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
                if n_shot == 0:
                    examples_for_prompt = None
                else:
                    # Escolhe exemplos balanceados de classes diferentes
                    n_shot_actual = min(n_shot, len(X_train))
                    ex_indices = np.random.choice(len(X_train), size=n_shot_actual, replace=False)
                    examples_for_prompt = []
                    for ei in ex_indices:
                        row = [X_train[ei, 0], X_train[ei, 1]]
                        if best['n_features'] >= 3:
                            row.extend(X_train_aug[ei, 2:].tolist())
                        row.append(nome_classe_0 if y_metric_train[ei] == 0 else nome_classe_1)
                        examples_for_prompt.append(tuple(row))

                extra_matrix_test = X_test_aug[:, 2:] if best['n_features'] >= 3 else None

                t_d = time.time()
                print(f"    [{problem_name} D seed={seed} {feat_0}/{feat_1}] n_shot={n_shot} | best n_feat={best['n_features']} | coletando LLM (n_test={len(X_test)})...", flush=True)
                y_llm_test, n_mal_d = collect_llm_decisions(
                    X_test, nome_classe_0, nome_classe_1,
                    examples=examples_for_prompt, verbose=False,
                    nome_feature_0=feat_0, nome_feature_1=feat_1,
                    extra_features_matrix=extra_matrix_test,
                    extra_feature_names=extra_features_names_test,
                    label_prefix=f"[{problem_name} D seed={seed} {feat_0}/{feat_1} n={n_shot}] ",
                )
                print(f"    [{problem_name} D seed={seed} {feat_0}/{feat_1}] n_shot={n_shot} coletado em {time.time()-t_d:.1f}s (malformadas={n_mal_d})", flush=True)

                llm_label_maps[(seed, feat_0, feat_1, n_shot)] = {
                    'X': X_test.copy(), 'y_llm': y_llm_test.copy(),
                    'y_true': y_test.copy(), 'y_metric': y_metric_test.copy(),
                }

                acc_llm_vs_metric = accuracy_score(y_metric_test, y_llm_test)
                acc_llm_vs_true = accuracy_score(y_test, y_llm_test)
                kappa = cohen_kappa_score(y_metric_test, y_llm_test)
                f1 = f1_score(y_metric_test, y_llm_test, zero_division=0)
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
                                eta=0.001, C=1.0, delta_gamma=0.05,  # Coelho et al. CILAMCE 2017, p. 16
                                max_epochs=50, tol=1e-4, verbose=False,
                                use_best_effort=True,
                            )
                            y_baseline = predict_with_metric(X_test_aug, c_ex, w_baseline)
                            acc_perc_baseline = accuracy_score(y_test, y_baseline)
                    except Exception as e:
                        acc_perc_baseline = None

                all_phase_e.append({
                    'problem_name': problem_name,
                    'seed': seed,
                    'feature_names': (feat_0, feat_1),
                    'n_features': best['n_features'],
                    'n_shot': n_shot,
                    'accuracy_llm_vs_metric': acc_llm_vs_metric,
                    'accuracy_llm_vs_true': acc_llm_vs_true,
                    'kappa_llm_vs_metric': kappa,
                    'f1_llm_vs_metric': f1,
                    'accuracy_perceptron_baseline_vs_true': acc_perc_baseline,
                    'n_malformed': n_mal_d,
                })

                if verbose:
                    baseline_str = f" | perc_baseline={acc_perc_baseline:.1%}" if acc_perc_baseline is not None else ""
                    print(
                        f"      D n_shot={n_shot}: LLM_vs_metric={acc_llm_vs_metric:.1%} | "
                        f"LLM_vs_real={acc_llm_vs_true:.1%}{baseline_str}"
                    )

    return {
        'phase_a_results': all_phase_a,
        'phase_e_results': all_phase_e,
        'llm_label_maps': llm_label_maps,
    }


def plot_problem_overview(
    X: np.ndarray,
    y_true: np.ndarray,
    title: str,
    feature_names: Tuple[str, str] = ("x1", "x2"),
    optimal_boundary_fn: Optional[callable] = None,
    filename: Optional[str] = None,
) -> None:
    """Visualiza um problema (scatter por classe + fronteira ótima opcional).

    Usado para os problemas novos (meia-lua, peso × altura) que nao aparecem
    nos gráficos `01_all_three_problems.png` (limitados a A/B/C).

    Args:
        optimal_boundary_fn: função f(x1, x2) cuja curva f=0 será sobreposta
            (ex.: classificador ótimo bayesiano do peso×altura é a elipse
            x2² - x2 + x1² - x1 + cte).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap_class = {0: "tab:blue", 1: "tab:red"}
    for c in [0, 1]:
        mask = y_true == c
        ax.scatter(
            X[mask, 0], X[mask, 1], c=cmap_class[c], s=50,
            edgecolor="black", linewidth=0.5, alpha=0.85,
            label=f"Classe {c}",
        )

    if optimal_boundary_fn is not None:
        pad = 0.5
        xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200)
        ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200)
        xx, yy = np.meshgrid(xs, ys)
        try:
            zz = optimal_boundary_fn(xx, yy)
            ax.contour(xx, yy, zz, levels=[0.0], colors="black",
                       linewidths=1.8, linestyles="--")
            ax.plot([], [], color="black", linestyle="--", label="Fronteira ótima")
        except Exception:
            pass

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches="tight")
    plt.close()


def plot_llm_labels_per_problem(
    X: np.ndarray,
    y_llm: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    title: str = "Rotulação do LLM",
    feature_names: Tuple[str, str] = ("x1", "x2"),
    filename: Optional[str] = None,
) -> None:
    """Visualiza scatter ponto-a-ponto das rotulações do LLM.

    Item G do plano (e-mail orientador 22:06):
        "Mostrar graficamente todas as rotulações feitas pela LLM."

    Se y_true for fornecido, gera 3 painéis: LLM, real, erros (LLM ≠ real).
    """
    n_panels = 3 if y_true is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    cmap_class = {0: "tab:blue", 1: "tab:red"}

    # Painel 1: rotulação do LLM
    for c in [0, 1]:
        mask = y_llm == c
        axes[0].scatter(X[mask, 0], X[mask, 1], c=cmap_class[c], s=40,
                        edgecolor="black", linewidth=0.4, alpha=0.85,
                        label=f"LLM = {c}")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].set_title(f"Rotulação do LLM\n(n={len(X)})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if y_true is not None:
        # Painel 2: rótulo real (ground truth)
        for c in [0, 1]:
            mask = y_true == c
            axes[1].scatter(X[mask, 0], X[mask, 1], c=cmap_class[c], s=40,
                            edgecolor="black", linewidth=0.4, alpha=0.85,
                            label=f"Real = {c}")
        axes[1].set_xlabel(feature_names[0])
        axes[1].set_ylabel(feature_names[1])
        acc = accuracy_score(y_true, y_llm)
        axes[1].set_title(f"Rótulo Real\n(acurácia do LLM = {acc:.1%})")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Painel 3: erros (LLM ≠ real)
        errors = y_llm != y_true
        axes[2].scatter(X[~errors, 0], X[~errors, 1], c="lightgray", s=25,
                        edgecolor="none", alpha=0.6, label="LLM acertou")
        axes[2].scatter(X[errors, 0], X[errors, 1], c="black", s=60,
                        marker="x", linewidths=2, label="LLM errou")
        axes[2].set_xlabel(feature_names[0])
        axes[2].set_ylabel(feature_names[1])
        axes[2].set_title(f"Erros do LLM\n(n_erros = {int(errors.sum())} / {len(X)})")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches="tight")
    plt.close()


def summarize_cross_linearity(
    results_abc_r3: List[dict],
    external_results: List[dict],
    pasta_execucao: str,
    results_abc_r4: Optional[List[dict]] = None,
) -> Optional[str]:
    """Sintetiza ganhos 2→3→4 features em problemas LINEARES (A/B/C) vs NÃO-LINEARES.

    Síntese cruzada dos 3 blocos (slide 62 do roteiro atual). Gera CSV consolidado
    `final_cross_linearity.csv`: para cada (problema, n_features), reporta fidelidade
    e acurácia média. Permite tabela final A/B/C × meia-lua × peso×altura.
    """
    rows = []

    # Resultados sintéticos A/B/C — R3 (3 features, hipérbole)
    if results_abc_r3:
        for r in results_abc_r3:
            rows.append({
                'problem': 'A_B_C_lineares',
                'is_nonlinear': False,
                'algorithm': 'perceptron',
                'n_features': 3,
                'fidelity_vs_llm': r.get('accuracy'),
                'accuracy_vs_true': None,
                'seed': r.get('seed'),
            })

    # Resultados sintéticos A/B/C — R4 (4 features, elipse — sanity check)
    if results_abc_r4:
        for r in results_abc_r4:
            rows.append({
                'problem': 'A_B_C_lineares',
                'is_nonlinear': False,
                'algorithm': 'perceptron',
                'n_features': 4,
                'fidelity_vs_llm': r.get('accuracy'),
                'accuracy_vs_true': None,
                'seed': r.get('seed'),
            })

    # Resultados dos problemas externos não-lineares
    for r in external_results:
        rows.append({
            'problem': r.get('problem_name'),
            'is_nonlinear': True,
            'algorithm': 'perceptron',
            'n_features': r.get('n_features'),
            'fidelity_vs_llm': r.get('fidelity_perc_vs_llm'),
            'accuracy_vs_true': r.get('accuracy_perc_vs_true'),
            'seed': r.get('seed'),
        })
        rows.append({
            'problem': r.get('problem_name'),
            'is_nonlinear': True,
            'algorithm': 'nnls',
            'n_features': r.get('n_features'),
            'fidelity_vs_llm': r.get('fidelity_nnls_vs_llm'),
            'accuracy_vs_true': r.get('accuracy_nnls_vs_true'),
            'seed': r.get('seed'),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    filename = os.path.join(pasta_execucao, "final_cross_linearity.csv")
    df.to_csv(filename, index=False)
    return filename


# =============================================================================
# VISUALIZAÇÕES DOS PROBLEMAS EXTERNOS (peso × altura, meia-lua)
# Os 4 plots abaixo respondem aos pedidos do e-mail 22:04 (pontos 3-5):
#   - Curva de aprendizado da Fase E com a melhor métrica
#   - Comparação 2/3/4 features (resposta visual a "houve ganho?")
#   - Fronteira de decisão por n_features (projetada em R2)
#   - LLM vs Perceptron baseline (reunião ~2110s)
# =============================================================================

def plot_external_learning_curve(
    phase_e_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Curva de aprendizado (acurácia × n_shot) por (problema, variante)."""
    if not phase_e_results:
        return
    df = pd.DataFrame(phase_e_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    df['group'] = df['problem_name'].astype(str) + ' | ' + df['feature_names_str']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('tab10')

    for idx, metric_col in enumerate(['accuracy_llm_vs_metric', 'accuracy_llm_vs_true']):
        ax = axes[idx]
        for i, group in enumerate(sorted(df['group'].unique())):
            sub = df[df['group'] == group].sort_values('n_shot')
            stats = sub.groupby('n_shot')[metric_col].agg(['mean', 'std']).reset_index()
            ax.errorbar(
                stats['n_shot'], stats['mean'], yerr=stats['std'],
                marker='o', label=group, color=cmap(i % 10), capsize=3,
            )
        ax.set_xlabel('n_shot')
        ax.set_ylabel(metric_col.replace('_', ' '))
        ax.set_title('LLM vs métrica' if idx == 0 else 'LLM vs rótulo real', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Fase E (problemas externos) — curva de aprendizado',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def plot_external_features_comparison(
    phase_a_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Bar chart comparativo 2/3/4 features por (problema, variante).

    Responde diretamente ao "Verificar se houve ganho?" do e-mail 22:04 ponto 3.
    """
    if not phase_a_results:
        return
    df = pd.DataFrame(phase_a_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    df['group'] = df['problem_name'].astype(str) + ' | ' + df['feature_names_str']
    groups = sorted(df['group'].unique())
    n_features_list = sorted(df['n_features'].unique())

    metrics = [
        ('fidelity_perc_vs_llm', 'Fidelidade Perc (vs LLM)'),
        ('accuracy_perc_vs_true', 'Acurácia Perc (vs real)'),
        ('fidelity_nnls_vs_llm', 'Fidelidade NNLS (vs LLM)'),
        ('accuracy_nnls_vs_true', 'Acurácia NNLS (vs real)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    cmap = plt.get_cmap('tab10')

    for ax, (col, title) in zip(axes, metrics):
        bar_width = 0.8 / len(n_features_list)
        x = np.arange(len(groups))
        for j, nf in enumerate(n_features_list):
            means, stds = [], []
            for g in groups:
                sub = df[(df['group'] == g) & (df['n_features'] == nf)]
                means.append(sub[col].mean() if len(sub) else 0)
                stds.append(sub[col].std() if len(sub) > 1 else 0)
            offset = (j - (len(n_features_list) - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                   color=cmap(j), edgecolor='black', alpha=0.8,
                   label=f'{nf} features')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)

    fig.suptitle('Fase A externos — comparação 2 / 3 / 4 features\n(orientador 22:04 ponto 3: "Verificar se houve ganho")',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def plot_external_decision_boundary(
    phase_a_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Fronteira de decisão projetada em R2 para cada (problema, variante, n_feat).

    Para cada combinação (problema, variante), mostra um painel por n_features
    com os pontos e a fronteira d_W(x, c_0) = d_W(x, c_1) reconstruída no espaço
    de 2/3/4 features mas projetada de volta em R2 via grid em (x1, x2).
    """
    if not phase_a_results:
        return
    df = pd.DataFrame(phase_a_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )

    # Agrega: para cada (problema, variante) usa a primeira seed disponível
    groups = df.groupby(['problem_name', 'feature_names_str'])
    n_panels = len(groups)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 3, figsize=(15, 5 * n_panels), squeeze=False)
    for row_idx, ((problem_name, fnames), sub) in enumerate(groups):
        seed = sub['seed'].iloc[0]
        sub_seed = sub[sub['seed'] == seed].sort_values('n_features')
        for col_idx, n_feat in enumerate([2, 3, 4]):
            ax = axes[row_idx, col_idx]
            row = sub_seed[sub_seed['n_features'] == n_feat]
            if len(row) == 0:
                ax.set_title(f'{problem_name} | {fnames} | n_feat={n_feat}\n(não rodado)',
                             fontsize=9)
                ax.axis('off')
                continue
            r = row.iloc[0]
            X_aug = r['X_aug']
            y_llm = r['y_llm']
            centroids = r['centroids']
            w = r['w_perc']

            X2 = X_aug[:, :2]
            pad = 0.5
            xs = np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + pad, 120)
            ys = np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + pad, 120)
            xx, yy = np.meshgrid(xs, ys)
            grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
            grid_aug = augment_features(grid_xy, n_feat)
            zz = predict_with_metric(grid_aug, centroids, w).reshape(xx.shape)

            ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                        colors=['tab:blue', 'tab:red'], alpha=0.18)
            ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=1.4)
            for c in [0, 1]:
                mask = y_llm == c
                ax.scatter(X2[mask, 0], X2[mask, 1],
                           c='tab:blue' if c == 0 else 'tab:red',
                           s=18, edgecolor='black', linewidth=0.3, alpha=0.85)
            ax.set_title(
                f'{problem_name} | {fnames}\n'
                f'n_feat={n_feat} | seed={seed} | acc_real={r["accuracy_perc_vs_true"]:.1%}',
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)

    fig.suptitle('Fronteira de decisão (Perceptron) — projetada em R2',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def plot_phase_e_llm_vs_perceptron(
    phase_e_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Bar chart comparativo: LLM vs Perceptron baseline na Fase E externa.

    Responde à instrução da reunião (~2110s): treinar Perceptron sobre os mesmos
    exemplos do expert e comparar com o LLM in-context.
    """
    if not phase_e_results:
        return
    df = pd.DataFrame(phase_e_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    df['group'] = df['problem_name'].astype(str) + ' | ' + df['feature_names_str']
    groups = sorted(df['group'].unique())
    n_shots = sorted(df['n_shot'].unique())

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    for col_idx, group in enumerate(groups):
        ax = axes[0, col_idx]
        sub = df[df['group'] == group]
        means_llm, means_perc = [], []
        stds_llm, stds_perc = [], []
        for ns in n_shots:
            ss = sub[sub['n_shot'] == ns]
            means_llm.append(ss['accuracy_llm_vs_true'].mean())
            stds_llm.append(ss['accuracy_llm_vs_true'].std() if len(ss) > 1 else 0)
            pb = ss['accuracy_perceptron_baseline_vs_true'].dropna()
            means_perc.append(pb.mean() if len(pb) else np.nan)
            stds_perc.append(pb.std() if len(pb) > 1 else 0)
        x = np.arange(len(n_shots))
        w = 0.4
        ax.bar(x - w/2, means_llm, w, yerr=stds_llm, capsize=3,
               color='tab:orange', edgecolor='black', alpha=0.85, label='LLM')
        ax.bar(x + w/2, means_perc, w, yerr=stds_perc, capsize=3,
               color='tab:blue', edgecolor='black', alpha=0.85, label='Perceptron baseline')
        ax.set_xticks(x)
        ax.set_xticklabels([str(ns) for ns in n_shots])
        ax.set_xlabel('n_shot')
        ax.set_ylabel('Acurácia (vs rótulo real)')
        ax.set_ylim(0, 1.05)
        ax.set_title(group, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)

    fig.suptitle('Fase E externa — LLM in-context vs Perceptron treinado nos mesmos exemplos',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def main():
    global client, async_client, MODEL_NAME, CURRENT_PROVIDER, CURRENT_TEMPERATURE

    # ─────────────────────────────────────────────────────────────────────
    # CRIAÇÃO DA PASTA DE EXECUÇÃO E INÍCIO DO LOG
    # ─────────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pasta_execucao = f"execucao_{timestamp}"
    os.makedirs(pasta_execucao, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" EXPERIMENTO INICIADO: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f" Pasta de saída criada: {pasta_execucao}/")
    print(f" Todos os arquivos (imagens, CSVs e log) serão salvos nesta pasta.")
    print(f"{'='*70}\n")

    tee = Tee()
    sys.stdout = tee

    print_section("EXPERIMENTO: CONSISTÊNCIA DECISIONAL DE LLMs VIA OTIMIZAÇÃO INVERSA (v5.0)", "=")
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

    models_str = '\n    '.join([f"- {p}/{m} (temp={t})" for p, m, t in MODELS_TO_TEST])

    print(f"""
    ═══════════════════════════════════════════════════
    MODELS TO TEST ({len(MODELS_TO_TEST)}):
    {models_str}
    ═══════════════════════════════════════════════════

    FLAGS ATIVAS: {', '.join(flags_active)}

    PHASES A-C:
      Few-shot sizes: {FEW_SHOT_SIZES}
      Class name variations: {len(NOMES_CLASSES)}
      Repetitions: {N_REPETICOES}
      Seeds: {RANDOM_SEEDS}

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
    all_results_example_order = []  # Para viés de ordem dos exemplos
    all_results_baselines = []  # Para baselines clássicos (k-NN, LR, SVM)
    results_r3_2feat = []  # Fidelidade da métrica R2 (2 pesos) sobre rótulos do LLM
    results_r3_3feat = []  # Fidelidade da métrica R3 (3 pesos, x3=x1*x2) sobre os mesmos rótulos
    results_r3_4feat = []  # Fidelidade da métrica R4 (4 pesos, +x1², x2²) — sanity check elipse em A/B/C
    # Cache de dados da Fase A para gráficos de erros
    phase_a_data_for_plots = {}
    # Dados detalhados por seed para visualizações abrangentes
    seed_detailed_data = {}

    for model_idx, (provider, model_name, temperature) in enumerate(MODELS_TO_TEST):
        client = get_client(provider)
        async_client = get_async_client(provider)
        MODEL_NAME = model_name
        CURRENT_PROVIDER = provider
        CURRENT_TEMPERATURE = temperature

        for seed_idx, seed in enumerate(RANDOM_SEEDS):
            print_section(
                f"BLOCO 1 — LLM como FONTE | MODEL {model_idx + 1}/{len(MODELS_TO_TEST)}: "
                f"{provider}/{model_name} (temp={temperature}) | SEED {seed_idx + 1}/{len(RANDOM_SEEDS)}: {seed}",
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
                print(f"  Imagem salva: 01_all_three_problems.png")

                y_expert_e_viz = expert_classify(X_e, EXPERT_W, EXPERT_CENTROIDS)
                visualize_problem_e_with_expert(
                    X_e, y_e, y_expert_e_viz, EXPERT_W, EXPERT_CENTROIDS,
                    filename=os.path.join(pasta_execucao, "bloco2_01_problema_e_expert.png")
                )
                print(f"  Imagem salva: 06_problem_d_expert.png")

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
                    for rep in range(N_REPETICOES):
                        print(f"  [ABC] n_shot={n_shot}, rep={rep+1}/{N_REPETICOES}, seed={seed}", flush=True)
                        verbose = (rep == 0 and n_shot == FEW_SHOT_SIZES[0] and seed_idx == 0)
                        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail_bc = \
                            _run_phase_abc_experiment(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                n_shot, "A", "B", rep, provider, model_name,
                                temperature, seed, seed_idx, phase_a_cache, verbose
                            )
                        all_results_abc.append(result)

                        # Guarda dados para plots de erros da Fase A e visualizações detalhadas
                        if n_shot == 0 and rep == 0 and learned_metric is not None:
                            cache_key_ab = (seed, "A", "B", "default")
                            if cache_key_ab in phase_a_cache:
                                cached = phase_a_cache[cache_key_ab]
                                y_metric_a = predict_with_metric(X_train_a, cached['metric'].centroids, cached['metric'].w)
                                phase_a_data_for_plots[seed] = {
                                    'X': X_train_a, 'y_llm': cached['y_llm'],
                                    'y_metric': y_metric_a,
                                    'w': cached['metric'].w,
                                    'centroids': cached['metric'].centroids,
                                }
                                seed_detailed_data[seed] = {
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

                print(f"  ✓ Experimento principal concluído ({len(FEW_SHOT_SIZES)} n_shots × {N_REPETICOES} reps)", flush=True)

                # Variações de nomes de classe
                print(f"\n>>> Fases A-C: Variações de nomes de classe (10-shot)", flush=True)
                n_shot_fixed = 10
                for nome_0, nome_1 in NOMES_CLASSES[1:]:
                    for rep in range(N_REPETICOES):
                        print(f"  [ABC-Nomes] classes=({nome_0},{nome_1}), rep={rep+1}/{N_REPETICOES}, seed={seed}", flush=True)
                        verbose = (rep == 0 and seed_idx == 0)
                        result, *_ = _run_phase_abc_experiment(
                            X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                            n_shot_fixed, nome_0, nome_1, rep, provider, model_name,
                            temperature, seed, seed_idx, phase_a_cache, verbose
                        )
                        all_results_abc.append(result)

                print(f"  ✓ Variações de nomes de classe concluídas", flush=True)

                # ═══════════════════════════════════════════════════════
                # TESTE DE INVERSÃO DE ORDEM DAS CLASSES
                # ═══════════════════════════════════════════════════════
                if RUN_CLASS_ORDER_BIAS:
                    print(f"\n>>> Fases A-C: Teste de inversão de ordem das classes", flush=True)
                    for nome_0, nome_1 in NOMES_CLASSES_INVERTIDAS:
                        for rep in range(N_REPETICOES):
                            print(f"  [ABC-Inversão] classes=({nome_0},{nome_1}), rep={rep+1}/{N_REPETICOES}, seed={seed}", flush=True)
                            verbose = (rep == 0 and seed_idx == 0)
                            result, *_ = _run_phase_abc_experiment(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                n_shot_fixed, nome_0, nome_1, rep, provider, model_name,
                                temperature, seed, seed_idx, phase_a_cache, verbose
                            )
                            all_results_abc.append(result)
                    print(f"  ✓ Teste de inversão de ordem das classes concluído", flush=True)

                # ═══════════════════════════════════════════════════════
                # TESTE DE VARIANTES DE PROMPT (Sensibilidade ao prompt)
                # ═══════════════════════════════════════════════════════
                if RUN_PROMPT_VARIANTS:
                    print(f"\n>>> Fases A-C: Teste de variantes de prompt", flush=True)
                    for variant_name in PROMPT_VARIANTS:
                        if variant_name == "default":
                            continue  # Já testado no loop principal
                        for rep in range(N_REPETICOES):
                            print(f"  [ABC-Prompt] variant={variant_name}, rep={rep+1}/{N_REPETICOES}, seed={seed}", flush=True)
                            verbose = (rep == 0 and seed_idx == 0)
                            result, *_ = _run_phase_abc_experiment(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                0, "A", "B", rep, provider, model_name,
                                temperature, seed, seed_idx, phase_a_cache, verbose,
                                prompt_variant=variant_name
                            )
                            all_results_abc.append(result)
                    print(f"  ✓ Teste de variantes de prompt concluído", flush=True)

                # ═══════════════════════════════════════════════════════
                # TESTE DE NOMES SEMÂNTICOS NAS FEATURES
                # ═══════════════════════════════════════════════════════
                if RUN_FEATURE_NAMES:
                    print(f"\n>>> Fases A-C: Teste de nomes semânticos nas features", flush=True)
                    for feat_0, feat_1 in NOMES_FEATURES[1:]:  # Pula o neutro (já testado)
                        print(f"    Features: {feat_0}/{feat_1}")
                        # Teste simplificado: 1 seed, classe A/B, zero-shot apenas
                        if seed_idx == 0:
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
                                    eta=0.001, C=1.0, delta_gamma=0.05,  # Coelho et al. CILAMCE 2017, p. 16
                                    max_epochs=50, tol=1e-4, verbose=False,
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
                if RUN_ALGORITHM_COMPARISON:
                    print(f"\n>>> Comparação de Algoritmos: Perceptron vs NNLS", flush=True)
                    cache_key_ab = (seed, "A", "B", "default")
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
                            if seed in seed_detailed_data:
                                seed_detailed_data[seed]['w_nnls'] = w_alt
                                seed_detailed_data[seed]['centroids_nnls'] = centroids_alt

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

            if RUN_ORACLE_VALIDATION:
                print(f"\n>>> Validação do Oráculo: Algoritmos recuperam W conhecido?", flush=True)
                oracle_results = run_oracle_validation(
                    X_a, y_a, X_b, y_b, X_c, y_c, X_e, y_e,
                    EXPERT_CONFIGS,
                    random_seed=seed,
                    verbose=(seed_idx == 0),
                )
                all_results_oracle.extend(oracle_results)
                print(f"  ✓ Validação do oráculo concluída para seed={seed}", flush=True)

            # ═══════════════════════════════════════════════════════════════
            # FASE E: LLM COMO APRENDIZ
            # ═══════════════════════════════════════════════════════════════

            if RUN_PHASE_E:
                # Determina quais configs de expert usar
                expert_configs_to_run = EXPERT_CONFIGS if RUN_MULTIPLE_EXPERTS else [EXPERT_CONFIGS[0]]

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
                                base_results = all_results_e[-N_REPETICOES:]
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

                            for rep in range(N_REPETICOES):
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
                if RUN_CLASSICAL_BASELINES:
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
                if RUN_DILUTION:
                    print(f"\n>>> Experimento de Diluição: 3 hard fixos + N easy progressivos", flush=True)
                    y_expert_dilution = expert_classify(X_e, EXPERT_W, EXPERT_CENTROIDS)
                    n_hard_fixed = 4  # 2 por classe (arredondado para par)
                    easy_additions = [0, 2, 4, 10, 16, 20]  # N easy adicionados

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

                        for rep in range(N_REPETICOES):
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

            if RUN_EXAMPLE_ORDER_BIAS and RUN_PHASE_E:
                print(f"\n>>> Experimento de Viés de Ordem dos Exemplos Few-Shot", flush=True)
                y_expert_order = expert_classify(X_e, EXPERT_W, EXPERT_CENTROIDS)
                expert_acc_order = accuracy_score(y_e, y_expert_order)

                # Usa n_shot=10 e estratégia "mixed" como configuração fixa
                # para isolar o efeito da ordenação
                ORDER_TEST_N_SHOTS = [5, 10, 20]

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

                        for rep in range(N_REPETICOES):
                            t_rep_start = time.time()
                            print(f"    [Ordem {ordering}] rep {rep+1}/{N_REPETICOES} (n_test={len(X_test_order)})...", flush=True)
                            y_llm_order, n_malf_order = collect_llm_decisions(
                                X_test_order, "A", "B",
                                examples=reordered,
                                verbose=(rep == 0 and seed_idx == 0 and ordering == EXAMPLE_ORDERINGS[0] and n_shot_order == ORDER_TEST_N_SHOTS[0]),
                                label_prefix=f"[Ordem {ordering}] "
                            )
                            cons_order = compute_consistency_metrics(y_llm_order, y_expert_test_order)
                            llm_acc_order = accuracy_score(y_gt_test_order, y_llm_order)
                            print(
                                f"    [Ordem {ordering}] rep {rep+1}/{N_REPETICOES} concluída em "
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
                cache_key_r3 = (seed, "A", "B", "default")
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
                            eta=0.001, C=1.0, delta_gamma=0.05,  # Coelho et al. CILAMCE 2017, p. 16
                            max_epochs=50, tol=1e-4, verbose=(seed_idx == 0),
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
                        })
                        results_r3_2feat.append({'accuracy': fid_r2, 'seed': seed})

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
                            eta=0.001, C=1.0, delta_gamma=0.05,  # Coelho et al. CILAMCE 2017, p. 16
                            max_epochs=50, tol=1e-4, verbose=False,
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
            X_hm, y_hm = create_problem_homem_mulher()
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
            print(f"  Gráfico salvo: 25_homem_mulher_overview.png")

            hm_pipeline = run_external_problem_pipeline(
                problem_name="homem_mulher",
                X=X_hm, y_true=y_hm,
                nome_classe_0="Homem", nome_classe_1="Mulher",
                feature_variants=[("x1", "x2"), ("peso", "altura")],
                seeds=RANDOM_SEEDS,
                n_shots_phase_e=FEW_SHOT_SIZES_PHASE_E,
                pasta_execucao=pasta_execucao,
                n_train_ratio=0.7,
                verbose=True,
            )
            external_phase_a_results.extend(hm_pipeline['phase_a_results'])
            external_phase_e_results.extend(hm_pipeline['phase_e_results'])
            external_llm_label_maps.update(hm_pipeline['llm_label_maps'])
            print("  ✓ Pipeline peso × altura concluído.", flush=True)
        except FileNotFoundError as exc:
            print(f"  ⚠ Base peso × altura não encontrada: {exc}")
        except Exception as exc:
            print(f"  ⚠ Erro no pipeline peso × altura: {exc}")
            traceback.print_exc()

    if RUN_PROBLEM_MEIALUA:
        try:
            print_section("BLOCO 1/2 — PROBLEMA D/F: MEIA-LUA (não-linear sintético)", "═")
            for seed in RANDOM_SEEDS:
                X_ml, y_ml = create_problem_d_meialua(n_samples=N_SAMPLES_PROBLEM_A, random_state=seed)
                print(f"  Meia-lua seed={seed}: shape={X_ml.shape}")

                # Visualização do problema antes de qualquer coleta LLM
                plot_problem_overview(
                    X_ml, y_ml,
                    title=f"Problema E — Meia-lua (sklearn.make_moons, seed={seed})",
                    feature_names=("x1", "x2"),
                    optimal_boundary_fn=None,
                    filename=os.path.join(pasta_execucao, f"bloco1_02_problema_d_meialua_seed{seed}_overview.png"),
                )
                print(f"  Gráfico salvo: 26_meia_lua_seed{seed}_overview.png")

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
                external_phase_a_results.extend(ml_pipeline['phase_a_results'])
                external_phase_e_results.extend(ml_pipeline['phase_e_results'])
                external_llm_label_maps.update(ml_pipeline['llm_label_maps'])
            print("  ✓ Pipeline meia-lua concluído.", flush=True)
        except Exception as exc:
            print(f"  ⚠ Erro no pipeline meia-lua: {exc}")
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZAÇÕES FINAIS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n[PASSO 5] Gerando visualizações finais...", flush=True)
    print_section("BLOCO 1 — VISUALIZAÇÕES (Fases A/B/C lineares + meia-lua)", "═")

    if all_results_abc:
        plot_consistency_comparison_extended(all_results_abc, filename=os.path.join(pasta_execucao, "bloco1_09_consistency_extended.png"))
        print(f"  Gráfico salvo: 02_consistency_extended.png")
        plot_class_names_effect(all_results_abc, filename=os.path.join(pasta_execucao, "bloco1_14a_class_names_effect.png"))
        print(f"  Gráfico salvo: 03_class_names_effect.png")

        if len(MODELS_TO_TEST) > 1:
            plot_model_comparison(all_results_abc, filename=os.path.join(pasta_execucao, "final_09_model_comparison.png"))
            print(f"  Gráfico salvo: 04_model_comparison.png")
        if len(RANDOM_SEEDS) > 1:
            plot_seed_comparison(all_results_abc, filename=os.path.join(pasta_execucao, "bloco1_08_seed_comparison.png"))
            print(f"  Gráfico salvo: 05_seed_comparison.png")

        # Distribuição de W
        plot_w_distribution(all_results_abc, filename=os.path.join(pasta_execucao, "bloco1_06_w_distribution.png"))
        print(f"  Gráfico salvo: 10_w_distribution.png")

        # Erros da métrica na Fase A
        for seed_key, data in phase_a_data_for_plots.items():
            fname = os.path.join(pasta_execucao, f"bloco1_05_fase_a_errors_seed{seed_key}.png")
            plot_metric_errors_phase_a(
                data['X'], data['y_llm'], data['y_metric'],
                data['w'], data['centroids'], filename=fname
            )
            print(f"  Gráfico salvo: 11_metric_errors_phase_a_seed{seed_key}.png")

        # Análise quantitativa de erros por região (complementa gráfico 11)
        if phase_a_data_for_plots:
            print_error_analysis_by_region(phase_a_data_for_plots)

        # Análise de sensibilidade dos hiperparâmetros do Perceptron
        if phase_a_data_for_plots:
            print_hyperparameter_sensitivity(phase_a_data_for_plots)

        # Viés de ordem das classes
        if RUN_CLASS_ORDER_BIAS:
            plot_class_order_bias(all_results_abc, filename=os.path.join(pasta_execucao, "bloco1_12_class_order_bias.png"))
            print(f"  Gráfico salvo: 15_class_order_bias.png")

        # Efeito de nomes de features
        if RUN_FEATURE_NAMES:
            plot_feature_names_effect(all_results_abc, filename=os.path.join(pasta_execucao, "bloco1_14b_feature_names_effect.png"))
            print(f"  Gráfico salvo: 19_feature_names_effect.png")

        # Variantes de prompt
        if RUN_PROMPT_VARIANTS:
            plot_prompt_variant_comparison(all_results_abc, filename=os.path.join(pasta_execucao, "bloco1_13_prompt_variants.png"))
            print(f"  Gráfico salvo: 17_prompt_variant_comparison.png")

    print_section("BLOCO 2 — VISUALIZAÇÕES (Fase E, LLM como aprendiz)", "═")

    if all_results_e:
        plot_phase_e_learning_curve(all_results_e, filename=os.path.join(pasta_execucao, "bloco2_04_phase_e_learning_curve.png"))
        print(f"  Gráfico salvo: bloco2_04_phase_e_learning_curve.png")
        plot_phase_e_strategy_comparison(all_results_e, filename=os.path.join(pasta_execucao, "bloco2_05_phase_e_strategy_comparison.png"))
        print(f"  Gráfico salvo: bloco2_05_phase_e_strategy_comparison.png")

    # Baselines clássicos
    if all_results_baselines and all_results_e:
        plot_classical_baselines_comparison(
            all_results_e, all_results_baselines,
            filename=os.path.join(pasta_execucao, "bloco2_09_classical_baselines.png")
        )
        print(f"  Gráfico salvo: 18_classical_baselines.png")

    # Experimento de diluição
    if all_results_dilution:
        plot_dilution_experiment(all_results_dilution, filename=os.path.join(pasta_execucao, "bloco2_06_dilution.png"))
        print(f"  Gráfico salvo: 12_dilution_experiment.png")

    # Viés de ordem dos exemplos few-shot
    if all_results_example_order:
        plot_example_order_bias(all_results_example_order, filename=os.path.join(pasta_execucao, "bloco2_07_example_order.png"))
        print(f"  Gráfico salvo: 16_example_order_bias.png")

        # Análise quantitativa do viés de ordem (recency bias)
        print_example_order_analysis(all_results_example_order)

    # Não-linearidade implícita: fidelidade R2 (2 pesos) vs R3 (3 pesos)
    if results_r3_2feat and results_r3_3feat:
        plot_r3_comparison(results_r3_2feat, results_r3_3feat,
                          filename=os.path.join(pasta_execucao, "bloco1_10_r3r4_comparison.png"))
        print(f"  Gráfico salvo: 13_r3_vs_r2.png")

    # Comparação de algoritmos — Perceptron × NNLS
    if all_results_abc_alternative and all_results_abc:
        perc_results = [r for r in all_results_abc
                       if r.n_shot == 0 and r.nomes_classes == ("A", "B")]
        if perc_results:
            plot_algorithm_comparison(perc_results, all_results_abc_alternative,
                                    filename=os.path.join(pasta_execucao, "bloco1_07_algorithm_comparison.png"))
            print(f"  Gráfico salvo: 14_algorithm_comparison.png")

    # NOVO: Validação do oráculo
    if all_results_oracle:
        plot_oracle_w_recovery(all_results_oracle,
            filename=os.path.join(pasta_execucao, "bloco1_03_oracle_w_recovery.png"))
        print(f"  Gráfico salvo: bloco1_03_oracle_w_recovery.png")
        plot_oracle_transfer(all_results_oracle,
            filename=os.path.join(pasta_execucao, "bloco1_04_oracle_transfer.png"))
        print(f"  Gráfico salvo: bloco1_04_oracle_transfer.png")

    # Diagnóstico da busca binária em γ (item b reunião 30/04/2026, ~520s)
    if PERCEPTRON_GAMMA_DIAGNOSTICS:
        plot_gamma_convergence(
            PERCEPTRON_GAMMA_DIAGNOSTICS,
            filename=os.path.join(pasta_execucao, "final_10_gamma_convergence.png"),
        )

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZAÇÕES DETALHADAS POR SEED
    # ═══════════════════════════════════════════════════════════════════

    print_section("FECHAMENTO — VISUALIZAÇÕES DETALHADAS POR SEED", "═")
    for seed_key, sdata in seed_detailed_data.items():
        print(f"\n  Gerando visualizações detalhadas para seed {seed_key}...")

        plot_dataset_overview(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"final_04_dataset_overview_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 16_dataset_overview_seed{seed_key}.png")

        plot_hits_and_errors(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"final_05_hits_errors_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 17_hits_errors_seed{seed_key}.png")

        plot_w_comparison_algorithms(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"final_06_w_algorithms_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 18_w_algorithms_seed{seed_key}.png")

        plot_confusion_matrices_detailed(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"final_02_confusion_matrices_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 20_confusion_matrices_seed{seed_key}.png")

        plot_margin_analysis_detailed(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"final_07_margin_analysis_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 21_margin_analysis_seed{seed_key}.png")

        plot_experiment_summary_dashboard(sdata, seed_key, all_results_abc, all_results_e,
            filename=os.path.join(pasta_execucao, f"final_03_dashboard_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 19_experiment_dashboard_seed{seed_key}.png")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    if all_results_abc:
        print_final_analysis(all_results_abc)
    if all_results_e:
        print_phase_e_analysis(all_results_e)

    # Sumário estatístico com Bootstrap CI, Wilcoxon e Cohen's d
    print_statistical_summary(all_results_abc, all_results_e)

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
                'seed': r['seed'], 'n_features': 3, 'algorithm': 'perceptron',
                'fidelidade': r['accuracy'],
                'w0': w[0], 'w1': w[1],
                'w2': w[2] if len(w) > 2 else np.nan,
                'w3': np.nan,
            })
            r3_rows.append({
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
                'seed': r['seed'], 'n_features': 4, 'algorithm': 'perceptron',
                'fidelidade': r['accuracy'],
                'w0': w[0], 'w1': w[1],
                'w2': w[2] if len(w) > 2 else np.nan,
                'w3': w[3] if len(w) > 3 else np.nan,
            })
            r3_rows.append({
                'seed': r['seed'], 'n_features': 4, 'algorithm': 'nnls',
                'fidelidade': r.get('accuracy_nnls', np.nan),
                'w0': w_nnls[0], 'w1': w_nnls[1],
                'w2': w_nnls[2] if len(w_nnls) > 2 else np.nan,
                'w3': w_nnls[3] if len(w_nnls) > 3 else np.nan,
            })
        for r in results_r3_2feat:
            r3_rows.append({
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

    # ─── Problemas externos não-lineares (peso×altura, meia-lua) ──────────
    if external_phase_a_results:
        rows_a = []
        for r in external_phase_a_results:
            rows_a.append({
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
                'gamma_perc': r.get('gamma_perc'),
                'n_malformed': r.get('n_malformed'),
            })
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

    # Tabela cruzada linear × não-linear
    cross_fname = summarize_cross_linearity(
        results_abc_r3=results_r3_3feat if results_r3_3feat else [],
        external_results=external_phase_a_results,
        pasta_execucao=pasta_execucao,
        results_abc_r4=results_r3_4feat if results_r3_4feat else None,
    )
    if cross_fname:
        print(f"  Comparação cruzada linear×não-linear salva em: {cross_fname}")

    # Visualização ponto-a-ponto das rotulações do LLM (item G, e-mail 22:06)
    if external_llm_label_maps:
        print(f"\n  Gerando scatter ponto-a-ponto das rotulações do LLM...")
        for key, data in external_llm_label_maps.items():
            seed_val, feat_0, feat_1, kind = key
            kind_label = f"n_shot={kind}" if isinstance(kind, int) else str(kind)
            fname_lbl = os.path.join(
                pasta_execucao,
                f"final_08_llm_labels_seed{seed_val}_{feat_0}_{feat_1}_{kind}.png",
            )
            try:
                plot_llm_labels_per_problem(
                    X=data['X'], y_llm=data['y_llm'], y_true=data.get('y_true'),
                    title=f"Rotulação LLM — seed={seed_val} | {feat_0}/{feat_1} | {kind_label}",
                    feature_names=(feat_0, feat_1),
                    filename=fname_lbl,
                )
            except Exception as exc:
                print(f"    ⚠ Falha em {fname_lbl}: {exc}")
        print(f"  ✓ Scatter ponto-a-ponto gerados em {pasta_execucao}/final_08_llm_labels_*.png")

    # ─── Plots adicionais para apresentação (problemas externos) ──────────
    if external_phase_e_results:
        try:
            plot_external_learning_curve(
                external_phase_e_results,
                filename=os.path.join(pasta_execucao, "bloco23_external_learning_curve.png"),
            )
            print(f"  Gráfico salvo: 27_external_learning_curve.png")
        except Exception as exc:
            print(f"  ⚠ Falha em 27_external_learning_curve.png: {exc}")

        try:
            plot_phase_e_llm_vs_perceptron(
                external_phase_e_results,
                filename=os.path.join(pasta_execucao, "bloco23_external_llm_vs_perceptron.png"),
            )
            print(f"  Gráfico salvo: 30_external_llm_vs_perceptron.png")
        except Exception as exc:
            print(f"  ⚠ Falha em 30_external_llm_vs_perceptron.png: {exc}")

    if external_phase_a_results:
        try:
            plot_external_features_comparison(
                external_phase_a_results,
                filename=os.path.join(pasta_execucao, "bloco23_external_features_comparison.png"),
            )
            print(f"  Gráfico salvo: 28_external_features_comparison.png")
        except Exception as exc:
            print(f"  ⚠ Falha em 28_external_features_comparison.png: {exc}")

        try:
            plot_external_decision_boundary(
                external_phase_a_results,
                filename=os.path.join(pasta_execucao, "bloco23_external_decision_boundary.png"),
            )
            print(f"  Gráfico salvo: 29_external_decision_boundary.png")
        except Exception as exc:
            print(f"  ⚠ Falha em 29_external_decision_boundary.png: {exc}")

        # ─── Resumo consolidado no log (auxilia roteiro da apresentação) ──────
        print_section("BLOCO 2/3 — RESUMO CONSOLIDADO: Pipeline externos (Fase A)", "═")
        df_ext = pd.DataFrame(external_phase_a_results)
        df_ext['variant'] = df_ext['feature_names'].apply(
            lambda t: '/'.join(t) if isinstance(t, tuple) else str(t)
        )
        for problem in sorted(df_ext['problem_name'].unique()):
            sub = df_ext[df_ext['problem_name'] == problem]
            for variant in sorted(sub['variant'].unique()):
                sv = sub[sub['variant'] == variant]
                print(f"\n  {problem} | {variant}:")
                for nf in sorted(sv['n_features'].unique()):
                    ss = sv[sv['n_features'] == nf]
                    print(
                        f"    n_features={nf}: "
                        f"fid_perc={ss['fidelity_perc_vs_llm'].mean():.1%}±{ss['fidelity_perc_vs_llm'].std():.1%} | "
                        f"acc_perc_real={ss['accuracy_perc_vs_true'].mean():.1%}±{ss['accuracy_perc_vs_true'].std():.1%} | "
                        f"llm_real={ss['llm_accuracy_vs_true'].mean():.1%}"
                    )
        best_overall = df_ext.loc[df_ext['accuracy_perc_vs_true'].idxmax()]
        print(
            f"\n  ★ Melhor configuração geral: {best_overall['problem_name']} | "
            f"{best_overall['variant']} | n_features={best_overall['n_features']} | "
            f"acc_real={best_overall['accuracy_perc_vs_true']:.1%}"
        )

    if external_phase_e_results:
        print_section("BLOCO 2/3 — RESUMO CONSOLIDADO: Fase E externa (LLM vs Perceptron)", "═")
        df_d = pd.DataFrame(external_phase_e_results)
        df_d['variant'] = df_d['feature_names'].apply(
            lambda t: '/'.join(t) if isinstance(t, tuple) else str(t)
        )
        for problem in sorted(df_d['problem_name'].unique()):
            sub = df_d[df_d['problem_name'] == problem]
            for variant in sorted(sub['variant'].unique()):
                sv = sub[sub['variant'] == variant]
                print(f"\n  {problem} | {variant}:")
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
    interactions_path = os.path.join(pasta_execucao, "llm_interactions.json")
    with open(interactions_path, 'w', encoding='utf-8') as f:
        json.dump(LLM_INTERACTIONS, f, ensure_ascii=False, indent=2)
    print(f"  Interações LLM salvas em: {interactions_path} ({len(LLM_INTERACTIONS)} chamadas)")

    # ─── Passo 7: Salvar log TXT ──────────
    print(f"\n[PASSO 7] Salvando log completo da execução...")
    sys.stdout = tee._stdout
    log_path = os.path.join(pasta_execucao, "log_execucao.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(tee.getvalue())

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