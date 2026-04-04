import os
from dotenv import load_dotenv
load_dotenv()


"""
EXPLICANDO DECISÕES DE LLMs VIA OTIMIZAÇÃO INVERSA
EXPERIMENTO: CONSISTÊNCIA DECISIONAL DE LLMs ENTRE PROBLEMAS DISTINTOS

=====================================================
CONTEXTO E MOTIVAÇÃO (ORIENTADOR: Prof. Raul)
=====================================================

A questão central da dissertação é: um LLM possui um processo decisório ESTÁVEL
e APRENDÍVEL? Quando classificamos pontos em um problema, o LLM utiliza algum
critério implícito. Esse critério pode ser capturado matematicamente via
otimização inversa (Problema Inverso). Se o critério for estável, aplicá-lo a
outros problemas com distribuições diferentes deverá produzir classificações
concordantes com as que o LLM faria por conta própria.

Matematicamente, isso equivale a mostrar que o LLM está "associando da mesma
maneira" entre problemas distintos — o que constitui evidência de CONSTÂNCIA
DECISIONAL mensurável, não apenas observável.

Referência do algoritmo de aprendizado: Perceptron Estruturado com Relaxação
de Margem (indicado pelo orientador na Reunião 1 para aprender a métrica W).

=====================================================
PROBLEMA DIRETO vs. PROBLEMA INVERSO
=====================================================

PROBLEMA DIRETO (ponto de partida):
  Dado um classificador com critério CONHECIDO, classificar novos pontos.
  → O LLM faz isso naturalmente (zero-shot ou few-shot).

PROBLEMA INVERSO (foco do experimento):
  Dadas as CLASSIFICAÇÕES de um especialista (o LLM), INFERIR qual critério
  (métrica W) ele utilizou implicitamente.
  → Usamos o Perceptron Estruturado para aprender W a partir das decisões do LLM.
  → Depois aplicamos W a novos problemas para ver se o LLM se repete.

NOTA DO ORIENTADOR (Reunião 1): "A gente pega outros exemplos, aplica a nossa
métrica, porque Mahalanobis não vai conseguir aprender totalmente. Vai ter que
ter uma matriz semidefinida, mas diferente, parametrizada."
→ A versão atual usa Mahalanobis DIAGONAL (restrição: O(d) parâmetros em vez
  de O(d²)). Isso é uma simplificação deliberada para o experimento inicial.
  PRÓXIMO PASSO: evoluir para uma métrica semidefinida positiva parametrizada
  mais geral, conforme indicado pelo orientador.

=====================================================
HIPÓTESES DO EXPERIMENTO
=====================================================

H1 (Consistência): O LLM mantém o mesmo critério decisório implícito quando
  aplicado a problemas com distribuições diferentes. A métrica Ŵ_LLM estimada
  a partir das decisões do LLM no Problema A prevê as classificações do LLM
  nos Problemas B e C com alta concordância (Kappa > 0.7).

H2 (Few-shot amplifica consistência): Fornecer ao LLM exemplos rotulados pela
  própria métrica W (ancoragem) aumenta a concordância LLM–métrica, pois
  direciona o LLM a seguir explicitamente o critério aprendido.

H3 (LLM como aprendiz — Fase D): O LLM consegue APRENDER o critério de um
  perito externo a partir de exemplos few-shot. A concordância LLM–perito
  cresce com o número de exemplos fornecidos.
  Sub-hipótese (do orientador, Reunião 3): "O zero-shot não vai acertar —
  ela não vai adivinhar o que você rotulou." → Esperamos concordância próxima
  ao acaso no zero-shot e melhora sistemática com mais exemplos.

H4 (Exemplos difíceis são mais informativos): Exemplos próximos à fronteira
  de decisão ("hard") são mais informativos para o LLM aprender o critério
  do perito. A estratégia "hard" supera "easy" e "random" em acurácia,
  especialmente com poucos exemplos few-shot.

H5 (Estabilidade): O comportamento do LLM é reproduzível entre execuções
  (verificado via múltiplas sementes e repetições).

=====================================================
METODOLOGIA — FASES DO EXPERIMENTO
=====================================================

FASE A — APRENDIZADO DA MÉTRICA (Problema A) [APENAS ZERO-SHOT]
---------------------------------------------------------------
Objetivo: capturar o critério decisório INTRÍNSECO do LLM sem influência
  de exemplos externos.

1. Gerar conjunto sintético (Problema A): duas gaussianas bem separadas
   no eixo x1 (separação horizontal clara)
2. LLM classifica pontos de treino em modo ZERO-SHOT
   → O LLM age como "perito intrínseco": não recebe nenhum exemplo
   → Capturamos seu critério natural, antes de qualquer ancoragem
3. Calcular centróides de cada classe a partir das decisões do LLM
4. Aprender métrica W com Perceptron Estruturado com Relaxação de Margem
   → W é uma matriz diagonal (semidefinida positiva restrita)
   → Busca binária no gamma ótimo para maximizar margem da fronteira
5. A métrica W_A é aprendida UMA ÚNICA VEZ e reutilizada nas Fases B e C

JUSTIFICATIVA DO ZERO-SHOT NA FASE A (Reunião 1):
- Queremos capturar o processo decisório INTRÍNSECO do LLM
- Few-shot enviesa o LLM em direção ao ground truth, não ao seu critério natural
- A métrica deve representar como o LLM classifica espontaneamente
- Isso garante que W reflita os critérios do próprio LLM

FASE B — TESTE DE CONSISTÊNCIA (Problema B)
------------------------------------------------------------
Objetivo: verificar se o LLM mantém o mesmo raciocínio em um problema com
  geometria DIFERENTE (centróides deslocados verticalmente).

1. Gerar Problema B: duas gaussianas com parâmetros DISTINTOS do Problema A
   → Mudança deliberada nos parâmetros das gaussianas (orientador, Reunião 2):
     "muda muito os parâmetros das gaussianas ali um pouco"
2. Aplicar W_A (com centróides do Problema A) para classificar Problema B
3. LLM classifica os mesmos pontos do Problema B (zero-shot ou few-shot)
4. CONSISTÊNCIA = concordância entre LLM e W_A

REGRA CRÍTICA DE FEW-SHOT (orientador, Reunião 2):
  "No fio-shot, você tem que usar os rótulos que a métrica definiu.
   Não é o que você gerou das gaussianas."
  → Os exemplos few-shot DEVEM usar rótulos preditos por W_A, NÃO o ground truth
  → Isso ancora o LLM ao critério aprendido, não ao padrão gerador dos dados
  → Seleção por Aprendizado Ativo: exemplos com maior margem (mais representativos)
    Margem = d_W(x, centróide_errado) - d_W(x, centróide_previsto)

FASE C — TESTE DE CONSISTÊNCIA ADICIONAL (Problema C)
-----------------------------------------------------
Objetivo: replicar o teste de consistência em uma terceira distribuição,
  com distorção geométrica ainda mais acentuada (maior separação diagonal).

1. Gerar Problema C: parâmetros mais distantes do Problema A que o B
2. Aplicar a mesma W_A do Problema A
3. Verificar se a consistência se mantém nesta distribuição mais difícil

JUSTIFICATIVA (orientador, Reunião 2): "Se a coisa permanecer [nos 2-3 problemas],
  já tem uma boa ideia de um artigo, que mostra que a linguagem consegue manter
  uma constância."

FASE D — LLM COMO APRENDIZ (v3.0)
-----------------------------------------
Objetivo: verificar se o LLM consegue EVOLUIR e aprender o critério de um
  perito externo a partir de exemplos few-shot (Reunião 3: "Agora eu quero
  saber se ela sabe evoluir, se ela sabe aprender a métrica do especialista").

Inversão de papéis: um PERITO EXTERNO com métrica W_expert CONHECIDA rotula
  os dados. O LLM deve aprender esse padrão a partir dos exemplos fornecidos.

1. Definir perito com W_expert conhecida (anisotrópica: w2 >> w1)
2. Perito rotula Problema D com sua métrica
3. Selecionar exemplos few-shot por estratégia de dificuldade:
   - "easy":   pontos LONGE da fronteira (margem alta) — exemplos óbvios
   - "hard":   pontos PRÓXIMOS à fronteira (margem baixa) — exemplos ambíguos
   - "mixed":  metade fáceis + metade difíceis — representação balanceada
   - "random": seleção aleatória — linha de base sem critério
   NOTA (orientador, Reunião 3): "Você não pode pegar exemplos muito claros,
   porque senão ela não aprende. Faz algumas variações."
   → O experimento testa TODAS as estratégias para comparação direta.
4. Fornecer exemplos ao LLM com rótulos do perito
5. LLM classifica pontos de teste (excluídos dos exemplos)
6. Medir concordância LLM–perito em função de n_shot e da estratégia
7. ESPERADO: zero-shot próximo ao acaso; crescimento sistemático com mais exemplos

=====================================================
DESIGN DE VARIABILIDADE (orientador, Reunião 1)
=====================================================

Para garantir que os resultados não sejam artefatos de uma execução específica:

- N_REPETICOES: múltiplas repetições por configuração na MESMA base de dados
  → "fazer em cima de uma mesma base pra saber se há variabilidade"
- RANDOM_SEEDS: múltiplas sementes para BASES DE DADOS DIFERENTES
  → "pode fazer em bases diferentes pra criar um padrão de aprendizado da métrica"
- NOMES_CLASSES: testar com diferentes nomes ("A/B", "0/1", "Positivo/Negativo",
  "Azul/Vermelho") para detectar viés semântico do LLM
  → "em vez de colocar o nome da classe AB, colocar outros nomes"
- Temperatura 0.0: reduzir estocasticidade do LLM para isolar o critério de classificação
  → "tirar o menos possível de estocasticidade do processo"

=====================================================
ESTRATÉGIA DE COMPARAÇÃO ENTRE MODELOS (planejado)
=====================================================

Atualmente: apenas GPT-4o-mini (OpenAI) em produção.

PRÓXIMO PASSO (orientador, Reunião 1): "depois, quando tiver tudo certinho,
  a gente pode fazer com outras — tipo, Claude (Anthropic), Gemini (Google)."
→ O código já suporta múltiplos provedores via MODELS_TO_TEST.
→ Objetivo: verificar se a consistência decisional é propriedade geral dos LLMs
  ou específica de um modelo/treinamento.

=====================================================
LIMITAÇÕES CONHECIDAS E PRÓXIMOS PASSOS
=====================================================

1. DADOS SINTÉTICOS 2D:
   Limitação: dados sintéticos em 2 dimensões são mais simples do que problemas reais.
   → Facilita visualização mas limita generalização dos achados.
   

3. CENTRÓIDES CROSS-PROBLEM:
   A métrica W_A usa os centróides do Problema A para classificar Problemas B/C.
   Isso cria um confound: "inconsistência" pode refletir desalinhamento geométrico
   entre problemas, não apenas variabilidade do LLM.

   Pergunta para orientador: transferir so o W, ou transferir W + centroides juntos?

   
4. LEVANTAMENTO DE TRABALHOS RELACIONADOS (próximo passo, Reunião 3):
   "Procurar artigos para embasar mais o que estou fazendo. Preferencialmente
   revisados por pares — periódicos e conferências."
   → Área com poucas publicações até o momento; relevante para situar a contribuição.

=====================================================
MELHORIAS DA VERSÃO 3.0
=====================================================
- Adicionada Fase D: LLM como Aprendiz (perito ensina LLM)
- Adicionadas estratégias de dificuldade de exemplos (easy/hard/mixed/random)
- Adicionado Problema C para validação adicional de consistência
- Tamanhos few-shot mais granulares na Fase D: [0, 3, 5, 10, 20]
- Adicionados Kappa de Cohen, F1-Score e análise detalhada de discordâncias
- Visualização aprimorada da fronteira de decisão (todos os 3 problemas)
- Tratamento robusto de respostas do LLM com retentativas para respostas malformadas
- Pasta por execução com log TXT de toda a saída
=====================================================

PROVEDORES SUPORTADOS: OpenAI, Gemini, Claude (Anthropic)
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
from collections import defaultdict
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

from sklearn.datasets import make_blobs
from sklearn.metrics import (
    accuracy_score, confusion_matrix, cohen_kappa_score,
    f1_score, precision_score, recall_score, classification_report
)
import seaborn as sns

from openai import OpenAI, AsyncOpenAI
import anthropic
import time
import asyncio

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
    """Cria o cliente assíncrono para chamadas concorrentes (apenas OpenAI/Gemini)."""
    config = PROVIDER_CONFIG[provider]
    if config["client_type"] == "anthropic":
        return None
    if config["base_url"]:
        return AsyncOpenAI(
            api_key=os.getenv(config["api_key_env"]),
            base_url=config["base_url"]
        )
    else:
        return AsyncOpenAI(api_key=os.getenv(config["api_key_env"]))


# Cliente global — será definido para cada modelo durante os experimentos
client = None
async_client = None  # Cliente assíncrono para chamadas concorrentes
MODEL_NAME = None
CURRENT_PROVIDER = None
CURRENT_TEMPERATURE = 0.0

# Log de todas as interações com a LLM (prompt, resposta bruta, parsing)
LLM_INTERACTIONS = []

# Fases A e D usam mais amostras porque envolvem aprendizado (A: treino do Perceptron, D: LLM aprendendo do expert).
# Fases B e C são apenas testes de consistência da métrica já aprendida — 100 pontos bastam para Kappa/F1 confiáveis.
N_SAMPLES_PROBLEM_A = 150
N_SAMPLES_PROBLEM_B = 100
N_SAMPLES_PROBLEM_C = 100
N_SAMPLES_PROBLEM_D = 150
FEW_SHOT_SIZES = [0, 5, 10]
FEW_SHOT_SIZES_PHASE_D = [0, 3, 5, 10, 20]  # NEW: More granular for Phase D
N_REPETICOES = 2

# Múltiplas sementes aleatórias para garantir robustez dos resultados
RANDOM_SEEDS = [42]
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

# Parâmetros do Problema B (instância anisotrópica — centróides deslocados verticalmente)
PROBLEM_B_CENTERS = [(-2.0, 0.8), (2.0, -0.8)]
PROBLEM_B_STD = 1.2

# Parâmetros do Problema C (distorção geométrica mais acentuada — teste de generalização mais severo)
PROBLEM_C_CENTERS = [(-2.0, 1.5), (2.0, -1.5)]
PROBLEM_C_STD = 1.2

# Parâmetros do Problema D (Fase D — LLM como Aprendiz)
# Geometria propositalmente distinta para que a métrica do perito não seja trivial
PROBLEM_D_CENTERS = [(-1.5, 1.0), (1.5, -1.0)]
PROBLEM_D_STD = 1.3

# Métrica do perito para a Fase D
# O perito usa uma métrica ANISOTRÓPICA que pondera x2 muito mais do que x1.
# Isso torna a classificação não trivial: o LLM não pode se basear apenas em x1.
EXPERT_W = np.array([0.3, 1.5])  # Weights x2 heavily
EXPERT_CENTROIDS = np.array([[-1.5, 1.0], [1.5, -1.0]])

# Estratégias de dificuldade de exemplos para a Fase D
EXAMPLE_STRATEGIES = ["easy", "hard", "mixed", "random"]

# Nomes de classe distintos para testar viés semântico do LLM (conjunto reduzido)
NOMES_CLASSES = [
    ("A", "B"),
    ("0", "1"),
    ("Positivo", "Negativo"),
    ("Azul", "Vermelho"),
]

# Nomes de classes com ordem invertida para testar viés de posição (Sprint 3)
NOMES_CLASSES_INVERTIDAS = [
    ("B", "A"),
    ("1", "0"),
    ("Negativo", "Positivo"),
    ("Vermelho", "Azul"),
]

# Nomes semânticos para features — testar viés semântico nas variáveis (Sprint 4)
NOMES_FEATURES = [
    ("x1", "x2"),                    # Original (neutro)
    ("altura", "peso"),              # Semântico
    ("feature_1", "feature_2"),      # Técnico
]

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DE EXPERTS MÚLTIPLOS (Fase D)
# ═══════════════════════════════════════════════════════════════════════════

EXPERT_CONFIGS = [
    {"name": "aniso_x2",  "w": np.array([0.3, 1.5]),  "desc": "x2 dominante (original)"},
    {"name": "aniso_x1",  "w": np.array([1.5, 0.3]),  "desc": "x1 dominante (invertido)"},
    {"name": "euclidean", "w": np.array([1.0, 1.0]),  "desc": "pesos iguais (Euclidiana)"},
]

# ═══════════════════════════════════════════════════════════════════════════
# FLAGS DE EXECUÇÃO — controla quais experimentos rodar
# ═══════════════════════════════════════════════════════════════════════════

RUN_PHASES_ABC = True              # Fases A-C padrão
RUN_PHASE_D = True                 # Fase D padrão
RUN_CLASS_ORDER_BIAS = True        # Teste de inversão de ordem das classes
RUN_FEATURE_NAMES = True           # Teste de nomes semânticos nas features
RUN_DILUTION = True                # Experimento de diluição
RUN_R3_EXPERIMENT = True           # Projeção R3 (kernel quadrático)
RUN_MULTIPLE_EXPERTS = True        # Múltiplas configs de expert na Fase D
RUN_ALGORITHM_COMPARISON = True    # Segundo algoritmo de otimização inversa

# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class LearnedMetric:
    """Armazena a métrica aprendida no Problema A."""
    w: np.ndarray
    centroids: np.ndarray
    gamma: float
    source_problem: str


@dataclass
class ConsistencyMetrics:
    """Armazena métricas detalhadas de consistência entre predições do LLM e da métrica aprendida."""
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
    feature_names: Tuple[str, str] = ("x1", "x2")


@dataclass
class ResultadoPhaseDExperimento:
    """
    Novo v3.0: Armazena os resultados do experimento da Fase D (LLM como Aprendiz).
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
    """Calcula métricas detalhadas de consistência entre predições do LLM e da métrica aprendida.

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

def create_problem_a(n_samples: int = 300, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema A para aprendizado da métrica.

    Problema A serve como base de treino: o LLM classifica estes pontos em zero-shot
    e a métrica W é aprendida a partir dessas decisões.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_A_CENTERS,
        cluster_std=PROBLEM_A_STD,
        random_state=random_state
    )
    return X, y


def create_problem_b(n_samples: int = 200, random_state: int = 43) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema B para teste de consistência.

    Problema B possui geometria diferente do A (centróides deslocados verticalmente).
    Isso verifica se a consistência LLM-métrica se mantém fora do domínio de treino.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_B_CENTERS,
        cluster_std=PROBLEM_B_STD,
        random_state=random_state
    )
    return X, y


def create_problem_c(n_samples: int = 200, random_state: int = 44) -> Tuple[np.ndarray, np.ndarray]:
    """Gera o conjunto de dados do Problema C para teste de consistência adicional.

    Problema C possui distorção geométrica ainda mais acentuada que o B,
    tornando o teste de generalização da métrica mais severo.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_C_CENTERS,
        cluster_std=PROBLEM_C_STD,
        random_state=random_state
    )
    return X, y


def create_problem_d(n_samples: int = 150, random_state: int = 45) -> Tuple[np.ndarray, np.ndarray]:
    """
    Novo v3.0: Gera o conjunto de dados do Problema D para a Fase D (LLM como Aprendiz).
    Usa geometria propositalmente distinta para que o LLM não possa aprender a métrica do perito
    por intuição simples — é necessário capturar o peso anisotrópico w2 >> w1.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=PROBLEM_D_CENTERS,
        cluster_std=PROBLEM_D_STD,
        random_state=random_state
    )
    return X, y


def visualize_all_problems(
    X_a: np.ndarray, y_a: np.ndarray,
    X_b: np.ndarray, y_b: np.ndarray,
    X_c: np.ndarray, y_c: np.ndarray,
    filename: str = None
):
    """Visualiza os três problemas sintéticos lado a lado para inspeção visual da geometria."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    problems = [
        (X_a, y_a, PROBLEM_A_CENTERS, PROBLEM_A_STD, "PROBLEMA A\n(Aprendizado da Métrica)"),
        (X_b, y_b, PROBLEM_B_CENTERS, PROBLEM_B_STD, "PROBLEMA B\n(Teste de Consistência 1)"),
        (X_c, y_c, PROBLEM_C_CENTERS, PROBLEM_C_STD, "PROBLEMA C\n(Teste de Consistência 2)")
    ]

    for ax, (X, y, centers, std, title) in zip(axes, problems):
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

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_problem_d_with_expert(
    X_d: np.ndarray, y_gt_d: np.ndarray,
    y_expert_d: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    filename: str = None
):
    """
    Novo v3.0: Visualiza o Problema D com a fronteira de decisão do perito.
    Compara os rótulos verdadeiros (ground truth) com os rótulos do perito lado a lado.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Gráfico 1: Rótulos verdadeiros (ground truth)
    ax = axes[0]
    ax.scatter(X_d[:, 0], X_d[:, 1], c=y_gt_d, cmap="coolwarm",
               alpha=0.7, edgecolor="k", s=60)
    ax.scatter(*PROBLEM_D_CENTERS[0], marker='*', s=300, c='blue',
               edgecolor='black', linewidth=2, label='Centro Verdadeiro 0', zorder=5)
    ax.scatter(*PROBLEM_D_CENTERS[1], marker='*', s=300, c='red',
               edgecolor='black', linewidth=2, label='Centro Verdadeiro 1', zorder=5)
    ax.set_title("Problema D: Rótulos Verdadeiros (Ground Truth)", fontsize=11, fontweight='bold')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)

    # Gráfico 2: Rótulos do perito com fronteira de decisão
    ax = axes[1]

    x_min, x_max = -6, 6
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, expert_centroids, expert_w)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
    ax.scatter(X_d[:, 0], X_d[:, 1], c=y_expert_d, cmap="coolwarm",
               alpha=0.7, edgecolor="k", s=60)
    ax.scatter(*expert_centroids[0], marker='X', s=300, c='blue',
               edgecolor='k', linewidth=2, label='Centróide do Perito 0', zorder=5)
    ax.scatter(*expert_centroids[1], marker='X', s=300, c='red',
               edgecolor='k', linewidth=2, label='Centróide do Perito 1', zorder=5)
    ax.set_title(f"Rótulos do Perito (W=[{expert_w[0]:.1f}, {expert_w[1]:.1f}])",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)

    # Gráfico 3: Mapa de calor da margem (confiança do perito)
    ax = axes[2]
    confidences, _ = compute_metric_confidence(grid_points, expert_centroids, expert_w)
    conf_grid = confidences.reshape(xx.shape)

    im = ax.contourf(xx, yy, conf_grid, levels=20, cmap="RdYlGn")
    ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
    plt.colorbar(im, ax=ax, label="Margem (confiança)")
    ax.scatter(X_d[:, 0], X_d[:, 1], c='black', alpha=0.3, s=20)
    ax.set_title("Margem do Perito (Distância à Fronteira)", fontsize=11, fontweight='bold')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# INTERAÇÃO COM O LLM (COM RETENTATIVA DE FORMATO)
# =============================================================================

def build_prompt_zero_shot(x1: float, x2: float,
                           nome_classe_0: str, nome_classe_1: str,
                           nome_feature_0: str = "x1", nome_feature_1: str = "x2",
                           extra_features: Optional[List[Tuple[str, float]]] = None) -> str:
    """Constrói o prompt ZERO-SHOT (sem exemplos fornecidos ao LLM).

    Args:
        extra_features: Lista de (nome, valor) para features adicionais (ex: R3).
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
    """Constrói o prompt FEW-SHOT (com exemplos rotulados pela métrica).

    Examples pode conter tuplas de 3 (x1, x2, label) ou 4+ elementos para R3.
    """
    n_dim = 2 + (len(extra_features) if extra_features else 0)
    dim_label = f"{n_dim}D"

    examples_lines = []
    for ex in examples:
        line = f"  {nome_feature_0} = {ex[0]:.4f}, {nome_feature_1} = {ex[1]:.4f}"
        if len(ex) > 3:
            # Extra features no exemplo (R3)
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


def llm_classify_point_openai(client: OpenAI, model_name: str, prompt: str,
                               temperature: float) -> str:
    """Classifica um ponto usando API compatível com OpenAI (OpenAI ou Gemini)."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a classifier. Respond only with the class label."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def llm_classify_point_anthropic(client: anthropic.Anthropic, model_name: str,
                                  prompt: str, temperature: float) -> str:
    """Classifica um ponto usando a API da Anthropic (Claude)."""
    response = client.messages.create(
        model=model_name,
        max_tokens=50,
        temperature=temperature,
        system="You are a classifier. Respond only with the class label.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text.strip()


async def async_llm_classify_point_openai(async_client, model_name: str, prompt: str,
                                           temperature: float) -> str:
    """Versão assíncrona de llm_classify_point_openai para chamadas concorrentes."""
    response = await async_client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a classifier. Respond only with the class label."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def parse_llm_response(
    label: str,
    nome_classe_0: str,
    nome_classe_1: str
) -> Tuple[Optional[str], bool]:
    """Interpreta a resposta do LLM e verifica se é válida (corresponde a uma das classes).

    Tentativas em ordem crescente de permissividade:
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
    extra_features: lista de (nome, valor) para features adicionais (ex: R3).
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

    # Fallback determinístico: usa os bits das coordenadas para decidir qual classe atribuir,
    # evitando o viés sistemático de sempre retornar a Classe 0.
    # O valor é determinístico por ponto (reproduzível entre execuções).
    coord_bits = abs(int(round(x1 * 1e4))) + abs(int(round(x2 * 1e4)))
    fallback = nome_classe_0 if (coord_bits % 2 == 0) else nome_classe_1

    warnings.warn(
        f"\n  ⚠️ RESPOSTA MALFORMADA após {MAX_FORMAT_RETRIES} tentativas para o ponto ({x1:.4f}, {x2:.4f}).\n"
        f"     Respostas recebidas: {all_responses}\n"
        f"     Esperado: '{nome_classe_0}' ou '{nome_classe_1}'\n"
        f"     Fallback determinístico: '{fallback}' (coord_bits={coord_bits})\n"
    )

    return fallback


async def async_llm_classify_point(
    x1: float, x2: float,
    nome_classe_0: str, nome_classe_1: str,
    examples: Optional[List] = None,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None
) -> Tuple[str, str, bool]:
    """Versão assíncrona de llm_classify_point. Retorna (label_parsed, raw_response, was_malformed)."""
    global async_client, MODEL_NAME, CURRENT_PROVIDER, CURRENT_TEMPERATURE

    if examples is None or len(examples) == 0:
        prompt = build_prompt_zero_shot(x1, x2, nome_classe_0, nome_classe_1,
                                        nome_feature_0, nome_feature_1, extra_features)
    else:
        prompt = build_prompt_few_shot(x1, x2, examples, nome_classe_0, nome_classe_1,
                                       nome_feature_0, nome_feature_1, extra_features)

    all_responses = []
    all_prompts = []

    for format_attempt in range(MAX_FORMAT_RETRIES):
        for rate_attempt in range(MAX_RETRIES):
            try:
                config = PROVIDER_CONFIG[CURRENT_PROVIDER]

                if config["client_type"] == "anthropic":
                    # Anthropic não tem async_client neste script; fallback sync em thread
                    label = await asyncio.get_event_loop().run_in_executor(
                        None, llm_classify_point_anthropic, client, MODEL_NAME, prompt, CURRENT_TEMPERATURE
                    )
                else:
                    label = await async_llm_classify_point_openai(async_client, MODEL_NAME, prompt, CURRENT_TEMPERATURE)

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

    # Fallback determinístico
    coord_bits = abs(int(round(x1 * 1e4))) + abs(int(round(x2 * 1e4)))
    fallback = nome_classe_0 if (coord_bits % 2 == 0) else nome_classe_1
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
        f"     Fallback determinístico: '{fallback}' (coord_bits={coord_bits})\n"
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
    extra_feature_names: Optional[List[str]] = None
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
                examples, nome_feature_0, nome_feature_1, extra_feats
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
    extra_feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, int]:
    """Coleta as decisões do LLM para todos os pontos do conjunto de dados.

    Wrapper síncrono que delega para async_collect_llm_decisions com chamadas concorrentes.
    """
    return asyncio.run(async_collect_llm_decisions(
        X, nome_classe_0, nome_classe_1,
        examples=examples, verbose=verbose, label_prefix=label_prefix,
        nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
        extra_features_matrix=extra_features_matrix,
        extra_feature_names=extra_feature_names
    ))


# =============================================================================
# APRENDIZADO DE MÉTRICA (PERCEPTRON ESTRUTURADO)
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
    eta: float = 0.1,
    C: float = 1.0,
    gamma_init: float = 0.1,
    delta_gamma: float = 0.1,
    max_epochs: int = 100,
    tol: float = 1e-5,
    verbose: bool = False,
    use_best_effort: bool = False
) -> Tuple[np.ndarray, float]:
    """Wrapper que delega para RelaxedPerceptron."""
    model = RelaxedPerceptron(
        eta=eta, C=C, gamma_init=gamma_init, delta_gamma=delta_gamma,
        max_epochs=max_epochs, tol=tol, use_best_effort=use_best_effort,
        verbose=verbose,
    )
    return model.fit(X, y, centroids)


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
    """Adiciona feature de interação x3 = x1 * x2 para projeção R3 (kernel quadrático).

    Conforme email do orientador: cria uma terceira feature dada pelo produto
    das duas primeiras. O vetor W terá 3 componentes. Se projetarmos de volta
    para R2, teremos a equação de uma hipérbole.
    """
    x3 = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, x3])


def predict_with_metric(X: np.ndarray, centroids: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Prediz a classe de cada ponto usando a métrica aprendida (vizinho mais próximo sob d_W)."""
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
    verbose: bool = True
) -> Tuple[LearnedMetric, np.ndarray, float, float, int]:
    """FASE A: Aprende a métrica W a partir das decisões do LLM no Problema A."""
    if verbose:
        print("\n  ══════════════════════════════════════════════════════════════")
        print("  FASE A: APRENDENDO A MÉTRICA W DO PROBLEMA A (ZERO-SHOT)")
        print("  ══════════════════════════════════════════════════════════════")
        print("  Objetivo: capturar o processo decisório INTRÍNSECO do LLM.")
        print("  O LLM classifica pontos sem exemplos (zero-shot), e a métrica")
        print("  W é aprendida a partir dessas decisões via Perceptron Estruturado.")

    if verbose:
        print(f"\n  Passo 1: Coletando decisões do LLM no Problema A (zero-shot)...")
        print(f"    Classes usadas: '{nome_classe_0}' e '{nome_classe_1}'")
        print(f"    Pontos de treino: {len(X_train_a)}")

    y_llm_train_a, n_malformed = collect_llm_decisions(
        X_train_a, nome_classe_0, nome_classe_1,
        examples=None, verbose=verbose,
        label_prefix="[Problema A] "
    )

    if len(np.unique(y_llm_train_a)) < 2:
        print("  ⚠ AVISO: O LLM classificou tudo como uma única classe! Verifique o modelo e os prompts.")
        return None, y_llm_train_a, 0.5, accuracy_score(y_true_train_a, y_llm_train_a), n_malformed

    centroids_a = compute_centroids(X_train_a, y_llm_train_a)

    if verbose:
        print(f"\n  Passo 2: Calculando centróides a partir das decisões do LLM...")
        print(f"    Centróide da Classe 0: ({centroids_a[0, 0]:.3f}, {centroids_a[0, 1]:.3f})")
        print(f"    Centróide da Classe 1: ({centroids_a[1, 0]:.3f}, {centroids_a[1, 1]:.3f})")

    if verbose:
        print("\n  Passo 3: Aprendendo a métrica W com o Perceptron Estruturado com Relaxação de Margem...")
        print("    (Busca binária no gamma ótimo para maximizar a margem da fronteira de decisão)")

    w_learned, gamma_optimal = train_relaxed_perceptron(
        X_train_a, y_llm_train_a, centroids_a,
        eta=0.05, C=10.0, delta_gamma=0.05,
        max_epochs=50, tol=1e-4, verbose=verbose,
        use_best_effort=True  # retorna melhor W parcial quando separação perfeita é impossível
    )

    if verbose:
        w_norm = w_learned / np.sum(w_learned) if np.sum(w_learned) > 0 else w_learned
        print(f"    >>> Ŵ_LLM estimado (bruto): [{w_learned[0]:.4f}, {w_learned[1]:.4f}]")
        print(f"    >>> Ŵ_LLM estimado (normalizado): [{w_norm[0]:.3f}, {w_norm[1]:.3f}]")
        w_ratio = w_learned[0] / w_learned[1] if w_learned[1] != 0 else float('inf')
        print(f"    >>> Razão w1/w2: {w_ratio:.4f}")
        print(f"    >>> Gamma ótimo encontrado: {gamma_optimal:.4f}")

    y_pred_metric_a = predict_with_metric(X_train_a, centroids_a, w_learned)
    fidelity = accuracy_score(y_llm_train_a, y_pred_metric_a)
    llm_accuracy = accuracy_score(y_true_train_a, y_llm_train_a)

    if verbose:
        print(f"\n  Passo 4: Verificando fidelidade da métrica aprendida...")
        print(f"    Fidelidade (métrica vs. LLM no Problema A): {fidelity:.1%}")
        print(f"    Acurácia do LLM vs. ground truth: {llm_accuracy:.1%}")
        if fidelity >= 0.9:
            print(f"    A métrica reproduz bem as decisões do LLM.")
        else:
            print(f"    ⚠ Fidelidade baixa — a métrica pode não representar bem o LLM.")

    learned_metric = LearnedMetric(
        w=w_learned, centroids=centroids_a,
        gamma=gamma_optimal, source_problem="Problem_A"
    )

    return learned_metric, y_llm_train_a, fidelity, llm_accuracy, n_malformed


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
    verbose: bool = True
) -> Tuple[ConsistencyMetrics, np.ndarray, np.ndarray, float, float, int, float]:
    """Teste de consistência genérico em um novo problema usando a métrica aprendida no Problema A.

    Garante que os exemplos few-shot (quando usados) são excluídos do conjunto de avaliação,
    evitando vazamento de dados (data leakage) entre seleção de exemplos e avaliação.
    """
    if verbose:
        print(f"\n  ══════════════════════════════════════════════════════════════")
        print(f"  {problem_name}: TESTE DE CONSISTÊNCIA")
        print(f"  ══════════════════════════════════════════════════════════════")
        print(f"  Aplicando a métrica W aprendida no Problema A a este novo problema.")
        print(f"  Objetivo: verificar se o LLM concorda com a métrica em dados não vistos.")

    confidences, y_pred_metric = compute_metric_confidence(
        X, learned_metric.centroids, learned_metric.w
    )

    if verbose:
        n_0 = np.sum(y_pred_metric == 0)
        n_1 = np.sum(y_pred_metric == 1)
        print(f"    Predições da métrica W_A: Classe 0={n_0}, Classe 1={n_1}")

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

    # Correção de vazamento (TODO 1): exclui exemplos few-shot da avaliação.
    # No modo zero-shot, selected_indices é vazio e test_mask mantém todos os pontos.
    test_mask = np.ones(len(X), dtype=bool)
    test_mask[selected_indices] = False
    X_test = X[test_mask]
    y_true_test = y_true[test_mask]
    y_metric_test = y_pred_metric[test_mask]

    if verbose:
        print(f"\n  Coletando decisões do LLM para {len(X_test)} pontos de teste...")

    y_llm, n_malformed = collect_llm_decisions(
        X_test, nome_classe_0, nome_classe_1,
        examples=examples, verbose=verbose,
        label_prefix=f"[{problem_name}] "
    )

    consistency_metrics = compute_consistency_metrics(y_llm, y_metric_test)
    llm_accuracy = accuracy_score(y_true_test, y_llm)
    metric_accuracy = accuracy_score(y_true_test, y_metric_test)

    # Linha de base Euclidiana: pesos uniformes para verificar limitação da métrica diagonal.
    # Se a métrica aprendida não superar a Euclidiana em >1%, sinalizamos limitação diagonal.
    w_euclidean = np.ones_like(learned_metric.w)
    y_pred_euclidean = predict_with_metric(X_test, learned_metric.centroids, w_euclidean)
    euclidean_metrics = compute_consistency_metrics(y_llm, y_pred_euclidean)
    consistency_euclidean = euclidean_metrics.accuracy

    if verbose:
        print(f"\n  RESULTADOS: {problem_name}")
        print(f"    CONSISTÊNCIA (LLM vs. Métrica W_A): {consistency_metrics.accuracy:.1%}")
        print(f"    Kappa de Cohen: {consistency_metrics.cohen_kappa:.3f}")
        print(f"    F1-Score: {consistency_metrics.f1_score:.3f}")
        print(f"    Concordâncias: {consistency_metrics.n_agreements} | Discordâncias: {consistency_metrics.n_disagreements}")
        print(f"    Consistência da linha de base Euclidiana: {consistency_euclidean:.1%}")
        print(f"    Conjunto de teste: {len(X_test)} pontos (excluídos {len(selected_indices)} exemplos few-shot)")
        if consistency_metrics.accuracy >= 0.85:
            print(f"    O LLM MANTÉM consistência elevada com a métrica W_A neste problema.")
        elif consistency_metrics.accuracy >= 0.7:
            print(f"    O LLM mantém consistência PARCIAL com a métrica W_A neste problema.")
        else:
            print(f"    ⚠ O LLM NÃO mantém consistência com a métrica W_A neste problema.")

    return consistency_metrics, y_llm, y_metric_test, llm_accuracy, metric_accuracy, n_malformed, consistency_euclidean


# =============================================================================
# FASE D: LLM COMO APRENDIZ (NOVO v3.0)
# =============================================================================

def expert_classify(
    X: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray
) -> np.ndarray:
    """
    Classifica pontos usando a métrica conhecida do perito.
    O perito atua como "verdade absoluta" (rótulo de referência) para a Fase D:
    o objetivo do LLM é aprender a reproduzir essa classificação a partir de exemplos.
    """
    return predict_with_metric(X, expert_centroids, expert_w)


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
    Novo v3.0: Seleciona exemplos few-shot usando estratégias de dificuldade.

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

    # Aviso de desbalanceamento (TODO 6): alerta se uma classe tiver menos pontos do que o solicitado
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


def phase_d_llm_as_learner(
    X_d: np.ndarray,
    y_gt_d: np.ndarray,
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
) -> ResultadoPhaseDExperimento:
    """
    Novo v3.0: Fase D — LLM como Aprendiz.

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
        print(f"  FASE D: LLM COMO APRENDIZ ({shot_label}, estratégia={strategy})")
        print(f"  Métrica do Perito W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
        print(f"  ══════════════════════════════════════════════════════════════")
        print(f"  O perito rotula os dados com sua métrica conhecida.")
        print(f"  O LLM recebe exemplos do perito e deve APRENDER o padrão de classificação.")

    # Passo 1: Perito classifica todos os pontos usando sua métrica W_expert conhecida
    y_expert = expert_classify(X_d, expert_w, expert_centroids)
    expert_accuracy = accuracy_score(y_gt_d, y_expert)

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
            X_d, y_expert, expert_w, expert_centroids,
            n_examples=n_shot,
            strategy=strategy,
            nome_classe_0=nome_classe_0,
            nome_classe_1=nome_classe_1,
            random_state=random_seed + repeticao,
            verbose=verbose
        )

    # Passo 3: Define o conjunto de teste — todos os pontos NÃO usados como exemplos
    # Isso garante avaliação honesta: o LLM não é testado nos pontos que recebeu como exemplos
    all_indices = np.arange(len(X_d))
    test_mask = np.ones(len(X_d), dtype=bool)
    if len(example_indices) > 0:
        test_mask[example_indices] = False
    test_indices = all_indices[test_mask]

    X_test = X_d[test_indices]
    y_expert_test = y_expert[test_indices]
    y_gt_test = y_gt_d[test_indices]

    if verbose:
        print(f"\n  Passo 3: LLM classificando {len(X_test)} pontos de teste ({shot_label})...")
        if n_shot > 0:
            print(f"    O LLM receberá {n_shot} exemplos rotulados pelo perito como contexto.")

    # Passo 4: LLM classifica os pontos de teste usando os exemplos como contexto few-shot
    y_llm_test, n_malformed = collect_llm_decisions(
        X_test, nome_classe_0, nome_classe_1,
        examples=examples, verbose=verbose,
        label_prefix="[Fase D] "
    )

    # Passo 5: Computa métricas de alinhamento (LLM vs. Perito) e acurácia vs. ground truth
    consistency = compute_consistency_metrics(y_llm_test, y_expert_test)
    llm_accuracy_vs_gt = accuracy_score(y_gt_test, y_llm_test)

    if verbose:
        print(f"\n  ═══════════════════════════════════════════════════════════")
        print(f"  RESULTADOS: FASE D ({shot_label}, estratégia={strategy})")
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

    return ResultadoPhaseDExperimento(
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
    y_llm_train_a_cache: Optional[np.ndarray] = None,
    fidelity_cache: Optional[float] = None,
    llm_accuracy_a_cache: Optional[float] = None,
    n_malformed_a_cache: Optional[int] = None,
    verbose: bool = True
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
        y_llm_train_a = y_llm_train_a_cache
        fidelity_a = fidelity_cache
        llm_accuracy_a = llm_accuracy_a_cache
        total_malformed += n_malformed_a_cache if n_malformed_a_cache else 0
    else:
        if verbose:
            print("\n  [Iniciando Fase A: Aprendizado da Métrica W a partir das decisões do LLM...]")
        result_a = phase_a_learn_metric(
            X_train_a, y_true_train_a,
            nome_classe_0, nome_classe_1, verbose=verbose
        )
        learned_metric, y_llm_train_a, fidelity_a, llm_accuracy_a, n_malformed_a = result_a
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
            ),
            None, y_llm_train_a, 0.5, llm_accuracy_a, total_malformed, {}
        )

    metrics_b, y_llm_b, y_pred_metric_b, llm_accuracy_b, metric_accuracy_b, n_malformed_b, euclidean_b = phase_consistency_test(
        X_b, y_true_b, learned_metric,
        nome_classe_0, nome_classe_1, n_shot, "FASE B (Problema B)", verbose=verbose
    )
    total_malformed += n_malformed_b

    metrics_c, y_llm_c, y_pred_metric_c, llm_accuracy_c, metric_accuracy_c, n_malformed_c, euclidean_c = phase_consistency_test(
        X_c, y_true_c, learned_metric,
        nome_classe_0, nome_classe_1, n_shot, "FASE C (Problema C)", verbose=verbose
    )
    total_malformed += n_malformed_c

    # Sinalização de limitação diagonal (TODO 5): verifica se a métrica aprendida supera a Euclidiana.
    # Se a melhoria for <= 1%, a restrição diagonal pode ser o gargalo — métricas não-diagonais
    # (ex.: Mahalanobis completo) poderiam ser mais expressivas neste caso.
    if metrics_b.accuracy <= euclidean_b + 0.01:
        diagonal_limitation_flag = 1
    else:
        diagonal_limitation_flag = 0

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
        ),
        learned_metric, y_llm_train_a, fidelity_a, llm_accuracy_a, total_malformed,
        {
            'y_llm_b': y_llm_b, 'y_llm_c': y_llm_c,
            'y_metric_b': y_pred_metric_b, 'y_metric_c': y_pred_metric_c,
            'metrics_b': metrics_b, 'metrics_c': metrics_c,
        }
    )


# =============================================================================
# VISUALIZAÇÕES (FASES A-C)
# =============================================================================

def plot_decision_boundary_all_problems(
    X_a: np.ndarray, y_llm_a: np.ndarray,
    X_b: np.ndarray, y_llm_b: np.ndarray,
    X_c: np.ndarray, y_llm_c: np.ndarray,
    learned_metric: LearnedMetric,
    title: str,
    filename: str = None
):
    """Visualiza as fronteiras de decisão da métrica aprendida nos três problemas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    problems = [
        (X_a, y_llm_a, "Problema A (Treino)"),
        (X_b, y_llm_b, "Problema B (Teste de Consistência 1)"),
        (X_c, y_llm_c, "Problema C (Teste de Consistência 2)")
    ]

    for ax, (X, y_llm, problem_name) in zip(axes, problems):
        x_min, x_max = min(X[:, 0].min() - 1.5, -6), max(X[:, 0].max() + 1.5, 6)
        y_min, y_max = min(X[:, 1].min() - 1.5, -5), max(X[:, 1].max() + 1.5, 5)

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_with_metric(grid_points, learned_metric.centroids, learned_metric.w)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        ax.scatter(X[:, 0], X[:, 1], c=y_llm, cmap="coolwarm", edgecolor="k", s=60, zorder=3)
        ax.scatter(*learned_metric.centroids[0], marker='X', s=300, c='blue',
                   edgecolor='k', linewidth=2, label='Centróide 0', zorder=5)
        ax.scatter(*learned_metric.centroids[1], marker='X', s=300, c='red',
                   edgecolor='k', linewidth=2, label='Centróide 1', zorder=5)
        ax.set_title(f"{problem_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_disagreement_analysis(
    X: np.ndarray,
    y_llm: np.ndarray,
    y_metric: np.ndarray,
    learned_metric: LearnedMetric,
    problem_name: str,
    filename: str = None
):
    """Visualiza os pontos onde o LLM e a métrica aprendida discordam, e analisa suas margens."""
    agreements = y_llm == y_metric
    disagreements = ~agreements

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, learned_metric.centroids, learned_metric.w)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors='k', linewidths=1.5, levels=[0.5])
    ax.scatter(X[agreements, 0], X[agreements, 1], c=y_llm[agreements],
               cmap="coolwarm", alpha=0.4, edgecolor="gray", s=40,
               label=f'Concordâncias ({np.sum(agreements)})')
    ax.scatter(X[disagreements, 0], X[disagreements, 1], c='yellow',
               edgecolor="black", s=150, linewidth=2, marker='s',
               label=f'Discordâncias ({np.sum(disagreements)})', zorder=5)
    ax.set_title(f"{problem_name}: Localização das Discordâncias", fontsize=11, fontweight='bold')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if np.sum(disagreements) > 0:
        confidences, _ = compute_metric_confidence(X, learned_metric.centroids, learned_metric.w)
        conf_agree = confidences[agreements]
        conf_disagree = confidences[disagreements]
        bins = np.linspace(0, max(confidences.max(), 1), 20)
        ax.hist(conf_agree, bins=bins, alpha=0.6, label=f'Concordâncias (n={len(conf_agree)})',
                color='green', edgecolor='black')
        ax.hist(conf_disagree, bins=bins, alpha=0.6, label=f'Discordâncias (n={len(conf_disagree)})',
                color='red', edgecolor='black')
        ax.set_xlabel("Margem (distância à fronteira)")
        ax.set_ylabel("Contagem")
        ax.set_title("Distribuição de Margem: Concordâncias vs. Discordâncias", fontsize=11, fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Sem discordâncias!", ha='center', va='center', fontsize=14, transform=ax.transAxes)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_grouped = df.groupby('n_shot').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_grouped.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']

    x = np.arange(len(df_grouped))
    width = 0.35

    ax = axes[0, 0]
    ax.bar(x - width/2, df_grouped['b_mean'], width, yerr=df_grouped['b_std'],
           label='Problema B', color='steelblue', capsize=3)
    ax.bar(x + width/2, df_grouped['c_mean'], width, yerr=df_grouped['c_std'],
           label='Problema C', color='coral', capsize=3)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n)}-shot' for n in df_grouped['n_shot']])
    ax.set_ylabel('Consistência')
    ax.set_title('Consistência: Problema B vs C', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)

    ax = axes[0, 1]
    df_kappa = df.groupby('n_shot').agg({'kappa_b': ['mean', 'std'], 'kappa_c': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']
    ax.bar(x - width/2, df_kappa['b_mean'], width, yerr=df_kappa['b_std'],
           label='Problema B', color='steelblue', capsize=3)
    ax.bar(x + width/2, df_kappa['c_mean'], width, yerr=df_kappa['c_std'],
           label='Problema C', color='coral', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n)}-shot' for n in df_kappa['n_shot']])
    ax.set_ylabel("Kappa de Cohen")
    ax.set_title("Kappa de Cohen: B vs C", fontweight='bold')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    ax = axes[1, 0]
    df_f1 = df.groupby('n_shot').agg({'f1_b': ['mean', 'std'], 'f1_c': ['mean', 'std']}).reset_index()
    df_f1.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']
    ax.bar(x - width/2, df_f1['b_mean'], width, yerr=df_f1['b_std'],
           label='Problema B', color='steelblue', capsize=3)
    ax.bar(x + width/2, df_f1['c_mean'], width, yerr=df_f1['c_std'],
           label='Problema C', color='coral', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n)}-shot' for n in df_f1['n_shot']])
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score: B vs C', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)

    ax = axes[1, 1]
    metrics_data = []
    for col, label in [('consistencia_b', 'Acu B'), ('consistencia_c', 'Acu C'),
                       ('kappa_b', 'Kappa B'), ('kappa_c', 'Kappa C'),
                       ('f1_b', 'F1 B'), ('f1_c', 'F1 C')]:
        for val in df[col]:
            metrics_data.append({'Metric': label, 'Value': val})
    df_metrics = pd.DataFrame(metrics_data)
    colors = ['steelblue', 'coral'] * 3
    positions = [0, 0.6, 1.5, 2.1, 3.0, 3.6]
    for i, (metric, color) in enumerate(zip(['Acu B', 'Acu C', 'Kappa B', 'Kappa C', 'F1 B', 'F1 C'], colors)):
        data = df_metrics[df_metrics['Metric'] == metric]['Value']
        bp = ax.boxplot([data], positions=[positions[i]], widths=0.4, patch_artist=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
    ax.set_xticks([0.3, 1.8, 3.3])
    ax.set_xticklabels(['Acurácia', "Kappa de Cohen", 'F1-Score'])
    ax.set_ylabel('Pontuação')
    ax.set_title('Distribuição de Todas as Métricas', fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Problema B'),
                       Patch(facecolor='coral', alpha=0.7, label='Problema C')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


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
# VISUALIZAÇÕES DA FASE D (NOVO v3.0)
# =============================================================================

def plot_phase_d_learning_curve(
    results_d: List[ResultadoPhaseDExperimento],
    filename: str = None
):
    """
    Novo v3.0: Curva de aprendizado mostrando como o LLM melhora com mais exemplos,
    discriminada por estratégia de seleção de exemplos.

    Esta é a VISUALIZAÇÃO PRINCIPAL da Fase D:
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
        for r in results_d
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

    metrics = [
        ('accuracy', 'Concordância LLM vs. Perito (Acurácia)', axes[0]),
        ('kappa', "Kappa de Cohen", axes[1]),
        ('f1', 'F1-Score', axes[2]),
    ]

    for metric_col, metric_name, ax in metrics:
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
        ax.set_title(f'Fase D: {metric_name}\nvs Número de Exemplos', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_d_strategy_comparison(
    results_d: List[ResultadoPhaseDExperimento],
    filename: str = None
):
    """
    Novo v3.0: Compara estratégias em cada nível de n_shot usando gráficos de barras agrupadas.
    Permite verificar qual estratégia de seleção de exemplos é mais eficaz para cada quantidade de shots.
    """
    df = pd.DataFrame([
        {
            'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
        }
        for r in results_d
    ])

    # Filtra apenas os resultados few-shot (exclui zero-shot, que é igual para todas as estratégias)
    df_fs = df[df['n_shot'] > 0]

    if len(df_fs) == 0:
        print("  Sem resultados few-shot para comparar estratégias.")
        return

    n_shots = sorted(df_fs['n_shot'].unique())

    fig, axes = plt.subplots(1, len(n_shots), figsize=(5 * len(n_shots), 5), squeeze=False)
    axes = axes[0]

    strategy_colors = {
        'easy': '#2ecc71',
        'hard': '#e74c3c',
        'mixed': '#3498db',
        'random': '#95a5a6',
    }

    for ax_idx, n in enumerate(n_shots):
        ax = axes[ax_idx]
        df_n = df_fs[df_fs['n_shot'] == n]

        df_strat = df_n.groupby('strategy').agg({
            'accuracy': ['mean', 'std']
        }).reset_index()
        df_strat.columns = ['strategy', 'mean', 'std']

        bars = ax.bar(
            range(len(df_strat)),
            df_strat['mean'],
            yerr=df_strat['std'],
            color=[strategy_colors.get(s, 'gray') for s in df_strat['strategy']],
            edgecolor='black',
            capsize=5
        )

        ax.set_xticks(range(len(df_strat)))
        ax.set_xticklabels([s.capitalize() for s in df_strat['strategy']], fontsize=10)
        ax.set_ylabel('Concordância LLM vs. Perito')
        ax.set_title(f'{n} Exemplos Few-Shot', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

        # Adiciona rótulos de valor no topo das barras para facilitar leitura
        for bar, mean_val in zip(bars, df_strat['mean']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean_val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Fase D: Comparação de Estratégias por Número de Exemplos',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_d_example_locations(
    X_d: np.ndarray,
    y_expert: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    n_examples: int = 10,
    random_state: int = 42,
    filename: str = None
):
    """
    Novo v3.0: Visualiza ONDE cada estratégia seleciona os exemplos.
    Exibe seleções easy/hard/mixed/random sobre a fronteira de decisão do perito.
    Fundamental para entender intuitivamente o que cada estratégia oferece ao LLM.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    strategies = ["easy", "hard", "mixed", "random"]

    x_min, x_max = -6, 6
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, expert_centroids, expert_w)
    Z = Z.reshape(xx.shape)

    for ax, strategy in zip(axes.flatten(), strategies):
        # Fronteira de decisão do perito
        ax.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=1.5, levels=[0.5])

        # Todos os pontos (transparentes — segundo plano)
        ax.scatter(X_d[:, 0], X_d[:, 1], c=y_expert, cmap="coolwarm",
                   alpha=0.2, edgecolor="gray", s=30)

        # Exemplos selecionados (destacados — primeiro plano)
        _, selected_indices = select_examples_by_strategy(
            X_d, y_expert, expert_w, expert_centroids,
            n_examples=n_examples, strategy=strategy,
            nome_classe_0="A", nome_classe_1="B",
            random_state=random_state, verbose=False
        )

        ax.scatter(X_d[selected_indices, 0], X_d[selected_indices, 1],
                   c=y_expert[selected_indices], cmap="coolwarm",
                   edgecolor="black", s=200, linewidth=2, marker='*',
                   zorder=5, label=f'Selecionados ({len(selected_indices)})')

        # Calcula as margens dos exemplos selecionados para exibir no título
        confidences, _ = compute_metric_confidence(X_d, expert_centroids, expert_w)
        sel_margins = confidences[selected_indices]

        ax.set_title(f'Estratégia: {strategy.upper()}\n'
                     f'Margem média: {sel_margins.mean():.3f} '
                     f'[{sel_margins.min():.3f}, {sel_margins.max():.3f}]',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.suptitle(f'Fase D: Estratégias de Seleção de Exemplos ({n_examples} exemplos)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# NOVOS GRÁFICOS (Reunião 13/03 — Pedidos do orientador)
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Scatter w0 vs w1 colorido por seed
    ax = axes[0]
    seeds = sorted(df['seed'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    for i, seed in enumerate(seeds):
        subset = df[df['seed'] == seed]
        ax.scatter(subset['w0'], subset['w1'], c=[colors[i]], s=100,
                   edgecolor='black', label=f'Semente {seed}', zorder=3)
    ax.set_xlabel('$w_1$ (peso da dimensão $x_1$)')
    ax.set_ylabel('$w_2$ (peso da dimensão $x_2$)')
    ax.set_title('Ŵ_LLM Estimado por Semente', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Boxplot de w0 e w1 por seed
    ax = axes[1]
    seeds_str = [str(s) for s in seeds]
    w0_by_seed = [df[df['seed'] == s]['w0'].values for s in seeds]
    w1_by_seed = [df[df['seed'] == s]['w1'].values for s in seeds]
    x = np.arange(len(seeds))
    width = 0.35
    bp1 = ax.boxplot(w0_by_seed, positions=x - width/2, widths=width*0.8, patch_artist=True)
    bp2 = ax.boxplot(w1_by_seed, positions=x + width/2, widths=width*0.8, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds_str)
    ax.set_xlabel('Semente Aleatória')
    ax.set_ylabel('Valor do Peso')
    ax.set_title('Distribuição de $w_1$ e $w_2$ por Semente', fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='steelblue', alpha=0.7, label='$w_1$'),
                       Patch(facecolor='coral', alpha=0.7, label='$w_2$')])

    # Plot 3: Razão w0/w1 por seed
    ax = axes[2]
    df_ratio = df.groupby('seed').agg({'ratio': ['mean', 'std']}).reset_index()
    df_ratio.columns = ['seed', 'mean', 'std']
    ax.bar(range(len(df_ratio)), df_ratio['mean'], yerr=df_ratio['std'],
           color='mediumpurple', edgecolor='black', capsize=5, alpha=0.8)
    ax.set_xticks(range(len(df_ratio)))
    ax.set_xticklabels([str(int(s)) for s in df_ratio['seed']])
    ax.set_xlabel('Semente Aleatória')
    ax.set_ylabel('Razão $w_1/w_2$')
    ax.set_title('Razão $w_1/w_2$ por Semente', fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Euclidiana (razão=1)')
    ax.legend()

    plt.suptitle('Distribuição do Vetor Ŵ_LLM Estimado entre Sementes e Repetições',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metric_errors_phase_a(
    X: np.ndarray, y_llm: np.ndarray, y_metric: np.ndarray,
    w: np.ndarray, centroids: np.ndarray, filename: str = None
):
    """Visualiza onde a métrica erra vs. a rotulação do LLM na Fase A (pedido do orientador)."""
    agreements = y_llm == y_metric
    disagreements = ~agreements

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Pontos com concordância e discordância sobre fronteira
    ax = axes[0]
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, centroids, w)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.15, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])

    ax.scatter(X[agreements, 0], X[agreements, 1], c=y_llm[agreements],
               cmap="coolwarm", alpha=0.5, edgecolor="gray", s=40, label=f'Concordam ({np.sum(agreements)})')
    ax.scatter(X[disagreements, 0], X[disagreements, 1],
               c='yellow', edgecolor="red", s=150, linewidth=2, marker='X',
               label=f'Discordam ({np.sum(disagreements)})', zorder=5)
    ax.scatter(*centroids[0], marker='D', s=200, c='blue', edgecolor='k', linewidth=2, zorder=6)
    ax.scatter(*centroids[1], marker='D', s=200, c='red', edgecolor='k', linewidth=2, zorder=6)

    fidelity = np.mean(agreements)
    ax.set_title(f'Fase A: Ŵ_LLM vs. Rotulação do LLM\nFidelidade: {fidelity:.1%}', fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Erros por distância à fronteira
    ax = axes[1]
    confidences, _ = compute_metric_confidence(X, centroids, w)

    # Dividir em faixas de distância
    bins = np.linspace(0, confidences.max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    error_rates = []
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if np.sum(mask) > 0:
            error_rates.append(np.mean(disagreements[mask]))
        else:
            error_rates.append(0)

    colors_bar = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.1 else '#2ecc71' for r in error_rates]
    ax.bar(range(len(error_rates)), error_rates, color=colors_bar, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(error_rates)))
    ax.set_xticklabels([f'{b:.1f}' for b in bin_centers], rotation=45)
    ax.set_xlabel('Margem (distância à fronteira)')
    ax.set_ylabel('Taxa de Erro')
    ax.set_title('Taxa de Discordância por Faixa de Margem', fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Análise de Erros da Métrica Ŵ_LLM na Fase A', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Consistência B e C
    ax = axes[0]
    x = np.arange(len(df_grouped))
    width = 0.35
    ax.bar(x - width/2, df_grouped['b_mean'], width, yerr=df_grouped['b_std'],
           label='Problema B', color='steelblue', capsize=3, edgecolor='black')
    ax.bar(x + width/2, df_grouped['c_mean'], width, yerr=df_grouped['c_std'],
           label='Problema C', color='coral', capsize=3, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped['pair'], rotation=45, ha='right')
    ax.set_ylabel('Consistência')
    ax.set_title('Consistência por Par de Classes\n(inclui pares invertidos)', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Fidelidade Fase A
    ax = axes[1]
    ax.bar(x, df_grouped['fid_mean'], yerr=df_grouped['fid_std'],
           color='mediumpurple', capsize=3, edgecolor='black', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped['pair'], rotation=45, ha='right')
    ax.set_ylabel('Fidelidade Fase A')
    ax.set_title('Fidelidade da Métrica por Par de Classes', fontweight='bold')
    ax.set_ylim(0, 1.1)

    plt.suptitle('Análise de Viés de Ordem/Posição das Classes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dilution_experiment(results_dilution: List[ResultadoPhaseDExperimento], filename: str = None):
    """Gráfico do experimento de diluição: 3 hard fixos + N easy progressivos."""
    if not results_dilution:
        return

    df = pd.DataFrame([{
        'n_shot': r.n_shot,
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'strategy': r.example_strategy,
    } for r in results_dilution])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Acurácia
    ax = axes[0]
    df_grouped = df.groupby('n_shot').agg({'accuracy': ['mean', 'std']}).reset_index()
    df_grouped.columns = ['n_shot', 'mean', 'std']
    df_grouped = df_grouped.sort_values('n_shot')
    ax.errorbar(df_grouped['n_shot'], df_grouped['mean'], yerr=df_grouped['std'],
                marker='o', linewidth=2, markersize=8, capsize=4, color='#e74c3c')
    # Referência: performance com 3 hard puros
    ref_3hard = df[df['n_shot'] == 3]
    if len(ref_3hard) > 0:
        ax.axhline(y=ref_3hard['accuracy'].mean(), color='gray', linestyle='--',
                   alpha=0.7, label=f'3 hard puros ({ref_3hard["accuracy"].mean():.1%})')
    ax.set_xlabel('Total de Exemplos (3 hard + N easy)')
    ax.set_ylabel('Concordância LLM vs. Perito')
    ax.set_title('Experimento de Diluição: Acurácia', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Kappa
    ax = axes[1]
    df_kappa = df.groupby('n_shot').agg({'kappa': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['n_shot', 'mean', 'std']
    df_kappa = df_kappa.sort_values('n_shot')
    ax.errorbar(df_kappa['n_shot'], df_kappa['mean'], yerr=df_kappa['std'],
                marker='s', linewidth=2, markersize=8, capsize=4, color='#3498db')
    ax.set_xlabel('Total de Exemplos (3 hard + N easy)')
    ax.set_ylabel('Kappa de Cohen')
    ax.set_title('Experimento de Diluição: Kappa', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    plt.suptitle('Experimento de Diluição: Hard Fixos + Easy Progressivos', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_r3_comparison(results_r2: List, results_r3: List, filename: str = None):
    """Compara resultados com 2 features vs 3 features (projeção R3)."""
    if not results_r2 or not results_r3:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ['2 Features\n($x_1$, $x_2$)', '3 Features\n($x_1$, $x_2$, $x_1 \\cdot x_2$)']
    means = [np.mean([r['accuracy'] for r in results_r2]), np.mean([r['accuracy'] for r in results_r3])]
    stds = [np.std([r['accuracy'] for r in results_r2]), np.std([r['accuracy'] for r in results_r3])]

    bars = ax.bar(labels, means, yerr=stds, color=['steelblue', 'coral'],
                  edgecolor='black', capsize=5, alpha=0.8)
    ax.set_ylabel('Concordância LLM vs. Métrica')
    ax.set_title('Projeção R3: Efeito da Feature de Interação $x_3 = x_1 \\cdot x_2$',
                 fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{mean:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(results_perceptron: List, results_alternative: List, filename: str = None):
    """Compara W aprendido por Perceptron vs método alternativo em todas as fases (A, B, C)."""
    if not results_perceptron or not results_alternative:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: W scatter
    ax = axes[0, 0]
    w_perc = np.array([[r.w_aprendido[0], r.w_aprendido[1]] for r in results_perceptron])
    w_alt = np.array([[r.w_aprendido[0], r.w_aprendido[1]] for r in results_alternative])
    ax.scatter(w_perc[:, 0], w_perc[:, 1], c='steelblue', s=100, edgecolor='k',
               label='Perceptron Estruturado', zorder=3)
    ax.scatter(w_alt[:, 0], w_alt[:, 1], c='coral', s=100, edgecolor='k',
               label='Mínimos Quadrados (NNLS)', zorder=3)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('Ŵ_LLM: Perceptron vs. NNLS', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Fidelidade (Fase A)
    ax = axes[0, 1]
    fid_perc = [r.fidelidade_problema_a for r in results_perceptron]
    fid_alt = [r.fidelidade_problema_a for r in results_alternative]
    bp = ax.boxplot([fid_perc, fid_alt], labels=['Perceptron', 'NNLS'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.7)
    ax.set_ylabel('Fidelidade')
    ax.set_title('Fase A: Fidelidade (Métrica vs. LLM)', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Consistência Fase B
    ax = axes[1, 0]
    cons_b_perc = [r.consistencia_problema_b for r in results_perceptron]
    cons_b_alt = [r.consistencia_problema_b for r in results_alternative]
    kappa_b_perc = [r.kappa_problema_b for r in results_perceptron]
    kappa_b_alt = [r.kappa_problema_b for r in results_alternative]

    x_pos = np.array([0, 1, 3, 4])
    colors = ['steelblue', 'coral', 'steelblue', 'coral']
    labels_x = ['Perc.', 'NNLS', 'Perc.', 'NNLS']

    means = [np.mean(cons_b_perc), np.mean(cons_b_alt), np.mean(kappa_b_perc), np.mean(kappa_b_alt)]
    stds = [np.std(cons_b_perc) if len(cons_b_perc) > 1 else 0,
            np.std(cons_b_alt) if len(cons_b_alt) > 1 else 0,
            np.std(kappa_b_perc) if len(kappa_b_perc) > 1 else 0,
            np.std(kappa_b_alt) if len(kappa_b_alt) > 1 else 0]
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, edgecolor='k', capsize=5)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(['Consistência', 'Kappa'])
    ax.set_title('Fase B: Consistência no Problema B', fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend([bars[0], bars[1]], ['Perceptron', 'NNLS'], loc='upper right')

    # Plot 4: Consistência Fase C
    ax = axes[1, 1]
    cons_c_perc = [r.consistencia_problema_c for r in results_perceptron]
    cons_c_alt = [r.consistencia_problema_c for r in results_alternative]
    kappa_c_perc = [r.kappa_problema_c for r in results_perceptron]
    kappa_c_alt = [r.kappa_problema_c for r in results_alternative]

    means = [np.mean(cons_c_perc), np.mean(cons_c_alt), np.mean(kappa_c_perc), np.mean(kappa_c_alt)]
    stds = [np.std(cons_c_perc) if len(cons_c_perc) > 1 else 0,
            np.std(cons_c_alt) if len(cons_c_alt) > 1 else 0,
            np.std(kappa_c_perc) if len(kappa_c_perc) > 1 else 0,
            np.std(kappa_c_alt) if len(kappa_c_alt) > 1 else 0]
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, edgecolor='k', capsize=5)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(['Consistência', 'Kappa'])
    ax.set_title('Fase C: Consistência no Problema C', fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend([bars[0], bars[1]], ['Perceptron', 'NNLS'], loc='upper right')

    plt.suptitle('Comparação de Algoritmos de Otimização Inversa\n(Fases A, B e C)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dataset_overview(data: dict, seed: int, filename: str = None):
    """Visão completa dos 4 datasets: ground truth vs classificação LLM."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    colors_gt = {0: '#3498db', 1: '#e74c3c'}
    colors_llm = {0: '#2980b9', 1: '#c0392b'}

    problems = [
        ('A', data['X_a'], data['y_gt_a'], data.get('y_llm_a')),
        ('B', data['X_b'], data['y_gt_b'], data.get('y_llm_b')),
        ('C', data['X_c'], data['y_gt_c'], data.get('y_llm_c')),
        ('D', data['X_d'], data['y_gt_d'], None),
    ]

    for col, (name, X, y_gt, y_llm) in enumerate(problems):
        ax_gt = axes[0, col]
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

        ax_llm = axes[1, col]
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

    fig.suptitle(f'Visão Geral dos Datasets — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hits_and_errors(data: dict, seed: int, filename: str = None):
    """Acertos e erros da métrica vs LLM para Problemas A, B, C."""
    problems = [
        ('A', data['X_a'], data.get('y_llm_a'), data.get('y_metric_a')),
        ('B', data['X_b'], data.get('y_llm_b'), data.get('y_metric_b')),
        ('C', data['X_c'], data.get('y_llm_c'), data.get('y_metric_c')),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    learned_metric = data.get('learned_metric')

    for row, (name, X, y_llm, y_metric) in enumerate(problems):
        if y_llm is None or y_metric is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, 'Sem dados', ha='center', va='center',
                                    transform=axes[row, col].transAxes, fontsize=12, color='gray')
                axes[row, col].set_title(f'Problema {name}')
            continue

        # Tamanhos podem diferir (B/C excluem exemplos few-shot do teste)
        n_min = min(len(y_llm), len(y_metric), len(X))
        X_plot = X[:n_min]
        y_llm_plot = y_llm[:n_min]
        y_metric_plot = y_metric[:n_min]

        hits = y_llm_plot == y_metric_plot
        errors = ~hits

        # Col 1: Acertos (verde) e erros (vermelho)
        ax1 = axes[row, 0]
        ax1.scatter(X_plot[hits, 0], X_plot[hits, 1], c='#27ae60', s=15, alpha=0.5, label=f'Acerto ({hits.sum()})')
        ax1.scatter(X_plot[errors, 0], X_plot[errors, 1], c='#e74c3c', s=30, alpha=0.8, marker='x', label=f'Erro ({errors.sum()})')
        acc = hits.sum() / len(hits) * 100
        ax1.set_title(f'Problema {name}: Acertos/Erros ({acc:.1f}%)', fontweight='bold', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Col 2: Fronteira de decisão
        ax2 = axes[row, 1]
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

        # Col 3: Mapa de confiança com erros destacados
        ax3 = axes[row, 2]
        if learned_metric is not None:
            confidences, _ = compute_metric_confidence(X_plot, learned_metric.centroids, learned_metric.w)
            sc = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=confidences, cmap='viridis', s=15, alpha=0.6)
            ax3.scatter(X_plot[errors, 0], X_plot[errors, 1], c='red', s=50, marker='x', linewidths=2, label=f'Erros ({errors.sum()})')
            plt.colorbar(sc, ax=ax3, label='Margem')
        ax3.set_title(f'Problema {name}: Confiança + Erros', fontweight='bold', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    fig.suptitle(f'Acertos e Erros: Métrica vs LLM — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


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
    n_cols = 3 if has_nnls else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 2:
        axes = list(axes) + [None]

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

    # Perceptron
    plot_boundary(axes[0], X, y_llm, w_perc, centroids_perc, 'Perceptron Estruturado')

    # NNLS
    if has_nnls:
        plot_boundary(axes[1], X, y_llm, w_nnls, centroids_nnls, 'Mínimos Quadrados (NNLS)')

        # Barras comparativas
        ax3 = axes[2]
        x_pos = np.arange(2)
        width = 0.3
        bars1 = ax3.bar(x_pos - width/2, w_perc, width, label='Perceptron', color='#3498db', alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, w_nnls, width, label='NNLS', color='#e67e22', alpha=0.8)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['w₁ (x1)', 'w₂ (x2)'])
        ax3.set_ylabel('Peso')
        ratio_p = w_perc[0] / w_perc[1] if w_perc[1] > 0 else float('inf')
        ratio_n = w_nnls[0] / w_nnls[1] if w_nnls[1] > 0 else float('inf')
        ax3.set_title(f'Comparação de Pesos\nRatio w1/w2: Perc={ratio_p:.2f}, NNLS={ratio_n:.2f}',
                      fontweight='bold', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'NNLS não executado\n(RUN_ALGORITHM_COMPARISON=False)',
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=11, color='gray')
        axes[1].set_title('Mínimos Quadrados (NNLS)')

    fig.suptitle(f'Comparação de Algoritmos de Otimização Inversa — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices_detailed(data: dict, seed: int, filename: str = None):
    """Matrizes de confusão: LLM vs Métrica e LLM vs Ground Truth para cada problema."""
    problems = [
        ('A', data.get('y_llm_a'), data.get('y_metric_a'), data['y_gt_a']),
        ('B', data.get('y_llm_b'), data.get('y_metric_b'), data['y_gt_b']),
        ('C', data.get('y_llm_c'), data.get('y_metric_c'), data['y_gt_c']),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, (name, y_llm, y_metric, y_gt) in enumerate(problems):
        # Linha 1: LLM vs Métrica
        ax1 = axes[0, col]
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

        # Linha 2: LLM vs Ground Truth
        ax2 = axes[1, col]
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

    fig.suptitle(f'Matrizes de Confusão — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_margin_analysis_detailed(data: dict, seed: int, filename: str = None):
    """Análise detalhada de margens/confiança da métrica aprendida."""
    learned_metric = data.get('learned_metric')
    if learned_metric is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Dados do Problema A
    X_a = data['X_a']
    y_llm_a = data.get('y_llm_a')
    y_metric_a = data.get('y_metric_a')
    conf_a, _ = compute_metric_confidence(X_a, learned_metric.centroids, learned_metric.w)

    # Painel 1: Histograma de margens
    ax1 = axes[0, 0]
    ax1.hist(conf_a, bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(np.median(conf_a), color='red', linestyle='--', label=f'Mediana: {np.median(conf_a):.2f}')
    ax1.axvline(np.mean(conf_a), color='orange', linestyle='--', label=f'Média: {np.mean(conf_a):.2f}')
    ax1.set_xlabel('Margem')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Distribuição de Margens — Problema A', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Painel 2: Taxa de erro por faixa de margem
    ax2 = axes[0, 1]
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
        bars = ax2.bar(bin_centers, error_rates, width=(bins[1]-bins[0])*0.8, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
        for bar, count in zip(bars, bin_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'n={count}', ha='center', va='bottom', fontsize=8)
        ax2.set_xlabel('Margem')
        ax2.set_ylabel('Taxa de Erro (%)')
        ax2.set_title('Taxa de Erro por Faixa de Margem', fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # Painel 3: Scatter colorido por margem
    ax3 = axes[1, 0]
    sc = ax3.scatter(X_a[:, 0], X_a[:, 1], c=conf_a, cmap='viridis', s=20, alpha=0.7)
    ax3.scatter(*learned_metric.centroids[0], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2)
    ax3.scatter(*learned_metric.centroids[1], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2)
    plt.colorbar(sc, ax=ax3, label='Margem')
    ax3.set_title('Mapa de Confiança — Problema A', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Painel 4: Comparação de margens entre problemas (violin)
    ax4 = axes[1, 1]
    all_margins = [conf_a]
    labels = ['A']
    for pname, X_p in [('B', data['X_b']), ('C', data['X_c'])]:
        conf_p, _ = compute_metric_confidence(X_p, learned_metric.centroids, learned_metric.w)
        all_margins.append(conf_p)
        labels.append(pname)
    parts = ax4.violinplot(all_margins, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.5)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels([f'Problema {l}' for l in labels])
    ax4.set_ylabel('Margem')
    ax4.set_title('Distribuição de Margens por Problema', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Análise de Margens — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_experiment_summary_dashboard(data: dict, seed: int,
                                      results_abc: list, results_d: list,
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
        ('D', data['X_d'], data['y_gt_d']),
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

    # ─── BLOCO 4: Fase D ───
    seed_d = [r for r in results_d if r.random_seed == seed]
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
        ax_d1.set_title('Fase D: Learning Curve por Estratégia', fontweight='bold', fontsize=9)
        ax_d1.legend(fontsize=8)
    else:
        ax_d1.text(0.5, 0.5, 'Fase D não executada', ha='center', va='center', transform=ax_d1.transAxes, color='gray')
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


def print_phase_d_analysis(results_d: List[ResultadoPhaseDExperimento]):
    """Novo v3.0: Imprime análise detalhada dos resultados da Fase D."""
    print_section("ANÁLISE DA FASE D: LLM COMO APRENDIZ", "═")

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
        for r in results_d
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
CONCLUSÃO DA FASE D: LLM COMO APRENDIZ

Métrica do perito: W = [{EXPERT_W[0]:.2f}, {EXPERT_W[1]:.2f}]
(Pondera a dimensão x2 {EXPERT_W[1]/EXPERT_W[0]:.1f}x mais do que x1)

Linha de base zero-shot: {overall_zero:.1%}
Melhor resultado few-shot: {overall_best:.1%} (estratégia {best_config[1]}, {best_config[0]}-shot)
Melhoria geral: {overall_best - overall_zero:+.1%}

{'✓ O LLM CONSEGUE aprender com exemplos do perito — o desempenho melhora com mais exemplos.' if overall_best - overall_zero > 0.1 else
 '~ O LLM mostra aprendizado MODERADO com exemplos do perito.' if overall_best - overall_zero > 0.05 else
 '✗ O LLM NÃO melhora significativamente com os exemplos do perito.'}

A dificuldade dos exemplos importa:
- Exemplos fáceis (longe da fronteira): padrões claros, mas podem não generalizar
- Exemplos difíceis (próximos à fronteira): mais difíceis de aprender, mas melhor conhecimento da fronteira
- Exemplos mistos: representação balanceada
- Exemplos aleatórios: comparação com linha de base
""")


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
# EXECUÇÃO PRINCIPAL
# =============================================================================

def _run_phase_abc_experiment(X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                              n_shot, nome_0, nome_1, rep, provider, model_name,
                              temperature, seed, seed_idx, phase_a_cache, verbose):
    """Helper para executar um experimento A-C com cache."""
    cache_key = (seed, nome_0, nome_1)
    if cache_key in phase_a_cache:
        cached = phase_a_cache[cache_key]
        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail = run_complete_experiment(
            X_train_a.copy(), y_train_a.copy(),
            X_b.copy(), y_b.copy(), X_c.copy(), y_c.copy(),
            n_shot=n_shot, nome_classe_0=nome_0, nome_classe_1=nome_1,
            repeticao=rep, provider=provider, model_name=model_name,
            temperature=temperature, random_seed=seed,
            learned_metric_cache=cached['metric'],
            y_llm_train_a_cache=cached['y_llm'],
            fidelity_cache=cached['fidelity'],
            llm_accuracy_a_cache=cached['llm_acc'],
            n_malformed_a_cache=cached['n_malformed'],
            verbose=verbose
        )
    else:
        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail = run_complete_experiment(
            X_train_a.copy(), y_train_a.copy(),
            X_b.copy(), y_b.copy(), X_c.copy(), y_c.copy(),
            n_shot=n_shot, nome_classe_0=nome_0, nome_classe_1=nome_1,
            repeticao=rep, provider=provider, model_name=model_name,
            temperature=temperature, random_seed=seed,
            verbose=verbose
        )
        if learned_metric is not None:
            phase_a_cache[cache_key] = {
                'metric': learned_metric, 'y_llm': y_llm_train_a,
                'fidelity': fidelity, 'llm_acc': llm_acc_a, 'n_malformed': n_malformed
            }
    return result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail


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

    print_section("EXPERIMENTO: CONSISTÊNCIA DECISIONAL DE LLMs VIA OTIMIZAÇÃO INVERSA (v4.0)", "=")
    print("  Inclui: Fase D com múltiplos experts, diluição, inversão de classes,")
    print("          nomes semânticos de features, projeção R3, comparação de algoritmos")
    print(f"  Pasta de execução: {pasta_execucao}/")
    print(f"  Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────────────────────────────
    # FLAGS DE EXECUÇÃO ATIVAS
    # ─────────────────────────────────────────────────────────────────────
    flags_active = []
    if RUN_PHASES_ABC: flags_active.append("Fases A-C")
    if RUN_PHASE_D: flags_active.append("Fase D")
    if RUN_CLASS_ORDER_BIAS: flags_active.append("Viés de Ordem")
    if RUN_FEATURE_NAMES: flags_active.append("Nomes de Features")
    if RUN_DILUTION: flags_active.append("Diluição")
    if RUN_R3_EXPERIMENT: flags_active.append("Projeção R3")
    if RUN_MULTIPLE_EXPERTS: flags_active.append("Múltiplos Experts")
    if RUN_ALGORITHM_COMPARISON: flags_active.append("Comparação Algoritmos")

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
      Few-shot sizes: {FEW_SHOT_SIZES_PHASE_D}
      Example strategies: {EXAMPLE_STRATEGIES}
      Expert configs: {[c['name'] for c in EXPERT_CONFIGS] if RUN_MULTIPLE_EXPERTS else ['aniso_x2 (original)']}
      Problem D samples: {N_SAMPLES_PROBLEM_D}
    ═══════════════════════════════════════════════════
    """)

    all_results_abc = []
    all_results_d = []
    all_results_dilution = []
    all_results_abc_alternative = []  # Para comparação de algoritmos
    results_r3_2feat = []  # Resultados R3 com 2 features
    results_r3_3feat = []  # Resultados R3 com 3 features
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
                f"MODEL {model_idx + 1}/{len(MODELS_TO_TEST)}: {provider}/{model_name} (temp={temperature}) | "
                f"SEED {seed_idx + 1}/{len(RANDOM_SEEDS)}: {seed}",
                "═"
            )

            np.random.seed(seed)

            # ─── Passo 1: Geração dos dados sintéticos ─────────────────────────
            print(f"\n[PASSO 1] Gerando conjuntos de dados sintéticos com semente aleatória {seed}...", flush=True)

            X_a, y_a = create_problem_a(N_SAMPLES_PROBLEM_A, seed)
            X_b, y_b = create_problem_b(N_SAMPLES_PROBLEM_B, seed + 1)
            X_c, y_c = create_problem_c(N_SAMPLES_PROBLEM_C, seed + 2)
            X_d, y_d = create_problem_d(N_SAMPLES_PROBLEM_D, seed + 3)

            print(f"  Dados gerados: A={len(X_a)}, B={len(X_b)}, C={len(X_c)}, D={len(X_d)}")

            # Salva dados sintéticos na pasta de execução
            seed_data_dir = os.path.join(pasta_execucao, f"dados_sinteticos_seed{seed}")
            os.makedirs(seed_data_dir, exist_ok=True)
            for name, X_data, y_data in [("A", X_a, y_a), ("B", X_b, y_b), ("C", X_c, y_c), ("D", X_d, y_d)]:
                df = pd.DataFrame({"x1": X_data[:, 0], "x2": X_data[:, 1], "y": y_data})
                df.to_csv(os.path.join(seed_data_dir, f"problem_{name}.csv"), index=False)
            print(f"  Dados salvos em: {seed_data_dir}/")

            if seed_idx == 0 and model_idx == 0:
                print(f"\n[PASSO 2] Gerando visualizações iniciais...")

                visualize_all_problems(X_a, y_a, X_b, y_b, X_c, y_c,
                                       filename=os.path.join(pasta_execucao, "01_all_three_problems.png"))
                print(f"  Imagem salva: 01_all_three_problems.png")

                y_expert_d_viz = expert_classify(X_d, EXPERT_W, EXPERT_CENTROIDS)
                visualize_problem_d_with_expert(
                    X_d, y_d, y_expert_d_viz, EXPERT_W, EXPERT_CENTROIDS,
                    filename=os.path.join(pasta_execucao, "06_problem_d_expert.png")
                )
                print(f"  Imagem salva: 06_problem_d_expert.png")

                plot_phase_d_example_locations(
                    X_d, y_expert_d_viz, EXPERT_W, EXPERT_CENTROIDS,
                    n_examples=10, random_state=seed,
                    filename=os.path.join(pasta_execucao, "07_phase_d_example_strategies.png")
                )
                print(f"  Imagem salva: 07_phase_d_example_strategies.png")

            X_train_a, y_train_a = X_a, y_a

            # ═══════════════════════════════════════════════════════════════
            # FASES A-C (Experimento original de consistência)
            # ═══════════════════════════════════════════════════════════════

            if RUN_PHASES_ABC:
                phase_a_cache = {}

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
                            cache_key_ab = (seed, "A", "B")
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
                                    'X_d': X_d, 'y_gt_d': y_d,
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
                # TESTE DE INVERSÃO DE ORDEM DAS CLASSES (Sprint 3)
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

                # ═══════════════════════════════════════════════════════
                # TESTE DE NOMES SEMÂNTICOS NAS FEATURES (Sprint 4)
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
                            if len(np.unique(y_llm_feat)) >= 2:
                                centroids_feat = compute_centroids(X_train_a, y_llm_feat)
                                w_feat, gamma_feat = train_relaxed_perceptron(
                                    X_train_a, y_llm_feat, centroids_feat,
                                    eta=0.05, C=10.0, delta_gamma=0.05,
                                    max_epochs=50, tol=1e-4, verbose=False,
                                    use_best_effort=True
                                )
                                y_metric_feat = predict_with_metric(X_train_a, centroids_feat, w_feat)
                                fid_feat = accuracy_score(y_llm_feat, y_metric_feat)
                                print(f"    Ŵ_LLM com features {feat_0}/{feat_1}: [{w_feat[0]:.4f}, {w_feat[1]:.4f}]")
                                print(f"    Fidelidade: {fid_feat:.1%}")

                # ═══════════════════════════════════════════════════════
                # COMPARAÇÃO DE ALGORITMOS (Sprint 4)
                # ═══════════════════════════════════════════════════════
                if RUN_ALGORITHM_COMPARISON:
                    print(f"\n>>> Comparação de Algoritmos: Perceptron vs Mínimos Quadrados", flush=True)
                    cache_key_ab = (seed, "A", "B")
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
                            metrics_b_alt, y_llm_b_alt, y_pred_b_alt, llm_acc_b_alt, metric_acc_b_alt, n_malf_b_alt, euc_b_alt = \
                                phase_consistency_test(
                                    X_b, y_b, learned_metric_alt,
                                    "A", "B", 0, "FASE B — NNLS", verbose=(seed_idx == 0)
                                )

                            # Fase C com W do NNLS
                            print(f"    [NNLS] Rodando Fase C (Problema C)...", flush=True)
                            metrics_c_alt, y_llm_c_alt, y_pred_c_alt, llm_acc_c_alt, metric_acc_c_alt, n_malf_c_alt, euc_c_alt = \
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
                            )
                            all_results_abc_alternative.append(result_alt)

                            # Guarda W do NNLS para visualizações
                            if seed in seed_detailed_data:
                                seed_detailed_data[seed]['w_nnls'] = w_alt
                                seed_detailed_data[seed]['centroids_nnls'] = centroids_alt

                            print(f"    --- Comparação Fase A (Fidelidade) ---", flush=True)
                            print(f"    Perceptron: W=[{cached['metric'].w[0]:.4f}, {cached['metric'].w[1]:.4f}], Fidelidade={cached['fidelity']:.1%}")
                            print(f"    Mín. Quad.: W=[{w_alt[0]:.4f}, {w_alt[1]:.4f}], Fidelidade={fid_alt:.1%}")
                            print(f"    --- Comparação Fase B (Consistência) ---", flush=True)
                            # Busca resultado do Perceptron para mesma seed, n_shot=0, classes A/B
                            perc_b = [r for r in all_results_abc
                                      if r.random_seed == seed and r.n_shot == 0
                                      and r.nomes_classes == ("A", "B") and r.repeticao == 0]
                            if perc_b:
                                print(f"    Perceptron: Consistência B={perc_b[0].consistencia_problema_b:.1%}, Kappa={perc_b[0].kappa_problema_b:.3f}")
                            print(f"    Mín. Quad.: Consistência B={metrics_b_alt.accuracy:.1%}, Kappa={metrics_b_alt.cohen_kappa:.3f}")
                            print(f"    --- Comparação Fase C (Consistência) ---", flush=True)
                            if perc_b:
                                print(f"    Perceptron: Consistência C={perc_b[0].consistencia_problema_c:.1%}, Kappa={perc_b[0].kappa_problema_c:.3f}")
                            print(f"    Mín. Quad.: Consistência C={metrics_c_alt.accuracy:.1%}, Kappa={metrics_c_alt.cohen_kappa:.3f}")

            # ═══════════════════════════════════════════════════════════════
            # FASE D: LLM COMO APRENDIZ
            # ═══════════════════════════════════════════════════════════════

            if RUN_PHASE_D:
                # Determina quais configs de expert usar
                expert_configs_to_run = EXPERT_CONFIGS if RUN_MULTIPLE_EXPERTS else [EXPERT_CONFIGS[0]]

                for expert_cfg in expert_configs_to_run:
                    expert_w = expert_cfg["w"]
                    expert_name = expert_cfg["name"]
                    expert_centroids = EXPERT_CENTROIDS

                    print(f"\n[PASSO 4] Fase D — Expert: {expert_name} ({expert_cfg['desc']})")
                    print(f"  W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
                    print_section(f"FASE D: Expert {expert_name}", "═")

                    total_d_combos = len(FEW_SHOT_SIZES_PHASE_D) * len(EXAMPLE_STRATEGIES)
                    combo_count = 0
                    for n_shot_d in FEW_SHOT_SIZES_PHASE_D:
                        for strategy in EXAMPLE_STRATEGIES:
                            combo_count += 1
                            print(f"  [Fase D] Expert={expert_name}, n_shot={n_shot_d}, strategy={strategy} ({combo_count}/{total_d_combos}), seed={seed}", flush=True)
                            if n_shot_d == 0 and strategy != EXAMPLE_STRATEGIES[0]:
                                base_results = all_results_d[-N_REPETICOES:]
                                for rep, prev_result in enumerate(base_results):
                                    dup_result = ResultadoPhaseDExperimento(
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
                                    all_results_d.append(dup_result)
                                continue

                            for rep in range(N_REPETICOES):
                                is_verbose = (
                                    rep == 0 and seed_idx == 0 and
                                    (n_shot_d in [0, FEW_SHOT_SIZES_PHASE_D[-1]])
                                )

                                result_d = phase_d_llm_as_learner(
                                    X_d.copy(), y_d.copy(),
                                    expert_w=expert_w,
                                    expert_centroids=expert_centroids,
                                    n_shot=n_shot_d,
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
                                result_d.expert_name = expert_name
                                all_results_d.append(result_d)

                print(f"  ✓ Fase D concluída para seed={seed}", flush=True)

                # ═══════════════════════════════════════════════════════
                # EXPERIMENTO DE DILUIÇÃO (Sprint 3)
                # ═══════════════════════════════════════════════════════
                if RUN_DILUTION:
                    print(f"\n>>> Experimento de Diluição: 3 hard fixos + N easy progressivos", flush=True)
                    y_expert_dilution = expert_classify(X_d, EXPERT_W, EXPERT_CENTROIDS)
                    n_hard_fixed = 4  # 2 por classe (arredondado para par)
                    easy_additions = [0, 2, 4, 10, 16, 20]  # N easy adicionados

                    for dil_idx, n_easy in enumerate(easy_additions):
                        n_total = n_hard_fixed + n_easy
                        print(f"  [Diluição] {n_hard_fixed} hard + {n_easy} easy = {n_total} total ({dil_idx+1}/{len(easy_additions)})", flush=True)

                        examples_dil, selected_dil = select_examples_dilution(
                            X_d, y_expert_dilution, EXPERT_W, EXPERT_CENTROIDS,
                            n_hard_fixed=n_hard_fixed, n_easy_added=n_easy,
                            nome_classe_0="A", nome_classe_1="B",
                            random_state=seed, verbose=True
                        )

                        # Monta conjunto de teste
                        test_mask = np.ones(len(X_d), dtype=bool)
                        test_mask[selected_dil] = False
                        X_test_dil = X_d[test_mask]
                        y_expert_test_dil = y_expert_dilution[test_mask]
                        y_gt_test_dil = y_d[test_mask]

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

                            result_dil = ResultadoPhaseDExperimento(
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

            # ═══════════════════════════════════════════════════════════════
            # PROJEÇÃO R3: KERNEL QUADRÁTICO (Sprint 4 — Email do professor)
            # ═══════════════════════════════════════════════════════════════

            if RUN_R3_EXPERIMENT and seed_idx == 0:
                print(f"\n>>> Projeção R3: x3 = x1 * x2", flush=True)

                X_a_r3 = augment_to_r3(X_train_a)
                print(f"  X_a original: {X_train_a.shape} → X_a_r3: {X_a_r3.shape}")

                # Salva dados R3
                df_r3 = pd.DataFrame({"x1": X_a_r3[:, 0], "x2": X_a_r3[:, 1], "x3": X_a_r3[:, 2], "y": y_train_a})
                seed_data_dir = os.path.join(pasta_execucao, f"dados_sinteticos_seed{seed}")
                os.makedirs(seed_data_dir, exist_ok=True)
                df_r3.to_csv(os.path.join(seed_data_dir, "problem_A_r3.csv"), index=False)
                print(f"  Dados R3 salvos em: {seed_data_dir}/problem_A_r3.csv")

                # Fase A com 3 features
                y_llm_r3, n_malf_r3 = collect_llm_decisions(
                    X_train_a, "A", "B",
                    examples=None, verbose=True,
                    label_prefix="[R3] ",
                    nome_feature_0="x1", nome_feature_1="x2",
                    extra_features_matrix=X_a_r3[:, 2:],
                    extra_feature_names=["x3"]
                )

                if len(np.unique(y_llm_r3)) >= 2:
                    # Aprende W com 3 features
                    centroids_r3 = compute_centroids(X_a_r3, y_llm_r3)
                    w_r3, gamma_r3 = train_relaxed_perceptron(
                        X_a_r3, y_llm_r3, centroids_r3,
                        eta=0.05, C=10.0, delta_gamma=0.05,
                        max_epochs=50, tol=1e-4, verbose=True,
                        use_best_effort=True
                    )
                    y_metric_r3 = predict_with_metric(X_a_r3, centroids_r3, w_r3)
                    fid_r3 = accuracy_score(y_llm_r3, y_metric_r3)

                    print(f"  Ŵ_LLM (3 features): [{w_r3[0]:.4f}, {w_r3[1]:.4f}, {w_r3[2]:.4f}]")
                    print(f"  Fidelidade R3: {fid_r3:.1%}")

                    results_r3_3feat.append({'accuracy': fid_r3, 'w': w_r3, 'seed': seed})

                    # Teste de concordância: LLM com 2 features vs dados rotulados por W3
                    # (conforme Email 2 do professor)
                    y_llm_r2_only, n_malf_r2 = collect_llm_decisions(
                        X_train_a, "A", "B",
                        examples=None, verbose=True,
                        label_prefix="[R2 only] "
                    )
                    fid_r2_vs_r3 = accuracy_score(y_metric_r3, y_llm_r2_only)
                    print(f"  Concordância LLM(2 feat) vs Métrica(3 feat): {fid_r2_vs_r3:.1%}")
                    results_r3_2feat.append({'accuracy': fid_r2_vs_r3, 'seed': seed})

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZAÇÕES FINAIS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n[PASSO 5] Gerando visualizações finais...", flush=True)
    print_section("VISUALIZAÇÕES: FASES A-C", "═")

    if all_results_abc:
        plot_consistency_comparison_extended(all_results_abc, filename=os.path.join(pasta_execucao, "02_consistency_extended.png"))
        print(f"  Gráfico salvo: 02_consistency_extended.png")
        plot_class_names_effect(all_results_abc, filename=os.path.join(pasta_execucao, "03_class_names_effect.png"))
        print(f"  Gráfico salvo: 03_class_names_effect.png")

        if len(MODELS_TO_TEST) > 1:
            plot_model_comparison(all_results_abc, filename=os.path.join(pasta_execucao, "04_model_comparison.png"))
            print(f"  Gráfico salvo: 04_model_comparison.png")
        if len(RANDOM_SEEDS) > 1:
            plot_seed_comparison(all_results_abc, filename=os.path.join(pasta_execucao, "05_seed_comparison.png"))
            print(f"  Gráfico salvo: 05_seed_comparison.png")

        # NOVO: Distribuição de W (Sprint 2)
        plot_w_distribution(all_results_abc, filename=os.path.join(pasta_execucao, "10_w_distribution.png"))
        print(f"  Gráfico salvo: 10_w_distribution.png")

        # NOVO: Erros da métrica na Fase A (Sprint 2)
        for seed_key, data in phase_a_data_for_plots.items():
            fname = os.path.join(pasta_execucao, f"11_metric_errors_phase_a_seed{seed_key}.png")
            plot_metric_errors_phase_a(
                data['X'], data['y_llm'], data['y_metric'],
                data['w'], data['centroids'], filename=fname
            )
            print(f"  Gráfico salvo: 11_metric_errors_phase_a_seed{seed_key}.png")

        # NOVO: Viés de ordem das classes (Sprint 3)
        if RUN_CLASS_ORDER_BIAS:
            plot_class_order_bias(all_results_abc, filename=os.path.join(pasta_execucao, "15_class_order_bias.png"))
            print(f"  Gráfico salvo: 15_class_order_bias.png")

    print_section("VISUALIZAÇÕES: FASE D (LLM COMO APRENDIZ)", "═")

    if all_results_d:
        plot_phase_d_learning_curve(all_results_d, filename=os.path.join(pasta_execucao, "08_phase_d_learning_curve.png"))
        print(f"  Gráfico salvo: 08_phase_d_learning_curve.png")
        plot_phase_d_strategy_comparison(all_results_d, filename=os.path.join(pasta_execucao, "09_phase_d_strategy_comparison.png"))
        print(f"  Gráfico salvo: 09_phase_d_strategy_comparison.png")

    # NOVO: Experimento de diluição (Sprint 3)
    if all_results_dilution:
        plot_dilution_experiment(all_results_dilution, filename=os.path.join(pasta_execucao, "12_dilution_experiment.png"))
        print(f"  Gráfico salvo: 12_dilution_experiment.png")

    # NOVO: Comparação R3 vs R2 (Sprint 4)
    if results_r3_2feat and results_r3_3feat:
        plot_r3_comparison(results_r3_2feat, results_r3_3feat,
                          filename=os.path.join(pasta_execucao, "13_r3_vs_r2.png"))
        print(f"  Gráfico salvo: 13_r3_vs_r2.png")

    # NOVO: Comparação de algoritmos (Sprint 4)
    if all_results_abc_alternative and all_results_abc:
        # Filtra resultados do perceptron (zero-shot, classe A/B)
        perc_results = [r for r in all_results_abc
                       if r.n_shot == 0 and r.nomes_classes == ("A", "B")]
        if perc_results:
            plot_algorithm_comparison(perc_results, all_results_abc_alternative,
                                    filename=os.path.join(pasta_execucao, "14_algorithm_comparison.png"))
            print(f"  Gráfico salvo: 14_algorithm_comparison.png")

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZAÇÕES DETALHADAS POR SEED
    # ═══════════════════════════════════════════════════════════════════

    print_section("VISUALIZAÇÕES DETALHADAS POR SEED", "═")
    for seed_key, sdata in seed_detailed_data.items():
        print(f"\n  Gerando visualizações detalhadas para seed {seed_key}...")

        plot_dataset_overview(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"16_dataset_overview_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 16_dataset_overview_seed{seed_key}.png")

        plot_hits_and_errors(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"17_hits_errors_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 17_hits_errors_seed{seed_key}.png")

        plot_w_comparison_algorithms(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"18_w_algorithms_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 18_w_algorithms_seed{seed_key}.png")

        plot_confusion_matrices_detailed(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"20_confusion_matrices_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 20_confusion_matrices_seed{seed_key}.png")

        plot_margin_analysis_detailed(sdata, seed_key,
            filename=os.path.join(pasta_execucao, f"21_margin_analysis_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 21_margin_analysis_seed{seed_key}.png")

        plot_experiment_summary_dashboard(sdata, seed_key, all_results_abc, all_results_d,
            filename=os.path.join(pasta_execucao, f"19_experiment_dashboard_seed{seed_key}.png"))
        print(f"  Gráfico salvo: 19_experiment_dashboard_seed{seed_key}.png")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    if all_results_abc:
        print_final_analysis(all_results_abc)
    if all_results_d:
        print_phase_d_analysis(all_results_d)

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
                'gamma': r.gamma_otimo,
                'n_disagreements_b': r.n_disagreements_b, 'n_disagreements_c': r.n_disagreements_c,
                'n_malformed_responses': r.n_malformed_responses,
                'euclidean_consistency_b': r.consistencia_euclidiana_problema_b,
                'euclidean_consistency_c': r.consistencia_euclidiana_problema_c,
                'diagonal_limitation_flag': r.diagonal_limitation_flag,
            }
            for r in all_results_abc
        ])
        filename_abc = os.path.join(pasta_execucao, f"results_phases_abc_v4_{timestamp_csv}.csv")
        df_abc.to_csv(filename_abc, index=False)
        print(f"  Resultados das Fases A-C salvos em: {filename_abc}")

    if all_results_d:
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
            for r in all_results_d
        ])
        filename_d = os.path.join(pasta_execucao, f"results_phase_d_v4_{timestamp_csv}.csv")
        df_d.to_csv(filename_d, index=False)
        print(f"  Resultados da Fase D salvos em: {filename_d}")

    if all_results_abc_alternative:
        df_alt = pd.DataFrame([
            {
                'algorithm': 'NNLS',
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'n_shot_phase_bc': r.n_shot,
                'class_0': r.nomes_classes[0], 'class_1': r.nomes_classes[1],
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
                'gamma': r.gamma_otimo,
                'n_disagreements_b': r.n_disagreements_b, 'n_disagreements_c': r.n_disagreements_c,
                'n_malformed_responses': r.n_malformed_responses,
                'euclidean_consistency_b': r.consistencia_euclidiana_problema_b,
                'euclidean_consistency_c': r.consistencia_euclidiana_problema_c,
                'diagonal_limitation_flag': r.diagonal_limitation_flag,
            }
            for r in all_results_abc_alternative
        ])
        filename_alt = os.path.join(pasta_execucao, f"results_algorithm_comparison_v4_{timestamp_csv}.csv")
        df_alt.to_csv(filename_alt, index=False)
        print(f"  Resultados NNLS (Fases A-C) salvos em: {filename_alt}")

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
        filename_dil = os.path.join(pasta_execucao, f"results_dilution_v4_{timestamp_csv}.csv")
        df_dilution.to_csv(filename_dil, index=False)
        print(f"  Resultados da Diluição salvos em: {filename_dil}")

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

    return all_results_abc, all_results_d


if __name__ == "__main__":
    results_abc, results_d = main()