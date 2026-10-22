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
    Fonseca Neto, CILAMCE 2017; eta=0.001, C=1.0 — no intervalo [0,1; 1] do artigo).
  - NNLS (Lawson-Hanson 1974, scipy.optimize.nnls; sem hiperparâmetros).

=====================================================
ORGANIZAÇÃO EM 3 BLOCOS AUTO-CONTIDOS (refactor 14/05/2026)
=====================================================

BLOCO 1 — LLM como ESPECIALISTA (otimização inversa)
  Problemas:
    A — linear sintético (centroides (−2, −0,5) e (2, 0,5) desde 15/10/2026; std=1.2)
    B — linear, centroides FIXOS (−2, 1,5) e (2, −1,5): rotação HORÁRIA de 2,0
        em relação ao A
    C — linear, centroides FIXOS (−2, −1,5) e (2, 1,5): rotação ANTI-HORÁRIA de 1,0
        em relação ao A (direção oposta à de B)
    D — meia-lua sintética (sklearn.make_moons, noise=0.15) [não-linear]
  Experimentos: validação do oráculo (sobre E3, E1 e E2), Problema A, Passos 2 a 6
    (W via Perc+NNLS), Problemas B e C, Passos 2 a 4 (transferência), R2 -> R3 -> R4
    no Problema A e na meia-lua (Problema D, Passos 2 a 6),
    vieses de classe, variantes de prompt, nomes semânticos.

BLOCO 2 — LLM como APRENDIZ (especialista externo via in-context learning)
  Problemas:
    E — gaussianas lineares (centroides ±1.5/±1.0, std=1.3) em três subproblemas:
        E1 (especialista W=[0.3, 1.5], principal), E2 (W=[1.5, 0.3]), E3 (W=[1, 1]).
    F — meia-lua sintética com especialista = ground truth do make_moons.
  Experimentos: Problema E, Passos 2 a 6 (estratégias easy/hard/mixed/random,
    múltiplos especialistas, diluição, recency bias), Problema F, Passos 2 a 4,
    baselines clássicos (k-NN, LR, SVM), LLM vs Perceptron baseline.

BLOCO 3 — Estudo de caso REAL (peso × altura, Problema G)
  Base do orientador (100 pessoas), classificador ótimo bayesiano com fronteira
    ELÍPTICA: f(x1,x2) = x2^2 - x2 + x1^2 - x1 + cte. Dois subproblemas com as
    mesmas pessoas e a mesma divisão 70/30: G1 (valores normalizados) e G2 (prompt
    em kg e m; métrica estimada no espaço normalizado).
  Experimentos: Problema G, Passos 2 a 7: métrica em R2/R3/R4 (R4 = elipse), acurácia
    vs rótulo REAL, paradoxo do overfitting, aprendizado em contexto, LLM vs Perceptron.

=====================================================
NOMENCLATURA POR PROBLEMA E PASSO (13/10/2026)
=====================================================

Letra = problema; cada problema tem a SUA sequência de passos, só com os passos que
existem nele. Desde 13/10/2026 os identificadores do código, as colunas e os valores dos
CSVs, as tags da auditoria e os nomes de arquivo também seguem essa nomenclatura
(mapa em renomeacao_passos_2026-10-13.md; execuções anteriores são migradas por
src/renomear_assets_legado.py).

Problema A (linear, Bloco 1): 1 gerar os 150 pontos; 2 LLM rotula em zero-shot;
         3 LLM rotula com k exemplos do rótulo real (k = 4, 10, 20, 40, 3 sorteios);
         4 estimar Ŵ (Perceptron e NNLS, R²/R³/R⁴) e medir fidelidade; 5 teto (W
         ajustado ao rótulo real); 6 eleger a melhor métrica para a transferência.
Problemas B e C (rotação horária de 2,0 / anti-horária de 1,0 em relação ao A, Bloco 1): 1 gerar os 100 pontos;
         2 LLM rotula em zero-shot; 3 aplicar cada métrica do A (só W, centróides
         locais) e medir consistência, κ e F1; 4 LLM rotula com k exemplos rotulados
         pela métrica do A (H2).
Problema D (meia-lua, LLM como especialista, Bloco 1): Passos 1 a 5 como o A, sem
         split, em R²/R³/R⁴ (adg_passos_2_a_5, flag RUN_PROBLEMA_D); 6 SVM RBF
         nos rótulos do LLM.
Problemas E1, E2, E3 (especialista linear, LLM como aprendiz, Bloco 2): 1 o
         especialista (W conhecido) rotula; 2 LLM em zero-shot; 3 LLM com k exemplos
         do especialista por estratégia (e_passos_2_3_aprendiz); 4 diluição; 5 ordem
         dos exemplos; 6 baselines clássicos.
Problema F (meia-lua como aprendiz, Bloco 2): 1 gerar e dividir 70/30; 2 LLM rotula
         o treino em zero-shot (métrica só para fidelidade); 3 LLM com k exemplos do
         rótulo real classifica o teste; 4 Perceptron nos mesmos exemplos
         (fg_executar(problem_name=F_meia_lua_seed{N})).
Problemas G1 e G2 (peso × altura, Bloco 3): 1 carregar e dividir 70/30; 2 zero-shot
         no treino; 3 k exemplos do rótulo real; 4 métrica em R²/R³/R⁴ e teto; 5 eleger
         a melhor métrica; 6 LLM com k exemplos rotulados pela métrica eleita classifica
         o teste; 7 Perceptron nos mesmos exemplos.
Oráculo sobre E3, E1 e E2: 1 rotular pelo W conhecido; 2 Perceptron e NNLS; 3 comparar.

=====================================================
PROBLEMA DIRETO vs. PROBLEMA INVERSO
=====================================================

PROBLEMA DIRETO: dado classificador conhecido, classificar novos pontos.
  -> LLM faz isso naturalmente (zero/few-shot).

PROBLEMA INVERSO: dadas as classificações do LLM, INFERIR o critério W.
  -> Estimamos W via Perceptron Estruturado E NNLS em paralelo (robustez ao
     método de estimação; cosseno entre os W's reportado por execução).
  -> Aplicamos W a outros problemas para ver se o LLM se repete.

Mahalanobis DIAGONAL: O(d) parâmetros vs O(d^2) na cheia. Simplificação
  deliberada; próximo passo é evoluir para forma SDP parametrizada mais
  geral (orientador, Reunião 1).

=====================================================
HIPÓTESES DO EXPERIMENTO
=====================================================

H1 (Consistência) — Bloco 1: o LLM mantém critério decisório implícito em
  problemas distintos. W estimada em A prevê classificações em B/C com
  Cohen's κ > 0.4 (piso da faixa "moderada" de Landis & Koch 1977 — limiar do artigo;
  testa a EXISTÊNCIA de um critério consistente, não seu domínio).

H2 (Few-shot amplifica) — Bloco 1: exemplos rotulados pela métrica
  aumentam a concordância LLM-métrica.

H3 (LLM como aprendiz) — Bloco 2: via in-context learning, o LLM reproduz
  o critério de um especialista externo. A acurácia (LLM vs especialista)
  cresce com k.

H4 (Exemplos difíceis) — Bloco 2: hipótese a priori "hard > easy". O veredicto
  é calculado por execução a partir dos dados (print_e_passos_2_3, log e CSVs).

H5 (Estabilidade) — Transversal: comportamento reproduzível entre 3 seeds
  (bases distintas); onde há sorteio de exemplos, 3 repetições por seed
  (regra única em `reps_para`).

=====================================================
DESIGN DE VARIABILIDADE
=====================================================

- N_REPETICOES=3: repetições SÓ onde há sorteio de exemplos few-shot (Passo 3 de A, D e G;
  Problema E, Passo 3, estratégia 'random'; Passo 3 do F e Passo 6 do G; random_state = seed + rep); seleções determinísticas e
  zero-shot rodam 1 coleta. O flip a T=0 é medido pela auditoria offline.
- RANDOM_SEEDS=[42, 123, 7]: BASES distintas.
- NOMES_CLASSES variados: detecta viés semântico (Bloco 1).
- Temperatura 0.0 REDUZ mas não elimina estocasticidade. Motivos:
    batching/arredondamento, hardware heterogêneo, atualizações silenciosas
    do provider, top-k residual, softmax paralelo não-associativo.

=====================================================
RECURSOS PRINCIPAIS (mapeados aos blocos)
=====================================================
- BLOCO 1: Problema A, Passos 2 a 6 (otim. inversa), B e C, Passos 2 a 4 (transferência), validação do oráculo,
  R2 -> R3 -> R4 em lineares + meia-lua, vieses, prompts.
- BLOCO 2: Problema E, Passos 2 a 6 (estratégias, múltiplos especialistas,
  diluição, recency bias), Problema F, Passos 2 a 4 (meia-lua), baselines clássicos,
  LLM vs Perceptron baseline.
- BLOCO 3: peso x altura (Problema G, Passos 2 a 7: zero-shot, k exemplos do rótulo real,
  métrica R2/R3/R4 e teto, desde 13/10/2026; paradoxo do overfitting; aprendizado em
  contexto com o especialista eleito entre todas as métricas; LLM vs Perceptron).
- Comum: média e faixa entre sementes, diferenças pareadas por semente, Cohen's d (sem bootstrap nem Wilcoxon desde 13/10/2026), concorrência
  asyncio.Semaphore(MAX_CONCURRENCY=50), parser 8 camadas (0-7) + fallback MD5, 3 seeds
  (x 3 reps onde há sorteio), centenas de PNGs (combinados + painéis) e 14 CSVs
  consolidados + 1 final_cross_linearity__<alias>.csv por modelo, nomeados com
  prefixos bloco{1,2,3}_, bloco23_, final_ e sufixo __<alias> nos derivados de LLM.

PROVEDOR: OpenRouter, único (todos os modelos, inclusive os da OpenAI, via
  id `<org>/<modelo>`; pin de infraestrutura por modelo em llm_client.MODEL_PROVIDER_PIN).
"""
import os
from dotenv import load_dotenv
load_dotenv()


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


from sklearn.metrics import accuracy_score
from classical_baselines import ClassicalBaselineRunner

import time
import asyncio
import contextlib
import io
import json
import relaxed_perceptron as _rp   # ULTIMA_MARGEM_MINIMA (15/10/2026)
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
from llm_client import (
    PROVIDER_CONFIG, get_async_client, get_extra_body,
    CreditosEsgotadosError, erro_de_credito, consultar_saldo_openrouter,
)
from data_problems import (
    PROBLEM_B_CENTERS,
    create_problem_a, create_problem_b, create_problem_c, create_problem_e_expert,
    create_problem_d_meialua, create_problem_homem_mulher, create_problem_homem_mulher_kg_m,
)
from plots import (  # reexport: extraído na etapa 1 da modularização, 2026
    _rotulo_fg,
    plot_a_passo4_perceptron_vs_nnls, plot_vies_nomes_classe, plot_vies_ordem_classes,
    plot_e_passo6_baselines, plot_final_matrizes_confusao, plot_bc_passo3_consistencia, plot_final_dados,
    plot_e_passo4_diluicao, plot_e_passo5_ordem, plot_final_painel_resumo, plot_fg_fronteiras,
    plot_fg_passos_2_e_4_r2_r3_r4, plot_fg_aprendiz_curva, plot_vies_nomes_atributo, plot_a_passo4_gamma,
    plot_final_acertos_erros, plot_fg_rotulos_llm, plot_final_margem, plot_d_passo6_svm,
    plot_a_passo4_erros, plot_final_comparacao_modelos, plot_d_passo5_teto,
    plot_oraculo_recuperacao_w, plot_e_passo3_exemplos, plot_e_passo3_curva, plot_fg_llm_vs_perceptron,
    plot_e_passo3_estrategias, plot_problema_passo1, plot_vies_variantes_prompt, plot_a_passo4_r2_r3_r4,
    plot_bc_passo3_sementes, plot_final_w_algoritmos, plot_a_passo4_w_distribuicao, plot_abc_passo1,
    plot_e_passo1,
    plot_rotulos_simbolos, plot_passos_2_a_5_curva, plot_g_passos_2_a_5_curva,
)
from relatorios import (  # reexport: extraído na etapa 1 da modularização, 2026
    print_a_passo4_erros_por_margem, print_e_passo5_ordem,
    print_abc_resumo, print_a_passo4_sensibilidade_perceptron, print_e_passos_2_3, print_section,
    print_sumario_estatistico, sintese_r2_r3_r4,
)
from execucao_io import (
    iniciar_cronometro,
    com_cor,  # reexport: extraído na etapa 1 da modularização, 2026
    # MODEL_ALIAS e _model_slug são reexportados aqui para os testes (dm.*).
    LLM_INTERACTIONS, LOG_CHUNK_LIMIT_BYTES, MODEL_ALIAS, RESULTADOS_DIR, Tee,
    _model_alias, _model_slug, asset_variant, checkpoint_interactions,
    llm_asset, salvar_json_em_chunks, salvar_log_em_chunks,
)
from resultados import (  # reexport: extraído na etapa 1 da modularização, 2026
    LearnedMetric, ResultadoABC, ResultadoAprendiz,
)

# Reexports consumidos pelos testes de outros módulos (test_metrics.py usa dm.d_W;
# test_e_passo3_exemplos.py usa dm.MODEL_ALIAS/_model_slug/asset_variant). Declarar em
# __all__ torna o uso explícito para o pyflakes sem depender de "# noqa".
__all__ = ["d_W", "MODEL_ALIAS", "_model_slug", "asset_variant"]
from protocolo import (  # reexport: constantes compartilhadas entre módulos
    EXAMPLE_STRATEGIES, EXPERT_CENTROIDS, EXPERT_W, PERCEPTRON_PARAMS,
    MIN_PONTOS_CLASSE_MINORITARIA, colapsou,
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

# Formato: (provider, model, temperature). Até 09/10/2026 havia um 4º campo `scope`
# ("full"/"core") e as seeds extras EXTRA_SEEDS_CORE; nenhum modelo usava "core" e as
# seeds extras estavam vazias, então o conceito foi removido: TODOS os modelos rodam
# o grid completo nas mesmas RANDOM_SEEDS.
MODELS_TO_TEST = [
    # Protocolo: 2 modelos, tratados de forma IGUAL: cada um gera o conjunto
    # completo de gráficos na raiz da execução, com seu alias (MODEL_ALIAS) no
    # nome do asset. CSVs consolidam ambos (colunas provider/model).
    # Provedor único: OpenRouter (id do modelo no formato <org>/<modelo>).
    ("openrouter", "openai/gpt-4o-mini", 0.0),
    ("openrouter", "google/gemini-2.5-flash-lite", 0.0),
]

# PROVIDER_CONFIG e get_async_client → llm_client.py (importados no topo).
# Ver tests/test_llm_client.py.


# Cliente assíncrono (OpenRouter). Desde 09/10/2026 fica None em produção: cada
# coleta (`async_collect_llm_decisions`) cria o seu cliente DENTRO do laço de eventos
# de `asyncio.run` e o fecha no fim da corrotina, com o laço ainda aberto. Antes, um
# cliente por modelo atravessava vários `asyncio.run`: as conexões ficavam presas ao
# laço já fechado e `AsyncClient.aclose()` estourava "Event loop is closed" no log.
# Um valor não-None aqui é ponto de injeção (dublês nos testes): é usado como está e
# NÃO é fechado pela coleta.
async_client = None
MODEL_NAME = None
CURRENT_PROVIDER = None
# Temperatura 0.0: REDUZ mas NÃO ELIMINA estocasticidade do LLM.
# Mesmo com temp=0, variabilidade pode ocorrer por: batching em GPU (arredondamentos float16),
# hardware heterogêneo entre requests, atualizações silenciosas do modelo pelo provider,
# e paralelismo não-determinístico em operações de ponto flutuante (softmax não-associativa).
# Por isso o experimento usa múltiplas seeds (RANDOM_SEEDS) e quantifica o flip a
# T=0 na auditoria offline (src/audit_interactions.py); as repetições (N_REPETICOES)
# variam o SORTEIO dos exemplos few-shot, não a consulta (ver `reps_para`).
CURRENT_TEMPERATURE = 0.0
# Tag de auditoria gravada em cada registro de LLM_INTERACTIONS ("bloco1|problema=D|
# passo=3|regime=passo3_exemplos_rotulo_real|k=10|rep=1|seed=42"): permite à auditoria offline
# separar malformadas/flip por regime. Definida pelo runner; vazia fora dele.
CURRENT_TAG = ""


# Bloco 1 (A) e Bloco 2 (E) usam mais amostras — envolvem aprendizado.
# Problemas B e C (testes de consistência) precisam só de 100 pontos para Cohen's κ/F1-Macro confiáveis.
N_PONTOS_A = 150  # Bloco 1: treino da métrica via otim. inversa
N_PONTOS_B = 100  # Bloco 1: teste de consistência (centróides fixos; rotação horária de 2,0 em relação ao A)
N_PONTOS_C = 100  # Bloco 1: teste de consistência (centróides fixos; rotação anti-horária de 1,0 em relação ao A)
N_PONTOS_E = 150  # Bloco 2: especialista linear (Problema E)


# Meia-lua (Bloco 1 = Problema D; Bloco 2 = Problema F) usa N_PONTOS_A=150 também
K_EXEMPLOS_APRENDIZ = [0, 4, 10, 20, 40]  # versão completa (pares: balanço exato)
N_REPETICOES = 3

# Pontos de ancoragem dos experimentos de viés (inversão de nomes de classe e
# variantes de prompt): mesmos k_exemplos para que os efeitos sejam comparáveis entre si.
K_VIESES = [0, 10]

# Grades dos experimentos auxiliares (reduzidas no --rapido):
DILUTION_EASY_ADDITIONS = [0, 2, 4, 10, 16, 20]  # N easy adicionados na diluição
K_ORDEM_EXEMPLOS = [4, 10, 20]              # valores_k do viés de ordem

# Passos 2 a 5 dos Problemas A e D (orientador, 01/10/2026): estimados em
# zero-shot, em few-shot com k exemplos ROTULADOS PELO GROUND TRUTH (sorteio aleatório
# balanceado, k/2 por classe, random_state = seed + rep, sem split: o LLM classifica os
# 150 − k pontos restantes) e com W ajustado diretamente ao ground truth (teto do
# estimador). O LLM vê sempre só x1, x2; a augmentação R3/R4 entra só na estimação.
K_EXEMPLOS_ROTULO_REAL = [4, 10, 20, 40]          # k do Passo 3 (A, D e G): exemplos do rótulo real; --rapido → [4]
K_EXEMPLOS_METRICA_BC = [0, 4, 10, 20, 40]       # B e C, Passo 4: k dos exemplos rotulados pela métrica do A (H2; pares: balanço exato de classes); CLI --bc-zero-shot → [0]
ATRIBUTOS_PASSO4 = {"A": (2, 3, 4), "D": (2, 3, 4), "G": (2, 3, 4)}   # n_features do Passo 4 (R², R³, R⁴)
# G, Passo 6 (14/10/2026, Metodologia IV.7): nº de atributos MOSTRADOS ao LLM no prompt do aprendizado em
# contexto, fator controlado e independente da métrica eleita: 2 = x1, x2; 3 = + x1·x2; 4 = + x1², x2².
# Os extras só acompanham exemplos (k > 0); o zero-shot (k = 0) é uma coleta só. O Problema F usa só 2.
ATRIBUTOS_PROMPT_PASSO6_G = (2, 3, 4)

# Múltiplas sementes aleatórias para garantir robustez dos resultados (protocolo
# uniforme: as mesmas 3 seeds em todos os experimentos e modelos; decisão 02/07/2026).
RANDOM_SEEDS = [42, 123, 7]


def reps_para(k_exemplos: int, sorteio_estocastico: bool = True) -> int:
    """Regra única de repetição do protocolo.

    Repetições existem para variar o SORTEIO dos exemplos few-shot (variância de
    amostragem): Passo 3 de A, D e G (k/2 por classe do rótulo real), estratégia 'random'
    do Passo 3 de E, Passo 3 do F e Passo 6 do G. Onde não há sorteio (zero-shot, ou
    seleção determinística por margem: B e C, Passo 4; easy/hard/mixed; diluição;
    ordenações) roda-se 1 coleta; o não-determinismo por consulta
    do LLM (flip a T=0) é quantificado pela auditoria offline
    (src/audit_interactions.py), com n ordens de magnitude maior que N_REPETICOES.
    """
    if k_exemplos == 0 or not sorteio_estocastico:
        return 1
    return N_REPETICOES


# `colapsou` (regra única de colapso) vive em protocolo.py desde 13/10/2026 e é
# reexportada daqui (`dissertacao_mestrado.colapsou`) para runner, plots e testes.

# Controle de limite de requisições (rate limit)
MAX_RETRIES = 5
INITIAL_BACKOFF = 10
# Timeouts (15/10/2026): um timeout de API_TIMEOUT_S não é sinal de capacidade do provedor
# e NÃO entra no backoff exponencial dos 429/5xx. Uma chamada que estoura o tempo é
# retentada no máximo MAX_RETRIES_TIMEOUT vezes (contagem de chamadas que terminaram em
# timeout, como MAX_RETRIES conta chamadas), com espera FIXA de ESPERA_TIMEOUT_S entre
# elas; ao esgotar, o ponto cai no fallback já existente (erro registrado + MD5). Antes,
# cada ponto que estourava custava 5 × 90 s + ~280 s de backoff (5 pontos da variante
# cot do Gemini ≈ 61 min na completa de 15/10/2026); agora ≤ 2 × 90 s + 5 s.
MAX_RETRIES_TIMEOUT = 2
ESPERA_TIMEOUT_S = 5.0
# Teto de espera por UMA requisição (segunda barreira, além do timeout do SDK). O SDK
# do OpenAI usa 60 s por tentativa e até 3 tentativas próprias (~180 s no pior caso);
# o teto aqui vale por chamada ao SDK e, ao estourar, vira erro RETENTÁVEL (backoff
# do laço de rate limit), não resposta vazia (até 08/10 consumia as 5 retentativas de
# FORMATO e caía no fallback MD5 sem registrar o erro).
API_TIMEOUT_S = 90.0

# Concorrência máxima para chamadas async à API (OpenRouter não publica limite para
# contas pagas; o teto prático é do provedor de inferência pinado).
# Figuras em processos paralelos (15/10/2026): as figuras por semente (fechamento, símbolos de
# A/B/C/D) e os 300+ scatters ponto-a-ponto (final_graphic_07) são independentes; cada uma é
# desenhada e salva (cor + cinza) num processo filho (`spawn`, seguro no macOS) e o texto que
# ela imprimiria volta para o log. `FIGURAS_PROCESSOS` = nº de processos; 1 desliga o paralelismo.
FIGURAS_PROCESSOS = max(1, min(8, (os.cpu_count() or 2) - 2))
MAX_CONCURRENCY = 50  # benchmark 08/10/2026: 10→13-15 chamadas/s; 30→21-39/s; 50→45-58/s, sem 429 (OpenAI e Google via OpenRouter); 30 rodou a completa de 08/10 em 2h23 sem 429

# Máximo de retentativas para respostas malformadas do LLM
MAX_FORMAT_RETRIES = 5

# ── Guarda de saldo (08/10/2026) ─────────────────────────────────────────────
# Custo medido no OpenRouter (prompts numéricos tokenizam a ~2 caracteres/token):
# ~US$ 0,09 por milhão de caracteres de prompt. A completa de 08/10/2026 (158 mil chamadas,
# ≈ 105 M caracteres) custou ≈ US$ 8–10; com o Passo 3 do G (13/10/2026) e o Passo 6 do G com
# 3 e 4 atributos (14/10/2026) a estimativa da completa é a soma dos blocos abaixo (17,0).
# Valores por bloco (2 modelos, 3 sementes), com folga.
CUSTO_ESTIMADO_USD = {
    "completa": 17.0,   # todos os blocos (= soma dos blocos abaixo)
    "bloco1": 3.5,      # A/B/C + Passos 2 a 5 de A e D + vieses (59 mil chamadas)
    "bloco2": 5.5,      # Problema E (E1/E2/E3, Passos 2 a 5), F (77 mil chamadas)
    "bloco3": 8.0,      # G1 + G2 (22 mil chamadas dos Passos 2 e 6 + ~30 mil do Passo 3
                        # com exemplos do rótulo real, 13/10/2026: 8 variantes × Σ(70−k) × 3 reps × 3
                        # sementes × 2 modelos, prompts mais longos; + ~35 mil do Passo 6 com 3 e 4
                        # atributos no prompt, 14/10/2026: 8 variantes × 2 dimensões × 4 k × 3 reps × 30 × 3 × 2)
    "oraculo": 0.0,     # 0 chamadas
}
CUSTO_ESTIMADO_SMOKE_USD = 0.6   # --rapido: ~23 mil chamadas curtas (medido: US$ 0,25)
MARGEM_SALDO = 1.2               # exige saldo ≥ estimativa × margem


def estimar_custo_usd(apenas, modo_rapido: bool) -> float:
    """Estimativa de custo (USD) da execução conforme --apenas e --rapido.

    Uma seleção sem chamadas pagas (`--apenas oraculo`) custa 0 também no smoke:
    sem isso, `--apenas oraculo --rapido` exigia chave e saldo para 0 chamadas.
    """
    if apenas:
        cheia = sum(CUSTO_ESTIMADO_USD[b] for b in set(apenas))
        if cheia == 0:
            return 0.0
        return CUSTO_ESTIMADO_SMOKE_USD if modo_rapido else cheia
    return CUSTO_ESTIMADO_SMOKE_USD if modo_rapido else CUSTO_ESTIMADO_USD["completa"]


def verificar_saldo(saldo_info, estimativa: float, ignorar: bool = False) -> str:
    """Compara saldo com a estimativa; levanta SystemExit se não cobrir (salvo `ignorar`).

    Devolve a linha de resumo impressa no log. `saldo_info` é o dict de
    `consultar_saldo_openrouter` ou None (consulta falhou → só avisa).
    """
    if saldo_info is None:
        return (f"  ⚠ Saldo do OpenRouter desconhecido; custo estimado US$ {estimativa:.2f}. "
                "Prosseguindo sem verificação.")
    saldo = saldo_info["saldo"]
    linha = (f"  Saldo OpenRouter: US$ {saldo:.2f} (créditos {saldo_info['total']:.2f}, "
             f"usados {saldo_info['usado']:.2f}) | custo estimado desta execução: "
             f"US$ {estimativa:.2f} (exigido ×{MARGEM_SALDO:.1f} = US$ {estimativa * MARGEM_SALDO:.2f})")
    if saldo < estimativa * MARGEM_SALDO:
        if ignorar:
            return linha + "\n  ⚠ --ignorar-saldo: saldo abaixo do exigido, prosseguindo por sua conta."
        raise SystemExit(
            linha + "\nERRO: saldo insuficiente. Recarregue ao menos "
            f"US$ {estimativa * MARGEM_SALDO - saldo:.2f} no OpenRouter ou use --ignorar-saldo. "
            "Um 402 no meio da coleta aborta a execução (ver CreditosEsgotadosError)."
        )
    return linha


# Contexto para abortar com segurança (pasta e Tee do log), preenchido por main().
_ABORT_CONTEXT: dict = {}


def _abortar_por_creditos(exc: BaseException) -> None:
    """Salva checkpoint das interações e o log parcial, depois encerra (SystemExit 3).

    SystemExit atravessa os `except Exception` dos blocos de main(): nada mais é
    coletado e nenhum rótulo de fallback MD5 entra nos resultados.
    """
    pasta = _ABORT_CONTEXT.get("pasta")
    print(f"\n{'!' * 70}\n  CRÉDITOS DO OPENROUTER ESGOTADOS — execução ABORTADA.\n  {str(exc)[:300]}\n{'!' * 70}")
    if pasta:
        checkpoint_interactions(pasta)
        tee = _ABORT_CONTEXT.get("tee")
        if tee is not None:
            try:
                sys.stdout = tee._stream
                sys.stderr = _ABORT_CONTEXT["tee_err"]._stream
                salvar_log_em_chunks(tee.getvalue(), pasta, nome_base="log_execucao")
            except Exception as e2:
                print(f"  ⚠ Falha ao salvar o log parcial: {e2}")
        print(f"  Interações salvas em {pasta}/llm_interactions_checkpoint.json; log parcial em log_execucao*.txt.")
    raise SystemExit(3)


class bloco_protegido:
    """Gestor de contexto que isola a falha de UM bloco lógico do `main()` (B3, 09/10/2026).

    Uso: ``with bloco_protegido("Bloco 1 (Problemas A, B e C) seed=42", falhas_fatais, pasta): ...``.
    Uma `Exception` dentro do bloco é registrada em `falhas_fatais` (com traceback no
    log), grava um checkpoint das interações (se `pasta` for dada) e é SUPRIMIDA: o
    fluxo continua no bloco seguinte e chega ao PASSO 7 (CSVs do que existir,
    llm_interactions, log_execucao.txt) e ao `SystemExit(2)` final. `SystemExit`,
    `KeyboardInterrupt` e `CreditosEsgotadosError` NÃO são engolidos: atravessam
    (o 402 já aborta com checkpoint + log em `_abortar_por_creditos`).
    """

    def __init__(self, nome: str, falhas_fatais: List[str], pasta: Optional[str] = None):
        self.nome, self.falhas_fatais, self.pasta = nome, falhas_fatais, pasta

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        if not isinstance(exc, Exception) or isinstance(exc, CreditosEsgotadosError):
            return False   # SystemExit, KeyboardInterrupt, 402: propagam
        print(f"\n  ⚠ ERRO em {self.nome}: {type(exc).__name__}: {exc}")
        traceback.print_exception(exc_type, exc, tb)
        self.falhas_fatais.append(f"{self.nome}: {type(exc).__name__}: {exc}")
        if self.pasta:
            try:
                checkpoint_interactions(self.pasta)
            except Exception as e2:   # o checkpoint é proteção extra, nunca derruba
                print(f"  ⚠ Falha ao gravar checkpoint após o erro: {e2}")
        print("  → bloco registrado em falhas_fatais; a execução continua (exit 2 no fim).")
        return True

# A geometria dos problemas (PROBLEM_*_CENTERS / PROBLEM_*_STD) e os geradores
# create_problem_* foram extraídos para data_problems.py (importados no topo).
# Ver tests/test_data_problems.py.



# PERCEPTRON_PARAMS (hiperparâmetros do Perceptron Estruturado, ponto ÚNICO de
# configuração dos 8 call sites) vive em protocolo.py — importado/reexportado acima.

# Coletor global para diagnóstico da busca binária em γ no Perceptron Estruturado
# (item b da reunião 30/04/2026, ~520s — orientador pediu para verificar se gamma
# converge crescentemente). Preenchido pela única chamada com
# train_relaxed_perceptron(..., return_history=True) — o Passo 4 do Problema A,
# por (modelo, seed) — e plotado em plot_a_passo4_gamma -> bloco1_graphic_09_A_passo4_gamma.png.
PERCEPTRON_GAMMA_DIAGNOSTICS: List[dict] = []

# Rótulos coletados nos Passos 2 e 3 dos Problemas A e D, por (provider, model, seed,
# problema completo, nome_classe_0, nome_classe_1, nome_feature_0, nome_feature_1,
# k, rep): evita recoletar e alimenta figuras/tabelas sem nova chamada. Os nomes de
# atributo entram na chave desde 13/10/2026 (Problema G: variantes não se cruzam).
CACHE_ROTULOS_LLM: Dict[tuple, dict] = {}

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
# CONFIGURAÇÕES DE ESPECIALISTAS MÚLTIPLOS (Problema E)
# ═══════════════════════════════════════════════════════════════════════════

# Nomenclatura (08/10/2026): os especialistas do Problema E são SUBPROBLEMAS E1, E2 e E3
# (mesmos dados, rótulo de outro especialista). Até 07/10 chamavam-se aniso_x2, aniso_x1
# e euclidean; execuções antigas mantêm esses nomes na coluna `expert_name`.
EXPERT_CONFIGS = [
    {"name": "E1", "w": np.array([0.3, 1.5]), "desc": "especialista W=[0,3; 1,5], x2 dominante (principal; antigo aniso_x2)"},
    {"name": "E2", "w": np.array([1.5, 0.3]), "desc": "especialista W=[1,5; 0,3], x1 dominante (antigo aniso_x1)"},
    {"name": "E3", "w": np.array([1.0, 1.0]), "desc": "especialista W=[1, 1], euclidiano (antigo euclidean)"},
]

# O oráculo cobre só as três configurações do texto do orientador (IV.5), todas sobre as
# gaussianas do Problema E (14/10/2026): Oráculo sobre E3 (W = [1, 1]) e sobre E1/E2
# ([0,3; 1,5] e [1,5; 0,3]), os Problemas 1/2/3 da tabela 17. As gaussianas anisotrópicas extras (x1_preciso/x2_preciso/forte_aniso)
# foram removidas em 13/10/2026: com qualquer separação ampla todo W classifica 100%
# dos pontos, W não é identificável e o teste não mede o estimador.

# ═══════════════════════════════════════════════════════════════════════════
# FLAGS DE EXECUÇÃO — controla quais experimentos rodar
# ═══════════════════════════════════════════════════════════════════════════

RUN_PROBLEMAS_ABC = True              # Bloco 1: Problema A (Passos 2 a 6), B e C (Passos 2 a 4)
RUN_PROBLEMA_E = True                 # Bloco 2: Problema E, Passos 2 e 3 — LLM como aprendiz (in-context learning)
RUN_VIES_ORDEM_CLASSES = True        # Teste de inversão de ordem das classes
RUN_VIES_NOMES_ATRIBUTO = True           # Teste de nomes semânticos nas features
RUN_E_PASSO4_DILUICAO = True                # Experimento de diluição
RUN_A_PASSO4_R3R4 = True           # Augmentação R3 (x1·x2) e R4 (x1², x2²) p/ detectar não-linearidade implícita
RUN_PROBLEMA_E2_E3 = True        # Múltiplas configs de especialista no Problema E (E1/E2/E3)
RUN_A_PASSO4_PERCEPTRON_VS_NNLS = True    # Comparação Perceptron × NNLS (robustez ao método)
RUN_ORACULO = True       # Validação: algoritmos recuperam W conhecido?
RUN_D_PASSO5_TETO = True          # Sub-flag do oráculo: Passo 5 do D sem LLM (teto da meia-lua em R2/R3/R4; 0 chamadas)
RUN_E_PASSO5_ORDEM = True      # Teste de viés de ordem dos exemplos few-shot
RUN_VIES_VARIANTES_PROMPT = True         # Teste de múltiplas variantes de prompt
RUN_E_PASSO6_BASELINES = True     # Comparação com baselines clássicos (k-NN, LR, SVM)
RUN_PROBLEMA_F = True         # Bloco 2: meia-lua como Problema F (aprendiz, Passos 1 a 4, split 70/30); o D é RUN_PROBLEMA_D
RUN_PROBLEMA_G = True            # Estudo de caso real: peso × altura (homem/mulher), elipse
RUN_PROBLEMA_G_CLASSES_AB = True       # Item 17 (reunião 20/05): repetir peso×altura com classes "A"/"B" (sem prior semântico)
RUN_PROBLEMA_D = True        # Bloco 1: meia-lua (Problema D, 150 pts, SEM split), Passos 2 a 6 (2 a 5 como o A + SVM)

# Item 17 (reunião 20/05 + e-mail): nomes de classe testados no peso×altura.
# Isola o prior semântico nos NOMES DAS CLASSES — "Homem"/"Mulher" carregam
# significado; "A"/"B" são neutros. Complementa o teste de nomes de FEATURE
# (x1/x2 vs peso/altura). Cada par extra multiplica o custo de API do bloco.
#
# CONVENÇÃO (tupla = (nome da classe 0, nome da classe 1, nome do problema)):
# na base peso_altura.csv a classe 0 corresponde a `male? = -1` e a classe 1 a
# `male? = +1` (arquivo-fonte Hghtwght_desired.txt); a classe 1 é a mais alta
# e mais pesada. Logo classe 0 = Mulher e classe 1 = Homem. Até 25/08/2026 a
# tupla estava invertida (("Homem", "Mulher")), o que fazia a resposta "Homem"
# virar y_llm = 0 e ser comparada com y_true = 1: todas as métricas "vs rótulo
# real" do Bloco 3 saíam como 1 - valor (corrigidas offline em 26/08/2026; a ferramenta foi removida em 13/10/2026).
#
# Nomenclatura (08/10/2026): o Problema G tem dois SUBPROBLEMAS com as mesmas 100
# pessoas, o mesmo rótulo e a mesma divisão 70/30 por semente, mudando só a
# apresentação ao LLM:
#   G1 — valores normalizados em [-1, 1] (como a base veio do NeuroSolutions);
#   G2 — valores em kg e metros (peso_altura_kg_m.csv, regra de três do orientador).
# A métrica é SEMPRE estimada no espaço normalizado (a conversão é linear; em kg e m
# o Perceptron ficaria dominado pelo peso: 1 unidade normalizada = 32,5 kg = 0,225 m).
# Execuções anteriores a 08/10 gravaram G1 como `homem_mulher`/`homem_mulher_classesAB`.
G_NOMES_CLASSE = [
    ("Mulher", "Homem", "G1_homem_mulher"),      # semântico (classe 0 = Mulher, 1 = Homem)
    ("A", "B", "G1_homem_mulher_classesAB"),     # neutro (item 17)
]
# G2: mesmas variantes de classe; prompt mostra kg e m. Atributos: nome físico
# (peso_kg/altura_m) e, como controle, nome neutro (x1/x2) com os MESMOS valores
# físicos, para separar o efeito da unidade do efeito do nome "peso".
G2_NOMES_CLASSE = [
    ("Mulher", "Homem", "G2_homem_mulher"),
    ("A", "B", "G2_homem_mulher_classesAB"),
]
G1_NOMES_ATRIBUTO = [("x1", "x2"), ("peso", "altura")]
G2_NOMES_ATRIBUTO = [("peso_kg", "altura_m"), ("x1", "x2")]


def nome_problema_f(seed: int) -> str:
    """`problem_name` do Problema F (meia-lua como aprendiz, Bloco 2), um por semente.

    Prefixo com a letra do problema, como em G1_/G2_. Execuções anteriores a
    08/10/2026 gravavam `meia_lua_seed{N}`.
    """
    return f"F_meia_lua_seed{seed}"

# CLI `--apenas <bloco> [<bloco> ...]`: roda SÓ os blocos selecionados. Cada bloco
# é definido pelas flags de topo que o ativam; as sub-flags (vieses dentro das
# Problemas A, B e C; diluição/baselines dentro do Problema E; A/B dentro do peso×altura,
# meia-lua dentro do oráculo) continuam valendo o que está configurado acima,
# porque já são gateadas pelo bloco-pai. As flags dos blocos NÃO selecionados são
# forçadas a False e as dos selecionados a True. A pasta de execução recebe o
# sufixo `_apenas-<blocos>` (nunca `_completa`), para não ser confundida com uma
# execução cheia pelas ferramentas que leem "a última _completa".
BLOCOS_APENAS = {
    # Meia-lua: Problema D (Bloco 1, LLM como especialista, Passos 2 a 5 em
    # R2/R3/R4, sem split) é RUN_PROBLEMA_D; Problema F (Bloco 2, LLM como
    # aprendiz, Passos 1 a 4 com split) é RUN_PROBLEMA_F.
    "bloco1":  ["RUN_PROBLEMAS_ABC", "RUN_A_PASSO4_R3R4", "RUN_PROBLEMA_D"],  # LLM como especialista (A/B/C + R3/R4 + D)
    "bloco2":  ["RUN_PROBLEMA_E", "RUN_PROBLEMA_F"],       # LLM como aprendiz (E + meia-lua F)
    "bloco3":  ["RUN_PROBLEMA_G"],                         # caso real peso × altura (Problema G)
    # Oráculo = chave própria (0 chamadas de API), embora seus assets sejam do
    # Bloco 1 (bloco1_graphic_19/21, bloco1_oraculo_*, bloco1_d_passo5_teto_*): `--apenas bloco1` NÃO o roda.
    "oraculo": ["RUN_ORACULO"],                    # validação sem LLM (0 chamadas)
}


def flags_para_apenas(selecao) -> dict:
    """Devolve {nome_da_flag: bool} para `--apenas selecao`.

    Flags de topo dos blocos selecionados ficam True; as dos demais, False. Só as
    flags listadas em BLOCOS_APENAS são tocadas. Levanta ValueError para bloco
    desconhecido ou seleção vazia.
    """
    selecao = list(selecao)
    if not selecao:
        raise ValueError("--apenas exige ao menos um bloco")
    desconhecidos = sorted(set(selecao) - set(BLOCOS_APENAS))
    if desconhecidos:
        raise ValueError(
            f"bloco(s) desconhecido(s) em --apenas: {desconhecidos}; "
            f"válidos: {list(BLOCOS_APENAS)}"
        )
    ativas = {flag for bloco in selecao for flag in BLOCOS_APENAS[bloco]}
    return {
        flag: (flag in ativas)
        for bloco in BLOCOS_APENAS
        for flag in BLOCOS_APENAS[bloco]
    }


def sufixo_apenas(selecao) -> str:
    """Sufixo de pasta para `--apenas`: ``apenas-bloco3`` ou ``apenas-bloco2+bloco3``."""
    ordenados = [b for b in BLOCOS_APENAS if b in set(selecao)]
    return "apenas-" + "+".join(ordenados)


def nome_pasta_execucao(timestamp: str, apenas=None, modo_rapido: bool = False, reestimada: bool = False) -> str:
    """Nome da pasta de saída: ``execucao_<timestamp>_<tipo>``.

    ``reestimada`` (15/10/2026, `--reestimar-de`): rótulos do LLM reaproveitados de outra execução e
    métricas reestimadas; o tipo recebe o sufixo ``_reestimada`` (``completa_reestimada``) para a pasta
    não ser confundida com uma coleta nova; é fonte do trabalho só se `reestimacao_lacunas.csv` não
    existir ou todas as lacunas tiverem sido coletadas na API (ver o fim do log).

    ``tipo`` distingue a execução cheia (``completa``, única fonte do trabalho) do
    smoke test (``smoke``, `--rapido`) e da execução parcial (``apenas-<blocos>``,
    `--apenas`, com ``_smoke`` no fim quando combinada com `--rapido`); uma parcial
    NUNCA leva ``completa``, para as ferramentas que filtram "a última _completa".
    """
    if apenas:
        tipo = sufixo_apenas(apenas) + ("_smoke" if modo_rapido else "")
    else:
        tipo = "smoke" if modo_rapido else "completa"
    if reestimada:
        tipo += "_reestimada"
    return f"execucao_{timestamp}_{tipo}"


# Grades reduzidas do smoke test (`--rapido`): TODOS os experimentos executam (cobertura
# completa de código/prompts), cada um UMA vez: 1 seed, 1 repetição e, onde há
# few-shot, só o zero-shot + UM few-shot (k = 4, par: balanço exato de classes).
GRADES_MODO_RAPIDO = {
    "RANDOM_SEEDS": [42],
    "N_REPETICOES": 1,
    "K_EXEMPLOS_APRENDIZ": [0, 4],
    "K_EXEMPLOS_METRICA_BC": [0, 4],          # B/C: zero-shot + um few-shot
    "K_EXEMPLOS_ROTULO_REAL": [4],         # um k no Passo 3 dos Problemas A, D e G
    "K_VIESES": [0, 4],               # zero-shot + um few-shot nos vieses
    "DILUTION_EASY_ADDITIONS": [0, 4],    # 2 pontos da curva de diluição
    "K_ORDEM_EXEMPLOS": [4],         # um few-shot (par) no viés de ordem
}


def grades_modo_rapido() -> dict:
    """Cópia das grades reduzidas do `--rapido` (função pura, para testes e para main())."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in GRADES_MODO_RAPIDO.items()}


def aplicar_modo_rapido(namespace: Optional[dict] = None) -> dict:
    """Aplica as grades do `--rapido` ao namespace do módulo (ou ao dict dado) e as devolve."""
    grades = grades_modo_rapido()
    (globals() if namespace is None else namespace).update(grades)
    return grades

# Estratégias de ordenação dos exemplos few-shot
EXAMPLE_ORDERINGS = ["class0_first", "class1_first", "shuffled", "alternating"]

# Variantes de prompt para teste de sensibilidade
# max_tokens = None em todas as variantes: NÃO se impõe limite de tokens. O
# gemini-2.5-flash-lite às vezes raciocina em texto antes do rótulo (o parser extrai a
# resposta pela camada 0/última linha); no smoke de 08/10/2026 um limite de 50 tokens
# truncou 256 respostas antes do rótulo (1,1% de fallback MD5). Sem limite: 2 malformadas.
PROMPT_VARIANTS = {
    "default": {
        "system_message": "You are a classifier. Respond only with the class label.",
        "max_tokens": None,
        "description": "Template padrão (baseline)",
    },
    "geometric": {
        "system_message": "You are a geometric classifier analyzing point positions in metric space. Respond only with the class label.",
        "max_tokens": None,
        "description": "Contexto geométrico/espacial explícito",
    },
    "cot": {
        "system_message": "You are a classifier. Think step by step, then give your final answer on the last line as just the class label.",
        "max_tokens": None,
        "description": "Chain-of-thought antes da resposta",
    },
    "tabular": {
        "system_message": "You are a classifier. Respond only with the class label.",
        "max_tokens": None,
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

# create_problem_a/b/c/e_expert/d_meialua/homem_mulher
# → data_problems.py (importados no topo). Ver tests/test_data_problems.py.








# =============================================================================
# INTERAÇÃO COM O LLM (COM RETENTATIVA DE FORMATO)
# =============================================================================

def prompt_zero_shot(x1: float, x2: float,
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


def prompt_com_exemplos(x1: float, x2: float,
                          examples: List,
                          nome_classe_0: str, nome_classe_1: str,
                          nome_feature_0: str = "x1", nome_feature_1: str = "x2",
                          extra_features: Optional[List[Tuple[str, float]]] = None) -> str:
    """Constrói o prompt FEW-SHOT com exemplos rotulados (origem varia por problema e passo).

    Origem dos rótulos depende do chamador:
      - Problemas B e C, Passo 4: rotulados pela métrica Ŵ_LLM do A (ancoragem em H2).
      - Problema E, Passo 3: rotulados pelo especialista externo (W_expert).
      - Problema F, Passo 3: rotulados pelo rótulo real (especialista = ground truth, D1 09/10/2026);
        Problema G, Passo 6: rotulados pela métrica eleita no Passo 5 (2/3/4 atributos).
      - Problemas A, D e G, Passo 3: rotulados pelo rótulo real (passo3_exemplos_rotulo_real).
    Examples pode conter tuplas de 3 (x1, x2, label) ou 4+ elementos com atributos extras.
    """
    n_dim = 2 + (len(extra_features) if extra_features else 0)
    dim_label = f"{n_dim}D"

    examples_lines = []
    for ex in examples:
        line = f"  {nome_feature_0} = {ex[0]:.4f}, {nome_feature_1} = {ex[1]:.4f}"
        if len(ex) > 3:
            # Atributos extras do exemplo (x1·x2 em R³; x1², x2² em R⁴), com os
            # MESMOS nomes e ordem dos atributos extras do ponto a classificar.
            # (Até 08/10/2026 o laço começava em 3 e descartava/trocava atributos.)
            for j, i in enumerate(range(2, len(ex) - 1)):
                nome = extra_features[j][0] if extra_features and j < len(extra_features) else f"x{i + 1}"
                line += f", {nome} = {ex[i]:.4f}"
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


def prompt_zero_shot_variante(
    variant: str, x1: float, x2: float,
    nome_classe_0: str, nome_classe_1: str,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None
) -> str:
    """Constrói prompt zero-shot usando a variante especificada."""
    if variant == "default":
        return prompt_zero_shot(x1, x2, nome_classe_0, nome_classe_1,
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


def prompt_com_exemplos_variante(
    variant: str, x1: float, x2: float,
    examples: List,
    nome_classe_0: str, nome_classe_1: str,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None
) -> str:
    """Constrói prompt few-shot usando a variante especificada."""
    if variant == "default":
        return prompt_com_exemplos(x1, x2, examples, nome_classe_0, nome_classe_1,
                                      nome_feature_0, nome_feature_1, extra_features)

    n_dim = 2 + (len(extra_features) if extra_features else 0)
    dim_label = f"{n_dim}D"

    # Monta linhas de exemplos
    examples_lines = []
    for ex in examples:
        line = f"  {nome_feature_0} = {ex[0]:.4f}, {nome_feature_1} = {ex[1]:.4f}"
        if len(ex) > 3:
            for j, i in enumerate(range(2, len(ex) - 1)):
                nome = extra_features[j][0] if extra_features and j < len(extra_features) else f"x{i + 1}"
                line += f", {nome} = {ex[i]:.4f}"
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
                for i in range(2, len(ex) - 1):
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


async def async_llm_classify_point_openai(async_client, model_name: str, prompt: str,
                                           temperature: float,
                                           system_message: str = "You are a classifier. Respond only with the class label.",
                                           max_tokens: Optional[int] = None) -> Tuple[str, dict]:
    """Chama a API compatível com OpenAI (OpenRouter) de forma assíncrona.

    Envolve a chamada em asyncio.wait_for(timeout=API_TIMEOUT_S) como segunda barreira
    contra travamentos (o SDK já tem 60 s × até 3 tentativas próprias): ao estourar,
    levanta ``TimeoutError`` com a palavra "timeout" na mensagem, que o laço de
    `async_llm_classify_point` trata como erro RETENTÁVEL (backoff exponencial), em vez
    de bloquear o semáforo ou devolver string vazia.

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
            **({"max_tokens": max_tokens} if max_tokens else {}),
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
        return await asyncio.wait_for(_do_call(), timeout=API_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        # Retentável pelo laço de rate limit (a mensagem contém "timeout"); NÃO aciona
        # o freio global (não é sinal de capacidade do provedor).
        raise TimeoutError(f"timeout de {API_TIMEOUT_S:.0f}s na chamada à API") from exc


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
_TOKENS_CAPACIDADE = ("error code: 429", "rate_limit", "overloaded", "temporarily")


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


def _montar_prompt(x1: float, x2: float, nome_classe_0: str, nome_classe_1: str,
                   examples: Optional[List], nome_feature_0: str, nome_feature_1: str,
                   extra_features: Optional[List[Tuple[str, float]]], prompt_variant: str,
                   reforco: str = "") -> str:
    """Prompt (zero ou few-shot) da variante, com o reforço de formato opcional no fim."""
    if examples is None or len(examples) == 0:
        prompt = prompt_zero_shot_variante(prompt_variant, x1, x2, nome_classe_0, nome_classe_1,
                                                nome_feature_0, nome_feature_1, extra_features)
    else:
        prompt = prompt_com_exemplos_variante(prompt_variant, x1, x2, examples, nome_classe_0, nome_classe_1,
                                               nome_feature_0, nome_feature_1, extra_features)
    return prompt + reforco


# Mensagens de HTTP 400 que o OpenRouter devolve de forma transitória (o endpoint
# compatível às vezes responde 400 para falha momentânea do provedor). Qualquer outro
# 400 (requisição inválida de fato) NÃO é retentável: até 08/10 todo 400 era retentado
# e um erro persistente prendia 40 workers por ~26 min cada.
_TOKENS_400_TRANSITORIO = ("could not parse", "provider returned error", "overloaded")


def _erro_retentavel(e: Exception) -> bool:
    """Decide se o erro da API merece nova tentativa com backoff (ver async_llm_classify_point)."""
    error_str = str(e).lower()
    _status = getattr(e, "status_code", None)
    _codigos = (408, 429, 500, 502, 503, 504)
    if _status in _codigos or any(f"error code: {c}" in error_str for c in _codigos):
        return True
    if _status == 400 or "error code: 400" in error_str:
        return any(tok in error_str for tok in _TOKENS_400_TRANSITORIO)
    return any(tok in error_str for tok in (
        "rate_limit", "overloaded", "could not parse",
        "timeout", "timed out",      # APITimeoutError do SDK ("Request timed out.")
        "connection", "temporarily",
        "provider returned error"))   # 404 transitório do OpenRouter


def _erro_timeout(e: Exception) -> bool:
    """Timeout da chamada: o `TimeoutError` do `asyncio.wait_for` (API_TIMEOUT_S) ou o
    `APITimeoutError` do SDK ("Request timed out."). Só esses seguem a regra de
    MAX_RETRIES_TIMEOUT; mensagens genéricas com "timeout" (ex.: gateway) continuam no
    backoff exponencial comum."""
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return True
    return type(e).__name__ == "APITimeoutError" or "timed out" in str(e).lower()


async def async_llm_classify_point(
    x1: float, x2: float,
    nome_classe_0: str, nome_classe_1: str,
    examples: Optional[List] = None,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    extra_features: Optional[List[Tuple[str, float]]] = None,
    prompt_variant: str = "default",
    tag: Optional[str] = None,
    client=None,
) -> Tuple[str, str, bool]:
    """Classifica um ponto via LLM (retries + fallback MD5). Retorna (label_parsed, raw_response, was_malformed).

    ``client``: cliente assíncrono da coleta corrente (criado por
    `async_collect_llm_decisions` dentro do laço); ``None`` usa o global ``async_client``
    (ponto de injeção dos testes).
    """
    prompt = _montar_prompt(x1, x2, nome_classe_0, nome_classe_1, examples,
                            nome_feature_0, nome_feature_1, extra_features, prompt_variant)
    cliente = client if client is not None else async_client

    # system_message da variante; max_tokens fica None (sem limite): ver nota em PROMPT_VARIANTS.
    variant_config = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["default"])
    system_msg = variant_config["system_message"]
    max_tok = variant_config["max_tokens"]

    all_responses = []
    all_prompts = []
    all_metas = []

    for format_attempt in range(MAX_FORMAT_RETRIES):
        n_timeouts = 0
        for rate_attempt in range(MAX_RETRIES):
            await _respeitar_rate_limit_global()
            try:
                label, call_meta = await async_llm_classify_point_openai(cliente, MODEL_NAME, prompt,
                                                                         CURRENT_TEMPERATURE, system_msg,
                                                                         max_tokens=max_tok)
                break
            except Exception as e:
                if erro_de_credito(e):
                    # 402: crédito/orçamento esgotado — fatal, nunca retentável.
                    raise CreditosEsgotadosError(str(e)[:300]) from e
                error_str = str(e).lower()
                _status = getattr(e, "status_code", None)
                # Retentáveis: rate limit (429), 408/5xx, timeout, conexão e 400 SÓ com
                # mensagem transitória (_erro_retentavel). O código HTTP vem do atributo
                # do SDK ou de "error code: NNN" na mensagem (não de substrings soltas
                # como "400" dentro de "14000").
                if _erro_timeout(e):
                    # Timeout (15/10/2026): espera FIXA curta e no máximo MAX_RETRIES_TIMEOUT
                    # chamadas estouradas; não aciona o freio global nem o backoff exponencial.
                    n_timeouts += 1
                    if n_timeouts >= MAX_RETRIES_TIMEOUT:
                        print(f"  ⏱ Timeout de {API_TIMEOUT_S:.0f}s, tentativa {n_timeouts}/{MAX_RETRIES_TIMEOUT}: "
                              f"desistindo do ponto (fallback).")
                        raise Exception(f"Máximo de tentativas por timeout ({MAX_RETRIES_TIMEOUT}) excedido "
                                        f"(timeout de {API_TIMEOUT_S:.0f}s persistente)") from e
                    print(f"  ⏱ Timeout de {API_TIMEOUT_S:.0f}s, tentativa {n_timeouts}/{MAX_RETRIES_TIMEOUT}. "
                          f"Aguardando {ESPERA_TIMEOUT_S:.0f}s...")
                    await asyncio.sleep(ESPERA_TIMEOUT_S)
                elif _erro_retentavel(e):
                    # Jitter de ±25%: sem ele, todos os workers que receberam 429
                    # no mesmo instante acordam juntos e re-disparam em rajada,
                    # re-acionando o rate limit. Só afeta TIMING de retry, nunca
                    # os dados — não compromete a reprodutibilidade do experimento.
                    wait_time = INITIAL_BACKOFF * (2 ** rate_attempt) * (0.75 + 0.5 * random.random())
                    if _status == 429 or any(tok in error_str for tok in _TOKENS_CAPACIDADE):
                        # Sinal de capacidade do provedor: TODOS os workers seguram
                        # as próximas chamadas até a janela passar (freio global).
                        _acionar_rate_limit_global(wait_time)
                    print(f"  ⏳ Erro retentável ({str(e)[:80]}). Aguardando {wait_time:.0f}s (tentativa {rate_attempt + 1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        else:
            raise Exception(f"Máximo de tentativas ({MAX_RETRIES}) excedido (erro retentável persistente)")

        all_responses.append(label)
        all_prompts.append(prompt)
        all_metas.append(call_meta)
        parsed_label, is_valid = parse_llm_response(label, nome_classe_0, nome_classe_1)

        if is_valid:
            # raw_response = a resposta de onde saiu parsed_label (após retry de
            # formato, a válida); as anteriores ficam em all_responses.
            raw = label
            LLM_INTERACTIONS.append({
                "point": {"x1": x1, "x2": x2},
                "prompt": all_prompts[0],
                "raw_response": raw,
                "all_responses": list(all_responses),
                "parsed_label": parsed_label,
                "model": MODEL_NAME,
                "model_resolved": call_meta.get("model_resolved"),
                "inference_provider": call_meta.get("inference_provider"),
                "provider": CURRENT_PROVIDER,
                "temperature": CURRENT_TEMPERATURE,
                "format_retries": format_attempt,
                "malformed": False,
                "tag": tag if tag is not None else CURRENT_TAG,
            })
            return parsed_label, raw, False

        if format_attempt < MAX_FORMAT_RETRIES - 1:
            # Reenvio com o MESMO estímulo (variante, nomes de atributo, atributos
            # extras e exemplos) mais a instrução de formato. (Até 08/10/2026 o
            # retry era um prompt fixo "2D points/x1/x2" que trocava o estímulo.)
            reforco = (f'\n\nIMPORTANT: Your response must be EXACTLY "{nome_classe_0}" or '
                       f'"{nome_classe_1}" with no other text.')
            prompt = _montar_prompt(x1, x2, nome_classe_0, nome_classe_1, examples,
                                    nome_feature_0, nome_feature_1, extra_features,
                                    prompt_variant, reforco=reforco)

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
        "tag": tag if tag is not None else CURRENT_TAG,
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
    prompt_variant: str = "default",
    tag: Optional[str] = None,
    semaphore: Optional["asyncio.Semaphore"] = None,
) -> Tuple[np.ndarray, int]:
    """Versão assíncrona de collect_llm_decisions com chamadas concorrentes à API.

    `semaphore` (15/10/2026): semáforo COMPARTILHADO entre várias coletas rodando no
    mesmo laço de eventos (`collect_llm_decisions_lote`), para o total em voo nunca passar
    de MAX_CONCURRENCY; sem ele, cada coleta cria o seu.

    `tag` (opcional) é gravada em cada registro de LLM_INTERACTIONS (auditoria por
    regime); sem ela vale o global CURRENT_TAG.

    Ciclo de vida do cliente (09/10/2026): se o global ``async_client`` é None, a
    corrotina cria o cliente de `get_async_client(CURRENT_PROVIDER)` AQUI (dentro do
    laço de eventos) e o fecha no ``finally``, ainda com o laço aberto, para que as
    conexões do httpx nunca sobrevivam ao `asyncio.run` que as criou ("Event loop is
    closed"). Um ``async_client`` injetado (testes) é usado e não é fechado."""
    n_examples = 0 if examples is None else len(examples)
    shot_type = "zero-shot" if n_examples == 0 else f"{n_examples}-shot"
    n_total = len(X)

    if verbose:
        print(f"  {label_prefix}Coletando decisões do LLM ({shot_type}) — {n_total} pontos [async, concurrency={MAX_CONCURRENCY}]...")
        sys.stdout.flush()

    semaphore = semaphore if semaphore is not None else asyncio.Semaphore(MAX_CONCURRENCY)
    cliente_proprio = async_client is None
    cliente = get_async_client(CURRENT_PROVIDER) if cliente_proprio else async_client
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
                    prompt_variant=prompt_variant, tag=tag, client=cliente,
                )
        except CreditosEsgotadosError:
            # Sem crédito NÃO há fallback: a exceção atravessa o gather (fail-fast),
            # o `finally` abaixo fecha o cliente e `collect_llm_decisions` aborta
            # (checkpoint + log + SystemExit 3) já FORA do laço. Antes de 09/10/2026
            # cada worker com 402 levantava SystemExit dentro do laço e os demais
            # apareciam como "unhandled exception during asyncio.run() shutdown".
            raise
        except Exception as exc:
            # Falha DEFINITIVA (esgotou retries ou erro não-retentável): não
            # derruba a execução inteira (gather é fail-fast) — registra como
            # malformada com o fallback MD5 e segue; a auditoria offline acusa.
            # Grava o prompt REAL (a chave de flip da auditoria é prompt+ponto) e
            # format_retries=0: não houve retentativa de FORMATO, houve erro de API.
            import hashlib
            hs = hashlib.md5(f"{x1:.6f}_{x2:.6f}".encode()).hexdigest()
            label = nome_classe_0 if int(hs, 16) % 2 == 0 else nome_classe_1
            raw, was_malformed = "", True
            LLM_INTERACTIONS.append({
                "point": {"x1": x1, "x2": x2},
                "prompt": _montar_prompt(x1, x2, nome_classe_0, nome_classe_1, examples,
                                         nome_feature_0, nome_feature_1, extra_feats, prompt_variant),
                "raw_response": "",
                "all_responses": [],
                "parsed_label": label,
                "model": MODEL_NAME,
                "model_resolved": None,
                "inference_provider": None,
                "provider": CURRENT_PROVIDER,
                "temperature": CURRENT_TEMPERATURE,
                "format_retries": 0,
                "malformed": True,
                "error": str(exc)[:300],
                "tag": tag if tag is not None else CURRENT_TAG,
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
    try:
        await asyncio.gather(*tasks)
    finally:
        # Fecha o cliente desta coleta com o laço AINDA ABERTO (também no abort por
        # 402/SystemExit): as conexões morrem no laço em que nasceram.
        if cliente_proprio:
            try:
                await cliente.close()
            except Exception as exc_close:   # fechar nunca derruba a coleta
                print(f"    ⚠ Falha ao fechar o cliente da API: {exc_close}")

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
    prompt_variant: str = "default",
    tag: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Coleta as decisões do LLM para todos os pontos do conjunto de dados.

    Wrapper síncrono que delega para async_collect_llm_decisions com chamadas concorrentes.
    Um 402 (`CreditosEsgotadosError`) sai do laço de eventos (cliente já fechado) e vira
    `_abortar_por_creditos` aqui: checkpoint + log parcial + SystemExit(3), sem fallback MD5.
    """
    try:
        return asyncio.run(async_collect_llm_decisions(
            X, nome_classe_0, nome_classe_1,
            examples=examples, verbose=verbose, label_prefix=label_prefix,
            nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
            extra_features_matrix=extra_features_matrix,
            extra_feature_names=extra_feature_names,
            prompt_variant=prompt_variant, tag=tag,
        ))
    except CreditosEsgotadosError as exc:
        _abortar_por_creditos(exc)


_COLLECT_LLM_DECISIONS_ORIGINAL = collect_llm_decisions   # p/ reconhecer dublês dos testes


async def _async_collect_lote(jobs: List[dict], rotulo: str) -> List[Tuple[np.ndarray, int]]:
    """Várias coletas no MESMO laço de eventos, com um cliente e um semáforo compartilhados."""
    global async_client
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    cliente_proprio = async_client is None
    cliente = get_async_client(CURRENT_PROVIDER) if cliente_proprio else async_client
    if cliente_proprio:
        async_client = cliente          # as coletas internas usam este cliente e NÃO o fecham
    n_pts = sum(len(j["X"]) for j in jobs)
    t0 = time.time()
    print(f"  {rotulo}lote de {len(jobs)} coletas ({n_pts} chamadas) [async, concurrency={MAX_CONCURRENCY}]...", flush=True)
    try:
        res = await asyncio.gather(*[
            async_collect_llm_decisions(**{**j, "verbose": False}, semaphore=sem) for j in jobs
        ])
    finally:
        if cliente_proprio:
            async_client = None
            try:
                await cliente.close()
            except Exception as exc:
                print(f"  ⚠ Falha ao fechar o cliente da API do lote ({type(exc).__name__}: {exc}); seguindo.")
    print(f"  {rotulo}lote concluído em {time.time() - t0:.1f}s ({n_pts} chamadas, "
          f"{sum(int(r[1]) for r in res)} malformadas)", flush=True)
    return list(res)


def collect_llm_decisions_lote(jobs: List[dict], rotulo: str = "") -> List[Tuple[np.ndarray, int]]:
    """Coleta VÁRIOS conjuntos de prompts de uma vez (15/10/2026).

    Cada item de `jobs` é um dict com os argumentos de `collect_llm_decisions` (`X`,
    `nome_classe_0`, `nome_classe_1`, `examples`, `nome_feature_*`, `extra_features_*`,
    `prompt_variant`, `tag`, `label_prefix`). Devolve, na MESMA ordem, a lista de
    `(y_llm, n_malformadas)`. Motivo: coletas pequenas (30 pontos no Passo 6 do G, ~110 no
    Passo 3 de A/D/G) eram limitadas pela latência de uma ida e volta à API cada, uma atrás
    da outra; num lote, todas as chamadas disputam o mesmo semáforo de MAX_CONCURRENCY. Nada muda nos prompts, nas tags nem nos
    resultados: só a ordem em que as chamadas saem. Quando `collect_llm_decisions` foi
    substituída por um dublê (testes) ou há um único job, chama a função coleta a coleta.
    """
    global CURRENT_TAG
    if not jobs:
        return []
    def _um(j):
        global CURRENT_TAG
        kw = {k: v for k, v in j.items() if k not in ("X", "nome_classe_0", "nome_classe_1")}
        tag_anterior = CURRENT_TAG
        if j.get("tag"):
            CURRENT_TAG = j["tag"]       # dublês antigos leem o global, não o argumento
        try:
            return collect_llm_decisions(j["X"], j["nome_classe_0"], j["nome_classe_1"], **kw)
        finally:
            CURRENT_TAG = tag_anterior   # a tag do lote não vaza para as coletas seguintes
    if globals()["collect_llm_decisions"] is not _COLLECT_LLM_DECISIONS_ORIGINAL or len(jobs) == 1:
        return [_um(j) for j in jobs]
    try:
        return asyncio.run(_async_collect_lote(jobs, rotulo))
    except CreditosEsgotadosError as exc:
        _abortar_por_creditos(exc)


# =============================================================================
# ESTIMAÇÃO DA MÉTRICA (OTIMIZAÇÃO INVERSA: PERCEPTRON ESTRUTURADO + NNLS)
# (augment_to_r3/r4 e augment_features, p/ R2/R3/R4, vivem em metrics.py)
# =============================================================================

# d_W e compute_centroids → metrics.py (importados no topo).


# train_relaxed_perceptron → relaxed_perceptron.py e train_least_squares_inverse →
# least_squares_inverse.py (funções de conveniência junto de suas classes; importadas no topo).


# augment_to_r3, augment_to_r4, augment_features, predict_with_metric e
# compute_metric_confidence → metrics.py (importados no topo). Ver tests/test_metrics.py.


# =============================================================================
# PROBLEMA A, PASSOS 2 E 4: ESTIMAÇÃO DA MÉTRICA SOBRE OS RÓTULOS ZERO-SHOT (O PASSO 3 ESTÁ ABAIXO)
# =============================================================================

def a_passo2_zero_shot_e_metrica(
    X_train_a: np.ndarray,
    y_true_train_a: np.ndarray,
    nome_classe_0: str,
    nome_classe_1: str,
    verbose: bool = True,
    prompt_variant: str = "default",
    nome_feature_0: str = "x1",
    nome_feature_1: str = "x2",
    random_seed: Optional[int] = None,
) -> Tuple[Optional[LearnedMetric], Optional[LearnedMetric], np.ndarray, float, float, float, int]:
    """Problema A, Passos 2 e 4 (antes de 13/10/2026: Fase A): o LLM rotula em zero-shot e estima-se a métrica Ŵ_LLM.

    Colapso (`colapsou`: classe única ou minoritária < MIN_PONTOS_CLASSE_MINORITARIA):
    devolve métricas None e fidelidades NaN. `nome_feature_0/1` permitem o experimento
    de nomes de atributo (altura/peso, feature_1/feature_2) passar pelo MESMO caminho
    da configuração base; `random_seed` só rotula o diagnóstico de γ (bloco1_graphic_09).

    Retorna:
        learned_metric: LearnedMetric via Perceptron Estruturado (algoritmo principal)
        learned_metric_nnls: LearnedMetric via NNLS (segundo estimador, robustez ao método)
        y_llm_train_a: decisões do LLM no Problema A
        fidelity: fidelidade do Perceptron (métrica vs. LLM)
        fidelity_nnls: fidelidade do NNLS (métrica vs. LLM)
        llm_accuracy: acurácia (vs rótulo real) do LLM, informação de contexto
        n_malformed: número de respostas malformadas

    (Antes de 13/10/2026: `phase_a_learn_metric`.)
    """
    variant_label = f" [prompt: {prompt_variant}]" if prompt_variant != "default" else ""
    if (nome_feature_0, nome_feature_1) != ("x1", "x2"):
        variant_label += f" [atributos: {nome_feature_0}/{nome_feature_1}]"
    if verbose:
        print("\n  ══════════════════════════════════════════════════════════════")
        print(f"  PROBLEMA A, PASSOS 2 E 4: ESTIMANDO A MÉTRICA W SOBRE OS RÓTULOS ZERO-SHOT{variant_label}")
        print("  ══════════════════════════════════════════════════════════════")
        print("  Objetivo: capturar o processo decisório INTRÍNSECO do LLM.")
        print("  O LLM classifica pontos sem exemplos (zero-shot), e a métrica")
        print("  Ŵ_LLM é inferida a partir dessas decisões via otimização inversa.")

    if verbose:
        print("\n  Passo 2: Coletando decisões do LLM no Problema A (zero-shot)...")
        print(f"    Classes usadas: '{nome_classe_0}' e '{nome_classe_1}'")
        print(f"    Pontos de treino: {len(X_train_a)}")

    y_llm_train_a, n_malformed = collect_llm_decisions(
        X_train_a, nome_classe_0, nome_classe_1,
        examples=None, verbose=verbose,
        label_prefix=f"[Problema A{variant_label}] ",
        nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
        prompt_variant=prompt_variant
    )

    if colapsou(y_llm_train_a):
        print(f"  ⚠ AVISO: colapso do LLM no Problema A, Passo 2 ({int(np.sum(y_llm_train_a == 0))}/"
              f"{int(np.sum(y_llm_train_a == 1))}; piso da classe minoritária = "
              f"{MIN_PONTOS_CLASSE_MINORITARIA}). Sem métrica: collapsed=1, medidas NaN.")
        return None, None, y_llm_train_a, float('nan'), float('nan'), accuracy_score(y_true_train_a, y_llm_train_a), n_malformed

    centroids_a = compute_centroids(X_train_a, y_llm_train_a)

    if verbose:
        print("\n  Passo 4: centróides a partir das decisões do LLM (Passo 2)...")
        print(f"    Centróide da Classe 0: ({centroids_a[0, 0]:.3f}, {centroids_a[0, 1]:.3f})")
        print(f"    Centróide da Classe 1: ({centroids_a[1, 0]:.3f}, {centroids_a[1, 1]:.3f})")

    # --- Algoritmo 1: Perceptron Estruturado com Relaxação de Margem ---
    if verbose:
        print("\n  Passo 4 (Perceptron): estimando a métrica W via Perceptron Estruturado com Relaxação de Margem...")
        print("    (Busca binária no gamma ótimo para maximizar a margem da fronteira de decisão)")

    w_learned, gamma_optimal, _gamma_hist = train_relaxed_perceptron(
        X_train_a, y_llm_train_a, centroids_a,
        **PERCEPTRON_PARAMS,
        verbose=verbose,
        use_best_effort=True,  # retorna melhor W parcial quando separação perfeita é impossível
        return_history=True,   # captura evolução de γ para diagnóstico (item b reunião 30/04)
    )
    margem_minima_perc = float(_rp.ULTIMA_MARGEM_MINIMA)   # 15/10/2026 (D): margem bruta mínima/||w|| de Ŵ
    # Anota diagnóstico de γ para esta chamada (Problema A, Passo 4); "model" permite
    # filtrar bloco1_graphic_09 por modelo. O rótulo identifica a configuração (seed,
    # nomes de classe, variante de prompt, atributos) e marca a configuração base.
    _base = (prompt_variant == "default" and (nome_classe_0, nome_classe_1) == ("A", "B")
             and (nome_feature_0, nome_feature_1) == ("x1", "x2"))
    _label = f"seed {random_seed} | {nome_classe_0}/{nome_classe_1} | {prompt_variant}"
    if (nome_feature_0, nome_feature_1) != ("x1", "x2"):
        _label += f" | {nome_feature_0}/{nome_feature_1}"
    PERCEPTRON_GAMMA_DIAGNOSTICS.append({
        "label": _label + (" [base]" if _base else ""),
        "base": _base,
        "model": MODEL_NAME,
        "seed": random_seed,
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

    # --- Algoritmo 2: Mínimos Quadrados (NNLS) — segundo estimador (robustez ao método) ---
    if verbose:
        print("\n  Passo 4 (NNLS): estimando a métrica W via Mínimos Quadrados Não-Negativos...")
        print("    (Segundo estimador: se ambos convergem para W similar, o resultado é robusto ao método)")

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
        print("\n  Passo 4: fidelidade (métrica vs LLM) das métricas estimadas...")
        print(f"    [Perceptron] Fidelidade (métrica vs. LLM no Problema A): {fidelity:.1%}")
        print(f"    [NNLS]       Fidelidade (métrica vs. LLM no Problema A): {fidelity_nnls:.1%}")
        print(f"    Acurácia (vs rótulo real) do LLM [contexto]: {llm_accuracy:.1%}")
        fids = [fidelity, fidelity_nnls]
        if all(f >= 0.9 for f in fids):
            print("    Ambas as métricas estimadas reproduzem bem as decisões do LLM.")
        elif any(f >= 0.9 for f in fids):
            print("    ⚠ Fidelidade divergente entre algoritmos — investigar.")
        else:
            print("    ⚠ Fidelidade baixa em ambos — a métrica pode não representar bem o LLM.")

    learned_metric = LearnedMetric(
        w=w_learned, centroids=centroids_a,
        gamma=gamma_optimal, source_problem="Problem_A", margem_minima=margem_minima_perc
    )

    learned_metric_nnls = LearnedMetric(
        w=w_nnls, centroids=centroids_a,
        gamma=0.0, source_problem="Problem_A"
    )

    return learned_metric, learned_metric_nnls, y_llm_train_a, fidelity, fidelity_nnls, llm_accuracy, n_malformed


# =============================================================================
# PROBLEMAS B E C, PASSOS 2 A 4: TESTE DE CONSISTÊNCIA EM NOVOS PROBLEMAS
# =============================================================================

def bc_passo4_exemplos_da_metrica(
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
    if len(selected_idx_0) + len(selected_idx_1) < n_per_class_0 + n_per_class_1:
        print(f"      ⚠ Só {len(selected_idx_0)}+{len(selected_idx_1)} exemplos disponíveis "
              f"(pedidos {n_per_class_0}+{n_per_class_1}): a métrica prediz poucos pontos numa classe.")

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


def bc_passos_2_a_4(
    X: np.ndarray,
    y_true: np.ndarray,
    learned_metric: LearnedMetric,
    nome_classe_0: str,
    nome_classe_1: str,
    k_exemplos: int,
    problem_name: str,
    verbose: bool = True,
    prompt_variant: str = "default",
    nome_feature_0: str = "x1",
    nome_feature_1: str = "x2",
) -> Tuple[Optional[ConsistencyMetrics], np.ndarray, np.ndarray, float, float, int, Dict, np.ndarray]:
    """Problemas B e C, Passos 2 a 4 (antes de 13/10/2026: Fases B e C): consistência em um novo problema com a métrica estimada no A.

    Consistência (seção IV.3): apenas o vetor W é transferido do Problema A; os
    centróides são recalculados LOCALMENTE a partir dos rótulos que o LLM atribui neste
    problema, isolando o efeito dos pesos da métrica do efeito da posição dos centróides
    entre distribuições diferentes.

    Exemplos few-shot (grade K_EXEMPLOS_METRICA_BC, padrão = grade completa [0, 4, 10, 20, 40];
    a CLI --bc-zero-shot reduz a [0] numa execução): rotulados por W_A COM OS CENTRÓIDES
    DO A (`bc_passo4_exemplos_da_metrica`, maior margem por classe), porque os rótulos locais
    do LLM ainda não existem antes da coleta; a consistência medida usa centróides locais
    (decisão D4, 09/10/2026, declarada aqui, no log e no rodapé das tabelas). Os exemplos
    ficam FORA do conjunto de avaliação (anti-leakage).

    Colapso do LLM neste problema (`colapsou`): consistência, κ e F1 ficam NaN
    (`consistency_metrics=None`, `extra['collapsed']=1`); até 08/10 usava-se os centróides
    do A como fallback, o que media outra coisa.

    Retorna:
        consistency_metrics: ConsistencyMetrics LLM vs. métrica (None no colapso)
        y_llm: decisões do LLM no conjunto de teste
        y_metric_test: predições da métrica no conjunto de teste (W_A + centróides locais); None se o LLM colapsou
        llm_accuracy: acurácia (vs rótulo real) do LLM, contexto
        metric_accuracy: acurácia (vs rótulo real) da métrica, contexto
        n_malformed: respostas malformadas do LLM
        extra: {'collapsed', 'consistency_centroides_a', 'kappa_centroides_a', 'y_true_test',
                'nome_feature_0', 'nome_feature_1'} — a ablação usa W_A com os
                centróides do A (só CSV e log)
        X_test: pontos efetivamente avaliados (sem os exemplos few-shot)

    (Antes de 13/10/2026: `phase_consistency_test`.)
    """
    if verbose:
        print("\n  ══════════════════════════════════════════════════════════════")
        print(f"  {problem_name}: " + ("zero-shot (Passo 2) e consistência (Passo 3)" if k_exemplos == 0
                                        else f"k={k_exemplos} exemplos da métrica (Passo 4), com a consistência (Passo 3)"))
        print("  ══════════════════════════════════════════════════════════════")
        print("  Aplicando a métrica W estimada no Problema A a este novo problema.")
        print("  Objetivo: verificar se o LLM concorda com a métrica em dados não vistos.")

    examples = None
    selected_indices = np.array([], dtype=int)

    if k_exemplos > 0:
        if verbose:
            print(f"\n  Selecionando {k_exemplos} exemplos few-shot (maior margem por classe, "
                  "rotulados por W_A com os centróides do A)...")

        examples, _, _, selected_indices = bc_passo4_exemplos_da_metrica(
            X, learned_metric.centroids, learned_metric.w,
            k_exemplos, nome_classe_0, nome_classe_1, verbose=verbose
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
        nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
        prompt_variant=prompt_variant
    )
    extra = {'collapsed': 0, 'consistency_centroides_a': float('nan'),
             'kappa_centroides_a': float('nan'), 'y_true_test': y_true_test,
             'nome_feature_0': nome_feature_0, 'nome_feature_1': nome_feature_1}

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
    # ABLAÇÃO: também calculamos a consistência com centróides transferidos (do
    # Problema A), gravada no CSV como consistency_centroides_a_b/c (I10, 09/10/2026).
    llm_accuracy = accuracy_score(y_true_test, y_llm)
    if colapsou(y_llm):
        # Colapso em B/C (M6): sem centróides locais não há consistência a medir.
        extra['collapsed'] = 1
        y_metric_test = None   # sem métrica: os painéis de fechamento mostram "sem dados" (13/10/2026; antes -1)
        if verbose:
            print(f"    ⚠ Colapso do LLM em {problem_name} ({int(np.sum(y_llm == 0))}/"
                  f"{int(np.sum(y_llm == 1))}; piso = {MIN_PONTOS_CLASSE_MINORITARIA}): "
                  "consistência, Cohen's κ e F1-Macro ficam NaN (collapsed=1).")
        return None, y_llm, y_metric_test, llm_accuracy, float('nan'), n_malformed, extra, X_test

    local_centroids = compute_centroids(X_test, y_llm)
    # Predições da métrica usando W do Problema A + centróides locais (abordagem principal)
    y_metric_test = predict_with_metric(X_test, local_centroids, learned_metric.w)

    # Ablação: predições com centróides TRANSFERIDOS do Problema A (só CSV e log).
    y_metric_centroides_a = predict_with_metric(
        X_test, learned_metric.centroids, learned_metric.w
    )
    consistency_centroides_a = compute_consistency_metrics(y_llm, y_metric_centroides_a)
    extra['consistency_centroides_a'] = float(consistency_centroides_a.accuracy)
    extra['kappa_centroides_a'] = float(consistency_centroides_a.cohen_kappa)

    consistency_metrics = compute_consistency_metrics(y_llm, y_metric_test)
    metric_accuracy = accuracy_score(y_true_test, y_metric_test)

    if verbose:
        n_0 = np.sum(y_metric_test == 0)
        n_1 = np.sum(y_metric_test == 1)
        print(f"    Predições da métrica (W_A + centróides locais): Classe 0={n_0}, Classe 1={n_1}")
        print(f"    [Ablação] Consistência com centróides LOCAIS:       {consistency_metrics.accuracy:.1%}")
        print(f"    [Ablação] Consistência com centróides do A (coluna consistency_centroides_a): {consistency_centroides_a.accuracy:.1%}")
        delta = consistency_metrics.accuracy - consistency_centroides_a.accuracy
        if delta > 0.05:
            print(f"    → Centróides locais melhoram em {delta:.1%}: recálculo justificado.")
        elif delta > -0.05:
            print(f"    → Diferença marginal ({delta:+.1%}): ambas as abordagens comparáveis.")
        else:
            print(f"    → ⚠ Centróides do A superiores ({delta:+.1%}): investigar.")

        print(f"\n  RESULTADOS: {problem_name}")
        print(f"    Consistência (LLM vs. métrica W_A + centróides locais): {consistency_metrics.accuracy:.1%}")
        print(f"    Cohen's κ: {consistency_metrics.cohen_kappa:.3f}")
        print(f"    F1-Macro: {consistency_metrics.f1_score:.3f}")
        print("    Pontos concordantes/discordantes (contagem): "
              f"{consistency_metrics.n_agreements}/{consistency_metrics.n_disagreements}")
        print(f"    Acurácia (vs rótulo real) [contexto]: LLM {llm_accuracy:.1%} | métrica {metric_accuracy:.1%}")
        print(f"    Conjunto de teste: {len(X_test)} pontos (excluídos {len(selected_indices)} exemplos few-shot)")
        if consistency_metrics.accuracy >= 0.85:
            print("    O LLM MANTÉM consistência elevada com a métrica (W_A + centróides locais).")
        elif consistency_metrics.accuracy >= 0.7:
            print("    O LLM mantém consistência PARCIAL com a métrica (W_A + centróides locais).")
        else:
            print("    ⚠ O LLM NÃO mantém consistência com a métrica (W_A + centróides locais).")

    return consistency_metrics, y_llm, y_metric_test, llm_accuracy, metric_accuracy, n_malformed, extra, X_test


# =============================================================================
# PROBLEMA E: LLM COMO APRENDIZ
# =============================================================================

def especialista_rotular(
    X: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray
) -> np.ndarray:
    """
    Classifica pontos usando a métrica conhecida do especialista.
    O especialista atua como "verdade absoluta" (rótulo de referência) no Problema E:
    o objetivo do LLM é aprender a reproduzir essa classificação a partir de exemplos.
    """
    return predict_with_metric(X, expert_centroids, expert_w)


# =============================================================================
# VALIDAÇÃO DO ORÁCULO: ALGORITMOS RECUPERAM W CONHECIDO?
# =============================================================================

def oraculo_passos_1_a_3(
    X_e: np.ndarray,
    expert_configs: List[dict],
    random_seed: int,
    verbose: bool = True,
) -> List[dict]:
    """Valida se os algoritmos de otimização inversa recuperam um W conhecido.

    Os três testes são as configurações da seção IV.5 do texto do orientador, os
    Problemas 1/2/3 da tabela 17, e desde 14/10/2026 rodam sobre UM MESMO conjunto, as
    gaussianas do Problema E (centróides `EXPERT_CENTROIDS`, que diferem nas duas
    coordenadas), só trocando a métrica "na frente" (sugestão da reunião de 02/10/2026):
      - Problema 1 = "Oráculo sobre E3": W = [1, 1] (euclidiano);
      - Problema 2 = "Oráculo sobre E1": W = [0,3; 1,5];
      - Problema 3 = "Oráculo sobre E2": W = [1,5; 0,3].
    Até 13/10/2026 o euclidiano rodava sobre os dados do Problema B ("Oráculo sobre B",
    `expert_name` "B"); os leitores (`plots.py`, `gerar_tabelas.py`) continuam aceitando
    esse nome nas execuções antigas. O Problema A de até 14/10/2026 não servia de base: com
    centróides (-2, 0) e (2, 0) a diferença de distâncias não dependia de w2 e a métrica não era
    identificável (mesmo motivo da remoção das gaussianas anisotrópicas extras em 13/10/2026);
    desde 15/10/2026 o A tem centróides (-2, -0,5) e (2, 0,5), mas o oráculo segue sobre o E.

    Os rótulos do oráculo são a regra do centróide mais próximo sob o W verdadeiro
    (`especialista_rotular`, centróides GERADORES, não `compute_centroids`), para eliminar
    a ambiguidade W/centróides e testar só a recuperação. A medida gravada é a ACURÁCIA da
    métrica recuperada vs os rótulos do oráculo (coluna `accuracy`).

    (Antes de 13/10/2026: `run_oracle_validation`; até 14/10/2026 recebia `X_b, y_b`.)
    """
    results = []

    # Euclidiano primeiro (Problema 1), depois os anisotrópicos na ordem de EXPERT_CONFIGS.
    configs = sorted(expert_configs, key=lambda c: 0 if np.allclose(c["w"], c["w"][0]) else 1)
    if verbose:
        print("\n  ═══ Oráculo sobre E3, E1 e E2: três métricas conhecidas sobre as gaussianas do Problema E ═══")

    for expert_cfg in configs:
        expert_w = np.asarray(expert_cfg["w"], dtype=float)
        expert_name = expert_cfg["name"]
        expert_desc = expert_cfg["desc"]
        expert_centroids = EXPERT_CENTROIDS

        # Rótulos gerados pelo especialista (oráculo determinístico)
        y_oraculo_e = especialista_rotular(X_e, expert_w, expert_centroids)

        n_classes = len(np.unique(y_oraculo_e))
        if n_classes < 2:
            if verbose:
                print(f"    ⚠ Especialista {expert_name} gerou apenas 1 classe — pulando")
            continue

        if verbose:
            print(f"\n  ── Oráculo sobre {expert_name}: W={expert_w.tolist()} ({expert_desc}), "
                  f"centróides={np.asarray(expert_centroids).tolist()}, n={len(X_e)}")

        for algo_name, train_fn in _oraculo_estimadores(X_e, y_oraculo_e, expert_centroids):
            w_recovered, _ = train_fn()
            _oraculo_registrar(
                results, X_e, y_oraculo_e, expert_centroids, expert_w, w_recovered,
                random_seed, f"Oráculo sobre {expert_name}", expert_name, algo_name, verbose,
            )

    return results

def d_passo5_teto(
    X_ml: np.ndarray,
    y_ml: np.ndarray,
    random_seed: int,
    n_features_list: Tuple[int, ...] = (2, 3, 4),
    verbose: bool = True,
) -> List[dict]:
    """Teto do Problema D: métrica diagonal ajustada ao rótulo real da meia-lua.

    A meia-lua é genuinamente não-linear: **não existe** um W diagonal que a
    *gere* (por isso não faz sentido gerar a meia-lua a partir de um W). Mas o
    orientador observou que pode existir uma **mudança de métrica que a aproxima**
    (~391-502s: "você pode ter um W que aproxima... uma mudança de métrica que
    aproxima um pouco da meia-lua").

    Aqui partimos dos rótulos REAIS da meia-lua e ajustamos W em espaço aumentado
    (2, 3, 4 atributos) via Perceptron e NNLS, medindo a ACURÁCIA da métrica
    resultante vs rótulo real (é o teto do Problema D, coluna `accuracy_vs_true`;
    "fidelidade" é reservado à comparação com o LLM). A função COMPARA R², R³ e R⁴
    (sem expectativa a priori sobre qual aproxima melhor a fronteira não-linear).

    Equivalência (09/10/2026): este cálculo é idêntico ao regime `passo5_teto` do
    Problema D em `adg_passos_2_a_5` (mesmos pontos, mesmo W ajustado ao rótulo real;
    tabela 18 = linhas "teto" da tabela 10). Fica como função própria para que
    `--apenas oraculo` (0 chamadas) produza o teto sem rodar os passos com LLM.
    """
    results = []
    if len(np.unique(y_ml)) < 2:
        return results

    for n_feat in n_features_list:
        X_aug = augment_features(X_ml, n_feat)
        centroids = compute_centroids(X_aug, y_ml)

        w_perc, _ = train_relaxed_perceptron(
            X_aug, y_ml, centroids,
            **PERCEPTRON_PARAMS,
            verbose=False,
            use_best_effort=True,
        )
        w_nnls, _ = train_least_squares_inverse(X_aug, y_ml, centroids, verbose=False)

        for algo_name, w_rec in (("perceptron", w_perc), ("nnls", w_nnls)):
            y_pred = predict_with_metric(X_aug, centroids, w_rec)
            acc_true = accuracy_score(y_ml, y_pred)
            n_err = int(np.sum(y_pred != y_ml))
            results.append({
                'random_seed': random_seed,
                'n_features': n_feat,
                'algorithm': algo_name,
                # Comparação com o rótulo REAL (teto do Problema D) = acurácia,
                # não fidelidade (que é vs LLM). Todos os pesos (R³/R⁴ inclusive).
                'accuracy_vs_true': acc_true,
                'n_errors': n_err,
                'n_samples': len(y_ml),
                **{f'w_recovered_{i}': float(w_rec[i]) if i < len(w_rec) else float('nan')
                   for i in range(4)},
            })
            if verbose:
                print(
                    f"    [Problema D, teto n_feat={n_feat} {algo_name.upper():10s}] "
                    f"acurácia vs rótulo real={acc_true:.1%} ({n_err}/{len(y_ml)} erros) | "
                    f"W=[{', '.join(f'{wi:.3f}' for wi in w_rec)}]"
                )
    return results




def _oraculo_estimadores(X, y, centroids):
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


def _oraculo_registrar(
    results, X, y_true, centroids, true_w, w_recovered,
    random_seed, problem_name, expert_name, algo_name, verbose,
):
    """Calcula métricas e adiciona resultado do oráculo à lista."""
    # Acurácia da métrica recuperada vs rótulos do ORÁCULO (gerados pelo W conhecido):
    # coluna `accuracy` (M8; até 08/10 chamava-se `fidelity`, nome reservado à
    # comparação com o LLM).
    y_pred = predict_with_metric(X, centroids, w_recovered)
    accuracy = accuracy_score(y_true, y_pred)

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
              f"Acurácia vs oráculo={accuracy:.1%} | Cosseno={cosine_sim:.4f} | "
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
        'accuracy': accuracy,
    })


def e_passo3_selecionar_exemplos(
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
        "mixed":  Metade fáceis + metade difíceis, EXATAMENTE 50/50 no total (I15,
                  09/10/2026): o excedente de cada classe alterna entre easy e hard
                  (classe 0 recebe ⌈n0/2⌉ easy, classe 1 recebe ⌊n1/2⌋ easy), de modo
                  que n=10 dá 5 fáceis + 5 difíceis (antes: 4 + 6).
                  → Representação balanceada; hipótese a priori (H4), avaliada por execução
        "random": Seleção aleatória (linha de base)
                  → Sem critério estratégico; referência comparativa

    Args:
        X: Matriz de características dos pontos
        y_expert: Rótulos do especialista para todos os pontos
        expert_w: Pesos da métrica do especialista
        expert_centroids: Centróides do especialista
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

    # Calcula as margens (confiança) de todos os pontos sob a métrica do especialista
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

        # Excedente alternado entre as classes para o total ficar 50/50 (I15):
        # classe 0 arredonda easy para CIMA, classe 1 para BAIXO.
        n_easy_0 = n_per_class_0 - n_per_class_0 // 2
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

    # Monta a lista de exemplos com os rótulos do ESPECIALISTA (não do ground truth)
    examples = []
    for i in selected_0:
        examples.append((X[i, 0], X[i, 1], nome_classe_0))
    for i in selected_1:
        examples.append((X[i, 0], X[i, 1], nome_classe_1))

    return examples, selected_indices


def e_passo4_exemplos_diluicao(
    X: np.ndarray,
    y_expert: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    n_hard_fixed: int,
    n_easy_added: int,
    nome_classe_0: str,
    nome_classe_1: str,
    verbose: bool = False
) -> Tuple[List[Tuple[float, float, str]], np.ndarray]:
    """Seleciona exemplos para o experimento de diluição: N hard fixos + M easy adicionais.

    Primeiro seleciona os top-N pontos hard (menor margem), depois adiciona
    M pontos easy (maior margem), excluindo os hard já selecionados. A seleção é
    DETERMINÍSTICA (só por margem): não há sorteio, logo não há random_state.
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


def e_passo5_reordenar_exemplos(
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


def e_passos_2_3_aprendiz(
    X_e: np.ndarray,
    y_gt_e: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    k_exemplos: int,
    strategy: str,
    nome_classe_0: str,
    nome_classe_1: str,
    provider: str,
    model_name: str,
    temperature: float,
    random_seed: int,
    repeticao: int,
    verbose: bool = True,
    y_llm_cached: Optional[np.ndarray] = None,
    n_malformed_cached: int = 0,
    tag: Optional[str] = None,
) -> ResultadoAprendiz:
    """
    Problema E, Passos 2 e 3 (antes de 13/10/2026: Fase E), LLM como Aprendiz.

    O especialista (com métrica W_expert conhecida) rotula os dados.
    O LLM recebe exemplos few-shot do especialista e tenta
    aprender o padrão de classificação do especialista.

    Testa se o LLM consegue MELHORAR seu alinhamento com o especialista
    conforme mais exemplos são fornecidos, e como a dificuldade dos exemplos
    afeta a curva de aprendizado.

    `y_llm_cached` (I16, 09/10/2026): rótulos do LLM já coletados para os MESMOS
    pontos de teste (só faz sentido em zero-shot, cujo prompt não depende do
    especialista): a coleta é feita 1× por (modelo, seed) e reaproveitada em
    E1/E2/E3, recalculando as métricas por especialista. `tag` sobrescreve
    CURRENT_TAG no registro das interações.

    Todas as medidas "do especialista" (acurácia vs rótulo real, contagens por
    classe) são calculadas sobre o CONJUNTO DE TESTE (P6), como na diluição e no
    viés de ordem.

    (Antes de 13/10/2026: `phase_e_llm_as_learner`.)
    """
    shot_label = "zero-shot" if k_exemplos == 0 else f"{k_exemplos}-shot"

    if verbose:
        print("\n  ══════════════════════════════════════════════════════════════")
        print(f"  PROBLEMA E, PASSO {2 if k_exemplos == 0 else 3}: LLM COMO APRENDIZ ({shot_label}, estratégia={strategy})")
        print(f"  Métrica do Especialista W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
        print("  ══════════════════════════════════════════════════════════════")
        print("  O especialista rotula os dados com sua métrica conhecida.")
        print("  O LLM recebe exemplos do especialista e deve APRENDER o padrão de classificação.")

    # (a) Especialista rotula todos os pontos com a métrica W_expert conhecida (Passo 1 do E)
    y_expert = especialista_rotular(X_e, expert_w, expert_centroids)

    if verbose:
        n_0 = np.sum(y_expert == 0)
        n_1 = np.sum(y_expert == 1)
        print("\n  (a) Especialista rotula os pontos (métrica W_expert conhecida; Passo 1 do E)")
        print(f"    Rótulos do especialista (todos os pontos): Classe 0={n_0}, Classe 1={n_1}")

    # (b) Seleciona os k exemplos conforme a estratégia (Passo 3; nada a selecionar no Passo 2)
    examples = None
    example_indices = np.array([], dtype=int)

    if k_exemplos > 0:
        if verbose:
            print(f"\n  (b) Selecionando {k_exemplos} exemplos (estratégia={strategy}; Passo 3)...")
            estrategia_descricao = {
                "easy": "pontos LONGE da fronteira (mais fáceis)",
                "hard": "pontos PRÓXIMOS à fronteira (mais difíceis)",
                "mixed": "metade fáceis + metade difíceis (balanceado)",
                "random": "seleção aleatória (linha de base)"
            }
            print(f"    Estratégia: {strategy.upper()} — {estrategia_descricao.get(strategy, '')}")

        examples, example_indices = e_passo3_selecionar_exemplos(
            X_e, y_expert, expert_w, expert_centroids,
            n_examples=k_exemplos,
            strategy=strategy,
            nome_classe_0=nome_classe_0,
            nome_classe_1=nome_classe_1,
            random_state=random_seed + repeticao,
            verbose=verbose
        )

    # (c) Conjunto de teste: todos os pontos NÃO usados como exemplos
    # Isso garante avaliação honesta: o LLM não é testado nos pontos que recebeu como exemplos
    all_indices = np.arange(len(X_e))
    test_mask = np.ones(len(X_e), dtype=bool)
    if len(example_indices) > 0:
        test_mask[example_indices] = False
    test_indices = all_indices[test_mask]

    X_test = X_e[test_indices]
    y_expert_test = y_expert[test_indices]
    y_gt_test = y_gt_e[test_indices]
    # Acurácia (vs rótulo real) do especialista sobre o TESTE (P6): mesma base da
    # diluição e do viés de ordem.
    expert_accuracy = accuracy_score(y_gt_test, y_expert_test)

    if verbose:
        print(f"\n  (d) LLM classifica {len(X_test)} pontos de teste ({shot_label}; Passo {2 if k_exemplos == 0 else 3})...")
        if k_exemplos > 0:
            print(f"    O LLM receberá {k_exemplos} exemplos rotulados pelo especialista como contexto.")

    # (d) LLM classifica os pontos de teste com os exemplos no prompt (Passo 2 sem exemplos, Passo 3 com)
    if y_llm_cached is not None and k_exemplos == 0:
        y_llm_test, n_malformed = np.asarray(y_llm_cached), n_malformed_cached
        if verbose:
            print("    [zero-shot reaproveitado: coleta única por (modelo, seed), I16]")
    else:
        y_llm_test, n_malformed = collect_llm_decisions(
            X_test, nome_classe_0, nome_classe_1,
            examples=examples, verbose=verbose,
            label_prefix=f"[Problema E, Passo {2 if k_exemplos == 0 else 3}] ", tag=tag,
        )

    # (e) Medidas: acurácia (LLM vs especialista), Cohen's κ, F1-Macro e acurácia (vs rótulo real) de contexto
    consistency = compute_consistency_metrics(y_llm_test, y_expert_test)
    llm_accuracy_vs_gt = accuracy_score(y_gt_test, y_llm_test)

    if verbose:
        print("\n  ═══════════════════════════════════════════════════════════")
        print(f"  RESULTADOS: PROBLEMA E, PASSO {2 if k_exemplos == 0 else 3} ({shot_label}, estratégia={strategy})")
        print("  ═══════════════════════════════════════════════════════════")
        print(f"    Acurácia (LLM vs especialista):        {consistency.accuracy:.1%}")
        print(f"    Cohen's κ:                             {consistency.cohen_kappa:.3f}")
        print(f"    F1-Macro:                              {consistency.f1_score:.3f}")
        print("    ")
        print(f"    Acurácia (vs rótulo real) do LLM [contexto]:          {llm_accuracy_vs_gt:.1%}")
        print(f"    Acurácia (vs rótulo real) do especialista [contexto]: {expert_accuracy:.1%}")
        print(f"    Pontos concordantes/discordantes (contagem): {consistency.n_agreements}/{consistency.n_disagreements}")

        if consistency.accuracy > 0.85:
            print("    → O LLM APRENDEU com sucesso o padrão de classificação do especialista!")
        elif consistency.accuracy > 0.7:
            print("    → O LLM tem ALINHAMENTO PARCIAL com o especialista.")
        else:
            print(f"    → O LLM NÃO conseguiu aprender o padrão do especialista com {k_exemplos} exemplos.")

    return ResultadoAprendiz(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        random_seed=random_seed,
        k=k_exemplos,
        example_strategy=strategy,
        nomes_classes=(nome_classe_0, nome_classe_1),
        repeticao=repeticao,
        accuracy_llm_vs_expert=consistency.accuracy,
        kappa_llm_vs_expert=consistency.cohen_kappa,
        f1_llm_vs_expert=consistency.f1_score,
        accuracy_expert_vs_gt=expert_accuracy,
        accuracy_llm_vs_gt=llm_accuracy_vs_gt,
        n_classe_0_expert=int(np.sum(y_expert_test == 0)),
        n_classe_1_expert=int(np.sum(y_expert_test == 1)),
        n_classe_0_llm=int(np.sum(y_llm_test == 0)),
        n_classe_1_llm=int(np.sum(y_llm_test == 1)),
        n_disagreements=consistency.n_disagreements,
        n_total_test=len(X_test),
        n_malformed_responses=n_malformed,
        expert_w=expert_w.copy(),
    )


# =============================================================================
# EXPERIMENTO COMPLETO (PROBLEMAS A, B E C)
# =============================================================================

def abc_executar(
    X_train_a: np.ndarray,
    y_true_train_a: np.ndarray,
    X_b: np.ndarray,
    y_true_b: np.ndarray,
    X_c: np.ndarray,
    y_true_c: np.ndarray,
    k_exemplos: int,
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
    prompt_variant: str = "default",
    nome_feature_0: str = "x1",
    nome_feature_1: str = "x2",
) -> Tuple[ResultadoABC, Optional[LearnedMetric], np.ndarray, float, float, int, dict]:
    """Executa o experimento completo: Problema A (Passos 2 e 4) + Problemas B e C (Passos 2 a 4).

    Utiliza cache do Problema A para evitar chamadas redundantes à API do LLM —
    uma vez estimada, Ŵ_LLM é reutilizada em todos os tamanhos de k_exemplos
    e repetições com os mesmos nomes de classe, atributos, variante e semente.
    Devolve também um dict `detail` (rótulos e métricas de B/C, inclusive as do NNLS
    PAREADAS com os mesmos rótulos do LLM, `metrics_nnls_b/c`, que alimentam
    `bloco1_a_passo4_perceptron_vs_nnls`) para as figuras por semente.

    (Antes de 13/10/2026: `run_complete_experiment`.)
    """
    shot_label = "zero-shot" if k_exemplos == 0 else f"{k_exemplos}-shot"

    if verbose:
        print_section(
            f"EXPERIMENTO: Problemas B e C, Passo {2 if k_exemplos == 0 else 4} ({shot_label}) | Classes: {nome_classe_0}/{nome_classe_1} | Rep: {repeticao+1} | Seed: {random_seed}",
            "─"
        )

    total_malformed = 0

    if learned_metric_cache is not None:
        if verbose:
            print("\n  [Usando a métrica do Problema A (Passo 4) do cache — evitando chamadas redundantes à API]")
        learned_metric = learned_metric_cache
        learned_metric_nnls = learned_metric_nnls_cache
        y_llm_train_a = y_llm_train_a_cache
        fidelity_a = fidelity_cache
        fidelity_nnls_a = fidelity_nnls_cache
        llm_accuracy_a = llm_accuracy_a_cache
        total_malformed += n_malformed_a_cache if n_malformed_a_cache else 0
    elif y_llm_train_a_cache is not None:
        # Colapso do Problema A (Passo 2) já em cache: reaproveita os rótulos, sem nova coleta.
        learned_metric, learned_metric_nnls = None, None
        y_llm_train_a = y_llm_train_a_cache
        fidelity_a = fidelity_nnls_a = float('nan')
        llm_accuracy_a = llm_accuracy_a_cache
        total_malformed += n_malformed_a_cache if n_malformed_a_cache else 0
    else:
        if verbose:
            print("\n  [Problema A, Passos 2 e 4: estimando a métrica W a partir das decisões do LLM...]")
        result_a = a_passo2_zero_shot_e_metrica(
            X_train_a, y_true_train_a,
            nome_classe_0, nome_classe_1, verbose=verbose,
            prompt_variant=prompt_variant,
            nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
            random_seed=random_seed,
        )
        learned_metric, learned_metric_nnls, y_llm_train_a, fidelity_a, fidelity_nnls_a, llm_accuracy_a, n_malformed_a = result_a
        total_malformed += n_malformed_a

    if learned_metric is None:
        # Colapso do LLM em classe única no Problema A (Passo 2): não há métrica. As medidas
        # ficam NaN e collapsed=1; main() guarda esse resultado FORA de
        # resultados_abc (só vai ao CSV), para não entrar em médias e figuras.
        # (Até 08/10/2026 gravava sentinelas 0,5 e κ=0 como se fossem medições.)
        _nan = float('nan')
        return (
            ResultadoABC(
                provider=provider, model_name=model_name, temperature=temperature,
                random_seed=random_seed, k_passo4_bc=k_exemplos,
                nomes_classes=(nome_classe_0, nome_classe_1), repeticao=repeticao,
                fidelidade_problema_a=_nan, acuracia_llm_vs_gt_problema_a=llm_accuracy_a,
                consistencia_problema_b=_nan, kappa_problema_b=_nan, f1_problema_b=_nan,
                acuracia_llm_vs_gt_problema_b=_nan, acuracia_metrica_vs_gt_problema_b=_nan,
                consistencia_problema_c=_nan, kappa_problema_c=_nan, f1_problema_c=_nan,
                acuracia_llm_vs_gt_problema_c=_nan, acuracia_metrica_vs_gt_problema_c=_nan,
                w_aprendido=np.full(2, _nan), gamma_otimo=_nan, margem_minima=_nan,
                n_classe_0_problema_a=np.sum(y_llm_train_a == 0),
                n_classe_1_problema_a=np.sum(y_llm_train_a == 1),
                n_classe_0_problema_b=0, n_classe_1_problema_b=0,
                n_classe_0_problema_c=0, n_classe_1_problema_c=0,
                n_disagreements_b=0, n_disagreements_c=0,
                n_malformed_responses=total_malformed,
                w_direction=None,
                w_cosine_sim_nnls=_nan,
                prompt_variant=prompt_variant,
                feature_names=(nome_feature_0, nome_feature_1),
                collapsed=1,
            ),
            None, y_llm_train_a, _nan, llm_accuracy_a, total_malformed, {}
        )

    # --- Problemas B e C com Perceptron (algoritmo principal) ---
    metrics_b, y_llm_b, y_pred_metric_b, llm_accuracy_b, metric_accuracy_b, n_malformed_b, extra_b, X_test_b = bc_passos_2_a_4(
        X_b, y_true_b, learned_metric,
        nome_classe_0, nome_classe_1, k_exemplos, f"PROBLEMA B, PASSO {2 if k_exemplos == 0 else 4}", verbose=verbose,
        prompt_variant=prompt_variant, nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
    )
    total_malformed += n_malformed_b

    metrics_c, y_llm_c, y_pred_metric_c, llm_accuracy_c, metric_accuracy_c, n_malformed_c, extra_c, X_test_c = bc_passos_2_a_4(
        X_c, y_true_c, learned_metric,
        nome_classe_0, nome_classe_1, k_exemplos, f"PROBLEMA C, PASSO {2 if k_exemplos == 0 else 4}", verbose=verbose,
        prompt_variant=prompt_variant, nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
    )
    total_malformed += n_malformed_c
    _nan = float('nan')
    collapsed_b, collapsed_c = int(extra_b['collapsed']), int(extra_c['collapsed'])

    def _m(cm, attr):
        return _nan if cm is None else float(getattr(cm, attr))

    # --- Problemas B e C com NNLS (segundo estimador, robustez ao método) ---
    # PAREADO: usa os MESMOS y_llm já coletados (0 chamadas); só troca o W. É o que
    # alimenta `bloco1_a_passo4_perceptron_vs_nnls` e `bloco1_graphic_06` (I12, 09/10/2026).
    nnls_detail = {}
    if learned_metric_nnls is not None:
        def _nnls_bc(X_test, y_llm, collapsed):
            if collapsed:
                return None, None
            local_c = compute_centroids(X_test, y_llm)
            y_pred = predict_with_metric(X_test, local_c, learned_metric_nnls.w)
            return compute_consistency_metrics(y_llm, y_pred), y_pred
        metrics_nnls_b, y_pred_nnls_b = _nnls_bc(X_test_b, y_llm_b, collapsed_b)
        metrics_nnls_c, y_pred_nnls_c = _nnls_bc(X_test_c, y_llm_c, collapsed_c)

        if verbose:
            print("\n  --- Comparação Perceptron vs NNLS (Problemas B e C, Passo 3, mesmos rótulos do LLM) ---")
            print(f"    [Perceptron] Consistência B: {_m(metrics_b, 'accuracy'):.1%} | Cohen's κ B: {_m(metrics_b, 'cohen_kappa'):.3f}")
            print(f"    [NNLS]       Consistência B: {_m(metrics_nnls_b, 'accuracy'):.1%} | Cohen's κ B: {_m(metrics_nnls_b, 'cohen_kappa'):.3f}")
            print(f"    [Perceptron] Consistência C: {_m(metrics_c, 'accuracy'):.1%} | Cohen's κ C: {_m(metrics_c, 'cohen_kappa'):.3f}")
            print(f"    [NNLS]       Consistência C: {_m(metrics_nnls_c, 'accuracy'):.1%} | Cohen's κ C: {_m(metrics_nnls_c, 'cohen_kappa'):.3f}")

        nnls_detail = {
            'metrics_nnls_b': metrics_nnls_b, 'metrics_nnls_c': metrics_nnls_c,
            'y_pred_nnls_b': y_pred_nnls_b, 'y_pred_nnls_c': y_pred_nnls_c,
            'fidelity_nnls_a': fidelity_nnls_a,
            'w_nnls': learned_metric_nnls.w,
            'metric_accuracy_nnls_b': (_nan if collapsed_b else float(accuracy_score(extra_b['y_true_test'], y_pred_nnls_b))),
            'metric_accuracy_nnls_c': (_nan if collapsed_c else float(accuracy_score(extra_c['y_true_test'], y_pred_nnls_c))),
        }

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
        ResultadoABC(
            provider=provider, model_name=model_name, temperature=temperature,
            random_seed=random_seed, k_passo4_bc=k_exemplos,
            nomes_classes=(nome_classe_0, nome_classe_1), repeticao=repeticao,
            fidelidade_problema_a=fidelity_a,
            acuracia_llm_vs_gt_problema_a=llm_accuracy_a,
            consistencia_problema_b=_m(metrics_b, 'accuracy'),
            kappa_problema_b=_m(metrics_b, 'cohen_kappa'),
            f1_problema_b=_m(metrics_b, 'f1_score'),
            acuracia_llm_vs_gt_problema_b=llm_accuracy_b,
            acuracia_metrica_vs_gt_problema_b=metric_accuracy_b,
            consistencia_problema_c=_m(metrics_c, 'accuracy'),
            kappa_problema_c=_m(metrics_c, 'cohen_kappa'),
            f1_problema_c=_m(metrics_c, 'f1_score'),
            acuracia_llm_vs_gt_problema_c=llm_accuracy_c,
            acuracia_metrica_vs_gt_problema_c=metric_accuracy_c,
            w_aprendido=learned_metric.w,
            gamma_otimo=learned_metric.gamma,
            margem_minima=learned_metric.margem_minima,
            n_classe_0_problema_a=np.sum(y_llm_train_a == 0),
            n_classe_1_problema_a=np.sum(y_llm_train_a == 1),
            n_classe_0_problema_b=np.sum(y_llm_b == 0),
            n_classe_1_problema_b=np.sum(y_llm_b == 1),
            n_classe_0_problema_c=np.sum(y_llm_c == 0),
            n_classe_1_problema_c=np.sum(y_llm_c == 1),
            n_disagreements_b=(float('nan') if collapsed_b else metrics_b.n_disagreements),   # NaN no colapso (14/10/2026)
            n_disagreements_c=(float('nan') if collapsed_c else metrics_c.n_disagreements),
            n_malformed_responses=total_malformed,
            w_direction=w_dir,
            w_cosine_sim_nnls=cos_sim_nnls,
            prompt_variant=prompt_variant,
            feature_names=(nome_feature_0, nome_feature_1),
            # `collapsed` marca SÓ o colapso do A no Passo 2 (linhas de resultados_abc_colapsados);
            # colapso em B ou C fica em collapsed_b/c, com as colunas de consistência NaN e a
            # fidelidade do A preservada (14/10/2026; antes collapsed=1 também aqui, o que apagava
            # linhas válidas do A das tabelas e figuras de vieses).
            collapsed=0,
            collapsed_b=collapsed_b, collapsed_c=collapsed_c,
            consistencia_centroides_a_problema_b=float(extra_b['consistency_centroides_a']),
            consistencia_centroides_a_problema_c=float(extra_c['consistency_centroides_a']),
            kappa_centroides_a_problema_b=float(extra_b['kappa_centroides_a']),
            kappa_centroides_a_problema_c=float(extra_c['kappa_centroides_a']),
        ),
        learned_metric, y_llm_train_a, fidelity_a, llm_accuracy_a, total_malformed,
        {
            'y_llm_b': y_llm_b, 'y_llm_c': y_llm_c,
            'y_metric_b': y_pred_metric_b, 'y_metric_c': y_pred_metric_c,
            'metrics_b': metrics_b, 'metrics_c': metrics_c,
            'collapsed_b': collapsed_b, 'collapsed_c': collapsed_c,
            'learned_metric_nnls': learned_metric_nnls,
            'llm_accuracy_b': llm_accuracy_b, 'llm_accuracy_c': llm_accuracy_c,
            **nnls_detail,
        }
    )


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def _abc_executar_configuracao(X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                              k_exemplos, nome_0, nome_1, rep, provider, model_name,
                              temperature, seed, cache_a_passo2, verbose,
                              prompt_variant="default", nome_feature_0="x1", nome_feature_1="x2"):
    """Helper para executar um experimento A-C com cache.

    A chave do cache do Problema A inclui provider/model, nomes de classe, variante de
    prompt e nomes de atributo (o experimento de nomes de atributo passa por aqui
    desde 09/10/2026, M5/I9, com consistência B/C por centróides locais como a base).
    """
    cache_key = (provider, model_name, seed, nome_0, nome_1, prompt_variant, nome_feature_0, nome_feature_1)
    if cache_key in cache_a_passo2:
        cached = cache_a_passo2[cache_key]
        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail = abc_executar(
            X_train_a.copy(), y_train_a.copy(),
            X_b.copy(), y_b.copy(), X_c.copy(), y_c.copy(),
            k_exemplos=k_exemplos, nome_classe_0=nome_0, nome_classe_1=nome_1,
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
            prompt_variant=prompt_variant,
            nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
        )
    else:
        result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail = abc_executar(
            X_train_a.copy(), y_train_a.copy(),
            X_b.copy(), y_b.copy(), X_c.copy(), y_c.copy(),
            k_exemplos=k_exemplos, nome_classe_0=nome_0, nome_classe_1=nome_1,
            repeticao=rep, provider=provider, model_name=model_name,
            temperature=temperature, random_seed=seed,
            verbose=verbose,
            prompt_variant=prompt_variant,
            nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
        )
        # Guarda também o colapso ('metric': None): os k_exemplos/reps seguintes e o
        # Passo 3 do Problema A reaproveitam os rótulos em vez de recoletar 150 pontos.
        cache_a_passo2[cache_key] = {
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
# PASSOS 2 A 5 DOS PROBLEMAS A, D E G
# Zero-shot | few-shot com exemplos rotulados pelo GROUND TRUTH (k/2 por classe,
# sem split, LLM classifica os n−k restantes) | W ajustado ao ground truth (teto).
# =============================================================================

def _tag(bloco: str, problema: str, passo: int, regime: str, k: int, rep: int, seed: int,
         extra: str = "") -> str:
    """Tag de auditoria gravada em LLM_INTERACTIONS (ver CURRENT_TAG).

    Os 7 campos fixos (`bloco|problema|passo|regime|k|rep|seed`) são o prefixo que
    `audit_interactions.por_regime` lê; `extra` (ex.: `nomes=Mulher/Homem|atributos=
    peso/altura`, no Problema G) entra depois deles, separado por `|`.
    `passo` é o passo DO PROBLEMA em que a chamada acontece (ver passos_por_problema.md);
    `regime` diz de onde vêm os exemplos: `passo2_zero_shot`, `passo3_exemplos_rotulo_real`,
    `exemplos_da_metrica`, `exemplos_rotulo_real`, `zero_shot` (aprendiz sem exemplos),
    `vies_<fator>`, a estratégia (`easy`…), `diluicao` ou a ordenação.
    (Antes de 13/10/2026: `_make_tag(fase, …)`, com o campo `fase=` no lugar de `passo=`.)
    """
    tag = f"{bloco}|problema={problema}|passo={passo}|regime={regime}|k={k}|rep={rep}|seed={seed}"
    return f"{tag}|{extra}" if extra else tag


def passo3_exemplos_rotulo_real(X: np.ndarray, y_true: np.ndarray, k: int,
                          nome_classe_0: str, nome_classe_1: str,
                          random_state: int,
                          X_display: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray, np.ndarray]:
    """k exemplos rotulados pelo GROUND TRUTH, k/2 por classe, sorteio sem reposição.

    Reusa `e_passo3_selecionar_exemplos(strategy="random")`, que já estratifica por
    classe de `y_expert` (aqui, o ground truth) com `RandomState(random_state)`.
    `w`/centróides só alimentam o cálculo de margens interno (irrelevante no "random").
    O SORTEIO usa `X` (espaço da estimação); os valores MOSTRADOS ao LLM nos exemplos
    vêm de `X_display` quando dado (Problema G2: kg e m no prompt, métrica no espaço
    normalizado), senão de `X`. Como o sorteio depende só de (y_true, k, random_state),
    variantes de atributo com o mesmo `seed + rep` recebem os MESMOS índices.
    Devolve (examples [(x1, x2, label)], índices sorteados, máscara dos pontos que o
    LLM vai classificar — anti-leakage: os exemplos ficam FORA da avaliação).
    """
    _, idx = e_passo3_selecionar_exemplos(
        X, y_true, np.ones(2), compute_centroids(X, y_true), k, "random",
        nome_classe_0, nome_classe_1, random_state=random_state, verbose=False,
    )
    idx = np.asarray(idx, dtype=int)
    X_show = X if X_display is None else np.asarray(X_display, dtype=float)
    examples = [(float(X_show[i, 0]), float(X_show[i, 1]),
                 nome_classe_0 if y_true[i] == 0 else nome_classe_1) for i in idx]
    eval_mask = np.ones(len(X), dtype=bool)
    eval_mask[idx] = False
    return examples, idx, eval_mask


def _linhas_passos_2_a_5(res: Optional[dict], *, base: dict, problem: str, n_feat: int,
                   regime: str, k: int, rep: int, random_state_sorteio,
                   n_total: int, n_examples: int,
                   y_llm_eval: np.ndarray, y_true_eval: np.ndarray,
                   X_full: np.ndarray, y_true_full: np.ndarray,
                   y_llm_zero: Optional[np.ndarray] = None,
                   n_malformed: int = 0, example_indices: str = "") -> List[dict]:
    """Converte um resultado de `passo4_estimar_metrica_r2_r3_r4` (ou None = colapso em classe
    única) em DUAS linhas (perceptron, nnls) do CSV `bloco1_ad_passos_2_a_5`.

    `accuracy_metric_vs_true_eval` é medida nos pontos avaliados (n−k, pareada com a
    fidelidade); `accuracy_metric_vs_true_full` aplica a métrica aos n pontos (comparável
    entre regimes). No regime `passo5_teto` (teto do estimador, M9) não há LLM: `fidelity_vs_llm`,
    `llm_accuracy_vs_true` e `n_classe_*_llm` são NaN, `accuracy_teto_vs_true` guarda a
    acurácia do W ajustado ao rótulo real sobre os n pontos (NaN nos outros regimes) e
    `fidelity_teto_vs_llm_passo2` mede quanto a métrica-GT reproduz os rótulos zero-shot
    do LLM (uma fidelidade; NaN se o zero-shot colapsou; até 08/10/2026 chamava-se
    accuracy_gt_metric_vs_llm_zero e, até 13/10/2026, fidelity_gt_metric_vs_llm_zero).
    No G as mesmas linhas vão para `bloco3_g_passos_2_a_5` (com problem_name, subproblem,
    class_names e feature_names acrescentados por `fg_executar`).
    """
    nan = float("nan")
    # No regime passo5_teto y_llm_eval É o rótulo real: contagens e acurácia do LLM não se aplicam.
    if regime == "passo5_teto":
        n0 = n1 = nan
    else:
        n0 = int(np.sum(y_llm_eval == 0)); n1 = int(np.sum(y_llm_eval == 1))
    llm_acc = nan if regime == "passo5_teto" else float(accuracy_score(y_true_eval, y_llm_eval))
    common = dict(base, problem=problem, n_features=n_feat, regime=regime, k=k, rep=rep,
                  random_state_sorteio=random_state_sorteio, n_total=n_total,
                  n_examples=n_examples, n_eval=int(len(y_llm_eval)),
                  n_classe_0_llm=n0, n_classe_1_llm=n1, llm_accuracy_vs_true=llm_acc,
                  n_malformed=n_malformed, example_indices=example_indices)
    algos = (("perceptron", "w_perc", "fidelity_perc_vs_llm", "accuracy_perc_vs_true", "n_errors_perc_vs_true"),
             ("nnls", "w_nnls", "fidelity_nnls_vs_llm", "accuracy_nnls_vs_true", "n_errors_nnls_vs_true"))
    rows = []
    cos_sim = nan
    if res is not None:
        wp, wn = np.asarray(res["w_perc"], float), np.asarray(res["w_nnls"], float)
        if np.linalg.norm(wp) > 0 and np.linalg.norm(wn) > 0:
            cos_sim = float(np.dot(wp, wn) / (np.linalg.norm(wp) * np.linalg.norm(wn)))
    for algo, wkey, fkey, akey, ekey in algos:
        row = dict(common, algorithm=algo)
        if res is None:
            row.update(collapsed=1, fidelity_vs_llm=nan, accuracy_metric_vs_true_eval=nan,
                       accuracy_metric_vs_true_full=nan, accuracy_teto_vs_true=nan,
                       fidelity_teto_vs_llm_passo2=nan,
                       n_errors_metric_vs_true_eval=nan, w_ratio=nan, gamma=nan, margem_minima=nan, cos_sim_perc_nnls=nan)
            for i in range(4):
                row[f"w_{i}"] = nan; row[f"w_unit_{i}"] = nan
        else:
            w = np.asarray(res[wkey], float)
            X_full_aug = augment_features(X_full, n_feat)
            y_full_pred = predict_with_metric(X_full_aug, res["centroids"], w)
            acc_full = float(accuracy_score(y_true_full, y_full_pred))
            gt_vs_zero = nan
            if (regime == "passo5_teto" and y_llm_zero is not None and len(y_llm_zero) == len(y_full_pred)
                    and not colapsou(y_llm_zero)):   # zero-shot colapsado não entra em médias (13/10/2026)
                gt_vs_zero = float(accuracy_score(y_llm_zero, y_full_pred))
            norm = float(np.linalg.norm(w))
            w_unit = w / norm if norm > 0 else w
            is_gt = regime == "passo5_teto"
            row.update(collapsed=0,
                       # No teto não há LLM: a "fidelidade" seria vs o próprio GT (M9).
                       fidelity_vs_llm=(nan if is_gt else float(res[fkey])),
                       accuracy_metric_vs_true_eval=float(res[akey]),
                       accuracy_metric_vs_true_full=acc_full,
                       accuracy_teto_vs_true=(acc_full if is_gt else nan),
                       fidelity_teto_vs_llm_passo2=gt_vs_zero,
                       n_errors_metric_vs_true_eval=int(res[ekey]),
                       w_ratio=(float(w[0] / w[1]) if w[1] != 0 else nan),
                       gamma=(float(res["gamma_perc"]) if algo == "perceptron" else nan),
                       # 15/10/2026 (D): margem bruta mínima/||w|| da solução; < 0 = não separa os rótulos († nas tabelas)
                       margem_minima=(float(res.get("margem_minima_perc", nan)) if algo == "perceptron" else nan),
                       cos_sim_perc_nnls=cos_sim)
            for i in range(4):
                row[f"w_{i}"] = float(w[i]) if i < len(w) else nan
                row[f"w_unit_{i}"] = float(w_unit[i]) if i < len(w_unit) else nan
        rows.append(row)
    return rows


def adg_passos_2_a_5(
    X: np.ndarray, y_true: np.ndarray,
    nome_classe_0: str, nome_classe_1: str,
    problem: str, seed: int, provider: str, model_name: str, temperature: float,
    n_features_list: Tuple[int, ...] = (2,),
    ks_rotulo_real: Optional[List[int]] = None,
    y_llm_zero_cached: Optional[np.ndarray] = None,
    n_malformed_zero: int = 0,
    verbose: bool = False, label_prefix: str = "",
    X_display: Optional[np.ndarray] = None,
    nome_feature_0: str = "x1", nome_feature_1: str = "x2",
    bloco: str = "bloco1", tag_extra: str = "",
) -> dict:
    """Passos 2 a 5 (antes de 13/10/2026: Fase A) em UM conjunto de pontos (Problema A, D ou treino do G), SEM split.

    Regimes:
      passo2_zero_shot — rótulos do LLM sem exemplos (reaproveita `y_llm_zero_cached` quando
                     dado; senão coleta: n chamadas);
      passo3_exemplos_rotulo_real (k) — k exemplos rotulados pelo RÓTULO REAL (k/2 por classe, random_state
                     = seed + rep), LLM classifica os n−k restantes (n−k chamadas por rep),
                     W estimado sobre as respostas do LLM, centróides pelos rótulos do LLM;
      passo5_teto  — W ajustado a y_true nos n pontos (0 chamadas): teto do estimador.
    Para cada regime e cada n_feat em `n_features_list`, Perceptron e NNLS via
    `passo4_estimar_metrica_r2_r3_r4` (o LLM vê sempre só x1, x2). Colapso (`colapsou`: classe
    única ou minoritária < MIN_PONTOS_CLASSE_MINORITARIA) gera linhas com NaN
    (collapsed=1), nunca é pulado.

    Generalização para o Problema G (Metodologia IV.7, 13/10/2026), sem alterar A e D:
      X_display      — valores MOSTRADOS ao LLM (exemplos few-shot e pontos a classificar);
                       sorteio, estimação, centróides e avaliações usam `X`. None = `X`.
      nome_feature_* — nomes dos atributos no prompt (repassados a `collect_llm_decisions`).
      bloco/tag_extra — tag de auditoria `<bloco>|problema=…|passo=…|regime=…|k=…|rep=…|seed=…`
                       seguida de `tag_extra` (ex.: `nomes=Mulher/Homem|atributos=peso/altura`).
    A chave de `CACHE_ROTULOS_LLM` inclui o nome completo do problema, os nomes de
    classe e os nomes de atributo: variantes nunca reaproveitam rótulos umas das outras.

    Devolve {'rows', 'labels', 'metrics', 'zero_shot_results', 'n_calls'}:
      rows    — linhas do CSV bloco1_ad_passos_2_a_5 (regime × k × rep × n_feat × algoritmo);
      labels  — {(regime, k, rep): {eval_idx, example_idx, y_llm_eval, examples}};
      metrics — {(regime, k, rep, n_feat): {w_perc, w_nnls, centroids, gamma_perc, collapsed}};
      zero_shot_results — {n_feat: dict de passo4_estimar_metrica_r2_r3_r4 | None}.

    (Antes de 13/10/2026: `phase_a_protocol`.)
    """
    ks_rotulo_real = list(K_EXEMPLOS_ROTULO_REAL) if ks_rotulo_real is None else list(ks_rotulo_real)
    base = {"provider": provider, "model": model_name, "temperature": temperature, "random_seed": seed}
    out = {"rows": [], "labels": {}, "metrics": {}, "zero_shot_results": {}, "n_calls": 0}
    n = len(X)
    all_idx = np.arange(n)
    y_llm_zero_holder = {"y": None}
    X_show = X if X_display is None else np.asarray(X_display, dtype=float)
    if X_show.shape != X.shape:
        raise ValueError(f"X_display {X_show.shape} deve ter a mesma forma de X {X.shape}")
    nomes_at = (nome_feature_0, nome_feature_1)
    # Chave de cache: (provider, modelo, seed, problema completo, classes, atributos, k, rep).
    chave = lambda k, rep: (provider, model_name, seed, problem, nome_classe_0, nome_classe_1,  # noqa: E731
                            nome_feature_0, nome_feature_1, k, rep)

    def _estimate(regime, k, rep, rs, X_eval, y_true_eval, y_llm_eval, n_malf, eval_idx, ex_idx, examples):
        # Regra única de colapso (I13); o regime passo5_teto usa y_true, que não colapsa.
        collapsed = regime != "passo5_teto" and colapsou(y_llm_eval)
        out["labels"][(regime, k, rep)] = {
            "eval_idx": np.asarray(eval_idx), "example_idx": np.asarray(ex_idx, dtype=int),
            "y_llm_eval": np.asarray(y_llm_eval), "examples": list(examples),
        }
        if collapsed and verbose:
            print(f"{label_prefix}  ⚠ {regime} k={k} rep={rep}: colapso (classe única ou minoritária < {MIN_PONTOS_CLASSE_MINORITARIA}) "
                  f"({int(np.sum(y_llm_eval == 0))}/{int(np.sum(y_llm_eval == 1))}) — linhas com NaN")
        for n_feat in n_features_list:
            res = None if collapsed else passo4_estimar_metrica_r2_r3_r4(
                X_eval, y_true_eval, n_feat, nome_classe_0, nome_classe_1, *nomes_at,
                y_llm_cached=y_llm_eval, verbose=False,
            )
            out["metrics"][(regime, k, rep, n_feat)] = {
                "w_perc": None if res is None else np.asarray(res["w_perc"], float),
                "w_nnls": None if res is None else np.asarray(res["w_nnls"], float),
                "centroids": None if res is None else np.asarray(res["centroids"], float),
                "gamma_perc": None if res is None else res["gamma_perc"],
                "collapsed": int(res is None),
            }
            if regime == "passo2_zero_shot":
                out["zero_shot_results"][n_feat] = res
            out["rows"].extend(_linhas_passos_2_a_5(
                res, base=base, problem=problem, n_feat=n_feat, regime=regime, k=k, rep=rep,
                random_state_sorteio=rs, n_total=n, n_examples=len(examples),
                y_llm_eval=y_llm_eval, y_true_eval=y_true_eval, X_full=X, y_true_full=y_true,
                y_llm_zero=y_llm_zero_holder["y"], n_malformed=n_malf,
                example_indices=";".join(str(int(i)) for i in ex_idx),
            ))
            if verbose and res is not None:
                if regime == "passo5_teto":
                    print(f"{label_prefix}  {regime:<12} k={k:>2} rep={rep} R{n_feat}: "
                          f"teto (acurácia do W ajustado ao rótulo real) perc={res['accuracy_perc_vs_true']:.1%} "
                          f"nnls={res['accuracy_nnls_vs_true']:.1%}")
                else:
                    print(f"{label_prefix}  {regime:<12} k={k:>2} rep={rep} R{n_feat}: "
                          f"fidelidade perc={res['fidelity_perc_vs_llm']:.1%} nnls={res['fidelity_nnls_vs_llm']:.1%} | "
                          f"acurácia (vs rótulo real) métrica={res['accuracy_perc_vs_true']:.1%} "
                          f"LLM={res['llm_accuracy_vs_true']:.1%}")

    # 1. zero-shot
    cache_key = chave(0, 0)
    if y_llm_zero_cached is not None:
        y_llm_zero, n_malf_zero = np.asarray(y_llm_zero_cached), n_malformed_zero
    elif cache_key in CACHE_ROTULOS_LLM:
        y_llm_zero = CACHE_ROTULOS_LLM[cache_key]["y_llm"]
        n_malf_zero = CACHE_ROTULOS_LLM[cache_key]["n_malformed"]
    else:
        y_llm_zero, n_malf_zero = collect_llm_decisions(
            X_show, nome_classe_0, nome_classe_1, examples=None, verbose=verbose,
            label_prefix=f"{label_prefix}[Problema {problem}, Passo 2] ",
            nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
            tag=_tag(bloco, problem, 2, "passo2_zero_shot", 0, 0, seed, extra=tag_extra),
        )
        out["n_calls"] += n
    CACHE_ROTULOS_LLM[cache_key] = {"y_llm": y_llm_zero, "eval_idx": all_idx,
                                         "example_idx": np.array([], dtype=int), "n_malformed": n_malf_zero}
    y_llm_zero_holder["y"] = y_llm_zero
    _estimate("passo2_zero_shot", 0, 0, None, X, y_true, y_llm_zero, n_malf_zero, all_idx, [], [])

    # 2. few-shot (ground truth nos exemplos; valores mostrados vêm de X_show).
    #    15/10/2026: os sorteios de todos os (k, rep) são feitos primeiro e as coletas que não
    #    estão no cache saem num ÚNICO lote (`collect_llm_decisions_lote`); a estimação roda
    #    depois, na mesma ordem de antes (k crescente, rep crescente).
    plano = []
    for k in ks_rotulo_real:
        for rep in range(reps_para(k)):
            rs = seed + rep
            examples, ex_idx, eval_mask = passo3_exemplos_rotulo_real(
                X, y_true, k, nome_classe_0, nome_classe_1, rs, X_display=X_show)
            ck = chave(k, rep)
            job = None if ck in CACHE_ROTULOS_LLM else dict(
                X=X_show[eval_mask], nome_classe_0=nome_classe_0, nome_classe_1=nome_classe_1,
                examples=examples, verbose=verbose,
                label_prefix=f"{label_prefix}[Problema {problem}, Passo 3, k={k}, rep {rep + 1}] ",
                nome_feature_0=nome_feature_0, nome_feature_1=nome_feature_1,
                tag=_tag(bloco, problem, 3, "passo3_exemplos_rotulo_real", k, rep, seed, extra=tag_extra),
            )
            plano.append((k, rep, rs, examples, ex_idx, eval_mask, ck, job))
    coletas = iter(collect_llm_decisions_lote([pl[-1] for pl in plano if pl[-1] is not None],
                                              rotulo=f"{label_prefix}[Problema {problem}, Passo 3] "))
    for k, rep, rs, examples, ex_idx, eval_mask, ck, job in plano:
        if job is None:
            y_llm_k = CACHE_ROTULOS_LLM[ck]["y_llm"]; n_malf_k = CACHE_ROTULOS_LLM[ck]["n_malformed"]
        else:
            y_llm_k, n_malf_k = next(coletas)
            out["n_calls"] += int(eval_mask.sum())
            CACHE_ROTULOS_LLM[ck] = {"y_llm": y_llm_k, "eval_idx": all_idx[eval_mask],
                                          "example_idx": ex_idx, "n_malformed": n_malf_k}
        _estimate("passo3_exemplos_rotulo_real", k, rep, rs, X[eval_mask], y_true[eval_mask], y_llm_k, n_malf_k,
                  all_idx[eval_mask], ex_idx, examples)

    # 3. ground truth (teto do estimador; 0 chamadas)
    _estimate("passo5_teto", 0, 0, None, X, y_true, y_true, 0, all_idx, [], [])
    return out


def bc_passo3_consistencia(X: np.ndarray, y_llm: np.ndarray, y_true: np.ndarray, problem: str,
                         metrics_a: Dict[tuple, dict], base: dict, n_features: int = 2) -> List[dict]:
    """Consistência do LLM (rótulos ZERO-SHOT já coletados em B ou C) contra TODAS as
    métricas estimadas no Problema A (zero-shot, cada k/rep, ground truth), sem chamadas.

    Só W é transferido; os centróides são LOCAIS (rótulos do LLM no próprio problema).
    Se o LLM colapsou em B/C (`colapsou`), consistência, κ e F1 ficam NaN com
    `collapsed_llm=1` (M6; até 08/10 o fallback usava os centróides do A).
    Uma linha por (métrica de origem, algoritmo) no CSV bloco1_bc_passo3_consistencia.

    Só as métricas do A com o mesmo `n_features` (padrão R²) são transferidas; as
    de R³/R⁴ não entram na transferência zero-shot.
    """
    nan = float("nan")
    rows = []
    X_aug = augment_features(X, n_features)
    y_llm = np.asarray(y_llm); y_true = np.asarray(y_true)
    collapsed_llm = colapsou(y_llm)
    local_c = None if collapsed_llm else compute_centroids(X_aug, y_llm)
    n0, n1 = int(np.sum(y_llm == 0)), int(np.sum(y_llm == 1))
    llm_acc = float(accuracy_score(y_true, y_llm))
    for (regime, k, rep, n_feat), m in sorted(metrics_a.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        if n_feat != n_features:
            continue
        for algo, wkey in (("perceptron", "w_perc"), ("nnls", "w_nnls")):
            row = dict(base, problem=problem, source_problem="A", metrica_regime=regime, metrica_k=k,
                       metrica_rep=rep, n_features=n_features, algorithm=algo, n_eval=int(len(X)),
                       n_classe_0_llm=n0, n_classe_1_llm=n1, llm_accuracy_vs_true=llm_acc,
                       collapsed_llm=int(collapsed_llm), metrica_fidelidade_a=nan, eleita_passo6=0)
            w = m.get(wkey)
            cents = local_c
            if m.get("collapsed") or w is None or cents is None:
                row.update(collapsed_metrica=int(bool(m.get("collapsed"))), consistency=nan, kappa=nan, f1=nan,
                           n_agreements=nan, n_disagreements=nan, metric_accuracy_vs_true=nan)
            else:
                y_metric = predict_with_metric(X_aug, cents, np.asarray(w, float))
                cm = compute_consistency_metrics(y_llm, y_metric)
                row.update(collapsed_metrica=0, consistency=float(cm.accuracy), kappa=float(cm.cohen_kappa),
                           f1=float(cm.f1_score), n_agreements=int(cm.n_agreements),
                           n_disagreements=int(cm.n_disagreements),
                           metric_accuracy_vs_true=float(accuracy_score(y_true, y_metric)))
            rows.append(row)
    return rows


def a_passo6_eleger_metrica(rows_a: List[dict], rows_bc: List[dict]) -> Dict[str, tuple]:
    """Elege, por modelo, a "melhor métrica do A": maior média de fidelity_vs_llm
    (Perceptron, problem=A, n_features=2, sem colapso) entre os regimes DERIVADOS DO LLM
    (passo2_zero_shot e cada k); `passo5_teto` não concorre (sua fidelidade é vs rótulo real).
    Marca eleita_passo6=1 e preenche metrica_fidelidade_a nas linhas de B/C.
    Devolve {model: (regime, k, fidelidade_media)}."""
    best: Dict[str, tuple] = {}
    if not rows_a:
        return best
    df = pd.DataFrame(rows_a)
    sub = df[(df["problem"] == "A") & (df["algorithm"] == "perceptron") & (df["n_features"] == 2)
             & (df["collapsed"] == 0) & (df["regime"] != "passo5_teto")]
    for model, g in sub.groupby("model"):
        means = g.groupby(["regime", "k"])["fidelity_vs_llm"].mean()
        if means.empty:
            continue
        regime, k = means.idxmax()
        best[model] = (regime, int(k), float(means.max()))
        for r in rows_bc:
            if r.get("model") != model:
                continue
            key = (r["metrica_regime"], r["metrica_k"])
            r["metrica_fidelidade_a"] = float(means.get(key, float("nan")))
            r["eleita_passo6"] = int(key == (regime, int(k)))
    return best


def _executar_figura(tarefa):
    """Roda UMA tarefa de figura `(func, args, kwargs)` capturando o que ela imprime (vai para o
    log pelo processo pai). Usada em série ou nos processos de `_figuras_em_paralelo`."""
    func, args, kwargs = tarefa
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            func(*args, **kwargs)
        except Exception as exc:
            print(f"  ⚠ Falha em {func.__name__} ({kwargs.get('filename', '')}): {exc}")
    return buf.getvalue()


def _figuras_em_paralelo(tarefas: list, rotulo: str = "") -> None:
    """Executa tarefas de figura em `FIGURAS_PROCESSOS` processos (ou em série) e imprime as
    saídas na ordem das tarefas. Qualquer falha do pool cai para a execução em série.

    Os processos filhos são criados por `spawn`: cada um REIMPORTA este módulo do zero, logo
    só vê o estado de import (as grades reduzidas do `--rapido`, aplicadas em `main()` por
    `aplicar_modo_rapido`, e qualquer outro global alterado em tempo de execução NÃO valem
    nos filhos; as tarefas devem receber tudo por argumento). Uma exceção dentro de uma
    figura é capturada por `_executar_figura`, vira a linha "⚠ Falha em ..." no log e NÃO
    entra em `falhas_fatais` (a execução termina com exit 0 mesmo sem essa figura)."""
    if not tarefas:
        return
    t0 = time.time()
    saidas = None
    if FIGURAS_PROCESSOS > 1 and len(tarefas) > 1:
        try:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(FIGURAS_PROCESSOS, len(tarefas)),
                                     mp_context=_mp.get_context("spawn")) as pool:
                saidas = list(pool.map(_executar_figura, tarefas))
        except Exception as exc:
            print(f"  ⚠ Figuras em paralelo falharam ({type(exc).__name__}: {exc}); gerando em série.")
            saidas = None
    if saidas is None:
        saidas = [_executar_figura(t) for t in tarefas]
    for txt in saidas:
        if txt.strip():
            print(txt.rstrip("\n"))
    print(f"  {rotulo}{len(tarefas)} figuras em {time.time() - t0:.1f}s "
          f"({'série' if FIGURAS_PROCESSOS <= 1 or len(tarefas) < 2 else f'{min(FIGURAS_PROCESSOS, len(tarefas))} processos'})", flush=True)


def _simbolos_tarefa(d: dict, problem: str, seed: int, model_name: str, best_m, bc, pasta_execucao: str) -> None:
    """Tarefa de figura: símbolos dos Passos 2 a 4 (A, D) / 2 e 3 (B, C) de um (modelo, seed)."""
    try:
        for _f in _plot_simbolos_passos(d, problem, seed, model_name, best_m, bc, pasta_execucao):
            print(f"  Gráfico salvo: {os.path.basename(_f)}")
    except Exception as exc:
        print(f"  ⚠ Falha nas figuras com símbolos ({problem}, seed {seed}): {exc}")


def _fechamento_seed_tarefa(sdata, seed_key, abc_m, e_m, model_name: str, pasta_execucao: str) -> None:
    """Tarefa de figura: as seis figuras de fechamento de um (modelo, seed) (final_graphic_01 a 06)."""
    print(f"\n  Gerando visualizações detalhadas para seed {seed_key} ({_model_alias(model_name)})...")
    for nome, func, extra in (
        ("final_graphic_01_ABCE_passo1_dados", plot_final_dados, ()),
        ("final_graphic_03_ABC_acertos_erros", plot_final_acertos_erros, ()),
        ("final_graphic_04_A_w_algoritmos", plot_final_w_algoritmos, ()),
        ("final_graphic_02_ABC_matrizes_confusao", plot_final_matrizes_confusao, ()),
        ("final_graphic_05_ABC_margem", plot_final_margem, ()),
        ("final_graphic_06_painel_resumo", plot_final_painel_resumo, (abc_m, e_m)),
    ):
        fname = llm_asset(pasta_execucao, f"{nome}_seed{seed_key}.png", model_name)
        func(sdata, seed_key, *extra, filename=fname)
        print(f"  Gráfico salvo: {os.path.basename(fname)}")


def _plot_simbolos_passos(d: dict, problem: str, seed: int, model_name: str, best_m,
                           bc: Optional[dict], pasta_execucao: str) -> List[str]:
    """Figuras com símbolos por classe dos Passos 2 a 4 (A, D) e 2 e 3 (B, C) para (modelo, seed):
    bloco1_graphic_02 (A: Passo 1 rótulo real | Passo 2 LLM zero-shot | Passo 3 LLM k-shot | Passo 4 métrica),
    bloco1_graphic_11 (B e C: rótulo real | LLM zero-shot | melhor métrica do A, com os centróides
    geradores e a rotação) e bloco1_graphic_13 (D: rótulo real | LLM zero-shot | LLM k-shot |
    métrica R² | métrica R⁴); k = 10 se estiver na grade, senão o menor de K_EXEMPLOS_ROTULO_REAL. Cada figura sai em `_color` (azul/vermelho) e `_gray`
    (preto, desenhada duas vezes); a lista devolvida traz o caminho `_gray` de cada uma.
    Colapso do zero-shot (F13): os painéis de métrica são omitidos e o título anota
    "zero-shot colapsado (n0/n1)" em vez de omitir em silêncio."""
    from data_problems import PROBLEM_A_CENTERS, PROBLEM_B_CENTERS, PROBLEM_C_CENTERS
    X, y_true, labels, metrics = d['X'], d['y_true'], d['labels'], d['metrics']
    saved = []
    zero = labels.get(("passo2_zero_shot", 0, 0))
    y_llm_zero = zero['y_llm_eval'] if zero else None

    def _part(y):
        y = np.asarray(y); return f"n0={int(np.sum(y == 0))}, n1={int(np.sum(y == 1))}"

    def _metric_scheme(regime, k, rep, n_feat, suffix, title):
        m = metrics.get((regime, k, rep, n_feat))
        if not m or m.get('collapsed') or m.get('w_perc') is None:
            return None
        y_m = predict_with_metric(augment_features(X, n_feat), m['centroids'], m['w_perc'])
        return {'suffix': suffix, 'title': title, 'y': y_m, 'subtitle': _part(y_m),
                'boundary': {'w': m['w_perc'], 'centroids': m['centroids'], 'n_features': n_feat}}

    schemes = [{'suffix': 'passo1_rotulo_real', 'title': 'Rótulo real (Passo 1)', 'y': y_true, 'subtitle': _part(y_true)}]
    zero_colapsado = y_llm_zero is not None and colapsou(y_llm_zero)
    if y_llm_zero is not None:
        schemes.append({'suffix': 'passo2_llm_zero_shot', 'title': 'LLM zero-shot (Passo 2)' + (' (colapso)' if zero_colapsado else ''),
                        'y': y_llm_zero, 'subtitle': _part(y_llm_zero)})
    sc = _metric_scheme("passo2_zero_shot", 0, 0, 2, 'passo4_metrica', 'Métrica estimada (Passo 4; zero-shot, R²)')
    if sc: schemes.append(sc)
    nota_colapso = f"; zero-shot colapsado ({_part(y_llm_zero)}), sem métrica" if zero_colapsado else ""

    def _esquema_passo3():
        # Painel "LLM k-shot" (k=10 ou o menor k disponível): A e D (F13).
        fs = labels.get(("passo3_exemplos_rotulo_real", 10, 0)) or next((labels[k] for k in sorted(labels) if k[0] == "passo3_exemplos_rotulo_real"), None)
        if fs is None:
            return None
        k_fs = next(k[1] for k in labels if labels[k] is fs)
        y_plot = np.asarray(y_true).copy(); y_plot[fs['eval_idx']] = fs['y_llm_eval']
        return {'suffix': f'passo3_llm_{k_fs}shot', 'title': f'LLM {k_fs}-shot (Passo 3; exemplos = rótulo real)',
                'y': y_plot, 'subtitle': _part(fs['y_llm_eval']), 'example_idx': fs['example_idx']}

    if problem == "A":
        fs_scheme = _esquema_passo3()
        if fs_scheme is not None:
            schemes.insert(2, fs_scheme)    # ordem canônica: Passo 1, 2, 3 (LLM k-shot), 4 (métrica)
        fname = llm_asset(pasta_execucao, f"bloco1_graphic_02_A_passos_2_a_4_simbolos_seed{seed}.png", model_name)
        plot_rotulos_simbolos(X, schemes, fname, suptitle=f"Problema A, Passos 2 a 4, seed {seed}, {model_name}{nota_colapso}")
        saved.append(com_cor(fname, "gray"))   # figura de símbolos: sai em _color e _gray; devolve a _gray
        # B e C: rótulos zero-shot do LLM × melhor métrica do A (centróides locais)
        if bc:
            if best_m:
                reg, kb = best_m[0], best_m[1]
            else:
                reg, kb = "passo2_zero_shot", 0
            mb = metrics.get((reg, kb, 0, 2))
            if not mb or mb.get('collapsed'):
                mb = metrics.get(("passo2_zero_shot", 0, 0, 2)); reg, kb = "passo2_zero_shot", 0
            for prob, Xk, yk, ylk, centers in (("B", bc['X_b'], bc['y_b'], bc['y_llm_b'], PROBLEM_B_CENTERS),
                                                ("C", bc['X_c'], bc['y_c'], bc['y_llm_c'], PROBLEM_C_CENTERS)):
                sch = [{'suffix': 'passo1_rotulo_real', 'title': 'Rótulo real (Passo 1)', 'y': yk, 'subtitle': _part(yk)},
                       {'suffix': 'passo2_llm_zero_shot', 'title': 'LLM zero-shot (Passo 2)', 'y': ylk, 'subtitle': _part(ylk)}]
                if mb and not mb.get('collapsed') and not colapsou(ylk):   # regra única de colapso
                    c_local = compute_centroids(np.asarray(Xk), np.asarray(ylk))
                    y_m = predict_with_metric(np.asarray(Xk), c_local, mb['w_perc'])
                    lab = "zero-shot" if reg == "passo2_zero_shot" else f"{kb}-shot"
                    sch.append({'suffix': 'passo3_metrica_eleita', 'title': f"Passo 3: melhor métrica do A ({lab}), centróides locais",
                                'y': y_m, 'subtitle': _part(y_m),
                                'rep': (None if reg == "passo2_zero_shot" else 0),   # rep real (0 = 1ª repetição)
                                'boundary': {'w': mb['w_perc'], 'centroids': c_local, 'n_features': 2}})
                sentido = "horária" if prob == "B" else "anti-horária"
                ref = {'from': PROBLEM_A_CENTERS, 'to': centers,
                       'label': f"centróides geradores: A {tuple(PROBLEM_A_CENTERS[0])}/{tuple(PROBLEM_A_CENTERS[1])} → "
                                f"{prob} {tuple(centers[0])}/{tuple(centers[1])} (rotação {sentido}, ±1,5)"}
                if colapsou(ylk):
                    sch[1]['title'] += ' (colapso)'
                fname = llm_asset(pasta_execucao, f"bloco1_graphic_11_{prob}_passos_2_3_simbolos_seed{seed}.png", model_name)
                plot_rotulos_simbolos(np.asarray(Xk), sch, fname, reference_centroids=ref,
                                    suptitle=f"Problema {prob}, Passos 2 e 3: seed {seed}, {model_name}"
                                             + (f"; zero-shot colapsado ({_part(ylk)}), sem métrica" if colapsou(ylk) else ""))
                saved.append(com_cor(fname, "gray"))   # figura de símbolos: sai em _color e _gray; devolve a _gray
    elif problem == "D":
        sc4 = _metric_scheme("passo2_zero_shot", 0, 0, 4, 'passo4_metrica_r4', 'Métrica estimada (Passo 4; zero-shot, R⁴: x1², x2²)')
        if sc4: schemes.append(sc4)
        fs_scheme = _esquema_passo3()
        if fs_scheme is not None:
            schemes.insert(2, fs_scheme)    # ordem canônica: Passo 1, 2, 3 (LLM k-shot), 4 (métrica R² e R⁴)
        fname = llm_asset(pasta_execucao, f"bloco1_graphic_13_D_passos_2_a_4_simbolos_seed{seed}.png", model_name)
        plot_rotulos_simbolos(X, schemes, fname, suptitle=f"Problema D (meia-lua), Passos 2 a 4, seed {seed}, {model_name}{nota_colapso}")
        saved.append(com_cor(fname, "gray"))   # figura de símbolos: sai em _color e _gray; devolve a _gray
    return saved

# =============================================================================
# PROBLEMAS F (meia-lua, Bloco 2) e G1/G2 (peso × altura, Bloco 3): split 70/30, LLM como aprendiz.
# Por semente: split 70/30; zero-shot no treino (Passo 2) com métrica em R²/R³/R⁴ (o LLM vê só x1, x2) para
# eleger a métrica de maior acurácia vs rótulo real (G, Passo 5); aprendizado em contexto (F Passo 3 / G Passo 6, k ∈
# K_EXEMPLOS_APRENDIZ, 3 reps, k/2 exemplos por classe) avaliada no teste. O
# ESPECIALISTA é o rótulo real quando `expert_labels` é dado (Problema F,
# decisão D1 de 09/10/2026) e a métrica eleita no Passo 5 nos demais (G1/G2).
# O Problema D (meia-lua com o LLM como especialista, Bloco 1) NÃO passa por aqui: adg_passos_2_a_5.
# Assets: bloco23_* (CSVs/curvas), final_graphic_07 (ponto a ponto); bloco1_graphic_12/14
# só se RUN_PROBLEMA_D=False.
# =============================================================================

def passo4_estimar_metrica_r2_r3_r4(
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
    """Estimação da métrica (Passo 4 de A, D e G; Passo 2 do F) generalizada para 2, 3 ou 4 atributos.

    O LLM SEMPRE vê apenas 2 atributos (x1, x2) — os atributos adicionais
    (x1·x2 ou x1², x2²) são aplicadas APENAS na construção da métrica W.
    Isso é proposital: queremos saber se uma métrica diagonal em espaço
    aumentado captura a não-linearidade no critério decisional do LLM.

    Retorna dict com Perceptron + NNLS, fidelidade (vs LLM) e
    acurácia (vs rótulo real); devolve None se o LLM colapsou (`colapsou`: classe
    única ou minoritária < MIN_PONTOS_CLASSE_MINORITARIA).
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

    if colapsou(y_llm):
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
        'margem_minima_perc': float(_rp.ULTIMA_MARGEM_MINIMA),   # 15/10/2026 (D)
        'y_metric_perc': y_metric_perc,
        'y_metric_nnls': y_metric_nnls,
        'fidelity_perc_vs_llm': accuracy_score(y_llm, y_metric_perc),
        'fidelity_nnls_vs_llm': accuracy_score(y_llm, y_metric_nnls),
        'accuracy_perc_vs_true': accuracy_score(y_true, y_metric_perc),
        'accuracy_nnls_vs_true': accuracy_score(y_true, y_metric_nnls),
        'llm_accuracy_vs_true': accuracy_score(y_true, y_llm),
        # Contagem absoluta de erros (conjuntos pequenos).
        'n_samples': len(y_true),
        'n_errors_perc_vs_true': int(np.sum(y_metric_perc != y_true)),
        'n_errors_nnls_vs_true': int(np.sum(y_metric_nnls != y_true)),
        'n_errors_llm_vs_true': int(np.sum(y_llm != y_true)),
        'n_malformed': n_malformed,
    }




def fg_exemplos_aprendiz(y_ref: np.ndarray, k: int, random_state: int,
                              log_prefix: str = "") -> np.ndarray:
    """k índices de exemplos few-shot, k/2 por classe sobre `y_ref` (rótulo de
    REFERÊNCIA do balanceamento: nos problemas com split F/G1/G2 é o rótulo REAL do
    treino, não o do especialista), sorteados sem reposição com
    `RandomState(random_state)` (I17; regra do rótulo real desde 09/10/2026).

    Classe 0 recebe ⌈k/2⌉ e classe 1 ⌊k/2⌋ (k par → exato). Se uma classe tiver menos
    pontos do que o pedido, completa com a outra e registra no log. O sorteio é
    determinístico em (y_ref, k, random_state): como o rótulo real não depende da
    variante de atributo nem do especialista, as variantes recebem os MESMOS índices
    na mesma ordem (os exemplos são então ROTULADOS pelo especialista de cada variante).
    """
    rng = np.random.RandomState(random_state)
    y_ref = np.asarray(y_ref)
    k = int(min(k, len(y_ref)))
    pedidos = {0: k - k // 2, 1: k // 2}
    escolhidos = {}
    for c in (0, 1):
        idx_c = np.where(y_ref == c)[0]
        n_c = min(pedidos[c], len(idx_c))
        escolhidos[c] = rng.choice(idx_c, size=n_c, replace=False) if n_c > 0 else np.array([], dtype=int)
    faltam = k - len(escolhidos[0]) - len(escolhidos[1])
    if faltam > 0:
        for c in (0, 1):
            idx_c = np.where(y_ref == c)[0]
            sobra = np.setdiff1d(idx_c, escolhidos[c])
            n_extra = min(faltam, len(sobra))
            if n_extra > 0:
                escolhidos[c] = np.concatenate([escolhidos[c], rng.choice(sobra, size=n_extra, replace=False)])
                faltam -= n_extra
        print(f"    ⚠ {log_prefix}k={k}: classe com menos de k/2 pontos no treino "
              f"(classe 0: {int(np.sum(y_ref == 0))}, classe 1: {int(np.sum(y_ref == 1))}); "
              f"sorteio completado com a outra classe ({len(escolhidos[0])}/{len(escolhidos[1])}).")
    return np.concatenate([escolhidos[0], escolhidos[1]]).astype(int)


def fg_executar(
    problem_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    nome_classe_0: str,
    nome_classe_1: str,
    feature_variants: List[Tuple[str, str]],
    seeds: List[int],
    ks_aprendiz: List[int],
    n_train_ratio: float = 0.7,
    verbose: bool = True,
    X_display: Optional[np.ndarray] = None,
    expert_labels: Optional[np.ndarray] = None,
    com_passos_3_a_5: bool = False,
    atributos_prompt_aprendiz: Tuple[int, ...] = (2,),
) -> dict:
    """Sequência completa de passos de um problema com split 70/30 (Problema F: Passos 1 a 4; Problema G: Passos 1 a 7).

    ``atributos_prompt_aprendiz`` (14/10/2026, Metodologia IV.7 do orientador: "alimenta-se o
    prompt com novos atributos, x1·x2 para R³ e x1², x2² para R⁴"): número de atributos
    MOSTRADOS ao LLM no aprendizado em contexto (Passo 6 do G; Passo 3 do F), fator
    controlado e independente da dimensão da métrica eleita, que define só o
    ESPECIALISTA. O G passa `ATRIBUTOS_PROMPT_PASSO6_G` = (2, 3, 4); o F usa (2,). Com 3
    o prompt leva x1·x2 (nos exemplos e no ponto de teste); com 4, x1² e x2². Os extras só
    acompanham exemplos: o zero-shot (k = 0) é UMA coleta, repetida nas linhas de cada
    dimensão (0 chamadas extras). Coluna `n_features` do CSV = atributos no prompt;
    `especialista_n_features` = dimensão R^n da métrica eleita (NaN no F). O Perceptron
    do Passo 7 (Passo 4 do F) é treinado nos mesmos exemplos e no mesmo espaço que o LLM viu.

    ``X_display`` (opcional, mesma forma de ``X``): valores MOSTRADOS ao LLM no
    prompt (pontos de treino do Passo 2, exemplos e pontos de teste do Passo 6, inclusive os
    atributos extras x1·x2, x1², x2² calculados sobre eles). A estimação da
    métrica, os centróides e todas as avaliações usam ``X``. Serve ao
    subproblema G2 (kg e m no prompt, métrica no espaço normalizado). Quando
    ``None``, o prompt mostra ``X``.

    ``expert_labels`` (opcional, alinhado a ``y_true``): rótulos do ESPECIALISTA do
    aprendizado em contexto (decisão D1, 09/10/2026). No Problema F passa-se ``y_true`` (o especialista
    é o rótulo real do make_moons): os exemplos few-shot, a referência das métricas
    (acurácia/κ/F1 LLM vs especialista) e o baseline Perceptron usam esses rótulos, o
    Passo 3 do F roda MESMO que o Passo 2 tenha colapsado (sem `best`: 2 atributos,
    sem extras no prompt) e `especialista_origem="rotulo_real"`. Sem ``expert_labels``
    (G1/G2) o especialista é a métrica eleita no Passo 5 (`especialista_origem=
    "metrica"`) e o Passo 6 depende de haver uma métrica.

    ``com_passos_3_a_5`` (Problema G, Metodologia IV.7 do orientador, 13/10/2026): os
    Passos 2 a 4 do G seguem os MESMOS passos do Problema A (`adg_passos_2_a_5`) nos
    pontos de treino: zero-shot (reaproveita os rótulos já coletados, 0 chamadas
    extras), few-shot com k ∈ K_EXEMPLOS_ROTULO_REAL exemplos rotulados pelo RÓTULO
    REAL (k/2 por classe, sorteio `seed + rep`, `reps_para` repetições; o LLM
    classifica os n_train − k restantes) e o teto (W ajustado ao rótulo real), cada
    regime estimado em R²/R³/R⁴ com Perceptron e NNLS. As linhas vão para
    'passos_2_a_5' (colunas de `bloco1_ad_passos_2_a_5` + `problem_name`,
    `subproblem`, `class_names`, `feature_names`; CSV `bloco3_g_passos_2_a_5`). O
    Problema F não usa os Passos 3 a 5 (False). Com eles, o ESPECIALISTA do Passo 6 é
    eleito entre TODAS as métricas vindas do LLM (zero-shot e few-shot, todos os k/rep e
    n_features, sem o teto e sem colapsos) pela maior acurácia vs rótulo real no treino
    (`accuracy_metric_vs_true_full`, Perceptron): o Passo 6 roda mesmo quando o zero-shot
    colapsa, desde que algum regime few-shot não tenha colapsado.

    Para cada (seed, feature_variant):
      1. Split treino/teste (n_train_ratio).
      2. Zero-shot no treino e métrica com n_features ∈ {2, 3, 4}, calculando fidelidade (vs LLM)
         e acurácia (vs rótulos reais, contexto); com `com_passos_3_a_5`, também
         few-shot (rótulo real) e teto, via `adg_passos_2_a_5`.
      3. Identifica a melhor configuração (maior acurácia vs real); com os Passos 3 a 5, a
         eleição considera zero-shot e few-shot (`especialista_regime`/`especialista_k`/`especialista_rep`).
      4. Aprendizado em contexto (F Passo 3 / G Passo 6): LLM aprende com k exemplos ROTULADOS pelo especialista. O sorteio
         (`fg_exemplos_aprendiz`, `RandomState(seed + rep)`) balanceia k/2 por
         classe pelo RÓTULO REAL do treino (`y_train`), não pelo rótulo do
         especialista (regra de 09/10/2026): os índices são idênticos entre as
         variantes de atributo (e a regra é a mesma em F e G), isolando o efeito do
         nome de atributo nesse passo. Em F (especialista = rótulo real) o balanço pelo
         especialista é exato; em G1/G2 (especialista = métrica estimada por
         variante) o balanço pelo rótulo do especialista pode não ser exato, e o log
         imprime a contagem por classe do especialista nos exemplos.

    Retorna dict com:
      - 'passo2': lista de resultados do zero-shot + métrica (todas configs)
      - 'passos_2_a_5': linhas dos Passos 2 a 5 do G (zero-shot, few-shot do rótulo real e teto; a
        eleição do Passo 5 lê estas linhas; vazia sem `com_passos_3_a_5`)
      - 'aprendiz': lista de resultados do aprendizado em contexto (colunas `accuracy/kappa/
        f1_llm_vs_expert`, `accuracy_llm_vs_true`, `especialista_origem`, `especialista_regime`,
        `especialista_k`, `especialista_rep`, `especialista_n_features`, `n_features` (atributos
        mostrados ao LLM, fator `atributos_prompt_aprendiz`), `example_indices`; em F,
        especialista = rótulo real, as colunas `especialista_*` ficam NaN)
      - 'llm_label_maps': dict {(problem_name, seed, feat_0, feat_1, kind)} só com a menor
        dimensão do prompt (2 atributos):
        {'X', 'y_llm', 'y_true', 'y_expert', ...}} — o problem_name na chave evita colisão
        entre problemas que usam os mesmos nomes de feature (ex.: meia-lua e
        peso×altura ambos com x1/x2)
                          para a visualização ponto-a-ponto

    (Antes de 13/10/2026: `run_external_problem_pipeline`.)
    """
    global CURRENT_TAG
    lista_passos_2_e_4 = []
    lista_aprendiz = []
    linhas_passos_2_a_5 = []
    llm_label_maps = {}
    bloco_tag = "bloco3" if problem_name.startswith("G") else "bloco2"
    subproblem = problem_name.split("_", 1)[0]          # G1_homem_mulher → G1
    class_names = f"{nome_classe_0}/{nome_classe_1}"
    especialista_origem = "rotulo_real" if expert_labels is not None else "metrica"
    passo_aprendiz = 6 if bloco_tag == "bloco3" else 3        # G Passo 6; F Passo 3
    regime_aprendiz = "exemplos_rotulo_real" if expert_labels is not None else "exemplos_da_metrica"
    if expert_labels is not None and len(expert_labels) != len(y_true):
        raise ValueError(f"expert_labels ({len(expert_labels)}) deve ter o tamanho de y_true ({len(y_true)})")

    if verbose:
        print_section(f"PROBLEMA {problem_name}: Passos 1 a {7 if com_passos_3_a_5 else 4}", "═")
        print(f"  Variantes de atributo: {feature_variants}")
        print(f"  Sementes: {seeds}")
        print(f"  N amostras total: {len(X)} | Split treino/teste: {n_train_ratio:.0%}/{1-n_train_ratio:.0%}")
        if com_passos_3_a_5:
            print(f"  Passos 2 a 5 do Problema G (mesmos passos do Problema A) nos pontos de treino: "
                  f"zero-shot + few-shot k={list(K_EXEMPLOS_ROTULO_REAL)} (exemplos do rótulo real, "
                  f"k/2 por classe, sorteio seed + rep, {N_REPETICOES} rep.) + teto, em R²/R³/R⁴.")

    for seed in seeds:
        # RNG LOCAL por seed (não o estado global) só para o split 70/30; o
        # sorteio de exemplos do aprendizado em contexto usa RandomState(seed + rep).
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        n_train = int(n_train_ratio * len(X))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        X_train, y_train = X[train_idx], y_true[train_idx]
        X_test, y_test = X[test_idx], y_true[test_idx]
        y_exp_train_gt = None if expert_labels is None else np.asarray(expert_labels)[train_idx]
        y_exp_test_gt = None if expert_labels is None else np.asarray(expert_labels)[test_idx]
        X_disp = X if X_display is None else np.asarray(X_display, dtype=float)
        if X_disp.shape != X.shape:
            raise ValueError(f"X_display {X_disp.shape} deve ter a mesma forma de X {X.shape}")
        X_disp_train, X_disp_test = X_disp[train_idx], X_disp[test_idx]

        if verbose:
            print(f"\n  ── Seed {seed} ── n_train={len(X_train)}, n_test={len(X_test)}")

        for feat_0, feat_1 in feature_variants:
            if verbose:
                print(f"\n    Variante de atributo: ({feat_0}, {feat_1})")

            # Coleta LLM uma única vez (n_features=2 visível para o LLM)
            t_collect = time.time()
            tag_extra = f"nomes={class_names}|atributos={feat_0}/{feat_1}"
            CURRENT_TAG = _tag(bloco_tag, problem_name, 2, "passo2_zero_shot", 0, 0, seed, extra=tag_extra)
            print(f"    [{problem_name} seed={seed} {feat_0}/{feat_1}] Passo 2: coletando LLM em zero-shot (n={len(X_train)})...", flush=True)
            y_llm_train, n_mal = collect_llm_decisions(
                X_disp_train, nome_classe_0, nome_classe_1,
                examples=None, verbose=False,
                nome_feature_0=feat_0, nome_feature_1=feat_1,
                label_prefix=f"[{problem_name} seed={seed} {feat_0}/{feat_1}] ",
            )
            print(f"    [{problem_name} seed={seed} {feat_0}/{feat_1}] Passo 2 coletado em {time.time()-t_collect:.1f}s (malformadas={n_mal})", flush=True)

            llm_label_maps[(problem_name, seed, feat_0, feat_1, 'train')] = {
                'X': X_train.copy(), 'X_display': X_disp_train.copy(),
                'y_llm': y_llm_train.copy(), 'y_true': y_train.copy(),
            }

            # Métrica com 2 / 3 / 4 atributos (reutiliza y_llm_train)
            results_by_nfeat = []
            for n_feat in [2, 3, 4]:
                r = passo4_estimar_metrica_r2_r3_r4(
                    X_train, y_train, n_feat,
                    nome_classe_0, nome_classe_1,
                    feat_0, feat_1,
                    y_llm_cached=y_llm_train,
                    verbose=False,
                )
                if r is None:
                    if verbose:
                        print(f"      n_features={n_feat}: colapso no zero-shot (classe única ou minoritária < {MIN_PONTOS_CLASSE_MINORITARIA}), pulando.")
                    continue
                r['seed'] = seed
                r['problem_name'] = problem_name
                results_by_nfeat.append(r)
                lista_passos_2_e_4.append(r)

                if verbose:
                    print(
                        f"      n_features={n_feat}: fid_perc={r['fidelity_perc_vs_llm']:.1%} | "
                        f"acc_perc_real={r['accuracy_perc_vs_true']:.1%} | "
                        f"fid_nnls={r['fidelity_nnls_vs_llm']:.1%} | "
                        f"acc_nnls_real={r['accuracy_nnls_vs_true']:.1%} | "
                        f"llm_acc_real={r['llm_accuracy_vs_true']:.1%}"
                    )

            # Problema G, Passos 3 e 4: few-shot com exemplos do rótulo real
            # + teto, em R²/R³/R⁴; o zero-shot reaproveita y_llm_train (0 chamadas extras).
            proto = None
            if com_passos_3_a_5:
                proto = adg_passos_2_a_5(
                    X_train, y_train, nome_classe_0, nome_classe_1,
                    problem=problem_name, seed=seed, provider=CURRENT_PROVIDER,
                    model_name=MODEL_NAME, temperature=CURRENT_TEMPERATURE,
                    n_features_list=ATRIBUTOS_PASSO4["G"], ks_rotulo_real=K_EXEMPLOS_ROTULO_REAL,
                    y_llm_zero_cached=y_llm_train, n_malformed_zero=n_mal,
                    X_display=X_disp_train, nome_feature_0=feat_0, nome_feature_1=feat_1,
                    bloco=bloco_tag, tag_extra=tag_extra, verbose=verbose,
                    label_prefix=f"[{problem_name} seed={seed} {feat_0}/{feat_1}] ",
                )
                for _row in proto['rows']:
                    _row.update(problem_name=problem_name, subproblem=subproblem,
                                class_names=class_names, feature_names=f"{feat_0}/{feat_1}")
                linhas_passos_2_a_5.extend(proto['rows'])
                if verbose:
                    print(f"    ✓ Passos 2 a 5 em {problem_name} ({feat_0}/{feat_1}, seed={seed}): "
                          f"{len(proto['rows'])} linhas, {proto['n_calls']} chamadas novas")

            # Seleciona a MELHOR configuração pela acurácia vs rótulo real (Perceptron).
            #   Sem os Passos 3 a 5 (F): entre as métricas zero-shot (R²/R³/R⁴).
            #   Com eles (G): entre TODAS as métricas vindas do LLM (zero-shot e com exemplos,
            #   todos os k/rep/n_features), sem o teto (passo5_teto) e sem colapsos; empate
            #   fica com a primeira na ordem zero-shot → k crescente → rep → R² → R³ → R⁴.
            #   Sem métrica (colapso com especialista = rótulo real) usa 2 atributos.
            best = g_passo5_eleger_especialista(proto, results_by_nfeat)
            if best is None and expert_labels is None:
                if verbose:
                    print("      ⚠ Nenhuma métrica disponível (colapso no zero-shot"
                          + (" em todos os regimes" if com_passos_3_a_5 else "")
                          + ") e o especialista é a métrica — pulando o Passo 6.")
                continue
            if not results_by_nfeat and verbose and expert_labels is not None:
                print("      ⚠ Passo 2 colapsou; o Passo 3 roda com o especialista = rótulo real "
                      "(2 atributos, sem extras).")

            # Δ de ganho 2→3 e 2→4 atributos
            if verbose:
                fid_by_nfeat = {r['n_features']: r['fidelity_perc_vs_llm'] for r in results_by_nfeat}
                acc_by_nfeat = {r['n_features']: r['accuracy_perc_vs_true'] for r in results_by_nfeat}
                if 2 in fid_by_nfeat:
                    base_fid = fid_by_nfeat[2]
                    base_acc = acc_by_nfeat[2]
                    print(f"    Ganhos vs 2 atributos — seed={seed} | {feat_0}/{feat_1}:")
                    for nf in (3, 4):
                        if nf in fid_by_nfeat:
                            d_fid = fid_by_nfeat[nf] - base_fid
                            d_acc = acc_by_nfeat[nf] - base_acc
                            marker = (
                                "✓ GANHO" if d_acc > 0.01
                                else ("≈ neutro" if abs(d_acc) <= 0.01 else "✗ perda")
                            )
                            print(
                                f"      2→{nf} atributos: Δ fidelidade={d_fid:+.1%} | "
                                f"Δ acurácia_real={d_acc:+.1%} ({marker})"
                            )

            # Atributos dos exemplos do aprendiz: no G (Passo 6) os da métrica eleita no Passo 5;
            # no F (Passo 3) sempre 2 (x1, x2), pois o especialista é o rótulo real e a métrica do
            # Passo 2 serve só para medir a fidelidade (13/10/2026; antes o F herdava R³/R⁴ da
            # melhor métrica zero-shot, o que variava por semente e não estava na lista de passos).
            n_feat_e = best['n_features'] if (best is not None and expert_labels is None) else 2
            if verbose and best is not None and expert_labels is None:
                print(
                    f"    ★ Passo 5: especialista eleito: regime={best['metrica_regime']} "
                    f"k={best['metrica_k']} rep={best['metrica_rep']} R{best['n_features']} "
                    f"(acurácia vs rótulo real no treino = {best['accuracy_perc_vs_true']:.1%}, Perceptron)"
                )

            # Aprendizado em contexto (F Passo 3 / G Passo 6): o ESPECIALISTA rotula treino (exemplos) e teste (referência).
            #   expert_labels dado (F): rótulo real; senão (G1/G2): métrica eleita no Passo 5, no
            #   seu espaço R^{n_feat_e}. O nº de atributos do PROMPT é outro fator
            #   (`atributos_prompt_aprendiz`, laço abaixo).
            X_test_aug = augment_features(X_test, n_feat_e)
            X_train_aug = augment_features(X_train, n_feat_e)
            if expert_labels is not None:
                y_expert_train, y_expert_test = y_exp_train_gt, y_exp_test_gt
            else:
                y_expert_train = predict_with_metric(X_train_aug, best['centroids'], best['w_perc'])
                y_expert_test = predict_with_metric(X_test_aug, best['centroids'], best['w_perc'])
            if verbose:
                print(f"    Especialista do aprendizado em contexto: {especialista_origem} "
                      f"(treino {int(np.sum(y_expert_train == 0))}/{int(np.sum(y_expert_train == 1))}, "
                      f"teste {int(np.sum(y_expert_test == 0))}/{int(np.sum(y_expert_test == 1))})")
                print("    Exemplos few-shot: sorteio RandomState(seed + rep) BALANCEADO PELO RÓTULO REAL "
                      "(k/2 por classe real); índices iguais entre variantes de atributo e mesma regra "
                      "em F e G; exemplos rotulados pelo especialista "
                      + ("(= rótulo real: balanço exato)." if especialista_origem != "metrica"
                         else "(= métrica desta variante: o balanço pelo rótulo do especialista pode "
                              "não ser exato; contagem por classe impressa a cada sorteio)."))

            # Passo 6 do G (14/10/2026, Metodologia IV.7): o número de atributos MOSTRADOS ao LLM
            # é um fator controlado, `atributos_prompt_aprendiz` (G: 2, 3 e 4; F: só 2),
            # independente da dimensão R^n da métrica eleita (`n_feat_e`), que define o
            # ESPECIALISTA. Com 3 atributos o prompt leva x1·x2; com 4, x1² e x2². Os extras só
            # acompanham EXEMPLOS: no zero-shot (k = 0) o LLM vê x1, x2 como no Passo 2, logo a
            # coleta de k = 0 é UMA só, reaproveitada nas linhas das outras dimensões.
            # 15/10/2026: três fases. (1) plano de todas as coletas (dimensão × k × rep), com os
            # sorteios e os prompts; (2) UM lote assíncrono com as coletas necessárias (o zero-shot,
            # sem extras, é coletado só na primeira dimensão e reaproveitado); (3) medidas e linhas
            # na mesma ordem de antes. Antes, as 37 coletas de 30 pontos de cada variante (1 zero-shot
            # + 3 dimensões × 4 k × 3 reps) saíam uma atrás da outra, cada uma presa à latência de
            # uma ida e volta à API.
            plano = []
            for n_feat_prompt in atributos_prompt_aprendiz:
                if n_feat_prompt == 2:
                    extra_features_names_test = None
                elif n_feat_prompt == 3:
                    extra_features_names_test = [f"{feat_0}*{feat_1}"]
                elif n_feat_prompt == 4:
                    extra_features_names_test = [f"{feat_0}²", f"{feat_1}²"]
                else:
                    raise ValueError(f"atributos_prompt_aprendiz aceita só 2, 3 ou 4 (recebido {n_feat_prompt})")

                # Valores mostrados no prompt (iguais a X quando não há X_display), com os extras
                X_disp_train_aug = augment_features(X_disp_train, n_feat_prompt)
                X_disp_test_aug = augment_features(X_disp_test, n_feat_prompt)
                # Espaço do Perceptron do Passo 7 (Passo 4 do F): os MESMOS atributos que o LLM viu
                X_train_aug_p = augment_features(X_train, n_feat_prompt)
                X_test_aug_p = augment_features(X_test, n_feat_prompt)
                tag_extra_dim = (f"{tag_extra}|n_atributos_prompt={n_feat_prompt}" if tag_extra
                                 else f"n_atributos_prompt={n_feat_prompt}")
                if verbose:
                    print(f"    [{problem_name} Passo {passo_aprendiz} seed={seed} {feat_0}/{feat_1}] "
                          f"atributos no prompt = {n_feat_prompt}"
                          + (f" ({', '.join(extra_features_names_test)} além de {feat_0}, {feat_1})"
                             if extra_features_names_test else f" ({feat_0}, {feat_1})")
                          + f" | especialista em R{n_feat_e}")

                for k_exemplos in ks_aprendiz:
                    # Repetições só onde há sorteio de exemplos (regra única reps_para):
                    # zero-shot tem prompt fixo → 1 coleta.
                    n_reps_eff = reps_para(k_exemplos)
                    for rep in range(n_reps_eff):
                        if k_exemplos == 0:
                            examples_for_prompt = None
                            ex_indices = None
                        else:
                            # Sorteio BALANCEADO pelo RÓTULO REAL (y_train): k/2 por classe
                            # real, sem reposição, RandomState(seed + rep) (I17; regra de
                            # 09/10/2026). Como y_train não depende da variante nem do
                            # especialista, a mesma repetição recebe os MESMOS índices em
                            # todas as variantes de atributo e dimensões do prompt (F e G,
                            # mesma regra); os exemplos são ROTULADOS pelo especialista
                            # (rótulo real em F, métrica em G1/G2, onde o balanço por
                            # especialista pode não ser exato: contagem impressa abaixo).
                            ex_indices = fg_exemplos_aprendiz(
                                y_train, k_exemplos, seed + rep,
                                log_prefix=f"[{problem_name} Passo {passo_aprendiz} seed={seed} {feat_0}/{feat_1}] ")
                            if verbose and n_feat_prompt == atributos_prompt_aprendiz[0]:
                                n_ex_real = (int(np.sum(y_train[ex_indices] == 0)), int(np.sum(y_train[ex_indices] == 1)))
                                n_ex_exp = (int(np.sum(y_expert_train[ex_indices] == 0)), int(np.sum(y_expert_train[ex_indices] == 1)))
                                print(f"    [{problem_name} Passo {passo_aprendiz} seed={seed} {feat_0}/{feat_1}] k={k_exemplos} rep={rep}: "
                                      f"exemplos por classe real {n_ex_real[0]}/{n_ex_real[1]}; "
                                      f"por classe do especialista ({especialista_origem}) {n_ex_exp[0]}/{n_ex_exp[1]}"
                                      + ("" if n_ex_real == n_ex_exp else " (balanço pelo especialista não exato)"))
                            examples_for_prompt = []
                            for ei in ex_indices:
                                row = [X_disp_train[ei, 0], X_disp_train[ei, 1]]
                                if n_feat_prompt >= 3:
                                    row.extend(X_disp_train_aug[ei, 2:].tolist())
                                row.append(nome_classe_0 if y_expert_train[ei] == 0 else nome_classe_1)
                                examples_for_prompt.append(tuple(row))

                        # Atributos extras (R³/R⁴) só acompanham EXEMPLOS: no zero-shot o
                        # LLM vê x1, x2 como no Passo 2, para a curva ser comparável.
                        mostra_extras = k_exemplos > 0 and n_feat_prompt >= 3
                        extra_matrix_test = X_disp_test_aug[:, 2:] if mostra_extras else None
                        # F Passo 3 / G Passo 6: sem exemplos = `zero_shot`; com exemplos, a origem deles.
                        tag_coleta = _tag(bloco_tag, problem_name, passo_aprendiz,
                                          "zero_shot" if k_exemplos == 0 else regime_aprendiz,
                                          k_exemplos, rep, seed, extra=tag_extra_dim)
                        reaproveita = (k_exemplos == 0 and n_feat_prompt != atributos_prompt_aprendiz[0])
                        job = None if reaproveita else dict(
                            X=X_disp_test, nome_classe_0=nome_classe_0, nome_classe_1=nome_classe_1,
                            examples=examples_for_prompt, verbose=False,
                            nome_feature_0=feat_0, nome_feature_1=feat_1,
                            extra_features_matrix=extra_matrix_test,
                            extra_feature_names=extra_features_names_test if mostra_extras else None,
                            tag=tag_coleta,
                            label_prefix=f"[{problem_name} Passo {passo_aprendiz} seed={seed} {feat_0}/{feat_1} k={k_exemplos} r{rep+1} R{n_feat_prompt}] ",
                        )
                        plano.append(dict(n_feat_prompt=n_feat_prompt, k=k_exemplos, rep=rep, n_reps_eff=n_reps_eff,
                                          ex_indices=ex_indices, examples=examples_for_prompt, job=job,
                                          X_train_aug_p=X_train_aug_p, X_test_aug_p=X_test_aug_p))

            # Fase 2: coleta em lote (0 chamadas para o zero-shot reaproveitado)
            CURRENT_TAG = ""
            rotulo_lote = f"[{problem_name} Passo {passo_aprendiz} seed={seed} {feat_0}/{feat_1}] "
            print(f"    {rotulo_lote}coletando {sum(1 for e in plano if e['job'] is not None)} conjuntos "
                  f"(n_test={len(X_test)} cada; especialista={especialista_origem} R{n_feat_e})...", flush=True)
            coletas = iter(collect_llm_decisions_lote([e["job"] for e in plano if e["job"] is not None], rotulo=rotulo_lote))
            zero_shot_cache = None
            for e in plano:
                if e["job"] is None:
                    e["y_llm"], e["n_mal"] = zero_shot_cache
                    print(f"    {rotulo_lote}k=0 | atributos no prompt={e['n_feat_prompt']} | zero-shot reaproveitado "
                          f"da dimensão {atributos_prompt_aprendiz[0]} (0 chamadas)", flush=True)
                else:
                    e["y_llm"], e["n_mal"] = next(coletas)
                    if e["k"] == 0:
                        zero_shot_cache = (np.asarray(e["y_llm"]).copy(), e["n_mal"])

            # Fase 3: medidas e linhas, na mesma ordem (dimensão, k, rep)
            for e in plano:
                n_feat_prompt, k_exemplos, rep, n_reps_eff = e["n_feat_prompt"], e["k"], e["rep"], e["n_reps_eff"]
                ex_indices, examples_for_prompt = e["ex_indices"], e["examples"]
                X_train_aug_p, X_test_aug_p = e["X_train_aug_p"], e["X_test_aug_p"]
                y_llm_test, n_mal_e = np.asarray(e["y_llm"]), e["n_mal"]

                # Guarda o mapa ponto-a-ponto só da 1ª repetição e da menor dimensão do
                # prompt (visualização final_graphic_07). 'y_expert' = rótulos do
                # especialista no teste ('y_metric' mantido como sinônimo para os leitores).
                if rep == 0 and n_feat_prompt == atributos_prompt_aprendiz[0]:
                    llm_label_maps[(problem_name, seed, feat_0, feat_1, k_exemplos)] = {
                        'X': X_test.copy(), 'X_display': X_disp_test.copy(),
                        'y_llm': y_llm_test.copy(),
                        'y_true': y_test.copy(), 'y_expert': y_expert_test.copy(),
                        'y_metric': y_expert_test.copy(), 'especialista_origem': especialista_origem,
                    }

                # Acurácia (LLM vs especialista), Cohen's κ e F1-Macro (M2/M13).
                cm_e = compute_consistency_metrics(y_llm_test, y_expert_test)
                acc_llm_vs_expert = float(cm_e.accuracy)
                acc_llm_vs_true = accuracy_score(y_test, y_llm_test)
                kappa = float(cm_e.cohen_kappa)
                f1 = float(cm_e.f1_score)
                # Nº absoluto de amostras erradas (peso×altura tem 30 pontos de teste).
                n_test_e = len(y_test)
                n_err_llm_vs_true = int(np.sum(y_llm_test != y_test))
                n_err_llm_vs_expert = int(np.sum(y_llm_test != y_expert_test))
                # Comparação com Perceptron diagonal treinado sobre os MESMOS exemplos
                # few-shot (rótulos do especialista), no MESMO espaço R^n que o LLM viu.
                acc_perc_baseline = None
                if examples_for_prompt is not None and k_exemplos >= 4:
                    try:
                        # Reconstroi (X_examples, y_examples) a partir dos índices few-shot
                        x_ex_aug = X_train_aug_p[ex_indices]
                        y_ex = y_expert_train[ex_indices]
                        if len(np.unique(y_ex)) >= 2:
                            c_ex = compute_centroids(x_ex_aug, y_ex)
                            w_baseline, _ = train_relaxed_perceptron(
                                x_ex_aug, y_ex, c_ex,
                                **PERCEPTRON_PARAMS,
                                verbose=False,
                                use_best_effort=True,
                            )
                            y_baseline = predict_with_metric(X_test_aug_p, c_ex, w_baseline)
                            acc_perc_baseline = accuracy_score(y_test, y_baseline)
                    except Exception as exc:
                        # Caminho de DADOS (não cosmético): a acurácia do baseline-perceptron
                        # entra no CSV externo. Falha silenciosa viraria None sem rastro —
                        # registra o motivo para auditabilidade.
                        acc_perc_baseline = None
                        print("    ⚠ Perceptron nos mesmos exemplos (F Passo 4 / G Passo 7) falhou "
                              f"[{problem_name}, k={k_exemplos}, R{n_feat_prompt}]: {exc}")

                lista_aprendiz.append({
                    'problem_name': problem_name,
                    'seed': seed,
                    'rep': rep,
                    'feature_names': (feat_0, feat_1),
                    # Atributos MOSTRADOS ao LLM (fator `atributos_prompt_aprendiz`; 14/10/2026).
                    # Até 13/10/2026 esta coluna era a dimensão da métrica eleita.
                    'n_features': n_feat_prompt,
                    'k': k_exemplos,
                    'especialista_origem': especialista_origem,
                    # Origem da métrica eleita (G, Passo 5): regime/k/rep/dimensão dos Passos 2 a 4;
                    # sem os Passos 3 a 5 é o zero-shot (k=0, rep=0); em F (rótulo real) NaN.
                    'especialista_regime': (best['metrica_regime'] if expert_labels is None else float('nan')),
                    'especialista_k': (best['metrica_k'] if expert_labels is None else float('nan')),
                    'especialista_rep': (best['metrica_rep'] if expert_labels is None else float('nan')),
                    'especialista_n_features': (n_feat_e if expert_labels is None else float('nan')),
                    'accuracy_llm_vs_expert': acc_llm_vs_expert,
                    'accuracy_llm_vs_true': acc_llm_vs_true,
                    'kappa_llm_vs_expert': kappa,
                    'f1_llm_vs_expert': f1,
                    'accuracy_perceptron_baseline_vs_true': acc_perc_baseline,
                    'n_test': n_test_e,
                    'n_errors_llm_vs_true': n_err_llm_vs_true,
                    'n_errors_llm_vs_expert': n_err_llm_vs_expert,
                    'n_malformed': n_mal_e,
                    'example_indices': ("" if ex_indices is None
                                        else ";".join(str(int(i)) for i in ex_indices)),
                })

                if verbose:
                    baseline_str = f" | perceptron (vs rótulo real)={acc_perc_baseline:.1%}" if acc_perc_baseline is not None else ""
                    print(
                        f"      Passo {passo_aprendiz} k={k_exemplos} rep={rep+1}/{n_reps_eff} R{n_feat_prompt}: acurácia (LLM vs especialista)={acc_llm_vs_expert:.1%} | "
                        f"Cohen's κ={kappa:.3f} | F1-Macro={f1:.3f} | acurácia (vs rótulo real)={acc_llm_vs_true:.1%} "
                        f"({n_err_llm_vs_true}/{n_test_e} discordantes){baseline_str}"
                    )

    CURRENT_TAG = ""
    return {
        'passo2': lista_passos_2_e_4,
        'passos_2_a_5': linhas_passos_2_a_5,
        'aprendiz': lista_aprendiz,
        'llm_label_maps': llm_label_maps,
    }
# As figuras dos Problemas F e G (bloco23_*) vivem em plots.py.


def g_passo5_eleger_especialista(proto: Optional[dict], results_by_nfeat: List[dict]) -> Optional[dict]:
    """Problema G, Passo 5: eleição do especialista (G1/G2), pela regra do Problema A.

    Com `proto` (saída de `adg_passos_2_a_5`): entre as linhas do Perceptron dos regimes
    `passo2_zero_shot` e `passo3_exemplos_rotulo_real` (todos os k, rep e n_features), excluídos o teto
    (`passo5_teto`) e os colapsos, a de maior `accuracy_metric_vs_true_full` (acurácia da
    métrica vs rótulo real nos n_train pontos). Empate: a primeira na ordem das linhas
    (zero-shot, depois k crescente, rep, R², R³, R⁴). Sem `proto`: a métrica zero-shot de
    maior `accuracy_perc_vs_true` em `results_by_nfeat` (regra anterior; F, sem os Passos 3 a 5).

    Devolve dict com `n_features`, `w_perc`, `centroids`, `accuracy_perc_vs_true`,
    `metrica_regime`, `metrica_k`, `metrica_rep`; None se não há métrica.

    (Antes de 13/10/2026: `_eleger_especialista_externo`.)
    """
    if proto is None:
        if not results_by_nfeat:
            return None
        r = max(results_by_nfeat, key=lambda r: r['accuracy_perc_vs_true'])
        return {'n_features': int(r['n_features']), 'w_perc': np.asarray(r['w_perc'], float),
                'centroids': np.asarray(r['centroids'], float),
                'accuracy_perc_vs_true': float(r['accuracy_perc_vs_true']),
                'metrica_regime': 'passo2_zero_shot', 'metrica_k': 0, 'metrica_rep': 0}
    cands = [r for r in proto['rows']
             if r['algorithm'] == 'perceptron' and r['regime'] in ('passo2_zero_shot', 'passo3_exemplos_rotulo_real')
             and int(r.get('collapsed', 0)) == 0 and not np.isnan(r['accuracy_metric_vs_true_full'])]
    if not cands:
        return None
    r = max(cands, key=lambda r: r['accuracy_metric_vs_true_full'])
    m = proto['metrics'][(r['regime'], r['k'], r['rep'], r['n_features'])]
    return {'n_features': int(r['n_features']), 'w_perc': np.asarray(m['w_perc'], float),
            'centroids': np.asarray(m['centroids'], float),
            'accuracy_perc_vs_true': float(r['accuracy_metric_vs_true_full']),
            'metrica_regime': r['regime'], 'metrica_k': int(r['k']), 'metrica_rep': int(r['rep'])}


def _instalar_reestimacao_offline(pasta: str, coletar_lacunas: bool = False) -> None:
    """Troca `collect_llm_decisions` por uma leitura dos rótulos gravados em ``pasta`` (15/10/2026).

    Chave EXATA: (modelo, tag, texto do prompt), depois (modelo, prompt) → registro de `llm_interactions*.json`
    (a tag primeiro, para o mesmo estímulo repetido em coletas diferentes manter a resposta daquela coleta,
    inclusive o flip de T=0). O prompt é
    reconstruído por `_montar_prompt` com os MESMOS argumentos da coleta, logo um registro só é
    reaproveitado quando o LLM veria exatamente o mesmo estímulo (mesmos exemplos, mesmos rótulos
    dos exemplos, mesmo ponto). Onde o prompt mudou (Passo 4 de B/C e vieses k > 0: exemplos
    escolhidos e rotulados pela métrica do A; Passo 6 do G: exemplos rotulados pela métrica eleita;
    ambas podem mudar quando o estimador muda), há duas saídas: com ``coletar_lacunas`` o ponto é
    coletado na API (só esses pontos); sem ela usa-se o registro do MESMO ponto e MESMA tag da
    execução de origem (o LLM respondeu àquele ponto com exemplos ligeiramente diferentes) e a
    coleta é contada como APROXIMADA. O resumo das lacunas sai no fim do log e em
    `reestimacao_lacunas.csv`; `reestimacao_origem.txt` guarda a pasta de origem. Ponto sem
    registro nem fallback é erro. O lote (`collect_llm_decisions_lote`) detecta a troca e chama
    coleta a coleta. Os registros reaproveitados são copiados para LLM_INTERACTIONS (a pasta nova
    fica com as interações completas e a auditoria funciona).
    """
    import glob as _glob
    if not os.path.isdir(pasta):
        pasta = str(RESULTADOS_DIR / pasta)
    arquivos = sorted(_glob.glob(os.path.join(pasta, "llm_interactions_parte*.json"))) or \
               sorted(_glob.glob(os.path.join(pasta, "llm_interactions.json")))
    if not arquivos:
        raise SystemExit(f"--reestimar-de: nenhum llm_interactions*.json em {pasta}")
    por_prompt_tag: dict = {}    # (modelo, tag, prompt): o MESMO estímulo na MESMA coleta (preserva o flip de T=0 entre coletas)
    por_prompt: dict = {}
    por_tag_ponto: dict = {}
    por_ponto_k0: dict = {}      # (modelo, configuração sem passo/regime/k/rep, ponto) → registro com k = 0

    def _config(tag: str) -> str:
        """Campos da tag que identificam a configuração (problema, seed, nomes, atributos...), sem passo/regime/k/rep."""
        return "|".join(c for c in tag.split("|") if not c.startswith(("passo=", "regime=", "k=", "rep=")))

    for f in arquivos:
        with open(f, encoding="utf-8") as fh:
            for r in json.load(fh):
                pt = r.get("point") or {}
                ponto = (round(float(pt.get("x1")), 6), round(float(pt.get("x2")), 6))
                tg = r.get("tag", "")
                por_prompt_tag.setdefault((r.get("model"), tg, r.get("prompt")), r)
                por_prompt.setdefault((r.get("model"), r.get("prompt")), r)
                por_tag_ponto.setdefault((r.get("model"), tg, ponto), r)
                if "|k=0|" in tg:
                    por_ponto_k0.setdefault((r.get("model"), _config(tg), ponto), r)
    print(f"  Reestimação offline: {len(por_prompt)} registros indexados de {len(arquivos)} arquivo(s) em {pasta}"
          + (" | lacunas serão coletadas na API" if coletar_lacunas else " | lacunas usam o mesmo ponto/tag da origem (aproximação)"))
    lacunas: List[dict] = []
    original = _COLLECT_LLM_DECISIONS_ORIGINAL

    def _coleta_reproduzida(X, nome_classe_0, nome_classe_1, examples=None, verbose=True, label_prefix="",
                            nome_feature_0="x1", nome_feature_1="x2", extra_features_matrix=None,
                            extra_feature_names=None, prompt_variant="default", tag=None):
        t = tag if tag is not None else CURRENT_TAG
        X = np.asarray(X, dtype=float)
        rotulos = np.zeros(len(X), dtype=int); n_mal = 0
        faltam: List[int] = []; n_aprox = 0; n_aprox_k0 = 0
        for i, (x1, x2) in enumerate(X[:, :2]):
            extra = None
            if extra_features_matrix is not None and extra_feature_names is not None:
                extra = list(zip(extra_feature_names, extra_features_matrix[i]))
            prompt = _montar_prompt(float(x1), float(x2), nome_classe_0, nome_classe_1, examples,
                                    nome_feature_0, nome_feature_1, extra, prompt_variant)
            r = por_prompt_tag.get((MODEL_NAME, t, prompt)) or por_prompt.get((MODEL_NAME, prompt))
            if r is None and not coletar_lacunas:
                ponto = (round(float(x1), 6), round(float(x2), 6))
                r = por_tag_ponto.get((MODEL_NAME, t, ponto))
                if r is not None:
                    n_aprox += 1
                else:
                    # o ponto era EXEMPLO na origem (não foi perguntado): usa a resposta zero-shot (k = 0)
                    # do mesmo ponto na mesma configuração
                    r = por_ponto_k0.get((MODEL_NAME, _config(t), ponto))
                    if r is not None:
                        n_aprox_k0 += 1
            if r is None:
                faltam.append(i)
                continue
            rotulos[i] = 1 if str(r.get("parsed_label")) == str(nome_classe_1) else 0
            n_mal += int(bool(r.get("malformed")))
            LLM_INTERACTIONS.append(dict(r, tag=t))
        if faltam and not coletar_lacunas:
            i = faltam[0]
            raise RuntimeError(f"reestimação offline: sem registro para modelo={MODEL_NAME} tag={t} "
                               f"ponto=({X[i, 0]:.6f}, {X[i, 1]:.6f}) (nem por prompt, nem por ponto/tag, nem no zero-shot)")
        n_coletados = 0
        if faltam:
            sub = original(X[faltam], nome_classe_0, nome_classe_1, examples=examples, verbose=verbose,
                           label_prefix=label_prefix + "[lacuna] ", nome_feature_0=nome_feature_0,
                           nome_feature_1=nome_feature_1,
                           extra_features_matrix=(None if extra_features_matrix is None else np.asarray(extra_features_matrix)[faltam]),
                           extra_feature_names=extra_feature_names, prompt_variant=prompt_variant, tag=t)
            y_sub, mal_sub = sub
            rotulos[faltam] = y_sub; n_mal += int(mal_sub); n_coletados = len(faltam)
        n_aprox_total = n_aprox + n_aprox_k0
        if n_aprox_total or n_coletados:
            lacunas.append({"tag": t, "n_pontos": int(len(X)), "n_exatos": int(len(X) - n_aprox_total - n_coletados),
                            "n_aproximados": int(n_aprox), "n_aproximados_zero_shot": int(n_aprox_k0),
                            "n_coletados_api": int(n_coletados)})
        if verbose:
            det = f"{len(X) - n_aprox_total - n_coletados} exatos"
            if n_aprox: det += f", {n_aprox} APROXIMADOS (mesmo ponto/tag, exemplos diferentes)"
            if n_aprox_k0: det += f", {n_aprox_k0} APROXIMADOS pelo zero-shot (eram exemplos na origem)"
            if n_coletados: det += f", {n_coletados} coletados na API"
            print(f"  {label_prefix}[reestimação offline] {len(X)} rótulos: {det} (malformadas={n_mal})")
        return rotulos, n_mal

    def _relatorio(pasta_saida: str) -> None:
        with open(os.path.join(pasta_saida, "reestimacao_origem.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"Rótulos do LLM reaproveitados de: {pasta}\n"
                     f"Modo: {'lacunas coletadas na API' if coletar_lacunas else 'lacunas aproximadas pelo mesmo ponto/tag'}\n")
        print(f"\n{'='*70}\n REESTIMAÇÃO OFFLINE: origem dos rótulos = {pasta}")
        if not lacunas:
            print(" Todas as coletas usaram prompts IDÊNTICOS aos da origem (0 lacunas).")
            return
        df = pd.DataFrame(lacunas)
        df.to_csv(os.path.join(pasta_saida, "reestimacao_lacunas.csv"), index=False)
        n_ap, n_k0, n_api = int(df["n_aproximados"].sum()), int(df["n_aproximados_zero_shot"].sum()), int(df["n_coletados_api"].sum())
        print(f" Coletas com prompt diferente da origem: {len(df)} (pontos aproximados pelo mesmo ponto/tag: {n_ap}; "
              f"pelo zero-shot: {n_k0}; coletados na API: {n_api}).")
        print(" Motivo: os exemplos desses prompts são escolhidos e/ou rotulados pela métrica estimada, que mudou com o estimador.")
        if n_ap:
            print(" ATENÇÃO: nas coletas APROXIMADAS o rótulo é a resposta da origem ao MESMO ponto, com exemplos ligeiramente")
            print(" diferentes no prompt; os números do Passo 4 de B/C, dos vieses com k > 0 e do Passo 6 do G são aproximações.")
        def _grupo(tg):
            c = dict(kv.split("=", 1) for kv in tg.split("|")[1:] if "=" in kv)
            return f"{c.get('problema','?')} | passo={c.get('passo','?')} | {c.get('regime','?')} | k={c.get('k','?')}"
        df["grupo"] = df["tag"].map(_grupo)
        g = df.groupby("grupo")[["n_pontos", "n_exatos", "n_aproximados", "n_aproximados_zero_shot", "n_coletados_api"]].sum()
        print(g.to_string())

    globals()["collect_llm_decisions"] = _coleta_reproduzida
    globals()["_REESTIMACAO_DE"] = pasta
    globals()["_REESTIMACAO_RELATORIO"] = _relatorio


_REESTIMACAO_RELATORIO = None
_REESTIMACAO_DE = None


def main():
    global MODEL_NAME, CURRENT_PROVIDER, CURRENT_TEMPERATURE
    # As grades do --rapido são aplicadas ao namespace do módulo por aplicar_modo_rapido()
    # (main() só as LÊ); K_EXEMPLOS_METRICA_BC continua global por causa de --bc-zero-shot.
    global MODELS_TO_TEST, K_EXEMPLOS_METRICA_BC, CURRENT_TAG

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
    parser.add_argument(
        "--apenas", dest="apenas", nargs="+", choices=list(BLOCOS_APENAS),
        metavar="BLOCO", default=None,
        help="Roda SÓ os blocos indicados (bloco1 | bloco2 | bloco3 | oraculo; aceita "
             "mais de um). Força as flags RUN_* de topo dos blocos escolhidos e desliga "
             "as dos demais; a pasta de saída ganha o sufixo _apenas-<blocos> em vez de "
             "_completa. Ex.: python src/dissertacao_mestrado.py --apenas bloco3",
    )
    parser.add_argument(
        "--bc-zero-shot", dest="bc_zero_shot", action="store_true",
        help="Problemas B e C só em zero-shot (K_EXEMPLOS_METRICA_BC=[0], sem o Passo 4) nesta execução — pedido do "
             "orientador (01/10/2026): few-shot em B/C pode contaminar a transferência. "
             "O Passo 3 dos Problemas A e D continua com k em K_EXEMPLOS_ROTULO_REAL.",
    )
    parser.add_argument(
        "--reestimar-de", dest="reestimar_de", default=None, metavar="PASTA",
        help="Reestimação OFFLINE (15/10/2026): reaproveita TODOS os rótulos do LLM gravados em "
             "<PASTA>/llm_interactions*.json (chave: modelo, tag, ponto) em vez de chamar a API; 0 chamadas. "
             "Serve para refazer estimação, tabelas e figuras depois de uma mudança nos estimadores. "
             "A pasta nova é uma _completa (mesmos rótulos, nova estimação) e o log diz de onde vieram os rótulos.")
    parser.add_argument(
        "--coletar-lacunas", dest="coletar_lacunas", action="store_true",
        help="Com --reestimar-de: os pontos cujo prompt mudou (exemplos escolhidos/rotulados pela métrica) são "
             "coletados na API em vez de aproximados pelo registro do mesmo ponto/tag da origem.")
    parser.add_argument(
        "--ignorar-saldo", dest="ignorar_saldo", action="store_true",
        help="Não aborta quando o saldo do OpenRouter é menor que a estimativa de custo "
             "da execução (a verificação roda antes da primeira chamada paga).",
    )
    args = parser.parse_args()
    if args.reestimar_de:
        _instalar_reestimacao_offline(args.reestimar_de, coletar_lacunas=args.coletar_lacunas)
        args.ignorar_saldo = not args.coletar_lacunas

    apenas = args.apenas
    if apenas:
        # Sobrescreve as flags de topo no namespace do módulo: main() só as LÊ,
        # então globals() evita declarar cada uma como `global` aqui.
        for _flag, _valor in flags_para_apenas(apenas).items():
            globals()[_flag] = _valor
        print(f"  --apenas {' '.join(apenas)}: rodando somente "
              f"{', '.join(b for b in BLOCOS_APENAS if b in apenas)}")

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
        # Regra do smoke test (GRADES_MODO_RAPIDO): todos os experimentos, uma vez cada.
        aplicar_modo_rapido()
    if args.bc_zero_shot:
        K_EXEMPLOS_METRICA_BC = [0]
        print("  --bc-zero-shot: Problemas B e C só em zero-shot (Passos 2 e 3; sem o Passo 4)")

    # Guard: valida as chaves de API de TODOS os modelos selecionados ANTES de
    # iniciar qualquer coleta — falha na hora zero, não no meio da execução paga.
    _estimativa_usd = estimar_custo_usd(apenas, modo_rapido)
    _chaves_faltando = []
    for _prov, _mod, _tmp in (MODELS_TO_TEST if _estimativa_usd > 0 else []):
        _env_var = PROVIDER_CONFIG[_prov]["api_key_env"]
        if not os.getenv(_env_var):
            _chaves_faltando.append(f"{_prov}/{_mod} → defina {_env_var}")
    if _chaves_faltando:
        raise SystemExit(
            "ERRO: chave(s) de API ausente(s) no ambiente/.env:\n  - "
            + "\n  - ".join(_chaves_faltando)
            + "\nDefina a(s) chave(s) no .env ou restrinja com --modelo <substring>."
        )

    # Guard de saldo: consulta GET /credits e compara com a estimativa de custo
    # ANTES de criar a pasta — um 402 no meio da coleta abortaria a execução.
    _linha_saldo = (verificar_saldo(consultar_saldo_openrouter(), _estimativa_usd, args.ignorar_saldo)
                    if _estimativa_usd > 0 else "  Execução sem chamadas pagas (0 chamadas): saldo não verificado.")
    print(_linha_saldo)

    # ─────────────────────────────────────────────────────────────────────
    # CRIAÇÃO DA PASTA DE EXECUÇÃO E INÍCIO DO LOG
    # ─────────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _inicio_execucao = datetime.now()   # para a duração no fim do log (14/10/2026)
    iniciar_cronometro()                 # ⏱ nos cabeçalhos de etapa (15/10/2026)
    # Sufixo no nome da pasta distingue smoke (--rapido) de execução completa:
    # ferramentas/skills que precisam de números finais filtram por "_completa".
    # Com --apenas a execução é PARCIAL: nunca leva "_completa" (leva
    # "_apenas-<blocos>", e "_smoke" no fim quando combinada com --rapido).
    pasta_execucao = str(RESULTADOS_DIR / nome_pasta_execucao(timestamp, apenas, modo_rapido, reestimada=bool(_REESTIMACAO_DE)))
    os.makedirs(pasta_execucao, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" EXPERIMENTO INICIADO: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    if _REESTIMACAO_DE:
        print(f" REESTIMAÇÃO OFFLINE: rótulos do LLM reaproveitados de {_REESTIMACAO_DE} (0 chamadas à API; "
              f"nova estimação das métricas, tabelas e figuras)")
    print(f" Pasta de saída criada: {pasta_execucao}/")
    print(" Todos os arquivos (imagens, CSVs e log) serão salvos nesta pasta.")
    print(f"{'='*70}\n")

    # stdout E stderr no MESMO buffer (ordem de chegada): warnings dos
    # estimadores (warnings.warn -> stderr) e tracebacks precisam constar no
    # log_execucao.txt persistido — sem isso o log diria "zero avisos" falsamente.
    tee = Tee()
    tee_err = Tee(sys.stderr, tee._buffer)
    sys.stdout = tee
    sys.stderr = tee_err
    _ABORT_CONTEXT.update(pasta=pasta_execucao, tee=tee, tee_err=tee_err)
    print(_linha_saldo)   # repete no log persistido

    print_section("EXPERIMENTO: CONSISTÊNCIA DECISIONAL DE LLMs VIA OTIMIZAÇÃO INVERSA", "=")
    if modo_rapido:
        print_section(
            "⚡ MODO RÁPIDO ATIVO (--rapido) — smoke test de cobertura completa\n"
            "   TODOS os experimentos, 1x cada | 1 seed (42) | 1 repetição | few-shot [0, 4]\n"
            "   (NÃO usar para resultados finais)",
            "="
        )
    if apenas:
        print_section(
            f"▶ EXECUÇÃO PARCIAL (--apenas {' '.join(apenas)})\n"
            f"   Somente: {', '.join(b for b in BLOCOS_APENAS if b in apenas)} | "
            "pasta sem sufixo _completa (não substitui a execução cheia)",
            "="
        )
    print("  Organizado em 3 BLOCOS auto-contidos:")
    print("    BLOCO 1 — LLM como ESPECIALISTA (otim. inversa em A/B/C lineares + D meia-lua)")
    print("    BLOCO 2 — LLM como APRENDIZ (Problema E, especialista linear, + Problema F, meia-lua)")
    print("    BLOCO 3 — Estudo de caso REAL (peso × altura, elipse)")
    print(f"  Pasta de execução: {pasta_execucao}/")
    print(f"  Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────────────────────────────
    # FLAGS DE EXECUÇÃO ATIVAS
    # ─────────────────────────────────────────────────────────────────────
    # Sub-flags só aparecem quando o bloco-pai que as gateia está ativo (com
    # --apenas, os blocos desligados não rodam nada, mesmo com sub-flag True).
    flags_active = []
    if RUN_PROBLEMAS_ABC: flags_active.append(f"Problemas A, B e C (B/C, Passo 4, k={K_EXEMPLOS_METRICA_BC})")
    if RUN_PROBLEMAS_ABC: flags_active.append(f"Problema A, Passos 2 a 6 (Passo 3 com exemplos do rótulo real, k={K_EXEMPLOS_ROTULO_REAL}; Passo 5 teto)")
    if RUN_PROBLEMA_D: flags_active.append(f"Problema D (meia-lua), Passos 2 a 6, R{'/R'.join(map(str, ATRIBUTOS_PASSO4['D']))}")
    if RUN_PROBLEMA_E: flags_active.append("Problema E, Passos 2 e 3")
    if RUN_PROBLEMAS_ABC and RUN_VIES_ORDEM_CLASSES: flags_active.append("Viés de Ordem")
    if RUN_PROBLEMAS_ABC and RUN_VIES_NOMES_ATRIBUTO: flags_active.append("Nomes de Features")
    if RUN_PROBLEMA_E and RUN_E_PASSO4_DILUICAO: flags_active.append("Diluição")
    if RUN_A_PASSO4_R3R4: flags_active.append("Não-linearidade Implícita (R3/R4)")
    if RUN_PROBLEMA_E and RUN_PROBLEMA_E2_E3: flags_active.append("Múltiplos Especialistas")
    if RUN_PROBLEMAS_ABC and RUN_A_PASSO4_PERCEPTRON_VS_NNLS: flags_active.append("Comparação Algoritmos")
    if RUN_ORACULO: flags_active.append("Validação Oráculo")
    if RUN_ORACULO and RUN_D_PASSO5_TETO: flags_active.append("Problema D, Passo 5 (teto, sem LLM)")
    if RUN_PROBLEMA_E and RUN_E_PASSO5_ORDEM: flags_active.append("Viés Ordem Exemplos")
    if RUN_PROBLEMAS_ABC and RUN_VIES_VARIANTES_PROMPT: flags_active.append("Variantes de Prompt")
    if RUN_PROBLEMA_E and RUN_E_PASSO6_BASELINES: flags_active.append("Baselines Clássicos")
    if RUN_PROBLEMA_G: flags_active.append(f"Peso×Altura G1 (normalizado) + G2 (kg/m), Passos 2 a 7 (Passo 3 com exemplos do rótulo real, k={K_EXEMPLOS_ROTULO_REAL})")
    if RUN_PROBLEMA_G and RUN_PROBLEMA_G_CLASSES_AB: flags_active.append("Peso×Altura classes A/B")
    if RUN_PROBLEMA_F: flags_active.append("Problema F (meia-lua, Passos 1 a 4)")

    models_str = '\n    '.join([f"- {p}/{m} (temp={t})" for p, m, t in MODELS_TO_TEST])

    print(f"""
    ═══════════════════════════════════════════════════
    MODELOS ({len(MODELS_TO_TEST)}):
    {models_str}
    ═══════════════════════════════════════════════════

    FLAGS ATIVAS: {', '.join(flags_active)}

    PROBLEMAS A, B e C:
      B e C, Passo 4, k (exemplos rotulados pela métrica do A): {K_EXEMPLOS_METRICA_BC}
      A e D, Passo 3, k (exemplos do rótulo real): {K_EXEMPLOS_ROTULO_REAL}
      Variações de nomes de classe: {len(NOMES_CLASSES)}
      Repetições (regra única): 1 coleta onde a seleção é determinística
        (zero-shot; margem em B/C; diluição; ordenações); {N_REPETICOES} onde há
        sorteio de exemplos (A, D e G Passo 3; E Passo 3; F Passo 3; G Passo 6).
        Flip de T=0 → auditoria.
      Seeds: {RANDOM_SEEDS} (as mesmas em todos os experimentos e modelos)
      Vieses ancorados em k = {K_VIESES}

    PROBLEMA E:
      Tamanhos few-shot (Passo 3): {K_EXEMPLOS_APRENDIZ}
      Estratégias de seleção: {EXAMPLE_STRATEGIES}
      Especialistas (configs): {[c['name'] for c in EXPERT_CONFIGS] if RUN_PROBLEMA_E2_E3 else ['E1 (principal)']}
      Amostras do Problema E: {N_PONTOS_E}
    ═══════════════════════════════════════════════════
    """)

    falhas_fatais: List[str] = []   # blocos que falharam por exceção (exit 2 no fim)
    resultados_abc = []
    resultados_abc_colapsados = []   # colapsos do Problema A (Passo 2): só CSV (collapsed=1), fora das médias

    def _registrar_resultado_abc(res):
        (resultados_abc_colapsados if getattr(res, "collapsed", 0) else resultados_abc).append(res)
    resultados_e = []
    resultados_e_passo4 = []
    # Comparação de algoritmos (NNLS) PAREADA: linhas construídas das métricas NNLS
    # calculadas sobre os MESMOS rótulos do LLM dos Problemas B e C (detail de
    # abc_executar), 0 chamadas (I12, 09/10/2026). Até 08/10 havia uma
    # recoleta de B/C (200 chamadas/seed) que misturava flip de T=0 com a diferença
    # de estimador; a recoleta foi REMOVIDA (o flip de T=0 segue mensurável pela
    # auditoria offline onde o mesmo prompt+ponto se repete: Problema D zero-shot ×
    # Problema F zero-shot, 105 pontos por seed).
    resultados_a_passo4_nnls = []
    resultados_oraculo = []  # Para validação do oráculo
    resultados_d_passo5 = []  # Problema D, Passo 5: teto da meia-lua sem LLM (origem: item 5 da reunião de 20/05/2026)
    resultados_d_passo6_svm = []  # Problema D, Passo 6: números do SVM RBF da figura bloco1_graphic_14 (16/10/2026)
    resultados_e_passo5 = []  # Para viés de ordem dos exemplos
    resultados_e_passo6 = []  # Para baselines clássicos (k-NN, LR, SVM)
    # R2/R3/R4 no Problema A: derivados das linhas ZERO-SHOT dos Passos 2 e 4
    # (0 reestimação; chaves 'fidelity'/'fidelity_nnls' + 'accuracy_vs_true*').
    results_r3_2feat = []  # métrica R2 (2 pesos) sobre rótulos zero-shot do LLM
    results_r3_3feat = []  # métrica R3 (3 pesos, x3=x1*x2) sobre os mesmos rótulos
    results_r3_4feat = []  # métrica R4 (4 pesos, +x1², x2²) — sanity check elipse no Problema A
    zero_shot_e_cache = {}  # (provider, model, seed) -> (y_llm zero-shot do Problema E, Passo 2, n_malformed) (I16)
    # Passos 2 a 5 (A e D) e transferência B/C contra todas as métricas do A
    linhas_passos_2_a_5 = []      # linhas do CSV bloco1_ad_passos_2_a_5
    linhas_bc_passo3 = []      # linhas do CSV bloco1_bc_passo3_consistencia
    rotulos_passos = {}    # (provider, model, seed, problema) -> {X, y_true, labels, metrics, zero_shot_results}
    bc_zero_shot = {}       # (provider, model, seed) -> rótulos zero-shot do LLM em B e C
    base_zero_shot_detail = {}  # (provider, model, seed) -> detail de abc_executar (base, zero-shot)
    # Cache de dados do Problema A (Passos 2 e 4) para gráficos de erros
    dados_a_passo2 = {}
    # Dados detalhados por seed para visualizações abrangentes
    seed_detailed_data = {}

    for model_idx, (provider, model_name, temperature) in enumerate(MODELS_TO_TEST):
        # Cliente da API: criado por coleta dentro do laço de eventos (ver async_client).
        MODEL_NAME = model_name
        CURRENT_PROVIDER = provider
        CURRENT_TEMPERATURE = temperature

        for seed_idx, seed in enumerate(RANDOM_SEEDS):
            # Checkpoint das interações coletadas até aqui (crash a qualquer
            # momento perde no máximo uma iteração de seed, não a execução inteira)
            checkpoint_interactions(pasta_execucao)

            print_section(
                f"MODEL {model_idx + 1}/{len(MODELS_TO_TEST)}: "
                f"{provider}/{model_name} (temp={temperature}) | "
                f"SEED {seed_idx + 1}/{len(RANDOM_SEEDS)}: {seed}",
                "═"
            )

            np.random.seed(seed)

            # ─── Passo 1: Geração dos dados sintéticos ─────────────────────────
            print(f"\n[ETAPA 1] Gerando conjuntos de dados sintéticos com semente aleatória {seed}...", flush=True)

            X_a, y_a = create_problem_a(N_PONTOS_A, seed)
            X_b, y_b = create_problem_b(N_PONTOS_B, seed + 1)
            X_c, y_c = create_problem_c(N_PONTOS_C, seed + 2)
            X_e, y_e = create_problem_e_expert(N_PONTOS_E, seed + 3)
            # Problema D (meia-lua, Bloco 1): mesma geração do Problema F e do oráculo
            X_d, y_d = create_problem_d_meialua(n_samples=N_PONTOS_A, random_state=seed)

            print(f"  Dados gerados: A={len(X_a)}, B={len(X_b)}, C={len(X_c)}, D={len(X_d)}, E={len(X_e)}")

            # Salva dados sintéticos na pasta de execução
            seed_data_dir = os.path.join(pasta_execucao, f"dados_sinteticos_seed{seed}")
            os.makedirs(seed_data_dir, exist_ok=True)
            for name, X_data, y_data in [("A", X_a, y_a), ("B", X_b, y_b), ("C", X_c, y_c), ("D", X_d, y_d), ("E", X_e, y_e)]:
                df = pd.DataFrame({"x1": X_data[:, 0], "x2": X_data[:, 1], "y": y_data})
                df.to_csv(os.path.join(seed_data_dir, f"problema_{name}.csv"), index=False)
            print(f"  Dados salvos em: {seed_data_dir}/")

            if seed_idx == 0 and model_idx == 0:
                print("\n[ETAPA 2] Gerando visualizações iniciais...")

                plot_abc_passo1(X_a, y_a, X_b, y_b, X_c, y_c,
                                       filename=os.path.join(pasta_execucao, "bloco1_graphic_01_ABC_passo1_dados.png"))
                print("  Imagem salva: bloco1_graphic_01_ABC_passo1_dados.png")

                y_expert_e_viz = especialista_rotular(X_e, EXPERT_W, EXPERT_CENTROIDS)
                plot_e_passo1(
                    X_e, y_e, y_expert_e_viz, EXPERT_W, EXPERT_CENTROIDS,
                    filename=os.path.join(pasta_execucao, "bloco2_graphic_01_E_passo1_especialista.png")
                )
                print("  Imagem salva: bloco2_graphic_01_E_passo1_especialista.png")

                plot_e_passo3_exemplos(
                    X_e, y_expert_e_viz, EXPERT_W, EXPERT_CENTROIDS,
                    n_examples=10, random_state=seed,
                    filename=os.path.join(pasta_execucao, "bloco2_graphic_02_E_passo3_estrategias_exemplos.png")
                )
                print("  Imagem salva: bloco2_graphic_02_E_passo3_estrategias_exemplos.png")

            X_train_a, y_train_a = X_a, y_a

            # ═══════════════════════════════════════════════════════════════
            # PROBLEMAS A, B E C (Experimento original de consistência)
            # ═══════════════════════════════════════════════════════════════

            cache_a_passo2 = {}

            with bloco_protegido(f"Bloco 1 (Problemas A, B, C e D, vieses) {model_name} seed={seed}",
                                 falhas_fatais, pasta_execucao):
                if RUN_PROBLEMAS_ABC:
                    print_section("BLOCO 1 — LLM como ESPECIALISTA (Problemas A, B e C lineares)", "─")
                    print("\n[ETAPA 3 DO RUNNER] Iniciando os Problemas A, B e C...")
                    print(f"  Modelo: {provider}/{model_name} | Seed: {seed} | Few-shot B/C: {K_EXEMPLOS_METRICA_BC}")

                    print("\n>>> Problemas A, B e C: experimento principal (A Passos 2 e 4; B/C Passos 2 a 4)", flush=True)
                    for k_exemplos in K_EXEMPLOS_METRICA_BC:
                        # Seleção de exemplos do Passo 4 de B e C é DETERMINÍSTICA (maior margem)
                        # → repetição não sorteia nada; roda-se 1 coleta (ver reps_para).
                        n_reps_abc = reps_para(k_exemplos, sorteio_estocastico=False)
                        for rep in range(n_reps_abc):
                            print(f"  [ABC] k={k_exemplos}, rep={rep+1}/{n_reps_abc}, seed={seed}", flush=True)
                            # A, B e C em zero-shot = Passo 2 de cada um; com k > 0 só B/C são coletados (Passo 4).
                            CURRENT_TAG = _tag("bloco1", "ABC", 2 if k_exemplos == 0 else 4,
                                               "passo2_zero_shot" if k_exemplos == 0 else "exemplos_da_metrica",
                                               k_exemplos, rep, seed)
                            verbose = (rep == 0 and k_exemplos == K_EXEMPLOS_METRICA_BC[0] and seed_idx == 0)
                            result, learned_metric, y_llm_train_a, fidelity, llm_acc_a, n_malformed, detail_bc = \
                                _abc_executar_configuracao(
                                    X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                    k_exemplos, "A", "B", rep, provider, model_name,
                                    temperature, seed, cache_a_passo2, verbose
                                )
                            _registrar_resultado_abc(result)
                            if k_exemplos == 0 and rep == 0:
                                base_zero_shot_detail[(provider, model_name, seed)] = detail_bc
                            if k_exemplos == 0 and rep == 0 and detail_bc.get('y_llm_b') is not None:
                                # Rótulos zero-shot de B/C: reavaliados contra TODAS as métricas
                                # do A (bc_passo3_consistencia), sem novas chamadas.
                                bc_zero_shot[(provider, model_name, seed)] = {
                                    'X_b': X_b, 'y_b': y_b, 'y_llm_b': detail_bc['y_llm_b'],
                                    'X_c': X_c, 'y_c': y_c, 'y_llm_c': detail_bc['y_llm_c'],
                                }

                            # Guarda dados para plots de erros do Problema A e visualizações detalhadas
                            # (TODOS os modelos alimentam as visualizações por seed —
                            # dicts chaveados por (provider, model_name, seed))
                            if k_exemplos == 0 and rep == 0 and learned_metric is not None:
                                cache_key_ab = (provider, model_name, seed, "A", "B", "default", "x1", "x2")
                                if cache_key_ab in cache_a_passo2:
                                    cached = cache_a_passo2[cache_key_ab]
                                    y_metric_a = predict_with_metric(X_train_a, cached['metric'].centroids, cached['metric'].w)
                                    dados_a_passo2[(provider, model_name, seed)] = {
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
                                        # rótulos do especialista principal (E1) no Problema E,
                                        # para o painel E de final_graphic_01 (F8)
                                        'y_expert_e': especialista_rotular(X_e, EXPERT_W, EXPERT_CENTROIDS),
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

                    print(f"  ✓ Experimento principal concluído ({len(K_EXEMPLOS_METRICA_BC)} valores de k × 1 coleta)", flush=True)

                    # Variações de nomes de classe — ancoradas em K_VIESES ({0, 10})
                    # para efeitos comparáveis com as demais análises de viés.
                    # Vieses de nomes de classe rodam em todos os modelos.
                    print(f"\n>>> Problema A, Passo 2, e B/C, Passos 3 e 4, com variações de nomes de classe (k={K_VIESES})", flush=True)
                    for nome_0, nome_1 in NOMES_CLASSES[1:]:
                        for k_vies in K_VIESES:
                            # Tag com k e o fator (P5): a auditoria separa zero de 10-shot e o par.
                            CURRENT_TAG = _tag("bloco1", "ABC", 2 if k_vies == 0 else 4, "vies_nomes_classe",
                                               k_vies, 0, seed, extra=f"nomes={nome_0}/{nome_1}")
                            print(f"  [ABC-Nomes] classes=({nome_0},{nome_1}), k={k_vies}, seed={seed}", flush=True)
                            verbose = (k_vies == K_VIESES[0] and seed_idx == 0)
                            result, *_ = _abc_executar_configuracao(
                                X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                k_vies, nome_0, nome_1, 0, provider, model_name,
                                temperature, seed, cache_a_passo2, verbose
                            )
                            _registrar_resultado_abc(result)

                    print("  ✓ Variações de nomes de classe concluídas", flush=True)

                    # ═══════════════════════════════════════════════════════
                    # TESTE DE INVERSÃO DE ORDEM DAS CLASSES
                    # ═══════════════════════════════════════════════════════
                    if RUN_VIES_ORDEM_CLASSES:
                        print(f"\n>>> Problema A, Passo 2, e B/C, Passos 3 e 4, com inversão de ordem das classes (k={K_VIESES})", flush=True)
                        for nome_0, nome_1 in NOMES_CLASSES_INVERTIDAS:
                            for k_vies in K_VIESES:
                                CURRENT_TAG = _tag("bloco1", "ABC", 2 if k_vies == 0 else 4, "vies_ordem_classes",
                                                   k_vies, 0, seed, extra=f"nomes={nome_0}/{nome_1}")
                                print(f"  [ABC-Inversão] classes=({nome_0},{nome_1}), k={k_vies}, seed={seed}", flush=True)
                                verbose = (k_vies == K_VIESES[0] and seed_idx == 0)
                                result, *_ = _abc_executar_configuracao(
                                    X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                    k_vies, nome_0, nome_1, 0, provider, model_name,
                                    temperature, seed, cache_a_passo2, verbose
                                )
                                _registrar_resultado_abc(result)
                        print("  ✓ Teste de inversão de ordem das classes concluído", flush=True)

                    # ═══════════════════════════════════════════════════════
                    # TESTE DE VARIANTES DE PROMPT (Sensibilidade ao prompt)
                    # ═══════════════════════════════════════════════════════
                    if RUN_VIES_VARIANTES_PROMPT:
                        print(f"\n>>> Problema A, Passo 2, e B/C, Passos 3 e 4, com variantes de prompt (k={K_VIESES})", flush=True)
                        for variant_name in PROMPT_VARIANTS:
                            if variant_name == "default":
                                continue  # Já testado no loop principal
                            for k_vies in K_VIESES:
                                CURRENT_TAG = _tag("bloco1", "ABC", 2 if k_vies == 0 else 4, "vies_variantes_prompt",
                                                   k_vies, 0, seed, extra=f"variante={variant_name}")
                                print(f"  [ABC-Prompt] variant={variant_name}, k={k_vies}, seed={seed}", flush=True)
                                verbose = (k_vies == K_VIESES[0] and seed_idx == 0)
                                result, *_ = _abc_executar_configuracao(
                                    X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                    k_vies, "A", "B", 0, provider, model_name,
                                    temperature, seed, cache_a_passo2, verbose,
                                    prompt_variant=variant_name
                                )
                                _registrar_resultado_abc(result)
                        print("  ✓ Teste de variantes de prompt concluído", flush=True)

                    # ═══════════════════════════════════════════════════════
                    # TESTE DE NOMES SEMÂNTICOS NAS FEATURES
                    # ═══════════════════════════════════════════════════════
                    if RUN_VIES_NOMES_ATRIBUTO:
                        # Nomes semânticos de atributo (altura/peso, feature_1/feature_2), classes
                        # A/B, zero-shot, todas as seeds. Desde 09/10/2026 (M5/I9) passa pelo
                        # MESMO caminho da configuração base (_abc_executar_configuracao →
                        # bc_passos_2_a_4): consistência B/C com centróides LOCAIS,
                        # NNLS pareado (log) e cache do Problema A chaveado pelos atributos.
                        print("\n>>> Problema A, Passo 2, e B/C, Passo 3, com nomes semânticos nos atributos (só zero-shot)", flush=True)
                        for feat_0, feat_1 in NOMES_FEATURES[1:]:  # Pula o neutro (já testado)
                            CURRENT_TAG = _tag("bloco1", "ABC", 2, "vies_nomes_atributo", 0, 0, seed,
                                               extra=f"atributos={feat_0}/{feat_1}")
                            print(f"  [ABC-Atributos] atributos=({feat_0},{feat_1}), zero-shot, seed={seed}", flush=True)
                            result_feat, metric_feat, _y_feat, fid_feat, _acc_feat, _nm_feat, detail_feat = \
                                _abc_executar_configuracao(
                                    X_train_a, y_train_a, X_b, y_b, X_c, y_c,
                                    0, "A", "B", 0, provider, model_name,
                                    temperature, seed, cache_a_passo2, verbose=(seed_idx == 0),
                                    nome_feature_0=feat_0, nome_feature_1=feat_1,
                                )
                            _registrar_resultado_abc(result_feat)
                            if metric_feat is not None:
                                # O NNLS pareado desta variante fica SÓ no log: o CSV
                                # bloco1_a_passo4_perceptron_vs_nnls é da configuração base (x1/x2) e
                                # bloco1_graphic_06 não filtra por atributo (descarte explícito).
                                _mb, _mc = detail_feat.get('metrics_nnls_b'), detail_feat.get('metrics_nnls_c')
                                print(f"    [Perceptron] atributos {feat_0}/{feat_1}: fidelidade={fid_feat:.1%}, "
                                      f"consistência B={result_feat.consistencia_problema_b:.1%}, "
                                      f"C={result_feat.consistencia_problema_c:.1%} (centróides locais)")
                                if _mb is not None and _mc is not None:
                                    print(f"    [NNLS]       atributos {feat_0}/{feat_1}: fidelidade={detail_feat.get('fidelity_nnls_a', float('nan')):.1%}, "
                                          f"consistência B={_mb.accuracy:.1%}, C={_mc.accuracy:.1%} (pareado; só no log)")
                            else:
                                print(f"    ⚠ Colapso no Problema A (Passo 2) com atributos {feat_0}/{feat_1}: linha com collapsed=1")
                        print("  ✓ Teste de nomes de atributos concluído", flush=True)

                    # ═══════════════════════════════════════════════════════
                    # COMPARAÇÃO DE ALGORITMOS
                    # ═══════════════════════════════════════════════════════
                    if RUN_A_PASSO4_PERCEPTRON_VS_NNLS:
                        # Perceptron × NNLS PAREADO (I12, 0 chamadas): as métricas do NNLS em
                        # B/C vêm de `detail` de abc_executar (mesmos rótulos do LLM
                        # da configuração base, zero-shot). A recoleta de B/C foi removida (ver
                        # comentário em resultados_a_passo4_nnls).
                        print("\n>>> Comparação de Algoritmos: Perceptron vs NNLS (pareada, 0 chamadas)", flush=True)
                        _det = base_zero_shot_detail.get((provider, model_name, seed))
                        _cached_ab = cache_a_passo2.get((provider, model_name, seed, "A", "B", "default", "x1", "x2"))
                        if _det and _det.get('metrics_nnls_b') is not None and _det.get('metrics_nnls_c') is not None and _cached_ab:
                            _mb, _mc = _det['metrics_nnls_b'], _det['metrics_nnls_c']
                            w_alt = np.asarray(_det['w_nnls'], float)
                            fid_alt = float(_det.get('fidelity_nnls_a', float('nan')))
                            result_alt = ResultadoABC(
                                provider=provider, model_name=model_name, temperature=temperature,
                                random_seed=seed, k_passo4_bc=0,
                                nomes_classes=("A", "B"), repeticao=0,
                                fidelidade_problema_a=fid_alt,
                                acuracia_llm_vs_gt_problema_a=_cached_ab['llm_acc'],
                                consistencia_problema_b=float(_mb.accuracy),
                                kappa_problema_b=float(_mb.cohen_kappa),
                                f1_problema_b=float(_mb.f1_score),
                                acuracia_llm_vs_gt_problema_b=_det['llm_accuracy_b'],
                                acuracia_metrica_vs_gt_problema_b=_det['metric_accuracy_nnls_b'],
                                consistencia_problema_c=float(_mc.accuracy),
                                kappa_problema_c=float(_mc.cohen_kappa),
                                f1_problema_c=float(_mc.f1_score),
                                acuracia_llm_vs_gt_problema_c=_det['llm_accuracy_c'],
                                acuracia_metrica_vs_gt_problema_c=_det['metric_accuracy_nnls_c'],
                                w_aprendido=w_alt, gamma_otimo=0.0, margem_minima=float('nan'),
                                n_classe_0_problema_a=int(np.sum(_cached_ab['y_llm'] == 0)),
                                n_classe_1_problema_a=int(np.sum(_cached_ab['y_llm'] == 1)),
                                n_classe_0_problema_b=int(np.sum(_det['y_llm_b'] == 0)),
                                n_classe_1_problema_b=int(np.sum(_det['y_llm_b'] == 1)),
                                n_classe_0_problema_c=int(np.sum(_det['y_llm_c'] == 0)),
                                n_classe_1_problema_c=int(np.sum(_det['y_llm_c'] == 1)),
                                n_disagreements_b=_mb.n_disagreements,
                                n_disagreements_c=_mc.n_disagreements,
                                n_malformed_responses=0,   # mesmos rótulos: malformadas já contadas na linha Perceptron
                                w_direction=w_alt / np.linalg.norm(w_alt) if np.linalg.norm(w_alt) > 0 else w_alt,
                                w_cosine_sim_nnls=0.0,  # Este JÁ é o NNLS, não há segundo algoritmo
                            )
                            resultados_a_passo4_nnls.append(result_alt)
                            if (provider, model_name, seed) in seed_detailed_data:
                                seed_detailed_data[(provider, model_name, seed)]['w_nnls'] = w_alt
                                seed_detailed_data[(provider, model_name, seed)]['centroids_nnls'] = _cached_ab['metric'].centroids
                            print("    --- Problema A, Passo 4 (fidelidade) ---", flush=True)
                            print(f"    Perceptron: W=[{_cached_ab['metric'].w[0]:.4f}, {_cached_ab['metric'].w[1]:.4f}], Fidelidade={_cached_ab['fidelity']:.1%}")
                            print(f"    NNLS:       W=[{w_alt[0]:.4f}, {w_alt[1]:.4f}], Fidelidade={fid_alt:.1%}")
                            _pb = _det['metrics_b']; _pc = _det['metrics_c']
                            print("    --- Problemas B e C, Passo 3 (consistência, mesmos rótulos do LLM) ---", flush=True)
                            if _pb is not None:
                                print(f"    Perceptron: Consistência B={_pb.accuracy:.1%}, Cohen's κ={_pb.cohen_kappa:.3f}")
                            print(f"    NNLS:       Consistência B={_mb.accuracy:.1%}, Cohen's κ={_mb.cohen_kappa:.3f}")
                            if _pc is not None:
                                print(f"    Perceptron: Consistência C={_pc.accuracy:.1%}, Cohen's κ={_pc.cohen_kappa:.3f}")
                            print(f"    NNLS:       Consistência C={_mc.accuracy:.1%}, Cohen's κ={_mc.cohen_kappa:.3f}")
                        else:
                            print("    (sem métricas NNLS pareadas: colapso no A ou em B/C nesta seed)")
                        print("  ✓ Comparação de algoritmos concluída", flush=True)

                # ═══════════════════════════════════════════════════════════════
                # PASSOS 2 A 5 DOS PROBLEMAS A E D (orientador, 01/10/2026):
                #   Problema A — zero-shot (cache; Passo 2) + few-shot com exemplos do GROUND
                #   TRUTH (Passo 3, k em K_EXEMPLOS_ROTULO_REAL) + W ajustado ao GT (teto, Passo 5), em R2/R3/R4;
                #   Problemas B e C, Passo 3 — consistência contra TODAS as métricas do A (0 chamadas);
                #   Problema D — meia-lua, 150 pontos, SEM split, mesmos passos.
                # ═══════════════════════════════════════════════════════════════
                if RUN_PROBLEMAS_ABC:
                    print_section("BLOCO 1 — PROBLEMA A, PASSOS 2 A 5: zero-shot | k exemplos do rótulo real | teto", "─")
                    cache_key_ab = (provider, model_name, seed, "A", "B", "default", "x1", "x2")
                    cached_ab = cache_a_passo2.get(cache_key_ab)
                    proto_a = adg_passos_2_a_5(
                        X_train_a, y_train_a, "A", "B", "A", seed, provider, model_name, temperature,
                        n_features_list=ATRIBUTOS_PASSO4["A"],
                        y_llm_zero_cached=(cached_ab['y_llm'] if cached_ab else None),
                        n_malformed_zero=(cached_ab['n_malformed'] if cached_ab else 0),
                        verbose=(seed_idx == 0), label_prefix=f"[A seed={seed}] ",
                    )
                    linhas_passos_2_a_5.extend(proto_a['rows'])
                    rotulos_passos[(provider, model_name, seed, "A")] = {
                        'X': X_train_a, 'y_true': y_train_a, 'labels': proto_a['labels'],
                        'metrics': proto_a['metrics'], 'zero_shot_results': proto_a['zero_shot_results'],
                    }
                    print(f"  ✓ Problema A, Passos 2 a 5, concluídos ({len(proto_a['rows'])} linhas, "
                          f"{proto_a['n_calls']} chamadas novas)", flush=True)
                    bc = bc_zero_shot.get((provider, model_name, seed))
                    if bc:
                        base_bc = {"provider": provider, "model": model_name, "temperature": temperature, "random_seed": seed}
                        for _prob, _Xk, _yk, _ylk in (("B", bc['X_b'], bc['y_b'], bc['y_llm_b']),
                                                      ("C", bc['X_c'], bc['y_c'], bc['y_llm_c'])):
                            linhas_bc_passo3.extend(bc_passo3_consistencia(_Xk, _ylk, _yk, _prob, proto_a['metrics'], base_bc))
                        _n_metricas_r2 = sum(1 for _k in proto_a['metrics'] if _k[3] == 2)
                        print(f"  ✓ Problemas B e C, Passo 3: transferência avaliada contra {_n_metricas_r2} métricas do A em R² (0 chamadas)", flush=True)

                if RUN_PROBLEMA_D:
                    print_section("BLOCO 1 — PROBLEMA D, PASSOS 2 A 6 (meia-lua, 150 pontos, sem split)", "─")
                    if model_idx == 0:
                        plot_problema_passo1(
                            X_d, y_d,
                            title=f"Problema D, Passo 1: meia-lua (sklearn.make_moons, seed={seed})",
                            feature_names=("x1", "x2"), optimal_boundary_fn=None,
                            filename=os.path.join(pasta_execucao, f"bloco1_graphic_12_D_passo1_dados_seed{seed}.png"),
                        )
                        print(f"  Gráfico salvo: bloco1_graphic_12_D_passo1_dados_seed{seed}.png")
                    proto_d = adg_passos_2_a_5(
                        X_d, y_d, "A", "B", "D", seed, provider, model_name, temperature,
                        n_features_list=ATRIBUTOS_PASSO4["D"],
                        verbose=(seed_idx == 0), label_prefix=f"[D seed={seed}] ",
                    )
                    linhas_passos_2_a_5.extend(proto_d['rows'])
                    rotulos_passos[(provider, model_name, seed, "D")] = {
                        'X': X_d, 'y_true': y_d, 'labels': proto_d['labels'],
                        'metrics': proto_d['metrics'], 'zero_shot_results': proto_d['zero_shot_results'],
                    }
                    print(f"  ✓ Problema D, Passos 2 a 5, concluídos ({len(proto_d['rows'])} linhas, "
                          f"{proto_d['n_calls']} chamadas novas)", flush=True)
                    # bloco1_graphic_14: SVM RBF vs rótulos zero-shot do LLM nos 150 pontos (antes: 105
                    # do split). A fronteira tracejada é a métrica de maior FIDELIDADE vs LLM (F13).
                    try:
                        zs = {nf: r for nf, r in proto_d['zero_shot_results'].items() if r is not None}
                        best_d = max(zs.values(), key=lambda r: r['fidelity_perc_vs_llm']) if zs else None
                        y_llm_zero_d = proto_d['labels'][("passo2_zero_shot", 0, 0)]['y_llm_eval']
                        fname_svm = llm_asset(pasta_execucao, f"bloco1_graphic_14_D_passo6_svm_seed{seed}.png", model_name)
                        if plot_d_passo6_svm(X_d, y_d, y_llm_zero_d, metric=best_d, seed=seed, filename=fname_svm):
                            print(f"  Gráfico salvo: {os.path.basename(fname_svm)}")
                        else:
                            print(f"  ⚠ Plot SVM meia-lua seed={seed} não gerado (classe única no ground truth).")
                        # Números da figura (tabela bloco1_table_19, 16/10/2026): os mesmos SVM RBF, nos 150 pontos
                        from classical_baselines import d_passo6_svm_medidas
                        med = d_passo6_svm_medidas(X_d, y_d, y_llm_zero_d, metric=best_d)
                        med.update({"provider": provider, "model": model_name, "random_seed": seed})
                        resultados_d_passo6_svm.append(med)
                        print(f"  Problema D, Passo 6, seed={seed}: SVM no rótulo real {med['svm_real_accuracy_vs_true']:.1%} vs real; "
                              + ("LLM colapsado (SVM nos rótulos do LLM não ajustável)" if med["collapsed"] else
                                 f"SVM nos rótulos do LLM: fidelidade {med['svm_llm_fidelity_vs_llm']:.1%}, {med['svm_llm_accuracy_vs_true']:.1%} vs real"))
                    except Exception as exc_svm:
                        print(f"  ⚠ Falha no plot SVM meia-lua seed={seed}: {exc_svm}")

            # ═══════════════════════════════════════════════════════════════
            # VALIDAÇÃO DO ORÁCULO: ALGORITMOS RECUPERAM W CONHECIDO?
            # ═══════════════════════════════════════════════════════════════

            # Oráculo é INDEPENDENTE do modelo (0 chamadas LLM) — roda uma vez, no
            # primeiro modelo, para todas as seeds (de graça: não usam o LLM).
            with bloco_protegido(f"Oráculo (validação sem LLM) seed={seed}", falhas_fatais, pasta_execucao):
                if RUN_ORACULO and model_idx == 0:
                    print("\n>>> Validação do Oráculo: Algoritmos recuperam W conhecido?", flush=True)
                    linhas_oraculo = oraculo_passos_1_a_3(
                        X_e, EXPERT_CONFIGS,
                        random_seed=seed,
                        verbose=(seed_idx == 0),
                    )
                    resultados_oraculo.extend(linhas_oraculo)  # acumula as 3 seeds (18 linhas)
                    print(f"  ✓ Validação do oráculo concluída para seed={seed}", flush=True)

                    # Passo 5 do Problema D calculado sem LLM (idêntico ao regime
                    # `passo5_teto` de bloco1_ad_passos_2_a_5): W ajustado ao rótulo real
                    # da meia-lua em R², R³ e R⁴, sem expectativa a priori sobre a ordem.
                    # Sub-flag do oráculo (0 chamadas LLM): independe das coletas do
                    # Problema D/F, para que `--apenas oraculo` valide também a meia-lua.
                    # (Origem: item 5 da reunião de 20/05/2026.)
                    if RUN_D_PASSO5_TETO:
                        linhas_d_passo5 = d_passo5_teto(
                            X_d, y_d, random_seed=seed, verbose=(seed_idx == 0),
                        )
                        resultados_d_passo5.extend(linhas_d_passo5)

            CURRENT_TAG = ""   # definida por experimento (Problema E, diluição, ordem) abaixo
            # ═══════════════════════════════════════════════════════════════
            # PROBLEMA E: LLM COMO APRENDIZ
            # ═══════════════════════════════════════════════════════════════

            with bloco_protegido(f"Bloco 2 (Problema E e auxiliares) {model_name} seed={seed}",
                                 falhas_fatais, pasta_execucao):
                if RUN_PROBLEMA_E:
                    expert_configs_to_run = EXPERT_CONFIGS if RUN_PROBLEMA_E2_E3 else [EXPERT_CONFIGS[0]]

                    # Problema E, Passo 2 (zero-shot): o prompt não depende do especialista, logo a coleta
                    # é feita UMA vez por (modelo, seed) e reaproveitada em E1/E2/E3 (I16);
                    # as métricas são recalculadas por especialista em e_passos_2_3_aprendiz.
                    if (provider, model_name, seed) not in zero_shot_e_cache:
                        CURRENT_TAG = _tag("bloco2", "E", 2, "passo2_zero_shot", 0, 0, seed)
                        print(f"  [Problema E, Passo 2] zero-shot único por (modelo, seed): {len(X_e)} pontos, seed={seed}", flush=True)
                        zero_shot_e_cache[(provider, model_name, seed)] = collect_llm_decisions(
                            X_e, "A", "B", examples=None, verbose=(seed_idx == 0),
                            label_prefix="[Problema E, Passo 2] ",
                        )
                    y_llm_zero_e, n_malf_zero_e = zero_shot_e_cache[(provider, model_name, seed)]

                    for expert_cfg in expert_configs_to_run:
                        expert_w = expert_cfg["w"]
                        expert_name = expert_cfg["name"]
                        expert_centroids = EXPERT_CENTROIDS

                        print(f"\n[ETAPA 4 DO RUNNER] Problema {expert_name}, Passos 2 e 3 — especialista {expert_name} ({expert_cfg['desc']})")
                        print(f"  W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
                        print_section(f"BLOCO 2 — LLM como APRENDIZ | PROBLEMA {expert_name}, PASSOS 2 E 3", "═")

                        total_e_combos = len(K_EXEMPLOS_APRENDIZ) * len(EXAMPLE_STRATEGIES)
                        combo_count = 0
                        for k_e in K_EXEMPLOS_APRENDIZ:
                            for strategy in EXAMPLE_STRATEGIES:
                                combo_count += 1
                                print(f"  [Problema {expert_name}, Passo {2 if k_e == 0 else 3}] k={k_e}, strategy={strategy} ({combo_count}/{total_e_combos}), seed={seed}", flush=True)
                                if k_e == 0 and strategy != EXAMPLE_STRATEGIES[0]:
                                    base_results = resultados_e[-reps_para(0):]
                                    for rep, prev_result in enumerate(base_results):
                                        dup_result = ResultadoAprendiz(
                                            provider=prev_result.provider,
                                            model_name=prev_result.model_name,
                                            temperature=prev_result.temperature,
                                            random_seed=prev_result.random_seed,
                                            k=0,
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
                                        resultados_e.append(dup_result)
                                    continue

                                # Zero-shot: 1 coleta (não há exemplos a sortear). Few-shot:
                                # só a estratégia "random" consome o sorteio (random_state=
                                # seed+rep); easy/hard/mixed são determinísticas por margem —
                                # repetir recoletaria o MESMO prompt (o flip de T=0 já é
                                # quantificado pela auditoria offline). Regra única: reps_para.
                                reps_e = reps_para(k_e, sorteio_estocastico=(strategy == "random"))
                                for rep in range(reps_e):
                                    is_verbose = (
                                        rep == 0 and seed_idx == 0 and
                                        (k_e in [0, K_EXEMPLOS_APRENDIZ[-1]])
                                    )
                                    # Tag (P5): zero-shot = Passo 2 (não leva o nome da estratégia); Passo 3 = estratégia.
                                    CURRENT_TAG = _tag("bloco2", expert_name, 2 if k_e == 0 else 3,
                                                       "passo2_zero_shot" if k_e == 0 else strategy,
                                                       k_e, rep, seed)

                                    result_e = e_passos_2_3_aprendiz(
                                        X_e.copy(), y_e.copy(),
                                        expert_w=expert_w,
                                        expert_centroids=expert_centroids,
                                        k_exemplos=k_e,
                                        strategy=strategy,
                                        nome_classe_0="A",
                                        nome_classe_1="B",
                                        provider=provider,
                                        model_name=model_name,
                                        temperature=temperature,
                                        random_seed=seed,
                                        repeticao=rep,
                                        verbose=is_verbose,
                                        y_llm_cached=(y_llm_zero_e if k_e == 0 else None),
                                        n_malformed_cached=(n_malf_zero_e if k_e == 0 else 0),
                                    )
                                    result_e.expert_name = expert_name
                                    resultados_e.append(result_e)

                    print(f"  ✓ Problema E, Passos 2 e 3, concluídos para seed={seed}", flush=True)

                    # ═══════════════════════════════════════════════════════
                    # BASELINES CLÁSSICOS (k-NN, LR, SVM)
                    # ═══════════════════════════════════════════════════════
                    # Baselines clássicos são INDEPENDENTES do LLM (treinam nos exemplos
                    # rotulados pelo especialista) — rodar uma vez, no primeiro modelo, evita
                    # linhas duplicadas nos CSVs/plots com múltiplos modelos.
                    # Mesmos exemplos do Passo 3 em TODAS as reps (15/10/2026): random_state =
                    # seed + rep, com reps_para (3 reps só na estratégia 'random', que é a única
                    # que sorteia; easy/hard/mixed são determinísticas e ficam em rep = 0). A
                    # coluna `rep` vai ao CSV. Roda para todos os especialistas de EXPERT_CONFIGS
                    # quando RUN_PROBLEMA_E2_E3 (custo zero de API); gate model_idx == 0, em
                    # todas as seeds do 1º modelo.
                    if RUN_E_PASSO6_BASELINES and model_idx == 0:
                        print("\n>>> Problema E, Passo 6: baselines clássicos", flush=True)
                        for expert_cfg_bl in (EXPERT_CONFIGS if RUN_PROBLEMA_E2_E3 else [EXPERT_CONFIGS[0]]):
                            expert_w_bl = expert_cfg_bl["w"]
                            expert_name_bl = expert_cfg_bl["name"]
                            y_expert_bl = especialista_rotular(X_e, expert_w_bl, EXPERT_CENTROIDS)

                            for k_bl in K_EXEMPLOS_APRENDIZ:
                                if k_bl == 0:
                                    continue  # Baselines precisam de dados de treino
                                for strategy_bl in EXAMPLE_STRATEGIES:
                                    for rep_bl in range(reps_para(k_bl, sorteio_estocastico=(strategy_bl == "random"))):
                                        _, indices_bl = e_passo3_selecionar_exemplos(
                                            X_e, y_expert_bl, expert_w_bl, EXPERT_CENTROIDS,
                                            n_examples=k_bl, strategy=strategy_bl,
                                            nome_classe_0="A", nome_classe_1="B",
                                            random_state=seed + rep_bl, verbose=False
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

                                        is_verbose_bl = (seed_idx == 0 and k_bl == K_EXEMPLOS_APRENDIZ[-1]
                                                         and strategy_bl == "mixed" and rep_bl == 0)
                                        if is_verbose_bl:
                                            print(f"  [Problema {expert_name_bl}, Passo 6: baselines] k={k_bl}, "
                                                  f"strategy={strategy_bl}, seed={seed}, rep={rep_bl}", flush=True)

                                        runner = ClassicalBaselineRunner(verbose=is_verbose_bl)
                                        baseline_results = runner.run(
                                            X_train_bl, y_train_bl,
                                            X_test_bl, y_expert_test_bl, y_gt_test_bl)

                                        for clf_name, metrics in baseline_results.items():
                                            resultados_e_passo6.append({
                                                'provider': 'classical',
                                                'model': clf_name,
                                                'random_seed': seed,
                                                'k': k_bl,
                                            'rep': rep_bl,
                                                'example_strategy': strategy_bl,
                                                'expert_name': expert_name_bl,
                                                'accuracy_vs_expert': metrics['accuracy_vs_expert'],
                                                'kappa_vs_expert': metrics['kappa_vs_expert'],
                                                'f1_vs_expert': metrics['f1_vs_expert'],
                                                'accuracy_vs_gt': metrics['accuracy_vs_gt'],
                                                'accuracy_expert_vs_gt': float(accuracy_score(y_gt_test_bl, y_expert_test_bl)),
                                                'n_total_test': len(X_test_bl),
                                            })

                        print(f"  ✓ Baselines clássicos concluídos para seed={seed}", flush=True)

                    # ═══════════════════════════════════════════════════════
                    # EXPERIMENTO DE DILUIÇÃO
                    # ═══════════════════════════════════════════════════════
                    if RUN_E_PASSO4_DILUICAO:
                        n_hard_fixed = 4  # 2 por classe (arredondado para par)
                        print("\n>>> Problema E1, Passo 4: diluição (4 difíceis fixos + N fáceis progressivos)", flush=True)
                        y_expert_dilution = especialista_rotular(X_e, EXPERT_W, EXPERT_CENTROIDS)
                        easy_additions = DILUTION_EASY_ADDITIONS  # N easy adicionados

                        for dil_idx, n_easy in enumerate(easy_additions):
                            n_total = n_hard_fixed + n_easy
                            print(f"  [Diluição] {n_hard_fixed} hard + {n_easy} easy = {n_total} total ({dil_idx+1}/{len(easy_additions)})", flush=True)

                            examples_dil, selected_dil = e_passo4_exemplos_diluicao(
                                X_e, y_expert_dilution, EXPERT_W, EXPERT_CENTROIDS,
                                n_hard_fixed=n_hard_fixed, n_easy_added=n_easy,
                                nome_classe_0="A", nome_classe_1="B",
                                verbose=True
                            )

                            # Monta conjunto de teste
                            test_mask = np.ones(len(X_e), dtype=bool)
                            test_mask[selected_dil] = False
                            X_test_dil = X_e[test_mask]
                            y_expert_test_dil = y_expert_dilution[test_mask]
                            y_gt_test_dil = y_e[test_mask]

                            # Seleção da diluição é determinística (margem) → 1 coleta.
                            for rep in range(reps_para(n_total, sorteio_estocastico=False)):
                                CURRENT_TAG = _tag("bloco2", "E1", 4, "diluicao", n_total, rep, seed, extra=f"easy={n_easy}")
                                y_llm_dil, n_malf_dil = collect_llm_decisions(
                                    X_test_dil, "A", "B",
                                    examples=examples_dil,
                                    verbose=(rep == 0 and seed_idx == 0),
                                    label_prefix=f"[E1 Passo 4 diluição k={n_total}] "
                                )
                                cons_dil = compute_consistency_metrics(y_llm_dil, y_expert_test_dil)
                                llm_acc_dil = accuracy_score(y_gt_test_dil, y_llm_dil)
                                expert_acc_dil = accuracy_score(y_gt_test_dil, y_expert_test_dil)

                                result_dil = ResultadoAprendiz(
                                    provider=provider, model_name=model_name,
                                    temperature=temperature, random_seed=seed,
                                    k=n_total,
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
                                resultados_e_passo4.append(result_dil)
                        print(f"  ✓ Experimento de diluição concluído para seed={seed}", flush=True)

                # ═══════════════════════════════════════════════════════════════
                # VIÉS DE ORDEM DOS EXEMPLOS FEW-SHOT (Recency Bias)
                # ═══════════════════════════════════════════════════════════════

                if RUN_E_PASSO5_ORDEM and RUN_PROBLEMA_E:
                    print("\n>>> Problema E1, Passo 5: viés de ordem dos exemplos (mesmos exemplos da estratégia mista, 4 ordenações)", flush=True)
                    y_expert_order = especialista_rotular(X_e, EXPERT_W, EXPERT_CENTROIDS)

                    # Usa os k_exemplos de K_ORDEM_EXEMPLOS e a estratégia "mixed"
                    # como configuração fixa para isolar o efeito da ordenação
                    for k_ordem in K_ORDEM_EXEMPLOS:
                        # Seleciona exemplos uma vez (estratégia mixed)
                        base_examples, base_indices = e_passo3_selecionar_exemplos(
                            X_e, y_expert_order, EXPERT_W, EXPERT_CENTROIDS,
                            n_examples=k_ordem,
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
                        # Acurácia (vs rótulo real) do especialista sobre o TESTE (P6)
                        expert_acc_order = accuracy_score(y_gt_test_order, y_expert_test_order)

                        for ordering in EXAMPLE_ORDERINGS:
                            reordered = e_passo5_reordenar_exemplos(
                                base_examples, ordering,
                                nome_classe_0="A", nome_classe_1="B",
                                random_state=seed
                            )
                            print(f"  [Problema E1, Passo 5: ordem] k={k_ordem}, ordering={ordering}, seed={seed}", flush=True)

                            # Compara ORDENAÇÕES FIXAS do mesmo conjunto de exemplos —
                            # não há sorteio por repetição → 1 coleta por ordenação.
                            n_reps_order = reps_para(k_ordem, sorteio_estocastico=False)
                            for rep in range(n_reps_order):
                                CURRENT_TAG = _tag("bloco2", "E1", 5, ordering, k_ordem, rep, seed)
                                t_rep_start = time.time()
                                print(f"    [Ordem {ordering}] rep {rep+1}/{n_reps_order} (n_test={len(X_test_order)})...", flush=True)
                                y_llm_order, n_malf_order = collect_llm_decisions(
                                    X_test_order, "A", "B",
                                    examples=reordered,
                                    verbose=(rep == 0 and seed_idx == 0 and ordering == EXAMPLE_ORDERINGS[0] and k_ordem == K_ORDEM_EXEMPLOS[0]),
                                    label_prefix=f"[Ordem {ordering}] "
                                )
                                cons_order = compute_consistency_metrics(y_llm_order, y_expert_test_order)
                                llm_acc_order = accuracy_score(y_gt_test_order, y_llm_order)
                                print(
                                    f"    [Ordem {ordering}] rep {rep+1}/{n_reps_order} concluída em "
                                    f"{time.time()-t_rep_start:.1f}s — acurácia (LLM vs especialista)={cons_order.accuracy:.1%}, "
                                    f"Cohen's κ={cons_order.cohen_kappa:.3f}, malformadas={n_malf_order}",
                                    flush=True,
                                )

                                result_order = ResultadoAprendiz(
                                    provider=provider, model_name=model_name,
                                    temperature=temperature, random_seed=seed,
                                    k=k_ordem,
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
                                resultados_e_passo5.append(result_order)

                    print(f"  ✓ Experimento de ordem dos exemplos concluído para seed={seed}", flush=True)

            # ═══════════════════════════════════════════════════════════════
            # NÃO-LINEARIDADE IMPLÍCITA: MÉTRICAS R3 E R4 SOBRE RÓTULOS R2 (Problema A)
            # O LLM classifica apenas com (x1, x2). Nos bastidores, adicionamos
            # x3 = x1*x2 (R3) e depois x1², x2² (R4) e estimamos métricas com 3 e 4
            # pesos. Se a fidelidade cresce com a dimensão, o LLM adota
            # implicitamente um critério não-linear.
            # ═══════════════════════════════════════════════════════════════

            with bloco_protegido(f"R2/R3/R4 no Problema A {model_name} seed={seed}", falhas_fatais, pasta_execucao):
                if RUN_A_PASSO4_R3R4:
                    print(f"\n>>> Projeção R3/R4 no Problema A: x3 = x1·x2; depois x1², x2² (seed={seed})", flush=True)
                    # Desde 09/10/2026 as métricas R2/R3/R4 vêm das linhas ZERO-SHOT do
                    # Passo 4 do Problema A (adg_passos_2_a_5, n_features 2/3/4 sobre os MESMOS
                    # rótulos do LLM): 0 reestimação (os números eram bit a bit iguais).
                    # O LLM só viu (x1, x2); se a fidelidade cresce com a dimensão, ele adota
                    # implicitamente um critério não-linear. Aqui ficam a gravação dos CSVs
                    # problem_A_r3/r4.csv, o log comparativo e as listas de bloco1_graphic_08.
                    _proto = rotulos_passos.get((provider, model_name, seed, "A"))
                    zs_a = {nf: r for nf, r in (_proto or {}).get('zero_shot_results', {}).items()}
                    if _proto is None:
                        print("  ⚠ Passos 2 e 4 do Problema A não disponíveis para esta seed, pulando R3/R4")
                    else:
                        X_a_r3 = augment_to_r3(X_train_a)
                        X_a_r4 = augment_to_r4(X_train_a)
                        seed_data_dir = os.path.join(pasta_execucao, f"dados_sinteticos_seed{seed}")
                        os.makedirs(seed_data_dir, exist_ok=True)
                        pd.DataFrame({"x1": X_a_r3[:, 0], "x2": X_a_r3[:, 1], "x3": X_a_r3[:, 2], "y": y_train_a}).to_csv(
                            os.path.join(seed_data_dir, "problema_A_r3.csv"), index=False)
                        pd.DataFrame({"x1": X_a_r4[:, 0], "x2": X_a_r4[:, 1], "x1_sq": X_a_r4[:, 2],
                                      "x2_sq": X_a_r4[:, 3], "y": y_train_a}).to_csv(
                            os.path.join(seed_data_dir, "problema_A_r4.csv"), index=False)
                        print(f"  Dados R3/R4 salvos em: {seed_data_dir}/problema_A_r3.csv e problema_A_r4.csv")

                        def _linha_r(nf):
                            r = zs_a.get(nf)
                            if r is None:
                                return None
                            return {'fidelity': float(r['fidelity_perc_vs_llm']), 'w': np.asarray(r['w_perc'], float),
                                    'fidelity_nnls': float(r['fidelity_nnls_vs_llm']), 'w_nnls': np.asarray(r['w_nnls'], float),
                                    'accuracy_vs_true': float(r['accuracy_perc_vs_true']),
                                    'accuracy_vs_true_nnls': float(r['accuracy_nnls_vs_true']),
                                    'seed': seed, 'provider': provider, 'model': model_name}
                        r2, r3, r4 = _linha_r(2), _linha_r(3), _linha_r(4)
                        if r2 is None or r3 is None:
                            print("  ⚠ Zero-shot colapsado nesta seed (sem métrica R2/R3): R3/R4 não comparável")
                        else:
                            results_r3_2feat.append(r2); results_r3_3feat.append(r3)
                            print(f"  [Perceptron] Ŵ_LLM (3 pesos): [{', '.join(f'{v:.4f}' for v in r3['w'])}]")
                            print(f"  [NNLS]       Ŵ_LLM (3 pesos): [{', '.join(f'{v:.4f}' for v in r3['w_nnls'])}]")
                            print("  --- Comparação de fidelidade (mesmos rótulos do LLM) ---")
                            print(f"  [Perceptron] Fidelidade R2 (2 pesos): {r2['fidelity']:.1%}  →  R3 (3 pesos): {r3['fidelity']:.1%}  (Δ = {r3['fidelity'] - r2['fidelity']:+.1%})")
                            print(f"  [NNLS]       Fidelidade R2 (2 pesos): {r2['fidelity_nnls']:.1%}  →  R3 (3 pesos): {r3['fidelity_nnls']:.1%}  (Δ = {r3['fidelity_nnls'] - r2['fidelity_nnls']:+.1%})")
                            if r3['fidelity'] > r2['fidelity']:
                                print("  ► Evidência de não-linearidade implícita: métrica R3 se ajusta melhor")
                            else:
                                print("  ► Sem evidência de não-linearidade: métrica R2 já é suficiente")
                            if r4 is not None:
                                results_r3_4feat.append(r4)
                                print(f"  [Perceptron] Ŵ_LLM (4 pesos): [{', '.join(f'{v:.4f}' for v in r4['w'])}]")
                                print(f"  [NNLS]       Ŵ_LLM (4 pesos): [{', '.join(f'{v:.4f}' for v in r4['w_nnls'])}]")
                                print(f"  [Perceptron] Fidelidade R3 (3 pesos): {r3['fidelity']:.1%}  →  R4 (4 pesos): {r4['fidelity']:.1%}  (Δ = {r4['fidelity'] - r3['fidelity']:+.1%})")
                                print(f"  [NNLS]       Fidelidade R3 (3 pesos): {r3['fidelity_nnls']:.1%}  →  R4 (4 pesos): {r4['fidelity_nnls']:.1%}  (Δ = {r4['fidelity_nnls'] - r3['fidelity_nnls']:+.1%})")
                                if r4['fidelity'] > r3['fidelity']:
                                    print("  ► R4 (elipse) supera R3 — sinal de não-linearidade quadrática isotrópica")
                                else:
                                    print("  ► R4 (elipse) NÃO supera R3 — problema linear, conforme esperado no Problema A")
                        print(f"  ✓ Experimento R2 vs R3 vs R4 concluído (seed={seed})", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # PROBLEMAS G1/G2 (peso × altura, Bloco 3) e F (meia-lua, Bloco 2): split 70/30
    # ═══════════════════════════════════════════════════════════════════

    resultados_fg_passos_2_e_4: List[dict] = []
    linhas_g_passos_2_a_5: List[dict] = []   # Problema G, Passos 2 a 5 (bloco3_g_passos_2_a_5)
    resultados_fg_aprendiz: List[dict] = []
    mapas_rotulos_fg: dict = {}

    # Base real carregada uma vez, fora de G1 e G2: falha aqui registra UMA entrada e
    # pula os dois subproblemas; falha em G1 não impede G2 (e vice-versa).
    X_hm = y_hm = None
    if RUN_PROBLEMA_G:
        try:
            print_section("BLOCO 3 — ESTUDO DE CASO REAL: PESO × ALTURA (base de 100 pessoas, G1 e G2)", "═")
            X_hm, y_hm = create_problem_homem_mulher(
                str(BASE_DIR / "dados_reais/homem_mulher/peso_altura.csv")
            )
            print(f"  Base: shape={X_hm.shape} | classes={dict(zip(*np.unique(y_hm, return_counts=True)))}")
            print("  Classificador ótimo: elipse x2² - x2 + x1² - x1 + cte")

            # Elipse do orientador (I7, 09/10/2026): a constante é calculada UMA vez sobre
            # X_hm, como o limiar de z = x2² - x2 + x1² - x1 que maximiza a acurácia vs
            # y_hm (varrendo os pontos médios entre valores consecutivos de z; classe 1 =
            # z acima do limiar). A função devolve z - cte vetorizada para a malha do
            # gráfico. (Até 08/10 indexava a malha 200×200 com y_hm de 100 posições:
            # IndexError engolido, e a fronteira nunca era desenhada.)
            _z_hm = X_hm[:, 1] ** 2 - X_hm[:, 1] + X_hm[:, 0] ** 2 - X_hm[:, 0]
            _zs = np.sort(np.unique(_z_hm))
            _cands = np.concatenate([[_zs[0] - 1.0], (_zs[:-1] + _zs[1:]) / 2.0, [_zs[-1] + 1.0]])
            _accs = [float(np.mean((_z_hm > c).astype(int) == y_hm)) for c in _cands]
            _accs_inv = [float(np.mean((_z_hm <= c).astype(int) == y_hm)) for c in _cands]
            if max(_accs_inv) > max(_accs):   # orientação da classe 1 em relação a z
                _accs, _sinal = _accs_inv, -1.0
            else:
                _sinal = 1.0
            ELIPSE_CTE = float(_cands[int(np.argmax(_accs))])
            ELIPSE_ACC = float(max(_accs))
            print(f"  Elipse ótima: cte={ELIPSE_CTE:.4f} (classe 1 = {'z > cte' if _sinal > 0 else 'z ≤ cte'}); "
                  f"acurácia (vs rótulo real) da elipse sobre as 100 pessoas = {ELIPSE_ACC:.1%}")

            def _elipse_otima(x1, x2):
                z = np.asarray(x2) ** 2 - np.asarray(x2) + np.asarray(x1) ** 2 - np.asarray(x1)
                return _sinal * (z - ELIPSE_CTE)
            plot_problema_passo1(
                X_hm, y_hm,
                title="Problema G1, Passo 1: peso × altura (valores normalizados), base real",
                feature_names=("peso", "altura"),
                optimal_boundary_fn=_elipse_otima,
                filename=os.path.join(pasta_execucao, "bloco3_graphic_01_G1_passo1_dados.png"),
                class_names=("Mulher", "Homem"),
            )
            print("  Gráfico salvo: bloco3_graphic_01_G1_passo1_dados.png")
        except Exception as exc:
            print(f"  ⚠ Base peso × altura não carregada: {exc}")
            traceback.print_exc()
            falhas_fatais.append(f"Bloco 3 (base peso × altura): {exc}")
            X_hm = None

    if RUN_PROBLEMA_G and X_hm is not None:
        print_section("BLOCO 3 — SUBPROBLEMA G1: PESO × ALTURA (valores normalizados)", "═")

        # Item 17 (reunião 20/05): roda o peso×altura uma vez por par de nomes
        # de classe. Com "Homem"/"Mulher" o LLM pode usar prior semântico; com
        # "A"/"B" ele fica cego ao significado. Comparar as duas isola o efeito
        # do nome da classe (complementa o teste de nomes de feature x1/x2).
        class_name_variants = (
            G_NOMES_CLASSE if RUN_PROBLEMA_G_CLASSES_AB
            else G_NOMES_CLASSE[:1]
        )
        # Comparação entre modelos ("central + caso real"): TODOS os modelos de
        # MODELS_TO_TEST rodam o caso real, em pé de igualdade. Os PNGs de todos
        # vão para a raiz da execução — o alias do modelo no nome do asset
        # (llm_asset) evita sobrescrita entre modelos.
        # `bloco_protegido` por (modelo, variante de nomes de classe) (15/10/2026), como por
        # (modelo, seed) no F: uma exceção numa variante registra a falha (exit 2 no fim), grava o
        # checkpoint e NÃO pula as variantes e os modelos seguintes.
        for ext_provider, ext_model, ext_temp in MODELS_TO_TEST:
            checkpoint_interactions(pasta_execucao)
            MODEL_NAME = ext_model
            CURRENT_PROVIDER = ext_provider
            CURRENT_TEMPERATURE = ext_temp

            for nc0, nc1, pname in class_name_variants:
                with bloco_protegido(f"Bloco 3 G1 (peso × altura) {ext_model} nomes={nc0}/{nc1} ({pname})",
                                     falhas_fatais, pasta_execucao):
                    print_section(
                        f"  → {ext_provider}/{ext_model} | nomes de classe: {nc0}/{nc1} ({pname})", "─")
                    saida_g = fg_executar(
                        problem_name=pname,
                        X=X_hm, y_true=y_hm,
                        nome_classe_0=nc0, nome_classe_1=nc1,
                        feature_variants=G1_NOMES_ATRIBUTO,
                        seeds=RANDOM_SEEDS,
                        ks_aprendiz=K_EXEMPLOS_APRENDIZ,
                        n_train_ratio=0.7,
                        verbose=True,
                        com_passos_3_a_5=True,     # mesmos passos do Problema A (IV.7)
                        atributos_prompt_aprendiz=ATRIBUTOS_PROMPT_PASSO6_G,   # 2, 3 e 4 atributos no prompt (IV.7, 14/10/2026)
                    )
                    for _r in (saida_g['passo2'] + saida_g['aprendiz']
                               + saida_g['passos_2_a_5']):
                        _r['provider'] = ext_provider
                        _r['model'] = ext_model
                    resultados_fg_passos_2_e_4.extend(saida_g['passo2'])
                    linhas_g_passos_2_a_5.extend(saida_g['passos_2_a_5'])
                    resultados_fg_aprendiz.extend(saida_g['aprendiz'])
                    # O scatter ponto-a-ponto (final_graphic_07) é gerado para TODOS os
                    # modelos e TODAS as variantes de classe (I18: a A/B entra; o nome
                    # do arquivo carrega o problem_name): a chave ganha o modelo na frente.
                    mapas_rotulos_fg.update({
                        (ext_model, *k): v
                        for k, v in saida_g['llm_label_maps'].items()
                    })
        print("  ✓ Subproblema G1 (valores normalizados) concluído (todos os modelos).", flush=True)


    if RUN_PROBLEMA_G and X_hm is not None:
        # G2 em blocos próprios: uma exceção em G1 não pula G2 (e vice-versa). A carga da base
        # em kg/m e a figura do Passo 1 ficam num `bloco_protegido` só delas; as coletas rodam
        # em um `bloco_protegido` por (modelo, variante de nomes de classe), como no G1.
        g2_base_ok = False   # só vira True se a carga, a checagem de alinhamento e a figura passarem
        with bloco_protegido("Bloco 3 G2 (base peso × altura em kg/m)", falhas_fatais, pasta_execucao):
            # ── G2: mesmas 100 pessoas e mesma divisão 70/30, prompt em kg e metros ──
            # (e-mail do orientador de 28/08/2026: regra de três 45–110 kg, 1,50–1,95 m;
            # e-mail de 05/10/2026: nomes semânticos peso/altura e homem/mulher.)
            # A métrica continua estimada em X_hm (normalizado); X_hm_fis só vai ao prompt.
            print_section("BLOCO 3 — SUBPROBLEMA G2: PESO × ALTURA EM KG E METROS", "═")
            X_hm_fis, y_hm_fis = create_problem_homem_mulher_kg_m(
                str(BASE_DIR / "dados_reais/homem_mulher/peso_altura_kg_m.csv")
            )
            if X_hm_fis.shape != X_hm.shape or not np.array_equal(y_hm_fis, y_hm):
                raise ValueError("peso_altura_kg_m.csv não está alinhado com peso_altura.csv (ordem/rótulos)")
            print(f"  Base física: peso {X_hm_fis[:, 0].min():.1f}–{X_hm_fis[:, 0].max():.1f} kg | "
                  f"altura {X_hm_fis[:, 1].min():.2f}–{X_hm_fis[:, 1].max():.2f} m")

            # Elipse ótima no espaço físico: desfaz o mapa linear (min/max da base) e
            # reutiliza _elipse_otima do espaço normalizado.
            _lo, _hi = X_hm_fis.min(axis=0), X_hm_fis.max(axis=0)
            _nlo, _nhi = X_hm.min(axis=0), X_hm.max(axis=0)
            def _elipse_otima_fis(kg, m):
                x1 = _nlo[0] + (kg - _lo[0]) * (_nhi[0] - _nlo[0]) / (_hi[0] - _lo[0])
                x2 = _nlo[1] + (m - _lo[1]) * (_nhi[1] - _nlo[1]) / (_hi[1] - _lo[1])
                return _elipse_otima(x1, x2)
            plot_problema_passo1(
                X_hm_fis, y_hm,
                title="Problema G2, Passo 1: peso × altura em kg e m, mesma base (regra de três)",
                feature_names=("peso (kg)", "altura (m)"),
                optimal_boundary_fn=_elipse_otima_fis,
                filename=os.path.join(pasta_execucao, "bloco3_graphic_02_G2_passo1_dados.png"),
                class_names=("Mulher", "Homem"),
            )
            print("  Gráfico salvo: bloco3_graphic_02_G2_passo1_dados.png")
            g2_base_ok = True

        if g2_base_ok:
            class_name_variants_g2 = (
                G2_NOMES_CLASSE if RUN_PROBLEMA_G_CLASSES_AB
                else G2_NOMES_CLASSE[:1]
            )
            for ext_provider, ext_model, ext_temp in MODELS_TO_TEST:
                checkpoint_interactions(pasta_execucao)
                MODEL_NAME = ext_model
                CURRENT_PROVIDER = ext_provider
                CURRENT_TEMPERATURE = ext_temp
                for nc0, nc1, pname in class_name_variants_g2:
                    with bloco_protegido(f"Bloco 3 G2 (peso × altura em kg/m) {ext_model} nomes={nc0}/{nc1} ({pname})",
                                         falhas_fatais, pasta_execucao):
                        print_section(
                            f"  → {ext_provider}/{ext_model} | G2 | nomes de classe: {nc0}/{nc1} ({pname})", "─")
                        saida_g = fg_executar(
                            problem_name=pname,
                            X=X_hm, y_true=y_hm,
                            nome_classe_0=nc0, nome_classe_1=nc1,
                            feature_variants=G2_NOMES_ATRIBUTO,
                            seeds=RANDOM_SEEDS,
                            ks_aprendiz=K_EXEMPLOS_APRENDIZ,
                            n_train_ratio=0.7,
                            verbose=True,
                            X_display=X_hm_fis,
                            com_passos_3_a_5=True,     # mesmos passos do Problema A (IV.7)
                            atributos_prompt_aprendiz=ATRIBUTOS_PROMPT_PASSO6_G,   # 2, 3 e 4 atributos no prompt (IV.7, 14/10/2026)
                        )
                        for _r in (saida_g['passo2'] + saida_g['aprendiz']
                                   + saida_g['passos_2_a_5']):
                            _r['provider'] = ext_provider
                            _r['model'] = ext_model
                        resultados_fg_passos_2_e_4.extend(saida_g['passo2'])
                        linhas_g_passos_2_a_5.extend(saida_g['passos_2_a_5'])
                        resultados_fg_aprendiz.extend(saida_g['aprendiz'])
                        mapas_rotulos_fg.update({   # todas as variantes (I18)
                            (ext_model, *k): v
                            for k, v in saida_g['llm_label_maps'].items()
                        })
            print("  ✓ Subproblema G2 (kg e m) concluído (todos os modelos).", flush=True)

    if RUN_PROBLEMA_F:
        # Cada (modelo, seed) do F roda em `bloco_protegido` (13/10/2026): uma exceção numa semente
        # registra a falha (exit 2 no fim), grava o checkpoint e NÃO pula as sementes e modelos seguintes.
        print_section("BLOCO 2 — PROBLEMA F, PASSOS 1 A 4: MEIA-LUA (LLM como aprendiz; split 70/30)", "═")
        # Todos os modelos rodam a meia-lua (barato, e testa se o colapso
        # zero-shot se repete entre modelos). PNGs de todos na raiz, com o
        # alias do modelo no nome.
        for _ml_idx, (ext_provider, ext_model, ext_temp) in enumerate(MODELS_TO_TEST):
            checkpoint_interactions(pasta_execucao)
            MODEL_NAME = ext_model
            CURRENT_PROVIDER = ext_provider
            CURRENT_TEMPERATURE = ext_temp
            print_section(f"  → Meia-lua com {ext_provider}/{ext_model}", "─")

            for seed in RANDOM_SEEDS:
                with bloco_protegido(f"Bloco 2 (Problema F) {ext_model} seed={seed}", falhas_fatais, pasta_execucao):
                    # Reseta o RNG global por semente (consistente com os demais loops):
                    # garante que qualquer estocasticidade que dependa do estado global
                    # do NumPy seja reprodutível por semente neste bloco.
                    np.random.seed(seed)
                    X_ml, y_ml = create_problem_d_meialua(n_samples=N_PONTOS_A, random_state=seed)
                    print(f"  Meia-lua seed={seed}: shape={X_ml.shape}")

                    # Visualização do problema (dado sintético, independe do
                    # modelo — 1 cópia basta: só no primeiro modelo do loop; com
                    # RUN_PROBLEMA_D o Bloco 1 já gerou este asset)
                    if _ml_idx == 0 and not RUN_PROBLEMA_D:
                        plot_problema_passo1(
                            X_ml, y_ml,
                            title=f"Problema D, Passo 1: meia-lua (sklearn.make_moons, seed={seed})",
                            feature_names=("x1", "x2"),
                            optimal_boundary_fn=None,
                            filename=os.path.join(pasta_execucao, f"bloco1_graphic_12_D_passo1_dados_seed{seed}.png"),
                        )
                        print(f"  Gráfico salvo: bloco1_graphic_12_D_passo1_dados_seed{seed}.png")

                    # Problema F: o ESPECIALISTA do Passo 3 é o rótulo real do make_moons
                    # (expert_labels=y_ml, decisão D1 de 09/10/2026); o Passo 3 roda mesmo
                    # que o zero-shot do Passo 2 colapse.
                    saida_f = fg_executar(
                        problem_name=nome_problema_f(seed),
                        X=X_ml, y_true=y_ml,
                        nome_classe_0="A", nome_classe_1="B",
                        feature_variants=[("x1", "x2")],
                        seeds=[seed],
                        ks_aprendiz=K_EXEMPLOS_APRENDIZ,
                        n_train_ratio=0.7,
                        verbose=True,
                        expert_labels=y_ml,
                        com_passos_3_a_5=False,    # os Passos 3 a 5 (exemplos do rótulo real) são só do G (IV.7)
                    )
                    for _r in saida_f['passo2'] + saida_f['aprendiz']:
                        _r['provider'] = ext_provider
                        _r['model'] = ext_model
                    resultados_fg_passos_2_e_4.extend(saida_f['passo2'])
                    resultados_fg_aprendiz.extend(saida_f['aprendiz'])
                    mapas_rotulos_fg.update({
                        (ext_model, *k): v
                        for k, v in saida_f['llm_label_maps'].items()
                    })

                    # Item 7 (reunião 20/05/2026): superfície SVM gaussiano (RBF) vs LLM
                    # na meia-lua. Usa os rótulos ZERO-SHOT do LLM (chave 'train') — LLM
                    # como especialista no Problema D, logo asset do Bloco 1 (bloco1_graphic_14)
                    # — e a métrica diagonal de maior FIDELIDADE vs LLM na fronteira tracejada (F13).
                    try:
                        lbl = None if RUN_PROBLEMA_D else saida_f['llm_label_maps'].get(
                            (nome_problema_f(seed), seed, 'x1', 'x2', 'train'))  # bloco1_graphic_14 já veio do Bloco 1
                        pa = saida_f['passo2']
                        best_metric = max(pa, key=lambda r: r['fidelity_perc_vs_llm']) if pa else None
                        if lbl is not None:
                            svm_saved = plot_d_passo6_svm(
                                lbl['X'], lbl['y_true'], lbl['y_llm'],
                                metric=best_metric, seed=seed,
                                filename=llm_asset(
                                    pasta_execucao,
                                    f"bloco1_graphic_14_D_passo6_svm_seed{seed}.png",
                                    ext_model,
                                ),
                            )
                            if svm_saved:
                                print(f"  Gráfico salvo: bloco1_graphic_14_D_passo6_svm_seed{seed}__{_model_alias(ext_model)}.png")
                            else:
                                print(f"  ⚠ Plot SVM meia-lua seed={seed} não gerado (classe única no ground truth).")
                    except Exception as exc_svm:
                        print(f"  ⚠ Falha no plot SVM meia-lua seed={seed}: {exc_svm}")
        print("  ✓ Problema F (meia-lua), Passos 1 a 4, concluído (todos os modelos).", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZAÇÕES FINAIS
    # ═══════════════════════════════════════════════════════════════════

    print("\n[ETAPA 5] Gerando visualizações finais...", flush=True)
    metrica_eleita_por_modelo = a_passo6_eleger_metrica(linhas_passos_2_a_5, linhas_bc_passo3)
    print_section("VISUALIZAÇÕES CONSOLIDADAS — comparação entre modelos + oráculo (Bloco 1)", "═")

    # Cada modelo de MODELS_TO_TEST gera o conjunto COMPLETO de visualizações
    # na raiz da execução, com seu alias (MODEL_ALIAS) no nome do asset; os
    # CSVs guardam todos os modelos e o final_graphic_08_comparacao_modelos.png compara todos.
    if resultados_abc and len(MODELS_TO_TEST) > 1:
        plot_final_comparacao_modelos(resultados_abc, filename=os.path.join(pasta_execucao, "final_graphic_08_comparacao_modelos.png"))
        print("  Gráfico salvo: final_graphic_08_comparacao_modelos.png")

    # Validação do oráculo (independe de modelo — nenhuma chamada de LLM)
    # (bloco1_graphic_20 "transferência do oráculo" foi removido em 09/10/2026, I6: não
    # havia transferência, cada expert_name tem um único problema.)
    if resultados_oraculo:
        with bloco_protegido("Visualizações do oráculo", falhas_fatais):
            plot_oraculo_recuperacao_w(resultados_oraculo,
                filename=os.path.join(pasta_execucao, "bloco1_graphic_19_oraculo_passos_1_a_3.png"))
            print("  Gráfico salvo: bloco1_graphic_19_oraculo_passos_1_a_3.png")

    # Problema D, Passo 5: teto da meia-lua sem LLM (origem: item 5 da reunião de 20/05/2026) (acurácia vs rótulo real por n_features)
    if resultados_d_passo5:
        with bloco_protegido("Visualização do teto da meia-lua", falhas_fatais):
            plot_d_passo5_teto(resultados_d_passo5,
                filename=os.path.join(pasta_execucao, "bloco1_graphic_21_D_passo5_teto.png"))
            print("  Gráfico salvo: bloco1_graphic_21_D_passo5_teto.png")

    for _viz_provider, _viz_model, _viz_temp in MODELS_TO_TEST:
        with bloco_protegido(f"Visualizações e análises de {_viz_model}", falhas_fatais):
            abc_m = [r for r in resultados_abc
                     if r.provider == _viz_provider and r.model_name == _viz_model]
            e_m = [r for r in resultados_e
                   if r.provider == _viz_provider and r.model_name == _viz_model]
            # Baselines clássicos independem do LLM (lista única, treinada 1×) —
            # entram na comparação de todos os modelos (são dicts com
            # provider='classical' e model=<nome do classificador>).
            baselines_m = resultados_e_passo6
            dilution_m = [r for r in resultados_e_passo4
                          if r.provider == _viz_provider and r.model_name == _viz_model]
            order_m = [r for r in resultados_e_passo5
                       if r.provider == _viz_provider and r.model_name == _viz_model]
            alternative_m = [r for r in resultados_a_passo4_nnls
                             if r.provider == _viz_provider and r.model_name == _viz_model]
            dados_a_passo2_m = {s: d for (p, m, s), d in dados_a_passo2.items()
                               if (p, m) == (_viz_provider, _viz_model)}
            seed_detail_m = {s: d for (p, m, s), d in seed_detailed_data.items()
                             if (p, m) == (_viz_provider, _viz_model)}
            if not (abc_m or e_m):
                continue  # modelo sem dados nesta execução (ex.: --rapido --modelo X)

            print_section(
                f"VISUALIZAÇÕES — {_viz_provider}/{_viz_model} "
                f"(alias: {_model_alias(_viz_model)})", "═")

            if abc_m:
                print_section("BLOCO 1 — VISUALIZAÇÕES (Problemas A, B e C, LLM como especialista)", "─")
                fname = llm_asset(pasta_execucao, "bloco1_graphic_10_BC_passo3_consistencia.png", _viz_model)
                plot_bc_passo3_consistencia(abc_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")
                fname = llm_asset(pasta_execucao, "bloco1_graphic_16_vies_nomes_classe.png", _viz_model)
                plot_vies_nomes_classe(abc_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

                if len(RANDOM_SEEDS) > 1:
                    fname = llm_asset(pasta_execucao, "bloco1_graphic_07_BC_passo3_sementes.png", _viz_model)
                    plot_bc_passo3_sementes(abc_m, filename=fname)
                    print(f"  Gráfico salvo: {os.path.basename(fname)}")

                # Distribuição de W
                fname = llm_asset(pasta_execucao, "bloco1_graphic_05_A_passo4_w_distribuicao.png", _viz_model)
                plot_a_passo4_w_distribuicao(abc_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

                # Erros da métrica no Problema A (Passo 4)
                for seed_key, data in dados_a_passo2_m.items():
                    fname = llm_asset(pasta_execucao, f"bloco1_graphic_03_A_passo4_erros_seed{seed_key}.png", _viz_model)
                    plot_a_passo4_erros(
                        data['X'], data['y_llm'], data['y_metric'],
                        data['w'], data['centroids'], filename=fname
                    )
                    print(f"  Gráfico salvo: {os.path.basename(fname)}")

                # Análise quantitativa de erros por região (complementa bloco1_graphic_03)
                if dados_a_passo2_m:
                    print_a_passo4_erros_por_margem(dados_a_passo2_m)

                # Análise de sensibilidade dos hiperparâmetros do Perceptron
                if dados_a_passo2_m:
                    print_a_passo4_sensibilidade_perceptron(dados_a_passo2_m)

                # Viés de ordem das classes
                if RUN_VIES_ORDEM_CLASSES:
                    fname = llm_asset(pasta_execucao, "bloco1_graphic_15_vies_ordem_classes.png", _viz_model)
                    plot_vies_ordem_classes(abc_m, filename=fname)
                    print(f"  Gráfico salvo: {os.path.basename(fname)}")

                # Efeito de nomes de features
                if RUN_VIES_NOMES_ATRIBUTO:
                    fname = llm_asset(pasta_execucao, "bloco1_graphic_17_vies_nomes_atributo.png", _viz_model)
                    plot_vies_nomes_atributo(abc_m, filename=fname)
                    print(f"  Gráfico salvo: {os.path.basename(fname)}")

                # Variantes de prompt
                if RUN_VIES_VARIANTES_PROMPT:
                    fname = llm_asset(pasta_execucao, "bloco1_graphic_18_vies_variantes_prompt.png", _viz_model)
                    plot_vies_variantes_prompt(abc_m, filename=fname)
                    print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # ── Passos 2 a 5 (A e D): figuras com símbolos (P&B) + curva vs k ──
            proto_m = {key: d for key, d in rotulos_passos.items()
                       if key[0] == _viz_provider and key[1] == _viz_model}
            if proto_m:
                print_section("BLOCO 1 — PROBLEMAS A E D, PASSOS 2 A 5: figuras com símbolos + curva vs k", "─")
                best_m = metrica_eleita_por_modelo.get(_viz_model)
                _figuras_em_paralelo([
                    (_simbolos_tarefa, (d, _prob, _seed, _viz_model, best_m, bc_zero_shot.get((_p, _m, _seed)), pasta_execucao), {})
                    for (_p, _m, _seed, _prob), d in sorted(proto_m.items(), key=lambda kv: (kv[0][2], kv[0][3]))
                ], rotulo="símbolos A/B/C/D: ")
                rows_m = [r for r in linhas_passos_2_a_5 if r.get('model') == _viz_model]
                if rows_m:
                    for _algo in ("perceptron", "nnls"):
                        fname = llm_asset(pasta_execucao, f"bloco1_graphic_04_AD_passos_2_a_5_curva{'' if _algo == 'perceptron' else '_nnls'}.png", _viz_model)
                        plot_passos_2_a_5_curva(rows_m, filename=fname, algorithm=_algo)
                        print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Não-linearidade implícita: fidelidade R2 (2 pesos) vs R3 (3) vs R4 (4)
            # no Problema A (o CSV r3r4 consolida todos os modelos)
            r3_2feat_viz = [r for r in results_r3_2feat if r.get('model') in (None, _viz_model)]
            r3_3feat_viz = [r for r in results_r3_3feat if r.get('model') in (None, _viz_model)]
            r3_4feat_viz = [r for r in results_r3_4feat if r.get('model') in (None, _viz_model)]
            if r3_2feat_viz and r3_3feat_viz:
                fname = llm_asset(pasta_execucao, "bloco1_graphic_08_A_passo4_R2_R3_R4.png", _viz_model)
                plot_a_passo4_r2_r3_r4(r3_2feat_viz, r3_3feat_viz, r3_4feat_viz or None, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Comparação de algoritmos — Perceptron × NNLS
            if alternative_m and abc_m:
                perc_results = [r for r in abc_m
                               if r.k_passo4_bc == 0 and r.nomes_classes == ("A", "B")
                               and r.prompt_variant == "default" and tuple(r.feature_names) == ("x1", "x2")]
                if perc_results:
                    fname = llm_asset(pasta_execucao, "bloco1_graphic_06_A_passo4_perceptron_vs_nnls.png", _viz_model)
                    plot_a_passo4_perceptron_vs_nnls(perc_results, alternative_m, filename=fname)
                    print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Diagnóstico da busca binária em γ (item b reunião 30/04/2026, ~520s)
            gamma_diag_m = [d for d in PERCEPTRON_GAMMA_DIAGNOSTICS
                            if d.get("model") in (None, _viz_model)]
            if gamma_diag_m:
                fname = llm_asset(pasta_execucao, "bloco1_graphic_09_A_passo4_gamma.png", _viz_model)
                plot_a_passo4_gamma(gamma_diag_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            print_section("BLOCO 2 — VISUALIZAÇÕES (Problema E, LLM como aprendiz)", "─")

            if e_m:
                fname = llm_asset(pasta_execucao, "bloco2_graphic_03_E1_passo3_curva.png", _viz_model)
                plot_e_passo3_curva(e_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")
                fname = llm_asset(pasta_execucao, "bloco2_graphic_04_E1_passo3_estrategias.png", _viz_model)
                plot_e_passo3_estrategias(e_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Baselines clássicos
            if baselines_m and e_m:
                fname = llm_asset(pasta_execucao, "bloco2_graphic_05_E1_passo6_baselines.png", _viz_model)
                plot_e_passo6_baselines(e_m, baselines_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Experimento de diluição (CSV guarda todos os modelos)
            if dilution_m:
                fname = llm_asset(pasta_execucao, "bloco2_graphic_06_E1_passo4_diluicao.png", _viz_model)
                plot_e_passo4_diluicao(dilution_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

            # Viés de ordem dos exemplos few-shot
            if order_m:
                fname = llm_asset(pasta_execucao, "bloco2_graphic_07_E1_passo5_ordem.png", _viz_model)
                plot_e_passo5_ordem(order_m, filename=fname)
                print(f"  Gráfico salvo: {os.path.basename(fname)}")

                # Análise quantitativa do viés de ordem (recency bias)
                print_e_passo5_ordem(order_m)

            # ═══════════════════════════════════════════════════════════════
            # VISUALIZAÇÕES DETALHADAS POR SEED (deste modelo)
            # ═══════════════════════════════════════════════════════════════

            print_section("FECHAMENTO — VISUALIZAÇÕES DETALHADAS POR SEED", "═")
            _figuras_em_paralelo([
                (_fechamento_seed_tarefa, (sdata, seed_key, abc_m, e_m, _viz_model, pasta_execucao), {})
                for seed_key, sdata in seed_detail_m.items()
            ], rotulo="fechamento por seed: ")

            # ═══════════════════════════════════════════════════════════════
            # ANALYSIS (deste modelo; CSVs guardam todos os modelos)
            # ═══════════════════════════════════════════════════════════════

            if abc_m:
                print_abc_resumo(abc_m)
            if e_m:
                print_e_passos_2_3(e_m)

            # Sumário estatístico: média e faixa entre sementes, diferenças pareadas, Cohen's d
            print_sumario_estatistico(abc_m, e_m)

    # ═══════════════════════════════════════════════════════════════════
    # SALVAR RESULTADOS
    # ═══════════════════════════════════════════════════════════════════

    print("\n[ETAPA 6] Salvando resultados em CSV...", flush=True)
    timestamp_csv = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Passos 2 a 5 (A e D) + transferência B/C contra as métricas do A
    # (metrica_eleita_por_modelo já calculada antes das visualizações)
    if linhas_passos_2_a_5:
        fname_proto = os.path.join(pasta_execucao, f"bloco1_ad_passos_2_a_5_{timestamp_csv}.csv")
        pd.DataFrame(linhas_passos_2_a_5).to_csv(fname_proto, index=False)
        print(f"  Passos 2 a 5 dos Problemas A e D salvos em: {fname_proto}")
    if linhas_bc_passo3:
        fname_transf = os.path.join(pasta_execucao, f"bloco1_bc_passo3_consistencia_{timestamp_csv}.csv")
        pd.DataFrame(linhas_bc_passo3).to_csv(fname_transf, index=False)
        print(f"  Transferência B/C contra as métricas do A salva em: {fname_transf}")
    for _m, (_reg, _k, _fid) in metrica_eleita_por_modelo.items():
        print(f"  Melhor métrica do A para {_m}: {_reg} k={_k} (fidelidade média {_fid:.1%})")

    if resultados_abc or resultados_abc_colapsados:
        df_abc = pd.DataFrame([
            {
                'collapsed': getattr(r, 'collapsed', 0),
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'k_passo4_bc': r.k_passo4_bc,
                'class_0': r.nomes_classes[0], 'class_1': r.nomes_classes[1],
                'feature_0': r.feature_names[0], 'feature_1': r.feature_names[1],
                'prompt_variant': r.prompt_variant,
                'rep': r.repeticao, 'fidelity_problem_a': r.fidelidade_problema_a,
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
                'margem_minima': r.margem_minima,   # 15/10/2026 (D): < 0 = Ŵ não separa os rótulos do LLM
                'n_disagreements_b': r.n_disagreements_b, 'n_disagreements_c': r.n_disagreements_c,
                'n_malformed_responses': r.n_malformed_responses,
                'collapsed_b': getattr(r, 'collapsed_b', 0), 'collapsed_c': getattr(r, 'collapsed_c', 0),
                # Ablação (I10): consistência/κ com W_A e os centróides TRANSFERIDOS do A
                'consistency_centroides_a_b': r.consistencia_centroides_a_problema_b,
                'consistency_centroides_a_c': r.consistencia_centroides_a_problema_c,
                'kappa_centroides_a_b': r.kappa_centroides_a_problema_b,
                'kappa_centroides_a_c': r.kappa_centroides_a_problema_c,
            }
            for r in resultados_abc + resultados_abc_colapsados
        ])
        filename_abc = os.path.join(pasta_execucao, f"bloco1_abc_passos_2_a_4_{timestamp_csv}.csv")
        df_abc.to_csv(filename_abc, index=False)
        print(f"  Resultados dos Problemas A, B e C salvos em: {filename_abc}")

    if resultados_e:
        # Linhas zero-shot: UMA coleta por (modelo, seed) reaproveitada em cada estratégia ×
        # especialista (12 linhas), com o mesmo n_malformed_responses repetido; não somar
        # entre essas linhas (ver docstring de ResultadoAprendiz).
        df_e = pd.DataFrame([
            {
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'k': r.k,
                'example_strategy': r.example_strategy,
                'expert_name': r.expert_name,
                'class_0': r.nomes_classes[0], 'class_1': r.nomes_classes[1],
                'rep': r.repeticao,
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
            for r in resultados_e
        ])
        filename_e = os.path.join(pasta_execucao, f"bloco2_e_passos_2_3_{timestamp_csv}.csv")
        df_e.to_csv(filename_e, index=False)
        print(f"  Resultados do Problema E salvos em: {filename_e}")

    if resultados_a_passo4_nnls:
        def _algo_row(r, algo_name):
            return {
                'algorithm': algo_name,
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'k_passo4_bc': r.k_passo4_bc,
                'class_0': r.nomes_classes[0], 'class_1': r.nomes_classes[1],
                'rep': r.repeticao, 'fidelity_problem_a': r.fidelidade_problema_a,
                'consistency_problem_b': r.consistencia_problema_b,
                'kappa_problem_b': r.kappa_problema_b, 'f1_problem_b': r.f1_problema_b,
                'consistency_problem_c': r.consistencia_problema_c,
                'kappa_problem_c': r.kappa_problema_c, 'f1_problem_c': r.f1_problema_c,
                'w_0': r.w_aprendido[0], 'w_1': r.w_aprendido[1],
                'w_ratio': r.w_aprendido[0] / r.w_aprendido[1] if r.w_aprendido[1] != 0 else float('inf'),
                'w_direction_0': r.w_direction[0] if r.w_direction is not None else 0.0,
                'w_direction_1': r.w_direction[1] if r.w_direction is not None else 0.0,
                'gamma': r.gamma_otimo,
                'margem_minima': r.margem_minima,   # 15/10/2026 (D): < 0 = Ŵ não separa os rótulos do LLM
                'n_disagreements_b': r.n_disagreements_b, 'n_disagreements_c': r.n_disagreements_c,
                'llm_accuracy_problem_a': r.acuracia_llm_vs_gt_problema_a,
                'llm_accuracy_problem_b': r.acuracia_llm_vs_gt_problema_b,
                'llm_accuracy_problem_c': r.acuracia_llm_vs_gt_problema_c,
                'metric_accuracy_problem_b': r.acuracia_metrica_vs_gt_problema_b,
                'metric_accuracy_problem_c': r.acuracia_metrica_vs_gt_problema_c,
                'n_malformed_responses': r.n_malformed_responses,
            }
        # Linhas NNLS PAREADAS com os mesmos rótulos do LLM da linha Perceptron (I12).
        alt_rows = [_algo_row(r, 'NNLS') for r in resultados_a_passo4_nnls]
        df_alt = pd.DataFrame(alt_rows)
        filename_alt = os.path.join(pasta_execucao, f"bloco1_a_passo4_perceptron_vs_nnls_{timestamp_csv}.csv")
        df_alt.to_csv(filename_alt, index=False)
        print(f"  Resultados comparação de algoritmos (NNLS pareado, mesmos rótulos do LLM) salvos em: {filename_alt}")

    if resultados_e_passo4:
        df_dilution = pd.DataFrame([
            {
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'k': r.k,
                'example_strategy': r.example_strategy,
                'rep': r.repeticao,
                'accuracy_llm_vs_expert': r.accuracy_llm_vs_expert,
                'kappa_llm_vs_expert': r.kappa_llm_vs_expert,
                'f1_llm_vs_expert': r.f1_llm_vs_expert,
                'accuracy_expert_vs_gt': r.accuracy_expert_vs_gt,
                'accuracy_llm_vs_gt': r.accuracy_llm_vs_gt,
                'n_disagreements': r.n_disagreements,
                'n_total_test': r.n_total_test,
                'n_malformed_responses': r.n_malformed_responses,
                'expert_w_0': r.expert_w[0], 'expert_w_1': r.expert_w[1],
            }
            for r in resultados_e_passo4
        ])
        filename_dil = os.path.join(pasta_execucao, f"bloco2_e_passo4_diluicao_{timestamp_csv}.csv")
        df_dilution.to_csv(filename_dil, index=False)
        print(f"  Resultados da Diluição salvos em: {filename_dil}")

    if results_r3_2feat or results_r3_3feat or results_r3_4feat:
        # Coluna `fidelity_vs_llm` (M17; até 08/10 chamava-se `fidelidade`) e as
        # chaves internas 'fidelity'/'fidelity_nnls' (antes 'accuracy'/'accuracy_nnls').
        r3_rows = []

        def _w4(w):
            w = np.asarray(w, float) if w is not None else np.array([])
            return {f'w{i}': (float(w[i]) if i < len(w) else np.nan) for i in range(4)}

        for n_feat, lista in ((2, results_r3_2feat), (3, results_r3_3feat), (4, results_r3_4feat)):
            for r in lista:
                for algo, fkey, wkey, akey in (('perceptron', 'fidelity', 'w', 'accuracy_vs_true'),
                                              ('nnls', 'fidelity_nnls', 'w_nnls', 'accuracy_vs_true_nnls')):
                    r3_rows.append({
                        'provider': r.get('provider'), 'model': r.get('model'),
                        'seed': r['seed'], 'n_features': n_feat, 'algorithm': algo,
                        'fidelity_vs_llm': r.get(fkey, np.nan),
                        'accuracy_vs_true': r.get(akey, np.nan),
                        # w0, w1 em TODAS as linhas (R2 inclusive); w2/w3 NaN onde não existem.
                        **_w4(r.get(wkey)),
                    })
        df_r3_csv = pd.DataFrame(r3_rows)
        filename_r3 = os.path.join(pasta_execucao, f"bloco1_a_passo4_r2_r3_r4_{timestamp_csv}.csv")
        df_r3_csv.to_csv(filename_r3, index=False)
        print(f"  Resultados R2 vs R3 vs R4 salvos em: {filename_r3}")

    if resultados_e_passo5:
        df_order = pd.DataFrame([
            {
                'provider': r.provider, 'model': r.model_name, 'temperature': r.temperature,
                'random_seed': r.random_seed, 'k': r.k,
                'example_ordering': r.example_strategy.replace("mixed_order_", ""),
                'rep': r.repeticao,
                'accuracy_llm_vs_expert': r.accuracy_llm_vs_expert,
                'kappa_llm_vs_expert': r.kappa_llm_vs_expert,
                'f1_llm_vs_expert': r.f1_llm_vs_expert,
                'accuracy_expert_vs_gt': r.accuracy_expert_vs_gt,
                'accuracy_llm_vs_gt': r.accuracy_llm_vs_gt,
                'n_disagreements': r.n_disagreements,
                'n_total_test': r.n_total_test,
                'n_malformed_responses': r.n_malformed_responses,
            }
            for r in resultados_e_passo5
        ])
        filename_order = os.path.join(pasta_execucao, f"bloco2_e_passo5_ordem_{timestamp_csv}.csv")
        df_order.to_csv(filename_order, index=False)
        print(f"  Resultados do Viés de Ordem salvos em: {filename_order}")

    if resultados_e_passo6:
        df_baselines = pd.DataFrame(resultados_e_passo6)
        filename_bl = os.path.join(pasta_execucao, f"bloco2_e_passo6_baselines_{timestamp_csv}.csv")
        df_baselines.to_csv(filename_bl, index=False)
        print(f"  Resultados dos Baselines Clássicos salvos em: {filename_bl}")

    if resultados_oraculo:
        df_oraculo = pd.DataFrame(resultados_oraculo)
        filename_oraculo = os.path.join(pasta_execucao, f"bloco1_oraculo_passos_1_a_3_{timestamp_csv}.csv")
        df_oraculo.to_csv(filename_oraculo, index=False)
        print(f"  Resultados da Validação do Oráculo salvos em: {filename_oraculo}")

    if resultados_d_passo6_svm:  # Problema D, Passo 6: números do SVM RBF (16/10/2026)
        filename_d_passo6 = os.path.join(pasta_execucao, f"bloco1_d_passo6_svm_{timestamp_csv}.csv")
        pd.DataFrame(resultados_d_passo6_svm).to_csv(filename_d_passo6, index=False)
        print(f"  Problema D, Passo 6 (SVM) salvo em: {filename_d_passo6}")
    if resultados_d_passo5:  # Problema D, Passo 5: teto da meia-lua sem LLM (origem: item 5 da reunião de 20/05/2026)
        df_d_passo5 = pd.DataFrame(resultados_d_passo5)
        filename_d_passo5 = os.path.join(pasta_execucao, f"bloco1_d_passo5_teto_{timestamp_csv}.csv")
        df_d_passo5.to_csv(filename_d_passo5, index=False)
        print(f"  Problema D, Passo 5 (teto) salvo em: {filename_d_passo5}")

    # ─── Problemas externos não-lineares (peso×altura, meia-lua) ──────────
    # CSVs guardam TODOS os modelos; plots/sínteses são gerados POR MODELO

    if resultados_fg_passos_2_e_4:
        rows_a = []

        def _w_comp(w, i):
            """Componente i do vetor W (ou None se ausente)."""
            try:
                return float(w[i]) if w is not None and len(w) > i else None
            except (TypeError, IndexError):
                return None

        for r in resultados_fg_passos_2_e_4:
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
        fname_ext_a = os.path.join(pasta_execucao, f"bloco23_fg_passos_2_e_4_{timestamp_csv}.csv")
        df_ext_a.to_csv(fname_ext_a, index=False)
        print(f"  Zero-shot e métrica dos Problemas F (Passo 2) e G (Passos 2 e 4) salvos em: {fname_ext_a}")

    if linhas_g_passos_2_a_5:
        fname_proto_g = os.path.join(pasta_execucao, f"bloco3_g_passos_2_a_5_{timestamp_csv}.csv")
        pd.DataFrame(linhas_g_passos_2_a_5).to_csv(fname_proto_g, index=False)
        print(f"  Problema G, Passos 2 a 5 (G1/G2) salvos em: {fname_proto_g}")

    if resultados_fg_aprendiz:
        df_ext_e = pd.DataFrame(resultados_fg_aprendiz)
        # converte tupla feature_names para string
        df_ext_e['feature_names'] = df_ext_e['feature_names'].apply(lambda t: '/'.join(t) if isinstance(t, tuple) else t)
        fname_ext_e = os.path.join(pasta_execucao, f"bloco23_fg_aprendiz_{timestamp_csv}.csv")
        df_ext_e.to_csv(fname_ext_e, index=False)
        print(f"  Aprendizado em contexto dos Problemas F (Passo 3) e G (Passo 6) salvo em: {fname_ext_e}")

    # Tabela cruzada linear × não-linear — uma por modelo (alias no nome)
    for _ext_provider, _ext_model, _ext_temp in MODELS_TO_TEST:
        r3_2feat_m = [r for r in results_r3_2feat if r.get('model') in (None, _ext_model)]
        r3_3feat_m = [r for r in results_r3_3feat if r.get('model') in (None, _ext_model)]
        r3_4feat_m = [r for r in results_r3_4feat if r.get('model') in (None, _ext_model)]
        ext_a_m = [
            r for r in resultados_fg_passos_2_e_4
            if r.get('provider') in (None, _ext_provider)
            and r.get('model') in (None, _ext_model)
        ]
        proto_m_rows = [r for r in linhas_passos_2_a_5 if r.get('model') == _ext_model]
        if not (r3_3feat_m or ext_a_m or proto_m_rows):
            continue
        cross_fname = sintese_r2_r3_r4(
            results_abc_r3=r3_3feat_m,
            resultados_fg_passos_2_e_4=ext_a_m,
            pasta_execucao=pasta_execucao,
            results_abc_r4=r3_4feat_m if r3_4feat_m else None,
            model_name=_ext_model,
            results_abc_r2=r3_2feat_m if r3_2feat_m else None,
            linhas_passos_2_a_5=proto_m_rows or None,   # linhas zero-shot do Problema D (I8)
        )
        if cross_fname:
            print(f"  Comparação cruzada linear×não-linear salva em: {cross_fname}")

    # Visualização ponto-a-ponto das rotulações do LLM (item 7 da reunião
    # 30/04/2026, e-mail 22:06) → final_graphic_07_FG_rotulos_llm_*
    # A chave carrega o modelo e o problema — um scatter por
    # (modelo, problema, seed, features, kind).
    if mapas_rotulos_fg:
        print("\n  Gerando scatter ponto-a-ponto das rotulações do LLM...")
        _tarefas_07 = []
        for key, data in mapas_rotulos_fg.items():
            model_lbl, prob_name, seed_val, feat_0, feat_1, kind = key
            # Passo do problema: zero-shot no treino (kind='train') = Passo 2; k exemplos no teste
            # (kind=k, inclusive k=0) = Passo 3 (F) / Passo 6 (G). O sufixo do arquivo diz o passo:
            # `_passo2` ou `_passo{3|6}_k{k}`.
            if isinstance(kind, int):
                _passo_rotulos = 3 if prob_name.startswith("F") else 6
                kind_label, kind_suf = f"k={kind}", f"passo{_passo_rotulos}_k{kind}"
            else:
                _passo_rotulos, kind_label, kind_suf = 2, "zero-shot (treino)", "passo2"
            # Nome do problema no arquivo desambigua problemas com os mesmos
            # nomes de atributo; o sufixo _seed{N} do F_meia_lua_seed{N} é
            # removido porque o seed já aparece como componente próprio.
            prob_slug = prob_name.replace(f"_seed{seed_val}", "")
            fname_lbl = llm_asset(
                pasta_execucao,
                f"final_graphic_07_FG_rotulos_llm_{prob_slug}_seed{seed_val}_{feat_0}_{feat_1}_{kind_suf}.png",
                model_lbl,
            )
            # Nomes de classe nas legendas: Mulher/Homem nas variantes semânticas do G,
            # A/B nas neutras e no F; y_metric = rótulos do especialista do aprendizado em contexto.
            _cn = ("Mulher", "Homem") if (prob_name.startswith("G") and "classesAB" not in prob_name) else ("A", "B")
            _tarefas_07.append((plot_fg_rotulos_llm, (), dict(
                X=data.get('X_display', data['X']), y_llm=data['y_llm'], y_true=data.get('y_true'),
                title=(f"Rotulação LLM ({_model_alias(model_lbl)}): Problema {_rotulo_fg(prob_slug)}, "
                       f"Passo {_passo_rotulos} | seed={seed_val} | {feat_0}/{feat_1} | {kind_label}"),
                feature_names=(feat_0, feat_1),
                filename=fname_lbl,
                y_metric=data.get('y_metric'),
                class_names=_cn,
            )))
        _figuras_em_paralelo(_tarefas_07, rotulo="final_graphic_07: ")   # 15/10/2026: em processos
        print(f"  ✓ Scatter ponto-a-ponto gerados em {pasta_execucao}/final_graphic_07_FG_rotulos_llm_*.png")

    # ─── Plots adicionais para apresentação (problemas externos) ──────────
    # Gerados POR MODELO, com o alias no nome do asset.
    for _ext_provider, _ext_model, _ext_temp in MODELS_TO_TEST:
        ext_e_m = [
            r for r in resultados_fg_aprendiz
            if r.get('provider') in (None, _ext_provider)
            and r.get('model') in (None, _ext_model)
        ]
        ext_a_m = [
            r for r in resultados_fg_passos_2_e_4
            if r.get('provider') in (None, _ext_provider)
            and r.get('model') in (None, _ext_model)
        ]
        if ext_e_m:
            try:
                plot_fg_aprendiz_curva(
                    ext_e_m,
                    filename=llm_asset(pasta_execucao, "bloco23_graphic_01_FG_aprendiz_curva.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_graphic_01_FG_aprendiz_curva__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_graphic_01_FG_aprendiz_curva ({_ext_model}): {exc}")

            try:
                plot_fg_llm_vs_perceptron(
                    ext_e_m,
                    filename=llm_asset(pasta_execucao, "bloco23_graphic_04_FG_llm_vs_perceptron.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_graphic_04_FG_llm_vs_perceptron__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_graphic_04_FG_llm_vs_perceptron ({_ext_model}): {exc}")

        if ext_a_m:
            try:
                plot_fg_passos_2_e_4_r2_r3_r4(
                    ext_a_m,
                    filename=llm_asset(pasta_execucao, "bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4 ({_ext_model}): {exc}")

            try:
                plot_fg_fronteiras(
                    ext_a_m,
                    filename=llm_asset(pasta_execucao, "bloco23_graphic_03_FG_fronteiras.png", _ext_model),
                )
                print(f"  Gráfico salvo: bloco23_graphic_03_FG_fronteiras__{_model_alias(_ext_model)}.png")
            except Exception as exc:
                print(f"  ⚠ Falha em bloco23_graphic_03_FG_fronteiras ({_ext_model}): {exc}")

        # Bloco 3: curva dos Passos 2 a 5 do Problema G (zero-shot → few-shot → teto), R²,
        # um painel por (subproblema, nomes de classe, nomes de atributo); variante _nnls.
        proto_g_m = [r for r in linhas_g_passos_2_a_5 if r.get('model') == _ext_model]
        if proto_g_m:
            for _algo in ("perceptron", "nnls"):
                _fname = llm_asset(pasta_execucao,
                                   f"bloco3_graphic_03_G_passos_2_a_5_curva{'' if _algo == 'perceptron' else '_nnls'}.png",
                                   _ext_model)
                try:
                    plot_g_passos_2_a_5_curva(proto_g_m, filename=_fname, algorithm=_algo)
                    print(f"  Gráfico salvo: {os.path.basename(_fname)}")
                except Exception as exc:
                    print(f"  ⚠ Falha em {os.path.basename(_fname)}: {exc}")

    if resultados_fg_passos_2_e_4:
        # ─── Resumo consolidado no log (auxilia roteiro da apresentação) ──────
        print_section("BLOCOS 2/3 — RESUMO CONSOLIDADO: zero-shot e métrica (F, Passo 2; G1/G2, Passos 2 e 4)", "═")
        df_ext = pd.DataFrame(resultados_fg_passos_2_e_4)
        df_ext['variant'] = df_ext['feature_names'].apply(
            lambda t: '/'.join(t) if isinstance(t, tuple) else str(t)
        )
        # Agrupa pelo nome-base do problema: F_meia_lua_seed{N} → F_meia_lua, para
        # que a média±desvio agregue os 3 seeds (cada seed da meia-lua é um
        # sequência própria porque o dataset sintético é regenerado por seed).
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

    if resultados_fg_aprendiz:
        print_section("BLOCOS 2/3 — RESUMO CONSOLIDADO: aprendizado em contexto (F, Passos 3 e 4; G1/G2, Passos 6 e 7; LLM vs Perceptron)", "═")
        df_ext_e_sum = pd.DataFrame(resultados_fg_aprendiz)
        df_ext_e_sum['variant'] = df_ext_e_sum['feature_names'].apply(
            lambda t: '/'.join(t) if isinstance(t, tuple) else str(t)
        )
        df_ext_e_sum['problem_base'] = df_ext_e_sum['problem_name'].str.replace(r'_seed\d+$', '', regex=True)
        for problem in sorted(df_ext_e_sum['problem_base'].unique()):
            sub_all = df_ext_e_sum[df_ext_e_sum['problem_base'] == problem]
            for model_sum in sorted(sub_all['model'].unique()):
                sub = sub_all[sub_all['model'] == model_sum]
                for variant in sorted(sub['variant'].unique()):
                  for n_feat_p in sorted(sub[sub['variant'] == variant]['n_features'].unique()):
                    sv = sub[(sub['variant'] == variant) & (sub['n_features'] == n_feat_p)]
                    print(f"\n  {problem} | {model_sum} | {variant} | {int(n_feat_p)} atributos no prompt:")
                    for nshot in sorted(sv['k'].unique()):
                        ss = sv[sv['k'] == nshot]
                        llm_real = ss['accuracy_llm_vs_true'].mean()
                        pb = ss['accuracy_perceptron_baseline_vs_true'].dropna()
                        if len(pb):
                            delta = llm_real - pb.mean()
                            marker = "LLM>Perceptron" if delta > 0.01 else (
                                "Perceptron>LLM" if delta < -0.01 else "≈ empate"
                            )
                            print(
                                f"    k={nshot}: llm_real={llm_real:.1%} | "
                                f"perceptron_real={pb.mean():.1%} | Δ={delta:+.1%} ({marker})"
                            )
                        else:
                            print(f"    k={nshot}: llm_real={llm_real:.1%}")

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
    print("\n  ── Auditoria do parser ──")
    print(f"     Chamadas ao LLM: {n_calls_total} | Fallback (hash) acionado: "
          f"{n_fallback} ({taxa_fallback:.2f}%)")
    if n_fallback == 0:
        print("     ✓ Nenhum fallback: métricas (κ/consistência) não contêm rótulos pseudo-aleatórios.")
    elif taxa_fallback <= 5.0:
        print("     ✓ Taxa ≤ 5%: impacto do fallback nas métricas é marginal.")
    else:
        print("     ⚠️ Taxa > 5%: rótulos de fallback podem poluir κ/consistência — "
              "reportar métricas com/sem fallback.")

    # ─── Passo 6b: tabelas (.tex em tabelas/ e .png na raiz), 0 chamadas ──────────
    # Catálogo completo de `gerar_tabelas.py` sobre os CSVs recém-salvos; a auditoria
    # usa as interações em memória (mesmo conteúdo dos llm_interactions*.json).
    print("\n[ETAPA 6b] Gerando as tabelas da execução (.tex + .png)...", flush=True)
    try:
        from audit_interactions import audit as _audit_fn
        from gerar_tabelas import gerar_tudo as _gerar_tabelas
        _auditoria = _audit_fn(LLM_INTERACTIONS) if LLM_INTERACTIONS else None
        _gerar_tabelas(pasta_execucao, audit=_auditoria)
    except Exception as _exc:   # tabelas são derivadas: nunca derrubam a execução
        print(f"  ⚠️ Tabelas não geradas ({type(_exc).__name__}: {_exc}); "
              f"rode `python src/gerar_tabelas.py {os.path.basename(pasta_execucao)}` depois.")

    if _REESTIMACAO_RELATORIO is not None:
        _REESTIMACAO_RELATORIO(pasta_execucao)   # lacunas da reestimação offline, antes de persistir o log

    if falhas_fatais:
        # Registrado ANTES de persistir o log, para constar em log_execucao.txt.
        print(f"\n{'!'*70}\n EXECUÇÃO INCOMPLETA — blocos que falharam por exceção:")
        for _f in falhas_fatais:
            print(f"   - {_f}")
        print(" A pasta NÃO deve ser usada como fonte do trabalho. Código de saída 2.")
        print(f"{'!'*70}")

    # Veredito, hora de fim e duração ANTES de gravar o log (14/10/2026: até então o log
    # terminava em "[ETAPA 7]" e não registrava se a execução foi concluída).
    _fim_execucao = datetime.now()
    print(f"\n EXPERIMENTO {'INCOMPLETO (código de saída 2)' if falhas_fatais else 'CONCLUÍDO (código de saída 0)'}: "
          f"início {_inicio_execucao:%Y-%m-%d %H:%M:%S}, fim {_fim_execucao:%Y-%m-%d %H:%M:%S}, "
          f"duração {str(_fim_execucao - _inicio_execucao).split('.')[0]}")

    # ─── Etapa 7: Salvar log TXT (fatiado em partes se passar de 10 MiB) ──────────
    print("\n[ETAPA 7] Salvando log completo da execução...")
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
    print(f" EXPERIMENTO {'INCOMPLETO (ver blocos que falharam acima)' if falhas_fatais else 'CONCLUÍDO!'}")
    print(f" Pasta de saída: {pasta_execucao}/")
    print(f" Total de arquivos gerados: {n_arquivos}")
    print("   - Imagens PNG: gráficos de todos os problemas e passos")
    print("   - CSVs: resultados numéricos detalhados")
    print(f"   - Log TXT: {log_path}")
    print(f"{'='*70}\n")

    if falhas_fatais:
        raise SystemExit(2)   # pasta gravada (log, CSVs parciais), mas execução incompleta


if __name__ == "__main__":
    main()