# CLAUDE.md — Projeto de Mestrado: Explicando Decisões de LLMs via Otimização Inversa

## Visão Geral

Pesquisa que investiga se LLMs possuem um processo de tomada de decisão **estável e aprendível**.
A hipótese central: se um LLM tem critérios de decisão consistentes, é possível aprender
uma métrica matemática (W) a partir de suas classificações e aplicá-la para prever
classificações em problemas diferentes (transferência de conhecimento).

**Abordagem:** Otimização inversa — aprender a função objetivo implícita do LLM a partir
de suas decisões observadas, usando métricas de Mahalanobis diagonais e dois algoritmos
de estimação (Perceptron Estruturado, NNLS).

**Organização em 3 blocos auto-contidos:**
- **Bloco 1 — LLM como FONTE:** Problemas A (linear), B (linear, rotação horária ±1.5), C (linear, rotação anti-horária ±1.5), D (meia-lua não-linear). LLM rotula; aprendemos W via Perceptron+NNLS. Inclui Oracle, R2→R3→R4, vieses e variantes de prompt.
- **Bloco 2 — LLM como APRENDIZ:** Problemas E (linear, perito W=[0.3,1.5]) e F (meia-lua, perito = ground truth). Fase E in-context; estratégias, refutação H4, baselines clássicos.
- **Bloco 3 — Estudo de caso REAL:** peso × altura (base do orientador, fronteira elíptica). Fase A R2/R3/R4 + Fase E + paradoxo do overfitting.

## Estrutura de Pastas

```
mestrado/
├── src/
│   ├── dissertacao_mestrado.py       # Script principal (~7500+ linhas)
│   ├── relaxed_perceptron.py         # Perceptron Estruturado com Relaxação de Margem
│   ├── least_squares_inverse.py      # Mínimos Quadrados Não-Negativos (NNLS)
│   └── arquivado/
│       └── max_margin_lp_inverse.py  # ARQUIVADO — LP Max-Margin removido (plano_trabalho item 12)
│   └── classical_baselines.py        # Baselines clássicos (k-NN, LR, SVM)
├── requirements.txt              # Dependências Python
├── CLAUDE.md                     # Este arquivo
├── .env                          # Chaves de API (carregadas via python-dotenv)
├── .venv/                        # Ambiente virtual Python
├── .claude/
│   └── settings.local.json       # Configurações locais do Claude Code
├── reunioes_orientador/           # Transcrições e planos de reuniões com orientador
├── artigos_referenciados/         # PDFs de artigos citados
├── trabalhos_referencias.txt      # Lista consolidada de referências bibliográficas
└── execucao_YYYY-MM-DD_HH-MM-SS/ # Pasta criada a cada execução (timestamp automático)
    ├── dados_sinteticos_seed{seed}/       # CSVs dos datasets por seed (A, B, C, D, A_r3)
    # === BLOCO 1 — LLM como FONTE (otim. inversa) ===
    ├── bloco1_01_problemas_lineares.png       # + painéis _problema_{a,b,c}.png
    ├── bloco1_02_problema_d_meialua_seed*_overview.png
    ├── bloco1_03_oracle_w_recovery.png        # Oracle: recuperação de W conhecido
    ├── bloco1_04_oracle_transfer.png          # Oracle: transferência entre geometrias
    ├── bloco1_05_fase_a_errors_seed*.png      # por seed (mapa de erros)
    ├── bloco1_06_w_distribution.png           # + boxplot, ratio, scatter
    ├── bloco1_07_algorithm_comparison.png     # Perceptron × NNLS
    ├── bloco1_08_seed_comparison.png
    ├── bloco1_09_consistency_extended.png     # + painéis por métrica
    ├── bloco1_10_r3r4_comparison.png          # R2/R3/R4 em lineares + meia-lua
    ├── bloco1_12_class_order_bias.png
    ├── bloco1_13_prompt_variants.png
    ├── bloco1_14a_class_names_effect.png
    ├── bloco1_14b_feature_names_effect.png
    # === BLOCO 2 — LLM como APRENDIZ (Fase E) ===
    ├── bloco2_01_problema_e_expert.png        # + painéis fronteira/ground_truth/margem
    ├── bloco2_03_phase_e_strategies.png       # + painéis por estratégia
    ├── bloco2_04_phase_e_learning_curve.png   # + painéis por métrica
    ├── bloco2_05_phase_e_strategy_comparison.png  # + painéis por n_shot
    ├── bloco2_06_dilution.png
    ├── bloco2_07_example_order.png            # recency bias
    ├── bloco2_09_classical_baselines.png      # k-NN, LR, SVM × LLM
    # === BLOCO 3 — Estudo de caso REAL ===
    ├── bloco3_01_peso_altura_overview.png
    # === BLOCO 2/3 — Pipelines externos (meia-lua + peso×altura) ===
    ├── bloco23_external_learning_curve.png
    ├── bloco23_external_features_comparison.png
    ├── bloco23_external_decision_boundary.png
    ├── bloco23_external_llm_vs_perceptron.png
    # === FECHAMENTO (visualizações por seed) ===
    ├── final_02_confusion_matrices_seed*.png
    ├── final_03_dashboard_seed*.png
    ├── final_04_dataset_overview_seed*.png
    ├── final_05_hits_errors_seed*.png
    ├── final_06_w_algorithms_seed*.png
    ├── final_07_margin_analysis_seed*.png
    ├── final_08_llm_labels_*.png              # scatter LLM-labels por config
    ├── final_09_model_comparison.png
    # === CSVs por bloco ===
    ├── bloco1_phases_abc_{timestamp}.csv
    ├── bloco1_oracle_validation_{timestamp}.csv
    ├── bloco1_algorithm_comparison_{timestamp}.csv
    ├── bloco1_r3r4_comparison_{timestamp}.csv
    ├── bloco2_phase_e_{timestamp}.csv
    ├── bloco2_dilution_{timestamp}.csv
    ├── bloco2_example_order_{timestamp}.csv
    ├── bloco2_classical_baselines_{timestamp}.csv
    ├── bloco23_external_phase_a_{timestamp}.csv
    ├── bloco23_external_phase_e_{timestamp}.csv
    ├── final_cross_linearity.csv
    ├── llm_interactions.json      # Log completo de prompts e respostas do LLM
    └── log_execucao.txt
```

## Regras para Apresentações

**Roteiro base:** Sempre que for gerar uma apresentação LaTeX Beamer, usar o arquivo
`roteiro_apresentacao.txt` (na raiz do projeto) como **índice mestre** da ordem dos slides
e do que cada um deve conter. O arquivo lista, slide a slide: título, tipo de conteúdo
(texto / tabela / gráfico / fórmula) e uma breve descrição do que deve aparecer.
Os números concretos (métricas, tabelas de resultados) devem ser extraídos dos CSVs e do
`log_execucao.txt` da execução mais recente.

Quando o usuário pedir para criar uma apresentação LaTeX Beamer com resultados de uma execução,
o arquivo `.tex` deve ser criado **dentro da pasta da execução correspondente**
(ex: `execucao_YYYY-MM-DD_HH-MM-SS/apresentacao/apresentacao.tex`), junto com os gráficos e CSVs.
Isso garante que a apresentação e seus assets fiquem co-localizados e o `\graphicspath` não seja necessário.

**Preâmbulo LaTeX padrão:** Toda apresentação Beamer (principal e guia) deve usar **exatamente** este preâmbulo,
sem adicionar ou remover pacotes/opções salvo necessidade explícita do usuário:

```latex
\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{siunitx}
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{caption}[numbered]
\setbeamerfont{caption}{size=\scriptsize}
\setbeamerfont{frametitle}{size=\large}
\graphicspath{{../}}
```

Padroniza visual (tema Madrid, aspect ratio 16:9, fonte 10pt), sem símbolos de navegação, com `\graphicspath{{../}}`
apontando para a pasta da execução onde os PNGs estão salvos.

**Formatação dos slides:** O conteúdo de cada slide deve **sempre caber dentro do espaço visível** do frame.
Se houver muito conteúdo para um único slide, **quebrar em 2 ou mais slides** (ex: "Resultados (1/2)", "Resultados (2/2)").
Prestar atenção especial à formatação do LaTeX: tamanhos de fonte, espaçamentos, margens e escala de figuras
devem ser ajustados para que nada ultrapasse os limites do slide.

**Nível de detalhe:** A apresentação é destinada a uma banca de mestrado, portanto deve conter
o **máximo de detalhes possível** — metodologia, configurações, métricas, análises e conclusões
devem ser apresentados de forma completa e rigorosa.

**Apresentação-guia:** Sempre criar, além da apresentação principal, um arquivo
`apresentacao_guia.tex` (na mesma pasta da execução) que funciona como **roteiro de fala**.
Regras da apresentação-guia:
- Deve ter **exatamente a mesma quantidade de slides** da apresentação principal.
- Cada slide contém **apenas texto** descrevendo o que o apresentador deve falar naquele momento — sem imagens ou figuras.
- O texto deve ser escrito com **naturalidade e fluidez**, como se uma pessoa estivesse explicando oralmente, para que não pareça uma leitura mecânica.
- Deve conter **muito detalhe** sobre como o experimento foi realizado, justificativas das escolhas e interpretação dos resultados, pois o professor precisa dessas informações.

## Como Executar

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar chaves de API em um arquivo .env (carregado via python-dotenv)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...   (opcional — só se ativar Claude em MODELS_TO_TEST)
# GOOGLE_API_KEY=...             (opcional — só se ativar Gemini)

# Executar o experimento completo
python src/dissertacao_mestrado.py
```

A execução cria automaticamente uma pasta `execucao_YYYY-MM-DD_HH-MM-SS/` com todos os outputs.
As chamadas à API do LLM são **assíncronas com concorrência limitada**
(`asyncio.Semaphore(MAX_CONCURRENCY)`), acelerando substancialmente a coleta de decisões.

## Configuração Principal

Constantes no topo de `src/dissertacao_mestrado.py`:

| Constante | Valor padrão | Descrição |
|---|---|---|
| `RANDOM_SEED` | `42` | Semente padrão |
| `RANDOM_SEEDS` | `[42, 123, 7]` | 3 seeds para robustez estatística |
| `N_SAMPLES_PROBLEM_A` | `150` | Amostras no Problema A |
| `N_SAMPLES_PROBLEM_B/C` | `100` | Amostras nos Problemas B e C |
| `N_SAMPLES_PROBLEM_D` | `150` | Amostras no Problema D |
| `FEW_SHOT_SIZES` | `[0, 5, 10, 20, 40]` | Tamanhos few-shot fases B e C |
| `FEW_SHOT_SIZES_PHASE_E` | `[0, 5, 10, 20, 40]` | Tamanhos few-shot fase D |
| `N_REPETICOES` | `3` | Repetições por configuração |
| `MAX_CONCURRENCY` | `10` | Chamadas paralelas à API do LLM |
| `MAX_FORMAT_RETRIES` | `5` | Reenvios para respostas malformadas |
| `EXAMPLE_STRATEGIES` | `["easy","hard","mixed","random"]` | Estratégias fase D |
| `EXPERT_W` | `[0.3, 1.5]` | Pesos do especialista padrão (x2 dominante) |
| `EXPERT_CENTROIDS` | `[[-1.5, 1.0], [1.5, -1.0]]` | Centróides do especialista (Fase E) |

### Flags de Execução

Controlam quais experimentos rodar sem precisar comentar/descomentar código:

| Flag | Padrão | Descrição |
|---|---|---|
| `RUN_PHASES_ABC` | `True` | Fases A-C padrão |
| `RUN_PHASE_E` | `True` | Fase E padrão |
| `RUN_CLASS_ORDER_BIAS` | `True` | Teste de inversão de ordem das classes |
| `RUN_FEATURE_NAMES` | `True` | Teste de nomes semânticos nas features |
| `RUN_DILUTION` | `True` | Experimento de diluição |
| `RUN_R3R4_EXPERIMENT` | `True` | Augmentação R3 (x1·x2) e R4 (x1², x2²) p/ detectar não-linearidade |
| `RUN_MULTIPLE_EXPERTS` | `True` | Múltiplas configs de expert na Fase E |
| `RUN_ALGORITHM_COMPARISON` | `True` | Comparação dos 3 algoritmos de otim. inversa |
| `RUN_ORACLE_VALIDATION` | `True` | **NOVO:** Validação: algoritmos recuperam W conhecido? |
| `RUN_EXAMPLE_ORDER_BIAS` | `True` | Teste de viés de ordem dos exemplos few-shot |
| `RUN_PROMPT_VARIANTS` | `True` | Teste de múltiplas variantes de prompt |
| `RUN_CLASSICAL_BASELINES` | `True` | Comparação com baselines clássicos (k-NN, LR, SVM) |
| `RUN_PROBLEM_E` | `True` | **NOVO:** Problema E (meia-lua) — não-linearidade explícita |
| `RUN_HOMEM_MULHER` | `True` | **NOVO:** Estudo de caso real peso×altura (homem/mulher, elipse) |

### Configurações de Experts Múltiplos (Fase E)

```python
EXPERT_CONFIGS = [
    {"name": "aniso_x2",  "w": [0.3, 1.5],  "desc": "x2 dominante (original)"},
    {"name": "aniso_x1",  "w": [1.5, 0.3],  "desc": "x1 dominante (invertido)"},
    {"name": "euclidean", "w": [1.0, 1.0],  "desc": "pesos iguais (Euclidiana)"},
]
```

### Configurações Anisotrópicas para Oracle Validation

```python
ORACLE_ANISO_CONFIGS = [
    {"name": "x2_dom",       "w": [0.3, 1.5], "desc": "x2 dominante"},
    {"name": "x1_dom",       "w": [1.5, 0.3], "desc": "x1 dominante"},
    {"name": "forte_aniso",  "w": [0.1, 2.0], "desc": "anisotropia extrema"},
]
```

### Nomes de Features Semânticos

```python
NOMES_FEATURES = [
    ("x1", "x2"),                    # Neutro (padrão)
    ("altura", "peso"),              # Semântico
    ("feature_1", "feature_2"),      # Técnico
]
```

### Variantes de Prompt

```python
PROMPT_VARIANTS = {
    "default":   { ... },  # Template atual (baseline)
    "geometric": { ... },  # Contexto geométrico/espacial explícito
    "cot":       { ... },  # Chain-of-thought (raciocínio passo a passo)
    "tabular":   { ... },  # Coordenadas apresentadas como tabela markdown
}
```

**Modelos suportados** (alternar em `MODELS_TO_TEST`):
- `gpt-4o-mini` (OpenAI — ativo por padrão)
- Claude (Anthropic — SDK já integrado, chamada comentada)
- Gemini 2.0 Flash (Google — comentado, pronto para ativar)

## Arquitetura do Código

### Estruturas de Dados Principais

```python
@dataclass
class ResultadoExperimento        # Resultado das fases A-C
    # consistencia, kappa, f1 para Problemas B e C
    # pesos aprendidos (w, w_nnls, w_lp), gamma (perceptron e LP)
    # fidelidades (perceptron, NNLS, LP) no Problema A
    # consistencia_{nnls,lp}_problema_b/c, kappa_{nnls,lp}_problema_b/c
    # w_cosine_sim_{nnls,lp,nnls_lp} (similaridade entre algoritmos)
    # w_ratio, w_direction, feature_names, prompt_variant

@dataclass
class ResultadoPhaseDExperimento  # Resultado da fase D
    # acurácia LLM vs. especialista, kappa, f1
    # estratégia de exemplos, ordem (para recency bias)
    # expert_name, baseline classicas opcionais

@dataclass
class LearnedMetric               # Métrica Mahalanobis aprendida
    # vetor w, centroides, gamma, problema de origem
```

### Funções Críticas

| Função | Propósito |
|---|---|
| `create_problem_{a,b,c,d}()` | Gera dados sintéticos 2D para cada problema |
| `create_anisotropic_problem()` | Gera dados para Oracle Validation (W conhecido) |
| `llm_classify_point()` / `async_llm_classify_point()` | Chama API do LLM (sync/async) |
| `async_collect_llm_decisions()` | Coleta paralela com `asyncio.Semaphore` |
| `parse_llm_response()` | Parser de 7 camadas para respostas do LLM; fallback por hash MD5 determinístico (substitui aritmética modular para evitar viés geométrico) |
| `train_relaxed_perceptron()` | Perceptron Estruturado com relaxação de margem |
| `train_least_squares_inverse()` | NNLS via scipy (mínimos quadrados não-negativos) |
| `create_problem_e_meia_lua()` | **NOVO:** Problema E (sklearn.make_moons) — fronteira não-linear |
| `create_problem_homem_mulher()` | **NOVO:** carrega base real peso×altura (~100 amostras, elipse) |
| `compute_consistency_metrics()` | Mede consistência LLM vs. métrica aprendida |
| `phase_a_learn_metric()` | Fase A: retorna 2 métricas (Perceptron, NNLS) simultâneas |
| `phase_a_multifeature()` | **NOVO:** Fase A parametrizada por `n_features ∈ {2,3,4}`; reporta fidelidade vs LLM **e** acurácia vs rótulo real |
| `phase_consistency_test()` | Fases B/C: testa consistência em novos problemas |
| `phase_e_llm_as_learner()` | Fase E: LLM aprendendo do especialista |
| `run_external_problem_pipeline()` | **NOVO:** pipeline completo para problemas externos (peso×altura, meia-lua) — combina variantes de prompt × `n_features` × seeds, identifica melhor métrica e roda Fase E |
| `summarize_cross_linearity()` | **NOVO:** tabela cruzada linear × não-linear (ganhos 2→3→4 features) |
| `plot_llm_labels_per_problem()` | **NOVO:** scatter ponto-a-ponto das rotulações do LLM (e-mail 22:06) |
| `run_oracle_validation()` | Validação em dados sintéticos com W conhecido |
| `_oracle_algorithms()` / `_append_oracle_result()` | Helpers da Oracle Validation |
| `select_examples_by_strategy()` | Estratégias de seleção (easy/hard/mixed/random) |
| `select_examples_dilution()` | Seleção para experimento de diluição |
| `reorder_examples()` | Reordena exemplos para teste de recency bias |
| `augment_to_r3()` | Projeção R3: adiciona x3 = x1 * x2 (hipérbole) |
| `augment_to_r4()` | **NOVO:** Projeção R4: adiciona x1², x2² (elipse — classificador ótimo do peso×altura) |
| `augment_features()` | **NOVO:** Wrapper para `n_features ∈ {2,3,4}` |
| `build_prompt_zero_shot{,_variant}()` | Prompts zero-shot (padrão e variantes) |
| `build_prompt_few_shot{,_variant}()` | Prompts few-shot (padrão e variantes) |
| `print_error_analysis_by_region()` | Análise dos erros por região do plano |
| `print_hyperparameter_sensitivity()` | Sensibilidade a eta/C/delta_gamma |
| `print_example_order_analysis()` | Sumariza viés de ordem dos exemplos |
| `print_statistical_summary()` | Bootstrap CI, Wilcoxon, Cohen's d |
| `bootstrap_ci()` | IC 95% via bootstrap (10k reamostragens) |
| `Tee` | Duplica stdout para terminal + buffer (→ log_execucao.txt) |

### Fluxo de Execução (v4.0)

```
 1. Criar pasta de execução (timestamp); carregar .env
 2. Gerar dados sintéticos (4 problemas) + salvar CSVs em dados_sinteticos_seed{seed}/
 3. Visualização inicial (scatter plots dos 3 problemas)
 4. FASE A: LLM classifica sem exemplos → aprende Ŵ_LLM com 2 algoritmos (Perceptron + NNLS)
 5. FASE B: Testa Ŵ_LLM em Problema B (geometria diferente)
 6. FASE C: Testa Ŵ_LLM em Problema C (rotação anti-horária dos centróides — orientação oposta a B)
 7. [RUN_CLASS_ORDER_BIAS] Teste com classes invertidas (B/A, 1/0, etc.)
 8. [RUN_FEATURE_NAMES] Teste com nomes semânticos nas features
 9. [RUN_ALGORITHM_COMPARISON] Perceptron × NNLS lado a lado
10. [RUN_ORACLE_VALIDATION] Validação: algoritmos recuperam W conhecido?
11. FASE E: LLM aprende métrica do(s) especialista(s) externo(s)
12. [RUN_MULTIPLE_EXPERTS] Itera sobre 3 configs de expert
13. [RUN_DILUTION] Experimento de diluição (hard fixos + easy progressivos)
14. [RUN_PROMPT_VARIANTS] Teste de variantes de prompt (sensibilidade)
15. [RUN_CLASSICAL_BASELINES] Baselines clássicos na Fase E (k-NN, LR, SVM)
16. [RUN_EXAMPLE_ORDER_BIAS] Viés de ordem dos exemplos few-shot
17. [RUN_R3R4_EXPERIMENT] Augmentação R3 (x1·x2) e R4 (x1², x2²) — comparação 2/3/4 features em A/B/C lineares + meia-lua
18. **[RUN_HOMEM_MULHER] Pipeline peso × altura:** Fase A com 2/3/4 features (variantes prompt x1/x2 vs peso/altura) → identifica melhor métrica → Fase E com ela
19. **[RUN_PROBLEM_E] Pipeline meia-lua:** mesma estrutura para meia-lua sintética (`sklearn.make_moons`)
20. Tabela cruzada linear × não-linear (`summarize_cross_linearity`)
21. Visualizações ponto-a-ponto das rotulações do LLM (`24_llm_labels_*.png`)
22. Análises estatísticas (bootstrap CI, Wilcoxon, Cohen's d)
23. Gerar 20+ gráficos (muitos com painéis individuais e por seed)
24. Exportar CSVs (incluindo `results_external_phase_a_*.csv`, `bloco23_external_phase_e_*.csv`, `results_cross_linearity.csv`) e llm_interactions.json
25. Salvar log completo da execução
```

## Algoritmos de Otimização Inversa

**Dois** algoritmos independentes estimam a métrica diagonal W. Roda ambos na Fase A
para demonstrar robustez e permitir comparação de similaridade (cosseno entre Ws).

| Algoritmo | Arquivo | Formulação | Hiperparâmetros |
|---|---|---|---|
| **Perceptron Estruturado** | `relaxed_perceptron.py` | Relaxação de margem + busca binária em γ | eta=1 (padrão), C, delta_gamma, max_iter |
| **NNLS (Mínimos Quadrados)** | `least_squares_inverse.py` | `min ‖Aw - b‖²  s.t. w ≥ 0`, via `scipy.optimize.nnls` | nenhum |

O LP Max-Margin foi **removido** do trabalho (decisão da reunião 30/04/2026 — bug
da restrição de convexidade não respeitada; módulo arquivado em `src/arquivado/`).
Dois algoritmos congruentes (Perceptron + NNLS) já bastam para demonstrar robustez
do método de estimação.

## Hipóteses do Experimento

| Hipótese | Descrição |
|---|---|
| **H1** (Consistência) | O LLM mantém o mesmo critério decisório implícito. A métrica Ŵ_LLM **estimada** prevê classificações nos Problemas B e C (Kappa > 0.7). |
| **H2** (Few-shot amplifica) | Exemplos rotulados pela métrica W aumentam a concordância LLM-métrica. |
| **H3** (LLM como aprendiz) | O LLM consegue aprender o critério de um perito externo via few-shot. |
| **H4** (Exemplos difíceis) | **REFUTADA.** Hipótese a priori: "hard" > "easy". Dados mostram o oposto — exemplos "easy" (alta margem) superam "hard" consistentemente. Interpretação: exemplos prototípicos funcionam como âncoras de classe; ambíguos não fornecem sinal claro. |
| **H5** (Estabilidade) | O comportamento é reproduzível entre execuções (3 sementes × múltiplas repetições). |

## Fases Experimentais

### Fase A — Aprendizado de Métrica (Zero-Shot)
O LLM classifica 150 pontos 2D sem exemplos. A partir dessas classificações, aprende-se
uma **métrica de Mahalanobis diagonal** (Ŵ_LLM) com **três algoritmos simultaneamente**
(Perceptron, NNLS, LP). A fidelidade de cada algoritmo no Problema A e a similaridade
de cosseno entre os Ws estimados são registradas.

### Fase B — Teste de Consistência 1
Aplica Ŵ_LLM aprendida na Fase A ao Problema B (centroides deslocados verticalmente).
Testa se o LLM mantém consistência em distribuições não vistas.

### Fase C — Teste de Consistência 2
Similar à Fase B, com distorção geométrica mais severa.

### Fase E — LLM como Aprendiz
**Papel invertido:** especialista externo com métrica W_expert conhecida rotula os dados.
O LLM recebe exemplos do especialista (few-shot) e deve aprender a reproduzir as classificações.

Estratégias de seleção de exemplos:
- **Easy:** Pontos longe da fronteira de decisão (alta margem)
- **Hard:** Pontos próximos da fronteira (baixa margem, ambíguos)
- **Mixed:** 50% easy + 50% hard
- **Random:** Baseline sem estratégia

Múltiplos experts testados (quando `RUN_MULTIPLE_EXPERTS=True`):
- **aniso_x2:** W=[0.3, 1.5] — x2 dominante (original)
- **aniso_x1:** W=[1.5, 0.3] — x1 dominante (invertido)
- **euclidean:** W=[1.0, 1.0] — pesos iguais

### Experimentos Auxiliares

#### Oracle Validation (NOVO)
Gera dados sintéticos com W conhecido (ex.: `[0.1, 2.0]`) e verifica se os três algoritmos
recuperam esse W a partir de rótulos corretamente rotulados pelo próprio W.
Responde: "os algoritmos funcionam quando o oráculo realmente existe?"

Também testa **transferência**: aprende W em uma configuração anisotrópica e aplica em
outra, medindo queda de consistência quando a geometria muda.

#### Viés de Ordem das Classes
Testa se a ordem "A ou B" vs "B ou A" altera o resultado do LLM.

#### Nomes Semânticos de Features
Testa se usar "altura"/"peso" ao invés de "x1"/"x2" altera os pesos aprendidos.

#### Experimento de Diluição
Fixa 4 exemplos hard e adiciona easy progressivamente (0, 2, 4, 10, 16, 20).
Verifica se em algum ponto os fáceis "diluem" os difíceis e a performance deteriora.

#### Projeção R3 (Kernel Quadrático)
Cria terceira feature x3 = x1 * x2. Compara: LLM com 2 features vs LLM com 3 features
vs Métrica com 3 features. Detecta se o LLM captura interação implícita.

#### Variantes de Prompt (Sensibilidade)
4 variantes (default, geometric, cot, tabular) rodam Fases A-C completas em todas as seeds.
Painel principal: scatter de W aprendidos — se W muda entre variantes, o prompt confunde a medição.

#### Viés de Ordem dos Exemplos Few-Shot (Recency Bias)
Usa os mesmos exemplos (estratégia "mixed") com 4 ordenações diferentes:
- **class0_first:** classe 0, depois 1
- **class1_first:** classe 1, depois 0
- **shuffled:** ordem aleatória
- **alternating:** 0, 1, 0, 1, ...

Testado com n_shot = [5, 10, 20, 40].

#### Baselines Clássicos (k-NN, Logistic Regression, SVM)
Treina classificadores clássicos nos mesmos exemplos few-shot da Fase E e avalia no mesmo
conjunto de teste. Responde: "O LLM faz algo que um classificador trivial não faria?"
- **k-NN:** k ajustado (máx. 5, ímpar)
- **Logistic Regression:** linear, regularizado
- **SVM:** kernel RBF

#### Comparação de Algoritmos
Perceptron × NNLS × LP lado a lado nas Fases A-C. Demonstra que os resultados são robustos
ao método de estimação e reporta similaridade de cosseno entre os Ws estimados.

## Outputs por Execução

### Gráficos (PNG)

Outputs organizados em **3 blocos auto-contidos** + visualizações de fechamento.
Muitos gráficos são salvos tanto como **figura combinada** quanto como **painéis individuais**
(ex.: `bloco2_04_phase_e_learning_curve.png` + `..._{accuracy,f1,kappa}.png`).

**BLOCO 1 — LLM como FONTE (otimização inversa)**

| Arquivo | Conteúdo |
|---|---|
| `bloco1_01_problemas_lineares.png` | Scatter dos Problemas A, B, C lineares |
| `bloco1_02_problema_d_meialua_seed*_overview.png` | Problema D meia-lua (por seed) |
| `bloco1_03_oracle_w_recovery.png` | Sanity: recuperação de W conhecido |
| `bloco1_04_oracle_transfer.png` | Sanity: transferência entre geometrias |
| `bloco1_05_fase_a_errors_seed*.png` | Erros da métrica vs LLM (por seed) |
| `bloco1_06_w_distribution.png` | Distribuição de Ŵ_LLM entre seeds (boxplot, ratio, scatter) |
| `bloco1_07_algorithm_comparison.png` | Perceptron × NNLS (fidelidade, fases B/C, scatter de W) |
| `bloco1_08_seed_comparison.png` | Robustez entre seeds |
| `bloco1_09_consistency_extended.png` | Consistência, Kappa, F1 em B/C |
| `bloco1_10_r3r4_comparison.png` | R2→R3→R4 em lineares e meia-lua |
| `bloco1_12_class_order_bias.png` | Viés de ordem/posição das classes |
| `bloco1_13_prompt_variants.png` | Comparação de variantes de prompt |
| `bloco1_14a_class_names_effect.png` | Efeito do nome de classes |
| `bloco1_14b_feature_names_effect.png` | Efeito dos nomes semânticos de features |

**BLOCO 2 — LLM como APRENDIZ (Fase E)**

| Arquivo | Conteúdo |
|---|---|
| `bloco2_01_problema_e_expert.png` | Problema E (linear perito) com fronteira do expert |
| `bloco2_03_phase_e_strategies.png` | Distribuição espacial de exemplos por estratégia |
| `bloco2_04_phase_e_learning_curve.png` | Acurácia vs n_shot por estratégia |
| `bloco2_05_phase_e_strategy_comparison.png` | Estratégias em múltiplas métricas (painéis por n_shot) |
| `bloco2_06_dilution.png` | Experimento de diluição |
| `bloco2_07_example_order.png` | Viés de ordem dos exemplos (recency bias) |
| `bloco2_09_classical_baselines.png` | LLM vs k-NN, LR, SVM |

**BLOCO 3 — Estudo de caso REAL**

| Arquivo | Conteúdo |
|---|---|
| `bloco3_01_peso_altura_overview.png` | Base real peso × altura com fronteira elíptica |

**BLOCO 2/3 — Pipelines externos (meia-lua + peso × altura)**

| Arquivo | Conteúdo |
|---|---|
| `bloco23_external_learning_curve.png` | Fase E externa (curvas LLM vs n_shot) |
| `bloco23_external_features_comparison.png` | R2/R3/R4 comparados em peso×altura e meia-lua |
| `bloco23_external_decision_boundary.png` | Fronteiras de decisão das métricas externas |
| `bloco23_external_llm_vs_perceptron.png` | LLM in-context vs Perceptron baseline |

**FECHAMENTO — visualizações detalhadas por seed**

| Arquivo | Conteúdo |
|---|---|
| `final_02_confusion_matrices_seed*.png` | Matrizes de confusão por seed |
| `final_03_dashboard_seed*.png` | Dashboard consolidado por seed |
| `final_04_dataset_overview_seed*.png` | Visão geral dos 4 problemas por seed |
| `final_05_hits_errors_seed*.png` | Mapa de acertos/erros por problema, por seed |
| `final_06_w_algorithms_seed*.png` | Scatter de W por algoritmo, por seed |
| `final_07_margin_analysis_seed*.png` | Análise de margem por seed |
| `final_08_llm_labels_*.png` | Scatter LLM-labels por configuração |
| `final_09_model_comparison.png` | Comparação entre modelos LLM (quando aplicável) |

### CSVs por bloco
- `bloco1_phases_abc_{timestamp}.csv` — Bloco 1: fases A/B/C (30+ colunas; algoritmo, w_ratio, feature_names, prompt_variant)
- `bloco1_oracle_validation_{timestamp}.csv` — Bloco 1: recuperação/transferência de W conhecido
- `bloco1_algorithm_comparison_{timestamp}.csv` — Bloco 1: comparação Perceptron × NNLS
- `bloco1_r3r4_comparison_{timestamp}.csv` — Bloco 1: R2/R3/R4 em lineares
- `bloco2_phase_e_{timestamp}.csv` — Bloco 2: Fase E (inclui expert_name)
- `bloco2_dilution_{timestamp}.csv` — Bloco 2: experimento de diluição
- `bloco2_example_order_{timestamp}.csv` — Bloco 2: viés de ordem dos exemplos
- `bloco2_classical_baselines_{timestamp}.csv` — Bloco 2: baselines clássicos
- `bloco23_external_phase_a_{timestamp}.csv` — Bloco 2/3: Fase A pipelines externos
- `bloco23_external_phase_e_{timestamp}.csv` — Bloco 2/3: Fase E pipelines externos
- `final_cross_linearity.csv` — Síntese cruzada R2→R4 nos 3 blocos

### Outros Artefatos
- `log_execucao.txt` — Transcript completo com detalhes algorítmicos e métricas
- `llm_interactions.json` — Log estruturado de **todas** as chamadas à API (prompt, resposta, metadata)
- `dados_sinteticos_seed{seed}/problem_{A,B,C,D,A_r3}.csv` — Datasets por seed (reprodutibilidade)

## Leitura de Resultados

**IMPORTANTE:** Antes de responder qualquer pergunta sobre resultados ou análise, leia sempre
os arquivos da **última execução disponível**.

A última execução está em: `execucao_2026-04-14_22-10-02/`

O padrão de nome das pastas é `execucao_YYYY-MM-DD_HH-MM-SS/`. Se houver pastas mais recentes,
leia a de timestamp mais alto.

Arquivos prioritários para leitura:
1. `log_execucao.txt` — visão completa dos resultados
2. `results_phases_abc_v4_{timestamp}.csv` — métricas fases A-C
3. `results_phase_e_v4_{timestamp}.csv` — métricas fase D
4. `results_oracle_validation_{timestamp}.csv` — validação dos algoritmos
5. `results_algorithm_comparison_v4_{timestamp}.csv` — comparação dos 3 algoritmos
6. `results_dilution_v4_{timestamp}.csv` — métricas do experimento de diluição

## Referencias bibliograficas

Leia as referencias caso precise referencias apresentações e artigos.

Todas as referencias bibliograficas estão no arquivo `trabalhos_referencias.txt`.
PDFs de artigos citados estão em `artigos_referenciados/`.

## Dependências

```
openai          # API OpenAI (GPT-4o-mini) — sync + async
anthropic       # SDK Claude (integração opcional)
python-dotenv   # Carrega .env com chaves de API
numpy           # Álgebra linear
matplotlib      # Visualizações
pandas          # Exportação CSV
scikit-learn    # Cohen's Kappa, F1, k-NN, LR, SVM
scipy           # nnls (NNLS), stats (Wilcoxon)
```

## Notas de Design

- **Métrica diagonal:** Simplificação proposital para tratabilidade (O(d) parâmetros vs O(d²))
- **Dados 2D sintéticos:** Permite visualização; próximo passo: 3D e dados reais (Iris)
- **Dois algoritmos de otimização inversa:** Perceptron (com hiperparâmetros) + NNLS (sem) — garante robustez do método de estimação. LP Max-Margin foi removido (ver `reunioes_orientador/reuniao_30_04_2026/plano_trabalho.md` item 12).
- **Oracle Validation:** Dá um piso de sanidade — se os algoritmos falharem em recuperar W sintético conhecido, qualquer achado sobre o LLM fica comprometido
- **Concorrência assíncrona (MAX_CONCURRENCY=10):** Reduz tempo de coleta do LLM em ~10× via `asyncio.Semaphore`
- **Fallback por hash MD5:** Respostas malformadas após retries são mapeadas via `hashlib.md5` de coordenadas (substituiu aritmética modular para não introduzir viés geométrico)
- **Temperatura = 0.0:** Minimiza estocasticidade para isolar o critério de decisão (não garante determinismo — ver Reunião 1)
- **Nomes de classe variados:** `["A"/"B", "0"/"1", "Positivo"/"Negativo", "Azul"/"Vermelho"]` — detecta viés semântico
- **Nomes de classe invertidos:** `["B"/"A", "1"/"0", ...]` — detecta viés de posição/ordem
- **Nomes de features semânticos:** `["altura"/"peso", "feature_1"/"feature_2"]` — detecta viés semântico nas variáveis
- **Parser de 7 camadas:** Estratégia robusta para lidar com respostas malformadas do LLM
- **3 sementes aleatórias × 3 repetições:** Garante robustez estatística
- **Análise estatística:** Bootstrap CI (10k reamostragens), Wilcoxon signed-rank, Cohen's d
- **Projeção R3:** x3=x1*x2 simula kernel quadrático sem sair do mundo linear
- **Painéis individuais:** cada figura combinada também é salva como PNGs separados, úteis para apresentações
- **Dados por seed salvos em CSV:** reprodutibilidade bit-a-bit independente do código
- **Terminologia:** Usar "Ŵ_LLM estimada/inferida" (não "W aprendida") para a métrica da Fase A
