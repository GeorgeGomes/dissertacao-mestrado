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
- **Bloco 3 — Estudo de caso REAL:** Problema G, peso × altura (base do orientador, fronteira elíptica). Fase A R2/R3/R4 + Fase E + paradoxo do overfitting.

**Letras dos problemas** (não confundir com as fases A/B/C/E): A, B, C lineares e D meia-lua
(Bloco 1); E perito linear e F meia-lua (Bloco 2); G peso × altura (Bloco 3). A meia-lua é
o mesmo dado em dois papéis (D: LLM fonte; F: LLM aprendiz), gerada por `create_problem_d_meialua`.

## Convenção de nomenclatura dos assets (a partir de 03/07/2026; painéis desde 01/10/2026)

**Assets cujos dados derivam de UM LLM específico levam o ALIAS curto do modelo antes
da extensão**, via `llm_asset()` + `MODEL_ALIAS` (`src/execucao_io.py`): ex.
`bloco1_06_w_distribution__gpt4mini.png`, `final_05_hits_errors_seed42__flashlite.png`,
`final_cross_linearity__flashlite.csv`.

| Modelo | Alias |
|---|---|
| `openai/gpt-4o-mini` (id no OpenRouter; `gpt-4o-mini` nas execuções até 07/2026) | `gpt4mini` |
| `google/gemini-2.5-flash-lite` | `flashlite` |

**TODOS os modelos são tratados de forma igual**: cada modelo de `MODELS_TO_TEST` gera
o conjunto COMPLETO de gráficos por modelo (bloco1_05–14b incluindo bloco1_11,
bloco2_04–07/09, bloco23_*, final_02–08, final_10, `final_cross_linearity__<alias>.csv`)
**na raiz da execução** — não existe subpasta por modelo nem o conceito de "modelo
principal". Modelo fora do protocolo sem entrada em `MODEL_ALIAS` cai no slug longo
(`_model_slug`); teste em `tests/test_selecao_exemplos.py` garante a cobertura dos 2.

**Painéis individuais e variantes de um asset** (`asset_variant()`): o sufixo do painel
entra ANTES do alias, para o alias ficar sempre imediatamente antes da extensão:
`bloco1_06_w_distribution_boxplot__gpt4mini.png`,
`final_04_dataset_overview_seed42_problema_e__flashlite.png`,
`final_cross_linearity_corrigido_hm__gpt4mini.csv`. (Nunca `<base>__<alias>_<painel>`.)
Execuções anteriores a 01/10/2026 foram migradas por `src/renomear_assets_legado.py`.

**NÃO levam sufixo de modelo:**
- Assets sem dados de LLM: dados sintéticos (`bloco1_01`, `bloco1_02`, `bloco3_01`,
  `dados_sinteticos_*`), oráculos (`bloco1_03/04/04b`, `bloco1_oracle_*`), perito
  (`bloco2_01`, `bloco2_03`) e baselines clássicos (`bloco2_classical_baselines_*.csv`).
- Assets consolidados multi-modelo, identificados pela coluna `provider`/`model` no
  conteúdo: todos os CSVs com timestamp, `llm_interactions*.json`, `log_execucao*.txt` e
  `final_09_model_comparison.png`. Única exceção entre os CSVs:
  `final_cross_linearity__<alias>.csv` é gerado UM por modelo (leva alias).

(A lista de arquivos abaixo mostra os nomes-base; aplicar mentalmente o sufixo
`__<alias>` aos derivados de LLM — um arquivo por modelo — e `_<painel>` antes dele.)

## Estrutura de Pastas

```
mestrado/
├── src/
│   ├── dissertacao_mestrado.py       # Runner principal (~4850 linhas): config, fases, coleta LLM, main()
│   ├── plots.py                      # TODAS as visualizações (plot_*/visualize_*, ~2900 linhas)
│   ├── relatorios.py                 # Relatórios de texto: print_*, bootstrap_ci, summarize_cross_linearity
│   ├── execucao_io.py                # Tee (log), chunking de log/JSON, checkpoint, MODEL_ALIAS/llm_asset/asset_variant
│   ├── resultados.py                 # Dataclasses de resultado (ResultadoExperimento, ResultadoPhaseEExperimento, LearnedMetric)
│   ├── protocolo.py                  # Constantes compartilhadas (EXPERT_W, EXPERT_CENTROIDS, EXAMPLE_STRATEGIES, PERCEPTRON_PARAMS)
│   ├── llm_client.py                 # Factory de cliente (OpenRouter, provedor único) + pins por modelo
│   ├── llm_parser.py                 # Parser de 8 camadas (0–7) das respostas do LLM
│   ├── metrics.py                    # d_W, centróides, augmentação R3/R4, métricas de consistência
│   ├── data_problems.py              # Geradores dos problemas sintéticos + base real
│   ├── audit_interactions.py         # Auditoria offline (malformadas, flip de T=0, breakdown por modelo)
│   ├── relaxed_perceptron.py         # Perceptron Estruturado com Relaxação de Margem
│   ├── least_squares_inverse.py      # Mínimos Quadrados Não-Negativos (NNLS)
│   ├── classical_baselines.py        # Baselines clássicos (k-NN, LR, SVM)
│   ├── gerar_amostras_homem_mulher.py       # Amostras X/T/R da base peso×altura (pedido do orientador, 25/08/2026)
│   ├── corrigir_mapeamento_homem_mulher.py  # Correção offline H/M do Bloco 3 → *_corrigido_hm__<alias>.csv
│   ├── renomear_assets_legado.py     # Migra assets de execuções antigas p/ a convenção atual (dry-run padrão)
│   └── animacao_perceptron/          # Animação HTML do Perceptron (gerar.py, template.html, core.js)
├── tests/                        # unittest (parser, métricas, estimadores, plots, --apenas, renomeação, auditoria)
├── dados_reais/homem_mulher/     # Base real peso × altura (peso_altura.csv) + amostras X/T/R
├── requirements.txt              # Dependências Python
├── CLAUDE.md                     # Este arquivo
├── README.md                     # Como reproduzir
├── glossario.md                  # Glossário dos termos do trabalho
├── roteiro_apresentacao.txt      # Índice mestre dos slides (fonte de verdade das apresentações)
├── .env                          # Chaves de API (carregadas via python-dotenv)
├── .venv/                        # Ambiente virtual Python
├── .claude/
│   ├── settings.local.json       # Configurações locais do Claude Code
│   └── skills/                   # 13 skills ativas (cópia canônica; skills/ na raiz é só documentação)
├── reunioes_orientador/           # Transcrições e planos de reuniões com orientador
├── artigos_referenciados/         # PDFs de artigos citados
├── trabalhos_referencias.txt      # Lista consolidada de referências bibliográficas
└── execucao_YYYY-MM-DD_HH-MM-SS_{completa|smoke|apenas-<blocos>[_smoke]}/ # Pasta criada a cada execução
    #   sufixo `_completa` = execução cheia (fonte para o trabalho);
    #   sufixo `_smoke` = execução `--rapido` (NUNCA usar como fonte de números);
    #   sufixo `_apenas-<blocos>` = execução parcial (`--apenas`; nunca `_completa`).
    #   (As pastas execucao_* são COMMITADAS por decisão do projeto.)
    ├── dados_sinteticos_seed{seed}/       # CSVs por seed: problem_{A,B,C,E,A_r3,A_r4}.csv
    #   (problem_E.csv = perito linear do Bloco 2; antes de 01/10/2026 gravado como problem_D.csv)
    # === BLOCO 1 — LLM como FONTE (otim. inversa) ===
    ├── bloco1_01_problemas_lineares.png       # + painéis _problema_{a,b,c}.png
    ├── bloco1_02_problema_d_meialua_seed*_overview.png
    ├── bloco1_03_oracle_w_recovery.png        # Oracle: recuperação de W conhecido (+ _w_scatter/_ratio/_cosseno/_fidelidade)
    ├── bloco1_04_oracle_transfer.png          # Oracle: transferência entre geometrias (+ painel por expert)
    ├── bloco1_04b_oracle_meialua.png          # Oracle: aproximação da meia-lua em R2/R3/R4 (0 chamadas)
    ├── bloco1_05_fase_a_errors_seed*.png      # por seed (mapa de erros; + _mapa_erros/_erros_por_margem)
    ├── bloco1_06_w_distribution.png           # + _boxplot, _ratio, _scatter
    ├── bloco1_07_algorithm_comparison.png     # Perceptron × NNLS (+ _w_scatter/_fidelidade/_fase_b/_fase_c)
    ├── bloco1_08_seed_comparison.png
    ├── bloco1_09_consistency_extended.png     # + painéis por métrica
    ├── bloco1_10_r3r4_comparison.png          # R2→R3→R4 no Problema A (+ _linear_vs_quadratica/_algoritmos_r3r4)
    ├── bloco1_11_meialua_svm_vs_llm_seed*.png # SVM RBF vs rótulos zero-shot do LLM na meia-lua (Problema D)
    ├── bloco1_12_class_order_bias.png
    ├── bloco1_13_prompt_variants.png
    ├── bloco1_14a_class_names_effect.png
    ├── bloco1_14b_feature_names_effect.png
    # === BLOCO 2 — LLM como APRENDIZ (Fase E) ===
    ├── bloco2_01_problema_e_expert.png        # + painéis _ground_truth/_fronteira_perito/_margem_perito
    ├── bloco2_03_phase_e_strategies.png       # + painéis por estratégia
    ├── bloco2_04_phase_e_learning_curve.png   # + painéis por métrica (perito principal aniso_x2)
    ├── bloco2_05_phase_e_strategy_comparison.png  # + painéis por n_shot (perito principal)
    ├── bloco2_06_dilution.png
    ├── bloco2_07_example_order.png            # recency bias
    ├── bloco2_09_classical_baselines.png      # k-NN, LR, SVM × LLM (perito principal)
    # === BLOCO 3 — Estudo de caso REAL (Problema G) ===
    ├── bloco3_01_peso_altura_overview.png
    # === BLOCO 2/3 — Pipelines externos (meia-lua + peso×altura) ===
    ├── bloco23_external_learning_curve.png
    ├── bloco23_external_features_comparison.png
    ├── bloco23_external_decision_boundary.png
    ├── bloco23_external_llm_vs_perceptron.png
    # === FECHAMENTO (visualizações por seed) ===
    ├── final_02_confusion_matrices_seed*.png  # + _problema_{a,b,c}
    ├── final_03_dashboard_seed*.png
    ├── final_04_dataset_overview_seed*.png    # A/B/C + E (perito); + _problema_{a,b,c,e}
    ├── final_05_hits_errors_seed*.png         # + _problema_{a,b,c}
    ├── final_06_w_algorithms_seed*.png        # + _perceptron/_nnls/_barras
    ├── final_07_margin_analysis_seed*.png     # + _histograma/_taxa_erro/_mapa_confianca/_violin
    ├── final_08_llm_labels_{meia_lua|homem_mulher}_seed*_{x1_x2|peso_altura}_{n_shot|train}.png
    ├── final_09_model_comparison.png          # único asset de fechamento SEM alias (compara todos)
    ├── final_10_gamma_convergence.png         # diagnóstico da busca binária em γ (Fase A, Problema A)
    # === CSVs por bloco (consolidam todos os modelos; sem alias) ===
    ├── bloco1_phases_abc_{timestamp}.csv
    ├── bloco1_oracle_validation_{timestamp}.csv
    ├── bloco1_oracle_meialua_{timestamp}.csv
    ├── bloco1_algorithm_comparison_{timestamp}.csv
    ├── bloco1_r3r4_comparison_{timestamp}.csv
    ├── bloco2_phase_e_{timestamp}.csv
    ├── bloco2_dilution_{timestamp}.csv
    ├── bloco2_example_order_{timestamp}.csv
    ├── bloco2_classical_baselines_{timestamp}.csv
    ├── bloco23_external_phase_a_{timestamp}.csv
    ├── bloco23_external_phase_e_{timestamp}.csv
    ├── final_cross_linearity__<alias>.csv     # UM por modelo (exceção: CSV com alias)
    ├── *_corrigido_hm[__<alias>].csv          # cópias corrigidas offline (corrigir_mapeamento_homem_mulher.py)
    ├── llm_interactions_parte{NNN}.json  # Log de prompts/respostas (particionado em 10 MiB; arquivo único se couber)
    └── log_execucao.txt                  # (log_execucao_parte{NNN}.txt se passar de 10 MiB)
```

## Regras para Apresentações

**Roteiro base:** Sempre que for gerar uma apresentação LaTeX Beamer, usar o arquivo
`roteiro_apresentacao.txt` (na raiz do projeto) como **índice mestre** da ordem dos slides
e do que cada um deve conter. O arquivo lista, slide a slide: título, tipo de conteúdo
(texto / tabela / gráfico / fórmula) e uma breve descrição do que deve aparecer.
Os números concretos (métricas, tabelas de resultados) devem ser extraídos dos CSVs e do
`log_execucao.txt` da execução **`_completa`** mais recente (nunca de uma `_smoke`).

**Ordem obrigatória de edição:** Ao criar ou alterar qualquer apresentação, **sempre atualizar
primeiro o `roteiro_apresentacao.txt`** e só depois os arquivos `.tex`
(`apresentacao.tex` e `apresentacao_guia.tex`). O roteiro é a fonte de verdade; os `.tex`
seguem o roteiro, nunca o contrário. Os três arquivos devem manter **a mesma quantidade e a mesma
numeração de slides** — ao adicionar, remover ou dividir um slide, refletir a mudança no roteiro
primeiro (renumerando as entradas "Slide N —") e então propagar para os dois `.tex`.

Quando o usuário pedir para criar uma apresentação LaTeX Beamer com resultados de uma execução,
o arquivo `.tex` deve ser criado **dentro da pasta da execução correspondente**
(ex: `execucao_YYYY-MM-DD_HH-MM-SS_completa/apresentacao/apresentacao.tex`), junto com os gráficos e CSVs.
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
# OPENROUTER_API_KEY=sk-or-...   (necessário p/ o Gemini via OpenRouter)
# (provedor único desde 01/10/2026: nenhuma outra chave é necessária)

# Executar o experimento completo (todos os modelos de MODELS_TO_TEST)
python src/dissertacao_mestrado.py

# Smoke test barato de UM modelo (antes de gastar na execução completa):
python src/dissertacao_mestrado.py --rapido --modelo gpt
python src/dissertacao_mestrado.py --rapido --modelo gemini

# Execução PARCIAL: só os blocos indicados (bloco1 | bloco2 | bloco3 | oraculo; aceita vários)
python src/dissertacao_mestrado.py --apenas bloco3                 # só peso×altura (Fase A + E)
python src/dissertacao_mestrado.py --apenas bloco2 bloco3 --modelo gpt
python src/dissertacao_mestrado.py --apenas oraculo               # 0 chamadas de API
```

```bash
# Ferramentas offline (0 chamadas de API):
python src/audit_interactions.py                       # auditoria da última _completa
python src/corrigir_mapeamento_homem_mulher.py         # correção H/M do Bloco 3 (*_corrigido_hm__<alias>.csv)
python src/renomear_assets_legado.py [--aplicar --docs] # migra assets antigos p/ a convenção (dry-run padrão)
python src/animacao_perceptron/gerar.py                # animação HTML do Perceptron
```

A execução cria automaticamente uma pasta `execucao_YYYY-MM-DD_HH-MM-SS_completa/`
(ou `..._smoke/` quando rodada com `--rapido`) com todos os outputs.
Com `--apenas`, a pasta recebe `..._apenas-<blocos>/` (ex.: `_apenas-bloco3`, ou
`_apenas-bloco3_smoke` se combinada com `--rapido`) e **nunca** `_completa`: é uma
execução parcial e não serve de fonte para o trabalho inteiro. `--apenas` força as flags
`RUN_*` de topo dos blocos escolhidos (`BLOCOS_APENAS` em `dissertacao_mestrado.py`) e
desliga as dos demais; as sub-flags (vieses, diluição, baselines, A/B, oráculo
meia-lua) continuam valendo o que está no código, pois já são gateadas pelo bloco-pai.
`RUN_R3R4_EXPERIMENT` é flag de topo do bloco1 (depende do cache da Fase A) e
`RUN_PROBLEM_MEIALUA` pertence a bloco1 E bloco2 (Problema D e F); `oraculo` é chave
própria (0 chamadas), embora seus assets sejam `bloco1_03/04/04b`.
As chamadas à API do LLM são **assíncronas com concorrência limitada**
(`asyncio.Semaphore(MAX_CONCURRENCY)`), acelerando substancialmente a coleta de decisões.

## Configuração Principal

Constantes no topo de `src/dissertacao_mestrado.py`:

| Constante | Valor padrão | Descrição |
|---|---|---|
| `RANDOM_SEED` | `42` | Semente padrão |
| `RANDOM_SEEDS` | `[42, 123, 7]` | 3 seeds para robustez estatística (grid completo) |
| `EXTRA_SEEDS_CORE` | `[]` (desativado) | Seeds extras do pipeline central dos modelos `scope="full"`; protocolo atual = 3 seeds uniformes em tudo. Reativar com `[2025, 314, 611]` habilita Wilcoxon pareado por seed no central |
| `BIAS_N_SHOTS` | `[0, 10]` | Âncoras dos experimentos de viés (nomes de classe, variantes de prompt) — efeitos comparáveis entre si |
| `N_SAMPLES_PROBLEM_A` | `150` | Amostras no Problema A |
| `N_SAMPLES_PROBLEM_B/C` | `100` | Amostras nos Problemas B e C |
| `N_SAMPLES_PROBLEM_E` | `150` | Amostras no Problema E (perito linear); a meia-lua (D/F) usa `N_SAMPLES_PROBLEM_A` |
| `FEW_SHOT_SIZES` | `[0, 4, 10, 20, 40]` | Tamanhos few-shot do Bloco 1 (Fases A/B/C; todos pares: balanço exato de classes no prompt) |
| `FEW_SHOT_SIZES_PHASE_E` | `[0, 4, 10, 20, 40]` | Tamanhos few-shot da Fase E (todos pares) |
| `DILUTION_EASY_ADDITIONS` | `[0, 2, 4, 10, 16, 20]` | Easy adicionados aos 4 hard fixos na diluição |
| `EXAMPLE_ORDER_N_SHOTS` | `[4, 10, 20]` | n_shot do viés de ordem dos exemplos |
| `PERCEPTRON_PARAMS` | `eta=0.001, C=1.0, delta_gamma=0.05, max_epochs=50, tol=1e-4` | Hiperparâmetros do Perceptron (ponto único, `protocolo.py`) |
| `N_REPETICOES` | `3` | Repetições ONDE HÁ SORTEIO de exemplos (regra única em `reps_para`) |
| `MAX_CONCURRENCY` | `10` | Chamadas paralelas à API do LLM |
| `MAX_FORMAT_RETRIES` | `5` | Reenvios para respostas malformadas |
| `EXAMPLE_STRATEGIES` | `["easy","hard","mixed","random"]` | Estratégias da Fase E |
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
| `RUN_ALGORITHM_COMPARISON` | `True` | Comparação Perceptron × NNLS |
| `RUN_ORACLE_VALIDATION` | `True` | Validação: algoritmos recuperam W conhecido? |
| `RUN_ORACLE_MEIALUA` | `True` | Sub-flag do oráculo: aproximação da meia-lua em R2/R3/R4 (0 chamadas LLM) |
| `RUN_EXAMPLE_ORDER_BIAS` | `True` | Teste de viés de ordem dos exemplos few-shot |
| `RUN_PROMPT_VARIANTS` | `True` | Teste de múltiplas variantes de prompt |
| `RUN_CLASSICAL_BASELINES` | `True` | Comparação com baselines clássicos (k-NN, LR, SVM) |
| `RUN_PROBLEM_MEIALUA` | `True` | Meia-lua — Problema D (Bloco 1) e F (Bloco 2): não-linearidade explícita |
| `RUN_HOMEM_MULHER` | `True` | Estudo de caso real peso×altura (Problema G; homem/mulher, elipse) |
| `RUN_HM_CLASS_NAMES_AB` | `True` | Sub-flag do Bloco 3: variante neutra de nomes de classe (A/B) no peso×altura |

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
# O W verdadeiro é implícito nos desvios por eixo (W_i ∝ 1/σ_i²); os rótulos do
# oráculo vêm da regra do centróide mais próximo sob esse W.
ORACLE_ANISO_CONFIGS = [
    {"name": "x2_dom",      "centers": [(-2.0, 0.5), (2.0, -0.5)], "std": [0.5, 2.0], "desc": "x1 preciso, x2 ruidoso"},
    {"name": "x1_dom",      "centers": [(-0.5, 2.0), (0.5, -2.0)], "std": [2.0, 0.5], "desc": "x1 ruidoso, x2 preciso"},
    {"name": "forte_aniso", "centers": [(-1.5, 1.0), (1.5, -1.0)], "std": [0.3, 1.5], "desc": "anisotropia forte"},
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

**Modelos** (`MODELS_TO_TEST`, formato `(provider, model, temperature, scope)`):
- Protocolo atual: **2 modelos com `scope="full"`** — o grid COMPLETO de
  experimentos roda em ambos, com 3 seeds uniformes:
  `openai/gpt-4o-mini` e `google/gemini-2.5-flash-lite`, AMBOS via **OpenRouter**,
  provedor único (`OPENROUTER_API_KEY`; `extra_body` com `allow_fallbacks=False`).
  Os provedores diretos (OpenAI, Gemini, Anthropic) foram removidos de `llm_client.py`
  em 01/10/2026: para usar outro modelo, acrescente-o a `MODELS_TO_TEST` com o id do
  OpenRouter (`<org>/<modelo>`), um alias em `MODEL_ALIAS` e um pin em `MODEL_PROVIDER_PIN`.
- **Pinagem de provedor de inferência POR MODELO** (`MODEL_PROVIDER_PIN` em
  `src/llm_client.py`): fixa qual infraestrutura serve cada modelo no OpenRouter
  (mesmo model-ID pode ser servido por empresas/quantizações distintas). Valor
  observado no smoke de 02/07/2026: Gemini → `Google`; GPT-4o-mini → `OpenAI`.
  Se o provedor pinado cair, as chamadas falham visivelmente (fallback resiliente +
  auditoria) em vez de migrar em silêncio — comportamento desejado p/ reprodutibilidade.
- `scope="core"` continua disponível para reduzir um modelo ao pipeline central
  (Fases A-C + Fase E perito principal + externos, sem experimentos auxiliares).
- CLI `--modelo <substring>` filtra `MODELS_TO_TEST` (smoke test isolado).
- CLI `--apenas <bloco> [...]` (`bloco1|bloco2|bloco3|oraculo`) roda só os blocos
  indicados; pasta `_apenas-<blocos>` (nunca `_completa`).
- **Todos os modelos são iguais para outputs**: cada um gera o conjunto completo de
  PNGs por modelo na raiz da execução, com seu alias (`MODEL_ALIAS`) no nome do asset.
  CSVs consolidam todos (colunas `provider`/`model`); `final_09_model_comparison.png`
  compara todos. Assets independentes de LLM (oráculos, perito, overviews sintéticos,
  baselines clássicos) são gerados 1× (gate `model_idx == 0`).

**Regra única de repetição** (`reps_para` em `dissertacao_mestrado.py`): repetições
existem para variar o SORTEIO dos exemplos few-shot (Fase E e externos:
`random_state = seed + rep`). Onde a seleção é determinística — zero-shot, seleção por
margem nas Fases B/C, diluição, ordenações fixas — roda-se **1 coleta**; o
não-determinismo por consulta (flip a T=0) é quantificado pela auditoria offline.

## Arquitetura do Código

### Estruturas de Dados Principais (`src/resultados.py`)

```python
@dataclass
class ResultadoExperimento        # Resultado das Fases A-C (Bloco 1)
    # provider, model_name, temperature, random_seed, n_shot, nomes_classes, repeticao
    # consistencia/kappa/f1 nos Problemas B e C; consistencia_euclidiana_{b,c}
    # acuracia_{llm,metrica}_vs_gt_problema_{a,b,c}
    # w_aprendido (Ŵ_LLM estimada, Perceptron), gamma_otimo
    # fidelidade_perceptron, fidelidade_nnls, w_cosine_sim_nnls
    # w_ratio, w_direction, feature_names, prompt_variant, diagonal_limitation_flag
    # n_classe_*, n_disagreements_*, n_malformed_responses

@dataclass
class ResultadoPhaseEExperimento  # Resultado da Fase E (Bloco 2)
    # accuracy/kappa/f1 LLM vs. perito; accuracy_expert_vs_gt, accuracy_llm_vs_gt
    # example_strategy: "easy"/"hard"/"mixed"/"random", ou "mixed_order_<ordem>"
    #   (viés de ordem) ou "dilution_<N>hard_<M>easy" (diluição)
    # expert_w, expert_name (aniso_x2 | aniso_x1 | euclidean)

@dataclass
class LearnedMetric               # Métrica Mahalanobis estimada
    # vetor w, centroids, gamma, source_problem
```

O Bloco 3 e os pipelines externos (`bloco23_*`) usam dicts (ver `run_external_problem_pipeline`).

### Funções Críticas

| Função | Propósito |
|---|---|
| `create_problem_{a,b,c}()` | Gera dados sintéticos 2D lineares (`data_problems.py`) |
| `create_problem_e_expert()` | Problema E (perito linear do Bloco 2) |
| `create_problem_d_meialua()` | Meia-lua (`sklearn.make_moons`) — Problema D (Bloco 1) / F (Bloco 2) |
| `create_problem_homem_mulher()` | Carrega base real peso×altura (Problema G, ~100 amostras, elipse) |
| `create_anisotropic_problem()` | Gera dados para Oracle Validation (W conhecido) |
| `async_llm_classify_point()` / `async_llm_classify_point_openai()` | Chamam a API (OpenRouter, endpoint compatível com OpenAI) |
| `async_collect_llm_decisions()` / `collect_llm_decisions()` | Coleta paralela com `asyncio.Semaphore` (+ wrapper síncrono) |
| `parse_llm_response()` | Parser de 8 camadas (0–7) (`llm_parser.py`); camada 6 usa padrões com a classe explícita; fallback por hash MD5 determinístico |
| `audit_interactions.audit()` | Auditoria offline dos `llm_interactions*.json` — malformadas, flip de `T=0`, breakdown por modelo |
| `train_relaxed_perceptron()` | Perceptron Estruturado com relaxação de margem (`**PERCEPTRON_PARAMS`) |
| `train_least_squares_inverse()` | NNLS via scipy (mínimos quadrados não-negativos) |
| `compute_consistency_metrics()` | Mede consistência LLM vs. métrica estimada |
| `phase_a_learn_metric()` | Fase A: retorna 2 métricas (Perceptron, NNLS) simultâneas |
| `phase_a_multifeature()` | Fase A parametrizada por `n_features ∈ {2,3,4}`; fidelidade vs LLM **e** acurácia vs rótulo real |
| `phase_consistency_test()` | Fases B/C: testa consistência em novos problemas (retorna também `X_test`) |
| `phase_e_llm_as_learner()` | Fase E: LLM aprendendo do especialista |
| `run_external_problem_pipeline()` | Pipeline dos problemas externos (meia-lua, peso×altura): variantes de prompt × `n_features` × seeds → melhor métrica → Fase E |
| `run_oracle_validation()` / `run_oracle_meialua()` | Validação em dados sintéticos com W conhecido / aproximação da meia-lua em R2/R3/R4 |
| `_oracle_algorithms()` / `_append_oracle_result()` | Helpers da Oracle Validation |
| `select_examples_by_strategy()` / `select_confident_examples()` | Estratégias de seleção (easy/hard/mixed/random) / seleção por margem (Fases B/C) |
| `select_examples_dilution()` | Seleção para experimento de diluição |
| `reorder_examples()` | Reordena exemplos para teste de recency bias |
| `augment_to_r3()` / `augment_to_r4()` / `augment_features()` | Projeções R3 (x1·x2), R4 (x1², x2²) e wrapper por `n_features` (`metrics.py`) |
| `build_prompt_zero_shot{,_variant}()` / `build_prompt_few_shot{,_variant}()` | Prompts (padrão e variantes) |
| `reps_para()` | Regra única de repetição (1 coleta onde a seleção é determinística) |
| `flags_para_apenas()` / `sufixo_apenas()` | CLI `--apenas`: flags por bloco e sufixo da pasta |
| `summarize_cross_linearity()` | Tabela cruzada linear (Problema A, R2/R3/R4) × não-linear → `final_cross_linearity__<alias>.csv` |
| `print_error_analysis_by_region()` / `print_hyperparameter_sensitivity()` / `print_example_order_analysis()` / `print_statistical_summary()` / `bootstrap_ci()` | Relatórios do log (`relatorios.py`) |
| `plot_llm_labels_per_problem()` / `plot_meialua_svm_vs_llm()` / `plot_gamma_convergence()` / `plot_oracle_meialua()` | `final_08`, `bloco1_11`, `final_10`, `bloco1_04b` (`plots.py`) |
| `llm_asset()` / `asset_variant()` / `_model_alias()` | Nomes de asset com alias e painéis (`execucao_io.py`) |
| `Tee` | Duplica stdout/stderr para terminal + buffer (→ log_execucao.txt) |

### Fluxo de Execução (ordem real do `main()`)

```
 1. Criar pasta de execução (timestamp + sufixo); carregar .env; validar chaves de API
 2. Por (modelo, seed): gerar A, B, C, E + salvar dados_sinteticos_seed{seed}/ (problem_{A,B,C,E}.csv)
 3. [seed0/model0] bloco1_01 (A/B/C), bloco2_01 (perito), bloco2_03 (estratégias)
 4. [RUN_PHASES_ABC] BLOCO 1 — FASE A (zero-shot → Ŵ_LLM Perceptron + NNLS) e Fases B/C
    (few-shot por margem), p/ cada nome de classe; dentro: [RUN_CLASS_ORDER_BIAS]
    nomes invertidos, [RUN_PROMPT_VARIANTS] variantes (BIAS_N_SHOTS), [RUN_FEATURE_NAMES]
    nomes semânticos, [RUN_ALGORITHM_COMPARISON] Perceptron × NNLS
 5. [RUN_ORACLE_VALIDATION] oráculo (model0): recuperação/transferência de W; [RUN_ORACLE_MEIALUA]
 6. [RUN_PHASE_E] BLOCO 2 — Fase E (perito principal; [RUN_MULTIPLE_EXPERTS] 3 peritos);
    [RUN_CLASSICAL_BASELINES] k-NN/LR/SVM (model0); [RUN_DILUTION]; [RUN_EXAMPLE_ORDER_BIAS]
 7. [RUN_R3R4_EXPERIMENT] R3/R4 sobre os rótulos zero-shot do Problema A (problem_A_r3/r4.csv)
 8. [RUN_HOMEM_MULHER] BLOCO 3 — pipeline externo peso×altura (Fase A 2/3/4 atributos × variantes
    de prompt → melhor métrica → Fase E); [RUN_HM_CLASS_NAMES_AB] variante A/B
 9. [RUN_PROBLEM_MEIALUA] pipeline externo meia-lua (Problema D/F) + bloco1_02 + bloco1_11 (SVM vs LLM)
10. Visualizações consolidadas: final_09 (se >1 modelo), bloco1_03/04/04b (oráculo)
11. Por modelo: BLOCO 1 (bloco1_09, 14a, 08, 06, 05, 12, 14b, 13, 10, 07, final_10) →
    BLOCO 2 (bloco2_04, 05, 09, 06, 07) → FECHAMENTO por seed (final_04, 05, 06, 02, 07, 03) →
    análises de texto (print_phase_e_analysis, print_statistical_summary, …)
12. Exportar CSVs por bloco (bloco1_*, bloco2_*, bloco23_*), final_cross_linearity__<alias>.csv
13. final_08_llm_labels_* (ponto-a-ponto) e bloco23_external_* (curvas/fronteiras externas)
14. Resumos consolidados dos pipelines externos no log; salvar llm_interactions (partes) e log
```

## Algoritmos de Otimização Inversa

**Dois** algoritmos independentes estimam a métrica diagonal W. Roda ambos na Fase A
para demonstrar robustez e permitir comparação de similaridade (cosseno entre Ws).

| Algoritmo | Arquivo | Formulação | Hiperparâmetros |
|---|---|---|---|
| **Perceptron Estruturado** | `relaxed_perceptron.py` | Relaxação de margem + busca binária em γ | `PERCEPTRON_PARAMS` (`protocolo.py`): eta=0.001, C=1.0, delta_gamma=0.05, max_epochs=50, tol=1e-4 |
| **NNLS (Mínimos Quadrados)** | `least_squares_inverse.py` | `min ‖Aw - b‖²  s.t. w ≥ 0`, via `scipy.optimize.nnls` | nenhum |

O LP Max-Margin foi **removido** do trabalho (decisão da reunião 30/04/2026 — bug
da restrição de convexidade não respeitada; o módulo foi apagado do repositório, não
existe pasta `src/arquivado/`).
Dois algoritmos congruentes (Perceptron + NNLS) já bastam para demonstrar robustez
do método de estimação.

## Hipóteses do Experimento

| Hipótese | Descrição |
|---|---|
| **H1** (Consistência) | O LLM mantém o mesmo critério decisório implícito. A métrica Ŵ_LLM **estimada** prevê classificações nos Problemas B e C com concordância ao menos moderada (Kappa > 0,4, piso de Landis & Koch 1977 — limiar do artigo; 0,7 seria "domínio", não "existência" do critério). |
| **H2** (Few-shot amplifica) | Exemplos rotulados pela métrica W aumentam a concordância LLM-métrica. |
| **H3** (LLM como aprendiz) | O LLM consegue aprender o critério de um perito externo via few-shot. |
| **H4** (Exemplos difíceis) | **REFUTADA.** Hipótese a priori: "hard" > "easy". Dados mostram o oposto — exemplos "easy" (alta margem) superam "hard" consistentemente. Interpretação: exemplos prototípicos funcionam como âncoras de classe; ambíguos não fornecem sinal claro. |
| **H5** (Estabilidade) | O comportamento é reproduzível entre execuções (3 sementes; 3 repetições onde há sorteio de exemplos, regra `reps_para`). |

## Fases Experimentais

### Fase A — Estimação da Métrica (Zero-Shot)
O LLM classifica 150 pontos 2D sem exemplos. A partir dessas classificações, estima-se
uma **métrica de Mahalanobis diagonal** (Ŵ_LLM) com **dois algoritmos simultaneamente**
(Perceptron Estruturado e NNLS). A fidelidade de cada algoritmo no Problema A e a
similaridade de cosseno entre os Ws estimados são registradas. Nos pipelines externos
(Problema D meia-lua, Problema G peso×altura) a Fase A roda com 2/3/4 atributos (R2/R3/R4).

### Fase B — Teste de Consistência 1
Aplica Ŵ_LLM estimada na Fase A ao Problema B (centróides rotacionados no sentido
horário, ±1.5). Testa se o LLM mantém consistência em distribuições não vistas.

### Fase C — Teste de Consistência 2
Similar à Fase B, com rotação anti-horária dos centróides (mesma magnitude, orientação
oposta à de B).

### Fase E — LLM como Aprendiz
**Papel invertido:** especialista externo com métrica W_expert conhecida rotula os dados
(Problema E; na meia-lua, Problema F, o "perito" é o ground truth; no Problema G, a
melhor métrica da Fase A externa). O LLM recebe exemplos do especialista (few-shot) e
deve aprender a reproduzir as classificações.

Estratégias de seleção de exemplos:
- **Easy:** Pontos longe da fronteira de decisão (alta margem)
- **Hard:** Pontos próximos da fronteira (baixa margem, ambíguos)
- **Mixed:** 50% easy + 50% hard
- **Random:** Baseline sem estratégia

Múltiplos experts testados (quando `RUN_MULTIPLE_EXPERTS=True`; os gráficos `bloco2_04/05/09`
e o dashboard mostram o perito principal `aniso_x2`; o log e o CSV cobrem os 3):
- **aniso_x2:** W=[0.3, 1.5] — x2 dominante (original)
- **aniso_x1:** W=[1.5, 0.3] — x1 dominante (invertido)
- **euclidean:** W=[1.0, 1.0] — pesos iguais

### Experimentos Auxiliares

#### Oracle Validation
Gera dados sintéticos com W conhecido (`ORACLE_ANISO_CONFIGS`: W implícito nos desvios por
eixo, ex. `std=[0.3, 1.5]`) e verifica se os dois algoritmos recuperam esse W a partir de
rótulos atribuídos pelo próprio W. `RUN_ORACLE_MEIALUA` acrescenta a aproximação da
meia-lua em R2/R3/R4 (`bloco1_04b`, `bloco1_oracle_meialua_*.csv`).
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

#### Projeções R3 e R4 (Kernel Quadrático)
Sobre os MESMOS rótulos zero-shot do Problema A (o LLM só viu x1, x2), estima a métrica
com 3 pesos (R3: x3 = x1·x2, hipérbole) e 4 pesos (R4: x1², x2², elipse). Se a fidelidade
cresce com a dimensão, o LLM adota implicitamente um critério não-linear; em problema
linear espera-se ganho ~nulo (`bloco1_10`, `bloco1_r3r4_comparison_*.csv`). Nos pipelines
externos (D e G) a mesma comparação 2/3/4 atributos vai para `bloco23_external_features_comparison`.

#### Variantes de Prompt (Sensibilidade)
3 variantes extras (geometric, cot, tabular) rodam Fases A-C com `BIAS_N_SHOTS=[0, 10]`
em todas as seeds, comparadas à `default` do grid principal.
Painel principal: scatter de W aprendidos — se W muda entre variantes, o prompt confunde a medição.

#### Viés de Ordem dos Exemplos Few-Shot (Recency Bias)
Usa os mesmos exemplos (estratégia "mixed") com 4 ordenações diferentes:
- **class0_first:** classe 0, depois 1
- **class1_first:** classe 1, depois 0
- **shuffled:** ordem aleatória
- **alternating:** 0, 1, 0, 1, ...

Testado com n_shot = `EXAMPLE_ORDER_N_SHOTS = [4, 10, 20]`.

#### Baselines Clássicos (k-NN, Logistic Regression, SVM)
Treina classificadores clássicos nos mesmos exemplos few-shot da Fase E e avalia no mesmo
conjunto de teste. Responde: "O LLM faz algo que um classificador trivial não faria?"
- **k-NN:** k ajustado (máx. 5, ímpar)
- **Logistic Regression:** linear, regularizado
- **SVM:** kernel RBF

#### Comparação de Algoritmos
Perceptron × NNLS lado a lado nas Fases A-C. Demonstra que os resultados são robustos
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
| `bloco1_04b_oracle_meialua.png` | Sanity: aproximação da meia-lua em R2/R3/R4 |
| `bloco1_05_fase_a_errors_seed*.png` | Erros da métrica vs LLM (por seed) |
| `bloco1_06_w_distribution.png` | Distribuição de Ŵ_LLM entre seeds (boxplot, ratio, scatter) |
| `bloco1_07_algorithm_comparison.png` | Perceptron × NNLS (fidelidade, fases B/C, scatter de W) |
| `bloco1_08_seed_comparison.png` | Robustez entre seeds |
| `bloco1_09_consistency_extended.png` | Consistência, Kappa, F1 em B/C |
| `bloco1_10_r3r4_comparison.png` | R2→R3→R4 no Problema A (painéis `_linear_vs_quadratica`, `_algoritmos_r3r4`) |
| `bloco1_11_meialua_svm_vs_llm_seed*.png` | SVM RBF vs rótulos zero-shot do LLM na meia-lua (Problema D) |
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

**BLOCO 3 — Estudo de caso REAL (Problema G)**

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
| `final_04_dataset_overview_seed*.png` | Visão geral dos 4 datasets sintéticos (A, B, C, E) por seed |
| `final_05_hits_errors_seed*.png` | Mapa de acertos/erros por problema, por seed |
| `final_06_w_algorithms_seed*.png` | Scatter de W por algoritmo, por seed |
| `final_07_margin_analysis_seed*.png` | Análise de margem por seed |
| `final_08_llm_labels_*.png` | Scatter LLM-labels por configuração |
| `final_09_model_comparison.png` | Comparação entre modelos LLM (quando aplicável; sem alias) |
| `final_10_gamma_convergence.png` | Diagnóstico da busca binária em γ do Perceptron (Fase A, Problema A) |

### CSVs por bloco
- `bloco1_phases_abc_{timestamp}.csv` — Bloco 1: fases A/B/C (30+ colunas; algoritmo, w_ratio, feature_names, prompt_variant)
- `bloco1_oracle_validation_{timestamp}.csv` — Bloco 1: recuperação/transferência de W conhecido
- `bloco1_oracle_meialua_{timestamp}.csv` — Bloco 1: oráculo da meia-lua em R2/R3/R4
- `bloco1_algorithm_comparison_{timestamp}.csv` — Bloco 1: comparação Perceptron × NNLS
- `bloco1_r3r4_comparison_{timestamp}.csv` — Bloco 1: R2/R3/R4 no Problema A
- `bloco2_phase_e_{timestamp}.csv` — Bloco 2: Fase E (inclui expert_name)
- `bloco2_dilution_{timestamp}.csv` — Bloco 2: experimento de diluição
- `bloco2_example_order_{timestamp}.csv` — Bloco 2: viés de ordem dos exemplos
- `bloco2_classical_baselines_{timestamp}.csv` — Bloco 2: baselines clássicos
- `bloco23_external_phase_a_{timestamp}.csv` — Bloco 2/3: Fase A pipelines externos
- `bloco23_external_phase_e_{timestamp}.csv` — Bloco 2/3: Fase E pipelines externos
- `final_cross_linearity__<alias>.csv` — Síntese cruzada R2→R4 nos 3 blocos (UM por modelo; linhas sintéticas rotuladas `A_linear`)
- `*_corrigido_hm[__<alias>].csv` — cópias corrigidas offline do Bloco 3 (`corrigir_mapeamento_homem_mulher.py`)

### Outros Artefatos
- `log_execucao.txt` — Transcript completo com detalhes algorítmicos e métricas (particionado em `log_execucao_parte{NNN}.txt` se passar de 10 MiB)
- `llm_interactions_parte{NNN}.json` — Log estruturado de **todas** as chamadas à API
  (prompt, `point`, `raw_response`, `parsed_label`, `model`, `model_resolved` — snapshot
  datado devolvido pela API, p/ reprodutibilidade —, `inference_provider` — provedor de
  inferência no OpenRouter —, `temperature`, `format_retries`,
  `malformed`), **particionado** em blocos de 10 MiB (`_parte001.json`, `_parte002.json`, …)
  para evitar um único arquivo gigante. Execuções pequenas (smoke, `--apenas`) e execuções
  antigas gravam o arquivo único `llm_interactions.json` — o auditor aceita ambos os esquemas.
- `dados_sinteticos_seed{seed}/problem_{A,B,C,E,A_r3,A_r4}.csv` — Datasets por seed (reprodutibilidade; `problem_E.csv` = perito linear)

### Auditoria Offline das Interações (`src/audit_interactions.py`)
Verifica, **sem chamar a API**, duas propriedades que sustentam a credibilidade de
κ/consistência, lendo os `llm_interactions_parte*.json` da execução:
1. **Taxa de fallback do parser** (`malformed=True`) — se alta, os rótulos são ruído do
   hash MD5 e as métricas ficam comprometidas.
2. **Determinismo de `T=0`** — mede a fração de pares (prompt, ponto) idênticos cuja
   resposta divergiu entre repetições ("taxa de flip"). `T=0` **não** é determinístico
   (tipicamente ~5% de flip); reportar esse número é honestidade experimental.
3. **Breakdown por (provider, modelo)** — fallback e flip por modelo, mais os snapshots
   resolvidos (`model_resolved`) e provedores de inferência observados (OpenRouter).
   A comparação de flip rates entre modelos é um resultado publicável por si só.

```bash
python src/audit_interactions.py                 # audita a execução COMPLETA mais recente
python src/audit_interactions.py execucao_2026-07-05_12-34-52_completa
python src/audit_interactions.py --json          # saída estruturada (CI/log)
```
Sai com código 1 se a taxa de malformadas ultrapassar `--max-malformed` (5% padrão).
O teste `tests/test_audit_interactions.py` roda esta auditoria na execução mais recente.

## Leitura de Resultados

**IMPORTANTE:** Antes de responder qualquer pergunta sobre resultados ou análise, leia sempre
os arquivos da **última execução COMPLETA disponível** — somente pastas com sufixo
`_completa`; pastas `_smoke` e `_apenas-*` NUNCA servem de fonte.

A última execução completa está em: `execucao_2026-07-05_12-34-52_completa/`

O padrão de nome das pastas é `execucao_YYYY-MM-DD_HH-MM-SS_{completa|smoke|apenas-<blocos>}/`.
Se houver pastas `_completa` mais recentes, leia a de timestamp mais alto. (Pastas legadas
sem sufixo: confirme no `log_execucao.txt` que não são `--rapido`.)

Arquivos prioritários para leitura:
1. `log_execucao.txt` — visão completa dos resultados
2. `bloco1_phases_abc_{timestamp}.csv` — métricas das Fases A-C (Bloco 1)
3. `bloco2_phase_e_{timestamp}.csv` — métricas da Fase E (Bloco 2; coluna `expert_name`)
4. `bloco1_oracle_validation_{timestamp}.csv` / `bloco1_oracle_meialua_*.csv` — validação dos algoritmos
5. `bloco1_algorithm_comparison_{timestamp}.csv` — comparação Perceptron × NNLS
6. `bloco1_r3r4_comparison_{timestamp}.csv` — R2/R3/R4 no Problema A
7. `bloco2_dilution_*.csv`, `bloco2_example_order_*.csv`, `bloco2_classical_baselines_*.csv` — auxiliares do Bloco 2
8. `bloco23_external_phase_a_*.csv`, `bloco23_external_phase_e_*.csv` — pipelines externos (meia-lua, peso×altura);
   para o Bloco 3 (`problem_name == homem_mulher`) use as cópias `*_corrigido_hm*.csv`
9. `final_cross_linearity__<alias>.csv` — síntese cruzada por modelo

## Referencias bibliograficas

Leia as referencias caso precise referencias apresentações e artigos.

Todas as referencias bibliograficas estão no arquivo `trabalhos_referencias.txt`.
PDFs de artigos citados estão em `artigos_referenciados/`.

## Dependências

```
openai          # SDK (endpoint compatível) usado para o OpenRouter — sync + async
python-dotenv   # Carrega .env com chaves de API
numpy           # Álgebra linear
matplotlib      # Visualizações
pandas          # Exportação CSV
scikit-learn    # Cohen's Kappa, F1, k-NN, LR, SVM
scipy           # nnls (NNLS), stats (Wilcoxon)
```

## Notas de Design

- **Métrica diagonal:** Simplificação proposital para tratabilidade (O(d) parâmetros vs O(d²))
- **Dados 2D sintéticos:** Permitem visualização; a dimensionalidade extra vem das projeções R3/R4 e o caso real é o peso × altura (Problema G)
- **Dois algoritmos de otimização inversa:** Perceptron (com hiperparâmetros) + NNLS (sem) — garante robustez do método de estimação. LP Max-Margin foi removido (ver `reunioes_orientador/reuniao_2026_04_30/plano_trabalho.md` item 12).
- **Oracle Validation:** Dá um piso de sanidade — se os algoritmos falharem em recuperar W sintético conhecido, qualquer achado sobre o LLM fica comprometido
- **Concorrência assíncrona (MAX_CONCURRENCY=10):** Reduz tempo de coleta do LLM em ~10× via `asyncio.Semaphore`
- **Fallback por hash MD5:** Respostas malformadas após retries são mapeadas via `hashlib.md5` de coordenadas (substituiu aritmética modular para não introduzir viés geométrico)
- **Temperatura = 0.0:** Minimiza estocasticidade para isolar o critério de decisão (não garante determinismo — ver Reunião 1)
- **Nomes de classe variados:** `["A"/"B", "0"/"1", "Positivo"/"Negativo", "Azul"/"Vermelho"]` — detecta viés semântico
- **Nomes de classe invertidos:** `["B"/"A", "1"/"0", ...]` — detecta viés de posição/ordem
- **Nomes de features semânticos:** `["altura"/"peso", "feature_1"/"feature_2"]` — detecta viés semântico nas variáveis
- **Parser de 8 camadas (0–7):** Estratégia robusta para lidar com respostas malformadas do LLM
- **3 sementes aleatórias (× 3 repetições onde há sorteio de exemplos, `reps_para`):** Garante robustez estatística
- **Análise estatística:** Bootstrap CI (10k reamostragens), Wilcoxon signed-rank, Cohen's d
- **Projeções R3/R4:** x3=x1·x2 (R3) e x1², x2² (R4) simulam kernel quadrático sem sair do mundo linear
- **Painéis individuais:** cada figura combinada também é salva como PNGs separados (`<base>_<painel>__<alias>.png`), úteis para apresentações
- **Dados por seed salvos em CSV:** reprodutibilidade bit-a-bit independente do código
- **Terminologia:** Usar "Ŵ_LLM estimada/inferida" (não "W aprendida") para a métrica da Fase A
