# CLAUDE.md — Projeto de Mestrado: Explicando Decisões de LLMs via Otimização Inversa

## Visão Geral

Pesquisa que investiga se LLMs possuem um processo de tomada de decisão **estável e aprendível**.
A hipótese central: se um LLM tem critérios de decisão consistentes, é possível aprender
uma métrica matemática (W) a partir de suas classificações e aplicá-la para prever
classificações em problemas diferentes (transferência de conhecimento).

**Abordagem:** Otimização inversa — aprender a função objetivo implícita do LLM a partir
de suas decisões observadas, usando métricas de Mahalanobis diagonais e Perceptron Estruturado.

## Estrutura de Pastas

```
mestrado/
├── src/
│   ├── dissertacao_mestrado.py       # Script principal (~4000+ linhas)
│   ├── relaxed_perceptron.py         # Classe RelaxedPerceptron (otimização inversa)
│   └── least_squares_inverse.py      # Classe LeastSquaresInverse (otimização inversa)
├── requirements.txt              # Dependências Python
├── CLAUDE.md                     # Este arquivo
├── .venv/                        # Ambiente virtual Python
├── .claude/
│   └── settings.local.json       # Configurações locais do Claude Code
├── reunioes_orientador/           # Transcrições e planos de reuniões com orientador
│   └── reuniao_13 03 2024/
│       ├── reuniao...transcricao.json
│       ├── emails.txt
│       ├── plano_trabalho.md
│       └── plano_implementacao.md
└── execucao_YYYY-MM-DD_HH-MM-SS/ # Pasta criada a cada execução (timestamp automático)
    ├── 01_all_three_problems.png
    ├── 02_consistency_extended.png
    ├── 03_class_names_effect.png
    ├── 05_seed_comparison.png
    ├── 06_problem_d_expert.png
    ├── 07_phase_d_example_strategies.png
    ├── 08_phase_d_learning_curve.png
    ├── 09_phase_d_strategy_comparison.png
    ├── 10_w_distribution.png
    ├── 11_metric_errors_phase_a_seed*.png
    ├── 12_dilution_experiment.png
    ├── 13_r3_vs_r2.png
    ├── 14_algorithm_comparison.png
    ├── 15_class_order_bias.png
    ├── results_phases_abc_v4_{timestamp}.csv
    ├── results_phase_d_v4_{timestamp}.csv
    ├── results_dilution_v4_{timestamp}.csv
    └── log_execucao.txt
```

## Como Executar

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar a chave de API (OpenAI por padrão)
export OPENAI_API_KEY="sk-..."

# Executar o experimento completo
python src/dissertacao_mestrado.py
```

A execução cria automaticamente uma pasta `execucao_YYYY-MM-DD_HH-MM-SS/` com todos os outputs.

## Configuração Principal

Constantes no topo de `src/dissertacao_mestrado.py`:

| Constante | Valor padrão | Descrição |
|---|---|---|
| `RANDOM_SEED` | `42` | Semente padrão |
| `RANDOM_SEEDS` | `[42, 123, 7, 256, 999]` | 5 seeds para robustez |
| `N_SAMPLES_PROBLEM_A` | `150` | Amostras no Problema A |
| `N_SAMPLES_PROBLEM_B/C` | `100` | Amostras nos Problemas B e C |
| `N_SAMPLES_PROBLEM_D` | `150` | Amostras no Problema D |
| `FEW_SHOT_SIZES` | `[0, 5, 10]` | Tamanhos few-shot fases A-C |
| `FEW_SHOT_SIZES_PHASE_D` | `[0, 3, 5, 10, 20]` | Tamanhos few-shot fase D |
| `N_REPETICOES` | `2` | Repetições por configuração |
| `EXAMPLE_STRATEGIES` | `["easy","hard","mixed","random"]` | Estratégias fase D |
| `EXPERT_W` | `[0.3, 1.5]` | Pesos do especialista padrão (x2 dominante) |

### Flags de Execução

Controlam quais experimentos rodar sem precisar comentar/descomentar código:

| Flag | Padrão | Descrição |
|---|---|---|
| `RUN_PHASES_ABC` | `True` | Fases A-C padrão |
| `RUN_PHASE_D` | `True` | Fase D padrão |
| `RUN_CLASS_ORDER_BIAS` | `True` | Teste de inversão de ordem das classes |
| `RUN_FEATURE_NAMES` | `True` | Teste de nomes semânticos nas features |
| `RUN_DILUTION` | `True` | Experimento de diluição |
| `RUN_R3_EXPERIMENT` | `True` | Projeção R3 (kernel quadrático, x3=x1*x2) |
| `RUN_MULTIPLE_EXPERTS` | `True` | Múltiplas configs de expert na Fase D |
| `RUN_ALGORITHM_COMPARISON` | `True` | Segundo algoritmo (Mínimos Quadrados) |

### Configurações de Experts Múltiplos (Fase D)

```python
EXPERT_CONFIGS = [
    {"name": "aniso_x2",  "w": [0.3, 1.5],  "desc": "x2 dominante (original)"},
    {"name": "aniso_x1",  "w": [1.5, 0.3],  "desc": "x1 dominante (invertido)"},
    {"name": "euclidean", "w": [1.0, 1.0],  "desc": "pesos iguais (Euclidiana)"},
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

**Modelos suportados** (alternar no código):
- `gpt-4o-mini` (ativo por padrão)
- Claude (comentado, pronto para ativar)
- Gemini 2.0 Flash (comentado, pronto para ativar)

## Arquitetura do Código

### Estruturas de Dados Principais

```python
@dataclass
class ResultadoExperimento        # Resultado das fases A-C
    # consistencia, kappa, f1 para Problemas B e C
    # pesos aprendidos, gamma, contagens de desacordo
    # w_ratio, feature_names (novos campos v4)

@dataclass
class ResultadoPhaseDExperimento  # Resultado da fase D
    # acurácia LLM vs. especialista, kappa, f1
    # estratégia de exemplos usada
    # expert_name (novo campo v4)

@dataclass
class LearnedMetric               # Métrica Mahalanobis aprendida
    # vetor w, centroides, gamma, problema de origem
```

### Funções Críticas

| Função | Propósito |
|---|---|
| `create_problem_*()` | Gera dados sintéticos 2D para cada problema |
| `llm_classify_point()` | Chama a API do LLM para classificar um ponto |
| `train_relaxed_perceptron()` | Perceptron Estruturado com relaxação de margem |
| `train_least_squares_inverse()` | Algoritmo alternativo via NNLS (scipy) |
| `compute_consistency_metrics()` | Mede consistência LLM vs. métrica aprendida |
| `phase_a_learn_metric()` | Fase A: aprende Ŵ_LLM a partir das decisões |
| `phase_consistency_test()` | Fases B/C: testa consistência em novos problemas |
| `phase_d_llm_as_learner()` | Fase D: LLM aprendendo do especialista |
| `select_examples_by_strategy()` | Estratégias de seleção (easy/hard/mixed/random) |
| `select_examples_dilution()` | Seleção para experimento de diluição |
| `augment_to_r3()` | Projeção R3: adiciona x3 = x1 * x2 |
| `build_prompt_zero_shot()` | Prompt zero-shot (suporta features extras) |
| `build_prompt_few_shot()` | Prompt few-shot (suporta features extras) |
| `parse_llm_response()` | Parser de 7 camadas para respostas do LLM |

### Fluxo de Execução (v4.0)

```
 1. Criar pasta de execução (timestamp)
 2. Gerar dados sintéticos (4 problemas)
 3. Visualização inicial (scatter plots)
 4. FASE A: LLM classifica sem exemplos → aprende Ŵ_LLM
 5. FASE B: Testa Ŵ_LLM em Problema B (geometria diferente)
 6. FASE C: Testa Ŵ_LLM em Problema C (distorção geométrica maior)
 7. [RUN_CLASS_ORDER_BIAS] Teste com classes invertidas (B/A, 1/0, etc.)
 8. [RUN_FEATURE_NAMES] Teste com nomes semânticos nas features
 9. [RUN_ALGORITHM_COMPARISON] Mínimos Quadrados vs. Perceptron
10. FASE D: LLM aprende métrica do(s) especialista(s) externo(s)
11. [RUN_MULTIPLE_EXPERTS] Itera sobre 3 configs de expert
12. [RUN_DILUTION] Experimento de diluição (hard fixos + easy progressivos)
13. [RUN_R3_EXPERIMENT] Projeção R3 (x3=x1*x2) e comparação 2 vs 3 features
14. Gerar 15+ gráficos de análise
15. Exportar CSVs com resultados numéricos
16. Salvar log completo da execução
```

## Hipóteses do Experimento

| Hipótese | Descrição |
|---|---|
| **H1** (Consistência) | O LLM mantém o mesmo critério decisório implícito. A métrica Ŵ_LLM **estimada** prevê classificações nos Problemas B e C (Kappa > 0.7). |
| **H2** (Few-shot amplifica) | Exemplos rotulados pela métrica W aumentam a concordância LLM-métrica. |
| **H3** (LLM como aprendiz) | O LLM consegue aprender o critério de um perito externo via few-shot. |
| **H4** (Exemplos difíceis) | Exemplos "hard" (próximos à fronteira) são **mais informativos** que "easy", especialmente com poucos exemplos. |
| **H5** (Estabilidade) | O comportamento é reproduzível entre execuções (5 sementes, múltiplas repetições). |

## Fases Experimentais

### Fase A — Aprendizado de Métrica (Zero-Shot)
O LLM classifica 150 pontos 2D sem exemplos. A partir dessas classificações,
aprende-se uma **métrica de Mahalanobis diagonal** (Ŵ_LLM) via Perceptron Estruturado com margem.

### Fase B — Teste de Consistência 1
Aplica a métrica Ŵ_LLM aprendida na Fase A ao Problema B (centroides deslocados verticalmente).
Testa se o LLM mantém consistência em distribuições não vistas.

### Fase C — Teste de Consistência 2
Similar à Fase B, com distorção geométrica mais severa. Validação adicional da
generalização da métrica.

### Fase D — LLM como Aprendiz (v4.0)
**Papel invertido:** Um especialista externo com métrica W_expert conhecida rotula os dados.
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

### Novos Experimentos (v4.0)

#### Viés de Ordem das Classes
Testa se a ordem "A ou B" vs "B ou A" altera o resultado do LLM.
Classes testadas: pares originais + pares invertidos.

#### Nomes Semânticos de Features
Testa se usar "altura"/"peso" ao invés de "x1"/"x2" no prompt altera os pesos aprendidos.

#### Experimento de Diluição
Fixa 4 exemplos hard e adiciona easy progressivamente (0, 2, 4, 10, 16, 20).
Verifica se em algum ponto os fáceis "diluem" os difíceis e a performance deteriora.

#### Projeção R3 (Kernel Quadrático)
Cria terceira feature x3 = x1 * x2. Testa se o LLM captura a importância dessa feature de interação.
Compara: LLM com 2 features vs LLM com 3 features vs Métrica com 3 features.

#### Comparação de Algoritmos
Compara Perceptron Estruturado vs Mínimos Quadrados (NNLS) para aprender W.
Demonstra que os resultados são robustos ao método de estimação.

## Outputs por Execução

### Gráficos (PNG)
| Arquivo | Conteúdo |
|---|---|
| `01_all_three_problems.png` | Scatter plots dos Problemas A, B, C |
| `02_consistency_extended.png` | Consistência, Kappa, F1, distribuição de métricas |
| `03_class_names_effect.png` | Consistência por variação de nomes de classes |
| `05_seed_comparison.png` | Robustez entre seeds |
| `06_problem_d_expert.png` | Problema D com fronteira de decisão do especialista |
| `07_phase_d_example_strategies.png` | Distribuição espacial dos exemplos por estratégia |
| `08_phase_d_learning_curve.png` | Acurácia vs. número de exemplos por estratégia |
| `09_phase_d_strategy_comparison.png` | Comparação de estratégias em múltiplas métricas |
| `10_w_distribution.png` | **NOVO:** Distribuição de Ŵ_LLM entre seeds/repetições |
| `11_metric_errors_phase_a_seed*.png` | **NOVO:** Erros da métrica vs. rotulação do LLM na Fase A |
| `12_dilution_experiment.png` | **NOVO:** Experimento de diluição (hard fixos + easy) |
| `13_r3_vs_r2.png` | **NOVO:** Comparação 2 features vs 3 features (R3) |
| `14_algorithm_comparison.png` | **NOVO:** Perceptron vs. Mínimos Quadrados |
| `15_class_order_bias.png` | **NOVO:** Viés de ordem/posição das classes |

### CSVs
- `results_phases_abc_v4_{timestamp}.csv` — 30+ colunas: fases A-C (inclui w_ratio, feature_names)
- `results_phase_d_v4_{timestamp}.csv` — 20+ colunas: fase D (inclui expert_name)
- `results_dilution_v4_{timestamp}.csv` — **NOVO:** resultados do experimento de diluição

### Log
- `log_execucao.txt` — Transcript completo com detalhes algorítmicos e métricas

## Leitura de Resultados

**IMPORTANTE:** Antes de responder qualquer pergunta sobre resultados ou análise do experimento, leia sempre os arquivos da **última execução disponível**.

A última execução está em: `execucao_2026-03-03_20-25-01/`

O padrão de nome das pastas é `execucao_YYYY-MM-DD_HH-MM-SS/`. Se houver pastas mais recentes, leia a de timestamp mais alto.

Arquivos prioritários para leitura:
1. `log_execucao.txt` — visão completa dos resultados
2. `results_phases_abc_v4_{timestamp}.csv` — métricas fases A-C
3. `results_phase_d_v4_{timestamp}.csv` — métricas fase D
4. `results_dilution_v4_{timestamp}.csv` — métricas do experimento de diluição

## Referencias bibliograficas

Leia as referencias caso precise referencias apresentações e artigos.

Todas as referencias bibliograficas estão no arquivo trabalhos_referencias.txt


## Dependências

```
openai          # API OpenAI (GPT-4o-mini)
numpy           # Álgebra linear
matplotlib      # Visualizações
pandas          # Exportação CSV
scikit-learn    # Cohen's Kappa, F1-score
scipy           # Operações científicas (inclui NNLS para algoritmo alternativo)
```

## Notas de Design

- **Métrica diagonal:** Simplificação proposital para tratabilidade (O(d) parâmetros vs O(d²))
- **Dados 2D sintéticos:** Permite visualização; próximo passo: 3D e dados reais (Iris)
- **Temperatura = 0.0:** Minimiza estocasticidade para isolar o critério de decisão
- **Nomes de classe variados:** `["A"/"B", "0"/"1", "Positivo"/"Negativo", "Azul"/"Vermelho"]` — detecta viés semântico
- **Nomes de classe invertidos:** `["B"/"A", "1"/"0", ...]` — detecta viés de posição/ordem
- **Nomes de features semânticos:** `["altura"/"peso", "feature_1"/"feature_2"]` — detecta viés semântico nas variáveis
- **Parser de 7 camadas:** Estratégia robusta para lidar com respostas malformadas do LLM
- **5 sementes aleatórias:** Garante robustez estatística dos resultados
- **Dois algoritmos de otimização:** Perceptron Estruturado + Mínimos Quadrados (NNLS) — demonstra robustez ao método
- **Projeção R3:** x3=x1*x2 simula kernel quadrático sem sair do mundo linear
- **Terminologia:** Usar "Ŵ_LLM estimada/inferida" (não "W aprendida") para a métrica da Fase A
