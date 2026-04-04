# Plano de Implementacao — Modificacoes no Codigo

Baseado nas solicitacoes do Prof. Raul na reuniao de 13/03/2024 e nos emails pos-reuniao.
Cada item mapeia o pedido do professor para as mudancas especificas no arquivo `Mestrado Trabalho.py`.

---

## BLOCO 1 — Mudancas Rapidas (Textuais e Configuracao)

### 1.1 Reescrever Hipoteses H1 e H4
**Arquivo:** `Mestrado Trabalho.py`
**Linhas:** 53-75 (bloco de hipoteses no docstring)

**H1 (linha 53-56):** Trocar "métrica W aprendida" por "métrica W_LLM estimada/inferida".
- Substituir todas as ocorrencias de "W aprendido" por "Ŵ_LLM" ou "W estimado" no codigo inteiro (comentarios, prints, nomes de variaveis de log)

**H4 (linha 69-72):** Inverter a hipotese. Texto atual diz que exemplos hard "transmitem mais informacao". O resultado confirmou isso, entao na verdade a hipotese ja esta alinhada com os dados. O professor pediu para reescrever com mais precisao:
- Novo texto: "Exemplos proximos a fronteira de decisao ('hard') sao mais informativos para o LLM aprender o criterio do perito. A estrategia 'hard' supera 'easy' e 'random' em acuracia, especialmente com poucos exemplos."

### 1.2 Aumentar Sementes Aleatorias
**Linha 338:** `RANDOM_SEEDS = [42, 123]` → `RANDOM_SEEDS = [42, 123, 7, 256, 999]`

### 1.3 Variar W do Expert na Fase D
**Linha 367-368:** Atualmente fixo em `EXPERT_W = np.array([0.3, 1.5])`

**Criar lista de experts:**
```python
EXPERT_CONFIGS = [
    {"name": "aniso_x2",   "w": np.array([0.3, 1.5]),  "desc": "x2 dominante (original)"},
    {"name": "aniso_x1",   "w": np.array([1.5, 0.3]),  "desc": "x1 dominante (invertido)"},
    {"name": "euclidean",  "w": np.array([1.0, 1.0]),  "desc": "pesos iguais (Euclidiana)"},
]
```
**Impacto:** O loop da Fase D (linha ~2964-3033) precisa iterar sobre cada config de expert. O `ResultadoPhaseDExperimento` ja tem campo `expert_w`, entao o CSV ja captura isso. Adicionar campo `expert_name` ao dataclass.

---

## BLOCO 2 — Exibir e Analisar W Estimado (Fase A)

### 2.1 Mostrar W Estimado nos Resultados
**Onde:** Funcao `phase_a_learn_metric()` (linha 1307) e `print_final_analysis()` (linha 2660)

**Adicionar apos linha ~1380 (fim da Fase A):**
```python
print(f"  >>> W estimado (Ŵ_LLM): [{w[0]:.4f}, {w[1]:.4f}]")
print(f"  >>> Razao w1/w2: {w[0]/w[1]:.4f}" if w[1] != 0 else "  >>> w2 = 0")
print(f"  >>> Gamma otimo: {gamma:.4f}")
```

### 2.2 Distribuicao de W entre Repeticoes e Sementes
**Nova funcao:** `plot_w_distribution(resultados, filename)`
**Onde inserir:** Apos as funcoes de plot existentes (~linha 2540)

**Logica:**
- Extrair `w_aprendido` de todos os `ResultadoExperimento` (campo `w_aprendido`, linhas 443-444)
- Plot 1: Scatter plot w0 vs w1 colorido por seed
- Plot 2: Boxplot de w0 e w1 por seed
- Plot 3: Razao w0/w1 por seed com barras de erro

**Chamar em:** Bloco de visualizacao (linha ~3044), salvar como `10_w_distribution.png`

### 2.3 Visualizar Erros da Metrica vs Rotulacao LLM (Fase A)
**Nova funcao:** `plot_metric_errors_phase_a(X, y_llm, y_metric, w, centroids, filename)`
**Onde inserir:** Apos `plot_disagreement_analysis()` (~linha 2078)

**Logica:**
- Scatter plot dos pontos do Problema A
- Pontos onde LLM e metrica concordam: cor da classe, marcador normal
- Pontos onde discordam: marcador 'X' vermelho, maior
- Desenhar fronteira de decisao da metrica W
- Imprimir % de erro por regiao (perto/longe da fronteira)

**Dados necessarios:** `y_llm_train_a_cache` ja existe no `run_complete_experiment()` (linha 1825). Precisa propagar para o nivel do main para plotar.

**Chamar em:** Bloco de visualizacao, salvar como `11_metric_errors_phase_a.png`

---

## BLOCO 3 — Verificar e Plotar Selecao de Exemplos

### 3.1 Plotar Exemplos Selecionados por Estrategia
**Funcao existente:** `plot_phase_d_example_locations()` (linha 2480) — ja faz isso parcialmente.

**Verificar:** Ler a funcao e confirmar que ela plota os pontos selecionados para CADA estrategia (easy/hard/mixed/random) com destaque visual. Se ja faz, apenas confirmar que esta funcionando e incluir no relatorio. Se nao, adicionar:
- Marcadores maiores para pontos selecionados
- Fronteira de decisao do expert no fundo
- Legenda clara

**Status:** Provavelmente ja implementado. Validar e ajustar se necessario.

---

## BLOCO 4 — Testes de Vies Semantico

### 4.1 Testar Inversao da Ordem das Classes
**Linha 374-379:** `NOMES_CLASSES` ja testa variacao de nomes, mas NAO testa inversao de ordem.

**Adicionar pares invertidos:**
```python
NOMES_CLASSES = [
    ("A", "B"), ("B", "A"),           # Inversao de ordem
    ("0", "1"), ("1", "0"),           # Inversao de ordem
    ("Positivo", "Negativo"), ("Negativo", "Positivo"),
    ("Azul", "Vermelho"), ("Vermelho", "Azul"),
]
```

**Impacto:** O loop principal ja itera sobre `NOMES_CLASSES` (verificar no main ~linha 2870+). Isso vai dobrar o numero de configs, mas o professor pediu especificamente.

**ATENCAO:** Isso duplica o numero de chamadas API. Considerar fazer isso em execucao separada ou com flag de controle.

### 4.2 Testar Nomes Semanticos nas Features
**Atualmente:** Prompts usam "x1" e "x2" como nomes de features (funcoes `build_prompt_*`, linhas 757, 772).

**Criar lista de nomes de features:**
```python
NOMES_FEATURES = [
    ("x1", "x2"),                    # Original (neutro)
    ("altura", "peso"),              # Semantico
    ("feature_1", "feature_2"),      # Tecnico
]
```

**Modificar `build_prompt_zero_shot()` e `build_prompt_few_shot()`:**
- Adicionar parametros `nome_feature_0="x1"` e `nome_feature_1="x2"`
- Substituir "x1" e "x2" no texto do prompt pelos nomes passados

**Modificar `llm_classify_point()`** (linha 913) e toda a cadeia de chamadas para propagar os nomes de features.

**Impacto:** Multiplica configs por 3. Considerar flag separado ou execucao isolada.

**NOTA:** O professor disse "nao criar 200 testes". Sugestao: fazer features semanticas como experimento separado, com 1 seed, 1 par de classes, para nao explodir combinatorias.

---

## BLOCO 5 — Experimento de Diluicao

### 5.1 Nova Funcao: Experimento de Diluicao
**Nova funcao:** `run_dilution_experiment(X_d, y_expert_d, expert_w, expert_centroids, ...)`
**Onde inserir:** Apos `phase_d_llm_as_learner()` (~linha 1820)

**Logica:**
1. Selecionar 3 exemplos hard (fixos)
2. Loop: adicionar N exemplos easy progressivamente: N = [0, 1, 2, 5, 10, 15, 20]
   - Total de exemplos: [3, 4, 5, 8, 13, 18, 23]
3. Para cada config, chamar `phase_d_llm_as_learner()` com os exemplos combinados
4. Registrar acuracia em cada ponto

**Nova funcao de selecao:** `select_examples_dilution(X, y_expert, expert_w, expert_centroids, n_hard_fixed, n_easy_added, ...)`
- Seleciona os top-3 hard
- Adiciona N easy (excluindo os hard ja selecionados)

**Novo plot:** `plot_dilution_experiment(results, filename)`
- Eixo X: numero total de exemplos (3, 4, 5, 8, 13, 18, 23)
- Eixo Y: acuracia LLM vs expert
- Linha de referencia: performance com 3 hard puros (sem diluicao)

**Chamar em:** Bloco da Fase D no main, salvar como `12_dilution_experiment.png`

**Novo dataclass ou reutilizar `ResultadoPhaseDExperimento`** com strategy="dilution_N" onde N = numero de easy adicionados.

---

## BLOCO 6 — Projecao R3 (Kernel Quadratico) — Email do Professor

### 6.1 Criar Feature de Interacao x3 = x1 * x2
**Novas funcoes de geracao de dados:**

**Modificar `create_problem_*()` ou criar wrappers:**
```python
def augment_to_r3(X):
    """Adiciona feature x3 = x1 * x2"""
    x3 = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, x3])
```

**Impacto em toda a pipeline:**
- `compute_centroids()` (linha 1104): funciona para qualquer dimensao ✓
- `d_W()` (linha 1095): funciona para qualquer dimensao ✓
- `train_relaxed_perceptron()` (linha 1115): funciona para qualquer dimensao ✓
- `predict_with_metric()` (linha 1247): funciona para qualquer dimensao ✓
- `build_prompt_*()`: PRECISA MUDAR — adicionar x3 no prompt

**Modificar prompts:**
```python
def build_prompt_zero_shot_3d(x1, x2, x3, nome_classe_0, nome_classe_1):
    # "Classifique o ponto (x1=..., x2=..., x3=...) como ..."
```

**Modificar `llm_classify_point()`** para aceitar numero variavel de features.

### 6.2 Teste de Concordancia: 2 Features vs 3 Features
**Conforme Email 2 do professor:**
- Aprender W = (w1, w2, w3) com 3 features
- Testar LLM com APENAS 2 features (x1, x2) usando dados rotulados por W com 3 features
- Comparar com LLM usando 3 features

**Nova funcao:** `run_r3_comparison_experiment(...)`
1. Gerar dados 2D normais
2. Augmentar para 3D com x3 = x1*x2
3. Fase A com 3 features → aprende W3 = (w1, w2, w3)
4. Rotular dados de teste com W3
5. Teste A: LLM classifica com 2 features (x1, x2) dados rotulados por W3
6. Teste B: LLM classifica com 3 features (x1, x2, x3) dados rotulados por W3
7. Comparar acuracias

**Impacto:** Este e um experimento separado. Sugestao: criar flag `RUN_R3_EXPERIMENT = True/False` no topo do arquivo.

---

## BLOCO 7 — Segundo Algoritmo de Otimizacao Inversa

### 7.1 Implementar Algoritmo Alternativo
**Referencia:** Ahuja & Orlin (otimizacao inversa classica). Aguardando artigos do professor.

**Abordagem sugerida enquanto espera artigos:** Implementar um metodo mais simples como baseline:
- **Least Squares Inverse Optimization:** Encontrar W que minimiza erro quadratico entre distancias observadas e fronteira de decisao
- Ou **SVM com kernel linear** como proxy para aprender W

**Nova funcao:** `train_alternative_method(X, y, centroids, method="least_squares")`
**Onde inserir:** Apos `train_relaxed_perceptron()` (~linha 1245)

**Retorna:** Mesmo formato: `(w_learned, gamma_optimal)`

**Modificar `phase_a_learn_metric()`** para aceitar parametro `method` e chamar o algoritmo correspondente.

**Novo plot:** `plot_algorithm_comparison(results_perceptron, results_alternative, filename)`
- Comparar W aprendido por cada algoritmo
- Comparar fidelidade e consistencia

**NOTA:** Este item depende dos artigos que o professor vai enviar. Implementar a infraestrutura (parametro de metodo, plot comparativo) e deixar pronto para plugar o algoritmo quando chegar.

---

## BLOCO 8 — Melhorias em Visualizacao

### 8.1 Novos Graficos a Adicionar

| # | Grafico | Funcao | Arquivo |
|---|---------|--------|---------|
| 10 | Distribuicao de W entre seeds/repeticoes | `plot_w_distribution()` | `10_w_distribution.png` |
| 11 | Erros da metrica vs LLM na Fase A | `plot_metric_errors_phase_a()` | `11_metric_errors_phase_a.png` |
| 12 | Experimento de diluicao | `plot_dilution_experiment()` | `12_dilution_experiment.png` |
| 13 | Comparacao R3 vs R2 | `plot_r3_comparison()` | `13_r3_vs_r2.png` |
| 14 | Comparacao de algoritmos | `plot_algorithm_comparison()` | `14_algorithm_comparison.png` |
| 15 | Vies de ordem das classes | `plot_class_order_bias()` | `15_class_order_bias.png` |

---

## BLOCO 9 — Modificacoes nos Dataclasses e CSVs

### 9.1 ResultadoPhaseDExperimento — Adicionar Campos
**Linha 462-492:**
```python
expert_name: str = "aniso_x2"     # Nome da config do expert
```

### 9.2 ResultadoExperimento — Adicionar Campos
**Linha 415-458:**
```python
w_ratio: float = 0.0              # Razao w0/w1 para analise rapida
feature_names: Tuple[str, str] = ("x1", "x2")  # Nomes das features usadas
```

### 9.3 CSV — Novas Colunas
- `expert_name` no CSV da Fase D
- `w_ratio`, `feature_name_0`, `feature_name_1` no CSV das Fases A-C

---

## ORDEM DE IMPLEMENTACAO RECOMENDADA

Agrupado por impacto e dependencias:

### Sprint 1 — Mudancas Minimas, Resultados Imediatos (1-2 dias)
1. **1.1** Reescrever hipoteses H1 e H4 (textual)
2. **1.2** Aumentar sementes para 5
3. **2.1** Exibir W estimado nos prints da Fase A
4. **3.1** Validar que plot de exemplos ja funciona

### Sprint 2 — Novos Graficos e Analises (2-3 dias)
5. **2.2** Plot distribuicao de W entre seeds
6. **2.3** Plot erros metrica vs LLM na Fase A
7. **9.1-9.3** Atualizar dataclasses e CSVs

### Sprint 3 — Novos Experimentos (3-4 dias)
8. **1.3** Variar W do expert (3 configs)
9. **4.1** Testar inversao da ordem das classes
10. **5.1** Experimento de diluicao

### Sprint 4 — Features Avancadas (3-5 dias)
11. **4.2** Nomes semanticos nas features
12. **6.1-6.2** Projecao R3 (kernel quadratico)
13. **7.1** Segundo algoritmo (quando artigos chegarem)
14. **8.1** Novos graficos finais

### Total Estimado: ~10-14 dias de implementacao

---

## FLAGS DE CONTROLE SUGERIDAS

Adicionar no topo do arquivo para controlar quais experimentos rodar sem comentar/descomentar codigo:

```python
# === FLAGS DE EXECUCAO ===
RUN_PHASES_ABC = True          # Fases A-C padrao
RUN_PHASE_D = True             # Fase D padrao
RUN_CLASS_ORDER_BIAS = False   # Teste de inversao de ordem (4.1)
RUN_FEATURE_NAMES = False      # Teste de nomes semanticos (4.2)
RUN_DILUTION = False           # Experimento de diluicao (5.1)
RUN_R3_EXPERIMENT = False      # Projecao R3 (6.1-6.2)
RUN_ALGORITHM_COMPARISON = False  # Segundo algoritmo (7.1)
EXPERT_CONFIGS_TO_RUN = ["aniso_x2"]  # Quais experts rodar (1.3)
```
