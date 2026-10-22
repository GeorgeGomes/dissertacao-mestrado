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
- **Bloco 1 — LLM como ESPECIALISTA:** Problemas A (linear; centróides (−2; −0,5) e (2; 0,5) desde 15/10/2026), B (linear, centróides (−2; 1,5) e (2; −1,5), rotação horária em relação ao A), C (linear, (−2; −1,5) e (2; 1,5), rotação anti-horária), D (meia-lua não-linear). O LLM rotula (Passo 2 zero-shot; Passo 3 few-shot do rótulo real em A e D) e a métrica W é estimada via Perceptron+NNLS (Passo 4), com teto (Passo 5) e eleição da melhor métrica (Passo 6 do A); B e C recebem as métricas do A (Passo 3) e exemplos rotulados por elas (Passo 4). Inclui Oráculo, R2→R3→R4, vieses e variantes de prompt.
- **Bloco 2 — LLM como APRENDIZ:** Problemas E (linear, especialista W=[0.3,1.5]) e F (meia-lua, especialista = GROUND TRUTH, desde 09/10/2026 de fato no código: `fg_executar(..., expert_labels=y)`, coluna `especialista_origem=rotulo_real`; o Passo 3 do F roda mesmo se o Passo 2 colapsar). Aprendizado em contexto: Passo 3 de E (estratégias), Passo 4 (diluição), Passo 5 (ordem dos exemplos), Passo 6 (baselines clássicos); no F, Passos 2 a 4. Veredito de H4 calculado dos dados.
- **Bloco 3 — Estudo de caso REAL:** Problema G, peso × altura (base do orientador, fronteira elíptica), em dois SUBPROBLEMAS com as mesmas 100 pessoas e a mesma divisão 70/30: **G1** (valores normalizados [-1,1], `problem_name` `G1_homem_mulher`/`G1_homem_mulher_classesAB`) e **G2** (prompt em kg e m via `peso_altura_kg_m.csv`, `G2_*`; métrica sempre estimada no espaço normalizado). Passos 2 a 5 nos 70 pontos de treino, os mesmos passos do Problema A (Metodologia IV.7 do orientador, 13/10/2026): zero-shot (Passo 2), few-shot com k ∈ {4, 10, 20, 40} exemplos do RÓTULO REAL (k/2 por classe, 3 repetições; Passo 3), estimação em R2/R3/R4 com Perceptron e NNLS mais o teto (Passo 4) e eleição da melhor métrica (Passo 5) (`fg_executar(..., com_passos_3_a_5=True)` chama `adg_passos_2_a_5` sobre o treino; CSV `bloco3_g_passos_2_a_5`, tabela `bloco3_table_06`, figura `bloco3_graphic_03`); Passo 6 com especialista = métrica ELEITA entre as de zero-shot e few-shot (`g_passo5_eleger_especialista`: maior acurácia vs rótulo real no treino, Perceptron; `especialista_origem=metrica`, colunas `especialista_regime`/`especialista_k`/`especialista_rep`; exemplos few-shot sorteados k/2 pelo RÓTULO REAL do treino, idênticos entre variantes de atributo); Passo 7 = Perceptron nos mesmos exemplos; + paradoxo do overfitting. O Problema F não tem few-shot do rótulo real na estimação: só zero-shot no Passo 2.

**Letras dos problemas** (não confundir com os passos, numerados dentro de cada problema): A, B, C lineares e D meia-lua
(Bloco 1); E especialista linear com 3 subproblemas E1/E2/E3 (especialistas [0,3;1,5], [1,5;0,3], [1;1]) e
F meia-lua (Bloco 2); G peso × altura com subproblemas G1/G2 (Bloco 3). Regra de nomes (08/10/2026):
letra = problema (dados + papel), número = subproblema (mesmos dados, outro rótulo ou outra
apresentação). O oráculo não é problema: é "Oráculo sobre E3 / E1 / E2" (sobre B até 13/10/2026). A meia-lua é
o mesmo dado em dois papéis (D: LLM especialista; F: LLM aprendiz), gerada por `create_problem_d_meialua`.

## Passos por problema (13/10/2026)

Regra: letra = problema; cada problema tem a SUA sequência de passos, numerada de 1 em diante,
só com os passos que existem nele. Nos textos visíveis (títulos, legendas, rodapés, log e
documentação) escreve-se "Problema A, Passo 3", "Passo 3 do Problema A" ou, em títulos curtos,
"A, Passo 3"; um asset que cobre vários passos é "Problema A, Passos 2 a 5". Os nomes antigos
(antes de 13/10/2026: "Fase A", "Fase B/C", "Fase E", "Fase A/E externa", "pipeline externo" e
"protocolo único da Fase A") deixaram de ser usados em texto visível (camada 1, decisão do autor,
13/10/2026). Em 13/10/2026 (camada 2) os identificadores internos de código também foram
renomeados pelo mesmo critério: funções (`adg_passos_2_a_5`, `e_passos_2_3_aprendiz`, `fg_executar`),
constantes (`K_EXEMPLOS_ROTULO_REAL`, `K_EXEMPLOS_APRENDIZ`, `RUN_PROBLEMA_E`, `RUN_PROBLEMAS_ABC`,
`RUN_PROBLEMA_D`), colunas de CSV (`k_passo4_bc`), valores de regime (`passo2_zero_shot`,
`passo3_exemplos_rotulo_real`, `passo5_teto`), tags de auditoria (`passo=2`, `passo=3`, ...),
nomes de CSV (`bloco1_ad_passos_2_a_5`, `bloco2_e_passos_2_3`, `bloco23_fg_passos_2_e_4`/`bloco23_fg_aprendiz`),
gráficos (`bloco3_graphic_03_G_passos_2_a_5_curva`), slugs de tabela, testes e ferramentas. O mapa
ÚNICO antigo → novo está em `renomeacao_passos_2026-10-13.md` (resumo na tabela "Equivalência de
identificadores" abaixo). Regra geral: nenhum nome novo contém `fase`, `phase`, `external`,
`pipeline`, `protocol`, `fewshot`/`few_shot` como etapa, `transfer` ou `oracle` (usa-se `oraculo`); o
nome diz PROBLEMA e PASSO quando é específico (prefixos `a_`, `bc_`, `d_`, `e_`, `f_`, `g_`, `fg_`,
`ad_`, `adg_`, `oraculo_`; sufixos `_passoN` ou `_passos_N_a_M`) e, quando é compartilhado, nome pelo
significado (`K_EXEMPLOS_ROTULO_REAL`). Siglas sem etapa ficam (`accuracy`, `fidelity`, `consistency`,
`kappa`, `f1`, `easy/hard/mixed/random`, `w_perc`, `n_features`).

**Problema A (gaussianas lineares)**
1. Gerar os 150 pontos (centróides (−2; −0,5) e (2; 0,5), σ = 1,2; até 14/10/2026 eram (−2, 0) e (2, 0): com x2 igual nos dois centros a diferença de distâncias não dependia de w2, a métrica estimada no A deixava w2 livre e H1 em B oscilava entre execuções; B e C mantêm as coordenadas de sempre).
2. O LLM rotula os 150 pontos em zero-shot.
3. O LLM rotula de novo com k exemplos do rótulo real no prompt, k ∈ {4, 10, 20, 40}, 3 sorteios por k.
4. Estimar a métrica Ŵ (Perceptron e NNLS, em R², R³ e R⁴) a partir de cada conjunto de rótulos do LLM; medir fidelidade.
5. Calcular o teto: W ajustado diretamente ao rótulo real.
6. Eleger a melhor métrica (maior fidelidade média) para a transferência.
Vieses (sobre o Passo 2 do A e os Passos 3 e 4 de B/C): nomes de classe, ordem das classes, variantes de prompt com k ∈ {0, 10}; nomes de atributo só zero-shot.

**Problemas B e C (transferência)**
1. Gerar os 100 pontos com os centróides rotacionados (B horária, 2,0 em relação ao A; C anti-horária, 1,0; coordenadas fixas (−2; ±1,5)/(2; ∓1,5)).
2. O LLM rotula os 100 pontos em zero-shot.
3. Aplicar cada métrica do A (só o W; centróides locais) e medir consistência, Cohen's κ e F1-Macro; destacar a métrica eleita no A.
4. O LLM rotula de novo com k exemplos rotulados pela métrica do A (k ∈ {4, 10, 20, 40}); medir se a consistência sobe (H2).

**Problema D (meia-lua, LLM como especialista)**
1. Gerar os 150 pontos da meia-lua.
2. O LLM rotula em zero-shot.
3. O LLM rotula com k exemplos do rótulo real, k ∈ {4, 10, 20, 40}.
4. Estimar a métrica em R², R³ e R⁴ e medir fidelidade.
5. Calcular o teto.
6. Treinar um SVM RBF nos rótulos do LLM para comparar a fronteira (figura `bloco1_graphic_14`; desde 16/10/2026 também os números: CSV `bloco1_d_passo6_svm` e tabela `bloco1_table_19_D_passo6_svm`, por `classical_baselines.d_passo6_svm_medidas`; para execuções anteriores, `python src/gerar_tabela_d_passo6_svm.py <pasta>`).

**Problemas E1, E2, E3 (especialista linear, LLM como aprendiz)**
1. Gerar os 150 pontos; o especialista (W conhecido) rotula todos.
2. O LLM classifica os 150 pontos em zero-shot (uma coleta, compartilhada pelos três especialistas).
3. O LLM recebe k exemplos rotulados pelo especialista, escolhidos pelas estratégias fácil, difícil, mista e aleatória, k ∈ {4, 10, 20, 40}, e classifica os 150 − k restantes.
4. Diluição: 4 exemplos difíceis fixos mais 0 a 20 fáceis (só E1).
5. Ordem dos exemplos: quatro ordenações do mesmo conjunto (só E1).
6. Baselines: k-NN, regressão logística e SVM treinados nos mesmos exemplos de cada estratégia (todas as 4, todos os k, só no primeiro modelo; não usam o LLM); a tabela `bloco2_table_06` e a figura `bloco2_graphic_05` mostram a estratégia mista.

**Problema F (meia-lua, LLM como aprendiz)**
1. Gerar os 150 pontos e dividir 70% de treino e 30% de teste (105 e 45 pontos).
2. O LLM rotula os 105 de treino em zero-shot; estimar a métrica em R², R³ e R⁴ só para medir a fidelidade.
3. O LLM recebe k exemplos do treino rotulados pelo rótulo real, k ∈ {0, 4, 10, 20, 40} (k = 0 é o LLM classificando o teste sem exemplos), e classifica os 45 de teste; exemplos e ponto de teste só com x1, x2 (desde 13/10/2026 o F não herda R³/R⁴ da métrica do Passo 2).
4. Perceptron treinado nos mesmos exemplos, para comparar.

**Problemas G1 e G2 (peso × altura)**
1. Carregar as 100 pessoas e dividir 70 de treino e 30 de teste (a mesma divisão para G1 e G2).
2. O LLM rotula os 70 de treino em zero-shot.
3. O LLM rotula os 70 com k exemplos do rótulo real, k ∈ {4, 10, 20, 40}.
4. Estimar a métrica em R², R³ e R⁴ a partir de cada conjunto de rótulos; medir fidelidade; calcular o teto (gravado com o valor de regime `passo5_teto`, numeração do A/D, no CSV `bloco3_g_passos_2_a_5`).
5. Eleger a melhor métrica (maior acurácia vs rótulo real no treino).
6. O LLM recebe k exemplos do treino rotulados pela métrica eleita, k ∈ {0, 4, 10, 20, 40} (k = 0 é o LLM classificando o teste sem exemplos), e classifica os 30 de teste (os mesmos pontos nas duas variantes de nome; o sorteio usa a mesma semente seed + rep do Passo 3, logo são os mesmos pontos sorteados lá, agora rotulados pela métrica eleita em vez do rótulo real). Desde 14/10/2026 (Metodologia IV.7: "alimenta-se o prompt com novos atributos") o número de atributos MOSTRADOS ao LLM é um fator controlado, `ATRIBUTOS_PROMPT_PASSO6_G = (2, 3, 4)` (2 = x1, x2; 3 = + x1·x2; 4 = + x1², x2²; nos exemplos e no ponto de teste), independente da dimensão da métrica eleita, que só define o especialista; os extras só acompanham exemplos, logo k = 0 é UMA coleta repetida nas linhas das três dimensões (coluna `n_features` do CSV = atributos no prompt; `especialista_n_features` = dimensão da eleita). O F usa só 2.
7. Perceptron treinado nos mesmos exemplos e no mesmo espaço que o LLM viu (2, 3 ou 4 atributos), para comparar.
Repetido para cada combinação de nomes de classe (Mulher/Homem, A/B) e de atributo (peso/altura ou kg/m, x1/x2).

**Oráculo sobre E3, E1 e E2** (até 13/10/2026: sobre B, E1 e E2)
1. Rotular as gaussianas do Problema E (centróides (−1,5; 1) e (1,5; −1), 150 pontos) pelo centróide mais próximo com o W conhecido: [1, 1] (E3, Problema 1), [0,3; 1,5] (E1, Problema 2) e [1,5; 0,3] (E2, Problema 3). Um único conjunto, só a métrica muda (14/10/2026); o Problema A não serve porque seus centróides têm x2 = 0 e w2 não seria identificável.
2. Rodar Perceptron e NNLS sobre esses rótulos.
3. Comparar o W recuperado ao verdadeiro: acurácia e cosseno.

**Equivalência com os nomes antigos**

| Nome antigo | Passos |
|---|---|
| "Fase A" (A e D), "protocolo único da Fase A" | Passos 2 a 5 do A e do D |
| "Fase A zero-shot" | Passo 2 (A, D) |
| "few-shot da Fase A", "few-shot GT" | Passo 3 (A, D) |
| "teto" | Passo 5 (A, D); Passo 4 do G |
| eleição da melhor métrica | Passo 6 do A |
| "Fases B e C" | Passos 2 a 4 de B e C |
| "B/C zero-shot" | Passo 2 de B e C |
| "transferência", "consistência" | Passo 3 de B e C |
| "few-shot H2", "exemplos da métrica" | Passo 4 de B e C |
| "Fase E" (Bloco 2) | Passos 2 e 3 de E (zero-shot = Passo 2; estratégias = Passo 3) |
| diluição | Passo 4 de E (só E1) |
| ordem dos exemplos | Passo 5 de E (só E1) |
| baselines clássicos | Passo 6 de E |
| "Fase A externa" | Passo 2 do F; Passos 2 a 4 do G |
| "Fase E externa" | Passo 3 do F; Passo 6 do G |
| "LLM vs Perceptron" | Passo 4 do F; Passo 7 do G |
| "Eleição do especialista" | Passo 5 do G |
| "Bloco 1/2/3" | continuam (agrupam problemas) |
| Hipóteses | H1 e H2 nos Passos 3 e 4 de B/C; H3 e H4 no Passo 3 de E; H5 nos Passos 4 a 6 do A |

**Equivalência de identificadores (camada 2, 13/10/2026)**: resumo do mapa `renomeacao_passos_2026-10-13.md`
(nomes de antes de 13/10/2026 → nomes atuais). A lista completa, inclusive painéis, chaves de dict e
variáveis do `main()`, está no mapa.

| Tipo | Nome antigo (antes de 13/10/2026) | Nome novo |
|---|---|---|
| constante | `FEW_SHOT_SIZES` | removida (era alias de `FEW_SHOT_SIZES_BC`) |
| constante | `FEW_SHOT_SIZES_FASE_A` / `FEW_SHOT_SIZES_BC` / `FEW_SHOT_SIZES_PHASE_E` / `BIAS_N_SHOTS` | `K_EXEMPLOS_ROTULO_REAL` / `K_EXEMPLOS_METRICA_BC` / `K_EXEMPLOS_APRENDIZ` / `K_VIESES` |
| constante | `PROTOCOLO_N_FEATURES` / `N_SAMPLES_PROBLEM_X` / `PHASE_A_PROTOCOL_CACHE` | `ATRIBUTOS_PASSO4` (com chave `"G"`) / `N_PONTOS_X` / `CACHE_ROTULOS_LLM` |
| constante | `HM_CLASS_NAME_VARIANTS` / `HM_G2_CLASS_NAME_VARIANTS` / `HM_G1_FEATURE_VARIANTS` / `HM_G2_FEATURE_VARIANTS` | `G_NOMES_CLASSE` / `G2_NOMES_CLASSE` / `G1_NOMES_ATRIBUTO` / `G2_NOMES_ATRIBUTO` |
| flag | `RUN_PHASES_ABC` / `RUN_PHASE_E` / `RUN_MULTIPLE_EXPERTS` | `RUN_PROBLEMAS_ABC` / `RUN_PROBLEMA_E` / `RUN_PROBLEMA_E2_E3` |
| flag | `RUN_CLASSICAL_BASELINES` / `RUN_DILUTION` / `RUN_EXAMPLE_ORDER_BIAS` | `RUN_E_PASSO6_BASELINES` / `RUN_E_PASSO4_DILUICAO` / `RUN_E_PASSO5_ORDEM` |
| flag | `RUN_R3R4_EXPERIMENT` / `RUN_ALGORITHM_COMPARISON` | `RUN_A_PASSO4_R3R4` / `RUN_A_PASSO4_PERCEPTRON_VS_NNLS` |
| flag | `RUN_CLASS_ORDER_BIAS` / `RUN_FEATURE_NAMES` / `RUN_PROMPT_VARIANTS` | `RUN_VIES_ORDEM_CLASSES` / `RUN_VIES_NOMES_ATRIBUTO` / `RUN_VIES_VARIANTES_PROMPT` |
| flag | `RUN_PROBLEM_D_FASE_A` / `RUN_PROBLEM_MEIALUA` / `RUN_HOMEM_MULHER` / `RUN_HM_CLASS_NAMES_AB` | `RUN_PROBLEMA_D` / `RUN_PROBLEMA_F` / `RUN_PROBLEMA_G` / `RUN_PROBLEMA_G_CLASSES_AB` |
| flag | `RUN_ORACLE_VALIDATION` / `RUN_ORACLE_MEIALUA` | `RUN_ORACULO` / `RUN_D_PASSO5_TETO` |
| função | `phase_a_learn_metric` / `phase_a_multifeature` / `phase_a_protocol` / `_protocol_rows` / `_balanced_gt_examples` | `a_passo2_zero_shot_e_metrica` / `passo4_estimar_metrica_r2_r3_r4` / `adg_passos_2_a_5` / `_linhas_passos_2_a_5` / `passo3_exemplos_rotulo_real` |
| função | `phase_consistency_test` / `select_confident_examples` / `evaluate_transfer_bc` / `marcar_melhor_metrica_a` | `bc_passos_2_a_4` / `bc_passo4_exemplos_da_metrica` / `bc_passo3_consistencia` / `a_passo6_eleger_metrica` |
| função | `run_complete_experiment` / `_run_phase_abc_experiment` / `_plot_protocol_symbols` / `_llm_fewshot_scheme` | `abc_executar` / `_abc_executar_configuracao` / `_plot_simbolos_passos` / `_esquema_passo3` |
| função | `phase_e_llm_as_learner` / `select_examples_by_strategy` / `select_examples_dilution` / `reorder_examples` / `expert_classify` | `e_passos_2_3_aprendiz` / `e_passo3_selecionar_exemplos` / `e_passo4_exemplos_diluicao` / `e_passo5_reordenar_exemplos` / `especialista_rotular` |
| função | `run_external_problem_pipeline(..., protocolo_fase_a=)` / `_balanced_example_indices` / `_eleger_especialista_externo` | `fg_executar(..., com_passos_3_a_5=)` / `fg_exemplos_aprendiz` / `g_passo5_eleger_especialista` |
| função | `run_oracle_validation` / `run_oracle_meialua` / `_oracle_algorithms` / `_append_oracle_result` | `oraculo_passos_1_a_3` / `d_passo5_teto` / `_oraculo_estimadores` / `_oraculo_registrar` |
| função | `_make_tag(fase, problema, regime, k, rep, seed, bloco, extra)` | `_tag(bloco, problema, passo, regime, k, rep, seed, extra)` |
| função | `build_prompt_zero_shot` / `build_prompt_few_shot` / `*_variant` | `prompt_zero_shot` / `prompt_com_exemplos` / `*_variante` |
| `main()` | `all_results_abc` / `all_results_e` / `all_phase_a_protocol`, `protocol_rows` / `external_phase_a_results` / `external_phase_e_results` / `external_phase_a_protocol_rows` / `all_results_oracle` / `all_results_oracle_meialua` / `phase_a_data_for_plots`, `phase_a_cache` / `hm_pipeline`, `ml_pipeline` | `resultados_abc` / `resultados_e` / `linhas_passos_2_a_5` / `resultados_fg_passos_2_e_4` / `resultados_fg_aprendiz` / `linhas_g_passos_2_a_5` / `resultados_oraculo` / `resultados_d_passo5` / `dados_a_passo2`, `cache_a_passo2` / `saida_g`, `saida_f`; também `all_results_abc_collapsed/alternative/dilution/example_order/baselines` → `resultados_abc_colapsados`, `resultados_a_passo4_nnls`, `resultados_e_passo4/5/6`; `transfer_rows` → `linhas_bc_passo3`; `protocol_labels` → `rotulos_passos`; `external_llm_label_maps` → `mapas_rotulos_fg`; `melhor_metrica_a_por_modelo` → `metrica_eleita_por_modelo`; em `fg_executar`, `all_phase_a/all_phase_e` → `lista_passos_2_e_4/lista_aprendiz` |
| chave de dict | `'phase_a_results'` / `'phase_e_results'` / `'phase_a_protocol_rows'` | `'passo2'` / `'aprendiz'` / `'passos_2_a_5'` |
| dataclass | `ResultadoExperimento` / `ResultadoPhaseEExperimento` | `ResultadoABC` / `ResultadoAprendiz` |
| campo | `n_shot` (ABC) / `consistencia_transferida_problema_b/c` / `kappa_transferido_problema_b/c` | `k_passo4_bc` / `consistencia_centroides_a_problema_b/c` / `kappa_centroides_a_problema_b/c` |
| relatórios | `print_phase_e_analysis` / `malformadas_fase_e` / `print_final_analysis` / `print_statistical_summary` / `summarize_cross_linearity` | `print_e_passos_2_3` / `malformadas_e` / `print_abc_resumo` / `print_sumario_estatistico` / `sintese_r2_r3_r4` |
| relatórios | `print_error_analysis_by_region` / `print_hyperparameter_sensitivity` / `print_example_order_analysis` | `print_a_passo4_erros_por_margem` / `print_a_passo4_sensibilidade_perceptron` / `print_e_passo5_ordem` |
| plots | `plot_metric_errors_phase_a` / `plot_fase_a_fewshot_curve` / `_g` / `plot_w_distribution` / `plot_algorithm_comparison` / `plot_seed_comparison` / `plot_consistency_comparison_extended` / `plot_r3_comparison` / `plot_gamma_convergence` | `plot_a_passo4_erros` / `plot_passos_2_a_5_curva` / `plot_g_passos_2_a_5_curva` / `plot_a_passo4_w_distribuicao` / `plot_a_passo4_perceptron_vs_nnls` / `plot_bc_passo3_sementes` / `plot_bc_passo3_consistencia` / `plot_a_passo4_r2_r3_r4` / `plot_a_passo4_gamma` |
| plots | `plot_class_names_effect` / `plot_class_order_bias` / `plot_feature_names_effect` / `plot_prompt_variant_comparison` / `plot_oracle_w_recovery` / `plot_oracle_meialua` | `plot_vies_nomes_classe` / `plot_vies_ordem_classes` / `plot_vies_nomes_atributo` / `plot_vies_variantes_prompt` / `plot_oraculo_recuperacao_w` / `plot_d_passo5_teto` |
| plots | `plot_phase_e_learning_curve` / `plot_phase_e_strategy_comparison` / `plot_phase_e_example_locations` / `plot_dilution_experiment` / `plot_example_order_bias` / `plot_classical_baselines_comparison` | `plot_e_passo3_curva` / `plot_e_passo3_estrategias` / `plot_e_passo3_exemplos` / `plot_e_passo4_diluicao` / `plot_e_passo5_ordem` / `plot_e_passo6_baselines` |
| plots | `plot_external_learning_curve` / `plot_external_features_comparison` / `plot_external_decision_boundary` / `plot_phase_e_llm_vs_perceptron` / `plot_meialua_svm_vs_llm` / `plot_llm_labels_per_problem` | `plot_fg_aprendiz_curva` / `plot_fg_passos_2_e_4_r2_r3_r4` / `plot_fg_fronteiras` / `plot_fg_llm_vs_perceptron` / `plot_d_passo6_svm` / `plot_fg_rotulos_llm` |
| plots | `visualize_all_problems` / `visualize_problem_e_with_expert` / `plot_problem_overview` / `plot_dataset_overview` / `plot_hits_and_errors` / `plot_w_comparison_algorithms` / `plot_confusion_matrices_detailed` / `plot_margin_analysis_detailed` / `plot_experiment_summary_dashboard` / `plot_model_comparison` / `plot_labels_symbols` | `plot_abc_passo1` / `plot_e_passo1` / `plot_problema_passo1` / `plot_final_dados` / `plot_final_acertos_erros` / `plot_final_w_algoritmos` / `plot_final_matrizes_confusao` / `plot_final_margem` / `plot_final_painel_resumo` / `plot_final_comparacao_modelos` / `plot_rotulos_simbolos` |
| tabelas | `gerar_tabelas_bloco1.tabela_T1..T9`, `gerar_tabelas.tab_*` (ex. `tab_G_protocolo`), `melhor_metrica_a()` (leitura da eleição) | `tab_<problema>_passoN_*` (ex. `tab_a_passos_2_a_5`, `tab_bc_passo3_todas_metricas`, `tab_g_passos_2_a_5`, `tab_final_cross`, `tab_final_comparacao_modelos`, `tab_final_auditoria_modelo`), `eleita_passo6()` |
| `carregar_tudo` | chaves `proto`, `bc`, `alg`, `r3r4`, `oraculo_ml`, `fase_e`, `dil`, `ordem`, `bl`, `ext_a`, `ext_e`, `proto_g` | `ad_passos_2_a_5`, `bc_passo3`, `a_passo4_perc_nnls`, `a_passo4_r3r4`, `d_passo5`, `e`, `e_passo4`, `e_passo5`, `e_passo6`, `fg_passos_2_e_4`, `fg_aprendiz`, `g_passos_2_a_5` (`abc`, `oraculo`, `cross` ficam) |
| coluna (ABC) | `n_shot_phase_bc` / `consistency_transferred_b/c` / `kappa_transferred_b/c` / `repetition` | `k_passo4_bc` / `consistency_centroides_a_b/c` / `kappa_centroides_a_b/c` / `rep` |
| coluna (Passos 2 a 5) | `regime` = `zero_shot` / `few_shot` / `ground_truth`; `fidelity_gt_metric_vs_llm_zero`; `accuracy_ceiling_vs_true` | `passo2_zero_shot` / `passo3_exemplos_rotulo_real` / `passo5_teto`; `fidelity_teto_vs_llm_passo2`; `accuracy_teto_vs_true` |
| coluna (B/C Passo 3) | `source_regime` / `source_k` / `source_rep` / `fidelity_source_mean` / `melhor_metrica_a` / `collapsed_source` | `metrica_regime` / `metrica_k` / `metrica_rep` / `metrica_fidelidade_a` / `eleita_passo6` / `collapsed_metrica` |
| coluna (aprendiz) | `n_shot` (E, F, G) / `expert_regime`, `expert_k`, `expert_rep` / `expert_source` = `ground_truth`, `metric` | `k` / `especialista_regime`, `especialista_k`, `especialista_rep` / `especialista_origem` = `rotulo_real`, `metrica` |
| tag | `fase=A` / `fase=ABC` zero-shot de B/C / B/C `regime=few_shot_metrica` / `fase=ABC_aux` / `fase=E` / `fase=E_diluicao` / `fase=E_ordem` / F `fase=A`, `fase=E` / G `fase=A`, `fase=E` | `passo=2` (zero-shot) ou `passo=3` (exemplos do rótulo real) / `passo=2` / `passo=4\|regime=exemplos_da_metrica` / `passo=2` (k = 0, A e B/C zero-shot) ou `passo=4` (k > 0, B/C com exemplos da métrica) com `regime=vies_<fator>` / `passo=2` ou `passo=3` / `passo=4` / `passo=5` / `passo=2`, `passo=3` / `passo=2` ou `passo=3`, `passo=6` |
| CSV | `bloco1_phases_abc` / `bloco1_fase_a_protocolo` / `bloco1_transferencia_bc` / `bloco1_algorithm_comparison` / `bloco1_r3r4_comparison` / `bloco1_oracle_validation` / `bloco1_oracle_meialua` | `bloco1_abc_passos_2_a_4` / `bloco1_ad_passos_2_a_5` / `bloco1_bc_passo3_consistencia` / `bloco1_a_passo4_perceptron_vs_nnls` / `bloco1_a_passo4_r2_r3_r4` / `bloco1_oraculo_passos_1_a_3` / `bloco1_d_passo5_teto` |
| CSV | `bloco2_phase_e` / `bloco2_dilution` / `bloco2_example_order` / `bloco2_classical_baselines` / `bloco23_external_phase_a` / `bloco23_external_phase_e` / `bloco3_fase_a_protocolo` / `problem_X.csv` | `bloco2_e_passos_2_3` / `bloco2_e_passo4_diluicao` / `bloco2_e_passo5_ordem` / `bloco2_e_passo6_baselines` / `bloco23_fg_passos_2_e_4` / `bloco23_fg_aprendiz` / `bloco3_g_passos_2_a_5` / `problema_X.csv` |
| gráfico | `bloco1_graphic_01_ABC_problemas_lineares` / `02_A_rotulos_simbolos` / `03_A_fase_a_erros` / `04_AD_fase_a_fewshot_curva` / `05_A_w_distribuicao` / `06_A_perceptron_vs_nnls` (painéis `_fase_b/_fase_c`) / `07_A_sementes` / `08_A_R2_R3_R4` / `09_A_gamma_convergencia` / `10_BC_consistencia` | `01_ABC_passo1_dados` / `02_A_passos_2_a_4_simbolos` (painéis `_passo1_rotulo_real`, `_passo2_llm_zero_shot`, `_passo3_llm_<k>shot`, `_passo4_metrica`) / `03_A_passo4_erros` / `04_AD_passos_2_a_5_curva` / `05_A_passo4_w_distribuicao` / `06_A_passo4_perceptron_vs_nnls` (`_bc_passo3_b/_bc_passo3_c`) / `07_BC_passo3_sementes` / `08_A_passo4_R2_R3_R4` / `09_A_passo4_gamma` / `10_BC_passo3_consistencia` |
| gráfico | `bloco1_graphic_11_{B,C}_rotulos_simbolos` / `12_D_meialua_seedS_overview` / `13_D_rotulos_simbolos` / `14_D_svm_vs_llm` / `19_oraculo_recuperacao_W` / `21_D_teto_oraculo_meialua` (15 a 18 `vies_*` ficam) | `11_{B,C}_passos_2_3_simbolos` (painéis `_passo1_rotulo_real`, `_passo2_llm_zero_shot`, `_passo3_metrica_eleita`) / `12_D_passo1_dados_seedS` / `13_D_passos_2_a_4_simbolos` / `14_D_passo6_svm` / `19_oraculo_passos_1_a_3` / `21_D_passo5_teto` |
| gráfico | `bloco2_graphic_01_E_especialista` / `02_E_estrategias_exemplos` / `03_E_curva_aprendizado` / `04_E_comparacao_estrategias` / `05_E_baselines_classicos` / `06_E_diluicao` / `07_E_ordem_exemplos` | `01_E_passo1_especialista` / `02_E_passo3_estrategias_exemplos` / `03_E1_passo3_curva` / `04_E1_passo3_estrategias` / `05_E1_passo6_baselines` / `06_E1_passo4_diluicao` / `07_E1_passo5_ordem` |
| gráfico | `bloco3_graphic_01_G1_peso_altura` / `02_G2_peso_altura_kg_m` / `03_G_fase_a_fewshot_curva` / `bloco23_graphic_01_FG_curva_aprendizado` / `02_FG_R2_R3_R4` / `03_FG_fronteiras_decisao` / `final_graphic_01_ABCE_dados` / `final_graphic_07_..._{train\|<k>}` | `01_G1_passo1_dados` / `02_G2_passo1_dados` / `03_G_passos_2_a_5_curva` / `01_FG_aprendiz_curva` / `02_FG_passos_2_e_4_R2_R3_R4` / `03_FG_fronteiras` / `01_ABCE_passo1_dados` / `_passo2` (treino) e `_passo3_k<k>` (F) ou `_passo6_k<k>` (G); `04_FG_llm_vs_perceptron`, `final_graphic_02..06` e `08` ficam |
| slug de tabela | `A_zero_fewshot_teto` / `A_zero_vs_fewshot_comparacao` / `A_escolha_melhor_metrica_transferencia` / `A_R2_R3_R4_zero_fewshot` / `A_perceptron_vs_nnls_por_semente` / `BC_transferencia_todas_metricas_A` / `BC_transferencia_zero_vs_melhor_por_semente` / `BC_comparativo_zero_vs_melhor` / `BC_fewshot_exemplos_da_metrica` | `A_passos_2_a_5` / `A_passo2_vs_passo3` / `A_passo6_eleicao` / `A_passo4_R2_R3_R4` / `A_passo4_perceptron_vs_nnls` / `BC_passo3_todas_metricas_A` / `BC_passo3_por_semente` / `BC_passo3_zero_vs_eleita` / `BC_passo4_exemplos_da_metrica` |
| slug de tabela | `D_meialua_R2_R3_R4` / `D_zero_vs_fewshot_comparacao` / `A_vs_D_por_regime` / `AD_estabilidade_W_por_semente` / `oraculo_recuperacao_W` / `D_teto_oraculo_meialua` (`vies_*` ficam) | `D_passos_2_a_5` / `D_passo2_vs_passo3` / `AD_por_passo` / `AD_passo4_estabilidade_W` / `oraculo_passos_1_a_3` / `D_passo5_teto` |
| slug de tabela | `E{1,2,3}_faseE_estrategia_x_nshot` / `E_zero_vs_fewshot_por_especialista` / `E_H4_media_fewshot_por_estrategia` / `E_LLM_vs_baselines_nmax` / `E1_diluicao` / `E1_ordem_exemplos` / `F_meialua_faseA_R2_R3_R4` / `F_meialua_faseE_nshot` | `E{1,2,3}_passo3_estrategia_x_k` / `E_passo2_vs_passo3` / `E_passo3_H4` / `E_passo6_baselines` / `E1_passo4_diluicao` / `E1_passo5_ordem` / `F_passo2_R2_R3_R4` / `F_passo3_k` |
| slug de tabela | `G_faseA_R2_R3_R4` / `G_faseE_nshot` / `G_faseE_LLM_vs_perceptron` / `G1_vs_G2_zero_shot` / `G_faseA_zero_fewshot_teto` (`G_nomes_semanticos_vs_neutros` e `final_*` ficam) | `G_passos_2_e_4_R2_R3_R4` / `G_passo6_k` / `G_passo7_LLM_vs_perceptron` / `G1_vs_G2_passos_2_e_4` / `G_passos_2_a_5` |
| teste | `test_fase_a_protocolo.py` / `test_fase_a_protocolo_g.py` / `test_selecao_exemplos.py` / `test_reorder_examples.py` | `test_adg_passos_2_a_5.py` / `test_g_passos_2_a_5.py` / `test_e_passo3_exemplos.py` / `test_e_passo5_ordem.py` (demais ficam, com classes/métodos renomeados) |
| ferramenta | `regerar_figuras_simbolos.py` / `corrigir_mapeamento_homem_mulher.py` / `renomear_assets_legado.py` | `regerar_simbolos.py` / removido (legado de 07/2026 sem uso) / ganha as regras do mapa (arquivos e colunas) com `--aplicar` |
| parâmetro (complemento de 14/10/2026) | `n_shot` como parâmetro/variável (`reps_para`, `bc_passos_2_a_4`, `abc_executar`, `e_passos_2_3_aprendiz`, laços do `main()`, `plots.py`, `relatorios.py`, `classical_baselines.py`, testes); `n_shots`/`n_shot_bias`/`n_shot_e`/… | `k_exemplos`; `valores_k`/`k_vies`/`k_e`/`k_bl`/`k_ordem`/`k_val` |
| constante (complemento de 14/10/2026) | `EXAMPLE_ORDER_N_SHOTS` (também chave de `GRADES_MODO_RAPIDO`) | `K_ORDEM_EXEMPLOS` |

## Convenção de nomes de gráficos e tabelas (09/10/2026)

```
<bloco>_<graphic|table>_<NN>_<assunto>[_<problema>][_seed<S>][_<painel>][__<alias>]_<color|gray>.png
```
- `<bloco>`: `bloco1`, `bloco2`, `bloco23` (F+G), `bloco3`, `final`; oráculo e teto da meia-lua em `bloco1`,
  auditoria e comparação de modelos em `final`. `<NN>` = ordem dentro do bloco, na ordem dos problemas
  (A → B/C → D → vieses → oráculo; E1..E3 → H4 → baselines → diluição → ordem → F; G; fechamento);
  gráficos e tabelas têm numerações independentes (`graphic_01…`, `table_01…`).
- **Cor por último**: todo gráfico sai em DOIS arquivos, `_color` (como desenhado) e `_gray` (preto e
  branco gerado por `estilo_cinza.para_cinza`: classes/séries por símbolo `+ − / * # @` (e `x`, `d` para a 7ª e a 8ª série, 15/10/2026), traços
  sólido/tracejado/pontilhado, hachuras nas barras, mapas em `gray`). As figuras de símbolos
  (`bloco1_graphic_02/11/13`) são desenhadas duas vezes (azul/vermelho → `_color`; preto → `_gray`); só as tabelas saem apenas em `_gray`. `.tex` de tabela = mesmo
  nome sem o campo de cor. Painel `_<painel>` entra antes do `__<alias>` (`asset_variant`), a cor depois
  (`execucao_io.com_cor`); `plots.salvar_figura` faz o salvamento duplo.
- **Legenda garantida** (09/10/2026): `estilo_cinza.garantir_legendas(fig)` roda em todo salvamento. Scatter `c=y`
  sem rótulo vira um scatter por classe ("classe 0/1"); scatters de cor única sem rótulo (≥ 4 pontos) viram
  "amostras (1)", "amostras (2)", …; marcadores grandes viram "centróide k"; contorno de linha sem rótulo ganha proxy
  "fronteira de decisão"; linhas e barras sem rótulo viram "série k" (ou o rótulo do eixo y). Tampas, hastes e
  linhas de `ErrorbarContainer` (barras de erro de `errorbar`/`bar(yerr=…)`) NUNCA são rotuladas nem viram série;
  barras com `label=''` num eixo que já tem legenda ficam fora dela; `axhline`/`axvline` são referências (preto,
  sem símbolo). Na versão `_gray` os cinzas de linhas e pontos nunca passam de 0,6 e séries com só 2 pontos mantêm
  traço e símbolo próprios. Rodada 2 de 09/10/2026: séries da MESMA cor mantêm o marcador e o traço originais (a chave
  que identifica a série na conversão P&B é cor + marcador + traço, não só a cor); a linha de dados de
  `errorbar(fmt='o')` é série, não tampa; proxies de fronteira (handles com 0 pontos) ficam pretos com o linestyle
  do contorno; boxplot sem `patch_artist` não vira série. Eixo com séries e ainda sem legenda imprime `⚠️ sem legenda em <arquivo>` no log (grep
  no log_execucao.txt deve dar zero). Toda `Tabela` tem caption e rodapé (testes em `tests/test_tabelas.py` e
  `tests/test_estilo_cinza.py`).
- Vale para execuções a partir de 09/10/2026; as anteriores (`execucao_2026-10-08_23-16-54_completa` e
  antigas) mantêm os nomes antigos (`bloco1_06_w_distribution__gpt4mini.png`, `T1_problema_A__*.tex`,
  `tabela_<slug>__<alias>.png`) e NÃO foram migradas (decisão de 09/10/2026). Correspondência antigo → novo
  no mapa `MAPA` do histórico (ver `revisao_codigo_2026-10-08.md`) e, resumida: bloco1 01→01, 15→02, 05→03,
  17→04, 06→05, 07→06, 08→07, 10→08, final_10→09, 09→10, 16→11, 02→12, 18→13, 11→14, 12→15, 14a→16, 14b→17,
  13→18, 03→19, 04→20, 04b→21; bloco2 01→01, 03→02, 04→03, 05→04, 09→05, 06→06, 07→07; final 04→01, 02→02,
  05→03, 06→04, 07→05, 03→06, 08→07, 09→08.
- **Camada de 13/10/2026**: os números e blocos ficam, só o `<assunto>` muda (ex.: antes de 13/10/2026 `bloco1_graphic_03_A_fase_a_erros`,
  hoje `bloco1_graphic_03_A_passo4_erros`; slug de tabela antes de 13/10/2026 `A_zero_fewshot_teto`, hoje `A_passos_2_a_5`). O mapa completo
  (gráficos, painéis, slugs, CSVs e colunas) está em `renomeacao_passos_2026-10-13.md`. Execuções anteriores a
  13/10/2026 (inclusive as desse dia) precisam de `python src/renomear_assets_legado.py --aplicar` ANTES de
  `gerar_tabelas.py`, pois `gerar_tabelas.carregar_tudo` não tem fallback para os nomes antigos.

## Convenção de nomenclatura dos assets (a partir de 03/07/2026; painéis desde 01/10/2026)

**Assets cujos dados derivam de UM LLM específico levam o ALIAS curto do modelo antes
da extensão**, via `llm_asset()` + `MODEL_ALIAS` (`src/execucao_io.py`): ex.
`bloco1_graphic_05_A_passo4_w_distribuicao__gpt4mini.png`, `final_graphic_03_ABC_acertos_erros_seed42__flashlite.png`,
`final_cross_linearity__flashlite.csv`.

| Modelo | Alias |
|---|---|
| `openai/gpt-4o-mini` (id no OpenRouter; `gpt-4o-mini` nas execuções até 07/2026) | `gpt4mini` |
| `google/gemini-2.5-flash-lite` | `flashlite` |

**TODOS os modelos são tratados de forma igual**: cada modelo de `MODELS_TO_TEST` gera
o conjunto COMPLETO de gráficos por modelo (`bloco1_graphic_02` a `18`, exceto 12;
`bloco2_graphic_03` a `07`; `bloco23_*`; `final_graphic_01` a `07`;
`final_cross_linearity__<alias>.csv`) **na raiz da execução**: não existe subpasta por
modelo nem o conceito de "modelo principal". Modelo fora de `MODELS_TO_TEST` sem entrada em
`MODEL_ALIAS` cai no slug longo (`_model_slug`); teste em `tests/test_e_passo3_exemplos.py`
garante a cobertura dos 2.

**Painéis individuais e variantes de um asset** (`asset_variant()`): o sufixo do painel
entra ANTES do alias, para o alias ficar sempre imediatamente antes da extensão:
`bloco1_graphic_05_A_passo4_w_distribuicao_boxplot__gpt4mini.png`,
`final_graphic_01_ABCE_passo1_dados_seed42_problema_e__flashlite.png`,
`final_cross_linearity_corrigido_hm__gpt4mini.csv`. (Nunca `<base>__<alias>_<painel>`.)
Execuções anteriores a 01/10/2026 foram migradas por `src/renomear_assets_legado.py`.

**NÃO levam sufixo de modelo:**
- Assets sem dados de LLM: dados sintéticos (`bloco1_graphic_01`, `bloco1_graphic_12`,
  `bloco3_graphic_01/02`, `dados_sinteticos_*`), oráculos (`bloco1_graphic_19/21`,
  `bloco1_oraculo_passos_1_a_3_*`, `bloco1_d_passo5_teto_*`), especialista (`bloco2_graphic_01/02`) e baselines clássicos
  (`bloco2_e_passo6_baselines_*.csv`).
- Assets consolidados multi-modelo, identificados pela coluna `provider`/`model` no
  conteúdo: todos os CSVs com timestamp, `llm_interactions*.json`, `log_execucao*.txt` e
  `final_graphic_08_comparacao_modelos.png`. Única exceção entre os CSVs:
  `final_cross_linearity__<alias>.csv` é gerado UM por modelo (leva alias).

(A lista de arquivos abaixo mostra os nomes-base; aplicar mentalmente o sufixo
`__<alias>` aos derivados de LLM — um arquivo por modelo — e `_<painel>` antes dele.)

## Estrutura de Pastas

```
mestrado/
├── src/
│   ├── dissertacao_mestrado.py       # Runner principal (~5600 linhas): config, passos dos problemas, coleta LLM, main()
│   ├── plots.py                      # TODAS as visualizações (plot_<problema>_passoN_* / plot_final_*, ~3100 linhas)
│   ├── relatorios.py                 # Relatórios de texto: print_*_passoN_*, resumo_entre_sementes, texto_h1/texto_h4, sintese_r2_r3_r4
│   ├── execucao_io.py                # Tee (log), chunking de log/JSON, checkpoint, MODEL_ALIAS/llm_asset/asset_variant
│   ├── resultados.py                 # Dataclasses de resultado (ResultadoABC, ResultadoAprendiz, LearnedMetric)
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
│   ├── renomear_assets_legado.py     # Migra uma execução inteira p/ a convenção atual: regras de 01/10/2026 (painéis/alias), de 09/10/2026 (`_color`/`_gray`) e de 13/10/2026 (mapa `renomeacao_passos_2026-10-13.md`: arquivos E colunas de CSV); dry-run padrão, `--aplicar` grava; não propõe nada em execução já na convenção
│   ├── tabelas.py                    # Motor: `Tabela` → .tex (booktabs) e .png (matplotlib); nomes <bloco>_table_NN_<assunto>__<alias>_gray; faixa_kappa (Landis e Koch)
│   ├── estilo_cinza.py               # para_cinza(fig) + garantir_legendas(fig): versão P&B de qualquer figura (símbolos + − / * # @, hachuras, cinzas ≤ 0,6)
│   ├── gerar_tabelas_bloco1.py       # Tabelas T1–T9 do Bloco 1 (objetos Tabela) + tabelas/todas_as_tabelas_bloco1.{tex,docx}; não toca todas_as_tabelas.tex
│   ├── gerar_tabelas.py              # CATÁLOGO COMPLETO (Blocos 1/2/3, oráculo, auditoria, fechamento): .tex em tabelas/ + .png na raiz; chamado por main()
│   ├── regerar_simbolos.py           # Regera bloco1_graphic_02/11/13 offline (CSVs + llm_interactions), p/ trocar estilo
│   ├── gerar_entrega_orientador.py   # Pasta de entrega ao orientador (.docx com os itens em vermelho do texto dele + figuras/tabelas soltas), 16/10/2026
│   ├── gerar_resultados_com_tabelas.py # Capítulo V do orientador reproduzido com as imagens novas sob cada vermelho + anexos (variantes de prompt, comparação com o artigo de 07/2025)
│   └── animacao_perceptron/          # Animação HTML do Perceptron: gerar.py (página `algoritmo_perceptron_animado.html` a partir da última _completa), template.html, core.js (porte JS de fit, idêntico bit a bit ao Python), pseudocodigo.py (fonte ÚNICA do pseudocódigo: página e .docx; tabela de correspondência com as linhas do .py, conferida por tests/test_animacao_pseudocodigo.py) e gerar_docx.py (`algoritmo_codigo_executado_pseudocodigo.docx`). Atualizados em 16/10/2026 para o Perceptron de 15/10/2026 (B, C, D, fator em [0, 1])
├── tests/                        # unittest, 390 testes em 15/10/2026 (1 pulado quando não há `--apenas oraculo` posterior a 14/10; `test_novidades_2026_10_15.py`: lote, figuras em processos, cronômetro, Passo 7 no espaço do prompt, oráculo E3, figuras FG, centróides do A) (parser, métricas, estimadores, plots, --apenas, renomeação, auditoria, tabelas, estilo_cinza, correções de 08/10 e 09/10, infraestrutura do runner em test_infra_main.py, Passos 2 a 5 de A/D/G em test_adg_passos_2_a_5.py e test_g_passos_2_a_5.py, atributos no prompt do Passo 6 do G em test_g_passo6_atributos_prompt.py, Passos 3 e 5 de E em test_e_passo3_exemplos.py e test_e_passo5_ordem.py, regressão numérica sobre a completa de 05/07/2026)
├── dados_reais/homem_mulher/     # Base real peso × altura (peso_altura.csv) + amostras X/T/R
├── requirements.txt              # Dependências Python
├── CLAUDE.md                     # Este arquivo
├── README.md                     # Como reproduzir
├── glossario.md                  # Glossário dos termos do trabalho
├── passos_por_problema.md        # Lista dos passos de cada problema (referência curta da nomenclatura "Problema X, Passo N")
├── problemas_tabela.html         # Tabela dos problemas + animação didática dos passos (página autônoma)
├── renomeacao_passos_2026-10-13.md  # Mapa ÚNICO antigo → novo da renomeação por passos (arquivos, colunas, tags; complemento de 14/10)
├── revisao_codigo_2026-10-08.md  # Relatório da revisão de código de 08/10/2026
├── revisao_codigo_2026-10-09.md  # Relatório da revisão de 09/10/2026 (rodadas, fechamento, decisões de 13/10)
├── revisao_codigo_2026-10-15.md  # Relatório da revisão de 15/10/2026 (achados pendentes, ver Notas de Design)
├── email_orientador_*.md         # Rascunhos de e-mails ao orientador
├── roteiro_apresentacao.txt      # Índice mestre dos slides (fonte de verdade das apresentações)
├── .env                          # Chaves de API (carregadas via python-dotenv)
├── .venv/                        # Ambiente virtual Python
├── .claude/
│   ├── settings.local.json       # Configurações locais do Claude Code
│   └── skills/                   # 13 skills ativas (cópia canônica; skills/ na raiz é só documentação)
├── reunioes_orientador/           # Transcrições e planos de reuniões com orientador
├── artigos_referenciados/         # PDFs de artigos citados
├── trabalhos_referencias.txt      # Lista consolidada de referências bibliográficas
└── resultados/execucao_YYYY-MM-DD_HH-MM-SS_{completa|smoke|apenas-<blocos>[_smoke]}/ # Pasta criada a cada execução
    #   sufixo `_completa` = execução cheia (fonte para o trabalho);
    #   sufixo `_smoke` = execução `--rapido` (NUNCA usar como fonte de números);
    #   sufixo `_apenas-<blocos>` = execução parcial (`--apenas`; nunca `_completa`).
    #   (As pastas resultados/execucao_* são COMMITADAS por decisão do projeto.)
    ├── dados_sinteticos_seed{seed}/       # CSVs por seed: problema_{A,B,C,D,E,A_r3,A_r4}.csv
    #   (problema_D.csv = meia-lua, desde 01/10/2026 [até 13/10/2026: problem_D.csv]; problema_E.csv = especialista linear do Bloco 2,
    #    que antes de 01/10/2026 era gravado como problem_D.csv; execuções antigas foram migradas)
    # === BLOCO 1 — LLM como ESPECIALISTA (otim. inversa) ===
    ├── bloco1_graphic_01_ABC_passo1_dados.png       # + painéis _problema_{a,b,c}.png
    ├── bloco1_graphic_12_D_passo1_dados_seed*.png
    ├── bloco1_graphic_19_oraculo_passos_1_a_3.png        # Oráculo sobre E3/E1/E2 (Problemas 1/2/3 do orientador, as três métricas sobre as gaussianas do E; sobre B até 13/10/2026): recuperação de W conhecido, 3 seeds agregadas (média, mín–máx); painéis _w_scatter/_ratio (escala log)/_cosseno/_acuracia
    #   (bloco1_graphic_20: REMOVIDO em 09/10/2026, não havia transferência; a numeração fica com lacuna)
    ├── bloco1_graphic_21_D_passo5_teto.png          # Teto do Problema D: W ajustado ao rótulo real da meia-lua em R2/R3/R4 (0 chamadas)
    ├── bloco1_graphic_03_A_passo4_erros_seed*.png      # por seed (mapa de erros; + _mapa_erros/_erros_por_margem)
    ├── bloco1_graphic_05_A_passo4_w_distribuicao.png           # entre sementes e nomes de classe (marcador por par de nomes); + _boxplot, _ratio, _scatter
    ├── bloco1_graphic_06_A_passo4_perceptron_vs_nnls.png     # Perceptron × NNLS (+ _w_scatter com vetor unitário/_fidelidade/_bc_passo3_b/_bc_passo3_c)
    ├── bloco1_graphic_07_BC_passo3_sementes.png               # só zero-shot (1 coleta por seed, sem barra de erro)
    ├── bloco1_graphic_10_BC_passo3_consistencia.png     # + painéis por métrica (Consistência, Cohen's κ, F1-Macro)
    ├── bloco1_graphic_08_A_passo4_R2_R3_R4.png          # R2→R3→R4 no Problema A (+ _linear_vs_quadratica/_algoritmos_r3r4)
    ├── bloco1_graphic_14_D_passo6_svm_seed*.png # SVM RBF vs rótulos zero-shot do LLM na meia-lua (fronteira escolhida por fidelidade vs LLM)
    ├── bloco1_graphic_15_vies_ordem_classes.png   # barras pareadas normal × invertido por regime; painéis _{consistencia_b,consistencia_c,fidelidade}_{0,10}shot
    ├── bloco1_graphic_18_vies_variantes_prompt.png  # painel de W com vetor unitário
    ├── bloco1_graphic_16_vies_nomes_classe.png    # só os 4 pares base; painéis _0shot/_10shot
    ├── bloco1_graphic_17_vies_nomes_atributo.png  # só zero-shot; painel de W com vetor unitário
    # --- Problema A, Passos 2 a 5 (orientador 01/10/2026): figuras P&B com símbolos ---
    ├── bloco1_graphic_02_A_passos_2_a_4_simbolos_seed*.png   # GT | LLM zero-shot | métrica zero-shot | LLM k-shot (painel `_passo3_llm_<k>shot`; k = 10 se estiver na grade, senão o menor de K_EXEMPLOS_ROTULO_REAL: 10 na completa, 4 no --rapido) (+ painéis)
    ├── bloco1_graphic_11_{B,C}_passos_2_3_simbolos_seed*.png  # GT | LLM zero-shot | melhor métrica do A (centróides geradores + rotação)
    ├── bloco1_graphic_04_AD_passos_2_a_5_curva.png       # fidelidade/acurácia × k, A e D (R²/R³/R⁴); + _nnls
    ├── bloco1_graphic_13_D_passos_2_a_4_simbolos_seed*.png   # meia-lua: rótulo real | LLM zero-shot | LLM k-shot | métrica R² | métrica R⁴ (painel `_passo3_llm_<k>shot`, mesmo k do graphic_02; os painéis `_passo4_metrica`/`_passo4_metrica_r4` só existem quando o zero-shot do D não colapsa; colapso anotado no título)
    # === BLOCO 2 — LLM como APRENDIZ (Problemas E e F) ===
    ├── bloco2_graphic_01_E_passo1_especialista.png        # + painéis _rotulo_real/_fronteira_perito/_margem_perito
    ├── bloco2_graphic_02_E_passo3_estrategias_exemplos.png       # + painéis por estratégia
    ├── bloco2_graphic_03_E1_passo3_curva.png   # + painéis por métrica (especialista principal E1)
    ├── bloco2_graphic_04_E1_passo3_estrategias.png  # + painéis por k (especialista principal)
    ├── bloco2_graphic_06_E1_passo4_diluicao.png
    ├── bloco2_graphic_07_E1_passo5_ordem.png            # recency bias
    ├── bloco2_graphic_05_E1_passo6_baselines.png      # k-NN (uma série; os k no rodapé), LR, SVM × LLM (especialista principal)
    # === BLOCO 3 — Estudo de caso REAL (Problema G) ===
    ├── bloco3_graphic_01_G1_passo1_dados.png       # G1 (normalizado), elipse ótima desenhada (constante calibrada pela acurácia máxima)
    ├── bloco3_graphic_02_G2_passo1_dados.png  # G2 (kg e m), mesma base, mesma elipse convertida
    ├── bloco3_graphic_03_G_passos_2_a_5_curva.png  # Problema G, Passos 2 a 5 (13/10/2026): fidelidade/acurácia × k em R², 8 painéis (subproblema × classes × atributos), teto tracejado; + _nnls; painéis _g1_mulher_homem_peso_altura, _g1_a_b_x1_x2, _g2_mulher_homem_peso_kg_altura_m etc.
    # === BLOCO 2/3 — Problemas F e G (meia-lua + peso×altura) ===
    ├── bloco23_graphic_01_FG_aprendiz_curva.png
    ├── bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4.png
    ├── bloco23_graphic_03_FG_fronteiras.png
    ├── bloco23_graphic_04_FG_llm_vs_perceptron.png
    # === FECHAMENTO (visualizações por seed) ===
    ├── final_graphic_02_ABC_matrizes_confusao_seed*.png  # + _problema_{a,b,c}
    ├── final_graphic_06_painel_resumo_seed*.png
    ├── final_graphic_01_ABCE_passo1_dados_seed*.png    # A/B/C + E (painel E com os rótulos do especialista); + _problema_{a,b,c,e}
    ├── final_graphic_03_ABC_acertos_erros_seed*.png         # + _problema_{a,b,c}
    ├── final_graphic_04_A_w_algoritmos_seed*.png        # + _perceptron/_nnls/_barras
    ├── final_graphic_05_ABC_margem_seed*.png     # + _histograma/_taxa_erro/_mapa_confianca/_violin
    ├── final_graphic_07_FG_rotulos_llm_{F_meia_lua|G1_homem_mulher[_classesAB]|G2_homem_mulher[_classesAB]}_seed*_{x1_x2|peso_altura|peso_kg_altura_m}_{passo2|passo3_k<k>|passo6_k<k>}.png  # inclui as variantes A/B; painel do especialista (y_metric); legendas Mulher/Homem
    ├── final_graphic_08_comparacao_modelos.png          # único asset de fechamento SEM alias (compara todos; barras "Problema B"/"Problema C")
    ├── bloco1_graphic_09_A_passo4_gamma.png         # diagnóstico da busca binária em γ (Problema A, Passo 4)
    # === CSVs por bloco (consolidam todos os modelos; sem alias) ===
    ├── bloco1_abc_passos_2_a_4_{timestamp}.csv
    ├── bloco1_oraculo_passos_1_a_3_{timestamp}.csv   # 18 linhas: E3/E1/E2 × Perceptron/NNLS × 3 seeds (sobre B até 13/10/2026)
    ├── bloco1_d_passo5_teto_{timestamp}.csv
    ├── bloco1_d_passo6_svm_{timestamp}.csv   # Problema D, Passo 6 (16/10/2026): SVM RBF no rótulo real (acurácia) e nos rótulos zero-shot do LLM (fidelidade, acurácia), por modelo × semente, + métrica diagonal de maior fidelidade
    ├── bloco1_a_passo4_perceptron_vs_nnls_{timestamp}.csv
    ├── bloco1_a_passo4_r2_r3_r4_{timestamp}.csv
    ├── bloco1_ad_passos_2_a_5_{timestamp}.csv   # Passos 2 a 5 de A e D: (modelo, seed, A/D, n_features, regime, k, rep, algoritmo)
    ├── bloco1_bc_passo3_consistencia_{timestamp}.csv   # B/C zero-shot × cada métrica do A em R² (eleita_passo6=1 na eleita)
    ├── <bloco>_table_<NN>_<assunto>[__<alias>]_gray.png  # TODAS as tabelas em PNG (71 na completa de 13/10/2026), na raiz, numeradas por bloco
    ├── tabelas/                           # mesmas tabelas em .tex (mesmo nome sem _gray), INDICE.md, todas_as_tabelas.{tex,docx}; todas_as_tabelas_bloco1.{tex,docx} se gerar_tabelas_bloco1.py for rodado
    ├── bloco2_e_passos_2_3_{timestamp}.csv
    ├── bloco2_e_passo4_diluicao_{timestamp}.csv
    ├── bloco2_e_passo5_ordem_{timestamp}.csv
    ├── bloco2_e_passo6_baselines_{timestamp}.csv
    ├── bloco23_fg_passos_2_e_4_{timestamp}.csv
    ├── bloco23_fg_aprendiz_{timestamp}.csv
    ├── bloco3_g_passos_2_a_5_{timestamp}.csv   # Passos 2 a 5 do G (13/10/2026): colunas de bloco1_ad_passos_2_a_5 + problem_name, subproblem, class_names, feature_names
    ├── final_cross_linearity__<alias>.csv     # UM por modelo (exceção: CSV com alias)
    ├── *_corrigido_hm[__<alias>].csv          # só na execução de 07/2026 (cópias corrigidas offline; a ferramenta que as gerou foi removida em 13/10/2026)
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
(ex: `resultados/execucao_YYYY-MM-DD_HH-MM-SS_completa/apresentacao/apresentacao.tex`), junto com os gráficos e CSVs.
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
# OPENROUTER_API_KEY=sk-or-...   (única chave: todos os modelos passam pelo OpenRouter)
# (provedor único desde 01/10/2026: nenhuma outra chave é necessária)

# Executar o experimento completo (todos os modelos de MODELS_TO_TEST)
python src/dissertacao_mestrado.py

# Smoke test barato de UM modelo (antes de gastar na execução completa):
python src/dissertacao_mestrado.py --rapido --modelo gpt
python src/dissertacao_mestrado.py --rapido --modelo gemini
# As grades reduzidas do --rapido vivem em GRADES_MODO_RAPIDO (1 seed, 1 repetição, zero-shot + k=4;
# aplicadas por aplicar_modo_rapido(); o nome da pasta vem de nome_pasta_execucao()).

# Execução PARCIAL: só os blocos indicados (bloco1 | bloco2 | bloco3 | oraculo; aceita vários)
python src/dissertacao_mestrado.py --apenas bloco3                 # só peso×altura (Problema G, Passos 1 a 7)
python src/dissertacao_mestrado.py --apenas bloco2 bloco3 --modelo gpt
python src/dissertacao_mestrado.py --apenas oraculo               # 0 chamadas de API

# Guarda de saldo (08/10/2026): toda execução consulta o saldo do OpenRouter antes de
# gastar e aborta se não cobrir a estimativa (ver CUSTO_ESTIMADO_USD); --ignorar-saldo pula.
# Custo medido: ~US$ 0,09 por milhão de caracteres de prompt → completa de 08/10/2026 (158 mil
# chamadas) = US$ 8–10; com o Passo 3 do G (few-shot do rótulo real, 13/10/2026, +29,7 mil chamadas, ≈187 mil
# no total) a estimativa passa a US$ 11–13; com o Passo 6 do G em 2, 3 e 4 atributos no prompt (14/10/2026, +≈35 mil
# chamadas, ≈222 mil no total) passa a US$ 13–16 (constante CUSTO_ESTIMADO_USD["completa"] = 17, com folga); as completas
# de 14/10/2026 (221.508 chamadas) e de 15/10/2026 (222.648) rodaram com essa guarda.
# Durações medidas: 2h23 com MAX_CONCURRENCY=30 na completa de 08/10/2026 (158 mil chamadas); 3h39 com 40 na de 13/10/2026
# (187.328); 3h58 com 50 na de 14/10/2026 (221.508); 2h57 com 50 na de 15/10/2026 de manhã (222.648, já com coletas em lote e figuras
# em processos; ≈ 61 min desses foram 5 pontos da variante cot do Gemini que estouraram 5 tentativas de 90 s); 2h21 na de 15/10/2026 à noite
# (222.648, os mesmos 5 timeouts limitados a 2 tentativas de 5 s de espera).
python src/dissertacao_mestrado.py --apenas bloco3 --ignorar-saldo
```

```bash
# Ferramentas offline (0 chamadas de API):
python src/audit_interactions.py                       # auditoria da última _completa
python src/dissertacao_mestrado.py --reestimar-de <pasta>   # REESTIMAÇÃO OFFLINE (15/10/2026): reaproveita os rótulos do LLM de
#   <pasta>/llm_interactions*.json e refaz estimação, CSVs, tabelas e figuras numa pasta nova `execucao_<ts>_completa_reestimada`.
#   Casamento em 3 níveis: (1) prompt byte-idêntico (exato); (2) mesmo ponto e mesma tag (o LLM viu exemplos ligeiramente
#   diferentes: APROXIMADO); (3) resposta zero-shot do mesmo ponto na mesma configuração (o ponto era exemplo na origem).
#   Os níveis 2 e 3 só ocorrem onde o prompt depende da métrica estimada (Passo 4 de B/C, vieses com k > 0, Passo 6 do G);
#   o resumo sai no fim do log e em `reestimacao_lacunas.csv` (origem em `reestimacao_origem.txt`). Com `--coletar-lacunas`
#   esses pontos são coletados na API (só eles) e a pasta fica exata. Sem lacunas (ou com elas coletadas) a pasta serve de fonte.
python src/renomear_assets_legado.py [pasta] [--aplicar --docs] # migra UMA execução inteira p/ a convenção atual (dry-run padrão; --aplicar grava).
#   Conjunto de regras (13/10/2026): (1) painéis/alias de 01/10/2026; (2) `_color`/`_gray` de 09/10/2026, reconhecidos sem
#   proposta nova; regra `problem_D.csv → problem_E.csv` (nomes de 01/10) só quando `problem_E.csv` não existe na mesma pasta;
#   (3) mapa de 13/10/2026 (`renomeacao_passos_2026-10-13.md`): renomeia arquivos (CSVs, gráficos, painéis, slugs de tabela,
#   `problema_X.csv`) E colunas/valores dentro dos CSVs (`k_passo4_bc`, valores de `regime`, `especialista_*`,
#   `metrica_*`, `rep` etc.). OBRIGATÓRIO em toda execução anterior a 13/10/2026 (inclusive as desse dia) ANTES de gerar tabelas:
#   `gerar_tabelas.carregar_tudo` não tem fallback para nomes antigos. Em execução já na convenção não propõe NADA.
python src/gerar_tabelas.py [pasta] [--sem-png --sem-docx] # TODAS as tabelas da execução: .tex em <pasta>/tabelas/ + .png na raiz (main() já faz isso na ETAPA 6b); em pasta antiga, rodar renomear_assets_legado.py --aplicar antes
python src/gerar_tabelas_bloco1.py [pasta] [--sem-docx] # só as tabelas do Bloco 1 (ex-T1 a T9) em .tex + tabelas/todas_as_tabelas_bloco1.{tex,docx} (não sobrescreve todas_as_tabelas.tex)
python src/regerar_simbolos.py [pasta]                 # regera as figuras com símbolos (bloco1_graphic_02/11/13) sem nova coleta
# A ferramenta de correção H/M da execução de 07/2026 foi removida em 13/10/2026 (as cópias *_corrigido_hm*.csv daquela execução continuam no repositório).

# Execução para o orientador (02/10/2026): Bloco 1 com os Passos 2 a 6 do A e B/C só até o Passo 3 (zero-shot)
python src/dissertacao_mestrado.py --apenas bloco1 --bc-zero-shot
python src/gerar_tabelas_bloco1.py                      # tabelas da última _apenas-bloco1
python src/animacao_perceptron/gerar.py                # animação HTML do Perceptron (última _completa; 45 conjuntos em 16/10/2026, JS = Python bit a bit)
python src/animacao_perceptron/gerar_docx.py           # algoritmo_codigo_executado_pseudocodigo.docx (pseudocódigo + correspondência com o .py)
python src/gerar_entrega_orientador.py [pasta]          # pasta <execucao>/entrega_orientador_<data>/ (16/10/2026): .docx com os itens em VERMELHO do texto do orientador
#   (reunião de 02/10/2026) na numeração DELE (Tabela 2…8, Figura 1…7), tabelas como tabelas do Word, figuras embutidas; figuras/ (color+gray),
#   tabelas/ (.tex+.png), csv/, variantes_de_prompt.txt (texto real dos 4 enunciados); seleção em ITENS, números em NUMEROS, perguntas em PERGUNTAS
python src/gerar_tabela_d_passo6_svm.py [pasta]          # CSV bloco1_d_passo6_svm + tabela 19 numa execução anterior a 16/10/2026 (0 chamadas)
python src/gerar_resultados_com_tabelas.py [pasta]      # Resultados_com_tabelas_e_figuras.docx na mesma pasta: o capítulo V revisado pelo orientador (PDF da reunião de
#   02/10/2026) reproduzido com o texto dele, vermelhos em vermelho, imagens das tabelas/figuras novas abaixo de cada vermelho, notas em azul,
#   Anexo A (variantes de prompt) e Anexo B (comparação com as Tabelas 2 a 8 do artigo de 19/07/2025, transcritas em ANTIGAS)
```

A execução cria automaticamente uma pasta `resultados/execucao_YYYY-MM-DD_HH-MM-SS_completa/`
(ou `..._smoke/` quando rodada com `--rapido`) com todos os outputs; o nome é montado por
`nome_pasta_execucao(timestamp, apenas, modo_rapido)`, e as grades do `--rapido` por
`aplicar_modo_rapido()` a partir de `GRADES_MODO_RAPIDO` (ambas puras, testadas em `tests/test_infra_main.py`).
Com `--apenas`, a pasta recebe `..._apenas-<blocos>/` (ex.: `_apenas-bloco3`, ou
`_apenas-bloco3_smoke` se combinada com `--rapido`) e **nunca** `_completa`: é uma
execução parcial e não serve de fonte para o trabalho inteiro. `--apenas` força as flags
`RUN_*` de topo dos blocos escolhidos (`BLOCOS_APENAS` em `dissertacao_mestrado.py`) e
desliga as dos demais; as sub-flags (vieses, diluição, baselines, A/B, oráculo
meia-lua) continuam valendo o que está no código, pois já são gateadas pelo bloco-pai.
`RUN_A_PASSO4_R3R4` é flag de topo do bloco1 (depende do cache do Passo 2 do A) e
`RUN_PROBLEMA_D` (meia-lua como Problema D, Passos 2 a 6) pertence ao bloco1 e
`RUN_PROBLEMA_F` (meia-lua como Problema F, Passos 1 a 4) ao bloco2; `oraculo` é chave
própria (0 chamadas; `estimar_custo_usd(["oraculo"], True) == 0`, logo não exige chave nem
saldo), embora seus assets sejam `bloco1_graphic_19/21`. `--bc-zero-shot` roda os
Problemas B e C só até o Passo 3 (zero-shot e transferência, sem o Passo 4 de exemplos) na execução corrente.
As chamadas à API do LLM são **assíncronas com concorrência limitada**
(`asyncio.Semaphore(MAX_CONCURRENCY)`), acelerando substancialmente a coleta de decisões.
Ciclo de vida do cliente (09/10/2026): o global `async_client` fica `None` em produção e serve só
para injeção de dublês nos testes; cada `async_collect_llm_decisions` cria o `AsyncOpenAI` DENTRO
do laço de eventos do `asyncio.run` e o fecha no `finally`, com o laço ainda aberto. Um 402
atravessa o `gather`, o cliente é fechado e `collect_llm_decisions` chama `_abortar_por_creditos`
FORA do laço: o log não tem mais tracebacks "Event loop is closed" do httpx.

**Códigos de saída** (09/10/2026): 0 = execução completa; 2 = "EXPERIMENTO INCOMPLETO",
algum bloco levantou exceção (Blocos 1 e 2 por (modelo, seed), G1, G2 e F estão todos
envolvidos por `bloco_protegido`: a exceção vai para `falhas_fatais` com traceback no log,
o checkpoint das interações é gravado e a execução CONTINUA nos blocos seguintes; CSVs,
tabelas e log saem sempre); 3 = créditos esgotados (HTTP 402, `CreditosEsgotadosError`:
checkpoint + log parcial, sem fallback MD5; o cliente da API já foi fechado dentro do laço de eventos).

## Configuração Principal

Constantes no topo de `src/dissertacao_mestrado.py`:

| Constante | Valor padrão | Descrição |
|---|---|---|
| `RANDOM_SEED` | `42` | Semente padrão |
| `RANDOM_SEEDS` | `[42, 123, 7]` | 3 seeds uniformes em todos os experimentos e modelos (`EXTRA_SEEDS_CORE`/`is_seed_extra` foram removidos em 09/10/2026) |
| `K_VIESES` | `[0, 10]` | Âncoras dos experimentos de viés: nomes de classe, inversão de ordem das classes e variantes de prompt (efeitos comparáveis entre si); os nomes de atributo rodam só zero-shot |
| `MIN_PONTOS_CLASSE_MINORITARIA` | `5` (`protocolo.py`) | Quase-colapso: coleta do LLM cuja classe minoritária tem menos de 5 pontos vira `collapsed=1` com medidas NaN (regra única `colapsou`, em todos os passos; antes só a classe única contava) |
| `N_PONTOS_A` | `150` | Amostras no Problema A |
| `N_PONTOS_B/C` | `100` | Amostras nos Problemas B e C |
| `N_PONTOS_E` | `150` | Amostras no Problema E (especialista linear); a meia-lua (D/F) usa `N_PONTOS_A` |
| `K_EXEMPLOS_METRICA_BC` | `[0, 4, 10, 20, 40]` | k dos Passos 2 (k = 0, zero-shot) e 4 (k > 0) de B e C (exemplos rotulados pela métrica do A, H2; todos pares: balanço exato de classes no prompt); a CLI `--bc-zero-shot` força `[0]` numa execução (pedido do orientador, 01/10/2026). A antiga `FEW_SHOT_SIZES` (alias desta grade) foi removida em 13/10/2026 |
| `K_EXEMPLOS_ROTULO_REAL` | `[4, 10, 20, 40]` | k do **Passo 3** de A, D e G: exemplos rotulados pelo GROUND TRUTH, k/2 por classe, sem split em A e D; `--rapido` → `[4]` |
| `ATRIBUTOS_PASSO4` | `{"A": (2,3,4), "D": (2,3,4), "G": (2,3,4)}` | Dimensões da métrica no Passo 4 de A, D e G (LLM vê só x1,x2; augmentação só na estimação) |
| `K_EXEMPLOS_APRENDIZ` | `[0, 4, 10, 20, 40]` | k do Passo 3 de E, do Passo 3 do F e do Passo 6 do G (0 = zero-shot; todos pares) |
| `ATRIBUTOS_PROMPT_PASSO6_G` | `(2, 3, 4)` | Nº de atributos MOSTRADOS ao LLM no Passo 6 do G (2 = x1, x2; 3 = + x1·x2; 4 = + x1², x2²; nos exemplos e no ponto de teste), fator independente da dimensão da métrica eleita (14/10/2026); k = 0 é UMA coleta repetida nas linhas das três dimensões; o F usa só 2; o Perceptron do Passo 7 usa o mesmo espaço |
| `DILUTION_EASY_ADDITIONS` | `[0, 2, 4, 10, 16, 20]` | Easy adicionados aos 4 hard fixos na diluição |
| `K_ORDEM_EXEMPLOS` | `[4, 10, 20]` | k do viés de ordem dos exemplos |
| `PERCEPTRON_PARAMS` | `eta=0.001, C=1.0, delta_gamma=0.05, max_epochs=50, tol=1e-4` | Hiperparâmetros do Perceptron (ponto único, `protocolo.py`). `gamma_init=0.1` e `max_iterations=50` (teto da busca em γ) são os defaults do construtor em `relaxed_perceptron.py` e NÃO estão em `PERCEPTRON_PARAMS` |
| `N_REPETICOES` | `3` | Repetições ONDE HÁ SORTEIO de exemplos (regra única em `reps_para`, sem exceção) |
| `FIGURAS_PROCESSOS` | `max(1, min(8, núcleos − 2))` | Processos (`spawn`) para as figuras por semente (fechamento, símbolos de A/B/C/D) e os scatters `final_graphic_07` (`_figuras_em_paralelo`, 15/10/2026); o texto que cada figura imprime volta ao log; 1 desliga |
| `MAX_CONCURRENCY` | `50` (desde 14/10/2026; 40 na completa de 13/10) | Chamadas paralelas à API do LLM (benchmark 08/10/2026: 10→13-15/s, 30→21-39/s, 50→45-58/s, sem 429). Durações medidas, sempre sem 429: 2h23 com 30 (158 mil chamadas, 08/10), 3h39 com 40 (187 mil, 13/10), 3h58 com 50 (221,5 mil, 14/10) e 2h57 com 50 (222,6 mil, 15/10, já com coletas em lote e figuras em processos; ≈ 61 min desses foram 5 pontos da variante cot do Gemini que estouraram 5 tentativas de 90 s) |
| `MAX_RETRIES` / `API_TIMEOUT_S` | `5` / `90` | Laço de erros retentáveis (429/5xx/conexão/timeout, backoff exponencial a partir de `INITIAL_BACKOFF=10` s; `_erro_retentavel` reconhece também a mensagem "timed out" do `APITimeoutError` do SDK, 09/10/2026 rodada 2). Cada tentativa é uma chamada ao SDK, que por sua vez faz até 3 tentativas próprias (`_MAX_RETRIES=2` em `llm_client.py`): até 15 tentativas HTTP por ponto. Timeout de 90 s por chamada ao SDK vira erro RETENTÁVEL (não resposta vazia); HTTP 400 só é retentável com mensagem transitória; falha definitiva grava o prompt real e `format_retries=0` |
| `MAX_RETRIES_TIMEOUT` / `ESPERA_TIMEOUT_S` | `2` / `5` s | 15/10/2026: um timeout de 90 s (`APITimeoutError`, "timed out", `asyncio.TimeoutError`) tenta no máximo 2 vezes com espera fixa de 5 s e então cai no fallback (registro com `error`, rótulo por hash); os demais erros retentáveis seguem `MAX_RETRIES = 5` com backoff. Motivo: na completa de 15/10 cinco pontos da variante cot do Gemini custaram ≈ 61 min com 5 × 90 s + backoff |
| `MAX_FORMAT_RETRIES` | `5` | Reenvios do MESMO estímulo para respostas malformadas; depois, fallback MD5 (`malformed=True`) |
| `CUSTO_ESTIMADO_USD` / `MARGEM_SALDO` | completa 17, bloco1 3,5, bloco2 5,5, bloco3 8 (desde 13/10/2026: +≈30 mil chamadas do Passo 3 do G; desde 14/10/2026: +≈35 mil do Passo 6 com 3 e 4 atributos no prompt), smoke 0,6; ×1,2 | Guarda de saldo: `main()` consulta `GET /api/v1/credits` do OpenRouter antes da 1ª chamada e aborta se saldo < estimativa×margem (`--ignorar-saldo` desliga). HTTP 402 no meio da coleta = `CreditosEsgotadosError`: checkpoint + log parcial + `SystemExit(3)`, nunca fallback MD5 |
| `EXAMPLE_STRATEGIES` | `["easy","hard","mixed","random"]` | Estratégias do Passo 3 de E |
| `EXPERT_W` | `[0.3, 1.5]` | Pesos do especialista padrão (x2 dominante) |
| `EXPERT_CENTROIDS` | `[[-1.5, 1.0], [1.5, -1.0]]` | Centróides do especialista (Problema E, Passo 1) |
| `GRADES_MODO_RAPIDO` | `RANDOM_SEEDS=[42]`, `N_REPETICOES=1`, `K_EXEMPLOS_APRENDIZ/K_EXEMPLOS_METRICA_BC=[0, 4]`, `K_EXEMPLOS_ROTULO_REAL=[4]`, `K_VIESES=[0, 4]`, `DILUTION_EASY_ADDITIONS=[0, 4]`, `K_ORDEM_EXEMPLOS=[4]` | Grades reduzidas do `--rapido` (todos os experimentos, uma vez cada); `grades_modo_rapido()` devolve cópia e `aplicar_modo_rapido(namespace=None)` as aplica ao módulo (ou ao dict dado) |
| `LIMIAR_H1_KAPPA` | `0.4` (`relatorios.py`) | Limiar de Cohen's κ do veredito de H1 impresso no log (`relatorios.texto_h1`) |

### Flags de Execução

Controlam quais experimentos rodar sem precisar comentar/descomentar código:

| Flag | Padrão | Descrição |
|---|---|---|
| `RUN_PROBLEMAS_ABC` | `True` | Problemas A, B e C (Passos 2 a 6 do A; Passos 2 a 4 de B e C) |
| `RUN_PROBLEMA_E` | `True` | Problema E, Passos 2 e 3 (zero-shot e estratégias) |
| `RUN_VIES_ORDEM_CLASSES` | `True` | Teste de inversão de ordem das classes |
| `RUN_VIES_NOMES_ATRIBUTO` | `True` | Teste de nomes semânticos nas features |
| `RUN_E_PASSO4_DILUICAO` | `True` | Experimento de diluição |
| `RUN_A_PASSO4_R3R4` | `True` | R3 (x1·x2) e R4 (x1², x2²) no Problema A: desde 09/10/2026 DERIVADO das linhas zero-shot do Passo 4 do A (0 reestimação); grava `problema_A_r3/r4.csv`, o CSV `bloco1_a_passo4_r2_r3_r4` e a figura `bloco1_graphic_08` |
| `RUN_PROBLEMA_E2_E3` | `True` | Múltiplas configs de especialista no Passo 3 de E (zero-shot coletado 1× por (modelo, seed) e reaproveitado em E1/E2/E3) |
| `RUN_A_PASSO4_PERCEPTRON_VS_NNLS` | `True` | Comparação Perceptron × NNLS PAREADA (mesmos rótulos do LLM em A/B/C, 0 chamadas; a recoleta de B/C de 1.200 chamadas foi removida em 09/10/2026) |
| `RUN_ORACULO` | `True` | Validação: algoritmos recuperam W conhecido? (rótulos pela regra do centróide mais próximo sob o W verdadeiro) |
| `RUN_D_PASSO5_TETO` | `True` | Sub-flag do oráculo: aproximação da meia-lua em R2/R3/R4 (0 chamadas LLM) |
| `RUN_E_PASSO5_ORDEM` | `True` | Teste de viés de ordem dos exemplos few-shot |
| `RUN_VIES_VARIANTES_PROMPT` | `True` | Teste de múltiplas variantes de prompt |
| `RUN_E_PASSO6_BASELINES` | `True` | Comparação com baselines clássicos (k-NN, LR, SVM) |
| `RUN_PROBLEMA_D` | `True` | Bloco 1: meia-lua (Problema D, 150 pts, SEM split), Passos 2 a 6 (zero-shot, few-shot do rótulo real, estimação em R²/R³/R⁴, teto, SVM) |
| `RUN_PROBLEMA_F` | `True` | Bloco 2: meia-lua com split 70/30: Problema F, Passos 1 a 4 (`problem_name` = `F_meia_lua_seed{N}`; antes de 08/10/2026 `meia_lua_seed{N}`) |
| `RUN_PROBLEMA_G` | `True` | Estudo de caso real peso×altura (Problema G = G1 normalizado + G2 kg/m; homem/mulher, elipse). Desde 13/10/2026 os Passos 2 a 5 nos 70 de treino seguem os do A (zero-shot + few-shot do rótulo real, k ∈ `K_EXEMPLOS_ROTULO_REAL`, + teto e eleição; `com_passos_3_a_5=True`). Desde 14/10/2026 o Passo 6 roda com 2, 3 e 4 atributos no prompt (`ATRIBUTOS_PROMPT_PASSO6_G`) |
| `RUN_PROBLEMA_G_CLASSES_AB` | `True` | Sub-flag do Bloco 3: variante neutra de nomes de classe (A/B) em G1 e G2 |

### Configurações de Especialistas Múltiplos (Problema E, Passo 3)

```python
EXPERT_CONFIGS = [
    {"name": "E1", "w": [0.3, 1.5], "desc": "x2 dominante (principal; antigo aniso_x2)"},
    {"name": "E2", "w": [1.5, 0.3], "desc": "x1 dominante (antigo aniso_x1)"},
    {"name": "E3", "w": [1.0, 1.0], "desc": "euclidiano (antigo euclidean)"},
]
```

### Configurações do oráculo

O oráculo roda exatamente as três configurações da seção IV.5 do texto do orientador, desde
14/10/2026 todas sobre UM MESMO conjunto, as gaussianas do Problema E (`EXPERT_CENTROIDS`, 150
pontos): sobre E3 (W = [1, 1], Problema 1), sobre E1 ([0,3; 1,5], Problema 2) e sobre E2
([1,5; 0,3], Problema 3), "Problemas 1/2/3 com as métricas na frente" (reunião de 02/10/2026).
Até 13/10/2026 o euclidiano rodava sobre os dados do B ("Oráculo sobre B", `expert_name` `"B"`;
até 08/10/2026 `"euclidiana"`); os leitores (`plots.py`, `gerar_tabelas.py`) aceitam esses nomes
nas execuções antigas (Problema 1 = B) e IGNORAM linhas de outras configurações. O Problema A não
serve de base até 14/10/2026: com centróides (−2, 0) e (2, 0) a diferença de distâncias não dependia de w2 e a
métrica não era identificável (desde 15/10/2026 o A tem centróides (−2; −0,5) e (2; 0,5), mas o oráculo segue sobre o E) (o texto do orientador, Resultados V.2, dizia "dados do problema A").
Em todas os rótulos vêm da regra do centróide mais próximo sob o W verdadeiro (desde 09/10/2026;
antes a Parte 1 usava os rótulos geradores do make_blobs).

As três gaussianas anisotrópicas extras que existiam até 13/10/2026 (`ORACLE_ANISO_CONFIGS`,
`x2_dom`/`x1_dom`/`forte_aniso`, gerador `create_anisotropic_problem`, flag
`RUN_ORACLE_ANISO_EXTRAS`) foram REMOVIDAS do código por decisão do autor em 13/10/2026: com
os centros afastados qualquer W diagonal separava os dados, o Perceptron devolvia W = [1, 1] com
acurácia 100% e o teste não media o estimador (falta de identificabilidade, não falha). A completa
de 08/10/2026 ainda tem essas linhas no CSV; tabela 17 e gráfico 19 as descartam.

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

**Modelos** (`MODELS_TO_TEST`, formato `(provider, model, temperature)`; até 09/10/2026
havia um 4º campo `scope`, removido junto com `EXTRA_SEEDS_CORE`):
- Protocolo atual: **2 modelos**, o grid COMPLETO de experimentos roda em ambos, com 3
  seeds uniformes:
  `openai/gpt-4o-mini` e `google/gemini-2.5-flash-lite`, AMBOS via **OpenRouter**,
  provedor único (`OPENROUTER_API_KEY`; `extra_body` com `allow_fallbacks=False`).
  Os provedores diretos (OpenAI, Gemini, Anthropic) foram removidos de `llm_client.py`
  em 01/10/2026: para usar outro modelo, acrescente-o a `MODELS_TO_TEST` com o id do
  OpenRouter (`<org>/<modelo>`), um alias em `MODEL_ALIAS` e um pin em `MODEL_PROVIDER_PIN`.
- **Pinagem de provedor de inferência POR MODELO** (`MODEL_PROVIDER_PIN` em
  `src/llm_client.py`): fixa qual infraestrutura serve cada modelo no OpenRouter
  (mesmo model-ID pode ser servido por empresas/quantizações distintas). Valor
  observado no smoke de 02/07/2026: Gemini → `Google`; GPT-4o-mini → `OpenAI`.
  Se o provedor pinado cair, as chamadas falham visivelmente (erro retentável, depois
  resposta malformada registrada e visível na auditoria) em vez de migrar em silêncio:
  comportamento desejado p/ reprodutibilidade.
- CLI `--modelo <substring>` filtra `MODELS_TO_TEST` (smoke test isolado).
- CLI `--apenas <bloco> [...]` (`bloco1|bloco2|bloco3|oraculo`) roda só os blocos
  indicados; pasta `_apenas-<blocos>` (nunca `_completa`).
- **Todos os modelos são iguais para outputs**: cada um gera o conjunto completo de
  PNGs por modelo na raiz da execução, com seu alias (`MODEL_ALIAS`) no nome do asset.
  CSVs consolidam todos (colunas `provider`/`model`); `final_graphic_08_comparacao_modelos.png`
  compara todos. Assets independentes de LLM (oráculos, especialista, overviews sintéticos,
  baselines clássicos) são gerados 1× (gate `model_idx == 0`).

**Regra única de repetição** (`reps_para` em `dissertacao_mestrado.py`): repetições
existem para variar o SORTEIO dos exemplos few-shot (Passo 3 de A, D e G; Passo 3 de E;
Passo 3 do F; Passo 6 do G: `random_state = seed + rep`). Onde a seleção é determinística (zero-shot, seleção por
margem no Passo 4 de B e C, diluição, ordenações fixas) roda-se **1 coleta**; o
não-determinismo por consulta (flip a T=0) é quantificado pela auditoria offline. Não há
exceção: a comparação Perceptron × NNLS é pareada sobre os mesmos rótulos (0 chamadas) e
o zero-shot do Problema E (Passo 2) é coletado 1× por (modelo, seed) e reaproveitado em E1/E2/E3.

## Arquitetura do Código

### Estruturas de Dados Principais (`src/resultados.py`)

```python
@dataclass
class ResultadoABC        # Resultado dos Problemas A, B e C (Bloco 1)
    # provider, model_name, temperature, random_seed, k_passo4_bc, nomes_classes, repeticao
    # consistencia/kappa/f1 (F1-Macro) nos Problemas B e C, com centróides LOCAIS;
    #   collapsed, collapsed_b/c; ablação consistencia_centroides_a_problema_{b,c} e
    #   kappa_centroides_a_problema_{b,c} (centróides do A; só CSV e log)
    # acuracia_{llm,metrica}_vs_gt_problema_{a,b,c} (contexto "vs rótulo real")
    # w_aprendido (Ŵ_LLM estimada, Perceptron), gamma_otimo
    # fidelidade_problema_a (Perceptron), w_cosine_sim_nnls (a fidelidade do NNLS vai só ao CSV bloco1_a_passo4_perceptron_vs_nnls e ao log)
    # w_ratio, w_direction, feature_names, prompt_variant
    # n_classe_*, n_disagreements_*, n_malformed_responses
    # (consistencia_euclidiana_{b,c} e diagonal_limitation_flag foram REMOVIDOS em 09/10/2026)

@dataclass
class ResultadoAprendiz  # Resultado do Problema E, Passos 2 a 5 (Bloco 2)
    # accuracy/kappa/f1 (F1-Macro) LLM vs. especialista; accuracy_expert_vs_gt e
    #   n_class_*_expert medidos sobre o conjunto de TESTE; accuracy_llm_vs_gt
    # example_strategy: "easy"/"hard"/"mixed"/"random", ou "mixed_order_<ordem>"
    #   (viés de ordem) ou "dilution_<N>hard_<M>easy" (diluição)
    # expert_w, expert_name (E1 | E2 | E3; execuções até 07/10/2026: aniso_x2 | aniso_x1 | euclidean)

@dataclass
class LearnedMetric               # Métrica Mahalanobis estimada
    # vetor w, centroids, gamma, source_problem
```

O Bloco 3 e os Problemas F e G (`bloco23_*`) usam dicts (ver `fg_executar`).

### Funções Críticas

| Função | Propósito |
|---|---|
| `create_problem_{a,b,c}()` | Gera dados sintéticos 2D lineares (`data_problems.py`) |
| `create_problem_e_expert()` | Problema E (especialista linear do Bloco 2) |
| `create_problem_d_meialua()` | Meia-lua (`sklearn.make_moons`) — Problema D (Bloco 1) / F (Bloco 2) |
| `create_problem_homem_mulher()` / `create_problem_homem_mulher_kg_m()` | Carrega base real peso×altura normalizada (G1) / em kg e m (G2, só para o prompt), 100 amostras, elipse |
| `async_llm_classify_point()` / `async_llm_classify_point_openai()` | Chamam a API (OpenRouter, endpoint compatível com OpenAI) |
| `collect_llm_decisions_lote(jobs, rotulo)` | 15/10/2026: várias coletas (dicts com os argumentos de `collect_llm_decisions`, inclusive `tag`) num ÚNICO laço de eventos, cliente e semáforo compartilhados (`semaphore=` em `async_collect_llm_decisions`); devolve `(y_llm, n_malformadas)` na ordem. Usada no Passo 3 de A/D/G (`adg_passos_2_a_5`) e no Passo 6 do G / Passo 3 do F (`fg_executar`: plano → lote → medidas). Coletas de 30 a 110 pontos eram limitadas pela latência (0,9 h só no Passo 6 do G na completa de 14/10). Com dublê de `collect_llm_decisions` (testes) ou 1 job, chama uma a uma e mantém `CURRENT_TAG` |
| `async_collect_llm_decisions()` / `collect_llm_decisions()` | Coleta paralela com `asyncio.Semaphore` (+ wrapper síncrono). Desde 09/10/2026 a corrotina cria o `AsyncOpenAI` dentro do laço de eventos quando o global `async_client` é `None` (produção; o global é só ponto de injeção dos testes) e o fecha no `finally` com o laço aberto; o 402 atravessa o `gather` e o wrapper chama `_abortar_por_creditos` fora do laço |
| `parse_llm_response()` | Parser de 8 camadas (0–7) (`llm_parser.py`): camada 0 usa o ÚLTIMO marcador ("Answer:", "Final Answer", …) e limpa markdown/LaTeX (`**A**`, `\boxed{A}`); marcador final e última linha decidindo classes diferentes = resposta contraditória = malformada; camada 3 ignora dígitos de decimais e usa o último inteiro; camada 6 usa padrões com a classe explícita; camada 7 exige fronteira de palavra e não decide se a outra classe também aparece; nomes de 1 letra são case-sensitive; fallback por hash MD5 determinístico |
| `audit_interactions.audit()` | Auditoria offline dos `llm_interactions*.json`: malformadas, flip de `T=0`, breakdown por modelo e por regime (`tag`) |
| `train_relaxed_perceptron()` | Perceptron Estruturado com relaxação de margem (`**PERCEPTRON_PARAMS`; γ = 0 = melhor esforço, marcado com † nas tabelas) |
| `train_least_squares_inverse()` | NNLS via scipy (mínimos quadrados não-negativos) |
| `compute_consistency_metrics()` | `ConsistencyMetrics` (`metrics.py`): acurácia da concordância (= Fidelidade em A/D, Consistência em B/C, Acurácia LLM vs especialista no Bloco 2), Cohen's κ, F1-Macro (`average='macro'`), contagens. Precision/recall foram removidos em 09/10/2026 |
| `colapsou()` | Regra única de colapso: classe única ou minoritária < `MIN_PONTOS_CLASSE_MINORITARIA` |
| `bloco_protegido` | Context manager: exceção num bloco → `falhas_fatais` + checkpoint; a execução continua e termina com exit 2 |
| `a_passo2_zero_shot_e_metrica()` | Passos 2 e 4 do A (coleta zero-shot e estimação): retorna 2 métricas (Perceptron, NNLS) simultâneas; o log usa "Passo 4 (Perceptron)"/"Passo 4 (NNLS)" |
| `passo4_estimar_metrica_r2_r3_r4()` | Passo 4 parametrizado por `n_features ∈ {2,3,4}`; fidelidade vs LLM **e** acurácia vs rótulo real |
| `bc_passos_2_a_4()` | Passos 2 a 4 de B e C (coleta zero-shot, consistência e exemplos da métrica); NÃO roda quando o A colapsou no Passo 2 (a linha do ABC sai com `collapsed=1` e B/C sem coleta): consistência com w do A e centróides LOCAIS (dos rótulos do LLM em B/C); colapso → NaN + `collapsed`; ablação com centróides do A (`consistency_centroides_a`); os exemplos few-shot de B/C (H2) são rotulados por w_A com os centróides de A (decisão D4, declarada nas docstrings e rodapés); retorna também `X_test` |
| `adg_passos_2_a_5(X, y_true, nome_classe_0, nome_classe_1, problem, seed, provider, model_name, temperature, n_features_list, ks_rotulo_real, y_llm_zero_cached, n_malformed_zero, verbose, label_prefix, X_display=None, nome_feature_0="x1", nome_feature_1="x2", bloco="bloco1", tag_extra="")` | **Passos 2 a 5** de A, D e, desde 13/10/2026, do treino do G (sem split em A e D): zero-shot + few-shot com exemplos do GROUND TRUTH (`passo3_exemplos_rotulo_real(..., X_display=None)`, k/2 por classe, `seed+rep`) + W ajustado ao GT (teto), em R²/R³/R⁴; linhas do CSV `bloco1_ad_passos_2_a_5` (A, D) ou `bloco3_g_passos_2_a_5` (G). `X_display` = valores mostrados ao LLM (kg e m no G2), `nome_feature_*` = nomes de atributo do prompt, `bloco`/`tag_extra` = tag de auditoria. A chave de `CACHE_ROTULOS_LLM` inclui o nome completo do problema e os nomes de atributo |
| `_tag(bloco, problema, passo, regime, k, rep, seed, extra="")` | Monta a tag de auditoria `<bloco>\|problema=<P>\|passo=<N>\|regime=<regime>\|k=<k>\|rep=<r>\|seed=<s>` (7 campos fixos) e, se houver, `\|<extra>` (no G: `nomes=Mulher/Homem\|atributos=peso/altura`; no Passo 6 do G e no Passo 3 do F ainda `\|n_atributos_prompt=<2\|3\|4>`, nº de atributos mostrados ao LLM, 14/10/2026). Até 13/10/2026 o 2º campo era `fase=` (`_make_tag`); os logs antigos continuam legíveis pela ferramenta de migração |
| `bc_passo3_consistencia()` / `a_passo6_eleger_metrica()` | B/C zero-shot contra todas as métricas do A em R² (0 chamadas; centróides locais) e eleição da melhor (maior fidelidade média no A, Perceptron, R²) |
| `_plot_simbolos_passos()` | Figuras `bloco1_graphic_02/11/13` (símbolos por classe, P&B) a partir dos rótulos guardados dos Passos 2 a 4; painéis na ordem Passo 1, 2, 3 (LLM k-shot), 4 (métrica) |
| `e_passos_2_3_aprendiz()` | Passos 2 e 3 de E: LLM aprendendo do especialista (sub-etapas internas (a) a (e) no log, sem numeração de passo) |
| `fg_executar(..., X_display=None, expert_labels=None, com_passos_3_a_5=False, atributos_prompt_aprendiz=(2,))` / `fg_exemplos_aprendiz()` | Sequência de passos dos Problemas F, G1 e G2 (`atributos_prompt_aprendiz`, 14/10/2026: nº de atributos mostrados ao LLM no Passo 6 do G, `ATRIBUTOS_PROMPT_PASSO6_G` = (2, 3, 4), fator independente da métrica eleita; k = 0 coletado 1×; o Perceptron do Passo 7 usa o mesmo espaço): variantes de nome de atributo × `n_features` × seeds → melhor métrica → aprendizado em contexto (Passo 3 do F, Passo 6 do G); com `com_passos_3_a_5=True` (G1 e G2; F passa False), depois do zero-shot dos 70 pontos de treino roda `adg_passos_2_a_5` sobre eles (zero-shot reaproveitado, 0 chamadas extras; few-shot k ∈ `K_EXEMPLOS_ROTULO_REAL` com exemplos do rótulo real, 3 reps; teto; R²/R³/R⁴; o LLM vê só os 2 atributos, em kg e m no G2 via `X_display`) e devolve as linhas em `'passos_2_a_5'` (`problem_name`, `subproblem` G1/G2, `class_names`, `feature_names`); `expert_labels=` (F) faz o especialista ser o rótulo real (`especialista_origem=rotulo_real`), senão a métrica eleita no Passo 5 (`metrica`); sorteio dos exemplos (Passo 3 do F, Passo 6 do G) BALANCEADO pelo RÓTULO REAL do treino (`fg_exemplos_aprendiz(y_train, k, seed + rep)`, k/2 por classe real, regra de 09/10/2026): índices idênticos entre variantes de atributo e mesma regra em F e G; os exemplos são ROTULADOS pelo especialista (rótulo real no F, melhor métrica no G, onde o balanço pelo rótulo do especialista pode não ser exato: contagem impressa no log) |
| `g_passo5_eleger_especialista(proto, results_by_nfeat)` | Passo 5 do G: eleição do especialista (13/10/2026). Com `proto` (G1/G2): entre as linhas do Perceptron dos regimes `passo2_zero_shot` e `passo3_exemplos_rotulo_real` (todos os k, rep e n_features), excluídos o teto (`passo5_teto`) e os colapsos, a de maior `accuracy_metric_vs_true_full` (acurácia vs rótulo real nos 70 de treino); empate → zero-shot, k crescente, rep, R² → R³ → R⁴. Sem `proto` (F): a métrica zero-shot de maior `accuracy_perc_vs_true`. Devolve `metrica_regime`/`metrica_k`/`metrica_rep` (→ `especialista_regime`/`especialista_k`/`especialista_rep` em `bloco23_fg_aprendiz`). Consequência: o Passo 6 do G roda mesmo com o zero-shot colapsado (ex.: gpt no G1 Mulher/Homem). Log: linha do especialista eleito ("regime=… k=… rep=… R…") |
| `oraculo_passos_1_a_3(X_e, EXPERT_CONFIGS, seed)` / `d_passo5_teto()` | Oráculo sobre E3 (W=[1,1]), E1 e E2, as três métricas sobre as gaussianas do E (14/10/2026; até 13/10 o euclidiano era sobre B; sem Problema A, pois w2 não é identificável nele; sem configurações extras desde 13/10/2026), rótulos pelo centróide mais próximo sob o W verdadeiro, coluna `accuracy` / aproximação da meia-lua em R2/R3/R4 (equivale ao regime `passo5_teto` do D) |
| `_oraculo_estimadores()` / `_oraculo_registrar()` | Helpers da Validação do oráculo |
| `e_passo3_selecionar_exemplos()` / `bc_passo4_exemplos_da_metrica()` | Estratégias de seleção (easy/hard/mixed/random) / seleção por margem (Passo 4 de B e C) |
| `e_passo4_exemplos_diluicao()` | Seleção para experimento de diluição |
| `e_passo5_reordenar_exemplos()` | Reordena exemplos para teste de recency bias |
| `augment_to_r3()` / `augment_to_r4()` / `augment_features()` | Projeções R3 (x1·x2), R4 (x1², x2²) e wrapper por `n_features` (`metrics.py`) |
| `prompt_zero_shot{,_variante}()` / `prompt_com_exemplos{,_variante}()` | Prompts (padrão e variantes) |
| `reps_para()` | Regra única de repetição (1 coleta onde a seleção é determinística) |
| `flags_para_apenas()` / `sufixo_apenas()` | CLI `--apenas`: flags por bloco e sufixo da pasta |
| `nome_pasta_execucao()` / `grades_modo_rapido()` / `aplicar_modo_rapido()` | Extraídas de `main()` em 09/10/2026: nome `execucao_<timestamp>_<completa|smoke|apenas-<blocos>[_smoke]>`; cópia das grades de `GRADES_MODO_RAPIDO`; aplicação ao módulo (ou a um dict) no `--rapido`. Testadas em `tests/test_infra_main.py` |
| `sintese_r2_r3_r4()` | Tabela cruzada linear (Problema A, R2/R3/R4) × não-linear → `final_cross_linearity__<alias>.csv` |
| `print_a_passo4_erros_por_margem()` / `print_a_passo4_sensibilidade_perceptron()` / `print_e_passo5_ordem()` / `print_sumario_estatistico()` / `resumo_entre_sementes()` | Relatórios do log (`relatorios.py`); sem bootstrap desde 13/10/2026 (pedido do orientador): média e faixa mín–máx entre sementes, diferenças pareadas por semente ("consistente" quando a faixa não contém zero), Cohen's d; sem Wilcoxon (removido em 13/10/2026 por decisão do autor) |
| `texto_h1(resultados, limiar=LIMIAR_H1_KAPPA)` | Veredito de H1 por modelo (09/10/2026), impresso por `print_sumario_estatistico` no bloco "H1 (CONSISTÊNCIA): VEREDITO POR MODELO": κ médio em B e C (configuração base A/B, prompt default, x1/x2, zero-shot, Perceptron, sem colapso), faixa de Landis e Koch, seeds com κ > 0,4 e a linha "H1 (κ > 0.4): SUSTENTADA/NÃO SUSTENTADA em B \| ... em C" |
| `malformadas_e(df)` | `(few_shot, zero_shot)` malformadas do Problema E (Passos 2 e 3) sem contar as 12 cópias do zero-shot (uma vez por (modelo, seed, rep)); o log imprime "few-shot X + zero-shot Y (contado uma vez por seed) = Z" |
| `plot_fg_rotulos_llm()` / `plot_d_passo6_svm()` / `plot_a_passo4_gamma()` / `plot_d_passo5_teto()` / `plot_oraculo_recuperacao_w()` | `final_graphic_07`, `bloco1_graphic_14`, `bloco1_graphic_09`, `bloco1_graphic_21`, `bloco1_graphic_19` (`plots.py`); a função do antigo gráfico 20 foi apagada em 09/10/2026 |
| `plot_rotulos_simbolos()` / `plot_passos_2_a_5_curva(rows, filename, algorithm, panels=None, ncols=None, suptitle=None)` / `plot_g_passos_2_a_5_curva(rows, filename, algorithm, n_features=2)` | Figuras P&B com símbolos (`bloco1_graphic_02/11/13`) e curva × k (`bloco1_graphic_04`); `plot_g_passos_2_a_5_curva` (13/10/2026) monta os 8 painéis (subproblema × classes × atributos) de `bloco3_graphic_03` a partir de `bloco3_g_passos_2_a_5` e delega a `plot_passos_2_a_5_curva` com `panels`/`ncols=2`/`suptitle` |
| `tab_g_passos_2_a_5(D, model)` | Tabela `bloco3_table_06_G_passos_2_a_5` (`gerar_tabelas.py`, chave `g_passos_2_a_5` de `carregar_tudo`): grupos por subproblema × classes × atributos; linhas zero-shot, cada k e teto; colunas Acur. LLM vs real, inv. perm. (variantes A/B), Fid. (vs LLM) R²/R³/R⁴ Perceptron, Acur. vs real R²; † de γ = 0; rodapé com os 70 pontos e "k = 40 deixa 30 para estimar" |
| `audit()` → `por_regime` | Auditoria por (problema, passo, regime, k) via campo `tag` dos registros, formato de 13/10/2026 `<bloco>\|problema=<P>\|passo=<N>\|regime=<regime>\|k=<k>\|rep=<r>\|seed=<s>[\|extra]`; devolve a chave `provider/modelo\|problema\|passo=N\|regime\|k=K`. `passo=2` = zero-shot (A, D, B/C, E, F, G), `passo=3` = exemplos do rótulo real (A, D, G) ou estratégias (E) ou aprendiz (F), `passo=4\|regime=exemplos_da_metrica` = B/C com exemplos da métrica, `passo=4`/`passo=5` = diluição/ordem (E), `passo=6` = aprendiz do G. Vieses: `passo=2` (k = 0, A e B/C zero-shot) ou `passo=4` (k > 0, B/C com exemplos da métrica) com `regime=vies_<fator>`, sempre com `k=` e o fator (`nomes=A/B`, `variante=cot`, `atributos=altura/peso`); o zero-shot do Problema E leva `passo=2\|regime=passo2_zero_shot`. Problemas G e F: `nomes=` e `atributos=` depois dos 7 campos fixos, ex. `bloco3\|problema=G1_homem_mulher\|passo=3\|regime=…\|k=4\|rep=0\|seed=42\|nomes=Mulher/Homem\|atributos=peso/altura`. Só o formato novo é aceito; logs anteriores a 13/10/2026 (campo `fase=`) passam antes por `renomear_assets_legado.py` |
| `llm_asset()` / `asset_variant()` / `_model_alias()` | Nomes de asset com alias e painéis (`execucao_io.py`) |
| `Tee` | Duplica stdout/stderr para terminal + buffer (→ log_execucao.txt) |

### Fluxo de Execução (ordem real do `main()`)

```
 1. Criar pasta de execução (`nome_pasta_execucao`: timestamp + sufixo; `--rapido` aplica `GRADES_MODO_RAPIDO`); carregar .env; validar chaves de API
 2. Por (modelo, seed): gerar A, B, C, D (meia-lua), E + salvar dados_sinteticos_seed{seed}/ (problema_{A,B,C,D,E}.csv)
 3. [seed0/model0] bloco1_01 (A/B/C), bloco2_01 (especialista), bloco2_03 (estratégias)
 4. [RUN_PROBLEMAS_ABC] BLOCO 1 — Problema A, Passos 2 e 4 (zero-shot → Ŵ_LLM Perceptron + NNLS) e Problemas B/C, Passos 2 a 4
    (few-shot por margem em K_EXEMPLOS_METRICA_BC; `--bc-zero-shot` → só Passos 2 e 3), p/ cada nome
    de classe; dentro: [RUN_VIES_ORDEM_CLASSES] nomes invertidos (K_VIESES), [RUN_VIES_VARIANTES_PROMPT] variantes
    (K_VIESES), [RUN_VIES_NOMES_ATRIBUTO] nomes semânticos (só zero-shot, via _abc_executar_configuracao:
    consistência com centróides locais), [RUN_A_PASSO4_PERCEPTRON_VS_NNLS] Perceptron × NNLS pareado (0 chamadas).
    Cada (modelo, seed) dos Blocos 1 e 2 roda dentro de `bloco_protegido` (exceção → exit 2 no fim)
 4b. [RUN_PROBLEMAS_ABC] Problema A, Passos 2 a 6: zero-shot (cache) + few-shot com
    exemplos do ground truth (k ∈ K_EXEMPLOS_ROTULO_REAL × reps_para) + W ajustado ao GT, em R²/R³/R⁴;
    B/C zero-shot avaliados contra todas as métricas do A em R² (bc_passo3_consistencia, 0 chamadas)
 4c. [RUN_PROBLEMA_D] Problema D, Passos 2 a 6 (mesmos passos do A; meia-lua, 150 pts, sem split) + bloco1_graphic_12
    (overview) + bloco1_graphic_14 (SVM vs LLM) + bloco1_graphic_13 (símbolos)
 5. [RUN_ORACULO] oráculo (model0): recuperação de W sobre E3/E1/E2 (sobre B até 13/10/2026; sem transferência;
    só as gaussianas do E); [RUN_D_PASSO5_TETO]
 6. [RUN_PROBLEMA_E] BLOCO 2: Problema E, Passos 2 e 3 (zero-shot coletado 1× por (modelo, seed); especialista principal;
    [RUN_PROBLEMA_E2_E3] 3 especialistas); [RUN_E_PASSO6_BASELINES] Passo 6, k-NN/LR/SVM (model0);
    [RUN_E_PASSO4_DILUICAO] Passo 4; [RUN_E_PASSO5_ORDEM] Passo 5
 7. [RUN_A_PASSO4_R3R4] R3/R4 do Problema A derivados das linhas zero-shot do Passo 4 do A (0 reestimação;
    grava problema_A_r3/r4.csv, bloco1_a_passo4_r2_r3_r4 e bloco1_graphic_08)
 8. [RUN_PROBLEMA_G] BLOCO 3: Problema G, Passos 1 a 7, G1 (normalizado: x1/x2, peso/altura) e depois
    G2 (prompt em kg/m: peso_kg/altura_m e x1/x2 como controle; `X_display`), ambos com `com_passos_3_a_5=True`
    (Passo 2: zero-shot nos 70 de treino × variantes de nome, 2/3/4 atributos; em seguida `adg_passos_2_a_5` sobre os
    mesmos 70: Passo 3 few-shot k ∈ K_EXEMPLOS_ROTULO_REAL com exemplos do rótulo real, 3 reps, Passo 4 estimação + teto, R²/R³/R⁴,
    Perceptron e NNLS, zero-shot reaproveitado → `bloco3_g_passos_2_a_5` → Passo 5: especialista ELEITO entre zero-shot e
    few-shot por `g_passo5_eleger_especialista` → Passo 6 com especialista = métrica eleita (Passo 7: Perceptron nos mesmos exemplos); exemplos sorteados k/2
    pelo rótulo REAL do treino via `fg_exemplos_aprendiz(y_train, k, seed + rep)`, idênticos entre variantes,
    rotulados pela métrica); [RUN_PROBLEMA_G_CLASSES_AB] variante A/B
 9. [RUN_PROBLEMA_F] Problema F, Passos 1 a 4 (Passo 3 com especialista = rótulo real; mesma regra
    de sorteio do G; `com_passos_3_a_5=False`: só zero-shot no Passo 2; bloco1_graphic_12 e bloco1_graphic_14 só se RUN_PROBLEMA_D=False)
10. Visualizações consolidadas: final_graphic_08 (se >1 modelo), bloco1_graphic_19/21 (oráculo)
11. Por modelo: BLOCO 1 (bloco1_graphic_10, 16, 07, 05, 03, 15, 17, 18, 02/11/13 símbolos, 04 curva, 08, 06, 09) →
    BLOCO 2 (bloco2_graphic_03, 04, 05, 06, 07) → FECHAMENTO por seed (final_graphic_01, 03, 04, 02, 05, 06) →
    análises de texto (print_e_passos_2_3 com veredito de H4 calculado e malformadas via `malformadas_e`,
    print_sumario_estatistico com o veredito de H1 por modelo via `texto_h1`, …)
12. Exportar CSVs por bloco (bloco1_* incl. ad_passos_2_a_5 e bc_passo3_consistencia, bloco2_*, bloco23_*), final_cross_linearity__<alias>.csv
13. final_graphic_07_FG_rotulos_llm_* (ponto-a-ponto) e bloco23_fg_* (curvas/fronteiras externas)
14. Resumos consolidados dos Problemas F e G no log; salvar llm_interactions (partes); ETAPA 6b: `gerar_tabelas.gerar_tudo` (todas as tabelas .tex + .png); salvar log
```

## Algoritmos de Otimização Inversa

**Dois** algoritmos independentes estimam a métrica diagonal W. Roda ambos no Passo 4 (A, D e G)
para demonstrar robustez e permitir comparação de similaridade (cosseno entre Ws).

| Algoritmo | Arquivo | Formulação | Hiperparâmetros |
|---|---|---|---|
| **Perceptron Estruturado** | `relaxed_perceptron.py` | Relaxação de margem + busca binária em γ | `PERCEPTRON_PARAMS` (`protocolo.py`): eta=0.001, C=1.0, delta_gamma=0.05, max_epochs=50, tol=1e-4; defaults do construtor fora do dicionário: gamma_init=0.1, max_iterations=50 (teto de iterações da busca em γ). Desde 15/10/2026 (decisão do autor, mudanças B, C e D): (B) ao fim de cada época w e α são divididos por ‖w‖ (a restrição é homogênea em w; sem isso w encolhia até ~10⁻³ e qualquer α > 0 satisfazia a margem relaxada, "γ viável" degenerado, 12 de 495 linhas de A/D na completa de 15/10); (C) o melhor esforço é a época com MENOS pontos mal classificados (margem bruta < 0), não com menos violações relaxadas; (D) a margem bruta mínima/‖w‖ da solução devolvida vai para `ULTIMA_MARGEM_MINIMA`/`margem_minima_` e para a coluna `margem_minima` dos CSVs (`bloco1_abc_passos_2_a_4`, `bloco1_ad_passos_2_a_5`, `bloco3_g_passos_2_a_5`). O teste de viabilidade continua o do texto do orientador (margem relaxada com α). Quando nenhum γ > 0 é viável, devolve W em melhor esforço (γ = 0); † nas tabelas = γ = 0 OU `margem_minima` < 0 (Ŵ não separa os rótulos). Validação de 15/10/2026: oráculo E3/E1/E2 100 % nas 3 sementes; gpt A seed 42 zero-shot W=[0,15; 0,99], fidelidade 0,860 (antes W=[0; 0,00065], 0,753; NNLS 0,847) |
| **NNLS (Mínimos Quadrados)** | `least_squares_inverse.py` | `min ‖Aw - b‖²  s.t. w ≥ 0`, via `scipy.optimize.nnls` | nenhum |

O LP Max-Margin foi **removido** do trabalho (decisão da reunião 30/04/2026 — bug
da restrição de convexidade não respeitada; o módulo foi apagado do repositório, não
existe pasta `src/arquivado/`).
Dois algoritmos congruentes (Perceptron + NNLS) já bastam para demonstrar robustez
do método de estimação.

## Métricas de avaliação (seção IV.3 do orientador)

Fonte: `reunioes_orientador/reuniao_2026_10_02/doc revisados orientador/Metodologia[90].pdf`,
seção IV.3. O sistema calcula, grava, plota e tabula **apenas** estas cinco métricas, com
estes nomes (desde 09/10/2026; a similaridade de cosseno, `w_ratio`, `w_direction` e `gamma`
são comparações de ESTIMADORES, não métricas de avaliação):

| Métrica | Definição (uma linha) | Onde se aplica no código | Nome exibido |
|---|---|---|---|
| **Fidelidade** | Fração dos pontos em que o centróide mais próximo sob o w estimado concorda com o rótulo do LLM: mede o quanto a métrica reproduz o LLM, não a sua correção | Problemas A e D (`fidelity_vs_llm`, `fidelidade_problema_a`, `fidelity_perc/nnls_vs_llm`) e, por extensão (decisão D3), F (Passo 2) e G1/G2 (Passo 4); também a melhor-métrica do A (`metrica_fidelidade_a`) e `fidelity_teto_vs_llm_passo2` | "Fidelidade (métrica vs LLM)", "Fid." nas tabelas |
| **Acurácia** | Fração de concordância com o rótulo real (`y_W = y_GT` ou `y_LLM = y_GT`) ou com o especialista (`y_LLM = y_especialista`) | Oráculo (coluna `accuracy`); Bloco 2 (`accuracy_llm_vs_expert`, Passo 3 de E e do F, Passo 6 do G); como CONTEXTO nos Blocos 1 e 3 (decisão D2: `llm_accuracy_vs_true`, `accuracy_metric_vs_true_*`, `accuracy_teto_vs_true`, `accuracy_perc/nnls_vs_true`), sempre com o rótulo completo e nunca numa coluna de fidelidade; nos pares invertidos e nas classes neutras A/B a coluna "inv. perm." = max(a, 1 − a) | "Acurácia (vs rótulo real)", "Acurácia (LLM vs especialista)"; "Acur. vs real", "Acur. LLM vs real", "Acur. vs espec." nas tabelas |
| **Consistência** | Mesma fórmula da fidelidade, com o centróide mais próximo calculado em relação aos centróides dos NOVOS problemas (rótulos do LLM em B/C); só o vetor w é transferido do A | Problemas B e C (`consistencia_problema_b/c`, `consistency` em `bloco1_bc_passo3_consistencia`); a variante com centróides transferidos do A é ablação (`consistency_centroides_a_b/c`, só CSV e log) | "Consistência", "Consist. B/C" |
| **Cohen's κ** | κ = (P0 − Pe)/(1 − Pe), Pe = Σ_k p_k p'_k; faixas de Landis e Koch (1977): < 0,2 fraca, 0,2–0,4 razoável, 0,4–0,6 moderada, 0,6–0,8 substancial, > 0,8 quase perfeita (`tabelas.faixa_kappa`; κ = 0,8 exato cai em "quase perfeita", decisão M18) | Junto com as anteriores (`cohen_kappa_score(y_metric, y_llm)`) | "Cohen's κ"; rodapé "κ = Cohen's κ (Landis e Koch 1977)" |
| **F1-Macro** | Por classe k: P_k = VP/(VP+FP), R_k = VP/(VP+FN), F1_k = 2 P_k R_k/(P_k+R_k); média entre as classes | Junto com as anteriores: `f1_score(..., average='macro')` em `metrics.py`, `classical_baselines.py` e no Passo 3 do F e Passo 6 do G (até 08/10/2026 havia três variantes, nenhuma macro) | "F1-Macro" |

**Removidos das saídas em 09/10/2026** (não estão na IV.3): consistência euclidiana e
`diagonal_limitation_flag` (campos, colunas `euclidean_consistency*`, teste do log),
precision/recall de `ConsistencyMetrics`, os nomes "Concordância", "Taxa de erro",
"F1-Score"/"F1" e "Kappa" solto, e a recoleta de B/C da comparação de algoritmos. A matriz
de confusão (`final_graphic_02`) fica como instrumento visual, não como métrica.

## Hipóteses do Experimento

| Hipótese | Descrição |
|---|---|
| **H1** (Consistência) | O LLM mantém o mesmo critério decisório implícito. A métrica Ŵ_LLM **estimada** prevê classificações nos Problemas B e C (Passo 3) com Consistência e Cohen's κ ao menos moderados (κ > 0,4, piso de Landis e Koch 1977, limiar do artigo; 0,7 seria "domínio", não "existência" do critério). O veredito é CALCULADO por execução (`relatorios.texto_h1`, `LIMIAR_H1_KAPPA = 0.4`) e impresso no log por `print_sumario_estatistico`, bloco "H1 (CONSISTÊNCIA): VEREDITO POR MODELO": κ médio em B e em C sobre a configuração base (classes A/B, prompt default, x1/x2, zero-shot, Perceptron, sem colapso), faixa de Landis e Koch, seeds com κ > 0,4 e a linha "H1 (κ > 0.4): SUSTENTADA/NÃO SUSTENTADA em B \| ... em C". |
| **H2** (Few-shot amplifica) | Exemplos rotulados pela métrica W (w_A com os centróides de A, decisão D4) aumentam a Consistência LLM-métrica em B/C (Passo 4 de B e C; `bloco1_table_09`). |
| **H3** (LLM como aprendiz) | O LLM consegue aprender o critério de um especialista externo via few-shot (Passo 3 de E): a Acurácia (LLM vs especialista) cresce com k. |
| **H4** (Exemplos difíceis) | Hipótese a priori: "hard" > "easy". O veredito é CALCULADO por execução a partir dos dados do Passo 3 de E (`relatorios.texto_h4`, no log, sobre o especialista E1; `bloco2_table_05` mostra E1/E2/E3). Na completa de 15/10/2026 (E1, média das médias por k > 0): gpt-4o-mini fácil 87,2 % vs difícil 77,9 % (aleatória 89,7 %, mista 86,3 %; em 40-shot a difícil vence, 97,9 % vs 88,2 %, cruzamento em relação à média), Gemini fácil 93,2 % vs difícil 72,4 % (aleatória 92,9 %, mista 91,1 %; sem cruzamento): "H4 REFUTADA nos dois modelos". Interpretação: exemplos prototípicos funcionam como âncoras de classe com poucos exemplos; com muitos, os difíceis delimitam a fronteira. (Histórico, completa de 08/10/2026, média few-shot de 2 modelos × 3 especialistas: easy 87,5% vs hard 76,7%; em 40-shot hard ≥ easy em 5 dos 6 pares (modelo, especialista), empate em gpt/E2 (77,5% vs 77,3%) e random ≥ easy em todos os pares exceto gemini/E1 e gemini/E2.) |
| **H5** (Estabilidade) | O comportamento é reproduzível entre execuções (Passos 4 a 6 do A; 3 sementes; 3 repetições onde há sorteio de exemplos, regra `reps_para`). |

## Passos por problema (detalhe)

### Problema A, Passos 2 a 6 (estimação da métrica; antes de 13/10/2026: Fase A)
O LLM classifica 150 pontos 2D sem exemplos (Passo 2) e de novo com k exemplos do rótulo
real (Passo 3). A partir de cada conjunto de rótulos, estima-se (Passo 4)
uma **métrica de Mahalanobis diagonal** (Ŵ_LLM) com **dois algoritmos simultaneamente**
(Perceptron Estruturado e NNLS). A fidelidade de cada algoritmo no Problema A e a
similaridade de cosseno entre os Ws estimados são registradas. Em A, D, F e G a estimação roda com 2/3/4 atributos (R2/R3/R4). O teto (Passo 5) é o W ajustado
diretamente ao rótulo real e a eleição (Passo 6 do A) escolhe a métrica de maior fidelidade média
para a transferência. No Problema G (G1 e G2; 13/10/2026,
Metodologia IV.7 do orientador) os Passos 2 a 5 sobre os 70 pontos de treino seguem os do Problema A:
zero-shot, few-shot com k ∈ {4, 10, 20, 40} exemplos rotulados pelo RÓTULO REAL (k/2 por classe, `seed + rep`,
3 repetições, o LLM classifica os 70 − k restantes) e teto (W ajustado ao rótulo real), em R²/R³/R⁴ com Perceptron
e NNLS; o LLM vê só os 2 atributos (kg e m no G2). O zero-shot é reaproveitado (0 chamadas extras); o custo são as
(66 + 60 + 50 + 30) × 3 reps chamadas por variante e semente. O Problema F tem só o zero-shot (Passo 2), sem few-shot do rótulo real na estimação.
R³/R⁴ não se aplicam a B e C (problemas lineares, só transferência do w do A).

### Problemas B e C, Passos 2 a 4 (transferência; antes de 13/10/2026: Fases B e C)
O LLM rotula os 100 pontos em zero-shot (Passo 2). Aplica-se cada métrica Ŵ_LLM estimada no A
(só o W; centróides locais) e mede-se Consistência, Cohen's κ e F1-Macro, com destaque para a
métrica eleita no Passo 6 do A (Passo 3, H1). O LLM rotula de novo com k exemplos rotulados pela
métrica do A, k ∈ {4, 10, 20, 40}, para medir se a consistência sobe (Passo 4, H2). Desde 15/10/2026 o A tem
centróides (−2; −0,5) e (2; 0,5); B mantém (−2; 1,5)/(2; −1,5) e C (−2; −1,5)/(2; 1,5), logo a rotação em relação
ao A é 2,0 em B (horária) e 1,0 em C (anti-horária), não mais simétrica (até 14/10/2026 o A tinha x2 = 0 e as
rotações eram ±1,5 de mesma magnitude). Testa se o LLM mantém consistência em distribuições não vistas.
Duas regras explícitas (auditoria de 13/10/2026): (1) quando o LLM colapsa no Passo 2 do A numa
configuração (nomes de classe, variante de prompt, nomes de atributo), B e C NÃO são coletados para ela
(`abc_executar` retorna antes de `bc_passos_2_a_4`; a linha sai com `collapsed=1`); (2) a métrica ELEITA
no Passo 6 do A entra só em `bloco1_bc_passo3_consistencia` (tabelas 06 a 08) e na figura 11; o CSV
`bloco1_abc_passos_2_a_4`, a figura 10, o veredito de H1 (`texto_h1`), as finais 02/03/05, a tabela 09 e
os exemplos do Passo 4 (H2) usam SEMPRE a métrica zero-shot do A (Passo 4 sobre os rótulos do Passo 2).
Em B/C colapsados o runner grava `y_metric_b/c = None` (os painéis de fechamento mostram "sem dados").

### Problema E, Passos 2 a 6 (LLM como aprendiz; antes de 13/10/2026: Fase E)
**Papel invertido:** especialista externo com métrica W_expert conhecida rotula os dados
(Problema E; na meia-lua, Problema F, o especialista é o ground truth, `especialista_origem=rotulo_real`;
no Problema G, a métrica ELEITA no Passo 5, `especialista_origem=metrica`: desde 13/10/2026
`g_passo5_eleger_especialista` escolhe, entre as métricas do Perceptron de zero-shot e few-shot (todos os k, rep e
R²/R³/R⁴, sem o teto nem colapsos), a de maior acurácia vs rótulo real nos 70 de treino, com empate resolvido por
zero-shot, k crescente, rep, R² → R³ → R⁴; `especialista_regime`/`especialista_k`/`especialista_rep` gravam a eleita, e o Passo 6 do G
roda mesmo com o zero-shot colapsado). O LLM classifica os 150 pontos em zero-shot (Passo 2) e depois recebe
exemplos do especialista (Passo 3, few-shot) e deve aprender a reproduzir as classificações. Métrica
principal: Acurácia (LLM vs especialista), com Cohen's κ e F1-Macro; `accuracy_expert_vs_gt`
e `n_class_*_expert` são medidos sobre o conjunto de teste. O zero-shot (prompt sem
especialista) é coletado 1× por (modelo, seed) e reaproveitado em E1/E2/E3 (tag
`passo=2|regime=passo2_zero_shot`).

Passo 3 do F e Passo 6 do G (F, G1, G2; regra de 09/10/2026): os k exemplos são sorteados por
`fg_exemplos_aprendiz(y_train, k, seed + rep)`, k/2 por classe pelo RÓTULO REAL do treino
(classe 0 recebe ⌈k/2⌉, classe 1 ⌊k/2⌋), sem reposição. Como o rótulo real não depende da variante
de atributo nem do especialista, os índices são idênticos entre as variantes (x1/x2, peso/altura,
peso_kg/altura_m) e a regra é a mesma em F e G; os exemplos continuam ROTULADOS pelo especialista
(rótulo real no F, métrica eleita no Passo 5 do G). No G o balanço pelo rótulo do especialista
pode não ser exato; o log imprime, por (problema, seed, variante, k, rep), a contagem por classe real
e por classe do especialista. Decisão do autor (13/10/2026): o objetivo da comparação entre variantes
não é a semântica das classes, então a prioridade é os MESMOS exemplos nas duas rodadas de nomes, com o
balanço mais próximo possível de k/2.

Estratégias de seleção de exemplos (Passo 3):
- **Easy:** Pontos longe da fronteira de decisão (alta margem)
- **Hard:** Pontos próximos da fronteira (baixa margem, ambíguos)
- **Mixed:** exatamente 50% easy + 50% hard (n = 10 → 5 + 5: quando n/2 por classe é ímpar, a classe 0 recebe o easy extra e a classe 1 o hard extra)
- **Random:** Baseline sem estratégia

Múltiplos especialistas testados (quando `RUN_PROBLEMA_E2_E3=True`; os gráficos `bloco2_graphic_03/04/05`
e o dashboard mostram o especialista principal `E1`; o log e o CSV cobrem os 3; nomes antigos entre parênteses):
- **E1 (aniso_x2):** W=[0.3, 1.5], x2 dominante (original)
- **E2 (aniso_x1):** W=[1.5, 0.3], x1 dominante (invertido)
- **E3 (euclidean):** W=[1.0, 1.0], pesos iguais

Os Passos 4 (diluição), 5 (ordem dos exemplos) e 6 (baselines clássicos) do Problema E estão em
"Experimentos Auxiliares"; o Problema F segue os Passos 1 a 4 da seção "Passos por problema".

### Experimentos Auxiliares

#### Validação do oráculo
Verifica se os dois algoritmos recuperam um W conhecido a partir de rótulos atribuídos pelo
próprio W (regra do centróide mais próximo sob o W verdadeiro). Por padrão são três testes,
os Problemas 1/2/3 da seção IV.5 do PDF do orientador, todos sobre as gaussianas do E (150 pts, 14/10/2026):
Oráculo sobre E3 (W = [1, 1]; até 13/10/2026 sobre B, 100 pts), sobre E1 ([0,3; 1,5]) e sobre E2 ([1,5; 0,3]); 2 algoritmos × 3 seeds = 18 linhas em
`bloco1_oraculo_passos_1_a_3`. A métrica é a Acurácia (coluna `accuracy`; painel `_acuracia` de
`bloco1_graphic_19`, que agrega as 3 seeds; `bloco1_table_17`).
Não há outras configurações: as gaussianas anisotrópicas extras foram removidas em 13/10/2026
(ver "Configurações do oráculo").
`RUN_D_PASSO5_TETO` acrescenta a aproximação da meia-lua em R2/R3/R4 (`bloco1_graphic_21`,
`bloco1_d_passo5_teto_*.csv`; equivale ao Passo 5 do Problema D, regime `passo5_teto`).
Responde: "os algoritmos funcionam quando o oráculo realmente existe?" Não há experimento de
transferência no oráculo (o gráfico 20 que o sugeria foi removido em 09/10/2026).

#### Viés de Ordem das Classes
Testa se a ordem "A ou B" vs "B ou A" altera o resultado do LLM, em `K_VIESES` (zero-shot e
10-shot na completa; 4-shot no `--rapido`), com barras pareadas normal × invertido por regime (`bloco1_graphic_15`); a acurácia vs
rótulo real dos pares invertidos é reportada invariante à permutação (max(a, 1 − a)).

#### Nomes Semânticos de Features
Testa se usar "altura"/"peso" ao invés de "x1"/"x2" altera os pesos aprendidos. Só zero-shot;
roda pelo mesmo caminho da configuração base (`_abc_executar_configuracao`), logo a consistência
B/C usa centróides locais como nos demais vieses (`bloco1_graphic_17`, `bloco1_table_15`).

#### Experimento de Diluição (Problema E, Passo 4; só E1)
Fixa 4 exemplos hard e adiciona easy progressivamente (0, 2, 4, 10, 16, 20).
Verifica se em algum ponto os fáceis "diluem" os difíceis e a performance deteriora.

#### Projeções R3 e R4 (Kernel Quadrático)
Sobre os MESMOS rótulos zero-shot do Problema A (o LLM só viu x1, x2), estima a métrica
com 3 pesos (R3: x3 = x1·x2, hipérbole) e 4 pesos (R4: x1², x2², elipse). Se a fidelidade
cresce com a dimensão, o LLM adota implicitamente um critério não-linear; em problema
linear espera-se ganho ~nulo (`bloco1_graphic_08`, `bloco1_a_passo4_r2_r3_r4_*.csv`, colunas
`fidelity_vs_llm` e `accuracy_vs_true`). Desde 09/10/2026 as linhas vêm das linhas zero-shot
do Passo 4 do A (R²/R³/R⁴ já estimados lá; 0 reestimação). Nos Problemas F (Passo 2)
e G (Passo 4) a mesma comparação 2/3/4 atributos vai para `bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4`.

#### Variantes de Prompt (Sensibilidade)
3 variantes extras (geometric, cot, tabular) rodam o Problema A (Passos 2 e 4) e B/C (Passos 2 a 4) com `K_VIESES=[0, 10]`
em todas as seeds, comparadas à `default` do grid principal.
Painel principal: scatter de W aprendidos — se W muda entre variantes, o prompt confunde a medição.

#### Viés de Ordem dos Exemplos Few-Shot (Recency Bias; Problema E, Passo 5, só E1)
Usa os mesmos exemplos (estratégia "mixed") com 4 ordenações diferentes:
- **class0_first:** classe 0, depois 1
- **class1_first:** classe 1, depois 0
- **shuffled:** ordem aleatória
- **alternating:** 0, 1, 0, 1, ...

Testado com k = `K_ORDEM_EXEMPLOS = [4, 10, 20]`.

#### Baselines Clássicos (k-NN, Logistic Regression, SVM; Problema E, Passo 6)
Treina classificadores clássicos nos mesmos exemplos few-shot do Passo 3 de E e avalia no mesmo
conjunto de teste, com Acurácia, Cohen's κ e F1-Macro vs especialista. Responde: "O LLM faz
algo que um classificador trivial não faria?"
- **k-NN:** k ajustado (máx. 5, ímpar); uma única série no gráfico, os k usados no rodapé
- **Logistic Regression:** linear, regularizado
- **SVM:** kernel RBF

#### Comparação de Algoritmos
Perceptron × NNLS lado a lado no Passo 4 do A e no Passo 3 de B e C, PAREADA: as métricas do NNLS em B/C usam os
MESMOS rótulos do LLM já coletados (0 chamadas; `_nnls_bc` em `abc_executar`).
Demonstra que os resultados são robustos ao método de estimação e reporta similaridade de
cosseno entre os Ws estimados. A recoleta de B/C (200 chamadas por seed), que misturava
flip de T=0 com diferença de estimador, foi removida em 09/10/2026.

## Outputs por Execução

### Gráficos (PNG)

Outputs organizados em **3 blocos auto-contidos** + visualizações de fechamento.
Muitos gráficos são salvos tanto como **figura combinada** quanto como **painéis individuais**
(ex.: `bloco2_graphic_03_E1_passo3_curva.png` + `..._{accuracy,f1,kappa}.png`).

**BLOCO 1 — LLM como ESPECIALISTA (otimização inversa)**

| Arquivo | Conteúdo |
|---|---|
| `bloco1_graphic_01_ABC_passo1_dados.png` | Scatter dos Problemas A, B, C lineares |
| `bloco1_graphic_12_D_passo1_dados_seed*.png` | Problema D meia-lua (por seed) |
| `bloco1_graphic_19_oraculo_passos_1_a_3.png` | Sanity: recuperação de W conhecido (oráculo sobre E3, E1 e E2, Problemas 1/2/3, sobre B até 13/10/2026; só eles), 3 seeds agregadas (média, mín–máx); painéis `_w_scatter`, `_ratio` (escala log), `_cosseno`, `_acuracia` |
| `bloco1_graphic_20` | REMOVIDO em 09/10/2026 (não havia transferência no oráculo); a numeração fica com lacuna |
| `bloco1_graphic_21_D_passo5_teto.png` | Sanity: aproximação da meia-lua em R2/R3/R4 (teto do Problema D) |
| `bloco1_graphic_03_A_passo4_erros_seed*.png` | Erros da métrica vs LLM (por seed); painel `_erros_por_margem` com W normalizado por ΣW |
| `bloco1_graphic_05_A_passo4_w_distribuicao.png` | Distribuição de Ŵ_LLM entre sementes e nomes de classe, zero-shot (boxplot dos pesos NORMALIZADOS por ΣW desde 14/10/2026, ratio, scatter; marcador por par de nomes) |
| `bloco1_graphic_06_A_passo4_perceptron_vs_nnls.png` | Perceptron × NNLS (fidelidade no A, Passo 3 de B e C pareado, scatter da direção de W como vetor unitário) |
| `bloco1_graphic_07_BC_passo3_sementes.png` | Consistência B/C por semente, só zero-shot (1 coleta por seed) |
| `bloco1_graphic_10_BC_passo3_consistencia.png` | Consistência, Cohen's κ e F1-Macro em B/C (painéis `_consistencia/_kappa/_f1` do Passo 3 e `_distribuicao`, boxplot que inclui o Passo 4 onde k > 0) |
| `bloco1_graphic_08_A_passo4_R2_R3_R4.png` | R2→R3→R4 no Problema A (painéis `_linear_vs_quadratica`, `_algoritmos_r3r4`), derivado do Passo 4 do A |
| `bloco1_graphic_02_A_passos_2_a_4_simbolos_seed*.png` | Problema A, Passos 2 a 4: GT, LLM zero-shot, métrica zero-shot, LLM k-shot (painel `_passo3_llm_<k>shot`; k = 10 se estiver na grade, senão o menor de `K_EXEMPLOS_ROTULO_REAL`: 10 na completa, 4 no `--rapido`); símbolos + (classe 0) e − (classe 1), centróides como o, exemplos circulados |
| `bloco1_graphic_11_{B,C}_passos_2_3_simbolos_seed*.png` | B e C: GT, LLM zero-shot, melhor métrica do A (centróides locais) + centróides geradores e rotação em relação ao A (2,0 em B, 1,0 em C; coordenadas fixas (−2; ±1,5)/(2; ∓1,5)) |
| `bloco1_graphic_04_AD_passos_2_a_5_curva.png` | Fidelidade, acurácia da métrica e do LLM vs GT em função de k (0 = zero-shot), teto tracejado; painéis A (R²) e D (R²/R³/R⁴); `_nnls` |
| `bloco1_graphic_13_D_passos_2_a_4_simbolos_seed*.png` | Meia-lua: GT, LLM zero-shot, métrica R², métrica R⁴ (elipse), LLM k-shot (painel `_passo3_llm_<k>shot`, mesmo k do `graphic_02`); painéis `_passo1_rotulo_real`, `_passo2_llm_zero_shot`, `_passo4_metrica` (R²) e `_passo4_metrica_r4` (estes dois só existem quando o zero-shot do D não colapsa); colapso anotado no título |
| `bloco1_graphic_14_D_passo6_svm_seed*.png` | SVM RBF vs rótulos zero-shot do LLM na meia-lua (Problema D); fronteira escolhida por fidelidade vs LLM |
| `bloco1_graphic_15_vies_ordem_classes.png` | Viés de ordem/posição das classes: barras pareadas normal × invertido por regime (painéis `_{consistencia_b,consistencia_c,fidelidade}_{0,10}shot`) |
| `bloco1_graphic_18_vies_variantes_prompt.png` | Comparação de variantes de prompt (painel de W com vetor unitário) |
| `bloco1_graphic_16_vies_nomes_classe.png` | Efeito do nome de classes, só os 4 pares base (painéis `_0shot`, `_10shot`) |
| `bloco1_graphic_17_vies_nomes_atributo.png` | Efeito dos nomes semânticos de atributos, só zero-shot (painéis `_consistencia`, `_fidelidade` e `_pesos`, este com vetor unitário) |
| `bloco1_graphic_09_A_passo4_gamma.png` | Diagnóstico da busca binária em γ do Perceptron (Problema A, Passo 4); curvas rotuladas "seed \| nomes \| variante"; desde 14/10/2026 a legenda (36 configurações na completa) fica abaixo da figura, em 4 colunas |

**BLOCO 2 — LLM como APRENDIZ (Problemas E e F)**

| Arquivo | Conteúdo |
|---|---|
| `bloco2_graphic_01_E_passo1_especialista.png` | Problema E (linear especialista) com fronteira do especialista |
| `bloco2_graphic_02_E_passo3_estrategias_exemplos.png` | Distribuição espacial de exemplos por estratégia |
| `bloco2_graphic_03_E1_passo3_curva.png` | Acurácia (LLM vs especialista) vs k por estratégia (painéis `_accuracy`, `_kappa`, `_f1`) |
| `bloco2_graphic_04_E1_passo3_estrategias.png` | Estratégias em Acurácia, Cohen's κ e F1-Macro (painéis por k) |
| `bloco2_graphic_06_E1_passo4_diluicao.png` | Experimento de diluição |
| `bloco2_graphic_07_E1_passo5_ordem.png` | Viés de ordem dos exemplos (recency bias) |
| `bloco2_graphic_05_E1_passo6_baselines.png` | LLM vs k-NN (uma série; os k no rodapé), Regressão Logística, SVM; F1-Macro nos dois lados. Desde 14/10/2026 as figuras do E usam os mesmos nomes das tabelas (Fácil/Difícil/Mista/Aleatória, "LLM (com exemplos)", "viés de recência"; `ESTRATEGIA_PT`/`CLASSIFICADOR_PT` em `plots.py`) e os eixos dizem "k exemplos" |

**BLOCO 3 — Estudo de caso REAL (Problema G)**

| Arquivo | Conteúdo |
|---|---|
| `bloco3_graphic_01_G1_passo1_dados.png` | Base real peso × altura (G1, normalizada) com a elipse ótima desenhada (constante calibrada pela acurácia máxima sobre as 100 pessoas) |
| `bloco3_graphic_02_G2_passo1_dados.png` | A mesma base em kg e m (G2), mesma elipse convertida para o espaço físico |
| `bloco3_graphic_03_G_passos_2_a_5_curva.png` | Problema G, Passos 2 a 5 (13/10/2026): fidelidade (métrica vs LLM), acurácia da métrica e do LLM vs rótulo real em função de k (0 = zero-shot), teto tracejado (W ajustado ao rótulo real pelo mesmo estimador; no G o Perceptron fica em melhor esforço, γ = 0, em todas as sementes, logo o teto é referência e não máximo garantido: métricas do LLM podem superá-lo), em R²; 8 painéis (subproblema × classes × atributos, Mulher/Homem antes de A/B), painéis individuais `_g1_mulher_homem_peso_altura`, `_g1_a_b_x1_x2`, `_g2_mulher_homem_peso_kg_altura_m` etc.; `_nnls` com o NNLS. Leva alias (`plot_g_passos_2_a_5_curva`) |

**BLOCO 2/3 — Problemas F e G (meia-lua + peso × altura)**

| Arquivo | Conteúdo |
|---|---|
| `bloco23_graphic_01_FG_aprendiz_curva.png` | Passo 3 do F e Passo 6 do G: desde 15/10/2026 um painel por (problema, variante de atributo), 9 na completa; em cada painel uma cor/marcador por nº de atributos no prompt (R², R³, R⁴), linha cheia = Acurácia (LLM vs rótulo real), tracejada = Acurácia (LLM vs especialista) |
| `bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4.png` | R2/R3/R4 comparados em peso×altura (G, Passos 2 e 4) e meia-lua (F, Passo 2) |
| `bloco23_graphic_03_FG_fronteiras.png` | Fronteiras de decisão das métricas externas |
| `bloco23_graphic_04_FG_llm_vs_perceptron.png` | LLM in-context vs Perceptron baseline; 25 painéis na completa (F + 8 combinações do G × 3 dimensões do prompt), 4 colunas |

**FECHAMENTO — visualizações detalhadas por seed**

| Arquivo | Conteúdo |
|---|---|
| `final_graphic_02_ABC_matrizes_confusao_seed*.png` | Matrizes de confusão por seed (instrumento visual; rótulo "Fidelidade"/"Consistência" em vez de "concordância") |
| `final_graphic_06_painel_resumo_seed*.png` | Dashboard consolidado por seed |
| `final_graphic_01_ABCE_passo1_dados_seed*.png` | Visão geral dos 4 datasets sintéticos (A, B, C, E) por seed; painel E com os rótulos do especialista |
| `final_graphic_03_ABC_acertos_erros_seed*.png` | Mapa de acertos/erros da métrica vs LLM por problema, por seed |
| `final_graphic_04_A_w_algoritmos_seed*.png` | Scatter de W por algoritmo, por seed |
| `final_graphic_05_ABC_margem_seed*.png` | Análise de margem por seed (1 − fidelidade por faixa de margem, não "taxa de erro") |
| `final_graphic_07_FG_rotulos_llm_*.png` | Scatter LLM-labels por configuração, incluindo as variantes `_classesAB`; painel do especialista (`y_metric`); legendas Mulher/Homem |
| `final_graphic_08_comparacao_modelos.png` | Comparação entre modelos LLM (quando aplicável; sem alias); barras "Problema B"/"Problema C" na legenda |

### CSVs por bloco
Contrato de colunas de 13/10/2026 (nomes de arquivo e colunas do mapa `renomeacao_passos_2026-10-13.md`;
as execuções anteriores, incluindo a completa de 08/10/2026, mantêm os nomes ANTIGOS e precisam de
`renomear_assets_legado.py --aplicar` antes de qualquer leitura: `gerar_tabelas.carregar_tudo` não tem
fallback para nomes antigos). Todos os CSVs usam a coluna `rep` (antes de 13/10/2026 o ABC usava `repetition`).
Novo ← antigo (as setas de 09/10/2026 ficam como histórico):
- `bloco1_abc_passos_2_a_4_{timestamp}.csv`: Bloco 1: Problemas A, B e C (Perceptron; w_ratio, w_direction, feature_names, prompt_variant; `collapsed=1` marca SÓ o colapso no Passo 2 do A, com métricas NaN e FORA das médias/figuras; desde 14/10/2026 o colapso em B ou C NÃO liga `collapsed`, fica em `collapsed_b`/`collapsed_c` com a consistência NaN e a fidelidade do A preservada, logo a linha continua nas tabelas e figuras de vieses). Novas: `collapsed_b`, `collapsed_c` (colapso em B/C → consistência NaN), `consistency_centroides_a_b/c` e `kappa_centroides_a_b/c` (ablação com centróides do A; só CSV e log). Removidas: `euclidean_consistency_b/c`, `diagonal_limitation_flag`
- `bloco1_oraculo_passos_1_a_3_{timestamp}.csv`: Bloco 1: recuperação de W conhecido. 18 linhas (Oráculo sobre E3/E1/E2 × Perceptron/NNLS × 3 seeds; sobre B/E1/E2 até 13/10/2026). `accuracy` ← `fidelity`; `expert_name` `B` ← `euclidiana`; linhas de configurações extras (`x2_dom`/`x1_dom`/`forte_aniso`, até 13/10/2026) são ignoradas pelos leitores; rótulos pela regra do centróide mais próximo em todas as partes
- `bloco1_d_passo5_teto_{timestamp}.csv`: Bloco 1: teto do Problema D (W ajustado ao rótulo real da meia-lua) em R2/R3/R4: `accuracy_vs_true`, `w_recovered_0..3`
- `bloco1_a_passo4_perceptron_vs_nnls_{timestamp}.csv`: Bloco 1: comparação Perceptron × NNLS PAREADA. O CSV contém SÓ as linhas NNLS (uma por modelo × seed, configuração base), estimadas sobre os mesmos rótulos zero-shot do LLM em A/B/C; as linhas Perceptron correspondentes são as de `bloco1_abc_passos_2_a_4` (mesmo modelo e seed, `k_passo4_bc=0`, nomes A/B, prompt default, x1/x2). Sem colunas euclidianas; a recoleta de B/C foi removida em 09/10/2026
- `bloco1_a_passo4_r2_r3_r4_{timestamp}.csv`: Bloco 1: R2/R3/R4 no Problema A, derivado das linhas zero-shot do Passo 4 do A. `fidelity_vs_llm` ← `fidelidade`; nova `accuracy_vs_true`; as linhas R2 gravam `w0, w1` (até a rodada 2 de 09/10/2026 ficavam vazias)
- `bloco1_ad_passos_2_a_5_{timestamp}.csv`: Bloco 1: Passos 2 a 5 de A e D: `regime ∈ {passo2_zero_shot, passo3_exemplos_rotulo_real, passo5_teto}`, `k`, `rep`, `n_features`, `algorithm`, `collapsed` (classe única OU minoritária < `MIN_PONTOS_CLASSE_MINORITARIA`), `fidelity_vs_llm` (NaN no regime `passo5_teto`, que não tem LLM), `accuracy_metric_vs_true_{eval,full}`, `accuracy_teto_vs_true` (nova: acurácia do teto; antes ia em `fidelity_vs_llm`), `llm_accuracy_vs_true`, `fidelity_teto_vs_llm_passo2` ← `accuracy_gt_metric_vs_llm_zero`, `w_0..3`, `w_unit_0..3`, `w_ratio`, `gamma`, `example_indices`
- `bloco1_bc_passo3_consistencia_{timestamp}.csv`: Bloco 1: B/C zero-shot contra cada métrica do A (`metrica_regime`, `metrica_k`, `metrica_rep`, `consistency`, `kappa`, `f1`, `metrica_fidelidade_a`, `eleita_passo6`); colapso do LLM em B/C → NaN + `collapsed_llm=1` (antes usava os centróides do A); sem `euclidean_consistency`
- `tabelas/<bloco>_table_<NN>_<assunto>__<alias>.tex`: Bloco 1 (09/10/2026): 01 `A_passos_2_a_5` (ex-T1), 02 `A_passo2_vs_passo3` (ex-T6), 03 `A_passo6_eleicao` (ex-T7), 04 `A_passo4_R2_R3_R4`, 05 `A_passo4_perceptron_vs_nnls`, 06 `BC_passo3_todas_metricas_A` (ex-T2; só `n_features == 2`), 07 `BC_passo3_por_semente` (ex-T8), 08 `BC_passo3_zero_vs_eleita` (ex-T9), 09 `BC_passo4_exemplos_da_metrica`, 10 `D_passos_2_a_5` (ex-T3), 11 `D_passo2_vs_passo3`, 12 `AD_por_passo`, 13 `AD_passo4_estabilidade_W` (ex-T4; travessão na célula do colapso), 14–16 `vies_nomes_classe/nomes_atributo/variantes_prompt` (14 e 16: regimes zero-shot e o(s) k presente(s) em `K_VIESES`/no CSV, 10-shot na completa e 4-shot no `--rapido`, nunca "10-shot" fixo, 09/10/2026 rodada 2; 15 só zero-shot), 17 `oraculo_passos_1_a_3` (Problemas 1/2/3 = E3/E1/E2, sobre B até 13/10/2026; linhas de configurações extras de execuções anteriores a 13/10/2026 são ignoradas), 18 `D_passo5_teto`, 19 `D_passo6_svm` (16/10/2026: números da figura 14, por modelo); `final_table_04_auditoria_flip_por_regime` (ex-T5). Convenções (09/10/2026): cabeçalhos "F1-Macro", "Acur. vs espec." (Acurácia LLM vs especialista), "Acur. LLM vs real"/"Acur. vs real" (contexto), rodapé "κ = Cohen's κ (Landis e Koch 1977)"; linha do teto com travessão em Fid. e valor em Acurácia (tabelas 01/10/11/12); marcador **†** nas linhas em que o Perceptron devolveu Ŵ em melhor esforço (γ = 0; tabelas 01, 02, 03, 04, 05, 10); coluna **"Acur. vs real (inv. perm.)"** = max(a, 1 − a) onde o nome da classe é arbitrário (zero-shot, pares invertidos, classes A/B: tabelas 01_14, 03_01, 03_02, 03_04, 03_05; o Δ semântico − neutro usa a invariante), com nota de permutação no rodapé; as tabelas 03/07/08 leem a eleição `eleita_passo6` do CSV (regra única do runner). `src/gerar_tabelas_bloco1.py` grava `tabelas/todas_as_tabelas_bloco1.tex/.docx` (não sobrescreve `todas_as_tabelas.tex`, que é do catálogo completo)
- **Catálogo completo de tabelas** (`src/gerar_tabelas.py`, 09/10/2026; `main()` chama na ETAPA 6b, 0 chamadas): cada tabela é um objeto `Tabela` (`src/tabelas.py`) escrito em `.tex` (`tabelas/`) **e** em `.png` na raiz com nome `<bloco>_table_<NN>_<assunto>[__<alias>]_gray.png`. Bloco 1: T1–T9 + `A_passo4_R2_R3_R4`, `A_passo4_perceptron_vs_nnls`, `BC_passo4_exemplos_da_metrica` (H2), `D_passo2_vs_passo3`, `AD_por_passo`, `vies_nomes_classe/atributo/variantes_prompt`, `oraculo_passos_1_a_3` (Problema 1/2/3), `D_passo5_teto`. Bloco 2: `E{1,2,3}_passo3_estrategia_x_k`, `E_passo2_vs_passo3` (média few-shot = média das médias por (estratégia, n)), `E_passo3_H4` (E1/E2/E3), `E_passo6_baselines` (slug ← `_40shot`; LLM no maior n da grade × k-NN/LR/SVM treinados nos exemplos da estratégia mista), `E1_passo4_diluicao`, `E1_passo5_ordem`, `F_passo2_R2_R3_R4`, `F_passo3_k`. Bloco 3: `G_passos_2_e_4_R2_R3_R4`, `G_passo6_k`, `G_passo7_LLM_vs_perceptron`, `G1_vs_G2_passos_2_e_4`, `G_nomes_semanticos_vs_neutros`, 06 `G_passos_2_a_5` (13/10/2026; `tab_g_passos_2_a_5`, lê `g_passos_2_a_5`: grupos por subproblema × classes × atributos, linhas zero-shot/cada k/teto, Acur. LLM vs real e inv. perm. nas variantes A/B, Fid. (vs LLM) R²/R³/R⁴ Perceptron, Acur. vs real R², † de γ = 0, rodapé com os 70 pontos e "com k = 40 sobram 30 para estimar"). Fechamento (`final_table_01..04`): `linearidade_cruzada_R2_R3_R4` (uma linha por (problem_name, feature_names); a meia-lua aparece como `D_meialua`; a nota de permutação do rodapé vale para TODAS as linhas zero-shot, pois os nomes de classe são A/B e a correspondência nome↔classe é arbitrária), `comparacao_modelos`, `auditoria_por_modelo`, `auditoria_flip_por_regime`. Índice em `tabelas/INDICE.md`. Tabelas cujo CSV não existe são puladas (`--apenas`, `--rapido`); a `bloco2_table_09` (Passo 2 do F) sai com uma linha "colapso" quando o zero-shot do F colapsou em todas as sementes do modelo (gpt na completa de 13/10/2026), em vez de sumir.
- `bloco2_e_passos_2_3_{timestamp}.csv`: Bloco 2: Problema E, Passos 2 e 3 (inclui `expert_name`); `accuracy_llm_vs_expert`, `kappa_llm_vs_expert`, `f1_llm_vs_expert` (F1-Macro); `accuracy_expert_vs_gt` e `n_class_*_expert` sobre o TESTE; zero-shot coletado 1× por (modelo, seed) e repetido nas linhas de E1/E2/E3: a MESMA coleta aparece em 12 linhas por (modelo, seed) (4 estratégias × 3 especialistas), só com as métricas recalculadas contra cada especialista. NÃO somar `n_malformed_responses` (nem contar chamadas) entre essas linhas, pois são cópias de uma única coleta; o log usa `relatorios.malformadas_e(df)`, que soma o few-shot normalmente e conta o zero-shot uma vez por (modelo, seed, rep), e imprime "few-shot X + zero-shot Y (contado uma vez por seed) = Z"
- `bloco2_e_passo4_diluicao_{timestamp}.csv`: Bloco 2: experimento de diluição (subconjunto das colunas de `bloco2_e_passos_2_3`: sem expert_name/class_*/n_class_*)
- `bloco2_e_passo5_ordem_{timestamp}.csv`: Bloco 2: viés de ordem dos exemplos
- `bloco2_e_passo6_baselines_{timestamp}.csv`: Bloco 2: baselines clássicos (`accuracy/kappa/f1_vs_expert`, F1-Macro); desde 15/10/2026 coluna `rep` (0 em fácil/difícil/mista; 0 a 2 na aleatória, `random_state = seed + rep`, os MESMOS exemplos do LLM em cada repetição)
- `bloco23_fg_passos_2_e_4_{timestamp}.csv`: Bloco 2/3: Passo 2 do F e Passos 2 e 4 do G (`fidelity_perc/nnls_vs_llm`, `accuracy_perc/nnls_vs_true`, `llm_accuracy_vs_true`)
- `bloco23_fg_aprendiz_{timestamp}.csv`: Bloco 2/3: Passo 3 do F e Passo 6 do G (com o Perceptron do Passo 4 do F e do Passo 7 do G). `accuracy_llm_vs_expert`/`kappa_llm_vs_expert`/`f1_llm_vs_expert`/`n_errors_llm_vs_expert` ← `*_llm_vs_metric`; novas `especialista_origem` (`rotulo_real` no F, `metrica` em G1/G2) e `example_indices` (sorteio balanceado k/2 por classe pelo RÓTULO REAL do treino, `seed + rep`; idênticos entre variantes de atributo); desde 13/10/2026 `especialista_regime`, `especialista_k`, `especialista_rep` (regime/k/rep da métrica eleita como especialista em G1/G2; NaN no F, cujo especialista é o rótulo real); desde 14/10/2026 `n_features` = nº de atributos MOSTRADOS ao LLM (2, 3 ou 4; fator `ATRIBUTOS_PROMPT_PASSO6_G`; até 13/10 era a dimensão da eleita) e `especialista_n_features` = dimensão da métrica eleita (NaN no F); as linhas k = 0 das três dimensões são a MESMA coleta
- `bloco3_g_passos_2_a_5_{timestamp}.csv`: Bloco 3 (13/10/2026): Passos 2 a 5 do G nos 70 pontos de treino, mesmas colunas de `bloco1_ad_passos_2_a_5` + `problem_name`, `subproblem` (G1/G2), `class_names`, `feature_names`; uma linha por (modelo, seed, subproblema, classes, atributos, regime, k, rep, n_features, algoritmo). Chave `g_passos_2_a_5` em `gerar_tabelas.carregar_tudo`
- `final_cross_linearity__<alias>.csv`: Síntese cruzada R2→R4 nos 3 blocos (UM por modelo). Novas colunas `problem` (estável: `A_linear`, `D_meialua`, `F_meia_lua`, `G1_homem_mulher`, `G1_homem_mulher_classesAB`, …), `problem_name` (completo) e `feature_names`; a meia-lua vem do Problema D (zero-shot do Passo 2; o F só entra se o D não rodou); `accuracy_vs_true` preenchida nas linhas `A_linear`
- `*_corrigido_hm[__<alias>].csv`: só na execução de 07/2026 (cópias corrigidas offline; a ferramenta que as gerou foi removida em 13/10/2026)

### Outros Artefatos
- `log_execucao.txt` — Transcript completo com detalhes algorítmicos e métricas; desde 15/10/2026 cada cabeçalho de seção (`print_section`, linhas `>>> ` e `[ETAPA`) traz `⏱ total (+desde a seção anterior)` (`execucao_io.iniciar_cronometro`/`decorrido`), para medir API × estimação × figuras × tabelas (particionado em `log_execucao_parte{NNN}.txt` se passar de 10 MiB)
- `llm_interactions_parte{NNN}.json` — Log estruturado de **todas** as chamadas à API
  (prompt, `point`, `raw_response`, `parsed_label`, `model`, `model_resolved` — snapshot
  datado devolvido pela API, p/ reprodutibilidade —, `inference_provider` — provedor de
  inferência no OpenRouter —, `temperature`, `format_retries`,
  `malformed`), **particionado** em blocos de 10 MiB (`_parte001.json`, `_parte002.json`, …)
  para evitar um único arquivo gigante. Execuções pequenas (smoke, `--apenas`) e execuções
  antigas gravam o arquivo único `llm_interactions.json` — o auditor aceita ambos os esquemas.
- `dados_sinteticos_seed{seed}/problema_{A,B,C,D,E,A_r3,A_r4}.csv`: Datasets por seed (reprodutibilidade; `problema_E.csv` = especialista linear)

### Auditoria Offline das Interações (`src/audit_interactions.py`)
Verifica, **sem chamar a API**, quatro propriedades que sustentam a credibilidade de
κ/consistência, lendo os `llm_interactions_parte*.json` da execução:
1. **Taxa de fallback do parser** (`malformed=True`) — se alta, os rótulos são ruído do
   hash MD5 e as métricas ficam comprometidas.
2. **Determinismo de `T=0`** — mede a fração de pares (prompt, ponto) idênticos cuja
   RESPOSTA INTERPRETADA (`parsed_label`) divergiu (`taxa_flip_T0`); a divergência do texto bruto
   fica em `taxa_flip_raw_T0` ("A" vs "A." não é flip de rótulo). `T=0` **não** é determinístico
   (completa de 13/10/2026: 1,4% no gpt-4o-mini, 0% no Gemini; na de 08/10/2026 1,6%; na de 07/2026 4,3%);
   reportar esse número é honestidade experimental.
3. **Breakdown por (provider, modelo)** — fallback e flip por modelo, mais os snapshots
   resolvidos (`model_resolved`) e provedores de inferência observados (OpenRouter).
   A comparação de flip rates entre modelos é um resultado publicável por si só.
4. **Breakdown por regime** (`por_regime`): a partir do campo `tag` gravado em cada
   registro desde 01/10/2026, no formato de 13/10/2026
   `<bloco>|problema=<P>|passo=<N>|regime=<regime>|k=<k>|rep=<r>|seed=<s>[|extra]` (7 campos fixos):
   malformadas e flip por (problema, passo, regime, k), chave `problema|passo|regime|k`. O campo `passo=`
   é o passo do problema (`passo=2` = zero-shot; `passo=3` = exemplos do rótulo real em A/D/G, estratégias em E,
   aprendiz no F; `passo=4|regime=exemplos_da_metrica` = B/C com exemplos da métrica; `passo=6` = aprendiz do G;
   vieses = `passo=2` (k = 0, A e B/C zero-shot) ou `passo=4` (k > 0, B/C com exemplos da métrica) com `regime=vies_<fator>`). O zero-shot de A, B e C da configuração
   base é UMA coleta com `problema=ABC` (150 + 100 + 100 chamadas), não separável por problema na auditoria. Nos Problemas G e F a tag
   leva, depois dos 7 campos fixos, `nomes=` e `atributos=`
   (`bloco3|problema=G1_homem_mulher|passo=3|regime=…|k=4|rep=0|seed=42|nomes=Mulher/Homem|atributos=peso/altura`);
   no Passo 6 do G e no Passo 3 do F entra ainda `n_atributos_prompt=<2|3|4>` depois de `atributos=` (nº de atributos
   mostrados ao LLM, 14/10/2026; no F é sempre 2).
   `por_regime` aceita só o formato novo; execuções anteriores a 13/10/2026 (2º campo `fase=`) passam antes por
   `renomear_assets_legado.py --aplicar`. O flip só é mensurável onde o mesmo
   prompt+ponto se repete no grupo (G: pontos de treino comuns entre sementes; D e F: o mesmo dado; diluição 4+0 = difícil k = 4; ordenação classe 0 primeiro = mista).

```bash
python src/audit_interactions.py                 # audita a execução COMPLETA mais recente
python src/audit_interactions.py resultados/execucao_2026-07-05_12-34-52_completa
python src/audit_interactions.py --json          # saída estruturada (CI/log)
```
Sai com código 1 se a taxa de malformadas ultrapassar `--max-malformed` (5% padrão).
O teste `tests/test_audit_interactions.py` roda esta auditoria na execução mais recente.

## Leitura de Resultados

**IMPORTANTE:** Antes de responder qualquer pergunta sobre resultados ou análise, leia sempre
os arquivos da **última execução COMPLETA disponível** — somente pastas com sufixo
`_completa`; pastas `_smoke` e `_apenas-*` NUNCA servem de fonte.

A última execução completa está em: `resultados/execucao_2026-10-15_20-15-20_completa/` (noite de 15/10/2026, Perceptron corrigido
B/C/D com a coluna `margem_minima`; 222.648 chamadas, 5 malformadas (os mesmos 5 pontos da variante cot do Gemini, agora com 2 tentativas de
90 s), flip 1,19 % gpt / 0 % Gemini; 2h21 com MAX_CONCURRENCY=50; exit 0; auditada em 15/10/2026: 0 linhas degeneradas do Perceptron em A/D/G;
H1 NÃO sustentada em B nos dois modelos (κ gpt 0,320, Gemini 0,258) e sustentada em C (0,725; 0,852); H2 exata (κ em B com k ≥ 10 ≈ 0,85–0,89);
H4 refutada; oráculo 100 %; zero-shot do G colapsa em 7 das 16 configurações (modelo × subproblema × classes × atributos), em todas as sementes,
por isso `bloco23_fg_passos_2_e_4` e `final_cross_linearity` não têm essas linhas (rodapé da final_table_01)). A anterior,
`execucao_2026-10-15_10-40-22_completa/` (Perceptron antigo, 13 linhas degeneradas em A/D), tem a REESTIMAÇÃO
com o Perceptron corrigido (15/10/2026 à noite, `--reestimar-de`, 0 chamadas) está em
`resultados/execucao_2026-10-15_19-32-06_completa_reestimada/`: use-a para A/D (Passos 2 a 5), B/C Passo 3 (H1), E, F e G Passos 2 a 5;
NÃO use dela o Passo 4 de B/C (H2), os vieses com k = 10 nem o Passo 6 do G, que são aproximações (16.700 pontos em 306 coletas,
`reestimacao_lacunas.csv`); `--coletar-lacunas` (≈ US$ 1,1) ou uma nova completa os tornam exatos. (A completa de 10-40-22, código de 15/10/2026: Problema A com centróides (−2; −0,5) e (2; 0,5), coletas em lote, figuras em processos, ⏱ no log; 222.648 chamadas, 5 malformadas, todas do Gemini nas variantes de prompt (a auditoria só distingue o regime `vies_variantes_prompt`: 4 em k = 0 e 1 em k = 10; o log mostra que foram timeouts de 90 s na variante cot, após 5 tentativas), flip 1,46 % gpt (43 de 2.936 perguntas repetidas) / 0 % Gemini; 2h57 com MAX_CONCURRENCY=50, dos quais ≈ 61 min perdidos nesses 5 pontos da variante cot do Gemini; exit 0; auditada em 15/10/2026). As duas anteriores, `execucao_2026-10-14_16-43-17_completa/` (A com x2 = 0, 221.508 chamadas) e `execucao_2026-10-13_22-27-34_completa/` (código final de 13/10/2026 com a
nomenclatura por passos; 187.328 chamadas, 6 malformadas, flip a T=0 de 1,4% no gpt-4o-mini e 0% no Gemini; 3h39 com
MAX_CONCURRENCY=40; exit 0; figuras 04/05/07/10/15–18/21, bloco2_01/03–07, bloco23_01/02/04, bloco3_03 e final_08 e todas as
tabelas foram REGENERADAS offline em 14/10/2026 a partir dos CSVs após a auditoria de resultados, ver
`revisao_codigo_2026-10-09.md`; as figuras 09, final_06 e final_07 só refletem as correções de texto na próxima execução).
A anterior, `execucao_2026-10-08_23-16-54_completa/` (código corrigido de 08/10/2026,
nomenclatura E1/E2/E3, F, G1/G2; a de julho, `execucao_2026-07-05_12-34-52_completa`, fica só para comparação histórica)

O padrão de nome das pastas é `resultados/execucao_YYYY-MM-DD_HH-MM-SS_{completa|smoke|apenas-<blocos>}/`.
Execuções anteriores a 13/10/2026 (inclusive a completa de 08/10/2026) estão com os nomes de arquivo e de
coluna ANTIGOS: antes de ler CSVs com as ferramentas offline (`gerar_tabelas.py`, `audit_interactions.py`) ou de
citar nomes de asset no trabalho, rode `python src/renomear_assets_legado.py <pasta> --aplicar` (mapa
`renomeacao_passos_2026-10-13.md`); a leitura manual dos CSVs antigos deve traduzir os nomes pela tabela
"Equivalência de identificadores".
Se houver pastas `_completa` mais recentes, leia a de timestamp mais alto. (Pastas legadas
sem sufixo: confirme no `log_execucao.txt` que não são `--rapido`.)

Arquivos prioritários para leitura:
1. `log_execucao.txt`: visão completa dos resultados; o veredito de H1 por modelo está no sumário estatístico,
   bloco "H1 (CONSISTÊNCIA): VEREDITO POR MODELO" (κ médio em B e C, faixa de Landis e Koch, seeds com κ > 0,4
   e a linha "H1 (κ > 0.4): SUSTENTADA/NÃO SUSTENTADA em B | ... em C"); o de H4 na análise do Problema E (Passo 3)
2. `bloco1_abc_passos_2_a_4_{timestamp}.csv` — métricas dos Problemas A, B e C (Bloco 1)
3. `bloco2_e_passos_2_3_{timestamp}.csv` — métricas do Problema E, Passos 2 e 3 (Bloco 2; coluna `expert_name`)
4. `bloco1_oraculo_passos_1_a_3_{timestamp}.csv` / `bloco1_d_passo5_teto_*.csv` — validação dos algoritmos
5. `bloco1_a_passo4_perceptron_vs_nnls_{timestamp}.csv`: comparação Perceptron × NNLS (só as linhas NNLS; o par Perceptron está em `bloco1_abc_passos_2_a_4`)
6. `bloco1_a_passo4_r2_r3_r4_{timestamp}.csv` — R2/R3/R4 no Problema A
7. `bloco2_e_passo4_diluicao_*.csv`, `bloco2_e_passo5_ordem_*.csv`, `bloco2_e_passo6_baselines_*.csv` — auxiliares do Bloco 2
8. `bloco23_fg_passos_2_e_4_*.csv`, `bloco23_fg_aprendiz_*.csv` — Problemas F e G (meia-lua, peso×altura);
   para o Bloco 3 da execução 2026-07-05 (`problem_name == homem_mulher`, nome antigo de G1) use as cópias `*_corrigido_hm*.csv` (a ferramenta que as gerou foi removida em 13/10/2026);
   `bloco3_g_passos_2_a_5_*.csv` (só em execuções a partir de 13/10/2026): Passos 2 a 5 do G (zero-shot, few-shot do rótulo real, estimação e teto)
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
scikit-learn    # Cohen's κ, F1-Macro, k-NN, LR, SVM
scipy           # nnls (NNLS), stats (Kruskal-Wallis, Mann-Whitney, ponto-bisserial nos relatórios)
seaborn         # heatmaps das matrizes de confusão (final_graphic_02)
httpx           # consulta de saldo do OpenRouter (llm_client.consultar_saldo_openrouter; já vem com openai)
```

## Notas de Design

- **Métrica diagonal:** Simplificação proposital para tratabilidade (O(d) parâmetros vs O(d²))
- **Dados 2D sintéticos:** Permitem visualização; a dimensionalidade extra vem das projeções R3/R4 e o caso real é o peso × altura (Problema G)
- **Dois algoritmos de otimização inversa:** Perceptron (com hiperparâmetros) + NNLS (sem) — garante robustez do método de estimação. LP Max-Margin foi removido (ver `reunioes_orientador/reuniao_2026_04_30/plano_trabalho.md` item 12).
- **Validação do oráculo:** Dá um piso de sanidade: se os algoritmos falharem em recuperar W sintético conhecido, qualquer achado sobre o LLM fica comprometido. Só os Problemas 1/2/3 do orientador (sobre E3, E1 e E2, as três métricas sobre as gaussianas do E); as gaussianas extras foram removidas em 13/10/2026 porque nelas o W não era identificável (qualquer W separa os dados), ver a nota na seção do oráculo
- **Concorrência assíncrona (MAX_CONCURRENCY=50, desde 14/10/2026):** Reduz tempo de coleta do LLM via `asyncio.Semaphore` (benchmark 08/10/2026 sem 429 até 50). O cliente `AsyncOpenAI` é criado por coleta, dentro do laço de eventos, e fechado no `finally` com o laço aberto (global `async_client = None` em produção, só injeção nos testes); o 402 é tratado fora do laço, depois do fechamento, sem "Event loop is closed" no log (09/10/2026)
- **Fallback por hash MD5:** Respostas malformadas após retries são mapeadas via `hashlib.md5` de coordenadas (substituiu aritmética modular para não introduzir viés geométrico)
- **Temperatura = 0.0:** Minimiza estocasticidade para isolar o critério de decisão (não garante determinismo — ver Reunião 1)
- **Nomes de classe variados:** `["A"/"B", "0"/"1", "Positivo"/"Negativo", "Azul"/"Vermelho"]` — detecta viés semântico
- **Nomes de classe invertidos:** `["B"/"A", "1"/"0", ...]` — detecta viés de posição/ordem
- **Nomes de features semânticos:** `["altura"/"peso", "feature_1"/"feature_2"]` — detecta viés semântico nas variáveis
- **Parser de 8 camadas (0–7):** Estratégia robusta para lidar com respostas malformadas do LLM
- **3 sementes aleatórias (× 3 repetições onde há sorteio de exemplos, `reps_para`):** Garante robustez estatística
- **Análise estatística:** média e faixa mín–máx entre sementes (× repetições), diferenças pareadas por semente (consistente = todas as sementes no mesmo sentido), Cohen's d. O intervalo de confiança por bootstrap foi removido em 13/10/2026 a pedido do orientador e o teste de Wilcoxon por decisão do autor na mesma data (com 3 sementes não acrescentava informação além da faixa observada).
- **Projeções R3/R4:** x3=x1·x2 (R3) e x1², x2² (R4) simulam kernel quadrático sem sair do mundo linear
- **Painéis individuais:** cada figura combinada também é salva como PNGs separados (`<base>_<painel>__<alias>.png`), úteis para apresentações
- **Dados por seed salvos em CSV:** reprodutibilidade bit-a-bit independente do código
- **Terminologia:** Usar "Ŵ_LLM estimada/inferida" (não "W aprendida") para a métrica estimada no Passo 4 (A, D, G)
- **Correções de 08/10/2026 (ver `revisao_codigo_2026-10-08.md`):** exemplos few-shot mostram os atributos
  extras (x1·x2, x1², x2²) com os mesmos nomes do ponto de teste; a retentativa de formato reenvia o MESMO
  estímulo; `raw_response` é a resposta de onde saiu o rótulo (`all_responses` guarda as anteriores);
  nenhum limite de `max_tokens` é imposto (limitar truncava o raciocínio do Gemini antes do rótulo;
  o parser extrai o rótulo de respostas longas); no Passo 3 do F e no Passo 6 do G os exemplos são sorteados com
  `RandomState(seed + rep)` (desde a noite de 09/10/2026 balanceados pelo RÓTULO REAL do treino, logo iguais
  entre variantes de atributo em F e em G1/G2; ver "Fechamento de 09/10/2026" abaixo) e o
  zero-shot externo vê só x1, x2;
  cada interação leva `tag` por bloco/problema/passo/regime/k/rep/seed (Bloco 2 e 3 incluídos; campo `passo=` desde 13/10/2026); colapso
  no Passo 2 (zero-shot do A) vira `collapsed=1` + NaN; parser casa nomes de 1 letra case-sensitive (o artigo inglês "a"
  não é classe A); HTTP 402 aborta (exit 3); exceção num bloco (G1, G2, F) é registrada e a execução
  termina com exit 2 ("EXPERIMENTO INCOMPLETO"); figuras e relatório do Bloco 1 agregam só a
  configuração base (prompt default, x1/x2); o veredito de H4 no log é sobre o especialista E1
  (`bloco2_table_05` cobre E1/E2/E3).
- **Correções de 09/10/2026 (ver `revisao_codigo_2026-10-09.md`, seção "Status"):** métricas de avaliação
  restritas às cinco da seção IV.3 (Fidelidade, Acurácia, Consistência, Cohen's κ, F1-Macro) com nomes
  únicos; especialista do Problema F = rótulo real (`especialista_origem`); `bloco_protegido` em todos os blocos
  (exit 2); comparação de algoritmos pareada (0 chamadas); oráculo rotulado pelo centróide mais próximo;
  zero-shot do Problema E (Passo 2) coletado 1×; mixed 50/50; sorteio balanceado no Passo 3 do F e no Passo 6 do G (pelo rótulo real desde a noite do mesmo dia); quase-colapso
  (`MIN_PONTOS_CLASSE_MINORITARIA`); `bloco1_graphic_20` removido; barras de erro imunes ao
  `garantir_legendas`; `MODELS_TO_TEST` sem `scope`; `tests/test_regressao_execucao.py` apontando para
  `execucao_2026-07-05_12-34-52_completa`. O aprendizado em contexto de F e G (Passo 3 do F, Passo 6 do G) de julho (07/2026) com `n_features ≥ 3`
  continua NÃO confiável (prompt sem os atributos extras; corrigido em 08/10/2026).
- **Rodada 2 de 09/10/2026 (ver `revisao_codigo_2026-10-09.md`, subseção "Rodada 2"):** `_erro_retentavel` aceita
  "timed out" do SDK; a reinterpretação offline das 157.854 respostas da completa de 08/10/2026 com o parser
  atual muda 7 rótulos (4 trocas de rótulo, 2 respostas contraditórias que agora são malformadas, 1 antes
  malformada agora aceita), todas respostas chain-of-thought do Gemini; `bloco1_a_passo4_r2_r3_r4` grava
  `w0, w1` nas linhas R2; tabelas 14/16 com o k de `K_VIESES`; `renomear_assets_legado` silencioso em
  execuções já na convenção; `estilo_cinza` com chave cor + marcador + traço (ver "Legenda garantida").
- **Fechamento de 09/10/2026, noite (ver `revisao_codigo_2026-10-09.md`, subseção "Fechamento das pendências"):**
  (1) Oráculo só sobre B/E1/E2 (Problemas 1/2/3 do PDF; desde 14/10/2026 o euclidiano é o E3 sobre as gaussianas do E); em 13/10/2026 as gaussianas extras, pouco
  identificáveis, foram removidas do código (configs, gerador e flag). (2) Passo 3 do F e Passo 6 do G: `fg_exemplos_aprendiz(y_train,
  k, seed + rep)` balanceia pelo RÓTULO REAL do treino, logo os índices são idênticos entre variantes de atributo
  também em G1/G2 (a limitação anterior, índices diferentes no G, deixou de existir); os exemplos continuam
  rotulados pelo especialista, e no G o balanço pelo especialista pode não ser exato (contagem no log).
  (3) Veredito de H1 no log (`texto_h1`, `LIMIAR_H1_KAPPA = 0.4`). (4) `malformadas_e` conta o zero-shot
  do Problema E uma vez por (modelo, seed, rep). (5) Cliente da API criado e fechado dentro do laço de eventos
  (sem "Event loop is closed"). (6) `GRADES_MODO_RAPIDO`, `grades_modo_rapido`, `aplicar_modo_rapido` e
  `nome_pasta_execucao` extraídas de `main()`, com `tests/test_infra_main.py` (suíte: 332 testes na época; 361 em 13/10/2026 à noite).
  (7) Itens 1 e 2 decididos pelo autor em 13/10/2026 (remover as extras; priorizar os mesmos exemplos entre
  variantes de nome, com o balanço mais próximo possível); a próxima execução completa pode rodar.
- **Correções e decisões de 13/10/2026 (ver `revisao_codigo_2026-10-09.md`, "Decisões de 13/10"):**
  (1) Passos 2 a 5 no Problema G (pedido do orientador, Metodologia IV.7: "o LLM classifica o mesmo conjunto
  de treino em zero-shot e few-shot, n ∈ {4, 10, 20, 40}, e as métricas são avaliadas seguindo os mesmos passos do
  problema A"): `fg_executar(..., com_passos_3_a_5=True)` em G1 e G2 (False no F) roda
  `adg_passos_2_a_5` sobre os 70 pontos de treino; `adg_passos_2_a_5` ganhou `X_display`, `nome_feature_0/1`,
  `bloco`, `tag_extra` (A e D inalterados); novo CSV `bloco3_g_passos_2_a_5`, tabela `bloco3_table_06`, figura
  `bloco3_graphic_03` (+ `_nnls`); especialista do Passo 6 eleito (Passo 5) por `g_passo5_eleger_especialista` entre zero-shot
  e few-shot (colunas `especialista_regime`/`especialista_k`/`especialista_rep`); tags externas com `nomes=` e `atributos=`.
  Chamadas extras na completa ≈ 29.664 (8 combinações subproblema × classes × atributos × (66 + 60 + 50 + 30) × 3 reps × 3 seeds × 2 modelos; ≈ 187 mil
  no total) e +1.056 no `--rapido`; `CUSTO_ESTIMADO_USD` completa 11 → 14,5 e bloco3 2 → 5,5. Suíte: 361 testes em 13/10/2026 (365 em 15/10/2026)
  (`tests/test_g_passos_2_a_5.py`, 14 testes). (2) Few-shot de B/C fica como está (grade {0, 4, 10, 20, 40},
  exemplos rotulados pela métrica zero-shot do A); o orientador ajusta o texto. (3) R³/R⁴ em B e C não se aplica
  (problemas lineares): só em A, D e nos Problemas F e G.
- **Nomenclatura por passos (13/10/2026, camada 1):** os textos visíveis (títulos, legendas, rodapés, log, documentação)
  usam "Problema X, Passo N" conforme a seção "Passos por problema".
- **Renomeação completa de 13/10/2026 (camada 2):** todos os identificadores internos (funções, constantes, flags,
  campos de dataclass, colunas e valores de CSV, tags de auditoria com `passo=`, nomes de CSV, gráficos, painéis,
  slugs de tabela, testes e ferramentas) seguem o mapa único `renomeacao_passos_2026-10-13.md` (resumo na tabela
  "Equivalência de identificadores" da seção "Passos por problema"). Nenhum nome novo contém `fase`/`phase`/`external`/
  `pipeline`/`protocol`/`oracle`. Consequências: `gerar_tabelas.carregar_tudo` e `audit_interactions.por_regime` aceitam
  SÓ os nomes novos; execuções anteriores a 13/10/2026 (inclusive as desse dia) precisam de
  `python src/renomear_assets_legado.py <pasta> --aplicar` antes de qualquer ferramenta offline (o migrador renomeia
  arquivos e colunas/valores de CSV e converte as tags `fase=` em `passo=`). O registro histórico
  (`revisao_codigo_2026-10-08.md`, `revisao_codigo_2026-10-09.md`) mantém os nomes da época.
- **Revisão de 15/10/2026 (ver `revisao_codigo_2026-10-15.md`), item 1 APLICADO à noite (seção 5 do relatório; ver a linha do Perceptron em "Algoritmos de Otimização Inversa"): o Perceptron
  aceitava soluções degeneradas com γ > 0 e ‖W‖ < 0,05 (o teste de margem usa a margem relaxada e o fator 1 − ηγ/‖w‖ encolhe
  w a cada correção; com ‖w‖ ≈ 10⁻³ um único αᵢ satisfaz qualquer ponto e a época termina com "0 violações"). Na completa de
  15/10/2026, 12 das 495 linhas do Perceptron em A/D (`bloco1_ad_passos_2_a_5`) são degeneradas, 8 no zero-shot e 4 no few-shot,
  nenhuma no G; a direção "só x2" do gpt-4o-mini no A é artefato dessas linhas e o κ zero-shot em B do gpt usa esse W; nelas o
  marcador † (γ = 0) inverte o sinal de alerta, pois as linhas com γ = 0 têm ‖W‖ ≈ 1 e são as saudáveis. Correção adiada
  (declarar viabilidade pela margem bruta e/ou renormalizar w; reestimação offline com 0 chamadas). Também pendente: timeouts
  de 90 s ainda seguem o laço comum de `MAX_RETRIES = 5` com backoff (a revisão sugere no máximo 2 tentativas com espera curta
  só para timeout, ou `API_TIMEOUT_S` maior para a variante cot).
