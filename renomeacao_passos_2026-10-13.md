# Renomeação completa para a nomenclatura por passos (camada 2, 13/10/2026)

Fonte da nomenclatura: `passos_por_problema.md`. Esta tabela é o mapa ÚNICO de renomeação
de identificadores, colunas, valores, tags, nomes de arquivo e slugs. Regras gerais:

1. Nenhum identificador, coluna, valor, tag, slug ou nome de arquivo novo contém `fase`,
   `phase`, `external`, `pipeline`, `protocol`, `protocolo`, `fewshot`/`few_shot` como nome
   de etapa, `transfer`/`transferred` (a transferência é o Passo 3 de B/C), `oracle`
   (usar `oraculo`), nem `meialua` como nome de etapa (o problema é D ou F).
2. O nome diz PROBLEMA e PASSO quando é específico: prefixo `a_`, `bc_`, `d_`, `e_`, `f_`,
   `g_`, `fg_` (F e G), `ad_` (A e D), `adg_` (A, D e G), `oraculo_`; sufixo `_passoN` ou
   `_passos_N_a_M`. Quando é compartilhado por vários problemas, nome pelo significado
   (ex.: `K_EXEMPLOS_ROTULO_REAL`).
3. Siglas inglesas que não carregam etapa (`accuracy`, `fidelity`, `consistency`, `kappa`,
   `f1`, `easy/hard/mixed/random`, `w_perc`, `n_features`) FICAM.
4. Tag da auditoria: `fase=` vira `passo=N`; o formato passa a ser
   `<bloco>|problema=<P>|passo=<N>|regime=<regime>|k=<k>|rep=<r>|seed=<s>[|extra]`
   (7 campos fixos; `audit_interactions.por_regime` precisa aceitar só o formato novo e
   os logs antigos com `fase=` continuam legíveis pela ferramenta de migração).
5. Execuções anteriores a 13/10/2026 (inclusive as de hoje): `renomear_assets_legado.py`
   ganha o conjunto de regras desta tabela (arquivos e colunas de CSV) para migrar uma
   pasta inteira; `gerar_tabelas.carregar_tudo` NÃO tem fallback para nomes antigos:
   migra-se a pasta antes.

## Constantes e flags (`dissertacao_mestrado.py`, `protocolo.py`)

| Antigo | Novo | Onde se aplica |
|---|---|---|
| `FEW_SHOT_SIZES` | remover (era só alias de `FEW_SHOT_SIZES_BC`) | |
| `FEW_SHOT_SIZES_FASE_A` | `K_EXEMPLOS_ROTULO_REAL` | A Passo 3, D Passo 3, G Passo 3 |
| `FEW_SHOT_SIZES_BC` | `K_EXEMPLOS_METRICA_BC` | B/C Passo 4 |
| `FEW_SHOT_SIZES_PHASE_E` | `K_EXEMPLOS_APRENDIZ` | E Passo 3, F Passo 3, G Passo 6 |
| `BIAS_N_SHOTS` | `K_VIESES` | vieses do A Passo 2 e B/C Passo 3 |
| `PROTOCOLO_N_FEATURES` | `ATRIBUTOS_PASSO4` (`{"A": (2,3,4), "D": (2,3,4), "G": (2,3,4)}`) | Passo 4 |
| `N_SAMPLES_PROBLEM_A/B/C/E` | `N_PONTOS_A/B/C/E` | Passo 1 |
| `RUN_PHASES_ABC` | `RUN_PROBLEMAS_ABC` | |
| `RUN_PHASE_E` | `RUN_PROBLEMA_E` | |
| `RUN_MULTIPLE_EXPERTS` | `RUN_PROBLEMA_E2_E3` | |
| `RUN_CLASSICAL_BASELINES` | `RUN_E_PASSO6_BASELINES` | |
| `RUN_DILUTION` | `RUN_E_PASSO4_DILUICAO` | |
| `RUN_EXAMPLE_ORDER_BIAS` | `RUN_E_PASSO5_ORDEM` | |
| `RUN_R3R4_EXPERIMENT` | `RUN_A_PASSO4_R3R4` | |
| `RUN_ALGORITHM_COMPARISON` | `RUN_A_PASSO4_PERCEPTRON_VS_NNLS` | |
| `RUN_CLASS_ORDER_BIAS` | `RUN_VIES_ORDEM_CLASSES` | |
| `RUN_FEATURE_NAMES` | `RUN_VIES_NOMES_ATRIBUTO` | |
| `RUN_PROMPT_VARIANTS` | `RUN_VIES_VARIANTES_PROMPT` | |
| `RUN_PROBLEM_D_FASE_A` | `RUN_PROBLEMA_D` | |
| `RUN_PROBLEM_MEIALUA` | `RUN_PROBLEMA_F` | |
| `RUN_HOMEM_MULHER` | `RUN_PROBLEMA_G` | |
| `RUN_HM_CLASS_NAMES_AB` | `RUN_PROBLEMA_G_CLASSES_AB` | |
| `RUN_ORACLE_VALIDATION` | `RUN_ORACULO` | |
| `RUN_ORACLE_MEIALUA` | `RUN_D_PASSO5_TETO` | |
| `HM_CLASS_NAME_VARIANTS`, `HM_G1_FEATURE_VARIANTS`, `HM_G2_FEATURE_VARIANTS` | `G_NOMES_CLASSE`, `G1_NOMES_ATRIBUTO`, `G2_NOMES_ATRIBUTO` | |
| `PHASE_A_PROTOCOL_CACHE` | `CACHE_ROTULOS_LLM` | |
| `EXAMPLE_STRATEGIES`, `EXPERT_W`, `EXPERT_CENTROIDS`, `EXPERT_CONFIGS`, `DILUTION_EASY_ADDITIONS`, `K_ORDEM_EXEMPLOS`, `EXAMPLE_ORDERINGS`, `NOMES_CLASSES*`, `NOMES_FEATURES`, `PROMPT_VARIANTS`, `PERCEPTRON_PARAMS`, `MODELS_TO_TEST`, `RANDOM_SEEDS`, `N_REPETICOES`, `MAX_*`, `CUSTO_*` | ficam | |
| `BLOCOS_APENAS` chaves `bloco1/bloco2/bloco3/oraculo` | ficam (CLI `--apenas`) | |
| CLI `--bc-zero-shot` | fica (força `K_EXEMPLOS_METRICA_BC = [0]`) | |
| `src/protocolo.py` (constantes compartilhadas) | fica | |

## Funções (`dissertacao_mestrado.py`)

| Antigo | Novo |
|---|---|
| `phase_a_learn_metric` | `a_passo2_zero_shot_e_metrica` |
| `phase_a_multifeature` | `passo4_estimar_metrica_r2_r3_r4` |
| `phase_a_protocol` | `adg_passos_2_a_5` |
| `_protocol_rows` | `_linhas_passos_2_a_5` |
| `_balanced_gt_examples` | `passo3_exemplos_rotulo_real` |
| `phase_consistency_test` | `bc_passos_2_a_4` |
| `select_confident_examples` | `bc_passo4_exemplos_da_metrica` |
| `evaluate_transfer_bc` | `bc_passo3_consistencia` |
| `marcar_melhor_metrica_a` | `a_passo6_eleger_metrica` |
| `run_complete_experiment` | `abc_executar` |
| `_run_phase_abc_experiment` | `_abc_executar_configuracao` |
| `_plot_protocol_symbols` | `_plot_simbolos_passos` |
| `_llm_fewshot_scheme` | `_esquema_passo3` |
| `phase_e_llm_as_learner` | `e_passos_2_3_aprendiz` |
| `select_examples_by_strategy` | `e_passo3_selecionar_exemplos` |
| `select_examples_dilution` | `e_passo4_exemplos_diluicao` |
| `reorder_examples` | `e_passo5_reordenar_exemplos` |
| `expert_classify` | `especialista_rotular` |
| `run_external_problem_pipeline` | `fg_executar` (F Passos 1 a 4; G Passos 1 a 7); parâmetro `protocolo_fase_a` → `com_passos_3_a_5` |
| `_balanced_example_indices` | `fg_exemplos_aprendiz` |
| `_eleger_especialista_externo` | `g_passo5_eleger_especialista` |
| `run_oracle_validation` | `oraculo_passos_1_a_3` |
| `run_oracle_meialua` | `d_passo5_teto` |
| `_oracle_algorithms`, `_append_oracle_result` | `_oraculo_estimadores`, `_oraculo_registrar` |
| `nome_problema_f` | fica |
| `_make_tag(fase, problema, regime, k, rep, seed, bloco, extra)` | `_tag(bloco, problema, passo, regime, k, rep, seed, extra)` |
| `build_prompt_zero_shot`, `build_prompt_few_shot`, `*_variant` | `prompt_zero_shot`, `prompt_com_exemplos`, `*_variante` |
| `colapsou`, `reps_para`, `bloco_protegido`, `aplicar_modo_rapido`, `nome_pasta_execucao`, `grades_modo_rapido` | ficam |
| variáveis do `main()`: `all_results_abc` → `resultados_abc`; `all_results_e` → `resultados_e`; `all_phase_a_protocol` / `protocol_rows` → `linhas_passos_2_a_5`; `external_phase_a_results` → `resultados_fg_passos_2_e_4`; `external_phase_e_results` → `resultados_fg_aprendiz`; `external_phase_a_protocol_rows` → `linhas_g_passos_2_a_5`; `all_results_oracle` → `resultados_oraculo`; `all_results_oracle_meialua` → `resultados_d_passo5`; `phase_a_data_for_plots`/`phase_a_cache` → `dados_a_passo2`/`cache_a_passo2`; `hm_pipeline`/`ml_pipeline` → `saida_g`/`saida_f`; chaves de dict `'phase_a_results'`/`'phase_e_results'`/`'phase_a_protocol_rows'` → `'passo2'`/`'aprendiz'`/`'passos_2_a_5'` | |

## Outros módulos

| Antigo | Novo |
|---|---|
| `resultados.ResultadoExperimento` | `ResultadoABC` |
| `resultados.ResultadoPhaseEExperimento` | `ResultadoAprendiz` |
| campos `consistencia_transferida_problema_b/c`, `kappa_transferido_problema_b/c` | `consistencia_centroides_a_problema_b/c`, `kappa_centroides_a_problema_b/c` (ablação com centróides do A) |
| campo `n_shot` em `ResultadoABC` | `k_passo4_bc` |
| campo `n_shot` em `ResultadoAprendiz` (E; 13/10/2026, noite) | `k` (a coluna de CSV já era `k`; os dicts internos de plots/relatorios e o eixo de `final_graphic_06` seguem) |
| rótulos LaTeX `tab:T1_A_…`, `tab:T2_BC_…`, `tab:T3_D_…`, `tab:T4_W_…`, `tab:T5_flip`, `tab:T6_zero_few_A_…`, `tab:T7_escolha_…`, `tab:T8_transf_…`, `tab:T9_comp_…` | `tab:<slug>_<alias>` (`tab:A_passos_2_a_5_…`, `tab:BC_passo3_todas_metricas_A_…`, `tab:D_passos_2_a_5_…`, `tab:AD_passo4_estabilidade_W_…`, `tab:final_auditoria_flip_por_passo`, `tab:A_passo2_vs_passo3_…`, `tab:A_passo6_eleicao_…`, `tab:BC_passo3_por_semente_…`, `tab:BC_passo3_zero_vs_eleita_…`) |
| `colapsou` (runner) | continua `colapsou`, mas vive em `protocolo.py` (13/10/2026, noite) e é usada também por `plots.py` (`bloco1_graphic_14`, centróides locais de B/C) |
| classes/métodos de teste `TestPipelineG2MostraKgM`, `TestMalformadasFaseE`, `TestFaseE`, `TestPipelineExternoEspecialista`, `TestPromptFewShotAtributosExtras`, `test_run_complete_experiment_*`, `test_plot_oracle_transfer_removido`, `test_sem_plot_oracle_transfer_no_runner`, `_COLS_EXT_E_ANTES` | `TestG2MostraKgM`, `TestMalformadasE`, `TestEPassos2e3`, `TestFGEspecialista`, `TestPromptComExemplosAtributosExtras`, `test_abc_executar_*`, `test_grafico_20_removido`, `test_sem_grafico_20_no_runner`, `_COLS_FG_APRENDIZ_ANTES` |
| `relatorios.print_phase_e_analysis` | `print_e_passos_2_3` |
| `relatorios.malformadas_fase_e` | `malformadas_e` |
| `relatorios.print_final_analysis` | `print_abc_resumo` |
| `relatorios.print_error_analysis_by_region` | `print_a_passo4_erros_por_margem` |
| `relatorios.print_hyperparameter_sensitivity` | `print_a_passo4_sensibilidade_perceptron` |
| `relatorios.print_example_order_analysis` | `print_e_passo5_ordem` |
| `relatorios.print_statistical_summary` | `print_sumario_estatistico` |
| `relatorios.summarize_cross_linearity` | `sintese_r2_r3_r4` |
| `plots.plot_metric_errors_phase_a` | `plot_a_passo4_erros` |
| `plots.plot_fase_a_fewshot_curve` / `_g` | `plot_passos_2_a_5_curva` / `plot_g_passos_2_a_5_curva` |
| `plots.plot_w_distribution` | `plot_a_passo4_w_distribuicao` |
| `plots.plot_algorithm_comparison` | `plot_a_passo4_perceptron_vs_nnls` |
| `plots.plot_seed_comparison` | `plot_bc_passo3_sementes` |
| `plots.plot_consistency_comparison_extended` | `plot_bc_passo3_consistencia` |
| `plots.plot_r3_comparison` | `plot_a_passo4_r2_r3_r4` |
| `plots.plot_gamma_convergence` | `plot_a_passo4_gamma` |
| `plots.plot_class_names_effect`, `plot_class_order_bias`, `plot_feature_names_effect`, `plot_prompt_variant_comparison` | `plot_vies_nomes_classe`, `plot_vies_ordem_classes`, `plot_vies_nomes_atributo`, `plot_vies_variantes_prompt` |
| `plots.plot_oracle_w_recovery`, `plot_oracle_meialua` | `plot_oraculo_recuperacao_w`, `plot_d_passo5_teto` |
| `plots.plot_phase_e_learning_curve`, `plot_phase_e_strategy_comparison`, `plot_phase_e_example_locations`, `plot_dilution_experiment`, `plot_example_order_bias`, `plot_classical_baselines_comparison` | `plot_e_passo3_curva`, `plot_e_passo3_estrategias`, `plot_e_passo3_exemplos`, `plot_e_passo4_diluicao`, `plot_e_passo5_ordem`, `plot_e_passo6_baselines` |
| `plots.plot_external_learning_curve`, `plot_external_features_comparison`, `plot_external_decision_boundary`, `plot_phase_e_llm_vs_perceptron` | `plot_fg_aprendiz_curva`, `plot_fg_passos_2_e_4_r2_r3_r4`, `plot_fg_fronteiras`, `plot_fg_llm_vs_perceptron` |
| `plots.plot_meialua_svm_vs_llm` | `plot_d_passo6_svm` |
| `plots.plot_llm_labels_per_problem` | `plot_fg_rotulos_llm` |
| `plots.visualize_all_problems`, `visualize_problem_e_with_expert`, `plot_problem_overview`, `plot_dataset_overview`, `plot_hits_and_errors`, `plot_w_comparison_algorithms`, `plot_confusion_matrices_detailed`, `plot_margin_analysis_detailed`, `plot_experiment_summary_dashboard`, `plot_model_comparison`, `plot_labels_symbols` | `plot_abc_passo1`, `plot_e_passo1`, `plot_problema_passo1`, `plot_final_dados`, `plot_final_acertos_erros`, `plot_final_w_algoritmos`, `plot_final_matrizes_confusao`, `plot_final_margem`, `plot_final_painel_resumo`, `plot_final_comparacao_modelos`, `plot_rotulos_simbolos` |
| `gerar_tabelas_bloco1.tabela_T1..T9` e `gerar_tabelas.tab_*` | nomes por passo: `tab_a_passos_2_a_5`, `tab_a_zero_vs_exemplos`, `tab_a_passo6_eleicao`, `tab_a_passo4_r2_r3_r4`, `tab_a_passo4_perceptron_vs_nnls`, `tab_bc_passo3_todas_metricas`, `tab_bc_passo3_zero_vs_melhor_por_semente`, `tab_bc_passo3_comparativo`, `tab_bc_passo4_exemplos_da_metrica`, `tab_d_passos_2_a_5`, `tab_d_zero_vs_exemplos`, `tab_ad_por_regime`, `tab_ad_estabilidade_w`, `tab_vies_*`, `tab_oraculo`, `tab_d_passo5_teto`, `tab_e_passo3_estrategia_x_k`, `tab_e_zero_vs_exemplos`, `tab_e_passo3_h4`, `tab_e_passo6_baselines`, `tab_e_passo4_diluicao`, `tab_e_passo5_ordem`, `tab_f_passo2`, `tab_f_passo3`, `tab_g_passos_2_e_4`, `tab_g_passo6`, `tab_g1_vs_g2`, `tab_g_nomes`, `tab_g_passos_2_a_5`, `tab_final_*` |
| `classical_baselines.ClassicalBaselineRunner` e chaves `*_vs_expert` | ficam (Passo 6 de E; referência = especialista) |
| `audit_interactions.por_regime` | lê `passo=`; devolve chave `provider/modelo|problema|passo=N|regime|k=K` (5 campos; `tab_final_auditoria_flip` depende desse formato) |
| `gerar_tabelas.carregar_tudo` chaves `abc`, `proto`, `bc`, `alg`, `r3r4`, `oraculo`, `oraculo_ml`, `fase_e`, `dil`, `ordem`, `bl`, `ext_a`, `ext_e`, `proto_g`, `cross` | `abc`, `ad_passos_2_a_5`, `bc_passo3`, `a_passo4_perc_nnls`, `a_passo4_r3r4`, `oraculo`, `d_passo5`, `e`, `e_passo4`, `e_passo5`, `e_passo6`, `fg_passos_2_e_4`, `fg_aprendiz`, `g_passos_2_a_5`, `cross` |

## Colunas de CSV

| CSV | Antigo | Novo |
|---|---|---|
| ABC | `n_shot_phase_bc` | `k_passo4_bc` |
| ABC | `consistency_transferred_b/c`, `kappa_transferred_b/c` | `consistency_centroides_a_b/c`, `kappa_centroides_a_b/c` |
| ABC | `repetition` | `rep` (todos os CSVs usam `rep`) |
| passos 2 a 5 (A, D, G) | `regime` valores `zero_shot` / `few_shot` / `ground_truth` | `passo2_zero_shot` / `passo3_exemplos_rotulo_real` / `passo5_teto` |
| passos 2 a 5 | `fidelity_gt_metric_vs_llm_zero` | `fidelity_teto_vs_llm_passo2` |
| passos 2 a 5 | `accuracy_ceiling_vs_true` | `accuracy_teto_vs_true` |
| bc passo3 | `source_regime`, `source_k`, `source_rep`, `fidelity_source_mean`, `melhor_metrica_a` | `metrica_regime`, `metrica_k`, `metrica_rep`, `metrica_fidelidade_a`, `eleita_passo6` |
| bc passo3 | `collapsed_source` | `collapsed_metrica` |
| aprendiz (E) | `n_shot` | `k` ; `example_strategy` fica |
| fg passo2 | `feature_names` etc. ficam; `fidelity_perc_vs_llm` fica |
| fg aprendiz | `n_shot` → `k`; `expert_regime/expert_k/expert_rep` → `especialista_regime/k/rep`; `expert_source` → `especialista_origem` (valores `rotulo_real` / `metrica`) |
| cross | fica |
| tags | `fase=A` → `passo=2` (zero-shot) ou `passo=3` (exemplos do rótulo real); `fase=ABC` zero-shot de B/C → `passo=2`; B/C few-shot `regime=few_shot_metrica` → `passo=4|regime=exemplos_da_metrica`; `fase=ABC_aux` vieses → `passo=2` (k = 0) ou `passo=4` (k > 0, B/C com exemplos da métrica) com `regime=vies_<fator>`; `fase=E` → `passo=2` (zero-shot) / `passo=3` (estratégia) / `passo=4` (`fase=E_diluicao`) / `passo=5` (`fase=E_ordem`); F `fase=A` → `passo=2`, `fase=E` → `passo=3`; G `fase=A` → `passo=2/3`, `fase=E` → `passo=6` |

## Valores de `regime` por tag (formato `<bloco>|problema=<P>|passo=<N>|regime=<regime>|k=<k>|rep=<r>|seed=<s>[|extra]`)

| Problema (campo `problema=`) | Passo | `regime` | Observação |
|---|---|---|---|
| A, D (`adg_passos_2_a_5`) | 2 | `passo2_zero_shot` | k = 0 |
| A, D | 3 | `passo3_exemplos_rotulo_real` | k ∈ `K_EXEMPLOS_ROTULO_REAL`, rep = sorteio |
| `ABC` (`abc_executar`, configuração base e vieses) | 2 | `passo2_zero_shot` | k = 0: a tag `problema=ABC\|passo=2` agrupa o zero-shot de A, B e C numa coleta só (um prompt por ponto dos três problemas) |
| `ABC` | 4 | `exemplos_da_metrica` | k ∈ `K_EXEMPLOS_METRICA_BC`: só B e C são coletados (exemplos rotulados pela métrica do A, H2) |
| `ABC` (vieses) | 2 (k = 0) ou 4 (k > 0) | `vies_nomes_classe`, `vies_ordem_classes`, `vies_variantes_prompt` | k ∈ `K_VIESES`; o Passo 4 é B/C com exemplos da métrica |
| `ABC` (vieses) | 2 | `vies_nomes_atributo` | só zero-shot |
| `E` | 2 | `passo2_zero_shot` | uma coleta por (modelo, seed), compartilhada por E1/E2/E3 |
| `E1`, `E2`, `E3` | 3 | `easy`, `hard`, `mixed`, `random` | estratégia de seleção, k ∈ `K_EXEMPLOS_APRENDIZ` \ {0} |
| `E1` | 4 | `diluicao` | extra `easy=<n>`; k = 4 + n |
| `E1` | 5 | `class0_first`, `class1_first`, `shuffled`, `alternating` | ordenações do mesmo conjunto (`EXAMPLE_ORDERINGS`) |
| `F_meia_lua_seed<S>` | 2 | `passo2_zero_shot` | zero-shot no treino (105 pontos); extra `nomes=A/B\|atributos=x1/x2` |
| `F_meia_lua_seed<S>` | 3 | `zero_shot` (k = 0) / `exemplos_rotulo_real` (k > 0) | aprendiz classificando os 45 de teste |
| `G1_*`, `G2_*` | 2 | `passo2_zero_shot` | zero-shot nos 70 de treino; extra `nomes=…\|atributos=…` |
| `G1_*`, `G2_*` | 3 | `passo3_exemplos_rotulo_real` | k ∈ `K_EXEMPLOS_ROTULO_REAL` nos 70 de treino |
| `G1_*`, `G2_*` | 6 | `zero_shot` (k = 0) / `exemplos_da_metrica` (k > 0) | aprendiz classificando os 30 de teste; exemplos rotulados pela métrica eleita no Passo 5 |

Os Passos 4 a 6 do A, 4 a 6 do D, 4, 5 e 7 do G, 4 do F, 6 do E e o oráculo não chamam o LLM, logo não têm tag.

## Nomes de arquivo (CSVs com timestamp)

| Antigo | Novo |
|---|---|
| `bloco1_phases_abc` | `bloco1_abc_passos_2_a_4` |
| `bloco1_fase_a_protocolo` | `bloco1_ad_passos_2_a_5` |
| `bloco1_transferencia_bc` | `bloco1_bc_passo3_consistencia` |
| `bloco1_algorithm_comparison` | `bloco1_a_passo4_perceptron_vs_nnls` |
| `bloco1_r3r4_comparison` | `bloco1_a_passo4_r2_r3_r4` |
| `bloco1_oracle_validation` | `bloco1_oraculo_passos_1_a_3` |
| `bloco1_oracle_meialua` | `bloco1_d_passo5_teto` |
| `bloco2_phase_e` | `bloco2_e_passos_2_3` |
| `bloco2_dilution` | `bloco2_e_passo4_diluicao` |
| `bloco2_example_order` | `bloco2_e_passo5_ordem` |
| `bloco2_classical_baselines` | `bloco2_e_passo6_baselines` |
| `bloco23_external_phase_a` | `bloco23_fg_passos_2_e_4` (F: Passo 2; G: Passos 2 e 4, a estimação da métrica no G é o Passo 4; nome intermediário de 13/10 `bloco23_fg_passo2`, também migrado pela ferramenta) |
| `bloco23_external_phase_e` | `bloco23_fg_aprendiz` |
| `bloco3_fase_a_protocolo` | `bloco3_g_passos_2_a_5` |
| `final_cross_linearity__<alias>` | fica |
| `dados_sinteticos_seed*/problem_X.csv` | `problema_X.csv` |

## Nomes de gráficos (só o `<assunto>`; número e bloco ficam)

| Antigo | Novo |
|---|---|
| `bloco1_graphic_01_ABC_problemas_lineares` | `bloco1_graphic_01_ABC_passo1_dados` |
| `bloco1_graphic_02_A_rotulos_simbolos_seedS` | `bloco1_graphic_02_A_passos_2_a_4_simbolos_seedS` (painéis `_passo1_rotulo_real`, `_passo2_llm_zero_shot`, `_passo3_llm_kshot`, `_passo4_metrica`) |
| `bloco1_graphic_03_A_fase_a_erros_seedS` | `bloco1_graphic_03_A_passo4_erros_seedS` |
| `bloco1_graphic_04_AD_fase_a_fewshot_curva` | `bloco1_graphic_04_AD_passos_2_a_5_curva` |
| `bloco1_graphic_05_A_w_distribuicao` | `bloco1_graphic_05_A_passo4_w_distribuicao` |
| `bloco1_graphic_06_A_perceptron_vs_nnls` (painéis `_fase_b/_fase_c`) | `bloco1_graphic_06_A_passo4_perceptron_vs_nnls` (painéis `_bc_passo3_b/_bc_passo3_c`) |
| `bloco1_graphic_07_A_sementes` | `bloco1_graphic_07_BC_passo3_sementes` |
| `bloco1_graphic_08_A_R2_R3_R4` | `bloco1_graphic_08_A_passo4_R2_R3_R4` |
| `bloco1_graphic_09_A_gamma_convergencia` | `bloco1_graphic_09_A_passo4_gamma` |
| `bloco1_graphic_10_BC_consistencia` | `bloco1_graphic_10_BC_passo3_consistencia` |
| `bloco1_graphic_11_{B,C}_rotulos_simbolos_seedS` | `bloco1_graphic_11_{B,C}_passos_2_3_simbolos_seedS` (painéis `_passo1_rotulo_real`, `_passo2_llm_zero_shot`, `_passo3_metrica_eleita`) |
| `bloco1_graphic_12_D_meialua_seedS_overview` | `bloco1_graphic_12_D_passo1_dados_seedS` |
| `bloco1_graphic_13_D_rotulos_simbolos_seedS` | `bloco1_graphic_13_D_passos_2_a_4_simbolos_seedS` |
| `bloco1_graphic_14_D_svm_vs_llm_seedS` | `bloco1_graphic_14_D_passo6_svm_seedS` |
| `bloco1_graphic_15..18_vies_*` | ficam |
| `bloco1_graphic_19_oraculo_recuperacao_W` | `bloco1_graphic_19_oraculo_passos_1_a_3` |
| `bloco1_graphic_21_D_teto_oraculo_meialua` | `bloco1_graphic_21_D_passo5_teto` |
| `bloco2_graphic_01_E_especialista` | `bloco2_graphic_01_E_passo1_especialista` (painel `_ground_truth` → `_rotulo_real`; `_fronteira_perito`/`_margem_perito` ficam) |
| `bloco2_graphic_02_E_estrategias_exemplos` | `bloco2_graphic_02_E_passo3_estrategias_exemplos` |
| `bloco2_graphic_03_E_curva_aprendizado` | `bloco2_graphic_03_E1_passo3_curva` |
| `bloco2_graphic_04_E_comparacao_estrategias` | `bloco2_graphic_04_E1_passo3_estrategias` |
| `bloco2_graphic_05_E_baselines_classicos` | `bloco2_graphic_05_E1_passo6_baselines` |
| `bloco2_graphic_06_E_diluicao` | `bloco2_graphic_06_E1_passo4_diluicao` |
| `bloco2_graphic_07_E_ordem_exemplos` | `bloco2_graphic_07_E1_passo5_ordem` |
| `bloco3_graphic_01_G1_peso_altura`, `_02_G2_peso_altura_kg_m` | `bloco3_graphic_01_G1_passo1_dados`, `bloco3_graphic_02_G2_passo1_dados` |
| `bloco3_graphic_03_G_fase_a_fewshot_curva` | `bloco3_graphic_03_G_passos_2_a_5_curva` |
| `bloco23_graphic_01_FG_curva_aprendizado` | `bloco23_graphic_01_FG_aprendiz_curva` |
| `bloco23_graphic_02_FG_R2_R3_R4` | `bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4` (intermediário de 13/10: `FG_passo2_R2_R3_R4`) |
| `bloco23_graphic_03_FG_fronteiras_decisao` | `bloco23_graphic_03_FG_fronteiras` |
| `bloco23_graphic_04_FG_llm_vs_perceptron` | fica |
| `final_graphic_01_ABCE_dados_seedS` | `final_graphic_01_ABCE_passo1_dados_seedS` |
| `final_graphic_02..06` | ficam (`_matrizes_confusao`, `_acertos_erros`, `_w_algoritmos`, `_margem`, `_painel_resumo`) |
| `final_graphic_07_FG_rotulos_llm_<prob>_seedS_<f0>_<f1>_<kind>` | `<kind>` = `train` → `passo2`; `<k>` → `passo3_k<k>` (F) / `passo6_k<k>` (G) |
| `final_graphic_08_comparacao_modelos` | fica |

## Slugs de tabela (`<bloco>_table_NN_<slug>`; número fica)

| Antigo | Novo |
|---|---|
| `A_zero_fewshot_teto` | `A_passos_2_a_5` |
| `A_zero_vs_fewshot_comparacao` | `A_passo2_vs_passo3` |
| `A_escolha_melhor_metrica_transferencia` | `A_passo6_eleicao` |
| `A_R2_R3_R4_zero_fewshot` | `A_passo4_R2_R3_R4` |
| `A_perceptron_vs_nnls_por_semente` | `A_passo4_perceptron_vs_nnls` |
| `BC_transferencia_todas_metricas_A` | `BC_passo3_todas_metricas_A` |
| `BC_transferencia_zero_vs_melhor_por_semente` | `BC_passo3_por_semente` |
| `BC_comparativo_zero_vs_melhor` | `BC_passo3_zero_vs_eleita` |
| `BC_fewshot_exemplos_da_metrica` | `BC_passo4_exemplos_da_metrica` |
| `D_meialua_R2_R3_R4` | `D_passos_2_a_5` |
| `D_zero_vs_fewshot_comparacao` | `D_passo2_vs_passo3` |
| `A_vs_D_por_regime` | `AD_por_passo` |
| `AD_estabilidade_W_por_semente` | `AD_passo4_estabilidade_W` |
| `vies_*` | ficam |
| `oraculo_recuperacao_W` | `oraculo_passos_1_a_3` |
| `D_teto_oraculo_meialua` | `D_passo5_teto` |
| `E{1,2,3}_faseE_estrategia_x_nshot` | `E{1,2,3}_passo3_estrategia_x_k` |
| `E_zero_vs_fewshot_por_especialista` | `E_passo2_vs_passo3` |
| `E_H4_media_fewshot_por_estrategia` | `E_passo3_H4` |
| `E_LLM_vs_baselines_nmax` | `E_passo6_baselines` |
| `E1_diluicao`, `E1_ordem_exemplos` | `E1_passo4_diluicao`, `E1_passo5_ordem` |
| `F_meialua_faseA_R2_R3_R4` | `F_passo2_R2_R3_R4` |
| `F_meialua_faseE_nshot` | `F_passo3_k` |
| `G_faseA_R2_R3_R4` | `G_passos_2_e_4_R2_R3_R4` (intermediário de 13/10: `G_passo2_R2_R3_R4`) |
| `G_faseE_nshot` | `G_passo6_k` |
| `G_faseE_LLM_vs_perceptron` | `G_passo7_LLM_vs_perceptron` |
| `G1_vs_G2_zero_shot` | `G1_vs_G2_passos_2_e_4` (intermediário de 13/10: `G1_vs_G2_passo2`) |
| `G_nomes_semanticos_vs_neutros` | fica (caption: "Problema G, Passos 2 e 4") |
| `G_faseA_zero_fewshot_teto` | `G_passos_2_a_5` |
| `final_*` | ficam |

Regra: o slug da tabela cita o passo PRINCIPAL (o que gera as linhas); a caption lista TODOS os passos
do conteúdo (ex.: slug `A_passo2_vs_passo3`, caption "Problema A, Passos 2 a 4"; slug `G_passo6_k`,
caption "Problema G, Passo 6"; `bloco3_table_05_G_nomes_semanticos_vs_neutros`, caption "Problema G, Passos 2 e 4").

## Testes (arquivos)

`test_fase_a_protocolo.py` → `test_adg_passos_2_a_5.py`; `test_fase_a_protocolo_g.py` → `test_g_passos_2_a_5.py`; `test_selecao_exemplos.py`, `test_reorder_examples.py` → `test_e_passo3_exemplos.py`, `test_e_passo5_ordem.py`; demais ficam, com classes/métodos renomeados onde citam fases.

## Ferramentas offline

`regerar_figuras_simbolos.py` → `regerar_simbolos.py`; `corrigir_mapeamento_homem_mulher.py` → remover (legado de 07/2026 sem uso); `renomear_assets_legado.py` ganha as regras desta tabela (arquivos e colunas) com `--aplicar`.

### Complemento de 14/10/2026 (parâmetros e variáveis internas)

| Nome antigo | Nome novo |
|---|---|
| parâmetro/variável `n_shot` (em `reps_para`, `bc_passos_2_a_4`, `abc_executar`, `_abc_executar_configuracao`, `e_passos_2_3_aprendiz`, laços do `main()`, `plots.py`, `relatorios.py`, `classical_baselines.py`, testes) | `k_exemplos` |
| `n_shots` / `n_shot_bias` / `n_shot_e` / `n_shot_bl` / `n_shot_order` / `n_shot_val` / `_config_base(n_shot_comum=)` / `_draw_n_shot` | `valores_k` / `k_vies` / `k_e` / `k_bl` / `k_ordem` / `k_val` / `_config_base(k_comum=)` / `_draw_k` |
| `EXAMPLE_ORDER_N_SHOTS` (também chave de `GRADES_MODO_RAPIDO`) | `K_ORDEM_EXEMPLOS` |
| `test_n_shot_impar_*` | `test_k_impar_*` |
| (novo) | `plots._ordem_fg` (F antes de G; Mulher/Homem antes de A/B nas figuras bloco23) |

