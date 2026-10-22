# Revisão de código — 08/10/2026

Escopo: `src/` (15.949 linhas), `tests/` (201 testes, todos passando). Cinco leituras completas em paralelo (runner em 3 fatias, `plots.py` + `relatorios.py`, módulos auxiliares + testes) mais varreduras automáticas (pyflakes, parâmetros sem uso, constantes órfãs). Cada achado abaixo foi confirmado lendo o código ou executando o trecho; os marcados com ✔ foram reproduzidos nesta sessão.

## Resumo

A nomenclatura nova (A–D, E1/E2/E3, F, G1/G2, Oráculo sobre B/E1/E2, especialista/aprendiz) está aplicada em todo o código e nos assets. O risco principal não é de nome, é de **corretude em quatro pontos que afetam números já coletados ou que serão coletados na execução completa**: (1) exemplos few-shot com atributos R³/R⁴ montados errado no prompt; (2) sorteio de exemplos da Fase E externa que difere entre variantes de atributo e não segue `seed + rep`; (3) tag de auditoria do Bloco 3/F herdada do Bloco 2; (4) parser que aceita o artigo inglês "a" como classe A. Além disso, várias figuras e relatórios do Bloco 1 misturam as variantes de viés nas médias da configuração base.

## Bloqueadores (corrigir antes da execução completa)

| # | Local | Problema | Correção |
|---|---|---|---|
| B1 ✔ | `dissertacao_mestrado.py:705, 816, 863` (`build_prompt_few_shot*`) | `for i in range(3, len(ex) - 1)`: com exemplo `(x1, x2, x3, label)` o laço é vazio e x1·x2 some; com `(x1, x2, x3, x4, label)` imprime só `ex[3]` (x2²) com o nome fixo `x3` e descarta x1². O ponto de teste recebe `x1*x2 = …`, `x1² = …`; os exemplos não. Afeta a Fase E externa de D/F, G1 e G2 quando a melhor métrica é R³/R⁴ (aconteceu nos smokes de hoje e na execução de julho). | `for j, i in enumerate(range(2, len(ex) - 1)): nome = extra_features[j][0] if extra_features else f"x{i+1}"; line += f", {nome} = {ex[i]:.4f}"` nos três pontos. Teste novo com exemplos de 4 e 5 elementos comparando o texto do exemplo com o do ponto. |
| B2 | `dissertacao_mestrado.py:3123–3126, 3255–3260` (`run_external_problem_pipeline`) | `rng = RandomState(seed)` faz o split e depois é consumido pelo sorteio de exemplos ao longo de `feature_variants × n_shots × reps`. A 2ª variante de atributos (peso/altura no G1, x1/x2 no G2) recebe exemplos diferentes da 1ª para o mesmo `(seed, n_shot, rep)`, confundindo efeito do nome com sorteio. CLAUDE.md diz `random_state = seed + rep`, o que o código não faz aqui. | `rng_ex = np.random.RandomState(seed + rep)` por `(n_shot, rep)`, independente da variante; manter o `rng` do split. Documentar. |
| B3 ✔ | `dissertacao_mestrado.py:4106` + Bloco 3 (4577–4722) + F (4727–4793) | `CURRENT_TAG` fica em `bloco2\|fase=E\|seed=<último>` e nunca é redefinida: no smoke de hoje 1.440 interações do G1/G2 levam tag do Bloco 2 e 700 levam tag dos vieses do Bloco 1. A auditoria por regime (T5) fica cega para o Bloco 3. | Setar a tag dentro de `run_external_problem_pipeline` por (problema, regime, k, rep, seed) e zerar ao sair do loop de modelos. Idem Fase E/diluição/ordem: incluir `problema=`, `regime=`, `k=`, `rep=`. |
| B4 ✔ | `llm_parser.py:88–96` (camada 4) | `parse("This is a borderline point", "A", "B")` → `("A", True)`: o artigo "a" vira classe A com `valid=True`, invisível à taxa de malformadas. Com a variante `cot` (raciocínio em inglês) e nomes de 1 letra, qualquer hedge sem um "B" isolado vira A. | Para nomes de 1 caractere, casar case-sensitive (`\bA\b` sem `re.IGNORECASE`) ou exigir contexto (`class|classe|answer|:`). Teste: o caso acima deve dar `(None, False)`. |
| B5 ✔ | `llm_client.py:97–106` (`erro_de_credito`, código de 08/10) | `"402"` como substring: `request_id=req_a402b`, `max_tokens 4020`, `context length 14020` disparam abort de execução paga. | Usar `getattr(exc, "status_code", None) == 402` e, no texto, `re.search(r"\b402\b")` só junto de "error code"/"status". Testes negativos com os três exemplos. |

## Importantes (afetam números ou figuras do trabalho)

| # | Local | Problema | Correção |
|---|---|---|---|
| I1 | `dissertacao_mestrado.py:1073–1100` | Prompt de retentativa de formato é fixo ("2D points", `x1`/`x2`, `-> {ex[2]}`): ignora nomes de atributo (G2: peso_kg/altura_m), atributos extras e variante; com exemplos de 4/5 elementos `ex[2]` é um atributo, não o rótulo. O registro guarda só o prompt original, então a auditoria não vê a troca. | Reconstruir o retry com `build_prompt_*_variant(...)` + frase "IMPORTANT: …". |
| I2 | `dissertacao_mestrado.py:1054–1058` | Após retry de formato, grava `raw = all_responses[0]` (a resposta malformada) com `malformed=False` e o rótulo vindo de outra resposta. A auditoria mede flip sobre `raw_response` e conta divergência espúria. | `raw = label` (resposta válida) e gravar `all_responses` também no ramo válido. |
| I3 ✔ | `audit_interactions.py:124–131` | Flip a T=0 compara `raw_response` (texto bruto): `"A"` vs `"A."` conta como flip. O número reportado (T5, 7,5% zero-shot no gpt) pode estar inflado por formatação. | Agrupar por `parsed_label`; reportar `flip_raw` separado se quiser. Teste: raw diferente, parsed igual → flip 0. |
| I4 | `dissertacao_mestrado.py:2422–2448` (`run_complete_experiment`, colapso) | Colapso na Fase A devolve `fidelidade=0.5`, `consistencia=0.5`, `kappa=0`, `w=[1,1]` que entram no CSV `bloco1_phases_abc`, nas médias e nos plots como medições. | NaN + coluna `collapsed=1` (como já faz `phase_a_protocol`); filtrar no export e nos plots. |
| I5 | `dissertacao_mestrado.py:3229–3236, 3269–3280` | No zero-shot da Fase E externa os atributos extras R³/R⁴ entram no prompt quando a melhor métrica é R³/R⁴; o ponto `n_shot=0` do `bloco23_external_learning_curve` e o `final_08_*_0` não são comparáveis ao zero-shot da Fase A (só x1, x2). | `extra_matrix_test = … if (n_shot > 0 and best['n_features'] >= 3) else None`, ou documentar. |
| I6 | `dissertacao_mestrado.py:1548, 1634` (Fases B/C few-shot e baselines) | Exemplos rotulados por `W_A + centróides do A` (transferidos), mas a consistência é medida com centróides locais. O rótulo dos exemplos pode discordar da métrica avaliada. Vale pouco com `--bc-zero-shot`, mas o código continua lá. | Selecionar/rotular com `W_A` + centróides locais, ou documentar na docstring. |
| I7 | `dissertacao_mestrado.py:2832–2853` (`evaluate_transfer_bc`) | Docstring promete "todas as métricas do A", mas só R² é transferida (`n_feat != n_features` descarta R³/R⁴); o print "contra N métricas" conta 3× mais. | Corrigir docstring e contagem, ou chamar também com `n_features=3, 4`. |
| I8 | `plots.py:254–265, 358–362, 461–465, 897–902, 1153–1154, 2328–2356`; `relatorios.py:523–547`; runner `4926–4927` | Figuras `bloco1_08/09/12/13/14a/07`, dashboard e `print_final_analysis` agregam `abc_m` inteiro: em k=0 e k=10 as médias misturam 4 nomes de classe × inversões × 4 variantes de prompt × nomes de atributo; em k=4/20/40 só A/B. Barras não comparáveis entre k, scatter de `bloco1_07` com ~18 pontos Perceptron contra 3 NNLS. `print_statistical_summary` filtra certo (`relatorios.py:612`). | Filtro único de configuração base (`prompt_variant == "default"`, `feature_names == ("x1","x2")`, nomes conforme a figura) aplicado em todas essas funções. |
| I9 ✔ | `relatorios.py:458–473` | A média por estratégia que decide "H4 refutada" usa `df` (E1+E2+E3), não `df_main`. | `df_main[df_main['n_shot'] > 0]`. |
| I10 | `plots.py:1926, 1939, 2192` | `final_05` e `final_07` desenham fronteira/margem de B e C com centróides do A, mas acertos/erros vêm de centróides locais. | Gravar centróides locais em `seed_detailed_data` e usá-los. |
| I11 | `plots.py:2672, 2715, 2777, 2847` | Pipelines externos agrupam por `problem_name`; `F_meia_lua_seed{N}` (e antes `meia_lua_seed{N}`) vira 3 grupos de n=1, sem barra de erro, enquanto o log agrega os 3 seeds. | `problem_base = problem_name.replace(r'_seed\d+$', '')` nas 4 funções. |
| I12 | `dissertacao_mestrado.py:4719–4723, 4791–4793` | `try/except Exception` em volta do Bloco 3 inteiro e do F: erro de código em G1 pula G2 em silêncio, a pasta ainda ganha `_completa` e o exit code é 0. | Um `try` por subproblema, lista `falhas_fatais`, `SystemExit(2)` no fim (ou sufixo `_incompleta`). |
| I13 | `dissertacao_mestrado.py:1840–1876` (`run_oracle_meialua`) | Chama de `fidelity_vs_true` uma medida contra o rótulo real (é acurácia, teto do D) e grava só `w_0`, `w_1` em R³/R⁴. | Renomear para `accuracy_vs_true` (consumidor `plots.py:214`) e gravar `w_recovered_{i}` para todos os pesos. |
| I14 | `animacao_perceptron/gerar.py:136–146, 112` | Nomes legados (`meia_lua_seed`, `homem_mulher`, ordem Homem/Mulher) e grupo "D" para o pipeline do F: numa `_completa` nova omite meia-lua e peso×altura em silêncio. | Ler `nome_problema_f`, `HM_*_CLASS_NAME_VARIANTS`; aceitar legado como fallback; rotular como F. |
| I15 | `corrigir_mapeamento_homem_mulher.py:157–174` | Sem guarda para máscara vazia: em execução nova grava cópias `_corrigido_hm` idênticas e tabelas vazias. | `if not mask.any(): raise SystemExit(...)`. |
| I16 | `plots.py:305, 319` | Boxplot do `bloco1_09` rotula consistência de B/C como "Acu B/C" e "Acurácia". | "Consist. B/C", "Consistência". |
| I17 | `plots.py:1738–1739, 1750, 1796–1808, 1652` | Oráculo: títulos "Oráculo sobre B, E1 e E2" mas a Parte 3 (anisotrópicos) entra nos mesmos eixos; "Fidelidade" e "acurácia" para a mesma grandeza na mesma figura; `replace('Problema ', '')` não casa mais; rótulo "Oráculo sobre E1\nE1" redundante; `bloco1_04` ("transfer") ficou com um tick por painel e repete o painel 4 do `bloco1_03`. | Um termo só ("acurácia vs rótulos do oráculo"), filtrar ou citar os anisotrópicos, simplificar rótulos; decidir se `bloco1_04` fica. |
| I18 | `dissertacao_mestrado.py:280, 298, 3450, 3573` | `FEW_SHOT_SIZES` só inicializa `FEW_SHOT_SIZES_BC` e é impressa; o cabeçalho do log mostra `[0,4,10,20,40]` numa execução `--bc-zero-shot`. | Imprimir `FEW_SHOT_SIZES_BC` e `FEW_SHOT_SIZES_FASE_A`; remover a reatribuição em `--rapido`. |

## Polimento (texto, código morto, consistência)

- **Docstrings e comentários desatualizados:** runner `:65–67` (D via pipeline externo), `:493` (`RUN_PROBLEM_MEIALUA` "D e F"; só F), `:137` ("11 CSVs"; são 13), `:22` ("C in [0.1, 1.0]"; C=1,0), `:329–331` (tiers da OpenAI; `MAX_CONCURRENCY` é 30), `:1314/1385/2465/3838` ("validação cruzada" para o NNLS; é segundo estimador), `:1904` ("rótulos verdadeiros" no oráculo; são do oráculo), `:1964` (mixed "hipoteticamente a melhor"; H4 refutada), `:2383/2447` (anotação de retorno com 6 elementos; são 7), `:2388` ("W aprendido" → "Ŵ_LLM estimada"), `:2395` (print em inglês); `audit_interactions.py:4–15` ("três propriedades"; são quatro); `execucao_io.py:49–55` (comentário do limite acima da constante errada); `gerar_tabelas_bloco1.py:193–195` (`kk` e comentário fora de lugar); `gerar_amostras_homem_mulher.py:185` (README usa âncoras do peso para a altura); `data_problems.py:33, 92` (data "14/05/2026" para a migração problem_D→E; CLAUDE.md diz 01/10/2026); `plots.py:195` (docstring "fidelidade" no teto do D), `:3–4` ("36 funções"; são 43), `:78` ("Aprendizado da Métrica" → "Estimação"), `:159` ("E1" fixo embora receba `expert_w`), `:2529` ("Meia-lua" → "Problema D"); referências a reunião/e-mail dentro de títulos de figura (`plots.py:227, 1619, 1623, 2530, 2751`).
- **Prints de seção fora da convenção:** runner `:4727` "BLOCO 1/2 — PROBLEMA D/F" (é F, Bloco 2); `:5406/5442` "BLOCOS 1/3 … meia-lua D + peso×altura G" (é F e G1/G2); `[PASSO N]` inconsistente (`:3643…5511`); `:3694` "Seeds: {seed}" (uma seed); "oracle" em inglês em comentários (`:3594, 4094, 4808, 5339`).
- **Código morto:** pyflakes: imports `d_W`, `PROBLEM_A_CENTERS`, `PROBLEM_C_CENTERS`, `MODEL_ALIAS`, `_model_slug`, `asset_variant` (runner `:168–202`), `glob`/`os` (`corrigir_mapeamento`), `glob` (`gerar_tabelas`, `regerar`), `numpy` (`test_gerar_tabelas`); `max_tok` (`:1006`), `global client` (`:3378`), 131 f-strings sem placeholder. Parâmetros nunca lidos: `run_oracle_validation(X_a, y_a, X_c, y_c)` (+ `gamma` descartado), `select_examples_dilution(random_state)`, `_run_phase_abc_experiment(seed_idx)`, `run_external_problem_pipeline(pasta_execucao)`, `ClassicalBaselineRunner.run(n_shot)`. Também: 2ª chamada de `marcar_melhor_metrica_a` (`:5168`), `_fallback_model` e os `fillna` (`:5345, 5412, 5445`), `ORDER_TEST_N_SHOTS` (`:4316`), `if run_aux` redundante (`:3846`), `ConsistencyMetrics.precision/recall` calculados e nunca lidos (`metrics.py:165–202`), `LearnedMetric.source_problem` só escrito, `PROVIDER_CONFIG["client_type"]` só nos testes, `data.get('centroids_nnls')` (`plots.py:1980`), colunas `model/nomes/fidelidade` criadas e não usadas (`plots.py:238–252, 502`), ramo inalcançável em `relaxed_perceptron.py:336–338`, `RUN_R3R4_EXPERIMENT` recalcula o que `phase_a_protocol` já grava (`bloco1_10` duplica linhas do protocolo).
- **CSV com colunas inconsistentes:** `bloco2_dilution` omite `n_malformed_responses`, `accuracy_*_vs_gt`, `expert_w_*` que `bloco2_example_order` exporta (`:5172` vs `:5287`); `bloco1_algorithm_comparison` descarta `llm_accuracy_*`, `metric_accuracy_*`, `n_malformed_responses` do resultado NNLS (`:5105–5125`).
- **Estimadores:** `relaxed_perceptron.py:270, 282` clips não citados no Algoritmo 1; `:284–290` `w_best_effort` é o w do fim da época mas as violações foram contadas durante a época; `least_squares_inverse.py:96–99` rival por distância não ponderada (irrelevante com 2 classes; documentar); teto efetivo de γ = `max_iterations·delta_gamma` só no warning.
- **Parser:** negação ignorada na camada 5 ("Não é Positivo" → Positivo); documentar.
- **Testes fracos:** `test_estimators.py:33` (`assertIsInstance(float(gamma), float)`), `test_audit_interactions.py:23–27` (`parsed_label == raw_response` esconde I3), faltam casos para B4 e B5; testes de alias dentro de `TestRegraRepeticao` (`test_selecao_exemplos.py:116–172`) e duplicado com `test_renomear_assets_legado.py:51–54`; `test_plots_nomes.py:34` usa ids legados; nomes de teste com "perito".
- **Outros:** `--apenas oraculo` ainda exige chave e consulta saldo; `find_latest_execution(root=".")` default não é `resultados/`; `regerar_figuras_simbolos` pode ler `llm_interactions_checkpoint.json` duplicado; `RESULTADOS_DIR` redefinido em 5 módulos em vez de importado de `execucao_io`; chave do `PHASE_A_PROTOCOL_CACHE` sem nomes de classe; `select_confident_examples` pode devolver menos de `n_shot` sem aviso; T2 põe em negrito Perceptron e NNLS mas o rodapé diz "Perceptron"; Wilcoxon nunca roda com 3 sementes mas a nota do log diz que foi usado; `consistency_transferred` do ablativo só é impresso.

## Conferido e correto

Anti-vazamento dos exemplos few-shot em todas as fases; k/2 por classe no protocolo da Fase A; `X_display` do G2 (prompt em kg/m, estimação e avaliação no espaço normalizado, elipse mapeada por afinidade exata); `reps_para` e `seed + rep` no protocolo; rotação B/C; gates `model_idx == 0`; nomes de asset via `llm_asset`/`asset_variant`; oráculo sobre B/E1/E2 sem A e sem E3; nenhuma função morta em `plots.py`/`relatorios.py`; nenhum "perito/expert/fonte" em texto; `print_statistical_summary`, `plot_w_distribution`, `plot_fase_a_fewshot_curve`, `summarize_cross_linearity` e `plot_classical_baselines_comparison` agregam o grupo certo.

## Ordem sugerida de correção

1. B1, B2, B3, B5 (prompt, sorteio, tag, 402) e I1, I2, I5: tudo que muda o que o LLM vê ou o que fica registrado. Precisam entrar antes da execução completa.
2. B4 e I3 (parser e auditoria): mudam a leitura das respostas; vale reauditar as execuções existentes depois.
3. I4, I8, I9, I10, I11, I16, I17 (CSV e figuras): afetam o que vai para o texto; podem ser regerados offline.
4. I6, I7, I12 a I15, I18 e o polimento.

## Status das correções (aplicadas em 08/10/2026, suíte com 212 testes passando)

**Bloqueadores:** B1, B2, B3, B4, B5 corrigidos (prompts com atributos extras nos 3 builders; sorteio
`RandomState(seed + rep)` igual entre variantes; tags por bloco/fase/problema/regime/k/rep/seed no pipeline
externo, na Fase E, na diluição e na ordem; parser case-sensitive para nomes de 1 letra nas camadas 4, 6 e
7; 402 só por `status_code` ou "error code: 402").

**Importantes:** I1 (retry reconstrói o mesmo estímulo), I2 (`raw_response` = resposta válida,
`all_responses` gravado), I3 (flip por `parsed_label`, `taxa_flip_raw_T0` à parte; na execução de julho os
dois coincidem: gpt 4,3%, Gemini 0%), I4 (`collapsed=1` + NaN, fora de `all_results_abc`), I5 (zero-shot
externo sem atributos extras), I6 (documentado), I7 (docstring + contagem em R²), I8 (`_config_base` em
`bloco1_08/09/12/13/14a`, dashboard, `bloco1_07` e `print_final_analysis`), I9 (H4 só E1), I10 (centróides
locais em `final_05`/`final_07`), I11 (`problem_base` nas figuras `bloco23`), I12 (G1, G2 e F em `try`
próprios, `falhas_fatais`, exit 2 e "EXPERIMENTO INCOMPLETO"), I13 (`accuracy_vs_true`, `w_recovered_0..3`),
I14 (animação lê nomes novos e legados), I15 (guarda de máscara vazia), I16, I17, I18.

**Polimento aplicado:** `max_tokens` da variante passa a ser enviado (cot = 300); parâmetros mortos
removidos (`X_a/y_a/X_c/y_c` e `gamma` do oráculo, `random_state` da diluição, `seed_idx`,
`pasta_execucao` do pipeline externo, `n_shot` dos baselines); imports e variáveis sem uso; 2ª chamada de
`marcar_melhor_metrica_a`; `_fallback_model`; `ORDER_TEST_N_SHOTS`; chave do cache do protocolo com nomes de
classe; aviso em `select_confident_examples`; CSVs de diluição e de comparação de algoritmos com as
colunas completas; `--apenas oraculo` sem chave e sem saldo; docstrings e comentários listados; textos de
figuras ("rótulo real", "Consistência", "atributos", "curva de aprendizado", "Problema D/F/G1/G2");
estimadores documentados (clips, best-effort, rival euclidiano, ramo inalcançável removido); testes
renomeados e fortalecidos.

**Não alterado, de propósito:** `RUN_R3R4_EXPERIMENT` continua recalculando o que o protocolo grava
(redundância documentada, resultados idênticos); `ConsistencyMetrics.precision/recall` e
`LearnedMetric.source_problem` permanecem (inócuos); o `[PASSO N]` do log não foi renumerado; os
reexports `d_W`, `asset_variant`, `MODEL_ALIAS`, `_model_slug` no runner são usados pelos testes
(pyflakes os aponta, mas são intencionais); 131 f-strings sem placeholder (cosmético).

## Segunda passada (mesmo dia, após as correções) — 213 testes passando

Cinco revisores releram tudo com foco em regressões e textos antigos. Achados e tratamento:

- **Regressões das correções (corrigidas):** a segunda chamada aos baselines (dentro da Fase E) ainda
  passava `n_shot`, removido da assinatura (derrubaria a execução completa); `compute_centroids` em
  `final_05`/`final_07` quebrava quando o LLM colapsa em classe única em B/C (agora `_centroides_locais`
  com a mesma guarda de `phase_consistency_test`); `--apenas oraculo` criava o cliente sem chave
  (cliente só quando há chamadas pagas); G2 dependia de variáveis criadas no `try` do G1 (base carregada
  num bloco próprio, G1 e G2 independentes); a lista de blocos falhados era impressa depois de salvar o
  log (agora consta em `log_execucao.txt`); `max_tokens` da variante era calculado e não enviado.
- **Lógica (corrigida):** colapso da Fase A entra no cache (sem recoletar os 150 pontos a cada n_shot/rep)
  e `run_complete_experiment` reaproveita os rótulos; sentinelas 0,5 removidas também de
  `phase_a_learn_metric`; registro de falha definitiva com `all_responses`; erros retentáveis por código
  de status (não por substring "400"/"500"); tags próprias para nomes de classe, inversão, variantes de
  prompt, nomes de atributo e para a recoleta zero-shot da comparação de algoritmos; `CURRENT_TAG`
  zerada ao sair do pipeline externo; `n_classe_*_llm` NaN no regime `ground_truth`; comparação
  Perceptron×NNLS no log filtra o modelo; `final_09` só com a configuração base; `bloco23` decision
  boundary e resumo da Fase E externa agregam por problema-base; `final_cross_linearity` ganha
  `problem_base`; CSV R³/R⁴ com linhas `perceptron`/`nnls` em R² (sem `llm_2feat`); custos por bloco
  somam o da execução completa; chave do cache do protocolo documentada; `boxplot(tick_labels=)`.
- **Textos (corrigidos):** cabeçalho do módulo (Semaphore 30, E1/E2/E3, G1/G2, protocolo no A, sem
  números de execução), cabeçalho do pipeline externo (F, G1/G2; sem itens de reunião), "Validação
  cruzada" no print do NNLS, "APRENDIZADO" → "ESTIMAÇÃO", "Oráculo sobre aniso …" na Parte 3, log em
  português, "oráculo" em vez de "oracle", referências a reunião/e-mail fora de títulos de figura
  (`final_10`, `bloco1_04b`, `bloco1_11`, `bloco23_features`), "rótulo real" no lugar de "ground
  truth"/"GT", "atributos" no lugar de "features", "Matriz de confusão", "Razão", travessão nos títulos
  de figura substituído por vírgula/dois-pontos; docstring do parser com a regra case-sensitive;
  clip do fator do Perceptron em [0, 1] com comentário coerente; glossário (κ > 0,4, 8 camadas,
  temperatura, concorrência, "estimada"); README (`--rapido`, árvore com `regerar_figuras_simbolos`);
  CLAUDE.md (oráculo, teto do D).
- **Limpezas:** `with tempfile` sem uso nos testes; teste do colapso passou a exercitar
  `run_complete_experiment` com dublê; `examples_dil if n_total > 0`; `main()` sem retorno não usado;
  animação lê o CSV mais recente, filtra zero-shot/colapso e importa `PERCEPTRON_PARAMS` de
  `protocolo`; `corrigir_cross_linearity` com guarda; regex de regime alinhada; `httpx` pinado.
- **Novo teste estático** `tests/test_assinaturas_chamadas.py`: toda chamada a função definida em
  `src/` é conferida contra a assinatura (kwargs inexistentes, posicionais a mais). Teria pegado o
  `n_shot` esquecido.
- **Mantido de propósito:** sufixos de painel `_fronteira_perito`/`_margem_perito` e
  `bloco2_01_problema_e_expert.png` (nomes de asset documentados; mudar exigiria migração);
  `ConsistencyMetrics.precision/recall` e `LearnedMetric.source_problem`; `[PASSO N]` do log;
  ids legados nos testes de nomes de asset; `RESULTADOS_DIR` repetido nos scripts offline (evita
  importar `execucao_io` neles); reexports no runner usados pelos testes.

**Smoke test final (22:36, 288 arquivos, exit 0):** tags por experimento corretas em todos os blocos,
exemplos few-shot com atributos extras iguais ao ponto de teste, `collapsed` no CSV, oráculo "sobre
B/E1/E2/aniso", `accuracy_vs_true` + `w_recovered_0..3`, R³/R⁴ com perceptron/nnls, diluição com 17
colunas, G1/G2/F concluídos. Um efeito colateral detectado e revertido: enviar `max_tokens` (50/300)
truncou 256 respostas do Gemini que raciocinam antes do rótulo (1,1% de fallback MD5, contra 2 no
smoke anterior). Decisão: `max_tokens = None` em todas as variantes (comportamento de julho), com
nota em `PROMPT_VARIANTS`; o texto do apêndice que fala em "limite ampliado de 50 para 300 tokens"
na variante cot deve ser corrigido para "sem limite de tokens". "Provider returned error" (404
transitório do OpenRouter, 2 casos) passou a ser retentável.

## Varredura final (após o smoke test das 22:36) — 213 testes passando

Cinco revisores releram o código definitivo. **Nenhum erro de lógica encontrado.** Ajustes aplicados:
cabeçalho do módulo virou docstring de fato (`load_dotenv()` passou para depois dele); oráculo
documentado com a Parte 3 anisotrópica; anotação de `phase_a_learn_metric` com `Optional` e nota do
colapso; mensagem de esgotamento de retentativas; freio global de 429 por código, não por substring;
`if True:` e `if run_aux` redundante removidos; `.get('X_test_c')` inócuo; "Problema D (meia-lua com o
LLM como especialista)"; "atributos" na docstring de `phase_a_multifeature` e no print de ganho;
"rep 1" só quando a métrica eleita é few-shot; títulos de figura do runner sem travessão e sem menção
ao orientador; log em português ("MODELO", "Teto do Problema D"); bloco "EXECUÇÃO INCOMPLETA" impresso
uma vez (no log) e resumido no fim; `main()` sem `return` morto; `plots.py` sem referências a reunião
em docstrings, `_draw_acuracia_oraculo`, coluna `rep` morta removida, travessão do `bloco1_10`;
`final_cross_linearity` com `problem_base` também nas linhas lineares; mensagens do relatório sem
travessão; animação usa a chave `F` para o Problema F; teste estático passa a acusar argumentos
obrigatórios faltando; `import tempfile` sem uso removido; CLAUDE.md (custos 11/3,5/5,5/2, "em R²",
variantes de nome de atributo, subconjunto de colunas da diluição, `problem_D.csv`, quatro
propriedades da auditoria, flip 4,3%/0%, `bloco3_02`, contagens de linhas, chave única do OpenRouter,
"Validação do oráculo") e README (flip 4,3%).
