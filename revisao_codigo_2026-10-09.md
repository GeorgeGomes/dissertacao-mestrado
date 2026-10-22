# Revisão de código — 09/10/2026

Escopo: `src/` (18.276 linhas com `tests/`), 223 testes (219 passam, 4 pulados). Oito leituras independentes em paralelo, uma por problema ou camada: Problema A (Fase A, protocolo, R3/R4, Perceptron × NNLS), Problemas B/C (transferência, H1/H2), Problema D e oráculo, vieses e prompts, Problema E (Bloco 2), Problemas F/G1/G2 (pipelines externos), fechamento (tabelas, estilo P&B, nomenclatura) e coleta do LLM/infra. Cada leitura cruzou código, comentários, CLAUDE.md, os CSVs e PNGs da última completa (`execucao_2026-10-08_23-16-54_completa`, nomes antigos) e os PNGs e `.tex` do último smoke (`execucao_2026-10-09_14-50-07_smoke`, nomenclatura nova). Varreduras automáticas: pyflakes, vulture. Achados marcados com ✔ foram reproduzidos nesta sessão (script ou leitura direta do trecho); os demais vêm das leituras com evidência citada. Nenhum arquivo do projeto foi alterado.

## Resumo

A estrutura (3 blocos, nomenclatura A–G, E1/E2/E3, G1/G2, convenção `<bloco>_<graphic|table>_<NN>` com `_color/_gray`) está aplicada e os números da completa batem com as constantes do protocolo (contagens de chamadas por bloco, k/2 por classe, `seed + rep`, cache da Fase A, tags 100% preenchidas, parser reproduzindo 157.854 rótulos). O que não bate é de três tipos:

1. **Três bloqueadores**: o "especialista" do Problema F não é o ground truth como o CLAUDE.md, o artigo e a tabela dizem; a conversão P&B rotula e traça as tampas das barras de erro como séries, o que vai estragar todas as figuras com barras de erro da próxima completa (no smoke já aparece em 6 figuras); e o laço principal (Blocos 1 e 2) não tem proteção de exceção, então uma execução de ~2h e US$ 10 pode terminar sem log e sem CSV.
2. **Resultados apresentados como coisa que não são**: acurácia "vs rótulo real" em zero-shot e em classes neutras A/B sem invariância à permutação (vários lugares); F1 de B/C que é a própria acurácia; "transferência" do oráculo que não existe; oráculo das Partes 1 e 3 rotulado pelo gerador e não pela regra do centróide mais próximo; gráficos do oráculo mostrando uma só seed; elipse "ótima" do Bloco 3 nunca desenhada; `final_table_01` misturando variantes de atributo.
3. **Dívida de manutenção**: 4 testes de regressão permanentemente pulados, cálculos descartados (NNLS pareado de B/C, baselines de B/C, `nnls` dos nomes de atributo), ~30 comentários/docstrings com numeração antiga de assets, README e glossário desatualizados.
4. **Métricas de avaliação fora da seção IV.3 do orientador** (seção própria abaixo): F1 nunca é macro (três variantes, a principal igual à acurácia), consistência violada nos vieses de atributo, acurácia e fidelidade usadas fora do escopo definido, e quatro famílias de métrica inexistentes no PDF nas saídas (consistência euclidiana, acurácia do especialista vs GT, "concordância"/"taxa de erro", precision/recall).

## Status em 09/10/2026 (fim do dia)

Onda 1 de correções aplicada em `src/` e `tests/` (300 testes; 336 após o fechamento da noite de 09/10, 332 após a remoção das configurações extras do oráculo em 13/10/2026 e 346 após o protocolo da Fase A no Problema G, na tarde de 13/10; o restante desta frase descreve o estado de 09/10: noite de 09/10, ver a subseção "Fechamento das pendências"); documentação (CLAUDE.md, README, glossário, skills) alinhada no mesmo dia. Decisões registradas: **D1** (B1) especialista do Problema F = rótulo real (`expert_labels`, coluna `expert_source`); **D2** (M10) acurácia vs rótulo real continua nos Blocos 1 e 3 como contexto, sempre "Acurácia (vs rótulo real)", nunca em coluna de fidelidade; **D3** (M11) fidelidade continua em F/G (mesmo conceito); **D4** (M7) exemplos few-shot de B/C rotulados por w_A com os centróides de A, consistência medida com centróides locais, declarado em docstrings e rodapés. Status conferido no código em 09/10/2026 (não apenas na lista de mudanças).

| Item | Status | Observação |
|---|---|---|
| B1 | feito (D1) | `run_external_problem_pipeline(..., expert_labels=)`; F passa `y_ml`; Fase E do F roda mesmo com colapso da Fase A; `expert_source` em `bloco23_external_phase_e` |
| B2 | feito | `estilo_cinza.py` ignora `ErrorbarContainer` (tampas/hastes), barras com `label=''` em eixo com legenda, `axhline/axvline`; `tests/test_estilo_cinza.py` (16 testes) |
| B3 | feito | `bloco_protegido` nos Blocos 1 e 2 por (modelo, seed) e em G1/G2/F: exceção → `falhas_fatais`, checkpoint, exit 2; CSVs/log sempre salvos |
| P1 | feito | `EXEC_REF` → `execucao_2026-07-05_12-34-52_completa`; 4 testes voltaram a rodar |
| P2 | feito | camada 7 com fronteira de palavra; não decide se a outra classe aparece |
| P3 | feito | camada 0 usa o ÚLTIMO marcador e limpa `**`/`\boxed{}`; contradição marcador × última linha = malformada; camada 3 ignora decimais e usa o último inteiro |
| P4 | feito | timeout (`API_TIMEOUT_S=90`) vira exceção retentável; 400 só retentável com mensagem transitória; falha definitiva grava prompt real e `format_retries=0`; `estimar_custo_usd(["oraculo"], True) == 0` |
| P5 | feito | tags dos vieses com `k=` e fator (`nomes=`, `variante=`, `atributos=`); Fase E zero-shot `regime=zero_shot` |
| P6 | feito | `accuracy_expert_vs_gt`/`n_class_*_expert` sobre o teste em Fase E, diluição e ordem; `diagonal_limitation_flag` removida (junto com M15) |
| P7 | feito | novos: `test_estilo_cinza.py` (B2), `test_correcoes_2026_10_09.py` (timeout → retentável, 400, checkpoint na falha, oráculo custo 0, colapso, zero-shot 1×, F com rótulo real, sorteio balanceado, H4, 3-tuplas, tag) e, na noite de 09/10, `test_infra_main.py` (grades do `--rapido` via `grades_modo_rapido`/`aplicar_modo_rapido`, `nome_pasta_execucao` com `_completa`/`_smoke`/`_apenas-<blocos>[_smoke]`, `Tee` restaurado após `_abortar_por_creditos`, roundtrip `salvar_json_em_chunks` → `load_interactions` com 1 e várias partes, ciclo de vida do cliente da API). Suíte: 336 testes |
| I1 | feito | F1 `average='macro'` em `metrics.py`, `classical_baselines.py` e Fase E externa (M1/M2) |
| I2 | feito | coluna "Acur. vs real (inv. perm.)" = max(a, 1 − a) nas tabelas 01_14, 03_01, 03_02, 03_04, 03_05; nota de permutação nos rodapés zero-shot |
| I3 | feito | idem; o Δ semântico − neutro da `bloco3_table_05` usa a invariante |
| I4 | feito | Partes 1 e 3 do oráculo rotuladas por `predict_with_metric(X, centros, W_verdadeiro)`; nomes `x1_preciso`/`x2_preciso`; `expert_name` "B" |
| I5 | feito | `bloco1_graphic_19` agrega 3 seeds (média, mín–máx) |
| I6 | feito | `bloco1_graphic_20` removido; `plot_oracle_transfer` apagada; texto de "transferência" retirado do CLAUDE.md |
| I7 | feito | `_elipse_otima` com constante calibrada pela acurácia máxima; elipse desenhada em `bloco3_graphic_01/02` |
| I8 | feito | `final_cross_linearity` com `problem`, `problem_name`, `feature_names`; meia-lua do Problema D; `accuracy_vs_true` em `A_linear`; `final_table_01` por (problem_name, feature_names) |
| I9 | feito | nomes de atributo roteados por `_run_phase_abc_experiment` (centróides locais, NNLS pareado); só zero-shot |
| I10 | feito (D4) | `consistency_transferred_b/c` e `kappa_transferred_b/c` no CSV (ablação); regra dos exemplos declarada |
| I11 | feito | tabelas 03/07/08 leem `melhor_metrica_a` do CSV (regra única `marcar_melhor_metrica_a`) |
| I12 | feito | comparação Perceptron × NNLS pareada (`_nnls_bc`, 0 chamadas); recoleta de B/C removida |
| I13 | feito | `MIN_PONTOS_CLASSE_MINORITARIA = 5` (`protocolo.py`), regra única `colapsou()` em todas as fases |
| I14 | feito | marcador † nas tabelas 01, 02, 03, 04, 05, 10 (γ = 0); `NOTA_GAMMA` no rodapé |
| I15 | feito | mixed exatamente 50/50 (n = 10 → 5 + 5) |
| I16 | feito | zero-shot da Fase E coletado 1× por (modelo, seed), `zero_shot_e_cache` |
| I17 | feito | `_balanced_example_indices` (k/2 por classe, `RandomState(seed + rep)`) na Fase E externa |
| I18 | feito | `final_graphic_07` inclui as variantes `_classesAB` |
| F1 | feito | título "sementes e nomes de classe"; marcador por par de nomes |
| F2 | feito | curvas de γ rotuladas "seed \| nomes \| variante" |
| F3 | feito | 15: barras pareadas normal × invertido por regime (painéis `_{consistencia_b,consistencia_c,fidelidade}_{0,10}shot`); 16: só os 4 pares base (`_0shot/_10shot`) |
| F4 | feito | painéis de W de 06/17/18 usam `_w_unitario` (direção) |
| F5 | parcial | fallback neutro "amostras (1), (2)…" para scatters de cor única; os scatters de acerto/erro de `final_graphic_06` ganharam rótulo explícito ("métrica ≠ LLM (n)"); o ramo de colapso de `bloco1_graphic_14` não foi conferido |
| F6 | feito | `label='Problema B'/'Problema C'` em `final_graphic_08` |
| F7 | feito | k-NN agrupado por família (uma série), k no rodapé; F1-Macro dos dois lados |
| F8 | feito | painel E de `final_graphic_01` com os rótulos do especialista |
| F9 | feito | `_draw_symbols_scheme` desenha `boundary['centroids']` quando há `boundary` |
| F10 | feito | margem com W normalizado por ΣW e ticks `.2g` em `_erros_por_margem` |
| F11 | feito | `bloco1_graphic_07` só zero-shot |
| F12 | feito | painel `_ratio` do gráfico 19 em escala log |
| F13 | feito | painel "LLM k-shot" no D; colapso anotado no título; fronteira do SVM escolhida por `fidelity_perc_vs_llm` |
| F14 | parcial | legendas Mulher/Homem (`_nomes_classes`) e painel do especialista (`y_metric`) em `final_graphic_07`; layout/suptitle e eixos do `bloco23_graphic_03` não conferidos |
| F15 | feito | cinzas ≤ 0,6; séries de 2 pontos mantêm traço/símbolo; `axhline/axvline` como referência; scatter `c=y` rotulado divide sem repetir contagem |
| F16 | feito | `bloco2_table_04` média das médias por (estratégia, n); slug `bloco2_E_LLM_vs_baselines_nmax`; tabela 15 só zero-shot; 13 com "—" no colapso; 06 filtra `n_features == 2`; `faltam` removido |
| F17 | feito | `gerar_tabelas_bloco1.py` grava `tabelas/todas_as_tabelas_bloco1.tex/.docx` |
| F18 | feito | `RE_PAINEL` exclui `color/gray` |
| M1 | feito | F1-Macro em `compute_consistency_metrics`; precision/recall removidos |
| M2 | feito | `average='macro'` em `classical_baselines.py` e na Fase E externa |
| M3 | feito | "F1-Macro" em figuras, tabelas e log (0 ocorrências de "F1-Score") |
| M4 | feito | "Cohen's κ" em figuras, tabelas e log (0 ocorrências de "Kappa" solto como rótulo) |
| M5 | feito | = I9 |
| M6 | feito | colapso em B/C → NaN + `collapsed_b/c`; `evaluate_transfer_bc` → NaN + `collapsed_llm` |
| M7 | feito (D4) | regra declarada em docstrings e rodapés; fórmula inalterada |
| M8 | feito | coluna `accuracy` (← `fidelity`) e painel `_acuracia` (← `_fidelidade`) no oráculo |
| M9 | feito | `fidelity_vs_llm` = NaN no regime `ground_truth`; `accuracy_ceiling_vs_true`; `fidelity_gt_metric_vs_llm_zero` |
| M10 | feito (D2) | acurácia vs rótulo real mantida como contexto, sempre nomeada "Acurácia (vs rótulo real)"/"Acur. vs real" |
| M11 | feito (D3) | fidelidade mantida em F/G com o mesmo conceito |
| M12 | feito | "Acurácia (LLM vs especialista)" no Bloco 2; "Acur. vs espec." nas tabelas |
| M13 | feito | `*_llm_vs_expert` ← `*_llm_vs_metric`; rodapé e CLAUDE.md alinhados com D1 |
| M14 | feito | baselines de B/C (que reutilizavam `*_vs_expert`) removidos com a recoleta; as chaves `*_vs_expert` só existem no Bloco 2, onde a referência é o especialista |
| M15 | feito | consistência euclidiana e `diagonal_limitation_flag` removidas de campos, CSVs, log e testes |
| M16 | feito | "Concordância"/"Taxa de erro"/"Acerto/Erro" renomeados (ex.: "1 − fidelidade por faixa de margem") |
| M17 | feito | `fidelity_vs_llm` (← `fidelidade`) e chaves internas `fidelity`/`fidelity_nnls` no R3/R4 |
| M18 | feito com decisão | κ = 0,8 exato continua "quase perfeita"; docstring de `faixa_kappa` cita o PDF e registra a decisão |
| M19 | feito | CLAUDE.md (seção "Métricas de avaliação", H2, H4, oráculo, gráfico 19), glossário (cinco verbetes reescritos; Precisão, Recall e "Euclidiana é o baseline" removidos), README e skills atualizados em 09/10/2026 |
| Código morto | feito, com 2 pendências | removidos: `confidences_test`, `y_pred_b/c_alt`, `examples_bl`, `gamma_r3`, `fid_feat_nnls`/`cons_*_feat_nnls` (via `_run_phase_abc_experiment`), `baselines_b/c`, `scope`/`EXTRA_SEEDS_CORE`/`is_seed_extra`, `faltam`, `tabelas.media_min_max`, `llm_client.get_client`; `nnls_detail` agora é persistido (I12); `RUN_R3R4_EXPERIMENT` derivado do protocolo; `com_cor` usada por `salvar_figura`; coluna `nomes` usada no marcador do gráfico 05; `corrigir_mapeamento_homem_mulher.py` documentado como legado. `run_oracle_meialua` mantido por decisão e documentado como equivalente ao regime `ground_truth` do D (tabela 18 = linhas "teto" da 10; gráfico 21 = linha "teto" do 04; serve a `--apenas oraculo`), não é pendência; 132 f-strings sem placeholder (cosmético, sem ação) |
| Comentários e docs | feito | numeração antiga de assets corrigida nos comentários de `dissertacao_mestrado.py`, `plots.py`, `gerar_tabelas*.py`, `regerar_figuras_simbolos.py`, `execucao_io.py` e `relaxed_perceptron.py` (sem "final_10"); nomes `x1_preciso/x2_preciso`; docstring de `phase_consistency_test` com a grade completa como padrão; o limiar κ > 0,4 de H1 passou a ser avaliado e impresso no log (`relatorios.texto_h1`, noite de 09/10). Sem pendências |

## Rodada 2 de validação (09/10/2026, noite)

Três auditores independentes releram o código depois da onda 1, cada um com um recorte: **Bloco 1** (Fase A, protocolo, R3/R4, Perceptron × NNLS, oráculo, vieses e as figuras/tabelas do bloco), **Blocos 2/3** (Fase E, especialistas E1/E2/E3, baselines, pipelines externos F/G1/G2 e suas tabelas) e **transversal** (coleta e cliente LLM, parser, `estilo_cinza`, `renomear_assets_legado`, contratos de CSV e documentação). O método foi o mesmo da onda 1: cruzar código, CSVs e PNGs da última completa (nomes antigos) e do último smoke (nomes novos), reinterpretar as 157.854 respostas da completa com o parser atual e conferir o CLAUDE.md contra o que o código faz.

Achados e status:

| Achado | Onde | Status |
|---|---|---|
| Exemplos da Fase E externa descritos como "iguais entre variantes de atributo"; só vale no Problema F (especialista = rótulo real). Em G1/G2 o especialista é a métrica estimada dos rótulos zero-shot do LLM, que muda com a variante, logo os índices sorteados diferem e o efeito do nome de atributo não fica isolado na Fase E | `run_external_problem_pipeline`, CLAUDE.md | **resolvido na noite de 09/10**: `_balanced_example_indices(y_train, k, seed + rep)` balanceia pelo rótulo REAL do treino, índices idênticos entre variantes em F e G (exemplos ainda rotulados pelo especialista); superado em 13/10/2026 (decisão do autor: mesmos exemplos entre variantes, ver Decisão final) |
| `bloco1_algorithm_comparison` descrito como "linhas NNLS pareadas com a linha Perceptron" sem dizer que o CSV só tem as linhas NNLS (uma por modelo × seed); o par Perceptron está em `bloco1_phases_abc` | CLAUDE.md | corrigido em 09/10 (rodada 2), documentação |
| Oráculo anisotrópico: `x1_preciso` e `x2_preciso` são quase perfeitamente separáveis por qualquer W (rótulos pelo centróide mais próximo coincidem com os euclidianos); o Perceptron devolve W = [1, 1], cosseno ≈ 0,75, acurácia 100%. Falta de identificabilidade, não falha do estimador | `ORACLE_ANISO_CONFIGS`, `bloco1_table_17`, `bloco1_graphic_19` | **resolvido por flag na noite de 09/10**: `RUN_ORACLE_ANISO_EXTRAS = False` deixa o oráculo só com B/E1/E2 (Problemas 1/2/3 do PDF); centros e desvios intactos; superado em 13/10/2026 (configurações extras removidas, ver Decisão final) |
| `bloco1_graphic_02`/`13`: 4º painel documentado como "LLM 10-shot" fixo; é "LLM k-shot" (painel `_llm_<k>shot`, k = 10 se estiver na grade, senão o menor de `FEW_SHOT_SIZES_FASE_A`: 10 na completa, 4 no `--rapido`) | `_llm_fewshot_scheme`, CLAUDE.md | corrigido em 09/10 (rodada 2), documentação |
| Duração da completa divergente entre CLAUDE.md ("1h15–1h45" e "2h23 com 30") e README ("1h15 a 2h30"); a única medida é 2h23 com `MAX_CONCURRENCY=30` | CLAUDE.md, README.md | corrigido em 09/10 (rodada 2) |
| Convenção resumida do README sem `[_<problema>][_seed<S>]` | README.md | corrigido em 09/10 (rodada 2) |
| `bloco1_r3r4_comparison`: linhas R2 sem `w0, w1` | `dissertacao_mestrado.py` | corrigido em 09/10 (rodada 2) |
| `bloco2_phase_e`: as linhas zero-shot são a MESMA coleta repetida 12× por (modelo, seed) (4 estratégias × 3 especialistas); somar `n_malformed_responses` entre elas infla a contagem | CLAUDE.md (nota na seção CSVs) | corrigido em 09/10 (rodada 2), documentação |
| `_tab_vies` (tabelas 14/15/16) com "10-shot" fixo no regime e no rodapé; no `--rapido` o k é 4 | `gerar_tabelas.py` | corrigido em 09/10 (rodada 2): regime = k presente em `BIAS_N_SHOTS`/no CSV |
| `final_table_01`: nota de permutação restrita a algumas linhas; vale para todas as linhas zero-shot (nomes A/B); meia-lua aparece como `D_meialua` | `gerar_tabelas.py` | corrigido em 09/10 (rodada 2) |
| `renomear_assets_legado.py` propunha `problem_D.csv → problem_E.csv` num smoke novo (já na convenção) | `renomear_assets_legado.py` | corrigido em 09/10 (rodada 2): regra só quando `problem_E.csv` não existe; `_color/_gray` reconhecidos; nada proposto em execução já na convenção |
| `_erro_retentavel` não reconhecia "timed out" (mensagem do `APITimeoutError` do SDK) | `dissertacao_mestrado.py` | corrigido em 09/10 (rodada 2) |
| Parser: reinterpretação da completa de 08/10 dá 7 divergências (4 trocas de rótulo, 2 respostas contraditórias agora malformadas, 1 antes malformada agora aceita), todas CoT do Gemini | `llm_parser.py` | verificado em 09/10 (rodada 2); números registrados no CLAUDE.md |
| `estilo_cinza`: gráfico 04 `_gray` com séries da mesma cor indistinguíveis (chave só por cor) | `estilo_cinza.py` | corrigido em 09/10 (rodada 2): chave cor + marcador + traço |
| `estilo_cinza`: linha de dados de `errorbar(fmt='o')` tratada como tampa no gráfico 19 | `estilo_cinza.py` | corrigido em 09/10 (rodada 2) |
| Proxies de fronteira (0 pontos) na legenda dos gráficos 14 e `bloco3_graphic_01` com cor/traço errados | `estilo_cinza.py` | corrigido em 09/10 (rodada 2): proxies pretos com o linestyle do contorno |
| Gráfico 09: handle colorido na versão `_gray` | `estilo_cinza.py` | corrigido em 09/10 (rodada 2) |
| `final_graphic_05`: Mediana e Média com o mesmo traço na `_gray` | `estilo_cinza.py`/`plots.py` | corrigido em 09/10 (rodada 2) |
| `bloco1_graphic_10`: linhas de referência sem legenda | `plots.py` | corrigido em 09/10 (rodada 2) |
| `bloco23_graphic_01`: legenda cobrindo as séries | `plots.py` | corrigido em 09/10 (rodada 2) |
| Boxplot sem `patch_artist` virava série na legenda garantida | `estilo_cinza.py` | corrigido em 09/10 (rodada 2) |
| Tabelas 10/11 mostrando 50,0 e 150/0 numa linha colapsada | `gerar_tabelas.py` | corrigido em 09/10 (rodada 2) |
| `print_box` truncando o veredito de H4 no log | `relatorios.py` | corrigido em 09/10 (rodada 2) |
| Travessões usados como vírgula em linhas adicionadas em 09/10 (CLAUDE.md, skills) | documentação | corrigido em 09/10 (rodada 2) |

Pendências que ficavam depois da rodada 2 (todas tratadas na subseção seguinte):

- Oráculo anisotrópico não identificável em `x1_preciso`/`x2_preciso`.
- Exemplos da Fase E externa não iguais entre as variantes de atributo em G1/G2.
- Testes de infraestrutura P7 (reduções do `--rapido`, pasta/sufixo em `main()`, restauração do `Tee` no abort, roundtrip `salvar_json_em_chunks` ↔ `load_interactions`).
- O limiar κ > 0,4 de H1 sem ser avaliado e impresso no log.
- Ruído do httpx no log: tracebacks "Event loop is closed" ao fechar conexões depois do `asyncio.run`.

## Fechamento das pendências (09/10/2026, noite)

Última rodada de código do dia, conferida símbolo a símbolo em `src/` e `tests/`. Nenhuma chamada à API; o smoke em andamento não foi tocado.

| # | Item | O que mudou | Status |
|---|---|---|---|
| 1 | Oráculo: gaussianas anisotrópicas extras | Flag `RUN_ORACLE_ANISO_EXTRAS = False` ao lado de `RUN_ORACLE_MEIALUA`; `run_oracle_validation(..., incluir_extras=None)` lê a flag. Por padrão o oráculo roda só sobre B, E1 e E2 (Problemas 1/2/3 da seção IV.5 do PDF do orientador); `x1_preciso`, `x2_preciso` e `forte_aniso` (acréscimo nosso, pouco identificáveis) só com a flag ligada. Gráfico 19 (título conta as configurações presentes), tabela 17 e CSV `bloco1_oracle_validation` (18 linhas = 3 × 2 algoritmos × 3 seeds) saem só com B/E1/E2. Centros e desvios não mudaram | superado em 13/10/2026 (ver Decisão final: removido de vez, sem flag) (e-mail de 09/10, pergunta 1) |
| 2 | Fase E externa (F/G1/G2): sorteio dos exemplos | `_balanced_example_indices(y_train, k, seed + rep)` balanceia k/2 pelo RÓTULO REAL do treino (classe 0 recebe ⌈k/2⌉, classe 1 ⌊k/2⌋, sem reposição). Os exemplos continuam rotulados pelo especialista (rótulo real no F, melhor métrica no G); índices idênticos entre variantes de atributo e mesma regra em F e G. No G o balanço pelo rótulo do especialista pode não ser exato; o log imprime as contagens por classe real e por classe do especialista. A "limitação conhecida" do CLAUDE.md foi retirada | resolvido; superado em 13/10/2026 (decisão do autor: mesmos exemplos entre variantes, ver Decisão final) |
| 3 | H1 no log | `relatorios.texto_h1(resultados, limiar=LIMIAR_H1_KAPPA)`, `LIMIAR_H1_KAPPA = 0.4`; `print_statistical_summary` imprime "H1 (CONSISTÊNCIA): VEREDITO POR MODELO": κ médio em B e C (configuração base A/B, prompt default, x1/x2, zero-shot, Perceptron, sem colapso), faixa de Landis e Koch, seeds com κ > 0,4 e a linha "H1 (κ > 0.4): SUSTENTADA/NÃO SUSTENTADA em B \| ... em C" | feito |
| 4 | Malformadas da Fase E | `relatorios.malformadas_fase_e(df)` soma o few-shot normalmente e conta o zero-shot uma vez por (modelo, seed, rep) em vez das 12 cópias; log: "few-shot X + zero-shot Y (contado uma vez por seed) = Z" | feito |
| 5 | Cliente da API | `async_collect_llm_decisions` cria o `AsyncOpenAI` dentro do laço de eventos quando o global `async_client` é `None` (produção; o global é só injeção nos testes) e o fecha no `finally` com o laço aberto; o 402 atravessa o `gather` e `collect_llm_decisions` chama `_abortar_por_creditos` fora do laço, depois do fechamento. Sem tracebacks "Event loop is closed" no log | feito (fecha a pendência "ruído do httpx") |
[6 testes em 09/10, 332 após 13/10 | feito (fecha P7) |
| 7 | E-mail ao orientador | Rascunho descartado em 13/10/2026: o autor decidiu os dois pontos (item 1: remover de vez as configurações extras do oráculo, inclusive a flag; item 2: priorizar os mesmos exemplos entre as variantes de nome, com o balanço mais próximo possível de k/2) | resolvido |

Decisão final (13/10/2026): as gaussianas anisotrópicas extras do oráculo (`ORACLE_ANISO_CONFIGS`, `create_anisotropic_problem`, `RUN_ORACLE_ANISO_EXTRAS`, Parte 3 de `run_oracle_validation`) foram removidas do código, dos testes e da documentação; tabela 17 e gráfico 19 ignoram essas linhas em execuções antigas. O sorteio da Fase E externa fica como implementado (balanceado pelo rótulo real, mesmos índices entre variantes, exemplos rotulados pelo especialista, balanço pelo especialista impresso no log). Suíte: 332 testes.

Pedido do orientador (13/10/2026): bootstrap removido. `relatorios.bootstrap_ci` deu lugar a `resumo_entre_sementes` (média, mínimo e máximo entre sementes) e `diferenca_consistente` (faixa da diferença pareada não contém zero); Cohen's d permanece. Também em 13/10 o orientador pediu a remoção do Wilcoxon: o teste (que nunca rodava com 3 sementes) saiu de `print_statistical_summary`, do artigo e da documentação. Atualizados código, testes, CLAUDE.md, README, skills, roteiro e a seção "Análise Estatística" do artigo. Pendente no artigo: a seção de resultados ainda cita valores de IC$_{95}$ da execução antiga (linhas ~871, 1058, 1414, 1446, 1572, 1603) e será reescrita com a nova completa.

### Decisões de 13/10/2026 (tarde): protocolo da Fase A no Problema G

| # | Item | O que foi feito | Status |
|---|---|---|---|
| 1 | Fase A few-shot no G (pedido do orientador, Metodologia IV.7: "o LLM classifica o mesmo conjunto de treino em zero-shot e few-shot, n ∈ {4, 10, 20, 40}, e as métricas são avaliadas seguindo os mesmos passos do problema A") | `run_external_problem_pipeline(..., protocolo_fase_a=False)`: com `True` (main passa True em G1 e G2, False em F), depois do zero-shot dos 70 pontos de treino roda `phase_a_protocol` sobre eles: few-shot com k ∈ `FEW_SHOT_SIZES_FASE_A` (4, 10, 20, 40) e exemplos do rótulo real (k/2 por classe, `seed + rep`, 3 repetições), teto (W ajustado ao rótulo real), R²/R³/R⁴, Perceptron e NNLS; zero-shot reaproveitado (0 chamadas extras); o LLM vê só os 2 atributos (kg e m no G2 via `X_display`). `phase_a_protocol` ganhou `X_display=None`, `nome_feature_0="x1"`, `nome_feature_1="x2"`, `bloco="bloco1"`, `tag_extra=""`; `_make_tag(..., bloco="bloco1", extra="")`; `_balanced_gt_examples(..., X_display=None)`; a chave de `PHASE_A_PROTOCOL_CACHE` inclui o problema completo e os nomes de atributo (A e D inalterados). Linhas em `'phase_a_protocol_rows'` (`problem_name`, `subproblem` G1/G2, `class_names`, `feature_names`) → CSV `bloco3_fase_a_protocolo_{timestamp}.csv` (chave `proto_g` em `gerar_tabelas.carregar_tudo`), tabela `bloco3_table_06_G_faseA_zero_fewshot_teto__<alias>` (`tab_G_protocolo`) e figura `bloco3_graphic_03_G_fase_a_fewshot_curva__<alias>` (+ `_nnls`; 8 painéis subproblema × classes × atributos em R², `plot_fase_a_fewshot_curve_g`; `plot_fase_a_fewshot_curve` ganhou `panels`, `ncols`, `suptitle`). Especialista da Fase E eleito por `_eleger_especialista_externo` entre as métricas do Perceptron de zero-shot e few-shot (todos k, rep, n_features; sem teto e colapsos) pela maior `accuracy_metric_vs_true_full`; empate → zero-shot, k crescente, rep, R² → R³ → R⁴; novas colunas `expert_regime`, `expert_k`, `expert_rep` em `bloco23_external_phase_e` (NaN no F); a Fase E do G roda mesmo com o zero-shot colapsado. Tags externas (G e F, Fases A e E) com `nomes=` além de `atributos=`. Custo: `CUSTO_ESTIMADO_USD` completa 11 → 14,5 e bloco3 2 → 5,5 (≈ 29.664 chamadas extras na completa, ≈ 187 mil no total; +1.056 no `--rapido`). Suíte: 346 testes (`tests/test_fase_a_protocolo_g.py`, 14 testes). O Problema F não muda | implementado |
| 2 | Few-shot de B/C | Fica como está: grade {0, 4, 10, 20, 40}, exemplos rotulados pela métrica zero-shot do A (decisão D4); o orientador ajusta o texto da metodologia | mantido; texto do orientador a ajustar |
| 3 | R³/R⁴ em B e C | Não se aplica: B e C são problemas lineares de transferência do w do A; R³/R⁴ ficam em A, D e nos externos F e G | não se aplica |

O que resta:

- Nova execução completa (estimativa US$ 11 a 13 com as ≈ 30 mil chamadas extras do protocolo no G; a medição anterior foi US$ 8 a 10 para 158 mil chamadas; cerca de 2 h ou mais) e, depois dela, regerar tabelas e atualizar roteiro, slides e artigo com os números novos (inclusive `bloco3_table_06`, `bloco3_graphic_03` e a eleição do especialista por regime).

### Nomenclatura por passos adotada em 13/10/2026 (camada 1: textos visíveis; camada 2, identificadores internos, executada em 13/10/2026)

Decisão do autor (13/10/2026): nos textos visíveis (títulos, legendas, rodapés, log e documentação) os nomes "Fase A", "Fase B/C", "Fase E", "Fase A/E externa", "pipeline externo" e "protocolo único da Fase A" dão lugar a "Problema X, Passo N": cada problema tem a sua sequência de passos, numerada de 1 em diante, só com os passos que existem nele (listas por problema e tabela de equivalência na seção "Passos por problema" do CLAUDE.md; resumo: Fase A = Passos 2 a 5 de A e D; Fases B e C = Passos 2 a 4; Fase E = Passos 2 e 3 de E, com diluição, ordem e baselines nos Passos 4 a 6; Fase A externa = Passo 2 do F e Passos 2 a 4 do G; Fase E externa = Passo 3 do F e Passo 6 do G). Camada 1 aplicada em CLAUDE.md, README.md, glossario.md, problemas_tabela.docx, roteiro_apresentacao.txt, artigo.tex (só nomenclatura), trabalhos_referencias.txt e nas skills; os textos visíveis do código (`src/`) seguem em paralelo. Camada 2 (identificadores internos: funções como `phase_a_protocol`, constantes como `FEW_SHOT_SIZES_FASE_A` e `RUN_PHASE_E`, colunas de CSV, tags de auditoria `fase=A`/`fase=E`, nomes de arquivo de asset como `bloco1_fase_a_protocolo` e `bloco3_graphic_03_G_fase_a_fewshot_curva`) foi executada em 13/10/2026 (camada 2) a partir do mapa único `renomeacao_passos_2026-10-13.md`: funções, constantes, flags, campos, colunas e valores de CSV, tags (`fase=` → `passo=`), nomes de CSV, gráficos, painéis, slugs de tabela, testes e ferramentas (`regerar_figuras_simbolos.py` → `regerar_simbolos.py`; `corrigir_mapeamento_homem_mulher.py` removido; `renomear_assets_legado.py` ganhou as regras do mapa, e as execuções anteriores a 13/10/2026 precisam dele antes de `gerar_tabelas.py`). O histórico das rodadas acima mantém os nomes da época, por ser registro do que foi feito.

## Bloqueadores (corrigir antes da próxima execução completa)

| # | Local | Problema | Correção |
|---|---|---|---|
| B1 ✔ | `dissertacao_mestrado.py:3272-3300` (`run_external_problem_pipeline`) | Os exemplos da Fase E externa são rotulados por `y_metric_train` (melhor métrica Perceptron da Fase A externa) para F **e** para G. O comentário na l.3300 diz "não há especialista aqui". CLAUDE.md (visão geral e Fase E), `artigo.tex:668` e o rodapé de `bloco2_table_10` (`gerar_tabelas.py:588`) afirmam que no F o especialista é o rótulo real. Consequências na completa: a Fase E do F fica gateada pelo zero-shot (gpt colapsou nas seeds 123 e 7, logo F com gpt só tem seed 42); o "professor" do LLM no F tem 46 a 68% de acurácia vs real. | Decidir uma das duas: rotular os exemplos do F com `y_train` (GT) e desacoplar a Fase E da Fase A; ou corrigir CLAUDE.md, artigo e rodapé para "especialista = métrica estimada na Fase A externa". |
| B2 ✔ | `estilo_cinza.py:179-191` (`_rotular_linhas_e_barras`), `:304-315` (`_cinza_linha`), `:124` | Toda `Line2D` com mais de 2 pontos e sem rótulo vira "série k", e na P&B ganha traço e símbolo. As tampas (caplines) de `errorbar`/`bar(yerr=…, capsize=…)` e as barras repetidas com `label=''` caem nessa regra. Visto no smoke: `bloco1_graphic_08` `_gray` (legenda "série 1..4" e linha preta cruzando o topo das barras), `bloco1_graphic_16/17/18` `_gray`, `bloco1_graphic_19` `_gray` (18 entradas tapando o painel), `bloco1_graphic_21`, `bloco2_graphic_04` (já na `_color`, pois o eixo não tinha legenda). Reproduzido offline com a grade da completa: `bloco2_graphic_03` ganha 12 "série" e 8 linhas espúrias por painel, `bloco2_graphic_07` idem. Não aparece na completa de 08/10 porque o código é posterior. | Ignorar linhas com `get_linestyle() == 'None'` ou pertencentes a `ErrorbarContainer` em `ax.containers` nas três funções; não rotular barras com `label=''` quando o eixo já tem legenda; em `_cinza_legenda` refazer a legenda a partir dos handles da legenda original. Teste novo com `bar(yerr, capsize)` de 4+ pontos. |
| B3 | `dissertacao_mestrado.py:3641-4583` (laço modelo × seed) | O corpo do laço (Fase A/B/C, protocolo, vieses, Fase E) não tem `try/except` nem `finally`. Os únicos `try` do `main()` protegem o plot do SVM, os Blocos 3 e F e as visualizações finais. Uma exceção derruba o processo antes do PASSO 7: o buffer do `Tee` se perde (sem `log_execucao.txt`), os CSVs não são exportados e as interações ficam só no último checkpoint (gravado 1× por seed, l.3659). CLAUDE.md promete "exceção num bloco é registrada e a execução termina com exit 2", o que só vale para G1/G2/F. | Envolver cada bloco em `except Exception` → `falhas_fatais.append(...)` como no Bloco 3, e um `finally` que grave checkpoint + log parcial (reaproveitar `_abortar_por_creditos` sem o `SystemExit(3)`). |

## Importantes: resultados mostrados como coisa que não são

| # | Local | Problema | Correção |
|---|---|---|---|
| I1 ✔ | `metrics.py:191-194` (`compute_consistency_metrics`) | A "F1 simétrica" (média da F1 weighted nas duas direções) é igual à acurácia para rótulos binários (diferença máxima 2e-16 em 500 casos). Na completa a coluna F1 da T2 é idêntica à coluna Consistência nas 24 linhas. Afeta `f1_problema_b/c`, `bloco1_transferencia_bc.f1`, tabelas 06/07/08, painel "F1-Score" e caixas de `bloco1_graphic_10`; o eixo diz "F1-Macro" (`plots.py:376`). | Remover F1 de B/C ou usar uma direção fixa com `average="macro"`; corrigir o rótulo. |
| I2 | `_protocol_rows` (`dissertacao_mestrado.py:2684-2750`), `plot_fase_a_fewshot_curve`, tabelas 01/02/04/12, `_tab_vies` (tabela 14), tabela 06, `final_graphic_02` | Em zero-shot o nome da classe é arbitrário; `llm_accuracy_vs_true` no A zero-shot = 0,21 a 0,37 na completa (abaixo do acaso = correspondência invertida). A curva `bloco1_graphic_04` e as tabelas mostram "LLM vs real 32%" em k=0 saltando para 84 a 94% em k=4; boa parte do salto é a fixação nome↔lado pelos exemplos. O mesmo nos pares invertidos (gemini A/B 25,1% vs B/A 77,6%, complementares, lidos como "efeito da ordem") e em B/C (C 17,3%). | Para zero-shot e pares invertidos reportar `max(acc, 1-acc)` em coluna própria ou nota de rodapé explícita; ou canonicalizar os rótulos dos pares invertidos ao conceito do par base. |
| I3 | `run_external_problem_pipeline` + `bloco3_table_04/05`, `bloco23_graphic_01..04` | Mesmo problema com classes neutras A/B no Bloco 3: `G1_homem_mulher_classesAB` (gemini) tem acurácia vs real 0,10 a 0,19, ou seja, 83 a 90% de consistência com rótulos trocados. `bloco3_table_05` reporta Δ semântico−neutro "+35,7 / +45,7" como se nomes neutros destruíssem o desempenho. | Idem I2 (`max(acc, 1-acc)` ou \|κ\| com marcação de inversão), e dizer no rodapé. |
| I4 ✔ | `dissertacao_mestrado.py:1752-1770` e `1816-1832` (`run_oracle_validation`, Partes 1 e 3) | Os rótulos do "oráculo" sobre B e sobre as 3 configurações anisotrópicas são os rótulos geradores do `make_blobs`, não a regra do centróide mais próximo sob o W verdadeiro (só a Parte 2, E1/E2, usa `expert_classify`). CLAUDE.md e o rodapé da tabela 17 dizem o contrário. Em `x1_dom`/`x2_dom` os rótulos geradores coincidem 100% tanto com o W verdadeiro quanto com a euclidiana: W não é identificável, o Perceptron devolve [1,1] intocado (cosseno 0,75 a 0,87) e isso é apresentado como falha de recuperação. | Rotular as Partes 1 e 3 por `predict_with_metric(X, centros, W_verdadeiro)`; tornar as gaussianas menos separáveis (ou aproximar os centros) para que o teste de sanidade teste algo; corrigir CLAUDE.md e rodapé. |
| I5 | `plots.py:1722-1726` (`plot_oracle_w_recovery`), `:1856-1860` (`plot_oracle_transfer`) | `setdefault(key, {})[algo] = r` sobrescreve a cada seed: `bloco1_graphic_19` mostra só a última seed (7) e `bloco1_graphic_20` só a primeira (42), sem dizer; a tabela 17 mostra a média das 3. | Agregar por (problema, algoritmo) com média e mín–máx, ou declarar a seed no título. |
| I6 | `plots.py:1838-1895` + CLAUDE.md ("Também testa transferência…", passo 5, descrição de `bloco1_graphic_20`) | Não existe transferência: cada `expert_name` tem um único problema, o gráfico 20 são 6 painéis com UM tick cada, repetindo o painel `_fidelidade` do gráfico 19; `run_oracle_validation` nunca aplica o W recuperado em outra configuração. Título imprime o float cru ("W = [11.111111111111, 0.444…]"). | Implementar a transferência cruzada ou remover o gráfico 20 e o texto do CLAUDE.md. |
| I7 ✔ | `dissertacao_mestrado.py:4593-4596` (`_elipse_otima`) com `plots.py:2648-2657` | `z[y_hm == 1]` recebe a malha 200×200 e `y_hm` tem 100 posições: `IndexError`, engolido por `except Exception: pass`. A "fronteira elíptica" prometida para `bloco3_graphic_01/02` nunca foi desenhada (confirmado nos PNGs da completa e do smoke: legenda só "Classe 0/1"). Além disso a calibração por mediana da classe 1 dá 75% de acurácia; a melhor constante dá ~91%. | Calcular a constante uma vez sobre `X_hm` fora da função e escolher o limiar que maximiza a acurácia; trocar o `except: pass` por aviso no log. |
| I8 | `relatorios.py:889-912` (`summarize_cross_linearity`), `gerar_tabelas.py:709-735` (`tab_cross`) | `final_cross_linearity__<alias>.csv` não guarda `feature_names` nem variante de classe; `final_table_01` faz UMA média por G1/G2 misturando x1/x2, peso/altura, Mulher/Homem e A/B (incluindo as invertidas de I3). Na completa (gpt) G2 = 83,6, média de 95,7 (semântico) e 71,4 (x1/x2). A linha "meia-lua" vem do F (105 pontos, split) e não do D (150 pontos), embora a tabela diga "D". As linhas `A_linear` gravam `accuracy_vs_true=None` embora o protocolo tenha o valor. | Gravar `feature_names` e `problem_name` completo no CSV cruzado; filtrar a configuração base; usar as linhas zero-shot do D ou documentar. |
| I9 | `dissertacao_mestrado.py:3882-3893` (nomes semânticos de atributo) | A consistência B/C das variantes altura/peso e feature_1/feature_2 usa `predict_with_metric(X_b, centroids_feat, w_feat)` (centróides TRANSFERIDOS do A); a base e os outros vieses passam por `phase_consistency_test` com centróides LOCAIS. `bloco1_graphic_17` e `bloco1_table_15` comparam protocolos diferentes (gpt altura/peso 57,7% vs x1/x2 70,7%). As mesmas linhas gravam placeholders `euclidean_consistency_b/c = 0.0` e `diagonal_limitation_flag = 0`; o NNLS é calculado e só impresso. | Rotear por `_run_phase_abc_experiment` passando `nome_feature_0/1`; ou no mínimo `compute_centroids(X_b, y_llm_b_feat)` e NaN nas colunas não medidas. Recalcular offline as 12 linhas de atributo da completa. |
| I10 | `dissertacao_mestrado.py:1569-1572`, `1598-1627` (`phase_consistency_test`) | Os exemplos few-shot de B/C (H2) são rotulados com os centróides transferidos do A, mas a consistência é medida contra centróides locais: o LLM é ancorado numa fronteira e avaliado contra outra (docstring admite; rodapé da tabela 09 e CLAUDE.md não). O teste ablativo mostra `transferidos > locais` em todas as 7 ocorrências do log (69% vs 89% etc.): quando o LLM mistura os blobs (77/23), os centróides locais caem ambos perto da origem e a fronteira local vira artefato. `consistency_transferred` não é gravada em CSV. Afeta H1 (κ_B ≈ 0,35). | Persistir `consistency_transferred/kappa_transferred` no CSV; decidir com o orientador qual regra reportar (ou as duas); usar os mesmos centróides nas duas etapas de H2. |
| I11 | `dissertacao_mestrado.py:2915-2939` vs `gerar_tabelas_bloco1.py:265-272` | Duas regras de "melhor métrica do A": o runner (`marcar_melhor_metrica_a`, coluna do CSV, negrito da tabela 06, painel 3 de `bloco1_graphic_11`) inclui o zero-shot; as tabelas 03/07/08 (`melhor_fewshot`) só few-shot com desempate pelo menor k. Coincidem na completa, divergem se o zero-shot vencer ou em `--rapido`. | Uma única função (tabelas lerem `melhor_metrica_a` do CSV) e documentar no CLAUDE.md. |
| I12 | `dissertacao_mestrado.py:2497-2524` vs `3938-4040` | `run_complete_experiment` calcula `metrics_nnls_b/c` pareadas com os MESMOS rótulos do LLM e as descarta; o que vai para `bloco1_algorithm_comparison`, `bloco1_graphic_06` e `bloco1_table_05` é o bloco "COMPARAÇÃO DE ALGORITMOS", que recoleta B/C (200 chamadas/seed, 1.200 na completa) e mistura diferença de estimador com flip de T=0 (smoke seed 42: B Perc 83,0 vs NNLS 67,0 com rótulos diferentes). O rodapé não avisa. | Exportar as métricas pareadas de `detail` ou usar os mesmos `y_llm_b/c`; dizer no rodapé qual comparação é; explicitar no CLAUDE.md que a recoleta é a exceção intencional a `reps_para`. |
| I13 | `dissertacao_mestrado.py:2784` (`collapsed = len(unique) < 2`) | "Colapso" é só classe única. No D, gpt zero-shot deu 149/1 nas seeds 42 e 7, tratado como válido: fidelidade 66,7/71/75% e métrica vs real 63% entram nas médias zero-shot das tabelas 10/11/12 e no k=0 das curvas D de `bloco1_graphic_04`; a métrica de um rótulo 149/1 é artefato (centróide da classe 1 = 1 ponto). | Piso para a classe minoritária (ex. 5 pontos) virando `collapsed` ou flag própria, excluído das médias, ou anotado na figura e nas tabelas. |
| I14 | `relaxed_perceptron.py:371-387` | O Perceptron devolve W em "melhor esforço" (γ=0) na maioria das chamadas: 837 avisos no log da completa; na configuração base do A, 3 de 6 estimativas têm γ=0. `w_best_effort` é o w do fim da época de menor contagem (o comentário l.283-287 admite que a contagem não é a do w devolvido). Nada nas tabelas 01/02/13 nem em `bloco1_graphic_06` sinaliza. | Exportar `gamma` nas tabelas (já está no CSV) ou marcar γ=0; opcionalmente recontar violações do w devolvido. |
| I15 | `dissertacao_mestrado.py:2040-2060` (`select_examples_by_strategy`, "mixed") | `n_easy = n_per_class // 2` e o resto hard: com n=10 dá 4 fáceis e 6 difíceis, não 50/50 como docstring, CLAUDE.md e rodapés prometem (n=4, 20, 40 saem exatos). Afeta as linhas mixed 10-shot dos 3 especialistas, o viés de ordem com n=10 e os baselines mixed n=10. | Alternar o excedente entre classes ou documentar a composição real. |
| I16 | `dissertacao_mestrado.py:4163-4214` (laço da Fase E) | O zero-shot (prompt idêntico, sem especialista no prompt) é coletado 3×, uma por especialista: 1.800 chamadas redundantes por completa, e a "linha de base zero-shot" difere entre E1/E2/E3 por flip (gpt: `n_class_0_llm` com 2 valores para a mesma seed), comparada na tabela 04. | Coletar 1× por (modelo, seed) e recalcular as métricas por especialista. |
| I17 | `dissertacao_mestrado.py:3254-3266`, `3340-3345` (Fase E externa) | Sorteio SEM balanceamento por classe (ao contrário da Fase E interna e do protocolo): na completa, prompts few-shot com uma só classe em 40/120 (k=4) e 18/120 (k=10). O Perceptron baseline vira `None` (barras somem em `bloco23_graphic_04`); o rodapé "treinado nos mesmos n exemplos" fica falso. Os rodapés de `bloco2_table_10` e `bloco3_table_03` dizem "perceptron linear", mas o baseline é treinado em `augment_features` (R³/R⁴ quando a melhor métrica tem 3 ou 4 atributos). | k/2 por classe sobre `y_metric_train` com `RandomState(seed+rep)`; rodapé "Perceptron diagonal no mesmo espaço R^n da melhor métrica". |
| I18 | `dissertacao_mestrado.py:4662, 4740` (filtro `HM_SEMANTIC_PROBLEMS` em `final_graphic_07`) | A variante A/B do G1 (a única que rodou Fase E com gpt, pois o G1 semântico colapsou) é excluída dos mapas de rótulos: não existe nenhum `final_graphic_07` de Fase E do G1 com gpt. O colapso em si é esperado e está no log. | Incluir a variante A/B nos mapas (o nome já desambigua). |

## Importantes: figuras e tabelas com conteúdo diferente do título ou legenda

| # | Local | Problema | Correção |
|---|---|---|---|
| F1 | `plots.py:790-870` (`bloco1_graphic_05`) | Suptitle "entre Sementes e Repetições", mas em zero-shot há 1 coleta: os 8 pontos por semente são os 8 pares de nomes de classe (4 normais + 4 invertidos). Coluna `nomes` calculada e nunca usada. | Título "sementes e nomes de classe" ou restringir a A/B. |
| F2 | `dissertacao_mestrado.py:1374-1379` (`bloco1_graphic_09`) | Todas as 33 curvas por modelo têm o mesmo rótulo "Fase A — Problema A". | `label=f"seed {seed} {nome_0}/{nome_1} {variante}"` ou só a configuração base. |
| F3 | `plots.py:945-1007` vs `403-442` (`bloco1_graphic_15` e `16`) | Os dois chamam `_config_base(n_shot_comum=True)` e desenham o MESMO gráfico (painel `_consistencia` idêntico nos PNGs): 8 pares em ordem alfabética, médias misturando zero e 10-shot, sem emparelhar normal × invertido. | 16 só com `NOMES_CLASSES`; 15 com barras pareadas normal × invertido por regime. |
| F4 | `plots.py:1282-1290` (`bloco1_graphic_18` `_w_scatter`), `:1066-1073` (`17` `_pesos`), `:1569-1580` (`06` `_w_scatter`) | Usam W bruto do Perceptron, cuja magnitude depende de γ (0,0003 a 0,53 entre seeds da MESMA variante): os painéis mostram escala, não direção (todos os pontos na origem exceto um). | Plotar `w_direction` (unitário) ou `w_ratio`, como já fazem as tabelas. |
| F5 | `estilo_cinza.py:155-162` (`_rotular_scatters`) | Dois scatters sem rótulo viram "classe 0/1" mesmo quando não são classes: `final_graphic_06` painéis "Fase B/C: Consistência" (acerto/erro) e `bloco1_graphic_14` no ramo de colapso (pontos coloridos pelo rótulo real sob o título "CLASSE ÚNICA"). | Label explícito nesses scatters; nome neutro no fallback. |
| F6 | `plots.py:475-477` (`final_graphic_08`) | Barras B (lisa) e C (hachurada) sem label e sem legenda; com `garantir_legendas` vira "série 1..2N" na próxima execução com 2 modelos. | `label='Problema B'/'Problema C'`. |
| F7 | `plots.py:1097-1200` + `classical_baselines.py:47` (`bloco2_graphic_05`) | O k-NN leva k no nome ("k=3" com 4 exemplos, "k=5" com 10+) e vira 2 séries da mesma cor, 2 entradas de legenda e curva quebrada. F1 dos baselines é unidirecional, o do LLM é simetrizado (= acurácia, I1). | Coluna `family`; k no rodapé; `compute_consistency_metrics` nos baselines. |
| F8 | `plots.py:1897-1945` (`final_graphic_01` painel E) | Painel inferior vazio ("Sem dados LLM") embora existam rótulos zero-shot e do especialista. | Desenhar rótulos do especialista ou omitir o painel. |
| F9 | `plots.py:2998-3005` (`_draw_symbols_scheme`) | Painel "Métrica estimada" desenha centróides da própria predição, não os `boundary['centroids']` que traçaram a fronteira (no PNG, c0/c1 em posições diferentes entre os painéis 2 e 3). Em `bloco1_graphic_11` a fronteira e os círculos c0/c1 ficam fora da legenda; o título diz "rep 1" para rep=0. | Quando houver `boundary`, desenhar `boundary['centroids']`. |
| F10 | `plots.py:900-925` (`bloco1_graphic_03` `_erros_por_margem`) | Eixo x todo "0.0" (margem na escala de W ~1e-3 formatada com `.1f`), só 3 barras. | Normalizar W antes de `compute_metric_confidence` ou formatar com `.2g`. |
| F11 | `plots.py:508-548` (`bloco1_graphic_07`) | Média ± desvio por semente sobre k ∈ {0..40} de B/C sem dizer; "Robustez entre seeds" no CLAUDE.md; com `--bc-zero-shot` o desvio vira NaN. | Filtrar `n_shot==0` ou declarar. |
| F12 | `plots.py:1746-1760` (`bloco1_graphic_19` `_ratio`) | Razões verdadeiras 16 e 25 cortadas em 10 sem anotação; 0,0625 invisível; `inf` pulado em silêncio. | Escala log ou anotação "> 10". |
| F13 | `dissertacao_mestrado.py:3000-3012` (`bloco1_graphic_13`), `4100-4103` (`bloco1_graphic_14`) | No colapso, os painéis de métrica são omitidos sem anotação (figura com 2 painéis); o D não tem painel few-shot, justamente onde tem sinal. A fronteira tracejada do SVM é escolhida por acurácia vs real, não por fidelidade vs LLM. | Painel "LLM k-shot" como no A; anotação de colapso; critério `fidelity_perc_vs_llm`. |
| F14 | `plots.py:2841-2908` (`bloco23_graphic_03`), `:2767-2772` (`01`), `:2641-2705` (`bloco3_graphic_01/02`, `final_graphic_07`) | Suptitle sobrepõe os títulos; painéis G2 "kg/m" com eixos normalizados sem rótulo; legenda cobre as séries; 9 grupos ilegíveis; legendas "Classe 0/1" em vez de Mulher/Homem; `y_metric` guardado e nunca desenhado (painel da Fase E compara com o GT, não com o professor). | Ajustar layout; rotular eixos; Mulher/Homem; desenhar `y_metric`. |
| F15 | `estilo_cinza.py:306`, `:258-260`, `:188`, `_Mapa` | Séries com exatamente 2 pontos viram preto sólido sem símbolo (smoke: curvas idênticas em `final_graphic_06`, `bloco2_graphic_03/05`, `bloco23_graphic_01`); scatter `c=y` dividido replica o rótulo inteiro ("Concordam (113), classe 0" e "…, classe 1"); barra única sem rótulo vira "valor"; série "Hard" em cinza 0,8 no painel κ de `bloco2_graphic_03` `_gray` com legenda preta (a confirmar: `_Mapa` único por figura indexado por cor). `final_graphic_03` `_gray`: "LLM Classe 0" cinza-claro sobre `contourf` cinza; `final_graphic_05` `_gray`: Mediana e Média com o mesmo traço. | Identificar referências por `axhline/axvline`, não por contagem; só acrescentar ", classe k"; não criar legenda de barra única; mapa por eixo. |
| F16 | Tabelas: `gerar_tabelas.py:423-427` (02_04), `:460-475` (02_06), `:290-333` (01_14/15/16), `gerar_tabelas_bloco1.py:213-236` (01_13), `:186-211` (01_06), `gerar_tabelas.py:560-563` (02_09) | `bloco2_table_04` "média few-shot" é média por linha (random pesa 3×: 87,1 vs 85,7); slug `40shot` fixo com `n_max` dinâmico (no smoke diz 40 e a caption 4); tabelas 14/15/16 repetem Fid. A / LLM vs real A / w1/w2 nas linhas zero e 10-shot sem dizer que a Fase A é sempre zero-shot, e a 15 tem "x1/x2 10-shot" sem contraparte; tabela 13 promete "—" para colapso mas `pivot_table` descarta a linha; tabela 06 não filtra `n_features==2` (inofensivo hoje); `tab_F_faseA` tem `faltam` sempre vazio e coluna com formatos misturados. | Média das médias por (estratégia, n); slug sem número; rodapés; `dropna=False`. |
| F17 | `gerar_tabelas_bloco1.py:494-517` | Rodar `gerar_tabelas_bloco1.py` depois do `main()` sobrescreve `tabelas/todas_as_tabelas.tex/.docx` com só as 9 tabelas do Bloco 1, sem aviso; o CLAUDE.md lista os dois comandos. | Manter o módulo só como biblioteca ou gravar com outro nome. |
| F18 | `renomear_assets_legado.py:44` (`RE_PAINEL`) | Trata qualquer sufixo após o alias como painel; nos nomes de 09/10 a cor vem depois do alias, então o script "corrige" nomes certos (`…_boxplot__gpt4mini_color.png` → `…_boxplot_color__gpt4mini.png`). A próxima completa apareceria como "a renomear" e `--aplicar` quebraria a convenção. | Excluir `color/gray` ou aposentar o script. |

## Importantes: infraestrutura, parser e testes

| # | Local | Problema | Correção |
|---|---|---|---|
| P1 ✔ | `tests/test_regressao_execucao.py:30` | `EXEC_REF` aponta para `execucao_2026-07-04_19-46-25_completa`, que não existe (os valores dourados citam o CSV de 05/07). Os 4 testes são sempre pulados: a rede de segurança numérica está morta desde a mudança para `resultados/`. | Apontar para `execucao_2026-07-05_12-34-52_completa` e reconferir os valores. |
| P2 | `llm_parser.py:150-157` (camada 7) | `startswith(nome)` sem fronteira de palavra: "Both A and B are possible" → B válido; "Based on…" → B; "Ambiguous" → A. Contradiz a política das camadas 4/5/6 de não decidir quando as duas classes aparecem. Impacto nos dados atuais nulo (0 registros pela camada 7). | `re.match(nome + r"\b")` e não decidir se a outra classe aparece. |
| P3 | `llm_parser.py:84-92` (camada 3), `:45-62` (camada 0) | Camada 3 conta todos os inteiros, inclusive pedaços de coordenadas ("x1 = 0.3 … Answer: 1" com nomes 0/1 → "0", válido). Camada 0 usa a PRIMEIRA ocorrência do marcador e não limpa `**`/`\boxed{}`: Gemini CoT teve 126 retentativas em 1.590 (7,9%). Sem dano na completa (respostas de 1 token nos pares 0/1). | Último número / último marcador; limpar `*`, `$`, `\boxed{}`. |
| P4 | `dissertacao_mestrado.py:955-960`, `1194-1216`, `1041-1049`, `331-337` | Timeout de 90 s devolve string vazia e consome as 5 tentativas de FORMATO (fallback MD5 sem campo `error`; o docstring diz 90 s "após retries do SDK", mas o SDK pode levar 180 s). Falha definitiva grava `format_retries=5` e `prompt=""` (agrupa pontos falhos na mesma chave de flip). HTTP 400 sempre retentável (~26 min por ponto se persistente, 40 workers presos). `--apenas oraculo --rapido` exige chave e saldo (0,6) com 0 chamadas. | Timeout → exceção retentável; `format_retries=0` + prompt real; 400 só com mensagens transitórias; oráculo → 0. |
| P5 | `dissertacao_mestrado.py:3783, 3804, 3822, 3843, 4208` | Tags dos vieses são `k=-\|rep=0` sem nomes/variante/k (a auditoria por regime não separa zero de 10-shot nem o par); zero-shot da Fase E leva `regime=easy\|k=0`. | Tag com k e o fator; `regime=zero_shot` quando `n_shot==0`. |
| P6 | `dissertacao_mestrado.py:2268/2366` vs `4344-4357` vs `4420-4432`; `2528-2531` | `accuracy_expert_vs_gt`/`n_class_*_expert` com bases diferentes (150 pontos na Fase E, teste na diluição, misto no viés de ordem). `diagonal_limitation_flag` usa só o Problema B; C é ignorado embora exista no CSV. | Unificar (tudo sobre o teste); incluir C. |
| P7 | Lacunas de teste | Sem teste para: reduções do `--rapido` e criação da pasta/sufixo em `main()`; checkpoint atômico e restauração do `Tee` no abort; timeout → malformada; roundtrip `salvar_json_em_chunks` ↔ `load_interactions`; `estilo_cinza` com barras de erro (B2). | Adicionar. |

## Código morto e cálculos descartados

| Local | Item | Observação |
|---|---|---|
| `dissertacao_mestrado.py:1606` | `confidences_test` | nunca usado |
| `dissertacao_mestrado.py:3972, 3980` | `y_pred_b_alt`, `y_pred_c_alt` (e `_bl_*_alt`, `_Xt_*_alt`) | nunca usados |
| `dissertacao_mestrado.py:4253` | `examples_bl` | nunca usado |
| `dissertacao_mestrado.py:4488` | `gamma_r3` | nunca usado |
| `dissertacao_mestrado.py:2497-2525, 2581` | `nnls_detail` (`metrics_nnls_b/c`, `y_pred_nnls_b/c`), `baselines_b/c` | calculados em todo few-shot de B/C e nunca persistidos (ver I12) |
| `dissertacao_mestrado.py:3861-3869, 3897-3900` | `fid_feat_nnls`, `cons_b/c_feat_nnls` | só impressos (ver I9) |
| `dissertacao_mestrado.py:4455-4575` | `RUN_R3R4_EXPERIMENT` | fidelidades bit a bit iguais às linhas `zero_shot` do protocolo (36/36 na completa); só `problem_A_r3/r4.csv` é exclusivo; `bloco1_graphic_08` e `bloco1_table_04` mostram os mesmos números |
| `dissertacao_mestrado.py:1837-1899` | `run_oracle_meialua` | idêntico ao regime `ground_truth` do D (tabela 18 = linhas "teto" da 10; gráfico 21 = linha "teto" do 04); docstring l.1855 espera o contrário do que os dados mostram. Serve a `--apenas oraculo`; documentar a equivalência ou derivar do protocolo |
| `dissertacao_mestrado.py:238-241, 3646-3665, 4148, 4634-4637, 4718` | `scope="core"`, `EXTRA_SEEDS_CORE`, `is_seed_extra` | nenhum modelo usa; ramos nunca exercitados |
| `plots.py:805`, `:296-308` | coluna `nomes`; colunas `model/nomes/fidelidade` | calculadas e não usadas |
| `gerar_tabelas.py:560-563` | `faltam` em `tab_F_faseA` | sempre vazio |
| `tabelas.py:301` | `media_min_max` | só usada em teste; `fmt_stat` faz o mesmo |
| `llm_client.py:67` | `get_client` | só usada em `tests/test_llm_client.py` |
| `execucao_io.py:226` | `com_cor` | `salvar_figura` não a usa (repete a lógica); só monta o nome impresso no log |
| `corrigir_mapeamento_homem_mulher.py` | script inteiro | sai por `SystemExit` em qualquer execução desde 08/10 (só `problem_name == "homem_mulher"` legado); CLAUDE.md e README o apresentam como rotina |
| pyflakes | 132 f-strings sem placeholder; imports `d_W`, `MODEL_ALIAS`, `_model_slug`, `asset_variant` | os imports são reexports usados por testes (noqa); as f-strings são cosméticas |

## Comentários, docstrings e documentação desatualizados

- **Numeração antiga de assets em comentários** (a convenção de 09/10 renumerou, e os números antigos hoje apontam para OUTRAS figuras): `dissertacao_mestrado.py:536, 564, 2944-2947, 3005/3014/3020/3024/3028/3034 ("só a versão _gray" é falso: sai `_color` e `_gray`), 3022-3023, 3098-3104, 4099, 4658, 4808, 4812, 4921/4929`; `plots.py:62-64, 794-795, 1636-1644, 2672 (docstring "final_08")`; `relatorios.py:45`; `relaxed_perceptron.py:92, 406`; `execucao_io.py:209, 212, 246-249`; `regerar_figuras_simbolos.py:2`; rodapés "Tabela T1"/"Tabela T3" em `gerar_tabelas_bloco1.py:181` e `gerar_tabelas.py:258`; cabeçalho `gerar_tabelas.py:22` (regra da T7) e `:33-35` ("zero e 10-shot" para a tabela de atributos, que só roda zero-shot); labels LaTeX `tab:T1/T4/T6`.
- **CLAUDE.md**: passos 4c, 9, 10, 11 do fluxo e a tabela de flags citam `bloco1_02/11`, `bloco1_03/04/04b`; "Também testa transferência" (I6); "rótulos do oráculo vêm da regra do centróide mais próximo" (I4); "Mixed: 50% easy + 50% hard" (I15); H4 "easy supera hard consistentemente" (na completa: empate em gpt E2, hard > easy em 40-shot, random ≥ easy em todos os pares); "H4 só o especialista E1" (a tabela 05 reporta E1/E2/E3); `BIAS_N_SHOTS` "nomes de classe, variantes de prompt" (também inversão de ordem; atributos só zero-shot não consta); "fallback resiliente" (l.451) não existe; `gamma_init=0.1` e `max_iterations=50` do Perceptron não estão em `PERCEPTRON_PARAMS` nem na tabela; `MAX_RETRIES × retries do SDK` (até 15 tentativas HTTP) não documentado; `bloco1_graphic_07` "Robustez entre seeds" (F11); `bloco1_graphic_13` descrito como "métrica R² | métrica R⁴" com painéis `_metrica_zero_shot/_metrica_r4_zero_shot` (ok, só alinhar o texto).
- **README.md**: manda `pytest` (não instalado); faltam `gerar_tabelas.py`, `tabelas.py`, `estilo_cinza.py`; `regerar_figuras_simbolos.py` "regera bloco1_15/16/18"; `gerar_tabelas_bloco1.py` "T1–T5"; `--ignorar-saldo` e a guarda de saldo ausentes; sem a convenção `_color/_gray`.
- **glossario.md:108** `MAX_CONCURRENCY` 30 (código: 40); `:100` flip "~4%" é de 07/2026 (completa de 10/2026: 1,6%). `.claude/skills/revisor-resultados/SKILL.md:82` cita `final_09_model_comparison.png`.
- `phase_consistency_test` docstring (l.1543-1546) diz que `--bc-zero-shot` é o protocolo atual; o padrão é a grade completa e a completa de 08/10 rodou few-shot em B/C.
- Nomes `x2_dom`/`x1_dom` em `ORACLE_ANISO_CONFIGS` descrevem a dimensão RUIDOSA, o W verdadeiro é o oposto (tabela 17 lê "x2_dom [4,00; 0,25]"). `expert_name` "euclidiana" para o Oráculo sobre B gera painel `_euclidiana` enquanto a tabela 17 usa "Problema 1/2/3": dois sistemas de nome.
- Ordem (altura, peso) no Bloco 1 e (peso, altura) no Bloco 3; cada bloco é coerente internamente.
- O limiar κ > 0,4 de H1 não é avaliado nem impresso em lugar nenhum do log (só as faixas de Landis & Koch nas tabelas 08/09).

## Verificado e OK (não precisa de nova auditoria)

- **Geração dos dados**: A/B/C com os centros, std 1,2 e seeds documentados; rotações de B e C com mesma magnitude e sentidos opostos; D/F com `make_moons(150, 0.15, seed)`; E com `EXPERT_CENTROIDS`/`EXPERT_W`; G com `peso_altura_kg_m.csv` igual ao mapa linear da base normalizada, mesma ordem e classe, classe 1 = Homem em todos os pontos (prompts, tabelas, scripts); split 70/30 idêntico em G1/G2 e entre variantes.
- **Protocolo da Fase A** (A e D): zero-shot reaproveita o cache (W, γ e fidelidade idênticos entre CSVs); k/2 por classe confirmado via `example_indices`; `seed + rep`; exemplos fora da avaliação; teto GT com 0 chamadas; LLM vê só x1,x2 e a augmentação entra só na estimação; exemplos few-shot em R³/R⁴ mostram os extras com os mesmos nomes do ponto de teste; 1.008 linhas = grade esperada.
- **Estimadores**: `relaxed_perceptron.py` (restrição, gradiente, projeção w ≥ 0, bisseção monotônica, RNG local) e `least_squares_inverse.py` (A = d²_errado − d²_certo, `scipy.nnls`) fazem o que as docstrings dizem; `metrics.py` (d_W, centróides, R3/R4, predição, margem) correto; NNLS do bloco de comparação = NNLS do protocolo.
- **Transferência B/C**: `evaluate_transfer_bc` reproduz `consistency_problem_b/c` nas 6 combinações; `marcar_melhor_metrica_a` marca a linha certa; `select_confident_examples` balanceado, rótulos da métrica, exemplos fora do teste; κ e acurácia simétricos; `--bc-zero-shot` força `[0]` e 1 coleta.
- **Fase E interna**: especialista por centróide mais próximo; easy/hard pela margem do ESPECIALISTA; balanceamento n/2; random `seed + rep`; easy/hard/mixed determinísticos com 1 coleta; teste = pontos não usados; baselines nos mesmos índices e mesmo teste; diluição com 4 hard fixos idênticos entre N; ordem com o mesmo conjunto reordenado; 504/432/36/72 linhas nos CSVs; figuras 03/04/05 e dashboard filtram E1.
- **Pipelines externos**: `X_display` só na exibição; zero-shot externo vê só x1,x2; sorteio igual entre variantes (mesmos 4 pontos nos prompts reais); conversão inversa da elipse correta; exceção por bloco registrada com exit 2.
- **Coleta e infra**: semáforo + freio global sem deadlock; backoff exponencial com jitter; 402 nunca retentado e abortando com checkpoint; retentativa de formato reenvia o mesmo estímulo; `raw_response` = resposta do rótulo (0 divergências em 157.854 registros); 100% dos registros com `tag` no formato; `model_resolved`/`inference_provider` iguais ao pin; contagens de chamadas por bloco batem com as constantes; `Tee`, chunking de 10 MiB e checkpoint atômico corretos; `--apenas`/`--rapido` reduzem o que o CLAUDE.md diz; parser reproduz 100% dos rótulos e cobre todos os pares de nomes (incluindo invertidos, Mulher/Homem e o artigo "a").
- **Vieses**: prompts reais batem com o código para as 4 variantes, 8 pares de nomes e 2 pares de atributo; cache da Fase A chaveado por configuração; filtros de configuração base coerentes em `plots._config_base`, `gerar_tabelas._base_abc` e `relatorios`.
- **Fechamento**: 300+ arquivos do smoke seguem a convenção `<bloco>_<graphic|table>_<NN>_…[_<painel>][__<alias>]_<color|gray>`; `com_cor` só em PNG; figuras de símbolos desenhadas 2× sem `para_cinza`; 36 (smoke) e 67 (completa) tabelas com caption e rodapé, colunas consistentes, `todas_as_tabelas.tex` compila; numeração NN segue a ordem do CLAUDE.md; `final_graphic_06` bate com os CSVs; `final_graphic_08` usa a configuração base; "sem legenda" = 0 no log.
- **Suspeita descartada** ✔: a fidelidade zero-shot 81,3% igual nos dois modelos em `final_table_02` é coincidência (122/150 em média), não cruzamento de cache: as fidelidades por seed, as partições e os W diferem (gpt 0,853/0,780/0,807; gemini 0,833/0,787/0,820).

## Métricas de avaliação: conformidade com a seção IV.3 do orientador

Fonte: `reunioes_orientador/reuniao_2026_10_02/doc revisados orientador/Metodologia[90].pdf`, seção "IV.3 – Métricas de Avaliação". O sistema deve calcular, gravar, plotar e tabular **apenas** estas cinco métricas, com estes nomes e estes conceitos:

| Métrica (nome exato) | Definição do PDF | Onde o PDF manda aplicar |
|---|---|---|
| **Fidelidade** | F = (1/n) Σ 1[y_W(v_i) = y_LLM(v_i)]: fração em que o centróide mais próximo sob o w estimado concorda com o LLM. Mede o quanto a métrica reproduz o LLM, não sua correção. | Problemas A e D |
| **Acurácia** | A = (1/n) Σ 1[y_W = y_GT] ou (1/n) Σ 1[y_LLM = y_GT]: concordância com o ground truth. | Teste-oráculo e Aprendizado em Contexto com o LLM como aprendiz (Bloco 2) |
| **Consistência** | Idem à fidelidade, com y_W calculado em relação aos centróides dos NOVOS problemas; só o vetor w é transferido do A. | Problemas B e C |
| **Cohen's k** | k = (P0 − Pe)/(1 − Pe), Pe = Σ_k p_k p'_k (p_k marginal da métrica, p'_k do LLM); faixas de Landis e Koch 1977: <0,2 fraca, 0,2–0,4 razoável, 0,4–0,6 moderada, 0,6–0,8 substancial, >0,8 quase perfeita; negativo = pior que o acaso. | Junto com as anteriores |
| **F1-Macro** | Por classe k: P_k = VP/(VP+FP), R_k = VP/(VP+FN), F1_k = 2 P_k R_k/(P_k+R_k); média entre as classes; de 0 a 1. | Junto com as anteriores |

IV.5 acrescenta: "são calculadas a fidelidade, consistência, coeficiente Kappa k e o F1-Macro entre a métrica estimada w e o LLM nos problemas A-D"; o teste-oráculo avalia "as respectivas acurácias"; a comparação Perceptron × NNLS usa similaridade do cosseno (comparação de estimadores, não métrica de avaliação; `w_cosine_sim`, `w_ratio`, `w_direction`, `gamma` ficam).

### Situação de cada métrica no código

| Métrica | Fórmula | Nome exibido | Escopo de aplicação | Veredito |
|---|---|---|---|---|
| Fidelidade | ✔ correta (`acc(y_llm, y_W)`) em `phase_a_learn_metric`, `_protocol_rows`, `evaluate_transfer_bc` (`fidelity_source_mean`), R3/R4 | ✔ "Fidelidade" em figuras, tabelas e log | ✘ usada também no oráculo (coluna `fidelity` e painel `_fidelidade` do gráfico 19, que são acurácia), em F/G1/G2 (`fidelity_perc_vs_llm`, tabelas 02_09, 03_01, 03_04, final_01, gráfico 23_02) e, dentro do próprio A/D, a coluna `fidelity_vs_llm` no regime `ground_truth` guarda a acurácia do teto (tabela 01 mostra 98,0 em "Fid. vs LLM"). O inverso também ocorre: `accuracy_gt_metric_vs_llm_zero` é uma fidelidade com nome "accuracy"; `bloco1_r3r4_comparison` grava a fidelidade na coluna `fidelidade` vinda de chaves internas `'accuracy'`. | Nome certo, escopo errado em 4 lugares |
| Acurácia | ✔ correta (`acc(y_GT, ·)`) | ✔ "Acurácia" no oráculo, no teto do D e nos externos; ✘ no Bloco 2 a métrica principal aparece como "Concordância LLM vs. Especialista" (figuras 02_03/04/05/06/07, log, `print_phase_e_analysis`) e como "Acurácia vs especialista" nas tabelas; na Fase E externa a coluna chama-se `accuracy_llm_vs_metric` e o gráfico 23_01 usa o nome cru da coluna como eixo | ✘ calculada e exibida fora do escopo em A, B, C, D e Fase A externa: `llm_accuracy_problem_a/b/c`, `metric_accuracy_problem_b/c`, `llm_accuracy_vs_true`, `accuracy_metric_vs_true_eval/full`, `accuracy_perc/nnls_vs_true`; colunas "LLM vs real"/"Métr. vs real" nas tabelas 01_01-04, 06, 10-12, 14-16, 02_09, 03_01/04/05, final_01/02; séries "acurácia ... vs rótulo real" do gráfico 01_04; critério de eleição da melhor métrica externa. No Bloco 2 há ainda `accuracy_llm_vs_gt` e `accuracy_expert_vs_gt` ("Especialista vs real"), que o PDF não prevê. | Conceito certo; nome errado no Bloco 2; escopo muito além do PDF |
| Consistência | ✔ correta no fluxo principal: `phase_consistency_test`, NNLS pareado, `evaluate_transfer_bc`, `_plot_protocol_symbols` usam w de A e centróides dos rótulos do LLM em B/C | ✔ "Consistência" | ✘ violada em três pontos: (i) experimento de nomes de atributo ([dissertacao_mestrado.py:3880-3899](src/dissertacao_mestrado.py#L3880-L3899)) usa `centroids_feat` do Problema A, e isso vai ao CSV, ao gráfico 01_17 e à tabela 01_15 como "Consist. B/C"; (ii) fallback no colapso do LLM em B/C (`phase_consistency_test` l.1598-1603 e `evaluate_transfer_bc` l.2900) usa centróides de A em vez de NaN; (iii) os exemplos few-shot de B/C (H2) e os baselines são rotulados por w_A com centróides transferidos de A (l.1491-1494, 1646), enquanto a consistência medida usa centróides locais (ver I10). A "consistência transferida" ablativa (l.1612-1631) só vai ao log, mas usa o nome "Consistência". | Fórmula certa no principal; violada nos vieses de atributo, no colapso e na rotulagem dos exemplos |
| Cohen's k | ✔ `cohen_kappa_score(y_metric, y_llm)` implementa exatamente a fórmula do PDF (verificado: exemplo manual = sklearn; simétrico, diferença 0 em 2000 pares); primeiro argumento = métrica (p_k), segundo = LLM (p'_k) | ✔ "Kappa de Cohen"/"κ" na maioria; ✘ só "Kappa", "Kappa B/C", "kappa=" em `plots.py:498, 1455, 1606, 2410`, `relatorios.py:272, 532, 641-642`, `dissertacao_mestrado.py:1663, 4035-4040, 4420` | ✔ | Fórmula e faixas certas; padronizar o nome |
| F1-Macro | ✘ **três definições, nenhuma macro**: (i) [metrics.py:191-194](src/metrics.py#L191-L194) `average='weighted'` simetrizado, que no binário é igual à acurácia (diferença 2e-16; coluna F1 = Consistência em todas as linhas das tabelas 01_06/07/08 e 02_07); (ii) [classical_baselines.py:154](src/classical_baselines.py#L154) `weighted` numa direção (diferença até 0,17 vs acurácia); (iii) [dissertacao_mestrado.py:3334](src/dissertacao_mestrado.py#L3334) `f1_score` padrão `binary` (classe positiva = 1) na Fase E externa (diferença até 0,93). O gráfico 02_05 mistura (i) e (ii) no mesmo painel. O F1-Macro do PDF difere da acurácia em até 0,41 nos mesmos pares e já é simétrico, portanto não precisa de simetrização | ✘ "F1-Score", "F1", "F1 (%)", "F1 média" em figuras, tabelas e log; "F1-Macro" só no boxplot do gráfico 01_10 (`plots.py:375`), onde é falso | ✔ | **Conceito errado em todo o sistema** |

Faixas de κ (`tabelas.py:321-333`, `faixa_kappa`): batem com o PDF; único detalhe é κ = 0,8 exato cair em "quase perfeita" (PDF escreve ">0,8"). Métricas de processo da auditoria (`taxa_malformadas`, `taxa_flip_T0`, final_table_03/04) não são métricas de avaliação e não conflitam com IV.3.

### Métricas que não existem no PDF e chegam às saídas

| Item | Onde aparece | O que fazer |
|---|---|---|
| Consistência euclidiana e `diagonal_limitation_flag` | `resultados.py:73-75`; `dissertacao_mestrado.py:1638-1642, 2526-2530, 2891, 4012-4014`; CSVs `euclidean_consistency_b/c` (phases_abc, algorithm_comparison) e `euclidean_consistency` (transferencia_bc); log "Consistência da linha de base Euclidiana" e "TESTE: Métrica estimada vs. Euclidiana" (`relatorios.py:684-699`); glossário l.118 | Remover das saídas (a linha "Euclidiana (razão=1)" do gráfico 01_05 é referência geométrica e pode ficar) |
| Precision e recall | `metrics.py:195-202` (só internos); verbetes no glossário | Remover de `ConsistencyMetrics` e do glossário |
| "Concordância", "Concordâncias/Discordâncias", "Acerto/Erro", "Taxa de erro", "Taxa de Discordância" como nomes de métrica | log `dissertacao_mestrado.py:1677, 2339`; `plots.py:906-909, 922-923` (gráfico 01_03), `1985-1988` (final_03 "Acerto/Erro (x%)"), `2170` (final_02 "x% concordância"), `2253-2254` (final_05 "Taxa de Erro"); ylabel das figuras 02_03/04/05/06/07; `relatorios.py:77-88, 383` | Renomear para Fidelidade (A/D), Consistência (B/C) ou Acurácia (Bloco 2); "taxa de erro" vira "1 − fidelidade por faixa de margem" ou sai |
| Contagens `n_agreements`, `n_disagreements*`, `n_errors_*`, "Erros (de 150)", "(erros)" | `resultados.py:68-69, 115`; CSVs de todos os blocos; tabelas 01_18, 02_09, 03_01/02 | Manter só como contagem auxiliar explícita ou remover dos CSVs e cabeçalhos |
| `accuracy_expert_vs_gt` ("Especialista vs real", "Acurácia do Especialista vs. rótulo real") e `accuracy_llm_vs_gt` no Problema E | `resultados.py:107-108`; tabela 02_04; log 2281, 2335-2336; `relatorios.py:380` | Remover das saídas ou confirmar com o orientador que quer a acurácia literal vs GT além da vs especialista |
| `accuracy_perceptron_baseline_vs_true`, baselines k-NN/LR/SVM | CSVs externos e `bloco2_classical_baselines`; chave `accuracy_vs_expert` reutilizada em B/C few-shot onde a referência é a métrica (`dissertacao_mestrado.py:1644-1668`) | São baselines, não métricas novas; rotular com a métrica do PDF que medem ("Acurácia" vs referência) e corrigir a chave em B/C |
| Matrizes de confusão (final_02, painel resumo) | instrumento visual | Podem ficar como instrumento; não chamar de métrica |

### Correções propostas (métricas)

| # | Local | O que está | O que deve ficar | Saídas que mudam |
|---|---|---|---|---|
| M1 | `metrics.py:191-202` | F1 `weighted` simetrizado + precision + recall | `f1_score(y_metric, y_llm, average='macro', zero_division=0)`, uma chamada; remover precision/recall (e `confusion_matrix` se ninguém usar) | `f1_problem_b/c`, `f1` (transferencia_bc), `f1_llm_vs_expert` (phase_e, dilution, example_order); gráficos 01_10, 02_03/05/06/07, final_06; tabelas 01_06/07/08, 02_07; `tests/test_metrics.py:96` |
| M2 | `classical_baselines.py:154`; `dissertacao_mestrado.py:3334` | `weighted` numa direção; `binary` | `average='macro'` (ou chamar `compute_consistency_metrics`) | `f1_vs_expert` (classical_baselines), `f1_llm_vs_metric` (external_phase_e); gráfico 02_05 |
| M3 | `plots.py:354-355, 593, 1147, 1352, 2410`; `gerar_tabelas.py:502`; `gerar_tabelas_bloco1.py:208, 382, 415`; log `dissertacao_mestrado.py:1676, 2335`; glossário l.23 | "F1-Score", "F1", "F1 (%)", "F1 média" | "F1-Macro" em todo lugar (o rótulo de `plots.py:375` só fica verdadeiro depois de M1) | rótulos |
| M4 | `plots.py:498, 1455, 1606, 2410`; `relatorios.py:272, 532, 641-642`; `dissertacao_mestrado.py:1663, 4035-4040, 4420` | "Kappa", "Kappa B/C", "kappa=" | "Cohen's κ" (ou "κ de Cohen") | rótulos |
| M5 | `dissertacao_mestrado.py:3880-3899` (nomes de atributo) | consistência com `centroids_feat` do A | `compute_centroids(X_b, y_llm_b_feat)` / `(X_c, y_llm_c_feat)`, ou passar por `phase_consistency_test` como a comparação de algoritmos | linhas de atributo em `bloco1_phases_abc`; gráfico 01_17; tabela 01_15 (recalculável offline a partir dos `llm_interactions`) |
| M6 | `dissertacao_mestrado.py:1598-1603, 2900` | colapso em B/C usa centróides de A | NaN com `collapsed`, como na Fase A | só linhas de colapso |
| M7 | `dissertacao_mestrado.py:1491-1494, 1646` | exemplos few-shot de B/C rotulados com centróides de A | decidir: rotular pela mesma regra da consistência (impossível antes da coleta, os rótulos locais não existem) ou declarar explicitamente nas tabelas 01_09/14/15/16 e no texto que os exemplos usam "w_A com os centróides de A" e só a consistência usa centróides locais (ver I10) | tabela 01_09 e linhas 10-shot das 14/15/16 se a regra mudar |
| M8 | `dissertacao_mestrado.py:1958` (`_append_oracle_result`); `gerar_tabelas.py:336, 348`; `plots.py:1800-1810, 1833, 1860-1870`; CLAUDE.md (descrição do gráfico 19) | coluna `fidelity` e painel `_fidelidade` para uma acurácia | coluna `accuracy`, painel `_acuracia` | `bloco1_oracle_validation`, gráfico 01_19, tabela 01_17 |
| M9 | `dissertacao_mestrado.py:2718, 2724` (`_protocol_rows`); `plots.py:3113-3116`; `gerar_tabelas_bloco1.py:124-135`; rodapés T1/T3 | regime `ground_truth` grava a acurácia do teto em `fidelity_vs_llm`; `accuracy_gt_metric_vs_llm_zero` é fidelidade | NaN em `fidelity_vs_llm` no teto e coluna própria `accuracy_ceiling_vs_true` (ou usar `accuracy_metric_vs_true_full`); renomear para `fidelity_gt_metric_vs_llm_zero` | `bloco1_fase_a_protocolo`, gráfico 01_04, tabelas 01_01/10/11/12 |
| M10 | acurácia em A/B/C/D e Fase A externa (ver lista acima) | calculada e exibida fora do escopo do PDF | decisão com o orientador: ou remover das saídas (colunas `llm_accuracy_*`, `metric_accuracy_*`, `accuracy_metric_vs_true_*`, `accuracy_perc/nnls_vs_true`; cabeçalhos "LLM vs real"/"Métr. vs real"; série de acurácia do gráfico 01_04; painel "LLM vs rótulo real" de final_02), ou o PDF passa a admitir acurácia como contexto nos Blocos 1 e 3, sempre com o rótulo completo "Acurácia (vs rótulo real)" e nunca numa coluna chamada "Fid." | muitas tabelas e CSVs (lista no inventário) |
| M11 | fidelidade em F/G1/G2 (`dissertacao_mestrado.py:3081-3082`; CSV `bloco23_external_phase_a`; `final_cross_linearity`; tabelas 02_09, 03_01, 03_04, final_01; gráfico 23_02) | "Fidelidade" fora de A/D | decisão com o orientador: estender a definição a F e G no PDF (mesmo conceito: métrica reproduz o LLM; recomendado) ou tirar essas colunas | nenhuma fórmula muda |
| M12 | Bloco 2: `plots.py:590, 690, 1147, 1350, 1446-1447, 2456`; `relatorios.py:272, 383, 812`; `dissertacao_mestrado.py:2332`; tabela 02_04 (`gerar_tabelas.py:433-439`) | "Concordância LLM vs. Especialista", "Acc média"; `accuracy_expert_vs_gt`, `accuracy_llm_vs_gt` nas saídas | "Acurácia (LLM vs especialista)" em todo lugar (coluna `accuracy_llm_vs_expert` já está certa); remover ou justificar as duas acurácias vs GT | gráficos 02_03-07, tabelas 02_01-08, log |
| M13 | Fase E externa: `dissertacao_mestrado.py:3330-3347`; `plots.py:2767-2768`; rodapé `gerar_tabelas.py:586-588`; CLAUDE.md l.23 e seção Fase E | `accuracy_llm_vs_metric`/`kappa_llm_vs_metric`/`f1_llm_vs_metric`; eixo com nome cru da coluna; rodapé "especialista do F é o rótulo real" | renomear para `*_llm_vs_expert` (o especialista de F/G é o rótulo que gerou os exemplos); rótulos "Acurácia (LLM vs especialista)" e "Acurácia (LLM vs rótulo real)"; alinhar rodapé e CLAUDE.md com a decisão de B1 | `bloco23_external_phase_e`, gráficos 23_01/04, tabelas 02_10, 03_02/03, `corrigir_mapeamento_homem_mulher.py:152-157` |
| M14 | `dissertacao_mestrado.py:1644-1668`; `classical_baselines.py:152-155` | chaves `*_vs_expert` reutilizadas em B/C onde a referência é a métrica | chaves genéricas (`*_vs_reference`) e rótulo "Consistência vs métrica" no log de B/C | log; `bloco2_classical_baselines` (nome já correto no Bloco 2) |
| M15 | consistência euclidiana e flag (lista acima) | nas saídas | remover (`resultados.py:73-75`; `dissertacao_mestrado.py:1638-1642, 2526-2530, 2891, 4012-4014, 5117-5119, 5180-5182`; `relatorios.py:684-699`; `tests/test_fase_a_protocolo.py:190`) | colunas somem de 3 CSVs; log |
| M16 | "Concordância/Acerto/Erro/Taxa de erro" e contagens (lista acima) | nomes paralelos para as mesmas métricas | renomear para a métrica do PDF correspondente; contagens só como auxiliares explícitas ou removidas | gráficos 01_03, final_02/03/05, 02_03-07; log; CSVs |
| M17 | `dissertacao_mestrado.py:4518-4523, 4563-4567, 5221-5258`; `relatorios.py:872-881` | chaves internas `'accuracy'`/`'accuracy_nnls'` e coluna `fidelidade` no R3/R4 | `fidelity` e `fidelity_vs_llm` | `bloco1_r3r4_comparison` |
| M18 | `tabelas.py:321-333` | κ = 0,8 em "quase perfeita" | opcional: "substancial" para seguir ">0,8" ao pé da letra; citar o PDF no docstring | tabelas 01_08/09 (só no limite exato) |
| M19 | CLAUDE.md (H2 "concordância LLM-métrica" → "consistência"; descrição do gráfico 01_10 "Kappa, F1" → "Cohen's κ, F1-Macro"; gráfico 19 `_fidelidade`; linha 23 e Fase E sobre o F); docstring H3 (`dissertacao_mestrado.py:99-100`); `glossario.md:21-25, 32-33, 117-118` | definições parciais; verbetes Precisão, Recall, "Métrica Euclidiana é o baseline" | reescrever as cinco entradas com as definições e escopos do PDF; remover os verbetes fora do PDF | documentação |

Resumo: Fidelidade, Consistência (no fluxo principal) e Cohen's k têm a fórmula certa; F1 está errada em todo o sistema (nenhuma das três variantes é macro, e a principal é idêntica à acurácia); a Consistência é violada no experimento de nomes de atributo, nos fallbacks de colapso e na rotulagem dos exemplos few-shot de B/C; Acurácia é usada fora do escopo em A/B/C/D e na Fase A externa, e Fidelidade fora do escopo no oráculo, na linha do teto e em F/G; quatro famílias de métrica que o PDF não tem chegam às saídas (consistência euclidiana e flag, acurácia do especialista vs GT, "concordância"/"taxa de erro"/contagens, precision/recall internos). As decisões M7, M10 e M11 dependem do orientador; as demais são mecânicas e recalculáveis offline a partir dos `llm_interactions` da completa de 08/10.

## Ordem sugerida

1. B1, B2, B3 e P1 antes de qualquer nova execução completa (B1 é decisão com o orientador: GT ou métrica como especialista do F).
2. M1 a M4 (F1-Macro e nomes) e M5, M6, M8, M9, M12, M13, M15, M16, M17: mecânicos, 0 chamadas, recalculáveis offline; M7, M10, M11 e M18 dependem do orientador.
3. I1 a I18 (corretude do que vai para o trabalho); vários são recalculáveis offline a partir dos CSVs e `llm_interactions` da completa de 08/10 (I1, I2, I3, I8, I9, I11), outros exigem recoleta (I16, I17) ou só mudam a apresentação (I5, I6, I13, I14).
4. F1 a F18 junto com `regerar_figuras_simbolos.py` e `gerar_tabelas.py` (0 chamadas).
5. Código morto, comentários e documentação (CLAUDE.md, README, glossário, skill), incluindo M19.

## Auditoria por problema dos nomes e comportamentos (13/10/2026, noite)

Sete auditores somente leitura (A e vieses; B e C; D; E1/E2/E3; F; G1/G2; oráculo, fechamento e
infraestrutura), com o briefing comum e `passos_por_problema.md`/`renomeacao_passos_2026-10-13.md` como
fonte de verdade, sobre o código e o smoke `execucao_2026-10-13_18-24-11_smoke`. Tudo o que segue foi
corrigido no mesmo dia; validação: 361 testes OK, pyflakes limpo, `--apenas oraculo --ignorar-saldo`
(`execucao_2026-10-13_21-24-18_apenas-oraculo`, exit 0, 18 linhas no CSV do oráculo, 0 "sem legenda"),
`gerar_tabelas.py` sobre o smoke (36 tabelas).

### Graves
- **Oráculo (main):** `resultados_oraculo = oraculo_passos_1_a_3(...); resultados_oraculo.extend(resultados_oraculo)`
  sobrescrevia o acumulador a cada seed e duplicava as linhas (smoke: 12 linhas em vez de 6; completa: só a
  última seed, em dobro). Corrigido (`linhas_oraculo`), teste `TestOraculoAcumulaSementes` lê a última
  `_apenas-oraculo` e exige 3 seeds × 3 problemas × 2 algoritmos.
- **Migrador:** `migrar_tag` quebrava (`int('-')`) nas 36 mil tags de vieses da completa de 08/10/2026
  (`k=-`); agora infere k do prompt (`k_prompt`, nº de linhas `-> classe` no bloco `Examples:`) e, sem prompt,
  grava `passo=-|k=-`. Também acrescenta `especialista_origem=metrica` a `bloco23_fg_aprendiz` de execuções
  anteriores a 09/10/2026 e renomeia `*_llm_vs_metric → *_llm_vs_expert` (o leitor do F é estrito).

### Médias
- `colapsou` (classe única OU minoritária < 5) agora vive em `protocolo.py` e é a regra em `plots.py`:
  `bloco1_graphic_14` (anota n0/n1 e não ajusta SVM no quase-colapso), `_centroides_locais` de B/C e a
  guarda do painel "métrica eleita" da figura 11. Em B/C colapsados `bc_passos_2_a_4` devolve
  `y_metric_test=None` (antes um vetor de -1 que a matriz de confusão e o mapa de acertos engoliam).
- `bloco1_table_12_AD_por_passo` exclui a acurácia do LLM colapsado, marca † e traz NOTA_COLAPSO/NOTA_GAMMA.
- `final_table_03` nunca preenchia Snapshot/Provedor (lia dicts; a auditoria grava listas). `final_table_01`
  explica o "—" do colapso no rodapé.
- Problema F: os exemplos do Passo 3 e o Perceptron do Passo 4 ficam SEMPRE em R² (antes herdavam R³/R⁴ da
  melhor métrica zero-shot, variando por semente: 4, 3 e 2 atributos na completa de 08/10); a linha
  "★ Passo 5: especialista eleito" só é impressa no G. Laço do F e blocos G1/G2 em `bloco_protegido`
  (F por (modelo, seed)), como o CLAUDE.md já afirmava.
- Textos de log com passo errado ou nome antigo: sub-numeração interna "Passo 2/3a/3b/4" do Passo 4 do A
  e "Passo 1..5" interno do Problema E (viravam texto visível) → "Passo 4 (Perceptron)" e sub-etapas (a)–(e);
  prefixos "[… E seed=…]" e "n_shot=" do aprendiz de F/G → "[… Passo 3/6 seed=…] k="; cabeçalhos dos vieses
  "B/C, Passos 3 e 4"; "[Protocolo A seed=…]" → "[A seed=…]"; "PASSOS 2 A 6" no D; "Passos 1 a 4" no F;
  "Passos 2 a 5" uniforme no G; "PIPELINE DOS PROBLEMAS EXTERNOS" → "PROBLEMAS F e G".
- `ResultadoAprendiz.n_shot` → `k` (o campo que faltou no mapa; eixo "n_shot" de `final_graphic_06` e tabela
  do sumário estatístico seguem); `bloco23_graphic_04` em grade de 3 colunas (9 painéis numa linha eram
  ilegíveis); painéis individuais de E (01/04/05/06/07) e de B/C (10) citam problema e passo; figura 02/13
  com os painéis na ordem Passo 1, 2, 3, 4; `bloco3_table_06` ganhou a coluna "Eleita (Passo 5)".
- Comportamentos que o código fazia e a lista canônica não dizia, agora documentados (CLAUDE.md,
  `passos_por_problema.md`, `problemas_tabela.html`): A colapsado ⇒ B/C não coletados nessa configuração; a
  eleição do Passo 6 do A só entra nas tabelas 06–08 e na figura 11 (H1/H2 usam a métrica zero-shot);
  Passo 6 do G sorteia os MESMOS pontos do Passo 3 (mesma semente seed + rep), rotulados pela métrica eleita;
  teto do G gravado como `passo5_teto` (numeração do A/D); baselines do E treinados em todas as estratégias
  (tabela e figura mostram a mista); `fidelity_teto_vs_llm_passo2` fica NaN se o zero-shot colapsou.

### Baixas (corrigidas)
Rótulos LaTeX `tab:T1…T9` → `tab:<slug>`; docstrings (`passo4_estimar_metrica_r2_r3_r4` cobre A, D, G e F;
`_plot_simbolos_passos` "Passos 2 a 4"; `_linhas_passos_2_a_5` cita o histórico de nomes e o G; `reps_para`
cita o Passo 3 de A/D/G); comentários obsoletos (ferramenta H/M removida, `test_selecao_exemplos`, razões do
oráculo, "Fase 1 da modularização"); `data_problems` (problem_E.csv em 01/10, problema_E.csv em 13/10);
tabela 06 de B/C com Perceptron antes do NNLS e o teto identificado como Passo 5; tabela 10 do F com k em vez
de n; caption da `bloco2_table_06` "Passos 3 e 6"; CLAUDE.md (parâmetro `ks_rotulo_real`, `metrica`,
`passo2_zero_shot` no E, `x2_dom/x1_dom`, 361 testes, k de `K_EXEMPLOS_*`); README dos dados reais; skill
revisor-resultados ("Oráculo"); mapa (chave de `por_regime`, `k`, rótulos, `colapsou`, nomes de teste);
testes renomeados (`TestG2MostraKgM`, `TestMalformadasE`, `TestEPassos2e3`, `TestFGEspecialista`,
`test_abc_executar_*`, `test_grafico_20_removido`, `_COLS_FG_APRENDIZ_ANTES`).

### Não corrigido (registrado)
- `bloco1_graphic_05` em P&B: o painel scatter perde a distinção "marcador = par de nomes" (todos viram +);
  exige regra nova em `estilo_cinza` (símbolo por série secundária).
- Coluna `seed` (não `random_seed`) em `bloco1_a_passo4_r2_r3_r4` e nos CSVs `bloco23_fg_*`: fora do mapa,
  mantida.
- Rótulos em inglês das estratégias e classificadores nas figuras do E (EASY/HARD, "Logistic Regression")
  versus Fácil/Difícil nas tabelas: siglas permitidas pelo mapa; harmonizar se o orientador pedir.
- O smoke de 18-24-11 foi gerado antes destas correções (CSV do oráculo com 12 linhas, supertítulo antigo em
  `bloco23_graphic_02`, log com "[PASSO 5]"); a próxima execução completa limpa tudo.


## Auditoria dos resultados da completa de 13/10/2026 (14/10/2026)

Execução `resultados/execucao_2026-10-13_22-27-34_completa/`: 187.328 chamadas, exit 0, 6 malformadas (todas gemini,
variante chain-of-thought, falhas de API após 5 tentativas, 1 ponto em 150 por seed), flip a T=0 1,4% no gpt e 0% no
gemini, 292 grupos de auditoria sem `sem_tag`, 0 avisos de legenda, 1.586 PNGs em pares `_color`/`_gray`, 70 tabelas,
3h39 com MAX_CONCURRENCY=40. Quatro auditores somente leitura (Bloco 1; Bloco 2; Bloco 3; fechamento e transversal)
conferiram completude, nomes, números (CSV × log × tabela × figura), comportamento e legibilidade. Nenhuma divergência
grave. Correções aplicadas no código e, onde o asset deriva só dos CSVs, regeneradas offline na própria pasta:

- **`collapsed` do CSV ABC**: o runner ligava `collapsed=1` também no colapso de B/C, o que apagava linhas válidas do A
  das tabelas 14/16 e dos painéis zero-shot das figuras 15/16 do gpt. Agora `collapsed` é só o colapso do A; o CSV da
  completa foi recalculado (`collapsed = fidelidade do A ausente`) e figuras 05/07/10/15/16/17/18, final_08 e tabelas
  regeneradas; `_tab_vies` anota "(n=…)" quando a linha tem menos sementes que o total.
- **Teto do G não é máximo**: o Perceptron fica em melhor esforço (γ = 0) em todas as sementes do G; rodapé da tabela 06
  e legenda das curvas (04 e bloco3_03) dizem "melhor esforço se γ = 0"; a tabela 06 informa a fração de estimativas com
  γ = 0, a regra de desempate da eleição, "—" na coluna Eleita quando o zero-shot colapsou em todas as sementes e lista
  Mulher/Homem antes de A/B; as curvas do G citam a permutação nome↔classe das variantes A/B.
- **Tabela 09 do F** sai com uma linha "colapso" quando o zero-shot do F colapsou em todas as sementes (gpt), em vez de
  sumir; rodapés das tabelas 01/04/05 do G usam a nota única de colapso; tabela 03 do G avisa que os atributos extras do
  G2 vão ao prompt em kg·m; `final_table_01` explica as variantes do G omitidas por colapso total.
- **Figuras**: curva A/D em 2×2 (eixos legíveis); gamma (09) com legenda abaixo da figura em 4 colunas; figuras do E com
  os nomes das tabelas (Fácil/Difícil/Mista/Aleatória, Regressão Logística, "LLM (com exemplos)", "viés de recência") e
  eixos "k exemplos"; títulos de E1 com problema e passo; boxplot do W normalizado por ΣW; rótulos "F (meia-lua)",
  "G1 Mulher/Homem" nas figuras de F/G; final_06 sem "n_shot"; SVM do D anota n0/n1.
- **Log**: `warnings.warn` do Perceptron virou `print` (eram 2.533 ecos com caminho e linha); cabeçalho de B/C sem
  "PASSO 2, PASSO 2"; "n_shot" nas caixas de H4 e do E virou "k"; veredito, hora de fim e duração impressos ANTES de
  gravar o log (o log terminava em "[ETAPA 7]").
- **Documentação**: CLAUDE.md aponta para a nova completa, descreve a semântica de `collapsed`, os painéis extras das
  figuras 10 e 17, a tag `problema=ABC` do zero-shot e o teto do G.

Não regenerados (dependem de dados que só a execução tem): `bloco1_graphic_09` (legenda ainda sobre os eixos nesta
pasta), `final_graphic_06` ("Consistência vs n_shot" no painel) e `final_graphic_07`; `bloco23_graphic_03` (centróides
não estão no CSV). Resultados para o texto, apontados pelos auditores: zero-shot do A/E/G com classes neutras sai
abaixo de 50% literal (permutação nome↔classe) e com Mulher/Homem em valores normalizados o LLM inverte os nomes (G1:
gpt colapsa em "Mulher", gemini 35%), o que some em kg e m (G2: 89,5%/84,3%); H1 sustentada em B só no gpt (κ 0,46)
e em C nos dois (0,74/0,85); H4 refutada nos dois modelos (easy > hard em média) mas com cruzamento em 40-shot no gpt;
a métrica eleita no A não melhora B/C em relação à zero-shot; gpt colapsa no zero-shot do D e do F nas 3 sementes;
Perceptron em melhor esforço (†) em 100% das estimativas de A/D e 93% das do G.

## Auditoria geral de 14/10/2026 (reunião do orientador × código; nomes, comentários, arquivos, figuras)

Sete auditores somente leitura: (1) pedidos do orientador (transcrição de 02/10, e-mail de 05/10, PDFs
Metodologia/Resultados/FundamTeo) × código; (2) A e vieses; (3) B/C e oráculo; (4) D; (5) E e F; (6) G;
(7) infraestrutura, fechamento e documentação. Nenhuma divergência grave de comportamento. Corrigido no mesmo dia
(361 testes OK, pyflakes limpo, assets derivados dos CSVs regenerados na completa de 13/10):

- **Figura 09 (γ):** `garantir_legendas` recriava uma legenda de 36 séries dentro de cada eixo mesmo com a
  legenda de figura; agora respeita `fig.legends` (só a próxima execução regenera a 09).
- **Figura 05 em P&B:** o scatter da direção de Ŵ passou a uma série por par de nomes (marcador e cor por par,
  semente anotada como número, eixos em vetor unitário), e o boxplot usa pesos normalizados por ΣW.
- **Colapsos visíveis:** figuras 15/16 anotam "colapso" nos slots vazios; a tabela 14 lista os pares cujo A
  colapsou em todas as sementes com "—" e o nº de sementes; a tabela de comparação de modelos exclui o D
  colapsado da acurácia do LLM; tabela 12 sem "††"; figura 11 e violino das margens usam a regra única
  `colapsou` (sem fallback aos centróides do A); `_nnls_bc` devolve None e `n_disagreements_b/c` NaN no colapso.
- **Nomes:** parâmetros e variáveis `n_shot*` → `k_exemplos`, `k_vies`, `k_e`, `k_bl`, `k_ordem`, `valores_k`,
  `k_comum`, `_draw_k`; `EXAMPLE_ORDER_N_SHOTS` → `K_ORDEM_EXEMPLOS`; tabelas do E com "k" em vez de "n";
  "Regressão Logística" também na tabela 06; `if True:` removido do laço do F; figuras de F/G ordenadas
  (F, depois G com Mulher/Homem antes de A/B) e `final_graphic_07` com "G2 Mulher/Homem" no título; teto do D
  (tabela 18) em Ŵ unitário e algoritmos capitalizados; "Oráculo Meia-lua" → "Problema D, Passo 5 (teto)".
- **Textos:** remoção do Wilcoxon atribuída ao autor (não ao orientador); concorrência 50 nos documentos;
  71 tabelas; flip 1,4%; `passos_por_problema.md` sem o parágrafo "identificadores não mudam"; glossário
  (baselines em todas as estratégias); `problemas_tabela.html` referenciando a completa.

### Reunião do orientador × código: o que ainda depende dele (para o texto ou para decisão)
1. Split 70/30 "em E e F" (IV.7): o E não tem split (avalia nos 150 − k); F e G têm.
2. Atributos extras (x1·x2, x1², x2²) no PROMPT do G (IV.7) × "na estimativa" (e-mail 05/10): o código põe só na
   estimação (Passos 2 a 4) e no prompt apenas no Passo 6 quando a métrica eleita é R³/R⁴ (em kg·m no G2).
3. IC bootstrap (IV.7, Tabela 4): removido a pedido dele por e-mail; o PDF ainda cita; arquivar o e-mail.
4. Few-shot de B/C "com a melhor métrica, n ∈ {4, 10}" (IV.5): o código usa a métrica zero-shot do A com
   k ∈ {4, 10, 20, 40}; a eleita só entra nas tabelas 06–08 e na figura 11 (decisão de 13/10).
5. "3 repetições por configuração" (IV.2): só onde há sorteio de exemplos.
6. Tabela 1 do PDF: E tem 150 pontos (σ 1,3; centróides (−1,5; 1,0)/(1,5; −1,0)), não 100; D e F 150.
7. Algoritmo 1 (pré-condição y ≠ ŷ) e Algoritmo 2 (γ multiplicativo a partir de 0) diferem do código (correção
   por margem; γ aditivo 0,05 a partir de 0,1; melhor esforço com γ = 0 em 100% de A/D e 93% do G).
8. Oráculo descrito sobre o A (V.2): é sobre B, E1 e E2; "100% na meia-lua" (V.4) era do oráculo gaussiano, o
   teto do D é 85–87%.
9. Especialista do G no aprendizado em contexto = métrica eleita (decisão do autor); diluição com 4 difíceis
   (PDF diz 3); nomes "feature_1/feature_2" (PDF: "atributo 1/2"); figura do SVM é do D, não do F.
