# Análise — Reunião 2026-10-02 (orientador) — o texto dele versus o que o sistema faz

Gerado em 14/10/2026 a partir de: transcrição da reunião, e-mail de 05/10/2026, PDFs
revisados (FundamTeo, Metodologia[90], Resultados), código em `src/` e execução
`resultados/execucao_2026-10-13_22-27-34_completa`.

## Resumo
O orientador assumiu a escrita (o texto é dele; George entrega tabelas, figuras e números).
Dos 25 itens extraídos, 16 estão feitos no código e na execução de 13/10, 4 estão parciais
e 5 pendem de decisão ou de texto. Os pontos em que o TEXTO diz uma coisa e o SISTEMA faz
outra estão na seção "Divergências", e são os que precisam de resposta a ele.

## Placar

| # | O que o texto/reunião pede | Onde checei | Status | Falta |
|---|---|---|---|---|
| 1 | Tabela 1: nº de amostras de D, E, F ("?") e centróides | `N_PONTOS_A/E`, `create_problem_d_meialua`, `fg_executar` 70/30 | ✅ código | texto: D 150, E 150 (centróides (−1,5; 1) e (1,5; −1), σ 1,3; na reunião George disse 100), F 150 (105/45) |
| 2 | B/C: centróides (−2; ±1,5)/(2; ∓1,5), σ 1,2, 100 amostras | `data_problems.py:25-31` | ✅ | — |
| 3 | 2 modelos, T = 0, sementes {42,123,7}, R = 3 | `MODELS_TO_TEST`, `RANDOM_SEEDS`, `N_REPETICOES` | ✅ | — |
| 4 | Nomes de classe (4 pares + inversões), atributos (x1/x2, altura/peso, técnico), 4 variantes de prompt | `NOMES_CLASSES`, `NOMES_CLASSES_INVERTIDAS`, `NOMES_FEATURES`, `PROMPT_VARIANTS` | ✅ | texto diz "atributo 1/atributo 2"; o código usa `feature_1/feature_2` |
| 5 | "limite de dez consultas em paralelo", 129.868 chamadas, 0 fallback, flip 4,3 % | `MAX_CONCURRENCY = 50`; `final_table_03` da completa de 13/10 | ⚠️ texto desatualizado | 50 em paralelo; 187.328 chamadas; 6 malformadas (Gemini, 0,006 %); flip 1,4 % gpt e 0 % Gemini |
| 6 | Few-shot n ∈ {4, 10} (especialista) e {4,10,20,40} (aprendiz) | `K_EXEMPLOS_ROTULO_REAL = [4,10,20,40]`, `K_EXEMPLOS_METRICA_BC = [0,4,10,20,40]`, `K_VIESES = [0,10]` | ⚠️ divergência | código usa a grade cheia no Passo 3 de A/D/G e no Passo 4 de B/C; só os vieses usam {0,10}. Orientador ajusta o texto (decisão 13/10) ou reduz-se a grade |
| 7 | Cinco métricas (IV.3): fidelidade, acurácia, consistência, κ (Landis e Koch), F1-Macro | `metrics.py`, `tabelas.faixa_kappa`, CSVs | ✅ (09/10) | — |
| 8 | Projeções R³ (x1·x2) e R⁴ (x1², x2²) | `augment_to_r3/r4` | ✅ | — |
| 9 | Bloco 1: A em zero-shot e few-shot, Perceptron principal + NNLS, cosseno | `adg_passos_2_a_5`, `bloco1_table_01/02/05` | ✅ | — |
| 10 | "Melhor métrica" transferida a B/C; LLM reclassifica zero-shot e few-shot | `a_passo6_eleger_metrica`, `bc_passo3_consistencia` (tabelas 06-08, figura 11) | ⚠️ parcial | a eleita entra SÓ no Passo 3 (consistência). O few-shot de B/C (Passo 4, H2), a tabela 09, a figura 10 e o veredito de H1 usam a métrica ZERO-SHOT do A (decisão 13/10). Texto precisa dizer isso ou o código muda |
| 11 | D com o mesmo protocolo do A; tabela do D | `RUN_PROBLEMA_D`, `bloco1_table_10..13`, `bloco1_graphic_04/13/14/21` | ✅ | — |
| 12 | Teste-oráculo com [1,1], [0,3;1,5], [1,5;0,3] = Problemas 1/2/3; tirar a 1ª linha | `oraculo_passos_1_a_3`, `bloco1_table_17`, `bloco1_graphic_19` (painéis separados) | ✅ | texto (Resultados V.2) diz "rotulação dos dados do problema A": o código rotula os dados de B para [1,1] e os de E para os anisotrópicos |
| 13 | Bloco 2: E1 (x2 5× mais relevante), F com especialista = rótulo real, 4 estratégias, baselines, ordem | `EXPERT_CONFIGS`, `fg_executar(expert_labels=y)`, `e_passo3_selecionar_exemplos`, `RUN_E_PASSO6_BASELINES`, `RUN_E_PASSO5_ORDEM` | ✅ | — |
| 14 | "O mesmo protocolo de divisão 70/30 é aplicado aos problemas E e F" | `e_passos_2_3_aprendiz`: classifica os 150 − k restantes (`test_mask`), sem split | ⛔ divergência (E) | F tem split; E não. Pendência já conhecida com o orientador |
| 15 | G: 100 amostras 50/50, 70/30 por semente, métrica nos 70 | `peso_altura.csv` (50/50), `fg_executar` | ✅ | — |
| 16 | G: LLM classifica o treino em zero-shot e few-shot {4,10,20,40} "seguindo os passos do A" | `com_passos_3_a_5=True`, `bloco3_g_passos_2_a_5` (8 combinações × 504 linhas, R²/R³/R⁴, teto), `bloco3_table_06`, `bloco3_graphic_03` | ✅ (13/10) | — |
| 17 | G: exemplos aleatórios do treino, avaliação nos 30 de teste | Passo 6 (`fg_exemplos_aprendiz`, k/2 por classe real) | ✅ | sorteio é balanceado por classe, não uniforme puro; vale uma frase no texto |
| 18 | G: viés de nomes de classe (A/B) e de atributo (x1/x2); nº absoluto de erros | `RUN_PROBLEMA_G_CLASSES_AB`, `G1/G2_NOMES_ATRIBUTO`, tabelas do G com "(n erros)" | ✅ (atende o e-mail de 05/10) | — |
| 19 | G: não linearidade "alimenta-se o PROMPT com x1·x2 (R³) e x1², x2² (R⁴)" | estimação: `bloco23_fg_passos_2_e_4`, `bloco3_g_passos_2_a_5` em R²/R³/R⁴ ✅. Prompt: `fg_executar` só mostra os atributos extras no Passo 6 quando a métrica eleita no Passo 5 é R³/R⁴, e só junto dos exemplos (zero-shot sempre x1, x2) | ⚠️ parcial | o e-mail de 05/10 ("na estimativa da métrica") está atendido; a frase da Metodologia (prompt com os atributos) só vale em 11 das 16 combinações modelo × variante da completa de 13/10 e não é uma comparação controlada 2 × 3 × 4 atributos no prompt. Decidir |
| 20 | IC bootstrap 95 % com 10⁴ reamostras | removido do código em 13/10 a pedido dele (`relatorios.py:644`) | ⚠️ texto desatualizado | tirar a frase de IV.7 e o "IC bootstrap" da Tabela 4 |
| 21 | Feedback imediato (II.6, "falta explicar") | nenhuma ocorrência em `src/` | ⛔ adiado | combinado na reunião (08:36): "deixa para depois"; só no problema real |
| 22 | Resultados: plotar amostras rotuladas pelo LLM vs rótulo real no A; Figura 2 "quebrar" | `bloco1_graphic_02` (símbolos); `bloco1_graphic_19` com painéis | ✅ | — |
| 23 | Tirar consistência euclidiana; resolver "100 % vs 86,7 %" da meia-lua; R³/R⁴ "em A, B e C" | euclidiana removida (09/10); teto do D (`bloco1_table_18`) = 84,7 a 87,3 % em R²/R³/R⁴, sem 100 %; R³/R⁴ só no A (`RUN_A_PASSO4_R3R4`; B/C decididos como não aplicáveis em 13/10) | ✅ / texto | Tabela 5 do texto deve ficar só com o A (ou A e D) |
| 24 | Explicar as 4 variantes de prompt por e-mail; corrigir "altura/peso" na Tabela 6; Tabela 7 com 61 % repetido; renomear "aniso_x2"; Figura 7 "3 difíceis" | `PROMPT_VARIANTS` existe, mas não localizei rascunho nem envio da explicação; Tabela 15 corrigida; 61 % é a MESMA coleta zero-shot (1× por modelo e semente, rodapé da tabela 03 explica); E1/E2/E3 desde 08/10; diluição fixa 4 difíceis (`n_hard_fixed = 4`) | ⚠️ | enviar a explicação das variantes; avisar que são 4 difíceis, não 3 |
| 25 | Algoritmos 1 e 2 (FundamTeo) = código? Orientador validou na reunião (suavização α/λ e busca binária) | `relaxed_perceptron.py:158-330` | ⚠️ 3 diferenças de pseudocódigo | (a) texto corrige só quando y ≠ ŷ E a margem falha; código corrige em qualquer violação de margem; (b) Algoritmo 2 parte de γ = 0 com incremento multiplicativo Δ; código parte de γ = 0,1 com incremento aditivo 0,05 e depois bissecção (tol 1e-4); (c) código devolve melhor esforço (γ = 0, marcado †) quando nenhum γ é viável; o texto não descreve. Sinal da correção e Max{0, ·} batem |

## Divergências entre o texto e o sistema (precisam de resposta ao orientador)
1. Atributos extras no PROMPT do G (item 19): hoje só quando a métrica eleita é R³/R⁴.
2. Split 70/30 no Problema E (item 14): não existe; o E classifica os 150 − k restantes.
3. Grade de k do especialista (item 6): texto {4, 10}; código {4, 10, 20, 40}.
4. Métrica usada no few-shot de B/C (item 10): texto "melhor métrica"; código métrica zero-shot do A.
5. Dados do oráculo (item 12): texto "problema A"; código B (para [1,1]) e E (anisotrópicos).

## Números para o texto dele (fonte: completa de 13/10/2026)
- Tabela 1: D 150, E 150, F 150 (105 treino / 45 teste); G 100 (50/50), 70/30.
- Protocolo: 50 chamadas em paralelo; 187.328 chamadas; 6 malformadas (0,003 % do total, todas do Gemini); flip a T = 0: 1,4 % gpt-4o-mini, 0 % Gemini.
- Teto do D (W ajustado ao rótulo real): 84,7 a 87,3 % (R², R³, R⁴), nunca 100 %.
- Diluição: 4 difíceis fixos + {0, 2, 4, 10, 16, 20} fáceis.
- Sem IC bootstrap e sem Wilcoxon; média, faixa mín–máx entre sementes e Cohen's d.
- Oráculo: Problemas 1/2/3 = sobre B ([1,1], 100 pts), E1 ([0,3;1,5], 150 pts), E2 ([1,5;0,3]).

## Próximos passos priorizados
1. E-mail curto ao orientador com as 5 divergências acima, perguntando o que prevalece (texto ou código). Destrava os itens 6, 10, 12, 14 e 19.
2. Enviar a explicação das 4 variantes de prompt (ele precisa para escrever IV.2/V.5). Fonte: `PROMPT_VARIANTS` em `src/dissertacao_mestrado.py`. Encaminhar: escritor-artigo.
3. Mandar a lista de números acima para ele atualizar Tabela 1, IV.2, IV.7, Tabela 4, Tabela 5 e Figura 7.
4. Alinhar o pseudocódigo dos Algoritmos 1 e 2 ao código (3 diferenças do item 25), depois de ele decidir qual prevalece.
5. Bloco 3: aguardar os resultados do NeuroSolutions (RBF/MLP) que ele vai fornecer; feedback imediato fica para depois, só no problema real.

## Resolução (14/10/2026, decisão do autor)
- Item 19 (atributos no prompt do G): resolvido no código. `fg_executar(..., atributos_prompt_aprendiz=ATRIBUTOS_PROMPT_PASSO6_G)` roda o Passo 6 do G com 2, 3 e 4 atributos no prompt para o mesmo especialista; k = 0 é uma coleta só; coluna `n_features` = atributos no prompt, nova `especialista_n_features`; tabelas 03_02/03_03 com coluna "Prompt"; teste `tests/test_g_passo6_atributos_prompt.py`.
- Item 12 (dados do oráculo): resolvido no código. `oraculo_passos_1_a_3(X_e, EXPERT_CONFIGS, seed)` roda [1,1] (E3), [0,3;1,5] (E1) e [1,5;0,3] (E2) sobre as gaussianas do E; "Oráculo sobre B" fica só nas execuções até 13/10/2026.
- Itens 6, 10 e 14 (grade de k, métrica dos exemplos de B/C, split no E): código mantido; frases prontas no e-mail `email_orientador_divergencias_texto_sistema.md`; só o split no E fica como opt-out para o orientador.
- Validação: 365 testes OK; smoke `resultados/execucao_2026-10-14_16-09-31_apenas-bloco3+oraculo_smoke` (Gemini, 2.048 chamadas, exit 0, 0 malformadas, 0 avisos de legenda). Falta a execução completa nova (≈ US$ 13 a 16, ~4 h).
