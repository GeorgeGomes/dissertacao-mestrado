# Plano de ação para a reunião de sexta, 02/10/2026

## O que o orientador confirmou (e-mail de 01/10/2026)

Respondendo ao e-mail de confirmação (`email_orientador_entrega_sexta.md`):

1. "É exatamente isto": Problema A em few-shot (exemplos rotulados pelo ground truth, k em 4, 10, 20, 40, sem split, nova tabela ao lado da zero-shot e da estimada pelo ground truth); B e C só zero-shot com a melhor métrica do A; D repete o protocolo do A, inclusive few-shot.
2. Texto é dele: "não se preocupe com o texto, pode deixar que eu escrevo". Entregar **somente resultados em tabelas e figuras**. Cai a pendência de reescrever bootstrap, Cohen's d e Algoritmo 1 no artigo para sexta.
3. **Vieses semânticos só no problema real** (peso × altura). Logo, nos Problemas A a D não se roda nomes de classe, nomes de atributo nem variantes de enunciado.
4. **Não linearidade na meia-lua com os novos atributos**: o Problema D entra em R², R³ e R⁴ (x1·x2; x1², x2²), em zero-shot, few-shot e ground truth.

Pedidos anteriores que continuam valendo (e-mail de 28/09, outro fio):
- Figura das 150 amostras do A rotuladas por ground truth, pelo LLM e pela métrica, com **símbolos** (+, x, o) por classe, legível em preto e branco.
- Mesma figura para B e C, com explicação da "rotação" (centróides (-2,0)/(2,0) passam a (-2,±1,5)/(2,∓1,5)).

Fora do escopo de sexta: retorno imediato (só peso × altura, depois), vetor alfa ("não agora"), decisão sobre o Algoritmo 1 (levar os dois Word da raiz para discutir ao vivo).

## Situação do código hoje

- `phase_a_learn_metric` (src/dissertacao_mestrado.py:1198) coleta só zero-shot (`examples=None`) e estima W pelos rótulos do LLM. Não existe estimação pelo ground truth dentro da execução; os números de 28/09 vieram de um script de scratch.
- Few-shot nas Fases B e C usa `select_confident_examples` (rotulados pela **métrica**, não pelo ground truth) e remove os exemplos do teste via `test_mask`. É o mesmo mecanismo a reaproveitar no A, trocando a origem dos rótulos.
- Problema D roda via `run_external_problem_pipeline` (linha 2607) com split 70/30, R²/R³/R⁴ e Fase E com a melhor métrica. O protocolo do A é sem split, então a Fase A do D precisa rodar nos 150 pontos.
- Ground truth na meia-lua já existe sem LLM em `run_oracle_meialua` (linha 1718) e como `accuracy_perc_vs_true` no pipeline externo.
- `--apenas bloco1` liga só `RUN_PHASES_ABC` e `RUN_R3R4_EXPERIMENT`; a meia-lua está em `bloco2` (`RUN_PROBLEM_MEIALUA`). Precisa de ajuste para a Fase A do D entrar no Bloco 1.
- Flags de viés (`RUN_CLASS_ORDER_BIAS`, `RUN_FEATURE_NAMES`, `RUN_PROMPT_VARIANTS`) hoje são True.

## Cronograma

Hoje é quinta, 01/10. A reunião é sexta à tarde. Há um dia útil de trabalho.

| Quando | O quê |
|---|---|
| Qui 01/10, manhã e tarde | Etapas 1 a 4 (código) + testes unitários |
| Qui 01/10, fim da tarde | Etapa 5: smoke test `--rapido --apenas bloco1 --modelo gpt` e `--modelo gemini`; conferir CSVs e figuras |
| Qui 01/10, noite | Etapa 6: execução `--apenas bloco1` (2 modelos, 3 sementes). Estimativa: ~22 mil chamadas, cerca de 1 h com concorrência 10 |
| Sex 02/10, manhã | Etapas 7 e 8: tabelas e figuras por script, auditoria das interações |
| Sex 02/10, início da tarde | Etapa 9: montar o pacote (só tabelas e figuras) e enviar antes da reunião |

Se a implementação não fechar até o fim da tarde de quinta, rodar a execução com o que estiver pronto (A few-shot e ground truth são o mínimo) e deixar D em R³/R⁴ para um segundo lote na sexta cedo.

Decisões já tomadas, não reabrir: exemplos rotulados pelo ground truth; k em {4, 10, 20, 40}; sem split; B e C só zero-shot; nada de texto.

## Etapa 1. Fase A com protocolo único (A e D)

Criar uma função `phase_a_protocol(X, y_true, nome_0, nome_1, n_features, seed, ...)` que devolve, para um mesmo conjunto de pontos, três famílias de métricas:

1. **zero-shot**: coleta `examples=None` (código atual).
2. **few-shot k**: para cada k em `FEW_SHOT_SIZES[1:]` e cada repetição `rep` em `range(reps_para(k))`:
   - sorteia k exemplos **balanceados** (k/2 por classe) do próprio conjunto, rotulados por `y_true`, com `random_state = seed + rep` (regra única já usada na Fase E);
   - coleta o LLM nos 150 − k pontos restantes com esses exemplos no prompt (`build_prompt_few_shot`, formato igual às Fases B/C);
   - estima W (Perceptron + NNLS) com os rótulos do LLM nesses 150 − k pontos; centróides pelos rótulos do LLM;
   - registra fidelidade vs LLM, acurácia vs ground truth, acurácia do próprio LLM vs ground truth, W, γ, partição de classes do LLM.
3. **ground truth**: estima W com `y_true` nos 150 pontos; fidelidade vs ground truth (teto do estimador) e acurácia dessa métrica vs rótulos do LLM zero-shot.

Observações:
- Seleção aleatória balanceada, não "easy": o orientador quer testar se exemplos melhoram a rotulação; uma estratégia por margem misturaria dois efeitos. Se der tempo, acrescentar "easy" como linha extra, nunca em lugar da aleatória.
- `n_features` em {2, 3, 4}: o LLM sempre vê só x1 e x2; a augmentação (`augment_features`) entra só na estimação de W. No A usar 2 (R³/R⁴ do A já existe em `RUN_R3R4_EXPERIMENT`); no D usar 2, 3 e 4.
- Guardar `y_llm` de cada coleta em um cache por `(provider, model, seed, problema, k, rep)` para reaproveitar em B/C e nas figuras sem chamar a API de novo.
- Não mexer em `relaxed_perceptron.py` nem em `least_squares_inverse.py`.

## Etapa 2. Fases B e C com a melhor métrica do A

- Manter a coleta zero-shot atual em B e C (100 pontos novos cada, só W transferido, centróides recalculados com rótulos do LLM no próprio B/C).
- Calcular consistência e kappa contra **todas** as métricas estimadas no A (zero-shot, cada k, ground truth), sem novas chamadas: só `predict_with_metric` sobre `y_llm_b` e `y_llm_c` já coletados.
- Marcar a "melhor métrica do A" pelo critério acordado: maior fidelidade vs LLM no A (média das 3 sementes, Perceptron). Reportar a linha da melhor em destaque e as demais como comparação.
- Desligar os tamanhos few-shot de B/C (`FEW_SHOT_SIZES` nesta execução vira `[0]` para B/C) para não gastar chamadas em algo que o orientador não quer.

## Etapa 3. Problema D no Bloco 1

- No laço do Bloco 1, para cada semente, gerar a meia-lua (`create_problem_d_meialua`, 150 pontos) e chamar `phase_a_protocol` com `n_features` em {2, 3, 4}, sem split.
- Manter `run_external_problem_pipeline` intocado para o Bloco 2 (Problema F, Fase E).
- Incluir a nova flag (por exemplo `RUN_PROBLEM_D_FASE_A`) em `BLOCOS_APENAS["bloco1"]` para `--apenas bloco1` cobrir A, B, C e D.
- Cuidado conhecido: o gpt-4o-mini colapsa em classe única na meia-lua zero-shot em 2 das 3 sementes. Registrar a linha mesmo assim (fidelidade = NaN, partição 150/0) e ver se o few-shot desfaz o colapso. Esse é um resultado em si.

## Etapa 4. Flags, CSVs e figuras

Flags para esta execução:
- `RUN_CLASS_ORDER_BIAS = False`, `RUN_FEATURE_NAMES = False`, `RUN_PROMPT_VARIANTS = False` (vieses só no real).
- `RUN_ALGORITHM_COMPARISON` e `RUN_ORACLE_VALIDATION` podem ficar True (0 ou poucas chamadas).
- Deixar os defaults originais anotados em comentário para a execução completa futura.

Novos CSVs (consolidados, colunas `provider`/`model`, nomenclatura do CLAUDE.md):
- `bloco1_fase_a_protocolo_{timestamp}.csv`: uma linha por (modelo, semente, problema A/D, n_features, regime zero/few-k/gt, rep, algoritmo) com fidelidade vs LLM, acurácia vs GT, acurácia do LLM vs GT, w, γ, n_classe_0/1.
- `bloco1_transferencia_bc_{timestamp}.csv`: uma linha por (modelo, semente, problema B/C, métrica de origem, algoritmo) com consistência, kappa, F1, flag `melhor_metrica_a`.

Novas figuras (uma por modelo, `llm_asset` com alias):
- `bloco1_15_rotulos_simbolos_A_seed{seed}__{alias}.png`: 3 painéis (ground truth, LLM zero-shot, métrica zero-shot), símbolos `+` e `x` em preto, centróides como `o` grande. Variante com 4 painéis acrescentando LLM 10-shot.
- `bloco1_16_rotulos_simbolos_{B,C}_seed{seed}__{alias}.png`: mesma coisa para B e C, com setas ou legenda indicando os centróides reais e a rotação.
- `bloco1_17_fase_a_fewshot_curva__{alias}.png`: fidelidade e acurácia vs GT em função de k (0, 4, 10, 20, 40), com a linha do ground truth como teto; painéis A e D (R², R³, R⁴).
- `bloco1_18_rotulos_simbolos_D_seed{seed}__{alias}.png`: meia-lua com os três esquemas de rotulação.

Testes: adicionar em `tests/` um teste da seleção balanceada por ground truth (k/2 por classe, exemplos fora do conjunto de estimação, reprodutível por `seed + rep`) e um teste de que `--apenas bloco1` liga a nova flag.

## Etapa 5. Smoke test

```bash
python src/dissertacao_mestrado.py --rapido --apenas bloco1 --modelo gpt
python src/dissertacao_mestrado.py --rapido --apenas bloco1 --modelo gemini
```
Conferir: CSVs novos com todas as combinações, nenhuma figura faltando, `python src/audit_interactions.py <pasta>` sem malformadas. Pasta `_smoke` nunca vira fonte de números.

## Etapa 6. Execução

```bash
python src/dissertacao_mestrado.py --apenas bloco1
```
Pasta `execucao_2026-10-0X_..._apenas-bloco1/`. Rodar à noite ou cedo, com `caffeinate`. Se um provedor pinado cair, as chamadas falham visivelmente: repetir só o modelo que falhou com `--modelo`.

Estimativa de chamadas por modelo: A zero-shot 450; A few-shot 4 k × 3 reps × 3 sementes × ~130 pontos ≈ 4.700; B e C zero-shot 600; D igual ao A ≈ 5.150. Total ≈ 11 mil por modelo, 22 mil nos dois.

## Etapa 7. Tabelas para o orientador

Média e mínimo/máximo das 3 sementes (sem IC bootstrap), Perceptron e NNLS, uma tabela por modelo:

- **T1, Problema A**: linhas zero-shot, 4, 10, 20, 40-shot, ground truth; colunas fidelidade vs LLM, acurácia da métrica vs GT, acurácia do LLM vs GT, partição de classes do LLM, W unitário.
- **T2, Problemas B e C**: consistência e kappa zero-shot com a melhor métrica do A; linhas extras com as demais métricas do A para comparação.
- **T3, Problema D**: mesma estrutura da T1 × n_features 2, 3, 4; coluna de ganho R² → R³ → R⁴.
- **T4, estabilidade**: W por regime e semente (razão w1/w2), para mostrar se o few-shot estabiliza a direção.

Gerar as tabelas por script a partir dos CSVs (pandas → LaTeX `booktabs`), nunca à mão.

## Etapa 8. Auditoria

`python src/audit_interactions.py <pasta_apenas-bloco1>`: taxa de malformadas (deve ser 0) e taxa de flip a T=0 por regime (zero-shot vs few-shot). Incluir a tabela de flip por k no pacote: confirma a observação do orientador de que exemplos estabilizam a resposta.

## Etapa 9. Pacote para sexta

- Um PDF curto (Beamer com o preâmbulo padrão, dentro da pasta da execução, ou um PDF de tabelas e figuras) **sem texto analítico**: título, tabela ou figura, legenda de uma linha dizendo o que é cada coluna. O texto é dele.
- Anexar os CSVs novos.
- Levar impressos ou abertos os dois Word do Algoritmo 1 (`algoritmo_comparacao_codigo_e_nova_versao.docx`, `algoritmo_completo_codigo_e_nova_versao.docx`) para fechar ao vivo qual versão prevalece.
- Depois da reunião: atualizar `roteiro_apresentacao.txt` antes dos `.tex`, e só então a apresentação principal.

## Riscos

- **Tempo**: se a implementação passar de quarta, cortar primeiro a variante "easy" e a figura de 4 painéis; nunca cortar T1 e T3.
- **Colapso do gpt-4o-mini na meia-lua**: pode zerar as linhas zero-shot de D nesse modelo. Reportar como está.
- **Provedor pinado do Gemini (Google via OpenRouter)**: checar no smoke que `inference_provider` continua `Google`.
- **Sobreposição das gaussianas no A** (erro de Bayes 4,8%): o ground truth não dá 100%, a tabela deve mostrar isso como teto do estimador para o orientador não estranhar.
