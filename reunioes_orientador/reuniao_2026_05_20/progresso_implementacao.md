# Progresso de implementação — itens da reunião 20/05/2026

Documento aditivo (não altera `plano_trabalho.md`). Registra o que foi feito no
código em resposta aos 20 itens, com correções de status descobertas na auditoria
e a análise dos itens que eram de re-análise (4 e 14).

## Resumo

Todas as **lacunas de código da Fase 1** foram implementadas e validadas
(sintaxe + testes sem-LLM). As alterações estão em `src/dissertacao_mestrado.py`.
Falta apenas **re-executar** o experimento (chamadas pagas à API) para gerar os
novos CSVs/PNGs e, então, atualizar os slides (Fase 2) e escrever o texto (Fase 3).

## Itens de código implementados

| Item | Mudança | Onde |
|---|---|---|
| **10** | Loop `N_REPETICOES` na Fase E do pipeline externo (peso×altura e meia-lua). Cada (n_shot>0) roda 3× com sorteios diferentes de exemplos; os gráficos já agregam média/std. Zero-shot roda 1× (determinístico). | `run_external_problem_pipeline` |
| **15** | Contagem **absoluta** de erros: na Fase A (`n_errors_perc_vs_true`, `n_errors_nnls_vs_true`, `n_errors_llm_vs_true`, `n_samples`) e na Fase E (`n_errors_llm_vs_true`, `n_errors_llm_vs_metric`, `n_test`). Exportadas no CSV externo. | `phase_a_multifeature`, pipeline e escrita de CSV |
| **17** | Variante de nomes de classe **A/B** no peso×altura (além de Homem/Mulher), via `HM_CLASS_NAME_VARIANTS` e flag `RUN_HM_CLASS_NAMES_AB`. Resultados separados por `problem_name` (`homem_mulher` vs `homem_mulher_classesAB`). | bloco `RUN_HOMEM_MULHER` |
| **5** | Oracle de **aproximação** da meia-lua: aprende W (Perceptron+NNLS) a partir do ground truth em 2/3/4 features e mede fidelidade vs GT. Novo gráfico `bloco1_04b_oracle_meialua.png` e CSV `bloco1_oracle_meialua_*.csv`. | `run_oracle_meialua`, `plot_oracle_meialua` |
| **7** | Plot da **superfície SVM gaussiano (RBF)** vs LLM na meia-lua: painel 1 = SVM no ground truth, painel 2 = SVM nos rótulos do LLM + fronteira da métrica diagonal (tracejada). `bloco23_external_svm_meialua_seed*.png`. | `plot_meialua_svm_vs_llm` |
| **3** | W **aprendido** exportado no CSV do Bloco 3 (`w_perc_0..3`, `w_nnls_0..3`, `w_perc_ratio`), permitindo tabular o W de peso×altura/meia-lua nos slides (antes só havia fidelidade). | escrita do CSV externo Fase A |
| **12** | **Já estava correto** — `augment_to_r3` = (x, y, x·y) hipérbole; `augment_to_r4` = (x, y, x², y²) elipse. Nada a mudar. | (confirmado) |

## Correções de status (auditoria)

Três itens já estavam **prontos no código/dados** e tinham sido mal-classificados:

- **Item 13** (peso×altura few-shot): o pipeline **já roda Fase E** com n_shot 0/5/10/20/40
  (30 linhas no `bloco23_external_phase_e`). Não era lacuna de código — falta apenas
  **mostrar nos slides**.
- **Itens 3 e 16** (mostrar W; tabela do prompt neutro x1/x2): os dados já existem nos
  CSVs (`w_0, w_1, w_ratio`; `feature_names` = x1/x2 vs peso/altura). Falta só montar a
  tabela no slide.

## Análise dos itens de re-análise

### Item 4 — fidelidade "baixa" no Problema A (NÃO é bug)

Números da execução `2026-05-14_21-43-18` (prompt default, média das 3 seeds):

- `fidelity_problem_a` ≈ **0,82** (a métrica reproduz 82% das decisões do LLM)
- `llm_accuracy_problem_a` ≈ **0,42** — o **LLM em zero-shot acerta só ~42%** vs ground
  truth no Problema A (abaixo do acaso).

Interpretação: o Problema A é fácil *para um classificador*, mas o **LLM em zero-shot
é praticamente cego** (42% ≈ loteria — exatamente a conclusão do item 19). Os centróides
da Fase A são calculados a partir dos rótulos (quase aleatórios) do LLM; uma métrica
diagonal determinística **não consegue reproduzir rótulos incoerentes**, então a
fidelidade satura em ~82%. A "fidelidade baixa" é, portanto, **consequência do ruído do
LLM no zero-shot**, não um erro de dados. (Reforça a tese da cegueira zero-shot.)

### Item 14 — queda de acurácia com 4 features no peso×altura

Números (peso/altura semântico, Fase A, vs ground truth):

| n_features | Perceptron (acc real) | NNLS (acc real) |
|---|---|---|
| 2 | 0,900 | 0,833 |
| 3 | 0,905 | 0,838 |
| 4 | **0,914** | **0,743** |

A "queda" está **só no NNLS** (0,838 → 0,743), **não no Perceptron** (que sobe
0,900 → 0,914, como esperado para erro de treino). O orientador pediu justamente
"rodar o Perceptron aqui para comparar" — e o Perceptron **não cai**. A queda do NNLS
é artefato da restrição de não-negatividade (w ≥ 0), que **zera termos aumentados**
úteis (ex.: o termo x1·x2 e um dos quadráticos vão a 0), piorando o ajuste. Como o
pipeline seleciona a melhor config pela acurácia do Perceptron, a escolha final
(n=4, 0,914) já é a correta.

Observação extra (item 16): no prompt **neutro x1/x2**, a acurácia do Perceptron vs real
fica em ~0,26 (pior que chutar) e **não melhora** com mais features — confirmando que,
sem viés semântico, aumentar features não ajuda.

### Nota sobre o Oracle da meia-lua (item 5)

No teste sem-LLM, a fidelidade da métrica **diagonal** vs ground truth da meia-lua fica
em ~**87% e não cresce** de 2→3→4 features (cai levemente no R4). Isso porque a meia-lua
**não é uma elipse** — x1²/x2² (alinhados aos eixos) não capturam sua fronteira, ao
contrário do peso×altura. O painel do SVM RBF (item 7) evidencia o contraste: um
classificador não-linear separa a meia-lua, enquanto a métrica diagonal satura. É um
resultado honesto e informativo para o texto.

## Status de código: COMPLETO

**Nenhuma lacuna de código restante.** Todas as alterações de Fase 1 (itens 5, 7, 10,
12, 15, 17 e o item 3 do Bloco 3) estão implementadas e validadas (compila + testes
sem-LLM). O que falta NÃO é implementação:

## Pendências (re-execução, slides pós-run, ou manuais)

- **Re-executar** `python src/dissertacao_mestrado.py` (chamadas pagas à API) para gerar
  os novos CSVs/PNGs com repetições, contagens absolutas, classes A/B, oracle e SVM da
  meia-lua, e o W do Bloco 3. ⚠️ custo/tempo: o bloco externo agora roda ~3× mais coletas
  na Fase E e dobra o peso×altura (variante A/B).
- **Fase 2 (slides) — já adiantados com dados atuais:** mapas de erro 1/slide (item 6),
  nº absoluto de erros (item 15), e já presentes no deck: few-shot peso×altura (item 13),
  prompt neutro (item 16), W do Bloco 1 (item 3). **Faltam (precisam do re-run):** slide do
  oracle meia-lua (item 5), slide do SVM meia-lua (item 7), comparação classes A/B (item 17),
  tabela do W do Bloco 3 (item 3 externo), limpeza do Bloco 3 (item 2), nomenclatura (item 1).
  Lembrar: **roteiro_apresentacao.txt primeiro**, depois os dois `.tex`.
- **Fase 3 (escrita):** itens 8, 9, 11, 18, 19 (texto da dissertação).
- **Fase 4 (leitura):** item 20 (8 artigos).
