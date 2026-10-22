# Levantamento: o que o orientador pediu em vermelho (reunião de 02/10/2026) e como entregar

Fontes: `transcricao_reuniao.txt` (02/10/2026), `email_2026_10_05.txt`, os três PDFs de
`doc revisados orientador/` (texto em vermelho extraído por cor: 2.067 caracteres em
`Resultados.pdf`, 172 em `Metodologia[90].pdf`, nenhum em `FundamTeo.pdf`) e a execução
completa `resultados/execucao_2026-10-15_20-15-20_completa` (Perceptron corrigido de 15/10/2026).
Levantamento feito em 16/10/2026.

Regra dita por ele na reunião (00:09:20): "Tudo que é tabela e figura, eu escrevi em vermelho".
O texto é dele; George entrega tabelas, figuras e os números para ele escrever.

## 1. Itens em vermelho, um a um

| # | Onde (PDF, página) | O que está em vermelho / o que ele pediu | Status hoje | O que entregar (execução de 15/10 à noite) | Observação para ele |
|---|---|---|---|---|---|
| R1 | Resultados p.1 | **Tabela 2**: fidelidade no A, 3 sementes, zero-shot. "Falta fazer experimentos com few-shot e gerar uma nova tabela." | ✅ feito | `bloco1_table_01_A_passos_2_a_5__<modelo>` (zero-shot, 4/10/20/40-shot, teto; Perceptron e NNLS; acurácia do LLM e partição) + `bloco1_table_02_A_passo2_vs_passo3` (zero × few lado a lado) + `bloco1_table_05_A_passo4_perceptron_vs_nnls` (cosseno) | Em zero-shot o gpt inverte nome↔classe (partição 109/41): acurácia literal 26,9 %, invariante 73 %; a tabela traz a nota |
| R2 | Resultados p.1 | "Seria interessante plotar as amostras das gaussianas rotuladas pelo LLM e pelo ground truth." | ✅ feito | `bloco1_graphic_02_A_passos_2_a_4_simbolos_seed{42,123,7}__<modelo>` (rótulo real, LLM zero-shot, métrica, LLM 10-shot; símbolos + e −, P&B) | Mesma figura para B e C (`bloco1_graphic_11_{B,C}_passos_2_3_simbolos`) e D (`bloco1_graphic_13`), pedidas no e-mail de 28/09 |
| R3 | Resultados p.1 | **Figura 1**: concordância dos estimadores, fidelidade no A e consistência em B e C | ✅ feito | `bloco1_graphic_06_A_passo4_perceptron_vs_nnls__<modelo>` (fidelidade A, B/C pareado, direção de W) + `bloco1_graphic_10_BC_passo3_consistencia__<modelo>` | Nenhuma dúvida pendente |
| R4 | Resultados p.2 | **Tabela 3** (oráculo): "Qual a diferença entre problema A e B? As métricas foram utilizadas para rotular o problema A? Renomear Expert aniso. Sugestão: Problema 1, 2, 3 (e 4) com as métricas na frente"; na reunião (00:15): tirar a primeira linha, deixar 1, 2 e 3 | ✅ feito | `bloco1_table_17_oraculo_passos_1_a_3` (Problema 1 = [1; 1], 2 = [0,3; 1,5], 3 = [1,5; 0,3]; 2 estimadores × 3 sementes) | Resposta às perguntas: as três métricas rotulam o MESMO conjunto, as gaussianas do Problema E (centróides (−1,5; 1) e (1,5; −1), 150 pontos). O Problema A não serve porque, até 14/10, seus centróides tinham x2 = 0 e w2 não era identificável. Resultado: Perceptron 100 % nas 9 combinações; NNLS 96,7 a 99,8 % |
| R5 | Resultados p.2 | **Figura 2** (validação dos estimadores): "quebrar", está pequena | ✅ feito | `bloco1_graphic_19_oraculo_passos_1_a_3` + painéis `_w_scatter`, `_ratio`, `_cosseno`, `_acuracia` | Entregar os 4 painéis separados |
| R6 | Resultados p.2 | **Tabela 4**: transferência B/C, consistência e κ, zero-shot e few-shot com 4 exemplos, "IC bootstrap 95 %" | ✅ feito, com 2 mudanças | `bloco1_table_08_BC_passo3_zero_vs_eleita` (zero-shot × métrica eleita), `bloco1_table_09_BC_passo4_exemplos_da_metrica` (k = 4, 10, 20, 40), `bloco1_table_07_BC_passo3_por_semente` | (a) Sem IC bootstrap: ele mesmo pediu para tirar em 13/10; as tabelas trazem média e faixa mín–máx entre sementes. (b) Grade k ∈ {4, 10, 20, 40}, não só 4. Números novos: κ_B zero-shot 0,320 (gpt) e 0,258 (Gemini); κ_C 0,725 e 0,852; com k ≥ 10, κ_B sobe para 0,85 a 0,89 |
| R7 | Resultados p.3 | "No teste-oráculo no problema Meia-Lua os estimadores recuperam a métrica… Perceptron 100 % e NNLS 95,3 %" (ele apontou a contradição com os 86,7 % de V.7) | ✅ número corrigido | `bloco1_table_18_D_passo5_teto` + `bloco1_graphic_21_D_passo5_teto` | Não existe 100 %: o teto da métrica diagonal na meia-lua é 86,9 % (R²), 87,1 % (R³) e 87,3 % (R⁴) no Perceptron; 86,4 a 86,7 % no NNLS. A frase de V.4 deve dizer isso; a de V.7 (86,7 %) está certa |
| R8 | Resultados p.3 | **Tabela 5**: fidelidade em R³ e R⁴ "para os problemas A, B e C" | ✅ só A | `bloco1_table_04_A_passo4_R2_R3_R4__<modelo>` + `bloco1_graphic_08` | B e C são lineares e só recebem o w do A: R³/R⁴ não se aplica (decisão de 13/10). A tabela fica "Problema A" |
| R9 | Resultados p.3 | "Falta resultados e tabela relativa ao problema D" (e na reunião, 00:04:49 e 00:31:30: "completar o meia-lua na primeira parte do Bloco 1") | ✅ feito | `bloco1_table_10_D_passos_2_a_5` (zero, few-shot, teto em R²/R³/R⁴), `bloco1_table_11_D_passo2_vs_passo3`, `bloco1_table_12_AD_por_passo` (A × D), `bloco1_graphic_04_AD_passos_2_a_5_curva`, `bloco1_graphic_13_D_passos_2_a_4_simbolos`, `bloco1_graphic_14_D_passo6_svm` | O gpt colapsa em zero-shot na meia-lua nas 3 sementes (rotula tudo com uma classe); com exemplos deixa de colapsar. O Gemini não colapsa |
| R10 | Resultados p.3 | **Tabela 6** (nomenclatura e variante de prompt) e, na reunião, "corrigir nomes peso/altura" (00:20:34) | ✅ feito | `bloco1_table_14_vies_nomes_classe`, `bloco1_table_15_vies_nomes_atributo` (nomes corrigidos), `bloco1_table_16_vies_variantes_prompt`; figuras `bloco1_graphic_15/16/17/18` | Valores do texto (99,3 %, 82,7 → 68,0 %, 84,7 %, 68 %, 64,7 %) são da execução de julho; todos mudam. A tabela 14 traz a acurácia invariante à permutação nos pares invertidos |
| R11 | Resultados p.4 | **Figura 3** (variantes de prompt) e "precisa explicar como é feita cada variante"; na reunião (00:02:07): mandar por e-mail o que é padrão, geométrico, cot e tabular | ⛔ pendente (só o e-mail) | `bloco1_graphic_18_vies_variantes_prompt__<modelo>`; texto de 4 parágrafos a partir de `PROMPT_VARIANTS` (`src/dissertacao_mestrado.py`) | Não localizei rascunho nem envio dessa explicação; é a única pendência de entrega que não é tabela nem figura |
| R12 | Resultados p.4 | "estranho o valor se repetir" (61 % nas quatro estratégias em zero-shot, Tabela 7) | ✅ explicado | `bloco2_table_04_E_passo2_vs_passo3` (rodapé explica) | Não é erro: em zero-shot não há exemplos, logo não há estratégia; é UMA coleta por (modelo, semente), repetida nas quatro linhas. Valores novos: E1 61,3 % (gpt) e 54,4 % (Gemini) |
| R13 | Resultados p.4 | **Figura 4** (curva de aprendizado por estratégia) e **Tabela 7** (estratégia × n); renomear "aniso_x2" (00:30:03) | ✅ feito | `bloco2_graphic_03_E1_passo3_curva`, `bloco2_graphic_04_E1_passo3_estrategias`; `bloco2_table_01/02/03_E{1,2,3}_passo3_estrategia_x_k`, `bloco2_table_05_E_passo3_H4` | Nomes: E1 = [0,3; 1,5] (x2 cinco vezes), E2 = [1,5; 0,3], E3 = [1; 1] (ele falou "P1, P2, P3"; no sistema é E1/E2/E3). H4 refutada: E1 fácil 88,1 % vs difícil 77,7 % (gpt), 93,2 % vs 72,4 % (Gemini); aleatória 89,9 % e 92,9 % |
| R14 | Resultados p.5 | **Tabela 8**: LLM (40-shot) × baselines (k-NN, RL, SVM), "variantes P3 e P4 descritas na Tabela 3" | ✅ feito | `bloco2_table_06_E_passo6_baselines__<modelo>` + `bloco2_graphic_05_E1_passo6_baselines` | Baselines treinados nos MESMOS exemplos da estratégia mista, 3 repetições; os três especialistas E1/E2/E3 |
| R15 | Resultados p.5 | **Figura 6**: meia-lua, semente 42, gpt-4o-mini, SVM RBF × zero-shot do LLM | ⚠️ mudou | `bloco1_graphic_14_D_passo6_svm_seed{S}__<modelo>` | Na execução nova o gpt colapsa nas 3 sementes (D e F), não só em 123 e 7: o painel do gpt mostra o colapso; a figura útil é a do Gemini (não colapsa). O texto de V.7 ("semente 42 executa, 53,3 % → 68 %") precisa ser reescrito com os números de `bloco2_table_10_F_passo3_k` |
| R16 | Resultados p.5 | **Figura 7**: esquerda diluição ("três exemplos difíceis"), direita viés de ordem | ✅ feito | `bloco2_graphic_06_E1_passo4_diluicao`, `bloco2_graphic_07_E1_passo5_ordem`; `bloco2_table_07`, `bloco2_table_08` | São 4 difíceis fixos (não 3) + 0, 2, 4, 10, 16, 20 fáceis; ordens: classe 0 primeiro, classe 1 primeiro, embaralhada, alternada |
| M1 | Metodologia p.1 | **Tabela 1**: "? amostras" em D, E e F; centróides do E | ✅ números prontos | nenhum asset; só números | D 150; E 150, centróides (−1,5; 1) e (1,5; −1), σ 1,3, rótulo pelo especialista; F 150 (105 treino / 45 teste); G 100 (50/50), 70/30. ATENÇÃO: o A mudou em 15/10 para centróides (−2; −0,5) e (2; 0,5) (σ 1,2); B (−2; 1,5)/(2; −1,5) e C (−2; −1,5)/(2; 1,5) continuam, logo a rotação em relação ao A é 2,0 em B e 1,0 em C, não ±1,5 |
| M2 | Metodologia p.3 | "especialista/aprendiz" (nome dos papéis) | ✅ | — | O sistema já usa especialista/aprendiz em títulos, tabelas e log |
| M3 | Metodologia p.6 | "Este mesmo protocolo de divisão [70/30] é aplicado aos problemas E e F" | ⛔ divergência | — | F tem o split; E não: o LLM classifica os 150 − k restantes. Ou ele ajusta a frase, ou o E passa a ter split (decisão dele; já consta no e-mail de divergências) |
| M4 | Metodologia p.7 | "Falta explicar o teste com novo esquema de prompt, feedback imediato" | ⏸ adiado por ele (00:08:36) | — | "Deixa para depois"; só no problema real |

Pedidos do e-mail de 05/10/2026 (não estão em vermelho, mas são entrega):

| # | Pedido | Status | Asset |
|---|---|---|---|
| E1 | Problema G com R³ (x1·x2) e R⁴ (x1², x2²) na estimativa da métrica | ✅ | `bloco3_table_01_G_passos_2_e_4_R2_R3_R4`, `bloco3_table_06_G_passos_2_a_5`, `bloco23_graphic_02`; e, desde 14/10, os atributos também no PROMPT do Passo 6 (`bloco3_table_02/03`, coluna "Prompt") |
| E2 | G com nomes semânticos (peso/altura, Mulher/Homem) × neutros | ✅ | `bloco3_table_05_G_nomes_semanticos_vs_neutros`, `bloco3_table_04_G1_vs_G2`, `bloco3_graphic_03` (8 painéis) |
| E3 | Variantes de prompt: padrão, geométrico, cot, tabular (explicação) | ⛔ pendente | = R11 |

## 2. Números do texto dele que mudaram (para ele atualizar, sem tabela)

| Onde no texto | Está | Deve ficar (execução de 15/10 à noite) |
|---|---|---|
| V (abertura), IV.2 | 129.868 chamadas; "limite de dez ao mesmo tempo"; 0 respostas não interpretadas; flip 4,3 % (gpt) | 222.648 chamadas; 50 em paralelo; 5 respostas sem rótulo interpretável (0,002 %, todas do Gemini na variante cot, por tempo limite); flip a T = 0 de 1,19 % no gpt e 0 % no Gemini |
| IV.2 | few-shot n ∈ {4, 10} (especialista) | {4, 10, 20, 40} em A, D e G (Passo 3) e em B/C (Passo 4) |
| IV.7 | IC bootstrap 95 % com 10⁴ reamostras | removido a pedido dele (13/10); média, faixa mín–máx entre sementes e Cohen's d |
| V.1 | fidelidade "de 81 % a 86 %"; LLM acerta "25 % a 31 %" | gpt zero-shot: fidelidade 85,3 a 88,0 %; acurácia literal 24,7 a 28,0 % (invariante 72 a 75 %) |
| V.3 | κ_B 0,33, κ_C 0,75 | κ_B 0,320 (gpt), 0,258 (Gemini); κ_C 0,725 (gpt), 0,852 (Gemini); H1 não sustentada em B nos dois modelos, sustentada em C |
| V.4 | "média sobe de 84,9 % (3 atributos) para 86,2 % (4)"; oráculo meia-lua 100 % / 95,3 % | ver `bloco1_table_04`; teto da meia-lua 86,9 / 87,1 / 87,3 % |
| V.6 | ordem Aleatória 90,3 > Mista 88,0 > Fácil 81,1 > Difícil 75,1; Gemini fácil 93,5 vs difícil 78,1 | gpt (E1): aleatória 89,9 > fácil 88,1 > mista 86,7 > difícil 77,7; Gemini: fácil 93,2 > aleatória 92,9 > mista 91,1 > difícil 72,4 |
| V.7 | gpt colapsa "em 2 das 3 sementes (123 e 7)"; semente 42 "53,3 % → ≈ 68 %" | gpt colapsa nas 3 sementes em zero-shot (D e F); com exemplos classifica (ver `bloco2_table_10`) |
| FundamTeo, Algoritmo 1 | pseudocódigo validado na reunião (00:43) | em 15/10 o código ganhou: w e α divididos por ‖w‖ ao fim de cada época; melhor esforço pela época com menos pontos mal classificados; margem bruta mínima registrada (†). O critério de parada (margem relaxada com α) não mudou. Documentos atualizados: `algoritmo_codigo_executado_pseudocodigo.docx` e `algoritmo_perceptron_animado.html` |

## 3. Como mostrar para ele, organizado

Princípio: seguir a ordem e a numeração DO TEXTO DELE (Tabela 2 … Tabela 8, Figura 1 … Figura 7,
Tabela 1 da Metodologia), não a numeração do sistema. Ele abre o pacote com o PDF dele ao lado e
preenche cada vermelho na sequência.

**Um único PDF, `entrega_orientador_2026-10-16.pdf`, dentro da pasta da execução, com 4 partes:**

1. **Capa + índice de correspondência** (1 página): tabela com três colunas: "no seu texto" (Tabela 2,
   Figura 1, …), "o que entrego" (nome do arquivo), "status" (feito / mudou / pendente). É a tabela da
   seção 1 deste levantamento, enxuta.
2. **Uma página por item em vermelho, na ordem do texto dele** (R1 a R16, M1): no alto, a frase
   dele em vermelho, copiada; no meio, a tabela ou figura (PNG `_gray`, versão para impressão; o
   gpt-4o-mini primeiro, o Gemini em seguida quando ele pediu "réplica de contraste"); embaixo, uma
   legenda de uma linha dizendo o que é cada coluna e, quando houver, a "observação para ele" da
   seção 1 (dois ou três números no máximo). Sem texto analítico: o texto é dele.
3. **Números para atualizar no texto** (1 página): a tabela da seção 2, "está / deve ficar".
4. **Perguntas e avisos** (1 página, lista curta): (a) split no Problema E (M3); (b) grade de k {4, 10, 20, 40}
   em vez de {4, 10}; (c) Tabela 5 só com o A; (d) gpt colapsa nas 3 sementes da meia-lua; (e) centróides
   do A mudaram para (−2; −0,5)/(2; 0,5) e por quê (w2 não identificável com x2 = 0); (f) Perceptron
   com as três mudanças de 15/10 e o †; (g) feedback imediato continua adiado.

**Junto com o PDF, num ZIP:** os `.tex` das tabelas usadas (`tabelas/`), os PNG `_gray` e `_color`
das figuras usadas e os CSVs de onde saem os números. Ele pediu para "mandar as tabelas" (00:04:49):
o `.tex` entra direto no Word dele via pandoc, ou ele copia os valores.

**Por e-mail, separado e curto** (regra das memórias: 1 a 2 frases por ponto, sem tabelas):
a explicação das 4 variantes de prompt (R11/E3), que é o único pedido que não é tabela nem figura,
e o aviso de que os números do texto mudaram porque o A foi inclinado e o Perceptron foi corrigido.

Ordem sugerida de produção: (1) e-mail das variantes de prompt hoje (10 minutos, texto de
`PROMPT_VARIANTS`); (2) PDF por script a partir da execução de 15/10 à noite (as tabelas e figuras já
existem; falta só o script que monta o PDF na ordem dele); (3) ZIP; (4) e-mail de envio com o link.

## 4. O que NÃO entra agora

- Bloco 3 (peso × altura): ele disse que ainda vai escrever (00:06:48 e 00:56) e que vai juntar os
  resultados do NeuroSolutions (RBF e MLP). As tabelas `bloco3_table_01..06` já existem para quando
  ele pedir; levar só se ele perguntar.
- Feedback imediato (M4): adiado por ele.
- Texto: nada de parágrafo analítico; só legendas de uma linha.
