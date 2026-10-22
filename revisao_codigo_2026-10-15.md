# Revisão completa de 15/10/2026: código, execução, tabelas, figuras, documentação e testes

Execução auditada: `resultados/execucao_2026-10-15_10-40-22_completa` (222.648 chamadas, 5 malformadas, exit 0, 2h57).
Seis frentes de revisão, somente leitura, com reprodução numérica onde cabia. Scripts de reprodução no scratchpad da
sessão (`repro.py`, `instr.py`, `verifica.py`, `test_propostas_2026-10-15.py`).

## 1. Bloqueadores (mudam números ou conclusões)

### B1. Perceptron: "γ viável" pode ser solução degenerada (‖W‖ → 0, α absorve as violações)
`src/relaxed_perceptron.py:247-283`. O teste de margem usa a margem RELAXADA (`+ λ·αi`) e o fator `1 − ηγ/‖w‖` encolhe
w a cada correção; quando ‖w‖ cai a ~10⁻³, um único αi = η já satisfaz qualquer ponto e a época termina com "0 violações",
embora dezenas de pontos tenham margem bruta negativa. Reproduzido com os rótulos zero-shot do gpt-4o-mini no A (seed 42):
W = [0; 0,00065], γ = 0,0499, fidelidade 0,753, com 37 de 150 pontos de margem negativa; W = [1, 1] dá 0,800 e o NNLS
0,847. Na execução: 12 das 495 linhas do Perceptron em `bloco1_ad_passos_2_a_5` são degeneradas (γ > 0 e ‖W‖ < 0,05; 8 no
zero-shot, 4 no few-shot), nenhuma no G. Consequências: a direção "W só de x2" do gpt no A é artefato; o κ zero-shot em B
do gpt usa esse W; o marcador † (γ = 0) inverte o sinal de alerta nessas linhas (as de γ = 0 têm ‖W‖ ≈ 1 e são as
saudáveis). Correção: declarar viabilidade pela margem BRUTA (ou exigir fidelidade = 1 na época de parada) e/ou renormalizar
w a ‖w‖ = 1 ao fim de cada época escalando α; gravar no CSV a margem mínima bruta/‖w‖; marcar † por ‖W‖ < limiar. Exige nova
execução das estimativas (0 chamadas: os rótulos estão em `llm_interactions`; dá para reestimar offline).

### B2. Log: "RESUMO DOS PROBLEMAS A, B E C" mistura o Passo 4 na consistência média e contradiz H1
`src/relatorios.py:626-639` (`print_abc_resumo`): `df_ab` não filtra `k_passo4_bc == 0`; a "Consistência Média B" sai 88 %
(todos os k) em vez de 70,5 % (Passo 3) e o veredito "✓ MANTÊM alta consistência" contradiz "H1 NÃO SUSTENTADA em B"
impresso 150 linhas depois. Tabelas 06 a 09 estão certas; só o log. Correção: filtrar k = 0 (ou rotular "Passos 3 e 4").

### B3. Figura 15 (viés de ordem das classes): painel "Fidelidade A, 10-shot" é cópia do zero-shot
`src/plots.py:1083-1152`: `fidelity_problem_a` é idêntica para todo k (a métrica do A é estimada uma vez, no Passo 2);
o painel "10-shot" de fidelidade não existe experimentalmente. Correção: um único painel de fidelidade (zero-shot) e os
painéis de consistência B/C por regime.

### B4. Versões em preto e branco que perdem a codificação
`src/estilo_cinza.py:563-567` reatribui o traço por série: em `bloco23_graphic_01_*_gray` a legenda "cheia = vs rótulo real,
tracejada = vs especialista" deixa de valer; `:503-509` zera o alpha e ignora a hachura própria: em `final_graphic_08_gray`
as barras de B e C ficam idênticas. Correção: preservar linestyle quando a figura já o usa como codificação (chave só
cor+marcador) e preservar hachura/alpha explícitos.

## 2. Comportamentos questionáveis e textos errados (⚠️)

- Timeouts tratados como rate limit (`dissertacao_mestrado.py:1084,1148-1163,1216-1222`): 5 pontos da variante cot do Gemini
  custaram ≈ 61 min (5 × 90 s + backoff até 198 s). Correção: para timeout, no máximo 2 tentativas e espera curta; ou
  `API_TIMEOUT_S` maior só para cot.
- `bloco_protegido` de G1/G2 envolve todos os modelos × classes × sementes (`:5149,5204`); uma exceção descarta o G inteiro.
  Mover o `with` para dentro do laço de variantes (como no F).
- Baselines do Passo 6 do E usam `random_state = seed` e o LLM `seed + rep`: na estratégia aleatória só a rep 0 tem os
  mesmos exemplos (`:2502` vs `:4833`). Laçar por rep ou declarar no rodapé da tabela 02_06.
- Linha "acaso = 0,5" nos painéis de κ (`plots.py:672,1335`): para κ o acaso é 0.
- `final_graphic_08` mistura Passo 3 e Passo 4 de B/C (sem filtro de k) sob o título "Passo 3" (`plots.py:505-570`).
- `print_sumario_estatistico` rotula como "B/C Passo 3" blocos com k > 0 (Passo 4) e repete a fidelidade do A por k como se
  variasse (`relatorios.py:741-906`); "Zero-shot vs 10-shot" é Passo 3 vs Passo 4 (H2) sem dizer.
- `final_graphic_06` e `final_graphic_04_barras`: W bruto (Perceptron ~6e-4 invisível ao lado do NNLS); estratégias do E
  em inglês; histogramas de margem com W bruto; faixa em branco sob o supertítulo.
- `bloco23_graphic_02/03`: títulos com `problem_name` cru; "Acurácia (vs rótulo real)" é da MÉTRICA; configurações
  colapsadas somem sem aviso; só a 1ª semente, sem dizer.
- Legendas cobrindo dados: `bloco1_graphic_10` (Consistência), 15, 16, `bloco23_graphic_04`.
- Rodapés com datas de changelog nas tabelas 02_10 e 03_02 (`gerar_tabelas.py:778,866`); "n" vs "k" inconsistente
  (03_03 cabeçalho, 01_09 rodapé, final_02 rodapé); final_02 cita "Passos 2 a 4" para linhas do Passo 3; `final_table_04`
  não explica os regimes `zero_shot`, `exemplos_rotulo_real`, `diluicao` e ordenações, e o F sai partido por semente.
- `tests/test_nomenclatura_g1_g2.py:56-64`: assert trivial (`accuracy >= 0`); `tests/test_g_passo6_atributos_prompt.py:107`
  não verifica o espaço do Passo 7; `tests/test_apenas_flag.py:115-129` passa com o esquema antigo do oráculo. Sem testes
  para lote, figuras em processos e cronômetro (25 testes prontos no scratchpad, todos passando).
- Documentação: "com 50 não há medida" (CLAUDE.md L482/L560, README L88) quando há duas completas com 50; oráculo "sobre
  B/E1/E2" em CLAUDE.md L362/778/1006 e glossário L172; "rotação ±1,5 de mesma magnitude" em B/C (CLAUDE.md L874/L1016,
  docstrings de `data_problems.py` e do runner) deixou de valer com o A inclinado (B gira 2,0; C 1,0); "361 testes" → 365;
  H4 citado da completa de 08/10; falta `ATRIBUTOS_PROMPT_PASSO6_G` na tabela de constantes e `n_atributos_prompt=` na
  descrição da tag; `problemas_tabela.html` ainda cita a completa de 14/10 como última; docstring do módulo do runner com
  "centroides ±2.0 horizontal"; `resultados.py:79-81` contraditório sobre `collapsed`; `gerar_tabelas.py:70-72` diz "sem
  fallback" e lista fallbacks; ramos mortos de "execução anterior a 10/10/2026".

## 3. Divergências conhecidas entre o código e os Algoritmos 1 e 2 do orientador (confirmadas + novas)
Condição de correção (y ≠ ŷ no texto; qualquer violação de margem no código); busca em γ (γ = 0 e multiplicativa no texto;
0,1 e aditiva 0,05 no código); melhor esforço (só no código); bisseção (o texto guarda w* mesmo inviável; o código restaura);
ordem das amostras permutada a cada época (código); salvaguardas `factor ∈ [0,1]` e `α ≤ 10⁶` (código); significado de γ
devolvido (maior γ candidato com época sem violações relaxadas, não a margem geométrica de w*).

## 4. O que está correto (conferido)
Tabelas: as 71 regeneradas byte a byte iguais às da execução e ~900 células recalculadas por script independente sem
divergência; `texto_h1`, `texto_h4`, `malformadas_e`, auditoria por modelo e por regime, `final_cross_linearity` conferem.
Runner: tags, caches, reaproveitamento do zero-shot (E e Passo 6 do G), `n_calls`, ordem das linhas dos CSVs, colapsos,
lote com cliente e semáforo compartilhados, 1.807 arquivos iguais aos de 14/10, nomes na convenção, 0 avisos de legenda.
Estimadores: NNLS, métricas (κ, F1-Macro), `colapsou`, dados (A reproduz bit a bit o CSV; base real 50/50, classe 1 =
Homem, kg/m por regra de três), parser (26 casos), baselines. Testes: 365 passam.

## 5. Status (noite de 15/10/2026): item 1 (Perceptron) aplicado

Decisão do autor ("Sim", 15/10/2026): aplicar só B, C e D em `relaxed_perceptron.py`; a alternativa A (viabilidade
pela margem BRUTA) foi descartada porque altera o critério de parada do texto do orientador (margem relaxada com α).
- **B** (fim de cada época): `w ← w/‖w‖`, `α ← α/‖w‖`. Só a direção de w define o centróide mais próximo; a folga
  deixa de "absorver" a margem quando ‖w‖ evapora.
- **C**: melhor esforço = época com menos pontos mal classificados (margem bruta < 0), não menos violações relaxadas.
- **D**: `ULTIMA_MARGEM_MINIMA`/`margem_minima_` = min_i (d_W(rival) − d_W(correto))/‖w‖ da solução devolvida; coluna
  `margem_minima` em `bloco1_abc_passos_2_a_4`, `bloco1_ad_passos_2_a_5` e `bloco3_g_passos_2_a_5`; † nas tabelas quando
  γ = 0 OU margem_minima < 0 (`marca_gamma`, `_g0`, `algum_g0`, `frac_gamma0`, `NOTA_GAMMA`).
- Validação: oráculo E3/E1/E2 100 % (Perceptron) nas 3 sementes; caso degenerado (gpt, A, seed 42, zero-shot):
  W = [0,15; 0,99], γ = 0 (melhor esforço honesto), margem mínima −4,96, fidelidade 0,860 (antes 0,753; NNLS 0,847;
  W = [1, 1] 0,800). Suíte: 396 testes (6 novos em `tests/test_perceptron_2026_10_15.py`); o teste de regressão de
  07/2026 passou a exigir cosseno > 0,99 com o W armazenado (medido 0,9977).
- Reestimação offline (`--reestimar-de`, pasta `_completa_reestimada`, `reestimacao_lacunas.csv`): ver README.
- Reestimação medida (`execucao_2026-10-15_19-32-06_completa_reestimada`, 23 min, exit 0): A/D 13 → 0 linhas degeneradas;
  fidelidade média 0,938 → 0,947 (sobe em 249 linhas, cai em 2 por ≤ 0,021); teto 0,916 → 0,922; os 6 W zero-shot da
  configuração base do A eram todos degenerados e passam a W ≈ [0,1–0,33; 0,94–0,99] (x2 dominante persiste; fidelidade do
  gpt 0,802 → 0,878, do Gemini 0,873 → 0,882); H1 em B continua NÃO sustentada (κ gpt 0,292 → 0,310; Gemini 0,268 → 0,258),
  em C sobe (0,742 → 0,764; 0,826 → 0,852); G Passos 2 a 5: fidelidade 0,919 → 0,922, teto 0,905 → 0,910. Lacunas: 306
  coletas (16.700 pontos, 7,5 %) do Passo 4 de B/C, dos vieses com k = 10 e do Passo 6 do G foram aproximadas; esses números
  (H2 inclusive) só valem depois de `--coletar-lacunas` (≈ US$ 1,1) ou de uma nova completa.
