# Nomenclatura por passos (decisão do autor, 13/10/2026)

Regra: letra = problema; cada problema tem a SUA sequência de passos, numerada de 1 em diante, só com os passos que existem nele. Os nomes "Fase A", "Fase B/C", "Fase E", "pipeline externo" e "protocolo único da Fase A" DEIXAM DE SER USADOS em qualquer texto visível (títulos, legendas, rodapés, log, documentação). Identificadores internos de código (funções, constantes, colunas, tags e nomes de arquivo) foram renomeados pelo mesmo critério em 13/10/2026 (camada 2; mapa em renomeacao_passos_2026-10-13.md); em 14/10/2026 os parâmetros `n_shot` viraram `k_exemplos` e `EXAMPLE_ORDER_N_SHOTS` virou `K_ORDEM_EXEMPLOS`.

## Problema A (gaussianas lineares)
1. Gerar os 150 pontos (centróides (−2; −0,5) e (2; 0,5), σ = 1,2; até 14/10/2026: (−2, 0) e (2, 0)).
2. O LLM rotula os 150 pontos em zero-shot.
3. O LLM rotula de novo com k exemplos do rótulo real no prompt, k ∈ {4, 10, 20, 40}, 3 sorteios por k.
4. Estimar a métrica Ŵ (Perceptron e NNLS, em R², R³ e R⁴) a partir de cada conjunto de rótulos do LLM; medir fidelidade.
5. Calcular o teto: W ajustado diretamente ao rótulo real.
6. Eleger a melhor métrica (maior fidelidade média) para a transferência.
Vieses (sobre o Passo 2 do A e os Passos 3 e 4 de B/C): nomes de classe, ordem das classes, variantes de prompt com k ∈ {0, 10}; nomes de atributo só zero-shot.

## Problemas B e C (transferência)
1. Gerar os 100 pontos com os centróides rotacionados (B horária, 2,0 em relação ao A; C anti-horária, 1,0; coordenadas fixas (−2; ±1,5)/(2; ∓1,5)).
2. O LLM rotula os 100 pontos em zero-shot (não roda na configuração em que o LLM colapsou no Passo 2 do A).
3. Aplicar cada métrica do A (só o W; centróides locais) e medir consistência, Cohen's κ e F1-Macro; destacar a métrica eleita no A (tabelas 06 a 08 e figura 11; o CSV bloco1_abc_passos_2_a_4, a figura 10 e as hipóteses H1/H2 usam a métrica zero-shot do A).
4. O LLM rotula de novo com k exemplos rotulados pela métrica do A (k ∈ {4, 10, 20, 40}); medir se a consistência sobe (H2).

## Problema D (meia-lua, LLM como especialista)
1. Gerar os 150 pontos da meia-lua.
2. O LLM rotula em zero-shot.
3. O LLM rotula com k exemplos do rótulo real, k ∈ {4, 10, 20, 40}.
4. Estimar a métrica em R², R³ e R⁴ e medir fidelidade.
5. Calcular o teto.
6. Treinar um SVM RBF nos rótulos do LLM para comparar a fronteira.

## Problemas E1, E2, E3 (especialista linear, LLM como aprendiz)
1. Gerar os 150 pontos; o especialista (W conhecido) rotula todos.
2. O LLM classifica os 150 pontos em zero-shot (uma coleta, compartilhada pelos três especialistas).
3. O LLM recebe k exemplos rotulados pelo especialista, escolhidos pelas estratégias fácil, difícil, mista e aleatória, k ∈ {4, 10, 20, 40}, e classifica os 150 − k restantes.
4. Diluição: 4 exemplos difíceis fixos mais 0 a 20 fáceis (só E1).
5. Ordem dos exemplos: quatro ordenações do mesmo conjunto (só E1).
6. Baselines: k-NN, regressão logística e SVM treinados nos mesmos exemplos de cada estratégia (todas, todos os k), só no primeiro modelo (não usam o LLM); tabela e figura mostram a estratégia mista.

## Problema F (meia-lua, LLM como aprendiz)
1. Gerar os 150 pontos e dividir 70% de treino e 30% de teste (105 e 45 pontos).
2. O LLM rotula os 105 de treino em zero-shot; estimar a métrica em R², R³ e R⁴ só para medir a fidelidade.
3. O LLM recebe k exemplos do treino rotulados pelo rótulo real, k ∈ {0, 4, 10, 20, 40} (k = 0 é o LLM classificando o teste sem exemplos), e classifica os 45 de teste; exemplos e ponto de teste só com x1, x2 (o F não herda R³/R⁴ da métrica do Passo 2).
4. Perceptron treinado nos mesmos exemplos, para comparar.

## Problemas G1 e G2 (peso × altura)
1. Carregar as 100 pessoas e dividir 70 de treino e 30 de teste (a mesma divisão para G1 e G2).
2. O LLM rotula os 70 de treino em zero-shot.
3. O LLM rotula os 70 com k exemplos do rótulo real, k ∈ {4, 10, 20, 40}.
4. Estimar a métrica em R², R³ e R⁴ a partir de cada conjunto de rótulos; medir fidelidade; calcular o teto (gravado com o valor de regime `passo5_teto`, numeração do A/D).
5. Eleger a melhor métrica (maior acurácia vs rótulo real no treino).
6. O LLM recebe k exemplos do treino rotulados pela métrica eleita, k ∈ {0, 4, 10, 20, 40} (k = 0 é o LLM classificando o teste sem exemplos), e classifica os 30 de teste (os mesmos pontos nas duas variantes de nome; mesma semente seed + rep do Passo 3, logo os mesmos pontos sorteados lá, agora rotulados pela métrica eleita). Repetido com 2, 3 e 4 atributos no prompt (x1, x2; + x1·x2; + x1², x2²), fator independente da métrica eleita (14/10/2026); k = 0 é uma coleta só.
7. Perceptron treinado nos mesmos exemplos e no mesmo espaço que o LLM viu, para comparar.
Repetido para cada combinação de nomes de classe (Mulher/Homem, A/B) e de atributo (peso/altura ou kg/m, x1/x2).

## Oráculo sobre E3, E1 e E2
1. Rotular as gaussianas do Problema E pelo centróide mais próximo com o W conhecido: [1, 1] (E3, Problema 1), [0,3; 1,5] (E1, Problema 2), [1,5; 0,3] (E2, Problema 3). Mesmo conjunto, só a métrica muda (14/10/2026; até 13/10 o euclidiano rodava sobre o B).
2. Rodar Perceptron e NNLS sobre esses rótulos.
3. Comparar o W recuperado ao verdadeiro: acurácia e cosseno.

## Equivalência com os nomes antigos
- "Fase A" (A e D) = Passos 2 a 5 do A e do D ("protocolo único da Fase A" = Passos 2 a 5; "Fase A zero-shot" = Passo 2; "few-shot da Fase A"/"few-shot GT" = Passo 3; "teto" = Passo 5; eleição = Passo 6 do A).
- "Fases B e C" = Passos 2 a 4 de B e C (zero-shot = Passo 2; "transferência"/"consistência" = Passo 3; "few-shot H2"/"exemplos da métrica" = Passo 4).
- "Fase E" (Bloco 2) = Passos 2 e 3 de E (zero-shot = Passo 2; estratégias = Passo 3); diluição = Passo 4; ordem = Passo 5; baselines = Passo 6.
- "Fase A externa" = Passo 2 do F; Passos 2 a 4 do G. "Fase E externa" = Passo 3 do F; Passo 6 do G. "LLM vs Perceptron" = Passo 4 do F; Passo 7 do G. "Eleição do especialista" = Passo 5 do G.
- "Bloco 1/2/3" continuam (agrupam problemas). Hipóteses: H1 e H2 nos Passos 3 e 4 de B/C; H3 e H4 no Passo 3 de E; H5 nos Passos 4 a 6 do A.

## Forma de escrever
"Problema A, Passo 3" / "Passo 3 do Problema A" / em títulos curtos "A, Passo 3". Para um asset que cobre vários passos: "Problema A, Passos 2 a 5". Nunca "fase".

## Nota (13/10/2026)
nomes de código: ver renomeacao_passos_2026-10-13.md (camada 2: identificadores, colunas, valores de regime, tags, CSVs, gráficos, painéis, slugs, testes e ferramentas renomeados pelo mesmo critério).
