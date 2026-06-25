# Plano de Trabalho — Reunião 20/05/2026

## Contexto da reunião

Reunião com o orientador Prof. Raul Fonseca Neto em 20/05/2026, com duração aproximada de 59 minutos. Apresentei a apresentação Beamer consolidada (84 slides) da execução de produção `execucao_2026-05-14_21-43-18`, organizada em 3 blocos e nos problemas A–G:

- **Bloco 1 (LLM como FONTE):** Problemas A (treino), B/C (consistência, rotações ±1,5), D (meia-lua, não-linear). Oracle/recuperação de W, augmentação R2/R3/R4, vieses de ordem/nome/prompt.
- **Bloco 2 (LLM como APRENDIZ):** Problema E (linear, perito artificial) e Problema F (meia-lua, perito = ground truth). Estratégias easy/hard/mixed/random, múltiplos peritos, diluição, baselines clássicos.
- **Bloco 3 (caso REAL):** Problema G — peso × altura (base do orientador, 100 amostras, fronteira elíptica).

A avaliação geral foi muito positiva (~2960s: *"os resultados, o trabalho que você fez está excelente, quer dizer, já dá insumo para escrever um artigo... até a dissertação"*). O foco agora é **refinar, eliminar inconsistências e começar a escrever o texto**. Após a reunião o orientador enviou um e-mail com refinamentos das features e a lista de artigos de referência.

Este plano consolida o que foi discutido na transcrição (com timestamps em segundos para rastreabilidade) e no e-mail.

## Fontes

- E-mail do orientador (após a reunião): `emails.txt`
- Diarização da reunião: `reuniao_raul_20_05_2026_20260610_213509_diarizacao.json`

## Participantes

- **Prof. Raul Fonseca Neto** — orientador (SPEAKER_00 na diarização)
- **George Gomes da Silva** — mestrando (SPEAKER_02)
- Terceiro participante esporádico (SPEAKER_01)

## Próxima reunião

Marcada para **quarta-feira seguinte (27/05/2026)** — *"pode até marcar então pra quarta que vem? Vamos, vamos"* (~3511s). O coorientador (Yuri/Saulo) participará; o orientador imprimirá o material, fará um refinamento e o repassará a ele (~3014s, 3028s).

---

## PARTE A — Nomenclatura e organização

### 1. Padronizar a nomenclatura dos problemas e blocos
- **O que fazer:** garantir nomenclatura única e consistente — Problemas A, B, C, D (Bloco 1), E, F (Bloco 2), G (Bloco 3). Evitar misturar "fase" com "problema".
- **Citação (~546-816s):** *"É só definir melhor esses problemas, né? Porque você começa falando de B, C e D, aí depois já fala de E, já fala de x1 domínio."* / *"Tem que ter uma nomenclatura única."* / *"organiza melhor tanto a nomenclatura quanto..."*
- **Status:** **em andamento** — a apresentação já foi convertida de "Fase X" para "Problema X" e o peso×altura passou a ser o **Problema G**. Confirmar que o texto/código também seguem.

### 2. Limpar o Bloco 3 de slides que não pertencem a ele
- **O que fazer:** remover do Bloco 3 conteúdos que não dizem respeito ao problema peso × altura (ex.: meia-lua aparecendo fora de lugar). Manter o bloco focado apenas no Problema G.
- **Citação (~2896-2910s):** *"esse é só o problema do amigo né, não ia misturar as coisas"* / *"eu tenho que dar uma revisada aqui que tem algumas sujeiras, as que não têm nada a ver aqui."*

---

## PARTE B — Bloco 1 (LLM como FONTE)

### 3. Mostrar o W aprendido nas tabelas/slides
- **O que fazer:** exibir explicitamente o vetor $W$ estimado pela otimização inversa nos slides/tabelas em que hoje só aparece a fidelidade.
- **Citação (~644-656s):** *"O W você não tem ali não, que ele aprendeu."* — George: *"O que ele aprendeu eu tenho, mas não tá aqui na tela... mas eu posso colocar."*

### 4. Investigar a fidelidade "baixa" no Problema A
- **O que fazer:** verificar por que o Problema A — quase linearmente separável, com dois centros bem separados — tem fidelidade **menor** do que o esperado. Deveria ser dos mais fáceis de acertar, até com métrica euclidiana. Confirmar se o resultado/data está correto.
- **Citação (~605-669s):** *"a fidelidade é menor ou por quê?"* / *"Ele é quase linearmente separável... Ele era pra ser mais fácil de acertar, né? Até numa métrica euclidiana."* / *"no primeiro problema não tem muita distorção... são dois centros bem separados."*

### 5. Oracle / recuperação de W na meia-lua (Problema D)
- **O que fazer:** estender a validação Oracle (recuperação de W conhecido) também à meia-lua, usando um **W que aproxime** a fronteira não-linear — não gerar a meia-lua a partir de um W (isso não faz sentido, pois a geração é não-linear), mas mostrar que existe uma mudança de métrica que aproxima.
- **Citação (~391-502s):** *"no Oracle com o recovery fazer do meia-lua também."* / Orientador: *"você pode ter um W que aproxima, você pode ter uma mudança ali de métrica que aproxima um pouco da meia-lua."*
- **Observação:** George marcou como item a fazer durante a reunião.

### 6. Mapas de erro maiores, um por slide
- **O que fazer:** apresentar os mapas de acerto/erro (onde a métrica concorda/discorda do LLM em B e C) em tamanho maior, um por slide.
- **Citação (~1010-1019s):** *"é isso é interessante fazer"* / *"posso depois até colocar para ficar maior né, colocar um em cada slide."*
- **Status:** **parcialmente feito** (slide de mapa de acertos/erros B/C, seed 42).

---

## PARTE C — Bloco 2 (LLM como APRENDIZ)

### 7. Plotar a superfície de separação com SVM gaussiano na meia-lua
- **O que fazer:** no problema da meia-lua, adicionar a **superfície de separação de um SVM gaussiano** para comparar com a fronteira de decisão induzida pelo LLM. Há artigos que fazem esse tipo de gráfico (comparação de decision boundaries).
- **Citação (~1285-1322s):** *"no outro problema da meia-lua você deveria colocar também a superfície de separação usando a SVM gaussiana, você tem como plotar isso?"* / *"a gente compara com a superfície de separação da LLM, né? Tem uns artigos que fazem esses gráficos."*

### 8. Esclarecer a coluna "acurácia vs métrica" no Problema F (meia-lua)
- **O que fazer:** explicar melhor (ou repensar) a coluna que compara o LLM com a métrica $\hat W$ aprendida em zero-shot. Causou confusão na reunião: a princípio pareceu **circular** (o W é ajustado em cima dos rótulos do LLM, então poderia dar ~100% trivialmente). Conclusão final do orientador: a coluna **faz sentido** se interpretada como "o quanto a métrica reproduz o LLM" — ela não chega a 100%, indicando que o aprendizado da métrica não foi perfeito. Deixar essa interpretação explícita no texto.
- **Citação (~1784-1809s):** *"essa coluna não faz sentido não, porque o W tá aprendendo em cima da coisa."* — e (~2164s): *"isso aqui faz sentido sim. Quer dizer que o aprendizado não foi 100%."*

### 9. Explicar a seleção dos exemplos few-shot
- **O que fazer:** documentar como os exemplos few-shot foram escolhidos no experimento da meia-lua. O ideal é seleção **random**; explicar isso no texto.
- **Citação (~1913-1991s):** *"Se eles são escolhidos randomicamente, isso aí você tem que explicar."* / *"Eu acho que o ideal aí é random."*

### 10. Re-rodar o Problema F (meia-lua, aprendiz) com múltiplas repetições
- **O que fazer:** re-executar o experimento, em especial o **n_shot = 20** (resultado não-monotônico estranho: subiu para ~89% em n=10 e caiu para ~44% no Perceptron baseline em n=20). Rodar **mais de uma vez** para tirar média e evitar resultados anômalos de execução única.
- **Citação (~2050-2126s):** *"esse salto pra 89 e depois cai de novo pra 44... tem que refazer esse experimento aqui porque tá muito estranho."* / *"Roda de novo esse 20"* / *"é interessante você rodar não só uma vez, pra ter uma média."*

### 11. Explicar F1 e Cohen's Kappa no texto
- **O que fazer:** incluir na dissertação a definição (com fórmula) das métricas **F1-score** e **Cohen's Kappa**. O orientador não reconheceu o Kappa e pediu que essas métricas sejam explicadas.
- **Citação (~2227-2257s):** *"esse capa, eu não sei o que é esse capa... essas coisas têm que ser explicadas na dissertação, não custa nada a formulinha."*

---

## PARTE D — Bloco 3 (Problema G — peso × altura)

### 12. Confirmar a forma das features aumentadas (R3 e R4)
- **O que fazer:** garantir que a augmentação siga exatamente a forma indicada no e-mail:
  - **3 features:** $x,\ y,\ x\cdot y$ → equação de uma **hipérbole**.
  - **4 features:** $x,\ y,\ x^2,\ y^2$ → equação de uma **elipse** (classificador ótimo do peso × altura).
- **Citação (e-mail):** *"quando se utiliza 3 features deve considerar x, y e x.y que dá a equação de uma hipérbole. Para 4 features, x, y, x² e y² que dá a equação de uma elipse."*

### 13. Rodar peso × altura com few-shot (não só zero-shot)
- **O que fazer:** o experimento R2/R3/R4 da fidelidade no peso × altura foi feito **apenas em zero-shot**; rodar também com **few-shot**. O orientador espera bom desempenho do LLM no homem/mulher com poucos exemplos.
- **Citação (~1375s):** *"agora eu quero ver no problema de homem e mulher, eu acho que ele com os shots vai ter desempenho bom."* / (~2412-2423s) *"ah, não fiz few-shot aqui... é interessante fazer também."*

### 14. Investigar a queda de acurácia REAL com 4 features
- **O que fazer:** revisar/re-rodar o resultado em que a **acurácia vs ground truth cai** ao usar 4 features (R4) no peso × altura. Como é **erro de treinamento** (não há conjunto de teste separado), um modelo mais complexo deveria ajustar melhor (ou igual), não pior — com uma curva quadrática é praticamente impossível errar no treino de um problema elíptico simples. Rodar também o **Perceptron** aqui para comparar.
- **Citação (~2547-2828s):** *"Não tá fazendo muito sentido cair essa acurácia com mais uma feature... só tá um erro de treinamento... talvez você roda o Perceptron aqui."* / Discussão do dilema **viés–variância** (mas ressaltando que aqui é só treino).

### 15. Reportar a quantidade de amostras classificadas erradas
- **O que fazer:** como o peso × altura é um problema pequeno e específico (100 amostras), reportar o **número absoluto de amostras erradas** (ex.: 3, 4 ou 5) além das porcentagens.
- **Citação (~2752-2790s):** *"você devia botar talvez também, como você tem um problema bem específico, a quantidade de amostra que errou — três, quatro ou cinco."*

### 16. Adicionar a tabela faltante do prompt neutro (x1, x2)
- **O que fazer:** incluir a tabela do resultado com **prompt neutro** ($x_1, x_2$, sem prior semântico) no peso × altura — hoje existe nos dados mas não foi colocada no slide. É importante mostrar que, num problema sem viés semântico, **aumentar features não melhora** (e a acurácia real fica pior que chutar).
- **Citação (~2838-2887s):** *"não deu... tenho aqui mas acho que não coloquei."* / *"interessante colocar aqui que é um problema que não tem viés semântico — mesmo aumentando as features, não melhora... fica pior do que chutar."*

### 17. Experimento com nomes de classe neutros (A/B) no peso × altura
- **O que fazer:** repetir a classificação do peso × altura perguntando por **classes "A" e "B"** em vez de "homem" e "mulher", para isolar o efeito do prior semântico nos **nomes das classes** (complementa o teste de features neutras x1/x2).
- **Citação (e-mail):** *"tenta fazer um experimento perguntando a classificação para A e B em vez de homem e mulher."*

---

## PARTE E — Escrita da dissertação e referências

### 18. Começar a escrever o texto (dissertação/artigo)
- **O que fazer:** com os experimentos essencialmente prontos, **iniciar a escrita textual**. O orientador enfatiza que já há insumo suficiente; agora é refinar, fechar pontas soltas e redigir. Usar a estrutura experimental dos artigos de referência como modelo.
- **Citação (~2960-3184s):** *"o trabalho que você fez está excelente... já dá insumo para escrever um artigo, até a dissertação."* / *"a gente já procura começar a botar isso numa forma de texto... começar a escrever textualmente mesmo."*
- **Título de referência:** *"Explicando o raciocínio das LLMs através do aprendizado inverso."*

### 19. Conclusões principais a destacar no texto
- Dependência **semântica e de contexto** do aprendizado in-context (prompt importa).
- Certa **estabilidade** do critério decisório do LLM.
- **"Cegueira" no zero-shot** — sem contexto, a classificação é praticamente uma loteria.
- O método precisa de algumas amostras para o LLM criar identificação do problema, mesmo com bilhões de parâmetros.
- **Citação (~2910-3184s):** *"ficou muito claro a questão semântica, a questão do contexto, a questão de uma certa estabilidade da LLM... a questão também da cegueira dela quando não tem nenhum tipo de informação no zero-shot — é uma loteria."*

### 20. Ler os artigos de referência enviados pelo orientador
Lista do e-mail (os 4 últimos são mais gerais):

1. **Rethinking the Role of Demonstration: What Makes In-Context Learning Work?**
2. **Probing the Decision Boundaries of In-Context Learning in Large Language Models**
3. **What Learning Algorithm is In-Context Learning? Investigation with Linear Models**
4. **What Can Transformers Learn In-Context? A Case Study of Simple Function Classes**
5. **Attention Is All You Need**
6. **Language Models are Few-Shot Learners** (Brown et al., 2022 — base do aprendizado em contexto; ~70 mil citações)
7. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al.)
8. **The Illusion of Thinking...**

- **Citação (~1114s, e-mail):** *"depois eu vou te passar uma lista dos artigos que eu li... pra embasar seu texto."* — usar especialmente a **parte de testes/experimentos** desses artigos como modelo de escrita.

---

## Resumo dos itens acionáveis

| # | Item | Bloco | Tipo |
|---|------|-------|------|
| 1 | Padronizar nomenclatura A–G / blocos | Geral | em andamento |
| 2 | Limpar slides fora de lugar no Bloco 3 | 3 | limpeza |
| 3 | Mostrar W aprendido nas tabelas | 1 | apresentação |
| 4 | Investigar fidelidade baixa do Problema A | 1 | re-análise |
| 5 | Oracle/recuperação de W na meia-lua (W aproximador) | 1 | novo experimento |
| 6 | Mapas de erro maiores, 1 por slide | 1 | apresentação (parcial) |
| 7 | Superfície de separação SVM gaussiano na meia-lua | 2 | novo gráfico |
| 8 | Esclarecer coluna "acc vs métrica" no Problema F | 2 | texto/explicação |
| 9 | Explicar seleção random dos exemplos few-shot | 2 | texto |
| 10 | Re-rodar Problema F (meia-lua), n_shot=20, múltiplas reps | 2 | re-execução |
| 11 | Explicar F1 e Kappa (fórmulas) | 2 | texto |
| 12 | Confirmar R3 = x,y,x·y (hipérbole) e R4 = x,y,x²,y² (elipse) | 3 | código |
| 13 | Rodar peso × altura com few-shot | 3 | novo experimento |
| 14 | Investigar queda de acurácia real com 4 features; rodar Perceptron | 3 | re-análise |
| 15 | Reportar nº absoluto de amostras erradas | 3 | apresentação |
| 16 | Adicionar tabela do prompt neutro (x1,x2) | 3 | apresentação |
| 17 | Experimento com nomes de classe A/B | 3 | novo experimento |
| 18 | Começar a escrever a dissertação/artigo | Escrita | escrita |
| 19 | Destacar conclusões principais | Escrita | escrita |
| 20 | Ler os 8 artigos de referência | Escrita | leitura |
