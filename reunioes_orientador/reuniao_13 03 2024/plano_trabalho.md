# Plano de Trabalho — Reuniao 13/03/2024

**Participantes:** Raul (orientador), Yuri (co-orientador), George
**Proxima reuniao:** 15 dias, segunda-feira 16h (horario de Brasilia)

---

## Acao Imediata

- [ ] **Enviar por email** a lista de entregas para Raul e Yuri aprovarem antes de comecar a implementar

---

## Entregas para a Proxima Reuniao

### 1. Exibir o W estimado na Fase A
- Mostrar explicitamente o valor do vetor W aprendido na Fase A
- Analisar a distribuicao do W entre repeticoes e sementes
- Visualizar graficamente onde a metrica W erra vs. a rotulacao do LLM

### 2. Reescrever Hipoteses
- **H1:** Trocar "W aprendido" por "W_LLM" ou "W chapeu". A metrica e "estimada/inferida", nao "aprendida"
- **H4:** Inverter — os resultados mostraram que exemplos dificeis (proximos da fronteira) funcionam melhor, contrariando a hipotese original. Reescrever para refletir os dados

### 3. Verificar Selecao de Exemplos
- Plotar no grafico quais pontos estao sendo selecionados como faceis/dificeis/mixed/random
- Confirmar visualmente que a selecao esta correta

### 4. Aumentar Robustez
- Aumentar de 2 para **pelo menos 5 sementes aleatorias**
- Gerar pelo menos **5 rotulacoes/problemas diferentes**

---

## Experimentos a Executar

### Prioridade Alta

| # | Experimento | Objetivo |
|---|---|---|
| 1 | Mostrar W estimado + distribuicao entre repeticoes | Transparencia nos resultados da Fase A |
| 2 | Plotar rotulacao LLM vs. erros da metrica W (Fase A) | Entender onde e por que a fidelidade e ~70-76% |
| 3 | Testar com 5 sementes (ao inves de 2) | Robustez estatistica |
| 4 | Testar inversao da ordem das classes no prompt ("A ou B" vs "B ou A") | Detectar vies de posicao |
| 5 | Testar nomes semanticos nas features (ex: "altura"/"peso" ao inves de X1/X2) | Detectar vies semantico nas variaveis |
| 6 | Variar W do expert na Fase D | Ver comportamento com diferentes razoes de peso |

### Variacoes do W Expert (Fase D)

- Inverter a razao (ex: W=[1.5, 0.3] ao inves de [0.3, 1.5])
- Pesos iguais (W1 = W2, equivalente a distancia Euclidiana)
- Usar solucao de KNN como expert

### Experimento de Diluicao

- Fixar 3 exemplos dificeis
- Ir adicionando exemplos faceis progressivamente (3+1, 3+2, 3+5, 3+10...)
- Verificar se em algum ponto os faceis "diluem" os dificeis e a performance cai

### Prioridade Media

| # | Experimento | Objetivo |
|---|---|---|
| 7 | Implementar segundo algoritmo de otimizacao inversa | Mostrar que instabilidade do W nao e culpa do algoritmo |
| 8 | Expandir para 3 dimensoes e 3 centroides | Maior complexidade para publicacao |
| 9 | Gerar Gaussianas mais separadas (problemas faceis vs dificeis) | Entender comportamento em diferentes niveis de dificuldade |
| 10 | Testar apresentacao como coordenadas "(x1, x2)" vs valores separados | Efeito do formato no prompt |

---

## Sugestoes Metodologicas do Professor

### Algoritmo de Otimizacao Inversa
- Implementar **pelo menos mais um algoritmo** alem do Perceptron Estruturado
- Raul mencionou o trabalho classico de **Ahuja/Orlin** e ficou de enviar artigos
- Objetivo: demonstrar que os resultados sao robustos ao metodo de estimacao

### Dimensionalidade
- Proximo passo: **3 dimensoes e 3 centroides** (dataset Iris como referencia)
- Visualizacao 3D ainda e viavel
- **Por enquanto, ficar no mundo linear** — nao complicar com hipoteses nao-lineares

### Sobre Outras LLMs
- **Nao testar em outras LLMs agora** (Claude, Llama, Grok) — "vai embolar demais"
- Primeiro lapidar metodologia e template de testes
- Focar exclusivamente no GPT-4o-mini por enquanto
- Para publicacao futura: reportar media com desvio padrao ou justificar uso de modelos com output deterministico

---

## Observacoes Importantes

### Pontos Positivos
- **A linha do trabalho esta correta** — Raul e Yuri aprovaram o escopo e a direcao
- **A Fase D e a mais promissora para artigo** — facil de explicar (LLM aprendendo de expert para escalabilidade), aplicacoes praticas claras (ex: classificacao de credito)

### Pontos de Atencao
- **Fases A-B-C sao mais dificeis de justificar** — tomar cuidado na escrita para explicar por que consistencia entre problemas e uma pergunta interessante
- **Fidelidade de 70-76% na Fase A precisa melhorar** — investigar, gerar mais exemplos, mostrar graficamente onde erra
- **Instabilidade do W entre sementes** — duas sementes geraram metricas "totalmente diferentes". Pode ser criticado. Solucao: testar com mais de um algoritmo
- **Piora no Problema B com 10 exemplos few-shot** — resultado estranho, precisa investigar com mais testes. "Nao da pra generalizar com base so nessas duas rodadas"
- **Possivel nao-linearidade do LLM** — se o LLM gera fronteira nao-linear (espiral, meia-lua), a metrica de Mahalanobis diagonal nunca vai capturar. Precisa verificar
- **Nao criar 200 testes** — "fazer coisas pequenas agora e escolher pra que direcao aprofundar"

---

## Artigos Pendentes (Raul vai enviar)

1. Artigo sobre **consistencia de LLM** — menciona que exemplos dificeis melhoram performance
2. Artigo sobre **sensibilidade ao prompt** — efeito da ordem das classes/palavras
3. Referencia do **algoritmo classico de otimizacao inversa** (Ahuja & Orlin)

---

## Emails do Professor (Pos-Reuniao)

### Email 1 — Projecao Explicita para R3 (Kernel Quadratico)
- Criar uma terceira feature: `x3 = x1 * x2`
- O vetor W tera 3 componentes
- Projetando de volta para R2, teremos equacao de uma hiperbole
- **Pergunta central:** A LLM captura a importancia dessa terceira feature?
- Esperanca: melhorar a acuracia do aprendizado de metrica

### Email 2 — Observacao Importante sobre Fase de Concordancia
- Na fase de concordancia, testar a LLM para dados rotulados pela metrica W=(w1,w2,w3) utilizando **somente as duas primeiras features**
- Comparar esses resultados com a LLM utilizando as 3 features
- Objetivo: entender se a terceira feature (interacao) faz diferenca para o LLM

---

## Resumo de Prioridades

1. Lapidar escrita e metodologia
2. Rodar testes de robustez (mais sementes, mais problemas)
3. Investigar e melhorar fidelidade da Fase A
4. Testes de vies semantico (ordem das classes, nomes das features)
5. Variacoes do expert na Fase D
6. Implementar projecao R3 (email do professor)
7. Segundo algoritmo de otimizacao inversa
