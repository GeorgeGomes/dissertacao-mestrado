# An Explanation of In-context Learning as Implicit Bayesian Inference

## Metadados
- **Autores:** Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma (Stanford University)
- **Ano:** 2022 (preprint arXiv:2111.02080, v1 em nov/2021 e v6 em jul/2022; publicado na ICLR 2022)
- **Publicação/Venue:** International Conference on Learning Representations (ICLR 2022). Campo "Comments" do arXiv: ICLR 2022; citado como ICLR 2022 nas bibliografias de Garg et al. (2022) e Min et al. (2022)
- **Arquivo:** An Explanation of In-context Learning as Implicit Bayesian Inference.pdf (versão arXiv v6)
- **PDF público:** https://arxiv.org/pdf/2111.02080

## Resumo completo

O artigo propõe uma explicação teórica para o **aprendizado em contexto** (*in-context learning*): por que um modelo de linguagem, treinado apenas para prever o próximo token, consegue executar uma tarefa nova só de condicionar em um *prompt* com exemplos entrada-saída, sem ter sido treinado para aprender com exemplos e apesar de o *prompt* (uma concatenação de exemplos independentes) ser um texto pouco natural.

A ideia central é um modelo generativo do pré-treinamento com **conceito latente**: cada documento é gerado sorteando-se primeiro um conceito `θ` de uma família `Θ` e, em seguida, uma sequência de tokens de um **Modelo Oculto de Markov (HMM)** cujas transições são parametrizadas por `θ` (Seção 2, equação 2). Para gerar continuações coerentes durante o pré-treinamento, o modelo precisa inferir o conceito latente do documento a partir das frases anteriores. No *prompt*, os exemplos são gerados a partir de um mesmo conceito `θ*`; o aprendizado em contexto ocorre quando o modelo infere esse **conceito compartilhado** entre os exemplos e prevê a saída pela **distribuição preditiva a posteriori**,

```
p(saída | prompt) = ∫ p(saída | conceito, prompt) · p(conceito | prompt) d(conceito),   (eq. 1)
```

de modo que, se `p(conceito | prompt)` concentra no conceito do *prompt* conforme os exemplos se acumulam, o modelo "seleciona" a tarefa por marginalização. É nesse sentido que o aprendizado em contexto é uma **inferência bayesiana implícita**.

A dificuldade técnica é que os *prompts* vêm de uma distribuição diferente da de pré-treinamento (os delimitadores e a concatenação de exemplos independentes são transições de baixa probabilidade). Os autores provam (Seção 3, Teoremas 1 a 3) que, apesar dessa discrepância, o erro assintótico do preditor em contexto é ótimo quando **o sinal sobre o conceito latente contido em cada exemplo supera o erro introduzido pela discrepância de distribuição**, e que o erro diminui com o comprimento de cada exemplo: a informação nas **entradas**, e não só no mapeamento entrada-saída, é útil para o aprendizado em contexto.

## Principais contribuições

- Uma **explicação bayesiana** do aprendizado em contexto: o modelo infere o conceito latente compartilhado pelos exemplos do *prompt* e prevê pela distribuição preditiva a posteriori, sem atualizar parâmetros.
- **Garantias teóricas** de que o aprendizado em contexto emerge de uma distribuição de pré-treinamento com estrutura de conceitos latentes (mistura de HMMs), mesmo com discrepância de distribuição entre *prompt* e pré-treinamento, com erro decrescente no número e no comprimento dos exemplos.
- O dataset sintético **GINC** (*Generative IN-Context learning*), pequeno e totalmente controlado, em que Transformers e LSTMs exibem aprendizado em contexto, permitindo ablações impossíveis em corpora reais.
- Reprodução, no GINC, de fenômenos observados em LLMs reais: ganho com a escala do modelo mesmo sob a mesma perda de pré-treinamento, **sensibilidade à ordem dos exemplos** (variação de 10 a 40 pontos entre permutações, como em Zhao et al., 2021) e casos em que *zero-shot* supera *few-shot*.

## Metodologia

Parte teórica: modelo generativo com conceito latente e HMM, definição da distribuição de *prompts* (n exemplos independentes de um mesmo conceito mais uma entrada de teste) e análise assintótica do preditor em contexto sob condições de distinguibilidade entre conceitos. Parte empírica (Seção 4): o GINC é uma mistura uniforme de 5 conceitos (HMMs) que gera 1000 documentos com cerca de 10 milhões de tokens; os *prompts* têm de 0 a 64 exemplos, com comprimentos de exemplo k em {3, 5, 8, 10}, 2500 *prompts* por configuração. Treinam-se Transformers baseados em GPT-2 (4, 12 e 16 camadas) e LSTMs em três tamanhos de vocabulário (50, 100 e 150), com média sobre 5 execuções de pré-treinamento. Ablações: pré-treinar com um único conceito, com transições aleatórias, ou testar com conceitos nunca vistos faz o aprendizado em contexto falhar, mostrando que a estrutura de mistura de conceitos é essencial.

## Conclusão do artigo

O aprendizado em contexto pode ser entendido como inferência bayesiana implícita sobre um conceito latente, herdada da coerência de longo alcance dos documentos de pré-treinamento; a estrutura latente da distribuição de pré-treinamento é o que o torna possível. A acurácia em contexto cresce com o número e o comprimento dos exemplos, e o modelo pode aproveitar a informação das entradas mesmo quando o mapeamento entrada-saída é pouco informativo. Ficam em aberto o efeito da escala e da arquitetura (LSTMs superam Transformers no GINC) e a extensão da teoria a corpora reais.

## Relação com este trabalho

A dissertação assume que o LLM tem um **critério decisório estável e aprendível** e que, no Bloco 2, ele consegue inferir o critério de um perito externo a partir de exemplos rotulados. A leitura bayesiana de Xie et al. dá forma a essa premissa: as demonstrações servem para o LLM inferir um **conceito latente compartilhado**, e a métrica do perito (o vetor W e os centróides) é, nessa leitura, o conceito a inferir. Três ligações concretas:

- **Bloco 2, Fase E (refutação de H4):** o teorema principal diz que o aprendizado em contexto funciona quando o sinal sobre o conceito latente em cada exemplo supera o erro da discrepância de distribuição. Exemplos **fáceis** (alta margem, prototípicos) são exatamente os que carregam sinal nítido sobre o critério; exemplos **difíceis** (baixa margem) são evidência ambígua. Isso é compatível com o resultado de que *easy* supera *hard* com poucos exemplos e de que *hard* só compensa em alto volume, quando o acúmulo de evidência concentra a posteriori. É uma leitura compatível, não uma previsão derivada formalmente, pois a teoria trata de sequências de tokens e HMMs, não de pontos no plano.
- **Fundamentação, §Aprendizado em Contexto:** complementa Garg et al. (2022) e Akyürek et al. (2023), que mostram que Transformers implementam algoritmos de aprendizado em contexto, com a explicação de **por que** esse comportamento emerge do pré-treinamento; reforça a premissa de que há um critério interno recuperável por otimização inversa.
- **Sensibilidade à ordem dos exemplos:** o GINC reproduz a variação com a permutação dos exemplos observada por Zhao et al. (2021), o mesmo fenômeno medido no experimento de ordem dos exemplos *few-shot* (recency bias) do Bloco 2.

## Onde citar

- **Fundamentação Teórica, §Aprendizado em Contexto em LLMs:** após Garg et al. e Akyürek et al., como a explicação teórica do fenômeno (inferência bayesiana implícita sobre conceito latente).
- **Trabalhos Relacionados, §Aprendizado em Contexto:** ao lado de Garg et al. e Akyürek et al., na família de trabalhos que explicam o mecanismo.
- **Resultados, Bloco 2 (interpretação da refutação de H4):** ao lado de Min et al. (2022), como leitura compatível com a vantagem das âncoras prototípicas com poucos exemplos.
