# What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

## Metadados
- **Autores:** Shivam Garg, Dimitris Tsipras, Percy Liang, Gregory Valiant (todos da Stanford University; Garg e Tsipras com contribuição igual)
- **Ano:** 2022 (publicado no NeurIPS 2022; versão arXiv v3 de 11 de agosto de 2023, arXiv:2208.01066)
- **Publicação/Venue:** Advances in Neural Information Processing Systems (NeurIPS) 2022
- **Arquivo:** What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.pdf

## Resumo completo

O artigo investiga o aprendizado em contexto (*in-context learning*, ICL) — a capacidade de um modelo condicionar-se a uma sequência de exemplos no prompt (pares entrada-saída de alguma tarefa) e, junto a uma nova entrada de consulta, gerar a saída correspondente, **sem qualquer atualização de parâmetros** no momento da inferência. Embora modelos de linguagem grandes como o GPT-3 exibam essa habilidade, não estava claro qual a relação entre as tarefas em que isso funciona e o que está presente nos dados de treino. Para tornar a questão tratável, os autores formalizam o problema como **aprender uma classe de funções a partir de exemplos em contexto**: dado um prompt `(x1, f(x1), ..., xk, f(xk), xquery)`, o modelo deve prever `f(xquery)` para "a maioria" das funções `f` de uma classe `F`.

O resultado central é empírico: um **Transformer padrão (decoder-only, família GPT-2, 12 camadas, 8 cabeças, embedding de 256 dimensões, ~9,5M de parâmetros)**, treinado do zero (sem texto, sem fine-tuning), consegue aprender em contexto a classe de **funções lineares** em 20 dimensões, atingindo erro comparável ao do **estimador ótimo de mínimos quadrados** para qualquer número de exemplos em contexto. Os autores argumentam que isso não pode ser explicado por memorização: o espaço de prompts é astronomicamente grande e mesmo restringir o treino a poucos vetores-peso distintos não degrada o desempenho — o modelo, portanto, codifica um **algoritmo de aprendizado** genuíno, executado em um único *forward pass*.

O ICL aprendido também **extrapola para fora da distribuição de treino**, de forma graciosa, sob dois tipos de deslocamento: (i) entre a distribuição de treino e os prompts de inferência (covariância enviesada, ruído nos rótulos, mudança de escala, subespaços de baixa dimensão) e (ii) entre os exemplos em contexto e a entrada de consulta (exemplos e consulta em ortantes diferentes, consulta ortogonal aos exemplos, consulta igual a um exemplo). Em vários desses casos o modelo permanece próximo dos mínimos quadrados, e notavelmente exibe a curva de *double descent* na regressão linear com ruído.

Além de funções lineares, os autores mostram que Transformers podem aprender em contexto **classes de funções mais complexas**: funções lineares esparsas (desempenho próximo ao Lasso, explorando a esparsidade), **árvores de decisão de profundidade 4** (superando aprendizado guloso de árvores e *boosting*/XGBoost) e **redes neurais ReLU de duas camadas** (comparável a treinar uma rede da mesma arquitetura por descida de gradiente nos exemplos). Por fim, exploram o papel da **capacidade do modelo** e da **dimensionalidade do problema**, e introduzem o **aprendizado por currículo** (*curriculum learning*), que acelera drasticamente o treino, sobretudo em dimensões altas.

## Principais contribuições

- **Formalização do ICL como aprendizado de classe de funções:** define operacionalmente que um modelo aprende em contexto uma classe `F` se prevê `f(xquery)` com erro médio ≤ ε para "a maioria" das `f ∈ F`, criando um banco de prova bem definido e independente de dados de linguagem.
- **Transformers aprendem regressão linear em contexto:** demonstram empiricamente que um Transformer treinado do zero atinge erro comparável ao estimador ótimo de mínimos quadrados em funções lineares de 20 dimensões.
- **Refutação da explicação por memorização:** argumentam (com cálculos empíricos) que o desempenho não vem de indexar prompts/vetores vistos no treino — o modelo codifica um algoritmo de aprendizado.
- **Robustez fora da distribuição:** o ICL se mantém sob diversos deslocamentos de distribuição entre treino/inferência e entre exemplos/consulta, indicando que o modelo aprendeu regressão linear com generalidade.
- **Classes mais complexas:** Transformers igualam ou superam algoritmos especializados (Lasso, redes treinadas por SGD, árvores de decisão por *boosting*) para funções esparsas, redes ReLU de 2 camadas e árvores de decisão.
- **Fatores que importam:** capacidade do modelo melhora o ICL (especialmente fora da distribuição) e o *curriculum learning* acelera muito o treino; pouca quantidade de dados distintos já basta para ICL não-trivial.
- **Perspectiva de engenharia reversa de algoritmos:** sugerem que entender os algoritmos codificados pelos Transformers pode revelar novos algoritmos de aprendizado eficientes (ex.: para árvores de decisão).

## Metodologia

A metodologia geral é treinar um modelo `Mθ` para o ICL de uma classe `F` com respeito a uma distribuição `DF` sobre funções e `DX` sobre entradas. Para cada prompt de treino: sorteia-se uma função `f ∼ DF`, sorteiam-se entradas `x1, ..., xk+1 ∼ DX` i.i.d. e avalia-se `f` para montar `P = (x1, f(x1), ..., xk+1, f(xk+1))`. No caso linear, `xi ∼ N(0, Id)` e `f(x) = wᵀx` com `w ∼ N(0, Id)`, `d = 20`. O treino minimiza o **erro quadrático médio sobre todos os prefixos do prompt** — ou seja, para cada prefixo `Pi` com `i` exemplos, o modelo prevê `f(xi+1)`, aproveitando que o Transformer calcula todas as previsões de prefixo em um único *forward pass*.

A arquitetura é um Transformer decoder-only da família GPT-2 (12 camadas, 8 cabeças, embedding 256, ~9,5M parâmetros). Entradas e saídas escalares são mapeadas ao espaço de embedding por transformações lineares aprendíveis (saídas `f(xi)` são completadas com zeros para igualar a dimensão das entradas); outra transformação linear mapeia o vetor previsto de volta a um escalar. O treino é **do zero** (sem texto real, sem fine-tuning), com lote de 64 e 500 mil passos. Aplica-se **aprendizado por currículo**: começa-se com subespaço de 5 dimensões e prompts curtos, aumentando gradualmente a dimensão do subespaço (+1) e o comprimento do prompt (+2) a cada 2.000 passos, até atingir a dimensão `d` e comprimento `2d+1`.

A avaliação compara o Transformer a **baselines**: mínimos quadrados (ótimo, limite inferior de erro), `n`-vizinhos mais próximos e estimação por média de `yi·xi`. Para as classes complexas, os baselines são Lasso (esparso), aprendizado guloso de árvores e XGBoost (árvores) e rede de 2 camadas treinada por Adam (redes ReLU). Análises adicionais incluem: estudo de o que a função condicionada ao prefixo representa (visualização ao variar a consulta numa direção aleatória; alinhamento do gradiente da previsão com `w` e com `proj(w)`), experimentos de deslocamento de distribuição, e *ablations* de capacidade, dimensão e quantidade de prompts/funções distintas no treino.

## Conclusão do artigo

Os autores concluem que Transformers treinados do zero conseguem aprender em contexto a classe de funções lineares com desempenho comparável ao estimador ótimo de mínimos quadrados, inclusive sob deslocamentos de distribuição, e que o mesmo vale para classes mais complexas (lineares esparsas, árvores de decisão e redes neurais de duas camadas) — problemas resolvidos na prática por algoritmos iterativos sofisticados. Isso mostra que Transformers padrão, com otimização padrão, podem **codificar algoritmos de aprendizado não-triviais que usam os exemplos em contexto de forma sofisticada**. Os autores ressalvam que estender essas conclusões aos modelos de linguagem reais ainda exige investigação. Apontam direções futuras: entender a complexidade do ICL em função da classe, capacidade e dados; explicar a aceleração do *curriculum learning*; comparar vieses indutivos entre famílias de modelos (ex.: Transformers vs. LSTMs); e fazer engenharia reversa dos algoritmos codificados, possivelmente descobrindo novos algoritmos eficientes.

## Relação com este trabalho

Este artigo é a **base teórica do papel de "LLM como aprendiz" (Bloco 2)** da dissertação. Onde a dissertação coloca o LLM para aprender, via *few-shot*, o critério de um perito externo com métrica conhecida, este artigo dá a fundamentação de que **Transformers efetivamente aprendem classes de funções simples em contexto**, sem atualização de parâmetros — e que, para a classe linear, o algoritmo que emerge aproxima a solução de mínimos quadrados. Há várias conexões diretas:

- **Funções lineares como classe-alvo:** a dissertação estuda fronteiras lineares (Problemas A/B/C) e aprende uma métrica de Mahalanobis diagonal `W`; este artigo mostra que o ICL de funções lineares é não só possível, mas próximo do ótimo, dando respaldo à expectativa de que o LLM reproduza critérios lineares de um perito.
- **Estabilidade/consistência (H1, H5):** a robustez do ICL a deslocamentos de distribuição (exemplos e consulta em ortantes diferentes, covariância enviesada) ecoa os testes de **transferência e consistência** da dissertação (aplicar `Ŵ_LLM` aprendida em A aos Problemas B/C com geometria distinta).
- **Few-shot e estratégias de exemplos (Fase E, H2/H4):** o achado de que o modelo executa regressão linear genérica a partir dos exemplos dá contexto para a discussão da dissertação sobre como a seleção/ordem de exemplos (easy/hard/mixed, recency bias) afeta o aprendizado.
- **Baselines clássicos:** o uso de mínimos quadrados, k-NN, Lasso e SVM/redes como referência espelha os **baselines clássicos** (k-NN, LR, SVM) da dissertação, reforçando a pergunta "o LLM faz algo que um classificador trivial não faria?".
- **Otimização inversa:** a proposta dos autores de "engenharia reversa do algoritmo codificado" conversa diretamente com a abordagem de **otimização inversa** da dissertação, que busca recuperar a função objetivo implícita (a métrica `W`) por trás das decisões do LLM.

## Onde citar

- **Introdução / fundamentação teórica:** ao motivar que LLMs aprendem em contexto e que isso pode ser formalizado como aprendizado de classes de funções.
- **Trabalhos relacionados:** como referência central de ICL formalizado e do achado de que Transformers aprendem funções lineares (e mais complexas) em contexto, próximo do ótimo.
- **Metodologia / Fase E (LLM como aprendiz, Bloco 2):** ao justificar o experimento de o LLM aprender o critério de um perito via few-shot.
- **Discussão de consistência e transferência (H1, H5):** ao comentar a robustez do ICL a deslocamentos de distribuição entre treino/aprendizado e teste.
- **Discussão sobre otimização inversa:** ao posicionar a recuperação da métrica implícita do LLM como uma forma de "entender o algoritmo de aprendizado codificado".
- **Comparação com baselines clássicos:** ao contrastar o desempenho do LLM com mínimos quadrados, k-NN, Lasso e SVM.
