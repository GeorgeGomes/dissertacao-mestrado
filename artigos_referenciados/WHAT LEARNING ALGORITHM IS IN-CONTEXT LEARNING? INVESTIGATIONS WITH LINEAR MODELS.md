# What Learning Algorithm Is In-Context Learning? Investigations with Linear Models

## Metadados
- **Autores:** Ekin Akyürek (MIT CSAIL / Google Research), Dale Schuurmans (Google Research), Jacob Andreas (MIT CSAIL), Tengyu Ma (Stanford / Google Research), Denny Zhou (Google Research). (Andreas, Ma e Zhou com orientação colaborativa compartilhada.)
- **Ano:** 2022 (pré-print arXiv:2211.15661); versão v3 de 17 de maio de 2023.
- **Publicação/Venue:** Publicado como conference paper na ICLR 2023 (International Conference on Learning Representations).
- **Arquivo:** WHAT LEARNING ALGORITHM IS IN-CONTEXT LEARNING? INVESTIGATIONS WITH LINEAR MODELS.pdf

## Resumo completo

O artigo investiga uma hipótese central sobre o fenômeno de *in-context learning* (ICL) em modelos de sequência baseados em transformers: a de que esses modelos implementam, de forma implícita, algoritmos de aprendizado padrão. A ideia é que o transformer codifica um "modelo menor" em suas ativações internas (estados ocultos) e o atualiza à medida que novos exemplos rotulados (x, f(x)) aparecem no contexto, sem qualquer atualização dos parâmetros da rede. Diferentemente de trabalhos anteriores que perguntavam *quais funções* o ICL consegue aprender, o foco aqui é *como* ele aprende — os vieses indutivos e as propriedades algorítmicas do ICL baseado em transformers.

Para tornar a investigação tratável, os autores escolhem a regressão linear como problema-protótipo, por ser um problema extremamente bem compreendido e com diversas soluções algorítmicas conhecidas (descida de gradiente, regressão ridge, mínimos quadrados/OLS, estimador Bayesiano). O trabalho oferece três fontes de evidência. Primeiro, evidência teórica: provam por construção que transformers conseguem implementar algoritmos de aprendizado de modelos lineares — um passo de descida de gradiente com O(d) de tamanho oculto e profundidade constante, e uma atualização de regressão ridge em forma fechada (via fórmula de Sherman–Morrison) com O(d²) de tamanho oculto.

Segundo, evidência comportamental: transformers treinados para ICL têm predições que coincidem de perto com as de preditores clássicos (OLS, ridge, mínimos quadrados exato e descida de gradiente). Em dados sem ruído, o ICL se comporta como OLS (o preditor de norma mínima na região subdeterminada); com ruído, segue o preditor de mínimo risco de Bayes (ridge com regularização σ²/τ²). Além disso, surgem "transições de fase algorítmicas": modelos rasos se aproximam de um passo de descida de gradiente, modelos de profundidade intermediária de ridge, e modelos profundos de OLS.

Terceiro, evidência algorítmica (mecanística): por meio de *probes* treinados sobre os estados ocultos, os autores mostram que quantidades intermediárias esperadas desses algoritmos — o vetor de momentos X^T·Y e o vetor de pesos de mínimos quadrados w_OLS — podem ser decodificadas das ativações das camadas finais do modelo (não-linearmente). O momento X^T·Y torna-se decodificável mais cedo (camada ~7) e o vetor de pesos depois (camada ~12), espelhando a ordem de cálculo dos algoritmos construídos teoricamente. A conclusão é que o ICL, ao menos no caso linear, é compreensível em termos algorítmicos e que os transformers parecem "redescobrir" algoritmos de estimação padrão a partir apenas da tarefa de modelagem de sequências.

## Principais contribuições
- Formaliza e testa a hipótese de que o ICL implementa implicitamente algoritmos de aprendizado conhecidos, codificando e atualizando um modelo interno nas ativações.
- Prova construtiva (Teorema 1) de que um transformer pode computar um passo de descida de gradiente sobre o objetivo de mínimos quadrados regularizado com profundidade constante e O(d) de espaço oculto.
- Prova construtiva (Teorema 2) de que um transformer pode realizar uma atualização da solução em forma fechada de regressão ridge (uma atualização de Sherman–Morrison) com profundidade constante e O(d²) de espaço oculto.
- Define primitivas computacionais implementáveis por uma única camada de transformer (mov, mul, div, aff — Lema 1), usadas para montar os algoritmos.
- Introduz duas métricas comportamentais para comparar preditores: SPD (Squared Prediction Difference, diferença de predição) e ILWD (Implicit Linear Weight Difference, diferença entre os pesos lineares implícitos).
- Evidência empírica de que o ICL casa com OLS em dados sem ruído e com o preditor de mínimo risco de Bayes (ridge) em dados ruidosos.
- Demonstração de "transições de fase algorítmicas" conforme profundidade e tamanho oculto variam (GD → ridge → OLS).
- Evidência mecanística via probing de que momentos (X^T·Y) e pesos (w_OLS) são codificados nas camadas profundas, na ordem prevista pelos algoritmos.

## Metodologia

A investigação combina teoria e experimentos. No lado teórico (Seção 3), os autores estabelecem primitivas computacionais (mover, multiplicar, dividir e aplicar transformações afins) que uma única camada de decoder transformer consegue implementar, e usam essas peças para construir, explicitamente, parametrizações que executam um passo de descida de gradiente e uma atualização de ridge em forma fechada. Esses resultados fornecem limites superiores de capacidade (camadas e unidades ocultas) suficientes para implementar — embora não necessariamente aprender — cada algoritmo; passos iterativos correspondem a "empilhar" grupos de camadas.

No lado empírico (Seção 4), treinam um decoder transformer autorregressivamente sobre o objetivo de ICL (Eq. 8) em problemas de regressão linear, com pesos w ~ N(0, I) e entradas x ~ N(0, I), pares (x, y) codificados como vetores de dimensão d+1. Fazem busca de hiperparâmetros sobre profundidade L ∈ {1,2,4,8,12,16}, tamanho oculto H ∈ {16,...,1024} e número de cabeças M ∈ {1,2,4,8}; a configuração principal (L=16, H=512, M=4) minimizou a perda de validação. Treinam por 500.000 iterações, com cada dataset in-context contendo 40 pares (x, y). Comparam o ICL contra preditores de referência (k-NN uniforme e ponderado, SGD de uma passada, GD de um passo em batch, ridge com vários λ, e OLS) usando SPD e ILWD, focando na região subdeterminada (menos exemplos que dimensões). Variam ruído de dados σ² e variância do prior τ² para testar o comportamento Bayesiano, e variam profundidade e tamanho oculto para identificar as transições de fase.

Na Seção 5, congelam um ICL treinado e treinam *probes* auxiliares (com atenção por posição, e variantes linear e MLP) para recuperar quantidades intermediárias (X^T·Y e w_OLS) a partir dos estados ocultos de cada camada e posição, num problema de d=4. O erro do probe mede se a quantidade está codificada e o mapa de atenção indica onde; uma tarefa de controle com peso fixo w=1 (sem necessidade de ICL) serve de baseline.

## Conclusão do artigo

Os autores apresentam um conjunto de experimentos que caracteriza as computações subjacentes ao in-context learning de funções lineares em transformers. Mostram que (i) esses modelos são, em teoria, capazes de implementar múltiplos algoritmos de regressão linear; (ii) empiricamente implementam essa gama de algoritmos, transitando entre eles conforme a capacidade do modelo e o ruído dos dados; e (iii) podem ser sondados (*probed*) para revelar quantidades intermediárias computadas por esses algoritmos. Embora restritos ao caso linear, argumentam que a metodologia pode ser estendida a classes de funções mais ricas e a modelos de linguagem em larga escala. O resultado geral é uma evidência inicial de que o fenômeno aparentemente misterioso do ICL pode ser entendido com o ferramental padrão de ML, e de que soluções de problemas de aprendizado descobertas por pesquisadores podem também ser descobertas, por conta própria, pela descida de gradiente durante o treinamento.

## Relação com este trabalho

Este artigo é a justificativa teórica central para tratar o ICL/few-shot como um verdadeiro algoritmo de aprendizado, e não apenas recuperação de padrões. Na dissertação "Explicando Decisões de LLMs via Otimização Inversa", o **Bloco 2 (LLM como APRENDIZ)** e a **Fase E** colocam o LLM para aprender, em contexto, o critério de um perito externo a partir de exemplos rotulados. Akyürek et al. sustentam exatamente a premissa de que isso é possível: o transformer constrói um preditor interno e o atualiza com os exemplos do contexto, implementando algo equivalente a OLS/ridge/descida de gradiente. Isso fundamenta a hipótese H3 (LLM como aprendiz) e dá respaldo à própria expectativa de que curvas de aprendizado por número de exemplos (n_shot) façam sentido.

Há também uma forte afinidade metodológica com o **Bloco 1 (LLM como FONTE)**: assim como o artigo recupera os *pesos lineares implícitos* (w_OLS) do ICL — via probing e via a métrica ILWD que ajusta o w linear mais próximo das predições do modelo —, a dissertação recupera, por otimização inversa, uma métrica de Mahalanobis diagonal W a partir das classificações do LLM. Ambos partem das decisões observadas e inferem o objeto linear que melhor as explica. A escolha de **modelos lineares e fronteiras de decisão simples** como bancada de testes, a ênfase em **estabilidade/consistência do comportamento** e a comparação contra **baselines clássicos (k-NN, regressão, etc.)** são paralelos diretos entre os dois trabalhos. A diferença importante a registrar: o artigo trata de **regressão** com transformers treinados especificamente para ICL, enquanto a dissertação trata de **classificação** com um LLM de propósito geral — uma diferença a mencionar ao transpor as conclusões.

## Onde citar

- **Introdução / Fundamentação teórica:** ao motivar por que faz sentido o LLM "aprender em contexto", citar como evidência de que o ICL implementa algoritmos de aprendizado conhecidos (regressão/descida de gradiente).
- **Hipótese H3 (LLM como aprendiz) e descrição da Fase E / Bloco 2:** suporte teórico de que o few-shot/in-context corresponde a um procedimento de estimação real.
- **Trabalhos relacionados:** como referência-chave da linha "ICL como algoritmo implícito" (junto a Garg et al. 2022, Xie et al. 2022, von Oswald et al., Müller et al. 2021).
- **Metodologia / Otimização inversa (Bloco 1):** ao justificar a recuperação dos pesos lineares implícitos do LLM, comparando com a métrica ILWD e o probing de w_OLS do artigo.
- **Baselines clássicos (Bloco 2):** ao posicionar a comparação LLM vs. k-NN/LR/SVM, ecoando a comparação ICL vs. preditores clássicos do artigo.
- **Discussão / Limitações:** ao apontar a transposição de regressão (artigo) para classificação (dissertação) e de transformer treinado-para-ICL para LLM de propósito geral.
