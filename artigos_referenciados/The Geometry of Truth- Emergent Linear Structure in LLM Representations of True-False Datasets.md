# The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets

## Metadados
- **Autores:** Samuel Marks (Northeastern University) e Max Tegmark (MIT)
- **Ano:** 2024 (versão arXiv v3 de 19/08/2024; pré-print original arXiv:2310.06824, outubro de 2023)
- **Publicação/Venue:** Conference on Language Modeling (COLM) 2024
- **Arquivo:** The Geometry of Truth- Emergent Linear Structure in LLM Representations of True:False Datasets.pdf

## Resumo completo

O artigo investiga, em detalhe, como Modelos de Linguagem de Grande Escala (LLMs) representam internamente a verdade ou falsidade de afirmações factuais. Os autores partem de um problema prático: LLMs frequentemente produzem afirmações falsas, às vezes mesmo quando "sabem" que são falsas. Trabalhos anteriores treinaram sondas (probes) lineares sobre as ativações internas dos modelos para inferir se uma afirmação é verdadeira, sugerindo a existência de uma "direção de verdade" no espaço latente. Contudo, esses resultados eram controversos: algumas sondas falhavam em generalizar de maneiras básicas (por exemplo, para afirmações contendo a palavra "não"), levantando a dúvida de que estariam capturando não a verdade, mas apenas atributos correlacionados com ela (como a probabilidade do texto).

Para esclarecer essa questão, os autores curam conjuntos de dados de alta qualidade com afirmações verdadeiras/falsas simples e inequívocas (cidades, traduções espanhol-inglês, comparações numéricas, negações, conjunções e disjunções), além de conjuntos não curados e de um conjunto especial chamado "likely" (texto não factual cujo último token é provável ou improvável). Apoiando-se em três linhas de evidência — visualização por PCA, experimentos de transferência/generalização de sondas, e intervenções causais cirúrgicas no forward pass do modelo —, eles argumentam que, em escala suficiente, LLMs computam e representam linearmente a verdade ou falsidade de afirmações factuais.

A análise mostra que a estrutura linear de separação verdadeiro/falso emerge nas representações (visível já nas duas primeiras componentes principais) e que ela se torna mais geral e abstrata conforme o modelo cresce (LLaMA-2-7B vs. 13B vs. 70B). Os autores localizam essa representação em um pequeno grupo de estados ocultos (sobre o token final da afirmação e a pontuação de fim de frase) via experimentos de patching. Eles também introduzem o "mass-mean probing" (sondagem por diferença de médias), uma alternativa simples e sem otimização à regressão logística, que generaliza tão bem quanto outras técnicas e, sobretudo, identifica direções mais fortemente implicadas causalmente nas saídas do modelo.

Por fim, intervenções causais — somar ou subtrair a direção de verdade nos estados ocultos — conseguem fazer o modelo tratar afirmações falsas como verdadeiras e vice-versa, mesmo em dados fora da distribuição de treino (OOD). As sondas mass-mean superam regressão logística e CCS em 7 de 8 condições experimentais quanto ao efeito causal, apesar de terem acurácia de classificação semelhante.

## Principais contribuições

- Evidência de que representações lineares de verdade **emergem com a escala**: modelos maiores possuem uma noção de verdade mais abstrata, que se aplica a entradas estrutural e topicamente diversas.
- **Localização** das representações de verdade a um pequeno grupo de estados ocultos causalmente implicados (sobre o token final e a pontuação de fim de frase), via experimentos de patching/mediação causal.
- Introdução do **mass-mean probing** (sondagem por diferença de médias, com correção de covariância equivalente à Análise de Discriminante Linear de Fisher): simples, sem otimização, tão acurado quanto regressão logística (LR) e CCS, mas com direções mais causais.
- Demonstração de que treinar em afirmações **e suas negações/opostos** melhora a generalização das sondas.
- Distinção empírica entre representar **verdade** e representar apenas **probabilidade do texto** (via o conjunto "likely" e datasets com anti-correlação entre verdade e probabilidade).
- Disponibilização pública de datasets curados, código e um data-explorer interativo (github.com/saprmarks/geometry-of-truth).

## Metodologia

Os autores trabalham com a família LLaMA-2 (7B, 13B e 70B) e extraem ativações do *residual stream*. A metodologia tem quatro pilares:

1. **Localização por patching (Seção 3):** rodam o modelo em prompts onde uma afirmação final é verdadeira ou falsa, fazem cache das ativações e trocam (patch) ativações entre as duas versões, medindo a diferença em log P(TRUE) − log P(FALSE). Isso identifica três grupos de estados ocultos causalmente relevantes; o grupo (b), sobre o token final e a pontuação, parece codificar a verdade da afirmação completa.

2. **Visualização por PCA (Seção 4):** projetam as ativações centradas (sem few-shot) nas duas primeiras componentes principais, revelando separação linear clara entre afirmações verdadeiras e falsas nos datasets curados. Estudam o alinhamento dos eixos de separação entre datasets e como ele varia com a escala e com as camadas (às vezes direções antipodais, ortogonais ou alinhadas).

3. **Sondagem e generalização (Seção 5):** treinam sondas lineares com três técnicas — regressão logística (LR), mass-mean (MM) e contrast-consistent search (CCS, não supervisionada) — em 80% de um dataset e avaliam a transferência para outros datasets. Discutem uma deficiência da LR (converge para o separador de margem máxima, distorcido por features não ortogonais à direção de verdade) e propõem o mass-mean: a direção θ_mm = μ⁺ − μ⁻ (média dos verdadeiros menos média dos falsos), com versão IID corrigida por Σ⁻¹ (equivalente à LDA de Fisher e interpretável como whitening de Mahalanobis).

4. **Intervenções causais (Seção 6):** deslocam as ativações do grupo (b) ao longo da direção da sonda, normalizada para que somar θ transforme a afirmação falsa média na verdadeira média. Medem os *normalized indirect effects* (NIEs) das transformações falso→verdadeiro e verdadeiro→falso em dados OOD (sp_en_trans), comparando LR, MM e CCS.

## Conclusão do artigo

Combinando visualizações, experimentos de sondagem/generalização e evidência causal, o trabalho conclui que, em escala suficiente, LLMs computam e **representam linearmente** a verdade/falsidade de afirmações factuais. Essas representações são localizáveis em estados ocultos específicos e tornam-se mais gerais com a escala. O mass-mean probing é apresentado como alternativa simples às demais técnicas de sondagem linear, identificando melhor a direção de verdade — mais acurada em generalização e, sobretudo, mais causalmente implicada nas saídas do modelo. As limitações reconhecidas incluem o foco em afirmações simples (não distinguindo verdade de noções próximas como "comumente acreditado" ou "verificável") e o estudo restrito à família LLaMA-2.

## Relação com este trabalho

Este artigo dá suporte empírico direto à hipótese central da dissertação "Explicando Decisões de LLMs via Otimização Inversa": que existe uma **estrutura linear/geométrica estável** por trás das decisões de um LLM, passível de ser capturada por uma métrica. Marks e Tegmark mostram que conceitos abstratos (verdade) são representados como **direções lineares** no espaço de ativações e que essas direções **transferem** entre datasets estrutural e topicamente diferentes — exatamente o tipo de transferência/consistência que a dissertação testa nos Blocos 1 (LLM como FONTE) e 2 (LLM como APRENDIZ). Há ainda uma conexão técnica forte: o **mass-mean probing** dos autores usa a direção de diferença de médias corrigida pela covariância (Σ⁻¹), explicitamente interpretada como **distância/whitening de Mahalanobis** e equivalente à LDA de Fisher — exatamente a família de métricas (Mahalanobis diagonal, W) que a dissertação aprende por otimização inversa. A constatação de que a separação verdadeiro/falso é capturada por componentes principais e por uma fronteira linear no espaço de ativações reforça a plausibilidade de aprender um critério geométrico (W) a partir das classificações do LLM. A ênfase em **explicabilidade** e em direções **causalmente implicadas** alinha-se ao objetivo de explicar decisões do LLM, e não apenas correlacioná-las.

## Onde citar

- **Introdução / Motivação:** ao argumentar que LLMs possuem estrutura interna estável e linear por trás de suas decisões, justificando a busca por uma métrica/critério geométrico aprendível.
- **Trabalhos Relacionados:** como referência central sobre representações lineares de conceitos em LLMs ("linear world models" e probing de verdade), e para distinguir a abordagem da dissertação (otimização inversa sobre classificações observadas) das sondas treinadas sobre ativações internas.
- **Metodologia / Fundamentação da métrica de Mahalanobis:** ao introduzir a métrica diagonal W, citando o mass-mean probing como precedente que usa diferença de médias corrigida por covariância (Mahalanobis/LDA de Fisher).
- **Discussão sobre transferência/consistência (Fases B, C e Bloco 2):** ao interpretar resultados de generalização de W entre geometrias, apoiando-se nos experimentos de transferência de sondas entre datasets.
- **Discussão / Limitações e Trabalhos Futuros:** ao comparar o nível de acesso (caixa-branca, ativações internas) deste artigo com a abordagem caixa-preta da dissertação (apenas decisões observadas).
