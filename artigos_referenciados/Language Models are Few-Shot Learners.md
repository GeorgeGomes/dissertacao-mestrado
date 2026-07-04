# Language Models are Few-Shot Learners

## Metadados
- **Autores:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (OpenAI)
- **Ano:** 2020 (versão arXiv v4, 22 de julho de 2020)
- **Publicação/Venue:** Advances in Neural Information Processing Systems 33 (NeurIPS 2020); pré-print arXiv:2005.14165 [cs.CL]
- **Arquivo:** Language Models are Few-Shot Learners.pdf

## Resumo completo

O artigo apresenta o **GPT-3**, um modelo de linguagem autorregressivo com **175 bilhões de parâmetros** (10× maior que qualquer modelo de linguagem denso anterior), e investiga sua capacidade de **aprendizado em poucos exemplos (few-shot learning)**. O ponto de partida é uma limitação do paradigma dominante na época — pré-treinar em um grande corpus e depois fazer *fine-tuning* específico de tarefa: esse processo ainda exige milhares a centenas de milhares de exemplos rotulados por tarefa. Em contraste, humanos aprendem novas tarefas linguísticas a partir de poucas demonstrações ou de uma simples instrução em linguagem natural. Os autores mostram que **escalar o tamanho do modelo** melhora substancialmente o desempenho *task-agnostic* em few-shot, às vezes chegando a competir com abordagens de *fine-tuning* estado-da-arte.

A contribuição metodológica central é o conceito de **in-context learning (aprendizado em contexto)**: o GPT-3 é avaliado **sem nenhuma atualização de gradiente ou fine-tuning** — a tarefa e as demonstrações são fornecidas inteiramente via texto, dentro da janela de contexto do modelo. Os autores definem três regimes ao longo de um espectro de quanta informação específica de tarefa é dada: **zero-shot** (apenas uma instrução em linguagem natural), **one-shot** (uma demonstração) e **few-shot** (tipicamente 10 a 100 demonstrações que cabem na janela de contexto de 2048 tokens). Esse processo é enquadrado como **meta-aprendizado** com estrutura de laço interno/externo: o laço externo é o pré-treinamento (que ajusta os pesos), e o laço interno é o in-context learning, que ocorre dentro de um único *forward pass*, sem alterar os pesos.

O GPT-3 é avaliado em mais de duas dúzias de datasets de NLP, além de várias tarefas novas projetadas para testar adaptação rápida e raciocínio "on-the-fly". Os resultados mostram desempenho forte em tradução, perguntas e respostas, tarefas cloze, raciocínio de senso comum, compreensão de leitura e SuperGLUE; e desempenho promissor em tarefas sintéticas como desembaralhar palavras, aritmética de 3 dígitos e usar uma palavra nova numa frase após vê-la definida apenas uma vez. Um achado recorrente: **a diferença entre zero-, one- e few-shot cresce com o tamanho do modelo**, sugerindo que modelos maiores são meta-aprendizes mais proficientes (as "curvas de aprendizado em contexto" são mais íngremes para modelos grandes).

O trabalho também identifica **limitações**: o GPT-3 ainda tem desempenho fraco em algumas tarefas (ex.: inferência de linguagem natural ANLI, WIC, e algumas tarefas de compreensão de leitura como RACE/QuAC), perde coerência em textos longos, e tem dificuldade com "física de senso comum". Além disso, os autores realizam um estudo sistemático de **contaminação de dados** (sobreposição treino-teste no Common Crawl), discutindo seu efeito sobre os resultados. Por fim, mostram que humanos têm dificuldade em distinguir notícias geradas pelo GPT-3 das escritas por pessoas, e discutem os **impactos sociais** (uso malicioso, viés, justiça, consumo de energia).

## Principais contribuições

- Introdução do **GPT-3**, modelo autorregressivo de 175 bilhões de parâmetros, e treinamento de uma família de 8 modelos (de 125M a 175B parâmetros) para estudar o efeito da escala.
- Formalização e estudo sistemático do **in-context learning** nos regimes zero-shot, one-shot e few-shot, **sem atualização de pesos** — apenas demonstrações no contexto textual.
- Demonstração empírica de que o **few-shot melhora dramaticamente com a escala do modelo**, com a vantagem do few-shot sobre o zero-shot crescendo conforme o modelo aumenta.
- Avaliação ampla em 20+ datasets de NLP, em alguns casos atingindo ou superando estado-da-arte de modelos com *fine-tuning* (ex.: TriviaQA few-shot; CoQA).
- Desenvolvimento de ferramentas para **medir e quantificar contaminação de dados** (sobreposição treino-teste) e caracterização honesta de seus efeitos.
- Análise dos **limites e fraquezas** do paradigma e discussão de **impactos sociais** (desinformação, viés, energia).

## Metodologia

**Modelo e arquitetura.** Mesma arquitetura do GPT-2 (inicialização modificada, pré-normalização, tokenização reversível), com a adição de padrões de atenção densa e esparsa localmente alternados (estilo Sparse Transformer). Foram treinados 8 modelos de tamanhos crescentes (125M, 350M, 760M, 1.3B, 2.7B, 6.7B, 13B e 175B parâmetros), todos com janela de contexto de **2048 tokens** e treinados por um total de **300 bilhões de tokens**.

**Dados de treinamento.** Mistura curada de Common Crawl filtrado (~410B tokens, 60% do mix), WebText2 (22%), Books1 (8%), Books2 (8%) e Wikipedia em inglês (3%). Filtragem por similaridade a corpora de referência de alta qualidade, deduplicação fuzzy em nível de documento, e amostragem ponderada (datasets de maior qualidade amostrados mais vezes).

**Regimes de avaliação (sem fine-tuning).** Para cada tarefa, o GPT-3 é avaliado em três condições, contrastadas com o *fine-tuning* tradicional:
- **Zero-shot:** apenas uma instrução em linguagem natural, nenhuma demonstração.
- **One-shot:** uma única demonstração + instrução.
- **Few-shot:** K demonstrações (tipicamente 10–100, limitado pela janela de contexto) + um exemplo final cujo completamento o modelo deve prever.

Nenhum regime envolve atualização de gradiente — toda "aprendizagem" ocorre via condicionamento no contexto (*forward pass*). O *fine-tuning* tradicional é descrito como referência, mas deliberadamente **não** é aplicado ao GPT-3, pois o foco é o desempenho *task-agnostic*.

**Controle de validade.** Estudo sistemático de contaminação treino-teste, com remoção (parcial, devido a um bug de filtragem reconhecido pelos autores) das sobreposições e relato dos casos potencialmente inflados.

## Conclusão do artigo

Os autores concluem que um modelo de linguagem de 175 bilhões de parâmetros exibe **desempenho forte em muitas tarefas de NLP nos regimes zero-shot, one-shot e few-shot**, em alguns casos quase igualando sistemas estado-da-arte com *fine-tuning*, além de gerar amostras de alta qualidade e bom desempenho qualitativo em tarefas definidas "on-the-fly". Documentaram **tendências de escala aproximadamente previsíveis** no desempenho, sem usar *fine-tuning*. Apesar de muitas limitações e fraquezas (coerência em textos longos, tarefas comparativas, ausência de bidirecionalidade, baixa eficiência amostral no pré-treino, custo de inferência, falta de *grounding* no mundo real, baixa interpretabilidade e herança de vieses dos dados), os resultados sugerem que **modelos de linguagem muito grandes podem ser um ingrediente importante no desenvolvimento de sistemas de linguagem adaptáveis e gerais**. Os autores destacam ainda uma incerteza conceitual: não é claro se o few-shot "aprende tarefas do zero" no tempo de inferência ou se apenas **reconhece tarefas já vistas durante o pré-treino** — uma direção importante para pesquisa futura.

## Relação com este trabalho

Este artigo é a **base teórica direta** do conceito de *in-context learning* / *few-shot learning* usado na dissertação, especialmente no **Bloco 2 — LLM como APRENDIZ** (Fase E). Naquela fase, o LLM (gpt-4o-mini, temperatura 0) recebe exemplos rotulados por um perito externo (métrica W conhecida) e deve reproduzir o critério de classificação **sem qualquer atualização de pesos** — exatamente o regime few-shot definido por Brown et al. As estratégias de seleção de exemplos da dissertação (easy/hard/mixed/random) e o estudo das **curvas de aprendizado em função do número de demonstrações** (`FEW_SHOT_SIZES = [0, 5, 10, 20, 40]`) são operacionalizações diretas das "in-context learning curves" introduzidas neste paper, inclusive o caso n=0 (zero-shot) usado na **Fase A** do Bloco 1.

Há também conexões com temas transversais da dissertação:
- A **incerteza levantada por Brown et al.** — se o LLM aprende de fato ou apenas reconhece padrões vistos no pré-treino — é central para a hipótese H1 da dissertação (o LLM tem um critério de decisão estável e aprendível) e para o uso de **otimização inversa** para recuperar a métrica implícita W.
- A observação dos autores de que as decisões do GPT são **pouco interpretáveis e calibradas** motiva a abordagem de explicabilidade da dissertação (aprender uma métrica de Mahalanobis diagonal a partir das classificações do LLM).
- A sensibilidade do GPT-3 à **formulação/frasing das tarefas e demonstrações** ressoa com os experimentos de **viés semântico de prompts**, viés de ordem das classes e nomes de features semânticos do Bloco 1.
- A discussão sobre **vieses herdados dos dados** sustenta a investigação de vieses de classe/posição do trabalho.

## Onde citar

- **Introdução / Fundamentação teórica:** ao definir *in-context learning* e *few-shot learning*, citar este artigo como origem do termo e do paradigma (zero/one/few-shot sem atualização de pesos).
- **Bloco 2 (LLM como APRENDIZ) / Fase E:** justificar o desenho experimental (demonstrações no contexto, curvas de aprendizado vs. número de shots, regime sem fine-tuning) como aplicação direta do método de Brown et al.
- **Trabalhos relacionados:** posicionar a dissertação na linha de pesquisa sobre capacidades emergentes de LLMs e meta-aprendizado.
- **Discussão / Hipótese H1:** citar a incerteza "aprende vs. reconhece" como motivação para investigar a estabilidade e a recuperabilidade do critério de decisão via otimização inversa.
- **Slides de motivação:** usar como referência ao introduzir LLMs, GPT e o conceito de few-shot para a banca; possível slide sobre "o que é in-context learning".
