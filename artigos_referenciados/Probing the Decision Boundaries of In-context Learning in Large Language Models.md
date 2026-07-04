# Probing the Decision Boundaries of In-context Learning in Large Language Models

## Metadados
- **Autores:** Siyan Zhao, Tung Nguyen, Aditya Grover (Department of Computer Science, University of California Los Angeles — UCLA)
- **Ano:** 2024
- **Publicação/Venue:** 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Preprint arXiv:2406.11233v3 [cs.LG], 9 dez. 2024. Código: https://github.com/siyan-zhao/ICL_decision_boundary
- **Arquivo:** Probing the Decision Boundaries of In-context Learning in Large Language Models.pdf
- **Observação:** há cópia duplicada "NeurIPS-2024-probing-the-decision-boundaries-of-in-context-learning-in-large-language-models-Paper-Conference.pdf" na pasta (mesmo artigo).

## Resumo completo

O artigo propõe um novo mecanismo para sondar (probe) e entender o aprendizado em contexto (in-context learning, ICL) em grandes modelos de linguagem (LLMs), olhando o problema sob a ótica das **fronteiras de decisão** em tarefas de classificação binária. A ideia central é tratar o ICL de um LLM como se fosse um algoritmo clássico de aprendizado de máquina e, então, usar uma ferramenta tradicional — a visualização de fronteiras de decisão em 2D — para revelar os vieses indutivos e a capacidade de generalização do modelo. Fronteiras de decisão são fáceis de visualizar e informam sobre o comportamento qualitativo do classificador.

A descoberta principal e surpreendente é que as fronteiras de decisão aprendidas por LLMs atuais em tarefas simples de classificação binária são frequentemente **irregulares e não-suaves (non-smooth)**, independentemente da separabilidade linear da tarefa subjacente. Isso ocorre mesmo em problemas linearmente separáveis, nos quais métodos clássicos como SVM atingem fronteiras suaves com poucos exemplos. O fenômeno persiste em modelos de diferentes tamanhos (1,3B a 13B parâmetros), com diferentes quantidades e ordenações de exemplos in-context, e com diferentes semânticas de rótulo. Mesmo o GPT-4o, o modelo mais poderoso testado, exibe regiões de decisão fragmentadas.

Os autores investigam quais fatores influenciam a suavidade dessas fronteiras: tamanho do modelo, dados/objetivos de pré-treino, número de exemplos in-context, nível de quantização, semântica dos rótulos e ordem dos exemplos. Constatam que aumentar o tamanho do modelo melhora a acurácia de teste, mas não torna a fronteira mais suave; mais exemplos in-context também elevam a acurácia sem suavizar a fronteira; quantização (8-bit → 4-bit) pode inverter decisões nas regiões de maior incerteza; e tanto os nomes dos rótulos (ex.: "Foo/Bar" vs. "Bar/Foo") quanto a ordem dos exemplos alteram a fronteira, mostrando dependência de conhecimento prévio semântico e sensibilidade à ordenação.

Por fim, o trabalho explora métodos para **suavizar** as fronteiras. Fine-tuning sobre os próprios exemplos in-context não ajuda, mas: (i) fine-tuning de camadas mais iniciais/intermediárias (embedding) suaviza mais do que ajustar a cabeça de predição; (ii) fine-tuning em uma coleção de ~1000 tarefas sintéticas de classificação (linear/círculo/lua), e mesmo só em tarefa linear, generaliza para tarefas não-lineares e multi-classe não vistas, produzindo fronteiras mais suaves; (iii) transformers treinados do zero (TNPs) conseguem aprender fronteiras suaves in-context; e (iv) **aprendizado ativo consciente de incerteza** (selecionar e rotular os pontos do grid mais incertos por entropia) suaviza a fronteira de forma eficiente em dados, superando amostragem aleatória.

## Principais contribuições

- Introduz um mecanismo novo para sondar e entender o ICL em LLMs por meio da visualização e análise de fronteiras de decisão em tarefas de classificação binária (tratando o ICL como um algoritmo de ML).
- Demonstra que LLMs estado-da-arte exibem fronteiras de decisão não-suaves e irregulares mesmo em tarefas linearmente separáveis, ao contrário de modelos clássicos de ML (SVM, MLP, k-NN, árvore de decisão).
- Estuda a influência de diversos fatores sobre a suavidade da fronteira: tamanho do modelo, dados/objetivos de pré-treino, número de exemplos in-context, níveis de quantização, semântica dos rótulos e ordem dos exemplos.
- Identifica métodos para melhorar a suavidade: fine-tuning de camadas iniciais (embedding), fine-tuning em tarefas sintéticas e aprendizado ativo consciente de incerteza (uncertainty-aware active learning).

## Metodologia

Os autores formalizam classificação in-context: dado um conjunto de n exemplos (xi, yi) amostrados de pdata e um ponto de teste, monta-se um prompt concatenando exemplos + ponto de consulta e o LLM prevê a classe escolhendo o token de classe com maior logit. Para visualizar a fronteira, geram um grid uniforme (escala 50×50 = 2500 consultas) cobrindo o espaço de features definido pelos exemplos in-context, consultam o modelo em cada ponto do grid e plotam os rótulos previstos.

Os datasets são gerados com scikit-learn em três geometrias de fronteira: **linear, círculo (circle) e meia-lua (moon)**, sempre com classes balanceadas, mais um conjunto de teste held-out de tamanho 100 para medir acurácia. Os modelos avaliados vão de 1,3B a 13B parâmetros: open-source (Llama2-7B, Llama2-13B, Llama3-8B, Mistral-7B-v0.1, sheared-Llama-1.3B) com quantização de 8 bits, e closed-source (GPT-4o, GPT-3.5-turbo), estes via geração do próximo token. Baselines clássicos: SVM (kernel polinomial/RBF) e MLP com duas camadas ocultas; resultados reportados com erro-padrão sobre 5 sementes.

Os experimentos seguem três perguntas: (1) como LLMs pré-treinados desempenham em classificação binária; (2) como diferentes fatores influenciam a fronteira — variando tamanho do modelo, número de exemplos (8 a 256), quantização (4/8/16 bits), nomes de rótulos e ordem dos exemplos; (3) como suavizar a fronteira. Para suavização testam: fine-tuning sobre exemplos in-context; fine-tuning sobre 1000 tarefas sintéticas (com variações LoRA nas camadas de atenção, só embedding, só cabeça linear); uma arquitetura modificada "CustomLLM" (backbone congelado + novas camadas de embedding/predição MLP que operam sobre valores numéricos brutos); TNPs (Transformer Neural Processes) treinados do zero; e um esquema de aprendizado ativo que seleciona iterativamente os top-k pontos mais incertos (entropia, com amostragem gulosa espacialmente distante) rotulados por uma regressão logística treinada como verdade-fundamental.

## Conclusão do artigo

Os autores concluem que sondar as fronteiras de decisão é uma forma reveladora de entender o ICL em LLMs. Apesar de atingirem alta acurácia de teste, os LLMs produzem fronteiras de decisão frequentemente irregulares e não-suaves, o que levanta preocupações sobre a confiabilidade e a generalização do ICL em implantações práticas. Por meio de experimentos extensivos, identificam fatores que afetam essa fronteira e mostram que métodos de fine-tuning (especialmente de camadas iniciais e sobre tarefas sintéticas) e de amostragem adaptativa consciente de incerteza são eficazes para melhorar a suavidade. As descobertas oferecem novos insights sobre os mecanismos do ICL e apontam caminhos de pesquisa e otimização.

## Relação com este trabalho

Este artigo é **altamente relacionado** à dissertação "Explicando Decisões de LLMs via Otimização Inversa" — talvez a referência mais próxima em desenho experimental. Ambos tratam o LLM como um classificador e investigam empiricamente o **comportamento de suas fronteiras de decisão** em tarefas de classificação binária 2D, geradas com scikit-learn, usando exatamente as mesmas geometrias (linear, círculo/elipse, meia-lua) que aparecem nos Blocos do projeto (problemas A/B/C lineares, D meia-lua, e o estudo de caso real peso×altura com fronteira elíptica).

Pontos de contato diretos: (a) **viés semântico de rótulos** — o artigo mostra que trocar "Foo/Bar" por "Bar/Foo" inverte predições, espelhando os experimentos de viés de ordem das classes e de nomes semânticos de features (altura/peso) da dissertação; (b) **sensibilidade à ordem dos exemplos few-shot** (recency bias), também medida no projeto; (c) **estabilidade/consistência** do critério de decisão do LLM, hipótese central (H1/H5) do mestrado. A diferença metodológica é complementar e pode ser explorada como contraste: enquanto Zhao et al. visualizam a fronteira diretamente sobre um grid denso (50×50), a dissertação **aprende uma métrica de Mahalanobis diagonal W via otimização inversa** (Perceptron Estruturado/NNLS) a partir das classificações do LLM e testa transferência/consistência entre geometrias. A constatação do artigo de que as fronteiras são "não-suaves e fragmentadas" oferece um forte argumento motivador: justifica por que aproximar o critério do LLM por uma métrica simples (linear/diagonal) captura apenas a tendência global, e contextualiza os achados de consistência imperfeita do projeto. Além disso, o uso de baselines clássicos (SVM, MLP, k-NN) como referência de suavidade espelha os baselines clássicos do Bloco 2.

## Onde citar

- **Introdução / Motivação:** justificar o interesse em fronteiras de decisão de LLMs e a relevância de tratar o LLM como um algoritmo de ML; ancorar a pergunta sobre estabilidade/consistência do critério decisório.
- **Trabalhos Relacionados:** referência central na subseção sobre entendimento prático do ICL e sobre fronteiras de decisão de LLMs; situar a abordagem por otimização inversa como complementar à visualização direta de fronteiras.
- **Metodologia:** ao descrever a geração de dados sintéticos (linear/círculo/meia-lua via scikit-learn) e o desenho de classificação binária in-context.
- **Discussão de viés semântico e de ordem:** comparar os achados do projeto (viés de ordem de classes, nomes de features, recency bias) com os reportados por Zhao et al.
- **Discussão de resultados / Limitações:** ao interpretar a consistência imperfeita da métrica Ŵ_LLM e a não-suavidade das decisões do LLM como evidência convergente com este trabalho.
