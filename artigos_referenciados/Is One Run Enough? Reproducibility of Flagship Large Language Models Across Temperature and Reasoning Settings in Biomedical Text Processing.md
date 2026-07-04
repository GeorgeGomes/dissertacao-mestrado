# Is One Run Enough? Reproducibility of Flagship Large Language Models Across Temperature and Reasoning Settings in Biomedical Text Processing

## Metadados
- **Autores:** Paul Windisch, Carole Koechli, Fabio Dennstädt, Daniel M. Aebersold, Daniel R. Zwahlen, Robert Förster, Christina Schröder (Department of Radiation Oncology, Cantonal Hospital Winterthur, e Inselspital, Bern University Hospital, University of Bern, Suíça)
- **Ano:** 2026 (preprint medRxiv, versão postada em 3 de fevereiro de 2026; não certificado por revisão por pares)
- **Publicação/Venue:** medRxiv preprint — doi: https://doi.org/10.64898/2026.02.02.26345352. Código e dados disponíveis em https://github.com/windisch-paul/llm_reproducibility
- **Arquivo:** Is One Run Enough? Reproducibility of Flagship Large Language Models Across Temperature and Reasoning Settings in Biomedical Text Processing.pdf

## Resumo completo

O artigo investiga uma pergunta prática e cada vez mais cobrada por revisores de periódicos biomédicos: ao reportar o desempenho de um LLM em uma tarefa, **basta uma única execução** ou é necessário repetir as chamadas para obter estimativas estáveis? Os autores observam que a maioria dos estudos de NLP biomédico reporta métricas (acurácia, F1) a partir de uma única invocação do modelo por entrada, assumindo implicitamente comportamento determinístico, apesar de os LLMs modernos usarem decodificação probabilística e, cada vez mais, mecanismos adaptativos de "esforço de raciocínio" (reasoning effort / thinking level) que podem introduzir variabilidade estocástica entre execuções idênticas.

Para responder à pergunta, os autores quantificam a **reprodutibilidade execução-a-execução** (run-to-run) de dois modelos comerciais de ponta — Gemini 3 Flash Preview (Google) e GPT-5.2 (OpenAI) — em uma tarefa de classificação binária de "sucesso de ensaio clínico" (POSITIVE/NEGATIVE). Usaram 250 abstracts de ensaios clínicos randomizados (RCTs) em oncologia, com rótulos de referência herdados de um estudo anterior do mesmo grupo. Cada configuração foi executada **três vezes**, varrendo sistematicamente a temperatura (0,0 a 2,0) e os níveis de raciocínio/pensamento suportados por cada modelo. A reprodutibilidade foi medida com o **Kappa de Fleiss** entre as três réplicas (endpoint primário), e o desempenho com F1 por execução e por voto majoritário (endpoint secundário), registrando também a taxa de saídas em formato inválido.

Os resultados mostram concordância quase perfeita em ambos os modelos. O Gemini teve Kappa de Fleiss entre 0,942 e 1,000, com concordância perfeita (κ=1,000) em temperatura 0 para todos os níveis de pensamento; a menor concordância (κ=0,942) ocorreu em pensamento "medium" com temperatura 2,0. O GPT-5.2 foi ainda mais estável (κ=0,984 a 0,995), sem nenhuma saída inválida. O desempenho ficou estável (F1 médio/majoritário entre 0,955 e 0,971) e o voto majoritário sobre três execuções trouxe ganhos apenas marginais. Curiosamente, o F1 médio do GPT-5.2 aumentou com mais esforço de raciocínio sem perder reprodutibilidade, o que não ocorreu com o Gemini.

A conclusão central é que, para classificação binária estrita com **espaço de saída fortemente restringido**, uma única execução costuma ser adequada, e a replicação mínima (poucas repetições) funciona como uma checagem prática de estabilidade — sem alterar materialmente as conclusões. Os autores ressalvam, porém, que isso não deve ser generalizado para tarefas de saída aberta (recomendações clínicas, sumarização, extração de campos múltiplos), onde a variabilidade tende a ser muito maior.

## Principais contribuições

- Trata a **reprodutibilidade como endpoint primário** (e não como detalhe secundário), usando uma estatística de concordância corrigida por acaso (Kappa de Fleiss) sobre execuções repetidas, com o desempenho (F1) como métrica de contexto.
- Avalia conjuntamente **dois eixos de configuração** — temperatura de decodificação e nível de raciocínio/pensamento — que estudos anteriores examinavam isoladamente, produzindo um "mapa de configurações" diretamente acionável para quem projeta pipelines de NLP biomédico.
- Cobre **dois modelos comerciais de ponta** (Gemini 3 Flash Preview e GPT-5.2) varrendo toda a faixa de parâmetros suportada por cada fornecedor.
- Reporta explicitamente as **taxas de saída inválida** ao lado do desempenho, separando "confiabilidade de formato" de "correção da classificação".
- Oferece orientação empírica para a questão prática "uma execução basta?", sugerindo replicação mínima como checagem de estabilidade e propondo políticas de replicação adaptativa (rodar uma vez por padrão; disparar execuções extras quando a confiança é baixa ou um ensemble inicial discorda).

## Metodologia

- **Dados:** 250 RCTs de oncologia (de sete grandes periódicos: BMJ, JAMA, JAMA Oncology, JCO, Lancet, Lancet Oncology, NEJM), publicados entre 2005 e 2023, restritos a desenhos de exatamente dois braços e um único endpoint primário. Rótulos de referência (POSITIVE se o endpoint primário foi atingido, NEGATIVE caso contrário) herdados de anotação dupla com resolução por consenso, sem reanotação no presente estudo.
- **Tarefa e prompt:** classificação binária com prompt de sistema fixo exigindo saída de exatamente uma palavra em maiúsculas (POSITIVE ou NEGATIVE); o prompt do usuário era o título + abstract. Respostas só eram válidas se, após remover espaços, fossem exatamente POSITIVE ou NEGATIVE.
- **Modelos:** GPT-5.2 (snapshot gpt-5.2-2025-12-11) e Gemini 3 Flash (gemini-3-flash-preview), via APIs dos fornecedores em pipeline local.
- **Varredura de configurações:** Gemini — todos os níveis de pensamento (minimal, low, medium, high) × temperaturas {0, 0,5, 1,0, 1,5, 2,0}, todas as permutações. GPT-5.2 — todos os níveis de reasoning effort (none, low, medium, high, xhigh); como o GPT-5.2 só permite especificar temperatura quando reasoning=none, a varredura completa de temperatura foi feita apenas nesse caso, e os demais níveis avaliados sem temperatura especificada. Nenhum seed foi fixado, para capturar a variabilidade natural.
- **Repetições:** todo o dataset foi rodado **três vezes** por combinação modelo×configuração, gerando três rótulos réplica por ensaio.
- **Análise estatística:** Kappa de Fleiss entre as três réplicas (reprodutibilidade, endpoint primário). Desempenho (endpoint secundário): F1 por execução, F1 médio e F1 por voto majoritário (rótulo escolhido se ao menos 2 de 3 réplicas concordavam); apenas predições válidas/bem-formatadas entraram no cálculo do F1.

## Conclusão do artigo

Para um problema de classificação binária biomédica restrita, com formatação de saída estrita, ambos os LLMs de ponta avaliados exibiram reprodutibilidade execução-a-execução quase perfeita em uma ampla faixa de configurações de temperatura e raciocínio, e o voto majoritário sobre três execuções ofereceu ganhos mínimos de desempenho. Isso sugere que o reporte de uma única execução pode ser frequentemente adequado para tarefas e prompts semelhantes, sobretudo em estocasticidade baixa a moderada. Os autores não contestam, porém, a recomendação mais ampla de que os pesquisadores devem sempre **reportar explicitamente as configurações de decodificação e raciocínio** e, quando viável, incluir ao menos uma checagem mínima de estabilidade baseada em replicação, para proteger contra variabilidade oculta e permitir comparações confiáveis entre estudos. Limitações reconhecidas: tarefa única binária com saída restrita (não generalizável a geração aberta), avaliação em um único snapshot/janela de execução (não cobre reprodutibilidade longitudinal entre atualizações de modelo), confusão potencial no GPT-5.2 por não poder fixar temperatura com raciocínio ligado, e dataset curado para reduzir ambiguidade (pode superestimar a reprodutibilidade frente a entradas mais "bagunçadas" do mundo real).

## Relação com este trabalho

Este artigo é diretamente relevante para a dissertação "Explicando Decisões de LLMs via Otimização Inversa", em vários pontos:

- **Justificativa para repetir experimentos / tirar média:** O trabalho usa 3 seeds × múltiplas repetições (N_REPETICOES=3) justamente para robustez estatística, atendendo ao pedido do orientador de rodar mais de uma vez e tirar média. Este artigo fornece evidência empírica e linguagem para justificar essa escolha: a replicação mínima serve como "checagem prática de estabilidade", mesmo quando uma execução já seria adequada.
- **Temperatura 0 e (não-)determinismo:** A dissertação usa gpt-4o-mini com temperatura 0,0 para minimizar a estocasticidade, observando (Reunião 1) que isso não garante determinismo. Este artigo corrobora exatamente esse ponto — encontra concordância **perfeita** (κ=1,000) em temperatura 0 para o Gemini, mas mostra que, fora de tarefas de saída super-restrita, a temperatura 0 não é garantia de reprodutibilidade perfeita; a estabilidade depende fortemente do formato/restrição da saída.
- **Estabilidade do critério de decisão (H5):** A hipótese H5 da dissertação afirma que o comportamento do LLM é reproduzível entre execuções. Este artigo dá suporte externo: para classificação binária com saída restrita (análoga à tarefa "Classe A ou B" da Fase A), os LLMs são altamente reprodutíveis entre execuções.
- **Efeito do formato de saída:** A dissertação restringe o LLM a responder com rótulos curtos (A/B, 0/1) e usa um parser de 7 camadas + fallback. O artigo reforça que "temperatura-induced instability" é fortemente dependente da tarefa e do formato — quando a saída é estritamente restrita a duas palavras, a variabilidade colapsa. Isso valida o desenho de prompt da dissertação como uma escolha que naturalmente reduz a variabilidade medida.
- **Limite de generalização:** O artigo lembra que reprodutibilidade alta em tarefa binária restrita não se estende a saídas abertas — útil para a discussão de ameaças à validade da dissertação.

## Onde citar

- **Introdução / Motivação:** ao introduzir a questão de reprodutibilidade e (não-)determinismo de LLMs e ao justificar por que o trabalho repete execuções (3 seeds × N repetições) e tira médias.
- **Metodologia / Notas de Design:** ao explicar a escolha de temperatura = 0,0 e a observação de que temperatura 0 não garante determinismo; e ao justificar o número de repetições como "checagem de estabilidade".
- **Discussão da Hipótese H5 (Estabilidade) e da H1 (Consistência):** como evidência externa de que LLMs em classificação binária com saída restrita são altamente reprodutíveis entre execuções (Kappa de Fleiss próximo de 1).
- **Trabalhos relacionados / Reprodutibilidade de LLMs:** como referência central de um estudo que trata reprodutibilidade como endpoint primário e varre temperatura × esforço de raciocínio.
- **Ameaças à validade / Limitações:** ao discutir os limites de generalizar achados de estabilidade obtidos em tarefas de saída restrita.
