# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?

## Metadados
- **Autores:** Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer (University of Washington, Meta AI, Allen Institute for AI)
- **Ano:** 2022 (versão arXiv v2, 20 de outubro de 2022; publicado na EMNLP 2022)
- **Publicação/Venue:** EMNLP 2022 / arXiv:2202.12837 [cs.CL]
- **Arquivo:** Rethinking the Role of Demonstrations- What Makes In-Context Learning Work?.pdf

## Resumo completo

O artigo investiga empiricamente **por que** o aprendizado em contexto (in-context learning, ICL) funciona em grandes modelos de linguagem (LLMs), questionando qual parte das demonstrações (os pares entrada-rótulo apresentados no prompt) realmente contribui para o ganho de desempenho sobre a inferência zero-shot. A descoberta central, contraintuitiva, é que **rótulos verdadeiros (ground truth) não são necessários** para um ICL eficaz: substituir os rótulos das demonstrações por rótulos aleatórios praticamente não prejudica o desempenho. Esse resultado se mantém consistente em 12 modelos diferentes, incluindo a família GPT-3, em tarefas de classificação e de múltipla escolha, com queda típica de apenas 0–5% absolutos.

A partir dessa observação, os autores decompõem a demonstração em quatro aspectos potenciais de sinal de aprendizado e medem o impacto de cada um isoladamente: (1) o **mapeamento entrada-rótulo** (se cada entrada está pareada com o rótulo correto); (2) a **distribuição do texto de entrada** (de qual distribuição vêm as entradas); (3) o **espaço de rótulos** (o conjunto de rótulos possíveis exibidos); e (4) o **formato** — especificamente o uso do pareamento entrada-rótulo como estrutura do prompt. As experiências mostram que o mapeamento correto entrada-rótulo importa muito pouco, enquanto o espaço de rótulos, a distribuição das entradas e o formato são os verdadeiros motores do desempenho.

Os autores mostram que basta especificar o espaço de rótulos e a distribuição das entradas, no formato correto, para reter a maior parte do ganho do ICL. Por exemplo, com o Direct MetaICL é possível reter cerca de 95% do ganho apenas amostrando frases aleatórias de um corpus e pareando-as aleatoriamente com o conjunto de rótulos; com modelos channel, é possível reter de 75% a 87% do ganho pareando cada entrada não rotulada com uma palavra inglesa aleatória. Remover o formato (apenas entradas, ou apenas rótulos, sem o pareamento) aproxima o desempenho do nível sem demonstrações, evidenciando a importância do formato.

Adicionalmente, o artigo analisa o efeito do **meta-treinamento** com objetivo explícito de ICL (modelo MetaICL): esse treinamento amplifica todos os efeitos observados — o modelo passa a explorar quase exclusivamente os aspectos mais simples das demonstrações (como o formato) e a ignorar quase completamente o mapeamento entrada-rótulo. O trabalho conclui propondo uma nova forma de entender como e por que o ICL funciona, sugerindo que o modelo "localiza" uma tarefa já aprendida no pré-treinamento, em vez de aprender uma nova correspondência entrada-rótulo no momento do teste.

## Principais contribuições

- Demonstra empiricamente, em 12 modelos (incluindo GPT-3, GPT-J, fairseq 13B, GPT-2 Large e MetaICL), que **substituir rótulos corretos por rótulos aleatórios nas demonstrações quase não afeta o desempenho** (queda de 0–5% absolutos), refutando a intuição de que o mapeamento entrada-rótulo correto seria essencial.
- Decompõe a demonstração em **quatro aspectos** (mapeamento entrada-rótulo, distribuição das entradas, espaço de rótulos e formato) e quantifica o impacto de cada um isoladamente por meio de variantes controladas das demonstrações.
- Identifica que **distribuição das entradas, espaço de rótulos e formato** são os reais responsáveis pelos ganhos do ICL, e que o formato (o pareamento entrada-rótulo) é crucial para reter o ganho mesmo quando apenas entradas ou apenas rótulos são fornecidos.
- Mostra que o **meta-treinamento com objetivo de ICL** intensifica esses efeitos, levando o modelo a explorar aspectos simples (formato/distribuição) e a ignorar quase totalmente o mapeamento entrada-rótulo.
- Eleva o **patamar do baseline zero-shot**: é possível atingir desempenho próximo ao k-shot sem nenhum dado rotulado, apenas pareando entradas não rotuladas com rótulos aleatórios no formato correto.
- Disponibiliza código reprodutível (github.com/Alrope123/rethinking-demonstrations).

## Metodologia

Os autores avaliam 12 configurações de modelo: 6 LLMs decoder-only densos (GPT-2 Large 774M, MetaICL 774M, GPT-J 6B, fairseq 6.7B, fairseq 13B e GPT-3 175B), cada um com dois métodos de inferência (**direct** e **channel**, seguindo Min et al. 2021a). As avaliações cobrem 26 datasets de classificação e múltipla escolha (análise de sentimento, detecção de paráfrase, inferência de linguagem natural, detecção de discurso de ódio, QA e completamento de sentenças), todos de baixo recurso (menos de 10K exemplos de treino) e de domínios diversos.

Por padrão, usam-se k = 16 exemplos como demonstrações, amostrados uniformemente do conjunto de treino, com 5 sementes aleatórias e 5 execuções (3 sementes e 6 datasets para fairseq 13B e GPT-3, por limitação de recursos). Reportam-se Macro-F1 para classificação e acurácia para múltipla escolha, com média por dataset sobre as sementes e, depois, macro-média entre datasets. Usam-se templates mínimos por padrão (e templates manuais como ablação).

A análise central compara três condições: (1) **sem demonstrações** (zero-shot); (2) **demonstrações com rótulos corretos (gold)**; e (3) **demonstrações com rótulos aleatórios** amostrados uniformemente do espaço de rótulos. Ablações adicionais variam: o número de rótulos corretos (0%, 25%, 50%, 75%, 100%), o número de exemplos k (4, 8, 16, 32), e o uso de templates manuais. Para isolar os quatro aspectos, criam-se variantes de demonstração: **OOD inputs** (entradas fora da distribuição, para medir o impacto da distribuição de entrada), **palavras inglesas aleatórias como rótulos** (para medir o espaço de rótulos), **demonstrações sem rótulos** e **demonstrações só com rótulos** (para medir o impacto do formato/pareamento). O modelo MetaICL serve para analisar o efeito do meta-treinamento.

## Conclusão do artigo

O mapeamento entrada-rótulo correto nas demonstrações importa muito menos do que se supunha — o ganho do ICL vem principalmente da especificação independente do espaço de entrada e do espaço de rótulos, e o modelo pode reter até 95% do ganho usando apenas as entradas ou apenas os rótulos, desde que o formato correto seja preservado; o meta-treinamento amplifica esses efeitos. Sob uma definição estrita de aprendizado (capturar a correspondência entrada-rótulo dos dados de treino), os LLMs **não aprendem uma nova tarefa no momento do teste** — em vez disso, usam conhecimento prévio do pré-treinamento, "localizando" uma capacidade já existente. Sob uma definição mais ampla, o modelo adapta-se às distribuições de entrada/rótulo e ao formato sugeridos pelas demonstrações. Isso implica que o ICL pode não funcionar em tarefas cuja correspondência entrada-rótulo não tenha sido capturada no pré-treinamento, e que o baseline zero-shot é mais alto do que se pensava. Limitações: o estudo cobre classificação e múltipla escolha (não geração), e datasets/tarefas diferentes podem se comportar de modo distinto (alguns pares modelo-dataset mostram lacunas maiores entre rótulos corretos e aleatórios).

## Relação com este trabalho

Este artigo é diretamente central para o **Bloco 2 (LLM como APRENDIZ)** da dissertação, em que o LLM gpt-4o-mini recebe exemplos few-shot (in-context) de um perito externo e deve reproduzir suas classificações. As descobertas têm implicações concretas:

- **Papel das demonstrações vs. estratégias de seleção de exemplos:** O trabalho de Min et al. mostra que a distribuição das entradas e o formato importam mais do que a correção dos rótulos. Isso dialoga com as estratégias **easy/hard/mixed/random** da Fase E e com a hipótese H4 (refutada) de que exemplos difíceis superariam os fáceis — ambos discutem qual propriedade das demonstrações de fato carrega o sinal de aprendizado.
- **Viés semântico de rótulos e prompts:** A análise do espaço de rótulos e do uso de "palavras inglesas aleatórias" como rótulos conecta-se diretamente aos experimentos de **nomes de classe variados/invertidos** e de **nomes de features semânticos** (altura/peso vs x1/x2) e variantes de prompt do Bloco 1.
- **Estabilidade do critério de decisão:** A tese central da dissertação — que o LLM tem um processo de decisão estável e aprendível, recuperável via otimização inversa de uma métrica de Mahalanobis W — ressoa com a conclusão de Min et al. de que o ICL "localiza" uma capacidade já presente no pré-treinamento, em vez de aprendê-la no teste. Isso oferece base teórica para interpretar a métrica Ŵ_LLM estimada como reflexo de um critério pré-existente.
- **Baseline zero-shot elevado:** A observação de que o desempenho zero-shot é mais alto do que se pensava reforça a importância da Fase A (zero-shot) como ponto de partida e da comparação com baselines clássicos no Bloco 2.

## Onde citar

- **Introdução / Motivação:** ao justificar por que estudar o papel das demonstrações e a estabilidade do critério de decisão do LLM.
- **Trabalhos Relacionados / Fundamentação:** seção sobre in-context learning e o que faz o ICL funcionar (papel do mapeamento entrada-rótulo, espaço de rótulos, formato).
- **Bloco 2 (LLM como APRENDIZ) — metodologia e discussão:** ao apresentar as estratégias de seleção de exemplos (easy/hard/mixed/random) e ao interpretar a refutação de H4.
- **Discussão de viés semântico:** nas seções sobre nomes de classe, nomes de features e variantes de prompt (Bloco 1).
- **Conclusão / Discussão:** ao argumentar que o LLM "localiza" um critério pré-existente em vez de aprender no teste, sustentando a interpretação da métrica Ŵ_LLM recuperada por otimização inversa.
