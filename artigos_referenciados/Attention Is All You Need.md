# Attention Is All You Need

## Metadados
- **Autores:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research / University of Toronto)
- **Ano:** 2017 (versão arXiv 1706.03762v7, revisada em 2 de agosto de 2023)
- **Publicação/Venue:** 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, EUA
- **Arquivo:** Attention Is All You Need.pdf

## Resumo completo
O artigo introduz o **Transformer**, uma arquitetura de rede neural para problemas de transdução de sequências (sequence transduction), como tradução automática. Até então, os melhores modelos eram baseados em redes recorrentes (RNN, LSTM, GRU) ou convolucionais, geralmente combinadas a um mecanismo de atenção que ligava codificador (encoder) e decodificador (decoder). A contribuição central é mostrar que é possível dispensar **completamente** a recorrência e as convoluções, apoiando o modelo **apenas em mecanismos de atenção**.

A motivação principal é a paralelização. Modelos recorrentes processam a sequência posição a posição, gerando estados ocultos de forma intrinsecamente sequencial, o que impede paralelizar o treinamento dentro de cada exemplo e se torna crítico em sequências longas. O Transformer substitui essa computação sequencial por **auto-atenção (self-attention)**, que relaciona todas as posições de uma sequência em um número constante de operações sequenciais, encurtando o caminho entre dependências de longo alcance e permitindo muito mais paralelismo.

O mecanismo proposto é a **Scaled Dot-Product Attention**: dados conjuntos de queries (Q), keys (K) e values (V), calcula-se `softmax(QKᵀ / √dk) V`, onde a divisão por `√dk` evita que produtos escalares grandes empurrem o softmax para regiões de gradiente quase nulo. Sobre isso constrói-se a **Multi-Head Attention** (8 cabeças no modelo base), em que Q, K e V são projetados linearmente várias vezes em subespaços distintos, a atenção é calculada em paralelo em cada um, e os resultados são concatenados e reprojetados — permitindo ao modelo atender simultaneamente a informações de diferentes subespaços de representação.

A arquitetura completa mantém a estrutura encoder-decoder: o encoder tem 6 camadas idênticas (auto-atenção multi-cabeça + rede feed-forward posicional), e o decoder tem 6 camadas com uma terceira sub-camada de atenção sobre a saída do encoder, além de mascaramento que preserva a propriedade auto-regressiva. Cada sub-camada usa conexões residuais e layer normalization. Como não há recorrência nem convolução, a ordem da sequência é injetada por **codificações posicionais** (positional encodings) baseadas em funções seno e cosseno de frequências diferentes.

Nos experimentos de tradução WMT 2014, o Transformer "big" atinge **28,4 BLEU** em inglês→alemão (mais de 2 BLEU acima do melhor resultado anterior, incluindo ensembles) e **41,8 BLEU** em inglês→francês (novo estado da arte para modelo único), treinando por apenas 3,5 dias em 8 GPUs P100 — uma fração do custo dos modelos competidores. O modelo também generaliza bem para **análise sintática de constituintes (constituency parsing)** em inglês, com bons resultados mesmo em regime de poucos dados.

## Principais contribuições
- Proposta do **Transformer**, primeira arquitetura de transdução de sequências baseada **inteiramente em atenção**, eliminando recorrência e convolução.
- Introdução da **Scaled Dot-Product Attention** e da **Multi-Head Attention**, que permitem atender a múltiplos subespaços de representação em paralelo.
- Demonstração de que a auto-atenção liga quaisquer duas posições com caminho de comprimento constante O(1) e número constante de operações sequenciais, contra O(n) das RNNs — facilitando o aprendizado de dependências de longo alcance.
- Uso de **codificações posicionais sinusoidais** para representar a ordem dos tokens sem recorrência.
- Novos estados da arte em tradução (WMT 2014 EN-DE e EN-FR) com custo de treinamento muito menor, além de boa generalização para parsing sintático.
- Indício de **maior interpretabilidade**: cabeças de atenção distintas parecem capturar relações sintáticas e semânticas (visualizações no apêndice).

## Metodologia
- **Modelo:** encoder-decoder com N=6 camadas cada, dmodel=512, dff=2048, h=8 cabeças, dk=dv=64. Versão "big": dmodel=1024, dff=4096, h=16.
- **Dados:** WMT 2014 inglês-alemão (~4,5M pares de sentenças, vocabulário compartilhado de ~37k tokens via byte-pair encoding) e WMT 2014 inglês-francês (36M sentenças, vocabulário de 32k word-pieces).
- **Treinamento:** 8 GPUs NVIDIA P100; modelo base por 100 mil passos (~12 h), modelo big por 300 mil passos (~3,5 dias). Otimizador **Adam** (β1=0,9, β2=0,98, ε=10⁻⁹) com agendamento de learning rate com warmup (4000 passos).
- **Regularização:** residual dropout (Pdrop=0,1) e label smoothing (εls=0,1).
- **Inferência:** beam search (tamanho 4, penalidade de comprimento α=0,6) e média de checkpoints.
- **Estudos de ablação (Tabela 3):** variação do número de cabeças, do tamanho de dk, do tamanho do modelo, do dropout e do tipo de codificação posicional (sinusoidal vs. aprendida — resultados quase idênticos).
- **Generalização:** análise sintática de constituintes no Penn Treebank (WSJ) em regimes supervisionado (~40K sentenças) e semi-supervisionado (~17M sentenças).

## Conclusão do artigo
O Transformer é o primeiro modelo de transdução de sequências baseado exclusivamente em atenção, substituindo as camadas recorrentes por auto-atenção multi-cabeça. Ele treina significativamente mais rápido que arquiteturas recorrentes ou convolucionais e atinge novo estado da arte em tradução EN-DE e EN-FR, superando inclusive ensembles previamente reportados. Os autores manifestam intenção de estender a arquitetura a outras modalidades (imagens, áudio, vídeo) e a investigar mecanismos de atenção restrita/local para entradas e saídas muito grandes, além de tornar a geração menos sequencial. O código foi disponibilizado no repositório tensor2tensor.

## Relação com este trabalho
Este artigo é a fundação técnica da arquitetura **Transformer**, que está na base de todos os LLMs estudados na dissertação (incluindo o gpt-4o-mini usado nos experimentos). Compreender o mecanismo de atenção é essencial para enquadrar conceitualmente o objeto de estudo: quando a dissertação pergunta se um LLM possui um processo de decisão **estável e aprendível** e tenta recuperar uma métrica de Mahalanobis (W) via otimização inversa, ela está, no fundo, tratando como caixa-preta um modelo cujo comportamento emerge de camadas de auto-atenção descritas aqui.

A conexão é especialmente relevante em dois pontos. Primeiro, no **Bloco 2 (LLM como APRENDIZ)**, que investiga in-context/few-shot learning: a capacidade de aprender o critério de um perito a partir de exemplos no prompt é um comportamento emergente de arquiteturas Transformer, e este artigo fornece o substrato para discutir *por que* o mecanismo de atenção sobre os exemplos do contexto possibilita esse aprendizado. Segundo, o tema de **interpretabilidade/explicabilidade**: o artigo já observa que cabeças de atenção distintas capturam estrutura sintática e semântica, oferecendo uma forma "interna" de explicar decisões. A dissertação propõe uma abordagem **complementar e externa** — explicar decisões não inspecionando pesos de atenção, mas aprendendo uma fronteira de decisão (a métrica W) a partir das classificações observadas. Mencionar o Transformer ajuda a contrastar essas duas vias de explicabilidade (interna via atenção × externa via otimização inversa).

## Onde citar
- **Introdução / Fundamentação teórica:** ao apresentar o que é um LLM e a arquitetura Transformer que sustenta o gpt-4o-mini, citar como a referência seminal da atenção.
- **Trabalhos relacionados / Explicabilidade:** ao discutir explicabilidade de transformers via atenção versus a abordagem de otimização inversa proposta.
- **Seção sobre in-context learning (Bloco 2):** ao contextualizar few-shot learning como propriedade emergente de modelos baseados em atenção.
- **Slides de abertura/contexto:** slide introdutório que situa "o que é um LLM" e "como ele toma decisões" antes de apresentar a hipótese central do trabalho.
