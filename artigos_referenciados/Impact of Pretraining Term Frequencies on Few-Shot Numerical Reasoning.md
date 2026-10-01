# Impact of Pretraining Term Frequencies on Few-Shot Numerical Reasoning

## Metadados
- **Autores:** Yasaman Razeghi (University of California, Irvine), Robert L. Logan IV (Dataminr Inc.; trabalho realizado na UC Irvine), Matt Gardner (Microsoft Semantic Machines), Sameer Singh (University of California, Irvine; Allen Institute for AI)
- **Ano:** 2022 (preprint arXiv:2202.07206, v1 em fev/2022 e v2 em mai/2022, com o título "Impact of Pretraining Term Frequencies on Few-Shot Reasoning"; versão publicada em dez/2022)
- **Publicação/Venue:** Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, pp. 840–854, Association for Computational Linguistics. DOI 10.18653/v1/2022.findings-emnlp.59
- **Arquivo:** Impact of Pretraining Term Frequencies on Few-Shot Numerical Reasoning.pdf
- **PDF público:** https://aclanthology.org/2022.findings-emnlp.59.pdf

## Resumo completo

O artigo pergunta se o bom desempenho de modelos de linguagem em **raciocínio numérico com poucos exemplos** (*few-shot*, via aprendizado em contexto) reflete uma capacidade de raciocínio robusta ou se depende das estatísticas do corpus de pré-treinamento. A intuição de partida (Seção 1) é simples: um modelo que aprendeu de fato a multiplicar deveria responder "24 vezes X" e "23 vezes X" com a mesma acurácia; na prática, a acurácia do GPT-J-6B em multiplicação cresce de forma quase monotônica com a frequência do operando no corpus (Figura 1), e a diferença entre 24 e 23, que diferem apenas em quantas vezes aparecem no pré-treinamento, passa de 20 pontos.

Para medir isso, os autores contam, no corpus de pré-treinamento, quantas vezes os **termos** de cada instância de teste aparecem: um termo isolado (`ω_{x1}`, por exemplo o número 23), um par de termos de entrada dentro de uma janela fixa (`ω_{x1,x2}`) e o par entrada-resposta (`ω_{x1,y}`). A partir dessas contagens definem o ***performance gap*** (Seção 2.3):

```
Δ(Ω) = Acc(Ω_{>90%}) − Acc(Ω_{<10%}),
```

a diferença entre a acurácia média nas instâncias cujos termos estão entre os 10% mais frequentes e nas instâncias cujos termos estão entre os 10% menos frequentes. Um modelo que raciocina de forma independente da frequência deveria ter Δ próximo de zero.

Os resultados mostram **correlação forte e consistente** entre a frequência dos termos e a acurácia, em todos os modelos e tarefas, com gaps que em alguns casos ultrapassam **70 pontos percentuais** absolutos. Para separar o efeito da frequência do efeito de memorização direta, os autores removem as instâncias cujas combinações de termos são frequentes no corpus (isto é, as que poderiam ter sido decoradas) e mostram que a correlação persiste (Figura 7): o fenômeno não se explica apenas por memorização. Observam ainda que mesmo os termos "raros" do estudo aparecem milhões de vezes no corpus, ou seja, não se trata de tokens desconhecidos.

## Principais contribuições

- Definição do ***performance gap*** como métrica para quantificar quanto o desempenho de um modelo em uma tarefa depende da **frequência dos termos das instâncias no corpus de pré-treinamento**.
- Evidência empírica, em três tamanhos de modelo e em várias tarefas numéricas, de que a acurácia *few-shot* em raciocínio numérico **acompanha a frequência dos números no pré-treinamento**, com gaps de dezenas de pontos.
- Demonstração de que o efeito **sobrevive à remoção das instâncias memorizadas**, portanto vai além da memorização direta de pares pergunta-resposta.
- Argumento metodológico de que verificações de sobreposição por n-gramas longos (como a checagem de 13-gramas de Brown et al., 2020) são insuficientes: estatísticas de **unigramas** já bastam para influenciar o desempenho.

## Metodologia

Os modelos avaliados são três tamanhos da família EleutherAI pré-treinados no **Pile** (GPT-Neo-1.3B, GPT-Neo-2.7B e GPT-J-6B), escolhidos justamente porque o corpus é público e permite contar frequências, ao contrário do GPT-3. As tarefas são de dedução numérica: aritmética (adição e multiplicação, com operandos como os inteiros de 0 a 99) e conversão de unidades, inclusive de tempo, formuladas como perguntas do tipo "Q: What is 24 times 18? A:" com poucos exemplos no *prompt* (por exemplo, 2-shot). Para cada tarefa, instanciam-se os termos, mede-se a acurácia média por termo (sobre os demais operandos e cinco sementes) e calcula-se o gap entre os decis extremos de frequência, para as contagens de unigramas e de coocorrências.

## Conclusão do artigo

Os autores concluem que o desempenho em contexto em raciocínio numérico é afetado de forma significativa por estatísticas de coocorrência de baixa ordem do corpus de pré-treinamento, o que põe em dúvida quanto os modelos de fato raciocinam nessas tarefas. Recomendam que qualquer avaliação de raciocínio leve em conta o corpus de pré-treinamento e que a comunidade deixe de tratar esses dados como caixa-preta. Não fazem uma afirmação causal (há confundidores não eliminados) e reconhecem como limitações o uso de padrões simples de frequência, de tarefas numéricas simples e a ausência de uma solução para tornar os modelos robustos.

## Relação com este trabalho

Todo o protocolo da dissertação apresenta os pontos ao LLM como **números decimais com quatro casas** (`x1 = -0.4321`, gerados em `build_prompt_zero_shot` e `build_prompt_few_shot`, em `src/dissertacao_mestrado.py`), e no Bloco 3 os atributos reais chegam normalizados em `[-1, 1]`, sem unidade. Esses valores são termos raros no pré-treinamento, muito mais raros do que inteiros pequenos ou do que alturas em metros e pesos em quilogramas. O artigo fornece a evidência de que o desempenho numérico *few-shot* de LLMs depende da frequência desses termos, o que tem três consequências para a leitura dos resultados:

- **Bloco 1 (Problemas A a D, LLM como fonte):** a representação numérica é uma fonte adicional de irregularidade no critério do LLM, ao lado das fronteiras irregulares observadas por Zhao, Nguyen e Grover (2024). Ajuda a explicar por que a fidelidade de Ŵ_LLM fica abaixo de 100% mesmo em problemas linearmente separáveis, e complementa, pelo lado das **entradas**, a terceira fonte de viés de Zhao, Wallace et al. (2021), a frequência de *tokens* no pré-treinamento, que naquele artigo se refere aos **rótulos**.
- **Bloco 3 (peso × altura):** oferece uma hipótese para a inversão física observada (o LLM rotula como "Homem" os pontos mais baixos e leves): a normalização afasta os números da distribuição vista no pré-treinamento e degrada o raciocínio numérico. É uma hipótese compatível com o achado, não uma explicação demonstrada.
- **Limitações:** justifica registrar a natureza numérica das entradas como limitação do protocolo, distinguindo o que pode vir do critério geométrico do que pode vir da representação dos números.

**Cuidado de fidelidade:** o artigo mede o efeito da frequência dos **operandos**, não dos nomes de rótulo. Não deve ser usado para explicar a queda de fidelidade com as classes `0/1` do Bloco 1, pois "0" e "1" são tokens frequentíssimos; esse efeito continua ancorado no *common token bias* de Zhao, Wallace et al. (2021).

## Onde citar

- **Fundamentação Teórica, §Viés Semântico em LLMs:** logo após a frase que lista as três fontes de viés de Zhao et al. (2021), como o trabalho que quantifica a terceira fonte para entradas numéricas.
- **Trabalhos Relacionados, §Aprendizado em Contexto:** ao lado de Zhao et al. (2021), como a linha empírica sobre dependência do pré-treinamento em tarefas numéricas.
- **Resultados, Bloco 3:** como hipótese compatível com a inversão física nos dados normalizados.
- **Limitações:** item sobre entradas numéricas (números decimais normalizados como termos raros).
