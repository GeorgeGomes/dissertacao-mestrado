# E-mail para o orientador: envio dos artigos Razeghi et al. (2022) e Xie et al. (2022)

**Assunto:** Re: Razeghi et al., 2022 e Xi et al., 2022 (artigos em anexo)

---

Professor,

Seguem em anexo os dois artigos:

1. **Razeghi, Y.; Logan IV, R. L.; Gardner, M.; Singh, S. "Impact of Pretraining Term Frequencies on Few-Shot Numerical Reasoning". Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, pp. 840–854. DOI 10.18653/v1/2022.findings-emnlp.59.**
   Cópia pública: https://aclanthology.org/2022.findings-emnlp.59.pdf
   (O preprint arXiv:2202.07206, de fevereiro de 2022, tem o título sem "Numerical".)

2. **Xie, S. M.; Raghunathan, A.; Liang, P.; Ma, T. "An Explanation of In-context Learning as Implicit Bayesian Inference". International Conference on Learning Representations (ICLR 2022).**
   Cópia pública: https://arxiv.org/pdf/2111.02080
   (Entendi que "Xi et al., 2022" se refere a este; é o Xie et al. (2022) citado por Min et al. (2022) e por Garg et al. (2022) como a explicação bayesiana do aprendizado em contexto. Se for outro, me avise.)

Sobre a citação: os dois trabalhos são citados no artigo de Min et al. (2022), "Rethinking the Role of Demonstrations", que estava na lista que o senhor me passou em maio, e imagino que seja dele que vêm os nomes. No meu artigo eles ainda não apareciam, nem no texto nem nas referências; conferi a versão de 21/07 e a atual.

Aproveitei para incluir os dois na bibliografia, porque são pertinentes ao trabalho:

- **Razeghi et al. (2022)** mostram que a acurácia de modelos de linguagem em tarefas numéricas com poucos exemplos cresce com a frequência dos números no corpus de pré-treinamento, com diferenças que chegam a 70 pontos percentuais entre os termos mais e menos frequentes, mesmo depois de remover as instâncias memorizadas. Todo o nosso protocolo apresenta os pontos ao LLM como números decimais com quatro casas, e no caso real os atributos chegam normalizados em [-1, 1], sem unidade, ou seja, termos raros no pré-treinamento. A citação entrou na Fundamentação (§Viés Semântico, logo após as três fontes de viés de Zhao et al., 2021), nos Trabalhos Relacionados, nos resultados do Bloco 3 (como hipótese compatível com a inversão física nos dados normalizados) e em um novo item das Limitações, sobre entradas numéricas. Não usei o artigo para explicar a queda com os nomes de classe 0/1, porque ele trata da frequência dos operandos e "0" e "1" são tokens muito frequentes; esse efeito continua ancorado no viés de tokens comuns de Zhao et al. (2021).

- **Xie et al. (2022)** explicam o aprendizado em contexto como inferência bayesiana implícita: o modelo infere o conceito latente compartilhado pelos exemplos do prompt e prevê pela distribuição preditiva a posteriori. Entrou na Fundamentação (§Aprendizado em Contexto, após Garg et al. e Akyürek et al.), nos Trabalhos Relacionados e na interpretação da refutação de H4 no Bloco 2, ao lado de Min et al.: com poucos exemplos, as âncoras prototípicas (alta margem) dão sinal nítido sobre o critério a inferir, e os exemplos ambíguos só compensam em alto volume. Deixei claro no texto que é uma leitura compatível, não uma previsão formal da teoria.

## Anexos

1. `Impact of Pretraining Term Frequencies on Few-Shot Numerical Reasoning.pdf` (Razeghi et al., 2022).
2. `An Explanation of In-context Learning as Implicit Bayesian Inference.pdf` (Xie et al., 2022).
3. `artigo.pdf` atualizado, com as duas referências e as citações acima.
4. `artigo_mudancas.pdf`, com as alterações marcadas em relação à versão de 21/07.

Abraços,
George
