# Learning a Distance Metric from Relative Comparisons

## Metadados
- **Autores:** Matthew Schultz, Thorsten Joachims (Department of Computer Science, Cornell University)
- **Ano:** 2003
- **Publicação/Venue:** Advances in Neural Information Processing Systems 16 (NIPS 2003), MIT Press
- **Arquivo:** Learning a Distance Metric from Relative Comparisons.pdf
- **PDF público:** https://proceedings.neurips.cc/paper/2003/file/d3b1fb02964aa64e257f9f26a31f72cf-Paper.pdf

## Resumo completo

O artigo propõe aprender uma função de distância a partir de **comparações relativas** do tipo "A está mais perto de B do que de C". Esse tipo de informação qualitativa é mais fácil de obter do que rótulos de classe ou valores numéricos de distância e aparece naturalmente em recuperação de informação e em interação com usuários. A família de distâncias considerada é a de Mahalanobis parametrizada, `d_{A,W}(x, y) = sqrt((x − y)ᵀ A W Aᵀ (x − y))`, em que `A` é uma matriz real qualquer (fixa, por exemplo a identidade ou uma transformação de kernel) e **`W` é uma matriz diagonal com entradas não-negativas**, que são os parâmetros a aprender. Com `A = I`, a distância é uma Euclidiana com um peso `Wⱼⱼ ≥ 0` por atributo, exatamente a métrica diagonal usada na dissertação.

Cada comparação relativa `(i, j, k)` do conjunto de treinamento (`xᵢ` está mais perto de `xⱼ` do que de `xₖ`) gera uma restrição de **margem unitária**,

```
(xᵢ − xₖ)ᵀ A W Aᵀ (xᵢ − xₖ) − (xᵢ − xⱼ)ᵀ A W Aᵀ (xᵢ − xⱼ) ≥ 1,
```

isto é, a diferença entre as distâncias quadráticas ao ponto "longe" e ao ponto "perto" deve ser pelo menos 1. Como a diferença é linear nos elementos diagonais de `W`, o conjunto de restrições é linear em `W`. Entre as infinitas soluções viáveis, os autores escolhem a que mantém a métrica o mais próxima possível da Euclidiana, minimizando `½ ‖A W Aᵀ‖²_F`, e, como nos SVMs, adicionam variáveis de folga `ξᵢⱼₖ ≥ 0` penalizadas por `C Σ ξᵢⱼₖ` para acomodar restrições que não podem ser satisfeitas simultaneamente. O resultado (Seção 3) é um **programa quadrático convexo com estrutura de SVM**, resolvido com um método padrão de treinamento de SVM; a soma das folgas é um limitante superior do número de restrições violadas.

## Principais contribuições

- Formulação do aprendizado de uma métrica de Mahalanobis **diagonal e não-negativa** a partir de comparações relativas como um programa quadrático convexo análogo ao SVM.
- Uso de **restrições de margem unitária** sobre diferenças de distâncias quadráticas, lineares nos parâmetros da métrica, com variáveis de folga para o caso não separável.
- Generalização para métricas em espaços transformados (matriz `A` fixa, inclusive via kernel).
- Validação em dados sintéticos e na coleção WebKB, com três métricas distintas aprendidas do mesmo conjunto de documentos a partir de três tipos de comparações, e comparação com a métrica Euclidiana e com métricas de referência.

## Metodologia

Dado um conjunto `P_train` de tríades `(i, j, k)`, resolve-se `min ½ ‖A W Aᵀ‖²_F + C Σ ξᵢⱼₖ` sujeito a uma restrição de margem `≥ 1 − ξᵢⱼₖ` por tríade e `ξ ≥ 0`, com `W` diagonal não-negativa. Nos experimentos, comparam-se a métrica aprendida e a Euclidiana pela taxa de erro em comparações relativas de teste e em tarefas de agrupamento e recuperação sobre documentos.

## Conclusão do artigo

Os autores concluem que comparações relativas são uma forma de supervisão suficiente e prática para aprender uma métrica, e que a formulação como programa quadrático convexo com margem herda a eficiência e as garantias dos SVMs.

## Relação com este trabalho

Este artigo é a **origem da construção das linhas da matriz `A` e do vetor `b` no estimador NNLS** da dissertação (`src/least_squares_inverse.py`). A condição de otimalidade da decisão do LLM sob a regra de centróide mais próximo, "o centróide `c_l` da classe atribuída está mais perto do que o centróide rival `c_k`", é escrita exatamente como a restrição de margem unitária deste artigo, trocando a tríade de pontos pela comparação **ponto versus dois centróides** (redução proposta por Coelho, Borges e Fonseca Neto, 2017):

```
Σⱼ wⱼ [(xᵢⱼ − c_k,j)² − (xᵢⱼ − c_l,j)²] ≥ 1   →   Aᵢⱼ = (xᵢⱼ − c_k,j)² − (xᵢⱼ − c_l,j)²,  bᵢ = 1.
```

Duas diferenças, ambas deliberadas:
- **Igualdade aproximada no lugar da desigualdade.** A dissertação relaxa a margem mínima `≥ 1` em uma margem-alvo `≈ 1` e resolve o sistema no sentido de mínimos quadrados, `min ‖Aw − b‖²` com `w ≥ 0` (NNLS de Lawson e Hanson, 1974), seguindo a estrutura de mínimos quadrados sobre resíduos de otimalidade de Keshavarz, Wang e Boyd (2011). Isso dispensa o hiperparâmetro `C` e as variáveis de folga.
- **Papel do valor 1.** Aqui, como lá, o 1 fixa apenas a escala de `w` (a regra de decisão é invariante a reescalonamentos positivos); é o que exclui a solução trivial `w = 0`.

O mesmo artigo é também a referência da formulação de margem usada no Perceptron Estruturado com Relaxação de Margem (`src/relaxed_perceptron.py`), por meio de Coelho, Borges e Fonseca Neto (2017), que citam Schultz e Joachims como estado da arte a ser simplificado (O(m) restrições ponto-centróide em vez de O(m³) tríades).

## Onde citar

- **Metodologia, §Estimadores da Métrica Inversa, item (ii) NNLS:** ao introduzir a margem unitária que define `A` e `b = 1`.
- **Trabalhos Relacionados, §Aprendizado de Métrica (opcional):** ao lado de Xing et al. (2002) e Weinberger e Saul (2009), como formulação de margem para métricas diagonais.
