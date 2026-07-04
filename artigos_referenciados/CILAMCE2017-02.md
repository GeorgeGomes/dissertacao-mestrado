# Uso de Predição Estruturada para o Aprendizado de Métrica

## Metadados
- **Autores:** Maurício Archanjo Nunes Coelho (Instituto Federal do Sudeste de Minas Gerais — Campus Rio Pomba); Carlos Cristiano Hasenclever Borges (Universidade Federal de Juiz de Fora); Raul Fonseca Neto (Universidade Federal de Juiz de Fora)
- **Ano:** 2017
- **Publicação/Venue:** CILAMCE 2017 — Proceedings of the XXXVIII Iberian Latin-American Congress on Computational Methods in Engineering (ABMEC), Florianópolis, SC, Brasil, 5–8 de novembro de 2017
- **Arquivo:** CILAMCE2017-02.pdf

## Resumo completo

O artigo trata do problema de **Aprendizado de Métrica** (Metric Learning) sob a ótica de um **problema inverso**: dado um conjunto de informações fornecidas por um especialista — indicando quais instâncias devem pertencer ao mesmo agrupamento — o objetivo é determinar uma métrica de distância tal que o conjunto de distâncias parametrizadas satisfaça as restrições impostas pelo especialista. A motivação é que algoritmos baseados em protótipos (como K-means e classificação pelo centroide/vizinho mais próximo) dependem fortemente da métrica usada para medir similaridade; uma métrica Euclidiana ingênua pode discordar do agrupamento desejado pelo especialista. Aprender a métrica equivale a um reescalonamento das coordenadas dos dados de modo que a solução do K-means concorde com a do especialista, sem alterar a dimensão dos dados (diferentemente do truque kernel).

O problema é formulado como uma **otimização convexa** que minimiza um conjunto parametrizado de **distâncias de Mahalanobis**, sujeita a restrições de não-negatividade e desigualdade triangular. O escopo restringe-se à **matriz diagonal** (equivalente a um vetor de parâmetros `w` que pondera cada dimensão, com complexidade O(d) em vez de O(d²)) e ao aprendizado **offline**. A contribuição central é reformular o aprendizado da métrica como um problema de **Predição Estruturada**, resolvido por uma variante do **algoritmo Perceptron Estruturado com margem** (Coelho et al., 2012/2016), com uma modificação que permite a **violação controlada de restrições** (relaxação/flexibilização da margem), viabilizando problemas não linearmente separáveis.

A formulação proposta baseia-se na comparação de cada ponto com **centroides** (em vez de pares de pontos), apoiada em duas premissas: (1) o problema de pairwise constraints (must-link / cannot-link) pode ser reduzido a um problema supervisionado com grupos disjuntos via fechos transitivos; e (2) o somatório das distâncias entre todos os pares de um mesmo grupo é igual ao somatório das distâncias de cada ponto ao seu centroide multiplicado pelo número de pontos. Como há infinitas soluções que satisfazem as restrições, escolhe-se aquela que **maximiza a margem** — a diferença entre as distâncias intergrupos.

A principal vantagem alegada sobre o estado da arte (notadamente Schultz e Joachims, 2003, e Xing et al., 2002) é que a formulação por comparação ponto-vs-centroide gera apenas uma quantidade **linear** de restrições — O(m) — em vez de O(m³) das comparações relativas entre tríades de pontos, e evita o uso de programação quadrática ou semidefinida, recorrendo a uma técnica de relaxação eficiente. Os experimentos, em bases artificiais (R² e R³) e em seis bases reais do UCI, mostraram melhora consistente do poder de classificação em relação à métrica Euclidiana.

## Principais contribuições

- Reformulação do **Aprendizado de Métrica** (matriz diagonal de Mahalanobis) como um problema de **Predição Estruturada**, resolvido pelo **Perceptron Estruturado com margem**.
- Modelagem por **comparação ponto-vs-centroide** que reduz o número de restrições de O(m³) (abordagem de comparações relativas de Schultz e Joachims) para **O(m)** — quantidade linear de restrições.
- Técnica de **relaxação com flexibilização de margem** (variáveis duais / multiplicadores de Lagrange com penalização λ=1/C) que permite a violação de algumas restrições, tratando dados **não linearmente separáveis** sem recorrer a programação quadrática/semidefinida.
- Escolha da solução de **máxima margem** (diferença das distâncias intergrupos) entre as infinitas soluções viáveis, com busca da margem máxima via incremento progressivo e **pesquisa binária**.
- Demonstração de que o aprendizado de métrica produz, como hipótese, um **classificador linear de máxima margem** (NCC parametrizado) quando se usa uma única matriz de métricas.
- Validação empírica em bases artificiais (R²/R³, 100% de acerto vs. especialista) e seis bases UCI, com ganhos de acurácia em treino (K-means parametrizado com margem) e em teste/generalização (NCC com a métrica aprendida).

## Metodologia

O trabalho parte da **distância de Mahalanobis parametrizada** `d_A(x,y) = (x−y)ᵀA(x−y)`, restrita ao caso da **matriz diagonal** A (vetor `w` com componentes não-negativas), o que equivale a reescalonar cada dimensão pela raiz quadrada do respectivo parâmetro e computar uma distância Euclidiana normalizada. Demonstra-se a equivalência entre a soma das distâncias intragrupo e a soma das distâncias de cada ponto ao seu centroide (resultado de Edwards e Cavalli-Sforza, 1965), permitindo trocar a função de pairwise distances pela função de distância parametrizada do **K-means**.

Em vez da estratégia de redução de diferenças entre K-means e especialista (Fagundes et al., 2016, baseada em gradiente descendente), os autores fixam o esquema de rótulos do especialista e impõem, para cada ponto `xᵢ` com centroide correto `cₗ`, a restrição de que sua distância parametrizada a `cₗ` seja menor do que a distância ao melhor centroide concorrente `cₖ`. Isso gera um **sistema de inequações** resolvido por **relaxação**. O problema de máxima margem (Eq. 27–28) é convertido na regra de correção do **Perceptron Estruturado com margem** (Eq. 29): quando uma restrição é violada, atualiza-se `w` (e as variáveis duais α) por um passo `η`, mantendo `w ≥ 0` para preservar a positividade da métrica. A margem γ inicia em zero; após cada execução do perceptron, calcula-se a margem de parada (a menor entre as classes), incrementa-se γ e repete-se até a inviabilidade, refinando a margem máxima por **busca binária**.

Nos experimentos, a taxa de aprendizado foi η fixa e a constante de penalização C variou de 1 a 0,1 conforme a dificuldade do problema. Os dados **não foram normalizados** (o que, segundo os autores, prejudica um pouco a acurácia). Todas as bases são não linearmente separáveis no espaço de entrada e linearmente separáveis no espaço de características com kernel polinomial de grau 2 ou 3. Compararam-se três variantes: **K-means1** (Euclidiano), **K-means2** (parametrizado) e **K-means3** (parametrizado com margem) para o treino; e **NCC1/NCC2/NCC3** para a generalização (50% treino / 50% teste, média de 20 execuções). Bases UCI usadas: Ionosphere, Wine, Balance, Breast Cancer, Iris e Parkinsons.

## Conclusão do artigo

Os autores concluem que a abordagem de predição estruturada resolve o Aprendizado de Métrica de forma eficiente, com **menor complexidade** (quantidade linear de restrições) e maior simplicidade do que o estado da arte, evitando programação quadrática e semidefinida. O método mostrou-se **estável** (convergiu em todos os experimentos) e os resultados foram promissores, melhorando a classificação em relação à métrica Euclidiana tanto em treino quanto em teste — e ainda gera um **classificador linear de máxima margem**. Como trabalhos futuros, propõem: comparar com SVM e outros classificadores de máxima margem; usar uma métrica apropriada por classe; estender para dados não linearmente separáveis via kernel (clustering no espaço de características); e desenvolver uma versão **online** da mesma metodologia.

## Relação com este trabalho

Este artigo é, muito provavelmente, **a base algorítmica direta do Perceptron Estruturado com Relaxação de Margem** usado na dissertação. O algoritmo central da dissertação — `train_relaxed_perceptron()` em `relaxed_perceptron.py`, que aprende uma **métrica de Mahalanobis diagonal `w`** com relaxação de margem e busca binária em γ — corresponde exatamente à formulação das Eq. 25–31 deste paper (Perceptron Estruturado com margem, regra de correção mantendo `w ≥ 0`, incremento de γ até a inviabilidade e refinamento por pesquisa binária). Os autores Coelho, Borges e Fonseca Neto são o grupo do orientador.

Pontos de conexão específicos:
- **Métrica diagonal de Mahalanobis (vetor `w` ≥ 0):** a dissertação adota exatamente a mesma simplificação O(d) descrita aqui (matriz diagonal = reescalonamento por dimensão).
- **Otimização inversa:** o artigo formula explicitamente o problema como o "inverso" do K-means (qual métrica reproduz os rótulos do especialista?). A dissertação faz o análogo trocando o "especialista" pelo **LLM** — aprende `Ŵ_LLM` a partir das classificações observadas do LLM (Bloco 1, LLM como FONTE).
- **Classificação por centroide/vizinho mais próximo (NCC):** a regra de decisão "ponto mais próximo do seu centroide do que de qualquer outro" do paper é a mesma base da comparação por centroides na dissertação.
- **Margem e relaxação:** o conceito de margem γ flexível e violação controlada de restrições (para dados não separáveis) é o mesmo mecanismo do `relaxed_perceptron.py` (hiperparâmetros eta, C, delta_gamma).
- **Bloco 3 (peso × altura) e fronteiras não-lineares:** a observação do artigo de que duas matrizes de parâmetros distintas geram uma função de decisão **quadrática** (Eq. 19–20, fronteira elíptica de Fisher) dialoga diretamente com o estudo de caso peso×altura (elipse) e as augmentações R3/R4 da dissertação.

A **diferença de papel** vale registrar: no artigo o especialista é humano/ground truth e a métrica reproduz seus agrupamentos; na dissertação o "especialista observado" é o LLM (Bloco 1) ou, invertendo, o LLM é o aprendiz de um perito sintético (Bloco 2).

## Onde citar

- **Metodologia / Algoritmos de otimização inversa:** ao descrever o `train_relaxed_perceptron()` e a formulação do Perceptron Estruturado com Relaxação de Margem — citação central, indicando que a implementação segue Coelho et al. (2017).
- **Fundamentação / Trabalhos relacionados:** ao introduzir Aprendizado de Métrica, distância de Mahalanobis diagonal e a redução de pairwise constraints a um problema supervisionado por centroides; também como ponte para Xing et al. (2002) e Schultz e Joachims (2003).
- **Justificativa da métrica diagonal:** ao explicar a escolha O(d) em vez de O(d²) (Notas de Design da dissertação).
- **Conexão NCC / classificador linear de máxima margem:** ao discutir a fronteira de decisão induzida pela métrica aprendida.
- **Bloco 3 e R3/R4:** ao motivar a fronteira quadrática/elíptica (peso×altura) a partir do uso de mais de uma matriz de parâmetros.
