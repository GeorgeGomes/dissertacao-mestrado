# Imputing a Convex Objective Function

## Metadados
- **Autores:** Arezou Keshavarz, Yang Wang, Stephen Boyd (Information Systems Laboratory, Electrical Engineering Department, Stanford University)
- **Ano:** 2011
- **Publicação/Venue:** 2011 IEEE International Symposium on Intelligent Control (ISIC), parte da 2011 IEEE Multi-Conference on Systems and Control, Denver, CO, EUA, 28 a 30 de setembro de 2011, pp. 613–619
- **Arquivo:** Imputing a Convex Objective Function.pdf
- **PDF público:** https://web.stanford.edu/~boyd/papers/pdf/imputed_objective_msc.pdf

## Resumo completo

O artigo trata do problema inverso da otimização paramétrica: um agente toma decisões `x(k)` resolvendo um problema de otimização convexa que depende de um parâmetro `p(k)`, mas a função que ele otimiza é desconhecida. Dadas `N` observações `(x(k), p(k))` de decisões ótimas ou aproximadamente ótimas, e um conhecimento prévio sobre a forma dessa função, os autores mostram como **imputar** (estimar) a função objetivo resolvendo um segundo problema de otimização convexa. As aplicações citadas são a estimação de funções de utilidade de consumidores a partir de compras, de funções valor em problemas de controle a partir de um controlador bom porém complexo, e de funções de custo em redes de fluxo.

A função a ser imputada é restrita a uma **parametrização afim finita**, `f = Σ αᵢ fᵢ`, onde `fᵢ` são funções-base convexas escolhidas a priori e `α` pertence a um conjunto convexo `A` que codifica o conhecimento prévio (por exemplo, `A = R₊^{K+1}`, isto é, pesos não-negativos, o que já garante a convexidade de `f`). A decisão observada é "consistente" com `f` se satisfaz as condições de otimalidade de Karush-Kuhn-Tucker (KKT) para o parâmetro correspondente. Como, com dados reais, raramente existe uma `f` exatamente consistente (ruído de medição, erro de modelagem, decisões apenas aproximadamente ótimas), os autores **relaxam as condições KKT em resíduos**: um resíduo de estacionariedade `r_stat(α, λ, ν)` e um resíduo de folga complementar `r_comp(λ)`, ambos lineares em `(α, λ, ν)`.

A formulação central é a **equação (3)** da Seção III:

```
minimizar   Σ_k φ(r_stat^(k), r_comp^(k))
sujeito a   λ^(k) ≥ 0,  k = 1, ..., N,   α ∈ A
```

com variáveis `α` (pesos das funções-base) e as variáveis duais `λ^(k)`, `ν^(k)` de cada observação. A penalidade `φ` é uma função convexa não-negativa que se anula apenas quando os resíduos são zero; pode ser qualquer norma, ou Huber, deadzone-linear, etc. Como os resíduos são lineares nos parâmetros, o problema é convexo e de dimensão finita. **Em todos os exemplos numéricos do artigo (comportamento do consumidor, §IV-A; controle por horizonte recuado, §VI; redes de fluxo, §VII) a penalidade adotada é a soma de quadrados**, `φ(r_stat, r_comp) = ‖r_stat‖₂² + ‖r_comp‖₂²`, o que torna a estimação do critério um problema de **mínimos quadrados com restrições** (não-negatividade de `λ` e restrição `α ∈ A`, tipicamente `α ≥ 0`).

Um ponto tratado com cuidado é o das **soluções triviais**: como os resíduos são homogêneos em `(α, λ, ν)`, `α = 0` sempre zera o objetivo. Para excluí-la é preciso normalizar, por exemplo fixando um dos coeficientes (`α₀ = 1`, quando parte da função é conhecida) ou impondo restrições em `A` que descartem funções constantes. Os autores observam ainda que, se o valor ótimo de (3) for zero e as observações forem primal-viáveis, a função imputada é exatamente consistente com os dados; se os resíduos ficarem grandes, o modelo de "agente otimizador" não descreve bem os dados.

## Principais contribuições

- Formulação geral e convexa do problema de **imputar a função objetivo** de um processo otimizador a partir de decisões observadas, para problemas convexos paramétricos com restrições.
- Tratamento explícito de decisões **aproximadamente ótimas**: relaxação das condições KKT em resíduos e minimização de uma penalidade convexa desses resíduos.
- Escolha de **mínimos quadrados** (`φ = ‖·‖₂²`) como penalidade nos exemplos, resultando em um problema de mínimos quadrados com restrições lineares e de não-negatividade.
- Discussão da **normalização** necessária para evitar a solução trivial `α = 0` e da não-unicidade da função imputada.
- Três aplicações numéricas: utilidade do consumidor (imputação de uma utilidade quadrática côncava, com boa previsão da demanda fora da amostra), redução de complexidade de controladores (função valor quadrática imputada a partir de um controlador por horizonte recuado, com desempenho quase idêntico ao original) e custos em redes de fluxo.

## Metodologia

Para cada observação `(x(k), p(k))`, introduzem-se as variáveis duais `λ(k)`, `ν(k)` e escrevem-se os resíduos de estacionariedade `∇f(x, p) + Σ λᵢ ∇gᵢ(x, p) + A(p)ᵀ ν` e de folga complementar `λᵢ gᵢ(x, p)`. Como `f = Σ αᵢ fᵢ`, o gradiente é linear em `α`, e os resíduos são lineares em todas as variáveis. Resolve-se então o problema (3) com `φ` quadrática, o que é um programa quadrático convexo (nos exemplos, resolvido com CVX). Nos exemplos, os dados são gerados por um agente que otimiza uma função conhecida, e a qualidade da imputação é medida pela capacidade preditiva da função imputada em novos parâmetros, não pela recuperação exata dos coeficientes (que, pela invariância a fatores multiplicativos e a composições monótonas, não é identificável).

## Conclusão do artigo

Os autores concluem que, quando se conhece a estrutura do problema de decisão (restrições e uma família de funções-base para o objetivo), é possível imputar de forma eficiente uma função objetivo convexa que explica decisões observadas, mesmo ruidosas, resolvendo um único problema convexo, tipicamente de mínimos quadrados. A função imputada serve para prever decisões futuras e para substituir um decisor complexo por um mais simples.

## Relação com este trabalho

Este artigo é a **âncora bibliográfica da formulação de mínimos quadrados em otimização inversa** usada pelo segundo estimador da dissertação (`train_least_squares_inverse()` em `src/least_squares_inverse.py`, resolvido por `scipy.optimize.nnls`). A correspondência é termo a termo:

| Keshavarz, Wang e Boyd (2011) | Dissertação (NNLS) |
|---|---|
| Critério parametrizado `f = Σ αᵢ fᵢ` com funções-base fixas | Distância `d_W(x, c)² = Σ wⱼ (xⱼ − cⱼ)²`: funções-base `(xⱼ − cⱼ)²`, pesos `wⱼ` |
| Restrição `α ∈ A`, tipicamente `α ≥ 0` (mantém `f` convexa) | `w ≥ 0` (mantém a métrica de Mahalanobis diagonal semidefinida positiva) |
| Decisão observada `x(k)` tomada pelo agente otimizador | Rótulo `y(i)` atribuído pelo LLM ao ponto `x(i)` (regra de centróide mais próximo) |
| Condição de otimalidade (KKT) da decisão, relaxada em resíduo linear nos parâmetros | Condição "centróide da classe atribuída mais perto que o rival", relaxada no resíduo `Aᵢ w − 1`, com `Aᵢⱼ = (xᵢⱼ − c_k,j)² − (xᵢⱼ − c_l,j)²` |
| Penalidade `φ = ‖·‖₂²` (soma de quadrados dos resíduos) | `min ‖Aw − b‖₂²` |
| Normalização para excluir `α = 0` (ex.: `α₀ = 1`) | Margem-alvo `b = 1` (fixa a escala de `w`, que é indeterminada pela invariância da regra de decisão, e exclui `w = 0`) |
| Programa quadrático convexo (CVX) | NNLS de Lawson e Hanson (1974), `scipy.optimize.nnls` |

A diferença principal é que a dissertação usa, no lugar do resíduo KKT completo (estacionariedade e folga complementar, com variáveis duais), o **resíduo de margem da decisão discreta** (à maneira de Schultz e Joachims, 2003), o que é natural para um classificador por centróide mais próximo: a "condição de otimalidade" da decisão é uma única desigualdade entre duas distâncias, sem variáveis duais. A estrutura, porém, é a mesma: pesos não-negativos de um critério linear nos parâmetros, estimados por mínimos quadrados sobre resíduos de otimalidade das decisões observadas.

## Onde citar

- **Fundamentação Teórica, §Otimização Inversa:** ao apresentar a passagem da formulação clássica (Ahuja e Orlin, 2001, perturbação mínima) para a formulação por resíduos e mínimos quadrados quando as decisões são apenas aproximadamente ótimas.
- **Metodologia, §Estimadores da Métrica Inversa, item (ii) NNLS:** citação central, indicando que o estimador segue esta formulação; explicar `A`, `b` e o papel da margem-alvo como normalização.
- **Trabalhos Relacionados, §Otimização Inversa:** ao lado de Ahuja e Orlin (2001) e Chan, Mahmood e Zhu (2025).
