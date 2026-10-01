# E-mail para o orientador: referência dos Mínimos Quadrados em Otimização Inversa

**Assunto:** Referência dos Mínimos Quadrados em Otimização Inversa (artigo em anexo) e correções no texto

---

Professor,

Sobre a sua pergunta de onde tirei a referência para os Mínimos Quadrados em Otimização Inversa: o senhor tinha razão, ela não estava na bibliografia do artigo. Ao revisar, percebi que o único trabalho que eu citava no parágrafo do NNLS era o livro de Lawson e Hanson (1974), que cobre apenas o algoritmo numérico de mínimos quadrados não-negativos (o que o scipy implementa), e não o uso de mínimos quadrados como método de otimização inversa. O método em si eu tinha adotado como segundo estimador, para ter um contraponto sem hiperparâmetros ao Perceptron, e faltava ancorá-lo na literatura.

Fui atrás e a referência que traz exatamente essa formulação é a seguinte, que envio em anexo:

**Keshavarz, A.; Wang, Y.; Boyd, S. "Imputing a Convex Objective Function". 2011 IEEE International Symposium on Intelligent Control (ISIC), Denver, pp. 613–619.**
Cópia pública: https://web.stanford.edu/~boyd/papers/pdf/imputed_objective_msc.pdf

## Onde está a formulação no artigo

- **Seção II (Problem statement):** o agente resolve um problema de otimização convexa paramétrico cuja função objetivo é desconhecida. Ela é restrita a uma combinação de funções-base com pesos, f = Σ αᵢ fᵢ, com α em um conjunto convexo A que codifica o conhecimento prévio (tipicamente α ≥ 0, o que garante a convexidade de f).
- **Seção III (Our method), equação (3):** como as decisões observadas são só aproximadamente ótimas, as condições de otimalidade (KKT) de cada decisão são relaxadas em resíduos, lineares nos pesos α e nas variáveis duais, e os pesos são estimados resolvendo

  minimizar  Σₖ φ(r_stat⁽ᵏ⁾, r_comp⁽ᵏ⁾)   sujeito a  λ⁽ᵏ⁾ ≥ 0,  α ∈ A.

- **Escolha de mínimos quadrados:** em todos os exemplos do artigo (consumidor, §IV-A; controle, §VI; rede de fluxo, §VII) os autores tomam φ(r_stat, r_comp) = ‖r_stat‖₂² + ‖r_comp‖₂², ou seja, a soma dos quadrados dos resíduos. O problema vira um problema de mínimos quadrados com restrições (não-negatividade e α ∈ A).
- **Normalização (Seção III, "Trivial solutions and normalization"):** como os resíduos são homogêneos nos parâmetros, α = 0 sempre zera o objetivo; para excluir essa solução trivial eles fixam a escala (por exemplo α₀ = 1) ou restringem A.

## Como isso corresponde ao que fiz na dissertação

| No artigo de Keshavarz, Wang e Boyd | No meu estimador NNLS |
|---|---|
| Critério parametrizado f = Σ αᵢ fᵢ | Distância d_W(x, c)² = Σⱼ wⱼ (xⱼ − cⱼ)²: as funções-base são (xⱼ − cⱼ)² e os pesos são wⱼ |
| Restrição α ∈ A, tipicamente α ≥ 0 | w ≥ 0 (matriz diagonal semidefinida positiva) |
| Decisão observada x⁽ᵏ⁾ do agente | Rótulo y⁽ⁱ⁾ que o LLM atribui ao ponto x⁽ⁱ⁾, sob a regra de centróide mais próximo |
| Condição de otimalidade relaxada em resíduo linear nos parâmetros | Condição "centróide da classe atribuída mais perto que o centróide rival", escrita como Σⱼ wⱼ[(xⱼ − c_k,j)² − (xⱼ − c_l,j)²] = 1; o resíduo é Aᵢw − 1 |
| Penalidade φ = soma de quadrados | min ‖Aw − b‖₂² |
| Normalização para excluir α = 0 | Margem-alvo b = 1, que fixa a escala de w (a regra de decisão é invariante a escala) e exclui w = 0 |
| Programa quadrático convexo | NNLS de Lawson e Hanson (1974), scipy.optimize.nnls |

A única diferença é que, em vez do resíduo KKT completo (estacionariedade e folga complementar, com variáveis duais), uso o resíduo de margem da decisão discreta, como em Schultz e Joachims (2003): para um classificador por centróide mais próximo, a condição de otimalidade da decisão é uma única desigualdade entre duas distâncias, então não há variáveis duais. A estrutura é a mesma: pesos não-negativos de um critério linear nos parâmetros, estimados por mínimos quadrados sobre os resíduos de otimalidade das decisões observadas.

## Correções que fiz no artigo a partir dessa comparação

1. **Corrigi uma frase errada na descrição do NNLS.** O texto dizia que "b codifica a decisão do LLM". Não é isso que o código faz: b é um vetor de uns (a margem-alvo), e a decisão do LLM entra na matriz A, ao definir qual centróide é o correto e, com isso, o sinal de cada linha. Reescrevi o parágrafo para descrever exatamente a implementação.
2. **Adicionei a equação explícita da construção de A e b** (nova equação, antes da equação do NNLS): para cada ponto, Σⱼ wⱼ[(x⁽ⁱ⁾ⱼ − c_k,j)² − (x⁽ⁱ⁾ⱼ − c_l,j)²] = 1, com c_l o centróide da classe atribuída pelo LLM e c_k o centróide rival mais próximo; portanto Aᵢⱼ = (x⁽ⁱ⁾ⱼ − c_k,j)² − (x⁽ⁱ⁾ⱼ − c_l,j)² e bᵢ = 1. Expliquei também o papel de b = 1 como normalização de escala e exclusão da solução trivial.
3. **Ancorei o estimador na literatura:** o parágrafo do NNLS agora abre com a formulação de Keshavarz, Wang e Boyd (2011) e cita Schultz e Joachims (2003) para a margem unitária; Lawson e Hanson (1974) fica só como o algoritmo que resolve o sistema.
4. **Fundamentação Teórica (§Otimização Inversa):** acrescentei a formulação clássica de Ahuja e Orlin (2001), que o senhor tinha mencionado na reunião de março (perturbação mínima dos parâmetros, normas L1 e L∞), e a passagem para a formulação por resíduos e mínimos quadrados quando as decisões são só aproximadamente ótimas, que é o caso das decisões do LLM.
5. **Trabalhos Relacionados (§Otimização Inversa):** incluí Ahuja e Orlin (2001) e Keshavarz, Wang e Boyd (2011) ao lado do survey de Chan, Mahmood e Zhu (2025), que já estava citado.
6. **Bibliografia:** três referências novas (Keshavarz, Wang e Boyd, 2011; Ahuja e Orlin, 2001; Schultz e Joachims, 2003). Os PDFs de Keshavarz e de Schultz e Joachims foram adicionados à pasta de artigos referenciados.

## Anexos

1. `Imputing a Convex Objective Function.pdf` (Keshavarz, Wang e Boyd, 2011): a formulação está na Seção III, equação (3), e a escolha de mínimos quadrados aparece nos exemplos das Seções IV-A, VI e VII.
2. `artigo.pdf` atualizado, com as correções acima (Seção 3.3, item (ii), e Seção de Fundamentação, §Otimização Inversa).

Se o senhor achar que a formulação por resíduo de margem precisa ser aproximada ainda mais da formulação KKT do artigo, ou que vale citar outra referência de mínimos quadrados em otimização inversa, ajusto o texto.

Obrigado,
George
