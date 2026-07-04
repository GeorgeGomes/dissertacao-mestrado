# Toy Models of Superposition

## Metadados
- **Autores:** Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, Christopher Olah (Anthropic; Harvard)
- **Ano:** 2022 (publicado em 14 de setembro de 2022)
- **Publicação/Venue:** Transformer Circuits Thread (artigo online da Anthropic, formato HTML)
- **Arquivo:** Toy Models of Superposition.pdf

## Resumo completo

O artigo investiga a **polissemanticidade** — o fenômeno empírico em que um único neurônio de uma rede neural responde a múltiplos conceitos não relacionados, dificultando a interpretabilidade. Os autores propõem que essa polissemanticidade surge de uma estratégia chamada **superposição (superposition)**: a rede armazena mais *features* (características) do que tem dimensões/neurônios, representando-as como direções **quase ortogonais** no espaço de ativações. Isso só é vantajoso quando as features são **esparsas** (raramente ativas ao mesmo tempo), pois a esparsidade reduz a "interferência" entre features que compartilham dimensões, e a não-linearidade (ReLU) permite filtrar pequenos ruídos.

Para estudar o fenômeno de forma controlada, os autores constroem **modelos de brinquedo (toy models)**: pequenas redes ReLU treinadas com dados sintéticos esparsos. A tarefa é projetar um vetor de alta dimensão `x ∈ Rⁿ` para um espaço menor `h ∈ Rᵐ` (com `m < n`) via `h = Wx` e depois recuperá-lo. Comparam-se dois modelos: um **modelo linear** (`x' = WᵀWx + b`, que não exibe superposição e recupera apenas as `m` features mais importantes, como o PCA) e um **modelo com ReLU na saída** (`x' = ReLU(WᵀWx + b)`, que exibe superposição). A diferença reside apenas na função de ativação final, mas o comportamento muda radicalmente.

Os principais achados são: (1) a superposição é um fenômeno **real e observável**; com features densas o modelo ReLU se comporta como o linear, mas conforme a esparsidade aumenta ele passa a representar features adicionais de forma não-ortogonal; (2) podem se formar tanto neurônios **monossemânticos** quanto **polissemânticos**; (3) é possível realizar **computação em superposição** (ex.: circuitos que computam a função valor absoluto sobrepostos); (4) o armazenamento de uma feature em superposição é governado por uma **mudança de fase (phase change)**, descrita por um "diagrama de fase" em função de importância e esparsidade.

Um resultado surpreendente é a **estrutura geométrica** da superposição: as features se organizam em polítopos uniformes — pares antipodais (digons), triângulos, tetraedros, pentágonos, antiprismas quadrados — análogos a soluções do problema de Thomson. Configurações mais complexas são entendidas como **produtos tegum** (polítopos embutidos em subespaços ortogonais, sem interferência cruzada). Em regimes não-uniformes (features com importâncias/esparsidades distintas ou correlacionadas), essas geometrias se deformam suavemente até "estalarem" para outro polítopo; features correlacionadas tendem a ficar ortogonais (em fatores tegum distintos) e anti-correlacionadas tendem a compartilhar o mesmo fator.

Por fim, os autores discutem implicações para a **interpretabilidade mecanicista** e a segurança de IA: a superposição é um obstáculo central porque impede decompor o espaço de ativações em features limpas (não se pode simplesmente aplicar Gram-Schmidt). Apontam ainda conexões preliminares com **exemplos adversariais**, **grokking** e modelos **mixture of experts**, e levantam a hipótese de que redes reais "simulam ruidosamente" redes muito maiores e altamente esparsas sem interferência.

## Principais contribuições

- Demonstração **direta e inequívoca** de que a superposição ocorre em redes neurais artificiais num cenário relativamente natural (pequenas redes ReLU com dados esparsos), sugerindo que também ocorra na prática.
- Uma **teoria de quando e por que** a superposição surge, formalizada como uma **mudança de fase** e um **diagrama de fase** em função da esparsidade e importância das features.
- Descoberta de que a superposição organiza features em **estruturas geométricas** (polítopos uniformes: digons/pares antipodais, triângulos, tetraedros, pentágonos, antiprismas quadrados), ligadas ao problema de Thomson e construídas via **produtos tegum**.
- Distinção formal de uma **hierarquia de propriedades** das representações: decomponibilidade, linearidade, superposição (`WᵀW` não invertível) e alinhamento à base (basis-aligned).
- Conceitos de **base privilegiada (privileged basis)** vs. **superposição** como duas forças opostas que explicam quando neurônios são interpretáveis.
- Evidência de que ao menos **alguns tipos de computação** podem ser feitos em superposição (circuito de valor absoluto sobreposto).
- Conexões preliminares com **exemplos adversariais**, **grokking** e **mixture of experts**, e implicações para interpretabilidade e segurança.

## Metodologia

Os autores usam **toy models** (modelos de brinquedo) — redes ReLU pequenas treinadas em dados sintéticos — para isolar o fenômeno. Os dados `x` simulam três propriedades que as features reais teriam: **esparsidade** (cada feature `xᵢ = 0` com probabilidade `Sᵢ`, senão uniforme em `[0,1]`), **mais features do que neurônios** e **variação de importância** (`Iᵢ`). A tarefa é compressão e reconstrução: `h = Wx` (projeção para dimensão menor) seguida de recuperação. Dois modelos diferem apenas na ativação final: o **modelo linear** (`x' = WᵀWx + b`) como baseline sem superposição, e o **modelo ReLU output** (`x' = ReLU(WᵀWx + b)`) que exibe superposição. A perda é o **erro quadrático médio ponderado pela importância** das features: `L = Σ Σ Iᵢ (xᵢ − x'ᵢ)²`.

A análise visual usa a matriz `WᵀW` (features × features) e o viés `b`; mede-se se uma feature é representada pela norma `‖Wᵢ‖` e quanto ela compartilha sua dimensão com outras via `Σⱼ≠ᵢ (Ŵᵢ · Wⱼ)²`. Variando os níveis de esparsidade (ex.: `n=20, m=5, Iᵢ=0.7ⁱ`), observa-se a transição: com features densas o ReLU imita o linear (PCA); com esparsidade crescente, emerge a superposição, começando por pares antipodais e evoluindo para outras geometrias. O estudo separa **superposição uniforme** (features idênticas) da **não-uniforme** (importâncias/esparsidades distintas ou correlações).

## Conclusão do artigo

Os autores concluem que a superposição é um fenômeno real, governado por uma mudança de fase e dotado de rica estrutura geométrica, mesmo em modelos extremamente simples. Argumentam que "resolver a superposição" — obter modelos sem superposição ou métodos para decodificá-la — é uma peça central da agenda de **interpretabilidade mecanicista** voltada à segurança de IA, já que a superposição derrota a decomposição do espaço de ativações (a "maldição da dimensionalidade"). Hipotetizam que redes reais agem como simulações ruidosas de redes muito maiores e esparsas. Reconhecem que generalizar dos toy models para redes reais é incerto, mas que alguns fenômenos (como a relação com exemplos adversariais) provavelmente generalizam. Encerram com **questões em aberto**, incluindo o papel dos neurônios polissemânticos e estratégias (arquiteturas, treino adversarial, mudança de ativação) para induzir modelos mais interpretáveis.

## Relação com este trabalho

A dissertação "Explicando Decisões de LLMs via Otimização Inversa" tenta extrair uma **métrica de Mahalanobis diagonal `W`** das classificações de um LLM, assumindo que o critério decisório do modelo pode ser capturado por uma representação geométrica **simples e linear** no espaço de features. *Toy Models of Superposition* é diretamente relevante a essa premissa: ele mostra que redes neurais codificam features como **direções no espaço de ativações** (hipótese da representação linear) — o que dá suporte conceitual à ideia de descrever o critério do LLM por direções/pesos — mas, ao mesmo tempo, alerta que essas direções podem estar em **superposição**, isto é, **mais features do que dimensões**, organizadas em geometrias não-triviais (polítopos, pares antipodais). Isso fundamenta a discussão sobre os **limites de extrair uma métrica simples**: se o LLM codifica seu critério em direções quase-ortogonais sobrepostas, uma `W` diagonal de baixa dimensão pode capturar apenas parte do critério (a parte "alinhada à base"/dominante), deixando interferência e não-linearidade de fora — exatamente o tipo de tensão que aparece nos blocos da dissertação (linear vs. meia-lua/elipse, augmentação R3/R4, e o "paradoxo do overfitting"). O artigo também sustenta a moldura de **interpretabilidade mecanicista** e **explicabilidade** do trabalho, e a distinção entre decomponibilidade, linearidade e alinhamento à base ajuda a posicionar o que a otimização inversa consegue e não consegue recuperar.

## Onde citar

- **Introdução / Motivação:** ao justificar a hipótese da representação linear (features como direções) que embasa a tentativa de aprender uma métrica `W` a partir das decisões do LLM.
- **Trabalhos Relacionados / Fundamentação:** como referência central de interpretabilidade mecanicista, polissemanticidade e superposição.
- **Discussão / Limitações:** ao discutir por que uma métrica diagonal/linear simples pode não capturar todo o critério do LLM (features em superposição, não-linearidade, interferência), motivando as augmentações R3/R4 e os problemas não-lineares (meia-lua, elipse).
- **Conclusão / Trabalhos Futuros:** ao apontar que critérios de decisão de LLMs podem residir em estruturas geométricas mais ricas do que uma única direção, sugerindo extensões além da métrica diagonal.
