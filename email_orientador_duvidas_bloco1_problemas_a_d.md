# E-mail para o orientador: dúvidas do Bloco 1, Problemas A a D

**Assunto:** Re: dúvidas Bloco 1, problemas A-D

---

Professor, bom dia.

Seguem as respostas, na ordem do seu e-mail (execução completa de 05/07, média de 3 sementes).

**Problema A**

- **Fidelidade:** o LLM rotula os 150 pontos em *zero-shot*. Com esses rótulos calculo os centróides e estimo W. A fidelidade é a fração dos pontos em que a regra do centróide mais próximo, com d_W, coincide com o rótulo do LLM.
- **Split:** não existe no A, a fidelidade é medida nos mesmos 150 pontos. Calculei agora com validação cruzada e a diferença é pequena (gpt-4o-mini, Perceptron: 84,0% na amostra e 81,0% fora dela). Se quiser posso adotar split 70/30 mas ainda não é feito.
- **Ground truth:** estimando com os rótulos verdadeiros, a fidelidade fica entre 93% e 98%. Não chega a 100% porque as duas gaussianas se sobrepõem (erro de Bayes de 4,8%). Com rótulos separáveis chega a 99% ou 100%. Isso confirma a sua leitura: a fidelidade de 81% a 86% vem do LLM, que em *zero-shot* acerta só 25% a 31% do ground truth.
- ***Few-shot*:** ainda não foi feito. Uma forma seria usar k exemplos rotulados pelo ground truth, k em {4, 10, 20, 40}, retirados do conjunto de estimação. É para implementar dessa forma?

**Problemas B e C**

- As amostras são novas: 100 pontos em cada problema, gerados de forma independente, com centróides em outras posições. Não são os pontos do A deslocados.
- Somente W é transferido. Os centróides são recalculados com os rótulos do LLM no próprio B ou C.
- Hoje a consistência é calculada só com a métrica *zero-shot*. É para calcular também com a métrica *few-shot* e reportar as duas?

**Problema D**

- Sim, é estimada uma nova métrica, a partir dos rótulos do LLM na meia-lua, em R², R³ e R⁴. A fidelidade fica em torno de 82% no gemini. O gpt-4o-mini coloca todos os pontos em uma única classe em 2 das 3 sementes.
- A estimação em relação ao ground truth já existe e dá cerca de 86%. A estimação em *few-shot* ainda não existe. É para repetir no D o mesmo protocolo do A?

Fico no aguardo da sua orientação sobre esses pontos.

Abraços,
George
