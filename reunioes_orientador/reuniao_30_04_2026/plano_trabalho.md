# Plano de Trabalho — Reunião 30/04/2026

## Contexto da reunião

Reunião com o orientador Prof. Raul Fonseca Neto e o coorientador (Yuri/Saulo) em 30/04/2026, com duração aproximada de 54 minutos. Apresentei os resultados consolidados da execução `execucao_2026-04-14_22-10-02`, que contempla:

- Fases A, B, C com Problemas sintéticos 2D gaussianos anisotrópicos
- Três algoritmos de otimização inversa: Perceptron Estruturado, NNLS e LP Max-Margin
- Fase D com perito externo (W conhecido) e quatro estratégias de seleção de exemplos (easy, hard, mixed, random)
- Experimentos auxiliares: Oracle Validation, diluição, viés de ordem das classes e dos exemplos, variantes de prompt, projeção R3, baselines clássicos
- Análises estatísticas (bootstrap CI, Wilcoxon, Cohen's d, Mann-Whitney, Kruskal-Wallis)

Após a reunião, no mesmo dia, o orientador enviou **três e-mails** consolidando e refinando as instruções:

- **19:15** — envio da base de dados real (peso × altura) com 100 amostras e a forma exata do classificador ótimo;
- **22:04** — esclarecimento dos 5 passos da Parte 2;
- **22:06** — adendo pedindo visualização gráfica das rotulações.

Posteriormente, em **06/05/2026 às 10:15**, o orientador marcou nova reunião para **07/05/2026 (quinta) às 17h**.

Este plano segue prioritariamente o conteúdo dos e-mails — versão definitiva — usando a transcrição apenas para contexto e justificativa.

## Fontes

- E-mails do orientador (19:15, 22:04 e 22:06 de 30/04/2026; 10:15 de 06/05/2026): `emails.txt`
- Diarização da reunião: `reuniao_raul_30_04_2026_20260513_112144_diarizacao.json`

## Participantes

- **Prof. Raul Fonseca Neto** — orientador (SPEAKER_00 na diarização)
- **Yuri / Saulo** — coorientador (SPEAKER_02)
- **George Gomes da Silva** — mestrando (SPEAKER_01)

## Prazo

**07/05/2026 (quinta-feira) — reunião marcada às 17h** (e-mail de 06/05/2026 às 10:15: *"se for preciso podemos nos reunir amanhã a tarde, por volta das 17 hrs"*). Mesma data já indicada na reunião como prazo ideal — *"então tá, então vamos ver se você não conseguir pra semana que vem, o ideal seria a semana que vem"* (~3155s).

---

## PARTE 1 — Ajuste único na primeira parte (já entregue)

> Citação do e-mail (22:04): *"A primeira parte do trabalho está ok. Somente modificar o problema C com centros rotacionados no sentido anti-horário."*

### 1. Modificar Problema C — rotação anti-horária dos centróides

- **O que fazer:** alterar a geração do Problema C para que os centróides sejam **rotacionados no sentido anti-horário** em relação ao Problema A — colocar a classe azul mais para baixo e a classe vermelha mais para cima. Hoje a transformação aplicada gera um Problema C visualmente muito similar ao Problema B.
- **Justificativa do orientador (~120s):** *"é, não, é porque aqui fica muito parecido apesar da distribuição ter dado dados diferentes né? Eu acho que é mais interessante a gente fazer o terceiro com a rotação anti-horária."* — e em ~138s: *"é que você faz uma rotação acho que se você faz uma rotação no sentido horário você podia fazer um mando anti-horário tá botar o azul pra baixo e o vermelho mais pra cima"*.
- **Arquivos:** `src/dissertacao_mestrado.py` (função `create_problem_c`).
- **Critério de pronto:** as figuras `01_all_three_problems.png` e os panéis individuais por problema mostram C visualmente distinto de B; a fronteira de decisão muda de orientação; resultados das Fases B e C divergem significativamente.

> **Todos os demais experimentos da primeira parte permanecem como estão:** Fases A/B/C originais, Oracle Validation, experimento de diluição, variantes de prompt, viés de ordem das classes, viés de ordem dos exemplos few-shot (recency bias), comparação de algoritmos, projeção R3 sintética, baselines clássicos. O orientador validou explicitamente que "está OK".

---

## PARTE 2 — Não-linearidade com peso × altura (NOVA)

> Citação do e-mail (22:04): *"A segunda parte, diz respeito a não linearidade"*

Plano de 5 passos numerados conforme o e-mail. **Importante:** a reunião (~2831s) reforça que **ambas as análises** devem ter problema não-linear; por isso a Parte 3 adiciona a meia-lua como segundo caso não-linear, complementar a peso×altura.

### 2. Usar o exemplo de peso × altura

- **O que fazer:** incorporar a base real **peso × altura (homem/mulher)** como problema central da Parte 2.
- **Origem da base:** enviada pelo orientador no e-mail de **30/04/2026 às 19:15** (*"Segue a base de dados. Consegui somente com cem amostras"*). Material da disciplina do orientador (NeuroSolutions / Cap2).
- **Tamanho:** **100 amostras** (atenção: bem menor que a expectativa inicial de ~1000 — pode impactar variância dos resultados, especialmente em few-shot com 40 exemplos onde restariam apenas 60 para teste).
- **Classificador ótimo (forma exata fornecida pelo orientador no e-mail de 19:15):**
  $$f(x_1, x_2) = x_2^2 - x_2 + x_1^2 - x_1 + \text{cte}$$
  — uma **curva elíptica** (fronteira quadrática). Substitui qualquer plano anterior de meia-lua.
- **Justificativa do orientador (~2030s):** *"até mesmo pra você ver se vier semântico peso e altura entendeu é um problema legal pra ser tratado aí como estudo de caso"*. Em ~2890s: *"você tem duas galcianas que tem superposição. Um classificador ótimo aqui é o baiseano, o ponto de paz. Ele é quadrático, quando há superposição"*.
- **Arquivos:** criar `src/dissertacao_mestrado.py::create_problem_homem_mulher` (ou equivalente) que carrega a base real fornecida pelo orientador; adicionar CSV à pasta `dados_sinteticos_seedXX/` ou pasta dedicada.

### 3. Variantes de prompt na nova base

- **O que fazer:** rotular os dados da base peso × altura pela LLM **variando a entrada do prompt**, conforme o e-mail:
  - **Variante neutra:** features rotuladas como `(x1, x2)`
  - **Variante semântica:** features rotuladas como `(peso, altura)`
  - **Pergunta:** classificar como **homem** ou **mulher**
- **Justificativa:** verificar se priors semânticos do LLM (associação peso/altura ↔ gênero) afetam a recuperação da métrica. Isso replica o experimento de "Nomes Semânticos de Features" da Parte 1, mas agora em base real.
- **Arquivos:** adaptar `build_prompt_zero_shot_variant` / `build_prompt_few_shot_variant` para suportar a pergunta homem/mulher; rodar o experimento sobre as duas variantes.

### 4. Fase A com 2, 3 e 4 features — análise de consistência da métrica

> Citação do e-mail (22:04): *"Na fase A inserir mais uma feature (hipérbole) ou duas features (elipse) e analisar a consistência da métrica aprendida utilizando 2, 3 ou 4 features nos dados peso e altura rotulados pela LLM seguindo o mesmo raciocínio da primeira parte do trabalho. Verificar se houve ganho."*

> **Reforço do e-mail de 19:15:** *"Talvez fosse interessante tb tentar com 4 features: x1, x2, x1^2 e x2^2"* — endossando explicitamente a versão de 4 features alinhada com a forma do classificador ótimo $x_2^2 - x_2 + x_1^2 - x_1 + cte$.

- **2 features:** $(peso, altura)$
- **3 features (hipérbole):** $(peso, altura, peso \cdot altura)$
- **4 features (elipse — alinhada com o classificador ótimo):** $(peso, altura, peso^2, altura^2)$
- **O que medir:** a **consistência da métrica aprendida** (fidelidade da métrica em reproduzir as rotulações do LLM) para cada configuração 2/3/4-features, replicando o pipeline da Parte 1 nos dados peso × altura rotulados pela LLM.
- **Expectativa:** o orientador deixou claro (~2740s) que se o problema for **linear**, ganhar componentes não deve melhorar; se houver **não-linearidade** no critério do LLM, a métrica de 3 ou 4 features deve mostrar ganho. É o teste decisivo para detectar não-linearidade no critério decisional do LLM. Citação (~2640s): *"se o esquema de rotulação da LLM tem alguma não linearidade a métrica com mais uma componente teria que refletir e se melhorar"*.
- **Arquivos:** `src/dissertacao_mestrado.py` — criar variante da Fase A parametrizada por número de features; `src/relaxed_perceptron.py` e `src/least_squares_inverse.py` já suportam $d$ features (verificar).
- **Critério de pronto:** tabela com fidelidade da métrica × número de features, separadamente para a variante neutra e a variante semântica do prompt.

### 5. Acurácia da métrica para a rotulação ORIGINAL

> Citação do e-mail: *"Apresentar também a acurácia da métrica para a rotulação da base de dados original."*

- **O que fazer:** além de medir a **fidelidade** (concordância da métrica com as decisões do LLM), também medir a **acurácia** da métrica em relação aos rótulos verdadeiros (homem/mulher) da base original.
- **Por quê:** isso compara o modelo substituto (métrica) diretamente com a verdade-terrestre, separando dois efeitos: (a) o quanto o LLM acerta, e (b) o quanto a métrica recupera a estratégia do LLM. Permite isolar o desempenho real do "shadow model" em relação à tarefa.
- **Apresentação:** tabela com colunas (n_features, fidelidade_vs_LLM, acuracia_vs_real, gap), para cada variante de prompt.

### 6. Fase D refeita com a melhor métrica

> Citação do e-mail: *"testar a consistência da LLM no aprendizado in-context (n-shot) com os dados rotulados pela melhor métrica (de 2, 3 ou 4 features), considerar os exemplos do prompt com 2, 3 ou 4 features."*

- **O que fazer:** identificar a **melhor métrica** entre 2/3/4 features (pelo critério dos itens 4 e 5) e usar essa métrica como **rotuladora** para a Fase D. Os exemplos do prompt devem incluir o **mesmo número de features** que a métrica vencedora.
- **Por quê:** se a melhor métrica usa 4 features (elipse), o prompt few-shot deve apresentar 4 features para cada exemplo, e o LLM precisa aprender o critério com essa representação aumentada.
- **Métricas a reportar:** acurácia, Kappa, F1 do LLM em relação aos rótulos da métrica vencedora, ao longo de $n$-shot $\in \{0, 5, 10, 20, 40\}$.
- **Arquivos:** `src/dissertacao_mestrado.py` — adaptar `phase_d_llm_as_learner`, `build_prompt_few_shot_variant`, `select_examples_by_strategy` para suportar número arbitrário de features.

### 7. Visualização gráfica de TODAS as rotulações do LLM

> Citação do e-mail (22:06 — adendo): *"Faltou uma coisa importante: Mostrar graficamente todas as rotulações feitas pela LLM."*

- **O que fazer:** gerar, para cada problema e cada configuração testada, um **scatter plot ponto-a-ponto** das rotulações da LLM (cores por classe predita pelo LLM).
- **Por quê:** as figuras atuais (consistência, kappa, fidelidade) são agregadas. Falta a visualização bruta das decisões individuais do LLM, que permite inspeção qualitativa de onde o LLM acerta/erra geometricamente.
- **Escopo mínimo:** gerar para a base peso × altura, ambas as variantes de prompt, todas as configurações da Fase A com 2/3/4 features, e para os $n$-shots da Fase D. Estender também para a meia-lua (Parte 3) e para os Problemas A/B/C/D da Parte 1.
- **Sugestão de nome:** `24_llm_labels_problem_*.png` ou similar.

---

## PARTE 3 — Segundo problema não-linear: meia-lua (NOVA)

> Citação da reunião (~2558s): *"mas aí você vai fazer o mesmo com o problema não linear. Então dá meia lua. Aí no não linear ele tende a melhorar"*. (~2831s): *"vai em todas as duas análises, você vai ter que colocar um problema não linear"*. (~2853s): *"A base da meia-lua você acha em qualquer lugar aí"*.

A reunião deixa claro que o R3 (kernel quadrático) **só faz sentido** em problemas com não-linearidade. Hoje os Problemas A/B/C são gaussianas anisotrópicas **lineares**, então adicionar $x_1 \cdot x_2$ não ajudou (de 84% caiu para 75% de fidelidade, ~2520s). A correção é introduzir um problema **explicitamente não-linear** — a meia-lua — e repetir Fase A e D nele com 2/3/4 features.

### 8. Adicionar problema meia-lua (Problema E)

- **O que fazer:** criar um novo problema sintético "meia-lua" com fronteira não-linear, usando `sklearn.datasets.make_moons(n_samples=150, noise=0.15, random_state=seed)`. Manter o mesmo número de amostras dos Problemas A/B/C/D (150) e as mesmas 3 sementes para consistência.
- **Justificativa:** a meia-lua é o caso canônico de não-linearidade na literatura clássica de ML; permite testar diretamente a hipótese de que o kernel quadrático melhora a recuperação da métrica quando a fronteira é genuinamente não-linear.
- **Arquivos:** `src/dissertacao_mestrado.py` — adicionar `create_problem_e_meia_lua()` espelhando o padrão dos `create_problem_*` existentes; adicionar flag `RUN_PROBLEM_E` (default `True`); salvar CSV em `dados_sinteticos_seedXX/problem_E.csv`.

### 9. Fase A meia-lua com 2 / 3 / 4 features

- **2 features:** $(x_1, x_2)$ originais.
- **3 features (hipérbole):** $(x_1, x_2, x_1 \cdot x_2)$ — extensão R3 já existente.
- **4 features (elipse):** $(x_1, x_2, x_1^2, x_2^2)$.
- **Métricas:** fidelidade da métrica em reproduzir o LLM **e** acurácia da métrica contra os rótulos verdadeiros da meia-lua. Mesmo formato da tabela do item 5.
- **Expectativa do orientador (~2740s):** *"se o problema é linear, matematicamente ele não vai melhorar [...] Agora se o problema é mais difícil, de repente se com a outra componente a gente conseguir melhorar a métrica, é sinal de que ela capturou uma não linearidade na estratégia da LLM"*.

### 10. Fase D meia-lua com a melhor métrica

- Espelhar o item 6 (Fase D peso × altura), agora com a meia-lua. Identificar melhor número de features (2/3/4) e testar n-shot $\in \{0, 5, 10, 20, 40\}$.

### 11. Comparação cruzada linear × não-linear

- Tabela consolidando os ganhos (ou ausência deles) ao passar de 2 para 3/4 features em **três cenários**: A/B/C lineares (já existe — sem ganho), meia-lua não-linear, peso × altura não-linear. Esse é o argumento central do artigo (~2843s): *"Eu acho que a gente fecha até um artigo já"*.

---

## PARTE 4 — Melhorias na apresentação Beamer

Pontos levantados pelo usuário com base na reunião e em revisão da apresentação atual (`execucao_2026-04-14_22-10-02/apresentacao/apresentacao.tex`).

### 12. Remover LP Max-Margin da apresentação e do trabalho

- **Decisão:** **remover** o LP Max-Margin do trabalho. Manter apenas Perceptron Estruturado + NNLS (Mínimos Quadrados).
- **Justificativa:** orientador na reunião (~552s, ~616s): *"dois são suficientes, Seu Jorge"* e *"se você não conseguir acertar, tira fora"*. O bug da restrição de convexidade não foi corrigido em tempo hábil, e dois algoritmos congruentes já bastam para demonstrar robustez do método de estimação. Substitui a indefinição anterior em "Anotações para próxima reunião (a)".
- **Ações concretas:**
  - Remover slide 13 (Algoritmo 3 — LP Max-Margin) da apresentação.
  - Remover colunas LP dos slides 14, 16, 19 (Comparação de Algoritmos) — manter scatter Perceptron × NNLS.
  - Remover figura `14_algorithm_comparison.png` ou regenerar com 2 algoritmos.
  - Não rodar `train_max_margin_lp` nas próximas execuções; flag `RUN_ALGORITHM_COMPARISON` pode ser mantida mas com 2 algoritmos.
  - Apagar / arquivar `src/max_margin_lp_inverse.py` (mover para `arquivado/` em vez de deletar).
  - Atualizar `CLAUDE.md` (seção Algoritmos de Otimização Inversa) — remover linha do LP.

### 13. Pseudo-código do Perceptron Estruturado com parâmetros completos

- **O que fazer:** no slide do Algoritmo 1, incluir explicitamente:
  - **Taxa de aprendizado** $\eta = 1$ (valor padrão, justificado pelo orientador em ~948s: *"Geralmente usa 1"*).
  - **$\Delta\gamma$** (incremento da busca binária na margem).
  - **C** (parâmetro de relaxação da margem).
  - **`max_iter`** (limite de iterações).
- **Justificativa:** orientador em ~920s notou que o pseudo-código apresentado omite esses parâmetros, dificultando a reprodutibilidade.
- **Arquivo:** `execucao_*/apresentacao/apresentacao.tex` — slide 11.

### 14. Pseudo-código dos Mínimos Quadrados (NNLS)

- **O que fazer:** adicionar ao slide do Algoritmo 2 o pseudo-código formal do NNLS:
  - Construção da matriz $A$ a partir das diferenças de distâncias aos centróides.
  - Construção do vetor $b$ (margem-alvo, tipicamente vetor de uns).
  - Chamada `w = scipy.optimize.nnls(A, b)` com restrição $w \geq 0$.
  - Normalização opcional final (ex.: $w / \|w\|$).
- **Justificativa:** simetria com o Perceptron e clareza para a banca (item explicitamente pedido pelo usuário em 13/05/2026).
- **Arquivo:** `apresentacao.tex` — slide 12.

### 15. Slide: motivação para múltiplas seeds

- **O que fazer:** criar um slide explicando **por que** o experimento usa três sementes ($42$, $123$, $7$) com três repetições por configuração.
- **Conteúdo:**
  - A geração dos dados é estocástica (gaussianas) — uma única semente pode produzir resultado atípico.
  - O LLM ainda tem alguma estocasticidade residual mesmo com temperatura zero.
  - Múltiplas seeds permitem: (i) estimar intervalos de confiança via bootstrap, (ii) testar reprodutibilidade (H5), (iii) descartar achados que dependam de configuração específica.
  - Citar literatura: prática padrão em ML empírico para evitar "p-hacking de seed".
- **Posição sugerida:** após "Configurações de Execução" (slide 10), antes de Fase A. Slide novo entre 10 e 11.

### 16. Slide: distribuição dos resultados por seed

- **O que fazer:** consolidar os gráficos por-seed (figuras `11_metric_errors_phase_a_seed*.png`, `16_dataset_overview_seed*.png`, `20_confusion_matrices_seed*.png`) em **um painel comparativo** mostrando lado a lado os resultados das três sementes.
- **Conteúdo:**
  - Boxplot/violin de fidelidade da Fase A por seed.
  - Distribuição do $w$ aprendido por seed (slide 18 já tem, mas reforçar com escala/comparação explícitas).
  - Tabela de Kappa, F1, acurácia por seed × algoritmo.
- **Mensagem:** demonstrar visualmente que os resultados são qualitativamente iguais entre seeds (suporta H5).
- **Arquivo:** novo slide entre 18 e 19.

### 17. Slides com prompts literais

- **O que fazer:** criar **2 a 4 slides** mostrando os prompts **exatamente como são enviados ao LLM**, em fonte monoespaçada (`\texttt{}` ou ambiente `verbatim` / `lstlisting`).
- **Conteúdo a mostrar (uma por slide ou agrupado):**
  - **Zero-shot default:** template completo do prompt zero-shot padrão.
  - **Few-shot default:** mesmo template com 5 exemplos inseridos.
  - **Variante geometric:** prompt com contexto espacial explícito.
  - **Variante CoT (chain-of-thought):** prompt pedindo raciocínio passo a passo.
  - **Variante tabular:** coordenadas como tabela markdown.
- **Justificativa:** a apresentação atual descreve as variantes (slide 28) mas não mostra o texto exato. Para a banca avaliar o experimento, é essencial ver o que de fato chega ao LLM. Os prompts estão em `build_prompt_zero_shot()` e `build_prompt_few_shot_variant()` em `src/dissertacao_mestrado.py`.
- **Posição sugerida:** após slide 28 (Design das Variantes de Prompt), antes do slide 29 (Resultados das Variantes).

### 18. Reorganizar estrutura da apresentação separando os dois paradigmas

- **O que fazer:** introduzir **dois blocos temáticos claros** na apresentação, com slides de transição:
  - **Bloco I — LLM como FONTE de decisões (otimização inversa)** — Problemas A, B, C. O LLM rotula; nós aprendemos $W$ a partir das decisões dele.
  - **Bloco II — LLM como APRENDIZ (in-context learning)** — Problemas D, peso×altura, meia-lua. Um perito (ou métrica) rotula; o LLM tenta reproduzir.
- **Por quê:** hoje a apresentação flui slide a slide sem demarcar essa distinção conceitual. Para a banca, é a separação **mais importante** do trabalho — duas teses experimentais distintas.
- **Implementação concreta:**
  - Slide de transição antes da Fase A: *"Bloco I — LLM como fonte de decisões"*.
  - Slide de transição antes da Fase D (atual slide 30): *"Bloco II — LLM como aprendiz"*.
  - Atualizar a Síntese das Hipóteses (slide 48) agrupando H1–H2 sob Bloco I e H3–H4 sob Bloco II.
- **Arquivo:** `apresentacao.tex` — inserir 2 frames de seção + ajustar narrativa.

### 19. Atualizar o `roteiro_apresentacao.txt`

- **O que fazer:** sincronizar o índice mestre `roteiro_apresentacao.txt` (na raiz do projeto) com todas as alterações dos itens 12–18:
  - **Remover** entrada do slide do Algoritmo 3 (LP Max-Margin) e referências em comparações.
  - **Atualizar** entradas dos slides do Perceptron e NNLS para refletir o pseudo-código completo (itens 13 e 14).
  - **Adicionar** entradas para: motivação de múltiplas seeds (item 15), distribuição por seed (item 16), prompts literais (item 17), e slides de transição Bloco I / Bloco II (item 18).
  - **Adicionar** entradas para Parte 3 (meia-lua): scatter inicial, Fase A 2/3/4 features, Fase D, comparação cruzada linear × não-linear (itens 8–11).
  - **Adicionar** entradas para Parte 2 (peso × altura) que ainda não estão no roteiro: itens 2–7.
- **Por quê:** o `CLAUDE.md` define o `roteiro_apresentacao.txt` como **índice mestre** da apresentação — qualquer slide novo ou removido deve ser refletido lá antes de regenerar o `.tex`. Manter o roteiro desatualizado quebra o fluxo "roteiro → tex" que o projeto usa.
- **Formato:** seguir o padrão atual do arquivo (slide a slide: título, tipo de conteúdo — texto/tabela/gráfico/fórmula —, e breve descrição).

### 20. Atualizar comentários e docstrings do código

- **O que fazer:** quando funções novas forem criadas (`create_problem_e_meia_lua`, `create_problem_homem_mulher`, `phase_a_multifeature`, etc.) e funções existentes forem alteradas (`create_problem_c`, `phase_d_llm_as_learner`, `build_prompt_*`), **atualizar os comentários e docstrings** das funções **e** o bloco-cabeçalho de `src/dissertacao_mestrado.py` (constantes, flags, descrição do fluxo).
- **Itens específicos:**
  - Remover qualquer referência a `train_max_margin_lp` ou LP nos comentários (item 12).
  - Atualizar docstring de `create_problem_c` descrevendo a rotação anti-horária (item 1).
  - Adicionar docstrings completas (Args, Returns, Raises) nas novas funções.
  - Atualizar a tabela "Configuração Principal" e "Flags de Execução" no `CLAUDE.md` com a nova flag `RUN_PROBLEM_E` e quaisquer outras adicionadas.
  - Atualizar a seção "Fluxo de Execução" do `CLAUDE.md` para incluir o Problema E e a Parte 2 com peso × altura.
- **Por quê:** comentários desatualizados confundem mais do que ajudam. Este passo foi explicitamente solicitado para garantir consistência entre código, documentação e apresentação.

---

## Anotações para a próxima reunião

Pontos levantados na reunião que **não constam dos e-mails** e que devem ser tratados como itens secundários (não bloqueantes para o prazo de 07/05/2026):

### a) LP Max-Margin — RESOLVIDO (decisão: remover)

- **Decisão final:** **remover** o LP Max-Margin do trabalho (ver item 12 da Parte 4). Manter apenas Perceptron + NNLS.
- **Citações da reunião que sustentam a remoção:**
  - (~552s) *"que dois são suficientes, Seu Jorge"*.
  - (~616s) *"se você não conseguir acertar, tira fora"*.
  - (~625s) *"Você não está nem respeitando aquela restrição de convexidade"*.

### b) Verificação da busca binária do $\gamma$ no Perceptron

- **Citação (~520s):** *"só você ver se o gama vai aumentando, porque a tendência dele é ficar igual a esse outro aqui. Entendeu? Às vezes você não está rodando o suficiente aqui para ele aproximar."*
- **Ação:** revisar `src/relaxed_perceptron.py` para garantir que a busca binária em $\gamma$ está convergindo (max_iter suficiente, critério de parada adequado).

### c) Pseudo-código (movido para Parte 4)

- Itens 13 e 14 da Parte 4 cobrem o pseudo-código de Perceptron e NNLS na apresentação. Resolvido.

### d) Formulação dos métodos — refinar na próxima reunião

- **Citação (~970s):** *"Deixa eu escrever aqui as coisas que a gente está perguntando. Na próxima reunião a gente já refina mais. Olha lá, formulação dos métodos, né? E os parâmetros do perceptro estruturado."*
- **Ação:** preparar uma seção de derivações/equivalências entre Perceptron Estruturado e LP Max-Margin para discussão.

---

## Mapeamento código → tarefa

| Item | Arquivos principais |
|---|---|
| 1. Problema C anti-horário | `src/dissertacao_mestrado.py::create_problem_c` |
| 2. Base peso × altura | nova função `create_problem_homem_mulher`; CSV em `dados_reais/homem_mulher/peso_altura.csv` (já criado) |
| 3. Variantes de prompt (peso × altura) | `build_prompt_zero_shot_variant`, `build_prompt_few_shot_variant`; constante `PROMPT_VARIANTS` |
| 4. Fase A com 2/3/4 features (peso × altura) | nova função `phase_a_multifeature`; reuso de `train_relaxed_perceptron` e `train_least_squares_inverse` |
| 5. Acurácia vs. rotulação original | `compute_consistency_metrics` com rótulos da base |
| 6. Fase D com melhor métrica (peso × altura) | adaptar `phase_d_llm_as_learner`, `select_examples_by_strategy`, `reorder_examples` para $d$ features |
| 7. Visualização ponto-a-ponto | nova rotina de plot; figuras `24_llm_labels_*.png` |
| 8. Problema E meia-lua | nova função `create_problem_e_meia_lua` usando `sklearn.datasets.make_moons`; flag `RUN_PROBLEM_E` |
| 9. Fase A meia-lua 2/3/4 features | reuso de `phase_a_multifeature` (item 4) |
| 10. Fase D meia-lua | reuso da adaptação do item 6 |
| 11. Comparação cruzada linear × não-linear | nova rotina de aggregation; figura/tabela consolidada |
| 12. Remover LP Max-Margin | mover `src/max_margin_lp_inverse.py` para `arquivado/`; limpar imports e chamadas em `src/dissertacao_mestrado.py`; remover slides LP de `apresentacao.tex` |
| 13. Pseudo-código Perceptron completo | `apresentacao.tex` slide 11 |
| 14. Pseudo-código NNLS | `apresentacao.tex` slide 12 |
| 15. Slide motivação múltiplas seeds | `apresentacao.tex` novo slide entre 10 e 11 |
| 16. Slide distribuição por seed | `apresentacao.tex` novo slide entre 18 e 19 |
| 17. Slides com prompts literais | `apresentacao.tex` novos slides após 28; usar `verbatim` ou `lstlisting`; conteúdo de `build_prompt_*` |
| 18. Reorganização em Bloco I / Bloco II | `apresentacao.tex` slides de transição + reorganização |
| 19. Atualizar `roteiro_apresentacao.txt` | `roteiro_apresentacao.txt` (raiz do projeto) — refletir itens 2–18 |
| 20. Atualizar comentários e docstrings | `src/dissertacao_mestrado.py` (cabeçalho + funções); `CLAUDE.md` (flags, fluxo, lista de problemas, remover LP) |
| (b) Busca binária $\gamma$ | `src/relaxed_perceptron.py` |

---

## Trechos críticos da transcrição (referência rápida)

- **~120s** — Problema C muito parecido com B; rotação anti-horária pedida.
- **~430–620s** — bug do LP Max-Margin; orientador considera remover se não funcionar.
- **~520s** — verificar busca binária de $\gamma$ no Perceptron.
- **~920s** — pseudo-código do Perceptron incompleto.
- **~970s** — anotações para a próxima reunião (formulação dos métodos).
- **~1480s** — viés de inversão das classes A/B → B/A altera até 25 p.p.
- **~1880s** — necessidade de problema não-linear; sugestão inicial de meia-lua.
- **~2030s, ~2890s** — base homem/mulher (peso × altura) como estudo de caso real.
- **~2110s** — comparar LLM com aprendizado de métrica via Perceptron na Fase D.
- **~2640–2740s** — repetir Fases A e D com componente quadrática para detectar não-linearidade.
- **~3155s** — prazo "quinta-feira da semana que vem".
