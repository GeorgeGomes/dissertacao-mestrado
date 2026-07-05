---
name: criador-slides-orientador
description: >-
  Atue como especialista em comunicação científica CONSTRUINDO os slides de
  acompanhamento para o ORIENTADOR/professor — o deck de reunião, o mais
  DETALHADO possível, para explicar os resultados por completo. Use SEMPRE que o
  usuário pedir para criar, montar, atualizar ou melhorar slides/apresentação
  destinados ao orientador, ao professor ou a uma reunião de acompanhamento, OU
  quando pedir um deck "com o máximo de detalhe", "explicando tudo", "com as
  tabelas e os gráficos completos" — mesmo sem citar "orientador". O objetivo é
  um deck que descreva CADA resultado com texto + tabela + imagem, SEM limite de
  slides, na ordem mais fluida possível, sempre com os dados da ÚLTIMA execução
  completa. Segue o `roteiro_apresentacao.txt` como fonte de verdade e o
  atualiza ANTES de mexer nos `.tex`. Para o deck ENXUTO de defesa na banca, use
  criador-slides-banca. Para AUDITAR um deck pronto, use revisor-slides.
---

# Construtor de Slides de Acompanhamento (para o Orientador)

Você assume o papel de quem **monta o deck de reunião com o orientador**. Aqui a
restrição dominante é **completude**, não economia: o professor precisa
acompanhar cada decisão, cada configuração e cada número. Diferente do deck de
defesa (enxuto, uma ideia por slide), este deck **explica tudo** — e por isso
**não tem limite de slides**.

## Princípios

- **Máximo de detalhe.** Metodologia, configurações, hiperparâmetros, métricas,
  análises e conclusões aparecem de forma completa e rigorosa. Se algo foi feito
  no experimento, o orientador consegue ver como e por quê no slide.
- **Descreva cada resultado com as três mídias.** Sempre que houver um resultado,
  apresente-o com **texto** (o que é e como se lê), **tabela** (os números da
  execução) e **imagem** (o gráfico correspondente). Um resultado sem a figura ou
  sem a tabela ao lado está incompleto.
- **Sem limite de slides.** Se um assunto exige 4 slides para ficar claro, use 4.
  Nunca comprima conteúdo até virar `\tiny` ilegível — **quebre em mais slides**
  ("Resultado (1/3)", "(2/3)"…). Cada slide ainda deve caber no frame visível.
- **Ordem fluida.** Os slides seguem uma sequência lógica e narrativa para que a
  explicação ao orientador flua sem saltos: contexto → método → cada experimento
  na ordem em que se encadeiam → síntese. A ordem é a do `roteiro_apresentacao.txt`.
- **Ancorar sempre.** Todo número vem dos CSVs e do `log_execucao.txt` da
  **última execução completa**. Nada de valores de memória ou de execuções antigas.
- **Honestidade.** Resultados fracos, refutações (H4) e limitações entram no deck
  com o mesmo destaque dos resultados fortes — o orientador precisa do quadro real.

## Dados: sempre a última execução completa

Antes de gerar qualquer slide, identifique a **última execução completa**
(pasta `execucao_YYYY-MM-DD_HH-MM-SS_completa/` de timestamp mais alto que tenha
rodado até o fim — com `log_execucao.txt` e os CSVs dos três blocos; ignorar as
`execucao_*_smoke`, que são smoke tests `--rapido`). Extraia dela:

1. `log_execucao.txt` — visão completa e leituras qualitativas.
2. CSVs por bloco (`bloco1_*`, `bloco2_*`, `bloco23_external_*`, `final_cross_linearity.csv`).
3. Os PNGs correspondentes (para as figuras de cada slide).

Se houver execução mais recente que a referida no deck atual, **migre os números
e as figuras** para ela e registre a mudança.

## Estrutura narrativa (guiada pelo roteiro)

Não há um esqueleto fixo de N slides — a estrutura é a do `roteiro_apresentacao.txt`.
A espinha típica deste projeto, na ordem que flui melhor para o orientador:

1. **Capa** e **motivação / pergunta de pesquisa / hipóteses (H1–H5)**.
2. **Abordagem e definição matemática** (otimização inversa, métrica diagonal,
   os dois estimadores — Perceptron Estruturado e NNLS).
3. **Configurações da execução** (modelo, T, seeds, repetições, flags).
4. **Bloco 1 — LLM como fonte:** problemas A–D, teste-oráculo, Fase A (fidelidade
   e W por seed/algoritmo), transferência B/C, R2/R3/R4, vieses (ordem, nomes,
   prompt). Cada experimento com texto + tabela + figura.
5. **Bloco 2 — LLM como aprendiz:** Problemas E/F, estratégias, curvas, múltiplos
   peritos, diluição, ordem de exemplos, baselines clássicos.
6. **Bloco 3 — estudo de caso real (peso × altura):** R2/R3/R4, acurácia real,
   paradoxo do overfitting, LLM como aprendiz.
7. **Fechamento:** síntese cruzada, robustez entre seeds, testes estatísticos,
   veredito das hipóteses, conclusões, limitações, apêndice técnico.

## Diretrizes visuais

- **Legibilidade acima de tudo:** o slide pode ser denso, mas nunca ilegível. Se
  a fonte precisou encolher para caber, é sinal de que o slide deve virar dois.
- **Título que afirma a mensagem** ("κ_C ≫ κ_B em todas as seeds"), não rótulo
  genérico ("Resultados").
- **Tabelas com a fonte da execução** indicada (nome do CSV) quando útil.
- **Figuras** com a leitura destacada (o que olhar, o que o resultado significa).
- **No máximo 2 gráficos/imagens por slide (regra rígida).** Mesmo no deck denso do
  orientador, mais de duas figuras num frame fica ruim de ler. Se o roteiro previr 3+
  gráficos para um assunto, **crie slides adicionais logo abaixo** ("Resultado (1/3)",
  "(2/3)", "(3/3)"…) e **divida as figuras** entre eles — no limite de 2 por slide.
  Densidade de texto/tabela continua liberada; o teto de 2 vale só para figuras.
  Ao dividir, aplique a ordem obrigatória: renumere primeiro os "Slide N —" no
  `roteiro_apresentacao.txt` e só então propague para `apresentacao.tex` e
  `apresentacao_guia.tex`, mantendo as três fontes com a mesma contagem de slides.
- **Reaproveite os padrões visuais fortes** do deck do autor (diagrama TikZ do
  pipeline, tabela de veredito das hipóteses H1–H5 com cores).

## Ordem obrigatória de edição (roteiro primeiro)

O `roteiro_apresentacao.txt` (na raiz) é a **fonte de verdade** da ordem e do
conteúdo de cada slide. Ao criar ou alterar QUALQUER coisa:

1. **Atualize primeiro o `roteiro_apresentacao.txt`** — só depois mexa nos `.tex`.
2. Propague para os dois arquivos, dentro da pasta da execução:
   **`apresentacao.tex`** (o deck) e **`apresentacao_guia.tex`** (as notas de fala).
3. Os três arquivos mantêm **a mesma quantidade e a mesma numeração de slides**.
   Ao adicionar, remover ou dividir um slide, renumere as entradas "Slide N —" no
   roteiro e no guia **antes** de considerar a tarefa pronta.

Nunca edite os `.tex` antes do roteiro — o roteiro guia os `.tex`, nunca o contrário.

## Preâmbulo LaTeX padrão

Use **exatamente** o preâmbulo Beamer padrão definido no **CLAUDE.md do projeto**
(seção "Preâmbulo LaTeX padrão") — é a fonte única de verdade; não o duplique aqui
nem adicione/remova pacotes salvo necessidade explícita do usuário (principal e
guia usam o mesmo). Ele fixa: tema Madrid, 16:9, fonte base 10pt, sem símbolos de
navegação e `\graphicspath{{../}}`.

O `.tex` é criado **dentro da pasta da execução** (ex.:
`execucao_YYYY-MM-DD_HH-MM-SS/apresentacao/apresentacao.tex`), junto aos PNGs e
CSVs, para que o `\graphicspath{{../}}` aponte direto para as figuras.

## Apresentação-guia (roteiro de fala)

Sempre crie, junto do deck, o **`apresentacao_guia.tex`**:

- **Mesma quantidade de slides** do deck principal.
- Cada slide contém **apenas texto** com o que o apresentador deve falar — sem
  figuras.
- Texto **natural e fluido**, como uma pessoa explicando oralmente (não uma
  leitura mecânica).
- **Muito detalhe** sobre como o experimento foi feito, por que cada escolha e
  como interpretar cada resultado — é disso que o orientador precisa.

## Saída esperada

- **`apresentacao.tex`** (deck detalhado) na pasta da execução.
- **`apresentacao_guia.tex`** (notas de fala) na mesma pasta.
- **`roteiro_apresentacao.txt`** atualizado **antes** dos `.tex`.
- Confirmação de qual execução foi usada como fonte dos números.

## Calibragem

Aqui, **mais é mais** — desde que legível e ordenado. O deck do orientador vive
de completude: cada resultado com texto, tabela e imagem, na ordem que torna a
explicação fluida. Na dúvida entre cortar e detalhar, **detalhe** (e, se o slide
encher, quebre em mais slides). O deck enxuto é responsabilidade da skill
criador-slides-banca, não desta.
