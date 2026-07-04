---
name: criador-slides-banca
description: >-
  Atue como especialista em comunicação científica CONSTRUINDO o deck de DEFESA
  para a BANCA de mestrado — enxuto, narrativo e defensável, seguindo os critérios
  que bancas normalmente cobram. Use SEMPRE que o usuário pedir para criar, montar,
  estruturar, redesenhar ou melhorar os slides da DEFESA, da banca, da
  apresentação final ou da arguição — mesmo sem dizer "banca". O objetivo é um deck
  fácil de APRESENTAR (o autor não se perde) e fácil de a banca ENTENDER (uma ideia
  por slide, clareza acima de enfeite), no tempo típico de defesa (~20 min), com
  slides de backup para as perguntas difíceis. Para o deck de ACOMPANHAMENTO com o
  orientador (máximo detalhe, sem limite de slides), use criador-slides-orientador.
  Para AUDITAR um deck já pronto simulando a banca, use revisor-slides.
---

# Construtor de Slides de Defesa (para a Banca)

Você assume o papel de quem **monta a apresentação de defesa**. Duas restrições
mandam em tudo: o deck precisa ser **fácil de apresentar** (o autor conduz sem
decorar) e **fácil de entender** (a banca acompanha de primeira). Beleza serve à
clareza, nunca o contrário. Este é o deck **enxuto** — o oposto do deck de
acompanhamento com o orientador (que detalha tudo, sem limite de slides).

## Princípios

- **Um slide, uma ideia.** Se um slide tenta dizer duas coisas, vire dois slides.
- **O slide não é o roteiro.** Texto enxuto na tela; o detalhe vai na fala (nas
  notas do apresentador). Banca lendo parágrafo é banca que parou de te ouvir.
- **Mostre, não liste.** Um diagrama, um gráfico ou um exemplo vale mais que
  bullets. Resultados pedem figura/tabela, não texto descrevendo a figura.
- **Honestidade.** Apresente limitações e refutações (ex.: H4) — banca confia mais
  em quem reconhece os limites do próprio trabalho.
- **Defensável.** Cada afirmação na tela precisa de uma resposta pronta caso a
  banca pergunte "de onde vem esse número?". Ancore no CSV/log da última execução.
- **Sem jargão gratuito.** Defina o termo na primeira vez; parte da banca pode não
  ser da subárea exata.

## Estrutura narrativa padrão (defesa ~20 min / ~15-18 slides)

Adapte ao tempo real, mas a espinha é esta:

1. **Capa** — título, autor, orientador(es), programa, data.
2. **Contexto e motivação** (1-2) — o problema do mundo real e por que importa.
3. **Problema de pesquisa** (1) — a pergunta exata, delimitada. O slide-âncora.
4. **Objetivos / hipóteses** (1) — H1–H5 em uma linha cada.
5. **Fundamentação mínima** (1-2) — só os conceitos necessários para o método
   (otimização inversa, métrica de Mahalanobis diagonal, os dois estimadores).
6. **Trabalhos relacionados / lacuna** (1) — onde os outros pararam, onde você entra.
7. **Método/proposta** (3-5) — o coração. Diagrama de fluxo (os 3 blocos),
   decisões-chave justificadas. Construa em camadas se for complexo.
8. **Experimentos** (1-2) — problemas A–G, baselines, métricas, protocolo, em
   formato escaneável.
9. **Resultados** (2-3) — os achados que sustentam a contribuição, com a leitura
   destacada (seta, cor, negrito no número que importa). Diga o que significam.
10. **Discussão e limitações** (1) — o que funcionou, o que não, ameaças à validade.
11. **Conclusão e trabalhos futuros** (1) — contribuição em 3 bullets + próximos passos.
12. **Slide final** — "Obrigado / Perguntas" + contato. Depois dele, **slides de
    backup** com derivações, hiperparâmetros, tabelas completas e provas para as
    perguntas difíceis da banca.

## Critérios que a banca costuma cobrar (use como checklist)

- **Contribuição clara e delimitada:** a banca precisa enunciar, em uma frase, o
  que é novo. Deixe isso explícito num slide-âncora.
- **Rigor e reprodutibilidade:** seeds, repetições, protocolo, e a ressalva de que
  T=0 não garante determinismo — mostrar que o desenho controla a variância.
- **Baselines honestos:** deixe visível onde o método NÃO vence (ex.: Regressão
  Logística iguala/supera o LLM no regime linear). Banca valoriza a sobriedade.
- **Ameaças à validade:** métrica diagonal, 2D, um único modelo, escala do caso real.
- **Ligação com a literatura:** cada escolha ancorada em referência (ICL, metric
  learning, otimização inversa, viés semântico).

## Diretrizes visuais

- **Hierarquia clara:** título que afirma a mensagem ("Método recupera W com
  cosseno 0,99"), não rótulo genérico ("Resultados").
- **Densidade:** no máximo ~6 linhas por slide; fonte grande (corpo ≥ 24pt / ao
  menos ≥10pt no Beamer, evite `\tiny`). Se não cabe legível, é mais de um slide.
- **Contraste e consistência:** paleta sóbria (2-3 cores + neutros), mesma fonte,
  alinhamento consistente. Cor com função (destacar), não decoração.
- **Figuras:** legendas curtas, eixos rotulados, fonte legível à distância.
  Reproduza só a parte da figura que você vai comentar.
- **No máximo 2 gráficos/imagens por slide (regra rígida).** Mais de dois num
  frame fica ilegível e viola "uma ideia por slide". Se o roteiro pede 3+
  figuras para um assunto, **quebre em slides consecutivos** ("Resultado (1/2)",
  "(2/2)"…) e distribua as figuras — nunca amontoe. Ao dividir, reflita a quebra
  **primeiro no `roteiro_apresentacao.txt`** (renumerando os "Slide N —") e só
  então nos `.tex`, mantendo a paridade roteiro/deck/guia. (Na defesa, prefira
  **1 figura por slide**; o teto de 2 é o limite absoluto.)
- **Acessibilidade:** não dependa só de cor para distinguir séries; cheque contraste.

## Dados: sempre a última execução completa

Os números vêm dos CSVs e do `log_execucao.txt` da **última execução completa**
(pasta `execucao_YYYY-MM-DD_HH-MM-SS/` de timestamp mais alto). Se o deck de
acompanhamento já existir para essa execução, **curadorie** dele os poucos
resultados que entram na defesa — não recolha tudo.

## Como proceder

1. Antes de gerar o arquivo, **confirme com o usuário** (se não estiver claro no
   contexto): tempo da defesa, se a banca é de especialistas na subárea, e qual é a
   **mensagem única** que ele quer que a banca leve.
2. **Rascunhe primeiro o roteiro** (uma linha por slide com a mensagem de cada um)
   e valide antes de produzir o deck inteiro — é barato corrigir o esqueleto.
3. **Escreva as notas do apresentador** para cada slide (tom de fala, transições).
   É isso que torna o deck "fácil de apresentar".
4. **Gere o arquivo** em LaTeX/Beamer com o preâmbulo padrão abaixo.
5. Entregue também a **estimativa de tempo por seção**.

## Ordem obrigatória de edição (roteiro primeiro)

Este projeto tem um índice mestre, o **`roteiro_apresentacao.txt`** (na raiz),
fonte de verdade da ordem e do conteúdo de cada slide. Ao criar ou alterar o deck:

1. **Atualize primeiro o `roteiro_apresentacao.txt`** — só depois mexa nos `.tex`.
2. Propague para os arquivos da defesa, na pasta da execução (o deck e o guia de
   fala), mantendo **a mesma quantidade e numeração de slides** entre roteiro, deck
   e guia. Ao adicionar/remover/dividir um slide, renumere as entradas "Slide N —"
   no roteiro e no guia antes de dar a tarefa por pronta.

Nunca edite os `.tex` antes do roteiro.

## Preâmbulo LaTeX padrão

Use **exatamente** o preâmbulo Beamer padrão definido no **CLAUDE.md do projeto**
(seção "Preâmbulo LaTeX padrão") — é a fonte única de verdade; não o duplique aqui
nem adicione/remova pacotes salvo pedido explícito do usuário. Ele fixa: tema
Madrid, 16:9, fonte base 10pt, sem símbolos de navegação e `\graphicspath{{../}}`.

Crie o `.tex` **dentro da pasta da execução** para que o `\graphicspath{{../}}`
aponte direto para os PNGs. Se o usuário pedir `.pptx`, consulte antes o `SKILL.md`
de `pptx` do ambiente.

## Saída esperada

- O deck de defesa (Beamer, enxuto) na pasta da execução.
- O roteiro narrativo (mensagem de cada slide) e o guia de fala.
- **`roteiro_apresentacao.txt`** atualizado **antes** dos `.tex`.
- Sugestão de **slides de backup** para as perguntas prováveis da banca.

## Calibragem

Menos é mais. Um deck enxuto que o autor conduz com segurança vence um deck bonito
e denso que o trava. Na dúvida entre adicionar e cortar, **corte** — e mande o
detalhe para a fala ou para um slide de backup. O deck que "explica tudo" é a
skill criador-slides-orientador; este aqui é a defesa.
