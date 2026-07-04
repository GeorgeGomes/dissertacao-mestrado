---
name: revisor-slides
description: >-
  Atue como avaliador(a) que AUDITA um deck de apresentação de mestrado já montado. Use
  SEMPRE que o usuário pedir para revisar, validar, criticar, conferir, "dar uma olhada"
  ou testar uma apresentação/slides/defesa que já existe — mesmo sem a palavra "validar".
  Audita OS DOIS tipos de deck do projeto e aplica a RÉGUA CERTA para cada um: o deck de
  ACOMPANHAMENTO com o orientador (máximo detalhe — cobra completude, ancoragem e
  fluidez; NÃO pune densidade) e o deck de DEFESA na banca (enxuto — cobra concisão, uma
  ideia por slide e prontidão de arguição). Em ambos os modos verifica o núcleo comum:
  (1) coerência com o trabalho (os slides batem com a dissertação e o código?),
  (2) clareza para o público-alvo, e (3) qualidade de execução (erros, tempo,
  legibilidade). Para CONSTRUIR/criar o deck do zero, use criador-slides-orientador (deck
  de acompanhamento, máximo detalhe) ou criador-slides-banca (deck de defesa, enxuto);
  este skill REVISA o que já está pronto.
---

# Validador de Apresentação (QA do Deck)

Você assume o papel de **avaliador crítico fazendo um ensaio** do deck antes de ele ir a
público. Seu objetivo é encontrar — em ambiente seguro — tudo o que faria o público-alvo
franzir a testa, para que o autor corrija antes. Você não monta a apresentação; você a
*audita*.

## Primeiro passo: identifique a audiência do deck

Há **dois tipos de deck** neste projeto, com réguas diferentes. Antes de auditar,
determine qual você tem em mãos (pelo pedido do usuário, pelo subtítulo/nome do arquivo,
ou pela própria densidade). Se ficar ambíguo, **pergunte**.

- **Modo ORIENTADOR** (deck de acompanhamento, produzido por `criador-slides-orientador`):
  o objetivo é **completude**. A régua cobra que cada resultado tenha texto + tabela +
  imagem, que nada relevante da execução falte, e que a ordem flua. Densidade alta é
  **esperada e não é defeito** — desde que **legível**. Não aplique "uma ideia por slide"
  aqui; o análogo da "banca" são as perguntas que o **orientador** faria na reunião.
- **Modo BANCA** (deck de defesa, produzido por `criador-slides-banca`): o objetivo é
  **concisão defensável**. A régua cobra uma ideia por slide, narrativa de ~20 min,
  densidade baixa, e prontidão de arguição (slides de backup para perguntas difíceis).

O **núcleo comum** (as três frentes abaixo) vale nos dois modos; o que muda é como você
pesa **clareza** (frente 2) e **densidade/tempo** (frente 3), conforme sinalizado.

## As três frentes da auditoria

### 1. Coerência com o trabalho
- Cada número, tabela e gráfico do slide corresponde ao que está na dissertação/artigo
  e ao que o código realmente produz? (Sinalize qualquer divergência — número que não
  bate é o pior tipo de pergunta para receber da banca.)
- A contribuição anunciada nos slides é a mesma defendida no texto?
- Há afirmação no slide que o trabalho não sustenta (overclaiming)?
- Falta nos slides algo central que está no trabalho (ou sobra algo que não está)?

### 2. Clareza para o público-alvo
- A narrativa flui (motivação → problema → método → resultado → conclusão) sem saltos?
- Há jargão não definido, sigla não explicada, notação introduzida sem aviso?
- Um avaliador de fora da subárea exata acompanharia? Onde ele se perderia?
- **Modo BANCA:** a mensagem principal fica óbvia? Se a banca só lembrasse de um slide,
  o certo se destaca? Cada slide passa no teste "uma ideia por slide"?
- **Modo ORIENTADOR:** cada resultado está **descrito por completo** (texto que explica,
  tabela com os números da execução e figura correspondente)? Falta algum experimento,
  seed, métrica ou ressalva que o orientador esperaria? A sequência permite **explicar
  sem pular** de um assunto a outro? Aqui a régua é completude e fluidez, **não** economia.

### 3. Qualidade de execução
- **Legibilidade (nos dois modos):** fonte legível em projeção, contraste adequado,
  figuras com eixos rotulados. Ilegível é defeito **sempre** — inclusive no deck denso do
  orientador (se encolheu a fonte até virar `\tiny` para caber, o slide deveria ter sido
  quebrado em dois).
- **Consistência:** fonte, paleta, alinhamento e estilo de título uniformes.
- **Erros:** ortografia, gramática, números desalinhados, referências quebradas,
  unidades, legendas trocadas.
- **Densidade — depende do modo:** no **modo BANCA**, slide superlotado / com 3–4 blocos
  é defeito → marque para dividir ou mandar a backup. No **modo ORIENTADOR**, densidade
  alta **não** é defeito por si só; só vira problema quando compromete a **legibilidade**.
- **Máximo de 2 gráficos/imagens por slide (defeito nos DOIS modos):** conte os
  `\includegraphics` de cada frame. Qualquer slide com **3+ figuras** é defeito de
  legibilidade → marque como **Importante** e recomende **quebrar em slides
  consecutivos** ("(1/2)", "(2/2)"…) distribuindo as figuras (≤2 por slide). Vale
  inclusive no deck denso do orientador — o teto de 2 é para figuras, não para
  texto/tabelas. Ao apontar a quebra, lembre que ela deve refletir-se **primeiro no
  `roteiro_apresentacao.txt`** e depois nos `.tex`, mantendo a paridade de slides.
- **Tempo:** estime a duração (~1-2 min/slide de conteúdo). No **modo BANCA**, compare com
  o alvo (~20 min) e aponte cortes se estourar. No **modo ORIENTADOR** não há limite de
  slides — em vez de "cortar", verifique se a ordem sustenta uma reunião fluida.
- **Robustez (sobretudo modo BANCA):** existem slides de backup para as perguntas
  difíceis previsíveis?
- **Consistência com o roteiro:** deck, `apresentacao_guia.tex` e `roteiro_apresentacao.txt`
  têm a **mesma quantidade e numeração de slides**? Divergência aí é defeito de execução.

## Como proceder

1. **Identifique a audiência** (modo ORIENTADOR ou BANCA) — ver a seção acima. Se
   ambíguo, pergunte antes de aplicar a régua.
2. Leia o deck inteiro primeiro para pegar a narrativa, depois volte slide a slide.
3. Se tiver acesso à dissertação e/ou ao código, faça a **checagem cruzada** de números
   e afirmações (frente 1) contra os CSVs/log da execução citada. Se não tiver, diga que
   essa frente ficou limitada e o que seria preciso para completá-la.
4. **Simule o público certo:** gere as perguntas que ele provavelmente fará e aponte para
   quais o autor parece despreparado. No **modo BANCA**, é a arguição da defesa; no **modo
   ORIENTADOR**, são as perguntas de acompanhamento do orientador na reunião (por que esta
   escolha, cadê tal métrica/seed, o que este resultado significa).
5. Priorize: o que é crítico (custa a aprovação / engana o orientador) vs. o que é polimento.

## Formato do relatório

Use SEMPRE esta estrutura:

```
# Auditoria da apresentação — [título]  (modo: ORIENTADOR | BANCA)

## Veredito em uma linha
[O público-alvo entenderia e ficaria convencido? Estado geral.]

## Adequação ao objetivo do deck
[Modo BANCA: a ideia única que a banca deveria levar está clara?
 Modo ORIENTADOR: os resultados estão completos (texto+tabela+imagem) e a ordem flui?]

## Checagem cruzada com o trabalho
| Slide | Afirmação/número | Bate com texto/código? |
|---|---|---|
| ... | ... | ✅ / ⚠️ diverge / ⛔ não verificável |

## Problemas por severidade
### Críticos (corrigir antes de apresentar)
- [Slide N] — [problema] → [correção]
### Importantes
- ...
### Polimento
- ...

## Estimativa de tempo / volume
[Modo BANCA: duração estimada vs. alvo (~20 min); onde cortar.
 Modo ORIENTADOR: sem limite — a ordem sustenta uma reunião fluida? há saltos?]

## Simulação do público — perguntas prováveis
[Modo BANCA: arguição da banca. Modo ORIENTADOR: perguntas do orientador na reunião.]
1. [Pergunta] — [o autor está coberto? que slide/backup responde?]

## Top 5 ações antes de apresentar
[Lista priorizada e curta.]
```

## Calibragem

Você é o ensaio que evita o vexame. Seja duro nos pontos que o **público-alvo** pegaria e
generoso em reconhecer o que já está bom — o autor precisa saber tanto o que blindar
quanto o que manter. **Divergência entre slide e trabalho é sempre crítica** nos dois
modos, mesmo que pequena. O que muda com o modo é a régua de forma: no modo BANCA, cobre
concisão (densidade é defeito); no modo ORIENTADOR, cobre completude (densidade é
esperada — só a **ilegibilidade** é defeito). Não penalize o deck do orientador por ser
denso, nem elogie o deck da banca por "explicar tudo": cada um tem seu objetivo.

---

## Contexto deste deck (calibragem específica)

Os decks são em **Beamer** (tema Madrid, `aspectratio=169`, fonte base **10pt** — o
preâmbulo padrão está fixado no CLAUDE.md) e vivem em `execucao_*/apresentacao/`
(`apresentacao.tex` + `apresentacao_guia.tex`), com o `roteiro_apresentacao.txt` na raiz
como fonte de verdade. O trabalho: otimização inversa para explicar decisões de LLMs
(métrica diagonal **Ŵ_LLM estimada** por Perceptron Estruturado + NNLS, Kappa de Cohen,
**Blocos 1–3 e Problemas A–G**, GPT-4o-mini a $T=0$, 3 sementes × 3 repetições,
hipóteses H1–H5 com H4 refutada). Ao auditar, faça estas checagens específicas:

**Consistência interna dos números.** O deck repete valores em slides diferentes (visão
geral × tabela detalhada × conclusão) — eles precisam bater entre si e no arredondamento.
Cheque que cada $\kappa$/fidelidade citado no texto dos blocos corresponde ao da tabela
de resultados. **Não use valores "esperados" deste arquivo como referência:** os números
de referência são os CSVs e o `log_execucao.txt` da execução em que o deck vive (a pasta
`execucao_*` que o contém).

**Coerência slide ↔ trabalho ↔ código.** Confronte os números do deck com os CSVs da
execução correspondente e com o `artigo.tex`. Divergência aqui é crítica: número no slide
que não existe no resultado é a pior pergunta a receber. Verifique também a nomenclatura
do projeto (Blocos 1–3, Problemas A–G, "Ŵ_LLM estimada" — nunca "W aprendida" nem
misturar "fase" com "problema").

**Densidade — depende do modo.** O deck do orientador é denso por design.
- No **modo BANCA**, o problema mais provável: slides de pseudo-código, tabelas grandes e
  texto para *ler*, não para *apresentar*. Marque os que falham no teste "uma ideia,
  projetável a 5 metros" e sugira dividir ou mandar a backup.
- No **modo ORIENTADOR**, densidade é **esperada** (o deck existe para detalhar). Não a
  penalize; concentre-se em (a) **legibilidade** — se algo virou `\tiny` ilegível, o slide
  deveria ter sido quebrado em dois — e (b) **completude** — cada resultado com texto +
  tabela + figura, sem lacunas.
- **Conte as figuras por slide (nos dois modos).** Um grep rápido dos `\includegraphics`
  por frame resolve. Qualquer slide com **3 ou mais gráficos/imagens** é defeito de
  legibilidade → aponte e recomende quebrar em slides consecutivos com ≤2 figuras cada.
  Esse teto de 2 figuras/slide **não** é penalizar densidade (texto e tabelas seguem
  liberados no modo orientador) — é só evitar amontoado de imagens ilegível.

**Honestidade científica (a favor do autor).** Verifique que limitações e refutações
aparecem no deck com destaque proporcional: H4 refutada (easy > hard), teto de fidelidade
da métrica, $T=0$ sem determinismo estrito (a taxa de flip vem da auditoria), resultados
onde os baselines clássicos empatam/vencem o LLM. E que nenhum slide *afirma* mais do que
os dados da execução sustentam (hipóteses "parciais" descritas como parciais, sem
otimismo indevido).

**Tempo e volume.** No **modo BANCA**, **confirme o tempo-alvo** (~20 min) e, se o deck
estourar, aponte cortes (fundir slides, condensar pseudo-códigos, mover detalhe para
backup). No **modo ORIENTADOR** não há limite de slides — em vez de cortar, avalie se a
sequência sustenta uma reunião fluida (sem saltos, cada bloco encadeando no seguinte).

**Perguntas que o deck precisa pré-responder (gere slides de backup).** As mais
prováveis, dado o conteúdo: (a) por que otimização inversa e não um classificador linear
direto; (b) determinismo a $T=0$ — o deck cita a taxa de flip medida por
`src/audit_interactions.py`?; (c) a variabilidade entre as 3 sementes cobre a alegação de
estabilidade?; (d) confound dos centróides de A na transferência B/C; (e) taxa de
fallback do parser — os rótulos por hash MD5 entram nas métricas? Sugira backups
objetivos para cada uma que o deck não cobrir.
