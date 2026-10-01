# Comitê de avaliação de mestrado — skills

Treze skills no formato `SKILL.md` (padrão Claude Code) que formam um "comitê" para
**planejar, produzir e avaliar** um trabalho de mestrado em computação inteligente
(Python). Cada skill é um papel especializado, com fronteiras propositalmente claras
para não se sobreporem.

## Onde os skills ficam (importante)

O Claude Code só descobre skills automaticamente em **`.claude/skills/<nome>/SKILL.md`**
(escopo do projeto) ou `~/.claude/skills/` (pessoal). Esta pasta `skills/` é **apenas
documentação** — os arquivos ativos vivem em **`.claude/skills/`**, que é a cópia
canônica. Edite lá. Para verificar quais estão ativos, use `/skills` no Claude Code.

## Os treze papéis

| Skill | Papel | Pergunta central | Executa código? |
|---|---|---|---|
| `analisador-reunioes` | Rastreador dos pedidos do orientador | O que foi pedido e o que já está feito? | Não (lê e cruza) |
| `revisor-cientifico` | Pesquisador sênior / banca | O método é bom e bem justificado? | Não |
| `revisor-codigo` | Code review de design | O código é limpo e manutenível? | Não (lê, não roda) |
| `testador-codigo` | QA executável | Roda, reproduz e os números batem? | **Sim** |
| `revisor-resultados` | Auditor da última execução | Os assets/CSVs batem com o código e o protocolo? | Não (lê a pasta `execucao_*`) |
| `escritor-artigo` | Redator científico | Como comunicar o trabalho por escrito? | Gera o texto |
| `conclusor-artigo` | Fecha o trabalho a partir da evidência | O que os experimentos mostraram, no fim? | Não (lê CSVs/log) |
| `revisor-artigo` | Revisor / peer review | O artigo (texto) convence e se sustenta? | Não |
| `revisor-referencias` | Bibliotecário das citações | Toda afirmação está citada e correta? | Não |
| `revisor-autoria` | Voz do autor e defensabilidade | Soa como o autor e resiste à arguição? | Não |
| `criador-slides-orientador` | Constrói o deck de acompanhamento (máximo detalhe) | Como explicar TUDO ao orientador? | Gera o arquivo |
| `criador-slides-banca` | Constrói o deck de defesa (enxuto) | Como apresentar isso com clareza em ~20 min? | Gera o arquivo |
| `revisor-slides` | Banca em ensaio | O deck convence e bate com o trabalho? | Não |

### Pares "produz × audita" (por que existem dois de cada)
- **`revisor-codigo` vs. `testador-codigo`**: o revisor *lê* e julga estilo e
  arquitetura sem rodar; o testador *executa* e confere reprodução. Design × QA.
  **`revisor-resultados`** audita a pasta de execução pronta (assets, nomes, números).
- **`escritor-artigo` / `conclusor-artigo` vs. `revisor-artigo`**: os primeiros
  *redigem* (seção sob demanda / conclusão a partir da evidência); o revisor *audita*
  um manuscrito pronto, simulando revisão por pares. `revisor-referencias` e
  `revisor-autoria` são lentes transversais (citações; voz e defensabilidade).
- **`criador-slides-orientador` / `criador-slides-banca` vs. `revisor-slides`**: os
  primeiros *constroem* o deck (acompanhamento detalhado / defesa enxuta); o segundo
  *audita* um deck pronto, simulando a banca.
- **`revisor-cientifico`** é transversal: julga o mérito científico do método, não a forma
  (código, texto ou slides).

## Fluxo de orquestração sugerido

Do conteúdo à entrega:

0. **`analisador-reunioes`** — comece sabendo o que o orientador pediu e o que ainda
   falta (cruza as reuniões com código/artigo/slides). Define a fila de trabalho.
1. **`revisor-cientifico`** — garanta primeiro que o mérito científico se sustenta.
2. **`revisor-codigo`** — limpe e organize o código.
3. **`testador-codigo`** e **`revisor-resultados`** — rode, confirme que os resultados
   reproduzem o que o texto afirma e que a pasta `execucao_*` está íntegra. (Antes de
   escrever/apresentar: os números têm que vir de algo verificado.)
4. **`escritor-artigo`** / **`conclusor-artigo`** — redija o artigo/dissertação a partir
   do trabalho já validado.
5. **`revisor-artigo`**, **`revisor-referencias`**, **`revisor-autoria`** — audite o
   manuscrito como um revisor de conferência, confira as citações e a voz do autor.
6. **`criador-slides-orientador`** / **`criador-slides-banca`** — construa o deck.
   (Regra do projeto: atualize sempre o `roteiro_apresentacao.txt` ANTES dos `.tex`.)
7. **`revisor-slides`** — ensaie/audite o deck contra o trabalho e a banca.

Repita os ciclos conforme o feedback. Os revisores são bons para rodar de novo a cada
iteração, porque são objetivos e rápidos.

## Ajustando ao seu trabalho

Os skills já trazem uma seção de calibragem específica deste projeto (otimização inversa
sobre decisões de LLM). Para afiar ainda mais, edite a seção "O que avaliar"/"Contexto"
de cada `SKILL.md` em `.claude/skills/` com o vocabulário da subárea (datasets, baselines
e métricas que a banca conhece). Quanto mais o skill souber do domínio, mais cirúrgico
fica o feedback.
