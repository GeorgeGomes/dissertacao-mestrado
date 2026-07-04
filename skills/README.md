# Comitê de avaliação de mestrado — skills

Oito skills no formato `SKILL.md` (padrão Claude Code) que formam um "comitê" para
**planejar, produzir e avaliar** um trabalho de mestrado em computação inteligente
(Python). Cada skill é um papel especializado, com fronteiras propositalmente claras
para não se sobreporem.

## Onde os skills ficam (importante)

O Claude Code só descobre skills automaticamente em **`.claude/skills/<nome>/SKILL.md`**
(escopo do projeto) ou `~/.claude/skills/` (pessoal). Esta pasta `skills/` é **apenas
documentação** — os arquivos ativos vivem em **`.claude/skills/`**, que é a cópia
canônica. Edite lá. Para verificar quais estão ativos, use `/skills` no Claude Code.

## Os oito papéis

| Skill | Papel | Pergunta central | Executa código? |
|---|---|---|---|
| `analisador-reunioes` | Rastreador dos pedidos do orientador | O que foi pedido e o que já está feito? | Não (lê e cruza) |
| `cientista-ic` | Pesquisador sênior / banca | O método é bom e bem justificado? | Não |
| `engenheiro-software` | Code review de design | O código é limpo e manutenível? | Não (lê, não roda) |
| `validador-codigo` | QA executável | Roda, reproduz e os números batem? | **Sim** |
| `escritor-artigo` | Redator científico | Como comunicar o trabalho por escrito? | Gera o texto |
| `validador-artigo` | Revisor / peer review | O artigo (texto) convence e se sustenta? | Não |
| `apresentador-slides` | Constrói o deck | Como apresentar isso com clareza? | Gera o arquivo |
| `validador-apresentacao` | Banca em ensaio | O deck convence e bate com o trabalho? | Não |

### Pares "produz × audita" (por que existem dois de cada)
- **`engenheiro-software` vs. `validador-codigo`**: o engenheiro *lê* e julga estilo e
  arquitetura sem rodar; o validador *executa* e confere reprodução. Design × QA.
- **`escritor-artigo` vs. `validador-artigo`**: o escritor *redige* o paper; o validador
  *audita* um manuscrito pronto, simulando revisão por pares.
- **`apresentador-slides` vs. `validador-apresentacao`**: o primeiro *constrói* o deck;
  o segundo *audita* um deck pronto, simulando a banca.
- **`cientista-ic`** é transversal: julga o mérito científico do método, não a forma
  (código, texto ou slides).

## Fluxo de orquestração sugerido

Do conteúdo à entrega:

0. **`analisador-reunioes`** — comece sabendo o que o orientador pediu e o que ainda
   falta (cruza as reuniões com código/artigo/slides). Define a fila de trabalho.
1. **`cientista-ic`** — garanta primeiro que o mérito científico se sustenta.
2. **`engenheiro-software`** — limpe e organize o código.
3. **`validador-codigo`** — rode e confirme que os resultados reproduzem o que o texto
   afirma. (Antes de escrever/apresentar: os números têm que vir de algo verificado.)
4. **`escritor-artigo`** — redija o artigo/dissertação a partir do trabalho já validado.
5. **`validador-artigo`** — audite o manuscrito como um revisor de conferência.
6. **`apresentador-slides`** — construa a defesa. (Regra do projeto: atualize sempre o
   `roteiro_apresentacao.txt` ANTES dos `.tex`.)
7. **`validador-apresentacao`** — ensaie/audite o deck contra o trabalho e a banca.

Repita os ciclos conforme o feedback. Os validadores são bons para rodar de novo a cada
iteração, porque são objetivos e rápidos.

## Ajustando ao seu trabalho

Os skills já trazem uma seção de calibragem específica deste projeto (otimização inversa
sobre decisões de LLM). Para afiar ainda mais, edite a seção "O que avaliar"/"Contexto"
de cada `SKILL.md` em `.claude/skills/` com o vocabulário da subárea (datasets, baselines
e métricas que a banca conhece). Quanto mais o skill souber do domínio, mais cirúrgico
fica o feedback.
