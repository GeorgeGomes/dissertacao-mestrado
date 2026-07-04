# Skills do projeto — convenção de nomes e registro

Este diretório contém as skills (slash-commands) do projeto de mestrado. Para evitar
nomes fora de padrão e skills duplicadas, **toda skill segue uma convenção única** e
ocupa **uma célula única** na matriz `ação × artefato` abaixo.

## Convenção de nomes

Formato: **`<ação-agente>-<artefato>`**

- **kebab-case**, tudo minúsculo, palavras inteiras em português (sem siglas).
- O *head* (primeira palavra) descreve a **ação**, não a profissão. Heads fixos:

| Ação | Head | Significado |
|---|---|---|
| Criar / produzir | `escritor-`, `criador-` | gera o artefato do zero |
| Auditar artefato pronto (read-only) | `revisor-` | critica algo já existente, **sem executar** |
| Executar / rodar para verificar | `testador-` | roda o código e compara números |
| Diagnosticar / rastrear | `analisador-` | lê e reporta estado, não implementa |
| Sintetizar / concluir | `conclusor-` | lê a evidência (código + última execução) e fecha os achados + a conclusão |

- O *tail* (segunda palavra) é o **artefato** no singular: `artigo`, `slides`, `codigo`,
  `referencias`, `reunioes`. Exceção: as **revisões transversais por lente** usam a
  **lente** como tail, por atravessarem vários artefatos: `revisor-cientifico` (lente do
  mérito) e `revisor-autoria` (lente de autenticidade/voz/defensabilidade).
- Quando um mesmo artefato é **criado para públicos diferentes**, o tail recebe um
  sufixo de audiência: `criador-slides-orientador` (deck de acompanhamento, máximo
  detalhe) e `criador-slides-banca` (deck de defesa, enxuto). São duas skills porque o
  público muda o que entra e como — não é duplicação.

## Matriz `ação × artefato` (registro oficial)

Cada célula tem **no máximo uma** skill. Antes de criar uma skill nova, ache a célula:
se já estiver ocupada, **estenda a skill existente** em vez de criar outra. **Exceção:**
a célula *Criar × Slides* comporta duas skills que se dividem por **audiência**
(orientador vs. banca) — ver a nota de sufixo de audiência acima.

| Artefato | Criar | Auditar (pronto) | Executar | Diagnóstico | Concluir |
|---|---|---|---|---|---|
| **Artigo / texto** | `escritor-artigo` | `revisor-artigo` | — | — | `conclusor-artigo` |
| **Slides / apresentação** | `criador-slides-orientador` (acompanhamento) · `criador-slides-banca` (defesa) | `revisor-slides` | — | — | — |
| **Código** | — | `revisor-codigo` (engenharia) · `revisor-cientifico` (mérito do método) | `testador-codigo` | — | — |
| **Referências** | — | `revisor-referencias` | — | — | — |
| **Reuniões** | — | — | — | `analisador-reunioes` | — |
| **Transversal (lente)** | — | `revisor-cientifico` (mérito) · `revisor-autoria` (autoria/voz) | — | — | — |

## As 12 skills

| Skill | Papel | NÃO faz (encaminha para) |
|---|---|---|
| `escritor-artigo` | Redige texto científico (abstract, seções, carta a revisores). | mérito → `revisor-cientifico`; auditar pronto → `revisor-artigo`; slides → `criador-slides-orientador`/`criador-slides-banca` |
| `revisor-artigo` | Audita o artigo pronto (revisão por pares). | mérito do método → `revisor-cientifico`; redigir → `escritor-artigo`; rodar → `testador-codigo` |
| `revisor-cientifico` | Julga o mérito científico (rigor, validade, contribuição). | estilo de código → `revisor-codigo`; execução → `testador-codigo` |
| `criador-slides-orientador` | Constrói o deck de acompanhamento com o orientador: máximo detalhe (texto + tabela + imagem por resultado), sem limite de slides, última execução completa. | deck de defesa → `criador-slides-banca`; auditar deck → `revisor-slides` |
| `criador-slides-banca` | Constrói o deck enxuto de defesa na banca (uma ideia por slide, ~20 min, backup para perguntas). | deck de acompanhamento → `criador-slides-orientador`; auditar deck → `revisor-slides` |
| `revisor-slides` | Audita um deck já montado (simula a banca). | construir → `criador-slides-orientador`/`criador-slides-banca` |
| `revisor-codigo` | Code review de engenharia, **sem rodar**. | resultados científicos → `revisor-cientifico`; executar → `testador-codigo` |
| `testador-codigo` | **Executa** o código: instala, roda, compara números. | estilo → `revisor-codigo`; mérito → `revisor-cientifico` |
| `revisor-referencias` | Integridade das citações (texto + slides) e PDFs em `artigos_referenciados/`. | mérito → `revisor-cientifico`; redigir → `escritor-artigo`; auditar artigo todo → `revisor-artigo` |
| `analisador-reunioes` | Lê `reunioes_orientador/`, cruza pedidos com código/texto/slides. | implementar → skill apropriada acima |
| `revisor-autoria` | Revisão transversal por lente: voz do autor, prontidão de defesa, rastreabilidade e disclosure de IA (texto + slides). **Não** ajuda a fraudar autoria nem burlar detecção. | redigir seções → `escritor-artigo`; mérito → `revisor-cientifico`; peer-review → `revisor-artigo`; slides → `criador-slides-orientador`/`criador-slides-banca`/`revisor-slides` |
| `conclusor-artigo` | Lê código + última execução, deriva os achados definitivos e fecha a conclusão / monta o artigo completo. | redigir seção sob demanda → `escritor-artigo`; mérito → `revisor-cientifico`; auditar pronto → `revisor-artigo`; rodar → `testador-codigo` |

## Regras para não duplicar

1. **Uma célula = uma skill.** Skill nova só nasce se abrir uma célula vazia da matriz.
2. **Fronteiras explícitas.** Toda `description` deve dizer o que a skill **NÃO** faz e
   para qual skill encaminhar (padrão "use `outra-skill`"). Atualize esses ponteiros ao
   renomear qualquer skill.
3. **Heads são fechados.** Use apenas `escritor`/`criador`/`revisor`/`testador`/
   `analisador`/`conclusor`. Um head novo só se surgir uma classe de ação genuinamente
   nova (foi o caso de `conclusor-`: sintetizar a evidência e fechar os achados — nem
   redigir sob demanda como `escritor-`, nem só diagnosticar como `analisador-`).
