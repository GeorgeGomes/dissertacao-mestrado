# Explicando Decisões de LLMs via Otimização Inversa

Código da pesquisa de mestrado que investiga se um LLM possui um critério decisório
**estável e aprendível**. A ideia central: tratar as classificações de um LLM como
decisões observadas e, por **otimização inversa**, inferir o critério implícito como
uma **métrica de Mahalanobis diagonal** `W = diag(w1, w2)`, estimada por um
**Perceptron Estruturado com margem** e por **NNLS**. A classificação é feita por
centróide mais próximo sob a distância deformada por `W`.

> A documentação detalhada do método, dos passos de cada problema e dos outputs está em
> [`CLAUDE.md`](CLAUDE.md). Este README cobre apenas **como reproduzir** a execução.

## Requisitos

- Python 3.12 (validado em 3.12; 3.10+ deve funcionar)
- Uma chave do OpenRouter (`OPENROUTER_API_KEY`): provedor único. A execução padrão roda
  `openai/gpt-4o-mini` (pinado à OpenAI) e `google/gemini-2.5-flash-lite` (pinado ao
  Google) com o grid completo; qualquer outro modelo entra pelo id do OpenRouter em
  `MODELS_TO_TEST` (tupla `(provider, model, temperature)`), com alias em `MODEL_ALIAS`
  e pin em `MODEL_PROVIDER_PIN`.

## Reprodução do zero

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Instalar dependências (versões fixadas em requirements.txt)
pip install -r requirements.txt

# 3. Configurar a chave de API num arquivo .env na raiz do projeto
#    (o .env é ignorado pelo git; nunca commite chaves)
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-...     # única chave: todos os modelos via OpenRouter
EOF

# 4. Executar o experimento completo A PARTIR DA RAIZ do projeto.
#    O Python coloca src/ no sys.path automaticamente (é a pasta do script),
#    e os dados reais (dados_reais/) e as pastas de saída (resultados/execucao_*/) são
#    ancorados à raiz via BASE_DIR, não ao diretório de trabalho.
python src/dissertacao_mestrado.py

# ou, para uma execução CURTA de teste (1 seed, 1 repetição, cada grade few-shot = [0, 4];
# grades em GRADES_MODO_RAPIDO, aplicadas por aplicar_modo_rapido()):
python src/dissertacao_mestrado.py --rapido
python src/dissertacao_mestrado.py --rapido --modelo gpt      # só um modelo (substring)

# ou, para rodar SÓ alguns blocos (bloco1 | bloco2 | bloco3 | oraculo; aceita vários):
python src/dissertacao_mestrado.py --apenas bloco3
python src/dissertacao_mestrado.py --apenas bloco2 bloco3 --modelo gpt
python src/dissertacao_mestrado.py --apenas oraculo          # 0 chamadas de API, não exige chave nem saldo

# execução para o orientador (02/10/2026): Bloco 1 com os Passos 2 a 6 do Problema A
# (few-shot com exemplos do ground truth + teto) e Problemas B/C só até o Passo 3 (zero-shot):
python src/dissertacao_mestrado.py --apenas bloco1 --bc-zero-shot

# guarda de saldo: toda execução paga consulta o saldo do OpenRouter antes da 1ª chamada
# e aborta se ele não cobrir a estimativa (CUSTO_ESTIMADO_USD × MARGEM_SALDO);
# --ignorar-saldo pula a consulta:
python src/dissertacao_mestrado.py --apenas bloco3 --ignorar-saldo
```

Opções da linha de comando: `--rapido` (= `--smoke`), `--modelo <substring>`,
`--apenas <bloco> [...]`, `--bc-zero-shot`, `--ignorar-saldo`.

Com `--apenas`, a pasta de saída recebe o sufixo `_apenas-<blocos>` (nunca `_completa`),
deixando claro que é uma execução parcial.

Cada execução cria uma pasta `resultados/execucao_AAAA-MM-DD_HH-MM-SS_<sufixo>/` com todos os outputs
(gráficos PNG, tabelas PNG e `.tex`, CSVs por bloco, `log_execucao.txt` e `llm_interactions_parte*.json`),
em que `<sufixo>` é `completa` (fonte para o trabalho), `smoke` (`--rapido`) ou `apenas-<blocos>`.

**Códigos de saída:** 0 = execução completa; 2 = "EXPERIMENTO INCOMPLETO" (algum bloco levantou
exceção; ela é registrada no log com traceback, o checkpoint das interações é gravado e a
execução continua nos blocos seguintes, com CSVs, tabelas e log sempre salvos); 3 = créditos
esgotados no OpenRouter (HTTP 402; checkpoint + log parcial, sem rótulos de fallback; o cliente da API
é criado e fechado dentro do laço de eventos de cada coleta, então o log não traz tracebacks
"Event loop is closed").

> **Custo/tempo:** a coleta faz milhares de chamadas à API do LLM (assíncronas, com
> `asyncio.Semaphore(MAX_CONCURRENCY)`, 50 chamadas paralelas desde 14/10/2026). Custo medido:
> ~US$ 0,09 por milhão de caracteres de prompt; a completa de 08/10/2026 (158 mil chamadas)
> custou US$ 8 a 10; desde 13/10/2026 o Passo 3 do Problema G (few-shot do rótulo real) acrescenta
> cerca de 30 mil chamadas (+19%, ≈187 mil no total), e a estimativa passa a US$ 11 a 13; desde 14/10/2026 o
> Passo 6 do G com 2, 3 e 4 atributos no prompt acrescenta cerca de 35 mil (≈222 mil no total), estimativa US$ 13 a 16
> (`CUSTO_ESTIMADO_USD["completa"] = 17`, com folga, é o que a guarda de saldo exige);
> durações medidas: 2h23 com `MAX_CONCURRENCY=30` (158 mil chamadas, completa de 08/10/2026), 3h39 com 40 (187.328, completa de 13/10/2026),
> 3h58 com 50 (221.508, completa de 14/10/2026) e 2h57 com 50 (222.648, completa de 15/10/2026, já com coletas em lote e figuras em
> processos; cerca de 61 min desses foram 5 pontos da variante cot do Gemini que estouraram 5 tentativas de 90 s). Para um *smoke-test* rápido e barato, use a flag `--rapido`
> (equivalente a `--smoke`): 1 seed (42), 1 repetição, cada grade few-shot reduzida a
> zero-shot + k=4 (`K_EXEMPLOS_METRICA_BC`, `K_EXEMPLOS_APRENDIZ`, `K_VIESES`, …) e
> `K_EXEMPLOS_ROTULO_REAL=[4]`, sem editar o código (dicionário `GRADES_MODO_RAPIDO` em
> `src/dissertacao_mestrado.py`). **Não use `--rapido` para resultados
> finais**: serve apenas para validar que o código roda. Para controle mais fino,
> ajuste as constantes no topo de `src/dissertacao_mestrado.py` ou desligue blocos
> inteiros via as flags `RUN_*`.

## Métricas de avaliação

O sistema calcula, grava, plota e tabula apenas as cinco métricas da seção IV.3 da
metodologia do orientador, sempre com estes nomes:

| Métrica | O que mede | Onde |
|---|---|---|
| **Fidelidade** ("Fidelidade (métrica vs LLM)") | fração em que o centróide mais próximo sob o w estimado concorda com o LLM | Problemas A e D (e a estimação no Passo 2 do F e no Passo 4 do G) |
| **Acurácia** ("Acurácia (vs rótulo real)", "Acurácia (LLM vs especialista)") | concordância com o rótulo real ou com o especialista | oráculo e Bloco 2; nos Blocos 1 e 3 só como contexto "vs rótulo real" |
| **Consistência** | mesma fórmula da fidelidade, com os centróides dos NOVOS problemas (só o w é transferido do A) | Problemas B e C |
| **Cohen's κ** | concordância descontado o acaso, faixas de Landis e Koch (1977) | junto com as anteriores |
| **F1-Macro** | média das F1 por classe (`average='macro'`) | junto com as anteriores |

Consistência euclidiana, precision/recall, "concordância" e "taxa de erro" não existem mais
nas saídas (removidos em 09/10/2026).

## Reprodutibilidade

- **Sementes:** `RANDOM_SEEDS = [42, 123, 7]`, as mesmas em todos os experimentos e modelos.
  O RNG global do NumPy é re-semeado no início de cada iteração de semente; a geração de
  dados e a seleção de exemplos usam `RandomState` local explícito (`seed + rep` onde há sorteio).
- **Determinismo do LLM:** `temperatura = 0.0` minimiza a estocasticidade, mas **não**
  garante saída idêntica (ver `CLAUDE.md`). `N_REPETICOES` varia o SORTEIO dos exemplos
  few-shot onde há sorteio (regra única `reps_para`); a estocasticidade por consulta é
  medida pela auditoria offline: na execução completa de 15/10/2026, 1,46% das consultas
  idênticas (mesmo prompt + ponto) do gpt-4o-mini devolveram rótulo diferente (43 de 2.936
  perguntas repetidas), 0% no Gemini (1,4% na completa de 13/10/2026; em 07/2026 eram 4,3% no gpt-4o-mini).
- **Dados por semente** são salvos em `resultados/execucao_*/dados_sinteticos_seed{seed}/` para
  reprodução independente do código.
- **Auditoria offline:** `python src/audit_interactions.py` lê os
  `llm_interactions_parte*.json` da execução completa mais recente e reporta, sem chamar
  a API, a taxa de fallback do parser (`malformed`), a taxa de flip de `T=0` e o breakdown
  por modelo e por regime (campo `tag`). Falha (exit 1) se a taxa de malformadas passar de 5%.
  As interações são gravadas particionadas (`_parte001.json`, `_parte002.json`, …, 10 MiB cada;
  execuções pequenas gravam um único `llm_interactions.json`).
- **Figuras em cor e em preto e branco:** todo gráfico sai em dois arquivos, `_color` (como
  desenhado) e `_gray` (versão P&B gerada por `src/estilo_cinza.py`: símbolos por classe,
  traços e hachuras); as tabelas saem em `.tex` (em `tabelas/`) e em `.png` (só `_gray`).
  Convenção de nomes: `<bloco>_<graphic|table>_<NN>_<assunto>[_<problema>][_seed<S>][_<painel>][__<alias>]_<color|gray>.png`.

## Testes

Testes de unidade das peças críticas (390 testes em 15/10/2026, incluindo `tests/test_novidades_2026_10_15.py`: parser de respostas do LLM,
métricas, estimadores, seleção de exemplos, Passos 2 a 5 de A, D e G (`test_adg_passos_2_a_5.py`,
`test_g_passos_2_a_5.py`), atributos no prompt do Passo 6 do G (`test_g_passo6_atributos_prompt.py`), tabelas, versão P&B das
figuras, auditoria, infraestrutura do runner, regressão numérica sobre a execução completa de 05/07/2026) ficam em
[`tests/`](tests/), em `unittest` (o `pytest` não está nas dependências):

```bash
python -m unittest discover -s tests          # a partir da raiz do projeto
```

O `test_audit_interactions.py` roda a auditoria offline na execução completa mais
recente (verifica malformadas ≤ 5% e `temperature = 0.0`); é pulado se não houver
execução com `llm_interactions_parte*.json` no repositório. O
`test_regressao_execucao.py` compara valores dourados com os CSVs de
`resultados/execucao_2026-07-05_12-34-52_completa`.

Arquivos da suíte: `test_apenas_flag`, `test_assinaturas_chamadas`, `test_audit_interactions`,
`test_classical_baselines`, `test_correcoes_2026_10_08`, `test_correcoes_2026_10_09`, `test_creditos`,
`test_data_problems`, `test_estilo_cinza`, `test_estimators`, `test_adg_passos_2_a_5`,
`test_g_passos_2_a_5` (Passos 2 a 5 nos 70 pontos de treino do Problema G: `com_passos_3_a_5`,
eleição do especialista entre zero-shot e few-shot, colunas `especialista_regime/especialista_k/especialista_rep`, CSV
`bloco3_g_passos_2_a_5`, tabela `bloco3_table_06`, figura `bloco3_graphic_03`),
`test_gerar_amostras_homem_mulher`, `test_gerar_tabelas_bloco1`, `test_infra_main` (grades do
`--rapido`, nome da pasta, `Tee` restaurado após o abort por 402, ida e volta
`salvar_json_em_chunks` → `load_interactions`, ciclo de vida do cliente da API), `test_json_chunk`,
`test_llm_client`, `test_llm_retry_fallback`, `test_log_chunk`, `test_metrics`,
`test_nomenclatura_g1_g2`, `test_parser`, `test_plots_nomes`, `test_regressao_execucao`,
`test_relatorios`, `test_renomear_assets_legado`, `test_e_passo5_ordem` (Passo 5 de E: ordenações dos exemplos),
`test_e_passo3_exemplos` (Passo 3 de E: estratégias de seleção; cobertura de `MODEL_ALIAS`), `test_tabelas`.
(Nomes de 13/10/2026; o mapa de renomeação dos arquivos de teste está em `renomeacao_passos_2026-10-13.md`.)

## Ferramentas offline (0 chamadas de API)

```bash
python src/dissertacao_mestrado.py --reestimar-de resultados/execucao_<ts>_completa   # reestimação offline (15/10/2026)
```
Reaproveita os rótulos do LLM gravados em `llm_interactions*.json` da pasta indicada e refaz estimação, CSVs,
tabelas e figuras numa pasta `execucao_<ts>_completa_reestimada` (0 chamadas). Onde o prompt depende da métrica
estimada (Passo 4 de B/C, vieses com k > 0, Passo 6 do G) a resposta é aproximada pelo mesmo ponto (mesma tag ou
zero-shot) e o fim do log e `reestimacao_lacunas.csv` listam essas coletas; `--coletar-lacunas` coleta só esses
pontos na API.


```bash
python src/audit_interactions.py [pasta] [--json]      # auditoria das interações
python src/gerar_tabelas.py [pasta] [--sem-png --sem-docx]  # TODAS as tabelas: .tex em <pasta>/tabelas/ + .png na raiz (main() já faz isso)
python src/gerar_tabelas_bloco1.py [pasta] [--sem-docx]     # só as tabelas do Bloco 1 (ex-T1 a T9) + tabelas/todas_as_tabelas_bloco1.{tex,docx}
python src/regerar_simbolos.py [pasta]                  # regera bloco1_graphic_02/11/13 a partir dos CSVs e das interações
python src/renomear_assets_legado.py [pasta] [--aplicar --docs]  # migra UMA execução inteira para a convenção atual (dry-run padrão):
#   regras de 01/10/2026 (painéis/alias), 09/10/2026 (_color/_gray) e 13/10/2026 (mapa renomeacao_passos_2026-10-13.md:
#   nomes de arquivo E colunas/valores de CSV, tags fase= -> passo=). OBRIGATÓRIO em execuções anteriores a 13/10/2026
#   antes de gerar_tabelas.py ou audit_interactions.py (não há fallback para nomes antigos). Silencioso nas já migradas.
python src/animacao_perceptron/gerar.py                 # animação HTML do Perceptron (última _completa)
python src/animacao_perceptron/gerar_docx.py            # .docx com o pseudocódigo do fit e a correspondência com o .py
python src/gerar_entrega_orientador.py                   # pasta de entrega ao orientador (.docx + figuras + tabelas) da última _completa
# A ferramenta de correção H/M da execução de 07/2026 foi removida em 13/10/2026 (as cópias *_corrigido_hm*.csv ficam).
```

## Estrutura

```
src/
├── dissertacao_mestrado.py   # runner principal (config, passos dos problemas, coleta LLM, main)
├── plots.py                  # todas as visualizações (PNGs combinados + painéis, _color e _gray)
├── relatorios.py             # relatórios de texto (print_<problema>_passoN_*), média e faixa entre sementes, vereditos de H1 e H4, sintese_r2_r3_r4
├── execucao_io.py            # log, chunking, checkpoint, MODEL_ALIAS/llm_asset/asset_variant/com_cor
├── resultados.py             # dataclasses de resultado (ResultadoABC, ResultadoAprendiz, LearnedMetric)
├── protocolo.py              # constantes compartilhadas (EXPERT_W, PERCEPTRON_PARAMS, MIN_PONTOS_CLASSE_MINORITARIA, …)
├── llm_client.py             # factory de cliente de API (OpenRouter, provedor único) + pins
├── llm_parser.py             # parser de 8 camadas das respostas do LLM
├── metrics.py                # d_W, centróides, augmentação R3/R4, ConsistencyMetrics (acurácia, Cohen's κ, F1-Macro)
├── data_problems.py          # geradores dos problemas sintéticos + base real
├── relaxed_perceptron.py     # Perceptron Estruturado com relaxação de margem
├── least_squares_inverse.py  # estimador NNLS
├── classical_baselines.py    # baselines clássicos (k-NN, LR, SVM)
├── audit_interactions.py     # auditoria offline das interações (malformadas + flip de T=0, por modelo e regime)
├── tabelas.py                # motor Tabela → .tex (booktabs) e .png; faixa_kappa (Landis e Koch)
├── estilo_cinza.py           # para_cinza(fig) e garantir_legendas(fig): versão P&B de qualquer figura
├── gerar_tabelas.py          # catálogo completo de tabelas (Blocos 1/2/3, oráculo, auditoria, fechamento)
├── gerar_tabelas_bloco1.py   # tabelas do Bloco 1 (ex-T1 a T9; tab_a_*, tab_bc_*, tab_d_*, tab_ad_*) + todas_as_tabelas_bloco1.tex/.docx
├── regerar_simbolos.py       # regera bloco1_graphic_02/11/13 offline (CSVs + llm_interactions)
├── gerar_amostras_homem_mulher.py    # amostras X/T/R da base peso×altura (orientador)
├── renomear_assets_legado.py # migra uma execução inteira para a convenção atual (regras de 01/10, 09/10 e 13/10/2026: arquivos, colunas e tags; silencioso nas já migradas)
└── animacao_perceptron/      # animação HTML do Perceptron (gerar.py + template + core.js) e pseudocódigo (pseudocodigo.py + gerar_docx.py)
tests/                        # unittest, 390 testes em 15/10/2026 (parser, métricas, estimadores, Passos 2 a 5 de A/D/G, atributos no prompt do Passo 6 do G, plots, tabelas, estilo_cinza, auditoria, infra do runner, regressão)
dados_reais/homem_mulher/     # base real peso × altura + amostras X/T/R
requirements.txt              # dependências com versões fixadas
CLAUDE.md                     # documentação detalhada do método e dos outputs
glossario.md                  # glossário dos termos do trabalho
passos_por_problema.md        # lista dos passos de cada problema (nomenclatura "Problema X, Passo N")
problemas_tabela.html         # tabela dos problemas + animação didática dos passos (página autônoma)
resultados/                   # uma pasta execucao_<timestamp>_<completa|smoke|apenas-...> por execução (commitadas)
```
