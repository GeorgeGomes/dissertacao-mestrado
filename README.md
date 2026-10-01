# Explicando Decisões de LLMs via Otimização Inversa

Código da pesquisa de mestrado que investiga se um LLM possui um critério decisório
**estável e aprendível**. A ideia central: tratar as classificações de um LLM como
decisões observadas e, por **otimização inversa**, inferir o critério implícito como
uma **métrica de Mahalanobis diagonal** `W = diag(w1, w2)`, estimada por um
**Perceptron Estruturado com margem** e por **NNLS**. A classificação é feita por
centróide/vizinho mais próximo sob a distância deformada por `W`.

> A documentação detalhada do método, das fases experimentais e dos outputs está em
> [`CLAUDE.md`](CLAUDE.md). Este README cobre apenas **como reproduzir** a execução.

## Requisitos

- Python 3.12 (validado em 3.12; 3.10+ deve funcionar)
- Uma chave do OpenRouter (`OPENROUTER_API_KEY`): provedor único. O protocolo roda
  `openai/gpt-4o-mini` (pinado à OpenAI) e `google/gemini-2.5-flash-lite` (pinado ao
  Google) com o grid completo; qualquer outro modelo entra pelo id do OpenRouter.

## Reprodução do zero

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Instalar dependências (versões fixadas em requirements.txt)
pip install -r requirements.txt

# 3. Configurar as chaves de API num arquivo .env na raiz do projeto
#    (o .env é ignorado pelo git — nunca commite chaves)
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-...     # única chave: todos os modelos via OpenRouter
EOF

# 4. Executar o experimento completo A PARTIR DA RAIZ do projeto.
#    O Python coloca src/ no sys.path automaticamente (é a pasta do script),
#    e os dados reais (dados_reais/) e as pastas de saída (execucao_*/) são
#    ancorados à raiz via BASE_DIR — não ao diretório de trabalho.
python src/dissertacao_mestrado.py

# ou, para uma execução CURTA de teste (few-shot [0, 4], 1 repetição, só a seed 42):
python src/dissertacao_mestrado.py --rapido

# ou, para rodar SÓ alguns blocos (bloco1 | bloco2 | bloco3 | oraculo; aceita vários):
python src/dissertacao_mestrado.py --apenas bloco3
python src/dissertacao_mestrado.py --apenas bloco2 bloco3 --modelo gpt
```

Com `--apenas`, a pasta de saída recebe o sufixo `_apenas-<blocos>` (nunca `_completa`),
deixando claro que é uma execução parcial.

Cada execução cria uma pasta `execucao_AAAA-MM-DD_HH-MM-SS_<sufixo>/` com todos os outputs
(gráficos PNG, CSVs por bloco, `log_execucao.txt` e `llm_interactions_parte*.json`), em que
`<sufixo>` é `completa` (fonte para o trabalho), `smoke` (`--rapido`) ou `apenas-<blocos>`.

> **Custo/tempo:** a coleta faz milhares de chamadas à API do LLM (assíncronas, com
> `asyncio.Semaphore`). Para um *smoke-test* rápido e barato, use a flag `--rapido`
> (equivalente a `--smoke`): reduz para `FEW_SHOT_SIZES=[0, 4]`, `N_REPETICOES=1` e
> `RANDOM_SEEDS=[42]`, sem editar o código. **Não use `--rapido` para resultados
> finais** — serve apenas para validar que o pipeline roda. Para controle mais fino,
> ajuste as constantes no topo de `src/dissertacao_mestrado.py` ou desligue blocos
> inteiros via as flags `RUN_*`.

## Reprodutibilidade

- **Sementes:** `RANDOM_SEEDS = [42, 123, 7]`. O RNG global do NumPy é re-semeado no
  início de cada iteração de semente; a geração de dados e a seleção de exemplos usam
  `RandomState` local explícito.
- **Determinismo do LLM:** `temperatura = 0.0` minimiza a estocasticidade, mas **não**
  garante saída idêntica (ver `CLAUDE.md`). `N_REPETICOES` varia o SORTEIO dos exemplos
  few-shot onde há sorteio (regra única `reps_para`); a estocasticidade por consulta é
  medida pela auditoria offline: ~5% das consultas idênticas (mesmo prompt+ponto)
  devolvem rótulo diferente entre repetições.
- **Dados por semente** são salvos em `execucao_*/dados_sinteticos_seed{seed}/` para
  reprodução independente do código.
- **Auditoria offline:** `python src/audit_interactions.py` lê os
  `llm_interactions_parte*.json` da execução completa mais recente e reporta, sem chamar
  a API, a taxa de fallback do parser (`malformed`) e a taxa de flip de `T=0`. Falha
  (exit 1) se a taxa de malformadas passar de 5%. As interações são gravadas
  particionadas (`_parte001.json`, `_parte002.json`, …, 10 MiB cada; execuções pequenas
  gravam um único `llm_interactions.json`).

## Testes

Testes de unidade das peças críticas (parser de respostas do LLM, métricas de
concordância, estimadores e o auditor de interações) ficam em [`tests/`](tests/):

```bash
cd src
../.venv/bin/python -m pytest ../tests -q     # ou: python -m unittest discover -s ../tests
```

O `test_audit_interactions.py` roda a auditoria offline na execução completa mais
recente (verifica malformadas ≤ 5% e `temperature = 0.0`); é pulado se não houver
execução com `llm_interactions_parte*.json` no repositório.

## Estrutura

```
src/
├── dissertacao_mestrado.py   # runner principal (config, fases, coleta LLM, main)
├── plots.py                  # todas as visualizações (PNGs combinados + painéis)
├── relatorios.py             # relatórios de texto, bootstrap CI, tabela cruzada
├── execucao_io.py            # log, chunking, checkpoint, MODEL_ALIAS/llm_asset/asset_variant
├── resultados.py             # dataclasses de resultado
├── protocolo.py              # constantes compartilhadas (EXPERT_W, PERCEPTRON_PARAMS, …)
├── llm_client.py             # factory de cliente de API (OpenRouter, provedor único) + pins
├── llm_parser.py             # parser de 8 camadas das respostas do LLM
├── metrics.py                # d_W, centróides, augmentação R3/R4, consistência
├── data_problems.py          # geradores dos problemas sintéticos + base real
├── relaxed_perceptron.py     # Perceptron Estruturado com relaxação de margem
├── least_squares_inverse.py  # estimador NNLS
├── classical_baselines.py    # baselines clássicos (k-NN, LR, SVM)
├── audit_interactions.py     # auditoria offline das interações (malformadas + flip de T=0)
├── gerar_amostras_homem_mulher.py    # amostras X/T/R da base peso×altura (orientador)
├── corrigir_mapeamento_homem_mulher.py  # correção offline H/M (*_corrigido_hm__<alias>.csv)
├── renomear_assets_legado.py # migra assets de execuções antigas para a convenção atual
└── animacao_perceptron/      # animação HTML do Perceptron (gerar.py + template)
tests/                        # testes de unidade (parser, métricas, estimadores, plots, auditoria)
requirements.txt              # dependências com versões fixadas
CLAUDE.md                     # documentação detalhada do método e dos outputs
```
