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
- Uma chave de API da OpenAI (GPT-4o-mini é o modelo ativo por padrão)

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
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...   # opcional (só se ativar Claude em MODELS_TO_TEST)
# GOOGLE_API_KEY=...             # opcional (só se ativar Gemini)
EOF

# 4. Executar o experimento completo A PARTIR DA RAIZ do projeto.
#    O Python coloca src/ no sys.path automaticamente (é a pasta do script),
#    e os dados reais (dados_reais/) e as pastas de saída (execucao_*/) são
#    ancorados à raiz via BASE_DIR — não ao diretório de trabalho.
python src/dissertacao_mestrado.py

# ou, para uma execução CURTA de teste (few-shot [0, 5], 1 repetição, só a seed 42):
python src/dissertacao_mestrado.py --rapido
```

Cada execução cria uma pasta `execucao_AAAA-MM-DD_HH-MM-SS/` com todos os outputs
(gráficos PNG, CSVs por bloco, `log_execucao.txt` e `llm_interactions_parte*.json`).

> **Custo/tempo:** a coleta faz milhares de chamadas à API do LLM (assíncronas, com
> `asyncio.Semaphore`). Para um *smoke-test* rápido e barato, use a flag `--rapido`
> (equivalente a `--smoke`): reduz para `FEW_SHOT_SIZES=[0, 5]`, `N_REPETICOES=1` e
> `RANDOM_SEEDS=[42]`, sem editar o código. **Não use `--rapido` para resultados
> finais** — serve apenas para validar que o pipeline roda. Para controle mais fino,
> ajuste as constantes no topo de `src/dissertacao_mestrado.py` ou desligue blocos
> inteiros via as flags `RUN_*`.

## Reprodutibilidade

- **Sementes:** `RANDOM_SEEDS = [42, 123, 7]`. O RNG global do NumPy é re-semeado no
  início de cada iteração de semente; a geração de dados e a seleção de exemplos usam
  `RandomState` local explícito.
- **Determinismo do LLM:** `temperatura = 0.0` minimiza a estocasticidade, mas **não**
  garante saída idêntica (ver `CLAUDE.md`). `N_REPETICOES` mede a variabilidade na
  mesma base. Medido na prática: ~5% das consultas idênticas (mesmo prompt+ponto)
  devolvem rótulo diferente entre repetições.
- **Dados por semente** são salvos em `execucao_*/dados_sinteticos_seed{seed}/` para
  reprodução independente do código.
- **Auditoria offline:** `python src/audit_interactions.py` lê os
  `llm_interactions_parte*.json` da execução completa mais recente e reporta, sem chamar
  a API, a taxa de fallback do parser (`malformed`) e a taxa de flip de `T=0`. Falha
  (exit 1) se a taxa de malformadas passar de 5%. As interações são gravadas
  particionadas (`_parte001.json`, `_parte002.json`, …, ~10 MB cada).

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
├── dissertacao_mestrado.py   # runner principal (config, cliente LLM, parser, fases, plots, I/O)
├── relaxed_perceptron.py     # Perceptron Estruturado com relaxação de margem
├── least_squares_inverse.py  # estimador NNLS
├── classical_baselines.py    # baselines clássicos (k-NN, LR, SVM)
└── audit_interactions.py     # auditoria offline das interações (malformadas + flip de T=0)
tests/                        # testes de unidade (parser, métricas, estimadores, auditoria)
requirements.txt              # dependências com versões fixadas
CLAUDE.md                     # documentação detalhada do método e dos outputs
```
