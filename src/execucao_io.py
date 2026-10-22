"""Infraestrutura de I/O da execução: log, particionamento e nomes de asset.

Extraído de ``dissertacao_mestrado.py`` (etapa 1 da modularização, 2026): ``Tee``
(stdout -> buffer p/ log_execucao.txt), particionamento de log/JSON em chunks
de ~10 MiB, checkpoint atômico das interações e a convenção de nomes de asset
por modelo (``MODEL_ALIAS``/``llm_asset``). Nada aqui depende do runner.

``LLM_INTERACTIONS`` é a lista global de interações com o LLM: o runner e a
coleta apenas fazem ``append`` (nunca rebind), então importá-la preserva o
aliasing: todos os módulos veem a MESMA lista.
"""
import json
import os
import re
import sys
import time
from pathlib import Path


class Tee:
    """Duplica um stream: exibe no terminal E armazena em buffer para o log TXT.

    Instalar em stdout E stderr (compartilhando o MESMO buffer, na ordem de
    chegada): os avisos dos estimadores (``warnings.warn`` → stderr) e
    tracebacks precisam constar no log_execucao.txt persistido; sem capturar
    stderr, o log diria "zero avisos" falsamente (3 avisos de não-convergência
    de γ foram perdidos assim no smoke de 03/07/2026).

    Uso no runner::

        tee = Tee()                                  # envolve sys.stdout
        sys.stdout = tee
        sys.stderr = Tee(sys.stderr, tee._buffer)    # mesmo buffer, ordem de chegada
    """
    def __init__(self, stream=None, buffer=None):
        self._stream = stream if stream is not None else sys.stdout
        self._buffer = buffer if buffer is not None else []

    def write(self, msg):
        # 15/10/2026: cabeçalhos de etapa (">>> " e "[ETAPA") ganham o tempo decorrido, para o
        # log mostrar quanto cada seção levou (API × estimação × figuras × tabelas).
        if _T0 is not None and msg.lstrip("\n").startswith((">>> ", "[ETAPA")) and not msg.endswith("\n"):
            msg = f"{msg}   {decorrido()}"
        self._stream.write(msg)
        self._buffer.append(msg)

    def flush(self):
        self._stream.flush()

    def getvalue(self):
        return ''.join(self._buffer)


# ── Cronômetro da execução (15/10/2026) ───────────────────────────────────────
# `iniciar_cronometro()` no começo de main(); `decorrido()` devolve "⏱ H:MM:SS (+M:SS)" com o
# total desde o início e o intervalo desde a marca anterior. Usado por `Tee.write` nos
# cabeçalhos ">>> "/"[ETAPA" e por `relatorios.print_section`.
_T0 = None
_T_MARCA = None


def iniciar_cronometro():
    global _T0, _T_MARCA
    _T0 = _T_MARCA = time.time()


def _fmt_s(seg: float) -> str:
    seg = int(round(seg)); h, r = divmod(seg, 3600); m, s_ = divmod(r, 60)
    return f"{h}:{m:02d}:{s_:02d}"


def decorrido(marcar: bool = True) -> str:
    """Tempo decorrido desde `iniciar_cronometro()`; vazio se o cronômetro não foi iniciado."""
    global _T_MARCA
    if _T0 is None:
        return ""
    agora = time.time()
    txt = f"⏱ {_fmt_s(agora - _T0)} (+{_fmt_s(agora - _T_MARCA)})"
    if marcar:
        _T_MARCA = agora
    return txt


# Raiz das execuções (08/10/2026): toda pasta execucao_* (novas e antigas) vive em
# <repo>/resultados/. Ferramentas offline e testes procuram aqui.
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = BASE_DIR / "resultados"

# Limite de tamanho antes de fatiar em partes (10 MiB). Vale para o log (.txt) e o JSON.
LOG_CHUNK_LIMIT_BYTES = 10 * 1024 * 1024


def _agrupar_por_tamanho(itens, tamanho_bytes, limite_bytes):
    """Agrupa `itens` em blocos (listas) cujo tamanho somado fica ~<= `limite_bytes`.

    Estratégia gulosa COMPARTILHADA por `salvar_log_em_chunks` (itens = linhas do log)
    e `salvar_json_em_chunks` (itens = elementos da lista de interações): fecha o bloco
    atual quando o próximo item o faria estourar o limite; nunca divide um item; e um
    item isolado maior que o limite fica sozinho no seu bloco (não entra em loop).

    `tamanho_bytes(item) -> int` devolve o tamanho serializado de um item, em bytes.
    """
    blocos = []
    atual = []
    bytes_atual = 0
    for item in itens:
        b = tamanho_bytes(item)
        if atual and bytes_atual + b > limite_bytes:
            blocos.append(atual)
            atual = []
            bytes_atual = 0
        atual.append(item)
        bytes_atual += b
    if atual:
        blocos.append(atual)
    return blocos


def salvar_log_em_chunks(conteudo, pasta_execucao, nome_base="log_execucao",
                         limite_bytes=LOG_CHUNK_LIMIT_BYTES):
    """Salva o log em TXT, fatiando em partes quando ultrapassa `limite_bytes`.

    - Se o conteúdo couber em `limite_bytes` (medido em UTF-8), grava um único
      arquivo `{nome_base}.txt` (comportamento antigo, retrocompatível).
    - Caso contrário, divide em `{nome_base}_parte001.txt`, `_parte002.txt`, ...,
      cada parte com no máximo ~`limite_bytes`. A quebra é feita em limites de
      LINHA (nunca no meio de uma linha), então cada parte permanece legível.
    - Cada parte recebe um cabeçalho "PARTE i/N" para facilitar a leitura posterior.

    Usa a mesma estratégia de agrupamento (`_agrupar_por_tamanho`) que
    `salvar_json_em_chunks`; a diferença é o que conta como "item" (aqui, linhas de
    texto) e como cada parte é escrita.

    Retorna a lista de caminhos gravados.
    """
    tamanho_total = len(conteudo.encode("utf-8"))

    # Caso comum: log pequeno → arquivo único, sem alterar o formato de sempre.
    if tamanho_total <= limite_bytes:
        caminho = os.path.join(pasta_execucao, f"{nome_base}.txt")
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
        return [caminho]

    # Log grande → agrupa LINHAS em blocos de até `limite_bytes` (quebra sempre em
    # limite de linha, mantendo cada parte legível).
    linhas = conteudo.splitlines(keepends=True)
    blocos = _agrupar_por_tamanho(linhas, lambda l: len(l.encode("utf-8")), limite_bytes)
    chunks = ["".join(bloco) for bloco in blocos]

    total_partes = len(chunks)
    caminhos = []
    for i, chunk in enumerate(chunks, start=1):
        caminho = os.path.join(pasta_execucao, f"{nome_base}_parte{i:03d}.txt")
        cabecalho = (
            f"# {nome_base} — PARTE {i}/{total_partes}\n"
            f"# Log fatiado automaticamente (limite de {limite_bytes // (1024*1024)} MiB por arquivo).\n"
            f"# Concatene as partes na ordem para reconstruir o log completo.\n"
            f"{'=' * 70}\n"
        )
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(cabecalho)
            f.write(chunk)
        caminhos.append(caminho)
    return caminhos


def salvar_json_em_chunks(dados, pasta_execucao, nome_base="llm_interactions",
                          limite_bytes=LOG_CHUNK_LIMIT_BYTES):
    """Salva uma lista em JSON, fatiando em partes quando ultrapassa `limite_bytes`.

    Diferente do log em texto, um JSON não pode ser cortado em qualquer ponto sem
    quebrar o parsing. Por isso a divisão é feita por ELEMENTOS da lista: cada parte
    `{nome_base}_parteNNN.json` é um array JSON VÁLIDO e independente, com um subconjunto
    dos itens. Para reconstruir a lista original, basta `json.load` de cada parte na
    ordem e concatenar as listas.

    - Se o JSON completo couber em `limite_bytes` (ou se `dados` não for uma lista),
      grava um único `{nome_base}.json` (retrocompatível).
    - Um item isolado maior que o limite fica sozinho em uma parte (não trava).

    Usa a mesma estratégia de agrupamento (`_agrupar_por_tamanho`) que
    `salvar_log_em_chunks`; aqui um "item" é um elemento da lista e cada parte é escrita
    como um array JSON válido. Observação: o limite é APROXIMADO, pois a soma dos tamanhos
    dos itens isolados não inclui a indentação/colchetes extras do array, então uma parte
    pode passar um pouco de `limite_bytes` (irrelevante para o alvo de 10 MiB).

    Retorna a lista de caminhos gravados.
    """
    conteudo = json.dumps(dados, ensure_ascii=False, indent=2)
    tamanho_total = len(conteudo.encode("utf-8"))

    # Caso comum (ou objeto não-lista, que não dá para fatiar por elemento): arquivo único.
    if tamanho_total <= limite_bytes or not isinstance(dados, list):
        caminho = os.path.join(pasta_execucao, f"{nome_base}.json")
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
        return [caminho]

    def _tamanho_item(item):
        return len(json.dumps(item, ensure_ascii=False, indent=2).encode("utf-8"))
    blocos = _agrupar_por_tamanho(dados, _tamanho_item, limite_bytes)

    caminhos = []
    for i, chunk in enumerate(blocos, start=1):
        caminho = os.path.join(pasta_execucao, f"{nome_base}_parte{i:03d}.json")
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        caminhos.append(caminho)
    return caminhos


# Log de todas as interações com a LLM (prompt, resposta bruta, parsing)
LLM_INTERACTIONS = []


def _model_slug(model_name: str) -> str:
    """Slug de nome de modelo para nomes de asset (fallback de ``MODEL_ALIAS``).

    Ex.: 'google/gemini-2.5-flash-lite' → 'google-gemini-2.5-flash-lite'.
    """
    return re.sub(r"[^a-zA-Z0-9.]+", "-", model_name).strip("-").lower()


# Aliases curtos por modelo para os nomes de asset (PNG/CSV). Todo modelo de
# MODELS_TO_TEST DEVE ter entrada aqui (teste garante); o fallback via
# _model_slug existe só para modelos fora do protocolo.
MODEL_ALIAS = {
    "openai/gpt-4o-mini":           "gpt4mini",   # id no OpenRouter (protocolo desde 01/10/2026)
    "gpt-4o-mini":                  "gpt4mini",   # id legado (OpenAI direto, execuções até 07/2026)
    "google/gemini-2.5-flash-lite": "flashlite",
}


def _model_alias(model_name: str) -> str:
    """Alias curto do modelo para nomes de asset; fallback = _model_slug."""
    return MODEL_ALIAS.get(model_name, _model_slug(model_name))


def llm_asset(pasta: str, filename: str, model_name: str) -> str:
    """Caminho de asset cujos dados derivam de UM LLM específico.

    Convenção de nomenclatura: injeta o ALIAS do modelo antes da extensão
    (ex.: 'bloco1_graphic_05_A_passo4_w_distribuicao.png' → 'bloco1_graphic_05_A_passo4_w_distribuicao__gpt4mini.png';
    o sufixo de cor ``_color``/``_gray`` entra depois, via ``com_cor``).
    Assets SEM dados de LLM (dados sintéticos, oráculos, especialista, baselines clássicos)
    e assets consolidados multi-modelo (CSVs com coluna model, llm_interactions,
    final_graphic_08_comparacao_modelos) NÃO usam esta função.
    """
    base, ext = os.path.splitext(filename)
    return os.path.join(pasta, f"{base}__{_model_alias(model_name)}{ext}")


# Sufixo de modelo no fim do stem: alias curto (MODEL_ALIAS) ou slug longo
# (_model_slug: [a-z0-9.-]). Só é reconhecido se vier após "__".
_ALIAS_SUFFIX_RE = re.compile(r"^(?P<stem>.+?)__(?P<alias>[a-z0-9][a-z0-9.-]*)$")


CORES_ASSET = ("color", "gray")


def com_cor(path: str, cor: str) -> str:
    """Sufixo de cor por ÚLTIMO (09/10/2026): 'p/x__gpt4mini.png', 'gray' → 'p/x__gpt4mini_gray.png'.

    Convenção dos assets gráficos e tabelas (09/10/2026):
      <bloco>_<graphic|table>_<NN>_<assunto>[_<problema>][_seed<S>][_<painel>][__<alias>]_<color|gray>.png
    Todo gráfico sai em `_color` e `_gray` (símbolos + − / * # @ no P&B); só as tabelas
    saem apenas em `_gray`. `.tex` de tabela: mesmo nome sem o campo de cor.
    """
    if cor not in CORES_ASSET:
        raise ValueError(f"cor inválida: {cor!r} (use {CORES_ASSET})")
    stem, ext = os.path.splitext(path)
    return f"{stem}_{cor}{ext}"


def asset_variant(path: str, suffix: str) -> str:
    """Caminho de uma VARIANTE (painel individual, cópia corrigida…) de um asset.

    Mantém a regra "alias sempre imediatamente antes da extensão": o sufixo da
    variante entra ANTES do ``__<alias>`` quando ele existe.

      'p/bloco1_graphic_05_A_passo4_w_distribuicao__gpt4mini.png', 'boxplot'
          → 'p/bloco1_graphic_05_A_passo4_w_distribuicao_boxplot__gpt4mini.png'
      'p/bloco1_graphic_19_oraculo_passos_1_a_3.png', 'ratio'        (asset sem alias)
          → 'p/bloco1_graphic_19_oraculo_passos_1_a_3_ratio.png'
      'p/final_cross_linearity__flashlite.csv', 'corrigido_hm'
          → 'p/final_cross_linearity_corrigido_hm__flashlite.csv'
    A cor (``_color``/``_gray``) é acrescentada por ``com_cor`` DEPOIS, no salvamento.
    """
    pasta, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    m = _ALIAS_SUFFIX_RE.match(stem)
    if m:
        new_stem = f"{m.group('stem')}_{suffix}__{m.group('alias')}"
    else:
        new_stem = f"{stem}_{suffix}"
    return os.path.join(pasta, new_stem + ext)


def checkpoint_interactions(pasta_execucao: str) -> None:
    """Salva um snapshot de LLM_INTERACTIONS (proteção contra crash em execução longa).

    Sobrescreve um único arquivo (escrita atômica via .tmp + os.replace); o salvamento
    oficial particionado acontece no fim da execução e remove este checkpoint. Se a
    execução morrer no meio, as interações coletadas até o último checkpoint ficam em
    llm_interactions_checkpoint.json (renomeie para llm_interactions.json para auditar
    com src/audit_interactions.py). Falha de escrita nunca derruba a execução.
    """
    if not LLM_INTERACTIONS:
        return
    path = os.path.join(pasta_execucao, "llm_interactions_checkpoint.json")
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(LLM_INTERACTIONS, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"  ⚠ Falha ao salvar checkpoint de interações: {exc}")
