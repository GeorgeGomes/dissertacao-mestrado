"""Versão preto e branco de uma figura matplotlib já desenhada (09/10/2026).

Toda figura do trabalho é salva duas vezes: ``_color`` (como desenhada) e ``_gray``.
Em vez de reescrever cada uma das ~45 funções de `plots.py`, `para_cinza(fig)` percorre os
artistas da figura pronta e troca cor por FORMA, de modo consistente em todos os painéis:

- pontos (scatter): cada cor vira um símbolo da sequência ``+ − / * # @`` (e ``x d`` para a
  7ª e a 8ª séries) em preto;
  um scatter colorido por classe (``c=y``) é dividido em um scatter por classe;
  scatter com cor contínua (confiança) vai para o mapa ``gray``;
- linhas: preto/cinzas com traço sólido, tracejado, pontilhado e os mesmos símbolos como
  marcadores; linhas de referência (``axhline``/``axvline`` ou 2 pontos sem marcador e sem
  rótulo) ficam pretas com o traço original; séries legítimas com 2 pontos (ex.: curvas do
  ``--rapido`` com k ∈ {0, 4}) recebem traço e símbolo como qualquer série;
- tampas e hastes de barras de erro (``errorbar``, ``bar(yerr=…, capsize=…)``) e linhas só de
  marcador sem rótulo ficam pretas, sem traço nem símbolo, e nunca entram na legenda; a linha
  de DADOS de um ``errorbar(fmt='o')`` não é tampa: recebe o símbolo da sua cor;
- linhas já pretas/cinzas mantêm marcador e traço originais (três séries pretas ``o-``,
  ``s--``, ``^:`` continuam distintas); a mesma cor usada com marcador/traço diferentes no
  mesmo eixo recebe símbolos diferentes (chave (cor, marcador, traço)); mas quando o eixo já usa
  o TRAÇO como codificação (a mesma cor e o mesmo marcador com traços diferentes, como em
  ``bloco23_graphic_01``: cheia = vs rótulo real, tracejada = vs especialista), o traço original é
  preservado e só o cinza e o símbolo variam com a cor (15/10/2026);
- uma linha marcada com o atributo ``_referencia = True`` (ex.: o arco ``||w|| = 1`` dos scatters
  de direção de Ŵ) é referência: fica preta com o traço original, sem símbolo;
- ``Line2D`` sem pontos (``ax.plot([], [], label=...)``) é referência de legenda: fica preta,
  mantém o traço e não ganha marcador; o contorno de linha da mesma cor recebe o mesmo traço,
  para a legenda bater com o desenho; proxies de marcador com linha branca ficam sem linha;
- caixas de ``boxplot`` sem ``patch_artist`` (linha fechada de 5 pontos) não são séries;
- barras, histogramas e caixas: face em tons de cinza com hachura (``//``, ``\\\\``,
  ``xx``, ``..``) por cor original, borda preta; uma hachura EXPLÍCITA da figura original é
  preservada, e a transparência (alpha) também, quando a mesma cor aparece com alphas
  diferentes (``final_graphic_08``: B sem hachura e alpha 0,8, C com ``//`` e alpha 0,5), para
  as barras continuarem distinguíveis (15/10/2026);
- mapas (contourf, heatmap, imagem): mapa de cores ``gray`` (ou ``gray_r`` quando o
  original ia do claro ao escuro, como ``Blues``), o que atualiza a barra de cores;
- textos coloridos ficam pretos (brancos sobre fundo escuro são mantidos);
- legendas: refeitas a partir das ENTRADAS da legenda original (mesmos rótulos, na mesma
  ordem), trocando cada entrada pelo artista já convertido; artistas que não estavam na
  legenda original não entram; um scatter rotulado dividido por classe vira uma entrada por
  classe; entradas de proxies (``Patch``/``Line2D`` avulsos) recebem a forma que a cor
  original ganhou.

A mesma cor original recebe sempre o mesmo símbolo/hachura dentro da figura, então
"Perceptron" (steelblue) e "NNLS" (coral) ficam iguais em todos os painéis. Os cinzas de
linhas e pontos nunca passam de 0.6, para que a 5ª ou 6ª cor de uma figura cheia continue
legível.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.contour import ContourSet
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# + − / * # @ e, para a 7ª e a 8ª séries (bloco1_graphic_05 com 7 pares de nomes), x e d (15/10/2026)
SIMBOLOS = ["+", "_", "$/$", "*", r"$\#$", "$@$", "x", "d"]
TRACOS = ["-", "--", ":", "-."]
# nunca acima de 0.6: a 6ª cor de uma figura cheia não pode virar um cinza quase invisível (F15)
CINZAS_LINHA = ["0.0", "0.45", "0.6", "0.25", "0.5", "0.35", "0.15", "0.55"]
# pares (face, hachura) distintos entre si: as 2 primeiras séries diferem pelo tom, a 3ª pela hachura…
FACES_AREA = [("0.85", ""), ("0.5", ""), ("1.0", "////"), ("0.25", ""), ("0.7", "xxxx"),
              ("0.4", "\\\\\\\\"), ("0.95", "...."), ("0.6", "++++")]
_MAX_GRUPOS_SCATTER = 6     # acima disso o scatter é tratado como cor contínua
_SEM_TRACO = ("None", "", " ")
_SEM_MARCADOR = ("", "None", " ", None)
_RE_CONTAGEM = re.compile(r"\((\d+)\)|(\d+)\s*pts")


def _lum(c) -> float:
    r, g, b, _ = to_rgba(c)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _chave(c) -> Tuple[float, float, float]:
    r, g, b, _ = to_rgba(c)
    return (round(r, 2), round(g, 2), round(b, 2))


def _invisivel(c) -> bool:
    """Cor que não aparece sobre fundo branco: 'none', transparente ou quase branca (lum > 0.97)."""
    if c is None or (isinstance(c, str) and c.lower() == "none"):
        return True
    try:
        r, g, b, a = to_rgba(c)
    except (ValueError, TypeError):
        return True
    return a == 0 or _lum((r, g, b, 1.0)) > 0.97


def _acromatica(c) -> bool:
    """Preto ou cinza visível (r ≈ g ≈ b): a figura original já não dependia da cor aqui."""
    if _invisivel(c):
        return False
    r, g, b, _ = to_rgba(c)
    return max(r, g, b) - min(r, g, b) < 0.05


def _forma(marker=None, ls=None) -> Tuple:
    """(marcador, traço) normalizados: ``None`` = ausente (vale como curinga na comparação)."""
    mk = None if marker in _SEM_MARCADOR else marker
    tr = None if ls in _SEM_TRACO or ls is None else ls
    return (mk, tr)


def _formas_compativeis(a: Tuple, b: Tuple) -> bool:
    return all(x is None or y is None or x == y for x, y in zip(a, b))


def _cmap_cinza(cmap):
    """'gray' ou 'gray_r': preserva o sentido claro→escuro do mapa original."""
    try:
        cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        return "gray_r" if _lum(cm(0.0)) > _lum(cm(1.0)) else "gray"
    except Exception:
        return "gray"


class _Mapa:
    """Cor original → (símbolo, cinza) para pontos/linhas e → (face, hachura) para áreas."""

    def __init__(self):
        self.sym: Dict[tuple, Tuple[str, str]] = {}
        self.forma: Dict[tuple, Tuple] = {}      # forma (marcador, traço) com que a cor foi vista primeiro
        self.face: Dict[tuple, Tuple[str, str]] = {}
        self._n_area = 0          # séries de área já atribuídas (exclui o branco original)
        self.n_centroides = 0     # marcadores grandes (centróides): preto, branco, preto…
        # cores (rgb) que aparecem em patches com alphas DIFERENTES no eixo corrente: o alpha
        # codifica uma série (final_graphic_08) e é preservado; preenchido por `_cinza_eixo`
        self.alphas_codificam: Set[tuple] = set()

    def _novo(self) -> Tuple[str, str]:
        i = len(self.sym)
        return (SIMBOLOS[i % len(SIMBOLOS)], CINZAS_LINHA[i % len(CINZAS_LINHA)])

    def simbolo(self, cor, marker=None, ls=None) -> Tuple[str, str]:
        """Símbolo e cinza da cor. A chave é a cor; quando a MESMA cor já apareceu na figura com
        outro marcador/traço (três séries pretas ``o-``, ``s--``, ``^:``), a chave passa a ser
        (cor, marcador, traço), para as séries não colapsarem no mesmo símbolo. Marcador ou traço
        ausentes (``None``) valem como curinga: um scatter ou um proxy só de marcador recebe o
        símbolo da cor, seja qual for o traço da série."""
        k = _chave(cor)
        f = _forma(marker, ls)
        if k not in self.sym:
            self.sym[k] = self._novo()
            self.forma[k] = f
            return self.sym[k]
        if _formas_compativeis(self.forma[k], f):
            return self.sym[k]
        k2 = (k, f)
        if k2 not in self.sym:
            self.sym[k2] = self._novo()
        return self.sym[k2]

    def area(self, cor, hatch: str = "") -> Tuple[str, str]:
        """Face e hachura da cor. A chave inclui a hachura explícita da figura original e, quando
        a cor aparece com alphas diferentes (`alphas_codificam`), o alpha: barras da mesma cor
        que a figura distinguia por hachura/alpha recebem entradas distintas. Uma hachura
        explícita é devolvida no lugar da hachura da sequência."""
        rgb = _chave(cor)
        alpha = round(float(to_rgba(cor)[3]), 2) if rgb in self.alphas_codificam else None
        k = (rgb, hatch or "", alpha)
        if k not in self.face:
            if _lum(cor) > 0.97 and to_rgba(cor)[3] > 0:      # já era branco: mantém sem hachura
                self.face[k] = ("1.0", hatch or "")
            else:
                face, hatch_seq = FACES_AREA[self._n_area % len(FACES_AREA)]
                self.face[k] = (face, hatch or hatch_seq)
                self._n_area += 1
        return self.face[k]


def _alphas_que_codificam(axes) -> Set[tuple]:
    """Cores (rgb) de patches que aparecem, nos eixos dados, com mais de um alpha: o alpha
    distingue séries (``final_graphic_08``: Problema B com alpha 0,8 e C com 0,5 na mesma cor).
    Avaliado por EIXO (um boxplot com alpha 0,7 noutro painel não muda a face das barras)."""
    vistos: Dict[tuple, Set[float]] = {}
    for ax in axes:
        for p in ax.patches:
            if not isinstance(p, Patch):
                continue
            fc = p.get_facecolor()
            a = float(to_rgba(fc)[3])
            if a == 0:
                continue
            vistos.setdefault(_chave(fc), set()).add(round(a, 2))
    return {k for k, s in vistos.items() if len(s) > 1}


def para_cinza(fig) -> None:
    """Converte a figura, no lugar, para preto e branco com símbolos/hachuras."""
    mapa = _Mapa()
    for ax in list(fig.axes):
        if getattr(ax, "_colorbar", None) is not None:      # eixo de barra de cores: segue o mappable
            continue
        _cinza_eixo(ax, mapa)
    mapa.alphas_codificam = _alphas_que_codificam(fig.axes)
    for leg in list(fig.legends):
        _cinza_legenda_proxies(leg, mapa)
    fig.patch.set_facecolor("white")


# ───────────────────────── classificação de linhas ─────────────────────────
def _sem_rotulo(art) -> bool:
    lab = art.get_label() if hasattr(art, "get_label") else ""
    return (not lab) or str(lab).startswith("_")


def _linhas_de_erro(ax) -> Tuple[Set[int], Set[int]]:
    """ids dos artistas que pertencem a um ``ErrorbarContainer`` do eixo.

    Devolve ``(todos, tampas)``: ``todos`` inclui a linha de dados (``lines[0]``), as tampas
    (``lines[1]``) e as hastes ``LineCollection`` (``lines[2]``); ``tampas`` só as duas últimas.
    ``bar(yerr=…)`` também cria um ``ErrorbarContainer`` (sem linha de dados)."""
    todos: Set[int] = set()
    tampas: Set[int] = set()
    for cont in ax.containers:
        if not isinstance(cont, ErrorbarContainer):
            continue
        linha_dados, caplines, hastes = cont.lines
        if linha_dados is not None:
            todos.add(id(linha_dados))
        for art in list(caplines) + list(hastes):
            todos.add(id(art))
            tampas.add(id(art))
    return todos, tampas


def _eh_tampa(ln: Line2D, tampas: Set[int], erro_dados: Set[int] = frozenset()) -> bool:
    """Tampa/haste de barra de erro, ou linha só de marcadores sem rótulo (fliers de boxplot):
    nunca recebe traço, símbolo nem rótulo. A linha de DADOS de um ``errorbar(fmt='o')``
    (``cont.lines[0]``, só marcador e com o rótulo no container) nunca é tampa: recebe o
    símbolo da sua cor como qualquer série só de marcadores."""
    if id(ln) in erro_dados:
        return False
    return id(ln) in tampas or (ln.get_linestyle() in _SEM_TRACO and _sem_rotulo(ln))


def _eh_caixa(ln: Line2D) -> bool:
    """Caixa de ``boxplot`` sem ``patch_artist``: linha fechada (primeiro ponto = último) de
    5 pontos, sem marcador e sem rótulo. Não é série: fica preta com o traço original."""
    if ln.get_marker() not in _SEM_MARCADOR or not _sem_rotulo(ln):
        return False
    x = np.atleast_1d(ln.get_xdata()); y = np.atleast_1d(ln.get_ydata())
    if len(x) != 5 or len(y) != 5:
        return False
    try:
        return bool(np.isclose(float(x[0]), float(x[-1])) and np.isclose(float(y[0]), float(y[-1])))
    except (TypeError, ValueError):
        return False


def _eh_proxy(ln: Line2D) -> bool:
    """``Line2D`` sem pontos (``ax.plot([], [], label=...)``): só existe para a legenda."""
    return len(np.atleast_1d(ln.get_xdata())) == 0


def _eh_referencia(ax, ln: Line2D, erro_todos: Set[int]) -> bool:
    """Linha de referência: ``axhline``/``axvline`` (transformação do eixo), linha marcada
    pelo autor com ``_referencia = True`` (arco ``||w|| = 1`` de `plots._eixos_w_unitario`), ou
    linha de 2 pontos sem marcador e sem rótulo (bigodes, medianas, segmentos desenhados à mão).
    Uma linha de dados de ``errorbar`` nunca é referência por contagem de pontos."""
    if getattr(ln, "_referencia", False):
        return True
    tr = ln.get_transform()
    try:
        if tr == ax.get_yaxis_transform(which="grid") or tr == ax.get_xaxis_transform(which="grid"):
            return True
    except Exception:
        pass
    if id(ln) in erro_todos:
        return False
    n = len(np.atleast_1d(ln.get_xdata()))
    return n <= 2 and ln.get_marker() in _SEM_MARCADOR and _sem_rotulo(ln)


def _eh_serie(ax, ln: Line2D, erro_todos: Set[int], tampas: Set[int]) -> bool:
    """Série de dados que merece legenda: não é tampa, não é referência, não é caixa de boxplot,
    não é proxy (0 pontos), não pertence a um ``ErrorbarContainer`` (cujo rótulo está no
    container) e tem mais de 2 pontos, ou 2 pontos com marcador ou rótulo."""
    if id(ln) in erro_todos or _eh_tampa(ln, tampas, erro_todos - tampas) or _eh_referencia(ax, ln, erro_todos):
        return False
    if _eh_caixa(ln) or _eh_proxy(ln):
        return False
    n = len(np.atleast_1d(ln.get_xdata()))
    return n > 2 or (n == 2 and (ln.get_marker() not in _SEM_MARCADOR or not _sem_rotulo(ln)))


# ───────────────────────── legendas garantidas ─────────────────────────
def garantir_legendas(fig) -> list:
    """Garante que todo eixo com séries tenha legenda (09/10/2026, pedido do usuário).

    - scatter colorido por classe (``c=y``) sem rótulo → dividido em um scatter por classe,
      rotulado ``classe <v>`` (mesmas cores);
    - scatters de cor única sem rótulo (≥ 4 pontos) → ``amostras (1)``, ``amostras (2)``, …
      (nome neutro: só o ``c=y`` numérico fala em classe; F5);
    - linhas (séries, ver `_eh_serie`), ``errorbar`` (pelo container) e barras sem rótulo →
      ``série k`` (ou o rótulo do eixo y, se única); barra única sem rótulo e sem rótulo do
      eixo y fica sem legenda;
    - tampas/hastes de barras de erro e linhas de ``ErrorbarContainer`` nunca são rotuladas (B2);
    - eixo que JÁ tem legenda ou artistas rotulados não ganha ``série k``: os artistas sem rótulo
      são propositais (padrão ``label=... if i == 0 else ''`` de `plots.py`; B2);
    - eixo com artistas rotulados e sem legenda → ``ax.legend()``.
    Devolve a lista de eixos que ainda ficaram sem legenda apesar de terem séries
    (mapas de calor/contorno puros não contam: a barra de cores é a legenda)."""
    pendentes = []
    if getattr(fig, "legends", None):
        # 14/10/2026: a figura já tem legenda de figura (fig.legend, ex.: bloco1_graphic_09 com 36
        # séries); não criar legendas por eixo, que cobririam as curvas.
        return pendentes
    for ax in list(fig.axes):
        if getattr(ax, "_colorbar", None) is not None:
            continue
        ja_rotulado = ax.get_legend() is not None or bool(ax.get_legend_handles_labels()[0])
        _rotular_scatters(ax)
        _rotular_linhas_e_barras(ax, ja_rotulado)
        handles, labels = ax.get_legend_handles_labels()
        if handles and ax.get_legend() is None:
            ax.legend(fontsize=8, framealpha=0.9, loc="best")
        if tem_series(ax) and ax.get_legend() is None:
            pendentes.append(ax.get_title() or ax.get_ylabel() or "(eixo sem título)")
    return pendentes


def tem_series(ax) -> bool:
    """O eixo tem séries que pedem legenda: linhas de dados (fora tampas, referências e linhas
    de barras de erro), scatters com ≥ 4 pontos, 2+ séries de barras ou um ``errorbar`` com
    linha de dados. Uma única série de barras não conta: sem rótulo do eixo y não há o que
    legendar."""
    erro_todos, tampas = _linhas_de_erro(ax)
    n_barras = sum(isinstance(c, BarContainer) for c in ax.containers)
    tem_errorbar = any(isinstance(c, ErrorbarContainer) and c.lines[0] is not None for c in ax.containers)
    return (any(_eh_serie(ax, l, erro_todos, tampas) for l in ax.lines)
            or any(isinstance(c, PathCollection) and len(c.get_offsets()) > 3 for c in ax.collections)
            or n_barras >= 2 or tem_errorbar)


def _rotular_scatters(ax) -> None:
    simples, centroides = [], []
    for coll in list(ax.collections):
        if not isinstance(coll, PathCollection) or not _sem_rotulo(coll):
            continue
        xy = np.asarray(coll.get_offsets()); n = len(xy)
        arr = coll.get_array()
        if arr is not None and len(arr) == n and 1 < len(np.unique(np.asarray(arr))) <= _MAX_GRUPOS_SCATTER:
            vals = np.asarray(arr); sizes = coll.get_sizes(); alpha = coll.get_alpha(); z = coll.get_zorder()
            for v in np.unique(vals):
                m = vals == v
                ax.scatter(xy[m, 0], xy[m, 1], color=coll.cmap(coll.norm(v)), s=(sizes[m] if len(sizes) == n else sizes),
                           alpha=alpha, zorder=z, label=f"classe {int(v) if float(v).is_integer() else v}")
            coll.remove()
        elif n >= 4:
            simples.append(coll)
        elif n >= 1 and len(coll.get_sizes()) and float(np.max(coll.get_sizes())) >= 100:
            centroides.append(coll)                       # marcador grande: centróide
    for i, c in enumerate(simples):
        arr = c.get_array()
        continua = arr is not None and len(arr) == len(c.get_offsets()) and len(np.unique(np.asarray(arr))) > _MAX_GRUPOS_SCATTER
        c.set_label("amostras (cor = valor da barra)" if continua else f"amostras ({i + 1})")
    for i, c in enumerate(centroides):
        c.set_label(f"centróide {i}" if len(centroides) > 1 else "centróide")
    # fronteira de decisão (contour de linha) sem rótulo → proxy na legenda, com o traço do contorno
    for coll in ax.collections:
        if isinstance(coll, ContourSet) and not coll.filled and _sem_rotulo(coll):
            ec = coll.get_edgecolor()
            cor = ec[0] if len(ec) else "black"
            ax.plot([], [], color=cor, linestyle=_traco_contorno(coll),
                    linewidth=1.5, label="fronteira de decisão")
            coll.set_label("_fronteira_rotulada")


def _traco_contorno(coll: ContourSet) -> str:
    """Traço de um contorno de linha como string (``'-'``, ``'--'`` ou ``':'``), lido do padrão
    de tracejado escalado que o ``ContourSet`` guarda."""
    try:
        ls = coll.get_linestyle()
        padrao = ls[0] if isinstance(ls, (list, tuple)) and ls else None
        if padrao is None or not isinstance(padrao, (tuple, list)) or padrao[1] is None:
            return "-"
        tracos = list(padrao[1])
        lw = coll.get_linewidth()
        lw = float(lw[0]) if np.ndim(lw) else float(lw or 1.0)
        # padrão pontilhado do matplotlib: traço ≈ 1 × largura; tracejado: ≈ 3,7 × largura
        return ":" if tracos and float(tracos[0]) <= 1.6 * max(lw, 0.1) else "--"
    except Exception:
        return "-"


def _rotular_linhas_e_barras(ax, ja_rotulado: bool = False) -> None:
    """Rotula linhas (séries) e barras sem rótulo com ``série k`` ou o rótulo do eixo y.

    ``ja_rotulado=True`` (eixo com legenda ou artistas rotulados antes desta passagem) deixa
    tudo como está: no matplotlib ``label=''`` é indistinguível de "sem rótulo", e o padrão
    ``label=... if i == 0 else ''`` de `plots.py` usa isso de propósito para uma entrada por
    série (B2). Tampas, hastes e linhas de ``ErrorbarContainer`` nunca são rotuladas."""
    if ja_rotulado:
        return
    erro_todos, tampas = _linhas_de_erro(ax)
    linhas = [l for l in ax.lines if _eh_serie(ax, l, erro_todos, tampas) and _sem_rotulo(l)]
    # errorbar sem rótulo: o rótulo vai para o CONTAINER (a linha de dados fica _nolegend_)
    linhas += [c for c in ax.containers
               if isinstance(c, ErrorbarContainer) and c.lines[0] is not None and _sem_rotulo(c)]
    if len(linhas) == 1:
        linhas[0].set_label(ax.get_ylabel() or "série")
    else:
        for i, l in enumerate(linhas):
            l.set_label(f"série {i + 1}")
    barras = [c for c in ax.containers if isinstance(c, BarContainer) and _sem_rotulo(c)]
    if len(barras) == 1:
        if ax.get_ylabel():                 # sem rótulo do eixo y não há legenda útil (F15c)
            barras[0].set_label(ax.get_ylabel())
    else:
        for i, c in enumerate(barras):
            c.set_label(f"série {i + 1}")


# ───────────────────────── por eixo ─────────────────────────
def _cinza_eixo(ax, mapa: _Mapa) -> None:
    _casar_contornos_com_proxies(ax)        # ANTES de enegrecer os contornos: casa pela cor original
    for im in ax.images:
        im.set_cmap(_cmap_cinza(im.get_cmap()))
    for coll in list(ax.collections):
        if isinstance(coll, ContourSet):
            if coll.filled:
                coll.set_cmap(_cmap_cinza(coll.get_cmap()))
            else:
                coll.set_edgecolor("black")
        elif isinstance(coll, QuadMesh):
            coll.set_cmap(_cmap_cinza(coll.get_cmap()))
        elif isinstance(coll, PathCollection):
            _cinza_scatter(ax, coll, mapa)
        elif isinstance(coll, PolyCollection):          # fill_between, violinos
            fc = coll.get_facecolor()
            face, hatch = mapa.area(fc[0] if len(fc) else "0.5")
            coll.set_facecolor(face); coll.set_edgecolor("black"); coll.set_hatch(hatch)
        elif isinstance(coll, LineCollection):          # barras de erro, bigodes
            coll.set_color("black")
    mapa.alphas_codificam = _alphas_que_codificam([ax])
    _cinza_patches(ax, mapa)
    erro_todos, tampas = _linhas_de_erro(ax)
    preservar_traco = _traco_codifica(ax, erro_todos, tampas)
    for ln in ax.lines:
        _cinza_linha(ax, ln, mapa, erro_todos, tampas, preservar_traco)
    for t in ax.texts:
        if _lum(t.get_color()) < 0.9:
            t.set_color("black")
    _cinza_legenda(ax, mapa)


def _rotulo_por_classe(label: str, n_total: int, n_classe: int) -> str:
    """Ajusta a contagem de um rótulo dividido por classe: ``Concordam (113)`` com 60 pontos
    na classe vira ``Concordam (60)``; se o número não for o total do scatter, é removido."""
    def _troca(m):
        n = int(m.group(1) or m.group(2))
        if n != n_total:
            return ""
        return f"({n_classe})" if m.group(1) else f"{n_classe} pts"
    return re.sub(r"\s{2,}", " ", _RE_CONTAGEM.sub(_troca, label)).strip()


def _cinza_scatter(ax, coll: PathCollection, mapa: _Mapa) -> None:
    xy = np.asarray(coll.get_offsets())
    n = len(xy)
    if n == 0:
        return
    arr = coll.get_array()
    fc = coll.get_facecolor()
    ec = coll.get_edgecolor()
    sizes = coll.get_sizes()
    alpha = coll.get_alpha()
    z = coll.get_zorder()
    label = coll.get_label() or ""
    rotulado = bool(label) and not label.startswith("_")
    # cor contínua (ex.: confiança) → mapa cinza
    if arr is not None and len(arr) == n and len(np.unique(np.asarray(arr))) > _MAX_GRUPOS_SCATTER:
        coll.set_cmap(_cmap_cinza(coll.get_cmap()))
        return
    # grupos: por valor de `c` ou por cor de face
    if arr is not None and len(arr) == n:
        chaves = np.asarray(arr)
        grupos = [(v, chaves == v) for v in np.unique(chaves)]
        cores = {v: coll.cmap(coll.norm(v)) for v, _ in grupos}
    elif len(fc) == n and n > 1 and len({_chave(c) for c in fc}) > 1:
        ks = [_chave(c) for c in fc]
        unicos = []
        for k in ks:
            if k not in unicos:
                unicos.append(k)
        grupos = [(k, np.array([kk == k for kk in ks])) for k in unicos]
        cores = {k: fc[ks.index(k)] for k in unicos}
    else:
        grupos = None
    if grupos is not None and 1 < len(grupos) <= _MAX_GRUPOS_SCATTER:
        for j, (v, m) in enumerate(grupos):
            sym, cinza = mapa.simbolo(cores[v])
            s = sizes[m] if len(sizes) == n else sizes
            if arr is not None and len(arr) == n:            # grupo = valor de classe (c=y)
                cls = f"classe {int(v) if float(v).is_integer() else v}"
                if not rotulado or "classe" in label.lower():
                    lab = cls
                else:                                        # só acrescenta ", classe k" (F15b)
                    lab = f"{_rotulo_por_classe(label, n, int(m.sum()))}, {cls}"
            elif not rotulado:
                lab = "_nolegend_"
            else:
                lab = f"{label} ({j})"
            novo = ax.scatter(xy[m, 0], xy[m, 1], marker=sym, s=s, color=cinza, alpha=alpha, zorder=z,
                              linewidths=1.3, label=lab)
            if rotulado:
                novo._rotulo_origem = label                  # entrada derivada da original (legenda)
        coll.remove()
        return
    # cor única: poucos pontos grandes = marcador especial (centróide), mantém a forma
    cor = fc[0] if len(fc) else (ec[0] if len(ec) else "black")
    if n <= 3 and len(sizes) and float(np.max(sizes)) >= 100:
        coll.set_facecolor("black" if mapa.n_centroides % 2 == 0 else "white")
        coll.set_edgecolor("black"); coll.set_linewidths(1.6)
        mapa.n_centroides += 1
        return
    sym, cinza = mapa.simbolo(cor)
    ax.scatter(xy[:, 0], xy[:, 1], marker=sym, s=sizes, color=cinza, alpha=alpha, zorder=z,
               linewidths=1.3, label=label if rotulado else "_nolegend_")
    coll.remove()


def _cinza_patches(ax, mapa: _Mapa) -> None:
    em_barra = set()
    for cont in ax.containers:
        if isinstance(cont, BarContainer):
            for p in cont.patches:
                em_barra.add(id(p))
                _cinza_patch(p, mapa)
    for p in ax.patches:
        if id(p) in em_barra or not isinstance(p, Patch):
            continue
        _cinza_patch(p, mapa)


def _cinza_patch(p: Patch, mapa: _Mapa) -> None:
    """Face cinza + hachura por cor original. Uma hachura EXPLÍCITA do patch é preservada e o
    alpha só é mantido quando a mesma cor aparece com alphas diferentes na figura (o alpha
    distingue séries); nos demais casos o alpha vai a 1,0 (tons de cinza como projetados)."""
    fc = p.get_facecolor()
    if to_rgba(fc)[3] == 0:            # sem preenchimento: só a borda
        p.set_edgecolor("black")
        return
    hatch_orig = p.get_hatch() or ""
    manter_alpha = _chave(fc) in mapa.alphas_codificam
    alpha = float(to_rgba(fc)[3])
    face, hatch = mapa.area(fc, hatch_orig)
    p.set_facecolor(face); p.set_edgecolor("black"); p.set_hatch(hatch)
    p.set_alpha(alpha if manter_alpha else 1.0)


def _casar_contornos_com_proxies(ax) -> None:
    """Dá a cada contorno de linha o traço do proxy de legenda da mesma cor (``ax.plot([], [],
    color=..., linestyle=..., label='Fronteira ótima')``), para a legenda bater com o desenho
    depois da conversão. Sem proxy da mesma cor o contorno mantém o traço que tem."""
    proxies = [ln for ln in ax.lines if _eh_proxy(ln) and not _sem_rotulo(ln)
               and ln.get_marker() in _SEM_MARCADOR]
    usados: Set[int] = set()
    for coll in ax.collections:
        if not isinstance(coll, ContourSet) or coll.filled:
            continue
        ec = coll.get_edgecolor()
        k = _chave(ec[0]) if len(ec) else None
        for p in proxies:
            if id(p) in usados or _chave(p.get_color()) != k:
                continue
            ls = p.get_linestyle()
            if ls not in _SEM_TRACO:
                coll.set_linestyle(ls)
            usados.add(id(p))
            break


def _traco_codifica(ax, erro_todos: Set[int], tampas: Set[int]) -> bool:
    """O eixo usa o TRAÇO como codificação: alguma (cor, marcador) de série colorida aparece com
    dois traços distintos (``bloco23_graphic_01``: cheia = vs rótulo real, tracejada = vs
    especialista, na mesma cor e marcador por dimensão). Nesse caso a conversão preserva o
    traço e varia só o cinza e o símbolo com a cor (15/10/2026)."""
    erro_dados = set(erro_todos) - set(tampas)
    tracos: Dict[tuple, Set[str]] = {}
    for ln in ax.lines:
        if _eh_proxy(ln) or _eh_tampa(ln, tampas, erro_dados) or _eh_referencia(ax, ln, erro_todos) or _eh_caixa(ln):
            continue
        if id(ln) not in erro_dados and not _eh_serie(ax, ln, erro_todos, tampas):
            continue
        ls = ln.get_linestyle()
        if ls in _SEM_TRACO or _acromatica(ln.get_color()) or _invisivel(ln.get_color()):
            continue
        tracos.setdefault((_chave(ln.get_color()), _forma(ln.get_marker())[0]), set()).add(str(ls))
    return any(len(s) > 1 for s in tracos.values())


def _cinza_linha(ax, ln: Line2D, mapa: _Mapa, erro_todos: Set[int] = frozenset(),
                 tampas: Set[int] = frozenset(), preservar_traco: bool = False) -> None:
    """Converte uma linha:
    - proxy de legenda (0 pontos): sem marcador vira referência preta com o traço original
      (fronteiras); com marcador recebe o símbolo da sua cor (ver `_cinza_proxy_linha`);
    - tampas/hastes, referências e caixas de boxplot ficam pretas com a forma original;
    - linha já preta/cinza mantém marcador e traço (a figura original não dependia da cor);
      só ganha símbolo se tiver traço e nenhum marcador;
    - séries coloridas (mesmo com 2 pontos) recebem cinza, traço e símbolo da sua cor; uma
      série só de marcadores (``plot(x, y, 'o')``, linha de dados de ``errorbar(fmt='o')``)
      recebe o símbolo e mantém-se sem traço;
    - ``preservar_traco=True`` (ver `_traco_codifica`): o traço original fica e o símbolo é o da
      cor + marcador, seja qual for o traço."""
    erro_dados = set(erro_todos) - set(tampas)
    if _eh_proxy(ln):
        _cinza_proxy_linha(ln, mapa)
        return
    if _eh_tampa(ln, tampas, erro_dados) or _eh_referencia(ax, ln, erro_todos) or _eh_caixa(ln):
        # tampas são só marcador: a cor visível é a do marcador, não a da linha
        ln.set_color("black"); ln.set_markeredgecolor("black"); ln.set_markerfacecolor("black")
        return
    n = len(np.atleast_1d(ln.get_xdata()))
    cor = ln.get_color()
    if _acromatica(cor):
        if ln.get_marker() in _SEM_MARCADOR and ln.get_linestyle() not in _SEM_TRACO and n > 2:
            sym, _ = mapa.simbolo(cor, None, ln.get_linestyle())
            ln.set_marker(sym); ln.set_markersize(7); ln.set_markeredgewidth(1.3)
            ln.set_markerfacecolor(cor); ln.set_markeredgecolor(cor)
            if n > 40:
                ln.set_markevery(max(1, n // 12))
        return
    if preservar_traco:
        sym, cinza = mapa.simbolo(cor, ln.get_marker(), None)     # traço = curinga: símbolo pela cor + marcador
    else:
        sym, cinza = mapa.simbolo(cor, ln.get_marker(), ln.get_linestyle())
    i = SIMBOLOS.index(sym)
    ln.set_color(cinza)
    if ln.get_linestyle() not in _SEM_TRACO and not preservar_traco:
        ln.set_linestyle(TRACOS[i % len(TRACOS)])
    ln.set_marker(sym); ln.set_markersize(7); ln.set_markeredgewidth(1.3)
    ln.set_markerfacecolor(cinza); ln.set_markeredgecolor(cinza)
    if n > 40:
        ln.set_markevery(max(1, n // 12))


def _cinza_proxy_linha(h: Line2D, mapa: _Mapa) -> None:
    """Converte um ``Line2D`` que só existe para a legenda (proxy avulso ou ``plot([], [])``).

    - Linha de cor invisível (branca/'none', só marcador): fica SEM linha e só o marcador é
      convertido pelo símbolo da cor da face do marcador;
    - sem marcador: é referência (fronteira), fica preta com o traço original;
    - com marcador e cor visível: símbolo e cinza da cor (traço da mesma sequência);
    - marcador de face já preta/cinza mantém a forma original."""
    tem_marcador = h.get_marker() not in _SEM_MARCADOR
    cor_linha = h.get_color()
    if _invisivel(cor_linha):
        h.set_linestyle("None")
        if not tem_marcador:
            return
        mfc = h.get_markerfacecolor()
        if _invisivel(mfc):
            mfc = h.get_markeredgecolor()
        if _acromatica(mfc):
            h.set_markerfacecolor(mfc); h.set_markeredgecolor("black")
            return
        sym, cinza = mapa.simbolo(mfc, h.get_marker(), None)
        h.set_marker(sym); h.set_markerfacecolor(cinza); h.set_markeredgecolor(cinza)
        h.set_markeredgewidth(1.3)
        return
    if not tem_marcador:
        h.set_color("black")                    # referência: mantém o traço original
        return
    mfc = h.get_markerfacecolor()
    if _acromatica(cor_linha) and (_acromatica(mfc) or _invisivel(mfc)):
        h.set_color("black"); h.set_markeredgecolor("black")   # já era preto: mantém a forma
        if not _invisivel(mfc):
            h.set_markerfacecolor("black")
        return
    base = mfc if not _invisivel(mfc) and not _acromatica(mfc) else cor_linha
    ls = h.get_linestyle() if len(np.atleast_1d(h.get_xdata())) > 1 else None
    sym, cinza = mapa.simbolo(base, h.get_marker(), ls)
    h.set_color(cinza); h.set_markerfacecolor(cinza); h.set_markeredgecolor(cinza)
    h.set_marker(sym); h.set_markeredgewidth(1.3)
    if ls is not None:
        h.set_linestyle(TRACOS[SIMBOLOS.index(sym) % len(TRACOS)])


# ───────────────────────── legendas ─────────────────────────
def _props_legenda(leg) -> dict:
    props = {"loc": getattr(leg, "_loc", "best"), "fontsize": leg.prop.get_size()}
    try:
        props["ncols"] = leg._ncols
    except AttributeError:
        pass
    # legenda ancorada fora do eixo (bbox_to_anchor): preserva âncora e transformação
    anc = getattr(leg, "_bbox_to_anchor", None)
    try:
        if anc is not None and hasattr(anc, "_bbox") and hasattr(anc, "_transform"):
            b = anc._bbox
            props["bbox_to_anchor"] = (b.x0, b.y0, b.width, b.height)
            props["bbox_transform"] = anc._transform
    except Exception:
        pass
    t = leg.get_title().get_text()
    if t:
        props["title"] = t
    return props


def _handles_legenda(leg) -> List:
    hs = getattr(leg, "legend_handles", None)
    if hs is None:
        hs = getattr(leg, "legendHandles", [])
    return list(hs)


def _cinza_legenda(ax, mapa: _Mapa) -> None:
    """Refaz a legenda do eixo a partir das ENTRADAS da legenda original (B2).

    Para cada rótulo da legenda original, na ordem: usa os artistas atuais do eixo com esse
    rótulo (já convertidos), ou os artistas derivados dele (scatter rotulado dividido por
    classe, marcados com ``_rotulo_origem``), ou, se nenhum artista do eixo corresponder, o
    próprio handle da legenda original (proxy) convertido pela cor. Artistas sem rótulo, ou
    rotulados mas fora da legenda original, não entram."""
    leg = ax.get_legend()
    if leg is None:
        return
    textos = [t.get_text() for t in leg.get_texts()]
    handles_orig = _handles_legenda(leg)
    hs, ls = ax.get_legend_handles_labels()
    atuais = list(zip(hs, ls))
    usados: Set[int] = set()
    novos_h, novos_l = [], []
    for i, txt in enumerate(textos):
        derivados = [(h, l) for h, l in atuais if id(h) not in usados and getattr(h, "_rotulo_origem", None) == txt]
        diretos = [(h, l) for h, l in atuais if id(h) not in usados and l == txt]
        if derivados:
            escolhidos = derivados
        elif diretos:
            escolhidos = diretos[:1]            # uma entrada original = um artista
        else:
            h = handles_orig[i] if i < len(handles_orig) else None
            if h is None:
                continue
            _cinza_proxy(h, mapa)
            escolhidos = [(h, txt)]
        for h, l in escolhidos:
            usados.add(id(h)); novos_h.append(h); novos_l.append(l)
    if novos_h:
        ax.legend(novos_h, novos_l, **_props_legenda(leg))
    else:
        _cinza_legenda_proxies(leg, mapa)


def _cinza_proxy(h, mapa: _Mapa) -> None:
    """Dá a um handle avulso de legenda (proxy) a forma que a sua cor original recebeu."""
    if isinstance(h, Line2D):
        _cinza_proxy_linha(h, mapa)
    elif isinstance(h, Patch):
        _cinza_patch(h, mapa)
    elif isinstance(h, PathCollection):
        fc = h.get_facecolor()
        sym, cinza = mapa.simbolo(fc[0] if len(fc) else "black")
        h.set_facecolor(cinza); h.set_edgecolor(cinza)


def _cinza_legenda_proxies(leg, mapa: _Mapa) -> None:
    """Legenda de figura ou só de proxies: converte cada handle no lugar."""
    for h in _handles_legenda(leg):
        _cinza_proxy(h, mapa)
    for t in leg.get_texts():
        t.set_color("black")
