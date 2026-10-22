"""Motor das tabelas do trabalho: uma `Tabela` vira LaTeX (booktabs) e PNG.

Cada tabela é descrita uma única vez (cabeçalhos, linhas já formatadas, grupos de
colunas, destaque, rodapé) e renderizada em dois formatos:

- ``.tex``  — tabular booktabs (``\\toprule``/``\\midrule``/``\\bottomrule``), com
  ``\\multicolumn`` + ``\\cmidrule`` para os grupos de colunas, pronto para o texto;
- ``.png``  — a mesma tabela desenhada com matplotlib (estilo booktabs: filetes
  horizontais, zebra leve, linha eleita em negrito), para apresentação e conferência.

Os textos das células podem conter fragmentos LaTeX simples (``$\\kappa$``, ``R$^2$``,
``\\textbf{}``, ``\\%``); `latex_para_texto` converte para Unicode no PNG.

Convenção de nomes (`Tabela.nome`, 09/10/2026):
``<bloco>_table_<NN>_<assunto>[__<alias>]_gray.png`` e o ``.tex`` com o mesmo nome sem a cor.
O slug é ``<bloco>_<assunto>`` (ex.: ``bloco1_A_passos_2_a_5``) e ``nn`` é a ordem da
tabela dentro do bloco; tabelas saem só em ``_gray`` (não têm cor de conteúdo).
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # alias do modelo (execucao_io está no mesmo diretório)
    from execucao_io import _model_alias
except Exception:  # pragma: no cover — uso isolado do módulo
    def _model_alias(model: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", model.split("/")[-1]).lower()


# ───────────────────────── LaTeX → texto ─────────────────────────
_SUBST = [
    (r"\\textbf\{([^}]*)\}", r"\1"), (r"\\texttt\{([^}]*)\}", r"\1"), (r"\\emph\{([^}]*)\}", r"\1"),
    (r"\$\\hat\{W\}\$", "Ŵ"), (r"\\hat\{W\}", "Ŵ"),
    (r"R\$\^2\$", "R²"), (r"R\$\^3\$", "R³"), (r"R\$\^4\$", "R⁴"),
    (r"\$\^2\$", "²"), (r"\$\^3\$", "³"), (r"\$\^4\$", "⁴"),
    (r"\$\\kappa\$", "κ"), (r"\\kappa", "κ"), (r"\$\\Delta\$", "Δ"), (r"\\Delta", "Δ"),
    (r"\$\\geq\$", "≥"), (r"\$\\leq\$", "≤"), (r"\$<\$", "<"), (r"\$>\$", ">"), (r"\$-\$", "−"),
    (r"\$\\times\$", "×"), (r"\\times", "×"), (r"\\sigma", "σ"), (r"\\gamma", "γ"),
    (r"\$150-k\$", "150−k"), (r"``", "“"), (r"''", "”"),
    (r"\\%", "%"), (r"\\_", "_"), (r"\\&", "&"), (r"\\#", "#"), (r"\\ ", " "), (r"\\,", " "),
    (r"\\footnotesize", ""), (r"\\small", ""), (r"\\\\", ""), (r"\$", ""),
]
_SUBST_RE = [(re.compile(p), r) for p, r in _SUBST]


def latex_para_texto(s: str) -> str:
    """Converte os fragmentos LaTeX usados nas tabelas para texto Unicode."""
    t = str(s)
    for rx, rep in _SUBST_RE:
        t = rx.sub(rep, t)
    return re.sub(r"\s+", " ", t).strip()


def esc(s: str) -> str:
    """Escapa `_`, `%` e `&` para LaTeX (texto cru vindo de CSV/modelos)."""
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ───────────────────────── a tabela ─────────────────────────
@dataclass
class Tabela:
    slug: str
    caption: str
    headers: Sequence[str]
    rows: List[Sequence[str]]
    align: str
    label: Optional[str] = None
    grupos: Sequence[Tuple[str, int, int]] = ()      # (título, col_inicial, col_final), 1-based, inclusivo
    destaque: Set[int] = field(default_factory=set)   # índices (0-based) das linhas em negrito
    midrules_apos: Set[int] = field(default_factory=set)
    rodape: Optional[str] = None
    tamanho: str = r"\small"
    modelo: Optional[str] = None                      # id do modelo → alias no nome do arquivo
    nn: int = 0                                       # ordem da tabela dentro do bloco (01, 02, …)

    # ---- nomes ----
    def alias(self) -> Optional[str]:
        return _model_alias(self.modelo) if self.modelo else None

    def bloco(self) -> str:
        return self.slug.split("_", 1)[0]

    def assunto(self) -> str:
        return self.slug.split("_", 1)[1] if "_" in self.slug else self.slug

    def nome(self, ext: str, cor: Optional[str] = None) -> str:
        """``<bloco>_table_<NN>_<assunto>[__<alias>][_<cor>].<ext>`` (convenção de 09/10/2026)."""
        base = f"{self.bloco()}_table_{self.nn:02d}_{self.assunto()}"
        al = self.alias()
        if al:
            base += f"__{al}"
        if cor:
            base += f"_{cor}"
        return f"{base}.{ext}"

    def nome_tex(self) -> str:
        return self.nome("tex")

    def nome_png(self) -> str:
        return self.nome("png", "gray")      # tabela não tem cor de conteúdo: só a versão _gray

    # ---- LaTeX ----
    def cabecalho_grupos(self) -> Optional[str]:
        if not self.grupos:
            return None
        ncol = len(self.headers)
        cells, cmid, col = [], [], 1
        grupos = sorted(self.grupos, key=lambda g: g[1])
        for titulo, a, b in grupos:
            while col < a:
                cells.append("")
                col += 1
            cells.append(rf"\multicolumn{{{b - a + 1}}}{{c}}{{{titulo}}}")
            cmid.append(rf"\cmidrule(lr){{{a}-{b}}}")
            col = b + 1
        while col <= ncol:
            cells.append("")
            col += 1
        return " & ".join(cells) + r" \\ " + " ".join(cmid)

    def tex(self) -> str:
        return booktabs(self.headers, self.rows, self.align, self.caption,
                        self.label or f"tab:{self.slug}" + (f"_{self.alias()}" if self.alias() else ""),
                        destaque=self.destaque, cabecalho_extra=self.cabecalho_grupos(),
                        rodape=self.rodape, midrules_apos=self.midrules_apos, tamanho=self.tamanho)

    # ---- PNG ----
    def png(self, caminho: Path, dpi: int = 160) -> Path:
        return renderizar_png(self, Path(caminho), dpi=dpi)


def booktabs(headers: Sequence[str], rows: Iterable[Sequence[str]], align: str,
             caption: str, label: str, destaque: Iterable[int] = (),
             cabecalho_extra: Optional[str] = None, rodape: Optional[str] = None,
             midrules_apos: Iterable[int] = (), tamanho: str = r"\small") -> str:
    """Emissor booktabs (a venv não tem jinja2, exigido por DataFrame.to_latex)."""
    destaque, midrules_apos = set(destaque), set(midrules_apos)
    L = [r"\begin{table}[htbp]", r"\centering", tamanho,
         rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    if cabecalho_extra:
        L.append(cabecalho_extra)
    L.append(" & ".join(headers) + r" \\")
    L.append(r"\midrule")
    rows = list(rows)
    for i, r in enumerate(rows):
        cells = [str(c) for c in r]
        if i in destaque:
            cells = [rf"\textbf{{{c}}}" if c else c for c in cells]
        L.append(" & ".join(cells) + r" \\")
        if i in midrules_apos and i < len(rows) - 1:
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    if rodape:
        L.append(rf"\multicolumn{{{len(headers)}}}{{p{{0.95\linewidth}}}}{{\footnotesize {rodape}}} \\")
    L += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ───────────────────────── PNG (matplotlib) ─────────────────────────
_LARGURA_MAX_IN = 15.0
_FONTE_BASE = 9.0


def renderizar_png(tab: Tabela, caminho: Path, dpi: int = 160) -> Path:
    """Desenha a tabela em estilo booktabs e grava o PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = [latex_para_texto(h) for h in tab.headers]
    R = [[latex_para_texto(c) for c in r] for r in tab.rows]
    ncol = len(H)
    R = [list(r) + [""] * (ncol - len(r)) for r in R]
    aligns = [a for a in tab.align if a in "lrc"]
    aligns += ["l"] * (ncol - len(aligns))

    # largura das colunas em caracteres (cabeçalho pode quebrar em 2 linhas)
    larg = []
    for j in range(ncol):
        cab = max((len(p) for p in _quebra_cab(H[j])), default=0)
        larg.append(max([cab] + [len(r[j]) for r in R] + [3]))
    fonte = _FONTE_BASE
    char_in = fonte * 0.0088           # largura média de um caractere (DejaVu Sans) em polegadas
    pad = 0.16
    col_w = [w * char_in + pad for w in larg]
    largura = sum(col_w) + 0.5
    if largura > _LARGURA_MAX_IN:      # encolhe a fonte até caber
        f = _LARGURA_MAX_IN / largura
        fonte = max(5.5, fonte * f)
        char_in = fonte * 0.0088
        col_w = [w * char_in + pad * f for w in larg]
        largura = sum(col_w) + 0.5
    linha_h = fonte * 2.1 / 72          # altura de uma linha em polegadas
    n_cab = 2 if any(len(_quebra_cab(h)) > 1 for h in H) else 1
    # rótulos de grupo: cabem na largura somada das colunas do grupo (quebra em 2 linhas e/ou fonte menor)
    grupos_lay = []
    for titulo, a, b in tab.grupos or []:
        disp = sum(col_w[a - 1:b]) - pad / 2
        grupos_lay.append(_ajusta_grupo(latex_para_texto(titulo), disp, fonte, char_in))
    n_grp = max([len(ls) for ls, _ in grupos_lay], default=1) if tab.grupos else 0

    chars_por_linha = max(40, int((largura - 0.5) / (fonte * 0.0085)))
    cap = textwrap.wrap(latex_para_texto(tab.caption), chars_por_linha)
    rod = textwrap.wrap(latex_para_texto(tab.rodape), chars_por_linha) if tab.rodape else []
    n_linhas = n_grp + n_cab + len(R)
    altura = (len(cap) * linha_h * 1.15 + 0.25 + n_linhas * linha_h + len(tab.midrules_apos) * 0.02
              + (len(rod) * linha_h * 0.95 + 0.15 if rod else 0.0) + 0.35)

    fig = plt.figure(figsize=(largura, altura), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, largura); ax.set_ylim(0, altura); ax.axis("off")
    x0 = 0.25
    xs = [x0]
    for w in col_w:
        xs.append(xs[-1] + w)
    x1 = xs[-1]

    y = altura - 0.15
    for ln in cap:                       # legenda (título) no topo
        ax.text(largura / 2, y, ln, ha="center", va="top", fontsize=fonte + 0.5, fontweight="bold")
        y -= linha_h * 1.15
    y -= 0.1
    ax.plot([x0, x1], [y, y], color="black", lw=1.3)     # toprule

    def _x(j: int) -> Tuple[float, str]:
        a = aligns[j]
        if a == "r":
            return xs[j + 1] - pad / 2, "right"
        if a == "c":
            return (xs[j] + xs[j + 1]) / 2, "center"
        return xs[j] + pad / 2, "left"

    if tab.grupos:                       # grupos de colunas + cmidrule
        yc = y - (n_grp * linha_h) / 2
        for (titulo, a, b), (linhas_g, fonte_g) in zip(tab.grupos, grupos_lay):
            xa, xb = xs[a - 1] + pad / 4, xs[b] - pad / 4
            ax.text((xa + xb) / 2, yc, "\n".join(linhas_g), ha="center", va="center", fontsize=fonte_g,
                    linespacing=1.05)
            ax.plot([xa, xb], [y - n_grp * linha_h + 0.01, y - n_grp * linha_h + 0.01], color="black", lw=0.6)
        y -= n_grp * linha_h

    yc = y - (n_cab * linha_h) / 2        # cabeçalho
    for j, h in enumerate(H):
        x, ha = _x(j)
        ax.text(x, yc, "\n".join(_quebra_cab(h)), ha=ha, va="center", fontsize=fonte, fontweight="bold",
                linespacing=1.05)
    y -= n_cab * linha_h
    ax.plot([x0, x1], [y, y], color="black", lw=0.8)      # midrule

    for i, r in enumerate(R):            # corpo
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((x0, y - linha_h), x1 - x0, linha_h, color="#f2f2f2", zorder=0))
        if i in tab.destaque:
            ax.add_patch(plt.Rectangle((x0, y - linha_h), x1 - x0, linha_h, color="#d9d9d9", zorder=0))
        yc = y - linha_h / 2
        for j, c in enumerate(r):
            x, ha = _x(j)
            ax.text(x, yc, c, ha=ha, va="center", fontsize=fonte,
                    fontweight="bold" if i in tab.destaque else "normal", zorder=2)
        y -= linha_h
        if i in tab.midrules_apos and i < len(R) - 1:
            ax.plot([x0, x1], [y - 0.01, y - 0.01], color="black", lw=0.6)
            y -= 0.02
    ax.plot([x0, x1], [y, y], color="black", lw=1.3)      # bottomrule
    y -= 0.12
    for ln in rod:                       # rodapé
        ax.text(x0, y, ln, ha="left", va="top", fontsize=fonte - 1.5, color="#333333")
        y -= linha_h * 0.95

    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=dpi, facecolor="white")
    plt.close(fig)
    return caminho


def _ajusta_grupo(txt: str, disponivel: float, fonte: float, char_in: float) -> Tuple[List[str], float]:
    """Rótulo de grupo do cabeçalho: (linhas, fonte) que cabem em ``disponivel`` polegadas.

    Primeiro tenta uma linha; se não cabe, quebra em até 2 linhas (no espaço mais central);
    se ainda não cabe, reduz a fonte do grupo até a linha mais longa caber (mínimo 5.5 pt).
    """
    if len(txt) * char_in <= disponivel:
        return [txt], fonte
    linhas = [txt]
    if " " in txt:
        meio = len(txt) // 2
        corte = min((i for i, ch in enumerate(txt) if ch == " "), key=lambda i: abs(i - meio))
        linhas = [txt[:corte], txt[corte + 1:]]
    maior = max(len(ln) for ln in linhas) * char_in
    if maior <= disponivel:
        return linhas, fonte
    return linhas, max(5.5, fonte * disponivel / maior)


def _quebra_cab(h: str) -> List[str]:
    """Quebra cabeçalhos longos em até 2 linhas (no espaço mais central)."""
    if len(h) <= 14 or " " not in h:
        return [h]
    meio = len(h) // 2
    cortes = [i for i, ch in enumerate(h) if ch == " "]
    corte = min(cortes, key=lambda i: abs(i - meio))
    return [h[:corte], h[corte + 1:]]


# ───────────────────────── formatação numérica compartilhada ─────────────────────────
def num(v, casas: int = 1, pct: bool = True) -> str:
    """Número em notação brasileira; NaN/None → —. pct=True multiplica por 100."""
    if v is None:
        return "—"
    try:
        if isinstance(v, str):
            return v
        if np.isnan(v):
            return "—"
    except TypeError:
        return str(v)
    x = 100 * v if pct else v
    return f"{x:.{casas}f}".replace(".", ",")


def delta(a, b, casas: int = 1, pct: bool = True) -> str:
    """b − a com sinal (+ / $-$); NaN → —; nunca '−0,0'."""
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return "—"
    d = b - a
    if abs(d) < 1e-12:
        d = 0.0
    return ("+" if d >= 0 else "$-$") + num(abs(d), casas, pct)


_LANDIS = [(0.2, "fraca"), (0.4, "razoável"), (0.6, "moderada"), (0.8, "substancial"), (np.inf, "quase perfeita")]


def faixa_kappa(k) -> str:
    """Faixa qualitativa de Cohen's κ segundo Landis e Koch (1977), como na seção IV.3
    da metodologia do orientador: < 0,2 fraca; 0,2 a 0,4 razoável; 0,4 a 0,6 moderada;
    0,6 a 0,8 substancial; > 0,8 quase perfeita; κ negativo = pior que o acaso.

    Convenção dos limites: cada faixa é fechada no limite inferior e aberta no superior
    (0,4 = moderada, 0,6 = substancial, 0,8 = quase perfeita). O PDF escreve "> 0,8"
    para a última faixa; o caso exato κ = 0,8 é tratado aqui como "quase perfeita"
    (decisão registrada na revisão de 09/10/2026, item M18; só afeta o limite exato).
    NaN ou None → "—".
    """
    if k is None or np.isnan(k):
        return "—"
    if k < 0:
        return "pior que o acaso"
    for lim, nome in _LANDIS:
        if k < lim:
            return nome
    return "quase perfeita"
