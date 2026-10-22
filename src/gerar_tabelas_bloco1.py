"""Tabelas do Bloco 1 (Passos 2 a 6 do A, 2 a 5 do D e 2 a 4 de B e C), 0 chamadas; nomenclatura por passos desde 13/10/2026.

Lê, da pasta de uma execução, os CSVs `bloco1_ad_passos_2_a_5_*.csv` e
`bloco1_bc_passo3_consistencia_*.csv` (e, se houver, `llm_interactions*.json` para a taxa de
flip por regime) e devolve objetos `Tabela` (ver `tabelas.py`), que o orquestrador
escreve em `<pasta>/tabelas/` como ``.tex`` (booktabs) e, via `gerar_tabelas.py`, também
como ``.png`` na raiz da execução:

- ``bloco1_table_01_A_passos_2_a_5__<alias>.tex`` (ex-T1)  A, Passos 2 a 5: zero-shot | 4/10/20/40-shot
  (exemplos = rótulo real) | W ajustado ao rótulo real (teto), Perceptron e NNLS;
- ``bloco1_table_02_A_passo2_vs_passo3`` (ex-T6)  A, Passos 2 a 5: zero-shot × cada k lado a lado, com Δ;
- ``bloco1_table_03_A_passo6_eleicao`` (ex-T7)  A, Passo 6: candidatas e eleita;
- ``bloco1_table_06_BC_passo3_todas_metricas_A`` (ex-T2)  B e C, Passo 3: contra cada métrica do A;
- ``bloco1_table_07_BC_passo3_por_semente`` (ex-T8)  B e C, Passo 3;
- ``bloco1_table_08_BC_passo3_zero_vs_eleita`` (ex-T9)  B e C, Passo 3: Δ e faixa de κ (Landis & Koch);
- ``bloco1_table_10_D_passos_2_a_5`` (ex-T3)  D, Passos 2 a 5: meia-lua × R²/R³/R⁴ × regime;
- ``bloco1_table_13_AD_passo4_estabilidade_W`` (ex-T4)  A e D, Passos 2 a 5: razão w1/w2 por regime × semente;
- ``final_table_04_auditoria_flip_por_regime`` (ex-T5)  malformadas e flip a T=0 por regime;
- ``todas_as_tabelas_bloco1.tex`` / ``.docx``  coleção standalone SÓ do Bloco 1 (pdflatex / pandoc;
  `--sem-docx` pula). O documento completo `todas_as_tabelas.tex` é escrito por `gerar_tabelas.py`
  e não é tocado por este módulo (decisão de 09/10/2026).

O catálogo COMPLETO (Blocos 1, 2, 3, oráculo, auditoria, fechamento; .tex + .png) está em
`src/gerar_tabelas.py`, que reaproveita as funções deste módulo.

Métricas (seção IV.3 do orientador): Fidelidade (métrica reproduz o LLM; A e D), Acurácia
(vs rótulo real ou vs especialista), Consistência (B e C, centróides locais), Cohen's κ e
F1-Macro. A acurácia (vs rótulo real) do LLM e da métrica aparece nos Blocos 1 e 3 só como
contexto, sempre com o nome completo no rodapé.

Contratos de CSV de 13/10/2026 (sem fallback de nomes de arquivo ou de coluna: execuções anteriores
passam antes por `renomear_assets_legado.py --aplicar`; `acc_cell` ainda aceita o teto gravado em
`fidelity_vs_llm` por execuções até 09/10/2026):
- `bloco1_ad_passos_2_a_5`: regime `passo5_teto` tem `fidelity_vs_llm` NaN e o teto em
  `accuracy_teto_vs_true` (antes: o teto ficava em `fidelity_vs_llm`);
  `fidelity_teto_vs_llm_passo2` (antes `accuracy_gt_metric_vs_llm_zero`).
- `bloco1_bc_passo3_consistencia`: a eleição da melhor métrica do A vem da coluna `eleita_passo6`
  (regra única do runner: maior fidelidade média no A, Perceptron, R², zero ou few-shot).

Médias e (mín–máx) das sementes/repetições; NaN (colapso ou quase-colapso em classe única,
`collapsed=1`) vira "—".

Uso:
    python src/gerar_tabelas_bloco1.py                      # última _apenas-bloco1* ou _completa
    python src/gerar_tabelas_bloco1.py resultados/execucao_2026-10-02_..._apenas-bloco1 [--saida tabelas]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execucao_io import _model_alias  # noqa: E402
from protocolo import MIN_PONTOS_CLASSE_MINORITARIA  # noqa: E402
from tabelas import Tabela, delta as _delta, esc as _esc, faixa_kappa, num as _num  # noqa: E402,F401

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = BASE_DIR / "resultados"   # raiz das pastas execucao_* (08/10/2026)
REGIME_ORDEM = {"passo2_zero_shot": 0, "passo3_exemplos_rotulo_real": 1, "passo5_teto": 2}

# Frases de rodapé compartilhadas (uma redação, várias tabelas).
NOTA_PASSOS_A = ("Passos do Problema A: zero-shot = Passo 2 (o LLM rotula sem exemplos); $k$-shot = Passo 3 (o LLM "
                 "rotula com $k$ exemplos do rótulo real); a métrica de cada linha é estimada no Passo 4; teto = Passo 5 "
                 "(W ajustado diretamente ao rótulo real).")
NOTA_PASSOS_D = NOTA_PASSOS_A.replace("Problema A", "Problema D")
NOTA_PASSOS_BC = ("Passos de B e C: o LLM rotula os 100 pontos em zero-shot (Passo 2) e cada métrica do A é aplicada a "
                  "esses rótulos (Passo 3).")
NOTA_KAPPA = "$\\kappa$ = Cohen's $\\kappa$ (faixas de Landis e Koch, 1977)."
NOTA_PERMUTACAO = ("Em zero-shot (e com nomes de classe neutros) a correspondência entre nome e classe é "
                   "arbitrária: a acurácia (vs rótulo real) do LLM pode sair abaixo de 50\\%; o valor invariante à "
                   "permutação é max($a$, 1 $-$ $a$).")
NOTA_GAMMA = ("† = em pelo menos uma estimativa o Perceptron não encontrou $\\hat{W}$ que separe os rótulos: "
              "melhor esforço ($\\gamma = 0$) ou margem bruta mínima negativa (algum ponto do lado errado da fronteira).")
NOTA_COLAPSO = (f"``—'' = colapso em classe única (ou quase-colapso: classe minoritária com menos de {MIN_PONTOS_CLASSE_MINORITARIA} pontos); "
                "na linha colapsada também a acurácia do LLM e a partição ficam como ``—''.")
NOTA_EXEMPLOS_BC = ("Os exemplos few-shot de B e C são rotulados por $\\hat{W}$ do A com os centróides do A; a consistência "
                    "usa os centróides locais, recalculados pelos rótulos do LLM no próprio problema.")


# ───────────────────────── colunas com fallback ─────────────────────────
def _col(df, novo: str, antigo: Optional[str] = None) -> Optional[str]:
    """Nome da coluna presente: `novo` (contrato de 13/10/2026) ou, se ausente, `antigo` (CSVs antigos).

    Aceita DataFrame, Series (linha), dict ou qualquer coleção de nomes. Devolve None se nenhum existe.
    """
    cols = df.columns if hasattr(df, "columns") else df
    if novo in cols:
        return novo
    if antigo is not None and antigo in cols:
        return antigo
    return None




# ───────────────────────── carga e agregação ─────────────────────────
def pasta_padrao(root: Path = RESULTADOS_DIR) -> Optional[Path]:
    """Última `_apenas-bloco1*` (sem `_smoke`); senão a última `_completa`."""
    cands = [p for p in sorted(root.glob("execucao_*_apenas-bloco1*")) if not p.name.endswith("_smoke")]
    if not cands:
        cands = sorted(root.glob("execucao_*_completa"))
    return cands[-1] if cands else None


def carregar(pasta: Path):
    def _ultimo(padrao):
        fs = sorted(pasta.glob(padrao))
        return pd.read_csv(fs[-1]) if fs else None
    return _ultimo("bloco1_ad_passos_2_a_5_*.csv"), _ultimo("bloco1_bc_passo3_consistencia_*.csv")


def agregar(df: pd.DataFrame, chaves: Sequence[str], colunas: Sequence[str]) -> pd.DataFrame:
    """Média, mínimo e máximo por grupo (NaN ignorado; grupo todo NaN → NaN). Colunas ausentes são ignoradas."""
    colunas = [c for c in colunas if c in df.columns]
    g = df.groupby(list(chaves), dropna=False)[list(colunas)].agg(["mean", "min", "max"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return g.reset_index()


# ───────────────────────── formatação ─────────────────────────
def fmt_stat(row, col: str, pct: bool = True, casas: int = 1) -> str:
    m, lo, hi = row.get(f"{col}_mean"), row.get(f"{col}_min"), row.get(f"{col}_max")
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "—"
    s = _num(m, casas, pct)
    if lo is not None and hi is not None and not np.isnan(lo) and not np.isnan(hi) and (hi - lo) > 1e-12:
        s += f" ({_num(lo, casas, pct)}–{_num(hi, casas, pct)})"
    return s


def fmt_particao(row) -> str:
    n0, n1 = row.get("n_classe_0_llm_mean"), row.get("n_classe_1_llm_mean")
    if n0 is None or np.isnan(n0):
        return "—"
    return f"{n0:.0f}/{n1:.0f}"


def fmt_w(row, n: int = 2) -> str:
    vals = [row.get(f"w_unit_{i}_mean") for i in range(n)]
    if any(v is None or np.isnan(v) for v in vals):
        return "—"
    return "[" + "; ".join(_num(v, 2, pct=False) for v in vals) + "]"


def regime_label(regime: str, k) -> str:
    if regime == "passo2_zero_shot":
        return "zero-shot"
    if regime == "passo5_teto":
        return "W ajustado ao rótulo real (teto)"
    return f"{int(k)}-shot"


def marca_gamma(row) -> str:
    """"†" se alguma estimativa do Perceptron no grupo ficou em melhor esforço (γ = 0); "" caso contrário."""
    if not hasattr(row, "get"):
        return ""
    g = row.get("gamma_min")
    mm = row.get("margem_minima_min")          # 15/10/2026 (D): margem bruta mínima/||w||
    g0 = g is not None and not (isinstance(g, float) and np.isnan(g)) and float(g) <= 0.0
    mm0 = mm is not None and not (isinstance(mm, float) and np.isnan(mm)) and float(mm) < 0.0
    return "†" if (g0 or mm0) else ""


def fid_cell(row, regime: Optional[str] = None, **kw) -> str:
    """Fidelidade (vs LLM) agregada; o teto (regime passo5_teto) não tem fidelidade → "—"."""
    regime = regime if regime is not None else row.get("regime")
    if regime == "passo5_teto" or not len(row):
        return "—"
    return fmt_stat(row, "fidelity_vs_llm", **kw)


def acc_cell(row, regime: Optional[str] = None, **kw) -> str:
    """Acurácia (vs rótulo real) da métrica, agregada. No teto lê `accuracy_teto_vs_true` (contrato novo),
    com fallback a `accuracy_metric_vs_true_full` e, por fim, ao valor antigo gravado em `fidelity_vs_llm`."""
    if not len(row):
        return "—"
    regime = regime if regime is not None else row.get("regime")
    if regime == "passo5_teto":
        for c in ("accuracy_teto_vs_true", "accuracy_metric_vs_true_full", "fidelity_vs_llm"):
            s = fmt_stat(row, c, **kw)
            if s != "—":
                return s
        return "—"
    return fmt_stat(row, "accuracy_metric_vs_true_full", **kw)




def _ordem(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_o"] = df["regime"].map(REGIME_ORDEM).fillna(9)
    return df.sort_values(["_o", "k"]).drop(columns="_o")


# ───────────────────────── eleição da melhor métrica do A ─────────────────────────
def eleita_passo6(df_bc: Optional[pd.DataFrame], model: str) -> Optional[Tuple[str, int]]:
    """(regime, k) eleito pelo runner (`a_passo6_eleger_metrica`: maior fidelidade média no A, Perceptron, R²,
    zero ou few-shot, sem colapso), lido da coluna `eleita_passo6` de `bloco1_bc_passo3_consistencia`.
    None se o CSV ou a coluna não existem ou nada foi marcado."""
    if df_bc is None or "eleita_passo6" not in df_bc.columns:
        return None
    sub = df_bc[(df_bc["model"] == model) & (df_bc["eleita_passo6"].fillna(0).astype(float) >= 0.5)]
    if sub.empty:
        return None
    pares = sub[["metrica_regime", "metrica_k"]].drop_duplicates()
    if len(pares) > 1:
        print(f"  (aviso: {len(pares)} métricas marcadas como eleita_passo6 para {model}; usando a primeira)")
    r = pares.iloc[0]
    return str(r["metrica_regime"]), int(r["metrica_k"])


# ───────────────────────── T1 / T3: Passos 2 a 5 de A e D ─────────────────────────
_COLS_A = ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "accuracy_teto_vs_true", "llm_accuracy_vs_true",
           "n_classe_0_llm", "n_classe_1_llm", "w_unit_0", "w_unit_1", "w_unit_2", "w_unit_3", "collapsed", "gamma", "margem_minima"]


def _linhas_regime(df_p: pd.DataFrame, n_feat: int) -> List[List[str]]:
    """Linhas (regime) com colunas Acur. LLM vs real | partição | Perc: fid, acur vs real, W | NNLS: idem."""
    sub = df_p[df_p["n_features"] == n_feat].copy()
    if "collapsed" in sub.columns:
        # Colapso (collapsed=1): a acurácia do LLM e a partição gravadas (ex.: 149/1) não medem um critério;
        # saem das médias como a fidelidade (NaN → "—").
        sub.loc[sub["collapsed"] == 1, ["llm_accuracy_vs_true", "n_classe_0_llm", "n_classe_1_llm"]] = np.nan
    ag = agregar(sub, ["regime", "k", "algorithm"], _COLS_A)
    perc = _ordem(ag[ag["algorithm"] == "perceptron"]).reset_index(drop=True)
    nnls = ag[ag["algorithm"] == "nnls"].set_index(["regime", "k"])
    rows = []
    for _, r in perc.iterrows():
        rn = nnls.loc[(r["regime"], r["k"])] if (r["regime"], r["k"]) in nnls.index else {}
        reg = r["regime"]
        gt = reg == "passo5_teto"
        ncol = int(sum(1 for i in range(4) if f"w_unit_{i}_mean" in r and not np.isnan(r[f"w_unit_{i}_mean"])) or 2)
        rows.append([regime_label(reg, r["k"]) + marca_gamma(r),
                     "—" if gt else fmt_stat(r, "llm_accuracy_vs_true"),
                     "—" if gt else fmt_particao(r),
                     fid_cell(r, reg), acc_cell(r, reg), fmt_w(r, ncol),
                     fid_cell(rn, reg) if len(rn) else "—",
                     acc_cell(rn, reg) if len(rn) else "—",
                     fmt_w(rn, ncol) if len(rn) else "—"])
    return rows


_HDR_A = ["Regime", "Acur.\\ LLM vs real", "Partição LLM (n0/n1)", "Fid.\\ (vs LLM)", "Acur.\\ vs real",
          "$\\hat{W}$ unit.", "Fid.\\ (vs LLM)", "Acur.\\ vs real", "$\\hat{W}$ unit."]
_GRUPOS_A = [("Perceptron", 4, 6), ("NNLS", 7, 9)]
_NOTA_COLUNAS_A = ("Fid.\\ (vs LLM) = fidelidade: fração dos pontos em que a métrica reproduz o rótulo do LLM. "
                   "Acur.\\ vs real = acurácia (vs rótulo real) da métrica aplicada aos 150 pontos; Acur.\\ LLM vs real = "
                   "acurácia (vs rótulo real) do próprio LLM, mostrada como contexto.")


def tab_a_passos_2_a_5(df_a: pd.DataFrame, model: str) -> Tabela:
    df_p = df_a[(df_a["model"] == model) & (df_a["problem"] == "A")]
    rows = _linhas_regime(df_p, 2)
    rod = ("Médias das sementes e repetições, mín–máx entre parênteses, em \\%. " + _NOTA_COLUNAS_A + " "
           "Few-shot: exemplos rotulados pelo rótulo real (k/2 por classe, sorteio aleatório), LLM classifica os "
           "$150-k$ restantes; fidelidade medida nesses pontos. Teto: W ajustado diretamente ao rótulo real; não tem "
           "fidelidade (não é estimado a partir do LLM), só acurácia (vs rótulo real), que não chega a 100\\% pela "
           "sobreposição das gaussianas do A. " + NOTA_PASSOS_A + " " + NOTA_PERMUTACAO + " " + NOTA_GAMMA + " " + NOTA_COLAPSO)
    return Tabela(slug="bloco1_A_passos_2_a_5", nn=1, modelo=model,
                  caption=f"Problema A, Passos 2 a 5 ({_esc(model)}): métrica estimada em zero-shot, few-shot e pelo rótulo real.",
                  headers=_HDR_A, rows=rows, align="lrrrrlrrl", label=f"tab:A_passos_2_a_5_{_model_alias(model)}",
                  grupos=_GRUPOS_A, rodape=rod, midrules_apos={len(rows) - 2} if len(rows) > 1 else set())


def tab_d_passos_2_a_5(df_a: pd.DataFrame, model: str) -> Tabela:
    df_p = df_a[(df_a["model"] == model) & (df_a["problem"] == "D")]
    rows, mids = [], []
    ag_fid = agregar(df_p[df_p["algorithm"] == "perceptron"], ["regime", "k", "n_features"], ["fidelity_vs_llm"])
    for nf in sorted(df_p["n_features"].unique()):
        linhas = _linhas_regime(df_p, int(nf))
        for ln in linhas:
            ln[0] = f"R$^{int(nf)}$ — {ln[0]}"
        rows.extend(linhas)
        mids.append(len(rows) - 1)
    base = ag_fid[ag_fid["n_features"] == 2].set_index(["regime", "k"])["fidelity_vs_llm_mean"]
    deltas = []
    for nf in sorted(df_p["n_features"].unique()):
        sub = _ordem(ag_fid[ag_fid["n_features"] == nf])
        for _, r in sub.iterrows():
            deltas.append("—" if r["regime"] == "passo5_teto"
                          else _delta(base.get((r["regime"], r["k"]), np.nan), r["fidelity_vs_llm_mean"]))
    if len(deltas) == len(rows):
        rows = [list(r) + [dlt] for r, dlt in zip(rows, deltas)]
        hdr, align = _HDR_A + [r"$\Delta$ fid.\ vs R$^2$"], "lrrrrlrrlr"
    else:
        hdr, align = _HDR_A, "lrrrrlrrl"
    rod = ("Meia-lua (150 pontos, sem split). O LLM vê sempre só x1, x2; R$^3$ acrescenta x1·x2 e R$^4$ "
           "acrescenta x1², x2² apenas na estimação da métrica. Colunas e convenções da bloco1\\_table\\_01: o teto "
           "(W ajustado ao rótulo real) só tem acurácia (vs rótulo real), não fidelidade. " + NOTA_PASSOS_D + " "
           + NOTA_PERMUTACAO + " " + NOTA_GAMMA + " " + NOTA_COLAPSO)
    return Tabela(slug="bloco1_D_passos_2_a_5", nn=10, modelo=model,
                  caption=f"Problema D, meia-lua, Passos 2 a 5 ({_esc(model)}): métrica em R$^2$, R$^3$ e R$^4$ por regime.",
                  headers=hdr, rows=rows, align=align, label=f"tab:D_passos_2_a_5_{_model_alias(model)}",
                  grupos=_GRUPOS_A, rodape=rod, midrules_apos=set(mids[:-1]), tamanho=r"\footnotesize")


# ───────────────────────── T2 / T4 / T5 ─────────────────────────
def tab_bc_passo3_todas_metricas(df_bc: pd.DataFrame, model: str) -> Tabela:
    sub = df_bc[df_bc["model"] == model]
    if "n_features" in sub.columns:
        sub = sub[sub["n_features"] == 2]
    ag = agregar(sub, ["problem", "metrica_regime", "metrica_k", "algorithm"],
                 ["consistency", "kappa", "f1", "eleita_passo6", "llm_accuracy_vs_true"])
    rows, destaque, mids = [], set(), []
    for prob in sorted(ag["problem"].unique()):
        a = ag[ag["problem"] == prob].copy()
        a["_o"] = a["metrica_regime"].map(REGIME_ORDEM).fillna(9)
        a["_alg"] = a["algorithm"].map({"perceptron": 0, "nnls": 1}).fillna(2)   # Perceptron antes do NNLS
        a = a.sort_values(["_o", "metrica_k", "_alg"])
        for _, r in a.iterrows():
            i = len(rows)
            rows.append([prob, regime_label(r["metrica_regime"], r["metrica_k"]), r["algorithm"],
                         fmt_stat(r, "consistency"), fmt_stat(r, "kappa", pct=False, casas=3), fmt_stat(r, "f1"),
                         fmt_stat(r, "llm_accuracy_vs_true")])
            if r.get("eleita_passo6_mean", 0) >= 0.5:
                destaque.add(i)
        mids.append(len(rows) - 1)
    rod = ("Rótulos ZERO-SHOT do LLM em B e C (100 pontos novos cada); só $\\hat{W}$ é transferido do A (R$^2$), os "
           "centróides são recalculados com os rótulos do LLM no próprio problema (consistência). " + NOTA_KAPPA + " "
           "F1-Macro entre a métrica e o LLM. Acur.\\ LLM vs real = acurácia (vs rótulo real) do LLM, contexto. "
           + NOTA_PERMUTACAO + " Em negrito: as linhas (Perceptron e NNLS) do regime/k eleito pelo runner "
           "(coluna eleita\\_passo6: maior fidelidade média no A, Perceptron, R$^2$, zero ou few-shot), excluído o teto. "
           + NOTA_PASSOS_BC + " A métrica de origem vem do Passo 4 do A (zero-shot = rótulos do Passo 2; $k$-shot = rótulos do Passo 3; "
           "``W ajustado ao rótulo real (teto)'' = Passo 5 do A, que não concorre à eleição).")
    return Tabela(slug="bloco1_BC_passo3_todas_metricas_A", nn=6, modelo=model,
                  caption=f"Problemas B e C, Passo 3 ({_esc(model)}): transferência zero-shot contra cada métrica estimada no A.",
                  headers=["Prob.", "Métrica de origem (A)", "Algoritmo", "Consistência (\\%)", "$\\kappa$", "F1-Macro (\\%)",
                           "Acur.\\ LLM vs real"],
                  rows=rows, align="llllrrr", label=f"tab:BC_passo3_todas_metricas_A_{_model_alias(model)}",
                  destaque=destaque, rodape=rod, midrules_apos=set(mids[:-1]))


def tab_ad_estabilidade_w(df_a: pd.DataFrame, model: str) -> Tabela:
    sub = df_a[(df_a["model"] == model) & (df_a["n_features"] == 2)]
    seeds = sorted(sub["random_seed"].unique())
    rows, mids = [], []
    for prob in sorted(sub["problem"].unique()):
        for algo in ("perceptron", "nnls"):
            s2 = sub[(sub["problem"] == prob) & (sub["algorithm"] == algo)]
            if s2.empty:
                continue
            ag = agregar(s2, ["regime", "k", "random_seed"], ["w_ratio"])
            # dropna=False: um regime em que TODAS as sementes colapsaram continua na tabela, como "—"
            piv = ag.pivot_table(index=["regime", "k"], columns="random_seed", values="w_ratio_mean", dropna=False)
            regimes = _ordem(s2[["regime", "k"]].drop_duplicates()).set_index(["regime", "k"]).index
            piv = piv.reindex(regimes).reindex(columns=seeds)
            for (reg, k), r in piv.iterrows():
                vals = [r.get(sd, np.nan) for sd in seeds]
                v = np.array([x for x in vals if not np.isnan(x)])
                cv = (v.std() / abs(v.mean())) if len(v) > 1 and v.mean() != 0 else np.nan
                rows.append([prob, {"perceptron": "Perceptron", "nnls": "NNLS"}.get(algo, algo), regime_label(reg, k)] +
                            [_num(x, 2, pct=False) for x in vals] + [_num(cv, 2, pct=False)])
            mids.append(len(rows) - 1)
    rod = ("Razão $w_1/w_2$ da métrica estimada (invariante de escala: só a direção importa) por semente; "
           "CV = desvio padrão / |média| entre sementes. " + NOTA_PASSOS_A + " O mesmo vale para o Problema D. " + NOTA_COLAPSO)
    return Tabela(slug="bloco1_AD_passo4_estabilidade_W", nn=13, modelo=model,
                  caption=f"Problemas A e D, Passos 2 a 5 ({_esc(model)}): estabilidade da direção de $\\hat{{W}}$ por regime e semente.",
                  headers=["Prob.", "Algoritmo", "Regime"] + [f"seed {s}" for s in seeds] + ["CV"], rows=rows,
                  align="lll" + "r" * len(seeds) + "r", label=f"tab:AD_passo4_estabilidade_W_{_model_alias(model)}",
                  rodape=rod, midrules_apos=set(mids[:-1]))


def tab_final_auditoria_flip(por_regime: Dict[str, dict]) -> Tabela:
    # 15/10/2026: o F chega partido por semente (problema=F_meia_lua_seed<S> na tag); agrupa-se em F_meia_lua
    # somando as contagens e recalculando as taxas, como A, D e G (uma linha por problema).
    import re as _re
    agreg: Dict[tuple, dict] = {}
    for nome, m in por_regime.items():
        prov_model, prob, passo, regime, k = nome.split("|")
        if regime == "sem_tag":
            continue
        prob = _re.sub(r"_seed\d+$", "", prob)
        a = agreg.setdefault((prov_model, prob, passo, regime, k),
                             {"total_chamadas": 0, "malformadas": 0, "queries_repetidas": 0, "queries_divergentes": 0})
        a["total_chamadas"] += int(m.get("total_chamadas", 0))
        a["malformadas"] += int(m.get("malformadas", round(m.get("taxa_malformadas", 0.0) * m.get("total_chamadas", 0))))
        a["queries_repetidas"] += int(m.get("queries_repetidas", 0))
        a["queries_divergentes"] += int(m.get("queries_divergentes", round(m.get("taxa_flip_T0", 0.0) * m.get("queries_repetidas", 0))))
    rows = []
    for (prov_model, prob, passo, regime, k), a in agreg.items():
        taxa_mal = a["malformadas"] / a["total_chamadas"] if a["total_chamadas"] else 0.0
        taxa_flip = a["queries_divergentes"] / a["queries_repetidas"] if a["queries_repetidas"] else 0.0
        rows.append([_esc(prov_model.split("/", 1)[-1]), _esc(prob), passo.replace("passo=", ""), _esc(regime),
                     k.replace("k=", ""),
                     f"{a['total_chamadas']:,}".replace(",", "."), _num(taxa_mal, 2),
                     f"{a['queries_repetidas']:,}".replace(",", "."),
                     "—" if a["queries_repetidas"] == 0 else _num(taxa_flip, 2)])
    rod = ("Auditoria offline das interações (\\texttt{src/audit\\_interactions.py}); métricas de processo, não de "
           "avaliação. Flip = fração dos pares (prompt, ponto) repetidos cuja resposta interpretada divergiu; só "
           "mensurável onde há repetição. Passo = passo do problema em que a chamada acontece; regime = origem dos "
           "exemplos: passo2\\_zero\\_shot = Passo 2, sem exemplos (A, B/C, D, E, F, G); passo3\\_exemplos\\_rotulo\\_real = "
           "Passo 3 de A, D e G, exemplos do rótulo real; zero\\_shot = k = 0 do aprendizado em contexto (Passo 3 do F, "
           "Passo 6 do G); exemplos\\_rotulo\\_real = Passo 3 do F; exemplos\\_da\\_metrica = métrica do A, em B e C "
           "(Passo 4), ou métrica eleita, em G (Passo 6); easy/hard/mixed/random = estratégias do Passo 3 do E; "
           "diluicao = Passo 4 do E1; class0\\_first/class1\\_first/shuffled/alternating = ordenações do Passo 5 do E1; "
           "vies\\_* = experimentos de viés. O Problema F agrega as três sementes numa linha (as tags gravam o seed no "
           "nome do problema).")
    return Tabela(slug="final_auditoria_flip_por_regime", nn=4,
                  caption="Malformadas e flip a $T=0$ por problema e Passo (regime de coleta).", label="tab:final_auditoria_flip_por_passo",
                  headers=["Modelo", "Prob.", "Passo", "Regime", "k", "Chamadas", "Malf.\\ (\\%)", "Repetidas", "Flip (\\%)"],
                  rows=rows, align="lllrrrrrr", rodape=rod)


# ───────────────────────── T6–T9: zero-shot × few-shot e transferência ─────────────────────────
def _agreg_a(df_a: pd.DataFrame, model: str) -> pd.DataFrame:
    """Problema A, R²: média/mín/máx por (regime, k, algoritmo)."""
    sub = df_a[(df_a["model"] == model) & (df_a["problem"] == "A") & (df_a["n_features"] == 2)]
    return agregar(sub, ["regime", "k", "algorithm"],
                   ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "accuracy_teto_vs_true",
                    "llm_accuracy_vs_true", "cos_sim_perc_nnls", "gamma", "margem_minima"])


def tab_a_zero_vs_exemplos(df_a: pd.DataFrame, model: str) -> Tabela:
    """Problema A: zero-shot × cada k, lado a lado, por algoritmo."""
    ag = _agreg_a(df_a, model)
    rows, mids = [], []
    for algo in ("perceptron", "nnls"):
        a = ag[ag["algorithm"] == algo].set_index(["regime", "k"])
        if ("passo2_zero_shot", 0) not in a.index:
            continue
        z = a.loc[("passo2_zero_shot", 0)]
        for k in sorted(int(k) for (r, k) in a.index if r == "passo3_exemplos_rotulo_real"):
            f = a.loc[("passo3_exemplos_rotulo_real", k)]
            rows.append([algo, f"{k}-shot",
                         fmt_stat(z, "fidelity_vs_llm") + marca_gamma(z), fmt_stat(f, "fidelity_vs_llm") + marca_gamma(f),
                         _delta(z["fidelity_vs_llm_mean"], f["fidelity_vs_llm_mean"]),
                         fmt_stat(z, "accuracy_metric_vs_true_full"), fmt_stat(f, "accuracy_metric_vs_true_full"),
                         _delta(z["accuracy_metric_vs_true_full_mean"], f["accuracy_metric_vs_true_full_mean"]),
                         fmt_stat(z, "llm_accuracy_vs_true"), fmt_stat(f, "llm_accuracy_vs_true"),
                         _delta(z["llm_accuracy_vs_true_mean"], f["llm_accuracy_vs_true_mean"]),
                         _num(z.get("cos_sim_perc_nnls_mean", np.nan), 3, pct=False),
                         _num(f.get("cos_sim_perc_nnls_mean", np.nan), 3, pct=False)])
        mids.append(len(rows) - 1)
    hdr = ["Algoritmo", "Few-shot", "zero", "few", "$\\Delta$", "zero", "few", "$\\Delta$", "zero", "few", "$\\Delta$",
           "zero", "few"]
    grupos = [("Fidelidade (vs LLM)", 3, 5), ("Acur.\\ da métrica vs real", 6, 8),
              ("Acur.\\ do LLM vs real", 9, 11), ("cos(Perc., NNLS)", 12, 13)]
    rod = ("Problema A em R$^2$, médias das sementes e repetições (mín–máx entre parênteses), em \\%. "
           "$\\Delta$ = few-shot $-$ zero-shot, em pontos percentuais. Few-shot: k exemplos rotulados pelo rótulo real "
           "(k/2 por classe). Acurácia (vs rótulo real) da métrica e do LLM como contexto. " + NOTA_PERMUTACAO + " "
           "cos = cosseno entre os $\\hat{W}$ do Perceptron e do NNLS na mesma coleta. " + NOTA_PASSOS_A + " " + NOTA_GAMMA)
    return Tabela(slug="bloco1_A_passo2_vs_passo3", nn=2, modelo=model,
                  caption=f"Problema A, Passos 2 a 4 ({_esc(model)}): zero-shot versus few-shot, lado a lado (o teto, Passo 5, está na tabela 01).",
                  headers=hdr, rows=rows, align="llrrrrrrrrrrr", label=f"tab:A_passo2_vs_passo3_{_model_alias(model)}",
                  grupos=grupos, rodape=rod, midrules_apos=set(mids[:-1]), tamanho=r"\footnotesize")


def tab_a_passo6_eleicao(df_a: pd.DataFrame, df_bc: Optional[pd.DataFrame], model: str) -> Tabela:
    """Candidatas à transferência (zero-shot e cada k do A): fidelidade no A e resultado em B/C; a eleita marcada."""
    ag = _agreg_a(df_a, model)
    perc = _ordem(ag[(ag["algorithm"] == "perceptron") & (ag["regime"] != "passo5_teto")]).reset_index(drop=True)
    eleita_rk = eleita_passo6(df_bc, model)
    bc = None
    if df_bc is not None:
        sub = df_bc[(df_bc["model"] == model) & (df_bc["algorithm"] == "perceptron")]
        if "n_features" in sub.columns:
            sub = sub[sub["n_features"] == 2]
        if len(sub):
            bc = agregar(sub, ["problem", "metrica_regime", "metrica_k"], ["consistency", "kappa"]).set_index(
                ["problem", "metrica_regime", "metrica_k"])
    rows, destaque = [], set()

    def _bc(prob, regime, k, col):
        if bc is None or (prob, regime, k) not in bc.index:
            return "—"
        r = bc.loc[(prob, regime, k)]
        return _num(r[f"{col}_mean"], 3, pct=False) if col == "kappa" else _num(r[f"{col}_mean"], 1)

    for _, r in perc.iterrows():
        reg, k = r["regime"], int(r["k"])
        eleita = eleita_rk is not None and (reg, k) == eleita_rk
        rows.append([regime_label(reg, k) + marca_gamma(r),
                     fmt_stat(r, "fidelity_vs_llm"), fmt_stat(r, "accuracy_metric_vs_true_full"),
                     _num(r.get("cos_sim_perc_nnls_mean", np.nan), 3, pct=False),
                     _bc("B", reg, k, "consistency"), _bc("B", reg, k, "kappa"),
                     _bc("C", reg, k, "consistency"), _bc("C", reg, k, "kappa"),
                     "eleita" if eleita else ""])
        if eleita:
            destaque.add(len(rows) - 1)
    origem = ("lida da coluna eleita\\_passo6 de bloco1\\_bc\\_passo3\\_consistencia (regra única do runner)"
              if eleita_rk is not None else "indisponível nesta execução (sem coluna eleita\\_passo6)")
    rod = ("Critério de eleição: maior fidelidade média no Problema A (Perceptron, R$^2$, sem colapso) entre o zero-shot "
           f"e cada k few-shot; o teto não concorre. Eleição {origem}. Acur.\\ vs real (A) = acurácia (vs rótulo real) "
           "da métrica, contexto. Consistência e $\\kappa$ em B e C: rótulos zero-shot do LLM contra a métrica de origem "
           "(Perceptron, só $\\hat{W}$ transferido, centróides recalculados pelos rótulos do LLM). " + NOTA_KAPPA + " "
           + NOTA_GAMMA + " " + NOTA_PASSOS_A + " Em negrito: a métrica eleita para a transferência (Passo 6).")
    return Tabela(slug="bloco1_A_passo6_eleicao", nn=3,
                  modelo=model,
                  caption=f"Problema A, Passo 6 ({_esc(model)}): escolha da melhor métrica para a transferência.",
                  headers=["Métrica de origem (A)", "Fid.\\ (vs LLM) no A", "Acur.\\ vs real (A)", "cos(Perc., NNLS)",
                           "Consist.\\ B", "$\\kappa$ B", "Consist.\\ C", "$\\kappa$ C", "Eleição"],
                  rows=rows, align="lrrrrrrrl", label=f"tab:A_passo6_eleicao_{_model_alias(model)}",
                  destaque=destaque, rodape=rod)


def _bc_origens(df_bc: pd.DataFrame, model: str):
    """Sub-DataFrame de B/C do modelo (R²) e as origens comparadas: zero-shot e a eleita (se for few-shot)."""
    sub = df_bc[df_bc["model"] == model]
    if "n_features" in sub.columns:
        sub = sub[sub["n_features"] == 2]
    eleita = eleita_passo6(df_bc, model)
    origens = [("passo2_zero_shot", 0, "zero-shot" + (" (eleita)" if eleita == ("passo2_zero_shot", 0) else ""))]
    if eleita is not None and eleita[0] == "passo3_exemplos_rotulo_real":
        origens.append(("passo3_exemplos_rotulo_real", eleita[1], f"eleita ({eleita[1]}-shot)"))
    return sub, origens


def tab_bc_passo3_zero_vs_melhor_por_semente(df_bc: pd.DataFrame, df_a: pd.DataFrame, model: str) -> Tabela:
    """B e C contra a métrica zero-shot e contra a eleita do A, por semente."""
    sub, origens = _bc_origens(df_bc, model)
    seeds = sorted(int(s) for s in sub["random_seed"].unique())
    rows, mids = [], []
    for prob in sorted(sub["problem"].unique()):
        for algo in ("perceptron", "nnls"):
            for reg, k, nome in origens:
                s2 = sub[(sub["problem"] == prob) & (sub["algorithm"] == algo)
                         & (sub["metrica_regime"] == reg) & (sub["metrica_k"] == k)]
                if s2.empty:
                    continue
                por_seed = s2.groupby("random_seed")[["consistency", "kappa", "f1"]].mean()
                cells = [prob, algo, nome]
                for col, casas, pct in (("consistency", 1, True), ("kappa", 3, False)):
                    cells += [_num(por_seed[col].get(sd, np.nan), casas, pct) for sd in seeds]
                    cells.append(_num(s2[col].mean(), casas, pct))
                cells.append(_num(s2["f1"].mean(), 1))
                rows.append(cells)
        mids.append(len(rows) - 1)
    n = len(seeds)
    hdr = (["Prob.", "Algoritmo", "Métrica de origem (A)"] + [f"seed {s}" for s in seeds] + ["média"]
           + [f"seed {s}" for s in seeds] + ["média", "F1-Macro"])
    grupos = [("Consistência (\\%)", 4, 4 + n), ("$\\kappa$", 5 + n, 5 + 2 * n)]
    rod = ("Rótulos zero-shot do LLM em B e C; só $\\hat{W}$ vem do A, centróides recalculados pelos rótulos do LLM "
           "no próprio problema. ``eleita'' = métrica do A marcada pelo runner (eleita\\_passo6). Na origem few-shot, "
           "o valor por semente é a média das repetições (sorteios) dessa semente. " + NOTA_KAPPA + " F1-Macro em \\%, "
           "média das sementes. " + NOTA_PASSOS_BC)
    return Tabela(slug="bloco1_BC_passo3_por_semente", nn=7,
                  modelo=model,
                  caption=f"Problemas B e C, Passo 3 ({_esc(model)}): transferência com a métrica zero-shot e com a eleita do A, por semente.",
                  headers=hdr, rows=rows, align="lll" + "r" * (2 * (n + 1) + 1), label=f"tab:BC_passo3_por_semente_{_model_alias(model)}",
                  grupos=grupos, rodape=rod, midrules_apos=set(mids[:-1]), tamanho=r"\footnotesize")


def tab_bc_passo3_comparativo(df_bc: pd.DataFrame, df_a: pd.DataFrame, model: str) -> Tabela:
    """Comparativo zero-shot × eleita few-shot em B e C, com Δ e faixa de κ."""
    sub, origens = _bc_origens(df_bc, model)
    slug, nn, label = "bloco1_BC_passo3_zero_vs_eleita", 8, f"tab:BC_passo3_zero_vs_eleita_{_model_alias(model)}"
    if len(origens) < 2:
        motivo = ("a métrica eleita é a própria zero-shot" if eleita_passo6(df_bc, model) == ("passo2_zero_shot", 0)
                  else "sem métrica few-shot eleita no A")
        return Tabela(slug=slug, nn=nn, modelo=model, caption=f"Problemas B e C, Passo 3 ({_esc(model)}): comparativo indisponível.",
                      headers=["Prob.", "Algoritmo"], rows=[["—", motivo]], align="ll", label=label,
                      rodape="Só há comparativo quando a métrica eleita no A (Passo 6, eleita\\_passo6) é few-shot. " + NOTA_PASSOS_BC)
    (rz, kz, _), (rf_, kf, nome_f) = origens
    rows, mids = [], []
    for prob in sorted(sub["problem"].unique()):
        for algo in ("perceptron", "nnls"):
            base = sub[(sub["problem"] == prob) & (sub["algorithm"] == algo)]
            z = base[(base["metrica_regime"] == rz) & (base["metrica_k"] == kz)][["consistency", "kappa", "f1"]].mean()
            f = base[(base["metrica_regime"] == rf_) & (base["metrica_k"] == kf)][["consistency", "kappa", "f1"]].mean()
            rows.append([prob, algo,
                         _num(z["consistency"], 1), _num(f["consistency"], 1), _delta(z["consistency"], f["consistency"]),
                         _num(z["kappa"], 3, pct=False), _num(f["kappa"], 3, pct=False), _delta(z["kappa"], f["kappa"], 3, pct=False),
                         faixa_kappa(z["kappa"]), faixa_kappa(f["kappa"]),
                         _num(z["f1"], 1), _num(f["f1"], 1), _delta(z["f1"], f["f1"])])
        mids.append(len(rows) - 1)
    hdr = ["Prob.", "Algoritmo", "zero", "eleita", "$\\Delta$", "zero", "eleita", "$\\Delta$",
           "zero", "eleita", "zero", "eleita", "$\\Delta$"]
    grupos = [("Consistência (\\%)", 3, 5), ("$\\kappa$", 6, 8), ("Faixa de $\\kappa$ (Landis \\& Koch)", 9, 10),
              ("F1-Macro (\\%)", 11, 13)]
    rod = (f"``zero'' = métrica zero-shot do A; ``eleita'' = {nome_f} do A (eleita\\_passo6: maior fidelidade no A). "
           "Médias das sementes (e repetições); $\\Delta$ = eleita $-$ zero. " + NOTA_KAPPA + " Faixas: $<$0,2 fraca, "
           "0,2–0,4 razoável, 0,4–0,6 moderada, 0,6–0,8 substancial, $>$0,8 quase perfeita. " + NOTA_PASSOS_BC)
    return Tabela(slug=slug, nn=nn, modelo=model,
                  caption=f"Problemas B e C, Passo 3 ({_esc(model)}): comparativo da transferência zero-shot versus métrica eleita do A.",
                  headers=hdr, rows=rows, align="llrrrrrrllrrr", label=label,
                  grupos=grupos, rodape=rod, midrules_apos=set(mids[:-1]), tamanho=r"\footnotesize")


# ───────────────────────── documento / DOCX ─────────────────────────
def documento(incluir: List[str], titulo: str = "Bloco 1 — Problemas A, B, C e D por passo: tabelas") -> str:
    L = [r"\documentclass[10pt]{article}", r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
         r"\usepackage[brazil]{babel}", r"\usepackage{booktabs}", r"\usepackage{amsmath}",
         # página grande: o documento conjunto é só para conferência visual; tabelas com muitas linhas
         # (auditoria por regime, Bloco 3) não cabem num A4 e travariam o posicionamento dos floats
         r"\usepackage[paperwidth=38cm,paperheight=56cm,margin=1.2cm]{geometry}",
         # tabelas sem flutuar (minipage): uma tabela mais alta que a página só transborda, em vez de
         # travar o posicionamento dos floats ("Output loop---100 consecutive dead cycles")
         r"\makeatletter",
         r"\renewenvironment{table}[1][]{\par\vspace{1.2em}\noindent\begin{minipage}{\linewidth}\def\@captype{table}}{\end{minipage}\par}",
         r"\makeatother", r"\begin{document}",
         rf"\section*{{{titulo}}}"]
    L += [rf"\input{{{os.path.basename(f)}}}" for f in incluir]
    L.append(r"\end{document}")
    return "\n".join(L) + "\n"


def exportar_docx(tex_files: List[Path], destino: Path) -> Optional[Path]:
    """Concatena as tabelas num .tex temporário e converte com pandoc (sem \\cmidrule, que o pandoc não lê).
    Devolve o caminho escrito ou None se o pandoc não estiver disponível."""
    import re
    import shutil
    import subprocess
    import tempfile
    if shutil.which("pandoc") is None:
        return None
    partes = []
    for f in tex_files:
        t = f.read_text(encoding="utf-8")
        partes.append(re.sub(r"\\cmidrule\(lr\)\{\d+-\d+\}\s*", "", t))
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(partes))
        nome = tmp.name
    try:
        subprocess.run(["pandoc", "-f", "latex", nome, "-o", str(destino)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"  (pandoc falhou: {exc.stderr.strip()[:200]})")
        return None
    finally:
        os.unlink(nome)
    return destino


# ───────────────────────── catálogo do Bloco 1 ─────────────────────────
def tabelas_bloco1(df_a: Optional[pd.DataFrame], df_bc: Optional[pd.DataFrame],
                   por_regime: Optional[Dict[str, dict]] = None) -> List[Tabela]:
    """Todas as tabelas T1–T9 (por modelo) + T5, na ordem de escrita."""
    out: List[Tabela] = []
    if df_a is None:
        return out
    for model in sorted(df_a["model"].unique()):
        tem_a = ((df_a["model"] == model) & (df_a["problem"] == "A")).any()
        tem_bc = df_bc is not None and (df_bc["model"] == model).any()
        if tem_a:
            out.append(tab_a_passos_2_a_5(df_a, model))
        if tem_bc:
            out.append(tab_bc_passo3_todas_metricas(df_bc, model))
        if ((df_a["model"] == model) & (df_a["problem"] == "D")).any():
            out.append(tab_d_passos_2_a_5(df_a, model))
        if tem_a:
            out += [tab_ad_estabilidade_w(df_a, model), tab_a_zero_vs_exemplos(df_a, model), tab_a_passo6_eleicao(df_a, df_bc, model)]
        if tem_a and tem_bc:
            out += [tab_bc_passo3_zero_vs_melhor_por_semente(df_bc, df_a, model), tab_bc_passo3_comparativo(df_bc, df_a, model)]
    if por_regime:
        out.append(tab_final_auditoria_flip(por_regime))
    return out


NOME_DOC_BLOCO1 = "todas_as_tabelas_bloco1"


def gerar_tabelas(pasta: Path, saida: str = "tabelas", por_regime: Optional[Dict[str, dict]] = None,
                  docx: bool = False, nome_doc: str = NOME_DOC_BLOCO1) -> List[Path]:
    """Escreve os .tex de T1–T9 (+ `<nome_doc>.tex/.docx`, só do Bloco 1) em <pasta>/<saida>/.

    Não toca em `todas_as_tabelas.tex/.docx`, que pertencem ao catálogo completo (`gerar_tabelas.py`)."""
    df_a, df_bc = carregar(pasta)
    if df_a is None:
        raise SystemExit(f"Nenhum bloco1_ad_passos_2_a_5_*.csv em {pasta}")
    if por_regime is None:
        por_regime = _auditoria_por_regime(pasta)
    out = pasta / saida
    out.mkdir(exist_ok=True)
    escritos: List[Path] = []
    for tab in tabelas_bloco1(df_a, df_bc, por_regime):
        f = out / tab.nome_tex()
        f.write_text(tab.tex(), encoding="utf-8")
        escritos.append(f)
    tex_tabelas = list(escritos)
    f = out / f"{nome_doc}.tex"
    f.write_text(documento([str(p) for p in tex_tabelas]), encoding="utf-8")
    escritos.append(f)
    if docx:
        d = exportar_docx(tex_tabelas, out / f"{nome_doc}.docx")
        if d is not None:
            escritos.append(d)
    return escritos


def _auditoria_por_regime(pasta: Path) -> Dict[str, dict]:
    try:
        from audit_interactions import audit, load_interactions
        recs = load_interactions(str(pasta))
        return audit(recs).get("por_regime", {}) if recs else {}
    except Exception as exc:  # auditoria é opcional
        print(f"  (auditoria por regime indisponível: {exc})")
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pasta", nargs="?", default=None)
    ap.add_argument("--saida", default="tabelas")
    ap.add_argument("--sem-docx", action="store_true", help=f"não gerar {NOME_DOC_BLOCO1}.docx (pandoc)")
    args = ap.parse_args(argv)
    pasta = Path(args.pasta) if args.pasta else pasta_padrao()
    if pasta is None:
        raise SystemExit("Nenhuma pasta execucao_* encontrada.")
    if not pasta.is_absolute():
        pasta = (BASE_DIR / pasta) if (BASE_DIR / pasta).exists() else (RESULTADOS_DIR / pasta)
    escritos = gerar_tabelas(pasta, args.saida, docx=not args.sem_docx)
    print(f"Tabelas do Bloco 1 em {pasta / args.saida}/ (o catálogo completo e todas_as_tabelas.tex ficam com "
          "src/gerar_tabelas.py):")
    for f in escritos:
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
