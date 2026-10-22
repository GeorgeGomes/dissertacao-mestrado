"""Catálogo COMPLETO de tabelas de uma execução — 0 chamadas de API.

Varre todos os CSVs da pasta de uma execução (Blocos 1, 2, 3, oráculo, auditoria e
fechamento) e escreve cada tabela em dois formatos:

- ``<pasta>/tabelas/<bloco>_table_<NN>_<assunto>[__<alias>].tex``  (booktabs), mais
  ``todas_as_tabelas.tex`` (compila com pdflatex), ``todas_as_tabelas.docx`` (pandoc, se
  instalado) e ``INDICE.md``;
- ``<pasta>/<bloco>_table_<NN>_<assunto>[__<alias>]_gray.png``  na RAIZ da execução, ao lado
  dos gráficos (``<bloco>_graphic_<NN>_…_color/_gray.png``). NN = ordem no bloco.

Assets derivados de um LLM levam o alias do modelo (`MODEL_ALIAS`); tabelas sem dados de
LLM (oráculo, teto da meia-lua, baselines) ou multi-modelo (auditoria, comparação de
modelos) não levam alias. Tabelas cujos CSVs não existem na execução são simplesmente
puladas (execuções `--apenas` e `--rapido` geram o subconjunto possível).

Catálogo (slug → o que mostra):

Nomenclatura (13/10/2026): letra = problema, cada problema com a sua sequência de passos (ver CLAUDE.md);
slugs, nomes de arquivo e colunas seguem o mapa `renomeacao_passos_2026-10-13.md`.

BLOCO 1 (por modelo)
  bloco1_A_passos_2_a_5                 T1: Problema A, Passos 2 a 5: zero/few-shot/teto, Perceptron e NNLS
  bloco1_A_passo2_vs_passo3        T6: A, Passos 2 a 5: zero-shot × cada k lado a lado, com Δ
  bloco1_A_passo6_eleicao  T7: A, Passo 6: candidatas e eleita (maior fidelidade no A, zero ou few-shot,
                                             lida de `eleita_passo6` em bloco1_bc_passo3_consistencia: regra única do runner)
  bloco1_A_passo4_R2_R3_R4             A, Passos 2 a 5, em R²/R³/R⁴ por regime (fidelidade, acurácia, Δ vs R²)
  bloco1_A_passo4_perceptron_vs_nnls    A, Passo 4, e B/C, Passo 3: fidelidade dos dois estimadores, cosseno, B/C por semente
  bloco1_BC_passo3_todas_metricas_A   T2: B e C, Passo 3: zero-shot contra cada métrica do A
  bloco1_BC_passo3_por_semente  T8: B e C, Passo 3
  bloco1_BC_passo3_zero_vs_eleita       T9: B e C, Passo 3: Δ e faixa de κ
  bloco1_BC_passo4_exemplos_da_metrica      H2: B e C, Passo 4: exemplos rotulados pela métrica zero-shot do A, por k
  bloco1_D_passos_2_a_5                  T3: D, Passos 2 a 5
  bloco1_D_passo2_vs_passo3        D, Passos 2 a 4: zero-shot × cada k, por espaço (R²/R³/R⁴)
  bloco1_AD_por_passo                   A e D, Passos 2 a 5, lado a lado por regime (fidelidade; acurácia vs rótulo real)
  bloco1_AD_passo4_estabilidade_W       T4: A e D, Passos 2 a 5
  bloco1_vies_nomes_classe                   A, Passo 2, e B/C, Passos 3 e 4, por par de nomes de classe (zero e 10-shot;
                                             acurácia invariante à permutação para os pares invertidos)
  bloco1_vies_nomes_atributo                 idem por par de nomes de atributo (só zero-shot)
  bloco1_vies_variantes_prompt               idem por variante de prompt (zero e 10-shot)
BLOCO 1 (sem alias)
  bloco1_oraculo_passos_1_a_3               Oráculo sobre E3/E1/E2 (B até 13/10/2026), Passos 1 a 3 (Problema 1/2/3): W verdadeiro × recuperado
  bloco1_D_passo5_teto              D, Passo 5: W ajustado ao rótulo real da meia-lua em R²/R³/R⁴
BLOCO 2 (por modelo)
  bloco2_E{1,2,3}_passo3_estrategia_x_k   E1/E2/E3, Passos 2 e 3: acurácia e κ vs especialista por estratégia × k
  bloco2_E_passo2_vs_passo3  E, Passos 2 e 3: zero-shot × melhor few-shot × média few-shot, por especialista
  bloco2_E_passo3_H4   E, Passo 3: média few-shot por estratégia, ordem e Δ difícil−fácil (H4)
  bloco2_E_passo6_baselines             E, Passo 6: LLM (maior n) × k-NN, Regressão Logística, SVM-RBF por especialista
  bloco2_E1_passo4_diluicao                         E1, Passo 4: 4 difíceis + N fáceis
  bloco2_E1_passo5_ordem                   E1, Passo 5: ordenações × n
  bloco2_F_passo2_R2_R3_R4            F, Passo 2: zero-shot no treino e métrica por semente e espaço
  bloco2_F_passo3_k               F, Passos 3 e 4: aprendizado em contexto por semente × k
BLOCO 3 (por modelo)
  bloco3_G_passos_2_e_4_R2_R3_R4             G, Passos 2 e 4: G1/G2 × classes × atributos × espaço
  bloco3_G_passo6_k                       G, Passo 6: acurácia (vs rótulo real) por k
  bloco3_G_passo7_LLM_vs_perceptron           G, Passos 6 e 7: LLM × Perceptron (vs rótulo real) por k
  bloco3_G1_vs_G2_passos_2_e_4               G, Passos 2 e 4: normalizado × kg/m lado a lado (LLM, métrica, fidelidade)
  bloco3_G_nomes_semanticos_vs_neutros       G, Passos 2 e 4: Mulher/Homem × A/B e peso/altura × x1/x2 (acurácia invariante à permutação)
  bloco3_G_passos_2_a_5           G, Passos 2 a 5 (mesmos passos do A, 13/10/2026): por (subproblema,
                                             classes, atributos), zero-shot × cada k × teto; fidelidade R²/R³/R⁴ e acurácia
                                             R² (Perceptron), lido de bloco3_g_passos_2_a_5
FECHAMENTO
  final_linearidade_cruzada_R2_R3_R4 (por modelo)   A linear × D/F × G1 × G2 por espaço e variante de atributo
  final_comparacao_modelos (sem alias)       resumo lado a lado dos modelos
  auditoria_flip_por_regime (T5), auditoria_por_modelo

Métricas (seção IV.3 do orientador): Fidelidade, Acurácia (vs rótulo real / LLM vs especialista), Consistência,
Cohen's κ e F1-Macro. Acurácia (vs rótulo real) nos Blocos 1 e 3 é contexto; com nomes de classe neutros ou em
zero-shot a correspondência nome↔classe é arbitrária, e as tabelas de vieses e do Bloco 3 trazem também o valor
invariante à permutação, max(a, 1−a). Os CSVs são lidos no contrato de 13/10/2026: execuções anteriores passam antes por
`renomear_assets_legado.py --aplicar` (não há fallback de nomes de ARQUIVO). `gerar_tabelas_bloco1._col` só
cobre nomes de COLUNA antigos em leitores pontuais (`fidelity` no oráculo, `problem_base` na síntese cruzada).

Uso:
    python src/gerar_tabelas.py                              # última _completa
    python src/gerar_tabelas.py resultados/execucao_... [--sem-png] [--sem-docx] [--saida tabelas]
O runner (`dissertacao_mestrado.main`) chama `gerar_tudo` ao final de cada execução.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execucao_io import _model_alias  # noqa: E402
from tabelas import Tabela, delta, esc, faixa_kappa, num  # noqa: E402
import gerar_tabelas_bloco1 as b1  # noqa: E402
from gerar_tabelas_bloco1 import (_col, acc_cell, fid_cell, marca_gamma, regime_label,  # noqa: E402
                                  NOTA_EXEMPLOS_BC, NOTA_GAMMA, NOTA_KAPPA, NOTA_PERMUTACAO,
                                  NOTA_PASSOS_A, NOTA_PASSOS_D)

BASE_DIR = b1.BASE_DIR
RESULTADOS_DIR = b1.RESULTADOS_DIR
_ESTRATEGIAS = [("easy", "Fácil"), ("hard", "Difícil"), ("mixed", "Mista"), ("random", "Aleatória")]
_ORDENS = [("class0_first", "classe 0 primeiro"), ("class1_first", "classe 1 primeiro"),
           ("shuffled", "embaralhada"), ("alternating", "alternada")]
# Nomes do orientador ("Problema 1/2/3") para os três oráculos principais; chave = `problem` do CSV.
# Desde 14/10/2026 os três rodam sobre as gaussianas do E ("Oráculo sobre E3" = euclidiano); "Oráculo sobre B"
# é o Problema 1 das execuções até 13/10/2026 (dados do B). Nunca os dois na mesma execução.
_ORACULO_NOMES = {"Oráculo sobre E3": ("Problema 1", "[1,0; 1,0]", "E (150 pts)"),
                  "Oráculo sobre B": ("Problema 1", "[1,0; 1,0]", "B (100 pts)"),
                  "Oráculo sobre E1": ("Problema 2", "[0,3; 1,5]", "E (150 pts)"),
                  "Oráculo sobre E2": ("Problema 3", "[1,5; 0,3]", "E (150 pts)")}
_ACUR_ESPEC = "Acur.\\ vs espec."
_NOTA_ACUR_ESPEC = "Acur.\\ vs espec.\\ = acurácia (LLM vs especialista): fração dos pontos de teste em que o LLM reproduz o rótulo do especialista."


def _inv_perm(s: pd.Series) -> pd.Series:
    """Acurácia invariante à permutação nome↔classe: max(a, 1 − a), elemento a elemento (NaN preservado)."""
    s = pd.to_numeric(s, errors="coerce")
    return pd.concat([s, 1.0 - s], axis=1).max(axis=1).where(s.notna())


# ───────────────────────── carga ─────────────────────────
def carregar_tudo(pasta: Path) -> Dict[str, pd.DataFrame]:
    """Último CSV de cada família → {chave: DataFrame}. Chaves ausentes = experimento não rodou.

    Lê SÓ os nomes de 13/10/2026 (sem fallback): uma execução anterior precisa ser migrada
    por ``renomear_assets_legado.py --aplicar`` (a função imprime o aviso e sai com código 1).
    """
    from renomear_assets_legado import tem_csv_antigo
    if tem_csv_antigo(pasta):
        print(f"execução anterior a 13/10/2026: rode `python src/renomear_assets_legado.py {pasta} --aplicar` antes")
        raise SystemExit(1)
    fam = {"ad_passos_2_a_5": "bloco1_ad_passos_2_a_5_*.csv", "g_passos_2_a_5": "bloco3_g_passos_2_a_5_*.csv",
           "bc_passo3": "bloco1_bc_passo3_consistencia_*.csv",
           "abc": "bloco1_abc_passos_2_a_4_*.csv", "a_passo4_perc_nnls": "bloco1_a_passo4_perceptron_vs_nnls_*.csv",
           "a_passo4_r3r4": "bloco1_a_passo4_r2_r3_r4_*.csv", "oraculo": "bloco1_oraculo_passos_1_a_3_*.csv",
           "d_passo5": "bloco1_d_passo5_teto_*.csv", "d_passo6_svm": "bloco1_d_passo6_svm_*.csv", "e": "bloco2_e_passos_2_3_*.csv",
           "e_passo4": "bloco2_e_passo4_diluicao_*.csv", "e_passo5": "bloco2_e_passo5_ordem_*.csv",
           "e_passo6": "bloco2_e_passo6_baselines_*.csv", "fg_passos_2_e_4": "bloco23_fg_passos_2_e_4_*.csv",
           "fg_aprendiz": "bloco23_fg_aprendiz_*.csv"}
    D: Dict[str, pd.DataFrame] = {}
    for k, pat in fam.items():
        fs = [f for f in sorted(pasta.glob(pat)) if "corrigido" not in f.name]
        if fs:
            D[k] = pd.read_csv(fs[-1])
    partes = []
    for f in sorted(pasta.glob("final_cross_linearity__*.csv")):
        if "corrigido" in f.name:
            continue
        df = pd.read_csv(f)
        df["_alias"] = f.stem.split("__")[-1]          # alias do modelo vem do nome do arquivo
        partes.append(df)
    if partes:
        D["cross"] = pd.concat(partes, ignore_index=True)
    return D


def modelos(D: Dict[str, pd.DataFrame]) -> List[str]:
    ms = set()
    for k, df in D.items():
        if isinstance(df, pd.DataFrame) and "model" in df.columns and k != "e_passo6":
            ms |= set(df["model"].dropna().unique())
    return sorted(ms)


def _ag(df: pd.DataFrame, chaves, cols) -> pd.DataFrame:
    return b1.agregar(df, chaves, cols)


def _st(r, col, casas=1, pct=True) -> str:
    return b1.fmt_stat(r, col, pct=pct, casas=casas)


def _base_abc(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Configuração base dos Problemas A, B e C: classes A/B, atributos x1/x2, prompt default, sem colapso."""
    return df[(df["model"] == model) & (df["class_0"] == "A") & (df["class_1"] == "B")
              & (df["feature_0"] == "x1") & (df["prompt_variant"] == "default") & (df["collapsed"] == 0)]


# ═════════════════════════ BLOCO 1 (extras) ═════════════════════════
def tab_a_passo4_r2_r3_r4(D, model) -> Optional[Tabela]:
    df = D.get("ad_passos_2_a_5")
    if df is None:
        return None
    sub = df[(df["model"] == model) & (df["problem"] == "A")]
    if sub.empty or sub["n_features"].nunique() < 2:
        return None
    ag = _ag(sub, ["regime", "k", "n_features", "algorithm"],
             ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "accuracy_teto_vs_true", "gamma", "margem_minima"])
    base = ag[(ag["n_features"] == 2) & (ag["algorithm"] == "perceptron")].set_index(["regime", "k"])["fidelity_vs_llm_mean"]
    rows, mids = [], []
    for nf in sorted(sub["n_features"].unique()):
        a = b1._ordem(ag[ag["n_features"] == nf])
        for (reg, k), g in a.groupby(["regime", "k"], sort=False):
            p = g[g["algorithm"] == "perceptron"].iloc[0] if (g["algorithm"] == "perceptron").any() else None
            n = g[g["algorithm"] == "nnls"].iloc[0] if (g["algorithm"] == "nnls").any() else None
            teto = reg == "passo5_teto"
            rows.append([f"R$^{int(nf)}$", b1.regime_label(reg, k) + (b1.marca_gamma(p) if p is not None else ""),
                         b1.fid_cell(p, reg) if p is not None else "—",
                         b1.acc_cell(p, reg) if p is not None else "—",
                         b1.fid_cell(n, reg) if n is not None else "—",
                         b1.acc_cell(n, reg) if n is not None else "—",
                         "—" if (p is None or teto) else delta(base.get((reg, k), np.nan), p["fidelity_vs_llm_mean"])])
        mids.append(len(rows) - 1)
    rod = ("Problema A (linear). O LLM vê só x1, x2; R$^3$ acrescenta x1·x2 e R$^4$ acrescenta x1², x2² apenas na "
           "estimação. Em problema linear espera-se ganho ~nulo com a dimensão. Fid.\\ (vs LLM) = fidelidade; "
           "Acur.\\ vs real = acurácia (vs rótulo real) da métrica, contexto; o teto (W ajustado ao rótulo real) só tem "
           "acurácia. $\\Delta$ = fidelidade Perceptron $-$ R$^2$. " + NOTA_PASSOS_A + " " + NOTA_GAMMA)
    return Tabela(slug="bloco1_A_passo4_R2_R3_R4", nn=4, modelo=model,
                  caption=f"Problema A, Passos 2 a 5 ({esc(model)}): métrica em R$^2$, R$^3$ e R$^4$ por regime.",
                  headers=["Espaço", "Regime", "Fid.\\ (vs LLM)", "Acur.\\ vs real", "Fid.\\ (vs LLM)", "Acur.\\ vs real", "$\\Delta$ fid.\\ vs R$^2$"],
                  rows=rows, align="llrrrrr", grupos=[("Perceptron", 3, 4), ("NNLS", 5, 6)],
                  rodape=rod, midrules_apos=set(mids[:-1]))


def tab_a_passo4_perceptron_vs_nnls(D, model) -> Optional[Tabela]:
    abc, alg = D.get("abc"), D.get("a_passo4_perc_nnls")
    if abc is None:
        return None
    p = _base_abc(abc, model)
    p = p[p["k_passo4_bc"] == 0]
    if p.empty:
        return None
    n = alg[(alg["model"] == model) & (alg["k_passo4_bc"] == 0)] if alg is not None else pd.DataFrame()
    if not n.empty and "algorithm" in n.columns:      # o CSV de comparação pode listar os dois estimadores
        so_nnls = n["algorithm"].astype(str).str.lower().str.startswith("nnls")
        n = n[so_nnls] if so_nnls.any() else n
    seeds = sorted(p["random_seed"].unique())

    def _g0(r) -> str:          # † quando o Perceptron não separou os rótulos (γ = 0 ou margem bruta mínima < 0) nessa semente
        if r is None:
            return ""
        g = r.get("gamma", np.nan); mm = r.get("margem_minima", np.nan)
        return "†" if ((g is not None and not pd.isna(g) and float(g) <= 0.0)
                       or (mm is not None and not pd.isna(mm) and float(mm) < 0.0)) else ""
    rows = []
    for sd in seeds:
        rp = p[p["random_seed"] == sd].iloc[0]
        rn = n[n["random_seed"] == sd].iloc[0] if (not n.empty and (n["random_seed"] == sd).any()) else None
        rows.append([str(sd), num(rp["fidelity_problem_a"]) + _g0(rp), num(rn["fidelity_problem_a"]) if rn is not None else "—",
                     num(rp["w_cosine_sim_nnls"], 3, pct=False),
                     num(rp["consistency_problem_b"]), num(rn["consistency_problem_b"]) if rn is not None else "—",
                     num(rp["consistency_problem_c"]), num(rn["consistency_problem_c"]) if rn is not None else "—",
                     num(rp["kappa_problem_b"], 3, pct=False), num(rp["kappa_problem_c"], 3, pct=False)])
    algum_g0 = "†" if (("gamma" in p.columns and (pd.to_numeric(p["gamma"], errors="coerce") <= 0).any())
                       or ("margem_minima" in p.columns and (pd.to_numeric(p["margem_minima"], errors="coerce") < 0).any())) else ""
    med = [["média", num(p["fidelity_problem_a"].mean()) + algum_g0, num(n["fidelity_problem_a"].mean()) if not n.empty else "—",
            num(p["w_cosine_sim_nnls"].mean(), 3, pct=False), num(p["consistency_problem_b"].mean()),
            num(n["consistency_problem_b"].mean()) if not n.empty else "—", num(p["consistency_problem_c"].mean()),
            num(n["consistency_problem_c"].mean()) if not n.empty else "—",
            num(p["kappa_problem_b"].mean(), 3, pct=False), num(p["kappa_problem_c"].mean(), 3, pct=False)]]
    rows += med
    # Desde 09/10/2026 as métricas do NNLS em B/C são PAREADAS (mesmos rótulos do LLM). Execuções anteriores
    # (coluna euclidean_consistency_b presente) vinham de uma RECOLETA de B/C e recebem a nota correspondente.
    pareado = alg is None or "euclidean_consistency_b" not in alg.columns
    nota_par = ("As colunas NNLS de B e C são PAREADAS: mesmos rótulos zero-shot do LLM para os dois estimadores "
                "(só a métrica muda)." if pareado else
                "Atenção: nesta execução (anterior a 09/10/2026) as colunas NNLS de B e C vêm de uma recoleta "
                "separada dos rótulos do LLM, e a diferença mistura estimador e flip a $T=0$.")
    rod = ("Zero-shot, configuração base (classes A/B, atributos x1/x2, prompt padrão). Fidelidade no A (Passo 4, sobre os "
           "rótulos zero-shot do Passo 2); consistência em B e C (Passo 3: só $\\hat{W}$ transferido, centróides locais). "
           + nota_par + " cos = cosseno entre os "
           "$\\hat{W}$ dos dois estimadores. " + NOTA_KAPPA + " $\\kappa$ do Perceptron. " + NOTA_GAMMA)
    return Tabela(slug="bloco1_A_passo4_perceptron_vs_nnls", nn=5, modelo=model,
                  caption=f"Perceptron Estruturado versus NNLS ({esc(model)}): fidelidade no A (Passo 4) e consistência em B e C (Passo 3), por semente.",
                  headers=["Semente", "Perc.", "NNLS", "cos", "Perc.", "NNLS", "Perc.", "NNLS", "$\\kappa$ B", "$\\kappa$ C"],
                  rows=rows, align="lrrrrrrrrr",
                  grupos=[("Fidelidade A (\\%)", 2, 3), ("Consist.\\ B (\\%)", 5, 6), ("Consist.\\ C (\\%)", 7, 8)],
                  rodape=rod, destaque={len(rows) - 1}, midrules_apos={len(rows) - 2})


def tab_bc_passo4_exemplos_da_metrica(D, model) -> Optional[Tabela]:
    abc = D.get("abc")
    if abc is None:
        return None
    p = _base_abc(abc, model)
    if p.empty or p["k_passo4_bc"].nunique() < 2:
        return None
    ag = _ag(p, ["k_passo4_bc"], ["consistency_problem_b", "kappa_problem_b", "f1_problem_b",
                                      "consistency_problem_c", "kappa_problem_c", "f1_problem_c"]).sort_values("k_passo4_bc")
    z = ag.iloc[0]
    rows = []
    for _, r in ag.iterrows():
        n = int(r["k_passo4_bc"])
        rows.append(["zero-shot" if n == 0 else f"{n}-shot",
                     _st(r, "consistency_problem_b"), num(r["kappa_problem_b_mean"], 3, pct=False), faixa_kappa(r["kappa_problem_b_mean"]),
                     delta(z["consistency_problem_b_mean"], r["consistency_problem_b_mean"]),
                     _st(r, "consistency_problem_c"), num(r["kappa_problem_c_mean"], 3, pct=False), faixa_kappa(r["kappa_problem_c_mean"]),
                     delta(z["consistency_problem_c_mean"], r["consistency_problem_c_mean"])])
    rod = ("Problemas B e C, Passo 4 (hipótese H2): os k exemplos mostrados ao LLM em B e C são rotulados pela métrica zero-shot do A "
           "(seleção por margem, k/2 por classe) e o LLM classifica os demais pontos. " + NOTA_EXEMPLOS_BC + " "
           "Consist.\\ = consistência (\\%). " + NOTA_KAPPA + " Médias das sementes (mín–máx); "
           "$\\Delta$ em relação ao zero-shot. Perceptron, configuração base.")
    return Tabela(slug="bloco1_BC_passo4_exemplos_da_metrica", nn=9, modelo=model,
                  caption=f"Problemas B e C, Passo 4 ({esc(model)}): few-shot com exemplos rotulados pela métrica do A (H2).",
                  headers=["Regime", "Consist.", "$\\kappa$", "Faixa", "$\\Delta$", "Consist.", "$\\kappa$", "Faixa", "$\\Delta$"],
                  rows=rows, align="lrrlrrrlr", grupos=[("Problema B", 2, 5), ("Problema C", 6, 9)], rodape=rod)


def tab_d_zero_vs_exemplos(D, model) -> Optional[Tabela]:
    df = D.get("ad_passos_2_a_5")
    if df is None:
        return None
    sub = df[(df["model"] == model) & (df["problem"] == "D") & (df["algorithm"] == "perceptron")].copy()
    if sub.empty:
        return None
    if "collapsed" in sub.columns:
        # Colapso (collapsed=1): a acurácia do LLM gravada (50%) e a partição 150/0 não são medidas
        # de um critério; ficam fora das médias como a fidelidade (NaN → "—").
        sub.loc[sub["collapsed"] == 1, ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "llm_accuracy_vs_true"]] = np.nan
    ag = _ag(sub, ["n_features", "regime", "k"], ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "llm_accuracy_vs_true"])
    rows, mids = [], []
    for nf in sorted(ag["n_features"].unique()):
        a = ag[ag["n_features"] == nf].set_index(["regime", "k"])
        if ("passo2_zero_shot", 0) not in a.index:
            continue
        z = a.loc[("passo2_zero_shot", 0)]
        for k in sorted(int(k) for (r, k) in a.index if r == "passo3_exemplos_rotulo_real"):
            f = a.loc[("passo3_exemplos_rotulo_real", k)]
            rows.append([f"R$^{int(nf)}$", f"{k}-shot", _st(z, "fidelity_vs_llm"), _st(f, "fidelity_vs_llm"),
                         delta(z["fidelity_vs_llm_mean"], f["fidelity_vs_llm_mean"]),
                         _st(z, "accuracy_metric_vs_true_full"), _st(f, "accuracy_metric_vs_true_full"),
                         delta(z["accuracy_metric_vs_true_full_mean"], f["accuracy_metric_vs_true_full_mean"]),
                         _st(z, "llm_accuracy_vs_true"), _st(f, "llm_accuracy_vs_true"),
                         delta(z["llm_accuracy_vs_true_mean"], f["llm_accuracy_vs_true_mean"])])
        mids.append(len(rows) - 1)
    rod = ("Meia-lua, Perceptron. Fidelidade (vs LLM); acurácia (vs rótulo real) da métrica e do LLM como contexto. "
           "Colapsos (e quase-colapsos) do zero-shot não entram em nenhuma média, nem na acurácia do LLM "
           "(``—'' quando todas as sementes colapsaram; detalhe na bloco1\\_table\\_10). "
           + NOTA_PERMUTACAO + " $\\Delta$ = few-shot $-$ zero-shot. " + NOTA_PASSOS_D)
    return Tabela(slug="bloco1_D_passo2_vs_passo3", nn=11, modelo=model,
                  caption=f"Problema D, meia-lua, Passos 2 a 4 ({esc(model)}): zero-shot versus few-shot por espaço.",
                  headers=["Espaço", "Few-shot", "zero", "few", "$\\Delta$", "zero", "few", "$\\Delta$", "zero", "few", "$\\Delta$"],
                  rows=rows, align="llrrrrrrrrr",
                  grupos=[("Fidelidade (vs LLM)", 3, 5), ("Acur.\\ da métrica vs real", 6, 8), ("Acur.\\ do LLM vs real", 9, 11)],
                  rodape=rod, midrules_apos=set(mids[:-1]), tamanho=r"\footnotesize")


def tab_ad_por_regime(D, model) -> Optional[Tabela]:
    df = D.get("ad_passos_2_a_5")
    if df is None:
        return None
    sub = df[(df["model"] == model) & (df["n_features"] == 2) & (df["algorithm"] == "perceptron")].copy()
    if sub["problem"].nunique() < 2:
        return None
    if "collapsed" in sub.columns:
        # Colapso (collapsed=1): a acurácia do LLM gravada não mede um critério; sai das médias como a
        # fidelidade (NaN → "—"), igual às tabelas 10 e 11 (13/10/2026).
        sub.loc[sub["collapsed"] == 1, ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "llm_accuracy_vs_true"]] = np.nan
    ag = b1.agregar(sub, ["problem", "regime", "k"], ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "accuracy_teto_vs_true",
                                                      "llm_accuracy_vs_true", "collapsed", "gamma", "margem_minima"])
    A = ag[ag["problem"] == "A"].set_index(["regime", "k"]); Dd = ag[ag["problem"] == "D"].set_index(["regime", "k"])
    rows = []
    for idx in b1._ordem(ag[ag["problem"] == "A"]).set_index(["regime", "k"]).index:
        a = A.loc[idx]; d = Dd.loc[idx] if idx in Dd.index else None
        reg = idx[0]
        g = lambda r, c: _st(r, c) if r is not None else "—"
        rows.append([b1.regime_label(*idx) + ("†" if (b1.marca_gamma(a) or (d is not None and b1.marca_gamma(d))) else ""),
                     b1.fid_cell(a, reg), b1.fid_cell(d, reg) if d is not None else "—",
                     b1.acc_cell(a, reg), b1.acc_cell(d, reg) if d is not None else "—",
                     "—" if reg == "passo5_teto" else g(a, "llm_accuracy_vs_true"),
                     "—" if reg == "passo5_teto" else g(d, "llm_accuracy_vs_true")])
    rod = ("Perceptron, R$^2$. Mesmos Passos 2 a 5 nos dois problemas (150 pontos, exemplos do rótulo real, sem split). "
           "Fidelidade (vs LLM); acurácia (vs rótulo real) da métrica e do LLM como contexto; o teto (W ajustado ao "
           "rótulo real) só tem acurácia. " + NOTA_PASSOS_A + " " + NOTA_PERMUTACAO + " " + NOTA_GAMMA + " "
           + b1.NOTA_COLAPSO)
    return Tabela(slug="bloco1_AD_por_passo", nn=12, modelo=model,
                  caption=f"Problema A (linear) versus Problema D (meia-lua), Passos 2 a 5 ({esc(model)}), por regime.",
                  headers=["Regime", "A", "D", "A", "D", "A", "D"], rows=rows, align="lrrrrrr",
                  grupos=[("Fidelidade (vs LLM)", 2, 3), ("Acur.\\ da métrica vs real", 4, 5), ("Acur.\\ do LLM vs real", 6, 7)],
                  rodape=rod, midrules_apos={len(rows) - 2} if len(rows) > 1 else set())


def _tab_vies(D, model, fator: str) -> Optional[Tabela]:
    abc = D.get("abc")
    if abc is None:
        return None
    df = abc[(abc["model"] == model) & (abc["collapsed"] == 0)]
    if fator == "classe":
        df = df[(df["feature_0"] == "x1") & (df["prompt_variant"] == "default")]
        chave, nome = ["class_0", "class_1"], lambda r: f"{r['class_0']} / {r['class_1']}"
        slug, nn, cap, ref = "bloco1_vies_nomes_classe", 14, "nomes de classe", "A / B"
    elif fator == "atributo":
        df = df[(df["class_0"] == "A") & (df["prompt_variant"] == "default")]
        chave, nome = ["feature_0", "feature_1"], lambda r: f"{r['feature_0']} / {r['feature_1']}"
        slug, nn, cap, ref = "bloco1_vies_nomes_atributo", 15, "nomes de atributo", "x1 / x2"
    else:
        df = df[(df["class_0"] == "A") & (df["feature_0"] == "x1")]
        chave, nome = ["prompt_variant"], lambda r: {"default": "padrão", "geometric": "geométrica", "cot": "chain-of-thought", "tabular": "tabular"}.get(r["prompt_variant"], r["prompt_variant"])
        slug, nn, cap, ref = "bloco1_vies_variantes_prompt", 16, "variantes de prompt", "padrão"
    # k das linhas: os valores de k_passo4_bc presentes nas VARIANTES (K_VIESES na completa,
    # [0, k_rapido] no smoke); a configuração de referência roda a grade inteira e só entra nesses k.
    # Nomes de atributo: só zero-shot (a métrica do A vem do Passo 2 e o experimento não tem contraparte few-shot).
    if df.empty:
        return None
    if fator == "atributo":
        ks = [0]
    else:
        eh_ref = df.apply(lambda r: nome(r) == ref, axis=1)
        fonte_k = df[~eh_ref] if (~eh_ref).any() else df
        ks = sorted(int(k) for k in fonte_k["k_passo4_bc"].dropna().unique())
    ks_few = [k for k in ks if k > 0]
    df = df[df["k_passo4_bc"].isin(ks)].copy()
    if df.empty:
        return None
    df["llm_acc_a_inv"] = _inv_perm(df["llm_accuracy_problem_a"])
    ag = _ag(df, chave + ["k_passo4_bc"], ["fidelity_problem_a", "llm_accuracy_problem_a", "llm_acc_a_inv",
                                               "consistency_problem_b", "kappa_problem_b", "consistency_problem_c",
                                               "kappa_problem_c", "w_ratio"])
    com_inv = fator == "classe"
    n_seeds_max = int(df["random_seed"].nunique()) if "random_seed" in df.columns else 0
    n_seeds_por = (df.groupby(chave + ["k_passo4_bc"])["random_seed"].nunique().to_dict()
                   if "random_seed" in df.columns else {})
    # 15/10/2026: colapso do LLM em B ou C numa semente (collapsed_b/c) deixa a consistência NaN só nessa
    # semente; as células de B/C ganham o sufixo (n=…) com o nº de sementes que entram na média.
    def _n_bc(r, n, prob):
        colc = f"collapsed_{prob}"
        if colc not in df.columns:
            return None
        m = (df["k_passo4_bc"] == n)
        for c in chave:
            m &= (df[c] == r[c])
        sub = df[m]
        ok = int((sub[colc] != 1).sum())
        return f" (n={ok})" if len(sub) and ok < len(sub) else ""
    rows, destaque = [], set()
    for n in sorted(ag["k_passo4_bc"].unique()):
        a = ag[ag["k_passo4_bc"] == n]
        for _, r in a.iterrows():
            lab = nome(r)
            _chave = tuple(r[c] for c in chave) + (n,)
            _ns = n_seeds_por.get(_chave if len(chave) > 1 else (_chave[0], n), n_seeds_max)
            _suf = f" (n={_ns})" if n_seeds_max and _ns < n_seeds_max else ""
            _sb, _sc = (_n_bc(r, n, "b") or ""), (_n_bc(r, n, "c") or "")
            rows.append([esc(lab), ("zero-shot" if n == 0 else f"{int(n)}-shot") + _suf, _st(r, "fidelity_problem_a"),
                         num(r["llm_accuracy_problem_a_mean"])]
                        + ([num(r["llm_acc_a_inv_mean"])] if com_inv else [])
                        + [num(r["w_ratio_mean"], 2, pct=False),
                           num(r["consistency_problem_b_mean"]) + _sb, num(r["kappa_problem_b_mean"], 3, pct=False) + _sb,
                           num(r["consistency_problem_c_mean"]) + _sc, num(r["kappa_problem_c_mean"], 3, pct=False) + _sc])
            if lab == ref:
                destaque.add(len(rows) - 1)
    # Variantes cujo A colapsou em TODAS as sementes (sem linha válida): entram com "—" e o nº de sementes (14/10/2026)
    col = abc[(abc["model"] == model) & (abc["collapsed"] == 1)]
    if fator == "classe":
        col = col[(col["feature_0"] == "x1") & (col["prompt_variant"] == "default")]
    elif fator == "atributo":
        col = col[(col["class_0"] == "A") & (col["prompt_variant"] == "default")]
    else:
        col = col[(col["class_0"] == "A") & (col["feature_0"] == "x1")]
    validas = {tuple(r[c] for c in chave) for _, r in df.iterrows()}
    for chv, grp in col.groupby(chave):
        chv = chv if isinstance(chv, tuple) else (chv,)
        if chv in validas:
            continue
        ns_col = int(grp["random_seed"].nunique()) if "random_seed" in grp.columns else len(grp)
        rows.append([esc(nome(grp.iloc[0])), f"zero-shot (colapso do A em {ns_col}/{n_seeds_max or ns_col} sementes)"]
                    + ["—"] * (7 if com_inv else 6))
    rotulo_k = ", ".join(f"{k}-shot" for k in ks_few)   # "10-shot" na completa, "4-shot" no smoke
    # Passos de B/C citados: Passo 3 (zero-shot contra a métrica) e, se há linhas few-shot, também o Passo 4
    passos_bc = "Passos 3 e 4" if ks_few else "Passo 3"
    rod = (f"Problema A, Passo 2, e B/C, {passos_bc}, por {cap}; os demais fatores ficam na configuração base. "
           "Médias das sementes (fidelidade com mín–máx). Fid.\\ A = fidelidade (vs LLM) no A; Acur.\\ LLM vs real A = "
           "acurácia (vs rótulo real) do LLM no A, contexto; $w_1/w_2$ = razão da métrica estimada (Perceptron). "
           "Fid.\\ A, Acur.\\ LLM vs real A e $w_1/w_2$ vêm do Problema A (Passo 4 sobre os rótulos do Passo 2), sempre zero-shot"
           + (f" (iguais nas linhas zero e {rotulo_k})" if ks_few else "") + ". " + NOTA_KAPPA + " "
           "Em negrito: a configuração de referência. (n=…) no Regime = número de sementes sem colapso do A na linha, quando "
           "menor que o total; (n=…) numa célula de B ou C = sementes sem colapso do LLM naquele problema (collapsed\\_b/c); "
           "as médias vêm só delas.")
    if com_inv:
        rod += (" Acur.\\ vs real (inv.\\ perm.) = max($a$, 1 $-$ $a$): nos pares invertidos (e em zero-shot) a correspondência "
                "entre nome e classe é arbitrária, e só o valor invariante à permutação compara as variantes.")
    else:
        rod += " " + NOTA_PERMUTACAO
    if ks_few:
        rod += f" {rotulo_k.capitalize()}: B e C, Passo 4, exemplos rotulados pela métrica zero-shot do A (H2). " + NOTA_EXEMPLOS_BC
    headers = (["Variante", "Regime", "Fid.\\ A", "Acur.\\ LLM vs real A"] + (["Acur.\\ vs real (inv.\\ perm.)"] if com_inv else [])
               + ["$w_1/w_2$", "Consist.\\ B", "$\\kappa$ B", "Consist.\\ C", "$\\kappa$ C"])
    return Tabela(slug=slug, nn=nn, modelo=model, caption=f"Problema A, Passo 2, e B/C, {passos_bc}, com {cap} ({esc(model)}): fidelidade, direção de $\\hat{{W}}$ e consistência.",
                  headers=headers, rows=rows, align="llrr" + ("r" if com_inv else "") + "rrrrr", rodape=rod, destaque=destaque,
                  midrules_apos={i for i in range(len(rows) - 1) if rows[i][1] != rows[i + 1][1]})


def tab_oraculo(D) -> Optional[Tabela]:
    df = D.get("oraculo")
    if df is None:
        return None
    acc = _col(df, "accuracy", "fidelity")           # `accuracy` (acurácia vs rótulo do oráculo); `fidelity` em CSVs antigos
    if acc is None:
        return None
    ag = _ag(df, ["problem", "algorithm"], [acc, "cosine_similarity", "recovered_w_norm_0", "recovered_w_norm_1", "true_w_0", "true_w_1"])

    # Só os Problemas 1/2/3 (E3 ou B, E1, E2); linhas de outras configurações (execuções anteriores a
    # 13/10/2026 tinham gaussianas anisotrópicas extras) são ignoradas.
    rows, mids = [], []
    p1 = "Oráculo sobre E3" if (ag["problem"] == "Oráculo sobre E3").any() else "Oráculo sobre B"
    for prob in _ORACULO_NOMES:
        a = ag[ag["problem"] == prob]
        if a.empty:
            continue
        nome, wtxt, pts = _ORACULO_NOMES[prob]
        for _, r in a.sort_values("algorithm", ascending=False).iterrows():
            rows.append([nome, wtxt, pts, r["algorithm"],
                         f"[{num(r['recovered_w_norm_0_mean'], 2, pct=False)}; {num(r['recovered_w_norm_1_mean'], 2, pct=False)}]",
                         _st(r, "cosine_similarity", 3, pct=False), _st(r, acc)])
        mids.append(len(rows) - 1)
    if not rows:
        return None
    p1_txt = ("Problema 1 = Oráculo sobre E3 (W = [1; 1]); Problema 2 = Oráculo sobre E1 (W = [0,3; 1,5]) e Problema 3 = "
              "Oráculo sobre E2 (W = [1,5; 0,3]), os três sobre os MESMOS pontos, as gaussianas do Problema E (centróides "
              "$(-1{,}5; 1)$ e $(1{,}5; -1)$, que diferem nas duas coordenadas; nos centróides do Problema A, com $x_2 = 0$, "
              "$w_2$ não seria identificável)."
              if p1 == "Oráculo sobre E3" else
              "Problema 1 = Oráculo sobre B (W = [1; 1], pontos do Problema B); Problema 2 = Oráculo sobre E1 (W = [0,3; 1,5]) e "
              "Problema 3 = Oráculo sobre E2 (W = [1,5; 0,3]), ambos nos pontos do Problema E.")
    rod = ("Teste-oráculo (não há LLM): Passo 1, os rótulos vêm da regra do centróide mais próximo sob o W verdadeiro; "
           "Passo 2, os dois estimadores tentam recuperá-lo; Passo 3, comparação por acurácia e cosseno. " + p1_txt
           + " W recuperado normalizado (só a direção "
           "importa). Acurácia (\\%) = acurácia da métrica recuperada contra o rótulo do oráculo. Médias das sementes (mín–máx).")
    return Tabela(slug="bloco1_oraculo_passos_1_a_3", nn=17,
                  caption=f"Oráculo sobre {'E3' if p1 == 'Oráculo sobre E3' else 'B'}, E1 e E2, Passos 1 a 3: recuperação de uma métrica conhecida pelos dois estimadores.",
                  headers=["Problema", "W verdadeiro", "Pontos", "Algoritmo", "W recuperado (norm. ΣW)", "Cosseno", "Acurácia (\\%)"],
                  rows=rows, align="lllllrr", rodape=rod, midrules_apos=set(mids[:-1]))


def tab_d_passo6_svm(D, model) -> Optional[Tabela]:
    """Tabela 19 do Bloco 1 (16/10/2026): números da figura bloco1_graphic_14 (Problema D, Passo 6)."""
    df = D.get("d_passo6_svm")
    if df is None or df.empty or "model" not in df.columns:
        return None
    sub = df[df["model"] == model].sort_values("random_seed")
    if sub.empty:
        return None

    def pc(v, casas=1):
        return "—" if pd.isna(v) else f"{100 * float(v):.{casas}f}".replace(".", ",")

    def inteiro(v):
        return "—" if pd.isna(v) else f"{int(v)}"

    rows = []
    for _, r in sub.iterrows():
        col = int(r.get("collapsed", 0)) == 1
        rows.append([str(int(r["random_seed"])), f"{int(r['n_classe_0_llm'])}/{int(r['n_classe_1_llm'])}" + (" (colapso)" if col else ""),
                     pc(r["llm_accuracy_vs_true"]), pc(r["svm_real_accuracy_vs_true"]), inteiro(r["svm_real_n_errors"]),
                     pc(r["svm_llm_fidelity_vs_llm"]), pc(r["svm_llm_accuracy_vs_true"]), inteiro(r["svm_llm_n_errors_vs_true"]),
                     ("—" if pd.isna(r["metric_n_features"]) else f"R$^{int(r['metric_n_features'])}$"),
                     pc(r["metric_fidelity_vs_llm"]), pc(r["metric_accuracy_vs_true"])])
    m = sub.mean(numeric_only=True)
    rows.append(["média", "", pc(m["llm_accuracy_vs_true"]), pc(m["svm_real_accuracy_vs_true"]), ("—" if pd.isna(m["svm_real_n_errors"]) else f"{m['svm_real_n_errors']:.1f}".replace(".", ",")),
                 pc(m["svm_llm_fidelity_vs_llm"]), pc(m["svm_llm_accuracy_vs_true"]), ("—" if pd.isna(m["svm_llm_n_errors_vs_true"]) else f"{m['svm_llm_n_errors_vs_true']:.1f}".replace(".", ",")),
                 "", pc(m["metric_fidelity_vs_llm"]), pc(m["metric_accuracy_vs_true"])])
    n_col = int(sub["collapsed"].sum()) if "collapsed" in sub.columns else 0
    return Tabela(slug="bloco1_D_passo6_svm", nn=19, modelo=model,
                  caption=f"Problema D, meia-lua, Passo 6 ({esc(model)}): SVM RBF treinado no rótulo real versus SVM RBF treinado nos rótulos zero-shot do LLM (Passo 2), nos 150 pontos; números da figura bloco1\\_graphic\\_14.",
                  headers=["Semente", "Partição LLM (n0/n1)", "Acur. LLM vs real", "Acur. vs real", "erros (de 150)", "Fid. (vs LLM)", "Acur. vs real", "erros vs real",
                           "Espaço", "Fid. (vs LLM)", "Acur. vs real"],
                  rows=rows, align="llr" + "rr" + "rrr" + "crr",
                  grupos=[("SVM RBF no rótulo real", 4, 5), ("SVM RBF nos rótulos do LLM", 6, 8), ("Métrica diagonal (Passo 4, maior fid.)", 9, 11)],
                  destaque={len(rows) - 1},
                  rodape=("SVM RBF com kernel gaussiano (gamma = `scale', C = 1), ajustado e avaliado nos mesmos 150 pontos: o treinado no rótulo real é a régua "
                          "de fronteira não linear; o treinado nos rótulos do LLM é a fronteira efetiva do LLM. Fid. (vs LLM) = fração dos pontos em que o SVM "
                          "reproduz o rótulo do LLM; Acur. vs real = fração de concordância com o rótulo real. A métrica diagonal é a de maior fidelidade entre "
                          "R$^2$, R$^3$ e R$^4$ no zero-shot (Passo 4). "
                          + (f"Em {n_col} semente(s) o LLM colapsou (classe única ou minoritária com menos de 5 pontos): sem SVM nem métrica, só a régua. " if n_col else "")
                          + "Acurácias vs real são literais: em zero-shot a correspondência entre nome e classe é arbitrária (o valor invariante à "
                            "permutação é max($a$, 1 $-$ $a$)); a fidelidade não depende disso. Não há conjunto de teste: a figura e a tabela "
                            "descrevem a separação induzida nos 150 pontos, não generalização."))


def tab_d_passo5_teto(D) -> Optional[Tabela]:
    df = D.get("d_passo5")
    if df is None:
        return None
    ag = _ag(df, ["n_features", "algorithm"], ["accuracy_vs_true", "n_errors", "w_recovered_0", "w_recovered_1", "w_recovered_2", "w_recovered_3"])
    rows = []
    for _, r in ag.sort_values(["n_features", "algorithm"], ascending=[True, False]).iterrows():
        nf = int(r["n_features"])
        w = np.array([r.get(f"w_recovered_{i}_mean", np.nan) for i in range(nf)], float)
        w = w / np.linalg.norm(w) if np.isfinite(w).all() and np.linalg.norm(w) > 0 else w   # unitário, como na tabela 10
        rows.append([f"R$^{nf}$", {"perceptron": "Perceptron", "nnls": "NNLS"}.get(r["algorithm"], r["algorithm"]),
                     _st(r, "accuracy_vs_true"), num(r["n_errors_mean"], 1, pct=False),
                     "[" + "; ".join(num(v, 2, pct=False) for v in w) + "]"])
    rod = ("W ajustado diretamente ao rótulo real da meia-lua (150 pontos): teto da métrica diagonal no Problema D, Passo 5 "
           "(0 chamadas; equivale ao regime ``teto'' da bloco1\\_table\\_10). Acurácia (vs rótulo real) da métrica ajustada; "
           "n erros = número médio de pontos errados em 150 (contagem auxiliar). R$^3$ acrescenta x1·x2, R$^4$ acrescenta "
           "x1², x2². Médias das sementes (mín–máx).")
    return Tabela(slug="bloco1_D_passo5_teto", nn=18,
                  caption="Problema D, Passo 5 (teto): acurácia da métrica diagonal ajustada ao rótulo real da meia-lua, em R$^2$, R$^3$ e R$^4$.",
                  headers=["Espaço", "Algoritmo", "Acurácia (vs rótulo real)", "n erros (de 150)", "$\\hat{W}$ (unitário)"], rows=rows, align="llrrl", rodape=rod)


# ═════════════════════════ BLOCO 2 ═════════════════════════
def _fe(D, model, expert=None) -> pd.DataFrame:
    df = D.get("e")
    if df is None:
        return pd.DataFrame()
    sub = df[(df["model"] == model) & df["example_strategy"].isin([e for e, _ in _ESTRATEGIAS])]
    return sub[sub["expert_name"] == expert] if expert else sub


def tab_e_passo3_estrategia_x_k(D, model, expert: str) -> Optional[Tabela]:
    sub = _fe(D, model, expert)
    if sub.empty:
        return None
    w = sub.iloc[0]
    wtxt = f"[{num(w['expert_w_0'], 1, pct=False)}; {num(w['expert_w_1'], 1, pct=False)}]"
    ag = _ag(sub, ["k", "example_strategy"], ["accuracy_llm_vs_expert", "kappa_llm_vs_expert"])
    rows = []
    for n in sorted(ag["k"].unique()):
        a = ag[ag["k"] == n].set_index("example_strategy")
        if n == 0:
            r = a.iloc[0]
            rows.append(["zero-shot (único valor)", _st(r, "accuracy_llm_vs_expert"), num(r["kappa_llm_vs_expert_mean"], 3, pct=False)] + [""] * 6)
            continue
        cells = [f"{int(n)}-shot"]
        for e, _ in _ESTRATEGIAS:
            r = a.loc[e] if e in a.index else None
            cells += [_st(r, "accuracy_llm_vs_expert") if r is not None else "—",
                      num(r["kappa_llm_vs_expert_mean"], 3, pct=False) if r is not None else "—"]
        rows.append(cells)
    hdr = ["k"] + [c for _ in _ESTRATEGIAS for c in (_ACUR_ESPEC, "$\\kappa$")]
    grupos = [(nome, 2 + 2 * i, 3 + 2 * i) for i, (_, nome) in enumerate(_ESTRATEGIAS)]
    rod = (f"Especialista {expert}, W = {wtxt}: rotula o Problema E pelo centróide mais próximo (Passo 1); o LLM classifica em "
           "zero-shot (Passo 2) e, no Passo 3, recebe k exemplos escolhidos pela estratégia e classifica os demais. "
           + _NOTA_ACUR_ESPEC + " Valores em \\% (mín–máx); "
           + NOTA_KAPPA + " Médias das sementes e sorteios. Em zero-shot não há exemplo a selecionar: valor único.")
    return Tabela(slug=f"bloco2_{expert}_passo3_estrategia_x_k", nn={"E1": 1, "E2": 2, "E3": 3}.get(expert, 0), modelo=model,
                  caption=f"Problema {expert}, Passos 2 e 3 ({esc(model)}): aprendizado em contexto por estratégia de seleção e k.",
                  headers=hdr, rows=rows, align="l" + "rr" * 4, grupos=grupos, rodape=rod)


def tab_e_zero_vs_exemplos(D, model) -> Optional[Tabela]:
    sub = _fe(D, model)
    if sub.empty:
        return None
    rows = []
    for ex in sorted(sub["expert_name"].unique()):
        s = sub[sub["expert_name"] == ex]
        z = s[s["k"] == 0]["accuracy_llm_vs_expert"].mean()
        fs = s[s["k"] > 0]
        if fs.empty:
            continue
        ag = fs.groupby(["example_strategy", "k"])["accuracy_llm_vs_expert"].mean()
        (be, bn), bv = ag.idxmax(), ag.max()
        media_few = ag.mean()      # média das médias por (estratégia, k): a aleatória (3 sorteios) não pesa 3×
        w = s.iloc[0]
        rows.append([ex, f"[{num(w['expert_w_0'], 1, pct=False)}; {num(w['expert_w_1'], 1, pct=False)}]", num(z),
                     num(media_few), delta(z, media_few),
                     num(bv), f"{dict(_ESTRATEGIAS)[be]}, k = {int(bn)}", delta(z, bv),
                     num(s["accuracy_expert_vs_gt"].mean()) if "accuracy_expert_vs_gt" in s.columns else "—"])
    rod = ("Problema E, Passos 2 (zero-shot) e 3 (few-shot). Acurácia (LLM vs especialista), em \\%. Média few-shot = média das médias por (estratégia, k), k > 0, de "
           "modo que cada célula (estratégia, k) pesa o mesmo, independentemente do número de sorteios. "
           "Melhor few-shot = maior média (estratégia, k). Última coluna: acurácia (vs rótulo real) do próprio "
           "especialista nas gaussianas do Problema E, contexto.")
    return Tabela(slug="bloco2_E_passo2_vs_passo3", nn=4, modelo=model,
                  caption=f"Problemas E1, E2 e E3, Passos 2 e 3 ({esc(model)}): zero-shot versus few-shot por especialista.",
                  headers=["Especialista", "W", "zero-shot", "média few-shot", "$\\Delta$", "melhor few-shot", "(estratégia, k)", "$\\Delta$",
                           "Acur.\\ do especialista vs real"],
                  rows=rows, align="llrrrrlrr", grupos=[("Acur.\\ vs espec.\\ (\\%)", 3, 6)], rodape=rod)


def tab_e_passo3_h4(D, model) -> Optional[Tabela]:
    sub = _fe(D, model)
    sub = sub[sub["k"] > 0] if not sub.empty else sub
    if sub.empty:
        return None
    rows = []
    for ex in sorted(sub["expert_name"].unique()):
        s = sub[sub["expert_name"] == ex].groupby("example_strategy")["accuracy_llm_vs_expert"].mean()
        ordem = " > ".join(dict(_ESTRATEGIAS)[e] for e in s.sort_values(ascending=False).index)
        rows.append([ex] + [num(s.get(e, np.nan)) for e, _ in _ESTRATEGIAS] + [ordem, delta(s.get("easy", np.nan), s.get("hard", np.nan))])
    rod = ("Problema E, Passo 3. Acurácia (LLM vs especialista), em \\%, média sobre os $k$ do few-shot, sementes e sorteios. H4 (a priori: "
           "difícil > fácil) é refutada quando $\\Delta$ difícil$-$fácil é negativo.")
    return Tabela(slug="bloco2_E_passo3_H4", nn=5, modelo=model,
                  caption=f"Problemas E1, E2 e E3, Passo 3 ({esc(model)}): média few-shot por estratégia de seleção (hipótese H4).",
                  headers=["Especialista"] + [n for _, n in _ESTRATEGIAS] + ["Ordem", "$\\Delta$ difícil $-$ fácil"],
                  rows=rows, align="lrrrrlr", rodape=rod)


def tab_e_passo6_baselines(D, model) -> Optional[Tabela]:
    bl, fe = D.get("e_passo6"), _fe(D, model)
    if bl is None or fe.empty:
        return None
    n_max = int(bl["k"].max())
    b = bl[(bl["k"] == n_max) & (bl["example_strategy"] == "mixed")]
    rows = []
    for ex in sorted(fe["expert_name"].unique()):
        llm = fe[(fe["expert_name"] == ex) & (fe["k"] == n_max)]
        if llm.empty:
            continue
        mix = llm[llm["example_strategy"] == "mixed"]["accuracy_llm_vs_expert"].mean()
        best = llm.groupby("example_strategy")["accuracy_llm_vs_expert"].mean()
        bb = b[b["expert_name"] == ex].groupby("model")["accuracy_vs_expert"].mean() if "model" in b.columns else pd.Series()
        cls = {c: bb.get(c, np.nan) for c in bb.index}
        rows.append([ex, num(mix), f"{num(best.max())} ({dict(_ESTRATEGIAS).get(best.idxmax(), best.idxmax())})"]
                    + [num(v) for _, v in sorted(cls.items())])
    _PT = {"Logistic Regression": "Regressão Logística"}   # mesmos nomes das figuras (CLASSIFICADOR_PT)
    nomes = sorted({c for c in (b.groupby("model").size().index if "model" in b.columns else [])})
    rod = (f"Problema E, Passos 3 (LLM) e 6 (baselines). Acurácia (LLM vs especialista), em \\%, com k = {n_max} exemplos (maior k da grade). Os classificadores "
           "clássicos (k-NN, Regressão Logística, SVM-RBF) são treinados nos MESMOS k exemplos da estratégia mista e "
           "avaliados no MESMO conjunto de teste; não dependem do LLM. Coluna LLM (mista): mesma estratégia dos baselines.")
    return Tabela(slug="bloco2_E_passo6_baselines", nn=6, modelo=model,
                  caption=f"Problema E, Passos 3 e 6 ({esc(model)}): LLM com {n_max} exemplos versus classificadores clássicos.",
                  headers=["Especialista", "LLM (mista)", "LLM (melhor estratégia)"] + [esc(_PT.get(n, n)) for n in nomes],
                  rows=rows, align="lrl" + "r" * len(nomes), grupos=[("Acur.\\ vs espec.\\ (\\%)", 2, 3 + len(nomes))], rodape=rod)


def tab_e_passo4_diluicao(D, model) -> Optional[Tabela]:
    df = D.get("e_passo4")
    if df is None:
        return None
    sub = df[df["model"] == model]
    if sub.empty:
        return None
    ag = _ag(sub, ["k", "example_strategy"], ["accuracy_llm_vs_expert", "kappa_llm_vs_expert", "f1_llm_vs_expert"]).sort_values("k")
    rows = []
    for _, r in ag.iterrows():
        m = re.search(r"(\d+)hard_(\d+)easy", str(r["example_strategy"]))
        nh, ne = (m.group(1), m.group(2)) if m else ("?", "?")
        rows.append([f"{int(r['k'])}", nh, ne, _st(r, "accuracy_llm_vs_expert"), num(r["kappa_llm_vs_expert_mean"], 3, pct=False), num(r["f1_llm_vs_expert_mean"])])
    rod = ("Problema E1, Passo 4. Exemplos difíceis fixos (menor margem) e fáceis acrescentados progressivamente. "
           + _NOTA_ACUR_ESPEC + " " + NOTA_KAPPA + " F1-Macro em \\%. Médias das sementes (mín–máx).")
    return Tabela(slug="bloco2_E1_passo4_diluicao", nn=7, modelo=model, caption=f"Problema E1, Passo 4 ({esc(model)}): experimento de diluição.",
                  headers=["k", "difíceis", "fáceis", _ACUR_ESPEC, "$\\kappa$", "F1-Macro"], rows=rows, align="rrrrrr", rodape=rod)


def tab_e_passo5_ordem(D, model) -> Optional[Tabela]:
    df = D.get("e_passo5")
    if df is None:
        return None
    sub = df[df["model"] == model]
    if sub.empty:
        return None
    ns = sorted(int(n) for n in sub["k"].unique())
    piv = sub.groupby(["example_ordering", "k"])["accuracy_llm_vs_expert"].mean()
    pk = sub.groupby(["example_ordering", "k"])["kappa_llm_vs_expert"].mean()
    rows = []
    for o, nome in _ORDENS:
        if o not in sub["example_ordering"].unique():
            continue
        rows.append([nome] + [num(piv.get((o, n), np.nan)) for n in ns] + [num(pk.get((o, n), np.nan), 3, pct=False) for n in ns])
    amp = [num(piv.xs(n, level=1).max() - piv.xs(n, level=1).min()) for n in ns]
    rows.append(["amplitude (máx $-$ mín)"] + amp + [""] * len(ns))
    rod = ("Problema E1, Passo 5: mesmos exemplos (estratégia mista) em quatro ordenações. " + _NOTA_ACUR_ESPEC + " "
           + NOTA_KAPPA + " Médias das sementes.")
    return Tabela(slug="bloco2_E1_passo5_ordem", nn=8, modelo=model, caption=f"Problema E1, Passo 5 ({esc(model)}): viés de ordem dos exemplos.",
                  headers=["Ordenação"] + [f"k = {n}" for n in ns] + [f"k = {n}" for n in ns], rows=rows,
                  align="l" + "r" * (2 * len(ns)), grupos=[("Acur.\\ vs espec.\\ (\\%)", 2, 1 + len(ns)), ("$\\kappa$", 2 + len(ns), 1 + 2 * len(ns))],
                  rodape=rod, destaque={len(rows) - 1}, midrules_apos={len(rows) - 2})


# ═════════════════════════ Problemas com split 70/30 (F, G1, G2) ═════════════════════════
def _ext(D, chave, model, prefixo) -> pd.DataFrame:
    df = D.get(chave)
    if df is None:
        return pd.DataFrame()
    sub = df[(df["model"] == model) & df["problem_name"].str.startswith(prefixo)].copy()
    sub["pb"] = sub["problem_name"].str.replace(r"_seed\d+$", "", regex=True)
    # acurácia (vs rótulo real) invariante à permutação nome↔classe (nomes neutros A/B e zero-shot)
    for c in ("llm_accuracy_vs_true", "accuracy_llm_vs_true", "accuracy_perc_vs_true", "accuracy_nnls_vs_true"):
        if c in sub.columns:        # ext_a usa llm_accuracy_vs_true; ext_e usa accuracy_llm_vs_true
            sub[c + "_inv"] = _inv_perm(sub[c])
    return sub


def _neutro(pb: str) -> bool:
    return pb.endswith("classesAB")


def _nomes_G(pb: str, feat: str):
    sp = "G1 (normalizado)" if pb.startswith("G1") else "G2 (kg e m)"
    classes = "A / B" if pb.endswith("classesAB") else "Mulher / Homem"
    return sp, classes, feat


def tab_f_passo2(D, model) -> Optional[Tabela]:
    sub = _ext(D, "fg_passos_2_e_4", model, "F_")
    if sub.empty:
        # O F rodou (há linhas do Passo 3) mas o zero-shot colapsou em TODAS as sementes: a tabela sai
        # com uma linha de registro, em vez de sumir sem explicação (14/10/2026).
        apr = _ext(D, "fg_aprendiz", model, "F_")
        if apr.empty:
            return None
        seeds = sorted(int(x) for x in apr["seed"].unique())
        rows = [["R$^2$/R$^3$/R$^4$", ", ".join(map(str, seeds)), "colapso", "—", "—", "—", "—"]]
        rod = ("Problema F, Passo 2: nas sementes listadas o LLM colapsou no zero-shot dos 105 pontos de treino (classe única ou "
               "minoritária com menos de 5 pontos), logo não há métrica estimada nem fidelidade a reportar; o Passo 3 rodou "
               "normalmente com o especialista = rótulo real (ver a tabela seguinte).")
        return Tabela(slug="bloco2_F_passo2_R2_R3_R4", nn=9, modelo=model,
                      caption=f"Problema F, meia-lua, Passo 2 ({esc(model)}): zero-shot no treino colapsado em todas as sementes.",
                      headers=["Espaço", "Semente", "Acur.\\ LLM vs real", "Fid.\\ Perc.", "Fid.\\ NNLS", "Acur.\\ Perc.\\ vs real", "Acur.\\ NNLS vs real"],
                      rows=rows, align="llrrrrr", rodape=rod)
    seeds = sorted(int(s) for s in sub["seed"].unique())
    rows, mids = [], []
    for nf in sorted(sub["n_features"].unique()):
        s = sub[sub["n_features"] == nf]
        for sd in seeds:
            r = s[s["seed"] == sd]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([f"R$^{int(nf)}$", str(sd), num(r["llm_accuracy_vs_true"]), num(r["fidelity_perc_vs_llm"]), num(r["fidelity_nnls_vs_llm"]),
                         num(r["accuracy_perc_vs_true"]), num(r["accuracy_nnls_vs_true"])])
        rows.append([f"R$^{int(nf)}$", "média", num(s["llm_accuracy_vs_true"].mean()), num(s["fidelity_perc_vs_llm"].mean()), num(s["fidelity_nnls_vs_llm"].mean()),
                     num(s["accuracy_perc_vs_true"].mean()), num(s["accuracy_nnls_vs_true"].mean())])
        mids.append(len(rows) - 1)
    n_tr = int(sub["n_samples"].iloc[0]) if "n_samples" in sub.columns else 105
    rod = (f"Problema F (meia-lua como aprendiz, divisão 70/30), Passo 2: o LLM rotula as {n_tr} amostras de treino em zero-shot "
           "e a métrica é estimada só para medir a fidelidade. "
           "Fid.\\ = fidelidade (vs LLM) de cada estimador; Acur.\\ vs real = acurácia (vs rótulo real) do LLM e da métrica, "
           "contexto. " + NOTA_PERMUTACAO + " Sementes ausentes = o LLM colapsou no zero-shot (classe única ou minoritária com menos de 5 pontos; sem métrica).")
    return Tabela(slug="bloco2_F_passo2_R2_R3_R4", nn=9, modelo=model,
                  caption=f"Problema F, meia-lua, Passo 2 ({esc(model)}): zero-shot no treino e métrica por semente e espaço.",
                  headers=["Espaço", "Semente", "Acur.\\ LLM vs real", "Fid.\\ Perc.", "Fid.\\ NNLS", "Acur.\\ Perc.\\ vs real", "Acur.\\ NNLS vs real"],
                  rows=rows, align="llrrrrr", rodape=rod, midrules_apos=set(mids[:-1]), destaque={i for i, r in enumerate(rows) if r[1] == "média"})


def tab_f_passo3(D, model) -> Optional[Tabela]:
    sub = _ext(D, "fg_aprendiz", model, "F_")
    if sub.empty:
        return None
    ns = sorted(int(n) for n in sub["k"].unique())
    acc_esp = "accuracy_llm_vs_expert"     # contrato de 13/10/2026 (sem fallback: migra-se a pasta antes)
    cols = ["accuracy_llm_vs_true", acc_esp, "accuracy_perceptron_baseline_vs_true"]
    rows = []
    for sd in sorted(int(s) for s in sub["seed"].unique()):
        s = sub[sub["seed"] == sd]
        g = s.groupby("k")[cols].mean()
        rows.append([str(sd)] + [num(g[c].get(n, np.nan)) for c in cols for n in ns])
    g = sub.groupby("k")[cols].mean()
    rows.append(["média"] + [num(g[c].get(n, np.nan)) for c in cols for n in ns])
    k = len(ns)
    n_test = int(sub["n_test"].iloc[0]) if "n_test" in sub.columns else 45
    fontes = set(sub["especialista_origem"].dropna().astype(str).unique())
    if fontes <= {"rotulo_real"}:
        quem = "o especialista do Problema F é o rótulo real (coluna especialista\\_origem = rotulo\\_real)"
    else:
        quem = ("o especialista do Problema F é " + ", ".join(sorted(esc(f) for f in fontes))
                + " (coluna especialista\\_origem)")
    rod = (f"Passo 3 do Problema F nas {n_test} amostras de teste; {quem}. Acur.\\ LLM vs espec.\\ = acurácia (LLM vs especialista); "
           "Acur.\\ LLM vs real = acurácia (vs rótulo real) do LLM. Perceptron (Passo 4) = Perceptron diagonal treinado nos mesmos $k$ "
           "exemplos, em R$^2$ (x1, x2, os mesmos atributos que o LLM vê; a métrica do Passo 2 em R$^3$/R$^4$ serve só "
           "para medir a fidelidade), avaliado contra o rótulo real. "
           "Médias dos sorteios por semente.")
    return Tabela(slug="bloco2_F_passo3_k", nn=10, modelo=model,
                  caption=f"Problema F, meia-lua, Passos 3 e 4 ({esc(model)}): aprendizado em contexto por semente e número de exemplos.",
                  headers=["Semente"] + [f"k={n}" for n in ns] * 3, rows=rows, align="l" + "r" * (3 * k),
                  grupos=[("Acur.\\ LLM vs real (\\%)", 2, 1 + k), ("Acur.\\ LLM vs espec.\\ (\\%)", 2 + k, 1 + 2 * k),
                          ("Acur.\\ Perceptron vs real (\\%)", 2 + 2 * k, 1 + 3 * k)],
                  rodape=rod, destaque={len(rows) - 1}, midrules_apos={len(rows) - 2}, tamanho=r"\footnotesize")


def tab_g_passos_2_e_4(D, model) -> Optional[Tabela]:
    sub = _ext(D, "fg_passos_2_e_4", model, "G")
    if sub.empty:
        return None
    ag = _ag(sub, ["pb", "feature_names", "n_features"], ["llm_accuracy_vs_true", "llm_accuracy_vs_true_inv", "n_errors_llm_vs_true",
                                                          "fidelity_perc_vs_llm", "fidelity_nnls_vs_llm", "accuracy_perc_vs_true",
                                                          "n_errors_perc_vs_true", "accuracy_nnls_vs_true", "n_samples"])
    rows, mids, destaque = [], [], set()
    for (pb, feat), g in ag.groupby(["pb", "feature_names"], sort=True):
        sp, classes, _ = _nomes_G(pb, feat)
        for _, r in g.sort_values("n_features").iterrows():
            rows.append([sp, classes, esc(feat), f"R$^{int(r['n_features'])}$",
                         f"{num(r['llm_accuracy_vs_true_mean'])} ({num(r['n_errors_llm_vs_true_mean'], 1, pct=False)})",
                         num(r["llm_accuracy_vs_true_inv_mean"]) if _neutro(pb) else "—",
                         _st(r, "fidelity_perc_vs_llm"), num(r["fidelity_nnls_vs_llm_mean"]),
                         f"{num(r['accuracy_perc_vs_true_mean'])} ({num(r['n_errors_perc_vs_true_mean'], 1, pct=False)})", num(r["accuracy_nnls_vs_true_mean"])])
            if pb == "G2_homem_mulher" and feat.startswith("peso"):
                destaque.add(len(rows) - 1)
        mids.append(len(rows) - 1)
    n = int(ag["n_samples_mean"].iloc[0]) if len(ag) else 70
    rod = (f"Problema G, Passo 2 (zero-shot nas {n} amostras de treino de cada semente) e métrica estimada sobre esses rótulos (Passo 4); "
           "médias das sementes (fidelidade com mín–máx). "
           "Fid.\\ = fidelidade (vs LLM) de cada estimador; Acur.\\ vs real = acurácia (vs rótulo real) do LLM e da métrica, "
           "contexto, com o número médio de erros entre parênteses (contagem auxiliar). Acur.\\ vs real (inv.\\ perm.) = "
           "max($a$, 1 $-$ $a$) nas linhas com classes neutras A/B: em zero-shot e com nomes neutros a correspondência "
           "entre nome e classe é arbitrária e a acurácia literal pode sair invertida (a métrica, estimada dos rótulos "
           "do LLM, herda a mesma permutação). Atributos extras (x1·x2; x1², x2²) só na estimação. "
           "Combinações ausentes = o LLM colapsou no zero-shot (classe única ou minoritária com menos de 5 pontos) nas três sementes. Em negrito: G2 com nomes semânticos.")
    return Tabela(slug="bloco3_G_passos_2_e_4_R2_R3_R4", nn=1, modelo=model,
                  caption=f"Problema G, Passos 2 e 4 ({esc(model)}): zero-shot e métrica em R$^2$, R$^3$ e R$^4$ por subproblema, nomes de classe e de atributo.",
                  headers=["Subproblema", "Classes", "Atributos", "Espaço", "Acur.\\ LLM vs real (n erros)", "Acur.\\ vs real (inv.\\ perm.)",
                           "Fid.\\ Perc.", "Fid.\\ NNLS", "Acur.\\ Perc.\\ vs real (n erros)", "Acur.\\ NNLS vs real"],
                  rows=rows, align="llllrrrrrr", rodape=rod, midrules_apos=set(mids[:-1]), destaque=destaque, tamanho=r"\footnotesize")


def tab_g_passo6(D, model, versus_perceptron: bool = False) -> Optional[Tabela]:
    sub = _ext(D, "fg_aprendiz", model, "G")
    if sub.empty:
        return None
    ns = sorted(int(n) for n in sub["k"].unique())
    rows, destaque = [], set()
    # 14/10/2026: `n_features` = atributos MOSTRADOS ao LLM (2 = x1, x2; 3 = + x1·x2; 4 = + x1², x2²), fator do
    # Passo 6 (`ATRIBUTOS_PROMPT_PASSO6_G`); até 13/10/2026 era a dimensão da métrica eleita.
    for (pb, feat, nf), g in sub.groupby(["pb", "feature_names", "n_features"], sort=True):
        sp, classes, _ = _nomes_G(pb, feat)
        dim = f"R$^{int(nf)}$"
        a = g.groupby("k")[["accuracy_llm_vs_true", "accuracy_llm_vs_true_inv", "n_errors_llm_vs_true",
                                 "accuracy_perceptron_baseline_vs_true", "n_test"]].mean()
        if versus_perceptron:
            cells = [sp, classes, esc(feat), dim] + [num(a["accuracy_llm_vs_true"].get(n, np.nan)) for n in ns] + [num(a["accuracy_perceptron_baseline_vs_true"].get(n, np.nan)) for n in ns]
        else:
            cells = ([sp, classes, esc(feat), dim]
                     + [f"{num(a['accuracy_llm_vs_true'].get(n, np.nan))} ({num(a['n_errors_llm_vs_true'].get(n, np.nan), 1, pct=False)})" for n in ns]
                     + [num(a["accuracy_llm_vs_true_inv"].get(n, np.nan)) if _neutro(pb) else "—" for n in ns])
        rows.append(cells)
        if pb == "G2_homem_mulher" and feat.startswith("peso"):
            destaque.add(len(rows) - 1)
    n_test = int(sub["n_test"].iloc[0])
    k = len(ns)
    if versus_perceptron:
        rod = (f"Problema G, Passo 6, nas {n_test} amostras de teste; o especialista é a métrica eleita no Passo 5. Prompt = "
               "atributos mostrados ao LLM (R$^2$ = x1, x2; R$^3$ = + x1·x2; R$^4$ = + x1², x2²), fator controlado independente da "
               "métrica eleita. Perceptron (Passo 7) = Perceptron diagonal treinado nos mesmos $k$ exemplos, no mesmo espaço que o LLM "
               "viu (coluna Prompt); se todos os regimes dos Passos 2 e 3 colapsaram, sem métrica eleita, o Passo 6 não roda e a linha "
               "não existe. Acurácia (vs rótulo "
               "real) do LLM e do Perceptron; médias das sementes e sorteios. Nas linhas com classes neutras A/B a acurácia "
               "literal do LLM pode sair invertida (ver bloco3\\_table\\_02)."
               + " Com 3 ou 4 atributos no prompt, os exemplos do G2 mostram os extras em unidades físicas (ex.: peso\\_kg·altura\\_m), que não são função monótona de x1·x2 normalizado.")
        return Tabela(slug="bloco3_G_passo7_LLM_vs_perceptron", nn=3, modelo=model,
                      caption=f"Problema G, Passos 6 e 7 ({esc(model)}): LLM versus Perceptron treinado nos mesmos exemplos, acurácia contra o rótulo real.",
                      headers=["Subproblema", "Classes", "Atributos", "Prompt"] + [f"k = {n}" for n in ns] * 2, rows=rows, align="llll" + "r" * (2 * k),
                      grupos=[("Acur.\\ LLM vs real (\\%)", 5, 4 + k), ("Acur.\\ Perceptron vs real (\\%)", 5 + k, 4 + 2 * k)], rodape=rod, destaque=destaque, tamanho=r"\footnotesize")
    rod = (f"Problema G, Passo 6, nas {n_test} amostras de teste de cada semente; o especialista é a métrica eleita no Passo 5 (por "
           f"acurácia vs rótulo real no treino). Acurácia (vs rótulo real) do LLM, n erros em {n_test} entre parênteses (contagem auxiliar); "
           "Acur.\\ vs real (inv.\\ perm.) = max($a$, 1 $-$ $a$) nas linhas com classes neutras A/B, onde a correspondência "
           "entre nome e classe é arbitrária e a acurácia literal pode sair invertida. Médias das sementes e sorteios. "
           "Prompt = atributos mostrados ao LLM (R$^2$ = x1, x2; R$^3$ = + x1·x2; R$^4$ = + x1², x2²), fator controlado e "
           "independente da métrica eleita; em k = 0 o prompt não tem exemplos nem extras, logo as linhas R$^2$, "
           "R$^3$ e R$^4$ repetem a mesma coleta. Em negrito: G2 com nomes semânticos.")
    return Tabela(slug="bloco3_G_passo6_k", nn=2, modelo=model,
                  caption=f"Problema G, Passo 6 ({esc(model)}): aprendizado em contexto, acurácia do LLM contra o rótulo real por número de exemplos e atributos no prompt.",
                  headers=["Subproblema", "Classes", "Atributos", "Prompt"] + [f"k = {n}" for n in ns] * 2, rows=rows, align="llll" + "r" * (2 * k),
                  grupos=[("Acur.\\ LLM vs real (n erros)", 5, 4 + k), ("Acur.\\ vs real (inv.\\ perm.)", 5 + k, 4 + 2 * k)],
                  rodape=rod, destaque=destaque, tamanho=r"\footnotesize")


def tab_g1_vs_g2(D, model) -> Optional[Tabela]:
    sub = _ext(D, "fg_passos_2_e_4", model, "G")
    sub = sub[sub["n_features"] == 2] if not sub.empty else sub
    if sub.empty or not sub["pb"].str.startswith("G2").any():
        return None
    ag = sub.groupby(["pb", "feature_names"])[["llm_accuracy_vs_true", "llm_accuracy_vs_true_inv", "accuracy_perc_vs_true",
                                               "accuracy_perc_vs_true_inv", "fidelity_perc_vs_llm"]].mean()

    def _g(pb, feats):
        for f in feats:
            if (pb, f) in ag.index:
                return ag.loc[(pb, f)]
        return None
    rows = []
    for classes, suf in (("Mulher / Homem", "homem_mulher"), ("A / B", "homem_mulher_classesAB")):
        for nome_at, f1, f2 in (("semânticos (peso/altura; peso\\_kg/altura\\_m)", ["peso/altura"], ["peso_kg/altura_m"]), ("neutros (x1/x2)", ["x1/x2"], ["x1/x2"])):
            g1, g2 = _g(f"G1_{suf}", f1), _g(f"G2_{suf}", f2)
            v = lambda r, c: r[c] if r is not None else np.nan
            rows.append([classes, nome_at, num(v(g1, "llm_accuracy_vs_true")), num(v(g2, "llm_accuracy_vs_true")), delta(v(g1, "llm_accuracy_vs_true"), v(g2, "llm_accuracy_vs_true")),
                         num(v(g1, "llm_accuracy_vs_true_inv")), num(v(g2, "llm_accuracy_vs_true_inv")),
                         num(v(g1, "accuracy_perc_vs_true")), num(v(g2, "accuracy_perc_vs_true")), delta(v(g1, "accuracy_perc_vs_true"), v(g2, "accuracy_perc_vs_true")),
                         num(v(g1, "fidelity_perc_vs_llm")), num(v(g2, "fidelity_perc_vs_llm"))])
    rod = ("Problema G, Passos 2 e 4 (zero-shot e métrica estimada sobre esses rótulos, R$^2$, Perceptron), médias das sementes. G1 mostra ao LLM os valores normalizados em $[-1, 1]$; G2 mostra "
           "os mesmos pontos em kg e m (regra de três: 45–110 kg, 1,50–1,95 m). A métrica é estimada no espaço normalizado nos dois. "
           "Acurácia (vs rótulo real) do LLM e da métrica, contexto; Fidelidade = métrica vs LLM. Acur.\\ LLM vs real (inv.\\ perm.) = "
           "max($a$, 1 $-$ $a$): em zero-shot e com nomes neutros (A/B) a correspondência entre nome e classe é arbitrária, e só o "
           "valor invariante à permutação compara as linhas. " + b1.NOTA_COLAPSO)
    return Tabela(slug="bloco3_G1_vs_G2_passos_2_e_4", nn=4, modelo=model,
                  caption=f"Problema G, Passos 2 e 4 ({esc(model)}): valores normalizados (G1) versus kg e m (G2), zero-shot e métrica estimada.",
                  headers=["Classes", "Atributos", "G1", "G2", "$\\Delta$", "G1 (inv.\\ perm.)", "G2 (inv.\\ perm.)", "G1", "G2", "$\\Delta$", "G1", "G2"],
                  rows=rows, align="llrrrrrrrrrr",
                  grupos=[("Acur.\\ LLM vs real (\\%)", 3, 7), ("Acur.\\ métrica (\\%)", 8, 10), ("Fidelidade (\\%)", 11, 12)],
                  rodape=rod, tamanho=r"\footnotesize")


def tab_g_nomes(D, model) -> Optional[Tabela]:
    sub = _ext(D, "fg_passos_2_e_4", model, "G")
    sub = sub[sub["n_features"] == 2] if not sub.empty else sub
    if sub.empty:
        return None
    ag = sub.groupby(["pb", "feature_names"])[["llm_accuracy_vs_true", "llm_accuracy_vs_true_inv",
                                               "accuracy_perc_vs_true", "accuracy_perc_vs_true_inv"]].mean()
    rows, mids = [], []
    for sp, pref in (("G1 (normalizado)", "G1"), ("G2 (kg e m)", "G2")):
        for feat in sorted({f for (pb, f) in ag.index if pb.startswith(pref)}):
            sem = ag.loc[(f"{pref}_homem_mulher", feat)] if (f"{pref}_homem_mulher", feat) in ag.index else None
            neu = ag.loc[(f"{pref}_homem_mulher_classesAB", feat)] if (f"{pref}_homem_mulher_classesAB", feat) in ag.index else None
            v = lambda r, c: r[c] if r is not None else np.nan
            # o Δ semântico − neutro usa a versão invariante à permutação nos dois lados (I3)
            rows.append([sp, esc(feat),
                         num(v(sem, "llm_accuracy_vs_true")), num(v(neu, "llm_accuracy_vs_true")),
                         num(v(sem, "llm_accuracy_vs_true_inv")), num(v(neu, "llm_accuracy_vs_true_inv")),
                         delta(v(neu, "llm_accuracy_vs_true_inv"), v(sem, "llm_accuracy_vs_true_inv")),
                         num(v(sem, "accuracy_perc_vs_true_inv")), num(v(neu, "accuracy_perc_vs_true_inv")),
                         delta(v(neu, "accuracy_perc_vs_true_inv"), v(sem, "accuracy_perc_vs_true_inv"))])
        if rows:
            mids.append(len(rows) - 1)
    rod = ("Problema G, Passos 2 e 4 (zero-shot e métrica estimada sobre esses rótulos, R$^2$, Perceptron): efeito do nome das classes (M/H = Mulher/Homem versus A/B) para cada par de nomes de atributo. "
           "Acurácia (vs rótulo real) do LLM e da métrica, contexto. Com nomes neutros (e em zero-shot) a correspondência "
           "entre nome e classe é arbitrária: a acurácia literal pode sair invertida; as colunas ``inv.\\ perm.'' usam o "
           "valor invariante à permutação max($a$, 1 $-$ $a$), e $\\Delta$ = semântico $-$ neutro é calculado sobre elas. "
           + b1.NOTA_COLAPSO)
    return Tabela(slug="bloco3_G_nomes_semanticos_vs_neutros", nn=5, modelo=model,
                  caption=f"Problema G, Passos 2 e 4 ({esc(model)}): nomes de classe semânticos versus neutros.",
                  headers=["Subproblema", "Atributos", "M/H literal", "A/B literal", "M/H inv.\\ perm.", "A/B inv.\\ perm.", "$\\Delta$",
                           "M/H inv.\\ perm.", "A/B inv.\\ perm.", "$\\Delta$"],
                  rows=rows, align="llrrrrrrrr",
                  grupos=[("Acur.\\ LLM vs real (\\%)", 3, 7), ("Acur.\\ métrica vs real (\\%)", 8, 10)],
                  rodape=rod, midrules_apos=set(mids[:-1]), tamanho=r"\footnotesize")


def tab_g_passos_2_a_5(D, model) -> Optional[Tabela]:
    """Passos 2 a 5 do Problema G (`bloco3_g_passos_2_a_5`): uma tabela por modelo, grupos de
    linhas por (subproblema, nomes de classe, nomes de atributo) e, dentro, zero-shot, cada k e o teto.
    Reaproveita `agregar`/`fid_cell`/`acc_cell`/`marca_gamma`/`regime_label` do Bloco 1."""
    df = D.get("g_passos_2_a_5")
    if df is None or "model" not in df.columns:
        return None
    sub = df[df["model"] == model].copy()
    if sub.empty:
        return None
    for c in ("subproblem", "class_names", "feature_names"):
        if c not in sub.columns:
            return None
    if "collapsed" in sub.columns:
        # Colapso: acurácia do LLM e partição não medem um critério; saem das médias (como em _linhas_regime).
        sub.loc[sub["collapsed"] == 1, ["llm_accuracy_vs_true"]] = np.nan
    sub["llm_accuracy_vs_true_inv"] = _inv_perm(sub["llm_accuracy_vs_true"])
    # Eleição do Passo 5 (13/10/2026): quantas sementes elegeram cada (regime, k) como especialista do
    # Passo 6, lida de bloco23_fg_aprendiz (especialista_regime/especialista_k por problem_name × atributos).
    eleitas: dict = {}
    apr = D.get("fg_aprendiz")
    if apr is not None and "especialista_regime" in apr.columns:
        a = apr[(apr["model"] == model) & apr["problem_name"].astype(str).str.startswith("G")]
        a = a.dropna(subset=["especialista_regime"]).drop_duplicates(["problem_name", "feature_names", "seed"])
        for _, r in a.iterrows():
            pn = str(r["problem_name"])
            chave = (pn[:2], "A/B" if pn.endswith("classesAB") else "Mulher/Homem", str(r["feature_names"]),
                     str(r["especialista_regime"]), int(float(r["especialista_k"])))
            eleitas[chave] = eleitas.get(chave, 0) + 1
    n_seeds = int(sub["random_seed"].nunique()) if "random_seed" in sub.columns else 0
    cols = ["fidelity_vs_llm", "accuracy_metric_vs_true_full", "accuracy_teto_vs_true",
            "llm_accuracy_vs_true", "llm_accuracy_vs_true_inv", "collapsed", "gamma", "margem_minima"]
    perc = sub[sub["algorithm"] == "perceptron"]
    ag = b1.agregar(perc, ["subproblem", "class_names", "feature_names", "regime", "k", "n_features"], cols)
    espacos = sorted(int(n) for n in ag["n_features"].unique())
    rows, mids, destaque = [], [], set()
    # Mulher/Homem antes de A/B (ordem do CLAUDE.md), depois os atributos
    grupos = sorted(set(zip(ag["subproblem"], ag["class_names"], ag["feature_names"])),
                    key=lambda t: (t[0], t[1].replace(" ", "") != "Mulher/Homem", t[2]))
    frac_gamma0 = float(((perc["gamma"].fillna(1.0) <= 0.0)
                         | (pd.to_numeric(perc["margem_minima"], errors="coerce").fillna(0.0) < 0.0 if "margem_minima" in perc.columns else False)).mean()) \
        if "gamma" in perc.columns and len(perc) else float("nan")
    for sp, cn, fn in grupos:
        g = ag[(ag["subproblem"] == sp) & (ag["class_names"] == cn) & (ag["feature_names"] == fn)]
        g2 = b1._ordem(g[g["n_features"] == 2]).reset_index(drop=True)
        por_nf = {nf: g[g["n_features"] == nf].set_index(["regime", "k"]) for nf in espacos}
        neutro = cn.replace(" ", "") in ("A/B", "B/A")
        primeiro = True
        for _, r in g2.iterrows():
            reg, k = r["regime"], r["k"]
            gt = reg == "passo5_teto"
            fids = []
            for nf in espacos:
                rn = por_nf[nf].loc[(reg, k)] if (reg, k) in por_nf[nf].index else {}
                fids.append(fid_cell(rn, reg) if len(rn) else "—")
            rows.append([("G1 (normalizado)" if sp == "G1" else "G2 (kg e m)" if sp == "G2" else esc(sp)) if primeiro else "",
                         esc(cn.replace("/", " / ")) if primeiro else "",
                         esc(fn) if primeiro else "",
                         regime_label(reg, k) + marca_gamma(r),
                         "—" if gt else b1.fmt_stat(r, "llm_accuracy_vs_true"),
                         "—" if (gt or not neutro) else b1.fmt_stat(r, "llm_accuracy_vs_true_inv"),
                         *fids,
                         acc_cell(r, reg),
                         ("—" if (gt or not eleitas or float(r.get("collapsed_mean", 0) or 0) >= 1.0)
                          else f"{eleitas.get((sp, cn, fn, reg, int(k)), 0)}/{n_seeds}")])
            if sp == "G2" and fn.startswith("peso") and not neutro:
                destaque.add(len(rows) - 1)
            primeiro = False
        mids.append(len(rows) - 1)
    n_tr = int(sub["n_total"].iloc[0]) if "n_total" in sub.columns else 70
    ks = sorted(int(k) for k in sub.loc[sub["regime"] == "passo3_exemplos_rotulo_real", "k"].unique())
    k_max = max(ks) if ks else 0
    rod = (f"Mesmos passos do Problema A nos {n_tr} pontos de treino de cada semente (divisão 70/30 "
           "do Problema G): zero-shot (Passo 2); few-shot (Passo 3) com $k$ exemplos rotulados pelo RÓTULO REAL ($k/2$ por classe, sorteio "
           f"aleatório por semente e repetição), o LLM classifica os ${n_tr}-k$ restantes (com $k = {k_max}$ sobram "
           f"{n_tr - k_max} pontos para estimar a métrica, Passo 4); teto (Passo 4) = W ajustado pelo MESMO estimador (Perceptron) diretamente ao rótulo real, que não tem "
           "fidelidade, só acurácia; como o Perceptron devolve W em melhor esforço quando não há margem viável ($\\gamma = 0$), o teto "
           "é uma referência, não um máximo garantido, e métricas estimadas dos rótulos do LLM podem superá-lo. A métrica (Perceptron) de maior acurácia vs rótulo real nos "
           f"{n_tr} pontos é a eleita no Passo 5 e vira o especialista do Passo 6 (empates resolvidos pela ordem zero-shot, $k$ crescente, repetição, R$^2$ → R$^3$ → R$^4$); "
           "a coluna Eleita conta em quantas sementes cada regime/$k$ foi eleito (``—'' = teto ou zero-shot colapsado em todas as sementes). "
           + (f"Perceptron sem $\\hat{{W}}$ separador (†) em {frac_gamma0:.0%} das estimativas deste modelo no G. " if frac_gamma0 == frac_gamma0 else "")
           + "Médias das sementes e repetições, mín–máx entre parênteses, em \\%. "
           "Fid.\\ (vs LLM) = fidelidade da métrica (Perceptron) nos pontos avaliados, em R$^2$/R$^3$/R$^4$ (o LLM vê "
           f"sempre só os dois atributos; x1·x2, x1², x2² só na estimação); Acur.\\ vs real = acurácia (vs rótulo real) da "
           f"métrica em R$^2$ aplicada aos {n_tr} pontos; Acur.\\ LLM vs real = acurácia do próprio LLM nos pontos avaliados, "
           "contexto. Acur.\\ LLM (inv.\\ perm.) = max($a$, 1 $-$ $a$) nas variantes com classes neutras A/B. "
           + NOTA_PERMUTACAO + " " + NOTA_GAMMA + " " + b1.NOTA_COLAPSO + " Em negrito: G2 com nomes semânticos.")
    hdr = ["Subproblema", "Classes", "Atributos", "Regime", "Acur.\\ LLM vs real", "Acur.\\ LLM (inv.\\ perm.)"]
    hdr += [f"R$^{nf}$" for nf in espacos] + ["Acur.\\ vs real R$^2$", "Eleita (Passo 5)"]
    n_e = len(espacos)
    return Tabela(slug="bloco3_G_passos_2_a_5", nn=6, modelo=model,
                  caption=f"Problema G, Passos 2 a 5 ({esc(model)}): zero-shot, few-shot (exemplos do rótulo real), teto e eleição, por subproblema, nomes de classe e de atributo.",
                  headers=hdr, rows=rows, align="llll" + "r" * (2 + n_e + 1) + "c",
                  grupos=[("Fid.\\ (vs LLM), Perceptron", 7, 6 + n_e)],
                  rodape=rod, midrules_apos=set(mids[:-1]), destaque=destaque, tamanho=r"\footnotesize")


# ═════════════════════════ fechamento ═════════════════════════
def tab_final_cross(D, model) -> Optional[Tabela]:
    df = D.get("cross")
    if df is None:
        return None
    sub = df[df["_alias"] == _model_alias(model)].copy()
    if sub.empty:
        return None
    # Rótulo da linha: `problem` (estável, sem seed: A_linear, D_meialua, G1_…) quando existe; senão
    # `problem_name`. Até 09/10 a meia-lua gravava problem_name="D" (letra solta), incoerente com as demais.
    pcol = "problem" if "problem" in sub.columns else _col(sub, "problem_name", "problem_base")
    sub["pb"] = sub[pcol].astype(str).str.replace(r"_seed\d+$", "", regex=True)
    com_feat = "feature_names" in sub.columns
    sub["feat"] = sub["feature_names"].fillna("x1/x2").astype(str) if com_feat else "—"
    ag = sub.groupby(["pb", "feat", "algorithm", "n_features"])[["fidelity_vs_llm", "accuracy_vs_true"]].mean()
    nfs = sorted(int(n) for n in sub["n_features"].unique())
    rows, mids = [], []
    ordem_pb = sorted(sub["pb"].unique(), key=lambda p: (0 if p.startswith("A") else 1 if p.startswith(("D", "F")) else 2, p))
    for pb in ordem_pb:
        feats = sorted(sub.loc[sub["pb"] == pb, "feat"].unique(), key=lambda f: (0 if f == "x1/x2" else 1, f))
        for feat in feats:
            for algo in ("perceptron", "nnls"):
                f = [ag["fidelity_vs_llm"].get((pb, feat, algo, n), np.nan) for n in nfs]
                a = [ag["accuracy_vs_true"].get((pb, feat, algo, n), np.nan) for n in nfs]
                rows.append([esc(pb), esc(feat), algo] + [num(x) for x in f] + [delta(f[0], f[-1])] + [num(x) for x in a] + [delta(a[0], a[-1])])
        mids.append(len(rows) - 1)
    k = len(nfs)
    tem_D = sub["pb"].str.startswith("D").any()
    rod = ("Síntese cruzada: o ganho de R$^2$ para R$^4$ indica se o LLM adota um critério não linear. A\\_linear = Problema A "
           "(rótulos zero-shot, Passo 2, métrica do Passo 4); " + ("meia-lua = Problema D (150 pontos, Passos 2 e 4)" if tem_D else
                                       "F = meia-lua com split (105 pontos de treino, Passo 2)")
           + "; G1, G2 = peso $\\times$ altura com split (Passo 2, zero-shot no treino). Fidelidade (vs LLM) e acurácia "
           "(vs rótulo real) da métrica, contexto. "
           + ("Uma linha por variante de nomes de atributo (coluna Atributos). Todas as linhas vêm de rótulos zero-shot "
              "com classes neutras A/B (A\\_linear, meia-lua e as variantes classesAB de G1/G2): a correspondência entre "
              "nome e classe é arbitrária e a acurácia (vs rótulo real) está sujeita à permutação em todas elas, não só "
              "nas classesAB; apenas em G1/G2 com Mulher/Homem o nome de classe carrega significado. " if com_feat else
              "Sem coluna feature\\_names: cada linha é a média sobre as variantes de nomes de atributo e de classe do subproblema. ")
           + "Médias das sementes. ``—'' = sem métrica (o zero-shot do LLM colapsou: classe única ou minoritária "
           "com menos de 5 pontos; a linha fica para registrar o colapso). Variantes de G1/G2 cujo zero-shot colapsou em "
           "TODAS as sementes não têm linha (o CSV bloco23\\_fg\\_passos\\_2\\_e\\_4 só grava coletas com métrica; "
           "os colapsos estão em bloco3\\_g\\_passos\\_2\\_a\\_5 e no log).")
    return Tabela(slug="final_linearidade_cruzada_R2_R3_R4", nn=1, modelo=model,
                  caption=f"Linearidade cruzada ({esc(model)}): fidelidade e acurácia em R$^2$, R$^3$ e R$^4$ nos problemas linear e não lineares; zero-shot do LLM (Passo 2) e métrica em R$^2$/R$^3$/R$^4$ (Passo 4 em A, D e G; Passo 2 no F).",
                  headers=["Problema", "Atributos", "Algoritmo"] + [f"R$^{n}$" for n in nfs] + ["$\\Delta$ R$^4$$-$R$^2$"] + [f"R$^{n}$" for n in nfs] + ["$\\Delta$"],
                  rows=rows, align="lll" + "r" * (2 * k + 2), grupos=[("Fidelidade (vs LLM) (\\%)", 4, 4 + k), ("Acur.\\ vs real (\\%)", 5 + k, 5 + 2 * k)],
                  rodape=rod, midrules_apos=set(mids[:-1]), tamanho=r"\footnotesize")


def tab_final_comparacao_modelos(D, audit: Optional[dict]) -> Optional[Tabela]:
    ms = modelos(D)
    if len(ms) < 2:
        return None
    linhas = []

    def _linha(nome, fn, casas=1, pct=True):
        vals = []
        for m in ms:
            try:
                v = fn(m)
            except Exception:
                v = np.nan
            vals.append(num(v, casas, pct) if not isinstance(v, str) else v)
        linhas.append([nome] + vals)
    proto, bc, fe, ea = D.get("ad_passos_2_a_5"), D.get("bc_passo3"), D.get("e"), D.get("fg_passos_2_e_4")
    if proto is not None:
        pa = proto[(proto["problem"] == "A") & (proto["n_features"] == 2) & (proto["algorithm"] == "perceptron")]
        _linha("A: fidelidade (vs LLM) zero-shot, Perceptron", lambda m: pa[(pa["model"] == m) & (pa["regime"] == "passo2_zero_shot")]["fidelity_vs_llm"].mean())
        _linha("A: acurácia (vs rótulo real) do LLM, zero-shot (inv.\\ perm.)",
               lambda m: _inv_perm(pa[(pa["model"] == m) & (pa["regime"] == "passo2_zero_shot")]["llm_accuracy_vs_true"]).mean())
        _linha("A: acurácia (vs rótulo real) do LLM, 10-shot", lambda m: pa[(pa["model"] == m) & (pa["k"] == 10)]["llm_accuracy_vs_true"].mean())

        def _eleita(m):
            e = b1.eleita_passo6(bc, m)
            return "—" if e is None else b1.regime_label(*e)
        _linha("A: métrica eleita para a transferência", _eleita)
        pd_ = proto[(proto["problem"] == "D") & (proto["n_features"] == 2) & (proto["algorithm"] == "perceptron")]
        if not pd_.empty:
            _linha("D: acurácia (vs rótulo real) do LLM, zero-shot (inv.\\ perm.)",
                   lambda m: _inv_perm(pd_[(pd_["model"] == m) & (pd_["regime"] == "passo2_zero_shot")
                                           & (pd_["collapsed"].fillna(0) == 0)]["llm_accuracy_vs_true"]).mean())   # colapso fora da média
            _linha("D: fidelidade (vs LLM) 10-shot, R$^2$", lambda m: pd_[(pd_["model"] == m) & (pd_["k"] == 10)]["fidelity_vs_llm"].mean())
    if bc is not None:
        z = bc[(bc["metrica_regime"] == "passo2_zero_shot") & (bc["algorithm"] == "perceptron")]
        _linha("B: Cohen's $\\kappa$ zero-shot (métrica zero-shot do A)", lambda m: z[(z["model"] == m) & (z["problem"] == "B")]["kappa"].mean(), 3, False)
        _linha("C: Cohen's $\\kappa$ zero-shot (métrica zero-shot do A)", lambda m: z[(z["model"] == m) & (z["problem"] == "C")]["kappa"].mean(), 3, False)
    if fe is not None:
        e1 = fe[(fe["expert_name"] == "E1") & fe["example_strategy"].isin([e for e, _ in _ESTRATEGIAS])]
        _linha("E1: acurácia (LLM vs especialista), zero-shot", lambda m: e1[(e1["model"] == m) & (e1["k"] == 0)]["accuracy_llm_vs_expert"].mean())
        _linha("E1: acurácia (LLM vs especialista), média few-shot (todas as estratégias)",
               lambda m: e1[(e1["model"] == m) & (e1["k"] > 0)].groupby(["example_strategy", "k"])["accuracy_llm_vs_expert"].mean().mean())
        _linha("E1: acurácia (LLM vs especialista), média few-shot, difícil", lambda m: e1[(e1["model"] == m) & (e1["k"] > 0) & (e1["example_strategy"] == "hard")]["accuracy_llm_vs_expert"].mean())
    if ea is not None:
        g2 = ea[(ea["problem_name"] == "G2_homem_mulher") & (ea["feature_names"] == "peso_kg/altura_m") & (ea["n_features"] == 2)]
        if not g2.empty:
            _linha("G2 (kg/m, Mulher/Homem): acurácia (vs rótulo real) do LLM, zero-shot", lambda m: g2[g2["model"] == m]["llm_accuracy_vs_true"].mean())
            _linha("G2: acurácia (vs rótulo real) da métrica, Perceptron", lambda m: g2[g2["model"] == m]["accuracy_perc_vs_true"].mean())
        g1 = ea[(ea["problem_name"] == "G1_homem_mulher") & (ea["feature_names"] == "peso/altura") & (ea["n_features"] == 2)]
        _linha("G1 (normalizado, Mulher/Homem): acurácia (vs rótulo real) do LLM", lambda m: g1[g1["model"] == m]["llm_accuracy_vs_true"].mean() if (g1["model"] == m).any() else "colapso")
    if audit:
        pm = audit.get("por_modelo", {})

        def _pm(m, campo):
            for k, v in pm.items():
                if k.endswith(m):
                    return v[campo]
            return np.nan
        _linha("Chamadas", lambda m: f"{int(_pm(m, 'total_chamadas')):,}".replace(",", "."))
        _linha("Malformadas (\\%)", lambda m: _pm(m, "taxa_malformadas"), 3)
        _linha("Flip a T=0 (\\%)", lambda m: _pm(m, "taxa_flip_T0"), 2)
    rod = ("Resumo lado a lado dos modelos; valores em \\% salvo indicação. Passos: A, Passos 2 a 6 (zero-shot = Passo 2; 10-shot = "
           "Passo 3; métrica eleita = Passo 6); B e C, Passo 3; D, Passos 2 e 3 (10-shot = Passo 3); E1, Passos 2 e 3; G, Passos 2 e 4. Métricas: fidelidade (métrica reproduz o LLM), "
           "acurácia (vs rótulo real, contexto; vs especialista no Bloco 2), Cohen's $\\kappa$. ``inv.\\ perm.'' = max($a$, 1 $-$ $a$), "
           "pois em zero-shot a correspondência entre nome e classe é arbitrária. Média few-shot = média das médias por "
           "(estratégia, k). Malformadas e flip são métricas de processo (auditoria). Detalhes em cada tabela por modelo.")
    return Tabela(slug="final_comparacao_modelos", nn=2, caption="Comparação entre os modelos: resumo dos principais resultados por problema (A, Passos 2 a 6; B e C, Passo 3; D, Passos 2 e 3; E1, Passos 2 e 3; G, Passos 2 e 4).",
                  headers=["Resultado"] + [esc(m) for m in ms], rows=linhas, align="l" + "r" * len(ms), rodape=rod)


def tab_final_auditoria_modelo(audit: Optional[dict]) -> Optional[Tabela]:
    if not audit or not audit.get("por_modelo"):
        return None
    rows = []
    for k, v in audit["por_modelo"].items():
        rows.append([esc(k.split("/", 1)[-1]), f"{v['total_chamadas']:,}".replace(",", "."), str(v.get("malformadas", 0)), num(v["taxa_malformadas"], 3),
                     f"{v.get('queries_repetidas', 0):,}".replace(",", "."), str(v.get("queries_divergentes", 0)), num(v["taxa_flip_T0"], 2),
                     esc(", ".join(v.get("models_resolved") or [])),        # listas gravadas por audit_interactions.audit
                     esc(", ".join(v.get("inference_providers") or []))])
    rod = ("Auditoria offline de todas as interações (todos os problemas e passos com chamadas ao LLM). Flip = perguntas idênticas "
           "(prompt + ponto) cuja resposta interpretada divergiu entre repetições, mesmo com T = 0. Snapshot e provedor de "
           "inferência observados no OpenRouter.")
    return Tabela(slug="final_auditoria_por_modelo", nn=3, caption="Auditoria das interações por modelo, todos os problemas e Passos: malformadas e flip a $T=0$.",
                  headers=["Modelo", "Chamadas", "Malf.", "Malf.\\ (\\%)", "Repetidas", "Divergentes", "Flip (\\%)", "Snapshot", "Provedor"],
                  rows=rows, align="lrrrrrrll", rodape=rod)


# ═════════════════════════ catálogo e orquestração ═════════════════════════
def catalogo(D: Dict[str, pd.DataFrame], audit: Optional[dict] = None) -> List[Tabela]:
    """Todas as tabelas possíveis com os CSVs presentes, na ordem bloco → problema."""
    out: List[Tabela] = []
    por_regime = (audit or {}).get("por_regime", {})
    # Bloco 1: T1–T9 (sem a T5, que entra no fechamento) + extras, por modelo
    b1_tabs = b1.tabelas_bloco1(D.get("ad_passos_2_a_5"), D.get("bc_passo3"), None)
    por_modelo: Dict[str, List[Tabela]] = {}
    for t in b1_tabs:
        por_modelo.setdefault(t.modelo, []).append(t)
    ms = modelos(D)
    for m in ms:
        out += por_modelo.get(m, [])
        for fn in (tab_a_passo4_r2_r3_r4, tab_a_passo4_perceptron_vs_nnls, tab_bc_passo4_exemplos_da_metrica, tab_d_zero_vs_exemplos, tab_ad_por_regime, tab_d_passo6_svm):
            t = fn(D, m)
            if t:
                out.append(t)
        for fator in ("classe", "atributo", "prompt"):
            t = _tab_vies(D, m, fator)
            if t:
                out.append(t)
    for fn in (tab_oraculo, tab_d_passo5_teto):
        t = fn(D)
        if t:
            out.append(t)
    # Bloco 2
    for m in ms:
        fe = _fe(D, m)
        for ex in sorted(fe["expert_name"].unique()) if not fe.empty else []:
            t = tab_e_passo3_estrategia_x_k(D, m, ex)
            if t:
                out.append(t)
        for fn in (tab_e_zero_vs_exemplos, tab_e_passo3_h4, tab_e_passo6_baselines, tab_e_passo4_diluicao, tab_e_passo5_ordem, tab_f_passo2, tab_f_passo3):
            t = fn(D, m)
            if t:
                out.append(t)
    # Bloco 3
    for m in ms:
        for fn in (tab_g_passos_2_e_4, tab_g_passo6, lambda D_, m_: tab_g_passo6(D_, m_, versus_perceptron=True), tab_g1_vs_g2, tab_g_nomes,
                   tab_g_passos_2_a_5):
            t = fn(D, m)
            if t:
                out.append(t)
    # Fechamento
    for m in ms:
        t = tab_final_cross(D, m)
        if t:
            out.append(t)
    for t in (tab_final_comparacao_modelos(D, audit), tab_final_auditoria_modelo(audit),
              b1.tab_final_auditoria_flip(por_regime) if por_regime else None):
        if t:
            out.append(t)
    ordem_bloco = {"bloco1": 1, "bloco2": 2, "bloco23": 3, "bloco3": 4, "final": 5}
    out.sort(key=lambda t: (ordem_bloco.get(t.bloco(), 9), t.nn, t.modelo or ""))
    return out


def _auditoria(pasta: Path) -> Optional[dict]:
    try:
        from audit_interactions import audit, load_interactions
        recs = load_interactions(str(pasta))
        return audit(recs) if recs else None
    except Exception as exc:
        print(f"  (auditoria indisponível: {exc})")
        return None


def gerar_tudo(pasta: Path, saida: str = "tabelas", png: bool = True, docx: bool = True,
               audit: Optional[dict] = None, verbose: bool = True) -> List[Path]:
    """Escreve .tex (em <pasta>/<saida>/), .png (na raiz de <pasta>), INDICE.md, todas_as_tabelas.tex/.docx."""
    pasta = Path(pasta)
    D = carregar_tudo(pasta)
    D["_pasta"] = pasta
    if not any(isinstance(v, pd.DataFrame) for v in D.values()):
        if verbose:
            print(f"  Nenhum CSV de resultados em {pasta}; nenhuma tabela gerada.")
        return []
    if audit is None:
        audit = _auditoria(pasta)
    tabs = catalogo(D, audit)
    out = pasta / saida
    out.mkdir(exist_ok=True)
    escritos: List[Path] = []
    indice = ["# Índice das tabelas", "", f"Execução: `{pasta.name}`. Cada tabela existe em `.tex` (esta pasta) e em `.png` (raiz da execução).", "",
              "| PNG (raiz da execução) | TeX | Modelo | Conteúdo |", "|---|---|---|---|"]
    for t in tabs:
        try:
            f_tex = out / t.nome_tex()
            f_tex.write_text(t.tex(), encoding="utf-8")
            escritos.append(f_tex)
            f_png = None
            if png:
                f_png = t.png(pasta / t.nome_png())
                escritos.append(f_png)
            indice.append(f"| `{f_png.name if f_png else '—'}` | `{f_tex.name}` | {t.modelo or '—'} | {_plain_caption(t.caption)} |")
        except Exception as exc:   # uma tabela ruim não derruba as demais
            print(f"  ⚠️ tabela {t.slug} ({t.modelo}): {type(exc).__name__}: {exc}")
    (out / "INDICE.md").write_text("\n".join(indice) + "\n", encoding="utf-8")
    escritos.append(out / "INDICE.md")
    tex_files = [p for p in escritos if p.suffix == ".tex"]
    f_doc = out / "todas_as_tabelas.tex"
    f_doc.write_text(b1.documento([str(p) for p in tex_files], "Tabelas da execução"), encoding="utf-8")
    escritos.append(f_doc)
    if docx:
        d = b1.exportar_docx(tex_files, out / "todas_as_tabelas.docx")
        if d is not None:
            escritos.append(d)
    if verbose:
        n_png = sum(1 for p in escritos if p.suffix == ".png")
        print(f"  Tabelas: {len(tex_files)} .tex em {out.name}/ e {n_png} .png na raiz da execução (índice em {out.name}/INDICE.md)")
    return escritos


def _plain_caption(c: str) -> str:
    from tabelas import latex_para_texto
    return latex_para_texto(c).replace("|", "/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pasta", nargs="?", default=None)
    ap.add_argument("--saida", default="tabelas")
    ap.add_argument("--sem-png", action="store_true")
    ap.add_argument("--sem-docx", action="store_true")
    args = ap.parse_args(argv)
    if args.pasta:
        pasta = Path(args.pasta)
        if not pasta.is_absolute():
            pasta = (BASE_DIR / pasta) if (BASE_DIR / pasta).exists() else (RESULTADOS_DIR / pasta)
    else:
        cands = sorted(RESULTADOS_DIR.glob("execucao_*_completa"))
        if not cands:
            raise SystemExit("Nenhuma execução _completa em resultados/.")
        pasta = cands[-1]
    escritos = gerar_tudo(pasta, args.saida, png=not args.sem_png, docx=not args.sem_docx)
    print(f"{len(escritos)} arquivos em {pasta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
