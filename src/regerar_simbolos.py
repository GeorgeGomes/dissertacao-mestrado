"""Regera OFFLINE (0 chamadas) as figuras com símbolos dos Passos 2 a 4 (A, D) e 2 e 3 (B, C)
(bloco1_graphic_02 A, bloco1_graphic_11 B/C, bloco1_graphic_13 D; cada uma em ``_color`` e
``_gray``) a partir dos artefatos de uma execução:

- pontos: ``dados_sinteticos_seed*/problema_{A,B,C,D}.csv``;
- rótulos do LLM: ``llm_interactions*.json`` (campo ``tag`` → problema/passo/regime/k/rep/seed;
  ponto casado pelas coordenadas);
- métricas: ``bloco1_ad_passos_2_a_5_*.csv`` (w_0..w_3, exemplos sorteados) com os
  centróides recalculados pelos rótulos do LLM, como no runner;
- melhor métrica do A: ``bloco1_bc_passo3_consistencia_*.csv`` (eleita_passo6=1).

Serve para trocar estilo (marcadores, legendas) sem repetir a coleta.

Uso:
    python src/regerar_simbolos.py                 # última _apenas-bloco1* ou _completa
    python src/regerar_simbolos.py resultados/execucao_2026-10-01_20-41-02_apenas-bloco1
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dissertacao_mestrado as dm  # noqa: E402
from metrics import augment_features, compute_centroids  # noqa: E402
from gerar_tabelas_bloco1 import pasta_padrao, BASE_DIR, RESULTADOS_DIR  # noqa: E402

_TAG = re.compile(r"problema=([A-Za-z0-9_]+)\|passo=\d+\|regime=([A-Za-z0-9_]+)\|k=(-|\d+)\|rep=(\d+)\|seed=(\d+)")


def _pt(p):
    return (round(float(p["x1"]), 4), round(float(p["x2"]), 4))


def _labels_por_grupo(pasta: Path, model: str):
    """{(problema, regime, k, rep, seed): {ponto: rótulo majoritário}}"""
    recs = []
    for f in sorted(pasta.glob("llm_interactions*.json")):
        if "checkpoint" in f.name:   # sobra de execução interrompida: duplicaria registros
            continue
        with open(f, encoding="utf-8") as fh:
            recs += json.load(fh)
    grupos = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("model") != model:
            continue
        m = _TAG.search(str(r.get("tag") or ""))
        if not m:
            continue
        prob, regime, k, rep, seed = m.groups()
        grupos[(prob, regime, k, int(rep), int(seed))][_pt(r["point"])].append(r.get("parsed_label"))
    return {g: {pt: Counter(v).most_common(1)[0][0] for pt, v in d.items()} for g, d in grupos.items()}


def _y_from(labels: dict, X: np.ndarray, c0="A"):
    y = np.full(len(X), -1)
    for i, row in enumerate(X):
        lab = labels.get((round(float(row[0]), 4), round(float(row[1]), 4)))
        if lab is not None:
            y[i] = 0 if lab == c0 else 1
    return y


def _reconstruir(pasta: Path, model: str, seed: int, prob: str, df_a: pd.DataFrame, grupos: dict):
    X_df = pd.read_csv(pasta / f"dados_sinteticos_seed{seed}" / f"problema_{prob}.csv")
    X, y_true = X_df[["x1", "x2"]].values, X_df["y"].values.astype(int)
    sub = df_a[(df_a.model == model) & (df_a.random_seed == seed) & (df_a.problem == prob)]
    labels, metrics = {}, {}
    n = len(X); all_idx = np.arange(n)
    for (regime, k, rep), g in sub.groupby(["regime", "k", "rep"]):
        ex = g["example_indices"].iloc[0]
        ex_idx = np.array([int(v) for v in str(ex).split(";") if v not in ("", "nan")], dtype=int)
        eval_mask = np.ones(n, bool); eval_mask[ex_idx] = False
        if regime == "passo5_teto":
            y_eval = y_true.copy()
        else:
            key_tag = ("ABC" if (prob == "A" and regime == "passo2_zero_shot") else prob, regime, str(int(k)), int(rep), seed)
            lab = grupos.get(key_tag)
            if lab is None and prob == "A" and regime == "passo2_zero_shot":
                lab = grupos.get(("A", "passo2_zero_shot", "0", 0, seed))
            if lab is None:
                continue
            y_full = _y_from(lab, X)
            y_eval = y_full[eval_mask]
            if (y_eval < 0).any():
                continue
        labels[(regime, int(k), int(rep))] = {"eval_idx": all_idx[eval_mask], "example_idx": ex_idx,
                                              "y_llm_eval": y_eval, "examples": []}
        for nf, gg in g.groupby("n_features"):
            r_p = gg[gg.algorithm == "perceptron"].iloc[0]
            collapsed = int(r_p["collapsed"]) == 1
            w = None if collapsed else np.array([r_p[f"w_{i}"] for i in range(int(nf))], float)
            cents = None if collapsed else compute_centroids(augment_features(X[eval_mask], int(nf)), y_eval)
            metrics[(regime, int(k), int(rep), int(nf))] = {"w_perc": w, "w_nnls": None, "centroids": cents,
                                                            "gamma_perc": None, "collapsed": int(collapsed)}
    return {"X": X, "y_true": y_true, "labels": labels, "metrics": metrics, "zero_shot_results": {}}


def regerar(pasta: Path) -> list:
    from renomear_assets_legado import tem_csv_antigo
    if tem_csv_antigo(pasta):
        print(f"execução anterior a 13/10/2026: rode `python src/renomear_assets_legado.py {pasta} --aplicar` antes")
        raise SystemExit(1)
    df_a = pd.read_csv(sorted(pasta.glob("bloco1_ad_passos_2_a_5_*.csv"))[-1])
    tf = sorted(pasta.glob("bloco1_bc_passo3_consistencia_*.csv"))
    df_t = pd.read_csv(tf[-1]) if tf else None
    saved = []
    for model in sorted(df_a.model.unique()):
        grupos = _labels_por_grupo(pasta, model)
        best = None
        if df_t is not None:
            b = df_t[(df_t.model == model) & (df_t.eleita_passo6 == 1)]
            if len(b):
                best = (b.metrica_regime.iloc[0], int(b.metrica_k.iloc[0]), float("nan"))
        for seed in sorted(df_a[df_a.model == model].random_seed.unique()):
            seed = int(seed)
            for prob in ("A", "D"):
                if not ((df_a.model == model) & (df_a.random_seed == seed) & (df_a.problem == prob)).any():
                    continue
                d = _reconstruir(pasta, model, seed, prob, df_a, grupos)
                bc = None
                if prob == "A":
                    lab = grupos.get(("ABC", "passo2_zero_shot", "0", 0, seed))
                    if lab:
                        bc = {}
                        for p_ in ("B", "C"):
                            Xp = pd.read_csv(pasta / f"dados_sinteticos_seed{seed}" / f"problema_{p_}.csv")
                            Xk, yk = Xp[["x1", "x2"]].values, Xp["y"].values.astype(int)
                            ylk = _y_from(lab, Xk)
                            if (ylk < 0).any():
                                bc = None; break
                            bc[f"X_{p_.lower()}"], bc[f"y_{p_.lower()}"], bc[f"y_llm_{p_.lower()}"] = Xk, yk, ylk
                saved += dm._plot_simbolos_passos(d, prob, seed, model, best, bc, str(pasta))
    return saved


def main(argv=None) -> int:
    pasta = Path(argv[0]) if argv else pasta_padrao()
    if pasta is None:
        raise SystemExit("Nenhuma pasta encontrada.")
    if not pasta.is_absolute():
        pasta = (BASE_DIR / pasta) if (BASE_DIR / pasta).exists() else (RESULTADOS_DIR / pasta)
    saved = regerar(pasta)
    print(f"{len(saved)} figuras regeradas em {pasta.name}/")
    for f in saved:
        print("  ", os.path.basename(f))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
