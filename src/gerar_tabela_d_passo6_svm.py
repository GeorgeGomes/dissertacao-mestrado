"""Gera, OFFLINE (0 chamadas à API), o CSV `bloco1_d_passo6_svm_<ts>.csv` e a tabela
`bloco1_table_19_D_passo6_svm` de uma execução já feita, a partir dos rótulos zero-shot do LLM
gravados em `llm_interactions*.json` (tag `bloco1|problema=D|passo=2|...`), dos pontos em
`dados_sinteticos_seed*/problema_D.csv` e da métrica zero-shot de maior fidelidade em
`bloco1_ad_passos_2_a_5` (Perceptron, R², R³ ou R⁴). Execuções a partir de 16/10/2026 gravam o CSV
na própria execução; esta ferramenta serve para as anteriores. Depois chama `gerar_tabelas.gerar_tudo`
para escrever a tabela (.tex + .png) e atualizar o índice.

Uso (raiz do projeto, .venv ativo):
    python src/gerar_tabela_d_passo6_svm.py [pasta_execucao]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from classical_baselines import d_passo6_svm_medidas  # noqa: E402


def _rotulos_zero_shot(pasta: Path):
    """{(modelo, seed): {(x1, x2): rótulo 0/1}} do Passo 2 do Problema D."""
    out = {}
    arquivos = sorted(glob.glob(str(pasta / "llm_interactions_parte*.json"))) or [str(pasta / "llm_interactions.json")]
    for f in arquivos:
        with open(f, encoding="utf-8") as fh:
            for r in json.load(fh):
                t = r.get("tag", "")
                if "problema=D|passo=2|regime=passo2_zero_shot|k=0|" not in t:
                    continue
                seed = int(t.split("|seed=")[1].split("|")[0])
                pt = r["point"]
                out.setdefault((r["model"], seed), {})[(round(float(pt["x1"]), 6), round(float(pt["x2"]), 6))] = 1 if str(r["parsed_label"]) == "B" else 0
    return out


def main():
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(str(RAIZ / "resultados" / "execucao_*_completa")))[-1])
    ad = pd.read_csv(sorted(pasta.glob("bloco1_ad_passos_2_a_5_*.csv"))[-1])
    zs = ad[(ad.problem == "D") & (ad.regime == "passo2_zero_shot") & (ad.algorithm == "perceptron")]
    rot = _rotulos_zero_shot(pasta)
    linhas = []
    for (modelo, seed), mapa in sorted(rot.items()):
        d = pd.read_csv(pasta / f"dados_sinteticos_seed{seed}" / "problema_D.csv")
        X = d[["x1", "x2"]].values; y_true = d["y"].values.astype(int)
        try:
            y_llm = np.array([mapa[(round(float(a), 6), round(float(b), 6))] for a, b in X])
        except KeyError:
            print(f"  ⚠ {modelo} seed={seed}: rótulos incompletos; pulando"); continue
        sub = zs[(zs.model == modelo) & (zs.random_seed == seed) & (zs.collapsed == 0)]
        metric = None
        if not sub.empty:
            r = sub.sort_values("fidelity_vs_llm", ascending=False).iloc[0]
            metric = {"n_features": int(r.n_features), "fidelity_perc_vs_llm": float(r.fidelity_vs_llm),
                      "accuracy_perc_vs_true": float(r.accuracy_metric_vs_true_full)}
        med = d_passo6_svm_medidas(X, y_true, y_llm, metric=metric)
        prov = ad[ad.model == modelo].provider.iloc[0] if "provider" in ad.columns and (ad.model == modelo).any() else "openrouter"
        med.update({"provider": prov, "model": modelo, "random_seed": seed})
        linhas.append(med)
        print(f"  {modelo:30s} seed={seed:3d}: SVM real {med['svm_real_accuracy_vs_true']:.3f} | "
              + ("LLM colapsado" if med["collapsed"] else f"SVM-LLM fid {med['svm_llm_fidelity_vs_llm']:.3f}, acur {med['svm_llm_accuracy_vs_true']:.3f}"))
    ts = sorted(pasta.glob("bloco1_ad_passos_2_a_5_*.csv"))[-1].stem.split("_")[-2:]
    destino = pasta / f"bloco1_d_passo6_svm_{'_'.join(ts)}.csv"
    pd.DataFrame(linhas).to_csv(destino, index=False)
    print(f"CSV: {destino.name} ({len(linhas)} linhas)")
    import gerar_tabelas as gt
    gt.gerar_tudo(pasta, png=True, docx=False, verbose=False)
    print("Tabelas regeradas (bloco1_table_19_D_passo6_svm__<modelo>).")


if __name__ == "__main__":
    main()
