"""Correção offline das métricas "vs rótulo real" do Bloco 3 (peso × altura).

Até 25/08/2026 `HM_CLASS_NAME_VARIANTS` trazia ("Homem", "Mulher"), fazendo a resposta
"Homem" do LLM virar `y_llm = 0`, enquanto na base (`Hghtwght_desired.txt`, coluna
`male?`) a classe 1 é o homem. Todas as comparações com `y_true` da variante semântica
(`problem_name == "homem_mulher"`) saíram, portanto, com a convenção invertida.

Como o problema é binário, a correção é exata e não exige nova coleta do LLM:

- a acurácia do LLM vs real vira ``1 - valor`` (troca direta do rótulo);
- a acurácia da métrica vs real também vira ``1 - valor``: trocar todos os rótulos
  troca a ordem dos dois centróides, o Perceptron e o NNLS estimam o MESMO w (a
  atualização e as linhas de A dependem apenas de "centróide correto vs rival", que
  são permutados juntos) e a regra de centróide mais próximo devolve a predição
  trocada; o mesmo vale para o Perceptron baseline da Fase E;
- contagens de erro viram ``n - erros``; fidelidade, kappa e F1 vs LLM não mudam.

A variante neutra (`homem_mulher_classesAB`) e a meia-lua não são afetadas.

Uso:
    python src/corrigir_mapeamento_homem_mulher.py                 # última _completa
    python src/corrigir_mapeamento_homem_mulher.py execucao_2026-07-05_12-34-52_completa

Escreve, ao lado de cada CSV afetado, uma cópia ``*_corrigido_hm.csv`` (os originais
não são alterados) e imprime as tabelas usadas no artigo (médias das 3 sementes).
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execucao_io import asset_variant  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
PROBLEMA_AFETADO = "homem_mulher"
SUFIXO = "_corrigido_hm"


def ultima_execucao_completa() -> Path:
    pastas = sorted(BASE_DIR.glob("execucao_*_completa"))
    if not pastas:
        raise SystemExit("nenhuma pasta execucao_*_completa encontrada")
    return pastas[-1]


def _flip(df: pd.DataFrame, mask: pd.Series, col_n: str | None) -> pd.DataFrame:
    """Inverte as colunas *_vs_true (acurácias) e *errors*_vs_true (contagens) em `mask`."""
    df = df.copy()
    for col in df.columns:
        if not col.endswith("_vs_true"):
            continue
        if "errors" in col:
            if col_n is None:
                raise ValueError(f"coluna de contagem {col} sem coluna de total")
            df.loc[mask, col] = df.loc[mask, col_n] - df.loc[mask, col]
        else:
            df.loc[mask, col] = 1.0 - df.loc[mask, col]
    return df


def corrigir_phase_a(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    mask = df["problem_name"] == PROBLEMA_AFETADO
    return _flip(df, mask, "n_samples")


def corrigir_phase_e(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    mask = df["problem_name"] == PROBLEMA_AFETADO
    return _flip(df, mask, "n_test")


def corrigir_cross_linearity(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    mask = df["problem"] == PROBLEMA_AFETADO
    return _flip(df, mask, None)


def _salvar(df: pd.DataFrame, original: Path) -> Path:
    # Sufixo ANTES do alias do modelo (final_cross_linearity_corrigido_hm__gpt4mini.csv)
    destino = Path(asset_variant(str(original), SUFIXO.lstrip("_")))
    df.to_csv(destino, index=False)
    return destino


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def relatorio(pa: pd.DataFrame, pe: pd.DataFrame) -> None:
    hm = pa[pa.problem_name.isin([PROBLEMA_AFETADO, PROBLEMA_AFETADO + "_classesAB"])]

    print("\n== Tabela tab:real_a (2 atributos, gpt-4o-mini, médias das 3 sementes) ==")
    sub = hm[(hm.problem_name == PROBLEMA_AFETADO) & (hm.n_features == 2)
             & hm.model.str.contains("gpt")]
    g = sub.groupby("feature_names")[["fidelity_perc_vs_llm", "llm_accuracy_vs_true",
                                      "accuracy_perc_vs_true", "n_errors_perc_vs_true",
                                      "n_samples"]].mean()
    for nome, r in g.iterrows():
        print(f"  {nome:12s} fid_perc={_pct(r.fidelity_perc_vs_llm)}  LLM_real={_pct(r.llm_accuracy_vs_true)}  "
              f"metrica_real={_pct(r.accuracy_perc_vs_true)}  erros≈{r.n_errors_perc_vs_true:.1f}/{r.n_samples:.0f}")
    print("  por semente (metrica_real perc):")
    for (nome, seed), r in sub.groupby(["feature_names", "seed"]).mean(numeric_only=True).iterrows():
        print(f"    {nome:12s} seed={seed:<4} LLM_real={_pct(r.llm_accuracy_vs_true)}  metrica_real={_pct(r.accuracy_perc_vs_true)}")

    print("\n== Tabela tab:real_nfeat (gpt-4o-mini, 2 vs 4 atributos, médias das 3 sementes) ==")
    for pname, rotulo in ((PROBLEMA_AFETADO, "homem/mulher"), (PROBLEMA_AFETADO + "_classesAB", "A/B")):
        sub = hm[(hm.problem_name == pname) & hm.model.str.contains("gpt")]
        g = sub.groupby(["feature_names", "n_features"])[["fidelity_perc_vs_llm", "fidelity_nnls_vs_llm",
                                                          "accuracy_perc_vs_true", "accuracy_nnls_vs_true"]].mean()
        for (nome, nf), r in g.iterrows():
            print(f"  {rotulo:12s} {nome:12s} nfeat={nf}  fid_perc={_pct(r.fidelity_perc_vs_llm)}  "
                  f"fid_nnls={_pct(r.fidelity_nnls_vs_llm)}  metrica_real_perc={_pct(r.accuracy_perc_vs_true)}  "
                  f"metrica_real_nnls={_pct(r.accuracy_nnls_vs_true)}")

    print("\n== Tabela tab:real_ab (acurácia Perceptron vs real, 2 atributos, gpt-4o-mini) ==")
    sub = hm[(hm.n_features == 2) & hm.model.str.contains("gpt")]
    g = sub.groupby(["problem_name", "feature_names"])[["accuracy_perc_vs_true", "n_errors_perc_vs_true",
                                                        "llm_accuracy_vs_true", "n_samples"]].mean()
    for (pname, nome), r in g.iterrows():
        print(f"  {pname:24s} {nome:12s} acc_real={r.accuracy_perc_vs_true:.3f}  erros≈{r.n_errors_perc_vs_true:.0f}/{r.n_samples:.0f}  "
              f"LLM_real={r.llm_accuracy_vs_true:.3f}")
    print("  por semente:")
    for (pname, nome, seed), r in sub.groupby(["problem_name", "feature_names", "seed"]).mean(numeric_only=True).iterrows():
        print(f"    {pname:24s} {nome:12s} seed={seed:<4} acc_real={r.accuracy_perc_vs_true:.3f}")

    print("\n== Gemini (2 atributos, médias das 3 sementes) ==")
    sub = hm[(hm.n_features == 2) & hm.model.str.contains("gemini")]
    g = sub.groupby(["problem_name", "feature_names"])[["fidelity_perc_vs_llm", "llm_accuracy_vs_true",
                                                        "accuracy_perc_vs_true"]].mean()
    for (pname, nome), r in g.iterrows():
        print(f"  {pname:24s} {nome:12s} fid_perc={_pct(r.fidelity_perc_vs_llm)}  LLM_real={_pct(r.llm_accuracy_vs_true)}  "
              f"metrica_real={_pct(r.accuracy_perc_vs_true)}")

    print("\n== Tabela tab:real_e (Fase E, gpt-4o-mini, peso/altura, por semente; LLM real e Perceptron real, %) ==")
    sub = pe[(pe.problem_name == PROBLEMA_AFETADO) & pe.model.str.contains("gpt")
             & (pe.feature_names == "peso/altura")]
    g = sub.groupby(["seed", "n_shot"])[["accuracy_llm_vs_true", "accuracy_perceptron_baseline_vs_true",
                                          "accuracy_llm_vs_metric"]].mean()
    for (seed, n_shot), r in g.iterrows():
        perc = "--" if pd.isna(r.accuracy_perceptron_baseline_vs_true) else _pct(r.accuracy_perceptron_baseline_vs_true)
        print(f"  seed={seed:<4} n_shot={n_shot:<3} LLM_real={_pct(r.accuracy_llm_vs_true)}  Perc_real={perc}  "
              f"LLM_vs_metrica={_pct(r.accuracy_llm_vs_metric)}")


def main() -> None:
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else ultima_execucao_completa()
    if not pasta.is_absolute():
        pasta = BASE_DIR / pasta
    print(f"Execução: {pasta.name}")

    escritos = []
    pa_path = next(p for p in pasta.glob("bloco23_external_phase_a_*.csv") if SUFIXO not in p.stem)
    pe_path = next(p for p in pasta.glob("bloco23_external_phase_e_*.csv") if SUFIXO not in p.stem)
    pa = corrigir_phase_a(pa_path)
    pe = corrigir_phase_e(pe_path)
    escritos += [_salvar(pa, pa_path), _salvar(pe, pe_path)]
    for cl_path in sorted(pasta.glob("final_cross_linearity*.csv")):
        if SUFIXO in cl_path.stem:
            continue
        escritos.append(_salvar(corrigir_cross_linearity(cl_path), cl_path))

    print("Arquivos corrigidos (originais preservados):")
    for p in escritos:
        print(f"  {p.name}")
    relatorio(pa, pe)


if __name__ == "__main__":
    main()
