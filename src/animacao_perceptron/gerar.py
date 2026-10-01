"""Gera `algoritmo_perceptron_animado.html` (raiz do projeto): página única, sem
dependências externas, que anima passo a passo o Perceptron Estruturado com
Relaxação de Margem implementado em `src/relaxed_perceptron.py`.

O que o script faz:
  1. Monta todos os conjuntos em R² que o LLM rotulou em zero-shot na execução completa
     mais recente, com o prompt padrão de cada experimento, nos dois modelos e nas três
     sementes: Problemas A, B, C (lineares, Bloco 1), Problema E (linear, Bloco 2),
     meia-lua (não linear; conjunto de treino 70% da Fase A externa) e peso × altura (não
     linear, base real; treino 70%, quatro variantes de enunciado). Os rótulos vêm do lote
     de coleta daquele conjunto em `llm_interactions_parte*.json`. Só os rótulos do LLM
     entram na página; o ground truth não é usado. Conjuntos em que o LLM deu a mesma
     classe a todos os pontos são omitidos (sem dois centróides o algoritmo não roda).
  2. Roda `train_relaxed_perceptron` (PERCEPTRON_PARAMS + use_best_effort=True) em
     cada conjunto e guarda w, γ, gamma_history e o aviso emitido: é a referência
     que a página compara, bit a bit, com o porte em JavaScript (`core.js`).
  3. Injeta dados, referência, o texto do método `fit` e o porte JS em `template.html`.

Uso (raiz do projeto, .venv ativo):
    python src/animacao_perceptron/gerar.py [pasta_execucao_completa]
"""
import contextlib
import csv
import glob
import io
import json
import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from relaxed_perceptron import train_relaxed_perceptron  # noqa: E402
from metrics import compute_centroids  # noqa: E402
from dissertacao_mestrado import PERCEPTRON_PARAMS  # noqa: E402
from data_problems import create_problem_d_meialua, create_problem_homem_mulher  # noqa: E402

MODELOS = [("gpt-4o-mini", "gpt4mini"), ("google/gemini-2.5-flash-lite", "flashlite")]
SEEDS = [42, 123, 7]


def execucao_completa_mais_recente() -> str:
    pastas = sorted(glob.glob(os.path.join(RAIZ, "execucao_*_completa")))
    if not pastas:
        raise SystemExit("Nenhuma pasta execucao_*_completa encontrada.")
    return pastas[-1]


def ler_csv(path):
    with open(path) as f:
        return [[float(r["x1"]), float(r["x2"])] for r in csv.DictReader(f)]


def chave(a, b):
    return (round(a, 10), round(b, 10))


def prompt_padrao(p, c0, c1, f0):
    """Prompt zero-shot padrão em 2D (sem exemplos), classes c0/c1 nessa ordem, 1º atributo f0."""
    return ("xample" not in p and "for 2D points.\n" in p and f"Class {c0} or Class {c1}." in p
            and "Answer ONLY" in p and f"\n{f0} = " in p)


def rotulos_do_lote(inter, X, modelo, c0, c1, f0, rot1):
    """Rótulos do lote de coleta deste conjunto: a primeira sequência contígua de chamadas
    (mesmo modelo e prompt) cujos pontos são exatamente os de X. Cada semente usa a sua
    própria coleta, mesmo quando os pontos se repetem entre sementes (flips em T=0)."""
    quer = {chave(*x): i for i, x in enumerate(X)}; n = len(X)
    L = [e for e in inter if e["model"] == modelo and prompt_padrao(e["prompt"], c0, c1, f0)]
    ks = [chave(e["point"]["x1"], e["point"]["x2"]) for e in L]
    for i in range(len(L) - n + 1):
        jan = ks[i:i + n]
        if jan[0] in quer and len(set(jan)) == n and all(k in quer for k in jan):
            lab = {quer[k]: L[i + j]["parsed_label"] for j, k in enumerate(jan)}
            return [1 if lab[q] == rot1 else 0 for q in range(n)]
    return None


def treino_70(X, seed):
    """Conjunto de treino da Fase A externa (run_external_problem_pipeline, n_train_ratio=0.7)."""
    idx = np.random.RandomState(seed).permutation(len(X))
    return [X[i] for i in idx[:int(0.7 * len(X))]]


def extrair_datasets(execucao: str):
    interacoes = []
    for f in sorted(glob.glob(os.path.join(execucao, "llm_interactions_parte*.json"))) or [
        os.path.join(execucao, "llm_interactions.json")
    ]:
        with open(f) as fh:
            interacoes.extend(json.load(fh))
    print(f"  {len(interacoes)} interações lidas de {os.path.basename(execucao)}")
    ext = pd.read_csv(sorted(glob.glob(os.path.join(execucao, "bloco23_external_phase_a_*_corrigido_hm.csv"))
                             or glob.glob(os.path.join(execucao, "bloco23_external_phase_a_*.csv")))[0])
    ext = ext[ext.n_features == 2]
    b1 = pd.read_csv(glob.glob(os.path.join(execucao, "bloco1_phases_abc_*.csv"))[0])
    b1 = b1[(b1.class_0 == "A") & (b1.class_1 == "B") & (b1.feature_0 == "x1") & (b1.prompt_variant == "default")]

    G = {"A": "Problema A (linear, Bloco 1: estimação da métrica)",
         "B": "Problema B (linear, Bloco 1: rotação horária)",
         "C": "Problema C (linear, Bloco 1: rotação anti-horária)",
         "E": "Problema E (linear, Bloco 2: perito W = [0,3; 1,5])",
         "D": "Meia-lua (não linear, Fase A externa)",
         "HM": "Peso × altura (não linear, base real, Fase A externa)"}
    cand = []   # (chave, grupo, nome, modelo, X, (c0, c1, f0, rot1), w_csv)
    Xh, _ = create_problem_homem_mulher()
    for seed in SEEDS:
        base = os.path.join(execucao, f"dados_sinteticos_seed{seed}")
        for p in "ABC":
            X = ler_csv(os.path.join(base, f"problem_{p}.csv"))
            for modelo, alias in MODELOS:
                w_csv = None
                if p == "A":
                    r = b1[(b1.model == modelo) & (b1.random_seed == seed)]
                    w_csv = [float(r.w_0.iloc[0]), float(r.w_1.iloc[0])] if len(r) else None
                cand.append((f"{p}_seed{seed}_{alias}", p, f"{G[p].split(' (')[0]}, seed {seed}, {modelo}", modelo, X, ("A", "B", "x1", "B"), w_csv))
        # problem_D.csv guarda o Problema E (perito linear do Bloco 2; create_problem_e_expert)
        X = ler_csv(os.path.join(base, "problem_D.csv"))
        for modelo, alias in MODELOS:
            cand.append((f"E_seed{seed}_{alias}", "E", f"Problema E, seed {seed}, {modelo}", modelo, X, ("A", "B", "x1", "B"), None))
        Xd = treino_70(create_problem_d_meialua(n_samples=150, random_state=seed)[0].tolist(), seed)
        for modelo, alias in MODELOS:
            r = ext[(ext.model == modelo) & (ext.problem_name == f"meia_lua_seed{seed}") & (ext.seed == seed)]
            cand.append((f"D_seed{seed}_{alias}", "D", f"Meia-lua, seed {seed}, {modelo} (treino, 105 pontos)", modelo, Xd, ("A", "B", "x1", "B"),
                         [float(r.w_perc_0.iloc[0]), float(r.w_perc_1.iloc[0])] if len(r) else None))
        Xt = treino_70(Xh.tolist(), seed)
        for c0, c1, rot1, pname, cl in [("Homem", "Mulher", "Homem", "homem_mulher", "Homem/Mulher"), ("A", "B", "B", "homem_mulher_classesAB", "A/B")]:
            for f0, fn in [("x1", "x1/x2"), ("peso", "peso/altura")]:
                for modelo, alias in MODELOS:
                    r = ext[(ext.model == modelo) & (ext.problem_name == pname) & (ext.seed == seed) & (ext.feature_names == fn)]
                    cand.append((f"HM_seed{seed}_{'HM' if c0 == 'Homem' else 'AB'}_{f0}_{alias}", "HM",
                                 f"Peso × altura, seed {seed}, {modelo}, classes {cl}, atributos {fn} (treino, 70 pontos)",
                                 modelo, Xt, (c0, c1, f0, rot1), [float(r.w_perc_0.iloc[0]), float(r.w_perc_1.iloc[0])] if len(r) else None))

    ordem = ["A", "B", "C", "E", "D", "HM"]
    cand.sort(key=lambda c: ordem.index(c[1]))   # estável: mantém seed e modelo na ordem
    datasets, omitidos = {}, []
    for k, grupo, nome, modelo, X, (c0, c1, f0, rot1), w_csv in cand:
        y = rotulos_do_lote(interacoes, X, modelo, c0, c1, f0, rot1)
        if y is None:
            omitidos.append(f"{nome}: rótulos do LLM incompletos no registro"); continue
        if min(y) == max(y):
            omitidos.append(f"{nome}: o LLM deu a mesma classe aos {len(X)} pontos"); continue
        datasets[k] = {"nome": nome, "grupo": G[grupo], "problema": grupo, "linear": grupo in "ABCE",
                       "X": X, "y": y, "origem": "llm", "w_csv": w_csv}
    print(f"  {len(datasets)} conjuntos; {len(omitidos)} omitidos")
    for o in omitidos:
        print("   omitido:", o)
    return datasets, omitidos


def referencia_python(datasets: dict) -> dict:
    ref = {}
    for nome, d in datasets.items():
        X = np.array(d["X"]); y = np.array(d["y"])
        c = compute_centroids(X, y)
        buf = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(buf):
                w, g, hist = train_relaxed_perceptron(
                    X, y, c, **PERCEPTRON_PARAMS, verbose=True, use_best_effort=True, return_history=True,
                )
        csv_ok = None if d["w_csv"] is None else bool(np.allclose(w, d["w_csv"], rtol=0, atol=1e-12))
        ref[nome] = {"w": w.tolist(), "gamma": float(g), "history": hist, "trace": buf.getvalue(), "n1": int(y.sum()), "csv_ok": csv_ok}
        print(f"  {nome:30s} w={np.round(w, 6)} γ={g:.6f} iterações={len(hist):2d} "
              f"{'viável' if g > 0 else 'melhor esforço':14s} CSV: {'-' if csv_ok is None else ('igual' if csv_ok else 'DIFERENTE')}")
    return ref


def fonte_fit():
    with open(os.path.join(RAIZ, "src", "relaxed_perceptron.py"), encoding="utf-8") as f:
        linhas = f.read().split("\n")
    ini = next(i for i, l in enumerate(linhas) if l.startswith("    def fit("))
    fim = next(i for i, l in enumerate(linhas) if l.startswith("def train_relaxed_perceptron"))
    while not linhas[fim - 1].strip():
        fim -= 1
    return "\n".join(linhas[ini:fim]), ini + 1


def fonte_fitgen(core: str, nome: str = "fitGen"):
    """Trecho de core.js que a animação executa (function* <nome> ... até o fecho)."""
    linhas = core.split("\n")
    ini = next(i for i, l in enumerate(linhas) if l.startswith(f"function* {nome}("))
    fim = next(i for i in range(ini + 1, len(linhas)) if linhas[i] == "}")
    return "\n".join(linhas[ini:fim + 1]), ini + 1


def main():
    execucao = sys.argv[1] if len(sys.argv) > 1 else execucao_completa_mais_recente()
    print("Extraindo conjuntos de dados...")
    datasets, omitidos = extrair_datasets(execucao)
    print("Rodando a referência em Python...")
    ref = referencia_python(datasets)
    fonte, primeira = fonte_fit()
    with open(os.path.join(AQUI, "template.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(AQUI, "core.js"), encoding="utf-8") as f:
        core = f.read()
    js_fontes = {}
    for chave, nome in [("codigo", "fitGen"), ("artigo", "fitGenArtigo"), ("professor", "fitGenProfessor")]:
        src, primeira_js = fonte_fitgen(core, nome)
        js_fontes[chave] = {"src": src, "first": primeira_js}
    esc = lambda s: s.replace("</script", "<\\/script")
    html = (html.replace("__DATASETS__", esc(json.dumps(datasets, ensure_ascii=False)))
                .replace("__OMITIDOS__", esc(json.dumps(omitidos, ensure_ascii=False)))
                .replace("__REFERENCE__", esc(json.dumps(ref, ensure_ascii=False)))
                .replace("__PY_SOURCE__", esc(json.dumps(fonte, ensure_ascii=False)))
                .replace("__PY_FIRST_LINE__", str(primeira))
                .replace("__JS_SOURCES__", esc(json.dumps(js_fontes, ensure_ascii=False)))
                .replace("__CORE_JS__", esc(core))
                .replace("__DATA_GERACAO__", date.today().strftime("%d/%m/%Y")))
    saida = os.path.join(RAIZ, "algoritmo_perceptron_animado.html")
    with open(saida, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Página gerada: {saida} ({os.path.getsize(saida) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
