"""Renomeia assets JÁ GERADOS (pastas ``execucao_*``) para a convenção de 01/10/2026.

Regras (puras sobre o nome do arquivo, aplicadas nesta ordem):

1. Painel/variante depois do alias → antes do alias (``asset_variant``):
   ``bloco1_06_w_distribution__gpt4mini_boxplot.png``
       → ``bloco1_06_w_distribution_boxplot__gpt4mini.png``
   ``final_cross_linearity__gpt4mini_corrigido_hm.csv``
       → ``final_cross_linearity_corrigido_hm__gpt4mini.csv``
2. Painel "Problema D" do ``final_04_dataset_overview`` mostra o Problema E (perito):
   ``..._seed42_problema_d[__alias].png`` → ``..._seed42_problema_e[__alias].png``
3. SVM da meia-lua é LLM como fonte no Problema D (Bloco 1), não pipeline externo:
   ``bloco23_external_svm_meialua_seedN…`` → ``bloco1_11_meialua_svm_vs_llm_seedN…``
4. ``dados_sinteticos_seed*/problem_D.csv`` guarda o Problema E (perito linear):
   → ``problem_E.csv``

O painel legado ``_algoritmos_r3`` NÃO é renomeado para ``_algoritmos_r3r4`` (o
conteúdo antigo só tem R3). As pastas ``execucao_*`` são rastreadas pelo git, então
arquivos rastreados são movidos com ``git mv`` (os demais com ``os.rename``).

Uso:
    python src/renomear_assets_legado.py                 # dry-run em todas as *_completa
    python src/renomear_assets_legado.py --aplicar       # executa
    python src/renomear_assets_legado.py execucao_2026-07-05_09-24-39_smoke --aplicar
    python src/renomear_assets_legado.py --aplicar --docs  # também artigo.tex, artigo_pandoc.tex, roteiro
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execucao_io import MODEL_ALIAS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PADRAO = ["artigo.tex", "artigo_pandoc.tex", "roteiro_apresentacao.txt"]

_ALIAS = "|".join(re.escape(a) for a in sorted(MODEL_ALIAS.values(), key=len, reverse=True))
RE_PAINEL = re.compile(rf"^(?P<base>.+?)__(?P<alias>{_ALIAS})_(?P<painel>[A-Za-z0-9_]+)\.(?P<ext>png|csv)$")
RE_PROB_D = re.compile(r"^(final_04_dataset_overview_seed\d+)_problema_d((?:__[a-z0-9.-]+)?)\.png$")
RE_SVM = re.compile(r"^bloco23_external_svm_meialua_seed(\d+)")


def regra_painel(nome: str) -> str:
    m = RE_PAINEL.match(nome)
    return f"{m['base']}_{m['painel']}__{m['alias']}.{m['ext']}" if m else nome


def regra_problema_e(nome: str) -> str:
    return RE_PROB_D.sub(r"\1_problema_e\2.png", nome)


def regra_svm(nome: str) -> str:
    return RE_SVM.sub(r"bloco1_11_meialua_svm_vs_llm_seed\1", nome)


def novo_nome(nome: str) -> str:
    """Nome novo (base, sem pasta) para um asset da raiz da execução."""
    for regra in (regra_painel, regra_problema_e, regra_svm):
        nome = regra(nome)
    return nome


def planejar(pasta: Path) -> list[tuple[Path, Path]]:
    """Lista (origem, destino) do que mudaria em ``pasta`` (vazia = já na convenção)."""
    pares = []
    for p in sorted(pasta.iterdir()):
        if p.is_file() and (n := novo_nome(p.name)) != p.name:
            pares.append((p, p.with_name(n)))
    for csv in sorted(pasta.glob("dados_sinteticos_seed*/problem_D.csv")):
        pares.append((csv, csv.with_name("problem_E.csv")))
    return pares


def rastreado(p: Path) -> bool:
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=BASE_DIR,
                           capture_output=True, env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
        return r.returncode == 0
    except OSError:
        return False


def aplicar(pares: list[tuple[Path, Path]], usar_git: bool = True) -> None:
    destinos = [d for _, d in pares]
    if len(set(destinos)) != len(destinos):
        raise SystemExit("colisão: dois arquivos mapeiam para o mesmo destino")
    for _, dst in pares:
        if dst.exists():
            raise SystemExit(f"destino já existe, abortando antes de mover: {dst}")
    for src, dst in pares:
        if usar_git and rastreado(src):
            subprocess.run(["git", "mv", str(src), str(dst)], cwd=BASE_DIR, check=True,
                           env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
        else:
            os.rename(src, dst)


def substituir_em_documento(caminho: Path, mapa: dict[str, str]) -> int:
    """Troca nomes antigos por novos (substituição literal). Devolve nº de ocorrências."""
    texto = caminho.read_text(encoding="utf-8")
    total = 0
    for antigo, novo in sorted(mapa.items(), key=lambda kv: -len(kv[0])):
        n = texto.count(antigo)
        if n:
            texto = texto.replace(antigo, novo)
            total += n
    if total:
        caminho.write_text(texto, encoding="utf-8")
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pastas", nargs="*", help="pastas execucao_* (padrão: todas as *_completa)")
    ap.add_argument("--aplicar", action="store_true", help="executa (sem a flag é dry-run)")
    ap.add_argument("--docs", action="store_true",
                    help=f"também atualiza referências em {', '.join(DOCS_PADRAO)}")
    args = ap.parse_args(argv)

    pastas = [Path(p) for p in args.pastas] or sorted(BASE_DIR.glob("execucao_*_completa"))
    pastas = [p if p.is_absolute() else BASE_DIR / p for p in pastas]
    mapa_docs: dict[str, str] = {}
    for pasta in pastas:
        pares = planejar(pasta)
        print(f"\n== {pasta.name}: {len(pares)} arquivo(s) a renomear ==")
        por_regra = {"painel/alias": 0, "problema_e": 0, "svm→bloco1_11": 0, "problem_E.csv": 0}
        for src, dst in pares:
            if src.name == "problem_D.csv":
                por_regra["problem_E.csv"] += 1
            elif RE_SVM.match(src.name):
                por_regra["svm→bloco1_11"] += 1
            elif RE_PROB_D.match(regra_painel(src.name)):
                por_regra["problema_e"] += 1
            else:
                por_regra["painel/alias"] += 1
            print(f"  {src.relative_to(pasta)}  ->  {dst.relative_to(pasta)}")
            mapa_docs[src.name] = dst.name
        print("  totais:", ", ".join(f"{k}={v}" for k, v in por_regra.items()))
        suspeitos = [p.name for p in pasta.iterdir()
                     if p.is_file() and "__" in p.name and novo_nome(p.name) == p.name
                     and not re.search(rf"__({_ALIAS})\.(png|csv)$", p.name)]
        if suspeitos:
            print("  ⚠ com '__' mas fora de qualquer regra:", suspeitos)
        if args.aplicar and pares:
            aplicar(pares)
            print("  ✓ aplicado")
    if args.docs:
        # Nomes base sem alias (o roteiro cita os assets sem sufixo de modelo)
        extra = {}
        for antigo, novo in mapa_docs.items():
            a0 = re.sub(rf"__({_ALIAS})(?=\.(png|csv)$)", "", antigo)
            n0 = re.sub(rf"__({_ALIAS})(?=\.(png|csv)$)", "", novo)
            if a0 != n0 and a0 != antigo:
                extra[a0] = n0
        mapa_docs.update(extra)
        for doc in DOCS_PADRAO:
            caminho = BASE_DIR / doc
            if not caminho.exists():
                continue
            if args.aplicar:
                n = substituir_em_documento(caminho, mapa_docs)
            else:
                texto = caminho.read_text(encoding="utf-8")
                n = sum(texto.count(a) for a in mapa_docs)
            print(f"  docs: {doc}: {n} ocorrência(s){'' if args.aplicar else ' (dry-run)'}")
    if not args.aplicar:
        print("\n(dry-run — nada foi alterado; use --aplicar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
