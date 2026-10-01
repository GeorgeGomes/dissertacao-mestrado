"""Gera as amostras da base homem × mulher pedidas pelo orientador (e-mail de 25/08/2026).

Produz 18 arquivos CSV a partir de ``dados_reais/homem_mulher/peso_altura.csv``:

- ``X1``, ``X2``, ``X3``: 3 conjuntos de TREINO com 20 amostras cada (colunas ``x1,x2``,
  valores normalizados em [-1, 1] como na base).
- ``T1``, ``T2``, ``T3``: 3 conjuntos de TESTE com 10 amostras cada (colunas ``x1,x2``).
- ``X'1`` … ``X'3`` e ``T'1`` … ``T'3``: as MESMAS amostras, na MESMA ordem, com as
  medidas convertidas para kg e metro (colunas ``peso_kg,altura_m``) pelo procedimento
  que o orientador indicou (e-mail de 28/08/2026): ancorar o menor valor das 100
  amostras em um mínimo plausível, o maior em um máximo plausível e obter os demais por
  regra de três (mapa linear), o que preserva a proporcionalidade entre as amostras.
  Faixas padrão: peso 45 a 110 kg, altura 1,50 a 1,95 m (ajustáveis pela CLI).
- ``RX1`` … ``RX3`` e ``RT1`` … ``RT3``: rótulos ``H``/``M`` alinhados linha a linha, válidos
  tanto para a versão normalizada quanto para a versão em kg/m.

Regras de sorteio (``numpy.random.default_rng(seed)``):

- Os 6 conjuntos são totalmente DISJUNTOS entre si (90 das 100 amostras, sem repetição
  dentro de um arquivo nem entre arquivos).
- Balanceados: cada X tem 10 H + 10 M; cada T tem 5 H + 5 M.
- As linhas de cada arquivo são embaralhadas, para as classes não aparecerem agrupadas.

Convenção de rótulo: no arquivo-fonte ``Hghtwght_desired.txt`` a coluna ``male?`` vale +1
para a ``classe 1`` do CSV (a classe mais alta e mais pesada). Logo classe 1 = H (Homem)
e classe 0 = M (Mulher).

Também gera ``manifest.csv`` (rastreabilidade: id da linha na base para cada linha de
cada arquivo) e ``README.md``; com ``--zip``, empacota os 18 CSVs + README para envio.

Uso:
    python src/gerar_amostras_homem_mulher.py            # pasta padrão, seed 42
    python src/gerar_amostras_homem_mulher.py --zip      # idem + zip para o e-mail
    python src/gerar_amostras_homem_mulher.py --peso-min 50 --peso-max 120
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_problems import create_problem_homem_mulher  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_BASE = BASE_DIR / "dados_reais" / "homem_mulher" / "peso_altura.csv"
SAIDA_PADRAO = BASE_DIR / "dados_reais" / "homem_mulher" / "amostras_orientador_2026-08-25"

N_TREINO_POR_CLASSE = 10   # 20 amostras por X (10 H + 10 M)
N_TESTE_POR_CLASSE = 5     # 10 amostras por T (5 H + 5 M)
N_CONJUNTOS = 3

ROTULO_POR_CLASSE = {1: "H", 0: "M"}   # male? = +1  ↔ classe 1
COLUNAS_NEUTRAS = ["x1", "x2"]
COLUNAS_FISICAS = ["peso_kg", "altura_m"]
NOME_ZIP = "amostras_homem_mulher_orientador.zip"

# Faixas de calibração (e-mail do orientador de 28/08/2026): o menor valor da base vira
# o mínimo da faixa, o maior vira o máximo, e o resto segue por regra de três.
PESO_KG: Tuple[float, float] = (45.0, 110.0)
ALTURA_M: Tuple[float, float] = (1.50, 1.95)
CASAS_DECIMAIS = {"peso_kg": 1, "altura_m": 2}


def desnormalizar(v, v_min: float, v_max: float, lo: float, hi: float) -> np.ndarray:
    """Regra de três sobre o intervalo: leva ``v_min`` em ``lo`` e ``v_max`` em ``hi``.

    ``original = lo + (v - v_min) * (hi - lo) / (v_max - v_min)``. É um mapa linear, logo
    a proporção entre diferenças de amostras é preservada (o que o orientador pediu).
    """
    v = np.asarray(v, dtype=float)
    if v_max == v_min:
        raise ValueError("v_max == v_min: não há intervalo para calibrar")
    return lo + (v - v_min) * (hi - lo) / (v_max - v_min)


def converter_para_fisico(X: np.ndarray, ancoras: np.ndarray, peso_kg=PESO_KG,
                          altura_m=ALTURA_M) -> np.ndarray:
    """Converte colunas (peso, altura) normalizadas para (kg, m) e arredonda.

    ``ancoras`` é ``[[v_min_peso, v_min_altura], [v_max_peso, v_max_altura]]`` medido
    nas 100 amostras da base (não no subconjunto), como pede o procedimento.
    """
    kg = desnormalizar(X[:, 0], ancoras[0, 0], ancoras[1, 0], *peso_kg)
    m = desnormalizar(X[:, 1], ancoras[0, 1], ancoras[1, 1], *altura_m)
    return np.column_stack([
        np.round(kg, CASAS_DECIMAIS["peso_kg"]),
        np.round(m, CASAS_DECIMAIS["altura_m"]),
    ])


def sortear_conjuntos(y: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Sorteia índices disjuntos e balanceados para X1..X3 e T1..T3.

    Para cada classe, permuta os índices e distribui, em sequência, 10 para cada X e
    5 para cada T. Depois embaralha a ordem das linhas dentro de cada conjunto.
    Retorna ``{"X1": idx, ..., "T3": idx}`` com índices na base original.
    """
    rng = np.random.default_rng(seed)
    por_classe = {c: rng.permutation(np.flatnonzero(y == c)) for c in ROTULO_POR_CLASSE}

    necessarios = N_CONJUNTOS * (N_TREINO_POR_CLASSE + N_TESTE_POR_CLASSE)
    for c, idx in por_classe.items():
        if len(idx) < necessarios:
            raise ValueError(
                f"classe {c} tem {len(idx)} amostras; precisa de {necessarios} para "
                f"{N_CONJUNTOS} conjuntos disjuntos"
            )

    conjuntos: Dict[str, np.ndarray] = {}
    cursor = {c: 0 for c in por_classe}
    for prefixo, n_por_classe in (("X", N_TREINO_POR_CLASSE), ("T", N_TESTE_POR_CLASSE)):
        for k in range(1, N_CONJUNTOS + 1):
            partes = []
            for c, idx in por_classe.items():
                ini = cursor[c]
                partes.append(idx[ini:ini + n_por_classe])
                cursor[c] = ini + n_por_classe
            juntos = np.concatenate(partes)
            conjuntos[f"{prefixo}{k}"] = rng.permutation(juntos)
    return conjuntos


def resumo_calibracao(X_fis: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Mínimo, máximo e média de peso (kg) e altura (m) por classe, na base inteira."""
    linhas = []
    for c, rotulo in sorted(ROTULO_POR_CLASSE.items(), reverse=True):
        sub = X_fis[y == c]
        linhas.append({
            "classe": rotulo,
            "peso_min": sub[:, 0].min(), "peso_max": sub[:, 0].max(), "peso_media": sub[:, 0].mean(),
            "altura_min": sub[:, 1].min(), "altura_max": sub[:, 1].max(), "altura_media": sub[:, 1].mean(),
        })
    return pd.DataFrame(linhas)


def _readme(seed: int, peso_kg, altura_m, ancoras: np.ndarray, resumo: pd.DataFrame) -> str:
    tabela = "\n".join(
        f"| {r.classe} | {r.peso_min:.1f} | {r.peso_max:.1f} | {r.peso_media:.1f} "
        f"| {r.altura_min:.2f} | {r.altura_max:.2f} | {r.altura_media:.2f} |"
        for r in resumo.itertuples()
    )
    return f"""# Amostras da base homem × mulher para o experimento com 3 pessoas

Gerado por `src/gerar_amostras_homem_mulher.py` com `seed = {seed}` a partir de
`dados_reais/homem_mulher/peso_altura.csv` (100 amostras, 50 H + 50 M).

## Arquivos (18)

| Arquivo | Conteúdo | Linhas | Colunas |
|---|---|---|---|
| `X1.csv`, `X2.csv`, `X3.csv` | treino, 20 amostras (10 H + 10 M), valores normalizados | 20 | `x1,x2` |
| `T1.csv`, `T2.csv`, `T3.csv` | teste, 10 amostras (5 H + 5 M), valores normalizados | 10 | `x1,x2` |
| `X'1.csv`, `X'2.csv`, `X'3.csv` | mesmas amostras e ordem de X1..X3, em kg e m | 20 | `peso_kg,altura_m` |
| `T'1.csv`, `T'2.csv`, `T'3.csv` | mesmas amostras e ordem de T1..T3, em kg e m | 10 | `peso_kg,altura_m` |
| `RX1.csv`, `RX2.csv`, `RX3.csv` | rótulos de X1..X3 e de X'1..X'3 | 20 | `classe` (H/M) |
| `RT1.csv`, `RT2.csv`, `RT3.csv` | rótulos de T1..T3 e de T'1..T'3 | 10 | `classe` (H/M) |

A linha `i` de `RX1.csv` é o rótulo da linha `i` de `X1.csv` e de `X'1.csv` (idem para os
demais). Todos os CSVs têm cabeçalho e separador vírgula.

## Sorteio

- Os 6 conjuntos (60 de treino + 30 de teste) são totalmente disjuntos: nenhuma amostra
  aparece em mais de um arquivo.
- Cada conjunto é balanceado (metade H, metade M) e tem as linhas embaralhadas, para que
  as classes não apareçam agrupadas.
- `manifest.csv` registra, para cada linha de cada arquivo, o índice da amostra na base
  original (`id_base`, começando em 0), permitindo cruzar com o ground truth depois.

## Rótulos

No arquivo-fonte `Hghtwght_desired.txt` a coluna `male?` vale +1 para a `classe 1` do CSV,
que é também a classe mais alta e mais pesada. Portanto `classe 1 = H` e `classe 0 = M`.

## Conversão para kg e metro (arquivos X' e T')

A base veio normalizada (arquivos `Hghtwght_input.txt` do NeuroSolutions), com peso e
altura entre {ancoras[0, 0]:g} e {ancoras[1, 0]:g}. Seguiu-se o procedimento indicado pelo
orientador: o menor valor das 100 amostras é associado a um mínimo plausível, o maior a
um máximo plausível, e todos os demais seguem por regra de três (mapa linear), o que
preserva a proporcionalidade entre as amostras:

```
peso_kg  = {peso_kg[0]:g} + (peso  - ({ancoras[0, 0]:g})) * ({peso_kg[1]:g} - {peso_kg[0]:g}) / ({ancoras[1, 0]:g} - ({ancoras[0, 0]:g}))
altura_m = {altura_m[0]:g} + (altura - ({ancoras[0, 1]:g})) * ({altura_m[1]:g} - {altura_m[0]:g}) / ({ancoras[1, 1]:g} - ({ancoras[0, 1]:g}))
```

Faixas usadas: peso de {peso_kg[0]:g} a {peso_kg[1]:g} kg e altura de {altura_m[0]:.2f} a
{altura_m[1]:.2f} m, calibradas para que o máximo ficasse em uma escala realista. Peso
arredondado a 1 casa decimal e altura a 2 casas. Resultado na base inteira:

| Classe | Peso mín (kg) | Peso máx (kg) | Peso médio (kg) | Altura mín (m) | Altura máx (m) | Altura média (m) |
|---|---|---|---|---|---|---|
{tabela}

Para outras faixas: `python src/gerar_amostras_homem_mulher.py --peso-min 50 --peso-max 120`
(idem `--altura-min`, `--altura-max`).
"""


def gerar(seed: int = 42, saida: Path = SAIDA_PADRAO, csv_base: Path = CSV_BASE,
          fazer_zip: bool = False, peso_kg=PESO_KG, altura_m=ALTURA_M) -> List[Path]:
    """Gera os 18 CSVs, o manifest e o README em ``saida``. Retorna os caminhos escritos."""
    X, y = create_problem_homem_mulher(str(csv_base))
    ancoras = np.vstack([X.min(axis=0), X.max(axis=0)])  # âncoras nas 100 amostras
    X_fis_base = converter_para_fisico(X, ancoras, peso_kg, altura_m)
    resumo = resumo_calibracao(X_fis_base, y)
    conjuntos = sortear_conjuntos(y, seed)

    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    escritos: List[Path] = []
    manifest_linhas = []

    for nome, idx in conjuntos.items():
        prefixo, k = nome[0], nome[1:]
        dados = X[idx]
        rotulos = [ROTULO_POR_CLASSE[int(c)] for c in y[idx]]

        df_neutro = pd.DataFrame(dados, columns=COLUNAS_NEUTRAS)
        df_fisico = pd.DataFrame(X_fis_base[idx], columns=COLUNAS_FISICAS)
        df_rotulo = pd.DataFrame({"classe": rotulos})

        for arquivo, df in (
            (f"{prefixo}{k}.csv", df_neutro),
            (f"{prefixo}'{k}.csv", df_fisico),
            (f"R{prefixo}{k}.csv", df_rotulo),
        ):
            caminho = saida / arquivo
            df.to_csv(caminho, index=False)
            escritos.append(caminho)

        for linha, (id_base, rotulo) in enumerate(zip(idx, rotulos), start=1):
            manifest_linhas.append(
                {"arquivo": f"{prefixo}{k}", "linha": linha,
                 "id_base": int(id_base), "classe": rotulo}
            )

    manifest = saida / "manifest.csv"
    pd.DataFrame(manifest_linhas).to_csv(manifest, index=False)
    escritos.append(manifest)

    readme = saida / "README.md"
    readme.write_text(_readme(seed, peso_kg, altura_m, ancoras, resumo), encoding="utf-8")
    escritos.append(readme)

    if fazer_zip:
        caminho_zip = saida / NOME_ZIP
        with zipfile.ZipFile(caminho_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for caminho in escritos:
                if caminho.name == "manifest.csv":
                    continue  # rastreabilidade interna, não vai para o orientador
                zf.write(caminho, arcname=caminho.name)
        escritos.append(caminho_zip)

    return escritos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    parser.add_argument("--zip", action="store_true", help="empacota os 18 CSVs + README")
    parser.add_argument("--peso-min", type=float, default=PESO_KG[0], help="kg associado ao menor peso da base")
    parser.add_argument("--peso-max", type=float, default=PESO_KG[1], help="kg associado ao maior peso da base")
    parser.add_argument("--altura-min", type=float, default=ALTURA_M[0], help="m associado à menor altura da base")
    parser.add_argument("--altura-max", type=float, default=ALTURA_M[1], help="m associado à maior altura da base")
    args = parser.parse_args()

    peso_kg = (args.peso_min, args.peso_max)
    altura_m = (args.altura_min, args.altura_max)
    escritos = gerar(seed=args.seed, saida=args.saida, fazer_zip=args.zip,
                     peso_kg=peso_kg, altura_m=altura_m)
    print(f"{len(escritos)} arquivos escritos em {args.saida}:")
    for caminho in escritos:
        print(f"  {caminho.name}")

    X, y = create_problem_homem_mulher(str(CSV_BASE))
    ancoras = np.vstack([X.min(axis=0), X.max(axis=0)])
    resumo = resumo_calibracao(converter_para_fisico(X, ancoras, peso_kg, altura_m), y)
    print(f"\nCalibração: peso {peso_kg[0]:g} a {peso_kg[1]:g} kg | altura "
          f"{altura_m[0]:.2f} a {altura_m[1]:.2f} m (âncoras {ancoras[0].tolist()} e {ancoras[1].tolist()})")
    print(resumo.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
