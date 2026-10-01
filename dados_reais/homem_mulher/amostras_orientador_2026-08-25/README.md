# Amostras da base homem × mulher para o experimento com 3 pessoas

Gerado por `src/gerar_amostras_homem_mulher.py` com `seed = 42` a partir de
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
altura entre -1 e 1. Seguiu-se o procedimento indicado pelo
orientador: o menor valor das 100 amostras é associado a um mínimo plausível, o maior a
um máximo plausível, e todos os demais seguem por regra de três (mapa linear), o que
preserva a proporcionalidade entre as amostras:

```
peso_kg  = 45 + (peso  - (-1)) * (110 - 45) / (1 - (-1))
altura_m = 1.5 + (altura - (-1)) * (1.95 - 1.5) / (1 - (-1))
```

Faixas usadas: peso de 45 a 110 kg e altura de 1.50 a
1.95 m, calibradas para que o máximo ficasse em uma escala realista. Peso
arredondado a 1 casa decimal e altura a 2 casas. Resultado na base inteira:

| Classe | Peso mín (kg) | Peso máx (kg) | Peso médio (kg) | Altura mín (m) | Altura máx (m) | Altura média (m) |
|---|---|---|---|---|---|---|
| H | 45.0 | 110.0 | 82.3 | 1.62 | 1.95 | 1.77 |
| M | 55.7 | 80.9 | 69.2 | 1.50 | 1.73 | 1.65 |

Para outras faixas: `python src/gerar_amostras_homem_mulher.py --peso-min 50 --peso-max 120`
(idem `--altura-min`, `--altura-max`).
