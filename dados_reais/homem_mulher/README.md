# Base real peso × altura (homem / mulher)

Base enviada pelo orientador, usada no Bloco 3 (Problema G) como estudo de caso real.
São 100 pessoas descritas por dois atributos, peso e altura, com o sexo como rótulo.

## Arquivos

| Arquivo | Conteúdo | Linhas |
|---|---|---|
| `Hghtwght_input.txt` | Arquivo-fonte original (formato NeuroSolutions). Cabeçalho `weight` `height`, separado por tabulação, 100 linhas de dados em notação científica. | 1 + 100 |
| `Hghtwght_desired.txt` | Arquivo-fonte original dos rótulos. Cabeçalho `male?`, uma linha por pessoa, valores `-1.00E+00` ou `1.00E+00`. | 1 + 100 |
| `Hghtwght_input.asc`, `Hghtwght_desired.asc` | Cópias idênticas, byte a byte, dos dois `.txt` acima (mesmo conteúdo, outra extensão). | idem |
| `peso_altura.csv` | Versão consolidada usada pelo código (`create_problem_homem_mulher` em `src/data_problems.py`). Colunas `peso,altura,classe`. | 1 + 100 |
| `peso_altura_kg_m.csv` | A mesma base convertida para kg e metros pela regra de três do orientador (seção "Conversão para kg e metros"). Colunas `peso_kg,altura_m,classe,sexo` (`sexo` = H/M). Mesma ordem de linhas. | 1 + 100 |
| `amostras_orientador_2026-08-25/` | Amostras de treino/teste sorteadas para o experimento com 3 pessoas (ver o README de lá). | |

A linha `i` de `Hghtwght_input.txt` corresponde à linha `i` de `Hghtwght_desired.txt` e à
linha `i` de `peso_altura.csv` (mesma ordem, sem embaralhar).

## Qual classe é homem e qual é mulher

No arquivo-fonte a coluna de rótulo chama-se `male?` ("é homem?"):

| `Hghtwght_desired.txt` (`male?`) | `peso_altura.csv` (`classe`) | Significado | Quantidade |
|---|---|---|---|
| `-1.00E+00` | `0` | **Mulher** | 50 |
| `1.00E+00` | `1` | **Homem** | 50 |

Ou seja, `classe = (male? + 1) / 2`. A base é balanceada, 50 mulheres e 50 homens.

Isso é coerente com os dados: a classe 1 tem média de peso e de altura maior que a classe 0
(tabela abaixo). Esta é a convenção adotada no código desde 26/08/2026
(`G_NOMES_CLASSE = ("Mulher", "Homem", ...)`, até 13/10/2026 `HM_CLASS_NAME_VARIANTS`, prompt "Mulher ou Homem"). Até
25/08/2026 o código tratava a classe 0 como "Homem", por isso os CSVs do Bloco 3 da execução
`2026-07-05` foram recalculados offline por uma ferramenta de correção (removida do repositório em 13/10/2026; as cópias `*_corrigido_hm*.csv` continuam na execução).

## Conferência do CSV contra os arquivos-fonte (08/10/2026)

Verificado por script: as colunas `peso` e `altura` são **idênticas** (diferença máxima 0,0)
às colunas `weight` e `height` de `Hghtwght_input.txt`, nas 100 linhas e na mesma ordem.
A coluna `classe` coincide em 100 de 100 linhas com `(male? + 1) / 2` de
`Hghtwght_desired.txt`, e em 0 de 100 com a convenção invertida. Os `.asc` são cópias
exatas dos `.txt`.

A única diferença é de escrita: o CSV usa decimal comum (`-0.44`) onde o arquivo-fonte usa
notação científica (`-4.40E-01`). O valor numérico é o mesmo.

## Notação científica (`E-01`, `E-02`, `E+00`)

Os arquivos-fonte escrevem os números no formato `mantissa E expoente`, que significa
`mantissa × 10^expoente`. O `E` é o "vezes dez elevado a":

| Escrito no arquivo | Conta | Valor |
|---|---|---|
| `-4.40E-01` | −4,40 × 10⁻¹ = −4,40 ÷ 10 | **−0,440** |
| `-8.41E-02` | −8,41 × 10⁻² = −8,41 ÷ 100 | **−0,0841** |
| `7.95E-02` | 7,95 × 10⁻² | 0,0795 |
| `1.00E+00` | 1,00 × 10⁰ = 1,00 × 1 | 1,0 |
| `-1.00E+00` | −1,00 × 10⁰ | −1,0 |

Regra prática: `E-01` desloca a vírgula uma casa para a esquerda, `E-02` duas casas,
`E+00` não desloca. A primeira linha de dados, `-4.40E-01  -8.41E-02`, é portanto uma pessoa
com peso −0,440 e altura −0,0841 (na escala normalizada), rotulada `-1.00E+00`, ou seja,
mulher (classe 0).

## Escala dos atributos

Os valores de `peso_altura.csv` **não estão em kg nem em metros**. A base veio já normalizada
pelo NeuroSolutions para o intervalo [−1, 1]: o menor peso das 100 amostras vale exatamente −1
e o maior exatamente +1, o mesmo para a altura. Não há registro dos valores originais.
O experimento do Bloco 3 usa os valores normalizados; a versão em unidades físicas está em
`peso_altura_kg_m.csv`.

## Conversão para kg e metros (`peso_altura_kg_m.csv`)

Procedimento indicado pelo orientador (e-mail de 28/08/2026): associar o menor valor das
100 amostras a um peso mínimo plausível, obter os demais por regra de três e calibrar o
mínimo até que o máximo caia numa escala apropriada (nem 150 kg, nem baixo demais). Como a
normalização do NeuroSolutions é linear, a conversão inversa também é linear e a proporção
entre diferenças de amostras é preservada.

Calibração adotada, a mesma dos arquivos `X'`/`T'` de `amostras_orientador_2026-08-25/`:

| Atributo | valor −1 (mínimo da base) | valor +1 (máximo da base) | Fórmula |
|---|---|---|---|
| peso | 45,0 kg | 110,0 kg | `peso_kg = 45 + (peso + 1) × 65 / 2` |
| altura | 1,50 m | 1,95 m | `altura_m = 1,50 + (altura + 1) × 0,45 / 2` |

Cada unidade da escala normalizada vale 32,5 kg ou 0,225 m. Peso arredondado a 1 casa
decimal, altura a 2 casas (erro de arredondamento máximo 0,05 kg e 0,005 m).
Exemplo com a primeira linha: peso −0,440 → 45 + 0,560 × 32,5 = 63,2 kg; altura −0,0841 →
1,50 + 0,9159 × 0,225 = 1,71 m.

Resultado por classe:

| Sexo | n | Peso mín | Peso máx | Peso médio | Altura mín | Altura máx | Altura média |
|---|---|---|---|---|---|---|---|
| H | 50 | 45,0 kg | 110,0 kg | 82,3 kg | 1,62 m | 1,95 m | 1,77 m |
| M | 50 | 55,7 kg | 80,9 kg | 69,2 kg | 1,50 m | 1,73 m | 1,65 m |

Gerado em 08/10/2026 com `converter_para_fisico` de `src/gerar_amostras_homem_mulher.py`;
conferido que os 90 pontos dos arquivos `X'1..3`/`T'1..3` coincidem com as linhas
correspondentes deste CSV (via `manifest.csv`). Para outras faixas, altere `PESO_KG` /
`ALTURA_M` nesse script (ou use `--peso-min`, `--peso-max`, `--altura-min`, `--altura-max`).

## Descrição dos dados

Estatísticas calculadas sobre `peso_altura.csv` (escala normalizada):

| Classe | n | Peso: média | desvio | mín | máx | Altura: média | desvio | mín | máx |
|---|---|---|---|---|---|---|---|---|---|
| 0 (Mulher) | 50 | −0,254 | 0,188 | −0,670 | 0,106 | −0,324 | 0,220 | −1,000 | 0,026 |
| 1 (Homem) | 50 | 0,148 | 0,410 | −1,000 | 1,000 | 0,180 | 0,295 | −0,453 | 1,000 |

Observações:

- **Homens são, em média, mais pesados e mais altos**, e também muito mais dispersos
  (desvio do peso 0,41 contra 0,19 das mulheres). As mulheres formam um grupo compacto
  no canto inferior esquerdo (peso e altura abaixo da média geral).
- **A classe 1 contém os extremos dos dois lados**: a pessoa mais leve da base (peso −1,0) e a
  mais pesada (+1,0) são ambas homens, assim como a mais alta (+1,0). A pessoa mais baixa
  (altura −1,0) é mulher. Por isso a fronteira ótima não é uma reta: o grupo das mulheres
  fica "envolvido" pelo dos homens, o que leva à fronteira elíptica descrita em
  `create_problem_homem_mulher` (termos x², y² no Passo 4 do G com 4 atributos, R⁴).
- Correlação peso × altura: 0,50 na base inteira, 0,02 dentro das mulheres e 0,28 dentro
  dos homens. A correlação global vem sobretudo da separação entre as classes.
- As classes se sobrepõem perto da origem: 6 mulheres têm peso ou altura acima de zero e
  5 homens têm peso e altura ambos abaixo de zero.
- Separabilidade de referência (reclassificação da própria base): centróide mais próximo
  com distância euclidiana 91 %, discriminante linear 90 %, discriminante quadrático 93 %
  (validação cruzada 10-fold: 90 % linear, 91 % quadrático). Nenhum classificador simples
  passa de ~93 %, o que dá o teto esperado para a métrica estimada no Bloco 3.

## Como carregar

```python
from src.data_problems import create_problem_homem_mulher
X, y = create_problem_homem_mulher()   # X: (100, 2) colunas (peso, altura); y em {0, 1}, 1 = Homem
```
