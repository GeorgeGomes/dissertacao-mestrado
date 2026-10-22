"""Pseudocódigo do método `RelaxedPerceptron.fit` (src/relaxed_perceptron.py), fonte ÚNICA
para a página `algoritmo_perceptron_animado.html` (gerar.py) e para o documento
`algoritmo_codigo_executado_pseudocodigo.docx` (gerar_docx.py).

Versão de 16/10/2026: inclui as mudanças de 15/10/2026 no Perceptron (B: w e α divididos
por ||w|| ao fim de cada época; C: melhor esforço = época com menos pontos mal classificados;
D: margem bruta mínima da solução devolvida) e o fator f truncado em [0, 1].

`CORRESPONDENCIA` liga cada linha do pseudocódigo às linhas do .py; `tests/test_animacao_pseudocodigo.py`
confere que cada linha citada contém o trecho esperado, para a tabela não envelhecer sem aviso.
"""
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_PY = os.path.abspath(os.path.join(AQUI, "..", "relaxed_perceptron.py"))

CABECALHO = [
    "Algoritmo (código implementado, RelaxedPerceptron.fit):",
    "Entrada: Dados {(v(i),y(i)), i=1,…,n}, v(i) ∊ R^d, y(i) ∊ {0,1} e centróides c(0) e c(1)",
    "Parâmetros: η, C, γ₀, δ, T, P, tol, s, melhor_esforço",
    "Saída: Vetor de pesos w ∊ R^d₊ (com ||w||₂ = 1), margem γ e margem bruta mínima μ",
    "Início",
]

# (número da linha, texto com a indentação)
LINHAS = [
    (1, " 1  λ ← 1/C;"),
    (2, " 2  Inicializar w ← [1,…,1]; α ← 0; γ ← γ₀;"),
    (3, " 3  γ_inf ← 0; γ_sup ← ∞;"),
    (4, " 4  w* ← w; α* ← α;"),
    (5, " 5  w_melhor ← w; erros_min ← ∞; viável ← F;"),
    (6, " 6  p ← 0;"),
    (7, " 7  Iniciar o gerador de números aleatórios com a semente s;"),
    (8, " 8  Enquanto p < P Faça"),
    (9, " 9      p ← p + 1;"),
    (10, "10      viol_final ← –1;"),
    (11, "11      Para t ← 1 até T Faça"),
    (12, "12          viol ← 0;"),
    (13, "13          π ← permutação aleatória de (1,…,n);"),
    (14, "14          Para cada i em π, nessa ordem, Faça"),
    (15, "15              l ← y(i);"),
    (16, "16              k ← Arg {Min_{j≠l} d_w(v(i),c(j))};"),
    (17, "17              N ← Max {||w||₂, 10⁻⁹};"),
    (18, "18              Se d_w(v(i),c(k)) – d_w(v(i),c(l)) + λ.α_i < γ.N Então"),
    (19, "19                  viol ← viol + 1;"),
    (20, "20                  g_j ← (v_j(i) – c_j(l))² – (v_j(i) – c_j(k))², j = 1,…,d;"),
    (21, "21                  f ← 1 – η.γ/N;"),
    (22, "22                  f ← Min {Max {f, 0}, 1};"),
    (23, "23                  w ← w.f – η.g;"),
    (24, "24                  w ← Max {0, w};"),
    (25, "25                  α ← α.f;"),
    (26, "26                  α_i ← α_i + η;"),
    (27, "27                  α_q ← Min {Max {α_q, 0}, 10⁶}, q = 1,…,n;"),
    (28, "28              Fim_Se;"),
    (29, "29          Fim_Para;"),
    (30, "30          viol_final ← viol;"),
    (31, "31          N ← ||w||₂;"),
    (32, "32          Se N > 0 Então w ← w/N; α ← α/N; Fim_Se;"),
    (33, "33          erros ← número de i em (1,…,n) com d_w(v(i),c(y(i))) > Min_{j≠y(i)} d_w(v(i),c(j));"),
    (34, "34          Se erros < erros_min Então"),
    (35, "35              erros_min ← erros; w_melhor ← w;"),
    (36, "36          Fim_Se;"),
    (37, "37          Se viol = 0 Então"),
    (38, "38              Sair do laço Para em t;"),
    (39, "39          Fim_Se;"),
    (40, "40      Fim_Para;"),
    (41, "41      Se viol_final = 0 Então"),
    (42, "42          viável ← V; γ_inf ← γ; w* ← w; α* ← α;"),
    (43, "43          Se γ_sup = ∞ Então"),
    (44, "44              γ ← γ + δ;"),
    (45, "45          Senão"),
    (46, "46              γ ← Min {γ + δ, (γ_inf + γ_sup)/2};"),
    (47, "47          Fim_Se;"),
    (48, "48      Senão"),
    (49, "49          γ_sup ← γ;"),
    (50, "50          γ ← (γ_inf + γ_sup)/2; w ← w*; α ← α*;"),
    (51, "51      Fim_Se;"),
    (52, "52      Se |γ_sup – γ_inf| ≤ tol Então"),
    (53, "53          Sair do laço Enquanto;"),
    (54, "54      Fim_Se;"),
    (55, "55  Fim_Enquanto;"),
    (56, "56  Se viável Então w_ret ← w*; γ_ret ← γ_inf;"),
    (57, "57  Senão Se melhor_esforço Então w_ret ← w_melhor; γ_ret ← 0;"),
    (58, "58  Senão w_ret ← w*; γ_ret ← 0;"),
    (59, "59  Fim_Se;"),
    (60, "60  μ ← Min_i {[d_w(v(i),c(k_i)) – d_w(v(i),c(y(i)))] / ||w_ret||₂}, k_i = Arg {Min_{j≠y(i)} d_w(v(i),c(j))};"),
    (61, "61  Retorne w_ret, γ_ret (e registre μ);"),
]
RODAPE = ["Fim."]

TEXTO = CABECALHO + [t for _, t in LINHAS] + RODAPE
IDS = [None] * len(CABECALHO) + [n for n, _ in LINHAS] + [None] * len(RODAPE)

# linha do .py que gera cada evento da animação -> linha do pseudocódigo
PY2PSEUDO = {202: 9, 251: 18, 277: 23, 289: 30, 317: 38, 344: 44, 346: 46, 353: 50, 358: 53,
             383: 56, 397: 57, 408: 58}

# Correspondência: linha do pseudocódigo -> (linhas do .py, trecho que DEVE aparecer nessas linhas).
# As linhas de fechamento (Fim_Se, Fim_Para, Fim_Enquanto) não entram.
CORRESPONDENCIA = [
    (1, [162], "lambda_ = 1.0 / self.C"),
    (2, [165, 169, 170], "w = np.ones(d)"),
    (3, [176, 177], "gamma_lo = 0.0"),
    (4, [180, 181], "w_star = w.copy()"),
    (5, [185, 186, 187], "min_global_erros = float"),
    (6, [189], "iteration = 0"),
    (7, [194], "self._rng = np.random.RandomState(self.random_state)"),
    (8, [201], "while iteration < self.max_iterations:"),
    (9, [202], "iteration += 1"),
    (10, [208], "final_violations = -1"),
    (11, [213], "for epoch in range(self.max_epochs):"),
    (12, [214], "violations = 0"),
    (13, [217], "indices = self._rng.permutation(m)"),
    (14, [219, 220], "for i in indices:"),
    (15, [228], "l = yi"),
    (16, [223, 229, 230, 231], "k = np.argmin(distances_temp)"),
    (17, [234, 235, 236], "norm_w = np.linalg.norm(w)"),
    (18, [245, 246, 247, 248, 249, 251], "if margin < gamma * norm_w:"),
    (19, [253], "violations += 1"),
    (20, [261], "g = (xi - centroids[l]) ** 2 - (xi - centroids[k]) ** 2"),
    (21, [272], "factor = 1 - self.eta * gamma / norm_w"),
    (22, [275], "factor = np.clip(factor, 0.0, 1.0)"),
    (23, [277], "w = w * factor - self.eta * g"),
    (24, [280], "w = np.maximum(0, w)"),
    (25, [285], "alpha = alpha * factor"),
    (26, [286], "alpha[i] = alpha[i] + self.eta"),
    (27, [287], "alpha = np.clip(alpha, 0, 1e6)"),
    (30, [289], "final_violations = violations"),
    (31, [297], "norm_fim = np.linalg.norm(w)"),
    (32, [298, 299, 300], "w = w / norm_fim"),
    (33, [304, 305, 306, 307, 308], "erros = int(sum("),
    (34, [309], "if erros < min_global_erros:"),
    (35, [310, 311], "w_best_effort = w.copy()"),
    (37, [314, 317], "if violations == 0:"),
    (41, [320, 331], "if viable_now:"),
    (42, [334, 335, 336, 337], "found_feasible_solution = True"),
    (43, [343], "if np.isinf(gamma_hi):"),
    (44, [344], "gamma = gamma + self.delta_gamma"),
    (46, [346], "gamma = min(gamma + self.delta_gamma, (gamma_lo + gamma_hi) / 2)"),
    (49, [349], "gamma_hi = gamma"),
    (50, [353, 354, 355], "gamma = (gamma_lo + gamma_hi) / 2"),
    (52, [358, 359], "if abs(gamma_hi - gamma_lo) <= self.tol:"),
    (56, [378, 380, 381], "self.gamma_ = gamma_lo"),
    (57, [386, 394, 395], "self.gamma_ = 0.0"),
    (58, [398, 405, 406], "self.w_ = w_star"),
    (60, [382, 396, 407, 410, 424, 425, 426], "_registrar_margem_minima"),
    (61, [383, 397, 408], "return w_"),
]

PARAMETROS = [
    # símbolo, nome no código, valor, significado
    ("η", "eta", "0,001", "Taxa de aprendizado, como no CILAMCE 2017"),
    ("C", "C", "1", "Constante de penalização; λ = 1/C = 1"),
    ("γ₀", "gamma_init", "0,1", "Primeira margem testada"),
    ("δ", "delta_gamma", "0,05", "Incremento de γ enquanto é viável"),
    ("T", "max_epochs", "50", "Máximo de épocas para cada valor de γ"),
    ("P", "max_iterations", "50", "Máximo de valores de γ testados"),
    ("tol", "tol", "0,0001", "Tolerância da busca binária em γ"),
    ("s", "random_state", "42", "Semente da ordem aleatória dos pontos"),
    ("melhor_esforço", "use_best_effort", "V", "O que retornar quando nenhum γ é viável"),
]

OBSERVACOES = [
    "As linhas 15 a 27 são as Eq. 28 e 29 do CILAMCE 2017. A restrição da linha 18 tem o termo λ.α_i, que faz a "
    "suavização da margem, e o vetor g da linha 20 é a derivada de d_w(v(i),c(l)) – d_w(v(i),c(k)) em relação a w. "
    "O fator f da linha 22 é truncado em [0, 1] (até 14/10/2026 o limite superior era 2; como η.γ/N > 0, f < 1 "
    "sempre, e o truncamento só protege de valores negativos).",
    "As linhas 31 e 32 (15/10/2026) dividem w e α por ||w||₂ ao fim de cada época. A restrição da linha 18 é "
    "homogênea em w, salvo a folga: sem a normalização, o fator f encolhia w a cada correção até ||w|| ≈ 10⁻³, a "
    "exigência γ.N ficava desprezível e qualquer α_i > 0 satisfazia a linha 18 mesmo com o ponto do lado errado "
    "da fronteira (um \"γ viável\" degenerado, encontrado em 13 das 495 estimativas de A/D da execução de 15/10/2026 "
    "de manhã). Só a direção de w define o centróide mais próximo, então a normalização não muda o critério; α é "
    "dividido pelo mesmo N para a folga λ.α_i acompanhar a escala de w.",
    "As linhas 33 a 36 (15/10/2026) escolhem o melhor esforço pela época com MENOS pontos mal classificados "
    "(margem bruta negativa, sem a folga α), e não pela época com menos violações relaxadas: a contagem de "
    "violações premiava as épocas em que w já tinha evaporado. w_melhor é o w ao fim dessa época, entre todas as "
    "épocas de todas as tentativas de γ, e não o w da última época.",
    "As linhas 41 a 51 seguem a segunda forma da Eq. 31 (γ ← γ + δ) e a pesquisa binária descrita logo depois dela "
    "no CILAMCE 2017. O código não calcula a margem de parada da Eq. 30 e começa em γ₀ = 0,1, enquanto o artigo "
    "começa em zero. O teste de viabilidade continua sendo o da linha 18 (margem relaxada com α), como no texto.",
    "Quando γ é viável, a tentativa seguinte continua do w e do α atuais, como sugere o CILAMCE 2017. Quando γ é "
    "inviável, w e α voltam a w* e α*, a última solução viável, ou a [1,…,1] e 0 se ainda não houve nenhuma.",
    "O gerador aleatório é iniciado uma vez, na linha 7, e não é reiniciado entre as tentativas de γ. Com a mesma "
    "entrada, o resultado é sempre o mesmo.",
    "A linha 60 (15/10/2026) calcula a margem bruta mínima μ do w devolvido: a menor diferença, entre todos os "
    "pontos, da distância ao centróide rival menos a distância ao centróide correto, dividida por ||w||₂, sem a "
    "folga α. μ < 0 significa que algum ponto fica do lado errado da fronteira sob o w devolvido, qualquer que "
    "seja o γ. O runner grava μ na coluna margem_minima dos CSVs (bloco1_abc_passos_2_a_4, bloco1_ad_passos_2_a_5, "
    "bloco3_g_passos_2_a_5) e as tabelas marcam † quando γ = 0 ou μ < 0. No código, μ é calculado por "
    "_registrar_margem_minima imediatamente antes de cada um dos três retornos (linhas 382, 396 e 407) e fica em "
    "margem_minima_ e no global ULTIMA_MARGEM_MINIMA.",
    "Em todas as chamadas do código melhor_esforço = V, então a linha 58 nunca é usada nos experimentos; nesse caso "
    "w* ainda seria [1,…,1].",
    "As linhas 17, 22 e 27 são proteções numéricas: ||w||₂ nunca menor que 10⁻⁹, f limitado ao intervalo [0, 1] e "
    "cada α_q limitado ao intervalo [0, 10⁶]. N é calculado antes da correção e é o mesmo valor usado no teste "
    "(linha 18) e no fator f (linha 21).",
    "A linha 10 só faria diferença se T = 0.",
    "No código, depois de γ_sup ← γ, existe o teste \"se γ_sup = ∞, sair do laço\", que nunca é verdadeiro porque γ é "
    "sempre finito. Por isso ele não aparece no pseudocódigo.",
    "O código também emite avisos no log, sem efeito no retorno: quando nenhum γ é viável (com o número de pontos "
    "mal classificados do w_melhor) e quando a busca para por P sem atingir tol. Também guarda o histórico das "
    "tentativas de γ, só para diagnóstico.",
    "A página HTML que acompanha este documento mostra este mesmo pseudocódigo e executa uma tradução do código para "
    "JavaScript, que reproduz o Python bit a bit nos conjuntos rotulados pelo LLM em zero-shot da execução completa "
    "mais recente (lineares: Problemas A, B, C e E; não lineares: meia-lua e peso × altura). Nos conjuntos em que o "
    "trabalho estima a métrica, o w calculado é igual ao gravado nos CSVs da execução.",
]


def linhas_py():
    with open(ARQUIVO_PY, encoding="utf-8") as f:
        return f.read().split("\n")


def tabela_correspondencia():
    """[(nº pseudocódigo, 'linhas do .py', código)] com o código lido do arquivo atual."""
    src = linhas_py()
    out = []
    for n, ls, _ in CORRESPONDENCIA:
        if len(ls) == 1:
            txt = str(ls[0])
        elif ls == list(range(ls[0], ls[-1] + 1)):
            txt = f"{ls[0]} a {ls[-1]}" if len(ls) > 2 else f"{ls[0]} e {ls[1]}"
        else:
            txt = ", ".join(str(l) for l in ls)
        codigo = "\n".join(src[l - 1].strip() for l in ls)
        out.append((n, txt, codigo))
    return out
