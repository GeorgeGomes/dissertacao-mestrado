"""Reproduz o capítulo "V - Resultados" revisado pelo orientador (PDF da reunião de 02/10/2026) num
.docx, mantendo o texto dele, as partes em VERMELHO em vermelho, e inserindo logo abaixo de cada
item vermelho as imagens das tabelas e figuras correspondentes da execução completa mais recente.
No fim, dois anexos: (A) as quatro variantes de prompt (pedido em vermelho) e (B) a comparação das
tabelas novas com as do artigo de 19/07/2025 (`artigo_mudancas_2025_07_19.pdf`), de onde vieram
os números do texto dele.

Uso (raiz do projeto, .venv ativo):
    python src/gerar_resultados_com_tabelas.py [pasta_execucao_completa]
Saída: <pasta_execucao>/entrega_orientador_<data>/Resultados_com_tabelas_e_figuras.docx
"""
import glob
import sys
from datetime import date
from pathlib import Path

import pymupdf
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
PDF = RAIZ / "reunioes_orientador" / "reuniao_2026_10_02" / "doc revisados orientador" / "Resultados.pdf"
VERMELHO = 0xFF0000
AZUL = RGBColor(0x1F, 0x4E, 0x9A)
LARG = Cm(17.0)

T, F, G = "tabela", "figura", "figura_gray"


def tab(nome):
    return (T, nome)


def fig(nome):
    return (F, nome)


def fig_pb(nome):
    return (G, nome)


# chave: começo do texto do bloco (ou trecho em vermelho contido nele) → o que inserir abaixo
INSERCOES = [
    ("Tabela 2: Fidelidade da métrica no Problema A", [tab("bloco1_table_01_A_passos_2_a_5"), tab("bloco1_table_05_A_passo4_perceptron_vs_nnls")],
     "Nova Tabela 2 (execução de 15/10/2026): zero-shot, few-shot com k ∈ {4, 10, 20, 40} exemplos do rótulo real e teto (W ajustado ao rótulo real), Perceptron e NNLS; a segunda tabela traz o cosseno entre os dois estimadores."),
    ("Falta fazer experimentos com few-shot", [tab("bloco1_table_02_A_passo2_vs_passo3")],
     "Feito: zero-shot versus cada k, lado a lado."),
    ("Seria interessante plotar as amostras", [fig_pb("bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42"), fig_pb("bloco1_graphic_11_B_passos_2_3_simbolos_seed42"),
                                               fig_pb("bloco1_graphic_11_C_passos_2_3_simbolos_seed42"), fig_pb("bloco1_graphic_13_D_passos_2_a_4_simbolos_seed42")],
     "Feito (semente 42; as outras sementes estão na pasta figuras/): rótulo real, LLM zero-shot, métrica e LLM com exemplos, símbolos + e −. Também para B, C e a meia-lua (D)."),
    ("Figura 1: Bloco 1", [fig("bloco1_graphic_06_A_passo4_perceptron_vs_nnls"), fig("bloco1_graphic_10_BC_passo3_consistencia")],
     "Nova Figura 1: Perceptron × NNLS (fidelidade no A, consistência em B e C, direção de Ŵ) e consistência, κ e F1-Macro em B e C."),
    ("Tabela 3: Teste-oráculo", [tab("bloco1_table_17_oraculo_passos_1_a_3")],
     "Nova Tabela 3: Problemas 1, 2 e 3 (métricas [1; 1], [0,3; 1,5] e [1,5; 0,3]), sem a primeira linha antiga."),
    ("Qual a diferença entre problema A e B", [],
     "Resposta: na versão nova não há A nem B no oráculo. As três métricas rotulam o MESMO conjunto, as gaussianas do Problema E (centróides (−1,5; 1) e (1,5; −1), 150 pontos), pelo centróide mais próximo; o Problema A não serve porque, com os centróides em x2 = 0, o peso w2 não era identificável."),
    ("Figura 2: Validação", [fig("bloco1_graphic_19_oraculo_passos_1_a_3_w_scatter"), fig("bloco1_graphic_19_oraculo_passos_1_a_3_ratio"),
                             fig("bloco1_graphic_19_oraculo_passos_1_a_3_cosseno"), fig("bloco1_graphic_19_oraculo_passos_1_a_3_acuracia")],
     "Nova Figura 2, em quatro painéis separados (W recuperado, razão w1/w2, cosseno, acurácia; 3 sementes)."),
    ("Tabela 4: Transferência", [tab("bloco1_table_08_BC_passo3_zero_vs_eleita"), tab("bloco1_table_09_BC_passo4_exemplos_da_metrica"), tab("bloco1_table_07_BC_passo3_por_semente")],
     "Nova Tabela 4: zero-shot (métrica zero-shot e métrica eleita do A), few-shot com k ∈ {4, 10, 20, 40} e valores por semente. Sem IC bootstrap (retirado a pedido de 13/10/2026): média e faixa mín–máx entre sementes."),
    ("Perceptron Estruturado 100% e NNLS 95,3%", [tab("bloco1_table_18_D_passo5_teto"), fig("bloco1_graphic_21_D_passo5_teto")],
     "Correção: não há 100 %. O teto da métrica diagonal ajustada ao rótulo real da meia-lua é 86,9 % (R²), 87,1 % (R³) e 87,3 % (R⁴) com o Perceptron e 86,4 a 86,7 % com o NNLS; o valor de V.7 (86,7 %) é o correto."),
    ("Tabela 5: Fidelidade média", [tab("bloco1_table_04_A_passo4_R2_R3_R4"), fig("bloco1_graphic_08_A_passo4_R2_R3_R4")],
     "Nova Tabela 5: só o Problema A (B e C são lineares e recebem apenas o vetor w do A; R³/R⁴ não se aplica a eles)."),
    ("Falta resultados e tabela relativa ao problema D", [tab("bloco1_table_10_D_passos_2_a_5"), tab("bloco1_table_11_D_passo2_vs_passo3"), tab("bloco1_table_12_AD_por_passo"),
                                                           fig("bloco1_graphic_04_AD_passos_2_a_5_curva"), fig("bloco1_graphic_14_D_passo6_svm_seed42")],
     "Feito: Problema D com os mesmos passos do A (zero-shot, few-shot do rótulo real, teto) em R², R³ e R⁴, comparação A × D, curva por k e SVM RBF × LLM. O gpt-4o-mini colapsa em zero-shot na meia-lua nas 3 sementes; com exemplos deixa de colapsar."),
    ("Tabela 6 - Efeito da nomenclatura", [tab("bloco1_table_14_vies_nomes_classe"), tab("bloco1_table_15_vies_nomes_atributo"), tab("bloco1_table_16_vies_variantes_prompt"),
                                           fig("bloco1_graphic_16_vies_nomes_classe"), fig("bloco1_graphic_15_vies_ordem_classes"), fig("bloco1_graphic_17_vies_nomes_atributo")],
     "Nova Tabela 6 em três partes (nomes de classe, nomes de atributo, variantes de prompt), zero-shot e 10-shot, 3 sementes; nomes de atributo corrigidos (altura/peso é a variante semântica do A)."),
    ("Figura 3 -  Fidelidade e consistência", [fig("bloco1_graphic_18_vies_variantes_prompt")],
     "Nova Figura 3."),
    ("“precisa explicar como é feita cada variante”", [],
     "As quatro variantes estão descritas no Anexo A, com o texto real de cada enunciado."),
    ("estranho o valor se repetir", [tab("bloco2_table_04_E_passo2_vs_passo3")],
     "Explicação: em zero-shot não há exemplos, logo não há estratégia de seleção; é UMA coleta por (modelo, semente), repetida nas quatro linhas. Valores atuais (E1): 61,3 % (gpt-4o-mini) e 54,4 % (gemini-2.5-flash-lite)."),
    ("Figura 4: Bloco 2", [tab("bloco2_table_01_E1_passo3_estrategia_x_k"), tab("bloco2_table_05_E_passo3_H4"), fig("bloco2_graphic_03_E1_passo3_curva"), fig("bloco2_graphic_04_E1_passo3_estrategias")],
     "Novas Tabela 7 (E1 = [0,3; 1,5]; E2 e E3 na pasta tabelas/) e Figura 4. H4 refutada: E1, média dos k > 0, gpt-4o-mini aleatória 89,9 > fácil 88,1 > mista 86,7 > difícil 77,7 %; Gemini fácil 93,2 > aleatória 92,9 > mista 91,1 > difícil 72,4 %."),
    ("Tabela 8 - Bloco 2", [tab("bloco2_table_06_E_passo6_baselines"), fig("bloco2_graphic_05_E1_passo6_baselines")],
     "Nova Tabela 8: classificadores treinados nos MESMOS exemplos mostrados ao LLM (estratégia mista, 3 sorteios); E1, E2 e E3."),
    ("SVM RBF treinado no ground truth", [fig("bloco1_graphic_14_D_passo6_svm_seed42"), tab("bloco1_table_19_D_passo6_svm"), tab("bloco2_table_10_F_passo3_k")],
     "Mudou: nesta execução o gpt-4o-mini colapsa em zero-shot nas 3 sementes da meia-lua (não só em 123 e 7); o painel dele registra o colapso e a figura com fronteira é a do Gemini. A tabela traz o Problema F (divisão 70/30) com k ∈ {0, 4, 10, 20, 40}: os números de V.7 (53,3 % → 68 %) precisam ser substituídos. A tabela nova (bloco1_table_19) traz os números da figura: acurácia do SVM RBF treinado no rótulo real, fidelidade e acurácia do SVM treinado nos rótulos do LLM e a métrica diagonal, por semente."),
    ("Figura 7 - Esquerda", [fig("bloco2_graphic_06_E1_passo4_diluicao"), fig("bloco2_graphic_07_E1_passo5_ordem"), tab("bloco2_table_07_E1_passo4_diluicao"), tab("bloco2_table_08_E1_passo5_ordem")],
     "Nova Figura 7 e tabelas: são QUATRO exemplos difíceis fixos (não três) mais 0, 2, 4, 10, 16 e 20 fáceis; ordenações classe 0 primeiro, classe 1 primeiro, embaralhada e alternada, k ∈ {4, 10, 20}."),
]

MODELOS = [("gpt4mini", "gpt-4o-mini"), ("flashlite", "gemini-2.5-flash-lite")]

# ---- Anexo B: tabelas do artigo de 19/07/2025 (valores transcritos do PDF) e o que mudou ----
ANTIGAS = [
    ("Tabela 2 (artigo): Fidelidade da estimação no Problema A, por semente (classes A/B, zero-shot).",
     ["Semente", "Fid. Perceptron (%)", "Fid. NNLS (%)"],
     [["42", "82,7", "83,3"], ["123", "86,7", "86,7"], ["7", "82,7", "87,3"], ["Média", "84,0", "85,8"]],
     "bloco1_table_01_A_passos_2_a_5",
     ["Estrutura: antes uma linha por semente, só zero-shot; agora uma linha por regime (zero-shot, 4, 10, 20, 40 exemplos do rótulo real e teto) com média e faixa entre sementes; as sementes separadas estão na tabela 13 (estabilidade de Ŵ).",
      "Números (gpt-4o-mini, zero-shot): fidelidade Perceptron 84,0 → 86,7 % (85,3 a 88,0); NNLS 85,8 → 85,3 %. Com exemplos a fidelidade sobe para 92 a 97 %.",
      "Motivos das diferenças: o Problema A mudou de centróides em 15/10/2026 ((−2; −0,5) e (2; 0,5) em vez de (−2; 0) e (2; 0)) e o Perceptron ganhou a normalização por época e o melhor esforço por pontos mal classificados; os rótulos do LLM são de uma coleta nova."]),
    ("Tabela 3 (artigo): Teste-oráculo: recuperação de W conhecido.",
     ["Config.", "W verd.", "Perceptron Fid. (%)", "cos", "NNLS Fid. (%)", "cos"],
     [["Prob. A (euclid.)", "[1,0; 1,0]", "98,0", "0,709", "98,0", "0,707"], ["Prob. B (euclid.)", "[1,0; 1,0]", "98,0", "0,9999", "98,0", "0,971"],
      ["Expert aniso. x2", "[0,3; 1,5]", "100,0", "0,9999", "95,3", "0,993"], ["Expert aniso. x1", "[1,5; 0,3]", "100,0", "0,998", "99,3", "0,989"]],
     "bloco1_table_17_oraculo_passos_1_a_3",
     ["Estrutura: a primeira linha (Prob. A euclidiano, cosseno 0,709, o \"cosseno esquisito\") foi retirada; as três restantes viraram Problemas 1, 2 e 3, todas sobre o mesmo conjunto (gaussianas do E, 150 pontos), com 3 sementes e a coluna chamada Acurácia (vs rótulo do oráculo), não Fidelidade.",
      "Números: Perceptron 100 % nos três problemas (cosseno 0,998 a 1,000); NNLS 99,8 / 96,7 / 98,2 % (antes 98,0 / 95,3 / 99,3)."]),
    ("Tabela 4 (artigo): Transferência: consistência e κ (Perceptron), IC bootstrap 95%.",
     ["Problema", "n-shot", "Consistência [IC95]", "κ [IC95]"],
     [["B", "0", "0,697 [0,670; 0,750]", "0,333 [0,255; 0,449]"], ["B", "4", "0,837 [0,719; 0,927]", "0,670 [0,432; 0,853]"],
      ["C", "0", "0,880 [0,860; 0,900]", "0,746 [0,703; 0,789]"], ["C", "4", "0,962 [0,948; 0,990]", "0,923 [0,895; 0,979]"]],
     "bloco1_table_09_BC_passo4_exemplos_da_metrica",
     ["Estrutura: sem IC bootstrap; faixa mín–máx entre sementes; k ∈ {0, 4, 10, 20, 40}; a tabela 08 compara a métrica zero-shot com a métrica eleita do A e a 07 mostra cada semente.",
      "Números (gpt-4o-mini): B zero-shot consistência 0,697 → 0,690 e κ 0,333 → 0,320; B com 4 exemplos 0,837 → 0,868 e κ 0,670 → 0,733; C zero-shot 0,880 → 0,870 e κ 0,746 → 0,725; C com 4 exemplos 0,962 → 0,948 e κ 0,923 → 0,896. Com 10 ou mais exemplos, κ em B chega a 0,85 a 0,87.",
      "A consistência com a métrica euclidiana, citada no texto, foi retirada do sistema em 09/10/2026 (não está na seção IV.3)."]),
    ("Tabela 5 (artigo): Fidelidade média (Perceptron) ao aumentar os atributos nos problemas lineares A/B/C, por semente.",
     ["n atributos", "Seed 42", "Seed 123", "Seed 7", "Média"],
     [["3 (R³, +x1x2)", "83,3", "88,7", "82,7", "84,9"], ["4 (R⁴, +x1², x2²)", "81,3", "86,7", "90,7", "86,2"]],
     "bloco1_table_04_A_passo4_R2_R3_R4",
     ["Estrutura: só o Problema A (B e C não recebem R³/R⁴); linhas por espaço (R², R³, R⁴) e regime (zero-shot, cada k, teto), Perceptron e NNLS, com a coluna Δ fid. vs R².",
      "Números (gpt-4o-mini, zero-shot): R² 86,7 %, R³ 88,7 % (antes 84,9), R⁴ 88,9 % (antes 86,2): ganho de 2 pontos, marginal como no texto."]),
    ("Tabela 6 (artigo): Efeito de rotulagem e prompt sobre a fidelidade (seed 42; classe: 10-shot; atributo/prompt: zero-shot).",
     ["Nome de classe", "Fid. (%)", "Nome de atributo", "Fid. (%)", "Prompt", "Fid. (%)"],
     [["Positivo/Negativo", "99,3", "altura/peso", "84,7", "default", "82,7"], ["Negativo/Positivo", "88,7", "feature_1/2", "84,0", "cot", "76,7"],
      ["Azul/Vermelho", "86,7", "x1/x2", "82,7", "geometric", "68,0"], ["1/0", "83,3", "", "", "tabular", "64,7"], ["A/B", "82,7", "", "", "", ""], ["B/A", "68,0", "", "", "", ""]],
     "bloco1_table_14_vies_nomes_classe",
     ["Estrutura: três tabelas (14 nomes de classe, 15 nomes de atributo, 16 variantes de prompt), 3 sementes, zero-shot e 10-shot, com a acurácia invariante à permutação e a consistência em B e C.",
      "Números (gpt-4o-mini, zero-shot, 3 sementes): Positivo/Negativo 99,3 → 99,3 %; A/B 82,7 → 86,7 %; B/A 68,0 → 70,0 %; Azul/Vermelho 86,7 → 92,2 %; Negativo/Positivo 88,7 → 97,1 %; 0/1 colapsou em 1 das 3 sementes (66,0 % nas outras duas). Atributos: altura/peso 84,7 → 90,2 %; feature_1/2 84,0 → 81,6 %. Prompt: cot 76,7 → 78,2 %; geométrica 68,0 → 79,6 %; tabular 64,7 → 72,2 %."]),
    ("Tabela 7 (artigo): Bloco 2: acurácia LLM × especialista por estratégia e n-shot (aniso. x2).",
     ["n-shot", "Easy", "Hard", "Mixed", "Random"],
     [["0", "61,1", "61,1", "61,1", "61,1"], ["4", "85,4", "62,1", "82,2", "84,3"], ["10", "86,7", "67,9", "79,3", "88,3"], ["20", "90,3", "83,8", "86,2", "92,1"], ["40", "87,6", "97,9", "98,5", "94,1"]],
     "bloco2_table_01_E1_passo3_estrategia_x_k",
     ["Estrutura: \"aniso. x2\" virou E1; nomes em português (Fácil, Difícil, Mista, Aleatória); κ ao lado de cada acurácia; faixa entre sementes; o zero-shot aparece uma vez (é uma coleta só).",
      "Números (gpt-4o-mini, E1): zero-shot 61,1 → 61,3 %; 4 exemplos fácil 85,4 → 84,7, difícil 62,1 → 62,6, mista 82,2 → 81,5, aleatória 84,3 → 85,0; 40 exemplos fácil 87,6 → 89,1, difícil 97,9 → 97,6, mista 98,5 → 97,6, aleatória 94,1 → 94,3. O quadro de H4 não muda."]),
    ("Tabela 8 (artigo): Bloco 2: LLM vs. baselines clássicos (40-shot).",
     ["Especialista", "LLM (hard)", "k-NN", "Reg. Log.", "SVM-RBF"],
     [["aniso. x2", "97,9 % (κ 0,957)", "94,5 % (κ 0,891)", "97,3 % (κ 0,947)", "94,5 % (κ 0,891)"], ["aniso. x1", "92,4 % (κ 0,850)", "95,6 % (κ 0,912)", "98,0 % (κ 0,961)", "95,5 % (κ 0,911)"]],
     "bloco2_table_06_E_passo6_baselines",
     ["Estrutura: os classificadores são treinados nos MESMOS exemplos mostrados ao LLM (estratégia mista, 3 sorteios); colunas LLM (mista), LLM (melhor estratégia), Regressão Logística, SVM e k-NN; três especialistas E1, E2, E3.",
      "Números (gpt-4o-mini, 40 exemplos): E1 LLM 97,6 % (mista e difícil) contra Regressão Logística 100,0 %, SVM 95,2 %, k-NN 91,2 %; E2 LLM 93,9 % (mista) e 96,0 % (aleatória) contra RL 100,0 %; E3 LLM 93,3 / 97,3 % contra RL 100,0 %. A conclusão do texto (o LLM não supera a Regressão Logística) fica mais forte: a RL acerta 100 % nos três."]),
]


def _rtexto(p, texto, vermelho=False, bold=False, italic=False, size=12, cor=None, mono=False):
    r = p.add_run(texto)
    r.font.size = Pt(size); r.bold = bold; r.italic = italic
    if vermelho:
        r.font.color.rgb = RGBColor(0xFF, 0, 0)
    elif cor is not None:
        r.font.color.rgb = cor
    if mono:
        r.font.name = "Consolas"
    return r


def _nota(doc, texto):
    p = doc.add_paragraph()
    _rtexto(p, "Nota (George, " + date.today().strftime("%d/%m/%Y") + "): " + texto, size=10, italic=True, cor=AZUL)
    p.paragraph_format.space_after = Pt(6)


def _legenda(doc, texto):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _rtexto(p, texto, size=9, italic=True); p.paragraph_format.space_after = Pt(8)


def _imagem(doc, caminho: Path, legenda: str):
    doc.add_picture(str(caminho), width=LARG)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    _legenda(doc, legenda)


def _inserir(doc, execucao: Path, itens):
    for tipo, nome in itens:
        if tipo == T:
            for alias, mnome in MODELOS + [(None, "")]:
                f = execucao / (f"{nome}__{alias}_gray.png" if alias else f"{nome}_gray.png")
                if f.exists():
                    _imagem(doc, f, f"{nome}" + (f" ({mnome})" if alias else "") + ". Tabela editável em tabelas/" + f.name.replace("_gray.png", ".tex") + ".")
                if alias is None:
                    break
        else:
            cor = "gray" if tipo == G else "color"
            for alias, mnome in MODELOS + [(None, "")]:
                f = execucao / (f"{nome}__{alias}_{cor}.png" if alias else f"{nome}_{cor}.png")
                if f.exists():
                    _imagem(doc, f, f"{nome}" + (f" ({mnome})" if alias else "") + ". Arquivo em figuras/.")
                if alias is None:
                    break


def _tabela_antiga(doc, cab, linhas):
    t = doc.add_table(rows=1, cols=len(cab)); t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, cab):
        c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9)
    for linha in linhas:
        cells = t.add_row().cells
        for c, v in zip(cells, linha):
            c.text = ""; r = c.paragraphs[0].add_run(v); r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _variantes():
    import dissertacao_mestrado as dm
    descr = {
        "default": "Padrão: enunciado direto, o ponto com seus atributos e a pergunta pela classe.",
        "geometric": "Geométrico: o enunciado descreve o ponto como posição num plano cartesiano e pede a classe pela proximidade espacial.",
        "cot": "Chain-of-thought (cot): pede ao modelo raciocinar passo a passo antes do rótulo; o rótulo é lido do fim da resposta.",
        "tabular": "Tabular: os atributos do ponto (e dos exemplos, quando há) são apresentados numa tabela em markdown.",
    }
    for v in ("default", "geometric", "cot", "tabular"):
        yield (descr[v], dm.PROMPT_VARIANTS.get(v, {}).get("system_message", ""),
               dm._montar_prompt(1.25, -0.5, "A", "B", None, "x1", "x2", None, v),
               dm._montar_prompt(1.25, -0.5, "A", "B", [(-2.1, 0.3, "A"), (1.9, -0.2, "B")], "x1", "x2", None, v))


def main():
    execucao = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(str(RAIZ / "resultados" / "execucao_*_completa")))[-1])
    out = execucao / f"entrega_orientador_{date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = sec.bottom_margin = Cm(2.0)
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(12)

    p = doc.add_paragraph()
    _rtexto(p, "Capítulo V (Resultados) revisado pelo orientador em 02/10/2026, com as tabelas e figuras novas abaixo de cada item em vermelho. "
            f"Texto: o do PDF da reunião, sem alteração. Imagens: execução completa de 15/10/2026 à noite ({execucao.name}). "
            "Notas em azul são de George. Anexo A: as quatro variantes de prompt. Anexo B: comparação com as tabelas do artigo de 19/07/2025.",
         size=9.5, italic=True, cor=AZUL)

    pdf = pymupdf.open(PDF)
    usados = set()
    for page in pdf:
        for b in page.get_text("dict")["blocks"]:
            if not b.get("lines"):
                continue
            spans = [s for l in b["lines"] for s in l["spans"]]
            texto = "".join(s["text"] for s in spans)
            if not texto.strip():
                continue
            par = doc.add_paragraph()
            par.paragraph_format.space_after = Pt(6)
            for s in spans:
                fonte = s["font"]
                _rtexto(par, s["text"], vermelho=(s["color"] == VERMELHO), bold="Bold" in fonte, italic="Italic" in fonte,
                     size=(8 if s["size"] < 10 else 12))
            # inserções: pelo começo do bloco ou por trecho vermelho contido nele
            for chave, itens, nota in INSERCOES:
                if chave in usados:
                    continue
                if texto.strip().startswith(chave) or (chave in texto and any(s["color"] == VERMELHO and chave in s["text"] for s in spans)) \
                        or (chave in texto and chave.startswith(("Perceptron Estruturado 100%", "estranho o valor"))):
                    usados.add(chave)
                    if nota:
                        _nota(doc, nota)
                    _inserir(doc, execucao, itens)
    faltou = [c for c, _, _ in INSERCOES if c not in usados]
    if faltou:
        print("  ⚠️ chaves não encontradas no PDF:", faltou)

    # Anexo A
    doc.add_page_break()
    p = doc.add_paragraph(); _rtexto(p, "Anexo A: as quatro variantes de prompt", bold=True, size=13)
    p = doc.add_paragraph(); _rtexto(p, "Todas recebem o mesmo ponto e os mesmos exemplos; muda só a forma do enunciado. Exemplo com o ponto (1,25; −0,5), classes A e B e, no few-shot, dois exemplos.", size=10)
    for d, sysmsg, zero, few in _variantes():
        p = doc.add_paragraph(); _rtexto(p, d, bold=True, size=11)
        if sysmsg:
            p = doc.add_paragraph(); _rtexto(p, "Mensagem de sistema: " + sysmsg, size=9, italic=True)
        for rot, txt in (("Zero-shot:", zero), ("Few-shot (2 exemplos):", few)):
            p = doc.add_paragraph(); _rtexto(p, rot, bold=True, size=9)
            p = doc.add_paragraph(); _rtexto(p, txt, size=8.5, mono=True); p.paragraph_format.space_after = Pt(6)

    # Anexo B
    doc.add_page_break()
    p = doc.add_paragraph(); _rtexto(p, "Anexo B: comparação das tabelas novas com as do artigo de 19/07/2025", bold=True, size=13)
    p = doc.add_paragraph()
    _rtexto(p, "Os números do capítulo V revisado vêm do artigo de 19/07/2025 (execução de 05/07/2026). Para cada tabela: a versão antiga transcrita do artigo, "
            "a versão nova (imagem; .tex editável em tabelas/) e o que mudou. Duas mudanças valem para todas: o Problema A passou a ter centróides "
            "(−2; −0,5) e (2; 0,5) em 15/10/2026 e o Perceptron ganhou três ajustes no mesmo dia (normalização de w e α por época, melhor esforço por pontos "
            "mal classificados, margem bruta mínima); os rótulos do LLM são de uma coleta nova (222.648 chamadas).", size=10)
    for cap, cab, linhas, nova, mudou in ANTIGAS:
        p = doc.add_paragraph(); _rtexto(p, cap, bold=True, size=11)
        _tabela_antiga(doc, cab, linhas)
        f = execucao / f"{nova}__gpt4mini_gray.png"
        if not f.exists():
            f = execucao / f"{nova}_gray.png"
        if f.exists():
            _imagem(doc, f, f"Versão nova: {f.name.replace('_gray.png', '')}" + (" (gpt-4o-mini; a do Gemini está em tabelas/)" if "gpt4mini" in f.name else "") + ".")
        for m in mudou:
            p = doc.add_paragraph(); _rtexto(p, "• " + m, size=10); p.paragraph_format.space_after = Pt(3)

    destino = out / "Resultados_com_tabelas_e_figuras.docx"
    doc.save(destino)
    print(f"Documento: {destino} ({destino.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
