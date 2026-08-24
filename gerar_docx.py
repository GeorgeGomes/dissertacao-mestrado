"""Gera artigo.docx bem formatado para o orientador, a partir de artigo.tex.

Pipeline:
  1. Renderiza o Algoritmo 1 (algorithm2e) como PNG de alta resolução via LaTeX
     standalone + pdfcrop (o pandoc não entende algorithm2e).
  2. Cria artigo_pandoc.tex:
       - tabelas e figuras são "desembrulhadas" e suas legendas viram
         PARÁGRAFOS DE TEXTO visíveis ("Tabela N: ..." acima da tabela;
         "Figura N: ..." abaixo da imagem), com a numeração real do PDF
         (lida do artigo.aux);
       - todas as \\ref/\\eqref são resolvidas para os números do PDF;
       - o ambiente algorithm vira imagem + legenda textual.
  3. Roda pandoc -> artigo.docx.
  4. Pós-processa o pacote direto no ZIP (preservando _rels/.rels e afins):
       - grade + largura 100% nas tabelas de dados (ordem correta do esquema
         OOXML dentro de <w:tblPr>);
       - fonte do tema trocada para "Latin Modern Roman" (a fonte do LaTeX;
         precisa estar instalada na máquina para o Word exibi-la).

Uso:  .venv/bin/python gerar_docx.py
"""
import os
import re
import subprocess
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(tempfile.gettempdir(), "artigo_docx_build")
os.makedirs(SCRATCH, exist_ok=True)
EXEC_DIR = "execucao_2026-07-05_12-34-52_completa"
os.chdir(ROOT)

art = open("artigo.tex", encoding="utf8").read()

# rótulo -> número, do artigo.aux (numeração idêntica à do PDF)
aux = open("artigo.aux", encoding="utf8").read()
labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))


def read_braced(s, i):
    """Conteúdo de um grupo {...} iniciando em s[i]=='{' (chaves aninhadas)."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("chave não fechada")


# ---------------------------------------------------------------- 1. algoritmo -> PNG
m = re.search(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", art, re.S)
assert m, "ambiente algorithm nao encontrado"
alg_block = m.group(0)
alg_body = re.sub(r"\\caption\{.*?\}\s*", "", alg_block, flags=re.S)
alg_body = re.sub(r"\\label\{.*?\}\s*", "", alg_body)

standalone = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[portuguese]{babel}
\usepackage{amsmath,amssymb}
\usepackage[portuguese,ruled,linesnumbered]{algorithm2e}
\SetAlFnt{\small}
\pagestyle{empty}
\begin{document}
""" + alg_body + "\n\\end{document}\n"

alg_dir = os.path.join(SCRATCH, "alg_build")
os.makedirs(alg_dir, exist_ok=True)
open(os.path.join(alg_dir, "alg.tex"), "w", encoding="utf8").write(standalone)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "alg.tex"],
               cwd=alg_dir, check=True, capture_output=True)
subprocess.run(["pdfcrop", "--margins", "6", "alg.pdf", "alg_crop.pdf"],
               cwd=alg_dir, check=True, capture_output=True)
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile",
                "alg_crop.pdf", "algoritmo1"], cwd=alg_dir, check=True)
alg_png = os.path.join(alg_dir, "algoritmo1.png")
print("algoritmo renderizado:", alg_png)

# ------------------------------------------------------- 2. tex pré-processado
caption_txt = ("\\textbf{Algoritmo 1:} Perceptron Estruturado com Relaxação de "
               "Margem (variante diagonal).")
img_block = ("\\begin{center}\n"
             f"\\includegraphics[width=0.8\\textwidth]{{{alg_png}}}\n\n"
             f"{caption_txt}\n"
             "\\end{center}\n")
tex = art.replace(alg_block, img_block)
tex = tex.replace(r"Algoritmo~\ref{alg:perceptron}", "Algoritmo 1")


def caption_and_body(block):
    """Extrai (caption, label, bloco sem caption/label)."""
    cap = ""
    ci = block.find(r"\caption{")
    if ci >= 0:
        cap, cend = read_braced(block, ci + len(r"\caption"))
        block = block[:ci] + block[cend:]
    lab = re.search(r"\\label\{((?:tab|fig):[^}]+)\}", block)
    num = labels.get(lab.group(1), "?") if lab else "?"
    block = re.sub(r"\\label\{(?:tab|fig):[^}]+\}", "", block)
    return cap, num, block


def unwrap_tables(m):
    cap, num, block = caption_and_body(m.group(0))
    body = re.search(r"\\begin\{tabular\}.*\\end\{tabular\}", block, re.S).group(0)
    return (f"\n\\bigskip\n\n\\noindent \\textbf{{Tabela {num}:}} {cap}\n\n"
            f"{body}\n\n\\bigskip\n")


def unwrap_figures(m):
    cap, num, block = caption_and_body(m.group(0))
    inner = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?", "", block)
    inner = inner.replace(r"\end{figure}", "").strip()
    return (f"\n{inner}\n\n\\noindent \\textbf{{Figura {num}:}} {cap}\n\n"
            "\\bigskip\n")


tex = re.sub(r"\\begin\{table\}.*?\\end\{table\}", unwrap_tables, tex, flags=re.S)
tex = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", unwrap_figures, tex, flags=re.S)

# resolve TODAS as refs pelos números do PDF (o pandoc deixaria de numerar
# tabelas/figuras desembrulhadas)
for lab, num in labels.items():
    tex = tex.replace(f"\\eqref{{{lab}}}", f"({num})")
    tex = tex.replace(f"\\ref{{{lab}}}", num)

open("artigo_pandoc.tex", "w", encoding="utf8").write(tex)
print("artigo_pandoc.tex gerado (legendas como texto, refs resolvidas)")

# ------------------------------------------------------------------- 3. pandoc
subprocess.run(["pandoc", "artigo_pandoc.tex", "-o", "artigo.docx",
                f"--resource-path=.:{EXEC_DIR}:{alg_dir}"], check=True)
print("pandoc ok")

# --------------------------- 4. pós-processamento no ZIP (bordas + fonte LM)
with zipfile.ZipFile("artigo.docx") as zin:
    entries = [(info, zin.read(info.filename)) for info in zin.infolist()]

parts = {i.filename: d for i, d in entries}
xml = parts["word/document.xml"].decode("utf8")

BORDERS = ('<w:tblBorders>'
           '<w:top w:val="single" w:sz="4" w:space="0" w:color="666666" />'
           '<w:left w:val="single" w:sz="4" w:space="0" w:color="666666" />'
           '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="666666" />'
           '<w:right w:val="single" w:sz="4" w:space="0" w:color="666666" />'
           '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="666666" />'
           '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="666666" />'
           '</w:tblBorders>')
WIDTH = '<w:tblW w:type="pct" w:w="5000" />'

# Ordem fixa dos filhos de <w:tblPr> no esquema OOXML:
#   tblStyle, tblW, jc, tblBorders, ..., tblLook, tblCaption
n_tab = 0


def fix_tblpr(m):
    global n_tab
    block = m.group(0)
    if 'w:val="Table"' not in block:
        return block
    n_tab += 1
    block = re.sub(r"<w:tblW[^>]*/>", WIDTH, block)
    return block.replace("<w:tblLook", BORDERS + "<w:tblLook")


xml = re.sub(r"<w:tblPr>.*?</w:tblPr>", fix_tblpr, xml, flags=re.S)
print(f"{n_tab} tabelas de dados com grade e largura total")

# fonte do tema -> Latin Modern Roman (títulos e corpo; Consolas fica p/ código)
theme = parts["word/theme/theme1.xml"].decode("utf8")
theme = re.sub(r'(<a:latin typeface=")[^"]*(")', r"\1Latin Modern Roman\2", theme)
print("fonte do tema: Latin Modern Roman")

out = os.path.join(ROOT, "artigo.docx")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
    for info, data in entries:
        if info.filename == "word/document.xml":
            data = xml.encode("utf8")
        elif info.filename == "word/theme/theme1.xml":
            data = theme.encode("utf8")
        zout.writestr(info, data)
print("artigo.docx final gravado:", out)
