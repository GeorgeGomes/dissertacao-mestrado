"""Testes do motor de tabelas (`tabelas.py`) e do catálogo completo (`gerar_tabelas.py`).

- `Tabela` → .tex booktabs com grupos (\\multicolumn + \\cmidrule) e → .png (matplotlib);
- `latex_para_texto` converte os fragmentos LaTeX usados nas células;
- nomes semânticos com alias do modelo antes da extensão;
- catálogo: roda sobre os CSVs da execução mais recente (copiados para uma pasta
  temporária, sem llm_interactions) e produz .tex + .png + INDICE.md sem exceções.

Execução: .venv/bin/python -m unittest tests.test_tabelas
"""
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import tabelas as tb  # noqa: E402
import gerar_tabelas as gt  # noqa: E402

RESULTADOS = Path(SRC_DIR).resolve().parent / "resultados"


def _tabela():
    return tb.Tabela(slug="bloco9_teste_x", modelo="openai/gpt-4o-mini",
                     caption="Teste ($\\kappa$, R$^2$) do motor.",
                     headers=["Item", "a", "b", "$\\Delta$"],
                     rows=[["um", "1,0", "2,0", "+1,0"], ["dois", "3,0", "3,0", "+0,0"], ["três", "—", "4,0", "—"]],
                     align="lrrr", grupos=[("Valores (\\%)", 2, 3)], destaque={1}, midrules_apos={1},
                     rodape="Rodapé com x1·x2 e \\textbf{negrito}.")


class TestMotor(unittest.TestCase):
    def test_nomes_com_alias(self):
        t = _tabela()
        t.nn = 7
        self.assertEqual(t.nome_png(), "bloco9_table_07_teste_x__gpt4mini_gray.png")
        self.assertEqual(t.nome_tex(), "bloco9_table_07_teste_x__gpt4mini.tex")
        t.modelo = None
        self.assertEqual(t.nome_png(), "bloco9_table_07_teste_x_gray.png")

    def test_tex_booktabs_com_grupos_e_destaque(self):
        tex = _tabela().tex()
        for marca in ("\\toprule", "\\midrule", "\\bottomrule", "\\multicolumn{2}{c}{Valores (\\%)}",
                      "\\cmidrule(lr){2-3}", "\\textbf{dois}", "\\caption{Teste", "\\label{tab:bloco9_teste_x_gpt4mini}"):
            self.assertIn(marca, tex)
        self.assertEqual(tex.count("\\midrule"), 2)          # cabeçalho + midrules_apos={1}
        self.assertNotIn("\\textbf{um}", tex)

    def test_latex_para_texto(self):
        self.assertEqual(tb.latex_para_texto("Fid.\\ vs LLM (\\%)"), "Fid. vs LLM (%)")
        self.assertEqual(tb.latex_para_texto("$\\kappa$ R$^2$ $\\Delta$ $-$0,5 \\textbf{x}"), "κ R² Δ −0,5 x")
        self.assertEqual(tb.latex_para_texto("peso\\_kg ``a''"), "peso_kg “a”")

    def test_png_e_escrito(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p = _tabela().png(tmp / "t.png")
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 5_000)
            with open(p, "rb") as fh:
                self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")
        finally:
            shutil.rmtree(tmp)

    def test_formatadores(self):
        self.assertEqual(tb.num(0.1234), "12,3"); self.assertEqual(tb.num(float("nan")), "—")
        self.assertEqual(tb.num(0.5, 3, pct=False), "0,500")
        self.assertEqual(tb.delta(0.5, 0.5), "+0,0"); self.assertEqual(tb.delta(0.6, 0.5), "$-$10,0")
        self.assertFalse(hasattr(tb, "media_min_max"))  # removida em 09/10/2026 (só o teste a usava)

    def test_faixa_kappa_landis_koch(self):
        # Seção IV.3 do orientador (Landis e Koch 1977); limite inferior inclusivo.
        self.assertEqual(tb.faixa_kappa(-0.1), "pior que o acaso")
        self.assertEqual(tb.faixa_kappa(0.0), "fraca"); self.assertEqual(tb.faixa_kappa(0.19), "fraca")
        self.assertEqual(tb.faixa_kappa(0.2), "razoável"); self.assertEqual(tb.faixa_kappa(0.39), "razoável")
        self.assertEqual(tb.faixa_kappa(0.4), "moderada"); self.assertEqual(tb.faixa_kappa(0.45), "moderada")
        self.assertEqual(tb.faixa_kappa(0.6), "substancial"); self.assertEqual(tb.faixa_kappa(0.79), "substancial")
        self.assertEqual(tb.faixa_kappa(0.8), "quase perfeita"); self.assertEqual(tb.faixa_kappa(1.0), "quase perfeita")
        self.assertEqual(tb.faixa_kappa(float("nan")), "—"); self.assertEqual(tb.faixa_kappa(None), "—")


class TestCatalogo(unittest.TestCase):
    """Roda o catálogo sobre os CSVs da execução mais recente (completa, senão qualquer), sem interações."""

    @classmethod
    def setUpClass(cls):
        pastas = sorted(RESULTADOS.glob("execucao_*_completa")) or sorted(RESULTADOS.glob("execucao_*"))
        if not pastas:
            raise unittest.SkipTest("nenhuma execução em resultados/")
        cls.origem = pastas[-1]
        cls.tmp = Path(tempfile.mkdtemp())
        for f in cls.origem.glob("*.csv"):
            shutil.copy(f, cls.tmp / f.name)
        # Execuções anteriores a 13/10/2026 (nomes antigos): migra a CÓPIA, nunca a pasta commitada.
        import renomear_assets_legado as rn
        rn.aplicar(rn.planejar(cls.tmp), usar_git=False)
        rn.migrar_conteudo(cls.tmp, aplicar=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_catalogo_gera_tex_png_e_indice(self):
        escritos = gt.gerar_tudo(self.tmp, png=True, docx=False, audit=None, verbose=False)
        tex = [p for p in escritos if p.suffix == ".tex" and p.name != "todas_as_tabelas.tex"]
        png = [p for p in escritos if p.suffix == ".png"]
        self.assertGreaterEqual(len(tex), 10)
        self.assertEqual(len(tex), len(png))                      # cada tabela nos dois formatos
        self.assertTrue(all(p.parent == self.tmp for p in png))    # PNG na raiz da execução
        self.assertTrue(all(p.parent == self.tmp / "tabelas" for p in tex))
        nomes = {p.name for p in png}
        self.assertTrue(any(n.startswith("bloco1_table_01_A_passos_2_a_5__") for n in nomes))
        self.assertTrue(all(re.match(r"^(bloco\d+|final)_table_\d{2}_.+_gray\.png$", n) for n in nomes))
        indice = (self.tmp / "tabelas" / "INDICE.md").read_text(encoding="utf-8")
        for p in png:
            self.assertIn(p.name, indice)
        self.assertTrue(any(p.name.startswith("bloco1_table_01_A_passos_2_a_5__") for p in tex))
        self.assertTrue(all(re.match(r"^(bloco\d+|final)_table_\d{2}_.+\.tex$", p.name) for p in tex))

    def test_toda_tabela_tem_legenda_e_rodape(self):
        D = gt.carregar_tudo(self.tmp); D["_pasta"] = self.tmp
        for t in gt.catalogo(D, None):
            self.assertTrue(t.caption and t.caption.strip(), t.slug)
            self.assertTrue(t.rodape and t.rodape.strip(), t.slug)

    def test_catalogo_sem_csv_nao_falha(self):
        vazio = Path(tempfile.mkdtemp())
        try:
            self.assertEqual(gt.gerar_tudo(vazio, verbose=False), [])
        finally:
            shutil.rmtree(vazio)


if __name__ == "__main__":
    unittest.main()
