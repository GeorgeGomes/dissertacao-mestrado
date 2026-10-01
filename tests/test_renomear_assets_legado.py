"""Testes do script de renomeação dos assets legados (``renomear_assets_legado``).

Garante que o mapa legado→novo é o mesmo que o código novo produz
(``llm_asset`` + ``asset_variant``), que assets já corretos não mudam (idempotência)
e que a aplicação em pasta temporária (sem git) funciona.

Execução: .venv/bin/python -m unittest tests.test_renomear_assets_legado
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import renomear_assets_legado as rn  # noqa: E402
from execucao_io import asset_variant, llm_asset  # noqa: E402

LEGADO = {
    "bloco1_06_w_distribution__gpt4mini_boxplot.png": "bloco1_06_w_distribution_boxplot__gpt4mini.png",
    "final_04_dataset_overview_seed42__flashlite_problema_d.png": "final_04_dataset_overview_seed42_problema_e__flashlite.png",
    "final_04_dataset_overview_seed7_problema_d.png": "final_04_dataset_overview_seed7_problema_e.png",
    "bloco23_external_svm_meialua_seed42__gpt4mini_gt.png": "bloco1_11_meialua_svm_vs_llm_seed42_gt__gpt4mini.png",
    "bloco23_external_svm_meialua_seed7__flashlite.png": "bloco1_11_meialua_svm_vs_llm_seed7__flashlite.png",
    "final_cross_linearity__gpt4mini_corrigido_hm.csv": "final_cross_linearity_corrigido_hm__gpt4mini.csv",
}
INALTERADOS = [
    "bloco1_02_problema_d_meialua_seed42_overview.png",
    "bloco1_04_oracle_transfer_euclidean_gt.png",
    "bloco1_06_w_distribution_boxplot__gpt4mini.png",
    "bloco1_10_r3r4_comparison_algoritmos_r3__gpt4mini.png",
    "bloco1_phases_abc_20260705_153049.csv",
    "final_09_model_comparison.png",
]


class TestNovoNome(unittest.TestCase):
    def test_mapa_legado(self):
        for antigo, novo in LEGADO.items():
            with self.subTest(antigo=antigo):
                self.assertEqual(rn.novo_nome(antigo), novo)

    def test_inalterados(self):
        for nome in INALTERADOS:
            with self.subTest(nome=nome):
                self.assertEqual(rn.novo_nome(nome), nome)

    def test_bate_com_asset_variant(self):
        novo = asset_variant(llm_asset("/x", "bloco1_06_w_distribution.png", "gpt-4o-mini"), "boxplot")
        self.assertEqual(rn.novo_nome("bloco1_06_w_distribution__gpt4mini_boxplot.png"),
                         os.path.basename(novo))


class TestPlanejarAplicar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # (sem o painel já correto de bloco1_06: colidiria com o destino do legado,
        #  e o script aborta de propósito quando o destino existe)
        for n in list(LEGADO) + [i for i in INALTERADOS if not i.startswith("bloco1_06_")]:
            (self.tmp / n).touch()
        (self.tmp / "dados_sinteticos_seed42").mkdir()
        (self.tmp / "dados_sinteticos_seed42" / "problem_D.csv").touch()
        (self.tmp / "dados_sinteticos_seed42" / "problem_A.csv").touch()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_planejar_e_aplicar_idempotente(self):
        pares = rn.planejar(self.tmp)
        esperado = dict(LEGADO)
        esperado["problem_D.csv"] = "problem_E.csv"
        self.assertEqual({a.name: b.name for a, b in pares}, esperado)
        rn.aplicar(pares, usar_git=False)
        self.assertEqual(rn.planejar(self.tmp), [])
        self.assertTrue((self.tmp / "bloco1_11_meialua_svm_vs_llm_seed42_gt__gpt4mini.png").exists())
        self.assertTrue((self.tmp / "dados_sinteticos_seed42" / "problem_E.csv").exists())
        self.assertFalse((self.tmp / "dados_sinteticos_seed42" / "problem_D.csv").exists())

    def test_substituicao_em_documento(self):
        doc = self.tmp / "artigo.tex"
        doc.write_text("\\includegraphics{bloco23_external_svm_meialua_seed42__gpt4mini_gt.png}\n"
                       "Imagem: bloco23_external_svm_meialua_seed42.png\n", encoding="utf-8")
        mapa = {"bloco23_external_svm_meialua_seed42__gpt4mini_gt.png":
                "bloco1_11_meialua_svm_vs_llm_seed42_gt__gpt4mini.png",
                "bloco23_external_svm_meialua_seed42.png": "bloco1_11_meialua_svm_vs_llm_seed42.png"}
        self.assertEqual(rn.substituir_em_documento(doc, mapa), 2)
        self.assertNotIn("bloco23_external_svm", doc.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
