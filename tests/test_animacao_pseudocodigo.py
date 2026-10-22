"""O pseudocódigo da animação/.docx (src/animacao_perceptron/pseudocodigo.py) continua fiel ao
`relaxed_perceptron.py`: cada linha citada na tabela de correspondência contém o trecho esperado,
o mapa evento → pseudocódigo aponta para linhas numeradas e o template usa os placeholders."""
import os
import sys
import unittest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "src", "animacao_perceptron"))
sys.path.insert(0, os.path.join(AQUI, "..", "src"))

import pseudocodigo as pc  # noqa: E402


class TestCorrespondencia(unittest.TestCase):
    def test_cada_linha_do_py_contem_o_trecho(self):
        src = pc.linhas_py()
        for n, ls, trecho in pc.CORRESPONDENCIA:
            bloco = "\n".join(src[l - 1] for l in ls)
            self.assertIn(trecho, bloco, f"pseudocódigo {n}: linhas {ls} do .py não contêm {trecho!r}; a tabela envelheceu")

    def test_numeracao_continua_e_ids(self):
        nums = [n for n, _ in pc.LINHAS]
        self.assertEqual(nums, list(range(1, len(nums) + 1)))
        self.assertEqual(len(pc.TEXTO), len(pc.IDS))
        self.assertEqual([i for i in pc.IDS if i is not None], nums)
        for n, t in pc.LINHAS:
            self.assertTrue(t.strip().startswith(str(n)), t)

    def test_py2pseudo_aponta_para_linhas_existentes(self):
        nums = {n for n, _ in pc.LINHAS}
        src = pc.linhas_py()
        for py, ps in pc.PY2PSEUDO.items():
            self.assertIn(ps, nums)
            self.assertTrue(src[py - 1].strip(), f"linha {py} do .py está vazia")

    def test_mudancas_de_15_10_estao_no_pseudocodigo(self):
        txt = "\n".join(pc.TEXTO)
        self.assertIn("w ← w/N; α ← α/N", txt)          # B
        self.assertIn("erros_min", txt)                   # C
        self.assertIn("μ ←", txt)                         # D
        self.assertIn("Min {Max {f, 0}, 1}", txt)         # clip [0, 1]

    def test_template_usa_placeholders_e_core_js_segue_o_perceptron(self):
        base = os.path.join(AQUI, "..", "src", "animacao_perceptron")
        with open(os.path.join(base, "template.html"), encoding="utf-8") as f:
            t = f.read()
        for ph in ("__PSEUDO_CODIGO__", "__PSEUDO_CODIGO_IDS__", "__PY2PSEUDO__", "__EXECUCAO__"):
            self.assertIn(ph, t)
        self.assertNotIn("min_global_violations", t)
        with open(os.path.join(base, "core.js"), encoding="utf-8") as f:
            js = f.read()
        self.assertIn("clip(factor, 0.0, 1.0)", js)
        self.assertIn("w = w.map(v => v / norm_fim)", js)
        self.assertIn("min_global_erros", js)
        self.assertIn("margemMinima(result.w)", js)


if __name__ == "__main__":
    unittest.main()
