"""Testes do gerador de amostras da base homem × mulher para o orientador.

Verifica tamanhos, balanceamento, disjunção total entre os 6 conjuntos, alinhamento dos
rótulos com a base (classe 1 = H), a conversão para kg/m (regra de três ancorada no
mínimo e máximo da base) e o determinismo.
"""
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import gerar_amostras_homem_mulher as g  # noqa: E402

NOMES = ["X1", "X2", "X3", "T1", "T2", "T3"]


class TestGerarAmostras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.saida = Path(cls.tmp.name) / "amostras"
        cls.escritos = g.gerar(seed=42, saida=cls.saida, fazer_zip=True)
        cls.base = pd.read_csv(g.CSV_BASE)
        cls.manifest = pd.read_csv(cls.saida / "manifest.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_arquivos_gerados(self):
        nomes = {p.name for p in self.escritos}
        esperados = set()
        for n in NOMES:
            prefixo, k = n[0], n[1:]
            esperados |= {f"{prefixo}{k}.csv", f"{prefixo}'{k}.csv", f"R{prefixo}{k}.csv"}
        esperados |= {"manifest.csv", "README.md", g.NOME_ZIP}
        self.assertEqual(nomes, esperados)
        self.assertEqual(len(esperados) - 3, 18)

    def test_tamanhos_e_balanceamento(self):
        for n in NOMES:
            esperado = 20 if n.startswith("X") else 10
            dados = pd.read_csv(self.saida / f"{n}.csv")
            rot = pd.read_csv(self.saida / f"R{n}.csv")
            self.assertEqual(len(dados), esperado, n)
            self.assertEqual(len(rot), esperado, n)
            self.assertEqual(list(dados.columns), ["x1", "x2"])
            self.assertEqual(list(rot.columns), ["classe"])
            contagem = rot["classe"].value_counts()
            self.assertEqual(contagem["H"], esperado // 2, n)
            self.assertEqual(contagem["M"], esperado // 2, n)

    def test_conjuntos_disjuntos(self):
        ids = self.manifest["id_base"].tolist()
        self.assertEqual(len(ids), 90)
        self.assertEqual(len(set(ids)), 90)

    def test_rotulos_alinhados_com_a_base(self):
        for n in NOMES:
            sub = self.manifest[self.manifest["arquivo"] == n].sort_values("linha")
            dados = pd.read_csv(self.saida / f"{n}.csv")
            rot = pd.read_csv(self.saida / f"R{n}.csv")
            base = self.base.iloc[sub["id_base"].values]
            np.testing.assert_allclose(dados[["x1", "x2"]].values,
                                       base[["peso", "altura"]].values)
            esperado = ["H" if c == 1 else "M" for c in base["classe"]]
            self.assertEqual(rot["classe"].tolist(), esperado, n)
            self.assertEqual(sub["classe"].tolist(), esperado, n)

    def test_linhas_embaralhadas(self):
        # Classes não podem aparecer todas agrupadas (primeira metade H, segunda M etc.)
        for n in NOMES:
            rot = pd.read_csv(self.saida / f"R{n}.csv")["classe"].tolist()
            metade = len(rot) // 2
            self.assertNotEqual(rot, ["H"] * metade + ["M"] * metade, n)
            self.assertNotEqual(rot, ["M"] * metade + ["H"] * metade, n)

    def test_versao_fisica_convertida_por_regra_de_tres(self):
        # Âncoras = mínimo e máximo da base inteira (−1 e +1), como pede o orientador.
        v_min = self.base[["peso", "altura"]].min().values
        v_max = self.base[["peso", "altura"]].max().values
        for n in NOMES:
            prefixo, k = n[0], n[1:]
            neutro = pd.read_csv(self.saida / f"{n}.csv")
            fisico = pd.read_csv(self.saida / f"{prefixo}'{k}.csv")
            self.assertEqual(list(fisico.columns), ["peso_kg", "altura_m"])
            self.assertEqual(len(fisico), len(neutro))
            kg_esp = np.round(g.desnormalizar(neutro["x1"], v_min[0], v_max[0], *g.PESO_KG), 1)
            m_esp = np.round(g.desnormalizar(neutro["x2"], v_min[1], v_max[1], *g.ALTURA_M), 2)
            np.testing.assert_allclose(fisico["peso_kg"].values, kg_esp, atol=1e-9)
            np.testing.assert_allclose(fisico["altura_m"].values, m_esp, atol=1e-9)
            self.assertTrue((fisico["peso_kg"] >= g.PESO_KG[0]).all() and (fisico["peso_kg"] <= g.PESO_KG[1]).all())
            self.assertTrue((fisico["altura_m"] >= g.ALTURA_M[0]).all() and (fisico["altura_m"] <= g.ALTURA_M[1]).all())

    def test_calibracao_extremos_e_medias_na_base(self):
        # Em toda a base: menor peso → 45 kg, maior → 110 kg; alturas 1,50 → 1,95 m.
        X = self.base[["peso", "altura"]].values
        anc = np.vstack([X.min(axis=0), X.max(axis=0)])
        fis = g.converter_para_fisico(X, anc)
        self.assertEqual(fis[:, 0].min(), g.PESO_KG[0])
        self.assertEqual(fis[:, 0].max(), g.PESO_KG[1])
        self.assertEqual(fis[:, 1].min(), g.ALTURA_M[0])
        self.assertEqual(fis[:, 1].max(), g.ALTURA_M[1])
        y = self.base["classe"].values
        self.assertGreater(fis[y == 1, 0].mean(), fis[y == 0, 0].mean())  # homens mais pesados
        self.assertGreater(fis[y == 1, 1].mean(), fis[y == 0, 1].mean())  # e mais altos

    def test_faixas_parametrizaveis(self):
        with tempfile.TemporaryDirectory() as outro:
            g.gerar(seed=42, saida=Path(outro), peso_kg=(50.0, 120.0), altura_m=(1.55, 2.00))
            fis = pd.concat(pd.read_csv(Path(outro) / f"{n[0]}'{n[1]}.csv") for n in NOMES)
            self.assertGreaterEqual(fis["peso_kg"].min(), 50.0)
            self.assertLessEqual(fis["peso_kg"].max(), 120.0)
            self.assertGreaterEqual(fis["altura_m"].min(), 1.55)
            self.assertLessEqual(fis["altura_m"].max(), 2.00)
            # Sorteio não depende das faixas: X1 idêntico.
            self.assertEqual((Path(outro) / "X1.csv").read_text(), (self.saida / "X1.csv").read_text())

    def test_zip_contem_18_csvs_e_readme(self):
        with zipfile.ZipFile(self.saida / g.NOME_ZIP) as zf:
            nomes = set(zf.namelist())
        self.assertEqual(len([n for n in nomes if n.endswith(".csv")]), 18)
        self.assertIn("README.md", nomes)
        self.assertNotIn("manifest.csv", nomes)

    def test_determinismo(self):
        with tempfile.TemporaryDirectory() as outro:
            g.gerar(seed=42, saida=Path(outro))
            for n in NOMES:
                a = (self.saida / f"{n}.csv").read_text()
                b = (Path(outro) / f"{n}.csv").read_text()
                self.assertEqual(a, b, n)
        with tempfile.TemporaryDirectory() as outro:
            g.gerar(seed=7, saida=Path(outro))
            diferente = any(
                (self.saida / f"{n}.csv").read_text() != (Path(outro) / f"{n}.csv").read_text()
                for n in NOMES
            )
            self.assertTrue(diferente)


class TestDesnormalizar(unittest.TestCase):
    def test_extremos_e_linearidade(self):
        v = np.array([-1.0, 0.0, 1.0])
        out = g.desnormalizar(v, -1.0, 1.0, 45.0, 110.0)
        np.testing.assert_allclose(out, [45.0, 77.5, 110.0])

    def test_intervalo_nulo(self):
        with self.assertRaises(ValueError):
            g.desnormalizar([0.0], 1.0, 1.0, 45.0, 110.0)


if __name__ == "__main__":
    unittest.main()
