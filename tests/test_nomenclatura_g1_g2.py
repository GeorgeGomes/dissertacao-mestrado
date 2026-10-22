"""Nomenclatura de 08/10/2026 (E1/E2/E3, G1/G2), o oráculo sobre E3/E1/E2 (14/10/2026) e o subproblema G2.

G2 mostra ao LLM os valores em kg e m (`X_display`) mas estima a métrica no espaço
normalizado (`X`). O teste substitui `collect_llm_decisions` por um dublê que registra
o que foi mostrado ao LLM e responde com o rótulo real, sem chamar a API.
"""
import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402
import data_problems as dp  # noqa: E402


class TestNomesDosSubproblemas(unittest.TestCase):
    def test_especialistas_do_problema_e_sao_e1_e2_e3(self):
        self.assertEqual([c["name"] for c in dm.EXPERT_CONFIGS], ["E1", "E2", "E3"])
        np.testing.assert_array_equal(dm.EXPERT_CONFIGS[0]["w"], dm.EXPERT_W)  # E1 = principal

    def test_nomes_do_bloco_3(self):
        self.assertEqual([v[2] for v in dm.G_NOMES_CLASSE],
                         ["G1_homem_mulher", "G1_homem_mulher_classesAB"])
        self.assertEqual([v[2] for v in dm.G2_NOMES_CLASSE],
                         ["G2_homem_mulher", "G2_homem_mulher_classesAB"])
        # classe 0 = Mulher, classe 1 = Homem nos dois subproblemas
        for v in dm.G_NOMES_CLASSE[:1] + dm.G2_NOMES_CLASSE[:1]:
            self.assertEqual(v[:2], ("Mulher", "Homem"))
        self.assertEqual(dm.G2_NOMES_ATRIBUTO[0], ("peso_kg", "altura_m"))

    def test_nome_do_problema_f(self):
        self.assertEqual(dm.nome_problema_f(42), "F_meia_lua_seed42")
        self.assertTrue(dm.nome_problema_f(7).startswith("F_"))

    def test_oraculo_sobre_e3_e1_e2(self):
        seed = 3
        X_e, _ = dp.create_problem_e_expert(n_samples=60, random_state=seed)
        res = dm.oraculo_passos_1_a_3(X_e, dm.EXPERT_CONFIGS, seed, verbose=False)
        problemas = [r["problem"] for r in res]
        # 14/10/2026: as três métricas sobre as MESMAS gaussianas do E; euclidiano (E3) primeiro = Problema 1
        self.assertEqual(list(dict.fromkeys(problemas)), ["Oráculo sobre E3", "Oráculo sobre E1", "Oráculo sobre E2"])
        self.assertEqual(len(res), 3 * 2)            # 3 problemas × 2 algoritmos; sem extras (13/10/2026)
        self.assertFalse(any("Problema A" in r["problem"] or "sobre B" in r["problem"] for r in res))
        self.assertEqual({r["expert_name"] for r in res if r["problem"] == "Oráculo sobre E3"}, {"E3"})
        self.assertEqual({(r["true_w_0"], r["true_w_1"]) for r in res if r["expert_name"] == "E3"}, {(1.0, 1.0)})
        self.assertTrue(all("accuracy" in r and "fidelity" not in r for r in res))
        self.assertTrue(all(0 <= r["accuracy"] <= 1 for r in res))
        self.assertFalse(hasattr(dm, "ORACLE_ANISO_CONFIGS"))
        self.assertFalse(hasattr(dm, "RUN_ORACLE_ANISO_EXTRAS"))
        self.assertFalse(hasattr(dp, "create_anisotropic_problem"))

    def test_oraculo_rotula_pelo_centroide_mais_proximo_sob_w(self):
        # I4: Partes 1 e 3 usam predict_with_metric(X, centróides geradores, W verdadeiro).
        # Verificação REAL (15/10/2026): a `accuracy` gravada em cada linha tem de ser exatamente a
        # concordância entre predict_with_metric(X_e, EXPERT_CENTROIDS, W_recuperado) e os rótulos
        # predict_with_metric(X_e, EXPERT_CENTROIDS, W_verdadeiro). Se o oráculo voltar a usar os
        # rótulos do make_blobs (y do gerador), a acurácia gravada deixa de bater com essa conta em
        # pelo menos uma linha (os rótulos de E1/E2 sob W anisotrópico diferem dos do gerador).
        from metrics import predict_with_metric
        seed = 3
        X_e, y_blobs = dp.create_problem_e_expert(n_samples=60, random_state=seed)
        C = np.array(dm.EXPERT_CENTROIDS, dtype=float)
        res = dm.oraculo_passos_1_a_3(X_e, dm.EXPERT_CONFIGS, seed, verbose=False)
        self.assertEqual(len(res), 6)
        bate_com_blobs = []
        for r in res:
            w_true = np.array([r["true_w_0"], r["true_w_1"]], dtype=float)
            w_rec = np.array([r["recovered_w_0"], r["recovered_w_1"]], dtype=float)
            y_oraculo = predict_with_metric(X_e, C, w_true)
            self.assertEqual(set(np.unique(y_oraculo).tolist()), {0, 1})
            y_pred = predict_with_metric(X_e, C, w_rec)
            acc_vs_oraculo = float(np.mean(y_pred == y_oraculo))
            self.assertAlmostEqual(r["accuracy"], acc_vs_oraculo, places=9,
                                   msg=f"{r['problem']}/{r['algorithm']}: accuracy não é vs rótulos do oráculo")
            # um W recuperado IGUAL ao verdadeiro reproduz 100% dos rótulos do oráculo
            self.assertEqual(float(np.mean(predict_with_metric(X_e, C, w_true) == y_oraculo)), 1.0)
            bate_com_blobs.append(np.isclose(r["accuracy"], float(np.mean(y_pred == y_blobs))))
        # discriminação: os rótulos do oráculo NÃO são os do make_blobs (ao menos uma linha difere)
        self.assertFalse(all(bate_com_blobs), "a accuracy de todas as linhas bate com os rótulos do make_blobs")
        self.assertTrue(any(not np.array_equal(predict_with_metric(X_e, C, np.array(c["w"], float)), y_blobs)
                            for c in dm.EXPERT_CONFIGS))
        # identificabilidade: nos centróides do A até 14/10/2026 (x2 = 0 nos dois) w2 não mudava rótulo algum;
        # com os centróides de 15/10/2026, (−2; −0,5)/(2; 0,5), w2 volta a importar
        c_antigo = np.array(dp.PROBLEM_A_CENTERS_ATE_2026_10_14)
        X_a, _ = dp.create_problem_a(n_samples=60, random_state=seed, centers=c_antigo)
        np.testing.assert_array_equal(predict_with_metric(X_a, c_antigo, np.array([1.0, 1.0])),
                                      predict_with_metric(X_a, c_antigo, np.array([1.0, 50.0])))
        X_a2, _ = dp.create_problem_a(n_samples=150, random_state=seed)
        c_novo = np.array(dp.PROBLEM_A_CENTERS)
        self.assertEqual(c_novo.tolist(), [[-2.0, -0.5], [2.0, 0.5]])
        self.assertFalse(np.array_equal(predict_with_metric(X_a2, c_novo, np.array([1.0, 1.0])),
                                        predict_with_metric(X_a2, c_novo, np.array([1.0, 50.0]))))


class TestG2MostraKgM(unittest.TestCase):
    def setUp(self):
        self.X, self.y = dp.create_problem_homem_mulher(
            csv_path=os.path.join(SRC_DIR, "..", "dados_reais", "homem_mulher", "peso_altura.csv"))
        self.X_fis, _ = dp.create_problem_homem_mulher_kg_m(
            csv_path=os.path.join(SRC_DIR, "..", "dados_reais", "homem_mulher", "peso_altura_kg_m.csv"))
        self.vistos = []          # (X mostrado, exemplos) de cada chamada ao "LLM"
        self._orig = dm.collect_llm_decisions
        fis = {tuple(np.round(r, 6)) for r in self.X_fis}

        def dublê(X, nome_classe_0, nome_classe_1, examples=None, **kw):
            self.vistos.append((np.asarray(X).copy(), examples))
            # responde com o rótulo real, localizando o ponto físico na base
            idx = [np.argmin(np.abs(self.X_fis - p).sum(axis=1)) for p in np.asarray(X)[:, :2]]
            return self.y[idx].copy(), 0
        dm.collect_llm_decisions = dublê
        self.fis = fis

    def tearDown(self):
        dm.collect_llm_decisions = self._orig

    def test_prompt_recebe_kg_m_e_metrica_usa_normalizado(self):
        out = dm.fg_executar(
                problem_name="G2_homem_mulher", X=self.X, y_true=self.y,
                nome_classe_0="Mulher", nome_classe_1="Homem",
                feature_variants=[("peso_kg", "altura_m")], seeds=[42],
                ks_aprendiz=[0, 4], n_train_ratio=0.7,
                verbose=False, X_display=self.X_fis,
            )
        self.assertTrue(self.vistos)
        for X_mostrado, exemplos in self.vistos:
            # tudo que o LLM viu está em kg e m (valores da base física)
            self.assertTrue(all(tuple(np.round(p[:2], 6)) in self.fis for p in X_mostrado))
            self.assertGreater(X_mostrado[:, 0].min(), 40.0)   # kg, não [-1, 1]
            if exemplos:
                for ex in exemplos:
                    self.assertGreater(ex[0], 40.0)
                    self.assertIn(ex[-1], ("Mulher", "Homem"))
        # a métrica foi estimada no espaço normalizado
        for r in out["passo2"]:
            self.assertLessEqual(np.abs(r["X_aug"][:, :2]).max(), 1.0 + 1e-9)
            self.assertEqual(r["feature_names"], ("peso_kg", "altura_m"))
        # o mapa ponto-a-ponto guarda os dois espaços
        mapa = out["llm_label_maps"][("G2_homem_mulher", 42, "peso_kg", "altura_m", "train")]
        self.assertLessEqual(np.abs(mapa["X"]).max(), 1.0)
        self.assertGreater(mapa["X_display"][:, 0].min(), 40.0)

    def test_sem_x_display_mostra_os_proprios_valores(self):
        if True:
            # dublê procura o ponto físico; aqui o LLM vê valores normalizados, então
            # troca-se o dublê por um que só registra e devolve o rótulo real por índice.
            dm.collect_llm_decisions = lambda X, c0, c1, examples=None, **kw: (
                self.vistos.append((np.asarray(X).copy(), examples)) or
                (self.y[[np.argmin(np.abs(self.X - p).sum(axis=1)) for p in np.asarray(X)[:, :2]]], 0))
            dm.fg_executar(
                problem_name="G1_homem_mulher", X=self.X, y_true=self.y,
                nome_classe_0="Mulher", nome_classe_1="Homem",
                feature_variants=[("x1", "x2")], seeds=[42],
                ks_aprendiz=[0], verbose=False,
            )
        for X_mostrado, _ in self.vistos:
            self.assertLessEqual(np.abs(X_mostrado[:, :2]).max(), 1.0)

    def test_x_display_com_forma_errada_falha(self):
        with self.assertRaises(ValueError):
            dm.fg_executar(
                problem_name="G2_homem_mulher", X=self.X, y_true=self.y,
                nome_classe_0="Mulher", nome_classe_1="Homem",
                feature_variants=[("peso_kg", "altura_m")], seeds=[42],
                ks_aprendiz=[0], verbose=False,
                X_display=self.X_fis[:10],
            )


if __name__ == "__main__":
    unittest.main()
