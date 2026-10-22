"""Problema G, Passo 6: número de atributos MOSTRADOS ao LLM como fator controlado (14/10/2026).

Metodologia IV.7 do orientador: "alimenta-se o prompt com novos atributos, x1·x2 para projeção
em R3 e x1², x2² para projeção em R4". Até 13/10/2026 o prompt só levava os extras quando a
métrica eleita no Passo 5 era R³/R⁴ (confundido com a eleição). Agora `fg_executar(...,
atributos_prompt_aprendiz=(2, 3, 4))` roda o aprendizado em contexto com 2, 3 e 4 atributos no
prompt para o MESMO especialista; o zero-shot (k = 0, sem extras) é uma coleta só.

Execução: .venv/bin/python -m unittest tests.test_g_passo6_atributos_prompt
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dissertacao_mestrado as dm  # noqa: E402


def _dados(n=80, seed=11):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X, y


class TestAtributosNoPrompt(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _dados()
        self.chamadas = []
        self._saved = {"collect": dm.collect_llm_decisions, "reps": dm.N_REPETICOES,
                       "k": list(dm.K_EXEMPLOS_ROTULO_REAL), "cache": dict(dm.CACHE_ROTULOS_LLM),
                       "model": dm.MODEL_NAME, "provider": dm.CURRENT_PROVIDER,
                       "perc": dm.train_relaxed_perceptron, "pred": dm.predict_with_metric}
        dm.N_REPETICOES = 1
        dm.K_EXEMPLOS_ROTULO_REAL = [4]
        dm.CACHE_ROTULOS_LLM.clear()
        dm.MODEL_NAME = "modelo-teste"
        dm.CURRENT_PROVIDER = "openrouter"

        def dublê(X, c0, c1, examples=None, **kw):
            X = np.asarray(X, dtype=float)
            self.chamadas.append({"X": X.copy(), "examples": examples, "tag": kw.get("tag") or dm.CURRENT_TAG,
                                  "extras": kw.get("extra_features_matrix"),
                                  "extras_nomes": kw.get("extra_feature_names")})
            return (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int), 0
        dm.collect_llm_decisions = dublê

        # Passo 7 (15/10/2026): registra a forma de cada treino do Perceptron (X e centróides) e a
        # dimensão de cada predição (X vs W), para verificar que o Perceptron dos mesmos exemplos
        # roda no MESMO espaço (R², R³ ou R⁴) que o LLM viu no Passo 6.
        self.treinos, self.predicoes = [], []
        orig_perc, orig_pred = dm.train_relaxed_perceptron, dm.predict_with_metric

        def perc(X, y, c, **kw):
            self.treinos.append((np.asarray(X).shape, np.asarray(c).shape))
            return orig_perc(X, y, c, **kw)

        def pred(X, c, w):
            self.predicoes.append((np.asarray(X).shape[1], len(np.asarray(w))))
            return orig_pred(X, c, w)
        dm.train_relaxed_perceptron = perc
        dm.predict_with_metric = pred

    def tearDown(self):
        dm.collect_llm_decisions = self._saved["collect"]
        dm.N_REPETICOES = self._saved["reps"]
        dm.K_EXEMPLOS_ROTULO_REAL = self._saved["k"]
        dm.CACHE_ROTULOS_LLM.clear(); dm.CACHE_ROTULOS_LLM.update(self._saved["cache"])
        dm.MODEL_NAME = self._saved["model"]
        dm.CURRENT_PROVIDER = self._saved["provider"]
        dm.train_relaxed_perceptron = self._saved["perc"]
        dm.predict_with_metric = self._saved["pred"]

    def _roda(self, **kw):
        args = dict(problem_name="G1_homem_mulher", X=self.X, y_true=self.y, nome_classe_0="Mulher",
                    nome_classe_1="Homem", feature_variants=[("peso", "altura")], seeds=[5],
                    ks_aprendiz=[0, 4], n_train_ratio=0.7, verbose=True, com_passos_3_a_5=True)
        args.update(kw)
        with redirect_stdout(io.StringIO()) as buf:
            out = dm.fg_executar(**args)
        return out, buf.getvalue()

    def test_tres_dimensoes_no_prompt_mesmo_especialista_e_zero_shot_unico(self):
        out, log = self._roda(atributos_prompt_aprendiz=(2, 3, 4))
        e = out["aprendiz"]
        self.assertEqual({r["n_features"] for r in e}, {2, 3, 4})
        self.assertEqual({(r["n_features"], r["k"]) for r in e}, {(d, k) for d in (2, 3, 4) for k in (0, 4)})
        # o especialista (métrica eleita) é o MESMO nas três dimensões do prompt
        esp = {(r["especialista_regime"], r["especialista_k"], r["especialista_rep"], r["especialista_n_features"]) for r in e}
        self.assertEqual(len(esp), 1)
        regime_e, k_e, rep_e, dim_e = list(esp)[0]
        # ... e é de fato a eleição do Passo 5: a linha do Perceptron (zero-shot ou few-shot, sem teto e
        # sem colapso) de maior acurácia vs rótulo real nos 70 de treino, lida das linhas dos Passos 2 a 5
        cands = [r for r in out["passos_2_a_5"]
                 if r["algorithm"] == "perceptron" and r["regime"] in ("passo2_zero_shot", "passo3_exemplos_rotulo_real")
                 and int(r.get("collapsed", 0)) == 0 and not np.isnan(r["accuracy_metric_vs_true_full"])]
        self.assertTrue(cands)
        melhor = max(r["accuracy_metric_vs_true_full"] for r in cands)
        eleita = [r for r in cands if (r["regime"], int(r["k"]), int(r["rep"]), int(r["n_features"])) == (regime_e, k_e, rep_e, dim_e)]
        self.assertEqual(len(eleita), 1, "a especialista gravada nas linhas do Passo 6 não existe nos Passos 2 a 5")
        self.assertAlmostEqual(eleita[0]["accuracy_metric_vs_true_full"], melhor)
        self.assertIn(dim_e, (2, 3, 4))
        # mesmos índices de exemplo nas três dimensões (o sorteio não depende do prompt)
        self.assertEqual(len({r["example_indices"] for r in e if r["k"] == 4}), 1)
        # zero-shot do Passo 6: coletado UMA vez (não tem extras), repetido nas linhas R3 e R4
        passo6_zero = [c for c in self.chamadas if "|passo=6|" in c["tag"] and c["examples"] is None]
        self.assertEqual(len(passo6_zero), 1)
        self.assertIn("n_atributos_prompt=2", passo6_zero[0]["tag"])
        zero_rows = [r for r in e if r["k"] == 0]
        self.assertEqual(len({r["accuracy_llm_vs_true"] for r in zero_rows}), 1)
        self.assertIn("zero-shot reaproveitado", log)
        # few-shot: o LLM vê 2, 3 ou 4 atributos (exemplos e ponto de teste) conforme a dimensão
        few = [c for c in self.chamadas if "|passo=6|" in c["tag"] and c["examples"]]
        self.assertEqual(len(few), 3)
        por_dim = {}
        for c in few:
            dim = int(c["tag"].split("n_atributos_prompt=")[1])
            por_dim[dim] = c
        self.assertEqual(set(por_dim), {2, 3, 4})
        self.assertIsNone(por_dim[2]["extras"]); self.assertIsNone(por_dim[2]["extras_nomes"])
        self.assertTrue(all(len(ex) == 3 for ex in por_dim[2]["examples"]))          # x1, x2, rótulo
        self.assertEqual(por_dim[3]["extras"].shape[1], 1); self.assertEqual(por_dim[3]["extras_nomes"], ["peso*altura"])
        self.assertTrue(all(len(ex) == 4 for ex in por_dim[3]["examples"]))
        self.assertEqual(por_dim[4]["extras"].shape[1], 2); self.assertEqual(por_dim[4]["extras_nomes"], ["peso²", "altura²"])
        self.assertTrue(all(len(ex) == 5 for ex in por_dim[4]["examples"]))
        # consistência dos extras: x1·x2 e x1², x2² calculados sobre os valores mostrados
        X3 = por_dim[3]["X"]; self.assertTrue(np.allclose(por_dim[3]["extras"][:, 0], X3[:, 0] * X3[:, 1]))
        X4 = por_dim[4]["X"]; self.assertTrue(np.allclose(por_dim[4]["extras"], X4 ** 2))
        for ex in por_dim[4]["examples"]:
            self.assertAlmostEqual(ex[2], ex[0] ** 2); self.assertAlmostEqual(ex[3], ex[1] ** 2)
        # Perceptron do Passo 7 (15/10/2026): treinado nos MESMOS 4 exemplos e no MESMO espaço do prompt,
        # um treino por dimensão (X com 2, 3 e 4 colunas; centróides na mesma dimensão) ...
        passo7 = sorted({(xs, cs) for xs, cs in self.treinos if xs[0] == 4})
        self.assertEqual(passo7, [((4, 2), (2, 2)), ((4, 3), (2, 3)), ((4, 4), (2, 4))])
        # ... nunca prediz com W de dimensão diferente da de X ...
        self.assertTrue(all(d == dw for d, dw in self.predicoes))
        # ... e grava a acurácia só onde há exemplos (k = 4), nunca no zero-shot (k = 0)
        linhas = {(r["n_features"], r["k"]): r for r in e}
        for d in (2, 3, 4):
            self.assertIsNotNone(linhas[(d, 4)]["accuracy_perceptron_baseline_vs_true"])
            self.assertIsNone(linhas[(d, 0)]["accuracy_perceptron_baseline_vs_true"])

    def test_padrao_continua_so_com_dois_atributos(self):
        out, _ = self._roda()
        e = out["aprendiz"]
        self.assertEqual({r["n_features"] for r in e}, {2})
        self.assertTrue(all("n_atributos_prompt=2" in c["tag"] for c in self.chamadas if "|passo=6|" in c["tag"]))
        self.assertTrue(all(c["extras"] is None for c in self.chamadas))

    def test_dimensao_invalida_falha(self):
        with self.assertRaises(ValueError):
            self._roda(atributos_prompt_aprendiz=(2, 5))

    def test_constante_e_custo(self):
        self.assertEqual(dm.ATRIBUTOS_PROMPT_PASSO6_G, (2, 3, 4))
        self.assertGreaterEqual(dm.CUSTO_ESTIMADO_USD["bloco3"], 8.0)
        self.assertAlmostEqual(dm.CUSTO_ESTIMADO_USD["completa"],
                               sum(v for k, v in dm.CUSTO_ESTIMADO_USD.items() if k != "completa"))


if __name__ == "__main__":
    unittest.main()
