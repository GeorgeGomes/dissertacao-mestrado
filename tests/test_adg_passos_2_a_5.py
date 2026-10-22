"""Testes dos Passos 2 a 5 dos Problemas A e D (`adg_passos_2_a_5`) e da transferência B/C (Passo 3)
contra todas as métricas do A — sem chamar a API (dublê de `async_llm_classify_point_openai`).

Contratos (pedido do orientador, 01/10/2026):
1. Exemplos few-shot rotulados pelo GROUND TRUTH, k/2 por classe, fora do conjunto
   avaliado (anti-leakage), reprodutíveis por `seed + rep`.
2. `adg_passos_2_a_5` produz uma linha por (regime × k × rep × n_features × algoritmo),
   faz n chamadas no zero-shot (0 com cache) e n−k por repetição few-shot, grava a `tag`
   de auditoria e nunca pula um colapso em classe única (linhas com NaN, collapsed=1).
3. No regime `passo5_teto` (teto do estimador) não há LLM: `fidelity_vs_llm` e a acurácia
   do LLM são NaN e `accuracy_teto_vs_true` guarda a acurácia do W ajustado ao rótulo real.
4. `bc_passo3_consistencia` gera uma linha por (métrica de origem, algoritmo) sem chamadas,
   e `a_passo6_eleger_metrica` elege exatamente uma (regime, k) por modelo, nunca o GT.

Execução: .venv/bin/python -m unittest tests.test_adg_passos_2_a_5
"""
import os
import re
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402

META = {"model_resolved": "modelo-teste", "inference_provider": "Teste"}
_RE_X1 = re.compile(r"x1\s*=\s*(-?\d+\.\d+)")


class _FakeLLM:
    """Responde 'A' se x1 < 0 e 'B' caso contrário (lê o ÚLTIMO x1 do prompt = ponto
    a classificar; os exemplos few-shot vêm antes). Com `constante`, responde sempre
    o mesmo rótulo (simula colapso em classe única)."""

    def __init__(self, constante=None):
        self.calls = 0
        self.constante = constante

    async def __call__(self, async_client, model_name, prompt, temperature, system_msg, **kwargs):
        self.calls += 1
        if self.constante is not None:
            return self.constante, dict(META)
        x1 = float(_RE_X1.findall(prompt)[-1])
        return ("A" if x1 < 0 else "B"), dict(META)


def _dados(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal([-2, 0], 0.8, size=(n // 2, 2)), rng.normal([2, 0], 0.8, size=(n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


class _Base(unittest.TestCase):
    def setUp(self):
        self._saved = {
            "INITIAL_BACKOFF": dm.INITIAL_BACKOFF,
            "fake": dm.async_llm_classify_point_openai,
            "async_client": dm.async_client, "MODEL_NAME": dm.MODEL_NAME,
            "CURRENT_PROVIDER": dm.CURRENT_PROVIDER, "N_REPETICOES": dm.N_REPETICOES,
            "interactions": list(dm.LLM_INTERACTIONS), "cache": dict(dm.CACHE_ROTULOS_LLM),
        }
        dm.INITIAL_BACKOFF = 0
        dm.async_client = object()
        dm.MODEL_NAME = "modelo-teste"
        dm.CURRENT_PROVIDER = "openrouter"
        dm.N_REPETICOES = 2
        dm.LLM_INTERACTIONS.clear()
        dm.CACHE_ROTULOS_LLM.clear()

    def tearDown(self):
        dm.INITIAL_BACKOFF = self._saved["INITIAL_BACKOFF"]
        dm.async_llm_classify_point_openai = self._saved["fake"]
        dm.async_client = self._saved["async_client"]
        dm.MODEL_NAME = self._saved["MODEL_NAME"]
        dm.CURRENT_PROVIDER = self._saved["CURRENT_PROVIDER"]
        dm.N_REPETICOES = self._saved["N_REPETICOES"]
        dm.LLM_INTERACTIONS.clear(); dm.LLM_INTERACTIONS.extend(self._saved["interactions"])
        dm.CACHE_ROTULOS_LLM.clear(); dm.CACHE_ROTULOS_LLM.update(self._saved["cache"])


class TestPasso3ExemplosRotuloReal(_Base):
    def test_k_meio_por_classe_fora_da_avaliacao_e_reprodutivel(self):
        X, y = _dados()
        ex, idx, mask = dm.passo3_exemplos_rotulo_real(X, y, 10, "A", "B", random_state=42 + 1)
        self.assertEqual(len(ex), 10)
        self.assertEqual(sum(1 for e in ex if e[-1] == "A"), 5)
        self.assertEqual(sum(1 for e in ex if e[-1] == "B"), 5)
        self.assertEqual(len(set(idx.tolist())), 10)
        self.assertFalse(mask[idx].any()); self.assertEqual(mask.sum(), len(X) - 10)
        # rótulo do exemplo = ground truth do índice sorteado
        for (x1, x2, lab), i in zip(ex, idx):
            self.assertEqual(lab, "A" if y[i] == 0 else "B")
            self.assertAlmostEqual(x1, X[i, 0]); self.assertAlmostEqual(x2, X[i, 1])
        ex2, idx2, _ = dm.passo3_exemplos_rotulo_real(X, y, 10, "A", "B", random_state=42 + 1)
        ex3, idx3, _ = dm.passo3_exemplos_rotulo_real(X, y, 10, "A", "B", random_state=42 + 2)
        self.assertEqual(idx.tolist(), idx2.tolist())
        self.assertNotEqual(idx.tolist(), idx3.tolist())


class TestAdgPassos2a5(_Base):
    def _run(self, fake, X, y, **kw):
        dm.async_llm_classify_point_openai = fake
        return dm.adg_passos_2_a_5(X, y, "A", "B", "A", 42, "openrouter", "modelo-teste", 0.0,
                                   n_features_list=(2, 3), ks_rotulo_real=[4], verbose=False, **kw)

    def test_linhas_chamadas_tag_e_cache(self):
        X, y = _dados(); n = len(X); fake = _FakeLLM()
        out = self._run(fake, X, y)
        regimes = [(r["regime"], r["k"], r["rep"]) for r in out["rows"]]
        # (passo2_zero_shot + 2 reps passo3 + passo5_teto) × 2 n_feat × 2 algoritmos
        self.assertEqual(len(out["rows"]), (1 + 2 + 1) * 2 * 2)
        self.assertEqual(fake.calls, n + 2 * (n - 4)); self.assertEqual(out["n_calls"], fake.calls)
        self.assertEqual(set(regimes), {("passo2_zero_shot", 0, 0), ("passo3_exemplos_rotulo_real", 4, 0), ("passo3_exemplos_rotulo_real", 4, 1), ("passo5_teto", 0, 0)})
        few = [r for r in out["rows"] if r["regime"] == "passo3_exemplos_rotulo_real"]
        self.assertTrue(all(r["n_eval"] == n - 4 and r["n_examples"] == 4 for r in few))
        self.assertTrue(all(r["random_state_sorteio"] == 42 + r["rep"] for r in few))
        self.assertTrue(all(len(r["example_indices"].split(";")) == 4 for r in few))
        tags = {i["tag"] for i in dm.LLM_INTERACTIONS}
        self.assertIn("bloco1|problema=A|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42", tags)
        self.assertIn("bloco1|problema=A|passo=3|regime=passo3_exemplos_rotulo_real|k=4|rep=1|seed=42", tags)
        # chave inclui os nomes de classe (08/10/2026) e de atributo (13/10/2026): outro par de
        # nomes não reaproveita rótulos
        self.assertEqual(set(dm.CACHE_ROTULOS_LLM),
                         {("openrouter", "modelo-teste", 42, "A", "A", "B", "x1", "x2", 0, 0),
                          ("openrouter", "modelo-teste", 42, "A", "A", "B", "x1", "x2", 4, 0),
                          ("openrouter", "modelo-teste", 42, "A", "A", "B", "x1", "x2", 4, 1)})
        # LLM "perfeito" (x1<0 → A) sobre dados separáveis: fidelidade e acurácia altas
        zs = [r for r in out["rows"] if r["regime"] == "passo2_zero_shot" and r["n_features"] == 2]
        self.assertTrue(all(r["collapsed"] == 0 and r["fidelity_vs_llm"] > 0.9 for r in zs))
        self.assertEqual(set(r["algorithm"] for r in zs), {"perceptron", "nnls"})
        self.assertTrue(all(0 <= r["accuracy_metric_vs_true_full"] <= 1 for r in out["rows"] if r["collapsed"] == 0))

    def test_cache_zero_shot_evita_chamadas(self):
        X, y = _dados(); fake = _FakeLLM()
        y_zero = (X[:, 0] >= 0).astype(int)
        out = self._run(fake, X, y, y_llm_zero_cached=y_zero)
        self.assertEqual(fake.calls, 2 * (len(X) - 4))
        np.testing.assert_array_equal(out["labels"][("passo2_zero_shot", 0, 0)]["y_llm_eval"], y_zero)

    def test_passo5_teto(self):
        X, y = _dados(); out = self._run(_FakeLLM(), X, y)
        gt = [r for r in out["rows"] if r["regime"] == "passo5_teto"]
        self.assertEqual(len(gt), 4)
        self.assertTrue(all(np.isnan(r["llm_accuracy_vs_true"]) for r in gt))
        # M9: no teto não há LLM → fidelidade NaN; a acurácia do teto vai em coluna própria
        self.assertTrue(all(np.isnan(r["fidelity_vs_llm"]) for r in gt))
        self.assertTrue(all(r["accuracy_teto_vs_true"] > 0.95 for r in gt))          # separável: teto ~1
        self.assertTrue(all(r["accuracy_teto_vs_true"] == r["accuracy_metric_vs_true_full"] for r in gt))
        self.assertTrue(all(0 <= r["fidelity_teto_vs_llm_passo2"] <= 1 for r in gt))
        outros = [r for r in out["rows"] if r["regime"] != "passo5_teto"]
        self.assertTrue(all(np.isnan(r["fidelity_teto_vs_llm_passo2"]) for r in outros))
        self.assertTrue(all(np.isnan(r["accuracy_teto_vs_true"]) for r in outros))
        self.assertFalse(any("accuracy_gt_metric_vs_llm_zero" in r for r in out["rows"]))

    def test_classe_minoritaria_abaixo_do_piso_e_colapso(self):
        # I13: 149/1 não é "válido"; com menos de MIN_PONTOS_CLASSE_MINORITARIA pontos na
        # classe menos frequente a coleta é colapso (collapsed=1, NaN).
        self.assertEqual(dm.MIN_PONTOS_CLASSE_MINORITARIA, 5)
        self.assertTrue(dm.colapsou(np.array([0] * 149 + [1])))
        self.assertTrue(dm.colapsou(np.array([0] * 146 + [1] * 4)))
        self.assertFalse(dm.colapsou(np.array([0] * 145 + [1] * 5)))
        self.assertTrue(dm.colapsou(np.zeros(10, int)))
        X, y = _dados()
        y_zero = np.zeros(len(X), int); y_zero[:3] = 1          # 57/3 → colapso
        out = self._run(_FakeLLM(), X, y, y_llm_zero_cached=y_zero)
        zs = [r for r in out["rows"] if r["regime"] == "passo2_zero_shot"]
        self.assertTrue(all(r["collapsed"] == 1 and np.isnan(r["fidelity_vs_llm"]) for r in zs))

    def test_colapso_gera_linhas_nan_sem_pular(self):
        X, y = _dados(); out = self._run(_FakeLLM(constante="A"), X, y)
        llm_rows = [r for r in out["rows"] if r["regime"] != "passo5_teto"]
        self.assertEqual(len(llm_rows), (1 + 2) * 2 * 2)
        self.assertTrue(all(r["collapsed"] == 1 and np.isnan(r["fidelity_vs_llm"]) and np.isnan(r["w_0"]) for r in llm_rows))
        self.assertTrue(all(r["n_classe_0_llm"] == r["n_eval"] and r["n_classe_1_llm"] == 0 for r in llm_rows))
        self.assertTrue(all(0 <= r["llm_accuracy_vs_true"] <= 1 for r in llm_rows))
        self.assertTrue(all(m["collapsed"] == 1 for k, m in out["metrics"].items() if k[0] != "passo5_teto"))
        # o teto pelo ground truth não depende do LLM
        self.assertTrue(all(r["collapsed"] == 0 for r in out["rows"] if r["regime"] == "passo5_teto"))


class TestBcPasso3Consistencia(_Base):
    def test_linhas_por_metrica_e_melhor_metrica(self):
        X, y = _dados(); dm.async_llm_classify_point_openai = _FakeLLM()
        out = dm.adg_passos_2_a_5(X, y, "A", "B", "A", 42, "openrouter", "modelo-teste", 0.0,
                                  n_features_list=(2,), ks_rotulo_real=[4], verbose=False)
        Xb, yb = _dados(n=40, seed=3)
        y_llm_b = (Xb[:, 0] >= 0).astype(int)
        base = {"provider": "openrouter", "model": "modelo-teste", "temperature": 0.0, "random_seed": 42}
        rows = dm.bc_passo3_consistencia(Xb, y_llm_b, yb, "B", out["metrics"], base)
        self.assertEqual(len(rows), len(out["metrics"]) * 2)   # 4 métricas (zero, 2×few, GT) × 2 algoritmos
        self.assertEqual({r["metrica_regime"] for r in rows}, {"passo2_zero_shot", "passo3_exemplos_rotulo_real", "passo5_teto"})
        self.assertTrue(all(r["collapsed_metrica"] == 0 and 0 <= r["consistency"] <= 1 and -1 <= r["kappa"] <= 1 for r in rows))
        self.assertTrue(all(r["n_eval"] == 40 and r["problem"] == "B" for r in rows))
        best = dm.a_passo6_eleger_metrica(out["rows"], rows)
        self.assertEqual(set(best), {"modelo-teste"})
        reg, k, fid = best["modelo-teste"]
        self.assertNotEqual(reg, "passo5_teto"); self.assertTrue(0 <= fid <= 1)
        marcadas = {(r["metrica_regime"], r["metrica_k"]) for r in rows if r["eleita_passo6"] == 1}
        self.assertEqual(marcadas, {(reg, k)})
        self.assertTrue(all(not np.isnan(r["metrica_fidelidade_a"]) for r in rows if r["metrica_regime"] != "passo5_teto"))

    def test_llm_colapsado_em_b_grava_nan_e_collapsed_llm(self):
        # M6: colapso do LLM em B/C → consistência/κ/F1 NaN e collapsed_llm=1 (sem fallback
        # aos centróides do A); a coluna euclidean_consistency não existe mais (M15).
        X, y = _dados(); dm.async_llm_classify_point_openai = _FakeLLM()
        out = dm.adg_passos_2_a_5(X, y, "A", "B", "A", 42, "openrouter", "modelo-teste", 0.0,
                                  n_features_list=(2,), ks_rotulo_real=[], verbose=False)
        Xb, yb = _dados(n=20, seed=5)
        rows = dm.bc_passo3_consistencia(Xb, np.zeros(20, int), yb, "C", out["metrics"],
                                       {"provider": "p", "model": "m", "temperature": 0.0, "random_seed": 42})
        self.assertTrue(rows)
        self.assertTrue(all(r["collapsed_llm"] == 1 for r in rows))
        self.assertTrue(all(np.isnan(r["consistency"]) and np.isnan(r["kappa"]) and np.isnan(r["f1"]) for r in rows))
        self.assertFalse(any("euclidean_consistency" in r for r in rows))
        # sem colapso: collapsed_llm=0 e consistência medida
        rows_ok = dm.bc_passo3_consistencia(Xb, (Xb[:, 0] >= 0).astype(int), yb, "C", out["metrics"],
                                          {"provider": "p", "model": "m", "temperature": 0.0, "random_seed": 42})
        self.assertTrue(all(r["collapsed_llm"] == 0 and 0 <= r["consistency"] <= 1 for r in rows_ok))


class TestFigurasSimbolos(_Base):
    def test_plot_simbolos_passos_gera_bloco1_02_11_13(self):
        import shutil, tempfile
        tmp = tempfile.mkdtemp()
        try:
            X, y = _dados(); dm.async_llm_classify_point_openai = _FakeLLM()
            out = dm.adg_passos_2_a_5(X, y, "A", "B", "A", 42, "openrouter", "openai/gpt-4o-mini", 0.0,
                                      n_features_list=(2,), ks_rotulo_real=[4], verbose=False)
            d = {'X': X, 'y_true': y, 'labels': out['labels'], 'metrics': out['metrics'],
                 'zero_shot_results': out['zero_shot_results']}
            Xb, yb = _dados(n=40, seed=3)
            bc = {'X_b': Xb, 'y_b': yb, 'y_llm_b': (Xb[:, 0] > 0).astype(int),
                  'X_c': Xb, 'y_c': yb, 'y_llm_c': (Xb[:, 0] > 0).astype(int)}
            saved = dm._plot_simbolos_passos(d, "A", 42, "openai/gpt-4o-mini", ("passo3_exemplos_rotulo_real", 4, 0.9), bc, tmp)
            nomes = sorted(os.path.basename(f) for f in saved)
            self.assertEqual(nomes, ["bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42__gpt4mini_gray.png",
                                     "bloco1_graphic_11_B_passos_2_3_simbolos_seed42__gpt4mini_gray.png",
                                     "bloco1_graphic_11_C_passos_2_3_simbolos_seed42__gpt4mini_gray.png"])
            self.assertTrue(all(os.path.exists(os.path.join(tmp, n)) for n in nomes))
            self.assertTrue(os.path.exists(os.path.join(tmp, "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo3_llm_4shot__gpt4mini_gray.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "bloco1_graphic_11_B_passos_2_3_simbolos_seed42_passo3_metrica_eleita__gpt4mini_gray.png")))
            self.assertTrue(all(os.path.exists(os.path.join(tmp, n.replace("_gray.png", "_color.png"))) for n in nomes))
            out_d = dm.adg_passos_2_a_5(X, y, "A", "B", "D", 42, "openrouter", "openai/gpt-4o-mini", 0.0,
                                        n_features_list=(2, 4), ks_rotulo_real=[4], verbose=False)
            dd = {'X': X, 'y_true': y, 'labels': out_d['labels'], 'metrics': out_d['metrics'],
                  'zero_shot_results': out_d['zero_shot_results']}
            saved_d = dm._plot_simbolos_passos(dd, "D", 42, "openai/gpt-4o-mini", None, None, tmp)
            self.assertEqual([os.path.basename(f) for f in saved_d], ["bloco1_graphic_13_D_passos_2_a_4_simbolos_seed42__gpt4mini_gray.png"])
            self.assertTrue(os.path.exists(os.path.join(tmp, "bloco1_graphic_13_D_passos_2_a_4_simbolos_seed42_passo4_metrica_r4__gpt4mini_gray.png")))
            # F13: o D também ganha o painel "LLM k-shot" (menor k disponível = 4)
            self.assertTrue(os.path.exists(os.path.join(tmp, "bloco1_graphic_13_D_passos_2_a_4_simbolos_seed42_passo3_llm_4shot__gpt4mini_gray.png")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
