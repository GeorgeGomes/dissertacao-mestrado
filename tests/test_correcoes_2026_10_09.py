"""Correções da revisão de código de 09/10/2026 (revisao_codigo_2026-10-09.md), parte do runner.

Cobre: B1 (especialista do F = rótulo real, Passo 3 independente do Passo 2), B3
(`bloco_protegido` registra a exceção e deixa o fluxo seguir; SystemExit e 402 passam),
I13 (`colapsou` com piso da classe minoritária), M6 (colapso em B/C → NaN), I12 (NNLS
pareado no detail), I15 (mixed 50/50), I16 (zero-shot do Problema E reaproveitado), I17
(sorteio balanceado k/2 por classe), M13/M2 (colunas `*_llm_vs_expert`, F1-Macro,
`example_indices`, `especialista_origem`), P4 (timeout retentável, 400 só transitório, falha
definitiva com prompt real e format_retries=0, custo 0 do oráculo no smoke), P5 (tag
zero-shot do Problema E), F2 (rótulo do diagnóstico de γ), H4 calculado dos dados e a
remoção de `scope`/euclidiana.

Nenhuma chamada à API: dublês de `async_llm_classify_point_openai` e de
`collect_llm_decisions`.
"""
import asyncio
import io
import os
import re
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402
import data_problems as dp  # noqa: E402
import llm_client as lc  # noqa: E402
from relatorios import texto_h4  # noqa: E402

META = {"model_resolved": "m", "inference_provider": "p"}
_RE_X1 = re.compile(r"x1\s*=\s*(-?\d+\.\d+)")


class _FakeOpenAI:
    """Roteiro de respostas/exceções; o último item se repete."""

    def __init__(self, roteiro):
        self.roteiro, self.calls = list(roteiro), 0

    async def __call__(self, async_client, model_name, prompt, temperature, system_msg, **kw):
        item = self.roteiro[min(self.calls, len(self.roteiro) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item, dict(META)


class _Base(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(dm, k) for k in ("INITIAL_BACKOFF", "async_llm_classify_point_openai",
                                                    "async_client", "MODEL_NAME", "CURRENT_PROVIDER",
                                                    "CURRENT_TEMPERATURE", "API_TIMEOUT_S", "N_REPETICOES")}
        self._inter = list(dm.LLM_INTERACTIONS)
        self._diag = list(dm.PERCEPTRON_GAMMA_DIAGNOSTICS)
        dm.INITIAL_BACKOFF = 0
        dm.async_client = object()
        dm.MODEL_NAME = "modelo-teste"
        dm.CURRENT_PROVIDER = "openrouter"
        dm.CURRENT_TEMPERATURE = 0.0
        dm.LLM_INTERACTIONS.clear()
        dm.PERCEPTRON_GAMMA_DIAGNOSTICS.clear()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(dm, k, v)
        dm.LLM_INTERACTIONS.clear(); dm.LLM_INTERACTIONS.extend(self._inter)
        dm.PERCEPTRON_GAMMA_DIAGNOSTICS.clear(); dm.PERCEPTRON_GAMMA_DIAGNOSTICS.extend(self._diag)


# ───────────────────────── B3: bloco_protegido ─────────────────────────
class TestBlocoProtegido(unittest.TestCase):
    def test_excecao_e_registrada_e_o_fluxo_continua(self):
        falhas = []
        depois = []
        with redirect_stdout(io.StringIO()):
            with dm.bloco_protegido("Bloco X seed=1", falhas):
                raise ValueError("boom")
            depois.append(1)          # chegou aqui: a exceção foi suprimida
        self.assertEqual(depois, [1])
        self.assertEqual(len(falhas), 1)
        self.assertIn("Bloco X seed=1", falhas[0]); self.assertIn("ValueError", falhas[0]); self.assertIn("boom", falhas[0])

    def test_sem_excecao_nao_registra(self):
        falhas = []
        with dm.bloco_protegido("ok", falhas):
            pass
        self.assertEqual(falhas, [])

    def test_system_exit_e_creditos_propagam(self):
        falhas = []
        with self.assertRaises(SystemExit):
            with dm.bloco_protegido("x", falhas):
                raise SystemExit(3)
        with self.assertRaises(lc.CreditosEsgotadosError):
            with dm.bloco_protegido("x", falhas):
                raise lc.CreditosEsgotadosError("402")
        with self.assertRaises(KeyboardInterrupt):
            with dm.bloco_protegido("x", falhas):
                raise KeyboardInterrupt()
        self.assertEqual(falhas, [])

    def test_grava_checkpoint_na_falha(self):
        import tempfile
        falhas = []
        n_antes = len(dm.LLM_INTERACTIONS)
        dm.LLM_INTERACTIONS.append({"point": {"x1": 0.0, "x2": 0.0}, "prompt": "p", "parsed_label": "A",
                                    "malformed": False, "tag": "teste"})
        try:
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                with dm.bloco_protegido("x", falhas, tmp):
                    raise RuntimeError("falhou")
                self.assertTrue(os.path.exists(os.path.join(tmp, "llm_interactions_checkpoint.json")))
        finally:
            del dm.LLM_INTERACTIONS[n_antes:]
        self.assertEqual(len(falhas), 1)


# ───────────────────────── P4: timeout / 400 / falha definitiva ─────────────────────────
class TestTimeoutERetentaveis(_Base):
    def test_timeout_vira_excecao_retentavel_e_retenta(self):
        # a chamada ao SDK demora mais que API_TIMEOUT_S: TimeoutError com "timeout" na
        # mensagem → o laço de rate limit re-tenta e sucede na 2ª chamada.
        dm.API_TIMEOUT_S = 0.01
        chamadas = []

        class _Resp:
            model = "m"; provider = "p"

            class _Choice:
                class message:
                    content = "A"
            choices = [_Choice()]

        class _Completions:
            async def create(self, **kw):
                chamadas.append(1)
                if len(chamadas) == 1:
                    await asyncio.sleep(0.2)
                return _Resp()

        class _Client:
            class chat:
                completions = _Completions()
        dm.async_client = _Client()
        label, raw, malformed = asyncio.run(dm.async_llm_classify_point(0.5, -0.5, "A", "B"))
        self.assertEqual((label, malformed), ("A", False))
        self.assertEqual(len(chamadas), 2)
        self.assertEqual(dm._RATE_LIMIT_GATE["until"], dm._RATE_LIMIT_GATE["until"])  # sem exceção

    def test_timed_out_do_sdk_e_retentavel(self):
        # APITimeoutError do SDK da OpenAI: str(e) == "Request timed out." (sem "timeout")
        class APITimeoutError(Exception):
            pass
        self.assertTrue(dm._erro_retentavel(APITimeoutError("Request timed out.")))
        self.assertTrue(dm._erro_retentavel(TimeoutError("timeout de 90 s")))
        self.assertFalse(dm._erro_retentavel(Exception("Error code: 401 - invalid api key")))

    def test_400_so_e_retentavel_com_mensagem_transitoria(self):
        self.assertFalse(dm._erro_retentavel(Exception("Error code: 400 - invalid request: messages[0] is empty")))
        self.assertTrue(dm._erro_retentavel(Exception("Error code: 400 - could not parse response")))
        self.assertTrue(dm._erro_retentavel(Exception("Error code: 400 - Provider returned error")))
        self.assertTrue(dm._erro_retentavel(Exception("Error code: 429 - rate_limit")))
        self.assertTrue(dm._erro_retentavel(Exception("Error code: 503")))
        self.assertTrue(dm._erro_retentavel(TimeoutError("timeout de 90s na chamada à API")))
        self.assertFalse(dm._erro_retentavel(Exception("Error code: 401 - User not found")))

        class E400(Exception):
            status_code = 400
        self.assertFalse(dm._erro_retentavel(E400("invalid")))
        self.assertTrue(dm._erro_retentavel(E400("overloaded")))

    def test_400_nao_transitorio_nao_retenta(self):
        fake = _FakeOpenAI([Exception("Error code: 400 - invalid request")])
        dm.async_llm_classify_point_openai = fake
        with self.assertRaises(Exception):
            asyncio.run(dm.async_llm_classify_point(0.5, -0.5, "A", "B"))
        self.assertEqual(fake.calls, 1)

    def test_falha_definitiva_grava_prompt_real_e_format_retries_zero(self):
        fake = _FakeOpenAI([Exception("Error code: 401 - User not found")])
        dm.async_llm_classify_point_openai = fake
        X = np.array([[0.1234, 0.2]])
        with redirect_stdout(io.StringIO()):
            dm.collect_llm_decisions(X, "A", "B", verbose=False, examples=[(1.0, 2.0, "A")],
                                     nome_feature_0="peso", nome_feature_1="altura")
        rec = dm.LLM_INTERACTIONS[-1]
        self.assertTrue(rec["malformed"]); self.assertIn("401", rec["error"])
        self.assertEqual(rec["format_retries"], 0)                 # erro de API, não de formato
        self.assertIn("peso = 0.1234", rec["prompt"])              # prompt real (chave de flip)
        self.assertIn("peso = 1.0000, altura = 2.0000 -> A", rec["prompt"])


class TestCustoOraculoSmoke(unittest.TestCase):
    def test_oraculo_custa_zero_tambem_no_rapido(self):
        self.assertEqual(dm.estimar_custo_usd(["oraculo"], True), 0.0)
        self.assertEqual(dm.estimar_custo_usd(["oraculo"], False), 0.0)
        self.assertEqual(dm.estimar_custo_usd(["bloco3"], True), dm.CUSTO_ESTIMADO_SMOKE_USD)
        self.assertEqual(dm.estimar_custo_usd(None, True), dm.CUSTO_ESTIMADO_SMOKE_USD)


# ───────────────────────── I13 / M6: colapso ─────────────────────────
class TestColapsoEmBC(_Base):
    def test_bc_passos_2_a_4_colapsado_devolve_nan(self):
        dm.async_llm_classify_point_openai = _FakeOpenAI(["A"])      # colapso em classe única
        X_b, y_b = dp.create_problem_b(n_samples=30, random_state=1)
        lm = dm.LearnedMetric(w=np.array([1.0, 1.0]), centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
                              gamma=0.1, source_problem="Problem_A")
        with redirect_stdout(io.StringIO()):
            cm, y_llm, y_metric, llm_acc, metric_acc, n_mal, extra, X_test = dm.bc_passos_2_a_4(
                X_b, y_b, lm, "A", "B", 0, "[Problema B, Passo 2]", verbose=True)
        self.assertIsNone(cm)
        self.assertEqual(extra["collapsed"], 1)
        self.assertTrue(np.isnan(metric_acc)); self.assertTrue(np.isnan(extra["consistency_centroides_a"]))
        self.assertEqual(len(y_llm), 30)
        self.assertFalse(any(k.startswith("euclid") for k in extra))

    def test_abc_executar_colapso_em_b_marca_collapsed(self):
        # Métrica do A em cache (válida); o LLM colapsa em B e C → linha com collapsed=1 e NaN.
        dm.async_llm_classify_point_openai = _FakeOpenAI(["A"])
        X_a, y_a = dp.create_problem_a(n_samples=40, random_state=1)
        X_b, y_b = dp.create_problem_b(n_samples=20, random_state=1)
        X_c, y_c = dp.create_problem_c(n_samples=20, random_state=1)
        lm = dm.LearnedMetric(w=np.array([1.0, 0.5]), centroids=np.array([[-2.0, 0.0], [2.0, 0.0]]),
                              gamma=0.1, source_problem="Problem_A")
        y_llm_a = (X_a[:, 0] > 0).astype(int)
        with redirect_stdout(io.StringIO()):
            result, *_ , detail = dm.abc_executar(
                X_a, y_a, X_b, y_b, X_c, y_c, k_exemplos=0, nome_classe_0="A", nome_classe_1="B",
                repeticao=0, provider="openrouter", model_name="m", temperature=0.0, random_seed=1,
                learned_metric_cache=lm, learned_metric_nnls_cache=lm, y_llm_train_a_cache=y_llm_a,
                fidelity_cache=0.9, fidelity_nnls_cache=0.9, llm_accuracy_a_cache=0.9,
                n_malformed_a_cache=0, verbose=False)
        # `collapsed` marca só o colapso do A (14/10/2026); B e C ficam em collapsed_b/c
        self.assertEqual((result.collapsed, result.collapsed_b, result.collapsed_c), (0, 1, 1))
        self.assertTrue(np.isnan(result.consistencia_problema_b) and np.isnan(result.kappa_problema_c))
        self.assertTrue(np.isnan(result.consistencia_centroides_a_problema_b))
        self.assertIsNone(detail["metrics_nnls_b"])
        self.assertIsNone(detail["y_metric_b"]); self.assertIsNone(detail["y_metric_c"])   # sem sentinela -1 (13/10/2026)
        self.assertFalse(hasattr(result, "consistencia_euclidiana_problema_b"))
        self.assertFalse(hasattr(result, "diagonal_limitation_flag"))

    def test_abc_executar_sem_colapso_tem_nnls_pareado_e_ablacao(self):
        # I12/I10: detail traz metrics_nnls_b/c pareadas e a ablação com centróides do A vai ao resultado.
        fake = _FakeOpenAI(["A"])

        async def por_x1(async_client, model_name, prompt, temperature, system_msg, **kw):
            fake.calls += 1
            return ("A" if float(_RE_X1.findall(prompt)[-1]) < 0 else "B"), dict(META)
        dm.async_llm_classify_point_openai = por_x1
        X_a, y_a = dp.create_problem_a(n_samples=40, random_state=1)
        X_b, y_b = dp.create_problem_b(n_samples=30, random_state=1)
        X_c, y_c = dp.create_problem_c(n_samples=30, random_state=1)
        with redirect_stdout(io.StringIO()):
            result, lm, y_llm_a, fid, acc, n_mal, detail = dm.abc_executar(
                X_a, y_a, X_b, y_b, X_c, y_c, k_exemplos=0, nome_classe_0="A", nome_classe_1="B",
                repeticao=0, provider="openrouter", model_name="m", temperature=0.0, random_seed=1,
                verbose=False)
        self.assertEqual(result.collapsed, 0)
        self.assertIsNotNone(detail["metrics_nnls_b"]); self.assertIsNotNone(detail["metrics_nnls_c"])
        self.assertTrue(0 <= detail["metric_accuracy_nnls_b"] <= 1)
        self.assertTrue(0 <= result.consistencia_centroides_a_problema_b <= 1)
        self.assertTrue(-1 <= result.kappa_centroides_a_problema_c <= 1)
        self.assertEqual(fake.calls, 40 + 30 + 30)
        # F2: o diagnóstico de γ identifica a configuração e marca a base
        self.assertEqual(len(dm.PERCEPTRON_GAMMA_DIAGNOSTICS), 1)
        d = dm.PERCEPTRON_GAMMA_DIAGNOSTICS[0]
        self.assertEqual(d["label"], "seed 1 | A/B | default [base]"); self.assertTrue(d["base"])


# ───────────────────────── I15 / I16 / P6: Problema E ─────────────────────────
class TestEPassos2e3(_Base):
    def test_zero_shot_reaproveitado_e_medidas_do_especialista_sobre_o_teste(self):
        fake = _FakeOpenAI(["A"])
        dm.async_llm_classify_point_openai = fake
        X_e, y_e = dp.create_problem_e_expert(n_samples=40, random_state=1)
        y_zero = (X_e[:, 1] > 0).astype(int)
        with redirect_stdout(io.StringIO()):
            r1 = dm.e_passos_2_3_aprendiz(X_e, y_e, dm.EXPERT_CONFIGS[0]["w"], dm.EXPERT_CENTROIDS, 0, "easy",
                                           "A", "B", "openrouter", "m", 0.0, 1, 0, verbose=False,
                                           y_llm_cached=y_zero, n_malformed_cached=0)
            r2 = dm.e_passos_2_3_aprendiz(X_e, y_e, dm.EXPERT_CONFIGS[1]["w"], dm.EXPERT_CENTROIDS, 0, "easy",
                                           "A", "B", "openrouter", "m", 0.0, 1, 0, verbose=False,
                                           y_llm_cached=y_zero, n_malformed_cached=0)
        self.assertEqual(fake.calls, 0)                                  # I16: nenhuma coleta
        self.assertEqual((r1.n_classe_0_llm, r1.n_classe_1_llm), (int(np.sum(y_zero == 0)), int(np.sum(y_zero == 1))))
        self.assertNotEqual(r1.accuracy_llm_vs_expert, r2.accuracy_llm_vs_expert)  # métricas por especialista
        self.assertEqual(r1.n_total_test, 40)
        # P6: contagens e acurácia do especialista sobre o teste (= todos os 40 em zero-shot)
        self.assertEqual(r1.n_classe_0_expert + r1.n_classe_1_expert, 40)
        with redirect_stdout(io.StringIO()):
            r3 = dm.e_passos_2_3_aprendiz(X_e, y_e, dm.EXPERT_CONFIGS[0]["w"], dm.EXPERT_CENTROIDS, 4, "easy",
                                           "A", "B", "openrouter", "m", 0.0, 1, 0, verbose=False,
                                           y_llm_cached=y_zero)        # few-shot ignora o cache
        self.assertEqual(fake.calls, 36)
        self.assertEqual(r3.n_classe_0_expert + r3.n_classe_1_expert, 36)
        self.assertEqual(r3.n_total_test, 36)


# ───────────────────────── B1 / I17 / M13: Problemas F e G (split 70/30) ─────────────────────────
class TestFGEspecialista(unittest.TestCase):
    def setUp(self):
        self.X, self.y = dp.create_problem_d_meialua(n_samples=60, random_state=3)
        self.chamadas = []
        self._orig = dm.collect_llm_decisions

    def tearDown(self):
        dm.collect_llm_decisions = self._orig

    def _dublê(self, resposta):
        def f(X, c0, c1, examples=None, **kw):
            self.chamadas.append((examples, kw.get("tag")))
            return resposta(np.asarray(X)), 0
        return f

    def test_f_com_rotulo_real_roda_passo3_mesmo_com_colapso_no_passo2(self):
        # zero-shot colapsa (tudo classe 0); few-shot responde pelo rótulo real.
        def resposta(X):
            if self.chamadas and self.chamadas[-1][0] is None:
                return np.zeros(len(X), int)
            idx = [np.argmin(np.abs(self.X - p).sum(axis=1)) for p in X[:, :2]]
            return self.y[idx].copy()
        dm.collect_llm_decisions = self._dublê(resposta)
        with redirect_stdout(io.StringIO()):
            out = dm.fg_executar(
                problem_name="F_meia_lua_seed3", X=self.X, y_true=self.y, nome_classe_0="A", nome_classe_1="B",
                feature_variants=[("x1", "x2")], seeds=[3], ks_aprendiz=[0, 4], verbose=True,
                expert_labels=self.y)
        self.assertEqual(out["passo2"], [])                  # Passo 2 colapsou
        e = out["aprendiz"]
        self.assertTrue(e)                                             # ... e o Passo 3 rodou (B1)
        self.assertTrue(all(r["especialista_origem"] == "rotulo_real" and r["n_features"] == 2 for r in e))
        for col in ("accuracy_llm_vs_expert", "kappa_llm_vs_expert", "f1_llm_vs_expert", "example_indices"):
            self.assertIn(col, e[0])
        self.assertFalse(any("accuracy_llm_vs_metric" in r for r in e))
        few = [r for r in e if r["k"] == 4]
        self.assertTrue(all(len(r["example_indices"].split(";")) == 4 for r in few))
        # exemplos rotulados pelo rótulo real, k/2 por classe (I17)
        for ex, _tag in self.chamadas:
            if ex:
                self.assertEqual(sum(1 for x in ex if x[-1] == "A"), 2)
                self.assertEqual(sum(1 for x in ex if x[-1] == "B"), 2)
                for x in ex:
                    i = int(np.argmin(np.abs(self.X - np.array(x[:2])).sum(axis=1)))
                    self.assertEqual(x[-1], "A" if self.y[i] == 0 else "B")
        # LLM = rótulo real no few-shot → acurácia (LLM vs especialista) = 1 e F1-Macro = 1
        self.assertTrue(all(abs(r["accuracy_llm_vs_expert"] - 1.0) < 1e-9 and abs(r["f1_llm_vs_expert"] - 1.0) < 1e-9 for r in few))
        self.assertTrue(all(r["accuracy_perceptron_baseline_vs_true"] is not None for r in few))
        mapa = out["llm_label_maps"][("F_meia_lua_seed3", 3, "x1", "x2", 4)]
        self.assertEqual(mapa["especialista_origem"], "rotulo_real")
        np.testing.assert_array_equal(mapa["y_expert"], mapa["y_true"])

    def test_g_sem_expert_labels_usa_metrica_e_pula_passo6_no_colapso(self):
        dm.collect_llm_decisions = self._dublê(lambda X: np.zeros(len(X), int))
        with redirect_stdout(io.StringIO()):
            out = dm.fg_executar(
                problem_name="G1_homem_mulher", X=self.X, y_true=self.y, nome_classe_0="Mulher", nome_classe_1="Homem",
                feature_variants=[("x1", "x2")], seeds=[3], ks_aprendiz=[0, 4], verbose=False)
        self.assertEqual(out["aprendiz"], [])
        dm.collect_llm_decisions = self._dublê(
            lambda X: self.y[[np.argmin(np.abs(self.X - p).sum(axis=1)) for p in X[:, :2]]].copy())
        with redirect_stdout(io.StringIO()):
            out = dm.fg_executar(
                problem_name="G1_homem_mulher", X=self.X, y_true=self.y, nome_classe_0="Mulher", nome_classe_1="Homem",
                feature_variants=[("x1", "x2")], seeds=[3], ks_aprendiz=[0, 4], verbose=False)
        self.assertTrue(all(r["especialista_origem"] == "metrica" for r in out["aprendiz"]))

    def test_sorteio_balanceado_reprodutivel_e_completa_classe_escassa(self):
        y = np.array([0] * 20 + [1] * 20)
        a = dm.fg_exemplos_aprendiz(y, 10, 42)
        b = dm.fg_exemplos_aprendiz(y, 10, 42)
        c = dm.fg_exemplos_aprendiz(y, 10, 43)
        self.assertEqual(a.tolist(), b.tolist()); self.assertNotEqual(a.tolist(), c.tolist())
        self.assertEqual(len(set(a.tolist())), 10)
        self.assertEqual(int(np.sum(y[a] == 0)), 5); self.assertEqual(int(np.sum(y[a] == 1)), 5)
        y_esc = np.array([0] * 30 + [1] * 2)
        with redirect_stdout(io.StringIO()) as buf:
            d = dm.fg_exemplos_aprendiz(y_esc, 10, 1)
        self.assertEqual(len(set(d.tolist())), 10)
        self.assertEqual(int(np.sum(y_esc[d] == 1)), 2)             # os 2 disponíveis + 8 da outra
        self.assertIn("completado com a outra classe", buf.getvalue())


# ───────────────────────── relatórios ─────────────────────────
class TestTextoH4(unittest.TestCase):
    def _df(self, easy, hard):
        rows = []
        for n in (4, 10):
            rows.append({"strategy": "easy", "k": n, "accuracy": easy[n]})
            rows.append({"strategy": "hard", "k": n, "accuracy": hard[n]})
            rows.append({"strategy": "random", "k": n, "accuracy": 0.5})
        return pd.DataFrame(rows)

    def test_veredicto_vem_dos_dados(self):
        t = texto_h4(self._df({4: .9, 10: .9}, {4: .7, 10: .7}))
        self.assertIn("H4 REFUTADA", t); self.assertIn("easy (90.0%) supera hard (70.0%)", t)
        t = texto_h4(self._df({4: .7, 10: .7}, {4: .9, 10: .9}))
        self.assertIn("H4 sustentável", t)
        t = texto_h4(self._df({4: .8, 10: .8}, {4: .8, 10: .8}))
        self.assertIn("EMPATE", t)
        # cruzamento: easy vence em média mas hard vence no maior k_exemplos
        t = texto_h4(self._df({4: .95, 10: .70}, {4: .60, 10: .80}))
        self.assertIn("CRUZAMENTO", t); self.assertIn("10-shot: hard", t)
        self.assertNotIn("consistentemente", t)


class TestRemocoes(unittest.TestCase):
    def test_modelos_sao_3_tuplas_sem_scope(self):
        self.assertTrue(all(len(m) == 3 for m in dm.MODELS_TO_TEST))
        self.assertFalse(hasattr(dm, "EXTRA_SEEDS_CORE"))

    def test_sem_grafico_20_no_runner(self):
        self.assertFalse(hasattr(dm, "plot_oracle_transfer"))

    def test_tag_e_passo2_zero_shot(self):
        # P5: o zero-shot único do Problema E é o Passo 2 com regime=passo2_zero_shot (string gerada no runner)
        with open(os.path.join(SRC_DIR, "dissertacao_mestrado.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('_tag("bloco2", "E", 2, "passo2_zero_shot", 0, 0, seed)', src)
        self.assertIn('"vies_nomes_classe",\n', src)
        self.assertIn('extra=f"variante={variant_name}"', src)
        self.assertIn('"vies_nomes_atributo", 0, 0, seed', src)


if __name__ == "__main__":
    unittest.main()


# ───────────────────────── pendências da rodada 2 (09/10/2026, noite) ─────────────────────────
class TestSorteioPeloRotuloRealNoG(unittest.TestCase):
    """Item 2: nos Problemas F e G o sorteio balanceia pelo RÓTULO REAL (k/2 por classe),
    logo os índices são iguais entre variantes de atributo mesmo com especialistas
    (métricas) diferentes; os exemplos continuam rotulados pelo especialista."""

    def setUp(self):
        self.X, self.y = dp.create_problem_a(n_samples=80, random_state=5)
        self.chamadas = []
        self._orig = dm.collect_llm_decisions

        def dublê(X, c0, c1, examples=None, **kw):
            X = np.asarray(X)
            idx = [int(np.argmin(np.abs(self.X - p).sum(axis=1))) for p in X[:, :2]]
            if kw.get("nome_feature_0") == "x1":
                rot = self.y[idx].copy()                     # métrica ≈ rótulo real
            else:
                rot = (X[:, 1] > np.median(self.X[:, 1])).astype(int)   # outra regra: outro especialista
            self.chamadas.append({"variante": kw.get("nome_feature_0"), "examples": examples, "k": 0 if examples is None else len(examples)})
            return rot, 0
        dm.collect_llm_decisions = dublê

    def tearDown(self):
        dm.collect_llm_decisions = self._orig

    def test_indices_iguais_entre_variantes_e_k_meio_por_classe_real(self):
        with redirect_stdout(io.StringIO()) as buf:
            out = dm.fg_executar(
                problem_name="G1_homem_mulher", X=self.X, y_true=self.y, nome_classe_0="Mulher",
                nome_classe_1="Homem", feature_variants=[("x1", "x2"), ("altura", "peso")],
                seeds=[5], ks_aprendiz=[0, 10], verbose=True)
        e = out["aprendiz"]
        self.assertTrue(all(r["especialista_origem"] == "metrica" for r in e))
        few = [r for r in e if r["k"] == 10]
        self.assertTrue(few)
        por_chave = {}
        for r in few:
            por_chave.setdefault((r["seed"], r["k"], r["rep"]), {})[r["feature_names"]] = r["example_indices"]
        rng = np.random.RandomState(5)
        train_idx = rng.permutation(len(self.X))[:int(0.7 * len(self.X))]
        y_train = self.y[train_idx]
        for chave, por_variante in por_chave.items():
            self.assertEqual(set(por_variante), {("x1", "x2"), ("altura", "peso")}, chave)
            a, b = por_variante[("x1", "x2")], por_variante[("altura", "peso")]
            self.assertEqual(a, b, f"índices diferem entre variantes em {chave}")
            idx = [int(i) for i in a.split(";")]
            self.assertEqual(len(idx), 10)
            self.assertEqual(int(np.sum(y_train[idx] == 0)), 5)       # k/2 por classe REAL
            self.assertEqual(int(np.sum(y_train[idx] == 1)), 5)
        # os especialistas das duas variantes são de fato diferentes (rotulam exemplos de outro jeito)
        rot = {}
        for c in self.chamadas:
            if c["examples"]:
                rot.setdefault(c["variante"], []).append(tuple(x[-1] for x in c["examples"]))
        self.assertNotEqual(rot["x1"], rot["altura"])
        # o log registra a regra e a contagem por classe do especialista
        log = buf.getvalue()
        self.assertIn("BALANCEADO PELO RÓTULO REAL", log)
        self.assertIn("por classe do especialista (metrica)", log)
