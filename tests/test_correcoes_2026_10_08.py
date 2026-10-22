"""Correções da revisão de código de 08/10/2026 (revisao_codigo_2026-10-08.md).

Cobre: atributos extras nos exemplos few-shot (B1), prompt de retentativa com o mesmo
estímulo (I1), raw_response = resposta válida (I2), sorteio seed+rep igual entre
variantes (B2), zero-shot externo sem atributos extras (I5), parser com artigo "a" (B4),
402 só por status (B5), flip da auditoria por rótulo (I3) e colapso marcado (I4).
"""
import asyncio
import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402
import llm_client as lc  # noqa: E402
from llm_parser import parse_llm_response  # noqa: E402
from audit_interactions import _stats_basicas  # noqa: E402


class TestPromptComExemplosAtributosExtras(unittest.TestCase):
    def test_exemplo_r3_mostra_x1x2_com_o_mesmo_nome_do_ponto(self):
        ex = [(0.5, -0.5, -0.25, "A"), (1.0, 2.0, 2.0, "B")]
        p = dm.prompt_com_exemplos(0.1, 0.2, ex, "A", "B", "x1", "x2",
                                     extra_features=[("x1*x2", 0.02)])
        self.assertIn("x1 = 0.5000, x2 = -0.5000, x1*x2 = -0.2500 -> A", p)
        self.assertIn("x1*x2 = 0.0200", p)              # ponto a classificar

    def test_exemplo_r4_mostra_os_dois_quadrados_na_ordem(self):
        ex = [(0.5, -0.5, 0.25, 0.25, "A")]
        p = dm.prompt_com_exemplos(0.1, 0.2, ex, "A", "B", "peso_kg", "altura_m",
                                     extra_features=[("peso_kg²", 0.01), ("altura_m²", 0.04)])
        self.assertIn("peso_kg = 0.5000, altura_m = -0.5000, peso_kg² = 0.2500, altura_m² = 0.2500 -> A", p)

    def test_variantes_tambem_mostram_extras(self):
        ex = [(0.5, -0.5, 0.25, 0.25, "A")]
        for variant in ("geometric", "cot"):
            p = dm.prompt_com_exemplos_variante(variant, 0.1, 0.2, ex, "A", "B", "x1", "x2",
                                                 extra_features=[("x1²", 0.01), ("x2²", 0.04)])
            self.assertIn("x1² = 0.2500, x2² = 0.2500 -> A", p, variant)
        tab = dm.prompt_com_exemplos_variante("tabular", 0.1, 0.2, ex, "A", "B", "x1", "x2",
                                               extra_features=[("x1²", 0.01), ("x2²", 0.04)])
        self.assertIn("| 0.5000 | -0.5000 | 0.2500 | 0.2500 | A |", tab)


class TestRetentativaDeFormato(unittest.TestCase):
    def setUp(self):
        self._orig = dm.async_llm_classify_point_openai
        self._n = len(dm.LLM_INTERACTIONS)

    def tearDown(self):
        dm.async_llm_classify_point_openai = self._orig
        del dm.LLM_INTERACTIONS[self._n:]

    def test_retry_mantem_estimulo_e_registra_resposta_valida(self):
        prompts, respostas = [], iter(["not sure", "Homem"])

        async def dublê(client, model, prompt, temp, system_msg, **kw):
            prompts.append(prompt)
            return next(respostas), {"model_resolved": "m", "inference_provider": "p"}
        dm.async_llm_classify_point_openai = dublê
        label, raw, malformed = asyncio.run(dm.async_llm_classify_point(
            63.2, 1.71, "Mulher", "Homem", examples=[(80.1, 1.6, "Mulher")],
            nome_feature_0="peso_kg", nome_feature_1="altura_m"))
        self.assertEqual((label, raw, malformed), ("Homem", "Homem", False))
        self.assertEqual(len(prompts), 2)
        self.assertIn("peso_kg = 63.2000", prompts[1])      # mesmo estímulo no retry
        self.assertIn("peso_kg = 80.1000, altura_m = 1.6000 -> Mulher", prompts[1])
        self.assertIn("IMPORTANT: Your response must be EXACTLY", prompts[1])
        rec = dm.LLM_INTERACTIONS[-1]
        self.assertEqual(rec["raw_response"], "Homem")     # resposta de onde saiu o rótulo
        self.assertEqual(rec["all_responses"], ["not sure", "Homem"])
        self.assertFalse(rec["malformed"])


class TestSorteioExternoSeedRep(unittest.TestCase):
    """Mesmo (seed, rep) → mesmos exemplos nas duas variantes de atributo; zero-shot sem extras."""

    def setUp(self):
        import data_problems as dp
        self.X, self.y = dp.create_problem_homem_mulher(
            csv_path=os.path.join(SRC_DIR, "..", "dados_reais", "homem_mulher", "peso_altura.csv"))
        self.chamadas = []
        self._orig = dm.collect_llm_decisions

        def dublê(X, c0, c1, examples=None, extra_features_matrix=None, **kw):
            self.chamadas.append((kw.get("nome_feature_0"), examples, extra_features_matrix))
            idx = [np.argmin(np.abs(self.X - p).sum(axis=1)) for p in np.asarray(X)[:, :2]]
            return self.y[idx].copy(), 0
        dm.collect_llm_decisions = dublê

    def tearDown(self):
        dm.collect_llm_decisions = self._orig

    def test_exemplos_iguais_entre_variantes_e_zero_shot_sem_extras(self):
        dm.fg_executar(
                problem_name="G1_homem_mulher", X=self.X, y_true=self.y,
                nome_classe_0="Mulher", nome_classe_1="Homem",
                feature_variants=[("x1", "x2"), ("peso", "altura")], seeds=[42],
                ks_aprendiz=[0, 4], verbose=False)
        por_variante = {}
        for feat0, ex, extras in self.chamadas:
            if ex is not None:
                por_variante.setdefault(feat0, []).append([tuple(e[:2]) for e in ex])
            else:
                self.assertIsNone(extras)      # zero-shot: só x1, x2
        self.assertEqual(por_variante["x1"], por_variante["peso"])   # mesmos pontos, mesma ordem


class TestParserArtigoIngles(unittest.TestCase):
    def test_artigo_a_nao_vira_classe_a(self):
        self.assertEqual(parse_llm_response("This is a borderline point", "A", "B"), (None, False))
        self.assertEqual(parse_llm_response("I think it is a point near the boundary.\nB", "A", "B"),
                         ("B", True))

    def test_maiusculas_continuam_valendo(self):
        self.assertEqual(parse_llm_response("The class is A", "A", "B"), ("A", True))
        self.assertEqual(parse_llm_response("Class B", "A", "B"), ("B", True))
        self.assertEqual(parse_llm_response("acho que é Positivo", "Positivo", "Negativo"), ("Positivo", True))


class TestErroDeCreditoPorStatus(unittest.TestCase):
    def test_402_em_request_id_ou_tokens_nao_e_credito(self):
        for msg in ("Error code: 429 - rate limit; request_id=req_a402b",
                    "max_tokens 4020 exceeds", "Error code: 400 - context length 14020"):
            self.assertFalse(lc.erro_de_credito(Exception(msg)), msg)

    def test_402_como_status_e_credito(self):
        class E(Exception):
            status_code = 402
        self.assertTrue(lc.erro_de_credito(E("x")))
        self.assertTrue(lc.erro_de_credito(Exception("Error code: 402 - {'message': 'oops'}")))


class TestAuditoriaFlipPorRotulo(unittest.TestCase):
    def test_formatacao_diferente_nao_e_flip(self):
        base = {"point": "p", "prompt": "P", "malformed": "False", "temperature": "0.0", "format_retries": "0"}
        recs = [dict(base, raw_response="A", parsed_label="A"),
                dict(base, raw_response="A.", parsed_label="A")]
        s = _stats_basicas(recs)
        self.assertEqual(s["taxa_flip_T0"], 0.0)
        self.assertEqual(s["taxa_flip_raw_T0"], 1.0)


class TestColapsoMarcado(unittest.TestCase):
    def setUp(self):
        self._orig = dm.a_passo2_zero_shot_e_metrica

    def tearDown(self):
        dm.a_passo2_zero_shot_e_metrica = self._orig

    def test_colapso_vira_nan_e_collapsed_1_sem_sentinelas(self):
        import data_problems as dp
        X_a, y_a = dp.create_problem_a(n_samples=40, random_state=1)
        X_b, y_b = dp.create_problem_b(n_samples=20, random_state=1)
        X_c, y_c = dp.create_problem_c(n_samples=20, random_state=1)
        y_unico = np.zeros(len(X_a), dtype=int)
        # dublê: o LLM colapsou em classe única no Problema A, Passo 2 (sem chamadas à API)
        dm.a_passo2_zero_shot_e_metrica = lambda *a, **k: (None, None, y_unico, float('nan'), float('nan'), 0.5, 0)
        result, metric, y_llm, fid, acc, n_mal, detail = dm.abc_executar(
            X_a, y_a, X_b, y_b, X_c, y_c, k_exemplos=0, nome_classe_0="A", nome_classe_1="B",
            repeticao=0, provider="openrouter", model_name="m", temperature=0.0, random_seed=1, verbose=False)
        self.assertEqual(result.collapsed, 1)
        self.assertIsNone(metric)
        for v in (result.fidelidade_problema_a, result.consistencia_problema_b, result.kappa_problema_c, fid):
            self.assertTrue(np.isnan(v))          # nunca 0,5 / 0,0 como se fossem medições
        self.assertTrue(np.isnan(result.w_aprendido).all())
        # com os rótulos em cache (colapso), não há nova coleta do Problema A
        chamadas = []
        dm.a_passo2_zero_shot_e_metrica = lambda *a, **k: chamadas.append(1)
        result2, *_ = dm.abc_executar(
            X_a, y_a, X_b, y_b, X_c, y_c, k_exemplos=0, nome_classe_0="A", nome_classe_1="B",
            repeticao=0, provider="openrouter", model_name="m", temperature=0.0, random_seed=1, verbose=False,
            learned_metric_cache=None, y_llm_train_a_cache=y_unico, fidelity_cache=float('nan'),
            llm_accuracy_a_cache=0.5, n_malformed_a_cache=0)
        self.assertEqual(chamadas, [])
        self.assertEqual(result2.collapsed, 1)


if __name__ == "__main__":
    unittest.main()
