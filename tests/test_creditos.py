"""Guarda de saldo do OpenRouter e abort em 402 (08/10/2026).

Sem rede: a consulta ao OpenRouter é substituída por dicts; a chamada à API é
substituída por um dublê que levanta o erro desejado.
"""
import asyncio
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402
import llm_client as lc  # noqa: E402


class TestClassificacaoDoErro(unittest.TestCase):
    def test_402_e_orcamento_sao_erro_de_credito(self):
        for msg in ("Error code: 402 - {'error': {'message': 'Insufficient credits'}}",
                    "in_flight_budget_exhausted", "weight_exceeds_budget", "Payment Required"):
            self.assertTrue(lc.erro_de_credito(Exception(msg)), msg)

    def test_429_500_timeout_nao_sao_erro_de_credito(self):
        for msg in ("Error code: 429 - rate_limit", "503 service overloaded", "Request timeout",
                    "Error code: 400 - could not parse"):
            self.assertFalse(lc.erro_de_credito(Exception(msg)), msg)

    def test_parse_creditos(self):
        info = lc.parse_creditos({"data": {"total_credits": 890, "total_usage": 881.74}})
        self.assertAlmostEqual(info["saldo"], 8.26, places=2)
        self.assertEqual(info["total"], 890.0)


class TestEstimativaEVerificacao(unittest.TestCase):
    def test_estimativa_por_modo(self):
        self.assertEqual(dm.estimar_custo_usd(None, False), dm.CUSTO_ESTIMADO_USD["completa"])
        self.assertEqual(dm.estimar_custo_usd(["bloco3"], False), dm.CUSTO_ESTIMADO_USD["bloco3"])
        self.assertEqual(dm.estimar_custo_usd(["bloco2", "bloco3"], False),
                         dm.CUSTO_ESTIMADO_USD["bloco2"] + dm.CUSTO_ESTIMADO_USD["bloco3"])
        self.assertEqual(dm.estimar_custo_usd(["bloco1"], True), dm.CUSTO_ESTIMADO_SMOKE_USD)
        self.assertEqual(dm.estimar_custo_usd(["oraculo"], False), 0.0)

    def test_saldo_insuficiente_aborta(self):
        with self.assertRaises(SystemExit) as cm:
            dm.verificar_saldo({"total": 890.0, "usado": 881.74, "saldo": 8.26}, estimativa=10.0)
        self.assertIn("saldo insuficiente", str(cm.exception))

    def test_saldo_suficiente_passa(self):
        linha = dm.verificar_saldo({"total": 100.0, "usado": 50.0, "saldo": 50.0}, estimativa=10.0)
        self.assertIn("US$ 50.00", linha)

    def test_ignorar_saldo_nao_aborta(self):
        linha = dm.verificar_saldo({"total": 10.0, "usado": 9.0, "saldo": 1.0}, estimativa=10.0, ignorar=True)
        self.assertIn("--ignorar-saldo", linha)

    def test_saldo_desconhecido_so_avisa(self):
        self.assertIn("desconhecido", dm.verificar_saldo(None, estimativa=10.0))


class TestAbortNo402(unittest.TestCase):
    def setUp(self):
        self._orig_api = dm.async_llm_classify_point_openai
        self._orig_ctx = dict(dm._ABORT_CONTEXT)
        self._orig_client = dm.async_client
        dm.async_client = object()   # injetado: a coleta não cria cliente real (09/10/2026)

    def tearDown(self):
        dm.async_llm_classify_point_openai = self._orig_api
        dm.async_client = self._orig_client
        dm._ABORT_CONTEXT.clear(); dm._ABORT_CONTEXT.update(self._orig_ctx)

    def test_402_vira_creditos_esgotados_sem_retry(self):
        chamadas = []

        async def dublê_402(*a, **k):
            chamadas.append(1)
            raise Exception("Error code: 402 - {'error': {'message': 'Insufficient credits'}}")
        dm.async_llm_classify_point_openai = dublê_402
        with self.assertRaises(lc.CreditosEsgotadosError):
            asyncio.run(dm.async_llm_classify_point(0.1, 0.2, "A", "B"))
        self.assertEqual(len(chamadas), 1)   # fatal: nenhuma nova tentativa

    def test_coletor_aborta_com_checkpoint_e_sem_fallback(self):
        async def dublê_402(*a, **k):
            raise Exception("Error code: 402 - Insufficient credits")
        dm.async_llm_classify_point_openai = dublê_402
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            dm._ABORT_CONTEXT.clear(); dm._ABORT_CONTEXT.update(pasta=tmp)
            n_antes = len(dm.LLM_INTERACTIONS)
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                dm.collect_llm_decisions(np.zeros((3, 2)), "A", "B", verbose=False)
            self.assertEqual(cm.exception.code, 3)
            # nenhum rótulo de fallback MD5 foi registrado
            self.assertEqual(len(dm.LLM_INTERACTIONS), n_antes)


if __name__ == "__main__":
    unittest.main()
