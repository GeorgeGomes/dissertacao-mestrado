"""Testes do loop de retry/fallback do caminho de chamadas ao LLM.

Este é o caminho mais crítico da coleta (milhares de chamadas na execução paga):
ele decide se um erro derruba a execução, vira retry com backoff ou vira fallback
MD5 auditável. Nenhuma chamada de rede é feita — `async_llm_classify_point_openai`
é substituída por um dublê programável e `INITIAL_BACKOFF` é zerado para que o
backoff exponencial (0 * 2^n = 0) não durma de verdade.

Contratos verificados em `async_llm_classify_point` e `collect_llm_decisions`:

1. Erro RETENTÁVEL (429/5xx/timeout) → re-tenta e, ao suceder, devolve o rótulo
   com `malformed=False` registrado em LLM_INTERACTIONS.
2. Erro NÃO-retentável (ex.: 401) → levanta imediatamente, sem re-tentar.
3. MAX_RETRIES esgotado em erro retentável → levanta (não silencia).
4. Resposta malformada persistente → MAX_FORMAT_RETRIES reenvios, depois fallback
   MD5 determinístico + registro `malformed=True` com todas as respostas.
5. Falha definitiva dentro da coleta (`classify_one`) → NÃO derruba o lote:
   ponto recebe fallback, registro ganha campo "error", e a coleta completa.

Execução: .venv/bin/python -m unittest tests.test_llm_retry_fallback
"""
import asyncio
import hashlib
import os
import sys
import unittest
import warnings

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402

C0, C1 = "A", "B"
META = {"model_resolved": "modelo-teste-2026-01-01", "inference_provider": "TestInfra"}


def _fallback_esperado(x1: float, x2: float) -> str:
    """Reproduz o fallback MD5 do runner (determinístico por coordenadas)."""
    hs = hashlib.md5(f"{x1:.6f}_{x2:.6f}".encode()).hexdigest()
    return C0 if int(hs, 16) % 2 == 0 else C1


class _FakeOpenAI:
    """Dublê de async_llm_classify_point_openai: devolve um roteiro programado.

    Cada item do roteiro é uma Exception (será levantada) ou uma string
    (será devolvida como resposta do "LLM"). O último item se repete
    indefinidamente, para roteiros do tipo "sempre falha".
    """

    def __init__(self, roteiro):
        self.roteiro = list(roteiro)
        self.calls = 0

    async def __call__(self, async_client, model_name, prompt, temperature, system_msg):
        item = self.roteiro[min(self.calls, len(self.roteiro) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item, dict(META)


class TestRetryFallback(unittest.TestCase):
    def setUp(self):
        # Salva e configura o estado global que o caminho de chamada consome.
        self._saved = {
            "INITIAL_BACKOFF": dm.INITIAL_BACKOFF,
            "async_llm_classify_point_openai": dm.async_llm_classify_point_openai,
            "async_client": dm.async_client,
            "MODEL_NAME": dm.MODEL_NAME,
            "CURRENT_PROVIDER": dm.CURRENT_PROVIDER,
            "CURRENT_TEMPERATURE": dm.CURRENT_TEMPERATURE,
            "interactions": list(dm.LLM_INTERACTIONS),
        }
        dm.INITIAL_BACKOFF = 0          # backoff 0 * 2^n = 0: sem espera real
        dm.async_client = object()      # nunca tocado: o dublê intercepta antes
        dm.MODEL_NAME = "modelo-teste"
        dm.CURRENT_PROVIDER = "openai"  # client_type "openai" → caminho async
        dm.CURRENT_TEMPERATURE = 0.0
        dm.LLM_INTERACTIONS.clear()

    def tearDown(self):
        dm.INITIAL_BACKOFF = self._saved["INITIAL_BACKOFF"]
        dm.async_llm_classify_point_openai = self._saved["async_llm_classify_point_openai"]
        dm.async_client = self._saved["async_client"]
        dm.MODEL_NAME = self._saved["MODEL_NAME"]
        dm.CURRENT_PROVIDER = self._saved["CURRENT_PROVIDER"]
        dm.CURRENT_TEMPERATURE = self._saved["CURRENT_TEMPERATURE"]
        dm.LLM_INTERACTIONS.clear()
        dm.LLM_INTERACTIONS.extend(self._saved["interactions"])

    def _classify(self, x1=0.5, x2=-0.5):
        import asyncio
        return asyncio.run(dm.async_llm_classify_point(x1, x2, C0, C1))

    # 1. Erro retentável → re-tenta → sucesso
    def test_erro_retentavel_retenta_e_sucede(self):
        fake = _FakeOpenAI([
            Exception("Error code: 429 - rate_limit exceeded"),
            Exception("upstream connection timeout"),
            C0,
        ])
        dm.async_llm_classify_point_openai = fake

        label, raw, malformed = self._classify()

        self.assertEqual(label, C0)
        self.assertEqual(raw, C0)
        self.assertFalse(malformed)
        self.assertEqual(fake.calls, 3, "2 erros retentáveis + 1 sucesso = 3 chamadas")
        rec = dm.LLM_INTERACTIONS[-1]
        self.assertFalse(rec["malformed"])
        self.assertEqual(rec["format_retries"], 0)
        self.assertEqual(rec["model_resolved"], META["model_resolved"])
        self.assertEqual(rec["inference_provider"], META["inference_provider"])

    # 2. Erro não-retentável → levanta imediatamente
    def test_erro_nao_retentavel_levanta_sem_retentar(self):
        fake = _FakeOpenAI([Exception("Error code: 401 - User not found")])
        dm.async_llm_classify_point_openai = fake

        with self.assertRaises(Exception) as ctx:
            self._classify()
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(fake.calls, 1, "erro não-retentável não pode re-tentar")

    # 3. MAX_RETRIES esgotado → levanta (não silencia)
    def test_max_retries_esgotado_levanta(self):
        fake = _FakeOpenAI([Exception("503 service overloaded")])
        dm.async_llm_classify_point_openai = fake

        with self.assertRaises(Exception) as ctx:
            self._classify()
        self.assertIn("Máximo de tentativas", str(ctx.exception))
        self.assertEqual(fake.calls, dm.MAX_RETRIES)

    # 4. Malformada persistente → reenvios → fallback MD5 registrado
    def test_malformada_persistente_cai_no_fallback_md5(self):
        # "xyzzy" é imparseável pelas 8 camadas para classes A/B (verificado);
        # atenção: respostas com as letras das classes (ex.: "banana") SÃO
        # recuperadas pelas camadas fuzzy do parser e não servem aqui.
        fake = _FakeOpenAI(["xyzzy"])
        dm.async_llm_classify_point_openai = fake
        x1, x2 = 0.5, -0.5

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            label, raw, malformed = self._classify(x1, x2)

        self.assertTrue(malformed)
        self.assertEqual(label, _fallback_esperado(x1, x2))
        self.assertEqual(raw, "xyzzy")
        self.assertEqual(fake.calls, dm.MAX_FORMAT_RETRIES,
                         "deve reenviar exatamente MAX_FORMAT_RETRIES vezes")
        self.assertTrue(any("MALFORMADA" in str(c.message) for c in caught),
                        "fallback não pode ser silencioso")
        rec = dm.LLM_INTERACTIONS[-1]
        self.assertTrue(rec["malformed"])
        self.assertEqual(rec["format_retries"], dm.MAX_FORMAT_RETRIES)
        self.assertEqual(len(rec["all_responses"]), dm.MAX_FORMAT_RETRIES)
        self.assertEqual(rec["parsed_label"], label)

    def test_fallback_md5_determinista_entre_chamadas(self):
        dm.async_llm_classify_point_openai = _FakeOpenAI(["???"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l1, _, _ = self._classify(1.234567, -9.87654)
            l2, _, _ = self._classify(1.234567, -9.87654)
        self.assertEqual(l1, l2, "mesmo ponto deve cair sempre na mesma classe")

    # 5. Falha definitiva no worker → coleta continua, ponto vira fallback + "error"
    def test_falha_definitiva_nao_derruba_a_coleta(self):
        fake = _FakeOpenAI([Exception("Error code: 401 - User not found")])
        dm.async_llm_classify_point_openai = fake
        X = np.array([[0.1, 0.2], [0.3, 0.4]])

        y_llm, n_malformed = dm.collect_llm_decisions(X, C0, C1, verbose=False)

        self.assertEqual(len(y_llm), 2, "todos os pontos devem receber rótulo")
        self.assertEqual(n_malformed, 2)
        esperados = [0 if _fallback_esperado(x1, x2) == C0 else 1 for x1, x2 in X]
        self.assertEqual(y_llm.tolist(), esperados,
                         "fallback dos pontos deve ser o MD5 determinístico")
        registros = dm.LLM_INTERACTIONS[-2:]
        for rec in registros:
            self.assertTrue(rec["malformed"])
            self.assertIn("401", rec["error"])


class TestFreioGlobalRateLimit(unittest.TestCase):
    """Freio global de cooldown: um 429 em QUALQUER worker segura todos os demais.

    Sem o freio, com MAX_CONCURRENCY workers, o que recebe 429 dorme mas os
    outros continuam disparando e realimentam o rate limit do provedor pinado
    (allow_fallbacks=False não deixa o OpenRouter desviar — por design).
    """

    def setUp(self):
        self._saved_until = dm._RATE_LIMIT_GATE["until"]
        dm._RATE_LIMIT_GATE["until"] = 0.0

    def tearDown(self):
        dm._RATE_LIMIT_GATE["until"] = self._saved_until

    def test_acionar_estende_e_nunca_encurta(self):
        dm._acionar_rate_limit_global(30)
        alvo = dm._RATE_LIMIT_GATE["until"]
        self.assertGreater(alvo, 0.0)
        dm._acionar_rate_limit_global(1)      # janela menor NÃO encurta a vigente
        self.assertEqual(dm._RATE_LIMIT_GATE["until"], alvo)
        dm._acionar_rate_limit_global(60)     # janela maior estende
        self.assertGreater(dm._RATE_LIMIT_GATE["until"], alvo)

    def test_respeitar_aguarda_a_janela(self):
        import time as _t
        dm._acionar_rate_limit_global(0.15)
        t0 = _t.monotonic()
        asyncio.run(dm._respeitar_rate_limit_global())
        self.assertGreaterEqual(_t.monotonic() - t0, 0.14,
                                "worker deve aguardar a janela de cooldown")

    def test_sem_janela_nao_espera(self):
        import time as _t
        t0 = _t.monotonic()
        asyncio.run(dm._respeitar_rate_limit_global())
        self.assertLess(_t.monotonic() - t0, 0.05)

    def test_429_de_um_worker_aciona_o_freio_global(self):
        # Integração: um 429 no caminho real de retry deve estender a janela
        # global (INITIAL_BACKOFF pequeno para o teste não dormir de verdade).
        saved = {
            "INITIAL_BACKOFF": dm.INITIAL_BACKOFF,
            "fake": dm.async_llm_classify_point_openai,
            "async_client": dm.async_client,
            "MODEL_NAME": dm.MODEL_NAME,
            "CURRENT_PROVIDER": dm.CURRENT_PROVIDER,
            "CURRENT_TEMPERATURE": dm.CURRENT_TEMPERATURE,
            "interactions": list(dm.LLM_INTERACTIONS),
        }
        try:
            dm.INITIAL_BACKOFF = 0.01
            dm.async_client = object()
            dm.MODEL_NAME = "modelo-teste"
            dm.CURRENT_PROVIDER = "openai"
            dm.CURRENT_TEMPERATURE = 0.0
            dm.LLM_INTERACTIONS.clear()
            dm.async_llm_classify_point_openai = _FakeOpenAI([
                Exception("Error code: 429 - rate_limit exceeded"), C0,
            ])

            label, _, _ = asyncio.run(dm.async_llm_classify_point(0.5, -0.5, C0, C1))

            self.assertEqual(label, C0)
            self.assertGreater(dm._RATE_LIMIT_GATE["until"], 0.0,
                               "429 deve acionar o freio global para os demais workers")
        finally:
            dm.INITIAL_BACKOFF = saved["INITIAL_BACKOFF"]
            dm.async_llm_classify_point_openai = saved["fake"]
            dm.async_client = saved["async_client"]
            dm.MODEL_NAME = saved["MODEL_NAME"]
            dm.CURRENT_PROVIDER = saved["CURRENT_PROVIDER"]
            dm.CURRENT_TEMPERATURE = saved["CURRENT_TEMPERATURE"]
            dm.LLM_INTERACTIONS.clear()
            dm.LLM_INTERACTIONS.extend(saved["interactions"])

    def test_timeout_nao_aciona_o_freio_global(self):
        # Timeout de conexão é problema da chamada, não de capacidade do
        # provedor: re-tenta com backoff próprio, mas NÃO segura os demais.
        saved = {
            "fake": dm.async_llm_classify_point_openai,
            "INITIAL_BACKOFF": dm.INITIAL_BACKOFF,
            "async_client": dm.async_client,
            "MODEL_NAME": dm.MODEL_NAME,
            "CURRENT_PROVIDER": dm.CURRENT_PROVIDER,
            "CURRENT_TEMPERATURE": dm.CURRENT_TEMPERATURE,
            "interactions": list(dm.LLM_INTERACTIONS),
        }
        try:
            dm.INITIAL_BACKOFF = 0.01
            dm.async_client = object()
            dm.MODEL_NAME = "modelo-teste"
            dm.CURRENT_PROVIDER = "openai"
            dm.CURRENT_TEMPERATURE = 0.0
            dm.LLM_INTERACTIONS.clear()
            dm.async_llm_classify_point_openai = _FakeOpenAI([
                Exception("upstream connection timeout"), C0,
            ])
            asyncio.run(dm.async_llm_classify_point(0.5, -0.5, C0, C1))
            self.assertEqual(dm._RATE_LIMIT_GATE["until"], 0.0,
                             "timeout avulso não deve acionar o freio global")
        finally:
            dm.async_llm_classify_point_openai = saved["fake"]
            dm.INITIAL_BACKOFF = saved["INITIAL_BACKOFF"]
            dm.async_client = saved["async_client"]
            dm.MODEL_NAME = saved["MODEL_NAME"]
            dm.CURRENT_PROVIDER = saved["CURRENT_PROVIDER"]
            dm.CURRENT_TEMPERATURE = saved["CURRENT_TEMPERATURE"]
            dm.LLM_INTERACTIONS.clear()
            dm.LLM_INTERACTIONS.extend(saved["interactions"])


if __name__ == "__main__":
    unittest.main()
