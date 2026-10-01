"""Testes de unidade da fábrica de clientes de LLM (`llm_client`).

Provedor único: OpenRouter. Verifica o roteamento (cliente OpenAI-compatível com a
URL do OpenRouter, síncrono e assíncrono), erro para provedor desconhecido, a chave
vinda de variável de ambiente, o corpo extra (fallback de provedor desligado) e a
pinagem de infraestrutura por modelo (`MODEL_PROVIDER_PIN`) para TODOS os modelos
do protocolo.

Não faz nenhuma chamada de rede: construir um cliente SDK apenas configura o objeto.
Uma chave dummy é injetada no ambiente para o construtor não levantar exceção.

Execução: .venv/bin/python -m unittest tests.test_llm_client
"""
import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import llm_client as lc  # noqa: E402
import dissertacao_mestrado as dm  # noqa: E402
from openai import OpenAI, AsyncOpenAI  # noqa: E402


class TestClientFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "dummy-key-para-testes"

    @classmethod
    def tearDownClass(cls):
        if cls._saved is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = cls._saved

    def test_openrouter_usa_endpoint_openai_compativel(self):
        c = lc.get_client("openrouter")
        self.assertIsInstance(c, OpenAI)
        self.assertIn("openrouter.ai", str(c.base_url))

    def test_async_openrouter_retorna_asyncopenai_com_url(self):
        c = lc.get_async_client("openrouter")
        self.assertIsInstance(c, AsyncOpenAI)
        self.assertIn("openrouter.ai", str(c.base_url))

    def test_provedor_desconhecido_levanta(self):
        with self.assertRaises(KeyError):
            lc.get_client("openai")          # provedor direto removido em 01/10/2026
        with self.assertRaises(KeyError):
            lc.get_async_client("anthropic")


class TestProviderConfig(unittest.TestCase):
    def test_provedor_unico_openrouter(self):
        self.assertEqual(list(lc.PROVIDER_CONFIG), ["openrouter"])

    def test_todos_os_provedores_tem_campos_obrigatorios(self):
        for provider, cfg in lc.PROVIDER_CONFIG.items():
            self.assertTrue(cfg["base_url"], provider)
            self.assertTrue(cfg["api_key_env"].endswith("_API_KEY"), provider)
            self.assertEqual(cfg["client_type"], "openai", provider)

    def test_modelos_do_protocolo_usam_openrouter_com_id_qualificado(self):
        for prov, model, _t, _s in dm.MODELS_TO_TEST:
            self.assertEqual(prov, "openrouter", model)
            self.assertIn("/", model, f"{model}: id do OpenRouter é <org>/<modelo>")


class TestExtraBody(unittest.TestCase):
    def test_openrouter_desliga_fallback_de_provedor(self):
        eb = lc.get_extra_body("openrouter")
        self.assertFalse(eb["provider"].get("allow_fallbacks", True))

    def test_provedor_desconhecido_devolve_vazio(self):
        self.assertEqual(lc.get_extra_body("qualquer-coisa"), {})

    def test_todos_os_modelos_do_protocolo_tem_pin(self):
        # Reprodutibilidade: um mesmo model-ID pode ser servido por várias
        # infraestruturas no OpenRouter; todo modelo do protocolo fixa quem serve.
        for _p, model, _t, _s in dm.MODELS_TO_TEST:
            self.assertIn(model, lc.MODEL_PROVIDER_PIN, model)

    def test_pin_fixa_provedor_de_inferencia(self):
        eb = lc.get_extra_body("openrouter", "google/gemini-2.5-flash-lite")
        self.assertEqual(eb["provider"]["order"], ["Google"])
        self.assertFalse(eb["provider"]["allow_fallbacks"])
        eb = lc.get_extra_body("openrouter", "openai/gpt-4o-mini")
        self.assertEqual(eb["provider"]["order"], ["OpenAI"])

    def test_todos_os_pins_tem_order_e_fallback_desligado(self):
        for model, pin in lc.MODEL_PROVIDER_PIN.items():
            self.assertTrue(pin.get("order"), model)
            self.assertFalse(pin.get("allow_fallbacks", True), model)

    def test_modelo_sem_pin_cai_no_default_do_provedor(self):
        eb = lc.get_extra_body("openrouter", "modelo/sem-pin")
        self.assertEqual(eb, lc.PROVIDER_EXTRA_BODY["openrouter"])


if __name__ == "__main__":
    unittest.main()
