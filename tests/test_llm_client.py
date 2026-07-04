"""Testes de unidade da fábrica de clientes de LLM (`llm_client`).

Verifica o roteamento de provedor: tipo de cliente correto (OpenAI vs Anthropic),
URL base correta (Gemini via endpoint compatível com OpenAI), ausência de cliente
assíncrono para a Anthropic, e erro para provedor desconhecido.

Não faz nenhuma chamada de rede: construir um cliente SDK apenas configura o objeto;
nenhuma requisição é disparada. Chaves de API dummy são injetadas no ambiente para
que os construtores (que exigem uma chave não-nula) não levantem exceção.

Execução (a partir de src/):
    ../.venv/bin/python -m pytest ../tests/test_llm_client.py -q
"""
import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import llm_client as lc  # noqa: E402
from openai import OpenAI, AsyncOpenAI  # noqa: E402
import anthropic  # noqa: E402

GEMINI_URL = "generativelanguage.googleapis.com"


class TestClientFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Chaves dummy: os construtores da OpenAI/Anthropic exigem api_key não-nula,
        # mas nenhuma chamada de rede é feita ao apenas construir o cliente.
        cls._saved = {}
        for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
                    "OPENROUTER_API_KEY"):
            cls._saved[var] = os.environ.get(var)
            os.environ[var] = "dummy-key-para-testes"

    @classmethod
    def tearDownClass(cls):
        for var, val in cls._saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val

    # --- get_client (síncrono) ---
    def test_openai_retorna_openai_com_url_padrao(self):
        c = lc.get_client("openai")
        self.assertIsInstance(c, OpenAI)
        self.assertIn("api.openai.com", str(c.base_url))

    def test_gemini_usa_endpoint_compativel_openai(self):
        c = lc.get_client("gemini")
        self.assertIsInstance(c, OpenAI)  # Gemini via API compatível com a OpenAI
        self.assertIn(GEMINI_URL, str(c.base_url))

    def test_anthropic_retorna_cliente_anthropic(self):
        c = lc.get_client("anthropic")
        self.assertIsInstance(c, anthropic.Anthropic)

    def test_provedor_desconhecido_levanta(self):
        with self.assertRaises(KeyError):
            lc.get_client("provedor-inexistente")

    def test_openrouter_usa_endpoint_openai_compativel(self):
        c = lc.get_client("openrouter")
        self.assertIsInstance(c, OpenAI)  # OpenRouter via API compatível com a OpenAI
        self.assertIn("openrouter.ai", str(c.base_url))

    # --- get_async_client (assíncrono) ---
    def test_async_openai_retorna_asyncopenai(self):
        c = lc.get_async_client("openai")
        self.assertIsInstance(c, AsyncOpenAI)

    def test_async_gemini_retorna_asyncopenai_com_url(self):
        c = lc.get_async_client("gemini")
        self.assertIsInstance(c, AsyncOpenAI)
        self.assertIn(GEMINI_URL, str(c.base_url))

    def test_async_anthropic_e_none(self):
        # A Anthropic não tem cliente assíncrono neste projeto — a factory deve
        # devolver None para sinalizar "sem caminho assíncrono".
        self.assertIsNone(lc.get_async_client("anthropic"))

    def test_async_provedor_desconhecido_levanta(self):
        with self.assertRaises(KeyError):
            lc.get_async_client("provedor-inexistente")

    def test_async_openrouter_retorna_asyncopenai_com_url(self):
        c = lc.get_async_client("openrouter")
        self.assertIsInstance(c, AsyncOpenAI)
        self.assertIn("openrouter.ai", str(c.base_url))


class TestProviderConfig(unittest.TestCase):
    def test_todos_os_provedores_tem_campos_obrigatorios(self):
        for provider, cfg in lc.PROVIDER_CONFIG.items():
            self.assertIn("base_url", cfg, provider)
            self.assertIn("api_key_env", cfg, provider)
            self.assertIn(cfg["client_type"], ("openai", "anthropic"), provider)

    def test_chaves_de_api_vem_de_variaveis_de_ambiente(self):
        # Corretude de segurança: nenhuma chave é hardcoded — a config só nomeia
        # a variável de ambiente de onde a chave é lida.
        for cfg in lc.PROVIDER_CONFIG.values():
            self.assertTrue(cfg["api_key_env"].endswith("_API_KEY"))


class TestExtraBody(unittest.TestCase):
    def test_openrouter_desliga_fallback_de_provedor(self):
        # Reprodutibilidade: o mesmo model-ID pode ser servido por provedores de
        # inferência diferentes no OpenRouter; o fallback deve vir desligado.
        eb = lc.get_extra_body("openrouter")
        self.assertIn("provider", eb)
        self.assertFalse(eb["provider"].get("allow_fallbacks", True))

    def test_demais_provedores_tem_corpo_extra_vazio(self):
        for provider in ("openai", "gemini", "anthropic"):
            self.assertEqual(lc.get_extra_body(provider), {})

    def test_provedor_desconhecido_devolve_vazio(self):
        self.assertEqual(lc.get_extra_body("qualquer-coisa"), {})

    # --- pinagem de provedor de inferência POR MODELO (MODEL_PROVIDER_PIN) ---
    def test_modelo_pinado_fixa_provedor_de_inferencia(self):
        # Reprodutibilidade: modelos de pesos abertos são servidos por várias
        # infraestruturas (quantizações distintas); o pin fixa quem serve.
        eb = lc.get_extra_body("openrouter", "meta-llama/llama-4-scout")
        self.assertEqual(eb["provider"]["order"], ["DeepInfra"])
        self.assertFalse(eb["provider"]["allow_fallbacks"])

    def test_todos_os_pins_tem_order_e_fallback_desligado(self):
        for model, pin in lc.MODEL_PROVIDER_PIN.items():
            self.assertTrue(pin.get("order"), model)
            self.assertFalse(pin.get("allow_fallbacks", True), model)

    def test_modelo_sem_pin_cai_no_default_do_provedor(self):
        eb = lc.get_extra_body("openrouter", "modelo/sem-pin")
        self.assertEqual(eb, lc.PROVIDER_EXTRA_BODY["openrouter"])

    def test_pin_nao_afeta_provedores_sem_corpo_extra(self):
        # gpt-4o-mini vai direto na OpenAI (sem OpenRouter): corpo extra vazio.
        self.assertEqual(lc.get_extra_body("openai", "gpt-4o-mini"), {})


if __name__ == "__main__":
    unittest.main()
