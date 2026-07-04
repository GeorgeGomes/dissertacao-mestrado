"""Fábrica de clientes de API de LLM por provedor (OpenAI, Gemini, Anthropic).

Isola a criação dos clientes SDK — registro de provedores + factories — do runner
principal. Cada factory lê a chave de API da variável de ambiente configurada em
``PROVIDER_CONFIG`` (nunca hardcoded) e devolve o cliente já configurado.

Extraído de ``dissertacao_mestrado.py`` para permitir testes de unidade do roteamento
de provedor (base_url, tipo de cliente, provedor inexistente) sem tocar no runner —
ver ``tests/test_llm_client.py``. Não carrega o ``.env``: quem consome (o runner
principal) é responsável por chamar ``load_dotenv()`` antes de usar as factories; as
chaves são lidas via ``os.getenv`` no momento da chamada.

Nota: este módulo NÃO depende de ``dissertacao_mestrado`` (evita import circular).
"""
import os
from typing import Optional, Union

import anthropic
from openai import OpenAI, AsyncOpenAI


# URLs base e variáveis de ambiente com chaves de API por provedor.
# Gemini é acessado pelo endpoint compatível com a API da OpenAI.
PROVIDER_CONFIG = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "client_type": "openai",
    },
    "anthropic": {
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "client_type": "openai",
    },
}

# Corpo extra por provedor, injetado via ``extra_body`` no chat.completions.create.
# OpenRouter: desliga o fallback de provedor de inferência (reprodutibilidade — o
# mesmo model-ID pode ser servido por provedores/quantizações distintos).
PROVIDER_EXTRA_BODY = {
    "openrouter": {"provider": {"allow_fallbacks": False}},
}

# Pinagem do provedor de inferência POR MODELO (OpenRouter) — fixa quem serve cada
# modelo, para que o não-determinismo medido seja do modelo e não da mistura de
# infraestruturas/quantizações. Valores observados no smoke test de 02/07/2026
# (campo ``inference_provider`` da auditoria offline). Se o provedor pinado ficar
# indisponível, as chamadas FALHAM visivelmente (malformadas + auditoria) em vez de
# migrar em silêncio — comportamento correto para reprodutibilidade.
MODEL_PROVIDER_PIN = {
    "deepseek/deepseek-v4-flash":   {"order": ["DeepInfra"], "allow_fallbacks": False},
    "meta-llama/llama-4-scout":     {"order": ["DeepInfra"], "allow_fallbacks": False},
    "google/gemini-2.5-flash-lite": {"order": ["Google"],    "allow_fallbacks": False},
}


def get_extra_body(provider: str, model_name: Optional[str] = None) -> dict:
    """Corpo extra da requisição (``{}`` quando não há).

    A pinagem por modelo (``MODEL_PROVIDER_PIN``) tem precedência sobre o corpo
    default do provedor (``PROVIDER_EXTRA_BODY``).
    """
    if model_name and model_name in MODEL_PROVIDER_PIN:
        return {"provider": MODEL_PROVIDER_PIN[model_name]}
    return PROVIDER_EXTRA_BODY.get(provider, {})


def get_client(provider: str) -> Union[OpenAI, anthropic.Anthropic]:
    """Cria o cliente de API para o provedor especificado (OpenAI, Anthropic ou Gemini).

    Aplica os mesmos timeout=60s e max_retries=2 do cliente assíncrono, para que uma
    requisição travada no caminho síncrono (Anthropic/fallback) não bloqueie sem teto.
    """
    config = PROVIDER_CONFIG[provider]

    if config["client_type"] == "anthropic":
        return anthropic.Anthropic(
            api_key=os.getenv(config["api_key_env"]),
            timeout=60.0,
            max_retries=2,
        )
    else:
        if config["base_url"]:
            return OpenAI(
                api_key=os.getenv(config["api_key_env"]),
                base_url=config["base_url"],
                timeout=60.0,
                max_retries=2,
            )
        else:
            return OpenAI(
                api_key=os.getenv(config["api_key_env"]),
                timeout=60.0,
                max_retries=2,
            )


def get_async_client(provider: str) -> Optional[AsyncOpenAI]:
    """Cria o cliente assíncrono para chamadas concorrentes (apenas OpenAI/Gemini).

    Timeout de 60s previne que uma única requisição travada bloqueie o semaphore
    indefinidamente (causa observada: terminal parou silenciosamente em run anterior).
    """
    config = PROVIDER_CONFIG[provider]
    if config["client_type"] == "anthropic":
        return None
    if config["base_url"]:
        return AsyncOpenAI(
            api_key=os.getenv(config["api_key_env"]),
            base_url=config["base_url"],
            timeout=60.0,
            max_retries=2,
        )
    else:
        return AsyncOpenAI(
            api_key=os.getenv(config["api_key_env"]),
            timeout=60.0,
            max_retries=2,
        )
