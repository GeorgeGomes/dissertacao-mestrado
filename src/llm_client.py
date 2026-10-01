"""Fábrica de clientes de API de LLM — provedor único: OpenRouter.

Todo modelo do protocolo é acessado pelo OpenRouter (endpoint compatível com a API
da OpenAI), inclusive os da OpenAI (``openai/gpt-4o-mini``): um só provedor, uma só
chave (``OPENROUTER_API_KEY``) e a pinagem do provedor de inferência por modelo
(``MODEL_PROVIDER_PIN``) para reprodutibilidade. Para usar outro modelo, basta
acrescentá-lo a ``MODELS_TO_TEST`` com o id do OpenRouter (``<org>/<modelo>``), dar
um alias em ``MODEL_ALIAS`` e, se quiser fixar a infraestrutura, um pin aqui.

Os provedores diretos (OpenAI, Gemini, Anthropic) foram removidos em 01/10/2026;
``PROVIDER_CONFIG`` mantém o formato de registro para que o runner e os testes
continuem genéricos.

Não carrega o ``.env``: quem consome (o runner principal) chama ``load_dotenv()``
antes de usar as factories; a chave é lida via ``os.getenv`` no momento da chamada.
Este módulo NÃO depende de ``dissertacao_mestrado`` (evita import circular).
"""
import os
from typing import Optional

from openai import OpenAI, AsyncOpenAI


# Registro de provedores: URL base e variável de ambiente com a chave de API.
PROVIDER_CONFIG = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "client_type": "openai",  # SDK da OpenAI (endpoint compatível)
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
# infraestruturas/quantizações. Gemini: valor observado no smoke test de 02/07/2026
# (campo ``inference_provider`` da auditoria offline). Se o provedor pinado ficar
# indisponível, as chamadas FALHAM visivelmente (malformadas + auditoria) em vez de
# migrar em silêncio — comportamento correto para reprodutibilidade.
MODEL_PROVIDER_PIN = {
    "openai/gpt-4o-mini":           {"order": ["OpenAI"], "allow_fallbacks": False},
    "google/gemini-2.5-flash-lite": {"order": ["Google"], "allow_fallbacks": False},
}

_TIMEOUT = 60.0     # previne requisição travada bloqueando o semaphore indefinidamente
_MAX_RETRIES = 2    # retries do SDK; os retries de formato/rate-limit ficam no runner


def get_extra_body(provider: str, model_name: Optional[str] = None) -> dict:
    """Corpo extra da requisição (``{}`` quando não há).

    A pinagem por modelo (``MODEL_PROVIDER_PIN``) tem precedência sobre o corpo
    default do provedor (``PROVIDER_EXTRA_BODY``).
    """
    if model_name and model_name in MODEL_PROVIDER_PIN:
        return {"provider": MODEL_PROVIDER_PIN[model_name]}
    return PROVIDER_EXTRA_BODY.get(provider, {})


def get_client(provider: str) -> OpenAI:
    """Cliente síncrono do provedor (``KeyError`` para provedor não registrado)."""
    config = PROVIDER_CONFIG[provider]
    return OpenAI(
        api_key=os.getenv(config["api_key_env"]),
        base_url=config["base_url"],
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )


def get_async_client(provider: str) -> AsyncOpenAI:
    """Cliente assíncrono para as chamadas concorrentes (``asyncio.Semaphore``)."""
    config = PROVIDER_CONFIG[provider]
    return AsyncOpenAI(
        api_key=os.getenv(config["api_key_env"]),
        base_url=config["base_url"],
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )
