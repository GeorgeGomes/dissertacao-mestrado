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
import re
from typing import Optional

from openai import AsyncOpenAI


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

# Timeout e retries do SDK da OpenAI (por chamada HTTP):
#   _TIMEOUT    = 60 s por tentativa, para uma requisição travada não bloquear o
#                 semaphore indefinidamente;
#   _MAX_RETRIES = 2 retries do SDK (429/5xx/conexão), ou seja, até 3 tentativas e
#                 até 3 × 60 s = 180 s numa única chamada ao cliente.
# O runner (``async_llm_classify_point_openai`` em ``dissertacao_mestrado.py``)
# envolve a chamada em ``asyncio.wait_for(timeout=API_TIMEOUT_S)``: na prática o teto
# por chamada é esse e a 2ª/3ª tentativa do SDK pode ser cortada pelo runner. O estouro
# (e o ``APITimeoutError`` do SDK, "Request timed out.") é ERRO RETENTÁVEL
# (``_erro_retentavel``): entra no laço de retentativas com backoff do runner, e só
# depois de ``MAX_RETRIES`` esgotadas a chamada falha (não é tratado como resposta
# vazia/malformada; isso valia até 08/10/2026). Os retries de FORMATO e de rate-limit
# com backoff ficam no runner (``MAX_FORMAT_RETRIES``, ``MAX_RETRIES``).
_TIMEOUT = 60.0
_MAX_RETRIES = 2


def get_extra_body(provider: str, model_name: Optional[str] = None) -> dict:
    """Corpo extra da requisição (``{}`` quando não há).

    A pinagem por modelo (``MODEL_PROVIDER_PIN``) tem precedência sobre o corpo
    default do provedor (``PROVIDER_EXTRA_BODY``).
    """
    if model_name and model_name in MODEL_PROVIDER_PIN:
        return {"provider": MODEL_PROVIDER_PIN[model_name]}
    return PROVIDER_EXTRA_BODY.get(provider, {})


def get_async_client(provider: str) -> AsyncOpenAI:
    """Cliente assíncrono para as chamadas concorrentes (``asyncio.Semaphore``).

    ``KeyError`` para provedor não registrado. É a única fábrica: toda a coleta é
    assíncrona, e a fábrica síncrona ``get_client`` (removida em 09/10/2026) só era
    usada pelos testes.
    """
    config = PROVIDER_CONFIG[provider]
    return AsyncOpenAI(
        api_key=os.getenv(config["api_key_env"]),
        base_url=config["base_url"],
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )


# ── Créditos do OpenRouter (08/10/2026) ──────────────────────────────────────
# Um 402 (crédito insuficiente ou orçamento "em voo" esgotado) NÃO é um erro
# transitório: toda chamada seguinte falha igual e viraria rótulo de fallback
# MD5 (malformada), contaminando os resultados em silêncio. O runner aborta.

class CreditosEsgotadosError(RuntimeError):
    """Saldo do OpenRouter acabou (HTTP 402) no meio da coleta."""


_SINAIS_CREDITO = (
    "insufficient credits", "insufficient_credits", "in_flight_budget",
    "exceeds_budget", "payment required",
)
# "402" só conta como código de status (não como substring de request_id,
# max_tokens, context length…): pelo atributo do SDK ou por "error code: 402".
_RE_STATUS_402 = re.compile(r"(?:error code|status(?: code)?)\s*[:=]?\s*402\b")


def erro_de_credito(exc: BaseException) -> bool:
    """True se a exceção da API indica falta de crédito (fatal, não retentável)."""
    if getattr(exc, "status_code", None) == 402:
        return True
    texto = str(exc).lower()
    if any(sinal in texto for sinal in _SINAIS_CREDITO):
        return True
    return bool(_RE_STATUS_402.search(texto))


def parse_creditos(payload: dict) -> dict:
    """Converte a resposta de GET /api/v1/credits em {total, usado, saldo} (USD)."""
    data = payload.get("data", payload)
    total = float(data["total_credits"])
    usado = float(data["total_usage"])
    return {"total": total, "usado": usado, "saldo": total - usado}


def consultar_saldo_openrouter(api_key: Optional[str] = None, timeout: float = 15.0) -> Optional[dict]:
    """Consulta o saldo da chave no OpenRouter. Devolve None se a consulta falhar."""
    import httpx
    key = api_key or os.getenv(PROVIDER_CONFIG["openrouter"]["api_key_env"])
    if not key:
        return None
    try:
        resp = httpx.get(f"{PROVIDER_CONFIG['openrouter']['base_url']}/credits",
                         headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        resp.raise_for_status()
        return parse_creditos(resp.json())
    except Exception as exc:  # rede, JSON, chave inválida: não derruba a execução
        print(f"  ⚠ Não foi possível consultar o saldo do OpenRouter: {str(exc)[:120]}")
        return None
