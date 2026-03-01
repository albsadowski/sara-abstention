from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.cache import SQLiteCache


@cache
def chat_model(model: str, no_cache: bool) -> BaseChatModel:
    kwargs, provider = {}, "openai"
    if not no_cache:
        set_llm_cache(SQLiteCache(database_path="./.cache"))

    if model == "claude-sonnet":
        model = "claude-sonnet-4-6"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "qwen3":
        model = "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "kimi-k25":
        model = "accounts/fireworks/models/kimi-k2p5"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "gemini-2.5-flash":
        provider = "google_genai"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )

    return init_chat_model(model, model_provider=provider, **kwargs)
