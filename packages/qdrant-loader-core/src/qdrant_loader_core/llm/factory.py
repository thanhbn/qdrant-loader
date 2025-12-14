from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .settings import LLMSettings
from .types import ChatClient, EmbeddingsClient, LLMProvider, TokenCounter

if TYPE_CHECKING:
    from .providers.ollama import OllamaProvider
    from .providers.openai import OpenAIProvider

# Lazy provider cache to avoid repeated imports
_provider_cache: dict[str, type] = {}


class _NoopEmbeddings(EmbeddingsClient):
    async def embed(self, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError("Embeddings provider not implemented")


class _NoopChat(ChatClient):
    async def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Chat provider not implemented")


class _NoopTokenizer(TokenCounter):
    def count(self, text: str) -> int:  # naive char-count fallback
        return len(text)


class _NoopProvider(LLMProvider):
    def embeddings(self) -> EmbeddingsClient:
        return _NoopEmbeddings()

    def chat(self) -> ChatClient:
        return _NoopChat()

    def tokenizer(self) -> TokenCounter:
        return _NoopTokenizer()


def _safe_hostname(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlparse(url).hostname
        return host.lower() if host else None
    except Exception:
        return None


def _get_openai_provider():
    """Lazily import OpenAI provider."""
    if "openai" not in _provider_cache:
        from .providers.openai import OpenAIProvider

        _provider_cache["openai"] = OpenAIProvider
    return _provider_cache["openai"]


def _get_ollama_provider():
    """Lazily import Ollama provider."""
    if "ollama" not in _provider_cache:
        from .providers.ollama import OllamaProvider

        _provider_cache["ollama"] = OllamaProvider
    return _provider_cache["ollama"]


def _get_azure_provider():
    """Lazily import Azure OpenAI provider."""
    if "azure" not in _provider_cache:
        try:
            from .providers.azure_openai import AzureOpenAIProvider

            _provider_cache["azure"] = AzureOpenAIProvider
        except Exception:
            _provider_cache["azure"] = None
    return _provider_cache["azure"]


def create_provider(settings: LLMSettings) -> LLMProvider:
    """Create a provider by settings.

    Phase 0: route OpenAI/OpenAI-compatible to OpenAIProvider when available; otherwise return a noop provider.
    Ollama returns a stub provider for now.
    """
    provider_name = (settings.provider or "").lower()
    base_url = settings.base_url or ""
    base_host = _safe_hostname(base_url)

    # Route Azure before generic OpenAI routing
    is_azure = "azure" in provider_name or (
        base_host is not None
        and (
            base_host == "openai.azure.com"
            or base_host.endswith(".openai.azure.com")
            or base_host == "cognitiveservices.azure.com"
            or base_host.endswith(".cognitiveservices.azure.com")
        )
    )
    AzureOpenAIProvider = _get_azure_provider()
    if is_azure and AzureOpenAIProvider is not None:
        try:
            return AzureOpenAIProvider(settings)
        except Exception:
            return _NoopProvider()

    if "openai" in provider_name or "openai" in base_url.lower():
        try:
            OpenAIProvider = _get_openai_provider()
            return OpenAIProvider(settings)
        except Exception:
            return _NoopProvider()

    if provider_name == "ollama" or (base_host in ("localhost", "127.0.0.1")):
        OllamaProvider = _get_ollama_provider()
        return OllamaProvider(settings)

    return _NoopProvider()
