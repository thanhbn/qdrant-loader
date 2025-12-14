# qdrant-loader-core package root
# Use lazy imports to improve CLI startup time

__all__ = [
    "EmbeddingsClient",
    "ChatClient",
    "TokenCounter",
    "LLMProvider",
    "LLMSettings",
    "RequestPolicy",
    "RateLimitPolicy",
    "EmbeddingPolicy",
    "create_provider",
]


def __getattr__(name: str):
    """Lazy import attributes to avoid loading heavy dependencies at startup."""
    if name in __all__:
        from .llm import (
            ChatClient,
            EmbeddingPolicy,
            EmbeddingsClient,
            LLMProvider,
            LLMSettings,
            RateLimitPolicy,
            RequestPolicy,
            TokenCounter,
            create_provider,
        )

        _exports = {
            "EmbeddingsClient": EmbeddingsClient,
            "ChatClient": ChatClient,
            "TokenCounter": TokenCounter,
            "LLMProvider": LLMProvider,
            "LLMSettings": LLMSettings,
            "RequestPolicy": RequestPolicy,
            "RateLimitPolicy": RateLimitPolicy,
            "EmbeddingPolicy": EmbeddingPolicy,
            "create_provider": create_provider,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
