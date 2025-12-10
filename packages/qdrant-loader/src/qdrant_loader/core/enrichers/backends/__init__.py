"""NER backend implementations for HsEntityEnricher.

POC3-005: Pluggable NER backend abstraction.

This package provides:
- NERBackend: Abstract interface for all NER backends
- SpaCyBackend: Default backend using spaCy models
- HuggingFaceBackend: Optional backend using HuggingFace Transformers

Backend selection is done via HsEntityEnricherConfig.backend parameter.
"""

from .base import NamedEntityAnnotation, NERBackend
from .spacy_backend import SpaCyBackend

__all__ = [
    "NERBackend",
    "NamedEntityAnnotation",
    "SpaCyBackend",
]

# Optional HuggingFace backend (requires transformers)
try:
    from .huggingface_backend import HuggingFaceBackend

    __all__.append("HuggingFaceBackend")
except ImportError:
    # HuggingFace backend not available
    HuggingFaceBackend = None  # type: ignore
