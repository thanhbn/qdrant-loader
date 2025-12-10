"""Pluggable metadata enrichers for document processing.

POC2: LlamaIndex-style Pluggable Enricher Pipeline
POC3: HierarchyEnricher and HsEntityEnricher (Haystack-inspired)

This module provides a flexible, plugin-based enrichment system that allows
users to configure and chain metadata enrichers. Enrichers process documents
and chunks to extract and add structured metadata.

Architecture:
- BaseEnricher: Abstract interface all enrichers must implement
- EnricherPipeline: Orchestrates running multiple enrichers in sequence
- Built-in enrichers: Entity, HsEntity, Keyword, Hierarchy

Example usage:
    # Option 1: Use factory for quick setup
    from qdrant_loader.core.enrichers import create_default_pipeline

    pipeline = create_default_pipeline(settings)
    result = await pipeline.enrich(document)

    # Option 2: Advanced pipeline with POC3 features
    from qdrant_loader.core.enrichers import (
        create_advanced_pipeline,
        HsEntityEnricherConfig,
    )

    config = HsEntityEnricherConfig(min_confidence=0.7)
    pipeline = create_advanced_pipeline(settings, config)

    # Option 3: Manual configuration
    from qdrant_loader.core.enrichers import (
        EnricherPipeline,
        HierarchyEnricher,
        HsEntityEnricher,
        KeywordEnricher,
    )

    pipeline = EnricherPipeline([
        HierarchyEnricher(settings),
        HsEntityEnricher(settings),
        KeywordEnricher(settings),
    ])

    enriched_doc = await pipeline.enrich(document)
"""

from .base_enricher import BaseEnricher, EnricherConfig, EnricherPriority, EnricherResult
from .entity_enricher import EntityEnricher, EntityEnricherConfig
from .factory import (
    create_advanced_pipeline,
    create_default_pipeline,
    create_full_pipeline,
    create_lightweight_pipeline,
)
from .hierarchy_enricher import HierarchyEnricher, HierarchyEnricherConfig, HierarchyMetadata
from .hs_entity_enricher import HsEntityEnricher, HsEntityEnricherConfig
from .keyword_enricher import KeywordEnricher, KeywordEnricherConfig
from .pipeline import EnricherPipeline, PipelineResult

# Re-export backend classes for advanced usage
from .backends import NamedEntityAnnotation, NERBackend, SpaCyBackend

__all__ = [
    # Base classes
    "BaseEnricher",
    "EnricherConfig",
    "EnricherPriority",
    "EnricherResult",
    # Pipeline
    "EnricherPipeline",
    "PipelineResult",
    # Factory functions
    "create_default_pipeline",
    "create_full_pipeline",
    "create_lightweight_pipeline",
    "create_advanced_pipeline",
    # Built-in enrichers
    "EntityEnricher",
    "EntityEnricherConfig",
    "KeywordEnricher",
    "KeywordEnricherConfig",
    # POC3: Hierarchy enricher
    "HierarchyEnricher",
    "HierarchyEnricherConfig",
    "HierarchyMetadata",
    # POC3: HsEntity enricher
    "HsEntityEnricher",
    "HsEntityEnricherConfig",
    # POC3: NER backends
    "NERBackend",
    "NamedEntityAnnotation",
    "SpaCyBackend",
]

# Optional HuggingFace backend
try:
    from .backends import HuggingFaceBackend

    __all__.append("HuggingFaceBackend")
except (ImportError, TypeError):
    # HuggingFace not installed
    pass
