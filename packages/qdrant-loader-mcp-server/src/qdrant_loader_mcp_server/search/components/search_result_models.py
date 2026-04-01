"""Backwards-compatible re-exports for search result models."""

from .models import (
    AttachmentInfo,
    BaseSearchResult,
    ChunkingContext,
    ContentAnalysis,
    ConversionInfo,
    CrossReferenceInfo,
    EnrichmentMetadata,
    HierarchyInfo,
    HybridSearchResult,
    NavigationContext,
    ProjectInfo,
    SectionInfo,
    SemanticAnalysis,
    create_hybrid_search_result,
)

__all__ = [
    "BaseSearchResult",
    "ProjectInfo",
    "HierarchyInfo",
    "AttachmentInfo",
    "SectionInfo",
    "ContentAnalysis",
    "SemanticAnalysis",
    "EnrichmentMetadata",
    "NavigationContext",
    "ChunkingContext",
    "ConversionInfo",
    "CrossReferenceInfo",
    "HybridSearchResult",
    "create_hybrid_search_result",
]
