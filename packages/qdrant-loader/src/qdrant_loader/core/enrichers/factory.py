"""Factory for creating enricher pipelines with default configurations.

POC2-005: Provides easy-to-use factory functions for creating enricher
pipelines with sensible defaults.

POC3-007: Added HierarchyEnricher and HsEntityEnricher support.
"""

from typing import TYPE_CHECKING

from .base_enricher import EnricherPriority
from .entity_enricher import EntityEnricher, EntityEnricherConfig
from .hierarchy_enricher import HierarchyEnricher, HierarchyEnricherConfig
from .hs_entity_enricher import HsEntityEnricher, HsEntityEnricherConfig
from .keyword_enricher import KeywordEnricher, KeywordEnricherConfig
from .pipeline import EnricherPipeline

from qdrant_loader.utils.logging import LoggingConfig

if TYPE_CHECKING:
    from qdrant_loader.config import Settings

logger = LoggingConfig.get_logger(__name__)


def create_default_pipeline(
    settings: "Settings",
    enable_hierarchy: bool = True,
    enable_entities: bool = True,
    enable_hs_entities: bool = False,
    enable_keywords: bool = True,
    parallel: bool = False,
    hs_entity_config: HsEntityEnricherConfig | None = None,
) -> EnricherPipeline:
    """Create an enricher pipeline with default enrichers.

    This factory function creates a pipeline with commonly used enrichers
    configured with sensible defaults.

    Args:
        settings: Application settings
        enable_hierarchy: Whether to include hierarchy detection (POC3)
        enable_entities: Whether to include entity extraction (requires spaCy)
        enable_hs_entities: Whether to use HsEntityEnricher instead of EntityEnricher (POC3)
        enable_keywords: Whether to include keyword extraction
        parallel: Whether to run enrichers in parallel
        hs_entity_config: Optional config for HsEntityEnricher

    Returns:
        Configured EnricherPipeline

    Example:
        from qdrant_loader.core.enrichers.factory import create_default_pipeline

        # Basic pipeline
        pipeline = create_default_pipeline(settings)

        # With HsEntityEnricher and confidence filtering
        config = HsEntityEnricherConfig(min_confidence=0.7)
        pipeline = create_default_pipeline(
            settings,
            enable_hs_entities=True,
            hs_entity_config=config,
        )

        result = await pipeline.enrich(document)
    """
    enrichers = []

    # HierarchyEnricher runs first (HIGHEST priority)
    if enable_hierarchy:
        try:
            hierarchy_config = HierarchyEnricherConfig()
            hierarchy_enricher = HierarchyEnricher(settings, hierarchy_config)
            enrichers.append(hierarchy_enricher)
            logger.debug("Added HierarchyEnricher to pipeline")
        except Exception as e:
            logger.warning(f"Failed to initialize HierarchyEnricher: {e}")

    # KeywordEnricher (NORMAL priority)
    if enable_keywords:
        keyword_config = KeywordEnricherConfig(
            max_keywords=20,
            include_bigrams=True,
            include_trigrams=False,
        )
        keyword_enricher = KeywordEnricher(settings, keyword_config)
        enrichers.append(keyword_enricher)
        logger.debug("Added KeywordEnricher to pipeline")

    # Entity extraction (HIGH priority)
    # Use HsEntityEnricher if explicitly enabled, otherwise fall back to EntityEnricher
    if enable_hs_entities:
        try:
            config = hs_entity_config or HsEntityEnricherConfig()
            hs_entity_enricher = HsEntityEnricher(settings, config)
            enrichers.append(hs_entity_enricher)
            logger.debug(f"Added HsEntityEnricher to pipeline (backend={config.backend})")
        except Exception as e:
            logger.warning(
                f"Failed to initialize HsEntityEnricher: {e}. "
                f"Falling back to EntityEnricher."
            )
            enable_entities = True
            enable_hs_entities = False

    if enable_entities and not enable_hs_entities:
        try:
            entity_config = EntityEnricherConfig(
                model_name="en_core_web_sm",
                extract_people=True,
                extract_organizations=True,
                extract_locations=True,
            )
            entity_enricher = EntityEnricher(settings, entity_config)
            enrichers.append(entity_enricher)
            logger.debug("Added EntityEnricher to pipeline")
        except Exception as e:
            logger.warning(
                f"Failed to initialize EntityEnricher (spaCy model may not be installed): {e}. "
                f"Entity extraction will be disabled."
            )

    pipeline = EnricherPipeline(enrichers=enrichers, parallel=parallel)
    logger.info(
        f"Created enricher pipeline with {len(enrichers)} enrichers "
        f"(parallel={parallel})"
    )

    return pipeline


def create_lightweight_pipeline(settings: "Settings") -> EnricherPipeline:
    """Create a lightweight pipeline with only keyword extraction.

    This is a faster alternative that skips entity extraction (which
    requires spaCy model loading). Still includes hierarchy detection
    as it's lightweight.

    Args:
        settings: Application settings

    Returns:
        Configured EnricherPipeline with HierarchyEnricher and KeywordEnricher
    """
    return create_default_pipeline(
        settings,
        enable_hierarchy=True,
        enable_entities=False,
        enable_keywords=True,
        parallel=False,
    )


def create_full_pipeline(settings: "Settings") -> EnricherPipeline:
    """Create a full pipeline with all available enrichers.

    Includes hierarchy detection, entity extraction, and keyword extraction
    running in parallel for maximum throughput.

    Args:
        settings: Application settings

    Returns:
        Configured EnricherPipeline with all enrichers
    """
    return create_default_pipeline(
        settings,
        enable_hierarchy=True,
        enable_entities=True,
        enable_keywords=True,
        parallel=True,
    )


def create_advanced_pipeline(
    settings: "Settings",
    hs_entity_config: HsEntityEnricherConfig | None = None,
) -> EnricherPipeline:
    """Create an advanced pipeline with POC3 enrichers.

    Uses HsEntityEnricher with confidence filtering and optionally
    HuggingFace backend for state-of-the-art NER.

    Args:
        settings: Application settings
        hs_entity_config: Optional HsEntityEnricher configuration.
            If None, uses spaCy backend with default settings.

    Returns:
        Configured EnricherPipeline with HierarchyEnricher, HsEntityEnricher,
        and KeywordEnricher

    Example:
        # With HuggingFace backend
        config = HsEntityEnricherConfig(
            backend="huggingface",
            hf_model="dslim/bert-base-NER",
            min_confidence=0.7,
        )
        pipeline = create_advanced_pipeline(settings, config)
    """
    return create_default_pipeline(
        settings,
        enable_hierarchy=True,
        enable_entities=False,
        enable_hs_entities=True,
        enable_keywords=True,
        parallel=True,
        hs_entity_config=hs_entity_config,
    )
