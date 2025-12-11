"""Enhanced entity enricher with Haystack-style features.

POC3-004: HsEntityEnricher with pluggable NER backends.

This module provides the HsEntityEnricher class with improvements over
the base EntityEnricher:

1. Confidence score filtering (min_confidence threshold)
2. Pluggable backends (spaCy + HuggingFace Transformers)
3. Haystack-compatible NamedEntityAnnotation format
4. Batch processing optimization

The "Hs" prefix indicates Haystack-inspired design patterns.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .backends import NamedEntityAnnotation, NERBackend, SpaCyBackend
from .base_enricher import (
    BaseEnricher,
    EnricherConfig,
    EnricherPriority,
    EnricherResult,
)

if TYPE_CHECKING:
    from qdrant_loader.config import Settings
    from qdrant_loader.core.document import Document


# Entity category mapping (OntoNotes + CoNLL labels)
ENTITY_CATEGORIES = {
    # OntoNotes scheme (spaCy default)
    "PERSON": "people",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "FAC": "locations",
    "PRODUCT": "products",
    "EVENT": "events",
    "WORK_OF_ART": "works",
    "LAW": "legal",
    "LANGUAGE": "languages",
    "DATE": "dates",
    "TIME": "times",
    "PERCENT": "percentages",
    "MONEY": "monetary",
    "QUANTITY": "quantities",
    "ORDINAL": "ordinals",
    "CARDINAL": "numbers",
    "NORP": "groups",
    # CoNLL scheme (common in HuggingFace models)
    "PER": "people",
    "MISC": "miscellaneous",
}


@dataclass
class HsEntityEnricherConfig(EnricherConfig):
    """Configuration for HsEntityEnricher.

    Attributes:
        backend: NER backend to use ("spacy" or "huggingface")
        spacy_model: spaCy model name (when backend="spacy")
        hf_model: HuggingFace model name (when backend="huggingface")
        hf_device: Device for HuggingFace model (None=auto, "cpu", "cuda")
        min_confidence: Minimum confidence score to accept entity (0.0-1.0)
        max_entities: Maximum entities to extract per document
        min_entity_length: Minimum entity text length
        include_categories: Only include these entity types (None=all)
        exclude_categories: Exclude these entity types
        deduplicate: Remove duplicate entities (same text+type)
        aggregation_strategy: HuggingFace token aggregation ("simple", "first", "max")
        batch_size: Batch size for processing
    """

    # Backend selection
    backend: str = "spacy"

    # spaCy options
    spacy_model: str = "en_core_web_sm"

    # HuggingFace options
    hf_model: str | None = None
    hf_device: str | None = None

    # Filtering options
    min_confidence: float = 0.0
    max_entities: int = 100
    min_entity_length: int = 2

    # Entity type filtering
    include_categories: list[str] | None = None
    exclude_categories: list[str] = field(default_factory=list)

    # Output options
    deduplicate: bool = True
    aggregation_strategy: str = "simple"

    # Performance
    batch_size: int = 32

    def __post_init__(self):
        """Set priority to HIGH (after hierarchy, before keywords)."""
        self.priority = EnricherPriority.HIGH


class HsEntityEnricher(BaseEnricher):
    """Enhanced entity enricher with Haystack-style features.

    Improvements over base EntityEnricher:
    - Confidence score filtering
    - Pluggable NER backends (spaCy + HuggingFace)
    - Haystack-compatible output format
    - Better batch processing

    Output Formats:
        1. Haystack format: named_entities list with NamedEntityAnnotation
        2. Legacy format: entities, entity_types, has_* flags

    Example usage:
        # Using spaCy (default)
        enricher = HsEntityEnricher(settings)

        # Using HuggingFace with confidence filtering
        config = HsEntityEnricherConfig(
            backend="huggingface",
            hf_model="dslim/bert-base-NER",
            min_confidence=0.7,
        )
        enricher = HsEntityEnricher(settings, config)

        result = await enricher.enrich(document)
    """

    def __init__(
        self,
        settings: "Settings",
        config: HsEntityEnricherConfig | None = None,
    ):
        """Initialize the HsEntityEnricher.

        Args:
            settings: Application settings
            config: Enricher-specific configuration
        """
        config = config or HsEntityEnricherConfig()
        super().__init__(settings, config)
        self.hs_config = config
        self._backend: NERBackend | None = None

    @property
    def name(self) -> str:
        """Unique identifier for this enricher."""
        return "hs_entity_enricher"

    @property
    def backend(self) -> NERBackend:
        """Lazy-load and return the NER backend.

        Returns:
            Configured NERBackend instance

        Raises:
            ValueError: If backend configuration is invalid
            ImportError: If required dependencies are missing
        """
        if self._backend is None:
            self._backend = self._create_backend()
            self._backend.warm_up()
        return self._backend

    def _create_backend(self) -> NERBackend:
        """Create the appropriate NER backend based on config.

        Returns:
            NERBackend instance

        Raises:
            ValueError: If backend type is unknown or misconfigured
        """
        if self.hs_config.backend == "huggingface":
            if not self.hs_config.hf_model:
                raise ValueError(
                    "hf_model is required when backend='huggingface'. "
                    "Example: 'dslim/bert-base-NER'"
                )

            # Import here to make HuggingFace optional
            try:
                from .backends.huggingface_backend import HuggingFaceBackend
            except ImportError as e:
                raise ImportError(
                    "HuggingFace backend requires: pip install transformers torch"
                ) from e

            return HuggingFaceBackend(
                model_name=self.hs_config.hf_model,
                device=self.hs_config.hf_device,
                aggregation_strategy=self.hs_config.aggregation_strategy,
                batch_size=self.hs_config.batch_size,
            )

        elif self.hs_config.backend == "spacy":
            return SpaCyBackend(
                model_name=self.hs_config.spacy_model,
                batch_size=self.hs_config.batch_size,
            )

        else:
            raise ValueError(
                f"Unknown backend: {self.hs_config.backend}. "
                f"Supported: 'spacy', 'huggingface'"
            )

    def should_process(self, document: "Document") -> tuple[bool, str | None]:
        """Check if document should be processed.

        Skips code files and very short content in addition to base checks.

        Args:
            document: Document to check

        Returns:
            Tuple of (should_process, skip_reason)
        """
        should, reason = super().should_process(document)
        if not should:
            return should, reason

        # Skip code files (entity extraction not useful)
        if document.content_type and "code" in document.content_type.lower():
            return False, "code_content"

        # Skip very short content
        content_length = len(document.content) if document.content else 0
        if content_length < 50:
            return False, f"content_too_short ({content_length} < 50)"

        return True, None

    async def enrich(self, document: "Document") -> EnricherResult:
        """Extract named entities with confidence filtering.

        Args:
            document: The document to enrich

        Returns:
            EnricherResult with both Haystack and legacy formats
        """
        try:
            # Extract entities using backend
            all_entities = self.backend.extract_entities([document.content])
            raw_entities = all_entities[0] if all_entities else []

            # Apply filters
            filtered_entities = self._filter_entities(raw_entities)

            # Build output metadata
            metadata = self._build_metadata(filtered_entities)

            return EnricherResult(metadata=metadata)

        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")
            return EnricherResult.error_result(str(e))

    def _filter_entities(
        self,
        entities: list[NamedEntityAnnotation],
    ) -> list[NamedEntityAnnotation]:
        """Apply confidence, category, and deduplication filters.

        Args:
            entities: Raw entity annotations from backend

        Returns:
            Filtered list of entities
        """
        filtered: list[NamedEntityAnnotation] = []
        seen: set[tuple[str, str]] = set()

        for ent in entities:
            # Confidence filter
            if ent.score < self.hs_config.min_confidence:
                continue

            # Length filter
            if len(ent.text.strip()) < self.hs_config.min_entity_length:
                continue

            # Category inclusion filter
            if self.hs_config.include_categories:
                if ent.entity not in self.hs_config.include_categories:
                    continue

            # Category exclusion filter
            if ent.entity in self.hs_config.exclude_categories:
                continue

            # Deduplication
            key = (ent.text.lower().strip(), ent.entity)
            if self.hs_config.deduplicate and key in seen:
                continue
            seen.add(key)

            filtered.append(ent)

            # Max entities limit
            if len(filtered) >= self.hs_config.max_entities:
                break

        return filtered

    def _build_metadata(
        self,
        entities: list[NamedEntityAnnotation],
    ) -> dict[str, Any]:
        """Build output metadata in both Haystack and legacy formats.

        Args:
            entities: Filtered entity annotations

        Returns:
            Metadata dictionary with both formats
        """
        # Haystack format: named_entities list
        named_entities = [ent.to_dict() for ent in entities]

        # Legacy format for backward compatibility
        entity_types: dict[str, list[str]] = defaultdict(list)
        legacy_entities: list[dict[str, Any]] = []

        for ent in entities:
            # Group by type
            entity_types[ent.entity].append(ent.text)

            # Legacy entity format
            legacy_entity = {
                "text": ent.text,
                "type": ent.entity,
                "category": ENTITY_CATEGORIES.get(ent.entity, "other"),
                "start": ent.start,
                "end": ent.end,
                "score": ent.score,
            }
            legacy_entities.append(legacy_entity)

        # Boolean flags for common entity types
        has_people = bool(entity_types.get("PERSON") or entity_types.get("PER"))
        has_orgs = bool(entity_types.get("ORG"))
        has_locations = bool(
            entity_types.get("GPE")
            or entity_types.get("LOC")
            or entity_types.get("FAC")
        )

        return {
            # Haystack-compatible format
            "named_entities": named_entities,
            # Legacy format
            "entities": legacy_entities,
            "entity_types": dict(entity_types),
            "entity_count": len(entities),
            # Convenience flags
            "has_people": has_people,
            "has_organizations": has_orgs,
            "has_locations": has_locations,
        }

    async def shutdown(self) -> None:
        """Clean up backend resources."""
        if self._backend:
            self._backend.shutdown()
            self._backend = None
        self.logger.debug("HsEntityEnricher shutdown")

    def get_metadata_keys(self) -> list[str]:
        """Return the list of metadata keys this enricher produces."""
        return [
            "named_entities",
            "entities",
            "entity_types",
            "entity_count",
            "has_people",
            "has_organizations",
            "has_locations",
        ]
