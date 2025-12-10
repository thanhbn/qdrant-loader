"""Unit tests for HsEntityEnricher.

POC3-009: Tests for HsEntityEnricher with pluggable NER backends.
"""

from unittest.mock import MagicMock, patch

import pytest

from qdrant_loader.core.document import Document
from qdrant_loader.core.enrichers.backends import NamedEntityAnnotation
from qdrant_loader.core.enrichers.base_enricher import EnricherPriority
from qdrant_loader.core.enrichers.hs_entity_enricher import (
    ENTITY_CATEGORIES,
    HsEntityEnricher,
    HsEntityEnricherConfig,
)


class MockSettings:
    """Mock settings for testing."""

    pass


def create_test_document(
    content: str = "Bill Gates founded Microsoft in Seattle.",
    url: str = "https://example.com/article",
    title: str = "Test Article",
    content_type: str = "text/plain",
    source_type: str = "publicdocs",
    source: str = "test-source",
    metadata: dict | None = None,
) -> Document:
    """Create a test document with default values."""
    return Document(
        title=title,
        content=content,
        content_type=content_type,
        source_type=source_type,
        source=source,
        url=url,
        metadata=metadata or {},
    )


def create_mock_entities() -> list[NamedEntityAnnotation]:
    """Create mock entities for testing."""
    return [
        NamedEntityAnnotation(
            entity="PERSON",
            text="Bill Gates",
            start=0,
            end=10,
            score=0.95,
        ),
        NamedEntityAnnotation(
            entity="ORG",
            text="Microsoft",
            start=19,
            end=28,
            score=0.92,
        ),
        NamedEntityAnnotation(
            entity="GPE",
            text="Seattle",
            start=32,
            end=39,
            score=0.88,
        ),
    ]


class TestEntityCategories:
    """Tests for ENTITY_CATEGORIES mapping."""

    def test_ontonotes_person_mapping(self):
        """Test OntoNotes PERSON maps to people."""
        assert ENTITY_CATEGORIES["PERSON"] == "people"

    def test_ontonotes_org_mapping(self):
        """Test OntoNotes ORG maps to organizations."""
        assert ENTITY_CATEGORIES["ORG"] == "organizations"

    def test_ontonotes_location_mappings(self):
        """Test OntoNotes location types map to locations."""
        assert ENTITY_CATEGORIES["GPE"] == "locations"
        assert ENTITY_CATEGORIES["LOC"] == "locations"
        assert ENTITY_CATEGORIES["FAC"] == "locations"

    def test_conll_person_mapping(self):
        """Test CoNLL PER maps to people."""
        assert ENTITY_CATEGORIES["PER"] == "people"

    def test_conll_misc_mapping(self):
        """Test CoNLL MISC maps to miscellaneous."""
        assert ENTITY_CATEGORIES["MISC"] == "miscellaneous"


class TestHsEntityEnricherConfig:
    """Tests for HsEntityEnricherConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HsEntityEnricherConfig()

        assert config.backend == "spacy"
        assert config.spacy_model == "en_core_web_sm"
        assert config.hf_model is None
        assert config.hf_device is None
        assert config.min_confidence == 0.0
        assert config.max_entities == 100
        assert config.min_entity_length == 2
        assert config.include_categories is None
        assert config.exclude_categories == []
        assert config.deduplicate is True
        assert config.aggregation_strategy == "simple"
        assert config.batch_size == 32
        assert config.priority == EnricherPriority.HIGH

    def test_spacy_backend_config(self):
        """Test configuration for spaCy backend."""
        config = HsEntityEnricherConfig(
            backend="spacy",
            spacy_model="en_core_web_md",
        )

        assert config.backend == "spacy"
        assert config.spacy_model == "en_core_web_md"

    def test_huggingface_backend_config(self):
        """Test configuration for HuggingFace backend."""
        config = HsEntityEnricherConfig(
            backend="huggingface",
            hf_model="dslim/bert-base-NER",
            hf_device="cuda",
            aggregation_strategy="first",
        )

        assert config.backend == "huggingface"
        assert config.hf_model == "dslim/bert-base-NER"
        assert config.hf_device == "cuda"
        assert config.aggregation_strategy == "first"

    def test_confidence_filtering_config(self):
        """Test confidence filtering configuration."""
        config = HsEntityEnricherConfig(
            min_confidence=0.7,
            max_entities=50,
        )

        assert config.min_confidence == 0.7
        assert config.max_entities == 50

    def test_category_filtering_config(self):
        """Test category filtering configuration."""
        config = HsEntityEnricherConfig(
            include_categories=["PERSON", "ORG"],
            exclude_categories=["DATE", "TIME"],
        )

        assert config.include_categories == ["PERSON", "ORG"]
        assert config.exclude_categories == ["DATE", "TIME"]


class TestHsEntityEnricher:
    """Tests for HsEntityEnricher class."""

    @pytest.fixture
    def settings(self):
        """Create mock settings."""
        return MockSettings()

    @pytest.fixture
    def mock_backend(self):
        """Create mock NER backend."""
        backend = MagicMock()
        backend.extract_entities.return_value = [create_mock_entities()]
        return backend

    @pytest.fixture
    def enricher_with_mock(self, settings, mock_backend):
        """Create enricher with mocked backend."""
        enricher = HsEntityEnricher(settings)
        enricher._backend = mock_backend
        return enricher

    def test_enricher_name(self, settings):
        """Test enricher name property."""
        enricher = HsEntityEnricher(settings)
        assert enricher.name == "hs_entity_enricher"

    def test_enricher_priority(self, settings):
        """Test that HsEntityEnricher has HIGH priority."""
        enricher = HsEntityEnricher(settings)
        assert enricher.config.priority == EnricherPriority.HIGH

    def test_get_metadata_keys(self, settings):
        """Test metadata keys produced by enricher."""
        enricher = HsEntityEnricher(settings)
        keys = enricher.get_metadata_keys()

        expected_keys = [
            "named_entities",
            "entities",
            "entity_types",
            "entity_count",
            "has_people",
            "has_organizations",
            "has_locations",
        ]
        for key in expected_keys:
            assert key in keys


class TestEntityExtraction:
    """Tests for entity extraction functionality."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.fixture
    def mock_backend(self):
        backend = MagicMock()
        backend.extract_entities.return_value = [create_mock_entities()]
        return backend

    @pytest.fixture
    def enricher(self, settings, mock_backend):
        enricher = HsEntityEnricher(settings)
        enricher._backend = mock_backend
        return enricher

    @pytest.mark.asyncio
    async def test_basic_extraction(self, enricher):
        """Test basic entity extraction."""
        doc = create_test_document()

        result = await enricher.enrich(doc)

        assert result.success is True
        assert "named_entities" in result.metadata
        assert "entities" in result.metadata
        assert result.metadata["entity_count"] == 3

    @pytest.mark.asyncio
    async def test_haystack_format_output(self, enricher):
        """Test Haystack-compatible named_entities format."""
        doc = create_test_document()

        result = await enricher.enrich(doc)

        named_entities = result.metadata["named_entities"]
        assert len(named_entities) == 3

        # Check entity structure
        first_entity = named_entities[0]
        assert "entity" in first_entity
        assert "text" in first_entity
        assert "start" in first_entity
        assert "end" in first_entity
        assert "score" in first_entity

    @pytest.mark.asyncio
    async def test_legacy_format_output(self, enricher):
        """Test legacy entities format for backward compatibility."""
        doc = create_test_document()

        result = await enricher.enrich(doc)

        entities = result.metadata["entities"]
        assert len(entities) == 3

        # Check legacy entity structure
        first_entity = entities[0]
        assert first_entity["text"] == "Bill Gates"
        assert first_entity["type"] == "PERSON"
        assert first_entity["category"] == "people"
        assert "start" in first_entity
        assert "end" in first_entity
        assert "score" in first_entity

    @pytest.mark.asyncio
    async def test_entity_types_grouping(self, enricher):
        """Test entity_types groups entities by type."""
        doc = create_test_document()

        result = await enricher.enrich(doc)

        entity_types = result.metadata["entity_types"]
        assert "PERSON" in entity_types
        assert "Bill Gates" in entity_types["PERSON"]
        assert "ORG" in entity_types
        assert "Microsoft" in entity_types["ORG"]

    @pytest.mark.asyncio
    async def test_has_people_flag(self, enricher):
        """Test has_people convenience flag."""
        doc = create_test_document()

        result = await enricher.enrich(doc)

        assert result.metadata["has_people"] is True

    @pytest.mark.asyncio
    async def test_has_organizations_flag(self, enricher):
        """Test has_organizations convenience flag."""
        doc = create_test_document()

        result = await enricher.enrich(doc)

        assert result.metadata["has_organizations"] is True

    @pytest.mark.asyncio
    async def test_has_locations_flag(self, enricher):
        """Test has_locations convenience flag."""
        doc = create_test_document()

        result = await enricher.enrich(doc)

        assert result.metadata["has_locations"] is True


class TestConfidenceFiltering:
    """Tests for confidence score filtering."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_filter_low_confidence_entities(self, settings):
        """Test filtering entities below confidence threshold."""
        config = HsEntityEnricherConfig(min_confidence=0.9)
        enricher = HsEntityEnricher(settings, config)

        # Mock backend with mixed confidence scores
        entities = [
            NamedEntityAnnotation("PERSON", "Bill Gates", 0, 10, 0.95),
            NamedEntityAnnotation("ORG", "Microsoft", 19, 28, 0.85),  # Below threshold
            NamedEntityAnnotation("GPE", "Seattle", 32, 39, 0.70),  # Below threshold
        ]
        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [entities]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        # Only Bill Gates should pass (0.95 >= 0.9)
        assert result.metadata["entity_count"] == 1
        assert result.metadata["entities"][0]["text"] == "Bill Gates"

    @pytest.mark.asyncio
    async def test_zero_confidence_threshold(self, settings):
        """Test that zero threshold accepts all entities."""
        config = HsEntityEnricherConfig(min_confidence=0.0)
        enricher = HsEntityEnricher(settings, config)

        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [create_mock_entities()]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        assert result.metadata["entity_count"] == 3


class TestCategoryFiltering:
    """Tests for entity category filtering."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_include_categories_filter(self, settings):
        """Test including only specific categories."""
        config = HsEntityEnricherConfig(include_categories=["PERSON"])
        enricher = HsEntityEnricher(settings, config)

        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [create_mock_entities()]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        # Only PERSON entities should be included
        assert result.metadata["entity_count"] == 1
        assert result.metadata["entities"][0]["type"] == "PERSON"

    @pytest.mark.asyncio
    async def test_exclude_categories_filter(self, settings):
        """Test excluding specific categories."""
        config = HsEntityEnricherConfig(exclude_categories=["GPE"])
        enricher = HsEntityEnricher(settings, config)

        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [create_mock_entities()]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        # GPE should be excluded
        assert result.metadata["entity_count"] == 2
        entity_types = [e["type"] for e in result.metadata["entities"]]
        assert "GPE" not in entity_types


class TestDeduplication:
    """Tests for entity deduplication."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_deduplicate_same_text_and_type(self, settings):
        """Test deduplication of identical entities."""
        config = HsEntityEnricherConfig(deduplicate=True)
        enricher = HsEntityEnricher(settings, config)

        # Create duplicate entities
        entities = [
            NamedEntityAnnotation("PERSON", "Bill Gates", 0, 10, 0.95),
            NamedEntityAnnotation("PERSON", "Bill Gates", 50, 60, 0.90),  # Duplicate
            NamedEntityAnnotation("ORG", "Microsoft", 19, 28, 0.92),
        ]
        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [entities]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        # Should deduplicate Bill Gates
        assert result.metadata["entity_count"] == 2
        person_count = sum(
            1 for e in result.metadata["entities"] if e["type"] == "PERSON"
        )
        assert person_count == 1

    @pytest.mark.asyncio
    async def test_no_deduplication_when_disabled(self, settings):
        """Test that duplicates are kept when deduplication is disabled."""
        config = HsEntityEnricherConfig(deduplicate=False)
        enricher = HsEntityEnricher(settings, config)

        entities = [
            NamedEntityAnnotation("PERSON", "Bill Gates", 0, 10, 0.95),
            NamedEntityAnnotation("PERSON", "Bill Gates", 50, 60, 0.90),
        ]
        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [entities]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        # Both should be kept
        assert result.metadata["entity_count"] == 2


class TestMaxEntitiesLimit:
    """Tests for max entities limit."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_max_entities_limit(self, settings):
        """Test limiting number of entities returned."""
        config = HsEntityEnricherConfig(max_entities=2)
        enricher = HsEntityEnricher(settings, config)

        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [create_mock_entities()]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        # Should only return first 2 entities
        assert result.metadata["entity_count"] == 2


class TestMinEntityLength:
    """Tests for minimum entity length filtering."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_filter_short_entities(self, settings):
        """Test filtering entities shorter than min length."""
        config = HsEntityEnricherConfig(min_entity_length=5)
        enricher = HsEntityEnricher(settings, config)

        entities = [
            NamedEntityAnnotation("PERSON", "Bill Gates", 0, 10, 0.95),
            NamedEntityAnnotation("ORG", "IBM", 20, 23, 0.90),  # Too short
            NamedEntityAnnotation("GPE", "US", 30, 32, 0.85),  # Too short
        ]
        mock_backend = MagicMock()
        mock_backend.extract_entities.return_value = [entities]
        enricher._backend = mock_backend

        doc = create_test_document()
        result = await enricher.enrich(doc)

        # Only Bill Gates should pass (10 chars > 5)
        assert result.metadata["entity_count"] == 1


class TestShouldProcess:
    """Tests for should_process document filtering."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.fixture
    def enricher(self, settings):
        return HsEntityEnricher(settings)

    def test_should_process_normal_document(self, enricher):
        """Test that normal documents are processed."""
        # Content must be >= 50 characters to pass should_process
        doc = create_test_document(
            content="Bill Gates founded Microsoft Corporation in the city of Seattle, Washington."
        )

        should, reason = enricher.should_process(doc)

        assert should is True
        assert reason is None

    def test_skip_code_content(self, enricher):
        """Test that code content is skipped."""
        doc = create_test_document(content_type="text/code")

        should, reason = enricher.should_process(doc)

        assert should is False
        assert reason == "code_content"

    def test_skip_short_content(self, enricher):
        """Test that very short content is skipped."""
        doc = create_test_document(content="Hi")

        should, reason = enricher.should_process(doc)

        assert should is False
        assert "content_too_short" in reason


class TestBackendCreation:
    """Tests for NER backend creation."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    def test_create_spacy_backend(self, settings):
        """Test creating spaCy backend."""
        config = HsEntityEnricherConfig(backend="spacy")
        enricher = HsEntityEnricher(settings, config)

        # Don't warm up, just check backend type
        backend = enricher._create_backend()
        assert backend.__class__.__name__ == "SpaCyBackend"

    def test_huggingface_backend_requires_model(self, settings):
        """Test that HuggingFace backend requires model name."""
        config = HsEntityEnricherConfig(
            backend="huggingface",
            hf_model=None,  # No model specified
        )
        enricher = HsEntityEnricher(settings, config)

        with pytest.raises(ValueError, match="hf_model is required"):
            enricher._create_backend()

    def test_unknown_backend_raises_error(self, settings):
        """Test that unknown backend raises ValueError."""
        config = HsEntityEnricherConfig(backend="unknown")
        enricher = HsEntityEnricher(settings, config)

        with pytest.raises(ValueError, match="Unknown backend"):
            enricher._create_backend()


class TestShutdown:
    """Tests for enricher shutdown."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_shutdown_releases_backend(self, settings):
        """Test that shutdown releases backend resources."""
        enricher = HsEntityEnricher(settings)
        mock_backend = MagicMock()
        enricher._backend = mock_backend

        await enricher.shutdown()

        mock_backend.shutdown.assert_called_once()
        assert enricher._backend is None


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_extraction_error_returns_error_result(self, settings):
        """Test that extraction errors return error result."""
        enricher = HsEntityEnricher(settings)
        mock_backend = MagicMock()
        mock_backend.extract_entities.side_effect = RuntimeError("Backend failed")
        enricher._backend = mock_backend

        # Content must be >= 50 characters to pass should_process
        doc = create_test_document(
            content="Bill Gates founded Microsoft Corporation in the city of Seattle, Washington."
        )
        result = await enricher.enrich(doc)

        assert result.success is False
        assert any("Backend failed" in err for err in result.errors)
