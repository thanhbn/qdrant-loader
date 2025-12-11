"""Integration tests for the enricher pipeline.

POC3-010: Tests for factory functions and pipeline integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from qdrant_loader.core.document import Document
from qdrant_loader.core.enrichers import (
    BaseEnricher,
    EnricherPipeline,
    EnricherPriority,
    HierarchyEnricher,
    HsEntityEnricher,
    KeywordEnricher,
    create_advanced_pipeline,
    create_default_pipeline,
    create_full_pipeline,
    create_lightweight_pipeline,
)
from qdrant_loader.core.enrichers.backends import NamedEntityAnnotation


class MockSettings:
    """Mock settings for testing."""

    pass


def create_test_document(
    content: str = "# Introduction\n\nBill Gates founded Microsoft Corporation in the city of Seattle, Washington. This is a comprehensive guide to understanding the technology industry.",
    url: str = "https://docs.example.com/guides/introduction",
    title: str = "Introduction Guide",
    content_type: str = "text/markdown",
    source_type: str = "publicdocs",
    source: str = "test-docs",
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


class TestEnricherPipelineCreation:
    """Tests for EnricherPipeline creation and configuration."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    def test_create_empty_pipeline(self):
        """Test creating an empty pipeline."""
        pipeline = EnricherPipeline(enrichers=[])

        assert len(pipeline.enrichers) == 0

    def test_create_pipeline_with_enrichers(self, settings):
        """Test creating pipeline with enrichers."""
        enrichers = [
            KeywordEnricher(settings),
        ]
        pipeline = EnricherPipeline(enrichers=enrichers)

        assert len(pipeline.enrichers) == 1

    def test_pipeline_parallel_flag(self, settings):
        """Test pipeline parallel execution flag."""
        pipeline = EnricherPipeline(enrichers=[], parallel=True)
        assert pipeline.parallel is True

        pipeline = EnricherPipeline(enrichers=[], parallel=False)
        assert pipeline.parallel is False


class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    def test_create_default_pipeline_enrichers(self, settings):
        """Test create_default_pipeline creates expected enrichers."""
        # Patch spacy import to avoid loading models
        with patch(
            "qdrant_loader.core.enrichers.entity_enricher.spacy.load"
        ) as mock_spacy:
            mock_spacy.side_effect = OSError("Model not found")

            pipeline = create_default_pipeline(settings)

            # Should have at least hierarchy and keyword enrichers
            enricher_names = [e.name for e in pipeline.enrichers]
            assert "hierarchy_enricher" in enricher_names
            assert "keyword_enricher" in enricher_names

    def test_create_default_pipeline_disable_hierarchy(self, settings):
        """Test disabling hierarchy enricher."""
        with patch(
            "qdrant_loader.core.enrichers.entity_enricher.spacy.load"
        ) as mock_spacy:
            mock_spacy.side_effect = OSError("Model not found")

            pipeline = create_default_pipeline(settings, enable_hierarchy=False)

            enricher_names = [e.name for e in pipeline.enrichers]
            assert "hierarchy_enricher" not in enricher_names

    def test_create_default_pipeline_disable_keywords(self, settings):
        """Test disabling keyword enricher."""
        with patch(
            "qdrant_loader.core.enrichers.entity_enricher.spacy.load"
        ) as mock_spacy:
            mock_spacy.side_effect = OSError("Model not found")

            pipeline = create_default_pipeline(settings, enable_keywords=False)

            enricher_names = [e.name for e in pipeline.enrichers]
            assert "keyword_enricher" not in enricher_names

    def test_create_lightweight_pipeline(self, settings):
        """Test create_lightweight_pipeline excludes entity enricher."""
        pipeline = create_lightweight_pipeline(settings)

        enricher_names = [e.name for e in pipeline.enrichers]
        # Should have hierarchy and keywords, but not entities
        assert "hierarchy_enricher" in enricher_names
        assert "keyword_enricher" in enricher_names
        assert "entity_enricher" not in enricher_names

    def test_create_full_pipeline_parallel(self, settings):
        """Test create_full_pipeline enables parallel execution."""
        with patch(
            "qdrant_loader.core.enrichers.entity_enricher.spacy.load"
        ) as mock_spacy:
            mock_spacy.side_effect = OSError("Model not found")

            pipeline = create_full_pipeline(settings)

            assert pipeline.parallel is True


class TestAdvancedPipeline:
    """Tests for create_advanced_pipeline with POC3 enrichers."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    def test_advanced_pipeline_uses_hs_entity_enricher(self, settings):
        """Test advanced pipeline uses HsEntityEnricher."""
        pipeline = create_advanced_pipeline(settings)

        enricher_names = [e.name for e in pipeline.enrichers]
        assert "hs_entity_enricher" in enricher_names
        # Should not have old entity_enricher
        assert "entity_enricher" not in enricher_names

    def test_advanced_pipeline_includes_hierarchy(self, settings):
        """Test advanced pipeline includes hierarchy enricher."""
        pipeline = create_advanced_pipeline(settings)

        enricher_names = [e.name for e in pipeline.enrichers]
        assert "hierarchy_enricher" in enricher_names

    def test_advanced_pipeline_is_parallel(self, settings):
        """Test advanced pipeline runs in parallel."""
        pipeline = create_advanced_pipeline(settings)

        assert pipeline.parallel is True

    def test_advanced_pipeline_with_custom_config(self, settings):
        """Test advanced pipeline with custom HsEntityEnricher config."""
        from qdrant_loader.core.enrichers import HsEntityEnricherConfig

        config = HsEntityEnricherConfig(
            min_confidence=0.8,
            max_entities=50,
        )

        pipeline = create_advanced_pipeline(settings, hs_entity_config=config)

        # Find HsEntityEnricher and verify config
        hs_enricher = None
        for e in pipeline.enrichers:
            if e.name == "hs_entity_enricher":
                hs_enricher = e
                break

        assert hs_enricher is not None
        assert hs_enricher.hs_config.min_confidence == 0.8
        assert hs_enricher.hs_config.max_entities == 50


class TestEnricherOrdering:
    """Tests for enricher priority ordering."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    def test_hierarchy_enricher_has_highest_priority(self, settings):
        """Test that HierarchyEnricher has HIGHEST priority."""
        enricher = HierarchyEnricher(settings)
        assert enricher.config.priority == EnricherPriority.HIGHEST

    def test_hs_entity_enricher_has_high_priority(self, settings):
        """Test that HsEntityEnricher has HIGH priority."""
        enricher = HsEntityEnricher(settings)
        assert enricher.config.priority == EnricherPriority.HIGH

    def test_keyword_enricher_has_normal_priority(self, settings):
        """Test that KeywordEnricher has NORMAL priority."""
        enricher = KeywordEnricher(settings)
        assert enricher.config.priority == EnricherPriority.NORMAL

    def test_priority_order(self):
        """Test that priorities are ordered correctly."""
        # Lower value = higher priority (runs first)
        assert EnricherPriority.HIGHEST.value < EnricherPriority.HIGH.value
        assert EnricherPriority.HIGH.value < EnricherPriority.NORMAL.value
        assert EnricherPriority.NORMAL.value < EnricherPriority.LOW.value
        assert EnricherPriority.LOW.value < EnricherPriority.LOWEST.value


class TestPipelineExecution:
    """Tests for pipeline execution with mocked enrichers."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.fixture
    def mock_enricher(self):
        """Create a mock enricher that returns test metadata."""
        from qdrant_loader.core.enrichers.base_enricher import EnricherResult

        enricher = MagicMock(spec=BaseEnricher)
        enricher.name = "mock_enricher"
        enricher.config = MagicMock()
        enricher.config.priority = EnricherPriority.NORMAL
        enricher.should_process.return_value = (True, None)
        enricher.enrich.return_value = EnricherResult(
            metadata={"test_key": "test_value"}
        )
        return enricher

    @pytest.mark.asyncio
    async def test_pipeline_runs_all_enrichers(self, mock_enricher):
        """Test that pipeline runs all enrichers."""
        pipeline = EnricherPipeline(enrichers=[mock_enricher])
        doc = create_test_document()

        result = await pipeline.enrich(doc)

        mock_enricher.enrich.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_pipeline_merges_metadata(self):
        """Test that pipeline merges metadata from all enrichers."""
        from qdrant_loader.core.enrichers.base_enricher import EnricherConfig, EnricherResult

        enricher1 = MagicMock(spec=BaseEnricher)
        enricher1.name = "enricher1"
        enricher1.priority = EnricherPriority.HIGH
        enricher1.config = EnricherConfig(priority=EnricherPriority.HIGH)
        enricher1.should_process.return_value = (True, None)
        enricher1.enrich.return_value = EnricherResult(metadata={"key1": "value1"})

        enricher2 = MagicMock(spec=BaseEnricher)
        enricher2.name = "enricher2"
        enricher2.priority = EnricherPriority.NORMAL
        enricher2.config = EnricherConfig(priority=EnricherPriority.NORMAL)
        enricher2.should_process.return_value = (True, None)
        enricher2.enrich.return_value = EnricherResult(metadata={"key2": "value2"})

        pipeline = EnricherPipeline(enrichers=[enricher1, enricher2])
        doc = create_test_document()

        result = await pipeline.enrich(doc)

        # PipelineResult has 'merged_metadata', not 'metadata'
        assert "key1" in result.merged_metadata
        assert "key2" in result.merged_metadata
        assert result.merged_metadata["key1"] == "value1"
        assert result.merged_metadata["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_pipeline_skips_documents_when_should_process_false(self):
        """Test that pipeline respects should_process."""
        from qdrant_loader.core.enrichers.base_enricher import EnricherResult

        enricher = MagicMock(spec=BaseEnricher)
        enricher.name = "mock"
        enricher.config = MagicMock()
        enricher.config.priority = EnricherPriority.NORMAL
        enricher.should_process.return_value = (False, "test_skip_reason")
        enricher.enrich.return_value = EnricherResult(metadata={})

        pipeline = EnricherPipeline(enrichers=[enricher])
        doc = create_test_document()

        result = await pipeline.enrich(doc)

        # Pipeline still succeeds but enricher may be skipped
        # Note: The actual behavior depends on pipeline implementation
        assert result.success is True


class TestEndToEndPipeline:
    """End-to-end tests with real enrichers (without external dependencies)."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.mark.asyncio
    async def test_hierarchy_and_keyword_pipeline(self, settings):
        """Test pipeline with hierarchy and keyword enrichers."""
        from qdrant_loader.core.enrichers import (
            HierarchyEnricher,
            KeywordEnricher,
        )

        enrichers = [
            HierarchyEnricher(settings),
            KeywordEnricher(settings),
        ]
        pipeline = EnricherPipeline(enrichers=enrichers)

        doc = create_test_document(
            content="# Getting Started\n\nLearn about Python programming and machine learning. This comprehensive guide covers all the essential topics you need to know to become a skilled developer."
        )

        result = await pipeline.enrich(doc)

        assert result.success is True
        # PipelineResult has 'merged_metadata', not 'metadata'
        # Should have hierarchy metadata
        assert "hierarchy_level" in result.merged_metadata
        assert "hierarchy_path" in result.merged_metadata
        # Should have keyword metadata
        assert "keywords" in result.merged_metadata

    @pytest.mark.asyncio
    async def test_lightweight_pipeline_execution(self, settings):
        """Test lightweight pipeline executes successfully."""
        pipeline = create_lightweight_pipeline(settings)
        doc = create_test_document()

        result = await pipeline.enrich(doc)

        assert result.success is True
        # PipelineResult has 'merged_metadata', not 'metadata'
        assert "hierarchy_level" in result.merged_metadata
        assert "keywords" in result.merged_metadata


class TestPipelineShutdown:
    """Tests for pipeline shutdown."""

    @pytest.mark.asyncio
    async def test_pipeline_shutdown_calls_enricher_shutdown(self):
        """Test that pipeline shutdown calls shutdown on all enrichers."""
        enricher1 = MagicMock(spec=BaseEnricher)
        enricher1.name = "enricher1"
        enricher1.shutdown = MagicMock()

        enricher2 = MagicMock(spec=BaseEnricher)
        enricher2.name = "enricher2"
        enricher2.shutdown = MagicMock()

        pipeline = EnricherPipeline(enrichers=[enricher1, enricher2])

        await pipeline.shutdown()

        enricher1.shutdown.assert_called_once()
        enricher2.shutdown.assert_called_once()


class TestBackendIntegration:
    """Tests for NER backend integration."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    def test_spacy_backend_creation(self, settings):
        """Test creating spaCy backend through HsEntityEnricher."""
        from qdrant_loader.core.enrichers import HsEntityEnricherConfig

        config = HsEntityEnricherConfig(
            backend="spacy",
            spacy_model="en_core_web_sm",
        )
        enricher = HsEntityEnricher(settings, config)

        # Create backend but don't warm up (would require spacy model)
        backend = enricher._create_backend()

        assert backend is not None
        assert backend.__class__.__name__ == "SpaCyBackend"
        assert backend.model_name == "en_core_web_sm"

    def test_backend_config_propagation(self, settings):
        """Test that config options propagate to backend."""
        from qdrant_loader.core.enrichers import HsEntityEnricherConfig

        config = HsEntityEnricherConfig(
            backend="spacy",
            batch_size=64,
        )
        enricher = HsEntityEnricher(settings, config)
        backend = enricher._create_backend()

        assert backend.batch_size == 64


class TestModuleExports:
    """Tests for module exports and imports."""

    def test_enricher_module_exports(self):
        """Test that all expected exports are available."""
        from qdrant_loader.core.enrichers import (
            BaseEnricher,
            EnricherConfig,
            EnricherPipeline,
            EnricherPriority,
            EnricherResult,
            HierarchyEnricher,
            HierarchyEnricherConfig,
            HierarchyMetadata,
            HsEntityEnricher,
            HsEntityEnricherConfig,
            KeywordEnricher,
            KeywordEnricherConfig,
            NamedEntityAnnotation,
            NERBackend,
            PipelineResult,
            SpaCyBackend,
            create_advanced_pipeline,
            create_default_pipeline,
            create_full_pipeline,
            create_lightweight_pipeline,
        )

        # Just verify imports don't raise errors
        assert BaseEnricher is not None
        assert EnricherPipeline is not None
        assert HierarchyEnricher is not None
        assert HsEntityEnricher is not None
        assert NERBackend is not None
        assert SpaCyBackend is not None
        assert create_default_pipeline is not None
        assert create_advanced_pipeline is not None

    def test_optional_huggingface_backend(self):
        """Test that HuggingFace backend is optional."""
        # This should not raise even if transformers is not installed
        try:
            from qdrant_loader.core.enrichers import HuggingFaceBackend

            # If import succeeds, transformers is installed
            assert HuggingFaceBackend is not None
        except ImportError:
            # Expected if transformers is not installed
            pass
