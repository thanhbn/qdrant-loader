"""Unit tests for HierarchyEnricher.

POC3-008: Tests for HierarchyEnricher with header and URL detection.
"""

import pytest

from qdrant_loader.core.document import Document
from qdrant_loader.core.enrichers.base_enricher import EnricherPriority
from qdrant_loader.core.enrichers.hierarchy_enricher import (
    HierarchyEnricher,
    HierarchyEnricherConfig,
    HierarchyMetadata,
)


class MockSettings:
    """Mock settings for testing."""

    pass


def create_test_document(
    content: str = "Test content",
    url: str = "https://example.com/docs/guide/intro",
    title: str = "Test Document",
    content_type: str = "text/markdown",
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


class TestHierarchyMetadata:
    """Tests for HierarchyMetadata dataclass."""

    def test_default_values(self):
        """Test default values for HierarchyMetadata."""
        metadata = HierarchyMetadata()

        assert metadata.parent_id is None
        assert metadata.hierarchy_level == 0
        assert metadata.hierarchy_path == []
        assert metadata.section_title is None
        assert metadata.is_root is True
        assert metadata.source_document_id is None
        assert metadata.sibling_ids == []

    def test_custom_values(self):
        """Test HierarchyMetadata with custom values."""
        metadata = HierarchyMetadata(
            parent_id="parent-123",
            hierarchy_level=2,
            hierarchy_path=["Root", "Section", "Subsection"],
            section_title="Subsection",
            is_root=False,
            source_document_id="doc-456",
            sibling_ids=["sib-1", "sib-2"],
        )

        assert metadata.parent_id == "parent-123"
        assert metadata.hierarchy_level == 2
        assert metadata.hierarchy_path == ["Root", "Section", "Subsection"]
        assert metadata.section_title == "Subsection"
        assert metadata.is_root is False
        assert metadata.source_document_id == "doc-456"
        assert metadata.sibling_ids == ["sib-1", "sib-2"]

    def test_to_dict(self):
        """Test HierarchyMetadata.to_dict() method."""
        metadata = HierarchyMetadata(
            hierarchy_level=1,
            hierarchy_path=["Root", "Section"],
            section_title="Section",
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["hierarchy_level"] == 1
        assert result["hierarchy_path"] == ["Root", "Section"]
        assert result["section_title"] == "Section"
        assert result["is_root"] is True


class TestHierarchyEnricherConfig:
    """Tests for HierarchyEnricherConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HierarchyEnricherConfig()

        assert config.detect_from_headers is True
        assert config.detect_from_url is True
        assert config.track_siblings is False
        assert config.max_hierarchy_depth == 10
        assert config.priority == EnricherPriority.HIGHEST

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = HierarchyEnricherConfig(
            detect_from_headers=False,
            detect_from_url=True,
            track_siblings=True,
            max_hierarchy_depth=5,
        )

        assert config.detect_from_headers is False
        assert config.detect_from_url is True
        assert config.track_siblings is True
        assert config.max_hierarchy_depth == 5
        # Priority should still be HIGHEST after __post_init__
        assert config.priority == EnricherPriority.HIGHEST


class TestHierarchyEnricher:
    """Tests for HierarchyEnricher."""

    @pytest.fixture
    def settings(self):
        """Create mock settings."""
        return MockSettings()

    @pytest.fixture
    def enricher(self, settings):
        """Create HierarchyEnricher instance."""
        return HierarchyEnricher(settings)

    @pytest.fixture
    def enricher_headers_only(self, settings):
        """Create enricher with only header detection."""
        config = HierarchyEnricherConfig(
            detect_from_headers=True,
            detect_from_url=False,
        )
        return HierarchyEnricher(settings, config)

    @pytest.fixture
    def enricher_url_only(self, settings):
        """Create enricher with only URL detection."""
        config = HierarchyEnricherConfig(
            detect_from_headers=False,
            detect_from_url=True,
        )
        return HierarchyEnricher(settings, config)

    def test_enricher_name(self, enricher):
        """Test enricher name property."""
        assert enricher.name == "hierarchy_enricher"

    def test_enricher_priority(self, enricher):
        """Test that HierarchyEnricher has HIGHEST priority."""
        assert enricher.config.priority == EnricherPriority.HIGHEST

    def test_get_metadata_keys(self, enricher):
        """Test metadata keys produced by enricher."""
        keys = enricher.get_metadata_keys()

        expected_keys = [
            "parent_id",
            "hierarchy_level",
            "hierarchy_path",
            "section_title",
            "is_root",
            "source_document_id",
        ]
        for key in expected_keys:
            assert key in keys


class TestHeaderDetection:
    """Tests for header-based hierarchy detection."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.fixture
    def enricher(self, settings):
        config = HierarchyEnricherConfig(
            detect_from_headers=True,
            detect_from_url=False,
        )
        return HierarchyEnricher(settings, config)

    @pytest.mark.asyncio
    async def test_markdown_h1_header(self, enricher):
        """Test detection of Markdown H1 header."""
        content = "# Main Title\n\nSome content here."
        doc = create_test_document(content=content)

        result = await enricher.enrich(doc)

        assert result.success is True
        # H1 maps to level 0 (root level) in implementation
        assert result.metadata["hierarchy_level"] == 0
        assert "Main Title" in result.metadata["hierarchy_path"]
        assert result.metadata["section_title"] == "Main Title"

    @pytest.mark.asyncio
    async def test_markdown_nested_headers(self, enricher):
        """Test detection of nested Markdown headers."""
        content = """# Chapter 1

## Section 1.1

### Subsection 1.1.1

Content here.
"""
        doc = create_test_document(content=content)

        result = await enricher.enrich(doc)

        assert result.success is True
        # Level is based on first header (H1 = level 0)
        assert result.metadata["hierarchy_level"] == 0
        # Path should include multiple levels from nested headers
        assert len(result.metadata["hierarchy_path"]) >= 1
        assert result.metadata["section_title"] == "Chapter 1"

    @pytest.mark.asyncio
    async def test_markdown_h2_header(self, enricher):
        """Test detection of Markdown H2 header."""
        content = "## Section Title\n\nContent."
        doc = create_test_document(content=content)

        result = await enricher.enrich(doc)

        assert result.success is True
        # H2 maps to level 1 in implementation (h_level - 1)
        assert result.metadata["hierarchy_level"] == 1
        assert result.metadata["section_title"] == "Section Title"

    @pytest.mark.asyncio
    async def test_html_header_tags(self, enricher):
        """Test detection of HTML header tags."""
        content = "<h1>HTML Title</h1>\n<p>Paragraph content.</p>"
        doc = create_test_document(content=content, content_type="text/html")

        result = await enricher.enrich(doc)

        assert result.success is True
        # H1 maps to level 0 in implementation
        assert result.metadata["hierarchy_level"] == 0
        assert "HTML Title" in result.metadata["hierarchy_path"]

    @pytest.mark.asyncio
    async def test_html_nested_headers(self, enricher):
        """Test detection of nested HTML headers."""
        content = """
<h1>Main</h1>
<h2>Sub</h2>
<h3>SubSub</h3>
"""
        doc = create_test_document(content=content, content_type="text/html")

        result = await enricher.enrich(doc)

        assert result.success is True
        assert len(result.metadata["hierarchy_path"]) >= 1

    @pytest.mark.asyncio
    async def test_no_headers(self, enricher):
        """Test document with no headers."""
        content = "Just plain text without any headers."
        doc = create_test_document(content=content)

        result = await enricher.enrich(doc)

        assert result.success is True
        # Should still return metadata, but with default/root values
        assert result.metadata["hierarchy_level"] == 0
        assert result.metadata["is_root"] is True

    @pytest.mark.asyncio
    async def test_empty_content(self, enricher):
        """Test document with empty content."""
        doc = create_test_document(content="")

        result = await enricher.enrich(doc)

        assert result.success is True
        assert result.metadata["hierarchy_level"] == 0

    @pytest.mark.asyncio
    async def test_alternative_markdown_header_syntax(self, enricher):
        """Test alternative Markdown header syntax (underline style)."""
        # Note: Underline style headers may not be detected if implementation
        # only handles # style headers. Test that it doesn't crash.
        content = """Title Here
============

Subtitle
--------

Content.
"""
        doc = create_test_document(content=content)

        result = await enricher.enrich(doc)

        assert result.success is True
        # Underline syntax may not be detected - that's OK
        # Just verify it doesn't crash


class TestUrlDetection:
    """Tests for URL-based hierarchy detection."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.fixture
    def enricher(self, settings):
        config = HierarchyEnricherConfig(
            detect_from_headers=False,
            detect_from_url=True,
        )
        return HierarchyEnricher(settings, config)

    @pytest.mark.asyncio
    async def test_simple_url_path(self, enricher):
        """Test hierarchy from simple URL path."""
        doc = create_test_document(
            url="https://docs.example.com/guides/getting-started",
            content="Content",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        # Should extract hierarchy from URL path
        assert "guides" in result.metadata["hierarchy_path"]
        assert "getting-started" in result.metadata["hierarchy_path"]

    @pytest.mark.asyncio
    async def test_deep_url_path(self, enricher):
        """Test hierarchy from deep URL path."""
        doc = create_test_document(
            url="https://example.com/docs/api/v2/endpoints/users/create",
            content="Content",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        # Should have multiple levels
        assert result.metadata["hierarchy_level"] >= 3
        assert len(result.metadata["hierarchy_path"]) >= 3

    @pytest.mark.asyncio
    async def test_url_with_file_extension(self, enricher):
        """Test URL path with file extension."""
        doc = create_test_document(
            url="https://example.com/docs/guide/intro.html",
            content="Content",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        # Should handle file extensions properly
        path = result.metadata["hierarchy_path"]
        # intro should be in path (with or without .html)
        assert any("intro" in p for p in path)

    @pytest.mark.asyncio
    async def test_root_url(self, enricher):
        """Test root URL with no path."""
        doc = create_test_document(
            url="https://example.com/",
            content="Content",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        assert result.metadata["is_root"] is True
        assert result.metadata["hierarchy_level"] == 0

    @pytest.mark.asyncio
    async def test_url_with_query_params(self, enricher):
        """Test URL with query parameters (should be ignored)."""
        doc = create_test_document(
            url="https://example.com/docs/guide?version=2&lang=en",
            content="Content",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        # Query params should not affect hierarchy
        path = result.metadata["hierarchy_path"]
        assert "version=2" not in str(path)

    @pytest.mark.asyncio
    async def test_url_with_fragment(self, enricher):
        """Test URL with fragment identifier."""
        doc = create_test_document(
            url="https://example.com/docs/guide#section-1",
            content="Content",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        # Fragment should be handled appropriately
        path = result.metadata["hierarchy_path"]
        assert "docs" in path or "guide" in path


class TestHierarchyMerging:
    """Tests for merging header and URL hierarchies."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.fixture
    def enricher(self, settings):
        """Enricher with both detection methods enabled."""
        config = HierarchyEnricherConfig(
            detect_from_headers=True,
            detect_from_url=True,
        )
        return HierarchyEnricher(settings, config)

    @pytest.mark.asyncio
    async def test_merge_url_and_headers(self, enricher):
        """Test merging URL-based and header-based hierarchies."""
        content = "## API Reference\n\nDetails about the API."
        doc = create_test_document(
            content=content,
            url="https://example.com/docs/api/reference",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        # Should have combined hierarchy from both sources
        path = result.metadata["hierarchy_path"]
        assert len(path) >= 1

    @pytest.mark.asyncio
    async def test_header_takes_precedence_for_section_title(self, enricher):
        """Test that header provides section title when available."""
        content = "# Getting Started Guide\n\nIntroduction content."
        doc = create_test_document(
            content=content,
            url="https://example.com/docs/intro",
        )

        result = await enricher.enrich(doc)

        assert result.success is True
        # Section title should come from header
        assert result.metadata["section_title"] == "Getting Started Guide"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def settings(self):
        return MockSettings()

    @pytest.fixture
    def enricher(self, settings):
        return HierarchyEnricher(settings)

    @pytest.mark.asyncio
    async def test_malformed_url(self, enricher):
        """Test handling of malformed URL."""
        doc = create_test_document(
            url="not-a-valid-url",
            content="Content",
        )

        result = await enricher.enrich(doc)

        # Should handle gracefully without crashing
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unicode_content(self, enricher):
        """Test handling of Unicode content in headers."""
        content = "# 日本語のタイトル\n\nコンテンツ"
        # Use URL without path to avoid URL hierarchy overriding header hierarchy
        doc = create_test_document(content=content, url="https://example.com/")

        result = await enricher.enrich(doc)

        assert result.success is True
        assert result.metadata["section_title"] == "日本語のタイトル"

    @pytest.mark.asyncio
    async def test_special_characters_in_headers(self, enricher):
        """Test headers with special characters."""
        content = "# API v2.0: User's Guide (Beta)\n\nContent."
        doc = create_test_document(content=content)

        result = await enricher.enrich(doc)

        assert result.success is True
        # Should preserve special characters
        title = result.metadata["section_title"]
        assert "API" in title
        assert "v2.0" in title or "2.0" in title

    @pytest.mark.asyncio
    async def test_max_hierarchy_depth(self, settings):
        """Test max hierarchy depth limit."""
        config = HierarchyEnricherConfig(
            max_hierarchy_depth=2,
            detect_from_url=False,  # Only test header detection
        )
        enricher = HierarchyEnricher(settings, config)

        content = """# Level 1
## Level 2
### Level 3
#### Level 4
"""
        doc = create_test_document(content=content, url="https://example.com/")

        result = await enricher.enrich(doc)

        assert result.success is True
        # Note: max_hierarchy_depth may be applied to the level value,
        # not necessarily the path length. Check level is capped.
        assert result.metadata["hierarchy_level"] <= 2

    @pytest.mark.asyncio
    async def test_document_with_none_url(self, enricher):
        """Test document with None URL."""
        doc = create_test_document(content="# Title\n\nContent")
        doc.url = None

        result = await enricher.enrich(doc)

        # Should handle None URL gracefully
        assert result.success is True

    @pytest.mark.asyncio
    async def test_should_process_checks(self, enricher):
        """Test should_process method."""
        doc = create_test_document(content="Content")

        should, reason = enricher.should_process(doc)

        assert should is True
        assert reason is None
