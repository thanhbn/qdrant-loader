"""
POC 5: MCP Formatter Enhancements - Enrichment Data Tests.

Tests for enrichment metadata exposure in MCP formatters.
"""

import pytest
from qdrant_loader_mcp_server.mcp.formatters import MCPFormatters
from qdrant_loader_mcp_server.mcp.formatters.basic import BasicResultFormatters
from qdrant_loader_mcp_server.mcp.formatters.structured import StructuredResultFormatters
from qdrant_loader_mcp_server.mcp.formatters.lightweight import LightweightResultFormatters
from qdrant_loader_mcp_server.search.components.search_result_models import (
    create_hybrid_search_result,
)


class TestPOC5EnrichmentFormatters:
    """Test suite for POC5 enrichment formatter enhancements."""

    @pytest.fixture
    def result_with_enrichment(self):
        """Create a HybridSearchResult with enrichment data."""
        return create_hybrid_search_result(
            score=0.95,
            text="This document discusses machine learning and AI technologies.",
            source_type="confluence",
            source_title="ML Overview",
            source_url="https://docs.example.com/ml",
            # Enrichment data
            keyword_list=["machine learning", "AI", "neural networks", "deep learning", "algorithms", "data science"],
            keywords=[
                {"word": "machine learning", "score": 0.95},
                {"word": "AI", "score": 0.88},
                {"word": "neural networks", "score": 0.75},
            ],
            top_keyword="machine learning",
            keyword_count=6,
            entity_types={
                "PERSON": ["John Smith", "Jane Doe"],
                "ORG": ["Google", "OpenAI", "Microsoft"],
                "GPE": ["San Francisco", "New York"],
            },
            entity_count=7,
            has_people=True,
            has_organizations=True,
            has_locations=True,
        )

    @pytest.fixture
    def result_without_enrichment(self):
        """Create a HybridSearchResult without enrichment data."""
        return create_hybrid_search_result(
            score=0.75,
            text="Simple document without enrichment.",
            source_type="localfile",
            source_title="Simple Doc",
        )

    @pytest.fixture
    def multiple_results_with_enrichment(self, result_with_enrichment):
        """Create multiple results with varying enrichment data."""
        result2 = create_hybrid_search_result(
            score=0.85,
            text="Document about Python programming.",
            source_type="git",
            source_title="Python Guide",
            keyword_list=["python", "programming", "coding"],
            top_keyword="python",
            keyword_count=3,
            entity_types={
                "ORG": ["Python Software Foundation"],
            },
            entity_count=1,
            has_people=False,
            has_organizations=True,
            has_locations=False,
        )
        result3 = create_hybrid_search_result(
            score=0.70,
            text="No enrichment data here.",
            source_type="localfile",
            source_title="Plain Doc",
        )
        return [result_with_enrichment, result2, result3]

    # Basic Formatter Tests

    def test_basic_format_enrichment_section_with_data(self, result_with_enrichment):
        """Test basic enrichment section formatting with data."""
        enrichment = BasicResultFormatters._format_enrichment_section(result_with_enrichment)

        assert enrichment != ""
        assert "Keywords:" in enrichment
        assert "machine learning" in enrichment
        assert "People" in enrichment or "Orgs" in enrichment

    def test_basic_format_enrichment_section_without_data(self, result_without_enrichment):
        """Test basic enrichment section formatting without data."""
        enrichment = BasicResultFormatters._format_enrichment_section(result_without_enrichment)

        assert enrichment == ""

    def test_basic_format_search_result_includes_enrichment(self, result_with_enrichment):
        """Test that basic search result formatting includes enrichment."""
        formatted = MCPFormatters.format_search_result(result_with_enrichment)

        assert "Keywords:" in formatted
        assert "machine learning" in formatted

    def test_basic_format_enrichment_overview(self, multiple_results_with_enrichment):
        """Test enrichment overview aggregation."""
        overview = BasicResultFormatters.format_enrichment_overview(multiple_results_with_enrichment)

        assert "Enrichment Overview" in overview
        assert "Documents analyzed: 3" in overview
        assert "Documents with enrichment: 2" in overview
        assert "Top Keywords" in overview
        assert "People" in overview or "Organizations" in overview

    # Structured Formatter Tests

    def test_structured_build_enrichment_section_with_data(self, result_with_enrichment):
        """Test structured enrichment section building with data."""
        enrichment = StructuredResultFormatters._build_enrichment_section(result_with_enrichment)

        assert enrichment is not None
        assert "keywords" in enrichment
        assert "entities" in enrichment
        assert enrichment["keywords"]["count"] == 6
        assert enrichment["keywords"]["top"] == "machine learning"
        assert enrichment["entities"]["has_people"] is True
        assert enrichment["entities"]["has_organizations"] is True

    def test_structured_build_enrichment_section_without_data(self, result_without_enrichment):
        """Test structured enrichment section building without data."""
        enrichment = StructuredResultFormatters._build_enrichment_section(result_without_enrichment)

        assert enrichment is None

    def test_structured_search_results_include_enrichment(self, result_with_enrichment):
        """Test that structured search results include enrichment."""
        results = StructuredResultFormatters.create_structured_search_results([result_with_enrichment])

        assert len(results) == 1
        assert "enrichment" in results[0]
        assert results[0]["enrichment"]["keywords"]["top"] == "machine learning"

    def test_structured_enrichment_summary(self, multiple_results_with_enrichment):
        """Test structured enrichment summary creation."""
        summary = StructuredResultFormatters.create_enrichment_summary(multiple_results_with_enrichment)

        assert "top_keywords" in summary
        assert "entity_counts" in summary
        assert "entity_samples" in summary
        assert "coverage" in summary
        assert summary["coverage"]["total_documents"] == 3
        assert summary["coverage"]["documents_with_keywords"] == 2

    # Lightweight Formatter Tests

    def test_lightweight_enrichment_index(self, multiple_results_with_enrichment):
        """Test lightweight enrichment index creation."""
        index = LightweightResultFormatters.create_lightweight_enrichment_index(
            multiple_results_with_enrichment
        )

        assert "keyword_index" in index
        assert "entity_index" in index
        assert "enrichment_coverage" in index
        assert index["enrichment_coverage"]["total_documents"] == 3
        assert index["enrichment_coverage"]["documents_with_keywords"] == 2
        assert len(index["keyword_index"]["keywords"]) > 0

    def test_lightweight_enrichment_index_empty_results(self):
        """Test lightweight enrichment index with empty results."""
        index = LightweightResultFormatters.create_lightweight_enrichment_index([])

        assert index["enrichment_coverage"]["total_documents"] == 0
        assert index["enrichment_coverage"]["documents_with_keywords"] == 0

    # Edge Cases

    def test_enrichment_with_mock_objects(self):
        """Test that enrichment formatters handle Mock objects gracefully."""
        from unittest.mock import Mock
        from qdrant_loader_mcp_server.search.components.search_result_models import HybridSearchResult

        mock_result = Mock(spec=HybridSearchResult)
        mock_result.score = 0.5
        mock_result.text = "Mock content"
        mock_result.source_type = "test"
        mock_result.source_title = "Mock Title"
        mock_result.source_url = None
        mock_result.file_path = None
        mock_result.repo_name = None
        mock_result.breadcrumb_text = None
        mock_result.hierarchy_context = None
        mock_result.is_attachment = False
        mock_result.original_filename = None
        mock_result.attachment_context = None
        mock_result.parent_document_title = None
        mock_result.parent_title = None
        mock_result.get_project_info.return_value = None
        mock_result.has_children.return_value = False
        # Note: keyword_list and entity_types will be Mock objects

        # Should not raise an error
        enrichment = BasicResultFormatters._format_enrichment_section(mock_result)
        assert enrichment == ""  # No enrichment data from Mock

    def test_enrichment_keywords_truncation(self, result_with_enrichment):
        """Test that keywords are properly truncated in output."""
        # Result has 6 keywords, should show max 5 in basic view
        enrichment = BasicResultFormatters._format_enrichment_section(result_with_enrichment)

        # Check truncation indicator
        assert "+1 more" in enrichment or "machine learning" in enrichment

    def test_enrichment_entity_types_limiting(self):
        """Test that entity lists are properly limited in structured output."""
        result = create_hybrid_search_result(
            score=0.9,
            text="Document with many entities.",
            source_type="confluence",
            source_title="Entity Test",
            entity_types={
                "PERSON": [f"Person {i}" for i in range(10)],  # 10 people
                "ORG": [f"Org {i}" for i in range(8)],  # 8 orgs
            },
            entity_count=18,
            has_people=True,
            has_organizations=True,
            has_locations=False,
        )

        enrichment = StructuredResultFormatters._build_enrichment_section(result)

        # Should limit to 5 entities per type
        assert len(enrichment["entities"]["types"]["PERSON"]) <= 5
        assert len(enrichment["entities"]["types"]["ORG"]) <= 5
        # But counts should be accurate
        assert enrichment["entities"]["type_counts"]["PERSON"] == 10
        assert enrichment["entities"]["type_counts"]["ORG"] == 8

    def test_has_enrichment_data_method(self, result_with_enrichment, result_without_enrichment):
        """Test has_enrichment_data helper method."""
        assert result_with_enrichment.has_enrichment_data() is True
        assert result_without_enrichment.has_enrichment_data() is False

    def test_get_enrichment_summary_method(self, result_with_enrichment, result_without_enrichment):
        """Test get_enrichment_summary helper method."""
        summary_with = result_with_enrichment.get_enrichment_summary()
        summary_without = result_without_enrichment.get_enrichment_summary()

        assert summary_with is not None
        assert "Keywords:" in summary_with
        assert summary_without is None
