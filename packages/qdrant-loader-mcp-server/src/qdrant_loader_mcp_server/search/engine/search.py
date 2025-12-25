"""
Search Operations - Main Search Functionality.

This module implements the core search operations including
basic document search using the hybrid search engine.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SearchEngine

from ...utils.logging import LoggingConfig
from ..components.models.hybrid import HybridSearchResult

logger = LoggingConfig.get_logger(__name__)


class SearchOperations:
    """Handles core search operations."""

    def __init__(self, engine: "SearchEngine"):
        """Initialize with search engine reference."""
        self.engine = engine

    # TODO [L2][AIKH-481][AIKH-553][TC-SEARCH-002]: Core search operation
    # Use Case: UC-001 - Basic Semantic Search
    # Business Rule: Delegates to HybridSearchEngine with limit constraint
    # Data Flow: SearchHandler -> SearchOperations -> HybridSearchEngine
    # Test: test_search_operations
    # -----------------------------------------------------------
    async def search(
        self,
        query: str,
        source_types: list[str] | None = None,
        limit: int = 5,
        project_ids: list[str] | None = None,
    ) -> list[HybridSearchResult]:
        """Search for documents using hybrid search.

        Args:
            query: Search query text
            source_types: Optional list of source types to filter by
            limit: Maximum number of results to return
            project_ids: Optional list of project IDs to filter by
        """
        hybrid = getattr(self.engine, "hybrid_search", None)
        if not hybrid:
            raise RuntimeError("Search engine not initialized")

        logger.debug(
            "Performing search",
            query=query,
            source_types=source_types,
            limit=limit,
            project_ids=project_ids,
        )

        try:
            # TODO [L1][AIKH-481][AIKH-553][TC-SEARCH-002][TC-SEARCH-006]: Pass filters to hybrid search
            # Use Case: UC-002, UC-004 - Limit and Project Filters
            # Business Rule: limit controls result count, project_ids restricts scope
            # Data Flow: SearchOps -> hybrid.search(limit, project_ids)
            # -----------------------------------------------------------
            results = await hybrid.search(
                query=query,
                source_types=source_types,
                limit=limit,
                project_ids=project_ids,
            )
            # -----------------------------------------------------------

            logger.info(
                "Search completed",
                query=query,
                result_count=len(results),
                project_ids=project_ids,
            )

            return results
        except Exception as e:
            logger.error("Search failed", error=str(e), query=query)
            raise
    # -----------------------------------------------------------
