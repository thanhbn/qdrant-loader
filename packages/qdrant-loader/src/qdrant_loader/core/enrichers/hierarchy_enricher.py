"""Hierarchy enricher for tracking parent-child document relationships.

POC3-001, POC3-002: Haystack-inspired hierarchy enricher.

This module provides the HierarchyEnricher class that detects hierarchical
relationships between documents based on:

1. Header structure (Markdown/HTML headers)
2. URL path structure
3. Existing parent metadata

The enricher runs with HIGHEST priority to establish hierarchy before
other enrichers process the document.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .base_enricher import (
    BaseEnricher,
    EnricherConfig,
    EnricherPriority,
    EnricherResult,
)

if TYPE_CHECKING:
    from qdrant_loader.config import Settings
    from qdrant_loader.core.document import Document


# Regex patterns for header detection
MARKDOWN_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*#*)?$", re.MULTILINE)
HTML_HEADER_PATTERN = re.compile(
    r"<h([1-6])(?:\s+[^>]*)?>(.+?)</h\1>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class HierarchyMetadata:
    """Hierarchy metadata structure.

    Attributes:
        parent_id: ID of the parent document or chunk
        hierarchy_level: Depth in hierarchy (0 = root)
        hierarchy_path: Path from root to current document
        section_title: Title extracted from headers
        is_root: Whether this is a root document
        source_document_id: ID of the original source document
        sibling_ids: IDs of sibling documents (same parent)
    """

    parent_id: str | None = None
    hierarchy_level: int = 0
    hierarchy_path: list[str] = field(default_factory=list)
    section_title: str | None = None
    is_root: bool = True
    source_document_id: str | None = None
    sibling_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for metadata storage."""
        return {
            "parent_id": self.parent_id,
            "hierarchy_level": self.hierarchy_level,
            "hierarchy_path": self.hierarchy_path,
            "section_title": self.section_title,
            "is_root": self.is_root,
            "source_document_id": self.source_document_id,
            "sibling_ids": self.sibling_ids,
        }


@dataclass
class HierarchyEnricherConfig(EnricherConfig):
    """Configuration for HierarchyEnricher.

    Attributes:
        detect_from_headers: Enable header-based hierarchy detection
        detect_from_url: Enable URL path-based hierarchy detection
        track_siblings: Include sibling information in metadata
        max_hierarchy_depth: Maximum hierarchy depth to track
        include_html_headers: Also parse HTML h1-h6 tags
        extract_first_header_as_title: Use first header as section_title
    """

    detect_from_headers: bool = True
    detect_from_url: bool = True
    track_siblings: bool = False  # Disabled by default (requires cross-doc analysis)
    max_hierarchy_depth: int = 10
    include_html_headers: bool = True
    extract_first_header_as_title: bool = True

    def __post_init__(self):
        """Set priority to HIGHEST so hierarchy runs first."""
        self.priority = EnricherPriority.HIGHEST


class HierarchyEnricher(BaseEnricher):
    """Enricher that extracts hierarchical relationships from documents.

    This enricher detects parent-child relationships to enable:
    - Parent-child navigation in search results
    - Hierarchy-aware context expansion
    - Section-level document retrieval
    - Breadcrumb-style navigation

    Example usage:
        enricher = HierarchyEnricher(settings)
        result = await enricher.enrich(document)

        # Result contains:
        # - parent_id: Reference to parent document
        # - hierarchy_level: Depth in document tree
        # - hierarchy_path: Path from root
        # - section_title: Extracted section name
    """

    def __init__(
        self,
        settings: "Settings",
        config: HierarchyEnricherConfig | None = None,
    ):
        """Initialize the HierarchyEnricher.

        Args:
            settings: Application settings
            config: Enricher-specific configuration
        """
        config = config or HierarchyEnricherConfig()
        super().__init__(settings, config)
        self.hierarchy_config = config

    @property
    def name(self) -> str:
        """Unique identifier for this enricher."""
        return "hierarchy_enricher"

    async def enrich(self, document: "Document") -> EnricherResult:
        """Extract hierarchy metadata from document.

        Detects hierarchy from multiple sources in order of priority:
        1. Existing parent_id in metadata (from source connector)
        2. Header structure in content
        3. URL path structure

        Args:
            document: The document to enrich

        Returns:
            EnricherResult with hierarchy metadata
        """
        try:
            hierarchy = HierarchyMetadata()

            # Preserve existing source_document_id if present
            if document.metadata.get("source_id"):
                hierarchy.source_document_id = document.metadata["source_id"]
            elif hasattr(document, "id") and document.id:
                hierarchy.source_document_id = str(document.id)

            # Strategy 1: Check for existing parent metadata (from connector)
            if document.metadata.get("parent_id"):
                hierarchy.parent_id = document.metadata["parent_id"]
                hierarchy.is_root = False

            # Strategy 2: Detect from headers in content
            if self.hierarchy_config.detect_from_headers and document.content:
                header_info = self._detect_from_headers(document.content)
                hierarchy = self._merge_hierarchy(hierarchy, header_info)

            # Strategy 3: Detect from URL structure
            if self.hierarchy_config.detect_from_url and document.url:
                url_info = self._detect_from_url(document.url)
                hierarchy = self._merge_hierarchy(hierarchy, url_info)

            # Limit hierarchy depth
            if hierarchy.hierarchy_level > self.hierarchy_config.max_hierarchy_depth:
                hierarchy.hierarchy_level = self.hierarchy_config.max_hierarchy_depth
                hierarchy.hierarchy_path = hierarchy.hierarchy_path[
                    : self.hierarchy_config.max_hierarchy_depth + 1
                ]

            # Build result metadata
            metadata = hierarchy.to_dict()

            # Remove sibling_ids if not tracking
            if not self.hierarchy_config.track_siblings:
                metadata.pop("sibling_ids", None)

            return EnricherResult(metadata=metadata)

        except Exception as e:
            self.logger.warning(f"Hierarchy extraction failed: {e}")
            return EnricherResult.error_result(str(e))

    def _detect_from_headers(self, content: str) -> HierarchyMetadata:
        """Parse headers to extract hierarchy information.

        Analyzes Markdown (#) and HTML (<h1>-<h6>) headers to determine:
        - Section title (from first header)
        - Hierarchy level (from header depth)
        - Hierarchy path (from nested headers)

        Args:
            content: Document content to parse

        Returns:
            HierarchyMetadata with detected information
        """
        hierarchy = HierarchyMetadata()
        headers: list[tuple[int, str]] = []

        # Extract Markdown headers (# Header)
        for match in MARKDOWN_HEADER_PATTERN.finditer(content):
            level = len(match.group(1))  # Number of # characters
            title = match.group(2).strip()
            if title:
                headers.append((level, title))

        # Extract HTML headers (<h1>...</h1>)
        if self.hierarchy_config.include_html_headers:
            for match in HTML_HEADER_PATTERN.finditer(content):
                level = int(match.group(1))
                # Clean HTML tags from title
                title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                if title:
                    headers.append((level, title))

        if not headers:
            return hierarchy

        # Sort headers by their position (already in order from regex)
        # Use first header as section title
        if self.hierarchy_config.extract_first_header_as_title:
            first_level, first_title = headers[0]
            hierarchy.section_title = first_title
            hierarchy.hierarchy_level = first_level - 1  # h1 = level 0

        # Build hierarchy path from headers
        # This tracks the path through nested headers
        path: list[str] = []
        current_level = 0

        for level, title in headers:
            # Adjust path based on header level
            if level > current_level:
                # Going deeper - add to path
                path.append(title)
            elif level == current_level:
                # Same level - replace last entry
                if path:
                    path[-1] = title
                else:
                    path.append(title)
            else:
                # Going up - truncate path and add new
                depth = level - 1
                path = path[:depth]
                path.append(title)

            current_level = level

        hierarchy.hierarchy_path = path

        # If we have headers, this is likely not a root (has structure)
        # But we only mark as non-root if there's a clear parent indicator
        if len(headers) > 1:
            # Multiple headers suggest this is a section within a larger doc
            pass

        return hierarchy

    def _detect_from_url(self, url: str) -> HierarchyMetadata:
        """Parse URL to extract hierarchy from path structure.

        Analyzes URL path segments to build hierarchy:
        - /docs/getting-started/installation -> level 2, path = ["docs", "getting-started", "installation"]

        Args:
            url: Document URL to parse

        Returns:
            HierarchyMetadata with URL-based hierarchy
        """
        hierarchy = HierarchyMetadata()

        try:
            parsed = urlparse(url)
            path = parsed.path

            # Remove leading/trailing slashes and split
            path = path.strip("/")
            if not path:
                hierarchy.is_root = True
                return hierarchy

            # Split path into segments
            segments = [s for s in path.split("/") if s]

            # Filter out common non-content segments
            filtered_segments = [
                s
                for s in segments
                if s.lower()
                not in {"index", "index.html", "index.md", "readme", "readme.md"}
            ]

            if not filtered_segments:
                hierarchy.is_root = True
                return hierarchy

            # Build hierarchy from path
            hierarchy.hierarchy_path = filtered_segments
            hierarchy.hierarchy_level = len(filtered_segments) - 1

            # Use last segment as section title (cleaned)
            last_segment = filtered_segments[-1]
            # Remove file extensions
            last_segment = re.sub(
                r"\.(html?|md|rst|txt)$", "", last_segment, flags=re.I
            )
            # Convert dashes/underscores to spaces and title case
            section_title = last_segment.replace("-", " ").replace("_", " ").strip()
            if section_title:
                hierarchy.section_title = section_title.title()

            # Determine if root based on path depth
            hierarchy.is_root = len(filtered_segments) <= 1

            # Parent path would be one level up
            if len(filtered_segments) > 1:
                # We can't set parent_id without knowing the actual document ID
                # But we can indicate this document has a parent
                hierarchy.is_root = False

        except Exception as e:
            self.logger.debug(f"URL parsing failed for {url}: {e}")

        return hierarchy

    def _merge_hierarchy(
        self,
        base: HierarchyMetadata,
        new: HierarchyMetadata,
    ) -> HierarchyMetadata:
        """Merge hierarchy information, preferring existing values.

        When merging, existing (non-default) values take precedence.
        This allows source-provided metadata to override detected values.

        Args:
            base: Existing hierarchy metadata
            new: New hierarchy metadata to merge

        Returns:
            Merged HierarchyMetadata
        """
        # parent_id: prefer existing if set
        if not base.parent_id and new.parent_id:
            base.parent_id = new.parent_id

        # hierarchy_level: take higher of two (more specific)
        if new.hierarchy_level > base.hierarchy_level:
            base.hierarchy_level = new.hierarchy_level

        # hierarchy_path: prefer longer path (more detail)
        if len(new.hierarchy_path) > len(base.hierarchy_path):
            base.hierarchy_path = new.hierarchy_path

        # section_title: prefer first non-empty
        if not base.section_title and new.section_title:
            base.section_title = new.section_title

        # is_root: only root if both say root
        base.is_root = base.is_root and new.is_root

        # source_document_id: prefer existing
        if not base.source_document_id and new.source_document_id:
            base.source_document_id = new.source_document_id

        # sibling_ids: merge lists (dedupe)
        all_siblings = set(base.sibling_ids) | set(new.sibling_ids)
        base.sibling_ids = list(all_siblings)

        return base

    def get_metadata_keys(self) -> list[str]:
        """Return the list of metadata keys this enricher produces."""
        keys = [
            "parent_id",
            "hierarchy_level",
            "hierarchy_path",
            "section_title",
            "is_root",
            "source_document_id",
        ]
        if self.hierarchy_config.track_siblings:
            keys.append("sibling_ids")
        return keys
