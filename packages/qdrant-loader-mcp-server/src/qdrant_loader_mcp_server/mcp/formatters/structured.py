"""
Structured Result Formatters - Complex Data Structure Formatting.

This module handles the creation of complex, structured result formats
for MCP responses that require detailed organization and presentation.

POC5: Enhanced to include enrichment metadata in structured outputs.
"""

from typing import Any

# Backward-compatible import for HybridSearchResult across branches
try:  # Prefer current location
    from ...search.components.search_result_models import (
        HybridSearchResult,  # type: ignore[assignment]
    )
except Exception:  # ImportError | ModuleNotFoundError
    # Fallback for older layout
    from ...search.components.models.hybrid import (
        HybridSearchResult,  # type: ignore[assignment]
    )
from .utils import FormatterUtils


class StructuredResultFormatters:
    """Handles structured result formatting operations."""

    @staticmethod
    def _build_enrichment_section(result: HybridSearchResult) -> dict[str, Any] | None:
        """POC5: Build enrichment section for a search result.

        Args:
            result: HybridSearchResult with potential enrichment data

        Returns:
            Enrichment dict or None if no enrichment data
        """
        # Get and validate enrichment data
        keyword_list = getattr(result, "keyword_list", None)
        entity_types = getattr(result, "entity_types", None)

        # Ensure proper types
        if not isinstance(keyword_list, list):
            keyword_list = []
        if not isinstance(entity_types, dict):
            entity_types = {}

        has_keywords = len(keyword_list) > 0
        has_entities = len(entity_types) > 0

        if not has_keywords and not has_entities:
            return None

        enrichment: dict[str, Any] = {}

        # Keywords section
        if has_keywords:
            top_keyword = getattr(result, "top_keyword", None)
            if not isinstance(top_keyword, str):
                top_keyword = None
            keywords_full = getattr(result, "keywords_full", None)
            if not isinstance(keywords_full, list):
                keywords_full = []

            enrichment["keywords"] = {
                "list": keyword_list[:10],  # Limit to 10 keywords in structured output
                "top": top_keyword,
                "count": len(keyword_list),
            }

            # Include top 5 keywords with scores if available
            if keywords_full:
                enrichment["keywords"]["top_with_scores"] = [
                    {"word": kw.get("word", ""), "score": round(kw.get("score", 0), 4)}
                    for kw in keywords_full[:5]
                    if isinstance(kw, dict)
                ]

        # Entities section
        if has_entities:
            entity_count = getattr(result, "entity_count", 0)
            if not isinstance(entity_count, int):
                entity_count = 0

            has_people = getattr(result, "has_people", False) is True
            has_organizations = getattr(result, "has_organizations", False) is True
            has_locations = getattr(result, "has_locations", False) is True

            enrichment["entities"] = {
                "types": {
                    ent_type: entities[:5]  # Limit to 5 entities per type
                    for ent_type, entities in entity_types.items()
                    if isinstance(entities, list)
                },
                "type_counts": {
                    ent_type: len(entities)
                    for ent_type, entities in entity_types.items()
                    if isinstance(entities, list)
                },
                "has_people": has_people,
                "has_organizations": has_organizations,
                "has_locations": has_locations,
                "total_count": entity_count,
            }

        return enrichment

    @staticmethod
    def create_structured_search_results(
        results: list[HybridSearchResult],
        query: str = "",
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Create structured search results as a list of formatted results.

        POC5: Now includes enrichment metadata when available.
        """
        formatted_results: list[dict[str, Any]] = []
        for result in results[:max_results]:
            raw_text = getattr(result, "text", None)
            if raw_text is None:
                normalized_text = ""
            elif isinstance(raw_text, str):
                normalized_text = raw_text
            else:
                normalized_text = str(raw_text)

            formatted_result: dict[str, Any] = {
                "document_id": getattr(result, "document_id", ""),
                "title": (
                    result.get_display_title()
                    if hasattr(result, "get_display_title")
                    else None
                )
                or getattr(result, "source_title", None)
                or "Untitled",
                "content": normalized_text,
                "content_snippet": (
                    normalized_text[:300] + "..."
                    if len(normalized_text) > 300
                    else normalized_text
                ),
                "source_type": getattr(result, "source_type", "unknown"),
                "source_url": getattr(result, "source_url", None),
                "file_path": getattr(result, "file_path", None),
                "score": getattr(result, "score", 0.0),
                "created_at": getattr(result, "created_at", None),
                "updated_at": getattr(result, "updated_at", None),
                "metadata": {
                    "breadcrumb": getattr(result, "breadcrumb_text", None),
                    "hierarchy_context": getattr(result, "hierarchy_context", None),
                    "project_info": (
                        result.get_project_info()
                        if hasattr(result, "get_project_info")
                        else None
                    ),
                    "project_id": (
                        ""
                        if getattr(result, "project_id", None) is None
                        else str(getattr(result, "project_id", ""))
                    ),
                    "file_path": getattr(result, "file_path", None),
                    "word_count": getattr(result, "word_count", None),
                    "chunk_index": getattr(result, "chunk_index", None),
                    "total_chunks": getattr(result, "total_chunks", None),
                    "is_attachment": getattr(result, "is_attachment", False),
                    "depth": FormatterUtils.extract_synthetic_depth(result),
                    "has_children": FormatterUtils.extract_has_children(result),
                },
            }

            # POC5: Add enrichment section if available
            enrichment = StructuredResultFormatters._build_enrichment_section(result)
            if enrichment:
                formatted_result["enrichment"] = enrichment

            formatted_results.append(formatted_result)

        return formatted_results

    @staticmethod
    def create_enrichment_summary(
        results: list[HybridSearchResult],
    ) -> dict[str, Any]:
        """POC5: Create aggregated enrichment summary across all results.

        Provides an overview of keywords and entities found across search results
        for faceted navigation and analysis.

        Args:
            results: List of search results to summarize

        Returns:
            Dictionary with aggregated enrichment information
        """
        all_keywords: dict[str, int] = {}
        all_entities: dict[str, set[str]] = {
            "PERSON": set(),
            "ORG": set(),
            "GPE": set(),
            "LOC": set(),
            "FAC": set(),
        }

        docs_with_keywords = 0
        docs_with_entities = 0

        for result in results:
            # Aggregate keywords - validate type
            keyword_list = getattr(result, "keyword_list", None)
            if not isinstance(keyword_list, list):
                keyword_list = []
            if keyword_list:
                docs_with_keywords += 1
                for kw in keyword_list:
                    if isinstance(kw, str):
                        all_keywords[kw] = all_keywords.get(kw, 0) + 1

            # Aggregate entities - validate type
            entity_types = getattr(result, "entity_types", None)
            if not isinstance(entity_types, dict):
                entity_types = {}
            if entity_types:
                docs_with_entities += 1
                for ent_type, entities in entity_types.items():
                    if ent_type in all_entities and isinstance(entities, list):
                        all_entities[ent_type].update(
                            e for e in entities if isinstance(e, str)
                        )

        # Sort keywords by frequency
        top_keywords = sorted(all_keywords.items(), key=lambda x: -x[1])[:20]

        return {
            "top_keywords": [
                {"keyword": kw, "frequency": freq} for kw, freq in top_keywords
            ],
            "entity_counts": {
                "people": len(all_entities["PERSON"]),
                "organizations": len(all_entities["ORG"]),
                "locations": len(all_entities["GPE"]) + len(all_entities["LOC"]) + len(all_entities["FAC"]),
            },
            "entity_samples": {
                "people": sorted(all_entities["PERSON"])[:10],
                "organizations": sorted(all_entities["ORG"])[:10],
                "locations": sorted(
                    all_entities["GPE"] | all_entities["LOC"] | all_entities["FAC"]
                )[:10],
            },
            "coverage": {
                "documents_with_keywords": docs_with_keywords,
                "documents_with_entities": docs_with_entities,
                "total_documents": len(results),
                "keyword_coverage_pct": round(
                    docs_with_keywords / len(results) * 100, 1
                ) if results else 0,
                "entity_coverage_pct": round(
                    docs_with_entities / len(results) * 100, 1
                ) if results else 0,
            },
        }

    @staticmethod
    def create_structured_hierarchy_results(
        organized_results: dict[str, list[HybridSearchResult]],
        query: str = "",
        max_depth: int = 100,
    ) -> dict[str, Any]:
        """Create structured hierarchical results with full organization.

        Parameters:
            organized_results: Mapping of group name to list of results.
            query: Original query string.
            max_depth: Maximum tree depth to attach children. This prevents
                stack overflows on very deep or cyclic hierarchies. Root level
                starts at depth 1. Children beyond max_depth are not attached.
        """
        hierarchy_data = []

        for group_name, results in organized_results.items():
            # Build hierarchical structure
            root_documents = []
            child_map = {}

            for result in results:
                parent_id = FormatterUtils.extract_synthetic_parent_id(result)
                raw_text = getattr(result, "text", None)
                if raw_text is None:
                    normalized_text = ""
                elif isinstance(raw_text, str):
                    normalized_text = raw_text
                else:
                    normalized_text = str(raw_text)

                doc_data: dict[str, Any] = {
                    "document_id": getattr(result, "document_id", ""),
                    "title": (
                        result.get_display_title()
                        if hasattr(result, "get_display_title")
                        else None
                    )
                    or getattr(result, "source_title", None)
                    or "Untitled",
                    "content_snippet": (
                        normalized_text[:200] + "..."
                        if len(normalized_text) > 200
                        else normalized_text
                    ),
                    "source_type": getattr(result, "source_type", "unknown"),
                    "score": getattr(result, "score", 0.0),
                    "depth": FormatterUtils.extract_synthetic_depth(result),
                    "parent_id": parent_id,
                    "has_children": FormatterUtils.extract_has_children(result),
                    "children": [],
                    "metadata": {
                        "breadcrumb": FormatterUtils.extract_synthetic_breadcrumb(
                            result
                        ),
                        "hierarchy_context": getattr(result, "hierarchy_context", None),
                        "file_path": getattr(result, "file_path", None),
                    },
                }

                # POC5: Add enrichment to hierarchy documents
                enrichment = StructuredResultFormatters._build_enrichment_section(result)
                if enrichment:
                    doc_data["enrichment"] = enrichment

                if parent_id:
                    if parent_id not in child_map:
                        child_map[parent_id] = []
                    child_map[parent_id].append(doc_data)
                else:
                    root_documents.append(doc_data)

            # Attach children to parents using an explicit stack and depth cap
            # to avoid unbounded recursion in deep hierarchies
            def attach_children_iterative(
                root_docs: list[dict[str, Any]],
                depth_limit: int,
                child_lookup: dict[str, list[dict[str, Any]]],
            ) -> None:
                if depth_limit <= 0:
                    return
                stack: list[tuple[dict[str, Any], int]] = [
                    (doc, 1) for doc in root_docs
                ]
                # Track visited to avoid cycles
                visited: set[str] = set()
                while stack:
                    current_doc, current_depth = stack.pop()
                    doc_id = current_doc.get("document_id")
                    if not doc_id or doc_id in visited:
                        continue
                    visited.add(doc_id)
                    # Only attach children if within depth limit
                    if current_depth >= depth_limit:
                        continue
                    children = child_lookup.get(doc_id)
                    if children:
                        current_doc["children"] = children
                        for child in children:
                            stack.append((child, current_depth + 1))

            attach_children_iterative(root_documents, max_depth, child_map)

            hierarchy_data.append(
                {
                    "group_name": FormatterUtils.generate_clean_group_name(
                        group_name, results
                    ),
                    "documents": root_documents,
                    "total_documents": len(results),
                    "max_depth": max(
                        (FormatterUtils.extract_synthetic_depth(r) for r in results),
                        default=0,
                    ),
                }
            )

        return {
            "query": query,
            "hierarchy_groups": hierarchy_data,
            "total_groups": len(organized_results),
            "total_documents": sum(
                len(results) for results in organized_results.values()
            ),
        }

    @staticmethod
    def create_structured_attachment_results(
        filtered_results: list[HybridSearchResult],
        attachment_filter: dict[str, Any],
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Create structured attachment results with detailed organization."""
        # Filter only attachment results
        attachment_results = [
            result
            for result in filtered_results
            if getattr(result, "is_attachment", False)
        ]

        # Group by file type
        organized_attachments = {}
        for result in attachment_results:
            file_type = FormatterUtils.extract_file_type_minimal(result)
            if file_type not in organized_attachments:
                organized_attachments[file_type] = []
            organized_attachments[file_type].append(result)

        attachment_data = []
        for file_type, results in organized_attachments.items():
            attachments = []

            for result in results:
                raw_text = getattr(result, "text", None)
                if raw_text is None:
                    normalized_text = ""
                elif isinstance(raw_text, str):
                    normalized_text = raw_text
                else:
                    normalized_text = str(raw_text)

                attachment_info = {
                    "document_id": getattr(result, "document_id", ""),
                    "filename": FormatterUtils.extract_safe_filename(result),
                    "file_type": FormatterUtils.extract_file_type_minimal(result),
                    "source_type": getattr(result, "source_type", "unknown"),
                    "score": getattr(result, "score", 0.0),
                    "content_snippet": (
                        normalized_text[:150] + "..."
                        if len(normalized_text) > 150
                        else normalized_text
                    ),
                }

                if include_metadata:
                    attachment_info["metadata"] = {
                        "original_filename": getattr(result, "original_filename", None),
                        "attachment_context": getattr(
                            result, "attachment_context", None
                        ),
                        "parent_document_title": getattr(
                            result, "parent_document_title", None
                        ),
                        "file_path": getattr(result, "file_path", None),
                        "source_url": getattr(result, "source_url", None),
                        "breadcrumb": getattr(result, "breadcrumb_text", None),
                    }

                attachments.append(attachment_info)

            attachment_data.append(
                {
                    "group_name": file_type,
                    "file_types": [file_type],
                    "attachments": attachments,
                    "total_attachments": len(attachments),
                    "metadata": (
                        {
                            "avg_score": (
                                sum(getattr(r, "score", 0) for r in results)
                                / len(results)
                                if results
                                else 0
                            ),
                            "source_types": list(
                                {getattr(r, "source_type", "unknown") for r in results}
                            ),
                        }
                        if include_metadata
                        else {}
                    ),
                }
            )

        def _normalized_text(r: Any) -> str:
            rt = getattr(r, "text", None)
            if rt is None:
                return ""
            if isinstance(rt, str):
                return rt
            return str(rt)

        return {
            "results": [
                {
                    "document_id": getattr(result, "document_id", ""),
                    "title": (
                        result.get_display_title()
                        if hasattr(result, "get_display_title")
                        else None
                    )
                    or getattr(result, "source_title", None)
                    or "Untitled",
                    "attachment_info": {
                        "filename": FormatterUtils.extract_safe_filename(result),
                        "file_type": FormatterUtils.extract_file_type_minimal(result),
                        "file_size": getattr(result, "file_size", None),
                    },
                    "source_type": getattr(result, "source_type", "unknown"),
                    "score": getattr(result, "score", 0.0),
                    "content_snippet": (
                        _normalized_text(result)[:150] + "..."
                        if len(_normalized_text(result)) > 150
                        else _normalized_text(result)
                    ),
                    "metadata": (
                        {
                            "file_path": getattr(result, "file_path", None),
                            "source_url": getattr(result, "source_url", None),
                            "breadcrumb": getattr(result, "breadcrumb_text", None),
                            "parent_document_title": getattr(
                                result, "parent_document_title", None
                            ),
                        }
                        if include_metadata
                        else {}
                    ),
                }
                for result in attachment_results
            ],
            "total_found": len(attachment_results),
            "attachment_summary": {
                "total_attachments": len(attachment_results),
                "file_types": list(organized_attachments.keys()),
                "groups_created": len(organized_attachments),
            },
            # Keep additional fields for backward compatibility
            "attachment_groups": attachment_data,
            "total_groups": len(organized_attachments),
            "total_attachments": len(attachment_results),
            "filter_criteria": attachment_filter,
            "metadata": (
                {
                    "all_file_types": list(organized_attachments.keys()),
                    "largest_group_size": max(
                        (len(results) for results in organized_attachments.values()),
                        default=0,
                    ),
                }
                if include_metadata
                else {}
            ),
        }
