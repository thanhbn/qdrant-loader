"""
Basic Result Formatters - Core Search Result Formatting.

This module handles basic formatting of search results, attachments,
and hierarchical results for display in the MCP interface.

POC5: Enhanced to include enrichment metadata in text output.
"""

from ...search.components.search_result_models import HybridSearchResult


class BasicResultFormatters:
    """Handles basic result formatting operations."""

    @staticmethod
    def _format_enrichment_section(result: HybridSearchResult) -> str:
        """POC5: Format enrichment data for display.

        Args:
            result: HybridSearchResult with potential enrichment data

        Returns:
            Formatted string with enrichment info, empty if no data
        """
        sections = []

        # Keywords section - ensure we have a proper list
        keyword_list = getattr(result, "keyword_list", None)
        if keyword_list and isinstance(keyword_list, list) and len(keyword_list) > 0:
            keywords_display = ", ".join(str(k) for k in keyword_list[:5])
            if len(keyword_list) > 5:
                keywords_display += f" (+{len(keyword_list) - 5} more)"
            sections.append(f"🏷️ Keywords: {keywords_display}")

        # Entity summary with counts - validate boolean values
        entity_parts = []
        has_people = getattr(result, "has_people", False)
        has_organizations = getattr(result, "has_organizations", False)
        has_locations = getattr(result, "has_locations", False)
        entity_types = getattr(result, "entity_types", None)

        # Ensure entity_types is a proper dict
        if not isinstance(entity_types, dict):
            entity_types = {}
        # Ensure boolean flags are actual booleans
        has_people = has_people is True
        has_organizations = has_organizations is True
        has_locations = has_locations is True

        if has_people:
            people = entity_types.get("PERSON", [])
            count = len(people)
            if count > 0 and people:
                samples = ", ".join(people[:2])
                entity_parts.append(f"👤 People ({count}): {samples}")
            else:
                entity_parts.append(f"👤 People ({count})")

        if has_organizations:
            orgs = entity_types.get("ORG", [])
            count = len(orgs)
            if count > 0 and orgs:
                samples = ", ".join(orgs[:2])
                entity_parts.append(f"🏢 Orgs ({count}): {samples}")
            else:
                entity_parts.append(f"🏢 Orgs ({count})")

        if has_locations:
            locs = entity_types.get("GPE", []) + entity_types.get("LOC", []) + entity_types.get("FAC", [])
            count = len(locs)
            if count > 0 and locs:
                samples = ", ".join(locs[:2])
                entity_parts.append(f"📍 Locations ({count}): {samples}")
            else:
                entity_parts.append(f"📍 Locations ({count})")

        if entity_parts:
            sections.append(" | ".join(entity_parts))

        return "\n".join(sections) if sections else ""

    @staticmethod
    def _format_common_fields(
        result: HybridSearchResult, is_attachment_view: bool = False
    ) -> str:
        """Build the base formatted string for a search result.

        This consolidates shared formatting between standard search results and
        attachment-focused views while preserving the original output order and
        conditional branches.
        """
        formatted_result = f"Score: {result.score}\n"
        formatted_result += f"Text: {result.text}\n"
        formatted_result += f"Source: {result.source_type}"

        if result.source_title:
            formatted_result += f" - {result.source_title}"

        # Project information (only shown in non-attachment view to preserve behavior)
        if not is_attachment_view:
            project_info = result.get_project_info()
            if project_info:
                formatted_result += f"\n🏗️ {project_info}"

        # Attachment info (shown if viewing attachments or the result itself is an attachment)
        if is_attachment_view or result.is_attachment:
            formatted_result += "\n📎 Attachment"
            if result.original_filename:
                formatted_result += f": {result.original_filename}"
            if result.attachment_context:
                formatted_result += f"\n📋 {result.attachment_context}"
            if result.parent_document_title:
                formatted_result += f"\n📄 Attached to: {result.parent_document_title}"

        # POC5: Add enrichment section
        enrichment = BasicResultFormatters._format_enrichment_section(result)
        if enrichment:
            formatted_result += f"\n{enrichment}"

        # Confluence breadcrumb path
        if result.source_type == "confluence" and result.breadcrumb_text:
            formatted_result += f"\n📍 Path: {result.breadcrumb_text}"

        # Source URL appended inline
        if result.source_url:
            formatted_result += f" ({result.source_url})"

        if result.file_path:
            formatted_result += f"\nFile: {result.file_path}"

        if result.repo_name:
            formatted_result += f"\nRepo: {result.repo_name}"

        # Additional hierarchy info for Confluence
        hierarchy_context = getattr(result, "hierarchy_context", None)
        if result.source_type == "confluence" and hierarchy_context:
            formatted_result += f"\n🏗️ {hierarchy_context}"

        # Parent info (for hierarchy, not for attachment items themselves)
        if result.parent_title and not result.is_attachment:
            formatted_result += f"\n⬆️ Parent: {result.parent_title}"

        # Children count
        if result.has_children():
            formatted_result += f"\n⬇️ Children: {result.children_count}"

        return formatted_result

    @staticmethod
    def format_search_result(result: HybridSearchResult) -> str:
        """Format a search result for display.

        POC5: Now includes enrichment metadata when available.
        """
        return BasicResultFormatters._format_common_fields(
            result, is_attachment_view=False
        )

    @staticmethod
    def format_attachment_search_result(result: HybridSearchResult) -> str:
        """Format an attachment search result for display.

        POC5: Now includes enrichment metadata when available.
        """
        return BasicResultFormatters._format_common_fields(
            result, is_attachment_view=True
        )

    @staticmethod
    def format_hierarchical_results(
        organized_results: dict[str, list[HybridSearchResult]],
    ) -> str:
        """Format hierarchically organized results for display.

        POC5: Now includes enrichment summary per result.
        """
        formatted_sections = []

        for root_title, results in organized_results.items():
            section = f"📁 **{root_title}** ({len(results)} results)\n"

            for result in results:
                indent = "  " * (getattr(result, "depth", 0) or 0)
                section += f"{indent}📄 {result.source_title}"
                if hasattr(result, "hierarchy_context") and result.hierarchy_context:
                    section += f" | {result.hierarchy_context}"
                section += f" (Score: {result.score:.3f})\n"

                # Add a snippet of the content
                content_snippet = (
                    result.text[:150] + "..." if len(result.text) > 150 else result.text
                )
                section += f"{indent}   {content_snippet}\n"

                # POC5: Add enrichment summary if available
                enrichment_summary = getattr(result, "get_enrichment_summary", lambda: None)()
                if enrichment_summary:
                    section += f"{indent}   📊 {enrichment_summary}\n"

                if result.source_url:
                    section += f"{indent}   🔗 {result.source_url}\n"
                section += "\n"

            formatted_sections.append(section)

        return (
            f"Found {sum(len(results) for results in organized_results.values())} results organized by hierarchy:\n\n"
            + "\n".join(formatted_sections)
        )

    @staticmethod
    def format_enrichment_overview(
        results: list[HybridSearchResult],
    ) -> str:
        """POC5: Format an overview of enrichment data across all results.

        Provides a human-readable summary of keywords and entities found.

        Args:
            results: List of search results

        Returns:
            Formatted enrichment overview string
        """
        if not results:
            return "No results to analyze."

        all_keywords: dict[str, int] = {}
        people_set: set[str] = set()
        orgs_set: set[str] = set()
        locs_set: set[str] = set()
        docs_with_enrichment = 0

        for result in results:
            keyword_list = getattr(result, "keyword_list", None)
            entity_types = getattr(result, "entity_types", None)

            # Validate types
            if not isinstance(keyword_list, list):
                keyword_list = []
            if not isinstance(entity_types, dict):
                entity_types = {}

            if keyword_list or entity_types:
                docs_with_enrichment += 1

            for kw in keyword_list:
                all_keywords[kw] = all_keywords.get(kw, 0) + 1

            people_set.update(entity_types.get("PERSON", []))
            orgs_set.update(entity_types.get("ORG", []))
            locs_set.update(
                entity_types.get("GPE", []) +
                entity_types.get("LOC", []) +
                entity_types.get("FAC", [])
            )

        output = "📊 **Enrichment Overview**\n\n"
        output += f"Documents analyzed: {len(results)}\n"
        output += f"Documents with enrichment: {docs_with_enrichment}\n\n"

        if all_keywords:
            top_keywords = sorted(all_keywords.items(), key=lambda x: -x[1])[:10]
            output += "🏷️ **Top Keywords:**\n"
            for kw, count in top_keywords:
                output += f"  - {kw} ({count} docs)\n"
            output += "\n"

        if people_set:
            output += f"👤 **People ({len(people_set)}):** {', '.join(sorted(people_set)[:10])}\n"
            if len(people_set) > 10:
                output += f"    (+{len(people_set) - 10} more)\n"

        if orgs_set:
            output += f"🏢 **Organizations ({len(orgs_set)}):** {', '.join(sorted(orgs_set)[:10])}\n"
            if len(orgs_set) > 10:
                output += f"    (+{len(orgs_set) - 10} more)\n"

        if locs_set:
            output += f"📍 **Locations ({len(locs_set)}):** {', '.join(sorted(locs_set)[:10])}\n"
            if len(locs_set) > 10:
                output += f"    (+{len(locs_set) - 10} more)\n"

        return output
