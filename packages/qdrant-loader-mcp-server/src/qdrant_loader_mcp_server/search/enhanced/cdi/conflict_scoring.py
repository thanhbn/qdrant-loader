# ============================================================
# LEARNING: conflict_scoring.py
# This file has been annotated with TODO markers for learning.
# To restore: git checkout -- packages/qdrant-loader-mcp-server/src/qdrant_loader_mcp_server/search/enhanced/cdi/conflict_scoring.py
# Learning Objectives:
# - [ ] Understand text conflict detection using spaCy NLP
# - [ ] Understand metadata conflict detection rules
# - [ ] Understand conflict categorization algorithm
# - [ ] Understand confidence scoring calculation
# ============================================================

from __future__ import annotations

from datetime import datetime
from typing import Any


def _safe_to_str(item: Any) -> str:
    """Safely convert item to string for set operations."""
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return str(item.text)
    return str(item)


def analyze_text_conflicts(
    detector: Any, doc1: Any, doc2: Any
) -> tuple[bool, str, float]:
    """spaCy-driven textual conflict heuristics (extracted)."""
    try:
        # TODO [L1]: Extract text from documents
        # Use Case: Get text content for NLP analysis
        # Data Flow: SearchResult -> .text attribute -> raw text string
        # Git: "Enhanced conflict detection" (commit 2a3cd6b)
        # -----------------------------------------------------------
        # Use .text attribute for SearchResult, fallback to .content if available
        doc1_text = getattr(doc1, "text", None) or getattr(doc1, "content", "")
        doc2_text = getattr(doc2, "text", None) or getattr(doc2, "content", "")

        # TODO [L1]: Analyze documents with spaCy semantic analysis
        # Use Case: Extract entities and keywords from documents
        # Data Flow: text -> spacy_analyzer.analyze_query_semantic -> analysis result
        # Git: "Enhanced conflict detection" (commit 2a3cd6b)
        # -----------------------------------------------------------
        doc1_analysis = detector.spacy_analyzer.analyze_query_semantic(doc1_text)
        doc2_analysis = detector.spacy_analyzer.analyze_query_semantic(doc2_text)

        # Safely extract entities and keywords as strings
        doc1_entities = {_safe_to_str(ent[0]).lower() for ent in doc1_analysis.entities}
        doc2_entities = {_safe_to_str(ent[0]).lower() for ent in doc2_analysis.entities}
        doc1_keywords = {_safe_to_str(kw).lower() for kw in doc1_analysis.semantic_keywords}
        doc2_keywords = {_safe_to_str(kw).lower() for kw in doc2_analysis.semantic_keywords}
        # -----------------------------------------------------------

        # TODO [L1]: Calculate entity and keyword overlap ratios
        # Use Case: Measure how much two documents share same topics
        # Business Rule: Jaccard similarity = intersection / union
        # Test: test_find_contradiction_patterns_version_conflicts
        # -----------------------------------------------------------
        entity_overlap = len(doc1_entities & doc2_entities) / max(
            len(doc1_entities | doc2_entities), 1
        )
        keyword_overlap = len(doc1_keywords & doc2_keywords) / max(
            len(doc1_keywords | doc2_keywords), 1
        )
        # -----------------------------------------------------------

        # TODO [L1]: Define conflict indicator keywords
        # Use Case: Detect conflicting language patterns in documents
        # Business Rule: Words like "never", "always", "must", "deprecated" suggest conflicts
        # Git: "Initial CDI implementation" (commit 0c9eac3)
        # Test: test_find_contradiction_patterns_procedural_conflicts
        # -----------------------------------------------------------
        conflict_indicators = [
            "should not",
            "avoid",
            "deprecated",
            "recommended",
            "best practice",
            "anti-pattern",
            "wrong",
            "correct",
            "instead",
            "better",
            "worse",
            "must",
            "never",
            "always",
            "required",
            "mandatory",
            "optional",
            "different",
            "conflicting",
            "inconsistent",
        ]

        doc1_indicators = sum(
            1 for indicator in conflict_indicators if indicator in doc1_text.lower()
        )
        doc2_indicators = sum(
            1 for indicator in conflict_indicators if indicator in doc2_text.lower()
        )
        # -----------------------------------------------------------

        # TODO [L1]: Apply conflict detection rules
        # Use Case: Determine if documents have conflicting information
        # Business Rule: Three conditions for conflict detection:
        #   1. High entity overlap (>0.2) + any indicator words
        #   2. Moderate entity overlap (>0.15) + high keyword overlap (>0.3)
        #   3. Both docs have multiple indicators + some overlap
        # Git: "Enhanced conflict detection" (commit 2a3cd6b)
        # Test: test_conflict_confidence_scoring
        # -----------------------------------------------------------
        # Condition 1: High entity overlap with indicator words
        if entity_overlap > 0.2 and (doc1_indicators > 0 or doc2_indicators > 0):
            confidence = min(
                entity_overlap * (doc1_indicators + doc2_indicators) / 8, 1.0
            )
            explanation = f"Similar topics with conflicting recommendations (entity overlap: {entity_overlap:.2f})"
            return True, explanation, max(0.3, confidence)

        # Condition 2: Moderate entity overlap + high keyword overlap
        if entity_overlap > 0.15 and keyword_overlap > 0.3:
            confidence = min((entity_overlap + keyword_overlap) / 2, 0.7)
            explanation = f"Overlapping keywords suggest potential conflicts (keyword overlap: {keyword_overlap:.2f})"
            return True, explanation, confidence

        # Condition 3: Multiple indicators in both docs
        if doc1_indicators > 1 and doc2_indicators > 1 and (entity_overlap > 0.1 or keyword_overlap > 0.1):
            confidence = min((doc1_indicators + doc2_indicators) / 10, 0.6)
            explanation = "Multiple conflicting statements detected in both documents"
            return True, explanation, confidence
        # -----------------------------------------------------------

        return False, "No textual conflicts detected", 0.0
    except Exception as e:  # pragma: no cover
        detector.logger.error(f"Error in text conflict analysis: {e}")
        return False, f"Text analysis error: {str(e)}", 0.0


def _parse_date(date_value: Any) -> datetime | None:
    """Parse date string to datetime object."""
    if date_value is None:
        return None
    if isinstance(date_value, datetime):
        return date_value
    if isinstance(date_value, str):
        # Try common ISO formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(date_value, fmt)
            except ValueError:
                continue
    return None


def analyze_metadata_conflicts(
    detector: Any, doc1: Any, doc2: Any
) -> tuple[bool, str, float]:
    """Metadata-driven conflict heuristics (extracted)."""
    try:
        conflicts: list[tuple[str, float, str]] = []
        total_weight = 0.0

        # TODO [L1]: Detect date-based conflicts
        # Use Case: Documents created far apart may have outdated information
        # Business Rule: Documents >180 days apart get weight 0.15-0.2
        # Git: "Streamlined conflict detection settings" (commit a7bd979)
        # -----------------------------------------------------------
        # Parse date strings to datetime objects
        doc1_date = _parse_date(getattr(doc1, "created_at", None))
        doc2_date = _parse_date(getattr(doc2, "created_at", None))
        if doc1_date and doc2_date:
            date_diff = abs((doc1_date - doc2_date).days)
            if date_diff > 180:  # Relaxed from 365 to 180 days
                weight = 0.2 if date_diff > 365 else 0.15
                conflicts.append(
                    ("date_conflict", weight, f"Documents created {date_diff} days apart")
                )
                total_weight += weight
        # -----------------------------------------------------------

        # TODO [L1]: Detect source type conflicts
        # Use Case: Different sources (confluence vs git) may have conflicting policies
        # Business Rule: Weight varies by source combination (e.g., confluence+git = 0.2)
        # Git: "Metadata conflict detection" (commit 58a9666)
        # -----------------------------------------------------------
        if doc1.source_type != doc2.source_type:
            source_conflicts = {
                ("confluence", "git"): 0.2,
                ("jira", "confluence"): 0.15,
                ("jira", "git"): 0.15,
                ("localfile", "confluence"): 0.1,
                ("localfile", "git"): 0.1,
                ("publicdocs", "confluence"): 0.1,
                ("publicdocs", "git"): 0.1,
            }
            conflict_key = tuple(sorted([doc1.source_type, doc2.source_type]))
            w = source_conflicts.get(conflict_key, 0.1)  # Default weight for any mismatch
            conflicts.append(
                (
                    "source_type_conflict",
                    w,
                    f"Different source types: {doc1.source_type} vs {doc2.source_type}",
                )
            )
            total_weight += w
        # -----------------------------------------------------------

        # TODO [L1]: Detect project conflicts
        # Use Case: Documents from different projects may have conflicting configurations
        # Business Rule: Different project_id = weight 0.15
        # -----------------------------------------------------------
        doc1_project = getattr(doc1, "project_id", None)
        doc2_project = getattr(doc2, "project_id", None)
        if doc1_project and doc2_project and doc1_project != doc2_project:
            conflicts.append(
                (
                    "project_conflict",
                    0.15,
                    f"Different projects: {doc1_project} vs {doc2_project}",
                )
            )
            total_weight += 0.15
        # -----------------------------------------------------------

        # TODO [L1]: Detect title similarity conflicts
        # Use Case: Similar titles from different sources = potential duplicate/conflict
        # Business Rule: Word overlap >50% in titles = weight 0.2
        # -----------------------------------------------------------
        doc1_title = getattr(doc1, "source_title", "") or ""
        doc2_title = getattr(doc2, "source_title", "") or ""
        if doc1_title and doc2_title:
            # Simple word overlap check for titles
            words1 = set(doc1_title.lower().split())
            words2 = set(doc2_title.lower().split())
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > 0.5:
                    conflicts.append(
                        (
                            "title_similarity",
                            0.2,
                            f"Similar titles: '{doc1_title[:50]}...' vs '{doc2_title[:50]}...'",
                        )
                    )
                    total_weight += 0.2
        # -----------------------------------------------------------

        # TODO [L1]: Apply minimum threshold for metadata conflicts
        # Use Case: Only report conflicts above significance threshold
        # Business Rule: total_weight > 0.1 to report (relaxed from 0.2)
        # Git: "Streamlined conflict detection settings" (commit a7bd979)
        # -----------------------------------------------------------
        if conflicts and total_weight > 0.1:
            explanation = "; ".join([c[2] for c in conflicts])
            return True, explanation, min(total_weight, 1.0)
        # -----------------------------------------------------------

        return False, "No metadata conflicts detected", 0.0
    except Exception as e:  # pragma: no cover
        detector.logger.error(f"Error in metadata conflict analysis: {e}")
        return False, f"Metadata analysis error: {str(e)}", 0.0


def categorize_conflict(_detector: Any, patterns) -> str:
    """Categorize conflict type based on detected patterns.

    Use Case: Group conflicts by type for reporting
    Business Rule: Categories are: version, procedural, data, general
    Git: "CDI Module Consolidation" (commit 9bc4aa9)
    Test: test_categorize_conflict_specific_patterns
    """
    # TODO [L1]: Implement conflict categorization
    # Business Rule: Check pattern keywords to determine category:
    #   - "version", "deprecated" -> version conflict
    #   - "procedure", "process", "should", "must", "never" -> procedural conflict
    #   - "data", "value", "number" -> data conflict
    #   - default -> general conflict
    # -----------------------------------------------------------
    if not patterns:
        return "unknown"
    for item in patterns:
        if isinstance(item, dict):
            pattern_text = item.get("type", "").lower()
        elif isinstance(item, tuple) and len(item) > 0:
            pattern_text = str(item[0]).lower()
        elif isinstance(item, str):
            pattern_text = item.lower()
        else:
            pattern_text = str(item).lower()

        if any(keyword in pattern_text for keyword in ["version", "deprecated"]):
            return "version"
        elif any(
            keyword in pattern_text
            for keyword in [
                "procedure",
                "process",
                "steps",
                "should",
                "must",
                "never",
                "always",
            ]
        ):
            return "procedural"
        elif any(
            keyword in pattern_text
            for keyword in [
                "data",
                "value",
                "number",
                "different values",
                "conflicting data",
            ]
        ):
            return "data"

    return "general"
    # -----------------------------------------------------------


def calculate_conflict_confidence(
    _detector: Any, patterns, doc1_score: float = 1.0, doc2_score: float = 1.0
) -> float:
    """Calculate confidence score for detected conflict.

    Use Case: Provide numeric confidence for conflict severity
    Business Rule: Confidence = pattern_strength * avg_document_score
    Git: "Confidence Scoring Fixes" - Return 0.0 when no indicators
    Test: test_conflict_confidence_scoring
    """
    # TODO [L1]: Implement confidence calculation
    # Business Rule:
    #   1. Extract confidence from each pattern (default 0.5)
    #   2. Average all pattern confidences
    #   3. Multiply by average document relevance score
    #   4. Clamp to [0.0, 1.0]
    # -----------------------------------------------------------
    if not patterns:
        return 0.0
    confidences: list[float] = []
    for pattern in patterns:
        if isinstance(pattern, dict):
            confidences.append(pattern.get("confidence", 0.5))
        elif isinstance(pattern, tuple) and len(pattern) >= 2:
            try:
                confidences.append(float(pattern[1]))
            except (ValueError, IndexError):
                confidences.append(0.5)
        else:
            pattern_text = str(pattern).lower()
            if any(
                ind in pattern_text
                for ind in [
                    "conflict",
                    "incompatible",
                    "contradicts",
                    "different values",
                ]
            ):
                confidences.append(0.8)
            elif any(
                ind in pattern_text
                for ind in ["different approach", "alternative method"]
            ):
                confidences.append(0.6)
            elif any(ind in pattern_text for ind in ["unclear", "possibly different"]):
                confidences.append(0.3)
            else:
                confidences.append(0.5)
    pattern_strength = sum(confidences) / len(confidences) if confidences else 0.5
    doc_score_avg = (doc1_score + doc2_score) / 2
    return min(1.0, pattern_strength * doc_score_avg)
    # -----------------------------------------------------------
