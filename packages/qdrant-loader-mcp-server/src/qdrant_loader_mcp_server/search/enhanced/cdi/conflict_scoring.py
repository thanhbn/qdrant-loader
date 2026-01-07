from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import get_config


@dataclass
class ConflictIndicator:
    """Represents a specific conflicting statement pair."""

    doc1_snippet: str
    doc2_snippet: str
    context: str
    conflict_type: str  # "value_mismatch", "contradictory_guidance", "version_conflict"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "doc1_snippet": self.doc1_snippet,
            "doc2_snippet": self.doc2_snippet,
            "context": self.context,
            "conflict_type": self.conflict_type,
            "confidence": self.confidence,
        }


@dataclass
class TextConflictResult:
    """Result of text conflict analysis with extracted snippets."""

    has_conflict: bool
    description: str
    confidence: float
    indicators: list[ConflictIndicator] = field(default_factory=list)

    def to_legacy_tuple(self) -> tuple[bool, str, float]:
        """Convert to legacy tuple format for backwards compatibility."""
        return (self.has_conflict, self.description, self.confidence)

    def get_structured_indicators(self) -> list[dict[str, Any]]:
        """Get indicators as list of dicts for serialization."""
        return [ind.to_dict() for ind in self.indicators]


def extract_conflict_snippets(
    doc1_content: str,
    doc2_content: str,
    conflict_keywords: list[str],
    context_window: int = 150,
) -> list[ConflictIndicator]:
    """Extract actual conflicting text snippets around conflict keywords."""
    indicators = []

    for keyword in conflict_keywords:
        # Find keyword in both docs with surrounding context
        pattern = rf"(.{{0,{context_window}}}{re.escape(keyword)}.{{0,{context_window}}})"

        doc1_match = re.search(pattern, doc1_content, re.IGNORECASE | re.DOTALL)
        doc2_match = re.search(pattern, doc2_content, re.IGNORECASE | re.DOTALL)

        if doc1_match and doc2_match:
            doc1_snippet = doc1_match.group(1).strip()
            doc2_snippet = doc2_match.group(1).strip()

            # Clean up snippets - remove excessive whitespace
            doc1_snippet = " ".join(doc1_snippet.split())
            doc2_snippet = " ".join(doc2_snippet.split())

            # Only add if snippets are actually different
            if doc1_snippet.lower() != doc2_snippet.lower():
                config = get_config()
                indicators.append(
                    ConflictIndicator(
                        doc1_snippet=doc1_snippet[:300],  # Limit length
                        doc2_snippet=doc2_snippet[:300],
                        context=f"Different content around '{keyword}'",
                        conflict_type="content_difference",
                        confidence=config.content_difference_confidence,
                    )
                )

    return indicators


def extract_value_conflicts(
    doc1_content: str, doc2_content: str
) -> list[ConflictIndicator]:
    """Extract conflicts where same config key has different values."""
    indicators = []

    # Pattern for key-value pairs (e.g., "ssl: true", "pool_size = 10", "SSL=false")
    kv_patterns = [
        r"(\w+)\s*[:=]\s*([^\n,;}\]]+)",  # key: value or key = value
        r'"(\w+)"\s*:\s*"?([^",\n}]+)"?',  # "key": "value" or "key": value
    ]

    doc1_kvs: dict[str, str] = {}
    doc2_kvs: dict[str, str] = {}

    for pattern in kv_patterns:
        for key, val in re.findall(pattern, doc1_content, re.IGNORECASE):
            doc1_kvs[key.lower().strip()] = val.strip()
        for key, val in re.findall(pattern, doc2_content, re.IGNORECASE):
            doc2_kvs[key.lower().strip()] = val.strip()

    # Find keys present in both with different values
    common_keys = set(doc1_kvs.keys()) & set(doc2_kvs.keys())

    # Filter out common non-meaningful keys
    skip_keys = {"type", "name", "id", "version", "description", "title", "content"}

    for key in common_keys:
        if key in skip_keys:
            continue

        val1 = doc1_kvs[key].strip().lower()
        val2 = doc2_kvs[key].strip().lower()

        # Skip if values are the same
        if val1 == val2:
            continue

        # Skip if values are too long (likely content, not config)
        if len(val1) > 100 or len(val2) > 100:
            continue

        config = get_config()
        indicators.append(
            ConflictIndicator(
                doc1_snippet=f"{key}: {doc1_kvs[key]}",
                doc2_snippet=f"{key}: {doc2_kvs[key]}",
                context=f"Configuration value mismatch for '{key}'",
                conflict_type="value_mismatch",
                confidence=config.value_conflict_confidence,
            )
        )

    return indicators


def extract_contradictory_statements(
    doc1_content: str, doc2_content: str
) -> list[ConflictIndicator]:
    """Extract statements that contradict each other."""
    indicators = []

    # Patterns that indicate recommendations/requirements
    recommendation_patterns = [
        (r"(should\s+(?:not\s+)?[^.]+\.)", "recommendation"),
        (r"(must\s+(?:not\s+)?[^.]+\.)", "requirement"),
        (r"(always\s+[^.]+\.)", "guideline"),
        (r"(never\s+[^.]+\.)", "prohibition"),
        (r"(recommended\s+(?:to\s+)?[^.]+\.)", "recommendation"),
        (r"(avoid\s+[^.]+\.)", "prohibition"),
        (r"(use\s+[^.]+\s+instead\s+of\s+[^.]+\.)", "preference"),
    ]

    doc1_statements: list[tuple[str, str]] = []
    doc2_statements: list[tuple[str, str]] = []

    for pattern, stmt_type in recommendation_patterns:
        for match in re.finditer(pattern, doc1_content, re.IGNORECASE):
            doc1_statements.append((match.group(1).strip(), stmt_type))
        for match in re.finditer(pattern, doc2_content, re.IGNORECASE):
            doc2_statements.append((match.group(1).strip(), stmt_type))

    # Look for contradictions between statements
    config = get_config()
    for stmt1, type1 in doc1_statements:
        for stmt2, type2 in doc2_statements:
            # Check for opposite recommendations on same topic
            if _statements_contradict(stmt1, stmt2):
                indicators.append(
                    ConflictIndicator(
                        doc1_snippet=stmt1[:300],
                        doc2_snippet=stmt2[:300],
                        context=f"Contradictory {type1} vs {type2}",
                        conflict_type="contradictory_guidance",
                        confidence=config.contradiction_confidence,
                    )
                )

    return indicators[:5]  # Limit to top 5 contradictions


def _statements_contradict(stmt1: str, stmt2: str) -> bool:
    """Check if two statements contradict each other."""
    stmt1_lower = stmt1.lower()
    stmt2_lower = stmt2.lower()

    # Extract key terms
    stmt1_words = set(re.findall(r"\b\w+\b", stmt1_lower))
    stmt2_words = set(re.findall(r"\b\w+\b", stmt2_lower))

    # Need some overlap to be about the same topic
    common_words = stmt1_words & stmt2_words
    # Filter out stop words
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "be",
        "to",
        "of",
        "and",
        "or",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "should",
        "must",
        "not",
        "always",
        "never",
    }
    meaningful_common = common_words - stop_words

    if len(meaningful_common) < 2:
        return False

    # Check for negation differences
    negation_words = {"not", "never", "avoid", "don't", "shouldn't", "mustn't"}

    stmt1_has_negation = bool(stmt1_words & negation_words)
    stmt2_has_negation = bool(stmt2_words & negation_words)

    # If one has negation and other doesn't, likely contradiction
    if stmt1_has_negation != stmt2_has_negation:
        return True

    return False


def _safe_extract_entity_text(entity) -> str | None:
    """Safely extract text from an entity regardless of its type."""
    if isinstance(entity, tuple) and len(entity) > 0:
        return str(entity[0]).lower()
    elif isinstance(entity, str):
        return entity.lower()
    elif hasattr(entity, "text"):
        return str(entity.text).lower()
    return None


def _safe_extract_keyword(keyword) -> str | None:
    """Safely extract keyword text regardless of its type."""
    if isinstance(keyword, str):
        return keyword.lower()
    elif hasattr(keyword, "text"):
        return str(keyword.text).lower()
    return None


def _get_text_content(doc: Any) -> str:
    """Safely extract text content from a document, handling various formats."""
    # Try text first (most common for SearchResult)
    text = getattr(doc, "text", None)
    if isinstance(text, str) and text:
        return text

    # Try content field
    content = getattr(doc, "content", None)
    if isinstance(content, str) and content:
        return content

    # If content is an object with text attribute
    if content is not None and hasattr(content, "text"):
        text_val = getattr(content, "text", None)
        if isinstance(text_val, str):
            return text_val

    return ""


def analyze_text_conflicts(
    detector: Any, doc1: Any, doc2: Any
) -> TextConflictResult:
    """Enhanced spaCy-driven textual conflict analysis with snippet extraction."""
    try:
        doc1_content = _get_text_content(doc1)
        doc2_content = _get_text_content(doc2)

        if not doc1_content or not doc2_content:
            return TextConflictResult(
                has_conflict=False,
                description="No content to analyze",
                confidence=0.0,
                indicators=[],
            )

        doc1_analysis = detector.spacy_analyzer.analyze_query_semantic(doc1_content)
        doc2_analysis = detector.spacy_analyzer.analyze_query_semantic(doc2_content)

        # Safely extract entities - handle various formats
        doc1_entities: set[str] = set()
        doc2_entities: set[str] = set()
        for ent in getattr(doc1_analysis, "entities", []):
            text = _safe_extract_entity_text(ent)
            if text:
                doc1_entities.add(text)
        for ent in getattr(doc2_analysis, "entities", []):
            text = _safe_extract_entity_text(ent)
            if text:
                doc2_entities.add(text)

        # Safely extract keywords
        doc1_keywords: set[str] = set()
        doc2_keywords: set[str] = set()
        for kw in getattr(doc1_analysis, "semantic_keywords", []):
            text = _safe_extract_keyword(kw)
            if text:
                doc1_keywords.add(text)
        for kw in getattr(doc2_analysis, "semantic_keywords", []):
            text = _safe_extract_keyword(kw)
            if text:
                doc2_keywords.add(text)

        entity_overlap = len(doc1_entities & doc2_entities) / max(
            len(doc1_entities | doc2_entities), 1
        )
        keyword_overlap = len(doc1_keywords & doc2_keywords) / max(
            len(doc1_keywords | doc2_keywords), 1
        )

        # Use config for conflict indicator words
        config = get_config()
        conflict_indicator_words = config.conflict_indicator_words

        doc1_indicator_count = sum(
            1 for ind in conflict_indicator_words if ind in doc1_content.lower()
        )
        doc2_indicator_count = sum(
            1 for ind in conflict_indicator_words if ind in doc2_content.lower()
        )

        # Collect all conflict indicators
        all_indicators: list[ConflictIndicator] = []

        # 1. Extract value conflicts (highest priority - most actionable)
        value_conflicts = extract_value_conflicts(doc1_content, doc2_content)
        all_indicators.extend(value_conflicts)

        # 2. Extract contradictory statements
        contradictions = extract_contradictory_statements(doc1_content, doc2_content)
        all_indicators.extend(contradictions)

        # 3. Extract snippet conflicts around shared keywords (if overlap is high)
        keyword_conflicts: list[ConflictIndicator] = []
        if keyword_overlap > config.keyword_overlap_threshold:
            shared_keywords = list(doc1_keywords & doc2_keywords)[:5]
            if shared_keywords:
                keyword_conflicts = extract_conflict_snippets(
                    doc1_content, doc2_content, shared_keywords
                )
                all_indicators.extend(keyword_conflicts)

        # Determine if there's a conflict
        has_conflict = False
        confidence = 0.0
        description_parts = []

        # Value conflicts are strong evidence
        if value_conflicts:
            has_conflict = True
            confidence = max(confidence, config.value_conflict_confidence)
            description_parts.append(
                f"Found {len(value_conflicts)} value mismatches"
            )

        # Contradictory statements are strong evidence
        if contradictions:
            has_conflict = True
            confidence = max(confidence, config.contradiction_confidence)
            description_parts.append(
                f"Found {len(contradictions)} contradictory statements"
            )

        # Keyword conflicts (different content around same keywords) are evidence
        if keyword_conflicts:
            has_conflict = True
            confidence = max(confidence, config.content_difference_confidence)
            description_parts.append(
                f"Found {len(keyword_conflicts)} content differences around shared keywords"
            )

        # High keyword overlap with conflict indicator words
        if keyword_overlap > config.keyword_overlap_threshold and (
            doc1_indicator_count > 0 or doc2_indicator_count > 0
        ):
            has_conflict = True
            base_confidence = min(
                keyword_overlap * (doc1_indicator_count + doc2_indicator_count) / 10, 1.0
            )
            confidence = max(confidence, base_confidence)
            description_parts.append(
                f"Overlapping keywords with conflict indicators (overlap: {keyword_overlap:.2f})"
            )

        if has_conflict:
            description = "; ".join(description_parts) if description_parts else "Conflict detected"
            return TextConflictResult(
                has_conflict=True,
                description=description,
                confidence=confidence,
                indicators=all_indicators,
            )

        return TextConflictResult(
            has_conflict=False,
            description="No textual conflicts detected",
            confidence=0.0,
            indicators=[],
        )

    except Exception as e:
        detector.logger.error(f"Error in text conflict analysis: {e}")
        return TextConflictResult(
            has_conflict=False,
            description=f"Text analysis error: {str(e)}",
            confidence=0.0,
            indicators=[],
        )


def _parse_date(date_value) -> "datetime | None":
    """Safely parse a date value that may be string or datetime."""
    from datetime import datetime

    if date_value is None:
        return None
    if isinstance(date_value, datetime):
        return date_value
    if isinstance(date_value, str):
        # Try common date formats
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
) -> tuple[bool, str, float, list[ConflictIndicator]]:
    """Metadata-driven conflict heuristics with indicator extraction."""
    try:
        config = get_config()
        conflicts: list[tuple[str, float, str]] = []
        indicators: list[ConflictIndicator] = []
        total_weight = 0.0

        # Safely parse dates
        doc1_date = _parse_date(getattr(doc1, "created_at", None))
        doc2_date = _parse_date(getattr(doc2, "created_at", None))
        if doc1_date and doc2_date:
            date_diff = abs((doc1_date - doc2_date).days)
            if date_diff > config.temporal_conflict_days:
                conflicts.append(
                    ("date_conflict", 0.3, f"Documents created {date_diff} days apart")
                )
                total_weight += 0.3
                indicators.append(
                    ConflictIndicator(
                        doc1_snippet=f"Created: {doc1_date}",
                        doc2_snippet=f"Created: {doc2_date}",
                        context=f"Documents created {date_diff} days apart - may have outdated information",
                        conflict_type="temporal_conflict",
                        confidence=config.temporal_conflict_confidence,
                    )
                )

        # NOTE: Removed metadata-based "conflicts" that aren't real conflicts:
        # - Different source types (confluence vs git) - expected behavior
        # - Similar titles - indicates related content, not conflict
        # - Different projects - expected behavior
        #
        # Real metadata conflicts are now only date-based (documents > 1 year apart
        # may have outdated information)

        if conflicts and total_weight >= config.metadata_conflict_min_weight:
            explanation = "; ".join([c[2] for c in conflicts])
            return True, explanation, min(total_weight, 1.0), indicators

        return False, "No metadata conflicts detected", 0.0, []

    except Exception as e:
        detector.logger.error(f"Error in metadata conflict analysis: {e}")
        return False, f"Metadata analysis error: {str(e)}", 0.0, []


def categorize_conflict(_detector: Any, patterns) -> str:
    """Categorize conflict based on patterns."""
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
                "mismatch",
            ]
        ):
            return "data"

    return "general"


def calculate_conflict_confidence(
    _detector: Any, patterns, doc1_score: float = 1.0, doc2_score: float = 1.0
) -> float:
    """Calculate overall conflict confidence from patterns."""
    if not patterns:
        return 0.0
    confidences: list[float] = []
    for pattern in patterns:
        if isinstance(pattern, dict):
            confidences.append(pattern.get("confidence", 0.5))
        elif isinstance(pattern, ConflictIndicator):
            confidences.append(pattern.confidence)
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
                    "mismatch",
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
