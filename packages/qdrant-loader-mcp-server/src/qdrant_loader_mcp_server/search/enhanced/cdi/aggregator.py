"""
Evidence Aggregator for CDI Conflict Detection.

This module aggregates evidence from multiple detection sources:
- Topic filtering (domain relevance)
- NLI-based contradiction detection
- Atomic fact comparison
- Legacy keyword-based detection

The aggregator provides a unified pipeline interface with
configurable weights and feature flags for gradual v2.0 rollout.

Workflow ID: W4-cdi-conflict-cff57ee0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .config import get_config
from .topic_filter import TopicFilter, TopicClassification, get_topic_filter
from .nli_detector import NLIDetector, NLIResult, get_nli_detector
from .atomic_facts import (
    AtomicFactExtractor,
    AtomicFact,
    ExtractionResult,
    FactComparator,
    get_fact_extractor,
)

logger = logging.getLogger(__name__)


class EvidenceSource(Enum):
    """Sources of conflict evidence."""

    TOPIC_FILTER = "topic_filter"
    NLI_DETECTION = "nli_detection"
    NLI = "nli"  # Alias for NLI-based detection
    ATOMIC_FACTS = "atomic_facts"
    KEYWORD_MATCH = "keyword_match"
    KEYWORD = "keyword"  # Alias for keyword-based detection
    NUMERIC_MISMATCH = "numeric_mismatch"
    NEGATION_MISMATCH = "negation_mismatch"
    METADATA = "metadata"  # Metadata-based conflicts
    LEGACY = "legacy"


class ConflictSeverity(Enum):
    """Severity levels for detected conflicts."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvidenceItem:
    """A single piece of evidence for a conflict.

    Attributes:
        source: The detection source
        score: Confidence score (0-1)
        description: Human-readable description
        details: Additional details specific to the source
    """

    source: EvidenceSource
    score: float
    description: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "score": self.score,
            "description": self.description,
            "details": self.details,
        }


@dataclass
class AggregatedResult:
    """Result of aggregated conflict detection.

    Attributes:
        text1: First document/text
        text2: Second document/text
        is_conflict: Whether a conflict was detected
        confidence: Overall confidence score
        severity: Conflict severity level
        evidence: List of evidence items
        filtered_by_topic: Whether comparison was skipped due to topic filter
        topic_info: Topic classification information
    """

    text1: str
    text2: str
    is_conflict: bool
    confidence: float
    severity: ConflictSeverity
    evidence: list[EvidenceItem] = field(default_factory=list)
    filtered_by_topic: bool = False
    topic_info: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_count(self) -> int:
        """Get number of evidence items."""
        return len(self.evidence)

    def get_top_evidence(self, n: int = 3) -> list[EvidenceItem]:
        """Get top N evidence items by score."""
        sorted_evidence = sorted(self.evidence, key=lambda e: e.score, reverse=True)
        return sorted_evidence[:n]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text1_preview": self.text1[:100] + "..." if len(self.text1) > 100 else self.text1,
            "text2_preview": self.text2[:100] + "..." if len(self.text2) > 100 else self.text2,
            "is_conflict": self.is_conflict,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "evidence_count": self.evidence_count,
            "evidence": [e.to_dict() for e in self.evidence],
            "filtered_by_topic": self.filtered_by_topic,
            "topic_info": self.topic_info,
        }


class EvidenceWeights:
    """Configurable weights for evidence sources.

    These weights determine how much each evidence source
    contributes to the final conflict score.
    """

    def __init__(
        self,
        nli_weight: float = 0.4,
        atomic_facts_weight: float = 0.3,
        topic_weight: float = 0.1,
        keyword_weight: float = 0.15,
        legacy_weight: float = 0.05,
    ):
        """Initialize weights.

        Args:
            nli_weight: Weight for NLI-based detection
            atomic_facts_weight: Weight for atomic fact comparison
            topic_weight: Weight for topic relevance
            keyword_weight: Weight for keyword matching
            legacy_weight: Weight for legacy detection
        """
        self.nli_weight = nli_weight
        self.atomic_facts_weight = atomic_facts_weight
        self.topic_weight = topic_weight
        self.keyword_weight = keyword_weight
        self.legacy_weight = legacy_weight

        # Normalize weights
        self._normalize()

    def _normalize(self):
        """Normalize weights to sum to 1.0."""
        total = (
            self.nli_weight
            + self.atomic_facts_weight
            + self.topic_weight
            + self.keyword_weight
            + self.legacy_weight
        )
        if total > 0:
            self.nli_weight /= total
            self.atomic_facts_weight /= total
            self.topic_weight /= total
            self.keyword_weight /= total
            self.legacy_weight /= total

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "nli_weight": self.nli_weight,
            "atomic_facts_weight": self.atomic_facts_weight,
            "topic_weight": self.topic_weight,
            "keyword_weight": self.keyword_weight,
            "legacy_weight": self.legacy_weight,
        }


class EvidenceAggregator:
    """Aggregates evidence from multiple conflict detection sources.

    This is the main entry point for the v2.0 conflict detection pipeline.
    It coordinates between topic filtering, NLI detection, and atomic
    fact comparison to produce aggregated conflict results.

    Example:
        aggregator = EvidenceAggregator()
        result = aggregator.detect_conflict(
            "Python is compiled to bytecode.",
            "Python is interpreted directly."
        )
        if result.is_conflict:
            print(f"Conflict detected with {result.confidence:.2%} confidence")
            for evidence in result.get_top_evidence():
                print(f"  - {evidence.description}")
    """

    def __init__(
        self,
        weights: EvidenceWeights | None = None,
        topic_filter: TopicFilter | None = None,
        nli_detector: NLIDetector | None = None,
        fact_extractor: AtomicFactExtractor | None = None,
    ):
        """Initialize the aggregator.

        Args:
            weights: Custom evidence weights
            topic_filter: Custom topic filter instance
            nli_detector: Custom NLI detector instance
            fact_extractor: Custom fact extractor instance
        """
        self.weights = weights or EvidenceWeights()
        self._topic_filter = topic_filter
        self._nli_detector = nli_detector
        self._fact_extractor = fact_extractor
        self._fact_comparator = FactComparator()

    def _get_topic_filter(self) -> TopicFilter:
        """Get or create topic filter."""
        if self._topic_filter is None:
            self._topic_filter = get_topic_filter()
        return self._topic_filter

    def _get_nli_detector(self) -> NLIDetector:
        """Get or create NLI detector."""
        if self._nli_detector is None:
            self._nli_detector = get_nli_detector()
        return self._nli_detector

    def _get_fact_extractor(self) -> AtomicFactExtractor:
        """Get or create fact extractor."""
        if self._fact_extractor is None:
            self._fact_extractor = get_fact_extractor()
        return self._fact_extractor

    def detect_conflict(
        self,
        text1: str,
        text2: str,
        skip_topic_filter: bool = False,
        include_legacy: bool = True,
    ) -> AggregatedResult:
        """Detect conflicts between two texts.

        Args:
            text1: First text
            text2: Second text
            skip_topic_filter: Whether to skip topic pre-filtering
            include_legacy: Whether to include legacy detection

        Returns:
            AggregatedResult with all evidence
        """
        config = get_config()
        evidence: list[EvidenceItem] = []
        topic_info: dict[str, Any] = {}

        # Step 1: Topic filtering (if enabled)
        if config.use_topic_filter and not skip_topic_filter:
            topic_filter = self._get_topic_filter()
            should_compare = topic_filter.should_compare(text1, text2)

            # Get topic classifications
            class1 = topic_filter.classify_topic(text1)
            class2 = topic_filter.classify_topic(text2)

            topic_info = {
                "text1_domain": class1.domain,
                "text1_topics": list(class1.topics),
                "text2_domain": class2.domain,
                "text2_topics": list(class2.topics),
                "should_compare": should_compare,
            }

            if not should_compare:
                # Return early - topics are incompatible
                return AggregatedResult(
                    text1=text1,
                    text2=text2,
                    is_conflict=False,
                    confidence=0.0,
                    severity=ConflictSeverity.NONE,
                    filtered_by_topic=True,
                    topic_info=topic_info,
                )

            # Add topic relevance as evidence
            similarity, explanation = topic_filter.calculate_topic_similarity(text1, text2)
            if similarity > 0:
                evidence.append(
                    EvidenceItem(
                        source=EvidenceSource.TOPIC_FILTER,
                        score=similarity,
                        description=f"Topic similarity: {similarity:.2%}",
                        details={"similarity": similarity, "explanation": explanation, **topic_info},
                    )
                )

        # Step 2: NLI detection (if enabled)
        if config.use_nli_model:
            nli_detector = self._get_nli_detector()
            nli_result = nli_detector.detect_contradiction(text1, text2)

            if nli_result.is_contradiction:
                evidence.append(
                    EvidenceItem(
                        source=EvidenceSource.NLI_DETECTION,
                        score=nli_result.contradiction_score,
                        description=f"NLI contradiction: {nli_result.contradiction_score:.2%}",
                        details={
                            "contradiction": nli_result.contradiction_score,
                            "entailment": nli_result.entailment_score,
                            "neutral": nli_result.neutral_score,
                            "label": nli_result.label.value,
                        },
                    )
                )

        # Step 3: Atomic fact comparison (if enabled)
        if config.use_atomic_facts:
            fact_extractor = self._get_fact_extractor()
            facts1 = fact_extractor.extract(text1)
            facts2 = fact_extractor.extract(text2)

            conflicts = self._fact_comparator.find_conflicting_facts(
                facts1.facts, facts2.facts
            )

            for conflict in conflicts:
                source = self._get_fact_conflict_source(conflict)
                evidence.append(
                    EvidenceItem(
                        source=source,
                        score=conflict["confidence"],
                        description=f"{conflict['conflict_type']}: {conflict['confidence']:.2%}",
                        details={
                            "fact1": conflict["fact1"],
                            "fact2": conflict["fact2"],
                            "conflict_type": conflict["conflict_type"],
                        },
                    )
                )

        # Step 4: Legacy keyword detection (if enabled)
        if include_legacy and config.enable_v2_detection:
            legacy_evidence = self._detect_keyword_conflicts(text1, text2)
            evidence.extend(legacy_evidence)

        # Calculate aggregated score
        aggregated_score = self._calculate_aggregated_score(evidence)

        # Determine severity
        severity = self._determine_severity(aggregated_score, len(evidence))

        # Determine if conflict
        is_conflict = aggregated_score >= 0.5 and len(evidence) > 0

        return AggregatedResult(
            text1=text1,
            text2=text2,
            is_conflict=is_conflict,
            confidence=aggregated_score,
            severity=severity,
            evidence=evidence,
            filtered_by_topic=False,
            topic_info=topic_info,
        )

    def _get_fact_conflict_source(self, conflict: dict[str, Any]) -> EvidenceSource:
        """Get the evidence source for a fact conflict."""
        conflict_type = conflict.get("conflict_type", "")
        if conflict_type == "numeric_mismatch":
            return EvidenceSource.NUMERIC_MISMATCH
        elif conflict_type == "negation_mismatch":
            return EvidenceSource.NEGATION_MISMATCH
        elif conflict_type == "semantic_contradiction":
            return EvidenceSource.NLI_DETECTION
        return EvidenceSource.ATOMIC_FACTS

    def _detect_keyword_conflicts(
        self, text1: str, text2: str
    ) -> list[EvidenceItem]:
        """Detect conflicts using keyword matching.

        This is a simplified version of the legacy detection logic.
        """
        evidence = []
        config = get_config()

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check for conflict indicator words
        for indicator in config.conflict_indicator_words:
            if indicator in text1_lower or indicator in text2_lower:
                # Check if the other text contradicts
                if indicator in text1_lower and indicator not in text2_lower:
                    evidence.append(
                        EvidenceItem(
                            source=EvidenceSource.KEYWORD_MATCH,
                            score=0.5,
                            description=f"Conflict indicator: '{indicator}' found in text1",
                            details={"indicator": indicator, "in_text1": True},
                        )
                    )
                elif indicator in text2_lower and indicator not in text1_lower:
                    evidence.append(
                        EvidenceItem(
                            source=EvidenceSource.KEYWORD_MATCH,
                            score=0.5,
                            description=f"Conflict indicator: '{indicator}' found in text2",
                            details={"indicator": indicator, "in_text2": True},
                        )
                    )

        return evidence

    def _calculate_aggregated_score(
        self, evidence: list[EvidenceItem]
    ) -> float:
        """Calculate weighted aggregated score from evidence."""
        if not evidence:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for item in evidence:
            weight = self._get_weight_for_source(item.source)
            weighted_sum += item.score * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return weighted_sum / weight_sum

    def _get_weight_for_source(self, source: EvidenceSource) -> float:
        """Get the weight for an evidence source."""
        if source == EvidenceSource.NLI_DETECTION:
            return self.weights.nli_weight
        elif source in (
            EvidenceSource.ATOMIC_FACTS,
            EvidenceSource.NUMERIC_MISMATCH,
            EvidenceSource.NEGATION_MISMATCH,
        ):
            return self.weights.atomic_facts_weight
        elif source == EvidenceSource.TOPIC_FILTER:
            return self.weights.topic_weight
        elif source == EvidenceSource.KEYWORD_MATCH:
            return self.weights.keyword_weight
        elif source == EvidenceSource.LEGACY:
            return self.weights.legacy_weight
        return 0.1  # Default weight

    def _determine_severity(
        self, score: float, evidence_count: int
    ) -> ConflictSeverity:
        """Determine conflict severity based on score and evidence."""
        if score < 0.3 or evidence_count == 0:
            return ConflictSeverity.NONE
        elif score < 0.5:
            return ConflictSeverity.LOW
        elif score < 0.7:
            return ConflictSeverity.MEDIUM
        elif score < 0.9:
            return ConflictSeverity.HIGH
        return ConflictSeverity.CRITICAL

    def detect_conflicts_batch(
        self,
        text_pairs: list[tuple[str, str]],
        skip_topic_filter: bool = False,
    ) -> list[AggregatedResult]:
        """Detect conflicts for multiple text pairs.

        Args:
            text_pairs: List of (text1, text2) tuples
            skip_topic_filter: Whether to skip topic pre-filtering

        Returns:
            List of AggregatedResults
        """
        return [
            self.detect_conflict(t1, t2, skip_topic_filter=skip_topic_filter)
            for t1, t2 in text_pairs
        ]

    def detect_conflicts_in_documents(
        self,
        documents: list[Any],
        text_extractor: Callable[[Any], str] | None = None,
    ) -> list[AggregatedResult]:
        """Detect conflicts between all pairs of documents.

        Args:
            documents: List of documents
            text_extractor: Function to extract text from document

        Returns:
            List of AggregatedResults for conflicting pairs
        """
        if text_extractor is None:
            text_extractor = lambda d: str(d) if not isinstance(d, str) else d

        results = []
        n = len(documents)

        for i in range(n):
            for j in range(i + 1, n):
                text1 = text_extractor(documents[i])
                text2 = text_extractor(documents[j])

                result = self.detect_conflict(text1, text2)
                if result.is_conflict:
                    results.append(result)

        return results


# Global singleton instance
_aggregator_instance: EvidenceAggregator | None = None


def get_evidence_aggregator() -> EvidenceAggregator:
    """Get the global evidence aggregator instance."""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = EvidenceAggregator()
    return _aggregator_instance


def reset_evidence_aggregator():
    """Reset the global aggregator instance (for testing)."""
    global _aggregator_instance
    _aggregator_instance = None


# Pipeline API for production use
class ConflictDetectionPipeline:
    """High-level pipeline for conflict detection.

    This class provides a simple interface for production use,
    handling feature flag checks and graceful degradation.

    Example:
        pipeline = ConflictDetectionPipeline()
        if pipeline.check_for_conflict("Text A", "Text B"):
            print("Potential conflict detected!")
    """

    def __init__(self):
        """Initialize the pipeline."""
        self._aggregator = None

    def _get_aggregator(self) -> EvidenceAggregator:
        """Get the aggregator instance."""
        if self._aggregator is None:
            self._aggregator = get_evidence_aggregator()
        return self._aggregator

    def check_for_conflict(
        self,
        text1: str,
        text2: str,
        min_confidence: float = 0.5,
    ) -> bool:
        """Quick check if two texts conflict.

        Args:
            text1: First text
            text2: Second text
            min_confidence: Minimum confidence threshold

        Returns:
            True if conflict detected above threshold
        """
        config = get_config()

        # If v2 features are disabled, fall back to simple check
        if not config.enable_v2_detection:
            return False

        aggregator = self._get_aggregator()
        result = aggregator.detect_conflict(text1, text2)

        return result.is_conflict and result.confidence >= min_confidence

    def get_conflict_details(
        self,
        text1: str,
        text2: str,
    ) -> AggregatedResult:
        """Get detailed conflict analysis.

        Args:
            text1: First text
            text2: Second text

        Returns:
            AggregatedResult with full details
        """
        aggregator = self._get_aggregator()
        return aggregator.detect_conflict(text1, text2)

    def analyze_documents(
        self,
        documents: list[str],
        include_non_conflicts: bool = False,
    ) -> dict[str, Any]:
        """Analyze a set of documents for conflicts.

        Args:
            documents: List of document texts
            include_non_conflicts: Whether to include non-conflicting pairs

        Returns:
            Analysis summary with conflicts
        """
        aggregator = self._get_aggregator()

        all_results = []
        conflicts = []
        n = len(documents)
        total_pairs = n * (n - 1) // 2

        for i in range(n):
            for j in range(i + 1, n):
                result = aggregator.detect_conflict(documents[i], documents[j])
                if result.is_conflict:
                    conflicts.append(
                        {
                            "doc1_index": i,
                            "doc2_index": j,
                            "confidence": result.confidence,
                            "severity": result.severity.value,
                            "evidence_count": result.evidence_count,
                        }
                    )
                if include_non_conflicts:
                    all_results.append(result)

        return {
            "total_documents": n,
            "total_pairs_analyzed": total_pairs,
            "conflicts_found": len(conflicts),
            "conflict_rate": len(conflicts) / total_pairs if total_pairs > 0 else 0,
            "conflicts": conflicts,
        }

    def is_v2_enabled(self) -> bool:
        """Check if v2 detection features are enabled."""
        config = get_config()
        return config.enable_v2_detection

    def get_enabled_features(self) -> dict[str, bool]:
        """Get status of all v2 features."""
        config = get_config()
        return {
            "v2_detection": config.enable_v2_detection,
            "topic_filter": config.use_topic_filter,
            "nli_model": config.use_nli_model,
            "atomic_facts": config.use_atomic_facts,
        }


# Convenience function for quick conflict check
def detect_conflict(text1: str, text2: str) -> AggregatedResult:
    """Detect conflicts between two texts.

    This is the main entry point for conflict detection.

    Args:
        text1: First text
        text2: Second text

    Returns:
        AggregatedResult with conflict details
    """
    aggregator = get_evidence_aggregator()
    return aggregator.detect_conflict(text1, text2)
