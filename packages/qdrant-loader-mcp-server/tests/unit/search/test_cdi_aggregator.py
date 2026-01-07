"""
Tests for CDI Evidence Aggregator module.

Tests cover:
- EvidenceSource and ConflictSeverity enums
- EvidenceItem and AggregatedResult dataclasses
- EvidenceWeights configuration
- EvidenceAggregator main functionality
- ConflictDetectionPipeline high-level API
- Singleton pattern
- Integration with topic filter, NLI, and atomic facts

Workflow ID: W4-cdi-conflict-cff57ee0
"""

import pytest
from unittest.mock import patch, MagicMock

from qdrant_loader_mcp_server.search.enhanced.cdi.aggregator import (
    EvidenceSource,
    ConflictSeverity,
    EvidenceItem,
    AggregatedResult,
    EvidenceWeights,
    EvidenceAggregator,
    ConflictDetectionPipeline,
    get_evidence_aggregator,
    reset_evidence_aggregator,
    detect_conflict,
)
from qdrant_loader_mcp_server.search.enhanced.cdi.config import (
    get_config,
    reset_config,
)
from qdrant_loader_mcp_server.search.enhanced.cdi.topic_filter import (
    reset_topic_filter,
)
from qdrant_loader_mcp_server.search.enhanced.cdi.nli_detector import (
    reset_nli_detector,
    MockNLIModel,
    NLIDetector,
)
from qdrant_loader_mcp_server.search.enhanced.cdi.atomic_facts import (
    reset_fact_extractor,
)


class TestEvidenceSource:
    """Tests for EvidenceSource enum."""

    def test_all_sources_exist(self):
        """Test that all expected sources exist."""
        assert EvidenceSource.TOPIC_FILTER.value == "topic_filter"
        assert EvidenceSource.NLI_DETECTION.value == "nli_detection"
        assert EvidenceSource.ATOMIC_FACTS.value == "atomic_facts"
        assert EvidenceSource.KEYWORD_MATCH.value == "keyword_match"
        assert EvidenceSource.NUMERIC_MISMATCH.value == "numeric_mismatch"
        assert EvidenceSource.NEGATION_MISMATCH.value == "negation_mismatch"
        assert EvidenceSource.LEGACY.value == "legacy"


class TestConflictSeverity:
    """Tests for ConflictSeverity enum."""

    def test_all_severities_exist(self):
        """Test that all severity levels exist."""
        assert ConflictSeverity.NONE.value == "none"
        assert ConflictSeverity.LOW.value == "low"
        assert ConflictSeverity.MEDIUM.value == "medium"
        assert ConflictSeverity.HIGH.value == "high"
        assert ConflictSeverity.CRITICAL.value == "critical"


class TestEvidenceItem:
    """Tests for EvidenceItem dataclass."""

    def test_basic_creation(self):
        """Test creating an evidence item."""
        item = EvidenceItem(
            source=EvidenceSource.NLI_DETECTION,
            score=0.85,
            description="NLI contradiction detected",
        )
        assert item.source == EvidenceSource.NLI_DETECTION
        assert item.score == 0.85
        assert item.description == "NLI contradiction detected"
        assert item.details == {}

    def test_with_details(self):
        """Test creating evidence item with details."""
        item = EvidenceItem(
            source=EvidenceSource.NUMERIC_MISMATCH,
            score=0.9,
            description="Numeric mismatch",
            details={"value1": 100, "value2": 200},
        )
        assert item.details["value1"] == 100
        assert item.details["value2"] == 200

    def test_to_dict(self):
        """Test converting to dictionary."""
        item = EvidenceItem(
            source=EvidenceSource.NLI_DETECTION,
            score=0.85,
            description="Test",
        )
        d = item.to_dict()
        assert d["source"] == "nli_detection"
        assert d["score"] == 0.85


class TestAggregatedResult:
    """Tests for AggregatedResult dataclass."""

    def test_basic_creation(self):
        """Test creating an aggregated result."""
        result = AggregatedResult(
            text1="Text A",
            text2="Text B",
            is_conflict=True,
            confidence=0.8,
            severity=ConflictSeverity.HIGH,
        )
        assert result.is_conflict is True
        assert result.confidence == 0.8
        assert result.severity == ConflictSeverity.HIGH

    def test_evidence_count(self):
        """Test evidence count property."""
        evidence = [
            EvidenceItem(EvidenceSource.NLI_DETECTION, 0.9, "Test 1"),
            EvidenceItem(EvidenceSource.KEYWORD_MATCH, 0.5, "Test 2"),
        ]
        result = AggregatedResult(
            text1="A",
            text2="B",
            is_conflict=True,
            confidence=0.7,
            severity=ConflictSeverity.MEDIUM,
            evidence=evidence,
        )
        assert result.evidence_count == 2

    def test_get_top_evidence(self):
        """Test getting top evidence items."""
        evidence = [
            EvidenceItem(EvidenceSource.KEYWORD_MATCH, 0.3, "Low"),
            EvidenceItem(EvidenceSource.NLI_DETECTION, 0.9, "High"),
            EvidenceItem(EvidenceSource.TOPIC_FILTER, 0.5, "Medium"),
        ]
        result = AggregatedResult(
            text1="A",
            text2="B",
            is_conflict=True,
            confidence=0.7,
            severity=ConflictSeverity.MEDIUM,
            evidence=evidence,
        )
        top = result.get_top_evidence(2)
        assert len(top) == 2
        assert top[0].score == 0.9
        assert top[1].score == 0.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = AggregatedResult(
            text1="Short text",
            text2="Another text",
            is_conflict=True,
            confidence=0.8,
            severity=ConflictSeverity.HIGH,
        )
        d = result.to_dict()
        assert d["is_conflict"] is True
        assert d["confidence"] == 0.8
        assert d["severity"] == "high"

    def test_to_dict_truncates_long_text(self):
        """Test that to_dict truncates long texts."""
        long_text = "A" * 200
        result = AggregatedResult(
            text1=long_text,
            text2="Short",
            is_conflict=False,
            confidence=0.0,
            severity=ConflictSeverity.NONE,
        )
        d = result.to_dict()
        assert len(d["text1_preview"]) == 103  # 100 + "..."


class TestEvidenceWeights:
    """Tests for EvidenceWeights configuration."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = EvidenceWeights()
        # Check weights are normalized
        total = (
            weights.nli_weight
            + weights.atomic_facts_weight
            + weights.topic_weight
            + weights.keyword_weight
            + weights.legacy_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_custom_weights(self):
        """Test custom weight values."""
        weights = EvidenceWeights(
            nli_weight=0.6,
            atomic_facts_weight=0.4,
            topic_weight=0.0,
            keyword_weight=0.0,
            legacy_weight=0.0,
        )
        # Should be normalized to sum to 1.0
        assert abs(weights.nli_weight + weights.atomic_facts_weight - 1.0) < 0.001

    def test_to_dict(self):
        """Test converting weights to dictionary."""
        weights = EvidenceWeights()
        d = weights.to_dict()
        assert "nli_weight" in d
        assert "atomic_facts_weight" in d


class TestEvidenceAggregator:
    """Tests for EvidenceAggregator class."""

    def setup_method(self):
        """Reset all singletons before each test."""
        reset_config()
        reset_topic_filter()
        reset_nli_detector()
        reset_fact_extractor()
        reset_evidence_aggregator()

    def test_detect_no_conflict_v2_disabled(self):
        """Test detection returns no conflict when v2 is disabled."""
        config = get_config()
        config.enable_v2_detection = False
        config.use_topic_filter = False
        config.use_nli_model = False
        config.use_atomic_facts = False

        aggregator = EvidenceAggregator()
        result = aggregator.detect_conflict("Text A", "Text B")

        # With all features disabled, no evidence should be collected
        assert result.evidence_count == 0

    def test_detect_with_topic_filter_incompatible(self):
        """Test that incompatible topics filter out comparison."""
        config = get_config()
        config.use_topic_filter = True
        config.enable_v2_detection = True

        aggregator = EvidenceAggregator()

        # Tech vs Food - should be filtered
        result = aggregator.detect_conflict(
            "OAuth API authentication with JWT tokens and SSL certificates",
            "Coffee brewing temperature and extraction time for espresso recipes"
        )

        assert result.filtered_by_topic is True
        assert result.is_conflict is False

    def test_detect_with_topic_filter_compatible(self):
        """Test that compatible topics allow comparison."""
        config = get_config()
        config.use_topic_filter = True
        config.enable_v2_detection = True

        aggregator = EvidenceAggregator()

        # Both tech - should compare (add more tech keywords)
        result = aggregator.detect_conflict(
            "Python API authentication uses OAuth2 with JWT tokens for server security",
            "JavaScript API authentication uses OAuth2 with session tokens for server security"
        )

        assert result.filtered_by_topic is False

    def test_detect_skip_topic_filter(self):
        """Test skipping topic filter."""
        config = get_config()
        config.use_topic_filter = True
        config.enable_v2_detection = True

        aggregator = EvidenceAggregator()

        # Even incompatible topics should be compared when skipping filter
        result = aggregator.detect_conflict(
            "OAuth API authentication",
            "Coffee brewing temperature",
            skip_topic_filter=True,
        )

        assert result.filtered_by_topic is False

    def test_custom_weights(self):
        """Test aggregator with custom weights."""
        # All zeros except nli to get nli_weight = 1.0 after normalization
        weights = EvidenceWeights(
            nli_weight=1.0,
            atomic_facts_weight=0.0,
            topic_weight=0.0,
            keyword_weight=0.0,
            legacy_weight=0.0,
        )
        aggregator = EvidenceAggregator(weights=weights)

        # After normalization, nli_weight should be 1.0
        assert aggregator.weights.nli_weight == 1.0

    def test_detect_conflicts_batch(self):
        """Test batch conflict detection."""
        config = get_config()
        config.enable_v2_detection = False

        aggregator = EvidenceAggregator()

        pairs = [
            ("Text A", "Text B"),
            ("Text C", "Text D"),
        ]
        results = aggregator.detect_conflicts_batch(pairs)

        assert len(results) == 2
        assert all(isinstance(r, AggregatedResult) for r in results)

    def test_detect_conflicts_in_documents(self):
        """Test conflict detection in document list."""
        config = get_config()
        config.enable_v2_detection = False

        aggregator = EvidenceAggregator()

        docs = ["Doc 1", "Doc 2", "Doc 3"]
        results = aggregator.detect_conflicts_in_documents(docs)

        # Should analyze all pairs (3 pairs for 3 docs)
        assert isinstance(results, list)


class TestEvidenceAggregatorWithNLI:
    """Tests for aggregator with NLI integration."""

    def setup_method(self):
        """Reset all singletons."""
        reset_config()
        reset_topic_filter()
        reset_nli_detector()
        reset_fact_extractor()
        reset_evidence_aggregator()

    def test_nli_evidence_collected(self):
        """Test that NLI evidence is collected when enabled."""
        config = get_config()
        config.use_nli_model = True
        config.use_topic_filter = False
        config.use_atomic_facts = False
        config.enable_v2_detection = True

        # Create mock NLI detector that returns contradiction
        mock_nli = MockNLIModel(
            default_scores={"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}
        )
        nli_detector = NLIDetector(model=mock_nli)

        aggregator = EvidenceAggregator(nli_detector=nli_detector)
        result = aggregator.detect_conflict(
            "The sky is blue.",
            "The sky is red."
        )

        # Should have NLI evidence
        nli_evidence = [
            e for e in result.evidence
            if e.source == EvidenceSource.NLI_DETECTION
        ]
        assert len(nli_evidence) >= 1
        assert nli_evidence[0].score == 0.9


class TestEvidenceAggregatorSeverity:
    """Tests for severity determination."""

    def setup_method(self):
        """Reset singletons."""
        reset_config()
        reset_evidence_aggregator()

    def test_severity_none(self):
        """Test NONE severity for low scores."""
        aggregator = EvidenceAggregator()
        severity = aggregator._determine_severity(0.2, 1)
        assert severity == ConflictSeverity.NONE

    def test_severity_low(self):
        """Test LOW severity."""
        aggregator = EvidenceAggregator()
        severity = aggregator._determine_severity(0.4, 1)
        assert severity == ConflictSeverity.LOW

    def test_severity_medium(self):
        """Test MEDIUM severity."""
        aggregator = EvidenceAggregator()
        severity = aggregator._determine_severity(0.6, 2)
        assert severity == ConflictSeverity.MEDIUM

    def test_severity_high(self):
        """Test HIGH severity."""
        aggregator = EvidenceAggregator()
        severity = aggregator._determine_severity(0.8, 3)
        assert severity == ConflictSeverity.HIGH

    def test_severity_critical(self):
        """Test CRITICAL severity."""
        aggregator = EvidenceAggregator()
        severity = aggregator._determine_severity(0.95, 5)
        assert severity == ConflictSeverity.CRITICAL


class TestConflictDetectionPipeline:
    """Tests for high-level pipeline API."""

    def setup_method(self):
        """Reset singletons."""
        reset_config()
        reset_topic_filter()
        reset_nli_detector()
        reset_fact_extractor()
        reset_evidence_aggregator()

    def test_check_for_conflict_v2_disabled(self):
        """Test quick check returns False when v2 disabled."""
        config = get_config()
        config.enable_v2_detection = False

        pipeline = ConflictDetectionPipeline()
        result = pipeline.check_for_conflict("A", "B")

        assert result is False

    def test_get_conflict_details(self):
        """Test getting detailed conflict analysis."""
        config = get_config()
        config.enable_v2_detection = True
        config.use_topic_filter = False
        config.use_nli_model = False
        config.use_atomic_facts = False

        pipeline = ConflictDetectionPipeline()
        result = pipeline.get_conflict_details("Text A", "Text B")

        assert isinstance(result, AggregatedResult)

    def test_is_v2_enabled(self):
        """Test checking v2 status."""
        config = get_config()
        config.enable_v2_detection = True

        pipeline = ConflictDetectionPipeline()
        assert pipeline.is_v2_enabled() is True

        config.enable_v2_detection = False
        assert pipeline.is_v2_enabled() is False

    def test_get_enabled_features(self):
        """Test getting feature status."""
        config = get_config()
        config.enable_v2_detection = True
        config.use_topic_filter = True
        config.use_nli_model = False
        config.use_atomic_facts = False

        pipeline = ConflictDetectionPipeline()
        features = pipeline.get_enabled_features()

        assert features["v2_detection"] is True
        assert features["topic_filter"] is True
        assert features["nli_model"] is False
        assert features["atomic_facts"] is False

    def test_analyze_documents(self):
        """Test document analysis."""
        config = get_config()
        config.enable_v2_detection = True
        config.use_topic_filter = False
        config.use_nli_model = False
        config.use_atomic_facts = False

        pipeline = ConflictDetectionPipeline()
        docs = ["Doc 1", "Doc 2", "Doc 3"]

        analysis = pipeline.analyze_documents(docs)

        assert analysis["total_documents"] == 3
        assert analysis["total_pairs_analyzed"] == 3  # 3 choose 2
        assert "conflicts_found" in analysis
        assert "conflict_rate" in analysis


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singletons."""
        reset_evidence_aggregator()

    def test_get_evidence_aggregator_singleton(self):
        """Test that get returns same instance."""
        agg1 = get_evidence_aggregator()
        agg2 = get_evidence_aggregator()
        assert agg1 is agg2

    def test_reset_evidence_aggregator(self):
        """Test that reset creates new instance."""
        agg1 = get_evidence_aggregator()
        reset_evidence_aggregator()
        agg2 = get_evidence_aggregator()
        assert agg1 is not agg2


class TestConvenienceFunction:
    """Tests for detect_conflict convenience function."""

    def setup_method(self):
        """Reset singletons."""
        reset_config()
        reset_evidence_aggregator()

    def test_detect_conflict_function(self):
        """Test the detect_conflict convenience function."""
        config = get_config()
        config.enable_v2_detection = False

        result = detect_conflict("Text A", "Text B")

        assert isinstance(result, AggregatedResult)


class TestAggregatorScoreCalculation:
    """Tests for score calculation logic."""

    def setup_method(self):
        """Reset singletons."""
        reset_config()
        reset_evidence_aggregator()

    def test_calculate_empty_evidence(self):
        """Test score calculation with no evidence."""
        aggregator = EvidenceAggregator()
        score = aggregator._calculate_aggregated_score([])
        assert score == 0.0

    def test_calculate_single_evidence(self):
        """Test score calculation with single evidence."""
        aggregator = EvidenceAggregator()
        evidence = [
            EvidenceItem(EvidenceSource.NLI_DETECTION, 0.8, "Test")
        ]
        score = aggregator._calculate_aggregated_score(evidence)
        # Use approximate comparison for floating point
        assert abs(score - 0.8) < 0.001

    def test_calculate_multiple_evidence(self):
        """Test score calculation with multiple evidence."""
        aggregator = EvidenceAggregator()
        evidence = [
            EvidenceItem(EvidenceSource.NLI_DETECTION, 0.8, "NLI"),
            EvidenceItem(EvidenceSource.KEYWORD_MATCH, 0.6, "Keyword"),
        ]
        score = aggregator._calculate_aggregated_score(evidence)
        # Should be weighted average
        assert 0.0 < score < 1.0

    def test_weight_for_sources(self):
        """Test getting weights for different sources."""
        aggregator = EvidenceAggregator()

        nli_weight = aggregator._get_weight_for_source(EvidenceSource.NLI_DETECTION)
        facts_weight = aggregator._get_weight_for_source(EvidenceSource.ATOMIC_FACTS)
        topic_weight = aggregator._get_weight_for_source(EvidenceSource.TOPIC_FILTER)

        # NLI should have highest weight by default
        assert nli_weight > topic_weight


class TestKeywordConflictDetection:
    """Tests for legacy keyword conflict detection."""

    def setup_method(self):
        """Reset singletons."""
        reset_config()
        reset_evidence_aggregator()

    def test_detect_conflict_indicator(self):
        """Test detecting conflict indicator words."""
        config = get_config()
        config.enable_v2_detection = True
        config.use_topic_filter = False
        config.use_nli_model = False
        config.use_atomic_facts = False

        aggregator = EvidenceAggregator()
        evidence = aggregator._detect_keyword_conflicts(
            "You should not use this method.",
            "This method is recommended."
        )

        # Should find "should not" and "recommended"
        assert len(evidence) >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Reset singletons."""
        reset_config()
        reset_evidence_aggregator()

    def test_empty_texts(self):
        """Test handling empty texts."""
        config = get_config()
        config.enable_v2_detection = False

        aggregator = EvidenceAggregator()
        result = aggregator.detect_conflict("", "")

        assert isinstance(result, AggregatedResult)
        assert result.is_conflict is False

    def test_same_text(self):
        """Test comparing identical texts."""
        config = get_config()
        config.enable_v2_detection = False

        aggregator = EvidenceAggregator()
        result = aggregator.detect_conflict("Same text", "Same text")

        # Identical texts shouldn't conflict
        assert result.is_conflict is False

    def test_very_long_texts(self):
        """Test handling very long texts."""
        config = get_config()
        config.enable_v2_detection = False

        aggregator = EvidenceAggregator()
        long_text = "A" * 10000

        result = aggregator.detect_conflict(long_text, long_text)
        assert isinstance(result, AggregatedResult)

    def test_unicode_texts(self):
        """Test handling unicode texts."""
        config = get_config()
        config.enable_v2_detection = False

        aggregator = EvidenceAggregator()
        result = aggregator.detect_conflict(
            "日本語テキスト",
            "中文文本"
        )

        assert isinstance(result, AggregatedResult)
