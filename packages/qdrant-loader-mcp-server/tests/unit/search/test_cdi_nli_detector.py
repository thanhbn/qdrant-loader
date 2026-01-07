"""
Tests for CDI NLI Detector module.

Tests cover:
- NLIResult dataclass and properties
- NLILabel enum
- MockNLIModel for testing
- NLIDetector with mock model
- Batch processing
- Caching behavior
- Singleton pattern
- Convenience functions

Workflow ID: W4-cdi-conflict-cff57ee0
"""

import pytest
from unittest.mock import patch, MagicMock

from qdrant_loader_mcp_server.search.enhanced.cdi.nli_detector import (
    NLILabel,
    NLIResult,
    MockNLIModel,
    NLIDetector,
    get_nli_detector,
    reset_nli_detector,
    cached_contradiction_check,
    detect_pairwise_contradictions,
)
from qdrant_loader_mcp_server.search.enhanced.cdi.config import (
    get_config,
    reset_config,
)


class TestNLILabel:
    """Tests for NLILabel enum."""

    def test_label_values(self):
        """Test that all expected labels exist."""
        assert NLILabel.ENTAILMENT.value == "entailment"
        assert NLILabel.CONTRADICTION.value == "contradiction"
        assert NLILabel.NEUTRAL.value == "neutral"

    def test_label_comparison(self):
        """Test label comparison."""
        assert NLILabel.CONTRADICTION == NLILabel.CONTRADICTION
        assert NLILabel.ENTAILMENT != NLILabel.CONTRADICTION


class TestNLIResult:
    """Tests for NLIResult dataclass."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_result_creation(self):
        """Test creating an NLIResult."""
        result = NLIResult(
            premise="The sky is blue.",
            hypothesis="The sky is red.",
            label=NLILabel.CONTRADICTION,
            contradiction_score=0.9,
            entailment_score=0.05,
            neutral_score=0.05,
        )
        assert result.premise == "The sky is blue."
        assert result.hypothesis == "The sky is red."
        assert result.label == NLILabel.CONTRADICTION
        assert result.contradiction_score == 0.9

    def test_is_contradiction_above_threshold(self):
        """Test is_contradiction property above threshold."""
        # Config default is 0.8
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.CONTRADICTION,
            contradiction_score=0.85,
            entailment_score=0.1,
            neutral_score=0.05,
        )
        assert result.is_contradiction is True

    def test_is_contradiction_below_threshold(self):
        """Test is_contradiction property below threshold."""
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.CONTRADICTION,
            contradiction_score=0.7,
            entailment_score=0.1,
            neutral_score=0.2,
        )
        assert result.is_contradiction is False

    def test_is_entailment_above_threshold(self):
        """Test is_entailment property above threshold."""
        # Config default is 0.7
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.ENTAILMENT,
            contradiction_score=0.1,
            entailment_score=0.8,
            neutral_score=0.1,
        )
        assert result.is_entailment is True

    def test_is_entailment_below_threshold(self):
        """Test is_entailment property below threshold."""
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.ENTAILMENT,
            contradiction_score=0.1,
            entailment_score=0.6,
            neutral_score=0.3,
        )
        assert result.is_entailment is False

    def test_confidence_for_contradiction(self):
        """Test confidence property for contradiction."""
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.CONTRADICTION,
            contradiction_score=0.9,
            entailment_score=0.05,
            neutral_score=0.05,
        )
        assert result.confidence == 0.9

    def test_confidence_for_entailment(self):
        """Test confidence property for entailment."""
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.ENTAILMENT,
            contradiction_score=0.05,
            entailment_score=0.9,
            neutral_score=0.05,
        )
        assert result.confidence == 0.9

    def test_confidence_for_neutral(self):
        """Test confidence property for neutral."""
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.NEUTRAL,
            contradiction_score=0.1,
            entailment_score=0.1,
            neutral_score=0.8,
        )
        assert result.confidence == 0.8

    def test_result_is_frozen(self):
        """Test that NLIResult is immutable."""
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.NEUTRAL,
            contradiction_score=0.1,
            entailment_score=0.1,
            neutral_score=0.8,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.premise = "C"


class TestMockNLIModel:
    """Tests for MockNLIModel."""

    def test_default_scores(self):
        """Test mock model returns default neutral scores."""
        model = MockNLIModel()
        results = model.predict([("A", "B")])
        assert len(results) == 1
        assert results[0]["contradiction"] == 0.1
        assert results[0]["entailment"] == 0.1
        assert results[0]["neutral"] == 0.8

    def test_custom_scores(self):
        """Test mock model with custom scores."""
        custom_scores = {
            "contradiction": 0.9,
            "entailment": 0.05,
            "neutral": 0.05,
        }
        model = MockNLIModel(default_scores=custom_scores)
        results = model.predict([("A", "B")])
        assert results[0]["contradiction"] == 0.9

    def test_batch_prediction(self):
        """Test mock model handles batches."""
        model = MockNLIModel()
        pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        results = model.predict(pairs)
        assert len(results) == 3

    def test_empty_input(self):
        """Test mock model handles empty input."""
        model = MockNLIModel()
        results = model.predict([])
        assert results == []

    def test_call_tracking(self):
        """Test mock model tracks calls."""
        model = MockNLIModel()
        model.predict([("A", "B")])
        model.predict([("C", "D")])
        assert model.call_count == 2
        assert model.last_inputs == [("C", "D")]


class TestNLIDetector:
    """Tests for NLIDetector class."""

    def setup_method(self):
        """Reset config and detector before each test."""
        reset_config()
        reset_nli_detector()

    def test_detector_with_mock_model(self):
        """Test detector with explicit mock model."""
        mock = MockNLIModel(
            default_scores={"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}
        )
        detector = NLIDetector(model=mock)

        result = detector.detect_contradiction(
            "The sky is blue.",
            "The sky is red."
        )
        assert result.label == NLILabel.CONTRADICTION
        assert result.contradiction_score == 0.9

    def test_detector_neutral_result(self):
        """Test detector returns neutral when appropriate."""
        mock = MockNLIModel()  # Default is neutral
        detector = NLIDetector(model=mock)

        result = detector.detect_contradiction("A", "B")
        assert result.label == NLILabel.NEUTRAL

    def test_is_contradictory_convenience(self):
        """Test is_contradictory convenience method."""
        mock = MockNLIModel(
            default_scores={"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}
        )
        detector = NLIDetector(model=mock)

        assert detector.is_contradictory("A", "B") is True

    def test_get_contradiction_score(self):
        """Test get_contradiction_score method."""
        mock = MockNLIModel(
            default_scores={"contradiction": 0.75, "entailment": 0.15, "neutral": 0.1}
        )
        detector = NLIDetector(model=mock)

        score = detector.get_contradiction_score("A", "B")
        assert score == 0.75


class TestNLIDetectorBatching:
    """Tests for batch processing."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()

    def test_batch_detection(self):
        """Test batch contradiction detection."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock)

        pairs = [
            ("Text 1", "Text 2"),
            ("Text 3", "Text 4"),
            ("Text 5", "Text 6"),
        ]
        results = detector.detect_contradictions_batch(pairs)

        assert len(results) == 3
        assert all(isinstance(r, NLIResult) for r in results)

    def test_batch_empty_input(self):
        """Test batch with empty input."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock)

        results = detector.detect_contradictions_batch([])
        assert results == []

    def test_batch_single_item(self):
        """Test batch with single item."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock)

        results = detector.detect_contradictions_batch([("A", "B")])
        assert len(results) == 1


class TestNLIDetectorCaching:
    """Tests for caching behavior."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()

    def test_cache_hit(self):
        """Test that repeated queries use cache."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock, use_cache=True)

        # First call
        result1 = detector.detect_contradiction("A", "B")
        # Second call (should use cache)
        result2 = detector.detect_contradiction("A", "B")

        assert result1 == result2
        assert mock.call_count == 1  # Only called once

    def test_cache_normalization(self):
        """Test that cache normalizes whitespace."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock, use_cache=True)

        # These should be treated as the same
        result1 = detector.detect_contradiction("Hello World", "Test Text")
        result2 = detector.detect_contradiction("hello  world", "test  text")

        assert mock.call_count == 1

    def test_cache_disabled(self):
        """Test detector with caching disabled."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock, use_cache=False)

        detector.detect_contradiction("A", "B")
        detector.detect_contradiction("A", "B")

        assert mock.call_count == 2

    def test_clear_cache(self):
        """Test clearing the cache."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock, use_cache=True)

        detector.detect_contradiction("A", "B")
        assert mock.call_count == 1

        detector.clear_cache()
        detector.detect_contradiction("A", "B")
        assert mock.call_count == 2

    def test_cache_stats(self):
        """Test cache statistics."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock, use_cache=True)

        detector.detect_contradiction("A", "B")
        detector.detect_contradiction("C", "D")

        stats = detector.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["cache_enabled"] is True

    def test_batch_uses_cache(self):
        """Test that batch processing uses cache."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock, use_cache=True)

        # Pre-populate cache
        detector.detect_contradiction("A", "B")
        assert mock.call_count == 1

        # Batch with one cached, one new
        pairs = [("A", "B"), ("C", "D")]
        results = detector.detect_contradictions_batch(pairs)

        assert len(results) == 2
        # Model should only be called for new pair
        assert mock.call_count == 2


class TestNLIDetectorSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()

    def test_get_nli_detector_singleton(self):
        """Test that get_nli_detector returns same instance."""
        detector1 = get_nli_detector()
        detector2 = get_nli_detector()
        assert detector1 is detector2

    def test_reset_nli_detector(self):
        """Test that reset_nli_detector creates new instance."""
        detector1 = get_nli_detector()
        reset_nli_detector()
        detector2 = get_nli_detector()
        assert detector1 is not detector2


class TestNLIDetectorWithConfig:
    """Tests for NLI detector integration with config."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()

    def test_uses_mock_when_nli_disabled(self):
        """Test that mock is used when NLI is disabled in config."""
        config = get_config()
        config.use_nli_model = False

        detector = NLIDetector()
        result = detector.detect_contradiction("A", "B")

        # Should work without error (uses mock)
        assert result is not None
        assert isinstance(result, NLIResult)

    def test_model_available_property(self):
        """Test is_model_available property."""
        config = get_config()
        config.use_nli_model = False

        detector = NLIDetector()
        detector.detect_contradiction("A", "B")

        # When disabled, model is not available
        assert detector.is_model_available is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()
        # Clear the LRU cache for cached_contradiction_check
        cached_contradiction_check.cache_clear()

    def test_cached_contradiction_check_uses_cache(self):
        """Test that cached_contradiction_check uses LRU cache."""
        # Configure to use mock by disabling NLI
        config = get_config()
        config.use_nli_model = False

        result1 = cached_contradiction_check("A", "B")
        result2 = cached_contradiction_check("A", "B")

        assert result1 == result2

    def test_detect_pairwise_contradictions_empty(self):
        """Test pairwise detection with insufficient texts."""
        result = detect_pairwise_contradictions([])
        assert result == []

        result = detect_pairwise_contradictions(["Single text"])
        assert result == []

    def test_detect_pairwise_contradictions_basic(self):
        """Test pairwise detection with multiple texts."""
        config = get_config()
        config.use_nli_model = False
        reset_nli_detector()

        # With mock model (returns neutral), no contradictions expected
        texts = ["Text A", "Text B", "Text C"]
        result = detect_pairwise_contradictions(texts)

        # Mock returns neutral, so no contradictions
        assert result == []

    def test_detect_pairwise_contradictions_with_labels(self):
        """Test pairwise detection with custom labels."""
        config = get_config()
        config.use_nli_model = False
        reset_nli_detector()

        texts = ["Text A", "Text B"]
        labels = ["Doc 1", "Doc 2"]
        result = detect_pairwise_contradictions(texts, labels)

        # Result format check (even if empty due to mock)
        assert isinstance(result, list)

    def test_detect_pairwise_with_contradicting_mock(self):
        """Test pairwise detection when contradictions found."""
        # Need to set up a custom detector that returns contradictions
        mock = MockNLIModel(
            default_scores={"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}
        )

        with patch(
            "qdrant_loader_mcp_server.search.enhanced.cdi.nli_detector.get_nli_detector"
        ) as mock_get:
            mock_detector = NLIDetector(model=mock)
            mock_get.return_value = mock_detector

            texts = ["The sky is blue", "The sky is red"]
            result = detect_pairwise_contradictions(texts)

            assert len(result) == 1
            assert result[0]["text1_index"] == 0
            assert result[0]["text2_index"] == 1
            assert result[0]["contradiction_score"] == 0.9


class TestNLIDetectorLabelDetermination:
    """Tests for label determination logic."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()

    def test_label_from_highest_score(self):
        """Test that label is determined by highest score."""
        # Test contradiction wins
        mock = MockNLIModel(
            default_scores={"contradiction": 0.6, "entailment": 0.3, "neutral": 0.1}
        )
        detector = NLIDetector(model=mock)
        result = detector.detect_contradiction("A", "B")
        assert result.label == NLILabel.CONTRADICTION

        # Test entailment wins
        mock2 = MockNLIModel(
            default_scores={"contradiction": 0.1, "entailment": 0.7, "neutral": 0.2}
        )
        detector2 = NLIDetector(model=mock2)
        result2 = detector2.detect_contradiction("A", "B")
        assert result2.label == NLILabel.ENTAILMENT

        # Test neutral wins
        mock3 = MockNLIModel(
            default_scores={"contradiction": 0.2, "entailment": 0.3, "neutral": 0.5}
        )
        detector3 = NLIDetector(model=mock3)
        result3 = detector3.detect_contradiction("A", "B")
        assert result3.label == NLILabel.NEUTRAL


class TestNLIDetectorEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()

    def test_empty_texts(self):
        """Test handling of empty texts."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock)

        result = detector.detect_contradiction("", "")
        assert result is not None

    def test_very_long_texts(self):
        """Test that long texts are handled."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock)

        long_text = "A" * 10000
        result = detector.detect_contradiction(long_text, long_text)
        assert result is not None

    def test_special_characters(self):
        """Test texts with special characters."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock)

        result = detector.detect_contradiction(
            "Text with émojis 🎉 and symbols @#$%",
            "Another text with ñ and ü"
        )
        assert result is not None

    def test_unicode_texts(self):
        """Test unicode text handling."""
        mock = MockNLIModel()
        detector = NLIDetector(model=mock)

        result = detector.detect_contradiction(
            "日本語テキスト",
            "中文文本"
        )
        assert result is not None


class TestNLIResultHashability:
    """Tests for NLIResult hashability and equality."""

    def test_result_hashable(self):
        """Test that NLIResult is hashable (frozen)."""
        result = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.NEUTRAL,
            contradiction_score=0.1,
            entailment_score=0.1,
            neutral_score=0.8,
        )
        # Should not raise
        hash_val = hash(result)
        assert isinstance(hash_val, int)

    def test_result_equality(self):
        """Test NLIResult equality."""
        result1 = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.NEUTRAL,
            contradiction_score=0.1,
            entailment_score=0.1,
            neutral_score=0.8,
        )
        result2 = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.NEUTRAL,
            contradiction_score=0.1,
            entailment_score=0.1,
            neutral_score=0.8,
        )
        assert result1 == result2

    def test_result_in_set(self):
        """Test NLIResult can be used in sets."""
        result1 = NLIResult(
            premise="A",
            hypothesis="B",
            label=NLILabel.NEUTRAL,
            contradiction_score=0.1,
            entailment_score=0.1,
            neutral_score=0.8,
        )
        result2 = NLIResult(
            premise="C",
            hypothesis="D",
            label=NLILabel.NEUTRAL,
            contradiction_score=0.1,
            entailment_score=0.1,
            neutral_score=0.8,
        )

        result_set = {result1, result2}
        assert len(result_set) == 2
