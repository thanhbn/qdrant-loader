"""
NLI-based Contradiction Detector for CDI Conflict Detection.

This module provides Natural Language Inference (NLI) capabilities
for detecting semantic contradictions between text pairs using
transformer-based cross-encoder models.

Key Features:
- Lazy model loading for resource efficiency
- Batch processing for performance
- Graceful fallback when model unavailable
- LRU caching for repeated comparisons

Workflow ID: W4-cdi-conflict-cff57ee0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from .config import get_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class NLILabel(Enum):
    """NLI classification labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class NLIResult:
    """Result of NLI classification for a text pair.

    Attributes:
        premise: The first text (premise)
        hypothesis: The second text (hypothesis)
        label: The predicted NLI label
        contradiction_score: Probability of contradiction (0-1)
        entailment_score: Probability of entailment (0-1)
        neutral_score: Probability of neutral (0-1)
        is_contradiction: Whether this is classified as a contradiction
        confidence: Confidence score for the predicted label
    """

    premise: str
    hypothesis: str
    label: NLILabel
    contradiction_score: float
    entailment_score: float
    neutral_score: float

    @property
    def is_contradiction(self) -> bool:
        """Check if this is a contradiction based on config threshold.

        Per AFEV/ContraDoc research: If entailment score exceeds threshold,
        the texts are semantically compatible (one supports/agrees with the other),
        so they should NOT be considered contradictory even if contradiction
        score is moderately high.

        Entailment filtering prevents false positives where texts discuss
        the same topic with compatible (not conflicting) information.
        """
        config = get_config()

        # ENTAILMENT FILTERING (per Internal Audit 2026-01-08)
        # If texts show high entailment, they are compatible, not conflicting
        if self.entailment_score >= config.nli_entailment_threshold:
            return False

        return self.contradiction_score >= config.nli_contradiction_threshold

    @property
    def is_entailment(self) -> bool:
        """Check if this is an entailment based on config threshold."""
        config = get_config()
        return self.entailment_score >= config.nli_entailment_threshold

    @property
    def is_compatible(self) -> bool:
        """Check if texts are semantically compatible (not conflicting).

        Returns True if:
        - Entailment score is high (one text supports/agrees with the other)
        - Or contradiction score is low (texts don't contradict)

        This is the inverse of is_contradiction but provides a clearer
        semantic meaning for conflict detection pipelines.
        """
        return not self.is_contradiction

    @property
    def confidence(self) -> float:
        """Get confidence score for the predicted label."""
        if self.label == NLILabel.CONTRADICTION:
            return self.contradiction_score
        elif self.label == NLILabel.ENTAILMENT:
            return self.entailment_score
        return self.neutral_score


class NLIModelProtocol(Protocol):
    """Protocol for NLI model implementations."""

    def predict(
        self, text_pairs: list[tuple[str, str]], **kwargs: Any
    ) -> list[dict[str, float]]:
        """Predict NLI labels for text pairs.

        Args:
            text_pairs: List of (premise, hypothesis) tuples
            **kwargs: Additional model-specific arguments

        Returns:
            List of dictionaries with 'contradiction', 'entailment', 'neutral' scores
        """
        ...


class MockNLIModel:
    """Mock NLI model for testing and fallback.

    Returns neutral predictions with configurable behavior.
    """

    def __init__(self, default_scores: dict[str, float] | None = None):
        """Initialize mock model.

        Args:
            default_scores: Default scores to return (defaults to neutral)
        """
        self.default_scores = default_scores or {
            "contradiction": 0.1,
            "entailment": 0.1,
            "neutral": 0.8,
        }
        self.call_count = 0
        self.last_inputs: list[tuple[str, str]] = []

    def predict(
        self, text_pairs: list[tuple[str, str]], **kwargs: Any
    ) -> list[dict[str, float]]:
        """Return mock predictions."""
        self.call_count += 1
        self.last_inputs = text_pairs
        return [self.default_scores.copy() for _ in text_pairs]


class CrossEncoderNLIModel:
    """NLI model using sentence-transformers CrossEncoder.

    This model uses the cross-encoder/nli-deberta-v3-small model
    for high-quality NLI classification.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the CrossEncoder model.

        Args:
            model_name: Model name/path (defaults to config value)
        """
        self._model = None
        self._model_name = model_name
        self._label_mapping = ["contradiction", "entailment", "neutral"]

    @property
    def model_name(self) -> str:
        """Get the model name."""
        if self._model_name is None:
            config = get_config()
            self._model_name = config.nli_model_name
        return self._model_name

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading NLI model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("NLI model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise

    def predict(
        self, text_pairs: list[tuple[str, str]], **kwargs: Any
    ) -> list[dict[str, float]]:
        """Predict NLI labels for text pairs.

        Args:
            text_pairs: List of (premise, hypothesis) tuples
            **kwargs: Additional arguments (e.g., batch_size)

        Returns:
            List of score dictionaries
        """
        if not text_pairs:
            return []

        self._load_model()

        config = get_config()
        batch_size = kwargs.get("batch_size", config.nli_batch_size)

        # Truncate texts to max length
        max_len = config.nli_max_text_length
        truncated_pairs = [
            (premise[:max_len], hypothesis[:max_len])
            for premise, hypothesis in text_pairs
        ]

        try:
            # CrossEncoder.predict returns logits, need softmax for probabilities
            import numpy as np

            scores = self._model.predict(
                truncated_pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )

            # Apply softmax to convert logits to probabilities
            if len(scores.shape) == 1:
                # Single prediction
                scores = scores.reshape(1, -1)

            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            results = []
            for prob_row in probs:
                results.append(
                    {
                        label: float(prob_row[i])
                        for i, label in enumerate(self._label_mapping)
                    }
                )
            return results

        except Exception as e:
            logger.error(f"NLI prediction failed: {e}")
            # Return neutral on error
            return [
                {"contradiction": 0.1, "entailment": 0.1, "neutral": 0.8}
                for _ in text_pairs
            ]


class NLIDetector:
    """Natural Language Inference detector for contradiction detection.

    This class provides a high-level interface for detecting semantic
    contradictions between text pairs using NLI models.

    Example:
        detector = NLIDetector()
        result = detector.detect_contradiction(
            "The sky is blue.",
            "The sky is red."
        )
        if result.is_contradiction:
            print("Texts contradict each other!")
    """

    def __init__(
        self, model: NLIModelProtocol | None = None, use_cache: bool = True
    ):
        """Initialize the NLI detector.

        Args:
            model: NLI model to use (auto-detected if None)
            use_cache: Whether to use LRU cache for results
        """
        self._model = model
        self._use_cache = use_cache
        self._cache: dict[tuple[str, str], NLIResult] = {}
        self._model_available: bool | None = None

    def _get_model(self) -> NLIModelProtocol:
        """Get or create the NLI model."""
        if self._model is not None:
            return self._model

        config = get_config()

        # Check if NLI is enabled
        if not config.use_nli_model:
            logger.debug("NLI model disabled in config")
            self._model = MockNLIModel()
            return self._model

        # Try to load the real model
        try:
            self._model = CrossEncoderNLIModel()
            self._model_available = True
        except ImportError:
            logger.warning("Using mock NLI model - sentence-transformers not available")
            self._model = MockNLIModel()
            self._model_available = False

        return self._model

    @property
    def is_model_available(self) -> bool:
        """Check if the real NLI model is available."""
        if self._model_available is None:
            self._get_model()
        return self._model_available or False

    def _create_cache_key(self, premise: str, hypothesis: str) -> tuple[str, str]:
        """Create a cache key for text pair.

        Normalizes texts for consistent caching.
        """
        # Normalize whitespace
        p = " ".join(premise.lower().split())
        h = " ".join(hypothesis.lower().split())
        return (p, h)

    def detect_contradiction(
        self, premise: str, hypothesis: str
    ) -> NLIResult:
        """Detect if hypothesis contradicts premise.

        Args:
            premise: The first text (premise/reference)
            hypothesis: The second text (hypothesis/claim)

        Returns:
            NLIResult with classification scores
        """
        # Check cache
        if self._use_cache:
            cache_key = self._create_cache_key(premise, hypothesis)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Get prediction
        model = self._get_model()
        predictions = model.predict([(premise, hypothesis)])

        if not predictions:
            # Fallback result
            result = NLIResult(
                premise=premise,
                hypothesis=hypothesis,
                label=NLILabel.NEUTRAL,
                contradiction_score=0.0,
                entailment_score=0.0,
                neutral_score=1.0,
            )
        else:
            scores = predictions[0]
            label = self._get_label_from_scores(scores)
            result = NLIResult(
                premise=premise,
                hypothesis=hypothesis,
                label=label,
                contradiction_score=scores.get("contradiction", 0.0),
                entailment_score=scores.get("entailment", 0.0),
                neutral_score=scores.get("neutral", 0.0),
            )

        # Cache result
        if self._use_cache:
            cache_key = self._create_cache_key(premise, hypothesis)
            self._cache[cache_key] = result

        return result

    def detect_contradictions_batch(
        self, text_pairs: list[tuple[str, str]]
    ) -> list[NLIResult]:
        """Detect contradictions for multiple text pairs.

        More efficient than calling detect_contradiction repeatedly
        as it processes all pairs in a single batch.

        Args:
            text_pairs: List of (premise, hypothesis) tuples

        Returns:
            List of NLIResults in same order as input
        """
        if not text_pairs:
            return []

        # Separate cached and uncached pairs
        results: list[NLIResult | None] = [None] * len(text_pairs)
        uncached_pairs: list[tuple[int, str, str]] = []

        for i, (premise, hypothesis) in enumerate(text_pairs):
            if self._use_cache:
                cache_key = self._create_cache_key(premise, hypothesis)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                    continue
            uncached_pairs.append((i, premise, hypothesis))

        # Process uncached pairs
        if uncached_pairs:
            model = self._get_model()
            pairs_to_predict = [(p, h) for _, p, h in uncached_pairs]
            predictions = model.predict(pairs_to_predict)

            for j, (i, premise, hypothesis) in enumerate(uncached_pairs):
                if j < len(predictions):
                    scores = predictions[j]
                    label = self._get_label_from_scores(scores)
                    result = NLIResult(
                        premise=premise,
                        hypothesis=hypothesis,
                        label=label,
                        contradiction_score=scores.get("contradiction", 0.0),
                        entailment_score=scores.get("entailment", 0.0),
                        neutral_score=scores.get("neutral", 0.0),
                    )
                else:
                    result = NLIResult(
                        premise=premise,
                        hypothesis=hypothesis,
                        label=NLILabel.NEUTRAL,
                        contradiction_score=0.0,
                        entailment_score=0.0,
                        neutral_score=1.0,
                    )

                results[i] = result

                # Cache result
                if self._use_cache:
                    cache_key = self._create_cache_key(premise, hypothesis)
                    self._cache[cache_key] = result

        return [r for r in results if r is not None]

    def _get_label_from_scores(self, scores: dict[str, float]) -> NLILabel:
        """Determine the NLI label from scores."""
        max_label = max(scores.items(), key=lambda x: x[1])
        label_name = max_label[0]

        if label_name == "contradiction":
            return NLILabel.CONTRADICTION
        elif label_name == "entailment":
            return NLILabel.ENTAILMENT
        return NLILabel.NEUTRAL

    def is_contradictory(self, premise: str, hypothesis: str) -> bool:
        """Quick check if texts contradict each other.

        Args:
            premise: The first text
            hypothesis: The second text

        Returns:
            True if texts are contradictory based on threshold
        """
        result = self.detect_contradiction(premise, hypothesis)
        return result.is_contradiction

    def get_contradiction_score(self, premise: str, hypothesis: str) -> float:
        """Get the contradiction probability for text pair.

        Args:
            premise: The first text
            hypothesis: The second text

        Returns:
            Contradiction probability (0-1)
        """
        result = self.detect_contradiction(premise, hypothesis)
        return result.contradiction_score

    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self._use_cache,
        }


# Global singleton instance
_nli_detector_instance: NLIDetector | None = None


def get_nli_detector() -> NLIDetector:
    """Get the global NLI detector instance."""
    global _nli_detector_instance
    if _nli_detector_instance is None:
        _nli_detector_instance = NLIDetector()
    return _nli_detector_instance


def reset_nli_detector():
    """Reset the global NLI detector instance (for testing)."""
    global _nli_detector_instance
    _nli_detector_instance = None


# Convenience functions for common operations
@lru_cache(maxsize=500)
def cached_contradiction_check(premise: str, hypothesis: str) -> bool:
    """Cached check for contradiction between texts.

    Uses LRU cache for efficiency with repeated checks.

    Args:
        premise: The first text
        hypothesis: The second text

    Returns:
        True if texts contradict each other
    """
    detector = get_nli_detector()
    return detector.is_contradictory(premise, hypothesis)


def detect_pairwise_contradictions(
    texts: list[str], labels: list[str] | None = None
) -> list[dict[str, Any]]:
    """Detect contradictions between all pairs of texts.

    Args:
        texts: List of texts to compare
        labels: Optional labels for each text

    Returns:
        List of contradiction results with text indices and scores
    """
    if len(texts) < 2:
        return []

    if labels is None:
        labels = [f"text_{i}" for i in range(len(texts))]

    detector = get_nli_detector()
    contradictions = []

    # Generate all pairs
    pairs = []
    pair_indices = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append((texts[i], texts[j]))
            pair_indices.append((i, j))

    # Batch process
    results = detector.detect_contradictions_batch(pairs)

    # Collect contradictions
    for (i, j), result in zip(pair_indices, results):
        if result.is_contradiction:
            contradictions.append(
                {
                    "text1_index": i,
                    "text2_index": j,
                    "text1_label": labels[i],
                    "text2_label": labels[j],
                    "contradiction_score": result.contradiction_score,
                    "label": result.label.value,
                }
            )

    return contradictions
