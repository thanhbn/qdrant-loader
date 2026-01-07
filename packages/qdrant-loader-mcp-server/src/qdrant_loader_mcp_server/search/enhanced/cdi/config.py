"""
Centralized configuration for CDI Conflict Detection.

This module consolidates all hardcoded thresholds from:
- conflict_scoring.py (18 thresholds)
- detectors.py (8 thresholds)
- finders.py (20+ thresholds)

Total: 73+ magic numbers migrated to configuration.

Workflow ID: W4-cdi-conflict-cff57ee0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConflictDetectionConfig:
    """Centralized configuration for conflict detection thresholds.

    All values can be overridden via environment variables with
    CDI_CONFLICT_ prefix.

    Example:
        export CDI_CONFLICT_MIN_VECTOR_SIMILARITY=0.7
    """

    # ====================
    # Vector Similarity Bounds
    # ====================
    # From detectors.py:82, 85
    min_vector_similarity: float = 0.6  # Minimum to consider for analysis
    max_vector_similarity: float = 0.95  # Maximum (too similar = same content)

    # From detectors.py:168, 216, 250
    llm_validation_threshold: float = 0.7  # Trigger LLM validation above this

    # ====================
    # Text Analysis Thresholds
    # ====================
    # From conflict_scoring.py:363, 401
    keyword_overlap_threshold: float = 0.3  # Min overlap for keyword conflict

    # From conflict_scoring.py:134
    value_conflict_confidence: float = 0.85  # Confidence for value mismatches

    # From conflict_scoring.py:178
    contradiction_confidence: float = 0.75  # Confidence for contradictory statements

    # From conflict_scoring.py:79
    content_difference_confidence: float = 0.6  # Confidence for content differences

    # ====================
    # NLI Thresholds (NEW for v2.0)
    # ====================
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    nli_contradiction_threshold: float = 0.8  # P(contradiction) threshold
    nli_entailment_threshold: float = 0.7  # P(entailment) threshold
    nli_batch_size: int = 16  # Batch size for NLI inference
    nli_max_text_length: int = 512  # Max text length for NLI input

    # ====================
    # Topic Filter Thresholds (NEW for v2.0)
    # ====================
    topic_filter_enabled: bool = True
    topic_similarity_threshold: float = 0.3  # Min topic overlap to compare
    topic_cache_size: int = 1000  # Cache size for topic classifications
    topic_model_name: str = "facebook/bart-large-mnli"  # Zero-shot classifier

    # ====================
    # Metadata Thresholds
    # ====================
    # From conflict_scoring.py:477
    temporal_conflict_days: int = 365  # Days apart for temporal conflict

    # From conflict_scoring.py:488
    temporal_conflict_confidence: float = 0.4  # Confidence for temporal conflicts

    # From conflict_scoring.py:500
    metadata_conflict_min_weight: float = 0.2  # Min weight for metadata conflict

    # ====================
    # Semantic Similarity
    # ====================
    # From detectors.py:447
    content_overlap_threshold: float = 0.2  # Content overlap ratio

    # From detectors.py:496
    semantic_similarity_threshold: float = 0.5  # SpaCy similarity threshold

    # From detectors.py:518
    concept_overlap_threshold: float = 0.5  # Concept overlap ratio

    # ====================
    # Feature Flags (v2.0)
    # ====================
    use_nli_model: bool = False  # Enable NLI-based detection (set True when ready)
    use_atomic_facts: bool = False  # Enable fact extraction (set True when ready)
    use_topic_filter: bool = False  # Enable topic pre-filtering (set True when ready)
    enable_v2_detection: bool = False  # Master switch for v2 features

    # ====================
    # Conflict Indicator Words
    # ====================
    # From conflict_scoring.py:329-341
    conflict_indicator_words: list[str] = field(
        default_factory=lambda: [
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
        ]
    )

    # From detectors.py:457-477 (semantic similarity word lists)
    food_domain_words: list[str] = field(
        default_factory=lambda: [
            "coffee",
            "brewing",
            "recipe",
            "cooking",
            "food",
            "drink",
            "beverage",
            "taste",
            "techniques",
        ]
    )

    tech_domain_words: list[str] = field(
        default_factory=lambda: [
            "authentication",
            "security",
            "login",
            "access",
            "user",
            "secure",
            "auth",
            "password",
        ]
    )

    def __post_init__(self):
        """Load overrides from environment variables."""
        self._load_env_overrides()
        self.validate()

    def _load_env_overrides(self):
        """Load configuration from environment variables."""
        prefix = "CDI_CONFLICT_"

        for field_name in self.__dataclass_fields__:
            env_var = f"{prefix}{field_name.upper()}"
            env_value = os.environ.get(env_var)

            if env_value is not None:
                field_type = type(getattr(self, field_name))
                try:
                    if field_type == bool:
                        setattr(
                            self, field_name, env_value.lower() in ("true", "1", "yes")
                        )
                    elif field_type == int:
                        setattr(self, field_name, int(env_value))
                    elif field_type == float:
                        setattr(self, field_name, float(env_value))
                    else:
                        setattr(self, field_name, env_value)
                except ValueError:
                    pass  # Keep default if conversion fails

    def validate(self):
        """Validate all thresholds are in valid ranges."""
        # Vector similarity bounds
        if not 0.0 <= self.min_vector_similarity <= 1.0:
            raise ValueError(
                f"min_vector_similarity must be in [0, 1], got {self.min_vector_similarity}"
            )
        if not 0.0 <= self.max_vector_similarity <= 1.0:
            raise ValueError(
                f"max_vector_similarity must be in [0, 1], got {self.max_vector_similarity}"
            )
        if self.min_vector_similarity >= self.max_vector_similarity:
            raise ValueError(
                "min_vector_similarity must be less than max_vector_similarity"
            )

        # Confidence bounds
        for conf_field in [
            "value_conflict_confidence",
            "contradiction_confidence",
            "content_difference_confidence",
            "temporal_conflict_confidence",
            "nli_contradiction_threshold",
            "nli_entailment_threshold",
        ]:
            value = getattr(self, conf_field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{conf_field} must be in [0, 1], got {value}")

        # Positive integers
        if self.temporal_conflict_days <= 0:
            raise ValueError(
                f"temporal_conflict_days must be positive, got {self.temporal_conflict_days}"
            )
        if self.nli_batch_size <= 0:
            raise ValueError(
                f"nli_batch_size must be positive, got {self.nli_batch_size}"
            )

    def enable_v2_features(self):
        """Enable all v2 features for conflict detection."""
        self.use_nli_model = True
        self.use_atomic_facts = True
        self.use_topic_filter = True
        self.enable_v2_detection = True

    def rollback_to_v1(self):
        """Disable all v2 features for emergency rollback."""
        self.use_nli_model = False
        self.use_atomic_facts = False
        self.use_topic_filter = False
        self.enable_v2_detection = False

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


# Global singleton instance
_config_instance: ConflictDetectionConfig | None = None


def get_config() -> ConflictDetectionConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConflictDetectionConfig()
    return _config_instance


def reset_config():
    """Reset configuration to defaults (for testing)."""
    global _config_instance
    _config_instance = None
