"""Tests for CDI conflict detection configuration.

Workflow ID: W4-cdi-conflict-cff57ee0
"""

import os
from unittest.mock import patch

import pytest

from qdrant_loader_mcp_server.search.enhanced.cdi.config import (
    ConflictDetectionConfig,
    get_config,
    reset_config,
)


class TestConflictDetectionConfig:
    """Test suite for ConflictDetectionConfig dataclass."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()

    def test_default_values(self):
        """Test that default config values are set correctly."""
        config = ConflictDetectionConfig()

        # Vector similarity bounds
        assert config.min_vector_similarity == 0.6
        assert config.max_vector_similarity == 0.95
        assert config.llm_validation_threshold == 0.7

        # Text analysis thresholds
        assert config.keyword_overlap_threshold == 0.3
        assert config.value_conflict_confidence == 0.85
        assert config.contradiction_confidence == 0.75
        # DOWNWEIGHTED from 0.6 to 0.3 per Internal Audit (2026-01-08)
        # Rationale: "content difference" ≠ "contradiction" (ContraDoc, AFEV papers)
        assert config.content_difference_confidence == 0.3

        # NLI thresholds
        assert config.nli_model_name == "cross-encoder/nli-deberta-v3-small"
        assert config.nli_contradiction_threshold == 0.8
        assert config.nli_entailment_threshold == 0.7
        assert config.nli_batch_size == 16
        assert config.nli_max_text_length == 512

        # Topic filter thresholds
        assert config.topic_filter_enabled is True
        assert config.topic_similarity_threshold == 0.3
        assert config.topic_cache_size == 1000

        # Metadata thresholds
        assert config.temporal_conflict_days == 365
        assert config.temporal_conflict_confidence == 0.4
        assert config.metadata_conflict_min_weight == 0.2

        # Semantic similarity
        assert config.content_overlap_threshold == 0.2
        assert config.semantic_similarity_threshold == 0.5
        assert config.concept_overlap_threshold == 0.5

        # Feature flags (v2.0) - ENABLED by default per Internal Audit (2026-01-08)
        # Rationale: V2 is research-backed (ContraDoc, AFEV), V1 keyword-based is flawed
        assert config.use_nli_model is True
        assert config.use_atomic_facts is True
        assert config.use_topic_filter is True
        assert config.enable_v2_detection is True

    def test_conflict_indicator_words_default(self):
        """Test default conflict indicator words list."""
        config = ConflictDetectionConfig()

        expected_words = [
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
        assert config.conflict_indicator_words == expected_words

    def test_domain_words_defaults(self):
        """Test default domain word lists."""
        config = ConflictDetectionConfig()

        assert "coffee" in config.food_domain_words
        assert "brewing" in config.food_domain_words
        assert "recipe" in config.food_domain_words

        assert "authentication" in config.tech_domain_words
        assert "security" in config.tech_domain_words
        assert "login" in config.tech_domain_words

    def test_validation_min_vector_similarity_out_of_range(self):
        """Test validation fails for min_vector_similarity out of range."""
        with pytest.raises(ValueError, match="min_vector_similarity must be in"):
            ConflictDetectionConfig(min_vector_similarity=-0.1)

        with pytest.raises(ValueError, match="min_vector_similarity must be in"):
            ConflictDetectionConfig(min_vector_similarity=1.5)

    def test_validation_max_vector_similarity_out_of_range(self):
        """Test validation fails for max_vector_similarity out of range."""
        with pytest.raises(ValueError, match="max_vector_similarity must be in"):
            ConflictDetectionConfig(max_vector_similarity=-0.1)

        with pytest.raises(ValueError, match="max_vector_similarity must be in"):
            ConflictDetectionConfig(max_vector_similarity=1.5)

    def test_validation_min_must_be_less_than_max(self):
        """Test validation fails when min >= max."""
        with pytest.raises(
            ValueError, match="min_vector_similarity must be less than max_vector_similarity"
        ):
            ConflictDetectionConfig(min_vector_similarity=0.9, max_vector_similarity=0.8)

        with pytest.raises(
            ValueError, match="min_vector_similarity must be less than max_vector_similarity"
        ):
            ConflictDetectionConfig(min_vector_similarity=0.8, max_vector_similarity=0.8)

    def test_validation_confidence_bounds(self):
        """Test validation for confidence fields."""
        confidence_fields = [
            "value_conflict_confidence",
            "contradiction_confidence",
            "content_difference_confidence",
            "temporal_conflict_confidence",
            "nli_contradiction_threshold",
            "nli_entailment_threshold",
        ]

        for field_name in confidence_fields:
            # Test negative value
            with pytest.raises(ValueError, match=f"{field_name} must be in"):
                ConflictDetectionConfig(**{field_name: -0.1})

            # Test value > 1
            with pytest.raises(ValueError, match=f"{field_name} must be in"):
                ConflictDetectionConfig(**{field_name: 1.1})

    def test_validation_temporal_conflict_days_positive(self):
        """Test validation requires positive temporal_conflict_days."""
        with pytest.raises(ValueError, match="temporal_conflict_days must be positive"):
            ConflictDetectionConfig(temporal_conflict_days=0)

        with pytest.raises(ValueError, match="temporal_conflict_days must be positive"):
            ConflictDetectionConfig(temporal_conflict_days=-10)

    def test_validation_nli_batch_size_positive(self):
        """Test validation requires positive nli_batch_size."""
        with pytest.raises(ValueError, match="nli_batch_size must be positive"):
            ConflictDetectionConfig(nli_batch_size=0)

        with pytest.raises(ValueError, match="nli_batch_size must be positive"):
            ConflictDetectionConfig(nli_batch_size=-5)

    def test_enable_v2_features(self):
        """Test enable_v2_features enables all v2 flags."""
        config = ConflictDetectionConfig()

        # First rollback to v1 to test enable
        config.rollback_to_v1()
        assert config.use_nli_model is False

        # Enable v2
        config.enable_v2_features()

        # All should be True
        assert config.use_nli_model is True
        assert config.use_atomic_facts is True
        assert config.use_topic_filter is True
        assert config.enable_v2_detection is True

    def test_rollback_to_v1(self):
        """Test rollback_to_v1 disables all v2 flags."""
        config = ConflictDetectionConfig()

        # All True by default (v2 enabled)
        assert config.enable_v2_detection is True

        # Rollback
        config.rollback_to_v1()

        # All should be False
        assert config.use_nli_model is False
        assert config.use_atomic_facts is False
        assert config.use_topic_filter is False
        assert config.enable_v2_detection is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = ConflictDetectionConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "min_vector_similarity" in config_dict
        assert "max_vector_similarity" in config_dict
        assert "nli_model_name" in config_dict
        assert "conflict_indicator_words" in config_dict

        # Verify values match
        assert config_dict["min_vector_similarity"] == 0.6
        assert config_dict["max_vector_similarity"] == 0.95

    def test_env_override_float(self):
        """Test environment variable override for float values."""
        with patch.dict(os.environ, {"CDI_CONFLICT_MIN_VECTOR_SIMILARITY": "0.75"}):
            reset_config()
            config = ConflictDetectionConfig()
            assert config.min_vector_similarity == 0.75

    def test_env_override_int(self):
        """Test environment variable override for int values."""
        with patch.dict(os.environ, {"CDI_CONFLICT_TEMPORAL_CONFLICT_DAYS": "180"}):
            reset_config()
            config = ConflictDetectionConfig()
            assert config.temporal_conflict_days == 180

    def test_env_override_bool_true(self):
        """Test environment variable override for bool values (true)."""
        for true_value in ["true", "True", "TRUE", "1", "yes"]:
            with patch.dict(os.environ, {"CDI_CONFLICT_USE_NLI_MODEL": true_value}):
                reset_config()
                config = ConflictDetectionConfig()
                assert config.use_nli_model is True

    def test_env_override_bool_false(self):
        """Test environment variable override for bool values (false)."""
        for false_value in ["false", "False", "FALSE", "0", "no"]:
            with patch.dict(os.environ, {"CDI_CONFLICT_TOPIC_FILTER_ENABLED": false_value}):
                reset_config()
                config = ConflictDetectionConfig()
                assert config.topic_filter_enabled is False

    def test_env_override_string(self):
        """Test environment variable override for string values."""
        with patch.dict(
            os.environ, {"CDI_CONFLICT_NLI_MODEL_NAME": "custom/model-name"}
        ):
            reset_config()
            config = ConflictDetectionConfig()
            assert config.nli_model_name == "custom/model-name"

    def test_env_override_invalid_keeps_default(self):
        """Test that invalid env values keep defaults."""
        with patch.dict(os.environ, {"CDI_CONFLICT_MIN_VECTOR_SIMILARITY": "not_a_number"}):
            reset_config()
            config = ConflictDetectionConfig()
            # Should keep default since "not_a_number" can't be converted to float
            assert config.min_vector_similarity == 0.6


class TestConfigSingleton:
    """Test suite for config singleton pattern."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()

    def test_get_config_returns_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """Test that reset_config clears the singleton."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_modifications_persist_in_singleton(self):
        """Test that modifications to singleton persist."""
        config1 = get_config()
        # V2 is enabled by default, so test rollback persistence
        config1.rollback_to_v1()

        config2 = get_config()
        assert config2.use_nli_model is False
        assert config2.enable_v2_detection is False


class TestConfigIntegration:
    """Integration tests for config with other modules."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()

    def test_import_from_cdi_package(self):
        """Test that config can be imported from cdi package."""
        from qdrant_loader_mcp_server.search.enhanced.cdi import (
            ConflictDetectionConfig,
            get_config,
            reset_config,
        )

        config = get_config()
        assert isinstance(config, ConflictDetectionConfig)

    def test_config_used_by_conflict_scoring(self):
        """Test that conflict_scoring uses centralized config."""
        # This test verifies the import works - actual behavior tested elsewhere
        from qdrant_loader_mcp_server.search.enhanced.cdi.conflict_scoring import (
            extract_conflict_snippets,
        )

        # Should not raise ImportError
        assert callable(extract_conflict_snippets)

    def test_config_used_by_detectors(self):
        """Test that detectors uses centralized config."""
        # This test verifies the import works
        from qdrant_loader_mcp_server.search.enhanced.cdi.detectors import (
            ConflictDetector,
        )

        # Should not raise ImportError
        assert ConflictDetector is not None
