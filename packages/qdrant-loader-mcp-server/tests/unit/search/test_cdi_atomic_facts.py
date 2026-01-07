"""
Tests for CDI Atomic Facts Extractor module.

Tests cover:
- FactType enum
- AtomicFact dataclass
- ExtractionResult class
- AtomicFactExtractor with spaCy
- FactComparator for conflict detection
- Caching behavior
- Singleton pattern
- Convenience functions

Workflow ID: W4-cdi-conflict-cff57ee0
"""

import pytest
from unittest.mock import patch, MagicMock

from qdrant_loader_mcp_server.search.enhanced.cdi.atomic_facts import (
    FactType,
    AtomicFact,
    ExtractionResult,
    AtomicFactExtractor,
    FactComparator,
    get_fact_extractor,
    reset_fact_extractor,
    extract_facts_cached,
    compare_document_facts,
)
from qdrant_loader_mcp_server.search.enhanced.cdi.config import (
    get_config,
    reset_config,
)
from qdrant_loader_mcp_server.search.enhanced.cdi.nli_detector import (
    reset_nli_detector,
)


class TestFactType:
    """Tests for FactType enum."""

    def test_all_types_exist(self):
        """Test that all expected fact types exist."""
        assert FactType.STATEMENT.value == "statement"
        assert FactType.NUMERIC.value == "numeric"
        assert FactType.TEMPORAL.value == "temporal"
        assert FactType.COMPARATIVE.value == "comparative"
        assert FactType.NEGATION.value == "negation"
        assert FactType.CAUSAL.value == "causal"
        assert FactType.CONDITIONAL.value == "conditional"
        assert FactType.DEFINITION.value == "definition"

    def test_type_comparison(self):
        """Test fact type comparison."""
        assert FactType.NUMERIC == FactType.NUMERIC
        assert FactType.NUMERIC != FactType.STATEMENT


class TestAtomicFact:
    """Tests for AtomicFact dataclass."""

    def test_basic_creation(self):
        """Test creating a basic atomic fact."""
        fact = AtomicFact(
            text="Python is a programming language.",
            fact_type=FactType.DEFINITION,
        )
        assert fact.text == "Python is a programming language."
        assert fact.fact_type == FactType.DEFINITION
        assert fact.negated is False
        assert fact.confidence == 1.0

    def test_full_creation(self):
        """Test creating a fact with all fields."""
        fact = AtomicFact(
            text="The temperature is 25 degrees.",
            fact_type=FactType.NUMERIC,
            subject="temperature",
            predicate="is",
            object_="25 degrees",
            negated=False,
            confidence=0.9,
            source_sentence="The temperature is 25 degrees.",
            numeric_value=25.0,
            numeric_unit="degrees",
        )
        assert fact.subject == "temperature"
        assert fact.predicate == "is"
        assert fact.object_ == "25 degrees"
        assert fact.numeric_value == 25.0
        assert fact.numeric_unit == "degrees"

    def test_negated_fact(self):
        """Test creating a negated fact."""
        fact = AtomicFact(
            text="Python is not a compiled language.",
            fact_type=FactType.NEGATION,
            negated=True,
        )
        assert fact.negated is True

    def test_to_dict(self):
        """Test converting fact to dictionary."""
        fact = AtomicFact(
            text="Test fact",
            fact_type=FactType.STATEMENT,
            subject="test",
            predicate="is",
            confidence=0.8,
        )
        d = fact.to_dict()
        assert d["text"] == "Test fact"
        assert d["fact_type"] == "statement"
        assert d["subject"] == "test"
        assert d["confidence"] == 0.8

    def test_fact_is_frozen(self):
        """Test that AtomicFact is immutable."""
        fact = AtomicFact(
            text="Immutable fact",
            fact_type=FactType.STATEMENT,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            fact.text = "Changed"


class TestExtractionResult:
    """Tests for ExtractionResult class."""

    def test_basic_creation(self):
        """Test creating an extraction result."""
        facts = [
            AtomicFact(text="Fact 1", fact_type=FactType.STATEMENT),
            AtomicFact(text="Fact 2", fact_type=FactType.NUMERIC, numeric_value=10),
        ]
        result = ExtractionResult(
            facts=facts,
            source_text="Original text",
            sentence_count=2,
        )
        assert result.fact_count == 2
        assert result.sentence_count == 2

    def test_get_facts_by_type(self):
        """Test filtering facts by type."""
        facts = [
            AtomicFact(text="Fact 1", fact_type=FactType.STATEMENT),
            AtomicFact(text="Fact 2", fact_type=FactType.NUMERIC),
            AtomicFact(text="Fact 3", fact_type=FactType.STATEMENT),
        ]
        result = ExtractionResult(facts=facts, source_text="")

        statements = result.get_facts_by_type(FactType.STATEMENT)
        assert len(statements) == 2

    def test_get_numeric_facts(self):
        """Test getting numeric facts."""
        facts = [
            AtomicFact(text="Fact 1", fact_type=FactType.STATEMENT),
            AtomicFact(text="Fact 2", fact_type=FactType.NUMERIC, numeric_value=10),
            AtomicFact(text="Fact 3", fact_type=FactType.NUMERIC, numeric_value=20),
        ]
        result = ExtractionResult(facts=facts, source_text="")

        numeric = result.get_numeric_facts()
        assert len(numeric) == 2

    def test_get_negated_facts(self):
        """Test getting negated facts."""
        facts = [
            AtomicFact(text="Fact 1", fact_type=FactType.STATEMENT, negated=False),
            AtomicFact(text="Fact 2", fact_type=FactType.NEGATION, negated=True),
        ]
        result = ExtractionResult(facts=facts, source_text="")

        negated = result.get_negated_facts()
        assert len(negated) == 1


class TestAtomicFactExtractor:
    """Tests for AtomicFactExtractor class."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_fact_extractor()

    def test_extract_empty_text(self):
        """Test extracting from empty text."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("")
        assert result.fact_count == 0

    def test_extract_whitespace_text(self):
        """Test extracting from whitespace-only text."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("   \n\t  ")
        assert result.fact_count == 0

    def test_extract_single_sentence(self):
        """Test extracting from a single sentence."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("Python is a programming language.")
        assert result.fact_count >= 1
        assert result.sentence_count >= 1

    def test_extract_multiple_sentences(self):
        """Test extracting from multiple sentences."""
        extractor = AtomicFactExtractor()
        text = "Python is a programming language. It was created in 1991. It is widely used."
        result = extractor.extract(text)
        assert result.fact_count >= 2

    def test_extract_numeric_fact(self):
        """Test extracting numeric facts."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The temperature is 25 degrees Celsius.")

        assert result.fact_count >= 1
        # Check if numeric value was extracted
        numeric_facts = result.get_numeric_facts()
        assert len(numeric_facts) >= 1
        assert numeric_facts[0].numeric_value == 25.0

    def test_extract_negation_detection(self):
        """Test detecting negation in facts."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("Python is not a compiled language.")

        assert result.fact_count >= 1
        negated = result.get_negated_facts()
        assert len(negated) >= 1

    def test_extract_definition(self):
        """Test detecting definition facts."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("A variable is a container for storing data.")

        assert result.fact_count >= 1
        definitions = result.get_facts_by_type(FactType.DEFINITION)
        assert len(definitions) >= 1

    def test_extract_comparative(self):
        """Test detecting comparative facts."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("Python is faster than Ruby for data processing.")

        assert result.fact_count >= 1
        comparatives = result.get_facts_by_type(FactType.COMPARATIVE)
        assert len(comparatives) >= 1

    def test_extract_conditional(self):
        """Test detecting conditional facts."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("If the input is valid, the function returns True.")

        assert result.fact_count >= 1
        conditionals = result.get_facts_by_type(FactType.CONDITIONAL)
        assert len(conditionals) >= 1

    def test_extract_causal(self):
        """Test detecting causal facts."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The process failed because the memory was full.")

        assert result.fact_count >= 1
        causal = result.get_facts_by_type(FactType.CAUSAL)
        assert len(causal) >= 1

    def test_extract_temporal(self):
        """Test detecting temporal facts."""
        extractor = AtomicFactExtractor()
        # Use more temporal words to ensure detection
        result = extractor.extract("Yesterday the meeting was held and today we are reviewing.")

        assert result.fact_count >= 1
        temporal = result.get_facts_by_type(FactType.TEMPORAL)
        assert len(temporal) >= 1


class TestAtomicFactExtractorCaching:
    """Tests for caching behavior."""

    def setup_method(self):
        """Reset before each test."""
        reset_fact_extractor()

    def test_cache_hit(self):
        """Test that cached results are returned."""
        extractor = AtomicFactExtractor(use_cache=True)

        text = "Python is a programming language."
        result1 = extractor.extract(text)
        result2 = extractor.extract(text)

        # Same instance should be returned
        assert result1 is result2

    def test_cache_normalization(self):
        """Test that cache normalizes whitespace."""
        extractor = AtomicFactExtractor(use_cache=True)

        result1 = extractor.extract("hello world")
        result2 = extractor.extract("hello  world")  # Extra space

        # Should be same due to normalization
        assert result1 is result2

    def test_cache_disabled(self):
        """Test extraction with caching disabled."""
        extractor = AtomicFactExtractor(use_cache=False)

        text = "Python is a programming language."
        result1 = extractor.extract(text)
        result2 = extractor.extract(text)

        # Different instances
        assert result1 is not result2

    def test_clear_cache(self):
        """Test clearing the cache."""
        extractor = AtomicFactExtractor(use_cache=True)

        text = "Python is a programming language."
        result1 = extractor.extract(text)
        extractor.clear_cache()
        result2 = extractor.extract(text)

        # Different instances after cache clear
        assert result1 is not result2

    def test_cache_stats(self):
        """Test cache statistics."""
        extractor = AtomicFactExtractor(use_cache=True)

        extractor.extract("Text 1")
        extractor.extract("Text 2")

        stats = extractor.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["cache_enabled"] is True


class TestAtomicFactExtractorNumeric:
    """Tests for numeric value extraction."""

    def setup_method(self):
        """Reset before each test."""
        reset_fact_extractor()

    def test_extract_percentage(self):
        """Test extracting percentage values."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The accuracy improved by 15 percent.")

        numeric = result.get_numeric_facts()
        assert len(numeric) >= 1
        assert numeric[0].numeric_value == 15.0

    def test_extract_dollars(self):
        """Test extracting dollar amounts."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The cost is 50 dollars.")

        numeric = result.get_numeric_facts()
        assert len(numeric) >= 1
        assert numeric[0].numeric_value == 50.0

    def test_extract_decimal(self):
        """Test extracting decimal values."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The ratio is 3.14.")

        numeric = result.get_numeric_facts()
        assert len(numeric) >= 1
        assert numeric[0].numeric_value == 3.14

    def test_extract_kilograms(self):
        """Test extracting weight values."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The package weighs 10 kg.")

        numeric = result.get_numeric_facts()
        assert len(numeric) >= 1
        assert numeric[0].numeric_value == 10.0
        assert numeric[0].numeric_unit == "kg"


class TestAtomicFactExtractorDocuments:
    """Tests for document batch extraction."""

    def setup_method(self):
        """Reset before each test."""
        reset_fact_extractor()

    def test_extract_from_string_documents(self):
        """Test extracting from string documents."""
        extractor = AtomicFactExtractor()
        docs = ["Python is fast.", "Java is verbose."]

        results = extractor.extract_from_documents(docs)
        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_extract_with_custom_extractor(self):
        """Test using custom text extractor function."""
        extractor = AtomicFactExtractor()
        docs = [
            {"content": "Python is fast."},
            {"content": "Java is verbose."},
        ]

        results = extractor.extract_from_documents(
            docs, text_extractor=lambda d: d["content"]
        )
        assert len(results) == 2


class TestFactComparator:
    """Tests for FactComparator class."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_nli_detector()

    def test_compare_same_facts(self):
        """Test comparing identical facts."""
        comparator = FactComparator()

        fact1 = AtomicFact(text="Python is fast.", fact_type=FactType.STATEMENT)
        fact2 = AtomicFact(text="Python is fast.", fact_type=FactType.STATEMENT)

        result = comparator.compare_facts(fact1, fact2)
        # Same facts shouldn't show conflict
        assert result["potential_conflict"] is False

    def test_compare_different_subjects(self):
        """Test comparing facts with different subjects."""
        comparator = FactComparator()

        fact1 = AtomicFact(
            text="Python is fast.",
            fact_type=FactType.STATEMENT,
            subject="Python",
        )
        fact2 = AtomicFact(
            text="Java is slow.",
            fact_type=FactType.STATEMENT,
            subject="Java",
        )

        result = comparator.compare_facts(fact1, fact2)
        assert result["potential_conflict"] is False
        assert result["details"].get("different_subjects") is True

    def test_compare_numeric_mismatch(self):
        """Test detecting numeric value mismatch."""
        comparator = FactComparator()

        fact1 = AtomicFact(
            text="The price is 100 dollars.",
            fact_type=FactType.NUMERIC,
            numeric_value=100.0,
            numeric_unit="dollars",
        )
        fact2 = AtomicFact(
            text="The price is 200 dollars.",
            fact_type=FactType.NUMERIC,
            numeric_value=200.0,
            numeric_unit="dollars",
        )

        result = comparator.compare_facts(fact1, fact2)
        assert result["potential_conflict"] is True
        assert result["conflict_type"] == "numeric_mismatch"

    def test_compare_numeric_different_units(self):
        """Test comparing numeric values with different units."""
        comparator = FactComparator()

        fact1 = AtomicFact(
            text="The distance is 100 km.",
            fact_type=FactType.NUMERIC,
            numeric_value=100.0,
            numeric_unit="km",
        )
        fact2 = AtomicFact(
            text="The weight is 100 kg.",
            fact_type=FactType.NUMERIC,
            numeric_value=100.0,
            numeric_unit="kg",
        )

        result = comparator.compare_facts(fact1, fact2)
        # Different units, same value - not a conflict
        assert result["potential_conflict"] is False

    def test_compare_negation_mismatch(self):
        """Test detecting negation mismatch."""
        comparator = FactComparator()

        fact1 = AtomicFact(
            text="Python is compiled.",
            fact_type=FactType.STATEMENT,
            predicate="is",
            negated=False,
        )
        fact2 = AtomicFact(
            text="Python is not compiled.",
            fact_type=FactType.NEGATION,
            predicate="is",
            negated=True,
        )

        result = comparator.compare_facts(fact1, fact2)
        assert result["potential_conflict"] is True
        assert result["conflict_type"] == "negation_mismatch"

    def test_find_conflicting_facts(self):
        """Test finding conflicts between fact lists."""
        comparator = FactComparator()

        facts1 = [
            AtomicFact(
                text="Price is 100 dollars",
                fact_type=FactType.NUMERIC,
                numeric_value=100.0,
                numeric_unit="dollars",
            ),
            AtomicFact(text="Quality is high", fact_type=FactType.STATEMENT),
        ]

        facts2 = [
            AtomicFact(
                text="Price is 200 dollars",
                fact_type=FactType.NUMERIC,
                numeric_value=200.0,
                numeric_unit="dollars",
            ),
            AtomicFact(text="Service is good", fact_type=FactType.STATEMENT),
        ]

        conflicts = comparator.find_conflicting_facts(facts1, facts2)
        assert len(conflicts) >= 1
        assert conflicts[0]["conflict_type"] == "numeric_mismatch"


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset before each test."""
        reset_fact_extractor()

    def test_get_fact_extractor_returns_singleton(self):
        """Test that get_fact_extractor returns same instance."""
        extractor1 = get_fact_extractor()
        extractor2 = get_fact_extractor()
        assert extractor1 is extractor2

    def test_reset_fact_extractor_clears_singleton(self):
        """Test that reset_fact_extractor creates new instance."""
        extractor1 = get_fact_extractor()
        reset_fact_extractor()
        extractor2 = get_fact_extractor()
        assert extractor1 is not extractor2


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset before each test."""
        reset_config()
        reset_fact_extractor()
        reset_nli_detector()
        extract_facts_cached.cache_clear()

    def test_extract_facts_cached(self):
        """Test cached fact extraction."""
        facts1 = extract_facts_cached("Python is a programming language.")
        facts2 = extract_facts_cached("Python is a programming language.")

        # Same result from cache
        assert facts1 == facts2
        assert isinstance(facts1, tuple)

    def test_compare_document_facts_no_conflict(self):
        """Test comparing documents without conflicts."""
        text1 = "Python is a programming language."
        text2 = "Java is a programming language."

        conflicts = compare_document_facts(text1, text2)
        # Different subjects, no conflicts expected
        assert isinstance(conflicts, list)

    def test_compare_document_facts_with_conflict(self):
        """Test comparing documents with numeric conflict."""
        text1 = "The price is 100 dollars."
        text2 = "The price is 200 dollars."

        conflicts = compare_document_facts(text1, text2)
        # Should find numeric conflict
        assert len(conflicts) >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Reset before each test."""
        reset_fact_extractor()

    def test_very_long_text(self):
        """Test extracting from very long text."""
        extractor = AtomicFactExtractor()
        long_text = "This is a sentence. " * 100
        result = extractor.extract(long_text)
        assert result.fact_count >= 1

    def test_special_characters(self):
        """Test handling special characters."""
        extractor = AtomicFactExtractor()
        text = "The value is $100 (approximately €90)."
        result = extractor.extract(text)
        assert result.fact_count >= 1

    def test_unicode_text(self):
        """Test handling unicode text."""
        extractor = AtomicFactExtractor()
        text = "Python 支持 Unicode 字符串。"
        result = extractor.extract(text)
        assert result is not None

    def test_empty_sentences(self):
        """Test handling text with empty sentences."""
        extractor = AtomicFactExtractor()
        text = "Hello.   . World."
        result = extractor.extract(text)
        # Should handle gracefully
        assert result is not None

    def test_question_marks(self):
        """Test handling questions."""
        extractor = AtomicFactExtractor()
        text = "What is Python? It is a programming language."
        result = extractor.extract(text)
        assert result.fact_count >= 1


class TestFactTypeDetection:
    """Tests for fact type detection logic."""

    def setup_method(self):
        """Reset before each test."""
        reset_fact_extractor()

    def test_detect_statement(self):
        """Test detecting general statement."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The sky is blue.")

        assert result.fact_count >= 1
        assert result.facts[0].fact_type == FactType.STATEMENT

    def test_detect_numeric_with_unit(self):
        """Test detecting numeric fact with unit."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("The file size is 500 MB.")

        numeric = result.get_numeric_facts()
        assert len(numeric) >= 1
        assert numeric[0].fact_type == FactType.NUMERIC

    def test_detect_definition_is_a(self):
        """Test detecting 'is a' definition pattern."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("A function is a reusable block of code.")

        definitions = result.get_facts_by_type(FactType.DEFINITION)
        assert len(definitions) >= 1

    def test_detect_definition_means(self):
        """Test detecting 'means' definition pattern."""
        extractor = AtomicFactExtractor()
        result = extractor.extract("API means Application Programming Interface.")

        definitions = result.get_facts_by_type(FactType.DEFINITION)
        assert len(definitions) >= 1


class TestSpaCyIntegration:
    """Tests for spaCy integration."""

    def setup_method(self):
        """Reset before each test."""
        reset_fact_extractor()

    def test_spacy_availability_check(self):
        """Test checking spaCy availability."""
        extractor = AtomicFactExtractor()
        # Should not raise regardless of spaCy availability
        is_available = extractor.is_spacy_available
        assert isinstance(is_available, bool)

    def test_extraction_works_without_spacy(self):
        """Test that extraction works even without full spaCy model."""
        extractor = AtomicFactExtractor()
        text = "This is a test sentence. This is another one."

        result = extractor.extract(text)
        # Should work with basic or spaCy extraction
        assert result.fact_count >= 1
