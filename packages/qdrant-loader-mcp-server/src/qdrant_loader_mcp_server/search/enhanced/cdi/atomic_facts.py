"""
Atomic Facts Extractor for CDI Conflict Detection.

This module extracts atomic facts (simple, verifiable statements) from
complex text, enabling more precise contradiction detection by comparing
individual claims rather than entire documents.

Key Features:
- Sentence-level decomposition
- Claim extraction using spaCy NLP
- Subject-predicate-object triplet extraction
- Numeric value extraction with units
- Negation detection
- LRU caching for performance

Workflow ID: W4-cdi-conflict-cff57ee0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

from .config import get_config

logger = logging.getLogger(__name__)


class FactType(Enum):
    """Types of atomic facts."""

    STATEMENT = "statement"  # General declarative statement
    NUMERIC = "numeric"  # Contains numeric value/measurement
    TEMPORAL = "temporal"  # Contains time/date reference
    COMPARATIVE = "comparative"  # Contains comparison
    NEGATION = "negation"  # Contains negation
    CAUSAL = "causal"  # Contains cause-effect relationship
    CONDITIONAL = "conditional"  # Contains if-then logic
    DEFINITION = "definition"  # Defines a term or concept


@dataclass(frozen=True)
class AtomicFact:
    """A single atomic fact extracted from text.

    Attributes:
        text: The original text of the fact
        fact_type: The type of fact
        subject: The subject of the statement (if extractable)
        predicate: The predicate/verb phrase (if extractable)
        object_: The object of the statement (if extractable)
        negated: Whether the fact is negated
        confidence: Extraction confidence (0-1)
        source_sentence: The original sentence this was extracted from
        numeric_value: Extracted numeric value (if applicable)
        numeric_unit: Unit for numeric value (if applicable)
        metadata: Additional extracted metadata
    """

    text: str
    fact_type: FactType
    subject: str | None = None
    predicate: str | None = None
    object_: str | None = None
    negated: bool = False
    confidence: float = 1.0
    source_sentence: str | None = None
    numeric_value: float | None = None
    numeric_unit: str | None = None
    metadata: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "fact_type": self.fact_type.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_,
            "negated": self.negated,
            "confidence": self.confidence,
            "source_sentence": self.source_sentence,
            "numeric_value": self.numeric_value,
            "numeric_unit": self.numeric_unit,
            "metadata": dict(self.metadata) if self.metadata else {},
        }


@dataclass
class ExtractionResult:
    """Result of fact extraction from text.

    Attributes:
        facts: List of extracted atomic facts
        source_text: The original text
        sentence_count: Number of sentences processed
        extraction_time_ms: Time taken for extraction (optional)
    """

    facts: list[AtomicFact]
    source_text: str
    sentence_count: int = 0
    extraction_time_ms: float | None = None

    @property
    def fact_count(self) -> int:
        """Get the number of extracted facts."""
        return len(self.facts)

    def get_facts_by_type(self, fact_type: FactType) -> list[AtomicFact]:
        """Get facts of a specific type."""
        return [f for f in self.facts if f.fact_type == fact_type]

    def get_numeric_facts(self) -> list[AtomicFact]:
        """Get all facts with numeric values."""
        return [f for f in self.facts if f.numeric_value is not None]

    def get_negated_facts(self) -> list[AtomicFact]:
        """Get all negated facts."""
        return [f for f in self.facts if f.negated]


class AtomicFactExtractor:
    """Extracts atomic facts from text using NLP.

    Uses spaCy for sentence segmentation and dependency parsing
    to decompose complex text into atomic, verifiable statements.

    Example:
        extractor = AtomicFactExtractor()
        result = extractor.extract("Python is a programming language. It was created in 1991.")
        for fact in result.facts:
            print(fact.text)
    """

    # Negation words and phrases
    NEGATION_WORDS = {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "none",
        "nothing",
        "nobody",
        "nowhere",
        "cannot",
        "can't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "doesn't",
        "don't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "haven't",
        "hasn't",
        "hadn't",
    }

    # Comparative indicators
    COMPARATIVE_WORDS = {
        "more",
        "less",
        "better",
        "worse",
        "higher",
        "lower",
        "greater",
        "smaller",
        "faster",
        "slower",
        "larger",
        "bigger",
        "than",
        "compared",
        "versus",
        "vs",
        "unlike",
        "similar",
        "different",
    }

    # Temporal indicators
    TEMPORAL_WORDS = {
        "today",
        "yesterday",
        "tomorrow",
        "now",
        "then",
        "before",
        "after",
        "during",
        "since",
        "until",
        "when",
        "while",
        "always",
        "never",
        "often",
        "sometimes",
        "daily",
        "weekly",
        "monthly",
        "yearly",
        "annually",
    }

    # Causal indicators
    CAUSAL_WORDS = {
        "because",
        "since",
        "therefore",
        "thus",
        "hence",
        "consequently",
        "as a result",
        "due to",
        "owing to",
        "causes",
        "caused",
        "leads to",
        "results in",
        "effect",
        "affects",
    }

    # Conditional indicators
    CONDITIONAL_WORDS = {
        "if",
        "unless",
        "provided",
        "assuming",
        "given",
        "when",
        "whenever",
        "otherwise",
        "else",
        "then",
    }

    # Numeric patterns
    NUMERIC_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*"
        r"(%|percent|percentage|dollars?|euros?|pounds?|"
        r"kg|kilograms?|g|grams?|mg|milligrams?|"
        r"km|kilometers?|m|meters?|cm|centimeters?|mm|millimeters?|"
        r"miles?|feet|ft|inches?|in|"
        r"hours?|hrs?|minutes?|mins?|seconds?|secs?|"
        r"days?|weeks?|months?|years?|"
        r"mb|gb|tb|kb|bytes?|"
        r"mhz|ghz|hz|"
        r"degrees?|celsius|fahrenheit|kelvin)?\b",
        re.IGNORECASE,
    )

    def __init__(self, use_cache: bool = True):
        """Initialize the extractor.

        Args:
            use_cache: Whether to cache extraction results
        """
        self._nlp = None
        self._use_cache = use_cache
        self._cache: dict[str, ExtractionResult] = {}
        self._spacy_available: bool | None = None

    def _load_spacy(self):
        """Lazy load spaCy model."""
        if self._nlp is not None:
            return

        try:
            import spacy

            # Try to load the model
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "en_core_web_sm not found. Using blank model. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                self._nlp = spacy.blank("en")
                # Add sentencizer for basic sentence splitting
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")

            self._spacy_available = True
        except ImportError:
            logger.warning("spaCy not available. Using basic extraction.")
            self._spacy_available = False

    @property
    def is_spacy_available(self) -> bool:
        """Check if spaCy is available."""
        if self._spacy_available is None:
            self._load_spacy()
        return self._spacy_available or False

    def _create_cache_key(self, text: str) -> str:
        """Create a cache key for text."""
        # Normalize whitespace
        return " ".join(text.lower().split())

    def extract(self, text: str) -> ExtractionResult:
        """Extract atomic facts from text.

        Args:
            text: The text to extract facts from

        Returns:
            ExtractionResult containing extracted facts
        """
        if not text or not text.strip():
            return ExtractionResult(facts=[], source_text=text, sentence_count=0)

        # Check cache
        if self._use_cache:
            cache_key = self._create_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Extract facts
        if self.is_spacy_available:
            result = self._extract_with_spacy(text)
        else:
            result = self._extract_basic(text)

        # Cache result
        if self._use_cache:
            cache_key = self._create_cache_key(text)
            self._cache[cache_key] = result

        return result

    def _extract_with_spacy(self, text: str) -> ExtractionResult:
        """Extract facts using spaCy NLP."""
        self._load_spacy()
        doc = self._nlp(text)

        facts = []
        sentences = list(doc.sents)

        for sent in sentences:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Determine fact type
            fact_type = self._determine_fact_type(sent_text)

            # Check for negation
            negated = self._is_negated(sent_text)

            # Extract subject-predicate-object if possible
            subject, predicate, object_ = self._extract_spo(sent)

            # Extract numeric values
            numeric_value, numeric_unit = self._extract_numeric(sent_text)

            # Determine confidence based on extraction quality
            confidence = self._calculate_confidence(
                subject, predicate, object_, sent_text
            )

            fact = AtomicFact(
                text=sent_text,
                fact_type=fact_type,
                subject=subject,
                predicate=predicate,
                object_=object_,
                negated=negated,
                confidence=confidence,
                source_sentence=sent_text,
                numeric_value=numeric_value,
                numeric_unit=numeric_unit,
            )
            facts.append(fact)

        return ExtractionResult(
            facts=facts, source_text=text, sentence_count=len(sentences)
        )

    def _extract_basic(self, text: str) -> ExtractionResult:
        """Extract facts using basic sentence splitting."""
        # Simple sentence splitting by common delimiters
        sentences = re.split(r"[.!?]+\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        facts = []
        for sent_text in sentences:
            fact_type = self._determine_fact_type(sent_text)
            negated = self._is_negated(sent_text)
            numeric_value, numeric_unit = self._extract_numeric(sent_text)

            fact = AtomicFact(
                text=sent_text,
                fact_type=fact_type,
                negated=negated,
                confidence=0.7,  # Lower confidence for basic extraction
                source_sentence=sent_text,
                numeric_value=numeric_value,
                numeric_unit=numeric_unit,
            )
            facts.append(fact)

        return ExtractionResult(
            facts=facts, source_text=text, sentence_count=len(sentences)
        )

    def _determine_fact_type(self, text: str) -> FactType:
        """Determine the type of fact from text."""
        text_lower = text.lower()
        words = set(text_lower.split())

        # Check for conditional
        if words & self.CONDITIONAL_WORDS:
            if "if" in text_lower or "unless" in text_lower:
                return FactType.CONDITIONAL

        # Check for causal
        if words & self.CAUSAL_WORDS:
            return FactType.CAUSAL

        # Check for comparative
        if words & self.COMPARATIVE_WORDS:
            return FactType.COMPARATIVE

        # Check for negation
        if words & self.NEGATION_WORDS:
            return FactType.NEGATION

        # Check for numeric
        if self.NUMERIC_PATTERN.search(text):
            return FactType.NUMERIC

        # Check for temporal
        if words & self.TEMPORAL_WORDS:
            return FactType.TEMPORAL

        # Check for definition patterns
        if self._is_definition(text_lower):
            return FactType.DEFINITION

        return FactType.STATEMENT

    def _is_negated(self, text: str) -> bool:
        """Check if text contains negation."""
        text_lower = text.lower()
        words = set(text_lower.split())
        return bool(words & self.NEGATION_WORDS)

    def _is_definition(self, text_lower: str) -> bool:
        """Check if text is a definition."""
        definition_patterns = [
            r"\bis\s+a\b",
            r"\bare\s+a\b",
            r"\bis\s+the\b",
            r"\bis\s+defined\s+as\b",
            r"\bmeans\b",
            r"\brefers\s+to\b",
            r"\bknown\s+as\b",
            r"\bcalled\b",
        ]
        for pattern in definition_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _extract_spo(self, sent) -> tuple[str | None, str | None, str | None]:
        """Extract subject-predicate-object from spaCy sentence.

        Args:
            sent: spaCy Span (sentence)

        Returns:
            Tuple of (subject, predicate, object) or None for each
        """
        subject = None
        predicate = None
        object_ = None

        # Find root verb
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break

        if root is None:
            return subject, predicate, object_

        predicate = root.text

        # Find subject
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass"):
                # Include compound modifiers
                subject_parts = [t.text for t in token.subtree if t.dep_ in ("compound", "amod")] + [token.text]
                subject = " ".join(subject_parts)
                break

        # Find object
        for token in sent:
            if token.dep_ in ("dobj", "pobj", "attr"):
                # Include modifiers
                obj_parts = [t.text for t in token.subtree if t.dep_ in ("compound", "amod")] + [token.text]
                object_ = " ".join(obj_parts)
                break

        return subject, predicate, object_

    def _extract_numeric(self, text: str) -> tuple[float | None, str | None]:
        """Extract numeric value and unit from text.

        Args:
            text: Text to extract from

        Returns:
            Tuple of (value, unit) or (None, None)
        """
        match = self.NUMERIC_PATTERN.search(text)
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2) if match.group(2) else None
                return value, unit
            except (ValueError, IndexError):
                pass
        return None, None

    def _calculate_confidence(
        self,
        subject: str | None,
        predicate: str | None,
        object_: str | None,
        text: str,
    ) -> float:
        """Calculate extraction confidence score."""
        confidence = 0.5  # Base confidence

        # Increase confidence for complete SPO extraction
        if subject:
            confidence += 0.15
        if predicate:
            confidence += 0.2
        if object_:
            confidence += 0.15

        # Adjust based on sentence length (very short or very long = less confident)
        word_count = len(text.split())
        if 5 <= word_count <= 25:
            confidence += 0.1
        elif word_count < 3:
            confidence -= 0.2
        elif word_count > 50:
            confidence -= 0.1

        return min(1.0, max(0.0, confidence))

    def extract_from_documents(
        self,
        documents: list[Any],
        text_extractor: Callable[[Any], str] | None = None,
    ) -> list[ExtractionResult]:
        """Extract facts from multiple documents.

        Args:
            documents: List of documents
            text_extractor: Function to extract text from document

        Returns:
            List of ExtractionResults
        """
        if text_extractor is None:
            text_extractor = lambda d: str(d) if not isinstance(d, str) else d

        results = []
        for doc in documents:
            text = text_extractor(doc)
            result = self.extract(text)
            results.append(result)

        return results

    def clear_cache(self):
        """Clear the extraction cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self._use_cache,
        }


class FactComparator:
    """Compares atomic facts for potential conflicts.

    Uses the NLI detector to compare individual facts
    and identify potential contradictions.
    """

    def __init__(self):
        """Initialize the comparator."""
        self._nli_detector = None

    def _get_nli_detector(self):
        """Lazy load NLI detector."""
        if self._nli_detector is None:
            from .nli_detector import get_nli_detector

            self._nli_detector = get_nli_detector()
        return self._nli_detector

    def compare_facts(
        self, fact1: AtomicFact, fact2: AtomicFact
    ) -> dict[str, Any]:
        """Compare two facts for potential conflict.

        Args:
            fact1: First fact
            fact2: Second fact

        Returns:
            Dictionary with comparison results
        """
        result = {
            "fact1": fact1.text,
            "fact2": fact2.text,
            "potential_conflict": False,
            "conflict_type": None,
            "confidence": 0.0,
            "details": {},
        }

        # Quick check: if both facts are about different subjects, no conflict
        if fact1.subject and fact2.subject:
            if fact1.subject.lower() != fact2.subject.lower():
                result["details"]["different_subjects"] = True
                return result

        # Check for numeric conflicts
        if fact1.numeric_value is not None and fact2.numeric_value is not None:
            if fact1.numeric_unit == fact2.numeric_unit:
                if fact1.numeric_value != fact2.numeric_value:
                    result["potential_conflict"] = True
                    result["conflict_type"] = "numeric_mismatch"
                    # Higher difference = higher confidence
                    diff_ratio = abs(fact1.numeric_value - fact2.numeric_value) / max(
                        abs(fact1.numeric_value), abs(fact2.numeric_value), 1
                    )
                    result["confidence"] = min(1.0, diff_ratio)
                    result["details"]["value1"] = fact1.numeric_value
                    result["details"]["value2"] = fact2.numeric_value
                    return result

        # Check for negation conflict
        if fact1.negated != fact2.negated:
            # One is negated, one is not - check if they're about the same thing
            if fact1.predicate and fact2.predicate:
                if fact1.predicate.lower() == fact2.predicate.lower():
                    result["potential_conflict"] = True
                    result["conflict_type"] = "negation_mismatch"
                    result["confidence"] = 0.8
                    return result

        # Use NLI for semantic comparison
        config = get_config()
        if config.use_nli_model:
            detector = self._get_nli_detector()
            nli_result = detector.detect_contradiction(fact1.text, fact2.text)

            if nli_result.is_contradiction:
                result["potential_conflict"] = True
                result["conflict_type"] = "semantic_contradiction"
                result["confidence"] = nli_result.contradiction_score
                result["details"]["nli_result"] = {
                    "contradiction": nli_result.contradiction_score,
                    "entailment": nli_result.entailment_score,
                    "neutral": nli_result.neutral_score,
                }

        return result

    def find_conflicting_facts(
        self, facts1: list[AtomicFact], facts2: list[AtomicFact]
    ) -> list[dict[str, Any]]:
        """Find conflicting facts between two lists.

        Args:
            facts1: First list of facts
            facts2: Second list of facts

        Returns:
            List of conflict dictionaries
        """
        conflicts = []

        for f1 in facts1:
            for f2 in facts2:
                comparison = self.compare_facts(f1, f2)
                if comparison["potential_conflict"]:
                    conflicts.append(comparison)

        # Sort by confidence
        conflicts.sort(key=lambda x: x["confidence"], reverse=True)
        return conflicts


# Global singleton instance
_extractor_instance: AtomicFactExtractor | None = None


def get_fact_extractor() -> AtomicFactExtractor:
    """Get the global fact extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = AtomicFactExtractor()
    return _extractor_instance


def reset_fact_extractor():
    """Reset the global fact extractor instance (for testing)."""
    global _extractor_instance
    _extractor_instance = None


# Convenience functions
@lru_cache(maxsize=200)
def extract_facts_cached(text: str) -> tuple[AtomicFact, ...]:
    """Extract facts with LRU caching.

    Args:
        text: Text to extract from

    Returns:
        Tuple of extracted facts (tuple for hashability)
    """
    extractor = get_fact_extractor()
    result = extractor.extract(text)
    return tuple(result.facts)


def compare_document_facts(
    text1: str, text2: str
) -> list[dict[str, Any]]:
    """Compare facts between two documents.

    Args:
        text1: First document text
        text2: Second document text

    Returns:
        List of potential conflicts
    """
    extractor = get_fact_extractor()
    result1 = extractor.extract(text1)
    result2 = extractor.extract(text2)

    comparator = FactComparator()
    return comparator.find_conflicting_facts(result1.facts, result2.facts)
