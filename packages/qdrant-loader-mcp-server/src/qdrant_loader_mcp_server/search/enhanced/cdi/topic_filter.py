"""
Topic Pre-Filter for CDI Conflict Detection.

This module implements topic-based pre-filtering to reduce false positives by
only comparing documents that share similar topics. Unrelated documents
(e.g., coffee brewing vs authentication) are filtered out before conflict
analysis.

Workflow ID: W4-cdi-conflict-cff57ee0
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .config import get_config


@dataclass
class TopicClassification:
    """Result of topic classification for a text."""

    topics: list[str]
    domain: str  # Primary domain: "tech", "food", "general", etc.
    confidence: float
    keywords: list[str] = field(default_factory=list)


class LRUCache:
    """Simple LRU cache for topic classifications."""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, TopicClassification] = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> TopicClassification | None:
        """Get item from cache, moving it to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: TopicClassification) -> None:
        """Add item to cache, evicting oldest if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class TopicFilter:
    """Topic-based pre-filter for conflict detection.

    Uses keyword-based topic classification to determine if two documents
    are likely to be about the same topic and thus worth comparing for
    conflicts.

    This is a lightweight, fast filter that runs before expensive
    conflict analysis to reduce false positives from unrelated documents.
    """

    # Domain keyword mappings
    TECH_KEYWORDS = {
        "authentication",
        "security",
        "login",
        "access",
        "user",
        "secure",
        "auth",
        "password",
        "api",
        "endpoint",
        "server",
        "database",
        "query",
        "request",
        "response",
        "http",
        "https",
        "token",
        "jwt",
        "oauth",
        "session",
        "cookie",
        "header",
        "encryption",
        "ssl",
        "tls",
        "certificate",
        "key",
        "hash",
        "algorithm",
        "code",
        "function",
        "class",
        "method",
        "variable",
        "import",
        "module",
        "package",
        "library",
        "framework",
        "deployment",
        "docker",
        "kubernetes",
        "container",
        "cloud",
        "aws",
        "azure",
        "gcp",
        "microservice",
        "config",
        "configuration",
        "environment",
        "variable",
        "timeout",
        "retry",
        "cache",
        "redis",
        "mongodb",
        "postgresql",
        "mysql",
    }

    FOOD_KEYWORDS = {
        "coffee",
        "brewing",
        "recipe",
        "cooking",
        "food",
        "drink",
        "beverage",
        "taste",
        "flavor",
        "ingredient",
        "meal",
        "dish",
        "kitchen",
        "oven",
        "stove",
        "bake",
        "roast",
        "fry",
        "grill",
        "boil",
        "steam",
        "simmer",
        "marinate",
        "season",
        "spice",
        "herb",
        "vegetable",
        "fruit",
        "meat",
        "fish",
        "dairy",
        "egg",
        "flour",
        "sugar",
        "salt",
        "pepper",
        "oil",
        "butter",
        "cream",
        "milk",
        "cheese",
        "wine",
        "beer",
        "restaurant",
        "chef",
        "menu",
    }

    BUSINESS_KEYWORDS = {
        "revenue",
        "profit",
        "budget",
        "cost",
        "expense",
        "invoice",
        "payment",
        "customer",
        "client",
        "vendor",
        "supplier",
        "contract",
        "agreement",
        "negotiation",
        "meeting",
        "presentation",
        "report",
        "analysis",
        "strategy",
        "marketing",
        "sales",
        "growth",
        "quarter",
        "annual",
        "forecast",
        "target",
        "kpi",
        "metric",
        "dashboard",
        "stakeholder",
        "management",
        "executive",
        "team",
        "project",
        "deadline",
        "milestone",
        "schedule",
    }

    LEGAL_KEYWORDS = {
        "contract",
        "agreement",
        "clause",
        "terms",
        "conditions",
        "liability",
        "warranty",
        "indemnity",
        "compliance",
        "regulation",
        "policy",
        "procedure",
        "law",
        "legal",
        "court",
        "judge",
        "attorney",
        "lawyer",
        "plaintiff",
        "defendant",
        "settlement",
        "arbitration",
        "mediation",
        "jurisdiction",
        "statute",
        "copyright",
        "trademark",
        "patent",
        "intellectual",
        "property",
        "license",
        "gdpr",
        "privacy",
        "consent",
    }

    SCIENCE_KEYWORDS = {
        "research",
        "experiment",
        "hypothesis",
        "theory",
        "observation",
        "analysis",
        "data",
        "study",
        "methodology",
        "results",
        "conclusion",
        "scientific",
        "laboratory",
        "sample",
        "specimen",
        "control",
        "variable",
        "statistical",
        "correlation",
        "causation",
        "peer",
        "review",
        "publication",
        "journal",
        "citation",
        "abstract",
        "thesis",
        "dissertation",
        "hypothesis",
        "empirical",
        "quantitative",
        "qualitative",
        "measurement",
        "precision",
        "accuracy",
    }

    HEALTH_KEYWORDS = {
        "medical",
        "health",
        "patient",
        "doctor",
        "physician",
        "nurse",
        "hospital",
        "clinic",
        "treatment",
        "diagnosis",
        "symptom",
        "disease",
        "illness",
        "medication",
        "prescription",
        "dosage",
        "therapy",
        "surgery",
        "procedure",
        "recovery",
        "wellness",
        "nutrition",
        "exercise",
        "diet",
        "vitamin",
        "supplement",
        "chronic",
        "acute",
        "infection",
        "vaccine",
        "immunization",
        "prevention",
        "screening",
        "consultation",
        "referral",
    }

    EDUCATION_KEYWORDS = {
        "learning",
        "teaching",
        "student",
        "teacher",
        "classroom",
        "curriculum",
        "course",
        "lesson",
        "lecture",
        "assignment",
        "homework",
        "exam",
        "test",
        "grade",
        "score",
        "assessment",
        "education",
        "school",
        "university",
        "college",
        "degree",
        "diploma",
        "certification",
        "training",
        "workshop",
        "seminar",
        "tutorial",
        "syllabus",
        "semester",
        "academic",
        "scholarship",
        "enrollment",
        "graduation",
        "faculty",
        "professor",
    }

    FINANCE_KEYWORDS = {
        "investment",
        "stock",
        "bond",
        "portfolio",
        "dividend",
        "equity",
        "asset",
        "liability",
        "balance",
        "statement",
        "income",
        "expense",
        "cash",
        "flow",
        "roi",
        "return",
        "interest",
        "rate",
        "loan",
        "mortgage",
        "credit",
        "debit",
        "account",
        "banking",
        "savings",
        "checking",
        "transaction",
        "transfer",
        "deposit",
        "withdrawal",
        "inflation",
        "recession",
        "market",
        "trading",
        "broker",
    }

    # Domains that should never be compared (incompatible pairs)
    # These pairs are very unlikely to have meaningful conflicts
    INCOMPATIBLE_DOMAINS = {
        frozenset({"tech", "food"}),
        frozenset({"tech", "health"}),
        frozenset({"food", "legal"}),
        frozenset({"food", "finance"}),
        frozenset({"health", "tech"}),
        frozenset({"education", "food"}),
        frozenset({"science", "food"}),
    }

    def __init__(self, enabled: bool | None = None):
        """Initialize the topic filter.

        Args:
            enabled: Override the config setting for whether filter is enabled.
                    If None, uses config.topic_filter_enabled.
        """
        config = get_config()
        self.enabled = enabled if enabled is not None else config.topic_filter_enabled
        self.similarity_threshold = config.topic_similarity_threshold
        self._cache = LRUCache(config.topic_cache_size)

        # Load domain words from config
        self.food_domain_words = set(config.food_domain_words)
        self.tech_domain_words = set(config.tech_domain_words)

    def _hash_text(self, text: str) -> str:
        """Generate a hash key for caching."""
        # Use first 500 chars for hash to avoid long text issues
        return hashlib.md5(text[:500].encode()).hexdigest()

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract keywords from text for topic analysis."""
        # Normalize text
        text_lower = text.lower()

        # Extract words (alphanumeric only)
        words = set(re.findall(r"\b[a-z][a-z0-9_]*\b", text_lower))

        # Filter out very short words and common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "as",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "you",
            "your",
            "we",
            "our",
            "they",
            "their",
            "what",
            "which",
            "who",
            "whom",
        }

        return {w for w in words if len(w) >= 3 and w not in stop_words}

    def classify_topic(self, text: str) -> TopicClassification:
        """Classify the topic/domain of a text.

        Args:
            text: The text to classify.

        Returns:
            TopicClassification with detected topics, domain, and confidence.
        """
        # Check cache first
        cache_key = self._hash_text(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        keywords = self._extract_keywords(text)

        # Count matches for each domain
        tech_matches = keywords & self.TECH_KEYWORDS
        food_matches = keywords & self.FOOD_KEYWORDS
        business_matches = keywords & self.BUSINESS_KEYWORDS
        legal_matches = keywords & self.LEGAL_KEYWORDS
        science_matches = keywords & self.SCIENCE_KEYWORDS
        health_matches = keywords & self.HEALTH_KEYWORDS
        education_matches = keywords & self.EDUCATION_KEYWORDS
        finance_matches = keywords & self.FINANCE_KEYWORDS

        # Also check config-based domain words
        tech_matches |= keywords & self.tech_domain_words
        food_matches |= keywords & self.food_domain_words

        # Determine primary domain
        domain_scores = {
            "tech": len(tech_matches),
            "food": len(food_matches),
            "business": len(business_matches),
            "legal": len(legal_matches),
            "science": len(science_matches),
            "health": len(health_matches),
            "education": len(education_matches),
            "finance": len(finance_matches),
        }

        max_score = max(domain_scores.values())
        if max_score == 0:
            domain = "general"
            confidence = 0.5
        else:
            domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            # Confidence based on how many keywords matched
            confidence = min(1.0, max_score / 5.0)

        # Combine topics from all detected domains
        topics = []
        if tech_matches:
            topics.append("technology")
        if food_matches:
            topics.append("food")
        if business_matches:
            topics.append("business")
        if legal_matches:
            topics.append("legal")
        if science_matches:
            topics.append("science")
        if health_matches:
            topics.append("health")
        if education_matches:
            topics.append("education")
        if finance_matches:
            topics.append("finance")

        if not topics:
            topics = ["general"]

        result = TopicClassification(
            topics=topics,
            domain=domain,
            confidence=confidence,
            keywords=list(keywords)[:20],  # Keep top 20 keywords
        )

        # Cache the result
        self._cache.put(cache_key, result)

        return result

    def calculate_topic_similarity(
        self, text1: str, text2: str
    ) -> tuple[float, str]:
        """Calculate topic similarity between two texts.

        Args:
            text1: First text to compare.
            text2: Second text to compare.

        Returns:
            Tuple of (similarity_score, explanation).
        """
        class1 = self.classify_topic(text1)
        class2 = self.classify_topic(text2)

        # Check domain match
        if class1.domain != class2.domain and class1.domain != "general" and class2.domain != "general":
            # Different specific domains - likely unrelated
            return 0.0, f"Different domains: {class1.domain} vs {class2.domain}"

        # Calculate topic overlap
        topics1 = set(class1.topics)
        topics2 = set(class2.topics)
        topic_overlap = len(topics1 & topics2) / max(len(topics1 | topics2), 1)

        # Calculate keyword overlap
        keywords1 = set(class1.keywords)
        keywords2 = set(class2.keywords)
        keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)

        # Combined similarity (weighted)
        similarity = 0.6 * topic_overlap + 0.4 * keyword_overlap

        # Explanation
        shared_topics = topics1 & topics2
        shared_keywords = keywords1 & keywords2
        explanation = f"Topic overlap: {topic_overlap:.2f}, Keywords: {keyword_overlap:.2f}"
        if shared_topics:
            explanation += f", Shared topics: {', '.join(shared_topics)}"
        if shared_keywords:
            explanation += f", Shared keywords: {', '.join(list(shared_keywords)[:5])}"

        return similarity, explanation

    def should_compare(self, text1: str, text2: str) -> bool:
        """Determine if two documents should be compared for conflicts.

        This is the main entry point for the filter. Returns True if the
        documents are similar enough in topic to warrant conflict analysis.

        Args:
            text1: First document text.
            text2: Second document text.

        Returns:
            True if documents should be compared, False otherwise.
        """
        if not self.enabled:
            return True  # Filter disabled, compare everything

        # Quick check for empty texts
        if not text1 or not text2 or len(text1.strip()) < 50 or len(text2.strip()) < 50:
            return True  # Don't filter very short texts

        # Get topic classifications
        class1 = self.classify_topic(text1)
        class2 = self.classify_topic(text2)

        # Check for incompatible domains - these pairs should never be compared
        # as they are very unlikely to have meaningful conflicts
        domain_pair = frozenset({class1.domain, class2.domain})
        if domain_pair in self.INCOMPATIBLE_DOMAINS:
            return False

        # Also check if domains are different and both are specific (not general)
        # Different specific domains are unlikely to have meaningful conflicts
        if (class1.domain != class2.domain and
            class1.domain != "general" and
            class2.domain != "general"):
            # Allow some domain pairs that might have overlap
            compatible_pairs = {
                frozenset({"business", "finance"}),
                frozenset({"business", "legal"}),
                frozenset({"science", "health"}),
                frozenset({"science", "education"}),
                frozenset({"tech", "science"}),
                frozenset({"education", "science"}),
            }
            if domain_pair not in compatible_pairs:
                return False

        # Calculate similarity
        similarity, _ = self.calculate_topic_similarity(text1, text2)

        return similarity >= self.similarity_threshold

    def filter_document_pairs(
        self, documents: list[Any], text_extractor: Any | None = None
    ) -> list[tuple[int, int]]:
        """Filter document pairs to only those worth comparing.

        Args:
            documents: List of documents to filter.
            text_extractor: Optional callable to extract text from document.
                           Defaults to using .text attribute.

        Returns:
            List of (i, j) tuples indicating pairs to compare.
        """
        if not self.enabled:
            # Return all pairs
            return [(i, j) for i in range(len(documents)) for j in range(i + 1, len(documents))]

        pairs_to_compare = []

        def get_text(doc: Any) -> str:
            if text_extractor:
                return text_extractor(doc)
            if hasattr(doc, "text"):
                return doc.text
            if isinstance(doc, str):
                return doc
            return str(doc)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                text1 = get_text(documents[i])
                text2 = get_text(documents[j])

                if self.should_compare(text1, text2):
                    pairs_to_compare.append((i, j))

        return pairs_to_compare

    def get_filter_stats(self) -> dict[str, Any]:
        """Get statistics about filter operation.

        Returns:
            Dictionary with cache stats and configuration.
        """
        return {
            "enabled": self.enabled,
            "similarity_threshold": self.similarity_threshold,
            "cache_size": len(self._cache.cache),
            "cache_max_size": self._cache.max_size,
        }

    def clear_cache(self) -> None:
        """Clear the topic classification cache."""
        self._cache.clear()


# Singleton instance for global access
_topic_filter_instance: TopicFilter | None = None


def get_topic_filter() -> TopicFilter:
    """Get the global TopicFilter instance."""
    global _topic_filter_instance
    if _topic_filter_instance is None:
        _topic_filter_instance = TopicFilter()
    return _topic_filter_instance


def reset_topic_filter() -> None:
    """Reset the global TopicFilter instance (for testing)."""
    global _topic_filter_instance
    _topic_filter_instance = None
