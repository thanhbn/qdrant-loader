"""Tests for CDI topic pre-filter.

Workflow ID: W4-cdi-conflict-cff57ee0
"""

import pytest

from qdrant_loader_mcp_server.search.enhanced.cdi.config import reset_config
from qdrant_loader_mcp_server.search.enhanced.cdi.topic_filter import (
    TopicFilter,
    TopicClassification,
    LRUCache,
    get_topic_filter,
    reset_topic_filter,
)


class TestLRUCache:
    """Test suite for LRU cache implementation."""

    def test_basic_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size=3)
        classification = TopicClassification(
            topics=["tech"], domain="tech", confidence=0.8
        )
        cache.put("key1", classification)
        result = cache.get("key1")
        assert result is not None
        assert result.domain == "tech"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LRUCache(max_size=3)
        result = cache.get("nonexistent")
        assert result is None

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        cache = LRUCache(max_size=2)

        class1 = TopicClassification(topics=["a"], domain="a", confidence=0.5)
        class2 = TopicClassification(topics=["b"], domain="b", confidence=0.5)
        class3 = TopicClassification(topics=["c"], domain="c", confidence=0.5)

        cache.put("key1", class1)
        cache.put("key2", class2)

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add key3, should evict key2 (least recently used)
        cache.put("key3", class3)

        assert cache.get("key1") is not None  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None  # Added

    def test_clear_cache(self):
        """Test cache clear operation."""
        cache = LRUCache(max_size=3)
        cache.put("key1", TopicClassification(topics=["a"], domain="a", confidence=0.5))
        cache.put("key2", TopicClassification(topics=["b"], domain="b", confidence=0.5))

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestTopicClassification:
    """Test suite for TopicClassification dataclass."""

    def test_basic_creation(self):
        """Test basic topic classification creation."""
        tc = TopicClassification(
            topics=["tech", "business"],
            domain="tech",
            confidence=0.85,
            keywords=["api", "server", "endpoint"],
        )

        assert tc.topics == ["tech", "business"]
        assert tc.domain == "tech"
        assert tc.confidence == 0.85
        assert tc.keywords == ["api", "server", "endpoint"]

    def test_default_keywords(self):
        """Test that keywords default to empty list."""
        tc = TopicClassification(topics=["general"], domain="general", confidence=0.5)
        assert tc.keywords == []


class TestTopicFilter:
    """Test suite for TopicFilter class."""

    def setup_method(self):
        """Reset state before each test."""
        reset_config()
        reset_topic_filter()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
        reset_topic_filter()

    def test_filter_enabled_by_default(self):
        """Test that filter is enabled by default from config."""
        tf = TopicFilter()
        assert tf.enabled is True

    def test_filter_can_be_disabled(self):
        """Test that filter can be explicitly disabled."""
        tf = TopicFilter(enabled=False)
        assert tf.enabled is False

    def test_classify_tech_text(self):
        """Test classification of technical text."""
        tf = TopicFilter()
        text = """
        The API endpoint requires OAuth 2.0 authentication.
        You need to set up the server configuration with proper
        SSL certificates and JWT tokens for secure access.
        """

        result = tf.classify_topic(text)

        assert "technology" in result.topics
        assert result.domain == "tech"
        assert result.confidence > 0.5

    def test_classify_food_text(self):
        """Test classification of food-related text."""
        tf = TopicFilter()
        text = """
        To prepare the perfect espresso, you need to grind
        the coffee beans fresh. Use 18 grams of ground coffee
        and brew at 93 degrees Celsius for the best flavor.
        """

        result = tf.classify_topic(text)

        assert "food" in result.topics
        assert result.domain == "food"
        # Confidence is based on keyword count, so just verify it's reasonable
        assert result.confidence > 0.0

    def test_classify_business_text(self):
        """Test classification of business text."""
        tf = TopicFilter()
        text = """
        The quarterly revenue report shows strong growth.
        Our marketing strategy has increased customer acquisition
        by 25%. The budget forecast for next year looks promising.
        """

        result = tf.classify_topic(text)

        assert "business" in result.topics
        assert result.domain == "business"

    def test_classify_general_text(self):
        """Test classification of generic text."""
        tf = TopicFilter()
        text = """
        Hello, this is a general message about nothing
        in particular. The weather is nice today and
        I hope you are having a good day.
        """

        result = tf.classify_topic(text)

        assert "general" in result.topics
        assert result.domain == "general"

    def test_should_compare_same_domain(self):
        """Test that documents in same domain should be compared."""
        tf = TopicFilter()

        text1 = "Configure the API authentication with OAuth tokens"
        text2 = "Set up server security using JWT authentication"

        assert tf.should_compare(text1, text2) is True

    def test_should_not_compare_different_domains(self):
        """Test that documents in different domains should not be compared."""
        tf = TopicFilter()

        text1 = "Configure the API authentication with OAuth tokens and JWT security"
        text2 = "The coffee brewing recipe requires fresh ground beans and hot water"

        assert tf.should_compare(text1, text2) is False

    def test_should_compare_when_disabled(self):
        """Test that filter returns True when disabled."""
        tf = TopicFilter(enabled=False)

        text1 = "Configure the API authentication"
        text2 = "The coffee brewing recipe"

        # Should return True even for different domains when disabled
        assert tf.should_compare(text1, text2) is True

    def test_should_compare_short_texts(self):
        """Test that very short texts are not filtered."""
        tf = TopicFilter()

        text1 = "Short text"
        text2 = "Another short text"

        # Short texts should pass through
        assert tf.should_compare(text1, text2) is True

    def test_calculate_topic_similarity_same_domain(self):
        """Test similarity calculation for same domain."""
        tf = TopicFilter()

        text1 = "The authentication API requires OAuth tokens"
        text2 = "Configure server authentication with JWT tokens"

        similarity, explanation = tf.calculate_topic_similarity(text1, text2)

        assert similarity > 0.3  # Should have reasonable similarity
        assert "tech" in explanation.lower() or "overlap" in explanation.lower()

    def test_calculate_topic_similarity_different_domains(self):
        """Test similarity calculation for different domains."""
        tf = TopicFilter()

        text1 = "The authentication API requires OAuth tokens and secure server config"
        text2 = "The coffee recipe requires fresh beans and proper brewing techniques"

        similarity, explanation = tf.calculate_topic_similarity(text1, text2)

        assert similarity < 0.3  # Should have low similarity
        assert "different domains" in explanation.lower()

    def test_caching_works(self):
        """Test that caching reduces computation."""
        tf = TopicFilter()
        text = "The API endpoint requires OAuth authentication and JWT tokens"

        # First call
        result1 = tf.classify_topic(text)

        # Second call should hit cache
        result2 = tf.classify_topic(text)

        assert result1.domain == result2.domain
        assert result1.topics == result2.topics

        # Verify cache has entry
        assert len(tf._cache.cache) > 0

    def test_filter_document_pairs(self):
        """Test filtering of document pairs."""
        tf = TopicFilter()

        class MockDoc:
            def __init__(self, text):
                self.text = text

        docs = [
            MockDoc("OAuth authentication API server security configuration setup"),
            MockDoc("JWT token authentication secure endpoint access control"),
            MockDoc("Fresh coffee beans espresso brewing recipe preparation"),
        ]

        pairs = tf.filter_document_pairs(docs)

        # Should include (0, 1) - both tech
        # Should NOT include (0, 2) or (1, 2) - tech vs food
        assert (0, 1) in pairs
        assert (0, 2) not in pairs
        assert (1, 2) not in pairs

    def test_filter_document_pairs_disabled(self):
        """Test that all pairs returned when filter disabled."""
        tf = TopicFilter(enabled=False)

        class MockDoc:
            def __init__(self, text):
                self.text = text

        docs = [
            MockDoc("OAuth authentication"),
            MockDoc("JWT tokens"),
            MockDoc("Coffee brewing"),
        ]

        pairs = tf.filter_document_pairs(docs)

        # All pairs should be included when disabled
        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs

    def test_get_filter_stats(self):
        """Test filter statistics reporting."""
        tf = TopicFilter()
        tf.classify_topic("Some test text for classification")

        stats = tf.get_filter_stats()

        assert "enabled" in stats
        assert "similarity_threshold" in stats
        assert "cache_size" in stats
        assert "cache_max_size" in stats
        assert stats["cache_size"] >= 1

    def test_clear_cache(self):
        """Test cache clearing."""
        tf = TopicFilter()
        tf.classify_topic("Some text")
        tf.classify_topic("More text")

        assert len(tf._cache.cache) > 0

        tf.clear_cache()

        assert len(tf._cache.cache) == 0


class TestTopicFilterSingleton:
    """Test suite for topic filter singleton pattern."""

    def setup_method(self):
        """Reset state before each test."""
        reset_config()
        reset_topic_filter()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
        reset_topic_filter()

    def test_get_topic_filter_returns_singleton(self):
        """Test that get_topic_filter returns same instance."""
        tf1 = get_topic_filter()
        tf2 = get_topic_filter()
        assert tf1 is tf2

    def test_reset_topic_filter_clears_singleton(self):
        """Test that reset clears the singleton."""
        tf1 = get_topic_filter()
        reset_topic_filter()
        tf2 = get_topic_filter()
        assert tf1 is not tf2


class TestTopicFilterEdgeCases:
    """Test edge cases for topic filter."""

    def setup_method(self):
        """Reset state before each test."""
        reset_config()
        reset_topic_filter()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
        reset_topic_filter()

    def test_empty_text(self):
        """Test classification of empty text."""
        tf = TopicFilter()
        result = tf.classify_topic("")

        assert result.domain == "general"
        assert result.topics == ["general"]

    def test_whitespace_only_text(self):
        """Test classification of whitespace-only text."""
        tf = TopicFilter()
        result = tf.classify_topic("   \n\t   ")

        assert result.domain == "general"

    def test_special_characters_text(self):
        """Test classification with special characters."""
        tf = TopicFilter()
        result = tf.classify_topic("!@#$%^&*() 123 456 789")

        # Should handle gracefully
        assert result is not None
        assert result.domain == "general"

    def test_mixed_domain_text(self):
        """Test classification of text with mixed domains."""
        tf = TopicFilter()
        text = """
        The restaurant's API uses OAuth authentication to process
        customer orders. The chef's recipe database is secured with
        JWT tokens for employee access.
        """

        result = tf.classify_topic(text)

        # Should detect both domains or pick primary
        assert result is not None
        # Multiple topics possible
        assert len(result.topics) >= 1

    def test_unicode_text(self):
        """Test classification with unicode characters."""
        tf = TopicFilter()
        result = tf.classify_topic("OAuth authentication 认证系统 аутентификация")

        # Should handle unicode gracefully
        assert result is not None
        assert "technology" in result.topics

    def test_custom_text_extractor(self):
        """Test filter_document_pairs with custom text extractor."""
        tf = TopicFilter()

        docs = [
            {"content": "OAuth authentication API server setup with secure endpoint configuration and token validation"},
            {"content": "JWT token security configuration for API authentication with proper SSL certificates"},
            {"content": "Coffee brewing recipe preparation with fresh ground beans and proper water temperature"},
        ]

        def extract_text(doc):
            return doc["content"]

        pairs = tf.filter_document_pairs(docs, text_extractor=extract_text)

        # Should work with custom extractor
        assert (0, 1) in pairs
        assert (0, 2) not in pairs

    def test_string_documents(self):
        """Test filter_document_pairs with string documents."""
        tf = TopicFilter()

        docs = [
            "OAuth authentication API server configuration with secure token handling and endpoint protection",
            "JWT token security endpoint setup with SSL certificates and proper access control mechanisms",
            "Coffee brewing recipe and techniques with fresh ground beans and optimal water temperature",
        ]

        pairs = tf.filter_document_pairs(docs)

        # Should work with string documents
        assert (0, 1) in pairs
        assert (0, 2) not in pairs


class TestNewDomainCategories:
    """Test suite for new domain categories added in Week 1 Day 5."""

    def setup_method(self):
        """Reset state before each test."""
        reset_config()
        reset_topic_filter()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
        reset_topic_filter()

    def test_classify_science_text(self):
        """Test classification of scientific text."""
        tf = TopicFilter()
        text = """
        The research methodology involved controlled experiments with
        multiple variables. Statistical analysis of the data showed
        a strong correlation between the hypothesis and observed results.
        """

        result = tf.classify_topic(text)

        assert "science" in result.topics
        assert result.domain == "science"

    def test_classify_health_text(self):
        """Test classification of health-related text."""
        tf = TopicFilter()
        text = """
        The patient presented with symptoms of chronic illness.
        The physician recommended treatment including medication
        and physical therapy for recovery. Follow-up screening
        was scheduled at the clinic.
        """

        result = tf.classify_topic(text)

        assert "health" in result.topics
        assert result.domain == "health"

    def test_classify_education_text(self):
        """Test classification of education-related text."""
        tf = TopicFilter()
        text = """
        The curriculum includes lectures and assignments for students.
        Assessment is through exams and coursework. The professor
        provides tutorials and the semester ends with graduation
        for those who complete their degree requirements.
        """

        result = tf.classify_topic(text)

        assert "education" in result.topics
        assert result.domain == "education"

    def test_classify_finance_text(self):
        """Test classification of finance-related text."""
        tf = TopicFilter()
        text = """
        The investment portfolio includes stocks and bonds.
        The dividend yield provides good return on investment.
        Consider the interest rate on the mortgage and loan
        options for banking transactions.
        """

        result = tf.classify_topic(text)

        assert "finance" in result.topics
        assert result.domain == "finance"


class TestIncompatibleDomains:
    """Test suite for incompatible domain detection."""

    def setup_method(self):
        """Reset state before each test."""
        reset_config()
        reset_topic_filter()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
        reset_topic_filter()

    def test_should_not_compare_tech_and_health(self):
        """Test that tech and health documents are not compared."""
        tf = TopicFilter()

        tech_text = "OAuth authentication API server configuration with JWT tokens and SSL certificates for secure endpoints"
        health_text = "The patient received diagnosis and treatment at the hospital clinic with medication and therapy"

        assert tf.should_compare(tech_text, health_text) is False

    def test_should_not_compare_food_and_finance(self):
        """Test that food and finance documents are not compared."""
        tf = TopicFilter()

        food_text = "The coffee brewing recipe requires fresh ground beans and proper water temperature for espresso"
        finance_text = "The investment portfolio with stocks and bonds shows good dividend yield and return on investment"

        assert tf.should_compare(food_text, finance_text) is False

    def test_should_not_compare_education_and_food(self):
        """Test that education and food documents are not compared."""
        tf = TopicFilter()

        education_text = "The curriculum for students includes lectures assignments and exams with assessment by the professor"
        food_text = "The chef prepared a recipe with fresh ingredients in the restaurant kitchen for the menu"

        assert tf.should_compare(education_text, food_text) is False

    def test_should_compare_compatible_domains_business_finance(self):
        """Test that business and finance documents CAN be compared."""
        tf = TopicFilter()

        # Both texts share overlapping keywords: revenue, report, growth, analysis, returns
        business_text = "The quarterly revenue report shows growth analysis with improved customer acquisition and sales metrics"
        finance_text = "The quarterly revenue report shows growth analysis with investment portfolio returns and market trading"

        # Business and finance are compatible domains with shared keywords
        assert tf.should_compare(business_text, finance_text) is True

    def test_should_compare_compatible_domains_science_health(self):
        """Test that science and health documents CAN be compared."""
        tf = TopicFilter()

        # Both texts share overlapping keywords: research, study, analysis, treatment, outcomes, correlation
        science_text = "The research study methodology used statistical analysis of treatment outcomes and correlation data"
        health_text = "The research study on patient treatment outcomes showed analysis and correlation between therapy methods"

        # Science and health are compatible domains with shared keywords
        assert tf.should_compare(science_text, health_text) is True

    def test_should_compare_same_domain(self):
        """Test that same domain documents are compared."""
        tf = TopicFilter()

        tech1 = "OAuth authentication with JWT tokens for API endpoint security configuration and SSL certificates"
        tech2 = "Server authentication setup using token validation and secure database connection with encryption"

        assert tf.should_compare(tech1, tech2) is True

    def test_incompatible_domains_class_attribute(self):
        """Test that INCOMPATIBLE_DOMAINS is properly defined."""
        tf = TopicFilter()

        # Check that incompatible domains set exists and has expected pairs
        assert hasattr(tf, 'INCOMPATIBLE_DOMAINS')
        assert frozenset({"tech", "food"}) in tf.INCOMPATIBLE_DOMAINS
        assert frozenset({"tech", "health"}) in tf.INCOMPATIBLE_DOMAINS
        assert frozenset({"food", "finance"}) in tf.INCOMPATIBLE_DOMAINS
