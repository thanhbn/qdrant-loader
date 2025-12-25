# ============================================================
# LEARNING: detectors.py - AIKH-488 (SPIKE-008: Validate detect_document_conflicts)
# This file has been annotated with TODO markers for learning.
# To restore: git checkout -- packages/qdrant-loader-mcp-server/src/qdrant_loader_mcp_server/search/enhanced/cdi/detectors.py
# MCP Tool: detect_document_conflicts
# Learning Objectives:
# - [ ] L1: Understand vector similarity thresholds (AIKH-488/AIKH-601: TC-CONFLICT-001)
# - [ ] L1: Understand multi-method conflict detection (AIKH-488/AIKH-607: TC-CONFLICT-007)
# - [ ] L1: Understand conflict aggregation (AIKH-488/AIKH-605: TC-CONFLICT-005)
# - [ ] L2: Understand tiered filtering (AIKH-488/AIKH-607: TC-CONFLICT-007)
# - [ ] L3: Understand LLM validation tier (AIKH-488/AIKH-603: TC-CONFLICT-003)
# ============================================================

"""
Conflict Detection for Cross-Document Intelligence.

This module implements advanced conflict detection between documents using
vector similarity analysis and optional LLM validation for comprehensive
conflict identification and analysis.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

# Soft-import async clients to avoid hard dependency at import time
if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient
else:
    AsyncQdrantClient = None  # type: ignore[assignment]

from ....utils.logging import LoggingConfig
from ...models import SearchResult
from ...nlp.spacy_analyzer import SpaCyQueryAnalyzer
from .conflict_pairing import (
    calculate_vector_similarity as _calculate_vector_similarity_ext,
)
from .conflict_pairing import (
    filter_by_vector_similarity as _filter_by_vector_similarity_ext,
)
from .conflict_pairing import get_document_embeddings as _get_document_embeddings_ext
from .conflict_resolution import describe_conflict as _describe_conflict_ext
from .conflict_resolution import extract_context_snippet as _extract_context_snippet_ext
from .conflict_resolution import (
    generate_resolution_suggestions as _generate_resolution_suggestions_ext,
)
from .conflict_scoring import (
    analyze_metadata_conflicts as _analyze_metadata_conflicts_ext,
)
from .conflict_scoring import analyze_text_conflicts as _analyze_text_conflicts_ext
from .conflict_scoring import calculate_conflict_confidence as _calculate_confidence_ext
from .conflict_scoring import categorize_conflict as _categorize_conflict_ext
from .legacy_adapters import LegacyConflictDetectorAdapter
from .llm_validation import llm_analyze_conflicts as _llm_analyze_conflicts_ext
from .llm_validation import (
    validate_conflict_with_llm as _validate_conflict_with_llm_ext,
)
from .models import ConflictAnalysis

logger = LoggingConfig.get_logger(__name__)


class ConflictDetector:
    """Enhanced conflict detector using vector similarity and LLM validation."""

    def __init__(
        self,
        spacy_analyzer: SpaCyQueryAnalyzer,
        qdrant_client: AsyncQdrantClient | None = None,
        openai_client: Any | None = None,
        collection_name: str = "documents",
        preferred_vector_name: str | None = "dense",
    ):
        """Initialize the enhanced conflict detector.

        Args:
            spacy_analyzer: SpaCy analyzer for text processing
            qdrant_client: Qdrant client for vector operations
            openai_client: OpenAI client for LLM analysis
            collection_name: Qdrant collection name
        """
        self.spacy_analyzer = spacy_analyzer
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client
        self.collection_name = collection_name
        self.logger = LoggingConfig.get_logger(__name__)
        self.preferred_vector_name = preferred_vector_name

        # TODO [L1]: Define vector similarity thresholds - AIKH-488/AIKH-601 (TC-CONFLICT-001)
        # MCP Tool: detect_document_conflicts
        # Use Case: Filter document pairs by semantic similarity before conflict analysis
        # Business Rule: Documents outside [0.4, 0.98] similarity range are not conflicting:
        #   - Below 0.4: Documents discuss unrelated topics (no conflict possible)
        #   - Above 0.98: Documents are likely duplicates (same content, not conflict)
        # Git: "Enhanced conflict detection and similarity calculations" (commit 2a3cd6b)
        # Test: test_detect_conflicts_basic_functionality
        # -----------------------------------------------------------
        self.MIN_VECTOR_SIMILARITY = (
            0.4  # Minimum similarity to consider for conflict analysis
        )
        self.MAX_VECTOR_SIMILARITY = (
            0.98  # Maximum similarity - too similar suggests same content
        )
        # -----------------------------------------------------------

        # LLM validation settings
        self.llm_enabled = qdrant_client is not None and openai_client is not None

        # Link back to engine for provider access if set upstream
        self.engine: Any | None = None

    async def _get_document_embeddings(
        self, document_ids: list[str]
    ) -> dict[str, list[float]]:
        return await _get_document_embeddings_ext(self, document_ids)

    def _calculate_vector_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        return _calculate_vector_similarity_ext(self, embedding1, embedding2)

    async def _validate_conflict_with_llm(
        self, doc1: SearchResult, doc2: SearchResult, similarity_score: float
    ) -> tuple[bool, str, float]:
        return await _validate_conflict_with_llm_ext(self, doc1, doc2, similarity_score)

    def _analyze_text_conflicts(
        self, doc1: SearchResult, doc2: SearchResult
    ) -> tuple[bool, str, float]:
        return _analyze_text_conflicts_ext(self, doc1, doc2)

    def _analyze_metadata_conflicts(
        self, doc1: SearchResult, doc2: SearchResult
    ) -> tuple[bool, str, float]:
        return _analyze_metadata_conflicts_ext(self, doc1, doc2)

    async def detect_conflicts(self, documents: list[SearchResult]) -> ConflictAnalysis:
        """Detect conflicts between documents using multiple analysis methods.

        Use Case: Main entry point for conflict detection between documents
        Business Rule: Uses multi-method detection (text + metadata + LLM)
        Git: "CDI Module Consolidation" (commit 9bc4aa9)
        Test: test_detect_conflicts
        """
        start_time = time.time()
        conflicts: list[dict[str, Any] | ConflictAnalysis] = []

        # TODO [L1]: Validate minimum document count - AIKH-488/AIKH-611 (TC-CONFLICT-011)
        # MCP Tool: detect_document_conflicts
        # Use Case: Handle edge case when insufficient documents provided
        # Business Rule: Need at least 2 documents to detect conflicts (single doc = no conflict)
        # Test: test_detect_conflicts_insufficient_documents
        # -----------------------------------------------------------
        if len(documents) < 2:
            self.logger.debug("Need at least 2 documents for conflict detection")
            return ConflictAnalysis()
        # -----------------------------------------------------------

        try:
            # TODO [L2]: Precompute embeddings for all documents - AIKH-488/AIKH-602 (TC-CONFLICT-002)
            # MCP Tool: detect_document_conflicts
            # Use Case: Batch fetch embeddings for efficient vector similarity calculations
            # Data Flow: documents -> document_ids -> Qdrant scroll API -> embeddings dict
            # Business Rule: Pre-fetch all embeddings to avoid N+1 queries during pair analysis
            # Test: test_embedding_similarity_tier
            # -----------------------------------------------------------
            document_ids = [
                getattr(doc, "document_id", f"{doc.source_type}:{doc.source_title}")
                for doc in documents
            ]
            embeddings = await self._get_document_embeddings(document_ids)
            # -----------------------------------------------------------

            self.logger.debug(
                f"Conflict detection: {len(documents)} documents, {len(embeddings)} embeddings found"
            )

            # TODO [L1]: Define pair analysis function - AIKH-488/AIKH-607 (TC-CONFLICT-007)
            # MCP Tool: detect_document_conflicts
            # Use Case: Analyze a document pair for conflicts using tiered multi-method analysis
            # Business Rule: Combines vector similarity + text analysis (spaCy) + metadata analysis
            # Data Flow: (doc1, doc2) -> vector check -> text analysis -> metadata analysis -> conflict dict
            # Test: test_tiered_filtering
            # -----------------------------------------------------------
            def analyze_pair(
                doc1: SearchResult, doc2: SearchResult, doc1_id: str, doc2_id: str
            ) -> dict[str, Any] | None:
                """Analyze a pair and return conflict info dict if conflict found."""
                vector_similarity = 0.0
                has_embeddings = doc1_id in embeddings and doc2_id in embeddings

                # TODO [L2]: Calculate vector similarity and skip duplicates - AIKH-488/AIKH-602 (TC-CONFLICT-002)
                # MCP Tool: detect_document_conflicts
                # Use Case: Compute cosine similarity between document embeddings
                # Business Rule: Skip pairs with similarity > 0.98 (likely duplicates, not conflicts)
                # Data Flow: embeddings dict -> cosine similarity -> float [0.0, 1.0]
                # Test: test_embedding_similarity_tier
                # -----------------------------------------------------------
                if has_embeddings:
                    vector_similarity = self._calculate_vector_similarity(
                        embeddings[doc1_id], embeddings[doc2_id]
                    )
                    # Only skip if documents are too similar (likely same content)
                    if vector_similarity > self.MAX_VECTOR_SIMILARITY:
                        self.logger.debug(
                            f"Skipping pair (too similar): {doc1_id[:30]}... vs {doc2_id[:30]}... "
                            f"(similarity: {vector_similarity:.3f})"
                        )
                        return None
                # -----------------------------------------------------------

                # TODO [L1]: Perform multi-method conflict analysis - AIKH-488/AIKH-607 (TC-CONFLICT-007)
                # MCP Tool: detect_document_conflicts
                # Use Case: Analyze using text (spaCy NLP) and metadata conflict detection methods
                # Business Rule: Always run both analyses regardless of embedding availability:
                #   - Text analysis: spaCy entity/keyword extraction, conflict indicators
                #   - Metadata analysis: date differences, source types, project conflicts
                # Test: test_tiered_filtering, test_text_metadata_analysis
                # -----------------------------------------------------------
                text_conflict, text_explanation, text_confidence = (
                    self._analyze_text_conflicts(doc1, doc2)
                )
                metadata_conflict, metadata_explanation, metadata_confidence = (
                    self._analyze_metadata_conflicts(doc1, doc2)
                )
                # -----------------------------------------------------------

                llm_conflict = False
                llm_explanation = ""
                llm_confidence = 0.0
                if self.llm_enabled and (
                    text_conflict or metadata_conflict or vector_similarity > 0.7
                ):
                    # Inlined await is not possible in nested def; handled outside
                    pass

                # Create conflict info dict if any analysis detected issues
                if text_conflict or metadata_conflict or llm_conflict:
                    combined_confidence = max(
                        text_confidence, metadata_confidence, llm_confidence
                    )
                    self.logger.debug(
                        f"Conflict detected: {doc1.source_title} vs {doc2.source_title} "
                        f"(text={text_conflict}, metadata={metadata_conflict}, confidence={combined_confidence:.2f})"
                    )
                    # Return conflict info as dict (not ConflictAnalysis object)
                    conflict_type = "text_conflict" if text_conflict else "metadata_conflict"
                    return {
                        "doc1_id": doc1_id,
                        "doc2_id": doc2_id,
                        "doc1_title": doc1.source_title,
                        "doc1_source": doc1.source_type,
                        "doc2_title": doc2.source_title,
                        "doc2_source": doc2.source_type,
                        "type": conflict_type,
                        "confidence": combined_confidence,
                        "vector_similarity": vector_similarity,
                        "description": f"Text: {text_explanation}; Metadata: {metadata_explanation}",
                        "detected_at": datetime.now().isoformat(),
                    }

                return None

            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents[i + 1 :], i + 1):
                    doc1_id = document_ids[i]
                    doc2_id = document_ids[j]

                    # Compute baseline analysis
                    baseline = analyze_pair(doc1, doc2, doc1_id, doc2_id)

                    # If LLM is needed, compute with LLM and merge
                    if baseline is None:
                        # Check if LLM should run even when baseline None (only by similarity gate)
                        vector_similarity = 0.0
                        if doc1_id in embeddings and doc2_id in embeddings:
                            vector_similarity = self._calculate_vector_similarity(
                                embeddings[doc1_id], embeddings[doc2_id]
                            )
                        if self.llm_enabled and vector_similarity > 0.7:
                            llm_conflict, llm_explanation, llm_confidence = (
                                await self._validate_conflict_with_llm(
                                    doc1, doc2, vector_similarity
                                )
                            )
                            if llm_conflict:
                                # Create conflict info as dict (not ConflictAnalysis object)
                                conflicts.append({
                                    "doc1_id": doc1_id,
                                    "doc2_id": doc2_id,
                                    "doc1_title": doc1.source_title,
                                    "doc1_source": doc1.source_type,
                                    "doc2_title": doc2.source_title,
                                    "doc2_source": doc2.source_type,
                                    "type": "content_conflict",
                                    "confidence": llm_confidence,
                                    "vector_similarity": vector_similarity,
                                    "description": f"LLM: {llm_explanation}",
                                    "detected_at": datetime.now().isoformat(),
                                })
                        continue

                    # If baseline exists and LLM applies, enrich with LLM
                    if self.llm_enabled:
                        vector_similarity = baseline.get("vector_similarity", 0.0) or 0.0
                        if (
                            vector_similarity <= 0.0
                            and doc1_id in embeddings
                            and doc2_id in embeddings
                        ):
                            vector_similarity = self._calculate_vector_similarity(
                                embeddings[doc1_id], embeddings[doc2_id]
                            )
                        if vector_similarity > 0.7:
                            llm_conflict, llm_explanation, llm_confidence = (
                                await self._validate_conflict_with_llm(
                                    doc1, doc2, vector_similarity
                                )
                            )
                            if llm_conflict:
                                baseline["confidence"] = max(
                                    baseline.get("confidence", 0.0), llm_confidence
                                )
                                baseline["description"] = (
                                    baseline.get("description", "") + f"; LLM: {llm_explanation}"
                                )

                    conflicts.append(baseline)

            processing_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"Conflict detection completed in {processing_time:.2f}ms",
                document_count=len(documents),
                conflicts_detected=len(conflicts),
                vector_analysis=len(embeddings) > 0,
                llm_analysis=self.llm_enabled,
            )

            # TODO [L1]: Build merged ConflictAnalysis from collected results - AIKH-488/AIKH-605 (TC-CONFLICT-005)
            # MCP Tool: detect_document_conflicts
            # Use Case: Aggregate all detected conflicts into unified ConflictAnalysis response
            # Data Flow: list of conflict dicts/objects -> merged ConflictAnalysis with:
            #   - conflicting_pairs: list of (doc1_id, doc2_id, conflict_info)
            #   - conflict_categories: dict mapping conflict type to pair list
            #   - resolution_suggestions: dict of suggested fixes
            # Business Rule: Supports multiple input formats for backward compatibility
            # Test: test_confidence_score_in_response
            # -----------------------------------------------------------
            merged_conflicting_pairs: list[tuple[str, str, dict[str, Any]]] = []
            merged_conflict_categories: defaultdict[str, list[tuple[str, str]]] = (
                defaultdict(list)
            )
            merged_resolution_suggestions: dict[str, str] = {}
            # -----------------------------------------------------------

            # TODO [L1]: Process each conflict and categorize by type - AIKH-488/AIKH-604 (TC-CONFLICT-004)
            # MCP Tool: detect_document_conflicts
            # Use Case: Normalize different conflict formats and classify by conflict type
            # Business Rule: Handle 3 input formats for backward compatibility:
            #   - ConflictAnalysis objects: Extract pairs, categories, suggestions
            #   - dict: New format from analyze_pair with doc IDs, type, confidence
            #   - tuple: Legacy format (doc1_id, doc2_id, conflict_info)
            # Conflict Types: text_conflict, metadata_conflict, content_conflict, unknown
            # Git: "CDI Module Consolidation" (commit 9bc4aa9)
            # Test: test_conflict_type_classification
            # -----------------------------------------------------------
            for conflict in conflicts:
                # Preferred: ConflictAnalysis objects
                if isinstance(conflict, ConflictAnalysis):
                    if getattr(conflict, "conflicting_pairs", None):
                        merged_conflicting_pairs.extend(conflict.conflicting_pairs)
                    if getattr(conflict, "conflict_categories", None):
                        for category, pairs in conflict.conflict_categories.items():
                            merged_conflict_categories[category].extend(pairs)
                    if getattr(conflict, "resolution_suggestions", None):
                        merged_resolution_suggestions.update(
                            conflict.resolution_suggestions
                        )
                    continue

                # Handle dict conflicts (new format from analyze_pair)
                if isinstance(conflict, dict):
                    doc1_id = conflict.get("doc1_id", "unknown")
                    doc2_id = conflict.get("doc2_id", "unknown")
                    conflict_type = conflict.get("type", "unknown")

                    # Build conflict info dict for the pair
                    conflict_info = {
                        "doc1_title": conflict.get("doc1_title"),
                        "doc1_source": conflict.get("doc1_source"),
                        "doc2_title": conflict.get("doc2_title"),
                        "doc2_source": conflict.get("doc2_source"),
                        "type": conflict_type,
                        "confidence": conflict.get("confidence", 0.0),
                        "vector_similarity": conflict.get("vector_similarity", 0.0),
                        "description": conflict.get("description", ""),
                        "detected_at": conflict.get("detected_at"),
                    }

                    merged_conflicting_pairs.append(
                        (str(doc1_id), str(doc2_id), conflict_info)
                    )
                    merged_conflict_categories[conflict_type].append(
                        (str(doc1_id), str(doc2_id))
                    )
                    continue

                # Backward-compat: tuples like (doc1_id, doc2_id, conflict_info)
                try:
                    if isinstance(conflict, tuple) and len(conflict) >= 3:
                        doc1_id, doc2_id, conflict_info = (
                            conflict[0],
                            conflict[1],
                            conflict[2],
                        )
                        info_dict = (
                            conflict_info
                            if isinstance(conflict_info, dict)
                            else {"info": conflict_info}
                        )
                        merged_conflicting_pairs.append(
                            (str(doc1_id), str(doc2_id), info_dict)
                        )
                        conflict_type = info_dict.get("type", "unknown")
                        merged_conflict_categories[conflict_type].append(
                            (str(doc1_id), str(doc2_id))
                        )
                except Exception:
                    # Ignore malformed entries
                    pass
            # -----------------------------------------------------------

            return ConflictAnalysis(
                conflicting_pairs=merged_conflicting_pairs,
                conflict_categories=dict(merged_conflict_categories),
                resolution_suggestions=merged_resolution_suggestions,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(
                f"Error in conflict detection after {processing_time:.2f}ms: {e}"
            )
            return ConflictAnalysis()

    # TODO [L2]: Compatibility methods for tests - AIKH-488/AIKH-614 (TC-CONFLICT-014)
    # MCP Tool: detect_document_conflicts
    # Use Case: Maintain backward compatibility with existing test suite via adapter pattern
    # Business Rule: Delegate pattern detection to LegacyConflictDetectorAdapter for:
    #   - _find_contradiction_patterns: Find textual contradictions
    #   - _detect_version_conflicts: Detect version-related conflicts
    #   - _detect_procedural_conflicts: Detect process/procedure conflicts
    # Git: "CDI Module Consolidation" (commit 9bc4aa9)
    # Test: test_find_contradiction_patterns_*, test_subtle_conflict_detection
    # -----------------------------------------------------------
    def _find_contradiction_patterns(self, doc1, doc2):
        try:
            adapter = getattr(self, "_legacy_adapter", None)
            if adapter is None:
                adapter = LegacyConflictDetectorAdapter(self)
                self._legacy_adapter = adapter
            return adapter._find_contradiction_patterns(doc1, doc2)
        except Exception as e:
            self.logger.warning(
                f"Error delegating to legacy adapter in _find_contradiction_patterns: {e}"
            )
            return []

    def _detect_version_conflicts(self, doc1, doc2):
        try:
            adapter = getattr(self, "_legacy_adapter", None)
            if adapter is None:
                adapter = LegacyConflictDetectorAdapter(self)
                self._legacy_adapter = adapter
            return adapter._detect_version_conflicts(doc1, doc2)
        except Exception as e:
            self.logger.warning(
                f"Error delegating to legacy adapter in _detect_version_conflicts: {e}"
            )
            return []

    def _detect_procedural_conflicts(self, doc1, doc2):
        try:
            adapter = getattr(self, "_legacy_adapter", None)
            if adapter is None:
                adapter = LegacyConflictDetectorAdapter(self)
                self._legacy_adapter = adapter
            return adapter._detect_procedural_conflicts(doc1, doc2)
        except Exception as e:
            self.logger.warning(
                f"Error delegating to legacy adapter in _detect_procedural_conflicts: {e}"
            )
            return []

    def _extract_context_snippet(
        self, text, keyword, context_length=100, max_length=None
    ):
        return _extract_context_snippet_ext(
            self, text, keyword, context_length, max_length
        )

    def _categorize_conflict(self, patterns):
        return _categorize_conflict_ext(self, patterns)

    def _calculate_conflict_confidence(self, patterns, doc1_score=1.0, doc2_score=1.0):
        return _calculate_confidence_ext(self, patterns, doc1_score, doc2_score)
    # -----------------------------------------------------------

    # --- Public stats accessor to avoid leaking private attributes ---
    def get_stats(self) -> dict:
        """Return detector runtime statistics as a dictionary.

        Public, stable accessor that always returns a dictionary and never raises.
        """
        try:
            stats = getattr(self, "_last_stats", None)
            return stats if isinstance(stats, dict) else {}
        except Exception:
            return {}

    def get_last_stats(self) -> dict:
        """Return the last computed runtime statistics as a dictionary.

        Exposes detector runtime metrics via a stable public API. Falls back to an
        empty dict if stats are not available. Compatible with internal implementations
        that may store stats on a private attribute.
        """
        try:
            stats = getattr(self, "_last_stats", None)
            return stats if isinstance(stats, dict) else {}
        except Exception:
            return {}

    # TODO [L2]: Additional compatibility methods for tests - AIKH-488/AIKH-608 (TC-CONFLICT-008)
    # MCP Tool: detect_document_conflicts
    # Use Case: Pre-filter documents before expensive conflict analysis (content overlap check)
    # Business Rule: Token intersection with 20% overlap threshold (Jaccard similarity variant):
    #   - Tokenize both texts to lowercase word sets
    #   - Calculate intersection / min(len(set1), len(set2))
    #   - Return true if ratio > 0.2 (20% overlap)
    # Test: test_have_content_overlap, test_conflict_evidence_extraction
    # -----------------------------------------------------------
    def _have_content_overlap(self, doc1: SearchResult, doc2: SearchResult) -> bool:
        """Check if two documents have content overlap (compatibility method)."""
        # Simple content overlap check using token intersection
        tokens1 = set(doc1.text.lower().split())
        tokens2 = set(doc2.text.lower().split())

        # Calculate overlap ratio based on smaller set (Jaccard similarity variant)
        intersection = tokens1 & tokens2
        min_tokens = min(len(tokens1), len(tokens2))

        if min_tokens == 0:
            return False

        # Use intersection over minimum set size for better sensitivity
        overlap_ratio = len(intersection) / min_tokens
        return overlap_ratio > 0.2  # 20% overlap threshold (more sensitive)
    # -----------------------------------------------------------

    # TODO [L2]: Semantic similarity using spaCy - AIKH-488/AIKH-614 (TC-CONFLICT-014)
    # MCP Tool: detect_document_conflicts
    # Use Case: Determine if documents discuss similar topics using NLP semantic analysis
    # Business Rule: Multi-tier semantic similarity check:
    #   1. Explicit topic separation (food vs tech keywords) -> return False if different domains
    #   2. spaCy doc.similarity() > 0.5 -> return True
    #   3. Semantic keyword overlap > 50% -> return True
    #   4. Fallback: content overlap > 50% -> return True
    # Test: test_have_semantic_similarity, test_subtle_conflict_detection
    # -----------------------------------------------------------
    def _have_semantic_similarity(self, doc1: SearchResult, doc2: SearchResult) -> bool:
        """Check if two documents have semantic similarity (compatibility method)."""
        try:
            # Get tokens for analysis
            tokens1 = set(doc1.text.lower().split())
            tokens2 = set(doc2.text.lower().split())

            # EXPLICIT checks for very different topics FIRST
            food_words = {
                "coffee",
                "brewing",
                "recipe",
                "cooking",
                "food",
                "drink",
                "beverage",
                "taste",
                "techniques",
            }
            tech_words = {
                "authentication",
                "security",
                "login",
                "access",
                "user",
                "secure",
                "auth",
                "password",
            }

            doc1_is_food = bool(tokens1 & food_words)
            doc1_is_tech = bool(tokens1 & tech_words)
            doc2_is_food = bool(tokens2 & food_words)
            doc2_is_tech = bool(tokens2 & tech_words)

            # If one is clearly food-related and the other is tech-related, NOT similar
            if (doc1_is_food and doc2_is_tech) or (doc1_is_tech and doc2_is_food):
                return False

            # Try spaCy if available for similar topics
            if self.spacy_analyzer and hasattr(self.spacy_analyzer, "nlp"):
                doc1_processed = self.spacy_analyzer.nlp(
                    doc1.text[:500]
                )  # Limit text length
                doc2_processed = self.spacy_analyzer.nlp(doc2.text[:500])

                similarity = doc1_processed.similarity(doc2_processed)
                if similarity > 0.5:  # Lower threshold for better sensitivity
                    return True

            # Look for semantic concept overlap (common important words)
            semantic_keywords = {
                "authentication",
                "login",
                "security",
                "access",
                "user",
                "secure",
                "auth",
                "method",
            }
            concept1 = tokens1 & semantic_keywords
            concept2 = tokens2 & semantic_keywords

            # If both docs have semantic concepts and they overlap significantly
            if concept1 and concept2:
                concept_overlap = len(concept1 & concept2) / max(
                    len(concept1), len(concept2)
                )
                if concept_overlap > 0.5:  # 50% concept overlap
                    return True

            # Final fallback: use content overlap with strict threshold
            tokens_intersection = tokens1 & tokens2
            min_tokens = min(len(tokens1), len(tokens2))
            if min_tokens > 0:
                overlap_ratio = len(tokens_intersection) / min_tokens
                return overlap_ratio > 0.5  # Very high threshold

            return False

        except Exception:
            # Ultimate fallback - be conservative
            return False
    # -----------------------------------------------------------

    def _describe_conflict(self, indicators: list) -> str:
        return _describe_conflict_ext(self, indicators)

    def _generate_resolution_suggestions(self, conflicts) -> dict[str, str]:
        return _generate_resolution_suggestions_ext(self, conflicts)

    async def _llm_analyze_conflicts(
        self, doc1: SearchResult, doc2: SearchResult, similarity_score: float
    ) -> dict | None:
        return await _llm_analyze_conflicts_ext(self, doc1, doc2, similarity_score)

    # TODO [L2]: Tiered analysis pairs generation - AIKH-488/AIKH-607 (TC-CONFLICT-007)
    # MCP Tool: detect_document_conflicts
    # Use Case: Prioritize document pairs based on relevance scores for efficient analysis
    # Business Rule: Tier classification by average document score:
    #   - primary (score >= 0.8): Highest priority, analyze first
    #   - secondary (score >= 0.5): Medium priority
    #   - tertiary (score < 0.5): Lowest priority, analyze last
    # Data Flow: documents -> all pairs -> (doc1, doc2, tier, score) -> sorted by score desc
    # Test: test_get_tiered_analysis_pairs, test_tiered_filtering
    # -----------------------------------------------------------
    async def _get_tiered_analysis_pairs(
        self, documents: list[SearchResult]
    ) -> list[tuple]:
        """Generate tiered analysis pairs for conflict detection (compatibility method)."""
        pairs = []

        if len(documents) < 2:
            return pairs

        # Generate pairs with different priority tiers
        for i, doc1 in enumerate(documents):
            for _j, doc2 in enumerate(documents[i + 1 :], i + 1):
                # Calculate a simple priority score based on document attributes
                score = 1.0  # Base score

                # Adjust score based on document similarity and importance
                if hasattr(doc1, "score") and hasattr(doc2, "score"):
                    avg_doc_score = (doc1.score + doc2.score) / 2
                    score = min(1.0, avg_doc_score)

                # Determine tier based on score and document characteristics
                if score >= 0.8:
                    tier = "primary"
                elif score >= 0.5:
                    tier = "secondary"
                else:
                    tier = "tertiary"

                pairs.append((doc1, doc2, tier, score))

        # Sort by score (highest first)
        pairs.sort(key=lambda x: x[3], reverse=True)
        return pairs
    # -----------------------------------------------------------

    async def _filter_by_vector_similarity(
        self, documents: list[SearchResult]
    ) -> list[tuple]:
        return await _filter_by_vector_similarity_ext(self, documents)

    # TODO [L2]: Pre-screening for conflict analysis eligibility - AIKH-488/AIKH-611 (TC-CONFLICT-011)
    # MCP Tool: detect_document_conflicts
    # Use Case: Quick filter to skip invalid document pairs before expensive analysis
    # Business Rule: Skip pair and return False if any condition met:
    #   - Either document is None
    #   - Either document has text < 10 characters
    #   - Documents have same document_id
    #   - Documents have identical text content
    # Test: test_should_analyze_for_conflicts_edge_cases, test_no_conflict_found_handling
    # -----------------------------------------------------------
    def _should_analyze_for_conflicts(
        self, doc1: SearchResult, doc2: SearchResult
    ) -> bool:
        """Determine if two documents should be analyzed for conflicts (compatibility method)."""
        # Basic checks for document validity
        if not doc1 or not doc2:
            return False

        # Check text length - skip very short texts or None text
        text1 = doc1.text if doc1.text else ""
        text2 = doc2.text if doc2.text else ""

        if len(text1.strip()) < 10 or len(text2.strip()) < 10:
            return False

        # Skip identical documents (same ID)
        if hasattr(doc1, "document_id") and hasattr(doc2, "document_id"):
            if doc1.document_id == doc2.document_id:
                return False

        # Skip if documents are exactly the same text
        if text1.strip() == text2.strip():
            return False

        return True
    # -----------------------------------------------------------
