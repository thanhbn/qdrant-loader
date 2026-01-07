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
from .config import get_config
from .conflict_scoring import (
    analyze_metadata_conflicts as _analyze_metadata_conflicts_ext,
)
from .conflict_scoring import analyze_text_conflicts as _analyze_text_conflicts_ext
from .conflict_scoring import calculate_conflict_confidence as _calculate_confidence_ext
from .conflict_scoring import categorize_conflict as _categorize_conflict_ext
from .conflict_scoring import TextConflictResult
from .legacy_adapters import LegacyConflictDetectorAdapter
from .llm_validation import llm_analyze_conflicts as _llm_analyze_conflicts_ext
from .llm_validation import (
    validate_conflict_with_llm as _validate_conflict_with_llm_ext,
)
from .models import ConflictAnalysis
from .topic_filter import get_topic_filter

# V2 CDI modules for semantic analysis
from .nli_detector import get_nli_detector, NLILabel
from .atomic_facts import get_fact_extractor, FactComparator
from .aggregator import (
    EvidenceAggregator,
    EvidenceItem,
    EvidenceSource,
    EvidenceWeights,
    ConflictSeverity,
)

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

        # Load vector similarity thresholds from centralized config
        config = get_config()
        self.MIN_VECTOR_SIMILARITY = config.min_vector_similarity
        self.MAX_VECTOR_SIMILARITY = config.max_vector_similarity

        # LLM validation settings
        self.llm_enabled = qdrant_client is not None and openai_client is not None

        # Link back to engine for provider access if set upstream
        self.engine: Any | None = None

        # V2 CDI components - initialized lazily when needed
        self._nli_detector = None
        self._fact_extractor = None
        self._fact_comparator = None
        self._evidence_aggregator = None
        self._v2_initialized = False

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
    ) -> TextConflictResult:
        """Analyze text conflicts and return structured result with indicators."""
        return _analyze_text_conflicts_ext(self, doc1, doc2)

    def _analyze_metadata_conflicts(
        self, doc1: SearchResult, doc2: SearchResult
    ) -> tuple[bool, str, float, list[dict]]:
        """Analyze metadata conflicts and return result with indicators."""
        return _analyze_metadata_conflicts_ext(self, doc1, doc2)

    def _init_v2_components(self):
        """Initialize v2 CDI components lazily."""
        if self._v2_initialized:
            return

        config = get_config()
        if not config.enable_v2_detection:
            return

        try:
            if config.use_nli_model:
                self._nli_detector = get_nli_detector()
                self.logger.debug("V2: NLI detector initialized")

            if config.use_atomic_facts:
                self._fact_extractor = get_fact_extractor()
                self._fact_comparator = FactComparator()
                self.logger.debug("V2: Fact extractor and comparator initialized")

            # Initialize evidence aggregator
            self._evidence_aggregator = EvidenceAggregator()
            self.logger.debug("V2: Evidence aggregator initialized")

            self._v2_initialized = True
            self.logger.info("V2 CDI components initialized successfully")

        except Exception as e:
            self.logger.warning(f"Failed to initialize V2 components: {e}")
            # Fallback to v1 behavior
            config.enable_v2_detection = False

    def _detect_v2_conflict(
        self, doc1: SearchResult, doc2: SearchResult
    ) -> tuple[bool, float, str, str]:
        """Detect conflict using v2 NLI-based semantic analysis.

        Returns:
            Tuple of (has_conflict, confidence, severity, explanation)
        """
        config = get_config()
        text1 = doc1.text if doc1.text else ""
        text2 = doc2.text if doc2.text else ""

        evidence_items: list[EvidenceItem] = []

        # 1. NLI-based contradiction detection
        if self._nli_detector and config.use_nli_model:
            try:
                nli_result = self._nli_detector.detect_contradiction(text1, text2)
                if nli_result.label == NLILabel.CONTRADICTION:
                    evidence_items.append(
                        EvidenceItem(
                            source=EvidenceSource.NLI,
                            score=nli_result.contradiction_score,
                            details={
                                "label": nli_result.label.value,
                                "contradiction_score": nli_result.contradiction_score,
                                "entailment_score": nli_result.entailment_score,
                            },
                        )
                    )
                    self.logger.debug(
                        f"V2 NLI: Contradiction detected (score={nli_result.contradiction_score:.2f})"
                    )
            except Exception as e:
                self.logger.warning(f"V2 NLI detection failed: {e}")

        # 2. Atomic fact extraction and comparison
        if self._fact_extractor and self._fact_comparator and config.use_atomic_facts:
            try:
                facts1 = self._fact_extractor.extract(text1)
                facts2 = self._fact_extractor.extract(text2)

                if facts1.facts and facts2.facts:
                    comparison = self._fact_comparator.compare(facts1.facts, facts2.facts)
                    if comparison.conflicts:
                        # Average conflict scores
                        avg_score = sum(c.score for c in comparison.conflicts) / len(
                            comparison.conflicts
                        )
                        evidence_items.append(
                            EvidenceItem(
                                source=EvidenceSource.ATOMIC_FACTS,
                                score=avg_score,
                                details={
                                    "conflict_count": len(comparison.conflicts),
                                    "conflicts": [
                                        {
                                            "fact1": c.fact1.text,
                                            "fact2": c.fact2.text,
                                            "conflict_type": c.conflict_type,
                                            "score": c.score,
                                        }
                                        for c in comparison.conflicts[:5]  # Limit details
                                    ],
                                },
                            )
                        )
                        self.logger.debug(
                            f"V2 Facts: {len(comparison.conflicts)} fact conflicts found"
                        )
            except Exception as e:
                self.logger.warning(f"V2 Fact extraction failed: {e}")

        # 3. Legacy keyword-based detection (always included for comparison)
        text_result = self._analyze_text_conflicts(doc1, doc2)
        if text_result.has_conflict:
            evidence_items.append(
                EvidenceItem(
                    source=EvidenceSource.KEYWORD,
                    score=text_result.confidence,
                    details={
                        "description": text_result.description,
                        "indicators": text_result.get_structured_indicators()[:5],
                    },
                )
            )

        # 4. Metadata conflicts
        metadata_conflict, metadata_desc, metadata_conf, _ = self._analyze_metadata_conflicts(
            doc1, doc2
        )
        if metadata_conflict:
            evidence_items.append(
                EvidenceItem(
                    source=EvidenceSource.METADATA,
                    score=metadata_conf,
                    details={"description": metadata_desc},
                )
            )

        # Aggregate all evidence
        if not evidence_items:
            return False, 0.0, "INFO", "No conflict detected"

        if self._evidence_aggregator:
            result = self._evidence_aggregator.aggregate(evidence_items)
            return (
                result.has_conflict,
                result.confidence,
                result.severity.value,
                result.explanation,
            )

        # Fallback: simple max confidence
        max_score = max(e.score for e in evidence_items)
        has_conflict = max_score >= config.nli_contradiction_threshold
        return has_conflict, max_score, "MEDIUM", "Conflict detected via evidence aggregation"

    async def detect_conflicts(self, documents: list[SearchResult]) -> ConflictAnalysis:
        """Detect conflicts between documents using multiple analysis methods."""
        start_time = time.time()
        conflicts: list[ConflictAnalysis] = []

        if len(documents) < 2:
            self.logger.debug("Need at least 2 documents for conflict detection")
            return ConflictAnalysis()

        # Initialize v2 components if enabled
        config = get_config()
        if config.enable_v2_detection:
            self._init_v2_components()
            self.logger.info("V2 conflict detection enabled")

        try:
            # Precompute embeddings once
            document_ids = [
                getattr(doc, "document_id", f"{doc.source_type}:{doc.source_title}")
                for doc in documents
            ]
            embeddings = await self._get_document_embeddings(document_ids)

            # Topic pre-filtering: only compare documents with similar topics
            # This reduces false positives from unrelated documents (e.g., coffee vs auth)
            config = get_config()
            pairs_to_analyze: set[tuple[int, int]] | None = None
            if config.use_topic_filter:
                topic_filter = get_topic_filter()
                # Define text extractor for SearchResult documents
                def extract_text(doc: SearchResult) -> str:
                    return doc.text if doc.text else ""

                filtered_pairs = topic_filter.filter_document_pairs(
                    documents, text_extractor=extract_text
                )
                pairs_to_analyze = set(filtered_pairs)
                self.logger.debug(
                    f"Topic filter reduced pairs from {len(documents) * (len(documents) - 1) // 2} to {len(pairs_to_analyze)}"
                )

            def analyze_pair(
                doc1: SearchResult, doc2: SearchResult, doc1_id: str, doc2_id: str
            ) -> ConflictAnalysis | None:
                vector_similarity = 0.0
                if doc1_id in embeddings and doc2_id in embeddings:
                    vector_similarity = self._calculate_vector_similarity(
                        embeddings[doc1_id], embeddings[doc2_id]
                    )
                    if (
                        vector_similarity > self.MAX_VECTOR_SIMILARITY
                        or vector_similarity < self.MIN_VECTOR_SIMILARITY
                    ):
                        return None

                config = get_config()

                # V2 Detection: Use NLI + atomic facts + evidence aggregation
                if config.enable_v2_detection and self._v2_initialized:
                    has_conflict, confidence, severity, explanation = self._detect_v2_conflict(
                        doc1, doc2
                    )
                    if not has_conflict:
                        return None

                    return ConflictAnalysis(
                        document1_title=doc1.source_title,
                        document1_source=doc1.source_type,
                        document2_title=doc2.source_title,
                        document2_source=doc2.source_type,
                        conflict_type=f"v2_{severity.lower()}_conflict",
                        confidence_score=confidence,
                        vector_similarity=vector_similarity,
                        analysis_method="v2_semantic_analysis",
                        explanation=explanation,
                        detected_at=datetime.now(),
                        structured_indicators=[
                            {
                                "type": "v2_detection",
                                "severity": severity,
                                "confidence": confidence,
                            }
                        ],
                    )

                # V1 Detection: Legacy keyword-based analysis
                # Get enhanced text conflict result with indicators
                text_result = self._analyze_text_conflicts(doc1, doc2)
                text_conflict = text_result.has_conflict
                text_explanation = text_result.description
                text_confidence = text_result.confidence
                text_indicators = text_result.get_structured_indicators()

                # Get metadata conflict result with indicators
                metadata_conflict, metadata_explanation, metadata_confidence, metadata_indicators = (
                    self._analyze_metadata_conflicts(doc1, doc2)
                )

                llm_conflict = False
                llm_explanation = ""
                llm_confidence = 0.0
                if self.llm_enabled and (
                    text_conflict or metadata_conflict or vector_similarity > config.llm_validation_threshold
                ):
                    # Inlined await is not possible in nested def; handled outside
                    pass

                if not (text_conflict or metadata_conflict or llm_conflict):
                    return None

                combined_confidence = max(
                    text_confidence, metadata_confidence, llm_confidence
                )

                # Combine all indicators
                all_indicators = text_indicators + [
                    ind.to_dict() if hasattr(ind, "to_dict") else ind
                    for ind in metadata_indicators
                ]

                return ConflictAnalysis(
                    document1_title=doc1.source_title,
                    document1_source=doc1.source_type,
                    document2_title=doc2.source_title,
                    document2_source=doc2.source_type,
                    conflict_type="text_conflict" if text_conflict else "metadata_conflict",
                    confidence_score=combined_confidence,
                    vector_similarity=vector_similarity,
                    analysis_method="v1_keyword_analysis",
                    explanation=f"Text: {text_explanation}; Metadata: {metadata_explanation}; LLM: {llm_explanation}",
                    detected_at=datetime.now(),
                    structured_indicators=all_indicators,
                )

            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents[i + 1 :], i + 1):
                    # Skip pairs filtered out by topic filter
                    if pairs_to_analyze is not None and (i, j) not in pairs_to_analyze:
                        continue

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
                        config = get_config()
                        if self.llm_enabled and vector_similarity > config.llm_validation_threshold:
                            llm_conflict, llm_explanation, llm_confidence = (
                                await self._validate_conflict_with_llm(
                                    doc1, doc2, vector_similarity
                                )
                            )
                            if llm_conflict:
                                conflicts.append(
                                    ConflictAnalysis(
                                        document1_title=doc1.source_title,
                                        document1_source=doc1.source_type,
                                        document2_title=doc2.source_title,
                                        document2_source=doc2.source_type,
                                        conflict_type="content_conflict",
                                        confidence_score=llm_confidence,
                                        vector_similarity=vector_similarity,
                                        analysis_method="multi_method",
                                        explanation=f"LLM: {llm_explanation}",
                                        detected_at=datetime.now(),
                                    )
                                )
                        continue

                    # If baseline exists and LLM applies, enrich with LLM
                    if self.llm_enabled:
                        vector_similarity = baseline.vector_similarity or 0.0
                        if (
                            vector_similarity <= 0.0
                            and doc1_id in embeddings
                            and doc2_id in embeddings
                        ):
                            vector_similarity = self._calculate_vector_similarity(
                                embeddings[doc1_id], embeddings[doc2_id]
                            )
                        config = get_config()
                        if vector_similarity > config.llm_validation_threshold:
                            llm_conflict, llm_explanation, llm_confidence = (
                                await self._validate_conflict_with_llm(
                                    doc1, doc2, vector_similarity
                                )
                            )
                            if llm_conflict:
                                baseline.confidence_score = max(
                                    baseline.confidence_score, llm_confidence
                                )
                                baseline.explanation = (
                                    baseline.explanation + f"; LLM: {llm_explanation}"
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

            # Build merged ConflictAnalysis from collected results
            merged_conflicting_pairs: list[tuple[str, str, dict[str, Any]]] = []
            merged_conflict_categories: defaultdict[str, list[tuple[str, str]]] = (
                defaultdict(list)
            )
            merged_resolution_suggestions: dict[str, str] = {}

            for conflict in conflicts:
                # Preferred: ConflictAnalysis objects
                if isinstance(conflict, ConflictAnalysis):
                    # Check if this is an individual conflict (has document titles)
                    if conflict.document1_title or conflict.document2_title:
                        # Build conflict_info dict from individual fields
                        doc1_id = f"{conflict.document1_source}:{conflict.document1_title}"
                        doc2_id = f"{conflict.document2_source}:{conflict.document2_title}"
                        conflict_info = {
                            "type": conflict.conflict_type,
                            "confidence": conflict.confidence_score,
                            "description": conflict.explanation,
                            "vector_similarity": conflict.vector_similarity,
                            "analysis_method": conflict.analysis_method,
                            "structured_indicators": conflict.structured_indicators,
                        }
                        merged_conflicting_pairs.append(
                            (doc1_id, doc2_id, conflict_info)
                        )
                        merged_conflict_categories[conflict.conflict_type].append(
                            (doc1_id, doc2_id)
                        )
                    # Also handle aggregate conflicting_pairs if present
                    elif getattr(conflict, "conflicting_pairs", None):
                        merged_conflicting_pairs.extend(conflict.conflicting_pairs)
                    if getattr(conflict, "conflict_categories", None):
                        for category, pairs in conflict.conflict_categories.items():
                            merged_conflict_categories[category].extend(pairs)
                    if getattr(conflict, "resolution_suggestions", None):
                        merged_resolution_suggestions.update(
                            conflict.resolution_suggestions
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

    # Compatibility methods for tests - delegate via legacy adapter
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

    # Additional compatibility methods for tests
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
        config = get_config()
        overlap_ratio = len(intersection) / min_tokens
        return overlap_ratio > config.content_overlap_threshold

    def _have_semantic_similarity(self, doc1: SearchResult, doc2: SearchResult) -> bool:
        """Check if two documents have semantic similarity (compatibility method)."""
        try:
            config = get_config()

            # Get tokens for analysis
            tokens1 = set(doc1.text.lower().split())
            tokens2 = set(doc2.text.lower().split())

            # EXPLICIT checks for very different topics FIRST
            # Use config for domain word lists
            food_words = set(config.food_domain_words)
            tech_words = set(config.tech_domain_words)

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
                if similarity > config.semantic_similarity_threshold:
                    return True

            # Look for semantic concept overlap (common important words)
            semantic_keywords = set(config.tech_domain_words)  # Use tech words as semantic concepts
            concept1 = tokens1 & semantic_keywords
            concept2 = tokens2 & semantic_keywords

            # If both docs have semantic concepts and they overlap significantly
            if concept1 and concept2:
                concept_overlap = len(concept1 & concept2) / max(
                    len(concept1), len(concept2)
                )
                if concept_overlap > config.concept_overlap_threshold:
                    return True

            # Final fallback: use content overlap with strict threshold
            tokens_intersection = tokens1 & tokens2
            min_tokens = min(len(tokens1), len(tokens2))
            if min_tokens > 0:
                overlap_ratio = len(tokens_intersection) / min_tokens
                return overlap_ratio > config.concept_overlap_threshold  # Use same threshold

            return False

        except Exception:
            # Ultimate fallback - be conservative
            return False

    def _describe_conflict(self, indicators: list) -> str:
        return _describe_conflict_ext(self, indicators)

    def _generate_resolution_suggestions(self, conflicts) -> dict[str, str]:
        return _generate_resolution_suggestions_ext(self, conflicts)

    async def _llm_analyze_conflicts(
        self, doc1: SearchResult, doc2: SearchResult, similarity_score: float
    ) -> dict | None:
        return await _llm_analyze_conflicts_ext(self, doc1, doc2, similarity_score)

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

    async def _filter_by_vector_similarity(
        self, documents: list[SearchResult]
    ) -> list[tuple]:
        return await _filter_by_vector_similarity_ext(self, documents)

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
