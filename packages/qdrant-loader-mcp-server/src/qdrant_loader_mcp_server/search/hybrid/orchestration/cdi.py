from __future__ import annotations

import inspect
from typing import Any

from ...components.search_result_models import HybridSearchResult
from ...enhanced.cdi import SimilarityMetric


async def analyze_document_relationships(
    engine: Any, documents: list[HybridSearchResult]
) -> dict[str, Any]:
    result = engine.cross_document_engine.analyze_document_relationships(documents)
    # Handle both async and sync implementations defensively
    if inspect.isawaitable(result):
        return await result  # type: ignore[no-any-return]
    return result  # type: ignore[no-any-return]


async def find_similar_documents(
    engine: Any,
    target_document: HybridSearchResult,
    documents: list[HybridSearchResult],
    similarity_metrics: list[SimilarityMetric] | None = None,
    max_similar: int = 5,
) -> list[dict[str, Any]]:
    # TODO [L2/AIKH-589]: Get similarity calculator from cross document engine
    # Use Case: TC-SIMILAR-001 - Find similar by target_query & comparison_query
    # Data Flow: engine.cross_document_engine -> similarity_calculator
    # -----------------------------------------------------------
    similarity_calculator = engine.cross_document_engine.similarity_calculator
    similar_docs = []
    # -----------------------------------------------------------

    # TODO [L2/AIKH-592]: Iterate documents and exclude target document
    # Use Case: TC-SIMILAR-004 - Exclude source document from results
    # Business Rule: Target document should NOT appear in similar documents list
    # Test: test_find_similar_documents
    # -----------------------------------------------------------
    for doc in documents:
        # Prefer ID-based comparison to avoid relying on object equality
        doc_id = getattr(doc, "document_id", getattr(doc, "id", None))
        target_id = getattr(
            target_document, "document_id", getattr(target_document, "id", None)
        )
        if doc_id is not None and target_id is not None:
            if doc_id == target_id:
                continue
        else:
            # Fallback defensively to identity check if IDs are unavailable
            if doc is target_document:
                continue
    # -----------------------------------------------------------

        # TODO [L2/AIKH-590]: Calculate similarity using DocumentSimilarityCalculator
        # Use Case: TC-SIMILAR-002 - Similarity score in response
        # Data Flow: target_document, doc, similarity_metrics -> DocumentSimilarity
        # Delegates to: L1 DocumentSimilarityCalculator.calculate_similarity()
        # -----------------------------------------------------------
        similarity = similarity_calculator.calculate_similarity(
            target_document, doc, similarity_metrics
        )
        # -----------------------------------------------------------

        # TODO [L2/AIKH-589]: Build result dict with document and similarity info
        # Use Case: Structured result for MCP response
        # Data Flow: doc, similarity -> result dict
        # -----------------------------------------------------------
        similar_docs.append(
            {
                "document_id": doc.document_id,
                "document": doc,
                "similarity_score": similarity.similarity_score,
                "metric_scores": similarity.metric_scores,
                "similarity_reasons": [similarity.get_display_explanation()],
            }
        )
        # -----------------------------------------------------------

    # TODO [L2/AIKH-591]: Sort by similarity score and limit results
    # Use Case: TC-SIMILAR-003 - Limit parameter respected
    # Business Rule: Sort descending by similarity_score, return top max_similar
    # Test: test_find_similar_documents
    # -----------------------------------------------------------
    similar_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar_docs[:max_similar]
    # -----------------------------------------------------------


async def detect_document_conflicts(
    engine: Any, documents: list[HybridSearchResult]
) -> dict[str, Any]:
    conflict_analysis = (
        await engine.cross_document_engine.conflict_detector.detect_conflicts(documents)
    )
    return {
        "conflicting_pairs": conflict_analysis.conflicting_pairs,
        "conflict_categories": conflict_analysis.conflict_categories,
        "resolution_suggestions": conflict_analysis.resolution_suggestions,
    }


async def find_complementary_content(
    engine: Any,
    target_document: HybridSearchResult,
    documents: list[HybridSearchResult],
    max_recommendations: int = 5,
) -> list[dict[str, Any]]:
    complementary_content = (
        engine.cross_document_engine.complementary_finder.find_complementary_content(
            target_document, documents
        )
    )
    recommendations = complementary_content.get_top_recommendations(max_recommendations)

    # Build robust document lookup with multiple key strategies
    doc_lookup = engine._build_document_lookup(documents, robust=True)

    enhanced_recommendations = []
    for rec in recommendations:
        doc_id = rec["document_id"]
        document = doc_lookup.get(doc_id)
        if document:
            enhanced_rec = {
                "document_id": rec["document_id"],
                "document": document,
                "title": document.get_display_title(),
                "source_type": document.source_type,
                "relevance_score": rec["relevance_score"],
                "recommendation_reason": rec["recommendation_reason"],
                "strategy": rec["strategy"],
            }
            enhanced_recommendations.append(enhanced_rec)
        else:
            engine.logger.warning(f"Document not found in lookup for ID: {doc_id}")
    return enhanced_recommendations
