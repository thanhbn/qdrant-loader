# ============================================================
# LEARNING: conflict_pairing.py
# This file has been annotated with TODO markers for learning.
# To restore: git checkout -- packages/qdrant-loader-mcp-server/src/qdrant_loader_mcp_server/search/enhanced/cdi/conflict_pairing.py
# Learning Objectives:
# - [ ] Understand async embedding retrieval from Qdrant
# - [ ] Understand cosine similarity calculation for vectors
# - [ ] Understand document pair filtering by similarity
# - [ ] Understand tiered analysis pair generation
# ============================================================

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np


async def get_document_embeddings(
    detector: Any, document_ids: list[str]
) -> dict[str, list[float]]:
    """Retrieve document embeddings from Qdrant using detector settings.

    This function mirrors ConflictDetector._get_document_embeddings and is extracted
    to reduce module size. It expects a detector with attributes:
    - qdrant_client, collection_name, preferred_vector_name, logger, _settings (optional)

    Use Case: Fetch pre-computed embeddings for conflict analysis
    Data Flow: document_ids -> Qdrant scroll API -> embeddings dict
    Git: "Async Embedding Retrieval" - Concurrent fetch with semaphore
    """
    # TODO [L2]: Check Qdrant client availability
    # Business Rule: Return empty dict if no client configured
    # -----------------------------------------------------------
    if not getattr(detector, "qdrant_client", None):
        return {}
    # -----------------------------------------------------------

    qdrant_client = detector.qdrant_client

    # Support mocked client in tests
    if hasattr(qdrant_client, "retrieve") and hasattr(
        qdrant_client.retrieve, "_mock_name"
    ):
        embeddings: dict[str, list[float]] = {}
        for doc_id in document_ids:
            try:
                points = await qdrant_client.retrieve(
                    collection_name=detector.collection_name,
                    ids=[doc_id],
                    with_vectors=True,
                )
                if points:
                    point = points[0]
                    if hasattr(point, "vector") and point.vector:
                        embeddings[doc_id] = point.vector
            except Exception as e:  # pragma: no cover (best-effort logging)
                detector.logger.warning(
                    f"Failed to retrieve embedding for {doc_id}: {e}"
                )
        return embeddings

    try:
        embeddings: dict[str, list[float]] = {}

        # TODO [L2]: Configure timeout and concurrency from settings
        # Use Case: Allow per-call customization of embedding retrieval
        # Business Rule: Default 5s timeout, max 5 concurrent requests
        # Git: "Improved conflict detection with temporary settings context manager" (commit 8aade99)
        # -----------------------------------------------------------
        settings = (
            getattr(detector, "_settings", {}) if hasattr(detector, "_settings") else {}
        )
        timeout_s = settings.get("conflict_embeddings_timeout_s", 5.0)
        max_cc = settings.get("conflict_embeddings_max_concurrency", 5)

        semaphore = asyncio.Semaphore(max_cc)
        # -----------------------------------------------------------

        async def fetch_embedding(doc_id: str) -> None:
            # TODO [L2]: Implement semaphore-controlled embedding fetch
            # Use Case: Rate-limit concurrent Qdrant requests
            # Data Flow: doc_id -> Qdrant scroll -> filter by document_id -> extract vector
            # -----------------------------------------------------------
            async with semaphore:
                try:
                    search_result = await asyncio.wait_for(
                        qdrant_client.scroll(
                            collection_name=detector.collection_name,
                            scroll_filter={
                                "must": [
                                    {
                                        "key": "document_id",
                                        "match": {"value": doc_id},
                                    }
                                ]
                            },
                            limit=1,
                            with_vectors=True,
                        ),
                        timeout=timeout_s,
                    )

                    if search_result and search_result[0]:
                        point = search_result[0][0]
                        if point.vector:
                            if isinstance(point.vector, dict):
                                vector_data = (
                                    point.vector.get(detector.preferred_vector_name)
                                    or point.vector.get("dense")
                                    or next(iter(point.vector.values()), None)
                                )
                            else:
                                vector_data = point.vector

                            if vector_data:
                                embeddings[doc_id] = vector_data
                            else:
                                detector.logger.warning(
                                    f"No vector data found for document {doc_id}"
                                )
                        else:
                            detector.logger.warning(
                                f"No vectors found for document {doc_id}"
                            )
                except TimeoutError:
                    detector.logger.warning(
                        f"Timeout retrieving embedding for document {doc_id}"
                    )
                except Exception as e:  # pragma: no cover
                    detector.logger.error(
                        f"Error retrieving embedding for document {doc_id}: {e}"
                    )
            # -----------------------------------------------------------

        # TODO [L2]: Concurrent embedding retrieval
        # Use Case: Fetch all embeddings in parallel for performance
        # Git: "Performance optimizations" - asyncio.gather for parallel I/O
        # -----------------------------------------------------------
        await asyncio.gather(
            *(fetch_embedding(doc_id) for doc_id in document_ids),
            return_exceptions=True,
        )
        # -----------------------------------------------------------
        return embeddings
    except Exception as e:  # pragma: no cover
        detector.logger.error(f"Error retrieving document embeddings: {e}")
        return {}


def calculate_vector_similarity(
    _detector: Any, embedding1: list[float], embedding2: list[float]
) -> float:
    """Cosine similarity with clipping to [-1, 1].

    Use Case: Measure semantic similarity between document embeddings
    Business Rule: cosine_similarity = dot(a,b) / (norm(a) * norm(b))
    """
    # TODO [L2]: Implement cosine similarity calculation
    # Data Flow: embedding1, embedding2 -> numpy arrays -> dot product / norms
    # Business Rule: Return 0.0 if any vector has zero norm
    # -----------------------------------------------------------
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    except Exception:  # pragma: no cover
        return 0.0
    # -----------------------------------------------------------


async def filter_by_vector_similarity(
    detector: Any, documents: list[Any]
) -> list[tuple]:
    """Filter document pairs by vector similarity within configured band.

    Use Case: Pre-filter document pairs before expensive analysis
    Business Rule: Only pairs with similarity <= MAX_VECTOR_SIMILARITY (0.98)
    Git: "Enhanced conflict detection and similarity calculations" (commit 2a3cd6b)
    """
    similar_pairs: list[tuple] = []

    # TODO [L2]: Validate minimum document count
    # Business Rule: Need at least 2 documents to form pairs
    # -----------------------------------------------------------
    if len(documents) < 2:
        return similar_pairs
    # -----------------------------------------------------------

    # TODO [L2]: Build document ID list and fetch embeddings
    # Data Flow: documents -> extract document_id or generate from source_type:source_title
    # -----------------------------------------------------------
    document_ids = [
        getattr(doc, "document_id", f"{doc.source_type}:{doc.source_title}")
        for doc in documents
    ]
    embeddings = await get_document_embeddings(detector, document_ids)
    # -----------------------------------------------------------

    # TODO [L2]: Generate pairs with similarity scores
    # Use Case: Create all possible pairs with their similarity scores
    # Business Rule: Only include pairs with similarity <= MAX_VECTOR_SIMILARITY
    #               (pairs > threshold are likely duplicates, not conflicts)
    # -----------------------------------------------------------
    for i, doc1 in enumerate(documents):
        for j, doc2 in enumerate(documents[i + 1 :], i + 1):
            doc1_id = document_ids[i]
            doc2_id = document_ids[j]
            similarity_score = 0.0
            if doc1_id in embeddings and doc2_id in embeddings:
                similarity_score = calculate_vector_similarity(
                    detector, embeddings[doc1_id], embeddings[doc2_id]
                )
            # Only filter out pairs that are too similar (likely duplicates)
            # Include pairs with low similarity for text/metadata analysis
            if similarity_score <= detector.MAX_VECTOR_SIMILARITY:
                similar_pairs.append((doc1, doc2, similarity_score))
    # -----------------------------------------------------------

    # TODO [L2]: Sort pairs by similarity (highest first)
    # Use Case: Process most similar (more likely conflicting) pairs first
    # -----------------------------------------------------------
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    # -----------------------------------------------------------
    return similar_pairs


def should_analyze_for_conflicts(_detector: Any, doc1: Any, doc2: Any) -> bool:
    """Pre-screen documents before conflict analysis.

    Use Case: Quick filter to skip invalid document pairs
    Business Rule: Skip if text too short, same ID, or identical content
    Git: "Edge Case Handling" - Empty document handling
    Test: test_should_analyze_for_conflicts_edge_cases
    """
    # TODO [L2]: Implement document eligibility checks
    # Business Rule:
    #   - Return False if either doc is None
    #   - Return False if text < 10 characters
    #   - Return False if same document_id
    #   - Return False if identical text content
    # -----------------------------------------------------------
    if not doc1 or not doc2:
        return False
    text1 = doc1.text if getattr(doc1, "text", None) else ""
    text2 = doc2.text if getattr(doc2, "text", None) else ""
    if len(text1.strip()) < 10 or len(text2.strip()) < 10:
        return False
    if hasattr(doc1, "document_id") and hasattr(doc2, "document_id"):
        if doc1.document_id == doc2.document_id:
            return False
    if text1.strip() == text2.strip():
        return False
    return True
    # -----------------------------------------------------------


async def get_tiered_analysis_pairs(detector: Any, documents: list[Any]) -> list[tuple]:
    """Generate tiered analysis pairs for conflict detection (extracted).

    Use Case: Prioritize document pairs based on relevance scores
    Business Rule: Tiers based on average document score:
      - primary: score >= 0.8 (analyze first)
      - secondary: score >= 0.5
      - tertiary: score < 0.5 (analyze last)
    Git: "Enhanced conflict detection - tiered analysis" (commit 58a9666)
    """
    pairs: list[tuple] = []

    # TODO [L2]: Validate minimum document count
    # -----------------------------------------------------------
    if len(documents) < 2:
        return pairs
    # -----------------------------------------------------------

    # TODO [L2]: Generate pairs with tier classification
    # Data Flow: documents -> pairs with (doc1, doc2, tier, score)
    # -----------------------------------------------------------
    for i, doc1 in enumerate(documents):
        for _j, doc2 in enumerate(documents[i + 1 :], i + 1):
            score = 1.0
            if hasattr(doc1, "score") and hasattr(doc2, "score"):
                avg_doc_score = (doc1.score + doc2.score) / 2
                score = min(1.0, avg_doc_score)

            if score >= 0.8:
                tier = "primary"
            elif score >= 0.5:
                tier = "secondary"
            else:
                tier = "tertiary"

            pairs.append((doc1, doc2, tier, score))
    # -----------------------------------------------------------

    # TODO [L2]: Sort by score descending
    # Use Case: Process highest-relevance pairs first
    # -----------------------------------------------------------
    pairs.sort(key=lambda x: x[3], reverse=True)
    # -----------------------------------------------------------
    return pairs
