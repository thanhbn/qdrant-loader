"""Semantic analysis models for search results.

POC5: Extended to include enrichment metadata from ingestion pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnrichmentMetadata:
    """POC5: Enrichment-specific metadata from ingestion pipeline.

    This dataclass captures the output from EntityEnricher and KeywordEnricher
    during document ingestion, making it available through MCP search results.

    Attributes:
        entity_types: Mapping of entity type (PERSON, ORG, etc.) to list of entity texts
        entity_count: Total number of unique entities extracted
        has_people: Whether document contains PERSON entities
        has_organizations: Whether document contains ORG entities
        has_locations: Whether document contains location entities (GPE, LOC, FAC)
        keywords: Full keyword objects with word, score, and frequency
        keyword_list: Simple list of keyword strings for filtering
        keyword_count: Total number of keywords extracted
        top_keyword: Highest-scored keyword
    """

    # Entity enrichment data
    entity_types: dict[str, list[str]] = field(default_factory=dict)
    entity_count: int = 0
    has_people: bool = False
    has_organizations: bool = False
    has_locations: bool = False

    # Keyword enrichment data
    keywords: list[dict[str, Any]] = field(default_factory=list)
    keyword_list: list[str] = field(default_factory=list)
    keyword_count: int = 0
    top_keyword: str | None = None


@dataclass
class SemanticAnalysis:
    """Semantic analysis information from document metadata.

    Combines spaCy-based analysis (entities, topics, key_phrases) with
    enrichment metadata from the ingestion pipeline (POC5).

    Attributes:
        entities: List of entity dicts or strings from spaCy NER
        topics: List of topic dicts or strings
        key_phrases: List of key phrase dicts or strings
        pos_tags: List of POS tag dicts
        enrichment: POC5 - Enrichment metadata from ingestion pipeline
    """

    entities: list[dict | str] = field(default_factory=list)
    topics: list[dict | str] = field(default_factory=list)
    key_phrases: list[dict | str] = field(default_factory=list)
    pos_tags: list[dict] = field(default_factory=list)

    # POC5: Enrichment metadata from ingestion pipeline
    enrichment: EnrichmentMetadata | None = None
