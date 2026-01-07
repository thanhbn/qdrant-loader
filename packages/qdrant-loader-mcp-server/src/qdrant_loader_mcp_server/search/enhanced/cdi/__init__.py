"""Cross-Document Intelligence (CDI) subpackage.

This package hosts extracted models, interfaces, utilities, and the pipeline
scaffolding for cross-document intelligence. The initial extraction focuses on
value types and contracts only, with no behavioral changes.

Workflow ID: W4-cdi-conflict-cff57ee0
"""

from .config import ConflictDetectionConfig, get_config, reset_config
from .topic_filter import (
    TopicFilter,
    TopicClassification,
    get_topic_filter,
    reset_topic_filter,
)
from .nli_detector import (
    NLIDetector,
    NLILabel,
    NLIResult,
    MockNLIModel,
    get_nli_detector,
    reset_nli_detector,
    cached_contradiction_check,
    detect_pairwise_contradictions,
)
from .atomic_facts import (
    AtomicFact,
    AtomicFactExtractor,
    ExtractionResult,
    FactComparator,
    FactType,
    get_fact_extractor,
    reset_fact_extractor,
    extract_facts_cached,
    compare_document_facts,
)
from .aggregator import (
    AggregatedResult,
    ConflictDetectionPipeline,
    ConflictSeverity,
    EvidenceAggregator,
    EvidenceItem,
    EvidenceSource,
    EvidenceWeights,
    detect_conflict,
    get_evidence_aggregator,
    reset_evidence_aggregator,
)
from .models import (
    CitationNetwork,
    ClusteringStrategy,
    ComplementaryContent,
    ConflictAnalysis,
    DocumentCluster,
    DocumentSimilarity,
    RelationshipType,
    SimilarityMetric,
)

__all__ = [
    # Configuration
    "ConflictDetectionConfig",
    "get_config",
    "reset_config",
    # Topic Filter
    "TopicFilter",
    "TopicClassification",
    "get_topic_filter",
    "reset_topic_filter",
    # NLI Detector
    "NLIDetector",
    "NLILabel",
    "NLIResult",
    "MockNLIModel",
    "get_nli_detector",
    "reset_nli_detector",
    "cached_contradiction_check",
    "detect_pairwise_contradictions",
    # Atomic Facts
    "AtomicFact",
    "AtomicFactExtractor",
    "ExtractionResult",
    "FactComparator",
    "FactType",
    "get_fact_extractor",
    "reset_fact_extractor",
    "extract_facts_cached",
    "compare_document_facts",
    # Evidence Aggregator
    "AggregatedResult",
    "ConflictDetectionPipeline",
    "ConflictSeverity",
    "EvidenceAggregator",
    "EvidenceItem",
    "EvidenceSource",
    "EvidenceWeights",
    "detect_conflict",
    "get_evidence_aggregator",
    "reset_evidence_aggregator",
    # Models
    "SimilarityMetric",
    "ClusteringStrategy",
    "RelationshipType",
    "DocumentSimilarity",
    "DocumentCluster",
    "CitationNetwork",
    "ComplementaryContent",
    "ConflictAnalysis",
]
