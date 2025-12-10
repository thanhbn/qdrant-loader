"""HuggingFace Transformers NER backend implementation.

POC3-006: HuggingFaceBackend for HsEntityEnricher.

This module provides an optional NER backend using HuggingFace Transformers.
Features:
- Access to state-of-the-art NER models (BERT, RoBERTa, etc.)
- Native confidence scores from model predictions
- Multiple aggregation strategies for subword tokens
- GPU acceleration support

Requires: pip install transformers torch
"""

from typing import Any

from qdrant_loader.utils.logging import LoggingConfig

from .base import NamedEntityAnnotation, NERBackend

logger = LoggingConfig.get_logger(__name__)


class HuggingFaceBackend(NERBackend):
    """HuggingFace Transformers NER backend.

    Uses HuggingFace's pipeline API for token classification (NER).
    Provides access to modern transformer models like:
    - dslim/bert-base-NER
    - dbmdz/bert-large-cased-finetuned-conll03-english
    - xlm-roberta-large-finetuned-conll03-english

    Features:
        - Native confidence scores from model logits
        - Subword token aggregation strategies
        - GPU/CPU device selection
        - Batch processing support

    Example:
        backend = HuggingFaceBackend(
            model_name="dslim/bert-base-NER",
            device="cuda",
        )
        backend.warm_up()

        entities = backend.extract_entities(["Bill Gates founded Microsoft."])
        # [[NamedEntityAnnotation(entity="PER", text="Bill Gates", score=0.95, ...)]]

    Note:
        Entity type labels depend on the model's training data.
        Common schemes:
        - OntoNotes: PERSON, ORG, GPE, etc.
        - CoNLL: PER, ORG, LOC, MISC
    """

    def __init__(
        self,
        model_name: str,
        device: str | int | None = None,
        aggregation_strategy: str = "simple",
        batch_size: int = 8,
    ):
        """Initialize the HuggingFace backend.

        Args:
            model_name: HuggingFace model identifier
                (e.g., "dslim/bert-base-NER")
            device: Device to run model on:
                - None: Auto-detect (GPU if available)
                - "cpu": Force CPU
                - "cuda": Use first GPU
                - "cuda:0", "cuda:1": Specific GPU
                - 0, 1, etc.: GPU device index
            aggregation_strategy: How to handle subword tokens:
                - "none": Return all tokens
                - "simple": Merge adjacent tokens with same label
                - "first": Use first subword's prediction
                - "average": Average predictions
                - "max": Use highest confidence prediction
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self.batch_size = batch_size
        self._pipeline: Any = None  # transformers.Pipeline
        self._is_warm = False

    def warm_up(self) -> None:
        """Load the HuggingFace model and create pipeline.

        Downloads the model if not cached locally.

        Raises:
            ImportError: If transformers is not installed
            RuntimeError: If model loading fails
        """
        if self._is_warm:
            return

        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "HuggingFace Transformers is required for HuggingFaceBackend. "
                "Install with: pip install transformers torch"
            ) from e

        logger.info(f"Loading HuggingFace model: {self.model_name}")

        try:
            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                device=self.device,
                aggregation_strategy=self.aggregation_strategy,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace model '{self.model_name}': {e}"
            ) from e

        self._is_warm = True
        device_info = (
            self._pipeline.device if hasattr(self._pipeline, "device") else "unknown"
        )
        logger.info(f"HuggingFace backend ready: {self.model_name} on {device_info}")

    def extract_entities(
        self,
        texts: list[str],
    ) -> list[list[NamedEntityAnnotation]]:
        """Extract entities from texts using HuggingFace pipeline.

        The pipeline handles batching internally for efficiency.

        Args:
            texts: List of text strings to process

        Returns:
            List of entity annotations per text with confidence scores
        """
        if not self._is_warm:
            self.warm_up()

        results: list[list[NamedEntityAnnotation]] = []

        # HuggingFace pipeline handles batching
        # Process in chunks for memory efficiency
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self._pipeline(batch)

            # Handle single text vs batch
            if len(batch) == 1:
                batch_results = [batch_results]

            for doc_entities in batch_results:
                entities: list[NamedEntityAnnotation] = []

                for ent in doc_entities:
                    # HuggingFace returns entity_group with aggregation
                    entity_type = ent.get("entity_group") or ent.get(
                        "entity", "UNKNOWN"
                    )

                    # Some models prefix with B-, I-, etc.
                    if entity_type.startswith(("B-", "I-", "E-", "S-")):
                        entity_type = entity_type[2:]

                    annotation = NamedEntityAnnotation(
                        entity=entity_type,
                        text=ent.get("word", ""),
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                        score=float(ent.get("score", 0.0)),
                    )
                    entities.append(annotation)

                results.append(entities)

        return results

    def shutdown(self) -> None:
        """Release HuggingFace model resources."""
        if self._pipeline is not None:
            # Clear model from memory
            del self._pipeline
            self._pipeline = None

            # Optionally clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        self._is_warm = False
        logger.debug("HuggingFace backend shutdown")

    def __repr__(self) -> str:
        return (
            f"HuggingFaceBackend("
            f"model={self.model_name!r}, "
            f"device={self.device}, "
            f"strategy={self.aggregation_strategy!r}, "
            f"warm={self._is_warm})"
        )
