"""spaCy NER backend implementation.

POC3-005: SpaCyBackend for HsEntityEnricher.

This module provides the default NER backend using spaCy models.
Features:
- Batch processing with nlp.pipe() for efficiency
- Lazy model loading
- Configurable pipeline optimization
"""

from typing import Any

from qdrant_loader.utils.logging import LoggingConfig

from .base import NamedEntityAnnotation, NERBackend

logger = LoggingConfig.get_logger(__name__)


class SpaCyBackend(NERBackend):
    """spaCy-based NER backend.

    Uses spaCy's pre-trained NER models for entity extraction.
    Supports all spaCy language models with NER pipeline component.

    Features:
        - Batch processing with nlp.pipe()
        - Pipeline optimization (disables unused components)
        - Support for all spaCy entity types (18 OntoNotes types)

    Limitations:
        - spaCy does not provide native confidence scores
        - Uses 1.0 as default score (can be overridden)

    Example:
        backend = SpaCyBackend(model_name="en_core_web_sm")
        backend.warm_up()

        entities = backend.extract_entities(["Bill Gates founded Microsoft."])
        # [[NamedEntityAnnotation(entity="PERSON", text="Bill Gates", ...)]]
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        batch_size: int = 32,
        disable_pipes: list[str] | None = None,
    ):
        """Initialize the spaCy backend.

        Args:
            model_name: spaCy model name (e.g., "en_core_web_sm")
            batch_size: Batch size for nlp.pipe() processing
            disable_pipes: Additional pipeline components to disable.
                By default, disables all except 'ner' and 'tok2vec'.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.disable_pipes = disable_pipes
        self._nlp: Any = None  # spacy.Language
        self._is_warm = False

    def warm_up(self) -> None:
        """Load the spaCy model and optimize pipeline.

        Loads the specified spaCy model and disables unused pipeline
        components for faster inference.

        Raises:
            ImportError: If spaCy is not installed
            OSError: If the model is not found
        """
        if self._is_warm:
            return

        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy is required for SpaCyBackend. " "Install with: pip install spacy"
            ) from e

        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            self._nlp = spacy.load(self.model_name)
        except OSError as e:
            raise OSError(
                f"spaCy model '{self.model_name}' not found. "
                f"Install with: python -m spacy download {self.model_name}"
            ) from e

        # Optimize pipeline by disabling unused components
        if self.disable_pipes:
            disable = self.disable_pipes
        else:
            # Keep only NER-essential components
            keep = {"ner", "tok2vec", "transformer"}
            disable = [p for p in self._nlp.pipe_names if p not in keep]

        if disable:
            for pipe_name in disable:
                if pipe_name in self._nlp.pipe_names:
                    self._nlp.disable_pipe(pipe_name)
            logger.debug(f"Disabled spaCy pipes: {disable}")

        self._is_warm = True
        logger.info(f"spaCy backend ready: {self.model_name}")

    def extract_entities(
        self,
        texts: list[str],
    ) -> list[list[NamedEntityAnnotation]]:
        """Extract entities from texts using spaCy.

        Uses nlp.pipe() for efficient batch processing.

        Args:
            texts: List of text strings to process

        Returns:
            List of entity annotations per text

        Note:
            spaCy doesn't provide confidence scores natively.
            All entities receive a score of 1.0.
        """
        if not self._is_warm:
            self.warm_up()

        results: list[list[NamedEntityAnnotation]] = []

        # Use nlp.pipe for batch processing
        for doc in self._nlp.pipe(texts, batch_size=self.batch_size):
            entities: list[NamedEntityAnnotation] = []

            for ent in doc.ents:
                # spaCy doesn't expose confidence scores by default
                # Use 1.0 as placeholder (entity was detected)
                annotation = NamedEntityAnnotation(
                    entity=ent.label_,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=1.0,
                )
                entities.append(annotation)

            results.append(entities)

        return results

    def shutdown(self) -> None:
        """Release spaCy model resources."""
        self._nlp = None
        self._is_warm = False
        logger.debug("spaCy backend shutdown")

    def __repr__(self) -> str:
        return f"SpaCyBackend(model={self.model_name!r}, warm={self._is_warm})"
