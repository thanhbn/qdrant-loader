"""Abstract base class for NER backends.

POC3-005: NERBackend interface inspired by Haystack's backend pattern.

This module defines:
- NamedEntityAnnotation: Haystack-compatible entity annotation format
- NERBackend: Abstract interface for NER model backends
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class NamedEntityAnnotation:
    """Haystack-compatible named entity annotation.

    This format is compatible with Haystack's NamedEntityExtractor output,
    enabling interoperability with Haystack pipelines.

    Attributes:
        entity: Entity type label (e.g., "PERSON", "ORG", "GPE")
        text: Surface form of the entity (e.g., "Bill Gates")
        start: Start character position in source text
        end: End character position in source text
        score: Confidence score from 0.0 to 1.0

    Example:
        annotation = NamedEntityAnnotation(
            entity="PERSON",
            text="Bill Gates",
            start=10,
            end=20,
            score=0.95,
        )
    """

    entity: str
    text: str
    start: int
    end: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"NamedEntityAnnotation("
            f"entity={self.entity!r}, text={self.text!r}, "
            f"start={self.start}, end={self.end}, score={self.score:.2f})"
        )


class NERBackend(ABC):
    """Abstract interface for Named Entity Recognition backends.

    All NER backends must implement this interface to be used with
    HsEntityEnricher. The interface supports:

    1. Lazy model loading via warm_up()
    2. Batch processing for efficiency
    3. Resource cleanup via shutdown()

    Example implementation:
        class MyNERBackend(NERBackend):
            def warm_up(self) -> None:
                self.model = load_my_model()

            def extract_entities(
                self, texts: list[str]
            ) -> list[list[NamedEntityAnnotation]]:
                results = []
                for text in texts:
                    entities = self.model.predict(text)
                    results.append([
                        NamedEntityAnnotation(...)
                        for ent in entities
                    ])
                return results

            def shutdown(self) -> None:
                self.model = None
    """

    @abstractmethod
    def warm_up(self) -> None:
        """Load the NER model into memory.

        This method is called lazily before the first extraction.
        Implementations should load model weights and prepare for inference.

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def extract_entities(
        self,
        texts: list[str],
    ) -> list[list[NamedEntityAnnotation]]:
        """Extract named entities from multiple texts.

        Batch processing is supported for efficiency. Each input text
        produces a list of NamedEntityAnnotation objects.

        Args:
            texts: List of text strings to process

        Returns:
            List of entity lists, one per input text.
            Empty lists for texts with no entities.

        Raises:
            RuntimeError: If extraction fails
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up model resources.

        Called when the enricher is being shut down. Implementations
        should release GPU memory, close connections, etc.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
