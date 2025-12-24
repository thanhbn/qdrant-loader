"""Chunking worker for processing documents into chunks."""

import sys
import asyncio
import concurrent.futures
from collections.abc import AsyncIterator

import psutil

from qdrant_loader.core.chunking.chunking_service import ChunkingService
from qdrant_loader.core.document import Document
from qdrant_loader.core.monitoring import prometheus_metrics
from qdrant_loader.utils.logging import LoggingConfig

from .base_worker import BaseWorker

logger = LoggingConfig.get_logger(__name__)

def _detect_environment() -> dict:
    """Detect runtime environment for adaptive scaling."""
    env_info = {
        "platform": sys.platform,
        "is_wsl": False,
        "is_windows": sys.platform == "win32",
        "is_mac": sys.platform == "darwin",
        "is_linux": sys.platform == "linux",
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_count": psutil.cpu_count() or 1,
    }

    # Detect WSL
    if sys.platform == "linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    env_info["is_wsl"] = True
        except Exception:
            pass

    return env_info


_ENV_INFO = _detect_environment()


def _get_memory_pressure() -> dict:
    """Get current memory pressure info."""
    mem = psutil.virtual_memory()
    return {
        "available_gb": mem.available / (1024**3),
        "percent_used": mem.percent,
        "is_high_pressure": mem.percent > 80,
        "is_critical": mem.percent > 90,
    }

class ChunkingWorker(BaseWorker):
    """Handles document chunking with controlled concurrency."""

    def __init__(
        self,
        chunking_service: ChunkingService,
        chunk_executor: concurrent.futures.ThreadPoolExecutor,
        max_workers: int = 10,
        queue_size: int = 1000,
        shutdown_event: asyncio.Event | None = None,
    ):
        super().__init__(max_workers, queue_size)
        self.chunking_service = chunking_service
        self.chunk_executor = chunk_executor
        self.shutdown_event = shutdown_event or asyncio.Event()
        self._active_tasks = 0  # NEW: Track concurrent tasks

        # NEW: Log environment info once
        logger.info(
            f"ChunkingWorker initialized - Environment: "
            f"platform={_ENV_INFO['platform']}, "
            f"is_wsl={_ENV_INFO['is_wsl']}, "
            f"total_memory={_ENV_INFO['total_memory_gb']:.1f}GB"
        )

    async def process(self, document: Document) -> list:
        """Process a single document into chunks.

        Args:
            document: The document to chunk

        Returns:
            List of chunks
        """
        logger.debug(f"Chunker_worker started for doc {document.id}")

        try:
            # Check for shutdown signal
            if self.shutdown_event.is_set():
                logger.debug(f"Chunker_worker {document.id} exiting due to shutdown")
                return []

            # Update metrics
            prometheus_metrics.CPU_USAGE.set(psutil.cpu_percent())
            prometheus_metrics.MEMORY_USAGE.set(psutil.virtual_memory().percent)

            # Run chunking in a thread pool for true parallelism
            with prometheus_metrics.CHUNKING_DURATION.time():
                # Calculate adaptive timeout based on document size
                adaptive_timeout = self._calculate_adaptive_timeout(document)

                # Log timeout decision for debugging
                logger.debug(
                    f"Adaptive timeout for {document.url}: {adaptive_timeout:.1f}s "
                    f"(size: {len(document.content)} bytes)"
                )

                # Add timeout to prevent hanging on chunking
                chunks = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        self.chunk_executor,
                        self.chunking_service.chunk_document,
                        document,
                    ),
                    timeout=adaptive_timeout,
                )

                # Check for shutdown before returning chunks
                if self.shutdown_event.is_set():
                    logger.debug(
                        f"Chunker_worker {document.id} exiting due to shutdown after chunking"
                    )
                    return []

                # Add document reference to chunk for later state tracking
                for chunk in chunks:
                    chunk.metadata["parent_document"] = document

                logger.debug(f"Chunked doc {document.id} into {len(chunks)} chunks")
                return chunks

        except asyncio.CancelledError:
            logger.debug(f"Chunker_worker {document.id} cancelled")
            raise
        except TimeoutError:
            # Provide more detailed timeout information
            doc_size = len(document.content)
            timeout_used = self._calculate_adaptive_timeout(document)
            logger.error(
                f"Chunking timed out for document '{document.url}' "
                f"(size: {doc_size:,} bytes, timeout: {timeout_used:.1f}s). "
                f"Consider increasing chunking timeout or checking document complexity."
            )
            raise
        except Exception as e:
            logger.error(f"Chunking failed for doc {document.url}: {e}")
            raise

    async def process_documents(self, documents: list[Document]) -> AsyncIterator:
        """Process documents into chunks.

        Args:
            documents: List of documents to process

        Yields:
            Chunks from processed documents
        """
        logger.debug("ChunkingWorker started")
        logger.info(f"🔄 Processing {len(documents)} documents for chunking...")

        try:
            # Process documents with controlled concurrency but stream results
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_and_yield(doc, doc_index):
                """Process a single document and yield its chunks."""
                try:
                    async with semaphore:
                        if self.shutdown_event.is_set():
                            logger.debug(
                                f"ChunkingWorker exiting due to shutdown (doc {doc_index})"
                            )
                            return

                        logger.debug(
                            f"🔄 Processing document {doc_index + 1}/{len(documents)}: {doc.id}"
                        )
                        chunks = await self.process(doc)

                        if chunks:
                            logger.debug(
                                f"✓ Document {doc_index + 1}/{len(documents)} produced {len(chunks)} chunks"
                            )
                            return chunks
                        else:
                            logger.debug(
                                f"⚠️ Document {doc_index + 1}/{len(documents)} produced no chunks"
                            )
                            return []

                except Exception as e:
                    logger.error(
                        f"❌ Chunking failed for document {doc_index + 1}/{len(documents)} ({doc.id}): {e}"
                    )
                    return []

            # Create tasks for all documents
            tasks = [process_and_yield(doc, i) for i, doc in enumerate(documents)]

            # Process tasks as they complete and yield chunks immediately
            chunk_count = 0
            completed_docs = 0

            for coro in asyncio.as_completed(tasks):
                if self.shutdown_event.is_set():
                    logger.debug("ChunkingWorker exiting due to shutdown")
                    break

                try:
                    chunks = await coro
                    completed_docs += 1

                    if chunks:
                        for chunk in chunks:
                            if not self.shutdown_event.is_set():
                                chunk_count += 1
                                yield chunk
                            else:
                                logger.debug("ChunkingWorker exiting due to shutdown")
                                return

                    # Log progress every 10 documents or at completion
                    if completed_docs % 10 == 0 or completed_docs == len(documents):
                        logger.info(
                            f"🔄 Chunking progress: {completed_docs}/{len(documents)} documents, {chunk_count} chunks generated"
                        )

                except Exception as e:
                    logger.error(f"❌ Error processing chunking task: {e}")
                    completed_docs += 1

            logger.info(
                f"✅ Chunking completed: {completed_docs}/{len(documents)} documents processed, {chunk_count} total chunks"
            )

        except asyncio.CancelledError:
            logger.debug("ChunkingWorker cancelled")
            raise
        finally:
            logger.debug("ChunkingWorker exited")

    def _calculate_adaptive_timeout(self, document: Document) -> float:
        """Calculate adaptive timeout based on document AND environment."""
        doc_size = len(document.content)

        # Base timeouts by document size
        if doc_size < 1_000:
            base_timeout = 30.0
        elif doc_size < 10_000:
            base_timeout = 60.0
        elif doc_size < 50_000:
            base_timeout = 120.0
        elif doc_size < 100_000:
            base_timeout = 240.0
        else:
            base_timeout = 360.0

        # === Environment-based adjustments ===
        
        # WSL has significant I/O overhead
        if _ENV_INFO["is_wsl"]:
            base_timeout *= 2.0

        # Low memory systems need more time
        if _ENV_INFO["total_memory_gb"] < 8:
            base_timeout *= 1.5

        # Check current memory pressure
        mem_pressure = _get_memory_pressure()
        if mem_pressure["is_high_pressure"]:
            base_timeout *= 1.5
        if mem_pressure["is_critical"]:
            base_timeout *= 2.0

        # === Content-based adjustments ===
        if document.content_type and document.content_type.lower() == "html":
            base_timeout *= 1.5

        if hasattr(document, "metadata") and document.metadata.get("conversion_method"):
            base_timeout *= 1.5

        # Size-based scaling
        size_factor = min(doc_size / 50000, 4.0)
        adaptive_timeout = base_timeout * (1 + size_factor)

        # Concurrency adjustment
        if self._active_tasks > 5:
            adaptive_timeout *= 1.2

        # Environment-aware max timeout
        max_timeout = 900.0 if _ENV_INFO["is_wsl"] else 600.0
        
        return min(adaptive_timeout, max_timeout)

    def _get_effective_max_workers(self) -> int:
        """Calculate effective max workers based on environment."""
        effective = self.max_workers

        if _ENV_INFO["is_wsl"]:
            effective = max(1, effective // 2)
            
        if _ENV_INFO["total_memory_gb"] < 8:
            effective = max(1, min(effective, 4))

        mem_pressure = _get_memory_pressure()
        if mem_pressure["is_high_pressure"]:
            effective = max(1, effective // 2)

        return effective