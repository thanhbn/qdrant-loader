"""CLI commands module.

Note: Commands are lazily imported to improve CLI startup time.
"""

__all__ = [
    "run_init",
    "run_pipeline_ingestion",
]


def __getattr__(name: str):
    """Lazy import commands to avoid loading heavy dependencies at startup."""
    if name == "run_init":
        from .init import run_init

        return run_init
    if name == "run_pipeline_ingestion":
        from .ingest import run_pipeline_ingestion

        return run_pipeline_ingestion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
