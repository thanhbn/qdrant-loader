"""Centralized logging configuration for the MCP Server application."""

import logging
import os
import re
import sys

import structlog


class QdrantVersionFilter(logging.Filter):
    """Filter to suppress Qdrant version check warnings."""

    def filter(self, record):
        return "Failed to obtain server version" not in str(record.msg)


class ApplicationFilter(logging.Filter):
    """Filter to only show logs from our application."""

    def filter(self, record):
        # Show logs from our application and related modules
        return (
            record.name.startswith("mcp_server")
            or record.name.startswith("src.")
            or record.name == "uvicorn"
            or record.name == "fastapi"
            or record.name == "__main__"  # Allow logs from main module
            or record.name == "asyncio"  # Allow logs from asyncio
            or record.name == "main"  # Allow logs when started as a script
            or record.name == "qdrant_loader_mcp_server"  # Allow logs from the package
        )


class CleanFormatter(logging.Formatter):
    """Formatter that removes ANSI color codes."""

    def format(self, record):
        # Get the formatted message
        message = super().format(record)
        # Remove ANSI color codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", message)


# NOTE: The core logging import is deferred to enable fast server startup.
# qdrant_loader_core imports heavy dependencies (openai, spacy, etc.)
# The import is done lazily in _get_core_logging_config() instead of at module level.
_core_logging_config_cache: type | None = None
_core_logging_checked = False


def _get_core_logging_config():
    """Lazily import and cache CoreLoggingConfig to defer heavy imports."""
    global _core_logging_config_cache, _core_logging_checked
    if not _core_logging_checked:
        try:
            from qdrant_loader_core.logging import (
                LoggingConfig as CoreLoggingConfig,  # type: ignore
            )

            _core_logging_config_cache = CoreLoggingConfig
        except Exception:  # pragma: no cover - core may not be available
            _core_logging_config_cache = None
        _core_logging_checked = True
    return _core_logging_config_cache


class LoggingConfig:
    """Wrapper that standardizes env handling and tracks current config.

    Delegates to core LoggingConfig when available, while maintaining
    _initialized and _current_config for MCP server tests and utilities.

    Supports two-phase startup:
    - minimal=True: Fast initialization without heavy core imports
    - minimal=False: Full initialization with core logging (default)
    """

    _initialized = False
    _minimal_mode = False
    _current_config: tuple[str, str, str | None, bool] | None = None

    @classmethod
    def setup(
        cls,
        level: str = "INFO",
        format: str = "console",
        file: str | None = None,
        suppress_qdrant_warnings: bool = True,
        minimal: bool = False,
    ) -> None:
        # Resolve from environment when present for level; for file only when using all defaults
        env_level = os.getenv("MCP_LOG_LEVEL")
        resolved_level = (env_level or level).upper()
        env_file = os.getenv("MCP_LOG_FILE")
        all_defaults = (
            level == "INFO"
            and format == "console"
            and file is None
            and suppress_qdrant_warnings is True
        )
        resolved_file = (
            file if file is not None else (env_file if all_defaults else None)
        )
        disable_console_logging = (
            os.getenv("MCP_DISABLE_CONSOLE_LOGGING", "").lower() == "true"
        )

        # Validate level
        if not hasattr(logging, resolved_level):
            raise ValueError(f"Invalid log level: {resolved_level}")

        numeric_level = getattr(logging, resolved_level)

        # In minimal mode, skip heavy core imports for fast startup
        # Core logging will be enabled later when upgrade_from_minimal() is called
        CoreLoggingConfig = None if minimal else _get_core_logging_config()
        if CoreLoggingConfig is not None:
            # Delegate to core implementation
            CoreLoggingConfig.setup(
                level=resolved_level,
                format=format,
                file=resolved_file,
                suppress_qdrant_warnings=suppress_qdrant_warnings,
                disable_console=disable_console_logging,
            )
        else:
            # Minimal/fallback behavior - basic logging without heavy imports
            handlers: list[logging.Handler] = []
            if not disable_console_logging:
                stderr_handler = logging.StreamHandler(sys.stderr)
                stderr_handler.setFormatter(logging.Formatter("%(message)s"))
                handlers.append(stderr_handler)
            if resolved_file:
                file_handler = logging.FileHandler(resolved_file)
                file_handler.setFormatter(CleanFormatter("%(message)s"))
                handlers.append(file_handler)
            logging.basicConfig(level=numeric_level, handlers=handlers, force=True)
            if suppress_qdrant_warnings:
                logging.getLogger("qdrant_client").addFilter(QdrantVersionFilter())

        cls._initialized = True
        cls._minimal_mode = minimal
        cls._current_config = (
            resolved_level,
            format,
            resolved_file,
            suppress_qdrant_warnings,
        )

    @classmethod
    def upgrade_from_minimal(cls) -> None:
        """Upgrade from minimal logging to full core logging.

        Called during two-phase startup after heavy imports are loaded.
        This re-initializes logging with the full core implementation.
        """
        if not cls._minimal_mode:
            return  # Already using full logging

        if cls._current_config is None:
            return  # Not initialized yet

        level, fmt, file, suppress = cls._current_config
        cls._minimal_mode = False
        cls._initialized = False  # Force re-initialization
        cls.setup(
            level=level,
            format=fmt,
            file=file,
            suppress_qdrant_warnings=suppress,
            minimal=False,
        )

    @classmethod
    def get_logger(cls, name: str | None = None):  # type: ignore
        if not cls._initialized:
            # Use minimal mode for auto-initialization to avoid heavy imports
            # during module loading. Call upgrade_from_minimal() later to get
            # full core logging after heavy imports are complete.
            cls.setup(minimal=True)
        return structlog.get_logger(name)

    @classmethod
    def reconfigure(cls, *, file: str | None = None, level: str | None = None) -> None:
        """Lightweight reconfiguration for file destination and optionally log level.

        If core logging is present and supports reconfigure, delegate to it.
        Otherwise, force-replace root handlers with a new file handler (and keep stderr
        if console is enabled via env).

        Note: In minimal mode, this skips the core logging import to maintain fast startup.

        Args:
            file: Path to log file (optional)
            level: New log level (optional, e.g., "DEBUG", "INFO")
        """
        disable_console_logging = (
            os.getenv("MCP_DISABLE_CONSOLE_LOGGING", "").lower() == "true"
        )

        # Skip core logging import if in minimal mode to maintain fast startup
        CoreLoggingConfig = None if cls._minimal_mode else _get_core_logging_config()
        if CoreLoggingConfig is not None and hasattr(CoreLoggingConfig, "reconfigure"):
            CoreLoggingConfig.reconfigure(file=file, level=level)  # type: ignore
        else:
            # Determine the level to use
            if level is not None:
                resolved_level = level.upper()
            elif cls._current_config is not None:
                resolved_level = cls._current_config[0]
            else:
                resolved_level = "INFO"

            handlers: list[logging.Handler] = []
            if not disable_console_logging:
                stderr_handler = logging.StreamHandler(sys.stderr)
                stderr_handler.setFormatter(logging.Formatter("%(message)s"))
                handlers.append(stderr_handler)
            if file:
                file_handler = logging.FileHandler(file)
                file_handler.setFormatter(CleanFormatter("%(message)s"))
                handlers.append(file_handler)
            logging.basicConfig(
                level=getattr(logging, resolved_level), handlers=handlers, force=True
            )

        if cls._current_config is not None:
            old_level, fmt, _, suppress = cls._current_config
            new_level = level.upper() if level is not None else old_level
            cls._current_config = (new_level, fmt, file, suppress)
