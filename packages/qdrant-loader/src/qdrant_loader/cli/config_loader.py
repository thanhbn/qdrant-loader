"""CLI configuration loader utilities.

Note: LoggingConfig and heavy config modules are lazily imported to improve CLI startup time.
DatabaseDirectoryError is imported from lightweight exceptions module to avoid pydantic import chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from click.exceptions import ClickException

# Import from top-level exceptions module instead of config.state module
# to avoid loading pydantic and the entire config chain at startup
from qdrant_loader.exceptions import DatabaseDirectoryError

if TYPE_CHECKING:
    from qdrant_loader.utils.logging import LoggingConfig


def _get_logging_config():
    """Lazily import LoggingConfig."""
    from qdrant_loader.utils.logging import LoggingConfig

    return LoggingConfig


# Back-compat shim for tests that import _get_logger from cli.cli
def _get_logger():
    return _get_logging_config().get_logger(__name__)


def setup_workspace(workspace_path: Path):
    try:
        from qdrant_loader.config.workspace import (
            create_workspace_structure,
        )
        from qdrant_loader.config.workspace import setup_workspace as _setup

        create_workspace_structure(workspace_path)
        workspace_config = _setup(workspace_path)
        return workspace_config
    except ValueError as e:
        raise ClickException(str(e)) from e
    except Exception as e:  # pragma: no cover - handled by CLI tests
        raise ClickException(f"Failed to setup workspace: {str(e)!s}") from e


def load_config_with_workspace(
    workspace_config: Any | None = None,
    config_path: Path | None = None,
    env_path: Path | None = None,
    skip_validation: bool = False,
) -> None:
    try:
        from qdrant_loader.config import initialize_config_with_workspace

        if workspace_config:
            _get_logging_config().get_logger(__name__).debug(
                "Loading configuration in workspace mode"
            )
            initialize_config_with_workspace(
                workspace_config, skip_validation=skip_validation
            )
        else:
            _get_logging_config().get_logger(__name__).debug(
                "Loading configuration in traditional mode"
            )
            load_config(config_path, env_path, skip_validation)
    except Exception as e:
        _get_logging_config().get_logger(__name__).error("config_load_failed", error=str(e))
        raise ClickException(f"Failed to load configuration: {str(e)!s}") from e


def create_database_directory(path: Path) -> bool:
    try:
        abs_path = path.resolve()
        _get_logging_config().get_logger(__name__).info(
            "The database directory does not exist", path=str(abs_path)
        )
        import click

        if click.confirm("Would you like to create this directory?", default=True):
            abs_path.mkdir(parents=True, mode=0o755, exist_ok=True)
            _get_logging_config().get_logger(__name__).info("Created directory: %s", abs_path)
            return True
        return False
    except Exception as e:  # pragma: no cover - interactive path
        raise ClickException(f"Failed to create directory: {str(e)!s}") from e


def load_config(
    config_path: Path | None = None,
    env_path: Path | None = None,
    skip_validation: bool = False,
) -> None:
    try:
        from qdrant_loader.config import initialize_config

        if config_path is not None:
            if not config_path.exists():
                _get_logging_config().get_logger(__name__).error(
                    "config_not_found", path=str(config_path)
                )
                raise ClickException(f"Config file not found: {str(config_path)!s}")
            initialize_config(config_path, env_path, skip_validation=skip_validation)
            return

        default_config = Path("config.yaml")
        if default_config.exists():
            initialize_config(default_config, env_path, skip_validation=skip_validation)
            return

        raise ClickException(
            f"No config file found. Please specify a config file or create config.yaml in the current directory: {str(default_config)!s}"
        )

    except DatabaseDirectoryError as e:
        if skip_validation:
            return
        error_path = e.path
        abs_path = error_path.resolve()
        if not create_database_directory(abs_path):
            raise ClickException(
                "Database directory creation declined. Exiting."
            ) from e
        # After successful creation, initialize once using the original config_path if provided,
        # otherwise fall back to default config.yaml
        from qdrant_loader.config import initialize_config

        target_config = config_path if config_path is not None else Path("config.yaml")
        initialize_config(target_config, env_path, skip_validation=skip_validation)
    except ClickException:
        raise
    except Exception as e:
        _get_logging_config().get_logger(__name__).error("config_load_failed", error=str(e))
        raise ClickException(f"Failed to load configuration: {str(e)!s}") from e
