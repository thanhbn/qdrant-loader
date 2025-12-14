"""Configuration exceptions.

This module contains lightweight exception classes that can be imported
without triggering heavy dependency chains like pydantic.
"""

from pathlib import Path


class DatabaseDirectoryError(Exception):
    """Exception raised when database directory needs to be created."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"Database directory does not exist: {path}")
