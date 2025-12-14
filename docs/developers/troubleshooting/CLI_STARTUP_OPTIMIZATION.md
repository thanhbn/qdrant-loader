# CLI Startup Performance Optimization

## Overview

This document captures the results of lazy import optimizations implemented to improve CLI responsiveness for `qdrant-loader`.

## Problem Statement

The CLI was taking **44+ seconds** to show `--help` or `--version` because heavy dependencies were being imported at module load time, even when not needed.

## Root Cause Analysis

Profile data identified the following heavy imports at startup:

| Import | Time (seconds) | Description |
|--------|----------------|-------------|
| `openai` SDK | ~20s | Via `qdrant_loader_core.llm` |
| `spacy` | ~10s | Via `text_processor.py` |
| `nltk` | ~5s | Via `text_processor.py` |
| `langchain_text_splitters` | ~3s | Via `text_processor.py` |
| `qdrant_client` | ~5s | Via `qdrant_manager.py` |

## Optimizations Implemented

### OPT-001: Lazy OpenAI SDK Import

**Files Modified:**
- `packages/qdrant-loader-core/src/qdrant_loader_core/__init__.py`
- `packages/qdrant-loader-core/src/qdrant_loader_core/llm/factory.py`

**Changes:**
- Added `__getattr__` lazy loading pattern to package `__init__.py`
- Created lazy provider loader functions with caching:
  - `_get_openai_provider()`
  - `_get_ollama_provider()`
  - `_get_azure_provider()`
- Used `TYPE_CHECKING` guard for type hints

### OPT-003: Lazy NLP Imports

**Files Modified:**
- `packages/qdrant-loader/src/qdrant_loader/core/text_processing/text_processor.py`
- `packages/qdrant-loader/src/qdrant_loader/core/chunking/strategy/base_strategy.py`

**Changes:**
- Moved `nltk`, `spacy`, `langchain_text_splitters` imports inside `TextProcessor.__init__`
- Moved `TextProcessor` import inside `BaseChunkingStrategy.__init__`
- Added docstring notes explaining the lazy import pattern

### OPT-005: Lazy qdrant_client Import

**Files Modified:**
- `packages/qdrant-loader/src/qdrant_loader/core/qdrant_manager.py`

**Changes:**
- Moved `QdrantClient` import inside `connect()` method
- Moved `Distance`, `VectorParams` import inside `create_collection()` method
- Moved `models` import inside methods that use it (`search_with_project_filter`, `delete_points_by_document_id`)
- Used `TYPE_CHECKING` guard for type hints

### OPT-006: Break Config Import Chain (December 2024)

**Problem:**
`config_loader.py` imported `DatabaseDirectoryError` from `qdrant_loader.config.state`, which triggered loading the entire `config` package including pydantic_settings, structlog, and rich (~3s overhead).

**Files Modified:**
- `packages/qdrant-loader/src/qdrant_loader/exceptions.py` (NEW)
- `packages/qdrant-loader/src/qdrant_loader/config/state.py`
- `packages/qdrant-loader/src/qdrant_loader/cli/config_loader.py`

**Changes:**
- Created new top-level `qdrant_loader/exceptions.py` with lightweight exception classes
- Moved `DatabaseDirectoryError` from `config/state.py` to `exceptions.py`
- Updated imports to use the lightweight exceptions module
- Re-exported `DatabaseDirectoryError` from `config/state.py` for backward compatibility

### OPT-007: Lazy Project Commands Group (December 2024)

**Problem:**
The `project_commands.py` module imports heavy dependencies (rich, sqlalchemy, pydantic_settings) at module level. When added to CLI, this adds ~4s to `--help` response time.

**Files Modified:**
- `packages/qdrant-loader/src/qdrant_loader/cli/cli.py`

**Changes:**
- Created `LazyProjectGroup` class extending `click.Group`
- Implemented lazy loading pattern that returns subcommand names without importing real module
- Real `project_commands` module only loaded when user actually invokes a project subcommand
- `--help` shows `project` command and its subcommands without triggering heavy imports

**Pattern:**
```python
class LazyProjectGroup(click.Group):
    _real_group = None

    def list_commands(self, ctx):
        # Return known subcommand names without importing
        return ["list", "status", "validate"]

    def get_command(self, ctx, cmd_name):
        if self._real_group is None:
            from .project_commands import project_cli
            LazyProjectGroup._real_group = project_cli
        return self._real_group.get_command(ctx, cmd_name)

@cli.group(name="project", cls=LazyProjectGroup)
def project():
    """Project management commands."""
    pass
```

## Results

| Metric | Before | Phase 1 | Phase 2 | Improvement |
|--------|--------|---------|---------|-------------|
| `qdrant-loader --help` | ~44+ sec | ~6.7 sec | **~0.77 sec** | **~98% faster** |
| `qdrant-loader --version` | ~44+ sec | ~5.5 sec | **~0.62 sec** | **~99% faster** |
| CLI module import | ~44+ sec | ~3.7 sec | **~0.96 sec** | **~98% faster** |

**Phase 1:** Initial lazy imports (OpenAI, NLP, qdrant_client)
**Phase 2:** Config import chain break + lazy project commands

## Key Principles

1. **Lazy imports defer, not eliminate** - The import cost is still paid when the functionality is actually used
2. **CLI responsiveness** - Fast `--help` and `--version` improve user experience
3. **TYPE_CHECKING guard** - Use for type hints without runtime import cost
4. **Method-level imports** - Import heavy dependencies inside methods that use them

## Pattern Examples

### Package-level `__getattr__` pattern:

```python
__all__ = ["SomeClass", "some_function"]

def __getattr__(name: str):
    """Lazy import attributes to avoid loading heavy dependencies at startup."""
    if name in __all__:
        from .heavy_module import SomeClass, some_function
        _exports = {
            "SomeClass": SomeClass,
            "some_function": some_function,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Method-level lazy import:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heavy_library import HeavyClass

class MyClass:
    def method_using_heavy_lib(self) -> "HeavyClass":
        from heavy_library import HeavyClass
        return HeavyClass()
```

## Future Enhancements

Potential areas for further optimization (already deferred with current optimizations):

1. **Lazy import `tiktoken`** in `base_strategy.py` (~2s) - Now deferred via lazy chunking
2. **Lazy import connectors** - Only import connector modules when that source type is used
3. **Lazy CLI command implementations** - Apply LazyGroup pattern to `init`, `ingest`, `config` commands
4. **Profile remaining imports** - Run `python -X importtime` to identify other slow imports

## Profiling Commands

```bash
# Profile import time
python -X importtime -c "from qdrant_loader.cli.cli import cli" 2>&1 | \
    grep -E "import time:.*[0-9]{6}" | sort -t'|' -k2 -rn | head -30

# Measure --help time
time python -c "from qdrant_loader.cli.cli import cli; cli(['--help'])"

# Measure --version time
time python -c "from qdrant_loader.cli.cli import cli; cli(['--version'])"
```

## Date

- Phase 1 Implemented: December 2024
- Phase 2 Implemented: December 2024
