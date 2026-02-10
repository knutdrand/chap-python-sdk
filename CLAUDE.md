# CLAUDE.md - AI Assistant Guidelines

This document provides guidelines for AI assistants working with the chap-python-sdk codebase.

## Testing Standards

### Test Execution
```bash
make test      # Run tests
make coverage  # With coverage reporting
make lint      # Ruff format, Ruff check, MyPy, Pyright
```

Always run `make lint` and `make test` after changes.

- Test paths: `tests/`
- Name pattern: `test_*.py`


## Dependency Management

Always use `uv` (not pip):
```bash
uv add <package>              # Runtime dependency
uv add --dev <package>        # Dev dependency
uv add <package>@latest       # Update specific
uv lock --upgrade             # Update all
```

Never manually edit `pyproject.toml` for dependencies.


## Project Overview

chap-python-sdk is a validation and testing framework for chapkit models. It provides:
- Test dataset management
- CLI for running validation tests
- pytest integration for automated testing
- Support for multiple dataset formats

## Architecture

```
src/chap_python_sdk/
├── __init__.py           # Public API exports
├── cli/                  # CLI commands
│   ├── __init__.py
│   └── main.py           # CLI entry point
├── datasets/             # Test dataset definitions
│   ├── __init__.py
│   └── loader.py         # Dataset loading logic
└── validation/           # Validation logic
    ├── __init__.py
    ├── runner.py         # Test runner
    └── results.py        # Result schemas
```

## Key Dependencies

Runtime:
- `pydantic` - Data validation
- `typer` or `click` - CLI framework
- `pandas` - Data manipulation

Dev:
- `pytest`, `pytest-asyncio` - Testing
- `mypy`, `pyright` - Type checking
- `ruff` - Formatting/linting

## Visualization Tools

For creating statistical illustrations and diagrams:

### Static Visualizations
- **Altair** - Declarative statistical visualization library
  - Use for all static plots, charts, and diagrams in documentation
  - Declarative API based on Vega-Lite grammar
  - Outputs to PNG/SVG for embedding in markdown
  - Ideal for: time series plots, ACF plots, residual plots, bootstrap distributions

### Animations
- **Manim** - Mathematical animation engine
  - Use for creating explanatory animations and educational content
  - Programmatic animation creation
  - Ideal for: explaining algorithms, showing temporal processes, illustrating concepts

### Usage Guidelines
- Save static plots to `docs/images/` directory
- Use descriptive filenames: `bootstrap-block-illustration.png`
- Reference in markdown with relative paths
- Prefer SVG for diagrams, PNG for plots with many data points

## Common Patterns

### Repository Pattern Methods
- `find_*`: Single entity or None
- `find_all_*`: Sequence
- `exists_*`: Boolean
- `count`: Integer

### Result Types
Use explicit result types for validation:
```python
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of a model validation run."""

    success: bool
    errors: list[str]
    warnings: list[str]
```
