# CLAUDE.md - AI Assistant Guidelines

This document provides guidelines for AI assistants working with the chap-python-sdk codebase.

## NO EMOJIS EVER

ABSOLUTELY NO EMOJIS IN ANY CONTEXT:
- NO emojis in commit messages
- NO emojis in PR titles or descriptions
- NO emojis in code or comments
- NO emojis in docstrings
- NO emojis in documentation
- NO emojis in markdown files
- NO emojis in any output

Use plain text alternatives:
- "[x]" instead of checkmarks
- "CRITICAL" instead of warning symbols
- "Note:" instead of note symbols
- "OK" instead of thumbs up

## Documentation Standards

Every Python file must have:
- One-line module docstring at the top
- One-line docstring for every class
- One-line docstring for every method/function

Format: Triple quotes `"""docstring"""`

Example:
```python
"""Module for validating chapkit models."""

class ModelValidator:
    """Validates chapkit model implementations against test datasets."""

    def validate(self, model_path: str) -> ValidationResult:
        """Run validation tests against the specified model."""
        ...
```

## Naming Conventions

CRITICAL: Always use full descriptive names, never abbreviations:
- `self.repository` (not `self.repo`)
- `config_repository` (not `config_repo`)
- `validation_result` (not `val_result`)
- `dataset_path` (not `ds_path`)

This applies to:
- Class attributes
- Local variables
- Function parameters
- Module names

## Code Quality Standards

### Language and Format
- Python 3.13+ required
- Line length: 120 characters maximum
- Quote style: Double quotes
- Type annotations: Required for all functions/methods
- Async/await: Preferred for I/O operations

### Type Checking
- MyPy strict mode enabled
- Pyright strict mode enabled

### Code Organization
- Class member order: public, protected, private
- `__all__` declarations: Only in `__init__.py` files

## Git Workflow

### Branch Naming Convention
- `feat/*` - New features
- `fix/*` - Bug fixes
- `refactor/*` - Code refactoring
- `docs/*` - Documentation changes
- `test/*` - Test additions
- `chore/*` - Dependencies/tooling/maintenance

### Commit Message Format
- Prefix: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`
- NEVER include "Co-Authored-By: Claude" or AI attribution
- NEVER use emojis
- Keep concise and descriptive
- Focus on what changed and why

Example:
```
feat: add dataset validation for climate models
fix: correct path resolution in CLI
docs: update installation instructions
```

### PR Requirements
- All tests must pass (`make test`)
- All linting must pass (`make lint`)
- No code coverage decrease
- Descriptive PR title and body
- NO EMOJIS in PR titles or descriptions

## Testing Standards

### Test Execution
```bash
make test      # Run tests
make coverage  # With coverage reporting
make lint      # Ruff format, Ruff check, MyPy, Pyright
```

Always run `make lint` and `make test` after changes.

### Test File Organization
- pytest with async support enabled
- Test paths: `tests/`
- Name pattern: `test_*.py`

Example test structure:
```python
"""Tests for model validation."""

class TestModelValidator:
    """Tests for the ModelValidator class."""

    async def test_validate_valid_model(self) -> None:
        """Test validation of a valid chapkit model."""
        ...
```

## Dependency Management

Always use `uv` (not pip):
```bash
uv add <package>              # Runtime dependency
uv add --dev <package>        # Dev dependency
uv add <package>@latest       # Update specific
uv lock --upgrade             # Update all
```

Never manually edit `pyproject.toml` for dependencies.

## Linting Configuration (Ruff)

Active rules:
- E, W: PEP 8 errors and warnings
- F: PyFlakes (undefined names, unused imports)
- I: isort (import ordering)
- D: pydocstyle (documentation)

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
- `chapkit` - Model framework being tested
- `pydantic` - Data validation
- `typer` or `click` - CLI framework
- `pandas` - Data manipulation

Dev:
- `pytest`, `pytest-asyncio` - Testing
- `mypy`, `pyright` - Type checking
- `ruff` - Formatting/linting

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

## Additional Resources

- [chapkit documentation](https://github.com/climateview/chapkit)
- [servicekit documentation](https://github.com/winterop-com/servicekit)