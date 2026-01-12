.PHONY: lint format check test coverage clean install dev

# Install dependencies
install:
	uv sync

# Install with dev dependencies
dev:
	uv sync --all-extras

# Run all linting and type checking
lint: format check

# Format code with ruff
format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Run type checkers (without fixing)
check:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run mypy src tests
	uv run pyright src tests

# Run tests (includes unit tests and documentation tests)
test:
	uv run pytest -v

# Run tests with coverage
coverage:
	uv run pytest -v --cov=src/chap_python_sdk --cov-report=term-missing --cov-report=html

# Clean build artifacts
clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
