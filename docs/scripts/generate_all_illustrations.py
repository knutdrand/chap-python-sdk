"""Generate all documentation illustrations."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import generate_bootstrap_illustrations
import generate_residual_bootstrap_illustrations
import generate_quantile_regression_illustrations


def main() -> None:
    """Generate all documentation illustrations."""
    print("=" * 70)
    print("Generating All Documentation Illustrations")
    print("=" * 70)
    print()

    print("PART 1: General Bootstrapping Illustrations")
    print("-" * 70)
    generate_bootstrap_illustrations.main()
    print()

    print("PART 2: Residual Bootstrapping Illustrations")
    print("-" * 70)
    generate_residual_bootstrap_illustrations.main()
    print()

    print("PART 3: Quantile Regression Illustrations")
    print("-" * 70)
    generate_quantile_regression_illustrations.main()
    print()

    print("=" * 70)
    print("All documentation illustrations generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
