"""Main CLI entry point for chap-python-sdk."""

import argparse
import re
import sys
from importlib import resources
from pathlib import Path


def to_class_name(project_name: str) -> str:
    """Convert project name to PascalCase class name."""
    # Replace hyphens and underscores with spaces, then title case
    words = re.sub(r"[-_]+", " ", project_name).split()
    return "".join(word.capitalize() for word in words)


def to_underscore_name(project_name: str) -> str:
    """Convert project name to snake_case module name."""
    # Replace hyphens with underscores
    return re.sub(r"-+", "_", project_name).lower()


def get_template_content(filename: str) -> str:
    """Read template file content from package resources."""
    template_files = resources.files("chap_python_sdk.template")
    template_path = template_files.joinpath(filename)
    return template_path.read_text()


def render_template(content: str, project_name: str) -> str:
    """Replace template placeholders with actual values."""
    class_name = to_class_name(project_name)
    underscore_name = to_underscore_name(project_name)

    return (
        content.replace("{{project_name}}", project_name)
        .replace("{{project_name_class}}", class_name)
        .replace("{{project_name_underscore}}", underscore_name)
    )


def init_project(project_name: str, output_dir: Path | None = None) -> None:
    """Initialize a new chapkit model project."""
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / project_name
    else:
        output_dir = output_dir / project_name

    underscore_name = to_underscore_name(project_name)

    # Check if directory already exists
    if output_dir.exists():
        print(f"Error: Directory '{output_dir}' already exists.")
        sys.exit(1)

    # Create directory structure
    src_dir = output_dir / "src" / underscore_name
    tests_dir = output_dir / "tests"

    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    # Template files and their destinations
    templates = [
        ("pyproject.toml.template", output_dir / "pyproject.toml"),
        ("README.md.template", output_dir / "README.md"),
        ("model.py.template", src_dir / "model.py"),
        ("__init__.py.template", src_dir / "__init__.py"),
        ("test_model.py.template", tests_dir / "test_model.py"),
    ]

    # Create files from templates
    for template_name, dest_path in templates:
        content = get_template_content(template_name)
        rendered = render_template(content, project_name)
        dest_path.write_text(rendered)
        print(f"  Created {dest_path.relative_to(output_dir)}")

    # Create empty tests/__init__.py
    (tests_dir / "__init__.py").write_text('"""Tests for the model."""\n')
    print("  Created tests/__init__.py")

    print(f"\nProject '{project_name}' created successfully at {output_dir}")
    print("\nNext steps:")
    print(f"  cd {project_name}")
    print("  uv sync")
    print("  pytest")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="chap-sdk",
        description="CHAP Python SDK - Tools for building and testing chapkit models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Create a new chapkit model project",
        description="Initialize a new chapkit model project with template files",
    )
    init_parser.add_argument(
        "project_name",
        help="Name of the project (e.g., 'my-model')",
    )
    init_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory (default: current directory)",
    )

    args = parser.parse_args()

    if args.command == "init":
        init_project(args.project_name, args.output_dir)
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
