"""Main CLI entry point for chap-python-sdk."""

import re
import sys
from importlib import resources
from pathlib import Path
from typing import Annotated

import cyclopts

app = cyclopts.App(
    name="chap-sdk",
    help="CHAP Python SDK - Tools for building and testing chapkit models",
)


def to_class_name(project_name: str) -> str:
    """Convert project name to PascalCase class name."""
    words = re.sub(r"[-_]+", " ", project_name).split()
    return "".join(word.capitalize() for word in words)


def to_underscore_name(project_name: str) -> str:
    """Convert project name to snake_case module name."""
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


@app.command
def init(
    project_name: Annotated[str, cyclopts.Parameter(help="Name of the project (e.g., 'my-model')")],
    output_dir: Annotated[Path | None, cyclopts.Parameter(name=["-o", "--output-dir"], help="Output directory")] = None,
) -> None:
    """Initialize a new chapkit model project with template files."""
    if output_dir is None:
        target_dir = Path.cwd() / project_name
    else:
        target_dir = output_dir / project_name

    underscore_name = to_underscore_name(project_name)

    if target_dir.exists():
        print(f"Error: Directory '{target_dir}' already exists.")
        sys.exit(1)

    src_dir = target_dir / "src" / underscore_name
    tests_dir = target_dir / "tests"

    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    templates = [
        ("pyproject.toml.template", target_dir / "pyproject.toml"),
        ("README.md.template", target_dir / "README.md"),
        ("MLproject.template", target_dir / "MLproject"),
        ("model.py.template", src_dir / "model.py"),
        ("cli.py.template", src_dir / "cli.py"),
        ("__init__.py.template", src_dir / "__init__.py"),
        ("test_model.py.template", tests_dir / "test_model.py"),
    ]

    for template_name, dest_path in templates:
        content = get_template_content(template_name)
        rendered = render_template(content, project_name)
        dest_path.write_text(rendered)
        print(f"  Created {dest_path.relative_to(target_dir)}")

    (tests_dir / "__init__.py").write_text('"""Tests for the model."""\n')
    print("  Created tests/__init__.py")

    print(f"\nProject '{project_name}' created successfully at {target_dir}")
    print("\nNext steps:")
    print(f"  cd {project_name}")
    print("  uv sync")
    print("  pytest")


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
