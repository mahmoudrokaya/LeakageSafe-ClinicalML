from pathlib import Path

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project")
OUTPUT_FILE = PROJECT_ROOT / "project_structure.txt"

SKIP_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "venv",
    ".venv",
    "env",
    "node_modules",
}

def build_tree(path: Path, prefix=""):
    lines = []

    try:
        entries = sorted(
            [p for p in path.iterdir() if p.name not in SKIP_DIRS],
            key=lambda p: (p.is_file(), p.name.lower())
        )
    except PermissionError:
        return [f"{prefix}[Permission denied]"]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(build_tree(entry, prefix + extension))

    return lines


def main():
    if not PROJECT_ROOT.exists():
        print(f"ERROR: Project folder does not exist:\n{PROJECT_ROOT}")
        return

    tree_lines = [PROJECT_ROOT.name + "/"]
    tree_lines.extend(build_tree(PROJECT_ROOT))

    OUTPUT_FILE.write_text(
        "\n".join(tree_lines),
        encoding="utf-8"
    )

    print("\n".join(tree_lines))
    print("\n" + "=" * 80)
    print(f"Project structure saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()