from pathlib import Path

# ============================================================
# HFAGM_Project - Generate data folder structure
# ============================================================

ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project\data"
)

OUTPUT_FILE = ROOT / "data_structure.txt"


def build_tree(folder: Path) -> list[str]:
    """Recursively generate a tree representation of a folder."""
    lines = [f"{folder.name}/"]

    def walk(current: Path, prefix: str = ""):
        try:
            items = sorted(
                current.iterdir(),
                key=lambda x: (x.is_file(), x.name.lower())
            )
        except PermissionError:
            lines.append(prefix + "└── [Permission Denied]")
            return

        # Do not include the generated structure file itself
        items = [
            item for item in items
            if item.resolve() != OUTPUT_FILE.resolve()
        ]

        for index, item in enumerate(items):
            is_last = index == len(items) - 1
            connector = "└── " if is_last else "├── "

            lines.append(prefix + connector + item.name)

            if item.is_dir():
                extension = "    " if is_last else "│   "
                walk(item, prefix + extension)

    walk(folder)

    return lines


def main():
    if not ROOT.exists():
        print(f"ERROR: Folder does not exist:\n{ROOT}")
        return

    if not ROOT.is_dir():
        print(f"ERROR: Path is not a directory:\n{ROOT}")
        return

    tree_lines = build_tree(ROOT)

    header = [
        "HFAGM_Project - Data Folder Structure",
        "=" * 60,
        f"Root: {ROOT}",
        "=" * 60,
        ""
    ]

    content = "\n".join(header + tree_lines)

    OUTPUT_FILE.write_text(
        content,
        encoding="utf-8"
    )

    print(content)
    print("\n" + "=" * 60)
    print("Structure successfully saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()