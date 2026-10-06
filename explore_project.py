"""
PROJECT STRUCTURE AUDIT
=======================

Purpose
-------
Explore the research project located at:

D:\\47\\472\\New-Papers\\Enhancing Fidelity_Array\\Paper4-Under-Processing

The script attempts to identify:

1. Source code
2. Raw and processed datasets
3. Experimental results
4. Models and checkpoints
5. Figures and plots
6. Logs
7. Configuration files
8. Documentation
9. Likely experiment entry points
10. Important project directories

The script is READ-ONLY with respect to the research project.
It creates only the folder:

    _project_audit

and writes audit reports there.

It does NOT modify, move, rename, or delete existing project files.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import os
import re
import sys
import traceback


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(
    r"D:\47\472\New-Papers\Enhancing Fidelity_Array\Paper4-Under-Processing"
)

OUTPUT_DIR = ROOT / "_project_audit"


# ============================================================
# 2. DIRECTORIES TO IGNORE
# ============================================================

SKIP_DIRS = {
    ".git",
    ".idea",
    ".vscode",

    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".ipynb_checkpoints",

    "node_modules",

    "venv",
    ".venv",
    "env",
    ".env",
    "virtualenv",

    "site-packages",

    "_project_audit",
}


# ============================================================
# 3. FILE EXTENSION GROUPS
# ============================================================

CODE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".r",
    ".m",
    ".jl",
    ".java",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".sh",
    ".bat",
    ".cmd",
    ".ps1",
}

TABULAR_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".arff",
    ".sav",
}

SERIALIZED_DATA_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
}

DATA_EXTENSIONS = (
    TABULAR_EXTENSIONS |
    SERIALIZED_DATA_EXTENSIONS
)

MODEL_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".keras",
    ".onnx",
    ".joblib",
    ".pb",
    ".tflite",
}

FIGURE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".eps",
    ".tif",
    ".tiff",
    ".bmp",
}

DOCUMENT_EXTENSIONS = {
    ".doc",
    ".docx",
    ".tex",
    ".md",
    ".txt",
    ".rtf",
}

CONFIG_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
}

LOG_EXTENSIONS = {
    ".log",
}

ARCHIVE_EXTENSIONS = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}


# ============================================================
# 4. DIRECTORY / FILENAME HINTS
# ============================================================

DATA_HINTS = {
    "data",
    "dataset",
    "datasets",
    "raw",
    "processed",
    "preprocessed",
    "cleaned",
    "input",
    "inputs",
    "source data",
    "training data",
    "testing data",
}

RESULT_HINTS = {
    "result",
    "results",
    "output",
    "outputs",
    "metric",
    "metrics",
    "evaluation",
    "evaluations",
    "benchmark",
    "benchmarks",
    "ablation",
    "analysis",
    "performance",
    "prediction",
    "predictions",
    "scores",
    "summary",
    "summaries",
}

EXPERIMENT_HINTS = {
    "experiment",
    "experiments",
    "exp",
    "runs",
    "run",
    "trial",
    "trials",
}

MODEL_HINTS = {
    "model",
    "models",
    "checkpoint",
    "checkpoints",
    "weights",
    "saved model",
    "saved_model",
    "trained model",
    "trained",
}

FIGURE_HINTS = {
    "figure",
    "figures",
    "fig",
    "plots",
    "plot",
    "charts",
    "chart",
    "visualization",
    "visualizations",
    "graphs",
    "graph",
}

LOG_HINTS = {
    "log",
    "logs",
    "history",
    "trace",
}

CODE_HINTS = {
    "src",
    "source",
    "code",
    "scripts",
    "script",
    "pipeline",
    "pipelines",
    "implementation",
}

CONFIG_HINTS = {
    "config",
    "configs",
    "configuration",
    "settings",
    "parameters",
    "params",
}

DOCUMENT_HINTS = {
    "readme",
    "documentation",
    "docs",
    "paper",
    "manuscript",
    "report",
}


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def normalize(text):
    """
    Convert a string to normalized lowercase words.
    """

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower()
    ).strip()


def path_words(path):
    """
    Return normalized words from a path.
    """

    return set(
        normalize(path).split()
    )


def contains_hint(path, hints):
    """
    Check whether normalized path text contains any hint.

    Supports both single-word and multi-word hints.
    """

    text = normalize(path)
    words = set(text.split())

    for hint in hints:

        hint_normalized = normalize(hint)

        if " " in hint_normalized:

            if hint_normalized in text:
                return True

        else:

            if hint_normalized in words:
                return True

    return False


def human_size(size):
    """
    Convert bytes to human-readable size.
    """

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    value = float(size)

    for unit in units:

        if value < 1024 or unit == units[-1]:

            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{size} B"


def safe_relative(path):
    """
    Return path relative to project root where possible.
    """

    try:

        return path.relative_to(ROOT)

    except Exception:

        return path


def should_skip(path):
    """
    Determine whether a path contains an ignored directory.
    """

    for part in path.parts:

        if part.lower() in SKIP_DIRS:
            return True

    return False


def unique_list(items):
    """
    Preserve order while removing duplicates.
    """

    seen = set()
    output = []

    for item in items:

        if item not in seen:

            seen.add(item)
            output.append(item)

    return output


# ============================================================
# 6. FILE CLASSIFICATION
# ============================================================

def classify_file(path):
    """
    Assign one or more research-oriented categories.

    Possible categories:

        CODE
        DATA
        RESULT
        MODEL
        FIGURE
        LOG
        CONFIG
        DOCUMENT
        ARCHIVE
        OTHER
    """

    ext = path.suffix.lower()

    rel = safe_relative(path)

    categories = []

    # --------------------------------------------------------
    # CODE
    # --------------------------------------------------------

    if ext in CODE_EXTENSIONS:

        categories.append("CODE")

    # --------------------------------------------------------
    # MODEL / CHECKPOINT
    # --------------------------------------------------------

    if ext in MODEL_EXTENSIONS:

        categories.append("MODEL")

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    if ext in LOG_EXTENSIONS:

        categories.append("LOG")

    elif (
        ext in {".txt", ".csv", ".json"}
        and contains_hint(rel, LOG_HINTS)
    ):

        categories.append("LOG")

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    if ext in CONFIG_EXTENSIONS:

        categories.append("CONFIG")

    elif (
        ext in {".json", ".txt"}
        and contains_hint(rel, CONFIG_HINTS)
    ):

        categories.append("CONFIG")

    # --------------------------------------------------------
    # DOCUMENTATION
    # --------------------------------------------------------

    if ext in DOCUMENT_EXTENSIONS:

        categories.append("DOCUMENT")

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    if ext in FIGURE_EXTENSIONS:

        categories.append("FIGURE")

    # --------------------------------------------------------
    # ARCHIVES
    # --------------------------------------------------------

    if ext in ARCHIVE_EXTENSIONS:

        categories.append("ARCHIVE")

    # --------------------------------------------------------
    # STRUCTURED DATA / RESULTS
    # --------------------------------------------------------

    if ext in DATA_EXTENSIONS:

        looks_like_result = (
            contains_hint(rel, RESULT_HINTS)
            or contains_hint(rel, EXPERIMENT_HINTS)
        )

        looks_like_data = contains_hint(
            rel,
            DATA_HINTS
        )

        # If clearly under result/experiment folders,
        # classify primarily as RESULT.
        if looks_like_result:

            categories.append("RESULT")

        # If explicitly data-oriented, classify as DATA.
        if looks_like_data:

            categories.append("DATA")

        # If ambiguous structured file, classify conservatively.
        if not looks_like_result and not looks_like_data:

            # CSV/XLSX/Parquet/etc. are commonly datasets or result tables.
            # Keep as DATA_OR_RESULT for later manual inspection.
            categories.append("DATA_OR_RESULT")

    # --------------------------------------------------------
    # MODEL HINTS
    # --------------------------------------------------------

    if (
        contains_hint(rel, MODEL_HINTS)
        and ext in MODEL_EXTENSIONS
    ):

        categories.append("MODEL")

    # --------------------------------------------------------
    # CODE HINTS
    # --------------------------------------------------------

    if (
        contains_hint(rel, CODE_HINTS)
        and ext in CODE_EXTENSIONS
    ):

        categories.append("CODE")

    # --------------------------------------------------------
    # FIGURE HINTS
    # --------------------------------------------------------

    if (
        contains_hint(rel, FIGURE_HINTS)
        and ext in FIGURE_EXTENSIONS
    ):

        categories.append("FIGURE")

    # --------------------------------------------------------
    # NOTHING MATCHED
    # --------------------------------------------------------

    categories = unique_list(categories)

    if not categories:

        categories.append("OTHER")

    return categories


# ============================================================
# 7. PROJECT SCANNER
# ============================================================

def scan_project():
    """
    Recursively scan the project.

    Returns:
        records
        scan_errors
    """

    records = []
    scan_errors = []

    for current_root, dirs, files in os.walk(ROOT):

        current = Path(current_root)

        # ----------------------------------------------------
        # Prevent traversal into ignored folders
        # ----------------------------------------------------

        dirs[:] = [
            d
            for d in dirs
            if d.lower() not in SKIP_DIRS
        ]

        for filename in files:

            path = current / filename

            if should_skip(path):
                continue

            try:

                stat = path.stat()

            except Exception as exc:

                scan_errors.append({
                    "path": str(path),
                    "error": str(exc),
                })

                continue

            try:

                categories = classify_file(path)

            except Exception as exc:

                categories = ["OTHER"]

                scan_errors.append({
                    "path": str(path),
                    "error": (
                        "Classification error: "
                        + str(exc)
                    ),
                })

            records.append({

                "absolute_path":
                    str(path),

                "relative_path":
                    str(safe_relative(path)),

                "parent_folder":
                    str(safe_relative(path.parent)),

                "filename":
                    path.name,

                "extension":
                    path.suffix.lower(),

                "size_bytes":
                    stat.st_size,

                "size":
                    human_size(stat.st_size),

                "categories":
                    ";".join(categories),
            })

    return records, scan_errors


# ============================================================
# 8. DIRECTORY TREE
# ============================================================

def build_tree(max_depth=6):
    """
    Build a readable project tree.

    To avoid producing a gigantic report, only six directory
    levels are displayed.
    """

    lines = [
        ROOT.name + "/"
    ]

    def walk(folder, prefix="", depth=0):

        if depth >= max_depth:
            return

        try:

            entries = [
                x
                for x in folder.iterdir()
                if x.name.lower() not in SKIP_DIRS
            ]

            entries = sorted(
                entries,
                key=lambda x: (
                    not x.is_dir(),
                    x.name.lower()
                )
            )

        except Exception as exc:

            lines.append(
                prefix
                + "[Unable to read directory: "
                + str(exc)
                + "]"
            )

            return

        for index, entry in enumerate(entries):

            last = (
                index == len(entries) - 1
            )

            connector = (
                "└── "
                if last
                else "├── "
            )

            lines.append(
                prefix
                + connector
                + entry.name
            )

            if entry.is_dir():

                extension = (
                    "    "
                    if last
                    else "│   "
                )

                walk(
                    entry,
                    prefix + extension,
                    depth + 1
                )

    walk(ROOT)

    return "\n".join(lines)


# ============================================================
# 9. DIRECTORY ANALYSIS
# ============================================================

def analyze_directories(records):
    """
    Count artifact categories by directory.
    """

    categories = [
        "CODE",
        "DATA",
        "DATA_OR_RESULT",
        "RESULT",
        "MODEL",
        "FIGURE",
        "LOG",
        "CONFIG",
        "DOCUMENT",
        "ARCHIVE",
        "OTHER",
    ]

    folder_stats = defaultdict(
        lambda: {
            "files": 0,
            **{
                category: 0
                for category in categories
            }
        }
    )

    for record in records:

        folder = record["parent_folder"]

        folder_stats[folder]["files"] += 1

        record_categories = (
            record["categories"].split(";")
        )

        for category in record_categories:

            if category in folder_stats[folder]:

                folder_stats[
                    folder
                ][category] += 1

    return folder_stats


def top_folders(
    folder_stats,
    category,
    limit=30
):
    """
    Rank directories by number of matching files.
    """

    ranked = []

    for folder, stats in folder_stats.items():

        count = stats.get(
            category,
            0
        )

        if count > 0:

            ranked.append(
                (
                    count,
                    folder,
                    stats["files"]
                )
            )

    ranked.sort(
        key=lambda x: (
            x[0],
            x[2]
        ),
        reverse=True
    )

    return ranked[:limit]


# ============================================================
# 10. CODE ENTRY-POINT DETECTION
# ============================================================

def detect_code_roles(records):
    """
    Detect scripts likely to start experiments or pipelines.
    """

    code_files = []

    entry_patterns = [
        "main",
        "run",
        "train",
        "experiment",
        "pipeline",
        "evaluate",
        "evaluation",
        "benchmark",
        "ablation",
        "generate",
        "generator",
        "preprocess",
        "prepare",
        "test",
        "validate",
        "validation",
        "compare",
        "analysis",
    ]

    high_priority_patterns = [
        "main",
        "run_all",
        "run_experiment",
        "run_experiments",
        "train",
        "pipeline",
    ]

    for record in records:

        categories = (
            record["categories"].split(";")
        )

        if "CODE" not in categories:
            continue

        name = Path(
            record["filename"]
        ).stem.lower()

        normalized_name = normalize(name)

        likely_entry = any(
            pattern in normalized_name
            for pattern in entry_patterns
        )

        high_priority = any(
            pattern in normalized_name
            for pattern in high_priority_patterns
        )

        score = 0

        if likely_entry:
            score += 1

        if high_priority:
            score += 2

        rel = record["relative_path"]

        if contains_hint(
            rel,
            EXPERIMENT_HINTS
        ):
            score += 1

        if contains_hint(
            rel,
            CODE_HINTS
        ):
            score += 1

        code_files.append({

            "relative_path":
                record["relative_path"],

            "filename":
                record["filename"],

            "likely_entry_point":
                likely_entry,

            "high_priority":
                high_priority,

            "entry_score":
                score,

            "size":
                record["size"],
        })

    code_files.sort(
        key=lambda x: (
            x["entry_score"],
            x["relative_path"]
        ),
        reverse=True
    )

    return code_files


# ============================================================
# 11. LARGE DIRECTORY DETECTION
# ============================================================

def identify_large_folders(
    folder_stats,
    limit=30
):
    """
    Find directories containing unusually many files.
    Useful for identifying environments, caches, generated
    artifacts, or large experiment collections.
    """

    rows = []

    for folder, stats in folder_stats.items():

        rows.append({
            "folder": folder,
            "files": stats["files"],
        })

    rows.sort(
        key=lambda x: x["files"],
        reverse=True
    )

    return rows[:limit]


# ============================================================
# 12. SAFE CSV WRITER
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames
):
    """
    Write dictionaries to CSV.

    IMPORTANT:
    extrasaction='ignore' prevents the error:

        ValueError:
        dict contains fields not in fieldnames

    This allows full inventory dictionaries to be reused
    for category-specific reports containing only selected
    columns.
    """

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 13. CATEGORY FILTER
# ============================================================

def select_category(
    records,
    category
):
    """
    Return records belonging to a category.
    """

    selected = []

    for record in records:

        categories = (
            record["categories"].split(";")
        )

        if category in categories:

            selected.append(record)

    return selected


# ============================================================
# 14. CREATE REPORTS
# ============================================================

def create_reports(
    records,
    scan_errors
):
    """
    Generate all audit reports.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FULL FILE INVENTORY
    # --------------------------------------------------------

    write_csv(

        OUTPUT_DIR
        / "file_inventory.csv",

        records,

        [
            "relative_path",
            "absolute_path",
            "parent_folder",
            "filename",
            "extension",
            "size_bytes",
            "size",
            "categories",
        ]
    )

    # --------------------------------------------------------
    # CATEGORY REPORTS
    # --------------------------------------------------------

    categories_to_export = [
        "CODE",
        "DATA",
        "DATA_OR_RESULT",
        "RESULT",
        "MODEL",
        "FIGURE",
        "LOG",
        "CONFIG",
        "DOCUMENT",
        "ARCHIVE",
    ]

    category_file_names = {
        "CODE":
            "code_files.csv",

        "DATA":
            "data_files.csv",

        "DATA_OR_RESULT":
            "ambiguous_data_or_results.csv",

        "RESULT":
            "result_files.csv",

        "MODEL":
            "model_files.csv",

        "FIGURE":
            "figure_files.csv",

        "LOG":
            "log_files.csv",

        "CONFIG":
            "config_files.csv",

        "DOCUMENT":
            "document_files.csv",

        "ARCHIVE":
            "archive_files.csv",
    }

    for category in categories_to_export:

        selected = select_category(
            records,
            category
        )

        write_csv(

            OUTPUT_DIR
            / category_file_names[
                category
            ],

            selected,

            [
                "relative_path",
                "filename",
                "parent_folder",
                "extension",
                "size",
            ]
        )

    # --------------------------------------------------------
    # CODE ENTRY POINTS
    # --------------------------------------------------------

    code_roles = detect_code_roles(
        records
    )

    write_csv(

        OUTPUT_DIR
        / "code_entry_points.csv",

        code_roles,

        [
            "relative_path",
            "filename",
            "likely_entry_point",
            "high_priority",
            "entry_score",
            "size",
        ]
    )

    # --------------------------------------------------------
    # DIRECTORY ANALYSIS
    # --------------------------------------------------------

    folder_stats = analyze_directories(
        records
    )

    folder_rows = []

    for folder, stats in folder_stats.items():

        row = {
            "folder": folder
        }

        row.update(stats)

        folder_rows.append(row)

    folder_rows.sort(
        key=lambda x: x["files"],
        reverse=True
    )

    folder_fieldnames = [
        "folder",
        "files",
        "CODE",
        "DATA",
        "DATA_OR_RESULT",
        "RESULT",
        "MODEL",
        "FIGURE",
        "LOG",
        "CONFIG",
        "DOCUMENT",
        "ARCHIVE",
        "OTHER",
    ]

    write_csv(

        OUTPUT_DIR
        / "folder_analysis.csv",

        folder_rows,

        folder_fieldnames
    )

    # --------------------------------------------------------
    # LARGE FOLDERS
    # --------------------------------------------------------

    large_folders = identify_large_folders(
        folder_stats
    )

    write_csv(

        OUTPUT_DIR
        / "largest_folders.csv",

        large_folders,

        [
            "folder",
            "files",
        ]
    )

    # --------------------------------------------------------
    # PROJECT TREE
    # --------------------------------------------------------

    tree = build_tree(
        max_depth=6
    )

    (
        OUTPUT_DIR
        / "project_tree.txt"
    ).write_text(
        tree,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # SCAN ERRORS
    # --------------------------------------------------------

    write_csv(

        OUTPUT_DIR
        / "scan_errors.csv",

        scan_errors,

        [
            "path",
            "error",
        ]
    )

    # --------------------------------------------------------
    # EXTENSION COUNTS
    # --------------------------------------------------------

    extension_counts = Counter(

        record["extension"]
        if record["extension"]
        else "[no extension]"

        for record in records
    )

    extension_rows = [

        {
            "extension": ext,
            "count": count,
        }

        for ext, count
        in extension_counts.most_common()
    ]

    write_csv(

        OUTPUT_DIR
        / "extension_counts.csv",

        extension_rows,

        [
            "extension",
            "count",
        ]
    )

    # --------------------------------------------------------
    # CATEGORY COUNTS
    # --------------------------------------------------------

    category_counts = Counter()

    for record in records:

        for category in (
            record["categories"].split(";")
        ):

            category_counts[
                category
            ] += 1

    # --------------------------------------------------------
    # JSON SUMMARY
    # --------------------------------------------------------

    summary = {

        "project_root":
            str(ROOT),

        "audit_output":
            str(OUTPUT_DIR),

        "total_files":
            len(records),

        "scan_errors":
            len(scan_errors),

        "category_counts":
            dict(category_counts),

        "extension_counts":
            dict(extension_counts),

        "likely_locations": {},
    }

    for category in [
        "CODE",
        "DATA",
        "DATA_OR_RESULT",
        "RESULT",
        "MODEL",
        "FIGURE",
        "LOG",
        "CONFIG",
        "DOCUMENT",
    ]:

        summary[
            "likely_locations"
        ][category] = [

            {
                "folder": folder,
                "matching_files": count,
                "total_files": total,
            }

            for count, folder, total
            in top_folders(
                folder_stats,
                category,
                limit=30
            )
        ]

    summary[
        "largest_folders"
    ] = large_folders

    with (
        OUTPUT_DIR
        / "project_summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # HUMAN-READABLE REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "PROJECT STRUCTURE AUDIT"
    )

    report.append(
        "=" * 80
    )

    report.append(
        "\nProject root:"
    )

    report.append(
        str(ROOT)
    )

    report.append(
        "\nAudit output:"
    )

    report.append(
        str(OUTPUT_DIR)
    )

    report.append(
        f"\nTotal files discovered: "
        f"{len(records)}"
    )

    report.append(
        f"Scan errors: "
        f"{len(scan_errors)}"
    )

    # --------------------------------------------------------
    # Category summary
    # --------------------------------------------------------

    report.append(
        "\n\nFILE CATEGORIES"
    )

    report.append(
        "-" * 80
    )

    for category, count in (
        category_counts.most_common()
    ):

        report.append(
            f"{category:20s}: {count}"
        )

    # --------------------------------------------------------
    # Largest folders
    # --------------------------------------------------------

    report.append(
        "\n\nLARGEST DIRECTORIES"
    )

    report.append(
        "-" * 80
    )

    for item in large_folders:

        report.append(
            f"{item['files']:8d} files  "
            f"{item['folder']}"
        )

    # --------------------------------------------------------
    # Code locations
    # --------------------------------------------------------

    report.append(
        "\n\nLIKELY CODE LOCATIONS"
    )

    report.append(
        "-" * 80
    )

    code_locations = top_folders(
        folder_stats,
        "CODE",
        limit=30
    )

    if code_locations:

        for count, folder, total in code_locations:

            report.append(
                f"{folder}"
                f"  -> "
                f"{count} code files / "
                f"{total} total"
            )

    else:

        report.append(
            "No code locations detected."
        )

    # --------------------------------------------------------
    # Data locations
    # --------------------------------------------------------

    report.append(
        "\n\nLIKELY DATA LOCATIONS"
    )

    report.append(
        "-" * 80
    )

    data_locations = top_folders(
        folder_stats,
        "DATA",
        limit=30
    )

    if data_locations:

        for count, folder, total in data_locations:

            report.append(
                f"{folder}"
                f"  -> "
                f"{count} data files / "
                f"{total} total"
            )

    else:

        report.append(
            "No explicit data directories detected."
        )

    # --------------------------------------------------------
    # Ambiguous data/result files
    # --------------------------------------------------------

    report.append(
        "\n\nAMBIGUOUS DATA / RESULT LOCATIONS"
    )

    report.append(
        "-" * 80
    )

    ambiguous_locations = top_folders(
        folder_stats,
        "DATA_OR_RESULT",
        limit=30
    )

    if ambiguous_locations:

        for count, folder, total in ambiguous_locations:

            report.append(
                f"{folder}"
                f"  -> "
                f"{count} ambiguous files / "
                f"{total} total"
            )

    else:

        report.append(
            "No ambiguous structured-data "
            "locations detected."
        )

    # --------------------------------------------------------
    # Result locations
    # --------------------------------------------------------

    report.append(
        "\n\nLIKELY RESULT LOCATIONS"
    )

    report.append(
        "-" * 80
    )

    result_locations = top_folders(
        folder_stats,
        "RESULT",
        limit=30
    )

    if result_locations:

        for count, folder, total in result_locations:

            report.append(
                f"{folder}"
                f"  -> "
                f"{count} result files / "
                f"{total} total"
            )

    else:

        report.append(
            "No explicit result directories detected."
        )

    # --------------------------------------------------------
    # Model locations
    # --------------------------------------------------------

    report.append(
        "\n\nLIKELY MODEL / CHECKPOINT LOCATIONS"
    )

    report.append(
        "-" * 80
    )

    model_locations = top_folders(
        folder_stats,
        "MODEL",
        limit=30
    )

    if model_locations:

        for count, folder, total in model_locations:

            report.append(
                f"{folder}"
                f"  -> "
                f"{count} model files / "
                f"{total} total"
            )

    else:

        report.append(
            "No saved models/checkpoints detected."
        )

    # --------------------------------------------------------
    # Figure locations
    # --------------------------------------------------------

    report.append(
        "\n\nLIKELY FIGURE LOCATIONS"
    )

    report.append(
        "-" * 80
    )

    figure_locations = top_folders(
        folder_stats,
        "FIGURE",
        limit=30
    )

    if figure_locations:

        for count, folder, total in figure_locations:

            report.append(
                f"{folder}"
                f"  -> "
                f"{count} figure files / "
                f"{total} total"
            )

    else:

        report.append(
            "No figure locations detected."
        )

    # --------------------------------------------------------
    # Log locations
    # --------------------------------------------------------

    report.append(
        "\n\nLIKELY LOG LOCATIONS"
    )

    report.append(
        "-" * 80
    )

    log_locations = top_folders(
        folder_stats,
        "LOG",
        limit=30
    )

    if log_locations:

        for count, folder, total in log_locations:

            report.append(
                f"{folder}"
                f"  -> "
                f"{count} log files / "
                f"{total} total"
            )

    else:

        report.append(
            "No logs detected."
        )

    # --------------------------------------------------------
    # Entry points
    # --------------------------------------------------------

    report.append(
        "\n\nPOSSIBLE CODE ENTRY POINTS"
    )

    report.append(
        "-" * 80
    )

    likely_entries = [
        item
        for item in code_roles
        if item["likely_entry_point"]
    ]

    if likely_entries:

        for item in likely_entries[:100]:

            report.append(

                f"[score={item['entry_score']}] "
                f"{item['relative_path']}"
            )

    else:

        report.append(
            "No obvious experiment entry points detected."
        )

    # --------------------------------------------------------
    # Extensions
    # --------------------------------------------------------

    report.append(
        "\n\nMOST COMMON FILE EXTENSIONS"
    )

    report.append(
        "-" * 80
    )

    for ext, count in (
        extension_counts.most_common(40)
    ):

        report.append(
            f"{ext:20s}: {count}"
        )

    # --------------------------------------------------------
    # Output explanation
    # --------------------------------------------------------

    report.append(
        "\n\nGENERATED AUDIT FILES"
    )

    report.append(
        "-" * 80
    )

    generated_descriptions = [

        (
            "PROJECT_AUDIT_REPORT.txt",
            "Human-readable project summary"
        ),

        (
            "project_summary.json",
            "Machine-readable project summary"
        ),

        (
            "project_tree.txt",
            "Project directory tree"
        ),

        (
            "file_inventory.csv",
            "Complete file inventory"
        ),

        (
            "folder_analysis.csv",
            "Artifact counts by directory"
        ),

        (
            "largest_folders.csv",
            "Directories containing the most files"
        ),

        (
            "code_files.csv",
            "Detected source-code files"
        ),

        (
            "code_entry_points.csv",
            "Likely experiment/pipeline scripts"
        ),

        (
            "data_files.csv",
            "Files confidently classified as datasets"
        ),

        (
            "ambiguous_data_or_results.csv",
            "Structured files requiring manual classification"
        ),

        (
            "result_files.csv",
            "Likely experimental result files"
        ),

        (
            "model_files.csv",
            "Models/checkpoints"
        ),

        (
            "figure_files.csv",
            "Figures and plots"
        ),

        (
            "log_files.csv",
            "Execution logs"
        ),

        (
            "config_files.csv",
            "Configuration files"
        ),

        (
            "document_files.csv",
            "Documentation/manuscript files"
        ),

        (
            "archive_files.csv",
            "ZIP/RAR/etc. files"
        ),

        (
            "extension_counts.csv",
            "File extension statistics"
        ),

        (
            "scan_errors.csv",
            "Files that could not be inspected"
        ),
    ]

    for filename, description in (
        generated_descriptions
    ):

        report.append(
            f"{filename:35s} "
            f"{description}"
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    (
        OUTPUT_DIR
        / "PROJECT_AUDIT_REPORT.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    return summary


# ============================================================
# 15. CONSOLE SUMMARY
# ============================================================

def print_summary(summary):
    """
    Print concise summary to PowerShell/terminal.
    """

    print()

    print(
        "=" * 80
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Reports saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        f"Total files scanned: "
        f"{summary['total_files']}"
    )

    print(
        f"Scan errors: "
        f"{summary['scan_errors']}"
    )

    print()

    print(
        "Category counts:"
    )

    for category, count in sorted(
        summary[
            "category_counts"
        ].items()
    ):

        print(
            f"  {category:20s}: "
            f"{count}"
        )

    print()

    print(
        "Generated files:"
    )

    try:

        files = sorted(
            OUTPUT_DIR.iterdir(),
            key=lambda x: x.name.lower()
        )

        for file in files:

            if file.is_file():

                print(
                    f"  {file.name}"
                )

    except Exception:

        pass

    print()

    print(
        "=" * 80
    )

    print(
        "FILES TO PROVIDE FOR PAPER/CODE AUDIT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Preferred option:"
    )

    print(
        "  ZIP the entire _project_audit folder."
    )

    print()

    print(
        "If uploading individual files, provide:"
    )

    print(
        "  1. PROJECT_AUDIT_REPORT.txt"
    )

    print(
        "  2. project_summary.json"
    )

    print(
        "  3. project_tree.txt"
    )

    print(
        "  4. folder_analysis.csv"
    )

    print(
        "  5. largest_folders.csv"
    )

    print(
        "  6. code_files.csv"
    )

    print(
        "  7. code_entry_points.csv"
    )

    print(
        "  8. data_files.csv"
    )

    print(
        "  9. ambiguous_data_or_results.csv"
    )

    print(
        " 10. result_files.csv"
    )

    print(
        " 11. model_files.csv"
    )

    print(
        " 12. figure_files.csv"
    )

    print()

    print(
        "These reports can be used to determine:"
    )

    print(
        "  - which code belongs to the paper"
    )

    print(
        "  - where the original datasets are located"
    )

    print(
        "  - where processed/generated data are located"
    )

    print(
        "  - which scripts execute the experiments"
    )

    print(
        "  - where numerical results are stored"
    )

    print(
        "  - where figures/tables originate"
    )

    print(
        "  - whether trained models/checkpoints exist"
    )

    print(
        "  - whether the project is sufficient "
        "for reproducibility"
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "PROJECT STRUCTURE AUDIT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Project:"
    )

    print(
        ROOT
    )

    # --------------------------------------------------------
    # Validate root
    # --------------------------------------------------------

    if not ROOT.exists():

        raise FileNotFoundError(
            "\nProject folder does not exist:\n"
            + str(ROOT)
        )

    if not ROOT.is_dir():

        raise NotADirectoryError(
            "\nProject path is not a directory:\n"
            + str(ROOT)
        )

    print()

    print(
        "Scanning project..."
    )

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    records, scan_errors = (
        scan_project()
    )

    print(
        f"Found {len(records)} files."
    )

    if scan_errors:

        print(
            f"Encountered "
            f"{len(scan_errors)} "
            f"scan warnings/errors."
        )

    print()

    print(
        "Generating audit reports..."
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    summary = create_reports(
        records,
        scan_errors
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print_summary(
        summary
    )


# ============================================================
# 17. SAFE EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print(
            "Audit cancelled by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()

        print(
            "=" * 80
        )

        print(
            "AUDIT FAILED"
        )

        print(
            "=" * 80
        )

        print()

        print(
            f"Error: {exc}"
        )

        print()

        print(
            "Full traceback:"
        )

        traceback.print_exc()

        sys.exit(1)