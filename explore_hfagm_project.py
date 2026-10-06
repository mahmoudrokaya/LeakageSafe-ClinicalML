"""
HFAGM PROJECT EXPLORER
======================

Target:
D:\\47\\472\\New-Papers\\Enhancing Fidelity_Array\\Paper4-Under-Processing\\HFAGM_Project

Purpose:
- Explore the HFAGM project structure
- Identify code locations and likely entry points
- Identify raw, processed, and synthetic data
- Identify experiment/result folders
- Identify models/checkpoints
- Identify figures, logs, configs, and documents
- Detect important Python scripts
- Produce machine-readable and human-readable audit reports

READ-ONLY:
The script does not modify the research project.
It creates only:

    HFAGM_Project\\_hfagm_audit
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import os
import re
import sys
import traceback
from datetime import datetime


# ============================================================
# 1. PROJECT PATH
# ============================================================

ROOT = Path(
    r"D:\47\472\New-Papers\Enhancing Fidelity_Array"
    r"\Paper4-Under-Processing\HFAGM_Project"
)

OUTPUT = ROOT / "_hfagm_audit"


# ============================================================
# 2. FOLDERS TO IGNORE
# ============================================================

SKIP_DIRS = {
    "_hfagm_audit",
    "_project_audit",
    "Paper4_Code_Evidence",

    ".git",
    ".idea",
    ".vscode",

    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".ipynb_checkpoints",

    "venv",
    ".venv",
    "env",
    ".env",
    "virtualenv",

    "node_modules",
    "site-packages",
}


# ============================================================
# 3. EXTENSIONS
# ============================================================

CODE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".r",
    ".m",
    ".jl",
    ".sh",
    ".bat",
    ".cmd",
    ".ps1",
}

DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".json",
    ".jsonl",
    ".parquet",
    ".feather",
    ".arff",
    ".sav",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
}

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

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".svg",
    ".eps",
}

DOCUMENT_EXTENSIONS = {
    ".md",
    ".txt",
    ".doc",
    ".docx",
    ".tex",
    ".rtf",
    ".pdf",
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
# 4. PATH HINTS
# ============================================================

RAW_DATA_HINTS = {
    "raw",
    "original",
    "source",
}

PROCESSED_DATA_HINTS = {
    "processed",
    "preprocessed",
    "cleaned",
    "scaled",
    "balanced",
    "normalized",
    "embedding",
    "embeddings",
    "graph_features",
}

SYNTHETIC_DATA_HINTS = {
    "synthetic",
    "generated",
    "generation",
    "generations",
}

RESULT_HINTS = {
    "result",
    "results",
    "metric",
    "metrics",
    "evaluation",
    "evaluations",
    "benchmark",
    "benchmarks",
    "ablation",
    "performance",
    "prediction",
    "predictions",
    "summary",
    "summaries",
}

EXPERIMENT_HINTS = {
    "experiment",
    "experiments",
    "exp",
    "runs",
    "trial",
    "trials",
    "scenario",
}

MODEL_HINTS = {
    "model",
    "models",
    "checkpoint",
    "checkpoints",
    "weights",
    "saved_models",
    "encoder",
    "decoder",
    "classifier",
    "generator",
    "discriminator",
}

FIGURE_HINTS = {
    "figure",
    "figures",
    "fig",
    "plot",
    "plots",
    "chart",
    "charts",
    "visualization",
}

LOG_HINTS = {
    "log",
    "logs",
    "history",
    "trace",
}

CONFIG_HINTS = {
    "config",
    "configuration",
    "settings",
    "parameter",
    "parameters",
    "params",
}


# ============================================================
# 5. UTILITY FUNCTIONS
# ============================================================

def normalize(text):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower()
    ).strip()


def contains_hint(path, hints):

    text = normalize(path)

    words = set(text.split())

    for hint in hints:

        h = normalize(hint)

        if " " in h:

            if h in text:
                return True

        elif h in words:
            return True

    return False


def relative(path):

    try:
        return path.relative_to(ROOT)

    except Exception:
        return path


def human_size(size):

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    value = float(size)

    for unit in units:

        if value < 1024 or unit == units[-1]:

            return f"{value:.2f} {unit}"

        value /= 1024

    return str(size)


def unique(items):

    output = []
    seen = set()

    for item in items:

        if item not in seen:

            seen.add(item)
            output.append(item)

    return output


# ============================================================
# 6. CLASSIFY FILE
# ============================================================

def classify_file(path):

    ext = path.suffix.lower()

    rel = relative(path)

    categories = []

    # --------------------------------------------------------
    # CODE
    # --------------------------------------------------------

    if ext in CODE_EXTENSIONS:
        categories.append("CODE")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if ext in MODEL_EXTENSIONS:
        categories.append("MODEL")

    # .pkl can also be a model
    if (
        ext in {".pkl", ".pickle", ".joblib"}
        and contains_hint(rel, MODEL_HINTS)
    ):
        categories.append("MODEL")

    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    if (
        ext in DATA_EXTENSIONS
        and contains_hint(
            rel,
            RAW_DATA_HINTS
        )
    ):
        categories.append("RAW_DATA")

    # --------------------------------------------------------
    # PROCESSED DATA
    # --------------------------------------------------------

    if (
        ext in DATA_EXTENSIONS
        and contains_hint(
            rel,
            PROCESSED_DATA_HINTS
        )
    ):
        categories.append(
            "PROCESSED_DATA"
        )

    # --------------------------------------------------------
    # SYNTHETIC DATA
    # --------------------------------------------------------

    if (
        ext in DATA_EXTENSIONS
        and contains_hint(
            rel,
            SYNTHETIC_DATA_HINTS
        )
    ):
        categories.append(
            "SYNTHETIC_DATA"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if (
        ext in DATA_EXTENSIONS
        and (
            contains_hint(
                rel,
                RESULT_HINTS
            )
            or contains_hint(
                rel,
                EXPERIMENT_HINTS
            )
        )
    ):
        categories.append("RESULT")

    # --------------------------------------------------------
    # AMBIGUOUS DATA
    # --------------------------------------------------------

    if (
        ext in DATA_EXTENSIONS
        and not any(
            x in categories
            for x in [
                "RAW_DATA",
                "PROCESSED_DATA",
                "SYNTHETIC_DATA",
                "RESULT",
                "MODEL",
            ]
        )
    ):
        categories.append(
            "DATA_OR_RESULT"
        )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    if ext in IMAGE_EXTENSIONS:

        if (
            contains_hint(
                rel,
                FIGURE_HINTS
            )
            or contains_hint(
                rel,
                EXPERIMENT_HINTS
            )
            or contains_hint(
                rel,
                RESULT_HINTS
            )
        ):

            categories.append(
                "FIGURE"
            )

        else:

            categories.append(
                "IMAGE_DATA"
            )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    if ext in DOCUMENT_EXTENSIONS:

        categories.append(
            "DOCUMENT"
        )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    if ext in CONFIG_EXTENSIONS:

        categories.append(
            "CONFIG"
        )

    elif (
        ext == ".json"
        and contains_hint(
            rel,
            CONFIG_HINTS
        )
    ):

        categories.append(
            "CONFIG"
        )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    if ext in LOG_EXTENSIONS:

        categories.append(
            "LOG"
        )

    elif (
        ext in {
            ".txt",
            ".csv",
            ".json"
        }
        and contains_hint(
            rel,
            LOG_HINTS
        )
    ):

        categories.append(
            "LOG"
        )

    # --------------------------------------------------------
    # ARCHIVE
    # --------------------------------------------------------

    if ext in ARCHIVE_EXTENSIONS:

        categories.append(
            "ARCHIVE"
        )

    categories = unique(
        categories
    )

    if not categories:

        categories = ["OTHER"]

    return categories


# ============================================================
# 7. SCAN PROJECT
# ============================================================

def scan_project():

    records = []
    errors = []

    for current_root, dirs, files in os.walk(ROOT):

        current = Path(current_root)

        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIRS
        ]

        for filename in files:

            path = current / filename

            try:

                stat = path.stat()

                categories = (
                    classify_file(path)
                )

                records.append({

                    "relative_path":
                        str(relative(path)),

                    "absolute_path":
                        str(path),

                    "parent_folder":
                        str(relative(
                            path.parent
                        )),

                    "filename":
                        path.name,

                    "extension":
                        path.suffix.lower(),

                    "size_bytes":
                        stat.st_size,

                    "size":
                        human_size(
                            stat.st_size
                        ),

                    "categories":
                        ";".join(
                            categories
                        ),
                })

            except Exception as exc:

                errors.append({

                    "path":
                        str(path),

                    "error":
                        str(exc),
                })

    return records, errors


# ============================================================
# 8. PYTHON SCRIPT ANALYSIS
# ============================================================

def analyze_python_file(path):

    result = {

        "relative_path":
            str(relative(path)),

        "filename":
            path.name,

        "has_main":
            False,

        "imports_torch":
            False,

        "imports_tensorflow":
            False,

        "imports_sklearn":
            False,

        "mentions_gan":
            False,

        "mentions_vae":
            False,

        "mentions_diffusion":
            False,

        "mentions_contrastive":
            False,

        "mentions_gat":
            False,

        "mentions_graph":
            False,

        "mentions_fairness":
            False,

        "mentions_ensemble":
            False,

        "mentions_generator":
            False,

        "mentions_discriminator":
            False,

        "mentions_training":
            False,

        "mentions_evaluation":
            False,

        "line_count":
            0,
    }

    try:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        lower = text.lower()

        result["line_count"] = (
            len(text.splitlines())
        )

        result["has_main"] = (
            "__name__" in lower
            and "__main__" in lower
        )

        result["imports_torch"] = (
            "import torch" in lower
            or "from torch" in lower
        )

        result["imports_tensorflow"] = (
            "tensorflow" in lower
        )

        result["imports_sklearn"] = (
            "sklearn" in lower
        )

        result["mentions_gan"] = bool(
            re.search(
                r"\bgan\b",
                lower
            )
        )

        result["mentions_vae"] = bool(
            re.search(
                r"\bvae\b",
                lower
            )
        )

        result[
            "mentions_diffusion"
        ] = (
            "diffusion" in lower
        )

        result[
            "mentions_contrastive"
        ] = (
            "contrastive" in lower
        )

        result["mentions_gat"] = bool(
            re.search(
                r"\bgat\b",
                lower
            )
        )

        result["mentions_graph"] = (
            "graph" in lower
        )

        result["mentions_fairness"] = (
            "fairness" in lower
            or "statistical parity"
            in lower
            or "equal opportunity"
            in lower
        )

        result["mentions_ensemble"] = (
            "ensemble" in lower
        )

        result["mentions_generator"] = (
            "generator" in lower
        )

        result[
            "mentions_discriminator"
        ] = (
            "discriminator" in lower
        )

        result["mentions_training"] = (
            "train" in lower
        )

        result["mentions_evaluation"] = (
            "evaluate" in lower
            or "evaluation" in lower
            or "metric" in lower
        )

    except Exception:
        pass

    return result


def analyze_python_files(records):

    rows = []

    for record in records:

        categories = (
            record[
                "categories"
            ].split(";")
        )

        if "CODE" not in categories:
            continue

        path = ROOT / record[
            "relative_path"
        ]

        if (
            path.suffix.lower()
            == ".py"
        ):

            rows.append(
                analyze_python_file(
                    path
                )
            )

    return rows


# ============================================================
# 9. LIKELY ENTRY POINTS
# ============================================================

def detect_entry_points(
    python_analysis
):

    rows = []

    important_words = {
        "main": 5,
        "run": 4,
        "train": 4,
        "experiment": 4,
        "pipeline": 5,
        "evaluate": 3,
        "benchmark": 3,
        "ablation": 3,
        "generate": 3,
        "preprocess": 3,
        "split": 2,
    }

    for item in python_analysis:

        name = normalize(
            item["filename"]
        )

        score = 0
        reasons = []

        if item["has_main"]:

            score += 5

            reasons.append(
                "__main__ block"
            )

        for word, value in (
            important_words.items()
        ):

            if word in name:

                score += value

                reasons.append(
                    f"filename:{word}"
                )

        if item[
            "mentions_training"
        ]:

            score += 1

        if item[
            "mentions_evaluation"
        ]:

            score += 1

        if score > 0:

            rows.append({

                "relative_path":
                    item[
                        "relative_path"
                    ],

                "filename":
                    item[
                        "filename"
                    ],

                "score":
                    score,

                "reasons":
                    "; ".join(
                        reasons
                    ),
            })

    rows.sort(
        key=lambda x: (
            x["score"],
            x["relative_path"]
        ),
        reverse=True
    )

    return rows


# ============================================================
# 10. FOLDER STATISTICS
# ============================================================

def folder_statistics(records):

    stats = defaultdict(
        lambda: {
            "total_files": 0,
            "total_bytes": 0,
            "CODE": 0,
            "RAW_DATA": 0,
            "PROCESSED_DATA": 0,
            "SYNTHETIC_DATA": 0,
            "RESULT": 0,
            "MODEL": 0,
            "FIGURE": 0,
            "IMAGE_DATA": 0,
            "DOCUMENT": 0,
            "CONFIG": 0,
            "LOG": 0,
        }
    )

    for record in records:

        folder = record[
            "parent_folder"
        ]

        stats[
            folder
        ]["total_files"] += 1

        stats[
            folder
        ]["total_bytes"] += (
            record["size_bytes"]
        )

        for category in (
            record[
                "categories"
            ].split(";")
        ):

            if category in (
                stats[folder]
            ):

                stats[
                    folder
                ][category] += 1

    rows = []

    for folder, values in (
        stats.items()
    ):

        row = {
            "folder": folder
        }

        row.update(values)

        row["total_size"] = (
            human_size(
                values[
                    "total_bytes"
                ]
            )
        )

        rows.append(row)

    rows.sort(
        key=lambda x:
            x["total_files"],
        reverse=True
    )

    return rows


# ============================================================
# 11. BUILD DIRECTORY TREE
# ============================================================

def build_tree(
    max_depth=8,
    max_files_per_folder=30
):

    lines = [
        ROOT.name + "/"
    ]

    def walk(
        folder,
        prefix="",
        depth=0
    ):

        if depth >= max_depth:
            return

        try:

            entries = [
                x
                for x
                in folder.iterdir()
                if x.name
                not in SKIP_DIRS
            ]

        except Exception:
            return

        dirs = sorted(
            [
                x for x in entries
                if x.is_dir()
            ],
            key=lambda x:
                x.name.lower()
        )

        files = sorted(
            [
                x for x in entries
                if x.is_file()
            ],
            key=lambda x:
                x.name.lower()
        )

        displayed_files = files[
            :max_files_per_folder
        ]

        display_entries = (
            dirs
            + displayed_files
        )

        for index, entry in enumerate(
            display_entries
        ):

            last = (
                index
                == len(
                    display_entries
                ) - 1
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
                    prefix
                    + extension,
                    depth + 1
                )

        omitted = (
            len(files)
            - len(
                displayed_files
            )
        )

        if omitted > 0:

            lines.append(
                prefix
                + f"... "
                + f"{omitted} additional "
                + "files omitted"
            )

    walk(ROOT)

    return "\n".join(lines)


# ============================================================
# 12. CSV WRITER
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames
):

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 13. EXPORT CATEGORY
# ============================================================

def export_category(
    records,
    category,
    filename
):

    selected = [

        r
        for r in records

        if category in (
            r[
                "categories"
            ].split(";")
        )
    ]

    write_csv(

        OUTPUT / filename,

        selected,

        [
            "relative_path",
            "parent_folder",
            "filename",
            "extension",
            "size",
            "categories",
        ]
    )

    return len(selected)


# ============================================================
# 14. CREATE REPORTS
# ============================================================

def create_reports(
    records,
    errors
):

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Complete inventory
    # --------------------------------------------------------

    write_csv(

        OUTPUT
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
    # Categories
    # --------------------------------------------------------

    category_exports = {

        "CODE":
            "code_files.csv",

        "RAW_DATA":
            "raw_data_files.csv",

        "PROCESSED_DATA":
            "processed_data_files.csv",

        "SYNTHETIC_DATA":
            "synthetic_data_files.csv",

        "DATA_OR_RESULT":
            "ambiguous_data_files.csv",

        "RESULT":
            "result_files.csv",

        "MODEL":
            "model_files.csv",

        "FIGURE":
            "figure_files.csv",

        "IMAGE_DATA":
            "image_dataset_files.csv",

        "DOCUMENT":
            "document_files.csv",

        "CONFIG":
            "config_files.csv",

        "LOG":
            "log_files.csv",

        "ARCHIVE":
            "archive_files.csv",
    }

    category_counts = {}

    for category, filename in (
        category_exports.items()
    ):

        category_counts[
            category
        ] = export_category(
            records,
            category,
            filename
        )

    # --------------------------------------------------------
    # Python analysis
    # --------------------------------------------------------

    python_analysis = (
        analyze_python_files(
            records
        )
    )

    python_fields = [
        "relative_path",
        "filename",
        "line_count",
        "has_main",
        "imports_torch",
        "imports_tensorflow",
        "imports_sklearn",
        "mentions_gan",
        "mentions_vae",
        "mentions_diffusion",
        "mentions_contrastive",
        "mentions_gat",
        "mentions_graph",
        "mentions_fairness",
        "mentions_ensemble",
        "mentions_generator",
        "mentions_discriminator",
        "mentions_training",
        "mentions_evaluation",
    ]

    write_csv(

        OUTPUT
        / "python_code_analysis.csv",

        python_analysis,

        python_fields
    )

    # --------------------------------------------------------
    # Entry points
    # --------------------------------------------------------

    entry_points = (
        detect_entry_points(
            python_analysis
        )
    )

    write_csv(

        OUTPUT
        / "likely_entry_points.csv",

        entry_points,

        [
            "relative_path",
            "filename",
            "score",
            "reasons",
        ]
    )

    # --------------------------------------------------------
    # Folder statistics
    # --------------------------------------------------------

    folder_rows = (
        folder_statistics(
            records
        )
    )

    write_csv(

        OUTPUT
        / "folder_statistics.csv",

        folder_rows,

        [
            "folder",
            "total_files",
            "total_bytes",
            "total_size",
            "CODE",
            "RAW_DATA",
            "PROCESSED_DATA",
            "SYNTHETIC_DATA",
            "RESULT",
            "MODEL",
            "FIGURE",
            "IMAGE_DATA",
            "DOCUMENT",
            "CONFIG",
            "LOG",
        ]
    )

    # --------------------------------------------------------
    # Tree
    # --------------------------------------------------------

    tree = build_tree()

    (
        OUTPUT
        / "project_tree.txt"
    ).write_text(
        tree,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    write_csv(

        OUTPUT
        / "scan_errors.csv",

        errors,

        [
            "path",
            "error",
        ]
    )

    # --------------------------------------------------------
    # Extension counts
    # --------------------------------------------------------

    extension_counts = Counter(

        r["extension"]
        if r["extension"]
        else "[none]"

        for r in records
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

        OUTPUT
        / "extension_counts.csv",

        extension_rows,

        [
            "extension",
            "count",
        ]
    )

    # --------------------------------------------------------
    # Framework signature
    # --------------------------------------------------------

    signature_fields = [
        "mentions_gan",
        "mentions_vae",
        "mentions_diffusion",
        "mentions_contrastive",
        "mentions_gat",
        "mentions_graph",
        "mentions_fairness",
        "mentions_ensemble",
        "mentions_generator",
        "mentions_discriminator",
    ]

    framework_signature = {}

    for field in signature_fields:

        matching = [

            x["relative_path"]

            for x in python_analysis

            if x[field]
        ]

        framework_signature[
            field
        ] = {

            "file_count":
                len(matching),

            "files":
                matching,
        }

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    total_bytes = sum(
        r["size_bytes"]
        for r in records
    )

    summary = {

        "created":
            datetime.now().isoformat(),

        "project_root":
            str(ROOT),

        "total_files":
            len(records),

        "total_bytes":
            total_bytes,

        "total_size":
            human_size(
                total_bytes
            ),

        "scan_errors":
            len(errors),

        "category_counts":
            category_counts,

        "extension_counts":
            dict(
                extension_counts
            ),

        "python_files":
            len(
                python_analysis
            ),

        "likely_entry_points":
            entry_points,

        "framework_signature":
            framework_signature,
    }

    (
        OUTPUT
        / "project_summary.json"
    ).write_text(

        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    report = []

    report.append(
        "HFAGM PROJECT AUDIT"
    )

    report.append(
        "=" * 80
    )

    report.append(
        f"\nProject:\n{ROOT}"
    )

    report.append(
        f"\nTotal files: "
        f"{len(records)}"
    )

    report.append(
        f"\nProject size: "
        f"{human_size(total_bytes)}"
    )

    report.append(
        f"\nPython files: "
        f"{len(python_analysis)}"
    )

    report.append(
        f"\nScan errors: "
        f"{len(errors)}"
    )

    report.append(
        "\n\nCATEGORY COUNTS"
    )

    report.append(
        "-" * 80
    )

    for category, count in (
        category_counts.items()
    ):

        report.append(
            f"{category:22s}: "
            f"{count}"
        )

    report.append(
        "\n\nLIKELY EXECUTION / "
        "EXPERIMENT ENTRY POINTS"
    )

    report.append(
        "-" * 80
    )

    for item in entry_points[
        :50
    ]:

        report.append(

            f"[score={item['score']}] "
            f"{item['relative_path']} "
            f"({item['reasons']})"
        )

    report.append(
        "\n\nFRAMEWORK SIGNATURE"
    )

    report.append(
        "-" * 80
    )

    labels = {

        "mentions_gan":
            "GAN",

        "mentions_vae":
            "VAE",

        "mentions_diffusion":
            "Diffusion",

        "mentions_contrastive":
            "Contrastive learning",

        "mentions_gat":
            "GAT",

        "mentions_graph":
            "Graph processing",

        "mentions_fairness":
            "Fairness",

        "mentions_ensemble":
            "Ensemble",

        "mentions_generator":
            "Generator",

        "mentions_discriminator":
            "Discriminator",
    }

    for field in signature_fields:

        info = (
            framework_signature[
                field
            ]
        )

        report.append(
            f"{labels[field]:25s}: "
            f"{info['file_count']} "
            f"Python file(s)"
        )

    report.append(
        "\n\nLARGEST DIRECTORIES"
    )

    report.append(
        "-" * 80
    )

    for row in folder_rows[
        :30
    ]:

        report.append(

            f"{row['total_files']:8d} files | "
            f"{row['total_size']:>12s} | "
            f"{row['folder']}"
        )

    report.append(
        "\n\nOUTPUT FILES"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "project_summary.json"
    )

    report.append(
        "project_tree.txt"
    )

    report.append(
        "file_inventory.csv"
    )

    report.append(
        "folder_statistics.csv"
    )

    report.append(
        "code_files.csv"
    )

    report.append(
        "python_code_analysis.csv"
    )

    report.append(
        "likely_entry_points.csv"
    )

    report.append(
        "raw_data_files.csv"
    )

    report.append(
        "processed_data_files.csv"
    )

    report.append(
        "synthetic_data_files.csv"
    )

    report.append(
        "result_files.csv"
    )

    report.append(
        "model_files.csv"
    )

    report.append(
        "figure_files.csv"
    )

    report.append(
        "image_dataset_files.csv"
    )

    report.append(
        "config_files.csv"
    )

    report.append(
        "document_files.csv"
    )

    (
        OUTPUT
        / "HFAGM_AUDIT_REPORT.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    return summary


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "HFAGM PROJECT EXPLORER"
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
    # Validate path
    # --------------------------------------------------------

    if not ROOT.exists():

        raise FileNotFoundError(
            "\nHFAGM project folder "
            "does not exist:\n"
            + str(ROOT)
        )

    if not ROOT.is_dir():

        raise NotADirectoryError(
            "\nThe specified path "
            "is not a folder:\n"
            + str(ROOT)
        )

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    print()

    print(
        "Scanning HFAGM project..."
    )

    records, errors = (
        scan_project()
    )

    print(
        f"Found {len(records)} files."
    )

    print(
        f"Scan errors: {len(errors)}"
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    print()

    print(
        "Analyzing code, data, "
        "experiments and results..."
    )

    summary = create_reports(
        records,
        errors
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "HFAGM AUDIT COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Total files: "
        f"{summary['total_files']}"
    )

    print(
        f"Project size: "
        f"{summary['total_size']}"
    )

    print(
        f"Python files: "
        f"{summary['python_files']}"
    )

    print()

    print(
        "Artifact categories:"
    )

    for category, count in (
        summary[
            "category_counts"
        ].items()
    ):

        print(
            f"  {category:22s}: "
            f"{count}"
        )

    print()

    print(
        "Framework signature:"
    )

    for field, info in (
        summary[
            "framework_signature"
        ].items()
    ):

        label = field.replace(
            "mentions_",
            ""
        )

        print(
            f"  {label:22s}: "
            f"{info['file_count']} "
            f"Python file(s)"
        )

    print()

    print(
        "Top likely entry points:"
    )

    for item in (
        summary[
            "likely_entry_points"
        ][:15]
    ):

        print(
            f"  [{item['score']:2d}] "
            f"{item['relative_path']}"
        )

    print()

    print(
        "Reports saved to:"
    )

    print(
        OUTPUT
    )

    print()

    print(
        "For detailed analysis, ZIP and upload "
        "the entire _hfagm_audit folder."
    )


# ============================================================
# 16. SAFE EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print(
            "Audit cancelled."
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

        traceback.print_exc()

        sys.exit(1)