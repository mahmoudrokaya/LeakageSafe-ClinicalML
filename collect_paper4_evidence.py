"""
COLLECT PAPER4 CODE / DATA / RESULTS EVIDENCE
=============================================

Project:
D:\\47\\472\\New-Papers\\Enhancing Fidelity_Array\\Paper4-Under-Processing

Purpose
-------
Create a compact evidence package containing the files needed to audit:

    manuscript -> code -> data -> experiments -> results

The script collects:

1. All Python/source code
2. Configuration files
3. Experiment metrics
4. Experiment run configurations
5. Saved models/checkpoints
6. Processed clinical/tabular data
7. Raw clinical CSV/XLSX files
8. NumPy artifacts such as embeddings/graph matrices
9. Fairness outputs
10. Experiment figures
11. Relevant project documentation
12. Existing audit reports
13. A manifest of the large ArSL image dataset
14. Small representative ArSL samples

The script DOES NOT:
- modify original files
- delete files
- move files
- copy the entire large image dataset

It creates:
    Paper4_Code_Evidence

and:
    Paper4_Code_Evidence.zip
"""

from pathlib import Path
from collections import Counter
import shutil
import csv
import json
import hashlib
import os
import sys
import traceback
from datetime import datetime


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(
    r"D:\47\472\New-Papers\Enhancing Fidelity_Array\Paper4-Under-Processing"
)

OUTPUT = ROOT / "Paper4_Code_Evidence"

ZIP_BASE = ROOT / "Paper4_Code_Evidence"


# ============================================================
# 2. SETTINGS
# ============================================================

# Number of representative files copied from each large
# image/label directory.
ARSL_SAMPLE_COUNT = 10

# Avoid accidentally copying gigantic individual files.
# Increase this if necessary.
MAX_GENERAL_FILE_MB = 200

MAX_GENERAL_FILE_BYTES = (
    MAX_GENERAL_FILE_MB * 1024 * 1024
)


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
    ".ps1",
}

CONFIG_EXTENSIONS = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}

DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".npy",
    ".npz",
    ".parquet",
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
    ".pkl",
}

FIGURE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".eps",
}

DOCUMENT_EXTENSIONS = {
    ".md",
    ".txt",
    ".docx",
    ".doc",
    ".tex",
    ".pdf",
}


# ============================================================
# 4. FOLDERS TO COMPLETELY EXCLUDE
# ============================================================

EXCLUDE_DIRS = {
    "Paper4_Code_Evidence",
    "_project_audit",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "site-packages",
}


# ============================================================
# 5. LARGE DATASET IDENTIFIERS
# ============================================================

# These directories should NOT be copied wholesale.
LARGE_DATASET_HINTS = {
    "arsl",
    "arsl21l",
}


# ============================================================
# 6. COLLECTION RECORDS
# ============================================================

manifest = []
errors = []
skipped = []


# ============================================================
# 7. HELPERS
# ============================================================

def relative(path):
    try:
        return path.relative_to(ROOT)
    except Exception:
        return path


def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024

    return str(size)


def sha256_file(path):
    """
    Calculate SHA-256 checksum.
    """

    digest = hashlib.sha256()

    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    except Exception:
        return ""


def is_excluded(path):
    return any(
        part in EXCLUDE_DIRS
        for part in path.parts
    )


def is_large_dataset_path(path):
    parts = {
        part.lower()
        for part in path.parts
    }

    return bool(
        parts.intersection(
            LARGE_DATASET_HINTS
        )
    )


def copy_file(source, category):
    """
    Copy file while preserving project-relative structure.
    """

    try:
        rel = relative(source)

        destination = (
            OUTPUT
            / "project_files"
            / rel
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        size = source.stat().st_size

        if size > MAX_GENERAL_FILE_BYTES:
            skipped.append({
                "relative_path": str(rel),
                "reason": (
                    f"Individual file exceeds "
                    f"{MAX_GENERAL_FILE_MB} MB"
                ),
                "size": human_size(size),
            })

            return False

        shutil.copy2(
            source,
            destination
        )

        manifest.append({
            "category": category,
            "relative_path": str(rel),
            "copied_to": str(
                destination.relative_to(OUTPUT)
            ),
            "size_bytes": size,
            "size": human_size(size),
            "sha256": sha256_file(source),
        })

        return True

    except Exception as exc:
        errors.append({
            "path": str(source),
            "error": str(exc),
        })

        return False


# ============================================================
# 8. CLASSIFY RELEVANT FILES
# ============================================================

def classify_for_collection(path):
    """
    Determine whether a file should be included.

    Returns category or None.
    """

    ext = path.suffix.lower()

    rel_text = str(
        relative(path)
    ).lower()

    filename = path.name.lower()

    # --------------------------------------------------------
    # Source code
    # --------------------------------------------------------

    if ext in CODE_EXTENSIONS:
        return "CODE"

    # --------------------------------------------------------
    # Models/checkpoints
    # --------------------------------------------------------

    if ext in MODEL_EXTENSIONS:

        if any(
            term in rel_text
            for term in [
                "model",
                "models",
                "checkpoint",
                "saved_models",
                "experiment",
                "experiments",
                "encoder",
                "classifier",
            ]
        ):
            return "MODEL"

    # --------------------------------------------------------
    # Experiment results
    # --------------------------------------------------------

    if ext in DATA_EXTENSIONS:

        if any(
            term in rel_text
            for term in [
                "experiment",
                "experiments",
                "result",
                "results",
                "metric",
                "metrics",
                "evaluation",
                "ablation",
                "fairness",
            ]
        ):
            return "RESULT"

    # --------------------------------------------------------
    # Clinical / processed data
    # --------------------------------------------------------

    if ext in DATA_EXTENSIONS:

        if any(
            term in rel_text
            for term in [
                "covid",
                "clinical",
                "preprocessed",
                "processed",
                "scaled",
                "balanced",
                "embedding",
                "graph_features",
                "adj_matrix",
                "x_train",
                "x_test",
                "y_train",
                "y_test",
            ]
        ):
            return "DATA"

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    if ext in CONFIG_EXTENSIONS:

        if any(
            term in filename
            for term in [
                "config",
                "setting",
                "parameter",
                "params",
            ]
        ):
            return "CONFIG"

    # --------------------------------------------------------
    # Figures/reports generated by experiments
    # --------------------------------------------------------

    if ext in FIGURE_EXTENSIONS:

        if any(
            term in rel_text
            for term in [
                "experiment",
                "experiments",
                "figure",
                "figures",
                "plot",
                "fairness",
                "report",
                "result",
            ]
        ):
            return "FIGURE_OR_REPORT"

    # --------------------------------------------------------
    # Relevant documentation
    # --------------------------------------------------------

    if ext in DOCUMENT_EXTENSIONS:

        if any(
            term in filename
            for term in [
                "readme",
                "hfagm",
                "framework",
                "architecture",
                "mathematical",
                "method",
                "experiment",
                "paper",
                "manuscript",
                "theoretical",
                "contribution",
            ]
        ):
            return "DOCUMENT"

    return None


# ============================================================
# 9. COLLECT MAIN PROJECT FILES
# ============================================================

def collect_project_files():

    print()
    print("Collecting relevant project files...")

    counts = Counter()

    for current_root, dirs, files in os.walk(ROOT):

        current = Path(current_root)

        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDE_DIRS
        ]

        for filename in files:

            path = current / filename

            if is_excluded(path):
                continue

            # Do not copy huge ArSL image dataset here.
            if is_large_dataset_path(path):
                continue

            category = classify_for_collection(
                path
            )

            if category is None:
                continue

            if copy_file(
                path,
                category
            ):
                counts[category] += 1

    return counts


# ============================================================
# 10. COLLECT EXISTING AUDIT
# ============================================================

def collect_existing_audit():

    audit_dir = ROOT / "_project_audit"

    if not audit_dir.exists():
        return 0

    destination = (
        OUTPUT
        / "existing_project_audit"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    for path in audit_dir.iterdir():

        if not path.is_file():
            continue

        try:
            shutil.copy2(
                path,
                destination / path.name
            )

            count += 1

        except Exception as exc:

            errors.append({
                "path": str(path),
                "error": str(exc),
            })

    return count


# ============================================================
# 11. LARGE DATASET MANIFEST
# ============================================================

def create_large_dataset_manifest():
    """
    Inventory ArSL-related files without copying the entire
    dataset.
    """

    print()
    print(
        "Creating ArSL dataset manifest "
        "(without copying full dataset)..."
    )

    rows = []

    extension_counts = Counter()
    folder_counts = Counter()

    for current_root, dirs, files in os.walk(ROOT):

        current = Path(current_root)

        if not is_large_dataset_path(current):
            continue

        for filename in files:

            path = current / filename

            try:
                size = path.stat().st_size
            except Exception:
                continue

            rel = relative(path)

            rows.append({
                "relative_path": str(rel),
                "folder": str(relative(path.parent)),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": size,
            })

            extension_counts[
                path.suffix.lower()
            ] += 1

            folder_counts[
                str(relative(path.parent))
            ] += 1

    report_dir = (
        OUTPUT
        / "large_dataset_manifest"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Full manifest
    manifest_path = (
        report_dir
        / "arsl_file_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "folder",
                "filename",
                "extension",
                "size_bytes",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # Folder summary
    folder_summary = (
        report_dir
        / "arsl_folder_summary.csv"
    )

    with folder_summary.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "folder",
            "file_count",
        ])

        for folder, count in (
            folder_counts.most_common()
        ):
            writer.writerow([
                folder,
                count,
            ])

    # JSON summary
    summary = {
        "total_files": len(rows),
        "extension_counts":
            dict(extension_counts),
        "folder_counts":
            dict(folder_counts),
    }

    (
        report_dir
        / "arsl_dataset_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return summary


# ============================================================
# 12. COPY REPRESENTATIVE ArSL SAMPLES
# ============================================================

def collect_arsl_samples():
    """
    Copy only a few representative files from each ArSL
    directory.
    """

    print()
    print(
        "Collecting representative ArSL samples..."
    )

    sample_root = (
        OUTPUT
        / "large_dataset_samples"
    )

    copied = 0

    for current_root, dirs, files in os.walk(ROOT):

        current = Path(current_root)

        if not is_large_dataset_path(current):
            continue

        candidate_files = sorted(
            [
                current / name
                for name in files
            ],
            key=lambda p: p.name.lower()
        )

        if not candidate_files:
            continue

        selected = (
            candidate_files[
                :ARSL_SAMPLE_COUNT
            ]
        )

        rel_folder = relative(current)

        destination_folder = (
            sample_root
            / rel_folder
        )

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for source in selected:

            try:

                shutil.copy2(
                    source,
                    destination_folder
                    / source.name
                )

                copied += 1

            except Exception as exc:

                errors.append({
                    "path": str(source),
                    "error": str(exc),
                })

    return copied


# ============================================================
# 13. WRITE COLLECTION MANIFEST
# ============================================================

def write_collection_manifest():

    path = (
        OUTPUT
        / "COLLECTED_FILES_MANIFEST.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "category",
            "relative_path",
            "copied_to",
            "size_bytes",
            "size",
            "sha256",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(manifest)


# ============================================================
# 14. WRITE SKIPPED FILE REPORT
# ============================================================

def write_skipped_report():

    path = (
        OUTPUT
        / "SKIPPED_FILES.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "reason",
                "size",
            ],
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(skipped)


# ============================================================
# 15. WRITE ERROR REPORT
# ============================================================

def write_error_report():

    path = (
        OUTPUT
        / "COLLECTION_ERRORS.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "error",
            ],
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(errors)


# ============================================================
# 16. PACKAGE SUMMARY
# ============================================================

def write_summary(
    counts,
    audit_count,
    arsl_summary,
    sample_count
):

    total_size = sum(
        item["size_bytes"]
        for item in manifest
    )

    summary = {
        "created":
            datetime.now().isoformat(),

        "project_root":
            str(ROOT),

        "evidence_folder":
            str(OUTPUT),

        "files_collected":
            len(manifest),

        "collected_size_bytes":
            total_size,

        "collected_size":
            human_size(total_size),

        "category_counts":
            dict(counts),

        "existing_audit_files":
            audit_count,

        "large_arsl_dataset": {
            "full_dataset_copied": False,
            "manifest_created": True,
            "representative_samples":
                sample_count,
            "dataset_summary":
                arsl_summary,
        },

        "skipped_files":
            len(skipped),

        "collection_errors":
            len(errors),
    }

    (
        OUTPUT
        / "EVIDENCE_PACKAGE_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return summary


# ============================================================
# 17. README
# ============================================================

def write_readme(summary):

    text = f"""
PAPER4 CODE EVIDENCE PACKAGE
============================

Original project
----------------
{ROOT}

Purpose
-------
This package was created to support a structured audit of the
relationship between:

    manuscript
        ->
    implementation
        ->
    datasets
        ->
    experiments
        ->
    numerical results
        ->
    figures/reports

Files collected
---------------
{summary['files_collected']}

Collected size
--------------
{summary['collected_size']}

Important directories
---------------------

project_files/
    Relevant files copied from the original project while
    preserving their original relative paths.

existing_project_audit/
    Previous project inventory and structural audit.

large_dataset_manifest/
    Manifest and summary of the ArSL/ArSL21L datasets.

large_dataset_samples/
    Small representative sample of the large image/label
    datasets. The complete ArSL datasets were intentionally
    NOT duplicated.

Important files
---------------

COLLECTED_FILES_MANIFEST.csv
    Complete list of collected files, original relative
    locations, sizes, and SHA-256 checksums.

EVIDENCE_PACKAGE_SUMMARY.json
    Machine-readable package summary.

SKIPPED_FILES.csv
    Files intentionally omitted because of size limits.

COLLECTION_ERRORS.csv
    Any errors encountered during collection.

Safety
------
The original project was not modified.

The full ArSL image dataset was not copied because it contains
many thousands of files. Instead, the package contains a
manifest plus representative samples.

Recommended audit
-----------------
Use this package to determine:

1. What algorithm is actually implemented.
2. Which scripts constitute the main experimental pipeline.
3. Which datasets were actually used.
4. How processed data were generated.
5. Which result files support manuscript tables and claims.
6. Which figures were generated by which experiments.
7. Whether saved models/checkpoints are available.
8. Whether the implementation matches the manuscript.
9. Whether the project is suitable for a public
   reproducibility repository.
"""

    (
        OUTPUT
        / "README_EVIDENCE_PACKAGE.txt"
    ).write_text(
        text.strip() + "\n",
        encoding="utf-8"
    )


# ============================================================
# 18. CREATE ZIP
# ============================================================

def create_zip():

    print()
    print("Creating ZIP archive...")

    zip_path = Path(
        shutil.make_archive(
            str(ZIP_BASE),
            "zip",
            root_dir=OUTPUT.parent,
            base_dir=OUTPUT.name,
        )
    )

    return zip_path


# ============================================================
# 19. MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PAPER4 CODE / DATA / RESULTS EVIDENCE COLLECTOR")
    print("=" * 80)

    print()
    print("Project:")
    print(ROOT)

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Project does not exist:\n{ROOT}"
        )

    # --------------------------------------------------------
    # Clean previous evidence package only
    # --------------------------------------------------------

    if OUTPUT.exists():

        print()
        print(
            "Removing previous evidence package..."
        )

        shutil.rmtree(OUTPUT)

    previous_zip = Path(
        str(ZIP_BASE) + ".zip"
    )

    if previous_zip.exists():
        previous_zip.unlink()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Main collection
    # --------------------------------------------------------

    counts = collect_project_files()

    # --------------------------------------------------------
    # Existing structural audit
    # --------------------------------------------------------

    print()
    print("Copying previous project audit...")

    audit_count = (
        collect_existing_audit()
    )

    # --------------------------------------------------------
    # Large dataset manifest
    # --------------------------------------------------------

    arsl_summary = (
        create_large_dataset_manifest()
    )

    # --------------------------------------------------------
    # Representative samples
    # --------------------------------------------------------

    sample_count = (
        collect_arsl_samples()
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    write_collection_manifest()
    write_skipped_report()
    write_error_report()

    summary = write_summary(
        counts,
        audit_count,
        arsl_summary,
        sample_count
    )

    write_readme(summary)

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    zip_path = create_zip()

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Evidence files collected: "
        f"{len(manifest)}"
    )

    print(
        f"Evidence package size: "
        f"{summary['collected_size']}"
    )

    print()
    print("Category counts:")

    for category, count in sorted(
        counts.items()
    ):
        print(
            f"  {category:22s}: "
            f"{count}"
        )

    print()
    print(
        f"Previous audit files copied: "
        f"{audit_count}"
    )

    print(
        f"ArSL files inventoried: "
        f"{arsl_summary['total_files']}"
    )

    print(
        f"Representative ArSL samples copied: "
        f"{sample_count}"
    )

    print(
        f"Skipped oversized files: "
        f"{len(skipped)}"
    )

    print(
        f"Collection errors: "
        f"{len(errors)}"
    )

    print()
    print("Evidence folder:")
    print(OUTPUT)

    print()
    print("ZIP archive:")
    print(zip_path)

    print()
    print(
        "Upload Paper4_Code_Evidence.zip "
        "for the manuscript/code/results audit."
    )


# ============================================================
# 20. SAFE EXECUTION
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Collection cancelled by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 80)
        print("COLLECTION FAILED")
        print("=" * 80)

        print()
        print(f"Error: {exc}")

        print()
        print("Full traceback:")

        traceback.print_exc()

        sys.exit(1)