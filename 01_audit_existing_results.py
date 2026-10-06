"""
01_audit_existing_results_v2.py
================================

Purpose
-------
Second-pass forensic audit of the HFAGM project.

This version corrects weaknesses in the first audit:

1. Excludes audit/revision scripts from evidence.
2. Does not search raw image/label filenames for metrics.
3. Separates raw multimodal assets from multimodal experiments.
4. Builds an experiment registry for New_EXPs, New_EXP2, New_EXP3,
   scenario1-scenario4, and any other experiment folders.
5. Extracts metric rows from CSV/JSON/TXT outputs.
6. Searches code for:
   - prediction generation/saving
   - confusion-matrix computation
   - FID/Frechet computation
   - fairness computation
   - scalability/runtime/memory measurement
   - ablation switches
   - multimodal loading/fusion
7. Detects actual prediction-like data by columns/content, not just filenames.
8. Maps manuscript claims only to legitimate experimental evidence.
9. Produces a cleaner audit for Reviewer #3 comments 40-54.

This script does NOT train models or alter experimental files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

AUDIT_DIR = PROJECT_ROOT / "outputs" / "revision_audit_v2"

EXCLUDE_DIR_NAMES = {
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
    "revision_audit",
    "revision_audit_v2",
    "new_code",              # excludes revision/audit scripts
}

EXCLUDE_FILE_NAMES = {
    "01_audit_existing_results.py",
    "01_audit_existing_results_v2.py",
}

# Raw assets should be inventoried, but not treated as experimental evidence.
RAW_ASSET_DIR_NAMES = {
    "images",
    "labels",
}

MAX_TEXT_BYTES = 30 * 1024 * 1024
MAX_TABLE_BYTES = 200 * 1024 * 1024
MAX_HASH_BYTES = 500 * 1024 * 1024

IMPORTANT_EXPERIMENT_HINTS = {
    "new_exps",
    "new_exp2",
    "new_exp3",
    "scenario1",
    "scenario2",
    "scenario3",
    "scenario4",
}


# ============================================================
# 2. FILE TYPES
# ============================================================

TEXT_EXTENSIONS = {
    ".txt", ".log", ".md", ".rst",
    ".py", ".yaml", ".yml",
    ".ini", ".cfg", ".toml",
}

TABLE_EXTENSIONS = {
    ".csv", ".tsv",
}

EXCEL_EXTENSIONS = {
    ".xlsx", ".xls",
}

JSON_EXTENSIONS = {
    ".json",
}

NUMPY_EXTENSIONS = {
    ".npy", ".npz",
}

FIGURE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff", ".bmp",
    ".svg", ".pdf",
}

MODEL_EXTENSIONS = {
    ".pt", ".pth", ".ckpt",
    ".pkl", ".pickle",
    ".joblib", ".h5", ".hdf5",
    ".onnx",
}

CODE_EXTENSIONS = {
    ".py", ".ipynb",
}


# ============================================================
# 3. CLAIMS TO VERIFY
# ============================================================

CLAIMS = [
    {
        "claim_id": "FID_HGF_1_5",
        "category": "fidelity",
        "description": "HGF/baseline fidelity reported around 1.5",
        "metric_terms": ["fid", "frechet", "fréchet"],
        "target_values": [1.5],
        "tolerance": 0.03,
    },
    {
        "claim_id": "FID_LIGHTWEIGHT_1_6",
        "category": "fidelity",
        "description": "Lightweight fidelity reported around 1.6",
        "metric_terms": ["fid", "frechet", "fréchet"],
        "target_values": [1.6],
        "tolerance": 0.03,
    },
    {
        "claim_id": "FID_LIGHTWEIGHT_3_2",
        "category": "fidelity",
        "description": "Lightweight fidelity reported as 3.2",
        "metric_terms": ["fid", "frechet", "fréchet"],
        "target_values": [3.2],
        "tolerance": 0.03,
    },
    {
        "claim_id": "SPD_0",
        "category": "fairness",
        "description": "SPD reported as 0",
        "metric_terms": ["spd", "statistical parity", "statistical_parity"],
        "target_values": [0.0],
        "tolerance": 1e-6,
    },
    {
        "claim_id": "SPD_00514",
        "category": "fairness",
        "description": "SPD reported as 0.0514",
        "metric_terms": ["spd", "statistical parity", "statistical_parity"],
        "target_values": [0.0514],
        "tolerance": 0.0002,
    },
    {
        "claim_id": "EOD_1",
        "category": "fairness",
        "description": "EOD reported as 1.0",
        "metric_terms": ["eod", "equal opportunity", "equal_opportunity"],
        "target_values": [1.0],
        "tolerance": 1e-6,
    },
    {
        "claim_id": "DI_1",
        "category": "fairness",
        "description": "DI reported as 1.0",
        "metric_terms": ["disparate impact", "disparate_impact", "impact ratio"],
        "target_values": [1.0],
        "tolerance": 0.001,
    },
    {
        "claim_id": "DI_0643",
        "category": "fairness",
        "description": "DI reported as 0.643",
        "metric_terms": ["disparate impact", "disparate_impact", "impact ratio"],
        "target_values": [0.643],
        "tolerance": 0.003,
    },
    {
        "claim_id": "DI_0209",
        "category": "fairness",
        "description": "DI reported as 0.209",
        "metric_terms": ["disparate impact", "disparate_impact", "impact ratio"],
        "target_values": [0.209],
        "tolerance": 0.003,
    },
    {
        "claim_id": "ACC_REAL_0983",
        "category": "utility",
        "description": "Real-data accuracy around 0.983",
        "metric_terms": ["accuracy", "acc"],
        "target_values": [0.983],
        "tolerance": 0.003,
    },
    {
        "claim_id": "ACC_BASELINE_0957",
        "category": "utility",
        "description": "Baseline synthetic accuracy around 0.957",
        "metric_terms": ["accuracy", "acc"],
        "target_values": [0.957],
        "tolerance": 0.003,
    },
    {
        "claim_id": "ACC_LIGHTWEIGHT_094",
        "category": "utility",
        "description": "Lightweight accuracy around 0.94",
        "metric_terms": ["accuracy", "acc"],
        "target_values": [0.94],
        "tolerance": 0.01,
    },
    {
        "claim_id": "ACC_LIGHTWEIGHT_0703",
        "category": "utility",
        "description": "Lightweight accuracy around 0.703",
        "metric_terms": ["accuracy", "acc"],
        "target_values": [0.703],
        "tolerance": 0.003,
    },
    {
        "claim_id": "F1_LIGHTWEIGHT_091",
        "category": "utility",
        "description": "Lightweight F1 around 0.91",
        "metric_terms": ["f1", "f1-score", "f1_score"],
        "target_values": [0.91],
        "tolerance": 0.01,
    },
    {
        "claim_id": "AUC_BASELINE_0996",
        "category": "utility",
        "description": "Baseline ROC-AUC around 0.996",
        "metric_terms": ["auc", "roc_auc", "roc-auc"],
        "target_values": [0.996],
        "tolerance": 0.003,
    },
    {
        "claim_id": "AUC_LIGHTWEIGHT_0801",
        "category": "utility",
        "description": "Lightweight ROC-AUC around 0.801",
        "metric_terms": ["auc", "roc_auc", "roc-auc"],
        "target_values": [0.801],
        "tolerance": 0.003,
    },
    {
        "claim_id": "RUNTIME_25_PERCENT",
        "category": "efficiency",
        "description": "Runtime improvement around 25%",
        "metric_terms": ["runtime", "training time", "faster", "speedup"],
        "target_values": [25.0],
        "tolerance": 2.0,
    },
    {
        "claim_id": "MEMORY_20_30_PERCENT",
        "category": "efficiency",
        "description": "GPU-memory reduction 20-30%",
        "metric_terms": ["gpu memory", "memory reduction", "vram"],
        "target_values": [20.0, 25.0, 30.0],
        "tolerance": 2.0,
    },
    {
        "claim_id": "MEMORY_40_PERCENT",
        "category": "efficiency",
        "description": "Hybrid memory increase around 40%",
        "metric_terms": ["gpu memory", "memory", "vram"],
        "target_values": [40.0],
        "tolerance": 2.0,
    },
    {
        "claim_id": "SYNTHETIC_N_500",
        "category": "scalability",
        "description": "Synthetic workload N=500",
        "metric_terms": ["sample", "samples", "dataset size", "synthetic size"],
        "target_values": [500.0],
        "tolerance": 0.0,
    },
    {
        "claim_id": "SYNTHETIC_N_100000",
        "category": "scalability",
        "description": "Synthetic workload N=100000",
        "metric_terms": ["sample", "samples", "dataset size", "synthetic size"],
        "target_values": [100000.0],
        "tolerance": 0.0,
    },
]


# ============================================================
# 4. REGEXES / TERMS
# ============================================================

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?%?"
)

PREDICTION_COLUMN_TERMS = {
    "y_true", "true_label", "ground_truth", "actual",
    "y_pred", "prediction", "predicted",
    "y_prob", "probability", "proba",
    "score", "prediction_score",
}

SENSITIVE_COLUMN_TERMS = {
    "gender", "sex", "nationality",
    "race", "ethnicity",
    "sensitive_attribute",
    "protected_group",
}

METRIC_COLUMN_TERMS = {
    "accuracy", "precision", "recall",
    "f1", "f1_score",
    "roc_auc", "auc",
    "spd", "eod",
    "disparate_impact", "di",
    "fid", "frechet_distance",
    "runtime", "training_time",
    "gpu_memory", "peak_memory",
    "throughput",
}

CODE_PATTERNS = {
    "prediction_creation": [
        r"\.predict\(",
        r"\.predict_proba\(",
        r"y_pred",
        r"predictions?",
        r"torch\.argmax",
    ],
    "prediction_saving": [
        r"np\.save\(",
        r"np\.savez\(",
        r"to_csv\(",
        r"csv\.writer",
        r"pickle\.dump",
        r"joblib\.dump",
    ],
    "confusion_matrix": [
        r"confusion_matrix\(",
        r"ConfusionMatrixDisplay",
        r"confusion matrix",
    ],
    "fairness": [
        r"statistical[_ ]parity",
        r"equal[_ ]opportunity",
        r"disparate[_ ]impact",
        r"\bSPD\b",
        r"\bEOD\b",
        r"fairlearn",
        r"aif360",
    ],
    "fidelity": [
        r"\bFID\b",
        r"frechet",
        r"Fr[eé]chet",
        r"sqrtm\(",
        r"cov\(",
    ],
    "runtime": [
        r"time\.time\(",
        r"perf_counter\(",
        r"elapsed",
        r"runtime",
        r"training_time",
    ],
    "gpu_memory": [
        r"max_memory_allocated",
        r"memory_allocated",
        r"nvidia-smi",
        r"gpu memory",
        r"peak_memory",
    ],
    "ablation": [
        r"ablation",
        r"static fusion",
        r"adaptive fusion",
        r"equal weights?",
        r"without fairness",
        r"no_fairness",
        r"disable_fairness",
        r"without scalability",
        r"no_scalability",
    ],
    "multimodal": [
        r"multimodal",
        r"image_encoder",
        r"text_encoder",
        r"tabular",
        r"fusion",
        r"ArSL",
        r"credit",
        r"finance",
    ],
}


# ============================================================
# 5. HELPERS
# ============================================================

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1


def modified_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat(timespec="seconds")
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    s = size_bytes(path)

    if s < 0 or s > MAX_HASH_BYTES:
        return ""

    h = hashlib.sha256()

    try:
        with path.open("rb") as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):
                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return ""


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    preferred_columns: Optional[List[str]] = None,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    keys = set()

    for row in rows:
        keys.update(row.keys())

    columns = []

    if preferred_columns:
        columns.extend(preferred_columns)

    for k in sorted(keys):
        if k not in columns:
            columns.append(k)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        if not columns:
            return

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def normalize_column_name(name: Any) -> str:
    text = str(name).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


def path_parts_lower(path: Path) -> set:
    return {
        p.lower()
        for p in path.parts
    }


def is_raw_asset(path: Path) -> bool:
    return bool(
        path_parts_lower(path)
        .intersection(RAW_ASSET_DIR_NAMES)
    )


def is_excluded(path: Path) -> bool:

    if path.name.lower() in {
        x.lower()
        for x in EXCLUDE_FILE_NAMES
    }:
        return True

    parts = path_parts_lower(path)

    if parts.intersection(
        {x.lower() for x in EXCLUDE_DIR_NAMES}
    ):
        return True

    return False


# ============================================================
# 6. PROJECT FILE WALK
# ============================================================

def iter_project_files() -> Iterable[Path]:

    for root, dirs, files in os.walk(PROJECT_ROOT):

        root_path = Path(root)

        dirs[:] = [
            d for d in dirs
            if d.lower() not in {
                x.lower()
                for x in EXCLUDE_DIR_NAMES
            }
        ]

        for filename in files:

            path = root_path / filename

            if is_excluded(path):
                continue

            if AUDIT_DIR in path.parents:
                continue

            yield path


# ============================================================
# 7. EXPERIMENT IDENTIFICATION
# ============================================================

def identify_experiment(path: Path) -> str:

    parts = [
        p.lower()
        for p in path.parts
    ]

    for hint in IMPORTANT_EXPERIMENT_HINTS:
        if hint in parts:
            return hint

    if "experiments" in parts:

        idx = parts.index("experiments")

        if idx + 1 < len(parts):
            return parts[idx + 1]

    return ""


def identify_dataset_from_path(path: Path) -> str:

    text = rel(path).lower()

    if "arsl" in text:
        return "ArSL"

    if "covid" in text:
        return "COVID-clinical"

    if "finance" in text or "credit" in text:
        return "Finance/Credit"

    return ""


# ============================================================
# 8. FILE CONTENT READERS
# ============================================================

def read_text_file(path: Path) -> str:

    if size_bytes(path) > MAX_TEXT_BYTES:
        return ""

    for enc in (
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ):
        try:
            return path.read_text(
                encoding=enc,
                errors="ignore"
            )
        except Exception:
            continue

    return ""


def read_csv_rows(
    path: Path,
    max_rows: Optional[int] = None
) -> Tuple[List[str], List[Dict[str, str]]]:

    delimiter = (
        "\t"
        if path.suffix.lower() == ".tsv"
        else ","
    )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="ignore",
            newline=""
        ) as f:

            reader = csv.DictReader(
                f,
                delimiter=delimiter
            )

            columns = reader.fieldnames or []

            rows = []

            for i, row in enumerate(reader):

                rows.append(
                    {
                        str(k): str(v)
                        for k, v in row.items()
                    }
                )

                if (
                    max_rows is not None
                    and i + 1 >= max_rows
                ):
                    break

        return columns, rows

    except Exception:
        return [], []


def read_json(path: Path) -> Any:

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        )
    except Exception:
        return None


# ============================================================
# 9. FILE INVENTORY
# ============================================================

def inventory_record(path: Path) -> Dict[str, Any]:

    ext = path.suffix.lower()
    experiment = identify_experiment(path)

    return {
        "relative_path": rel(path),
        "filename": path.name,
        "extension": ext,
        "size_bytes": size_bytes(path),
        "modified_time": modified_time(path),
        "sha256": sha256_file(path),
        "experiment": experiment,
        "dataset_hint": identify_dataset_from_path(path),
        "is_raw_asset": int(is_raw_asset(path)),
        "is_table": int(
            ext in TABLE_EXTENSIONS
            or ext in EXCEL_EXTENSIONS
        ),
        "is_code": int(
            ext in CODE_EXTENSIONS
        ),
        "is_model": int(
            ext in MODEL_EXTENSIONS
        ),
        "is_figure": int(
            ext in FIGURE_EXTENSIONS
        ),
    }


# ============================================================
# 10. TABLE SCHEMA ANALYSIS
# ============================================================

def analyze_table_schema(
    path: Path
) -> Optional[Dict[str, Any]]:

    if path.suffix.lower() not in TABLE_EXTENSIONS:
        return None

    columns, rows = read_csv_rows(
        path,
        max_rows=None
    )

    normalized = [
        normalize_column_name(c)
        for c in columns
    ]

    prediction_matches = [
        c for c in normalized
        if c in PREDICTION_COLUMN_TERMS
        or any(
            term in c
            for term in PREDICTION_COLUMN_TERMS
        )
    ]

    sensitive_matches = [
        c for c in normalized
        if c in SENSITIVE_COLUMN_TERMS
        or any(
            term in c
            for term in SENSITIVE_COLUMN_TERMS
        )
    ]

    metric_matches = [
        c for c in normalized
        if c in METRIC_COLUMN_TERMS
        or any(
            term in c
            for term in METRIC_COLUMN_TERMS
        )
    ]

    return {
        "relative_path": rel(path),
        "experiment": identify_experiment(path),
        "dataset_hint": identify_dataset_from_path(path),
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": "; ".join(columns),
        "prediction_columns":
            "; ".join(prediction_matches),
        "sensitive_columns":
            "; ".join(sensitive_matches),
        "metric_columns":
            "; ".join(metric_matches),
        "is_prediction_table":
            int(bool(prediction_matches)),
        "is_metric_table":
            int(bool(metric_matches)),
    }


# ============================================================
# 11. METRIC ROW EXTRACTION
# ============================================================

def safe_float(value: Any) -> Optional[float]:

    if value is None:
        return None

    text = str(value).strip()

    text = text.rstrip("%")

    try:
        return float(text)
    except Exception:
        return None


def extract_metric_rows(
    path: Path
) -> List[Dict[str, Any]]:

    if path.suffix.lower() not in TABLE_EXTENSIONS:
        return []

    columns, rows = read_csv_rows(path)

    if not columns:
        return []

    normalized_map = {
        c: normalize_column_name(c)
        for c in columns
    }

    metric_columns = [
        original
        for original, norm in normalized_map.items()
        if (
            norm in METRIC_COLUMN_TERMS
            or any(
                term in norm
                for term in METRIC_COLUMN_TERMS
            )
        )
    ]

    if not metric_columns:
        return []

    output = []

    for row_index, row in enumerate(
        rows,
        start=1
    ):

        context = {
            k: v
            for k, v in row.items()
            if normalize_column_name(k)
            not in METRIC_COLUMN_TERMS
        }

        for metric_col in metric_columns:

            raw = row.get(metric_col)

            value = safe_float(raw)

            if value is None:
                continue

            output.append({
                "relative_path": rel(path),
                "experiment":
                    identify_experiment(path),
                "dataset_hint":
                    identify_dataset_from_path(path),
                "row_index": row_index,
                "metric_name":
                    normalize_column_name(metric_col),
                "metric_column": metric_col,
                "value": value,
                "raw_value": raw,
                "context":
                    json.dumps(
                        context,
                        ensure_ascii=False
                    ),
            })

    return output


# ============================================================
# 12. JSON CONFIG / RESULT EXTRACTION
# ============================================================

def flatten_json(
    obj: Any,
    prefix: str = ""
) -> List[Tuple[str, Any]]:

    output = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            new_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            output.extend(
                flatten_json(
                    value,
                    new_prefix
                )
            )

    elif isinstance(obj, list):

        for i, value in enumerate(obj):

            new_prefix = (
                f"{prefix}[{i}]"
            )

            output.extend(
                flatten_json(
                    value,
                    new_prefix
                )
            )

    else:
        output.append(
            (prefix, obj)
        )

    return output


def extract_json_values(
    path: Path
) -> List[Dict[str, Any]]:

    if path.suffix.lower() != ".json":
        return []

    obj = read_json(path)

    if obj is None:
        return []

    rows = []

    for key, value in flatten_json(obj):

        numeric = safe_float(value)

        rows.append({
            "relative_path": rel(path),
            "experiment":
                identify_experiment(path),
            "dataset_hint":
                identify_dataset_from_path(path),
            "json_key": key,
            "raw_value": value,
            "numeric_value":
                "" if numeric is None else numeric,
        })

    return rows


# ============================================================
# 13. CODE FORENSICS
# ============================================================

def audit_code_file(
    path: Path
) -> List[Dict[str, Any]]:

    if path.suffix.lower() != ".py":
        return []

    text = read_text_file(path)

    if not text:
        return []

    rows = []

    for category, patterns in CODE_PATTERNS.items():

        matches = []

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):
                matches.append(pattern)

        if matches:

            rows.append({
                "relative_path": rel(path),
                "experiment":
                    identify_experiment(path),
                "dataset_hint":
                    identify_dataset_from_path(path),
                "evidence_category":
                    category,
                "matched_patterns":
                    "; ".join(matches),
            })

    return rows


def find_save_statements(
    path: Path
) -> List[Dict[str, Any]]:

    if path.suffix.lower() != ".py":
        return []

    text = read_text_file(path)

    if not text:
        return []

    lines = text.splitlines()

    save_patterns = [
        r"np\.save",
        r"np\.savez",
        r"to_csv",
        r"torch\.save",
        r"joblib\.dump",
        r"pickle\.dump",
        r"savefig",
    ]

    rows = []

    for line_no, line in enumerate(
        lines,
        start=1
    ):

        if any(
            re.search(
                p,
                line,
                flags=re.IGNORECASE
            )
            for p in save_patterns
        ):
            rows.append({
                "relative_path": rel(path),
                "experiment":
                    identify_experiment(path),
                "line_number": line_no,
                "statement":
                    line.strip()[:2000],
            })

    return rows


# ============================================================
# 14. CLAIM EVIDENCE SEARCH
# ============================================================

def extract_numbers(text: str) -> List[float]:

    values = []

    for match in NUMBER_PATTERN.finditer(text):

        raw = match.group(0)

        try:
            values.append(
                float(raw.rstrip("%"))
            )
        except Exception:
            pass

    return values


def values_close(
    found: float,
    target: float,
    tolerance: float
) -> bool:

    if tolerance == 0:
        return found == target

    return abs(found - target) <= tolerance


def legitimate_evidence_text(
    path: Path
) -> str:
    """
    Only searchable experimental evidence.
    Never raw image/label assets.
    Never revision code.
    """

    if is_raw_asset(path):
        return ""

    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return read_text_file(path)

    if ext in TABLE_EXTENSIONS:

        columns, rows = read_csv_rows(path)

        pieces = [
            " | ".join(columns)
        ]

        for row in rows:

            pieces.append(
                " | ".join(
                    f"{k}={v}"
                    for k, v in row.items()
                )
            )

        return "\n".join(pieces)

    if ext == ".json":

        obj = read_json(path)

        if obj is not None:
            return json.dumps(
                obj,
                ensure_ascii=False,
                indent=2
            )

    return ""


def search_claims(
    path: Path
) -> List[Dict[str, Any]]:

    text = legitimate_evidence_text(path)

    if not text:
        return []

    lines = text.splitlines()

    rows = []

    for claim in CLAIMS:

        for line_no, line in enumerate(
            lines,
            start=1
        ):

            lower = line.lower()

            if not any(
                term.lower() in lower
                for term in claim["metric_terms"]
            ):
                continue

            numbers = extract_numbers(line)

            for found in numbers:

                for target in claim[
                    "target_values"
                ]:

                    if values_close(
                        found,
                        target,
                        claim["tolerance"]
                    ):

                        rows.append({
                            "claim_id":
                                claim["claim_id"],
                            "category":
                                claim["category"],
                            "description":
                                claim["description"],
                            "target_value":
                                target,
                            "found_value":
                                found,
                            "relative_path":
                                rel(path),
                            "experiment":
                                identify_experiment(path),
                            "dataset_hint":
                                identify_dataset_from_path(path),
                            "line_number":
                                line_no,
                            "context":
                                line.strip()[:2000],
                        })

    return rows


# ============================================================
# 15. EXPERIMENT REGISTRY
# ============================================================

def build_experiment_registry(
    inventory: List[Dict[str, Any]],
    schemas: List[Dict[str, Any]],
    metric_rows: List[Dict[str, Any]],
    code_evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    experiments = defaultdict(
        lambda: {
            "files": set(),
            "datasets": set(),
            "tables": set(),
            "metric_tables": set(),
            "prediction_tables": set(),
            "code_files": set(),
            "models": set(),
            "figures": set(),
            "metrics": set(),
            "code_evidence": set(),
        }
    )

    for row in inventory:

        exp = row.get(
            "experiment"
        )

        if not exp:
            continue

        e = experiments[exp]

        e["files"].add(
            row["relative_path"]
        )

        if row.get("dataset_hint"):
            e["datasets"].add(
                row["dataset_hint"]
            )

        if row.get("is_table"):
            e["tables"].add(
                row["relative_path"]
            )

        if row.get("is_code"):
            e["code_files"].add(
                row["relative_path"]
            )

        if row.get("is_model"):
            e["models"].add(
                row["relative_path"]
            )

        if row.get("is_figure"):
            e["figures"].add(
                row["relative_path"]
            )

    for row in schemas:

        exp = row.get("experiment")

        if not exp:
            continue

        if row.get("is_metric_table"):
            experiments[exp][
                "metric_tables"
            ].add(
                row["relative_path"]
            )

        if row.get("is_prediction_table"):
            experiments[exp][
                "prediction_tables"
            ].add(
                row["relative_path"]
            )

    for row in metric_rows:

        exp = row.get("experiment")

        if exp:
            experiments[exp][
                "metrics"
            ].add(
                row["metric_name"]
            )

    for row in code_evidence:

        exp = row.get("experiment")

        if exp:
            experiments[exp][
                "code_evidence"
            ].add(
                row[
                    "evidence_category"
                ]
            )

    output = []

    for exp, e in sorted(
        experiments.items()
    ):

        output.append({
            "experiment": exp,
            "datasets":
                "; ".join(
                    sorted(e["datasets"])
                ),
            "total_files":
                len(e["files"]),
            "table_count":
                len(e["tables"]),
            "metric_table_count":
                len(e["metric_tables"]),
            "prediction_table_count":
                len(e["prediction_tables"]),
            "code_file_count":
                len(e["code_files"]),
            "model_count":
                len(e["models"]),
            "figure_count":
                len(e["figures"]),
            "metrics_present":
                "; ".join(
                    sorted(e["metrics"])
                ),
            "code_evidence":
                "; ".join(
                    sorted(
                        e["code_evidence"]
                    )
                ),
            "metric_tables":
                "; ".join(
                    sorted(
                        e["metric_tables"]
                    )
                ),
            "prediction_tables":
                "; ".join(
                    sorted(
                        e[
                            "prediction_tables"
                        ]
                    )
                ),
            "code_files":
                "; ".join(
                    sorted(
                        e["code_files"]
                    )
                ),
        })

    return output


# ============================================================
# 16. MULTIMODAL EVIDENCE CLASSIFICATION
# ============================================================

def classify_multimodal_evidence(
    inventory: List[Dict[str, Any]],
    code_evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    output = []

    code_multimodal_files = {
        row["relative_path"]
        for row in code_evidence
        if row[
            "evidence_category"
        ] == "multimodal"
    }

    for row in inventory:

        text = row[
            "relative_path"
        ].lower()

        raw_asset = bool(
            row["is_raw_asset"]
        )

        relevant = any(
            term in text
            for term in [
                "arsl",
                "multimodal",
                "image",
                "text",
                "finance",
                "credit",
                "embedding",
            ]
        )

        if not relevant:
            continue

        if raw_asset:
            evidence_type = (
                "RAW_ASSET_ONLY"
            )

        elif (
            row["relative_path"]
            in code_multimodal_files
        ):
            evidence_type = (
                "MULTIMODAL_CODE_EVIDENCE"
            )

        elif row[
            "extension"
        ] in TABLE_EXTENSIONS:
            evidence_type = (
                "POTENTIAL_MULTIMODAL_RESULT"
            )

        else:
            evidence_type = (
                "RELATED_FILE"
            )

        output.append({
            **row,
            "multimodal_evidence_type":
                evidence_type,
        })

    return output


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HFAGM PROJECT - EXISTING RESULTS AUDIT V2")
    print("=" * 80)

    if not PROJECT_ROOT.exists():
        print(
            f"\nERROR: Project does not exist:\n"
            f"{PROJECT_ROOT}"
        )
        sys.exit(1)

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = list(
        iter_project_files()
    )

    print(
        f"\nFiles to inspect: "
        f"{len(files):,}"
    )

    inventory = []
    schemas = []
    metric_rows = []
    json_values = []
    code_evidence = []
    save_statements = []
    claim_hits = []

    for i, path in enumerate(
        files,
        start=1
    ):

        if (
            i == 1
            or i % 500 == 0
            or i == len(files)
        ):
            print(
                f"Scanning {i:,}/"
                f"{len(files):,}"
            )

        inventory.append(
            inventory_record(path)
        )

        schema = analyze_table_schema(
            path
        )

        if schema:
            schemas.append(schema)

        metric_rows.extend(
            extract_metric_rows(path)
        )

        json_values.extend(
            extract_json_values(path)
        )

        code_evidence.extend(
            audit_code_file(path)
        )

        save_statements.extend(
            find_save_statements(path)
        )

        claim_hits.extend(
            search_claims(path)
        )

    # --------------------------------------------------------
    # Build experiment registry
    # --------------------------------------------------------

    experiment_registry = (
        build_experiment_registry(
            inventory,
            schemas,
            metric_rows,
            code_evidence,
        )
    )

    multimodal_evidence = (
        classify_multimodal_evidence(
            inventory,
            code_evidence,
        )
    )

    # --------------------------------------------------------
    # Claim mapping summary
    # --------------------------------------------------------

    grouped_claim_hits = defaultdict(list)

    for hit in claim_hits:
        grouped_claim_hits[
            hit["claim_id"]
        ].append(hit)

    claim_mapping = []

    for claim in CLAIMS:

        hits = grouped_claim_hits.get(
            claim["claim_id"],
            []
        )

        files_found = sorted({
            h["relative_path"]
            for h in hits
        })

        experiments_found = sorted({
            h["experiment"]
            for h in hits
            if h["experiment"]
        })

        datasets_found = sorted({
            h["dataset_hint"]
            for h in hits
            if h["dataset_hint"]
        })

        values_found = sorted({
            h["found_value"]
            for h in hits
        })

        claim_mapping.append({
            "claim_id":
                claim["claim_id"],
            "category":
                claim["category"],
            "description":
                claim["description"],
            "target_values":
                "; ".join(
                    str(v)
                    for v in claim[
                        "target_values"
                    ]
                ),
            "status":
                (
                    "FOUND_IN_EXPERIMENTAL_EVIDENCE"
                    if hits
                    else
                    "NOT_FOUND"
                ),
            "hit_count":
                len(hits),
            "found_values":
                "; ".join(
                    str(v)
                    for v in values_found
                ),
            "experiments":
                "; ".join(
                    experiments_found
                ),
            "datasets":
                "; ".join(
                    datasets_found
                ),
            "candidate_files":
                "; ".join(
                    files_found
                ),
        })

    # --------------------------------------------------------
    # Specialized subsets
    # --------------------------------------------------------

    prediction_tables = [
        row
        for row in schemas
        if row.get(
            "is_prediction_table"
        )
    ]

    metric_tables = [
        row
        for row in schemas
        if row.get(
            "is_metric_table"
        )
    ]

    actual_multimodal_code = [
        row
        for row in code_evidence
        if row.get(
            "evidence_category"
        ) == "multimodal"
    ]

    fairness_code = [
        row
        for row in code_evidence
        if row.get(
            "evidence_category"
        ) == "fairness"
    ]

    fidelity_code = [
        row
        for row in code_evidence
        if row.get(
            "evidence_category"
        ) == "fidelity"
    ]

    confusion_code = [
        row
        for row in code_evidence
        if row.get(
            "evidence_category"
        ) == "confusion_matrix"
    ]

    prediction_code = [
        row
        for row in code_evidence
        if row.get(
            "evidence_category"
        ) in {
            "prediction_creation",
            "prediction_saving",
        }
    ]

    runtime_code = [
        row
        for row in code_evidence
        if row.get(
            "evidence_category"
        ) in {
            "runtime",
            "gpu_memory",
        }
    ]

    ablation_code = [
        row
        for row in code_evidence
        if row.get(
            "evidence_category"
        ) == "ablation"
    ]

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    write_csv(
        AUDIT_DIR /
        "file_inventory_v2.csv",
        inventory
    )

    write_csv(
        AUDIT_DIR /
        "table_schema_inventory.csv",
        schemas
    )

    write_csv(
        AUDIT_DIR /
        "prediction_tables.csv",
        prediction_tables
    )

    write_csv(
        AUDIT_DIR /
        "metric_tables.csv",
        metric_tables
    )

    write_csv(
        AUDIT_DIR /
        "all_metric_rows.csv",
        metric_rows
    )

    write_csv(
        AUDIT_DIR /
        "json_config_values.csv",
        json_values
    )

    write_csv(
        AUDIT_DIR /
        "code_evidence.csv",
        code_evidence
    )

    write_csv(
        AUDIT_DIR /
        "prediction_code_evidence.csv",
        prediction_code
    )

    write_csv(
        AUDIT_DIR /
        "confusion_matrix_code_evidence.csv",
        confusion_code
    )

    write_csv(
        AUDIT_DIR /
        "fairness_code_evidence.csv",
        fairness_code
    )

    write_csv(
        AUDIT_DIR /
        "fidelity_code_evidence.csv",
        fidelity_code
    )

    write_csv(
        AUDIT_DIR /
        "runtime_memory_code_evidence.csv",
        runtime_code
    )

    write_csv(
        AUDIT_DIR /
        "ablation_code_evidence.csv",
        ablation_code
    )

    write_csv(
        AUDIT_DIR /
        "multimodal_code_evidence.csv",
        actual_multimodal_code
    )

    write_csv(
        AUDIT_DIR /
        "save_statements.csv",
        save_statements
    )

    write_csv(
        AUDIT_DIR /
        "claim_evidence_v2.csv",
        claim_hits
    )

    write_csv(
        AUDIT_DIR /
        "manuscript_claim_mapping_v2.csv",
        claim_mapping
    )

    write_csv(
        AUDIT_DIR /
        "experiment_registry.csv",
        experiment_registry
    )

    write_csv(
        AUDIT_DIR /
        "multimodal_evidence_classification.csv",
        multimodal_evidence
    )

    # --------------------------------------------------------
    # Important experiment files only
    # --------------------------------------------------------

    important_experiment_files = [
        row
        for row in inventory
        if row.get(
            "experiment"
        ) in IMPORTANT_EXPERIMENT_HINTS
    ]

    write_csv(
        AUDIT_DIR /
        "important_experiment_files.csv",
        important_experiment_files
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    raw_multimodal_count = sum(
        1
        for row in multimodal_evidence
        if row[
            "multimodal_evidence_type"
        ] == "RAW_ASSET_ONLY"
    )

    actual_multimodal_code_count = len(
        actual_multimodal_code
    )

    located_claims = [
        c
        for c in claim_mapping
        if c["status"]
        == "FOUND_IN_EXPERIMENTAL_EVIDENCE"
    ]

    missing_claims = [
        c
        for c in claim_mapping
        if c["status"]
        == "NOT_FOUND"
    ]

    summary = [
        "=" * 80,
        "HFAGM PROJECT - EXISTING RESULTS AUDIT V2",
        "=" * 80,
        "",
        f"Generated: "
        f"{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Project root: {PROJECT_ROOT}",
        "",
        f"Total non-revision files audited: "
        f"{len(inventory):,}",
        "",
        "IMPORTANT COUNTS",
        "-" * 80,
        f"CSV/TSV tables: "
        f"{len(schemas):,}",
        f"Prediction-like tables: "
        f"{len(prediction_tables):,}",
        f"Metric tables: "
        f"{len(metric_tables):,}",
        f"Extracted metric rows: "
        f"{len(metric_rows):,}",
        f"Prediction-related code files: "
        f"{len(prediction_code):,}",
        f"Confusion-matrix code files: "
        f"{len(confusion_code):,}",
        f"Fairness code files: "
        f"{len(fairness_code):,}",
        f"Fidelity/Frechet code files: "
        f"{len(fidelity_code):,}",
        f"Runtime/memory code files: "
        f"{len(runtime_code):,}",
        f"Ablation code files: "
        f"{len(ablation_code):,}",
        "",
        "MULTIMODAL AUDIT",
        "-" * 80,
        f"Raw image/label assets: "
        f"{raw_multimodal_count:,}",
        f"Files with actual multimodal code evidence: "
        f"{actual_multimodal_code_count:,}",
        "",
        "CLAIM AUDIT",
        "-" * 80,
        f"Claims checked: {len(CLAIMS)}",
        f"Claims found in legitimate experimental evidence: "
        f"{len(located_claims)}",
        f"Claims not found: "
        f"{len(missing_claims)}",
        "",
        "EXPERIMENT REGISTRY",
        "-" * 80,
    ]

    for row in experiment_registry:

        summary.extend([
            f"Experiment: "
            f"{row['experiment']}",
            f"  Datasets: "
            f"{row['datasets'] or 'unknown'}",
            f"  Files: "
            f"{row['total_files']}",
            f"  Metric tables: "
            f"{row['metric_table_count']}",
            f"  Prediction tables: "
            f"{row['prediction_table_count']}",
            f"  Metrics: "
            f"{row['metrics_present'] or 'none detected'}",
            f"  Code evidence: "
            f"{row['code_evidence'] or 'none detected'}",
            "",
        ])

    summary.extend([
        "",
        "CLAIMS NOT FOUND IN LEGITIMATE EVIDENCE",
        "-" * 80,
    ])

    for row in missing_claims:
        summary.extend([
            f"{row['claim_id']}: "
            f"{row['description']}",
        ])

    summary.extend([
        "",
        "NEXT FILES TO REVIEW",
        "-" * 80,
        "1. experiment_registry.csv",
        "2. manuscript_claim_mapping_v2.csv",
        "3. all_metric_rows.csv",
        "4. prediction_tables.csv",
        "5. prediction_code_evidence.csv",
        "6. confusion_matrix_code_evidence.csv",
        "7. fairness_code_evidence.csv",
        "8. fidelity_code_evidence.csv",
        "9. runtime_memory_code_evidence.csv",
        "10. ablation_code_evidence.csv",
        "11. multimodal_code_evidence.csv",
        "12. json_config_values.csv",
        "13. save_statements.csv",
        "",
        "Interpretation rule:",
        "A manuscript claim is considered 'found' only when the matching value",
        "occurs in non-revision experimental/configuration/output evidence.",
        "Raw image filenames, label filenames, audit scripts, and revision files",
        "are not accepted as evidence.",
        "",
        "=" * 80,
    ])

    summary_path = (
        AUDIT_DIR /
        "audit_summary_v2.txt"
    )

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("AUDIT V2 COMPLETE")
    print("=" * 80)

    print(
        f"\nResults directory:\n"
        f"{AUDIT_DIR}"
    )

    print(
        "\nMost important outputs:"
    )

    print(
        AUDIT_DIR /
        "experiment_registry.csv"
    )

    print(
        AUDIT_DIR /
        "manuscript_claim_mapping_v2.csv"
    )

    print(
        AUDIT_DIR /
        "audit_summary_v2.txt"
    )


if __name__ == "__main__":
    main()