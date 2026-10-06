"""
02_reconstruct_and_recompute_primary_metrics.py
================================================

Purpose
-------
Reconstruct and independently recompute the primary classification metrics
used in the HFAGM manuscript.

This script is intentionally conservative.

It DOES:
    1. Discover existing experiment outputs.
    2. Identify existing true labels, predictions, probabilities, and scores.
    3. Inspect existing experiment CSV/JSON files.
    4. Inspect source code for prediction-generation logic.
    5. Match compatible y_true / y_pred / y_prob arrays.
    6. Recompute:
           - Accuracy
           - Precision
           - Recall / Sensitivity
           - Specificity
           - F1-score
           - ROC-AUC
           - Confusion matrix
    7. Compare recomputed metrics with previously reported metrics.
    8. Save canonical prediction tables when actual predictions exist.
    9. Explicitly identify experiments that cannot be reconstructed.

It DOES NOT:
    - fabricate predictions;
    - infer probabilities from accuracy/F1/AUC;
    - retrain models automatically;
    - overwrite historical experiment files;
    - assume that metrics from different datasets/configurations are comparable.

Output
------
outputs/revision_primary_metrics/

    source_file_inventory.csv
    experiment_metric_inventory.csv
    array_candidates.csv
    prediction_source_candidates.csv
    reconstruction_candidates.csv
    recomputed_primary_metrics.csv
    historical_vs_recomputed.csv
    confusion_matrices.csv
    canonical_prediction_tables/
    unreconstructable_experiments.csv
    primary_metrics_audit_summary.txt

Requirements
------------
Python 3.9+
numpy
pandas
scikit-learn

Optional:
joblib
torch
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
)

CANONICAL_PREDICTION_DIR = (
    OUTPUT_DIR
    / "canonical_prediction_tables"
)

# Existing audit directory from Script 01 V2
AUDIT_V2_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_audit_v2"
)

# Important experiment families detected by Audit V2.
IMPORTANT_EXPERIMENTS = {
    "new_exps",
    "new_exp2",
    "new_exp3",
    "scenario1",
    "scenario2",
    "scenario3",
    "scenario4",
}

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "revision_audit",
    "revision_audit_v2",
    "revision_primary_metrics",
    "new_code",
}

# These directories may contain tens of thousands of raw files and
# should not be searched for prediction arrays.
RAW_ASSET_DIR_NAMES = {
    "images",
    "labels",
}

MAX_CSV_SIZE_BYTES = 500 * 1024 * 1024
MAX_TEXT_SIZE_BYTES = 30 * 1024 * 1024

# Classification threshold if probabilities are available and no
# explicit threshold is documented.
DEFAULT_BINARY_THRESHOLD = 0.5

# Tolerance used only for reporting whether recomputed and historical
# values are numerically close.
METRIC_COMPARISON_TOLERANCE = 1e-6


# ============================================================
# 2. COLUMN TERMINOLOGY
# ============================================================

TRUE_LABEL_NAMES = {
    "y_true",
    "true",
    "true_label",
    "true_labels",
    "label",
    "labels",
    "target",
    "targets",
    "ground_truth",
    "groundtruth",
    "actual",
    "actual_label",
    "actual_labels",
}

PRED_LABEL_NAMES = {
    "y_pred",
    "pred",
    "prediction",
    "predictions",
    "predicted",
    "predicted_label",
    "predicted_labels",
    "class_prediction",
}

PROBABILITY_NAMES = {
    "y_prob",
    "y_probability",
    "prob",
    "proba",
    "probability",
    "probabilities",
    "prediction_probability",
    "prediction_probabilities",
    "positive_probability",
    "positive_prob",
    "score",
    "scores",
    "y_score",
}

METRIC_NAMES = {
    "accuracy",
    "acc",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "f1_score",
    "auc",
    "roc_auc",
    "macro_auc",
    "macro_f1",
    "accuracy_mean",
    "accuracy_std",
    "f1_mean",
    "f1_std",
    "auc_mean",
    "auc_std",
    "eod",
    "eod_mean",
    "eod_std",
}

IDENTIFIER_NAMES = {
    "id",
    "sample_id",
    "patient_id",
    "record_id",
    "index",
}

SEED_NAMES = {
    "seed",
    "random_seed",
    "random_state",
}

MODEL_NAMES = {
    "model",
    "model_name",
    "generator",
    "generation",
    "configuration",
    "config",
    "variant",
    "method",
    "approach",
}

SPLIT_NAMES = {
    "split",
    "partition",
    "subset",
    "fold",
    "set",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1


def safe_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat(timespec="seconds")
    except Exception:
        return ""


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    preferred_columns: Optional[List[str]] = None,
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    keys = set()
    for row in rows:
        keys.update(row.keys())

    columns: List[str] = []

    if preferred_columns:
        columns.extend(preferred_columns)

    for key in sorted(keys):
        if key not in columns:
            columns.append(key)

    if not columns:
        path.write_text("", encoding="utf-8")
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def path_parts_lower(path: Path) -> List[str]:
    return [
        p.lower()
        for p in path.parts
    ]


def is_raw_asset_path(path: Path) -> bool:
    parts = set(path_parts_lower(path))
    return bool(parts.intersection(RAW_ASSET_DIR_NAMES))


def identify_experiment(path: Path) -> str:

    parts = path_parts_lower(path)

    for exp in IMPORTANT_EXPERIMENTS:
        if exp in parts:
            return exp

    if "experiments" in parts:
        idx = parts.index("experiments")

        if idx + 1 < len(parts):
            return parts[idx + 1]

    return ""


def identify_dataset_hint(path: Path) -> str:

    text = relative_path(path).lower()

    if "arsl" in text:
        return "ArSL"

    if "covid" in text:
        return "COVID-clinical"

    if "finance" in text or "credit" in text:
        return "Finance/Credit"

    return ""


def iter_project_files() -> Iterable[Path]:

    for root, dirs, files in os.walk(PROJECT_ROOT):

        root_path = Path(root)

        dirs[:] = [
            d
            for d in dirs
            if d.lower() not in EXCLUDED_DIRS
        ]

        for filename in files:

            path = root_path / filename

            if OUTPUT_DIR in path.parents:
                continue

            yield path


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:

    if safe_size(path) < 0:
        return None

    if safe_size(path) > MAX_CSV_SIZE_BYTES:
        return None

    for kwargs in [
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8"},
        {"encoding": "latin-1"},
    ]:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            continue

    return None


def safe_read_json(path: Path) -> Optional[Any]:

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except Exception:
        return None


def safe_read_text(path: Path) -> str:

    if safe_size(path) < 0:
        return ""

    if safe_size(path) > MAX_TEXT_SIZE_BYTES:
        return ""

    for encoding in [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ]:
        try:
            return path.read_text(
                encoding=encoding,
                errors="ignore",
            )
        except Exception:
            continue

    return ""


def safe_numeric_series(
    series: pd.Series
) -> Optional[np.ndarray]:

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = numeric.notna()

    if valid.sum() == 0:
        return None

    return numeric.to_numpy()


def unique_non_nan(values: np.ndarray) -> np.ndarray:

    arr = np.asarray(values)

    if np.issubdtype(arr.dtype, np.number):
        return np.unique(
            arr[np.isfinite(arr)]
        )

    return np.unique(arr)


# ============================================================
# 4. EXPERIMENT METRIC INVENTORY
# ============================================================

def extract_existing_metric_rows(
    path: Path
) -> List[Dict[str, Any]]:

    if path.suffix.lower() != ".csv":
        return []

    df = safe_read_csv(path)

    if df is None or df.empty:
        return []

    normalized = {
        col: normalize_name(col)
        for col in df.columns
    }

    metric_cols = [
        original
        for original, normalized_name
        in normalized.items()
        if normalized_name in METRIC_NAMES
    ]

    if not metric_cols:
        return []

    context_cols = [
        col
        for col in df.columns
        if col not in metric_cols
    ]

    rows: List[Dict[str, Any]] = []

    for row_idx, row in df.iterrows():

        context = {
            str(col): row[col]
            for col in context_cols
        }

        for metric_col in metric_cols:

            raw = row[metric_col]

            try:
                value = float(raw)
            except Exception:
                continue

            rows.append({
                "experiment":
                    identify_experiment(path),
                "dataset_hint":
                    identify_dataset_hint(path),
                "relative_path":
                    relative_path(path),
                "row_index":
                    int(row_idx),
                "metric_name":
                    normalize_name(metric_col),
                "metric_value":
                    value,
                "context":
                    json.dumps(
                        context,
                        default=str,
                        ensure_ascii=False,
                    ),
            })

    return rows


# ============================================================
# 5. ARRAY / PREDICTION CANDIDATE DISCOVERY
# ============================================================

@dataclass
class ArrayCandidate:
    path: Path
    source_type: str
    variable_name: str
    array: np.ndarray
    experiment: str
    dataset_hint: str


def classify_column_role(
    column_name: str
) -> str:

    name = normalize_name(column_name)

    if name in TRUE_LABEL_NAMES:
        return "y_true"

    if name in PRED_LABEL_NAMES:
        return "y_pred"

    if name in PROBABILITY_NAMES:
        return "y_prob"

    if any(
        token in name
        for token in [
            "y_true",
            "true_label",
            "ground_truth",
        ]
    ):
        return "y_true"

    if any(
        token in name
        for token in [
            "y_pred",
            "predicted",
            "prediction",
        ]
    ):
        return "y_pred"

    if any(
        token in name
        for token in [
            "probability",
            "proba",
            "y_prob",
            "y_score",
        ]
    ):
        return "y_prob"

    return ""


def discover_csv_arrays(
    path: Path
) -> List[ArrayCandidate]:

    if path.suffix.lower() != ".csv":
        return []

    if is_raw_asset_path(path):
        return []

    df = safe_read_csv(path)

    if df is None or df.empty:
        return []

    candidates: List[ArrayCandidate] = []

    for col in df.columns:

        role = classify_column_role(col)

        if not role:
            continue

        numeric = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if numeric.notna().sum() != len(df):
            continue

        candidates.append(
            ArrayCandidate(
                path=path,
                source_type="csv_column",
                variable_name=f"{role}:{col}",
                array=numeric.to_numpy(),
                experiment=identify_experiment(path),
                dataset_hint=identify_dataset_hint(path),
            )
        )

    return candidates


def infer_npy_role(
    path: Path
) -> str:

    name = normalize_name(path.stem)

    if any(
        term in name
        for term in TRUE_LABEL_NAMES
    ):
        return "y_true"

    if any(
        term in name
        for term in PRED_LABEL_NAMES
    ):
        return "y_pred"

    if any(
        term in name
        for term in PROBABILITY_NAMES
    ):
        return "y_prob"

    return ""


def discover_numpy_arrays(
    path: Path
) -> List[ArrayCandidate]:

    if path.suffix.lower() not in {
        ".npy",
        ".npz",
    }:
        return []

    if is_raw_asset_path(path):
        return []

    candidates: List[ArrayCandidate] = []

    try:

        if path.suffix.lower() == ".npy":

            role = infer_npy_role(path)

            if not role:
                return []

            arr = np.load(
                path,
                allow_pickle=False,
            )

            candidates.append(
                ArrayCandidate(
                    path=path,
                    source_type="npy",
                    variable_name=role,
                    array=np.asarray(arr),
                    experiment=identify_experiment(path),
                    dataset_hint=identify_dataset_hint(path),
                )
            )

        else:

            archive = np.load(
                path,
                allow_pickle=False,
            )

            for key in archive.files:

                role = classify_column_role(key)

                if not role:
                    continue

                arr = np.asarray(
                    archive[key]
                )

                candidates.append(
                    ArrayCandidate(
                        path=path,
                        source_type="npz",
                        variable_name=f"{role}:{key}",
                        array=arr,
                        experiment=identify_experiment(path),
                        dataset_hint=identify_dataset_hint(path),
                    )
                )

    except Exception:
        pass

    return candidates


def discover_known_label_files(
    path: Path
) -> List[ArrayCandidate]:
    """
    Explicit support for existing y_train.csv / y_test.csv type files,
    even when the internal column name is generic.
    """

    if path.suffix.lower() != ".csv":
        return []

    stem = normalize_name(path.stem)

    role = ""

    if stem in {
        "y_test",
        "y_test_scaled",
        "test_labels",
    }:
        role = "y_true"

    elif stem in {
        "y_train",
        "y_train_scaled",
        "train_labels",
    }:
        role = "y_train"

    if not role:
        return []

    df = safe_read_csv(path)

    if df is None or df.empty:
        return []

    numeric_cols = []

    for col in df.columns:

        numeric = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if numeric.notna().sum() == len(df):
            numeric_cols.append(
                numeric.to_numpy()
            )

    if len(numeric_cols) != 1:
        return []

    return [
        ArrayCandidate(
            path=path,
            source_type="known_label_file",
            variable_name=role,
            array=numeric_cols[0],
            experiment=identify_experiment(path),
            dataset_hint=identify_dataset_hint(path),
        )
    ]


# ============================================================
# 6. SOURCE CODE FORENSICS
# ============================================================

def inspect_prediction_code(
    path: Path
) -> List[Dict[str, Any]]:

    if path.suffix.lower() != ".py":
        return []

    text = safe_read_text(path)

    if not text:
        return []

    patterns = {
        "predict": [
            r"\.predict\(",
            r"\.predict_proba\(",
            r"torch\.argmax",
            r"argmax\(",
        ],

        "classification_metrics": [
            r"accuracy_score",
            r"f1_score",
            r"roc_auc_score",
            r"precision_score",
            r"recall_score",
        ],

        "confusion_matrix": [
            r"confusion_matrix",
            r"ConfusionMatrixDisplay",
        ],

        "prediction_save": [
            r"np\.save",
            r"np\.savez",
            r"to_csv",
            r"torch\.save",
        ],

        "model_load": [
            r"joblib\.load",
            r"pickle\.load",
            r"torch\.load",
            r"load_state_dict",
        ],
    }

    rows = []

    lines = text.splitlines()

    for category, regexes in patterns.items():

        for line_no, line in enumerate(
            lines,
            start=1
        ):

            if not any(
                re.search(
                    pattern,
                    line,
                    flags=re.IGNORECASE,
                )
                for pattern in regexes
            ):
                continue

            rows.append({
                "experiment":
                    identify_experiment(path),
                "dataset_hint":
                    identify_dataset_hint(path),
                "relative_path":
                    relative_path(path),
                "category":
                    category,
                "line_number":
                    line_no,
                "statement":
                    line.strip()[:2000],
            })

    return rows


# ============================================================
# 7. METRIC RECOMPUTATION
# ============================================================

def clean_binary_labels(
    values: np.ndarray
) -> Optional[np.ndarray]:

    arr = np.asarray(values).reshape(-1)

    try:
        arr = arr.astype(float)
    except Exception:
        return None

    if np.any(~np.isfinite(arr)):
        return None

    unique = np.unique(arr)

    # Standard binary labels.
    if set(unique.tolist()).issubset(
        {0.0, 1.0}
    ):
        return arr.astype(int)

    # Common alternative {-1, +1}.
    if set(unique.tolist()).issubset(
        {-1.0, 1.0}
    ):
        return (
            (arr > 0)
            .astype(int)
        )

    return None


def clean_binary_predictions(
    values: np.ndarray
) -> Optional[np.ndarray]:

    return clean_binary_labels(values)


def clean_probability_array(
    values: np.ndarray,
) -> Optional[np.ndarray]:

    arr = np.asarray(values)

    # Binary probability vector.
    if arr.ndim == 1:

        try:
            arr = arr.astype(float)
        except Exception:
            return None

        if np.any(~np.isfinite(arr)):
            return None

        if np.any(arr < 0) or np.any(arr > 1):
            return None

        return arr

    # n x 1
    if arr.ndim == 2 and arr.shape[1] == 1:
        return clean_probability_array(
            arr[:, 0]
        )

    # Binary two-column probabilities:
    # assume second column is positive class probability.
    if arr.ndim == 2 and arr.shape[1] == 2:

        try:
            arr = arr.astype(float)
        except Exception:
            return None

        if np.any(~np.isfinite(arr)):
            return None

        if np.any(arr < 0) or np.any(arr > 1):
            return None

        row_sum = arr.sum(axis=1)

        if not np.allclose(
            row_sum,
            1.0,
            atol=1e-3,
        ):
            return None

        return arr[:, 1]

    return None


def calculate_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    if cm.shape != (2, 2):
        return float("nan")

    tn, fp, fn, tp = cm.ravel()

    denom = tn + fp

    if denom == 0:
        return float("nan")

    return float(
        tn / denom
    )


def recompute_binary_metrics(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = DEFAULT_BINARY_THRESHOLD,
) -> Tuple[Optional[Dict[str, Any]], str]:

    true_clean = clean_binary_labels(
        y_true
    )

    if true_clean is None:
        return None, (
            "y_true is not a valid binary label vector"
        )

    pred_clean = None
    prob_clean = None

    if y_prob is not None:

        prob_clean = clean_probability_array(
            y_prob
        )

        if prob_clean is None:
            return None, (
                "y_prob is not a valid binary probability vector"
            )

        if len(prob_clean) != len(true_clean):
            return None, (
                "y_prob length differs from y_true"
            )

    if y_pred is not None:

        pred_clean = clean_binary_predictions(
            y_pred
        )

        if pred_clean is None:
            return None, (
                "y_pred is not a valid binary label vector"
            )

        if len(pred_clean) != len(true_clean):
            return None, (
                "y_pred length differs from y_true"
            )

    if pred_clean is None and prob_clean is not None:

        pred_clean = (
            prob_clean >= threshold
        ).astype(int)

    if pred_clean is None:
        return None, (
            "Neither valid y_pred nor y_prob is available"
        )

    cm = confusion_matrix(
        true_clean,
        pred_clean,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        true_clean,
        pred_clean,
    )

    precision = precision_score(
        true_clean,
        pred_clean,
        zero_division=0,
    )

    recall = recall_score(
        true_clean,
        pred_clean,
        zero_division=0,
    )

    f1 = f1_score(
        true_clean,
        pred_clean,
        zero_division=0,
    )

    specificity = calculate_specificity(
        true_clean,
        pred_clean,
    )

    auc = float("nan")

    if prob_clean is not None:

        if len(np.unique(true_clean)) == 2:

            try:
                auc = roc_auc_score(
                    true_clean,
                    prob_clean,
                )
            except Exception:
                auc = float("nan")

    result = {
        "n_samples":
            len(true_clean),

        "n_negative":
            int((true_clean == 0).sum()),

        "n_positive":
            int((true_clean == 1).sum()),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "sensitivity":
            float(recall),

        "specificity":
            float(specificity),

        "f1":
            float(f1),

        "roc_auc":
            float(auc)
            if np.isfinite(auc)
            else "",

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "threshold":
            threshold
            if prob_clean is not None
            else "",

        "auc_source":
            "probability"
            if prob_clean is not None
            else "unavailable",
    }

    return result, ""


# ============================================================
# 8. MATCHING ARRAY CANDIDATES
# ============================================================

def candidate_role(
    candidate: ArrayCandidate
) -> str:

    return (
        candidate.variable_name
        .split(":")[0]
    )


def compatibility_score(
    y_true: ArrayCandidate,
    other: ArrayCandidate,
) -> int:

    score = 0

    if (
        y_true.experiment
        and other.experiment
        and y_true.experiment
        == other.experiment
    ):
        score += 5

    if (
        y_true.dataset_hint
        and other.dataset_hint
        and y_true.dataset_hint
        == other.dataset_hint
    ):
        score += 4

    if (
        len(np.asarray(y_true.array).reshape(-1))
        ==
        len(np.asarray(other.array).reshape(-1))
    ):
        score += 10

    if y_true.path.parent == other.path.parent:
        score += 5

    if (
        y_true.path.parent.parent
        ==
        other.path.parent.parent
    ):
        score += 2

    return score


def build_reconstruction_candidates(
    arrays: List[ArrayCandidate]
) -> List[Dict[str, Any]]:

    y_true_candidates = [
        c
        for c in arrays
        if candidate_role(c) == "y_true"
    ]

    y_pred_candidates = [
        c
        for c in arrays
        if candidate_role(c) == "y_pred"
    ]

    y_prob_candidates = [
        c
        for c in arrays
        if candidate_role(c) == "y_prob"
    ]

    rows: List[Dict[str, Any]] = []

    for y_true in y_true_candidates:

        true_len = len(
            np.asarray(
                y_true.array
            ).reshape(-1)
        )

        for y_pred in y_pred_candidates:

            pred_len = len(
                np.asarray(
                    y_pred.array
                ).reshape(-1)
            )

            if pred_len != true_len:
                continue

            rows.append({
                "candidate_type":
                    "y_true+y_pred",
                "score":
                    compatibility_score(
                        y_true,
                        y_pred,
                    ),
                "experiment_true":
                    y_true.experiment,
                "experiment_pred":
                    y_pred.experiment,
                "dataset_true":
                    y_true.dataset_hint,
                "dataset_pred":
                    y_pred.dataset_hint,
                "n_samples":
                    true_len,
                "y_true_path":
                    relative_path(
                        y_true.path
                    ),
                "y_true_variable":
                    y_true.variable_name,
                "prediction_path":
                    relative_path(
                        y_pred.path
                    ),
                "prediction_variable":
                    y_pred.variable_name,
            })

        for y_prob in y_prob_candidates:

            prob_arr = np.asarray(
                y_prob.array
            )

            prob_len = prob_arr.shape[0]

            if prob_len != true_len:
                continue

            rows.append({
                "candidate_type":
                    "y_true+y_prob",
                "score":
                    compatibility_score(
                        y_true,
                        y_prob,
                    ),
                "experiment_true":
                    y_true.experiment,
                "experiment_pred":
                    y_prob.experiment,
                "dataset_true":
                    y_true.dataset_hint,
                "dataset_pred":
                    y_prob.dataset_hint,
                "n_samples":
                    true_len,
                "y_true_path":
                    relative_path(
                        y_true.path
                    ),
                "y_true_variable":
                    y_true.variable_name,
                "prediction_path":
                    relative_path(
                        y_prob.path
                    ),
                "prediction_variable":
                    y_prob.variable_name,
            })

    rows.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return rows


# ============================================================
# 9. EXACT SAME-TABLE RECOMPUTATION
# ============================================================

def recompute_from_prediction_table(
    path: Path
) -> Tuple[
    List[Dict[str, Any]],
    Optional[pd.DataFrame]
]:

    if path.suffix.lower() != ".csv":
        return [], None

    df = safe_read_csv(path)

    if df is None or df.empty:
        return [], None

    roles = defaultdict(list)

    for col in df.columns:

        role = classify_column_role(
            col
        )

        if role:
            roles[role].append(col)

    if (
        not roles["y_true"]
        or (
            not roles["y_pred"]
            and not roles["y_prob"]
        )
    ):
        return [], None

    results = []

    # Prefer exact same-table combinations because
    # row alignment is guaranteed.
    for true_col in roles["y_true"]:

        pred_cols = (
            roles["y_pred"]
            if roles["y_pred"]
            else [None]
        )

        prob_cols = (
            roles["y_prob"]
            if roles["y_prob"]
            else [None]
        )

        # Avoid combinatorial duplication.
        if roles["y_pred"] and roles["y_prob"]:

            combinations = [
                (
                    roles["y_pred"][0],
                    roles["y_prob"][0],
                )
            ]

        elif roles["y_pred"]:

            combinations = [
                (col, None)
                for col in roles["y_pred"]
            ]

        else:

            combinations = [
                (None, col)
                for col in roles["y_prob"]
            ]

        for pred_col, prob_col in combinations:

            y_true = df[
                true_col
            ].to_numpy()

            y_pred = (
                df[pred_col].to_numpy()
                if pred_col
                else None
            )

            y_prob = (
                df[prob_col].to_numpy()
                if prob_col
                else None
            )

            metrics, reason = (
                recompute_binary_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_prob=y_prob,
                )
            )

            if metrics is None:

                results.append({
                    "status":
                        "FAILED",
                    "relative_path":
                        relative_path(path),
                    "experiment":
                        identify_experiment(path),
                    "dataset_hint":
                        identify_dataset_hint(path),
                    "y_true_column":
                        true_col,
                    "y_pred_column":
                        pred_col or "",
                    "y_prob_column":
                        prob_col or "",
                    "reason":
                        reason,
                })

                continue

            result = {
                "status":
                    "RECOMPUTED",
                "reconstruction_type":
                    "same_prediction_table",
                "relative_path":
                    relative_path(path),
                "experiment":
                    identify_experiment(path),
                "dataset_hint":
                    identify_dataset_hint(path),
                "y_true_column":
                    true_col,
                "y_pred_column":
                    pred_col or "",
                "y_prob_column":
                    prob_col or "",
                **metrics,
            }

            results.append(result)

            # Save canonical prediction table.
            true_clean = clean_binary_labels(
                y_true
            )

            if pred_col:
                pred_clean = (
                    clean_binary_predictions(
                        y_pred
                    )
                )
            else:
                prob_clean = (
                    clean_probability_array(
                        y_prob
                    )
                )

                pred_clean = (
                    prob_clean
                    >= DEFAULT_BINARY_THRESHOLD
                ).astype(int)

            canonical = pd.DataFrame({
                "row_id":
                    np.arange(
                        len(true_clean)
                    ),
                "y_true":
                    true_clean,
                "y_pred":
                    pred_clean,
            })

            if y_prob is not None:

                prob_clean = (
                    clean_probability_array(
                        y_prob
                    )
                )

                if prob_clean is not None:
                    canonical[
                        "y_prob"
                    ] = prob_clean

            return results, canonical

    return results, None


# ============================================================
# 10. HISTORICAL VS RECOMPUTED COMPARISON
# ============================================================

def compare_historical_metrics(
    recomputed_rows: List[Dict[str, Any]],
    historical_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    comparison = []

    metric_aliases = {
        "accuracy":
            {"accuracy", "acc", "accuracy_mean"},

        "f1":
            {
                "f1",
                "f1_score",
                "macro_f1",
                "f1_mean",
            },

        "roc_auc":
            {
                "auc",
                "roc_auc",
                "macro_auc",
                "auc_mean",
            },
    }

    for rec in recomputed_rows:

        if rec.get(
            "status"
        ) != "RECOMPUTED":
            continue

        exp = rec.get(
            "experiment",
            ""
        )

        dataset = rec.get(
            "dataset_hint",
            ""
        )

        for recomputed_metric, aliases in metric_aliases.items():

            value = rec.get(
                recomputed_metric
            )

            if value in (
                "",
                None,
            ):
                continue

            historical_candidates = []

            for hist in historical_rows:

                if (
                    exp
                    and hist.get(
                        "experiment"
                    )
                    and hist["experiment"]
                    != exp
                ):
                    continue

                if hist[
                    "metric_name"
                ] not in aliases:
                    continue

                historical_candidates.append(
                    hist
                )

            if not historical_candidates:

                comparison.append({
                    "experiment":
                        exp,
                    "dataset_hint":
                        dataset,
                    "metric":
                        recomputed_metric,
                    "recomputed_value":
                        value,
                    "historical_value":
                        "",
                    "difference":
                        "",
                    "match_status":
                        "NO_HISTORICAL_VALUE",
                    "historical_source":
                        "",
                })

                continue

            for hist in historical_candidates:

                historical_value = (
                    hist[
                        "metric_value"
                    ]
                )

                difference = (
                    float(value)
                    - float(
                        historical_value
                    )
                )

                comparison.append({
                    "experiment":
                        exp,
                    "dataset_hint":
                        dataset,
                    "metric":
                        recomputed_metric,
                    "recomputed_value":
                        value,
                    "historical_value":
                        historical_value,
                    "difference":
                        difference,
                    "absolute_difference":
                        abs(difference),
                    "match_status":
                        (
                            "MATCH"
                            if abs(difference)
                            <= METRIC_COMPARISON_TOLERANCE
                            else
                            "DIFFERENT"
                        ),
                    "historical_source":
                        hist[
                            "relative_path"
                        ],
                    "historical_context":
                        hist.get(
                            "context",
                            ""
                        ),
                })

    return comparison


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:

    print("=" * 84)
    print("HFAGM - RECONSTRUCT AND RECOMPUTE PRIMARY METRICS")
    print("=" * 84)

    if not PROJECT_ROOT.exists():

        print(
            "\nERROR: PROJECT_ROOT does not exist:"
        )
        print(PROJECT_ROOT)

        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANONICAL_PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = list(
        iter_project_files()
    )

    print(
        f"\nProject files scanned: "
        f"{len(files):,}"
    )

    # --------------------------------------------------------
    # A. Inventory relevant source files
    # --------------------------------------------------------

    source_inventory = []

    historical_metric_rows = []

    array_candidates: List[
        ArrayCandidate
    ] = []

    code_evidence = []

    prediction_source_candidates = []

    candidate_csv_prediction_tables = []

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

        ext = path.suffix.lower()

        exp = identify_experiment(
            path
        )

        dataset = identify_dataset_hint(
            path
        )

        relevant = (
            bool(exp)
            or ext in {
                ".csv",
                ".npy",
                ".npz",
                ".json",
                ".py",
                ".pkl",
                ".joblib",
                ".pt",
                ".pth",
                ".ckpt",
            }
        )

        if relevant:

            source_inventory.append({
                "relative_path":
                    relative_path(path),
                "extension":
                    ext,
                "experiment":
                    exp,
                "dataset_hint":
                    dataset,
                "size_bytes":
                    safe_size(path),
                "modified_time":
                    safe_mtime(path),
                "is_raw_asset":
                    int(
                        is_raw_asset_path(
                            path
                        )
                    ),
            })

        # Historical metric extraction.
        if ext == ".csv":

            historical_metric_rows.extend(
                extract_existing_metric_rows(
                    path
                )
            )

            csv_arrays = (
                discover_csv_arrays(
                    path
                )
            )

            known_arrays = (
                discover_known_label_files(
                    path
                )
            )

            array_candidates.extend(
                csv_arrays
            )

            array_candidates.extend(
                known_arrays
            )

            if any(
                candidate_role(c)
                == "y_true"
                for c in csv_arrays
            ) and any(
                candidate_role(c)
                in {
                    "y_pred",
                    "y_prob",
                }
                for c in csv_arrays
            ):
                candidate_csv_prediction_tables.append(
                    path
                )

        # NumPy arrays.
        elif ext in {
            ".npy",
            ".npz",
        }:

            array_candidates.extend(
                discover_numpy_arrays(
                    path
                )
            )

        # Python code.
        elif ext == ".py":

            code_rows = (
                inspect_prediction_code(
                    path
                )
            )

            code_evidence.extend(
                code_rows
            )

            if code_rows:

                prediction_source_candidates.append({
                    "relative_path":
                        relative_path(path),
                    "experiment":
                        exp,
                    "dataset_hint":
                        dataset,
                    "evidence_categories":
                        "; ".join(
                            sorted({
                                r[
                                    "category"
                                ]
                                for r
                                in code_rows
                            })
                        ),
                })

    # --------------------------------------------------------
    # B. Save source inventories
    # --------------------------------------------------------

    write_csv(
        OUTPUT_DIR /
        "source_file_inventory.csv",
        source_inventory,
    )

    write_csv(
        OUTPUT_DIR /
        "experiment_metric_inventory.csv",
        historical_metric_rows,
    )

    write_csv(
        OUTPUT_DIR /
        "prediction_source_candidates.csv",
        prediction_source_candidates,
    )

    write_csv(
        OUTPUT_DIR /
        "prediction_code_evidence.csv",
        code_evidence,
    )

    # --------------------------------------------------------
    # C. Array candidate summary
    # --------------------------------------------------------

    array_candidate_rows = []

    for c in array_candidates:

        arr = np.asarray(c.array)

        role = candidate_role(c)

        shape = list(arr.shape)

        flat_len = (
            int(arr.size)
            if arr.ndim == 1
            else int(arr.shape[0])
        )

        unique_preview = ""

        if arr.size <= 100000:

            try:

                flattened = (
                    arr.reshape(-1)
                )

                unique = np.unique(
                    flattened
                )

                unique_preview = "; ".join(
                    str(x)
                    for x
                    in unique[:20]
                )

            except Exception:
                pass

        array_candidate_rows.append({
            "role":
                role,
            "variable_name":
                c.variable_name,
            "relative_path":
                relative_path(
                    c.path
                ),
            "source_type":
                c.source_type,
            "experiment":
                c.experiment,
            "dataset_hint":
                c.dataset_hint,
            "shape":
                str(shape),
            "n_first_dimension":
                flat_len,
            "dtype":
                str(arr.dtype),
            "unique_preview":
                unique_preview,
        })

    write_csv(
        OUTPUT_DIR /
        "array_candidates.csv",
        array_candidate_rows,
    )

    # --------------------------------------------------------
    # D. Build compatibility registry
    # --------------------------------------------------------

    reconstruction_candidates = (
        build_reconstruction_candidates(
            array_candidates
        )
    )

    write_csv(
        OUTPUT_DIR /
        "reconstruction_candidates.csv",
        reconstruction_candidates,
    )

    # --------------------------------------------------------
    # E. Recompute same-table metrics
    # --------------------------------------------------------

    recomputed_rows = []

    canonical_files = []

    for path in sorted(
        set(
            candidate_csv_prediction_tables
        )
    ):

        results, canonical = (
            recompute_from_prediction_table(
                path
            )
        )

        recomputed_rows.extend(
            results
        )

        if canonical is not None:

            safe_name = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                relative_path(path)
            )

            canonical_path = (
                CANONICAL_PREDICTION_DIR
                / f"{safe_name}.canonical_predictions.csv"
            )

            canonical.to_csv(
                canonical_path,
                index=False,
            )

            canonical_files.append({
                "source":
                    relative_path(path),
                "canonical_prediction_file":
                    relative_path(
                        canonical_path
                    ),
                "n_samples":
                    len(canonical),
            })

    # --------------------------------------------------------
    # F. Try high-confidence cross-file matches
    # --------------------------------------------------------
    #
    # This is deliberately conservative:
    # only candidates with same length and strong path/experiment
    # compatibility are considered.
    # --------------------------------------------------------

    y_true_candidates = [
        c
        for c in array_candidates
        if candidate_role(c)
        == "y_true"
    ]

    y_pred_candidates = [
        c
        for c in array_candidates
        if candidate_role(c)
        == "y_pred"
    ]

    y_prob_candidates = [
        c
        for c in array_candidates
        if candidate_role(c)
        == "y_prob"
    ]

    used_signatures = set()

    for y_true in y_true_candidates:

        possible_pred = []

        for candidate in (
            y_pred_candidates
            + y_prob_candidates
        ):

            score = compatibility_score(
                y_true,
                candidate,
            )

            true_n = len(
                np.asarray(
                    y_true.array
                ).reshape(-1)
            )

            candidate_array = (
                np.asarray(
                    candidate.array
                )
            )

            candidate_n = (
                candidate_array.shape[0]
            )

            if true_n != candidate_n:
                continue

            possible_pred.append(
                (score, candidate)
            )

        possible_pred.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        # Need strong evidence to align different files.
        # Score >= 15 normally means matching length plus same experiment
        # and/or same parent location.
        for score, candidate in possible_pred:

            if score < 15:
                continue

            signature = (
                str(y_true.path),
                str(candidate.path),
                candidate.variable_name,
            )

            if signature in used_signatures:
                continue

            used_signatures.add(
                signature
            )

            role = candidate_role(
                candidate
            )

            y_pred = (
                candidate.array
                if role == "y_pred"
                else None
            )

            y_prob = (
                candidate.array
                if role == "y_prob"
                else None
            )

            metrics, reason = (
                recompute_binary_metrics(
                    y_true=y_true.array,
                    y_pred=y_pred,
                    y_prob=y_prob,
                )
            )

            if metrics is None:

                recomputed_rows.append({
                    "status":
                        "FAILED",
                    "reconstruction_type":
                        "cross_file",
                    "compatibility_score":
                        score,
                    "experiment":
                        candidate.experiment
                        or y_true.experiment,
                    "dataset_hint":
                        candidate.dataset_hint
                        or y_true.dataset_hint,
                    "y_true_source":
                        relative_path(
                            y_true.path
                        ),
                    "prediction_source":
                        relative_path(
                            candidate.path
                        ),
                    "prediction_role":
                        role,
                    "reason":
                        reason,
                })

                continue

            recomputed_rows.append({
                "status":
                    "RECOMPUTED",
                "reconstruction_type":
                    "cross_file_high_confidence",
                "compatibility_score":
                    score,
                "experiment":
                    candidate.experiment
                    or y_true.experiment,
                "dataset_hint":
                    candidate.dataset_hint
                    or y_true.dataset_hint,
                "y_true_source":
                    relative_path(
                        y_true.path
                    ),
                "prediction_source":
                    relative_path(
                        candidate.path
                    ),
                "prediction_role":
                    role,
                **metrics,
            })

            true_clean = (
                clean_binary_labels(
                    y_true.array
                )
            )

            if role == "y_pred":

                pred_clean = (
                    clean_binary_predictions(
                        candidate.array
                    )
                )

                prob_clean = None

            else:

                prob_clean = (
                    clean_probability_array(
                        candidate.array
                    )
                )

                pred_clean = (
                    prob_clean
                    >= DEFAULT_BINARY_THRESHOLD
                ).astype(int)

            if (
                true_clean is not None
                and pred_clean is not None
            ):

                canonical = pd.DataFrame({
                    "row_id":
                        np.arange(
                            len(true_clean)
                        ),
                    "y_true":
                        true_clean,
                    "y_pred":
                        pred_clean,
                })

                if prob_clean is not None:
                    canonical[
                        "y_prob"
                    ] = prob_clean

                file_name = (
                    "crossfile_"
                    + re.sub(
                        r"[^A-Za-z0-9_.-]+",
                        "_",
                        (
                            relative_path(
                                candidate.path
                            )
                        ),
                    )
                    + ".canonical_predictions.csv"
                )

                canonical_path = (
                    CANONICAL_PREDICTION_DIR
                    / file_name
                )

                canonical.to_csv(
                    canonical_path,
                    index=False,
                )

                canonical_files.append({
                    "source":
                        relative_path(
                            candidate.path
                        ),
                    "true_label_source":
                        relative_path(
                            y_true.path
                        ),
                    "canonical_prediction_file":
                        relative_path(
                            canonical_path
                        ),
                    "n_samples":
                        len(canonical),
                })

            # Do not pair the same y_true with many ambiguous arrays.
            break

    # --------------------------------------------------------
    # G. Save recomputed metrics
    # --------------------------------------------------------

    write_csv(
        OUTPUT_DIR /
        "recomputed_primary_metrics.csv",
        recomputed_rows,
        preferred_columns=[
            "status",
            "reconstruction_type",
            "experiment",
            "dataset_hint",
            "n_samples",
            "n_negative",
            "n_positive",
            "accuracy",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "threshold",
            "auc_source",
            "y_true_source",
            "prediction_source",
            "relative_path",
            "reason",
        ],
    )

    write_csv(
        OUTPUT_DIR /
        "canonical_prediction_files.csv",
        canonical_files,
    )

    # --------------------------------------------------------
    # H. Confusion matrix table
    # --------------------------------------------------------

    confusion_rows = []

    for row in recomputed_rows:

        if row.get(
            "status"
        ) != "RECOMPUTED":
            continue

        if not all(
            key in row
            for key in [
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ):
            continue

        confusion_rows.append({
            "experiment":
                row.get(
                    "experiment",
                    ""
                ),
            "dataset_hint":
                row.get(
                    "dataset_hint",
                    ""
                ),
            "reconstruction_type":
                row.get(
                    "reconstruction_type",
                    ""
                ),
            "n_samples":
                row.get(
                    "n_samples",
                    ""
                ),
            "TN":
                row["tn"],
            "FP":
                row["fp"],
            "FN":
                row["fn"],
            "TP":
                row["tp"],
            "accuracy":
                row.get(
                    "accuracy",
                    ""
                ),
            "f1":
                row.get(
                    "f1",
                    ""
                ),
            "roc_auc":
                row.get(
                    "roc_auc",
                    ""
                ),
        })

    write_csv(
        OUTPUT_DIR /
        "confusion_matrices.csv",
        confusion_rows,
    )

    # --------------------------------------------------------
    # I. Historical comparison
    # --------------------------------------------------------

    historical_comparison = (
        compare_historical_metrics(
            recomputed_rows,
            historical_metric_rows,
        )
    )

    write_csv(
        OUTPUT_DIR /
        "historical_vs_recomputed.csv",
        historical_comparison,
    )

    # --------------------------------------------------------
    # J. Determine experiment reconstruction status
    # --------------------------------------------------------

    known_experiments = set()

    for row in source_inventory:
        if row[
            "experiment"
        ]:
            known_experiments.add(
                row[
                    "experiment"
                ]
            )

    for row in historical_metric_rows:
        if row[
            "experiment"
        ]:
            known_experiments.add(
                row[
                    "experiment"
                ]
            )

    unreconstructable = []

    for exp in sorted(
        known_experiments
    ):

        historical_count = sum(
            1
            for row
            in historical_metric_rows
            if row[
                "experiment"
            ] == exp
        )

        recomputed_count = sum(
            1
            for row
            in recomputed_rows
            if (
                row.get(
                    "experiment"
                ) == exp
                and row.get(
                    "status"
                ) == "RECOMPUTED"
            )
        )

        code_count = sum(
            1
            for row
            in code_evidence
            if row[
                "experiment"
            ] == exp
        )

        array_count = sum(
            1
            for c
            in array_candidates
            if c.experiment == exp
        )

        if recomputed_count > 0:
            status = "RECOMPUTED"
            recommendation = (
                "Use canonical prediction table and recomputed metrics."
            )

        elif array_count > 0:
            status = "PARTIAL"
            recommendation = (
                "Relevant arrays exist but could not be safely aligned. "
                "Inspect reconstruction_candidates.csv."
            )

        elif code_count > 0:
            status = "REGENERATION_POSSIBLE"
            recommendation = (
                "No saved predictions found, but prediction-generating "
                "code exists. Regenerate predictions using the existing "
                "trained model/configuration without changing the model."
            )

        else:
            status = "CANNOT_RECOMPUTE"
            recommendation = (
                "Only summary metrics appear to remain. "
                "Underlying predictions must be regenerated from the "
                "original experiment or the result cannot be independently verified."
            )

        unreconstructable.append({
            "experiment":
                exp,
            "historical_metric_rows":
                historical_count,
            "array_candidates":
                array_count,
            "prediction_code_evidence":
                code_count,
            "recomputed_metric_sets":
                recomputed_count,
            "status":
                status,
            "recommendation":
                recommendation,
        })

    write_csv(
        OUTPUT_DIR /
        "experiment_reconstruction_status.csv",
        unreconstructable,
    )

    # ========================================================
    # K. SUMMARY
    # ========================================================

    successful = [
        r
        for r in recomputed_rows
        if r.get(
            "status"
        ) == "RECOMPUTED"
    ]

    failed = [
        r
        for r in recomputed_rows
        if r.get(
            "status"
        ) == "FAILED"
    ]

    summary_lines = [
        "=" * 84,
        "HFAGM - PRIMARY METRIC RECONSTRUCTION AUDIT",
        "=" * 84,
        "",
        f"Generated: "
        f"{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Project root:",
        str(PROJECT_ROOT),
        "",
        "SOURCE DISCOVERY",
        "-" * 84,
        f"Relevant source files inventoried: "
        f"{len(source_inventory):,}",
        f"Historical metric values extracted: "
        f"{len(historical_metric_rows):,}",
        f"Candidate label/prediction/probability arrays: "
        f"{len(array_candidates):,}",
        f"Prediction-generating code evidence rows: "
        f"{len(code_evidence):,}",
        "",
        "RECONSTRUCTION",
        "-" * 84,
        f"Compatible reconstruction candidate pairs: "
        f"{len(reconstruction_candidates):,}",
        f"Successfully recomputed metric sets: "
        f"{len(successful):,}",
        f"Failed reconstruction attempts: "
        f"{len(failed):,}",
        f"Canonical prediction tables created: "
        f"{len(canonical_files):,}",
        "",
        "EXPERIMENT STATUS",
        "-" * 84,
    ]

    for row in unreconstructable:

        summary_lines.extend([
            f"Experiment: {row['experiment']}",
            f"  Historical metric rows: "
            f"{row['historical_metric_rows']}",
            f"  Array candidates: "
            f"{row['array_candidates']}",
            f"  Prediction-code evidence: "
            f"{row['prediction_code_evidence']}",
            f"  Recomputed metric sets: "
            f"{row['recomputed_metric_sets']}",
            f"  Status: "
            f"{row['status']}",
            f"  Recommendation: "
            f"{row['recommendation']}",
            "",
        ])

    summary_lines.extend([
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 84,
        "",
        "1. Historical metric CSVs are treated as historical evidence only.",
        "2. Accuracy/F1/AUC are considered independently verified only when",
        "   they can be recomputed from actual y_true/y_pred/y_prob arrays.",
        "3. Metrics from New_EXPs, New_EXP2, and New_EXP3 are NOT assumed",
        "   to represent the same model, dataset, or experiment.",
        "4. ROC-AUC is recomputed only from probabilities/scores. It is not",
        "   reconstructed from hard class predictions.",
        "5. No model is retrained by this script.",
        "6. No missing prediction is fabricated from summary statistics.",
        "7. Cross-file array matching requires strong structural evidence;",
        "   simple equality of sample count is not considered sufficient.",
        "",
        "FILES TO REVIEW NEXT",
        "-" * 84,
        "1. experiment_reconstruction_status.csv",
        "2. recomputed_primary_metrics.csv",
        "3. historical_vs_recomputed.csv",
        "4. reconstruction_candidates.csv",
        "5. array_candidates.csv",
        "6. prediction_source_candidates.csv",
        "7. prediction_code_evidence.csv",
        "8. confusion_matrices.csv",
        "9. canonical_prediction_files.csv",
        "10. experiment_metric_inventory.csv",
        "",
        "=" * 84,
    ])

    summary_path = (
        OUTPUT_DIR
        / "primary_metrics_audit_summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 84)
    print("SCRIPT 02 COMPLETE")
    print("=" * 84)

    print(
        f"\nSuccessful metric recomputations: "
        f"{len(successful)}"
    )

    print(
        f"Canonical prediction tables: "
        f"{len(canonical_files)}"
    )

    print(
        "\nResults written to:"
    )
    print(OUTPUT_DIR)

    print(
        "\nSend these files first:"
    )
    print(
        OUTPUT_DIR
        / "primary_metrics_audit_summary.txt"
    )
    print(
        OUTPUT_DIR
        / "experiment_reconstruction_status.csv"
    )
    print(
        OUTPUT_DIR
        / "recomputed_primary_metrics.csv"
    )
    print(
        OUTPUT_DIR
        / "prediction_source_candidates.csv"
    )
    print(
        OUTPUT_DIR
        / "reconstruction_candidates.csv"
    )


if __name__ == "__main__":
    main()