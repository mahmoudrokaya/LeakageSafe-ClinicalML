"""
02C_verify_split_and_leakage.py
================================

Purpose
-------
Forensically verify whether the preserved HFAGM COVID-clinical train/test
split is genuinely leakage-safe.

This script investigates:

1. Raw/preprocessed/balanced dataset sizes and schemas.
2. Whether train/test rows overlap exactly.
3. Whether train/test rows overlap after rounding.
4. Near-duplicate train/test records.
5. Whether y appears directly or indirectly among X features.
6. Whether preprocessing was performed before or after splitting.
7. Whether balancing was performed before or after splitting.
8. Whether the scaler was fitted on training data only.
9. Whether synthetic generation may have used test records.
10. Whether the saved ensemble model appears to have been trained on
    the preserved training set.
11. Whether suspicious target-derived columns exist.
12. Whether the perfect 40/40 test performance can be considered
    leakage-safe based on available project evidence.

IMPORTANT
---------
This script DOES NOT:
- retrain models;
- change train/test files;
- rebalance datasets;
- rescale data;
- create synthetic samples;
- overwrite original outputs;
- assume leakage when evidence is absent.

Outputs
-------
outputs/revision_primary_metrics/split_leakage_audit/

    dataset_inventory.csv
    schema_comparison.csv
    train_test_exact_overlap.csv
    train_test_near_duplicates.csv
    target_feature_audit.csv
    feature_target_correlations.csv
    preprocessing_code_evidence.csv
    split_code_evidence.csv
    balancing_code_evidence.csv
    scaling_code_evidence.csv
    synthetic_generation_code_evidence.csv
    training_code_evidence.csv
    file_timestamp_audit.csv
    leakage_findings.csv
    leakage_verdict.csv
    split_leakage_audit_summary.txt

Requirements
------------
numpy
pandas
scikit-learn
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
    / "split_leakage_audit"
)

# Preserved split
X_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "X_train_scaled.csv"
)

X_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "X_test_scaled.csv"
)

Y_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "y_train.csv"
)

Y_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "y_test.csv"
)

# Main data candidates
RAW_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "covid_clinical.csv"
)

RAW_XLSX_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "covid_clinical.xlsx"
)

PREPROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "covid_clinical_preprocessed.csv"
)

BALANCED_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "covid_clinical_balanced.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifiers"
    / "ensemble_model.pkl"
)

# Code files known from prior audit
KNOWN_CODE_FILES = [
    PROJECT_ROOT / "preprocess_covid_clinical.py",
    PROJECT_ROOT / "split_and_standardize.py",

    PROJECT_ROOT
    / "models"
    / "classifiers"
    / "train_ensemble.py",

    PROJECT_ROOT
    / "training"
    / "ensemble"
    / "train_hfagm.py",

    PROJECT_ROOT
    / "training"
    / "hfagm"
    / "train_hfagm.py",
]

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "revision_audit",
    "revision_audit_v2",
    "revision_primary_metrics",
}

# Number of decimal places for rounded-overlap checks.
ROUNDING_LEVELS = [8, 6, 4, 3]

# A very small standardized Euclidean distance can indicate
# exact/almost-exact cross-split duplication.
NEAR_DUPLICATE_DISTANCE_THRESHOLDS = [
    1e-12,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
]

MAX_TEXT_FILE_BYTES = 20 * 1024 * 1024


# ============================================================
# 2. TERMINOLOGY
# ============================================================

TARGET_TERMS = {
    "status",
    "target",
    "outcome",
    "label",
    "class",
    "death",
    "deceased",
    "mortality",
    "survival",
    "survived",
    "recovered",
    "recovery",
    "y",
}

SPLIT_PATTERNS = {
    "train_test_split":
        r"\btrain_test_split\s*\(",

    "stratify":
        r"\bstratify\s*=",

    "test_size":
        r"\btest_size\s*=",

    "random_state":
        r"\brandom_state\s*=",

    "shuffle":
        r"\bshuffle\s*=",

    "X_train":
        r"\bX_train\b",

    "X_test":
        r"\bX_test\b",

    "y_train":
        r"\by_train\b",

    "y_test":
        r"\by_test\b",
}

SCALING_PATTERNS = {
    "StandardScaler":
        r"\bStandardScaler\b",

    "MinMaxScaler":
        r"\bMinMaxScaler\b",

    "RobustScaler":
        r"\bRobustScaler\b",

    "fit":
        r"\.fit\s*\(",

    "fit_transform":
        r"\.fit_transform\s*\(",

    "transform":
        r"\.transform\s*\(",

    "scaler":
        r"\bscaler\b",
}

BALANCING_PATTERNS = {
    "SMOTE":
        r"\bSMOTE\b",

    "ADASYN":
        r"\bADASYN\b",

    "RandomOverSampler":
        r"\bRandomOverSampler\b",

    "RandomUnderSampler":
        r"\bRandomUnderSampler\b",

    "resample":
        r"\bresample\s*\(",

    "oversample":
        r"oversampl",

    "undersample":
        r"undersampl",

    "balanced":
        r"\bbalanc",
}

SYNTHETIC_PATTERNS = {
    "synthetic":
        r"\bsynthetic\b",

    "generator":
        r"\bgenerator\b",

    "generate":
        r"\bgenerat",

    "GAN":
        r"\bGAN\b",

    "VAE":
        r"\bVAE\b",

    "diffusion":
        r"\bdiffusion\b",

    "sample":
        r"\.sample\s*\(",
}

TRAINING_PATTERNS = {
    "model_fit":
        r"\.fit\s*\(",

    "VotingClassifier":
        r"\bVotingClassifier\b",

    "LogisticRegression":
        r"\bLogisticRegression\b",

    "RandomForestClassifier":
        r"\bRandomForestClassifier\b",

    "GradientBoostingClassifier":
        r"\bGradientBoostingClassifier\b",

    "joblib_dump":
        r"\bjoblib\.dump\s*\(",

    "ensemble_model":
        r"\bensemble_model\b",

    "X_train":
        r"\bX_train\b",

    "y_train":
        r"\by_train\b",
}


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def normalize_name(value: Any) -> str:

    text = str(value).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


def relative_path(path: Path) -> str:

    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )
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
        ).isoformat(
            timespec="seconds"
        )
    except Exception:
        return ""


def sha256_file(path: Path) -> str:

    try:

        h = hashlib.sha256()

        with path.open("rb") as f:

            while True:

                chunk = f.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return ""


def safe_read_csv(
    path: Path
) -> Optional[pd.DataFrame]:

    if not path.exists():
        return None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]:

        try:

            return pd.read_csv(
                path,
                encoding=encoding
            )

        except Exception:
            continue

    return None


def safe_read_excel(
    path: Path
) -> Optional[pd.DataFrame]:

    if not path.exists():
        return None

    try:
        return pd.read_excel(
            path
        )
    except Exception:
        return None


def safe_read_text(
    path: Path
) -> str:

    if not path.exists():
        return ""

    if safe_size(path) > MAX_TEXT_FILE_BYTES:
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
                errors="ignore"
            )

        except Exception:
            continue

    return ""


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    preferred_columns: Optional[
        List[str]
    ] = None,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    keys = set()

    for row in rows:
        keys.update(
            row.keys()
        )

    columns = []

    if preferred_columns:
        columns.extend(
            preferred_columns
        )

    for key in sorted(keys):

        if key not in columns:
            columns.append(key)

    if not columns:

        path.write_text(
            "",
            encoding="utf-8"
        )

        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# 4. PROJECT PYTHON FILE DISCOVERY
# ============================================================

def iter_python_files() -> Iterable[Path]:

    for root, dirs, files in os.walk(
        PROJECT_ROOT
    ):

        dirs[:] = [
            d
            for d in dirs
            if d.lower()
            not in EXCLUDED_DIRS
        ]

        for filename in files:

            if not filename.lower().endswith(
                ".py"
            ):
                continue

            yield (
                Path(root)
                / filename
            )


# ============================================================
# 5. DATASET INVENTORY
# ============================================================

def identify_target_column(
    df: pd.DataFrame
) -> str:

    normalized = {
        normalize_name(col): col
        for col in df.columns
    }

    priority = [
        "status",
        "outcome",
        "target",
        "label",
        "class",
    ]

    for item in priority:

        if item in normalized:
            return normalized[item]

    return ""


def inventory_dataframe(
    label: str,
    path: Path,
    df: Optional[pd.DataFrame],
) -> Dict[str, Any]:

    if df is None:

        return {
            "dataset":
                label,

            "relative_path":
                relative_path(path),

            "exists":
                int(
                    path.exists()
                ),

            "readable":
                0,
        }

    target_col = (
        identify_target_column(
            df
        )
    )

    result = {
        "dataset":
            label,

        "relative_path":
            relative_path(path),

        "exists":
            1,

        "readable":
            1,

        "rows":
            len(df),

        "columns":
            df.shape[1],

        "target_column":
            target_col,

        "column_names":
            json.dumps(
                [
                    str(x)
                    for x
                    in df.columns
                ],
                ensure_ascii=False
            ),

        "duplicate_rows":
            int(
                df.duplicated().sum()
            ),

        "file_size_bytes":
            safe_size(path),

        "modified_time":
            safe_mtime(path),

        "sha256":
            sha256_file(path),
    }

    if target_col:

        counts = (
            df[target_col]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        result[
            "target_counts"
        ] = json.dumps(
            counts,
            default=str
        )

    else:

        result[
            "target_counts"
        ] = ""

    return result


# ============================================================
# 6. LOAD PRESERVED SPLIT
# ============================================================

def load_single_label_vector(
    path: Path
) -> Tuple[
    Optional[np.ndarray],
    str
]:

    df = safe_read_csv(
        path
    )

    if df is None or df.empty:
        return None, ""

    if df.shape[1] == 1:

        col = df.columns[0]

    else:

        col = (
            identify_target_column(
                df
            )
        )

        if not col:
            return None, ""

    y = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    if y.isna().any():
        return None, col

    return (
        y.to_numpy(),
        col
    )


def numeric_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:

    result = df.copy()

    for col in result.columns:

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    return result


# ============================================================
# 7. SCHEMA COMPARISON
# ============================================================

def schema_comparison(
    datasets: Dict[
        str,
        Optional[pd.DataFrame]
    ]
) -> List[Dict[str, Any]]:

    rows = []

    names = list(
        datasets.keys()
    )

    for i, left_name in enumerate(
        names
    ):

        left = datasets[
            left_name
        ]

        if left is None:
            continue

        left_cols = [
            normalize_name(c)
            for c
            in left.columns
        ]

        for right_name in names[
            i + 1:
        ]:

            right = datasets[
                right_name
            ]

            if right is None:
                continue

            right_cols = [
                normalize_name(c)
                for c
                in right.columns
            ]

            left_set = set(
                left_cols
            )

            right_set = set(
                right_cols
            )

            rows.append({
                "dataset_a":
                    left_name,

                "dataset_b":
                    right_name,

                "n_columns_a":
                    len(left_cols),

                "n_columns_b":
                    len(right_cols),

                "common_columns":
                    len(
                        left_set
                        & right_set
                    ),

                "only_a":
                    "; ".join(
                        sorted(
                            left_set
                            - right_set
                        )
                    ),

                "only_b":
                    "; ".join(
                        sorted(
                            right_set
                            - left_set
                        )
                    ),

                "exact_same_column_set":
                    int(
                        left_set
                        == right_set
                    ),

                "exact_same_order":
                    int(
                        left_cols
                        == right_cols
                    ),
            })

    return rows


# ============================================================
# 8. EXACT TRAIN/TEST OVERLAP
# ============================================================

def row_signature(
    row: np.ndarray,
    decimals: Optional[int] = None,
) -> Tuple:

    arr = np.asarray(
        row,
        dtype=float
    )

    if decimals is not None:

        arr = np.round(
            arr,
            decimals
        )

    return tuple(
        arr.tolist()
    )


def exact_overlap_audit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> List[Dict[str, Any]]:

    if list(
        X_train.columns
    ) != list(
        X_test.columns
    ):

        raise RuntimeError(
            "X_train and X_test feature columns differ."
        )

    train_numeric = (
        numeric_dataframe(
            X_train
        )
    )

    test_numeric = (
        numeric_dataframe(
            X_test
        )
    )

    if (
        train_numeric.isna().any().any()
        or
        test_numeric.isna().any().any()
    ):

        raise RuntimeError(
            "Train/test contain non-numeric or missing values."
        )

    rows = []

    for decimals in [
        None,
        *ROUNDING_LEVELS
    ]:

        train_map = defaultdict(
            list
        )

        for i, row in enumerate(
            train_numeric.to_numpy()
        ):

            sig = row_signature(
                row,
                decimals
            )

            train_map[
                sig
            ].append(i)

        matches = []

        for j, row in enumerate(
            test_numeric.to_numpy()
        ):

            sig = row_signature(
                row,
                decimals
            )

            if sig in train_map:

                for train_idx in train_map[
                    sig
                ]:

                    matches.append(
                        (
                            train_idx,
                            j
                        )
                    )

        rows.append({
            "comparison":
                (
                    "exact"
                    if decimals is None
                    else
                    f"rounded_{decimals}_decimals"
                ),

            "n_train":
                len(
                    train_numeric
                ),

            "n_test":
                len(
                    test_numeric
                ),

            "overlap_pairs":
                len(matches),

            "unique_test_rows_overlapping":
                len({
                    x[1]
                    for x in matches
                }),

            "unique_train_rows_overlapping":
                len({
                    x[0]
                    for x in matches
                }),

            "match_pairs_preview":
                json.dumps(
                    matches[:100]
                ),
        })

    return rows


# ============================================================
# 9. NEAR-DUPLICATE CROSS-SPLIT AUDIT
# ============================================================

def near_duplicate_audit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> List[Dict[str, Any]]:

    train = numeric_dataframe(
        X_train
    ).to_numpy(
        dtype=float
    )

    test = numeric_dataframe(
        X_test
    ).to_numpy(
        dtype=float
    )

    if (
        not np.isfinite(
            train
        ).all()
        or
        not np.isfinite(
            test
        ).all()
    ):

        raise RuntimeError(
            "Train/test arrays contain NaN/Inf."
        )

    distances = pairwise_distances(
        test,
        train,
        metric="euclidean"
    )

    nearest_train = (
        np.argmin(
            distances,
            axis=1
        )
    )

    nearest_distance = (
        np.min(
            distances,
            axis=1
        )
    )

    rows = []

    for test_idx in range(
        len(test)
    ):

        rows.append({
            "test_index":
                test_idx,

            "nearest_train_index":
                int(
                    nearest_train[
                        test_idx
                    ]
                ),

            "euclidean_distance":
                float(
                    nearest_distance[
                        test_idx
                    ]
                ),
        })

    return rows


def summarize_near_duplicates(
    near_rows: List[
        Dict[str, Any]
    ]
) -> List[Dict[str, Any]]:

    distances = np.asarray(
        [
            row[
                "euclidean_distance"
            ]
            for row
            in near_rows
        ],
        dtype=float
    )

    output = []

    for threshold in (
        NEAR_DUPLICATE_DISTANCE_THRESHOLDS
    ):

        output.append({
            "distance_threshold":
                threshold,

            "test_cases_at_or_below":
                int(
                    np.sum(
                        distances
                        <= threshold
                    )
                ),

            "fraction_of_test":
                float(
                    np.mean(
                        distances
                        <= threshold
                    )
                ),
        })

    return output


# ============================================================
# 10. TARGET LEAKAGE FEATURE AUDIT
# ============================================================

def target_feature_audit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]]
]:

    audit_rows = []

    corr_rows = []

    normalized_columns = {
        normalize_name(col): col
        for col in X_train.columns
    }

    for normalized, original in (
        normalized_columns.items()
    ):

        suspicious = False

        reasons = []

        if normalized in TARGET_TERMS:

            suspicious = True

            reasons.append(
                "feature name equals target-related term"
            )

        for term in TARGET_TERMS:

            if (
                len(term) >= 4
                and term in normalized
            ):

                suspicious = True

                reasons.append(
                    f"contains target-related token '{term}'"
                )

        audit_rows.append({
            "feature":
                original,

            "normalized_feature":
                normalized,

            "name_suspicious":
                int(
                    suspicious
                ),

            "reason":
                "; ".join(
                    sorted(
                        set(reasons)
                    )
                ),
        })

    # Correlation with target.
    Xtr = numeric_dataframe(
        X_train
    )

    Xte = numeric_dataframe(
        X_test
    )

    for col in Xtr.columns:

        train_values = (
            Xtr[col]
            .to_numpy(
                dtype=float
            )
        )

        test_values = (
            Xte[col]
            .to_numpy(
                dtype=float
            )
        )

        def corr(
            x: np.ndarray,
            y: np.ndarray
        ) -> float:

            if (
                len(
                    np.unique(x)
                ) <= 1
                or
                len(
                    np.unique(y)
                ) <= 1
            ):
                return float("nan")

            return float(
                np.corrcoef(
                    x,
                    y
                )[0, 1]
            )

        train_corr = corr(
            train_values,
            y_train
        )

        test_corr = corr(
            test_values,
            y_test
        )

        corr_rows.append({
            "feature":
                col,

            "train_target_correlation":
                (
                    train_corr
                    if np.isfinite(
                        train_corr
                    )
                    else ""
                ),

            "test_target_correlation":
                (
                    test_corr
                    if np.isfinite(
                        test_corr
                    )
                    else ""
                ),

            "abs_train_correlation":
                (
                    abs(
                        train_corr
                    )
                    if np.isfinite(
                        train_corr
                    )
                    else ""
                ),

            "abs_test_correlation":
                (
                    abs(
                        test_corr
                    )
                    if np.isfinite(
                        test_corr
                    )
                    else ""
                ),
        })

    corr_rows.sort(
        key=lambda r: (
            -float(
                r[
                    "abs_train_correlation"
                ]
            )
            if r[
                "abs_train_correlation"
            ] != ""
            else 0
        )
    )

    return (
        audit_rows,
        corr_rows
    )


# ============================================================
# 11. CODE FORENSICS
# ============================================================

def inspect_code_patterns(
    path: Path,
    patterns: Dict[
        str,
        str
    ],
    category: str,
) -> List[Dict[str, Any]]:

    text = safe_read_text(
        path
    )

    if not text:
        return []

    rows = []

    lines = (
        text.splitlines()
    )

    for line_no, line in enumerate(
        lines,
        start=1
    ):

        for pattern_name, regex in (
            patterns.items()
        ):

            if re.search(
                regex,
                line,
                flags=re.IGNORECASE
            ):

                rows.append({
                    "category":
                        category,

                    "relative_path":
                        relative_path(
                            path
                        ),

                    "line_number":
                        line_no,

                    "pattern":
                        pattern_name,

                    "statement":
                        line.strip()[
                            :3000
                        ],
                })

    return rows


# ============================================================
# 12. ORDER-OF-OPERATIONS FORENSICS
# ============================================================

def infer_code_operation_order(
    path: Path
) -> List[Dict[str, Any]]:

    text = safe_read_text(
        path
    )

    if not text:
        return []

    rows = []

    patterns = {
        "load_data":
            r"(read_csv|read_excel)",

        "split":
            r"train_test_split\s*\(",

        "scaler_fit":
            r"(scaler|standardscaler|minmaxscaler|robustscaler).*fit",

        "fit_transform":
            r"fit_transform\s*\(",

        "transform":
            r"\.transform\s*\(",

        "balance":
            r"(smote|adasyn|oversampl|undersampl|resample)",

        "model_fit":
            r"\.fit\s*\(",

        "synthetic":
            r"(synthetic|generator|generate|gan|vae|diffusion)",
    }

    for line_no, line in enumerate(
        text.splitlines(),
        start=1
    ):

        stripped = (
            line.strip()
        )

        if not stripped:
            continue

        for operation, regex in (
            patterns.items()
        ):

            if re.search(
                regex,
                stripped,
                flags=re.IGNORECASE
            ):

                rows.append({
                    "relative_path":
                        relative_path(
                            path
                        ),

                    "line_number":
                        line_no,

                    "operation":
                        operation,

                    "statement":
                        stripped[
                            :3000
                        ],
                })

    return rows


# ============================================================
# 13. TIMESTAMP AUDIT
# ============================================================

def timestamp_audit(
    paths: Sequence[Path]
) -> List[Dict[str, Any]]:

    rows = []

    for path in paths:

        rows.append({
            "relative_path":
                relative_path(
                    path
                ),

            "exists":
                int(
                    path.exists()
                ),

            "modified_time":
                safe_mtime(
                    path
                ),

            "size_bytes":
                safe_size(
                    path
                ),

            "sha256":
                (
                    sha256_file(
                        path
                    )
                    if path.exists()
                    and path.is_file()
                    else ""
                ),
        })

    return rows


# ============================================================
# 14. FINDINGS ENGINE
# ============================================================

def add_finding(
    findings: List[
        Dict[str, Any]
    ],
    finding_id: str,
    severity: str,
    status: str,
    finding: str,
    evidence: str,
    implication: str,
) -> None:

    findings.append({
        "finding_id":
            finding_id,

        "severity":
            severity,

        "status":
            status,

        "finding":
            finding,

        "evidence":
            evidence,

        "implication":
            implication,
    })


def evaluate_findings(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    exact_rows: List[
        Dict[str, Any]
    ],
    near_rows: List[
        Dict[str, Any]
    ],
    target_audit: List[
        Dict[str, Any]
    ],
    correlation_rows: List[
        Dict[str, Any]
    ],
    split_code: List[
        Dict[str, Any]
    ],
    scaling_code: List[
        Dict[str, Any]
    ],
    balancing_code: List[
        Dict[str, Any]
    ],
    training_code: List[
        Dict[str, Any]
    ],
    synthetic_code: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:

    findings = []

    # --------------------------------------------------------
    # F01 - Split dimensions
    # --------------------------------------------------------

    if (
        len(
            X_train
        ) == len(
            y_train
        )
        and
        len(
            X_test
        ) == len(
            y_test
        )
    ):

        add_finding(
            findings,
            "F01",
            "INFO",
            "PASS",
            "Feature/label row counts are internally consistent.",
            (
                f"Train={len(X_train)}; "
                f"Test={len(X_test)}"
            ),
            "No row-count misalignment detected."
        )

    else:

        add_finding(
            findings,
            "F01",
            "CRITICAL",
            "FAIL",
            "Feature/label row counts are inconsistent.",
            (
                f"X_train={len(X_train)}, "
                f"y_train={len(y_train)}, "
                f"X_test={len(X_test)}, "
                f"y_test={len(y_test)}"
            ),
            "Evaluation cannot be considered reliable."
        )

    # --------------------------------------------------------
    # F02 - Exact duplication
    # --------------------------------------------------------

    exact_result = next(
        (
            row
            for row
            in exact_rows
            if row[
                "comparison"
            ] == "exact"
        ),
        None
    )

    exact_overlap = (
        int(
            exact_result[
                "overlap_pairs"
            ]
        )
        if exact_result
        else 0
    )

    if exact_overlap == 0:

        add_finding(
            findings,
            "F02",
            "HIGH",
            "PASS",
            "No exact feature-vector duplication was found between train and test.",
            "Exact cross-split overlap = 0.",
            "Direct row duplication is not evident."
        )

    else:

        add_finding(
            findings,
            "F02",
            "CRITICAL",
            "FAIL",
            "Exact train/test duplicates were detected.",
            (
                f"Exact overlap pairs = "
                f"{exact_overlap}"
            ),
            (
                "The preserved test set is not independent "
                "of training data."
            )
        )

    # --------------------------------------------------------
    # F03 - Rounded duplicates
    # --------------------------------------------------------

    rounded_max = max(
        [
            int(
                row[
                    "overlap_pairs"
                ]
            )
            for row in exact_rows
            if row[
                "comparison"
            ] != "exact"
        ]
        or [0]
    )

    if rounded_max == 0:

        add_finding(
            findings,
            "F03",
            "MEDIUM",
            "PASS",
            "No rounded near-identical train/test duplicates were detected.",
            (
                "No overlap after rounding "
                "to 8, 6, 4, or 3 decimal places."
            ),
            "Numerically near-identical standardized records are not evident."
        )

    else:

        add_finding(
            findings,
            "F03",
            "HIGH",
            "WARNING",
            "Some train/test records become identical after rounding.",
            (
                f"Maximum rounded overlap pairs = "
                f"{rounded_max}"
            ),
            (
                "Records may be near duplicates and require "
                "manual inspection."
            )
        )

    # --------------------------------------------------------
    # F04 - Very close Euclidean matches
    # --------------------------------------------------------

    distances = np.asarray(
        [
            float(
                r[
                    "euclidean_distance"
                ]
            )
            for r in near_rows
        ]
    )

    n_extremely_close = int(
        np.sum(
            distances <= 1e-6
        )
    )

    if n_extremely_close == 0:

        add_finding(
            findings,
            "F04",
            "MEDIUM",
            "PASS",
            "No test samples have essentially identical standardized train counterparts.",
            (
                f"Minimum nearest-train distance = "
                f"{float(np.min(distances)):.12g}"
            ),
            "Strong near-duplicate leakage is not evident."
        )

    else:

        add_finding(
            findings,
            "F04",
            "HIGH",
            "WARNING",
            "Extremely close cross-split records were detected.",
            (
                f"{n_extremely_close} test records have "
                "nearest-train Euclidean distance <= 1e-6."
            ),
            (
                "Potential duplicate/near-duplicate leakage "
                "should be investigated."
            )
        )

    # --------------------------------------------------------
    # F05 - Target feature names
    # --------------------------------------------------------

    suspicious_features = [
        row
        for row
        in target_audit
        if int(
            row[
                "name_suspicious"
            ]
        ) == 1
    ]

    if not suspicious_features:

        add_finding(
            findings,
            "F05",
            "CRITICAL",
            "PASS",
            "No obvious target-named feature is present in X.",
            "No feature name matched outcome/status/death/survival-related terms.",
            "Direct target-column leakage is not evident."
        )

    else:

        add_finding(
            findings,
            "F05",
            "CRITICAL",
            "FAIL",
            "Potential target-derived feature names were detected.",
            "; ".join(
                row[
                    "feature"
                ]
                for row
                in suspicious_features
            ),
            "Direct or semantic target leakage may exist."
        )

    # --------------------------------------------------------
    # F06 - Perfect correlation feature
    # --------------------------------------------------------

    perfect_corr = []

    for row in correlation_rows:

        value = row[
            "abs_train_correlation"
        ]

        if value == "":
            continue

        if float(
            value
        ) >= 0.999999:

            perfect_corr.append(
                row[
                    "feature"
                ]
            )

    if not perfect_corr:

        add_finding(
            findings,
            "F06",
            "HIGH",
            "PASS",
            "No individual numeric training feature has near-perfect linear correlation with the target.",
            "No |r| >= 0.999999 feature detected.",
            "Simple single-column label encoding is not evident."
        )

    else:

        add_finding(
            findings,
            "F06",
            "CRITICAL",
            "FAIL",
            "One or more features are essentially perfectly correlated with the training target.",
            "; ".join(
                perfect_corr
            ),
            "Strong direct target leakage is possible."
        )

    # --------------------------------------------------------
    # F07 - Split code
    # --------------------------------------------------------

    if split_code:

        add_finding(
            findings,
            "F07",
            "HIGH",
            "EVIDENCE_FOUND",
            "Train/test split implementation exists in project code.",
            (
                f"{len(split_code)} split-related code statements found."
            ),
            (
                "Operation ordering should be assessed from "
                "split_code_evidence.csv."
            )
        )

    else:

        add_finding(
            findings,
            "F07",
            "HIGH",
            "UNVERIFIED",
            "No explicit train/test split implementation was detected.",
            "No split-pattern evidence found.",
            (
                "The origin of preserved train/test files "
                "cannot be established from code."
            )
        )

    # --------------------------------------------------------
    # F08 - Scaling code
    # --------------------------------------------------------

    if scaling_code:

        add_finding(
            findings,
            "F08",
            "HIGH",
            "EVIDENCE_FOUND",
            "Scaling/preprocessing code exists.",
            (
                f"{len(scaling_code)} scaling-related statements found."
            ),
            (
                "Need to verify scaler.fit() uses X_train only, "
                "not the full dataset."
            )
        )

    else:

        add_finding(
            findings,
            "F08",
            "HIGH",
            "UNVERIFIED",
            "Scaler fitting procedure could not be verified.",
            "No scaling-code evidence found.",
            (
                "Feature preprocessing leakage remains unresolved."
            )
        )

    # --------------------------------------------------------
    # F09 - Balancing
    # --------------------------------------------------------

    if balancing_code:

        add_finding(
            findings,
            "F09",
            "CRITICAL",
            "EVIDENCE_FOUND",
            "Balancing/resampling logic exists in project code.",
            (
                f"{len(balancing_code)} balancing-related "
                "statements detected."
            ),
            (
                "Must determine whether balancing occurred before "
                "or after train/test separation."
            )
        )

    else:

        add_finding(
            findings,
            "F09",
            "MEDIUM",
            "UNVERIFIED",
            "No explicit balancing implementation was detected in scanned Python code.",
            "Balanced dataset exists but code provenance may be elsewhere.",
            "Ordering of balancing relative to splitting remains uncertain."
        )

    # --------------------------------------------------------
    # F10 - Model training provenance
    # --------------------------------------------------------

    if training_code:

        add_finding(
            findings,
            "F10",
            "HIGH",
            "EVIDENCE_FOUND",
            "Classifier training logic exists in project code.",
            (
                f"{len(training_code)} training-related statements found."
            ),
            (
                "Need to verify that only X_train/y_train are "
                "passed into model.fit()."
            )
        )

    else:

        add_finding(
            findings,
            "F10",
            "HIGH",
            "UNVERIFIED",
            "Training procedure for the persisted classifier could not be verified.",
            "No classifier-training evidence found.",
            "Saved-model provenance remains incomplete."
        )

    # --------------------------------------------------------
    # F11 - Synthetic generation interaction
    # --------------------------------------------------------

    if synthetic_code:

        add_finding(
            findings,
            "F11",
            "HIGH",
            "EVIDENCE_FOUND",
            "Synthetic/generative code exists in the project.",
            (
                f"{len(synthetic_code)} generative-related statements found."
            ),
            (
                "Need to confirm that the untouched test set was "
                "not used to fit the generator."
            )
        )

    else:

        add_finding(
            findings,
            "F11",
            "MEDIUM",
            "UNVERIFIED",
            "Synthetic-generation interaction with the test set was not established.",
            "No matching generative statements in inspected code.",
            "Generator/test-set independence cannot be inferred automatically."
        )

    return findings


# ============================================================
# 15. FINAL VERDICT
# ============================================================

def determine_verdict(
    findings: List[
        Dict[str, Any]
    ]
) -> Dict[str, Any]:

    critical_failures = [
        f
        for f in findings
        if (
            f[
                "severity"
            ] == "CRITICAL"
            and f[
                "status"
            ] == "FAIL"
        )
    ]

    high_warnings = [
        f
        for f in findings
        if (
            f[
                "severity"
            ] in {
                "CRITICAL",
                "HIGH"
            }
            and f[
                "status"
            ] in {
                "WARNING",
                "UNVERIFIED"
            }
        )
    ]

    if critical_failures:

        verdict = (
            "LEAKAGE_DETECTED_OR_STRONGLY_SUSPECTED"
        )

        manuscript_use = (
            "DO_NOT_USE_PERFECT_TEST_METRICS_AS_VALIDATED RESULTS"
        )

        explanation = (
            "At least one critical leakage test failed."
        )

    elif high_warnings:

        verdict = (
            "NO_DIRECT_LEAKAGE_DETECTED_BUT_PROVENANCE_INCOMPLETE"
        )

        manuscript_use = (
            "DO_NOT_YET_TREAT_1.000_METRICS_AS FULLY VALIDATED"
        )

        explanation = (
            "Direct row/target leakage may not be present, but one "
            "or more high-impact preprocessing/training provenance "
            "questions remain unresolved."
        )

    else:

        verdict = (
            "NO_LEAKAGE_DETECTED_WITH_AVAILABLE_EVIDENCE"
        )

        manuscript_use = (
            "PRESERVED_TEST_RESULTS_CAN_BE_REPORTED WITH SAMPLE-SIZE CAUTION"
        )

        explanation = (
            "No direct duplication, target leakage, or unresolved "
            "high-severity provenance issue was identified."
        )

    return {
        "verdict":
            verdict,

        "manuscript_use":
            manuscript_use,

        "critical_failures":
            len(
                critical_failures
            ),

        "high_unresolved_findings":
            len(
                high_warnings
            ),

        "explanation":
            explanation,
    }


# ============================================================
# 16. MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 92
    )

    print(
        "HFAGM - SPLIT AND DATA-LEAKAGE FORENSIC AUDIT"
    )

    print(
        "=" * 92
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Required preserved split
    # --------------------------------------------------------

    required = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required preserved split file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load preserved split
    # --------------------------------------------------------

    X_train = safe_read_csv(
        X_TRAIN_PATH
    )

    X_test = safe_read_csv(
        X_TEST_PATH
    )

    y_train, y_train_col = (
        load_single_label_vector(
            Y_TRAIN_PATH
        )
    )

    y_test, y_test_col = (
        load_single_label_vector(
            Y_TEST_PATH
        )
    )

    if (
        X_train is None
        or X_test is None
        or y_train is None
        or y_test is None
    ):

        raise RuntimeError(
            "Unable to read preserved train/test split."
        )

    print(
        f"\nX_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_train: {len(y_train)}"
    )

    print(
        f"y_test : {len(y_test)}"
    )

    print(
        f"Train class counts: "
        f"{dict(pd.Series(y_train).value_counts())}"
    )

    print(
        f"Test class counts: "
        f"{dict(pd.Series(y_test).value_counts())}"
    )

    # --------------------------------------------------------
    # Load other datasets
    # --------------------------------------------------------

    raw_csv_df = (
        safe_read_csv(
            RAW_CSV_PATH
        )
    )

    raw_excel_df = (
        safe_read_excel(
            RAW_XLSX_PATH
        )
    )

    preprocessed_df = (
        safe_read_csv(
            PREPROCESSED_PATH
        )
    )

    balanced_df = (
        safe_read_csv(
            BALANCED_PATH
        )
    )

    datasets = {
        "raw_csv":
            raw_csv_df,

        "raw_excel":
            raw_excel_df,

        "preprocessed":
            preprocessed_df,

        "balanced":
            balanced_df,

        "X_train":
            X_train,

        "X_test":
            X_test,
    }

    dataset_inventory_rows = []

    data_path_map = {
        "raw_csv":
            RAW_CSV_PATH,

        "raw_excel":
            RAW_XLSX_PATH,

        "preprocessed":
            PREPROCESSED_PATH,

        "balanced":
            BALANCED_PATH,

        "X_train":
            X_TRAIN_PATH,

        "X_test":
            X_TEST_PATH,
    }

    for label, df in (
        datasets.items()
    ):

        dataset_inventory_rows.append(
            inventory_dataframe(
                label,
                data_path_map[
                    label
                ],
                df
            )
        )

    write_csv(
        OUTPUT_DIR
        / "dataset_inventory.csv",
        dataset_inventory_rows
    )

    # --------------------------------------------------------
    # Schema comparison
    # --------------------------------------------------------

    schema_rows = schema_comparison(
        datasets
    )

    write_csv(
        OUTPUT_DIR
        / "schema_comparison.csv",
        schema_rows
    )

    # --------------------------------------------------------
    # Exact/rounded overlap
    # --------------------------------------------------------

    print(
        "\nChecking exact and rounded train/test overlap..."
    )

    exact_rows = exact_overlap_audit(
        X_train,
        X_test
    )

    write_csv(
        OUTPUT_DIR
        / "train_test_exact_overlap.csv",
        exact_rows
    )

    # --------------------------------------------------------
    # Nearest-neighbor duplication
    # --------------------------------------------------------

    print(
        "Checking cross-split near duplicates..."
    )

    near_rows = near_duplicate_audit(
        X_train,
        X_test
    )

    write_csv(
        OUTPUT_DIR
        / "train_test_near_duplicates.csv",
        near_rows
    )

    near_summary = (
        summarize_near_duplicates(
            near_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "train_test_near_duplicate_summary.csv",
        near_summary
    )

    # --------------------------------------------------------
    # Target leakage
    # --------------------------------------------------------

    print(
        "Checking direct and correlation-based target leakage..."
    )

    (
        target_audit_rows,
        correlation_rows
    ) = target_feature_audit(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    write_csv(
        OUTPUT_DIR
        / "target_feature_audit.csv",
        target_audit_rows
    )

    write_csv(
        OUTPUT_DIR
        / "feature_target_correlations.csv",
        correlation_rows
    )

    # --------------------------------------------------------
    # Code inspection
    # --------------------------------------------------------

    python_files = list(
        iter_python_files()
    )

    print(
        f"\nPython files inspected: "
        f"{len(python_files)}"
    )

    split_code = []

    scaling_code = []

    balancing_code = []

    synthetic_code = []

    training_code = []

    operation_order_rows = []

    for path in python_files:

        split_code.extend(
            inspect_code_patterns(
                path,
                SPLIT_PATTERNS,
                "split"
            )
        )

        scaling_code.extend(
            inspect_code_patterns(
                path,
                SCALING_PATTERNS,
                "scaling"
            )
        )

        balancing_code.extend(
            inspect_code_patterns(
                path,
                BALANCING_PATTERNS,
                "balancing"
            )
        )

        synthetic_code.extend(
            inspect_code_patterns(
                path,
                SYNTHETIC_PATTERNS,
                "synthetic"
            )
        )

        training_code.extend(
            inspect_code_patterns(
                path,
                TRAINING_PATTERNS,
                "training"
            )
        )

        order_rows = (
            infer_code_operation_order(
                path
            )
        )

        operation_order_rows.extend(
            order_rows
        )

    write_csv(
        OUTPUT_DIR
        / "split_code_evidence.csv",
        split_code
    )

    write_csv(
        OUTPUT_DIR
        / "scaling_code_evidence.csv",
        scaling_code
    )

    write_csv(
        OUTPUT_DIR
        / "balancing_code_evidence.csv",
        balancing_code
    )

    write_csv(
        OUTPUT_DIR
        / "synthetic_generation_code_evidence.csv",
        synthetic_code
    )

    write_csv(
        OUTPUT_DIR
        / "training_code_evidence.csv",
        training_code
    )

    write_csv(
        OUTPUT_DIR
        / "code_operation_order.csv",
        operation_order_rows
    )

    # --------------------------------------------------------
    # Timestamp audit
    # --------------------------------------------------------

    timestamp_paths = [
        RAW_CSV_PATH,
        RAW_XLSX_PATH,
        PREPROCESSED_PATH,
        BALANCED_PATH,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        MODEL_PATH,
        *KNOWN_CODE_FILES,
    ]

    timestamp_rows = timestamp_audit(
        timestamp_paths
    )

    write_csv(
        OUTPUT_DIR
        / "file_timestamp_audit.csv",
        timestamp_rows
    )

    # --------------------------------------------------------
    # Findings
    # --------------------------------------------------------

    findings = evaluate_findings(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        exact_rows=exact_rows,
        near_rows=near_rows,
        target_audit=target_audit_rows,
        correlation_rows=correlation_rows,
        split_code=split_code,
        scaling_code=scaling_code,
        balancing_code=balancing_code,
        training_code=training_code,
        synthetic_code=synthetic_code,
    )

    write_csv(
        OUTPUT_DIR
        / "leakage_findings.csv",
        findings,
        preferred_columns=[
            "finding_id",
            "severity",
            "status",
            "finding",
            "evidence",
            "implication",
        ]
    )

    verdict = determine_verdict(
        findings
    )

    write_csv(
        OUTPUT_DIR
        / "leakage_verdict.csv",
        [
            verdict
        ]
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    exact_result = next(
        (
            row
            for row
            in exact_rows
            if row[
                "comparison"
            ] == "exact"
        ),
        {}
    )

    min_distance = min(
        [
            float(
                row[
                    "euclidean_distance"
                ]
            )
            for row
            in near_rows
        ]
    )

    suspicious_target_names = [
        row[
            "feature"
        ]
        for row
        in target_audit_rows
        if int(
            row[
                "name_suspicious"
            ]
        ) == 1
    ]

    top_corr = (
        correlation_rows[:10]
    )

    summary = [
        "=" * 92,
        "HFAGM - SPLIT AND DATA-LEAKAGE FORENSIC AUDIT",
        "=" * 92,
        "",
        f"Generated: "
        f"{datetime.now().isoformat(timespec='seconds')}",
        "",
        "PRESERVED SPLIT",
        "-" * 92,
        f"X_train: "
        f"{X_train.shape[0]} rows x "
        f"{X_train.shape[1]} features",
        f"X_test: "
        f"{X_test.shape[0]} rows x "
        f"{X_test.shape[1]} features",
        f"y_train: "
        f"{len(y_train)}",
        f"y_test: "
        f"{len(y_test)}",
        f"Training label column: "
        f"{y_train_col}",
        f"Test label column: "
        f"{y_test_col}",
        "",
        "TRAIN/TEST DUPLICATION",
        "-" * 92,
        f"Exact overlap pairs: "
        f"{exact_result.get('overlap_pairs', 'NA')}",
        f"Unique overlapping test records: "
        f"{exact_result.get('unique_test_rows_overlapping', 'NA')}",
        f"Minimum test-to-train Euclidean distance: "
        f"{min_distance:.12g}",
        "",
        "TARGET LEAKAGE",
        "-" * 92,
        (
            "Suspicious target-related feature names: "
            + (
                ", ".join(
                    suspicious_target_names
                )
                if suspicious_target_names
                else "NONE"
            )
        ),
        "",
        "TOP ABSOLUTE TRAIN FEATURE-TARGET CORRELATIONS",
        "-" * 92,
    ]

    for row in top_corr:

        summary.append(
            f"{row['feature']}: "
            f"{row['train_target_correlation']}"
        )

    summary.extend([
        "",
        "CODE EVIDENCE COUNTS",
        "-" * 92,
        f"Split-related statements: "
        f"{len(split_code)}",
        f"Scaling-related statements: "
        f"{len(scaling_code)}",
        f"Balancing-related statements: "
        f"{len(balancing_code)}",
        f"Synthetic-generation statements: "
        f"{len(synthetic_code)}",
        f"Training-related statements: "
        f"{len(training_code)}",
        "",
        "FINDINGS",
        "-" * 92,
    ])

    for finding in findings:

        summary.extend([
            (
                f"{finding['finding_id']} "
                f"[{finding['severity']}] "
                f"{finding['status']}"
            ),
            f"  {finding['finding']}",
            f"  Evidence: "
            f"{finding['evidence']}",
            f"  Implication: "
            f"{finding['implication']}",
            "",
        ])

    summary.extend([
        "FINAL AUTOMATED VERDICT",
        "-" * 92,
        f"Verdict: "
        f"{verdict['verdict']}",
        f"Manuscript use: "
        f"{verdict['manuscript_use']}",
        f"Critical failures: "
        f"{verdict['critical_failures']}",
        f"High unresolved findings: "
        f"{verdict['high_unresolved_findings']}",
        "",
        f"Explanation: "
        f"{verdict['explanation']}",
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 92,
        "",
        "This automated verdict is intentionally conservative.",
        "",
        "A PASS for duplicate detection does not by itself prove that the test set",
        "was untouched. The operation ordering in split_and_standardize.py,",
        "preprocess_covid_clinical.py, classifier training code, balancing code,",
        "and synthetic-generation code must still agree with the preserved files.",
        "",
        "In particular, the following must be established before the regenerated",
        "1.000 test metrics are treated as validated manuscript results:",
        "",
        "1. Train/test separation occurred before any operation that learned",
        "   information from the full dataset.",
        "2. Any scaler was fitted exclusively on X_train.",
        "3. Oversampling/balancing did not use or create derivatives of test rows.",
        "4. Synthetic generators were fitted only on training data when evaluated",
        "   against the preserved real test set.",
        "5. ensemble_model.pkl was fitted only using training-side information.",
        "6. No outcome-derived variable entered the 51 predictor features.",
        "",
        "FILES TO REVIEW NEXT",
        "-" * 92,
        "1. leakage_verdict.csv",
        "2. leakage_findings.csv",
        "3. code_operation_order.csv",
        "4. split_code_evidence.csv",
        "5. scaling_code_evidence.csv",
        "6. balancing_code_evidence.csv",
        "7. training_code_evidence.csv",
        "8. synthetic_generation_code_evidence.csv",
        "9. train_test_exact_overlap.csv",
        "10. train_test_near_duplicate_summary.csv",
        "11. target_feature_audit.csv",
        "12. feature_target_correlations.csv",
        "13. dataset_inventory.csv",
        "14. file_timestamp_audit.csv",
        "",
        "=" * 92,
    ])

    summary_path = (
        OUTPUT_DIR
        / "split_leakage_audit_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            summary
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 92
    )

    print(
        "02C COMPLETE"
    )

    print(
        "=" * 92
    )

    print(
        f"\nExact train/test overlap pairs: "
        f"{exact_result.get('overlap_pairs', 'NA')}"
    )

    print(
        f"Minimum nearest train/test distance: "
        f"{min_distance:.8g}"
    )

    print(
        f"Suspicious target feature names: "
        f"{len(suspicious_target_names)}"
    )

    print(
        f"\nAutomated verdict:"
    )

    print(
        verdict[
            "verdict"
        ]
    )

    print(
        f"\nManuscript use:"
    )

    print(
        verdict[
            "manuscript_use"
        ]
    )

    print(
        "\nResults written to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nUpload these files first:"
    )

    print(
        OUTPUT_DIR
        / "split_leakage_audit_summary.txt"
    )

    print(
        OUTPUT_DIR
        / "leakage_verdict.csv"
    )

    print(
        OUTPUT_DIR
        / "leakage_findings.csv"
    )

    print(
        OUTPUT_DIR
        / "code_operation_order.csv"
    )

    print(
        OUTPUT_DIR
        / "split_code_evidence.csv"
    )

    print(
        OUTPUT_DIR
        / "scaling_code_evidence.csv"
    )

    print(
        OUTPUT_DIR
        / "balancing_code_evidence.csv"
    )

    print(
        OUTPUT_DIR
        / "training_code_evidence.csv"
    )

    print(
        OUTPUT_DIR
        / "target_feature_audit.csv"
    )

    print(
        OUTPUT_DIR
        / "feature_target_correlations.csv"
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n" + "=" * 92
        )

        print(
            "02C FAILED SAFELY"
        )

        print(
            "=" * 92
        )

        print(
            f"\n{type(exc).__name__}: "
            f"{exc}"
        )

        print(
            "\nNo dataset, model, split, or source file was modified."
        )

        raise