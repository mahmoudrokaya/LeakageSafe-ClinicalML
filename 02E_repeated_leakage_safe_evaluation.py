"""
02E_repeated_leakage_safe_evaluation.py
=======================================

Repeated leakage-safe evaluation for the HFAGM clinical classification
benchmark.

RATIONALE
---------
The historical 160/40 split was invalidated because exact train/test
duplication was detected.

02D corrected that problem by starting from the raw 193-record dataset
and splitting BEFORE learned preprocessing. A single corrected split
(seed 42) nevertheless produced perfect test performance.

02E tests whether that performance is stable across multiple
predeclared leakage-safe stratified partitions.

PROTOCOL PER SEED
-----------------
Raw 193 observations
        |
        v
same 51 historical predictors + status target
        |
        v
stratified train/test split FIRST
        |
        v
fit imputation on training only
        |
        v
fit MinMaxScaler on training only
        |
        v
fit StandardScaler on training only
        |
        +----------------------------------+
        |                                  |
        v                                  v
unbalanced training              training-only oversampling
        |                                  |
        v                                  v
clone preserved ensemble          clone preserved ensemble
        |                                  |
        +----------------+-----------------+
                         |
                         v
                untouched real test
                         |
                         v
      Accuracy / Precision / Sensitivity /
      Specificity / F1 / ROC-AUC / CM

NO model tuning is performed.

NO seed is selected based on performance.

NO test observation is used to fit preprocessing.

NO test observation is used in oversampling.

NO historical result is overwritten.

The script must be run in the same compatible environment used for
02B/02D, preferably scikit-learn 1.5.2.

Outputs
-------
outputs/revision_primary_metrics/repeated_leakage_safe_evaluation/

    repeated_seed_metrics.csv
    repeated_predictions.csv
    repeated_confusion_matrices.csv
    repeated_split_audit.csv
    repeated_summary_statistics.csv
    paired_variant_differences.csv
    paired_variant_tests.csv
    source_case_prediction_frequency.csv
    pooled_prediction_summary.csv
    repeated_evaluation_provenance.csv
    repeated_evaluation_summary.txt
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import traceback
import warnings

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy import stats

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

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

PREPROCESSED_193_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "covid_clinical_preprocessed.csv"
)

HISTORICAL_X_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "X_train_scaled.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifiers"
    / "ensemble_model.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
    / "repeated_leakage_safe_evaluation"
)

# -------------------------------------------------------------------------
# Predeclared seeds.
#
# Do NOT change/remove seeds after seeing results.
# -------------------------------------------------------------------------

SEEDS = [
    42,
    47,
    53,
    59,
    71,
    83,
    97,
    101,
    113,
    127,
]

TEST_SIZE = 0.20

RUN_UNBALANCED = True
RUN_TRAIN_ONLY_OVERSAMPLED = True

USE_MINMAX_SCALER = True
USE_STANDARD_SCALER = True

TARGET_CANDIDATES = [
    "status",
    "Status",
    "STATUS",
    "target",
    "Target",
    "label",
    "Label",
    "outcome",
    "Outcome",
]

METRICS_TO_SUMMARIZE = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
]

VARIANT_UNBALANCED = "unbalanced_training"
VARIANT_OVERSAMPLED = "training_only_oversampled"


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()

    return "".join(
        ch
        for ch in text
        if ch.isalnum()
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def safe_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(
            value
        )


def write_csv(
    path: Path,
    rows: Sequence[
        Dict[str, Any]
    ],
    columns: Optional[
        List[str]
    ] = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        pd.DataFrame(
            columns=(
                columns
                if columns
                else []
            )
        ).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        return

    df = pd.DataFrame(
        rows
    )

    if columns:
        remaining = [
            c
            for c in df.columns
            if c not in columns
        ]

        df = df[
            columns + remaining
        ]

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def read_csv_robust(
    path: Path
) -> pd.DataFrame:
    errors = []

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )

        except Exception as exc:
            errors.append(
                repr(
                    exc
                )
            )

    raise RuntimeError(
        f"Could not read CSV: {path}\n"
        + "\n".join(
            errors
        )
    )


# =============================================================================
# 3. LOAD ORIGINAL DATA
# =============================================================================

def load_original_dataset(
) -> Tuple[
    pd.DataFrame,
    Path,
    str,
]:
    candidates = []

    if RAW_CSV_PATH.exists():
        try:
            candidates.append(
                (
                    read_csv_robust(
                        RAW_CSV_PATH
                    ),
                    RAW_CSV_PATH,
                    "raw_csv",
                )
            )
        except Exception:
            pass

    if RAW_XLSX_PATH.exists():
        try:
            candidates.append(
                (
                    pd.read_excel(
                        RAW_XLSX_PATH
                    ),
                    RAW_XLSX_PATH,
                    "raw_excel",
                )
            )
        except Exception:
            pass

    for df, path, source_type in candidates:
        if len(
            df
        ) == 193:
            return (
                df.copy(),
                path,
                source_type,
            )

    if candidates:
        raise RuntimeError(
            "Readable raw dataset found, but no raw source "
            "contains exactly 193 records."
        )

    if PREPROCESSED_193_PATH.exists():
        df = read_csv_robust(
            PREPROCESSED_193_PATH
        )

        if len(
            df
        ) != 193:
            raise RuntimeError(
                "Fallback preprocessed data does not contain "
                "193 observations."
            )

        warnings.warn(
            "Using preprocessed 193-row dataset because raw data "
            "could not be loaded. Results remain provisional."
        )

        return (
            df.copy(),
            PREPROCESSED_193_PATH,
            "preprocessed_fallback",
        )

    raise FileNotFoundError(
        "No usable 193-row clinical dataset found."
    )


# =============================================================================
# 4. TARGET
# =============================================================================

def identify_target_column(
    df: pd.DataFrame
) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    normalized = {
        normalize_name(
            c
        ):
            c

        for c in df.columns
    }

    for candidate in TARGET_CANDIDATES:
        key = normalize_name(
            candidate
        )

        if key in normalized:
            return normalized[
                key
            ]

    raise RuntimeError(
        "Could not identify target column."
    )


def normalize_binary_target(
    y_raw: pd.Series
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:
    if y_raw.isna().any():
        raise RuntimeError(
            "Target contains missing values."
        )

    numeric = pd.to_numeric(
        y_raw,
        errors="coerce",
    )

    if numeric.notna().all():
        unique = sorted(
            numeric.unique()
            .tolist()
        )

        if unique == [
            0,
            1,
        ]:
            return (
                numeric.astype(
                    int
                ).to_numpy(),
                {
                    "mapping":
                        "native_0_1"
                },
            )

    text = (
        y_raw
        .astype(str)
        .str.strip()
        .str.lower()
    )

    negative_terms = {
        "recovered",
        "survived",
        "survivor",
        "alive",
        "negative",
        "no",
        "0",
    }

    positive_terms = {
        "deceased",
        "death",
        "dead",
        "died",
        "positive",
        "yes",
        "1",
    }

    unique = sorted(
        text.unique()
        .tolist()
    )

    if len(
        unique
    ) != 2:
        raise RuntimeError(
            f"Expected binary target; found {unique}"
        )

    mapping = {}

    for value in unique:
        if value in negative_terms:
            mapping[
                value
            ] = 0

        elif value in positive_terms:
            mapping[
                value
            ] = 1

    if len(
        mapping
    ) != 2:
        raise RuntimeError(
            "Target labels cannot be mapped without "
            "inventing class semantics."
        )

    y = (
        text.map(
            mapping
        )
        .astype(
            int
        )
        .to_numpy()
    )

    return (
        y,
        {
            "mapping":
                mapping
        },
    )


# =============================================================================
# 5. RECOVER SAME 51 HISTORICAL FEATURES
# =============================================================================

def load_historical_features(
) -> List[str]:
    if not HISTORICAL_X_TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Historical X_train_scaled.csv not found."
        )

    historical = read_csv_robust(
        HISTORICAL_X_TRAIN_PATH
    )

    if historical.shape[
        1
    ] != 51:
        raise RuntimeError(
            f"Expected 51 historical predictors, "
            f"found {historical.shape[1]}."
        )

    return [
        str(
            c
        )
        for c
        in historical.columns
    ]


def map_features(
    df: pd.DataFrame,
    historical_features: List[str],
    target_col: str,
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    normalized = defaultdict(
        list
    )

    for col in df.columns:
        if col == target_col:
            continue

        normalized[
            normalize_name(
                col
            )
        ].append(
            col
        )

    output = {}
    audit = []

    for historical_name in historical_features:
        if historical_name in df.columns:
            source = historical_name
            method = "exact"

        else:
            candidates = normalized[
                normalize_name(
                    historical_name
                )
            ]

            if len(
                candidates
            ) != 1:
                raise RuntimeError(
                    "Could not uniquely map historical feature "
                    f"'{historical_name}'. Candidates={candidates}"
                )

            source = candidates[
                0
            ]

            method = (
                "normalized"
            )

        if source == target_col:
            raise RuntimeError(
                "Target column mapped into predictors."
            )

        output[
            historical_name
        ] = df[
            source
        ].copy()

        audit.append(
            {
                "historical_feature":
                    historical_name,

                "source_feature":
                    source,

                "mapping_method":
                    method,
            }
        )

    X = pd.DataFrame(
        output
    )

    if X.shape[
        1
    ] != 51:
        raise RuntimeError(
            "Feature mapping did not produce 51 predictors."
        )

    return (
        X,
        audit,
    )


# =============================================================================
# 6. NUMERIC CONVERSION
# =============================================================================

def numeric_series(
    series: pd.Series
) -> pd.Series:
    if pd.api.types.is_numeric_dtype(
        series
    ):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    text = (
        series.astype(
            str
        )
        .str.strip()
        .replace(
            {
                "":
                    np.nan,

                "nan":
                    np.nan,

                "None":
                    np.nan,

                "NA":
                    np.nan,

                "N/A":
                    np.nan,

                "-":
                    np.nan,
            }
        )
        .str.replace(
            ",",
            "",
            regex=False,
        )
    )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def convert_features_to_numeric(
    X: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    out = pd.DataFrame(
        index=X.index
    )

    audit = []

    for col in X.columns:
        source_nonmissing = int(
            X[
                col
            ].notna().sum()
        )

        converted = numeric_series(
            X[
                col
            ]
        )

        converted_nonmissing = int(
            converted.notna().sum()
        )

        failed = (
            source_nonmissing
            - converted_nonmissing
        )

        failure_fraction = (
            failed
            / source_nonmissing

            if source_nonmissing
            else 0.0
        )

        if failure_fraction > 0.05:
            raise RuntimeError(
                f"Feature {col} has excessive numeric "
                f"conversion failures: {failed}/"
                f"{source_nonmissing}"
            )

        out[
            col
        ] = converted

        audit.append(
            {
                "feature":
                    col,

                "source_nonmissing":
                    source_nonmissing,

                "converted_nonmissing":
                    converted_nonmissing,

                "conversion_failures":
                    failed,

                "failure_fraction":
                    failure_fraction,
            }
        )

    return (
        out,
        audit,
    )


# =============================================================================
# 7. OVERLAP CHECK
# =============================================================================

def row_signature(
    row: pd.Series
) -> Tuple[Any, ...]:
    result = []

    for value in row:
        if pd.isna(
            value
        ):
            result.append(
                "__MISSING__"
            )
        else:
            result.append(
                float(
                    value
                )
            )

    return tuple(
        result
    )


def count_cross_split_overlap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[
    int,
    int,
    List[Dict[str, Any]],
]:
    train_map = defaultdict(
        list
    )

    for train_row in range(
        len(
            X_train
        )
    ):
        sig = row_signature(
            X_train.iloc[
                train_row
            ]
        )

        train_map[
            sig
        ].append(
            train_row
        )

    matches = []

    for test_row in range(
        len(
            X_test
        )
    ):
        sig = row_signature(
            X_test.iloc[
                test_row
            ]
        )

        for train_row in train_map.get(
            sig,
            [],
        ):
            matches.append(
                {
                    "train_partition_row":
                        train_row,

                    "test_partition_row":
                        test_row,
                }
            )

    unique_test = len(
        {
            row[
                "test_partition_row"
            ]
            for row
            in matches
        }
    )

    return (
        len(
            matches
        ),
        unique_test,
        matches,
    )


# =============================================================================
# 8. TRAIN-ONLY PREPROCESSING
# =============================================================================

def preprocess_train_only(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train_raw
    )

    X_test = imputer.transform(
        X_test_raw
    )

    if USE_MINMAX_SCALER:
        minmax = MinMaxScaler()

        X_train = minmax.fit_transform(
            X_train
        )

        X_test = minmax.transform(
            X_test
        )

    if USE_STANDARD_SCALER:
        standard = StandardScaler()

        X_train = standard.fit_transform(
            X_train
        )

        X_test = standard.transform(
            X_test
        )

    if not np.isfinite(
        X_train
    ).all():
        raise RuntimeError(
            "Nonfinite processed training values."
        )

    if not np.isfinite(
        X_test
    ).all():
        raise RuntimeError(
            "Nonfinite processed test values."
        )

    return (
        X_train,
        X_test,
    )


# =============================================================================
# 9. TRAIN-ONLY OVERSAMPLING
# =============================================================================

def oversample_training_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    classes, counts = np.unique(
        y_train,
        return_counts=True,
    )

    if len(
        classes
    ) != 2:
        raise RuntimeError(
            "Expected binary training target."
        )

    majority_class = classes[
        np.argmax(
            counts
        )
    ]

    minority_class = classes[
        np.argmin(
            counts
        )
    ]

    majority_count = int(
        max(
            counts
        )
    )

    majority_idx = np.where(
        y_train
        == majority_class
    )[0]

    minority_idx = np.where(
        y_train
        == minority_class
    )[0]

    before = {
        int(
            cls
        ):
            int(
                count
            )
        for cls, count
        in zip(
            classes,
            counts,
        )
    }

    if len(
        minority_idx
    ) == majority_count:
        return (
            X_train.copy(),
            y_train.copy(),
            {
                "before":
                    before,

                "after":
                    before,

                "new_training_duplicates":
                    0,
            },
        )

    minority_resampled = resample(
        minority_idx,
        replace=True,
        n_samples=majority_count,
        random_state=seed,
    )

    selected = np.concatenate(
        [
            majority_idx,
            minority_resampled,
        ]
    )

    rng = np.random.RandomState(
        seed
    )

    rng.shuffle(
        selected
    )

    X_bal = X_train[
        selected
    ]

    y_bal = y_train[
        selected
    ]

    after = {
        int(
            cls
        ):
            int(
                count
            )

        for cls, count
        in zip(
            *np.unique(
                y_bal,
                return_counts=True,
            )
        )
    }

    return (
        X_bal,
        y_bal,
        {
            "before":
                before,

            "after":
                after,

            "new_training_duplicates":
                int(
                    majority_count
                    - len(
                        minority_idx
                    )
                ),
        },
    )


# =============================================================================
# 10. MODEL
# =============================================================================

def load_preserved_model(
) -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    try:
        return joblib.load(
            MODEL_PATH
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not load preserved ensemble. "
            "Run in sklearn 1.5.2 environment.\n"
            f"{repr(exc)}"
        )


def positive_score(
    model: Any,
    X: np.ndarray,
) -> Tuple[
    Optional[np.ndarray],
    str,
]:
    if hasattr(
        model,
        "predict_proba"
    ):
        probs = model.predict_proba(
            X
        )

        classes = list(
            model.classes_
        )

        if 1 not in classes:
            raise RuntimeError(
                "Positive class 1 not found."
            )

        col = classes.index(
            1
        )

        return (
            probs[
                :,
                col
            ].astype(
                float
            ),
            "predict_proba",
        )

    if hasattr(
        model,
        "decision_function"
    ):
        values = model.decision_function(
            X
        )

        return (
            np.asarray(
                values
            ).reshape(
                -1
            ).astype(
                float
            ),
            "decision_function",
        )

    return (
        None,
        "unavailable",
    )


# =============================================================================
# 11. METRICS
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[
        np.ndarray
    ],
) -> Dict[str, Any]:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = [
        int(
            x
        )
        for x
        in (
            cm[
                0,
                0
            ],
            cm[
                0,
                1
            ],
            cm[
                1,
                0
            ],
            cm[
                1,
                1
            ],
        )
    ]

    specificity = (
        tn
        / (
            tn
            + fp
        )
        if (
            tn
            + fp
        )
        else np.nan
    )

    roc_auc = np.nan

    if (
        y_score is not None
        and len(
            np.unique(
                y_true
            )
        )
        == 2
    ):
        roc_auc = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "sensitivity":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "specificity":
            float(
                specificity
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "roc_auc":
            roc_auc,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,
    }


# =============================================================================
# 12. ONE SEED / ONE VARIANT
# =============================================================================

def train_and_evaluate(
    seed: int,
    variant: str,
    preserved_model: Any,
    X_train_processed: np.ndarray,
    y_train: np.ndarray,
    X_test_processed: np.ndarray,
    y_test: np.ndarray,
    test_source_indices: np.ndarray,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    if variant == VARIANT_UNBALANCED:
        X_fit = X_train_processed
        y_fit = y_train

        balance_info = {
            "method":
                "none",

            "train_rows_before":
                int(
                    len(
                        y_train
                    )
                ),

            "train_rows_after":
                int(
                    len(
                        y_train
                    )
                ),

            "class_counts_before":
                safe_json(
                    dict(
                        Counter(
                            y_train.tolist()
                        )
                    )
                ),

            "class_counts_after":
                safe_json(
                    dict(
                        Counter(
                            y_train.tolist()
                        )
                    )
                ),
        }

    elif variant == VARIANT_OVERSAMPLED:
        (
            X_fit,
            y_fit,
            info,
        ) = oversample_training_only(
            X_train_processed,
            y_train,
            seed=seed,
        )

        balance_info = {
            "method":
                "training_only_random_oversampling",

            "train_rows_before":
                int(
                    len(
                        y_train
                    )
                ),

            "train_rows_after":
                int(
                    len(
                        y_fit
                    )
                ),

            "class_counts_before":
                safe_json(
                    info[
                        "before"
                    ]
                ),

            "class_counts_after":
                safe_json(
                    info[
                        "after"
                    ]
                ),

            "new_training_duplicates":
                info[
                    "new_training_duplicates"
                ],
        }

    else:
        raise ValueError(
            f"Unknown variant: {variant}"
        )

    # Clone original architecture/hyperparameters.
    model = clone(
        preserved_model
    )

    model.fit(
        X_fit,
        y_fit,
    )

    y_pred = np.asarray(
        model.predict(
            X_test_processed
        )
    ).astype(
        int
    )

    y_score, score_source = positive_score(
        model,
        X_test_processed,
    )

    metric_values = calculate_metrics(
        y_test,
        y_pred,
        y_score,
    )

    metric_row = {
        "seed":
            seed,

        "variant":
            variant,

        "n_train_original":
            int(
                len(
                    y_train
                )
            ),

        "n_train_fitted":
            int(
                len(
                    y_fit
                )
            ),

        "n_test":
            int(
                len(
                    y_test
                )
            ),

        "train_class_0":
            int(
                np.sum(
                    y_train
                    == 0
                )
            ),

        "train_class_1":
            int(
                np.sum(
                    y_train
                    == 1
                )
            ),

        "test_class_0":
            int(
                np.sum(
                    y_test
                    == 0
                )
            ),

        "test_class_1":
            int(
                np.sum(
                    y_test
                    == 1
                )
            ),

        **metric_values,

        "score_source":
            score_source,

        **balance_info,
    }

    predictions = []

    for i in range(
        len(
            y_test
        )
    ):
        row = {
            "seed":
                seed,

            "variant":
                variant,

            "test_partition_row":
                i,

            "original_source_row_index":
                int(
                    test_source_indices[
                        i
                    ]
                ),

            "y_true":
                int(
                    y_test[
                        i
                    ]
                ),

            "y_pred":
                int(
                    y_pred[
                        i
                    ]
                ),

            "correct":
                int(
                    y_true_eq := (
                        y_test[
                            i
                        ]
                        == y_pred[
                            i
                        ]
                    )
                ),

            "score_source":
                score_source,
        }

        if y_score is not None:
            row[
                "y_score_positive"
            ] = float(
                y_score[
                    i
                ]
            )

        predictions.append(
            row
        )

    return (
        metric_row,
        predictions,
    )


# =============================================================================
# 13. SUMMARY STATISTICS
# =============================================================================

def mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    n = len(
        values
    )

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(
        np.mean(
            values
        )
    )

    if n == 1:
        return (
            mean,
            np.nan,
            np.nan,
            np.nan,
        )

    sd = float(
        np.std(
            values,
            ddof=1,
        )
    )

    se = sd / math.sqrt(
        n
    )

    alpha = (
        1.0
        - confidence
    )

    critical = stats.t.ppf(
        1.0
        - alpha
        / 2.0,
        df=n - 1,
    )

    lower = (
        mean
        - critical
        * se
    )

    upper = (
        mean
        + critical
        * se
    )

    # Metrics are bounded.
    lower = max(
        0.0,
        lower,
    )

    upper = min(
        1.0,
        upper,
    )

    return (
        mean,
        sd,
        lower,
        upper,
    )


def summarize_metrics(
    metrics_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows = []

    for variant in sorted(
        metrics_df[
            "variant"
        ].unique()
    ):
        subset = metrics_df[
            metrics_df[
                "variant"
            ]
            == variant
        ]

        for metric in METRICS_TO_SUMMARIZE:
            values = pd.to_numeric(
                subset[
                    metric
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            (
                mean,
                sd,
                ci_low,
                ci_high,
            ) = mean_ci(
                finite
            )

            rows.append(
                {
                    "variant":
                        variant,

                    "metric":
                        metric,

                    "n_runs":
                        int(
                            len(
                                finite
                            )
                        ),

                    "mean":
                        mean,

                    "sd":
                        sd,

                    "95ci_lower":
                        ci_low,

                    "95ci_upper":
                        ci_high,

                    "minimum":
                        (
                            float(
                                np.min(
                                    finite
                                )
                            )
                            if len(
                                finite
                            )
                            else np.nan
                        ),

                    "maximum":
                        (
                            float(
                                np.max(
                                    finite
                                )
                            )
                            if len(
                                finite
                            )
                            else np.nan
                        ),

                    "median":
                        (
                            float(
                                np.median(
                                    finite
                                )
                            )
                            if len(
                                finite
                            )
                            else np.nan
                        ),
                }
            )

    return rows


# =============================================================================
# 14. PAIRED VARIANT COMPARISON
# =============================================================================

def paired_variant_analysis(
    metrics_df: pd.DataFrame,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not (
        RUN_UNBALANCED
        and RUN_TRAIN_ONLY_OVERSAMPLED
    ):
        return (
            [],
            [],
        )

    difference_rows = []
    test_rows = []

    for metric in METRICS_TO_SUMMARIZE:
        pivot = metrics_df.pivot(
            index="seed",
            columns="variant",
            values=metric,
        )

        required = {
            VARIANT_UNBALANCED,
            VARIANT_OVERSAMPLED,
        }

        if not required.issubset(
            pivot.columns
        ):
            continue

        paired = pivot[
            [
                VARIANT_UNBALANCED,
                VARIANT_OVERSAMPLED,
            ]
        ].dropna()

        differences = (
            paired[
                VARIANT_OVERSAMPLED
            ]
            -
            paired[
                VARIANT_UNBALANCED
            ]
        )

        for seed, row in paired.iterrows():
            difference_rows.append(
                {
                    "seed":
                        int(
                            seed
                        ),

                    "metric":
                        metric,

                    "unbalanced":
                        float(
                            row[
                                VARIANT_UNBALANCED
                            ]
                        ),

                    "oversampled":
                        float(
                            row[
                                VARIANT_OVERSAMPLED
                            ]
                        ),

                    "difference_oversampled_minus_unbalanced":
                        float(
                            row[
                                VARIANT_OVERSAMPLED
                            ]
                            -
                            row[
                                VARIANT_UNBALANCED
                            ]
                        ),
                }
            )

        n = len(
            differences
        )

        mean_diff = float(
            differences.mean()
        ) if n else np.nan

        sd_diff = float(
            differences.std(
                ddof=1
            )
        ) if n > 1 else np.nan

        # ---------------------------------------------------------
        # Paired t-test
        # ---------------------------------------------------------

        if (
            n > 1
            and not np.allclose(
                differences,
                0.0,
            )
        ):
            t_stat, t_p = stats.ttest_rel(
                paired[
                    VARIANT_OVERSAMPLED
                ],
                paired[
                    VARIANT_UNBALANCED
                ],
            )

            t_stat = float(
                t_stat
            )

            t_p = float(
                t_p
            )

        elif n > 1:
            t_stat = 0.0
            t_p = 1.0

        else:
            t_stat = np.nan
            t_p = np.nan

        # ---------------------------------------------------------
        # Wilcoxon signed-rank
        # ---------------------------------------------------------

        if (
            n > 0
            and not np.allclose(
                differences,
                0.0,
            )
        ):
            try:
                w_stat, w_p = stats.wilcoxon(
                    paired[
                        VARIANT_OVERSAMPLED
                    ],
                    paired[
                        VARIANT_UNBALANCED
                    ],
                    zero_method="wilcox",
                    alternative="two-sided",
                )

                w_stat = float(
                    w_stat
                )

                w_p = float(
                    w_p
                )

            except Exception:
                w_stat = np.nan
                w_p = np.nan

        elif n > 0:
            w_stat = 0.0
            w_p = 1.0

        else:
            w_stat = np.nan
            w_p = np.nan

        test_rows.append(
            {
                "metric":
                    metric,

                "n_paired_runs":
                    n,

                "mean_difference_oversampled_minus_unbalanced":
                    mean_diff,

                "sd_difference":
                    sd_diff,

                "paired_t_statistic":
                    t_stat,

                "paired_t_pvalue":
                    t_p,

                "wilcoxon_statistic":
                    w_stat,

                "wilcoxon_pvalue":
                    w_p,
            }
        )

    return (
        difference_rows,
        test_rows,
    )


# =============================================================================
# 15. SOURCE CASE FREQUENCY
# =============================================================================

def source_case_frequency(
    predictions_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows = []

    grouped = predictions_df.groupby(
        [
            "variant",
            "original_source_row_index",
        ]
    )

    for (
        variant,
        source_idx,
    ), group in grouped:
        y_values = group[
            "y_true"
        ].unique()

        if len(
            y_values
        ) != 1:
            raise RuntimeError(
                "Same source row has inconsistent true labels."
            )

        errors = int(
            (
                group[
                    "correct"
                ]
                == 0
            ).sum()
        )

        scores = pd.to_numeric(
            group.get(
                "y_score_positive",
                pd.Series(
                    dtype=float
                )
            ),
            errors="coerce",
        )

        rows.append(
            {
                "variant":
                    variant,

                "original_source_row_index":
                    int(
                        source_idx
                    ),

                "y_true":
                    int(
                        y_values[
                            0
                        ]
                    ),

                "times_in_test":
                    int(
                        len(
                            group
                        )
                    ),

                "times_correct":
                    int(
                        (
                            group[
                                "correct"
                            ]
                            == 1
                        ).sum()
                    ),

                "times_incorrect":
                    errors,

                "error_rate_when_tested":
                    float(
                        errors
                        / len(
                            group
                        )
                    ),

                "mean_positive_score":
                    (
                        float(
                            scores.mean()
                        )
                        if scores.notna().any()
                        else np.nan
                    ),

                "min_positive_score":
                    (
                        float(
                            scores.min()
                        )
                        if scores.notna().any()
                        else np.nan
                    ),

                "max_positive_score":
                    (
                        float(
                            scores.max()
                        )
                        if scores.notna().any()
                        else np.nan
                    ),
            }
        )

    return rows


# =============================================================================
# 16. POOLED PREDICTION SUMMARY
# =============================================================================

def pooled_prediction_summary(
    predictions_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows = []

    for variant in sorted(
        predictions_df[
            "variant"
        ].unique()
    ):
        subset = predictions_df[
            predictions_df[
                "variant"
            ]
            == variant
        ].copy()

        y_true = subset[
            "y_true"
        ].to_numpy(
            dtype=int
        )

        y_pred = subset[
            "y_pred"
        ].to_numpy(
            dtype=int
        )

        if (
            "y_score_positive"
            in subset.columns
        ):
            score = pd.to_numeric(
                subset[
                    "y_score_positive"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            score = (
                score
                if np.isfinite(
                    score
                ).all()
                else None
            )

        else:
            score = None

        metrics = calculate_metrics(
            y_true,
            y_pred,
            score,
        )

        rows.append(
            {
                "variant":
                    variant,

                "pooled_test_predictions":
                    int(
                        len(
                            subset
                        )
                    ),

                "unique_source_cases_appearing_in_test":
                    int(
                        subset[
                            "original_source_row_index"
                        ].nunique()
                    ),

                **metrics,
            }
        )

    return rows


# =============================================================================
# 17. MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 96
    )

    print(
        "HFAGM - REPEATED LEAKAGE-SAFE EVALUATION"
    )

    print(
        "=" * 96
    )

    print(
        f"\nSeeds: {SEEDS}"
    )

    print(
        f"Test size: {TEST_SIZE}"
    )

    print(
        f"scikit-learn: {sklearn.__version__}"
    )

    # -----------------------------------------------------------------
    # Original source
    # -----------------------------------------------------------------

    (
        original_df,
        original_path,
        source_type,
    ) = load_original_dataset()

    if len(
        original_df
    ) != 193:
        raise RuntimeError(
            f"Expected 193 raw records; got {len(original_df)}."
        )

    print(
        f"\nData source: {original_path}"
    )

    print(
        f"Source type: {source_type}"
    )

    print(
        f"Rows: {len(original_df)}"
    )

    target_col = identify_target_column(
        original_df
    )

    y, target_mapping = normalize_binary_target(
        original_df[
            target_col
        ]
    )

    historical_features = load_historical_features()

    (
        X_source,
        mapping_rows,
    ) = map_features(
        original_df,
        historical_features,
        target_col,
    )

    (
        X,
        numeric_audit_rows,
    ) = convert_features_to_numeric(
        X_source
    )

    if X.shape != (
        193,
        51,
    ):
        raise RuntimeError(
            f"Expected feature matrix (193, 51); got {X.shape}."
        )

    write_csv(
        OUTPUT_DIR
        / "feature_mapping.csv",
        mapping_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "numeric_conversion_audit.csv",
        numeric_audit_rows,
    )

    # -----------------------------------------------------------------
    # Verify original dataset has no duplicated feature rows.
    # -----------------------------------------------------------------

    original_feature_duplicates = int(
        X.duplicated(
            keep=False
        ).sum()
    )

    original_xy = X.copy()

    original_xy[
        "__target__"
    ] = y

    original_xy_duplicates = int(
        original_xy.duplicated(
            keep=False
        ).sum()
    )

    print(
        f"Original feature-duplicate rows: "
        f"{original_feature_duplicates}"
    )

    print(
        f"Original feature+target duplicate rows: "
        f"{original_xy_duplicates}"
    )

    if original_feature_duplicates > 0:
        raise RuntimeError(
            "Original 193-row dataset contains duplicated feature "
            "vectors. A grouped split is required."
        )

    # -----------------------------------------------------------------
    # Load preserved ensemble configuration once.
    # -----------------------------------------------------------------

    preserved_model = load_preserved_model()

    print(
        "\nPreserved model:"
    )

    print(
        f"{preserved_model.__class__.__module__}."
        f"{preserved_model.__class__.__name__}"
    )

    # -----------------------------------------------------------------
    # Repeated evaluation
    # -----------------------------------------------------------------

    all_metric_rows = []
    all_prediction_rows = []
    all_confusion_rows = []
    all_split_audit_rows = []

    variants = []

    if RUN_UNBALANCED:
        variants.append(
            VARIANT_UNBALANCED
        )

    if RUN_TRAIN_ONLY_OVERSAMPLED:
        variants.append(
            VARIANT_OVERSAMPLED
        )

    indices = np.arange(
        len(
            X
        )
    )

    for run_number, seed in enumerate(
        SEEDS,
        start=1,
    ):
        print(
            "\n" + "=" * 96
        )

        print(
            f"RUN {run_number}/{len(SEEDS)} - SEED {seed}"
        )

        print(
            "=" * 96
        )

        (
            train_idx,
            test_idx,
        ) = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )

        X_train_raw = (
            X.iloc[
                train_idx
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        X_test_raw = (
            X.iloc[
                test_idx
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        y_train = y[
            train_idx
        ].astype(
            int
        )

        y_test = y[
            test_idx
        ].astype(
            int
        )

        (
            overlap_pairs,
            overlapping_test_rows,
            overlap_details,
        ) = count_cross_split_overlap(
            X_train_raw,
            X_test_raw,
        )

        split_row = {
            "run_number":
                run_number,

            "seed":
                seed,

            "n_train":
                len(
                    train_idx
                ),

            "n_test":
                len(
                    test_idx
                ),

            "train_class_0":
                int(
                    np.sum(
                        y_train
                        == 0
                    )
                ),

            "train_class_1":
                int(
                    np.sum(
                        y_train
                        == 1
                    )
                ),

            "test_class_0":
                int(
                    np.sum(
                        y_test
                        == 0
                    )
                ),

            "test_class_1":
                int(
                    np.sum(
                        y_test
                        == 1
                    )
                ),

            "exact_overlap_pairs":
                overlap_pairs,

            "unique_overlapping_test_rows":
                overlapping_test_rows,

            "train_source_indices":
                safe_json(
                    [
                        int(
                            x
                        )
                        for x
                        in train_idx
                    ]
                ),

            "test_source_indices":
                safe_json(
                    [
                        int(
                            x
                        )
                        for x
                        in test_idx
                    ]
                ),
        }

        all_split_audit_rows.append(
            split_row
        )

        print(
            f"Train: {len(train_idx)} "
            f"{dict(Counter(y_train.tolist()))}"
        )

        print(
            f"Test : {len(test_idx)} "
            f"{dict(Counter(y_test.tolist()))}"
        )

        print(
            f"Exact cross-split overlap: "
            f"{overlap_pairs}"
        )

        if overlap_pairs > 0:
            raise RuntimeError(
                f"Seed {seed} produced exact cross-split overlap "
                f"from the original dataset. Evaluation stopped."
            )

        # -------------------------------------------------------------
        # Preprocessing fit only on training subset.
        # -------------------------------------------------------------

        (
            X_train_processed,
            X_test_processed,
        ) = preprocess_train_only(
            X_train_raw,
            X_test_raw,
        )

        # -------------------------------------------------------------
        # Run both variants on SAME test set.
        # -------------------------------------------------------------

        for variant in variants:
            (
                metric_row,
                prediction_rows,
            ) = train_and_evaluate(
                seed=seed,
                variant=variant,
                preserved_model=preserved_model,
                X_train_processed=X_train_processed,
                y_train=y_train,
                X_test_processed=X_test_processed,
                y_test=y_test,
                test_source_indices=test_idx,
            )

            metric_row[
                "run_number"
            ] = run_number

            metric_row[
                "cross_split_overlap_pairs"
            ] = overlap_pairs

            all_metric_rows.append(
                metric_row
            )

            all_prediction_rows.extend(
                prediction_rows
            )

            all_confusion_rows.append(
                {
                    "run_number":
                        run_number,

                    "seed":
                        seed,

                    "variant":
                        variant,

                    "tn":
                        metric_row[
                            "tn"
                        ],

                    "fp":
                        metric_row[
                            "fp"
                        ],

                    "fn":
                        metric_row[
                            "fn"
                        ],

                    "tp":
                        metric_row[
                            "tp"
                        ],
                }
            )

            print(
                f"\n{variant}"
            )

            print(
                f"  Accuracy    : "
                f"{metric_row['accuracy']:.6f}"
            )

            print(
                f"  Precision   : "
                f"{metric_row['precision']:.6f}"
            )

            print(
                f"  Sensitivity : "
                f"{metric_row['sensitivity']:.6f}"
            )

            print(
                f"  Specificity : "
                f"{metric_row['specificity']:.6f}"
            )

            print(
                f"  F1          : "
                f"{metric_row['f1']:.6f}"
            )

            print(
                f"  ROC-AUC     : "
                f"{metric_row['roc_auc']:.6f}"
            )

            print(
                f"  CM          : "
                f"TN={metric_row['tn']} "
                f"FP={metric_row['fp']} "
                f"FN={metric_row['fn']} "
                f"TP={metric_row['tp']}"
            )

    # -----------------------------------------------------------------
    # Save raw repeated results.
    # -----------------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_metric_rows
    )

    predictions_df = pd.DataFrame(
        all_prediction_rows
    )

    metrics_df.to_csv(
        OUTPUT_DIR
        / "repeated_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        OUTPUT_DIR
        / "repeated_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_csv(
        OUTPUT_DIR
        / "repeated_confusion_matrices.csv",
        all_confusion_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "repeated_split_audit.csv",
        all_split_audit_rows,
    )

    # -----------------------------------------------------------------
    # Summary statistics.
    # -----------------------------------------------------------------

    summary_rows = summarize_metrics(
        metrics_df
    )

    write_csv(
        OUTPUT_DIR
        / "repeated_summary_statistics.csv",
        summary_rows,
    )

    # -----------------------------------------------------------------
    # Paired comparison between variants.
    # -----------------------------------------------------------------

    (
        difference_rows,
        paired_test_rows,
    ) = paired_variant_analysis(
        metrics_df
    )

    write_csv(
        OUTPUT_DIR
        / "paired_variant_differences.csv",
        difference_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_variant_tests.csv",
        paired_test_rows,
    )

    # -----------------------------------------------------------------
    # Source-case recurrence/error audit.
    # -----------------------------------------------------------------

    source_rows = source_case_frequency(
        predictions_df
    )

    write_csv(
        OUTPUT_DIR
        / "source_case_prediction_frequency.csv",
        source_rows,
    )

    # -----------------------------------------------------------------
    # Pooled descriptive results.
    #
    # IMPORTANT:
    # These pooled results are descriptive only because the same source
    # case may appear in the test set for multiple seeds.
    # -----------------------------------------------------------------

    pooled_rows = pooled_prediction_summary(
        predictions_df
    )

    write_csv(
        OUTPUT_DIR
        / "pooled_prediction_summary.csv",
        pooled_rows,
    )

    # -----------------------------------------------------------------
    # Provenance.
    # -----------------------------------------------------------------

    provenance = {
        "generated":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "original_source":
            str(
                original_path
            ),

        "original_source_type":
            source_type,

        "original_source_sha256":
            sha256_file(
                original_path
            ),

        "original_rows":
            len(
                original_df
            ),

        "target_column":
            target_col,

        "target_mapping":
            safe_json(
                target_mapping
            ),

        "n_features":
            len(
                historical_features
            ),

        "seeds":
            safe_json(
                SEEDS
            ),

        "n_seeds":
            len(
                SEEDS
            ),

        "test_size":
            TEST_SIZE,

        "stratification":
            True,

        "split_before_preprocessing":
            True,

        "imputer_fit_scope":
            "training_only",

        "minmax_fit_scope":
            (
                "training_only"
                if USE_MINMAX_SCALER
                else "disabled"
            ),

        "standard_scaler_fit_scope":
            (
                "training_only"
                if USE_STANDARD_SCALER
                else "disabled"
            ),

        "oversampling_scope":
            "training_only",

        "preserved_model_path":
            str(
                MODEL_PATH
            ),

        "preserved_model_sha256":
            sha256_file(
                MODEL_PATH
            ),

        "model_configuration":
            "cloned_from_preserved_ensemble",

        "model_tuning_performed":
            False,

        "seed_selection_based_on_results":
            False,

        "threshold_tuning":
            False,

        "test_used_for_training":
            False,

        "test_used_for_preprocessing_fit":
            False,

        "test_used_for_oversampling":
            False,

        "original_feature_duplicate_rows":
            original_feature_duplicates,

        "original_feature_target_duplicate_rows":
            original_xy_duplicates,

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "sklearn_version":
            sklearn.__version__,

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "scipy_version":
            scipy.__version__,
    }

    write_csv(
        OUTPUT_DIR
        / "repeated_evaluation_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    lines = [
        "=" * 96,
        "HFAGM - REPEATED LEAKAGE-SAFE EVALUATION",
        "=" * 96,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "PURPOSE",
        "-" * 96,
        (
            "Assess whether the perfect result obtained in the single corrected "
            "seed-42 evaluation remains stable across multiple leakage-safe "
            "stratified train/test partitions."
        ),
        "",
        "DATA",
        "-" * 96,
        f"Source: {original_path}",
        f"Source type: {source_type}",
        f"Observations: {len(original_df)}",
        f"Predictors: {len(historical_features)}",
        f"Target: {target_col}",
        f"Target counts: {dict(Counter(y.tolist()))}",
        "",
        "REPEATED PROTOCOL",
        "-" * 96,
        f"Seeds: {SEEDS}",
        f"Number of seeds: {len(SEEDS)}",
        f"Test size: {TEST_SIZE}",
        "Split performed before all learned preprocessing.",
        "Median imputation fitted on training only.",
        "MinMaxScaler fitted on training only.",
        "StandardScaler fitted on training only.",
        "Oversampling, when used, applied only to training records.",
        "Preserved ensemble architecture/hyperparameters cloned for every run.",
        "No hyperparameter tuning was performed.",
        "No seed was selected or excluded based on performance.",
        "",
        "SPLIT INTEGRITY",
        "-" * 96,
        (
            f"Original feature-duplicate rows: "
            f"{original_feature_duplicates}"
        ),
        (
            f"Original feature+target duplicate rows: "
            f"{original_xy_duplicates}"
        ),
        (
            "Cross-split exact overlaps across repeated runs: "
            f"{sum(row['exact_overlap_pairs'] for row in all_split_audit_rows)}"
        ),
        "",
        "SUMMARY STATISTICS",
        "-" * 96,
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    for variant in variants:
        lines.append(
            ""
        )

        lines.append(
            f"Variant: {variant}"
        )

        variant_summary = summary_df[
            summary_df[
                "variant"
            ]
            == variant
        ]

        for _, row in variant_summary.iterrows():
            lines.append(
                (
                    f"  {row['metric']:<12} "
                    f"mean={row['mean']:.6f} "
                    f"SD={row['sd']:.6f} "
                    f"95% CI=[{row['95ci_lower']:.6f}, "
                    f"{row['95ci_upper']:.6f}] "
                    f"min={row['minimum']:.6f} "
                    f"max={row['maximum']:.6f}"
                )
            )

    lines.extend(
        [
            "",
            "PER-SEED RESULTS",
            "-" * 96,
        ]
    )

    ordered_metrics = metrics_df.sort_values(
        [
            "seed",
            "variant",
        ]
    )

    for _, row in ordered_metrics.iterrows():
        lines.append(
            (
                f"Seed={int(row['seed']):>3} "
                f"Variant={row['variant']:<26} "
                f"Acc={row['accuracy']:.6f} "
                f"F1={row['f1']:.6f} "
                f"AUC={row['roc_auc']:.6f} "
                f"CM=[TN={int(row['tn'])}, "
                f"FP={int(row['fp'])}, "
                f"FN={int(row['fn'])}, "
                f"TP={int(row['tp'])}]"
            )
        )

    lines.extend(
        [
            "",
            "PAIRED VARIANT COMPARISON",
            "-" * 96,
        ]
    )

    if paired_test_rows:
        for row in paired_test_rows:
            lines.append(
                (
                    f"{row['metric']}: "
                    f"mean delta="
                    f"{row['mean_difference_oversampled_minus_unbalanced']:.6f}; "
                    f"paired t p={row['paired_t_pvalue']}; "
                    f"Wilcoxon p={row['wilcoxon_pvalue']}"
                )
            )
    else:
        lines.append(
            "Paired variant analysis not applicable."
        )

    lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 96,
            (
                "The repeated-seed mean, SD, confidence interval, and range "
                "should be used to assess stability. A single perfect split "
                "must not be treated as evidence of universal perfect "
                "classification."
            ),
            "",
            (
                "pooled_prediction_summary.csv is DESCRIPTIVE ONLY because "
                "the same original subject can appear in the test partition "
                "for more than one seed. It must not be treated as an "
                "independent pooled sample."
            ),
            "",
            (
                "If repeated performance remains extremely high, the next "
                "scientific audit should examine feature-label separability "
                "and whether any clinically post-outcome or outcome-proxy "
                "variables are present among the 51 predictors."
            ),
            "",
            "PRIMARY OUTPUTS",
            "-" * 96,
            "repeated_seed_metrics.csv",
            "repeated_summary_statistics.csv",
            "repeated_predictions.csv",
            "repeated_confusion_matrices.csv",
            "repeated_split_audit.csv",
            "paired_variant_differences.csv",
            "paired_variant_tests.csv",
            "source_case_prediction_frequency.csv",
            "pooled_prediction_summary.csv",
            "repeated_evaluation_provenance.csv",
            "",
            "=" * 96,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "repeated_evaluation_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console final summary.
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 96
    )

    print(
        "02E COMPLETE"
    )

    print(
        "=" * 96
    )

    print(
        f"\nRuns completed: "
        f"{len(SEEDS)} seeds x {len(variants)} variants"
    )

    print(
        "\nSummary:"
    )

    for variant in variants:
        print(
            f"\n{variant}"
        )

        sub = summary_df[
            summary_df[
                "variant"
            ]
            == variant
        ]

        for _, row in sub.iterrows():
            print(
                f"  {row['metric']:<12}: "
                f"{row['mean']:.6f} ± {row['sd']:.6f} "
                f"95% CI "
                f"[{row['95ci_lower']:.6f}, "
                f"{row['95ci_upper']:.6f}] "
                f"range "
                f"[{row['minimum']:.6f}, "
                f"{row['maximum']:.6f}]"
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

    for filename in [
        "repeated_evaluation_summary.txt",
        "repeated_seed_metrics.csv",
        "repeated_summary_statistics.csv",
        "repeated_predictions.csv",
        "repeated_confusion_matrices.csv",
        "repeated_split_audit.csv",
        "paired_variant_tests.csv",
        "source_case_prediction_frequency.csv",
        "repeated_evaluation_provenance.csv",
    ]:
        print(
            OUTPUT_DIR
            / filename
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            "\n" + "=" * 96
        )

        print(
            "02E FAILED SAFELY"
        )

        print(
            "=" * 96
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo source data, historical model, or prior "
            "experimental output was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(
            1
        )