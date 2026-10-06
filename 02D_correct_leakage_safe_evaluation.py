"""
02D_correct_leakage_safe_evaluation.py
======================================

HFAGM corrected leakage-safe classification evaluation.

PURPOSE
-------
The preserved HFAGM evaluation was found to contain exact train/test
duplication. This script creates a NEW leakage-safe evaluation without
modifying any historical files.

The corrected protocol is:

    original 193-row clinical dataset
            |
            v
    identify same 51 predictor features + status target
            |
            v
    stratified train/test split FIRST
            |
            v
    fit all learned preprocessing on training only
            |
            v
    transform untouched test set
            |
            +-----------------------------+
            |                             |
            v                             v
    unbalanced training            training-only oversampling
            |                             |
            v                             v
     clone original ensemble        clone original ensemble
            |                             |
            +-------------+---------------+
                          |
                          v
             SAME untouched real test set
                          |
                          v
          Accuracy / Precision / Recall /
       Specificity / F1 / ROC-AUC / CM

IMPORTANT
---------
This script:
- does NOT overwrite the historical split;
- does NOT overwrite ensemble_model.pkl;
- does NOT modify raw/preprocessed datasets;
- does NOT balance before splitting;
- does NOT fit imputation/scalers on test data;
- does NOT tune thresholds;
- does NOT tune hyperparameters;
- does NOT use test performance to select a model;
- does NOT fabricate missing results.

It attempts to reproduce the PRESERVED ensemble architecture and
hyperparameters by loading ensemble_model.pkl and cloning its estimators.

If the original model cannot be reconstructed safely, the script FAILS
instead of silently using default classifiers.

RUN THIS SCRIPT WITH THE scikit-learn 1.5.2 ENVIRONMENT USED TO LOAD
THE PRESERVED MODEL.

Expected environment:
    scikit-learn == 1.5.2
    numpy        == 1.26.4
    scipy        == 1.13.1
    pandas       == 2.2.3
    joblib       == 1.4.2
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import platform
import sys
import traceback
import warnings

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn

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

NEW_CODE_DIR = PROJECT_ROOT / "New_Code"

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

HISTORICAL_X_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "X_test_scaled.csv"
)

HISTORICAL_Y_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "y_train.csv"
)

HISTORICAL_Y_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "y_test.csv"
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
    / "corrected_leakage_safe_evaluation"
)

# Preserve the historical intended split settings where possible.
TEST_SIZE = 0.20
RANDOM_STATE = 42

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

# Original preprocessing contained MinMax scaling followed later by
# StandardScaler. We preserve that transformation family, but both
# scalers are fitted strictly on training data here.
USE_MINMAX_SCALER = True
USE_STANDARD_SCALER = True

# Train-only balancing variant.
RUN_UNBALANCED_VARIANT = True
RUN_TRAIN_ONLY_OVERSAMPLING_VARIANT = True

# Threshold is fixed; it is NOT optimized using the test set.
CLASSIFICATION_THRESHOLD = 0.50


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(
        df,
        index=True,
    ).values.tobytes()

    return hashlib.sha256(payload).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        if columns:
            pd.DataFrame(
                columns=columns
            ).to_csv(
                path,
                index=False,
                encoding="utf-8-sig",
            )
        else:
            path.write_text(
                "",
                encoding="utf-8",
            )

        return

    df = pd.DataFrame(rows)

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


def normalize_column_name(value: Any) -> str:
    text = str(value).strip().lower()

    text = "".join(
        ch
        for ch in text
        if ch.isalnum()
    )

    return text


def safe_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


# =============================================================================
# 3. LOAD DATA
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
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
                f"{encoding}: {repr(exc)}"
            )

    raise RuntimeError(
        f"Could not read CSV:\n{path}\n\n"
        + "\n".join(errors)
    )


def load_original_dataset() -> Tuple[pd.DataFrame, Path, str]:
    """
    Prefer the raw 193-row dataset.

    We intentionally do NOT use covid_clinical_balanced.csv because
    that file was the source of pre-split duplication.

    covid_clinical_preprocessed.csv is only a fallback if the raw
    source cannot be read, and the fallback is clearly flagged.
    """

    candidates = []

    if RAW_CSV_PATH.exists():
        try:
            df = read_csv_robust(
                RAW_CSV_PATH
            )

            candidates.append(
                (
                    df,
                    RAW_CSV_PATH,
                    "raw_csv",
                )
            )

        except Exception:
            pass

    if RAW_XLSX_PATH.exists():
        try:
            df = pd.read_excel(
                RAW_XLSX_PATH
            )

            candidates.append(
                (
                    df,
                    RAW_XLSX_PATH,
                    "raw_excel",
                )
            )

        except Exception:
            pass

    # Select an original 193-row source if available.
    for df, path, source_type in candidates:
        if len(df) == 193:
            return (
                df.copy(),
                path,
                source_type,
            )

    # If both raw sources were readable but had different row count,
    # do not silently assume which is correct.
    if candidates:
        details = "\n".join(
            f"{path}: {len(df)} rows"
            for df, path, _
            in candidates
        )

        raise RuntimeError(
            "Raw clinical dataset(s) were readable, but none "
            "contained the expected 193 original records.\n"
            + details
        )

    # Last-resort fallback.
    if PREPROCESSED_193_PATH.exists():
        df = read_csv_robust(
            PREPROCESSED_193_PATH
        )

        if len(df) != 193:
            raise RuntimeError(
                "Fallback preprocessed dataset does not contain "
                f"193 records: found {len(df)}."
            )

        warnings.warn(
            "\nRAW DATA COULD NOT BE READ.\n"
            "Using covid_clinical_preprocessed.csv as a fallback.\n"
            "Because this file may already contain preprocessing learned "
            "from the full dataset, results MUST NOT be considered fully "
            "leakage-safe without further provenance verification.\n"
        )

        return (
            df.copy(),
            PREPROCESSED_193_PATH,
            "preprocessed_fallback",
        )

    raise FileNotFoundError(
        "No usable original 193-row clinical dataset was found."
    )


# =============================================================================
# 4. TARGET AND FEATURE IDENTIFICATION
# =============================================================================

def identify_target_column(
    df: pd.DataFrame,
) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    normalized = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for candidate in TARGET_CANDIDATES:
        norm = normalize_column_name(
            candidate
        )

        if norm in normalized:
            return normalized[norm]

    raise RuntimeError(
        "Unable to identify target/status column in original dataset.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def load_historical_feature_names() -> List[str]:
    """
    Use historical X_train column names as the authoritative predictor set.
    This prevents us from inventing a new feature definition.
    """

    if not HISTORICAL_X_TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Historical X_train_scaled.csv is required to recover "
            "the exact predictor set."
        )

    historical_X = read_csv_robust(
        HISTORICAL_X_TRAIN_PATH
    )

    if historical_X.shape[1] != 51:
        raise RuntimeError(
            "Expected 51 historical predictors, but found "
            f"{historical_X.shape[1]}."
        )

    return [
        str(c)
        for c in historical_X.columns
    ]


def map_historical_features_to_original(
    historical_features: List[str],
    original_df: pd.DataFrame,
    target_column: str,
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    """
    Match the historical 51 predictors to the original source columns.

    Matching strategy:
    1. exact column name;
    2. normalized name.

    The script FAILS on missing or ambiguous mappings.
    """

    source_columns = [
        c
        for c in original_df.columns
        if c != target_column
    ]

    normalized_source: Dict[str, List[str]] = {}

    for col in source_columns:
        key = normalize_column_name(
            col
        )

        normalized_source.setdefault(
            key,
            [],
        ).append(
            col
        )

    mapped = {}
    audit_rows = []

    for historical_name in historical_features:
        selected = None
        method = None

        if historical_name in original_df.columns:
            selected = historical_name
            method = "exact"

        else:
            key = normalize_column_name(
                historical_name
            )

            candidates = normalized_source.get(
                key,
                [],
            )

            if len(candidates) == 1:
                selected = candidates[0]
                method = "normalized"

            elif len(candidates) > 1:
                raise RuntimeError(
                    "Ambiguous original-column mapping for historical "
                    f"feature '{historical_name}': {candidates}"
                )

        if selected is None:
            raise RuntimeError(
                "Could not map historical feature to original data:\n"
                f"{historical_name}"
            )

        if selected == target_column:
            raise RuntimeError(
                "Target column was mapped into predictors, which would "
                "constitute direct target leakage."
            )

        mapped[
            historical_name
        ] = selected

        audit_rows.append(
            {
                "historical_feature":
                    historical_name,

                "original_source_column":
                    selected,

                "mapping_method":
                    method,
            }
        )

    if len(mapped) != 51:
        raise RuntimeError(
            f"Expected 51 mapped predictors, got {len(mapped)}."
        )

    X = pd.DataFrame(
        {
            historical_name:
                original_df[
                    source_name
                ].copy()

            for historical_name, source_name
            in mapped.items()
        }
    )

    return (
        X,
        audit_rows,
    )


# =============================================================================
# 5. TARGET NORMALIZATION
# =============================================================================

def normalize_binary_target(
    y_raw: pd.Series,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preserve native 0/1 labels whenever possible.

    Otherwise infer a binary mapping only when unambiguous.
    """

    original_non_null = y_raw.dropna()

    if len(original_non_null) != len(y_raw):
        raise RuntimeError(
            "Target contains missing values."
        )

    # Numeric first.
    numeric = pd.to_numeric(
        y_raw,
        errors="coerce",
    )

    if numeric.notna().all():
        unique = sorted(
            numeric.unique().tolist()
        )

        if unique == [0, 1]:
            return (
                numeric.astype(int).to_numpy(),
                {
                    "mapping_type":
                        "native_numeric_0_1",

                    "mapping":
                        {"0": 0, "1": 1},
                },
            )

        if unique == [-1, 1]:
            mapped = (
                numeric
                .map({
                    -1: 0,
                    1: 1,
                })
                .astype(int)
                .to_numpy()
            )

            return (
                mapped,
                {
                    "mapping_type":
                        "numeric_minus1_plus1",

                    "mapping":
                        {"-1": 0, "1": 1},
                },
            )

    # String labels.
    values = (
        y_raw
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unique = sorted(
        values.unique().tolist()
    )

    if len(unique) != 2:
        raise RuntimeError(
            "Expected binary outcome, but target values are:\n"
            f"{unique}"
        )

    # Explicit semantic mappings.
    negative_terms = {
        "recovered",
        "recover",
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
        "mortality",
        "positive",
        "yes",
        "1",
    }

    mapping = {}

    for value in unique:
        if value in negative_terms:
            mapping[value] = 0

        elif value in positive_terms:
            mapping[value] = 1

    if len(mapping) != 2:
        raise RuntimeError(
            "Binary target values cannot be safely assigned to "
            "0/1 without inventing label semantics.\n"
            f"Observed values: {unique}"
        )

    y = (
        values
        .map(mapping)
        .astype(int)
        .to_numpy()
    )

    return (
        y,
        {
            "mapping_type":
                "semantic_binary_mapping",

            "mapping":
                mapping,
        },
    )


# =============================================================================
# 6. NUMERIC FEATURE CLEANING
# =============================================================================

def normalize_numeric_text(
    series: pd.Series,
) -> pd.Series:
    """
    Conservative numeric conversion.

    Removes whitespace and common thousands separators only.
    It does NOT invent category encodings.
    """

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    text = (
        series
        .astype(str)
        .str.strip()
    )

    text = text.replace(
        {
            "":
                np.nan,

            "nan":
                np.nan,

            "None":
                np.nan,

            "none":
                np.nan,

            "NA":
                np.nan,

            "N/A":
                np.nan,

            "-":
                np.nan,
        }
    )

    # Remove standard thousands commas.
    text = text.str.replace(
        ",",
        "",
        regex=False,
    )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def convert_features_to_numeric(
    X: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    converted = pd.DataFrame(
        index=X.index
    )

    audit = []

    for col in X.columns:
        original_nonmissing = int(
            X[col].notna().sum()
        )

        numeric = normalize_numeric_text(
            X[col]
        )

        converted_nonmissing = int(
            numeric.notna().sum()
        )

        newly_failed = (
            original_nonmissing
            - converted_nonmissing
        )

        failure_fraction = (
            newly_failed
            / original_nonmissing

            if original_nonmissing > 0

            else 0.0
        )

        unique_raw_preview = (
            X[col]
            .dropna()
            .astype(str)
            .unique()[:10]
            .tolist()
        )

        audit.append(
            {
                "feature":
                    col,

                "original_nonmissing":
                    original_nonmissing,

                "numeric_nonmissing":
                    converted_nonmissing,

                "newly_failed_numeric_conversions":
                    newly_failed,

                "failure_fraction":
                    failure_fraction,

                "raw_values_preview":
                    safe_json(
                        unique_raw_preview
                    ),
            }
        )

        # Do not silently encode a categorical feature.
        if failure_fraction > 0.05:
            raise RuntimeError(
                f"Feature '{col}' cannot be safely converted to numeric.\n"
                f"{newly_failed}/{original_nonmissing} nonmissing values "
                "failed conversion.\n"
                "The script will not invent a categorical encoding."
            )

        converted[col] = numeric

    return (
        converted,
        audit,
    )


# =============================================================================
# 7. ORIGINAL-DATA DUPLICATE AUDIT
# =============================================================================

def original_duplicate_audit(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    audit_df = X.copy()

    audit_df[
        "__target__"
    ] = y

    exact_dup_mask = audit_df.duplicated(
        keep=False
    )

    feature_dup_mask = X.duplicated(
        keep=False
    )

    rows = []

    for idx in range(
        len(X)
    ):
        rows.append(
            {
                "source_row_index":
                    int(idx),

                "feature_duplicate":
                    int(
                        feature_dup_mask.iloc[
                            idx
                        ]
                    ),

                "feature_plus_target_duplicate":
                    int(
                        exact_dup_mask.iloc[
                            idx
                        ]
                    ),

                "target":
                    int(
                        y[idx]
                    ),
            }
        )

    return (
        audit_df,
        rows,
    )


# =============================================================================
# 8. LEAKAGE-SAFE SPLIT
# =============================================================================

def create_stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Dict[str, Any]:
    indices = np.arange(
        len(X)
    )

    (
        train_idx,
        test_idx,
    ) = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
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

    y_train = (
        y[
            train_idx
        ]
        .astype(int)
    )

    y_test = (
        y[
            test_idx
        ]
        .astype(int)
    )

    return {
        "train_source_indices":
            np.asarray(
                train_idx,
                dtype=int,
            ),

        "test_source_indices":
            np.asarray(
                test_idx,
                dtype=int,
            ),

        "X_train_raw":
            X_train_raw,

        "X_test_raw":
            X_test_raw,

        "y_train":
            y_train,

        "y_test":
            y_test,
    }


# =============================================================================
# 9. SPLIT DUPLICATION CHECK
# =============================================================================

def normalized_row_signature(
    row: pd.Series,
) -> Tuple[Any, ...]:
    signature = []

    for value in row.tolist():
        if pd.isna(value):
            signature.append(
                "__MISSING__"
            )
        else:
            signature.append(
                float(value)
            )

    return tuple(
        signature
    )


def detect_cross_split_overlap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> List[Dict[str, Any]]:
    train_map: Dict[
        Tuple[Any, ...],
        List[int],
    ] = {}

    for i in range(
        len(X_train)
    ):
        sig = normalized_row_signature(
            X_train.iloc[i]
        )

        train_map.setdefault(
            sig,
            [],
        ).append(i)

    matches = []

    for j in range(
        len(X_test)
    ):
        sig = normalized_row_signature(
            X_test.iloc[j]
        )

        if sig not in train_map:
            continue

        for i in train_map[sig]:
            matches.append(
                {
                    "train_row":
                        int(i),

                    "test_row":
                        int(j),
                }
            )

    return matches


# =============================================================================
# 10. TRAIN-ONLY PREPROCESSING
# =============================================================================

def fit_train_only_preprocessing(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Every learned transformation is fitted ONLY on the training split.
    """

    # ---------------------------------------------------------
    # Median imputation
    # ---------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train_raw
    )

    X_test = imputer.transform(
        X_test_raw
    )

    # ---------------------------------------------------------
    # MinMax scaler
    # ---------------------------------------------------------

    minmax = None

    if USE_MINMAX_SCALER:
        minmax = MinMaxScaler()

        X_train = minmax.fit_transform(
            X_train
        )

        X_test = minmax.transform(
            X_test
        )

    # ---------------------------------------------------------
    # Standard scaler
    # ---------------------------------------------------------

    standard = None

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
            "Non-finite values appeared in processed training features."
        )

    if not np.isfinite(
        X_test
    ).all():
        raise RuntimeError(
            "Non-finite values appeared in processed test features."
        )

    provenance = {
        "imputer":
            "SimpleImputer(strategy='median')",

        "imputer_fit_scope":
            "training_only",

        "minmax_scaler":
            (
                "MinMaxScaler"
                if USE_MINMAX_SCALER
                else "disabled"
            ),

        "minmax_fit_scope":
            (
                "training_only"
                if USE_MINMAX_SCALER
                else "not_applicable"
            ),

        "standard_scaler":
            (
                "StandardScaler"
                if USE_STANDARD_SCALER
                else "disabled"
            ),

        "standard_fit_scope":
            (
                "training_only"
                if USE_STANDARD_SCALER
                else "not_applicable"
            ),

        "n_features":
            int(
                X_train.shape[1]
            ),
    }

    return (
        X_train,
        X_test,
        provenance,
    )


# =============================================================================
# 11. LOAD ORIGINAL ENSEMBLE CONFIGURATION
# =============================================================================

def load_preserved_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Preserved model does not exist:\n{MODEL_PATH}"
        )

    print(
        "\nLoading preserved classifier configuration:"
    )

    print(
        MODEL_PATH
    )

    try:
        model = joblib.load(
            MODEL_PATH
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not load preserved ensemble model.\n"
            "Run this script in the compatible scikit-learn 1.5.2 "
            "environment.\n\n"
            f"Original error:\n{repr(exc)}"
        )

    return model


def inspect_preserved_model(
    model: Any,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_class":
            f"{model.__class__.__module__}.{model.__class__.__name__}",

        "runtime_sklearn_version":
            sklearn.__version__,
    }

    if hasattr(
        model,
        "voting",
    ):
        info[
            "voting"
        ] = model.voting

    if hasattr(
        model,
        "weights",
    ):
        info[
            "weights"
        ] = safe_json(
            model.weights
        )

    if hasattr(
        model,
        "flatten_transform",
    ):
        info[
            "flatten_transform"
        ] = model.flatten_transform

    if hasattr(
        model,
        "estimators",
    ):
        estimator_info = []

        for name, estimator in model.estimators:
            estimator_info.append(
                {
                    "name":
                        name,

                    "class":
                        (
                            f"{estimator.__class__.__module__}."
                            f"{estimator.__class__.__name__}"
                        ),

                    "params":
                        estimator.get_params(
                            deep=False
                        ),
                }
            )

        info[
            "estimators"
        ] = safe_json(
            estimator_info
        )

    if hasattr(
        model,
        "get_params",
    ):
        try:
            info[
                "ensemble_params"
            ] = safe_json(
                model.get_params(
                    deep=False
                )
            )
        except Exception:
            pass

    return info


def create_unfitted_ensemble_clone(
    preserved_model: Any,
) -> Any:
    """
    sklearn.clone() preserves estimator hyperparameters but removes
    fitted state.

    This is exactly what we want: same classifier specification,
    freshly fitted on leakage-safe training data.
    """

    try:
        new_model = clone(
            preserved_model
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to clone the preserved ensemble architecture.\n"
            "The script will not replace it with invented defaults.\n\n"
            f"clone() error:\n{repr(exc)}"
        )

    return new_model


# =============================================================================
# 12. TRAIN-ONLY RANDOM OVERSAMPLING
# =============================================================================

def oversample_training_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Randomly oversample the minority class to the majority count.

    IMPORTANT:
    This occurs AFTER split and AFTER train-only preprocessing.

    No test rows are included.
    """

    classes, counts = np.unique(
        y_train,
        return_counts=True,
    )

    if len(classes) != 2:
        raise RuntimeError(
            "Training-only oversampling currently requires binary labels."
        )

    class_counts_before = {
        int(cls):
            int(count)

        for cls, count
        in zip(
            classes,
            counts,
        )
    }

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
        np.max(
            counts
        )
    )

    majority_idx = np.where(
        y_train == majority_class
    )[0]

    minority_idx = np.where(
        y_train == minority_class
    )[0]

    if len(
        minority_idx
    ) == majority_count:
        return (
            X_train.copy(),
            y_train.copy(),
            {
                "method":
                    "none_already_balanced",

                "class_counts_before":
                    safe_json(
                        class_counts_before
                    ),

                "class_counts_after":
                    safe_json(
                        class_counts_before
                    ),

                "generated_duplicate_training_rows":
                    0,
            },
        )

    minority_resampled_idx = resample(
        minority_idx,
        replace=True,
        n_samples=majority_count,
        random_state=RANDOM_STATE,
    )

    final_idx = np.concatenate(
        [
            majority_idx,
            minority_resampled_idx,
        ]
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    rng.shuffle(
        final_idx
    )

    X_balanced = X_train[
        final_idx
    ]

    y_balanced = y_train[
        final_idx
    ]

    _, after_counts = np.unique(
        y_balanced,
        return_counts=True,
    )

    class_counts_after = {
        int(cls):
            int(count)

        for cls, count
        in zip(
            classes,
            after_counts,
        )
    }

    generated_duplicates = (
        majority_count
        - len(
            minority_idx
        )
    )

    provenance = {
        "method":
            "RandomOverSampling_using_sklearn.utils.resample",

        "scope":
            "training_only",

        "random_state":
            RANDOM_STATE,

        "minority_class":
            int(
                minority_class
            ),

        "majority_class":
            int(
                majority_class
            ),

        "class_counts_before":
            safe_json(
                class_counts_before
            ),

        "class_counts_after":
            safe_json(
                class_counts_after
            ),

        "generated_duplicate_training_rows":
            int(
                generated_duplicates
            ),
    }

    return (
        X_balanced,
        y_balanced,
        provenance,
    )


# =============================================================================
# 13. PREDICTION HELPERS
# =============================================================================

def get_positive_class_score(
    model: Any,
    X: np.ndarray,
) -> Tuple[
    Optional[np.ndarray],
    str,
]:
    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = model.predict_proba(
            X
        )

        if probabilities.ndim != 2:
            raise RuntimeError(
                "predict_proba returned an unexpected shape."
            )

        classes = getattr(
            model,
            "classes_",
            None,
        )

        if classes is None:
            raise RuntimeError(
                "Model exposes predict_proba but has no classes_."
            )

        classes_list = list(
            classes
        )

        if 1 not in classes_list:
            raise RuntimeError(
                f"Positive class 1 is absent from model.classes_: {classes_list}"
            )

        positive_col = classes_list.index(
            1
        )

        return (
            probabilities[
                :,
                positive_col
            ].astype(float),

            "predict_proba_positive_class",
        )

    if hasattr(
        model,
        "decision_function",
    ):
        scores = model.decision_function(
            X
        )

        scores = np.asarray(
            scores
        ).reshape(-1)

        return (
            scores.astype(float),
            "decision_function",
        )

    return (
        None,
        "unavailable",
    )


# =============================================================================
# 14. METRICS
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
) -> Dict[str, Any]:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        int(cm[0, 0]),
        int(cm[0, 1]),
        int(cm[1, 0]),
        int(cm[1, 1]),
    )

    specificity = (
        tn / (tn + fp)
        if (
            tn + fp
        ) > 0
        else np.nan
    )

    result = {
        "n_test":
            int(
                len(y_true)
            ),

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

        "recall_sensitivity":
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

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,
    }

    if (
        y_score is not None
        and len(
            np.unique(
                y_true
            )
        ) == 2
    ):
        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

    else:
        result[
            "roc_auc"
        ] = np.nan

    return result


# =============================================================================
# 15. RUN A SINGLE VARIANT
# =============================================================================

def run_variant(
    variant_name: str,
    preserved_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_source_indices: np.ndarray,
    feature_names: List[str],
    balancing_info: Dict[str, Any],
) -> Dict[str, Any]:
    print(
        "\n" + "=" * 88
    )

    print(
        f"VARIANT: {variant_name}"
    )

    print(
        "=" * 88
    )

    model = create_unfitted_ensemble_clone(
        preserved_model
    )

    print(
        f"Training rows: {len(y_train)}"
    )

    print(
        f"Training class counts: "
        f"{dict(Counter(y_train.tolist()))}"
    )

    print(
        f"Untouched test rows: {len(y_test)}"
    )

    # ---------------------------------------------------------
    # Fit only on training-side data
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Predict untouched test
    # ---------------------------------------------------------

    y_pred_native = model.predict(
        X_test
    )

    y_pred_native = np.asarray(
        y_pred_native
    ).reshape(-1)

    # Expect actual binary predictions.
    unique_pred = set(
        np.unique(
            y_pred_native
        ).tolist()
    )

    if not unique_pred.issubset(
        {0, 1}
    ):
        raise RuntimeError(
            f"Unexpected predicted classes: {sorted(unique_pred)}"
        )

    y_pred = y_pred_native.astype(
        int
    )

    y_score, score_source = get_positive_class_score(
        model,
        X_test,
    )

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )

    # ---------------------------------------------------------
    # Output row-level predictions
    # ---------------------------------------------------------

    prediction_rows = []

    for i in range(
        len(y_test)
    ):
        row = {
            "variant":
                variant_name,

            "test_row_id":
                int(i),

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
                    y_test[
                        i
                    ]
                    == y_pred[
                        i
                    ]
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

        prediction_rows.append(
            row
        )

    prediction_path = (
        OUTPUT_DIR
        / f"predictions_{variant_name}.csv"
    )

    write_csv(
        prediction_path,
        prediction_rows,
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metric_row = {
        "variant":
            variant_name,

        **metrics,

        "score_source":
            score_source,

        "classification_threshold":
            CLASSIFICATION_THRESHOLD,

        "model_source":
            relative_path(
                MODEL_PATH
            ),

        "model_configuration":
            "cloned_preserved_ensemble",

        "balancing":
            safe_json(
                balancing_info
            ),
    }

    write_csv(
        OUTPUT_DIR
        / f"metrics_{variant_name}.csv",
        [
            metric_row
        ],
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm_rows = [
        {
            "actual_class":
                0,

            "predicted_0":
                metrics[
                    "tn"
                ],

            "predicted_1":
                metrics[
                    "fp"
                ],
        },
        {
            "actual_class":
                1,

            "predicted_0":
                metrics[
                    "fn"
                ],

            "predicted_1":
                metrics[
                    "tp"
                ],
        },
    ]

    write_csv(
        OUTPUT_DIR
        / f"confusion_matrix_{variant_name}.csv",
        cm_rows,
    )

    # ---------------------------------------------------------
    # Save freshly fitted corrected model as a NEW artifact
    # ---------------------------------------------------------

    corrected_model_path = (
        OUTPUT_DIR
        / f"model_{variant_name}.joblib"
    )

    joblib.dump(
        model,
        corrected_model_path,
    )

    metric_row[
        "saved_corrected_model"
    ] = relative_path(
        corrected_model_path
    )

    print(
        f"Accuracy : {metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: {metrics['precision']:.6f}"
    )

    print(
        f"Recall   : {metrics['recall_sensitivity']:.6f}"
    )

    print(
        f"Specificity: {metrics['specificity']:.6f}"
    )

    print(
        f"F1       : {metrics['f1']:.6f}"
    )

    if np.isfinite(
        metrics[
            "roc_auc"
        ]
    ):
        print(
            f"ROC-AUC  : {metrics['roc_auc']:.6f}"
        )
    else:
        print(
            "ROC-AUC  : unavailable"
        )

    print(
        f"CM       : "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )

    return {
        "metric_row":
            metric_row,

        "prediction_rows":
            prediction_rows,

        "model":
            model,

        "model_path":
            corrected_model_path,
    }


# =============================================================================
# 16. HISTORICAL SPLIT COMPARISON
# =============================================================================

def historical_split_summary() -> Dict[str, Any]:
    result = {}

    if HISTORICAL_X_TRAIN_PATH.exists():
        old_X_train = read_csv_robust(
            HISTORICAL_X_TRAIN_PATH
        )

        result[
            "historical_train_rows"
        ] = len(
            old_X_train
        )

    if HISTORICAL_X_TEST_PATH.exists():
        old_X_test = read_csv_robust(
            HISTORICAL_X_TEST_PATH
        )

        result[
            "historical_test_rows"
        ] = len(
            old_X_test
        )

    if HISTORICAL_Y_TRAIN_PATH.exists():
        old_y_train = read_csv_robust(
            HISTORICAL_Y_TRAIN_PATH
        )

        ycol = old_y_train.columns[
            0
        ]

        result[
            "historical_train_class_counts"
        ] = safe_json(
            old_y_train[
                ycol
            ].value_counts(
                dropna=False
            ).to_dict()
        )

    if HISTORICAL_Y_TEST_PATH.exists():
        old_y_test = read_csv_robust(
            HISTORICAL_Y_TEST_PATH
        )

        ycol = old_y_test.columns[
            0
        ]

        result[
            "historical_test_class_counts"
        ] = safe_json(
            old_y_test[
                ycol
            ].value_counts(
                dropna=False
            ).to_dict()
        )

    return result


# =============================================================================
# 17. MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 88
    )

    print(
        "HFAGM - CORRECTED LEAKAGE-SAFE EVALUATION"
    )

    print(
        "=" * 88
    )

    print(
        f"\nPython: {sys.version.split()[0]}"
    )

    print(
        f"scikit-learn: {sklearn.__version__}"
    )

    print(
        f"numpy: {np.__version__}"
    )

    print(
        f"pandas: {pd.__version__}"
    )

    # ---------------------------------------------------------
    # Load original source
    # ---------------------------------------------------------

    print(
        "\nLoading original clinical dataset..."
    )

    (
        original_df,
        original_path,
        source_type,
    ) = load_original_dataset()

    print(
        f"Source: {original_path}"
    )

    print(
        f"Source type: {source_type}"
    )

    print(
        f"Rows: {len(original_df)}"
    )

    print(
        f"Columns: {original_df.shape[1]}"
    )

    if len(
        original_df
    ) != 193:
        raise RuntimeError(
            f"Expected 193 original records; found {len(original_df)}."
        )

    # ---------------------------------------------------------
    # Identify target
    # ---------------------------------------------------------

    target_column = identify_target_column(
        original_df
    )

    print(
        f"Target column: {target_column}"
    )

    y, target_mapping_info = normalize_binary_target(
        original_df[
            target_column
        ]
    )

    print(
        f"Target counts: {dict(Counter(y.tolist()))}"
    )

    if len(
        np.unique(
            y
        )
    ) != 2:
        raise RuntimeError(
            "Corrected evaluation requires a binary outcome."
        )

    # ---------------------------------------------------------
    # Recover exact historical feature definition
    # ---------------------------------------------------------

    historical_features = load_historical_feature_names()

    print(
        f"Historical predictor count: {len(historical_features)}"
    )

    (
        X_source,
        feature_mapping_rows,
    ) = map_historical_features_to_original(
        historical_features,
        original_df,
        target_column,
    )

    write_csv(
        OUTPUT_DIR
        / "feature_mapping.csv",
        feature_mapping_rows,
    )

    # ---------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------

    (
        X_numeric,
        numeric_conversion_rows,
    ) = convert_features_to_numeric(
        X_source
    )

    write_csv(
        OUTPUT_DIR
        / "numeric_conversion_audit.csv",
        numeric_conversion_rows,
    )

    if X_numeric.shape != (
        193,
        51,
    ):
        raise RuntimeError(
            "Unexpected final predictor matrix shape: "
            f"{X_numeric.shape}; expected (193, 51)."
        )

    # ---------------------------------------------------------
    # Original duplicate audit
    # ---------------------------------------------------------

    _, original_dup_rows = original_duplicate_audit(
        X_numeric,
        y,
    )

    write_csv(
        OUTPUT_DIR
        / "original_193_duplicate_audit.csv",
        original_dup_rows,
    )

    n_feature_duplicate_rows = sum(
        row[
            "feature_duplicate"
        ]
        for row
        in original_dup_rows
    )

    n_feature_target_duplicate_rows = sum(
        row[
            "feature_plus_target_duplicate"
        ]
        for row
        in original_dup_rows
    )

    print(
        f"\nOriginal rows participating in feature duplication: "
        f"{n_feature_duplicate_rows}"
    )

    print(
        f"Original rows participating in feature+target duplication: "
        f"{n_feature_target_duplicate_rows}"
    )

    # ---------------------------------------------------------
    # Create split BEFORE learned preprocessing
    # ---------------------------------------------------------

    print(
        "\nCreating leakage-safe stratified split BEFORE preprocessing..."
    )

    split = create_stratified_split(
        X_numeric,
        y,
    )

    train_idx = split[
        "train_source_indices"
    ]

    test_idx = split[
        "test_source_indices"
    ]

    X_train_raw = split[
        "X_train_raw"
    ]

    X_test_raw = split[
        "X_test_raw"
    ]

    y_train = split[
        "y_train"
    ]

    y_test = split[
        "y_test"
    ]

    print(
        f"Corrected training rows: {len(y_train)}"
    )

    print(
        f"Corrected test rows: {len(y_test)}"
    )

    print(
        f"Training counts: {dict(Counter(y_train.tolist()))}"
    )

    print(
        f"Test counts: {dict(Counter(y_test.tolist()))}"
    )

    # ---------------------------------------------------------
    # Check that original-row split itself has no exact overlap
    # ---------------------------------------------------------

    overlap_pairs = detect_cross_split_overlap(
        X_train_raw,
        X_test_raw,
    )

    print(
        f"Exact feature overlap pairs in corrected raw split: "
        f"{len(overlap_pairs)}"
    )

    write_csv(
        OUTPUT_DIR
        / "corrected_split_overlap_check.csv",
        overlap_pairs,
        columns=[
            "train_row",
            "test_row",
        ],
    )

    # If original dataset genuinely contains duplicate participants/rows,
    # a row-level split could still place identical source records across
    # train/test. Do NOT allow that silently.
    if overlap_pairs:
        raise RuntimeError(
            "\nThe original 193-record dataset itself produces exact "
            "cross-split duplicate feature vectors under the requested "
            "random split.\n"
            f"Detected overlap pairs: {len(overlap_pairs)}\n\n"
            "A grouped/deduplicated patient-level split is required "
            "before training. The script has stopped rather than "
            "claiming a leakage-safe evaluation."
        )

    # ---------------------------------------------------------
    # Save split manifest BEFORE preprocessing
    # ---------------------------------------------------------

    split_manifest_rows = []

    for corrected_train_row, source_idx in enumerate(
        train_idx
    ):
        split_manifest_rows.append(
            {
                "partition":
                    "train",

                "corrected_partition_row":
                    int(
                        corrected_train_row
                    ),

                "original_source_row_index":
                    int(
                        source_idx
                    ),

                "target":
                    int(
                        y[
                            source_idx
                        ]
                    ),
            }
        )

    for corrected_test_row, source_idx in enumerate(
        test_idx
    ):
        split_manifest_rows.append(
            {
                "partition":
                    "test",

                "corrected_partition_row":
                    int(
                        corrected_test_row
                    ),

                "original_source_row_index":
                    int(
                        source_idx
                    ),

                "target":
                    int(
                        y[
                            source_idx
                        ]
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "corrected_split_manifest.csv",
        split_manifest_rows,
    )

    # ---------------------------------------------------------
    # Fit preprocessing STRICTLY on training
    # ---------------------------------------------------------

    print(
        "\nFitting preprocessing on training data only..."
    )

    (
        X_train_processed,
        X_test_processed,
        preprocessing_info,
    ) = fit_train_only_preprocessing(
        X_train_raw,
        X_test_raw,
    )

    # Save matrices for reproducibility.
    pd.DataFrame(
        X_train_processed,
        columns=historical_features,
    ).to_csv(
        OUTPUT_DIR
        / "X_train_corrected_scaled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        X_test_processed,
        columns=historical_features,
    ).to_csv(
        OUTPUT_DIR
        / "X_test_corrected_scaled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "status":
                y_train
        }
    ).to_csv(
        OUTPUT_DIR
        / "y_train_corrected.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "status":
                y_test
        }
    ).to_csv(
        OUTPUT_DIR
        / "y_test_corrected.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # Load preserved classifier specification
    # ---------------------------------------------------------

    preserved_model = load_preserved_model()

    preserved_model_info = inspect_preserved_model(
        preserved_model
    )

    write_csv(
        OUTPUT_DIR
        / "preserved_model_configuration.csv",
        [
            preserved_model_info
        ],
    )

    # ---------------------------------------------------------
    # Run corrected variants
    # ---------------------------------------------------------

    variant_results = []

    # ---- Variant A: no balancing ---------------------------------

    if RUN_UNBALANCED_VARIANT:
        no_balance_info = {
            "method":
                "none",

            "scope":
                "training_only_no_resampling",

            "class_counts":
                safe_json(
                    dict(
                        Counter(
                            y_train.tolist()
                        )
                    )
                ),
        }

        result = run_variant(
            variant_name="unbalanced_training",
            preserved_model=preserved_model,
            X_train=X_train_processed,
            y_train=y_train,
            X_test=X_test_processed,
            y_test=y_test,
            test_source_indices=test_idx,
            feature_names=historical_features,
            balancing_info=no_balance_info,
        )

        variant_results.append(
            result[
                "metric_row"
            ]
        )

    # ---- Variant B: training-only oversampling --------------------

    if RUN_TRAIN_ONLY_OVERSAMPLING_VARIANT:
        (
            X_train_balanced,
            y_train_balanced,
            oversampling_info,
        ) = oversample_training_only(
            X_train_processed,
            y_train,
        )

        pd.DataFrame(
            X_train_balanced,
            columns=historical_features,
        ).to_csv(
            OUTPUT_DIR
            / "X_train_corrected_training_only_balanced.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            {
                "status":
                    y_train_balanced
            }
        ).to_csv(
            OUTPUT_DIR
            / "y_train_corrected_training_only_balanced.csv",
            index=False,
            encoding="utf-8-sig",
        )

        result = run_variant(
            variant_name="training_only_oversampled",
            preserved_model=preserved_model,
            X_train=X_train_balanced,
            y_train=y_train_balanced,
            X_test=X_test_processed,
            y_test=y_test,
            test_source_indices=test_idx,
            feature_names=historical_features,
            balancing_info=oversampling_info,
        )

        variant_results.append(
            result[
                "metric_row"
            ]
        )

    # ---------------------------------------------------------
    # Consolidated metrics
    # ---------------------------------------------------------

    write_csv(
        OUTPUT_DIR
        / "corrected_evaluation_metrics.csv",
        variant_results,
        columns=[
            "variant",
            "n_test",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )

    # ---------------------------------------------------------
    # Comparison between variants
    # ---------------------------------------------------------

    comparison_rows = []

    if len(
        variant_results
    ) >= 2:
        first = variant_results[0]
        second = variant_results[1]

        for metric in [
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
            "roc_auc",
        ]:
            v1 = first.get(
                metric
            )

            v2 = second.get(
                metric
            )

            if (
                v1 is not None
                and v2 is not None
                and np.isfinite(
                    float(v1)
                )
                and np.isfinite(
                    float(v2)
                )
            ):
                delta = float(
                    v2
                ) - float(
                    v1
                )

            else:
                delta = np.nan

            comparison_rows.append(
                {
                    "metric":
                        metric,

                    "unbalanced_training":
                        v1,

                    "training_only_oversampled":
                        v2,

                    "delta_oversampled_minus_unbalanced":
                        delta,
                }
            )

    write_csv(
        OUTPUT_DIR
        / "variant_comparison.csv",
        comparison_rows,
    )

    # ---------------------------------------------------------
    # Provenance
    # ---------------------------------------------------------

    historical_summary = historical_split_summary()

    provenance = {
        "generated":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "original_source":
            str(
                original_path
            ),

        "original_source_type":
            source_type,

        "original_rows":
            int(
                len(original_df)
            ),

        "original_columns":
            int(
                original_df.shape[1]
            ),

        "original_sha256":
            sha256_file(
                original_path
            ),

        "original_dataframe_sha256":
            sha256_dataframe(
                original_df
            ),

        "target_column":
            target_column,

        "target_mapping":
            safe_json(
                target_mapping_info
            ),

        "predictor_count":
            len(
                historical_features
            ),

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "split_method":
            "sklearn.model_selection.train_test_split",

        "stratified":
            True,

        "split_before_learned_preprocessing":
            True,

        "corrected_train_rows":
            len(
                y_train
            ),

        "corrected_test_rows":
            len(
                y_test
            ),

        "corrected_train_class_counts":
            safe_json(
                dict(
                    Counter(
                        y_train.tolist()
                    )
                )
            ),

        "corrected_test_class_counts":
            safe_json(
                dict(
                    Counter(
                        y_test.tolist()
                    )
                )
            ),

        "corrected_cross_split_exact_overlap_pairs":
            len(
                overlap_pairs
            ),

        "preprocessing":
            safe_json(
                preprocessing_info
            ),

        "preserved_model_path":
            str(
                MODEL_PATH
            ),

        "preserved_model_sha256":
            sha256_file(
                MODEL_PATH
            ),

        "preserved_model_configuration_used":
            True,

        "threshold_tuned":
            False,

        "test_set_used_for_model_selection":
            False,

        "test_set_used_for_preprocessing_fit":
            False,

        "test_set_used_for_balancing":
            False,

        "historical_split_summary":
            safe_json(
                historical_summary
            ),

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
    }

    write_csv(
        OUTPUT_DIR
        / "corrected_evaluation_provenance.csv",
        [
            provenance
        ],
    )

    # ---------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------

    lines = [
        "=" * 88,
        "HFAGM - CORRECTED LEAKAGE-SAFE EVALUATION",
        "=" * 88,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "WHY THIS EVALUATION WAS REQUIRED",
        "-" * 88,
        (
            "The historical 160/40 evaluation split was previously found "
            "to contain exact train/test duplicates."
        ),
        (
            "The historical perfect 1.000 test metrics therefore must not "
            "be treated as independently validated performance."
        ),
        "",
        "CORRECTED DATA SOURCE",
        "-" * 88,
        f"Source: {original_path}",
        f"Source type: {source_type}",
        f"Original observations: {len(original_df)}",
        f"Target column: {target_column}",
        f"Predictors: {len(historical_features)}",
        f"Target counts: {dict(Counter(y.tolist()))}",
        "",
        "CORRECTED PROTOCOL",
        "-" * 88,
        "1. Start from the original 193 observations.",
        "2. Define the same historical 51 predictor features.",
        "3. Perform stratified train/test splitting BEFORE learned preprocessing.",
        "4. Fit median imputation on training only.",
        "5. Fit MinMaxScaler on training only.",
        "6. Fit StandardScaler on training only.",
        "7. Transform the untouched test partition using training-fitted objects.",
        "8. Clone the preserved ensemble architecture and hyperparameters.",
        "9. Fit classifiers exclusively on training-side data.",
        "10. Evaluate on the same untouched real test set.",
        "",
        "CORRECTED SPLIT",
        "-" * 88,
        f"Training observations: {len(y_train)}",
        f"Test observations: {len(y_test)}",
        f"Training class counts: {dict(Counter(y_train.tolist()))}",
        f"Test class counts: {dict(Counter(y_test.tolist()))}",
        f"Exact train/test feature overlap pairs: {len(overlap_pairs)}",
        "",
        "ORIGINAL-DATA DUPLICATE AUDIT",
        "-" * 88,
        (
            "Original rows participating in duplicated feature vectors: "
            f"{n_feature_duplicate_rows}"
        ),
        (
            "Original rows participating in duplicated feature+target rows: "
            f"{n_feature_target_duplicate_rows}"
        ),
        "",
        "CORRECTED RESULTS",
        "-" * 88,
    ]

    for row in variant_results:
        lines.extend(
            [
                f"Variant: {row['variant']}",
                f"  N test       : {row['n_test']}",
                f"  Accuracy     : {row['accuracy']:.10f}",
                f"  Precision    : {row['precision']:.10f}",
                f"  Sensitivity  : {row['recall_sensitivity']:.10f}",
                f"  Specificity  : {row['specificity']:.10f}",
                f"  F1           : {row['f1']:.10f}",
                (
                    f"  ROC-AUC      : {row['roc_auc']:.10f}"
                    if np.isfinite(
                        float(
                            row[
                                "roc_auc"
                            ]
                        )
                    )
                    else
                    "  ROC-AUC      : unavailable"
                ),
                (
                    f"  Confusion    : "
                    f"TN={row['tn']}, "
                    f"FP={row['fp']}, "
                    f"FN={row['fn']}, "
                    f"TP={row['tp']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "INTERPRETATION RULE",
            "-" * 88,
            (
                "These corrected results replace the contaminated 1.000 "
                "metrics for scientific interpretation only if the script "
                "completed with zero corrected train/test overlap and used "
                "a true raw 193-row data source."
            ),
            "",
            (
                "If source_type is 'preprocessed_fallback', the results "
                "remain provisional because that file may already contain "
                "full-dataset preprocessing."
            ),
            "",
            "NO historical file was overwritten.",
            "NO threshold was tuned on the test set.",
            "NO test observation was used for training-side oversampling.",
            "NO test observation was used to fit preprocessing.",
            "",
            "PRIMARY OUTPUTS",
            "-" * 88,
            "corrected_evaluation_metrics.csv",
            "variant_comparison.csv",
            "predictions_unbalanced_training.csv",
            "predictions_training_only_oversampled.csv",
            "confusion_matrix_unbalanced_training.csv",
            "confusion_matrix_training_only_oversampled.csv",
            "corrected_split_manifest.csv",
            "corrected_split_overlap_check.csv",
            "corrected_evaluation_provenance.csv",
            "preserved_model_configuration.csv",
            "feature_mapping.csv",
            "numeric_conversion_audit.csv",
            "original_193_duplicate_audit.csv",
            "",
            "=" * 88,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "corrected_evaluation_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 88
    )

    print(
        "02D COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        f"\nCorrected split:"
    )

    print(
        f"Train = {len(y_train)}"
    )

    print(
        f"Test  = {len(y_test)}"
    )

    print(
        f"Exact corrected cross-split overlap = {len(overlap_pairs)}"
    )

    print(
        "\nResults written to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nUpload these files:"
    )

    for filename in [
        "corrected_evaluation_summary.txt",
        "corrected_evaluation_metrics.csv",
        "variant_comparison.csv",
        "predictions_unbalanced_training.csv",
        "predictions_training_only_oversampled.csv",
        "confusion_matrix_unbalanced_training.csv",
        "confusion_matrix_training_only_oversampled.csv",
        "corrected_split_manifest.csv",
        "corrected_split_overlap_check.csv",
        "corrected_evaluation_provenance.csv",
        "preserved_model_configuration.csv",
        "feature_mapping.csv",
        "original_193_duplicate_audit.csv",
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
            "\n" + "=" * 88
        )

        print(
            "02D FAILED SAFELY"
        )

        print(
            "=" * 88
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo historical dataset, split, or model was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(
            1
        )