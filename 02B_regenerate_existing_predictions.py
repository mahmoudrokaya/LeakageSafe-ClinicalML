"""
02B_regenerate_existing_predictions.py
======================================

Purpose
-------
Regenerate predictions from the EXISTING saved classifier using the
preserved real-data test split.

This script:

1. Loads:
       data/preprocessed/X_test_scaled.csv
       data/preprocessed/y_test.csv
       models/classifiers/ensemble_model.pkl

2. Regenerates:
       y_true
       y_pred
       y_prob, where genuine probability output is supported

3. Recomputes:
       Accuracy
       Precision
       Recall / Sensitivity
       Specificity
       F1-score
       ROC-AUC
       Confusion matrix

4. Saves a canonical row-level prediction file for all subsequent analyses.

5. Performs consistency and provenance checks.

IMPORTANT
---------
This script DOES NOT:
- retrain the classifier;
- modify the saved classifier;
- tune thresholds;
- fabricate probabilities;
- infer ROC-AUC from hard predictions;
- change the preserved test split.

Outputs
-------
outputs/revision_primary_metrics/regenerated_existing/

    regenerated_existing_predictions.csv
    regenerated_existing_metrics.csv
    regenerated_confusion_matrix.csv
    model_provenance.csv
    test_data_audit.csv
    regeneration_summary.txt

Requirements
------------
numpy
pandas
scikit-learn
joblib
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

try:
    import joblib
except ImportError:
    joblib = None


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

X_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "X_test_scaled.csv"
)

Y_TEST_PATH = (
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
    / "regenerated_existing"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "regenerated_existing_predictions.csv"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "regenerated_existing_metrics.csv"
)

CONFUSION_PATH = (
    OUTPUT_DIR
    / "regenerated_confusion_matrix.csv"
)

MODEL_PROVENANCE_PATH = (
    OUTPUT_DIR
    / "model_provenance.csv"
)

TEST_AUDIT_PATH = (
    OUTPUT_DIR
    / "test_data_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "regeneration_summary.txt"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DEFAULT_THRESHOLD = 0.5

# We do NOT automatically force feature order changes unless the
# trained model explicitly exposes feature_names_in_.
STRICT_FEATURE_ALIGNMENT = True


# ============================================================
# 3. UTILITIES
# ============================================================

def normalize_column_name(value: Any) -> str:
    """
    Normalize a column name only for comparison.
    Original column names are preserved for model inference.
    """

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1


def safe_modified_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat(timespec="seconds")
    except Exception:
        return ""


def save_single_row_csv(
    path: Path,
    row: Dict[str, Any]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(
        [row]
    ).to_csv(
        path,
        index=False
    )


def require_file(
    path: Path,
    description: str
) -> None:

    if not path.exists():

        print(
            f"\nERROR: Required {description} not found:"
        )

        print(path)

        sys.exit(1)


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

def load_x_test() -> pd.DataFrame:

    print("\nLoading X_test:")
    print(X_TEST_PATH)

    df = pd.read_csv(
        X_TEST_PATH
    )

    if df.empty:
        raise RuntimeError(
            "X_test_scaled.csv is empty."
        )

    print(
        f"X_test shape: {df.shape}"
    )

    return df


def identify_label_column(
    df: pd.DataFrame
) -> str:
    """
    y_test.csv is expected to contain one target column.

    If more than one column exists, try standard target names.
    """

    if df.shape[1] == 1:
        return df.columns[0]

    preferred_names = {
        "y",
        "label",
        "labels",
        "target",
        "outcome",
        "class",
        "y_test",
    }

    normalized_map = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for name in preferred_names:

        if name in normalized_map:
            return normalized_map[name]

    raise RuntimeError(
        "y_test.csv contains multiple columns and the "
        "target column cannot be identified safely. "
        f"Columns: {list(df.columns)}"
    )


def load_y_test() -> Tuple[np.ndarray, str]:

    print("\nLoading y_test:")
    print(Y_TEST_PATH)

    df = pd.read_csv(
        Y_TEST_PATH
    )

    if df.empty:
        raise RuntimeError(
            "y_test.csv is empty."
        )

    target_col = identify_label_column(
        df
    )

    y_numeric = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    if y_numeric.isna().any():

        bad_count = int(
            y_numeric.isna().sum()
        )

        raise RuntimeError(
            f"y_test contains {bad_count} "
            "non-numeric/missing labels."
        )

    y = y_numeric.to_numpy()

    # Standardize common binary representations.
    unique = sorted(
        np.unique(y).tolist()
    )

    if set(unique).issubset(
        {0, 1}
    ):
        y = y.astype(int)

    elif set(unique).issubset(
        {-1, 1}
    ):
        y = (
            y > 0
        ).astype(int)

    else:
        raise RuntimeError(
            "This regeneration script currently expects binary "
            f"classification. Found labels: {unique}"
        )

    print(
        f"y_test samples: {len(y)}"
    )

    print(
        f"Class counts: "
        f"0={(y == 0).sum()}, "
        f"1={(y == 1).sum()}"
    )

    return y, target_col


# ============================================================
# 5. LOAD EXISTING MODEL
# ============================================================

def load_existing_model(
    path: Path
) -> Tuple[Any, str]:

    print("\nLoading existing model:")
    print(path)

    errors: List[str] = []

    # Prefer joblib for sklearn-style artifacts.
    if joblib is not None:

        try:

            model = joblib.load(
                path
            )

            return model, "joblib"

        except Exception as exc:

            errors.append(
                f"joblib: {repr(exc)}"
            )

    # Fallback to standard pickle.
    try:

        with path.open(
            "rb"
        ) as f:

            model = pickle.load(f)

        return model, "pickle"

    except Exception as exc:

        errors.append(
            f"pickle: {repr(exc)}"
        )

    raise RuntimeError(
        "Unable to load ensemble_model.pkl.\n"
        + "\n".join(errors)
    )


# ============================================================
# 6. MODEL PROVENANCE
# ============================================================

def inspect_model(
    model: Any,
    load_method: str
) -> Dict[str, Any]:

    provenance: Dict[str, Any] = {
        "model_path":
            str(MODEL_PATH),
        "model_file_size_bytes":
            safe_file_size(
                MODEL_PATH
            ),
        "model_modified_time":
            safe_modified_time(
                MODEL_PATH
            ),
        "load_method":
            load_method,
        "python_class":
            model.__class__.__name__,
        "python_module":
            model.__class__.__module__,
        "has_predict":
            int(
                hasattr(
                    model,
                    "predict"
                )
            ),
        "has_predict_proba":
            int(
                hasattr(
                    model,
                    "predict_proba"
                )
            ),
        "has_decision_function":
            int(
                hasattr(
                    model,
                    "decision_function"
                )
            ),
    }

    if hasattr(
        model,
        "classes_"
    ):

        try:

            provenance[
                "classes"
            ] = json.dumps(
                np.asarray(
                    model.classes_
                ).tolist()
            )

        except Exception:
            provenance[
                "classes"
            ] = str(
                model.classes_
            )

    else:
        provenance[
            "classes"
        ] = ""

    if hasattr(
        model,
        "n_features_in_"
    ):

        provenance[
            "n_features_in"
        ] = int(
            model.n_features_in_
        )

    else:
        provenance[
            "n_features_in"
        ] = ""

    if hasattr(
        model,
        "feature_names_in_"
    ):

        try:

            provenance[
                "feature_names_in"
            ] = json.dumps(
                [
                    str(x)
                    for x
                    in model.feature_names_in_
                ],
                ensure_ascii=False
            )

        except Exception:

            provenance[
                "feature_names_in"
            ] = ""

    else:
        provenance[
            "feature_names_in"
        ] = ""

    if hasattr(
        model,
        "get_params"
    ):

        try:

            params = model.get_params(
                deep=False
            )

            provenance[
                "model_parameters"
            ] = json.dumps(
                params,
                default=str,
                ensure_ascii=False
            )

        except Exception:

            provenance[
                "model_parameters"
            ] = ""

    else:
        provenance[
            "model_parameters"
        ] = ""

    return provenance


# ============================================================
# 7. FEATURE ALIGNMENT
# ============================================================

def align_features(
    X: pd.DataFrame,
    model: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    audit: Dict[str, Any] = {
        "input_n_rows":
            len(X),
        "input_n_columns":
            X.shape[1],
        "original_columns":
            json.dumps(
                list(
                    map(str, X.columns)
                ),
                ensure_ascii=False
            ),
        "feature_alignment":
            "unchanged",
        "missing_model_features":
            "",
        "extra_test_features":
            "",
    }

    # --------------------------------------------------------
    # Check explicit number of model features.
    # --------------------------------------------------------

    if hasattr(
        model,
        "n_features_in_"
    ):

        expected_n = int(
            model.n_features_in_
        )

        audit[
            "model_expected_features"
        ] = expected_n

        if X.shape[1] != expected_n:

            raise RuntimeError(
                "Feature-count mismatch.\n"
                f"Model expects {expected_n} features, "
                f"but X_test contains {X.shape[1]} columns.\n"
                "No automatic feature deletion/addition will be performed."
            )

    else:

        audit[
            "model_expected_features"
        ] = ""

    # --------------------------------------------------------
    # Check exact model feature names if available.
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        model_columns = [
            str(x)
            for x
            in model.feature_names_in_
        ]

        test_columns = [
            str(x)
            for x
            in X.columns
        ]

        model_set = set(
            model_columns
        )

        test_set = set(
            test_columns
        )

        missing = [
            col
            for col
            in model_columns
            if col not in test_set
        ]

        extra = [
            col
            for col
            in test_columns
            if col not in model_set
        ]

        audit[
            "missing_model_features"
        ] = "; ".join(
            missing
        )

        audit[
            "extra_test_features"
        ] = "; ".join(
            extra
        )

        if missing or extra:

            raise RuntimeError(
                "Feature-name mismatch between saved model and "
                "X_test_scaled.csv.\n"
                f"Missing model features: {missing}\n"
                f"Extra test features: {extra}"
            )

        if test_columns != model_columns:

            X = X[
                model_columns
            ].copy()

            audit[
                "feature_alignment"
            ] = (
                "reordered_to_model_feature_names_in"
            )

    # --------------------------------------------------------
    # Numeric validation.
    # --------------------------------------------------------

    for col in X.columns:

        converted = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        if converted.isna().any():

            raise RuntimeError(
                f"Feature column '{col}' contains "
                "missing or non-numeric values."
            )

        X[col] = converted

    values = X.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            "X_test contains NaN or infinite values."
        )

    audit[
        "final_n_rows"
    ] = len(X)

    audit[
        "final_n_columns"
    ] = X.shape[1]

    audit[
        "final_columns"
    ] = json.dumps(
        list(
            map(str, X.columns)
        ),
        ensure_ascii=False
    )

    return X, audit


# ============================================================
# 8. PREDICTION GENERATION
# ============================================================

def normalize_binary_predictions(
    predictions: Any
) -> np.ndarray:

    arr = np.asarray(
        predictions
    ).reshape(-1)

    # Numeric conversion if possible.
    try:
        arr = arr.astype(float)
    except Exception:

        raise RuntimeError(
            "Model predict() output is not numeric."
        )

    if not np.isfinite(
        arr
    ).all():

        raise RuntimeError(
            "Model predictions contain NaN/Inf."
        )

    unique = set(
        np.unique(
            arr
        ).tolist()
    )

    if unique.issubset(
        {0.0, 1.0}
    ):
        return arr.astype(int)

    if unique.issubset(
        {-1.0, 1.0}
    ):
        return (
            arr > 0
        ).astype(int)

    raise RuntimeError(
        "predict() did not return recognizable binary labels. "
        f"Unique values: {sorted(unique)}"
    )


def determine_positive_class_index(
    model: Any,
    n_columns: int
) -> int:
    """
    For two-column probabilities, determine which column corresponds
    to positive class label 1.
    """

    if n_columns != 2:

        raise RuntimeError(
            "Expected exactly two probability columns "
            f"for binary classification; found {n_columns}."
        )

    if hasattr(
        model,
        "classes_"
    ):

        classes = np.asarray(
            model.classes_
        )

        positions = np.where(
            classes == 1
        )[0]

        if len(
            positions
        ) == 1:

            return int(
                positions[0]
            )

        # String case.
        positions = np.where(
            classes.astype(str)
            == "1"
        )[0]

        if len(
            positions
        ) == 1:

            return int(
                positions[0]
            )

        raise RuntimeError(
            "Model has classes_, but positive class label 1 "
            f"cannot be located. classes_={classes.tolist()}"
        )

    # Standard sklearn convention is [0,1], but do NOT assume
    # silently if the model does not expose classes_.
    raise RuntimeError(
        "predict_proba returned two columns but model.classes_ "
        "is unavailable. Positive-class column cannot be "
        "identified safely."
    )


def get_probabilities(
    model: Any,
    X: pd.DataFrame
) -> Tuple[
    Optional[np.ndarray],
    str,
    str
]:
    """
    Returns:
        probability_or_score,
        score_type,
        explanation

    Preference:
        1. predict_proba
        2. decision_function
        3. unavailable

    decision_function is accepted for ROC-AUC because AUC can be
    computed from continuous decision scores. It is NOT interpreted
    as probability.
    """

    # --------------------------------------------------------
    # Genuine probability
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            proba = np.asarray(
                model.predict_proba(
                    X
                )
            )

            if proba.ndim == 1:

                values = proba.astype(
                    float
                )

                if np.any(
                    values < 0
                ) or np.any(
                    values > 1
                ):

                    raise RuntimeError(
                        "1D predict_proba output contains "
                        "values outside [0,1]."
                    )

                return (
                    values,
                    "probability",
                    "predict_proba returned one probability column",
                )

            if (
                proba.ndim == 2
                and proba.shape[1] == 2
            ):

                positive_idx = (
                    determine_positive_class_index(
                        model,
                        proba.shape[1],
                    )
                )

                values = proba[
                    :,
                    positive_idx
                ].astype(float)

                return (
                    values,
                    "probability",
                    (
                        "Positive-class probability extracted "
                        f"from predict_proba column {positive_idx}"
                    ),
                )

            raise RuntimeError(
                "Unexpected predict_proba shape: "
                f"{proba.shape}"
            )

        except Exception as exc:

            print(
                "\nWARNING: predict_proba exists but could "
                "not be used:"
            )

            print(
                repr(exc)
            )

    # --------------------------------------------------------
    # Continuous decision score
    # --------------------------------------------------------

    if hasattr(
        model,
        "decision_function"
    ):

        try:

            score = np.asarray(
                model.decision_function(
                    X
                )
            )

            if score.ndim == 1:

                return (
                    score.astype(float),
                    "decision_score",
                    "decision_function used for ROC-AUC only",
                )

            if (
                score.ndim == 2
                and score.shape[1] == 2
            ):

                positive_idx = (
                    determine_positive_class_index(
                        model,
                        score.shape[1],
                    )
                )

                return (
                    score[
                        :,
                        positive_idx
                    ].astype(float),
                    "decision_score",
                    "Positive-class decision score used for ROC-AUC",
                )

        except Exception as exc:

            print(
                "\nWARNING: decision_function exists but "
                "could not be used:"
            )

            print(
                repr(exc)
            )

    return (
        None,
        "unavailable",
        (
            "Neither usable predict_proba nor decision_function "
            "was available."
        ),
    )


# ============================================================
# 9. METRICS
# ============================================================

def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    denominator = (
        tn + fp
    )

    if denominator == 0:
        return float("nan")

    return float(
        tn / denominator
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    score_type: str,
) -> Dict[str, Any]:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    metrics: Dict[str, Any] = {
        "n_samples":
            len(y_true),

        "n_negative":
            int(
                np.sum(
                    y_true == 0
                )
            ),

        "n_positive":
            int(
                np.sum(
                    y_true == 1
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "sensitivity":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "specificity":
            float(
                compute_specificity(
                    y_true,
                    y_pred
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "score_type":
            score_type,

        "roc_auc":
            "",
    }

    if (
        y_score is not None
        and len(
            np.unique(
                y_true
            )
        ) == 2
    ):

        metrics[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                y_score
            )
        )

    return metrics


# ============================================================
# 10. TEST DATA AUDIT
# ============================================================

def build_test_audit(
    X: pd.DataFrame,
    y: np.ndarray,
    label_column: str,
    feature_audit: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "x_test_path":
            str(X_TEST_PATH),

        "y_test_path":
            str(Y_TEST_PATH),

        "x_test_modified_time":
            safe_modified_time(
                X_TEST_PATH
            ),

        "y_test_modified_time":
            safe_modified_time(
                Y_TEST_PATH
            ),

        "x_test_rows":
            len(X),

        "x_test_columns":
            X.shape[1],

        "y_test_rows":
            len(y),

        "label_column":
            label_column,

        "class_0_count":
            int(
                np.sum(
                    y == 0
                )
            ),

        "class_1_count":
            int(
                np.sum(
                    y == 1
                )
            ),

        "class_1_prevalence":
            float(
                np.mean(
                    y == 1
                )
            ),

        "feature_columns":
            json.dumps(
                list(
                    map(
                        str,
                        X.columns
                    )
                ),
                ensure_ascii=False
            ),
    }

    result.update(
        feature_audit
    )

    return result


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:

    print("=" * 88)
    print("HFAGM - REGENERATE EXISTING MODEL PREDICTIONS")
    print("=" * 88)

    print(
        "\nThis script uses the existing saved model only."
    )

    print(
        "No training or hyperparameter modification will occur."
    )

    # --------------------------------------------------------
    # Required-file checks
    # --------------------------------------------------------

    require_file(
        X_TEST_PATH,
        "X_test dataset"
    )

    require_file(
        Y_TEST_PATH,
        "y_test labels"
    )

    require_file(
        MODEL_PATH,
        "saved ensemble model"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_test = load_x_test()

    y_true, label_column = (
        load_y_test()
    )

    if len(
        X_test
    ) != len(
        y_true
    ):

        raise RuntimeError(
            "X_test and y_test row counts differ.\n"
            f"X_test = {len(X_test)}, "
            f"y_test = {len(y_true)}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, load_method = (
        load_existing_model(
            MODEL_PATH
        )
    )

    print(
        "\nLoaded model class:"
    )

    print(
        f"{model.__class__.__module__}."
        f"{model.__class__.__name__}"
    )

    provenance = inspect_model(
        model,
        load_method
    )

    save_single_row_csv(
        MODEL_PROVENANCE_PATH,
        provenance
    )

    # --------------------------------------------------------
    # Align/check features
    # --------------------------------------------------------

    X_aligned, feature_audit = (
        align_features(
            X_test.copy(),
            model
        )
    )

    test_audit = build_test_audit(
        X=X_aligned,
        y=y_true,
        label_column=label_column,
        feature_audit=feature_audit,
    )

    save_single_row_csv(
        TEST_AUDIT_PATH,
        test_audit
    )

    # --------------------------------------------------------
    # Generate hard predictions
    # --------------------------------------------------------

    if not hasattr(
        model,
        "predict"
    ):

        raise RuntimeError(
            "The saved model does not expose predict()."
        )

    print(
        "\nGenerating hard predictions..."
    )

    raw_pred = model.predict(
        X_aligned
    )

    y_pred = (
        normalize_binary_predictions(
            raw_pred
        )
    )

    if len(
        y_pred
    ) != len(
        y_true
    ):

        raise RuntimeError(
            "Generated prediction count differs from y_test."
        )

    # --------------------------------------------------------
    # Generate genuine probability / decision score
    # --------------------------------------------------------

    print(
        "Attempting to recover probability/decision scores..."
    )

    y_score, score_type, score_note = (
        get_probabilities(
            model,
            X_aligned
        )
    )

    if y_score is not None:

        if len(
            y_score
        ) != len(
            y_true
        ):

            raise RuntimeError(
                "Probability/score length differs from y_test."
            )

        if not np.isfinite(
            y_score
        ).all():

            raise RuntimeError(
                "Probability/decision score contains NaN/Inf."
            )

    # --------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        score_type=score_type,
    )

    metrics.update({
        "evaluation_type":
            "Existing model on preserved real test set",

        "model_file":
            str(
                MODEL_PATH
            ),

        "x_test_file":
            str(
                X_TEST_PATH
            ),

        "y_test_file":
            str(
                Y_TEST_PATH
            ),

        "model_class":
            model.__class__.__name__,

        "score_note":
            score_note,

        "prediction_generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "retraining_performed":
            False,

        "threshold_tuning_performed":
            False,
    })

    save_single_row_csv(
        METRICS_PATH,
        metrics
    )

    # --------------------------------------------------------
    # Canonical row-level prediction table
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "test_row_id":
            np.arange(
                len(
                    y_true
                )
            ),

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "correct":
            (
                y_true == y_pred
            ).astype(int),
    })

    if y_score is not None:

        if score_type == "probability":

            prediction_df[
                "y_prob"
            ] = y_score

        elif score_type == "decision_score":

            prediction_df[
                "y_score"
            ] = y_score

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        [
            {
                "actual_class":
                    0,

                "predicted_0":
                    metrics["TN"],

                "predicted_1":
                    metrics["FP"],
            },
            {
                "actual_class":
                    1,

                "predicted_0":
                    metrics["FN"],

                "predicted_1":
                    metrics["TP"],
            }
        ]
    )

    confusion_df.to_csv(
        CONFUSION_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    summary_lines = [
        "=" * 88,
        "HFAGM - EXISTING MODEL PREDICTION REGENERATION",
        "=" * 88,
        "",
        f"Generated: "
        f"{datetime.now().isoformat(timespec='seconds')}",
        "",
        "EVALUATION PROVENANCE",
        "-" * 88,
        f"Model: {MODEL_PATH}",
        f"Model class: "
        f"{model.__class__.__module__}."
        f"{model.__class__.__name__}",
        f"Model load method: "
        f"{load_method}",
        "",
        f"X_test: {X_TEST_PATH}",
        f"y_test: {Y_TEST_PATH}",
        "",
        f"Test samples: "
        f"{metrics['n_samples']}",
        f"Negative class samples: "
        f"{metrics['n_negative']}",
        f"Positive class samples: "
        f"{metrics['n_positive']}",
        "",
        "PRIMARY METRICS",
        "-" * 88,
        f"Accuracy: "
        f"{metrics['accuracy']:.10f}",
        f"Precision: "
        f"{metrics['precision']:.10f}",
        f"Recall / Sensitivity: "
        f"{metrics['recall']:.10f}",
        f"Specificity: "
        f"{metrics['specificity']:.10f}",
        f"F1-score: "
        f"{metrics['f1']:.10f}",
    ]

    if metrics[
        "roc_auc"
    ] != "":

        summary_lines.append(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.10f}"
        )

        summary_lines.append(
            f"ROC-AUC source: "
            f"{score_type}"
        )

    else:

        summary_lines.append(
            "ROC-AUC: NOT AVAILABLE"
        )

        summary_lines.append(
            "Reason: no genuine continuous probability/"
            "decision score was available."
        )

    summary_lines.extend([
        "",
        "CONFUSION MATRIX",
        "-" * 88,
        f"TN: {metrics['TN']}",
        f"FP: {metrics['FP']}",
        f"FN: {metrics['FN']}",
        f"TP: {metrics['TP']}",
        "",
        "PREDICTION SCORE",
        "-" * 88,
        f"Score type: "
        f"{score_type}",
        f"Details: "
        f"{score_note}",
        "",
        "INTEGRITY CONDITIONS",
        "-" * 88,
        "Existing saved model used: YES",
        "Existing preserved test set used: YES",
        "Model retrained: NO",
        "Hyperparameters changed: NO",
        "Threshold tuned: NO",
        "Predictions fabricated: NO",
        "ROC-AUC calculated from hard labels: NO",
        "",
        "OUTPUTS",
        "-" * 88,
        f"Predictions:",
        str(PREDICTIONS_PATH),
        "",
        f"Metrics:",
        str(METRICS_PATH),
        "",
        f"Confusion matrix:",
        str(CONFUSION_PATH),
        "",
        f"Model provenance:",
        str(MODEL_PROVENANCE_PATH),
        "",
        f"Test-data audit:",
        str(TEST_AUDIT_PATH),
        "",
        "=" * 88,
    ])

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print("\n" + "=" * 88)
    print("REGENERATION COMPLETE")
    print("=" * 88)

    print(
        f"\nSamples evaluated: "
        f"{metrics['n_samples']}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.6f}"
    )

    if metrics[
        "roc_auc"
    ] != "":

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            f"AUC source: "
            f"{score_type}"
        )

    else:

        print(
            "ROC-AUC unavailable "
            "(no genuine score/probability)."
        )

    print(
        "\nConfusion matrix:"
    )

    print(
        f"TN={metrics['TN']} "
        f"FP={metrics['FP']} "
        f"FN={metrics['FN']} "
        f"TP={metrics['TP']}"
    )

    print(
        "\nResults written to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nUpload these files next:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        PREDICTIONS_PATH
    )

    print(
        METRICS_PATH
    )

    print(
        MODEL_PROVENANCE_PATH
    )

    print(
        TEST_AUDIT_PATH
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n" + "=" * 88
        )

        print(
            "SCRIPT FAILED SAFELY"
        )

        print(
            "=" * 88
        )

        print(
            f"\n{type(exc).__name__}: "
            f"{exc}"
        )

        print(
            "\nNo model retraining or source-file "
            "modification was performed."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(1)