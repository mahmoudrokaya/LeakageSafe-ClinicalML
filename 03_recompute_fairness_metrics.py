"""
03_recompute_fairness_metrics.py
================================

HFAGM - Leakage-safe fairness reconstruction and recomputation.

PURPOSE
-------
The historical manuscript contains fairness claims involving:
    - Statistical Parity Difference (SPD)
    - Equal Opportunity Difference (EOD)
    - Disparate Impact (DI)

Earlier forensic auditing found that the historical evaluation split was
contaminated by preprocessing/balancing before the train/test split.

Therefore, historical fairness metrics must NOT be treated as validated merely
because they appear in old CSVs or figures.

This script recomputes fairness metrics using the corrected repeated
leakage-safe row-level predictions produced by:

    02E_repeated_leakage_safe_evaluation.py

The script directly addresses Reviewer #3 concerns C16-C18:

C16:
    Recalculate SPD and EOD.
    Enforce their mathematical definitions and expected bounds.

C17:
    Recalculate DI and avoid describing DI farther from 1.0 as improvement.

C18:
    Explicitly identify:
        - sensitive attribute,
        - observed subgroup values,
        - operational reference group,
        - favorable outcome,
        - subgroup sample sizes.

IMPORTANT DESIGN RULES
----------------------
1. No historical fairness number is copied into the new result.
2. No sensitive-group meaning is invented.
3. Numeric/string group codes are preserved as observed.
4. Favorable outcome is explicitly configured.
5. Reference groups are either manually configured or selected reproducibly
   as the largest observed subgroup.
6. Largest-group selection is operational only; it is NOT interpreted as
   privileged, advantaged, ethically preferred, or normatively superior.
7. Fairness is recomputed separately for every repeated seed.
8. The same subject may occur in the test partition for multiple seeds.
   Therefore repeated-seed summaries are stability summaries, not independent
   population samples.
9. Pairwise subgroup comparisons are also reported so the analysis does not
   depend solely on one arbitrary reference group.
10. DI is undefined when the reference-group favorable prediction rate is 0.
    Such cases remain NaN rather than being forced to a value.
11. EOD uses true favorable-outcome cases as its denominator.
12. SPD and EOD are signed differences:
        comparison group - reference group
    and therefore lie in [-1, 1].
13. DI is a nonnegative ratio:
        comparison favorable-rate / reference favorable-rate.
14. Values are never clipped merely to make them look valid.

EXPECTED 02E INPUT SCHEMA
-------------------------
The script supports the actual 02E repeated_predictions.csv columns:

    seed
    variant
    test_partition_row
    original_source_row_index
    y_true
    y_pred
    correct
    score_source
    y_score_positive

The following aliases are also supported for robustness.

EXPECTED OUTPUT
---------------
outputs/revision_fairness/recomputed_fairness/

Main files:
    sensitive_attribute_inventory.csv
    prediction_input_audit.csv
    predictions_with_sensitive_attributes.csv
    fairness_per_seed_reference.csv
    fairness_per_seed_pairwise.csv
    fairness_summary_reference.csv
    fairness_summary_pairwise.csv
    subgroup_performance_per_seed.csv
    subgroup_counts_per_seed.csv
    sensitive_attribute_missingness.csv
    fairness_bounds_audit.csv
    fairness_interpretation_audit.csv
    fairness_provenance.csv
    fairness_summary.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import traceback

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

REPEATED_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
    / "repeated_leakage_safe_evaluation"
    / "repeated_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_fairness"
    / "recomputed_fairness"
)

PRIMARY_VARIANT = "unbalanced_training"

# Existing project target semantics:
#   0 = recovered / survived
#   1 = deceased
#
# Fairness favorable outcome:
#   recovery/survival
FAVORABLE_LABEL = 0
FAVORABLE_OUTCOME_DESCRIPTION = "recovered/survived"

UNFAVORABLE_LABEL = 1
UNFAVORABLE_OUTCOME_DESCRIPTION = "deceased"

SENSITIVE_ATTRIBUTE_CANDIDATES = [
    "Gender",
    "gender",
    "Sex",
    "sex",
    "Nationality",
    "nationality",
]

# Optional manual reference-group overrides.
#
# Leave None unless the study documentation explicitly justifies a particular
# reference/privileged group.
REFERENCE_GROUP_OVERRIDES: Dict[str, Optional[Any]] = {
    "Gender": None,
    "Sex": None,
    "Nationality": None,
}

MIN_RECOMMENDED_SUBGROUP_N = 10
BOUND_TOLERANCE = 1e-10


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def safe_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


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


def read_csv_robust(
    path: Path,
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
                f"{encoding}: {repr(exc)}"
            )

    raise RuntimeError(
        f"Could not read CSV:\n{path}\n"
        + "\n".join(
            errors
        )
    )


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

        pd.DataFrame(
            columns=columns or []
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


def safe_float(
    value: Any,
) -> float:

    try:

        result = float(
            value
        )

        if np.isfinite(
            result
        ):
            return result

    except Exception:
        pass

    return np.nan


def format_metric(
    value: Any,
) -> str:

    value = safe_float(
        value
    )

    if np.isnan(
        value
    ):
        return "NA"

    return f"{value:.6f}"


# =============================================================================
# 3. LOAD RAW DATA
# =============================================================================

def load_raw_dataset(
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

        details = "\n".join(
            f"{path}: {len(df)} rows"
            for df, path, _
            in candidates
        )

        raise RuntimeError(
            "Raw sources were readable, but none had the expected "
            "193 observations.\n"
            + details
        )

    raise FileNotFoundError(
        "No usable 193-row raw clinical dataset found."
    )


# =============================================================================
# 4. SENSITIVE ATTRIBUTE DISCOVERY
# =============================================================================

def discover_sensitive_attributes(
    raw_df: pd.DataFrame,
) -> List[str]:

    normalized_columns = {
        normalize_name(
            col
        ):
            col

        for col in raw_df.columns
    }

    found = []

    for candidate in SENSITIVE_ATTRIBUTE_CANDIDATES:

        normalized_candidate = normalize_name(
            candidate
        )

        if normalized_candidate in normalized_columns:

            actual_col = normalized_columns[
                normalized_candidate
            ]

            if actual_col not in found:

                found.append(
                    actual_col
                )

    return found


def normalize_group_value(
    value: Any,
) -> Any:

    if pd.isna(
        value
    ):
        return np.nan

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):

        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):

        if float(
            value
        ).is_integer():

            return int(
                value
            )

        return float(
            value
        )

    text = str(
        value
    ).strip()

    if text == "":
        return np.nan

    return text


def observed_group_counts(
    series: pd.Series,
) -> Dict[Any, int]:

    normalized = series.map(
        normalize_group_value
    )

    normalized = normalized[
        normalized.notna()
    ]

    return normalized.value_counts(
        dropna=False
    ).to_dict()


def resolve_reference_group(
    attribute: str,
    counts: Dict[Any, int],
) -> Tuple[Any, str]:

    if not counts:

        raise RuntimeError(
            f"No nonmissing groups available for {attribute}."
        )

    normalized_override_map = {
        normalize_name(
            key
        ):
            value

        for key, value
        in REFERENCE_GROUP_OVERRIDES.items()
    }

    override = normalized_override_map.get(
        normalize_name(
            attribute
        ),
        None,
    )

    if override is not None:

        available = list(
            counts.keys()
        )

        if override not in available:

            raise RuntimeError(
                f"Configured reference group {override!r} for "
                f"{attribute!r} does not occur in the raw dataset.\n"
                f"Available groups: {available}"
            )

        return (
            override,
            "MANUAL_OVERRIDE",
        )

    ordered = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            str(
                item[0]
            ),
        ),
    )

    return (
        ordered[0][0],
        "AUTO_LARGEST_GROUP_NOT_NORMATIVE",
    )


# =============================================================================
# 5. PREDICTION INPUT AUDIT
# =============================================================================

def find_column(
    df: pd.DataFrame,
    candidate_names: List[str],
) -> Optional[str]:

    normalized = {
        normalize_name(
            col
        ):
            col

        for col in df.columns
    }

    for candidate in candidate_names:

        key = normalize_name(
            candidate
        )

        if key in normalized:

            return normalized[
                key
            ]

    return None


def audit_and_standardize_predictions(
    pred_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:

    audit = []

    seed_col = find_column(
        pred_df,
        [
            "seed",
            "random_seed",
        ],
    )

    variant_col = find_column(
        pred_df,
        [
            "variant",
            "training_variant",
            "model_variant",
        ],
    )

    # -----------------------------------------------------------------
    # UPDATED FOR ACTUAL 02E SCHEMA
    #
    # The actual 02E output uses:
    #     original_source_row_index
    #
    # This must be preferred over test_partition_row because
    # test_partition_row is only the local position within each test set.
    # -----------------------------------------------------------------

    source_index_col = find_column(
        pred_df,
        [
            "original_source_row_index",
            "source_index",
            "source_row_index",
            "original_index",
            "original_row_index",
            "row_index",
            "sample_index",
            "source_row",
            "test_partition_row",
        ],
    )

    y_true_col = find_column(
        pred_df,
        [
            "y_true",
            "true_label",
            "actual",
            "target",
        ],
    )

    y_pred_col = find_column(
        pred_df,
        [
            "y_pred",
            "predicted_label",
            "prediction",
            "pred",
        ],
    )

    # -----------------------------------------------------------------
    # UPDATED FOR ACTUAL 02E SCHEMA
    #
    # Actual 02E output uses:
    #     y_score_positive
    # -----------------------------------------------------------------

    y_prob_col = find_column(
        pred_df,
        [
            "y_score_positive",
            "y_prob",
            "y_score",
            "probability",
            "positive_probability",
            "positive_class_score",
            "score",
        ],
    )

    test_partition_row_col = find_column(
        pred_df,
        [
            "test_partition_row",
            "test_row",
            "partition_row",
        ],
    )

    score_source_col = find_column(
        pred_df,
        [
            "score_source",
            "probability_source",
        ],
    )

    correct_col = find_column(
        pred_df,
        [
            "correct",
            "is_correct",
        ],
    )

    required = {
        "seed":
            seed_col,

        "variant":
            variant_col,

        "source_index":
            source_index_col,

        "y_true":
            y_true_col,

        "y_pred":
            y_pred_col,
    }

    missing = [
        key
        for key, value
        in required.items()
        if value is None
    ]

    if missing:

        raise RuntimeError(
            "Repeated prediction file is missing required columns: "
            + ", ".join(
                missing
            )
            + "\nObserved columns:\n"
            + safe_json(
                pred_df.columns.tolist()
            )
        )

    standardized = pd.DataFrame(
        {
            "seed":
                pd.to_numeric(
                    pred_df[
                        seed_col
                    ],
                    errors="raise",
                ).astype(
                    int
                ),

            "variant":
                pred_df[
                    variant_col
                ].astype(
                    str
                ),

            "source_index":
                pd.to_numeric(
                    pred_df[
                        source_index_col
                    ],
                    errors="raise",
                ).astype(
                    int
                ),

            "y_true":
                pd.to_numeric(
                    pred_df[
                        y_true_col
                    ],
                    errors="raise",
                ).astype(
                    int
                ),

            "y_pred":
                pd.to_numeric(
                    pred_df[
                        y_pred_col
                    ],
                    errors="raise",
                ).astype(
                    int
                ),
        }
    )

    if y_prob_col is not None:

        standardized[
            "y_prob"
        ] = pd.to_numeric(
            pred_df[
                y_prob_col
            ],
            errors="coerce",
        )

    else:

        standardized[
            "y_prob"
        ] = np.nan

    if test_partition_row_col is not None:

        standardized[
            "test_partition_row"
        ] = pd.to_numeric(
            pred_df[
                test_partition_row_col
            ],
            errors="coerce",
        )

    else:

        standardized[
            "test_partition_row"
        ] = np.nan

    if score_source_col is not None:

        standardized[
            "score_source"
        ] = pred_df[
            score_source_col
        ].astype(
            str
        )

    else:

        standardized[
            "score_source"
        ] = ""

    if correct_col is not None:

        standardized[
            "original_correct_field"
        ] = pred_df[
            correct_col
        ]

    else:

        standardized[
            "original_correct_field"
        ] = np.nan

    # -----------------------------------------------------------------
    # Verify that source_index is genuinely the original raw row index.
    #
    # Because original_source_row_index is available in 02E, it should have
    # been chosen before test_partition_row.
    # -----------------------------------------------------------------

    if source_index_col == test_partition_row_col:

        raise RuntimeError(
            "The script resolved test_partition_row as the source index. "
            "This is unsafe because test_partition_row is not the original "
            "193-row dataset index. The 02E file should contain "
            "original_source_row_index."
        )

    audit.append(
        {
            "check":
                "resolved_source_index_column",

            "value":
                source_index_col,

            "expected_preferred":
                "original_source_row_index",

            "status":
                (
                    "PASS"
                    if normalize_name(
                        source_index_col
                    )
                    == normalize_name(
                        "original_source_row_index"
                    )
                    else "REVIEW"
                ),
        }
    )

    audit.append(
        {
            "check":
                "resolved_probability_column",

            "value":
                y_prob_col,

            "expected_preferred":
                "y_score_positive",

            "status":
                (
                    "PASS"
                    if y_prob_col is not None
                    else "REVIEW"
                ),
        }
    )

    # -----------------------------------------------------------------
    # Validate source-index range.
    # -----------------------------------------------------------------

    min_index = int(
        standardized[
            "source_index"
        ].min()
    )

    max_index = int(
        standardized[
            "source_index"
        ].max()
    )

    if min_index < 0:

        raise RuntimeError(
            "Prediction source index contains negative values."
        )

    if max_index >= len(
        raw_df
    ):

        raise RuntimeError(
            f"Prediction source index reaches {max_index}, "
            f"but raw dataset contains only {len(raw_df)} rows."
        )

    # -----------------------------------------------------------------
    # Binary-label audit.
    # -----------------------------------------------------------------

    true_labels = sorted(
        standardized[
            "y_true"
        ].unique().tolist()
    )

    pred_labels = sorted(
        standardized[
            "y_pred"
        ].unique().tolist()
    )

    if not set(
        true_labels
    ).issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"y_true is not binary 0/1: {true_labels}"
        )

    if not set(
        pred_labels
    ).issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"y_pred is not binary 0/1: {pred_labels}"
        )

    # -----------------------------------------------------------------
    # Verify "correct" field against y_true/y_pred if present.
    # -----------------------------------------------------------------

    standardized[
        "correct_recomputed"
    ] = (
        standardized[
            "y_true"
        ]
        ==
        standardized[
            "y_pred"
        ]
    ).astype(
        int
    )

    correct_mismatch_count = 0

    if correct_col is not None:

        original_correct = pd.to_numeric(
            pred_df[
                correct_col
            ],
            errors="coerce",
        )

        valid_correct = original_correct.notna()

        if valid_correct.any():

            mismatch = (
                original_correct[
                    valid_correct
                ].astype(
                    int
                ).to_numpy()
                !=
                standardized.loc[
                    valid_correct,
                    "correct_recomputed"
                ].to_numpy(
                    dtype=int
                )
            )

            correct_mismatch_count = int(
                np.sum(
                    mismatch
                )
            )

    audit.append(
        {
            "check":
                "correct_field_vs_recomputed",

            "value":
                correct_mismatch_count,

            "expected":
                0,

            "status":
                (
                    "PASS"
                    if correct_mismatch_count == 0
                    else "REVIEW"
                ),
        }
    )

    # -----------------------------------------------------------------
    # Verify each seed/variant/source row is unique.
    # -----------------------------------------------------------------

    duplicates = standardized.duplicated(
        subset=[
            "seed",
            "variant",
            "source_index",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicates.sum()
    )

    if duplicate_count > 0:

        raise RuntimeError(
            "Repeated predictions contain duplicate original source rows "
            "within the same seed/variant. "
            f"Duplicated rows: {duplicate_count}"
        )

    # -----------------------------------------------------------------
    # Check each seed/variant test size.
    #
    # 193 * 0.20 -> 39 rows using sklearn split behavior in 02E.
    # -----------------------------------------------------------------

    group_sizes = (
        standardized
        .groupby(
            [
                "seed",
                "variant",
            ]
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    for _, row in group_sizes.iterrows():

        audit.append(
            {
                "check":
                    "seed_variant_test_size",

                "seed":
                    int(
                        row[
                            "seed"
                        ]
                    ),

                "variant":
                    row[
                        "variant"
                    ],

                "value":
                    int(
                        row[
                            "n"
                        ]
                    ),

                "expected":
                    39,

                "status":
                    (
                        "PASS"
                        if int(
                            row[
                                "n"
                            ]
                        ) == 39
                        else "REVIEW"
                    ),
            }
        )

    # -----------------------------------------------------------------
    # Probability-score audit.
    #
    # y_score_positive refers to probability/score of class 1 (deceased).
    # That is acceptable for subgroup AUC reporting.
    # Fairness SPD/EOD/DI themselves are computed from y_pred and favorable
    # label 0, not from probabilities.
    # -----------------------------------------------------------------

    finite_probabilities = int(
        np.isfinite(
            standardized[
                "y_prob"
            ].to_numpy(
                dtype=float
            )
        ).sum()
    )

    probability_outside_01 = 0

    if finite_probabilities > 0:

        probs = standardized[
            "y_prob"
        ].to_numpy(
            dtype=float
        )

        finite = np.isfinite(
            probs
        )

        probability_outside_01 = int(
            np.sum(
                (
                    probs[
                        finite
                    ] < 0
                )
                |
                (
                    probs[
                        finite
                    ] > 1
                )
            )
        )

    audit.extend(
        [
            {
                "check":
                    "rows",

                "value":
                    len(
                        standardized
                    ),

                "status":
                    "INFO",
            },

            {
                "check":
                    "unique_seeds",

                "value":
                    standardized[
                        "seed"
                    ].nunique(),

                "status":
                    "INFO",
            },

            {
                "check":
                    "variants",

                "value":
                    safe_json(
                        sorted(
                            standardized[
                                "variant"
                            ].unique().tolist()
                        )
                    ),

                "status":
                    "INFO",
            },

            {
                "check":
                    "source_index_min",

                "value":
                    min_index,

                "status":
                    "PASS",
            },

            {
                "check":
                    "source_index_max",

                "value":
                    max_index,

                "status":
                    "PASS",
            },

            {
                "check":
                    "duplicate_seed_variant_source_rows",

                "value":
                    duplicate_count,

                "status":
                    (
                        "PASS"
                        if duplicate_count == 0
                        else "FAIL"
                    ),
            },

            {
                "check":
                    "probability_column_available",

                "value":
                    int(
                        y_prob_col is not None
                    ),

                "resolved_column":
                    y_prob_col,

                "status":
                    (
                        "PASS"
                        if y_prob_col is not None
                        else "REVIEW"
                    ),
            },

            {
                "check":
                    "finite_probability_scores",

                "value":
                    finite_probabilities,

                "status":
                    "INFO",
            },

            {
                "check":
                    "probability_scores_outside_0_1",

                "value":
                    probability_outside_01,

                "status":
                    (
                        "PASS"
                        if probability_outside_01 == 0
                        else "REVIEW"
                    ),
            },
        ]
    )

    return (
        standardized,
        audit,
    )


# =============================================================================
# 6. FAIRNESS DEFINITIONS
# =============================================================================

def favorable_indicator(
    labels: np.ndarray,
) -> np.ndarray:

    return (
        labels
        == FAVORABLE_LABEL
    ).astype(
        int
    )


def safe_rate(
    numerator: int,
    denominator: int,
) -> float:

    if denominator <= 0:
        return np.nan

    return float(
        numerator
        / denominator
    )


def subgroup_fairness_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:

    y_true_fav = favorable_indicator(
        y_true
    )

    y_pred_fav = favorable_indicator(
        y_pred
    )

    n = len(
        y_true
    )

    n_true_favorable = int(
        np.sum(
            y_true_fav == 1
        )
    )

    n_true_unfavorable = int(
        np.sum(
            y_true_fav == 0
        )
    )

    n_pred_favorable = int(
        np.sum(
            y_pred_fav == 1
        )
    )

    n_pred_unfavorable = int(
        np.sum(
            y_pred_fav == 0
        )
    )

    favorable_prediction_rate = safe_rate(
        n_pred_favorable,
        n,
    )

    true_favorable_correct = int(
        np.sum(
            (
                y_true_fav == 1
            )
            &
            (
                y_pred_fav == 1
            )
        )
    )

    favorable_tpr = safe_rate(
        true_favorable_correct,
        n_true_favorable,
    )

    false_favorable_count = int(
        np.sum(
            (
                y_true_fav == 0
            )
            &
            (
                y_pred_fav == 1
            )
        )
    )

    false_favorable_rate = safe_rate(
        false_favorable_count,
        n_true_unfavorable,
    )

    return {
        "n":
            n,

        "n_true_favorable":
            n_true_favorable,

        "n_true_unfavorable":
            n_true_unfavorable,

        "n_pred_favorable":
            n_pred_favorable,

        "n_pred_unfavorable":
            n_pred_unfavorable,

        "favorable_prediction_rate":
            favorable_prediction_rate,

        "favorable_true_positive_rate":
            favorable_tpr,

        "false_favorable_rate":
            false_favorable_rate,
    }


def fairness_difference(
    comparison_rate: float,
    reference_rate: float,
) -> float:

    if (
        np.isnan(
            comparison_rate
        )
        or
        np.isnan(
            reference_rate
        )
    ):
        return np.nan

    return float(
        comparison_rate
        - reference_rate
    )


def disparate_impact(
    comparison_rate: float,
    reference_rate: float,
) -> float:

    if (
        np.isnan(
            comparison_rate
        )
        or
        np.isnan(
            reference_rate
        )
    ):
        return np.nan

    if reference_rate == 0:
        return np.nan

    return float(
        comparison_rate
        / reference_rate
    )


# =============================================================================
# 7. SUBGROUP CLASSIFICATION PERFORMANCE
# =============================================================================

def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:

    result = {
        "n":
            len(
                y_true
            ),

        "accuracy":
            np.nan,

        "precision_label1":
            np.nan,

        "recall_label1":
            np.nan,

        "f1_label1":
            np.nan,

        "specificity_label0":
            np.nan,

        "roc_auc_label1":
            np.nan,

        "tn":
            np.nan,

        "fp":
            np.nan,

        "fn":
            np.nan,

        "tp":
            np.nan,
    }

    if len(
        y_true
    ) == 0:

        return result

    result[
        "accuracy"
    ] = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    result[
        "precision_label1"
    ] = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    result[
        "recall_label1"
    ] = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    result[
        "f1_label1"
    ] = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = cm.ravel()

    result.update(
        {
            "tn":
                int(
                    tn
                ),

            "fp":
                int(
                    fp
                ),

            "fn":
                int(
                    fn
                ),

            "tp":
                int(
                    tp
                ),

            "specificity_label0":
                safe_rate(
                    int(
                        tn
                    ),
                    int(
                        tn + fp
                    ),
                ),
        }
    )

    if (
        y_prob is not None
        and
        len(
            np.unique(
                y_true
            )
        ) == 2
        and
        np.all(
            np.isfinite(
                y_prob
            )
        )
    ):

        try:

            result[
                "roc_auc_label1"
            ] = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                )
            )

        except Exception:
            pass

    return result


# =============================================================================
# 8. BUILD SENSITIVE-ATTRIBUTE INVENTORY
# =============================================================================

def build_sensitive_attribute_inventory(
    raw_df: pd.DataFrame,
    sensitive_attributes: List[str],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:

    rows = []
    references = {}

    for attribute in sensitive_attributes:

        normalized_series = raw_df[
            attribute
        ].map(
            normalize_group_value
        )

        counts = observed_group_counts(
            normalized_series
        )

        (
            reference_group,
            reference_method,
        ) = resolve_reference_group(
            attribute,
            counts,
        )

        references[
            attribute
        ] = {
            "reference_group":
                reference_group,

            "reference_method":
                reference_method,
        }

        missing_count = int(
            normalized_series.isna().sum()
        )

        for group_value, n in sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                str(
                    item[0]
                ),
            ),
        ):

            rows.append(
                {
                    "sensitive_attribute":
                        attribute,

                    "observed_group":
                        group_value,

                    "n_raw_dataset":
                        int(
                            n
                        ),

                    "raw_dataset_fraction":
                        float(
                            n
                            / len(
                                raw_df
                            )
                        ),

                    "reference_group":
                        reference_group,

                    "is_reference_group":
                        int(
                            group_value
                            == reference_group
                        ),

                    "reference_selection_method":
                        reference_method,

                    "normative_privileged_status_claimed":
                        False,

                    "favorable_label":
                        FAVORABLE_LABEL,

                    "favorable_outcome_description":
                        FAVORABLE_OUTCOME_DESCRIPTION,

                    "missing_sensitive_attribute_rows":
                        missing_count,
                }
            )

    return (
        rows,
        references,
    )


# =============================================================================
# 9. ATTACH SENSITIVE ATTRIBUTES TO 02E PREDICTIONS
# =============================================================================

def attach_sensitive_attributes(
    pred_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    sensitive_attributes: List[str],
) -> pd.DataFrame:

    result = pred_df.copy()

    for attribute in sensitive_attributes:

        values = []

        for source_index in result[
            "source_index"
        ].tolist():

            raw_value = raw_df.iloc[
                int(
                    source_index
                )
            ][
                attribute
            ]

            values.append(
                normalize_group_value(
                    raw_value
                )
            )

        result[
            attribute
        ] = values

    return result


# =============================================================================
# 10. PER-SEED REFERENCE-GROUP FAIRNESS
# =============================================================================

def compute_reference_fairness(
    pred_df: pd.DataFrame,
    sensitive_attributes: List[str],
    references: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    fairness_rows = []
    subgroup_performance_rows = []
    subgroup_count_rows = []

    grouped = pred_df.groupby(
        [
            "seed",
            "variant",
        ],
        sort=True,
    )

    for (
        seed,
        variant,
    ), run_df in grouped:

        for attribute in sensitive_attributes:

            reference_group = references[
                attribute
            ][
                "reference_group"
            ]

            reference_method = references[
                attribute
            ][
                "reference_method"
            ]

            valid_df = run_df[
                run_df[
                    attribute
                ].notna()
            ].copy()

            observed_groups = sorted(
                valid_df[
                    attribute
                ].unique().tolist(),
                key=lambda value:
                    str(
                        value
                    ),
            )

            if reference_group not in observed_groups:

                for comparison_group in observed_groups:

                    fairness_rows.append(
                        {
                            "seed":
                                int(
                                    seed
                                ),

                            "variant":
                                variant,

                            "sensitive_attribute":
                                attribute,

                            "reference_group":
                                reference_group,

                            "comparison_group":
                                comparison_group,

                            "reference_selection_method":
                                reference_method,

                            "status":
                                "REFERENCE_GROUP_ABSENT_IN_THIS_TEST_SPLIT",

                            "spd":
                                np.nan,

                            "eod":
                                np.nan,

                            "di":
                                np.nan,
                        }
                    )

                continue

            group_components = {}

            for group_value in observed_groups:

                group_df = valid_df[
                    valid_df[
                        attribute
                    ]
                    ==
                    group_value
                ]

                y_true = group_df[
                    "y_true"
                ].to_numpy(
                    dtype=int
                )

                y_pred = group_df[
                    "y_pred"
                ].to_numpy(
                    dtype=int
                )

                if group_df[
                    "y_prob"
                ].notna().all():

                    y_prob = group_df[
                        "y_prob"
                    ].to_numpy(
                        dtype=float
                    )

                else:

                    y_prob = None

                components = subgroup_fairness_components(
                    y_true,
                    y_pred,
                )

                group_components[
                    group_value
                ] = components

                perf = subgroup_performance(
                    y_true,
                    y_pred,
                    y_prob,
                )

                subgroup_count_rows.append(
                    {
                        "seed":
                            int(
                                seed
                            ),

                        "variant":
                            variant,

                        "sensitive_attribute":
                            attribute,

                        "group":
                            group_value,

                        "is_reference_group":
                            int(
                                group_value
                                == reference_group
                            ),

                        "n":
                            components[
                                "n"
                            ],

                        "n_true_favorable":
                            components[
                                "n_true_favorable"
                            ],

                        "n_true_unfavorable":
                            components[
                                "n_true_unfavorable"
                            ],

                        "n_pred_favorable":
                            components[
                                "n_pred_favorable"
                            ],

                        "n_pred_unfavorable":
                            components[
                                "n_pred_unfavorable"
                            ],

                        "favorable_prediction_rate":
                            components[
                                "favorable_prediction_rate"
                            ],

                        "favorable_true_positive_rate":
                            components[
                                "favorable_true_positive_rate"
                            ],

                        "false_favorable_rate":
                            components[
                                "false_favorable_rate"
                            ],

                        "subgroup_size_flag":
                            (
                                "SMALL_SUBGROUP"
                                if components[
                                    "n"
                                ]
                                < MIN_RECOMMENDED_SUBGROUP_N
                                else
                                "ADEQUATE_FOR_DESCRIPTIVE_REPORTING"
                            ),
                    }
                )

                subgroup_performance_rows.append(
                    {
                        "seed":
                            int(
                                seed
                            ),

                        "variant":
                            variant,

                        "sensitive_attribute":
                            attribute,

                        "group":
                            group_value,

                        "is_reference_group":
                            int(
                                group_value
                                == reference_group
                            ),

                        **perf,
                    }
                )

            ref = group_components[
                reference_group
            ]

            for comparison_group in observed_groups:

                if comparison_group == reference_group:
                    continue

                comp = group_components[
                    comparison_group
                ]

                spd = fairness_difference(
                    comp[
                        "favorable_prediction_rate"
                    ],
                    ref[
                        "favorable_prediction_rate"
                    ],
                )

                eod = fairness_difference(
                    comp[
                        "favorable_true_positive_rate"
                    ],
                    ref[
                        "favorable_true_positive_rate"
                    ],
                )

                di = disparate_impact(
                    comp[
                        "favorable_prediction_rate"
                    ],
                    ref[
                        "favorable_prediction_rate"
                    ],
                )

                fairness_rows.append(
                    {
                        "seed":
                            int(
                                seed
                            ),

                        "variant":
                            variant,

                        "sensitive_attribute":
                            attribute,

                        "reference_group":
                            reference_group,

                        "comparison_group":
                            comparison_group,

                        "reference_selection_method":
                            reference_method,

                        "favorable_label":
                            FAVORABLE_LABEL,

                        "favorable_outcome":
                            FAVORABLE_OUTCOME_DESCRIPTION,

                        "n_reference":
                            ref[
                                "n"
                            ],

                        "n_comparison":
                            comp[
                                "n"
                            ],

                        "reference_favorable_prediction_rate":
                            ref[
                                "favorable_prediction_rate"
                            ],

                        "comparison_favorable_prediction_rate":
                            comp[
                                "favorable_prediction_rate"
                            ],

                        "reference_favorable_tpr":
                            ref[
                                "favorable_true_positive_rate"
                            ],

                        "comparison_favorable_tpr":
                            comp[
                                "favorable_true_positive_rate"
                            ],

                        "spd":
                            spd,

                        "eod":
                            eod,

                        "di":
                            di,

                        "abs_spd":
                            (
                                abs(
                                    spd
                                )
                                if np.isfinite(
                                    spd
                                )
                                else np.nan
                            ),

                        "abs_eod":
                            (
                                abs(
                                    eod
                                )
                                if np.isfinite(
                                    eod
                                )
                                else np.nan
                            ),

                        "di_distance_from_1":
                            (
                                abs(
                                    di - 1.0
                                )
                                if np.isfinite(
                                    di
                                )
                                else np.nan
                            ),

                        "status":
                            "COMPUTED",
                    }
                )

    return (
        fairness_rows,
        subgroup_performance_rows,
        subgroup_count_rows,
    )


# =============================================================================
# 11. PAIRWISE FAIRNESS
# =============================================================================

def compute_pairwise_fairness(
    pred_df: pd.DataFrame,
    sensitive_attributes: List[str],
) -> List[Dict[str, Any]]:

    rows = []

    grouped = pred_df.groupby(
        [
            "seed",
            "variant",
        ],
        sort=True,
    )

    for (
        seed,
        variant,
    ), run_df in grouped:

        for attribute in sensitive_attributes:

            valid_df = run_df[
                run_df[
                    attribute
                ].notna()
            ]

            groups = sorted(
                valid_df[
                    attribute
                ].unique().tolist(),
                key=lambda value:
                    str(
                        value
                    ),
            )

            components = {}

            for group in groups:

                group_df = valid_df[
                    valid_df[
                        attribute
                    ]
                    ==
                    group
                ]

                components[
                    group
                ] = subgroup_fairness_components(
                    group_df[
                        "y_true"
                    ].to_numpy(
                        dtype=int
                    ),
                    group_df[
                        "y_pred"
                    ].to_numpy(
                        dtype=int
                    ),
                )

            for i in range(
                len(
                    groups
                )
            ):

                for j in range(
                    i + 1,
                    len(
                        groups
                    )
                ):

                    group_a = groups[
                        i
                    ]

                    group_b = groups[
                        j
                    ]

                    a = components[
                        group_a
                    ]

                    b = components[
                        group_b
                    ]

                    spd_a_minus_b = fairness_difference(
                        a[
                            "favorable_prediction_rate"
                        ],
                        b[
                            "favorable_prediction_rate"
                        ],
                    )

                    eod_a_minus_b = fairness_difference(
                        a[
                            "favorable_true_positive_rate"
                        ],
                        b[
                            "favorable_true_positive_rate"
                        ],
                    )

                    di_a_over_b = disparate_impact(
                        a[
                            "favorable_prediction_rate"
                        ],
                        b[
                            "favorable_prediction_rate"
                        ],
                    )

                    di_b_over_a = disparate_impact(
                        b[
                            "favorable_prediction_rate"
                        ],
                        a[
                            "favorable_prediction_rate"
                        ],
                    )

                    rows.append(
                        {
                            "seed":
                                int(
                                    seed
                                ),

                            "variant":
                                variant,

                            "sensitive_attribute":
                                attribute,

                            "group_a":
                                group_a,

                            "group_b":
                                group_b,

                            "n_group_a":
                                a[
                                    "n"
                                ],

                            "n_group_b":
                                b[
                                    "n"
                                ],

                            "group_a_favorable_prediction_rate":
                                a[
                                    "favorable_prediction_rate"
                                ],

                            "group_b_favorable_prediction_rate":
                                b[
                                    "favorable_prediction_rate"
                                ],

                            "group_a_favorable_tpr":
                                a[
                                    "favorable_true_positive_rate"
                                ],

                            "group_b_favorable_tpr":
                                b[
                                    "favorable_true_positive_rate"
                                ],

                            "spd_a_minus_b":
                                spd_a_minus_b,

                            "eod_a_minus_b":
                                eod_a_minus_b,

                            "di_a_over_b":
                                di_a_over_b,

                            "di_b_over_a":
                                di_b_over_a,

                            "abs_spd":
                                (
                                    abs(
                                        spd_a_minus_b
                                    )
                                    if np.isfinite(
                                        spd_a_minus_b
                                    )
                                    else np.nan
                                ),

                            "abs_eod":
                                (
                                    abs(
                                        eod_a_minus_b
                                    )
                                    if np.isfinite(
                                        eod_a_minus_b
                                    )
                                    else np.nan
                                ),
                        }
                    )

    return rows


# =============================================================================
# 12. SUMMARY STATISTICS
# =============================================================================

def descriptive_summary(
    values: np.ndarray,
) -> Dict[str, Any]:

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

        return {
            "n":
                0,

            "mean":
                np.nan,

            "sd":
                np.nan,

            "median":
                np.nan,

            "min":
                np.nan,

            "max":
                np.nan,

            "ci95_lower":
                np.nan,

            "ci95_upper":
                np.nan,
        }

    mean = float(
        np.mean(
            values
        )
    )

    median = float(
        np.median(
            values
        )
    )

    minimum = float(
        np.min(
            values
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    if n >= 2:

        sd = float(
            np.std(
                values,
                ddof=1,
            )
        )

        se = sd / math.sqrt(
            n
        )

        critical = float(
            stats.t.ppf(
                0.975,
                df=n - 1,
            )
        )

        ci_lower = (
            mean
            - critical
            * se
        )

        ci_upper = (
            mean
            + critical
            * se
        )

    else:

        sd = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    return {
        "n":
            n,

        "mean":
            mean,

        "sd":
            sd,

        "median":
            median,

        "min":
            minimum,

        "max":
            maximum,

        "ci95_lower":
            ci_lower,

        "ci95_upper":
            ci_upper,
    }


def summarize_reference_fairness(
    fairness_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    df = pd.DataFrame(
        fairness_rows
    )

    if df.empty:
        return []

    df = df[
        df[
            "status"
        ]
        ==
        "COMPUTED"
    ].copy()

    if df.empty:
        return []

    summary_rows = []

    group_cols = [
        "variant",
        "sensitive_attribute",
        "reference_group",
        "comparison_group",
        "reference_selection_method",
    ]

    for keys, subset in df.groupby(
        group_cols,
        dropna=False,
        sort=True,
    ):

        (
            variant,
            attribute,
            reference_group,
            comparison_group,
            reference_method,
        ) = keys

        row = {
            "variant":
                variant,

            "sensitive_attribute":
                attribute,

            "reference_group":
                reference_group,

            "comparison_group":
                comparison_group,

            "reference_selection_method":
                reference_method,

            "favorable_label":
                FAVORABLE_LABEL,

            "favorable_outcome":
                FAVORABLE_OUTCOME_DESCRIPTION,

            "repeated_split_dependence_note":
                (
                    "Same participants may recur across test partitions; "
                    "summary is descriptive stability evidence."
                ),
        }

        for metric in [
            "spd",
            "eod",
            "di",
            "abs_spd",
            "abs_eod",
            "di_distance_from_1",
        ]:

            stats_row = descriptive_summary(
                subset[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )

            for stat_name, value in stats_row.items():

                row[
                    f"{metric}_{stat_name}"
                ] = value

        row[
            "n_seeds_total"
        ] = int(
            subset[
                "seed"
            ].nunique()
        )

        row[
            "mean_reference_n"
        ] = float(
            subset[
                "n_reference"
            ].mean()
        )

        row[
            "mean_comparison_n"
        ] = float(
            subset[
                "n_comparison"
            ].mean()
        )

        summary_rows.append(
            row
        )

    return summary_rows


def summarize_pairwise_fairness(
    pairwise_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    df = pd.DataFrame(
        pairwise_rows
    )

    if df.empty:
        return []

    summary_rows = []

    group_cols = [
        "variant",
        "sensitive_attribute",
        "group_a",
        "group_b",
    ]

    for keys, subset in df.groupby(
        group_cols,
        dropna=False,
        sort=True,
    ):

        (
            variant,
            attribute,
            group_a,
            group_b,
        ) = keys

        row = {
            "variant":
                variant,

            "sensitive_attribute":
                attribute,

            "group_a":
                group_a,

            "group_b":
                group_b,

            "repeated_split_dependence_note":
                (
                    "Same participants may recur across test partitions; "
                    "summary is descriptive stability evidence."
                ),
        }

        for metric in [
            "spd_a_minus_b",
            "eod_a_minus_b",
            "di_a_over_b",
            "di_b_over_a",
            "abs_spd",
            "abs_eod",
        ]:

            stats_row = descriptive_summary(
                subset[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )

            for stat_name, value in stats_row.items():

                row[
                    f"{metric}_{stat_name}"
                ] = value

        row[
            "n_seeds_total"
        ] = int(
            subset[
                "seed"
            ].nunique()
        )

        summary_rows.append(
            row
        )

    return summary_rows


# =============================================================================
# 13. METRIC BOUNDS AUDIT
# =============================================================================

def audit_metric_bounds(
    reference_rows: List[Dict[str, Any]],
    pairwise_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    rows = []

    def check_difference_metric(
        source: str,
        metric_name: str,
        value: Any,
        identifying: Dict[str, Any],
    ) -> None:

        val = safe_float(
            value
        )

        if np.isnan(
            val
        ):

            status = (
                "UNDEFINED_OR_MISSING"
            )

        elif (
            val
            <
            -1.0
            - BOUND_TOLERANCE
            or
            val
            >
            1.0
            + BOUND_TOLERANCE
        ):

            status = (
                "INVALID_OUTSIDE_MINUS1_PLUS1"
            )

        else:

            status = (
                "PASS"
            )

        rows.append(
            {
                "source":
                    source,

                "metric":
                    metric_name,

                "value":
                    val,

                "status":
                    status,

                **identifying,
            }
        )

    def check_di(
        source: str,
        metric_name: str,
        value: Any,
        identifying: Dict[str, Any],
    ) -> None:

        val = safe_float(
            value
        )

        if np.isnan(
            val
        ):

            status = (
                "UNDEFINED_OR_MISSING"
            )

        elif val < -BOUND_TOLERANCE:

            status = (
                "INVALID_NEGATIVE_DI"
            )

        else:

            status = (
                "PASS"
            )

        rows.append(
            {
                "source":
                    source,

                "metric":
                    metric_name,

                "value":
                    val,

                "status":
                    status,

                **identifying,
            }
        )

    for row in reference_rows:

        identifying = {
            "seed":
                row.get(
                    "seed"
                ),

            "variant":
                row.get(
                    "variant"
                ),

            "sensitive_attribute":
                row.get(
                    "sensitive_attribute"
                ),

            "reference_group":
                row.get(
                    "reference_group"
                ),

            "comparison_group":
                row.get(
                    "comparison_group"
                ),
        }

        check_difference_metric(
            "reference",
            "SPD",
            row.get(
                "spd"
            ),
            identifying,
        )

        check_difference_metric(
            "reference",
            "EOD",
            row.get(
                "eod"
            ),
            identifying,
        )

        check_di(
            "reference",
            "DI",
            row.get(
                "di"
            ),
            identifying,
        )

    for row in pairwise_rows:

        identifying = {
            "seed":
                row.get(
                    "seed"
                ),

            "variant":
                row.get(
                    "variant"
                ),

            "sensitive_attribute":
                row.get(
                    "sensitive_attribute"
                ),

            "group_a":
                row.get(
                    "group_a"
                ),

            "group_b":
                row.get(
                    "group_b"
                ),
        }

        check_difference_metric(
            "pairwise",
            "SPD",
            row.get(
                "spd_a_minus_b"
            ),
            identifying,
        )

        check_difference_metric(
            "pairwise",
            "EOD",
            row.get(
                "eod_a_minus_b"
            ),
            identifying,
        )

        check_di(
            "pairwise",
            "DI_A_OVER_B",
            row.get(
                "di_a_over_b"
            ),
            identifying,
        )

        check_di(
            "pairwise",
            "DI_B_OVER_A",
            row.get(
                "di_b_over_a"
            ),
            identifying,
        )

    return rows


# =============================================================================
# 14. INTERPRETATION AUDIT
# =============================================================================

def build_interpretation_audit(
    summary_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    rows = []

    for row in summary_rows:

        mean_spd = safe_float(
            row.get(
                "spd_mean"
            )
        )

        mean_eod = safe_float(
            row.get(
                "eod_mean"
            )
        )

        mean_di = safe_float(
            row.get(
                "di_mean"
            )
        )

        abs_spd = (
            abs(
                mean_spd
            )
            if np.isfinite(
                mean_spd
            )
            else np.nan
        )

        abs_eod = (
            abs(
                mean_eod
            )
            if np.isfinite(
                mean_eod
            )
            else np.nan
        )

        di_distance = (
            abs(
                mean_di
                - 1.0
            )
            if np.isfinite(
                mean_di
            )
            else np.nan
        )

        rows.append(
            {
                "variant":
                    row[
                        "variant"
                    ],

                "sensitive_attribute":
                    row[
                        "sensitive_attribute"
                    ],

                "reference_group":
                    row[
                        "reference_group"
                    ],

                "comparison_group":
                    row[
                        "comparison_group"
                    ],

                "mean_spd":
                    mean_spd,

                "mean_abs_spd":
                    abs_spd,

                "mean_eod":
                    mean_eod,

                "mean_abs_eod":
                    abs_eod,

                "mean_di":
                    mean_di,

                "di_distance_from_1":
                    di_distance,

                "spd_parity_target":
                    0.0,

                "eod_parity_target":
                    0.0,

                "di_parity_target":
                    1.0,

                "direction_rule":
                    (
                        "For SPD/EOD, values closer to 0 indicate "
                        "greater parity. For DI, values closer to 1 "
                        "indicate greater parity."
                    ),

                "important_warning":
                    (
                        "Do not call a lower DI value an improvement "
                        "when it moves farther from 1.0."
                    ),

                "universal_fairness_threshold_claimed":
                    False,
            }
        )

    return rows


# =============================================================================
# 15. MAIN
# =============================================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )

    print(
        "HFAGM - RECOMPUTE FAIRNESS METRICS"
    )

    print(
        "=" * 100
    )

    print(
        f"\nscikit-learn: {sklearn.__version__}"
    )

    print(
        f"numpy: {np.__version__}"
    )

    print(
        f"pandas: {pd.__version__}"
    )

    print(
        f"scipy: {scipy.__version__}"
    )

    # -----------------------------------------------------------------
    # Raw data.
    # -----------------------------------------------------------------

    (
        raw_df,
        raw_path,
        raw_source_type,
    ) = load_raw_dataset()

    print(
        f"\nRaw dataset: {raw_path}"
    )

    print(
        f"Rows: {len(raw_df)}"
    )

    # -----------------------------------------------------------------
    # Sensitive attributes.
    # -----------------------------------------------------------------

    sensitive_attributes = discover_sensitive_attributes(
        raw_df
    )

    if not sensitive_attributes:

        raise RuntimeError(
            "No configured sensitive attribute was found in the raw dataset.\n"
            f"Searched for: {SENSITIVE_ATTRIBUTE_CANDIDATES}\n"
            f"Observed columns: {raw_df.columns.tolist()}"
        )

    print(
        "\nSensitive attributes found:"
    )

    for attribute in sensitive_attributes:

        print(
            f"  - {attribute}: "
            f"{observed_group_counts(raw_df[attribute])}"
        )

    (
        attribute_inventory_rows,
        references,
    ) = build_sensitive_attribute_inventory(
        raw_df,
        sensitive_attributes,
    )

    write_csv(
        OUTPUT_DIR
        / "sensitive_attribute_inventory.csv",
        attribute_inventory_rows,
    )

    print(
        "\nOperational references:"
    )

    for attribute in sensitive_attributes:

        print(
            f"  {attribute}: "
            f"{references[attribute]['reference_group']!r} "
            f"({references[attribute]['reference_method']})"
        )

    print(
        f"\nFavorable outcome: "
        f"label {FAVORABLE_LABEL} = "
        f"{FAVORABLE_OUTCOME_DESCRIPTION}"
    )

    # -----------------------------------------------------------------
    # Load 02E predictions.
    # -----------------------------------------------------------------

    if not REPEATED_PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            "02E repeated prediction file not found:\n"
            f"{REPEATED_PREDICTIONS_PATH}"
        )

    raw_pred_df = read_csv_robust(
        REPEATED_PREDICTIONS_PATH
    )

    print(
        "\nObserved 02E prediction columns:"
    )

    print(
        raw_pred_df.columns.tolist()
    )

    (
        pred_df,
        prediction_audit_rows,
    ) = audit_and_standardize_predictions(
        raw_pred_df,
        raw_df,
    )

    write_csv(
        OUTPUT_DIR
        / "prediction_input_audit.csv",
        prediction_audit_rows,
    )

    print(
        f"\nPrediction rows: {len(pred_df)}"
    )

    print(
        f"Seeds: "
        f"{sorted(pred_df['seed'].unique().tolist())}"
    )

    print(
        f"Variants: "
        f"{sorted(pred_df['variant'].unique().tolist())}"
    )

    # -----------------------------------------------------------------
    # Attach raw sensitive attributes using original_source_row_index.
    # -----------------------------------------------------------------

    pred_df = attach_sensitive_attributes(
        pred_df,
        raw_df,
        sensitive_attributes,
    )

    joined_columns = [
        "seed",
        "variant",
        "test_partition_row",
        "source_index",
        "y_true",
        "y_pred",
        "correct_recomputed",
        "y_prob",
        "score_source",
        *sensitive_attributes,
    ]

    existing_joined_columns = [
        col
        for col in joined_columns
        if col in pred_df.columns
    ]

    pred_df[
        existing_joined_columns
    ].to_csv(
        OUTPUT_DIR
        / "predictions_with_sensitive_attributes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Reference-group fairness.
    # -----------------------------------------------------------------

    (
        fairness_reference_rows,
        subgroup_performance_rows,
        subgroup_count_rows,
    ) = compute_reference_fairness(
        pred_df,
        sensitive_attributes,
        references,
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_per_seed_reference.csv",
        fairness_reference_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "subgroup_performance_per_seed.csv",
        subgroup_performance_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "subgroup_counts_per_seed.csv",
        subgroup_count_rows,
    )

    # -----------------------------------------------------------------
    # Pairwise fairness.
    # -----------------------------------------------------------------

    pairwise_rows = compute_pairwise_fairness(
        pred_df,
        sensitive_attributes,
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_per_seed_pairwise.csv",
        pairwise_rows,
    )

    # -----------------------------------------------------------------
    # Repeated-seed summaries.
    # -----------------------------------------------------------------

    reference_summary_rows = summarize_reference_fairness(
        fairness_reference_rows
    )

    pairwise_summary_rows = summarize_pairwise_fairness(
        pairwise_rows
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_summary_reference.csv",
        reference_summary_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_summary_pairwise.csv",
        pairwise_summary_rows,
    )

    # -----------------------------------------------------------------
    # Bounds audit.
    # -----------------------------------------------------------------

    bounds_rows = audit_metric_bounds(
        fairness_reference_rows,
        pairwise_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_bounds_audit.csv",
        bounds_rows,
    )

    invalid_bounds = [
        row
        for row in bounds_rows
        if str(
            row[
                "status"
            ]
        ).startswith(
            "INVALID"
        )
    ]

    # -----------------------------------------------------------------
    # Interpretation audit.
    # -----------------------------------------------------------------

    interpretation_rows = build_interpretation_audit(
        reference_summary_rows
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_interpretation_audit.csv",
        interpretation_rows,
    )

    # -----------------------------------------------------------------
    # Primary variant.
    # -----------------------------------------------------------------

    primary_rows = [
        row
        for row in reference_summary_rows
        if row[
            "variant"
        ]
        ==
        PRIMARY_VARIANT
    ]

    # -----------------------------------------------------------------
    # Missing sensitive attributes.
    # -----------------------------------------------------------------

    missing_sensitive_summary = []

    for attribute in sensitive_attributes:

        raw_missing = int(
            raw_df[
                attribute
            ].isna().sum()
        )

        prediction_missing = int(
            pred_df[
                attribute
            ].isna().sum()
        )

        missing_sensitive_summary.append(
            {
                "sensitive_attribute":
                    attribute,

                "raw_missing":
                    raw_missing,

                "raw_missing_fraction":
                    raw_missing
                    / len(
                        raw_df
                    ),

                "prediction_rows_missing":
                    prediction_missing,

                "prediction_missing_fraction":
                    prediction_missing
                    / len(
                        pred_df
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "sensitive_attribute_missingness.csv",
        missing_sensitive_summary,
    )

    # -----------------------------------------------------------------
    # Provenance.
    # -----------------------------------------------------------------

    provenance = {
        "generated":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "script":
            "03_recompute_fairness_metrics.py",

        "raw_source":
            str(
                raw_path
            ),

        "raw_source_type":
            raw_source_type,

        "raw_source_sha256":
            sha256_file(
                raw_path
            ),

        "raw_rows":
            len(
                raw_df
            ),

        "prediction_source":
            str(
                REPEATED_PREDICTIONS_PATH
            ),

        "prediction_source_sha256":
            sha256_file(
                REPEATED_PREDICTIONS_PATH
            ),

        "prediction_rows":
            len(
                pred_df
            ),

        "source_index_column":
            "original_source_row_index",

        "probability_column":
            "y_score_positive",

        "probability_semantics":
            "score/probability for original class label 1",

        "seeds":
            safe_json(
                sorted(
                    pred_df[
                        "seed"
                    ].unique().tolist()
                )
            ),

        "variants":
            safe_json(
                sorted(
                    pred_df[
                        "variant"
                    ].unique().tolist()
                )
            ),

        "primary_variant":
            PRIMARY_VARIANT,

        "sensitive_attributes":
            safe_json(
                sensitive_attributes
            ),

        "reference_groups":
            safe_json(
                references
            ),

        "favorable_label":
            FAVORABLE_LABEL,

        "favorable_outcome_description":
            FAVORABLE_OUTCOME_DESCRIPTION,

        "spd_definition":
            (
                "P(predicted favorable | comparison) - "
                "P(predicted favorable | reference)"
            ),

        "eod_definition":
            (
                "P(predicted favorable | true favorable, comparison) - "
                "P(predicted favorable | true favorable, reference)"
            ),

        "di_definition":
            (
                "P(predicted favorable | comparison) / "
                "P(predicted favorable | reference)"
            ),

        "historical_fairness_metrics_reused":
            False,

        "predictions_from_corrected_02E":
            True,

        "reference_largest_group_is_normative_privileged_claim":
            False,

        "repeated_seed_samples_independent":
            False,

        "fairness_summary_type":
            "descriptive_repeated_split_stability",

        "sklearn_version":
            sklearn.__version__,

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "scipy_version":
            scipy.__version__,

        "python_version":
            sys.version,
    }

    write_csv(
        OUTPUT_DIR
        / "fairness_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    lines = [
        "=" * 100,
        "HFAGM - RECOMPUTED LEAKAGE-SAFE FAIRNESS METRICS",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "INPUT",
        "-" * 100,
        f"Raw source: {raw_path}",
        f"Raw rows: {len(raw_df)}",
        f"Prediction source: {REPEATED_PREDICTIONS_PATH}",
        f"Prediction rows: {len(pred_df)}",
        (
            "Seeds: "
            + str(
                sorted(
                    pred_df[
                        "seed"
                    ].unique().tolist()
                )
            )
        ),
        (
            "Variants: "
            + str(
                sorted(
                    pred_df[
                        "variant"
                    ].unique().tolist()
                )
            )
        ),
        "",
        "02E COLUMN MAPPING",
        "-" * 100,
        "original_source_row_index -> source_index",
        "y_score_positive -> y_prob",
        "test_partition_row retained only as local test-row metadata",
        "",
        "FAIRNESS DEFINITIONS",
        "-" * 100,
        (
            f"Favorable outcome: label {FAVORABLE_LABEL} = "
            f"{FAVORABLE_OUTCOME_DESCRIPTION}"
        ),
        (
            "SPD = P(predicted favorable | comparison) - "
            "P(predicted favorable | reference)"
        ),
        (
            "EOD = P(predicted favorable | true favorable, comparison) - "
            "P(predicted favorable | true favorable, reference)"
        ),
        (
            "DI = P(predicted favorable | comparison) / "
            "P(predicted favorable | reference)"
        ),
        "",
        "Parity targets:",
        "  SPD = 0",
        "  EOD = 0",
        "  DI  = 1",
        "",
        (
            "SPD and EOD must lie within [-1, 1]. "
            "DI must be nonnegative when defined."
        ),
        "",
        "SENSITIVE ATTRIBUTES",
        "-" * 100,
    ]

    for attribute in sensitive_attributes:

        counts = observed_group_counts(
            raw_df[
                attribute
            ]
        )

        lines.append(
            f"{attribute}: {counts}"
        )

        lines.append(
            (
                f"  operational reference = "
                f"{references[attribute]['reference_group']!r}"
            )
        )

        lines.append(
            (
                f"  selection method = "
                f"{references[attribute]['reference_method']}"
            )
        )

        if (
            references[
                attribute
            ][
                "reference_method"
            ]
            ==
            "AUTO_LARGEST_GROUP_NOT_NORMATIVE"
        ):

            lines.append(
                (
                    "  NOTE: largest group is used solely for reproducible "
                    "metric orientation; no privileged-group interpretation "
                    "is asserted."
                )
            )

    lines.extend(
        [
            "",
            "PRIMARY VARIANT",
            "-" * 100,
            PRIMARY_VARIANT,
            "",
        ]
    )

    if primary_rows:

        for row in primary_rows:

            lines.extend(
                [
                    (
                        f"{row['sensitive_attribute']}: "
                        f"comparison={row['comparison_group']!r}, "
                        f"reference={row['reference_group']!r}"
                    ),

                    (
                        f"  SPD: "
                        f"{format_metric(row.get('spd_mean'))} "
                        f"± {format_metric(row.get('spd_sd'))}; "
                        f"range "
                        f"[{format_metric(row.get('spd_min'))}, "
                        f"{format_metric(row.get('spd_max'))}]"
                    ),

                    (
                        f"  EOD: "
                        f"{format_metric(row.get('eod_mean'))} "
                        f"± {format_metric(row.get('eod_sd'))}; "
                        f"range "
                        f"[{format_metric(row.get('eod_min'))}, "
                        f"{format_metric(row.get('eod_max'))}]"
                    ),

                    (
                        f"  DI : "
                        f"{format_metric(row.get('di_mean'))} "
                        f"± {format_metric(row.get('di_sd'))}; "
                        f"range "
                        f"[{format_metric(row.get('di_min'))}, "
                        f"{format_metric(row.get('di_max'))}]"
                    ),

                    "",
                ]
            )

    else:

        lines.append(
            "No reference-group summary rows were produced "
            "for the primary variant."
        )

    lines.extend(
        [
            "BOUNDS AUDIT",
            "-" * 100,
            (
                f"Invalid mathematical-bound records: "
                f"{len(invalid_bounds)}"
            ),
        ]
    )

    if invalid_bounds:

        lines.append(
            "WARNING: invalid fairness values were detected."
        )

    else:

        lines.append(
            "All defined SPD/EOD/DI values satisfy their mathematical bounds."
        )

    lines.extend(
        [
            "",
            "INTERPRETATION RULES",
            "-" * 100,

            (
                "1. SPD closer to 0 indicates greater statistical-parity "
                "alignment."
            ),

            (
                "2. EOD closer to 0 indicates greater equal-opportunity "
                "alignment."
            ),

            (
                "3. DI closer to 1 indicates greater parity."
            ),

            (
                "4. A DI value moving from 1.0 toward 0.64 or 0.21 is NOT "
                "an improvement in parity."
            ),

            (
                "5. The sign of SPD/EOD depends on which subgroup is written "
                "as comparison minus reference."
            ),

            (
                "6. Repeated test splits overlap in participants, so the "
                "10-seed summaries are stability summaries rather than "
                "10 independent population samples."
            ),

            (
                "7. Small subgroup counts must be considered before drawing "
                "strong substantive fairness conclusions."
            ),

            "",
            "MANUSCRIPT USE",
            "-" * 100,

            (
                "Use these recomputed metrics instead of historical SPD/EOD/DI "
                "values whenever the fairness result is tied to the corrected "
                "clinical classifier evaluation."
            ),

            (
                "Do not describe a lower DI as a fairness improvement unless "
                "it is demonstrably closer to the parity target DI=1."
            ),

            (
                "Do not call the automatically selected largest subgroup "
                "privileged or advantaged unless that interpretation is "
                "supported separately by the study design/documentation."
            ),

            "",
            "PRIMARY OUTPUTS",
            "-" * 100,

            "sensitive_attribute_inventory.csv",
            "prediction_input_audit.csv",
            "predictions_with_sensitive_attributes.csv",
            "fairness_per_seed_reference.csv",
            "fairness_per_seed_pairwise.csv",
            "fairness_summary_reference.csv",
            "fairness_summary_pairwise.csv",
            "subgroup_performance_per_seed.csv",
            "subgroup_counts_per_seed.csv",
            "sensitive_attribute_missingness.csv",
            "fairness_bounds_audit.csv",
            "fairness_interpretation_audit.csv",
            "fairness_provenance.csv",

            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "fairness_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console output.
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "03 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nSensitive attributes: "
        f"{sensitive_attributes}"
    )

    print(
        f"Reference fairness rows: "
        f"{len(fairness_reference_rows)}"
    )

    print(
        f"Pairwise fairness rows: "
        f"{len(pairwise_rows)}"
    )

    print(
        f"Invalid bound records: "
        f"{len(invalid_bounds)}"
    )

    print(
        "\nPrimary variant:"
    )

    print(
        PRIMARY_VARIANT
    )

    print(
        "\nPrimary repeated fairness summaries:"
    )

    if primary_rows:

        for row in primary_rows:

            print(
                "\n"
                f"{row['sensitive_attribute']} | "
                f"{row['comparison_group']!r} vs "
                f"{row['reference_group']!r}"
            )

            print(
                f"  SPD = "
                f"{format_metric(row.get('spd_mean'))} "
                f"± {format_metric(row.get('spd_sd'))}"
            )

            print(
                f"  EOD = "
                f"{format_metric(row.get('eod_mean'))} "
                f"± {format_metric(row.get('eod_sd'))}"
            )

            print(
                f"  DI  = "
                f"{format_metric(row.get('di_mean'))} "
                f"± {format_metric(row.get('di_sd'))}"
            )

    else:

        print(
            "No primary summary rows."
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
        "fairness_summary.txt",
        "sensitive_attribute_inventory.csv",
        "fairness_summary_reference.csv",
        "fairness_per_seed_reference.csv",
        "fairness_summary_pairwise.csv",
        "subgroup_counts_per_seed.csv",
        "fairness_bounds_audit.csv",
        "fairness_interpretation_audit.csv",
        "prediction_input_audit.csv",
        "fairness_provenance.csv",
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
            "\n" + "=" * 100
        )

        print(
            "03 FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo historical model, dataset, fairness output, "
            "or manuscript file was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(
            1
        )