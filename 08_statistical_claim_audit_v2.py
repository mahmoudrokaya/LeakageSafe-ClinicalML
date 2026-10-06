from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_statistics_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PARTICIPANTS = 193
EXPECTED_SEEDS = [42, 47, 53, 59, 71, 83, 97, 101, 113, 127]

PRIMARY_SEARCH_ROOTS = [
    PROJECT_ROOT / "outputs" / "revision_primary_metrics",
]

FAIRNESS_SEARCH_ROOTS = [
    PROJECT_ROOT / "outputs" / "revision_fairness",
]

PRIMARY_FILENAME_PREFERENCES = [
    "repeated_seed_metrics.csv",
    "repeated_leakage_safe_metrics.csv",
    "repeated_metrics.csv",
]

PRIMARY_EXCLUSION_TERMS = (
    "single_feature",
    "chronology",
    "proxy",
    "feature_timing",
    "feature_audit",
    "fairness",
    "prediction",
    "confusion",
    "summary",
    "provenance",
    "audit",
)

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".log",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".docx",
}

EXCLUDED_DIR_EXACT = {
    ".git",
    "__pycache__",
    "site-packages",
    "dist-packages",
    "node_modules",
    "build",
    "dist",
    ".idea",
    ".vscode",
    ".pytest_cache",
}

EXCLUDED_DIR_PREFIXES = (
    ".venv",
    "venv",
    ".env",
)

SELF_MARKERS = (
    "outputs/revision_statistics_v2",
    "outputs\\revision_statistics_v2",
)

MAX_TEXT_BYTES = 25 * 1024 * 1024


PRIMARY_METRIC_ALIASES = {
    "accuracy": (
        "accuracy",
        "acc",
    ),
    "precision": (
        "precision",
        "prec",
    ),
    "recall_sensitivity": (
        "recall_sensitivity",
        "recall",
        "sensitivity",
        "tpr",
    ),
    "specificity": (
        "specificity",
        "tnr",
    ),
    "f1": (
        "f1",
        "f1_score",
        "f1score",
    ),
    "roc_auc": (
        "roc_auc",
        "rocauc",
        "auc",
    ),
}

FAIRNESS_METRIC_ALIASES = {
    "spd": (
        "spd",
        "statistical_parity_difference",
        "statisticalparitydifference",
    ),
    "eod": (
        "eod",
        "equal_opportunity_difference",
        "equalopportunitydifference",
    ),
    "di": (
        "di",
        "disparate_impact",
        "disparateimpact",
    ),
}

CONDITION_ALIASES = {
    "unbalanced": (
        "unbalanced",
        "real_training_unbalanced",
        "without_oversampling",
        "no_oversampling",
        "original",
    ),
    "oversampled": (
        "oversampled",
        "train_only_oversampled",
        "oversampling",
        "balanced",
        "smote",
    ),
}

SIGNIFICANCE_PATTERNS = [
    r"\bstatistically significant\b",
    r"\bstatistical significance\b",
    r"\bsignificantly\b",
    r"\bsignificant\b",
    r"\bp\s*[<=>]\s*0?\.\d+",
    r"\bp[\-\s]?value\b",
    r"\bp[\-\s]?values\b",
]


def normalize_name(value: Any) -> str:
    return "".join(
        char
        for char in str(value).strip().lower()
        if char.isalnum()
    )


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


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

    df = pd.DataFrame(rows)

    if columns:
        extras = [
            col
            for col in df.columns
            if col not in columns
        ]

        df = df[
            columns + extras
        ]

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def is_excluded_path(path: Path) -> bool:
    try:
        parts = [
            part.lower()
            for part in path.relative_to(PROJECT_ROOT).parts
        ]
    except Exception:
        parts = [
            part.lower()
            for part in path.parts
        ]

    for part in parts:
        if part in EXCLUDED_DIR_EXACT:
            return True

        if any(
            part.startswith(prefix)
            for prefix in EXCLUDED_DIR_PREFIXES
        ):
            return True

    full = str(path).lower()

    if any(
        marker.lower() in full
        for marker in SELF_MARKERS
    ):
        return True

    return False


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    normalized = {
        normalize_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def canonical_metric(
    column: str,
    aliases: Dict[str, Sequence[str]],
) -> Optional[str]:
    key = normalize_name(column)

    for canonical, values in aliases.items():
        for value in values:
            if key == normalize_name(value):
                return canonical

    return None


def canonical_condition(value: Any) -> str:
    lower = str(value).strip().lower()

    for canonical, aliases in CONDITION_ALIASES.items():
        if any(
            alias in lower
            for alias in aliases
        ):
            return canonical

    return lower or "unspecified"


def descriptive_summary(
    values: np.ndarray,
) -> Dict[str, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(
            np.mean(values)
        ),
        "sd": (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if len(values) > 1
            else 0.0
        ),
        "median": float(
            np.median(values)
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


def all_csvs_under(
    roots: Sequence[Path],
) -> List[Path]:
    paths: List[Path] = []

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            if is_excluded_path(path):
                continue

            paths.append(path)

    return sorted(
        set(paths),
        key=lambda p: str(p).lower(),
    )


def assess_primary_candidate(
    path: Path,
) -> Dict[str, Any]:
    df = read_csv_safe(path)

    if df is None:
        return {
            "file": relative_path(path),
            "absolute_path": str(path),
            "readable": 0,
            "rows": 0,
            "columns": 0,
            "seed_column": "",
            "condition_column": "",
            "metric_columns": safe_json({}),
            "unique_seeds": safe_json([]),
            "conditions": safe_json([]),
            "expected_seed_match": 0,
            "expected_conditions_found": 0,
            "six_primary_metrics_found": 0,
            "excluded_term": "",
            "eligible": 0,
            "selection_score": 0,
            "selected": 0,
            "reason": "CSV_READ_FAILED",
        }

    seed_col = find_column(
        df,
        [
            "seed",
            "random_seed",
            "random_state",
            "split_seed",
            "run_seed",
        ],
    )

    condition_col = find_column(
        df,
        [
            "condition",
            "variant",
            "setting",
            "training_condition",
            "scenario",
        ],
    )

    metric_cols: Dict[str, str] = {}

    for column in df.columns:
        metric = canonical_metric(
            str(column),
            PRIMARY_METRIC_ALIASES,
        )

        if metric:
            metric_cols[column] = metric

    filename_lower = path.name.lower()

    excluded_term = next(
        (
            term
            for term in PRIMARY_EXCLUSION_TERMS
            if term in filename_lower
        ),
        None,
    )

    preferred_rank = 999

    for index, preferred in enumerate(
        PRIMARY_FILENAME_PREFERENCES
    ):
        if path.name.lower() == preferred.lower():
            preferred_rank = index
            break

    seed_values: List[int] = []

    if seed_col is not None:
        seed_values = (
            pd.to_numeric(
                df[seed_col],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

    expected_seed_match = (
        sorted(set(seed_values))
        ==
        sorted(EXPECTED_SEEDS)
    )

    conditions: List[str] = []

    if condition_col is not None:
        conditions = sorted(
            {
                canonical_condition(value)
                for value in df[condition_col].dropna()
            }
        )

    condition_match = (
        "unbalanced" in conditions
        and "oversampled" in conditions
    )

    required_metrics = {
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    }

    six_metrics_found = (
        required_metrics.issubset(
            set(metric_cols.values())
        )
    )

    eligible = bool(
        seed_col is not None
        and condition_col is not None
        and six_metrics_found
        and expected_seed_match
        and condition_match
        and excluded_term is None
    )

    score = 0

    if eligible:
        score += 1000

    if preferred_rank < 999:
        score += 100 - preferred_rank

    if len(df) == 20:
        score += 50

    if expected_seed_match:
        score += 20

    if condition_match:
        score += 20

    if six_metrics_found:
        score += 20

    return {
        "file": relative_path(path),
        "absolute_path": str(path),
        "readable": 1,
        "rows": len(df),
        "columns": len(df.columns),
        "seed_column": seed_col or "",
        "condition_column": condition_col or "",
        "metric_columns": safe_json(metric_cols),
        "unique_seeds": safe_json(
            sorted(set(seed_values))
        ),
        "conditions": safe_json(conditions),
        "expected_seed_match": int(
            expected_seed_match
        ),
        "expected_conditions_found": int(
            condition_match
        ),
        "six_primary_metrics_found": int(
            six_metrics_found
        ),
        "excluded_term": excluded_term or "",
        "eligible": int(eligible),
        "selection_score": score,
        "selected": 0,
        "reason": "",
    }


def resolve_primary_source(
) -> Tuple[Path, List[Dict[str, Any]]]:
    candidates = all_csvs_under(
        PRIMARY_SEARCH_ROOTS
    )

    audit_rows = [
        assess_primary_candidate(path)
        for path in candidates
    ]

    eligible_rows = [
        row
        for row in audit_rows
        if row.get("eligible", 0) == 1
    ]

    if not eligible_rows:
        write_csv(
            OUTPUT_DIR / "primary_source_resolution.csv",
            audit_rows,
        )

        raise RuntimeError(
            "No eligible corrected 02E repeated-seed primary metric file "
            "was found. Inspect primary_source_resolution.csv."
        )

    eligible_rows = sorted(
        eligible_rows,
        key=lambda row: (
            -row.get(
                "selection_score",
                0,
            ),
            row.get(
                "file",
                "",
            ),
        ),
    )

    chosen = eligible_rows[0]

    chosen_path = Path(
        chosen["absolute_path"]
    )

    for row in audit_rows:
        if (
            row.get("file", "")
            ==
            chosen.get("file", "")
        ):
            row["selected"] = 1
            row["reason"] = (
                "Selected as corrected 02E primary repeated-seed result."
            )

        elif row.get("eligible", 0) == 1:
            row["reason"] = (
                "Eligible but lower-priority duplicate/alternative."
            )

        elif row.get("excluded_term", ""):
            row["reason"] = (
                "Excluded from primary statistics because filename indicates "
                "a chronology/proxy/fairness/audit/non-primary experiment."
            )

        elif not row.get("readable", 0):
            row["reason"] = (
                row.get("reason", "")
                or "CSV_READ_FAILED"
            )

        else:
            row["reason"] = (
                "Does not satisfy the exact corrected 02E primary schema."
            )

    return (
        chosen_path,
        audit_rows,
    )


def extract_primary_metrics(
    path: Path,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)

    seed_col = find_column(
        df,
        [
            "seed",
            "random_seed",
            "random_state",
            "split_seed",
            "run_seed",
        ],
    )

    condition_col = find_column(
        df,
        [
            "condition",
            "variant",
            "setting",
            "training_condition",
            "scenario",
        ],
    )

    if seed_col is None or condition_col is None:
        raise RuntimeError(
            "Selected primary file lacks required seed/condition columns."
        )

    metric_cols: Dict[str, str] = {}

    for column in df.columns:
        metric = canonical_metric(
            str(column),
            PRIMARY_METRIC_ALIASES,
        )

        if metric:
            metric_cols[column] = metric

    required_metrics = {
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    }

    if not required_metrics.issubset(
        set(metric_cols.values())
    ):
        raise RuntimeError(
            "Selected primary result file does not contain all six required "
            "classifier metrics."
        )

    rows: List[Dict[str, Any]] = []

    for source_row, record in df.iterrows():
        try:
            seed = int(
                float(
                    record[seed_col]
                )
            )
        except Exception:
            continue

        condition = canonical_condition(
            record[condition_col]
        )

        if condition not in {
            "unbalanced",
            "oversampled",
        }:
            continue

        for source_col, metric in metric_cols.items():
            value = pd.to_numeric(
                pd.Series(
                    [
                        record[source_col]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.isna(value):
                continue

            rows.append(
                {
                    "source_file": relative_path(path),
                    "source_row": int(source_row),
                    "seed": seed,
                    "condition": condition,
                    "metric": metric,
                    "value": float(value),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "No primary repeated metrics could be extracted."
        )

    duplicate_mask = result.duplicated(
        subset=[
            "seed",
            "condition",
            "metric",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_path = (
            OUTPUT_DIR
            / "primary_duplicate_key_error.csv"
        )

        result[
            duplicate_mask
        ].to_csv(
            duplicate_path,
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "Primary metric file contains duplicate "
            "seed×condition×metric keys. "
            "See primary_duplicate_key_error.csv."
        )

    for condition in [
        "unbalanced",
        "oversampled",
    ]:
        for metric in sorted(
            required_metrics
        ):
            subset = result[
                (
                    result["condition"]
                    ==
                    condition
                )
                &
                (
                    result["metric"]
                    ==
                    metric
                )
            ]

            seeds = sorted(
                subset["seed"]
                .astype(int)
                .tolist()
            )

            if seeds != sorted(
                EXPECTED_SEEDS
            ):
                raise RuntimeError(
                    f"Primary metric seed coverage mismatch for "
                    f"{condition}/{metric}: {seeds}"
                )

    expected_count = (
        len(EXPECTED_SEEDS)
        * 2
        * 6
    )

    if len(result) != expected_count:
        raise RuntimeError(
            f"Primary metric count mismatch: "
            f"found {len(result)}, "
            f"expected {expected_count}."
        )

    return result.to_dict(
        orient="records"
    )


def summarize_primary(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    df = pd.DataFrame(rows)

    summaries: List[Dict[str, Any]] = []

    for (
        condition,
        metric,
    ), group in df.groupby(
        [
            "condition",
            "metric",
        ],
        sort=True,
    ):
        stats = descriptive_summary(
            group["value"]
            .astype(float)
            .to_numpy()
        )

        summaries.append(
            {
                "condition": condition,
                "metric": metric,
                "n_seeds": int(
                    group["seed"].nunique()
                ),
                "seeds": safe_json(
                    sorted(
                        group["seed"]
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                ),
                **stats,
                "interpretation": (
                    "descriptive stability across repeated "
                    "leakage-safe stratified holdouts"
                ),
            }
        )

    return summaries


def pair_primary_conditions(
    rows: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    df = pd.DataFrame(rows)

    paired_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for metric in sorted(
        df["metric"].unique()
    ):
        unbalanced = (
            df[
                (
                    df["metric"]
                    ==
                    metric
                )
                &
                (
                    df["condition"]
                    ==
                    "unbalanced"
                )
            ][
                [
                    "seed",
                    "value",
                ]
            ]
            .rename(
                columns={
                    "value":
                        "unbalanced_value"
                }
            )
        )

        oversampled = (
            df[
                (
                    df["metric"]
                    ==
                    metric
                )
                &
                (
                    df["condition"]
                    ==
                    "oversampled"
                )
            ][
                [
                    "seed",
                    "value",
                ]
            ]
            .rename(
                columns={
                    "value":
                        "oversampled_value"
                }
            )
        )

        merged = unbalanced.merge(
            oversampled,
            on="seed",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(
            EXPECTED_SEEDS
        ):
            raise RuntimeError(
                f"Primary pairing produced "
                f"{len(merged)} rows "
                f"for {metric}; expected "
                f"{len(EXPECTED_SEEDS)}."
            )

        merged[
            "difference_oversampled_minus_unbalanced"
        ] = (
            merged["oversampled_value"]
            -
            merged["unbalanced_value"]
        )

        for _, record in merged.iterrows():
            paired_rows.append(
                {
                    "metric": metric,
                    "seed": int(
                        record["seed"]
                    ),
                    "unbalanced_value": float(
                        record[
                            "unbalanced_value"
                        ]
                    ),
                    "oversampled_value": float(
                        record[
                            "oversampled_value"
                        ]
                    ),
                    "difference_oversampled_minus_unbalanced": float(
                        record[
                            "difference_oversampled_minus_unbalanced"
                        ]
                    ),
                }
            )

        stats = descriptive_summary(
            merged[
                "difference_oversampled_minus_unbalanced"
            ]
            .astype(float)
            .to_numpy()
        )

        summary_rows.append(
            {
                "metric": metric,
                "n_paired_seeds": len(merged),
                "difference_mean": stats["mean"],
                "difference_sd": stats["sd"],
                "difference_median": stats["median"],
                "difference_min": stats["min"],
                "difference_max": stats["max"],
                "formal_p_value": "",
                "interpretation": (
                    "descriptive paired seed-level difference only; "
                    "not an inferential significance test"
                ),
            }
        )

    return (
        paired_rows,
        summary_rows,
    )


def assess_fairness_candidate(
    path: Path,
) -> Dict[str, Any]:
    df = read_csv_safe(path)

    if df is None:
        return {
            "file": relative_path(path),
            "absolute_path": str(path),
            "readable": 0,
            "rows": 0,
            "columns": 0,
            "seed_column": "",
            "condition_column": "",
            "sensitive_attribute_column": "",
            "fairness_metric_columns": safe_json({}),
            "eligible": 0,
            "selection_score": 0,
            "selected": 0,
            "reason": "CSV_READ_FAILED",
        }

    seed_col = find_column(
        df,
        [
            "seed",
            "random_seed",
            "split_seed",
            "run_seed",
        ],
    )

    condition_col = find_column(
        df,
        [
            "condition",
            "variant",
            "setting",
            "training_condition",
        ],
    )

    sensitive_col = find_column(
        df,
        [
            "sensitive_attribute",
            "sensitive",
            "attribute",
            "protected_attribute",
            "group_variable",
        ],
    )

    fairness_cols: Dict[str, str] = {}

    for column in df.columns:
        metric = canonical_metric(
            str(column),
            FAIRNESS_METRIC_ALIASES,
        )

        if metric:
            fairness_cols[column] = metric

    eligible = bool(
        seed_col is not None
        and condition_col is not None
        and sensitive_col is not None
        and fairness_cols
    )

    score = (
        1000
        if eligible
        else 0
    )

    if "fairness" in path.name.lower():
        score += 20

    if "repeated" in path.name.lower():
        score += 10

    if len(fairness_cols) >= 3:
        score += 30

    return {
        "file": relative_path(path),
        "absolute_path": str(path),
        "readable": 1,
        "rows": len(df),
        "columns": len(df.columns),
        "seed_column": seed_col or "",
        "condition_column": condition_col or "",
        "sensitive_attribute_column": sensitive_col or "",
        "fairness_metric_columns": safe_json(
            fairness_cols
        ),
        "eligible": int(eligible),
        "selection_score": score,
        "selected": 0,
        "reason": "",
    }


def resolve_fairness_sources(
) -> Tuple[
    List[Path],
    List[Dict[str, Any]],
]:
    candidates = all_csvs_under(
        FAIRNESS_SEARCH_ROOTS
    )

    audit_rows = [
        assess_fairness_candidate(path)
        for path in candidates
    ]

    eligible_rows = [
        row
        for row in audit_rows
        if row.get("eligible", 0) == 1
    ]

    selected_paths: List[Path] = []
    seen_hashes = set()

    for row in sorted(
        eligible_rows,
        key=lambda item: (
            -item.get(
                "selection_score",
                0,
            ),
            item.get(
                "file",
                "",
            ),
        ),
    ):
        path = Path(
            row["absolute_path"]
        )

        file_hash = sha256_file(path)

        if file_hash in seen_hashes:
            row["reason"] = (
                "Exact duplicate content of another selected fairness table."
            )
            continue

        seen_hashes.add(
            file_hash
        )

        selected_paths.append(
            path
        )

        row["selected"] = 1

        row["reason"] = (
            "Selected as corrected repeated fairness evidence."
        )

    for row in audit_rows:
        if (
            row.get("selected", 0) == 0
            and not row.get(
                "reason",
                "",
            )
        ):
            row["reason"] = (
                "Does not contain seed + condition + sensitive attribute + "
                "fairness metric schema."
            )

    return (
        selected_paths,
        audit_rows,
    )


def optional_identifier_columns(
    df: pd.DataFrame,
) -> List[str]:
    candidate_groups = [
        [
            "reference_group",
            "reference",
            "privileged_group",
        ],
        [
            "comparison_group",
            "comparison",
            "unprivileged_group",
        ],
        [
            "favorable_label",
            "positive_label",
        ],
    ]

    found: List[str] = []

    for candidates in candidate_groups:
        col = find_column(
            df,
            candidates,
        )

        if col is not None:
            found.append(col)

    return found


def extract_fairness_rows(
    paths: Sequence[Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path in paths:
        df = pd.read_csv(path)

        seed_col = find_column(
            df,
            [
                "seed",
                "random_seed",
                "split_seed",
                "run_seed",
            ],
        )

        condition_col = find_column(
            df,
            [
                "condition",
                "variant",
                "setting",
                "training_condition",
            ],
        )

        sensitive_col = find_column(
            df,
            [
                "sensitive_attribute",
                "sensitive",
                "attribute",
                "protected_attribute",
                "group_variable",
            ],
        )

        if (
            seed_col is None
            or condition_col is None
            or sensitive_col is None
        ):
            continue

        fairness_cols: Dict[str, str] = {}

        for column in df.columns:
            metric = canonical_metric(
                str(column),
                FAIRNESS_METRIC_ALIASES,
            )

            if metric:
                fairness_cols[column] = metric

        identifier_cols = (
            optional_identifier_columns(df)
        )

        for source_row, record in df.iterrows():
            try:
                seed = int(
                    float(
                        record[seed_col]
                    )
                )
            except Exception:
                continue

            condition = canonical_condition(
                record[condition_col]
            )

            if condition not in {
                "unbalanced",
                "oversampled",
            }:
                continue

            sensitive_attribute = str(
                record[sensitive_col]
            ).strip()

            identifiers: Dict[str, str] = {}

            for column in identifier_cols:
                identifiers[
                    normalize_name(column)
                ] = str(
                    record[column]
                )

            comparison_key = safe_json(
                identifiers
            )

            for source_col, metric in fairness_cols.items():
                value = pd.to_numeric(
                    pd.Series(
                        [
                            record[source_col]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                if pd.isna(value):
                    continue

                rows.append(
                    {
                        "source_file": relative_path(path),
                        "source_row": int(source_row),
                        "seed": seed,
                        "condition": condition,
                        "sensitive_attribute": sensitive_attribute,
                        "comparison_identifiers": comparison_key,
                        "metric": metric,
                        "value": float(value),
                    }
                )

    if not rows:
        return []

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        subset=[
            "seed",
            "condition",
            "sensitive_attribute",
            "comparison_identifiers",
            "metric",
            "value",
        ]
    )

    return df.to_dict(
        orient="records"
    )


def summarize_fairness(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    df = pd.DataFrame(rows)

    summaries: List[Dict[str, Any]] = []

    for (
        condition,
        attribute,
        identifiers,
        metric,
    ), group in df.groupby(
        [
            "condition",
            "sensitive_attribute",
            "comparison_identifiers",
            "metric",
        ],
        dropna=False,
        sort=True,
    ):
        stats = descriptive_summary(
            group["value"]
            .astype(float)
            .to_numpy()
        )

        summaries.append(
            {
                "condition": condition,
                "sensitive_attribute": attribute,
                "comparison_identifiers": identifiers,
                "metric": metric,
                "n_unique_seeds": int(
                    group["seed"].nunique()
                ),
                "n_rows": len(group),
                **stats,
                "interpretation": (
                    "descriptive corrected classifier fairness across "
                    "repeated holdouts; not synthetic-generator fairness"
                ),
            }
        )

    return summaries


def pair_fairness_conditions(
    rows: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not rows:
        return (
            [],
            [],
        )

    df = pd.DataFrame(rows)

    paired_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    group_columns = [
        "sensitive_attribute",
        "comparison_identifiers",
        "metric",
    ]

    for group_key, group in df.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        (
            sensitive_attribute,
            identifiers,
            metric,
        ) = group_key

        unbalanced = (
            group[
                group["condition"]
                ==
                "unbalanced"
            ][
                [
                    "seed",
                    "value",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "seed"
                ]
            )
            .rename(
                columns={
                    "value":
                        "unbalanced_value"
                }
            )
        )

        oversampled = (
            group[
                group["condition"]
                ==
                "oversampled"
            ][
                [
                    "seed",
                    "value",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "seed"
                ]
            )
            .rename(
                columns={
                    "value":
                        "oversampled_value"
                }
            )
        )

        merged = unbalanced.merge(
            oversampled,
            on="seed",
            how="inner",
            validate="one_to_one",
        )

        if merged.empty:
            continue

        if len(merged) > len(
            EXPECTED_SEEDS
        ):
            raise RuntimeError(
                f"Fairness pairing inflation detected for "
                f"{sensitive_attribute}/{metric}: "
                f"{len(merged)} pairs > "
                f"{len(EXPECTED_SEEDS)} expected maximum."
            )

        merged[
            "difference_oversampled_minus_unbalanced"
        ] = (
            merged["oversampled_value"]
            -
            merged["unbalanced_value"]
        )

        for _, record in merged.iterrows():
            paired_rows.append(
                {
                    "sensitive_attribute": sensitive_attribute,
                    "comparison_identifiers": identifiers,
                    "metric": metric,
                    "seed": int(
                        record["seed"]
                    ),
                    "unbalanced_value": float(
                        record[
                            "unbalanced_value"
                        ]
                    ),
                    "oversampled_value": float(
                        record[
                            "oversampled_value"
                        ]
                    ),
                    "difference_oversampled_minus_unbalanced": float(
                        record[
                            "difference_oversampled_minus_unbalanced"
                        ]
                    ),
                }
            )

        stats = descriptive_summary(
            merged[
                "difference_oversampled_minus_unbalanced"
            ]
            .astype(float)
            .to_numpy()
        )

        summary_rows.append(
            {
                "sensitive_attribute": sensitive_attribute,
                "comparison_identifiers": identifiers,
                "metric": metric,
                "n_paired_seeds": len(merged),
                "difference_mean": stats["mean"],
                "difference_sd": stats["sd"],
                "difference_median": stats["median"],
                "difference_min": stats["min"],
                "difference_max": stats["max"],
                "formal_p_value": "",
                "interpretation": (
                    "descriptive condition difference paired by "
                    "seed + sensitive-attribute comparison only"
                ),
            }
        )

    return (
        paired_rows,
        summary_rows,
    )


def build_excluded_source_inventory(
    primary_audit_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in primary_audit_rows:
        if row.get(
            "selected",
            0,
        ) == 1:
            continue

        file_value = row.get(
            "file",
            "",
        )

        file_lower = (
            file_value.lower()
        )

        category = (
            "other_nonprimary"
        )

        if any(
            token in file_lower
            for token in (
                "single_feature",
                "chronology",
                "proxy",
            )
        ):
            category = (
                "single_feature_or_chronology_experiment"
            )

        elif "fairness" in file_lower:
            category = (
                "fairness_separate_analysis"
            )

        elif not row.get(
            "readable",
            0,
        ):
            category = (
                "unreadable_csv"
            )

        rows.append(
            {
                "file": file_value,
                "category": category,
                "excluded_from_primary_statistics": 1,
                "reason": row.get(
                    "reason",
                    "",
                ),
            }
        )

    return rows


def read_docx_text(
    path: Path,
) -> Optional[str]:
    try:
        with zipfile.ZipFile(
            path,
            "r",
        ) as archive:
            if (
                "word/document.xml"
                not in archive.namelist()
            ):
                return None

            xml = archive.read(
                "word/document.xml"
            )

        root = ET.fromstring(xml)

        text_parts: List[str] = []

        for element in root.iter():
            if (
                element.tag.endswith("}t")
                and element.text
            ):
                text_parts.append(
                    element.text
                )

            elif element.tag.endswith(
                "}p"
            ):
                text_parts.append(
                    "\n"
                )

        return " ".join(
            text_parts
        )

    except Exception:
        return None


def read_text_safe(
    path: Path,
) -> Optional[str]:
    try:
        if (
            path.stat().st_size
            >
            MAX_TEXT_BYTES
        ):
            return None

    except Exception:
        return None

    if path.suffix.lower() == ".docx":
        return read_docx_text(path)

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]:
        try:
            return path.read_text(
                encoding=encoding,
                errors="ignore",
            )

        except Exception:
            pass

    return None


def discover_text_sources(
) -> List[Path]:
    paths: List[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if is_excluded_path(path):
            continue

        if (
            path.suffix.lower()
            not in TEXT_EXTENSIONS
        ):
            continue

        if (
            path.name.lower()
            ==
            "08_statistical_claim_audit_v2.py"
        ):
            continue

        paths.append(path)

    return sorted(
        paths,
        key=lambda p: str(p).lower(),
    )


def audit_significance_language(
) -> List[Dict[str, Any]]:
    compiled = [
        re.compile(
            pattern,
            flags=re.IGNORECASE,
        )
        for pattern in SIGNIFICANCE_PATTERNS
    ]

    rows: List[Dict[str, Any]] = []

    for path in discover_text_sources():
        text = read_text_safe(path)

        if not text:
            continue

        lines = text.splitlines()

        for line_no, line in enumerate(
            lines,
            start=1,
        ):
            matched = [
                pattern.pattern
                for pattern in compiled
                if pattern.search(line)
            ]

            if not matched:
                continue

            start = max(
                0,
                line_no - 3,
            )

            stop = min(
                len(lines),
                line_no + 2,
            )

            context = " | ".join(
                value.strip()
                for value in lines[
                    start:stop
                ]
            )[:4000]

            lower_context = (
                context.lower()
            )

            cautionary = any(
                phrase in lower_context
                for phrase in (
                    "not significant",
                    "not statistically",
                    "cannot claim",
                    "do not claim",
                    "should not",
                    "unsupported",
                    "descriptive",
                    "not independent",
                    "remove significant",
                )
            )

            explicit_p = bool(
                re.search(
                    r"\bp\s*[<=>]\s*0?\.\d+",
                    line,
                    flags=re.IGNORECASE,
                )
            )

            significant_word = bool(
                re.search(
                    r"\bsignificant(?:ly)?\b",
                    line,
                    flags=re.IGNORECASE,
                )
            )

            if cautionary:
                classification = (
                    "KEEP_AS_CAUTION_OR_LIMITATION"
                )

                action = (
                    "Retain if the statement explicitly rejects unsupported "
                    "inferential interpretation."
                )

            elif explicit_p:
                classification = (
                    "VERIFY_OR_REMOVE_P_VALUE"
                )

                action = (
                    "Retain only if tied to a valid documented inferential "
                    "design and clearly defined experimental unit."
                )

            elif significant_word:
                classification = (
                    "REMOVE_OR_REWRITE_SIGNIFICANCE_WORDING"
                )

                action = (
                    "Replace significant/significantly with descriptive "
                    "language supported by corrected repeated-run results."
                )

            else:
                classification = (
                    "REVIEW_STATISTICAL_LANGUAGE"
                )

                action = (
                    "Ensure wording is explicitly descriptive rather than "
                    "inferential."
                )

            rows.append(
                {
                    "file": relative_path(path),
                    "line": line_no,
                    "matched_patterns": safe_json(
                        matched
                    ),
                    "text": line.strip()[:3000],
                    "context": context,
                    "classification": classification,
                    "recommended_action": action,
                }
            )

    return rows


def build_verification_matrix(
    primary_rows: List[Dict[str, Any]],
    primary_paired_summary: List[Dict[str, Any]],
    fairness_paired_summary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    exact_primary_count = (
        len(EXPECTED_SEEDS)
        * 2
        * 6
    )

    primary_pair_counts_valid = (
        len(primary_paired_summary)
        ==
        6
        and all(
            row.get(
                "n_paired_seeds",
                0,
            )
            ==
            len(EXPECTED_SEEDS)
            for row in primary_paired_summary
        )
    )

    fairness_pair_counts_valid = all(
        row.get(
            "n_paired_seeds",
            0,
        )
        <=
        len(EXPECTED_SEEDS)
        for row in fairness_paired_summary
    )

    return [
        {
            "criterion": (
                "primary_02E_source_resolved"
            ),
            "passed": int(
                bool(primary_rows)
            ),
            "detail": (
                "Corrected 02E repeated-seed classifier result was "
                "explicitly resolved."
            ),
        },
        {
            "criterion": (
                "primary_metric_count_exact"
            ),
            "passed": int(
                len(primary_rows)
                ==
                exact_primary_count
            ),
            "detail": (
                f"Recovered {len(primary_rows)} primary metric values; "
                f"expected {exact_primary_count} = "
                f"10 seeds × 2 conditions × 6 metrics."
            ),
        },
        {
            "criterion": (
                "primary_seed_pairing_exact"
            ),
            "passed": int(
                primary_pair_counts_valid
            ),
            "detail": (
                "Each primary metric comparison contains exactly "
                "10 one-to-one paired seeds."
            ),
        },
        {
            "criterion": (
                "single_feature_experiments_excluded_from_primary"
            ),
            "passed": 1,
            "detail": (
                "Primary statistics are restricted to the selected 02E "
                "classifier table."
            ),
        },
        {
            "criterion": (
                "fairness_paired_without_cartesian_inflation"
            ),
            "passed": int(
                fairness_pair_counts_valid
            ),
            "detail": (
                "Fairness pairing is grouped by sensitive attribute, "
                "comparison identifiers, metric, and seed."
            ),
        },
        {
            "criterion": (
                "inferential_confidence_intervals_removed"
            ),
            "passed": 1,
            "detail": (
                "Only mean, SD, median, minimum, and maximum are reported."
            ),
        },
        {
            "criterion": (
                "formal_p_values_not_generated"
            ),
            "passed": 1,
            "detail": (
                "No p-value or inferential significance test is produced."
            ),
        },
        {
            "criterion": (
                "independent_experimental_replicates_available"
            ),
            "passed": 0,
            "detail": (
                "Repeated holdouts reuse the same 193 participants and are "
                "not independent datasets."
            ),
        },
    ]


def build_final_verdict(
    matrix: List[Dict[str, Any]],
) -> Dict[str, Any]:
    critical = {
        row["criterion"]:
            row["passed"]
        for row in matrix
    }

    required_passes = [
        "primary_02E_source_resolved",
        "primary_metric_count_exact",
        "primary_seed_pairing_exact",
        "single_feature_experiments_excluded_from_primary",
        "fairness_paired_without_cartesian_inflation",
        "inferential_confidence_intervals_removed",
        "formal_p_values_not_generated",
    ]

    clean = all(
        critical.get(
            criterion,
            0,
        ) == 1
        for criterion in required_passes
    )

    if clean:
        verdict = (
            "DESCRIPTIVE_STABILITY_EVIDENCE_VALID_"
            "FORMAL_SIGNIFICANCE_NOT_SUPPORTED"
        )

        next_action = (
            "USE_MEAN_SD_MIN_MAX_AND_REMOVE_UNSUPPORTED_SIGNIFICANCE_WORDING"
        )

        manuscript_action = (
            "Report the corrected 02E repeated-holdout classifier results "
            "using descriptive mean ± SD, optionally accompanied by median "
            "and range. Treat seed-level differences as descriptive only. "
            "Do not claim statistically significant superiority from these "
            "dependent repeated holdouts."
        )

    else:
        verdict = (
            "STATISTICAL_AUDIT_REQUIRES_FURTHER_CORRECTION"
        )

        next_action = (
            "DO_NOT_USE_STATISTICS_IN_MANUSCRIPT_UNTIL_FAILED_CHECKS_ARE_RESOLVED"
        )

        manuscript_action = (
            "One or more v2 integrity checks failed. Do not transfer the "
            "generated statistics into the manuscript until the relevant "
            "source or pairing problem is corrected."
        )

    return {
        "verdict": verdict,
        "next_action": next_action,
        "manuscript_action": manuscript_action,
        "formal_significance_supported": False,
        "independent_dataset_replicates": 0,
        "original_participants": (
            EXPECTED_PARTICIPANTS
        ),
        "expected_repeated_seeds": (
            len(EXPECTED_SEEDS)
        ),
        "new_model_training_performed": False,
        "new_synthetic_generation_performed": False,
        "new_inferential_experiment_created": False,
    }


def main() -> None:
    print("=" * 100)
    print(
        "HFAGM - STATISTICAL CLAIM AUDIT V2"
    )
    print("=" * 100)

    print()
    print("V2 restrictions:")
    print(
        "  - primary statistics use only corrected 02E repeated-seed metrics"
    )
    print(
        "  - fairness is analyzed separately"
    )
    print(
        "  - fairness pairing uses seed + sensitive attribute/comparison"
    )
    print(
        "  - no inferential CI"
    )
    print(
        "  - no p-value"
    )
    print(
        "  - no significance claim"
    )

    primary_path, primary_source_audit = (
        resolve_primary_source()
    )

    write_csv(
        OUTPUT_DIR
        / "primary_source_resolution.csv",
        primary_source_audit,
    )

    print()
    print(
        "Selected primary 02E file:"
    )
    print(
        primary_path
    )

    primary_rows = extract_primary_metrics(
        primary_path
    )

    write_csv(
        OUTPUT_DIR
        / "primary_repeated_metric_long.csv",
        primary_rows,
    )

    primary_summary = summarize_primary(
        primary_rows
    )

    write_csv(
        OUTPUT_DIR
        / "primary_repeated_metric_summary.csv",
        primary_summary,
    )

    (
        primary_paired_rows,
        primary_paired_summary,
    ) = pair_primary_conditions(
        primary_rows
    )

    write_csv(
        OUTPUT_DIR
        / "primary_paired_seed_differences.csv",
        primary_paired_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "primary_paired_difference_summary.csv",
        primary_paired_summary,
    )

    excluded_sources = (
        build_excluded_source_inventory(
            primary_source_audit
        )
    )

    write_csv(
        OUTPUT_DIR
        / "excluded_repeated_sources.csv",
        excluded_sources,
    )

    (
        fairness_paths,
        fairness_source_audit,
    ) = resolve_fairness_sources()

    write_csv(
        OUTPUT_DIR
        / "fairness_source_resolution.csv",
        fairness_source_audit,
    )

    fairness_rows = (
        extract_fairness_rows(
            fairness_paths
        )
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_metric_long.csv",
        fairness_rows,
    )

    fairness_summary = (
        summarize_fairness(
            fairness_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_metric_summary.csv",
        fairness_summary,
    )

    (
        fairness_paired_rows,
        fairness_paired_summary,
    ) = pair_fairness_conditions(
        fairness_rows
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_paired_differences.csv",
        fairness_paired_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_paired_difference_summary.csv",
        fairness_paired_summary,
    )

    statistical_language_rows = (
        audit_significance_language()
    )

    write_csv(
        OUTPUT_DIR
        / "statistical_claim_language_audit.csv",
        statistical_language_rows,
    )

    inferential_rows = [
        {
            "criterion": (
                "multiple_repeated_seeds"
            ),
            "status": "PASS",
            "detail": (
                f"{len(EXPECTED_SEEDS)} repeated leakage-safe stratified "
                "holdout seeds are available."
            ),
        },
        {
            "criterion": (
                "same_participant_pool_reused"
            ),
            "status": "YES",
            "detail": (
                "All repeated runs originate from the same "
                "193-participant cohort."
            ),
        },
        {
            "criterion": (
                "independent_dataset_replicates"
            ),
            "status": "NO",
            "detail": (
                "No independent external cohorts or independent "
                "experimental datasets are present."
            ),
        },
        {
            "criterion": (
                "formal_significance_from_seed_replicates"
            ),
            "status": (
                "NOT_SUPPORTED"
            ),
            "detail": (
                "Seed-level repeated holdouts are dependent resamples and "
                "are interpreted descriptively."
            ),
        },
        {
            "criterion": (
                "descriptive_mean_sd_range"
            ),
            "status": "SUPPORTED",
            "detail": (
                "Mean, SD, median, minimum, maximum, and paired seed-level "
                "differences are appropriate descriptive stability summaries."
            ),
        },
    ]

    write_csv(
        OUTPUT_DIR
        / "inferential_eligibility_v2.csv",
        inferential_rows,
    )

    matrix = build_verification_matrix(
        primary_rows,
        primary_paired_summary,
        fairness_paired_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "statistical_verification_matrix_v2.csv",
        matrix,
    )

    verdict = build_final_verdict(
        matrix
    )

    write_csv(
        OUTPUT_DIR
        / "statistical_verdict_v2.csv",
        [
            verdict
        ],
    )

    provenance = {
        "generated": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "script": (
            "08_statistical_claim_audit_v2.py"
        ),
        "project_root": (
            str(PROJECT_ROOT)
        ),
        "primary_source": (
            relative_path(
                primary_path
            )
        ),
        "primary_metric_values": (
            len(primary_rows)
        ),
        "primary_summary_rows": (
            len(primary_summary)
        ),
        "primary_paired_rows": (
            len(primary_paired_rows)
        ),
        "primary_paired_summary_rows": (
            len(primary_paired_summary)
        ),
        "fairness_sources_selected": (
            len(fairness_paths)
        ),
        "fairness_metric_rows": (
            len(fairness_rows)
        ),
        "fairness_summary_rows": (
            len(fairness_summary)
        ),
        "fairness_paired_rows": (
            len(fairness_paired_rows)
        ),
        "fairness_paired_summary_rows": (
            len(fairness_paired_summary)
        ),
        "significance_language_rows": (
            len(statistical_language_rows)
        ),
        "original_participants": (
            EXPECTED_PARTICIPANTS
        ),
        "expected_seeds": (
            safe_json(
                EXPECTED_SEEDS
            )
        ),
        "formal_inferential_test_performed": False,
        "p_value_generated": False,
        "confidence_interval_generated": False,
        "new_training_performed": False,
        "new_synthetic_data_generated": False,
        "verdict": verdict["verdict"],
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    write_csv(
        OUTPUT_DIR
        / "statistical_provenance_v2.csv",
        [
            provenance
        ],
    )

    lines = [
        "=" * 100,
        (
            "HFAGM - STATISTICAL CLAIM / DESCRIPTIVE UNCERTAINTY AUDIT V2"
        ),
        "=" * 100,
        "",
        (
            f"Generated: "
            f"{provenance['generated']}"
        ),
        "",
        "V2 CORRECTIONS",
        "-" * 100,
        (
            "Primary performance statistics are restricted to the corrected "
            "02E repeated leakage-safe classifier evaluation."
        ),
        (
            "Single-feature chronology/proxy experiments are excluded from "
            "primary performance aggregation."
        ),
        (
            "Fairness statistics are analyzed separately and paired by seed "
            "plus sensitive-attribute/comparison identifiers."
        ),
        (
            "No inferential confidence intervals or p-values are generated."
        ),
        "",
        "PRIMARY SOURCE",
        "-" * 100,
        relative_path(
            primary_path
        ),
        "",
        "PRIMARY DATA INTEGRITY",
        "-" * 100,
        (
            f"Expected seeds: "
            f"{EXPECTED_SEEDS}"
        ),
        (
            f"Primary metric values: "
            f"{len(primary_rows)} "
            "(expected 120 = "
            "10 seeds × 2 conditions × 6 metrics)"
        ),
        (
            f"Primary metric summaries: "
            f"{len(primary_summary)}"
        ),
        (
            f"Primary paired metric summaries: "
            f"{len(primary_paired_summary)}"
        ),
        "",
        "PRIMARY PERFORMANCE",
        "-" * 100,
    ]

    summary_lookup = {
        (
            row["condition"],
            row["metric"],
        ):
            row
        for row in primary_summary
    }

    difference_lookup = {
        row["metric"]:
            row
        for row in primary_paired_summary
    }

    metric_order = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    ]

    for metric in metric_order:
        unbalanced = summary_lookup[
            (
                "unbalanced",
                metric,
            )
        ]

        oversampled = summary_lookup[
            (
                "oversampled",
                metric,
            )
        ]

        difference = (
            difference_lookup[
                metric
            ]
        )

        lines.append(
            f"{metric}: "
            f"unbalanced mean="
            f"{unbalanced['mean']:.10f}, "
            f"SD="
            f"{unbalanced['sd']:.10f}, "
            f"median="
            f"{unbalanced['median']:.10f}, "
            f"range=["
            f"{unbalanced['min']:.10f}, "
            f"{unbalanced['max']:.10f}]"
        )

        lines.append(
            f"    oversampled mean="
            f"{oversampled['mean']:.10f}, "
            f"SD="
            f"{oversampled['sd']:.10f}, "
            f"median="
            f"{oversampled['median']:.10f}, "
            f"range=["
            f"{oversampled['min']:.10f}, "
            f"{oversampled['max']:.10f}]"
        )

        lines.append(
            f"    descriptive mean difference "
            f"(oversampled - unbalanced)="
            f"{difference['difference_mean']:.10f}; "
            f"paired seeds="
            f"{difference['n_paired_seeds']}"
        )

    lines.extend(
        [
            "",
            "FAIRNESS ANALYSIS",
            "-" * 100,
            (
                f"Selected fairness source files: "
                f"{len(fairness_paths)}"
            ),
            (
                f"Fairness metric rows: "
                f"{len(fairness_rows)}"
            ),
            (
                f"Fairness paired summaries: "
                f"{len(fairness_paired_summary)}"
            ),
        ]
    )

    if fairness_paired_summary:
        for row in fairness_paired_summary:
            lines.append(
                f"{row['sensitive_attribute']} | "
                f"{row['metric']} | "
                f"paired seeds="
                f"{row['n_paired_seeds']} | "
                f"mean difference="
                f"{row['difference_mean']:.10f}"
            )

    else:
        lines.append(
            "No eligible repeated fairness pair was reconstructed."
        )

    lines.extend(
        [
            "",
            "STATISTICAL INTERPRETATION",
            "-" * 100,
            (
                "The ten random seeds correspond to repeated train/test "
                "partitions of the same 193-participant cohort."
            ),
            (
                "They are therefore dependent resamples rather than ten "
                "independent clinical datasets."
            ),
            (
                "The corrected statistics support descriptive stability "
                "reporting using mean ± SD and range."
            ),
            (
                "No formal claim of statistical significance is made."
            ),
            "",
            "VERIFICATION MATRIX",
            "-" * 100,
        ]
    )

    for row in matrix:
        state = (
            "PASS"
            if row["passed"] == 1
            else "MISSING/NO"
        )

        lines.append(
            f"{state}: "
            f"{row['criterion']}"
        )

        lines.append(
            f"    "
            f"{row['detail']}"
        )

    lines.extend(
        [
            "",
            "FINAL VERDICT",
            "-" * 100,
            verdict["verdict"],
            "",
            "NEXT ACTION",
            "-" * 100,
            verdict["next_action"],
            "",
            "MANUSCRIPT ACTION",
            "-" * 100,
            verdict["manuscript_action"],
            "",
            "RECOMMENDED REPORTING STYLE",
            "-" * 100,
            (
                "Preferred: 'Across ten repeated leakage-safe stratified "
                "holdouts, performance remained stable, with ROC-AUC "
                "reported as mean ± SD.'"
            ),
            (
                "Avoid: 'The method significantly outperformed the baseline' "
                "unless a separate valid inferential design is available."
            ),
            "",
            "IMPORTANT LIMITATION",
            "-" * 100,
            (
                "Repeated resampling does not increase the independent "
                "sample size beyond the original 193 participants."
            ),
            "",
            "SAFETY CONFIRMATION",
            "-" * 100,
            "New model training: NO",
            "New synthetic generation: NO",
            "Formal significance test: NO",
            "P-value generated: NO",
            (
                "Inferential confidence interval generated: NO"
            ),
            (
                "Historical project files modified: NO"
            ),
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "statistical_claim_audit_v2_summary.txt"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("08 V2 COMPLETE")
    print("=" * 100)
    print()

    print(
        f"Primary metric values: "
        f"{len(primary_rows)}"
    )

    print(
        f"Expected primary metric values: "
        f"{len(EXPECTED_SEEDS) * 2 * 6}"
    )

    print(
        f"Primary paired summaries: "
        f"{len(primary_paired_summary)}"
    )

    print(
        f"Fairness metric rows: "
        f"{len(fairness_rows)}"
    )

    print(
        f"Fairness paired summaries: "
        f"{len(fairness_paired_summary)}"
    )

    print()
    print("FINAL VERDICT:")
    print(
        verdict["verdict"]
    )

    print()
    print("NEXT ACTION:")
    print(
        verdict["next_action"]
    )

    print()
    print("Results written to:")
    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Upload these files first:"
    )

    for filename in [
        "statistical_claim_audit_v2_summary.txt",
        "statistical_verdict_v2.csv",
        "statistical_verification_matrix_v2.csv",
        "primary_source_resolution.csv",
        "primary_repeated_metric_summary.csv",
        "primary_paired_difference_summary.csv",
        "fairness_source_resolution.csv",
        "fairness_metric_summary.csv",
        "fairness_paired_difference_summary.csv",
        "excluded_repeated_sources.csv",
        "inferential_eligibility_v2.csv",
        "statistical_claim_language_audit.csv",
        "statistical_provenance_v2.csv",
    ]:
        print(
            OUTPUT_DIR
            / filename
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 100)
        print(
            "08 V2 FAILED SAFELY"
        )
        print("=" * 100)
        print()

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "No model was trained, no synthetic data were generated, "
            "and no historical artifact was modified."
        )

        print()
        print(
            "Full traceback:"
        )
        print()

        traceback.print_exc()

        sys.exit(1)