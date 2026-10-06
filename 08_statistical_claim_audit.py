"""
08_statistical_claim_audit.py
=============================

HFAGM revision - statistical claim and uncertainty audit.

PURPOSE
-------
Reviewer #3 requested statistical support for statements such as
"significantly improved".

Earlier corrected experiments established:

    - the original evaluation suffered leakage;
    - corrected evaluation uses leakage-safe split-first processing;
    - repeated evaluation uses multiple random stratified holdout seeds;
    - those repeated holdouts reuse the same 193 participants;
    - therefore repeated seeds are NOT independent datasets.

Accordingly, this script:

1. Audits historical/revision files for statistical-significance language.
2. Finds corrected repeated-run result tables.
3. Reconstructs descriptive mean, SD, median, min, max, and 95% uncertainty
   intervals where raw repeated-run values are available.
4. Computes paired per-seed differences ONLY as descriptive stability evidence.
5. Does NOT convert repeated overlapping holdouts into independent replicates.
6. Does NOT claim statistical significance from the 10 repeated holdouts.
7. Audits whether any historical p-value/significance claim has an identifiable
   valid experimental basis.
8. Produces exact manuscript guidance:
       - KEEP
       - REWRITE_AS_DESCRIPTIVE
       - REMOVE_UNSUPPORTED_SIGNIFICANCE_CLAIM

STRICT PRINCIPLE
----------------
Repeated random holdouts over the same small participant dataset are dependent
resamples. They can characterize stability but should not be presented as
independent experimental replicates supporting conventional inferential
significance claims.

OUTPUT
------
outputs/revision_statistics/

    statistical_source_inventory.csv
    significance_language_evidence.csv
    repeated_result_candidate_inventory.csv
    repeated_metric_long.csv
    repeated_metric_summary.csv
    paired_condition_differences.csv
    paired_difference_summary.csv
    inferential_test_eligibility.csv
    historical_statistical_claim_audit.csv
    statistical_verification_matrix.csv
    statistical_verdict.csv
    statistical_provenance.csv
    statistical_claim_audit_summary.txt
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
import traceback
import zipfile

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_statistics"
)

EXPECTED_REAL_PARTICIPANTS = 193

# -------------------------------------------------------------------------
# Corrected result locations from prior revision stages.
# -------------------------------------------------------------------------

CORRECTED_OUTPUT_ROOTS = [
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics",

    PROJECT_ROOT
    / "outputs"
    / "revision_fairness",

    PROJECT_ROOT
    / "outputs"
    / "revision_utility",
]

# -------------------------------------------------------------------------
# Text formats audited for significance claims.
# -------------------------------------------------------------------------

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

MAX_TEXT_BYTES = 25 * 1024 * 1024

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

# -------------------------------------------------------------------------
# Do not allow this script's own output to become evidence.
# -------------------------------------------------------------------------

SELF_OUTPUT_MARKERS = (
    "outputs/revision_statistics",
    "outputs\\revision_statistics",
)

SELF_SCRIPT_NAMES = {
    "08_statistical_claim_audit.py",
}

# -------------------------------------------------------------------------
# Significance vocabulary.
# -------------------------------------------------------------------------

SIGNIFICANCE_PATTERNS = [
    r"\bsignificant\b",
    r"\bsignificantly\b",
    r"\bstatistically significant\b",
    r"\bstatistical significance\b",
    r"\bp\s*[<=>]\s*0?\.\d+",
    r"\bp[\-\s]?value\b",
    r"\bp[\-\s]?values\b",
    r"\bconfidence interval\b",
    r"\bconfidence intervals\b",
    r"\b95\s*%\s*ci\b",
    r"\b95\s*%\s*confidence\b",
]

METRIC_ALIASES = {
    "accuracy": [
        "accuracy",
        "acc",
    ],

    "precision": [
        "precision",
        "prec",
    ],

    "recall_sensitivity": [
        "recall_sensitivity",
        "recall",
        "sensitivity",
        "tpr",
    ],

    "specificity": [
        "specificity",
        "tnr",
    ],

    "f1": [
        "f1",
        "f1_score",
        "f1score",
    ],

    "roc_auc": [
        "roc_auc",
        "auc",
        "rocauc",
    ],

    "spd": [
        "spd",
        "statistical_parity_difference",
        "statistical parity difference",
    ],

    "eod": [
        "eod",
        "equal_opportunity_difference",
        "equal opportunity difference",
    ],

    "di": [
        "di",
        "disparate_impact",
        "disparate impact",
    ],
}

CONDITION_ALIASES = {
    "unbalanced": [
        "unbalanced",
        "real_training_unbalanced",
        "original",
        "without_oversampling",
        "no_oversampling",
    ],

    "oversampled": [
        "oversampled",
        "balanced",
        "smote",
        "oversampling",
        "train_only_oversampled",
    ],
}


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        )
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


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
    return "".join(
        char
        for char in str(value).strip().lower()
        if char.isalnum()
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

    df = pd.DataFrame(rows)

    if columns:
        extras = [
            column
            for column in df.columns
            if column not in columns
        ]

        df = df[
            columns + extras
        ]

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def is_excluded(path: Path) -> bool:
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

    text = str(path).lower()

    if any(
        marker.lower() in text
        for marker in SELF_OUTPUT_MARKERS
    ):
        return True

    if path.name.lower() in SELF_SCRIPT_NAMES:
        return True

    return False


def read_plain_text(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
    except Exception:
        return None

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


def read_docx_text(path: Path) -> Optional[str]:
    """
    Read DOCX text without executing macros or requiring python-docx.
    """

    try:
        with zipfile.ZipFile(
            path,
            "r",
        ) as zf:

            if "word/document.xml" not in zf.namelist():
                return None

            xml_bytes = zf.read(
                "word/document.xml"
            )

        root = ET.fromstring(
            xml_bytes
        )

        texts = []

        for element in root.iter():
            if element.tag.endswith(
                "}t"
            ):
                if element.text:
                    texts.append(
                        element.text
                    )

            elif element.tag.endswith(
                "}p"
            ):
                texts.append(
                    "\n"
                )

        return " ".join(
            texts
        )

    except Exception:
        return None


def read_text(path: Path) -> Optional[str]:
    if path.suffix.lower() == ".docx":
        return read_docx_text(
            path
        )

    return read_plain_text(
        path
    )


def line_context(
    text: str,
    line_number: int,
    radius: int = 2,
) -> str:
    lines = text.splitlines()

    start = max(
        0,
        line_number - 1 - radius,
    )

    stop = min(
        len(lines),
        line_number + radius,
    )

    return " | ".join(
        line.strip()
        for line in lines[
            start:stop
        ]
    )[:4000]


# =============================================================================
# 3. SOURCE AUDIT
# =============================================================================

def discover_text_sources() -> List[Path]:
    paths = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if is_excluded(path):
            continue

        if (
            path.suffix.lower()
            not in TEXT_EXTENSIONS
        ):
            continue

        paths.append(path)

    return sorted(
        paths,
        key=lambda p: str(p).lower(),
    )


def extract_significance_language(
    path: Path,
    text: str,
) -> List[Dict[str, Any]]:
    rows = []

    compiled = [
        re.compile(
            pattern,
            flags=re.IGNORECASE,
        )
        for pattern in SIGNIFICANCE_PATTERNS
    ]

    lines = text.splitlines()

    for line_no, line in enumerate(
        lines,
        start=1,
    ):
        matches = []

        for pattern in compiled:
            found = pattern.findall(
                line
            )

            if found:
                matches.extend(
                    [
                        str(item)
                        for item in found
                    ]
                )

        if not matches:
            continue

        context = line_context(
            text,
            line_no,
            radius=3,
        )

        rows.append(
            {
                "file":
                    relative_path(path),

                "line":
                    line_no,

                "matched_statistical_language":
                    safe_json(
                        matches
                    ),

                "text":
                    line.strip()[:3000],

                "context":
                    context,

                "contains_explicit_p_value":
                    int(
                        bool(
                            re.search(
                                r"\bp\s*[<=>]\s*0?\.\d+",
                                line,
                                flags=re.IGNORECASE,
                            )
                        )
                    ),

                "contains_significant_word":
                    int(
                        bool(
                            re.search(
                                r"\bsignificant(?:ly)?\b",
                                line,
                                flags=re.IGNORECASE,
                            )
                        )
                    ),
            }
        )

    return rows


# =============================================================================
# 4. REPEATED RESULT DISCOVERY
# =============================================================================

def read_table(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(
                path
            )

        if suffix == ".tsv":
            return pd.read_csv(
                path,
                sep="\t",
            )

    except Exception:
        return None

    return None


def discover_corrected_result_tables() -> List[Path]:
    candidates = []

    for root in CORRECTED_OUTPUT_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".csv",
                ".tsv",
            }:
                continue

            candidates.append(
                path
            )

    return sorted(
        set(
            candidates
        ),
        key=lambda p: str(p).lower(),
    )


def find_seed_column(
    df: pd.DataFrame,
) -> Optional[str]:
    normalized = {
        normalize_name(column):
            column
        for column in df.columns
    }

    candidates = [
        "seed",
        "random_seed",
        "random_state",
        "split_seed",
        "run_seed",
    ]

    for candidate in candidates:
        key = normalize_name(
            candidate
        )

        if key in normalized:
            return normalized[
                key
            ]

    return None


def find_condition_column(
    df: pd.DataFrame,
) -> Optional[str]:
    normalized = {
        normalize_name(column):
            column
        for column in df.columns
    }

    candidates = [
        "condition",
        "variant",
        "setting",
        "scenario",
        "training_condition",
        "experiment",
    ]

    for candidate in candidates:
        key = normalize_name(
            candidate
        )

        if key in normalized:
            return normalized[
                key
            ]

    return None


def canonical_metric_column(
    column: str,
) -> Optional[str]:
    key = normalize_name(
        column
    )

    for canonical, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if key == normalize_name(
                alias
            ):
                return canonical

    return None


def infer_condition_from_text(
    value: Any,
) -> str:
    lower = str(
        value
    ).strip().lower()

    for canonical, aliases in CONDITION_ALIASES.items():
        if any(
            alias in lower
            for alias in aliases
        ):
            return canonical

    return lower or "unspecified"


def extract_repeated_metrics_from_table(
    path: Path,
    df: pd.DataFrame,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    seed_col = find_seed_column(
        df
    )

    condition_col = find_condition_column(
        df
    )

    metric_columns = {}

    for column in df.columns:
        canonical = canonical_metric_column(
            str(column)
        )

        if canonical:
            metric_columns[
                column
            ] = canonical

    diagnostics = {
        "file":
            relative_path(path),

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "seed_column":
            seed_col or "",

        "condition_column":
            condition_col or "",

        "metric_columns":
            safe_json(
                metric_columns
            ),

        "eligible_repeated_metric_table":
            int(
                seed_col is not None
                and
                bool(
                    metric_columns
                )
            ),
    }

    if seed_col is None:
        return (
            [],
            diagnostics,
        )

    if not metric_columns:
        return (
            [],
            diagnostics,
        )

    rows = []

    for row_index, source_row in df.iterrows():

        seed_value = source_row[
            seed_col
        ]

        try:
            seed_value = int(
                float(
                    seed_value
                )
            )
        except Exception:
            continue

        if condition_col is not None:
            condition = infer_condition_from_text(
                source_row[
                    condition_col
                ]
            )
        else:
            condition = infer_condition_from_text(
                path.stem
            )

        for source_metric_col, metric in metric_columns.items():

            value = pd.to_numeric(
                pd.Series(
                    [
                        source_row[
                            source_metric_col
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.isna(
                value
            ):
                continue

            rows.append(
                {
                    "source_file":
                        relative_path(path),

                    "source_row":
                        int(
                            row_index
                        ),

                    "seed":
                        seed_value,

                    "condition":
                        condition,

                    "metric":
                        metric,

                    "value":
                        float(
                            value
                        ),
                }
            )

    return (
        rows,
        diagnostics,
    )


# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================

def t_like_interval(
    values: np.ndarray,
) -> Tuple[
    float,
    float,
]:
    """
    Descriptive 95% mean interval.

    Uses fixed small-sample t critical values for n <= 30 when available.
    This is reported as DESCRIPTIVE uncertainty only, not as evidence that
    repeated holdout runs are independent inferential replicates.
    """

    n = len(
        values
    )

    mean = float(
        np.mean(
            values
        )
    )

    if n <= 1:
        return (
            mean,
            mean,
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

    t_critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }

    critical = t_critical.get(
        n,
        1.96,
    )

    margin = critical * se

    return (
        mean - margin,
        mean + margin,
    )


def summarize_repeated_metrics(
    long_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not long_rows:
        return []

    df = pd.DataFrame(
        long_rows
    )

    rows = []

    grouped = df.groupby(
        [
            "source_file",
            "condition",
            "metric",
        ],
        dropna=False,
    )

    for (
        source_file,
        condition,
        metric,
    ), group in grouped:

        values = group[
            "value"
        ].astype(
            float
        ).to_numpy()

        unique_seeds = sorted(
            group[
                "seed"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        ci_low, ci_high = t_like_interval(
            values
        )

        rows.append(
            {
                "source_file":
                    source_file,

                "condition":
                    condition,

                "metric":
                    metric,

                "n_rows":
                    len(values),

                "n_unique_seeds":
                    len(
                        unique_seeds
                    ),

                "seeds":
                    safe_json(
                        unique_seeds
                    ),

                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "sd":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                    if len(
                        values
                    ) > 1
                    else 0.0,

                "median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "min":
                    float(
                        np.min(
                            values
                        )
                    ),

                "max":
                    float(
                        np.max(
                            values
                        )
                    ),

                "descriptive_95_interval_low":
                    float(
                        ci_low
                    ),

                "descriptive_95_interval_high":
                    float(
                        ci_high
                    ),

                "interval_interpretation":
                    (
                        "descriptive repeated-holdout uncertainty; "
                        "not independent-dataset inferential CI"
                    ),
            }
        )

    return rows


# =============================================================================
# 6. PAIRED CONDITION DIFFERENCES
# =============================================================================

def build_paired_differences(
    long_rows: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not long_rows:
        return (
            [],
            [],
        )

    df = pd.DataFrame(
        long_rows
    )

    paired_rows = []

    summary_rows = []

    # -----------------------------------------------------------------
    # Pair only conditions that share source file, metric, and seed.
    # -----------------------------------------------------------------

    for source_file in sorted(
        df[
            "source_file"
        ].unique()
    ):

        source_df = df[
            df[
                "source_file"
            ] == source_file
        ]

        for metric in sorted(
            source_df[
                "metric"
            ].unique()
        ):

            metric_df = source_df[
                source_df[
                    "metric"
                ] == metric
            ]

            conditions = sorted(
                metric_df[
                    "condition"
                ].unique()
            )

            if len(
                conditions
            ) < 2:
                continue

            # Prefer the known corrected comparison.
            preferred_pairs = []

            if (
                "unbalanced"
                in conditions
                and
                "oversampled"
                in conditions
            ):
                preferred_pairs.append(
                    (
                        "unbalanced",
                        "oversampled",
                    )
                )

            # Otherwise compare all unique pairs.
            if not preferred_pairs:
                for i in range(
                    len(
                        conditions
                    )
                ):
                    for j in range(
                        i + 1,
                        len(
                            conditions
                        )
                    ):
                        preferred_pairs.append(
                            (
                                conditions[i],
                                conditions[j],
                            )
                        )

            for condition_a, condition_b in preferred_pairs:

                a = metric_df[
                    metric_df[
                        "condition"
                    ] == condition_a
                ][
                    [
                        "seed",
                        "value",
                    ]
                ].rename(
                    columns={
                        "value":
                            "value_a"
                    }
                )

                b = metric_df[
                    metric_df[
                        "condition"
                    ] == condition_b
                ][
                    [
                        "seed",
                        "value",
                    ]
                ].rename(
                    columns={
                        "value":
                            "value_b"
                    }
                )

                merged = a.merge(
                    b,
                    on="seed",
                    how="inner",
                )

                if merged.empty:
                    continue

                merged[
                    "difference_b_minus_a"
                ] = (
                    merged[
                        "value_b"
                    ]
                    -
                    merged[
                        "value_a"
                    ]
                )

                for _, row in merged.iterrows():
                    paired_rows.append(
                        {
                            "source_file":
                                source_file,

                            "metric":
                                metric,

                            "condition_a":
                                condition_a,

                            "condition_b":
                                condition_b,

                            "seed":
                                int(
                                    row[
                                        "seed"
                                    ]
                                ),

                            "value_a":
                                float(
                                    row[
                                        "value_a"
                                    ]
                                ),

                            "value_b":
                                float(
                                    row[
                                        "value_b"
                                    ]
                                ),

                            "difference_b_minus_a":
                                float(
                                    row[
                                        "difference_b_minus_a"
                                    ]
                                ),
                        }
                    )

                differences = merged[
                    "difference_b_minus_a"
                ].astype(
                    float
                ).to_numpy()

                ci_low, ci_high = t_like_interval(
                    differences
                )

                summary_rows.append(
                    {
                        "source_file":
                            source_file,

                        "metric":
                            metric,

                        "condition_a":
                            condition_a,

                        "condition_b":
                            condition_b,

                        "n_paired_seeds":
                            len(
                                differences
                            ),

                        "mean_difference_b_minus_a":
                            float(
                                np.mean(
                                    differences
                                )
                            ),

                        "sd_difference":
                            float(
                                np.std(
                                    differences,
                                    ddof=1,
                                )
                            )
                            if len(
                                differences
                            ) > 1
                            else 0.0,

                        "median_difference":
                            float(
                                np.median(
                                    differences
                                )
                            ),

                        "min_difference":
                            float(
                                np.min(
                                    differences
                                )
                            ),

                        "max_difference":
                            float(
                                np.max(
                                    differences
                                )
                            ),

                        "descriptive_95_interval_low":
                            float(
                                ci_low
                            ),

                        "descriptive_95_interval_high":
                            float(
                                ci_high
                            ),

                        "formal_p_value_reported":
                            False,

                        "reason_no_formal_significance_claim":
                            (
                                "Repeated holdouts reuse the same "
                                "participant pool and are not independent "
                                "experimental replicates."
                            ),
                    }
                )

    return (
        paired_rows,
        summary_rows,
    )


# =============================================================================
# 7. INFERENTIAL ELIGIBILITY
# =============================================================================

def build_inferential_eligibility(
    repeated_summary: List[Dict[str, Any]],
    paired_summary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    max_seeds = 0

    if repeated_summary:
        max_seeds = max(
            row[
                "n_unique_seeds"
            ]
            for row in repeated_summary
        )

    rows = [
        {
            "criterion":
                "multiple_random_seeds_available",

            "status":
                (
                    "PASS"
                    if max_seeds >= 2
                    else
                    "MISSING"
                ),

            "detail":
                (
                    f"Maximum recovered repeated seeds = {max_seeds}."
                ),
        },
        {
            "criterion":
                "independent_datasets_or_external_test_replicates",

            "status":
                "MISSING",

            "detail":
                (
                    "Repeated runs reuse the same 193-participant dataset."
                ),
        },
        {
            "criterion":
                "nonoverlapping_independent_test_sets",

            "status":
                "MISSING",

            "detail":
                (
                    "Random holdout test sets overlap across seeds."
                ),
        },
        {
            "criterion":
                "predefined_inferential_unit",

            "status":
                "MISSING",

            "detail":
                (
                    "Recovered protocol does not define each random seed as "
                    "an independent scientific experimental unit."
                ),
        },
        {
            "criterion":
                "paired_seed_differences_available",

            "status":
                (
                    "PASS"
                    if bool(
                        paired_summary
                    )
                    else
                    "MISSING"
                ),

            "detail":
                (
                    f"{len(paired_summary)} paired descriptive comparison(s) "
                    "could be reconstructed."
                ),
        },
        {
            "criterion":
                "formal_significance_claim_supported",

            "status":
                "NO",

            "detail":
                (
                    "Available repeated holdouts support descriptive stability "
                    "analysis, not conventional independent-replicate "
                    "significance claims."
                ),
        },
    ]

    return rows


# =============================================================================
# 8. HISTORICAL CLAIM CLASSIFICATION
# =============================================================================

def classify_statistical_claims(
    significance_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for source in significance_rows:

        context = (
            source[
                "context"
            ]
            or ""
        ).lower()

        # -----------------------------------------------------------------
        # Revision audit outputs can mention significance in a cautionary way.
        # -----------------------------------------------------------------

        cautionary_terms = (
            "no significance",
            "not significant",
            "should not",
            "cannot claim",
            "do not claim",
            "descriptive",
            "not independent",
            "unsupported",
        )

        is_cautionary = any(
            term in context
            for term in cautionary_terms
        )

        if is_cautionary:

            classification = (
                "KEEP_AS_LIMITATION_OR_METHOD_CAUTION"
            )

            action = (
                "No change required if wording explicitly rejects "
                "unsupported significance interpretation."
            )

        elif source[
            "contains_significant_word"
        ] == 1:

            classification = (
                "REMOVE_UNSUPPORTED_SIGNIFICANCE_CLAIM"
            )

            action = (
                "Replace 'significant/significantly' with neutral descriptive "
                "language unless a valid inferential test with an appropriate "
                "independent experimental unit is documented."
            )

        elif source[
            "contains_explicit_p_value"
        ] == 1:

            classification = (
                "VERIFY_OR_REMOVE_P_VALUE"
            )

            action = (
                "Retain only if the p-value can be tied to a clearly defined "
                "test, independent unit, comparison, and raw data."
            )

        else:

            classification = (
                "VERIFY_STATISTICAL_WORDING"
            )

            action = (
                "Check whether the interval or statistical terminology is "
                "descriptive or inferential and label it explicitly."
            )

        rows.append(
            {
                **source,

                "classification":
                    classification,

                "recommended_action":
                    action,
            }
        )

    return rows


# =============================================================================
# 9. VERIFICATION MATRIX AND VERDICT
# =============================================================================

def build_verification_matrix(
    repeated_rows: List[Dict[str, Any]],
    paired_summary: List[Dict[str, Any]],
    statistical_claim_rows: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    repeated_seeds = sorted(
        {
            int(
                row[
                    "seed"
                ]
            )
            for row in repeated_rows
        }
    )

    unsupported_claim_count = sum(
        row[
            "classification"
        ]
        in {
            "REMOVE_UNSUPPORTED_SIGNIFICANCE_CLAIM",
            "VERIFY_OR_REMOVE_P_VALUE",
        }
        for row in statistical_claim_rows
    )

    matrix = [
        {
            "criterion":
                "corrected_repeated_run_values_available",

            "passed":
                int(
                    bool(
                        repeated_rows
                    )
                ),

            "detail":
                (
                    f"{len(repeated_rows)} repeated metric values recovered."
                ),
        },
        {
            "criterion":
                "multiple_corrected_seeds_available",

            "passed":
                int(
                    len(
                        repeated_seeds
                    ) >= 2
                ),

            "detail":
                (
                    f"{len(repeated_seeds)} unique seed(s): "
                    f"{repeated_seeds}"
                ),
        },
        {
            "criterion":
                "paired_condition_differences_available",

            "passed":
                int(
                    bool(
                        paired_summary
                    )
                ),

            "detail":
                (
                    f"{len(paired_summary)} paired descriptive summary row(s)."
                ),
        },
        {
            "criterion":
                "independent_experimental_replicates_available",

            "passed":
                0,

            "detail":
                (
                    "Repeated holdouts reuse the same 193 participants."
                ),
        },
        {
            "criterion":
                "formal_significance_claim_justified_by_corrected_protocol",

            "passed":
                0,

            "detail":
                (
                    "The corrected repeated-holdout protocol provides "
                    "stability evidence but not independent-replicate "
                    "inferential significance."
                ),
        },
        {
            "criterion":
                "historical_significance_language_requires_revision",

            "passed":
                int(
                    unsupported_claim_count > 0
                ),

            "detail":
                (
                    f"{unsupported_claim_count} significance/p-value "
                    "statement(s) require verification or removal."
                ),
        },
    ]

    if repeated_rows:

        verdict = (
            "DESCRIPTIVE_UNCERTAINTY_SUPPORTED_FORMAL_SIGNIFICANCE_NOT_SUPPORTED"
        )

        next_action = (
            "REPORT_REPEATED_RUN_MEAN_SD_AND_REMOVE_UNSUPPORTED_SIGNIFICANCE_WORDING"
        )

        manuscript_action = (
            "Report corrected repeated-holdout performance using descriptive "
            "mean and variability across seeds. Do not describe differences "
            "as statistically significant unless a separate valid inferential "
            "design with an appropriate independent experimental unit is "
            "available. Replace unsupported 'significant/significantly' "
            "language with descriptive wording."
        )

    else:

        verdict = (
            "NO_REPRODUCIBLE_STATISTICAL_EVIDENCE_AVAILABLE"
        )

        next_action = (
            "REMOVE_ALL_UNSUPPORTED_SIGNIFICANCE_CLAIMS"
        )

        manuscript_action = (
            "No repeated corrected run-level evidence was recovered. Remove "
            "unsupported significance claims and retain only directly "
            "reproducible point estimates."
        )

    verdict_row = {
        "verdict":
            verdict,

        "next_action":
            next_action,

        "manuscript_action":
            manuscript_action,

        "unique_corrected_seeds":
            len(
                repeated_seeds
            ),

        "repeated_metric_values":
            len(
                repeated_rows
            ),

        "paired_descriptive_comparisons":
            len(
                paired_summary
            ),

        "independent_dataset_replicates":
            0,

        "formal_significance_supported":
            False,

        "new_model_training_performed":
            False,

        "new_statistical_experiment_created":
            False,

        "new_synthetic_data_generated":
            False,
    }

    return (
        matrix,
        verdict_row,
    )


# =============================================================================
# 10. MAIN
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
        "HFAGM - STATISTICAL CLAIM / UNCERTAINTY AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "\nNo new model training or synthetic generation will be performed."
    )

    # -----------------------------------------------------------------
    # A. Search statistical language.
    # -----------------------------------------------------------------

    source_paths = discover_text_sources()

    source_inventory = []

    significance_rows = []

    for path in source_paths:

        source_inventory.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "extension":
                    path.suffix.lower(),

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

        text = read_text(
            path
        )

        if not text:
            continue

        significance_rows.extend(
            extract_significance_language(
                path,
                text,
            )
        )

    write_csv(
        OUTPUT_DIR
        / "statistical_source_inventory.csv",
        source_inventory,
    )

    write_csv(
        OUTPUT_DIR
        / "significance_language_evidence.csv",
        significance_rows,
    )

    # -----------------------------------------------------------------
    # B. Recover corrected repeated-run tables.
    # -----------------------------------------------------------------

    result_tables = discover_corrected_result_tables()

    candidate_inventory = []

    repeated_rows = []

    for path in result_tables:

        df = read_table(
            path
        )

        if df is None:
            continue

        rows, diagnostics = (
            extract_repeated_metrics_from_table(
                path,
                df,
            )
        )

        candidate_inventory.append(
            diagnostics
        )

        repeated_rows.extend(
            rows
        )

    write_csv(
        OUTPUT_DIR
        / "repeated_result_candidate_inventory.csv",
        candidate_inventory,
    )

    # -----------------------------------------------------------------
    # Remove exact duplicate extracted metric records.
    # -----------------------------------------------------------------

    if repeated_rows:

        repeated_df = pd.DataFrame(
            repeated_rows
        ).drop_duplicates(
            subset=[
                "source_file",
                "source_row",
                "seed",
                "condition",
                "metric",
                "value",
            ]
        )

        repeated_rows = repeated_df.to_dict(
            orient="records"
        )

    write_csv(
        OUTPUT_DIR
        / "repeated_metric_long.csv",
        repeated_rows,
    )

    repeated_summary = summarize_repeated_metrics(
        repeated_rows
    )

    write_csv(
        OUTPUT_DIR
        / "repeated_metric_summary.csv",
        repeated_summary,
    )

    # -----------------------------------------------------------------
    # C. Paired descriptive comparisons.
    # -----------------------------------------------------------------

    (
        paired_rows,
        paired_summary,
    ) = build_paired_differences(
        repeated_rows
    )

    write_csv(
        OUTPUT_DIR
        / "paired_condition_differences.csv",
        paired_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_difference_summary.csv",
        paired_summary,
    )

    # -----------------------------------------------------------------
    # D. Inferential eligibility.
    # -----------------------------------------------------------------

    inferential_rows = build_inferential_eligibility(
        repeated_summary,
        paired_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "inferential_test_eligibility.csv",
        inferential_rows,
    )

    # -----------------------------------------------------------------
    # E. Historical language classification.
    # -----------------------------------------------------------------

    statistical_claim_rows = classify_statistical_claims(
        significance_rows
    )

    write_csv(
        OUTPUT_DIR
        / "historical_statistical_claim_audit.csv",
        statistical_claim_rows,
    )

    # -----------------------------------------------------------------
    # F. Verification and verdict.
    # -----------------------------------------------------------------

    matrix, verdict = build_verification_matrix(
        repeated_rows,
        paired_summary,
        statistical_claim_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "statistical_verification_matrix.csv",
        matrix,
    )

    write_csv(
        OUTPUT_DIR
        / "statistical_verdict.csv",
        [
            verdict
        ],
    )

    # -----------------------------------------------------------------
    # G. Provenance.
    # -----------------------------------------------------------------

    provenance = {
        "generated":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "script":
            "08_statistical_claim_audit.py",

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "expected_original_participants":
            EXPECTED_REAL_PARTICIPANTS,

        "text_sources_scanned":
            len(
                source_inventory
            ),

        "statistical_language_rows":
            len(
                significance_rows
            ),

        "corrected_result_tables_scanned":
            len(
                result_tables
            ),

        "repeated_metric_values":
            len(
                repeated_rows
            ),

        "repeated_metric_summary_rows":
            len(
                repeated_summary
            ),

        "paired_difference_rows":
            len(
                paired_rows
            ),

        "paired_difference_summary_rows":
            len(
                paired_summary
            ),

        "verdict":
            verdict[
                "verdict"
            ],

        "new_training_performed":
            False,

        "new_synthetic_generation_performed":
            False,

        "new_inferential_experiment_created":
            False,

        "python_version":
            sys.version,

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,
    }

    write_csv(
        OUTPUT_DIR
        / "statistical_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # H. Human-readable summary.
    # -----------------------------------------------------------------

    unique_seeds = sorted(
        {
            int(
                row[
                    "seed"
                ]
            )
            for row in repeated_rows
        }
    )

    lines = [
        "=" * 100,
        "HFAGM - STATISTICAL CLAIM / UNCERTAINTY FORENSIC AUDIT",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "PURPOSE",
        "-" * 100,
        (
            "Audit whether claims of statistical significance are supported "
            "by the corrected experimental design and recoverable raw "
            "repeated-run evidence."
        ),
        "",
        "SOURCE AUDIT",
        "-" * 100,
        (
            f"Text/statistical sources scanned: "
            f"{len(source_inventory)}"
        ),
        (
            f"Statistical/significance language rows: "
            f"{len(significance_rows)}"
        ),
        (
            f"Corrected result tables scanned: "
            f"{len(result_tables)}"
        ),
        "",
        "CORRECTED REPEATED-RUN EVIDENCE",
        "-" * 100,
        (
            f"Repeated metric values recovered: "
            f"{len(repeated_rows)}"
        ),
        (
            f"Unique seeds recovered: "
            f"{len(unique_seeds)}"
        ),
        (
            f"Seeds: {unique_seeds}"
        ),
        (
            f"Paired descriptive comparison summaries: "
            f"{len(paired_summary)}"
        ),
        "",
        "STATISTICAL INTERPRETATION",
        "-" * 100,
        (
            "Repeated stratified holdouts reuse the same participant pool."
        ),
        (
            "Therefore the seed-level runs are not independent clinical "
            "datasets or independent experimental replicates."
        ),
        (
            "Mean, SD, ranges, and seed-level paired differences may be "
            "reported as descriptive stability evidence."
        ),
        (
            "Conventional claims that one condition is 'statistically "
            "significantly' better than another are not supported by this "
            "repeated-holdout design alone."
        ),
        "",
        "VERIFICATION MATRIX",
        "-" * 100,
    ]

    for row in matrix:

        state = (
            "PASS"
            if row[
                "passed"
            ] == 1
            else
            "MISSING"
        )

        lines.append(
            f"{state}: {row['criterion']}"
        )

        lines.append(
            f"    {row['detail']}"
        )

    if repeated_summary:

        lines.extend(
            [
                "",
                "RECOVERED DESCRIPTIVE SUMMARIES",
                "-" * 100,
            ]
        )

        for row in repeated_summary:

            lines.append(
                (
                    f"{row['condition']} | "
                    f"{row['metric']} | "
                    f"n seeds={row['n_unique_seeds']} | "
                    f"mean={row['mean']:.10f} | "
                    f"SD={row['sd']:.10f} | "
                    f"descriptive interval="
                    f"[{row['descriptive_95_interval_low']:.10f}, "
                    f"{row['descriptive_95_interval_high']:.10f}]"
                )
            )

    if paired_summary:

        lines.extend(
            [
                "",
                "PAIRED SEED-LEVEL DIFFERENCES",
                "-" * 100,
            ]
        )

        for row in paired_summary:

            lines.append(
                (
                    f"{row['condition_b']} - "
                    f"{row['condition_a']} | "
                    f"{row['metric']} | "
                    f"paired seeds={row['n_paired_seeds']} | "
                    f"mean difference="
                    f"{row['mean_difference_b_minus_a']:.10f}"
                )
            )

        lines.append(
            ""
        )

        lines.append(
            (
                "These are descriptive paired seed differences only; "
                "no formal significance p-value is asserted."
            )
        )

    lines.extend(
        [
            "",
            "FINAL VERDICT",
            "-" * 100,
            verdict[
                "verdict"
            ],
            "",
            "NEXT ACTION",
            "-" * 100,
            verdict[
                "next_action"
            ],
            "",
            "MANUSCRIPT ACTION",
            "-" * 100,
            verdict[
                "manuscript_action"
            ],
            "",
            "RECOMMENDED WORDING",
            "-" * 100,
            (
                "Use wording such as: 'Performance remained stable across "
                "repeated leakage-safe stratified holdouts' and report the "
                "mean ± SD."
            ),
            (
                "Avoid wording such as: 'significantly outperformed', "
                "'statistically superior', or 'significant improvement' "
                "unless supported by a separately valid inferential design."
            ),
            "",
            "IMPORTANT LIMITATION",
            "-" * 100,
            (
                "The original cohort contains 193 participants. Repeating "
                "train/test splits does not create additional independent "
                "participants."
            ),
            "",
            "SAFETY CONFIRMATION",
            "-" * 100,
            "New model training performed: NO",
            "New synthetic data generated: NO",
            "New inferential experiment created: NO",
            "Historical files modified: NO",
            "",
            "PRIMARY OUTPUTS",
            "-" * 100,
            "statistical_source_inventory.csv",
            "significance_language_evidence.csv",
            "repeated_result_candidate_inventory.csv",
            "repeated_metric_long.csv",
            "repeated_metric_summary.csv",
            "paired_condition_differences.csv",
            "paired_difference_summary.csv",
            "inferential_test_eligibility.csv",
            "historical_statistical_claim_audit.csv",
            "statistical_verification_matrix.csv",
            "statistical_verdict.csv",
            "statistical_provenance.csv",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "statistical_claim_audit_summary.txt"
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
        "08 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nStatistical language rows: "
        f"{len(significance_rows)}"
    )

    print(
        f"Repeated metric values: "
        f"{len(repeated_rows)}"
    )

    print(
        f"Unique corrected seeds: "
        f"{len(unique_seeds)}"
    )

    print(
        f"Paired descriptive comparisons: "
        f"{len(paired_summary)}"
    )

    print(
        "\nFINAL VERDICT:"
    )

    print(
        verdict[
            "verdict"
        ]
    )

    print(
        "\nNEXT ACTION:"
    )

    print(
        verdict[
            "next_action"
        ]
    )

    print(
        "\nMANUSCRIPT ACTION:"
    )

    print(
        verdict[
            "manuscript_action"
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

    for filename in [
        "statistical_claim_audit_summary.txt",
        "statistical_verdict.csv",
        "statistical_verification_matrix.csv",
        "repeated_result_candidate_inventory.csv",
        "repeated_metric_long.csv",
        "repeated_metric_summary.csv",
        "paired_condition_differences.csv",
        "paired_difference_summary.csv",
        "inferential_test_eligibility.csv",
        "historical_statistical_claim_audit.csv",
        "significance_language_evidence.csv",
        "statistical_provenance.csv",
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
            "08 FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo model was trained, no synthetic data were generated, "
            "and no historical artifact was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(1)