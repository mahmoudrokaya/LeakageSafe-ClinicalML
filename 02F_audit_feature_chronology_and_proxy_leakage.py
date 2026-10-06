"""
02F_audit_feature_chronology_and_proxy_leakage.py
=================================================

HFAGM feature chronology and outcome-proxy forensic audit.

PURPOSE
-------
02C demonstrated that the historical 160/40 evaluation split was
contaminated by exact train/test duplicates.

02D and 02E corrected the computational leakage and showed that the
classification task nevertheless remains highly separable under repeated
leakage-safe evaluation.

02F investigates a DIFFERENT possible source of optimistic performance:

    temporal / clinical outcome-proxy leakage

A predictor can be computationally leakage-free but still invalidate a
prospective prediction claim if it is:
- recorded after the prediction time point;
- derived from subsequent clinical evolution;
- measured close to death/recovery;
- generated from the outcome itself;
- a treatment/discharge/post-outcome field;
- otherwise unavailable at the intended decision point.

This script DOES NOT remove any feature automatically.

It:
1. inventories the same historical 51 predictors;
2. searches project text/code/configuration files for timing/context evidence;
3. computes descriptive feature-target associations;
4. computes raw single-feature ROC-AUC where possible;
5. performs repeated leakage-safe single-feature logistic-regression tests;
6. identifies unusually predictive individual variables;
7. classifies chronology only when supported by explicit evidence;
8. flags unresolved chronology for manual verification.

IMPORTANT
---------
High predictive strength alone is NOT evidence of leakage.

A feature is NOT called an outcome proxy merely because its AUC is high.

Chronology conclusions must be based on actual documentation/code evidence.

Unknown timing remains UNKNOWN rather than being invented.

Run under the same environment used for 02D/02E.

Recommended:
    scikit-learn 1.5.2
    numpy 1.26.4
    scipy 1.13.1
    pandas 2.2.3
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import traceback
import warnings
import zipfile

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

HISTORICAL_X_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "X_train_scaled.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
    / "feature_chronology_proxy_audit"
)

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

# -------------------------------------------------------------------------
# Evidence scanning
# -------------------------------------------------------------------------

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".csv",
    ".log",
}

DOCX_EXTENSION = ".docx"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "outputs",
}

# Exclude forensic scripts themselves to avoid self-contamination.
EXCLUDED_FILE_PREFIXES = (
    "01_",
    "02_",
    "02B_",
    "02C_",
    "02D_",
    "02E_",
    "02F_",
)

# -------------------------------------------------------------------------
# Timing vocabulary
#
# These terms DO NOT automatically prove timing. They are evidence markers
# that must be reviewed in context.
# -------------------------------------------------------------------------

BASELINE_TERMS = [
    "baseline",
    "at baseline",
    "on admission",
    "at admission",
    "upon admission",
    "initial assessment",
    "initial laboratory",
    "initial lab",
    "presentation",
    "at presentation",
    "first measurement",
    "first laboratory",
    "pretreatment",
    "pre-treatment",
]

POST_OUTCOME_TERMS = [
    "at discharge",
    "discharge",
    "after discharge",
    "post discharge",
    "post-discharge",
    "after outcome",
    "post outcome",
    "post-outcome",
    "final status",
    "final outcome",
    "after recovery",
    "after death",
    "date of death",
    "death date",
    "mortality date",
    "follow-up",
    "follow up",
    "followup",
]

LONGITUDINAL_TERMS = [
    "during hospitalization",
    "during hospitalisation",
    "hospital course",
    "serial measurement",
    "serial measurements",
    "repeated measurement",
    "repeated measurements",
    "maximum value",
    "minimum value",
    "peak value",
    "last measurement",
    "latest measurement",
    "mean during",
    "trajectory",
]

OUTCOME_DERIVED_TERMS = [
    "derived from outcome",
    "outcome-derived",
    "outcome derived",
    "survival duration",
    "time to death",
    "time-to-death",
    "days until death",
    "days to death",
    "days to recovery",
    "length until outcome",
]

# -------------------------------------------------------------------------
# Statistical proxy warning thresholds
#
# These are screening thresholds, NOT definitions of leakage.
# -------------------------------------------------------------------------

VERY_HIGH_AUC = 0.95
HIGH_AUC = 0.90

VERY_HIGH_ABS_CORRELATION = 0.80
HIGH_ABS_CORRELATION = 0.70


# =============================================================================
# 2. HELPERS
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()

    return "".join(
        ch
        for ch in text
        if ch.isalnum()
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
        f"Could not read CSV:\n{path}\n"
        + "\n".join(errors)
    )


# =============================================================================
# 3. LOAD RAW DATA
# =============================================================================

def load_raw_dataset() -> Tuple[pd.DataFrame, Path, str]:

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
        if len(df) == 193:
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
            "Raw source(s) were readable but none contained "
            "the expected 193 observations.\n"
            + details
        )

    raise FileNotFoundError(
        "No usable raw 193-row COVID clinical dataset found."
    )


# =============================================================================
# 4. TARGET
# =============================================================================

def identify_target_column(df: pd.DataFrame) -> str:

    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    for candidate in TARGET_CANDIDATES:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    raise RuntimeError(
        "Could not identify target/status column."
    )


def normalize_binary_target(
    y_raw: pd.Series,
) -> Tuple[np.ndarray, Dict[str, Any]]:

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
            numeric.unique().tolist()
        )

        if unique == [0, 1]:

            return (
                numeric.astype(int).to_numpy(),
                {
                    "type": "native_numeric_0_1",
                    "mapping": {
                        "0": 0,
                        "1": 1,
                    },
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

    unique = sorted(
        text.unique().tolist()
    )

    if len(unique) != 2:
        raise RuntimeError(
            f"Expected binary target; found {unique}"
        )

    mapping = {}

    for value in unique:

        if value in negative_terms:
            mapping[value] = 0

        elif value in positive_terms:
            mapping[value] = 1

    if len(mapping) != 2:

        raise RuntimeError(
            "Could not assign target semantics safely.\n"
            f"Observed labels: {unique}"
        )

    y = (
        text.map(mapping)
        .astype(int)
        .to_numpy()
    )

    return (
        y,
        {
            "type": "semantic_binary_mapping",
            "mapping": mapping,
        },
    )


# =============================================================================
# 5. RECOVER SAME 51 FEATURES
# =============================================================================

def load_historical_feature_names() -> List[str]:

    if not HISTORICAL_X_TRAIN_PATH.exists():

        raise FileNotFoundError(
            "Historical X_train_scaled.csv not found."
        )

    historical = read_csv_robust(
        HISTORICAL_X_TRAIN_PATH
    )

    if historical.shape[1] != 51:

        raise RuntimeError(
            f"Expected 51 historical predictors; "
            f"found {historical.shape[1]}."
        )

    return [
        str(c)
        for c in historical.columns
    ]


def map_features_to_raw(
    raw_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:

    normalized = defaultdict(list)

    for col in raw_df.columns:

        if col == target_col:
            continue

        normalized[
            normalize_name(col)
        ].append(col)

    mapped = {}
    audit = []

    for historical_feature in feature_names:

        if historical_feature in raw_df.columns:

            source_col = historical_feature
            method = "exact"

        else:

            candidates = normalized[
                normalize_name(
                    historical_feature
                )
            ]

            if len(candidates) != 1:

                raise RuntimeError(
                    "Could not uniquely map feature "
                    f"'{historical_feature}'. "
                    f"Candidates={candidates}"
                )

            source_col = candidates[0]
            method = "normalized"

        if source_col == target_col:

            raise RuntimeError(
                f"Target column mapped to predictor "
                f"{historical_feature}."
            )

        mapped[
            historical_feature
        ] = raw_df[
            source_col
        ].copy()

        audit.append(
            {
                "historical_feature":
                    historical_feature,

                "raw_source_column":
                    source_col,

                "mapping_method":
                    method,
            }
        )

    X = pd.DataFrame(mapped)

    if X.shape != (193, 51):

        raise RuntimeError(
            f"Expected X shape (193,51); found {X.shape}."
        )

    return (
        X,
        audit,
    )


# =============================================================================
# 6. NUMERIC CONVERSION
# =============================================================================

def numeric_series(series: pd.Series) -> pd.Series:

    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(
            series,
            errors="coerce",
        )

    text = (
        series
        .astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "none": np.nan,
                "NA": np.nan,
                "N/A": np.nan,
                "-": np.nan,
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


def convert_numeric(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:

    out = pd.DataFrame(
        index=X.index
    )

    audit = []

    for feature in X.columns:

        source_nonmissing = int(
            X[feature].notna().sum()
        )

        numeric = numeric_series(
            X[feature]
        )

        numeric_nonmissing = int(
            numeric.notna().sum()
        )

        failures = (
            source_nonmissing
            - numeric_nonmissing
        )

        failure_fraction = (
            failures / source_nonmissing
            if source_nonmissing
            else 0.0
        )

        if failure_fraction > 0.05:

            raise RuntimeError(
                f"Feature '{feature}' has too many "
                f"numeric conversion failures: "
                f"{failures}/{source_nonmissing}"
            )

        out[feature] = numeric

        audit.append(
            {
                "feature":
                    feature,

                "source_nonmissing":
                    source_nonmissing,

                "numeric_nonmissing":
                    numeric_nonmissing,

                "conversion_failures":
                    failures,

                "failure_fraction":
                    failure_fraction,
            }
        )

    return (
        out,
        audit,
    )


# =============================================================================
# 7. TEXT EXTRACTION
# =============================================================================

def should_scan_file(path: Path) -> bool:

    if any(
        part.lower() in EXCLUDED_DIR_NAMES
        for part in path.parts
    ):
        return False

    filename = path.name

    if filename.startswith(
        EXCLUDED_FILE_PREFIXES
    ):
        return False

    return (
        path.suffix.lower() in TEXT_EXTENSIONS
        or
        path.suffix.lower() == DOCX_EXTENSION
    )


def read_text_file(path: Path) -> Optional[str]:

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
            continue

    return None


def read_docx_text(path: Path) -> Optional[str]:
    """
    Extract plain text from DOCX without requiring python-docx.
    """

    try:

        with zipfile.ZipFile(
            path,
            "r",
        ) as z:

            if (
                "word/document.xml"
                not in z.namelist()
            ):
                return None

            xml = (
                z.read(
                    "word/document.xml"
                )
                .decode(
                    "utf-8",
                    errors="ignore",
                )
            )

            # Restore paragraph/tab separators.
            xml = re.sub(
                r"</w:p>",
                "\n",
                xml,
            )

            xml = re.sub(
                r"<w:tab[^>]*/>",
                "\t",
                xml,
            )

            text = re.sub(
                r"<[^>]+>",
                "",
                xml,
            )

            text = (
                text
                .replace(
                    "&amp;",
                    "&",
                )
                .replace(
                    "&lt;",
                    "<",
                )
                .replace(
                    "&gt;",
                    ">",
                )
            )

            return text

    except Exception:

        return None


def load_scannable_documents() -> List[Dict[str, Any]]:

    documents = []

    print(
        "\nScanning project documents/code for "
        "feature chronology evidence..."
    )

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if not should_scan_file(path):
            continue

        suffix = path.suffix.lower()

        if suffix == DOCX_EXTENSION:

            text = read_docx_text(path)

        else:

            text = read_text_file(path)

        if not text:
            continue

        documents.append(
            {
                "path":
                    path,

                "relative_path":
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "text":
                    text,
            }
        )

    print(
        f"Scannable files loaded: "
        f"{len(documents)}"
    )

    return documents


# =============================================================================
# 8. FEATURE EVIDENCE SEARCH
# =============================================================================

def feature_search_variants(
    feature: str,
) -> List[str]:

    variants = {
        feature.lower(),
        feature.lower().replace("_", " "),
        feature.lower().replace("-", " "),
    }

    aliases = {
        "age": [
            "age",
        ],

        "albumin": [
            "albumin",
        ],

        "calclum": [
            "calclum",
            "calcium",
        ],

        "calcium": [
            "calcium",
            "calclum",
        ],

        "hemo": [
            "hemo",
            "hemoglobin",
            "haemoglobin",
        ],

        "rbc": [
            "rbc",
            "red blood cell",
            "red blood cells",
        ],

        "wbc": [
            "wbc",
            "white blood cell",
            "white blood cells",
        ],

        "blood urea": [
            "blood urea",
            "urea",
        ],

        "ptt": [
            "ptt",
            "partial thromboplastin",
        ],

        "pt": [
            "pt",
            "prothrombin time",
        ],

        "vit d": [
            "vit d",
            "vitamin d",
        ],

        "vit b12": [
            "vit b12",
            "vitamin b12",
        ],
    }

    key = feature.lower().strip()

    if key in aliases:

        variants.update(
            aliases[key]
        )

    return sorted(
        variants,
        key=len,
        reverse=True,
    )


def extract_context(
    text: str,
    start: int,
    end: int,
    radius: int = 240,
) -> str:

    left = max(
        0,
        start - radius,
    )

    right = min(
        len(text),
        end + radius,
    )

    context = text[
        left:right
    ]

    context = re.sub(
        r"\s+",
        " ",
        context,
    )

    return context.strip()


def timing_markers(
    context_lower: str,
) -> Dict[str, List[str]]:

    return {
        "baseline_terms":
            [
                term
                for term in BASELINE_TERMS
                if term in context_lower
            ],

        "post_outcome_terms":
            [
                term
                for term in POST_OUTCOME_TERMS
                if term in context_lower
            ],

        "longitudinal_terms":
            [
                term
                for term in LONGITUDINAL_TERMS
                if term in context_lower
            ],

        "outcome_derived_terms":
            [
                term
                for term in OUTCOME_DERIVED_TERMS
                if term in context_lower
            ],
    }


def search_feature_evidence(
    feature_names: List[str],
    documents: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
]:

    all_rows = []
    by_feature = defaultdict(list)

    for feature in feature_names:

        variants = feature_search_variants(
            feature
        )

        for doc in documents:

            text = doc["text"]
            text_lower = text.lower()

            seen_spans = set()

            for variant in variants:

                if len(variant) <= 2:
                    # Avoid massive accidental matching for
                    # short abbreviations such as PT.
                    pattern = (
                        r"\b"
                        + re.escape(variant)
                        + r"\b"
                    )

                else:
                    pattern = re.escape(
                        variant
                    )

                for match in re.finditer(
                    pattern,
                    text_lower,
                    flags=re.IGNORECASE,
                ):

                    span = (
                        match.start(),
                        match.end(),
                    )

                    if span in seen_spans:
                        continue

                    seen_spans.add(
                        span
                    )

                    context = extract_context(
                        text,
                        match.start(),
                        match.end(),
                    )

                    markers = timing_markers(
                        context.lower()
                    )

                    row = {
                        "feature":
                            feature,

                        "matched_variant":
                            variant,

                        "file":
                            doc[
                                "relative_path"
                            ],

                        "context":
                            context,

                        "baseline_terms":
                            safe_json(
                                markers[
                                    "baseline_terms"
                                ]
                            ),

                        "post_outcome_terms":
                            safe_json(
                                markers[
                                    "post_outcome_terms"
                                ]
                            ),

                        "longitudinal_terms":
                            safe_json(
                                markers[
                                    "longitudinal_terms"
                                ]
                            ),

                        "outcome_derived_terms":
                            safe_json(
                                markers[
                                    "outcome_derived_terms"
                                ]
                            ),
                    }

                    all_rows.append(
                        row
                    )

                    by_feature[
                        feature
                    ].append(
                        row
                    )

    return (
        all_rows,
        by_feature,
    )


# =============================================================================
# 9. EVIDENCE-BASED CHRONOLOGY CLASSIFICATION
# =============================================================================

def classify_chronology_from_evidence(
    feature: str,
    evidence_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:

    baseline_hits = []
    post_hits = []
    longitudinal_hits = []
    derived_hits = []

    evidence_files = set()

    for row in evidence_rows:

        evidence_files.add(
            row["file"]
        )

        baseline = json.loads(
            row["baseline_terms"]
        )

        post = json.loads(
            row["post_outcome_terms"]
        )

        longitudinal = json.loads(
            row["longitudinal_terms"]
        )

        derived = json.loads(
            row["outcome_derived_terms"]
        )

        baseline_hits.extend(
            baseline
        )

        post_hits.extend(
            post
        )

        longitudinal_hits.extend(
            longitudinal
        )

        derived_hits.extend(
            derived
        )

    # ---------------------------------------------------------
    # Important:
    # Do NOT infer chronology from feature name alone.
    # ---------------------------------------------------------

    if derived_hits:

        classification = (
            "POTENTIAL_OUTCOME_DERIVED"
        )

        risk = "CRITICAL_REVIEW"

    elif post_hits:

        classification = (
            "POTENTIALLY_POST_OUTCOME"
        )

        risk = "HIGH_REVIEW"

    elif longitudinal_hits:

        classification = (
            "POTENTIALLY_LONGITUDINAL_OR_POST_BASELINE"
        )

        risk = "HIGH_REVIEW"

    elif baseline_hits:

        classification = (
            "BASELINE_OR_ADMISSION_EVIDENCE_FOUND"
        )

        risk = "LOWER_RISK_BUT_VERIFY"

    else:

        classification = (
            "TIMING_UNKNOWN"
        )

        risk = "REQUIRES_DOCUMENTATION"

    return {
        "feature":
            feature,

        "chronology_classification":
            classification,

        "chronology_review_level":
            risk,

        "evidence_hits":
            len(
                evidence_rows
            ),

        "evidence_file_count":
            len(
                evidence_files
            ),

        "baseline_marker_count":
            len(
                baseline_hits
            ),

        "post_outcome_marker_count":
            len(
                post_hits
            ),

        "longitudinal_marker_count":
            len(
                longitudinal_hits
            ),

        "outcome_derived_marker_count":
            len(
                derived_hits
            ),

        "evidence_files":
            safe_json(
                sorted(
                    evidence_files
                )
            ),

        "baseline_markers":
            safe_json(
                sorted(
                    set(
                        baseline_hits
                    )
                )
            ),

        "post_outcome_markers":
            safe_json(
                sorted(
                    set(
                        post_hits
                    )
                )
            ),

        "longitudinal_markers":
            safe_json(
                sorted(
                    set(
                        longitudinal_hits
                    )
                )
            ),

        "outcome_derived_markers":
            safe_json(
                sorted(
                    set(
                        derived_hits
                    )
                )
            ),
    }


# =============================================================================
# 10. DESCRIPTIVE SINGLE-FEATURE ASSOCIATION
# =============================================================================

def descriptive_feature_statistics(
    X: pd.DataFrame,
    y: np.ndarray,
) -> List[Dict[str, Any]]:

    rows = []

    for feature in X.columns:

        values = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

        valid = values.notna()

        x_valid = (
            values[
                valid
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_valid = y[
            valid.to_numpy()
        ]

        n_valid = len(
            x_valid
        )

        n_missing = int(
            len(values)
            - n_valid
        )

        correlation = np.nan
        correlation_p = np.nan

        raw_auc = np.nan
        directionless_auc = np.nan

        mean_0 = np.nan
        mean_1 = np.nan

        median_0 = np.nan
        median_1 = np.nan

        if n_valid > 2:

            if (
                np.std(
                    x_valid
                ) > 0
                and
                len(
                    np.unique(
                        y_valid
                    )
                ) == 2
            ):

                try:

                    correlation, correlation_p = (
                        stats.pointbiserialr(
                            y_valid,
                            x_valid,
                        )
                    )

                    correlation = float(
                        correlation
                    )

                    correlation_p = float(
                        correlation_p
                    )

                except Exception:
                    pass

                try:

                    raw_auc = float(
                        roc_auc_score(
                            y_valid,
                            x_valid,
                        )
                    )

                    directionless_auc = max(
                        raw_auc,
                        1.0 - raw_auc,
                    )

                except Exception:
                    pass

            class0 = x_valid[
                y_valid == 0
            ]

            class1 = x_valid[
                y_valid == 1
            ]

            if len(class0):

                mean_0 = float(
                    np.mean(
                        class0
                    )
                )

                median_0 = float(
                    np.median(
                        class0
                    )
                )

            if len(class1):

                mean_1 = float(
                    np.mean(
                        class1
                    )
                )

                median_1 = float(
                    np.median(
                        class1
                    )
                )

        abs_corr = (
            abs(
                correlation
            )
            if np.isfinite(
                correlation
            )
            else np.nan
        )

        if (
            np.isfinite(
                directionless_auc
            )
            and
            directionless_auc
            >= VERY_HIGH_AUC
        ):

            strength_flag = (
                "VERY_HIGH_SINGLE_FEATURE_SEPARATION"
            )

        elif (
            np.isfinite(
                directionless_auc
            )
            and
            directionless_auc
            >= HIGH_AUC
        ):

            strength_flag = (
                "HIGH_SINGLE_FEATURE_SEPARATION"
            )

        elif (
            np.isfinite(
                abs_corr
            )
            and
            abs_corr
            >= VERY_HIGH_ABS_CORRELATION
        ):

            strength_flag = (
                "VERY_HIGH_CORRELATION"
            )

        elif (
            np.isfinite(
                abs_corr
            )
            and
            abs_corr
            >= HIGH_ABS_CORRELATION
        ):

            strength_flag = (
                "HIGH_CORRELATION"
            )

        else:

            strength_flag = (
                "NO_EXTREME_UNIVARIATE_FLAG"
            )

        rows.append(
            {
                "feature":
                    feature,

                "n_total":
                    len(values),

                "n_valid":
                    n_valid,

                "n_missing":
                    n_missing,

                "point_biserial_r":
                    correlation,

                "point_biserial_p":
                    correlation_p,

                "abs_correlation":
                    abs_corr,

                "raw_auc_class1_high":
                    raw_auc,

                "directionless_auc":
                    directionless_auc,

                "mean_class_0":
                    mean_0,

                "mean_class_1":
                    mean_1,

                "median_class_0":
                    median_0,

                "median_class_1":
                    median_1,

                "statistical_strength_flag":
                    strength_flag,
            }
        )

    return rows


# =============================================================================
# 11. REPEATED LEAKAGE-SAFE SINGLE-FEATURE LOGISTIC REGRESSION
# =============================================================================

def run_single_feature_repeated_evaluation(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    per_seed_rows = []
    summary_rows = []

    indices = np.arange(
        len(X)
    )

    total_features = len(
        X.columns
    )

    print(
        "\nRunning repeated leakage-safe "
        "single-feature logistic evaluation..."
    )

    for feature_number, feature in enumerate(
        X.columns,
        start=1,
    ):

        print(
            f"[{feature_number:02d}/{total_features}] "
            f"{feature}"
        )

        feature_values = (
            X[
                [feature]
            ]
            .copy()
        )

        for seed in SEEDS:

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
                feature_values.iloc[
                    train_idx
                ]
            )

            X_test_raw = (
                feature_values.iloc[
                    test_idx
                ]
            )

            y_train = y[
                train_idx
            ]

            y_test = y[
                test_idx
            ]

            # -------------------------------------------------
            # Every learned preprocessing step is training only.
            # -------------------------------------------------

            imputer = SimpleImputer(
                strategy="median"
            )

            X_train = imputer.fit_transform(
                X_train_raw
            )

            X_test = imputer.transform(
                X_test_raw
            )

            scaler = StandardScaler()

            X_train = scaler.fit_transform(
                X_train
            )

            X_test = scaler.transform(
                X_test
            )

            model = LogisticRegression(
                max_iter=1000,
                random_state=seed,
            )

            model.fit(
                X_train,
                y_train,
            )

            y_pred = model.predict(
                X_test
            )

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            accuracy = float(
                accuracy_score(
                    y_test,
                    y_pred,
                )
            )

            f1 = float(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            )

            auc = float(
                roc_auc_score(
                    y_test,
                    y_prob,
                )
            )

            per_seed_rows.append(
                {
                    "feature":
                        feature,

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

                    "accuracy":
                        accuracy,

                    "f1":
                        f1,

                    "roc_auc":
                        auc,

                    "absolute_auc":
                        max(
                            auc,
                            1.0 - auc,
                        ),

                    "test_class_0":
                        int(
                            np.sum(
                                y_test == 0
                            )
                        ),

                    "test_class_1":
                        int(
                            np.sum(
                                y_test == 1
                            )
                        ),
                }
            )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    for feature in X.columns:

        subset = per_seed_df[
            per_seed_df[
                "feature"
            ]
            == feature
        ]

        row = {
            "feature":
                feature,

            "n_runs":
                len(
                    subset
                ),
        }

        for metric in [
            "accuracy",
            "f1",
            "roc_auc",
        ]:

            values = (
                subset[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )

            n = len(
                values
            )

            mean = float(
                np.mean(
                    values
                )
            )

            sd = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            se = (
                sd
                / math.sqrt(
                    n
                )
            )

            critical = stats.t.ppf(
                0.975,
                df=n - 1,
            )

            ci_lower = max(
                0.0,
                mean
                - critical
                * se,
            )

            ci_upper = min(
                1.0,
                mean
                + critical
                * se,
            )

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_sd"
            ] = sd

            row[
                f"{metric}_95ci_lower"
            ] = ci_lower

            row[
                f"{metric}_95ci_upper"
            ] = ci_upper

            row[
                f"{metric}_min"
            ] = float(
                np.min(
                    values
                )
            )

            row[
                f"{metric}_max"
            ] = float(
                np.max(
                    values
                )
            )

        mean_auc = row[
            "roc_auc_mean"
        ]

        directionless_mean_auc = max(
            mean_auc,
            1.0 - mean_auc,
        )

        row[
            "directionless_mean_auc"
        ] = (
            directionless_mean_auc
        )

        if (
            directionless_mean_auc
            >= VERY_HIGH_AUC
        ):

            row[
                "single_feature_model_flag"
            ] = (
                "VERY_HIGH_PREDICTIVE_STRENGTH"
            )

        elif (
            directionless_mean_auc
            >= HIGH_AUC
        ):

            row[
                "single_feature_model_flag"
            ] = (
                "HIGH_PREDICTIVE_STRENGTH"
            )

        else:

            row[
                "single_feature_model_flag"
            ] = (
                "NO_EXTREME_FLAG"
            )

        summary_rows.append(
            row
        )

    return (
        per_seed_rows,
        summary_rows,
    )


# =============================================================================
# 12. COMBINE CHRONOLOGY + STATISTICAL EVIDENCE
# =============================================================================

def build_combined_risk_table(
    feature_names: List[str],
    chronology_rows: List[Dict[str, Any]],
    descriptive_rows: List[Dict[str, Any]],
    repeated_summary_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    chronology = {
        row["feature"]: row
        for row in chronology_rows
    }

    descriptive = {
        row["feature"]: row
        for row in descriptive_rows
    }

    repeated = {
        row["feature"]: row
        for row in repeated_summary_rows
    }

    combined = []

    for feature in feature_names:

        c = chronology[
            feature
        ]

        d = descriptive[
            feature
        ]

        r = repeated[
            feature
        ]

        chronology_class = c[
            "chronology_classification"
        ]

        repeated_auc = r[
            "directionless_mean_auc"
        ]

        descriptive_auc = d[
            "directionless_auc"
        ]

        # ---------------------------------------------------------
        # Evidence-based review priority.
        #
        # High AUC alone does NOT produce "leakage confirmed".
        # ---------------------------------------------------------

        if chronology_class == (
            "POTENTIAL_OUTCOME_DERIVED"
        ):

            overall_priority = (
                "CRITICAL_MANUAL_REVIEW"
            )

            reason = (
                "Project evidence contains outcome-derived timing markers."
            )

        elif chronology_class in {
            "POTENTIALLY_POST_OUTCOME",
            "POTENTIALLY_LONGITUDINAL_OR_POST_BASELINE",
        }:

            overall_priority = (
                "HIGH_MANUAL_REVIEW"
            )

            reason = (
                "Project evidence suggests feature may not be strictly "
                "available at baseline."
            )

        elif (
            chronology_class
            == "TIMING_UNKNOWN"
            and
            repeated_auc
            >= VERY_HIGH_AUC
        ):

            overall_priority = (
                "HIGH_MANUAL_REVIEW"
            )

            reason = (
                "Timing is undocumented and the feature alone has "
                "very high repeated predictive strength."
            )

        elif (
            chronology_class
            == "TIMING_UNKNOWN"
            and
            repeated_auc
            >= HIGH_AUC
        ):

            overall_priority = (
                "MODERATE_MANUAL_REVIEW"
            )

            reason = (
                "Timing is undocumented and single-feature predictive "
                "strength is high."
            )

        elif chronology_class == (
            "TIMING_UNKNOWN"
        ):

            overall_priority = (
                "TIMING_DOCUMENTATION_REQUIRED"
            )

            reason = (
                "No explicit measurement-time evidence was found."
            )

        else:

            overall_priority = (
                "LOWER_PRIORITY_VERIFY"
            )

            reason = (
                "Baseline/admission timing evidence was found; "
                "manual confirmation remains appropriate."
            )

        combined.append(
            {
                "feature":
                    feature,

                "chronology_classification":
                    chronology_class,

                "chronology_review_level":
                    c[
                        "chronology_review_level"
                    ],

                "evidence_hits":
                    c[
                        "evidence_hits"
                    ],

                "evidence_files":
                    c[
                        "evidence_files"
                    ],

                "baseline_marker_count":
                    c[
                        "baseline_marker_count"
                    ],

                "post_outcome_marker_count":
                    c[
                        "post_outcome_marker_count"
                    ],

                "longitudinal_marker_count":
                    c[
                        "longitudinal_marker_count"
                    ],

                "outcome_derived_marker_count":
                    c[
                        "outcome_derived_marker_count"
                    ],

                "point_biserial_r":
                    d[
                        "point_biserial_r"
                    ],

                "directionless_raw_auc":
                    descriptive_auc,

                "repeated_single_feature_auc_mean":
                    r[
                        "roc_auc_mean"
                    ],

                "repeated_single_feature_auc_sd":
                    r[
                        "roc_auc_sd"
                    ],

                "repeated_single_feature_directionless_auc":
                    repeated_auc,

                "repeated_single_feature_accuracy_mean":
                    r[
                        "accuracy_mean"
                    ],

                "repeated_single_feature_f1_mean":
                    r[
                        "f1_mean"
                    ],

                "single_feature_model_flag":
                    r[
                        "single_feature_model_flag"
                    ],

                "overall_review_priority":
                    overall_priority,

                "review_reason":
                    reason,
            }
        )

    priority_order = {
        "CRITICAL_MANUAL_REVIEW": 0,
        "HIGH_MANUAL_REVIEW": 1,
        "MODERATE_MANUAL_REVIEW": 2,
        "TIMING_DOCUMENTATION_REQUIRED": 3,
        "LOWER_PRIORITY_VERIFY": 4,
    }

    combined.sort(
        key=lambda row: (
            priority_order.get(
                row[
                    "overall_review_priority"
                ],
                99,
            ),
            -float(
                row[
                    "repeated_single_feature_directionless_auc"
                ]
            ),
        )
    )

    return combined


# =============================================================================
# 13. MAIN
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
        "HFAGM - FEATURE CHRONOLOGY AND OUTCOME-PROXY AUDIT"
    )

    print(
        "=" * 96
    )

    print(
        f"\nscikit-learn: "
        f"{sklearn.__version__}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    # -----------------------------------------------------------------
    # Load raw source.
    # -----------------------------------------------------------------

    (
        raw_df,
        raw_path,
        source_type,
    ) = load_raw_dataset()

    print(
        f"\nRaw source: {raw_path}"
    )

    print(
        f"Source type: {source_type}"
    )

    print(
        f"Rows: {len(raw_df)}"
    )

    target_col = identify_target_column(
        raw_df
    )

    y, target_mapping = normalize_binary_target(
        raw_df[
            target_col
        ]
    )

    print(
        f"Target: {target_col}"
    )

    print(
        f"Target counts: "
        f"{dict(Counter(y.tolist()))}"
    )

    # -----------------------------------------------------------------
    # Recover same 51 features.
    # -----------------------------------------------------------------

    feature_names = load_historical_feature_names()

    (
        X_source,
        feature_mapping_rows,
    ) = map_features_to_raw(
        raw_df,
        feature_names,
        target_col,
    )

    (
        X,
        numeric_conversion_rows,
    ) = convert_numeric(
        X_source
    )

    write_csv(
        OUTPUT_DIR
        / "feature_mapping.csv",
        feature_mapping_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "numeric_conversion_audit.csv",
        numeric_conversion_rows,
    )

    # -----------------------------------------------------------------
    # Documentation/code evidence.
    # -----------------------------------------------------------------

    documents = load_scannable_documents()

    (
        feature_evidence_rows,
        evidence_by_feature,
    ) = search_feature_evidence(
        feature_names,
        documents,
    )

    write_csv(
        OUTPUT_DIR
        / "feature_timing_evidence.csv",
        feature_evidence_rows,
    )

    chronology_rows = []

    for feature in feature_names:

        chronology_rows.append(
            classify_chronology_from_evidence(
                feature,
                evidence_by_feature.get(
                    feature,
                    [],
                ),
            )
        )

    write_csv(
        OUTPUT_DIR
        / "feature_chronology_classification.csv",
        chronology_rows,
    )

    # -----------------------------------------------------------------
    # Descriptive statistical association.
    # -----------------------------------------------------------------

    print(
        "\nComputing descriptive feature-target associations..."
    )

    descriptive_rows = descriptive_feature_statistics(
        X,
        y,
    )

    descriptive_rows_sorted = sorted(
        descriptive_rows,
        key=lambda row: (
            -float(
                row["directionless_auc"]
            )
            if np.isfinite(
                row["directionless_auc"]
            )
            else 999
        ),
    )

    write_csv(
        OUTPUT_DIR
        / "univariate_descriptive_statistics.csv",
        descriptive_rows_sorted,
    )

    # -----------------------------------------------------------------
    # Repeated single-feature leakage-safe models.
    # -----------------------------------------------------------------

    (
        repeated_per_seed_rows,
        repeated_summary_rows,
    ) = run_single_feature_repeated_evaluation(
        X,
        y,
    )

    write_csv(
        OUTPUT_DIR
        / "single_feature_repeated_seed_metrics.csv",
        repeated_per_seed_rows,
    )

    repeated_summary_rows_sorted = sorted(
        repeated_summary_rows,
        key=lambda row: (
            -float(
                row[
                    "directionless_mean_auc"
                ]
            )
        ),
    )

    write_csv(
        OUTPUT_DIR
        / "single_feature_repeated_summary.csv",
        repeated_summary_rows_sorted,
    )

    # -----------------------------------------------------------------
    # Combined risk table.
    # -----------------------------------------------------------------

    combined_rows = build_combined_risk_table(
        feature_names,
        chronology_rows,
        descriptive_rows,
        repeated_summary_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "feature_proxy_review_priority.csv",
        combined_rows,
    )

    # -----------------------------------------------------------------
    # Counts / findings.
    # -----------------------------------------------------------------

    chronology_counts = Counter(
        row[
            "chronology_classification"
        ]
        for row
        in chronology_rows
    )

    review_counts = Counter(
        row[
            "overall_review_priority"
        ]
        for row
        in combined_rows
    )

    very_high_features = [
        row
        for row in combined_rows
        if (
            row[
                "repeated_single_feature_directionless_auc"
            ]
            >= VERY_HIGH_AUC
        )
    ]

    high_features = [
        row
        for row in combined_rows
        if (
            HIGH_AUC
            <= row[
                "repeated_single_feature_directionless_auc"
            ]
            < VERY_HIGH_AUC
        )
    ]

    possible_temporal_risk_features = [
        row
        for row in combined_rows
        if row[
            "chronology_classification"
        ] in {
            "POTENTIAL_OUTCOME_DERIVED",
            "POTENTIALLY_POST_OUTCOME",
            "POTENTIALLY_LONGITUDINAL_OR_POST_BASELINE",
        }
    ]

    # -----------------------------------------------------------------
    # Automated verdict.
    #
    # Conservative by design.
    # -----------------------------------------------------------------

    if any(
        row[
            "chronology_classification"
        ]
        == "POTENTIAL_OUTCOME_DERIVED"
        for row in combined_rows
    ):

        verdict = (
            "POTENTIAL_OUTCOME_PROXY_EVIDENCE_REQUIRES_MANUAL_RESOLUTION"
        )

        manuscript_use = (
            "DO_NOT_YET_PRESENT_REPEATED_CLASSIFICATION_AS_PROSPECTIVE_UTILITY"
        )

    elif possible_temporal_risk_features:

        verdict = (
            "POTENTIAL_FEATURE_TIMING_RISK_REQUIRES_MANUAL_RESOLUTION"
        )

        manuscript_use = (
            "CLASSIFICATION_RESULTS_REMAIN_PROVISIONAL_UNTIL_TIMING_IS_VERIFIED"
        )

    elif any(
        row[
            "chronology_classification"
        ]
        == "TIMING_UNKNOWN"
        and
        row[
            "repeated_single_feature_directionless_auc"
        ]
        >= VERY_HIGH_AUC
        for row in combined_rows
    ):

        verdict = (
            "NO_DIRECT_PROXY_EVIDENCE_BUT_HIGHLY_PREDICTIVE_FEATURE_TIMING_UNKNOWN"
        )

        manuscript_use = (
            "VERIFY_MEASUREMENT_TIME_BEFORE_PROSPECTIVE_UTILITY_CLAIM"
        )

    else:

        verdict = (
            "NO_DIRECT_OUTCOME_PROXY_EVIDENCE_DETECTED_WITH_AVAILABLE_PROJECT_EVIDENCE"
        )

        manuscript_use = (
            "REPEATED_CLASSIFICATION_RESULTS_CAN_BE_USED_WITH_DOCUMENTED_TIMING_LIMITATION"
        )

    # -----------------------------------------------------------------
    # Provenance.
    # -----------------------------------------------------------------

    provenance = {
        "generated":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "source":
            str(
                raw_path
            ),

        "source_type":
            source_type,

        "source_sha256":
            sha256_file(
                raw_path
            ),

        "rows":
            len(
                raw_df
            ),

        "features":
            len(
                feature_names
            ),

        "target":
            target_col,

        "target_mapping":
            safe_json(
                target_mapping
            ),

        "seeds":
            safe_json(
                SEEDS
            ),

        "test_size":
            TEST_SIZE,

        "single_feature_model":
            "LogisticRegression",

        "imputation_scope":
            "training_only",

        "scaling_scope":
            "training_only",

        "seed_selection_based_on_results":
            False,

        "feature_removal_performed":
            False,

        "chronology_inferred_from_feature_name":
            False,

        "high_auc_interpreted_as_leakage_automatically":
            False,

        "scannable_project_files":
            len(
                documents
            ),

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
        / "feature_proxy_audit_provenance.csv",
        [
            provenance
        ],
    )

    write_csv(
        OUTPUT_DIR
        / "feature_proxy_audit_verdict.csv",
        [
            {
                "verdict":
                    verdict,

                "manuscript_use":
                    manuscript_use,

                "n_potential_temporal_risk_features":
                    len(
                        possible_temporal_risk_features
                    ),

                "n_very_high_single_feature_auc":
                    len(
                        very_high_features
                    ),

                "n_high_single_feature_auc":
                    len(
                        high_features
                    ),

                "n_timing_unknown":
                    chronology_counts[
                        "TIMING_UNKNOWN"
                    ],
            }
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    lines = [
        "=" * 96,
        "HFAGM - FEATURE CHRONOLOGY AND OUTCOME-PROXY AUDIT",
        "=" * 96,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "DATA SOURCE",
        "-" * 96,
        f"Source: {raw_path}",
        f"Source type: {source_type}",
        f"Rows: {len(raw_df)}",
        f"Predictors: {len(feature_names)}",
        f"Target: {target_col}",
        f"Target counts: {dict(Counter(y.tolist()))}",
        "",
        "PURPOSE",
        "-" * 96,
        (
            "Determine whether the extremely high repeated leakage-safe "
            "classification performance might still reflect clinical "
            "feature chronology or outcome-proxy leakage."
        ),
        "",
        (
            "High feature-target association alone is NOT interpreted "
            "as evidence of leakage."
        ),
        "",
        (
            "Timing is classified only from explicit project evidence. "
            "Missing timing information remains UNKNOWN."
        ),
        "",
        "CHRONOLOGY CLASSIFICATION COUNTS",
        "-" * 96,
    ]

    for key, value in sorted(
        chronology_counts.items()
    ):

        lines.append(
            f"{key}: {value}"
        )

    lines.extend(
        [
            "",
            "REVIEW PRIORITY COUNTS",
            "-" * 96,
        ]
    )

    for key, value in sorted(
        review_counts.items()
    ):

        lines.append(
            f"{key}: {value}"
        )

    lines.extend(
        [
            "",
            "TOP SINGLE-FEATURE REPEATED PREDICTORS",
            "-" * 96,
        ]
    )

    top_features = sorted(
        combined_rows,
        key=lambda row:
            -float(
                row[
                    "repeated_single_feature_directionless_auc"
                ]
            ),
    )[:15]

    for rank, row in enumerate(
        top_features,
        start=1,
    ):

        lines.append(
            (
                f"{rank:02d}. "
                f"{row['feature']}: "
                f"mean single-feature AUC="
                f"{row['repeated_single_feature_auc_mean']:.6f}; "
                f"directionless AUC="
                f"{row['repeated_single_feature_directionless_auc']:.6f}; "
                f"mean accuracy="
                f"{row['repeated_single_feature_accuracy_mean']:.6f}; "
                f"chronology="
                f"{row['chronology_classification']}; "
                f"review="
                f"{row['overall_review_priority']}"
            )
        )

    lines.extend(
        [
            "",
            "FEATURES WITH VERY HIGH SINGLE-FEATURE SEPARATION",
            "-" * 96,
        ]
    )

    if very_high_features:

        for row in very_high_features:

            lines.append(
                (
                    f"{row['feature']}: "
                    f"directionless repeated mean AUC="
                    f"{row['repeated_single_feature_directionless_auc']:.6f}; "
                    f"chronology="
                    f"{row['chronology_classification']}; "
                    f"review="
                    f"{row['overall_review_priority']}"
                )
            )

    else:

        lines.append(
            "NONE"
        )

    lines.extend(
        [
            "",
            "FEATURES WITH DIRECT TIMING/PROXY REVIEW FLAGS",
            "-" * 96,
        ]
    )

    if possible_temporal_risk_features:

        for row in possible_temporal_risk_features:

            lines.append(
                (
                    f"{row['feature']}: "
                    f"{row['chronology_classification']}; "
                    f"evidence files="
                    f"{row['evidence_files']}"
                )
            )

    else:

        lines.append(
            "NONE DETECTED FROM AVAILABLE PROJECT TEXT."
        )

    lines.extend(
        [
            "",
            "AUTOMATED VERDICT",
            "-" * 96,
            f"Verdict: {verdict}",
            f"Manuscript use: {manuscript_use}",
            "",
            "IMPORTANT INTERPRETATION",
            "-" * 96,
            (
                "This script cannot establish clinical measurement timing "
                "when the dataset documentation does not state it."
            ),
            "",
            (
                "A laboratory variable with AUC > 0.95 is not automatically "
                "a leakage variable. It becomes a temporal leakage concern "
                "only if it was unavailable at the intended prediction time "
                "or was measured after substantial outcome evolution."
            ),
            "",
            (
                "If high-performing predictors have unknown timing, the "
                "appropriate manuscript response is to document that "
                "uncertainty and narrow prospective/predictive claims unless "
                "measurement timing can be verified from the source dataset "
                "documentation."
            ),
            "",
            "PRIMARY OUTPUTS",
            "-" * 96,
            "feature_proxy_review_priority.csv",
            "feature_chronology_classification.csv",
            "feature_timing_evidence.csv",
            "univariate_descriptive_statistics.csv",
            "single_feature_repeated_summary.csv",
            "single_feature_repeated_seed_metrics.csv",
            "feature_proxy_audit_verdict.csv",
            "feature_proxy_audit_provenance.csv",
            "",
            "=" * 96,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "feature_proxy_audit_summary.txt"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console summary.
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 96
    )

    print(
        "02F COMPLETE"
    )

    print(
        "=" * 96
    )

    print(
        f"\nVery-high single-feature "
        f"predictors (AUC >= {VERY_HIGH_AUC}): "
        f"{len(very_high_features)}"
    )

    print(
        f"High single-feature predictors "
        f"({HIGH_AUC} <= AUC < {VERY_HIGH_AUC}): "
        f"{len(high_features)}"
    )

    print(
        f"Potential temporal/proxy review features: "
        f"{len(possible_temporal_risk_features)}"
    )

    print(
        f"Features with unknown timing: "
        f"{chronology_counts['TIMING_UNKNOWN']}"
    )

    print(
        "\nAutomated verdict:"
    )

    print(
        verdict
    )

    print(
        "\nManuscript use:"
    )

    print(
        manuscript_use
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
        "feature_proxy_audit_summary.txt",
        "feature_proxy_review_priority.csv",
        "feature_chronology_classification.csv",
        "feature_timing_evidence.csv",
        "univariate_descriptive_statistics.csv",
        "single_feature_repeated_summary.csv",
        "feature_proxy_audit_verdict.csv",
        "feature_proxy_audit_provenance.csv",
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
            "02F FAILED SAFELY"
        )

        print(
            "=" * 96
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo source dataset, historical result, "
            "or classifier artifact was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(1)