"""
02F_v2_audit_feature_chronology_and_proxy_leakage.py
====================================================

HFAGM feature chronology and outcome-proxy forensic audit - V2.

WHY V2 IS REQUIRED
------------------
The original 02F statistical analysis was useful, but its chronology scanner
was contaminated because it traversed the Python virtual environment and
third-party package sources under paths such as:

    New_Code\\.venv_sklearn152\\Lib\\site-packages\\...

This produced false timing evidence for ordinary feature names such as Age.

V2 preserves the valid statistical component while making chronology evidence
much more conservative.

V2 PRINCIPLES
-------------
1. Use the raw 193-record clinical dataset.
2. Recover exactly the same historical 51 predictors.
3. Do NOT remove features.
4. Do NOT interpret high AUC alone as leakage.
5. Search chronology evidence only in genuine project/documentation sources.
6. Exclude:
       - virtual environments;
       - site-packages/dist-packages;
       - package metadata;
       - outputs/revision results;
       - ArSL bulk datasets;
       - generated audit code/results;
       - project structure dumps;
       - cache/build folders.
7. Require feature name and timing marker to occur in the same sentence or
   compact text segment.
8. Keep unsupported feature timing as UNKNOWN.
9. Run repeated leakage-safe single-feature logistic-regression evaluation
   using the same predeclared seeds used in 02E.

IMPORTANT
---------
A feature with AUC >= 0.95 is NOT automatically leakage.

Temporal leakage is only supported when project/dataset documentation shows
that the feature was unavailable at the intended prediction time, measured
after admission when the prediction target is admission-time prognosis, or
derived from subsequent/final outcome information.

Run in the same environment used for 02D/02E:
    scikit-learn 1.5.2
    numpy 1.26.4
    scipy 1.13.1
    pandas 2.2.3
"""

from __future__ import annotations

import hashlib
import html
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    / "feature_chronology_proxy_audit_v2"
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
# Statistical screening thresholds.
#
# These are review thresholds only. They are NOT leakage definitions.
# -------------------------------------------------------------------------

VERY_HIGH_AUC = 0.95
HIGH_AUC = 0.90

VERY_HIGH_ABS_CORRELATION = 0.80
HIGH_ABS_CORRELATION = 0.70


# =============================================================================
# 2. SCANNER CONFIGURATION
# =============================================================================

# -------------------------------------------------------------------------
# Allowed evidence-document extensions.
#
# CSV is deliberately excluded from chronology text scanning because large
# numeric data files generally do not document measurement timing.
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
}

DOCX_EXTENSION = ".docx"

# PDF is not parsed here because OCR / PDF extraction is outside this script.
# If project PDF documentation exists, it should be manually checked or
# extracted separately.

# -------------------------------------------------------------------------
# Directory / path exclusions.
# -------------------------------------------------------------------------

EXCLUDED_EXACT_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist-packages",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "outputs",
    "arsl",
    "arsl21l",
    "arsl_dataset",
    "images",
    "labels",
}

EXCLUDED_DIR_PREFIXES = (
    ".venv",
    "venv",
    "env",
    ".env",
)

EXCLUDED_DIR_SUFFIXES = (
    ".dist-info",
    ".egg-info",
)

# -------------------------------------------------------------------------
# File exclusions.
# -------------------------------------------------------------------------

EXCLUDED_FILE_PREFIXES = (
    "01_",
    "02_",
    "02b_",
    "02c_",
    "02d_",
    "02e_",
    "02f_",
)

EXCLUDED_FILE_NAMES = {
    "hfagm_project_structure.txt",
    "project_structure.txt",
    "audit_summary.txt",
    "audit_summary_v2.txt",
    "primary_metrics_audit_summary.txt",
    "split_leakage_audit_summary.txt",
    "corrected_evaluation_summary.txt",
    "repeated_evaluation_summary.txt",
    "feature_proxy_audit_summary.txt",
}

EXCLUDED_FILENAME_SUBSTRINGS = (
    "structure",
    "audit_summary",
    "leakage_audit",
    "repeated_evaluation",
    "corrected_evaluation",
    "feature_proxy_audit",
    "revision_primary_metrics",
)

# -------------------------------------------------------------------------
# Paths that are particularly relevant and should be retained whenever they
# survive general exclusions.
# -------------------------------------------------------------------------

HIGH_VALUE_PATH_TERMS = (
    "readme",
    "documentation",
    "docs",
    "method",
    "methods",
    "dataset",
    "covid",
    "clinical",
    "preprocess",
    "manuscript",
    "paper",
    "description",
    "procedure",
    "protocol",
    "metadata",
    "data_dictionary",
    "dictionary",
)

# -------------------------------------------------------------------------
# Timing evidence markers.
# -------------------------------------------------------------------------

BASELINE_TERMS = [
    "at admission",
    "on admission",
    "upon admission",
    "admission time",
    "admission-time",
    "baseline",
    "at baseline",
    "initial assessment",
    "initial clinical assessment",
    "initial laboratory",
    "initial laboratories",
    "initial lab",
    "first measurement",
    "first laboratory",
    "first blood test",
    "at presentation",
    "upon presentation",
    "presentation",
    "pretreatment",
    "pre-treatment",
    "before treatment",
    "before therapy",
    "before hospitalization",
]

POST_OUTCOME_TERMS = [
    "at discharge",
    "after discharge",
    "post-discharge",
    "post discharge",
    "discharge value",
    "discharge measurement",
    "final outcome",
    "final status",
    "after outcome",
    "post-outcome",
    "post outcome",
    "after recovery",
    "after death",
    "date of death",
    "death date",
    "mortality date",
    "time of death",
    "recovery date",
]

LONGITUDINAL_TERMS = [
    "during hospitalization",
    "during hospitalisation",
    "during admission",
    "hospital course",
    "throughout hospitalization",
    "throughout hospitalisation",
    "serial measurement",
    "serial measurements",
    "repeated measurement",
    "repeated measurements",
    "longitudinal measurement",
    "longitudinal measurements",
    "last measurement",
    "latest measurement",
    "final measurement",
    "peak value",
    "maximum value",
    "minimum value",
    "mean during hospitalization",
    "mean during hospitalisation",
    "trajectory",
    "follow-up measurement",
    "follow up measurement",
]

OUTCOME_DERIVED_TERMS = [
    "derived from outcome",
    "outcome-derived",
    "outcome derived",
    "time to death",
    "time-to-death",
    "days to death",
    "days until death",
    "days to recovery",
    "time to recovery",
    "survival duration",
    "survival time",
    "length until outcome",
    "days before death",
]

# -------------------------------------------------------------------------
# Sentence segmentation.
#
# Evidence is only counted when feature and timing marker are in same segment.
# -------------------------------------------------------------------------

MAX_SEGMENT_LENGTH = 600


# =============================================================================
# 3. GENERAL HELPERS
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
# 4. LOAD RAW DATA
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
            "Readable raw sources were found but none had "
            "the expected 193 observations.\n"
            + details
        )

    raise FileNotFoundError(
        "No usable raw 193-row COVID clinical dataset found."
    )


# =============================================================================
# 5. TARGET
# =============================================================================

def identify_target_column(
    df: pd.DataFrame,
) -> str:

    for candidate in TARGET_CANDIDATES:

        if candidate in df.columns:
            return candidate

    normalized = {
        normalize_name(c):
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
                    "type":
                        "native_numeric_0_1",

                    "mapping":
                        {
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
            "Could not infer target semantics safely.\n"
            f"Observed labels: {unique}"
        )

    y = (
        text
        .map(mapping)
        .astype(int)
        .to_numpy()
    )

    return (
        y,
        {
            "type":
                "semantic_binary_mapping",

            "mapping":
                mapping,
        },
    )


# =============================================================================
# 6. RECOVER SAME 51 HISTORICAL FEATURES
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
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:

    normalized = defaultdict(list)

    for col in raw_df.columns:

        if col == target_col:
            continue

        normalized[
            normalize_name(
                col
            )
        ].append(
            col
        )

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
                f"Target column mapped into predictor "
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

    X = pd.DataFrame(
        mapped
    )

    if X.shape != (193, 51):

        raise RuntimeError(
            f"Expected X shape (193, 51); "
            f"found {X.shape}."
        )

    return (
        X,
        audit,
    )


# =============================================================================
# 7. NUMERIC CONVERSION
# =============================================================================

def numeric_series(
    series: pd.Series,
) -> pd.Series:

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
        .replace(
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
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:

    out = pd.DataFrame(
        index=X.index
    )

    audit = []

    for feature in X.columns:

        source_nonmissing = int(
            X[
                feature
            ].notna().sum()
        )

        numeric = numeric_series(
            X[
                feature
            ]
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
                f"Feature '{feature}' has excessive "
                f"numeric conversion failures: "
                f"{failures}/{source_nonmissing}"
            )

        out[
            feature
        ] = numeric

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
# 8. PATH FILTERING
# =============================================================================

def is_excluded_path(
    path: Path,
) -> Tuple[bool, str]:

    try:

        relative = path.relative_to(
            PROJECT_ROOT
        )

        parts = [
            part.lower()
            for part in relative.parts
        ]

    except Exception:

        parts = [
            part.lower()
            for part in path.parts
        ]

    for part in parts:

        if part in EXCLUDED_EXACT_DIR_NAMES:

            return (
                True,
                f"excluded_dir_exact:{part}",
            )

        if any(
            part.startswith(prefix)
            for prefix in EXCLUDED_DIR_PREFIXES
        ):

            return (
                True,
                f"excluded_dir_prefix:{part}",
            )

        if any(
            part.endswith(suffix)
            for suffix in EXCLUDED_DIR_SUFFIXES
        ):

            return (
                True,
                f"excluded_dir_suffix:{part}",
            )

    filename = path.name.lower()

    if filename in EXCLUDED_FILE_NAMES:

        return (
            True,
            f"excluded_filename:{filename}",
        )

    if any(
        filename.startswith(prefix.lower())
        for prefix in EXCLUDED_FILE_PREFIXES
    ):

        return (
            True,
            f"excluded_file_prefix:{filename}",
        )

    if any(
        token.lower() in filename
        for token in EXCLUDED_FILENAME_SUBSTRINGS
    ):

        return (
            True,
            f"excluded_filename_substring:{filename}",
        )

    return (
        False,
        "",
    )


def is_supported_document(
    path: Path,
) -> bool:

    suffix = path.suffix.lower()

    return (
        suffix in TEXT_EXTENSIONS
        or
        suffix == DOCX_EXTENSION
    )


# =============================================================================
# 9. TEXT EXTRACTION
# =============================================================================

def read_text_file(
    path: Path,
) -> Optional[str]:

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


def read_docx_text(
    path: Path,
) -> Optional[str]:

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
                " ",
                xml,
            )

            text = html.unescape(
                text
            )

            text = re.sub(
                r"[ \t]+",
                " ",
                text,
            )

            return text

    except Exception:

        return None


# =============================================================================
# 10. LOAD SCANNABLE PROJECT DOCUMENTS
# =============================================================================

def calculate_document_priority(
    path: Path,
) -> str:

    relative = str(
        path.relative_to(
            PROJECT_ROOT
        )
    ).lower()

    filename = path.name.lower()

    if (
        "readme" in filename
        or
        "documentation" in relative
        or
        "docs" in path.parts
        or
        "dataset" in filename
        or
        "clinical" in filename
        or
        "covid" in filename
        or
        "procedure" in filename
        or
        "description" in filename
        or
        "metadata" in filename
    ):

        return "HIGH"

    if (
        path.suffix.lower() == ".docx"
        or
        "preprocess" in filename
        or
        "method" in filename
        or
        "manuscript" in filename
    ):

        return "MEDIUM"

    return "LOW"


def load_scannable_documents(
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    documents = []
    audit_rows = []

    print(
        "\nScanning genuine project sources for chronology evidence..."
    )

    all_files = [
        path
        for path
        in PROJECT_ROOT.rglob("*")
        if path.is_file()
    ]

    for path in all_files:

        excluded, exclusion_reason = is_excluded_path(
            path
        )

        relative = str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

        if excluded:

            audit_rows.append(
                {
                    "file":
                        relative,

                    "included":
                        0,

                    "reason":
                        exclusion_reason,

                    "priority":
                        "",
                }
            )

            continue

        if not is_supported_document(
            path
        ):

            audit_rows.append(
                {
                    "file":
                        relative,

                    "included":
                        0,

                    "reason":
                        "unsupported_extension",

                    "priority":
                        "",
                }
            )

            continue

        suffix = path.suffix.lower()

        if suffix == DOCX_EXTENSION:

            text = read_docx_text(
                path
            )

        else:

            text = read_text_file(
                path
            )

        if not text:

            audit_rows.append(
                {
                    "file":
                        relative,

                    "included":
                        0,

                    "reason":
                        "empty_or_unreadable",

                    "priority":
                        "",
                }
            )

            continue

        priority = calculate_document_priority(
            path
        )

        documents.append(
            {
                "path":
                    path,

                "relative_path":
                    relative,

                "text":
                    text,

                "priority":
                    priority,
            }
        )

        audit_rows.append(
            {
                "file":
                    relative,

                "included":
                    1,

                "reason":
                    "included",

                "priority":
                    priority,

                "characters":
                    len(
                        text
                    ),
            }
        )

    print(
        f"Included scannable project files: "
        f"{len(documents)}"
    )

    excluded_count = sum(
        1
        for row
        in audit_rows
        if row[
            "included"
        ] == 0
    )

    print(
        f"Excluded/non-scannable files: "
        f"{excluded_count}"
    )

    return (
        documents,
        audit_rows,
    )


# =============================================================================
# 11. SEGMENT TEXT
# =============================================================================

def segment_text(
    text: str,
) -> List[str]:

    # Normalize whitespace while keeping sentence boundaries.
    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Split on:
    # - sentence punctuation;
    # - paragraph/newline boundaries;
    # - semicolon where it helps separate independent statements.
    raw_segments = re.split(
        r"(?<=[.!?])\s+|\n+|;\s+",
        text,
    )

    segments = []

    for segment in raw_segments:

        segment = segment.strip()

        if not segment:
            continue

        if len(segment) > MAX_SEGMENT_LENGTH:

            # Split unusually long code/doc segments conservatively.
            subsegments = re.split(
                r",\s+|\s{2,}|:\s+",
                segment,
            )

            for sub in subsegments:

                sub = sub.strip()

                if (
                    sub
                    and
                    len(sub)
                    <= MAX_SEGMENT_LENGTH
                ):

                    segments.append(
                        sub
                    )

        else:

            segments.append(
                segment
            )

    return segments


# =============================================================================
# 12. FEATURE SEARCH VARIANTS
# =============================================================================

def feature_search_variants(
    feature: str,
) -> List[str]:

    feature_lower = feature.lower().strip()

    variants = {
        feature_lower,
        feature_lower.replace(
            "_",
            " ",
        ),
        feature_lower.replace(
            "-",
            " ",
        ),
    }

    aliases = {
        "age": [
            "age",
            "patient age",
            "participants' age",
            "participant age",
        ],

        "albumin": [
            "albumin",
            "serum albumin",
        ],

        "calclum": [
            "calclum",
            "calcium",
            "serum calcium",
        ],

        "calcium": [
            "calcium",
            "calclum",
            "serum calcium",
        ],

        "hemo": [
            "hemo",
            "hemoglobin",
            "haemoglobin",
        ],

        "hema": [
            "hema",
            "hematocrit",
            "haematocrit",
        ],

        "rbc": [
            "rbc",
            "red blood cell",
            "red blood cells",
            "red blood cell count",
        ],

        "wbc": [
            "wbc",
            "white blood cell",
            "white blood cells",
            "white blood cell count",
        ],

        "blood urea": [
            "blood urea",
            "urea",
            "serum urea",
        ],

        "pt": [
            "pt",
            "prothrombin time",
        ],

        "ptt": [
            "ptt",
            "partial thromboplastin time",
            "activated partial thromboplastin time",
        ],

        "inr": [
            "inr",
            "international normalized ratio",
        ],

        "crp": [
            "crp",
            "c-reactive protein",
            "c reactive protein",
        ],

        "ldh": [
            "ldh",
            "lactate dehydrogenase",
        ],

        "glucose": [
            "glucose",
            "blood glucose",
            "serum glucose",
        ],
    }

    if feature_lower in aliases:

        variants.update(
            aliases[
                feature_lower
            ]
        )

    return sorted(
        {
            variant.strip()
            for variant
            in variants
            if variant.strip()
        },
        key=len,
        reverse=True,
    )


# =============================================================================
# 13. ROBUST FEATURE MATCHING
# =============================================================================

def variant_matches_segment(
    variant: str,
    segment_lower: str,
) -> bool:

    variant = variant.lower().strip()

    if not variant:
        return False

    # Short abbreviations require strict boundaries.
    if len(variant) <= 3:

        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(
                variant
            )
            + r"(?![A-Za-z0-9])"
        )

        return (
            re.search(
                pattern,
                segment_lower,
                flags=re.IGNORECASE,
            )
            is not None
        )

    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(
            variant
        )
        + r"(?![A-Za-z0-9])"
    )

    return (
        re.search(
            pattern,
            segment_lower,
            flags=re.IGNORECASE,
        )
        is not None
    )


# =============================================================================
# 14. TIMING MARKERS
# =============================================================================

def find_terms(
    segment_lower: str,
    terms: Iterable[str],
) -> List[str]:

    hits = []

    for term in terms:

        if term in segment_lower:

            hits.append(
                term
            )

    return sorted(
        set(
            hits
        )
    )


def timing_markers_in_segment(
    segment_lower: str,
) -> Dict[str, List[str]]:

    return {
        "baseline_terms":
            find_terms(
                segment_lower,
                BASELINE_TERMS,
            ),

        "post_outcome_terms":
            find_terms(
                segment_lower,
                POST_OUTCOME_TERMS,
            ),

        "longitudinal_terms":
            find_terms(
                segment_lower,
                LONGITUDINAL_TERMS,
            ),

        "outcome_derived_terms":
            find_terms(
                segment_lower,
                OUTCOME_DERIVED_TERMS,
            ),
    }


# =============================================================================
# 15. SEARCH FEATURE TIMING EVIDENCE
# =============================================================================

def search_feature_timing_evidence(
    feature_names: List[str],
    documents: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
]:

    evidence_rows = []
    by_feature = defaultdict(
        list
    )

    print(
        "\nSearching for SAME-SEGMENT feature/timing evidence..."
    )

    prepared_documents = []

    for doc in documents:

        segments = segment_text(
            doc[
                "text"
            ]
        )

        prepared_documents.append(
            {
                **doc,
                "segments":
                    segments,
            }
        )

    for feature_index, feature in enumerate(
        feature_names,
        start=1,
    ):

        variants = feature_search_variants(
            feature
        )

        feature_evidence_count = 0

        for doc in prepared_documents:

            for segment_number, segment in enumerate(
                doc[
                    "segments"
                ],
                start=1,
            ):

                segment_lower = segment.lower()

                matched_variants = [
                    variant
                    for variant
                    in variants
                    if variant_matches_segment(
                        variant,
                        segment_lower,
                    )
                ]

                if not matched_variants:
                    continue

                markers = timing_markers_in_segment(
                    segment_lower
                )

                has_timing_marker = any(
                    markers[
                        key
                    ]
                    for key
                    in [
                        "baseline_terms",
                        "post_outcome_terms",
                        "longitudinal_terms",
                        "outcome_derived_terms",
                    ]
                )

                # Only retain chronology evidence when timing evidence exists
                # in the same sentence/segment.
                if not has_timing_marker:
                    continue

                row = {
                    "feature":
                        feature,

                    "matched_variants":
                        safe_json(
                            matched_variants
                        ),

                    "file":
                        doc[
                            "relative_path"
                        ],

                    "document_priority":
                        doc[
                            "priority"
                        ],

                    "segment_number":
                        segment_number,

                    "segment_text":
                        segment,

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

                evidence_rows.append(
                    row
                )

                by_feature[
                    feature
                ].append(
                    row
                )

                feature_evidence_count += 1

        print(
            f"[{feature_index:02d}/{len(feature_names)}] "
            f"{feature}: "
            f"{feature_evidence_count} chronology-evidence segments"
        )

    return (
        evidence_rows,
        by_feature,
    )


# =============================================================================
# 16. EVIDENCE QUALITY
# =============================================================================

def priority_weight(
    priority: str,
) -> int:

    mapping = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    return mapping.get(
        priority,
        0,
    )


def evidence_strength(
    rows: List[Dict[str, Any]],
    marker_key: str,
) -> Tuple[
    int,
    int,
    List[str],
]:

    count = 0
    weighted = 0
    files = set()

    for row in rows:

        terms = json.loads(
            row[
                marker_key
            ]
        )

        if not terms:
            continue

        count += 1

        weighted += priority_weight(
            row[
                "document_priority"
            ]
        )

        files.add(
            row[
                "file"
            ]
        )

    return (
        count,
        weighted,
        sorted(
            files
        ),
    )


# =============================================================================
# 17. CLASSIFY FEATURE CHRONOLOGY
# =============================================================================

def classify_chronology(
    feature: str,
    evidence_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:

    (
        baseline_count,
        baseline_weight,
        baseline_files,
    ) = evidence_strength(
        evidence_rows,
        "baseline_terms",
    )

    (
        post_count,
        post_weight,
        post_files,
    ) = evidence_strength(
        evidence_rows,
        "post_outcome_terms",
    )

    (
        longitudinal_count,
        longitudinal_weight,
        longitudinal_files,
    ) = evidence_strength(
        evidence_rows,
        "longitudinal_terms",
    )

    (
        derived_count,
        derived_weight,
        derived_files,
    ) = evidence_strength(
        evidence_rows,
        "outcome_derived_terms",
    )

    # -----------------------------------------------------------------
    # Conservative evidence logic.
    # -----------------------------------------------------------------

    conflicting = (
        baseline_count > 0
        and
        (
            post_count > 0
            or
            longitudinal_count > 0
            or
            derived_count > 0
        )
    )

    if derived_count > 0:

        classification = (
            "POTENTIAL_OUTCOME_DERIVED"
        )

        review_level = (
            "CRITICAL_MANUAL_REVIEW"
        )

    elif post_count > 0:

        classification = (
            "POTENTIALLY_POST_OUTCOME"
        )

        review_level = (
            "HIGH_MANUAL_REVIEW"
        )

    elif longitudinal_count > 0:

        classification = (
            "POTENTIALLY_LONGITUDINAL_OR_POST_BASELINE"
        )

        review_level = (
            "HIGH_MANUAL_REVIEW"
        )

    elif baseline_count > 0:

        classification = (
            "BASELINE_OR_ADMISSION_EVIDENCE_FOUND"
        )

        review_level = (
            "LOWER_RISK_BUT_VERIFY"
        )

    else:

        classification = (
            "TIMING_UNKNOWN"
        )

        review_level = (
            "REQUIRES_DOCUMENTATION"
        )

    if conflicting:

        classification = (
            "CONFLICTING_TIMING_EVIDENCE"
        )

        review_level = (
            "HIGH_MANUAL_REVIEW"
        )

    evidence_files = sorted(
        {
            row[
                "file"
            ]
            for row
            in evidence_rows
        }
    )

    return {
        "feature":
            feature,

        "chronology_classification":
            classification,

        "chronology_review_level":
            review_level,

        "evidence_segments":
            len(
                evidence_rows
            ),

        "evidence_file_count":
            len(
                evidence_files
            ),

        "baseline_segment_count":
            baseline_count,

        "baseline_weight":
            baseline_weight,

        "post_outcome_segment_count":
            post_count,

        "post_outcome_weight":
            post_weight,

        "longitudinal_segment_count":
            longitudinal_count,

        "longitudinal_weight":
            longitudinal_weight,

        "outcome_derived_segment_count":
            derived_count,

        "outcome_derived_weight":
            derived_weight,

        "conflicting_timing_evidence":
            int(
                conflicting
            ),

        "evidence_files":
            safe_json(
                evidence_files
            ),

        "baseline_evidence_files":
            safe_json(
                baseline_files
            ),

        "post_outcome_evidence_files":
            safe_json(
                post_files
            ),

        "longitudinal_evidence_files":
            safe_json(
                longitudinal_files
            ),

        "outcome_derived_evidence_files":
            safe_json(
                derived_files
            ),
    }


# =============================================================================
# 18. DESCRIPTIVE FEATURE STATISTICS
# =============================================================================

def descriptive_feature_statistics(
    X: pd.DataFrame,
    y: np.ndarray,
) -> List[Dict[str, Any]]:

    rows = []

    for feature in X.columns:

        values = pd.to_numeric(
            X[
                feature
            ],
            errors="coerce",
        )

        valid_mask = values.notna()

        x_valid = (
            values[
                valid_mask
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_valid = y[
            valid_mask.to_numpy()
        ]

        n_valid = len(
            x_valid
        )

        n_missing = int(
            len(
                values
            )
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

        if (
            n_valid > 2
            and
            len(
                np.unique(
                    y_valid
                )
            ) == 2
        ):

            if np.std(
                x_valid
            ) > 0:

                try:

                    (
                        correlation,
                        correlation_p,
                    ) = stats.pointbiserialr(
                        y_valid,
                        x_valid,
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

            if len(
                class0
            ):

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

            if len(
                class1
            ):

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
            directionless_auc >= VERY_HIGH_AUC
        ):

            strength_flag = (
                "VERY_HIGH_SINGLE_FEATURE_SEPARATION"
            )

        elif (
            np.isfinite(
                directionless_auc
            )
            and
            directionless_auc >= HIGH_AUC
        ):

            strength_flag = (
                "HIGH_SINGLE_FEATURE_SEPARATION"
            )

        elif (
            np.isfinite(
                abs_corr
            )
            and
            abs_corr >= VERY_HIGH_ABS_CORRELATION
        ):

            strength_flag = (
                "VERY_HIGH_CORRELATION"
            )

        elif (
            np.isfinite(
                abs_corr
            )
            and
            abs_corr >= HIGH_ABS_CORRELATION
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
                    len(
                        values
                    ),

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
# 19. REPEATED LEAKAGE-SAFE SINGLE-FEATURE EVALUATION
# =============================================================================

def repeated_single_feature_evaluation(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    per_seed_rows = []

    indices = np.arange(
        len(
            X
        )
    )

    print(
        "\nRunning repeated leakage-safe single-feature models..."
    )

    for feature_number, feature in enumerate(
        X.columns,
        start=1,
    ):

        print(
            f"[{feature_number:02d}/{len(X.columns)}] "
            f"{feature}"
        )

        feature_df = X[
            [
                feature
            ]
        ].copy()

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

            X_train_raw = feature_df.iloc[
                train_idx
            ]

            X_test_raw = feature_df.iloc[
                test_idx
            ]

            y_train = y[
                train_idx
            ]

            y_test = y[
                test_idx
            ]

            # ---------------------------------------------------------
            # Training-only preprocessing.
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

                    "directionless_auc":
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

    summary_rows = []

    for feature in X.columns:

        subset = per_seed_df[
            per_seed_df[
                "feature"
            ] == feature
        ]

        summary = {
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

            se = sd / math.sqrt(
                n
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

            summary[
                f"{metric}_mean"
            ] = mean

            summary[
                f"{metric}_sd"
            ] = sd

            summary[
                f"{metric}_95ci_lower"
            ] = ci_lower

            summary[
                f"{metric}_95ci_upper"
            ] = ci_upper

            summary[
                f"{metric}_min"
            ] = float(
                np.min(
                    values
                )
            )

            summary[
                f"{metric}_max"
            ] = float(
                np.max(
                    values
                )
            )

        mean_auc = summary[
            "roc_auc_mean"
        ]

        directionless_mean_auc = max(
            mean_auc,
            1.0 - mean_auc,
        )

        summary[
            "directionless_mean_auc"
        ] = (
            directionless_mean_auc
        )

        if (
            directionless_mean_auc
            >= VERY_HIGH_AUC
        ):

            summary[
                "single_feature_model_flag"
            ] = (
                "VERY_HIGH_PREDICTIVE_STRENGTH"
            )

        elif (
            directionless_mean_auc
            >= HIGH_AUC
        ):

            summary[
                "single_feature_model_flag"
            ] = (
                "HIGH_PREDICTIVE_STRENGTH"
            )

        else:

            summary[
                "single_feature_model_flag"
            ] = (
                "NO_EXTREME_FLAG"
            )

        summary_rows.append(
            summary
        )

    return (
        per_seed_rows,
        summary_rows,
    )


# =============================================================================
# 20. COMBINED REVIEW TABLE
# =============================================================================

def build_combined_review_table(
    feature_names: List[str],
    chronology_rows: List[Dict[str, Any]],
    descriptive_rows: List[Dict[str, Any]],
    repeated_summary_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    chronology_map = {
        row[
            "feature"
        ]:
            row

        for row
        in chronology_rows
    }

    descriptive_map = {
        row[
            "feature"
        ]:
            row

        for row
        in descriptive_rows
    }

    repeated_map = {
        row[
            "feature"
        ]:
            row

        for row
        in repeated_summary_rows
    }

    combined = []

    for feature in feature_names:

        c = chronology_map[
            feature
        ]

        d = descriptive_map[
            feature
        ]

        r = repeated_map[
            feature
        ]

        chronology_class = c[
            "chronology_classification"
        ]

        repeated_auc = r[
            "directionless_mean_auc"
        ]

        # ---------------------------------------------------------
        # Manual review priority.
        #
        # Statistical strength alone never confirms leakage.
        # ---------------------------------------------------------

        if chronology_class == (
            "POTENTIAL_OUTCOME_DERIVED"
        ):

            review_priority = (
                "CRITICAL_MANUAL_REVIEW"
            )

            review_reason = (
                "Project documentation contains same-segment "
                "outcome-derived timing evidence."
            )

        elif chronology_class in {
            "POTENTIALLY_POST_OUTCOME",
            "POTENTIALLY_LONGITUDINAL_OR_POST_BASELINE",
            "CONFLICTING_TIMING_EVIDENCE",
        }:

            review_priority = (
                "HIGH_MANUAL_REVIEW"
            )

            review_reason = (
                "Project documentation contains timing evidence "
                "that may place this feature after baseline, or "
                "the available timing evidence conflicts."
            )

        elif (
            chronology_class
            == "TIMING_UNKNOWN"
            and
            repeated_auc
            >= VERY_HIGH_AUC
        ):

            review_priority = (
                "HIGH_MANUAL_REVIEW"
            )

            review_reason = (
                "Feature timing is undocumented and the feature "
                "alone has very high leakage-safe predictive strength."
            )

        elif (
            chronology_class
            == "TIMING_UNKNOWN"
            and
            repeated_auc
            >= HIGH_AUC
        ):

            review_priority = (
                "MODERATE_MANUAL_REVIEW"
            )

            review_reason = (
                "Feature timing is undocumented and single-feature "
                "predictive strength is high."
            )

        elif chronology_class == (
            "TIMING_UNKNOWN"
        ):

            review_priority = (
                "TIMING_DOCUMENTATION_REQUIRED"
            )

            review_reason = (
                "No explicit timing evidence was found in genuine "
                "project/documentation sources."
            )

        else:

            review_priority = (
                "LOWER_PRIORITY_VERIFY"
            )

            review_reason = (
                "Baseline/admission evidence was found, but its "
                "clinical interpretation should still be verified."
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

                "evidence_segments":
                    c[
                        "evidence_segments"
                    ],

                "evidence_file_count":
                    c[
                        "evidence_file_count"
                    ],

                "baseline_segment_count":
                    c[
                        "baseline_segment_count"
                    ],

                "post_outcome_segment_count":
                    c[
                        "post_outcome_segment_count"
                    ],

                "longitudinal_segment_count":
                    c[
                        "longitudinal_segment_count"
                    ],

                "outcome_derived_segment_count":
                    c[
                        "outcome_derived_segment_count"
                    ],

                "conflicting_timing_evidence":
                    c[
                        "conflicting_timing_evidence"
                    ],

                "evidence_files":
                    c[
                        "evidence_files"
                    ],

                "point_biserial_r":
                    d[
                        "point_biserial_r"
                    ],

                "directionless_raw_auc":
                    d[
                        "directionless_auc"
                    ],

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
                    review_priority,

                "review_reason":
                    review_reason,
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
# 21. MAIN
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
        "HFAGM - FEATURE CHRONOLOGY AND OUTCOME-PROXY AUDIT V2"
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

    print(
        f"Seeds: {SEEDS}"
    )

    # -----------------------------------------------------------------
    # Raw source.
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

    (
        y,
        target_mapping,
    ) = normalize_binary_target(
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
    # Same historical 51 predictors.
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

    if X.shape != (
        193,
        51,
    ):

        raise RuntimeError(
            f"Expected X shape (193,51); "
            f"found {X.shape}."
        )

    write_csv(
        OUTPUT_DIR
        / "feature_mapping_v2.csv",
        feature_mapping_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "numeric_conversion_audit_v2.csv",
        numeric_conversion_rows,
    )

    # -----------------------------------------------------------------
    # Genuine-project document scanner.
    # -----------------------------------------------------------------

    (
        documents,
        scanner_audit_rows,
    ) = load_scannable_documents()

    write_csv(
        OUTPUT_DIR
        / "scanner_file_inclusion_audit.csv",
        scanner_audit_rows,
    )

    (
        timing_evidence_rows,
        timing_evidence_by_feature,
    ) = search_feature_timing_evidence(
        feature_names,
        documents,
    )

    write_csv(
        OUTPUT_DIR
        / "feature_timing_evidence_v2.csv",
        timing_evidence_rows,
    )

    chronology_rows = []

    for feature in feature_names:

        chronology_rows.append(
            classify_chronology(
                feature,
                timing_evidence_by_feature.get(
                    feature,
                    [],
                ),
            )
        )

    write_csv(
        OUTPUT_DIR
        / "feature_chronology_classification_v2.csv",
        chronology_rows,
    )

    # -----------------------------------------------------------------
    # Descriptive statistics.
    # -----------------------------------------------------------------

    print(
        "\nComputing univariate descriptive statistics..."
    )

    descriptive_rows = descriptive_feature_statistics(
        X,
        y,
    )

    descriptive_rows_sorted = sorted(
        descriptive_rows,
        key=lambda row: (
            -float(
                row[
                    "directionless_auc"
                ]
            )
            if np.isfinite(
                row[
                    "directionless_auc"
                ]
            )
            else 999
        ),
    )

    write_csv(
        OUTPUT_DIR
        / "univariate_descriptive_statistics_v2.csv",
        descriptive_rows_sorted,
    )

    # -----------------------------------------------------------------
    # Repeated single-feature leakage-safe evaluation.
    # -----------------------------------------------------------------

    (
        repeated_seed_rows,
        repeated_summary_rows,
    ) = repeated_single_feature_evaluation(
        X,
        y,
    )

    write_csv(
        OUTPUT_DIR
        / "single_feature_repeated_seed_metrics_v2.csv",
        repeated_seed_rows,
    )

    repeated_summary_rows_sorted = sorted(
        repeated_summary_rows,
        key=lambda row:
            -float(
                row[
                    "directionless_mean_auc"
                ]
            ),
    )

    write_csv(
        OUTPUT_DIR
        / "single_feature_repeated_summary_v2.csv",
        repeated_summary_rows_sorted,
    )

    # -----------------------------------------------------------------
    # Combined review table.
    # -----------------------------------------------------------------

    combined_rows = build_combined_review_table(
        feature_names,
        chronology_rows,
        descriptive_rows,
        repeated_summary_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "feature_proxy_review_priority_v2.csv",
        combined_rows,
    )

    # -----------------------------------------------------------------
    # Summary counts.
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
        for row
        in combined_rows
        if (
            row[
                "repeated_single_feature_directionless_auc"
            ]
            >= VERY_HIGH_AUC
        )
    ]

    high_features = [
        row
        for row
        in combined_rows
        if (
            HIGH_AUC
            <= row[
                "repeated_single_feature_directionless_auc"
            ]
            < VERY_HIGH_AUC
        )
    ]

    direct_temporal_flags = [
        row
        for row
        in combined_rows
        if row[
            "chronology_classification"
        ] in {
            "POTENTIAL_OUTCOME_DERIVED",
            "POTENTIALLY_POST_OUTCOME",
            "POTENTIALLY_LONGITUDINAL_OR_POST_BASELINE",
            "CONFLICTING_TIMING_EVIDENCE",
        }
    ]

    unknown_high_auc_features = [
        row
        for row
        in combined_rows
        if (
            row[
                "chronology_classification"
            ] == "TIMING_UNKNOWN"
            and
            row[
                "repeated_single_feature_directionless_auc"
            ] >= HIGH_AUC
        )
    ]

    # -----------------------------------------------------------------
    # Automated verdict.
    #
    # Conservative and evidence-based.
    # -----------------------------------------------------------------

    has_outcome_derived = any(
        row[
            "chronology_classification"
        ]
        == "POTENTIAL_OUTCOME_DERIVED"

        for row
        in combined_rows
    )

    has_post_or_longitudinal = any(
        row[
            "chronology_classification"
        ] in {
            "POTENTIALLY_POST_OUTCOME",
            "POTENTIALLY_LONGITUDINAL_OR_POST_BASELINE",
            "CONFLICTING_TIMING_EVIDENCE",
        }

        for row
        in combined_rows
    )

    if has_outcome_derived:

        verdict = (
            "POTENTIAL_OUTCOME_DERIVED_FEATURE_EVIDENCE_FOUND"
        )

        manuscript_use = (
            "DO_NOT_PRESENT_AS_PROSPECTIVE_PREDICTIVE_UTILITY_UNTIL_MANUALLY_RESOLVED"
        )

    elif has_post_or_longitudinal:

        verdict = (
            "POTENTIAL_TEMPORAL_FEATURE_RISK_FOUND_IN_PROJECT_DOCUMENTATION"
        )

        manuscript_use = (
            "VERIFY_FLAGGED_FEATURE_TIMING_BEFORE_PROSPECTIVE_UTILITY_CLAIMS"
        )

    elif unknown_high_auc_features:

        verdict = (
            "NO_DIRECT_TEMPORAL_LEAKAGE_EVIDENCE_BUT_HIGHLY_PREDICTIVE_FEATURE_TIMING_REMAINS_UNDOCUMENTED"
        )

        manuscript_use = (
            "HIGH_CLASSIFICATION_RESULTS_MAY_BE_REPORTED_AS_INTERNAL_ASSOCIATION_WITH_TIMING_LIMITATION"
        )

    else:

        verdict = (
            "NO_DIRECT_OUTCOME_PROXY_OR_POST_OUTCOME_EVIDENCE_FOUND_IN_AVAILABLE_PROJECT_DOCUMENTATION"
        )

        manuscript_use = (
            "REPEATED_CLASSIFICATION_RESULTS_CAN_BE_USED_WITH_APPROPRIATE_INTERNAL_VALIDATION_LIMITATIONS"
        )

    verdict_row = {
        "verdict":
            verdict,

        "manuscript_use":
            manuscript_use,

        "n_features":
            len(
                feature_names
            ),

        "n_direct_temporal_flags":
            len(
                direct_temporal_flags
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

        "n_unknown_high_auc_features":
            len(
                unknown_high_auc_features
            ),

        "n_scannable_project_files":
            len(
                documents
            ),
    }

    write_csv(
        OUTPUT_DIR
        / "feature_proxy_audit_verdict_v2.csv",
        [
            verdict_row
        ],
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
            "02F_v2_audit_feature_chronology_and_proxy_leakage.py",

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

        "single_feature_imputation":
            "training_only_median",

        "single_feature_scaling":
            "training_only_standard_scaler",

        "feature_removal_performed":
            False,

        "high_auc_automatically_interpreted_as_leakage":
            False,

        "chronology_inferred_from_feature_name_alone":
            False,

        "chronology_requires_same_segment_timing_marker":
            True,

        "virtual_environments_excluded":
            True,

        "site_packages_excluded":
            True,

        "outputs_excluded":
            True,

        "arsl_bulk_data_excluded":
            True,

        "audit_scripts_excluded":
            True,

        "csv_files_scanned_for_chronology":
            False,

        "scannable_project_files":
            len(
                documents
            ),

        "timing_evidence_segments":
            len(
                timing_evidence_rows
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
        / "feature_proxy_audit_provenance_v2.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    lines = [
        "=" * 100,
        "HFAGM - FEATURE CHRONOLOGY AND OUTCOME-PROXY AUDIT V2",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "WHY V2 WAS REQUIRED",
        "-" * 100,
        (
            "The original 02F chronology scanner traversed virtual-environment "
            "and third-party package files, contaminating chronology evidence."
        ),
        (
            "V2 excludes virtual environments, site-packages, package metadata, "
            "outputs, ArSL bulk data, and audit scripts."
        ),
        (
            "Feature and timing marker must occur in the SAME sentence/compact "
            "text segment before chronology evidence is recorded."
        ),
        "",
        "DATA SOURCE",
        "-" * 100,
        f"Source: {raw_path}",
        f"Source type: {source_type}",
        f"Rows: {len(raw_df)}",
        f"Predictors: {len(feature_names)}",
        f"Target: {target_col}",
        f"Target counts: {dict(Counter(y.tolist()))}",
        "",
        "SCANNER",
        "-" * 100,
        f"Included genuine project files: {len(documents)}",
        f"Timing-evidence segments retained: {len(timing_evidence_rows)}",
        (
            "CSV data files were not used as chronology-language evidence."
        ),
        "",
        "CHRONOLOGY CLASSIFICATION COUNTS",
        "-" * 100,
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
            "-" * 100,
        ]
    )

    for key, value in sorted(
        review_counts.items()
    ):

        lines.append(
            f"{key}: {value}"
        )

    # -----------------------------------------------------------------
    # Top repeated single-feature predictors.
    # -----------------------------------------------------------------

    lines.extend(
        [
            "",
            "TOP SINGLE-FEATURE REPEATED PREDICTORS",
            "-" * 100,
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
                f"mean AUC="
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
            "VERY HIGH SINGLE-FEATURE PREDICTIVE STRENGTH",
            "-" * 100,
        ]
    )

    if very_high_features:

        for row in very_high_features:

            lines.append(
                (
                    f"{row['feature']}: "
                    f"directionless repeated AUC="
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
            "DIRECT TEMPORAL / OUTCOME-PROXY FLAGS",
            "-" * 100,
        ]
    )

    if direct_temporal_flags:

        for row in direct_temporal_flags:

            lines.append(
                (
                    f"{row['feature']}: "
                    f"{row['chronology_classification']}; "
                    f"evidence_segments="
                    f"{row['evidence_segments']}; "
                    f"files={row['evidence_files']}"
                )
            )

    else:

        lines.append(
            "NONE DETECTED FROM FILTERED GENUINE PROJECT DOCUMENTATION."
        )

    lines.extend(
        [
            "",
            "HIGH-PREDICTIVE FEATURES WITH UNKNOWN TIMING",
            "-" * 100,
        ]
    )

    if unknown_high_auc_features:

        for row in unknown_high_auc_features:

            lines.append(
                (
                    f"{row['feature']}: "
                    f"directionless repeated AUC="
                    f"{row['repeated_single_feature_directionless_auc']:.6f}; "
                    f"timing=UNKNOWN"
                )
            )

    else:

        lines.append(
            "NONE"
        )

    lines.extend(
        [
            "",
            "AUTOMATED VERDICT",
            "-" * 100,
            f"Verdict: {verdict}",
            f"Manuscript use: {manuscript_use}",
            "",
            "INTERPRETATION",
            "-" * 100,
            (
                "High predictive performance alone is not evidence of leakage."
            ),
            (
                "A feature is considered a temporal-risk candidate only when "
                "genuine project documentation places it after baseline, during "
                "later clinical evolution, after outcome, or derives it from "
                "the outcome."
            ),
            (
                "Where timing remains undocumented, the appropriate response is "
                "to state the limitation and avoid claiming prospective "
                "prediction unless the source documentation can verify the "
                "measurement time."
            ),
            "",
            "PRIMARY OUTPUTS",
            "-" * 100,
            "scanner_file_inclusion_audit.csv",
            "feature_timing_evidence_v2.csv",
            "feature_chronology_classification_v2.csv",
            "feature_proxy_review_priority_v2.csv",
            "univariate_descriptive_statistics_v2.csv",
            "single_feature_repeated_summary_v2.csv",
            "single_feature_repeated_seed_metrics_v2.csv",
            "feature_proxy_audit_verdict_v2.csv",
            "feature_proxy_audit_provenance_v2.csv",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "feature_proxy_audit_summary_v2.txt"
    )

    summary_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console summary.
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "02F V2 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nScannable genuine project files: "
        f"{len(documents)}"
    )

    print(
        f"Timing-evidence segments: "
        f"{len(timing_evidence_rows)}"
    )

    print(
        f"Direct temporal/proxy flags: "
        f"{len(direct_temporal_flags)}"
    )

    print(
        f"Very-high single-feature predictors "
        f"(AUC >= {VERY_HIGH_AUC}): "
        f"{len(very_high_features)}"
    )

    print(
        f"High single-feature predictors "
        f"({HIGH_AUC} <= AUC < {VERY_HIGH_AUC}): "
        f"{len(high_features)}"
    )

    print(
        f"Features with unknown timing: "
        f"{chronology_counts['TIMING_UNKNOWN']}"
    )

    print(
        f"Unknown-timing features with AUC >= {HIGH_AUC}: "
        f"{len(unknown_high_auc_features)}"
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
        "feature_proxy_audit_summary_v2.txt",
        "feature_proxy_audit_verdict_v2.csv",
        "feature_proxy_review_priority_v2.csv",
        "feature_chronology_classification_v2.csv",
        "feature_timing_evidence_v2.csv",
        "scanner_file_inclusion_audit.csv",
        "single_feature_repeated_summary_v2.csv",
        "univariate_descriptive_statistics_v2.csv",
        "feature_proxy_audit_provenance_v2.csv",
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
            "02F V2 FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo source dataset, historical model, historical result, "
            "or prior audit output was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(
            1
        )