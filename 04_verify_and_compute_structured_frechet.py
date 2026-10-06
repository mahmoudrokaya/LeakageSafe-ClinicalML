"""
04_verify_and_compute_structured_frechet.py
===========================================

HFAGM - Fidelity implementation audit and Structured Fréchet Distance (SFD)
recomputation.

PURPOSE
-------
Reviewer #3 questioned the manuscript's use of "FID" for structured clinical
data. The manuscript appears to describe a Fréchet-type distance calculated
from means and covariance matrices of normalized structured variables rather
than conventional Fréchet Inception Distance calculated in an Inception or
other learned image-feature space.

This script therefore:

1. Audits the existing project for actual fidelity/FID/Fréchet code.
2. Searches for actual structured synthetic datasets.
3. Determines whether conventional image FID is implemented anywhere.
4. Reconstructs the common 51-variable structured clinical feature space.
5. Computes a clearly named STRUCTURED FRÉCHET DISTANCE (SFD) only for
   genuine synthetic datasets that can be mapped to all 51 real predictors.
6. Audits the historical manuscript values approximately 1.5, 1.6, and 3.2.
7. Does NOT generate, simulate, infer, tune, or fabricate synthetic data.
8. Does NOT relabel an ordinary tabular moment distance as conventional FID.

IMPORTANT INTERPRETATION
------------------------
The quantity computed here is:

    SFD(R, S)
      = ||mu_R - mu_S||_2^2
        + Tr(
            Sigma_R + Sigma_S
            - 2 * (Sigma_R^(1/2) Sigma_S Sigma_R^(1/2))^(1/2)
          )

where R and S are represented in a REAL-REFERENCE standardized structured
feature space.

This has the same Gaussian Fréchet/Bures moment form commonly underlying FID,
but because no Inception image embedding is used it MUST NOT be described as
Fréchet Inception Distance.

The script reports:
    structured_frechet_distance

and explicitly labels:
    representation = standardized_structured_51_feature_space

If an actual learned embedding is later discovered, it is audited separately
but is never silently substituted into this calculation.

EXPECTED REAL DATA
------------------
    data/raw/covid_clinical.csv
or
    data/raw/covid_clinical.xlsx

Historical 51-column schema:
    data/preprocessed/X_train_scaled.csv

OUTPUT
------
outputs/revision_fidelity/structured_frechet_audit/

Files:
    fidelity_code_evidence.csv
    historical_fidelity_value_evidence.csv
    synthetic_candidate_inventory.csv
    synthetic_candidate_feature_mapping.csv
    structured_frechet_results.csv
    structured_feature_space_audit.csv
    covariance_diagnostics.csv
    historical_claim_comparison.csv
    fidelity_terminology_verdict.csv
    fidelity_provenance.csv
    fidelity_summary.txt

SAFETY
------
No historical file is modified.
No synthetic data are generated.
No unsupported manuscript value is recreated.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sys
import traceback
import zipfile

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn

from sklearn.impute import SimpleImputer
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
    / "revision_fidelity"
    / "structured_frechet_audit"
)

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
# Historical manuscript fidelity claims to audit.
#
# These are NOT accepted values. They are search targets only.
# -------------------------------------------------------------------------

HISTORICAL_FID_CLAIMS = {
    "FID_HGF_OR_BASELINE_1_5": 1.5,
    "FID_LIGHTWEIGHT_1_6": 1.6,
    "FID_LIGHTWEIGHT_3_2": 3.2,
}

CLAIM_MATCH_TOLERANCE = 0.02

# -------------------------------------------------------------------------
# Feature requirements.
#
# We compute primary SFD only if ALL 51 predictors can be mapped.
# -------------------------------------------------------------------------

EXPECTED_FEATURE_COUNT = 51
MIN_FEATURE_MATCH_FOR_CANDIDATE = 0.70
REQUIRE_ALL_FEATURES_FOR_PRIMARY_SFD = True

# Minimum useful number of synthetic rows.
MIN_SYNTHETIC_ROWS = 10

# Numerical tolerance.
PSD_EIGEN_TOLERANCE = 1e-10
NEGATIVE_DISTANCE_TOLERANCE = 1e-8

# -------------------------------------------------------------------------
# Files to inspect for fidelity implementation evidence.
# -------------------------------------------------------------------------

TEXT_CODE_EXTENSIONS = {
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

# Candidate data formats.
TABULAR_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
}

ARRAY_EXTENSIONS = {
    ".npy",
    ".npz",
}

# -------------------------------------------------------------------------
# Directory exclusions.
#
# Do not scan virtual environments / third-party package code, because the
# presence of scipy.linalg.sqrtm inside scipy is NOT evidence that this project
# implemented a Fréchet metric.
# -------------------------------------------------------------------------

EXCLUDED_DIR_EXACT = {
    ".git",
    "__pycache__",
    "site-packages",
    "dist-packages",
    "node_modules",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}

EXCLUDED_DIR_PREFIXES = (
    ".venv",
    "venv",
    ".env",
)

EXCLUDED_DIR_SUFFIXES = (
    ".dist-info",
    ".egg-info",
)

# -------------------------------------------------------------------------
# Exclude our own revision/audit code from historical implementation evidence.
# -------------------------------------------------------------------------

REVISION_SCRIPT_PREFIXES = (
    "01_",
    "02_",
    "02b_",
    "02c_",
    "02d_",
    "02e_",
    "02f_",
    "03_",
    "04_",
)

# Do not treat revision outputs as historical experimental evidence.
REVISION_OUTPUT_PATH_TERMS = (
    "outputs/revision_primary_metrics",
    "outputs\\revision_primary_metrics",
    "outputs/revision_fairness",
    "outputs\\revision_fairness",
    "outputs/revision_fidelity",
    "outputs\\revision_fidelity",
)

# -------------------------------------------------------------------------
# Real/known tables that must NEVER be mistaken for synthetic candidates.
# -------------------------------------------------------------------------

KNOWN_REAL_OR_AUDIT_FILENAMES = {
    "covid_clinical.csv",
    "covid_clinical.xlsx",
    "covid_clinical_preprocessed.csv",
    "covid_clinical_balanced.csv",
    "x_train_scaled.csv",
    "x_test_scaled.csv",
    "y_train.csv",
    "y_test.csv",
    "repeated_predictions.csv",
    "predictions_with_sensitive_attributes.csv",
}

# -------------------------------------------------------------------------
# Hints that make a table more likely to be genuine synthetic data.
# -------------------------------------------------------------------------

SYNTHETIC_PATH_HINTS = (
    "synthetic",
    "generated",
    "generator",
    "gan",
    "vae",
    "diffusion",
    "hfagm",
    "sample",
    "fake",
)

# -------------------------------------------------------------------------
# Implementation evidence patterns.
# -------------------------------------------------------------------------

FID_TERMS = [
    r"\bfid\b",
    r"fr[eé]chet inception distance",
    r"frechet inception distance",
]

FRECHET_TERMS = [
    r"\bfrechet\b",
    r"\bfr[eé]chet\b",
    r"bures",
]

MOMENT_TERMS = [
    r"np\.cov",
    r"numpy\.cov",
    r"\bcovariance\b",
    r"\bmean\s*\(",
    r"np\.mean",
]

MATRIX_SQRT_TERMS = [
    r"sqrtm",
    r"matrix_square_root",
    r"matrix sqrt",
    r"eigh",
    r"eigval",
]

INCEPTION_TERMS = [
    r"\binception\b",
    r"inception_v3",
    r"inceptionv3",
    r"torchvision\.models\.inception",
]

EMBEDDING_TERMS = [
    r"\bembedding\b",
    r"\bencoder\b",
    r"feature representation",
    r"latent representation",
]

SAVE_OR_METRIC_TERMS = [
    r"fid_score",
    r"fid_value",
    r"frechet_distance",
    r"frechet_score",
    r"fidelity",
]


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
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors = []

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]:
        try:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","

            return pd.read_csv(
                path,
                encoding=encoding,
                sep=sep,
            )

        except Exception as exc:
            errors.append(
                f"{encoding}: {repr(exc)}"
            )

    raise RuntimeError(
        f"Could not read tabular text file:\n{path}\n"
        + "\n".join(errors)
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


def path_relative(path: Path) -> str:
    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )
    except Exception:
        return str(path)


def is_revision_output(path: Path) -> bool:
    text = str(path).lower()

    return any(
        token.lower() in text
        for token in REVISION_OUTPUT_PATH_TERMS
    )


def is_excluded_path(path: Path) -> Tuple[bool, str]:
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

        if part in EXCLUDED_DIR_EXACT:
            return (
                True,
                f"excluded_dir:{part}",
            )

        if any(
            part.startswith(prefix)
            for prefix in EXCLUDED_DIR_PREFIXES
        ):
            return (
                True,
                f"excluded_env:{part}",
            )

        if any(
            part.endswith(suffix)
            for suffix in EXCLUDED_DIR_SUFFIXES
        ):
            return (
                True,
                f"excluded_package_metadata:{part}",
            )

    return (
        False,
        "",
    )


def is_revision_script(path: Path) -> bool:
    name = path.name.lower()

    return any(
        name.startswith(prefix.lower())
        for prefix in REVISION_SCRIPT_PREFIXES
    )


# =============================================================================
# 3. REAL DATA / 51-FEATURE SPACE
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
            "Readable raw clinical sources were found, but none "
            "contained the expected 193 rows.\n"
            + details
        )

    raise FileNotFoundError(
        "No usable 193-row raw COVID clinical dataset found."
    )


def identify_target_column(
    df: pd.DataFrame,
) -> str:
    for candidate in TARGET_CANDIDATES:

        if candidate in df.columns:
            return candidate

    normalized = {
        normalize_name(col): col
        for col in df.columns
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


def load_historical_feature_names() -> List[str]:
    if not HISTORICAL_X_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Historical schema not found:\n"
            f"{HISTORICAL_X_TRAIN_PATH}"
        )

    historical = read_csv_robust(
        HISTORICAL_X_TRAIN_PATH
    )

    features = [
        str(col)
        for col in historical.columns
    ]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} historical predictors; "
            f"found {len(features)}."
        )

    return features


def map_dataframe_to_features(
    df: pd.DataFrame,
    feature_names: List[str],
    excluded_column: Optional[str] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """
    Attempt to map a DataFrame to the required historical 51-feature schema.

    Primary computation requires all 51 features.
    """

    normalized_lookup = defaultdict(list)

    for col in df.columns:

        if (
            excluded_column is not None
            and
            col == excluded_column
        ):
            continue

        normalized_lookup[
            normalize_name(
                col
            )
        ].append(
            col
        )

    mapping_rows = []
    mapped = {}
    missing = []
    ambiguous = []

    for feature in feature_names:

        if feature in df.columns:
            source_col = feature
            method = "exact"

        else:
            candidates = normalized_lookup[
                normalize_name(
                    feature
                )
            ]

            if len(candidates) == 1:
                source_col = candidates[0]
                method = "normalized"

            elif len(candidates) == 0:
                source_col = None
                method = "missing"
                missing.append(
                    feature
                )

            else:
                source_col = None
                method = "ambiguous"
                ambiguous.append(
                    {
                        "feature": feature,
                        "candidates": candidates,
                    }
                )

        mapping_rows.append(
            {
                "required_feature":
                    feature,

                "source_column":
                    source_col,

                "mapping_method":
                    method,
            }
        )

        if source_col is not None:
            mapped[
                feature
            ] = df[
                source_col
            ].copy()

    matched = len(
        mapped
    )

    fraction = matched / len(
        feature_names
    )

    diagnostics = {
        "required_features":
            len(feature_names),

        "matched_features":
            matched,

        "feature_match_fraction":
            fraction,

        "missing_features":
            missing,

        "ambiguous_features":
            ambiguous,
    }

    if (
        missing
        or
        ambiguous
    ):
        return (
            None,
            mapping_rows,
            diagnostics,
        )

    X = pd.DataFrame(
        mapped
    )

    return (
        X,
        mapping_rows,
        diagnostics,
    )


def numeric_series(series: pd.Series) -> pd.Series:
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

                "none":
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


def convert_numeric(
    X: pd.DataFrame,
    dataset_label: str,
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    out = pd.DataFrame(
        index=X.index
    )

    audit = []

    for feature in X.columns:

        original_nonmissing = int(
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
            original_nonmissing
            - numeric_nonmissing
        )

        failure_fraction = (
            failures / original_nonmissing
            if original_nonmissing > 0
            else 0.0
        )

        out[
            feature
        ] = numeric

        audit.append(
            {
                "dataset":
                    dataset_label,

                "feature":
                    feature,

                "original_nonmissing":
                    original_nonmissing,

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
# 4. TEXT EXTRACTION FOR HISTORICAL IMPLEMENTATION AUDIT
# =============================================================================

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

            xml = z.read(
                "word/document.xml"
            ).decode(
                "utf-8",
                errors="ignore",
            )

            xml = re.sub(
                r"</w:p>",
                "\n",
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

            return re.sub(
                r"[ \t]+",
                " ",
                text,
            )

    except Exception:
        return None


def regex_hits(
    text: str,
    patterns: Iterable[str],
) -> List[str]:
    hits = []

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            hits.append(
                pattern
            )

    return hits


def audit_fidelity_implementation(
) -> List[Dict[str, Any]]:
    rows = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        excluded, reason = is_excluded_path(
            path
        )

        if excluded:
            continue

        if is_revision_output(
            path
        ):
            continue

        if is_revision_script(
            path
        ):
            continue

        suffix = path.suffix.lower()

        if (
            suffix not in TEXT_CODE_EXTENSIONS
            and
            suffix != DOCX_EXTENSION
        ):
            continue

        if suffix == DOCX_EXTENSION:
            text = read_docx_text(
                path
            )
        else:
            text = read_text_file(
                path
            )

        if not text:
            continue

        fid_hits = regex_hits(
            text,
            FID_TERMS,
        )

        frechet_hits = regex_hits(
            text,
            FRECHET_TERMS,
        )

        moment_hits = regex_hits(
            text,
            MOMENT_TERMS,
        )

        sqrt_hits = regex_hits(
            text,
            MATRIX_SQRT_TERMS,
        )

        inception_hits = regex_hits(
            text,
            INCEPTION_TERMS,
        )

        embedding_hits = regex_hits(
            text,
            EMBEDDING_TERMS,
        )

        metric_hits = regex_hits(
            text,
            SAVE_OR_METRIC_TERMS,
        )

        # Only retain potentially relevant files.
        if not (
            fid_hits
            or
            frechet_hits
            or
            metric_hits
        ):
            continue

        rows.append(
            {
                "file":
                    path_relative(
                        path
                    ),

                "suffix":
                    suffix,

                "fid_term_found":
                    int(
                        bool(
                            fid_hits
                        )
                    ),

                "frechet_term_found":
                    int(
                        bool(
                            frechet_hits
                        )
                    ),

                "moment_code_found":
                    int(
                        bool(
                            moment_hits
                        )
                    ),

                "matrix_sqrt_code_found":
                    int(
                        bool(
                            sqrt_hits
                        )
                    ),

                "inception_reference_found":
                    int(
                        bool(
                            inception_hits
                        )
                    ),

                "embedding_reference_found":
                    int(
                        bool(
                            embedding_hits
                        )
                    ),

                "metric_term_found":
                    int(
                        bool(
                            metric_hits
                        )
                    ),

                "fid_patterns":
                    safe_json(
                        fid_hits
                    ),

                "frechet_patterns":
                    safe_json(
                        frechet_hits
                    ),

                "moment_patterns":
                    safe_json(
                        moment_hits
                    ),

                "matrix_sqrt_patterns":
                    safe_json(
                        sqrt_hits
                    ),

                "inception_patterns":
                    safe_json(
                        inception_hits
                    ),

                "embedding_patterns":
                    safe_json(
                        embedding_hits
                    ),
            }
        )

    return rows


# =============================================================================
# 5. HISTORICAL VALUE AUDIT
# =============================================================================

def extract_numeric_values(text: str) -> List[float]:
    values = []

    pattern = re.compile(
        r"(?<![A-Za-z0-9_.])"
        r"[-+]?"
        r"(?:\d+\.\d+|\d+)"
        r"(?:[eE][-+]?\d+)?"
        r"(?![A-Za-z0-9_.])"
    )

    for match in pattern.finditer(
        text
    ):
        try:
            values.append(
                float(
                    match.group(0)
                )
            )
        except Exception:
            pass

    return values


def audit_historical_fidelity_values(
) -> List[Dict[str, Any]]:
    rows = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        excluded, _ = is_excluded_path(
            path
        )

        if excluded:
            continue

        if is_revision_output(
            path
        ):
            continue

        if is_revision_script(
            path
        ):
            continue

        suffix = path.suffix.lower()

        # Historical numeric claims should be searched only in manageable
        # text/config/result files, not images or raw ArSL labels.
        if (
            suffix not in TEXT_CODE_EXTENSIONS
            and
            suffix != DOCX_EXTENSION
            and
            suffix not in {".csv", ".tsv"}
        ):
            continue

        try:
            if suffix == DOCX_EXTENSION:
                text = read_docx_text(
                    path
                )

            elif suffix in {
                ".csv",
                ".tsv",
            }:
                # Avoid enormous raw data tables.
                if path.stat().st_size > 20 * 1024 * 1024:
                    continue

                text = read_text_file(
                    path
                )

            else:
                text = read_text_file(
                    path
                )

        except Exception:
            continue

        if not text:
            continue

        lower = text.lower()

        # Require fidelity context.
        has_fidelity_context = (
            "fid" in lower
            or
            "frechet" in lower
            or
            "fréchet" in lower
            or
            "fidelity" in lower
        )

        if not has_fidelity_context:
            continue

        numbers = extract_numeric_values(
            text
        )

        for claim_name, claim_value in HISTORICAL_FID_CLAIMS.items():

            matched_values = [
                value
                for value in numbers
                if abs(
                    value
                    - claim_value
                )
                <= CLAIM_MATCH_TOLERANCE
            ]

            if not matched_values:
                continue

            # Classify source conservatively.
            relative = path_relative(
                path
            )

            lower_path = relative.lower()

            if (
                "manuscript" in lower_path
                or
                "paper" in lower_path
                or
                suffix == ".docx"
            ):
                evidence_type = (
                    "MANUSCRIPT_OR_NARRATIVE_ONLY"
                )

            elif (
                "metric" in lower_path
                or
                "result" in lower_path
                or
                "experiment" in lower_path
                or
                "output" in lower_path
            ):
                evidence_type = (
                    "POTENTIAL_EXPERIMENTAL_OUTPUT"
                )

            elif suffix == ".py":
                evidence_type = (
                    "CODE_OR_HARDCODED_VALUE"
                )

            else:
                evidence_type = (
                    "UNCLASSIFIED_CONTEXT"
                )

            rows.append(
                {
                    "claim_name":
                        claim_name,

                    "claim_target":
                        claim_value,

                    "file":
                        relative,

                    "evidence_type":
                        evidence_type,

                    "matched_values":
                        safe_json(
                            matched_values
                        ),

                    "independently_recomputed":
                        False,
                }
            )

    return rows


# =============================================================================
# 6. SYNTHETIC DATA DISCOVERY
# =============================================================================

def candidate_path_score(path: Path) -> int:
    text = str(
        path
    ).lower()

    score = 0

    for hint in SYNTHETIC_PATH_HINTS:

        if hint in text:
            score += 1

    return score


def load_tabular_file(
    path: Path,
) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()

    try:
        if suffix in {
            ".csv",
            ".tsv",
        }:
            return read_csv_robust(
                path
            )

        if suffix in {
            ".xlsx",
            ".xls",
        }:
            return pd.read_excel(
                path
            )

        if suffix == ".parquet":
            return pd.read_parquet(
                path
            )

    except Exception:
        return None

    return None


def discover_synthetic_tabular_candidates(
    feature_names: List[str],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, pd.DataFrame],
    Dict[str, List[Dict[str, Any]]],
]:
    inventory = []
    usable_tables = {}
    mapping_by_file = {}

    normalized_required = {
        normalize_name(
            feature
        )
        for feature in feature_names
    }

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        excluded, reason = is_excluded_path(
            path
        )

        if excluded:
            continue

        if is_revision_output(
            path
        ):
            continue

        suffix = path.suffix.lower()

        if suffix not in TABULAR_EXTENSIONS:
            continue

        name_lower = path.name.lower()

        if name_lower in KNOWN_REAL_OR_AUDIT_FILENAMES:
            continue

        # Skip obvious metrics / audit tables.
        lower_path = str(
            path
        ).lower()

        if any(
            term in lower_path
            for term in [
                "metric",
                "summary",
                "confusion",
                "prediction",
                "audit",
                "provenance",
                "fairness",
                "registry",
                "claim_mapping",
            ]
        ):
            continue

        path_hint_score = candidate_path_score(
            path
        )

        df = load_tabular_file(
            path
        )

        if df is None:
            continue

        if df.empty:
            inventory.append(
                {
                    "file":
                        path_relative(
                            path
                        ),

                    "rows":
                        0,

                    "columns":
                        len(
                            df.columns
                        ),

                    "path_synthetic_hint_score":
                        path_hint_score,

                    "matched_feature_count":
                        0,

                    "feature_match_fraction":
                        0.0,

                    "candidate_status":
                        "EMPTY_TABLE",
                }
            )

            continue

        normalized_columns = {
            normalize_name(
                col
            )
            for col in df.columns
        }

        matched_count = len(
            normalized_required
            &
            normalized_columns
        )

        match_fraction = (
            matched_count
            / len(
                feature_names
            )
        )

        # To avoid calling arbitrary tables "synthetic", require either:
        # a synthetic path/name hint OR very strong feature schema agreement.
        candidate_like = (
            path_hint_score > 0
            or
            match_fraction >= MIN_FEATURE_MATCH_FOR_CANDIDATE
        )

        if not candidate_like:
            continue

        (
            mapped,
            mapping_rows,
            diagnostics,
        ) = map_dataframe_to_features(
            df,
            feature_names,
            excluded_column=None,
        )

        relative = path_relative(
            path
        )

        mapping_by_file[
            relative
        ] = mapping_rows

        all_features = (
            diagnostics[
                "matched_features"
            ]
            ==
            EXPECTED_FEATURE_COUNT
            and
            not diagnostics[
                "missing_features"
            ]
            and
            not diagnostics[
                "ambiguous_features"
            ]
        )

        if len(
            df
        ) < MIN_SYNTHETIC_ROWS:
            status = (
                "TOO_FEW_ROWS"
            )

        elif not all_features:
            status = (
                "INCOMPLETE_51_FEATURE_SCHEMA"
            )

        else:
            status = (
                "ELIGIBLE_STRUCTURED_SYNTHETIC_CANDIDATE"
            )

            usable_tables[
                relative
            ] = mapped

        inventory.append(
            {
                "file":
                    relative,

                "rows":
                    len(
                        df
                    ),

                "columns":
                    len(
                        df.columns
                    ),

                "path_synthetic_hint_score":
                    path_hint_score,

                "matched_feature_count":
                    diagnostics[
                        "matched_features"
                    ],

                "feature_match_fraction":
                    diagnostics[
                        "feature_match_fraction"
                    ],

                "missing_feature_count":
                    len(
                        diagnostics[
                            "missing_features"
                        ]
                    ),

                "ambiguous_feature_count":
                    len(
                        diagnostics[
                            "ambiguous_features"
                        ]
                    ),

                "missing_features":
                    safe_json(
                        diagnostics[
                            "missing_features"
                        ]
                    ),

                "candidate_status":
                    status,
            }
        )

    return (
        inventory,
        usable_tables,
        mapping_by_file,
    )


# =============================================================================
# 7. OPTIONAL NUMPY ARRAY DISCOVERY
# =============================================================================

def discover_structured_array_candidates(
    feature_count: int,
) -> List[Dict[str, Any]]:
    rows = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        excluded, _ = is_excluded_path(
            path
        )

        if excluded:
            continue

        if is_revision_output(
            path
        ):
            continue

        if path.suffix.lower() not in ARRAY_EXTENSIONS:
            continue

        hint_score = candidate_path_score(
            path
        )

        if hint_score <= 0:
            continue

        relative = path_relative(
            path
        )

        try:
            if path.suffix.lower() == ".npy":

                arr = np.load(
                    path,
                    allow_pickle=False,
                )

                arrays = {
                    "array": arr
                }

            else:

                archive = np.load(
                    path,
                    allow_pickle=False,
                )

                arrays = {
                    key: archive[
                        key
                    ]
                    for key in archive.files
                }

            for key, arr in arrays.items():

                shape = tuple(
                    arr.shape
                )

                eligible = (
                    arr.ndim == 2
                    and
                    arr.shape[1]
                    == feature_count
                    and
                    arr.shape[0]
                    >= MIN_SYNTHETIC_ROWS
                )

                rows.append(
                    {
                        "file":
                            relative,

                        "array_key":
                            key,

                        "shape":
                            safe_json(
                                shape
                            ),

                        "dtype":
                            str(
                                arr.dtype
                            ),

                        "path_synthetic_hint_score":
                            hint_score,

                        "potential_51_column_array":
                            int(
                                eligible
                            ),

                        "status":
                            (
                                "POTENTIAL_ARRAY_BUT_FEATURE_ORDER_UNVERIFIED"
                                if eligible
                                else
                                "NOT_51_COLUMN_STRUCTURED_ARRAY"
                            ),
                    }
                )

        except Exception as exc:

            rows.append(
                {
                    "file":
                        relative,

                    "array_key":
                        "",

                    "shape":
                        "",

                    "dtype":
                        "",

                    "path_synthetic_hint_score":
                        hint_score,

                    "potential_51_column_array":
                        0,

                    "status":
                        "ARRAY_READ_FAILED",

                    "error":
                        repr(
                            exc
                        ),
                }
            )

    return rows


# =============================================================================
# 8. REAL-REFERENCE STRUCTURED FEATURE TRANSFORM
# =============================================================================

def fit_real_reference_transform(
    X_real_numeric: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    SimpleImputer,
    StandardScaler,
    List[Dict[str, Any]],
]:
    """
    Fit preprocessing ONLY on the real reference dataset.

    For fidelity measurement this establishes a common coordinate system.

    This is not the classifier evaluation protocol and should not be interpreted
    as a predictive train/test operation.
    """

    diagnostics = []

    imputer = SimpleImputer(
        strategy="median"
    )

    X_real_imputed = imputer.fit_transform(
        X_real_numeric
    )

    scaler = StandardScaler()

    X_real_standardized = scaler.fit_transform(
        X_real_imputed
    )

    for index, feature in enumerate(
        X_real_numeric.columns
    ):

        diagnostics.append(
            {
                "feature":
                    feature,

                "real_missing_count":
                    int(
                        X_real_numeric[
                            feature
                        ].isna().sum()
                    ),

                "real_imputation_median":
                    float(
                        imputer.statistics_[
                            index
                        ]
                    ),

                "real_scaler_mean":
                    float(
                        scaler.mean_[
                            index
                        ]
                    ),

                "real_scaler_scale":
                    float(
                        scaler.scale_[
                            index
                        ]
                    ),

                "representation":
                    "real_reference_standardized_structured_feature",
            }
        )

    return (
        X_real_standardized,
        imputer,
        scaler,
        diagnostics,
    )


def transform_synthetic_with_real_reference(
    X_synth_numeric: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply the real-reference transform to the synthetic table.

    Critical:
        imputer.fit is NOT called on synthetic data
        scaler.fit is NOT called on synthetic data
    """

    X_synth_imputed = imputer.transform(
        X_synth_numeric
    )

    return scaler.transform(
        X_synth_imputed
    )


# =============================================================================
# 9. MATRIX FUNCTIONS
# =============================================================================

def symmetric_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    return (
        matrix
        + matrix.T
    ) / 2.0


def psd_matrix_sqrt(
    matrix: np.ndarray,
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:
    """
    Symmetric PSD square root using eigen decomposition.

    Negative eigenvalues within numerical tolerance are set to zero.
    Materially negative eigenvalues are recorded.
    """

    matrix = symmetric_matrix(
        np.asarray(
            matrix,
            dtype=float,
        )
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix
    )

    min_eigenvalue = float(
        np.min(
            eigenvalues
        )
    )

    materially_negative = int(
        np.sum(
            eigenvalues
            <
            -PSD_EIGEN_TOLERANCE
        )
    )

    clipped = np.clip(
        eigenvalues,
        a_min=0.0,
        a_max=None,
    )

    root = (
        eigenvectors
        @ np.diag(
            np.sqrt(
                clipped
            )
        )
        @ eigenvectors.T
    )

    root = symmetric_matrix(
        root
    )

    diagnostics = {
        "min_eigenvalue":
            min_eigenvalue,

        "materially_negative_eigenvalue_count":
            materially_negative,

        "zero_or_clipped_eigenvalue_count":
            int(
                np.sum(
                    eigenvalues
                    <= PSD_EIGEN_TOLERANCE
                )
            ),
    }

    return (
        root,
        diagnostics,
    )


def covariance_matrix(
    X: np.ndarray,
) -> np.ndarray:
    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2-D matrix; got shape {X.shape}"
        )

    if X.shape[0] < 2:
        raise ValueError(
            "At least two rows are required for covariance."
        )

    cov = np.cov(
        X,
        rowvar=False,
        ddof=1,
    )

    if np.ndim(
        cov
    ) == 0:
        cov = np.array(
            [
                [
                    float(
                        cov
                    )
                ]
            ]
        )

    return symmetric_matrix(
        np.asarray(
            cov,
            dtype=float,
        )
    )


# =============================================================================
# 10. STRUCTURED FRÉCHET DISTANCE
# =============================================================================

def structured_frechet_distance(
    X_real: np.ndarray,
    X_synth: np.ndarray,
) -> Tuple[
    float,
    Dict[str, Any],
]:
    """
    Compute Gaussian Fréchet/Bures squared moment distance:

        ||mu_r - mu_s||^2
        + Tr(C_r + C_s - 2*(C_r^1/2 C_s C_r^1/2)^1/2)

    Returned value is the usual squared Fréchet moment expression, analogous
    to the scalar commonly reported as FID, but here named SFD because this is
    NOT an Inception representation.
    """

    X_real = np.asarray(
        X_real,
        dtype=float,
    )

    X_synth = np.asarray(
        X_synth,
        dtype=float,
    )

    if X_real.ndim != 2:
        raise ValueError(
            "Real matrix must be 2-D."
        )

    if X_synth.ndim != 2:
        raise ValueError(
            "Synthetic matrix must be 2-D."
        )

    if X_real.shape[1] != X_synth.shape[1]:
        raise ValueError(
            "Real and synthetic feature dimensions differ: "
            f"{X_real.shape[1]} vs {X_synth.shape[1]}"
        )

    if not np.all(
        np.isfinite(
            X_real
        )
    ):
        raise ValueError(
            "Real transformed matrix contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            X_synth
        )
    ):
        raise ValueError(
            "Synthetic transformed matrix contains non-finite values."
        )

    mu_real = np.mean(
        X_real,
        axis=0,
    )

    mu_synth = np.mean(
        X_synth,
        axis=0,
    )

    cov_real = covariance_matrix(
        X_real
    )

    cov_synth = covariance_matrix(
        X_synth
    )

    mean_term = float(
        np.sum(
            (
                mu_real
                - mu_synth
            )
            ** 2
        )
    )

    real_sqrt, real_sqrt_diag = psd_matrix_sqrt(
        cov_real
    )

    middle = (
        real_sqrt
        @ cov_synth
        @ real_sqrt
    )

    middle = symmetric_matrix(
        middle
    )

    middle_sqrt, middle_sqrt_diag = psd_matrix_sqrt(
        middle
    )

    trace_real = float(
        np.trace(
            cov_real
        )
    )

    trace_synth = float(
        np.trace(
            cov_synth
        )
    )

    trace_cross_root = float(
        np.trace(
            middle_sqrt
        )
    )

    covariance_term = float(
        trace_real
        + trace_synth
        - 2.0
        * trace_cross_root
    )

    distance = float(
        mean_term
        + covariance_term
    )

    # Only correct tiny negative roundoff.
    if (
        distance < 0
        and
        distance
        >= -NEGATIVE_DISTANCE_TOLERANCE
    ):
        distance = 0.0

    diagnostics = {
        "n_real":
            X_real.shape[0],

        "n_synthetic":
            X_synth.shape[0],

        "dimension":
            X_real.shape[1],

        "mean_term":
            mean_term,

        "covariance_term":
            covariance_term,

        "trace_cov_real":
            trace_real,

        "trace_cov_synthetic":
            trace_synth,

        "trace_cross_root":
            trace_cross_root,

        "real_cov_rank":
            int(
                np.linalg.matrix_rank(
                    cov_real
                )
            ),

        "synthetic_cov_rank":
            int(
                np.linalg.matrix_rank(
                    cov_synth
                )
            ),

        "real_cov_condition_number":
            float(
                np.linalg.cond(
                    cov_real
                )
            ),

        "synthetic_cov_condition_number":
            float(
                np.linalg.cond(
                    cov_synth
                )
            ),

        "real_cov_min_eigenvalue":
            float(
                np.min(
                    np.linalg.eigvalsh(
                        cov_real
                    )
                )
            ),

        "synthetic_cov_min_eigenvalue":
            float(
                np.min(
                    np.linalg.eigvalsh(
                        cov_synth
                    )
                )
            ),

        "real_sqrt_materially_negative_eigenvalues":
            real_sqrt_diag[
                "materially_negative_eigenvalue_count"
            ],

        "middle_sqrt_materially_negative_eigenvalues":
            middle_sqrt_diag[
                "materially_negative_eigenvalue_count"
            ],
    }

    return (
        distance,
        diagnostics,
    )


# =============================================================================
# 11. SELF-CHECKS
# =============================================================================

def run_sfd_self_checks(
    X_real: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []

    # Identical array should be approximately zero.
    value, diag = structured_frechet_distance(
        X_real,
        X_real.copy(),
    )

    rows.append(
        {
            "check":
                "identity_real_vs_real",

            "value":
                value,

            "expected":
                "approximately 0",

            "status":
                (
                    "PASS"
                    if abs(
                        value
                    )
                    <= 1e-7
                    else "REVIEW"
                ),
        }
    )

    # Row permutation should not change moments.
    rng = np.random.default_rng(
        20260902
    )

    permuted = X_real[
        rng.permutation(
            len(
                X_real
            )
        )
    ]

    value_perm, _ = structured_frechet_distance(
        X_real,
        permuted,
    )

    rows.append(
        {
            "check":
                "row_permutation_invariance",

            "value":
                value_perm,

            "expected":
                "approximately 0",

            "status":
                (
                    "PASS"
                    if abs(
                        value_perm
                    )
                    <= 1e-7
                    else "REVIEW"
                ),
        }
    )

    return rows


# =============================================================================
# 12. HISTORICAL CLAIM COMPARISON
# =============================================================================

def compare_computed_to_historical_claims(
    results_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    if not results_rows:

        for claim_name, value in HISTORICAL_FID_CLAIMS.items():

            rows.append(
                {
                    "claim_name":
                        claim_name,

                    "historical_reported_value":
                        value,

                    "synthetic_dataset":
                        "",

                    "computed_structured_frechet_distance":
                        np.nan,

                    "absolute_difference":
                        np.nan,

                    "matches_historical_value":
                        False,

                    "status":
                        "NO_ACTUAL_SYNTHETIC_DATASET_AVAILABLE_FOR_RECOMPUTATION",
                }
            )

        return rows

    for claim_name, claim_value in HISTORICAL_FID_CLAIMS.items():

        for result in results_rows:

            computed = result[
                "structured_frechet_distance"
            ]

            difference = abs(
                computed
                - claim_value
            )

            rows.append(
                {
                    "claim_name":
                        claim_name,

                    "historical_reported_value":
                        claim_value,

                    "synthetic_dataset":
                        result[
                            "synthetic_dataset"
                        ],

                    "computed_structured_frechet_distance":
                        computed,

                    "absolute_difference":
                        difference,

                    "matches_historical_value":
                        bool(
                            difference
                            <= CLAIM_MATCH_TOLERANCE
                        ),

                    "status":
                        (
                            "NUMERICALLY_CLOSE_BUT_PROVENANCE_STILL_REQUIRED"
                            if difference
                            <= CLAIM_MATCH_TOLERANCE
                            else
                            "DOES_NOT_MATCH"
                        ),
                }
            )

    return rows


# =============================================================================
# 13. MAIN
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
        "HFAGM - VERIFY FID IMPLEMENTATION AND COMPUTE STRUCTURED FRECHET DISTANCE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nPython: {sys.version.split()[0]}"
    )

    print(
        f"NumPy: {np.__version__}"
    )

    print(
        f"pandas: {pd.__version__}"
    )

    print(
        f"SciPy: {scipy.__version__}"
    )

    print(
        f"scikit-learn: {sklearn.__version__}"
    )

    # -----------------------------------------------------------------
    # Load real data and exact 51-feature schema.
    # -----------------------------------------------------------------

    (
        raw_df,
        raw_path,
        raw_source_type,
    ) = load_raw_dataset()

    target_col = identify_target_column(
        raw_df
    )

    feature_names = load_historical_feature_names()

    (
        X_real_source,
        real_mapping_rows,
        real_mapping_diag,
    ) = map_dataframe_to_features(
        raw_df,
        feature_names,
        excluded_column=target_col,
    )

    if X_real_source is None:
        raise RuntimeError(
            "Could not map all 51 historical predictors to raw real data.\n"
            f"Diagnostics: {safe_json(real_mapping_diag)}"
        )

    (
        X_real_numeric,
        real_numeric_audit,
    ) = convert_numeric(
        X_real_source,
        dataset_label="real_reference",
    )

    print(
        f"\nReal source: {raw_path}"
    )

    print(
        f"Rows: {len(raw_df)}"
    )

    print(
        f"Target excluded from fidelity space: {target_col}"
    )

    print(
        f"Structured predictors: {len(feature_names)}"
    )

    # -----------------------------------------------------------------
    # Fit real-reference representation.
    # -----------------------------------------------------------------

    (
        X_real_standardized,
        real_imputer,
        real_scaler,
        feature_space_rows,
    ) = fit_real_reference_transform(
        X_real_numeric
    )

    for row in feature_space_rows:
        row[
            "real_source"
        ] = str(
            raw_path
        )

    write_csv(
        OUTPUT_DIR
        / "structured_feature_space_audit.csv",
        feature_space_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "real_feature_mapping.csv",
        real_mapping_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "real_numeric_conversion_audit.csv",
        real_numeric_audit,
    )

    # -----------------------------------------------------------------
    # Internal numerical self-check.
    # -----------------------------------------------------------------

    self_check_rows = run_sfd_self_checks(
        X_real_standardized
    )

    write_csv(
        OUTPUT_DIR
        / "structured_frechet_self_checks.csv",
        self_check_rows,
    )

    failed_self_checks = [
        row
        for row in self_check_rows
        if row[
            "status"
        ] != "PASS"
    ]

    if failed_self_checks:
        raise RuntimeError(
            "Structured Fréchet numerical self-check failed:\n"
            + safe_json(
                failed_self_checks
            )
        )

    # -----------------------------------------------------------------
    # Historical implementation audit.
    # -----------------------------------------------------------------

    print(
        "\nAuditing historical fidelity/FID implementation..."
    )

    fidelity_code_rows = audit_fidelity_implementation()

    write_csv(
        OUTPUT_DIR
        / "fidelity_code_evidence.csv",
        fidelity_code_rows,
    )

    actual_inception_code_rows = [
        row
        for row in fidelity_code_rows
        if (
            row[
                "fid_term_found"
            ] == 1
            and
            row[
                "inception_reference_found"
            ] == 1
        )
    ]

    actual_moment_frechet_rows = [
        row
        for row in fidelity_code_rows
        if (
            (
                row[
                    "fid_term_found"
                ] == 1
                or
                row[
                    "frechet_term_found"
                ] == 1
            )
            and
            row[
                "moment_code_found"
            ] == 1
        )
    ]

    print(
        f"Potential historical fidelity files: "
        f"{len(fidelity_code_rows)}"
    )

    print(
        f"Files combining FID and Inception evidence: "
        f"{len(actual_inception_code_rows)}"
    )

    print(
        f"Files combining Fréchet/FID and moment-code evidence: "
        f"{len(actual_moment_frechet_rows)}"
    )

    # -----------------------------------------------------------------
    # Historical value evidence.
    # -----------------------------------------------------------------

    historical_value_rows = audit_historical_fidelity_values()

    write_csv(
        OUTPUT_DIR
        / "historical_fidelity_value_evidence.csv",
        historical_value_rows,
    )

    # -----------------------------------------------------------------
    # Search for genuine structured synthetic tables.
    # -----------------------------------------------------------------

    print(
        "\nSearching for genuine structured synthetic datasets..."
    )

    (
        candidate_inventory_rows,
        usable_tables,
        mapping_by_file,
    ) = discover_synthetic_tabular_candidates(
        feature_names
    )

    write_csv(
        OUTPUT_DIR
        / "synthetic_candidate_inventory.csv",
        candidate_inventory_rows,
    )

    mapping_flat_rows = []

    for filename, mapping_rows in mapping_by_file.items():

        for row in mapping_rows:

            mapping_flat_rows.append(
                {
                    "synthetic_dataset":
                        filename,

                    **row,
                }
            )

    write_csv(
        OUTPUT_DIR
        / "synthetic_candidate_feature_mapping.csv",
        mapping_flat_rows,
    )

    # -----------------------------------------------------------------
    # Find possible raw arrays but do NOT compute from them automatically
    # because feature ordering cannot be proven merely from shape.
    # -----------------------------------------------------------------

    array_candidate_rows = discover_structured_array_candidates(
        EXPECTED_FEATURE_COUNT
    )

    write_csv(
        OUTPUT_DIR
        / "synthetic_array_candidate_inventory.csv",
        array_candidate_rows,
    )

    print(
        f"Tabular synthetic candidates inspected: "
        f"{len(candidate_inventory_rows)}"
    )

    print(
        f"Eligible full 51-feature tables: "
        f"{len(usable_tables)}"
    )

    potential_unverified_arrays = [
        row
        for row in array_candidate_rows
        if row[
            "potential_51_column_array"
        ] == 1
    ]

    print(
        f"Potential 51-column arrays with unverified feature order: "
        f"{len(potential_unverified_arrays)}"
    )

    # -----------------------------------------------------------------
    # Compute SFD for each genuine full-schema table.
    # -----------------------------------------------------------------

    result_rows = []
    covariance_rows = []
    synthetic_numeric_audit_rows = []

    for synthetic_name, X_synth_source in usable_tables.items():

        print(
            f"\nComputing SFD for:\n  {synthetic_name}"
        )

        (
            X_synth_numeric,
            numeric_audit,
        ) = convert_numeric(
            X_synth_source,
            dataset_label=synthetic_name,
        )

        synthetic_numeric_audit_rows.extend(
            numeric_audit
        )

        # Reject a synthetic table if any feature is entirely missing/non-numeric.
        unusable_features = []

        for feature in feature_names:

            if (
                X_synth_numeric[
                    feature
                ].notna().sum()
                == 0
            ):
                unusable_features.append(
                    feature
                )

        if unusable_features:

            result_rows.append(
                {
                    "synthetic_dataset":
                        synthetic_name,

                    "status":
                        "FAILED_ALL_MISSING_OR_NONNUMERIC_FEATURE",

                    "n_real":
                        len(
                            X_real_numeric
                        ),

                    "n_synthetic":
                        len(
                            X_synth_numeric
                        ),

                    "dimension":
                        EXPECTED_FEATURE_COUNT,

                    "structured_frechet_distance":
                        np.nan,

                    "unusable_features":
                        safe_json(
                            unusable_features
                        ),
                }
            )

            continue

        try:
            X_synth_standardized = (
                transform_synthetic_with_real_reference(
                    X_synth_numeric,
                    real_imputer,
                    real_scaler,
                )
            )

            (
                sfd_value,
                diagnostics,
            ) = structured_frechet_distance(
                X_real_standardized,
                X_synth_standardized,
            )

            result_rows.append(
                {
                    "synthetic_dataset":
                        synthetic_name,

                    "status":
                        "COMPUTED",

                    "representation":
                        (
                            "standardized_structured_51_feature_space"
                        ),

                    "metric_name":
                        "Structured Frechet Distance",

                    "metric_abbreviation":
                        "SFD",

                    "conventional_fid":
                        False,

                    "inception_features_used":
                        False,

                    "real_reference_transform":
                        (
                            "median imputation and StandardScaler "
                            "fit on real reference only"
                        ),

                    "target_included":
                        False,

                    "n_real":
                        diagnostics[
                            "n_real"
                        ],

                    "n_synthetic":
                        diagnostics[
                            "n_synthetic"
                        ],

                    "dimension":
                        diagnostics[
                            "dimension"
                        ],

                    "structured_frechet_distance":
                        sfd_value,

                    "mean_term":
                        diagnostics[
                            "mean_term"
                        ],

                    "covariance_term":
                        diagnostics[
                            "covariance_term"
                        ],
                }
            )

            covariance_rows.append(
                {
                    "synthetic_dataset":
                        synthetic_name,

                    **diagnostics,
                }
            )

        except Exception as exc:

            result_rows.append(
                {
                    "synthetic_dataset":
                        synthetic_name,

                    "status":
                        "COMPUTATION_FAILED",

                    "representation":
                        (
                            "standardized_structured_51_feature_space"
                        ),

                    "metric_name":
                        "Structured Frechet Distance",

                    "metric_abbreviation":
                        "SFD",

                    "conventional_fid":
                        False,

                    "n_real":
                        len(
                            X_real_standardized
                        ),

                    "n_synthetic":
                        len(
                            X_synth_numeric
                        ),

                    "dimension":
                        EXPECTED_FEATURE_COUNT,

                    "structured_frechet_distance":
                        np.nan,

                    "error":
                        repr(
                            exc
                        ),
                }
            )

    write_csv(
        OUTPUT_DIR
        / "structured_frechet_results.csv",
        result_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "covariance_diagnostics.csv",
        covariance_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "synthetic_numeric_conversion_audit.csv",
        synthetic_numeric_audit_rows,
    )

    # -----------------------------------------------------------------
    # Compare to old 1.5 / 1.6 / 3.2 claims.
    # -----------------------------------------------------------------

    computed_rows = [
        row
        for row in result_rows
        if (
            row.get(
                "status"
            )
            == "COMPUTED"
            and
            np.isfinite(
                row.get(
                    "structured_frechet_distance",
                    np.nan,
                )
            )
        )
    ]

    historical_comparison_rows = compare_computed_to_historical_claims(
        computed_rows
    )

    write_csv(
        OUTPUT_DIR
        / "historical_claim_comparison.csv",
        historical_comparison_rows,
    )

    # -----------------------------------------------------------------
    # Terminology verdict.
    # -----------------------------------------------------------------

    conventional_fid_implementation_verified = bool(
        actual_inception_code_rows
    )

    structured_moment_implementation_found = bool(
        actual_moment_frechet_rows
    )

    computed_sfd_count = len(
        computed_rows
    )

    if conventional_fid_implementation_verified:

        terminology_verdict = (
            "POTENTIAL_CONVENTIONAL_FID_IMPLEMENTATION_FOUND_REQUIRES_MANUAL_PROVENANCE_REVIEW"
        )

        manuscript_action = (
            "Do not rename automatically. Inspect the exact implementation, "
            "input modality, pretrained representation, and synthetic files "
            "before deciding whether any specific result qualifies as FID."
        )

    elif computed_sfd_count > 0:

        terminology_verdict = (
            "USE_STRUCTURED_FRECHET_DISTANCE_NOT_FRECHET_INCEPTION_DISTANCE"
        )

        manuscript_action = (
            "For the structured clinical experiment, report the recomputed "
            "quantity as Structured Frechet Distance (SFD), explicitly define "
            "the 51-feature standardized representation, and remove the term "
            "Fréchet Inception Distance unless a separate Inception-based image "
            "experiment is independently verified."
        )

    else:

        terminology_verdict = (
            "NO_REPRODUCIBLE_FID_OR_STRUCTURED_FRECHET_RESULT_AVAILABLE"
        )

        manuscript_action = (
            "Remove unsupported absolute FID values from the structured-data "
            "results until an actual synthetic dataset with verified provenance "
            "is available. Do not recreate 1.5, 1.6, or 3.2 from the manuscript."
        )

    terminology_row = {
        "conventional_fid_implementation_verified":
            conventional_fid_implementation_verified,

        "historical_inception_fid_evidence_file_count":
            len(
                actual_inception_code_rows
            ),

        "historical_structured_moment_frechet_evidence_file_count":
            len(
                actual_moment_frechet_rows
            ),

        "eligible_structured_synthetic_tables":
            len(
                usable_tables
            ),

        "computed_structured_frechet_results":
            computed_sfd_count,

        "potential_51_column_arrays_with_unverified_order":
            len(
                potential_unverified_arrays
            ),

        "terminology_verdict":
            terminology_verdict,

        "manuscript_action":
            manuscript_action,

        "representation_for_new_sfd":
            (
                "51 raw clinical predictors; target excluded; median "
                "imputation and StandardScaler parameters fit on real "
                "reference; same transform applied to synthetic records"
            ),

        "cross_modality_absolute_comparability_claim":
            False,
    }

    write_csv(
        OUTPUT_DIR
        / "fidelity_terminology_verdict.csv",
        [
            terminology_row
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
            "04_verify_and_compute_structured_frechet.py",

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "real_source":
            str(
                raw_path
            ),

        "real_source_type":
            raw_source_type,

        "real_source_sha256":
            sha256_file(
                raw_path
            ),

        "historical_feature_schema":
            str(
                HISTORICAL_X_TRAIN_PATH
            ),

        "historical_feature_schema_sha256":
            sha256_file(
                HISTORICAL_X_TRAIN_PATH
            ),

        "real_rows":
            len(
                raw_df
            ),

        "feature_count":
            len(
                feature_names
            ),

        "target_excluded":
            target_col,

        "representation":
            "standardized_structured_51_feature_space",

        "imputation":
            "median fitted on real reference",

        "scaling":
            "StandardScaler fitted on real reference",

        "metric":
            "Structured Frechet Distance",

        "metric_formula":
            (
                "||mu_r-mu_s||^2 + Tr(Sigma_r + Sigma_s "
                "- 2*(Sigma_r^(1/2) Sigma_s Sigma_r^(1/2))^(1/2))"
            ),

        "conventional_fid_claim_for_new_metric":
            False,

        "inception_features_used_for_new_metric":
            False,

        "synthetic_data_generated_by_script":
            False,

        "historical_values_reused_as_results":
            False,

        "historical_claims_audited":
            safe_json(
                HISTORICAL_FID_CLAIMS
            ),

        "eligible_synthetic_tables":
            len(
                usable_tables
            ),

        "computed_sfd_results":
            computed_sfd_count,

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "scipy_version":
            scipy.__version__,

        "sklearn_version":
            sklearn.__version__,

        "python_version":
            sys.version,
    }

    write_csv(
        OUTPUT_DIR
        / "fidelity_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    lines = [
        "=" * 100,
        "HFAGM - FID IMPLEMENTATION AUDIT AND STRUCTURED FRECHET RECOMPUTATION",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "REAL REFERENCE",
        "-" * 100,
        f"Source: {raw_path}",
        f"Rows: {len(raw_df)}",
        f"Predictors: {len(feature_names)}",
        f"Target excluded from fidelity representation: {target_col}",
        "",
        "NEW STRUCTURED REPRESENTATION",
        "-" * 100,
        (
            "Representation: 51 structured clinical predictors in a "
            "real-reference standardized feature space."
        ),
        (
            "Missing values: median imputation fitted on the real "
            "reference dataset."
        ),
        (
            "Scaling: StandardScaler fitted on the real reference dataset "
            "and applied unchanged to synthetic records."
        ),
        (
            "Target/outcome is excluded from the Fréchet feature vector."
        ),
        "",
        "METRIC TERMINOLOGY",
        "-" * 100,
        (
            "Metric name used by this audit: Structured Frechet Distance (SFD)."
        ),
        (
            "This is NOT called Fréchet Inception Distance because the "
            "calculation does not use Inception image features."
        ),
        (
            "Absolute distances produced in different feature representations "
            "must not be interpreted as directly comparable."
        ),
        "",
        "HISTORICAL IMPLEMENTATION AUDIT",
        "-" * 100,
        (
            f"Potential historical fidelity/FID files: "
            f"{len(fidelity_code_rows)}"
        ),
        (
            f"Files combining FID terminology with Inception evidence: "
            f"{len(actual_inception_code_rows)}"
        ),
        (
            f"Files combining FID/Fréchet terminology with moment-code "
            f"evidence: {len(actual_moment_frechet_rows)}"
        ),
        "",
        "SYNTHETIC DATA DISCOVERY",
        "-" * 100,
        (
            f"Candidate structured tables: "
            f"{len(candidate_inventory_rows)}"
        ),
        (
            f"Eligible tables containing the full verified 51-feature schema: "
            f"{len(usable_tables)}"
        ),
        (
            f"Potential 51-column array files whose feature order cannot be "
            f"verified automatically: {len(potential_unverified_arrays)}"
        ),
        "",
        "RECOMPUTED STRUCTURED FRECHET RESULTS",
        "-" * 100,
    ]

    if computed_rows:

        for result in computed_rows:

            lines.extend(
                [
                    (
                        f"Synthetic dataset: "
                        f"{result['synthetic_dataset']}"
                    ),

                    (
                        f"  n_real = {result['n_real']}; "
                        f"n_synthetic = {result['n_synthetic']}"
                    ),

                    (
                        f"  dimension = "
                        f"{result['dimension']}"
                    ),

                    (
                        f"  Structured Frechet Distance = "
                        f"{result['structured_frechet_distance']:.10f}"
                    ),

                    (
                        f"  mean term = "
                        f"{result['mean_term']:.10f}"
                    ),

                    (
                        f"  covariance term = "
                        f"{result['covariance_term']:.10f}"
                    ),

                    "",
                ]
            )

    else:

        lines.append(
            "NO REPRODUCIBLE STRUCTURED FRECHET RESULT WAS COMPUTED."
        )

        lines.append(
            (
                "Reason: no actual eligible full-schema synthetic dataset "
                "was available, or all candidate calculations failed."
            )
        )

    lines.extend(
        [
            "",
            "HISTORICAL MANUSCRIPT CLAIMS",
            "-" * 100,
        ]
    )

    for claim_name, value in HISTORICAL_FID_CLAIMS.items():

        evidence_count = sum(
            1
            for row
            in historical_value_rows
            if row[
                "claim_name"
            ] == claim_name
        )

        lines.append(
            (
                f"{claim_name}: reported value={value}; "
                f"historical context hits={evidence_count}"
            )
        )

    lines.extend(
        [
            "",
            (
                "A textual or hard-coded occurrence of 1.5, 1.6, or 3.2 "
                "does not validate the value."
            ),
            (
                "Only a value recomputed from a provenanced real/synthetic "
                "dataset pair is considered independently verified."
            ),
            "",
            "TERMINOLOGY VERDICT",
            "-" * 100,
            terminology_verdict,
            "",
            "MANUSCRIPT ACTION",
            "-" * 100,
            manuscript_action,
            "",
            "NUMERICAL SELF-CHECKS",
            "-" * 100,
        ]
    )

    for row in self_check_rows:

        lines.append(
            (
                f"{row['check']}: "
                f"value={row['value']:.12g}; "
                f"status={row['status']}"
            )
        )

    lines.extend(
        [
            "",
            "PRIMARY OUTPUTS",
            "-" * 100,
            "fidelity_code_evidence.csv",
            "historical_fidelity_value_evidence.csv",
            "synthetic_candidate_inventory.csv",
            "synthetic_candidate_feature_mapping.csv",
            "synthetic_array_candidate_inventory.csv",
            "real_feature_mapping.csv",
            "real_numeric_conversion_audit.csv",
            "synthetic_numeric_conversion_audit.csv",
            "structured_feature_space_audit.csv",
            "structured_frechet_self_checks.csv",
            "structured_frechet_results.csv",
            "covariance_diagnostics.csv",
            "historical_claim_comparison.csv",
            "fidelity_terminology_verdict.csv",
            "fidelity_provenance.csv",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "fidelity_summary.txt"
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
        "04 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nHistorical fidelity/FID evidence files: "
        f"{len(fidelity_code_rows)}"
    )

    print(
        f"FID + Inception evidence files: "
        f"{len(actual_inception_code_rows)}"
    )

    print(
        f"Structured synthetic tables eligible: "
        f"{len(usable_tables)}"
    )

    print(
        f"Structured Fréchet distances computed: "
        f"{computed_sfd_count}"
    )

    if computed_rows:

        print(
            "\nComputed SFD values:"
        )

        for result in computed_rows:

            print(
                f"  {result['synthetic_dataset']}"
            )

            print(
                f"    SFD = "
                f"{result['structured_frechet_distance']:.10f}"
            )

    print(
        "\nTerminology verdict:"
    )

    print(
        terminology_verdict
    )

    print(
        "\nManuscript action:"
    )

    print(
        manuscript_action
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
        "fidelity_summary.txt",
        "fidelity_terminology_verdict.csv",
        "fidelity_code_evidence.csv",
        "synthetic_candidate_inventory.csv",
        "structured_frechet_results.csv",
        "covariance_diagnostics.csv",
        "historical_claim_comparison.csv",
        "historical_fidelity_value_evidence.csv",
        "synthetic_array_candidate_inventory.csv",
        "fidelity_provenance.csv",
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
            "04 FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo real data, synthetic data, historical result, model, "
            "or manuscript file was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(
            1
        )