"""
04B_reconstruct_structured_synthetic_generation.py
===================================================

HFAGM - Forensic reconstruction of structured synthetic-data generation.

PURPOSE
-------
04_verify_and_compute_structured_frechet.py established that:

    - no reproducible conventional FID implementation was verified;
    - no eligible 51-feature structured synthetic table was found;
    - therefore no Structured Frechet Distance (SFD) could be recomputed.

Before deleting structured fidelity evaluation entirely, this script asks:

    CAN THE ORIGINAL STRUCTURED SYNTHETIC DATA BE REGENERATED
    FROM EXISTING PROJECT CODE + EXISTING TRAINED GENERATOR ARTIFACTS?

This script is intentionally conservative.

IT DOES:
---------
1. Reconstruct the exact 51-feature clinical schema.
2. Inventory generator/HFAGM source code.
3. Inventory generator/checkpoint/model artifacts.
4. Inspect checkpoint metadata/state-dict structure where safely possible.
5. Search code for:
       - generator classes,
       - GAN/VAE/diffusion classes,
       - forward/sample/generate methods,
       - latent/noise dimensions,
       - output dimensions,
       - torch.save / torch.load,
       - joblib/pickle save/load,
       - synthetic-data generation statements,
       - CSV/NumPy save statements,
       - inverse scaling / decoding.
6. Search for overlooked structured synthetic tables.
7. Determine whether a complete provenance chain exists:

       model architecture
           +
       trained checkpoint
           +
       generation procedure
           +
       output dimension/schema
           +
       preprocessing/inverse transform
           =
       reproducible structured generation

8. Copy an ALREADY-EXISTING verified 51-feature synthetic table into the
   revision output directory if one is found.

IT DOES NOT:
------------
- retrain a GAN, VAE, diffusion model, or HFAGM;
- initialize a new generator and pretend it is trained;
- infer missing latent dimensions;
- infer checkpoint/model pairing merely from filename similarity;
- execute training scripts;
- overwrite old project files;
- create random synthetic clinical rows;
- recreate FID/SFD values from manuscript claims;
- dynamically execute arbitrary project modules.

WHY THIS MATTERS
----------------
A saved model file by itself is not enough.

For defensible regeneration we need evidence that establishes:

    checkpoint -> architecture -> generation call -> structured output schema

If any essential link is missing, the script reports the exact gap.

PRIMARY VERDICTS
----------------
ALREADY_HAS_VERIFIED_STRUCTURED_SYNTHETIC_TABLE

REGENERATION_READY_WITH_EXISTING_CHECKPOINT_AND_EXPLICIT_GENERATION_PATH

CHECKPOINT_FOUND_BUT_GENERATION_PATH_INCOMPLETE

GENERATION_CODE_FOUND_BUT_NO_TRAINED_GENERATOR_CHECKPOINT

STRUCTURED_GENERATOR_CODE_FOUND_BUT_OUTPUT_SCHEMA_UNVERIFIED

NO_STRUCTURED_GENERATOR_PROVENANCE_FOUND

OUTPUT
------
outputs/revision_fidelity/structured_generation_reconstruction/

    structured_generation_summary.txt
    project_generation_code_inventory.csv
    generator_definition_evidence.csv
    generation_operation_evidence.csv
    checkpoint_inventory.csv
    checkpoint_structure_audit.csv
    checkpoint_code_linkage.csv
    synthetic_table_inventory.csv
    synthetic_table_feature_mapping.csv
    existing_verified_synthetic_tables.csv
    preprocessing_generation_evidence.csv
    generation_provenance_chain.csv
    generation_reconstruction_verdict.csv
    reconstruction_provenance.csv

If an existing verified table is found:
    recovered_existing_synthetic/
        <original_filename>

IMPORTANT
---------
A "READY" verdict does not itself generate new records. It means the evidence
is sufficient to write a narrowly targeted 04C regeneration script using the
specific architecture/checkpoint/generation procedure discovered here.

That separation prevents this forensic audit from accidentally retraining,
using randomly initialized models, or executing unrelated project code.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import traceback
import zipfile

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    / "structured_generation_reconstruction"
)

RECOVERED_EXISTING_DIR = (
    OUTPUT_DIR
    / "recovered_existing_synthetic"
)

EXPECTED_REAL_ROWS = 193
EXPECTED_FEATURE_COUNT = 51

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
# High-value project locations.
# -------------------------------------------------------------------------

HIGH_VALUE_PATHS = [
    PROJECT_ROOT / "models" / "generators",
    PROJECT_ROOT / "saved_models" / "generators",
    PROJECT_ROOT / "models" / "classifiers",
    PROJECT_ROOT / "training" / "hfagm",
    PROJECT_ROOT / "training" / "ensemble",
    PROJECT_ROOT / "models" / "encoders",
    PROJECT_ROOT / "saved_models" / "encoders",
    PROJECT_ROOT / "synthetic",
    PROJECT_ROOT / "experiments",
    PROJECT_ROOT / "config",
]

# -------------------------------------------------------------------------
# Model/checkpoint file extensions.
# -------------------------------------------------------------------------

CHECKPOINT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npz",
    ".npy",
    ".h5",
    ".hdf5",
    ".keras",
    ".onnx",
}

# -------------------------------------------------------------------------
# Tabular outputs potentially containing structured synthetic samples.
# -------------------------------------------------------------------------

TABULAR_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
}

TEXT_CODE_EXTENSIONS = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".txt",
    ".md",
    ".rst",
}

# -------------------------------------------------------------------------
# Exclusions.
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
    "04b_",
)

REVISION_OUTPUT_MARKERS = (
    "outputs/revision_primary_metrics",
    "outputs\\revision_primary_metrics",
    "outputs/revision_fairness",
    "outputs\\revision_fairness",
    "outputs/revision_fidelity",
    "outputs\\revision_fidelity",
)

# Avoid huge ArSL asset traversal as substantive generation evidence.
BULK_ASSET_DIR_HINTS = {
    "arsl",
    "arsl21l",
    "arsl_dataset",
    "images",
    "labels",
}

# -------------------------------------------------------------------------
# Existing files known not to be generated synthetic clinical datasets.
# -------------------------------------------------------------------------

KNOWN_NONSYNTHETIC_TABLES = {
    "covid_clinical.csv",
    "covid_clinical.xlsx",
    "covid_clinical_preprocessed.csv",
    "covid_clinical_balanced.csv",
    "x_train_scaled.csv",
    "x_test_scaled.csv",
    "y_train.csv",
    "y_test.csv",
}

# -------------------------------------------------------------------------
# Synthetic path/name hints.
# -------------------------------------------------------------------------

SYNTHETIC_HINTS = (
    "synthetic",
    "generated",
    "fake",
    "sampled",
    "samples",
    "generation",
    "generator",
    "gan",
    "vae",
    "diffusion",
    "hfagm",
)

# -------------------------------------------------------------------------
# Generator-related language.
# -------------------------------------------------------------------------

GENERATOR_CLASS_TERMS = (
    "generator",
    "gan",
    "vae",
    "variationalautoencoder",
    "diffusion",
    "denoiser",
    "decoder",
    "hfagm",
)

GENERATION_METHOD_NAMES = {
    "generate",
    "generate_samples",
    "generate_synthetic",
    "sample",
    "sampling",
    "synthesize",
    "synthesise",
    "decode",
    "reverse_diffusion",
    "p_sample",
}

MODEL_LOAD_TERMS = (
    "torch.load",
    "load_state_dict",
    "joblib.load",
    "pickle.load",
    "load_model",
)

MODEL_SAVE_TERMS = (
    "torch.save",
    "state_dict",
    "joblib.dump",
    "pickle.dump",
    "save_model",
)

SYNTHETIC_SAVE_TERMS = (
    "to_csv",
    "np.save",
    "numpy.save",
    "savez",
    "to_excel",
    "to_parquet",
)

PREPROCESSING_TERMS = (
    "standardscaler",
    "minmaxscaler",
    "fit_transform",
    "inverse_transform",
    "imputer",
    "normaliz",
    "scaler",
)

LATENT_TERMS = (
    "latent_dim",
    "z_dim",
    "noise_dim",
    "nz",
    "latent_size",
    "embedding_dim",
)

OUTPUT_DIM_TERMS = (
    "output_dim",
    "input_dim",
    "feature_dim",
    "num_features",
    "n_features",
    "data_dim",
)

# Maximum text-file size read into memory.
MAX_TEXT_FILE_BYTES = 10 * 1024 * 1024

# Maximum table file size automatically read.
MAX_TABULAR_FILE_BYTES = 100 * 1024 * 1024


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def normalize_name(value: Any) -> str:
    return "".join(
        ch
        for ch in str(value).strip().lower()
        if ch.isalnum()
    )


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

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def relative_path(path: Path) -> str:

    try:

        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

    except Exception:

        return str(
            path
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

        extra = [
            col
            for col in df.columns
            if col not in columns
        ]

        df = df[
            columns + extra
        ]

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def read_csv_robust(
    path: Path,
) -> pd.DataFrame:

    errors = []

    separator = (
        "\t"
        if path.suffix.lower() == ".tsv"
        else ","
    )

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
                sep=separator,
            )

        except Exception as exc:

            errors.append(
                f"{encoding}: {repr(exc)}"
            )

    raise RuntimeError(
        f"Could not read {path}\n"
        + "\n".join(
            errors
        )
    )


def load_table(
    path: Path,
) -> Optional[pd.DataFrame]:

    if (
        path.stat().st_size
        > MAX_TABULAR_FILE_BYTES
    ):
        return None

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


def read_text_file(
    path: Path,
) -> Optional[str]:

    try:

        if (
            path.stat().st_size
            > MAX_TEXT_FILE_BYTES
        ):

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
            continue

    return None


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

        if part in EXCLUDED_DIR_EXACT:

            return (
                True,
                f"excluded_exact:{part}",
            )

        if part in BULK_ASSET_DIR_HINTS:

            return (
                True,
                f"excluded_bulk_asset:{part}",
            )

        if any(
            part.startswith(
                prefix
            )
            for prefix
            in EXCLUDED_DIR_PREFIXES
        ):

            return (
                True,
                f"excluded_environment:{part}",
            )

        if any(
            part.endswith(
                suffix
            )
            for suffix
            in EXCLUDED_DIR_SUFFIXES
        ):

            return (
                True,
                f"excluded_package_metadata:{part}",
            )

    return (
        False,
        "",
    )


def is_revision_output(
    path: Path,
) -> bool:

    text = str(
        path
    ).lower()

    return any(
        marker.lower()
        in text

        for marker
        in REVISION_OUTPUT_MARKERS
    )


def is_revision_script(
    path: Path,
) -> bool:

    name = path.name.lower()

    return any(
        name.startswith(
            prefix.lower()
        )

        for prefix
        in REVISION_SCRIPT_PREFIXES
    )


def synthetic_hint_score(
    path: Path,
) -> int:

    text = str(
        path
    ).lower()

    return sum(
        int(
            hint in text
        )
        for hint
        in SYNTHETIC_HINTS
    )


# =============================================================================
# 3. REAL 51-FEATURE SCHEMA
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

    for (
        df,
        path,
        source_type,
    ) in candidates:

        if len(
            df
        ) == EXPECTED_REAL_ROWS:

            return (
                df.copy(),
                path,
                source_type,
            )

    if candidates:

        detail = "\n".join(
            f"{path}: {len(df)} rows"
            for df, path, _
            in candidates
        )

        raise RuntimeError(
            "Raw clinical source exists but expected "
            f"{EXPECTED_REAL_ROWS} rows were not found.\n"
            + detail
        )

    raise FileNotFoundError(
        "No usable raw COVID clinical dataset found."
    )


def identify_target_column(
    df: pd.DataFrame,
) -> str:

    normalized = {
        normalize_name(
            col
        ):
            col

        for col
        in df.columns
    }

    for candidate in TARGET_CANDIDATES:

        if candidate in df.columns:

            return candidate

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


def load_feature_schema(
) -> List[str]:

    if not HISTORICAL_X_TRAIN_PATH.exists():

        raise FileNotFoundError(
            "Historical feature schema not found:\n"
            f"{HISTORICAL_X_TRAIN_PATH}"
        )

    df = read_csv_robust(
        HISTORICAL_X_TRAIN_PATH
    )

    features = [
        str(
            col
        )
        for col
        in df.columns
    ]

    if len(
        features
    ) != EXPECTED_FEATURE_COUNT:

        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(features)}."
        )

    return features


def map_table_to_feature_schema(
    df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[
    Optional[pd.DataFrame],
    List[Dict[str, Any]],
    Dict[str, Any],
]:

    normalized_lookup = defaultdict(
        list
    )

    for col in df.columns:

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
            method = "EXACT"

        else:

            candidates = normalized_lookup[
                normalize_name(
                    feature
                )
            ]

            if len(
                candidates
            ) == 1:

                source_col = candidates[
                    0
                ]

                method = "NORMALIZED"

            elif len(
                candidates
            ) == 0:

                source_col = None
                method = "MISSING"

                missing.append(
                    feature
                )

            else:

                source_col = None
                method = "AMBIGUOUS"

                ambiguous.append(
                    {
                        "feature":
                            feature,

                        "candidates":
                            candidates,
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
            ]

    diagnostics = {
        "required_feature_count":
            len(
                feature_names
            ),

        "matched_feature_count":
            len(
                mapped
            ),

        "feature_match_fraction":
            (
                len(
                    mapped
                )
                /
                len(
                    feature_names
                )
            ),

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

    return (
        pd.DataFrame(
            mapped
        ),
        mapping_rows,
        diagnostics,
    )


# =============================================================================
# 4. PROJECT SOURCE INVENTORY
# =============================================================================

def project_source_files(
) -> List[Path]:

    files = []

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

        if (
            path.suffix.lower()
            not in TEXT_CODE_EXTENSIONS
        ):
            continue

        files.append(
            path
        )

    return sorted(
        files,
        key=lambda p:
            str(
                p
            ).lower(),
    )


def high_value_path_flag(
    path: Path,
) -> int:

    try:

        resolved = path.resolve()

    except Exception:

        resolved = path

    for root in HIGH_VALUE_PATHS:

        try:

            if root.exists():

                if (
                    resolved == root.resolve()
                    or
                    root.resolve()
                    in resolved.parents
                ):

                    return 1

        except Exception:

            pass

    return 0


# =============================================================================
# 5. PYTHON AST ANALYSIS
# =============================================================================

def ast_literal_value(
    node: ast.AST,
) -> Optional[Any]:

    try:

        return ast.literal_eval(
            node
        )

    except Exception:

        return None


def python_definition_inventory(
    path: Path,
    text: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    definitions = []
    operations = []

    try:

        tree = ast.parse(
            text,
            filename=str(
                path
            ),
        )

    except Exception as exc:

        operations.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "line":
                    np.nan,

                "evidence_type":
                    "AST_PARSE_FAILED",

                "symbol":
                    "",

                "statement":
                    "",

                "detail":
                    repr(
                        exc
                    ),
            }
        )

        return (
            definitions,
            operations,
        )

    source_lines = text.splitlines()

    # -----------------------------------------------------------------
    # Classes / methods.
    # -----------------------------------------------------------------

    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            ast.ClassDef,
        ):

            class_name = node.name

            lower_name = class_name.lower()

            generator_like = int(
                any(
                    term in lower_name
                    for term
                    in GENERATOR_CLASS_TERMS
                )
            )

            base_names = []

            for base in node.bases:

                try:

                    base_names.append(
                        ast.unparse(
                            base
                        )
                    )

                except Exception:

                    base_names.append(
                        type(
                            base
                        ).__name__
                    )

            methods = [
                child.name
                for child in node.body
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
            ]

            definitions.append(
                {
                    "file":
                        relative_path(
                            path
                        ),

                    "line":
                        getattr(
                            node,
                            "lineno",
                            np.nan,
                        ),

                    "definition_type":
                        "class",

                    "name":
                        class_name,

                    "generator_like":
                        generator_like,

                    "bases":
                        safe_json(
                            base_names
                        ),

                    "methods":
                        safe_json(
                            methods
                        ),

                    "has_forward":
                        int(
                            "forward"
                            in methods
                        ),

                    "has_generation_method":
                        int(
                            any(
                                method.lower()
                                in GENERATION_METHOD_NAMES
                                for method
                                in methods
                            )
                        ),

                    "high_value_path":
                        high_value_path_flag(
                            path
                        ),
                }
            )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            function_name = node.name

            lower_name = function_name.lower()

            generation_like = int(
                lower_name
                in GENERATION_METHOD_NAMES
                or
                any(
                    token
                    in lower_name
                    for token
                    in [
                        "generate",
                        "synthetic",
                        "sample",
                        "diffusion",
                        "decode",
                    ]
                )
            )

            if generation_like:

                arguments = [
                    arg.arg
                    for arg
                    in node.args.args
                ]

                definitions.append(
                    {
                        "file":
                            relative_path(
                                path
                            ),

                        "line":
                            getattr(
                                node,
                                "lineno",
                                np.nan,
                            ),

                        "definition_type":
                            "function_or_method",

                        "name":
                            function_name,

                        "generator_like":
                            1,

                        "bases":
                            "",

                        "methods":
                            "",

                        "arguments":
                            safe_json(
                                arguments
                            ),

                        "has_forward":
                            0,

                        "has_generation_method":
                            1,

                        "high_value_path":
                            high_value_path_flag(
                                path
                            ),
                    }
                )

    # -----------------------------------------------------------------
    # Assignments with latent/output dimensions.
    # -----------------------------------------------------------------

    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):

            if isinstance(
                node,
                ast.Assign,
            ):

                targets = node.targets
                value_node = node.value

            else:

                targets = [
                    node.target
                ]

                value_node = node.value

            for target in targets:

                try:

                    target_text = ast.unparse(
                        target
                    )

                except Exception:

                    target_text = ""

                normalized_target = (
                    target_text
                    .lower()
                    .replace(
                        "self.",
                        "",
                    )
                )

                literal_value = (
                    ast_literal_value(
                        value_node
                    )
                    if value_node
                    is not None
                    else None
                )

                if any(
                    term
                    in normalized_target
                    for term
                    in LATENT_TERMS
                ):

                    operations.append(
                        {
                            "file":
                                relative_path(
                                    path
                                ),

                            "line":
                                getattr(
                                    node,
                                    "lineno",
                                    np.nan,
                                ),

                            "evidence_type":
                                "LATENT_DIMENSION_ASSIGNMENT",

                            "symbol":
                                target_text,

                            "statement":
                                safe_source_line(
                                    source_lines,
                                    node,
                                ),

                            "detail":
                                safe_json(
                                    literal_value
                                ),
                        }
                    )

                if any(
                    term
                    in normalized_target
                    for term
                    in OUTPUT_DIM_TERMS
                ):

                    operations.append(
                        {
                            "file":
                                relative_path(
                                    path
                                ),

                            "line":
                                getattr(
                                    node,
                                    "lineno",
                                    np.nan,
                                ),

                            "evidence_type":
                                "OUTPUT_OR_FEATURE_DIMENSION_ASSIGNMENT",

                            "symbol":
                                target_text,

                            "statement":
                                safe_source_line(
                                    source_lines,
                                    node,
                                ),

                            "detail":
                                safe_json(
                                    literal_value
                                ),
                        }
                    )

    # -----------------------------------------------------------------
    # Calls.
    # -----------------------------------------------------------------

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        try:

            call_name = ast.unparse(
                node.func
            )

        except Exception:

            call_name = ""

        lower_call = call_name.lower()

        evidence_type = None

        if any(
            term
            in lower_call
            for term
            in MODEL_LOAD_TERMS
        ):

            evidence_type = (
                "MODEL_OR_CHECKPOINT_LOAD"
            )

        elif any(
            term
            in lower_call
            for term
            in MODEL_SAVE_TERMS
        ):

            evidence_type = (
                "MODEL_OR_CHECKPOINT_SAVE"
            )

        elif any(
            term
            in lower_call
            for term
            in SYNTHETIC_SAVE_TERMS
        ):

            evidence_type = (
                "DATA_SAVE_OPERATION"
            )

        elif any(
            term
            in lower_call
            for term
            in PREPROCESSING_TERMS
        ):

            evidence_type = (
                "PREPROCESSING_OPERATION"
            )

        elif any(
            token
            in lower_call
            for token
            in [
                "randn",
                "normal",
                "noise",
            ]
        ):

            evidence_type = (
                "NOISE_OR_LATENT_SAMPLING"
            )

        elif (
            lower_call.split(
                "."
            )[-1]
            in GENERATION_METHOD_NAMES
        ):

            evidence_type = (
                "GENERATION_METHOD_CALL"
            )

        if evidence_type is None:
            continue

        positional_literals = []

        for arg in node.args:

            literal = ast_literal_value(
                arg
            )

            if literal is not None:

                positional_literals.append(
                    literal
                )

        keyword_literals = {}

        for keyword in node.keywords:

            if keyword.arg is None:
                continue

            literal = ast_literal_value(
                keyword.value
            )

            if literal is not None:

                keyword_literals[
                    keyword.arg
                ] = literal

        operations.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "line":
                    getattr(
                        node,
                        "lineno",
                        np.nan,
                    ),

                "evidence_type":
                    evidence_type,

                "symbol":
                    call_name,

                "statement":
                    safe_source_line(
                        source_lines,
                        node,
                    ),

                "detail":
                    safe_json(
                        {
                            "positional_literals":
                                positional_literals,

                            "keyword_literals":
                                keyword_literals,
                        }
                    ),

                "high_value_path":
                    high_value_path_flag(
                        path
                    ),
            }
        )

    return (
        definitions,
        operations,
    )


def safe_source_line(
    source_lines: List[str],
    node: ast.AST,
) -> str:

    lineno = getattr(
        node,
        "lineno",
        None,
    )

    if lineno is None:
        return ""

    index = lineno - 1

    if (
        index < 0
        or
        index >= len(
            source_lines
        )
    ):

        return ""

    return source_lines[
        index
    ].strip()[:1000]


# =============================================================================
# 6. TEXTUAL GENERATION EVIDENCE
# =============================================================================

def textual_generation_inventory(
    source_files: List[Path],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    file_inventory = []
    definition_rows = []
    operation_rows = []

    for path in source_files:

        text = read_text_file(
            path
        )

        if not text:
            continue

        lower = text.lower()

        generator_terms_found = sorted(
            {
                term
                for term
                in GENERATOR_CLASS_TERMS
                if term
                in lower
            }
        )

        generation_terms_found = sorted(
            {
                term
                for term
                in [
                    "generate",
                    "synthetic",
                    "sample",
                    "noise",
                    "latent",
                    "decoder",
                    "diffusion",
                    "generator",
                    "vae",
                    "gan",
                ]
                if term
                in lower
            }
        )

        load_terms_found = sorted(
            {
                term
                for term
                in MODEL_LOAD_TERMS
                if term
                in lower
            }
        )

        save_terms_found = sorted(
            {
                term
                for term
                in MODEL_SAVE_TERMS
                if term
                in lower
            }
        )

        preprocessing_found = sorted(
            {
                term
                for term
                in PREPROCESSING_TERMS
                if term
                in lower
            }
        )

        generation_related = bool(
            generator_terms_found
            or
            generation_terms_found
            or
            load_terms_found
            or
            save_terms_found
        )

        if not generation_related:
            continue

        file_inventory.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "suffix":
                    path.suffix.lower(),

                "size_bytes":
                    path.stat().st_size,

                "modified_time":
                    datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).isoformat(
                        timespec="seconds"
                    ),

                "high_value_path":
                    high_value_path_flag(
                        path
                    ),

                "generator_terms":
                    safe_json(
                        generator_terms_found
                    ),

                "generation_terms":
                    safe_json(
                        generation_terms_found
                    ),

                "model_load_terms":
                    safe_json(
                        load_terms_found
                    ),

                "model_save_terms":
                    safe_json(
                        save_terms_found
                    ),

                "preprocessing_terms":
                    safe_json(
                        preprocessing_found
                    ),
            }
        )

        if path.suffix.lower() == ".py":

            (
                definitions,
                operations,
            ) = python_definition_inventory(
                path,
                text,
            )

            definition_rows.extend(
                definitions
            )

            operation_rows.extend(
                operations
            )

    return (
        file_inventory,
        definition_rows,
        operation_rows,
    )


# =============================================================================
# 7. CHECKPOINT INVENTORY
# =============================================================================

def checkpoint_priority(
    path: Path,
) -> str:

    text = str(
        path
    ).lower()

    if (
        "saved_models"
        in text
        and
        "generator"
        in text
    ):

        return (
            "VERY_HIGH"
        )

    if (
        "generator"
        in text
        or
        "hfagm"
        in text
        or
        "gan"
        in text
        or
        "vae"
        in text
        or
        "diffusion"
        in text
    ):

        return (
            "HIGH"
        )

    if (
        "checkpoint"
        in text
        or
        "model"
        in text
    ):

        return (
            "MODERATE"
        )

    return (
        "LOW"
    )


def inventory_checkpoints(
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

        if (
            path.suffix.lower()
            not in CHECKPOINT_EXTENSIONS
        ):
            continue

        stat = path.stat()

        rows.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "extension":
                    path.suffix.lower(),

                "size_bytes":
                    stat.st_size,

                "modified_time":
                    datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(
                        timespec="seconds"
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),

                "priority":
                    checkpoint_priority(
                        path
                    ),

                "generator_name_hint":
                    int(
                        any(
                            term
                            in path.name.lower()
                            for term
                            in [
                                "generator",
                                "gan",
                                "vae",
                                "diffusion",
                                "hfagm",
                            ]
                        )
                    ),

                "high_value_path":
                    high_value_path_flag(
                        path
                    ),
            }
        )

    return rows


# =============================================================================
# 8. CHECKPOINT STRUCTURE AUDIT
# =============================================================================

def inspect_numpy_checkpoint(
    path: Path,
) -> List[Dict[str, Any]]:

    rows = []

    try:

        if path.suffix.lower() == ".npy":

            arr = np.load(
                path,
                allow_pickle=False,
            )

            rows.append(
                {
                    "file":
                        relative_path(
                            path
                        ),

                    "inspection_method":
                        "numpy_load_allow_pickle_false",

                    "object_type":
                        "ndarray",

                    "key":
                        "array",

                    "shape":
                        safe_json(
                            tuple(
                                arr.shape
                            )
                        ),

                    "dtype":
                        str(
                            arr.dtype
                        ),

                    "possible_output_dimension_51":
                        int(
                            arr.ndim >= 1
                            and
                            51
                            in arr.shape
                        ),

                    "status":
                        "INSPECTED",
                }
            )

        else:

            archive = np.load(
                path,
                allow_pickle=False,
            )

            for key in archive.files:

                arr = archive[
                    key
                ]

                rows.append(
                    {
                        "file":
                            relative_path(
                                path
                            ),

                        "inspection_method":
                            "numpy_npz_allow_pickle_false",

                        "object_type":
                            "ndarray",

                        "key":
                            key,

                        "shape":
                            safe_json(
                                tuple(
                                    arr.shape
                                )
                            ),

                        "dtype":
                            str(
                                arr.dtype
                            ),

                        "possible_output_dimension_51":
                            int(
                                arr.ndim >= 1
                                and
                                51
                                in arr.shape
                            ),

                        "status":
                            "INSPECTED",
                    }
                )

    except Exception as exc:

        rows.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "inspection_method":
                    "numpy",

                "object_type":
                    "",

                "key":
                    "",

                "shape":
                    "",

                "dtype":
                    "",

                "possible_output_dimension_51":
                    0,

                "status":
                    "INSPECTION_FAILED",

                "error":
                    repr(
                        exc
                    ),
            }
        )

    return rows


def inspect_torch_checkpoint(
    path: Path,
) -> List[Dict[str, Any]]:

    rows = []

    try:

        import torch

    except Exception as exc:

        return [
            {
                "file":
                    relative_path(
                        path
                    ),

                "inspection_method":
                    "torch_unavailable",

                "status":
                    "NOT_INSPECTED",

                "error":
                    repr(
                        exc
                    ),
            }
        ]

    # -----------------------------------------------------------------
    # Prefer weights_only=True when supported.
    # This avoids general pickle deserialization.
    # -----------------------------------------------------------------

    try:

        try:

            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            method = (
                "torch.load(weights_only=True)"
            )

        except TypeError:

            # Older torch versions may not support weights_only.
            return [
                {
                    "file":
                        relative_path(
                            path
                        ),

                    "inspection_method":
                        "torch_weights_only_not_supported",

                    "status":
                        "NOT_INSPECTED_FOR_SAFETY",

                    "error":
                        (
                            "Installed torch does not support "
                            "weights_only=True."
                        ),
                }
            ]

        except Exception as exc:

            return [
                {
                    "file":
                        relative_path(
                            path
                        ),

                    "inspection_method":
                        "torch.load(weights_only=True)",

                    "status":
                        "INSPECTION_FAILED",

                    "error":
                        repr(
                            exc
                        ),
                }
            ]

        # -----------------------------------------------------------------
        # Identify state dict.
        # -----------------------------------------------------------------

        candidate_dicts = []

        if isinstance(
            obj,
            dict,
        ):

            tensor_like_values = sum(
                int(
                    hasattr(
                        value,
                        "shape",
                    )
                )
                for value
                in obj.values()
            )

            if tensor_like_values > 0:

                candidate_dicts.append(
                    (
                        "root",
                        obj,
                    )
                )

            for key, value in obj.items():

                if isinstance(
                    value,
                    dict,
                ):

                    tensor_count = sum(
                        int(
                            hasattr(
                                inner,
                                "shape",
                            )
                        )
                        for inner
                        in value.values()
                    )

                    if tensor_count > 0:

                        candidate_dicts.append(
                            (
                                str(
                                    key
                                ),
                                value,
                            )
                        )

        if not candidate_dicts:

            rows.append(
                {
                    "file":
                        relative_path(
                            path
                    ),

                    "inspection_method":
                        method,

                    "object_type":
                        type(
                            obj
                        ).__name__,

                    "key":
                        "",

                    "shape":
                        "",

                    "dtype":
                        "",

                    "possible_output_dimension_51":
                        0,

                    "status":
                        "LOADED_BUT_NO_STATE_DICT_IDENTIFIED",

                    "top_level_keys":
                        (
                            safe_json(
                                list(
                                    obj.keys()
                                )[:100]
                            )
                            if isinstance(
                                obj,
                                dict,
                            )
                            else ""
                        ),
                }
            )

            return rows

        for container_name, state in candidate_dicts:

            for key, value in state.items():

                if not hasattr(
                    value,
                    "shape",
                ):
                    continue

                try:

                    shape = tuple(
                        value.shape
                    )

                except Exception:

                    shape = ()

                possible_51 = int(
                    51
                    in shape
                )

                possible_generator_output_layer = int(
                    possible_51
                    and
                    any(
                        term
                        in str(
                            key
                        ).lower()
                        for term
                        in [
                            "generator",
                            "decoder",
                            "output",
                            "fc",
                            "linear",
                        ]
                    )
                )

                rows.append(
                    {
                        "file":
                            relative_path(
                                path
                            ),

                        "inspection_method":
                            method,

                        "object_type":
                            "state_dict_tensor",

                        "container":
                            container_name,

                        "key":
                            str(
                                key
                            ),

                        "shape":
                            safe_json(
                                shape
                            ),

                        "dtype":
                            str(
                                getattr(
                                    value,
                                    "dtype",
                                    "",
                                )
                            ),

                        "possible_output_dimension_51":
                            possible_51,

                        "possible_generator_output_layer":
                            possible_generator_output_layer,

                        "status":
                            "INSPECTED",
                    }
                )

    except Exception as exc:

        rows.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "inspection_method":
                    "torch",

                "status":
                    "INSPECTION_FAILED",

                "error":
                    repr(
                        exc
                    ),
            }
        )

    return rows


def inspect_checkpoint_structures(
    checkpoint_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    rows = []

    for checkpoint in checkpoint_rows:

        path = (
            PROJECT_ROOT
            / checkpoint[
                "file"
            ]
        )

        suffix = path.suffix.lower()

        if suffix in {
            ".npy",
            ".npz",
        }:

            rows.extend(
                inspect_numpy_checkpoint(
                    path
                )
            )

        elif suffix in {
            ".pt",
            ".pth",
            ".ckpt",
            ".bin",
        }:

            rows.extend(
                inspect_torch_checkpoint(
                    path
                )
            )

        elif suffix in {
            ".pkl",
            ".pickle",
            ".joblib",
        }:

            # Do not unpickle arbitrary serialized Python objects here.
            rows.append(
                {
                    "file":
                        relative_path(
                            path
                        ),

                    "inspection_method":
                        "not_unpickled_for_forensic_safety",

                    "object_type":
                        "",

                    "key":
                        "",

                    "shape":
                        "",

                    "dtype":
                        "",

                    "possible_output_dimension_51":
                        0,

                    "status":
                        (
                            "SERIALIZED_PYTHON_OBJECT_PRESENT_"
                            "REQUIRES_KNOWN_CLASS_FOR_SAFE_USE"
                        ),
                }
            )

        else:

            rows.append(
                {
                    "file":
                        relative_path(
                            path
                        ),

                    "inspection_method":
                        "metadata_only",

                    "status":
                        "FORMAT_NOT_AUTOMATICALLY_INSPECTED",
                }
            )

    return rows


# =============================================================================
# 9. CHECKPOINT <-> CODE LINKAGE
# =============================================================================

def extract_filename_literals(
    text: str,
) -> List[str]:

    patterns = [
        r"""["']([^"']+\.(?:pt|pth|ckpt|pkl|pickle|joblib|bin|h5|keras|onnx))["']""",
    ]

    results = []

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            results.append(
                match.group(
                    1
                )
            )

    return sorted(
        set(
            results
        )
    )


def link_checkpoints_to_code(
    checkpoint_rows: List[Dict[str, Any]],
    source_files: List[Path],
) -> List[Dict[str, Any]]:

    code_cache = {}

    for path in source_files:

        text = read_text_file(
            path
        )

        if text:

            code_cache[
                path
            ] = text

    linkage = []

    for checkpoint in checkpoint_rows:

        checkpoint_rel = checkpoint[
            "file"
        ]

        checkpoint_name = Path(
            checkpoint_rel
        ).name

        checkpoint_stem = Path(
            checkpoint_rel
        ).stem.lower()

        found_links = 0

        for code_path, text in code_cache.items():

            lower = text.lower()

            reasons = []

            if checkpoint_name.lower() in lower:

                reasons.append(
                    "EXACT_CHECKPOINT_FILENAME_REFERENCED"
                )

            literal_names = extract_filename_literals(
                text
            )

            if checkpoint_name in [
                Path(
                    item
                ).name
                for item
                in literal_names
            ]:

                reasons.append(
                    "CHECKPOINT_LITERAL_REFERENCED"
                )

            # Weak stem link only when generator-related terms also present.
            if (
                checkpoint_stem
                and
                len(
                    checkpoint_stem
                ) >= 5
                and
                checkpoint_stem
                in lower
                and
                any(
                    term
                    in lower
                    for term
                    in [
                        "generator",
                        "hfagm",
                        "gan",
                        "vae",
                        "diffusion",
                    ]
                )
            ):

                reasons.append(
                    "CHECKPOINT_STEM_WITH_GENERATOR_CONTEXT"
                )

            if reasons:

                found_links += 1

                linkage.append(
                    {
                        "checkpoint":
                            checkpoint_rel,

                        "code_file":
                            relative_path(
                                code_path
                            ),

                        "link_strength":
                            (
                                "STRONG"
                                if (
                                    "EXACT_CHECKPOINT_FILENAME_REFERENCED"
                                    in reasons
                                    or
                                    "CHECKPOINT_LITERAL_REFERENCED"
                                    in reasons
                                )
                                else "WEAK"
                            ),

                        "link_reasons":
                            safe_json(
                                sorted(
                                    set(
                                        reasons
                                    )
                                )
                            ),
                    }
                )

        if found_links == 0:

            linkage.append(
                {
                    "checkpoint":
                        checkpoint_rel,

                    "code_file":
                        "",

                    "link_strength":
                        "NONE",

                    "link_reasons":
                        "[]",
                }
            )

    return linkage


# =============================================================================
# 10. SYNTHETIC TABLE DISCOVERY
# =============================================================================

def inspect_synthetic_tables(
    feature_names: List[str],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    inventory = []
    mapping_rows_all = []
    verified_rows = []

    required_normalized = {
        normalize_name(
            feature
        )
        for feature
        in feature_names
    }

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

        if (
            path.suffix.lower()
            not in TABULAR_EXTENSIONS
        ):
            continue

        if (
            path.name.lower()
            in KNOWN_NONSYNTHETIC_TABLES
        ):
            continue

        lower_path = str(
            path
        ).lower()

        # Exclude obvious results/metrics tables.
        if any(
            token
            in lower_path
            for token
            in [
                "metric",
                "summary",
                "confusion",
                "prediction",
                "fairness",
                "audit",
                "registry",
                "claim_mapping",
                "provenance",
            ]
        ):

            continue

        hint_score = synthetic_hint_score(
            path
        )

        df = load_table(
            path
        )

        if df is None:
            continue

        normalized_columns = {
            normalize_name(
                col
            )
            for col
            in df.columns
        }

        schema_overlap = len(
            required_normalized
            &
            normalized_columns
        )

        overlap_fraction = (
            schema_overlap
            /
            EXPECTED_FEATURE_COUNT
        )

        # Retain if filename/path suggests synthetic OR schema is highly similar.
        if (
            hint_score == 0
            and
            overlap_fraction < 0.70
        ):

            continue

        (
            mapped,
            mapping_rows,
            diagnostics,
        ) = map_table_to_feature_schema(
            df,
            feature_names,
        )

        relative = relative_path(
            path
        )

        for mapping_row in mapping_rows:

            mapping_rows_all.append(
                {
                    "source_table":
                        relative,

                    **mapping_row,
                }
            )

        fully_mapped = (
            mapped is not None
            and
            diagnostics[
                "matched_feature_count"
            ]
            == EXPECTED_FEATURE_COUNT
        )

        # -----------------------------------------------------------------
        # Distinguish "schema-compatible" from "verified synthetic".
        #
        # A table is called VERIFIED only if:
        #   - full 51-feature schema is present
        #   - path/name has explicit synthetic/generation hint
        #
        # Merely resembling the real data schema is insufficient.
        # -----------------------------------------------------------------

        verified_synthetic = bool(
            fully_mapped
            and
            hint_score > 0
        )

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

                "synthetic_hint_score":
                    hint_score,

                "matched_feature_count":
                    diagnostics[
                        "matched_feature_count"
                    ],

                "feature_match_fraction":
                    diagnostics[
                        "feature_match_fraction"
                    ],

                "missing_features":
                    safe_json(
                        diagnostics[
                            "missing_features"
                        ]
                    ),

                "ambiguous_features":
                    safe_json(
                        diagnostics[
                            "ambiguous_features"
                        ]
                    ),

                "full_51_feature_schema":
                    int(
                        fully_mapped
                    ),

                "verified_existing_structured_synthetic":
                    int(
                        verified_synthetic
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

        if verified_synthetic:

            destination = (
                RECOVERED_EXISTING_DIR
                / path.name
            )

            RECOVERED_EXISTING_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Avoid overwrite; add hash suffix if needed.
            if destination.exists():

                short_hash = sha256_file(
                    path
                )[:10]

                destination = (
                    RECOVERED_EXISTING_DIR
                    /
                    (
                        path.stem
                        + "_"
                        + short_hash
                        + path.suffix
                    )
                )

            shutil.copy2(
                path,
                destination,
            )

            verified_rows.append(
                {
                    "source_file":
                        relative,

                    "source_sha256":
                        sha256_file(
                            path
                        ),

                    "rows":
                        len(
                            df
                        ),

                    "feature_count":
                        EXPECTED_FEATURE_COUNT,

                    "copied_to":
                        str(
                            destination
                        ),

                    "copy_sha256":
                        sha256_file(
                            destination
                        ),

                    "status":
                        "COPIED_EXISTING_FILE_NO_GENERATION_PERFORMED",
                }
            )

    return (
        inventory,
        mapping_rows_all,
        verified_rows,
    )


# =============================================================================
# 11. PREPROCESSING / GENERATION EVIDENCE
# =============================================================================

def extract_preprocessing_generation_rows(
    operation_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    selected = []

    relevant_types = {
        "PREPROCESSING_OPERATION",
        "DATA_SAVE_OPERATION",
        "NOISE_OR_LATENT_SAMPLING",
        "GENERATION_METHOD_CALL",
        "LATENT_DIMENSION_ASSIGNMENT",
        "OUTPUT_OR_FEATURE_DIMENSION_ASSIGNMENT",
        "MODEL_OR_CHECKPOINT_LOAD",
        "MODEL_OR_CHECKPOINT_SAVE",
    }

    for row in operation_rows:

        if (
            row.get(
                "evidence_type"
            )
            in relevant_types
        ):

            selected.append(
                row
            )

    return selected


# =============================================================================
# 12. PROVENANCE CHAIN SCORING
# =============================================================================

def contains_generator_definition(
    definitions: List[Dict[str, Any]],
) -> bool:

    return any(
        row.get(
            "generator_like"
        ) == 1

        for row
        in definitions
    )


def contains_explicit_generation_method(
    definitions: List[Dict[str, Any]],
    operations: List[Dict[str, Any]],
) -> bool:

    definition_evidence = any(
        row.get(
            "has_generation_method"
        ) == 1

        for row
        in definitions
    )

    operation_evidence = any(
        row.get(
            "evidence_type"
        )
        == "GENERATION_METHOD_CALL"

        for row
        in operations
    )

    # Forward-only generator can also be usable if noise sampling exists.
    forward_generator = any(
        (
            row.get(
                "definition_type"
            )
            == "class"
            and
            row.get(
                "generator_like"
            )
            == 1
            and
            row.get(
                "has_forward"
            )
            == 1
        )

        for row
        in definitions
    )

    latent_sampling = any(
        row.get(
            "evidence_type"
        )
        == "NOISE_OR_LATENT_SAMPLING"

        for row
        in operations
    )

    return bool(
        definition_evidence
        or
        operation_evidence
        or
        (
            forward_generator
            and
            latent_sampling
        )
    )


def output_dimension_51_evidence(
    definitions: List[Dict[str, Any]],
    operations: List[Dict[str, Any]],
    checkpoint_structure_rows: List[Dict[str, Any]],
) -> Tuple[
    bool,
    List[str],
]:

    reasons = []

    for row in operations:

        if (
            row.get(
                "evidence_type"
            )
            != "OUTPUT_OR_FEATURE_DIMENSION_ASSIGNMENT"
        ):
            continue

        detail = str(
            row.get(
                "detail",
                "",
            )
        )

        if re.search(
            r"(^|[^0-9])51([^0-9]|$)",
            detail,
        ):

            reasons.append(
                (
                    f"CODE:{row.get('file')}:"
                    f"{row.get('line')}"
                )
            )

    for row in checkpoint_structure_rows:

        if (
            row.get(
                "possible_generator_output_layer",
                0,
            )
            == 1
        ):

            reasons.append(
                (
                    f"CHECKPOINT:{row.get('file')}:"
                    f"{row.get('key')}"
                )
            )

    return (
        bool(
            reasons
        ),
        reasons,
    )


def checkpoint_generation_candidates(
    checkpoint_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    candidates = []

    for row in checkpoint_rows:

        priority = row.get(
            "priority",
            "LOW",
        )

        if (
            row.get(
                "generator_name_hint",
                0,
            )
            == 1
            or
            priority
            in {
                "VERY_HIGH",
                "HIGH",
            }
        ):

            candidates.append(
                row
            )

    return candidates


def build_provenance_chain(
    definitions: List[Dict[str, Any]],
    operations: List[Dict[str, Any]],
    checkpoints: List[Dict[str, Any]],
    checkpoint_structures: List[Dict[str, Any]],
    checkpoint_links: List[Dict[str, Any]],
    verified_tables: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:

    generator_definition = contains_generator_definition(
        definitions
    )

    generation_method = contains_explicit_generation_method(
        definitions,
        operations,
    )

    generator_checkpoints = checkpoint_generation_candidates(
        checkpoints
    )

    trained_checkpoint = bool(
        generator_checkpoints
    )

    strong_checkpoint_link = any(
        row.get(
            "link_strength"
        )
        == "STRONG"

        for row
        in checkpoint_links
    )

    (
        output_51,
        output_51_reasons,
    ) = output_dimension_51_evidence(
        definitions,
        operations,
        checkpoint_structures,
    )

    latent_dimension_evidence = any(
        row.get(
            "evidence_type"
        )
        == "LATENT_DIMENSION_ASSIGNMENT"

        for row
        in operations
    )

    model_load_evidence = any(
        row.get(
            "evidence_type"
        )
        == "MODEL_OR_CHECKPOINT_LOAD"

        for row
        in operations
    )

    model_save_evidence = any(
        row.get(
            "evidence_type"
        )
        == "MODEL_OR_CHECKPOINT_SAVE"

        for row
        in operations
    )

    preprocessing_evidence = any(
        row.get(
            "evidence_type"
        )
        == "PREPROCESSING_OPERATION"

        for row
        in operations
    )

    synthetic_save_evidence = any(
        row.get(
            "evidence_type"
        )
        == "DATA_SAVE_OPERATION"

        for row
        in operations
    )

    existing_verified_table = bool(
        verified_tables
    )

    criteria = [
        (
            "generator_architecture_definition",
            generator_definition,
            (
                "Generator-like class/function found."
                if generator_definition
                else
                "No generator-like class/function found."
            ),
        ),
        (
            "explicit_generation_procedure",
            generation_method,
            (
                "Generation/sample procedure found."
                if generation_method
                else
                "No explicit generation/sample procedure found."
            ),
        ),
        (
            "trained_generator_checkpoint_candidate",
            trained_checkpoint,
            (
                f"{len(generator_checkpoints)} generator-related "
                "checkpoint candidate(s) found."
                if trained_checkpoint
                else
                "No generator-related checkpoint candidate found."
            ),
        ),
        (
            "checkpoint_explicitly_linked_to_code",
            strong_checkpoint_link,
            (
                "At least one checkpoint filename is explicitly linked "
                "to code."
                if strong_checkpoint_link
                else
                "No strong checkpoint-to-code filename linkage found."
            ),
        ),
        (
            "51_feature_output_evidence",
            output_51,
            (
                "51-dimensional output evidence found: "
                + safe_json(
                    output_51_reasons
                )
                if output_51
                else
                "No verified 51-dimensional generator output evidence."
            ),
        ),
        (
            "latent_or_noise_dimension_evidence",
            latent_dimension_evidence,
            (
                "Latent/noise dimension evidence found."
                if latent_dimension_evidence
                else
                "No explicit latent/noise dimension evidence."
            ),
        ),
        (
            "checkpoint_load_evidence",
            model_load_evidence,
            (
                "Checkpoint load operation found."
                if model_load_evidence
                else
                "No checkpoint load operation found."
            ),
        ),
        (
            "checkpoint_save_evidence",
            model_save_evidence,
            (
                "Checkpoint save operation found."
                if model_save_evidence
                else
                "No checkpoint save operation found."
            ),
        ),
        (
            "preprocessing_or_inverse_transform_evidence",
            preprocessing_evidence,
            (
                "Preprocessing/scaling evidence found."
                if preprocessing_evidence
                else
                "No preprocessing/inverse-transform evidence found."
            ),
        ),
        (
            "synthetic_output_save_evidence",
            synthetic_save_evidence,
            (
                "Potential generated-data save operation found."
                if synthetic_save_evidence
                else
                "No generated-data save operation found."
            ),
        ),
        (
            "existing_verified_51_feature_synthetic_table",
            existing_verified_table,
            (
                f"{len(verified_tables)} verified existing table(s) found."
                if existing_verified_table
                else
                "No verified existing 51-feature synthetic table found."
            ),
        ),
    ]

    chain_rows = []

    for (
        criterion,
        passed,
        detail,
    ) in criteria:

        chain_rows.append(
            {
                "criterion":
                    criterion,

                "passed":
                    int(
                        passed
                    ),

                "detail":
                    detail,
            }
        )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    if existing_verified_table:

        verdict = (
            "ALREADY_HAS_VERIFIED_STRUCTURED_SYNTHETIC_TABLE"
        )

        manuscript_action = (
            "Use the recovered existing table as the candidate provenance "
            "source for rerunning 04 SFD, after manually confirming that "
            "the table was actually generated by the intended HGF/baseline "
            "condition rather than merely stored under a synthetic path."
        )

        next_action = (
            "MANUAL_PROVENANCE_CONFIRMATION_THEN_RERUN_04"
        )

    elif (
        generator_definition
        and
        generation_method
        and
        trained_checkpoint
        and
        strong_checkpoint_link
        and
        output_51
        and
        latent_dimension_evidence
        and
        model_load_evidence
    ):

        verdict = (
            "REGENERATION_READY_WITH_EXISTING_CHECKPOINT_"
            "AND_EXPLICIT_GENERATION_PATH"
        )

        manuscript_action = (
            "Do not report fidelity yet. Write a narrow 04C script using "
            "only the specific architecture/checkpoint/generation path "
            "identified by this audit, generate a versioned 51-feature "
            "synthetic table, and then rerun 04 to compute SFD."
        )

        next_action = (
            "WRITE_04C_EXACT_REGENERATION_SCRIPT"
        )

    elif (
        trained_checkpoint
        and
        (
            not generation_method
            or
            not generator_definition
            or
            not strong_checkpoint_link
            or
            not output_51
        )
    ):

        verdict = (
            "CHECKPOINT_FOUND_BUT_GENERATION_PATH_INCOMPLETE"
        )

        manuscript_action = (
            "Do not generate synthetic records yet. The checkpoint exists "
            "but architecture/generation/schema provenance is incomplete. "
            "Resolve the missing links manually before any regeneration."
        )

        next_action = (
            "INSPECT_CHECKPOINT_AND_ASSOCIATED_CODE_MANUALLY"
        )

    elif (
        generator_definition
        and
        generation_method
        and
        not trained_checkpoint
    ):

        verdict = (
            "GENERATION_CODE_FOUND_BUT_NO_TRAINED_GENERATOR_CHECKPOINT"
        )

        manuscript_action = (
            "The project contains generation code but no verified trained "
            "generator artifact. Do not initialize a new generator merely "
            "to reproduce old fidelity claims. Remove unsupported historical "
            "FID/SFD values unless a trained artifact can be recovered."
        )

        next_action = (
            "SEARCH_EXTERNAL_BACKUP_OR_REMOVE_FID_CLAIMS"
        )

    elif (
        generator_definition
        and
        not output_51
    ):

        verdict = (
            "STRUCTURED_GENERATOR_CODE_FOUND_BUT_OUTPUT_SCHEMA_UNVERIFIED"
        )

        manuscript_action = (
            "Generator-like code exists, but there is insufficient evidence "
            "that it generates the 51-feature clinical representation. "
            "Do not use it for structured fidelity without verifying the "
            "output schema."
        )

        next_action = (
            "MANUALLY_INSPECT_GENERATOR_OUTPUT_SCHEMA"
        )

    else:

        verdict = (
            "NO_STRUCTURED_GENERATOR_PROVENANCE_FOUND"
        )

        manuscript_action = (
            "No defensible structured generation provenance chain was "
            "recovered. Remove unsupported absolute FID/SFD claims from "
            "the structured experiment."
        )

        next_action = (
            "REMOVE_UNSUPPORTED_FID_SFD_CLAIMS"
        )

    verdict_row = {
        "verdict":
            verdict,

        "next_action":
            next_action,

        "manuscript_action":
            manuscript_action,

        "generator_definition_found":
            int(
                generator_definition
            ),

        "explicit_generation_procedure_found":
            int(
                generation_method
            ),

        "generator_checkpoint_candidates":
            len(
                generator_checkpoints
            ),

        "strong_checkpoint_code_link_found":
            int(
                strong_checkpoint_link
            ),

        "output_dimension_51_evidence":
            int(
                output_51
            ),

        "output_dimension_51_evidence_detail":
            safe_json(
                output_51_reasons
            ),

        "latent_dimension_evidence":
            int(
                latent_dimension_evidence
            ),

        "checkpoint_load_evidence":
            int(
                model_load_evidence
            ),

        "checkpoint_save_evidence":
            int(
                model_save_evidence
            ),

        "preprocessing_evidence":
            int(
                preprocessing_evidence
            ),

        "synthetic_save_evidence":
            int(
                synthetic_save_evidence
            ),

        "existing_verified_synthetic_tables":
            len(
                verified_tables
            ),

        "new_training_performed":
            False,

        "new_synthetic_rows_generated":
            False,
    }

    return (
        chain_rows,
        verdict_row,
    )


# =============================================================================
# 13. CHECKPOINT CANDIDATE SUMMARY
# =============================================================================

def summarize_checkpoint_candidates(
    checkpoints: List[Dict[str, Any]],
    structures: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    structure_by_file = defaultdict(
        list
    )

    for row in structures:

        structure_by_file[
            row.get(
                "file",
                "",
            )
        ].append(
            row
        )

    links_by_file = defaultdict(
        list
    )

    for row in links:

        links_by_file[
            row.get(
                "checkpoint",
                "",
            )
        ].append(
            row
        )

    rows = []

    for checkpoint in checkpoints:

        file_name = checkpoint[
            "file"
        ]

        structure_rows = structure_by_file[
            file_name
        ]

        link_rows = links_by_file[
            file_name
        ]

        has_51_shape = any(
            row.get(
                "possible_output_dimension_51",
                0,
            )
            == 1
            for row
            in structure_rows
        )

        generator_51 = any(
            row.get(
                "possible_generator_output_layer",
                0,
            )
            == 1
            for row
            in structure_rows
        )

        strongest_link = (
            "STRONG"
            if any(
                row.get(
                    "link_strength"
                )
                == "STRONG"
                for row
                in link_rows
            )
            else
            "WEAK"
            if any(
                row.get(
                    "link_strength"
                )
                == "WEAK"
                for row
                in link_rows
            )
            else
            "NONE"
        )

        rows.append(
            {
                **checkpoint,

                "structure_rows":
                    len(
                        structure_rows
                    ),

                "contains_shape_dimension_51":
                    int(
                        has_51_shape
                    ),

                "possible_generator_output_layer_51":
                    int(
                        generator_51
                    ),

                "strongest_code_link":
                    strongest_link,

                "linked_code_files":
                    safe_json(
                        sorted(
                            {
                                row.get(
                                    "code_file",
                                    "",
                                )
                                for row
                                in link_rows
                                if row.get(
                                    "code_file"
                                )
                            }
                        )
                    ),
            }
        )

    return rows


# =============================================================================
# 14. MAIN
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
        "HFAGM - RECONSTRUCT STRUCTURED SYNTHETIC GENERATION"
    )

    print(
        "=" * 100
    )

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        "\nThis is a forensic reconstruction."
    )

    print(
        "No training and no new synthetic-data generation will be performed."
    )

    # -----------------------------------------------------------------
    # Real data/schema verification.
    # -----------------------------------------------------------------

    (
        raw_df,
        raw_path,
        raw_source_type,
    ) = load_raw_dataset()

    target_col = identify_target_column(
        raw_df
    )

    feature_names = load_feature_schema()

    print(
        f"\nRaw clinical source: {raw_path}"
    )

    print(
        f"Rows: {len(raw_df)}"
    )

    print(
        f"Target: {target_col}"
    )

    print(
        f"Structured feature schema: {len(feature_names)} predictors"
    )

    # -----------------------------------------------------------------
    # Source inventory and AST analysis.
    # -----------------------------------------------------------------

    print(
        "\nScanning project generation code..."
    )

    source_files = project_source_files()

    (
        project_code_inventory,
        generator_definitions,
        generation_operations,
    ) = textual_generation_inventory(
        source_files
    )

    write_csv(
        OUTPUT_DIR
        / "project_generation_code_inventory.csv",
        project_code_inventory,
    )

    write_csv(
        OUTPUT_DIR
        / "generator_definition_evidence.csv",
        generator_definitions,
    )

    write_csv(
        OUTPUT_DIR
        / "generation_operation_evidence.csv",
        generation_operations,
    )

    print(
        f"Generation-related source/config files: "
        f"{len(project_code_inventory)}"
    )

    print(
        f"Generator-like definitions: "
        f"{len(generator_definitions)}"
    )

    print(
        f"Generation/preprocessing/save/load operations: "
        f"{len(generation_operations)}"
    )

    # -----------------------------------------------------------------
    # Checkpoints.
    # -----------------------------------------------------------------

    print(
        "\nInventorying model/checkpoint artifacts..."
    )

    checkpoint_rows = inventory_checkpoints()

    write_csv(
        OUTPUT_DIR
        / "checkpoint_inventory.csv",
        checkpoint_rows,
    )

    print(
        f"Checkpoint/model artifacts: "
        f"{len(checkpoint_rows)}"
    )

    generator_checkpoint_candidates = (
        checkpoint_generation_candidates(
            checkpoint_rows
        )
    )

    print(
        f"Generator-related checkpoint candidates: "
        f"{len(generator_checkpoint_candidates)}"
    )

    # -----------------------------------------------------------------
    # Checkpoint structure.
    # -----------------------------------------------------------------

    print(
        "\nInspecting checkpoint structures where safe..."
    )

    checkpoint_structure_rows = (
        inspect_checkpoint_structures(
            checkpoint_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "checkpoint_structure_audit.csv",
        checkpoint_structure_rows,
    )

    # -----------------------------------------------------------------
    # Checkpoint -> code linkage.
    # -----------------------------------------------------------------

    checkpoint_link_rows = (
        link_checkpoints_to_code(
            checkpoint_rows,
            source_files,
        )
    )

    write_csv(
        OUTPUT_DIR
        / "checkpoint_code_linkage.csv",
        checkpoint_link_rows,
    )

    checkpoint_candidate_summary = (
        summarize_checkpoint_candidates(
            checkpoint_rows,
            checkpoint_structure_rows,
            checkpoint_link_rows,
        )
    )

    write_csv(
        OUTPUT_DIR
        / "checkpoint_candidate_summary.csv",
        checkpoint_candidate_summary,
    )

    # -----------------------------------------------------------------
    # Search for overlooked existing synthetic tables.
    # -----------------------------------------------------------------

    print(
        "\nSearching for overlooked structured synthetic tables..."
    )

    (
        synthetic_table_inventory,
        synthetic_mapping_rows,
        verified_existing_tables,
    ) = inspect_synthetic_tables(
        feature_names
    )

    write_csv(
        OUTPUT_DIR
        / "synthetic_table_inventory.csv",
        synthetic_table_inventory,
    )

    write_csv(
        OUTPUT_DIR
        / "synthetic_table_feature_mapping.csv",
        synthetic_mapping_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "existing_verified_synthetic_tables.csv",
        verified_existing_tables,
    )

    print(
        f"Candidate synthetic tables: "
        f"{len(synthetic_table_inventory)}"
    )

    print(
        f"Existing verified 51-feature synthetic tables: "
        f"{len(verified_existing_tables)}"
    )

    # -----------------------------------------------------------------
    # Preprocessing/generation evidence.
    # -----------------------------------------------------------------

    preprocessing_generation_rows = (
        extract_preprocessing_generation_rows(
            generation_operations
        )
    )

    write_csv(
        OUTPUT_DIR
        / "preprocessing_generation_evidence.csv",
        preprocessing_generation_rows,
    )

    # -----------------------------------------------------------------
    # Build complete provenance chain.
    # -----------------------------------------------------------------

    (
        provenance_chain_rows,
        verdict_row,
    ) = build_provenance_chain(
        generator_definitions,
        generation_operations,
        checkpoint_rows,
        checkpoint_structure_rows,
        checkpoint_link_rows,
        verified_existing_tables,
    )

    write_csv(
        OUTPUT_DIR
        / "generation_provenance_chain.csv",
        provenance_chain_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "generation_reconstruction_verdict.csv",
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
            "04B_reconstruct_structured_synthetic_generation.py",

        "project_root":
            str(
                PROJECT_ROOT
            ),

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

        "target":
            target_col,

        "feature_schema_source":
            str(
                HISTORICAL_X_TRAIN_PATH
            ),

        "feature_schema_sha256":
            sha256_file(
                HISTORICAL_X_TRAIN_PATH
            ),

        "feature_count":
            len(
                feature_names
            ),

        "source_files_scanned":
            len(
                source_files
            ),

        "generation_related_source_files":
            len(
                project_code_inventory
            ),

        "generator_definitions":
            len(
                generator_definitions
            ),

        "generation_operations":
            len(
                generation_operations
            ),

        "checkpoint_artifacts":
            len(
                checkpoint_rows
            ),

        "generator_checkpoint_candidates":
            len(
                generator_checkpoint_candidates
            ),

        "synthetic_table_candidates":
            len(
                synthetic_table_inventory
            ),

        "verified_existing_synthetic_tables":
            len(
                verified_existing_tables
            ),

        "verdict":
            verdict_row[
                "verdict"
            ],

        "new_training_performed":
            False,

        "new_synthetic_rows_generated":
            False,

        "arbitrary_project_modules_executed":
            False,

        "serialized_python_objects_unpickled":
            False,

        "torch_checkpoints_inspected_with_weights_only_when_supported":
            True,

        "python_version":
            sys.version,

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,
    }

    write_csv(
        OUTPUT_DIR
        / "reconstruction_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    passed_criteria = [
        row
        for row
        in provenance_chain_rows
        if row[
            "passed"
        ] == 1
    ]

    failed_criteria = [
        row
        for row
        in provenance_chain_rows
        if row[
            "passed"
        ] == 0
    ]

    lines = [
        "=" * 100,
        "HFAGM - STRUCTURED SYNTHETIC GENERATION RECONSTRUCTION",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "PURPOSE",
        "-" * 100,
        (
            "Determine whether the original structured synthetic-data "
            "generation can be reproduced from EXISTING project artifacts."
        ),
        (
            "This audit performs no retraining and generates no new "
            "synthetic clinical records."
        ),
        "",
        "REFERENCE DATA",
        "-" * 100,
        f"Raw source: {raw_path}",
        f"Rows: {len(raw_df)}",
        f"Target: {target_col}",
        f"Structured predictors: {len(feature_names)}",
        "",
        "GENERATION CODE",
        "-" * 100,
        (
            f"Generation-related source/config files: "
            f"{len(project_code_inventory)}"
        ),
        (
            f"Generator-like class/function definitions: "
            f"{len(generator_definitions)}"
        ),
        (
            f"Generation/preprocessing/save/load evidence rows: "
            f"{len(generation_operations)}"
        ),
        "",
        "CHECKPOINTS / MODELS",
        "-" * 100,
        (
            f"All model/checkpoint artifacts found: "
            f"{len(checkpoint_rows)}"
        ),
        (
            f"Generator-related checkpoint candidates: "
            f"{len(generator_checkpoint_candidates)}"
        ),
        "",
    ]

    if generator_checkpoint_candidates:

        lines.append(
            "Generator-related checkpoint candidates:"
        )

        for row in generator_checkpoint_candidates:

            lines.append(
                (
                    f"  - {row['file']} | "
                    f"priority={row['priority']} | "
                    f"size={row['size_bytes']} bytes"
                )
            )

    else:

        lines.append(
            "No generator-related checkpoint candidate was found."
        )

    lines.extend(
        [
            "",
            "EXISTING STRUCTURED SYNTHETIC TABLES",
            "-" * 100,
            (
                f"Candidate tables reviewed: "
                f"{len(synthetic_table_inventory)}"
            ),
            (
                f"Verified existing 51-feature synthetic tables: "
                f"{len(verified_existing_tables)}"
            ),
        ]
    )

    if verified_existing_tables:

        for row in verified_existing_tables:

            lines.append(
                (
                    f"  - {row['source_file']} "
                    f"({row['rows']} rows)"
                )
            )

            lines.append(
                (
                    f"    copied to: "
                    f"{row['copied_to']}"
                )
            )

    lines.extend(
        [
            "",
            "PROVENANCE CHAIN",
            "-" * 100,
        ]
    )

    for row in provenance_chain_rows:

        symbol = (
            "PASS"
            if row[
                "passed"
            ] == 1
            else "MISSING"
        )

        lines.append(
            (
                f"{symbol}: "
                f"{row['criterion']}"
            )
        )

        lines.append(
            (
                f"    {row['detail']}"
            )
        )

    lines.extend(
        [
            "",
            "RECONSTRUCTION VERDICT",
            "-" * 100,
            verdict_row[
                "verdict"
            ],
            "",
            "NEXT ACTION",
            "-" * 100,
            verdict_row[
                "next_action"
            ],
            "",
            "MANUSCRIPT CONSEQUENCE",
            "-" * 100,
            verdict_row[
                "manuscript_action"
            ],
            "",
            "SAFETY CONFIRMATION",
            "-" * 100,
            "New model training performed: NO",
            "New synthetic clinical rows generated: NO",
            "Historical model/checkpoint overwritten: NO",
            "Historical synthetic table overwritten: NO",
            "Arbitrary project Python modules executed: NO",
            "Unknown pickle/joblib objects deserialized: NO",
            "",
            "PRIMARY OUTPUTS",
            "-" * 100,
            "project_generation_code_inventory.csv",
            "generator_definition_evidence.csv",
            "generation_operation_evidence.csv",
            "checkpoint_inventory.csv",
            "checkpoint_structure_audit.csv",
            "checkpoint_code_linkage.csv",
            "checkpoint_candidate_summary.csv",
            "synthetic_table_inventory.csv",
            "synthetic_table_feature_mapping.csv",
            "existing_verified_synthetic_tables.csv",
            "preprocessing_generation_evidence.csv",
            "generation_provenance_chain.csv",
            "generation_reconstruction_verdict.csv",
            "reconstruction_provenance.csv",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "structured_generation_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console.
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "04B COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nGenerator definitions found: "
        f"{len(generator_definitions)}"
    )

    print(
        f"Generator checkpoint candidates: "
        f"{len(generator_checkpoint_candidates)}"
    )

    print(
        f"Verified existing structured synthetic tables: "
        f"{len(verified_existing_tables)}"
    )

    print(
        f"\nProvenance criteria passed: "
        f"{len(passed_criteria)}/{len(provenance_chain_rows)}"
    )

    print(
        "\nVerdict:"
    )

    print(
        verdict_row[
            "verdict"
        ]
    )

    print(
        "\nNext action:"
    )

    print(
        verdict_row[
            "next_action"
        ]
    )

    print(
        "\nManuscript consequence:"
    )

    print(
        verdict_row[
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
        "structured_generation_summary.txt",
        "generation_reconstruction_verdict.csv",
        "generation_provenance_chain.csv",
        "checkpoint_candidate_summary.csv",
        "checkpoint_inventory.csv",
        "checkpoint_structure_audit.csv",
        "checkpoint_code_linkage.csv",
        "generator_definition_evidence.csv",
        "generation_operation_evidence.csv",
        "synthetic_table_inventory.csv",
        "existing_verified_synthetic_tables.csv",
        "preprocessing_generation_evidence.csv",
        "reconstruction_provenance.csv",
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
            "04B FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo model was trained, no synthetic data were generated, "
            "and no historical project artifact was modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(
            1
        )