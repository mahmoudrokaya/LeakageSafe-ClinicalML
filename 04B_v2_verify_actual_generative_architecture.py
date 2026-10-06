"""
04B_v2_verify_actual_generative_architecture.py
===============================================

HFAGM - Verification of actual generative architecture in recovered project.

PURPOSE
-------
The earlier 04B reconstruction produced false-positive "generator" evidence
because it:

1. treated generic .sample() calls as generative sampling;
2. treated ordinary .npy arrays as generator checkpoints;
3. treated encoder/classifier artifacts as generator checkpoints;
4. treated the class name "HFAGM" as proof of a generative model.

This V2 script applies stricter semantic requirements.

PRIMARY QUESTION
----------------
Does the recovered project contain an implemented and reproducible:

    GAN
    VAE
    diffusion model
    or hybrid GAN/VAE/diffusion generator

that can generate the 51-feature COVID structured clinical records?

THIS SCRIPT DOES
----------------
1. Audits all project Python code except revision scripts and environments.
2. Inspects actual classes/functions using AST.
3. Detects genuine GAN components:
       Generator
       Discriminator
       adversarial losses
       latent/noise sampling
4. Detects genuine VAE components:
       encoder
       decoder
       mu/logvar
       reparameterization
       KL divergence
5. Detects genuine diffusion components:
       timestep/noise schedule
       forward noising
       reverse denoising
       noise prediction
6. Detects hybrid integration evidence.
7. Determines whether model output dimension is explicitly 51.
8. Distinguishes:
       classifier
       encoder
       graph model
       generative model
9. Audits checkpoint artifacts conservatively.
10. Links only plausible generative checkpoints to actual generative code.
11. Searches for explicit structured synthetic-data generation and saving.
12. Produces a final provenance verdict.

THIS SCRIPT DOES NOT
--------------------
- train any model;
- initialize models for inference;
- generate synthetic data;
- execute arbitrary project modules;
- unpickle unknown .pkl/.joblib files;
- infer architecture from filenames alone;
- classify pandas.DataFrame.sample() as generation;
- classify np.random sampling alone as a generative model;
- classify an encoder checkpoint as a generator checkpoint;
- classify adjacency/embedding .npy files as generator checkpoints.

OUTPUT
------
outputs/revision_fidelity/actual_generative_architecture_verification/

    python_file_inventory.csv
    class_architecture_inventory.csv
    function_inventory.csv
    generative_semantic_evidence.csv
    gan_evidence.csv
    vae_evidence.csv
    diffusion_evidence.csv
    hybrid_integration_evidence.csv
    classifier_graph_evidence.csv
    structured_output_evidence.csv
    checkpoint_inventory_v2.csv
    checkpoint_generative_linkage.csv
    generation_save_evidence.csv
    architecture_verification_matrix.csv
    actual_generative_architecture_verdict.csv
    actual_generative_architecture_summary.txt
    verification_provenance.csv

FINAL VERDICTS
--------------
VERIFIED_STRUCTURED_HYBRID_GENERATIVE_ARCHITECTURE

VERIFIED_STRUCTURED_GAN_ONLY

VERIFIED_STRUCTURED_VAE_ONLY

VERIFIED_STRUCTURED_DIFFUSION_ONLY

GENERATIVE_COMPONENTS_FOUND_BUT_NO_VERIFIED_51_FEATURE_OUTPUT

GENERATIVE_CODE_FOUND_BUT_NO_TRAINED_GENERATOR_CHECKPOINT

CLASSIFIER_OR_ENCODER_ONLY_NO_STRUCTURED_GENERATOR

NO_ACTUAL_GENERATIVE_ARCHITECTURE_FOUND
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import traceback

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    / "revision_fidelity"
    / "actual_generative_architecture_verification"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "covid_clinical.csv"
)

FEATURE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "X_train_scaled.csv"
)

EXPECTED_FEATURE_COUNT = 51

# -------------------------------------------------------------------------
# Revision files must not count as historical project implementation.
# -------------------------------------------------------------------------

REVISION_PREFIXES = (
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

# -------------------------------------------------------------------------
# Excluded directories.
# -------------------------------------------------------------------------

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
    ".mypy_cache",
}

EXCLUDED_PREFIXES = (
    ".venv",
    "venv",
    ".env",
)

EXCLUDED_SUFFIXES = (
    ".dist-info",
    ".egg-info",
)

# -------------------------------------------------------------------------
# Model/checkpoint extensions.
# -------------------------------------------------------------------------

CHECKPOINT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".pkl",
    ".pickle",
    ".joblib",
    ".h5",
    ".hdf5",
    ".keras",
    ".onnx",
}

# -------------------------------------------------------------------------
# Strong semantic indicators.
# -------------------------------------------------------------------------

GAN_CLASS_TERMS = {
    "generator",
    "discriminator",
    "gan",
    "wgan",
    "cgan",
}

VAE_CLASS_TERMS = {
    "vae",
    "variationalautoencoder",
    "encoder",
    "decoder",
}

DIFFUSION_CLASS_TERMS = {
    "diffusion",
    "denoiser",
    "unet",
    "noise_predictor",
}

CLASSIFIER_TERMS = {
    "classifier",
    "classification",
    "logisticregression",
    "randomforest",
    "gradientboosting",
    "votingclassifier",
}

GRAPH_TERMS = {
    "graph",
    "gcn",
    "gat",
    "adjacency",
    "subgraph",
    "edge_index",
}

# -------------------------------------------------------------------------
# GAN-specific operations.
# -------------------------------------------------------------------------

GAN_STRONG_PATTERNS = [
    r"\bdiscriminator\b",
    r"\bgenerator\b",
    r"adversarial",
    r"binary_cross_entropy",
    r"bcewithlogitsloss",
    r"wasserstein",
]

# -------------------------------------------------------------------------
# VAE-specific operations.
# -------------------------------------------------------------------------

VAE_STRONG_PATTERNS = [
    r"\blogvar\b",
    r"\blog_var\b",
    r"\bmu\b",
    r"\breparameter",
    r"\bkl_div",
    r"\bkld\b",
    r"kl divergence",
    r"\bdecoder\b",
]

# -------------------------------------------------------------------------
# Diffusion-specific operations.
# -------------------------------------------------------------------------

DIFFUSION_STRONG_PATTERNS = [
    r"\btimestep",
    r"\btimesteps",
    r"\bbeta_schedule",
    r"\bnoise_schedule",
    r"\balphas_cumprod",
    r"\bq_sample",
    r"\bp_sample",
    r"reverse_diffusion",
    r"predict_noise",
    r"noise_pred",
]

# -------------------------------------------------------------------------
# Genuine generation method names.
#
# NOTE:
# Generic method "sample" alone is NOT accepted unless it belongs to a
# verified generative class or is accompanied by latent/noise/model evidence.
# -------------------------------------------------------------------------

STRONG_GENERATION_METHOD_NAMES = {
    "generate",
    "generate_samples",
    "generate_synthetic",
    "synthesize",
    "synthesise",
    "reverse_diffusion",
    "p_sample",
    "decode",
}

# -------------------------------------------------------------------------
# Latent/noise parameter terms.
# -------------------------------------------------------------------------

LATENT_PARAMETER_TERMS = {
    "latent_dim",
    "z_dim",
    "noise_dim",
    "latent_size",
    "nz",
}

OUTPUT_DIMENSION_TERMS = {
    "output_dim",
    "data_dim",
    "feature_dim",
    "num_features",
    "n_features",
    "input_dim",
}

# -------------------------------------------------------------------------
# DataFrame.sample must explicitly be excluded.
# -------------------------------------------------------------------------

NON_GENERATIVE_SAMPLE_RECEIVERS = {
    "df",
    "data",
    "dataset",
    "train",
    "test",
    "x",
    "y",
    "df_balanced",
    "dataframe",
}

# -------------------------------------------------------------------------
# Known non-generative checkpoint hints.
# -------------------------------------------------------------------------

NON_GENERATIVE_CHECKPOINT_HINTS = {
    "encoder",
    "classifier",
    "ensemble",
    "randomforest",
    "logistic",
    "gradientboost",
    "adj_matrix",
    "graph_features",
    "embedding",
    "contrastive",
}

GENERATOR_CHECKPOINT_HINTS = {
    "generator",
    "decoder",
    "gan",
    "vae",
    "diffusion",
    "denoiser",
}

MAX_TEXT_SIZE = 10 * 1024 * 1024


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


def normalize_name(value: Any) -> str:
    return "".join(
        ch
        for ch in str(value).lower()
        if ch.isalnum()
    )


def relative_path(path: Path) -> str:
    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        )
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

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
        extra = [
            c
            for c in df.columns
            if c not in columns
        ]

        df = df[
            columns + extra
        ]

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def read_text(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_TEXT_SIZE:
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


def is_excluded(path: Path) -> bool:
    try:
        parts = [
            p.lower()
            for p in path.relative_to(
                PROJECT_ROOT
            ).parts
        ]
    except Exception:
        parts = [
            p.lower()
            for p in path.parts
        ]

    for part in parts:
        if part in EXCLUDED_DIR_EXACT:
            return True

        if any(
            part.startswith(prefix)
            for prefix
            in EXCLUDED_PREFIXES
        ):
            return True

        if any(
            part.endswith(suffix)
            for suffix
            in EXCLUDED_SUFFIXES
        ):
            return True

    return False


def is_revision_script(path: Path) -> bool:
    name = path.name.lower()

    return any(
        name.startswith(prefix)
        for prefix in REVISION_PREFIXES
    )


def is_revision_output(path: Path) -> bool:
    lower = str(path).lower()

    return any(
        marker.lower() in lower
        for marker in REVISION_OUTPUT_MARKERS
    )


def source_line(
    text: str,
    node: ast.AST,
) -> str:
    lineno = getattr(
        node,
        "lineno",
        None,
    )

    if lineno is None:
        return ""

    lines = text.splitlines()

    if (
        lineno < 1
        or
        lineno > len(lines)
    ):
        return ""

    return lines[
        lineno - 1
    ].strip()[:1500]


def ast_unparse_safe(
    node: ast.AST,
) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def literal_safe(
    node: Optional[ast.AST],
) -> Any:
    if node is None:
        return None

    try:
        return ast.literal_eval(node)
    except Exception:
        return None


# =============================================================================
# 3. LOAD VERIFIED FEATURE SCHEMA
# =============================================================================

def load_feature_schema() -> List[str]:
    if not FEATURE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema file missing:\n"
            f"{FEATURE_SCHEMA_PATH}"
        )

    df = pd.read_csv(
        FEATURE_SCHEMA_PATH
    )

    features = list(
        map(str, df.columns)
    )

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} features; "
            f"found {len(features)}."
        )

    return features


# =============================================================================
# 4. PYTHON FILE DISCOVERY
# =============================================================================

def python_files() -> List[Path]:
    results = []

    for path in PROJECT_ROOT.rglob("*.py"):
        if not path.is_file():
            continue

        if is_excluded(path):
            continue

        if is_revision_output(path):
            continue

        if is_revision_script(path):
            continue

        results.append(path)

    return sorted(
        results,
        key=lambda p: str(p).lower(),
    )


# =============================================================================
# 5. CLASS / FUNCTION AST ANALYSIS
# =============================================================================

def analyze_python_file(
    path: Path,
    text: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    class_rows = []
    function_rows = []
    semantic_rows = []

    try:
        tree = ast.parse(
            text,
            filename=str(path),
        )

    except Exception as exc:
        semantic_rows.append(
            {
                "file":
                    relative_path(path),

                "line":
                    np.nan,

                "category":
                    "PARSE_ERROR",

                "symbol":
                    "",

                "detail":
                    repr(exc),
            }
        )

        return (
            class_rows,
            function_rows,
            semantic_rows,
        )

    # -----------------------------------------------------------------
    # Class definitions.
    # -----------------------------------------------------------------

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.ClassDef,
        ):
            continue

        class_name = node.name

        lower_name = class_name.lower()

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

        method_lower = {
            method.lower()
            for method in methods
        }

        body_source = ""

        try:
            body_source = ast.get_source_segment(
                text,
                node,
            ) or ""
        except Exception:
            pass

        body_lower = body_source.lower()

        gan_score = sum(
            int(
                re.search(
                    pattern,
                    body_lower,
                    flags=re.IGNORECASE,
                )
                is not None
            )
            for pattern in GAN_STRONG_PATTERNS
        )

        vae_score = sum(
            int(
                re.search(
                    pattern,
                    body_lower,
                    flags=re.IGNORECASE,
                )
                is not None
            )
            for pattern in VAE_STRONG_PATTERNS
        )

        diffusion_score = sum(
            int(
                re.search(
                    pattern,
                    body_lower,
                    flags=re.IGNORECASE,
                )
                is not None
            )
            for pattern in DIFFUSION_STRONG_PATTERNS
        )

        classifier_score = sum(
            int(term in body_lower)
            for term in CLASSIFIER_TERMS
        )

        graph_score = sum(
            int(term in body_lower)
            for term in GRAPH_TERMS
        )

        strong_generation_method = bool(
            method_lower
            &
            STRONG_GENERATION_METHOD_NAMES
        )

        # -------------------------------------------------------------
        # Name alone is not enough.
        # -------------------------------------------------------------

        name_gan_hint = int(
            any(
                term in lower_name
                for term in GAN_CLASS_TERMS
            )
        )

        name_vae_hint = int(
            any(
                term in lower_name
                for term in VAE_CLASS_TERMS
            )
        )

        name_diffusion_hint = int(
            any(
                term in lower_name
                for term in DIFFUSION_CLASS_TERMS
            )
        )

        # -------------------------------------------------------------
        # A class is considered genuinely generative only when actual
        # semantic evidence exists inside the implementation.
        # -------------------------------------------------------------

        genuine_gan = bool(
            (
                gan_score >= 2
                and
                (
                    "generator" in body_lower
                    or
                    "discriminator" in body_lower
                    or
                    "adversarial" in body_lower
                )
            )
        )

        genuine_vae = bool(
            (
                vae_score >= 3
                and
                (
                    "logvar" in body_lower
                    or
                    "reparameter" in body_lower
                )
            )
        )

        genuine_diffusion = bool(
            diffusion_score >= 3
        )

        genuinely_generative = bool(
            genuine_gan
            or
            genuine_vae
            or
            genuine_diffusion
        )

        class_rows.append(
            {
                "file":
                    relative_path(path),

                "line":
                    node.lineno,

                "class_name":
                    class_name,

                "bases":
                    safe_json(
                        [
                            ast_unparse_safe(base)
                            for base in node.bases
                        ]
                    ),

                "methods":
                    safe_json(methods),

                "has_forward":
                    int(
                        "forward"
                        in method_lower
                    ),

                "has_strong_generation_method":
                    int(
                        strong_generation_method
                    ),

                "name_gan_hint":
                    name_gan_hint,

                "name_vae_hint":
                    name_vae_hint,

                "name_diffusion_hint":
                    name_diffusion_hint,

                "gan_semantic_score":
                    gan_score,

                "vae_semantic_score":
                    vae_score,

                "diffusion_semantic_score":
                    diffusion_score,

                "classifier_score":
                    classifier_score,

                "graph_score":
                    graph_score,

                "verified_gan_component":
                    int(
                        genuine_gan
                    ),

                "verified_vae_component":
                    int(
                        genuine_vae
                    ),

                "verified_diffusion_component":
                    int(
                        genuine_diffusion
                    ),

                "verified_generative_component":
                    int(
                        genuinely_generative
                    ),
            }
        )

    # -----------------------------------------------------------------
    # Function definitions.
    # -----------------------------------------------------------------

    for node in ast.walk(tree):

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        function_name = node.name

        lower_name = (
            function_name.lower()
        )

        parent_generation_name = int(
            lower_name
            in STRONG_GENERATION_METHOD_NAMES
        )

        function_rows.append(
            {
                "file":
                    relative_path(path),

                "line":
                    node.lineno,

                "function_name":
                    function_name,

                "arguments":
                    safe_json(
                        [
                            arg.arg
                            for arg
                            in node.args.args
                        ]
                    ),

                "strong_generation_name":
                    parent_generation_name,
            }
        )

    # -----------------------------------------------------------------
    # Semantic calls / assignments.
    # -----------------------------------------------------------------

    for node in ast.walk(tree):

        # -------------------------------------------------------------
        # Assignments.
        # -------------------------------------------------------------

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

                target_text = (
                    ast_unparse_safe(target)
                )

                normalized = (
                    target_text
                    .lower()
                    .replace(
                        "self.",
                        "",
                    )
                )

                literal = literal_safe(
                    value_node
                )

                if any(
                    term in normalized
                    for term in LATENT_PARAMETER_TERMS
                ):
                    semantic_rows.append(
                        {
                            "file":
                                relative_path(path),

                            "line":
                                node.lineno,

                            "category":
                                "LATENT_DIMENSION",

                            "symbol":
                                target_text,

                            "detail":
                                safe_json(literal),

                            "statement":
                                source_line(
                                    text,
                                    node,
                                ),
                        }
                    )

                if any(
                    term in normalized
                    for term in OUTPUT_DIMENSION_TERMS
                ):
                    semantic_rows.append(
                        {
                            "file":
                                relative_path(path),

                            "line":
                                node.lineno,

                            "category":
                                "OUTPUT_DIMENSION",

                            "symbol":
                                target_text,

                            "detail":
                                safe_json(literal),

                            "statement":
                                source_line(
                                    text,
                                    node,
                                ),
                        }
                    )

        # -------------------------------------------------------------
        # Calls.
        # -------------------------------------------------------------

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        call_name = (
            ast_unparse_safe(
                node.func
            )
        )

        lower_call = (
            call_name.lower()
        )

        final_method = (
            lower_call.split(".")[-1]
        )

        receiver = ""

        if "." in lower_call:
            receiver = (
                lower_call
                .rsplit(
                    ".",
                    1,
                )[0]
            )

        category = None

        # -------------------------------------------------------------
        # EXPLICITLY reject DataFrame.sample()
        # -------------------------------------------------------------

        if final_method == "sample":

            receiver_tail = (
                receiver
                .split(".")[-1]
            )

            if (
                receiver_tail
                in NON_GENERATIVE_SAMPLE_RECEIVERS
                or
                "dataframe" in lower_call
                or
                "df_" in lower_call
            ):
                semantic_rows.append(
                    {
                        "file":
                            relative_path(path),

                        "line":
                            node.lineno,

                        "category":
                            "NON_GENERATIVE_DATAFRAME_SAMPLE",

                        "symbol":
                            call_name,

                        "detail":
                            "Excluded from synthetic-generation evidence.",

                        "statement":
                            source_line(
                                text,
                                node,
                            ),
                    }
                )

                continue

        if final_method in STRONG_GENERATION_METHOD_NAMES:

            category = (
                "EXPLICIT_GENERATION_CALL"
            )

        elif any(
            token in lower_call
            for token in [
                "torch.randn",
                "np.random.randn",
                "numpy.random.randn",
                "torch.normal",
            ]
        ):

            category = (
                "LATENT_OR_NOISE_SAMPLING"
            )

        elif (
            "binary_cross_entropy"
            in lower_call
            or
            "bcewithlogitsloss"
            in lower_call
        ):

            category = (
                "ADVERSARIAL_COMPATIBLE_LOSS_CALL"
            )

        elif (
            "kl_div"
            in lower_call
            or
            "kld"
            in lower_call
        ):

            category = (
                "VAE_KL_OPERATION"
            )

        elif (
            "torch.save"
            in lower_call
            or
            "state_dict"
            in lower_call
        ):

            category = (
                "MODEL_SAVE"
            )

        elif (
            "torch.load"
            in lower_call
            or
            "load_state_dict"
            in lower_call
        ):

            category = (
                "MODEL_LOAD"
            )

        elif (
            "to_csv"
            in lower_call
            or
            "np.save"
            in lower_call
            or
            "to_excel"
            in lower_call
        ):

            category = (
                "DATA_SAVE"
            )

        if category:

            semantic_rows.append(
                {
                    "file":
                        relative_path(path),

                    "line":
                        node.lineno,

                    "category":
                        category,

                    "symbol":
                        call_name,

                    "detail":
                        "",

                    "statement":
                        source_line(
                            text,
                            node,
                        ),
                }
            )

    return (
        class_rows,
        function_rows,
        semantic_rows,
    )


# =============================================================================
# 6. CHECKPOINT AUDIT V2
# =============================================================================

def checkpoint_inventory_v2(
) -> List[Dict[str, Any]]:
    rows = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if is_excluded(path):
            continue

        if is_revision_output(path):
            continue

        if (
            path.suffix.lower()
            not in CHECKPOINT_EXTENSIONS
        ):
            continue

        lower_name = (
            path.name.lower()
        )

        lower_path = (
            str(path).lower()
        )

        positive_hints = [
            hint
            for hint in GENERATOR_CHECKPOINT_HINTS
            if (
                hint in lower_name
                or
                hint in lower_path
            )
        ]

        negative_hints = [
            hint
            for hint in NON_GENERATIVE_CHECKPOINT_HINTS
            if (
                hint in lower_name
                or
                hint in lower_path
            )
        ]

        # -------------------------------------------------------------
        # Do not classify by location alone.
        # -------------------------------------------------------------

        plausible_generator_checkpoint = bool(
            positive_hints
            and
            not (
                negative_hints
                and
                not any(
                    strong in lower_name
                    for strong in [
                        "generator",
                        "decoder",
                    ]
                )
            )
        )

        # ArSL encoder checkpoint explicitly not a structured generator.
        if (
            "new_exp2"
            in lower_path
            and
            "encoder"
            in lower_name
        ):
            plausible_generator_checkpoint = False

        # sklearn ensemble explicitly non-generative.
        if (
            "ensemble_model"
            in lower_name
        ):
            plausible_generator_checkpoint = False

        rows.append(
            {
                "file":
                    relative_path(path),

                "extension":
                    path.suffix.lower(),

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(path),

                "positive_generator_hints":
                    safe_json(
                        positive_hints
                    ),

                "negative_non_generator_hints":
                    safe_json(
                        negative_hints
                    ),

                "plausible_generative_checkpoint":
                    int(
                        plausible_generator_checkpoint
                    ),

                "status":
                    (
                        "PLAUSIBLE_GENERATIVE_CHECKPOINT"
                        if plausible_generator_checkpoint
                        else
                        "NON_GENERATIVE_OR_UNVERIFIED_CHECKPOINT"
                    ),
            }
        )

    return rows


# =============================================================================
# 7. CHECKPOINT-CODE LINKAGE
# =============================================================================

def checkpoint_code_linkage(
    checkpoint_rows: List[Dict[str, Any]],
    python_texts: Dict[Path, str],
    class_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results = []

    verified_generative_files = {
        row[
            "file"
        ]
        for row in class_rows
        if row[
            "verified_generative_component"
        ] == 1
    }

    for checkpoint in checkpoint_rows:

        if (
            checkpoint[
                "plausible_generative_checkpoint"
            ] != 1
        ):
            continue

        checkpoint_name = Path(
            checkpoint[
                "file"
            ]
        ).name.lower()

        found = False

        for path, text in python_texts.items():

            relative = (
                relative_path(path)
            )

            lower = text.lower()

            exact_filename = (
                checkpoint_name in lower
            )

            code_is_generative = (
                relative
                in verified_generative_files
            )

            if exact_filename:

                results.append(
                    {
                        "checkpoint":
                            checkpoint[
                                "file"
                            ],

                        "code_file":
                            relative,

                        "exact_filename_reference":
                            1,

                        "code_verified_generative":
                            int(
                                code_is_generative
                            ),

                        "valid_generative_link":
                            int(
                                code_is_generative
                            ),
                    }
                )

                found = True

        if not found:

            results.append(
                {
                    "checkpoint":
                        checkpoint[
                            "file"
                        ],

                    "code_file":
                        "",

                    "exact_filename_reference":
                        0,

                    "code_verified_generative":
                        0,

                    "valid_generative_link":
                        0,
                }
            )

    return results


# =============================================================================
# 8. ARCHITECTURE EVIDENCE AGGREGATION
# =============================================================================

def architecture_evidence(
    class_rows: List[Dict[str, Any]],
    semantic_rows: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    gan_rows = []
    vae_rows = []
    diffusion_rows = []
    classifier_graph_rows = []
    hybrid_rows = []

    by_file = defaultdict(
        lambda: {
            "gan": False,
            "vae": False,
            "diffusion": False,
        }
    )

    for row in class_rows:

        file = row[
            "file"
        ]

        if row[
            "verified_gan_component"
        ] == 1:

            gan_rows.append(row)

            by_file[
                file
            ][
                "gan"
            ] = True

        if row[
            "verified_vae_component"
        ] == 1:

            vae_rows.append(row)

            by_file[
                file
            ][
                "vae"
            ] = True

        if row[
            "verified_diffusion_component"
        ] == 1:

            diffusion_rows.append(row)

            by_file[
                file
            ][
                "diffusion"
            ] = True

        if (
            row[
                "classifier_score"
            ] > 0
            or
            row[
                "graph_score"
            ] > 0
        ):

            classifier_graph_rows.append(
                row
            )

    for file, flags in by_file.items():

        count = sum(
            int(value)
            for value in flags.values()
        )

        if count >= 2:

            hybrid_rows.append(
                {
                    "file":
                        file,

                    "gan_component":
                        int(
                            flags[
                                "gan"
                            ]
                        ),

                    "vae_component":
                        int(
                            flags[
                                "vae"
                            ]
                        ),

                    "diffusion_component":
                        int(
                            flags[
                                "diffusion"
                            ]
                        ),

                    "hybrid_component_count":
                        count,
                }
            )

    return (
        gan_rows,
        vae_rows,
        diffusion_rows,
        hybrid_rows,
        classifier_graph_rows,
    )


# =============================================================================
# 9. STRUCTURED OUTPUT EVIDENCE
# =============================================================================

def structured_output_evidence(
    semantic_rows: List[Dict[str, Any]],
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    rows = []

    feature_set_normalized = {
        normalize_name(
            feature
        )
        for feature in feature_names
    }

    for row in semantic_rows:

        if (
            row[
                "category"
            ] != "OUTPUT_DIMENSION"
        ):
            continue

        detail = str(
            row.get(
                "detail",
                "",
            )
        )

        explicit_51 = bool(
            re.search(
                r"(^|[^0-9])51([^0-9]|$)",
                detail,
            )
        )

        rows.append(
            {
                **row,

                "explicit_output_dimension_51":
                    int(
                        explicit_51
                    ),
            }
        )

    return rows


# =============================================================================
# 10. DATA SAVE EVIDENCE
# =============================================================================

def generation_save_evidence(
    semantic_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        row
        for row in semantic_rows
        if row[
            "category"
        ]
        in {
            "DATA_SAVE",
            "EXPLICIT_GENERATION_CALL",
            "LATENT_OR_NOISE_SAMPLING",
        }
    ]


# =============================================================================
# 11. FINAL VERDICT
# =============================================================================

def build_verdict(
    gan_rows: List[Dict[str, Any]],
    vae_rows: List[Dict[str, Any]],
    diffusion_rows: List[Dict[str, Any]],
    hybrid_rows: List[Dict[str, Any]],
    classifier_graph_rows: List[Dict[str, Any]],
    structured_rows: List[Dict[str, Any]],
    checkpoint_rows: List[Dict[str, Any]],
    checkpoint_links: List[Dict[str, Any]],
    semantic_rows: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    gan_found = bool(
        gan_rows
    )

    vae_found = bool(
        vae_rows
    )

    diffusion_found = bool(
        diffusion_rows
    )

    hybrid_found = bool(
        hybrid_rows
    )

    any_generative = bool(
        gan_found
        or
        vae_found
        or
        diffusion_found
    )

    output_51 = any(
        row.get(
            "explicit_output_dimension_51"
        ) == 1
        for row in structured_rows
    )

    latent_evidence = any(
        row[
            "category"
        ] == "LATENT_DIMENSION"
        for row in semantic_rows
    )

    noise_sampling = any(
        row[
            "category"
        ] == "LATENT_OR_NOISE_SAMPLING"
        for row in semantic_rows
    )

    explicit_generation_call = any(
        row[
            "category"
        ] == "EXPLICIT_GENERATION_CALL"
        for row in semantic_rows
    )

    plausible_checkpoints = [
        row
        for row in checkpoint_rows
        if row[
            "plausible_generative_checkpoint"
        ] == 1
    ]

    valid_checkpoint_links = [
        row
        for row in checkpoint_links
        if row[
            "valid_generative_link"
        ] == 1
    ]

    trained_generator_available = bool(
        plausible_checkpoints
        and
        valid_checkpoint_links
    )

    classifier_or_graph_present = bool(
        classifier_graph_rows
    )

    criteria = [
        {
            "criterion":
                "verified_gan_component",

            "passed":
                int(
                    gan_found
                ),

            "detail":
                f"{len(gan_rows)} verified GAN class(es).",
        },
        {
            "criterion":
                "verified_vae_component",

            "passed":
                int(
                    vae_found
                ),

            "detail":
                f"{len(vae_rows)} verified VAE class(es).",
        },
        {
            "criterion":
                "verified_diffusion_component",

            "passed":
                int(
                    diffusion_found
                ),

            "detail":
                (
                    f"{len(diffusion_rows)} verified "
                    "diffusion class(es)."
                ),
        },
        {
            "criterion":
                "hybrid_generative_integration",

            "passed":
                int(
                    hybrid_found
                ),

            "detail":
                (
                    f"{len(hybrid_rows)} file(s) contain "
                    "multiple verified generative paradigms."
                ),
        },
        {
            "criterion":
                "explicit_51_feature_generator_output",

            "passed":
                int(
                    output_51
                ),

            "detail":
                (
                    "Explicit 51-dimensional output found."
                    if output_51
                    else
                    "No explicit 51-dimensional generative output found."
                ),
        },
        {
            "criterion":
                "latent_dimension_evidence",

            "passed":
                int(
                    latent_evidence
                ),

            "detail":
                (
                    "Latent dimension assignment found."
                    if latent_evidence
                    else
                    "No latent dimension assignment found."
                ),
        },
        {
            "criterion":
                "latent_or_noise_sampling",

            "passed":
                int(
                    noise_sampling
                ),

            "detail":
                (
                    "Noise/latent sampling found."
                    if noise_sampling
                    else
                    "No genuine latent/noise sampling found."
                ),
        },
        {
            "criterion":
                "explicit_generation_call",

            "passed":
                int(
                    explicit_generation_call
                ),

            "detail":
                (
                    "Explicit generative method call found."
                    if explicit_generation_call
                    else
                    "No explicit generative method call found."
                ),
        },
        {
            "criterion":
                "plausible_trained_generative_checkpoint",

            "passed":
                int(
                    bool(
                        plausible_checkpoints
                    )
                ),

            "detail":
                (
                    f"{len(plausible_checkpoints)} plausible "
                    "generative checkpoint(s)."
                ),
        },
        {
            "criterion":
                "checkpoint_linked_to_verified_generative_code",

            "passed":
                int(
                    trained_generator_available
                ),

            "detail":
                (
                    f"{len(valid_checkpoint_links)} valid checkpoint/code "
                    "link(s)."
                ),
        },
    ]

    # -----------------------------------------------------------------
    # Final classification.
    # -----------------------------------------------------------------

    if (
        hybrid_found
        and
        output_51
        and
        trained_generator_available
    ):

        verdict = (
            "VERIFIED_STRUCTURED_HYBRID_GENERATIVE_ARCHITECTURE"
        )

        next_action = (
            "WRITE_04C_EXACT_REGENERATION_SCRIPT"
        )

    elif (
        gan_found
        and
        output_51
        and
        trained_generator_available
    ):

        verdict = (
            "VERIFIED_STRUCTURED_GAN_ONLY"
        )

        next_action = (
            "WRITE_04C_GAN_REGENERATION_SCRIPT"
        )

    elif (
        vae_found
        and
        output_51
        and
        trained_generator_available
    ):

        verdict = (
            "VERIFIED_STRUCTURED_VAE_ONLY"
        )

        next_action = (
            "WRITE_04C_VAE_REGENERATION_SCRIPT"
        )

    elif (
        diffusion_found
        and
        output_51
        and
        trained_generator_available
    ):

        verdict = (
            "VERIFIED_STRUCTURED_DIFFUSION_ONLY"
        )

        next_action = (
            "WRITE_04C_DIFFUSION_REGENERATION_SCRIPT"
        )

    elif (
        any_generative
        and
        not output_51
    ):

        verdict = (
            "GENERATIVE_COMPONENTS_FOUND_BUT_NO_VERIFIED_51_FEATURE_OUTPUT"
        )

        next_action = (
            "MANUALLY_VERIFY_GENERATOR_OUTPUT_SCHEMA"
        )

    elif (
        any_generative
        and
        output_51
        and
        not trained_generator_available
    ):

        verdict = (
            "GENERATIVE_CODE_FOUND_BUT_NO_TRAINED_GENERATOR_CHECKPOINT"
        )

        next_action = (
            "SEARCH_BACKUP_FOR_GENERATIVE_CHECKPOINT_OR_REMOVE_FID_CLAIMS"
        )

    elif (
        not any_generative
        and
        classifier_or_graph_present
    ):

        verdict = (
            "CLASSIFIER_OR_ENCODER_ONLY_NO_STRUCTURED_GENERATOR"
        )

        next_action = (
            "DO_NOT_REGENERATE_REMOVE_UNSUPPORTED_GENERATIVE_FID_CLAIMS"
        )

    else:

        verdict = (
            "NO_ACTUAL_GENERATIVE_ARCHITECTURE_FOUND"
        )

        next_action = (
            "DO_NOT_REGENERATE_REMOVE_UNSUPPORTED_GENERATIVE_FID_CLAIMS"
        )

    manuscript_consequence = {
        "VERIFIED_STRUCTURED_HYBRID_GENERATIVE_ARCHITECTURE":
            (
                "A recoverable structured hybrid generator exists. "
                "Regeneration may proceed only through a dedicated 04C "
                "script using the verified checkpoint and architecture."
            ),

        "VERIFIED_STRUCTURED_GAN_ONLY":
            (
                "A structured GAN is verified, but the recovered evidence "
                "does not establish the claimed GAN-VAE-diffusion hybrid. "
                "The manuscript architecture claims must be narrowed."
            ),

        "VERIFIED_STRUCTURED_VAE_ONLY":
            (
                "A structured VAE is verified, but the claimed hybrid "
                "architecture is not."
            ),

        "VERIFIED_STRUCTURED_DIFFUSION_ONLY":
            (
                "A structured diffusion model is verified, but the claimed "
                "hybrid architecture is not."
            ),

        "GENERATIVE_COMPONENTS_FOUND_BUT_NO_VERIFIED_51_FEATURE_OUTPUT":
            (
                "Some genuine generative code exists, but it is not yet "
                "proven to generate the 51-feature clinical dataset. "
                "Do not use it for structured fidelity."
            ),

        "GENERATIVE_CODE_FOUND_BUT_NO_TRAINED_GENERATOR_CHECKPOINT":
            (
                "Generative code exists, but no trained generator artifact "
                "can be tied to it. Old FID/SFD values remain unsupported."
            ),

        "CLASSIFIER_OR_ENCODER_ONLY_NO_STRUCTURED_GENERATOR":
            (
                "Recovered implementation supports classifier/encoder/graph "
                "processing rather than a verified structured synthetic "
                "generator. The manuscript's structured generative-model "
                "claims and FID values are not supported by the recovered "
                "implementation."
            ),

        "NO_ACTUAL_GENERATIVE_ARCHITECTURE_FOUND":
            (
                "No actual GAN, VAE, diffusion, or hybrid structured "
                "generator was verified in the recovered project. "
                "Do not regenerate or retain historical FID values."
            ),
    }[
        verdict
    ]

    verdict_row = {
        "verdict":
            verdict,

        "next_action":
            next_action,

        "manuscript_consequence":
            manuscript_consequence,

        "verified_gan":
            int(
                gan_found
            ),

        "verified_vae":
            int(
                vae_found
            ),

        "verified_diffusion":
            int(
                diffusion_found
            ),

        "verified_hybrid":
            int(
                hybrid_found
            ),

        "explicit_51_feature_output":
            int(
                output_51
            ),

        "latent_dimension_evidence":
            int(
                latent_evidence
            ),

        "latent_noise_sampling":
            int(
                noise_sampling
            ),

        "explicit_generation_call":
            int(
                explicit_generation_call
            ),

        "plausible_generative_checkpoints":
            len(
                plausible_checkpoints
            ),

        "valid_generative_checkpoint_code_links":
            len(
                valid_checkpoint_links
            ),

        "classifier_or_graph_components_found":
            int(
                classifier_or_graph_present
            ),

        "new_training_performed":
            False,

        "new_synthetic_data_generated":
            False,
    }

    return (
        criteria,
        verdict_row,
    )


# =============================================================================
# 12. MAIN
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
        "HFAGM - VERIFY ACTUAL GENERATIVE ARCHITECTURE V2"
    )

    print(
        "=" * 100
    )

    print(
        "\nNo training, inference, or synthetic generation will be performed."
    )

    feature_names = (
        load_feature_schema()
    )

    print(
        f"\nVerified structured feature count: "
        f"{len(feature_names)}"
    )

    # -----------------------------------------------------------------
    # Scan Python files.
    # -----------------------------------------------------------------

    files = python_files()

    python_texts: Dict[
        Path,
        str
    ] = {}

    file_inventory = []

    all_class_rows = []
    all_function_rows = []
    all_semantic_rows = []

    for path in files:

        text = read_text(path)

        if not text:
            continue

        python_texts[
            path
        ] = text

        (
            class_rows,
            function_rows,
            semantic_rows,
        ) = analyze_python_file(
            path,
            text,
        )

        all_class_rows.extend(
            class_rows
        )

        all_function_rows.extend(
            function_rows
        )

        all_semantic_rows.extend(
            semantic_rows
        )

        file_inventory.append(
            {
                "file":
                    relative_path(path),

                "size_bytes":
                    path.stat().st_size,

                "classes":
                    len(
                        class_rows
                    ),

                "functions":
                    len(
                        function_rows
                    ),

                "semantic_evidence_rows":
                    len(
                        semantic_rows
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "python_file_inventory.csv",
        file_inventory,
    )

    write_csv(
        OUTPUT_DIR
        / "class_architecture_inventory.csv",
        all_class_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "function_inventory.csv",
        all_function_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "generative_semantic_evidence.csv",
        all_semantic_rows,
    )

    # -----------------------------------------------------------------
    # Aggregate architectural evidence.
    # -----------------------------------------------------------------

    (
        gan_rows,
        vae_rows,
        diffusion_rows,
        hybrid_rows,
        classifier_graph_rows,
    ) = architecture_evidence(
        all_class_rows,
        all_semantic_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "gan_evidence.csv",
        gan_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "vae_evidence.csv",
        vae_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "diffusion_evidence.csv",
        diffusion_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "hybrid_integration_evidence.csv",
        hybrid_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "classifier_graph_evidence.csv",
        classifier_graph_rows,
    )

    # -----------------------------------------------------------------
    # Structured output evidence.
    # -----------------------------------------------------------------

    structured_rows = (
        structured_output_evidence(
            all_semantic_rows,
            feature_names,
        )
    )

    write_csv(
        OUTPUT_DIR
        / "structured_output_evidence.csv",
        structured_rows,
    )

    # -----------------------------------------------------------------
    # Data generation/save evidence.
    # -----------------------------------------------------------------

    save_rows = (
        generation_save_evidence(
            all_semantic_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "generation_save_evidence.csv",
        save_rows,
    )

    # -----------------------------------------------------------------
    # Checkpoint inventory.
    # -----------------------------------------------------------------

    checkpoint_rows = (
        checkpoint_inventory_v2()
    )

    write_csv(
        OUTPUT_DIR
        / "checkpoint_inventory_v2.csv",
        checkpoint_rows,
    )

    # -----------------------------------------------------------------
    # Checkpoint-code linkage.
    # -----------------------------------------------------------------

    checkpoint_links = (
        checkpoint_code_linkage(
            checkpoint_rows,
            python_texts,
            all_class_rows,
        )
    )

    write_csv(
        OUTPUT_DIR
        / "checkpoint_generative_linkage.csv",
        checkpoint_links,
    )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    (
        verification_matrix,
        verdict,
    ) = build_verdict(
        gan_rows,
        vae_rows,
        diffusion_rows,
        hybrid_rows,
        classifier_graph_rows,
        structured_rows,
        checkpoint_rows,
        checkpoint_links,
        all_semantic_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "architecture_verification_matrix.csv",
        verification_matrix,
    )

    write_csv(
        OUTPUT_DIR
        / "actual_generative_architecture_verdict.csv",
        [
            verdict
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
            (
                "04B_v2_verify_actual_generative_architecture.py"
            ),

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "feature_schema":
            str(
                FEATURE_SCHEMA_PATH
            ),

        "feature_schema_sha256":
            sha256_file(
                FEATURE_SCHEMA_PATH
            ),

        "feature_count":
            len(
                feature_names
            ),

        "python_files_scanned":
            len(
                python_texts
            ),

        "class_definitions":
            len(
                all_class_rows
            ),

        "verified_gan_classes":
            len(
                gan_rows
            ),

        "verified_vae_classes":
            len(
                vae_rows
            ),

        "verified_diffusion_classes":
            len(
                diffusion_rows
            ),

        "verified_hybrid_files":
            len(
                hybrid_rows
            ),

        "checkpoint_files":
            len(
                checkpoint_rows
            ),

        "verdict":
            verdict[
                "verdict"
            ],

        "new_training_performed":
            False,

        "new_inference_performed":
            False,

        "new_synthetic_rows_generated":
            False,

        "arbitrary_project_code_executed":
            False,

        "unknown_pickle_objects_loaded":
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
        / "verification_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    lines = [
        "=" * 100,
        "HFAGM - ACTUAL GENERATIVE ARCHITECTURE VERIFICATION V2",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "QUESTION",
        "-" * 100,
        (
            "Does the recovered project actually implement a GAN, VAE, "
            "diffusion model, or hybrid generative architecture capable "
            "of producing the 51-feature structured COVID records?"
        ),
        "",
        "STRICT EVIDENCE RULES",
        "-" * 100,
        (
            "A class name containing 'HFAGM' is NOT sufficient evidence "
            "of a generative model."
        ),
        (
            "pandas/DataFrame .sample() is NOT treated as synthetic-data "
            "generation."
        ),
        (
            "Encoder, classifier, graph, embedding, and adjacency artifacts "
            "are NOT classified as generator checkpoints."
        ),
        (
            "A checkpoint is useful only if it can be linked to verified "
            "generative code."
        ),
        "",
        "SOURCE AUDIT",
        "-" * 100,
        f"Python files scanned: {len(python_texts)}",
        f"Class definitions found: {len(all_class_rows)}",
        "",
        "VERIFIED GENERATIVE COMPONENTS",
        "-" * 100,
        f"Verified GAN classes: {len(gan_rows)}",
        f"Verified VAE classes: {len(vae_rows)}",
        f"Verified diffusion classes: {len(diffusion_rows)}",
        f"Verified hybrid integration files: {len(hybrid_rows)}",
        "",
        "STRUCTURED OUTPUT",
        "-" * 100,
        (
            f"Explicit 51-feature output evidence: "
            f"{verdict['explicit_51_feature_output']}"
        ),
        (
            f"Latent dimension evidence: "
            f"{verdict['latent_dimension_evidence']}"
        ),
        (
            f"Latent/noise sampling evidence: "
            f"{verdict['latent_noise_sampling']}"
        ),
        (
            f"Explicit generation call evidence: "
            f"{verdict['explicit_generation_call']}"
        ),
        "",
        "CHECKPOINTS",
        "-" * 100,
        (
            f"Plausible generative checkpoints: "
            f"{verdict['plausible_generative_checkpoints']}"
        ),
        (
            f"Valid checkpoint -> verified generative-code links: "
            f"{verdict['valid_generative_checkpoint_code_links']}"
        ),
        "",
        "VERIFICATION MATRIX",
        "-" * 100,
    ]

    for row in verification_matrix:

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
            "MANUSCRIPT CONSEQUENCE",
            "-" * 100,
            verdict[
                "manuscript_consequence"
            ],
            "",
            "SAFETY CONFIRMATION",
            "-" * 100,
            "New training performed: NO",
            "New inference performed: NO",
            "New synthetic data generated: NO",
            "Arbitrary project modules executed: NO",
            "Unknown pickle/joblib models loaded: NO",
            "",
            "PRIMARY OUTPUTS",
            "-" * 100,
            "python_file_inventory.csv",
            "class_architecture_inventory.csv",
            "function_inventory.csv",
            "generative_semantic_evidence.csv",
            "gan_evidence.csv",
            "vae_evidence.csv",
            "diffusion_evidence.csv",
            "hybrid_integration_evidence.csv",
            "classifier_graph_evidence.csv",
            "structured_output_evidence.csv",
            "checkpoint_inventory_v2.csv",
            "checkpoint_generative_linkage.csv",
            "generation_save_evidence.csv",
            "architecture_verification_matrix.csv",
            "actual_generative_architecture_verdict.csv",
            "verification_provenance.csv",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "actual_generative_architecture_summary.txt"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console.
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "04B V2 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nVerified GAN classes: "
        f"{len(gan_rows)}"
    )

    print(
        f"Verified VAE classes: "
        f"{len(vae_rows)}"
    )

    print(
        f"Verified diffusion classes: "
        f"{len(diffusion_rows)}"
    )

    print(
        f"Verified hybrid files: "
        f"{len(hybrid_rows)}"
    )

    print(
        f"\nExplicit 51-feature output: "
        f"{verdict['explicit_51_feature_output']}"
    )

    print(
        f"Plausible generative checkpoints: "
        f"{verdict['plausible_generative_checkpoints']}"
    )

    print(
        f"Valid generative checkpoint/code links: "
        f"{verdict['valid_generative_checkpoint_code_links']}"
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
        "\nMANUSCRIPT CONSEQUENCE:"
    )

    print(
        verdict[
            "manuscript_consequence"
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
        "actual_generative_architecture_summary.txt",
        "actual_generative_architecture_verdict.csv",
        "architecture_verification_matrix.csv",
        "class_architecture_inventory.csv",
        "gan_evidence.csv",
        "vae_evidence.csv",
        "diffusion_evidence.csv",
        "hybrid_integration_evidence.csv",
        "structured_output_evidence.csv",
        "checkpoint_inventory_v2.csv",
        "checkpoint_generative_linkage.csv",
        "generative_semantic_evidence.csv",
        "classifier_graph_evidence.csv",
        "verification_provenance.csv",
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
            "04B V2 FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo training, inference, synthetic generation, "
            "or historical project modification was performed."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(1)