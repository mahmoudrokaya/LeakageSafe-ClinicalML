"""
07_ablation_reproducibility_audit.py
====================================

HFAGM revision - ablation reproducibility forensic audit.

PURPOSE
-------
Reviewer #3 requested ablation experiments covering, at minimum:

    1. full HGF / full HFAGM
    2. GAN-only
    3. VAE-only
    4. diffusion-only
    5. static fusion
    6. adaptive fusion
    7. fairness controller ON
    8. fairness controller OFF
    9. scalability mechanism ON
    10. scalability mechanism OFF

Earlier audits established that the recovered implementation does not contain a
verified structured GAN/VAE/diffusion generator. Therefore this script must NOT
construct those missing ablation conditions.

PRIMARY QUESTION
----------------
Do recoverable project artifacts contain actual executable code and/or
numerical experimental outputs for the claimed ablation conditions?

STRICT RULES
------------
- Do not infer an ablation from manuscript wording alone.
- Do not treat a filename containing "GAN", "VAE", "diffusion", "fusion", or
  "fairness" as proof that an ablation was executed.
- Do not treat classifier hyperparameter changes as generator ablations.
- Do not create missing variants.
- Do not retrain models.
- Do not generate synthetic data.
- Do not treat revision-generated outputs as historical evidence.
- A condition is considered reproducible only if recoverable evidence includes:
      a. executable/implemented condition semantics, AND
      b. numerical output associated with that condition, OR
      c. a preserved trained artifact explicitly tied to that condition.

OUTPUT
------
outputs/revision_ablation/

    ablation_source_inventory.csv
    ablation_keyword_evidence.csv
    architecture_variant_evidence.csv
    fusion_ablation_evidence.csv
    fairness_ablation_evidence.csv
    scalability_ablation_evidence.csv
    numerical_ablation_results.csv
    checkpoint_ablation_linkage.csv
    ablation_condition_matrix.csv
    ablation_claim_reconstruction.csv
    ablation_verdict.csv
    ablation_provenance.csv
    ablation_audit_summary.txt

EXPECTED CONSERVATIVE VERDICT
-----------------------------
ABLATION_STUDY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS

if no valid historical condition/result pairs are found.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
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
    / "revision_ablation"
)

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".log",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".csv",
    ".tsv",
}

RESULT_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".txt",
    ".log",
}

CHECKPOINT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".joblib",
    ".keras",
    ".h5",
    ".hdf5",
    ".onnx",
}

MAX_TEXT_BYTES = 20 * 1024 * 1024

# -------------------------------------------------------------------------
# Exclusions
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
}

EXCLUDED_DIR_PREFIXES = (
    ".venv",
    "venv",
    ".env",
)

REVISION_DIR_MARKERS = (
    "new_code",
    "outputs/revision_primary_metrics",
    "outputs\\revision_primary_metrics",
    "outputs/revision_fairness",
    "outputs\\revision_fairness",
    "outputs/revision_fidelity",
    "outputs\\revision_fidelity",
    "outputs/revision_utility",
    "outputs\\revision_utility",
    "outputs/revision_scalability",
    "outputs\\revision_scalability",
    "outputs/revision_ablation",
    "outputs\\revision_ablation",
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
    "05_",
    "06_",
    "07_",
)

# -------------------------------------------------------------------------
# Ablation conditions
# -------------------------------------------------------------------------

CONDITIONS = {
    "full_hybrid": {
        "display": "Full HGF/HFAGM hybrid",
        "keywords": [
            "full model",
            "full hfagm",
            "full hgf",
            "hybrid model",
            "hybrid generative",
            "gan vae diffusion",
            "gan-vae-diffusion",
        ],
    },

    "gan_only": {
        "display": "GAN-only",
        "keywords": [
            "gan only",
            "gan-only",
            "only gan",
            "without vae",
            "without diffusion",
        ],
    },

    "vae_only": {
        "display": "VAE-only",
        "keywords": [
            "vae only",
            "vae-only",
            "only vae",
            "without gan",
            "without diffusion",
        ],
    },

    "diffusion_only": {
        "display": "Diffusion-only",
        "keywords": [
            "diffusion only",
            "diffusion-only",
            "only diffusion",
            "without gan",
            "without vae",
        ],
    },

    "static_fusion": {
        "display": "Static fusion",
        "keywords": [
            "static fusion",
            "fixed fusion",
            "fixed weights",
            "uniform fusion",
            "equal weights",
        ],
    },

    "adaptive_fusion": {
        "display": "Adaptive fusion",
        "keywords": [
            "adaptive fusion",
            "dynamic fusion",
            "adaptive weights",
            "dynamic weights",
        ],
    },

    "fairness_on": {
        "display": "Fairness controller ON",
        "keywords": [
            "fairness on",
            "fairness enabled",
            "with fairness",
            "fairness controller",
            "fairness regularization",
        ],
    },

    "fairness_off": {
        "display": "Fairness controller OFF",
        "keywords": [
            "fairness off",
            "fairness disabled",
            "without fairness",
            "no fairness",
        ],
    },

    "scalability_on": {
        "display": "Scalability mechanism ON",
        "keywords": [
            "scalability on",
            "scalability enabled",
            "with scalability",
            "scalable mode",
        ],
    },

    "scalability_off": {
        "display": "Scalability mechanism OFF",
        "keywords": [
            "scalability off",
            "scalability disabled",
            "without scalability",
            "no scalability",
        ],
    },
}

# -------------------------------------------------------------------------
# Generic terms
# -------------------------------------------------------------------------

GENERATOR_TERMS = (
    "generator",
    "discriminator",
    "gan",
    "vae",
    "variational",
    "decoder",
    "diffusion",
    "denoiser",
    "noise schedule",
    "reverse diffusion",
)

FUSION_TERMS = (
    "fusion",
    "adaptive",
    "static",
    "weight",
    "weighted",
    "attention",
)

FAIRNESS_TERMS = (
    "fairness",
    "demographic parity",
    "statistical parity",
    "spd",
    "equal opportunity",
    "eod",
    "disparate impact",
    "di",
)

SCALABILITY_TERMS = (
    "scalability",
    "runtime",
    "execution time",
    "memory",
    "gpu memory",
    "throughput",
)

METRIC_TERMS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "roc_auc",
    "fid",
    "sfd",
    "spd",
    "eod",
    "di",
    "runtime",
    "memory",
)

NON_ABLATION_TERMS = (
    "classifier",
    "randomforest",
    "votingclassifier",
    "logistic regression",
    "graph feature",
    "contrastive encoder",
)


# =============================================================================
# 2. HELPERS
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


def read_text(path: Path) -> Optional[str]:
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

    full = str(path).lower()

    if any(
        marker.lower() in full
        for marker in REVISION_DIR_MARKERS
    ):
        return True

    if path.suffix.lower() == ".py":
        name = path.name.lower()

        if any(
            name.startswith(prefix)
            for prefix in REVISION_SCRIPT_PREFIXES
        ):
            return True

    return False


def contains_any(
    text: str,
    terms: Sequence[str],
) -> bool:
    lower = text.lower()

    return any(
        term.lower() in lower
        for term in terms
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
        for line in lines[start:stop]
    )[:3000]


# =============================================================================
# 3. DISCOVERY
# =============================================================================

def discover_files() -> List[Path]:
    files = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if is_excluded(path):
            continue

        if (
            path.suffix.lower()
            not in TEXT_EXTENSIONS
            and
            path.suffix.lower()
            not in CHECKPOINT_EXTENSIONS
        ):
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda p: str(p).lower(),
    )


# =============================================================================
# 4. CONDITION KEYWORD EVIDENCE
# =============================================================================

def find_condition_mentions(
    path: Path,
    text: str,
) -> List[Dict[str, Any]]:
    rows = []

    lines = text.splitlines()

    for line_no, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        for condition_key, config in CONDITIONS.items():
            matched = [
                keyword
                for keyword in config["keywords"]
                if keyword in lower
            ]

            if not matched:
                continue

            context = line_context(
                text,
                line_no,
            )

            metric_context = contains_any(
                context,
                METRIC_TERMS,
            )

            generator_context = contains_any(
                context,
                GENERATOR_TERMS,
            )

            non_ablation_context = contains_any(
                context,
                NON_ABLATION_TERMS,
            )

            rows.append(
                {
                    "file":
                        relative_path(path),

                    "line":
                        line_no,

                    "condition":
                        condition_key,

                    "condition_display":
                        config["display"],

                    "matched_keywords":
                        safe_json(
                            matched
                        ),

                    "metric_context":
                        int(
                            metric_context
                        ),

                    "generator_context":
                        int(
                            generator_context
                        ),

                    "non_ablation_context":
                        int(
                            non_ablation_context
                        ),

                    "text":
                        line.strip()[:2000],

                    "context":
                        context,
                }
            )

    return rows


# =============================================================================
# 5. ARCHITECTURE VARIANT EVIDENCE
# =============================================================================

def inspect_python_architecture(
    path: Path,
    text: str,
) -> List[Dict[str, Any]]:
    rows = []

    try:
        tree = ast.parse(
            text,
            filename=str(path),
        )
    except Exception:
        return rows

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        name = getattr(
            node,
            "name",
            "",
        )

        try:
            source = ast.get_source_segment(
                text,
                node,
            ) or ""
        except Exception:
            source = ""

        lower = source.lower()

        gan = (
            "generator" in lower
            and
            "discriminator" in lower
        )

        vae = (
            (
                "logvar" in lower
                or
                "log_var" in lower
                or
                "reparameter" in lower
            )
            and
            "decoder" in lower
        )

        diffusion = (
            (
                "timestep" in lower
                or
                "timesteps" in lower
            )
            and
            (
                "noise" in lower
                or
                "denois" in lower
            )
        )

        fusion = contains_any(
            lower,
            FUSION_TERMS,
        )

        fairness = contains_any(
            lower,
            FAIRNESS_TERMS,
        )

        scalability = contains_any(
            lower,
            SCALABILITY_TERMS,
        )

        if not any(
            [
                gan,
                vae,
                diffusion,
                fusion,
                fairness,
                scalability,
            ]
        ):
            continue

        rows.append(
            {
                "file":
                    relative_path(path),

                "line":
                    getattr(
                        node,
                        "lineno",
                        np.nan,
                    ),

                "symbol":
                    name,

                "symbol_type":
                    type(node).__name__,

                "gan_semantics":
                    int(gan),

                "vae_semantics":
                    int(vae),

                "diffusion_semantics":
                    int(diffusion),

                "fusion_semantics":
                    int(fusion),

                "fairness_semantics":
                    int(fairness),

                "scalability_semantics":
                    int(scalability),

                "verified_generative_variant":
                    int(
                        gan
                        or
                        vae
                        or
                        diffusion
                    ),
            }
        )

    return rows


# =============================================================================
# 6. NUMERICAL RESULT EXTRACTION
# =============================================================================

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?"
    r"(?:\d+\.\d+|\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z0-9_])"
)


def extract_numerical_results(
    path: Path,
    text: str,
) -> List[Dict[str, Any]]:
    rows = []

    lines = text.splitlines()

    for line_no, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        if not contains_any(
            line,
            METRIC_TERMS,
        ):
            continue

        numbers = NUMBER_PATTERN.findall(
            line
        )

        if not numbers:
            continue

        condition_matches = []

        for condition_key, config in CONDITIONS.items():
            if any(
                keyword in lower
                for keyword in config["keywords"]
            ):
                condition_matches.append(
                    condition_key
                )

        # Also inspect immediate context.
        context = line_context(
            text,
            line_no,
            radius=3,
        )

        if not condition_matches:
            context_lower = context.lower()

            for condition_key, config in CONDITIONS.items():
                if any(
                    keyword in context_lower
                    for keyword in config["keywords"]
                ):
                    condition_matches.append(
                        condition_key
                    )

        rows.append(
            {
                "file":
                    relative_path(path),

                "line":
                    line_no,

                "condition_matches":
                    safe_json(
                        sorted(
                            set(
                                condition_matches
                            )
                        )
                    ),

                "numbers":
                    safe_json(
                        numbers
                    ),

                "text":
                    line.strip()[:2000],

                "context":
                    context,
            }
        )

    return rows


# =============================================================================
# 7. CHECKPOINT CONDITION LINKAGE
# =============================================================================

def checkpoint_condition_linkage(
    path: Path,
) -> List[Dict[str, Any]]:
    lower = str(path).lower()

    matched = []

    for condition_key, config in CONDITIONS.items():
        if any(
            keyword.replace(
                " ",
                "_",
            ) in lower
            or
            keyword.replace(
                " ",
                "-",
            ) in lower
            or
            keyword in lower
            for keyword in config["keywords"]
        ):
            matched.append(
                condition_key
            )

    # Strictly exclude obviously non-generative checkpoints.
    non_generative = contains_any(
        lower,
        (
            "encoder",
            "classifier",
            "ensemble",
            "contrastive",
            "graph",
        ),
    )

    plausible = bool(
        matched
        and
        not non_generative
    )

    return [
        {
            "file":
                relative_path(path),

            "extension":
                path.suffix.lower(),

            "sha256":
                sha256_file(path),

            "condition_matches":
                safe_json(
                    matched
                ),

            "non_generative_hint":
                int(
                    non_generative
                ),

            "plausible_ablation_checkpoint":
                int(
                    plausible
                ),
        }
    ]


# =============================================================================
# 8. CONDITION MATRIX
# =============================================================================

def build_condition_matrix(
    keyword_rows: List[Dict[str, Any]],
    architecture_rows: List[Dict[str, Any]],
    numerical_rows: List[Dict[str, Any]],
    checkpoint_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    matrix = []

    for condition_key, config in CONDITIONS.items():

        keyword_evidence = [
            row
            for row in keyword_rows
            if row[
                "condition"
            ] == condition_key
        ]

        numerical_evidence = []

        for row in numerical_rows:
            try:
                matches = json.loads(
                    row[
                        "condition_matches"
                    ]
                )
            except Exception:
                matches = []

            if condition_key in matches:
                numerical_evidence.append(
                    row
                )

        checkpoint_evidence = []

        for row in checkpoint_rows:
            try:
                matches = json.loads(
                    row[
                        "condition_matches"
                    ]
                )
            except Exception:
                matches = []

            if (
                condition_key in matches
                and
                row[
                    "plausible_ablation_checkpoint"
                ] == 1
            ):
                checkpoint_evidence.append(
                    row
                )

        # -------------------------------------------------------------
        # Architecture-specific evidence.
        # -------------------------------------------------------------

        if condition_key == "gan_only":
            implementation_evidence = [
                row
                for row in architecture_rows
                if row[
                    "gan_semantics"
                ] == 1
            ]

        elif condition_key == "vae_only":
            implementation_evidence = [
                row
                for row in architecture_rows
                if row[
                    "vae_semantics"
                ] == 1
            ]

        elif condition_key == "diffusion_only":
            implementation_evidence = [
                row
                for row in architecture_rows
                if row[
                    "diffusion_semantics"
                ] == 1
            ]

        elif condition_key == "full_hybrid":
            implementation_evidence = [
                row
                for row in architecture_rows
                if (
                    row[
                        "gan_semantics"
                    ]
                    +
                    row[
                        "vae_semantics"
                    ]
                    +
                    row[
                        "diffusion_semantics"
                    ]
                ) >= 2
            ]

        elif condition_key in {
            "static_fusion",
            "adaptive_fusion",
        }:
            implementation_evidence = [
                row
                for row in architecture_rows
                if row[
                    "fusion_semantics"
                ] == 1
            ]

        elif condition_key in {
            "fairness_on",
            "fairness_off",
        }:
            implementation_evidence = [
                row
                for row in architecture_rows
                if row[
                    "fairness_semantics"
                ] == 1
            ]

        elif condition_key in {
            "scalability_on",
            "scalability_off",
        }:
            implementation_evidence = [
                row
                for row in architecture_rows
                if row[
                    "scalability_semantics"
                ] == 1
            ]

        else:
            implementation_evidence = []

        # -------------------------------------------------------------
        # Reproducibility decision.
        # -------------------------------------------------------------

        implementation_found = bool(
            implementation_evidence
        )

        numerical_found = bool(
            numerical_evidence
        )

        checkpoint_found = bool(
            checkpoint_evidence
        )

        reproducible = bool(
            implementation_found
            and
            (
                numerical_found
                or
                checkpoint_found
            )
        )

        matrix.append(
            {
                "condition":
                    condition_key,

                "display":
                    config["display"],

                "keyword_mentions":
                    len(
                        keyword_evidence
                    ),

                "implementation_rows":
                    len(
                        implementation_evidence
                    ),

                "numerical_result_rows":
                    len(
                        numerical_evidence
                    ),

                "plausible_checkpoint_rows":
                    len(
                        checkpoint_evidence
                    ),

                "implementation_found":
                    int(
                        implementation_found
                    ),

                "numerical_result_found":
                    int(
                        numerical_found
                    ),

                "checkpoint_found":
                    int(
                        checkpoint_found
                    ),

                "reproducible_condition":
                    int(
                        reproducible
                    ),

                "status":
                    (
                        "REPRODUCIBLE"
                        if reproducible
                        else
                        "NOT_REPRODUCIBLE"
                    ),
            }
        )

    return matrix


# =============================================================================
# 9. CLAIM RECONSTRUCTION
# =============================================================================

def build_claim_reconstruction(
    condition_matrix: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    lookup = {
        row[
            "condition"
        ]:
            row
        for row in condition_matrix
    }

    claims = [
        (
            "Full HGF compared against GAN-only",
            [
                "full_hybrid",
                "gan_only",
            ],
        ),
        (
            "Full HGF compared against VAE-only",
            [
                "full_hybrid",
                "vae_only",
            ],
        ),
        (
            "Full HGF compared against diffusion-only",
            [
                "full_hybrid",
                "diffusion_only",
            ],
        ),
        (
            "Adaptive fusion compared against static fusion",
            [
                "adaptive_fusion",
                "static_fusion",
            ],
        ),
        (
            "Fairness ON compared against fairness OFF",
            [
                "fairness_on",
                "fairness_off",
            ],
        ),
        (
            "Scalability ON compared against scalability OFF",
            [
                "scalability_on",
                "scalability_off",
            ],
        ),
    ]

    rows = []

    for claim, conditions in claims:
        reproducible = all(
            lookup[
                condition
            ][
                "reproducible_condition"
            ] == 1
            for condition in conditions
        )

        rows.append(
            {
                "claim":
                    claim,

                "required_conditions":
                    safe_json(
                        conditions
                    ),

                "reconstructable":
                    int(
                        reproducible
                    ),

                "verdict":
                    (
                        "REPRODUCIBLE"
                        if reproducible
                        else
                        "NOT_REPRODUCIBLE"
                    ),
            }
        )

    return rows


# =============================================================================
# 10. FINAL VERDICT
# =============================================================================

def build_verdict(
    condition_matrix: List[Dict[str, Any]],
    claim_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reproducible_conditions = [
        row
        for row in condition_matrix
        if row[
            "reproducible_condition"
        ] == 1
    ]

    reproducible_claims = [
        row
        for row in claim_rows
        if row[
            "reconstructable"
        ] == 1
    ]

    required_count = len(
        CONDITIONS
    )

    if len(
        reproducible_conditions
    ) == required_count:

        verdict = (
            "FULL_ABLATION_STUDY_REPRODUCIBLE"
        )

        next_action = (
            "RECOMPUTE_ONLY_VERIFIED_ABLATION_RESULTS"
        )

        manuscript_action = (
            "Use only the recovered, reproducible ablation conditions and "
            "recalculate their metrics from preserved artifacts."
        )

    elif reproducible_conditions:

        verdict = (
            "PARTIAL_ABLATION_EVIDENCE_ONLY"
        )

        next_action = (
            "KEEP_ONLY_VERIFIED_ABLATIONS_REMOVE_UNSUPPORTED_VARIANTS"
        )

        manuscript_action = (
            "Retain only ablation variants for which implementation and "
            "numerical/checkpoint provenance are both recoverable. Remove "
            "unsupported variants and do not create missing historical "
            "experiments."
        )

    else:

        verdict = (
            "ABLATION_STUDY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS"
        )

        next_action = (
            "REMOVE_UNSUPPORTED_ABLATION_CLAIMS"
        )

        manuscript_action = (
            "Remove historical claims comparing full HGF with GAN-only, "
            "VAE-only, diffusion-only, static/adaptive fusion, fairness "
            "on/off, or scalability on/off unless independently preserved "
            "evidence is recovered. Do not generate replacement ablations "
            "and present them as the original study."
        )

    return {
        "verdict":
            verdict,

        "next_action":
            next_action,

        "manuscript_action":
            manuscript_action,

        "required_conditions":
            required_count,

        "reproducible_conditions":
            len(
                reproducible_conditions
            ),

        "reproducible_pairwise_claims":
            len(
                reproducible_claims
            ),

        "new_training_performed":
            False,

        "new_synthetic_data_generated":
            False,

        "new_ablation_experiment_created":
            False,
    }


# =============================================================================
# 11. MAIN
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
        "HFAGM - ABLATION REPRODUCIBILITY FORENSIC AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "\nNo new ablation models will be trained."
    )

    print(
        "No synthetic data will be generated."
    )

    # -----------------------------------------------------------------
    # Discover source files.
    # -----------------------------------------------------------------

    files = discover_files()

    source_inventory = []

    keyword_rows = []

    architecture_rows = []

    numerical_rows = []

    checkpoint_rows = []

    for path in files:

        suffix = path.suffix.lower()

        source_inventory.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "extension":
                    suffix,

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(
                        path
                    ),

                "is_text":
                    int(
                        suffix
                        in TEXT_EXTENSIONS
                    ),

                "is_checkpoint":
                    int(
                        suffix
                        in CHECKPOINT_EXTENSIONS
                    ),
            }
        )

        # -------------------------------------------------------------
        # Checkpoints.
        # -------------------------------------------------------------

        if suffix in CHECKPOINT_EXTENSIONS:
            checkpoint_rows.extend(
                checkpoint_condition_linkage(
                    path
                )
            )

        # -------------------------------------------------------------
        # Text/code evidence.
        # -------------------------------------------------------------

        if suffix not in TEXT_EXTENSIONS:
            continue

        text = read_text(
            path
        )

        if not text:
            continue

        keyword_rows.extend(
            find_condition_mentions(
                path,
                text,
            )
        )

        numerical_rows.extend(
            extract_numerical_results(
                path,
                text,
            )
        )

        if suffix == ".py":
            architecture_rows.extend(
                inspect_python_architecture(
                    path,
                    text,
                )
            )

    # -----------------------------------------------------------------
    # Outputs.
    # -----------------------------------------------------------------

    write_csv(
        OUTPUT_DIR
        / "ablation_source_inventory.csv",
        source_inventory,
    )

    write_csv(
        OUTPUT_DIR
        / "ablation_keyword_evidence.csv",
        keyword_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "architecture_variant_evidence.csv",
        architecture_rows,
    )

    fusion_rows = [
        row
        for row in architecture_rows
        if row[
            "fusion_semantics"
        ] == 1
    ]

    fairness_rows = [
        row
        for row in architecture_rows
        if row[
            "fairness_semantics"
        ] == 1
    ]

    scalability_rows = [
        row
        for row in architecture_rows
        if row[
            "scalability_semantics"
        ] == 1
    ]

    write_csv(
        OUTPUT_DIR
        / "fusion_ablation_evidence.csv",
        fusion_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "fairness_ablation_evidence.csv",
        fairness_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "scalability_ablation_evidence.csv",
        scalability_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "numerical_ablation_results.csv",
        numerical_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "checkpoint_ablation_linkage.csv",
        checkpoint_rows,
    )

    # -----------------------------------------------------------------
    # Condition matrix.
    # -----------------------------------------------------------------

    condition_matrix = build_condition_matrix(
        keyword_rows,
        architecture_rows,
        numerical_rows,
        checkpoint_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "ablation_condition_matrix.csv",
        condition_matrix,
    )

    claim_rows = build_claim_reconstruction(
        condition_matrix
    )

    write_csv(
        OUTPUT_DIR
        / "ablation_claim_reconstruction.csv",
        claim_rows,
    )

    verdict = build_verdict(
        condition_matrix,
        claim_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "ablation_verdict.csv",
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
            "07_ablation_reproducibility_audit.py",

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "files_scanned":
            len(
                source_inventory
            ),

        "keyword_evidence_rows":
            len(
                keyword_rows
            ),

        "architecture_variant_rows":
            len(
                architecture_rows
            ),

        "numerical_result_rows":
            len(
                numerical_rows
            ),

        "checkpoint_rows":
            len(
                checkpoint_rows
            ),

        "required_ablation_conditions":
            len(
                CONDITIONS
            ),

        "reproducible_conditions":
            verdict[
                "reproducible_conditions"
            ],

        "verdict":
            verdict[
                "verdict"
            ],

        "new_training_performed":
            False,

        "new_synthetic_generation_performed":
            False,

        "new_ablation_experiment_created":
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
        / "ablation_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    lines = [
        "=" * 100,
        "HFAGM - ABLATION REPRODUCIBILITY FORENSIC AUDIT",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "PURPOSE",
        "-" * 100,
        (
            "Determine whether the ablation conditions claimed or requested "
            "for the manuscript are recoverable from historical code, model "
            "artifacts, and numerical outputs."
        ),
        (
            "No missing condition is created by this script."
        ),
        "",
        "SOURCE AUDIT",
        "-" * 100,
        (
            f"Files scanned: "
            f"{len(source_inventory)}"
        ),
        (
            f"Keyword evidence rows: "
            f"{len(keyword_rows)}"
        ),
        (
            f"Architecture evidence rows: "
            f"{len(architecture_rows)}"
        ),
        (
            f"Numerical result rows: "
            f"{len(numerical_rows)}"
        ),
        (
            f"Checkpoint rows: "
            f"{len(checkpoint_rows)}"
        ),
        "",
        "CONDITION MATRIX",
        "-" * 100,
    ]

    for row in condition_matrix:

        state = (
            "PASS"
            if row[
                "reproducible_condition"
            ] == 1
            else
            "MISSING"
        )

        lines.append(
            f"{state}: {row['display']}"
        )

        lines.append(
            (
                f"    keyword mentions = "
                f"{row['keyword_mentions']}"
            )
        )

        lines.append(
            (
                f"    implementation rows = "
                f"{row['implementation_rows']}"
            )
        )

        lines.append(
            (
                f"    numerical result rows = "
                f"{row['numerical_result_rows']}"
            )
        )

        lines.append(
            (
                f"    checkpoint rows = "
                f"{row['plausible_checkpoint_rows']}"
            )
        )

    lines.extend(
        [
            "",
            "PAIRWISE ABLATION CLAIMS",
            "-" * 100,
        ]
    )

    for row in claim_rows:

        lines.append(
            f"{row['verdict']}: {row['claim']}"
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
            "INTERPRETATION",
            "-" * 100,
            (
                "Ablation evidence requires an actual implemented variant plus "
                "a corresponding historical result or preserved trained "
                "artifact."
            ),
            (
                "Textual mentions, proposed methodology diagrams, or parameter "
                "names alone do not establish that an ablation experiment was "
                "executed."
            ),
            (
                "Classifier hyperparameter variants must not be relabeled as "
                "GAN/VAE/diffusion generator ablations."
            ),
            "",
            "SAFETY CONFIRMATION",
            "-" * 100,
            "New models trained: NO",
            "New synthetic rows generated: NO",
            "New ablation variants created: NO",
            "Historical files modified: NO",
            "",
            "PRIMARY OUTPUTS",
            "-" * 100,
            "ablation_source_inventory.csv",
            "ablation_keyword_evidence.csv",
            "architecture_variant_evidence.csv",
            "fusion_ablation_evidence.csv",
            "fairness_ablation_evidence.csv",
            "scalability_ablation_evidence.csv",
            "numerical_ablation_results.csv",
            "checkpoint_ablation_linkage.csv",
            "ablation_condition_matrix.csv",
            "ablation_claim_reconstruction.csv",
            "ablation_verdict.csv",
            "ablation_provenance.csv",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "ablation_audit_summary.txt"
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
        "07 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nRequired ablation conditions: "
        f"{len(CONDITIONS)}"
    )

    print(
        f"Reproducible conditions: "
        f"{verdict['reproducible_conditions']}"
    )

    print(
        f"Reproducible pairwise claims: "
        f"{verdict['reproducible_pairwise_claims']}"
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
        "ablation_audit_summary.txt",
        "ablation_verdict.csv",
        "ablation_condition_matrix.csv",
        "ablation_claim_reconstruction.csv",
        "architecture_variant_evidence.csv",
        "numerical_ablation_results.csv",
        "checkpoint_ablation_linkage.csv",
        "fusion_ablation_evidence.csv",
        "fairness_ablation_evidence.csv",
        "scalability_ablation_evidence.csv",
        "ablation_provenance.csv",
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
            "07 FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print(
            "\nNo model was trained, no synthetic rows were generated, "
            "and no historical project files were modified."
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        sys.exit(1)