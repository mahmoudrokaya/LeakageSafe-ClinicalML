"""
06_scalability_benchmark.py
===========================

HFAGM revision - scalability / computational-efficiency forensic audit.

PURPOSE
-------
Reviewer #3 requested clarification and reproducibility for scalability claims,
including workload size, runtime, memory use, hardware, and the meaning of the
reported efficiency improvement.

Earlier forensic stages established:

    - no verified structured GAN;
    - no verified structured VAE;
    - no verified structured diffusion model;
    - no verified hybrid GAN-VAE-diffusion generator;
    - no recoverable 51-feature synthetic table with provenance.

Therefore this script MUST NOT fabricate a new synthetic-generation benchmark.

PRIMARY QUESTIONS
-----------------
1. Is there existing executable generative code capable of producing the
   51-feature COVID records?

2. Are there historical runtime measurements associated with that actual
   generator?

3. Are there historical GPU-memory measurements associated with that actual
   generator?

4. Are workload sizes such as N=500 and N=100000 present in genuine
   experimental records rather than manuscript/audit text?

5. Are the claimed efficiency improvements:
       ~25% faster
       ~20-30% lower GPU memory
   numerically reconstructable from existing results?

6. Which hardware configuration was actually used?

STRICT RULE
-----------
Classifier, graph, encoder, or preprocessing speed MUST NOT be substituted for
synthetic-generator speed.

Duplicating/resampling the 193-row dataset MUST NOT be interpreted as increasing
independent information.

If no genuine generator and corresponding measurement provenance exist, the
correct verdict is:

    GENERATIVE_SCALABILITY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS

OUTPUT
------
outputs/revision_scalability/

    scalability_source_inventory.csv
    scalability_numeric_evidence.csv
    workload_size_evidence.csv
    runtime_evidence.csv
    memory_evidence.csv
    hardware_evidence.csv
    generative_scalability_linkage.csv
    efficiency_claim_reconstruction.csv
    scalability_verification_matrix.csv
    scalability_verdict.csv
    scalability_provenance.csv
    scalability_audit_summary.txt
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import sys
import traceback

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
    / "revision_scalability"
)

EXPECTED_FEATURE_COUNT = 51

# -------------------------------------------------------------------------
# Historical manuscript claims being audited.
# -------------------------------------------------------------------------

CLAIM_WORKLOAD_MIN = 500
CLAIM_WORKLOAD_MAX = 100000

CLAIM_RUNTIME_REDUCTION_PERCENT = 25.0

CLAIM_MEMORY_REDUCTION_MIN_PERCENT = 20.0
CLAIM_MEMORY_REDUCTION_MAX_PERCENT = 30.0

# Manuscript also reportedly contains a conflicting +40% GPU-memory statement.
CLAIM_MEMORY_INCREASE_PERCENT = 40.0


# =============================================================================
# 2. SEARCH / FILTER CONFIGURATION
# =============================================================================

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

MODEL_EXTENSIONS = {
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
# Do not use revision audit outputs as historical evidence.
# -------------------------------------------------------------------------

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
)

# -------------------------------------------------------------------------
# Terms.
# -------------------------------------------------------------------------

GENERATOR_TERMS = (
    "generator",
    "generative",
    "gan",
    "vae",
    "variational",
    "diffusion",
    "denoiser",
    "synthetic generation",
    "generate_synthetic",
    "generate_samples",
)

CLASSIFIER_TERMS = (
    "classifier",
    "votingclassifier",
    "randomforest",
    "logisticregression",
    "gradientboosting",
    "predict_proba",
)

RUNTIME_TERMS = (
    "runtime",
    "execution time",
    "elapsed",
    "duration",
    "wall time",
    "wall-clock",
    "wall_clock",
    "seconds",
    "sec",
    "time.time",
    "perf_counter",
    "cuda event",
)

MEMORY_TERMS = (
    "memory",
    "gpu memory",
    "vram",
    "max_memory_allocated",
    "memory_allocated",
    "memory_reserved",
    "nvidia-smi",
    "allocated memory",
    "peak memory",
)

HARDWARE_TERMS = (
    "rtx",
    "gpu",
    "cuda",
    "nvidia",
    "geforce",
    "tesla",
    "quadro",
    "cpu",
    "intel",
    "amd",
)

WORKLOAD_TERMS = (
    "n_samples",
    "num_samples",
    "sample_size",
    "synthetic_size",
    "n_synthetic",
    "generation_size",
    "dataset_size",
    "batch_size",
)

MAX_TEXT_BYTES = 20 * 1024 * 1024


# =============================================================================
# 3. HELPERS
# =============================================================================

def normalize_path(path: Path) -> str:
    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        )
    except Exception:
        return str(path)


def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


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


def line_context(
    text: str,
    line_number: int,
    radius: int = 2,
) -> str:
    lines = text.splitlines()

    if line_number < 1:
        return ""

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
    )[:3000]


def contains_any(
    text: str,
    terms: Sequence[str],
) -> bool:
    lower = text.lower()

    return any(
        term.lower() in lower
        for term in terms
    )


# =============================================================================
# 4. FILE INVENTORY
# =============================================================================

def discover_source_files() -> List[Path]:
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
            not in MODEL_EXTENSIONS
        ):
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda item: str(item).lower(),
    )


# =============================================================================
# 5. NUMERIC EVIDENCE EXTRACTION
# =============================================================================

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?"
    r"(?:\d+\.\d+|\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z0-9_])"
)


def extract_numeric_evidence(
    path: Path,
    text: str,
) -> List[Dict[str, Any]]:
    rows = []

    lines = text.splitlines()

    for index, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        categories = []

        if contains_any(
            line,
            RUNTIME_TERMS,
        ):
            categories.append("runtime")

        if contains_any(
            line,
            MEMORY_TERMS,
        ):
            categories.append("memory")

        if contains_any(
            line,
            HARDWARE_TERMS,
        ):
            categories.append("hardware")

        if contains_any(
            line,
            WORKLOAD_TERMS,
        ):
            categories.append("workload")

        if contains_any(
            line,
            GENERATOR_TERMS,
        ):
            categories.append("generative")

        if not categories:
            continue

        numbers = NUMBER_PATTERN.findall(
            line
        )

        rows.append(
            {
                "file":
                    normalize_path(path),

                "line":
                    index,

                "categories":
                    safe_json(categories),

                "numbers":
                    safe_json(numbers),

                "text":
                    line.strip()[:2000],

                "context":
                    line_context(
                        text,
                        index,
                    ),
            }
        )

    return rows


# =============================================================================
# 6. SPECIALIZED EVIDENCE
# =============================================================================

def extract_workload_evidence(
    numeric_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for row in numeric_rows:
        categories = row[
            "categories"
        ]

        if "workload" not in categories:
            continue

        values = []

        try:
            values = json.loads(
                row["numbers"]
            )
        except Exception:
            pass

        for value in values:
            try:
                number = float(value)
            except Exception:
                continue

            if number < 1:
                continue

            rows.append(
                {
                    "file":
                        row[
                            "file"
                        ],

                    "line":
                        row[
                            "line"
                        ],

                    "value":
                        number,

                    "is_claim_500":
                        int(
                            math.isclose(
                                number,
                                CLAIM_WORKLOAD_MIN,
                            )
                        ),

                    "is_claim_100000":
                        int(
                            math.isclose(
                                number,
                                CLAIM_WORKLOAD_MAX,
                            )
                        ),

                    "text":
                        row[
                            "text"
                        ],

                    "context":
                        row[
                            "context"
                        ],
                }
            )

    return rows


def extract_runtime_evidence(
    numeric_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for row in numeric_rows:
        if "runtime" not in row[
            "categories"
        ]:
            continue

        generative_context = contains_any(
            row["context"],
            GENERATOR_TERMS,
        )

        classifier_context = contains_any(
            row["context"],
            CLASSIFIER_TERMS,
        )

        rows.append(
            {
                "file":
                    row[
                        "file"
                    ],

                "line":
                    row[
                        "line"
                    ],

                "generative_context":
                    int(
                        generative_context
                    ),

                "classifier_context":
                    int(
                        classifier_context
                    ),

                "numbers":
                    row[
                        "numbers"
                    ],

                "text":
                    row[
                        "text"
                    ],

                "context":
                    row[
                        "context"
                    ],
            }
        )

    return rows


def extract_memory_evidence(
    numeric_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for row in numeric_rows:
        if "memory" not in row[
            "categories"
        ]:
            continue

        generative_context = contains_any(
            row["context"],
            GENERATOR_TERMS,
        )

        rows.append(
            {
                "file":
                    row[
                        "file"
                    ],

                "line":
                    row[
                        "line"
                    ],

                "generative_context":
                    int(
                        generative_context
                    ),

                "numbers":
                    row[
                        "numbers"
                    ],

                "text":
                    row[
                        "text"
                    ],

                "context":
                    row[
                        "context"
                    ],
            }
        )

    return rows


def extract_hardware_evidence(
    numeric_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for row in numeric_rows:
        if "hardware" not in row[
            "categories"
        ]:
            continue

        rows.append(
            {
                "file":
                    row[
                        "file"
                    ],

                "line":
                    row[
                        "line"
                    ],

                "text":
                    row[
                        "text"
                    ],

                "context":
                    row[
                        "context"
                    ],
            }
        )

    return rows


# =============================================================================
# 7. GENERATIVE IMPLEMENTATION AUDIT
# =============================================================================

def inspect_python_for_generator(
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
            ast.ClassDef,
        ):
            continue

        class_name = node.name

        method_names = [
            child.name.lower()
            for child in node.body
            if isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
        ]

        source = ""

        try:
            source = ast.get_source_segment(
                text,
                node,
            ) or ""
        except Exception:
            pass

        lower_source = source.lower()

        has_generator = (
            "generator" in class_name.lower()
            or
            "generator" in lower_source
        )

        has_discriminator = (
            "discriminator" in lower_source
        )

        has_vae = (
            (
                "logvar" in lower_source
                or
                "log_var" in lower_source
            )
            and
            "decoder" in lower_source
        )

        has_diffusion = (
            (
                "timestep" in lower_source
                or
                "timesteps" in lower_source
            )
            and
            (
                "noise" in lower_source
                or
                "denois" in lower_source
            )
        )

        has_generation_method = any(
            method in {
                "generate",
                "generate_samples",
                "generate_synthetic",
                "synthesize",
                "decode",
                "p_sample",
                "reverse_diffusion",
            }
            for method in method_names
        )

        verified = bool(
            (
                has_generator
                and
                has_discriminator
            )
            or
            has_vae
            or
            has_diffusion
        )

        rows.append(
            {
                "file":
                    normalize_path(path),

                "line":
                    node.lineno,

                "class_name":
                    class_name,

                "methods":
                    safe_json(
                        method_names
                    ),

                "gan_semantics":
                    int(
                        has_generator
                        and
                        has_discriminator
                    ),

                "vae_semantics":
                    int(
                        has_vae
                    ),

                "diffusion_semantics":
                    int(
                        has_diffusion
                    ),

                "generation_method":
                    int(
                        has_generation_method
                    ),

                "verified_generative_architecture":
                    int(
                        verified
                    ),
            }
        )

    return rows


# =============================================================================
# 8. EFFICIENCY CLAIM RECONSTRUCTION
# =============================================================================

def reconstruct_efficiency_claims(
    runtime_rows: List[Dict[str, Any]],
    memory_rows: List[Dict[str, Any]],
    workload_rows: List[Dict[str, Any]],
    generator_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    verified_generator_files = {
        row[
            "file"
        ]
        for row in generator_rows
        if row[
            "verified_generative_architecture"
        ] == 1
    }

    runtime_generator_rows = [
        row
        for row in runtime_rows
        if row[
            "file"
        ]
        in verified_generator_files
        and
        row[
            "generative_context"
        ] == 1
    ]

    memory_generator_rows = [
        row
        for row in memory_rows
        if row[
            "file"
        ]
        in verified_generator_files
        and
        row[
            "generative_context"
        ] == 1
    ]

    workload_generator_rows = [
        row
        for row in workload_rows
        if row[
            "file"
        ]
        in verified_generator_files
    ]

    evidence_500 = any(
        row[
            "is_claim_500"
        ] == 1
        for row in workload_generator_rows
    )

    evidence_100000 = any(
        row[
            "is_claim_100000"
        ] == 1
        for row in workload_generator_rows
    )

    rows = [
        {
            "claim":
                "synthetic workload includes N=500",

            "historical_claim_value":
                500,

            "reconstructable":
                int(
                    evidence_500
                ),

            "supporting_rows":
                sum(
                    row[
                        "is_claim_500"
                    ]
                    for row in workload_generator_rows
                ),

            "verdict":
                (
                    "SUPPORTED"
                    if evidence_500
                    else
                    "NOT_REPRODUCIBLE"
                ),
        },
        {
            "claim":
                "synthetic workload includes N=100000",

            "historical_claim_value":
                100000,

            "reconstructable":
                int(
                    evidence_100000
                ),

            "supporting_rows":
                sum(
                    row[
                        "is_claim_100000"
                    ]
                    for row in workload_generator_rows
                ),

            "verdict":
                (
                    "SUPPORTED"
                    if evidence_100000
                    else
                    "NOT_REPRODUCIBLE"
                ),
        },
        {
            "claim":
                "approximately 25 percent faster",

            "historical_claim_value":
                25.0,

            "reconstructable":
                0,

            "supporting_rows":
                len(
                    runtime_generator_rows
                ),

            "verdict":
                (
                    "RAW_RUNTIME_EVIDENCE_REQUIRES_MANUAL_RECONSTRUCTION"
                    if len(
                        runtime_generator_rows
                    ) >= 2
                    else
                    "NOT_REPRODUCIBLE"
                ),
        },
        {
            "claim":
                "20-30 percent lower GPU memory",

            "historical_claim_value":
                "20-30%",

            "reconstructable":
                0,

            "supporting_rows":
                len(
                    memory_generator_rows
                ),

            "verdict":
                (
                    "RAW_MEMORY_EVIDENCE_REQUIRES_MANUAL_RECONSTRUCTION"
                    if len(
                        memory_generator_rows
                    ) >= 2
                    else
                    "NOT_REPRODUCIBLE"
                ),
        },
        {
            "claim":
                "approximately 40 percent greater GPU memory",

            "historical_claim_value":
                "+40%",

            "reconstructable":
                0,

            "supporting_rows":
                len(
                    memory_generator_rows
                ),

            "verdict":
                (
                    "CONFLICT_REQUIRES_RAW_MEASUREMENT_AUDIT"
                    if len(
                        memory_generator_rows
                    ) >= 2
                    else
                    "NOT_REPRODUCIBLE"
                ),
        },
    ]

    return rows


# =============================================================================
# 9. VERIFICATION MATRIX / FINAL VERDICT
# =============================================================================

def build_verification(
    generator_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    memory_rows: List[Dict[str, Any]],
    workload_rows: List[Dict[str, Any]],
    hardware_rows: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    verified_generator_files = {
        row[
            "file"
        ]
        for row in generator_rows
        if row[
            "verified_generative_architecture"
        ] == 1
    }

    generator_exists = bool(
        verified_generator_files
    )

    generator_runtime = [
        row
        for row in runtime_rows
        if row[
            "file"
        ]
        in verified_generator_files
    ]

    generator_memory = [
        row
        for row in memory_rows
        if row[
            "file"
        ]
        in verified_generator_files
    ]

    generator_workload = [
        row
        for row in workload_rows
        if row[
            "file"
        ]
        in verified_generator_files
    ]

    evidence_500 = any(
        row[
            "is_claim_500"
        ] == 1
        for row in generator_workload
    )

    evidence_100000 = any(
        row[
            "is_claim_100000"
        ] == 1
        for row in generator_workload
    )

    matrix = [
        {
            "criterion":
                "verified_structured_generative_architecture",

            "passed":
                int(
                    generator_exists
                ),

            "detail":
                (
                    f"{len(verified_generator_files)} verified "
                    "generative implementation file(s)."
                ),
        },
        {
            "criterion":
                "generator_runtime_measurements",

            "passed":
                int(
                    bool(
                        generator_runtime
                    )
                ),

            "detail":
                (
                    f"{len(generator_runtime)} runtime evidence row(s) "
                    "linked to verified generator code."
                ),
        },
        {
            "criterion":
                "generator_gpu_memory_measurements",

            "passed":
                int(
                    bool(
                        generator_memory
                    )
                ),

            "detail":
                (
                    f"{len(generator_memory)} memory evidence row(s) "
                    "linked to verified generator code."
                ),
        },
        {
            "criterion":
                "workload_N_500_provenance",

            "passed":
                int(
                    evidence_500
                ),

            "detail":
                (
                    "N=500 found in verified generator workload evidence."
                    if evidence_500
                    else
                    "No verified generator workload evidence for N=500."
                ),
        },
        {
            "criterion":
                "workload_N_100000_provenance",

            "passed":
                int(
                    evidence_100000
                ),

            "detail":
                (
                    "N=100000 found in verified generator workload evidence."
                    if evidence_100000
                    else
                    "No verified generator workload evidence for N=100000."
                ),
        },
        {
            "criterion":
                "historical_hardware_documentation",

            "passed":
                int(
                    bool(
                        hardware_rows
                    )
                ),

            "detail":
                (
                    f"{len(hardware_rows)} hardware-related evidence row(s) "
                    "found; manual consistency assessment still required."
                ),
        },
    ]

    # -----------------------------------------------------------------
    # Most important gating rule.
    # -----------------------------------------------------------------

    if not generator_exists:

        verdict = (
            "GENERATIVE_SCALABILITY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS"
        )

        next_action = (
            "REMOVE_UNSUPPORTED_RUNTIME_MEMORY_AND_SYNTHETIC_SCALE_CLAIMS"
        )

        manuscript_action = (
            "Remove quantitative synthetic-generation scalability claims, "
            "including N=500 to N=100000, approximately 25% faster execution, "
            "20-30% lower GPU memory, and any conflicting +40% memory claim, "
            "unless independently documented raw measurements are recovered. "
            "Do not replace these measurements with classifier or preprocessing "
            "benchmarks."
        )

    elif (
        not generator_runtime
        or
        not generator_memory
    ):

        verdict = (
            "GENERATOR_FOUND_BUT_SCALABILITY_MEASUREMENTS_INCOMPLETE"
        )

        next_action = (
            "INSPECT_HISTORICAL_LOGS_BEFORE_ANY_NEW_BENCHMARK"
        )

        manuscript_action = (
            "A generator implementation exists, but historical runtime/memory "
            "measurements are incomplete. Do not retain quantitative efficiency "
            "claims until raw measurements are reconstructed."
        )

    else:

        verdict = (
            "HISTORICAL_SCALABILITY_EVIDENCE_REQUIRES_MANUAL_RECONSTRUCTION"
        )

        next_action = (
            "RECONSTRUCT_MATCHED_RUNTIME_MEMORY_COMPARISONS"
        )

        manuscript_action = (
            "Raw scalability evidence may exist. Reconstruct matched baseline "
            "and proposed-model measurements before reporting percentage "
            "improvements."
        )

    verdict_row = {
        "verdict":
            verdict,

        "next_action":
            next_action,

        "manuscript_action":
            manuscript_action,

        "verified_generative_architecture":
            int(
                generator_exists
            ),

        "verified_generator_files":
            len(
                verified_generator_files
            ),

        "generator_runtime_rows":
            len(
                generator_runtime
            ),

        "generator_memory_rows":
            len(
                generator_memory
            ),

        "generator_workload_rows":
            len(
                generator_workload
            ),

        "hardware_evidence_rows":
            len(
                hardware_rows
            ),

        "N500_verified":
            int(
                evidence_500
            ),

        "N100000_verified":
            int(
                evidence_100000
            ),

        "new_model_training_performed":
            False,

        "new_synthetic_rows_generated":
            False,

        "new_runtime_benchmark_performed":
            False,

        "new_gpu_memory_benchmark_performed":
            False,
    }

    return (
        matrix,
        verdict_row,
    )


# =============================================================================
# 10. HARDWARE SUMMARY
# =============================================================================

def summarize_hardware(
    hardware_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    combined = " ".join(
        row[
            "context"
        ]
        for row in hardware_rows
    ).lower()

    gpu_models = []

    patterns = [
        r"rtx\s*6000",
        r"rtx\s*3080",
        r"rtx\s*3090",
        r"rtx\s*4090",
        r"a100",
        r"v100",
        r"t4",
    ]

    for pattern in patterns:
        if re.search(
            pattern,
            combined,
            flags=re.IGNORECASE,
        ):
            gpu_models.append(
                pattern
                .replace(
                    r"\s*",
                    " ",
                )
            )

    return {
        "unique_detected_gpu_terms":
            safe_json(
                sorted(
                    set(
                        gpu_models
                    )
                )
            ),

        "multiple_gpu_models_detected":
            int(
                len(
                    set(
                        gpu_models
                    )
                ) > 1
            ),

        "hardware_evidence_rows":
            len(
                hardware_rows
            ),
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
        "HFAGM - SCALABILITY / COMPUTATIONAL EFFICIENCY AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "\nImportant: no new generator, synthetic rows, runtime benchmark, "
        "or GPU-memory benchmark will be created."
    )

    source_paths = discover_source_files()

    source_inventory = []

    numeric_rows = []

    generator_rows = []

    for path in source_paths:

        suffix = path.suffix.lower()

        source_inventory.append(
            {
                "file":
                    normalize_path(
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

                "is_text_source":
                    int(
                        suffix
                        in TEXT_EXTENSIONS
                    ),

                "is_model_artifact":
                    int(
                        suffix
                        in MODEL_EXTENSIONS
                    ),
            }
        )

        if suffix not in TEXT_EXTENSIONS:
            continue

        text = read_text(
            path
        )

        if not text:
            continue

        numeric_rows.extend(
            extract_numeric_evidence(
                path,
                text,
            )
        )

        if suffix == ".py":
            generator_rows.extend(
                inspect_python_for_generator(
                    path,
                    text,
                )
            )

    write_csv(
        OUTPUT_DIR
        / "scalability_source_inventory.csv",
        source_inventory,
    )

    write_csv(
        OUTPUT_DIR
        / "scalability_numeric_evidence.csv",
        numeric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "generative_scalability_linkage.csv",
        generator_rows,
    )

    # -----------------------------------------------------------------
    # Specialized evidence.
    # -----------------------------------------------------------------

    workload_rows = extract_workload_evidence(
        numeric_rows
    )

    runtime_rows = extract_runtime_evidence(
        numeric_rows
    )

    memory_rows = extract_memory_evidence(
        numeric_rows
    )

    hardware_rows = extract_hardware_evidence(
        numeric_rows
    )

    write_csv(
        OUTPUT_DIR
        / "workload_size_evidence.csv",
        workload_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "runtime_evidence.csv",
        runtime_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "memory_evidence.csv",
        memory_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "hardware_evidence.csv",
        hardware_rows,
    )

    # -----------------------------------------------------------------
    # Historical efficiency claims.
    # -----------------------------------------------------------------

    efficiency_rows = reconstruct_efficiency_claims(
        runtime_rows,
        memory_rows,
        workload_rows,
        generator_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "efficiency_claim_reconstruction.csv",
        efficiency_rows,
    )

    # -----------------------------------------------------------------
    # Verification / verdict.
    # -----------------------------------------------------------------

    matrix, verdict = build_verification(
        generator_rows,
        runtime_rows,
        memory_rows,
        workload_rows,
        hardware_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "scalability_verification_matrix.csv",
        matrix,
    )

    write_csv(
        OUTPUT_DIR
        / "scalability_verdict.csv",
        [
            verdict
        ],
    )

    hardware_summary = summarize_hardware(
        hardware_rows
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
            "06_scalability_benchmark.py",

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "source_files_scanned":
            len(
                source_inventory
            ),

        "numeric_evidence_rows":
            len(
                numeric_rows
            ),

        "runtime_evidence_rows":
            len(
                runtime_rows
            ),

        "memory_evidence_rows":
            len(
                memory_rows
            ),

        "hardware_evidence_rows":
            len(
                hardware_rows
            ),

        "workload_evidence_rows":
            len(
                workload_rows
            ),

        "verified_generative_classes":
            sum(
                row[
                    "verified_generative_architecture"
                ]
                for row in generator_rows
            ),

        "verdict":
            verdict[
                "verdict"
            ],

        "new_training_performed":
            False,

        "new_synthetic_generation_performed":
            False,

        "new_runtime_benchmark_performed":
            False,

        "new_gpu_memory_benchmark_performed":
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
        / "scalability_provenance.csv",
        [
            provenance
        ],
    )

    # -----------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------

    verified_generator_files = sorted(
        {
            row[
                "file"
            ]
            for row in generator_rows
            if row[
                "verified_generative_architecture"
            ] == 1
        }
    )

    lines = [
        "=" * 100,
        "HFAGM - SCALABILITY / COMPUTATIONAL EFFICIENCY FORENSIC AUDIT",
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        "",
        "PURPOSE",
        "-" * 100,
        (
            "Determine whether historical synthetic-generation scalability "
            "claims can be reconstructed from existing project artifacts."
        ),
        (
            "No new scalability experiment is created when the original "
            "generative architecture is not recoverable."
        ),
        "",
        "SOURCE AUDIT",
        "-" * 100,
        (
            f"Historical/relevant files scanned: "
            f"{len(source_inventory)}"
        ),
        (
            f"Numeric evidence rows: "
            f"{len(numeric_rows)}"
        ),
        "",
        "GENERATIVE IMPLEMENTATION",
        "-" * 100,
        (
            f"Verified generative architecture files: "
            f"{len(verified_generator_files)}"
        ),
    ]

    if verified_generator_files:
        for filename in verified_generator_files:
            lines.append(
                f"  - {filename}"
            )
    else:
        lines.append(
            "  None."
        )

    lines.extend(
        [
            "",
            "SCALABILITY EVIDENCE",
            "-" * 100,
            (
                f"Workload-related evidence rows: "
                f"{len(workload_rows)}"
            ),
            (
                f"Runtime-related evidence rows: "
                f"{len(runtime_rows)}"
            ),
            (
                f"Memory-related evidence rows: "
                f"{len(memory_rows)}"
            ),
            (
                f"Hardware-related evidence rows: "
                f"{len(hardware_rows)}"
            ),
            "",
            "HARDWARE CONSISTENCY",
            "-" * 100,
            (
                "Detected GPU terms: "
                + hardware_summary[
                    "unique_detected_gpu_terms"
                ]
            ),
            (
                "Multiple GPU configurations detected: "
                + (
                    "YES"
                    if hardware_summary[
                        "multiple_gpu_models_detected"
                    ] == 1
                    else
                    "NO"
                )
            ),
            "",
            "HISTORICAL CLAIM AUDIT",
            "-" * 100,
        ]
    )

    for row in efficiency_rows:
        lines.extend(
            [
                f"Claim: {row['claim']}",
                f"  Historical value: {row['historical_claim_value']}",
                f"  Verdict: {row['verdict']}",
                (
                    f"  Supporting generator-linked rows: "
                    f"{row['supporting_rows']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "VERIFICATION MATRIX",
            "-" * 100,
        ]
    )

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
                "A larger requested synthetic N would represent computational "
                "workload only. It would not increase the number of independent "
                "clinical observations beyond the original 193 patients."
            ),
            (
                "Classifier inference speed, graph processing speed, or "
                "preprocessing speed must not be presented as synthetic-data "
                "generation scalability."
            ),
            (
                "Any percentage reduction in runtime or GPU memory requires "
                "matched raw measurements under the same hardware, batch size, "
                "software environment, and workload."
            ),
            "",
            "SAFETY CONFIRMATION",
            "-" * 100,
            "New generative model trained: NO",
            "New synthetic rows generated: NO",
            "New runtime benchmark created: NO",
            "New GPU-memory benchmark created: NO",
            "Historical files modified: NO",
            "",
            "PRIMARY OUTPUTS",
            "-" * 100,
            "scalability_source_inventory.csv",
            "scalability_numeric_evidence.csv",
            "workload_size_evidence.csv",
            "runtime_evidence.csv",
            "memory_evidence.csv",
            "hardware_evidence.csv",
            "generative_scalability_linkage.csv",
            "efficiency_claim_reconstruction.csv",
            "scalability_verification_matrix.csv",
            "scalability_verdict.csv",
            "scalability_provenance.csv",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "scalability_audit_summary.txt"
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
        "06 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nVerified generative architecture files: "
        f"{len(verified_generator_files)}"
    )

    print(
        f"Runtime evidence rows: "
        f"{len(runtime_rows)}"
    )

    print(
        f"Memory evidence rows: "
        f"{len(memory_rows)}"
    )

    print(
        f"Workload evidence rows: "
        f"{len(workload_rows)}"
    )

    print(
        f"Hardware evidence rows: "
        f"{len(hardware_rows)}"
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
        "scalability_audit_summary.txt",
        "scalability_verdict.csv",
        "scalability_verification_matrix.csv",
        "efficiency_claim_reconstruction.csv",
        "runtime_evidence.csv",
        "memory_evidence.csv",
        "workload_size_evidence.csv",
        "hardware_evidence.csv",
        "generative_scalability_linkage.csv",
        "scalability_provenance.csv",
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
            "06 FAILED SAFELY"
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