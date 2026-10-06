from __future__ import annotations

import json
import re
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

PAPER_ROOT = PROJECT_ROOT.parent

MANUSCRIPT_PATH = (
    PAPER_ROOT / "HFAGM.docx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_master_evidence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GENERATED_AT = datetime.now().isoformat(
    timespec="seconds"
)

EXPECTED_PARTICIPANTS = 193

# This audit consolidates prior evidence only.
# It does NOT train models, generate synthetic data, run OCR, or modify
# historical outputs/manuscript files.

# =============================================================================
# 2. KNOWN PRIOR OUTPUT LOCATIONS
# =============================================================================

# Stage 02D/02E/02F
REPEATED_METRICS_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
    / "repeated_leakage_safe_evaluation"
    / "repeated_seed_metrics.csv",
]

FEATURE_CHRONOLOGY_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
    / "feature_chronology_proxy_audit"
    / "feature_chronology_proxy_audit_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_primary_metrics"
    / "feature_chronology_proxy_audit_v2"
    / "feature_chronology_proxy_audit_summary.txt",
]

# Stage 03
FAIRNESS_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_fairness"
    / "recomputed_fairness"
    / "fairness_summary_reference.csv",

    PROJECT_ROOT
    / "outputs"
    / "revision_fairness"
    / "recomputed_fairness"
    / "fairness_per_seed_reference.csv",
]

# Stage 04
FRECHET_SUMMARY_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_structured_frechet"
    / "structured_frechet_audit_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_fid_audit"
    / "structured_frechet_audit_summary.txt",
]

# Stage 04B
GENERATOR_AUDIT_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_generator_audit"
    / "actual_generative_architecture_verification_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_generator_audit_v2"
    / "actual_generative_architecture_verification_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_structured_generation"
    / "actual_generative_architecture_verification_summary.txt",
]

# Stage 05
UTILITY_SUMMARY_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_downstream_utility"
    / "common_real_test_utility_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_utility"
    / "common_real_test_utility_summary.txt",
]

# Stage 06
SCALABILITY_SUMMARY_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_scalability"
    / "scalability_computational_efficiency_audit_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_scalability_audit"
    / "scalability_computational_efficiency_audit_summary.txt",
]

# Stage 07
ABLATION_SUMMARY_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_ablation"
    / "ablation_audit_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_ablation_audit"
    / "ablation_audit_summary.txt",
]

# Stage 08
STATISTICS_SUMMARY_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "revision_statistics_v2"
    / "statistical_claim_audit_v2_summary.txt",

    PROJECT_ROOT
    / "outputs"
    / "revision_statistics_v2"
    / "statistical_claim_audit_summary.txt",
]

# Stage 09
STAGE09_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_multimodal_figure_audit_v2"
)

STAGE09_SUMMARY = (
    STAGE09_DIR
    / "multimodal_figure_results_audit_v2_summary.txt"
)

STAGE09_FINDINGS = (
    STAGE09_DIR
    / "results_consistency_findings_v2.csv"
)

STAGE09_VERDICT = (
    STAGE09_DIR
    / "multimodal_figure_verdict_v2.csv"
)

STAGE09_FIGURE_INVENTORY = (
    STAGE09_DIR
    / "figure_inventory_v2.csv"
)

STAGE09_DUPLICATE_NUMBERS = (
    STAGE09_DIR
    / "duplicate_figure_numbers_v2.csv"
)

STAGE09_METRIC_CLAIMS = (
    STAGE09_DIR
    / "manuscript_metric_claims_v2.csv"
)

STAGE09_MULTIMODAL_CLAIMS = (
    STAGE09_DIR
    / "manuscript_multimodal_claims_v2.csv"
)


# =============================================================================
# 3. REVIEWER COMMENT MAP
# =============================================================================

REVIEWER_MAP = {
    "C6": "HGF mathematical precision and reproducibility",
    "C7": "Correct GAN/VAE/diffusion objectives",
    "C8": "Fairness controller dimensional consistency",
    "C9": "SPD/EOD differentiability",
    "C10": "Wall-clock non-differentiability",
    "C11": "Latent fusion dimensions/alignment/normalization",
    "C12": "Unsupported Pareto-optimal claim",
    "C13": "Scope inconsistency across healthcare/finance/multimodal",
    "C14": "Multimodal reproducibility",
    "C15": "Duplicate/mislabeled multimodal figures",
    "C16": "Fairness recalculation and EOD interpretation",
    "C17": "DI interpretation",
    "C18": "Sensitive attribute/reference/favorable outcome/subgroup n",
    "C19": "FID definition/feature space",
    "C20": "PR-GM/MSS definitions",
    "C21": "Common untouched real test downstream utility",
    "C22": "Accuracy/confusion-matrix consistency",
    "C23": "Synthetic N interpretation",
    "C24": "Conflicting fidelity/FID values",
    "C25": "Hardware/efficiency inconsistency",
    "C26": "Implementation details/pseudocode",
    "C27": "Baseline tuning fairness",
    "C28": "Requested ablations",
    "C29": "Statistical significance/uncertainty",
    "C30": "Unsupported privacy claims",
    "C31": "Novelty scope",
    "C32": "Reference DOI/bibliographic audit",
    "C33": "Major restructuring and consolidated table",
}


# =============================================================================
# 4. EVIDENCE RECORD
# =============================================================================

@dataclass
class EvidenceRecord:
    evidence_id: str
    reviewer_comments: str
    topic: str
    manuscript_claim_or_issue: str
    reproducibility_status: str
    evidence_status: str
    validated_evidence: str
    invalid_or_unsupported_evidence: str
    final_value_or_interpretation: str
    manuscript_action: str
    required_wording_or_constraint: str
    source_stage: str
    source_files: str
    severity: str
    final_disposition: str


# =============================================================================
# 5. GENERAL HELPERS
# =============================================================================

def normalize_ws(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def first_existing(
    candidates: Sequence[Path],
) -> Optional[Path]:
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def read_text(
    path: Path,
) -> str:
    if not path.exists():
        return ""

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):
        try:
            return path.read_text(
                encoding=encoding,
                errors="ignore",
            )
        except Exception:
            continue

    return ""


def read_csv_safe(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
        )
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def fmt_float(
    value: Any,
    digits: int = 6,
) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def safe_json(
    obj: Any,
) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        default=str,
    )


def relative(
    path: Optional[Path],
) -> str:
    if path is None:
        return ""

    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )
    except Exception:
        return str(path)


def add_record(
    rows: List[EvidenceRecord],
    *,
    evidence_id: str,
    reviewer_comments: str,
    topic: str,
    manuscript_claim_or_issue: str,
    reproducibility_status: str,
    evidence_status: str,
    validated_evidence: str,
    invalid_or_unsupported_evidence: str,
    final_value_or_interpretation: str,
    manuscript_action: str,
    required_wording_or_constraint: str,
    source_stage: str,
    source_files: str,
    severity: str,
    final_disposition: str,
) -> None:
    rows.append(
        EvidenceRecord(
            evidence_id=evidence_id,
            reviewer_comments=reviewer_comments,
            topic=topic,
            manuscript_claim_or_issue=manuscript_claim_or_issue,
            reproducibility_status=reproducibility_status,
            evidence_status=evidence_status,
            validated_evidence=validated_evidence,
            invalid_or_unsupported_evidence=invalid_or_unsupported_evidence,
            final_value_or_interpretation=final_value_or_interpretation,
            manuscript_action=manuscript_action,
            required_wording_or_constraint=required_wording_or_constraint,
            source_stage=source_stage,
            source_files=source_files,
            severity=severity,
            final_disposition=final_disposition,
        )
    )


# =============================================================================
# 6. STAGE 02E REPEATED LEAKAGE-SAFE METRICS
# =============================================================================

def summarize_repeated_metrics(
    path: Optional[Path],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": path,
        "available": False,
        "rows": 0,
        "conditions": {},
    }

    if path is None:
        return result

    df = read_csv_safe(path)

    if df.empty:
        return result

    result["available"] = True
    result["rows"] = len(df)

    columns_lower = {
        c.lower(): c
        for c in df.columns
    }

    condition_col = None

    for candidate in (
        "condition",
        "evaluation_condition",
        "variant",
        "setting",
    ):
        if candidate in columns_lower:
            condition_col = columns_lower[
                candidate
            ]
            break

    metric_candidates = [
        "accuracy",
        "precision",
        "sensitivity",
        "recall",
        "specificity",
        "f1",
        "auc",
        "roc_auc",
    ]

    metric_cols = []

    for metric in metric_candidates:
        if metric in columns_lower:
            metric_cols.append(
                columns_lower[metric]
            )

    if condition_col is None:
        groups = [("all", df)]
    else:
        groups = list(
            df.groupby(
                condition_col,
                dropna=False,
            )
        )

    for condition, g in groups:
        cond_summary = {}

        for col in metric_cols:
            values = pd.to_numeric(
                g[col],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            cond_summary[col] = {
                "n": int(
                    values.shape[0]
                ),
                "mean": float(
                    values.mean()
                ),
                "sd": float(
                    values.std(
                        ddof=1
                    )
                ) if len(values) > 1 else 0.0,
                "median": float(
                    values.median()
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                ),
            }

        result["conditions"][
            str(condition)
        ] = cond_summary

    return result


# =============================================================================
# 7. STAGE 03 FAIRNESS
# =============================================================================

def summarize_fairness(
    candidates: Sequence[Path],
) -> Dict[str, Any]:
    selected = first_existing(
        candidates
    )

    result: Dict[str, Any] = {
        "path": selected,
        "available": False,
        "rows": 0,
        "summary": {},
    }

    if selected is None:
        return result

    df = read_csv_safe(selected)

    if df.empty:
        return result

    result["available"] = True
    result["rows"] = len(df)

    lower = {
        c.lower(): c
        for c in df.columns
    }

    metric_cols = {}

    for key in (
        "spd",
        "eod",
        "di",
    ):
        if key in lower:
            metric_cols[key] = lower[key]

    # If this is the per-seed file, aggregate.
    possible_group_cols = []

    for candidate in (
        "condition",
        "sensitive_attribute",
        "comparison",
        "attribute",
    ):
        if candidate in lower:
            possible_group_cols.append(
                lower[candidate]
            )

    if metric_cols:
        if possible_group_cols:
            grouped = df.groupby(
                possible_group_cols,
                dropna=False,
            )

            for group_key, g in grouped:
                key = (
                    group_key
                    if isinstance(
                        group_key,
                        tuple,
                    )
                    else (group_key,)
                )

                group_name = " | ".join(
                    str(x)
                    for x in key
                )

                result["summary"][
                    group_name
                ] = {}

                for metric, col in (
                    metric_cols.items()
                ):
                    values = pd.to_numeric(
                        g[col],
                        errors="coerce",
                    ).dropna()

                    if values.empty:
                        continue

                    result["summary"][
                        group_name
                    ][metric] = {
                        "mean": float(
                            values.mean()
                        ),
                        "sd": float(
                            values.std(
                                ddof=1
                            )
                        ) if len(values) > 1 else 0.0,
                        "min": float(
                            values.min()
                        ),
                        "max": float(
                            values.max()
                        ),
                        "n": int(
                            len(values)
                        ),
                    }

    return result


# =============================================================================
# 8. TEXTUAL PRIOR AUDIT DETECTION
# =============================================================================

def textual_status(
    candidates: Sequence[Path],
    required_terms: Sequence[str],
) -> Dict[str, Any]:
    path = first_existing(
        candidates
    )

    result = {
        "path": path,
        "available": False,
        "matched_terms": [],
        "text_excerpt": "",
    }

    if path is None:
        return result

    text = read_text(path)

    if not text:
        return result

    result["available"] = True

    upper = text.upper()

    matched = [
        term
        for term in required_terms
        if term.upper() in upper
    ]

    result["matched_terms"] = matched

    if matched:
        first = matched[0]
        idx = upper.find(
            first.upper()
        )

        start = max(
            0,
            idx - 500,
        )

        end = min(
            len(text),
            idx + 1200,
        )

        result["text_excerpt"] = normalize_ws(
            text[start:end]
        )

    return result


# =============================================================================
# 9. STAGE 09 LOAD
# =============================================================================

def load_stage09() -> Dict[str, Any]:
    return {
        "summary_text": read_text(
            STAGE09_SUMMARY
        ),
        "findings": read_csv_safe(
            STAGE09_FINDINGS
        ),
        "verdict": read_csv_safe(
            STAGE09_VERDICT
        ),
        "figures": read_csv_safe(
            STAGE09_FIGURE_INVENTORY
        ),
        "duplicates": read_csv_safe(
            STAGE09_DUPLICATE_NUMBERS
        ),
        "metrics": read_csv_safe(
            STAGE09_METRIC_CLAIMS
        ),
        "multimodal": read_csv_safe(
            STAGE09_MULTIMODAL_CLAIMS
        ),
    }


# =============================================================================
# 10. BUILD MASTER EVIDENCE TABLE
# =============================================================================

def build_master_records() -> Tuple[
    List[EvidenceRecord],
    Dict[str, Any],
]:
    records: List[
        EvidenceRecord
    ] = []

    diagnostics: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Stage 02E
    # -------------------------------------------------------------------------
    repeated_path = first_existing(
        REPEATED_METRICS_CANDIDATES
    )

    repeated = summarize_repeated_metrics(
        repeated_path
    )

    diagnostics[
        "stage02e_repeated_metrics"
    ] = repeated

    if repeated["available"]:
        cond_text_parts = []

        for condition, metrics in (
            repeated["conditions"].items()
        ):
            metric_parts = []

            for metric, stats in (
                metrics.items()
            ):
                metric_parts.append(
                    f"{metric}: "
                    f"{stats['mean']:.6f}±"
                    f"{stats['sd']:.6f}"
                )

            cond_text_parts.append(
                f"{condition} -> "
                + "; ".join(
                    metric_parts
                )
            )

        validated = " | ".join(
            cond_text_parts
        )

    else:
        validated = (
            "Repeated leakage-safe metric file "
            "not resolved automatically."
        )

    add_record(
        records,
        evidence_id="E02E-01",
        reviewer_comments="C21,C22,C27,C29",
        topic="Repeated leakage-safe classifier performance",
        manuscript_claim_or_issue=(
            "Classifier performance must be based on leakage-safe "
            "evaluation rather than the historically contaminated split."
        ),
        reproducibility_status=(
            "REPRODUCIBLE"
            if repeated["available"]
            else "SOURCE_NOT_AUTO_RESOLVED"
        ),
        evidence_status=(
            "VALIDATED"
            if repeated["available"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=validated,
        invalid_or_unsupported_evidence=(
            "Historical perfect metrics obtained from the earlier "
            "leakage-contaminated workflow must not be used as primary "
            "headline evidence."
        ),
        final_value_or_interpretation=(
            "Use repeated leakage-safe holdouts descriptively. "
            "Ten seeds reuse the same participant cohort and therefore "
            "represent stability analyses rather than independent datasets."
        ),
        manuscript_action=(
            "RETAIN corrected repeated-holdout classifier results; "
            "REMOVE/REPLACE leakage-contaminated primary claims."
        ),
        required_wording_or_constraint=(
            "Report mean±SD/range across repeated leakage-safe partitions; "
            "do not describe the ten seeds as independent experimental "
            "replicates."
        ),
        source_stage="02E",
        source_files=relative(
            repeated_path
        ),
        severity="CRITICAL",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # Stage 02F chronology/proxy
    # -------------------------------------------------------------------------
    chronology = textual_status(
        FEATURE_CHRONOLOGY_CANDIDATES,
        [
            (
                "NO_DIRECT_TEMPORAL_LEAKAGE_EVIDENCE_"
                "BUT_HIGHLY_PREDICTIVE_FEATURE_TIMING_REMAINS_UNDOCUMENTED"
            ),
        ],
    )

    diagnostics[
        "stage02f_feature_chronology"
    ] = chronology

    add_record(
        records,
        evidence_id="E02F-01",
        reviewer_comments="C21,C27",
        topic="Feature chronology and proxy leakage",
        manuscript_claim_or_issue=(
            "The chronology of highly predictive clinical features "
            "is incompletely documented."
        ),
        reproducibility_status=(
            "AUDITED"
            if chronology["available"]
            else "SOURCE_NOT_AUTO_RESOLVED"
        ),
        evidence_status=(
            "LIMITATION"
        ),
        validated_evidence=(
            "No direct temporal/outcome proxy leakage was established "
            "in the recovered audit."
        ),
        invalid_or_unsupported_evidence=(
            "Absence of direct proxy evidence does not prove that all "
            "feature timing was prospectively valid."
        ),
        final_value_or_interpretation=(
            "Feature timing remains a documented limitation, particularly "
            "because some predictors are highly discriminative."
        ),
        manuscript_action=(
            "RETAIN as a limitation; avoid claims of fully prospective "
            "or deployment-ready validation."
        ),
        required_wording_or_constraint=(
            "State that no direct temporal leakage was identified, "
            "but predictor timing relative to outcome determination "
            "could not be fully verified."
        ),
        source_stage="02F_v2",
        source_files=relative(
            chronology["path"]
        ),
        severity="MAJOR",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # Stage 03 fairness
    # -------------------------------------------------------------------------
    fairness = summarize_fairness(
        FAIRNESS_CANDIDATES
    )

    diagnostics[
        "stage03_fairness"
    ] = fairness

    add_record(
        records,
        evidence_id="E03-01",
        reviewer_comments="C16,C17,C18,C29",
        topic="Recomputed fairness metrics",
        manuscript_claim_or_issue=(
            "Historical SPD/EOD/DI interpretations are inconsistent "
            "with standard parity interpretation."
        ),
        reproducibility_status=(
            "REPRODUCIBLE"
            if fairness["available"]
            else "SOURCE_NOT_AUTO_RESOLVED"
        ),
        evidence_status=(
            "VALIDATED_CLASSIFIER_FAIRNESS"
            if fairness["available"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            safe_json(
                fairness["summary"]
            )
            if fairness["available"]
            else
            "Fairness output not resolved automatically."
        ),
        invalid_or_unsupported_evidence=(
            "EOD=1 cannot be described as perfect fairness under the "
            "standard signed-difference interpretation. Historical DI "
            "values 1 -> 0.643 -> 0.209 do not demonstrate improvement "
            "toward parity."
        ),
        final_value_or_interpretation=(
            "SPD≈0, EOD≈0, and DI≈1 indicate parity. The corrected "
            "fairness audit applies to the real-outcome classifier, "
            "not to an unverified synthetic generator."
        ),
        manuscript_action=(
            "REPLACE historical fairness interpretation with corrected "
            "classifier-fairness reporting; REMOVE generator-fairness "
            "claims unsupported by implementation evidence."
        ),
        required_wording_or_constraint=(
            "Explicitly state sensitive attribute, reference group, "
            "favorable outcome, subgroup counts, and that the results "
            "are classifier fairness metrics."
        ),
        source_stage="03",
        source_files=relative(
            fairness["path"]
        ),
        severity="CRITICAL",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # Stage 04 SFD/FID
    # -------------------------------------------------------------------------
    frechet = textual_status(
        FRECHET_SUMMARY_CANDIDATES,
        [
            "NO_REPRODUCIBLE_FID_OR_STRUCTURED_FRECHET_RESULT_AVAILABLE",
        ],
    )

    diagnostics[
        "stage04_structured_frechet"
    ] = frechet

    add_record(
        records,
        evidence_id="E04-01",
        reviewer_comments="C19,C24",
        topic="FID / Structured Fréchet Distance",
        manuscript_claim_or_issue=(
            "The manuscript reports FID-like fidelity values without "
            "recoverable generator or feature-space provenance."
        ),
        reproducibility_status=(
            "NOT_REPRODUCIBLE"
        ),
        evidence_status="UNSUPPORTED",
        validated_evidence=(
            "The audit established that the recovered data consist of "
            "structured predictors and that no provenance-valid synthetic "
            "table was available for recomputation."
        ),
        invalid_or_unsupported_evidence=(
            "Absolute historical values 1.5, 1.6, and 3.2 are unsupported. "
            "Conventional image FID is not established for the structured "
            "clinical feature space."
        ),
        final_value_or_interpretation=(
            "No reproducible FID/SFD result is available from recovered "
            "artifacts."
        ),
        manuscript_action=(
            "REMOVE absolute FID claims and any comparison that relies "
            "on those values."
        ),
        required_wording_or_constraint=(
            "Do not call a structured-feature Fréchet distance conventional "
            "FID unless a valid feature space and synthetic source are "
            "explicitly established."
        ),
        source_stage="04",
        source_files=relative(
            frechet["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE",
    )

    # -------------------------------------------------------------------------
    # Stage 04B generator audit
    # -------------------------------------------------------------------------
    generator = textual_status(
        GENERATOR_AUDIT_CANDIDATES,
        [
            "CLASSIFIER_OR_ENCODER_ONLY_NO_STRUCTURED_GENERATOR",
            "DO_NOT_REGENERATE_REMOVE_UNSUPPORTED_GENERATIVE_FID_CLAIMS",
        ],
    )

    diagnostics[
        "stage04b_generator"
    ] = generator

    add_record(
        records,
        evidence_id="E04B-01",
        reviewer_comments="C6,C7,C8,C9,C10,C11,C19,C20,C26,C28,C30,C31",
        topic="Recovered HFAGM generative implementation",
        manuscript_claim_or_issue=(
            "The manuscript describes a GAN/VAE/diffusion hybrid "
            "generator, adaptive fairness controller, and synthetic-data "
            "generation pipeline."
        ),
        reproducibility_status="NOT_REPRODUCIBLE",
        evidence_status="IMPLEMENTATION_MISSING",
        validated_evidence=(
            "Recovered implementation supports classifier/encoder/graph "
            "processing. No verified GAN, VAE, diffusion generator, "
            "hybrid generative integration, latent/noise sampling, "
            "51-feature generator output, or generator checkpoint "
            "was established."
        ),
        invalid_or_unsupported_evidence=(
            "Textual mathematical descriptions, diagrams, BCE losses, "
            "class names, and empty generator directories do not establish "
            "an executable synthetic generator."
        ),
        final_value_or_interpretation=(
            "The recovered codebase cannot substantiate the manuscript's "
            "historical structured synthetic generator claims."
        ),
        manuscript_action=(
            "MAJOR REFRAME: remove unsupported implementation/results "
            "claims for GAN/VAE/diffusion hybrid generation. Do not "
            "retroactively create a new generator and present it as "
            "the original experiment."
        ),
        required_wording_or_constraint=(
            "Any future generator implementation must be explicitly labeled "
            "as a new/corrected experiment, not a reconstruction of the "
            "historical study."
        ),
        source_stage="04B_v2",
        source_files=relative(
            generator["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE_OR_MAJOR_REFRAME",
    )

    # -------------------------------------------------------------------------
    # Stage 05 utility
    # -------------------------------------------------------------------------
    utility = textual_status(
        UTILITY_SUMMARY_CANDIDATES,
        [
            "SYNTHETIC_UTILITY_NOT_RECOMPUTABLE_FROM_RECOVERED_ARTIFACTS",
        ],
    )

    diagnostics[
        "stage05_utility"
    ] = utility

    add_record(
        records,
        evidence_id="E05-01",
        reviewer_comments="C21,C22,C27",
        topic="Common untouched real-test downstream utility",
        manuscript_claim_or_issue=(
            "Synthetic-vs-real downstream utility comparisons require "
            "the same untouched real test set."
        ),
        reproducibility_status="PARTIALLY_REPRODUCIBLE",
        evidence_status="REAL_REFERENCE_ONLY",
        validated_evidence=(
            "A leakage-safe real-data reference using the same locked "
            "test protocol was recoverable."
        ),
        invalid_or_unsupported_evidence=(
            "No provenance-valid labeled 51-feature synthetic dataset "
            "was recovered, so synthetic-utility conditions cannot be "
            "recomputed."
        ),
        final_value_or_interpretation=(
            "Retain the real benchmark only as a controlled reference; "
            "do not claim validated synthetic-vs-real downstream utility."
        ),
        manuscript_action=(
            "REMOVE historical synthetic utility superiority claims. "
            "RETAIN only explicitly reproducible real-data evaluation."
        ),
        required_wording_or_constraint=(
            "Do not convert the single locked-test perfect score into "
            "a headline superiority claim; use repeated leakage-safe "
            "stability evidence for broader reporting."
        ),
        source_stage="05",
        source_files=relative(
            utility["path"]
        ),
        severity="CRITICAL",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # Stage 06 scalability
    # -------------------------------------------------------------------------
    scalability = textual_status(
        SCALABILITY_SUMMARY_CANDIDATES,
        [
            "GENERATIVE_SCALABILITY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS",
            "REMOVE_UNSUPPORTED_RUNTIME_MEMORY_AND_SYNTHETIC_SCALE_CLAIMS",
        ],
    )

    diagnostics[
        "stage06_scalability"
    ] = scalability

    add_record(
        records,
        evidence_id="E06-01",
        reviewer_comments="C10,C23,C25",
        topic="Generative scalability and computational efficiency",
        manuscript_claim_or_issue=(
            "Claims such as N=500→100000, ~25% faster, "
            "20–30% lower GPU memory, and conflicting +40% memory "
            "require matched generator-linked measurements."
        ),
        reproducibility_status="NOT_REPRODUCIBLE",
        evidence_status="UNSUPPORTED",
        validated_evidence=(
            "Recovered artifacts did not establish a verified generator-linked "
            "runtime or GPU-memory benchmark."
        ),
        invalid_or_unsupported_evidence=(
            "Synthetic scale and runtime/memory percentages are unsupported. "
            "Classifier/preprocessing timing cannot substitute for generator "
            "scalability."
        ),
        final_value_or_interpretation=(
            "Large synthetic N, if mentioned at all, is computational "
            "workload and not independent clinical information."
        ),
        manuscript_action=(
            "REMOVE unsupported scalability/runtime/memory claims."
        ),
        required_wording_or_constraint=(
            "If computational efficiency is discussed, frame it as an "
            "unverified design objective or future evaluation requirement."
        ),
        source_stage="06",
        source_files=relative(
            scalability["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE",
    )

    # -------------------------------------------------------------------------
    # Stage 07 ablation
    # -------------------------------------------------------------------------
    ablation = textual_status(
        ABLATION_SUMMARY_CANDIDATES,
        [
            "ABLATION_STUDY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS",
            "REMOVE_UNSUPPORTED_ABLATION_CLAIMS",
        ],
    )

    diagnostics[
        "stage07_ablation"
    ] = ablation

    add_record(
        records,
        evidence_id="E07-01",
        reviewer_comments="C11,C28",
        topic="Generator/fusion/fairness/scalability ablations",
        manuscript_claim_or_issue=(
            "Reviewer requested full HGF vs GAN/VAE/diffusion, "
            "adaptive vs static fusion, fairness on/off, and scalability "
            "on/off ablations."
        ),
        reproducibility_status="NOT_REPRODUCIBLE",
        evidence_status="MISSING_EXECUTED_VARIANTS",
        validated_evidence=(
            "Recovered project does not contain the required executed "
            "historical ablation variants and result provenance."
        ),
        invalid_or_unsupported_evidence=(
            "Generic implementation keywords or classifier hyperparameters "
            "cannot be relabeled as generator ablations."
        ),
        final_value_or_interpretation=(
            "The requested ablation study cannot be reconstructed from "
            "existing artifacts."
        ),
        manuscript_action=(
            "REMOVE unsupported historical ablation claims and figures. "
            "If new ablations are later run, label them explicitly as "
            "new corrected experiments."
        ),
        required_wording_or_constraint=(
            "Do not imply that missing ablation variants were evaluated."
        ),
        source_stage="07",
        source_files=relative(
            ablation["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE",
    )

    # -------------------------------------------------------------------------
    # Stage 08 statistics
    # -------------------------------------------------------------------------
    statistics = textual_status(
        STATISTICS_SUMMARY_CANDIDATES,
        [
            (
                "DESCRIPTIVE_STABILITY_EVIDENCE_VALID_"
                "FORMAL_SIGNIFICANCE_NOT_SUPPORTED"
            ),
            "FORMAL_SIGNIFICANCE_NOT_SUPPORTED",
        ],
    )

    diagnostics[
        "stage08_statistics"
    ] = statistics

    add_record(
        records,
        evidence_id="E08-01",
        reviewer_comments="C29",
        topic="Statistical significance and uncertainty",
        manuscript_claim_or_issue=(
            "Claims using 'significantly', statistical superiority, "
            "or inferential p-values require independent replicates "
            "or an appropriate inferential design."
        ),
        reproducibility_status="AUDITED",
        evidence_status="DESCRIPTIVE_ONLY",
        validated_evidence=(
            "Ten leakage-safe repeated holdouts can support descriptive "
            "mean, SD, median, and range."
        ),
        invalid_or_unsupported_evidence=(
            "The same 193-participant cohort is reused across holdouts; "
            "formal p-values and independent-replicate significance claims "
            "are not supported."
        ),
        final_value_or_interpretation=(
            "Treat the repeated holdouts as stability analyses."
        ),
        manuscript_action=(
            "REMOVE 'significantly improved/outperformed' unless supported "
            "by a future valid inferential design."
        ),
        required_wording_or_constraint=(
            "Across ten leakage-safe stratified holdouts, performance was "
            "evaluated descriptively using mean, standard deviation, "
            "median, and range; no inferential p-values are reported."
        ),
        source_stage="08_v2",
        source_files=relative(
            statistics["path"]
        ),
        severity="MAJOR",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # Stage 09 multimodal / figures
    # -------------------------------------------------------------------------
    stage09 = load_stage09()

    diagnostics[
        "stage09"
    ] = {
        "summary_available": bool(
            stage09["summary_text"]
        ),
        "findings_rows": len(
            stage09["findings"]
        ),
        "verdict_rows": len(
            stage09["verdict"]
        ),
        "figure_rows": len(
            stage09["figures"]
        ),
        "duplicate_number_rows": len(
            stage09["duplicates"]
        ),
        "metric_rows": len(
            stage09["metrics"]
        ),
        "multimodal_rows": len(
            stage09["multimodal"]
        ),
    }

    add_record(
        records,
        evidence_id="E09-01",
        reviewer_comments="C13,C14",
        topic="Multimodal clinical scope",
        manuscript_claim_or_issue=(
            "The manuscript claims multimodal clinical evaluation "
            "and cross-domain/multimodal generalization."
        ),
        reproducibility_status="NOT_REPRODUCIBLE",
        evidence_status="UNSUPPORTED_SCOPE",
        validated_evidence=(
            "Stage 09 found 27 multimodal manuscript claims, "
            "0 strict executable clinical multimodal+fusion candidates, "
            "0 clinical image artifacts, and a large unrelated "
            "sign-language image repository."
        ),
        invalid_or_unsupported_evidence=(
            "Presence of ArSL/sign-language images elsewhere in the project "
            "does not establish multimodal clinical HFAGM evaluation."
        ),
        final_value_or_interpretation=(
            "The reproducible clinical study should be described as "
            "structured/tabular clinical data."
        ),
        manuscript_action=(
            "REMOVE or reframe multimodal clinical claims."
        ),
        required_wording_or_constraint=(
            "Do not state that fairness, utility, or generator behavior "
            "generalizes across modalities unless a recoverable multimodal "
            "clinical implementation and experiment are provided."
        ),
        source_stage="09_v2",
        source_files=relative(
            STAGE09_SUMMARY
        ),
        severity="CRITICAL",
        final_disposition="REMOVE_OR_MAJOR_REFRAME",
    )

    duplicate_count = len(
        stage09["duplicates"]
    )

    add_record(
        records,
        evidence_id="E09-02",
        reviewer_comments="C15",
        topic="Figure numbering and figure provenance",
        manuscript_claim_or_issue=(
            "Figure numbering/captions contain inconsistencies."
        ),
        reproducibility_status="AUDITED",
        evidence_status="CORRECTION_REQUIRED",
        validated_evidence=(
            f"Stage 09 detected {duplicate_count} duplicate "
            "figure-number groups and no exact/perceptual duplicate "
            "image pairs."
        ),
        invalid_or_unsupported_evidence=(
            "A duplicated figure number is distinct from an exact "
            "duplicate image. Unsupported generative/multimodal figures "
            "cannot be fixed by renumbering alone."
        ),
        final_value_or_interpretation=(
            "Figure numbering must be corrected after unsupported "
            "figures are removed/reframed."
        ),
        manuscript_action=(
            "REMOVE unsupported figures first, then renumber remaining "
            "figures sequentially and align every caption/reference."
        ),
        required_wording_or_constraint=(
            "Do not preserve a figure solely for continuity if its "
            "experimental claim is unsupported."
        ),
        source_stage="09_v2",
        source_files="; ".join(
            [
                relative(
                    STAGE09_FIGURE_INVENTORY
                ),
                relative(
                    STAGE09_DUPLICATE_NUMBERS
                ),
            ]
        ),
        severity="MAJOR",
        final_disposition="CORRECT_AFTER_RESTRUCTURE",
    )

    add_record(
        records,
        evidence_id="E09-03",
        reviewer_comments="C22",
        topic="Accuracy and confusion-matrix consistency",
        manuscript_claim_or_issue=(
            "Confusion-matrix-derived accuracy must match the same "
            "evaluation condition."
        ),
        reproducibility_status="PARTIAL",
        evidence_status="MANUAL_SOURCE_LINK_REQUIRED",
        validated_evidence=(
            "Context-aware Stage 09 parsing found no directly assigned "
            "accuracy values and kept AUC 0.95/0.99 separate from accuracy."
        ),
        invalid_or_unsupported_evidence=(
            "No complete textual TP/TN/FP/FN set was recovered from the "
            "manuscript; image-only confusion matrices cannot be numerically "
            "validated without source result counts."
        ),
        final_value_or_interpretation=(
            "For each retained confusion matrix, recompute accuracy as "
            "(TP+TN)/(TP+TN+FP+FN) from the source evaluation condition."
        ),
        manuscript_action=(
            "RETAIN only confusion matrices with source-linked counts; "
            "remove or replace unsupported matrices."
        ),
        required_wording_or_constraint=(
            "Never substitute AUC for accuracy."
        ),
        source_stage="09_v2",
        source_files=relative(
            STAGE09_SUMMARY
        ),
        severity="MAJOR",
        final_disposition="VERIFY_BEFORE_RETAINING",
    )

    add_record(
        records,
        evidence_id="E09-04",
        reviewer_comments="C23",
        topic="Synthetic sample size",
        manuscript_claim_or_issue=(
            "Synthetic sample count must not be interpreted as "
            "additional independent patients."
        ),
        reproducibility_status="AUDITED",
        evidence_status="INTERPRETATION_CONSTRAINT",
        validated_evidence=(
            "Stage 09 found no directly parsed large synthetic-N statement "
            "in the selected manuscript version."
        ),
        invalid_or_unsupported_evidence=(
            "Historical N=500→100000 generator-scale claims remain "
            "unsupported by Stage 06."
        ),
        final_value_or_interpretation=(
            f"Independent clinical information remains limited to the "
            f"original cohort of {EXPECTED_PARTICIPANTS} participants."
        ),
        manuscript_action=(
            "REMOVE unsupported historical synthetic-scale statements."
        ),
        required_wording_or_constraint=(
            "If a synthetic row count is ever reported, call it generated "
            "computational workload, not increased independent sample size."
        ),
        source_stage="06,09_v2",
        source_files=relative(
            STAGE09_SUMMARY
        ),
        severity="MAJOR",
        final_disposition="REWRITE_OR_REMOVE",
    )

    # -------------------------------------------------------------------------
    # Structural/claim-level consequences not requiring another experiment
    # -------------------------------------------------------------------------
    add_record(
        records,
        evidence_id="ESTR-01",
        reviewer_comments="C6,C7,C8,C9,C10,C11,C12,C26,C31",
        topic="Mathematical HGF/HFAGM framework",
        manuscript_claim_or_issue=(
            "The current equations and narrative imply an executable "
            "hybrid generative optimizer that is not recovered in code."
        ),
        reproducibility_status="NOT_SUPPORTED_AS_IMPLEMENTED",
        evidence_status="THEORETICAL_ONLY_AT_BEST",
        validated_evidence=(
            "A conceptual mathematical framework may be discussed only "
            "if clearly separated from implemented experiments."
        ),
        invalid_or_unsupported_evidence=(
            "Equations cannot be presented as implemented training "
            "mechanics when generator/fairness/scalability modules are absent."
        ),
        final_value_or_interpretation=(
            "Separate conceptual design from empirically verified components."
        ),
        manuscript_action=(
            "REWRITE Methods to distinguish conceptual framework from "
            "recovered executable classifier/encoder pipeline; remove "
            "unsupported implementation-specific equations or claims."
        ),
        required_wording_or_constraint=(
            "Do not describe non-differentiable SPD/EOD or wall-clock "
            "terms as direct gradient objectives unless an actual "
            "differentiable surrogate/proxy is verified."
        ),
        source_stage="04B_v2,06,07",
        source_files="; ".join(
            filter(
                None,
                [
                    relative(
                        generator["path"]
                    ),
                    relative(
                        scalability["path"]
                    ),
                    relative(
                        ablation["path"]
                    ),
                ],
            )
        ),
        severity="CRITICAL",
        final_disposition="MAJOR_REWRITE",
    )

    add_record(
        records,
        evidence_id="ESTR-02",
        reviewer_comments="C12",
        topic="Pareto-optimal wording",
        manuscript_claim_or_issue=(
            "Pareto-optimal/optimal-frontier claims require explicit "
            "frontier/convergence evidence."
        ),
        reproducibility_status="NOT_ESTABLISHED",
        evidence_status="UNSUPPORTED",
        validated_evidence=(
            "No verified Pareto-front/convergence artifact was recovered."
        ),
        invalid_or_unsupported_evidence=(
            "Multi-objective language alone does not establish Pareto optimality."
        ),
        final_value_or_interpretation=(
            "The work may discuss trade-offs, not proven Pareto optimality."
        ),
        manuscript_action="REMOVE Pareto-optimal wording.",
        required_wording_or_constraint=(
            "Use 'trade-off' or 'multi-objective design motivation' "
            "instead of 'Pareto-optimal' unless new evidence is produced."
        ),
        source_stage="Master synthesis",
        source_files="",
        severity="MAJOR",
        final_disposition="REMOVE",
    )

    add_record(
        records,
        evidence_id="ESTR-03",
        reviewer_comments="C20",
        topic="PR-GM and MSS metrics",
        manuscript_claim_or_issue=(
            "PR-GM/MSS are undefined or unreported in reproducible outputs."
        ),
        reproducibility_status="NOT_ESTABLISHED",
        evidence_status="UNSUPPORTED_UNLESS_SOURCE_FOUND",
        validated_evidence=(
            "No validated PR-GM/MSS result is currently part of the "
            "reproducible evidence set."
        ),
        invalid_or_unsupported_evidence=(
            "Undefined metrics cannot remain as evidence of fidelity/utility."
        ),
        final_value_or_interpretation=(
            "Either define and reproduce them from source outputs or remove them."
        ),
        manuscript_action=(
            "Default to REMOVE unless a provenance-valid source is recovered."
        ),
        required_wording_or_constraint=(
            "Do not invent definitions or back-calculate missing values."
        ),
        source_stage="Master synthesis",
        source_files="",
        severity="MAJOR",
        final_disposition="REMOVE_UNLESS_PROVEN",
    )

    add_record(
        records,
        evidence_id="ESTR-04",
        reviewer_comments="C25",
        topic="Hardware reporting",
        manuscript_claim_or_issue=(
            "Conflicting RTX 6000 vs RTX 3080 hardware descriptions exist."
        ),
        reproducibility_status="UNRESOLVED",
        evidence_status="MANUAL_PROVENANCE_REQUIRED",
        validated_evidence=(
            "Hardware must be tied to the exact retained experiment."
        ),
        invalid_or_unsupported_evidence=(
            "A hardware configuration from an unrelated experiment "
            "must not be imported into this manuscript."
        ),
        final_value_or_interpretation=(
            "Retain only hardware supported by the exact experiment logs/code."
        ),
        manuscript_action=(
            "VERIFY retained experiment hardware before final Methods."
        ),
        required_wording_or_constraint=(
            "Do not use unrelated RTX 3080 evidence to resolve the "
            "manuscript's RTX 6000/3080 conflict."
        ),
        source_stage="06 + master synthesis",
        source_files=relative(
            scalability["path"]
        ),
        severity="MAJOR",
        final_disposition="VERIFY_BEFORE_RETAINING",
    )

    add_record(
        records,
        evidence_id="ESTR-05",
        reviewer_comments="C27",
        topic="Baseline tuning fairness",
        manuscript_claim_or_issue=(
            "Model comparisons must use comparable tuning/search budgets."
        ),
        reproducibility_status="PARTIAL",
        evidence_status="REWRITE_REQUIRED",
        validated_evidence=(
            "Corrected classifier evaluation can be retained where the "
            "pipeline and split are reproducible."
        ),
        invalid_or_unsupported_evidence=(
            "Historical superiority claims across differently tuned models "
            "cannot be interpreted as a controlled architecture comparison."
        ),
        final_value_or_interpretation=(
            "Describe baseline results descriptively unless tuning parity "
            "is documented."
        ),
        manuscript_action=(
            "REMOVE claims that attribute performance differences solely "
            "to model architecture when tuning budgets differ or are unknown."
        ),
        required_wording_or_constraint=(
            "Report known hyperparameters and acknowledge unequal/unknown "
            "search budgets where applicable."
        ),
        source_stage="02E,05,master synthesis",
        source_files=relative(
            repeated_path
        ),
        severity="MAJOR",
        final_disposition="REWRITE",
    )

    add_record(
        records,
        evidence_id="ESTR-06",
        reviewer_comments="C30",
        topic="Privacy claims",
        manuscript_claim_or_issue=(
            "Synthetic data are described as privacy-preserving without "
            "a verified privacy evaluation."
        ),
        reproducibility_status="NOT_ESTABLISHED",
        evidence_status="UNSUPPORTED_AS_RESULT",
        validated_evidence=(
            "Privacy can remain as motivation/background only."
        ),
        invalid_or_unsupported_evidence=(
            "No verified membership inference, attribute inference, "
            "differential privacy, or equivalent privacy experiment "
            "was recovered."
        ),
        final_value_or_interpretation=(
            "Synthetic data generation does not automatically establish privacy."
        ),
        manuscript_action=(
            "REMOVE empirical privacy-preservation claims; retain cautious "
            "motivation and explicitly state privacy was not evaluated."
        ),
        required_wording_or_constraint=(
            "Use wording such as 'may reduce direct exposure of original "
            "records' only as motivation, not as a demonstrated guarantee."
        ),
        source_stage="Master synthesis",
        source_files="",
        severity="CRITICAL",
        final_disposition="REWRITE",
    )

    add_record(
        records,
        evidence_id="ESTR-07",
        reviewer_comments="C31,C33",
        topic="Novelty and paper scope",
        manuscript_claim_or_issue=(
            "Current novelty claims depend on unsupported generator, "
            "multimodal, fairness-generation, ablation, and scalability results."
        ),
        reproducibility_status="NARROWER_SCOPE_REQUIRED",
        evidence_status="REFRAME",
        validated_evidence=(
            "The strongest reproducible contribution is a leakage-aware "
            "evaluation/re-audit of structured clinical classification, "
            "with corrected fairness/stability interpretation."
        ),
        invalid_or_unsupported_evidence=(
            "Broad claims of a validated adaptive hybrid generative framework "
            "cannot be sustained by recovered artifacts."
        ),
        final_value_or_interpretation=(
            "Novelty must be narrowed to what the retained evidence actually "
            "supports."
        ),
        manuscript_action=(
            "REWRITE title, abstract, contributions, Methods, Results, "
            "Discussion, and Conclusion around reproducible evidence."
        ),
        required_wording_or_constraint=(
            "Do not claim validated multimodal generalization, synthetic "
            "generation superiority, or proven generator fairness."
        ),
        source_stage="04B_v2,05,06,07,08_v2,09_v2",
        source_files="; ".join(
            filter(
                None,
                [
                    relative(
                        generator["path"]
                    ),
                    relative(
                        utility["path"]
                    ),
                    relative(
                        scalability["path"]
                    ),
                    relative(
                        ablation["path"]
                    ),
                    relative(
                        statistics["path"]
                    ),
                    relative(
                        STAGE09_SUMMARY
                    ),
                ],
            )
        ),
        severity="CRITICAL",
        final_disposition="MAJOR_REWRITE",
    )

    add_record(
        records,
        evidence_id="ESTR-08",
        reviewer_comments="C32",
        topic="Reference audit",
        manuscript_claim_or_issue=(
            "Reference numbering/DOI relevance must be checked independently "
            "of the experimental evidence audit."
        ),
        reproducibility_status="NOT_AUDITED_BY_STAGE10",
        evidence_status="PENDING_REFERENCE_AUDIT",
        validated_evidence=(
            "Stage 10 does not infer or invent bibliographic corrections."
        ),
        invalid_or_unsupported_evidence=(
            "Potential citation mismatch remains possible and must be checked "
            "against actual bibliographic records."
        ),
        final_value_or_interpretation=(
            "Reference audit remains a separate manuscript-finalization task."
        ),
        manuscript_action=(
            "VERIFY DOI, title, journal, author, year, and relevance for "
            "every retained reference."
        ),
        required_wording_or_constraint=(
            "Do not retain a reference solely because it appears in the "
            "current bibliography if it does not support the cited statement."
        ),
        source_stage="Pending final reference audit",
        source_files="",
        severity="MAJOR",
        final_disposition="PENDING",
    )

    return records, diagnostics


# =============================================================================
# 11. CLAIM-TO-SECTION RESTRUCTURING PLAN
# =============================================================================

def build_restructuring_plan(
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "order": 1,
            "section": "Title",
            "action": "REWRITE",
            "target": (
                "Narrow title to the reproducible study scope. "
                "Do not foreground scalability, multimodal validation, "
                "or a verified hybrid generator unless independently supported."
            ),
            "evidence_ids": "E04B-01,E06-01,E09-01,ESTR-07",
        },
        {
            "order": 2,
            "section": "Abstract",
            "action": "REWRITE",
            "target": (
                "Remove unsupported FID, generator superiority, "
                "multimodal generalization, privacy guarantee, scalability, "
                "ablation, and statistical-significance claims. "
                "Report only corrected reproducible classifier/fairness "
                "stability evidence."
            ),
            "evidence_ids": (
                "E02E-01,E03-01,E04-01,E04B-01,E05-01,"
                "E06-01,E07-01,E08-01,E09-01,ESTR-06"
            ),
        },
        {
            "order": 3,
            "section": "Introduction",
            "action": "REWRITE",
            "target": (
                "Retain synthetic-data/fairness motivation cautiously, "
                "but distinguish motivation from demonstrated properties."
            ),
            "evidence_ids": "E04B-01,ESTR-06,ESTR-07",
        },
        {
            "order": 4,
            "section": "Contributions",
            "action": "REWRITE",
            "target": (
                "List only contributions directly supported by retained "
                "experiments and audits."
            ),
            "evidence_ids": "E02E-01,E03-01,E08-01,ESTR-07",
        },
        {
            "order": 5,
            "section": "Methods - Framework",
            "action": "MAJOR_REWRITE",
            "target": (
                "Separate conceptual framework from executable recovered "
                "implementation. Remove claims that non-recovered GAN/VAE/"
                "diffusion/fairness/scalability modules were executed."
            ),
            "evidence_ids": "E04B-01,ESTR-01,ESTR-02",
        },
        {
            "order": 6,
            "section": "Methods - Dataset",
            "action": "RETAIN_AND_CLARIFY",
            "target": (
                f"Describe the structured clinical dataset and original "
                f"cohort size ({EXPECTED_PARTICIPANTS}) exactly. "
                "Do not call the clinical dataset multimodal."
            ),
            "evidence_ids": "E09-01,E09-04",
        },
        {
            "order": 7,
            "section": "Methods - Preprocessing/Splits",
            "action": "REWRITE",
            "target": (
                "Use split-first leakage-safe preprocessing and clearly "
                "distinguish corrected evaluation from invalid historical "
                "pipeline behavior."
            ),
            "evidence_ids": "E02E-01,E02F-01,E05-01",
        },
        {
            "order": 8,
            "section": "Methods - Fairness",
            "action": "REWRITE",
            "target": (
                "Define favorable outcome, sensitive attribute, reference "
                "group, subgroup counts, SPD/EOD/DI formulas, and interpret "
                "them as classifier fairness metrics."
            ),
            "evidence_ids": "E03-01",
        },
        {
            "order": 9,
            "section": "Methods - Statistics",
            "action": "REWRITE",
            "target": (
                "Report repeated holdouts descriptively. No formal p-values "
                "or independent-replicate significance claims."
            ),
            "evidence_ids": "E08-01",
        },
        {
            "order": 10,
            "section": "Methods - Hardware",
            "action": "VERIFY",
            "target": (
                "Retain only hardware tied to retained experiments."
            ),
            "evidence_ids": "ESTR-04",
        },
        {
            "order": 11,
            "section": "Results - Primary performance",
            "action": "REBUILD",
            "target": (
                "Use corrected leakage-safe repeated-holdout performance. "
                "Do not headline contaminated perfect scores."
            ),
            "evidence_ids": "E02E-01,E05-01",
        },
        {
            "order": 12,
            "section": "Results - Fairness",
            "action": "REBUILD",
            "target": (
                "Use corrected classifier fairness metrics with cautious "
                "descriptive interpretation."
            ),
            "evidence_ids": "E03-01,E08-01",
        },
        {
            "order": 13,
            "section": "Results - Fidelity/FID",
            "action": "REMOVE",
            "target": (
                "Remove unsupported FID/SFD values and generator-fidelity "
                "claims."
            ),
            "evidence_ids": "E04-01,E04B-01",
        },
        {
            "order": 14,
            "section": "Results - Synthetic utility",
            "action": "REMOVE_OR_REFRAME",
            "target": (
                "Remove synthetic-vs-real superiority claims; retain only "
                "the reproducible real reference if needed."
            ),
            "evidence_ids": "E05-01",
        },
        {
            "order": 15,
            "section": "Results - Scalability",
            "action": "REMOVE",
            "target": (
                "Remove unsupported N/runtime/memory generator-scale claims."
            ),
            "evidence_ids": "E06-01",
        },
        {
            "order": 16,
            "section": "Results - Ablations",
            "action": "REMOVE",
            "target": (
                "Remove unreproducible historical generator/fusion/fairness/"
                "scalability ablation claims and plots."
            ),
            "evidence_ids": "E07-01",
        },
        {
            "order": 17,
            "section": "Results - Multimodal/Cross-domain",
            "action": "REMOVE",
            "target": (
                "Remove unsupported ArSL→clinical/multimodal claims from "
                "the clinical evidence narrative."
            ),
            "evidence_ids": "E09-01",
        },
        {
            "order": 18,
            "section": "Figures",
            "action": "REBUILD_AND_RENUMBER",
            "target": (
                "Remove unsupported figures first, then renumber retained "
                "figures sequentially and verify all textual references."
            ),
            "evidence_ids": "E09-02,E09-03",
        },
        {
            "order": 19,
            "section": "Discussion",
            "action": "MAJOR_REWRITE",
            "target": (
                "Discuss only verified classifier/fairness stability results, "
                "limitations of feature timing, lack of generator provenance, "
                "lack of formal privacy evaluation, and limited generalizability."
            ),
            "evidence_ids": (
                "E02F-01,E03-01,E04B-01,E06-01,E07-01,"
                "E08-01,E09-01,ESTR-06,ESTR-07"
            ),
        },
        {
            "order": 20,
            "section": "Limitations",
            "action": "EXPAND",
            "target": (
                "Explicitly state cohort size, repeated-holdout dependence, "
                "feature chronology uncertainty, missing generator provenance, "
                "absence of validated multimodal/generalization/privacy claims, "
                "and non-reproducible historical ablations/scalability."
            ),
            "evidence_ids": (
                "E02F-01,E04B-01,E06-01,E07-01,E08-01,E09-01,ESTR-06"
            ),
        },
        {
            "order": 21,
            "section": "Conclusion",
            "action": "REWRITE",
            "target": (
                "Conclude only what is supported by retained leakage-safe "
                "structured-clinical evaluation."
            ),
            "evidence_ids": "E02E-01,E03-01,E08-01,ESTR-07",
        },
        {
            "order": 22,
            "section": "References",
            "action": "AUDIT",
            "target": (
                "Verify every retained reference and DOI independently."
            ),
            "evidence_ids": "ESTR-08",
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# 12. BUILD REVIEWER COMMENT DISPOSITION
# =============================================================================

def build_comment_disposition(
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for comment_id, topic in REVIEWER_MAP.items():
        mask = (
            evidence_df[
                "reviewer_comments"
            ]
            .fillna("")
            .str.contains(
                rf"(?<!\d){re.escape(comment_id)}(?!\d)",
                regex=True,
            )
        )

        matches = evidence_df[
            mask
        ]

        if matches.empty:
            rows.append(
                {
                    "reviewer_comment": comment_id,
                    "reviewer_topic": topic,
                    "evidence_ids": "",
                    "stage10_status": "PENDING_DIRECT_MANUSCRIPT_EDIT",
                    "recommended_disposition": "MANUAL_REVIEW",
                    "basis": (
                        "No dedicated master-evidence row was mapped "
                        "automatically."
                    ),
                }
            )
            continue

        dispositions = sorted(
            {
                str(x)
                for x in matches[
                    "final_disposition"
                ].dropna()
            }
        )

        statuses = sorted(
            {
                str(x)
                for x in matches[
                    "evidence_status"
                ].dropna()
            }
        )

        rows.append(
            {
                "reviewer_comment": comment_id,
                "reviewer_topic": topic,
                "evidence_ids": ",".join(
                    matches[
                        "evidence_id"
                    ].astype(
                        str
                    ).tolist()
                ),
                "stage10_status": (
                    "EVIDENCE_SYNTHESIZED"
                ),
                "recommended_disposition": ";".join(
                    dispositions
                ),
                "basis": ";".join(
                    statuses
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 13. FINAL CLAIM WHITELIST / BLACKLIST
# =============================================================================

def build_claim_policy() -> pd.DataFrame:
    rows = [
        {
            "claim_class": "ALLOW",
            "claim": (
                "The study uses a structured clinical dataset "
                f"with {EXPECTED_PARTICIPANTS} participants."
            ),
            "condition": (
                "Retain only if dataset description is unchanged "
                "and source documentation supports the count."
            ),
        },
        {
            "claim_class": "ALLOW",
            "claim": (
                "Repeated leakage-safe holdouts provide descriptive "
                "stability evidence."
            ),
            "condition": (
                "Report mean/SD/median/range; no independent-replicate "
                "significance claim."
            ),
        },
        {
            "claim_class": "ALLOW",
            "claim": (
                "Corrected SPD/EOD/DI quantify fairness of the "
                "real-outcome classifier."
            ),
            "condition": (
                "State sensitive attribute, reference group, favorable "
                "outcome, subgroup counts, and classifier scope."
            ),
        },
        {
            "claim_class": "ALLOW_WITH_LIMITATION",
            "claim": (
                "No direct temporal/outcome proxy leakage was identified."
            ),
            "condition": (
                "Also state that feature timing remains incompletely documented."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "A verified GAN/VAE/diffusion hybrid structured-data "
                "generator was implemented and evaluated."
            ),
            "condition": (
                "Recovered project does not support this."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": "Validated conventional FID = 1.5/1.6/3.2.",
            "condition": (
                "No provenance-valid generator/feature-space evidence."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "HFAGM is proven multimodal or cross-domain on "
                "clinical + ArSL data."
            ),
            "condition": (
                "No strict executable clinical multimodal+fusion evidence."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "Synthetic data preserve or improve fairness based on "
                "the corrected Stage 03 metrics."
            ),
            "condition": (
                "Stage 03 fairness applies to real-outcome classifier."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "Synthetic utility exceeds real-data utility."
            ),
            "condition": (
                "No provenance-valid labeled synthetic 51-feature dataset "
                "was recovered for common-test evaluation."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "Generator is ~25% faster or uses 20–30% less GPU memory."
            ),
            "condition": (
                "No matched generator-linked computational benchmark."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "N=100000 represents more independent information "
                "than the original cohort."
            ),
            "condition": (
                "Synthetic rows are computational workload only."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": "Pareto-optimal solution/frontier.",
            "condition": (
                "No explicit frontier/convergence evidence."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "Significantly improved/outperformed based solely "
                "on the ten repeated holdouts."
            ),
            "condition": (
                "Same cohort reused; descriptive stability only."
            ),
        },
        {
            "claim_class": "DO_NOT_CLAIM",
            "claim": (
                "Synthetic-data privacy protection was empirically proven."
            ),
            "condition": (
                "No validated privacy experiment recovered."
            ),
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# 14. REPORT WRITING
# =============================================================================

def write_master_summary(
    evidence_df: pd.DataFrame,
    restructure_df: pd.DataFrame,
    comment_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
) -> None:
    critical = evidence_df[
        evidence_df["severity"]
        == "CRITICAL"
    ]

    remove_like = evidence_df[
        evidence_df[
            "final_disposition"
        ].isin(
            [
                "REMOVE",
                "REMOVE_OR_MAJOR_REFRAME",
                "REMOVE_UNLESS_PROVEN",
            ]
        )
    ]

    rewrite_like = evidence_df[
        evidence_df[
            "final_disposition"
        ].isin(
            [
                "REWRITE",
                "MAJOR_REWRITE",
                "REWRITE_OR_REMOVE",
                "CORRECT_AFTER_RESTRUCTURE",
                "VERIFY_BEFORE_RETAINING",
            ]
        )
    ]

    lines = [
        "=" * 110,
        "HFAGM - STAGE 10 MASTER EVIDENCE AND MANUSCRIPT RESTRUCTURING AUDIT",
        "=" * 110,
        "",
        f"Generated: {GENERATED_AT}",
        f"Project root: {PROJECT_ROOT}",
        f"Manuscript target: {MANUSCRIPT_PATH}",
        "",
        "PURPOSE",
        "-" * 110,
        (
            "Consolidate Stages 02E/02F, 03, 04/04B, 05, 06, 07, 08, and 09 "
            "into one controlling evidence table for manuscript revision."
        ),
        (
            "This stage does not create new experimental evidence. It determines "
            "what may be retained, rewritten, verified, or removed."
        ),
        "",
        "COUNTS",
        "-" * 110,
        f"Master evidence records: {len(evidence_df)}",
        f"Critical records: {len(critical)}",
        f"Remove/remove-or-reframe records: {len(remove_like)}",
        f"Rewrite/verify/correct records: {len(rewrite_like)}",
        f"Restructuring actions: {len(restructure_df)}",
        f"Reviewer comments mapped: {len(comment_df)}",
        f"Claim-policy rows: {len(policy_df)}",
        "",
        "CONTROLLING CONCLUSIONS",
        "-" * 110,
        (
            "1. The recovered project does not substantiate a verified structured "
            "GAN/VAE/diffusion hybrid generator."
        ),
        (
            "2. Historical FID/SFD, generator scalability, synthetic utility, "
            "generator ablations, multimodal clinical generalization, and "
            "generator-fairness claims are not reproducible from recovered artifacts."
        ),
        (
            "3. The strongest reproducible empirical evidence is the corrected "
            "leakage-safe structured-clinical classifier evaluation and its "
            "descriptive fairness/stability analysis."
        ),
        (
            "4. Repeated holdouts reuse the same participant cohort and therefore "
            "support descriptive stability, not independent-replicate significance."
        ),
        (
            "5. Figures and numbering must be rebuilt only after unsupported "
            "result figures are removed/reframed."
        ),
        (
            "6. Any future generator/ablation/privacy/scalability experiment must "
            "be labeled as a new corrected experiment, not historical reconstruction."
        ),
        "",
        "FINAL STAGE-10 VERDICT",
        "-" * 110,
        "MAJOR_EVIDENCE_CONSTRAINED_MANUSCRIPT_RESTRUCTURING_REQUIRED",
        "",
        "NEXT ACTION",
        "-" * 110,
        (
            "Use master_evidence_table.csv and manuscript_restructuring_plan.csv "
            "as the controlling source to rewrite the manuscript section-by-section."
        ),
        "",
        "RECOMMENDED MANUSCRIPT ORDER",
        "-" * 110,
    ]

    for _, row in (
        restructure_df.sort_values(
            "order"
        ).iterrows()
    ):
        lines.append(
            f"{int(row['order']):02d}. "
            f"{row['section']} | {row['action']} | {row['target']}"
        )

    lines.extend(
        [
            "",
            "SAFETY",
            "-" * 110,
            "New model training: NO",
            "Synthetic generation: NO",
            "New ablation creation: NO",
            "OCR: NO",
            "Image modification: NO",
            "Historical output modification: NO",
            "Manuscript modification: NO",
            "",
            "=" * 110,
        ]
    )

    (
        OUTPUT_DIR
        / "master_evidence_and_restructuring_audit_summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 15. MAIN
# =============================================================================

def main() -> None:
    print("=" * 110)
    print(
        "HFAGM - STAGE 10 MASTER EVIDENCE AND "
        "MANUSCRIPT RESTRUCTURING AUDIT"
    )
    print("=" * 110)

    print()
    print("Restrictions:")
    print("  - consolidate existing evidence only")
    print("  - no model training")
    print("  - no synthetic generation")
    print("  - no new ablation creation")
    print("  - no OCR")
    print("  - no historical output modification")
    print("  - no manuscript modification")

    records, diagnostics = (
        build_master_records()
    )

    evidence_df = pd.DataFrame(
        [
            asdict(record)
            for record in records
        ]
    )

    restructure_df = (
        build_restructuring_plan(
            evidence_df
        )
    )

    comment_df = (
        build_comment_disposition(
            evidence_df
        )
    )

    policy_df = build_claim_policy()

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    evidence_path = (
        OUTPUT_DIR
        / "master_evidence_table.csv"
    )

    evidence_df.to_csv(
        evidence_path,
        index=False,
        encoding="utf-8-sig",
    )

    restructure_path = (
        OUTPUT_DIR
        / "manuscript_restructuring_plan.csv"
    )

    restructure_df.to_csv(
        restructure_path,
        index=False,
        encoding="utf-8-sig",
    )

    comment_path = (
        OUTPUT_DIR
        / "reviewer_comment_disposition.csv"
    )

    comment_df.to_csv(
        comment_path,
        index=False,
        encoding="utf-8-sig",
    )

    policy_path = (
        OUTPUT_DIR
        / "final_claim_policy.csv"
    )

    policy_df.to_csv(
        policy_path,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics_path = (
        OUTPUT_DIR
        / "source_resolution_diagnostics.json"
    )

    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Compact source-resolution table
    source_rows = []

    for key, value in diagnostics.items():
        if isinstance(
            value,
            dict,
        ):
            path = value.get(
                "path"
            )

            source_rows.append(
                {
                    "stage_key": key,
                    "resolved_path": (
                        str(path)
                        if path
                        else ""
                    ),
                    "available": int(
                        bool(
                            value.get(
                                "available",
                                value.get(
                                    "summary_available",
                                    False,
                                ),
                            )
                        )
                    ),
                    "diagnostic": safe_json(
                        {
                            k: v
                            for k, v in value.items()
                            if k not in {
                                "path",
                                "summary",
                                "conditions",
                                "text_excerpt",
                            }
                        }
                    ),
                }
            )

    pd.DataFrame(
        source_rows
    ).to_csv(
        OUTPUT_DIR
        / "source_resolution_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Reproducibility matrix
    reproducibility_df = evidence_df[
        [
            "evidence_id",
            "reviewer_comments",
            "topic",
            "reproducibility_status",
            "evidence_status",
            "severity",
            "final_disposition",
            "source_stage",
            "source_files",
        ]
    ].copy()

    reproducibility_df.to_csv(
        OUTPUT_DIR
        / "reproducibility_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # High-priority removal list
    removal_mask = evidence_df[
        "final_disposition"
    ].isin(
        [
            "REMOVE",
            "REMOVE_OR_MAJOR_REFRAME",
            "REMOVE_UNLESS_PROVEN",
        ]
    )

    evidence_df[
        removal_mask
    ].to_csv(
        OUTPUT_DIR
        / "claims_to_remove_or_majorly_reframe.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # High-priority retained/rewrite evidence
    retain_mask = ~removal_mask

    evidence_df[
        retain_mask
    ].to_csv(
        OUTPUT_DIR
        / "claims_to_retain_rewrite_or_verify.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_master_summary(
        evidence_df,
        restructure_df,
        comment_df,
        policy_df,
        diagnostics,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    critical_count = int(
        (
            evidence_df[
                "severity"
            ]
            ==
            "CRITICAL"
        ).sum()
    )

    remove_count = int(
        removal_mask.sum()
    )

    rewrite_count = int(
        evidence_df[
            "final_disposition"
        ].isin(
            [
                "REWRITE",
                "MAJOR_REWRITE",
                "REWRITE_OR_REMOVE",
                "CORRECT_AFTER_RESTRUCTURE",
                "VERIFY_BEFORE_RETAINING",
            ]
        ).sum()
    )

    print()
    print("=" * 110)
    print("STAGE 10 COMPLETE")
    print("=" * 110)

    print(
        f"Master evidence records: "
        f"{len(evidence_df)}"
    )

    print(
        f"Critical records: "
        f"{critical_count}"
    )

    print(
        f"Remove/remove-or-majorly-reframe records: "
        f"{remove_count}"
    )

    print(
        f"Rewrite/verify/correct records: "
        f"{rewrite_count}"
    )

    print(
        f"Reviewer comments mapped: "
        f"{len(comment_df)}"
    )

    print()
    print("FINAL VERDICT:")
    print(
        "MAJOR_EVIDENCE_CONSTRAINED_"
        "MANUSCRIPT_RESTRUCTURING_REQUIRED"
    )

    print()
    print("NEXT ACTION:")
    print(
        "USE_MASTER_EVIDENCE_TABLE_TO_REWRITE_"
        "MANUSCRIPT_SECTION_BY_SECTION"
    )

    print()
    print("Results written to:")
    print(OUTPUT_DIR)

    print()
    print("Upload these files first:")

    for filename in [
        "master_evidence_and_restructuring_audit_summary.txt",
        "master_evidence_table.csv",
        "manuscript_restructuring_plan.csv",
        "reviewer_comment_disposition.csv",
        "final_claim_policy.csv",
        "reproducibility_matrix.csv",
        "claims_to_remove_or_majorly_reframe.csv",
        "claims_to_retain_rewrite_or_verify.csv",
        "source_resolution_table.csv",
        "source_resolution_diagnostics.json",
    ]:
        print(
            OUTPUT_DIR / filename
        )


# =============================================================================
# 16. SAFE EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 110)
        print("STAGE 10 FAILED SAFELY")
        print("=" * 110)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "No model was trained, no synthetic data were generated, "
            "no ablation was created, no OCR was performed, "
            "no historical output was modified, and no manuscript "
            "file was changed."
        )

        print()
        traceback.print_exc()

        sys.exit(1)
