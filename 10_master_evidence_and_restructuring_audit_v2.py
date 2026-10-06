from __future__ import annotations

import json
import re
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

OUTPUTS_ROOT = (
    PROJECT_ROOT / "outputs"
)

OUTPUT_DIR = (
    OUTPUTS_ROOT
    / "revision_master_evidence_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GENERATED_AT = datetime.now().isoformat(
    timespec="seconds"
)

EXPECTED_PARTICIPANTS = 193

MAX_TEXT_BYTES = 25 * 1024 * 1024


# =============================================================================
# 2. UNIQUE VERDICT / SIGNATURE TERMS FOR PRIOR STAGES
# =============================================================================

STAGE_SIGNATURES = {
    "02F_v2": [
        "NO_DIRECT_TEMPORAL_LEAKAGE_EVIDENCE_BUT_HIGHLY_PREDICTIVE_FEATURE_TIMING_REMAINS_UNDOCUMENTED",
    ],
    "04": [
        "NO_REPRODUCIBLE_FID_OR_STRUCTURED_FRECHET_RESULT_AVAILABLE",
    ],
    "04B_v2": [
        "CLASSIFIER_OR_ENCODER_ONLY_NO_STRUCTURED_GENERATOR",
        "DO_NOT_REGENERATE_REMOVE_UNSUPPORTED_GENERATIVE_FID_CLAIMS",
    ],
    "05": [
        "SYNTHETIC_UTILITY_NOT_RECOMPUTABLE_FROM_RECOVERED_ARTIFACTS",
    ],
    "06": [
        "GENERATIVE_SCALABILITY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS",
        "REMOVE_UNSUPPORTED_RUNTIME_MEMORY_AND_SYNTHETIC_SCALE_CLAIMS",
    ],
    "07": [
        "ABLATION_STUDY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS",
        "REMOVE_UNSUPPORTED_ABLATION_CLAIMS",
    ],
    "08_v2": [
        "DESCRIPTIVE_STABILITY_EVIDENCE_VALID_FORMAL_SIGNIFICANCE_NOT_SUPPORTED",
        "FORMAL_SIGNIFICANCE_NOT_SUPPORTED",
    ],
    "09_v2": [
        "ARRAY_MANUSCRIPT_REQUIRES_MULTIMODAL_GENERATIVE_REFRAMING",
        "RETAIN_REPRODUCIBLE_STRUCTURED_CLINICAL_RESULTS_AND_CORRECT_FIGURE_NUMBERING_AND_NUMERICS",
    ],
}


# =============================================================================
# 3. FIXED PATHS FOR STAGES THAT ARE KNOWN
# =============================================================================

REPEATED_METRICS_PATH = (
    OUTPUTS_ROOT
    / "revision_primary_metrics"
    / "repeated_leakage_safe_evaluation"
    / "repeated_seed_metrics.csv"
)

FAIRNESS_DIR = (
    OUTPUTS_ROOT
    / "revision_fairness"
    / "recomputed_fairness"
)

FAIRNESS_PER_SEED_PATH = (
    FAIRNESS_DIR
    / "fairness_per_seed_reference.csv"
)

FAIRNESS_SUMMARY_PATH = (
    FAIRNESS_DIR
    / "fairness_summary_reference.csv"
)

STAGE09_DIR = (
    OUTPUTS_ROOT
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
# 4. REVIEWER COMMENT MAP
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
# 5. EVIDENCE RECORD
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
# 6. GENERAL HELPERS
# =============================================================================

def normalize_ws(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""

    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
    except Exception:
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


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
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


def relative(path: Optional[Path]) -> str:
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


def safe_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        default=str,
    )


def add_record(
    rows: List[EvidenceRecord],
    **kwargs: Any,
) -> None:
    rows.append(
        EvidenceRecord(**kwargs)
    )


# =============================================================================
# 7. ROBUST OUTPUT SEARCH BY UNIQUE VERDICT
# =============================================================================

def candidate_text_files() -> List[Path]:
    files: List[Path] = []

    if not OUTPUTS_ROOT.exists():
        return files

    for path in OUTPUTS_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if OUTPUT_DIR in path.parents:
            continue

        if path.suffix.lower() not in {
            ".txt",
            ".log",
            ".md",
            ".csv",
            ".json",
        }:
            continue

        files.append(path)

    return files


def resolve_stage_by_signatures(
    stage_key: str,
    signatures: Sequence[str],
    files: Sequence[Path],
) -> Dict[str, Any]:
    """
    Search the outputs tree for files containing stage-specific final verdict
    strings. This avoids brittle hard-coded filenames.

    If multiple files match, rank them by:
      1. number of matched signatures
      2. preference for .txt summary-like files
      3. filename containing summary/audit
      4. newest mtime
    """
    matches: List[Dict[str, Any]] = []

    for path in files:
        text = read_text(path)

        if not text:
            continue

        upper = text.upper()

        matched = [
            term
            for term in signatures
            if term.upper() in upper
        ]

        if not matched:
            continue

        score = len(matched) * 100

        if path.suffix.lower() == ".txt":
            score += 20

        name_lower = path.name.lower()

        if "summary" in name_lower:
            score += 10

        if "audit" in name_lower:
            score += 5

        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0

        matches.append(
            {
                "path": path,
                "matched_terms": matched,
                "score": score,
                "mtime": mtime,
                "text": text,
            }
        )

    matches.sort(
        key=lambda item: (
            -item["score"],
            -item["mtime"],
            str(item["path"]).lower(),
        )
    )

    if not matches:
        return {
            "stage": stage_key,
            "resolved": False,
            "path": None,
            "matched_terms": [],
            "candidate_count": 0,
            "ambiguity": False,
            "all_candidates": [],
            "text_excerpt": "",
        }

    best = matches[0]
    top_score = best["score"]

    tied = [
        item
        for item in matches
        if item["score"] == top_score
    ]

    ambiguity = len(tied) > 1

    excerpt = ""

    if best["matched_terms"]:
        term = best["matched_terms"][0]
        upper = best["text"].upper()
        idx = upper.find(term.upper())
        start = max(0, idx - 600)
        end = min(
            len(best["text"]),
            idx + 1400,
        )
        excerpt = normalize_ws(
            best["text"][start:end]
        )

    return {
        "stage": stage_key,
        "resolved": True,
        "path": best["path"],
        "matched_terms": best["matched_terms"],
        "candidate_count": len(matches),
        "ambiguity": ambiguity,
        "all_candidates": [
            {
                "path": str(item["path"]),
                "matched_terms": item["matched_terms"],
                "score": item["score"],
                "mtime": item["mtime"],
            }
            for item in matches
        ],
        "text_excerpt": excerpt,
    }


# =============================================================================
# 8. REPEATED METRICS
# =============================================================================

def summarize_repeated_metrics(
    path: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": path,
        "available": False,
        "rows": 0,
        "conditions": {},
    }

    df = read_csv_safe(path)

    if df.empty:
        return result

    result["available"] = True
    result["rows"] = len(df)

    lower = {
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
        if candidate in lower:
            condition_col = lower[candidate]
            break

    metric_aliases = {
        "accuracy": ("accuracy",),
        "precision": ("precision",),
        "sensitivity": (
            "sensitivity",
            "recall",
        ),
        "specificity": ("specificity",),
        "f1": (
            "f1",
            "f1_score",
        ),
        "roc_auc": (
            "roc_auc",
            "auc",
        ),
    }

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
        cond_summary: Dict[
            str,
            Any,
        ] = {}

        for standard_name, aliases in (
            metric_aliases.items()
        ):
            col = None

            for alias in aliases:
                if alias in lower:
                    col = lower[alias]
                    break

            if col is None:
                continue

            values = pd.to_numeric(
                g[col],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            cond_summary[
                standard_name
            ] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(
                    values.std(ddof=1)
                ) if len(values) > 1 else 0.0,
                "median": float(
                    values.median()
                ),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        result["conditions"][
            str(condition)
        ] = cond_summary

    return result


# =============================================================================
# 9. FAIRNESS V2 PARSER
# =============================================================================

def detect_metric_columns(
    df: pd.DataFrame,
) -> Dict[str, str]:
    lower = {
        c.lower(): c
        for c in df.columns
    }

    aliases = {
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

    found: Dict[str, str] = {}

    for metric, names in aliases.items():
        for name in names:
            if name in lower:
                found[metric] = lower[name]
                break

    return found


def summarize_fairness_v2() -> Dict[str, Any]:
    """
    Prefer per-seed file because it permits explicit aggregation.
    If unavailable, fall back to summary file and preserve raw rows.
    """
    result: Dict[str, Any] = {
        "selected_path": None,
        "available": False,
        "source_kind": "",
        "rows": 0,
        "metric_columns": {},
        "group_columns": [],
        "summary": {},
        "raw_rows": [],
    }

    selected = None
    source_kind = ""

    if FAIRNESS_PER_SEED_PATH.exists():
        selected = FAIRNESS_PER_SEED_PATH
        source_kind = "per_seed"
    elif FAIRNESS_SUMMARY_PATH.exists():
        selected = FAIRNESS_SUMMARY_PATH
        source_kind = "summary"
    else:
        return result

    df = read_csv_safe(selected)

    if df.empty:
        return result

    result["selected_path"] = selected
    result["available"] = True
    result["source_kind"] = source_kind
    result["rows"] = len(df)

    metric_cols = detect_metric_columns(df)
    result["metric_columns"] = metric_cols

    lower = {
        c.lower(): c
        for c in df.columns
    }

    group_aliases = [
        "condition",
        "sensitive_attribute",
        "sensitive attribute",
        "attribute",
        "comparison",
        "reference_group",
        "reference group",
        "compared_group",
        "compared group",
    ]

    group_cols: List[str] = []

    for alias in group_aliases:
        if alias in lower:
            col = lower[alias]
            if col not in group_cols:
                group_cols.append(col)

    result["group_columns"] = group_cols

    result["raw_rows"] = (
        df.head(50)
        .where(
            pd.notna(df.head(50)),
            None,
        )
        .to_dict(
            orient="records"
        )
    )

    if not metric_cols:
        # Preserve summary rows even if the file uses long format.
        long_metric_col = None
        long_value_col = None

        for candidate in (
            "metric",
            "fairness_metric",
            "measure",
        ):
            if candidate in lower:
                long_metric_col = lower[candidate]
                break

        for candidate in (
            "value",
            "mean",
            "metric_value",
        ):
            if candidate in lower:
                long_value_col = lower[candidate]
                break

        if (
            long_metric_col is not None
            and long_value_col is not None
        ):
            temp = df.copy()

            temp["_metric_norm"] = (
                temp[
                    long_metric_col
                ]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            for metric in (
                "spd",
                "eod",
                "di",
            ):
                subset = temp[
                    temp["_metric_norm"]
                    .str.contains(
                        metric,
                        regex=False,
                    )
                ]

                if subset.empty:
                    continue

                vals = pd.to_numeric(
                    subset[
                        long_value_col
                    ],
                    errors="coerce",
                ).dropna()

                if vals.empty:
                    continue

                result["summary"][
                    metric
                ] = {
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "sd": float(
                        vals.std(ddof=1)
                    ) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }

        return result

    if group_cols:
        grouped = df.groupby(
            group_cols,
            dropna=False,
        )
    else:
        grouped = [("all", df)]

    for group_key, g in grouped:
        if isinstance(
            group_key,
            tuple,
        ):
            group_name = " | ".join(
                str(x)
                for x in group_key
            )
        else:
            group_name = str(
                group_key
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
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(
                    values.std(ddof=1)
                ) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }

    return result


# =============================================================================
# 10. STAGE 09 LOAD
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
# 11. BUILD MASTER EVIDENCE TABLE
# =============================================================================

def build_master_records(
    resolved: Dict[str, Dict[str, Any]],
) -> Tuple[
    List[EvidenceRecord],
    Dict[str, Any],
]:
    records: List[
        EvidenceRecord
    ] = []

    diagnostics: Dict[str, Any] = {
        "resolved_stage_sources": {},
    }

    repeated = summarize_repeated_metrics(
        REPEATED_METRICS_PATH
    )

    diagnostics[
        "stage02e_repeated_metrics"
    ] = repeated

    fairness = summarize_fairness_v2()

    diagnostics[
        "stage03_fairness"
    ] = fairness

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

    for key, info in resolved.items():
        diagnostics[
            "resolved_stage_sources"
        ][key] = {
            "resolved": info["resolved"],
            "path": str(
                info["path"]
            ) if info["path"] else None,
            "matched_terms": info[
                "matched_terms"
            ],
            "candidate_count": info[
                "candidate_count"
            ],
            "ambiguity": info[
                "ambiguity"
            ],
            "all_candidates": info[
                "all_candidates"
            ],
            "text_excerpt": info[
                "text_excerpt"
            ],
        }

    # -------------------------------------------------------------------------
    # 02E
    # -------------------------------------------------------------------------
    if repeated["available"]:
        parts: List[str] = []

        for condition, metrics in (
            repeated["conditions"].items()
        ):
            metric_parts: List[str] = []

            for metric, stats in (
                metrics.items()
            ):
                metric_parts.append(
                    f"{metric}="
                    f"{stats['mean']:.6f}±"
                    f"{stats['sd']:.6f}"
                )

            parts.append(
                f"{condition}: "
                + "; ".join(
                    metric_parts
                )
            )

        repeated_text = " | ".join(parts)
    else:
        repeated_text = (
            "Repeated leakage-safe metrics not resolved."
        )

    add_record(
        records,
        evidence_id="E02E-01",
        reviewer_comments="C21,C22,C27,C29",
        topic="Repeated leakage-safe classifier performance",
        manuscript_claim_or_issue=(
            "Primary classifier performance must be based on "
            "leakage-safe evaluation rather than the historical "
            "contaminated workflow."
        ),
        reproducibility_status=(
            "REPRODUCIBLE"
            if repeated["available"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "VALIDATED"
            if repeated["available"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=repeated_text,
        invalid_or_unsupported_evidence=(
            "Historical perfect metrics from the contaminated "
            "pre-split scaling/oversampling workflow must not be "
            "used as headline evidence."
        ),
        final_value_or_interpretation=(
            "Ten seeds represent descriptive stability across "
            "repeated leakage-safe holdouts of the same cohort."
        ),
        manuscript_action=(
            "RETAIN corrected repeated-holdout results; "
            "REMOVE/REPLACE contaminated primary claims."
        ),
        required_wording_or_constraint=(
            "Report mean±SD/range; do not call seeds independent "
            "replicates."
        ),
        source_stage="02E",
        source_files=relative(
            REPEATED_METRICS_PATH
        ),
        severity="CRITICAL",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # 02F
    # -------------------------------------------------------------------------
    r02f = resolved[
        "02F_v2"
    ]

    add_record(
        records,
        evidence_id="E02F-01",
        reviewer_comments="C21,C27",
        topic="Feature chronology and proxy leakage",
        manuscript_claim_or_issue=(
            "Timing of highly predictive features relative to "
            "outcome determination is incompletely documented."
        ),
        reproducibility_status=(
            "AUDITED"
            if r02f["resolved"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "LIMITATION"
            if r02f["resolved"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            "No direct temporal/outcome proxy leakage was identified "
            "by the recovered audit."
            if r02f["resolved"]
            else "Stage 02F source not resolved."
        ),
        invalid_or_unsupported_evidence=(
            "Absence of a direct proxy flag does not prove all "
            "predictor timing was prospectively valid."
        ),
        final_value_or_interpretation=(
            "Feature timing remains an explicit limitation."
        ),
        manuscript_action=(
            "RETAIN as a limitation; avoid deployment-ready or "
            "fully prospective validation claims."
        ),
        required_wording_or_constraint=(
            "State that no direct temporal leakage was identified, "
            "but predictor timing could not be fully verified."
        ),
        source_stage="02F_v2",
        source_files=relative(
            r02f["path"]
        ),
        severity="MAJOR",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # 03 fairness
    # -------------------------------------------------------------------------
    fairness_valid = (
        fairness["available"]
        and bool(
            fairness["summary"]
            or fairness["raw_rows"]
        )
    )

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
            if fairness_valid
            else "SOURCE_OR_PARSE_MISSING"
        ),
        evidence_status=(
            "VALIDATED_CLASSIFIER_FAIRNESS"
            if fairness_valid
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            safe_json(
                fairness["summary"]
            )
            if fairness["summary"]
            else safe_json(
                fairness["raw_rows"]
            )
        ),
        invalid_or_unsupported_evidence=(
            "EOD=1 is not perfect fairness under the standard "
            "signed-difference interpretation. DI 1→0.643→0.209 "
            "does not demonstrate movement toward parity."
        ),
        final_value_or_interpretation=(
            "SPD≈0, EOD≈0, DI≈1 indicate parity. These corrected "
            "metrics apply to the real-outcome classifier, not to "
            "an unverified synthetic generator."
        ),
        manuscript_action=(
            "REPLACE historical fairness interpretation with "
            "corrected classifier-fairness reporting."
        ),
        required_wording_or_constraint=(
            "State sensitive attribute, reference group, favorable "
            "outcome, subgroup counts, and classifier scope."
        ),
        source_stage="03",
        source_files=relative(
            fairness["selected_path"]
        ),
        severity="CRITICAL",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # 04
    # -------------------------------------------------------------------------
    r04 = resolved["04"]

    add_record(
        records,
        evidence_id="E04-01",
        reviewer_comments="C19,C24",
        topic="FID / Structured Fréchet Distance",
        manuscript_claim_or_issue=(
            "FID-like fidelity values lack recoverable generator "
            "and feature-space provenance."
        ),
        reproducibility_status=(
            "NOT_REPRODUCIBLE"
            if r04["resolved"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "UNSUPPORTED"
            if r04["resolved"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            "The Stage 04 audit found no reproducible conventional "
            "FID or structured Fréchet result from provenance-valid "
            "synthetic artifacts."
        ),
        invalid_or_unsupported_evidence=(
            "Historical values 1.5, 1.6, and 3.2 are unsupported; "
            "structured predictors do not establish conventional "
            "image FID."
        ),
        final_value_or_interpretation=(
            "No reproducible FID/SFD result is available."
        ),
        manuscript_action=(
            "REMOVE absolute FID claims and dependent comparisons."
        ),
        required_wording_or_constraint=(
            "Do not call a structured-feature Fréchet distance "
            "conventional FID without validated feature-space provenance."
        ),
        source_stage="04",
        source_files=relative(
            r04["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE",
    )

    # -------------------------------------------------------------------------
    # 04B
    # -------------------------------------------------------------------------
    r04b = resolved["04B_v2"]

    add_record(
        records,
        evidence_id="E04B-01",
        reviewer_comments="C6,C7,C8,C9,C10,C11,C19,C20,C26,C28,C30,C31",
        topic="Recovered HFAGM generative implementation",
        manuscript_claim_or_issue=(
            "The manuscript describes a GAN/VAE/diffusion hybrid "
            "generator, adaptive fairness controller, and synthetic "
            "generation pipeline."
        ),
        reproducibility_status=(
            "NOT_REPRODUCIBLE"
            if r04b["resolved"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "IMPLEMENTATION_MISSING"
            if r04b["resolved"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            "Recovered implementation supports classifier/encoder/graph "
            "processing rather than a verified structured GAN/VAE/"
            "diffusion generator."
        ),
        invalid_or_unsupported_evidence=(
            "Textual equations, diagrams, BCE losses, empty generator "
            "directories, or class names do not establish executable "
            "structured synthetic generation."
        ),
        final_value_or_interpretation=(
            "Historical structured generator claims are not substantiated "
            "by recovered code/artifacts."
        ),
        manuscript_action=(
            "REMOVE unsupported generator implementation/results claims "
            "or explicitly separate them as conceptual design."
        ),
        required_wording_or_constraint=(
            "Any future generator implementation must be labeled as "
            "a new corrected experiment, not a reconstruction."
        ),
        source_stage="04B_v2",
        source_files=relative(
            r04b["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE_OR_MAJOR_REFRAME",
    )

    # -------------------------------------------------------------------------
    # 05
    # -------------------------------------------------------------------------
    r05 = resolved["05"]

    add_record(
        records,
        evidence_id="E05-01",
        reviewer_comments="C21,C22,C27",
        topic="Common untouched real-test downstream utility",
        manuscript_claim_or_issue=(
            "Synthetic-vs-real downstream utility requires the same "
            "untouched real test set and a provenance-valid synthetic "
            "training condition."
        ),
        reproducibility_status=(
            "PARTIALLY_REPRODUCIBLE"
            if r05["resolved"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "REAL_REFERENCE_ONLY"
            if r05["resolved"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            "A leakage-safe real-data reference condition was recoverable."
        ),
        invalid_or_unsupported_evidence=(
            "No provenance-valid labeled 51-feature synthetic dataset "
            "was recovered for common-test comparison."
        ),
        final_value_or_interpretation=(
            "Retain real-data reference only; do not claim validated "
            "synthetic utility superiority."
        ),
        manuscript_action=(
            "REMOVE historical synthetic-vs-real utility superiority claims."
        ),
        required_wording_or_constraint=(
            "Do not use the single locked-test perfect score as "
            "a headline superiority result."
        ),
        source_stage="05",
        source_files=relative(
            r05["path"]
        ),
        severity="CRITICAL",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # 06
    # -------------------------------------------------------------------------
    r06 = resolved["06"]

    add_record(
        records,
        evidence_id="E06-01",
        reviewer_comments="C10,C23,C25",
        topic="Generative scalability and computational efficiency",
        manuscript_claim_or_issue=(
            "N=500→100000, ~25% faster, 20–30% lower GPU memory, "
            "and +40% memory claims require matched generator-linked evidence."
        ),
        reproducibility_status=(
            "NOT_REPRODUCIBLE"
            if r06["resolved"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "UNSUPPORTED"
            if r06["resolved"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            "No verified generator-linked runtime/GPU-memory benchmark "
            "was recovered."
        ),
        invalid_or_unsupported_evidence=(
            "Synthetic-scale/runtime/memory percentages are unsupported; "
            "classifier/preprocessing timing cannot substitute."
        ),
        final_value_or_interpretation=(
            "Synthetic N is computational workload, not independent "
            "clinical information."
        ),
        manuscript_action=(
            "REMOVE unsupported scalability/runtime/memory claims."
        ),
        required_wording_or_constraint=(
            "Efficiency may be discussed only as an unverified design "
            "objective or future evaluation requirement."
        ),
        source_stage="06",
        source_files=relative(
            r06["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE",
    )

    # -------------------------------------------------------------------------
    # 07
    # -------------------------------------------------------------------------
    r07 = resolved["07"]

    add_record(
        records,
        evidence_id="E07-01",
        reviewer_comments="C11,C28",
        topic="Generator/fusion/fairness/scalability ablations",
        manuscript_claim_or_issue=(
            "Requested full-vs-component, fusion, fairness, and "
            "scalability ablations require actual historical variants."
        ),
        reproducibility_status=(
            "NOT_REPRODUCIBLE"
            if r07["resolved"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "MISSING_EXECUTED_VARIANTS"
            if r07["resolved"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            "Required executed historical ablation variants and "
            "result provenance were not recovered."
        ),
        invalid_or_unsupported_evidence=(
            "Generic keywords or classifier hyperparameters cannot "
            "be relabeled as generator ablations."
        ),
        final_value_or_interpretation=(
            "Historical ablation study cannot be reconstructed."
        ),
        manuscript_action=(
            "REMOVE unsupported ablation claims and plots."
        ),
        required_wording_or_constraint=(
            "Any later ablations must be labeled as new corrected experiments."
        ),
        source_stage="07",
        source_files=relative(
            r07["path"]
        ),
        severity="CRITICAL",
        final_disposition="REMOVE",
    )

    # -------------------------------------------------------------------------
    # 08
    # -------------------------------------------------------------------------
    r08 = resolved["08_v2"]

    add_record(
        records,
        evidence_id="E08-01",
        reviewer_comments="C29",
        topic="Statistical significance and uncertainty",
        manuscript_claim_or_issue=(
            "Formal significance claims require an appropriate inferential design."
        ),
        reproducibility_status=(
            "AUDITED"
            if r08["resolved"]
            else "SOURCE_MISSING"
        ),
        evidence_status=(
            "DESCRIPTIVE_ONLY"
            if r08["resolved"]
            else "MANUAL_CHECK_REQUIRED"
        ),
        validated_evidence=(
            "Ten leakage-safe repeated holdouts support descriptive "
            "mean, SD, median, minimum, and maximum."
        ),
        invalid_or_unsupported_evidence=(
            "Repeated holdouts reuse the same 193 participants and "
            "do not provide independent experimental replicates."
        ),
        final_value_or_interpretation=(
            "Treat repeated holdouts as stability analyses."
        ),
        manuscript_action=(
            "REMOVE unsupported 'significantly improved/outperformed' wording."
        ),
        required_wording_or_constraint=(
            "Report descriptive statistics only unless a separate valid "
            "inferential design is available."
        ),
        source_stage="08_v2",
        source_files=relative(
            r08["path"]
        ),
        severity="MAJOR",
        final_disposition="REWRITE",
    )

    # -------------------------------------------------------------------------
    # 09
    # -------------------------------------------------------------------------
    add_record(
        records,
        evidence_id="E09-01",
        reviewer_comments="C13,C14",
        topic="Multimodal clinical scope",
        manuscript_claim_or_issue=(
            "The manuscript claims multimodal clinical evaluation and "
            "cross-domain/multimodal generalization."
        ),
        reproducibility_status="NOT_REPRODUCIBLE",
        evidence_status="UNSUPPORTED_SCOPE",
        validated_evidence=(
            "Stage 09 found 27 multimodal claims, 0 strict executable "
            "clinical multimodal+fusion candidates, 0 clinical image "
            "artifacts, and a large unrelated sign-language image repository."
        ),
        invalid_or_unsupported_evidence=(
            "Unrelated ArSL/sign-language assets do not establish "
            "multimodal clinical HFAGM evaluation."
        ),
        final_value_or_interpretation=(
            "The reproducible clinical study is structured/tabular."
        ),
        manuscript_action=(
            "REMOVE or reframe multimodal clinical claims."
        ),
        required_wording_or_constraint=(
            "Do not claim cross-modal fairness/utility/generalization."
        ),
        source_stage="09_v2",
        source_files=relative(
            STAGE09_SUMMARY
        ),
        severity="CRITICAL",
        final_disposition="REMOVE_OR_MAJOR_REFRAME",
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
            f"Stage 09 detected {len(stage09['duplicates'])} duplicate "
            "figure-number groups and no exact/perceptual duplicate image pairs."
        ),
        invalid_or_unsupported_evidence=(
            "Unsupported generative/multimodal figures cannot be fixed "
            "by renumbering alone."
        ),
        final_value_or_interpretation=(
            "Remove/reframe unsupported figures first, then renumber."
        ),
        manuscript_action=(
            "REBUILD AND RENUMBER figures after evidence-constrained pruning."
        ),
        required_wording_or_constraint=(
            "Every retained figure must map to a reproducible result/source."
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
            "Confusion-matrix accuracy must match the same evaluation condition."
        ),
        reproducibility_status="PARTIAL",
        evidence_status="MANUAL_SOURCE_LINK_REQUIRED",
        validated_evidence=(
            "Stage 09 context-aware parsing found no direct accuracy values "
            "and kept AUC 0.95/0.99 separate from accuracy."
        ),
        invalid_or_unsupported_evidence=(
            "No complete textual TP/TN/FP/FN set was recovered."
        ),
        final_value_or_interpretation=(
            "Recompute each retained confusion-matrix accuracy as "
            "(TP+TN)/(TP+TN+FP+FN) from source counts."
        ),
        manuscript_action=(
            "RETAIN only source-linked confusion matrices."
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
            "Synthetic row count must not be interpreted as "
            "additional independent patients."
        ),
        reproducibility_status="AUDITED",
        evidence_status="INTERPRETATION_CONSTRAINT",
        validated_evidence=(
            "Stage 09 found no directly parsed large synthetic-N statement "
            "in the selected manuscript."
        ),
        invalid_or_unsupported_evidence=(
            "Historical N=500→100000 scale claims remain unsupported."
        ),
        final_value_or_interpretation=(
            f"Independent clinical information remains limited to "
            f"{EXPECTED_PARTICIPANTS} participants."
        ),
        manuscript_action=(
            "REMOVE unsupported historical synthetic-scale statements."
        ),
        required_wording_or_constraint=(
            "Synthetic rows, if discussed, are computational workload only."
        ),
        source_stage="06,09_v2",
        source_files="; ".join(
            filter(
                None,
                [
                    relative(
                        r06["path"]
                    ),
                    relative(
                        STAGE09_SUMMARY
                    ),
                ],
            )
        ),
        severity="MAJOR",
        final_disposition="REWRITE_OR_REMOVE",
    )

    # -------------------------------------------------------------------------
    # Structural synthesis
    # -------------------------------------------------------------------------
    add_record(
        records,
        evidence_id="ESTR-01",
        reviewer_comments="C6,C7,C8,C9,C10,C11,C12,C26,C31",
        topic="Mathematical HGF/HFAGM framework",
        manuscript_claim_or_issue=(
            "Current equations imply an executable hybrid generative "
            "optimizer not recovered in code."
        ),
        reproducibility_status="NOT_SUPPORTED_AS_IMPLEMENTED",
        evidence_status="THEORETICAL_ONLY_AT_BEST",
        validated_evidence=(
            "A conceptual framework may be discussed only if clearly "
            "separated from empirically verified components."
        ),
        invalid_or_unsupported_evidence=(
            "Non-differentiable SPD/EOD or wall-clock terms cannot be "
            "presented as direct gradient objectives without a verified surrogate."
        ),
        final_value_or_interpretation=(
            "Separate conceptual design from executable recovered implementation."
        ),
        manuscript_action=(
            "MAJOR REWRITE Methods; remove unsupported implementation mechanics."
        ),
        required_wording_or_constraint=(
            "Do not present unverified update order, gradient interaction, "
            "or controller equations as executed training procedures."
        ),
        source_stage="04B_v2,06,07",
        source_files="; ".join(
            filter(
                None,
                [
                    relative(r04b["path"]),
                    relative(r06["path"]),
                    relative(r07["path"]),
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
            "Pareto-optimal/frontier claims require explicit frontier "
            "and convergence evidence."
        ),
        reproducibility_status="NOT_ESTABLISHED",
        evidence_status="UNSUPPORTED",
        validated_evidence=(
            "No verified Pareto-front/convergence artifact was recovered."
        ),
        invalid_or_unsupported_evidence=(
            "Multi-objective language does not establish Pareto optimality."
        ),
        final_value_or_interpretation=(
            "Discuss trade-offs only."
        ),
        manuscript_action="REMOVE Pareto-optimal wording.",
        required_wording_or_constraint=(
            "Use trade-off/multi-objective design motivation instead."
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
            "PR-GM/MSS are undefined or unreported in validated outputs."
        ),
        reproducibility_status="NOT_ESTABLISHED",
        evidence_status="UNSUPPORTED_UNLESS_SOURCE_FOUND",
        validated_evidence=(
            "No validated PR-GM/MSS result is part of the controlling "
            "evidence set."
        ),
        invalid_or_unsupported_evidence=(
            "Undefined metrics cannot support fidelity/utility claims."
        ),
        final_value_or_interpretation=(
            "Define and reproduce from provenance-valid source or remove."
        ),
        manuscript_action=(
            "REMOVE by default unless source evidence is recovered."
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
            "Conflicting hardware descriptions exist."
        ),
        reproducibility_status="UNRESOLVED",
        evidence_status="MANUAL_PROVENANCE_REQUIRED",
        validated_evidence=(
            "Hardware must be tied to the exact retained experiment."
        ),
        invalid_or_unsupported_evidence=(
            "Unrelated hardware evidence must not be imported."
        ),
        final_value_or_interpretation=(
            "Retain only hardware supported by exact logs/code."
        ),
        manuscript_action=(
            "VERIFY hardware before final Methods."
        ),
        required_wording_or_constraint=(
            "Do not use unrelated RTX 3080 evidence to resolve "
            "the manuscript's hardware conflict."
        ),
        source_stage="06 + master synthesis",
        source_files=relative(
            r06["path"]
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
            "Model comparisons require comparable tuning/search budgets."
        ),
        reproducibility_status="PARTIAL",
        evidence_status="REWRITE_REQUIRED",
        validated_evidence=(
            "Corrected classifier evaluation may be retained where "
            "pipeline and split are reproducible."
        ),
        invalid_or_unsupported_evidence=(
            "Historical superiority claims across differently tuned "
            "models are not controlled architecture comparisons."
        ),
        final_value_or_interpretation=(
            "Describe baseline differences descriptively unless tuning "
            "parity is documented."
        ),
        manuscript_action=(
            "REMOVE architecture-causal superiority claims when search "
            "budgets differ or are unknown."
        ),
        required_wording_or_constraint=(
            "Report known hyperparameters and acknowledge tuning-budget limits."
        ),
        source_stage="02E,05",
        source_files="; ".join(
            filter(
                None,
                [
                    relative(
                        REPEATED_METRICS_PATH
                    ),
                    relative(
                        r05["path"]
                    ),
                ],
            )
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
            "Synthetic data are described as privacy-preserving "
            "without verified privacy evaluation."
        ),
        reproducibility_status="NOT_ESTABLISHED",
        evidence_status="UNSUPPORTED_AS_RESULT",
        validated_evidence=(
            "Privacy may remain as motivation/background only."
        ),
        invalid_or_unsupported_evidence=(
            "No membership-inference, attribute-inference, "
            "differential-privacy, or equivalent experiment was recovered."
        ),
        final_value_or_interpretation=(
            "Synthetic generation does not automatically establish privacy."
        ),
        manuscript_action=(
            "REMOVE empirical privacy guarantees; retain cautious motivation."
        ),
        required_wording_or_constraint=(
            "State explicitly that privacy was not empirically evaluated."
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
            "Current novelty depends on unsupported generator, multimodal, "
            "fairness-generation, ablation, and scalability results."
        ),
        reproducibility_status="NARROWER_SCOPE_REQUIRED",
        evidence_status="REFRAME",
        validated_evidence=(
            "Strongest reproducible evidence is corrected leakage-safe "
            "structured clinical classification with fairness/stability analysis."
        ),
        invalid_or_unsupported_evidence=(
            "Broad validated hybrid-generative framework claims are not "
            "supported by recovered artifacts."
        ),
        final_value_or_interpretation=(
            "Novelty must be narrowed to reproducible evidence."
        ),
        manuscript_action=(
            "REWRITE title, abstract, contributions, Methods, Results, "
            "Discussion, and Conclusion."
        ),
        required_wording_or_constraint=(
            "Do not claim validated multimodal generalization, generator "
            "fairness, synthetic superiority, or scalability."
        ),
        source_stage="04B_v2,05,06,07,08_v2,09_v2",
        source_files="; ".join(
            filter(
                None,
                [
                    relative(r04b["path"]),
                    relative(r05["path"]),
                    relative(r06["path"]),
                    relative(r07["path"]),
                    relative(r08["path"]),
                    relative(STAGE09_SUMMARY),
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
            "Reference numbering/DOI relevance requires independent "
            "bibliographic verification."
        ),
        reproducibility_status="NOT_AUDITED_BY_STAGE10_V2",
        evidence_status="PENDING_REFERENCE_AUDIT",
        validated_evidence=(
            "Stage 10 V2 does not invent bibliographic corrections."
        ),
        invalid_or_unsupported_evidence=(
            "Potential citation mismatch remains possible."
        ),
        final_value_or_interpretation=(
            "Reference audit remains a separate finalization task."
        ),
        manuscript_action=(
            "VERIFY DOI/title/journal/authors/year/relevance for every "
            "retained reference."
        ),
        required_wording_or_constraint=(
            "Do not retain references solely because they appear in the "
            "current bibliography."
        ),
        source_stage="Pending final reference audit",
        source_files="",
        severity="MAJOR",
        final_disposition="PENDING",
    )

    return records, diagnostics


# =============================================================================
# 12. RESTRUCTURING PLAN
# =============================================================================

def build_restructuring_plan() -> pd.DataFrame:
    rows = [
        (1, "Title", "REWRITE",
         "Narrow title to reproducible structured-clinical scope; do not foreground scalability, multimodal validation, or a verified hybrid generator.",
         "E04B-01,E06-01,E09-01,ESTR-07"),

        (2, "Abstract", "REWRITE",
         "Remove unsupported FID, generator superiority, multimodal generalization, privacy guarantee, scalability, ablation, and significance claims; report only corrected reproducible classifier/fairness stability evidence.",
         "E02E-01,E03-01,E04-01,E04B-01,E05-01,E06-01,E07-01,E08-01,E09-01,ESTR-06"),

        (3, "Introduction", "REWRITE",
         "Retain synthetic-data/fairness motivation cautiously, but distinguish motivation from demonstrated properties.",
         "E04B-01,ESTR-06,ESTR-07"),

        (4, "Contributions", "REWRITE",
         "List only contributions directly supported by retained evidence.",
         "E02E-01,E03-01,E08-01,ESTR-07"),

        (5, "Methods - Framework", "MAJOR_REWRITE",
         "Separate conceptual framework from executable recovered implementation. Remove claims that non-recovered GAN/VAE/diffusion/fairness/scalability modules were executed.",
         "E04B-01,ESTR-01,ESTR-02"),

        (6, "Methods - Dataset", "RETAIN_AND_CLARIFY",
         f"Describe structured clinical dataset and original cohort size ({EXPECTED_PARTICIPANTS}) exactly; do not call the clinical dataset multimodal.",
         "E09-01,E09-04"),

        (7, "Methods - Preprocessing/Splits", "REWRITE",
         "Use split-first leakage-safe preprocessing and clearly distinguish corrected evaluation from invalid historical workflow.",
         "E02E-01,E02F-01,E05-01"),

        (8, "Methods - Fairness", "REWRITE",
         "Define favorable outcome, sensitive attribute, reference group, subgroup counts, SPD/EOD/DI formulas, and classifier scope.",
         "E03-01"),

        (9, "Methods - Statistics", "REWRITE",
         "Report repeated holdouts descriptively; no formal p-values or independent-replicate significance claims.",
         "E08-01"),

        (10, "Methods - Hardware", "VERIFY",
         "Retain only hardware tied to retained experiments.",
         "ESTR-04"),

        (11, "Results - Primary performance", "REBUILD",
         "Use corrected leakage-safe repeated-holdout performance; do not headline contaminated perfect scores.",
         "E02E-01,E05-01"),

        (12, "Results - Fairness", "REBUILD",
         "Use corrected classifier fairness metrics with descriptive interpretation.",
         "E03-01,E08-01"),

        (13, "Results - Fidelity/FID", "REMOVE",
         "Remove unsupported FID/SFD values and generator-fidelity claims.",
         "E04-01,E04B-01"),

        (14, "Results - Synthetic utility", "REMOVE_OR_REFRAME",
         "Remove synthetic-vs-real superiority claims; retain only reproducible real reference if needed.",
         "E05-01"),

        (15, "Results - Scalability", "REMOVE",
         "Remove unsupported synthetic-N/runtime/memory generator-scale claims.",
         "E06-01"),

        (16, "Results - Ablations", "REMOVE",
         "Remove unreproducible generator/fusion/fairness/scalability ablation claims and plots.",
         "E07-01"),

        (17, "Results - Multimodal/Cross-domain", "REMOVE",
         "Remove unsupported ArSL→clinical/multimodal claims from the clinical evidence narrative.",
         "E09-01"),

        (18, "Figures", "REBUILD_AND_RENUMBER",
         "Remove unsupported figures first, then renumber retained figures sequentially and verify all references.",
         "E09-02,E09-03"),

        (19, "Discussion", "MAJOR_REWRITE",
         "Discuss verified classifier/fairness stability, feature-timing limitation, missing generator provenance, absent privacy evaluation, and limited generalizability.",
         "E02F-01,E03-01,E04B-01,E06-01,E07-01,E08-01,E09-01,ESTR-06,ESTR-07"),

        (20, "Limitations", "EXPAND",
         "State cohort size, repeated-holdout dependence, chronology uncertainty, missing generator provenance, absence of validated multimodal/privacy claims, and non-reproducible ablations/scalability.",
         "E02F-01,E04B-01,E06-01,E07-01,E08-01,E09-01,ESTR-06"),

        (21, "Conclusion", "REWRITE",
         "Conclude only what is supported by leakage-safe structured-clinical evaluation.",
         "E02E-01,E03-01,E08-01,ESTR-07"),

        (22, "References", "AUDIT",
         "Verify every retained reference and DOI independently.",
         "ESTR-08"),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "order",
            "section",
            "action",
            "target",
            "evidence_ids",
        ],
    )


# =============================================================================
# 13. COMMENT DISPOSITION
# =============================================================================

def build_comment_disposition(
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    for comment_id, topic in (
        REVIEWER_MAP.items()
    ):
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
                    "stage10_v2_status": "PENDING_DIRECT_MANUSCRIPT_EDIT",
                    "recommended_disposition": "MANUAL_REVIEW",
                    "basis": (
                        "No dedicated master-evidence row mapped automatically."
                    ),
                }
            )
            continue

        rows.append(
            {
                "reviewer_comment": comment_id,
                "reviewer_topic": topic,
                "evidence_ids": ",".join(
                    matches[
                        "evidence_id"
                    ].astype(str)
                ),
                "stage10_v2_status": "EVIDENCE_SYNTHESIZED",
                "recommended_disposition": ";".join(
                    sorted(
                        set(
                            matches[
                                "final_disposition"
                            ].astype(str)
                        )
                    )
                ),
                "basis": ";".join(
                    sorted(
                        set(
                            matches[
                                "evidence_status"
                            ].astype(str)
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 14. CLAIM POLICY
# =============================================================================

def build_claim_policy() -> pd.DataFrame:
    rows = [
        ("ALLOW",
         f"The study uses a structured clinical dataset with {EXPECTED_PARTICIPANTS} participants.",
         "Retain only if source dataset documentation supports the count."),

        ("ALLOW",
         "Repeated leakage-safe holdouts provide descriptive stability evidence.",
         "Report mean/SD/median/range; no independent-replicate significance claim."),

        ("ALLOW",
         "Corrected SPD/EOD/DI quantify fairness of the real-outcome classifier.",
         "State attribute, reference group, favorable outcome, subgroup counts, and classifier scope."),

        ("ALLOW_WITH_LIMITATION",
         "No direct temporal/outcome proxy leakage was identified.",
         "Also state that feature timing remains incompletely documented."),

        ("DO_NOT_CLAIM",
         "A verified GAN/VAE/diffusion hybrid structured-data generator was implemented and evaluated.",
         "Recovered project does not support this."),

        ("DO_NOT_CLAIM",
         "Validated conventional FID = 1.5/1.6/3.2.",
         "No provenance-valid generator/feature-space evidence."),

        ("DO_NOT_CLAIM",
         "HFAGM is proven multimodal or cross-domain on clinical + ArSL data.",
         "No strict executable clinical multimodal+fusion evidence."),

        ("DO_NOT_CLAIM",
         "Synthetic data preserve/improve fairness based on corrected Stage 03 metrics.",
         "Stage 03 fairness applies to the real-outcome classifier."),

        ("DO_NOT_CLAIM",
         "Synthetic utility exceeds real-data utility.",
         "No provenance-valid labeled synthetic dataset for common-test evaluation."),

        ("DO_NOT_CLAIM",
         "Generator is ~25% faster or uses 20–30% less GPU memory.",
         "No matched generator-linked benchmark."),

        ("DO_NOT_CLAIM",
         "N=100000 represents more independent clinical information than the original cohort.",
         "Synthetic rows are computational workload only."),

        ("DO_NOT_CLAIM",
         "Pareto-optimal solution/frontier.",
         "No explicit frontier/convergence evidence."),

        ("DO_NOT_CLAIM",
         "Significantly improved/outperformed based only on the ten repeated holdouts.",
         "Same cohort reused; descriptive stability only."),

        ("DO_NOT_CLAIM",
         "Synthetic-data privacy protection was empirically proven.",
         "No validated privacy experiment recovered."),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "claim_class",
            "claim",
            "condition",
        ],
    )


# =============================================================================
# 15. STAGE CLOSURE CHECK
# =============================================================================

def build_closure_matrix(
    resolved: Dict[str, Dict[str, Any]],
    repeated: Dict[str, Any],
    fairness: Dict[str, Any],
    stage09: Dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "criterion": "02E repeated leakage-safe metrics resolved",
            "passed": int(
                repeated["available"]
            ),
            "detail": str(
                REPEATED_METRICS_PATH
            ),
        },
        {
            "criterion": "02F chronology audit resolved by verdict",
            "passed": int(
                resolved[
                    "02F_v2"
                ]["resolved"]
            ),
            "detail": str(
                resolved[
                    "02F_v2"
                ]["path"]
                or ""
            ),
        },
        {
            "criterion": "03 fairness evidence parsed",
            "passed": int(
                fairness["available"]
                and bool(
                    fairness[
                        "summary"
                    ]
                    or fairness[
                        "raw_rows"
                    ]
                )
            ),
            "detail": str(
                fairness[
                    "selected_path"
                ]
                or ""
            ),
        },
        {
            "criterion": "04 FID/SFD audit resolved by verdict",
            "passed": int(
                resolved["04"][
                    "resolved"
                ]
            ),
            "detail": str(
                resolved["04"][
                    "path"
                ]
                or ""
            ),
        },
        {
            "criterion": "04B generator audit resolved by verdict",
            "passed": int(
                resolved[
                    "04B_v2"
                ]["resolved"]
            ),
            "detail": str(
                resolved[
                    "04B_v2"
                ]["path"]
                or ""
            ),
        },
        {
            "criterion": "05 utility audit resolved by verdict",
            "passed": int(
                resolved["05"][
                    "resolved"
                ]
            ),
            "detail": str(
                resolved["05"][
                    "path"
                ]
                or ""
            ),
        },
        {
            "criterion": "06 scalability audit resolved by verdict",
            "passed": int(
                resolved["06"][
                    "resolved"
                ]
            ),
            "detail": str(
                resolved["06"][
                    "path"
                ]
                or ""
            ),
        },
        {
            "criterion": "07 ablation audit resolved by verdict",
            "passed": int(
                resolved["07"][
                    "resolved"
                ]
            ),
            "detail": str(
                resolved["07"][
                    "path"
                ]
                or ""
            ),
        },
        {
            "criterion": "08 statistics audit resolved by verdict",
            "passed": int(
                resolved[
                    "08_v2"
                ]["resolved"]
            ),
            "detail": str(
                resolved[
                    "08_v2"
                ]["path"]
                or ""
            ),
        },
        {
            "criterion": "09 multimodal/figure audit available",
            "passed": int(
                bool(
                    stage09[
                        "summary_text"
                    ]
                )
            ),
            "detail": str(
                STAGE09_SUMMARY
            ),
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# 16. SUMMARY WRITER
# =============================================================================

def write_summary(
    evidence_df: pd.DataFrame,
    restructure_df: pd.DataFrame,
    comment_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    closure_df: pd.DataFrame,
    resolved: Dict[str, Dict[str, Any]],
    repeated: Dict[str, Any],
    fairness: Dict[str, Any],
) -> None:
    all_pass = bool(
        closure_df[
            "passed"
        ].eq(1).all()
    )

    critical_count = int(
        (
            evidence_df[
                "severity"
            ]
            == "CRITICAL"
        ).sum()
    )

    removal_mask = evidence_df[
        "final_disposition"
    ].isin(
        [
            "REMOVE",
            "REMOVE_OR_MAJOR_REFRAME",
            "REMOVE_UNLESS_PROVEN",
        ]
    )

    rewrite_mask = evidence_df[
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

    if all_pass:
        final_verdict = (
            "MASTER_EVIDENCE_PROVENANCE_RESOLVED_"
            "READY_FOR_SECTION_BY_SECTION_REWRITE"
        )

        next_action = (
            "BEGIN_MANUSCRIPT_REWRITE_WITH_TITLE_THEN_ABSTRACT"
        )
    else:
        final_verdict = (
            "MASTER_EVIDENCE_PROVENANCE_STILL_INCOMPLETE"
        )

        next_action = (
            "RESOLVE_FAILED_CLOSURE_ITEMS_BEFORE_REWRITE"
        )

    lines = [
        "=" * 112,
        "HFAGM - STAGE 10 V2 MASTER EVIDENCE AND MANUSCRIPT RESTRUCTURING AUDIT",
        "=" * 112,
        "",
        f"Generated: {GENERATED_AT}",
        f"Project root: {PROJECT_ROOT}",
        f"Manuscript target: {MANUSCRIPT_PATH}",
        "",
        "V2 CORRECTIONS",
        "-" * 112,
        "1. Prior audit files are resolved by unique verdict/signature strings across the outputs tree rather than brittle hard-coded filenames.",
        "2. Stage 02F, 04, 04B, 05, and 06 provenance are explicitly resolved and recorded.",
        "3. Stage 03 fairness prefers the per-seed file and aggregates it; if unavailable, the summary file is preserved without inventing values.",
        "4. A formal closure matrix determines whether Stage 10 can be closed.",
        "",
        "COUNTS",
        "-" * 112,
        f"Master evidence records: {len(evidence_df)}",
        f"Critical records: {critical_count}",
        f"Remove/remove-or-majorly-reframe records: {int(removal_mask.sum())}",
        f"Rewrite/verify/correct records: {int(rewrite_mask.sum())}",
        f"Restructuring actions: {len(restructure_df)}",
        f"Reviewer comments mapped: {len(comment_df)}",
        f"Claim-policy rows: {len(policy_df)}",
        "",
        "SOURCE RESOLUTION",
        "-" * 112,
    ]

    for stage_key in (
        "02F_v2",
        "04",
        "04B_v2",
        "05",
        "06",
        "07",
        "08_v2",
        "09_v2",
    ):
        info = resolved[stage_key]

        lines.append(
            f"{stage_key}: "
            f"{'RESOLVED' if info['resolved'] else 'MISSING'} | "
            f"path={info['path']} | "
            f"candidates={info['candidate_count']} | "
            f"ambiguity={info['ambiguity']}"
        )

    lines.extend(
        [
            "",
            "FAIRNESS SOURCE",
            "-" * 112,
            (
                f"selected={fairness['selected_path']} | "
                f"source_kind={fairness['source_kind']} | "
                f"rows={fairness['rows']} | "
                f"metric_columns={safe_json(fairness['metric_columns'])} | "
                f"group_columns={safe_json(fairness['group_columns'])}"
            ),
            "",
            "CLOSURE MATRIX",
            "-" * 112,
        ]
    )

    for _, row in (
        closure_df.iterrows()
    ):
        lines.append(
            f"{'PASS' if int(row['passed']) == 1 else 'FAIL'}: "
            f"{row['criterion']}"
        )
        lines.append(
            f"    {row['detail']}"
        )

    lines.extend(
        [
            "",
            "CONTROLLING CONCLUSIONS",
            "-" * 112,
            "1. The recovered project does not substantiate a verified structured GAN/VAE/diffusion hybrid generator.",
            "2. Historical FID/SFD, generator scalability, synthetic utility, generator ablations, multimodal clinical generalization, and generator-fairness claims are not reproducible from recovered artifacts.",
            "3. The strongest reproducible empirical evidence is the corrected leakage-safe structured-clinical classifier evaluation and descriptive fairness/stability analysis.",
            "4. Repeated holdouts reuse the same participant cohort and therefore support descriptive stability, not independent-replicate significance.",
            "5. Unsupported figures/results must be removed before final renumbering.",
            "6. Any future generator/ablation/privacy/scalability experiment must be labeled as new corrected work, not historical reconstruction.",
            "",
            "FINAL STAGE-10 V2 VERDICT",
            "-" * 112,
            final_verdict,
            "",
            "NEXT ACTION",
            "-" * 112,
            next_action,
            "",
            "RECOMMENDED MANUSCRIPT ORDER",
            "-" * 112,
        ]
    )

    for _, row in (
        restructure_df.sort_values(
            "order"
        ).iterrows()
    ):
        lines.append(
            f"{int(row['order']):02d}. "
            f"{row['section']} | "
            f"{row['action']} | "
            f"{row['target']}"
        )

    lines.extend(
        [
            "",
            "SAFETY",
            "-" * 112,
            "New model training: NO",
            "Synthetic generation: NO",
            "New ablation creation: NO",
            "OCR: NO",
            "Image modification: NO",
            "Historical output modification: NO",
            "Manuscript modification: NO",
            "",
            "=" * 112,
        ]
    )

    (
        OUTPUT_DIR
        / "master_evidence_and_restructuring_audit_v2_summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 17. MAIN
# =============================================================================

def main() -> None:
    print("=" * 112)
    print(
        "HFAGM - STAGE 10 V2 MASTER EVIDENCE AND "
        "MANUSCRIPT RESTRUCTURING AUDIT"
    )
    print("=" * 112)

    print()
    print("Restrictions:")
    print("  - consolidate existing evidence only")
    print("  - resolve prior audits by verdict/signature")
    print("  - no model training")
    print("  - no synthetic generation")
    print("  - no new ablation creation")
    print("  - no OCR")
    print("  - no historical output modification")
    print("  - no manuscript modification")

    files = candidate_text_files()

    resolved: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for stage_key, signatures in (
        STAGE_SIGNATURES.items()
    ):
        resolved[
            stage_key
        ] = resolve_stage_by_signatures(
            stage_key,
            signatures,
            files,
        )

    records, diagnostics = (
        build_master_records(
            resolved
        )
    )

    evidence_df = pd.DataFrame(
        [
            asdict(record)
            for record in records
        ]
    )

    restructure_df = (
        build_restructuring_plan()
    )

    comment_df = (
        build_comment_disposition(
            evidence_df
        )
    )

    policy_df = build_claim_policy()

    repeated = summarize_repeated_metrics(
        REPEATED_METRICS_PATH
    )

    fairness = summarize_fairness_v2()

    stage09 = load_stage09()

    closure_df = build_closure_matrix(
        resolved,
        repeated,
        fairness,
        stage09,
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------
    evidence_df.to_csv(
        OUTPUT_DIR
        / "master_evidence_table_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    restructure_df.to_csv(
        OUTPUT_DIR
        / "manuscript_restructuring_plan_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comment_df.to_csv(
        OUTPUT_DIR
        / "reviewer_comment_disposition_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    policy_df.to_csv(
        OUTPUT_DIR
        / "final_claim_policy_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    closure_df.to_csv(
        OUTPUT_DIR
        / "stage10_v2_closure_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
        / "reproducibility_matrix_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
        / "claims_to_remove_or_majorly_reframe_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    evidence_df[
        ~removal_mask
    ].to_csv(
        OUTPUT_DIR
        / "claims_to_retain_rewrite_or_verify_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Source-resolution table
    # -------------------------------------------------------------------------
    resolution_rows: List[
        Dict[str, Any]
    ] = []

    for stage_key, info in (
        resolved.items()
    ):
        resolution_rows.append(
            {
                "stage": stage_key,
                "resolved": int(
                    info["resolved"]
                ),
                "selected_path": (
                    str(info["path"])
                    if info["path"]
                    else ""
                ),
                "matched_terms": safe_json(
                    info[
                        "matched_terms"
                    ]
                ),
                "candidate_count": info[
                    "candidate_count"
                ],
                "ambiguity": int(
                    info["ambiguity"]
                ),
                "all_candidates": safe_json(
                    info[
                        "all_candidates"
                    ]
                ),
            }
        )

    pd.DataFrame(
        resolution_rows
    ).to_csv(
        OUTPUT_DIR
        / "source_resolution_table_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Fairness audit export
    # -------------------------------------------------------------------------
    pd.DataFrame(
        fairness[
            "raw_rows"
        ]
    ).to_csv(
        OUTPUT_DIR
        / "fairness_source_rows_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fairness_summary_rows: List[
        Dict[str, Any]
    ] = []

    for group, metrics in (
        fairness["summary"].items()
    ):
        if (
            isinstance(
                metrics,
                dict,
            )
            and all(
                isinstance(
                    value,
                    dict,
                )
                for value in metrics.values()
            )
        ):
            for metric, stats in (
                metrics.items()
            ):
                row = {
                    "group": group,
                    "metric": metric,
                }

                row.update(stats)

                fairness_summary_rows.append(
                    row
                )
        else:
            row = {
                "group": "all",
                "metric": group,
            }

            if isinstance(
                metrics,
                dict,
            ):
                row.update(metrics)

            fairness_summary_rows.append(
                row
            )

    pd.DataFrame(
        fairness_summary_rows
    ).to_csv(
        OUTPUT_DIR
        / "fairness_aggregated_summary_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    diagnostics[
        "closure_matrix"
    ] = closure_df.to_dict(
        orient="records"
    )

    diagnostics[
        "final_closure_pass"
    ] = bool(
        closure_df[
            "passed"
        ].eq(1).all()
    )

    (
        OUTPUT_DIR
        / "source_resolution_diagnostics_v2.json"
    ).write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    write_summary(
        evidence_df,
        restructure_df,
        comment_df,
        policy_df,
        closure_df,
        resolved,
        repeated,
        fairness,
    )

    all_pass = bool(
        closure_df[
            "passed"
        ].eq(1).all()
    )

    print()
    print("=" * 112)
    print("STAGE 10 V2 COMPLETE")
    print("=" * 112)

    print(
        f"Prior output text files scanned: "
        f"{len(files)}"
    )

    for stage_key in (
        "02F_v2",
        "04",
        "04B_v2",
        "05",
        "06",
        "07",
        "08_v2",
        "09_v2",
    ):
        info = resolved[
            stage_key
        ]

        print(
            f"{stage_key}: "
            f"{'RESOLVED' if info['resolved'] else 'MISSING'}"
            f" | candidates={info['candidate_count']}"
            f" | ambiguity={info['ambiguity']}"
        )

        if info["path"]:
            print(
                f"    {info['path']}"
            )

    print()
    print(
        f"Fairness source: "
        f"{fairness['selected_path']}"
    )

    print(
        f"Fairness source kind: "
        f"{fairness['source_kind']}"
    )

    print(
        f"Fairness rows: "
        f"{fairness['rows']}"
    )

    print(
        f"Fairness metric columns: "
        f"{fairness['metric_columns']}"
    )

    print()
    print("CLOSURE MATRIX:")

    for _, row in (
        closure_df.iterrows()
    ):
        print(
            f"  "
            f"{'PASS' if int(row['passed']) == 1 else 'FAIL'} "
            f"- {row['criterion']}"
        )

    print()

    if all_pass:
        print("FINAL VERDICT:")
        print(
            "MASTER_EVIDENCE_PROVENANCE_RESOLVED_"
            "READY_FOR_SECTION_BY_SECTION_REWRITE"
        )

        print()
        print("NEXT ACTION:")
        print(
            "BEGIN_MANUSCRIPT_REWRITE_WITH_TITLE_"
            "THEN_ABSTRACT"
        )
    else:
        print("FINAL VERDICT:")
        print(
            "MASTER_EVIDENCE_PROVENANCE_STILL_INCOMPLETE"
        )

        print()
        print("NEXT ACTION:")
        print(
            "RESOLVE_FAILED_CLOSURE_ITEMS_BEFORE_REWRITE"
        )

    print()
    print("Results written to:")
    print(OUTPUT_DIR)

    print()
    print("Upload these files first:")

    for filename in [
        "master_evidence_and_restructuring_audit_v2_summary.txt",
        "stage10_v2_closure_matrix.csv",
        "source_resolution_table_v2.csv",
        "master_evidence_table_v2.csv",
        "manuscript_restructuring_plan_v2.csv",
        "reviewer_comment_disposition_v2.csv",
        "final_claim_policy_v2.csv",
        "reproducibility_matrix_v2.csv",
        "claims_to_remove_or_majorly_reframe_v2.csv",
        "claims_to_retain_rewrite_or_verify_v2.csv",
        "fairness_aggregated_summary_v2.csv",
        "fairness_source_rows_v2.csv",
        "source_resolution_diagnostics_v2.json",
    ]:
        print(
            OUTPUT_DIR
            / filename
        )


# =============================================================================
# 18. SAFE EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 112)
        print("STAGE 10 V2 FAILED SAFELY")
        print("=" * 112)

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
