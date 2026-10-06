from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import pandas as pd

try:
    from PIL import Image
except Exception:
    Image = None


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Array\Paper4-Under-Processing\HFAGM_Project"
)

PAPER_ROOT = PROJECT_ROOT.parent

# Preferred exact filename. If this exact file is absent, the resolver below
# searches for a unique strong filename match but NEVER silently falls back to
# an arbitrary DOCX such as HFAGM.docx.
MANUSCRIPT_FILENAME = (
    "HFAGM.docx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_multimodal_figure_audit_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_PARTICIPANTS = 193

MAX_TEXT_BYTES = 25 * 1024 * 1024
MAX_SCAN_FILES = 100000

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".docx",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

TABULAR_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}

ARRAY_EXTENSIONS = {
    ".npy",
    ".npz",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    "site-packages",
    "dist-packages",
    "node_modules",
    ".idea",
    ".vscode",
    ".pytest_cache",
}

EXCLUDED_DIR_PREFIXES = (
    ".venv",
    "venv",
    ".env",
)

DOCX_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


# =============================================================================
# 2. REVIEWER-SPECIFIC TERMS
# =============================================================================

MULTIMODAL_TERMS = (
    "multimodal",
    "multi-modal",
    "multi modal",
    "multiple modalities",
    "modality",
    "modalities",
    "imaging",
    "thermal",
    "rgb",
    "visual modality",
    "audio modality",
    "text modality",
    "sensor modality",
    "cross-modal",
    "cross modal",
)

CLINICAL_TERMS = (
    "covid",
    "clinical",
    "patient",
    "patients",
    "vitamin d",
    "vitamin b12",
    "calcium",
    "phosphorus",
    "magnesium",
    "hemoglobin",
    "haemoglobin",
    "status",
    "mortality",
    "survivor",
    "recovered",
    "deceased",
)

UNRELATED_DATASET_TERMS = (
    "arsl",
    "karsl",
    "sign language",
    "arabic sign",
    "asl",
    "gesture",
    "gestures",
)

FUSION_TERMS = (
    "fusion",
    "fuse",
    "fused",
    "concatenate",
    "concatenation",
    "torch.cat",
    "np.concatenate",
    "hstack",
    "vstack",
    "late fusion",
    "early fusion",
    "attention fusion",
    "cross-modal",
    "cross modal",
)

GENERATOR_TERMS = (
    "gan",
    "generator",
    "discriminator",
    "vae",
    "variational autoencoder",
    "diffusion",
    "denoising",
    "synthetic generation",
    "synthetic data generation",
)


# =============================================================================
# 3. CAPTION / METRIC PATTERNS
# =============================================================================

FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.)\s*([0-9]+)\s*[:.\-]?\s*(.*)$",
    re.I,
)

TABLE_CAPTION_RE = re.compile(
    r"^\s*Table\s*([0-9]+)\s*[:.\-]?\s*(.*)$",
    re.I,
)

NUMBER_RE = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*%?)"

# IMPORTANT:
# Values are only captured when they directly follow their metric label.
# Therefore "accuracy (AUC ≈ 0.99)" will NOT assign 0.99 to accuracy.
DIRECT_METRIC_PATTERNS = {
    "accuracy": re.compile(
        rf"\baccuracy\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "precision": re.compile(
        rf"\bprecision\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "recall": re.compile(
        rf"\b(?:recall|sensitivity)\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "specificity": re.compile(
        rf"\bspecificity\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "f1": re.compile(
        rf"\b(?:f1(?:[- ]?score)?)\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "auc": re.compile(
        rf"\b(?:roc[- ]?auc|auc)\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "fid": re.compile(
        rf"\bfid\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "spd": re.compile(
        rf"\bspd\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "eod": re.compile(
        rf"\beod\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
    "di": re.compile(
        rf"\b(?:di|disparate impact)\b\s*(?:[:=≈~]|of)?\s*{NUMBER_RE}",
        re.I,
    ),
}

SAMPLE_PATTERNS = [
    re.compile(r"\b[Nn]\s*=\s*([0-9][0-9,]*)"),
    re.compile(
        r"\b([0-9][0-9,]*)\s+(?:synthetic\s+)?samples\b",
        re.I,
    ),
    re.compile(
        r"\b([0-9][0-9,]*)\s+"
        r"(?:patients?|records?|instances?|observations?)\b",
        re.I,
    ),
]

CONFUSION_LABEL_RE = re.compile(
    r"\b(TP|TN|FP|FN)\b\s*(?:[:=]\s*)?([0-9]+)",
    re.I,
)


# =============================================================================
# 4. GENERAL HELPERS
# =============================================================================

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PAPER_ROOT))
    except Exception:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        pd.DataFrame(columns=columns or []).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        return

    df = pd.DataFrame(rows)

    if columns:
        extras = [col for col in df.columns if col not in columns]
        df = df[columns + extras]

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def is_excluded_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]

    for part in parts:
        if part in EXCLUDED_DIR_NAMES:
            return True

        if any(
            part.startswith(prefix)
            for prefix in EXCLUDED_DIR_PREFIXES
        ):
            return True

    try:
        if OUTPUT_DIR in path.parents:
            return True
    except Exception:
        pass

    return False


# =============================================================================
# 5. SAFE MANUSCRIPT RESOLUTION
# =============================================================================

def resolve_exact_manuscript() -> Path:
    """
    Resolve the intended Array manuscript safely.

    Priority
    --------
    1. Exact MANUSCRIPT_FILENAME under PAPER_ROOT.
    2. If absent, inspect DOCX files directly under PAPER_ROOT and select a
       UNIQUE strong filename match.
    3. Otherwise fail safely and print the candidate list.

    This function NEVER silently substitutes an arbitrary DOCX.
    """

    exact_path = PAPER_ROOT / MANUSCRIPT_FILENAME

    if exact_path.exists() and exact_path.is_file():
        print()
        print("Exact expected manuscript filename found.")
        return exact_path

    candidates = [
        path
        for path in PAPER_ROOT.glob("*.docx")
        if (
            path.is_file()
            and not path.name.startswith("~$")
            and not is_excluded_path(path)
        )
    ]

    if not candidates:
        raise RuntimeError(
            f"No DOCX manuscripts were found under:\n{PAPER_ROOT}"
        )

    expected_tokens = {
        "final",
        "advancements",
        "synthetic",
        "data",
        "eaai",
    }

    scored: List[Tuple[int, Path]] = []

    for path in candidates:
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            path.stem.lower(),
        )

        tokens = set(normalized.split())

        score = len(expected_tokens.intersection(tokens))

        if "advancements in synthetic data" in normalized:
            score += 5

        if "eaai" in normalized:
            score += 3

        if "final" in normalized:
            score += 2

        # Useful but weaker signals.
        if "synthetic" in normalized:
            score += 1

        if "advancements" in normalized:
            score += 1

        scored.append((score, path))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].name.lower(),
        )
    )

    best_score = scored[0][0]
    best_matches = [
        path
        for score, path in scored
        if score == best_score
    ]

    if best_score >= 5 and len(best_matches) == 1:
        selected = best_matches[0]

        print()
        print("Exact expected filename was not found.")
        print("A unique strong manuscript-name match was found:")
        print(selected)

        return selected

    candidate_text = "\n".join(
        f"  score={score:02d}  {path.name}"
        for score, path in scored
    )

    raise RuntimeError(
        "The exact manuscript filename was not found and no unique "
        "strong replacement could be selected safely.\n\n"
        f"Expected:\n  {MANUSCRIPT_FILENAME}\n\n"
        f"DOCX candidates under:\n  {PAPER_ROOT}\n\n"
        f"{candidate_text}\n\n"
        "Rename/copy the intended manuscript to the expected filename, "
        "or set MANUSCRIPT_FILENAME to the correct current filename."
    )


# =============================================================================
# 6. DOCX EXTRACTION
# =============================================================================

def paragraph_text(paragraph: ET.Element) -> str:
    return normalize_ws(
        " ".join(
            node.text or ""
            for node in paragraph.findall(
                ".//w:t",
                DOCX_NS,
            )
        )
    )


def extract_docx_structure(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "paragraphs": [],
        "images": [],
        "captions": [],
        "tables": [],
        "full_text": "",
    }

    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())

        if "word/document.xml" not in names:
            raise RuntimeError(
                f"DOCX has no word/document.xml: {path}"
            )

        root = ET.fromstring(
            archive.read("word/document.xml")
        )

        rel_map: Dict[str, str] = {}
        rel_name = "word/_rels/document.xml.rels"

        if rel_name in names:
            rel_root = ET.fromstring(
                archive.read(rel_name)
            )

            for rel in rel_root:
                rid = rel.attrib.get("Id", "")
                target = rel.attrib.get("Target", "")

                if not rid or not target:
                    continue

                if target.startswith("../"):
                    target = target[3:]
                elif not target.startswith("word/"):
                    target = "word/" + target.lstrip("/")

                rel_map[rid] = target.replace("\\", "/")

        body = root.find("w:body", DOCX_NS)

        if body is None:
            return result

        paragraph_index = 0

        for child in list(body):
            tag = child.tag.split("}")[-1]

            if tag == "p":
                text = paragraph_text(child)

                result["paragraphs"].append(
                    {
                        "paragraph_index": paragraph_index,
                        "text": text,
                    }
                )

                figure_match = FIGURE_CAPTION_RE.match(text)

                if figure_match:
                    result["captions"].append(
                        {
                            "paragraph_index": paragraph_index,
                            "caption_type": "figure",
                            "number": int(
                                figure_match.group(1)
                            ),
                            "caption": text,
                            "caption_body": figure_match.group(
                                2
                            ).strip(),
                        }
                    )

                table_match = TABLE_CAPTION_RE.match(text)

                if table_match:
                    result["captions"].append(
                        {
                            "paragraph_index": paragraph_index,
                            "caption_type": "table",
                            "number": int(
                                table_match.group(1)
                            ),
                            "caption": text,
                            "caption_body": table_match.group(
                                2
                            ).strip(),
                        }
                    )

                for blip in child.findall(
                    ".//a:blip",
                    DOCX_NS,
                ):
                    rid = blip.attrib.get(
                        f"{{{DOCX_NS['r']}}}embed",
                        "",
                    )

                    target = rel_map.get(rid, "")

                    data = (
                        archive.read(target)
                        if target in names
                        else b""
                    )

                    result["images"].append(
                        {
                            "paragraph_index": paragraph_index,
                            "relationship_id": rid,
                            "media_target": target,
                            "sha256": (
                                sha256_bytes(data)
                                if data
                                else ""
                            ),
                            "bytes": len(data),
                        }
                    )

                paragraph_index += 1

            elif tag == "tbl":
                rows: List[List[str]] = []

                for tr in child.findall(
                    ".//w:tr",
                    DOCX_NS,
                ):
                    cells: List[str] = []

                    for tc in tr.findall(
                        "./w:tc",
                        DOCX_NS,
                    ):
                        cells.append(
                            normalize_ws(
                                " ".join(
                                    paragraph_text(p)
                                    for p in tc.findall(
                                        ".//w:p",
                                        DOCX_NS,
                                    )
                                )
                            )
                        )

                    rows.append(cells)

                result["tables"].append(
                    {
                        "table_index": len(
                            result["tables"]
                        ),
                        "rows": rows,
                        "text": " | ".join(
                            " | ".join(row)
                            for row in rows
                        ),
                    }
                )

        figure_captions = [
            caption
            for caption in result["captions"]
            if caption["caption_type"] == "figure"
        ]

        # Associate each embedded image with the nearest figure caption.
        # This is structural proximity only; it is NOT semantic image analysis.
        for image in result["images"]:
            candidates = sorted(
                figure_captions,
                key=lambda caption: abs(
                    caption["paragraph_index"]
                    - image["paragraph_index"]
                ),
            )

            chosen = None

            for candidate in candidates:
                distance = abs(
                    candidate["paragraph_index"]
                    - image["paragraph_index"]
                )

                if distance <= 4:
                    chosen = candidate
                    break

            image["figure_number"] = (
                chosen["number"]
                if chosen
                else None
            )

            image["caption"] = (
                chosen["caption"]
                if chosen
                else ""
            )

            image["caption_body"] = (
                chosen["caption_body"]
                if chosen
                else ""
            )

        result["full_text"] = "\n".join(
            row["text"]
            for row in result["paragraphs"]
            if row["text"]
        )

    return result


# =============================================================================
# 7. MANUSCRIPT CLAIM EXTRACTION
# =============================================================================

def split_sentences(text: str) -> List[str]:
    cleaned = normalize_ws(
        text.replace("\n", " ")
    )

    if not cleaned:
        return []

    return [
        normalize_ws(part)
        for part in re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9(])",
            cleaned,
        )
        if normalize_ws(part)
    ]


def metric_value_to_float(
    text: str,
) -> Optional[float]:
    try:
        stripped = text.strip()

        if stripped.endswith("%"):
            return float(
                stripped[:-1].strip()
            ) / 100.0

        return float(stripped)

    except Exception:
        return None


def extract_metric_claims(
    text: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sentence_index, sentence in enumerate(
        split_sentences(text),
        start=1,
    ):
        for metric, pattern in (
            DIRECT_METRIC_PATTERNS.items()
        ):
            for match in pattern.finditer(sentence):
                value_text = match.group(1).strip()
                value = metric_value_to_float(
                    value_text
                )

                if value is None:
                    continue

                prefix = sentence[
                    max(0, match.start() - 40):
                    match.start()
                ]

                context = sentence[
                    max(0, match.start() - 80):
                    min(
                        len(sentence),
                        match.end() + 80,
                    )
                ]

                rows.append(
                    {
                        "sentence_index": sentence_index,
                        "metric": metric,
                        "value_text": value_text,
                        "value_numeric": value,
                        "match_text": match.group(0),
                        "prefix": prefix,
                        "context": context,
                        "sentence": sentence,
                    }
                )

    return rows


def extract_multimodal_claims(
    text: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sentence_index, sentence in enumerate(
        split_sentences(text),
        start=1,
    ):
        lower = sentence.lower()

        matched = sorted(
            {
                term
                for term in MULTIMODAL_TERMS
                if term in lower
            }
        )

        if not matched:
            continue

        rows.append(
            {
                "sentence_index": sentence_index,
                "matched_terms": safe_json(
                    matched
                ),
                "clinical_anchor_present": int(
                    any(
                        term in lower
                        for term in CLINICAL_TERMS
                    )
                ),
                "generator_term_present": int(
                    any(
                        term in lower
                        for term in GENERATOR_TERMS
                    )
                ),
                "sentence": sentence,
            }
        )

    return rows


def extract_sample_size_claims(
    text: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sentence_index, sentence in enumerate(
        split_sentences(text),
        start=1,
    ):
        for pattern in SAMPLE_PATTERNS:
            for match in pattern.finditer(sentence):
                value = int(
                    match.group(1).replace(",", "")
                )

                lower = sentence.lower()

                rows.append(
                    {
                        "sentence_index": sentence_index,
                        "sample_size": value,
                        "synthetic_context": int(
                            "synthetic" in lower
                        ),
                        "original_clinical_context": int(
                            any(
                                term in lower
                                for term in CLINICAL_TERMS
                            )
                        ),
                        "sentence": sentence,
                    }
                )

    unique: Dict[
        Tuple[int, int, str],
        Dict[str, Any],
    ] = {}

    for row in rows:
        key = (
            row["sentence_index"],
            row["sample_size"],
            row["sentence"],
        )
        unique[key] = row

    return list(unique.values())


def extract_confusion_text_claims(
    text: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sentence_index, sentence in enumerate(
        split_sentences(text),
        start=1,
    ):
        lower = sentence.lower()

        if (
            "confusion" not in lower
            and not all(
                label.lower() in lower
                for label in (
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                )
            )
        ):
            continue

        labels: Dict[str, int] = {}

        for match in CONFUSION_LABEL_RE.finditer(
            sentence
        ):
            labels[
                match.group(1).upper()
            ] = int(match.group(2))

        complete = all(
            key in labels
            for key in (
                "TP",
                "TN",
                "FP",
                "FN",
            )
        )

        accuracy = None
        total = None

        if complete:
            total = sum(labels.values())

            if total > 0:
                accuracy = (
                    labels["TP"]
                    + labels["TN"]
                ) / total

        rows.append(
            {
                "sentence_index": sentence_index,
                "tp": labels.get("TP"),
                "tn": labels.get("TN"),
                "fp": labels.get("FP"),
                "fn": labels.get("FN"),
                "complete_numeric_confusion": int(
                    complete
                ),
                "n_from_confusion": total,
                "recomputed_accuracy": accuracy,
                "sentence": sentence,
            }
        )

    return rows


# =============================================================================
# 8. PROJECT EVIDENCE
# =============================================================================

def read_text_file(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
    except Exception:
        return None

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

    return None


def discover_project_files() -> List[Path]:
    files: List[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if len(files) >= MAX_SCAN_FILES:
            break

        if (
            not path.is_file()
            or is_excluded_path(path)
        ):
            continue

        files.append(path)

    return files


def classify_scope(
    path: Path,
    text: str = "",
) -> str:
    haystack = (
        str(path)
        + " "
        + text
    ).lower()

    if any(
        term in haystack
        for term in UNRELATED_DATASET_TERMS
    ):
        return (
            "UNRELATED_IMAGE_OR_SIGN_LANGUAGE_CONTEXT"
        )

    if any(
        term in haystack
        for term in CLINICAL_TERMS
    ):
        return (
            "CLINICAL_OR_COVID_CONTEXT"
        )

    return "UNRESOLVED_CONTEXT"


def audit_multimodal_implementation(
    files: Sequence[Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path in files:
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        if path.suffix.lower() == ".docx":
            try:
                text = extract_docx_structure(
                    path
                )["full_text"]
            except Exception:
                continue
        else:
            text = read_text_file(path)

        if not text:
            continue

        lower = text.lower()

        multimodal_hits = sorted(
            {
                term
                for term in MULTIMODAL_TERMS
                if term in lower
            }
        )

        fusion_hits = sorted(
            {
                term
                for term in FUSION_TERMS
                if term in lower
            }
        )

        clinical_hits = sorted(
            {
                term
                for term in CLINICAL_TERMS
                if term in lower
            }
        )

        unrelated_hits = sorted(
            {
                term
                for term in UNRELATED_DATASET_TERMS
                if term in lower
            }
        )

        if (
            not multimodal_hits
            and not fusion_hits
        ):
            continue

        strict = int(
            path.suffix.lower() == ".py"
            and bool(multimodal_hits)
            and bool(fusion_hits)
            and bool(clinical_hits)
            and not unrelated_hits
        )

        rows.append(
            {
                "file": relative_path(path),
                "extension": path.suffix.lower(),
                "scope_classification": classify_scope(
                    path,
                    text,
                ),
                "executable_python": int(
                    path.suffix.lower() == ".py"
                ),
                "multimodal_terms": safe_json(
                    multimodal_hits
                ),
                "fusion_terms": safe_json(
                    fusion_hits
                ),
                "clinical_anchor_terms": safe_json(
                    clinical_hits
                ),
                "unrelated_dataset_terms": safe_json(
                    unrelated_hits
                ),
                "strict_clinical_multimodal_candidate": strict,
            }
        )

    return rows


def audit_data_modalities(
    files: Sequence[Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    accepted = (
        IMAGE_EXTENSIONS
        | TABULAR_EXTENSIONS
        | ARRAY_EXTENSIONS
    )

    for path in files:
        suffix = path.suffix.lower()

        if suffix not in accepted:
            continue

        scope = classify_scope(path)

        rows.append(
            {
                "file": relative_path(path),
                "extension": suffix,
                "scope_classification": scope,
                "candidate_image_modality": int(
                    suffix in IMAGE_EXTENSIONS
                ),
                "candidate_tabular_modality": int(
                    suffix in TABULAR_EXTENSIONS
                ),
                "candidate_array_artifact": int(
                    suffix in ARRAY_EXTENSIONS
                ),
            }
        )

    return rows


# =============================================================================
# 9. FIGURE AUDIT
# =============================================================================

def image_metadata(
    data: bytes,
) -> Tuple[
    Optional[int],
    Optional[int],
    str,
    str,
]:
    if not data or Image is None:
        return (
            None,
            None,
            "",
            "",
        )

    try:
        import io

        with Image.open(
            io.BytesIO(data)
        ) as img:
            width, height = img.size
            image_format = img.format or ""

            gray = img.convert("L").resize(
                (8, 8)
            )

            pixels = list(gray.getdata())
            mean = sum(pixels) / len(pixels)

            bits = "".join(
                "1" if value >= mean else "0"
                for value in pixels
            )

            average_hash = (
                f"{int(bits, 2):016x}"
            )

            return (
                width,
                height,
                image_format,
                average_hash,
            )

    except Exception:
        return (
            None,
            None,
            "",
            "",
        )


def figure_classification(
    caption: str,
) -> Tuple[str, str]:
    lower = caption.lower()

    if any(
        term in lower
        for term in UNRELATED_DATASET_TERMS
    ):
        return (
            "UNRELATED_DATASET_FIGURE",
            (
                "Caption explicitly references unrelated "
                "sign-language/image dataset context."
            ),
        )

    if any(
        term in lower
        for term in MULTIMODAL_TERMS
    ):
        return (
            "MULTIMODAL_SUPPORT_REQUIRED",
            (
                "Caption contains multimodal terminology "
                "requiring recoverable clinical multimodal implementation."
            ),
        )

    if (
        "confusion" in lower
        and "matrix" in lower
    ):
        return (
            "CONFUSION_MATRIX_REVIEW",
            (
                "Confusion-matrix figure requires numerical consistency "
                "check against the same evaluation condition."
            ),
        )

    if any(
        term in lower
        for term in CLINICAL_TERMS
    ):
        return (
            "STRUCTURED_CLINICAL_CANDIDATE",
            (
                "Caption contains clinical anchors; provenance still "
                "requires manual/source confirmation."
            ),
        )

    if (
        "fid" in lower
        or "synthetic" in lower
        or any(
            term in lower
            for term in GENERATOR_TERMS
        )
    ):
        return (
            "GENERATIVE_RESULT_SUPPORT_REQUIRED",
            (
                "Generative/synthetic figure requires generator "
                "and result provenance."
            ),
        )

    return (
        "UNCLASSIFIED_FIGURE",
        (
            "No decisive scope term was found "
            "in the caption."
        ),
    )


def build_figure_inventory(
    manuscript: Path,
    structure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with zipfile.ZipFile(
        manuscript,
        "r",
    ) as archive:
        names = set(archive.namelist())

        for occurrence, image in enumerate(
            structure["images"],
            start=1,
        ):
            target = image.get(
                "media_target",
                "",
            )

            data = (
                archive.read(target)
                if target in names
                else b""
            )

            (
                width,
                height,
                image_format,
                average_hash,
            ) = image_metadata(data)

            caption = image.get(
                "caption",
                "",
            )

            category, reason = (
                figure_classification(caption)
            )

            rows.append(
                {
                    "image_occurrence": occurrence,
                    "paragraph_index": image.get(
                        "paragraph_index"
                    ),
                    "figure_number": image.get(
                        "figure_number"
                    ),
                    "caption": caption,
                    "caption_body": image.get(
                        "caption_body",
                        "",
                    ),
                    "media_target": target,
                    "sha256": image.get(
                        "sha256",
                        "",
                    ),
                    "average_hash": average_hash,
                    "bytes": image.get(
                        "bytes",
                        0,
                    ),
                    "width": width,
                    "height": height,
                    "image_format": image_format,
                    "figure_classification": category,
                    "classification_reason": reason,
                }
            )

    return rows


def duplicate_figure_number_audit(
    structure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    captions = [
        caption
        for caption in structure["captions"]
        if caption["caption_type"] == "figure"
    ]

    grouped: Dict[
        int,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for caption in captions:
        grouped[
            caption["number"]
        ].append(caption)

    rows: List[Dict[str, Any]] = []

    for number, group in sorted(
        grouped.items()
    ):
        if len(group) <= 1:
            continue

        rows.append(
            {
                "figure_number": number,
                "caption_count": len(group),
                "paragraph_indices": safe_json(
                    [
                        item["paragraph_index"]
                        for item in group
                    ]
                ),
                "captions": safe_json(
                    [
                        item["caption"]
                        for item in group
                    ]
                ),
                "status": "DUPLICATE_FIGURE_NUMBER",
                "requires_correction": 1,
            }
        )

    return rows


def exact_duplicate_image_audit(
    figure_rows: Sequence[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in figure_rows:
        image_hash = row.get(
            "sha256",
            "",
        )

        if image_hash:
            grouped[
                image_hash
            ].append(row)

    rows: List[Dict[str, Any]] = []
    group_id = 0

    for image_hash, group in (
        grouped.items()
    ):
        if len(group) <= 1:
            continue

        group_id += 1

        rows.append(
            {
                "duplicate_group": group_id,
                "sha256": image_hash,
                "occurrences": len(group),
                "figure_numbers": safe_json(
                    [
                        item.get(
                            "figure_number"
                        )
                        for item in group
                    ]
                ),
                "captions": safe_json(
                    [
                        item.get(
                            "caption",
                            "",
                        )
                        for item in group
                    ]
                ),
                "status": (
                    "EXACT_DUPLICATE_EMBEDDED_IMAGE"
                ),
            }
        )

    return rows


def hamming_distance_hex(
    left: str,
    right: str,
) -> Optional[int]:
    if not left or not right:
        return None

    try:
        return (
            int(left, 16)
            ^ int(right, 16)
        ).bit_count()

    except Exception:
        return None


def perceptual_similarity_audit(
    figure_rows: Sequence[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for left_index in range(
        len(figure_rows)
    ):
        for right_index in range(
            left_index + 1,
            len(figure_rows),
        ):
            left = figure_rows[
                left_index
            ]

            right = figure_rows[
                right_index
            ]

            distance = hamming_distance_hex(
                left.get(
                    "average_hash",
                    "",
                ),
                right.get(
                    "average_hash",
                    "",
                ),
            )

            if (
                distance is None
                or distance > 6
            ):
                continue

            if (
                left.get("sha256")
                == right.get("sha256")
            ):
                continue

            rows.append(
                {
                    "left_occurrence": left.get(
                        "image_occurrence"
                    ),
                    "right_occurrence": right.get(
                        "image_occurrence"
                    ),
                    "left_figure_number": left.get(
                        "figure_number"
                    ),
                    "right_figure_number": right.get(
                        "figure_number"
                    ),
                    "hamming_distance_64bit_ahash": distance,
                    "left_caption": left.get(
                        "caption",
                        "",
                    ),
                    "right_caption": right.get(
                        "caption",
                        "",
                    ),
                    "status": (
                        "VISUALLY_SIMILAR_CANDIDATE_"
                        "MANUAL_REVIEW"
                    ),
                }
            )

    return rows


def caption_tokens(
    caption: str,
) -> set[str]:
    stop = {
        "figure",
        "fig",
        "the",
        "a",
        "an",
        "of",
        "for",
        "and",
        "to",
        "in",
        "on",
        "with",
        "using",
        "result",
        "results",
        "show",
        "shows",
        "performance",
    }

    tokens = set(
        re.findall(
            r"[a-z0-9]+",
            caption.lower(),
        )
    )

    return {
        token
        for token in tokens
        if (
            token not in stop
            and len(token) > 1
        )
    }


def caption_similarity_audit(
    structure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    captions = [
        caption
        for caption in structure["captions"]
        if caption["caption_type"] == "figure"
    ]

    rows: List[Dict[str, Any]] = []

    for left_index in range(
        len(captions)
    ):
        for right_index in range(
            left_index + 1,
            len(captions),
        ):
            left = captions[
                left_index
            ]
            right = captions[
                right_index
            ]

            left_tokens = caption_tokens(
                left["caption_body"]
            )
            right_tokens = caption_tokens(
                right["caption_body"]
            )

            if (
                not left_tokens
                or not right_tokens
            ):
                continue

            union = (
                left_tokens
                | right_tokens
            )

            score = (
                len(
                    left_tokens
                    & right_tokens
                )
                / len(union)
            ) if union else 0.0

            confusion_pair = (
                "confusion"
                in left["caption"].lower()
                and "confusion"
                in right["caption"].lower()
            )

            if (
                score >= 0.70
                or (
                    confusion_pair
                    and score >= 0.50
                )
            ):
                rows.append(
                    {
                        "left_figure_number": left[
                            "number"
                        ],
                        "right_figure_number": right[
                            "number"
                        ],
                        "jaccard_similarity": score,
                        "left_caption": left[
                            "caption"
                        ],
                        "right_caption": right[
                            "caption"
                        ],
                        "both_confusion_matrix_captions": int(
                            confusion_pair
                        ),
                        "status": (
                            "SIMILAR_EXPERIMENTAL_"
                            "CAPTION_MANUAL_REVIEW"
                        ),
                    }
                )

    return rows


def figure_caption_audit(
    structure: Dict[str, Any],
    figure_rows: Sequence[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    captions = [
        caption
        for caption in structure["captions"]
        if caption["caption_type"] == "figure"
    ]

    number_counts = Counter(
        caption["number"]
        for caption in captions
    )

    image_numbers = Counter(
        row.get("figure_number")
        for row in figure_rows
        if row.get(
            "figure_number"
        ) is not None
    )

    rows: List[Dict[str, Any]] = []

    for caption in captions:
        lower = caption[
            "caption"
        ].lower()

        issues: List[str] = []

        if (
            number_counts[
                caption["number"]
            ]
            > 1
        ):
            issues.append(
                "DUPLICATE_FIGURE_NUMBER"
            )

        if (
            image_numbers[
                caption["number"]
            ]
            == 0
        ):
            issues.append(
                "CAPTION_WITHOUT_MATCHED_EMBEDDED_IMAGE"
            )

        if (
            "confusion" in lower
            and "matrix" in lower
        ):
            issues.append(
                "CONFUSION_MATRIX_NUMERICAL_CROSSCHECK_REQUIRED"
            )

        if any(
            term in lower
            for term in MULTIMODAL_TERMS
        ):
            issues.append(
                "MULTIMODAL_IMPLEMENTATION_SUPPORT_REQUIRED"
            )

        if any(
            term in lower
            for term in UNRELATED_DATASET_TERMS
        ):
            issues.append(
                "UNRELATED_DATASET_SCOPE_REVIEW"
            )

        if "fid" in lower:
            issues.append(
                "FID_PROVENANCE_REVIEW"
            )

        rows.append(
            {
                "figure_number": caption[
                    "number"
                ],
                "paragraph_index": caption[
                    "paragraph_index"
                ],
                "caption": caption[
                    "caption"
                ],
                "matched_embedded_images": image_numbers[
                    caption["number"]
                ],
                "issues": safe_json(issues),
                "issue_count": len(issues),
            }
        )

    return rows


# =============================================================================
# 10. PRIOR AUDIT CROSS-CHECK
# =============================================================================

def find_prior_audit_evidence(
) -> List[Dict[str, Any]]:
    targets = {
        "generator_missing": (
            "CLASSIFIER_OR_ENCODER_ONLY_NO_STRUCTURED_GENERATOR",
            "DO_NOT_REGENERATE_REMOVE_UNSUPPORTED_GENERATIVE_FID_CLAIMS",
        ),
        "fid_unsupported": (
            "NO_REPRODUCIBLE_FID_OR_STRUCTURED_FRECHET_RESULT_AVAILABLE",
            "REMOVE_UNSUPPORTED_GENERATIVE_FID_CLAIMS",
        ),
        "scalability_unsupported": (
            "GENERATIVE_SCALABILITY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS",
            "REMOVE_UNSUPPORTED_RUNTIME_MEMORY_AND_SYNTHETIC_SCALE_CLAIMS",
        ),
        "ablation_unsupported": (
            "ABLATION_STUDY_NOT_REPRODUCIBLE_FROM_RECOVERED_ARTIFACTS",
            "REMOVE_UNSUPPORTED_ABLATION_CLAIMS",
        ),
        "statistics_descriptive_only": (
            "DESCRIPTIVE_STABILITY_EVIDENCE_VALID_FORMAL_SIGNIFICANCE_NOT_SUPPORTED",
            "FORMAL_SIGNIFICANCE_NOT_SUPPORTED",
        ),
    }

    rows: List[Dict[str, Any]] = []
    outputs_root = PROJECT_ROOT / "outputs"

    if not outputs_root.exists():
        return rows

    for path in outputs_root.rglob("*"):
        if (
            not path.is_file()
            or is_excluded_path(path)
        ):
            continue

        if path.suffix.lower() not in {
            ".txt",
            ".csv",
            ".log",
            ".md",
            ".json",
        }:
            continue

        text = read_text_file(path)

        if not text:
            continue

        upper = text.upper()

        for category, terms in (
            targets.items()
        ):
            matched = [
                term
                for term in terms
                if term.upper() in upper
            ]

            if not matched:
                continue

            rows.append(
                {
                    "category": category,
                    "file": relative_path(path),
                    "matched_terms": safe_json(
                        matched
                    ),
                }
            )

    return rows


# =============================================================================
# 11. FINDINGS
# =============================================================================

def unique_metric_values(
    rows: Sequence[
        Dict[str, Any]
    ],
    metric: str,
) -> List[float]:
    return sorted(
        {
            float(
                row["value_numeric"]
            )
            for row in rows
            if row.get("metric") == metric
        }
    )


def build_findings(
    multimodal_claims,
    implementation_rows,
    data_rows,
    duplicate_numbers,
    exact_duplicates,
    perceptual_duplicates,
    caption_similarities,
    caption_rows,
    metric_rows,
    sample_rows,
    confusion_rows,
    prior_rows,
):
    strict_multimodal = [
        row
        for row in implementation_rows
        if row[
            "strict_clinical_multimodal_candidate"
        ] == 1
    ]

    clinical_images = [
        row
        for row in data_rows
        if (
            row["candidate_image_modality"] == 1
            and row[
                "scope_classification"
            ]
            == "CLINICAL_OR_COVID_CONTEXT"
        )
    ]

    unrelated_images = [
        row
        for row in data_rows
        if (
            row["candidate_image_modality"] == 1
            and row[
                "scope_classification"
            ]
            == "UNRELATED_IMAGE_OR_SIGN_LANGUAGE_CONTEXT"
        )
    ]

    prior_categories = {
        row["category"]
        for row in prior_rows
    }

    accuracy_values = unique_metric_values(
        metric_rows,
        "accuracy",
    )

    fid_values = unique_metric_values(
        metric_rows,
        "fid",
    )

    sample_sizes = sorted(
        {
            int(row["sample_size"])
            for row in sample_rows
        }
    )

    large_synthetic = sorted(
        {
            int(row["sample_size"])
            for row in sample_rows
            if (
                row["synthetic_context"]
                and int(
                    row["sample_size"]
                )
                > EXPECTED_PARTICIPANTS
            )
        }
    )

    complete_confusions = [
        row
        for row in confusion_rows
        if row[
            "complete_numeric_confusion"
        ] == 1
    ]

    findings = [
        {
            "reviewer_comment": "C13-C14",
            "topic": (
                "multimodal_scope_and_reproducibility"
            ),
            "status": (
                "SUPPORTED"
                if strict_multimodal
                else
                "NOT_REPRODUCIBLE_FROM_RECOVERED_PROJECT"
            ),
            "evidence": (
                f"manuscript multimodal claims="
                f"{len(multimodal_claims)}; "
                f"strict executable clinical multimodal+fusion "
                f"candidates={len(strict_multimodal)}; "
                f"clinical image artifacts="
                f"{len(clinical_images)}; "
                f"unrelated image/sign-language artifacts="
                f"{len(unrelated_images)}"
            ),
            "manuscript_action": (
                "If strict clinical multimodal evidence remains absent, "
                "remove or reframe multimodal claims and describe the "
                "evaluated study as structured clinical/tabular data only."
            ),
        },
        {
            "reviewer_comment": "C15",
            "topic": (
                "figure_numbering_and_duplicate_content"
            ),
            "status": (
                "CORRECTION_REQUIRED"
                if (
                    duplicate_numbers
                    or exact_duplicates
                    or perceptual_duplicates
                    or caption_similarities
                )
                else
                "NO_DUPLICATE_SIGNAL_DETECTED"
            ),
            "evidence": (
                f"duplicate figure-number groups="
                f"{len(duplicate_numbers)}; "
                f"exact duplicate image groups="
                f"{len(exact_duplicates)}; "
                f"visually similar image candidate pairs="
                f"{len(perceptual_duplicates)}; "
                f"similar-caption candidate pairs="
                f"{len(caption_similarities)}"
            ),
            "manuscript_action": (
                "Correct duplicate figure numbering. Review exact, "
                "visual, and caption-similarity candidates manually, "
                "especially confusion matrices, before retaining any "
                "apparently distinct experiment."
            ),
        },
        {
            "reviewer_comment": "C22",
            "topic": (
                "accuracy_and_confusion_consistency"
            ),
            "status": "REVIEW_REQUIRED",
            "evidence": (
                f"directly parsed accuracy values="
                f"{safe_json(accuracy_values)}; "
                f"complete TP/TN/FP/FN text claims="
                f"{len(complete_confusions)}"
            ),
            "manuscript_action": (
                "For each retained confusion matrix, tie TP/TN/FP/FN "
                "to one documented evaluation condition and ensure "
                "stated accuracy equals (TP+TN)/N. "
                "Do not treat AUC values as accuracy."
            ),
        },
        {
            "reviewer_comment": "C23",
            "topic": (
                "synthetic_sample_size_interpretation"
            ),
            "status": (
                "REVIEW_REQUIRED"
                if large_synthetic
                else
                "NO_LARGE_SYNTHETIC_N_DIRECTLY_PARSED"
            ),
            "evidence": (
                f"sample sizes="
                f"{safe_json(sample_sizes)}; "
                f"large synthetic sizes>"
                f"{EXPECTED_PARTICIPANTS}="
                f"{safe_json(large_synthetic)}"
            ),
            "manuscript_action": (
                "Describe any synthetic N above the original cohort "
                "only as generated computational workload, never as "
                "additional independent patient information."
            ),
        },
        {
            "reviewer_comment": "C24",
            "topic": (
                "fid_consistency_and_reproducibility"
            ),
            "status": (
                "UNSUPPORTED"
                if (
                    "fid_unsupported"
                    in prior_categories
                    or "generator_missing"
                    in prior_categories
                )
                else (
                    "REVIEW_REQUIRED"
                    if fid_values
                    else
                    "NO_FID_VALUE_PARSED"
                )
            ),
            "evidence": (
                f"directly parsed manuscript FID values="
                f"{safe_json(fid_values)}; "
                f"prior FID/generator unsupported="
                f"{int('fid_unsupported' in prior_categories or 'generator_missing' in prior_categories)}"
            ),
            "manuscript_action": (
                "Remove absolute FID claims unless generator provenance "
                "and the exact feature space are recovered. Do not label "
                "raw structured-feature Fréchet distance as conventional "
                "image FID."
            ),
        },
    ]

    major = (
        not strict_multimodal
        or "fid_unsupported" in prior_categories
        or "generator_missing" in prior_categories
    )

    findings.append(
        {
            "reviewer_comment": (
                "C13-C15,C22-C24"
            ),
            "topic": (
                "overall_results_integrity"
            ),
            "status": (
                "MAJOR_REFRAME_REQUIRED"
                if major
                else
                "TARGETED_CORRECTIONS_REQUIRED"
            ),
            "evidence": (
                f"strict clinical multimodal implementation="
                f"{len(strict_multimodal)}; "
                f"duplicate figure numbers="
                f"{len(duplicate_numbers)}; "
                f"exact duplicate images="
                f"{len(exact_duplicates)}; "
                f"FID/generator prior unsupported="
                f"{int('fid_unsupported' in prior_categories or 'generator_missing' in prior_categories)}; "
                f"caption issue count="
                f"{sum(int(row['issue_count']) for row in caption_rows)}"
            ),
            "manuscript_action": (
                "Consolidate the manuscript around reproducible "
                "structured-clinical evidence and remove unsupported "
                "multimodal/generator/FID/scalability interpretations."
            ),
        }
    )

    return findings


# =============================================================================
# 12. VERIFICATION MATRIX
# =============================================================================

def build_verification_matrix(
    manuscript,
    figure_rows,
    structure,
    duplicate_numbers,
    implementation_rows,
    metric_rows,
    prior_rows,
):
    strict_multimodal = [
        row
        for row in implementation_rows
        if row[
            "strict_clinical_multimodal_candidate"
        ] == 1
    ]

    prior_categories = {
        row["category"]
        for row in prior_rows
    }

    figure_captions = [
        caption
        for caption in structure["captions"]
        if caption["caption_type"] == "figure"
    ]

    accuracy_values = unique_metric_values(
        metric_rows,
        "accuracy",
    )

    auc_values = unique_metric_values(
        metric_rows,
        "auc",
    )

    return [
        {
            "criterion": (
                "intended_array_manuscript_selected"
            ),
            "passed": int(
                manuscript.suffix.lower() == ".docx"
                and (
                    manuscript.name
                    == MANUSCRIPT_FILENAME
                    or (
                        "synthetic"
                        in manuscript.name.lower()
                        and "advancement"
                        in manuscript.name.lower()
                    )
                )
            ),
            "detail": str(manuscript),
        },
        {
            "criterion": (
                "figure_inventory_completed"
            ),
            "passed": 1,
            "detail": (
                f"embedded images="
                f"{len(figure_rows)}; "
                f"figure captions="
                f"{len(figure_captions)}"
            ),
        },
        {
            "criterion": (
                "duplicate_figure_numbers_explicitly_audited"
            ),
            "passed": 1,
            "detail": (
                f"duplicate figure-number groups="
                f"{len(duplicate_numbers)}"
            ),
        },
        {
            "criterion": (
                "context_aware_metric_extraction_used"
            ),
            "passed": 1,
            "detail": (
                f"direct accuracy values="
                f"{safe_json(accuracy_values)}; "
                f"direct AUC values="
                f"{safe_json(auc_values)}"
            ),
        },
        {
            "criterion": (
                "clinical_multimodal_implementation_verified"
            ),
            "passed": int(
                bool(strict_multimodal)
            ),
            "detail": (
                "strict executable clinical multimodal+fusion "
                f"candidates={len(strict_multimodal)}"
            ),
        },
        {
            "criterion": (
                "fid_claims_crosschecked_with_prior_generator_audit"
            ),
            "passed": int(
                (
                    "fid_unsupported"
                    in prior_categories
                )
                or (
                    "generator_missing"
                    in prior_categories
                )
            ),
            "detail": (
                f"prior categories="
                f"{safe_json(sorted(prior_categories))}"
            ),
        },
    ]


# =============================================================================
# 13. FINAL VERDICT
# =============================================================================

def build_verdict(
    findings,
    matrix,
):
    intended_manuscript = next(
        (
            row["passed"]
            for row in matrix
            if row["criterion"]
            == "intended_array_manuscript_selected"
        ),
        0,
    )

    multimodal_verified = next(
        (
            row["passed"]
            for row in matrix
            if row["criterion"]
            == "clinical_multimodal_implementation_verified"
        ),
        0,
    )

    fid_unsupported = any(
        (
            row["topic"]
            == "fid_consistency_and_reproducibility"
        )
        and row["status"] == "UNSUPPORTED"
        for row in findings
    )

    duplicate_issue = any(
        (
            row["topic"]
            == "figure_numbering_and_duplicate_content"
        )
        and row["status"] == "CORRECTION_REQUIRED"
        for row in findings
    )

    if not intended_manuscript:
        verdict = (
            "AUDIT_INVALID_WRONG_OR_UNCONFIRMED_MANUSCRIPT"
        )

        next_action = (
            "DO_NOT_USE_RESULTS_RESOLVE_INTENDED_ARRAY_MANUSCRIPT"
        )

    elif (
        not multimodal_verified
        or fid_unsupported
    ):
        verdict = (
            "ARRAY_MANUSCRIPT_REQUIRES_"
            "MULTIMODAL_GENERATIVE_REFRAMING"
        )

        next_action = (
            "RETAIN_REPRODUCIBLE_STRUCTURED_CLINICAL_RESULTS_"
            "AND_CORRECT_FIGURE_NUMBERING_AND_NUMERICS"
        )

    elif duplicate_issue:
        verdict = (
            "ARRAY_MANUSCRIPT_REQUIRES_"
            "FIGURE_AND_RESULT_CONSISTENCY_CORRECTIONS"
        )

        next_action = (
            "CORRECT_FIGURE_NUMBERING_CAPTIONS_"
            "AND_NUMERICAL_LINKAGE"
        )

    else:
        verdict = (
            "TARGETED_RESULT_CONSISTENCY_REVIEW_REQUIRED"
        )

        next_action = (
            "VERIFY_REMAINING_CONFUSION_MATRIX_"
            "AND_NUMERICAL_CLAIMS"
        )

    return {
        "verdict": verdict,
        "next_action": next_action,
        "new_training_performed": False,
        "new_synthetic_generation_performed": False,
        "ocr_performed": False,
        "image_modification_performed": False,
        "historical_files_modified": False,
    }


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:
    print("=" * 100)

    print(
        "HFAGM - 09 V2 MULTIMODAL / FIGURE / "
        "RESULTS CONSISTENCY AUDIT"
    )

    print("=" * 100)

    print()
    print("Restrictions:")
    print("  - exact manuscript preferred")
    print(
        "  - unique strong filename match allowed if exact filename is absent"
    )
    print("  - no arbitrary DOCX fallback")
    print("  - context-aware metric parsing")
    print("  - duplicate figure-number audit")
    print(
        "  - exact and perceptual image duplicate audit"
    )
    print(
        "  - no training, no synthetic generation, "
        "no OCR, no image modification"
    )

    manuscript = resolve_exact_manuscript()

    print()
    print("Selected manuscript:")
    print(manuscript)

    structure = extract_docx_structure(
        manuscript
    )

    manuscript_text = structure[
        "full_text"
    ]

    multimodal_claims = (
        extract_multimodal_claims(
            manuscript_text
        )
    )

    metric_claims = (
        extract_metric_claims(
            manuscript_text
        )
    )

    sample_claims = (
        extract_sample_size_claims(
            manuscript_text
        )
    )

    confusion_claims = (
        extract_confusion_text_claims(
            manuscript_text
        )
    )

    project_files = discover_project_files()

    implementation_rows = (
        audit_multimodal_implementation(
            project_files
        )
    )

    data_rows = audit_data_modalities(
        project_files
    )

    figure_rows = build_figure_inventory(
        manuscript,
        structure,
    )

    duplicate_numbers = (
        duplicate_figure_number_audit(
            structure
        )
    )

    exact_duplicates = (
        exact_duplicate_image_audit(
            figure_rows
        )
    )

    perceptual_duplicates = (
        perceptual_similarity_audit(
            figure_rows
        )
    )

    caption_similarities = (
        caption_similarity_audit(
            structure
        )
    )

    caption_rows = figure_caption_audit(
        structure,
        figure_rows,
    )

    prior_rows = find_prior_audit_evidence()

    findings = build_findings(
        multimodal_claims,
        implementation_rows,
        data_rows,
        duplicate_numbers,
        exact_duplicates,
        perceptual_duplicates,
        caption_similarities,
        caption_rows,
        metric_claims,
        sample_claims,
        confusion_claims,
        prior_rows,
    )

    matrix = build_verification_matrix(
        manuscript,
        figure_rows,
        structure,
        duplicate_numbers,
        implementation_rows,
        metric_claims,
        prior_rows,
    )

    verdict = build_verdict(
        findings,
        matrix,
    )

    outputs = {
        "manuscript_multimodal_claims_v2.csv":
            multimodal_claims,
        "manuscript_metric_claims_v2.csv":
            metric_claims,
        "manuscript_sample_size_claims_v2.csv":
            sample_claims,
        "manuscript_confusion_text_claims_v2.csv":
            confusion_claims,
        "multimodal_implementation_evidence_v2.csv":
            implementation_rows,
        "data_modality_inventory_v2.csv":
            data_rows,
        "figure_inventory_v2.csv":
            figure_rows,
        "duplicate_figure_numbers_v2.csv":
            duplicate_numbers,
        "exact_duplicate_figure_groups_v2.csv":
            exact_duplicates,
        "perceptual_figure_similarity_v2.csv":
            perceptual_duplicates,
        "figure_caption_similarity_v2.csv":
            caption_similarities,
        "figure_caption_audit_v2.csv":
            caption_rows,
        "prior_audit_crosscheck_v2.csv":
            prior_rows,
        "results_consistency_findings_v2.csv":
            findings,
        "multimodal_figure_verification_matrix_v2.csv":
            matrix,
        "multimodal_figure_verdict_v2.csv":
            [verdict],
    }

    for filename, rows in outputs.items():
        write_csv(
            OUTPUT_DIR / filename,
            rows,
        )

    prior_categories = sorted(
        {
            row["category"]
            for row in prior_rows
        }
    )

    provenance = {
        "generated": datetime.now().isoformat(
            timespec="seconds"
        ),
        "script": (
            "09_multimodal_figure_results_audit_v2.py"
        ),
        "project_root": str(PROJECT_ROOT),
        "paper_root": str(PAPER_ROOT),
        "preferred_manuscript_filename":
            MANUSCRIPT_FILENAME,
        "selected_manuscript": str(
            manuscript
        ),
        "manuscript_sha256": sha256_file(
            manuscript
        ),
        "project_files_scanned": len(
            project_files
        ),
        "multimodal_claim_rows": len(
            multimodal_claims
        ),
        "metric_claim_rows": len(
            metric_claims
        ),
        "sample_size_claim_rows": len(
            sample_claims
        ),
        "confusion_text_claim_rows": len(
            confusion_claims
        ),
        "embedded_image_occurrences": len(
            figure_rows
        ),
        "duplicate_figure_number_groups": len(
            duplicate_numbers
        ),
        "exact_duplicate_figure_groups": len(
            exact_duplicates
        ),
        "perceptual_similarity_pairs": len(
            perceptual_duplicates
        ),
        "caption_similarity_pairs": len(
            caption_similarities
        ),
        "strict_clinical_multimodal_candidates": sum(
            int(
                row[
                    "strict_clinical_multimodal_candidate"
                ]
            )
            for row in implementation_rows
        ),
        "prior_audit_categories": safe_json(
            prior_categories
        ),
        "new_training_performed": False,
        "new_synthetic_generation_performed": False,
        "ocr_performed": False,
        "image_modification_performed": False,
        "historical_files_modified": False,
        "verdict": verdict["verdict"],
        "python_version": sys.version,
        "pandas_version": pd.__version__,
    }

    write_csv(
        OUTPUT_DIR
        / "multimodal_figure_provenance_v2.csv",
        [provenance],
    )

    accuracy_values = unique_metric_values(
        metric_claims,
        "accuracy",
    )

    auc_values = unique_metric_values(
        metric_claims,
        "auc",
    )

    fid_values = unique_metric_values(
        metric_claims,
        "fid",
    )

    lines = [
        "=" * 100,
        (
            "HFAGM - 09 V2 MULTIMODAL / FIGURE / "
            "RESULTS CONSISTENCY AUDIT"
        ),
        "=" * 100,
        "",
        f"Generated: {provenance['generated']}",
        (
            f"Preferred manuscript filename: "
            f"{MANUSCRIPT_FILENAME}"
        ),
        (
            f"Selected manuscript: "
            f"{manuscript}"
        ),
        (
            f"Manuscript SHA256: "
            f"{provenance['manuscript_sha256']}"
        ),
        "",
        "CORRECTIONS FROM 09 V1",
        "-" * 100,
        (
            "1. Exact expected manuscript filename is preferred; "
            "if absent, only a unique strong filename match is accepted."
        ),
        (
            "2. The script never silently substitutes an arbitrary DOCX."
        ),
        (
            "3. Accuracy is parsed only when a numeric value directly "
            "follows the accuracy label."
        ),
        (
            "4. AUC values are parsed independently and cannot be "
            "reassigned to accuracy."
        ),
        (
            "5. Duplicate figure numbers are audited independently "
            "of image hashes."
        ),
        (
            "6. Exact-image, perceptual-image, and caption-similarity "
            "signals are separated."
        ),
        (
            "7. No OCR or image-content interpretation is performed."
        ),
        "",
        "COUNTS",
        "-" * 100,
        (
            f"Project files scanned: "
            f"{len(project_files)}"
        ),
        (
            f"Multimodal manuscript claim rows: "
            f"{len(multimodal_claims)}"
        ),
        (
            f"Metric claim rows: "
            f"{len(metric_claims)}"
        ),
        (
            f"Sample-size claim rows: "
            f"{len(sample_claims)}"
        ),
        (
            f"Confusion-text claim rows: "
            f"{len(confusion_claims)}"
        ),
        (
            f"Embedded image occurrences: "
            f"{len(figure_rows)}"
        ),
        (
            f"Duplicate figure-number groups: "
            f"{len(duplicate_numbers)}"
        ),
        (
            f"Exact duplicate image groups: "
            f"{len(exact_duplicates)}"
        ),
        (
            f"Perceptual similarity pairs: "
            f"{len(perceptual_duplicates)}"
        ),
        (
            f"Similar-caption pairs: "
            f"{len(caption_similarities)}"
        ),
        (
            "Strict executable clinical multimodal+fusion "
            f"candidates: "
            f"{provenance['strict_clinical_multimodal_candidates']}"
        ),
        "",
        "CONTEXT-AWARE METRIC PARSING",
        "-" * 100,
        (
            f"Direct accuracy values: "
            f"{accuracy_values}"
        ),
        (
            f"Direct AUC values: "
            f"{auc_values}"
        ),
        (
            f"Direct FID values: "
            f"{fid_values}"
        ),
        "",
        "FINDINGS",
        "-" * 100,
    ]

    for row in findings:
        lines.append(
            f"{row['reviewer_comment']} | "
            f"{row['topic']} | "
            f"{row['status']}"
        )
        lines.append(
            f"    Evidence: "
            f"{row['evidence']}"
        )
        lines.append(
            f"    Manuscript action: "
            f"{row['manuscript_action']}"
        )

    lines.extend(
        [
            "",
            "VERIFICATION MATRIX",
            "-" * 100,
        ]
    )

    for row in matrix:
        state = (
            "PASS"
            if int(row["passed"]) == 1
            else
            "MISSING/NO"
        )

        lines.append(
            f"{state}: "
            f"{row['criterion']}"
        )

        lines.append(
            f"    {row['detail']}"
        )

    lines.extend(
        [
            "",
            "FINAL VERDICT",
            "-" * 100,
            verdict["verdict"],
            "",
            "NEXT ACTION",
            "-" * 100,
            verdict["next_action"],
            "",
            "INTERPRETATION RULES",
            "-" * 100,
            (
                "1. AUC and accuracy are different metrics; "
                "an AUC value is never reassigned to accuracy."
            ),
            (
                "2. Duplicate figure number and duplicate image "
                "content are independent problems."
            ),
            (
                "3. Similar image hashes or captions are candidates "
                "for manual review, not proof of duplication."
            ),
            (
                "4. An unrelated image dataset in the repository "
                "does not establish multimodal clinical evaluation."
            ),
            (
                "5. Synthetic N does not increase the independent "
                "clinical sample beyond the original cohort."
            ),
            (
                "6. FID remains unsupported if generator and "
                "feature-space provenance are absent."
            ),
            "",
            "SAFETY",
            "-" * 100,
            "New training: NO",
            "Synthetic generation: NO",
            "OCR: NO",
            "Image modification: NO",
            "Historical project modification: NO",
            "",
            "=" * 100,
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "multimodal_figure_results_audit_v2_summary.txt"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("09 V2 COMPLETE")
    print("=" * 100)

    print(
        f"Selected manuscript: "
        f"{manuscript.name}"
    )
    print(
        f"Direct accuracy values: "
        f"{accuracy_values}"
    )
    print(
        f"Direct AUC values: "
        f"{auc_values}"
    )
    print(
        f"Duplicate figure-number groups: "
        f"{len(duplicate_numbers)}"
    )
    print(
        f"Exact duplicate image groups: "
        f"{len(exact_duplicates)}"
    )
    print(
        f"Perceptual similarity pairs: "
        f"{len(perceptual_duplicates)}"
    )
    print(
        "Strict clinical multimodal candidates: "
        f"{provenance['strict_clinical_multimodal_candidates']}"
    )

    print()
    print("FINAL VERDICT:")
    print(verdict["verdict"])

    print()
    print("NEXT ACTION:")
    print(verdict["next_action"])

    print()
    print("Results written to:")
    print(OUTPUT_DIR)

    print()
    print("Upload these files first:")

    for filename in [
        "multimodal_figure_results_audit_v2_summary.txt",
        "multimodal_figure_verdict_v2.csv",
        "multimodal_figure_verification_matrix_v2.csv",
        "results_consistency_findings_v2.csv",
        "duplicate_figure_numbers_v2.csv",
        "figure_inventory_v2.csv",
        "figure_caption_audit_v2.csv",
        "perceptual_figure_similarity_v2.csv",
        "figure_caption_similarity_v2.csv",
        "manuscript_metric_claims_v2.csv",
        "manuscript_multimodal_claims_v2.csv",
        "manuscript_sample_size_claims_v2.csv",
        "multimodal_implementation_evidence_v2.csv",
        "prior_audit_crosscheck_v2.csv",
        "multimodal_figure_provenance_v2.csv",
    ]:
        print(
            OUTPUT_DIR / filename
        )


# =============================================================================
# 15. SAFE EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 100)
        print("09 V2 FAILED SAFELY")
        print("=" * 100)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "No model was trained, no synthetic data were generated, "
            "no OCR was performed, no image was modified, "
            "and no historical artifact was changed."
        )

        print()

        traceback.print_exc()

        sys.exit(1)
