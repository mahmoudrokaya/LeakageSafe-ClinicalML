from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
import zipfile
from collections import defaultdict
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "revision_multimodal_figure_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANUSCRIPT_NAME_HINTS = (
    "final_advancements in synthetic data-eaai-w(1).docx",
    "final_advancements in synthetic data-eaai-w.docx",
    "advancements in synthetic data generation2-revised.docx",
    "advancements in synthetic data generation3.docx",
)

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

MAX_TEXT_BYTES = 25 * 1024 * 1024
MAX_SCAN_FILES = 100000


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
    "image",
    "images",
    "imaging",
    "thermal",
    "rgb",
    "audio",
    "text modality",
    "visual modality",
    "sensor modality",
)

CLINICAL_ANCHOR_TERMS = (
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
    "status",
)

UNRELATED_DATASET_HINTS = (
    "arsl",
    "karsl",
    "sign language",
    "arabic sign",
    "asl",
    "gesture",
)

FUSION_TERMS = (
    "fusion",
    "fuse",
    "concatenate",
    "concatenation",
    "torch.cat",
    "np.concatenate",
    "hstack",
    "vstack",
    "cross-modal",
    "cross modal",
    "attention fusion",
    "late fusion",
    "early fusion",
)

METRIC_PATTERNS = {
    "accuracy": re.compile(
        r"\b(?:accuracy|acc)\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?\s*%|0?\.\d+|1\.0+)",
        re.I,
    ),
    "precision": re.compile(
        r"\bprecision\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?\s*%|0?\.\d+|1\.0+)",
        re.I,
    ),
    "recall": re.compile(
        r"\b(?:recall|sensitivity)\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?\s*%|0?\.\d+|1\.0+)",
        re.I,
    ),
    "specificity": re.compile(
        r"\bspecificity\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?\s*%|0?\.\d+|1\.0+)",
        re.I,
    ),
    "f1": re.compile(
        r"\b(?:f1|f1-score|f1 score)\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?\s*%|0?\.\d+|1\.0+)",
        re.I,
    ),
    "auc": re.compile(
        r"\b(?:roc[- ]?auc|auc)\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?\s*%|0?\.\d+|1\.0+)",
        re.I,
    ),
    "fid": re.compile(
        r"\bFID\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?)",
        re.I,
    ),
    "spd": re.compile(
        r"\bSPD\b[^\n\r]{0,80}?"
        r"(-?[0-9]+(?:\.[0-9]+)?)",
        re.I,
    ),
    "eod": re.compile(
        r"\bEOD\b[^\n\r]{0,80}?"
        r"(-?[0-9]+(?:\.[0-9]+)?)",
        re.I,
    ),
    "di": re.compile(
        r"\b(?:DI|disparate impact)\b[^\n\r]{0,80}?"
        r"([0-9]+(?:\.[0-9]+)?)",
        re.I,
    ),
}

SAMPLE_SIZE_PATTERNS = [
    re.compile(
        r"\bN\s*=\s*([0-9][0-9,]*)",
        re.I,
    ),
    re.compile(
        r"\bn\s*=\s*([0-9][0-9,]*)",
        re.I,
    ),
    re.compile(
        r"\b([0-9][0-9,]*)\s+"
        r"(?:synthetic\s+)?samples\b",
        re.I,
    ),
    re.compile(
        r"\b([0-9][0-9,]*)\s+"
        r"(?:patient|patients|records|instances|observations)\b",
        re.I,
    ),
]

FIGURE_CAPTION_RE = re.compile(
    r"^\s*(Figure|Fig\.)\s*([0-9]+)\s*[:.\-]?\s*(.*)$",
    re.I,
)

TABLE_CAPTION_RE = re.compile(
    r"^\s*Table\s*([0-9]+)\s*[:.\-]?\s*(.*)$",
    re.I,
)

DOCX_NS = {
    "w":
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",

    "a":
        "http://schemas.openxmlformats.org/drawingml/2006/main",

    "r":
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def normalize_ws(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def relative_path(path: Path) -> str:
    try:
        return str(
            path.relative_to(
                PAPER_ROOT
            )
        )
    except Exception:
        return str(path)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

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
        extras = [
            col
            for col in df.columns
            if col not in columns
        ]

        df = df[
            columns + extras
        ]

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def is_excluded_path(path: Path) -> bool:
    parts = [
        part.lower()
        for part in path.parts
    ]

    for part in parts:
        if part in EXCLUDED_DIR_NAMES:
            return True

        if any(
            part.startswith(prefix)
            for prefix in EXCLUDED_DIR_PREFIXES
        ):
            return True

    return False


def read_text_file(
    path: Path,
) -> Optional[str]:
    try:
        if (
            path.stat().st_size
            >
            MAX_TEXT_BYTES
        ):
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
            pass

    return None


def paragraph_text(
    paragraph: ET.Element,
) -> str:
    texts: List[str] = []

    for node in paragraph.findall(
        ".//w:t",
        DOCX_NS,
    ):
        if node.text:
            texts.append(
                node.text
            )

    return normalize_ws(
        " ".join(texts)
    )


# =============================================================================
# 4. MANUSCRIPT RESOLUTION
# =============================================================================

def discover_manuscripts() -> List[Path]:
    found: List[Path] = []

    if PAPER_ROOT.exists():
        for path in PAPER_ROOT.glob(
            "*.docx"
        ):
            if path.is_file():
                found.append(path)

    def rank(
        path: Path,
    ) -> Tuple[int, int, str]:
        name = path.name.lower()

        hint_rank = 999

        for index, hint in enumerate(
            MANUSCRIPT_NAME_HINTS
        ):
            if name == hint:
                hint_rank = index
                break

        try:
            mtime_rank = -int(
                path.stat().st_mtime
            )
        except Exception:
            mtime_rank = 0

        return (
            hint_rank,
            mtime_rank,
            name,
        )

    return sorted(
        set(found),
        key=rank,
    )


def choose_manuscript() -> Optional[Path]:
    candidates = discover_manuscripts()

    if not candidates:
        return None

    return candidates[0]


# =============================================================================
# 5. DOCX STRUCTURE AND FIGURES
# =============================================================================

def extract_docx_structure(
    path: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "paragraphs": [],
        "images": [],
        "captions": [],
        "tables": [],
        "full_text": "",
    }

    with zipfile.ZipFile(
        path,
        "r",
    ) as archive:
        names = set(
            archive.namelist()
        )

        if (
            "word/document.xml"
            not in names
        ):
            raise RuntimeError(
                f"DOCX has no word/document.xml: {path}"
            )

        doc_root = ET.fromstring(
            archive.read(
                "word/document.xml"
            )
        )

        rel_map: Dict[str, str] = {}

        if (
            "word/_rels/document.xml.rels"
            in names
        ):
            rel_root = ET.fromstring(
                archive.read(
                    "word/_rels/document.xml.rels"
                )
            )

            for rel in rel_root:
                rid = rel.attrib.get(
                    "Id",
                    "",
                )

                target = rel.attrib.get(
                    "Target",
                    "",
                )

                if rid and target:
                    if not target.startswith(
                        "word/"
                    ):
                        target = (
                            "word/"
                            +
                            target.lstrip("/")
                        )

                    target = str(
                        Path(target)
                    ).replace(
                        "\\",
                        "/",
                    )

                    while "word/../" in target:
                        target = target.replace(
                            "word/../",
                            "",
                        )

                    rel_map[
                        rid
                    ] = target

        body = doc_root.find(
            "w:body",
            DOCX_NS,
        )

        if body is None:
            return result

        paragraph_records: List[
            Dict[str, Any]
        ] = []

        image_occurrences: List[
            Dict[str, Any]
        ] = []

        caption_records: List[
            Dict[str, Any]
        ] = []

        table_records: List[
            Dict[str, Any]
        ] = []

        paragraph_index = 0

        for child in list(body):
            tag = child.tag.split(
                "}"
            )[-1]

            if tag == "p":
                text = paragraph_text(
                    child
                )

                paragraph_records.append(
                    {
                        "paragraph_index":
                            paragraph_index,

                        "text":
                            text,
                    }
                )

                figure_match = (
                    FIGURE_CAPTION_RE.match(
                        text
                    )
                )

                if figure_match:
                    caption_records.append(
                        {
                            "paragraph_index":
                                paragraph_index,

                            "caption_type":
                                "figure",

                            "number":
                                int(
                                    figure_match.group(
                                        2
                                    )
                                ),

                            "caption":
                                text,

                            "caption_body":
                                figure_match.group(
                                    3
                                ).strip(),
                        }
                    )

                table_match = (
                    TABLE_CAPTION_RE.match(
                        text
                    )
                )

                if table_match:
                    caption_records.append(
                        {
                            "paragraph_index":
                                paragraph_index,

                            "caption_type":
                                "table",

                            "number":
                                int(
                                    table_match.group(
                                        1
                                    )
                                ),

                            "caption":
                                text,

                            "caption_body":
                                table_match.group(
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

                    target = rel_map.get(
                        rid,
                        "",
                    )

                    media_bytes = (
                        archive.read(target)
                        if target in names
                        else b""
                    )

                    image_occurrences.append(
                        {
                            "paragraph_index":
                                paragraph_index,

                            "relationship_id":
                                rid,

                            "media_target":
                                target,

                            "sha256":
                                (
                                    sha256_bytes(
                                        media_bytes
                                    )
                                    if media_bytes
                                    else ""
                                ),

                            "bytes":
                                len(
                                    media_bytes
                                ),
                        }
                    )

                paragraph_index += 1

            elif tag == "tbl":
                rows = []

                for tr in child.findall(
                    ".//w:tr",
                    DOCX_NS,
                ):
                    cells = []

                    for tc in tr.findall(
                        "./w:tc",
                        DOCX_NS,
                    ):
                        cell_text = normalize_ws(
                            " ".join(
                                paragraph_text(p)
                                for p in tc.findall(
                                    ".//w:p",
                                    DOCX_NS,
                                )
                            )
                        )

                        cells.append(
                            cell_text
                        )

                    rows.append(cells)

                table_records.append(
                    {
                        "table_index":
                            len(
                                table_records
                            ),

                        "rows":
                            rows,

                        "text":
                            " | ".join(
                                " | ".join(row)
                                for row in rows
                            ),
                    }
                )

        figure_captions = [
            caption
            for caption in caption_records
            if (
                caption[
                    "caption_type"
                ]
                ==
                "figure"
            )
        ]

        for image in image_occurrences:
            after = [
                caption
                for caption in figure_captions
                if (
                    caption[
                        "paragraph_index"
                    ]
                    >=
                    image[
                        "paragraph_index"
                    ]
                )
            ]

            before = [
                caption
                for caption in figure_captions
                if (
                    caption[
                        "paragraph_index"
                    ]
                    <
                    image[
                        "paragraph_index"
                    ]
                )
            ]

            chosen = None

            if after:
                chosen = min(
                    after,
                    key=lambda caption:
                        (
                            caption[
                                "paragraph_index"
                            ]
                            -
                            image[
                                "paragraph_index"
                            ]
                        ),
                )

                if (
                    chosen[
                        "paragraph_index"
                    ]
                    -
                    image[
                        "paragraph_index"
                    ]
                    >
                    4
                ):
                    chosen = None

            if (
                chosen is None
                and before
            ):
                candidate = max(
                    before,
                    key=lambda caption:
                        caption[
                            "paragraph_index"
                        ],
                )

                if (
                    image[
                        "paragraph_index"
                    ]
                    -
                    candidate[
                        "paragraph_index"
                    ]
                    <=
                    3
                ):
                    chosen = candidate

            image[
                "figure_number"
            ] = (
                chosen[
                    "number"
                ]
                if chosen
                else None
            )

            image[
                "caption"
            ] = (
                chosen[
                    "caption"
                ]
                if chosen
                else ""
            )

            image[
                "caption_body"
            ] = (
                chosen[
                    "caption_body"
                ]
                if chosen
                else ""
            )

        result[
            "paragraphs"
        ] = paragraph_records

        result[
            "images"
        ] = image_occurrences

        result[
            "captions"
        ] = caption_records

        result[
            "tables"
        ] = table_records

        result[
            "full_text"
        ] = "\n".join(
            row["text"]
            for row in paragraph_records
            if row["text"]
        )

        return result


# =============================================================================
# 6. MANUSCRIPT CLAIM EXTRACTION
# =============================================================================

def split_sentences(
    text: str,
) -> List[str]:
    text = normalize_ws(
        text.replace(
            "\n",
            " ",
        )
    )

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(])",
        text,
    )

    return [
        normalize_ws(part)
        for part in parts
        if normalize_ws(part)
    ]


def extract_multimodal_claims(
    text: str,
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    for index, sentence in enumerate(
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

        if matched:
            rows.append(
                {
                    "sentence_index":
                        index,

                    "matched_terms":
                        safe_json(
                            matched
                        ),

                    "clinical_anchor_present":
                        int(
                            any(
                                term in lower
                                for term in
                                CLINICAL_ANCHOR_TERMS
                            )
                        ),

                    "sentence":
                        sentence,
                }
            )

    return rows


def extract_metric_claims(
    text: str,
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    sentences = split_sentences(
        text
    )

    for index, sentence in enumerate(
        sentences,
        start=1,
    ):
        for metric, pattern in (
            METRIC_PATTERNS.items()
        ):
            for match in pattern.finditer(
                sentence
            ):
                rows.append(
                    {
                        "sentence_index":
                            index,

                        "metric":
                            metric,

                        "value_text":
                            match.group(
                                1
                            ).strip(),

                        "sentence":
                            sentence,
                    }
                )

    return rows


def extract_sample_size_claims(
    text: str,
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    for index, sentence in enumerate(
        split_sentences(text),
        start=1,
    ):
        for pattern in (
            SAMPLE_SIZE_PATTERNS
        ):
            for match in pattern.finditer(
                sentence
            ):
                value_text = match.group(
                    1
                )

                value_num = int(
                    value_text.replace(
                        ",",
                        "",
                    )
                )

                rows.append(
                    {
                        "sentence_index":
                            index,

                        "sample_size":
                            value_num,

                        "value_text":
                            value_text,

                        "synthetic_context":
                            int(
                                "synthetic"
                                in sentence.lower()
                            ),

                        "sentence":
                            sentence,
                    }
                )

    return rows


# =============================================================================
# 7. PROJECT EVIDENCE AUDIT
# =============================================================================

def discover_project_files() -> List[Path]:
    files: List[Path] = []

    if not PROJECT_ROOT.exists():
        return files

    for path in PROJECT_ROOT.rglob(
        "*"
    ):
        if len(files) >= MAX_SCAN_FILES:
            break

        if not path.is_file():
            continue

        if is_excluded_path(path):
            continue

        files.append(path)

    return files


def classify_dataset_scope(
    path: Path,
    text: str,
) -> str:
    haystack = (
        str(path)
        +
        " "
        +
        (text or "")
    ).lower()

    if any(
        term in haystack
        for term in
        UNRELATED_DATASET_HINTS
    ):
        return (
            "UNRELATED_IMAGE_OR_SIGN_LANGUAGE_CONTEXT"
        )

    if any(
        term in haystack
        for term in
        CLINICAL_ANCHOR_TERMS
    ):
        return (
            "CLINICAL_OR_COVID_CONTEXT"
        )

    return "UNRESOLVED_CONTEXT"


def read_source_text(
    path: Path,
) -> Optional[str]:
    suffix = path.suffix.lower()

    if suffix == ".docx":
        try:
            return extract_docx_structure(
                path
            )[
                "full_text"
            ]
        except Exception:
            return None

    if suffix in TEXT_EXTENSIONS:
        return read_text_file(
            path
        )

    return None


def audit_multimodal_implementation(
    files: Sequence[Path],
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    for path in files:
        if (
            path.suffix.lower()
            not in TEXT_EXTENSIONS
        ):
            continue

        text = read_source_text(
            path
        )

        if not text:
            continue

        lower = text.lower()

        multimodal_hits = sorted(
            {
                term
                for term in
                MULTIMODAL_TERMS
                if term in lower
            }
        )

        fusion_hits = sorted(
            {
                term
                for term in
                FUSION_TERMS
                if term in lower
            }
        )

        clinical_hits = sorted(
            {
                term
                for term in
                CLINICAL_ANCHOR_TERMS
                if term in lower
            }
        )

        unrelated_hits = sorted(
            {
                term
                for term in
                UNRELATED_DATASET_HINTS
                if term in lower
            }
        )

        if (
            multimodal_hits
            or fusion_hits
        ):
            scope = (
                classify_dataset_scope(
                    path,
                    text,
                )
            )

            executable = int(
                path.suffix.lower()
                ==
                ".py"
            )

            strict_candidate = int(
                executable
                and bool(
                    multimodal_hits
                )
                and bool(
                    fusion_hits
                )
                and bool(
                    clinical_hits
                )
                and not bool(
                    unrelated_hits
                )
            )

            rows.append(
                {
                    "file":
                        relative_path(
                            path
                        ),

                    "extension":
                        path.suffix.lower(),

                    "scope_classification":
                        scope,

                    "executable_python":
                        executable,

                    "multimodal_terms":
                        safe_json(
                            multimodal_hits
                        ),

                    "fusion_terms":
                        safe_json(
                            fusion_hits
                        ),

                    "clinical_anchor_terms":
                        safe_json(
                            clinical_hits
                        ),

                    "unrelated_dataset_terms":
                        safe_json(
                            unrelated_hits
                        ),

                    "strict_clinical_multimodal_candidate":
                        strict_candidate,
                }
            )

    return rows


def audit_data_modalities(
    files: Sequence[Path],
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    accepted = (
        IMAGE_EXTENSIONS
        |
        {
            ".csv",
            ".xlsx",
            ".xls",
            ".npy",
            ".npz",
        }
    )

    for path in files:
        suffix = (
            path.suffix.lower()
        )

        if suffix not in accepted:
            continue

        path_lower = str(
            path
        ).lower()

        scope = classify_dataset_scope(
            path,
            "",
        )

        rows.append(
            {
                "file":
                    relative_path(
                        path
                    ),

                "extension":
                    suffix,

                "scope_classification":
                    scope,

                "clinical_anchor_in_path":
                    int(
                        any(
                            term in path_lower
                            for term in
                            CLINICAL_ANCHOR_TERMS
                        )
                    ),

                "unrelated_dataset_hint_in_path":
                    int(
                        any(
                            term in path_lower
                            for term in
                            UNRELATED_DATASET_HINTS
                        )
                    ),

                "candidate_image_modality":
                    int(
                        suffix
                        in IMAGE_EXTENSIONS
                    ),

                "candidate_tabular_modality":
                    int(
                        suffix
                        in {
                            ".csv",
                            ".xlsx",
                            ".xls",
                        }
                    ),

                "candidate_array_artifact":
                    int(
                        suffix
                        in {
                            ".npy",
                            ".npz",
                        }
                    ),
            }
        )

    return rows


# =============================================================================
# 8. FIGURE AUDIT
# =============================================================================

def image_metadata_from_bytes(
    data: bytes,
) -> Tuple[
    Optional[int],
    Optional[int],
    str,
]:
    if (
        not data
        or Image is None
    ):
        return (
            None,
            None,
            "",
        )

    try:
        import io

        with Image.open(
            io.BytesIO(data)
        ) as img:
            return (
                img.width,
                img.height,
                img.format or "",
            )

    except Exception:
        return (
            None,
            None,
            "",
        )


def build_figure_inventory(
    manuscript: Path,
    structure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    with zipfile.ZipFile(
        manuscript,
        "r",
    ) as archive:
        names = set(
            archive.namelist()
        )

        for index, image in enumerate(
            structure[
                "images"
            ],
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
            ) = image_metadata_from_bytes(
                data
            )

            caption = image.get(
                "caption",
                "",
            )

            caption_lower = (
                caption.lower()
            )

            rows.append(
                {
                    "image_occurrence":
                        index,

                    "paragraph_index":
                        image.get(
                            "paragraph_index"
                        ),

                    "figure_number":
                        image.get(
                            "figure_number"
                        ),

                    "caption":
                        caption,

                    "media_target":
                        target,

                    "sha256":
                        image.get(
                            "sha256",
                            "",
                        ),

                    "bytes":
                        image.get(
                            "bytes",
                            0,
                        ),

                    "width":
                        width,

                    "height":
                        height,

                    "image_format":
                        image_format,

                    "caption_mentions_confusion_matrix":
                        int(
                            (
                                "confusion"
                                in caption_lower
                            )
                            and
                            (
                                "matrix"
                                in caption_lower
                            )
                        ),

                    "caption_mentions_multimodal":
                        int(
                            any(
                                term
                                in caption_lower
                                for term in
                                MULTIMODAL_TERMS
                            )
                        ),

                    "caption_missing":
                        int(
                            not bool(
                                caption
                            )
                        ),
                }
            )

    return rows


def duplicate_figure_groups(
    figure_rows: Sequence[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    by_hash: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in figure_rows:
        image_hash = row.get(
            "sha256",
            "",
        )

        if image_hash:
            by_hash[
                image_hash
            ].append(row)

    rows: List[
        Dict[str, Any]
    ] = []

    group_id = 0

    for image_hash, group in (
        by_hash.items()
    ):
        if len(group) < 2:
            continue

        group_id += 1

        figure_numbers = [
            item.get(
                "figure_number"
            )
            for item in group
        ]

        captions = [
            item.get(
                "caption",
                "",
            )
            for item in group
        ]

        confusion_count = sum(
            int(
                item.get(
                    "caption_mentions_confusion_matrix",
                    0,
                )
            )
            for item in group
        )

        rows.append(
            {
                "duplicate_group":
                    group_id,

                "sha256":
                    image_hash,

                "occurrences":
                    len(group),

                "figure_numbers":
                    safe_json(
                        figure_numbers
                    ),

                "captions":
                    safe_json(
                        captions
                    ),

                "confusion_matrix_caption_count":
                    confusion_count,

                "requires_manual_review":
                    1,

                "reason":
                    (
                        "Exact embedded-image duplicate. "
                        "If figure numbers or captions describe different "
                        "experiments/modalities, the manuscript is internally "
                        "inconsistent."
                    ),
            }
        )

    return rows


def audit_figure_captions(
    structure: Dict[str, Any],
    figure_rows: Sequence[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    figure_captions = [
        caption
        for caption in structure[
            "captions"
        ]
        if (
            caption[
                "caption_type"
            ]
            ==
            "figure"
        )
    ]

    number_counts: Dict[
        int,
        int,
    ] = defaultdict(int)

    for caption in (
        figure_captions
    ):
        number_counts[
            caption[
                "number"
            ]
        ] += 1

    image_figures = {
        row.get(
            "figure_number"
        )
        for row in figure_rows
        if (
            row.get(
                "figure_number"
            )
            is not None
        )
    }

    for caption in (
        figure_captions
    ):
        issues: List[str] = []

        lower = caption[
            "caption"
        ].lower()

        if (
            number_counts[
                caption[
                    "number"
                ]
            ]
            >
            1
        ):
            issues.append(
                "DUPLICATE_FIGURE_NUMBER"
            )

        if (
            caption[
                "number"
            ]
            not in image_figures
        ):
            issues.append(
                "CAPTION_WITHOUT_NEARBY_EMBEDDED_IMAGE"
            )

        if (
            "confusion"
            in lower
            and
            "matrix"
            in lower
        ):
            issues.append(
                "CONFUSION_MATRIX_CAPTION_REVIEW"
            )

        if any(
            term in lower
            for term in
            MULTIMODAL_TERMS
        ):
            issues.append(
                "MULTIMODAL_CAPTION_REQUIRES_IMPLEMENTATION_SUPPORT"
            )

        rows.append(
            {
                "figure_number":
                    caption[
                        "number"
                    ],

                "paragraph_index":
                    caption[
                        "paragraph_index"
                    ],

                "caption":
                    caption[
                        "caption"
                    ],

                "issues":
                    safe_json(
                        issues
                    ),

                "issue_count":
                    len(
                        issues
                    ),
            }
        )

    return rows


# =============================================================================
# 9. PRIOR AUDIT CROSS-CHECK
# =============================================================================

def find_prior_audit_evidence(
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    outputs_root = (
        PROJECT_ROOT
        / "outputs"
    )

    if not outputs_root.exists():
        return rows

    target_terms = {
        "generator_missing": (
            "CLASSIFIER_OR_ENCODER_ONLY_NO_STRUCTURED_GENERATOR",
            "NO_STRUCTURED_GENERATOR",
            "DO_NOT_REGENERATE_REMOVE_UNSUPPORTED_GENERATIVE_FID_CLAIMS",
        ),

        "fid_unsupported": (
            "NO_REPRODUCIBLE_FID_OR_STRUCTURED_FRECHET_RESULT_AVAILABLE",
            "NO_REPRODUCIBLE_FID",
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

    for path in outputs_root.rglob(
        "*"
    ):
        if not path.is_file():
            continue

        if is_excluded_path(path):
            continue

        if (
            path.suffix.lower()
            not in {
                ".txt",
                ".csv",
                ".log",
                ".md",
                ".json",
            }
        ):
            continue

        text = read_text_file(
            path
        )

        if not text:
            continue

        upper = text.upper()

        for category, terms in (
            target_terms.items()
        ):
            matched = [
                term
                for term in terms
                if (
                    term.upper()
                    in upper
                )
            ]

            if matched:
                rows.append(
                    {
                        "category":
                            category,

                        "file":
                            relative_path(
                                path
                            ),

                        "matched_terms":
                            safe_json(
                                matched
                            ),
                    }
                )

    return rows


# =============================================================================
# 10. CONSISTENCY FINDINGS
# =============================================================================

def unique_metric_values(
    metric_rows: Sequence[
        Dict[str, Any]
    ],
    metric: str,
) -> List[str]:
    values = []

    for row in metric_rows:
        if row.get(
            "metric"
        ) == metric:
            values.append(
                str(
                    row.get(
                        "value_text",
                        "",
                    )
                ).strip()
            )

    return sorted(
        set(values)
    )


def build_consistency_findings(
    multimodal_claims: Sequence[Dict[str, Any]],
    implementation_rows: Sequence[Dict[str, Any]],
    data_rows: Sequence[Dict[str, Any]],
    duplicate_rows: Sequence[Dict[str, Any]],
    caption_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    sample_rows: Sequence[Dict[str, Any]],
    prior_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    findings: List[
        Dict[str, Any]
    ] = []

    strict_multimodal = [
        row
        for row in implementation_rows
        if (
            row.get(
                "strict_clinical_multimodal_candidate"
            )
            ==
            1
        )
    ]

    clinical_images = [
        row
        for row in data_rows
        if (
            row.get(
                "candidate_image_modality"
            )
            ==
            1
            and
            row.get(
                "scope_classification"
            )
            ==
            "CLINICAL_OR_COVID_CONTEXT"
        )
    ]

    unrelated_images = [
        row
        for row in data_rows
        if (
            row.get(
                "candidate_image_modality"
            )
            ==
            1
            and
            row.get(
                "scope_classification"
            )
            ==
            "UNRELATED_IMAGE_OR_SIGN_LANGUAGE_CONTEXT"
        )
    ]

    prior_categories = {
        row[
            "category"
        ]
        for row in prior_rows
    }

    findings.append(
        {
            "reviewer_comment":
                "C13-C14",

            "topic":
                "multimodal_scope_and_reproducibility",

            "status":
                (
                    "SUPPORTED"
                    if strict_multimodal
                    else
                    "NOT_REPRODUCIBLE_FROM_RECOVERED_PROJECT"
                ),

            "evidence":
                (
                    f"Manuscript multimodal claim sentences="
                    f"{len(multimodal_claims)}; "
                    f"strict clinical multimodal implementation candidates="
                    f"{len(strict_multimodal)}; "
                    f"clinical image artifacts="
                    f"{len(clinical_images)}; "
                    f"unrelated image/sign-language artifacts="
                    f"{len(unrelated_images)}."
                ),

            "manuscript_action":
                (
                    "Retain multimodal claims only if a concrete clinical "
                    "multimodal implementation, aligned modalities, "
                    "preprocessing, fusion, and result provenance can be "
                    "demonstrated. Otherwise remove or explicitly reframe "
                    "the work as structured clinical/tabular data."
                ),
        }
    )

    findings.append(
        {
            "reviewer_comment":
                "C15",

            "topic":
                "duplicate_figures_and_confusion_matrices",

            "status":
                (
                    "REVIEW_REQUIRED"
                    if duplicate_rows
                    else
                    "NO_EXACT_DUPLICATE_EMBEDDED_IMAGES_DETECTED"
                ),

            "evidence":
                (
                    f"Exact duplicate embedded-image groups="
                    f"{len(duplicate_rows)}."
                ),

            "manuscript_action":
                (
                    "For each duplicate group, verify that repeated images "
                    "are intentional. Remove duplicate confusion matrices "
                    "and correct captions when a repeated image is presented "
                    "as a different experiment or modality."
                ),
        }
    )

    accuracy_values = (
        unique_metric_values(
            metric_rows,
            "accuracy",
        )
    )

    findings.append(
        {
            "reviewer_comment":
                "C22",

            "topic":
                "accuracy_and_confusion_consistency",

            "status":
                "REVIEW_REQUIRED",

            "evidence":
                (
                    "Distinct manuscript accuracy value strings detected="
                    f"{safe_json(accuracy_values)}. "
                    "Confusion-related figure captions require cross-check "
                    "against their source counts."
                ),

            "manuscript_action":
                (
                    "Recalculate every accuracy stated next to a confusion "
                    "matrix as (TP + TN) / N for the same evaluation "
                    "condition. Retain only values tied to a documented "
                    "evaluation condition."
                ),
        }
    )

    sample_sizes = sorted(
        {
            int(
                row[
                    "sample_size"
                ]
            )
            for row in sample_rows
        }
    )

    findings.append(
        {
            "reviewer_comment":
                "C23",

            "topic":
                "synthetic_sample_size_interpretation",

            "status":
                (
                    "REVIEW_REQUIRED"
                    if any(
                        value >= 500
                        for value in
                        sample_sizes
                    )
                    else
                    "NO_LARGE_SAMPLE_SIZE_CLAIM_DETECTED"
                ),

            "evidence":
                (
                    "Detected sample-size claims="
                    f"{safe_json(sample_sizes)}."
                ),

            "manuscript_action":
                (
                    "Any increase from the original clinical cohort to "
                    "hundreds or thousands of synthetic rows must be "
                    "described as synthetic computational workload, not an "
                    "increase in independent patient information or "
                    "effective clinical sample size."
                ),
        }
    )

    fid_values = unique_metric_values(
        metric_rows,
        "fid",
    )

    fid_prior_unsupported = (
        "fid_unsupported"
        in prior_categories
        or
        "generator_missing"
        in prior_categories
    )

    findings.append(
        {
            "reviewer_comment":
                "C24",

            "topic":
                "fid_consistency_and_reproducibility",

            "status":
                (
                    "UNSUPPORTED"
                    if fid_prior_unsupported
                    else
                    (
                        "REVIEW_REQUIRED"
                        if len(
                            fid_values
                        )
                        >
                        1
                        else
                        "NO_PRIOR_UNSUPPORTED_FID_VERDICT_FOUND"
                    )
                ),

            "evidence":
                (
                    "Distinct manuscript FID value strings="
                    f"{safe_json(fid_values)}; "
                    "prior audit marks FID/generator unsupported="
                    f"{int(fid_prior_unsupported)}."
                ),

            "manuscript_action":
                (
                    "Remove absolute FID claims unless the feature space and "
                    "synthetic-data provenance are recovered. For structured "
                    "data, do not call a raw-feature Fréchet distance "
                    "conventional image FID."
                ),
        }
    )

    findings.append(
        {
            "reviewer_comment":
                "C13-C15,C22-C24",

            "topic":
                "overall_results_integrity",

            "status":
                (
                    "MAJOR_REFRAME_REQUIRED"
                    if (
                        not strict_multimodal
                        or fid_prior_unsupported
                    )
                    else
                    "TARGETED_CORRECTIONS_REQUIRED"
                ),

            "evidence":
                (
                    f"strict clinical multimodal implementation="
                    f"{len(strict_multimodal)}; "
                    f"duplicate groups="
                    f"{len(duplicate_rows)}; "
                    f"FID prior unsupported="
                    f"{int(fid_prior_unsupported)}; "
                    f"caption issues="
                    f"{sum(int(row.get('issue_count', 0)) for row in caption_rows)}."
                ),

            "manuscript_action":
                (
                    "Consolidate final results around only reproducible "
                    "structured-clinical evidence. Remove unsupported "
                    "multimodal, generator, FID, scalability, and synthetic-"
                    "sample-size interpretations unless direct provenance "
                    "exists."
                ),
        }
    )

    return findings


# =============================================================================
# 11. VERIFICATION MATRIX
# =============================================================================

def build_verification_matrix(
    manuscript: Optional[Path],
    structure: Dict[str, Any],
    figure_rows: Sequence[Dict[str, Any]],
    duplicate_rows: Sequence[Dict[str, Any]],
    multimodal_claims: Sequence[Dict[str, Any]],
    implementation_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    sample_rows: Sequence[Dict[str, Any]],
    prior_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    strict_candidates = [
        row
        for row in implementation_rows
        if (
            row.get(
                "strict_clinical_multimodal_candidate"
            )
            ==
            1
        )
    ]

    prior_categories = {
        row[
            "category"
        ]
        for row in prior_rows
    }

    fid_values = unique_metric_values(
        metric_rows,
        "fid",
    )

    figure_caption_count = len(
        [
            caption
            for caption in structure.get(
                "captions",
                [],
            )
            if (
                caption.get(
                    "caption_type"
                )
                ==
                "figure"
            )
        ]
    )

    return [
        {
            "criterion":
                "manuscript_resolved",

            "passed":
                int(
                    manuscript is not None
                ),

            "detail":
                (
                    relative_path(
                        manuscript
                    )
                    if manuscript
                    else
                    "No manuscript DOCX resolved."
                ),
        },

        {
            "criterion":
                "figure_inventory_completed",

            "passed":
                int(
                    manuscript is not None
                ),

            "detail":
                (
                    f"Embedded image occurrences="
                    f"{len(figure_rows)}; "
                    f"figure captions="
                    f"{figure_caption_count}."
                ),
        },

        {
            "criterion":
                "exact_duplicate_images_audited",

            "passed":
                1,

            "detail":
                (
                    f"Exact duplicate embedded-image groups="
                    f"{len(duplicate_rows)}."
                ),
        },

        {
            "criterion":
                "multimodal_claims_identified",

            "passed":
                1,

            "detail":
                (
                    "Multimodal-related manuscript sentences identified="
                    f"{len(multimodal_claims)}."
                ),
        },

        {
            "criterion":
                "clinical_multimodal_implementation_verified",

            "passed":
                int(
                    bool(
                        strict_candidates
                    )
                ),

            "detail":
                (
                    "Strict executable clinical multimodal+fusion candidates="
                    f"{len(strict_candidates)}."
                ),
        },

        {
            "criterion":
                "fid_claims_cross_checked",

            "passed":
                int(
                    (
                        "fid_unsupported"
                        in prior_categories
                    )
                    or
                    (
                        "generator_missing"
                        in prior_categories
                    )
                    or
                    not fid_values
                ),

            "detail":
                (
                    f"FID values="
                    f"{safe_json(fid_values)}; "
                    "prior unsupported/generator-missing evidence="
                    f"{int('fid_unsupported' in prior_categories or 'generator_missing' in prior_categories)}."
                ),
        },

        {
            "criterion":
                "sample_size_claims_inventory_completed",

            "passed":
                1,

            "detail":
                (
                    f"Sample-size claim rows="
                    f"{len(sample_rows)}."
                ),
        },

        {
            "criterion":
                "accuracy_metric_claims_inventory_completed",

            "passed":
                1,

            "detail":
                (
                    f"Accuracy claim rows="
                    f"{len([row for row in metric_rows if row.get('metric') == 'accuracy'])}."
                ),
        },
    ]


# =============================================================================
# 12. FINAL VERDICT
# =============================================================================

def build_verdict(
    findings: Sequence[Dict[str, Any]],
    matrix: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    clinical_multimodal_verified = any(
        (
            row.get(
                "criterion"
            )
            ==
            "clinical_multimodal_implementation_verified"
        )
        and
        (
            row.get(
                "passed"
            )
            ==
            1
        )
        for row in matrix
    )

    fid_unsupported = any(
        (
            row.get(
                "topic"
            )
            ==
            "fid_consistency_and_reproducibility"
        )
        and
        (
            row.get(
                "status"
            )
            ==
            "UNSUPPORTED"
        )
        for row in findings
    )

    if (
        not clinical_multimodal_verified
        or fid_unsupported
    ):
        verdict = (
            "MULTIMODAL_AND_GENERATIVE_RESULT_CLAIMS_REQUIRE_REMOVAL_OR_MAJOR_REFRAMING"
        )

        next_action = (
            "RETAIN_ONLY_REPRODUCIBLE_STRUCTURED_CLINICAL_RESULTS_AND_CORRECT_FIGURES_CAPTIONS_NUMBERS"
        )

    else:
        verdict = (
            "TARGETED_FIGURE_AND_RESULT_CONSISTENCY_CORRECTIONS_REQUIRED"
        )

        next_action = (
            "CORRECT_DUPLICATES_CAPTIONS_AND_NUMERICAL_INCONSISTENCIES"
        )

    return {
        "verdict":
            verdict,

        "next_action":
            next_action,

        "new_training_performed":
            False,

        "new_synthetic_generation_performed":
            False,

        "ocr_performed":
            False,

        "historical_files_modified":
            False,
    }


# =============================================================================
# 13. MAIN
# =============================================================================

def main() -> None:
    print(
        "=" * 100
    )

    print(
        "HFAGM - 09 MULTIMODAL / FIGURE / RESULTS CONSISTENCY AUDIT"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "Restrictions:"
    )

    print(
        "  - no model training"
    )

    print(
        "  - no synthetic generation"
    )

    print(
        "  - no OCR"
    )

    print(
        "  - no image modification"
    )

    print(
        "  - no historical artifact modification"
    )

    print(
        "  - exact-image hashes are used only for duplicate detection"
    )

    manuscript = (
        choose_manuscript()
    )

    if manuscript is None:
        raise RuntimeError(
            f"No manuscript DOCX found directly under {PAPER_ROOT}. "
            "Place the current manuscript there or update "
            "PAPER_ROOT/MANUSCRIPT_NAME_HINTS."
        )

    print()
    print(
        "Selected manuscript:"
    )

    print(
        manuscript
    )

    structure = (
        extract_docx_structure(
            manuscript
        )
    )

    manuscript_text = (
        structure[
            "full_text"
        ]
    )

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

    project_files = (
        discover_project_files()
    )

    multimodal_impl = (
        audit_multimodal_implementation(
            project_files
        )
    )

    data_modalities = (
        audit_data_modalities(
            project_files
        )
    )

    figure_inventory = (
        build_figure_inventory(
            manuscript,
            structure,
        )
    )

    duplicate_groups = (
        duplicate_figure_groups(
            figure_inventory
        )
    )

    caption_audit = (
        audit_figure_captions(
            structure,
            figure_inventory,
        )
    )

    prior_evidence = (
        find_prior_audit_evidence()
    )

    findings = (
        build_consistency_findings(
            multimodal_claims,
            multimodal_impl,
            data_modalities,
            duplicate_groups,
            caption_audit,
            metric_claims,
            sample_claims,
            prior_evidence,
        )
    )

    matrix = (
        build_verification_matrix(
            manuscript,
            structure,
            figure_inventory,
            duplicate_groups,
            multimodal_claims,
            multimodal_impl,
            metric_claims,
            sample_claims,
            prior_evidence,
        )
    )

    verdict = (
        build_verdict(
            findings,
            matrix,
        )
    )

    write_csv(
        OUTPUT_DIR
        / "manuscript_multimodal_claims.csv",
        multimodal_claims,
    )

    write_csv(
        OUTPUT_DIR
        / "manuscript_metric_claims.csv",
        metric_claims,
    )

    write_csv(
        OUTPUT_DIR
        / "manuscript_sample_size_claims.csv",
        sample_claims,
    )

    write_csv(
        OUTPUT_DIR
        / "multimodal_implementation_evidence.csv",
        multimodal_impl,
    )

    write_csv(
        OUTPUT_DIR
        / "data_modality_inventory.csv",
        data_modalities,
    )

    write_csv(
        OUTPUT_DIR
        / "figure_inventory.csv",
        figure_inventory,
    )

    write_csv(
        OUTPUT_DIR
        / "duplicate_figure_groups.csv",
        duplicate_groups,
    )

    write_csv(
        OUTPUT_DIR
        / "figure_caption_audit.csv",
        caption_audit,
    )

    write_csv(
        OUTPUT_DIR
        / "prior_audit_crosscheck.csv",
        prior_evidence,
    )

    write_csv(
        OUTPUT_DIR
        / "results_consistency_findings.csv",
        findings,
    )

    write_csv(
        OUTPUT_DIR
        / "multimodal_figure_verification_matrix.csv",
        matrix,
    )

    write_csv(
        OUTPUT_DIR
        / "multimodal_figure_verdict.csv",
        [
            verdict
        ],
    )

    provenance = {
        "generated":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "script":
            "09_multimodal_figure_results_audit.py",

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "paper_root":
            str(
                PAPER_ROOT
            ),

        "selected_manuscript":
            str(
                manuscript
            ),

        "manuscript_sha256":
            sha256_file(
                manuscript
            ),

        "project_files_scanned":
            len(
                project_files
            ),

        "manuscript_multimodal_claims":
            len(
                multimodal_claims
            ),

        "manuscript_metric_claims":
            len(
                metric_claims
            ),

        "manuscript_sample_size_claims":
            len(
                sample_claims
            ),

        "embedded_image_occurrences":
            len(
                figure_inventory
            ),

        "exact_duplicate_figure_groups":
            len(
                duplicate_groups
            ),

        "multimodal_implementation_rows":
            len(
                multimodal_impl
            ),

        "strict_clinical_multimodal_candidates":
            sum(
                int(
                    row.get(
                        "strict_clinical_multimodal_candidate",
                        0,
                    )
                )
                for row in
                multimodal_impl
            ),

        "new_training_performed":
            False,

        "new_synthetic_generation_performed":
            False,

        "ocr_performed":
            False,

        "historical_files_modified":
            False,

        "verdict":
            verdict[
                "verdict"
            ],

        "python_version":
            sys.version,

        "pandas_version":
            pd.__version__,
    }

    write_csv(
        OUTPUT_DIR
        / "multimodal_figure_provenance.csv",
        [
            provenance
        ],
    )

    lines = [
        "=" * 100,
        (
            "HFAGM - 09 MULTIMODAL / FIGURE / "
            "RESULTS CONSISTENCY AUDIT"
        ),
        "=" * 100,
        "",
        (
            f"Generated: "
            f"{provenance['generated']}"
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
        "SCOPE",
        "-" * 100,
        (
            "Reviewer #3 C13-C15 and C22-C24: "
            "multimodal scope/reproducibility, duplicate or mismatched "
            "figures, confusion-matrix consistency, synthetic-N "
            "interpretation, and FID consistency."
        ),
        "",
        "COUNTS",
        "-" * 100,
        (
            f"Project files scanned: "
            f"{len(project_files)}"
        ),
        (
            f"Manuscript multimodal-related sentences: "
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
            f"Embedded image occurrences: "
            f"{len(figure_inventory)}"
        ),
        (
            f"Exact duplicate embedded-image groups: "
            f"{len(duplicate_groups)}"
        ),
        (
            f"Multimodal implementation evidence rows: "
            f"{len(multimodal_impl)}"
        ),
        (
            "Strict executable clinical multimodal+fusion candidates: "
            f"{provenance['strict_clinical_multimodal_candidates']}"
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
            if int(
                row.get(
                    "passed",
                    0,
                )
            ) == 1
            else
            "MISSING/NO"
        )

        lines.append(
            f"{state}: "
            f"{row['criterion']}"
        )

        lines.append(
            f"    "
            f"{row['detail']}"
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
            "INTERPRETATION RULES",
            "-" * 100,
            (
                "1. An image dataset elsewhere in the repository does not "
                "establish multimodal clinical evaluation."
            ),
            (
                "2. A multimodal claim is reproducible only when the "
                "clinical modalities, alignment, preprocessing, fusion "
                "operation, and result provenance are recoverable."
            ),
            (
                "3. Exact duplicate image hashes flag repeated embedded "
                "figures but do not by themselves prove an error; captions "
                "and experimental meaning must be checked."
            ),
            (
                "4. Accuracy associated with a confusion matrix must equal "
                "(TP + TN) / N for the same evaluation condition."
            ),
            (
                "5. Synthetic row count is computational workload and does "
                "not increase independent patient information beyond the "
                "original cohort."
            ),
            (
                "6. Historical FID values remain unsupported when no "
                "verified structured generator and feature-space provenance "
                "are recoverable."
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
        / "multimodal_figure_results_audit_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "09 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Manuscript multimodal claims: "
        f"{len(multimodal_claims)}"
    )

    print(
        "Strict clinical multimodal implementation candidates: "
        f"{provenance['strict_clinical_multimodal_candidates']}"
    )

    print(
        f"Embedded image occurrences: "
        f"{len(figure_inventory)}"
    )

    print(
        f"Exact duplicate figure groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Metric claim rows: "
        f"{len(metric_claims)}"
    )

    print(
        f"Sample-size claim rows: "
        f"{len(sample_claims)}"
    )

    print()
    print(
        "FINAL VERDICT:"
    )

    print(
        verdict[
            "verdict"
        ]
    )

    print()
    print(
        "NEXT ACTION:"
    )

    print(
        verdict[
            "next_action"
        ]
    )

    print()
    print(
        "Results written to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Upload these files first:"
    )

    for filename in [
        "multimodal_figure_results_audit_summary.txt",
        "multimodal_figure_verdict.csv",
        "multimodal_figure_verification_matrix.csv",
        "results_consistency_findings.csv",
        "figure_inventory.csv",
        "duplicate_figure_groups.csv",
        "figure_caption_audit.csv",
        "manuscript_multimodal_claims.csv",
        "manuscript_metric_claims.csv",
        "manuscript_sample_size_claims.csv",
        "multimodal_implementation_evidence.csv",
        "prior_audit_crosscheck.csv",
        "multimodal_figure_provenance.csv",
    ]:
        print(
            OUTPUT_DIR
            / filename
        )


# =============================================================================
# 14. SAFE EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print(
            "=" * 100
        )

        print(
            "09 FAILED SAFELY"
        )

        print(
            "=" * 100
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "No model was trained, no synthetic data were generated, "
            "no OCR was performed, and no historical artifact was modified."
        )

        print()

        traceback.print_exc()

        sys.exit(1)