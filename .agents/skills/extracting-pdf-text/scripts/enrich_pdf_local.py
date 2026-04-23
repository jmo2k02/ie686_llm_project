#!/usr/bin/env python3
"""
Build a traceable local PDF enrichment pipeline.

This script:
- extracts raw markdown per page using pymupdf4llm
- extracts embedded images with stable page / image / xref identifiers
- runs local Tesseract OCR when available
- creates a manifest and a vision review queue
- writes an enriched markdown file with reinjected image blocks

Usage:
    uv run enrich_pdf_local.py input.pdf
    uv run enrich_pdf_local.py input.pdf --force --ocr-mode auto

Outputs (next to the PDF by default):
    input.md                raw markdown
    input.enriched.md       markdown with injected image blocks
    input.assets/
      manifest.json
      review_queue.json
      images/
      ocr/
      pages/

The script keeps the raw markdown as a source artifact and writes the enriched
markdown separately to make review and reruns safe.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pymupdf4llm>=0.0.17",
#   "pymupdf>=1.25.0",
#   "Pillow>=10.0.0",
#   "pytesseract>=0.3.10",
# ]
# ///

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

PLACEHOLDER_RE = re.compile(
    r"\*\*==> picture \[(\d+) x (\d+)\] intentionally omitted <==\*\*"
)
BLOCK_START_RE = re.compile(r"<!-- image-block:start image_id=(.+?) -->")
PICTURE_TEXT_BLOCK_RE = re.compile(
    r"\*\*----- Start of picture text -----\*\*<br>\s*.*?\*\*----- End of picture text -----\*\*<br>\s*",
    re.MULTILINE | re.DOTALL,
)
DATE_LINE_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z .'-]+,\s+)?\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*$"
)
GENERIC_SOURCE_LINE_RE = re.compile(r"^Source:\s+[a-z]\.?$", re.IGNORECASE)
GENERIC_DATE_SOURCE_LINE_RE = re.compile(
    r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s+Source:\s+[a-z]\.?$",
    re.IGNORECASE,
)
PLACEHOLDER_FOOTER_LINE_RE = re.compile(
    r"^(?:Beispiel.?Fu.szeile\s+)?TT\.MM\.JJJJ$",
    re.IGNORECASE,
)
NUMBERED_MARKER_RE = re.compile(r"^[^A-Za-z0-9]*[A-Za-z]?([0-9]{1,2})[^A-Za-z0-9]*$")
RUNNING_HEADER_RE = re.compile(
    r"^(?:Industrial AI|AI in|Lecture|Chapter|Section|Page|Slide)\s*\d*\s*$",
    re.IGNORECASE,
)
VISION_SCHEMA_VERSION = "1.2"
VISION_SYSTEM_PROMPT = (
    "You are a conservative analyst for extracted PDF visuals. Your job is to create reliable JSON sidecars that preserve the actual visible content of each figure for downstream agents. "
    "Accuracy matters more than completeness. Only claim what is visually supported. Do not guess proper nouns, places, institutions, products, numbers, or technical terms. "
    "CRITICAL FOR CHARTS/GRAPHS: You MUST extract: (1) ALL axis labels with units, (2) ALL legend entries exactly as written, (3) data point labels and values if visible, (4) colorbar/gradient values and increments. "
    "CRITICAL FOR LISTS: If the image contains a numbered list, checklist, steps, grid of recommendations, or labeled callouts, enumerate EVERY readable item in order inside structured_items. NEVER summarize as 'a numbered list of X items'. "
    "For diagrams, capture ALL labeled components, their relationships, and any text inside shapes. For tables, capture headers and ALL key cells when readable. For screenshots, capture the visible UI text and states that matter. "
    "Use summary for a concrete one-sentence statement of what is shown with SPECIFIC content. Put readable labels and longer extracted text in visible_text, and use structured_items for ordered or grouped content. "
    "NEVER summarize when you can enumerate. If you see 'S-Naive, ETS, LSTM-CL' in a legend, put those EXACT strings in structured_items, not 'various forecasting methods'. "
    "When content is unclear, mark the unclear portion as [unclear] rather than guessing. If the image is mostly decorative, duplicated branding, or layout chrome, say that briefly. Return valid JSON only and follow the schema exactly."
)
DEFAULT_VISION_PROMPT = (
    "Task: Analyze this PDF image and write an EXPLANATORY description for educational context. "
    "Your goal is to help a reader UNDERSTAND what the image shows and WHY it matters, not just list its contents. "
    "\n\n"
    "INSTRUCTIONS:\n"
    "1. EXPLAIN the image's purpose and what it demonstrates or teaches\n"
    "2. Describe the key insights, relationships, or patterns visible\n"
    "3. For CHARTS: Explain what the data shows, trends, comparisons, or conclusions that can be drawn\n"
    "4. For DIAGRAMS: Explain the process, workflow, architecture, or relationships being illustrated\n"
    "5. For PHOTOS: Describe the scene, context, and relevance to the educational content\n"
    "6. Include specific values, labels, and text only when they support understanding\n"
    "\n"
    "Write a clear, educational description (2-4 sentences) that helps someone understand the image "
    "without seeing it. Focus on MEANING over mechanics. Avoid generic phrases like 'The image shows...' - "
    "start directly with what the image communicates."
)


@dataclass
class OcrResult:
    status: str
    text: str
    mean_confidence: float | None
    best_psm: int | None
    words: list[dict[str, Any]]
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "text": self.text,
            "mean_confidence": self.mean_confidence,
            "best_psm": self.best_psm,
            "words": self.words,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "OcrResult":
        return cls(
            status=str(payload.get("status") or "unavailable"),
            text=str(payload.get("text") or ""),
            mean_confidence=(
                float(payload["mean_confidence"])
                if payload.get("mean_confidence") is not None
                else None
            ),
            best_psm=(
                int(payload["best_psm"])
                if payload.get("best_psm") is not None
                else None
            ),
            words=list(payload.get("words") or []),
            note=(str(payload["note"]) if payload.get("note") is not None else None),
        )


@dataclass
class VisionResult:
    status: str
    payload: dict[str, Any] | None
    note: str | None = None


def hash_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traceable local PDF enrichment pipeline"
    )
    parser.add_argument("pdf", help="Path to the source PDF")
    parser.add_argument("--markdown", help="Path for raw markdown output")
    parser.add_argument("--enriched-markdown", help="Path for enriched markdown output")
    parser.add_argument("--assets-dir", help="Directory for sidecar assets")
    parser.add_argument(
        "--lang", default="eng", help="Tesseract language (default: eng)"
    )
    parser.add_argument(
        "--ocr-mode",
        choices=("auto", "all", "none"),
        default="auto",
        help="How aggressively to run OCR on extracted images",
    )
    parser.add_argument(
        "--min-dimension",
        type=int,
        default=48,
        help="Skip OCR for images smaller than this on both sides in auto mode",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing assets and markdown outputs",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory for reusable OCR / vision cache entries",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the entire cache directory before this run",
    )
    parser.add_argument(
        "--clear-cache-key",
        action="append",
        default=[],
        help="Clear a specific cache key (typically an image sha256); may be passed multiple times",
    )
    parser.add_argument(
        "--vision-backend",
        choices=("manual", "openrouter"),
        default="manual",
        help="Vision enrichment backend (default: manual review queue only)",
    )
    parser.add_argument(
        "--vision-model",
        help="OpenRouter model for selective visual explanations (CLI overrides .env)",
    )
    parser.add_argument(
        "--vision-max-images",
        type=int,
        default=0,
        help="Limit OpenRouter calls per run (0 means no explicit limit)",
    )
    parser.add_argument(
        "--vision-only-image-id",
        action="append",
        default=[],
        help="Restrict OpenRouter analysis to specific image_id values; may be passed multiple times",
    )
    parser.add_argument(
        "--vision-prompt-file",
        help="Optional text file overriding the default OpenRouter vision prompt",
    )
    parser.add_argument(
        "--vision-cache-mode",
        choices=("use", "refresh", "bypass"),
        default="use",
        help="OpenRouter vision cache behavior: use cache by default, refresh to force rerun and rewrite, bypass for dev-only no-cache runs",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slugify_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def find_workspace_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return path.parent


def derive_paths(pdf_path: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    markdown_path = (
        Path(args.markdown) if args.markdown else pdf_path.with_suffix(".md")
    )
    enriched_path = (
        Path(args.enriched_markdown)
        if args.enriched_markdown
        else pdf_path.with_suffix(".enriched.md")
    )
    assets_dir = (
        Path(args.assets_dir)
        if args.assets_dir
        else pdf_path.with_suffix("").with_suffix(".assets")
    )
    return markdown_path, enriched_path, assets_dir


def derive_cache_dir(pdf_path: Path, args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir)
    workspace_root = find_workspace_root(pdf_path.resolve())
    return workspace_root / ".pdf-enrich-cache"


def serialize_rect(rect: Any) -> list[float]:
    return [round(float(value), 3) for value in rect]


def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            value.startswith(('"', "'"))
            and value.endswith(('"', "'"))
            and len(value) >= 2
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_vision_prompt(args: argparse.Namespace) -> str:
    if not args.vision_prompt_file:
        return DEFAULT_VISION_PROMPT
    return Path(args.vision_prompt_file).read_text(encoding="utf-8").strip()


def prepare_cache_dir(cache_dir: Path, args: argparse.Namespace) -> None:
    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subdir in (cache_dir / "ocr", cache_dir / "vision"):
        subdir.mkdir(parents=True, exist_ok=True)
    for key in args.clear_cache_key:
        normalized = key.strip()
        if not normalized:
            continue
        for extension in ("json", "txt"):
            target = cache_dir / "ocr" / f"{normalized}.{extension}"
            if target.exists():
                target.unlink()
        vision_target = cache_dir / "vision" / f"{normalized}.json"
        if vision_target.exists():
            vision_target.unlink()


def resolve_tesseract_binary() -> str | None:
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        "/home/linuxbrew/.linuxbrew/bin/tesseract",
        str(Path.home() / ".local/bin/tesseract"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def check_tesseract(lang: str) -> tuple[str | None, str]:
    binary = resolve_tesseract_binary()
    if not binary:
        return None, "tesseract binary not found"
    try:
        pytesseract = importlib.import_module("pytesseract")
        pytesseract.pytesseract.tesseract_cmd = binary

        langs = set(pytesseract.get_languages(config=""))
    except Exception as exc:  # pragma: no cover - defensive runtime check
        return None, f"failed to query tesseract languages: {exc}"
    if lang not in langs:
        return None, f"tesseract language '{lang}' not installed"
    return binary, f"tesseract available at {binary}"


def classify_image(
    info: dict[str, Any], duplicate_of: str | None, min_dimension: int
) -> dict[str, Any]:
    width = int(info["width"])
    height = int(info["height"])
    area = width * height
    repeated = duplicate_of is not None
    tiny = width < min_dimension and height < min_dimension
    likely_logo = repeated and width <= 260 and height <= 120
    extreme_banner = max(width / max(height, 1), height / max(width, 1)) > 8
    likely_text_candidate = area >= 12000 and not tiny and not likely_logo
    likely_visual_candidate = area >= 25000 and not tiny
    skip_reason = None
    if tiny:
        skip_reason = "tiny"
    elif likely_logo:
        skip_reason = "duplicate_logo_or_badge"
    return {
        "tiny": tiny,
        "repeated": repeated,
        "duplicate_of": duplicate_of,
        "likely_logo": likely_logo,
        "extreme_banner": extreme_banner,
        "likely_text_candidate": likely_text_candidate,
        "likely_visual_candidate": likely_visual_candidate,
        "skip_reason": skip_reason,
    }


def detect_fixed_image_digests(
    images: list[dict[str, Any]], page_count: int, ocr_dir: Path
) -> set[str]:
    stats_by_digest: dict[str, dict[str, Any]] = {}
    for image in images:
        digest = str(image["digest_sha256"])
        ocr = image["ocr"]
        sample_ocr_text = ""
        ocr_file = ocr.get("text_file")
        if ocr_file:
            raw_path = ocr_dir / Path(str(ocr_file)).name
            if raw_path.exists():
                sample_ocr_text = compact_text(
                    raw_path.read_text(encoding="utf-8"), max_length=140
                )
        stats = stats_by_digest.setdefault(
            digest,
            {
                "count": 0,
                "pages": set(),
                "classification": image["classification"],
                "render_dimensions": image["render_dimensions"],
                "ocr_confidence": float(ocr.get("mean_confidence") or 0.0),
                "ocr_text": sample_ocr_text,
            },
        )
        stats["count"] += 1
        stats["pages"].add(int(image["page_number"]))

    fixed: set[str] = set()
    minimum_page_coverage = 0.6
    minimum_page_hits = 4
    for digest, stats in stats_by_digest.items():
        pages_seen = len(stats["pages"])
        if pages_seen < minimum_page_hits:
            continue
        if pages_seen / max(page_count, 1) < minimum_page_coverage:
            continue
        classification = stats["classification"]
        dimensions = stats["render_dimensions"]
        width = int(dimensions["width"])
        height = int(dimensions["height"])
        ocr_text_length, _ = cleaned_text_score(str(stats["ocr_text"]))
        ocr_confidence = float(stats["ocr_confidence"])
        if classification.get("likely_logo") or classification.get("extreme_banner"):
            fixed.add(digest)
            continue
        if (
            pages_seen / max(page_count, 1) >= 0.85
            and ocr_text_length <= 80
            and ocr_confidence <= 80
            and max(width, height) >= 400
            and min(width, height) <= 260
        ):
            fixed.add(digest)
            continue
        if (
            not classification.get("likely_visual_candidate")
            and min(width, height) <= 180
            and max(width, height) <= 700
        ):
            fixed.add(digest)
    return fixed


def run_tesseract(image_path: Path, lang: str, tesseract_cmd: str) -> OcrResult:
    pytesseract = importlib.import_module("pytesseract")
    image_module = importlib.import_module("PIL.Image")
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    configs = [6, 11]
    image = image_module.open(image_path)
    best: OcrResult | None = None

    for psm in configs:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
        words: list[dict[str, Any]] = []
        texts: list[str] = []
        confs: list[float] = []
        for idx, raw_text in enumerate(data["text"]):
            text = (raw_text or "").strip()
            conf_raw = data["conf"][idx]
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = -1.0
            if not text:
                continue
            bbox = {
                "left": int(data["left"][idx]),
                "top": int(data["top"][idx]),
                "width": int(data["width"][idx]),
                "height": int(data["height"][idx]),
            }
            words.append({"text": text, "confidence": confidence, "bbox": bbox})
            texts.append(text)
            if confidence >= 0:
                confs.append(confidence)
        mean_conf = statistics.mean(confs) if confs else None
        result = OcrResult(
            status="ok" if texts else "empty",
            text=" ".join(texts).strip(),
            mean_confidence=mean_conf,
            best_psm=psm,
            words=words,
            note=None if texts else "tesseract produced no text",
        )
        if best is None:
            best = result
            continue
        best_score = ((best.mean_confidence or 0.0), len(best.text))
        new_score = ((result.mean_confidence or 0.0), len(result.text))
        if new_score > best_score:
            best = result

    assert best is not None
    return best


def load_cached_ocr(cache_dir: Path, digest: str) -> OcrResult | None:
    payload_path = cache_dir / "ocr" / f"{digest}.json"
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return OcrResult.from_json(payload)


def save_cached_ocr(cache_dir: Path, digest: str, result: OcrResult) -> None:
    payload_path = cache_dir / "ocr" / f"{digest}.json"
    payload_path.write_text(pretty_json(result.to_json()), encoding="utf-8")


def load_cached_vision(cache_dir: Path, cache_key: str) -> dict[str, Any] | None:
    payload_path = cache_dir / "vision" / f"{cache_key}.json"
    if not payload_path.exists():
        return None
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_cached_vision(
    cache_dir: Path, cache_key: str, payload: dict[str, Any]
) -> None:
    payload_path = cache_dir / "vision" / f"{cache_key}.json"
    payload_path.write_text(pretty_json(payload), encoding="utf-8")


def materialize_cached_vision(
    cache_dir: Path, cache_key: str, destination: Path
) -> dict[str, Any] | None:
    payload = load_cached_vision(cache_dir, cache_key)
    if payload is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(pretty_json(payload), encoding="utf-8")
    return payload


def write_ocr_sidecars(
    ocr_dir: Path, base_name: str, result: OcrResult
) -> tuple[str | None, str | None]:
    if result.status == "unavailable":
        return None, None
    txt_path = ocr_dir / f"{base_name}.txt"
    json_path = ocr_dir / f"{base_name}.json"
    txt_path.write_text(result.text + ("\n" if result.text else ""), encoding="utf-8")
    json_path.write_text(
        pretty_json(
            {
                "status": result.status,
                "mean_confidence": result.mean_confidence,
                "best_psm": result.best_psm,
                "note": result.note,
                "words": result.words,
            }
        ),
        encoding="utf-8",
    )
    return str(txt_path.name), str(json_path.name)


def should_run_ocr(classification: dict[str, Any], ocr_mode: str) -> bool:
    if ocr_mode == "none":
        return False
    if ocr_mode == "all":
        return True
    if classification["skip_reason"]:
        return False
    return (
        classification["likely_text_candidate"]
        or classification["likely_visual_candidate"]
    )


def build_vision_cache_key(digest: str, prompt_text: str) -> str:
    prompt_hash = hash_text(prompt_text)
    return slugify_filename(f"{digest}-{prompt_hash[:12]}")


def vision_payload_matches_request(
    payload: dict[str, Any],
    backend: str,
    model: str | None,
    prompt_hash: str,
    schema_version: str,
) -> bool:
    payload_backend = str(payload.get("backend") or "manual")
    payload_model = payload.get("model")
    payload_prompt_hash = str(payload.get("prompt_hash") or "")
    payload_schema_version = str(payload.get("schema_version") or "")
    if payload_backend != backend:
        return False
    if backend == "openrouter" and str(payload_model or "") != str(model or ""):
        return False
    if payload_schema_version and payload_schema_version != schema_version:
        return False
    if payload_prompt_hash and payload_prompt_hash != prompt_hash:
        return False
    return True


def fix_ocr_errors(text: str) -> str:
    corrections = {
        "Dectection": "Detection",
        "dectection": "detection",
        "Dectector": "Detector",
        "dectector": "detector",
        "Dectect": "Detect",
        "dectect": "detect",
        "Reccognition": "Recognition",
        "reccognition": "recognition",
        "Anallysis": "Analysis",
        "anallysis": "analysis",
        "Proccssing": "Processing",
        "proccssing": "processing",
        "lntelligence": "Intelligence",
        "lntelligent": "Intelligent",
        "Apllication": "Application",
        "apllication": "application",
        "Systcm": "System",
        "systcm": "system",
        "lechnology": "Technology",
        "lechnologies": "Technologies",
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text


def filtered_ocr_text(
    ocr: dict[str, Any], minimum_word_confidence: float = 72.0
) -> str:
    words = list(ocr.get("words") or [])
    kept: list[str] = []
    for word in words:
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(word.get("confidence", -1.0))
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence >= minimum_word_confidence:
            kept.append(text)
    return fix_ocr_errors(" ".join(kept).strip())


def should_request_vision_summary(
    image: dict[str, Any], backend: str, _ocr_dir: Path, args: argparse.Namespace
) -> bool:
    """Determine if an image should be sent to vision API for description."""
    if backend != "openrouter":
        return False
    classification = image["classification"]
    if classification["skip_reason"]:
        return False
    if classification["repeated"] or classification["extreme_banner"]:
        return False
    allowed_image_ids = {
        value.strip() for value in args.vision_only_image_id if value.strip()
    }
    if allowed_image_ids and image["image_id"] not in allowed_image_ids:
        return False
    if not classification["likely_visual_candidate"]:
        return False
    # Visual candidates always get vision analysis - OCR can't describe visual content
    return True


def openrouter_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or None


def resolve_vision_model(args: argparse.Namespace) -> str:
    return (
        args.vision_model
        or os.environ.get("OPENROUTER_MODEL")
        or "google/gemini-2.5-flash"
    )


def run_openrouter_vision(
    image_path: Path,
    image: dict[str, Any],
    prompt_text: str,
    model: str,
    api_key: str,
) -> VisionResult:
    started = time.time()
    mime_type = "image/png"
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"

    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    schema = {
        "name": "pdf_visual_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "maxLength": 480},
                "description": {"type": "string", "maxLength": 1200},
                "content_type": {
                    "type": "string",
                    "enum": [
                        "chart",
                        "diagram",
                        "table",
                        "photo",
                        "infographic",
                        "branding",
                        "screenshot",
                        "unknown",
                    ],
                },
                "visible_text": {"type": "string", "maxLength": 1600},
                "structured_items": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 240},
                    "maxItems": 20,
                },
                "text_extracted": {"type": "string", "maxLength": 300},
                "chart_trend": {"type": "string", "maxLength": 240},
                "diagram_components": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "uncertainty_notes": {"type": "string", "maxLength": 240},
            },
            "required": [
                "summary",
                "description",
                "content_type",
                "visible_text",
                "structured_items",
                "text_extracted",
                "chart_trend",
                "diagram_components",
                "confidence",
                "uncertainty_notes",
            ],
            "additionalProperties": False,
        },
    }
    request_payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 800,
        "messages": [
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt_text}\n\n"
                            "Context:\n"
                            f"- image_id: {image['image_id']}\n"
                            f"- page_number: {image['page_number']}\n"
                            f"- ocr_status: {image['ocr']['status']}\n"
                            f"- ocr_mean_confidence: {image['ocr']['mean_confidence']}\n"
                            "Use the OCR context only as a reliability hint, not as evidence about visual meaning."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                        },
                    },
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema,
        },
    }
    body = json.dumps(request_payload).encode("utf-8")
    request = urllib_request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    max_retries = 5
    retry_delay = 2.0
    payload = None
    for attempt in range(max_retries):
        try:
            with urllib_request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
                break
        except urllib_error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                detail = exc.read().decode("utf-8", errors="replace")
                wait_time = retry_delay * (2**attempt)
                time.sleep(wait_time)
                retry_delay = min(wait_time, 60)
                continue
            detail = exc.read().decode("utf-8", errors="replace")
            return VisionResult(
                status="error",
                payload=None,
                note=f"OpenRouter HTTP {exc.code}: {detail}",
            )
        except urllib_error.URLError as exc:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                time.sleep(wait_time)
                retry_delay = min(wait_time, 60)
                continue
            return VisionResult(
                status="error", payload=None, note=f"OpenRouter request failed: {exc}"
            )
        except TimeoutError as exc:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                time.sleep(wait_time)
                retry_delay = min(wait_time, 60)
                continue
            return VisionResult(
                status="error", payload=None, note=f"OpenRouter timeout: {exc}"
            )
        except json.JSONDecodeError as exc:
            return VisionResult(
                status="error",
                payload=None,
                note=f"OpenRouter response was not valid JSON: {exc}",
            )
    else:
        # Loop completed without break - all retries exhausted
        return VisionResult(
            status="error", payload=None, note="OpenRouter max retries exceeded"
        )

    try:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        result_json = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        return VisionResult(
            status="error",
            payload=None,
            note=f"OpenRouter structured output parse failed: {exc}",
        )

    normalized = {
        "schema_version": VISION_SCHEMA_VERSION,
        "backend": "openrouter",
        "model": model,
        "summary": str(result_json.get("summary") or "").strip(),
        "description": str(result_json.get("description") or "").strip(),
        "content_type": str(result_json.get("content_type") or "unknown").strip(),
        "visible_text": str(result_json.get("visible_text") or "").strip(),
        "structured_items": [
            str(item).strip()
            for item in list(result_json.get("structured_items") or [])
            if str(item).strip()
        ],
        "text_extracted": str(result_json.get("text_extracted") or "").strip(),
        "chart_trend": str(result_json.get("chart_trend") or "").strip(),
        "diagram_components": [
            str(component).strip()
            for component in list(result_json.get("diagram_components") or [])
            if str(component).strip()
        ],
        "confidence": str(result_json.get("confidence") or "medium").strip(),
        "uncertainty_notes": str(result_json.get("uncertainty_notes") or "").strip(),
        "analysis": {
            "temperature": 0.0,
            "latency_ms": int((time.time() - started) * 1000),
        },
    }
    return VisionResult(status="ok", payload=normalized)


def format_quote_block(text: str, fallback: str) -> str:
    content = (text or "").strip()
    if not content:
        content = fallback
    lines = [line.rstrip() for line in content.splitlines() if line.strip()]
    if not lines:
        lines = [fallback]
    return "\n".join(f"> {line}" for line in lines)


def cleaned_text_score(text: str) -> tuple[int, float]:
    stripped = " ".join(text.split())
    alnum = sum(ch.isalnum() for ch in stripped)
    ratio = (alnum / len(stripped)) if stripped else 0.0
    return len(stripped), ratio


def compact_text(text: str, max_length: int = 160) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def compact_blank_lines(text: str) -> str:
    compacted = re.sub(r"\n{3,}", "\n\n", text)
    return compacted.strip() + "\n"


def normalize_inline_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    collapsed = re.sub(r"\s+([,.;:!?])", r"\1", collapsed)
    return collapsed.strip()


def marker_number(text: str) -> int | None:
    match = NUMBERED_MARKER_RE.match((text or "").strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def reconstruct_ocr_lines(words: list[dict[str, Any]]) -> list[str]:
    if not words:
        return []
    ordered = sorted(words, key=lambda word: (word["top"], word["left"]))
    median_height = statistics.median(word["height"] for word in ordered)
    line_threshold = max(10.0, median_height * 0.75)
    lines: list[list[dict[str, Any]]] = []
    for word in ordered:
        center_y = word["top"] + (word["height"] / 2.0)
        if not lines:
            lines.append([word])
            continue
        last_line = lines[-1]
        last_center_y = statistics.mean(
            item["top"] + (item["height"] / 2.0) for item in last_line
        )
        if abs(center_y - last_center_y) <= line_threshold:
            last_line.append(word)
        else:
            lines.append([word])
    rendered: list[str] = []
    for line in lines:
        pieces = [
            str(word["text"]).strip()
            for word in sorted(line, key=lambda word: word["left"])
        ]
        text = normalize_inline_text(" ".join(piece for piece in pieces if piece))
        if text:
            rendered.append(text)
    return rendered


def ocr_words_for_layout(
    ocr: dict[str, Any], minimum_word_confidence: float = 45.0
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for raw_word in list(ocr.get("words") or []):
        text = str(raw_word.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(raw_word.get("confidence", -1.0))
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < minimum_word_confidence:
            continue
        bbox = dict(raw_word.get("bbox") or {})
        left = float(bbox.get("left") or 0.0)
        top = float(bbox.get("top") or 0.0)
        width = float(bbox.get("width") or 0.0)
        height = float(bbox.get("height") or 0.0)
        prepared.append(
            {
                "text": text,
                "confidence": confidence,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "right": left + width,
                "bottom": top + height,
            }
        )
    return prepared


def extract_numbered_ocr_items(ocr: dict[str, Any]) -> list[str]:
    words = ocr_words_for_layout(ocr)
    markers: list[dict[str, Any]] = []
    for word in words:
        number = marker_number(str(word["text"]))
        if number is None:
            continue
        markers.append(
            {
                **word,
                "number": number,
                "center_x": word["left"] + (word["width"] / 2.0),
            }
        )
    if len(markers) < 2:
        return []

    image_right = max((word["right"] for word in words), default=0.0)
    image_bottom = max((word["bottom"] for word in words), default=0.0)
    cluster_threshold = max(60.0, image_right * 0.12)
    columns: list[dict[str, Any]] = []
    for marker in sorted(markers, key=lambda item: item["center_x"]):
        if (
            not columns
            or abs(marker["center_x"] - columns[-1]["center_x"]) > cluster_threshold
        ):
            columns.append({"center_x": marker["center_x"], "markers": [marker]})
            continue
        columns[-1]["markers"].append(marker)
        centers = [item["center_x"] for item in columns[-1]["markers"]]
        columns[-1]["center_x"] = statistics.mean(centers)

    boundaries = [0.0]
    for idx in range(len(columns) - 1):
        boundaries.append(
            (columns[idx]["center_x"] + columns[idx + 1]["center_x"]) / 2.0
        )
    boundaries.append(image_right + 20.0)

    extracted: list[tuple[int, float, str]] = []
    for column_index, column in enumerate(columns):
        column_markers = sorted(
            column["markers"], key=lambda item: (item["top"], item["left"])
        )
        for marker_index, marker in enumerate(column_markers):
            next_top = (
                column_markers[marker_index + 1]["top"]
                if marker_index + 1 < len(column_markers)
                else image_bottom + 20.0
            )
            y_min = marker["top"] - max(marker["height"], 16.0)
            y_max = (marker["top"] + next_top) / 2.0
            x_min = boundaries[column_index] - 8.0
            x_max = boundaries[column_index + 1] + 8.0
            region_words: list[dict[str, Any]] = []
            for word in words:
                if (
                    word["left"] == marker["left"]
                    and word["top"] == marker["top"]
                    and word["text"] == marker["text"]
                ):
                    continue
                center_y = word["top"] + (word["height"] / 2.0)
                center_x = word["left"] + (word["width"] / 2.0)
                if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                    region_words.append(word)
            text = normalize_inline_text(" ".join(reconstruct_ocr_lines(region_words)))
            if text:
                extracted.append((marker["number"], marker["top"], text))

    extracted.sort(key=lambda item: (item[0], item[1]))
    deduped: list[str] = []
    seen_numbers: set[int] = set()
    for number, _, text in extracted:
        if number in seen_numbers:
            continue
        seen_numbers.add(number)
        deduped.append(f"{number}. {text}")
    return deduped if len(deduped) >= 2 else []


def extract_visible_ocr_block(ocr: dict[str, Any]) -> str:
    numbered_items = extract_numbered_ocr_items(ocr)
    if numbered_items:
        return "\n".join(numbered_items)

    words = ocr_words_for_layout(ocr, minimum_word_confidence=72.0)
    lines = reconstruct_ocr_lines(words)
    meaningful = [line for line in lines if len(line) >= 4]
    if len(meaningful) >= 2:
        return "\n".join(meaningful[:8])
    return ""


def build_document_frontmatter(
    pdf_path: Path, page_count: int, image_count: int
) -> str:
    return f"""---
document_type: enriched-markdown
source_pdf: {pdf_path.name}
page_count: {page_count}
image_count: {image_count}
extraction_tool: enrich_pdf_local.py
---
"""


def cleanup_agent_markdown(text: str) -> str:
    cleaned = PICTURE_TEXT_BLOCK_RE.sub("", text)
    kept_lines: list[str] = []
    lines = cleaned.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if GENERIC_SOURCE_LINE_RE.match(stripped):
            continue
        if GENERIC_DATE_SOURCE_LINE_RE.match(stripped):
            continue
        if DATE_LINE_RE.match(stripped):
            continue
        if PLACEHOLDER_FOOTER_LINE_RE.match(stripped):
            continue
        if stripped.startswith("Beispiel-Fu00dfzeile") or stripped.startswith(
            "Beispielu2010Fu00dfzeile"
        ):
            continue
        if RUNNING_HEADER_RE.match(stripped):
            continue
        heading_bullet = re.match(r"^#{1,6}\s*[•–-]\s*(.+)$", stripped)
        if heading_bullet:
            kept_lines.append(f"- {heading_bullet.group(1).strip()}")
            continue
        if stripped.isdigit() and len(stripped) <= 3:
            continue
        kept_lines.append(line.rstrip())
    return normalize_header_hierarchy(compact_blank_lines("\n".join(kept_lines)))


def normalize_header_hierarchy(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    first_header_seen = False
    for line in lines:
        stripped = line.strip()
        header_match = re.match(r"^(#{1,6})\s*(.+)$", stripped)
        if header_match:
            hashes, content = header_match.groups()
            current_level = len(hashes)
            content_stripped = content.strip()
            if not first_header_seen:
                output.append(f"# {content_stripped}")
                first_header_seen = True
            else:
                section_num = re.match(r"^(\d+)\.\s+", content_stripped)
                if section_num:
                    output.append(f"## {content_stripped}")
                elif current_level >= 2:
                    output.append(f"### {content_stripped}")
                else:
                    output.append(f"## {content_stripped}")
        else:
            output.append(line)
    return "\n".join(output)


def build_block_comment(image: dict[str, Any], vision_meta: str) -> str:
    return f"<!-- figure:{image['image_id']} p{image['page_number']} -->"

    bbox = ", ".join(str(value) for value in image["bbox"])
    parts = [
        f"image_id={image['image_id']}",
        f"page={image['page_number']}",
        f"page_image_index={image['page_image_index']}",
        f"placeholder_index={image['placeholder_index']}",
        f"xref={image['xref']}",
        f"bbox=[{bbox}]",
    ]
    if vision_meta:
        parts.append(vision_meta)
    return "<!-- image-block:start " + "; ".join(parts) + " -->"


def classify_alt_kind(text: str, image: dict[str, Any]) -> str:
    lowered = (text or "").lower()
    if "photo" in lowered or "photograph" in lowered or "courtyard" in lowered:
        return "Photo"
    if "chart" in lowered or "plot" in lowered or "graph" in lowered:
        return "Chart"
    if "diagram" in lowered or "workflow" in lowered or "pipeline" in lowered:
        return "Diagram"
    if "table" in lowered:
        return "Table"
    if "equation" in lowered or "formula" in lowered:
        return "Equation"
    if "screenshot" in lowered or "interface" in lowered:
        return "Screenshot"
    if "logo" in lowered or image["classification"].get("likely_logo"):
        return "Logo"
    if image["classification"].get("likely_visual_candidate"):
        return "Figure"
    return "Graphic"


def build_fallback_vision_text(image: dict[str, Any], ocr_dir: Path) -> str:
    classification = image["classification"]
    duplicate_of = classification.get("duplicate_of")
    if classification.get("repeated") and duplicate_of:
        return "Repeated decorative image."
    if classification.get("likely_logo"):
        return "Branding graphic."

    ocr = image["ocr"]
    confidence = float(ocr.get("mean_confidence") or 0.0)
    ocr_file = ocr.get("text_file")
    raw_text = ""
    if ocr_file:
        raw_path = ocr_dir / Path(ocr_file).name
        if raw_path.exists():
            raw_text = raw_path.read_text(encoding="utf-8")
    raw_text = compact_text(raw_text, max_length=140)

    if raw_text and confidence >= 55:
        if classification.get("extreme_banner"):
            return f"Text banner containing: {raw_text}"
        if classification.get("likely_text_candidate"):
            return f"Text-centric figure containing: {raw_text}"

    if classification.get("extreme_banner"):
        return "Banner or title graphic."
    if classification.get("likely_visual_candidate"):
        return "Figure extracted from the PDF."
    return "Supporting graphic."


def build_image_alt_text(image: dict[str, Any], vision_text: str, ocr_text: str) -> str:
    candidate = compact_text(vision_text or ocr_text, max_length=48)
    kind = classify_alt_kind(candidate, image)
    if candidate:
        return f"{kind}: {candidate}"
    classification = image["classification"]
    duplicate_of = classification.get("duplicate_of")
    if classification.get("repeated") and duplicate_of:
        return f"Graphic: repeated image matching {duplicate_of}"
    if classification.get("likely_logo"):
        return "Logo"
    if classification.get("extreme_banner"):
        return "Banner"
    if classification.get("likely_visual_candidate"):
        return "Figure"
    return "Graphic"


def summarize_ocr_for_markdown(
    image: dict[str, Any], ocr_dir: Path
) -> tuple[str, str, str]:
    ocr = image["ocr"]
    if ocr["status"] == "ok":
        visible_block = extract_visible_ocr_block(ocr)
        ocr_text = filtered_ocr_text(ocr)
        text_length, alnum_ratio = cleaned_text_score(ocr_text)
        confidence = ocr["mean_confidence"] or 0.0
        if visible_block:
            return (
                "ocr=structured",
                compact_text(ocr_text, max_length=200),
                visible_block,
            )
        if confidence >= 88 and text_length >= 24 and alnum_ratio >= 0.72:
            return "ocr=high-confidence", compact_text(ocr_text, max_length=200), ""
        return "", "", ""
    return "", "", ""


def format_structured_items(items: list[str]) -> str:
    cleaned_items = [
        normalize_inline_text(item) for item in items if normalize_inline_text(item)
    ]
    if not cleaned_items:
        return ""
    if all(re.match(r"^\d+\.\s+", item) for item in cleaned_items):
        return "\n".join(cleaned_items)
    if all(re.match(r"^\d+\s+", item) for item in cleaned_items):
        return "\n".join(
            re.sub(r"^(\d+)\s+", r"\1. ", item, count=1) for item in cleaned_items
        )
    return "\n".join(f"- {item}" for item in cleaned_items)


def format_vision_detail(payload: dict[str, Any]) -> str:
    structured = format_structured_items(list(payload.get("structured_items") or []))
    if structured:
        return structured

    visible_text = normalize_inline_text(str(payload.get("visible_text") or ""))
    text_extracted = normalize_inline_text(str(payload.get("text_extracted") or ""))
    if visible_text and len(visible_text) >= 40:
        return visible_text
    if (
        text_extracted
        and len(text_extracted) >= 40
        and text_extracted.lower() not in visible_text.lower()
    ):
        return text_extracted

    diagram_components = [
        normalize_inline_text(component)
        for component in list(payload.get("diagram_components") or [])
        if normalize_inline_text(component)
    ]
    if diagram_components:
        return "\n".join(f"- {component}" for component in diagram_components[:10])
    return ""


def is_generic_vision_summary(text: str) -> bool:
    lowered = normalize_inline_text(text).lower()
    if not lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "numbered list of",
            "bullet list of",
            "text-centric figure containing:",
            "text banner containing:",
            "grid of recommendations",
        )
    )


def load_vision_summary(
    assets_dir: Path, image: dict[str, Any]
) -> tuple[str, str, str]:
    sidecar_rel = image["vision"]["sidecar_file"]
    sidecar_path = assets_dir / sidecar_rel
    if not sidecar_path.exists():
        status = image["vision"]["status"]
        if status == "pending":
            return (
                "vision_status=pending",
                build_fallback_vision_text(image, assets_dir / "ocr"),
                "",
            )
        return (
            "vision_status=optional",
            build_fallback_vision_text(image, assets_dir / "ocr"),
            "",
        )
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return (
            "vision_status=parse_error",
            "Figure present but vision sidecar could not be parsed.",
            "",
        )
    summary = (
        payload.get("summary")
        or payload.get("description")
        or payload.get("caption")
        or ""
    )
    content_type = payload.get("content_type") or "visual"
    confidence = payload.get("confidence") or "unknown"
    return (
        f"content_type={content_type}; confidence={confidence}",
        compact_text(summary, 420),
        format_vision_detail(payload),
    )


def should_drop_image_from_markdown(assets_dir: Path, image: dict[str, Any]) -> bool:
    classification = image["classification"]
    if (
        classification.get("likely_logo")
        or classification.get("skip_reason") == "duplicate_logo_or_badge"
    ):
        return True

    sidecar_rel = image["vision"]["sidecar_file"]
    sidecar_path = assets_dir / sidecar_rel
    if not sidecar_path.exists():
        return False

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    content_type = str(payload.get("content_type") or "").strip().lower()
    if "branding" in content_type:
        return True

    summary = compact_text(
        str(payload.get("summary") or payload.get("description") or ""),
        max_length=160,
    ).lower()
    return (
        summary.startswith("logo of ")
        or summary.startswith("logo for ")
        or " corporate logo" in summary
    )


def build_image_block(
    image: dict[str, Any],
    image_rel_path: str,
    ocr_status_line: str,
    ocr_text: str,
    ocr_detail: str,
    vision_status_line: str,
    vision_text: str,
    vision_detail: str,
) -> str:
    alt_text = build_image_alt_text(image, vision_text, ocr_text)
    detail_blocks: list[str] = []
    if vision_detail:
        detail_blocks.append(vision_detail)
    if ocr_detail and ocr_detail.lower() not in vision_detail.lower():
        detail_blocks.append(ocr_detail)

    summary_lines: list[str] = []
    detail_starts_with_number = bool(
        detail_blocks
        and re.match(r"^(?:-\s+)?\d{1,2}[.)]\s+", detail_blocks[0].splitlines()[0])
    )
    if vision_text and not (
        detail_blocks
        and (is_generic_vision_summary(vision_text) or detail_starts_with_number)
    ):
        summary_lines.append(f"> {vision_text}")
    elif vision_text and not detail_blocks:
        summary_lines.append(f"> {vision_text}")
    if ocr_text and ocr_text.lower() not in vision_text.lower() and not ocr_detail:
        summary_lines.append(f"> OCR cue: {ocr_text}")

    parts = [
        build_block_comment(image, vision_status_line),
        f"![{alt_text}]({image_rel_path})",
        *summary_lines,
    ]
    if detail_blocks and summary_lines:
        parts.append("")
    parts.extend(detail_blocks)
    parts.append(f"<!-- /figure:{image['image_id']} -->")
    return "\n".join(parts)


def build_unmapped_placeholder_block(
    page_number: int, placeholder_index: int, placeholder_match: re.Match[str]
) -> str:
    width = int(placeholder_match.group(1))
    height = int(placeholder_match.group(2))
    area = width * height
    # Require minimum 60x60 = 3600 pixels to avoid false positives from layout artifacts
    # Many false positives are small inline markers or layout elements
    if area <= 3600:
        return ""
    # Skip very narrow or very short elements (likely layout artifacts, not figures)
    aspect_ratio = max(width, height) / max(min(width, height), 1)
    if aspect_ratio > 12:
        return ""
    image_id = f"page-{page_number:03d}-placeholder-{placeholder_index:03d}-unmapped"
    return f"<!-- figure:{image_id} p{page_number} placeholder unmapped -->"


def replace_placeholder(text: str, occurrence_index: int, replacement: str) -> str:
    matches = list(PLACEHOLDER_RE.finditer(text))
    if occurrence_index >= len(matches):
        return text
    match = matches[occurrence_index]
    return text[: match.start()] + replacement + text[match.end() :]


def replace_placeholders(text: str, replacements: list[str]) -> str:
    matches = list(PLACEHOLDER_RE.finditer(text))
    if not replacements:
        return text
    parts: list[str] = []
    cursor = 0
    for idx, match in enumerate(matches):
        parts.append(text[cursor : match.start()])
        if idx < len(replacements):
            parts.append(replacements[idx])
        else:
            parts.append(match.group(0))
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)


def main() -> None:
    pymupdf = importlib.import_module("pymupdf")
    pymupdf4llm = importlib.import_module("pymupdf4llm")

    args = parse_args()
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    skill_dir = Path(__file__).parent.parent
    load_dotenv_file(skill_dir / ".env")

    markdown_path, enriched_path, assets_dir = derive_paths(pdf_path, args)
    cache_dir = derive_cache_dir(pdf_path, args)
    vision_prompt = load_vision_prompt(args)
    vision_prompt_hash = hash_text(vision_prompt)
    openrouter_key = openrouter_api_key()
    vision_model = resolve_vision_model(args)
    images_dir = assets_dir / "images"
    ocr_dir = assets_dir / "ocr"
    pages_dir = assets_dir / "pages"
    vision_dir = assets_dir / "vision"
    manifest_path = assets_dir / "manifest.json"
    review_queue_path = assets_dir / "review_queue.json"

    for path in (markdown_path, enriched_path, manifest_path, review_queue_path):
        if path.exists() and not args.force:
            pass
    for directory in (assets_dir, images_dir, ocr_dir, pages_dir, vision_dir):
        directory.mkdir(parents=True, exist_ok=True)
    prepare_cache_dir(cache_dir, args)

    tesseract_cmd, tesseract_note = check_tesseract(args.lang)
    tesseract_ok = tesseract_cmd is not None
    vision_calls = 0

    doc = pymupdf.open(pdf_path)
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    raw_markdown = (
        "\n\n".join(chunk["text"].rstrip() for chunk in page_chunks).rstrip() + "\n"
    )
    markdown_path.write_text(raw_markdown, encoding="utf-8")

    all_entries: list[dict[str, Any]] = []
    page_records: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    page_outputs: list[str] = []
    global_image_index = 0

    for page_number, chunk in enumerate(page_chunks, start=1):
        page = doc[page_number - 1]
        page_text = chunk["text"].rstrip() + "\n"
        raw_page_path = pages_dir / f"page-{page_number:03d}.raw.md"
        raw_page_path.write_text(page_text, encoding="utf-8")

        placeholders = list(PLACEHOLDER_RE.finditer(page_text))
        image_infos = sorted(
            page.get_image_info(xrefs=True),
            key=lambda item: (
                float(item["bbox"][1]),
                float(item["bbox"][0]),
                int(item["number"]),
            ),
        )

        page_entries: list[dict[str, Any]] = []
        mapped_count = min(len(placeholders), len(image_infos))

        for page_image_index in range(mapped_count):
            info = image_infos[page_image_index]
            placeholder = placeholders[page_image_index]
            xref = int(info.get("xref") or 0)
            if xref:
                extracted = doc.extract_image(xref)
                image_bytes = extracted["image"]
                ext = extracted.get("ext") or "png"
            else:
                rect = pymupdf.Rect(info["bbox"])
                pix = page.get_pixmap(clip=rect, dpi=200)
                image_bytes = pix.tobytes("png")
                ext = "png"

            global_image_index += 1
            image_id = f"page-{page_number:03d}-img-{page_image_index + 1:03d}"
            base_name = image_id if not xref else f"{image_id}-xref-{xref}"
            image_file = images_dir / f"{base_name}.{ext}"
            image_file.write_bytes(image_bytes)
            digest = sha256_bytes(image_bytes)
            duplicate_of = seen_hashes.get(digest)
            if duplicate_of is None:
                seen_hashes[digest] = image_id

            classification = classify_image(info, duplicate_of, args.min_dimension)
            ocr_result = OcrResult(
                status="unavailable",
                text="",
                mean_confidence=None,
                best_psm=None,
                words=[],
                note=tesseract_note if not tesseract_ok else None,
            )
            if tesseract_ok and should_run_ocr(classification, args.ocr_mode):
                cached_ocr = load_cached_ocr(cache_dir, digest)
                if cached_ocr is not None:
                    ocr_result = cached_ocr
                    if ocr_result.note:
                        ocr_result.note = f"{ocr_result.note} (loaded from cache)"
                    else:
                        ocr_result.note = "OCR result loaded from cache"
                else:
                    ocr_result = run_tesseract(image_file, args.lang, tesseract_cmd)
                    save_cached_ocr(cache_dir, digest, ocr_result)
            elif args.ocr_mode == "none":
                ocr_result.note = "OCR disabled by --ocr-mode none"
            elif classification["skip_reason"]:
                ocr_result.note = (
                    f"OCR skipped in auto mode ({classification['skip_reason']})"
                )

            ocr_text_name, ocr_json_name = write_ocr_sidecars(
                ocr_dir, base_name, ocr_result
            )
            needs_vision = (
                should_request_vision_summary(
                    {
                        "image_id": image_id,
                        "classification": classification,
                        "ocr": {
                            "status": ocr_result.status,
                            "mean_confidence": ocr_result.mean_confidence,
                            "text_file": str(Path("ocr") / ocr_text_name)
                            if ocr_text_name
                            else None,
                            "words": ocr_result.words,
                        },
                    },
                    args.vision_backend,
                    ocr_dir,
                    args,
                )
                or classification["likely_visual_candidate"]
                or ocr_result.status == "unavailable"
                or (
                    ocr_result.mean_confidence is not None
                    and ocr_result.mean_confidence < 85
                )
                or (ocr_result.status == "empty" and not classification["skip_reason"])
            )
            vision_cache_key = build_vision_cache_key(
                digest,
                vision_prompt,
            )
            vision_sidecar_rel = Path("vision") / f"{base_name}.json"
            vision_sidecar_path = assets_dir / vision_sidecar_rel
            cached_vision = None
            if args.vision_cache_mode in ("use", "refresh"):
                cached_vision = materialize_cached_vision(
                    cache_dir, vision_cache_key, vision_sidecar_path
                )
            if (
                cached_vision is None
                and args.vision_cache_mode == "use"
                and vision_sidecar_path.exists()
            ):
                try:
                    existing_sidecar_payload = json.loads(
                        vision_sidecar_path.read_text(encoding="utf-8")
                    )
                    if vision_payload_matches_request(
                        existing_sidecar_payload,
                        args.vision_backend,
                        vision_model if args.vision_backend == "openrouter" else None,
                        vision_prompt_hash,
                        VISION_SCHEMA_VERSION,
                    ):
                        save_cached_vision(
                            cache_dir,
                            vision_cache_key,
                            existing_sidecar_payload,
                        )
                except json.JSONDecodeError:
                    pass
            if (
                cached_vision is None
                and args.vision_backend == "openrouter"
                and needs_vision
                and openrouter_key
                and (
                    args.vision_max_images <= 0 or vision_calls < args.vision_max_images
                )
                and should_request_vision_summary(
                    {
                        "image_id": image_id,
                        "classification": classification,
                        "ocr": {
                            "status": ocr_result.status,
                            "mean_confidence": ocr_result.mean_confidence,
                            "text_file": str(Path("ocr") / ocr_text_name)
                            if ocr_text_name
                            else None,
                            "words": ocr_result.words,
                        },
                    },
                    args.vision_backend,
                    ocr_dir,
                    args,
                )
            ):
                vision_result = run_openrouter_vision(
                    image_file,
                    {
                        "image_id": image_id,
                        "page_number": page_number,
                        "ocr": {
                            "status": ocr_result.status,
                            "mean_confidence": ocr_result.mean_confidence,
                        },
                    },
                    vision_prompt,
                    vision_model,
                    openrouter_key,
                )
                if vision_result.status == "ok" and vision_result.payload is not None:
                    vision_payload = {
                        **vision_result.payload,
                        "digest_sha256": digest,
                        "prompt_hash": vision_prompt_hash,
                        "provenance": {
                            "page_number": page_number,
                            "page_image_index": page_image_index + 1,
                            "xref": xref or None,
                            "trigger_reason": {
                                "ocr_status": ocr_result.status,
                                "ocr_mean_confidence": ocr_result.mean_confidence,
                                "classification": classification,
                            },
                        },
                    }
                    vision_sidecar_path.write_text(
                        pretty_json(vision_payload), encoding="utf-8"
                    )
                    if args.vision_cache_mode != "bypass":
                        save_cached_vision(cache_dir, vision_cache_key, vision_payload)
                    cached_vision = vision_payload
                    vision_calls += 1
            image_entry = {
                "image_id": image_id,
                "global_image_index": global_image_index,
                "pdf_path": str(pdf_path),
                "page_number": page_number,
                "page_image_index": page_image_index + 1,
                "placeholder_index": page_image_index + 1,
                "placeholder_text": placeholder.group(0),
                "placeholder_dimensions": {
                    "width": int(placeholder.group(1)),
                    "height": int(placeholder.group(2)),
                },
                "image_number": int(info["number"]),
                "xref": xref or None,
                "bbox": serialize_rect(info["bbox"]),
                "render_dimensions": {
                    "width": int(info["width"]),
                    "height": int(info["height"]),
                    "xres": int(info["xres"]),
                    "yres": int(info["yres"]),
                },
                "digest_sha256": digest,
                "classification": classification,
                "image_file": str(Path("images") / image_file.name),
                "ocr": {
                    "status": ocr_result.status,
                    "mean_confidence": ocr_result.mean_confidence,
                    "best_psm": ocr_result.best_psm,
                    "note": ocr_result.note,
                    "text_file": str(Path("ocr") / ocr_text_name)
                    if ocr_text_name
                    else None,
                    "json_file": str(Path("ocr") / ocr_json_name)
                    if ocr_json_name
                    else None,
                },
                "vision": {
                    "status": (
                        "cached"
                        if cached_vision is not None
                        else ("pending" if needs_vision else "optional")
                    ),
                    "backend": args.vision_backend,
                    "model": vision_model
                    if args.vision_backend == "openrouter"
                    else None,
                    "cache_key": vision_cache_key,
                    "sidecar_file": str(vision_sidecar_rel),
                },
                "rendering": {
                    "drop_from_enriched_markdown": False,
                },
                "mapping": {
                    "status": "exact_index_within_page",
                    "page_placeholder_count": len(placeholders),
                    "page_embedded_image_count": len(image_infos),
                },
            }
            page_entries.append(image_entry)
            all_entries.append(image_entry)

            if needs_vision and cached_vision is None:
                review_queue.append(
                    {
                        "image_id": image_id,
                        "image_file": str(image_file),
                        "pdf_path": str(pdf_path),
                        "page_number": page_number,
                        "prompt": (
                            "Describe this PDF figure for study notes. If it contains readable text, repeat it. "
                            "If it is a chart, explain the trend. If it is a technical or academic diagram, "
                            "explain the components and takeaway. Be explicit about uncertainty."
                        ),
                        "reason": {
                            "ocr_status": ocr_result.status,
                            "ocr_mean_confidence": ocr_result.mean_confidence,
                            "classification": classification,
                        },
                    }
                )

        if len(placeholders) != len(image_infos):
            review_queue.append(
                {
                    "type": "mapping_mismatch",
                    "pdf_path": str(pdf_path),
                    "page_number": page_number,
                    "placeholder_count": len(placeholders),
                    "embedded_image_count": len(image_infos),
                    "note": "Page-level placeholder count and embedded image count differ; review mapping manually.",
                }
            )

        page_records.append(
            {
                "page_number": page_number,
                "page_text": page_text,
                "placeholders": placeholders,
                "mapped_count": mapped_count,
                "page_entries": page_entries,
            }
        )

    fixed_image_digests = detect_fixed_image_digests(
        all_entries, doc.page_count, ocr_dir
    )
    for image in all_entries:
        image["rendering"] = {
            "drop_from_enriched_markdown": image["digest_sha256"] in fixed_image_digests
            or should_drop_image_from_markdown(assets_dir, image)
        }

    for page_record in page_records:
        page_number = int(page_record["page_number"])
        page_text = str(page_record["page_text"])
        placeholders = list(page_record["placeholders"])
        mapped_count = int(page_record["mapped_count"])
        page_entries = list(page_record["page_entries"])
        replacements: list[str] = []
        for image in page_entries:
            if image.get("rendering", {}).get("drop_from_enriched_markdown"):
                replacements.append("")
                continue
            image_rel_path = f"{assets_dir.name}/{image['image_file']}"
            ocr_status_line, ocr_text, ocr_detail = summarize_ocr_for_markdown(
                image, ocr_dir
            )
            vision_status_line, vision_text, vision_detail = load_vision_summary(
                assets_dir, image
            )
            replacements.append(
                build_image_block(
                    image,
                    image_rel_path,
                    ocr_status_line,
                    ocr_text,
                    ocr_detail,
                    vision_status_line,
                    vision_text,
                    vision_detail,
                )
            )

        if len(placeholders) > mapped_count:
            for placeholder_index in range(mapped_count, len(placeholders)):
                replacements.append(
                    build_unmapped_placeholder_block(
                        page_number,
                        placeholder_index + 1,
                        placeholders[placeholder_index],
                    )
                )

        page_enriched = cleanup_agent_markdown(
            replace_placeholders(page_text, replacements)
        )
        enriched_page_path = pages_dir / f"page-{page_number:03d}.enriched.md"
        enriched_page_path.write_text(page_enriched, encoding="utf-8")
        page_outputs.append(page_enriched.rstrip())

    enriched_markdown = compact_blank_lines("\n\n".join(page_outputs))
    frontmatter = build_document_frontmatter(pdf_path, doc.page_count, len(all_entries))
    enriched_with_frontmatter = frontmatter + "\n" + enriched_markdown
    enriched_path.write_text(enriched_with_frontmatter, encoding="utf-8")

    manifest = {
        "manifest_version": "1.0",
        "document": {
            "pdf_path": str(pdf_path),
            "markdown_path": str(markdown_path),
            "enriched_markdown_path": str(enriched_path),
            "assets_dir": str(assets_dir),
            "cache_dir": str(cache_dir),
            "page_count": doc.page_count,
            "ocr_mode": args.ocr_mode,
            "ocr_language": args.lang,
            "tesseract_available": tesseract_ok,
            "tesseract_note": tesseract_note,
            "vision_backend": args.vision_backend,
            "vision_model": vision_model
            if args.vision_backend == "openrouter"
            else None,
            "vision_prompt_hash": vision_prompt_hash,
            "vision_schema_version": VISION_SCHEMA_VERSION,
            "vision_cache_mode": args.vision_cache_mode,
            "vision_calls": vision_calls,
            "openrouter_api_key_present": openrouter_key is not None,
            "suppressed_fixed_image_digest_count": len(fixed_image_digests),
        },
        "pages": [
            {
                "page_number": page_number,
                "raw_markdown": str(Path("pages") / f"page-{page_number:03d}.raw.md"),
                "enriched_markdown": str(
                    Path("pages") / f"page-{page_number:03d}.enriched.md"
                ),
            }
            for page_number in range(1, doc.page_count + 1)
        ],
        "images": all_entries,
        "review_queue_count": len(review_queue),
    }
    manifest_path.write_text(pretty_json(manifest), encoding="utf-8")
    review_queue_path.write_text(pretty_json(review_queue), encoding="utf-8")

    summary = {
        "pdf": str(pdf_path),
        "raw_markdown": str(markdown_path),
        "enriched_markdown": str(enriched_path),
        "assets_dir": str(assets_dir),
        "images": len(all_entries),
        "review_queue": len(review_queue),
        "tesseract_available": tesseract_ok,
        "tesseract_note": tesseract_note,
        "vision_backend": args.vision_backend,
        "vision_cache_mode": args.vision_cache_mode,
        "vision_calls": vision_calls,
        "suppressed_fixed_image_digest_count": len(fixed_image_digests),
    }
    print(pretty_json(summary))


if __name__ == "__main__":
    main()
