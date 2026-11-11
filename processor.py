#!/usr/bin/env python3
"""
Paperless preprocessor (Docker/NAS-friendly).

Watches INCOMING_DIR for stable PDFs, normalizes and OCRs them, renders the
first MAX_PAGES pages, and makes two Ollama vision calls per page (handwriting
and context). Appends an LLM metadata summary, publishes to OUT_DIR, archives
the original under incoming/processed, and optionally cleans the work folder.

Configuration (environment variables):
  INCOMING_DIR=/nas/incoming  OUT_DIR=/nas/consume  WORK_DIR=/nas/work
  STABLE_SECONDS=20  POLL_SECONDS=5  MAX_PAGES=4  RENDER_DPI=150
  OCR_LANG=eng  OCR_JOBS=3  OCR_THREADS=<OCR_JOBS>
  JOBID_PREFIX=1  CLEANUP_WORKDIR=1
  OLLAMA_URL=http://ollama:11434  OLLAMA_MODEL=gemma4:latest
  OLLAMA_CONNECT_TIMEOUT=10  OLLAMA_READ_TIMEOUT=600

The model tag is configurable: use the exact tag installed on your Ollama
server and a model that supports images and structured output. The default
matches the owner's current deployment; it is not a universal recommendation.

Single worker; one PDF at a time. Any required Ollama call that exhausts its
retries, including invalid structured output, fails the document job. No final
PDF is published for that attempt. The main loop moves the original to
incoming/failed and keeps working on other documents. After resolving the
problem, move the original back into incoming to retry it. A successfully
analyzed page may legitimately have an empty handwritten_entries list.
"""

import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---- Crash logging ----
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass

def _fatal(msg: str) -> None:
    print(f"[FATAL] {msg}", flush=True)
    time.sleep(1.0)
    sys.exit(1)

def _excepthook(exc_type, exc, tb):
    print("[UNCAUGHT EXCEPTION]", flush=True)
    traceback.print_exception(exc_type, exc, tb)
    time.sleep(1.0)
    sys.exit(1)

sys.excepthook = _excepthook

print("### PREPROCESSOR VERSION: 2026-11-11a ###", flush=True)
print(f"[startup] pid={os.getpid()} python={sys.version.split()[0]} cwd={os.getcwd()}", flush=True)

try:
    import pikepdf
    import requests
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception as e:
    print("[startup] import failure:", flush=True)
    traceback.print_exc()
    _fatal(f"ImportError: {e!r}")

# -------------------------
# Environment / configuration
# -------------------------

INCOMING_DIR = Path(os.environ.get("INCOMING_DIR", "/nas/incoming"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/nas/consume"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/nas/work"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")

OCR_LANG = os.environ.get("OCR_LANG", "eng")
STABLE_SECONDS = int(os.environ.get("STABLE_SECONDS", "20"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "5"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "4"))
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))
OCR_JOBS = int(os.environ.get("OCR_JOBS", "3"))
OCR_THREADS = int(os.environ.get("OCR_THREADS", str(OCR_JOBS)))
JOBID_PREFIX = os.environ.get("JOBID_PREFIX", "1").strip() != "0"
CLEANUP_WORKDIR = os.environ.get("CLEANUP_WORKDIR", "1").strip() != "0"
OLLAMA_CONNECT_TIMEOUT = int(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "10"))
OLLAMA_READ_TIMEOUT = int(os.environ.get("OLLAMA_READ_TIMEOUT", "600"))

print(
    f"[startup] INCOMING_DIR={INCOMING_DIR} OUT_DIR={OUT_DIR} WORK_DIR={WORK_DIR} "
    f"STABLE_SECONDS={STABLE_SECONDS} POLL_SECONDS={POLL_SECONDS} MAX_PAGES={MAX_PAGES} DPI={RENDER_DPI} "
    f"OCR_JOBS={OCR_JOBS} OCR_THREADS={OCR_THREADS} JOBID_PREFIX={JOBID_PREFIX} CLEANUP_WORKDIR={CLEANUP_WORKDIR} "
    f"OLLAMA_URL={OLLAMA_URL} MODEL={OLLAMA_MODEL}",
    flush=True,
)

for d in (INCOMING_DIR, OUT_DIR, WORK_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[startup] cannot create/access directory: {d}", flush=True)
        traceback.print_exc()
        _fatal(f"Directory error for {d}: {e!r}")

# -------------------------
# Vision structured output schema
# -------------------------

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "purpose": {"type": "string"},
        "issuer": {"type": "string"},
        "handwritten_entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location_hint": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["location_hint", "text"],
                "additionalProperties": False,
            },
        },
        "non_text_visuals": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "document_type", "purpose", "issuer", "handwritten_entries",
        "non_text_visuals", "confidence",
    ],
    "additionalProperties": False,
}

PROMPT_TEMPLATE = """SYSTEM:
You are a handwriting + visual-context extraction assistant for a paperless archival system.
Faithfulness is more important than completeness.

TASK:
You will receive ONE full page image from a document.

IMPORTANT:
Below is the OCR text already extracted from THIS SAME PAGE.
You MUST NOT repeat printed text that already appears in the OCR text.
Your job is to add what OCR tends to miss: handwriting + non-text visuals + context.

OCR_TEXT_START
{ocr_text}
OCR_TEXT_END

WHAT TO OUTPUT (JSON ONLY, matching the schema):
1) handwritten_entries:
- ONLY handwritten text (pen/pencil) that is NOT already present in OCR text above.
- If handwriting exists but cannot be read: include ONE entry with text "[UNREADABLE]".
- NEVER output placeholders like "[Handwritten]" by itself.
- Do NOT guess missing characters.
- If unsure whether something is handwritten vs printed: treat it as printed and EXCLUDE it.

2) non_text_visuals:
- Note signatures, stamps, checked checkboxes, logos, drawings/photos, seals, tables, etc.
- Keep each item short.

3) context:
- document_type: short phrase (e.g. "medical form", "registration form", "endorsement", "letter", "invoice", "statement")
- purpose: one short sentence about what the document is for
- issuer: organization name if visible (letterhead/logo/footer), else ""

OUTPUT RULES:
- Return ONLY valid JSON, no commentary.
- Keep strings concise. Do NOT dump paragraphs of printed text.
"""

PROMPT_ECHO_MARKERS = [
    "You are an OCR and document-analysis assistant", "Text extraction (OCR):",
    "VISUAL DESCRIPTION:", "OUTPUT FORMAT:", "OUTPUT RULES:",
    "FAILURE HANDLING:", "CLASSIFICATION RULES:", "INSTRUCTIONS:", "TASK:",
    "SYSTEM:", "You will receive a single image", "Extract ALL readable text",
    "Do NOT summarize or paraphrase",
]

# -------------------------
# Helpers
# -------------------------

def run(cmd: List[str], extra_env: Dict[str, str] | None = None) -> None:
    """Run a subprocess; apply any supplied CPU/thread caps."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.check_call(cmd, env=env)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sanitize_text(s: str, max_len: int = 500) -> str:
    if s is None:
        return ""
    s = str(s)
    for m in PROMPT_ECHO_MARKERS:
        if m in s:
            return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    s = " ".join(lines).strip()
    return s[:max_len] if s else ""

def extract_ocr_text_by_page(pdf: Path, page_no: int) -> str:
    try:
        out = subprocess.check_output(
            ["pdftotext", "-f", str(page_no), "-l", str(page_no), str(pdf), "-"],
            stderr=subprocess.DEVNULL, text=True, timeout=30,
        )
        return (out or "").strip()
    except Exception:
        return ""

def _page_num_from_path(p: Path) -> int:
    m = re.search(r"page-(\d+)$", p.stem)
    return int(m.group(1)) if m else 10**9

def pdftoimages(pdf: Path, out_dir: Path, max_pages: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    run([
        "pdftoppm", "-png", "-r", str(RENDER_DPI), "-f", "1",
        "-l", str(max_pages), str(pdf), str(prefix),
    ])
    imgs = sorted(list(out_dir.glob("page-*.png")), key=_page_num_from_path)
    print(f"[render] pdf={pdf.name} requested_pages=1..{max_pages} produced_images={len(imgs)}", flush=True)
    if imgs:
        print("[render] images=" + ", ".join(p.name for p in imgs), flush=True)
    return imgs

def extract_json_object(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = s.strip().replace("```json", "").replace("```", "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = s[start:end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

class OllamaAnalysisError(RuntimeError):
    """A required page analysis could not be completed successfully."""

def validate_analysis(value: Any, schema: Dict[str, Any], path: str = "result") -> None:
    """Validate the subset of JSON Schema used by this script, not arbitrary schemas.

    Reject missing fields and wrong types rather than confusing a malformed
    response with a valid analysis that found no handwriting. Error messages
    identify field paths without including document text.
    """
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object")
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key}: missing required field")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path}: unexpected field(s)")
        for key, child_schema in properties.items():
            if key in value:
                validate_analysis(value[key], child_schema, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        for index, item in enumerate(value):
            validate_analysis(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected number")
        if not math.isfinite(value):
            raise ValueError(f"{path}: expected finite number")
    else:
        raise ValueError(f"Unsupported schema type: {kind}")

def post_ollama(payload: Dict[str, Any], tag: str, page_no: int, tries: int = 3) -> Dict[str, Any]:
    if tries < 1:
        raise ValueError("tries must be at least 1")
    last_err = "unknown failure"
    for attempt in range(1, tries + 1):
        t0 = time.time()
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat", json=payload,
                timeout=(OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT),
            )
            dt = time.time() - t0
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                print(f"[ollama:{tag}] page={page_no} attempt={attempt} status={r.status_code} time={dt:.2f}s", flush=True)
            else:
                data = r.json()
                content = (data.get("message", {}) or {}).get("content", "") or ""
                parsed = extract_json_object(content)
                if not parsed:
                    last_err = "JSON_PARSE_FAIL or empty response object"
                    print(f"[ollama:{tag}] page={page_no} attempt={attempt} status=200 time={dt:.2f}s {last_err}", flush=True)
                else:
                    validate_analysis(parsed, payload["format"])
                    print(f"[ollama:{tag}] page={page_no} attempt={attempt} status=200 time={dt:.2f}s ok", flush=True)
                    return parsed
        except Exception as e:
            dt = time.time() - t0
            # Avoid dumping model output, HTTP bodies, or exception text that
            # might contain document content or credentials in a server URL.
            last_err = f"{type(e).__name__} during request/response validation"
            print(f"[ollama:{tag}] page={page_no} attempt={attempt} EXCEPTION time={dt:.2f}s err={last_err}", flush=True)
        if attempt < tries:
            time.sleep(1.0 * attempt)

    message = f"analysis failed: page={page_no} stage={tag} after {tries} attempts ({last_err})"
    print(f"[ollama:{tag}] ANALYSIS_FAILED {message}", flush=True)
    raise OllamaAnalysisError(message)

def norm_cmp(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

# -------------------------
# LLM extraction
# -------------------------

def ollama_extract(img: Path, ocr_text: str, page_no: int) -> Dict[str, Any]:
    handwriting_schema = {
        "type": "object",
        "properties": {
            "handwritten_entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"location_hint": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["location_hint", "text"],
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "number"},
        },
        "required": ["handwritten_entries", "confidence"],
        "additionalProperties": False,
    }

    handwriting_prompt = (
        "You will be shown a single PAGE image. "
        "Return ONLY JSON (no commentary). "
        "Task: TRANSCRIBE HANDWRITTEN TEXT (pen/pencil) visible in the image. "
        "Rules: "
        "- Output field 'handwritten_entries': array of {location_hint, text}. "
        "- location_hint: short phrase like 'top right', 'signature area', 'row 3' (be concise). "
        "- text: transcribe handwritten characters exactly as seen; if handwriting is present but unreadable, use '[UNREADABLE]'. "
        "- Do NOT transcribe printed/typed text. "
        "- If you are not highly confident something is handwritten, EXCLUDE it. "
        "- If there's no handwriting, return an empty array."
    )

    b64 = base64.b64encode(img.read_bytes()).decode("utf-8")
    payload_hw = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": handwriting_prompt, "images": [b64]}],
        "stream": False, "format": handwriting_schema,
    }

    hw_parsed = post_ollama(payload_hw, tag="hw", page_no=page_no, tries=3)
    hw_entries_raw = hw_parsed.get("handwritten_entries", []) if isinstance(hw_parsed, dict) else []
    try:
        hw_conf = float(hw_parsed.get("confidence", 0.0) or 0.0) if isinstance(hw_parsed, dict) else 0.0
    except Exception:
        hw_conf = 0.0

    ocr_text = (ocr_text or "").strip()
    if len(ocr_text) > 6000:
        ocr_text = ocr_text[:6000] + "\n[TRUNCATED]"

    context_prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    payload_ctx = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": context_prompt, "images": [b64]}],
        "stream": False, "format": JSON_SCHEMA,
    }

    ctx_out = {
        "document_type": "", "purpose": "", "issuer": "",
        "handwritten_entries": [], "non_text_visuals": [], "confidence": 0.0,
    }
    ctx_parsed = post_ollama(payload_ctx, tag="ctx", page_no=page_no, tries=3)
    if isinstance(ctx_parsed, dict):
        ctx_out.update(ctx_parsed)

    def normalize_hw_list(raw_list):
        items = []
        for e in raw_list or []:
            if not isinstance(e, dict):
                continue
            loc = sanitize_text(e.get("location_hint", ""), max_len=120) or "unspecified area"
            txt = sanitize_text(e.get("text", ""), max_len=500)
            if not txt:
                continue
            if txt.lower() in ("[handwritten]",):
                continue
            items.append({"location_hint": loc, "text": txt})
        return items

    hw_entries = normalize_hw_list(hw_entries_raw)
    if not hw_entries:
        hw_entries = normalize_hw_list(ctx_out.get("handwritten_entries", []))

    print(f"[debug] page={page_no} hw_entries_before_filter={len(hw_entries)}", flush=True)
    ocr_norm = norm_cmp(ocr_text)
    filtered_hw = []
    for e in hw_entries:
        txt_norm = norm_cmp(e["text"])
        if txt_norm and txt_norm in ocr_norm and len(txt_norm) > 3:
            continue
        filtered_hw.append(e)
    print(f"[debug] page={page_no} hw_entries_after_filter={len(filtered_hw)}", flush=True)

    visuals = ctx_out.get("non_text_visuals", []) or []
    visuals_clean: List[str] = []
    for v in visuals:
        sv = sanitize_text(v, max_len=80)
        if sv:
            visuals_clean.append(sv)
    seen = set()
    visuals_dedup: List[str] = []
    for v in visuals_clean:
        k = v.lower()
        if k not in seen:
            visuals_dedup.append(v)
            seen.add(k)

    try:
        ctx_conf = float(ctx_out.get("confidence", 0.0) or 0.0)
    except Exception:
        ctx_conf = 0.0
    conf = max(hw_conf, ctx_conf) if filtered_hw else (ctx_conf or hw_conf)

    return {
        "document_type": sanitize_text(ctx_out.get("document_type", ""))[:120],
        "purpose": sanitize_text(ctx_out.get("purpose", ""))[:240],
        "issuer": sanitize_text(ctx_out.get("issuer", ""))[:120],
        "handwritten_entries": filtered_hw[:25],
        "non_text_visuals": visuals_dedup[:40],
        "confidence": max(0.0, min(1.0, conf)),
    }

# -------------------------
# PDF summary page
# -------------------------

def build_metadata_pdf(meta_pdf: Path, per_page: List[Dict[str, Any]]) -> None:
    c = canvas.Canvas(str(meta_pdf), pagesize=letter)
    _, page_h = letter
    x0 = 36
    y = page_h - 36
    line_h = 11

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = page_h - 36

    def draw_wrapped(text: str, font: str = "Helvetica", size: int = 9,
                     x: int = x0, wrap_chars: int = 110) -> None:
        nonlocal y
        c.setFont(font, size)
        wrapped = textwrap.wrap(text, width=wrap_chars) if text else [""]
        for ln in wrapped:
            c.drawString(x, y, ln)
            y -= line_h
            if y < 48:
                new_page()
                c.setFont(font, size)

    def heading(text: str) -> None:
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x0, y, text)
        y -= 16
        if y < 48:
            new_page()

    def label_value(label: str, value: str) -> None:
        value = value.strip() or "(unknown)"
        draw_wrapped(f"{label} {value}", font="Helvetica", size=10, wrap_chars=105)

    def first_nonempty(key: str) -> str:
        for p in per_page:
            v = sanitize_text(str(p.get(key, "")).strip(), max_len=240)
            if v:
                return v
        return ""

    doc_type = first_nonempty("document_type")
    issuer = first_nonempty("issuer")
    purpose = first_nonempty("purpose")
    hw: List[Tuple[int, str, str]] = []
    visuals: List[str] = []
    confidences: List[float] = []

    for idx, p in enumerate(per_page, start=1):
        try:
            confidences.append(float(p.get("confidence", 0.0)))
        except Exception:
            pass
        for e in (p.get("handwritten_entries") or []):
            loc = sanitize_text(e.get("location_hint", ""), max_len=60) or "unspecified area"
            txt = sanitize_text(e.get("text", ""), max_len=300)
            if txt:
                hw.append((idx, loc, txt))
        for v in (p.get("non_text_visuals") or []):
            sv = sanitize_text(v, max_len=60)
            if sv:
                visuals.append(sv)

    seen_v = set()
    visuals_dedup: List[str] = []
    for v in visuals:
        lv = v.lower()
        if lv not in seen_v:
            visuals_dedup.append(v)
            seen_v.add(lv)
    visuals = visuals_dedup[:40]
    conf_vals = [c for c in confidences if c > 0]
    avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0

    heading("LLM_METADATA (handwriting + context)")
    draw_wrapped(
        f"Model: {OLLAMA_MODEL} | Pages analyzed: {len(per_page)} (fixed) | Render DPI: {RENDER_DPI} | Confidence(avg): {avg_conf:.2f}",
        font="Helvetica-Oblique", size=8, wrap_chars=120,
    )
    y -= 6
    label_value("Document type:", doc_type)
    label_value("Issuer/source:", issuer)
    label_value("Purpose:", purpose)
    y -= 8
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0, y, "Handwriting captured (delta vs OCR text):")
    y -= 14
    if y < 48:
        new_page()

    if not hw:
        draw_wrapped("(none detected)", font="Helvetica-Oblique", size=9, x=x0 + 6)
    else:
        for (page_no, loc, txt) in hw[:25]:
            draw_wrapped(f"- p{page_no} {loc}: {txt}", font="Helvetica", size=9, x=x0 + 6, wrap_chars=112)

    if visuals:
        y -= 8
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x0, y, "Non-text visuals noticed:")
        y -= 14
        if y < 48:
            new_page()
        draw_wrapped(", ".join(visuals), font="Helvetica", size=9, x=x0 + 6, wrap_chars=112)
    c.showPage()
    c.save()

# -------------------------
# PDF assembly
# -------------------------

def append_pdf(base_pdf: Path, extra_pdf: Path, out_pdf: Path) -> None:
    with pikepdf.open(str(base_pdf)) as base, pikepdf.open(str(extra_pdf)) as extra:
        for page in extra.pages:
            base.pages.append(page)
        base.save(str(out_pdf))

# -------------------------
# Processing pipeline
# -------------------------

def process_one(src_pdf: Path) -> None:
    job_id = sha256(src_pdf)[:16]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    local_in = job_dir / "input.pdf"
    shutil.copy2(src_pdf, local_in)
    normalized = job_dir / "normalized.pdf"
    ocred = job_dir / "ocred.pdf"
    meta_page = job_dir / "llm_meta.pdf"
    final_pdf = job_dir / "final.pdf"

    print(f"[job] start file={src_pdf.name} job_id={job_id} max_pages={MAX_PAGES} dpi={RENDER_DPI} ocr_jobs={OCR_JOBS}", flush=True)
    run(["qpdf", "--warning-exit-0", "--linearize", str(local_in), str(normalized)])

    ocr_env = {
        "OMP_THREAD_LIMIT": str(OCR_THREADS), "OMP_NUM_THREADS": str(OCR_THREADS),
        "OPENBLAS_NUM_THREADS": str(OCR_THREADS), "MKL_NUM_THREADS": str(OCR_THREADS),
        "NUMEXPR_NUM_THREADS": str(OCR_THREADS),
    }
    run([
        "ocrmypdf", "--skip-text", "--rotate-pages", "--deskew", "--optimize", "1",
        "--jobs", str(OCR_JOBS), "-l", OCR_LANG, str(normalized), str(ocred),
    ], extra_env=ocr_env)

    img_dir = job_dir / "images"
    imgs = pdftoimages(ocred, img_dir, max_pages=MAX_PAGES)
    if not imgs:
        raise OllamaAnalysisError("analysis failed: PDF rendering produced no page images")
    per_page: List[Dict[str, Any]] = []
    for img in imgs:
        page_no = _page_num_from_path(img)
        ocr_text = extract_ocr_text_by_page(ocred, page_no)
        print(f"[page] analyzing page={page_no} img={img.name} ocr_chars={len(ocr_text)}", flush=True)
        result = ollama_extract(img, ocr_text, page_no)
        hw_chars = sum(len(e.get("text", "")) for e in result.get("handwritten_entries", []))
        print(f"[page] page={page_no} handwriting_entries={len(result.get('handwritten_entries', []))} hw_chars={hw_chars} visuals={len(result.get('non_text_visuals', []))}", flush=True)
        per_page.append(result)

    build_metadata_pdf(meta_page, per_page)
    append_pdf(ocred, meta_page, final_pdf)

    out_name = src_pdf.name
    if JOBID_PREFIX:
        out_name = f"{job_id}__{out_name}"
    tmp_out = OUT_DIR / (out_name + ".tmp")
    final_out = OUT_DIR / out_name
    shutil.copy2(final_pdf, tmp_out)
    os.replace(tmp_out, final_out)

    archive_dir = INCOMING_DIR / "processed"
    archive_dir.mkdir(exist_ok=True)
    os.replace(src_pdf, archive_dir / src_pdf.name)
    print(f"[job] done file={src_pdf.name} -> {final_out}", flush=True)
    if CLEANUP_WORKDIR:
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass

# -------------------------
# Main loop
# -------------------------

def main() -> None:
    """Track size/mtime across polls and process stable PDFs."""
    seen: Dict[str, Dict[str, float]] = {}
    while True:
        now = time.time()
        pdfs = sorted(INCOMING_DIR.glob("*.pdf"))
        current_keys = {str(p) for p in pdfs}
        for k in list(seen.keys()):
            if k not in current_keys:
                seen.pop(k, None)

        ready: List[Path] = []
        for p in pdfs:
            key = str(p)
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            size = float(st.st_size)
            mtime = float(st.st_mtime)
            if key not in seen:
                seen[key] = {"size": size, "mtime": mtime, "last_change": now}
                continue
            rec = seen[key]
            if rec["size"] != size or rec["mtime"] != mtime:
                rec["size"] = size
                rec["mtime"] = mtime
                rec["last_change"] = now
                continue
            if (now - rec["last_change"]) >= STABLE_SECONDS:
                ready.append(p)

        for pdf in ready:
            key = str(pdf)
            try:
                process_one(pdf)
            except Exception as e:
                bad = INCOMING_DIR / "failed"
                bad.mkdir(exist_ok=True)
                try:
                    os.replace(pdf, bad / pdf.name)
                except Exception:
                    pass
                print(f"[job] FAILED file={pdf.name} err={e!r}", flush=True)
                traceback.print_exc()
            finally:
                seen.pop(key, None)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL] main() crashed:", flush=True)
        traceback.print_exc()
        time.sleep(1.0)
        raise
