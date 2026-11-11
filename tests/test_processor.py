"""Offline regression tests for processor.py.

HTTP and PDF libraries are stubbed, so these tests need no Docker, GPU, Ollama
server, or real documents. They cover the control flow and validation logic only;
see scripts/smoke_test.py for the real-tool container check.
"""
import importlib
import math
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _stub_modules():
    """Install lightweight stand-ins for third-party modules if they are missing."""
    for name in ("pikepdf", "requests"):
        sys.modules[name] = mock.MagicMock(name=name)
    rl = types.ModuleType("reportlab")
    lib = types.ModuleType("reportlab.lib")
    pagesizes = types.ModuleType("reportlab.lib.pagesizes")
    pagesizes.letter = (612, 792)
    pdfgen = types.ModuleType("reportlab.pdfgen")
    canvas = types.ModuleType("reportlab.pdfgen.canvas")
    canvas.Canvas = mock.MagicMock(name="Canvas")
    pdfgen.canvas = canvas
    for n, m in (
        ("reportlab", rl), ("reportlab.lib", lib), ("reportlab.lib.pagesizes", pagesizes),
        ("reportlab.pdfgen", pdfgen), ("reportlab.pdfgen.canvas", canvas),
    ):
        sys.modules[n] = m


@pytest.fixture()
def proc(tmp_path, monkeypatch):
    _stub_modules()
    for var, sub in (("INCOMING_DIR", "incoming"), ("OUT_DIR", "consume"), ("WORK_DIR", "work")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("processor", None)
    module = importlib.import_module("processor")
    yield module
    sys.modules.pop("processor", None)


VALID_CTX = {
    "document_type": "letter", "purpose": "test", "issuer": "",
    "handwritten_entries": [], "non_text_visuals": [], "confidence": 0.5,
}


def _resp(status, content=""):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = {"message": {"content": content}}
    return r


# ---------- validate_analysis ----------

def test_validate_accepts_valid_with_empty_handwriting(proc):
    proc.validate_analysis(dict(VALID_CTX), proc.JSON_SCHEMA)


@pytest.mark.parametrize("missing", list(VALID_CTX))
def test_validate_rejects_missing_field(proc, missing):
    bad = dict(VALID_CTX)
    del bad[missing]
    with pytest.raises(ValueError):
        proc.validate_analysis(bad, proc.JSON_SCHEMA)


@pytest.mark.parametrize("field,value", [
    ("document_type", 5), ("handwritten_entries", "none"), ("non_text_visuals", {}),
    ("confidence", "high"), ("confidence", True), ("confidence", math.nan),
    ("confidence", math.inf),
])
def test_validate_rejects_wrong_type(proc, field, value):
    bad = dict(VALID_CTX)
    bad[field] = value
    with pytest.raises(ValueError):
        proc.validate_analysis(bad, proc.JSON_SCHEMA)


def test_validate_rejects_unexpected_and_malformed_entries(proc):
    bad = dict(VALID_CTX, extra=1)
    with pytest.raises(ValueError):
        proc.validate_analysis(bad, proc.JSON_SCHEMA)
    bad = dict(VALID_CTX, handwritten_entries=[{"text": "x"}])
    with pytest.raises(ValueError):
        proc.validate_analysis(bad, proc.JSON_SCHEMA)


# ---------- post_ollama ----------

def test_post_ollama_success(proc, monkeypatch):
    import json
    proc.requests.post.return_value = _resp(200, json.dumps(VALID_CTX))
    out = proc.post_ollama({"format": proc.JSON_SCHEMA}, "ctx", 1)
    assert out["confidence"] == 0.5


@pytest.mark.parametrize("response", [
    _resp(500), _resp(200, ""), _resp(200, "not json"), _resp(200, '{"confidence": 1}'),
])
def test_post_ollama_raises_after_retries(proc, monkeypatch, response):
    monkeypatch.setattr(proc.time, "sleep", lambda s: None)
    proc.requests.post.return_value = response
    proc.requests.post.side_effect = None
    with pytest.raises(proc.OllamaAnalysisError):
        proc.post_ollama({"format": proc.JSON_SCHEMA}, "ctx", 2, tries=3)
    assert proc.requests.post.call_count == 3


def test_post_ollama_request_exception_does_not_leak_text(proc, monkeypatch, capsys):
    monkeypatch.setattr(proc.time, "sleep", lambda s: None)
    proc.requests.post.side_effect = RuntimeError("secret-token-in-url")
    with pytest.raises(proc.OllamaAnalysisError) as exc:
        proc.post_ollama({"format": proc.JSON_SCHEMA}, "hw", 1, tries=2)
    assert "secret-token-in-url" not in str(exc.value)
    assert "secret-token-in-url" not in capsys.readouterr().out
    proc.requests.post.side_effect = None


# ---------- ollama_extract ----------

def _png(tmp_path):
    p = tmp_path / "page-1.png"
    p.write_bytes(b"\x89PNG fake")
    return p


def test_extract_empty_handwriting_is_valid(proc, tmp_path, monkeypatch):
    calls = []

    def fake(payload, tag, page_no, tries=3):
        calls.append(tag)
        if tag == "hw":
            return {"handwritten_entries": [], "confidence": 0.9}
        return dict(VALID_CTX)

    monkeypatch.setattr(proc, "post_ollama", fake)
    out = proc.ollama_extract(_png(tmp_path), "printed text", 1)
    assert calls == ["hw", "ctx"]  # two vision calls per page
    assert out["handwritten_entries"] == []


@pytest.mark.parametrize("failing", ["hw", "ctx"])
def test_extract_propagates_required_call_failure(proc, tmp_path, monkeypatch, failing):
    def fake(payload, tag, page_no, tries=3):
        if tag == failing:
            raise proc.OllamaAnalysisError("boom")
        return {"handwritten_entries": [], "confidence": 0.1} if tag == "hw" else dict(VALID_CTX)

    monkeypatch.setattr(proc, "post_ollama", fake)
    with pytest.raises(proc.OllamaAnalysisError):
        proc.ollama_extract(_png(tmp_path), "", 1)


def test_extract_filters_ocr_duplicates_and_caps_entries(proc, tmp_path, monkeypatch):
    entries = [{"location_hint": "top", "text": "Already In OCR"}] + [
        {"location_hint": f"row {i}", "text": f"note{i}x"} for i in range(40)
    ]
    entries.append({"location_hint": "x", "text": "[Handwritten]"})

    def fake(payload, tag, page_no, tries=3):
        if tag == "hw":
            return {"handwritten_entries": entries, "confidence": 0.8}
        return dict(VALID_CTX)

    monkeypatch.setattr(proc, "post_ollama", fake)
    out = proc.ollama_extract(_png(tmp_path), "xx already in ocr yy", 1)
    texts = [e["text"] for e in out["handwritten_entries"]]
    assert "Already In OCR" not in texts
    assert "[Handwritten]" not in texts
    assert len(texts) == 25  # entry cap


# ---------- process_one ----------

def _stub_pipeline(proc, monkeypatch, imgs, extract):
    cmds = []
    monkeypatch.setattr(proc, "run", lambda cmd, extra_env=None: cmds.append(list(cmd)))
    monkeypatch.setattr(proc, "pdftoimages", lambda pdf, d, max_pages: imgs)
    monkeypatch.setattr(proc, "extract_ocr_text_by_page", lambda pdf, n: "")
    monkeypatch.setattr(proc, "ollama_extract", extract)
    monkeypatch.setattr(proc, "build_metadata_pdf", lambda p, pp: None)
    monkeypatch.setattr(proc, "append_pdf", lambda a, b, out: out.write_bytes(b"%PDF final"))
    return cmds


def _src(proc):
    src = proc.INCOMING_DIR / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 synthetic")
    return src


def test_qpdf_warning_flag_scoped_to_qpdf(proc, monkeypatch):
    img = proc.WORK_DIR / "page-1.png"
    cmds = _stub_pipeline(proc, monkeypatch, [img], lambda *a: dict(VALID_CTX))
    proc.process_one(_src(proc))
    qpdf = next(c for c in cmds if c[0] == "qpdf")
    ocr = next(c for c in cmds if c[0] == "ocrmypdf")
    assert "--warning-exit-0" in qpdf
    assert "--warning-exit-0" not in ocr
    assert ocr[ocr.index("--optimize") + 1] == "1"


def test_success_publishes_and_archives(proc, monkeypatch):
    img = proc.WORK_DIR / "page-1.png"
    _stub_pipeline(proc, monkeypatch, [img], lambda *a: dict(VALID_CTX))
    src = _src(proc)
    proc.process_one(src)
    outs = list(proc.OUT_DIR.glob("*.pdf"))
    assert len(outs) == 1 and outs[0].name.endswith("__doc.pdf")
    assert not list(proc.OUT_DIR.glob("*.tmp"))
    assert (proc.INCOMING_DIR / "processed" / "doc.pdf").exists()
    assert not src.exists()


def test_vision_failure_publishes_nothing(proc, monkeypatch):
    img = proc.WORK_DIR / "page-1.png"

    def boom(*a):
        raise proc.OllamaAnalysisError("analysis failed")

    _stub_pipeline(proc, monkeypatch, [img], boom)
    src = _src(proc)
    with pytest.raises(proc.OllamaAnalysisError):
        proc.process_one(src)
    assert not list(proc.OUT_DIR.iterdir())
    assert src.exists()  # not archived; main loop quarantines it


def test_zero_rendered_pages_fails(proc, monkeypatch):
    _stub_pipeline(proc, monkeypatch, [], lambda *a: dict(VALID_CTX))
    with pytest.raises(proc.OllamaAnalysisError):
        proc.process_one(_src(proc))
    assert not list(proc.OUT_DIR.iterdir())


def test_subprocess_error_still_fails(proc, monkeypatch):
    import subprocess

    def failing(cmd, extra_env=None):
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(proc, "run", failing)
    with pytest.raises(subprocess.CalledProcessError):
        proc.process_one(_src(proc))
    assert not list(proc.OUT_DIR.iterdir())


# ---------- main loop quarantine ----------

def test_main_moves_failed_source_to_failed_dir(proc, monkeypatch):
    src = _src(proc)
    monkeypatch.setattr(proc, "STABLE_SECONDS", 0)

    def boom(pdf):
        raise proc.OllamaAnalysisError("analysis failed")

    monkeypatch.setattr(proc, "process_one", boom)
    sleeps = {"n": 0}

    def fake_sleep(s):
        sleeps["n"] += 1
        if sleeps["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(proc.time, "sleep", fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        proc.main()
    assert (proc.INCOMING_DIR / "failed" / "doc.pdf").exists()
    assert not src.exists()
