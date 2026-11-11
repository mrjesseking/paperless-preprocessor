#!/usr/bin/env python3
"""Container smoke test: real qpdf, OCRmyPDF, Poppler and PDF assembly.

Only the Ollama server is mocked (a local HTTP server returning schema-shaped
JSON, or deliberate failures). No GPU, no real model, and only a synthetic
generated PDF are used. This does NOT measure handwriting accuracy.

Run inside the built image, with this directory mounted, e.g.:
  docker run --rm --entrypoint python -v "$PWD/scripts:/smoke:ro" IMAGE /smoke/smoke_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODE = {"value": "ok"}
REQUESTS = []

CTX = {
    "document_type": "synthetic letter", "purpose": "smoke test", "issuer": "Example Co",
    "handwritten_entries": [], "non_text_visuals": [], "confidence": 0.5,
}
HW = {"handwritten_entries": [{"location_hint": "margin", "text": "synthetic note"}], "confidence": 0.7}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        fmt = body.get("format", {})
        REQUESTS.append(sorted(fmt.get("properties", {})))
        if MODE["value"] == "http500":
            self.send_response(500)
            self.end_headers()
            return
        is_hw = "document_type" not in fmt.get("properties", {})
        payload = HW if is_hw else CTX
        if MODE["value"] == "invalid_ctx" and not is_hw:
            payload = {"document_type": "x"}  # missing required fields
        data = json.dumps({"message": {"content": json.dumps(payload)}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_pdf(path: Path, pages: int) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=letter)
    for i in range(1, pages + 1):
        c.setFont("Helvetica", 24)
        c.drawString(72, 700, f"Synthetic smoke test document page {i}")
        c.drawString(72, 650, "The quick brown fox jumps over the lazy dog")
        c.showPage()
    c.save()


def page_count(path: Path) -> int:
    import pikepdf
    with pikepdf.open(str(path)) as pdf:
        return len(pdf.pages)


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    for tool in ("qpdf", "pdftoppm", "pdftotext", "ocrmypdf", "tesseract", "gs"):
        r = subprocess.run([tool, "--version"], capture_output=True, text=True)
        check(r.returncode == 0, f"{tool} available")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp())
    for sub in ("incoming", "consume", "work"):
        (tmp / sub).mkdir()
    os.environ.update({
        "INCOMING_DIR": str(tmp / "incoming"), "OUT_DIR": str(tmp / "consume"),
        "WORK_DIR": str(tmp / "work"), "OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
        "MAX_PAGES": "2", "OCR_JOBS": "1", "OLLAMA_CONNECT_TIMEOUT": "3", "OLLAMA_READ_TIMEOUT": "30",
    })
    sys.path.insert(0, "/app")
    import processor

    # 1) success path: 3-page PDF, only 2 pages go to vision, OCR covers all pages.
    src = tmp / "incoming" / "ok.pdf"
    make_pdf(src, 3)
    processor.process_one(src)
    outs = list((tmp / "consume").glob("*__ok.pdf"))
    check(len(outs) == 1, "output published into consume")
    check(page_count(outs[0]) >= 4, "output = 3 original pages + metadata page(s)")
    check(len(REQUESTS) == 4, "two vision requests per analyzed page (2 pages -> 4 requests)")
    check((tmp / "incoming" / "processed" / "ok.pdf").exists(), "source archived to processed")
    text = subprocess.check_output(["pdftotext", str(outs[0]), "-"], text=True)
    check("LLM_METADATA" in text, "metadata summary is searchable text")
    check("synthetic note" in text, "mock handwriting entry appears in summary")
    check("fox" in text, "OCR/text layer present")

    # 2) failure paths: nothing published, source remains for the main loop to quarantine.
    for mode in ("http500", "invalid_ctx"):
        MODE["value"] = mode
        before = set((tmp / "consume").iterdir())
        bad = tmp / "incoming" / f"{mode}.pdf"
        make_pdf(bad, 1)
        try:
            processor.process_one(bad)
            check(False, f"{mode}: should have raised")
        except processor.OllamaAnalysisError:
            check(True, f"{mode}: OllamaAnalysisError raised")
        check(set((tmp / "consume").iterdir()) == before, f"{mode}: nothing published")
        check(bad.exists(), f"{mode}: source left in place")
        bad.unlink()
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
