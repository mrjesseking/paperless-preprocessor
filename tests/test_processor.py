"""Offline regression tests; PDF libraries and HTTP are stubbed, not integration-tested."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "processor.py"


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = io.StringIO()
        self.redirect_out = contextlib.redirect_stdout(self.log)
        self.redirect_err = contextlib.redirect_stderr(self.log)
        self.redirect_out.__enter__()
        self.redirect_err.__enter__()
        self.addCleanup(self.redirect_out.__exit__, None, None, None)
        self.addCleanup(self.redirect_err.__exit__, None, None, None)
        self.p = self.load_processor()

    def load_processor(self, model=None):
        modules = {name: ModuleType(name) for name in (
            "pikepdf", "requests", "reportlab", "reportlab.lib",
            "reportlab.lib.pagesizes", "reportlab.pdfgen",
        )}
        modules["requests"].post = Mock()
        modules["reportlab.lib.pagesizes"].letter = (612, 792)
        modules["reportlab.pdfgen"].canvas = SimpleNamespace(Canvas=Mock())
        config = {
            "INCOMING_DIR": str(self.root / "incoming"),
            "OUT_DIR": str(self.root / "consume"),
            "WORK_DIR": str(self.root / "work"),
            "STABLE_SECONDS": "0", "POLL_SECONDS": "0",
        }
        spec = importlib.util.spec_from_file_location("processor_under_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, config), patch.dict(sys.modules, modules), patch.object(sys, "excepthook", sys.excepthook):
            os.environ.pop("OLLAMA_MODEL", None)
            if model is not None:
                os.environ["OLLAMA_MODEL"] = model
            spec.loader.exec_module(module)
        return module

    def response(self, value=None, status=200, raw=None):
        response = Mock(status_code=status)
        response.json.return_value = {
            "message": {"content": raw if raw is not None else json.dumps(value)}
        }
        return response

    def context(self, **changes):
        value = dict(document_type="form", purpose="Test fixture", issuer="",
                     handwritten_entries=[], non_text_visuals=[], confidence=0.8)
        value.update(changes)
        return value

    def post_context(self):
        return self.p.post_ollama({"format": self.p.JSON_SCHEMA}, "ctx", 1)

    def test_default_and_environment_override(self):
        self.assertEqual(self.p.OLLAMA_MODEL, "gemma4:latest")
        custom = self.load_processor("my-installed-vision-model:custom")
        self.assertEqual(custom.OLLAMA_MODEL, "my-installed-vision-model:custom")

    def test_successful_no_handwriting_is_valid(self):
        self.p.requests.post.return_value = self.response(self.context())
        self.assertEqual(self.post_context()["handwritten_entries"], [])
        self.assertEqual(self.p.requests.post.call_count, 1)

    def test_http_errors_exhaust_retries_and_raise(self):
        self.p.requests.post.return_value = self.response(status=503)
        with patch.object(self.p.time, "sleep") as sleep:
            with self.assertRaisesRegex(self.p.OllamaAnalysisError, "analysis failed.*stage=ctx"):
                self.post_context()
        self.assertEqual(self.p.requests.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_connection_failure_raises(self):
        self.p.requests.post.side_effect = TimeoutError("private document snippet")
        with patch.object(self.p.time, "sleep"):
            with self.assertRaises(self.p.OllamaAnalysisError):
                self.post_context()
        self.assertNotIn("private document snippet", self.log.getvalue())

    def test_invalid_response_variants_are_not_success(self):
        invalid = [
            self.response(raw="not JSON: private document snippet"),
            self.response({}),
            self.response({"unrelated": "value"}),
            self.response(self.context(handwritten_entries="none")),
            self.response(self.context(handwritten_entries=[{"text": "missing location"}])),
            self.response(self.context(confidence=True)),
            self.response(self.context(confidence=float("nan"))),
            self.response(self.context(non_text_visuals=[123])),
        ]
        with patch.object(self.p.time, "sleep"):
            for response in invalid:
                with self.subTest(response=response):
                    self.p.requests.post.reset_mock()
                    self.p.requests.post.return_value = response
                    with self.assertRaises(self.p.OllamaAnalysisError):
                        self.post_context()
                    self.assertEqual(self.p.requests.post.call_count, 3)
        self.assertNotIn("private document snippet", self.log.getvalue())

    def test_retry_can_recover(self):
        self.p.requests.post.side_effect = [self.response(status=503), self.response(self.context())]
        with patch.object(self.p.time, "sleep"):
            self.assertEqual(self.post_context()["document_type"], "form")
        self.assertEqual(self.p.requests.post.call_count, 2)

    def test_handwriting_and_context_use_configured_model(self):
        image = self.root / "page-1.png"
        image.write_bytes(b"synthetic test image placeholder")
        handwriting = {"handwritten_entries": [{"location_hint": "top", "text": "Call Tuesday"}], "confidence": 0.9}
        self.p.requests.post.side_effect = [self.response(handwriting), self.response(self.context())]
        result = self.p.ollama_extract(image, "Printed title", 1)
        self.assertEqual(result["handwritten_entries"][0]["text"], "Call Tuesday")
        self.assertEqual(self.p.requests.post.call_count, 2)
        for call in self.p.requests.post.call_args_list:
            self.assertEqual(call.kwargs["json"]["model"], "gemma4:latest")

    def test_context_failure_after_successful_handwriting_fails_page(self):
        image = self.root / "page-1.png"
        image.write_bytes(b"synthetic test image placeholder")
        self.p.requests.post.side_effect = [
            self.response({"handwritten_entries": [], "confidence": 0.8}),
            self.response(status=503), self.response(status=503), self.response(status=503),
        ]
        with patch.object(self.p.time, "sleep"):
            with self.assertRaisesRegex(self.p.OllamaAnalysisError, "stage=ctx"):
                self.p.ollama_extract(image, "", 1)

    def test_failure_quarantines_original_without_publishing_pdf(self):
        original = self.p.INCOMING_DIR / "synthetic.pdf"
        original.write_bytes(b"synthetic PDF placeholder")
        with patch.object(self.p, "run"), \
             patch.object(self.p, "pdftoimages", return_value=[self.root / "page-1.png"]), \
             patch.object(self.p, "extract_ocr_text_by_page", return_value=""), \
             patch.object(self.p, "ollama_extract", side_effect=self.p.OllamaAnalysisError("analysis failed")), \
             patch.object(self.p, "build_metadata_pdf") as summary, \
             patch.object(self.p.time, "sleep", side_effect=[None, StopIteration]):
            with self.assertRaises(StopIteration):
                self.p.main()
        self.assertFalse(original.exists())
        self.assertEqual((self.p.INCOMING_DIR / "failed" / "synthetic.pdf").read_bytes(), b"synthetic PDF placeholder")
        self.assertEqual(list(self.p.OUT_DIR.iterdir()), [])
        summary.assert_not_called()
        self.assertNotIn("[job] done", self.log.getvalue())

    def test_no_rendered_pages_fails_without_publishing(self):
        original = self.p.INCOMING_DIR / "synthetic.pdf"
        original.write_bytes(b"synthetic PDF placeholder")
        with patch.object(self.p, "run"), patch.object(self.p, "pdftoimages", return_value=[]):
            with self.assertRaisesRegex(self.p.OllamaAnalysisError, "no page images"):
                self.p.process_one(original)
        self.assertTrue(original.exists())
        self.assertEqual(list(self.p.OUT_DIR.iterdir()), [])

    def test_success_publishes_and_archives(self):
        original = self.p.INCOMING_DIR / "synthetic.pdf"
        original.write_bytes(b"synthetic PDF placeholder")
        def assemble(base, extra, destination):
            destination.write_bytes(b"synthetic completed PDF")
        with patch.object(self.p, "run") as commands, \
             patch.object(self.p, "pdftoimages", return_value=[self.root / "page-1.png"]), \
             patch.object(self.p, "extract_ocr_text_by_page", return_value=""), \
             patch.object(self.p, "ollama_extract", return_value=self.context()), \
             patch.object(self.p, "build_metadata_pdf"), \
             patch.object(self.p, "append_pdf", side_effect=assemble):
            self.p.process_one(original)
        self.assertIn("--warning-exit-0", commands.call_args_list[0].args[0])
        self.assertEqual(len(list(self.p.OUT_DIR.glob("*.pdf"))), 1)
        self.assertTrue((self.p.INCOMING_DIR / "processed" / "synthetic.pdf").exists())
        self.assertFalse(original.exists())
        self.assertEqual(list(self.p.WORK_DIR.iterdir()), [])

    def test_real_subprocess_errors_are_still_fatal(self):
        with patch.object(self.p.subprocess, "check_call", side_effect=subprocess.CalledProcessError(2, ["qpdf"])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.p.run(["qpdf"])


if __name__ == "__main__":
    unittest.main()
