# Revision notes — 2026-11-11

This handoff contains a revised complete processor.py, a Claude Code repository-creation prompt, and offline regression tests. It is not yet a Docker repository or published image: the original Dockerfile and package versions have not been supplied.

Changes from the script supplied in the conversation:

- Kept qpdf --warning-exit-0, scoped to normalization.
- Changed the fallback OLLAMA_MODEL from llama3.2-vision to gemma4:latest, matching the supplied Compose setting. An explicitly configured environment value still wins. The exact tag must exist on the configured server and support the requested vision/structured output.
- Added OllamaAnalysisError. HTTP errors, request failures, JSON parse failures, and response-schema failures are retried three times per required call, then fail the document. Either the handwriting or context request failing prevents publication. The existing main-loop exception path moves the original to incoming/failed.
- Added validation for the subset of schema used by the script: required object fields, additional-property restriction, nested arrays/objects, strings and finite numeric confidence. This is not a general-purpose JSON Schema validator and does not enforce a new confidence range.
- Kept valid empty handwriting lists as successful results. The summary reports none detected only for the successful analysis path (after existing text sanitization/filtering).
- Reject zero rendered page images instead of creating an empty analysis summary.
- Stopped printing model-response/error-body snippets and arbitrary request exception text in Ollama error logs. Logs still include filenames, configured server URL and other operational information; this is not comprehensive log redaction.
- Retry delays are only between attempts, not after the final failed attempt. Updated version banner and documentation. Reformatted sections without changing the pipeline, prompts or existing output limits.

Behavior intentionally retained:

- One document at a time, two vision calls per analyzed page, first four pages by default. Full-document conventional OCR is separate from the vision page limit.
- Existing prompts, OCR text filtering, 25-entry limits, text truncation, metadata layout and model-generated confidence treatment. None of these guarantees handwriting accuracy or completeness.
- Original archiving, consume publication, work cleanup and file-stability polling. Existing filename collision/overwrite behavior is unchanged; this revision is not a comprehensive reliability audit.
- Failure routing uses the pre-existing move attempt. If permissions prevent that move, the original code logs the job failure but suppresses the move exception; no guarantee is made that an inaccessible file can be quarantined.

## Validation

Run from this folder:

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile processor.py tests/test_processor.py
```

The tests use synthetic bytes, mocked HTTP, mocked PDF libraries and mocked subprocesses. They verify retry/failure behavior, response validation, model selection, a successful document lifecycle and withholding output/quarantining a failed job. They do not run actual OCR, Docker, a Raspberry Pi, or a real Ollama model. The Claude Code prompt requires those feasible build/smoke checks separately. No production stack has been changed by creating this handoff.
