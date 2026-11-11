# paperless-preprocessor

Watches a folder for PDFs, normalizes and OCRs them, asks a vision model (served by your own
[Ollama](https://ollama.com) server) to transcribe handwriting and describe document context,
appends a searchable metadata page, and drops the result into [Paperless-ngx](https://docs.paperless-ngx.com/)'s
consume folder.

Image: `ghcr.io/mrjesseking/paperless-preprocessor` (linux/amd64, linux/arm64).

## Architecture

```
incoming/ ──> qpdf normalize ──> OCRmyPDF ──> Poppler render (first MAX_PAGES pages)
                                                   │ page images + selected OCR text
                                                   ▼
                                       Ollama at OLLAMA_URL (2 calls/page: handwriting, context)
                                                   │
          consume/  <── append LLM_METADATA page <─┘        incoming/processed (source archived)
```

- The container (e.g. on a Raspberry Pi) does the PDF/OCR work. Ollama can run on another machine. Paperless consumes the output.
- **Privacy:** page images and up to 6000 characters of OCR text per page are sent to `OLLAMA_URL`. The endpoint you configure decides where document content goes. The image contains no model weights.
- One worker, one PDF at a time.

## Behavior you should know about

- **Vision is limited to the first `MAX_PAGES` pages (default 4).** Raise `MAX_PAGES` to analyze more. Ordinary OCR (OCRmyPDF) still processes the *entire* PDF regardless of this limit.
- **LLM output can be wrong.** `confidence` is a number the model reports about itself, not a calibrated accuracy figure. Treat handwriting transcriptions as hints.
- **Not every handwritten mark is kept.** Entries whose text already appears in the OCR text (length > 3 after normalization) are dropped, `[Handwritten]` placeholders are dropped, text is truncated (500 chars/entry), and at most 25 entries per page and 40 visuals are kept.
- **Empty handwriting is valid.** A well-formed response with `handwritten_entries: []` means "none detected" and is not an error.
- **Failures are strict.** If either vision call fails after 3 attempts (HTTP error, unparsable output, missing or wrongly typed fields), or no page images render, the document fails and **nothing is published to `consume/`**. The main loop moves the source to `incoming/failed/` and continues with other files. There is no fallback to a different model.
- `qpdf --warning-exit-0` is used only for the normalization step. Any actual subprocess error still fails the job.

## Quick start

```bash
cp .env.example .env            # edit OLLAMA_URL / OLLAMA_MODEL
cp docker-compose.example.yml docker-compose.yml   # edit host paths
docker compose up -d
docker compose logs -f preprocessor
```

Drop a PDF in the host `incoming` folder. After `STABLE_SECONDS` of unchanged size/mtime it is processed.

### Host directories

The container runs as root by default, so bind-mounted folders need to be readable and writable by root (or run with `user:` set to an account that owns them).
It must be able to: read/write/rename in `incoming` (creates `incoming/processed` and `incoming/failed`), write to `consume`, and write to `work`. Keep `incoming` and `consume` on filesystems where renames/copies are permitted; the publish step writes `<name>.tmp` then renames it so Paperless never sees a partial file.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `INCOMING_DIR` | `/nas/incoming` | Watched folder (`*.pdf` at top level only) |
| `OUT_DIR` | `/nas/consume` | Finished PDFs (Paperless consume folder) |
| `WORK_DIR` | `/nas/work` | Per-job scratch (`<job_id>/`) |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama base URL; document content is sent here |
| `OLLAMA_MODEL` | `gemma4:latest` | Exact model tag; must support images and structured output |
| `OLLAMA_CONNECT_TIMEOUT` | `10` | Seconds |
| `OLLAMA_READ_TIMEOUT` | `600` | Seconds |
| `STABLE_SECONDS` | `20` | Seconds a file must be unchanged before processing |
| `POLL_SECONDS` | `5` | Folder poll interval |
| `MAX_PAGES` | `4` | Pages rendered and sent to the vision model |
| `RENDER_DPI` | `150` | Render resolution for vision |
| `OCR_LANG` | `eng` | Tesseract language(s). Only English data is installed in the image |
| `OCR_JOBS` | `3` | OCRmyPDF `--jobs` |
| `OCR_THREADS` | `OCR_JOBS` | Caps OMP/OpenBLAS/MKL/NumExpr threads for OCR |
| `JOBID_PREFIX` | `1` | `1` = output named `<16-hex-sha256>__<original>.pdf`; `0` = original name |
| `CLEANUP_WORKDIR` | `1` | `0` keeps the job folder for debugging |

### Choosing and confirming the model

`gemma4:latest` is the owner's configured tag and the script's default. It is not a claim that your server has it. Confirm on the Ollama host:

```bash
ollama list                       # is the exact tag installed?
ollama show gemma4:latest         # check capabilities include vision
curl -s http://YOUR_OLLAMA:11434/api/tags | grep -o '"name":"[^"]*"'
```

Set `OLLAMA_MODEL` to the exact tag. Use a vision-capable model that honors the structured-output `format` schema. There is no silent fallback to any other model.

## Archive, failures and retries

- Success: result in `OUT_DIR`; source moved to `incoming/processed/`.
- Failure: no output; source moved to `incoming/failed/`; the loop continues.
- **Retry:** fix the cause (Ollama down, wrong tag, ...) and move the PDF from `incoming/failed/` back into `incoming/`.

## Logs

`docker compose logs preprocessor` (stdout). Per-request lines report page, stage, attempt, status and timing. Failed-request logs no longer dump model responses, HTTP bodies or exception text. Filenames and job ids still appear in logs, and `OLLAMA_URL` appears in the startup line.

## Architectures

The published manifest targets `linux/amd64` and `linux/arm64`. A 32-bit Raspberry Pi OS (armv7) is **not** supported. Check with `uname -m` (`aarch64` is fine, `armv7l` is not). The architecture of your Pi has not been verified by this project.

## Development and testing

- `python -m pip install -r requirements-dev.txt && python -m pytest tests` (Python 3.12). These **mocked** offline tests stub HTTP and PDF libraries and test control flow and validation only.
- `scripts/smoke_test.py` runs inside the built image with real qpdf, OCRmyPDF, Poppler and PDF assembly on a generated synthetic PDF, against a mock Ollama HTTP server (valid responses plus deliberate HTTP 500 and invalid-schema failures). No GPU. CI runs it. Locally:
  ```bash
  docker build -t paperless-preprocessor:dev .
  docker run --rm --entrypoint python -v "$PWD/scripts:/smoke:ro" paperless-preprocessor:dev /smoke/smoke_test.py
  ```
- **Not covered by CI: real-model handwriting accuracy.** That needs your actual Ollama server and a representative *non-sensitive* scan. Run the container against it and review the `LLM_METADATA` page.

CI (`.github/workflows/ci.yml`) runs tests and builds (no push, read-only permissions) on PRs and pushes. `publish.yml` publishes only from `main`, `v*` tags, or a manual run started from those refs. Tags: `latest` and `sha-<short>` for `main`; `X.Y.Z` (+ `sha-<short>`) for `vX.Y.Z` tags.

## Upgrading from the local image

See [docs/MIGRATION.md](docs/MIGRATION.md).

## License

No license has been chosen yet. Until one is added, all rights are reserved by the author.
