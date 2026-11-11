# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# qpdf: normalization. poppler-utils: pdftoppm + pdftotext. tesseract (+eng data)
# and ghostscript: used by OCRmyPDF. No model weights are included; Ollama runs elsewhere.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      qpdf \
      poppler-utils \
      tesseract-ocr \
      tesseract-ocr-eng \
      ghostscript \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY processor.py /app/processor.py

CMD ["python", "/app/processor.py"]
