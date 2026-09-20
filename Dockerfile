# syntax=docker/dockerfile:1
#
# Image for the systematic mapping pipeline.
#
#   docker build -t sms-pipeline .
#   docker run --rm -v "$PWD/pdfs:/work/pdfs:ro" -v "$PWD/output:/work/output" sms-pipeline
#
# The corpus is mounted rather than copied in, so the image carries no
# copyrighted material and the same image serves any corpus.

FROM python:3.12-slim AS base

# Tesseract is the OCR engine, needed only for scanned PDFs with no text
# layer. It is a system program; the `tesseract` package on PyPI is an
# unrelated astronomy library and will not work. Including it here means the
# image handles scanned documents that a bare pip install cannot.
# The build toolchain is present because hdbscan and umap-learn compile from
# source where no wheel matches the platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HF_HOME=/work/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

WORKDIR /work

# Dependencies first, so editing the notebook does not invalidate this layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the sentence-transformer weights into the image. Without this the first
# run downloads ~90 MB, which fails in an offline or air-gapped environment and
# makes runs non-reproducible when the upstream model changes.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')" \
    && chmod -R a+rX /work/.cache

COPY sms_pipeline.ipynb README.md ./
COPY experiments/ ./experiments/
COPY docker-entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint \
    && mkdir -p /work/pdfs /work/output \
    && useradd -m -u 1000 sms \
    && chown -R sms:sms /work
USER sms

EXPOSE 8888
ENTRYPOINT ["entrypoint"]
CMD ["run"]
