# Automating Systematic Mapping Studies

An unsupervised machine-learning pipeline that turns a folder of full-text
research papers into the structured data a systematic mapping study needs: the
three mapping facets, thematic clusters, and the figures that make up the map.

MSc Data Science dissertation work, University of Dar es Salaam.

## What it does

The pipeline consumes no labelled training data at any stage. Facets are
assigned zero-shot, by comparing each paper against natural-language
descriptions of the published category definitions rather than by matching
keywords or learning from examples. Every assignment carries a confidence
margin, and low-margin assignments are flagged for review.

`sms_pipeline.ipynb` runs in two phases:

**Part A, steps 1-7** reads the PDFs once and writes `output/dataset.csv`: one
row per paper with the title, abstract, publication year, author keywords, the
six body sections and the full text. This is the slow phase.

**Part B, steps 8-28** models from that table and never opens a PDF again, so it
re-runs in under a minute while you tune it. It compares four text
representations, clusters with k-means and HDBSCAN, assigns the facets, models
themes with BERTopic, tunes hyperparameters by grid search, and writes the
figures and evaluation reports.

## Running it

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m jupyter notebook sms_pipeline.ipynb
```

Put your PDFs in `pdfs/` and run the notebook top to bottom. Each modelling run
writes to its own timestamped folder under `output/runs/`, and `output/latest`
points at the most recent, so runs never overwrite one another.

Scanned PDFs with no text layer additionally need the Tesseract OCR engine,
which pip cannot install:

```bash
sudo apt-get install -y tesseract-ocr
```

Note that the `tesseract` package on PyPI is an unrelated astronomy library, not
the OCR engine.

## What is not in this repository

The corpus of 126 primary studies is not included. Those are published papers
from IEEE, ACM, Springer and Elsevier, and they are not ours to redistribute;
neither the PDFs nor the full text extracted from them is committed here.
Supply your own PDFs and Part A will rebuild the dataset.

The trained BERTopic model and the pickled pipeline are also excluded. They are
88 MB and 10 MB respectively and are rebuilt by Part B in under a minute.

## Layout

```
sms_pipeline.ipynb     the pipeline, as 28 explained steps
requirements.txt       pinned to the versions the results were produced with
experiments/           standalone analyses: algorithm benchmark, representation
                       comparison, heading census, outlier impact
output/runs/<run>/     per-run results, figures and evaluation reports
```

## A caution on the results

The evaluation establishes internal consistency and aggregate correspondence
with a published manual mapping. It does not establish per-paper accuracy,
because no hand-annotated sample was available. The pipeline writes a
pre-populated labelling template on every run so that this can be remedied.
