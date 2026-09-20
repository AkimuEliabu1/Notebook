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

## Running it locally

Docker is the recommended route and is covered below. To run without it:

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
the OCR engine. The Docker image includes the real one, which is one reason to
prefer it.

## Running with Docker, step by step

Docker is the recommended way to run this. It supplies the Tesseract OCR engine,
which pip cannot install, and it carries the sentence-transformer weights, so
the pipeline works offline and gives the same answers on any machine.

### 1. Check Docker is installed

```bash
docker --version
docker compose version
```

If either is missing, install Docker Engine and the Compose plugin:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"      # then log out and back in
```

### 2. Get the code

```bash
git clone https://github.com/AkimuEliabu1/Notebook.git
cd Notebook
```

### 3. Add your papers

The corpus is not distributed with this repository. Put your own PDFs in
`pdfs/`, one file per primary study:

```bash
mkdir -p pdfs output
cp /path/to/your/papers/*.pdf pdfs/
ls pdfs/*.pdf | wc -l
```

### 4. Build the image

```bash
docker compose build
```

This takes ten to twenty minutes the first time and needs roughly 6 GB of free
disk. It downloads PyTorch, which is large, and then bakes the
sentence-transformer weights into the image. Later builds reuse the cached
layers and finish in seconds unless `requirements.txt` changes.

Check the free space first, because the build fails confusingly when the disk
is full:

```bash
df -h /
```

### 5. Run the pipeline

```bash
docker compose run --rm pipeline
```

The notebook executes end to end: it reads the PDFs, builds the dataset, then
clusters, assigns the facets, models the themes and writes the figures. Expect
roughly a minute of modelling for about 120 papers, plus the time to read the
PDFs.

### 6. Collect the results

Everything is written to `output/` on the host, not inside the container:

```
output/dataset.csv          the corpus: one row per paper, full text included
output/runs/run_<stamp>/    this run's tables, figures and evaluation reports
output/latest               a symlink to the most recent run
```

A new run never overwrites an old one, so runs can be compared.

### 7. Work in the notebook instead

To edit and run cells interactively rather than executing the whole notebook:

```bash
docker compose up notebook
```

Then open <http://127.0.0.1:8888> in a browser. The server is bound to localhost
only, because it runs without a token; do not publish the port to a network.
Stop it with Ctrl-C.

### Other commands

```bash
docker compose run --rm pipeline shell     # a shell inside the container
docker compose build --no-cache            # rebuild from scratch
docker compose down -v                     # remove containers and the model cache
```

### If something goes wrong

**The build runs out of space.** `docker system df` shows what Docker is
holding; `docker image prune -a` reclaims images nothing is using.

**Scanned PDFs are skipped.** Documents with no text layer need OCR. The image
already contains Tesseract, so this should not happen inside Docker; if it does,
the PDF is likely damaged rather than merely scanned.

**Permission errors on `output/`.** The container runs as uid 1000. If your host
user differs, `sudo chown -R $(id -u):$(id -g) output/` after the run.

**The first run is slow.** The sentence-transformer weights are in the image,
but BERTopic still fits from scratch on every run. That is expected.

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
