#!/usr/bin/env python3
"""Which headings do these 126 PDFs actually use, and which go unmapped?

Loads the notebook's own extractor, runs the heading detector over every PDF,
and splits what it finds into headings that map to a canonical section and
headings that do not. The unmapped list, ranked by how many papers use it, is
the evidence for what the alias table is still missing.
"""
import json, re, warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path("/home/naedatatz/Desktop/Notebook")

# --- load the extractor straight out of the notebook -----------------------
ns = {"__name__": "heading_census"}
nb = json.loads((BASE / "sms_pipeline.ipynb").read_text())
# run code cells in order until the extractor class exists
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if src.lstrip().startswith(("dataset =", "print(", "pipeline")):
        continue
    try:
        exec(compile(src, "<cell>", "exec"), ns)
    except Exception as e:
        print(f"  (cell skipped: {str(e)[:70]})")
    if "RobustPDFExtractor" in ns:
        break

RobustPDFExtractor = ns["RobustPDFExtractor"]
HEADING_RX = ns["_HEADING_RX"]
print(f"loaded extractor; {len(HEADING_RX)} canonical sections\n")

ex = RobustPDFExtractor()
mapped = defaultdict(Counter)      # section -> heading text -> papers
unmapped = Counter()
unmapped_papers = defaultdict(set)

pdfs = sorted(f for f in BASE.joinpath("pdfs").iterdir()
              if f.suffix.lower() == ".pdf")
for pdf in pdfs:
    try:
        lines, _ocr, _nat = ex._read_document_lines(__import__("pymupdf").open(pdf))
    except Exception:
        continue
    if not lines:
        continue
    body = ex._body_size(lines) if hasattr(ex, "_body_size") else 10.0
    seen = set()
    for ln in lines:
        t = ln.clean if hasattr(ln, "clean") else ln.text
        if not ex._is_heading(ln, body):
            continue
        canon = ex._canonical(t)
        key = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s*", "", t).strip().lower()
        if not key or len(key) > 45:
            continue
        if canon:
            mapped[canon][key] += 1
        elif key not in seen:
            unmapped[key] += 1
            unmapped_papers[key].add(pdf.name)
            seen.add(key)

print("=" * 74)
print("UNMAPPED HEADINGS, most common first")
print("=" * 74)
for head, n in unmapped.most_common(45):
    print(f"  {n:>4}  {head}")
print(f"\n{len(unmapped)} distinct unmapped headings across {len(pdfs)} PDFs")

out = BASE / "experiments" / "unmapped_headings.csv"
import csv
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["heading", "papers"])
    for head, n in unmapped.most_common():
        w.writerow([head, n])
print(f"wrote {out}")
