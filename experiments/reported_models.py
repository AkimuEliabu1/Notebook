#!/usr/bin/env python3
"""Which models and tools do the primary studies report good results with?

Two counts are kept apart, because they mean different things:

  mentioned  - the paper names the technique anywhere
  endorsed   - the technique is named in a sentence that also reports a
               positive outcome ("outperformed", "achieved 92% accuracy",
               "reduced effort by", "improved", "best results")

A high mention count only says the technique is popular. The endorsement count
is the one that answers "what worked". Both are raw counts of sentences the
papers wrote; nothing here is inferred.
"""
import re, warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path("/home/naedatatz/Desktop/Notebook")

TECHNIQUES = {
    # --- BDD / acceptance-testing tooling ---
    "Cucumber": [r"\bcucumber\b"],
    "JBehave": [r"\bjbehave\b"],
    "SpecFlow": [r"\bspecflow\b"],
    "Behat": [r"\bbehat\b"],
    "FitNesse": [r"\bfitnesse\b"],
    "Robot Framework": [r"\brobot framework\b"],
    "Selenium": [r"\bselenium\b"],
    "JUnit": [r"\bjunit\b"],
    "Gherkin": [r"\bgherkin\b"],
    # --- classical machine learning ---
    "SVM": [r"\bsvm\b", r"support vector machine"],
    "Random Forest": [r"random forest"],
    "Naive Bayes": [r"na[iï]ve bayes"],
    "Decision Tree": [r"decision tree"],
    "k-NN": [r"\bk-?nn\b", r"nearest neighbou?r"],
    "Logistic Regression": [r"logistic regression"],
    "K-Means": [r"k-?means"],
    "LDA (topic model)": [r"\blda\b", r"latent dirichlet"],
    # --- neural / transformer ---
    "BERT": [r"\bbert\b(?!opic)"],
    "Sentence-BERT": [r"sentence-?bert", r"\bsbert\b"],
    "BERTopic": [r"\bbertopic\b"],
    "GPT / LLM": [r"\bgpt\b", r"large language model", r"\bllm\b", r"chatgpt"],
    "LSTM / RNN": [r"\blstm\b", r"\brnn\b", r"recurrent neural"],
    "CNN": [r"\bcnn\b", r"convolutional neural"],
    "word2vec / GloVe": [r"word2vec", r"\bglove\b", r"word embedding"],
    "Transformer": [r"\btransformer\b"],
    # --- NLP tooling & representation ---
    "TF-IDF": [r"tf-?idf"],
    "spaCy": [r"\bspacy\b"],
    "NLTK": [r"\bnltk\b"],
    "Stanford / CoreNLP": [r"stanford (?:nlp|parser|corenlp)", r"corenlp"],
    "WordNet": [r"\bwordnet\b"],
    "POS tagging": [r"part-of-speech", r"\bpos tagg"],
    "Named Entity Recognition": [r"named entity recognition", r"\bner\b"],
    # --- model-driven / formal ---
    "Ontology / OWL": [r"\bontolog", r"\bowl\b"],
    "UML": [r"\buml\b"],
    "State machine": [r"state ?machine", r"statechart"],
    "Model-driven engineering": [r"model-driven", r"\bmde\b"],
    "Petri net": [r"petri net"],
}

POSITIVE = re.compile(
    r"\b(?:outperform\w*|out-?performed|achiev\w+|best (?:result|performance|accuracy)"
    r"|highest (?:accuracy|precision|recall|f1|f-measure|score)"
    r"|improv\w+|increas\w+ (?:the )?(?:accuracy|precision|recall|coverage|quality)"
    r"|reduc\w+ (?:the )?(?:effort|time|cost|defect)"
    r"|effective\w*|promising|successful\w*|significant\w* better"
    r"|accuracy of \d|precision of \d|recall of \d|f1[ -]?score of \d|\d\d(?:\.\d+)?\s*%)\b",
    re.I)

COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in TECHNIQUES.items()}


def sentences(text, lo=30, hi=400):
    text = re.sub(r"\s+", " ", str(text))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if lo < len(s) < hi]


def main():
    df = pd.read_csv(BASE / "output" / "dataset.csv")
    print(f"scanning the full text of {len(df)} papers\n")

    mentioned = defaultdict(set)
    endorsed = defaultdict(set)
    quotes = defaultdict(list)

    for _, row in df.iterrows():
        pid, full = row["paper_id"], str(row.get("full_text", "") or "")
        low = full.lower()
        present = [t for t, pats in COMPILED.items() if any(p.search(low) for p in pats)]
        for t in present:
            mentioned[t].add(pid)
        if not present:
            continue
        for s in sentences(full):
            if not POSITIVE.search(s):
                continue
            for t in present:
                if any(p.search(s) for p in COMPILED[t]):
                    endorsed[t].add(pid)
                    if len(quotes[t]) < 3 and len(s) < 230:
                        quotes[t].append((pid, s))

    rows = [{"technique": t,
             "papers_mentioning": len(mentioned[t]),
             "papers_endorsing": len(endorsed[t]),
             "endorsement_rate": round(len(endorsed[t]) / len(mentioned[t]), 3)
                                 if mentioned[t] else 0.0}
            for t in TECHNIQUES if mentioned[t]]
    tab = pd.DataFrame(rows).sort_values(
        ["papers_endorsing", "papers_mentioning"], ascending=False)
    tab.to_csv(BASE / "experiments" / "reported_models.csv", index=False)

    print("=" * 78)
    print("TECHNIQUES REPORTED WITH POSITIVE RESULTS")
    print("=" * 78)
    print(tab.head(20).to_string(index=False))

    print("\n" + "=" * 78)
    print("WHAT THE PAPERS ACTUALLY SAY  (top 6 by endorsement)")
    print("=" * 78)
    for t in tab.head(6).technique:
        print(f"\n{t}  -  endorsed in {len(endorsed[t])} papers")
        for pid, s in quotes[t][:2]:
            print(f"   [{pid}] “{s[:185]}…”")

    print(f"\nwrote {BASE / 'experiments' / 'reported_models.csv'}")


if __name__ == "__main__":
    main()
