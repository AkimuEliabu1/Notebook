#!/usr/bin/env python3
"""Do the over-long documents actually damage the classification?

The length check flags ~10 papers that together hold about half the corpus
text, two of which are whole proceedings volumes rather than single studies.
This asks whether that matters, four ways:

  1. Are the flagged papers assigned differently from the rest?
  2. Does the facet distribution shift when they are removed?
  3. Does cross-validated F1 change on the papers that remain?
  4. Do the clusters themselves change?
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
BASE = Path("/home/naedatatz/Desktop/Notebook")
SEED = 42
FACETS = ["research_type", "contribution_type", "evaluation_method"]


def main():
    df = pd.read_csv(BASE / "output" / "latest" / "sms_results.csv")
    w = df["word_count"].astype(float)
    q1, q3 = w.quantile(0.25), w.quantile(0.75)
    hi = q3 + 1.5 * (q3 - q1)
    df["is_long"] = w > hi
    n_long = int(df.is_long.sum())
    print(f"{len(df)} papers, {n_long} flagged as over-long "
          f"(> {int(hi):,} words), holding "
          f"{100 * w[df.is_long].sum() / w.sum():.1f}% of all text\n")

    # 1. are the flagged papers assigned differently?
    print("1. WHERE THE OVER-LONG PAPERS LAND")
    for facet in FACETS:
        if facet not in df:
            continue
        long_mix = df[df.is_long][facet].value_counts(normalize=True)
        rest_mix = df[~df.is_long][facet].value_counts(normalize=True)
        top_long = long_mix.index[0] if len(long_mix) else "-"
        print(f"   {facet:<20} most common among the long: {top_long} "
              f"({100 * long_mix.iloc[0]:.0f}%)  vs among the rest: "
              f"{100 * rest_mix.get(top_long, 0):.0f}%")
    if "cluster_id" in df:
        print(f"   cluster_id           long papers spread over "
              f"{df[df.is_long].cluster_id.nunique()} of "
              f"{df.cluster_id.nunique()} clusters")

    # embeddings for both corpora
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()
    E = normalize(np.asarray(enc.encode(texts, show_progress_bar=False)))
    keep = (~df.is_long).to_numpy()

    # 2 & 3. distribution shift and F1 on the surviving papers
    print("\n2/3. EFFECT ON CLASSIFICATION (evaluated on the same surviving papers)")
    rows = []
    for facet in FACETS:
        if facet not in df:
            continue
        y = df[facet].to_numpy()
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        f_all = f1_score(y, cross_val_predict(LogisticRegression(max_iter=2000),
                                              E, y, cv=cv),
                         average="macro", zero_division=0)
        yk = y[keep]
        f_keep = f1_score(yk, cross_val_predict(LogisticRegression(max_iter=2000),
                                                E[keep], yk, cv=cv),
                          average="macro", zero_division=0)
        share_all = df[facet].value_counts(normalize=True)
        share_keep = df[~df.is_long][facet].value_counts(normalize=True)
        drift = (share_all - share_keep).abs().sum() / 2
        rows.append({"facet": facet, "macro_f1_all": round(f_all, 4),
                     "macro_f1_without_long": round(f_keep, 4),
                     "change": round(f_keep - f_all, 4),
                     "distribution_shift": round(float(drift), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    # 4. do the clusters move?
    print("\n4. EFFECT ON CLUSTERING")
    k = int(df.cluster_id.nunique()) if "cluster_id" in df else 4
    lab_all = KMeans(k, random_state=SEED, n_init=10).fit_predict(E)
    lab_keep = KMeans(k, random_state=SEED, n_init=10).fit_predict(E[keep])
    ari = adjusted_rand_score(lab_all[keep], lab_keep)
    print(f"   agreement on the surviving papers, with vs without the long ones:")
    print(f"   ARI = {ari:.4f}   (1.0 = the long papers changed nothing)")

    pd.DataFrame(rows).to_csv(BASE / "experiments" / "outlier_impact.csv", index=False)


if __name__ == "__main__":
    main()
