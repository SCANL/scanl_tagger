"""
annotation_review.py
--------------------
Joins model predictions into every row of tagger_data.tsv, producing
input/tagger_data_review.tsv.

Every row from tagger_data.tsv is preserved in original order with two
extra columns appended:

  pred_tags   – model prediction  (identical to GRAMMAR_PATTERN if correct)
  error_count – number of wrong tokens  (0 if correct)

Workflow:
  1. Run this script
  2. Open input/tagger_data_review.tsv
  3. Filter / sort on error_count to surface mistakes
  4. Fix GRAMMAR_PATTERN where the annotation is wrong
  5. Delete pred_tags and error_count columns
  6. Save as input/tagger_data.tsv  (replaces the original)
"""

import os
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SCRIPT_DIR)
ERRORS_PATH  = os.path.join(ROOT, "output", "error_analysis", "identifier_errors.csv")
TSV_PATH     = os.path.join(ROOT, "input", "tagger_data.tsv")
OUTPUT_PATH  = os.path.join(ROOT, "input", "tagger_data_review.tsv")


def norm(s: str) -> str:
    return " ".join(str(s).strip().split())


def main():
    tsv_df    = pd.read_csv(TSV_PATH, sep="\t", dtype=str).fillna("")
    errors_df = pd.read_csv(ERRORS_PATH, dtype=str).fillna("")

    # Only tagger_data predictions are relevant here
    errors_df = errors_df[errors_df["data_source"] == "tagger_data"].copy()
    errors_df["error_count"] = pd.to_numeric(
        errors_df["error_count"], errors="coerce"
    ).fillna(0).astype(int)

    # Normalised join keys on both sides
    tsv_df["_tok"] = tsv_df["SPLIT"].apply(norm)
    tsv_df["_ctx"] = tsv_df["CONTEXT"].str.strip().str.upper()
    tsv_df["_lng"] = tsv_df["LANGUAGE"].str.strip().str.upper()

    errors_df["_tok"] = errors_df["tokens"].apply(norm)
    errors_df["_ctx"] = errors_df["context"].str.strip().str.upper()
    errors_df["_lng"] = errors_df["language"].str.strip().str.upper()

    pred_lookup = errors_df.set_index(["_tok", "_ctx", "_lng"])[
        ["pred_tags", "error_count"]
    ]

    # Join predictions onto every TSV row
    def lookup(row):
        key = (row["_tok"], row["_ctx"], row["_lng"])
        if key in pred_lookup.index:
            hit = pred_lookup.loc[key]
            # handle duplicate index entries (take first)
            if isinstance(hit, pd.DataFrame):
                hit = hit.iloc[0]
            return pd.Series({"pred_tags": hit["pred_tags"],
                              "error_count": int(hit["error_count"])})
        # Not in error list → model was correct
        return pd.Series({"pred_tags": row["GRAMMAR_PATTERN"], "error_count": 0})

    extras = tsv_df.apply(lookup, axis=1)
    out = pd.concat([tsv_df.drop(columns=["_tok", "_ctx", "_lng"]), extras], axis=1)

    out.to_csv(OUTPUT_PATH, sep="\t", index=False)

    n_errors = (out["error_count"] > 0).sum()
    print(f"Wrote {len(out)} rows → {OUTPUT_PATH}")
    print(f"  {n_errors} rows have model errors  |  {len(out) - n_errors} are correct")


if __name__ == "__main__":
    main()
