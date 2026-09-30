"""
Combine input/tagger_data.tsv with the reconstructed code links into input/tagger_data_new.tsv.

Rows without a usable declaration (no link, file or name missing at the commit, or only a
plain use of the name) are dropped and written to output/identifier_links/dropped_rows.tsv.
The remaining rows keep tagger_data.tsv's order; TAGGER_DATA_LINE gives the original line.
The REVIEW column explains why a row may need a manual check and is empty otherwise.

Run after reconstruct_identifier_links.py and add_closed_category_lines.py:
    python scripts/build_tagger_data_new.py
"""
import argparse

import pandas as pd

LINK_COLUMNS = ["ORIGINAL_NAME", "CODE_URL", "COMMIT", "FILE", "SRC_LINE", "LINK_SOURCE", "MATCH",
                "LINE_CHECK", "FILE_AT_COMMIT", "SRCML_CONTEXT", "SRCML_TYPE", "SNAPSHOT_DATE"]


def drop_reason(row):
    if row.LINK_SOURCE == "none":
        return "name not found in snapshot"
    if row.LINE_CHECK == "file_unreadable":
        return "file missing at commit"
    if row.LINE_CHECK == "not_found":
        return "name not found in file at commit"
    if row.MATCH in ("usage-only", "none"):
        return "no declaration found"
    return ""


def review_reason(row):
    if row.MATCH == "forward-declaration":
        return "class linked to a forward declaration; no definition found"
    if row.MATCH == "text-declaration":
        return "srcML could not parse the declaration; found by text search"
    if row.MATCH == "name-only":
        return "context differs; first declaration with this name"
    return ""


def main():
    base = pd.read_csv(ARGS.tagger_data, sep="\t", keep_default_na=False, dtype=str)
    links = pd.read_csv(ARGS.links, sep="\t", keep_default_na=False, dtype=str)
    if len(base) != len(links) or (base.SPLIT.values != links.SPLIT.values).any():
        raise SystemExit("tagger_data.tsv and the links file are out of sync; rerun the link scripts")
    for column in LINK_COLUMNS:
        if column not in links:
            links[column] = ""
    out = pd.concat([base.reset_index(drop=True), links[LINK_COLUMNS].reset_index(drop=True)], axis=1)
    out.insert(0, "TAGGER_DATA_LINE", out.index + 2)
    out["DROP"] = out.apply(drop_reason, axis=1)
    dropped = out[out.DROP != ""]
    dropped.to_csv(ARGS.dropped, sep="\t", index=False)
    out = out[out.DROP == ""].drop(columns=["DROP"])
    out["REVIEW"] = out.apply(review_reason, axis=1)
    out.to_csv(ARGS.out, sep="\t", index=False)
    print(f"Wrote {ARGS.out} ({len(out)} rows); dropped {len(dropped)} rows -> {ARGS.dropped}")
    print(dropped.DROP.value_counts().to_string())
    print(out.REVIEW.replace("", "(ok)")
          .str.replace(r"adjusted\([+-]\d+\)", "adjusted", regex=True).value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tagger-data", default="input/tagger_data.tsv")
    parser.add_argument("--links", default="output/identifier_links/tagger_data_with_links.tsv")
    parser.add_argument("--out", default="input/tagger_data_new.tsv")
    parser.add_argument("--dropped", default="output/identifier_links/dropped_rows.tsv")
    ARGS = parser.parse_args()
    main()
