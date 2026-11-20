"""List identifiers where every seed of a training run disagrees with gold, for review.

Usage:
  python scripts/disagreements.py RUN [--seeds 209 7 42] [--previous OLD.tsv] [--out OUT.tsv]

Reads output/reports/RUN_seed<N>_predictions.csv and _cv_predictions.csv (holdout and
out-of-fold), so every identifier, real and synthetic, is scored once per seed.

Rows are grouped by error signature (which tag the model swaps for which, and where:
`PRE>NM@first` = gold PRE on the first word, model NM), largest group first, then by gold
pattern. SUGGESTED is pre-filled from word evidence: how the same word in the same position
is tagged in other gold rows. If those rows mostly tag it like the model, the model's
answer is suggested; mixed evidence leaves SUGGESTED blank. It ignores the word's role, so
check it. `--previous` copies SUGGESTED/WHY/REVIEW/NOTES from an earlier sheet for rows
that are still wrong with the same gold.
"""
import argparse
import collections
import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "output" / "reports"
KEY = ["tokens", "true_tags", "context", "type", "system_name", "subset"]


def position(i, n):
    return "only" if n == 1 else "first" if i == 0 else "last" if i == n - 1 else "mid"


def family(signature):
    pairs = [set(p.split("@")[0].split(">")) for p in signature.split(" + ")]
    if any("PRE" in p for p in pairs):
        return "PRE boundary"
    if any(p & {"V", "VM"} and p & {"N", "NM", "NPL"} for p in pairs):
        return "verb vs noun"
    if any(p & {"P", "D", "DT", "CJ", "VM"} for p in pairs):
        return "closed-category word"
    if any(p in ({"N", "NPL"}, {"NM", "NPL"}) for p in pairs):
        return "plural"
    if all(p == {"N", "NM"} for p in pairs):
        return "head position (N vs NM)"
    return "other"


def load(run, seeds):
    merged = None
    for seed in seeds:
        d = pd.concat([pd.read_csv(REPORTS / f"{run}_seed{seed}_{kind}.csv", keep_default_na=False)
                       for kind in ("predictions", "cv_predictions")])
        d["dup"] = d.groupby(KEY).cumcount()
        d = d.set_index(KEY + ["dup"]).rename(columns={"pred_tags": f"p{seed}"})
        merged = d[[f"p{seed}", "language", "eval_split"]] if merged is None else merged.join(d[[f"p{seed}"]])
    return merged.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[209, 7, 42])
    parser.add_argument("--previous", help="earlier sheet to copy SUGGESTED/WHY/REVIEW/NOTES from")
    parser.add_argument("--out", default=str(REPORTS / f"agree_wrong_{datetime.date.today()}.tsv"))
    args = parser.parse_args()

    m = load(args.run, args.seeds)
    preds = [f"p{s}" for s in args.seeds]
    m["n_wrong"] = sum((m[p] != m.true_tags).astype(int) for p in preds)

    stats = collections.defaultdict(collections.Counter)  # (word, position) -> gold tag counts
    for tokens, tags in zip(m.tokens, m.true_tags):
        words, labels = tokens.split(), tags.split()
        for i, (w, t) in enumerate(zip(words, labels)):
            stats[(w.lower(), position(i, len(words)))][t] += 1

    data = pd.read_csv(REPO / "input" / "tagger_data.tsv", sep="\t", dtype=str, keep_default_na=False)
    lines = collections.defaultdict(list)
    for r in data.itertuples():
        lines[(r.SPLIT, r.CONTEXT, r.SYSTEM_NAME, r.DATA_SOURCE)].append(r.TAGGER_DATA_LINE)

    rows = []
    for r in m[m.n_wrong == len(preds)].itertuples():
        votes = collections.Counter(getattr(r, p) for p in preds)
        majority, agree = votes.most_common(1)[0]
        words, gold, pred = r.tokens.split(), r.true_tags.split(), majority.split()
        n = len(words)
        diffs = [(i, gold[i], pred[i]) for i in range(n) if gold[i] != pred[i]]
        evidence, for_model, for_gold = [], 0, 0
        for i, g, p in diffs:
            counts = stats[(words[i].lower(), position(i, n))].copy()
            counts[g] -= 1  # leave this row out
            counts = +counts
            total = sum(counts.values())
            if total:
                evidence.append(f"'{words[i]}'@{position(i, n)}: " + ", ".join(f"{k} {v}" for k, v in counts.most_common(4)))
            if total >= 2 and counts[p] > counts[g]:
                for_model += 1
            elif total >= 2:
                for_gold += 1
        if for_model and not for_gold:
            suggested, why = majority, "other gold rows tag this word like the model"
        elif for_model:
            suggested, why = "", "sibling evidence is mixed"
        else:
            suggested, why = r.true_tags, "other gold rows agree with gold" if evidence else "no sibling evidence against gold"
        rows.append(dict(
            SIGNATURE=" + ".join(f"{g}>{p}@{position(i, n)}" for i, g, p in diffs),
            TOKENS=r.tokens, GOLD=r.true_tags, PREDICTED=majority, SEEDS_AGREE=agree,
            SUGGESTED=suggested, WHY=why, WORD_EVIDENCE=" | ".join(evidence),
            CONTEXT=r.context, TYPE=r.type, LANGUAGE=r.language, SYSTEM=r.system_name,
            SUBSET=r.subset, SPLIT=r.eval_split,
            TAGGER_DATA_LINE=";".join(lines.get((r.tokens, r.context, r.system_name, r.subset), [])),
            REVIEW="", NOTES=""))
    out = pd.DataFrame(rows)

    if args.previous:
        prev = pd.read_csv(args.previous, sep="\t", dtype=str, keep_default_na=False)
        prev = prev.drop_duplicates(["TOKENS", "GOLD", "SYSTEM"]).set_index(["TOKENS", "GOLD", "SYSTEM"])
        carried = 0
        for i, r in out.iterrows():
            key = (r.TOKENS, r.GOLD, r.SYSTEM)
            if key in prev.index:
                old = prev.loc[key]
                for col in ("SUGGESTED", "WHY", "REVIEW", "NOTES"):
                    out.at[i, col] = old[col]
                carried += 1
        print(f"carried {carried} rows from {args.previous}")

    out["FAMILY"] = out.SIGNATURE.map(family)
    out["SIG_COUNT"] = out.groupby("SIGNATURE").SIGNATURE.transform("size")
    out["GOLD_PAT_COUNT"] = out.groupby(["SIGNATURE", "GOLD"]).GOLD.transform("size")
    out = out.sort_values(["SIG_COUNT", "SIGNATURE", "GOLD_PAT_COUNT", "GOLD", "TOKENS"],
                          ascending=[False, True, False, True, True])
    first = ["FAMILY", "SIG_COUNT", "SIGNATURE", "GOLD_PAT_COUNT"]
    out[first + [c for c in out.columns if c not in first]].to_csv(args.out, sep="\t", index=False)
    print(f"{len(out)} identifiers wrong in all {len(preds)} seeds -> {args.out}")
    print(out.groupby("FAMILY").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
