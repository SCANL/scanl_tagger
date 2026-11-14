"""
Evaluation metrics for identifier tagging, and a paired comparison of training runs.

The headline metric is identifier-level accuracy on real code: an identifier counts only if
every one of its tags is right. Token macro-F1 on real code is the guardrail: it weighs each
tag equally, so it drops when rare tags (CJ, VM, PRE, DT) are sacrificed, which barely moves
identifier accuracy.

Compare runs saved with `--run-name` (output/reports/<run>_seed<N>_*predictions.csv):

    python -m src.lm_based_tagger.evaluation RUN_A RUN_B
    python -m src.lm_based_tagger.evaluation RUN_A          # one run's metrics only

Every seed of each run is loaded, real identifiers are rescored against the current gold in
input/tagger_data.tsv (so label fixes apply to old runs too), and only identifiers scored in
every seed of both runs are compared. Differences come with a paired bootstrap 95% interval.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REAL_SOURCE = "tagger_data"
REAL_SUBSETS = ("original_set", "closed_category_set", "headnoun_set")
DEFAULT_REPORTS_DIR = os.path.join("output", "reports")
DEFAULT_GOLD = os.path.join("input", "tagger_data.tsv")


def normalize_split(split: str) -> str:
    """Comparison key for an identifier split or tag sequence: lowercased, single-spaced."""
    return " ".join(str(split).lower().split())


def _label_counts(true: Sequence[List[str]], pred: Sequence[List[str]], labels: List[str]) -> np.ndarray:
    """Per-identifier [tp, fp, fn] counts for each label, shape [n, len(labels), 3]."""
    index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(true), len(labels), 3), dtype=np.int32)
    for row, (t_tags, p_tags) in enumerate(zip(true, pred)):
        for t, p in zip(t_tags, p_tags):
            if t == p:
                counts[row, index[t], 0] += 1
            else:
                counts[row, index[t], 2] += 1
                if p in index:
                    counts[row, index[p], 1] += 1
    return counts


def _macro_f1_from_counts(summed: np.ndarray) -> np.ndarray:
    """Macro-F1 over the labels present in the gold tags; `summed` is [..., labels, 3]."""
    tp, fp, fn = summed[..., 0], summed[..., 1], summed[..., 2]
    support = tp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(support > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return np.nanmean(f1, axis=-1)


def token_macro_f1(true: Sequence[List[str]], pred: Sequence[List[str]]) -> float:
    """Token macro-F1 over the tags that occur in `true`."""
    labels = sorted({tag for tags in true for tag in tags} | {tag for tags in pred for tag in tags})
    if not labels:
        return float("nan")
    return float(_macro_f1_from_counts(_label_counts(true, pred, labels).sum(axis=0)))


def accuracy_by_source(preds_df: pd.DataFrame, column: str = "data_source") -> pd.DataFrame:
    """
    Identifier accuracy, token accuracy and token macro-F1 for each value of `column`.

    "data_source" gives real (tagger_data, every subset together) vs. synthetic; "subset"
    breaks real down into original_set, closed_category_set and headnoun_set. Synthetic rows
    are much easier than real identifiers, so the combined score hides how the model does on
    real code. Expects the columns written to holdout_predictions.csv.
    """
    rows = []
    for source, group in preds_df.groupby(column, sort=True):
        true = group["true_tags"].str.split().tolist()
        pred = group["pred_tags"].str.split().tolist()
        tokens = sum(len(t) for t in true)
        correct = sum(a == b for t, p in zip(true, pred) for a, b in zip(t, p))
        rows.append({
            "source": source,
            "identifiers": len(group),
            "token_accuracy": correct / tokens if tokens else float("nan"),
            "identifier_accuracy": (group["true_tags"] == group["pred_tags"]).mean(),
            "macro_f1": token_macro_f1(true, pred),
        })
    return pd.DataFrame(rows)


def real_code_summary(preds_df: pd.DataFrame) -> pd.DataFrame:
    """accuracy_by_source rows for real code combined ("real") and for each real subset."""
    real = preds_df[preds_df["data_source"] == REAL_SOURCE]
    if real.empty:
        return pd.DataFrame(columns=["source", "identifiers", "token_accuracy", "identifier_accuracy", "macro_f1"])
    combined = accuracy_by_source(real.assign(_all="real"), "_all")
    return pd.concat([combined, accuracy_by_source(real, "subset")], ignore_index=True)


# === Comparing runs ===

_PRED_FILE = re.compile(r"_seed(\d+)_(cv_)?predictions\.csv$")


def load_run(run: str, reports_dir: str = DEFAULT_REPORTS_DIR) -> Dict[int, pd.DataFrame]:
    """Every seed's predictions for `run` (holdout plus CV out-of-fold when saved), by seed."""
    seeds: Dict[int, List[pd.DataFrame]] = {}
    prefix = os.path.join(reports_dir, f"{run}_seed")
    for path in sorted(glob.glob(glob.escape(prefix) + "*predictions.csv")):
        match = _PRED_FILE.search(path)
        if not match or not path.startswith(prefix) or path[len(prefix):].split("_")[0] != match.group(1):
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "eval_split" not in frame.columns:
            frame["eval_split"] = "cv" if match.group(2) else "holdout"
        seeds.setdefault(int(match.group(1)), []).append(frame)
    if not seeds:
        raise FileNotFoundError(f"No predictions for run '{run}' in {reports_dir}")
    return {seed: pd.concat(frames, ignore_index=True) for seed, frames in sorted(seeds.items())}


def _keys(frame: pd.DataFrame) -> pd.Series:
    return (frame["data_source"] + "|" + frame["tokens"].map(normalize_split) + "|"
            + frame["context"].str.strip().str.upper())


def _gold_map(gold_path: str) -> Dict[str, str]:
    """Current gold tags for real identifiers, by key; keys that occur twice are left out."""
    gold = pd.read_csv(gold_path, sep="\t", dtype=str, keep_default_na=False)
    keys = (REAL_SOURCE + "|" + gold["SPLIT"].map(normalize_split) + "|"
            + gold["CONTEXT"].str.strip().str.upper())
    unique = ~keys.duplicated(keep=False)
    return dict(zip(keys[unique], gold.loc[unique, "GRAMMAR_PATTERN"].map(lambda s: " ".join(s.split()))))


def _prepare(frame: pd.DataFrame, gold: Dict[str, str] | None) -> pd.DataFrame:
    frame = frame.copy()
    frame["key"] = _keys(frame)
    frame = frame[~frame["key"].duplicated(keep=False)]
    if gold is not None:
        real = frame["data_source"] == REAL_SOURCE
        frame = frame[~real | frame["key"].isin(gold)].copy()
        real = frame["data_source"] == REAL_SOURCE
        frame.loc[real, "true_tags"] = frame.loc[real, "key"].map(gold)
        frame = frame[frame["true_tags"].str.split().str.len() == frame["pred_tags"].str.split().str.len()]
    return frame.set_index("key")


class _RunScores:
    """Per-identifier correctness and per-label counts for each seed of a run."""

    def __init__(self, seeds: Dict[int, pd.DataFrame], keys: List[str], labels: List[str]):
        self.seeds = sorted(seeds)
        frames = [seeds[s].loc[keys] for s in self.seeds]
        true = frames[0]["true_tags"].str.split().tolist()
        self.correct = np.stack([(f["true_tags"] == f["pred_tags"]).to_numpy(float) for f in frames])  # [S, n]
        self.counts = np.stack([_label_counts(true, f["pred_tags"].str.split().tolist(), labels) for f in frames])

    def identifier_accuracy(self, weights: np.ndarray) -> np.ndarray:
        """Seed-averaged identifier accuracy for each row of bootstrap `weights` [B, n]."""
        return (weights @ self.correct.mean(axis=0)) / weights.sum(axis=1)

    def macro_f1(self, weights: np.ndarray) -> np.ndarray:
        """Seed-averaged token macro-F1 for each row of bootstrap `weights` [B, n]."""
        per_seed = [_macro_f1_from_counts(np.einsum("bn,nlk->blk", weights, c)) for c in self.counts]
        return np.mean(per_seed, axis=0)

    def per_seed_accuracy(self) -> np.ndarray:
        return self.correct.mean(axis=1)


def _bootstrap_weights(n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.integers(0, n, size=(reps, n))
    return np.stack([np.bincount(row, minlength=n) for row in draws]).astype(np.float64)


def _interval(diffs: np.ndarray) -> str:
    low, high = np.percentile(diffs, [2.5, 97.5])
    return f"[{low * 100:+.2f}, {high * 100:+.2f}]"


def compare_runs(
    runs: List[str],
    reports_dir: str = DEFAULT_REPORTS_DIR,
    gold_path: str | None = DEFAULT_GOLD,
    eval_split: str = "all",
    reps: int = 2000,
    seed: int = 0,
) -> str:
    """Text report of each run's metrics and, for two runs, the paired differences (B - A)."""
    gold = _gold_map(gold_path) if gold_path else None
    loaded = {run: {s: _prepare(f, gold) for s, f in load_run(run, reports_dir).items()} for run in runs}
    if eval_split != "all":
        loaded = {run: {s: f[f["eval_split"] == eval_split] for s, f in seeds.items()} for run, seeds in loaded.items()}

    shared = set.intersection(*[set(f.index) for seeds in loaded.values() for f in seeds.values()])
    first = next(iter(next(iter(loaded.values())).values())).loc[sorted(shared)]
    labels = sorted({tag for tags in first["true_tags"] for tag in tags.split()})
    rng = np.random.default_rng(seed)

    lines = [
        f"Runs: {', '.join(f'{r} (seeds {sorted(loaded[r])})' for r in runs)}",
        f"Identifiers scored in every seed of every run: {len(shared)}"
        + (f" (eval split: {eval_split})" if eval_split != "all" else "")
        + ("; real identifiers rescored against " + gold_path if gold_path else ""),
        "Accuracy is identifier-level (every tag right); F1 is token macro-F1. Values are seed means, in %.",
    ]
    groups = [("real", first["data_source"] == REAL_SOURCE)]
    groups += [(s, first["subset"] == s) for s in REAL_SUBSETS]
    groups += [(s, (first["data_source"] == s)) for s in sorted(set(first["data_source"]) - {REAL_SOURCE})]

    header = f"\n{'group':<22}{'n':>6}"
    for run in runs:
        header += f"{run[:14] + ' acc':>20}{'F1':>8}"
    if len(runs) == 2:
        header += f"{'acc diff [95% CI]':>30}{'F1 diff [95% CI]':>30}"
    lines.append(header)

    real_scores = {}
    for name, mask in groups:
        keys = list(first.index[mask.to_numpy()])
        if not keys:
            continue
        weights = _bootstrap_weights(len(keys), reps, rng)
        full = np.ones((1, len(keys)))
        scores = {run: _RunScores(loaded[run], keys, labels) for run in runs}
        if name == "real":
            real_scores = scores
        line = f"{name:<22}{len(keys):>6}"
        for run in runs:
            acc = scores[run].identifier_accuracy(full)[0] * 100
            spread = np.std(scores[run].per_seed_accuracy(), ddof=1) * 100 if len(scores[run].seeds) > 1 else 0.0
            line += f"{acc:>13.2f} ±{spread:<4.2f}{scores[run].macro_f1(full)[0] * 100:>8.2f}"
        if len(runs) == 2:
            a, b = (scores[r] for r in runs)
            acc_diff = b.identifier_accuracy(weights) - a.identifier_accuracy(weights)
            f1_diff = b.macro_f1(weights) - a.macro_f1(weights)
            point_acc = (b.identifier_accuracy(full) - a.identifier_accuracy(full))[0] * 100
            point_f1 = (b.macro_f1(full) - a.macro_f1(full))[0] * 100
            line += f"{point_acc:>+10.2f} {_interval(acc_diff):<19}{point_f1:>+10.2f} {_interval(f1_diff):<19}"
        lines.append(line)
    lines.append("(± after accuracy: standard deviation across seeds)")

    if real_scores:
        lines.append("\nPer-tag F1 on real identifiers (seed means, %):")
        support = next(iter(real_scores.values())).counts[0].sum(axis=0)
        lines.append(f"{'tag':<6}{'support':>9}" + "".join(f"{run[:14]:>16}" for run in runs))
        for i, label in enumerate(labels):
            row = f"{label:<6}{support[i, 0] + support[i, 2]:>9}"
            for run in runs:
                summed = real_scores[run].counts.sum(axis=1)[:, i, :]  # [S, 3]
                tp, fp, fn = summed[:, 0], summed[:, 1], summed[:, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    f1 = np.nanmean(np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan))
                row += f"{f1 * 100:>16.2f}"
            lines.append(row)
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare SCALAR training runs saved with --run-name.")
    parser.add_argument("runs", nargs="+", help="One run name to summarize, or two to compare (B - A).")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--gold", default=DEFAULT_GOLD, help="Gold TSV to rescore real identifiers against.")
    parser.add_argument("--no-rescore", action="store_true", help="Score against each run's own gold tags.")
    parser.add_argument("--split", choices=["all", "cv", "holdout"], default="all",
                        help="Which predictions to use: CV out-of-fold, holdout, or both (default).")
    parser.add_argument("--reps", type=int, default=2000, help="Bootstrap resamples.")
    args = parser.parse_args(argv)
    if len(args.runs) > 2:
        parser.error("give one or two run names")
    print(compare_runs(
        args.runs,
        reports_dir=args.reports_dir,
        gold_path=None if args.no_rescore else args.gold,
        eval_split=args.split,
        reps=args.reps,
    ))


if __name__ == "__main__":
    main()
