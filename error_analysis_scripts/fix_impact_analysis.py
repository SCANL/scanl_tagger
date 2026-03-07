"""
fix_impact_analysis.py
----------------------
Answers the question:
  "Which single-tag corrections, if we resolved them perfectly,
   would give us the biggest gain in identifier-level accuracy?"

Focuses heavily on *one-off* identifiers -- those where exactly one
token is wrong.  Fixing any one-off identifier's single mistake makes
the whole identifier correct.

Usage (standalone):
    python error_analysis_scripts/fix_impact_analysis.py \
        --predictions output/holdout_predictions.csv \
        --output_dir  output/fix_impact

Or import and call analyze_fix_impact() directly from lm_error_analysis.
"""

import argparse
import os
from collections import defaultdict
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split(value: str) -> List[str]:
    if pd.isna(value):
        return []
    return str(value).strip().split()


def _build_error_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand every token-level error into its own row with full context."""
    rows = []
    for _, id_row in df.iterrows():
        tokens = _split(id_row["tokens"])
        true_tags = _split(id_row["true_tags"])
        pred_tags = _split(id_row["pred_tags"])
        error_count = sum(t != p for t, p in zip(true_tags, pred_tags))
        for pos, (tok, true_tag, pred_tag) in enumerate(zip(tokens, true_tags, pred_tags)):
            if true_tag != pred_tag:
                rows.append({
                    "identifier":   id_row["tokens"],
                    "true_tags":    id_row["true_tags"],
                    "pred_tags":    id_row["pred_tags"],
                    "token":        tok,
                    "position":     pos,
                    "identifier_len": len(tokens),
                    "true_tag":     true_tag,
                    "pred_tag":     pred_tag,
                    "substitution": f"{pred_tag} → {true_tag}",
                    "identifier_error_count": error_count,
                    "is_one_off":   error_count == 1,
                    "context":      id_row.get("context", ""),
                    "language":     id_row.get("language", ""),
                    "data_source":  id_row.get("data_source", ""),
                    "system_name":  id_row.get("system_name", ""),
                })
    return pd.DataFrame(rows)


def _aggregate_by_substitution(error_df: pd.DataFrame, total_identifiers: int) -> pd.DataFrame:
    """
    Per (pred_tag → true_tag) substitution:
      - one_off_gain:   identifiers we'd fix by resolving this substitution in one-off cases
      - total_errors:   total tokens wrong with this substitution (across all identifiers)
      - one_off_pct:    what % of all identifiers this one fix unlocks
    Sorted by one_off_gain desc, then total_errors desc.
    """
    one_off = error_df[error_df["is_one_off"]]
    agg = (
        error_df
        .groupby(["pred_tag", "true_tag", "substitution"])
        .agg(
            total_errors=("identifier", "count"),
            one_off_gain=("is_one_off", "sum"),
        )
        .reset_index()
    )
    agg["one_off_gain"] = agg["one_off_gain"].astype(int)
    agg["one_off_pct_of_all"] = (agg["one_off_gain"] / total_identifiers * 100).round(2)
    agg = agg.sort_values(["one_off_gain", "total_errors"], ascending=False).reset_index(drop=True)
    return agg


def _aggregate_by_position(error_df: pd.DataFrame, total_identifiers: int) -> pd.DataFrame:
    """Breakdown by (pred_tag → true_tag, position) -- useful to see positional patterns."""
    agg = (
        error_df
        .groupby(["pred_tag", "true_tag", "substitution", "position"])
        .agg(
            total_errors=("identifier", "count"),
            one_off_gain=("is_one_off", "sum"),
        )
        .reset_index()
    )
    agg["one_off_gain"] = agg["one_off_gain"].astype(int)
    agg["one_off_pct_of_all"] = (agg["one_off_gain"] / total_identifiers * 100).round(2)
    agg = agg.sort_values(["one_off_gain", "total_errors"], ascending=False).reset_index(drop=True)
    return agg


def _aggregate_by_token(error_df: pd.DataFrame, total_identifiers: int) -> pd.DataFrame:
    """Breakdown by (specific token, pred_tag → true_tag) -- shows concrete vocabulary errors."""
    agg = (
        error_df
        .groupby(["token", "pred_tag", "true_tag", "substitution"])
        .agg(
            total_errors=("identifier", "count"),
            one_off_gain=("is_one_off", "sum"),
        )
        .reset_index()
    )
    agg["one_off_gain"] = agg["one_off_gain"].astype(int)
    agg["one_off_pct_of_all"] = (agg["one_off_gain"] / total_identifiers * 100).round(2)
    agg = agg.sort_values(["one_off_gain", "total_errors"], ascending=False).reset_index(drop=True)
    return agg


def _write_summary(
    f,
    total_identifiers: int,
    correct_identifiers: int,
    one_off_df: pd.DataFrame,
    by_sub: pd.DataFrame,
    by_pos: pd.DataFrame,
    by_tok: pd.DataFrame,
    error_df: pd.DataFrame,
):
    id_acc = correct_identifiers / total_identifiers if total_identifiers else 0.0
    n_one_off = int(one_off_df["identifier"].nunique()) if len(one_off_df) > 0 else 0
    max_possible_gain = (correct_identifiers + n_one_off) / total_identifiers if total_identifiers else 0.0

    f.write("Fix-Impact Analysis\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total identifiers:          {total_identifiers}\n")
    f.write(f"Currently correct:          {correct_identifiers}  ({id_acc:.1%})\n")
    f.write(f"One-off errors:             {n_one_off}  "
            f"({n_one_off / total_identifiers:.1%} of all identifiers have exactly 1 wrong tag)\n")
    f.write(f"Max achievable id-accuracy (by fixing all one-offs): {max_possible_gain:.1%}\n\n")

    f.write("─" * 80 + "\n")
    f.write("Top 20 substitutions by one-off identifier gain\n")
    f.write("(how many whole identifiers become correct if this single-tag mistake is fixed)\n")
    f.write("─" * 80 + "\n")
    header = f"  {'Substitution':<15}  {'One-off gain':>12}  {'% of total':>10}  {'Total errors':>13}\n"
    f.write(header)
    f.write("  " + "-" * (len(header) - 3) + "\n")
    for _, row in by_sub.head(20).iterrows():
        f.write(
            f"  {row['substitution']:<15}  {int(row['one_off_gain']):>12}  "
            f"{row['one_off_pct_of_all']:>9.2f}%  {int(row['total_errors']):>13}\n"
        )

    f.write("\n")
    f.write("─" * 80 + "\n")
    f.write("Top 20 substitutions by position (pred → true, position number)\n")
    f.write("─" * 80 + "\n")
    header2 = f"  {'Substitution':<15}  {'Pos':>4}  {'One-off gain':>12}  {'Total errors':>13}\n"
    f.write(header2)
    f.write("  " + "-" * (len(header2) - 3) + "\n")
    for _, row in by_pos.head(20).iterrows():
        f.write(
            f"  {row['substitution']:<15}  {int(row['position']):>4}  "
            f"{int(row['one_off_gain']):>12}  {int(row['total_errors']):>13}\n"
        )

    f.write("\n")
    f.write("─" * 80 + "\n")
    f.write("Top 20 specific tokens with one-off impact\n")
    f.write("─" * 80 + "\n")
    header3 = f"  {'Token':<20}  {'Substitution':<15}  {'One-off gain':>12}  {'Total errors':>13}\n"
    f.write(header3)
    f.write("  " + "-" * (len(header3) - 3) + "\n")
    for _, row in by_tok.head(20).iterrows():
        f.write(
            f"  {row['token']:<20}  {row['substitution']:<15}  "
            f"{int(row['one_off_gain']):>12}  {int(row['total_errors']):>13}\n"
        )

    f.write("\n")
    f.write("─" * 80 + "\n")
    f.write("One-off error identifiers (full list)\n")
    f.write("─" * 80 + "\n")
    one_off_ids = (
        one_off_df[["identifier", "true_tags", "pred_tags", "token", "position",
                    "true_tag", "pred_tag", "substitution", "context", "language"]]
        .sort_values(["substitution", "identifier"])
    )
    for _, row in one_off_ids.iterrows():
        f.write(
            f"  [{row['substitution']}]  pos={int(row['position'])}  "
            f"token='{row['token']}'  "
            f"id='{row['identifier']}'  "
            f"true='{row['true_tags']}'  pred='{row['pred_tags']}'  "
            f"ctx={row['context']}  lang={row['language']}\n"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_fix_impact(
    predictions_path: str,
    output_dir: str,
) -> dict:
    """
    Analyze holdout predictions to rank single-tag fixes by identifier-level gain.

    Produces:
      fix_impact_by_substitution.csv  -- ranked by one_off_gain desc
      fix_impact_by_position.csv      -- same, broken down by token position
      fix_impact_by_token.csv         -- same, broken down by specific token word
      fix_impact_summary.txt          -- human-readable report

    Returns dict with all output paths and key statistics.
    """
    df = pd.read_csv(predictions_path)
    os.makedirs(output_dir, exist_ok=True)

    df["row_correct"] = df["true_tags"] == df["pred_tags"]
    total_identifiers = len(df)
    correct_identifiers = int(df["row_correct"].sum())

    error_df = _build_error_rows(df)
    if error_df.empty:
        print("No errors found in predictions — nothing to analyze.")
        return {}

    one_off_df = error_df[error_df["is_one_off"]].copy()

    by_sub = _aggregate_by_substitution(error_df, total_identifiers)
    by_pos = _aggregate_by_position(error_df, total_identifiers)
    by_tok = _aggregate_by_token(error_df, total_identifiers)

    sub_path = os.path.join(output_dir, "fix_impact_by_substitution.csv")
    pos_path = os.path.join(output_dir, "fix_impact_by_position.csv")
    tok_path = os.path.join(output_dir, "fix_impact_by_token.csv")
    summary_path = os.path.join(output_dir, "fix_impact_summary.txt")

    by_sub.to_csv(sub_path, index=False)
    by_pos.to_csv(pos_path, index=False)
    by_tok.to_csv(tok_path, index=False)

    with open(summary_path, "w") as f:
        _write_summary(f, total_identifiers, correct_identifiers,
                       one_off_df, by_sub, by_pos, by_tok, error_df)

    print(f"Wrote fix-impact summary to:          {summary_path}")
    print(f"Wrote fix-impact by substitution to:  {sub_path}")
    print(f"Wrote fix-impact by position to:      {pos_path}")
    print(f"Wrote fix-impact by token to:         {tok_path}")

    return {
        "summary_path": summary_path,
        "sub_path": sub_path,
        "pos_path": pos_path,
        "tok_path": tok_path,
        "total_identifiers": total_identifiers,
        "correct_identifiers": correct_identifiers,
        "one_off_count": int(one_off_df["identifier"].nunique()) if len(one_off_df) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank single-tag fixes by identifier-level accuracy gain."
    )
    parser.add_argument(
        "--predictions",
        default="output/holdout_predictions.csv",
        help="Path to holdout_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="output/fix_impact",
        help="Directory to write output files",
    )
    args = parser.parse_args()

    result = analyze_fix_impact(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
    )

    if result:
        n_one_off = result["one_off_count"]
        total = result["total_identifiers"]
        correct = result["correct_identifiers"]
        print(
            f"\nCurrent identifier accuracy: {correct / total:.1%}"
            f"  ({correct}/{total})"
        )
        print(
            f"One-off errors: {n_one_off}"
            f"  — fixing all one-offs would raise accuracy to"
            f" {(correct + n_one_off) / total:.1%}"
        )
