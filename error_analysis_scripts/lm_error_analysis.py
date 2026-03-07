import os
from collections import Counter
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from error_analysis_scripts.fix_impact_analysis import analyze_fix_impact


def _split_tags(value: str) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    return text.split() if text else []


def _tokenize_identifier(value: str) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    return text.split() if text else []


def _flatten_predictions(df: pd.DataFrame) -> Tuple[List[str], List[str], List[dict]]:
    y_true: List[str] = []
    y_pred: List[str] = []
    token_rows: List[dict] = []

    for _, row in df.iterrows():
        tokens = _tokenize_identifier(row["tokens"])
        true_tags = _split_tags(row["true_tags"])
        pred_tags = _split_tags(row["pred_tags"])

        for idx, (token, true_tag, pred_tag) in enumerate(zip(tokens, true_tags, pred_tags)):
            token_rows.append({
                "token": token,
                "position": idx,
                "true_tag": true_tag,
                "pred_tag": pred_tag,
                "correct": true_tag == pred_tag,
                "context": row.get("context", ""),
                "language": row.get("language", ""),
                "data_source": row.get("data_source", ""),
                "identifier": row["tokens"],
            })
            y_true.append(true_tag)
            y_pred.append(pred_tag)

    return y_true, y_pred, token_rows


def run_lm_diagnostics(
    predictions_path: str,
    output_dir: str,
    report_path: str | None = None,
) -> dict:
    """Generate diagnostic summaries for LM holdout predictions."""
    df = pd.read_csv(predictions_path)
    os.makedirs(output_dir, exist_ok=True)

    df["true_tag_list"] = df["true_tags"].apply(_split_tags)
    df["pred_tag_list"] = df["pred_tags"].apply(_split_tags)
    df["token_list"] = df["tokens"].apply(_tokenize_identifier)
    df["token_count"] = df["token_list"].apply(len)
    df["row_correct"] = df["true_tags"] == df["pred_tags"]
    df["error_count"] = df.apply(
        lambda row: sum(
            1 for true_tag, pred_tag in zip(row["true_tag_list"], row["pred_tag_list"]) if true_tag != pred_tag
        ),
        axis=1,
    )

    incorrect_df = df[~df["row_correct"]].copy()
    y_true, y_pred, token_rows = _flatten_predictions(df)
    token_df = pd.DataFrame(token_rows)
    error_token_df = token_df[~token_df["correct"]].copy()

    confusion_pairs = Counter(zip(error_token_df["true_tag"], error_token_df["pred_tag"]))
    token_confusions = Counter(
        zip(error_token_df["token"], error_token_df["true_tag"], error_token_df["pred_tag"])
    )
    position_errors = Counter(error_token_df["position"])
    context_errors = Counter(incorrect_df.get("context", pd.Series(dtype=str)).fillna(""))
    language_errors = Counter(incorrect_df.get("language", pd.Series(dtype=str)).fillna(""))
    source_errors = Counter(incorrect_df.get("data_source", pd.Series(dtype=str)).fillna(""))

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion_path = os.path.join(output_dir, "token_confusion_matrix.csv")
    confusion_df.to_csv(confusion_path)

    detail_path = os.path.join(output_dir, "identifier_errors.csv")
    incorrect_df.drop(columns=["true_tag_list", "pred_tag_list", "token_list"], errors="ignore").to_csv(detail_path, index=False)

    token_detail_path = os.path.join(output_dir, "token_errors.csv")
    error_token_df.to_csv(token_detail_path, index=False)

    summary_path = os.path.join(output_dir, "error_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("LM Holdout Error Analysis\n")
        f.write("=" * 80 + "\n")
        f.write(f"Predictions file: {predictions_path}\n")
        if report_path:
            f.write(f"Holdout report: {report_path}\n")
        f.write(f"Total identifiers: {len(df)}\n")
        f.write(f"Incorrect identifiers: {len(incorrect_df)}\n")
        f.write(f"Identifier accuracy: {df['row_correct'].mean():.4f}\n")
        f.write(f"Total token errors: {len(error_token_df)}\n\n")

        f.write("Classification report:\n")
        f.write(classification_report(y_true, y_pred, digits=4))
        f.write("\nTop tag confusions:\n")
        for (true_tag, pred_tag), count in confusion_pairs.most_common(15):
            f.write(f"  {true_tag} -> {pred_tag}: {count}\n")

        f.write("\nTop token-level confusions:\n")
        for (token, true_tag, pred_tag), count in token_confusions.most_common(20):
            f.write(f"  {token}: {true_tag} -> {pred_tag} ({count})\n")

        f.write("\nErrors by token position:\n")
        for position, count in sorted(position_errors.items()):
            f.write(f"  position {position}: {count}\n")

        if context_errors:
            f.write("\nErrors by context:\n")
            for context, count in context_errors.most_common():
                f.write(f"  {context or '<missing>'}: {count}\n")

        if language_errors:
            f.write("\nErrors by language:\n")
            for language, count in language_errors.most_common():
                f.write(f"  {language or '<missing>'}: {count}\n")

        if source_errors:
            f.write("\nErrors by data source:\n")
            for source, count in source_errors.most_common():
                f.write(f"  {source or '<missing>'}: {count}\n")

        f.write("\nMost error-prone identifiers:\n")
        for _, row in incorrect_df.sort_values(["error_count", "token_count"], ascending=[False, False]).head(25).iterrows():
            f.write(
                f"  {row['tokens']} | true={row['true_tags']} | pred={row['pred_tags']} | errors={row['error_count']}\n"
            )

    print(f"Wrote LM diagnostics summary to: {summary_path}")
    print(f"Wrote identifier-level errors to: {detail_path}")
    print(f"Wrote token-level errors to: {token_detail_path}")
    print(f"Wrote token confusion matrix to: {confusion_path}")

    fix_impact_dir = os.path.join(output_dir, "fix_impact")
    fix_result = analyze_fix_impact(
        predictions_path=predictions_path,
        output_dir=fix_impact_dir,
    )

    return {
        "summary_path": summary_path,
        "detail_path": detail_path,
        "token_detail_path": token_detail_path,
        "confusion_path": confusion_path,
        "identifier_accuracy": float(df["row_correct"].mean()),
        "token_errors": int(len(error_token_df)),
        "fix_impact": fix_result,
    }