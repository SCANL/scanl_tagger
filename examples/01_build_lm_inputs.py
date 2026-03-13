from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABELS = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}


TOY_ROWS = [
    {
        "tokens": ["get", "employee", "name"],
        "tags": ["V", "NM", "N"],
        "CONTEXT": "FUNCTION",
        "TYPE": "std::string",
        "LANGUAGE": "C++",
        "SYSTEM_NAME": "employee-service",
    },
    {
        "tokens": ["max", "retry", "count"],
        "tags": ["NM", "NM", "N"],
        "CONTEXT": "ATTRIBUTE",
        "TYPE": "int",
        "LANGUAGE": "Java",
        "SYSTEM_NAME": "retry-manager",
    },
    {
        "tokens": ["user", "id", "2"],
        "tags": ["NM", "N", "D"],
        "CONTEXT": "PARAMETER",
        "TYPE": "Guid",
        "LANGUAGE": "C#",
        "SYSTEM_NAME": "user-api",
    },
]


def build_input_sequence(row, build_model_input_tokens, normalize_selected_features):
    active_features = normalize_selected_features()
    full_tokens, feature_token_count = build_model_input_tokens(
        row,
        row["tokens"],
        selected_features=active_features,
    )

    aligned_labels = [-100] * feature_token_count
    for tag in row["tags"]:
        aligned_labels.extend([-100, LABEL2ID[tag]])
    return full_tokens, aligned_labels, active_features


def pretty_label(label_id):
    if label_id == -100:
        return "IGNORE"
    return LABELS[label_id]


def main():
    print("Example 1: how the repository turns one identifier into a token-classification sequence")
    print()

    try:
        from src.lm_based_tagger.distilbert_preprocessing import (
            DEFAULT_FEATURES,
            build_model_input_tokens,
            get_feature_tokens,
            get_number_of_features,
            normalize_selected_features,
        )
    except ImportError as exc:
        print("This example uses the repository's real preprocessing module.")
        print("Install the project dependencies first, then re-run this file.")
        print(f"Underlying error: {exc}")
        return

    print("Default active features:", DEFAULT_FEATURES)
    print()

    for example_number, row in enumerate(TOY_ROWS, start=1):
        full_tokens, aligned_labels, active_features = build_input_sequence(
            row,
            build_model_input_tokens,
            normalize_selected_features,
        )
        feature_tokens = get_feature_tokens(row, row["tokens"], selected_features=active_features)

        print(f"=== Row {example_number} ===")
        print("Original tokens:", row["tokens"])
        print("Gold tags:      ", row["tags"])
        print("Feature names:  ", active_features)
        print("Feature tokens: ", feature_tokens)
        print("Feature count:  ", get_number_of_features(active_features))
        print("Full sequence:  ", full_tokens)
        print("Label ids:      ", aligned_labels)
        print("Label names:    ", [pretty_label(label_id) for label_id in aligned_labels])
        print()

    print("What to notice:")
    print("- This example imports the repository's real preprocessing helpers, so it stays in sync with training.")
    print("- Feature tokens are inserted before any identifier words.")
    print("- Each real word gets a position marker immediately before it.")
    print("- Only real words receive trainable labels.")
    print("- Feature and position tokens are masked with -100, so the loss ignores them.")


if __name__ == "__main__":
    main()