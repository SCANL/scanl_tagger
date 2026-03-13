from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABELS = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def describe_labels(label_ids):
    described = []
    for label_id in label_ids:
        if label_id == -100:
            described.append("IGNORE")
        else:
            described.append(ID2LABEL[label_id])
    return described


def main():
    print("Example 4: the repository's real preprocessing helpers on a tiny in-memory dataset")
    print()

    try:
        import pandas as pd
        from transformers import DistilBertTokenizerFast

        from src.lm_based_tagger.distilbert_preprocessing import prepare_dataset, tokenize_and_align_labels
    except ImportError as exc:
        print("This example depends on the repository's training stack.")
        print("Install the project dependencies first, then re-run this file.")
        print(f"Underlying error: {exc}")
        return

    frame = pd.DataFrame(
        [
            {
                "tokens": ["get", "employee", "name"],
                "tags": ["V", "NM", "N"],
                "CONTEXT": "FUNCTION",
                "TYPE": "String",
                "LANGUAGE": "Java",
                "SYSTEM_NAME": "employee-service",
            },
            {
                "tokens": ["retry", "count"],
                "tags": ["NM", "N"],
                "CONTEXT": "ATTRIBUTE",
                "TYPE": "int",
                "LANGUAGE": "Java",
                "SYSTEM_NAME": "retry-manager",
            },
        ]
    )

    prepared_dataset = prepare_dataset(frame, LABEL2ID)

    print("Prepared dataset rows before DistilBERT tokenization:")
    for row_number in range(len(prepared_dataset)):
        row = prepared_dataset[row_number]
        print(f"Row {row_number + 1} tokens: {row['tokens']}")
        print(f"Row {row_number + 1} labels: {describe_labels(row['ner_tags'])}")
        print()

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    except Exception as exc:
        print("Could not load distilbert-base-uncased.")
        print("If this is the first run, you may need network access or a cached local copy.")
        print(f"Underlying error: {exc}")
        return

    tokenized_dataset = prepared_dataset.map(
        lambda sample: tokenize_and_align_labels(sample, tokenizer),
        batched=False,
    )

    first_row = tokenized_dataset[0]
    tokens = tokenizer.convert_ids_to_tokens(first_row["input_ids"])

    print("First tokenized row after DistilBERT tokenization:")
    print("Subword pieces:", tokens)
    print("Aligned labels:", describe_labels(first_row["labels"]))
    print()
    print("This is the boundary where the repository moves from preprocessing into model training.")
    print("After this point, the trainer passes input_ids, attention_mask, and labels into DistilBERT plus the CRF layer.")


if __name__ == "__main__":
    main()