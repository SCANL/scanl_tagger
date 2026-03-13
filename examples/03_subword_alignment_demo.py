from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABELS = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def main():
    print("Example 3: how word-level labels align with DistilBERT subword tokenization")
    print()

    input_tokens = [
        "@func",
        "@lang_java",
        "@pos_0",
        "get",
        "@pos_1",
        "employeeName",
        "@pos_2",
        "details",
    ]
    word_level_labels = [-100, -100, -100, LABEL2ID["V"], -100, LABEL2ID["NM"], -100, LABEL2ID["N"]]

    try:
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    except Exception as exc:
        print("Could not import or load distilbert-base-uncased.")
        print("If this is the first run, you may need to install dependencies and allow a model download.")
        print(f"Underlying error: {exc}")
        return

    encoded = tokenizer(input_tokens, is_split_into_words=True, truncation=True)
    word_ids = encoded.word_ids()
    pieces = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    aligned_labels = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id == previous_word_id:
            aligned_labels.append(-100)
        else:
            aligned_labels.append(word_level_labels[word_id])
        previous_word_id = word_id

    print("Original tokens:")
    print(input_tokens)
    print()
    print("Tokenizer output pieces:")
    print(pieces)
    print()
    print("Piece-by-piece alignment:")
    for piece, word_id, label_id in zip(pieces, word_ids, aligned_labels):
        if label_id == -100:
            label_name = "IGNORE"
        else:
            label_name = ID2LABEL[label_id]
        print(f"  piece={piece:15s} word_id={str(word_id):5s} label={label_name}")

    print()
    print("What to notice:")
    print("- Special tokens like [CLS] and [SEP] have no word id.")
    print("- If one original token becomes multiple pieces, only the first piece keeps the word label.")
    print("- Later pieces are ignored so training and metrics stay aligned to the original word list.")


if __name__ == "__main__":
    main()