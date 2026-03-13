from pathlib import Path
import random
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


random.seed(7)


LABELS = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
IGNORE_INDEX = -100


TOY_DATA = [
    {
        "tokens": ["@func", "@lang_java", "@pos_0", "get", "@pos_1", "user", "@pos_2", "name"],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, LABEL2ID["V"], IGNORE_INDEX, LABEL2ID["NM"], IGNORE_INDEX, LABEL2ID["N"]],
    },
    {
        "tokens": ["@func", "@lang_java", "@pos_0", "set", "@pos_1", "user", "@pos_2", "name"],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, LABEL2ID["V"], IGNORE_INDEX, LABEL2ID["NM"], IGNORE_INDEX, LABEL2ID["N"]],
    },
    {
        "tokens": ["@attr", "@lang_java", "@pos_0", "user", "@pos_1", "id", "@pos_2", "2"],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, LABEL2ID["NM"], IGNORE_INDEX, LABEL2ID["N"], IGNORE_INDEX, LABEL2ID["D"]],
    },
    {
        "tokens": ["@attr", "@lang_java", "@pos_0", "max", "@pos_1", "retry", "@pos_2", "count"],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, LABEL2ID["NM"], IGNORE_INDEX, LABEL2ID["NM"], IGNORE_INDEX, LABEL2ID["N"]],
    },
    {
        "tokens": ["@param", "@lang_csharp", "@pos_0", "with", "@pos_1", "new", "@pos_1", "cache", "@pos_2", "key"],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, LABEL2ID["P"], IGNORE_INDEX, LABEL2ID["NM"], IGNORE_INDEX, LABEL2ID["NM"], IGNORE_INDEX, LABEL2ID["N"]],
    },
]


def build_vocab(rows):
    vocab = {"<pad>": 0}
    for row in rows:
        for token in row["tokens"]:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_row(row, vocab):
    token_ids = [vocab[token] for token in row["tokens"]]
    return token_ids, row["labels"]


def pad_batch(encoded_rows, pad_token_id):
    max_length = max(len(token_ids) for token_ids, _ in encoded_rows)
    batch_inputs = []
    batch_labels = []
    for token_ids, labels in encoded_rows:
        padding_length = max_length - len(token_ids)
        batch_inputs.append(token_ids + [pad_token_id] * padding_length)
        batch_labels.append(labels + [IGNORE_INDEX] * padding_length)
    return batch_inputs, batch_labels


def readable_predictions(tokens, predicted_ids, gold_ids):
    rows = []
    for token, predicted_id, gold_id in zip(tokens, predicted_ids, gold_ids):
        predicted_label = ID2LABEL[predicted_id] if gold_id != IGNORE_INDEX else "IGNORE"
        gold_label = ID2LABEL[gold_id] if gold_id != IGNORE_INDEX else "IGNORE"
        rows.append((token, predicted_label, gold_label))
    return rows


def main():
    print("Example 2: a tiny neural network that learns token labels")
    print()

    try:
        import importlib

        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as exc:
        print("This example needs PyTorch, which is not installed in the active virtual environment.")
        print("Install the project dependencies first, then re-run this file.")
        print(f"Underlying error: {exc}")
        return

    torch.manual_seed(7)

    class TinySequenceTagger(nn.Module):
        def __init__(self, vocab_size, embedding_dim, num_labels):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_labels)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids)
            return self.classifier(embedded)

    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    vocab = build_vocab(TOY_DATA)
    encoded_rows = [encode_row(row, vocab) for row in TOY_DATA]
    print("Encoded rows (token ids and label ids):")
    pp.pprint(encoded_rows)
    padded_inputs, padded_labels = pad_batch(encoded_rows, pad_token_id=vocab["<pad>"])
    print("Padded input token ids:")
    pp.pprint(padded_inputs)
    print("Padded label ids:")
    pp.pprint(padded_labels)
    input_tensor = torch.tensor(padded_inputs)
    label_tensor = torch.tensor(padded_labels)

    model = TinySequenceTagger(vocab_size=len(vocab), embedding_dim=16, num_labels=len(LABELS))
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    print("Vocabulary:", vocab)
    print("Input tensor shape:", tuple(input_tensor.shape))
    print("Label tensor shape:", tuple(label_tensor.shape))
    print()

    logits_before = model(input_tensor[:1])
    print("Logits for the first example before training:")
    pp.pprint(logits_before[0, :, :4].detach())
    print()

    for epoch in range(1, 201):
        optimizer.zero_grad()
        logits = model(input_tensor)
        loss = loss_fn(logits.view(-1, len(LABELS)), label_tensor.view(-1))
        loss.backward()
        optimizer.step()

        if epoch in {1, 20, 50, 100, 200}:
            print(f"Epoch {epoch:>3} | loss = {loss.item():.4f}")

    print()
    with torch.no_grad():
        trained_logits = model(input_tensor)
        predictions = trained_logits.argmax(dim=-1)

    for row, predicted_ids in zip(TOY_DATA, predictions):
        print("Tokens:", row["tokens"])
        for token, predicted_label, gold_label in readable_predictions(row["tokens"], predicted_ids.tolist(), row["labels"]):
            print(f"  {token:12s} predicted={predicted_label:6s} gold={gold_label}")
        print()

    print("What this example is doing:")
    print("- The embedding layer learns a vector for every token in the toy vocabulary.")
    print("- The linear layer maps each token vector to one score per label.")
    print("- Cross-entropy pushes the correct label score upward for non-ignored positions.")
    print("- This mirrors the final token-classification step in the real model, but without context from neighboring tokens.")


if __name__ == "__main__":
    main()