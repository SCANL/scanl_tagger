import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import DistilBertConfig, DistilBertTokenizerFast  # noqa: E402

from src.lm_based_tagger.distilbert_crf import DistilBertCRFForTokenClassification  # noqa: E402

LABEL_LIST = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# A small WordPiece vocabulary so tests never download distilbert-base-uncased.
# Feature/position tokens split on punctuation (e.g. "@pos_0" -> "@", "pos", "_", "0").
VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "@", "_", "pos", "0", "1", "2",
    "get", "set", "user", "name", "token", "max", "size", "items", "is", "valid",
    "em", "##ploy", "##ee", "func", "param", "attr", "decl", "class",
]


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory):
    vocab_dir = tmp_path_factory.mktemp("vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    return DistilBertTokenizerFast(vocab=str(vocab_file), do_lower_case=True, model_max_length=128)


def make_config(**overrides):
    kwargs = dict(
        vocab_size=len(VOCAB),
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=128,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    kwargs.update(overrides)
    return DistilBertConfig(**kwargs)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = DistilBertCRFForTokenClassification(
        num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID, config=make_config()
    )
    model.eval()
    return model


def save_checkpoint(model, tokenizer, path):
    """Save a checkpoint the same way train_lm does (config + model.safetensors + tokenizer)."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    model.config.save_pretrained(path)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(path, "model.safetensors"))
    tokenizer.save_pretrained(path)
    return str(path)
