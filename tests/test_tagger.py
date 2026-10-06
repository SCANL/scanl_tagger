import pytest
import torch

from conftest import ID2LABEL, LABEL2ID, LABEL_LIST, save_checkpoint
from src.lm_based_tagger.distilbert_preprocessing import LEGACY_FEATURES
from src.lm_based_tagger.distilbert_tagger import DistilBertTagger


def _tagger(model, tokenizer, path):
    return DistilBertTagger(save_checkpoint(model, tokenizer, path), local=True, device="cpu")


def test_tagger_uses_saved_features(tiny_model, tokenizer, tmp_path):
    tiny_model.config.selected_features = ["context", "type"]
    tagger = _tagger(tiny_model, tokenizer, tmp_path / "ckpt")
    assert tagger.selected_features == ["context", "type"]

    tags = tagger.tag_identifier(["get", "employee", "name"], "FUNCTION", "int", "C++", "drill")
    assert len(tags) == 3
    assert set(tags) <= set(LABEL_LIST)


def test_tagger_falls_back_to_legacy_features(tiny_model, tokenizer, tmp_path):
    # Checkpoints published before the feature list was saved (e.g. sourceslicer/scalar_lm_best)
    assert not hasattr(tiny_model.config, "selected_features")
    tagger = _tagger(tiny_model, tokenizer, tmp_path / "ckpt")
    assert tagger.selected_features == LEGACY_FEATURES


def test_tagger_reads_first_subword_of_each_word(tiny_model, tokenizer, tmp_path):
    """Each word's tag must come from the CRF prediction at that word's first subword."""
    tagger = _tagger(tiny_model, tokenizer, tmp_path / "ckpt")
    seen = {}

    def fake_forward(input_ids, attention_mask, word_mask):
        seen["word_mask"] = word_mask
        ids = input_ids[0].tolist()
        # predictions exclude [CLS]; label each inner position by its token id
        return {"predictions": [[tok % len(LABEL_LIST) for tok in ids[1:-1]]]}

    tagger.model.forward = fake_forward
    words = ["max", "employee", "size"]
    tags = tagger.tag_identifier(words, "DECLARATION", "", "", "")

    first_subwords = ["max", "em", "size"]
    expected = [ID2LABEL[tokenizer.convert_tokens_to_ids(t) % len(LABEL_LIST)] for t in first_subwords]
    assert tags == expected
    assert int(seen["word_mask"].sum()) == len(words)


def test_tagger_end_to_end_respects_forbidden_transitions(tiny_model, tokenizer, tmp_path):
    # Make every word prefer N, then NM: the unconstrained answer would be N N.
    with torch.no_grad():
        tiny_model.classifier.weight.zero_()
        tiny_model.classifier.bias.fill_(-5.0)
        tiny_model.classifier.bias[LABEL2ID["N"]] = 5.0
        tiny_model.classifier.bias[LABEL2ID["NM"]] = 4.0
        tiny_model.crf.transitions.zero_()
        tiny_model.crf.start_transitions.zero_()
        tiny_model.crf.end_transitions.zero_()
        tiny_model.crf.end_transitions[LABEL2ID["N"]] = 1.0
    tagger = _tagger(tiny_model, tokenizer, tmp_path / "ckpt")
    assert tagger.tag_identifier(["max", "size"], "DECLARATION", "int", "C", "x") == ["NM", "N"]
    # N NM N is allowed (and scores higher than NM NM N); only N->N/NPL and NPL->NPL are ruled out
    assert tagger.tag_identifier(["max", "user", "size"], "DECLARATION", "int", "C", "x") == ["N", "NM", "N"]


def test_tagger_rejects_identifiers_that_do_not_fit(tiny_model, tokenizer, tmp_path):
    tagger = _tagger(tiny_model, tokenizer, tmp_path / "ckpt")
    with pytest.raises(ValueError, match="too long"):
        tagger.tag_identifier(["name"] * 100, "DECLARATION", "", "", "")
