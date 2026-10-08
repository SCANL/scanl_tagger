import random

import numpy as np
import pandas as pd
import pytest
from transformers import DataCollatorForTokenClassification, TrainingArguments

from conftest import LABEL2ID
from src.lm_based_tagger import train_model
from src.lm_based_tagger.distilbert_preprocessing import prepare_dataset, tokenize_and_align_labels


def _rows(*rows, source=train_model.REAL_SOURCE):
    return pd.DataFrame([
        {
            "CONTEXT": ctx, "tokens": tokens, "tags": tags, "TYPE": "", "LANGUAGE": "", "SYSTEM_NAME": "",
            "SPLIT": " ".join(tokens), "DATA_SOURCE": source,
        }
        for ctx, tokens, tags in rows
    ])


def _source_frame(source, n, prefix):
    contexts = ["FUNCTION", "ATTRIBUTE", "DECLARATION", "PARAMETER", "CLASS"]
    return _rows(
        *[(contexts[i % len(contexts)], [prefix, f"w{i}"], ["NM", "N"]) for i in range(n)],
        source=source,
    )


def test_labels_match_tests():
    assert train_model.LABEL2ID == LABEL2ID


def test_accuracy_by_source():
    preds = pd.DataFrame({
        "data_source": ["tagger_data", "tagger_data", "synthetic_pos_data_full"],
        "true_tags":   ["V NM N",      "N",           "NM N"],
        "pred_tags":   ["V N N",       "N",           "NM N"],
    })
    by_source = train_model.accuracy_by_source(preds).set_index("source")
    assert by_source.loc["tagger_data", "identifiers"] == 2
    assert by_source.loc["tagger_data", "token_accuracy"] == pytest.approx(3 / 4)
    assert by_source.loc["tagger_data", "identifier_accuracy"] == pytest.approx(0.5)
    assert by_source.loc["synthetic_pos_data_full", "identifier_accuracy"] == 1.0


def test_compute_metrics_uses_position_aligned_predictions():
    labels = np.array([
        [-100, -100, 3, -100, 4, -100],
        [-100, 8, -100, 3, -100, -100],
    ])
    preds = np.array([
        [0, 0, 3, 9, 4, 0],   # garbage at ignored positions must not matter
        [0, 8, 0, 4, 0, 0],   # second word wrong
    ])
    metrics = train_model.compute_metrics((preds, labels))
    assert metrics["eval_token_accuracy"] == pytest.approx(0.75)
    assert metrics["eval_identifier_accuracy"] == pytest.approx(0.5)


def test_verb_augmentation_only_touches_function_rows():
    df = _rows(
        ("FUNCTION", ["get", "user", "name"], ["V", "NM", "N"]),
        ("ATTRIBUTE", ["adjusted", "load"], ["NM", "V"]),
        ("FUNCTION", ["user", "name"], ["NM", "N"]),
    )
    augmented = train_model._augment_verb_examples(df, random.Random(0))
    assert len(augmented) == 2
    assert (augmented["CONTEXT"] == "FUNCTION").all()
    for _, row in augmented.iterrows():
        assert row["tags"] == ["V", "NM", "N"]
        assert row["tokens"][1:] == ["user", "name"]
        assert row["tokens"][0] in train_model.VERB_SYNONYMS["get"]


def test_prepare_training_frame_upsamples_and_augments():
    df = _rows(
        ("FUNCTION", ["get", "name"], ["V", "N"]),        # low-frequency tag (V) -> x3, then augmented
        ("DECLARATION", ["user", "name"], ["NM", "N"]),
    )
    prepared, n_aug = train_model._prepare_training_frame(df, augmentation_seed=1)
    assert n_aug == 3 * 2   # three copies of the V row, two synonyms each
    assert len(prepared) == 1 + 3 + n_aug
    assert len(df) == 2     # input frame is not modified


def test_prepare_training_frame_can_amplify_real_rows_only():
    real = _rows(("FUNCTION", ["get", "name"], ["V", "N"]))
    synthetic = _rows(("FUNCTION", ["set", "name"], ["V", "N"]), source="synthetic")
    df = pd.concat([real, synthetic], ignore_index=True)

    prepared, n_aug = train_model._prepare_training_frame(df, augmentation_seed=1, amplify="real")
    assert n_aug == 3 * 2   # only the real row is tripled and augmented
    assert (prepared["DATA_SOURCE"] == "synthetic").sum() == 1
    assert len(prepared) == 2 + 2 + n_aug

    _, n_aug_all = train_model._prepare_training_frame(df, augmentation_seed=1, amplify="all")
    assert n_aug_all == 2 * 3 * 2
    with pytest.raises(ValueError):
        train_model._prepare_training_frame(df, augmentation_seed=1, amplify="none")


def test_real_holdout_does_not_depend_on_synthetic_data():
    real = _source_frame(train_model.REAL_SOURCE, 100, "real")
    _, real_only_holdout = train_model._split_holdout(real)

    for n_synthetic in (50, 80):
        synthetic = _source_frame("synthetic", n_synthetic, "syn")
        train, holdout = train_model._split_holdout(pd.concat([real, synthetic], ignore_index=True))
        real_holdout = holdout[holdout["DATA_SOURCE"] == train_model.REAL_SOURCE]
        assert sorted(real_holdout["SPLIT"]) == sorted(real_only_holdout["SPLIT"])
        assert (holdout["DATA_SOURCE"] == "synthetic").sum() == round(n_synthetic * train_model.HOLDOUT_RATIO)
        assert not set(train.index) & set(holdout.index)


def test_drop_synthetic_overlap():
    real = _rows(("FUNCTION", ["get", "Name"], ["V", "N"]))
    synthetic = _rows(
        ("PARAMETER", ["get", "name"], ["V", "N"]),        # real twin, any context or case
        ("ATTRIBUTE", ["max", "size"], ["NM", "N"]),
        ("ATTRIBUTE", ["Max", "Size"], ["NM", "N"]),       # duplicate of the row above
        ("FUNCTION", ["max", "size"], ["NM", "N"]),        # same words, different context: kept
        source="synthetic",
    )
    filtered, dropped = train_model._drop_synthetic_overlap(pd.concat([real, synthetic], ignore_index=True))
    assert dropped == {"real_twins": 1, "duplicates": 1}
    assert list(filtered["SPLIT"]) == ["get Name", "max size", "max size"]
    assert list(filtered["CONTEXT"]) == ["FUNCTION", "ATTRIBUTE", "FUNCTION"]


def test_retrain_epochs_use_the_median_fold():
    assert train_model._select_retrain_epochs([3.0, 5.0, 6.0, 7.0, 5.0]) == 5.0
    assert train_model._select_retrain_epochs([]) == float(train_model.EPOCHS)


def test_crf_trainer_gives_crf_its_own_learning_rate(tiny_model, tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True, learning_rate=5e-5)
    trainer = train_model.CRFTrainer(model=tiny_model, args=args)
    optimizer = trainer.create_optimizer()

    crf_ids = {id(p) for p in tiny_model.crf.parameters()}
    crf_groups = [g for g in optimizer.param_groups if any(id(p) in crf_ids for p in g["params"])]
    assert len(crf_groups) == 1
    assert crf_groups[0]["lr"] == train_model.CRF_LEARNING_RATE
    assert {id(p) for p in crf_groups[0]["params"]} == crf_ids
    other_lrs = {g["lr"] for g in optimizer.param_groups if g is not crf_groups[0]}
    assert other_lrs == {5e-5}


def test_viterbi_predict_aligns_with_labels(tiny_model, tokenizer, monkeypatch):
    monkeypatch.setattr(train_model, "device", tiny_model.crf.transitions.device)
    df = _rows(
        ("FUNCTION", ["get", "employee", "name"], ["V", "NM", "N"]),
        ("DECLARATION", ["size"], ["N"]),
    )
    dataset = prepare_dataset(df, LABEL2ID, selected_features=["context"]).map(lambda s: tokenize_and_align_labels(s, tokenizer))
    preds, labels = train_model.viterbi_predict(
        tiny_model, dataset, DataCollatorForTokenClassification(tokenizer=tokenizer), batch_size=2
    )
    assert len(preds) == len(labels) == 2
    for row_preds, row_labels, expected_words in zip(preds, labels, [3, 1]):
        word_preds = [p for l, p in zip(row_labels[1:-1], row_preds) if l != -100]
        assert len(word_preds) == expected_words
