import pandas as pd
import pytest

from conftest import LABEL2ID, REPO_ROOT
from src.lm_based_tagger.distilbert_preprocessing import (
    AVAILABLE_FEATURES,
    LEGACY_FEATURES,
    build_model_input_tokens,
    get_feature_tokens,
    get_plural_suffix_feature_tokens,
    get_type_feature_tokens,
    get_type_overlap_feature_tokens,
    normalize_selected_features,
    prepare_dataset,
    tokenize_and_align_labels,
)

ROW = {"CONTEXT": "FUNCTION", "TYPE": "const char*", "LANGUAGE": "C++", "SYSTEM_NAME": "drill"}


def test_normalize_selected_features():
    assert normalize_selected_features(None) == AVAILABLE_FEATURES
    # Canonical order, whatever order the caller used
    assert normalize_selected_features(["digit", "context"]) == ["context", "digit"]
    assert normalize_selected_features(LEGACY_FEATURES) == ["context", "hungarian", "cvr", "digit"]
    with pytest.raises(ValueError, match="nope"):
        normalize_selected_features(["context", "nope"])


def test_legacy_features_match_the_original_four():
    tokens = ["get", "user", "name"]
    assert get_feature_tokens(ROW, tokens, LEGACY_FEATURES) == ["@func", "@hung_none", "@cvr_low", "@no_digit"]


def test_build_model_input_tokens_interleaves_positions():
    tokens = ["get", "user", "name"]
    full, n_features = build_model_input_tokens(ROW, tokens, ["context"])
    assert n_features == 1
    assert full == ["@func", "@pos_0", "get", "@pos_1", "user", "@pos_2", "name"]

    single, _ = build_model_input_tokens(ROW, ["size"], ["context"])
    assert single == ["@func", "@pos_2", "size"]


def test_no_features():
    full, n_features = build_model_input_tokens(ROW, ["get", "name"], [])
    assert n_features == 0
    assert full == ["@pos_0", "get", "@pos_2", "name"]
    assert normalize_selected_features([]) == []


def test_multi_token_features_are_counted():
    tokens = ["get", "user", "name"]
    full, n_features = build_model_input_tokens(ROW, tokens, AVAILABLE_FEATURES)
    assert n_features == len(get_feature_tokens(ROW, tokens, AVAILABLE_FEATURES))
    assert n_features > len(AVAILABLE_FEATURES)  # several features emit more than one token
    assert full[n_features:] == ["@pos_0", "get", "@pos_1", "user", "@pos_2", "name"]


def test_type_features():
    tokens = get_type_feature_tokens("const char*")
    assert tokens[0] == "@type_char_like"
    assert {"@type_ptr", "@type_const"} <= set(tokens)
    assert get_type_feature_tokens("") == ["@type_unknown"]
    assert get_type_feature_tokens("std::vector<int>")[0] == "@type_container_like"
    assert get_type_feature_tokens("uint32_t")[0] == "@type_int_like"
    assert get_type_overlap_feature_tokens(["user", "list"], "UserList") == [
        "@typeov_head_exact", "@typeov_lead_2plus", "@typeov_any_2plus",
    ]


def test_plural_suffix_features():
    assert get_plural_suffix_feature_tokens(["items"]) == [
        "@plural_suffix_present", "@plural_suffix_s", "@plural_suffix_tail",
    ]
    assert get_plural_suffix_feature_tokens(["status", "class"]) == ["@plural_suffix_none"]


def test_prepare_dataset_labels_only_words():
    df = pd.DataFrame([{**ROW, "tokens": ["get", "user", "name"], "tags": ["V", "NM", "N"]}])
    ds = prepare_dataset(df, LABEL2ID, selected_features=["context", "digit"])
    assert ds[0]["tokens"] == ["@func", "@no_digit", "@pos_0", "get", "@pos_1", "user", "@pos_2", "name"]
    assert ds[0]["ner_tags"] == [-100, -100, -100, LABEL2ID["V"], -100, LABEL2ID["NM"], -100, LABEL2ID["N"]]


def test_continuation_subwords_are_ignored(tokenizer):
    df = pd.DataFrame([{**ROW, "tokens": ["get", "employee"], "tags": ["V", "N"]}])
    sample = prepare_dataset(df, LABEL2ID, selected_features=["context"])[0]
    tokenized = tokenize_and_align_labels(sample, tokenizer)

    pieces = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    labeled = [(piece, label) for piece, label in zip(pieces, tokenized["labels"]) if label != -100]
    # "employee" -> em ##ploy ##ee: only the first subword carries the tag
    assert labeled == [("get", LABEL2ID["V"]), ("em", LABEL2ID["N"])]
    assert pieces[pieces.index("em") + 1:pieces.index("em") + 3] == ["##ploy", "##ee"]


def test_every_training_row_builds_features():
    from src.lm_based_tagger.train_model import LABEL2ID as TRAIN_LABEL2ID, _load_lm_training_dataframe

    df, _ = _load_lm_training_dataframe(REPO_ROOT)
    ds = prepare_dataset(df, TRAIN_LABEL2ID)
    assert len(ds) == len(df)
    for tokens, tags in zip(ds["tokens"], ds["ner_tags"]):
        assert len(tokens) == len(tags)
        assert not any(t in ("@nan", "@lang_nan") or "nan" == t for t in tokens)
