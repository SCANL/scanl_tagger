import os
from collections import Counter

import pandas as pd
import pytest

from conftest import LABEL_LIST, REPO_ROOT
from src.lm_based_tagger.distilbert_crf import DistilBertCRFForTokenClassification
from src.lm_based_tagger.distilbert_preprocessing import CONTEXT_MAP

DATA_FILES = {
    "tagger_data": (os.path.join(REPO_ROOT, "input", "tagger_data.tsv"), {"sep": "\t"}),
    "synthetic_pos_data_full": (os.path.join(REPO_ROOT, "input", "synthetic_pos_data_full.csv"), {}),
}
REQUIRED_COLUMNS = ["TYPE", "SPLIT", "CONTEXT", "GRAMMAR_PATTERN", "LANGUAGE", "SYSTEM_NAME"]


@pytest.fixture(scope="module", params=sorted(DATA_FILES))
def dataset(request):
    path, kwargs = DATA_FILES[request.param]
    return pd.read_csv(path, dtype=str, **kwargs)


def test_required_columns_present_and_filled(dataset):
    assert set(REQUIRED_COLUMNS) <= set(dataset.columns)
    assert dataset[REQUIRED_COLUMNS].isna().sum().sum() == 0
    assert len(dataset) > 1000


def test_tokens_and_tags_line_up(dataset):
    lengths = dataset["SPLIT"].str.split().str.len() != dataset["GRAMMAR_PATTERN"].str.split().str.len()
    assert not lengths.any(), dataset[lengths][["SPLIT", "GRAMMAR_PATTERN"]].head().to_string()


def test_tags_and_contexts_are_known(dataset):
    tags = {tag for pattern in dataset["GRAMMAR_PATTERN"] for tag in pattern.split()}
    assert tags <= set(LABEL_LIST)
    assert set(dataset["CONTEXT"].str.strip().str.upper()) <= set(CONTEXT_MAP)


def test_forbidden_transitions_never_occur(dataset):
    # The CRF rules these bigrams out at decode time, which is only safe if the data agrees.
    bigrams = Counter()
    for pattern in dataset["GRAMMAR_PATTERN"]:
        tags = pattern.split()
        bigrams.update(zip(tags, tags[1:]))
    for bigram in DistilBertCRFForTokenClassification.FORBIDDEN_TRANSITIONS:
        assert bigrams[bigram] == 0, bigram


def test_training_loader_combines_both_sources():
    from src.lm_based_tagger.train_model import _load_lm_training_dataframe

    df, sources = _load_lm_training_dataframe(REPO_ROOT)
    assert sources == ["tagger_data", "synthetic_pos_data_full"]
    counts = df["DATA_SOURCE"].value_counts()
    assert counts["tagger_data"] > 2500
    assert counts["synthetic_pos_data_full"] > 1800
    assert (df["tokens"].str.len() == df["tags"].str.len()).all()

    only_tagger, _ = _load_lm_training_dataframe(REPO_ROOT, use_synthetic_data=False)
    assert set(only_tagger["DATA_SOURCE"]) == {"tagger_data"}

    with pytest.raises(ValueError):
        _load_lm_training_dataframe(REPO_ROOT, use_tagger_data=False, use_synthetic_data=False)
