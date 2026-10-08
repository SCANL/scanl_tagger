import glob
import os
from collections import Counter

import pandas as pd
import pytest

from conftest import LABEL_LIST, REPO_ROOT
from src.lm_based_tagger.distilbert_crf import DistilBertCRFForTokenClassification
from src.lm_based_tagger.distilbert_preprocessing import CONTEXT_MAP

REAL_DATA = os.path.join(REPO_ROOT, "input", "tagger_data.tsv")
SYNTHETIC_FILES = sorted(glob.glob(os.path.join(REPO_ROOT, "input", "synthetic*.csv")))
# Kept unchanged for comparison with later versions; the training loader drops its real twins.
SYNTHETIC_V1 = os.path.join(REPO_ROOT, "input", "synthetic_pos_data_full.csv")
DATA_FILES = {
    "tagger_data": (REAL_DATA, {"sep": "\t"}),
    **{os.path.splitext(os.path.basename(path))[0]: (path, {}) for path in SYNTHETIC_FILES},
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


@pytest.mark.parametrize("path", [
    pytest.param(
        path,
        id=os.path.basename(path),
        marks=[pytest.mark.xfail(strict=True, reason="v1 is kept unchanged; the loader drops its real twins")]
        if path == SYNTHETIC_V1 else [],
    )
    for path in SYNTHETIC_FILES
])
def test_synthetic_file_has_no_real_twins(path):
    # A synthetic copy of a real identifier can leak a real holdout answer into training.
    from src.lm_based_tagger.train_model import normalize_split

    real = set(pd.read_csv(REAL_DATA, sep="\t", dtype=str)["SPLIT"].map(normalize_split))
    synthetic = pd.read_csv(path, dtype=str)
    twins = synthetic[synthetic["SPLIT"].map(normalize_split).isin(real)]
    assert twins.empty, twins[["SPLIT", "CONTEXT"]].head(10).to_string()


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


def test_training_loader_reads_the_chosen_synthetic_file(tmp_path):
    from src.lm_based_tagger.train_model import _load_lm_training_dataframe

    path = tmp_path / "synthetic_v2.csv"
    pd.read_csv(SYNTHETIC_V1, dtype=str).head(20).to_csv(path, index=False)
    df, sources = _load_lm_training_dataframe(REPO_ROOT, synthetic_path=str(path))
    assert sources == ["tagger_data", "synthetic_v2"]
    assert df["DATA_SOURCE"].value_counts()["synthetic_v2"] == 20


def test_class_rows_use_class_as_their_type(dataset):
    # A class's TYPE is the keyword `class`, never the class's own name (which would make
    # type_overlap report that the whole name is its type).
    classes = dataset[dataset["CONTEXT"] == "CLASS"]
    assert (classes["TYPE"].str.lower() == "class").all(), classes[classes["TYPE"].str.lower() != "class"][["SPLIT", "TYPE"]].head()
