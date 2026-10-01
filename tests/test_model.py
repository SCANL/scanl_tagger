"""Checks against the real checkpoint in output/best_model. Skipped when it isn't present."""

import csv
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "output", "best_model")
DATA_PATH = os.path.join(ROOT, "input", "tagger_data_new.tsv")

pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="no local model in output/best_model")


@pytest.fixture(scope="module")
def backend():
    from src.tagging_backend import TaggingBackend
    return TaggingBackend(MODEL_DIR, local=True)


def gold_rows(limit=None):
    with open(DATA_PATH, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[:limit] if limit else rows


def test_batched_tags_match_one_at_a_time(backend):
    rows = [
        {
            "tokens": row["SPLIT"].split(),
            "context": row["CONTEXT"],
            "type_str": row["TYPE"],
            "language": row["LANGUAGE"],
            "system_name": row["SYSTEM_NAME"],
        }
        for row in gold_rows(limit=300)
    ]
    batched = backend.lm_model.tag_identifiers(rows, batch_size=64)
    single = [backend.lm_model.tag_identifier(**row) for row in rows]
    assert batched == single


def test_tag_batch_end_to_end(backend):
    response = backend.tag_batch({
        "id": "r1",
        "identifiers": [
            {"key": 1, "name": "getUserToken", "context": "FUNCTION", "type": "string"},
            {"key": 2, "name": "~Foo", "context": "FUNCTION"},
            {"key": 3, "name": "a" * 2000, "context": "DECLARATION"},
        ],
    })
    assert response["model"]["revision"].startswith("sha256:")
    first, second, third = response["results"]
    assert [t["tag"] for t in first["tokens"]] == ["V", "NM", "N"]
    assert second["error"]["code"] == "UNSUPPORTED_IDENTIFIER"
    assert third["error"]["code"] == "IDENTIFIER_TOO_LONG"
