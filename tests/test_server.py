import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from src import tag_identifier


class FakeTagger:
    def __init__(self):
        self.calls = []

    def tag_identifier(self, tokens, context, type_str, language, system_name):
        self.calls.append((tokens, context, type_str, language, system_name))
        return ["V"] + ["NM"] * (len(tokens) - 2) + ["N"] if len(tokens) > 1 else ["N"]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the cache directory is relative to the working directory
    fake = FakeTagger()
    monkeypatch.setattr(tag_identifier, "lm_model", fake)
    tag_identifier.app.english_words = {"get", "user", "name"}
    tag_identifier.app.words = tag_identifier.WordList(str(tmp_path / "missing.csv"))
    return tag_identifier.app.test_client(), fake


def test_tag_route_uses_distilbert_tagger(client):
    http, fake = client
    response = http.get("/getUserName/FUNCTION?language=Java&type=String&system_name=demo")
    assert response.status_code == 200
    assert response.get_json() == {"words": [
        {"get": {"tag": "V", "dictionary": "DW"}},
        {"User": {"tag": "NM", "dictionary": "DW"}},
        {"Name": {"tag": "N", "dictionary": "DW"}},
    ]}
    assert fake.calls == [(["get", "User", "Name"], "FUNCTION", "String", "Java", "demo")]


def test_tag_route_caches_results(client, tmp_path):
    http, fake = client
    os.makedirs("cache")
    first = http.get("/getUserName/FUNCTION/proj").get_json()
    second = http.get("/getUserName/FUNCTION/proj").get_json()
    assert len(fake.calls) == 1
    assert second["words"] == first["words"]


def test_tree_based_model_type_is_gone():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "main"), "--mode", "train", "--model_type", "tree_based"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2
    assert "invalid choice: 'tree_based'" in result.stderr
    assert not os.path.exists(os.path.join(REPO_ROOT, "src", "tree_based_tagger"))
