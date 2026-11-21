import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from scalar_tagger import tag_identifier
from scalar_tagger.tagging_backend import TaggingBackend


class FakeTagger:
    selected_features = ["context"]

    def __init__(self):
        self.calls = []

    def tag_identifiers(self, rows, batch_size=64):
        tags = []
        for row in rows:
            tokens = row["tokens"]
            self.calls.append((tokens, row["context"], row["type_str"], row["language"], row["system_name"]))
            tags.append(["V"] + ["NM"] * (len(tokens) - 2) + ["N"] if len(tokens) > 1 else ["N"])
        return tags


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the cache directory is relative to the working directory
    fake = FakeTagger()
    backend = TaggingBackend("fake/model", tagger=fake, english_words={"get", "user", "name"})
    monkeypatch.setattr(tag_identifier.app, "backend", backend, raising=False)
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
    assert not os.path.exists(os.path.join(REPO_ROOT, "scalar_tagger", "tree_based_tagger"))


def _run_main(monkeypatch, *argv):
    """Run the `main` script in-process with train_lm replaced, returning what it was called with."""
    import runpy
    from scalar_tagger.lm_based_tagger import train_model

    calls = []
    monkeypatch.setattr(train_model, "train_lm", lambda script_dir, **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["main", *argv])
    try:
        runpy.run_path(os.path.join(REPO_ROOT, "main"), run_name="__main__")
    except SystemExit as exc:  # `main` exits with the CLI's status; only failures propagate
        if exc.code:
            raise
    return calls


def test_train_defaults_to_every_non_opt_in_feature(monkeypatch):
    from scalar_tagger.lm_based_tagger.distilbert_preprocessing import DEFAULT_FEATURES

    assert _run_main(monkeypatch, "--mode", "train") == [{"selected_features": DEFAULT_FEATURES}]


def test_train_features_flag_is_passed_through(monkeypatch):
    calls = _run_main(monkeypatch, "--mode", "train", "--features", "type", "context")
    assert calls == [{"selected_features": ["type", "context"]}]  # train_lm puts them in canonical order


@pytest.mark.parametrize("feature", ["bogus", "hungarian_legacy"])
def test_train_rejects_unknown_features(monkeypatch, capsys, feature):
    with pytest.raises(SystemExit) as exit_info:
        _run_main(monkeypatch, "--mode", "train", "--features", "context", feature)
    assert exit_info.value.code == 2
    assert f"invalid choice: '{feature}'" in capsys.readouterr().err


def test_train_no_features_flag(monkeypatch):
    assert _run_main(monkeypatch, "--mode", "train", "--no-features") == [{"selected_features": []}]


def test_features_and_no_features_are_exclusive(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run_main(monkeypatch, "--mode", "train", "--no-features", "--features", "context")
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_train_seed_flag(monkeypatch):
    assert _run_main(monkeypatch, "--mode", "train", "--features", "context", "--seed", "7") == [
        {"selected_features": ["context"], "train_seed": 7}
    ]
    # Without --seed, train_lm's own default applies
    assert "train_seed" not in _run_main(monkeypatch, "--mode", "train")[0]


def test_train_synthetic_data_flags(monkeypatch, capsys):
    calls = _run_main(monkeypatch, "--mode", "train", "--no-features", "--synthetic-data", "input/v2.csv")
    assert calls == [{"selected_features": [], "synthetic_path": os.path.join(REPO_ROOT, "input", "v2.csv")}]

    calls = _run_main(monkeypatch, "--mode", "train", "--no-features", "--no-synthetic-data")
    assert calls == [{"selected_features": [], "use_synthetic_data": False}]

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--mode", "train", "--synthetic-data", "x.csv", "--no-synthetic-data")
    assert "not allowed with argument" in capsys.readouterr().err


def test_train_amplify_flag(monkeypatch):
    calls = _run_main(monkeypatch, "--mode", "train", "--no-features", "--amplify", "real")
    assert calls == [{"selected_features": [], "amplify": "real"}]


def test_train_run_name_flag(monkeypatch):
    calls = _run_main(monkeypatch, "--mode", "train", "--no-features", "--run-name", "tc")
    assert calls == [{"selected_features": [], "run_name": "tc"}]


def test_train_marker_tokens_flag(monkeypatch):
    calls = _run_main(monkeypatch, "--mode", "train", "--no-features", "--marker-tokens")
    assert calls == [{"selected_features": [], "marker_tokens": True}]


def test_seed_help_states_the_real_default():
    from scalar_tagger.lm_based_tagger.train_model import TRAIN_SEED

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "main"), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert f"Default: {TRAIN_SEED}" in " ".join(result.stdout.split())


def test_train_folds_flag(monkeypatch):
    calls = _run_main(monkeypatch, "--mode", "train", "--no-features", "--folds", "3")
    assert calls == [{"selected_features": [], "folds": 3}]


def test_train_folds_flag_rejects_one(monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--mode", "train", "--folds", "1")
