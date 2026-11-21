import argparse
import os

import pytest

from scalar_tagger import cli, tagging_backend


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["serve", "--stdio"], ["--mode", "serve", "--stdio"]),
        (["train", "--features", "context"], ["--mode", "train", "--features", "context"]),
        (["--mode", "run"], ["--mode", "run"]),
        (["--version"], ["--version"]),
        ([], []),
    ],
)
def test_normalize_argv(argv, expected):
    assert cli.normalize_argv(argv) == expected


def test_version_does_not_need_a_mode(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"SCALAR tagger {cli.__version__}"


def test_installed_command_does_not_read_serve_json_by_default():
    assert cli.build_parser(discover_local=False).parse_args(["--mode", "serve"]).config_path is None
    assert cli.build_parser(discover_local=True).parse_args(["--mode", "serve"]).config_path == "serve.json"


def args(**overrides):
    values = {"model_dir": None, "local": False, "revision": None}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def base(tmp_path):
    return lambda path: None if not path else (path if os.path.isabs(path) else str(tmp_path / path))


def test_default_is_the_pinned_release_model(base):
    assert cli.resolve_model(args(), {}, base, discover_local=False) == (cli.RELEASE_MODEL, False, cli.RELEASE_REVISION)


def test_revision_flag_overrides_the_pin(base):
    assert cli.resolve_model(args(revision="main"), {}, base, False) == (cli.RELEASE_MODEL, False, "main")


def test_local_model_dir_is_only_discovered_from_a_checkout(base, tmp_path):
    (tmp_path / "output" / "best_model").mkdir(parents=True)
    local = str(tmp_path / "output" / "best_model")
    assert cli.resolve_model(args(), {}, base, discover_local=True) == (local, True, None)
    assert cli.resolve_model(args(), {}, base, discover_local=False)[0] == cli.RELEASE_MODEL


def test_model_dir_wins(base, tmp_path):
    assert cli.resolve_model(args(model_dir="m"), {"model": "x/y"}, base, True) == (str(tmp_path / "m"), True, None)


def test_config_model(base, tmp_path):
    assert cli.resolve_model(args(), {"model": "org/other"}, base, False) == ("org/other", False, None)
    assert cli.resolve_model(args(), {"model": "org/other", "revision": "abc"}, base, False) == ("org/other", False, "abc")
    assert cli.resolve_model(args(), {"model": cli.RELEASE_MODEL}, base, False) == (cli.RELEASE_MODEL, False, cli.RELEASE_REVISION)
    assert cli.resolve_model(args(), {"model": "rel", "local": True}, base, False) == (str(tmp_path / "rel"), True, None)


def test_bundled_word_list():
    words = tagging_backend.load_english_words(extra_words_path=None)
    assert len(words) == 234377
    assert {"get", "user", "token", "basic"} <= words
    assert all(word == word.lower() for word in words)


def test_bundled_word_list_matches_nltk():
    nltk = pytest.importorskip("nltk")
    try:
        expected = set(w.lower() for w in nltk.corpus.words.words())
    except LookupError:
        pytest.skip("NLTK words corpus not downloaded")
    assert tagging_backend.load_english_words(extra_words_path=None) == expected


def test_extra_words_file(tmp_path):
    extra = tmp_path / "en.txt"
    extra.write_text("Grpc\nkubectl\n")
    words = tagging_backend.load_english_words(extra_words_path=str(extra))
    assert {"grpc", "kubectl"} <= words


def test_file_revision_is_cached_until_the_file_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    first = tagging_backend.file_revision(str(weights))
    assert first.startswith("sha256:") and len(first) == len("sha256:") + 16
    assert (tmp_path / "cache" / "scalar-tagger" / "revisions.json").exists()

    calls = []
    real_sha256 = tagging_backend.hashlib.sha256
    monkeypatch.setattr(tagging_backend.hashlib, "sha256", lambda: calls.append(1) or real_sha256())
    assert tagging_backend.file_revision(str(weights)) == first
    assert calls == []  # served from the cache

    weights.write_bytes(b"second, longer")
    assert tagging_backend.file_revision(str(weights)) != first
    assert calls == [1]


def test_cached_snapshot_ignores_branch_names():
    assert tagging_backend.cached_snapshot(cli.RELEASE_MODEL, "main") is None
    assert tagging_backend.cached_snapshot(cli.RELEASE_MODEL, None) is None
