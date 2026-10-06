# Changelog

All notable changes to SCALAR are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and SCALAR follows [Semantic Versioning](https://semver.org). See the "Versioning" section of the README for what each version bump means.

## [Unreleased]

### Added

- A `pyproject.toml` package, `scalar-tagger`, with a `scalar-tagger` command (`scalar-tagger serve --stdio`, `scalar-tagger train ...`). Install it with `pip install "scalar-tagger @ git+https://github.com/SCANL/scanl_tagger.git"`. Training dependencies are in the `[train]` extra.
- `--revision` (and `"revision"` in the config file) to load a specific Hugging Face commit, branch, or tag.
- The release model is pinned: `sourceslicer/scalar_lm_release_test` at commit `41a3c953a6ec5612834a16b0da54d809531af2ee` (`RELEASE_MODEL` and `RELEASE_REVISION` in `scalar_tagger/cli.py`).
- `--mode serve --stdio`: serves the batch contract as JSON lines over stdin/stdout. It prints a `ready` line once the model has loaded, sends all logging to stderr, answers `{"command": "info"}`, and exits when stdin closes.
- `POST /tag` and `GET /info` HTTP endpoints, using the same contract as stdio.
- `--device` (and `"device"` in the config file) to choose `cpu`, `cuda`, or `cuda:N` instead of detecting the GPU automatically.
- `--mode serve`, the same as `--mode run`.
- `TaggingBackend.tag_batch(request)`: one batch-tagging function for every transport, following the JSON contract in `src/contract.py`.
- Each token in a response is an object with `text`, `start`, `end`, `tag`, and `dictionary`. `start` and `end` are character offsets into the original name.
- Callers can send their own `tokens` to skip splitting.
- Every response carries a `model` block: `name`, `revision`, `features`, `postprocess`, `device`, and `scalar_version`.
- Per-identifier errors with codes (`EMPTY_IDENTIFIER`, `NO_TOKENS`, `UNSUPPORTED_IDENTIFIER`, `INVALID_CONTEXT`, `INVALID_TOKENS`, `INVALID_IDENTIFIER`, `IDENTIFIER_TOO_LONG`, `INTERNAL_ERROR`), so one bad name doesn't fail the batch.
- `DistilBertTagger.tag_identifiers(rows, batch_size)` for batched inference.
- A pytest suite in `tests/`, run in CI.

### Changed

- The package moved from `src/` to `scalar_tagger/`, and the version from `version.py` to `scalar_tagger/version.py`. `python main` still works from a checkout.
- The installed command doesn't read `serve.json` or use `output/best_model` unless told to. `python main` still does both.
- Startup takes about 7 seconds to a ready line, down from about 9, and the tagger warms up before reporting ready, so the first request isn't slower. `--version` returns in well under a second.
  - The `dictionary` word list is bundled (`scalar_tagger/data/english_words.txt.gz`), so NLTK data is never downloaded, and loading it takes 0.07 seconds instead of 2.3.
  - Inference no longer imports `pandas` or `datasets`.
  - Loading a checkpoint no longer downloads or reads the base `distilbert-base-uncased` weights first.
  - A pinned model that is already downloaded loads without contacting Hugging Face, so the tagger starts offline.
  - The hash that identifies a local checkpoint is cached in `~/.cache/scalar-tagger/revisions.json`.
- `requirements.txt` pins `transformers` and `accelerate`, and adds `pytest`.
- The legacy `GET /<name>/<context>` route goes through the shared backend. Its responses are unchanged.
- The `dictionary` flag in every transport uses the NLTK words corpus plus `words/en.txt`, if that file exists.

### Removed

- `setup.py`, the root `__init__.py`, and the `nltk` direct dependency. `nltk` is still installed, because the `spiral` splitter needs it.

### Fixed

- Inference runs on the GPU when one is available. It always ran on the CPU before (about 25 identifiers per second instead of about 360).
- An identifier too long for the model input is reported as an error, instead of silently returning fewer tags than tokens.

## [3.0.0] - 2026-10-01

### Removed

- The tree-based (Gradient Boosting) tagger: `src/tree_based_tagger/`, `models/model_GradientBoostingClassifier.pkl`, and the `gensim` dependency. It was slower (8.6 vs. about 490 identifiers per second) and less accurate than the DistilBERT+CRF model.
- `--model_type tree_based`. `--model_type` is now optional and accepts only `lm_based`.
- The `diagnostics` run mode, whose `error_analysis_scripts` module had already been removed from the repository.

### Changed

- `python main --version` works without `--mode`, and prints `SCALAR tagger <version>`.
- `version.py` is the single source of the version, and is validated as a semantic version at import time.
- Run mode downloads only the NLTK `words` corpus, and only when it is missing. The tree-based model's NLTK resources are no longer downloaded.
- The Docker image serves the DistilBERT model on port 8080, instead of starting training.

### Fixed

- `main` no longer fails at startup with `ModuleNotFoundError: error_analysis_scripts`.

## [2.2.0]

Last release with the tree-based tagger.

[Unreleased]: https://github.com/SCANL/scanl_tagger/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/SCANL/scanl_tagger/compare/v2.2.0...v3.0.0
[2.2.0]: https://github.com/SCANL/scanl_tagger/releases/tag/v2.2.0
