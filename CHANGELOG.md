# Changelog

All notable changes to SCALAR are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and SCALAR follows [Semantic Versioning](https://semver.org). See the "Versioning" section of the README for what each version bump means.

## [Unreleased]

### Added

- `TaggingBackend.tag_batch(request)`: one batch-tagging function for every transport, following the JSON contract in `src/contract.py`.
- Each token in a response is an object with `text`, `start`, `end`, `tag`, and `dictionary`. `start` and `end` are character offsets into the original name.
- Callers can send their own `tokens` to skip splitting.
- Every response carries a `model` block: `name`, `revision`, `features`, `postprocess`, `device`, and `scalar_version`.
- Per-identifier errors with codes (`EMPTY_IDENTIFIER`, `NO_TOKENS`, `UNSUPPORTED_IDENTIFIER`, `INVALID_CONTEXT`, `INVALID_TOKENS`, `INVALID_IDENTIFIER`, `IDENTIFIER_TOO_LONG`, `INTERNAL_ERROR`), so one bad name doesn't fail the batch.
- `DistilBertTagger.tag_identifiers(rows, batch_size)` for batched inference.
- A pytest suite in `tests/`, run in CI.

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
