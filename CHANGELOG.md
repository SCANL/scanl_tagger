# Changelog

All notable changes to SCALAR are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and SCALAR follows [Semantic Versioning](https://semver.org). See the "Versioning" section of the README for what each version bump means.

## [Unreleased]

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
