# Changelog

All notable changes to SCALAR are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and SCALAR follows [Semantic Versioning](https://semver.org). See the "Versioning" section of the README for what each version bump means.

## 3.0.0 - Unreleased

### Changed

- The CRF runs over identifier words only. Training and Viterbi decoding used to run over every inner token (feature tokens, `@pos_N` tokens, and continuation subwords), so the learned transitions never connected adjacent words. Decoding also rules out the tag bigrams `N N`, `N NPL`, and `NPL NPL`, which never occur in the training data. Checkpoints trained before this change should be retrained.
- Continuation subwords (e.g. `##ployee`) are no longer labeled during training; only the first subword of each word carries its tag.
- The CRF transition parameters train with their own learning rate (`5e-3`), instead of the encoder's `5e-5`.
- The LM model trains on `input/tagger_data.tsv` (replaced with the corrected, cross-validated annotations) plus `input/synthetic_pos_data_full.csv`.
- LM training: 20% holdout, stratified 5-fold CV by context, low-frequency tag upsampling and verb-synonym augmentation (FUNCTION rows only) applied inside each fold, then a final retrain on the whole training split for the best fold's epoch count.
- The holdout report shows token and identifier accuracy for each data source, since synthetic rows are much easier than real identifiers.
- `python main --mode train --features ...` chooses which feature tokens to train with, and `--no-features` trains without any. The default is all of them.
- New feature tokens for the LM model: digit connectors, plural suffixes, declared type, type/name overlap, language, and system-name overlap. The feature list is saved in the checkpoint config, and inference uses it. Checkpoints without one (such as `sourceslicer/scalar_lm_best`) use the original four features.
- Inference runs on the GPU when one is available.
- `version.py` is the single source of the version, and is validated as a semantic version at import time.
- `python main --version` works without `--mode`, and prints `SCALAR tagger <version>`.
- `--model_type` is optional and accepts only `lm_based`.
- Run mode downloads only the NLTK `words` corpus (used by the `dictionary` flag), and only when it is missing.
- The Docker image serves the DistilBERT model over HTTP on port 8080, instead of starting training.
- Dependencies are updated to their latest releases, including `torch` 2.14.1, `transformers` 5.19.0, `datasets` 5.1.0, `pandas` 3.0.6, and `scikit-learn` 1.9.1. `transformers`, `accelerate`, `huggingface_hub`, `safetensors`, `numpy`, and `pandas` are now pinned directly.

### Removed

- The tree-based (Gradient Boosting) tagger: `src/tree_based_tagger/`, `models/model_GradientBoostingClassifier.pkl`, its training database `input/scanl_tagger_training_db_11_29_2024.db`, and the `gensim` dependency. It was slower (8.6 vs. about 360 identifiers per second) and less accurate than the DistilBERT+CRF model.
- `--model_type tree_based` and `--model_dir`, which only applied to the tree-based model.
- The unused `aiohttp` and `pipdeptree` dependencies.

### Added

- A pytest suite in `tests/`, run in CI.
- This changelog.

### Fixed

- `--address`, `--port`, `--protocol`, and `--word` apply when serving the DistilBERT model; they used to apply only to the tree-based model.
- `--config_path` is honored; the server always read `serve.json` before.
- Fold and holdout evaluation decode with the CRF, instead of taking the argmax of the emission scores.
- The `hungarian` feature looked for its prefix inside the first word *after* splitting, so it almost never fired (20 of 4,383 training rows). It now detects a single lowercase letter followed by a capitalized word (`f Matcher`, `b Force`), which is a preamble about 80% of the time in the training data. Checkpoints trained with `hungarian` before this fix should be retrained; checkpoints without a saved feature list (such as `sourceslicer/scalar_lm_best`) keep the original behavior.
- An identifier too long for the model input raises an error, instead of silently returning fewer tags than tokens.

## 2.2.0

Last release before semantic versioning was enforced.
