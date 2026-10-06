# Changelog

All notable changes to SCALAR are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and SCALAR follows [Semantic Versioning](https://semver.org). See the "Versioning" section of the README for what each version bump means.

## 3.0.0 - Unreleased

### Changed

- The CRF runs over identifier words only. Training and Viterbi decoding used to run over every inner token (feature tokens, `@pos_N` tokens, and continuation subwords), so the learned transitions never connected adjacent words. Decoding also rules out the tag bigrams `N N`, `N NPL`, and `NPL NPL`, which never occur in the training data. Checkpoints trained before this change should be retrained.
- Continuation subwords (e.g. `##ployee`) are no longer labeled during training; only the first subword of each word carries its tag.
- The CRF transition parameters train with their own learning rate (`5e-3`), instead of the encoder's `5e-5`.
- The LM model trains on `input/tagger_data.tsv` (replaced with the corrected, cross-validated annotations) plus `input/synthetic_pos_data_full.csv`.
- LM training: 20% holdout, stratified 5-fold CV by context, low-frequency tag upsampling and verb-synonym augmentation (FUNCTION rows only) applied inside each fold, then a final retrain on the whole training split for the best fold's epoch count.
- New feature tokens for the LM model: digit connectors, plural suffixes, declared type, type/name overlap, language, and system-name overlap. The feature list is saved in the checkpoint config, and inference uses it. Checkpoints without one (such as `sourceslicer/scalar_lm_best`) use the original four features.
- Inference runs on the GPU when one is available.
- `version.py` is the single source of the version, and is validated as a semantic version at import time.
- `python main --version` works without `--mode` and `--model_type`, and prints `SCALAR tagger <version>`.

### Added

- A pytest suite in `tests/`, run in CI.
- This changelog.

### Fixed

- Fold and holdout evaluation decode with the CRF, instead of taking the argmax of the emission scores.
- An identifier too long for the model input raises an error, instead of silently returning fewer tags than tokens.

## 2.2.0

Last release before semantic versioning was enforced.
