# Changelog

All notable changes to SCALAR are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and SCALAR follows [Semantic Versioning](https://semver.org). See the "Versioning" section of the README for what each version bump means.

## 3.0.0-rc.1 - 2026-11-21

Release candidate for 3.0.0, tagged `v3.0.0-rc.1` for downstream review. Everything below is in it.

### Changed

- The CRF runs over identifier words only. Training and Viterbi decoding used to run over every inner token (feature tokens, `@pos_N` tokens, and continuation subwords), so the learned transitions never connected adjacent words. Decoding also rules out tag bigrams that never occur in the training data: `N N`, `N NPL`, `NPL NPL`, PRE after any tag other than PRE (PRE only starts an identifier), `PRE VM`, `DT D` and `CJ CJ`. Checkpoints trained before this change should be retrained.
- Continuation subwords (e.g. `##ployee`) are no longer labeled during training; only the first subword of each word carries its tag.
- The CRF transition parameters train with their own learning rate (`5e-3`), instead of the encoder's `5e-5`.
- The LM model trains on `input/tagger_data.tsv` (replaced with the corrected, cross-validated annotations) plus `input/synthetic_pos_data_full.csv`.
- LM training: 20% holdout, stratified 5-fold CV by context, low-frequency tag upsampling and verb-synonym augmentation (FUNCTION rows only) applied inside each fold, then a final retrain on the whole training split.
- `python main --mode train --seed N` sets the training seed (default 209). The holdout split and CV folds don't depend on it, so runs with different seeds are scored on the same holdout set.
- The holdout is split per data source (20% of each, stratified by context), and CV folds are stratified by source and context. The real holdout no longer changes when the synthetic file does. This changes which identifiers are held out, so scores from earlier runs aren't comparable.
- CV folds and the final retrain train a fixed 5 epochs, with no early stopping or best-epoch pick (folds nearly always peaked at their last epoch). Fold models predict their test slice out-of-fold, saved to `output/cv_predictions.csv`, so with the holdout every real identifier is scored once per run. The holdout report's "Real-Code Evaluation" section gives identifier accuracy, token accuracy and token macro-F1 on real code, per subset and combined, and a per-tag report.
- The headline "real" score in reports and the evaluation tool covers only `original_set`, `closed_category_set` and `headnoun_set`. Any other real subset added to `input/tagger_data.tsv` (listed in `NEW_REAL_SUBSETS`) is reported on its own and in a `real+new` row, so the headline stays comparable with earlier runs.
- `python main --mode train --folds K` sets the number of CV folds (default 5) for faster screening runs; the evaluation tool warns when compared runs used different fold counts.
- `python main --mode train --run-name NAME` also saves the report and predictions to `output/reports/NAME_seed<N>.*`, and `python -m scalar_tagger.lm_based_tagger.evaluation RUN_A RUN_B` compares saved runs over all their seeds with paired-bootstrap 95% intervals, rescoring real identifiers against the current gold.
- `python main --mode train --synthetic-data PATH` chooses the synthetic file, and `--no-synthetic-data` trains on real identifiers only. `--amplify real` limits low-frequency upsampling and verb augmentation to real rows, and `--amplify none` turns them off.
- Training drops synthetic rows that repeat a real identifier (43 in `synthetic_pos_data_full.csv`) or another synthetic row in the same context. The holdout report lists the synthetic file, its row count, and what was dropped.
- Class rows in `input/tagger_data.tsv` use `class` as their TYPE. 260 of them had the class's own name, which made `type_overlap` report that the whole name was its type.
- The holdout report shows token and identifier accuracy for each data source, since synthetic rows are much easier than real identifiers.
- `python main --mode train --features ...` chooses which feature tokens to train with, and `--no-features` trains without any. The default is all of them except the opt-in `lead_word`.
- New `lead_shape` feature: the first word's length, case, and separator (underscore or camelCase), which the uncased model can't otherwise see. A short first word followed by an underscore is a preamble far more often than one in camelCase.
- New `lead_word` feature: signals that the first word is a preamble, computed from the first word alone, with no list of known prefixes. It reports whether the word is a single letter, a number, letters mixed with digits, an English word (NLTK `words` plus WordNet, inflections included), or a non-dictionary term. It also gives the word's consonant/vowel ratio, whether it matches the system name, and whether a number follows it. It is opt-in (`--features ... lead_word`): it needs the NLTK `words` and `wordnet` corpora and didn't improve accuracy over `type context` in 3-seed screening.
- New feature tokens for the LM model: digit connectors, plural suffixes, declared type, type/name overlap, language, and system-name overlap. The feature list is saved in the checkpoint config, and inference uses it. Checkpoints without one (such as `sourceslicer/scalar_lm_best`) use the original four features.
- Inference runs on the GPU when one is available.
- The package moved from `src/` to `scalar_tagger/`, and the version from `version.py` to `scalar_tagger/version.py`. `python main` still works from a checkout; the evaluation tool is now `python -m scalar_tagger.lm_based_tagger.evaluation`.
- The release model is `sourceslicer/scalar_lm_v3` at commit `f01a24fe8533995c1246c494c6cb7478c3c74c2d` (the `tc_k5_rev` seed-42 checkpoint: 90.45% identifier accuracy on real code, 3-seed mean), with `confidence_temperature` 1.5. It replaces `sourceslicer/scalar_lm_best`, which is left unchanged for 2.x installs that load it unpinned.
- The installed command doesn't read `serve.json` or use `output/best_model` unless told to. `python main` still does both.
- The tagger warms up (splits and tags one identifier) before reporting ready, so the first request isn't slower than the rest. A pinned model that is already downloaded loads without contacting Hugging Face, so the tagger starts offline, and the hash that identifies a local checkpoint is cached in `~/.cache/scalar-tagger/revisions.json`.
- The legacy `GET /<name>/<context>` route goes through the shared batch backend. Its responses are unchanged.
- Loading a checkpoint builds the encoder from the saved config instead of downloading the base `distilbert-base-uncased` weights first.
- `version.py` is the single source of the version, and is validated as a semantic version at import time.
- `python main --version` works without `--mode`, and prints `SCALAR tagger <version>`.
- `--model_type` is optional and accepts only `lm_based`.
- The `dictionary` word list is bundled (`scalar_tagger/data/english_words.txt.gz`, the NLTK `words` corpus), so no NLTK data is downloaded to serve. Only the opt-in `lead_word` training feature still needs NLTK corpora.
- The Docker image installs the package and serves the release model over HTTP on port 8080, instead of starting training.
- Dependencies are updated to their latest releases, including `torch` 2.14.1, `transformers` 5.19.0, `datasets` 5.1.0, `pandas` 3.0.6, and `scikit-learn` 1.9.1. `transformers`, `accelerate`, `huggingface_hub`, `safetensors`, `numpy`, and `pandas` are now pinned directly.

### Removed

- The tree-based (Gradient Boosting) tagger: `src/tree_based_tagger/`, `models/model_GradientBoostingClassifier.pkl`, its training database `input/scanl_tagger_training_db_11_29_2024.db`, and the `gensim` dependency. It was slower (8.6 vs. about 360 identifiers per second) and less accurate than the DistilBERT+CRF model.
- `--model_type tree_based` and `--model_dir`, which only applied to the tree-based model.
- The unused `aiohttp` and `pipdeptree` dependencies.
- `setup.py` and the root `__init__.py`; `pyproject.toml` replaces them.
- `--input_path`, which was never used.

### Added

- A `pyproject.toml` package, `scalar-tagger`, with a `scalar-tagger` command (`scalar-tagger serve --stdio`, `scalar-tagger train ...`). Install it with `pip install "scalar-tagger @ git+https://github.com/SCANL/scanl_tagger.git@v3.0.0-rc.1"`. Training dependencies are in the `[train]` extra.
- `TaggingBackend.tag_batch(request)`: one batch-tagging function for every transport, following the JSON contract in `scalar_tagger/contract.py`. Each token is an object with `text`, `start`, `end` (character offsets into the original name), `tag`, and `dictionary`; callers can send their own `tokens` to skip splitting; every response carries a `model` block (`name`, `revision`, `features`, `postprocess`, `confidence_temperature`, `device`, `scalar_version`); and a bad identifier gets a coded error (`EMPTY_IDENTIFIER`, `NO_TOKENS`, `UNSUPPORTED_IDENTIFIER`, `INVALID_CONTEXT`, `INVALID_TOKENS`, `INVALID_IDENTIFIER`, `IDENTIFIER_TOO_LONG`, `INTERNAL_ERROR`) without failing the batch.
- Per-token confidence: with `options.confidence`, each token gets `p`, the probability of its tag from the CRF's forward-backward marginals (computed with the same forbidden transitions as Viterbi), and `alt`, the most probable other tag. A checkpoint's config can set `confidence_temperature` to soften overconfident probabilities without changing any tag.
- `--mode serve --stdio`: serves the batch contract as JSON lines over stdin/stdout. It prints a `ready` line once the model has loaded, sends all logging to stderr, answers `{"command": "info"}`, and exits when stdin closes.
- `POST /tag` and `GET /info` HTTP endpoints, using the same contract as stdio.
- `--mode serve`, the same as `--mode run`.
- `python -m scalar_tagger ...`, the same as the installed `scalar-tagger` command.
- `--revision` (and `"revision"` in the config file) to load a specific Hugging Face commit, branch, or tag, and a pinned release model (`RELEASE_MODEL` and `RELEASE_REVISION` in `scalar_tagger/cli.py`; see `RELEASE_MODEL.md`).
- `--device` (and `"device"` in the config file) to choose `cpu`, `cuda`, or `cuda:N`.
- `--model_dir` to serve a local checkpoint, or to choose where training saves it.
- `DistilBertTagger.tag_identifiers(rows, batch_size)` for batched inference: about 780 identifiers per second on a GPU and 90 on a CPU.
- The README documents what callers should send in `context`, `type`, and `system`.
- A pytest suite in `tests/`, run in CI.
- This changelog.

### Fixed

- `--address`, `--port`, `--protocol`, and `--word` apply when serving the DistilBERT model; they used to apply only to the tree-based model.
- `--config_path` is honored; the server always read `serve.json` before.
- Fold and holdout evaluation decode with the CRF, instead of taking the argmax of the emission scores.
- The `hungarian` feature looked for its prefix inside the first word *after* splitting, so it almost never fired (20 of 4,383 training rows). It now detects a single lowercase letter followed by a capitalized word (`f Matcher`, `b Force`), which is a preamble about 80% of the time in the training data. Checkpoints trained with `hungarian` before this fix should be retrained; checkpoints without a saved feature list (such as `sourceslicer/scalar_lm_best`) keep the original behavior.
- An identifier too long for the model input raises an error (`IDENTIFIER_TOO_LONG` in the batch contract), instead of silently returning fewer tags than tokens.
- `pyproject.toml` requires setuptools 77 or later, which the SPDX `license` string needs; older setuptools failed to build the package.

## 2.2.0

Last release before semantic versioning was enforced.
