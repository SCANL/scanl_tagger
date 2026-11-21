# Release Model

This file records the model the installed `scalar-tagger` command serves, so the published artifact, the pin in the code, and the reported metrics stay aligned.

## Current release candidate

- Checkpoint: `output/best_model` from the `tc_k5_rev` run, seed 42 (`python main --mode train --features type context --folds 5 --seed 42 --run-name tc_k5_rev`).
- Local revision: `sha256:714800a14254839e` (hash of `model.safetensors`).
- Published as: `sourceslicer/scalar_lm_v3`, commit `d65d72568388965e3ce5a3bac8db7eafa475b457` (the same weights as `f01a24fe`, which 3.0.0-rc.1 pins, with the calibration map added to `config.json`). `RELEASE_MODEL` and `RELEASE_REVISION` in [scalar_tagger/cli.py](scalar_tagger/cli.py) point at it.
- Features: `context`, `type`.
- Confidence: temperature 1.5, then an isotonic calibration map (`confidence_calibration` in `config.json`). The map was fitted on the out-of-fold predictions of the `tc_k5_cal` run, which retrained this recipe with seed 42 and reproduced the checkpoint exactly (the same `model.safetensors` hash and the same out-of-fold and holdout predictions), on its 9,214 real tokens. Neither changes a tag.

## Metrics

| Evaluation | Identifier accuracy | Token macro-F1 |
|---|---|---|
| Real code (4,261 identifiers, 5-fold out-of-fold + holdout), mean of seeds 209, 7, 42 | 90.45% (±0.29) | 0.951 |
| This checkpoint's holdout (1,209 identifiers, real and synthetic) | 93.38% | 0.965 |
| Real identifiers in that holdout (853) | 92.4% | |
| 375 random identifiers from 15 unseen projects, reviewed by hand | 97.3% (C/C++/Java), 100% (C#) | |

Confidence on the real holdout (853 identifiers, 2,323 tokens, 78 wrong tags), which neither the model nor the calibration saw:

| Calibrated `p` | Share of tokens | Share correct | Mean `p` |
|---|---|---|---|
| Under 0.5 | 0.9% | 40.0% | 0.30 |
| 0.5 to 0.8 | 5.6% | 73.6% | 0.66 |
| 0.8 to 0.9 | 1.5% | 82.9% | 0.86 |
| 0.9 to 0.95 | 5.9% | 94.1% | 0.92 |
| 0.95 to 0.99 | 57.0% | 98.7% | 0.98 |
| 0.99 to 0.995 | 5.6% | 100.0% | 0.991 |
| 0.995 or higher | 23.6% | 99.8% | 0.997 |

Flagging tokens under 0.9 catches 52 of the 78 errors (7.9% of tokens flagged); under 0.95, 60 (13.8%). Before calibration (raw marginals at temperature 1.5, all 4,261 real identifiers out-of-fold plus holdout, 506 wrong tags), tokens between 0.95 and 0.99 were right 80.3% of the time and tokens between 0.9 and 0.95 67.7%. `python -m scalar_tagger.lm_based_tagger.evaluation tc_k5_cal` prints both tables.

Throughput: about 780 identifiers per second on a GPU and 90 on a CPU (batch path, real identifiers).

## Publishing a new release model

The installed command loads `RELEASE_MODEL` at the exact commit in `RELEASE_REVISION`, so pushing to the Hugging Face repo doesn't change what users run until the pin changes. Publish to a repo that older SCALAR versions don't load unpinned (2.x loads `sourceslicer/scalar_lm_best` at its latest commit), so a new checkpoint can't break them.

1. Upload the checkpoint: `hf upload <repo> output/best_model . --exclude training_args.bin --commit-message "<SCALAR version, metrics>"`. The tagger needs `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`; `README.md` is the model card.
2. Copy the commit hash it prints (or find it in the repo's history).
3. Set `RELEASE_MODEL` and `RELEASE_REVISION` in `scalar_tagger/cli.py`, and update this file and the README's metrics.
4. Record the model and revision in `CHANGELOG.md`. A new release model is a MINOR version change.

Check the result with `scalar-tagger serve --stdio`: its `ready` line reports `model.name` and `model.revision`.
