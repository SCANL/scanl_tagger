# Release Model

This file records the model the installed `scalar-tagger` command serves, so the published artifact, the pin in the code, and the reported metrics stay aligned.

## Current release candidate

- Checkpoint: `output/best_model` from the `tc_k5_rev` run, seed 42 (`python main --mode train --features type context --folds 5 --seed 42 --run-name tc_k5_rev`).
- Local revision: `sha256:714800a14254839e` (hash of `model.safetensors`).
- Published as: `sourceslicer/scalar_lm_v3`, commit `f01a24fe8533995c1246c494c6cb7478c3c74c2d`. `RELEASE_MODEL` in [scalar_tagger/cli.py](scalar_tagger/cli.py) points at it; `RELEASE_REVISION` is set once the upload's commit hash is known.
- Features: `context`, `type`. Confidence temperature: 1.5, chosen on the real holdout (cross-fit on its two halves, log-loss of the true tags 23–30% lower than at 1.0). It changes `p` and `alt`, never the tags.

## Metrics

| Evaluation | Identifier accuracy | Token macro-F1 |
|---|---|---|
| Real code (4,261 identifiers, 5-fold out-of-fold + holdout), mean of seeds 209, 7, 42 | 90.45% (±0.29) | 0.951 |
| This checkpoint's holdout (1,209 identifiers, real and synthetic) | 93.38% | 0.965 |
| Real identifiers in that holdout (853) | 92.4% | |
| 375 random identifiers from 15 unseen projects, reviewed by hand | 97.3% (C/C++/Java), 100% (C#) | |

Throughput: about 780 identifiers per second on a GPU and 90 on a CPU (batch path, real identifiers).

## Publishing a new release model

The installed command loads `RELEASE_MODEL` at the exact commit in `RELEASE_REVISION`, so pushing to the Hugging Face repo doesn't change what users run until the pin changes. Publish to a repo that older SCALAR versions don't load unpinned (2.x loads `sourceslicer/scalar_lm_best` at its latest commit), so a new checkpoint can't break them.

1. Upload the checkpoint: `hf upload <repo> output/best_model . --exclude training_args.bin --commit-message "<SCALAR version, metrics>"`. The tagger needs `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`; `README.md` is the model card.
2. Copy the commit hash it prints (or find it in the repo's history).
3. Set `RELEASE_MODEL` and `RELEASE_REVISION` in `scalar_tagger/cli.py`, and update this file and the README's metrics.
4. Record the model and revision in `CHANGELOG.md`. A new release model is a MINOR version change.

Check the result with `scalar-tagger serve --stdio`: its `ready` line reports `model.name` and `model.revision`.
