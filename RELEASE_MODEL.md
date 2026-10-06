# LM Release Candidate

This file records the current LM release candidate so the published artifact, local fallback path, and reported metrics stay aligned.

## Candidate Summary

- Serving config: `serve.release.json`
- Default local model directory: `output/best_model`
- Serving command: `python main --mode run --config_path serve.release.json`
- Training command: `python main --mode train --pattern-postprocessing --features context`
- Pinned release model: `RELEASE_MODEL` and `RELEASE_REVISION` in [scalar_tagger/cli.py](scalar_tagger/cli.py), currently `sourceslicer/scalar_lm_release_test` at `41a3c953a6ec5612834a16b0da54d809531af2ee`

## Holdout Metrics

- Macro F1: `0.9527`
- Token accuracy: `0.9569`
- Identifier accuracy: `0.9053`
- Inference throughput: `486.81` identifiers/sec

## Publication Notes

- While the candidate is still under verification, keep `serve.release.json` pointing to `output/best_model`.
- After publishing the candidate to a separate Hugging Face repo, update `serve.release.json` so `model` is the repo id and `local` is `false`.
- Once the candidate is verified, either retarget the main production repo or copy the tested checkpoint into the production release repo.

## Publishing a New Release Model

The installed `scalar-tagger` command loads `RELEASE_MODEL` at the exact commit in `RELEASE_REVISION`, so pushing to the Hugging Face repo doesn't change what users run until the pin changes.

1. Upload the checkpoint: `hf upload <repo> output/best_model . --commit-message "<SCALAR version, metrics>"`.
2. Copy the commit hash it prints (or find it on the repo's "Files" tab history).
3. Set `RELEASE_MODEL` and `RELEASE_REVISION` in `scalar_tagger/cli.py`, and update the metrics above and in the README.
4. Release a new MINOR version of SCALAR, and record the revision in `CHANGELOG.md`.

Check the result with `scalar-tagger serve --stdio`: its `ready` line reports `model.name` and `model.revision`.