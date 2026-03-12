# LM Release Candidate

This file records the current LM release candidate so the published artifact, local fallback path, and reported metrics stay aligned.

## Candidate Summary

- Serving config: `serve.release.json`
- Default local model directory: `output/best_model`
- Serving command: `python main --mode run --model_type lm_based --config_path serve.release.json`
- Training command: `python main --mode train --model_type lm_based --pattern-postprocessing --features context`

## Holdout Metrics

- Macro F1: `0.9527`
- Token accuracy: `0.9569`
- Identifier accuracy: `0.9053`
- Inference throughput: `486.81` identifiers/sec

## Publication Notes

- While the candidate is still under verification, keep `serve.release.json` pointing to `output/best_model`.
- After publishing the candidate to a separate Hugging Face repo, update `serve.release.json` so `model` is the repo id and `local` is `false`.
- Once the candidate is verified, either retarget the main production repo or copy the tested checkpoint into the production release repo.