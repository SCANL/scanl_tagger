# SCALAR Part-of-Speech Tagger for Identifiers

**SCALAR** is a part-of-speech tagger for source code identifiers. It supports two model types:

- **DistilBERT-based model with CRF layer** (Recommended: faster, more accurate)
- Legacy Gradient Boosting model (for compatibility)

---

## Installation

Make sure you have `python3.12` installed. Then:

```bash
git clone https://github.com/SCANL/scanl_tagger.git
cd scanl_tagger
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

You can run SCALAR in multiple ways:

### CLI (with DistilBERT or GradientBoosting model)

```bash
python main --mode run --model_type lm_based         # DistilBERT; prefers serve.release.json or local output/best_model when configured/present
python main --mode run --model_type lm_based --local # Load the locally trained model from output/best_model
python main --mode run --model_type lm_based --pattern-postprocessing
python main --mode run --model_type lm_based --config_path serve.release.json
python main --mode run --model_type tree_based       # Legacy model
```

`--config_path` is honored in run mode, so server address, port, protocol, word list, and optional LM defaults can come from an alternate JSON file.

For LM inference, selected feature tokens are loaded from the saved model config automatically. Postprocessing can be controlled in three places:

- saved model default from training
- server startup override via `--pattern-postprocessing` or `--no-pattern-postprocessing`
- per-request override via `?pattern_postprocessing=true` or `?pattern_postprocessing=false`

## Release Path

The checked-in release-serving entry point is `serve.release.json`.

```bash
python main --mode run --model_type lm_based --config_path serve.release.json
```

That config is intended to represent the exact release candidate setup. Today it points at the local retrained checkpoint in `output/best_model`; once the candidate is published to Hugging Face, update the `model` and `local` fields there so fresh clones resolve to the published release artifact.

Then query like:

```
http://127.0.0.1:8080/GetValue/FUNCTION
```

Supports context types:
- FUNCTION
- CLASS
- ATTRIBUTE
- DECLARATION
- PARAMETER

---

## Training

You can retrain either model (default parameters are currently hardcoded):

```bash
python main --mode train --model_type lm_based
python main --mode train --model_type tree_based
```

For LM training, you can choose which datasets to include:

```bash
python main --mode train --model_type lm_based
python main --mode train --model_type lm_based --no-synthetic-data
python main --mode train --model_type lm_based --no-tagger-data
python main --mode train --model_type lm_based --tagger-data --synthetic-data
python main --mode train --model_type lm_based --features context hungarian digit
python main --mode train --model_type lm_based --features context type type_overlap sys_sim
python main --mode train --model_type lm_based --model_dir release_models/lm_v1
```

When `--model_dir` is provided for LM training, the best saved checkpoint is written there. The same path can be reused with `python main --mode run --model_type lm_based --local --model_dir release_models/lm_v1`.

---

## Evaluation Results

### DistilBERT (LM-Based Model) — Recommended

| Metric                   | Score   |
|--------------------------|---------|
| **Macro F1**             | 0.9527  |
| **Token Accuracy**       | 0.9569  |
| **Identifier Accuracy**  | 0.9053  |

| Label | Precision | Recall | F1    | Support |
|-------|-----------|--------|-------|---------|
| CJ    | 0.88      | 0.88   | 0.88  | 8       |
| D     | 0.98      | 0.96   | 0.97  | 52      |
| DT    | 0.95      | 0.93   | 0.94  | 45      |
| N     | 0.94      | 0.94   | 0.94  | 418     |
| NM    | 0.91      | 0.93   | 0.92  | 440     |
| NPL   | 0.97      | 0.97   | 0.97  | 79      |
| P     | 0.94      | 0.92   | 0.93  | 79      |
| PRE   | 0.79      | 0.79   | 0.79  | 68      |
| V     | 0.89      | 0.84   | 0.86  | 110     |
| VM    | 0.79      | 0.85   | 0.81  | 13      |

Current release-candidate training command:

```bash
python main --mode train --model_type lm_based --pattern-postprocessing --features context
```

**Inference Performance:**
- Identifiers/sec: 486.81

---

### Gradient Boost Model (Legacy)

| Metric               | Score     |
|----------------------|-----------|
| Accuracy             | 0.8216    |
| Balanced Accuracy    | 0.9160    |
| Weighted Recall      | 0.8216    |
| Weighted Precision   | 0.8245    |
| Weighted F1          | 0.8220    |
| Inference Time       | 249.05s   |

**Inference Performance:**
- Identifiers/sec: 8.6

---

## Supported Tagset

| Tag   | Meaning                            | Examples                       |
|-------|------------------------------------|--------------------------------|
| N     | Noun                               | `user`, `Data`, `Array`        |
| DT    | Determiner                         | `this`, `that`, `those`        |
| CJ    | Conjunction                        | `and`, `or`, `but`             |
| P     | Preposition                        | `with`, `for`, `in`            |
| NPL   | Plural Noun                        | `elements`, `indices`          |
| NM    | Noun Modifier (adjective-like)     | `max`, `total`, `employee`     |
| V     | Verb                               | `get`, `set`, `delete`         |
| VM    | Verb Modifier (adverb-like)        | `quickly`, `deeply`            |
| D     | Digit                              | `1`, `2`, `10`, `0xAF`         |
| PRE   | Preamble / Prefix                  | `m`, `b`, `GL`, `p`            |

---

## Docker Support (Legacy only)

For the legacy server, you can also use Docker:

```bash
docker compose pull
docker compose up
```

---

## Notes

- **Kebab case** is not supported (e.g., `do-something-cool`).
- Feature and position tokens (e.g., `@pos_0`) are inserted automatically.
- Internally uses [WordNet](https://wordnet.princeton.edu/) for lexical features.
- Input must be parsed into identifier tokens. We recommend [srcML](https://www.srcml.org/) but any AST-based parser works.

---

## Citations

Please cite:

```
@inproceedings{newman2025scalar,
  author    = {Christian Newman and Brandon Scholten and Sophia Testa and others},
  title     = {SCALAR: A Part-of-speech Tagger for Identifiers},
  booktitle = {ICPC Tool Demonstrations Track},
  year      = {2025}
}

@article{newman2021ensemble,
  title={An Ensemble Approach for Annotating Source Code Identifiers with Part-of-speech Tags},
  author={Newman, Christian and Decker, Michael and AlSuhaibani, Reem and others},
  journal={IEEE Transactions on Software Engineering},
  year={2021},
  doi={10.1109/TSE.2021.3098242}
}
```

---

## Training Data

You can find the most recent SCALAR training dataset [here](https://github.com/SCANL/scanl_tagger/blob/master/input/tagger_data.tsv)

---

## More from SCANL

- [SCANL Website](https://www.scanl.org/)
- [Identifier Name Structure Catalogue](https://github.com/SCANL/identifier_name_structure_catalogue)

---

## Trouble?

Please [open an issue](https://github.com/SCANL/scanl_tagger/issues) if you encounter problems!
