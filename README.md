# SCALAR Part-of-Speech Tagger for Identifiers

The Source Code Analysis and Lexical Annotation Runtime (**SCALAR**) is a part-of-speech tagger for source code identifiers. It uses a DistilBERT model with a CRF layer.

The legacy Gradient Boosting (tree-based) model was removed in 3.0.0. It was slower and less accurate than the DistilBERT model. Version 2.2.0 is the last release that includes it.

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

### CLI

```bash
python main --version                                # Print the SCALAR version
python main --mode run                               # Uses the model in serve.json, else local output/best_model if present, else the Hugging Face release
python main --mode run --local                       # Load the locally trained model from output/best_model
python main --mode run --pattern-postprocessing
python main --mode run --config_path serve.release.json
```

`--model_type lm_based` is still accepted, for compatibility with older scripts.

`--config_path` is honored in run mode, so server address, port, protocol, word list, and optional LM defaults can come from an alternate JSON file.

For LM inference, selected feature tokens are loaded from the saved model config automatically. Postprocessing can be controlled in three places:

- saved model default from training
- server startup override via `--pattern-postprocessing` or `--no-pattern-postprocessing`
- per-request override via `?pattern_postprocessing=true` or `?pattern_postprocessing=false`

## Release Path

The checked-in release-serving entry point is `serve.release.json`.

```bash
python main --mode run --config_path serve.release.json
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

You can retrain the model (default parameters are currently hardcoded):

```bash
python main --mode train
```

For LM training, you can choose which datasets to include:

```bash
python main --mode train
python main --mode train --no-synthetic-data
python main --mode train --no-tagger-data
python main --mode train --tagger-data --synthetic-data
python main --mode train --features context hungarian digit
python main --mode train --features context type type_overlap sys_sim
python main --mode train --model_dir release_models/lm_v1
```

When `--model_dir` is provided for LM training, the best saved checkpoint is written there. The same path can be reused with `python main --mode run --local --model_dir release_models/lm_v1`.

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
python main --mode train --pattern-postprocessing --features context
```

**Inference Performance:**
- Identifiers/sec: 486.81

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

See [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md) for how to tag ambiguous words such as `in`, `if`, `no` and single-letter prefixes.

---

## Docker Support

The image serves the DistilBERT model over HTTP on port 8080:

```bash
docker compose pull
docker compose up
```

---

## Notes

- **Kebab case** is not supported (e.g., `do-something-cool`).
- Feature and position tokens (e.g., `@pos_0`) are inserted automatically.
- The `dictionary` field in responses comes from the NLTK `words` corpus. It is downloaded on first run.
- Input must be parsed into identifier tokens. We recommend [srcML](https://www.srcml.org/) but any AST-based parser works.

---

## Versioning

SCALAR follows [Semantic Versioning](https://semver.org). The version lives in [version.py](version.py), and `python main --version` prints it. Changes are recorded in [CHANGELOG.md](CHANGELOG.md), and each release is tagged `vMAJOR.MINOR.PATCH` in git.

The version covers the software and its interfaces: the command line, the HTTP and stdio request/response formats, and the tagset.

- **MAJOR**: a breaking change to any of those interfaces, such as a removed option, a renamed response field, or a tag added to or removed from the tagset.
- **MINOR**: a backward-compatible addition, such as a new endpoint, a new optional response field, or a new default release model.
- **PATCH**: a bug fix that keeps every interface the same.

The model checkpoint is versioned separately, by its Hugging Face revision. A retrained model can change individual tags without changing the interface, so for reproducible results, record both the SCALAR version and the model revision.

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

You can find the most recent SCALAR training dataset [here](https://github.com/SCANL/scanl_tagger/blob/master/input/tagger_data_new.tsv). Each identifier links to its declaration in the original source (`CODE_URL`)

---

## More from SCANL

- [SCANL Website](https://www.scanl.org/)
- [Identifier Name Structure Catalogue](https://github.com/SCANL/identifier_name_structure_catalogue)

---

## Trouble?

Please [open an issue](https://github.com/SCANL/scanl_tagger/issues) if you encounter problems!
