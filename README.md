# SCALAR Part-of-Speech Tagger for Identifiers

**SCALAR** is a part-of-speech tagger for source code identifiers. It uses a DistilBERT model with a CRF layer.

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
python main --mode run           # Serve the published model (sourceslicer/scalar_lm_best)
python main --mode run --local   # Serve a locally trained model from output/best_model
```

`--address`, `--port`, `--protocol`, and `--word` override the values in `serve.json`.

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

Default parameters are currently hardcoded:

```bash
python main --mode train                                  # every feature (the default)
python main --mode train --features context type language # only the listed features
python main --mode train --no-features                    # no feature tokens at all
python main --mode train --seed 7                         # a different training seed (default 209)
```

### Comparing configurations

One training run varies by about ±1 point of identifier accuracy on real code from the seed alone, so compare configurations over several seeds. The holdout split and CV folds are fixed, so every run is scored on the same holdout set; `--seed` changes only weight initialization, batch order, augmentation and dropout. Each run overwrites `holdout_report.txt`, so copy it aside:

```bash
mkdir -p output/reports
for seed in 209 7 42; do
  python main --mode train --features context type language --seed $seed
  cp holdout_report.txt "output/reports/ctl_seed$seed.txt"
done
```

Compare the `tagger_data` line under "Held-Out Accuracy by Data Source"; synthetic identifiers are much easier and inflate the combined score.

### Feature tokens

Each identifier is prefixed with feature tokens that describe it. `--features` chooses which ones the model trains with, and `--no-features` turns them all off; **by default, all ten are on**. The chosen list is saved in the checkpoint's config, and the tagger uses the same list when serving, so nothing needs to be set at run time. A checkpoint without a saved list (such as `sourceslicer/scalar_lm_best`) uses `context`, `hungarian`, `cvr`, and `digit`.

| Feature | What it encodes | Input it reads |
|---------|-----------------|----------------|
| `context` | Where the identifier is declared (`@func`, `@param`, `@attr`, `@decl`, `@class`) | context |
| `hungarian` | A single lowercase letter followed by a capitalized word (`fMatcher`, `bForce16bpp`). Underscore prefixes like `m_value` aren't detected, because splitting drops the underscore | name |
| `cvr` | Average consonant/vowel ratio of the words (low, mid, high) | name |
| `digit` | Whether any word contains a digit | name |
| `digit_connector` | A `2` used as a connector (`to`), and where it sits (head, middle, tail) | name |
| `plural_suffix` | Plural-looking suffixes (`s`, `es`, `ies`) and where they sit | name |
| `type` | Bucketed declared type (bool, int, float, string, container, …) plus pointer, reference, array, const | type |
| `type_overlap` | Whether the name's words repeat the type's words (e.g. `userList` of type `UserList`) | name, type |
| `language` | Programming language | language |
| `sys_sim` | Whether the name's words repeat the system (project) name, e.g. a `gimp` prefix in GIMP | name, system |

The model trains on `input/tagger_data.tsv` and `input/synthetic_pos_data_full.csv`. It writes the checkpoint to `output/best_model`, per-identifier holdout predictions to `output/holdout_predictions.csv`, and metrics to `holdout_report.txt`. Serve a locally trained checkpoint with `python main --mode run --local`.

---

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

The tests build a tiny randomly initialized DistilBERT, so they don't download a model and run on CPU in under a minute.

---

## Evaluation Results

### DistilBERT + CRF

| Metric                   | Score   |
|--------------------------|---------|
| **Macro F1**             | 0.9032  |
| **Token Accuracy**       | 0.9223  |
| **Identifier Accuracy**  | 0.8291  |

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

**Inference Performance:**
- Identifiers/sec: 225.8

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

## Docker Support

The Docker image serves the DistilBERT model over HTTP on port 8080:

```bash
docker compose pull
docker compose up
```

---

## Notes

- **Kebab case** is not supported (e.g., `do-something-cool`).
- Feature and position tokens (e.g., `@pos_0`) are inserted automatically.
- The `dictionary` flag in each response uses the NLTK words corpus, plus `words/en.txt` if that file exists.
- Input must be parsed into identifier tokens. We recommend [srcML](https://www.srcml.org/) but any AST-based parser works.

---

## Versioning

SCALAR follows [Semantic Versioning](https://semver.org). The version lives in [version.py](version.py), and `python main --version` prints it. Changes are recorded in [CHANGELOG.md](CHANGELOG.md), and each release is tagged `vMAJOR.MINOR.PATCH` in git.

The version covers the software and its interfaces: the command line, the HTTP request/response format, and the tagset.

- **MAJOR**: a breaking change to any of those interfaces, such as a removed option, a renamed response field, or a tag added to or removed from the tagset.
- **MINOR**: a backward-compatible addition, such as a new endpoint, a new optional response field, or a new default model.
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

You can find the most recent SCALAR training dataset [here](https://github.com/SCANL/scanl_tagger/blob/master/input/tagger_data.tsv)

---

## More from SCANL

- [SCANL Website](https://www.scanl.org/)
- [Identifier Name Structure Catalogue](https://github.com/SCANL/identifier_name_structure_catalogue)

---

## Trouble?

Please [open an issue](https://github.com/SCANL/scanl_tagger/issues) if you encounter problems!
