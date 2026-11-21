# SCALAR Part-of-Speech Tagger for Identifiers

**SCALAR** is a part-of-speech tagger for source code identifiers. It uses a DistilBERT model with a CRF layer.

---

## Installation

SCALAR needs Python 3.12 or later.

To use the tagger, install the package. This gives you the `scalar-tagger` command:

```bash
pip install "scalar-tagger @ git+https://github.com/SCANL/scanl_tagger.git@v3.0.0"
scalar-tagger --version
```

To train models or work on SCALAR itself, clone the repository and install the pinned environment:

```bash
git clone https://github.com/SCANL/scanl_tagger.git
cd scanl_tagger
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The installed package has only what tagging needs. Training also needs `pip install "scalar-tagger[train]"`, which `requirements.txt` already includes.

### Which model runs

The installed `scalar-tagger` command uses the release model, pinned to an exact Hugging Face commit (`RELEASE_MODEL` and `RELEASE_REVISION` in [scalar_tagger/cli.py](scalar_tagger/cli.py); see [RELEASE_MODEL.md](RELEASE_MODEL.md)). It is downloaded on first use. After that, the tagger starts from the local copy without contacting Hugging Face, so it also works offline. `--revision` picks a different commit, branch, or tag, and `--model_dir` loads a local checkpoint.

Running `python main` from a checkout behaves the same, except that it also reads `serve.json` and uses `output/best_model` if that directory exists.

---

## Usage

### CLI

```bash
scalar-tagger --version                              # Print the SCALAR version
scalar-tagger serve --stdio                          # JSON lines over stdin/stdout (see "Stdio transport")
scalar-tagger serve --port 8080                      # HTTP server
scalar-tagger serve --stdio --model_dir path/to/checkpoint
scalar-tagger train --features type context          # Needs the [train] extra and the input/ data

python main --mode run                               # From a checkout: serve.json's model, else output/best_model if present, else the release model
python main --mode run --local                       # Serve the locally trained model in output/best_model
```

`scalar-tagger serve` is the same as `scalar-tagger --mode serve`, and `run` is the same as `serve`. `--model_type lm_based` is still accepted, for compatibility with older scripts.

`--device cpu`, `--device cuda`, or `--device cuda:N` picks where inference runs. The default, `auto`, uses the GPU when there is one. The config file can also set `"device"`. `--address`, `--port`, `--protocol`, and `--word` override the values in the config file (`serve.json` by default, or `--config_path`).

The feature tokens a checkpoint was trained with are saved in its config and used automatically when serving.

---

## Batch Tagging Contract

`TaggingBackend.tag_batch` tags a whole batch of identifiers with one JSON request. The stdio and HTTP transports carry the same request and response.

```python
from scalar_tagger.tagging_backend import TaggingBackend

backend = TaggingBackend("output/best_model", local=True)
response = backend.tag_batch({
    "id": 1,
    "options": {"confidence": True},
    "identifiers": [
        {"key": "a1", "name": "getUserToken", "context": "FUNCTION",
         "type": "string", "language": "C++", "system": "myproj", "tokens": None},
        {"key": "a2", "name": "openFile", "context": "DECLARATION", "type": "bool"},
    ],
})
```

```json
{"id": 1,
 "model": {"name": "output/best_model", "revision": "sha256:714800a14254839e",
           "features": ["context", "type"], "postprocess": false, "confidence_temperature": 1.5,
           "device": "cuda", "scalar_version": "3.0.0"},
 "results": [
   {"key": "a1", "tokens": [
     {"text": "get",   "start": 0, "end": 3,  "tag": "V",  "p": 0.9977, "alt": {"tag": "DT", "p": 0.0004}, "dictionary": true},
     {"text": "User",  "start": 3, "end": 7,  "tag": "NM", "p": 0.9979, "alt": {"tag": "P",  "p": 0.0004}, "dictionary": true},
     {"text": "Token", "start": 7, "end": 12, "tag": "N",  "p": 0.9986, "alt": {"tag": "NM", "p": 0.0003}, "dictionary": true}]},
   {"key": "a2", "tokens": [
     {"text": "open",  "start": 0, "end": 4,  "tag": "V",  "p": 0.966,  "alt": {"tag": "NM", "p": 0.0298}, "dictionary": true},
     {"text": "File",  "start": 4, "end": 8,  "tag": "N",  "p": 0.9973, "alt": {"tag": "NM", "p": 0.0005}, "dictionary": true}]}]}
```

- `id` and `key` are echoed back unchanged. Results are always in request order.
- `start` and `end` are character offsets into `name`, found by matching each token in order, ignoring case. A caller-supplied token that isn't in the name gets `null` offsets.
- Set `tokens` to a list of strings to skip splitting, for example to keep `IPv4` whole.
- `dictionary` is true when the word is in the bundled English word list (the NLTK `words` corpus; see [scalar_tagger/data/README.md](scalar_tagger/data/README.md)), plus `words/en.txt` if that file exists.
- `model.revision` is the Hugging Face commit for a hub model, or a hash of `model.safetensors` for a local directory. Record it, together with `scalar_version`, to make results reproducible.
- `options.postprocess` is accepted for compatibility but ignored, and `model.postprocess` is always `false`: 3.0 has no tag postprocessing. The CRF's transition constraints replaced it.

### Per-token confidence

With `"options": {"confidence": true}`, each token also gets:

- `p`: the probability of the returned tag, from the CRF's forward-backward marginals. The marginals use the same forbidden transitions as Viterbi decoding, so `p` describes the tag actually returned.
- `alt`: the most probable other tag and its probability.

Viterbi returns the best *sequence*, so in rare cases `alt.p` can exceed `p`. Confidence costs about 20% throughput on a GPU and nothing measurable on a CPU. Without the option, `p` and `alt` are omitted.

How to read `p`, from the release model on the 853 real holdout identifiers it never trained on:

- Most tokens are confident. 91% get `p` ≥ 0.99, and 98.9% of those are right.
- Below that, `p` roughly tracks accuracy. Tokens with `p` < 0.95 are right 62% of the time, and tokens between 0.8 and 0.9 are right 68% of the time.
- At the identifier level, flagging identifiers whose lowest `p` is below 0.9 marks 6.4% of them and catches 43% of the mistakes (28 of 65). Many mistakes are still made confidently, so a high `p` is not a guarantee.

A checkpoint can set `confidence_temperature` in its config to soften overconfident probabilities. Every score is divided by it before the marginals are computed, which never changes the returned tags. The value in use is reported in `model.confidence_temperature`. The release model uses 1.5. Without it (raw marginals, 1.0), 97% of tokens get `p` ≥ 0.99, including most of the errors. The value was chosen on the holdout: fitting it on one half and checking it on the other lowered the log-loss of the true tags by 23–30%.

### Errors

An identifier that can't be tagged gets an error in place of `tokens`, and the rest of the batch is still tagged:

```json
{"key": "a2", "error": {"code": "UNSUPPORTED_IDENTIFIER", "message": "operator names are not tagged"}}
```

| Code | Meaning |
|------|---------|
| `EMPTY_IDENTIFIER` | The name is empty or only whitespace |
| `NO_TOKENS` | Splitting left no words, for example `___` |
| `UNSUPPORTED_IDENTIFIER` | An operator (`operator==`), destructor (`~Foo`), or qualified name (`ns::name`, `self.x`). Send the unqualified name instead |
| `INVALID_CONTEXT` | `context` isn't one of the five contexts (case doesn't matter) |
| `INVALID_TOKENS` | `tokens` isn't `null` or a non-empty list of non-empty strings |
| `INVALID_IDENTIFIER` | `name`, `type`, `language`, or `system` isn't a string |
| `IDENTIFIER_TOO_LONG` | The name has too many words to fit in the model input |
| `INTERNAL_ERROR` | The model failed on this identifier |

A request that can't be read at all, such as one missing the `identifiers` list, gets a top-level `{"error": {"code": "INVALID_REQUEST", ...}}` instead of `results`.

On real identifiers, the batch path tags about 780 identifiers per second on a GPU and about 90 per second on a CPU.

### What to send in `context`, `type`, and `system`

The model learned from how these fields are filled in the training data, so send them the same way.

**`context`** is one of `FUNCTION`, `CLASS`, `ATTRIBUTE`, `DECLARATION`, `PARAMETER`. It describes what the identifier names, not only where it is declared:

| Construct | Context |
|---|---|
| Function or method, including a prototype or declaration | `FUNCTION` |
| Variable, field, or parameter initialized with a lambda or method reference | `FUNCTION` |
| Variable, field, or parameter of a function-pointer, callback, or functional-interface type (`std::function`, `Runnable`, `Comparator`, `*_cb`) | `FUNCTION` |
| Class, struct, interface, enum, or union | `CLASS` |
| Namespace | `CLASS` |
| Field or property of a class, including enum constants | `ATTRIBUTE` |
| Local or global variable | `DECLARATION` |
| Function parameter or template parameter | `PARAMETER` |
| Constructor or destructor | Skip (the name is the class name) |
| Macro | Skip for now |

**`type`** is the declared type, exactly as the parser gives it. Don't normalize it. The `type` feature reads pointers (`*`), references (`&`), arrays, templates (`<...>`), `const`, `unsigned`, and `signed` from the raw string, and buckets the base type (bool, int, float, string, container, function, object). The training data mixes bare names (`int`, `List`) with full types (`std::vector<Foo>`, `HostEntry &`; about 450 rows), so either form works.

| Case | Value to send |
|---|---|
| Classes | `class` |
| Functions | The return type (`void` for none) |
| Integer flags used as booleans | The declared type (`int`), not `bool` |
| Unknown type (common in Python) | The empty string, which the model sees as an unknown type |

**`system`** is the project name, such as `drill` or `gimp`, not the containing class. The `sys_sim` feature uses it to recognize project prefixes (a `gimp` prefix in GIMP). Default it to the repository's directory name. The release model doesn't use `sys_sim`, but other checkpoints may.

`language` is the source language (`C`, `C++`, `Java`, `C#`, ...). The release model doesn't use it either.

### Stdio transport

For a parent process that spawns the tagger, such as a CLI or an MCP server:

```bash
scalar-tagger serve --stdio
```

- Each line on stdin is one JSON message, and each line on stdout is one JSON response, in order. Blank lines are ignored.
- Once the model has loaded, the first line on stdout is `{"ready": true, "model": {...}}`. Wait for it before sending. If the model fails to load, the line is `{"ready": false, "error": {"code": "MODEL_LOAD_FAILED", ...}}` and the process exits with status 1.
- A message without a `command` is a tagging request. `{"id": 2, "command": "info"}` returns `{"id": 2, "model": {...}}` without tagging anything.
- A line that isn't valid JSON gets an `INVALID_JSON` error, and an unknown command gets `UNKNOWN_COMMAND`. The session continues either way.
- Only protocol messages go to stdout. All logging goes to stderr.
- The process exits with status 0 when stdin closes.

Time from launch to the ready line is about 11 seconds on our development machine (WSL2), on either a CPU or a GPU, once the model is downloaded. About 6.5 seconds of that is importing torch and transformers. The tagger has already split and tagged one identifier when it reports ready, so the first request isn't slower than the rest. Allow at least 60 seconds the first time, when the model is downloaded from Hugging Face. `scalar-tagger --version` returns immediately, so it is a cheap way to check that the command is installed.

### HTTP transport

`scalar-tagger serve` (or `python main --mode run`) also serves the contract over HTTP:

- `POST /tag` takes the same JSON request body. It returns 200 when the request could be read, even if some identifiers have errors, and 400 with a top-level `error` when it couldn't.
- `GET /info` returns `{"id": null, "model": {...}}`.

The older `GET /<name>/<context>` route still works (for example `http://127.0.0.1:8080/GetValue/FUNCTION?type=int`), but it puts type strings like `std::map<std::string, int>` into the URL. Prefer `POST /tag` for new clients.

---

## Training

Default parameters are currently hardcoded:

```bash
python main --mode train                                  # every feature (the default)
python main --mode train --features context type language # only the listed features
python main --mode train --no-features                    # no feature tokens at all
python main --mode train --seed 7                         # a different training seed (default 209)
python main --mode train --synthetic-data input/other.csv # a different synthetic file
python main --mode train --no-synthetic-data              # real identifiers only
python main --mode train --amplify real                   # upsample/augment real identifiers only
python main --mode train --amplify none                   # no upsampling or verb augmentation
```

### Comparing configurations

The metric that matters is **identifier-level accuracy on real code**: an identifier counts only if every one of its tags is right. **Token macro-F1 on real code** is the guardrail: it weighs each tag equally, so it falls when rare tags (CJ, VM, PRE, DT) are sacrificed, which barely moves identifier accuracy. A change is an improvement when it raises real identifier accuracy without a significant drop in real macro-F1.

Every real identifier is scored exactly once per run. The 5 CV folds train a fixed number of epochs and predict the training portion out-of-fold (`output/cv_predictions.csv`); the final model, retrained on the whole training portion, predicts the 20% holdout (`output/holdout_predictions.csv`). "Real-Code Evaluation" in `holdout_report.txt` combines the two, per subset and overall, with a per-tag report below it. That sample is about five times the holdout alone. The holdout split and CV folds are fixed by `SPLIT_SEED`, and each data source is split separately, so the real identifiers land in the same place whichever seed or synthetic file is used.

One run still varies with the seed (`--seed` changes weight initialization, batch order, augmentation and dropout), so train each configuration with several seeds and save each run under a name:

```bash
for seed in 209 7 42; do
  python main --mode train --features context type --seed $seed --run-name tc
done
```

`--run-name tc` copies the report and both prediction files to `output/reports/tc_seed<N>.*`. Then compare two runs:

```bash
python -m scalar_tagger.lm_based_tagger.evaluation tc all_features   # B - A, per subset, with 95% intervals
python -m scalar_tagger.lm_based_tagger.evaluation tc                 # one run's metrics
```

The comparison averages over every saved seed, rescores real identifiers against the current `input/tagger_data.tsv` (so label fixes apply to old runs too), uses only identifiers scored in every seed of both runs, and gives paired-bootstrap 95% intervals for the differences in identifier accuracy and macro-F1, plus per-tag F1. `--split cv` or `--split holdout` restricts it to one kind of prediction; runs saved before CV predictions were kept are compared on the holdout.

For quicker screening, `--folds 3` trains with 3 CV folds instead of 5 (about 40% less training). Each fold then sees less data, so out-of-fold accuracy comes out lower: compare a 3-fold run only with other 3-fold runs (the comparison tool warns when fold counts differ), and confirm a promising result with 5 folds. `input/tagger_data.tsv` records each row's subset in its `DATA_SOURCE` column (`original_set`, `closed_category_set`, `headnoun_set`).

### Feature tokens

Each identifier is prefixed with feature tokens that describe it. `--features` chooses which ones the model trains with, and `--no-features` turns them all off; **by default, all eleven are on**. The chosen list is saved in the checkpoint's config, and the tagger uses the same list when serving, so nothing needs to be set at run time. A checkpoint without a saved list (such as `sourceslicer/scalar_lm_best`) uses `context`, `hungarian`, `cvr`, and `digit`.

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
| `lead_shape` | The first word's length (1, 2–3, 4+ characters), case, and whether an underscore or a camelCase break follows it. The model is uncased, so this is how it sees `m_count` vs. `xOffset`; the separator is read from the next word's case | name |

The model trains on `input/tagger_data.tsv` and a synthetic file, `input/synthetic_pos_data_full.csv` by default. Synthetic rows that repeat a real identifier (compared lowercased) are dropped, as are synthetic rows that repeat another synthetic row in the same context; the holdout report lists the file and how many rows were dropped. By default, low-frequency tags (V, PRE, VM, CJ) are upsampled and FUNCTION verbs are augmented with synonyms in every row; `--amplify real` does this for real rows only, and `--amplify none` turns it off. The CV folds and the final retrain all train for `EPOCHS` (5) epochs. Training writes the checkpoint to `output/best_model`, out-of-fold predictions to `output/cv_predictions.csv`, holdout predictions to `output/holdout_predictions.csv`, and metrics to `holdout_report.txt`. Serve a locally trained checkpoint with `python main --mode run --local`.

---

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

The tests build a tiny randomly initialized DistilBERT, so they don't download a model and run on CPU in under a minute.

---

## Evaluation Results

SCALAR 3.0 is evaluated three ways. The headline is the first.

| Evaluation | Identifier accuracy | Token macro-F1 |
|---|---|---|
| **Real code**, every one of the 4,261 real identifiers scored once (5-fold out-of-fold + holdout), mean of 3 training seeds | **90.45%** (±0.29) | 0.951 |
| Holdout set of the release checkpoint (1,209 identifiers, real and synthetic) | 93.38% | 0.965 |
| Real identifiers in that holdout (853) | 92.4% | |
| 375 random identifiers from 15 open-source projects that aren't in the training data, reviewed by hand | 97.3% (C, C++, Java; 300) and 100% (C#; 75) | |

The unseen-project sample is random, so it holds many easy names, and the reviewer saw the model's tags while reviewing. Read it as a check that accuracy holds up on new code, not as a better estimate than the first row.

Per-tag F1 on real code (mean of 3 seeds):

| Tag | Support | F1 |
|-----|---------|------|
| CJ  | 75   | 0.948 |
| D   | 335  | 0.989 |
| DT  | 406  | 0.978 |
| N   | 3720 | 0.969 |
| NM  | 3845 | 0.956 |
| NPL | 636  | 0.980 |
| P   | 606  | 0.956 |
| PRE | 732  | 0.924 |
| V   | 1080 | 0.922 |
| VM  | 102  | 0.885 |

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

The Docker image serves the release model over HTTP on port 8080:

```bash
docker compose pull
docker compose up
```

---

## Notes

- **Kebab case** is not supported (e.g., `do-something-cool`).
- Feature and position tokens (e.g., `@pos_0`) are inserted automatically.
- Input must be parsed into identifier tokens. We recommend [srcML](https://www.srcml.org/) but any AST-based parser works.

---

## Versioning

SCALAR follows [Semantic Versioning](https://semver.org). The version lives in [scalar_tagger/version.py](scalar_tagger/version.py), and `python main --version` prints it. Changes are recorded in [CHANGELOG.md](CHANGELOG.md), and each release is tagged `vMAJOR.MINOR.PATCH` in git.

The version covers the software and its interfaces: the command line, the batch contract (stdio and HTTP), and the tagset.

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
