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

`--device cpu`, `--device cuda`, or `--device cuda:N` picks where inference runs. The default, `auto`, uses the GPU when there is one. The config file can also set `"device"`.

`--config_path` is honored in run mode, so server address, port, protocol, word list, and optional LM defaults can come from an alternate JSON file.

For LM inference, selected feature tokens are loaded from the saved model config automatically. Postprocessing can be controlled in three places:

- saved model default from training
- server startup override via `--pattern-postprocessing` or `--no-pattern-postprocessing`
- per-request override via `?pattern_postprocessing=true` or `?pattern_postprocessing=false`

## Batch Tagging Contract

`TaggingBackend.tag_batch` tags a whole batch of identifiers with one JSON request. The stdio and HTTP transports will carry the same request and response.

```python
from src.tagging_backend import TaggingBackend

backend = TaggingBackend("output/best_model", local=True)
response = backend.tag_batch({
    "id": 1,
    "options": {"postprocess": True},
    "identifiers": [
        {"key": "a1", "name": "getUserToken", "context": "FUNCTION",
         "type": "string", "language": "C++", "system": "myproj", "tokens": None},
    ],
})
```

```json
{"id": 1,
 "model": {"name": "output/best_model", "revision": "sha256:922b805f818441bd",
           "features": ["context"], "postprocess": true, "device": "cuda", "scalar_version": "3.0.0"},
 "results": [
   {"key": "a1", "tokens": [
     {"text": "get",   "start": 0, "end": 3,  "tag": "V",  "dictionary": true},
     {"text": "User",  "start": 3, "end": 7,  "tag": "NM", "dictionary": true},
     {"text": "Token", "start": 7, "end": 12, "tag": "N",  "dictionary": true}]}]}
```

- `id` and `key` are echoed back unchanged. Results are always in request order.
- `start` and `end` are character offsets into `name`, found by matching each token in order, ignoring case. A caller-supplied token that isn't in the name gets `null` offsets.
- Set `tokens` to a list of strings to skip splitting, for example to keep `IPv4` whole.
- `options.postprocess` overrides the model's postprocessing default. `null` or omitted uses the default.
- `dictionary` is true when the word is in the NLTK English word list.
- `model.revision` is the Hugging Face commit for a hub model, or a hash of `model.safetensors` for a local directory. Record it, together with `scalar_version`, to make results reproducible.

An identifier that can't be tagged gets an error in place of `tokens`, and the rest of the batch is still tagged:

```json
{"key": "a2", "error": {"code": "UNSUPPORTED_IDENTIFIER", "message": "operator names are not tagged"}}
```

| Code | Meaning |
|------|---------|
| `EMPTY_IDENTIFIER` | The name is empty or only whitespace |
| `NO_TOKENS` | Splitting left no words, for example `___` |
| `UNSUPPORTED_IDENTIFIER` | An operator (`operator==`), destructor (`~Foo`), or qualified name (`ns::name`, `self.x`). Send the unqualified name instead |
| `INVALID_CONTEXT` | `context` isn't one of the five contexts below (case doesn't matter) |
| `INVALID_TOKENS` | `tokens` isn't `null` or a non-empty list of non-empty strings |
| `INVALID_IDENTIFIER` | `name`, `type`, `language`, or `system` isn't a string |
| `IDENTIFIER_TOO_LONG` | The name has too many words to fit in the model input |
| `INTERNAL_ERROR` | The model failed on this identifier |

A request that can't be read at all, such as one missing the `identifiers` list, gets a top-level `{"error": {"code": "INVALID_REQUEST", ...}}` instead of `results`. Per-token confidence (`options.confidence`) isn't supported yet. Requesting it adds a `CONFIDENCE_UNAVAILABLE` warning to the response.

Inference runs on the GPU when one is available. On the training data, the batch path tags about 360 identifiers per second on a GPU and about 25 per second on a CPU.

### Stdio transport

For a parent process that spawns the tagger, such as a CLI or an MCP server:

```bash
python main --mode serve --stdio
```

- Each line on stdin is one JSON message, and each line on stdout is one JSON response, in order. Blank lines are ignored.
- Once the model has loaded, the first line on stdout is `{"ready": true, "model": {...}}`. Wait for it before sending. If the model fails to load, the line is `{"ready": false, "error": {"code": "MODEL_LOAD_FAILED", ...}}` and the process exits with status 1.
- A message without a `command` is a tagging request. `{"id": 2, "command": "info"}` returns `{"id": 2, "model": {...}}` without tagging anything.
- A line that isn't valid JSON gets an `INVALID_JSON` error, and an unknown command gets `UNKNOWN_COMMAND`. The session continues either way.
- Only protocol messages go to stdout. All logging goes to stderr.
- The process exits with status 0 when stdin closes.

Time from launch to the ready line is about 9 seconds with the model already downloaded, on either a CPU or a GPU. About 7 seconds of that is importing torch and transformers. Allow at least 30 seconds the first time, when the model is downloaded from Hugging Face.

### HTTP transport

`python main --mode run` (or `--mode serve`) also serves the contract over HTTP:

- `POST /tag` takes the same JSON request body. It returns 200 when the request could be read, even if some identifiers have errors, and 400 with a top-level `error` when it couldn't.
- `GET /info` returns `{"id": null, "model": {...}}`.

The older `GET /<name>/<context>` route still works, but it puts type strings like `std::map<std::string, int>` into the URL. Prefer `POST /tag` for new clients.

---

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
