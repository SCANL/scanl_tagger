# Tool Integration Wish List

Requests from the identifier-naming analysis tool: a C++ core (srcML fact extraction, a grammar-pattern parser, and a rule engine) with CLI, MCP, and later LSP front ends. The tool calls SCALAR to tag identifiers. These changes are ordered by when the tool needs them.

The tool will call the tagger through a small client interface with two transports:

- **stdio** (default): the CLI or MCP server spawns the tagger as a child process and exchanges JSON lines. No ports. The tagger's lifetime is tied to its parent. The CLI starts it lazily, only when its local cache misses.
- **HTTP** (later): a long-running daemon for the LSP or a shared lab machine, over TCP or a Unix socket.

Both transports carry the same JSON contract (item 4).

---

## Milestone 1: needed for the first CLI

### 1. A single batch-tagging function

> **Status (3.x):** Done: `TaggingBackend.tag_batch(request)` in `scalar_tagger/tagging_backend.py`.

One `tag_batch(requests) -> results` function that every transport calls. `TaggingBackend.tag_identifier_batch` already does most of this. The tool tags a whole file or project at a time, so tagging one identifier per request will be a bottleneck.

### 2. A stdio serve mode

> **Status (3.x):** Done: `python main --mode serve --stdio`. Model load failures send `{"ready": false, ...}` and exit with status 1.

```bash
scalar-tagger serve --stdio        # or: python main --mode serve --stdio
```

- Read one JSON request per line on stdin, and write one JSON response per line on stdout.
- Send **all** logging to stderr. Anything else on stdout corrupts the protocol.
- Print one line, `{"ready": true, "model": {...}}`, once the model has loaded, so the client knows when to start sending.
- Exit cleanly when stdin closes.

### 3. A `POST /tag` HTTP endpoint

> **Status (3.x):** Done. `GET /info` is also available, and the GET route remains.

It takes the same JSON body as stdio. The current `GET /<name>/<context>?type=...` puts type strings like `std::map<std::string, int>` into the URL, which is fragile. The GET route can stay for backward compatibility.

### 4. A structured request/response contract

> **Status (3.x):** Done, in `scalar_tagger/contract.py`. `p` and `alt` are not filled in yet (item 9); requesting them adds a `CONFIDENCE_UNAVAILABLE` warning.

Request:

```json
{"id": 1,
 "options": {"postprocess": true, "confidence": true},
 "identifiers": [
   {"key": "a1", "name": "getUserToken", "context": "FUNCTION",
    "type": "string", "language": "C++", "system": "myproj",
    "tokens": null}
 ]}
```

Response:

```json
{"id": 1,
 "model": {"name": "...", "revision": "...", "features": ["context"], "postprocess": true},
 "results": [
   {"key": "a1", "tokens": [
     {"text": "get",   "start": 0, "end": 3,  "tag": "V",  "p": 0.98, "alt": {"tag": "NM", "p": 0.01}, "dictionary": true},
     {"text": "User",  "start": 3, "end": 7,  "tag": "NM", "p": 0.95, "alt": {"tag": "N",  "p": 0.04}, "dictionary": true},
     {"text": "Token", "start": 7, "end": 12, "tag": "N",  "p": 0.99, "alt": {"tag": "NM", "p": 0.01}, "dictionary": true}
   ]}
 ]}
```

- `id` echoes the request ID. `key` echoes the per-identifier key, so results can be matched up even if their order changes.
- Each token is an object rather than the current `{word: {tag, dictionary}}` shape, so fields can be added without breaking clients.
- `p` and `alt` are present only when `options.confidence` is true (item 9).

### 5. Character offsets for each token

> **Status (3.x):** Done.

`start` and `end` are positions in the original identifier string, so `end - start` equals the length of `text`. The tool needs them to highlight a single word and to build rename suggestions. Ronin drops underscores and may split digits, so the original text can't be reconstructed from the tokens alone.

### 6. Caller-supplied tokens (optional)

> **Status (3.x):** Done. A supplied token that is not in the name gets `null` offsets.

If `tokens` is a non-null list of strings, skip `ronin.split` and tag those tokens. The tool uses this for:

- gold splits during evaluation
- domain terms that shouldn't be split (`IPv4`, `utf8`)
- splits the user has corrected

When tokens are supplied, offsets can be computed by matching each token in order against the name, ignoring case.

### 7. Model metadata in every response, plus an `info` command

> **Status (3.x):** Done. The model block has `name`, `revision`, `features`, `postprocess`, `device`, and `scalar_version`. `{"command": "info"}` works over stdio, and `GET /info` over HTTP.

`model.name`, `model.revision` (a checkpoint hash or Hugging Face revision), the feature set, and the postprocessing flag. The tool records these with every finding so results are reproducible, and it uses them as part of its cache key. Retagging only happens when the model changes.

For stdio, `{"command": "info"}` returns this block without tagging anything. For HTTP, use `GET /info`.

### 8. Structured errors per identifier

> **Status (3.x):** Done. Error codes are listed in the README.

One bad identifier shouldn't fail the whole batch:

```json
{"key": "a2", "error": {"code": "UNSUPPORTED_IDENTIFIER", "message": "operator names are not tagged"}}
```

Inputs to handle without crashing: empty strings, `operator==`, `~Foo`, `ns::name`, non-ASCII names, very long names, and names made entirely of digits or underscores. The tool will filter most of these out, but the tagger shouldn't depend on that.

---

## Milestone 2: confidence-aware rules

### 9. Per-token confidence

- `p`: the probability of the chosen tag, from the CRF's per-token marginal probabilities (forward-backward), not from a softmax over raw emission scores.
- `alt`: the runner-up tag and its probability.
- Optionally, a sequence-level probability for the whole identifier.

Rules use these to soften or drop findings when a tag is uncertain. For example: "*get* is tagged as a verb (p = 0.55; noun modifier p = 0.41), so this finding is low confidence."

### 10. Keep the `dictionary` flag, and add `likely_abbreviation`

The prototype tool used `dictionary` to detect abbreviations. A separate `likely_abbreviation` boolean would support abbreviation-related naming rules directly.

---

## Contract clarifications (mostly documentation)

### 11. Type string format

The training `TYPE` column appears to hold the base type name plus pointer stars, with no templates, qualifiers, or namespaces (`int`, `void*`, `List`, `vector`, `String`). Please confirm this and document it in the README. The tool will normalize to the same shape (for example, `const std::vector<Foo>&` becomes `vector`). If the model would benefit from template arguments (`vector<Foo>`) or reference markers, say so, and the tool can send them in an extra field.

### 12. Contexts beyond the five

The tool will see constructors, destructors, enum constants, namespaces, template parameters, lambdas, macros, and Python module-level names. For each one, document which of the five contexts to send, or say that the tool should skip it. Proposed defaults:

| Construct | Proposed context |
|---|---|
| constructor / destructor | skip (the name is the class name) |
| enum constant | ATTRIBUTE |
| namespace | CLASS |
| template parameter | PARAMETER |
| lambda variable | DECLARATION |
| macro | skip for now |

### 13. `system` means the project name

In the training data and in the `sys_sim` feature, `SYSTEM_NAME` is the project (`drill`, `rigraph`), and the feature is used to detect project prefixes such as `gimp`. Please state this in the README. The earlier prototype mistakenly sent the containing class name. The tool will send a configurable project name, defaulting to the repository directory name.

---

## Distribution

### 14. A pip-installable package with a console command

> **Status (3.x):** Done. `pip install "scalar-tagger @ git+https://github.com/SCANL/scanl_tagger.git@<tag>"` installs the `scalar-tagger` command; `scalar-tagger serve --stdio` starts the stdio transport. Publishing to PyPI needs `spiral` published there first, because PyPI rejects packages with git dependencies.

Provide `pip install scalar-tagger` (or an install from git) with a `scalar-tagger` command, so the C++ tool can find and start it without knowing the path to a checked-out repo or a `main` script.

### 15. Faster startup

> **Status (3.x):** Done. About 7 seconds to a warmed-up `ready` line, documented in the README; about 5 seconds of that is importing torch and transformers. `--version` returns immediately, NLTK data is no longer downloaded, and a pinned model that is already cached starts offline.

Spawning over stdio makes import and model-load time matter. Ideas:

- ~~Import the tree-based model's dependencies only when it is selected.~~ Done in 3.0.0: the tree-based model was removed, and `main` imports torch and transformers only for the mode it runs.
- Make the NLTK words-corpus lookup optional, or bundle the word list, so a first run doesn't need to download it.
- Report the measured time from launch to the `ready` line in the README, so the tool can set a sensible timeout.

### 16. Publish the release model at a pinned revision

> **Status (3.x):** Pinning done: `RELEASE_REVISION` in `scalar_tagger/cli.py`, overridable with `--revision`. Pushing the verified 90.5% candidate to the Hub and updating the pin is still to do; see RELEASE_MODEL.md.

`RELEASE_MODEL.md` already plans this. The tool's config will pin a revision, so a fresh install resolves to the exact model the reported metrics describe.