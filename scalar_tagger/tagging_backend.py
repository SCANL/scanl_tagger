import gzip
import hashlib
import json
import os
import re
from importlib import resources

from spiral import ronin

from scalar_tagger import contract
from scalar_tagger.calibration import confidence_fields
from scalar_tagger.lm_based_tagger.distilbert_tagger import DistilBertTagger
from scalar_tagger.version import __version__


def load_english_words(extra_words_path: str = os.path.join("words", "en.txt")) -> set[str]:
    """
    Return the bundled English word list (the NLTK words corpus, lowercased; see
    scalar_tagger/data/README.md), plus one word per line from `extra_words_path` if that
    file exists.
    """
    bundled = resources.files("scalar_tagger").joinpath("data", "english_words.txt.gz")
    with bundled.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as handle:
        english_words = set(handle.read().split())

    if extra_words_path and os.path.exists(extra_words_path):
        with open(extra_words_path) as handle:
            english_words.update(line.strip().lower() for line in handle if line.strip())
    return english_words


def _revision_cache_path() -> str:
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache_home, "scalar-tagger", "revisions.json")


def file_revision(weight_path: str) -> str:
    """
    "sha256:" plus the first 16 hex digits of the file's hash. Hashes are cached by path,
    size, and modification time, so an unchanged checkpoint isn't re-read on every start.
    """
    stat = os.stat(weight_path)
    key = os.path.realpath(weight_path)
    fingerprint = [stat.st_size, stat.st_mtime_ns]
    cache_path = _revision_cache_path()

    try:
        with open(cache_path) as handle:
            cache = json.load(handle)
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(key)
    if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint:
        return entry["revision"]

    digest = hashlib.sha256()
    with open(weight_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    revision = f"sha256:{digest.hexdigest()[:16]}"

    cache[key] = {"fingerprint": fingerprint, "revision": revision}
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as handle:
            json.dump(cache, handle)
    except OSError:
        pass  # a read-only home directory just means hashing again next time
    return revision


def model_revision(model_path: str, revision: str | None = None) -> str | None:
    """
    Identify the exact checkpoint being served.

    For a Hugging Face repo this is the commit hash of the downloaded snapshot. For a local
    directory it is a hash of model.safetensors (see `file_revision`).
    """
    if os.path.isdir(model_path):
        weight_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(weight_path):
            return None
        return file_revision(weight_path)

    from huggingface_hub import hf_hub_download
    try:
        # The model is already downloaded, so this resolves from the local cache:
        # .../models--<repo>/snapshots/<commit>/config.json
        config_path = hf_hub_download(
            repo_id=model_path, filename="config.json", revision=revision, local_files_only=True
        )
    except Exception:
        return None
    return os.path.basename(os.path.dirname(config_path))


def cached_snapshot(repo_id: str, revision: str | None) -> str | None:
    """
    The local snapshot directory for a pinned commit that is already downloaded, or None.

    A commit hash can't change, so a cached copy can be used without asking the Hub whether
    it is current. That keeps startup fast and lets the tagger start offline.
    """
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return None
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
    except Exception:
        return None


class TaggingBackend:
    def __init__(
        self,
        model_path: str,
        local: bool = False,
        revision: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
        tagger=None,
        english_words: set[str] | None = None,
    ):
        """
        Args:
            model_path: local checkpoint directory or Hugging Face repo id.
            revision: Hugging Face commit, branch, or tag to load; ignored for local directories.
            device: "cpu", "cuda", "cuda:1", or None/"auto" to use the GPU when there is one.
            tagger: an already-loaded tagger to use instead of loading `model_path`.
            english_words: lowercased words for the `dictionary` flag; defaults to the bundled list.
        """
        if not model_path:
            raise ValueError("Tagging requires a model path or HuggingFace repo id.")

        self.model_path = model_path
        self.batch_size = batch_size
        snapshot = None if local or tagger is not None else cached_snapshot(model_path, revision)
        self.lm_model = tagger or DistilBertTagger(
            snapshot or model_path,
            local=local or snapshot is not None,
            device=device,
            revision=None if snapshot else revision,
        )
        self.english_words = english_words if english_words is not None else load_english_words()
        if tagger is not None:
            self.revision = None
        elif snapshot:
            self.revision = revision
        else:
            self.revision = model_revision(model_path, revision)
        if tagger is None:
            self.warm_up()

    def warm_up(self) -> None:
        """
        Run one identifier through the splitter and the model. Both do one-time setup on first
        use (Ronin loads its dictionary; the GPU initializes its kernels), so doing it here keeps
        that cost out of the first real request.
        """
        self.lm_model.tag_identifiers([{"tokens": ronin.split("getUserName"), "context": "FUNCTION"}])

    def model_info(self) -> dict:
        """
        Describe the model serving requests. `postprocess` is always false: 3.0 has no tag
        postprocessing (the CRF's transition constraints replaced it), and the request option
        is accepted only for compatibility.
        """
        return {
            "name": self.model_path,
            "revision": self.revision,
            "features": list(self.lm_model.selected_features),
            "postprocess": False,
            "confidence_temperature": getattr(self.lm_model, "confidence_temperature", 1.0),
            "confidence_calibration": (getattr(self.lm_model, "confidence_calibration", None) or {}).get("method"),
            "device": str(getattr(self.lm_model, "device", "cpu")),
            "scalar_version": __version__,
        }

    def tag_batch(self, request) -> dict:
        """
        Tag every identifier in a request, following the contract in `scalar_tagger/contract.py`.

        Never raises for bad input: a malformed request gets a top-level "error", and a bad
        identifier gets an "error" in its own result while the rest of the batch is tagged.
        """
        request_id = request.get("id") if isinstance(request, dict) else None
        try:
            parsed = contract.parse_request(request)
        except contract.ContractError as exc:
            return {"id": request_id, "model": self.model_info(), "error": exc.to_json()}

        options = parsed.options
        response = {"id": parsed.id, "model": self.model_info()}

        tag_indices = [i for i, (_, item) in enumerate(parsed.items) if isinstance(item, contract.IdentifierInput)]
        rows = [
            {
                "tokens": parsed.items[i][1].tokens,
                "context": parsed.items[i][1].context,
                "type_str": parsed.items[i][1].type_str,
                "language": parsed.items[i][1].language,
                "system_name": parsed.items[i][1].system,
            }
            for i in tag_indices
        ]
        tags_by_index = dict(zip(tag_indices, self._tag_rows(rows, options.confidence)))

        results = []
        for index, (key, item) in enumerate(parsed.items):
            if isinstance(item, contract.ContractError):
                results.append({"key": key, "error": item.to_json()})
                continue

            tags = tags_by_index[index]
            if isinstance(tags, contract.ContractError):
                results.append({"key": key, "error": tags.to_json()})
                continue

            marginals = None
            if options.confidence:
                tags, marginals = tags["tags"], tags["marginals"]

            offsets = contract.token_offsets(item.name, item.tokens)
            tokens = []
            for position, (token, tag, (start, end)) in enumerate(zip(item.tokens, tags, offsets)):
                entry = {"text": token, "start": start, "end": end, "tag": tag}
                if marginals is not None:
                    entry.update(confidence_fields(tag, marginals[position], getattr(self.lm_model, "confidence_calibration", None)))
                entry["dictionary"] = token.lower() in self.english_words
                tokens.append(entry)
            results.append({"key": key, "tokens": tokens})

        response["results"] = results
        return response

    def _tag_rows(self, rows, confidence: bool = False):
        """
        Tag rows in batches, returning tags (or, with `confidence`, {"tags", "marginals"}) or a
        ContractError for each row. If a batch fails, its rows are retried one at a time, so a
        single bad identifier can't fail the others.
        """
        options = {"return_marginals": True} if confidence else {}
        try:
            predictions = self.lm_model.tag_identifiers(rows, batch_size=self.batch_size, **options)
        except Exception:
            predictions = []
            for row in rows:
                try:
                    predictions.append(self.lm_model.tag_identifiers([row], **options)[0])
                except Exception as exc:
                    predictions.append(contract.ContractError(contract.INTERNAL_ERROR, str(exc)))

        too_long = contract.ContractError(
            contract.IDENTIFIER_TOO_LONG,
            "identifier has too many words to fit in the model input",
        )
        return [too_long if tags is None else tags for tags in predictions]

    def tag_identifier(
        self,
        identifier_name: str,
        context: str,
        type_str: str = "",
        language: str = "",
        system_name: str = "",
    ) -> dict:
        words = ronin.split(identifier_name)
        tags = self.lm_model.tag_identifiers([{
            "tokens": words,
            "context": context,
            "type_str": type_str,
            "language": language,
            "system_name": system_name,
        }])[0]
        if tags is None:
            raise ValueError("Identifier is too long for the model input.")
        return {"tokens": words, "tags": list(tags)}

    def tag_identifier_batch(self, records, batch_size: int = 64):
        rows = [
            {
                "tokens": ronin.split(record["identifier_name"]),
                "context": record["context"],
                "type_str": record.get("type_str", ""),
                "language": record.get("language", ""),
                "system_name": record.get("system_name", ""),
            }
            for record in records
        ]
        batch_predictions = self.lm_model.tag_identifiers(rows, batch_size=batch_size)
        return [
            {"tokens": row["tokens"], "tags": tags}
            for row, tags in zip(rows, batch_predictions)
        ]
