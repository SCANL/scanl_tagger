import hashlib
import os

import nltk
from spiral import ronin

from src import contract
from src.lm_based_tagger.distilbert_tagger import DistilBertTagger
from version import __version__


def load_english_words() -> set[str]:
    """Return the lowercased NLTK words corpus, downloading it on first use."""
    try:
        words = nltk.corpus.words.words()
    except LookupError:
        nltk.download("words", quiet=True)
        words = nltk.corpus.words.words()
    return set(w.lower() for w in words)


def model_revision(model_path: str, local: bool) -> str | None:
    """
    Identify the exact checkpoint being served.

    For a Hugging Face repo this is the commit hash of the downloaded snapshot. For a local
    directory it is "sha256:" plus the first 16 hex digits of model.safetensors' hash.
    """
    if os.path.isdir(model_path):
        weight_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(weight_path):
            return None
        digest = hashlib.sha256()
        with open(weight_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()[:16]}"

    from huggingface_hub import hf_hub_download
    try:
        # The model is already downloaded, so this resolves from the local cache:
        # .../models--<repo>/snapshots/<commit>/config.json
        config_path = hf_hub_download(repo_id=model_path, filename="config.json", local_files_only=True)
    except Exception:
        return None
    return os.path.basename(os.path.dirname(config_path))


class TaggingBackend:
    def __init__(
        self,
        model_path: str,
        local: bool = False,
        pattern_postprocessing: bool | None = None,
        batch_size: int = 64,
        tagger=None,
        english_words: set[str] | None = None,
    ):
        """
        Args:
            model_path: local checkpoint directory or Hugging Face repo id.
            tagger: an already-loaded tagger to use instead of loading `model_path`.
            english_words: lowercased words for the `dictionary` flag; defaults to NLTK's corpus.
        """
        if not model_path:
            raise ValueError("Tagging requires a model path or HuggingFace repo id.")

        self.model_path = model_path
        self.pattern_postprocessing = pattern_postprocessing
        self.batch_size = batch_size
        self.lm_model = tagger or DistilBertTagger(
            model_path,
            local=local,
            pattern_postprocessing=pattern_postprocessing,
        )
        self.english_words = english_words if english_words is not None else load_english_words()
        self.revision = model_revision(model_path, local) if tagger is None else None

    def model_info(self, postprocess: bool | None = None) -> dict:
        """
        Describe the model serving this request. `postprocess` is the request's override, if
        any; the reported flag is the one actually applied.
        """
        return {
            "name": self.model_path,
            "revision": self.revision,
            "features": list(self.lm_model.selected_features),
            "postprocess": self.lm_model.pattern_postprocessing if postprocess is None else postprocess,
            "device": str(getattr(self.lm_model, "device", "cpu")),
            "scalar_version": __version__,
        }

    def tag_batch(self, request) -> dict:
        """
        Tag every identifier in a request, following the contract in `src/contract.py`.

        Never raises for bad input: a malformed request gets a top-level "error", and a bad
        identifier gets an "error" in its own result while the rest of the batch is tagged.
        """
        request_id = request.get("id") if isinstance(request, dict) else None
        try:
            parsed = contract.parse_request(request)
        except contract.ContractError as exc:
            return {"id": request_id, "model": self.model_info(), "error": exc.to_json()}

        options = parsed.options
        response = {"id": parsed.id, "model": self.model_info(options.postprocess)}
        if options.confidence:
            response["warnings"] = [{
                "code": "CONFIDENCE_UNAVAILABLE",
                "message": "per-token confidence is not supported yet; 'p' and 'alt' are omitted",
            }]

        tag_indices = [i for i, (_, item) in enumerate(parsed.items) if isinstance(item, contract.IdentifierInput)]
        rows = [
            {
                "tokens": parsed.items[i][1].tokens,
                "context": parsed.items[i][1].context,
                "type_str": parsed.items[i][1].type_str,
                "language": parsed.items[i][1].language,
                "system_name": parsed.items[i][1].system,
                "pattern_postprocessing": options.postprocess,
            }
            for i in tag_indices
        ]
        tags_by_index = dict(zip(tag_indices, self._tag_rows(rows)))

        results = []
        for index, (key, item) in enumerate(parsed.items):
            if isinstance(item, contract.ContractError):
                results.append({"key": key, "error": item.to_json()})
                continue

            tags = tags_by_index[index]
            if isinstance(tags, contract.ContractError):
                results.append({"key": key, "error": tags.to_json()})
                continue

            offsets = contract.token_offsets(item.name, item.tokens)
            results.append({
                "key": key,
                "tokens": [
                    {
                        "text": token,
                        "start": start,
                        "end": end,
                        "tag": tag,
                        "dictionary": token.lower() in self.english_words,
                    }
                    for token, tag, (start, end) in zip(item.tokens, tags, offsets)
                ],
            })

        response["results"] = results
        return response

    def _tag_rows(self, rows):
        """
        Tag rows in batches, returning tags or a ContractError for each row. If a batch fails,
        its rows are retried one at a time, so a single bad identifier can't fail the others.
        """
        try:
            predictions = self.lm_model.tag_identifiers(rows, batch_size=self.batch_size)
        except Exception:
            predictions = []
            for row in rows:
                try:
                    predictions.append(self.lm_model.tag_identifiers([row])[0])
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
        pattern_postprocessing: bool | None = None,
    ) -> dict:
        words = ronin.split(identifier_name)
        tags = self.lm_model.tag_identifier(
            tokens=words,
            context=context,
            type_str=type_str,
            language=language,
            system_name=system_name,
            pattern_postprocessing=pattern_postprocessing,
        )
        return {"tokens": words, "tags": list(tags)}

    def tag_identifier_batch(self, records, batch_size: int = 64):
        rows = [
            {
                "tokens": ronin.split(record["identifier_name"]),
                "context": record["context"],
                "type_str": record.get("type_str", ""),
                "language": record.get("language", ""),
                "system_name": record.get("system_name", ""),
                "pattern_postprocessing": record.get("pattern_postprocessing"),
            }
            for record in records
        ]
        batch_predictions = self.lm_model.tag_identifiers(rows, batch_size=batch_size)
        return [
            {"tokens": row["tokens"], "tags": tags}
            for row, tags in zip(rows, batch_predictions)
        ]
