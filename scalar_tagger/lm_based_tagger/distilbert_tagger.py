import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from .distilbert_crf import DistilBertCRFForTokenClassification 
from .distilbert_preprocessing import *

NOMINAL_TAGS = {"N", "NM", "NPL"}
POSITION0_PRIOR_TAGS = {"PRE", "NM", "N"}
POSITION0_PRIOR_SOURCE_TAGS = {"PRE", "NM"}

class DistilBertTagger:
    """
    A lightweight wrapper around a DistilBERT+CRF or DistilBERT-only model for tagging identifier tokens
    with part-of-speech-like grammar labels (e.g., V, NM, N, etc.).

    Automatically handles:
    - Tokenization (with custom feature and position tokens)
    - Running the model
    - Post-processing the raw logits or CRF predictions
    - Aligning subword tokens back to word-level predictions
    """
    def __init__(
        self,
        model_path: str,
        local: bool = False,
        pattern_postprocessing: bool | None = None,
        device: str | None = None,
        revision: str | None = None,
    ):
        # Run on the GPU when there is one, unless the caller picks a device. An explicit
        # device skips torch.cuda.is_available(), which can crash on some broken CUDA setups.
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load tokenizer from local directory or remote HuggingFace path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=local, revision=revision)

        # Try loading CRF-enhanced model; fallback to plain classifier if not available
        try:
            self.model = DistilBertCRFForTokenClassification.from_pretrained(model_path, local=local, revision=revision)
        except Exception:
            self.model = DistilBertForTokenClassification.from_pretrained(model_path, local_files_only=local, revision=revision)

        # disable dropout, etc. for inference
        self.model.to(self.device)
        self.model.eval()

        self.selected_features = normalize_selected_features(
            getattr(self.model.config, "selected_features", None)
        )
        saved_postprocessing_default = getattr(
            self.model.config,
            "pattern_postprocessing_default",
            False,
        )
        self.pattern_postprocessing = (
            bool(saved_postprocessing_default)
            if pattern_postprocessing is None
            else bool(pattern_postprocessing)
        )
        self.position0_label_priors = getattr(self.model.config, "position0_label_priors", {}) or {}
        
        # map label IDs to strings
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def _apply_position0_prior(self, pred_tags, tokens, context):
        if not pred_tags or not tokens:
            return list(pred_tags)

        repaired = list(pred_tags)
        current = repaired[0]
        if current not in POSITION0_PRIOR_SOURCE_TAGS:
            return repaired

        token = str(tokens[0]).strip().lower()
        if not token:
            return repaired

        by_context = self.position0_label_priors.get("by_context", {})
        context_map = by_context.get(str(context or ""), {})
        prior_label = context_map.get(token)

        if prior_label is None:
            prior_label = self.position0_label_priors.get("global", {}).get(token)

        if prior_label in POSITION0_PRIOR_TAGS:
            repaired[0] = prior_label

        return repaired

    def _repair_nominal_chunk(self, chunk_tags, next_tag):
        repaired = list(chunk_tags)
        head_positions = [i for i, tag in enumerate(repaired) if tag in {"N", "NPL"}]

        if not head_positions and repaired and next_tag == "D":
            repaired[-1] = "N"
            head_positions = [len(repaired) - 1]

        if head_positions:
            head_pos = head_positions[-1]
            for i in range(head_pos):
                if repaired[i] == "N":
                    repaired[i] = "NM"

        return repaired

    def _postprocess_pattern(self, pred_tags):
        repaired = list(pred_tags)
        chunk_start = None

        for idx, tag in enumerate(repaired + [None]):
            if tag in NOMINAL_TAGS:
                if chunk_start is None:
                    chunk_start = idx
                continue

            if chunk_start is not None:
                repaired[chunk_start:idx] = self._repair_nominal_chunk(
                    repaired[chunk_start:idx],
                    next_tag=tag,
                )
                chunk_start = None

        return repaired

    def postprocess_tags(self, pred_tags, tokens=None, context=None, language=None, system_name=None):
        repaired = self._apply_position0_prior(pred_tags, tokens or [], context)
        return self._postprocess_pattern(repaired)

    def tag_identifier(
        self,
        tokens,
        context,
        type_str,
        language,
        system_name,
        pattern_postprocessing: bool | None = None,
    ):
        """
        Tag a split identifier, returning one grammar tag per token (e.g., ["V", "NM", "N"]).

        Raises:
            ValueError: if the identifier is too long to fit in the model's input.
        """
        tags = self.tag_identifiers(
            [{
                "tokens": tokens,
                "context": context,
                "type_str": type_str,
                "language": language,
                "system_name": system_name,
                "pattern_postprocessing": pattern_postprocessing,
            }]
        )[0]
        if tags is None:
            raise ValueError("Identifier is too long for the model input.")
        return tags

    def tag_identifiers(self, rows, batch_size: int = 64):
        """
        Tag many split identifiers, running the model `batch_size` identifiers at a time.

        Each row is a dict with "tokens" (a non-empty list of words), "context", and optionally
        "type_str", "language", "system_name", and "pattern_postprocessing".

        Steps, per batch:
        1) Build each row's input tokens:
              [feature tokens] + [@pos_0, w1, @pos_1, w2, ..., @pos_2, wn]
        2) Tokenize the batch with is_split_into_words=True, padding to the longest row
        3) Use word_ids() to find the first subtoken of each identifier word
              - Skip special tokens (None)
              - Skip feature tokens (index < that row's feature count)
              - Use only the *second* token in each [@pos_X, word] pair (the word)
              - Skip repeated subword tokens (only use the first subtoken per word)
        4) Run the model forward pass with a word mask, so the CRF decodes over words only
        5) Map label IDs back to tags, and optionally postprocess

        Returns:
            List[List[str] | None]: tags aligned to each row's tokens, in input order. A row is
            None when truncation dropped some of its words, so it can't be tagged in full.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        results = [None] * len(rows)
        # Group rows of similar length so each batch carries little padding.
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tokens"]))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            batch_rows = [rows[i] for i in batch_indices]
            for index, tags in zip(batch_indices, self._tag_batch(batch_rows)):
                results[index] = tags
        return results

    def _tag_batch(self, rows):
        # Step 1: Feature tokens + alternating position/word tokens
        inputs, feature_counts = [], []
        for row in rows:
            feature_row = {
                "CONTEXT": row["context"],
                "SYSTEM_NAME": row.get("system_name", ""),
                "TYPE": row.get("type_str", ""),
                "LANGUAGE": row.get("language", ""),
            }
            input_tokens, feature_count = build_model_input_tokens(
                feature_row, row["tokens"], self.selected_features
            )
            inputs.append(input_tokens)
            feature_counts.append(feature_count)

        # Step 2: Tokenize using word-alignment aware tokenizer
        encoded = self.tokenizer(
            inputs,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Step 3: Find the first subtoken of each identifier word
        word_positions = []
        encoded = encoded.to(self.device)
        word_mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
        for row_index, feature_count in enumerate(feature_counts):
            positions, previous_word_idx = [], None
            for idx, word_idx in enumerate(encoded.word_ids(batch_index=row_index)):
                if word_idx is None:
                    continue  # special token (CLS, SEP, PAD, etc.)
                if word_idx < feature_count:
                    continue  # feature tokens (shouldn't be labeled)
                if (word_idx - feature_count) % 2 == 0:
                    continue  # position tokens (e.g., @pos_0)
                if word_idx == previous_word_idx:
                    continue  # skip repeated subword tokens
                positions.append(idx)
                previous_word_idx = word_idx
            word_positions.append(positions)
            word_mask[row_index, positions] = True

        # Step 4: Forward pass (the CRF decodes over word positions only)
        with torch.inference_mode():
            out = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                word_mask=word_mask,
            )

        # Step 5: Read the label at each word position
        if isinstance(out, dict) and "predictions" in out:
            # CRF predictions exclude [CLS], so they lag input positions by 1
            batch_labels = [
                [labels[idx - 1] for idx in positions]
                for labels, positions in zip(out["predictions"], word_positions)
            ]
        else:
            logits = out[0] if isinstance(out, (tuple, list)) else out
            argmax = torch.argmax(logits, dim=-1).cpu().tolist()
            batch_labels = [
                [labels[idx] for idx in positions]
                for labels, positions in zip(argmax, word_positions)
            ]

        # Step 6: Map label IDs back to string labels
        results = []
        for row, pred_labels in zip(rows, batch_labels):
            tokens = row["tokens"]
            if len(pred_labels) != len(tokens):
                results.append(None)  # truncation dropped some words
                continue

            pred_tag_strings = [self.id2label[i] for i in pred_labels]
            override = row.get("pattern_postprocessing")
            should_postprocess = self.pattern_postprocessing if override is None else bool(override)
            if should_postprocess:
                pred_tag_strings = self.postprocess_tags(
                    pred_tag_strings,
                    tokens=tokens,
                    context=row["context"],
                    language=row.get("language", ""),
                    system_name=row.get("system_name", ""),
                )
            results.append(pred_tag_strings)
        return results
