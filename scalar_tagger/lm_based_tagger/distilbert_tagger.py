import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from .distilbert_crf import DistilBertCRFForTokenClassification
from .distilbert_preprocessing import *

class DistilBertTagger:
    """
    A lightweight wrapper around a DistilBERT+CRF or DistilBERT-only model for tagging identifier tokens
    with part-of-speech-like grammar labels (e.g., V, NM, N, etc.).

    Automatically handles:
    - Tokenization (with custom feature and position tokens)
    - Running the model, many identifiers per batch
    - Aligning subword tokens back to word-level predictions
    """
    def __init__(self, model_path: str, local: bool = False, device: str | None = None, revision: str | None = None):
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

        # Use the same feature tokens the checkpoint was trained with. An empty list means the
        # model was trained without features; only a missing list means a legacy checkpoint.
        saved_features = getattr(self.model.config, "selected_features", None)
        self.selected_features = normalize_selected_features(
            LEGACY_FEATURES if saved_features is None else saved_features
        )

        # Temperature for confidence marginals, fitted on held-out data and saved in the
        # checkpoint config; 1.0 (raw CRF marginals) when the checkpoint has none.
        self.confidence_temperature = float(getattr(self.model.config, "confidence_temperature", 1.0) or 1.0)

        # map label IDs to strings
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def tag_identifier(self, tokens, context, type_str, language, system_name):
        """
        Tag a split identifier, returning one grammar tag per token (e.g., ["V", "NM", "N"]).

        Raises:
            ValueError: if the identifier is too long to fit in the model's input.
        """
        tags = self.tag_identifiers([{
            "tokens": tokens,
            "context": context,
            "type_str": type_str,
            "language": language,
            "system_name": system_name,
        }])[0]
        if tags is None:
            raise ValueError("Identifier is too long for the model input.")
        return tags

    def tag_identifiers(self, rows, batch_size: int = 64, return_marginals: bool = False):
        """
        Tag many split identifiers, running the model `batch_size` identifiers at a time.

        Each row is a dict with "tokens" (a non-empty list of words), "context", and optionally
        "type_str", "language", and "system_name".

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
        5) Map label IDs back to tags

        Returns:
            List[List[str] | None]: tags aligned to each row's tokens, in input order. A row is
            None when truncation dropped some of its words, so it can't be tagged in full.
            With `return_marginals`, each tagged row is instead {"tags": [...], "marginals":
            [{tag: probability, ...} per word]}, the CRF's forward-backward marginals.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        results = [None] * len(rows)
        # Group rows of similar length so each batch carries little padding.
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tokens"]))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            batch = self._tag_batch([rows[i] for i in batch_indices], return_marginals)
            for index, tags in zip(batch_indices, batch):
                results[index] = tags
        return results

    def _tag_batch(self, rows, return_marginals: bool = False):
        # Step 1: Feature tokens + alternating position/word tokens
        inputs, feature_counts = [], []
        for row in rows:
            feature_row = {
                "CONTEXT": row["context"],
                "SYSTEM_NAME": row.get("system_name") or "",
                "TYPE": row.get("type_str") or "",
                "LANGUAGE": row.get("language") or "",
            }
            input_tokens, feature_count = build_model_input_tokens(feature_row, row["tokens"], self.selected_features)
            inputs.append(input_tokens)
            feature_counts.append(feature_count)

        # Step 2: Tokenize using word-alignment aware tokenizer
        encoded = self.tokenizer(
            inputs,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        # Step 3: Find the first subtoken of each identifier word
        word_positions = []
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

        encoded = encoded.to(self.device)
        word_mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
        for row_index, positions in enumerate(word_positions):
            word_mask[row_index, positions] = True

        # Step 4: Forward pass (the CRF decodes over word positions only)
        with torch.inference_mode():
            if isinstance(self.model, DistilBertCRFForTokenClassification):
                out = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    word_mask=word_mask,
                    **({"return_marginals": True, "marginal_temperature": self.confidence_temperature}
                       if return_marginals else {}),
                )
            else:
                out = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

        # Step 5: Read the label at each word position
        if isinstance(out, dict) and "predictions" in out:
            # CRF predictions exclude [CLS], so they lag input positions by 1
            batch_labels = [
                [labels[idx - 1] for idx in positions]
                for labels, positions in zip(out["predictions"], word_positions)
            ]
        else:
            logits = out[0] if isinstance(out, (tuple, list)) else out.logits
            argmax = torch.argmax(logits, dim=-1).tolist()
            batch_labels = [
                [labels[idx] for idx in positions]
                for labels, positions in zip(argmax, word_positions)
            ]

        batch_marginals = None
        if return_marginals:
            if isinstance(out, dict) and "marginals" in out:
                batch_marginals = out["marginals"]
            else:  # a plain classifier has no CRF; fall back to a softmax over emissions
                probs = torch.softmax(logits.float(), dim=-1).tolist()
                batch_marginals = [[row[idx] for idx in positions] for row, positions in zip(probs, word_positions)]

        # Step 6: Map label IDs back to string labels
        results = []
        for row_index, (row, pred_labels) in enumerate(zip(rows, batch_labels)):
            if len(pred_labels) != len(row["tokens"]):
                results.append(None)  # truncation dropped some words
                continue
            tags = [self.id2label[i] for i in pred_labels]
            if batch_marginals is None:
                results.append(tags)
                continue
            marginals = [
                {self.id2label[label]: probability for label, probability in enumerate(word)}
                for word in batch_marginals[row_index]
            ]
            results.append({"tags": tags, "marginals": marginals})
        return results
