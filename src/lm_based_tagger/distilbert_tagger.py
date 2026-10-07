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
    - Running the model
    - Post-processing the raw logits or CRF predictions
    - Aligning subword tokens back to word-level predictions
    """
    def __init__(self, model_path: str, local: bool = False, device: str | None = None):
        # Run on the GPU when there is one, unless the caller picks a device.
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load tokenizer from local directory or remote HuggingFace path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=local)

        # Try loading CRF-enhanced model; fallback to plain classifier if not available
        try:
            self.model = DistilBertCRFForTokenClassification.from_pretrained(model_path, local=local)
        except Exception:
            self.model = DistilBertForTokenClassification.from_pretrained(model_path, local_files_only=local)

        # disable dropout, etc. for inference
        self.model.to(self.device)
        self.model.eval()

        # Use the same feature tokens the checkpoint was trained with. An empty list means the
        # model was trained without features; only a missing list means a legacy checkpoint.
        saved_features = getattr(self.model.config, "selected_features", None)
        self.selected_features = normalize_selected_features(
            LEGACY_FEATURES if saved_features is None else saved_features
        )

        # map label IDs to strings
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def tag_identifier(self, tokens, context, type_str, language, system_name):
        """
        Tag a split identifier using the model, returning a sequence of grammar pattern labels (e.g., ["V", "NM", "N"]).

        Steps:
        1) Build full input token list:
              [feature tokens] + [@pos_0, w1, @pos_1, w2, ..., @pos_2, wn]
        2) Tokenize using HuggingFace tokenizer with is_split_into_words=True
        3) Use word_ids() to find the first subtoken of each identifier word
              - Skip special tokens (None)
              - Skip feature tokens (index < number of feature tokens)
              - Use only the *second* token in each [@pos_X, word] pair (the word)
              - Skip repeated subword tokens (only use the first subtoken per word)
        4) Run the model forward pass with a word mask, so the CRF decodes over words only
        5) Return a list of string labels corresponding to the original identifier tokens.

        Returns:
            List[str]: a list of grammar tags (e.g., ['V', 'NM', 'N']) aligned to `tokens`

        Raises:
            ValueError: if the identifier is too long to fit in the model's input.
        """
        row = {
            "CONTEXT": context,
            "SYSTEM_NAME": system_name or "",
            "TYPE": type_str or "",
            "LANGUAGE": language or "",
        }

        # Step 1: Feature tokens + alternating position/word tokens
        input_tokens, feature_count = build_model_input_tokens(row, tokens, self.selected_features)

        # Step 2: Tokenize using word-alignment aware tokenizer
        encoded = self.tokenizer(
            input_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Step 3: Find the first subtoken of each identifier word
        positions, previous_word_idx = [], None
        for idx, word_idx in enumerate(encoded.word_ids()):
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

        if len(positions) != len(tokens):
            raise ValueError("Identifier is too long for the model input.")

        encoded = encoded.to(self.device)
        word_mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
        word_mask[0, positions] = True

        # Step 4: Forward pass (the CRF decodes over word positions only)
        with torch.inference_mode():
            if isinstance(self.model, DistilBertCRFForTokenClassification):
                out = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    word_mask=word_mask,
                )
            else:
                out = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

        # Step 5: Read the label at each word position
        if isinstance(out, dict) and "predictions" in out:
            # CRF predictions exclude [CLS], so they lag input positions by 1
            labels_per_token = out["predictions"][0]
            pred_labels = [labels_per_token[idx - 1] for idx in positions]
        else:
            logits = out[0] if isinstance(out, (tuple, list)) else out.logits
            labels_per_token = torch.argmax(logits, dim=-1)[0].tolist()
            pred_labels = [labels_per_token[idx] for idx in positions]

        # Step 6: Map label IDs back to string labels
        return [self.id2label[i] for i in pred_labels]
