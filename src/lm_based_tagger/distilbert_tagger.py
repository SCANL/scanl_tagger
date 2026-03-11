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
    def __init__(self, model_path: str, local: bool = False, pattern_postprocessing: bool = False):
        # Load tokenizer from local directory or remote HuggingFace path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=local)

        # Try loading CRF-enhanced model; fallback to plain classifier if not available
        try:
            self.model = DistilBertCRFForTokenClassification.from_pretrained(model_path, local=local)
        except Exception:
            self.model = DistilBertForTokenClassification.from_pretrained(model_path, local_files_only=local)

        # disable dropout, etc. for inference
        self.model.eval()

        self.selected_features = normalize_selected_features(
            getattr(self.model.config, "selected_features", None)
        )
        self.pattern_postprocessing = pattern_postprocessing
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

    def tag_identifier(self, tokens, context, type_str, language, system_name):
        """
        Tag a split identifier using the model, returning a sequence of grammar pattern labels (e.g., ["V", "NM", "N"]).

        Steps:
        1) Build full input token list:
              [feature tokens] + [@pos_0, w1, @pos_1, w2, ..., @pos_2, wn]
        2) Tokenize using HuggingFace tokenizer with is_split_into_words=True
        3) Run the model forward pass (handles CRF or logits automatically)
        4) Use word_ids() to align predictions back to full words
              - Skip special tokens (None)
              - Skip feature tokens (index < configured feature count)
              - Use only the *second* token in each [@pos_X, word] pair (the word)
              - Skip repeated subword tokens (only use the first subtoken per word)
        5) Return a list of string labels corresponding to the original identifier tokens.

        Returns:
            List[str]: a list of grammar tags (e.g., ['V', 'NM', 'N']) aligned to `tokens`
        """
        row = {
            "CONTEXT": context,
            "SYSTEM_NAME": system_name,
            "TYPE": type_str,
            "LANGUAGE": language
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

        # Step 3: Forward pass
        with torch.no_grad():
            out = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        # Step 4: Get predictions depending on model type (CRF vs logits)
        if isinstance(out, dict) and "predictions" in out:
            labels_per_token = out["predictions"][0]
        else:
            logits = out[0] if isinstance(out, (tuple, list)) else out
            labels_per_token = torch.argmax(logits, dim=-1).squeeze().tolist()

        # Step 5: Convert subtoken-level predictions to word-level predictions
        pred_labels, previous_word_idx = [], None
        word_ids = encoded.word_ids()

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # special token (CLS, SEP, PAD, etc.)
            if word_idx < feature_count:
                continue  # feature tokens (shouldn't be labeled)
            if (word_idx - feature_count) % 2 == 0:
                continue  # position tokens (e.g., @pos_0)
            if word_idx == previous_word_idx:
                continue  # skip repeated subword tokens
            
            # Heuristic: labels lag by 1 position relative to input_ids
            label_idx = idx - 1
            if label_idx < len(labels_per_token):
                pred_labels.append(labels_per_token[label_idx])
            previous_word_idx = word_idx
        
        # Step 6: Map label IDs back to string labels
        pred_tag_strings = [self.id2label[i] for i in pred_labels]
        if self.pattern_postprocessing:
            pred_tag_strings = self.postprocess_tags(
                pred_tag_strings,
                tokens=tokens,
                context=context,
                language=language,
                system_name=system_name,
            )
        return pred_tag_strings
