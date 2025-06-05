import torch
from nltk import pos_tag
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from .distilbert_crf import DistilBertCRFForTokenClassification 
from .distilbert_preprocessing import *

class DistilBertTagger:
    def __init__(self, model_path: str, local: bool = False):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=local)

        try:
            self.model = DistilBertCRFForTokenClassification.from_pretrained(model_path, local=local)
        except Exception:
            self.model = DistilBertForTokenClassification.from_pretrained(model_path, local_files_only=local)

        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def tag_identifier(self, tokens, context, type_str, language, system_name):
        """
        1) Build the “feature tokens + position tokens + identifier tokens” sequence
        2) Tokenize with `is_split_into_words=True`
        3) Run the model, take argmax over token logits
        4) Align via `word_ids()`, skipping:
              - Any word_id = None
              - Any word_id < N (number of feature tokens) => labels = -100
              - Repeated word_ids (so we pick only the first sub-token of each “(pos, identifier-word)” pair)
        5) Return a list of string labels by mapping numeric IDs through `self.id2label`.
        """
        row = {
            "CONTEXT": context,
            "SYSTEM_NAME": system_name,
            "TYPE": type_str,
            "LANGUAGE": language
        }
        feature_tokens = get_feature_tokens(row, tokens)

        length = len(tokens)
        pos_tokens = ["@pos_2"] if length == 1 else ["@pos_0"] + ["@pos_1"] * (length - 2) + ["@pos_2"]
        tokens_with_pos = [val for pair in zip(pos_tokens, tokens) for val in pair]

        input_tokens = feature_tokens + tokens_with_pos

        encoded = self.tokenizer(
            input_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            out = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        if isinstance(out, dict) and "predictions" in out:
            labels_per_token = out["predictions"][0]
        else:
            logits = out[0] if isinstance(out, (tuple, list)) else out
            labels_per_token = torch.argmax(logits, dim=-1).squeeze().tolist()

        pred_labels, previous_word_idx = [], None
        word_ids = encoded.word_ids()

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx < NUMBER_OF_FEATURES:
                continue
            if (word_idx - NUMBER_OF_FEATURES) % 2 == 0:
                continue
            if word_idx == previous_word_idx:
                continue

            label_idx = idx - 1
            if label_idx < len(labels_per_token):
                pred_labels.append(labels_per_token[label_idx])
            previous_word_idx = word_idx

        pred_tag_strings = [self.id2label[i] for i in pred_labels]
        return pred_tag_strings
