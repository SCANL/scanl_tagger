import re
import torch
from nltk import pos_tag
import nltk
from difflib import SequenceMatcher
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

# Make sure we have the same NLTK tagset
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('universal_tagset', quiet=True)

VOWELS = set("aeiou")
CONTEXT_MAP = {
    "FUNCTION": "@func",
    "PARAMETER": "@param",
    "ATTRIBUTE": "@attr",
    "DECLARATION": "@decl",
    "CLASS": "@class"
}


def detect_hungarian_prefix(first_token):
    m = re.match(r'^([a-zA-Z]{1,3})[A-Z_]', first_token)
    if m:
        return f"@hung_{m.group(1).lower()}"
    return "@hung_none"


def detect_digit_feature(tokens):
    for token in tokens:
        if any(char.isdigit() for char in token):
            return "@has_digit"
    return "@no_digit"


def consonant_vowel_ratio_bucket(tokens):
    def ratio(tok):
        tok_lower = tok.lower()
        num_vowels = sum(1 for c in tok_lower if c in VOWELS)
        num_consonants = sum(1 for c in tok_lower if c.isalpha() and c not in VOWELS)
        return num_consonants / (num_vowels + 1e-5)

    ratios = [ratio(tok) for tok in tokens if tok.isalpha()]
    if not ratios:
        return "@cvr_none"
    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio < 1.5:
        return "@cvr_low"
    elif avg_ratio < 3.0:
        return "@cvr_mid"
    else:
        return "@cvr_high"


def normalize_type(type_str):
    ts = type_str.strip().lower()
    ts = ts.replace("*", "_ptr")
    ts = ts.replace(" ", "_")
    return f"@{ts}"


def normalize_language(lang_str):
    return "@lang_" + lang_str.strip().lower().replace("++", "pp").replace("#", "sharp")


def system_prefix_similarity(first_token, system_name):
    if not first_token or not system_name:
        return "@sim_none"
    sys_lower = system_name.strip().lower()
    tok_lower = first_token.strip().lower()
    r = SequenceMatcher(None, tok_lower, sys_lower).ratio()
    if r > 0.9:
        return "@sim_high"
    elif r > 0.6:
        return "@sim_mid"
    elif r > 0.3:
        return "@sim_low"
    else:
        return "@sim_none"


class DistilBertTagger:
    def __init__(self, model_path: str):
        """
        Expects `model_path` to be a folder where the fine-tuned DistilBertForTokenClassification
        (and its tokenizer) have been saved via `trainer.save_model(...)` and `tokenizer.save_pretrained(...)`.
        """
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    def tag_identifier(self, tokens, context, type_str, language, system_name):
        """
        1) Build the “feature tokens + position tokens + identifier tokens” sequence
        2) Tokenize with `is_split_into_words=True`
        3) Run the model, take argmax over token logits
        4) Align via `word_ids()`, skipping:
              - Any word_id = None
              - Any word_id < 9 (because first 9 tokens were “feature tokens” => labels = -100)
              - Repeated word_ids (so we pick only the first sub-token of each “(pos, identifier‐word)” pair)
        5) Return a list of numeric labels.  (If you want strings, you can map via id2label externally.)
        """

        # 1. Re–compute exactly the same feature tokens as in training:
        context_token = CONTEXT_MAP.get(context.strip().upper(), "@unknown")
        system_token = f"@system_{system_name.strip().lower().replace(' ', '_')}"
        hungarian_token = detect_hungarian_prefix(tokens[0]) if tokens else "@hung_none"
        cvr_token = consonant_vowel_ratio_bucket(tokens)
        digit_token = detect_digit_feature(tokens)
        sim_token = system_prefix_similarity(tokens[0], system_name) if tokens else "@sim_none"
        type_token = normalize_type(type_str)
        lang_token = normalize_language(language)

        # Position tags for each identifier token
        length = len(tokens)
        if length == 1:
            pos_tokens = ["@pos_2"]
        else:
            pos_tokens = ["@pos_0"] + ["@pos_1"] * (length - 2) + ["@pos_2"]

        # NLTK POS feature
        nltk_tags = pos_tag(tokens, tagset="universal")
        universal_tags = [tag.lower() for _, tag in nltk_tags]
        nltk_feature = f"@nltk_{'-'.join(universal_tags)}"

        # Interleave pos_tokens + identifier tokens
        tokens_with_pos = [val for pair in zip(pos_tokens, tokens) for val in pair]

        # Build the full input token sequence (exactly what training saw):
        input_tokens = [
            context_token,
            system_token,
            hungarian_token,
            cvr_token,
            digit_token,
            sim_token,
            type_token,
            lang_token,
            nltk_feature
        ] + tokens_with_pos

        # 2. Tokenize
        encoded = self.tokenizer(
            input_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # 3. Inference
        with torch.no_grad():
            logits = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )[0]

        # 4. Take argmax, then align via word_ids()
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        word_ids = encoded.word_ids()

        pred_labels = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            # Skip if special token (None), or if it's part of the first 9 “feature tokens”
            if word_idx is None or word_idx < 9:
                continue
            # Skip if it’s the same word_idx as the previous (to avoid sub-token duplicates)
            if word_idx == previous_word_idx:
                continue

            pred_labels.append(predictions[idx])
            previous_word_idx = word_idx

        # Now, pred_labels is a list of numeric IDs (length == len(tokens)),
        # in the same order as your original “tokens” list.
        return pred_labels
