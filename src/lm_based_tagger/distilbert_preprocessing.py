import re
from nltk import pos_tag
import nltk
from difflib import SequenceMatcher
import pandas as pd
from datasets import Dataset

# Download once (we’ll just do it quietly here)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('universal_tagset', quiet=True)

# === Constants ===
VOWELS = set("aeiou")
LOW_FREQ_TAGS = {"CJ", "VM", "PRE", "V"}

# Map of context strings ➔ “feature tokens”
CONTEXT_MAP = {
    "FUNCTION": "@func",
    "PARAMETER": "@param",
    "ATTRIBUTE": "@attr",
    "DECLARATION": "@decl",
    "CLASS": "@class"
}

FEATURES = [
    "context",
    "hungarian",
    "cvr",
    "digit",
    "nltk"
]

FEATURE_FUNCTIONS = {
    "context": lambda row, tokens: CONTEXT_MAP.get(row["CONTEXT"].strip().upper(), "@unknown"),
    "hungarian": lambda row, tokens: detect_hungarian_prefix(tokens[0]) if tokens else "@hung_none",
    "cvr": lambda row, tokens: consonant_vowel_ratio_bucket(tokens),
    "digit": lambda row, tokens: detect_digit_feature(tokens),
    "nltk": lambda row, tokens: "@nltk_" + '-'.join(tag.lower() for _, tag in pos_tag(tokens, tagset="universal"))
}

def get_feature_tokens(row, tokens):
    return [FEATURE_FUNCTIONS[feat](row, tokens) for feat in FEATURES]

NUMBER_OF_FEATURES = len(FEATURES)

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

def normalize_type(type_str):
    ts = type_str.strip().lower()
    ts = ts.replace("*", "_ptr")
    ts = ts.replace(" ", "_")
    return f"@{ts}"

def normalize_language(lang_str):
    return "@lang_" + lang_str.strip().lower().replace("++", "pp").replace("#", "sharp")

def prepare_dataset(df: pd.DataFrame, label2id: dict):
    rows = []
    for _, row in df.iterrows():
        tokens = row["tokens"]
        tags = row["tags"]
        feature_tokens = get_feature_tokens(row, tokens)

        length = len(tokens)
        pos_tokens = ["@pos_2"] if length == 1 else ["@pos_0"] + ["@pos_1"] * (length - 2) + ["@pos_2"]
        tokens_with_pos = [val for pair in zip(pos_tokens, tokens) for val in pair]

        full_tokens = feature_tokens + tokens_with_pos
        ner_tags_with_pos = [val for tag in tags for val in (-100, label2id[tag])]
        full_labels = [-100] * NUMBER_OF_FEATURES + ner_tags_with_pos

        rows.append({
            "tokens": full_tokens,
            "ner_tags": full_labels
        })

    return Dataset.from_dict({
        "tokens": [r["tokens"] for r in rows],
        "ner_tags": [r["ner_tags"] for r in rows]
    })

def tokenize_and_align_labels(example, tokenizer):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    word_ids = tokenized.word_ids()

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id < len(example["ner_tags"]):
            labels.append(example["ner_tags"][word_id])
        else:
            labels.append(-100)

    tokenized["labels"] = labels
    return tokenized
