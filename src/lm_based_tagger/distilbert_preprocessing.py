import re
from difflib import SequenceMatcher
import pandas as pd
from datasets import Dataset

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
]

FEATURE_FUNCTIONS = {
    "context": lambda row, tokens: CONTEXT_MAP.get(row["CONTEXT"].strip().upper(), "@unknown"),
    "hungarian": lambda row, tokens: detect_hungarian_prefix(tokens[0]) if tokens else "@hung_none",
    "cvr": lambda row, tokens: consonant_vowel_ratio_bucket(tokens),
    "digit": lambda row, tokens: detect_digit_feature(tokens),
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
    """
    Converts a DataFrame of identifier tokens and grammar tags into a HuggingFace Dataset
    formatted for NER training with feature and position tokens.

    Each row in the input DataFrame should contain:
        - tokens: List[str] (e.g., ['get', 'Employee', 'Name'])
        - tags:   List[str] (e.g., ['V', 'NM', 'N'])
        - CONTEXT: str (e.g., 'function')

    The function adds:
        - Feature tokens: ['@hung_get', '@no_digit', '@cvr_mid', '@func']
        - Interleaved position and real tokens:
            ['@pos_0', 'get', '@pos_1', 'Employee', '@pos_2', 'Name']

    The NER tags are aligned so that:
        - Feature tokens and position markers get label -100 (ignored in loss)
        - Real tokens are converted from grammar tags using `label2id`

    Example Input:
        df = pd.DataFrame([{
            "tokens": ["get", "Employee", "Name"],
            "tags": ["V", "NM", "N"],
            "CONTEXT": "function"
        }])

    Example Output:
        Dataset with:
            tokens:    ['@hung_get', '@no_digit', '@cvr_mid', '@func',
                        '@pos_0', 'get', '@pos_1', 'Employee', '@pos_2', 'Name']
            ner_tags:  [-100, -100, -100, -100,
                        -100, 1, -100, 2, -100, 3]  # assuming label2id = {"V": 1, "NM": 2, "N": 3}
    """
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

def tokenize_and_align_labels(sample, tokenizer):
    """
    Tokenizes an example and aligns NER labels with subword tokens.

    The input `example` comes from `prepare_dataset()` and contains:
        - tokens: List[str], including feature and position tokens
        - ner_tags: List[int], aligned with `tokens`, with -100 for ignored tokens

    This function:
        - Uses `is_split_into_words=True` to tokenize each item in `tokens`
        - Uses `tokenizer.word_ids()` to map each subword back to its original token index
        - Assigns the corresponding label (or -100) for each subword token

    Example Input:
        example = {
            "tokens": ['@hung_get', '@no_digit', '@cvr_mid', '@func',
                       '@pos_0', 'get', '@pos_1', 'Employee', '@pos_2', 'Name'],
            "ner_tags": [-100, -100, -100, -100,
                         -100, 1, -100, 2, -100, 3]
        }

    Assuming 'Employee' is tokenized to ['Em', '##ployee'],
    Example Output:
        tokenized["labels"] = [-100, -100, -100, -100,
                               -100, 1, -100, 2, 2, -100, 3]
    """
    tokenized = tokenizer(
        sample["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    word_ids = tokenized.word_ids()

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id < len(sample["ner_tags"]):
            labels.append(sample["ner_tags"][word_id])
        else:
            labels.append(-100)

    tokenized["labels"] = labels
    return tokenized
