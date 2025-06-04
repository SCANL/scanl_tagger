# distilbert_preprocessing.py

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


def detect_hungarian_prefix(first_token):
    """
    If the first token starts with 1–3 letters followed by an uppercase or underscore,
    return "@hung_<prefix>". Otherwise "@hung_none".
    """
    m = re.match(r'^([a-zA-Z]{1,3})[A-Z_]', first_token)
    if m:
        return f"@hung_{m.group(1).lower()}"
    return "@hung_none"


def detect_digit_feature(tokens):
    """
    If any token has a digit, return "@has_digit", else "@no_digit".
    """
    for token in tokens:
        if any(char.isdigit() for char in token):
            return "@has_digit"
    return "@no_digit"


def consonant_vowel_ratio_bucket(tokens):
    """
    Compute the average consonant/vowel ratio across all alphabetic tokens,
    then bucket into low/mid/high.
    """
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
    """
    Compute a SequenceMatcher ratio against the system name, then bucket:
      >0.9 ➔ "@sim_high", >0.6 ➔ "@sim_mid", >0.3 ➔ "@sim_low", else "@sim_none".
    """
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


def prepare_dataset(df: pd.DataFrame, label2id: dict):
    """
    Takes a DataFrame with columns:
       - "tokens"      : List[str] (split identifier)
       - "tags"        : List[str] (gold PoS tags, same length as tokens)
       - "CONTEXT"     : e.g. "FUNCTION", "PARAMETER", etc.
       - "SYSTEM_NAME" : string

    Returns a HuggingFace `datasets.Dataset` with two fields:
       - "tokens"   : List[List[str]]  (the FULL token sequence, including exactly 7 feature tokens + position tokens + identifier tokens)
       - "ner_tags" : List[List[int]]  (the aligned label IDs, with -100 in front for each feature token)
    """
    rows = []
    for _, row in df.iterrows():
        tokens = row["tokens"]
        tags = row["tags"]

        # 1. Build 7 feature tokens (context, system, hungarian, cvr, digit, sim, nltk)
        context_token   = CONTEXT_MAP.get(row["CONTEXT"].strip().upper(), "@unknown")
        system_token    = f"@system_{row['SYSTEM_NAME'].strip().lower().replace(' ', '_')}"
        hungarian_token = detect_hungarian_prefix(tokens[0]) if tokens else "@hung_none"
        cvr_token       = consonant_vowel_ratio_bucket(tokens)
        digit_token     = detect_digit_feature(tokens)
        sim_token       = system_prefix_similarity(tokens[0], row["SYSTEM_NAME"]) if tokens else "@sim_none"

        # 2. NLTK POS tags (universal tagset)
        nltk_tags = pos_tag(tokens, tagset="universal")
        universal_tags = [tag.lower() for _, tag in nltk_tags]
        nltk_feature = f"@nltk_{'-'.join(universal_tags)}"

        # 3. Position tags: interleave with identifier tokens
        length = len(tokens)
        if length == 1:
            pos_tokens = ["@pos_2"]
        else:
            pos_tokens = ["@pos_0"] + ["@pos_1"] * (length - 2) + ["@pos_2"]
        tokens_with_pos = [val for pair in zip(pos_tokens, tokens) for val in pair]

        # 4. Build the “full” token list (7 feature tokens + 2*len(tokens) position‐identifier tokens)
        full_tokens = [
            context_token,
            system_token,
            hungarian_token,
            cvr_token,
            digit_token,
            sim_token,
            nltk_feature,
        ] + tokens_with_pos

        # 5. Build the aligned labels array:
        #    - First 7 entries → -100 (because they are feature tokens)
        #    - Then for each identifier token, [-100, label2id[tag]]
        ner_tags_with_pos = [val for tag in tags for val in (-100, label2id[tag])]
        full_labels = [0] * 7 + ner_tags_with_pos      # ← use 0, not -100

        rows.append({
            "tokens":   full_tokens,
            "ner_tags": full_labels
        })

    return Dataset.from_dict({
        "tokens":   [r["tokens"]   for r in rows],
        "ner_tags": [r["ner_tags"] for r in rows]
    })


def tokenize_and_align_labels(example, tokenizer):
    """
    example: a dict with
      - "tokens"   : List[str] (the full token sequence, including exactly 7 feature tokens)
      - "ner_tags" : List[int] (same length as above)

    We run `tokenizer(example["tokens"], is_split_into_words=True, truncation=True)`,
    then align `word_ids()` with `example["ner_tags"]`
    """
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
            # Just in case of truncation
            labels.append(-100)

    tokenized["labels"] = labels
    return tokenized
