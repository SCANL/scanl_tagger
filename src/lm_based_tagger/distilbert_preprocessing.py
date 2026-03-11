import re
from difflib import SequenceMatcher
from typing import Iterable, List, Optional
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

AVAILABLE_FEATURES = [
    "context",
    "hungarian",
    "cvr",
    "digit",
    "digit_connector",
    "plural_suffix",
    "type",
    "type_overlap",
    "language",
    "sys_sim"
]

DEFAULT_FEATURES = list(AVAILABLE_FEATURES)

FEATURE_FUNCTIONS = {
    "context": lambda row, tokens: CONTEXT_MAP.get(row["CONTEXT"].strip().upper(), "@unknown"),
    "hungarian": lambda row, tokens: detect_hungarian_prefix(tokens[0]) if tokens else "@hung_none",
    "cvr": lambda row, tokens: consonant_vowel_ratio_bucket(tokens),
    "digit": lambda row, tokens: detect_digit_feature(tokens),
    "digit_connector": lambda row, tokens: get_digit_connector_feature_tokens(tokens),
    "plural_suffix": lambda row, tokens: get_plural_suffix_feature_tokens(tokens),
    "type": lambda row, tokens: get_type_feature_tokens(row.get("TYPE", "")),
    "type_overlap": lambda row, tokens: get_type_overlap_feature_tokens(tokens, row.get("TYPE", "")),
    "language": lambda row, tokens: normalize_language(row.get("LANGUAGE", "")),
    "sys_sim": lambda row, tokens: get_system_overlap_feature_tokens(tokens, row.get("SYSTEM_NAME", "")),
}

TYPE_BOOL_TOKENS = {"bool", "boolean", "glboolean", "boolean_t"}
TYPE_VOID_TOKENS = {"void"}
TYPE_INT_TOKENS = {
    "int", "integer", "byte", "word", "dword", "qword", "short", "long",
    "size_t", "ssize_t", "ptrdiff_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "u8", "u16", "u32", "u64",
    "i8", "i16", "i32", "i64", "usize", "isize"
}
TYPE_FLOAT_TOKENS = {"float", "double", "decimal", "real", "numeric", "number"}
TYPE_CHAR_TOKENS = {"char", "wchar", "tchar", "char16_t", "char32_t", "rune"}
TYPE_STRING_TOKENS = {"string", "str", "text", "utf8", "utf16", "utf32", "cstring"}
TYPE_CONTAINER_TOKENS = {
    "list", "array", "vector", "map", "set", "queue", "stack", "deque",
    "collection", "iterator", "iterable", "stream", "buffer", "slice"
}
TYPE_FUNCTION_TOKENS = {"func", "function", "callback", "predicate", "consumer", "supplier", "runnable", "lambda"}
TYPE_QUALIFIER_TOKENS = {"const", "volatile", "static", "signed", "unsigned", "struct", "class", "enum", "union"}
NON_PLURAL_S_SUFFIX_STOPLIST = {
    "this", "is", "was", "has", "plus", "minus", "pass", "class", "glass",
    "status", "alias", "bias", "canvas", "http", "https"
}


def _looks_int_like(normalized_type: str, type_tokens: List[str]) -> bool:
    if any(token in TYPE_INT_TOKENS for token in type_tokens):
        return True
    if any(re.fullmatch(r"(?:u|i)?int\d+", token) for token in type_tokens):
        return True
    return bool(re.search(r"(?:^|\W)(?:u?_?int\d*_t|int\d*_t|size_t|ssize_t|ptrdiff_t)(?:$|\W)", normalized_type))

def normalize_selected_features(selected_features: Optional[Iterable[str]] = None) -> List[str]:
    """
    Validate and canonicalize a selected feature list.

    Args:
        selected_features: Iterable of feature names, or ``None`` to use all features.

    Returns:
        A feature list ordered according to ``AVAILABLE_FEATURES``.
    """
    if selected_features is None:
        return list(DEFAULT_FEATURES)

    selected_list = [feature.strip() for feature in selected_features if str(feature).strip()]
    selected_set = set(selected_list)
    invalid_features = sorted(selected_set - set(AVAILABLE_FEATURES))
    if invalid_features:
        raise ValueError(
            "Unknown lm_based feature(s): "
            f"{', '.join(invalid_features)}. "
            f"Available features: {', '.join(AVAILABLE_FEATURES)}"
        )

    return [feature for feature in AVAILABLE_FEATURES if feature in selected_set]


def get_feature_tokens(row, tokens, selected_features: Optional[Iterable[str]] = None):
    active_features = normalize_selected_features(selected_features)
    feature_tokens = []
    for feature in active_features:
        raw_value = FEATURE_FUNCTIONS[feature](row, tokens)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            if raw_value:
                feature_tokens.append(raw_value)
            continue
        feature_tokens.extend(str(value) for value in raw_value if str(value))
    return feature_tokens


def get_number_of_features(selected_features: Optional[Iterable[str]] = None) -> int:
    return len(normalize_selected_features(selected_features))


def build_model_input_tokens(row, tokens, selected_features: Optional[Iterable[str]] = None):
    feature_tokens = get_feature_tokens(row, tokens, selected_features)

    length = len(tokens)
    pos_tokens = ["@pos_2"] if length == 1 else ["@pos_0"] + ["@pos_1"] * (length - 2) + ["@pos_2"]
    tokens_with_pos = [value for pair in zip(pos_tokens, tokens) for value in pair]
    return feature_tokens + tokens_with_pos, len(feature_tokens)

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


def _is_alpha_like_token(token: str) -> bool:
    return bool(token) and token.isalpha()


def get_digit_connector_feature_tokens(tokens: List[str]) -> List[str]:
    feature_tokens = []

    for index, token in enumerate(tokens):
        normalized = token.strip().lower()
        if normalized != "2":
            continue

        feature_tokens.append("@digit_connector_2")

        if index == 0:
            feature_tokens.append("@digit_connector_head")
        elif index == len(tokens) - 1:
            feature_tokens.append("@digit_connector_tail")
        else:
            feature_tokens.append("@digit_connector_mid")

        prev_is_alpha = index > 0 and _is_alpha_like_token(tokens[index - 1])
        next_is_alpha = index + 1 < len(tokens) and _is_alpha_like_token(tokens[index + 1])
        if prev_is_alpha and next_is_alpha:
            feature_tokens.append("@digit_connector_alpha_bridge")
        elif prev_is_alpha or next_is_alpha:
            feature_tokens.append("@digit_connector_alpha_adjacent")

    if not feature_tokens:
        return ["@digit_connector_none"]
    return _dedupe_preserve_order(feature_tokens)


def _plural_suffix_bucket(token: str) -> str | None:
    normalized = token.strip().lower()
    if len(normalized) < 4 or not normalized.isalpha():
        return None
    if normalized in NON_PLURAL_S_SUFFIX_STOPLIST:
        return None

    if normalized.endswith("ies") and len(normalized) > 4:
        return "ies"
    if normalized.endswith(("ses", "xes", "zes", "ches", "shes", "oes")):
        return "es"
    if normalized.endswith("es") and not normalized.endswith(("sses", "uses", "ises")):
        return "es"
    if normalized.endswith("s") and not normalized.endswith(("ss", "us", "is")):
        return "s"
    return None


def get_plural_suffix_feature_tokens(tokens: List[str]) -> List[str]:
    suffixes = []
    positions = []

    for index, token in enumerate(tokens):
        suffix = _plural_suffix_bucket(token)
        if suffix is None:
            continue
        suffixes.append(suffix)
        if index == len(tokens) - 1:
            positions.append("tail")
        elif index == 0:
            positions.append("head")
        else:
            positions.append("mid")

    if not suffixes:
        return ["@plural_suffix_none"]

    feature_tokens = ["@plural_suffix_present"]
    feature_tokens.extend(f"@plural_suffix_{suffix}" for suffix in _dedupe_preserve_order(suffixes))
    feature_tokens.extend(f"@plural_suffix_{position}" for position in _dedupe_preserve_order(positions))
    return feature_tokens

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


def _normalize_text_for_split(text: str) -> str:
    text = re.sub(r"::|->|<|>|\(|\)|\[|\]|,|\*|&|/|\\", " ", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-.]+", " ", text)
    return text


def _split_identifier_like_text(text: str) -> List[str]:
    if not text:
        return []
    normalized = _normalize_text_for_split(str(text))
    return [part.lower() for part in re.findall(r"[A-Za-z]+\d*|\d+[A-Za-z]*", normalized) if part]


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _token_match(identifier_token: str, lexicon_token: str) -> str | None:
    left = identifier_token.strip().lower()
    right = lexicon_token.strip().lower()
    if not left or not right:
        return None
    if left == right:
        return "exact"
    if len(left) >= 2 and len(right) >= 2 and (left.startswith(right) or right.startswith(left)):
        return "prefix"
    return None


def _count_any_overlap(identifier_tokens: List[str], lexicon_tokens: List[str]) -> int:
    count = 0
    for identifier_token in identifier_tokens:
        if any(_token_match(identifier_token, lexicon_token) for lexicon_token in lexicon_tokens):
            count += 1
    return count


def _count_leading_overlap(identifier_tokens: List[str], lexicon_tokens: List[str]) -> int:
    span = 0
    for identifier_token in identifier_tokens:
        if any(_token_match(identifier_token, lexicon_token) for lexicon_token in lexicon_tokens):
            span += 1
        else:
            break
    return span


def _bucket_overlap_count(count: int, prefix: str) -> str:
    if count <= 0:
        return f"@{prefix}_0"
    if count == 1:
        return f"@{prefix}_1"
    return f"@{prefix}_2plus"


def _get_head_match_token(identifier_tokens: List[str], lexicon_tokens: List[str], prefix: str) -> str:
    if not identifier_tokens or not lexicon_tokens:
        return f"@{prefix}_head_none"

    head = identifier_tokens[0]
    match_strength = None
    for lexicon_token in lexicon_tokens:
        match_strength = _token_match(head, lexicon_token)
        if match_strength == "exact":
            break

    if match_strength == "exact":
        return f"@{prefix}_head_exact"
    if match_strength == "prefix":
        return f"@{prefix}_head_prefix"
    return f"@{prefix}_head_none"


def _normalize_type_for_bucketing(type_str: str) -> str:
    return str(type_str or "").strip().lower()


def _normalize_type_tokens(type_str: str) -> List[str]:
    raw_tokens = _split_identifier_like_text(type_str)
    return [token for token in raw_tokens if token not in TYPE_QUALIFIER_TOKENS]


def get_type_feature_tokens(type_str: str) -> List[str]:
    normalized_type = _normalize_type_for_bucketing(type_str)
    type_tokens = _normalize_type_tokens(type_str)
    feature_tokens = []

    if not normalized_type:
        return ["@type_unknown"]

    if "*" in normalized_type:
        feature_tokens.append("@type_ptr")
    if "&" in normalized_type:
        feature_tokens.append("@type_ref")
    if "[]" in normalized_type or "array" in type_tokens:
        feature_tokens.append("@type_array")
    if "<" in normalized_type and ">" in normalized_type:
        feature_tokens.append("@type_generic")
    if "const" in normalized_type:
        feature_tokens.append("@type_const")
    if "unsigned" in normalized_type:
        feature_tokens.append("@type_unsigned")
    if "signed" in normalized_type:
        feature_tokens.append("@type_signed")

    bucket = "@type_object_like"
    if any(token in TYPE_BOOL_TOKENS for token in type_tokens):
        bucket = "@type_bool"
    elif any(token in TYPE_VOID_TOKENS for token in type_tokens):
        bucket = "@type_void"
    elif any(token in TYPE_CONTAINER_TOKENS for token in type_tokens):
        bucket = "@type_container_like"
    elif any(token in TYPE_STRING_TOKENS for token in type_tokens):
        bucket = "@type_string_like"
    elif any(token in TYPE_CHAR_TOKENS for token in type_tokens):
        bucket = "@type_char_like"
    elif any(token in TYPE_FLOAT_TOKENS for token in type_tokens):
        bucket = "@type_float_like"
    elif _looks_int_like(normalized_type, type_tokens):
        bucket = "@type_int_like"
    elif "enum" in normalized_type:
        bucket = "@type_enum_like"
    elif any(token in TYPE_FUNCTION_TOKENS for token in type_tokens):
        bucket = "@type_function_like"

    feature_tokens.insert(0, bucket)
    return _dedupe_preserve_order(feature_tokens)


def get_type_overlap_feature_tokens(identifier_tokens: List[str], type_str: str) -> List[str]:
    normalized_identifier_tokens = [token.lower() for token in identifier_tokens if token]
    type_tokens = _normalize_type_tokens(type_str)
    if not type_tokens:
        return ["@typeov_head_none", "@typeov_lead_0", "@typeov_any_0"]

    leading_overlap = _count_leading_overlap(normalized_identifier_tokens, type_tokens)
    any_overlap = _count_any_overlap(normalized_identifier_tokens, type_tokens)
    return [
        _get_head_match_token(normalized_identifier_tokens, type_tokens, "typeov"),
        _bucket_overlap_count(leading_overlap, "typeov_lead"),
        _bucket_overlap_count(any_overlap, "typeov_any"),
    ]


def get_system_overlap_feature_tokens(identifier_tokens: List[str], system_name: str) -> List[str]:
    normalized_identifier_tokens = [token.lower() for token in identifier_tokens if token]
    system_tokens = _split_identifier_like_text(system_name)
    if not system_tokens:
        return ["@sys_head_none", "@sys_lead_0", "@sys_any_0", "@sim_none"]

    leading_overlap = _count_leading_overlap(normalized_identifier_tokens, system_tokens)
    any_overlap = _count_any_overlap(normalized_identifier_tokens, system_tokens)
    similarity_token = system_prefix_similarity(normalized_identifier_tokens[0] if normalized_identifier_tokens else "", system_name)
    return [
        _get_head_match_token(normalized_identifier_tokens, system_tokens, "sys"),
        _bucket_overlap_count(leading_overlap, "sys_lead"),
        _bucket_overlap_count(any_overlap, "sys_any"),
        similarity_token,
    ]

def normalize_language(lang_str):
    return "@lang_" + lang_str.strip().lower().replace("++", "pp").replace("#", "sharp")

def prepare_dataset(
    df: pd.DataFrame,
    label2id: dict,
    selected_features: Optional[Iterable[str]] = None,
):
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
    active_features = normalize_selected_features(selected_features)

    rows = []
    for _, row in df.iterrows():
        tokens = row["tokens"]
        tags = row["tags"]
        full_tokens, num_features = build_model_input_tokens(row, tokens, active_features)
        ner_tags_with_pos = [val for tag in tags for val in (-100, label2id[tag])]
        full_labels = [-100] * num_features + ner_tags_with_pos

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
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special tokens (CLS, SEP, PAD)
            labels.append(-100)
        elif word_id == prev_word_id:
            # Continuation subword token (e.g. ##ployee) — ignore in loss/metrics
            labels.append(-100)
        elif word_id < len(sample["ner_tags"]):
            labels.append(sample["ner_tags"][word_id])
        else:
            labels.append(-100)
        prev_word_id = word_id

    tokenized["labels"] = labels
    return tokenized
