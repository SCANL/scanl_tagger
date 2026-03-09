import os
import time
import random
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .distilbert_crf import DistilBertCRFForTokenClassification

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

from transformers import (
    Trainer,
    TrainingArguments,
    DistilBertTokenizerFast,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

from datasets import Dataset
from src.lm_based_tagger.distilbert_preprocessing import (
    AVAILABLE_FEATURES,
    normalize_selected_features,
    prepare_dataset,
    tokenize_and_align_labels,
)

# If CUDA is available, use it; otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def _configure_torch_runtime() -> None:
    """Enable faster CUDA execution settings when reproducibility constraints allow it."""
    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

# === Random Seeds ===
RAND_STATE = 209
random.seed(RAND_STATE)
np.random.seed(RAND_STATE)
torch.manual_seed(RAND_STATE)
torch.cuda.manual_seed_all(RAND_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_configure_torch_runtime()

# === Hyperparameters / Config ===
K = 5                     # number of CV folds
HOLDOUT_RATIO = 0.30      # 30% held out for final evaluation
EPOCHS = 5            # number of epochs per fold
EARLY_STOP = 2            # patience for early stopping
LOW_FREQ_TAGS = {"CJ", "VM", "PRE", "V"}

# Curated mapping: common programming verbs → synonyms that are also clearly verbs.
# Only words that are rarely ambiguous as NM/N in identifier naming are included.
# Used by _augment_verb_examples() to synthesise new V-tagged training rows.
VERB_SYNONYMS: Dict[str, List[str]] = {
    # Read / Load / Fetch
    "read":        ["load", "fetch", "parse"],
    "load":        ["read", "fetch", "import"],
    "fetch":       ["get", "retrieve", "load"],
    "get":         ["fetch", "retrieve", "obtain"],
    "parse":       ["read", "decode", "extract"],
    "retrieve":    ["fetch", "get", "load"],
    "import":      ["load", "read", "fetch"],
    # Write / Save / Store
    "write":       ["save", "store", "output"],
    "save":        ["write", "store", "persist"],
    "store":       ["save", "write", "persist"],
    "output":      ["write", "emit", "print"],
    "export":      ["save", "write", "output"],
    "emit":        ["send", "dispatch", "publish"],
    # Compute
    "compute":     ["calculate", "evaluate", "process"],
    "calculate":   ["compute", "evaluate", "solve"],
    "calc":        ["compute", "calculate", "evaluate"],
    "evaluate":    ["compute", "calculate", "check"],
    "process":     ["compute", "handle", "transform"],
    "solve":       ["compute", "calculate", "resolve"],
    # Transform / Convert
    "convert":     ["transform", "encode", "decode"],
    "transform":   ["convert", "encode", "map"],
    "encode":      ["convert", "serialize", "pack"],
    "decode":      ["parse", "convert", "deserialize"],
    "serialize":   ["encode", "convert", "write"],
    "deserialize": ["decode", "parse", "read"],
    "transpose":   ["transform", "convert", "rotate"],
    "invert":      ["negate", "reverse", "flip"],
    # Search / Filter / Sort
    "find":        ["search", "locate", "lookup"],
    "search":      ["find", "scan", "lookup"],
    "filter":      ["select", "prune", "screen"],
    "sort":        ["order", "arrange", "rank"],
    "select":      ["filter", "choose", "pick"],
    "scan":        ["search", "parse", "traverse"],
    # Create / Build
    "create":      ["build", "make", "generate"],
    "build":       ["create", "make", "construct"],
    "make":        ["create", "build", "generate"],
    "generate":    ["create", "produce", "build"],
    "compile":     ["build", "assemble"],
    "clone":       ["copy", "duplicate", "replicate"],
    # CRUD
    "add":         ["insert", "append", "attach"],
    "insert":      ["add", "append", "push"],
    "remove":      ["delete", "detach", "erase"],
    "delete":      ["remove", "erase", "purge"],
    "update":      ["modify", "refresh", "change"],
    "modify":      ["update", "change", "alter"],
    "change":      ["update", "modify", "alter"],
    "set":         ["assign", "update", "put"],
    # Init / Setup / Clear
    "init":        ["initialize", "setup", "create"],
    "initialize":  ["init", "setup", "create"],
    "setup":       ["init", "initialize", "configure"],
    "configure":   ["setup", "initialize", "set"],
    "reset":       ["clear", "reinitialize"],
    "clear":       ["reset", "flush", "purge"],
    # Messaging / Events
    "send":        ["transmit", "dispatch", "publish"],
    "dispatch":    ["send", "emit", "route"],
    "publish":     ["emit", "send", "broadcast"],
    "receive":     ["accept", "collect", "handle"],
    "handle":      ["process", "manage", "respond"],
    "trigger":     ["invoke", "dispatch", "emit"],
    # Render / Display
    "render":      ["draw", "display", "paint"],
    "draw":        ["render", "paint", "display"],
    "display":     ["render", "show", "print"],
    "print":       ["output", "display", "log"],
    "show":        ["display", "render", "print"],
    "preview":     ["show", "display", "render"],
    # Validate
    "validate":    ["check", "verify", "assert"],
    "check":       ["validate", "verify", "test"],
    "verify":      ["validate", "check", "confirm"],
    "test":        ["check", "validate", "verify"],
    # Execute / Run
    "run":         ["execute", "invoke", "call"],
    "execute":     ["run", "invoke", "call"],
    "invoke":      ["call", "run", "execute"],
    "call":        ["invoke", "execute", "run"],
    "start":       ["begin", "launch", "run"],
    "stop":        ["end", "halt", "terminate"],
    "launch":      ["start", "run", "execute"],
    # Connect / Register
    "open":        ["connect", "launch"],
    "close":       ["disconnect", "terminate"],
    "connect":     ["link", "attach", "bind"],
    "disconnect":  ["unlink", "detach", "unbind"],
    "register":    ["add", "bind", "attach"],
    "unregister":  ["remove", "detach", "deregister"],
    "attach":      ["connect", "link", "bind"],
    "detach":      ["disconnect", "unlink", "unbind"],
    "link":        ["connect", "attach", "bind"],
    # IO / Streams
    "stream":      ["transmit", "pipe", "transfer"],
    "echo":        ["print", "output", "display"],
    "log":         ["record", "trace", "output"],
    # Misc
    "copy":        ["clone", "duplicate", "replicate"],
    "move":        ["transfer", "relocate"],
    "merge":       ["combine", "join", "concat"],
    "split":       ["divide", "separate", "partition"],
    "compare":     ["match", "check", "evaluate"],
    "hash":        ["digest", "compute"],
    "format":      ["serialize", "convert", "encode"],
    "compress":    ["pack", "encode"],
    "decompress":  ["unpack", "decode"],
    "upload":      ["send", "push", "transfer"],
    "download":    ["fetch", "pull", "retrieve"],
    "backup":      ["copy", "archive", "save"],
    "restore":     ["recover", "reload"],
    "release":     ["free", "dispose"],
    "free":        ["release", "dispose"],
    "allocate":    ["create", "reserve"],
    "resize":      ["scale", "adjust"],
    "swap":        ["exchange", "replace"],
    "defragment":  ["compact", "reorganize"],
    "downsample":  ["reduce", "scale"],
    "aggregate":   ["collect", "combine", "merge"],
    "reload":      ["refresh", "restart", "reinitialize"],
    "refresh":     ["reload", "update", "redraw"],
    "sync":        ["synchronize", "update", "align"],
    "synchronize": ["sync", "update", "align"],
    "zip":         ["compress", "pack"],
    "unzip":       ["decompress", "unpack"],
    "encrypt":     ["encode", "protect"],
    "decrypt":     ["decode", "decipher"],
}


def _augment_verb_examples(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """
    Generate synthetic training rows by substituting V-tagged tokens with synonyms.

    For each row containing at least one V-tagged token whose lowercase form appears
    in VERB_SYNONYMS, produce up to MAX_SYNS new rows where that token is replaced
    by a randomly sampled synonym.  The rest of the row (context, tags, other tokens)
    is unchanged, so the new rows are valid training examples.

    Applied across all contexts (FUNCTION, PARAMETER, ATTRIBUTE, DECLARATION, CLASS)
    to improve V recall outside the previously-dominant FUNCTION context.

    Args:
        df:  Fold training DataFrame with 'tokens' (List[str]) and 'tags' (List[str]).
        rng: Seeded Random instance for reproducibility.

    Returns:
        DataFrame of newly-synthesised rows (may be empty if nothing is substitutable).
    """
    MAX_SYNS = 2  # max new synthetic examples per original row

    new_rows = []
    for _, row in df.iterrows():
        tokens: List[str] = list(row["tokens"])
        tags:   List[str] = list(row["tags"])

        # Positions that are V-tagged and whose lowercase token has synonyms
        substitutable = [
            (i, tokens[i])
            for i, tag in enumerate(tags)
            if tag == "V" and tokens[i].lower() in VERB_SYNONYMS
        ]
        if not substitutable:
            continue

        generated = 0
        for pos, token in substitutable:
            if generated >= MAX_SYNS:
                break
            synonyms = VERB_SYNONYMS[token.lower()]
            # Preserve original casing style (Title-case → Title-case, lower → lower)
            capitalize = token[0].isupper() if token else False
            n_to_sample = min(MAX_SYNS - generated, len(synonyms))
            sampled = rng.sample(synonyms, n_to_sample)
            for syn in sampled:
                if generated >= MAX_SYNS:
                    break
                syn_tok = syn[0].upper() + syn[1:] if capitalize else syn
                new_tokens = tokens.copy()
                new_tokens[pos] = syn_tok
                new_row = row.copy()
                new_row["tokens"] = new_tokens
                new_rows.append(new_row)
                generated += 1

    if not new_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(new_rows)

# === Label List & Mappings ===
LABEL_LIST = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL   = {i: label for label, i in LABEL2ID.items()}

def dual_print(*args, file, **kwargs):
    print(*args, **kwargs)         # stdout
    print(*args, file=file, **kwargs)  # file


def _write_run_metadata(
    file,
    source_names: List[str],
    selected_features: List[str],
    run_metadata: Dict[str, Any] | None = None,
):
    """Write reproducibility metadata for the current LM training run."""
    run_metadata = run_metadata or {}
    cli_options = run_metadata.get("cli_options", {})

    dual_print("\nRun Configuration:", file=file)
    dual_print(f"Command: {run_metadata.get('command', '<unknown>')}", file=file)
    dual_print(f"Seed: {RAND_STATE}", file=file)
    dual_print(f"Datasets: {', '.join(source_names)}", file=file)
    dual_print(
        f"Features: {', '.join(selected_features) if selected_features else '<none>'}",
        file=file,
    )
    dual_print("CLI options:", file=file)
    for key, value in cli_options.items():
        dual_print(f"  {key}: {value}", file=file)


# 11) compute_metrics function (macro-F1) 
def compute_metrics(eval_pred):
    """
    Computes macro-F1, token-level accuracy, and identifier-level accuracy.

    Supports both:
    - Raw logits from the model (shape [B, T, C])
    - Viterbi-decoded label paths from CRF models (List[List[int]])

    Args:
        eval_pred: Either a tuple (preds, labels) or a HuggingFace EvalPrediction object.
                   `preds` can be:
                       • [B, T, C] logits (e.g., output of a classifier head)
                       • [B, T] label IDs
                       • List[List[int]] variable-length decoded paths (CRF)

    Returns:
        dict with:
            - "eval_macro_f1": F1 averaged over classes (not tokens)
            - "eval_token_accuracy": token-level accuracy (ignores -100)
            - "eval_identifier_accuracy": percentage of rows where all tokens matched

    Example (logits of shape [B=2, T=3, C=4]):
        preds = np.array([
            [  # Example 1 (B=0)
                [0.1, 2.5, 0.3, -1.0],  # Token 1 → class 1 (NM)
                [1.5, 0.4, 0.2, -0.5], # Token 2 → class 0 (V)
                [0.3, 0.1, 3.2, 0.0],  # Token 3 → class 2 (N)
            ],
            [  # Example 2 (B=1)
                [0.2, 0.1, 0.4, 2.1],  # Token 1 → class 3 (P)
                [0.9, 1.0, 0.3, 0.0],  # Token 2 → class 1 (NM)
                [1.1, 1.1, 1.1, 1.1],  # Token 3 → tie (say model picks class 0)
            ]
        ])

        Converted via argmax(preds, axis=-1):
            → [[1, 0, 2],  # Example 1 predictions
               [3, 1, 0]]  # Example 2 predictions

        Gold:  [V, NM, N]    → label_row = [-100, 1, -100, 2, -100, 3]
        Pred:  [V, NM, N]    → pred_row  =         [1,        2,       3]
        All tokens match → example_correct = True
    """
    # 1) Extract predictions and labels
    if isinstance(eval_pred, tuple):  # older HuggingFace versions
        preds, labels = eval_pred
    else:  # EvalPrediction object
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

    # 2) Normalize predictions format
    # Convert [B, T, C] logits → [B, T] class IDs
    if isinstance(preds, np.ndarray) and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    # Convert CRF list-of-lists → numpy object array
    elif isinstance(preds, list):
        preds = np.array(preds, dtype=object)

    # 3) Compare predictions to labels, ignoring -100
    all_true, all_pred, id_correct_flags = [], [], []

    for pred_row, label_row in zip(preds, labels):
        example_correct = True

        for i, lbl in enumerate(label_row):   # iterate gold labels with position index
            if lbl == -100:                   # skip padding / specials
                continue

            # Use the same position index into pred_row for correct alignment
            if isinstance(pred_row, (list, np.ndarray)):
                pred_lbl = pred_row[i]
            else:                             # pred_row is scalar
                pred_lbl = pred_row

            all_true.append(lbl)
            all_pred.append(pred_lbl)
            if pred_lbl != lbl:
                example_correct = False

        id_correct_flags.append(example_correct)

    # 4) Compute metrics from flattened predictions
    macro_f1  = f1_score(all_true, all_pred, average="macro")
    token_acc = accuracy_score(all_true, all_pred)
    id_acc    = float(sum(id_correct_flags)) / len(id_correct_flags)

    return {
        "eval_macro_f1":          macro_f1,
        "eval_token_accuracy":    token_acc,
        "eval_identifier_accuracy": id_acc,
    }


def _cuda_supports_bf16() -> bool:
    if device.type != "cuda":
        return False
    if not torch.cuda.is_bf16_supported():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def _get_lm_runtime_config(train_examples: int, eval_examples: int) -> Dict[str, Any]:
    """Choose throughput-oriented runtime settings for the active device."""
    if device.type != "cuda":
        return {
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "warmup_steps": max(1, math.ceil(0.1 * (train_examples / 16) * EPOCHS)),
            "fp16": False,
            "bf16": False,
            "dataloader_num_workers": min(4, os.cpu_count() or 1),
            "pin_memory": False,
            "persistent_workers": False,
            "pad_to_multiple_of": None,
            "eval_accumulation_steps": None,
            "optim": "adamw_torch",
            "torch_compile": False,
            "viterbi_batch_size": min(32, max(16, eval_examples)),
        }

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    supports_bf16 = _cuda_supports_bf16()
    cpu_count = os.cpu_count() or 1
    num_workers = min(6, max(2, cpu_count - 2))

    if total_vram_gb >= 11:
        train_batch_size = 32
        eval_batch_size = 64
    elif total_vram_gb >= 7.5:
        train_batch_size = 16
        eval_batch_size = 32
    else:
        train_batch_size = 8
        eval_batch_size = 16

    effective_batch = train_batch_size
    warmup_steps = max(1, math.ceil(0.1 * (train_examples / effective_batch) * EPOCHS))

    return {
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": 1,
        "warmup_steps": warmup_steps,
        "fp16": not supports_bf16,
        "bf16": supports_bf16,
        "dataloader_num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "pad_to_multiple_of": 16,
        "eval_accumulation_steps": 8,
        "optim": "adamw_torch_fused",
        "torch_compile": False,
        "viterbi_batch_size": eval_batch_size,
    }

def viterbi_predict(
    model,
    dataset,
    data_collator,
    batch_size=16,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
):
    """
    Run CRF Viterbi decoding over a dataset without using the HuggingFace Trainer.

    Returns:
        all_preds:  List[List[int]] — Viterbi-decoded label IDs per example,
                    length T-2 (inner tokens, no CLS/SEP), shorter for padded sequences.
        all_labels: List[List[int]] — padded label IDs per example, length T
                    (with -100 for CLS, SEP, padding, and ignored tokens).

    Caller is responsible for aligning: use sent_labels[1:-1] to strip CLS/SEP
    before zipping with sent_preds.
    """
    # Strip raw string columns (tokens, ner_tags) that the collator cannot tensorise
    COLLATABLE = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    extra_cols = [c for c in dataset.column_names if c not in COLLATABLE]
    if extra_cols:
        dataset = dataset.remove_columns(extra_cols)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels").tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            if "predictions" in out:
                all_preds.extend(out["predictions"])
            else:
                # Fallback for non-CRF models
                all_preds.extend(torch.argmax(out["logits"], dim=-1).tolist())
            all_labels.extend(batch_labels)

    return all_preds, all_labels


def _load_lm_training_dataframe(
    script_dir: str,
    use_tagger_data: bool = True,
    use_synthetic_data: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and combine the requested LM training sources.

    Returns:
        A tuple of:
        - combined training dataframe
        - list of source names that were included
    """
    source_configs = [
        {
            "enabled": use_tagger_data,
            "name": "tagger_data",
            "path": os.path.join(script_dir, "input", "tagger_data.tsv"),
            "read_kwargs": {"sep": "\t", "dtype": str},
        },
        {
            "enabled": use_synthetic_data,
            "name": "synthetic_pos_data_full",
            "path": os.path.join(script_dir, "input", "synthetic_pos_data_full.csv"),
            "read_kwargs": {"dtype": str},
        },
    ]

    selected_sources = [cfg for cfg in source_configs if cfg["enabled"]]
    if not selected_sources:
        raise ValueError("At least one LM training dataset must be enabled.")

    required_columns = ["SPLIT", "GRAMMAR_PATTERN"]
    dataframes = []
    source_names = []

    for source in selected_sources:
        df = pd.read_csv(source["path"], **source["read_kwargs"])
        df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
        df = df.dropna(subset=required_columns)
        df = df[df["SPLIT"].str.strip().astype(bool)].copy()
        df["tokens"] = df["SPLIT"].apply(lambda x: x.strip().split())
        df["tags"] = df["GRAMMAR_PATTERN"].apply(lambda x: x.strip().split())
        df = df[df.apply(lambda r: len(r["tokens"]) == len(r["tags"]), axis=1)].copy()
        df["DATA_SOURCE"] = source["name"]

        dataframes.append(df)
        source_names.append(source["name"])

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df, source_names


def train_lm(
    script_dir: str,
    use_tagger_data: bool = True,
    use_synthetic_data: bool = True,
    selected_features: List[str] | None = None,
    run_metadata: Dict[str, Any] | None = None,
):
    """
    Trains a DistilBERT+CRF model using k-fold cross-validation for token-level grammar tagging.
    Performs model selection based on macro F1 score, and evaluates the best model on a final hold-out set.

    Enabled input datasets must contain:
        - SPLIT: tokenized identifier as space-separated subtokens (e.g., "get Employee Name")
        - GRAMMAR_PATTERN: space-separated labels (e.g., "V NM N")
        - CONTEXT: usage context string (e.g., FUNCTION, PARAMETER, ...)

    Example input row:
        SPLIT="get Employee Name", GRAMMAR_PATTERN="V NM N", CONTEXT="FUNCTION"

    Output:
        - Trained model checkpoints (best fold + final eval)
        - Hold-out predictions and metrics (saved to output/holdout_predictions.csv)
        - Text report of macro-F1, token-level and identifier-level accuracy
    """
    # 1) Paths
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    selected_features = normalize_selected_features(selected_features)

    # 2) Read the requested datasets and build “tokens” / “tags” columns
    df, source_names = _load_lm_training_dataframe(
        script_dir=script_dir,
        use_tagger_data=use_tagger_data,
        use_synthetic_data=use_synthetic_data,
    )
    print(
        "Loaded LM training data from "
        f"{', '.join(source_names)}: {len(df)} total examples"
    )
    print(f"Active LM features: {', '.join(selected_features) if selected_features else '<none>'}")
    if "DATA_SOURCE" in df.columns:
        print(df["DATA_SOURCE"].value_counts().sort_index().to_string())

    # 3) Initial Train/Val Split (15% hold-out) 
    train_df, val_df = train_test_split(
        df,
        test_size=HOLDOUT_RATIO,
        random_state=RAND_STATE,
        stratify=df["CONTEXT"]
    )

    # 4) Tokenizer (upsampling now happens per-fold to prevent cross-fold leakage)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # 6) Prepare final hold-out “validation” Dataset 
    val_dataset = prepare_dataset(val_df, LABEL2ID, selected_features=selected_features)
    tokenized_val = val_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=False
    )

    # 7) Set up K-Fold
    kf = KFold(n_splits=K, shuffle=True, random_state=RAND_STATE)
    best_macro_f1 = -1.0
    best_model_dir = None

    fold = 1
    for train_idx, test_idx in kf.split(train_df):
        print(f"\n=== Fold {fold} ===")

        # 7a) Split this fold’s train/test from the base training set
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_test_df  = train_df.iloc[test_idx].reset_index(drop=True)

        # Upsample low-frequency tags inside the fold to prevent cross-fold leakage
        low_freq_fold = fold_train_df[fold_train_df["tags"].apply(lambda tags: any(t in LOW_FREQ_TAGS for t in tags))]
        fold_train_df = pd.concat([fold_train_df] + [low_freq_fold] * 2, ignore_index=True)

        # Verb synonym augmentation: synthesise new rows by substituting V-tagged tokens
        # with synonyms drawn from VERB_SYNONYMS.  Applied across all contexts so the model
        # sees V signal in DECLARATION, PARAMETER, ATTRIBUTE, and CLASS frames, not just FUNCTION.
        _fold_rng = random.Random(RAND_STATE + fold)
        verb_aug_df = _augment_verb_examples(fold_train_df, _fold_rng)
        if not verb_aug_df.empty:
            fold_train_df = pd.concat([fold_train_df, verb_aug_df], ignore_index=True)
            print(f"  Verb augmentation added {len(verb_aug_df)} synthetic rows "
                  f"(fold train size: {len(fold_train_df)})")

        # 7b) Build HuggingFace Datasets via prepare_dataset(...) 
        fold_train_dataset = prepare_dataset(fold_train_df, LABEL2ID, selected_features=selected_features)
        fold_test_dataset  = prepare_dataset(fold_test_df, LABEL2ID, selected_features=selected_features)

        # 7c) Tokenize + align labels (exactly as before) 
        tokenized_train = fold_train_dataset.map(
            lambda sample: tokenize_and_align_labels(sample, tokenizer),
            batched=False
        )
        tokenized_test = fold_test_dataset.map(
            lambda sample: tokenize_and_align_labels(sample, tokenizer),
            batched=False
        )

        runtime_config = _get_lm_runtime_config(
            train_examples=len(fold_train_df),
            eval_examples=len(fold_test_df),
        )
        print(
            "  Runtime config: "
            f"train_bs={runtime_config['train_batch_size']}, "
            f"eval_bs={runtime_config['eval_batch_size']}, "
            f"workers={runtime_config['dataloader_num_workers']}, "
            f"precision={'bf16' if runtime_config['bf16'] else 'fp16' if runtime_config['fp16'] else 'fp32'}, "
            f"optim={runtime_config['optim']}"
        )

        # 8) Build fresh model + config for this fold 
        model = DistilBertCRFForTokenClassification(
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            pretrained_name="distilbert-base-uncased",
            dropout_prob=0.1
        ).to(device)
        model.config.selected_features = selected_features

        # 9) TrainingArguments (with early stopping)
        if device.type == "cpu":
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"fold_{fold}"),
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=runtime_config["train_batch_size"],
                per_device_eval_batch_size=runtime_config["eval_batch_size"],
                num_train_epochs=EPOCHS,
                weight_decay=0.01,
                warmup_steps=runtime_config["warmup_steps"],
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_macro_f1",
                greater_is_better=True,
                save_total_limit=1,
                report_to="none",
                seed=RAND_STATE,
                dataloader_num_workers=runtime_config["dataloader_num_workers"],
                group_by_length=True,
            )
        else:
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"fold_{fold}"),
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=runtime_config["train_batch_size"],
                per_device_eval_batch_size=runtime_config["eval_batch_size"],
                gradient_accumulation_steps=runtime_config["gradient_accumulation_steps"],
                num_train_epochs=EPOCHS,
                weight_decay=0.01,
                warmup_steps=runtime_config["warmup_steps"],
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_macro_f1",
                greater_is_better=True,
                save_total_limit=1,
                report_to="none",
                seed=RAND_STATE,
                optim=runtime_config["optim"],
                fp16=runtime_config["fp16"],
                bf16=runtime_config["bf16"],
                dataloader_num_workers=runtime_config["dataloader_num_workers"],
                dataloader_pin_memory=runtime_config["pin_memory"],
                dataloader_persistent_workers=runtime_config["persistent_workers"],
                eval_accumulation_steps=runtime_config["eval_accumulation_steps"],
                group_by_length=True,
                torch_compile=runtime_config["torch_compile"],
            )

        # 10) Define collator that handles dynamic padding + label alignment
        #     For example, if two tokenized examples have:
        #         input_ids = [[101, 2121, 5661, 2171, 102], [101, 2064, 102]]
        #     the collator will pad them to the same length and align
        #     their attention_mask and labels accordingly.
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            pad_to_multiple_of=runtime_config["pad_to_multiple_of"],
        )


        # 11) Initialize Trainer for this fold with early stopping
        #     Trainer handles batching, optimizer, eval, LR scheduling, logging, etc.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP)],
            compute_metrics=compute_metrics
        )

        # 12) Train model on this fold
        #     During training, the CRF computes loss using both:
        #         - emission scores (per-token label likelihoods from DistilBERT)
        #         - transition scores (likelihoods of label sequences)
        #     It uses the Viterbi algorithm to find the most likely label path
        #     and compares it to the true label sequence to compute loss.
        trainer.train()


        # 13) Evaluate fold performance using CRF Viterbi decoding
        viterbi_preds, fold_labels = viterbi_predict(
            model,
            tokenized_test,
            data_collator,
            batch_size=runtime_config["viterbi_batch_size"],
            num_workers=runtime_config["dataloader_num_workers"],
            pin_memory=runtime_config["pin_memory"],
            persistent_workers=runtime_config["persistent_workers"],
        )

        # Align: Viterbi output is length T-2 (no CLS/SEP); strip CLS/SEP from labels
        true_labels_list = [
            ID2LABEL[l]
            for sent_labels, sent_preds in zip(fold_labels, viterbi_preds)
            for (l, p) in zip(sent_labels[1:-1], sent_preds)
            if l != -100
        ]

        pred_labels_list = [
            ID2LABEL[p]
            for sent_labels, sent_preds in zip(fold_labels, viterbi_preds)
            for (l, p) in zip(sent_labels[1:-1], sent_preds)
            if l != -100
        ]

        fold_macro_f1 = f1_score(true_labels_list, pred_labels_list, average="macro")
        print(f"Fold {fold} Macro F1: {fold_macro_f1:.4f}")

        # 14) Save model checkpoint if this fold is the best so far
        #     This ensures we retain the model with highest validation performance
        if fold_macro_f1 > best_macro_f1:
            best_macro_f1 = fold_macro_f1
            best_model_dir = os.path.join(output_dir, "best_model")
            # Clear stale files (e.g. added_tokens.json from prior runs) before saving
            if os.path.exists(best_model_dir):
                import shutil
                shutil.rmtree(best_model_dir)
            trainer.save_model(best_model_dir)
            model.config.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

        fold += 1

    # 15) Final summary after cross-validation
    #     Reports where the best model is saved and its macro F1 on fold validation data
    print(f"\nBest fold model saved at: {best_model_dir}, Macro F1 = {best_macro_f1:.4f}")

    # 16) Load best model and prepare for final evaluation on held-out set
    best_model = DistilBertCRFForTokenClassification.from_pretrained(best_model_dir)
    best_model.to(device)
    holdout_runtime_config = _get_lm_runtime_config(
        train_examples=len(train_df),
        eval_examples=len(val_df),
    )

    # 17) Run prediction on hold-out set and record inference time
    holdout_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=holdout_runtime_config["pad_to_multiple_of"],
    )
    start_time = time.perf_counter()
    val_preds, val_labels = viterbi_predict(
        best_model,
        tokenized_val,
        holdout_collator,
        batch_size=holdout_runtime_config["viterbi_batch_size"],
        num_workers=holdout_runtime_config["dataloader_num_workers"],
        pin_memory=holdout_runtime_config["pin_memory"],
        persistent_workers=holdout_runtime_config["persistent_workers"],
    )
    end_time = time.perf_counter()

    # Align: Viterbi output is length T-2 (no CLS/SEP); strip CLS/SEP from labels
    flat_true = [
        ID2LABEL[l]
        for sent_labels, sent_preds in zip(val_labels, val_preds)
        for (l, p) in zip(sent_labels[1:-1], sent_preds)
        if l != -100
    ]
    flat_pred = [
        ID2LABEL[p]
        for sent_labels, sent_preds in zip(val_labels, val_preds)
        for (l, p) in zip(sent_labels[1:-1], sent_preds)
        if l != -100
    ]

    # 18) Output predictions per row to CSV for inspection or error analysis
    from .distilbert_tagger import DistilBertTagger

    # Re-instantiate the exact same DistilBERT tagger we saved
    tagger = DistilBertTagger(best_model_dir)

    rows = []
    for _, row in val_df.iterrows():
        tokens     = row["tokens"]            # e.g. ["my", "Identifier", "Name"]
        true_tags  = row["tags"]              # e.g. ["NM", "DT", "DT"]
        context    = row.get("CONTEXT", "")   # e.g. "FUNCTION"
        type_str   = row.get("TYPE", "")      # if present; otherwise ""
        language   = row.get("LANGUAGE", "")  # if present; otherwise ""
        system_name= row.get("SYSTEM_NAME", "")  # if present; otherwise ""

        # `tag_identifier` now returns a list of string labels, not IDs
        pred_tags = tagger.tag_identifier(tokens, context, type_str, language, system_name)

        rows.append({
            "tokens":      " ".join(tokens),
            "true_tags":   " ".join(true_tags),
            "pred_tags":   " ".join(pred_tags),
            "context":     context,
            "type":        type_str,
            "language":    language,
            "system_name": system_name,
            "data_source": row.get("DATA_SOURCE", ""),
        })

    preds_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "holdout_predictions.csv")
    preds_df.to_csv(csv_path, index=False)
    print(f"\nWrote hold-out predictions to: {csv_path}")

    # Now also compute identifier-level accuracy from the “flat_true/flat_pred” folds:
    # We need to compare per-example (not flattened) again, so re-run a grouping logic.
    df = pd.read_csv(os.path.join(output_dir, "holdout_predictions.csv"))
    df["row_correct"] = df["true_tags"] == df["pred_tags"]
    id_level_acc = df["row_correct"].mean()
    
    # Report evaluation metrics and timing info
    total_tokens = sum(len(ex["tokens"]) for ex in val_dataset)
    total_examples = len(val_dataset)
    elapsed = end_time - start_time
    final_macro_f1 = f1_score(flat_true, flat_pred, average="macro")
    final_accuracy = accuracy_score(flat_true, flat_pred)
    
    print("\nFinal Evaluation on Held-Out Set:")
    with open('holdout_report.txt', 'w') as f:
        report = classification_report(flat_true, flat_pred)
        dual_print(report, file=f)
        _write_run_metadata(
            file=f,
            source_names=source_names,
            selected_features=selected_features,
            run_metadata=run_metadata,
        )
        dual_print(f"\nInference Time: {elapsed:.2f}s for {total_examples} identifiers ({total_tokens} tokens)", file=f)
        dual_print(f"Tokens/sec: {total_tokens / elapsed:.2f}", file=f)
        dual_print(f"Identifiers/sec: {total_examples / elapsed:.2f}", file=f)
        dual_print(f"\nFinal Macro F1 on Held-Out Set: {final_macro_f1:.4f}", file=f)
        dual_print(f"Final Token-level Accuracy on Held-Out Set: {final_accuracy:.4f}", file=f)
        dual_print(f"Final Identifier-level Accuracy on Held-Out Set: {id_level_acc:.4f}", file=f)