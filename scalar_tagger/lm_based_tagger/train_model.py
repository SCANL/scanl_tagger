import os
import re
import time
import random
import math
import inspect
import shutil
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .distilbert_crf import DistilBertCRFForTokenClassification
from scalar_tagger.version import __version__

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from transformers import (
    Trainer,
    TrainingArguments,
    DistilBertTokenizerFast,
    DataCollatorForTokenClassification,
)

from datasets import Dataset
from scalar_tagger.lm_based_tagger.evaluation import accuracy_by_source, normalize_split, real_code_summary
from scalar_tagger.lm_based_tagger.distilbert_preprocessing import (
    add_marker_tokens,
    collect_marker_tokens,
    init_marker_embeddings,
    normalize_selected_features,
    prepare_dataset,
    tokenize_and_align_labels,
)

# If CUDA is available, use it; otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def _configure_torch_runtime() -> None:
    """Configure CUDA for reproducible training runs."""
    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

# === Random Seeds ===
SPLIT_SEED = 658   # holdout split and CV folds; fixed so every run is scored on the same holdout
TRAIN_SEED = 209   # default training seed (weight init, batch order, augmentation, dropout)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_seed_everything(TRAIN_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_configure_torch_runtime()

# === Hyperparameters / Config ===
K = 5                     # number of CV folds
HOLDOUT_RATIO = 0.20      # 20% held out for final evaluation
# Epochs for every CV fold and for the final retrain. Folds train a fixed number of epochs, with
# no early stopping or best-checkpoint pick on their own test slice, so their out-of-fold
# predictions are unbiased. (When folds did pick their best epoch, it was nearly always the last.)
EPOCHS = 5
CRF_LEARNING_RATE = 5e-3  # CRF transitions need a much higher LR than the encoder (5e-5)
LOW_FREQ_TAGS = {"CJ", "VM", "PRE", "V"}

# === Data sources ===
REAL_SOURCE = "tagger_data"   # DATA_SOURCE of the real, commit-linked identifiers
DEFAULT_SYNTHETIC_PATH = os.path.join("input", "synthetic_pos_data_full.csv")
AMPLIFY_MODES = ("all", "real", "none")  # which rows get low-frequency upsampling and verb augmentation

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

    Only FUNCTION rows are augmented. Augmenting every context taught the model that
    verb-looking words are V wherever they appear, so modifiers such as `adjusted` in
    adjustedGradient or `conv` in conv_in_channels_ were tagged V.

    Args:
        df:  Fold training DataFrame with 'tokens' (List[str]) and 'tags' (List[str]).
        rng: Seeded Random instance for reproducibility.

    Returns:
        DataFrame of newly-synthesised rows (may be empty if nothing is substitutable).
    """
    MAX_SYNS = 2  # max new synthetic examples per original row

    new_rows = []
    for _, row in df.iterrows():
        if str(row.get("CONTEXT", "")).strip().upper() != "FUNCTION":
            continue
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


def _prepare_training_frame(
    df: pd.DataFrame,
    *,
    augmentation_seed: int,
    amplify: str = "all",
) -> Tuple[pd.DataFrame, int]:
    """
    Apply fold-safe resampling and augmentation to a training frame.

    `amplify` chooses which rows are upsampled and augmented: "all", "real" for
    REAL_SOURCE rows only (synthetic rows are then used once, as written), or "none".
    """
    if amplify not in AMPLIFY_MODES:
        raise ValueError(f"amplify must be one of {AMPLIFY_MODES}, not {amplify!r}")
    prepared_df = df.reset_index(drop=True).copy()
    if amplify == "none":
        return prepared_df, 0

    if amplify == "real":
        amplifiable = prepared_df["DATA_SOURCE"] == REAL_SOURCE
    else:
        amplifiable = pd.Series(True, index=prepared_df.index)

    low_freq_rows = prepared_df[
        amplifiable
        & prepared_df["tags"].apply(lambda tags: any(tag in LOW_FREQ_TAGS for tag in tags))
    ]
    prepared_df = pd.concat([prepared_df] + [low_freq_rows] * 2, ignore_index=True)

    augment_from = prepared_df
    if amplify == "real":
        augment_from = prepared_df[prepared_df["DATA_SOURCE"] == REAL_SOURCE]
    verb_aug_df = _augment_verb_examples(augment_from, random.Random(augmentation_seed))
    if not verb_aug_df.empty:
        prepared_df = pd.concat([prepared_df, verb_aug_df], ignore_index=True)

    return prepared_df, len(verb_aug_df)


# === Label List & Mappings ===
LABEL_LIST = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL   = {i: label for label, i in LABEL2ID.items()}

def dual_print(*args, file, **kwargs):
    print(*args, **kwargs)         # stdout
    print(*args, file=file, **kwargs)  # file


def _write_run_metadata(
    file,
    dataset_lines: List[str],
    selected_features: List[str],
    train_seed: int,
    amplify: str,
    marker_tokens: bool = False,
    folds: int = K,
):
    """Write reproducibility metadata for the current LM training run."""
    dual_print("\nRun Configuration:", file=file)
    dual_print(f"SCALAR version: {__version__}", file=file)
    dual_print(f"Split seed: {SPLIT_SEED}", file=file)
    dual_print(f"Train seed: {train_seed}", file=file)
    dual_print(f"CV folds: {folds}, holdout ratio: {HOLDOUT_RATIO}, epochs (folds and final retrain): {EPOCHS}", file=file)
    dual_print("Datasets:", file=file)
    for line in dataset_lines:
        dual_print(f"  {line}", file=file)
    dual_print(f"Amplification (upsampling + verb augmentation): {amplify} rows", file=file)
    dual_print(
        f"Features: {', '.join(selected_features) if selected_features else '<none>'}",
        file=file,
    )
    dual_print(f"Markers as single tokens: {'yes' if marker_tokens else 'no'}", file=file)


class CRFTrainer(Trainer):
    """
    Trainer that gives the CRF transition parameters their own learning rate.

    The CRF only has num_labels^2 + 2 * num_labels parameters and starts from small random
    values; at the encoder learning rate (5e-5) they barely move during fine-tuning.
    """

    def create_optimizer(self, *args, **kwargs):
        # transformers <5.3 takes no `model` argument; newer versions accept an optional one.
        if self.optimizer is None:
            super().create_optimizer(*args, **kwargs)
            opt_model = kwargs.get("model") or (args[0] if args else None) or self.model
            crf_param_ids = {
                id(p) for n, p in opt_model.named_parameters()
                if re.search(r"(^|\.)crf\.", n) and p.requires_grad
            }
            crf_params = []
            for group in self.optimizer.param_groups:
                crf_params.extend(p for p in group["params"] if id(p) in crf_param_ids)
                group["params"] = [p for p in group["params"] if id(p) not in crf_param_ids]
            if crf_params:
                self.optimizer.add_param_group({
                    "params": crf_params,
                    "lr": CRF_LEARNING_RATE,
                    "weight_decay": 0.0,
                })
        return self.optimizer


# 11) compute_metrics function (macro-F1) 
def compute_metrics(eval_pred, row_mask=None):
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

    `row_mask` (one bool per eval row, in eval order) limits the metrics to those rows; training
    passes the real rows, so the per-epoch log tracks real code rather than synthetic rows.

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

    if row_mask is not None:
        keep = np.asarray(row_mask, dtype=bool)
        preds = [row for row, k in zip(preds, keep) if k]
        labels = [row for row, k in zip(labels, keep) if k]
        if not labels:
            return {}

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


def _training_arguments_supports(name: str) -> bool:
    return name in inspect.signature(TrainingArguments.__init__).parameters

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
            labels = batch.pop("labels")
            batch_labels = labels.tolist()
            # Word positions are exactly the labeled positions; the CRF decodes over those only
            batch["word_mask"] = labels != -100
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
    synthetic_path: str | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and combine the requested LM training sources.

    `synthetic_path` (relative to `script_dir` unless absolute) chooses the synthetic file;
    the default is DEFAULT_SYNTHETIC_PATH. Its DATA_SOURCE is the file name without extension.

    Returns:
        A tuple of:
        - combined training dataframe, real rows first
        - list of source names that were included
    """
    synthetic_path = os.path.join(script_dir, synthetic_path or DEFAULT_SYNTHETIC_PATH)
    source_configs = [
        {
            "enabled": use_tagger_data,
            "name": REAL_SOURCE,
            "path": os.path.join(script_dir, "input", "tagger_data.tsv"),
            "read_kwargs": {"sep": "\t", "dtype": str},
        },
        {
            "enabled": use_synthetic_data,
            "name": os.path.splitext(os.path.basename(synthetic_path))[0],
            "path": synthetic_path,
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
        if "TYPE" in df.columns:
            # Typeless identifiers (enum constants, macros) have a blank TYPE; keep it ""
            # so the type feature reads @type_unknown rather than the string "nan".
            df["TYPE"] = df["TYPE"].fillna("")
        df["tokens"] = df["SPLIT"].apply(lambda x: x.strip().split())
        df["tags"] = df["GRAMMAR_PATTERN"].apply(lambda x: x.strip().split())
        df = df[df.apply(lambda r: len(r["tokens"]) == len(r["tags"]), axis=1)].copy()
        # tagger_data.tsv names the collection each row came from in its own DATA_SOURCE
        # column (original_set, ...); keep that as SUBSET. DATA_SOURCE becomes the file name,
        # which decides what counts as real for twin-dropping and amplification.
        file_subsets = df.get("DATA_SOURCE", pd.Series("", index=df.index)).fillna("").str.strip()
        df["SUBSET"] = file_subsets.where(file_subsets != "", source["name"])
        df["DATA_SOURCE"] = source["name"]

        dataframes.append(df)
        source_names.append(source["name"])

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df, source_names


def _drop_synthetic_overlap(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Drop synthetic rows that duplicate a real identifier or an earlier synthetic row.

    A synthetic twin of a real identifier lets the model see a real holdout answer during
    training. Synthetic duplicates are matched on split and context, since the same words can
    take different tags in a different context.

    Returns the filtered frame and {"real_twins": n, "duplicates": n} dropped.
    """
    keys = df["SPLIT"].map(normalize_split)
    synthetic = df["DATA_SOURCE"] != REAL_SOURCE
    real_twin = synthetic & keys.isin(set(keys[~synthetic]))
    duplicate = synthetic & ~real_twin & pd.DataFrame({
        "source": df["DATA_SOURCE"], "key": keys, "context": df["CONTEXT"].str.strip().str.upper(),
    }).duplicated()
    dropped = {"real_twins": int(real_twin.sum()), "duplicates": int(duplicate.sum())}
    return df[~(real_twin | duplicate)].reset_index(drop=True), dropped


def _subsets(df: pd.DataFrame) -> pd.Series:
    """Each row's subset (original_set, headnoun_set, ...), or its DATA_SOURCE if it has none."""
    return df["SUBSET"] if "SUBSET" in df.columns else df["DATA_SOURCE"]


def _split_holdout(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out HOLDOUT_RATIO of each subset separately, stratified by context.

    Splitting per subset means each subset's holdout depends only on its own rows, so adding
    a subset or swapping the synthetic file can't change which identifiers are scored in the others.
    """
    train_parts, holdout_parts = [], []
    for _, group in df.groupby(_subsets(df), sort=False):
        train_part, holdout_part = train_test_split(
            group,
            test_size=HOLDOUT_RATIO,
            random_state=SPLIT_SEED,
            stratify=group["CONTEXT"],
        )
        train_parts.append(train_part)
        holdout_parts.append(holdout_part)
    return pd.concat(train_parts), pd.concat(holdout_parts)


def _decoded_tags(viterbi_preds, labels) -> List[List[str]]:
    """Word-level tag strings per row from viterbi_predict output (strips CLS/SEP from labels)."""
    return [
        [ID2LABEL[p] for l, p in zip(row_labels[1:-1], row_preds) if l != -100]
        for row_labels, row_preds in zip(labels, viterbi_preds)
    ]


def _prediction_rows(df: pd.DataFrame, pred_tags: List[List[str]], eval_split: str, fold: int | None = None):
    """One holdout_predictions.csv-style record per row of `df`."""
    rows = []
    for (_, row), tags in zip(df.iterrows(), pred_tags):
        record = {
            "tokens":      " ".join(row["tokens"]),
            "true_tags":   " ".join(row["tags"]),
            "pred_tags":   " ".join(tags),
            "context":     row.get("CONTEXT", ""),
            "type":        row.get("TYPE", ""),
            "language":    row.get("LANGUAGE", ""),
            "system_name": row.get("SYSTEM_NAME", ""),
            "data_source": row.get("DATA_SOURCE", ""),
            "subset":      row.get("SUBSET", ""),
            "eval_split":  eval_split,
        }
        if fold is not None:
            record["fold"] = fold
        rows.append(record)
    return rows


def _write_summary(title: str, summary: pd.DataFrame, file) -> None:
    dual_print(f"\n{title}", file=file)
    for row in summary.itertuples():
        dual_print(
            f"  {row.source}: {row.identifiers} identifiers, identifier {row.identifier_accuracy:.4f}, "
            f"token {row.token_accuracy:.4f}, macro-F1 {row.macro_f1:.4f}",
            file=file,
        )


def _new_model(selected_features: List[str], tokenizer, marker_pieces: dict | None):
    """A fresh DistilBERT+CRF model, with single-token markers if `marker_pieces` is given."""
    model = DistilBertCRFForTokenClassification(
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        pretrained_name="distilbert-base-uncased",
        dropout_prob=0.1
    )
    if marker_pieces is not None:
        init_marker_embeddings(model, tokenizer, marker_pieces)
        model.config.marker_tokens = True
    model.config.selected_features = selected_features
    return model.to(device)


def save_run_reports(script_dir: str, run_name: str, train_seed: int) -> List[str]:
    """Copy this run's report and predictions to output/reports/<run_name>_seed<N>.*"""
    reports_dir = os.path.join(script_dir, "output", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    stem = os.path.join(reports_dir, f"{run_name}_seed{train_seed}")
    copies = [
        (os.path.join(script_dir, "holdout_report.txt"), f"{stem}.txt"),
        (os.path.join(script_dir, "output", "holdout_predictions.csv"), f"{stem}_predictions.csv"),
        (os.path.join(script_dir, "output", "cv_predictions.csv"), f"{stem}_cv_predictions.csv"),
    ]
    saved = []
    for source, target in copies:
        if os.path.exists(source):
            shutil.copyfile(source, target)
            saved.append(target)
    return saved


def train_lm(
    script_dir: str,
    use_tagger_data: bool = True,
    use_synthetic_data: bool = True,
    selected_features: List[str] | None = None,
    model_dir: str | None = None,
    train_seed: int = TRAIN_SEED,
    synthetic_path: str | None = None,
    amplify: str = "all",
    run_name: str | None = None,
    marker_tokens: bool = False,
    folds: int = K,
):
    """
    Trains a DistilBERT+CRF model for token-level grammar tagging and evaluates it two ways.

    K-fold CV on the training portion gives out-of-fold predictions for every training row;
    the final model, retrained on the whole training portion, is scored on the hold-out set.
    Together they score every real identifier exactly once ("real-code evaluation" in the
    report), which is a much larger and steadier sample than the hold-out set alone.

    Enabled input datasets must contain:
        - SPLIT: tokenized identifier as space-separated subtokens (e.g., "get Employee Name")
        - GRAMMAR_PATTERN: space-separated labels (e.g., "V NM N")
        - CONTEXT: usage context string (e.g., FUNCTION, PARAMETER, ...)

    Example input row:
        SPLIT="get Employee Name", GRAMMAR_PATTERN="V NM N", CONTEXT="FUNCTION"

    `train_seed` varies weight initialization, batch order, augmentation and dropout. The
    holdout split and CV folds always use SPLIT_SEED, so runs with different seeds are scored
    on the same holdout set. Each source is split separately, so the real holdout also stays
    the same whichever synthetic file (`synthetic_path`) is used, or none.

    `amplify` is "all" to upsample and augment every row, "real" for real rows only, or "none".

    `marker_tokens` adds the feature and position markers to the tokenizer as single tokens
    (initialized from the subwords they used to split into) instead of letting WordPiece split them.

    `folds` is the number of CV folds (default K). Fewer folds train faster, but each fold sees
    less data, so out-of-fold accuracy is lower: compare only runs with the same fold count.

    `run_name`, if given, copies the report and predictions to output/reports/<run_name>_seed<N>.*
    for later comparison with `python -m scalar_tagger.lm_based_tagger.evaluation`.

    Output:
        - Final model checkpoint (output/best_model)
        - Out-of-fold CV predictions (output/cv_predictions.csv)
        - Hold-out predictions (output/holdout_predictions.csv)
        - Text report (holdout_report.txt): identifier accuracy, token accuracy and macro-F1
    """
    # 1) Paths
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    best_model_dir = model_dir or os.path.join(output_dir, "best_model")
    if not os.path.isabs(best_model_dir):
        best_model_dir = os.path.join(script_dir, best_model_dir)
    best_model_root = os.path.dirname(best_model_dir)
    os.makedirs(best_model_root, exist_ok=True)
    selected_features = normalize_selected_features(selected_features)
    _seed_everything(train_seed)

    # 2) Read the requested datasets and build “tokens” / “tags” columns
    df, source_names = _load_lm_training_dataframe(
        script_dir=script_dir,
        use_tagger_data=use_tagger_data,
        use_synthetic_data=use_synthetic_data,
        synthetic_path=synthetic_path,
    )
    df, dropped = _drop_synthetic_overlap(df)
    source_counts = df["DATA_SOURCE"].value_counts()
    dataset_lines = []
    for name in source_names:
        line = f"{name}: {source_counts.get(name, 0)} rows"
        if name == REAL_SOURCE:
            subset_counts = df.loc[df["DATA_SOURCE"] == name, "SUBSET"].value_counts().sort_index()
            line += " (" + ", ".join(f"{s} {n}" for s, n in subset_counts.items()) + ")"
        else:
            line += (
                f" ({synthetic_path or DEFAULT_SYNTHETIC_PATH}; dropped "
                f"{dropped['real_twins']} real twins and {dropped['duplicates']} duplicates)"
            )
        dataset_lines.append(line)
    print(f"Loaded LM training data: {len(df)} total examples")
    for line in dataset_lines:
        print(f"  {line}")
    print(f"Active LM features: {', '.join(selected_features) if selected_features else '<none>'}")

    # 3) Initial Train/Val Split (20% hold-out of each source)
    train_df, val_df = _split_holdout(df)

    # 4) Tokenizer (upsampling now happens per-fold to prevent cross-fold leakage)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    marker_pieces = None
    if marker_tokens:
        markers = collect_marker_tokens(df, selected_features)
        marker_pieces = add_marker_tokens(tokenizer, markers)
        print(f"Added {len(marker_pieces)} feature/position markers to the tokenizer as single tokens")

    # 6) Prepare final hold-out “validation” Dataset 
    val_dataset = prepare_dataset(val_df, LABEL2ID, selected_features=selected_features)
    tokenized_val = val_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=False
    )

    # 7) Set up K-Fold
    #    Stratify by subset and context, so every fold has its share of each subset.
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SPLIT_SEED)
    fold_strata = _subsets(train_df) + "|" + train_df["CONTEXT"]
    cv_rows = []
    fold = 1
    for train_idx, test_idx in kf.split(train_df, fold_strata):
        print(f"\n=== Fold {fold} ===")

        # 7a) Split this fold’s train/test from the base training set
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_test_df  = train_df.iloc[test_idx].reset_index(drop=True)

        # Apply resampling and verb augmentation only inside the fold training slice.
        fold_train_df, verb_aug_count = _prepare_training_frame(
            fold_train_df,
            augmentation_seed=train_seed + fold,
            amplify=amplify,
        )
        if verb_aug_count:
            print(f"  Verb augmentation added {verb_aug_count} synthetic rows "
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
        model = _new_model(selected_features, tokenizer, marker_pieces)

        # 9) TrainingArguments (with early stopping)
        # Fixed epochs: the per-epoch eval is only a learning-curve log (real rows, emission argmax).
        training_kwargs = {
            "output_dir": os.path.join(output_dir, f"fold_{fold}"),
            "eval_strategy": "epoch",
            "save_strategy": "no",
            "learning_rate": 5e-5,
            "per_device_train_batch_size": runtime_config["train_batch_size"],
            "per_device_eval_batch_size": runtime_config["eval_batch_size"],
            "num_train_epochs": EPOCHS,
            "weight_decay": 0.01,
            "warmup_steps": runtime_config["warmup_steps"],
            "lr_scheduler_type": "cosine",
            "report_to": "none",
            "seed": train_seed,
            "dataloader_num_workers": runtime_config["dataloader_num_workers"],
        }

        if _training_arguments_supports("data_seed"):
            training_kwargs["data_seed"] = train_seed

        if _training_arguments_supports("group_by_length"):
            training_kwargs["group_by_length"] = True

        if device.type != "cpu":
            training_kwargs.update({
                "gradient_accumulation_steps": runtime_config["gradient_accumulation_steps"],
                "optim": runtime_config["optim"],
                "fp16": runtime_config["fp16"],
                "bf16": runtime_config["bf16"],
                "dataloader_pin_memory": runtime_config["pin_memory"],
                "dataloader_persistent_workers": runtime_config["persistent_workers"],
                "eval_accumulation_steps": runtime_config["eval_accumulation_steps"],
            })
            if _training_arguments_supports("torch_compile"):
                training_kwargs["torch_compile"] = runtime_config["torch_compile"]

        training_args = TrainingArguments(**training_kwargs)

        # 10) Define collator that handles dynamic padding + label alignment
        #     For example, if two tokenized examples have:
        #         input_ids = [[101, 2121, 5661, 2171, 102], [101, 2064, 102]]
        #     the collator will pad them to the same length and align
        #     their attention_mask and labels accordingly.
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            pad_to_multiple_of=runtime_config["pad_to_multiple_of"],
        )


        # 11) Initialize Trainer for this fold
        #     Trainer handles batching, optimizer, eval, LR scheduling, logging, etc.
        trainer = CRFTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=partial(
                compute_metrics,
                row_mask=(fold_test_df["DATA_SOURCE"] == REAL_SOURCE).to_numpy(),
            ),
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

        fold_pred_tags = _decoded_tags(viterbi_preds, fold_labels)
        fold_rows = _prediction_rows(fold_test_df, fold_pred_tags, "cv", fold)
        cv_rows.extend(fold_rows)
        fold_real = real_code_summary(pd.DataFrame(fold_rows))
        if not fold_real.empty:
            real = fold_real.iloc[0]
            print(f"Fold {fold} real code: identifier {real.identifier_accuracy:.4f}, macro-F1 {real.macro_f1:.4f}")

        fold += 1

    # 15) Save the out-of-fold predictions, then retrain once on all of train_df.
    cv_df = pd.DataFrame(cv_rows)
    cv_path = os.path.join(output_dir, "cv_predictions.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"\nWrote out-of-fold CV predictions to: {cv_path}")
    best_num_train_epochs = EPOCHS

    full_train_df, final_verb_aug_count = _prepare_training_frame(
        train_df,
        augmentation_seed=train_seed + folds + 1,
        amplify=amplify,
    )
    if final_verb_aug_count:
        print(
            f"Final retrain verb augmentation added {final_verb_aug_count} synthetic rows "
            f"(train size: {len(full_train_df)})"
        )

    full_train_dataset = prepare_dataset(full_train_df, LABEL2ID, selected_features=selected_features)
    tokenized_full_train = full_train_dataset.map(
        lambda sample: tokenize_and_align_labels(sample, tokenizer),
        batched=False
    )

    final_runtime_config = _get_lm_runtime_config(
        train_examples=len(full_train_df),
        eval_examples=len(val_df),
    )
    print(
        "Final retrain runtime config: "
        f"train_bs={final_runtime_config['train_batch_size']}, "
        f"workers={final_runtime_config['dataloader_num_workers']}, "
        f"precision={'bf16' if final_runtime_config['bf16'] else 'fp16' if final_runtime_config['fp16'] else 'fp32'}, "
        f"optim={final_runtime_config['optim']}"
    )

    final_model = _new_model(selected_features, tokenizer, marker_pieces)

    final_training_kwargs = {
        "output_dir": os.path.join(output_dir, "final_retrain"),
        "eval_strategy": "no",
        "save_strategy": "no",
        "learning_rate": 5e-5,
        "per_device_train_batch_size": final_runtime_config["train_batch_size"],
        "per_device_eval_batch_size": final_runtime_config["eval_batch_size"],
        "num_train_epochs": best_num_train_epochs,
        "weight_decay": 0.01,
        "warmup_steps": max(1, math.ceil(0.1 * (len(full_train_df) / final_runtime_config["train_batch_size"]) * best_num_train_epochs)),
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "seed": train_seed,
        "dataloader_num_workers": final_runtime_config["dataloader_num_workers"],
    }

    if _training_arguments_supports("data_seed"):
        final_training_kwargs["data_seed"] = train_seed

    if _training_arguments_supports("group_by_length"):
        final_training_kwargs["group_by_length"] = True

    if device.type != "cpu":
        final_training_kwargs.update({
            "gradient_accumulation_steps": final_runtime_config["gradient_accumulation_steps"],
            "optim": final_runtime_config["optim"],
            "fp16": final_runtime_config["fp16"],
            "bf16": final_runtime_config["bf16"],
            "dataloader_pin_memory": final_runtime_config["pin_memory"],
            "dataloader_persistent_workers": final_runtime_config["persistent_workers"],
        })
        if _training_arguments_supports("torch_compile"):
            final_training_kwargs["torch_compile"] = final_runtime_config["torch_compile"]

    final_training_args = TrainingArguments(**final_training_kwargs)
    final_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=final_runtime_config["pad_to_multiple_of"],
    )
    final_trainer = CRFTrainer(
        model=final_model,
        args=final_training_args,
        train_dataset=tokenized_full_train,
        processing_class=tokenizer,
        data_collator=final_collator,
    )
    final_trainer.train()

    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    final_trainer.save_model(best_model_dir)
    final_model.config.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Final retrained model saved at: {best_model_dir}")

    # 16) Load final model and prepare for final evaluation on held-out set
    best_model = DistilBertCRFForTokenClassification.from_pretrained(best_model_dir)
    best_model.to(device)
    holdout_runtime_config = _get_lm_runtime_config(
        train_examples=len(full_train_df),
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

    # Align: Viterbi output is length T-2 (no CLS/SEP); strip CLS/SEP from labels.
    # Batched decoding should agree with the per-identifier tagger below; the report shows both.
    viterbi_true = [
        l
        for sent_labels, sent_preds in zip(val_labels, val_preds)
        for (l, p) in zip(sent_labels[1:-1], sent_preds)
        if l != -100
    ]
    viterbi_pred = [
        p
        for sent_labels, sent_preds in zip(val_labels, val_preds)
        for (l, p) in zip(sent_labels[1:-1], sent_preds)
        if l != -100
    ]

    # 18) Output predictions per row to CSV for inspection or error analysis
    from .distilbert_tagger import DistilBertTagger

    # Re-instantiate the exact same DistilBERT tagger we saved
    tagger = DistilBertTagger(best_model_dir, local=True)

    holdout_pred_tags = []
    flat_true = []
    flat_pred = []
    for _, row in val_df.iterrows():
        pred_tags = tagger.tag_identifier(
            row["tokens"],
            row.get("CONTEXT", ""),
            row.get("TYPE", ""),
            row.get("LANGUAGE", ""),
            row.get("SYSTEM_NAME", ""),
        )
        holdout_pred_tags.append(pred_tags)
        flat_true.extend(row["tags"])
        flat_pred.extend(pred_tags)
    rows = _prediction_rows(val_df, holdout_pred_tags, "holdout")

    preds_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "holdout_predictions.csv")
    preds_df.to_csv(csv_path, index=False)
    print(f"\nWrote hold-out predictions to: {csv_path}")

    # Identifier-level accuracy: every tag in the identifier must match
    id_level_acc = (preds_df["true_tags"] == preds_df["pred_tags"]).mean()

    # Report evaluation metrics and timing info
    total_tokens = sum(len(ex["tokens"]) for ex in val_dataset)
    total_examples = len(val_dataset)
    elapsed = end_time - start_time
    final_macro_f1 = f1_score(flat_true, flat_pred, average="macro")
    final_accuracy = accuracy_score(flat_true, flat_pred)

    print("\nFinal Evaluation on Held-Out Set:")
    with open(os.path.join(script_dir, "holdout_report.txt"), "w") as f:
        dual_print(classification_report(flat_true, flat_pred), file=f)
        _write_run_metadata(
            file=f,
            dataset_lines=dataset_lines,
            selected_features=selected_features,
            train_seed=train_seed,
            amplify=amplify,
            marker_tokens=marker_tokens,
            folds=folds,
        )
        dual_print(f"Model dir: {best_model_dir}", file=f)
        dual_print(f"\nInference Time: {elapsed:.2f}s for {total_examples} identifiers ({total_tokens} tokens)", file=f)
        dual_print(f"Tokens/sec: {total_tokens / elapsed:.2f}", file=f)
        dual_print(f"Identifiers/sec: {total_examples / elapsed:.2f}", file=f)
        dual_print(f"\nFinal Macro F1 on Held-Out Set: {final_macro_f1:.4f}", file=f)
        dual_print(f"Final Token-level Accuracy on Held-Out Set: {final_accuracy:.4f}", file=f)
        dual_print(f"Final Identifier-level Accuracy on Held-Out Set: {id_level_acc:.4f}", file=f)
        dual_print(f"Batched Viterbi Token-level Accuracy (consistency check): {accuracy_score(viterbi_true, viterbi_pred):.4f}", file=f)
        for title, column in (("Data Source", "data_source"), ("Subset", "subset")):
            _write_summary(f"Held-Out Accuracy by {title}:", accuracy_by_source(preds_df, column), f)

        # Every real identifier scored once: CV folds score the training portion, the final
        # model scores the hold-out set. This is the number to compare runs on.
        all_df = pd.concat([cv_df, preds_df], ignore_index=True)
        _write_summary(
            "Real-Code Evaluation (CV out-of-fold + hold-out; every real identifier scored once):",
            real_code_summary(all_df), f,
        )
        _write_summary("CV Out-of-Fold by Data Source:", accuracy_by_source(cv_df, "data_source"), f)
        real_all = all_df[all_df["data_source"] == REAL_SOURCE]
        if not real_all.empty:
            dual_print("\nPer-Tag Report on Real Code (CV out-of-fold + hold-out):", file=f)
            dual_print(classification_report(
                [t for tags in real_all["true_tags"] for t in tags.split()],
                [t for tags in real_all["pred_tags"] for t in tags.split()],
                digits=4, zero_division=0,
            ), file=f)

    if run_name:
        for path in save_run_reports(script_dir, run_name, train_seed):
            print(f"Saved {path}")
