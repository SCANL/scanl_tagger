import os
import time
import random
from typing import List, Tuple

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
from src.lm_based_tagger.distilbert_preprocessing import prepare_dataset, tokenize_and_align_labels

# If CUDA is available, use it; otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Random Seeds ===
RAND_STATE = 69
random.seed(RAND_STATE)
np.random.seed(RAND_STATE)
torch.manual_seed(RAND_STATE)
torch.cuda.manual_seed_all(RAND_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Hyperparameters / Config ===
K = 5                     # number of CV folds
HOLDOUT_RATIO = 0.15      # 15% held out for final evaluation
EPOCHS = 5            # number of epochs per fold
EARLY_STOP = 2            # patience for early stopping
LOW_FREQ_TAGS = {"CJ", "VM", "PRE", "V"}

# === Label List & Mappings ===
LABEL_LIST = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL   = {i: label for label, i in LABEL2ID.items()}

def dual_print(*args, file, **kwargs):
    print(*args, **kwargs)         # stdout
    print(*args, file=file, **kwargs)  # file


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

def viterbi_predict(model, dataset, data_collator, batch_size=16):
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

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
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
    val_dataset = prepare_dataset(val_df, LABEL2ID)
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

        # 7b) Build HuggingFace Datasets via prepare_dataset(...) 
        fold_train_dataset = prepare_dataset(fold_train_df, LABEL2ID)
        fold_test_dataset  = prepare_dataset(fold_test_df, LABEL2ID)

        # 7c) Tokenize + align labels (exactly as before) 
        tokenized_train = fold_train_dataset.map(
            lambda sample: tokenize_and_align_labels(sample, tokenizer),
            batched=False
        )
        tokenized_test = fold_test_dataset.map(
            lambda sample: tokenize_and_align_labels(sample, tokenizer),
            batched=False
        )

        # 8) Build fresh model + config for this fold 
        model = DistilBertCRFForTokenClassification(
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            pretrained_name="distilbert-base-uncased",
            dropout_prob=0.1
        ).to(device)

        # 9) TrainingArguments (with early stopping)
        # Compute warmup_steps as ~10% of total training steps for this fold
        if device.type == "cpu":
            _effective_batch = 16
        else:
            _effective_batch = 8 * 2  # per_device_batch * gradient_accumulation_steps
        _warmup_steps = max(1, int(0.1 * (len(fold_train_df) / _effective_batch) * EPOCHS))

        if device.type == "cpu":
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"fold_{fold}"),
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=EPOCHS,
                weight_decay=0.01,
                warmup_steps=_warmup_steps,
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_macro_f1",
                greater_is_better=True,
                save_total_limit=1,
                report_to="none",
                seed=RAND_STATE
            )
        else:
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"fold_{fold}"),
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,

                # Use more of your VRAM — try 8 or even 16 depending on sequence length
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=2,  # adjust if needed to match total batch size

                num_train_epochs=EPOCHS,
                weight_decay=0.01,
                warmup_steps=_warmup_steps,
                lr_scheduler_type="cosine",

                load_best_model_at_end=True,
                metric_for_best_model="eval_macro_f1",
                greater_is_better=True,
                save_total_limit=1,

                report_to="none",
                seed=RAND_STATE,

                fp16=True,                     # ✅ Enable mixed-precision training
                dataloader_pin_memory=True    # ✅ Enable pinned memory for faster host-device transfers
            )

        # 10) Define collator that handles dynamic padding + label alignment
        #     For example, if two tokenized examples have:
        #         input_ids = [[101, 2121, 5661, 2171, 102], [101, 2064, 102]]
        #     the collator will pad them to the same length and align
        #     their attention_mask and labels accordingly.
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


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
        viterbi_preds, fold_labels = viterbi_predict(model, tokenized_test, data_collator)

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

    # 17) Run prediction on hold-out set and record inference time
    holdout_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    start_time = time.perf_counter()
    val_preds, val_labels = viterbi_predict(best_model, tokenized_val, holdout_collator)
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
            "pred_tags":   " ".join(pred_tags)
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
        dual_print(f"\nInference Time: {elapsed:.2f}s for {total_examples} identifiers ({total_tokens} tokens)", file=f)
        dual_print(f"Tokens/sec: {total_tokens / elapsed:.2f}", file=f)
        dual_print(f"Identifiers/sec: {total_examples / elapsed:.2f}", file=f)
        dual_print(f"\nFinal Macro F1 on Held-Out Set: {final_macro_f1:.4f}", file=f)
        dual_print(f"Final Token-level Accuracy on Held-Out Set: {final_accuracy:.4f}", file=f)
        dual_print(f"Final Identifier-level Accuracy on Held-Out Set: {id_level_acc:.4f}", file=f)