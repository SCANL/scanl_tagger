import os
import time
import random

import numpy as np
import pandas as pd
import torch
from .distilbert_crf import DistilBertCRFForTokenClassification

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

from transformers import (
    Trainer,
    TrainingArguments,
    DistilBertTokenizerFast,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

from datasets import Dataset
from src.lm_based_tagger.distilbert_preprocessing import prepare_dataset, tokenize_and_align_labels

# If CUDA is available, use it; otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Random Seeds ===
RAND_STATE = 209
random.seed(RAND_STATE)
np.random.seed(RAND_STATE)
torch.manual_seed(RAND_STATE)
torch.cuda.manual_seed_all(RAND_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Hyperparameters / Config ===
K = 5                     # number of CV folds
HOLDOUT_RATIO = 0.15      # 15% held out for final evaluation
EPOCHS = 10            # number of epochs per fold
EARLY_STOP = 2            # patience for early stopping
LOW_FREQ_TAGS = {"CJ", "VM", "PRE", "V"}

# === Label List & Mappings ===
LABEL_LIST = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL   = {i: label for label, i in LABEL2ID.items()}

def dual_print(*args, file, **kwargs):
    print(*args, **kwargs)         # stdout
    print(*args, file=file, **kwargs)  # file

def train_lm(script_dir: str):
    # 1) Paths
    input_path = os.path.join(script_dir, "input", "tagger_data.tsv")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2) Read the TSV & build “tokens” / “tags” columns 
    df = pd.read_csv(input_path, sep="\t", dtype=str).dropna(subset=["SPLIT", "GRAMMAR_PATTERN"])
    df = df[df["SPLIT"].str.strip().astype(bool)]
    df["tokens"] = df["SPLIT"].apply(lambda x: x.strip().split())
    df["tags"]   = df["GRAMMAR_PATTERN"].apply(lambda x: x.strip().split())
    # Keep only rows where len(tokens) == len(tags)
    df = df[df.apply(lambda r: len(r["tokens"]) == len(r["tags"]), axis=1)]

    # 3) Initial Train/Val Split (15% hold-out) 
    train_df, val_df = train_test_split(
        df,
        test_size=HOLDOUT_RATIO,
        random_state=RAND_STATE,
        stratify=df["CONTEXT"]
    )

    # 4) Upsample low-frequency tags **in the training set only** 
    low_freq_df = train_df[train_df["tags"].apply(lambda tags: any(t in LOW_FREQ_TAGS for t in tags))]
    train_df_upsampled = pd.concat([train_df] + [low_freq_df] * 2, ignore_index=True)

    # 5) Tokenizer
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
    for train_idx, test_idx in kf.split(train_df_upsampled):
        print(f"\n=== Fold {fold} ===")

        # 7a) Split the upsampled DataFrame into this fold’s train/test
        fold_train_df = train_df_upsampled.iloc[train_idx].reset_index(drop=True)
        fold_test_df  = train_df_upsampled.iloc[test_idx].reset_index(drop=True)

        # 7b) Build HuggingFace Datasets via prepare_dataset(...) 
        fold_train_dataset = prepare_dataset(fold_train_df, LABEL2ID)
        fold_test_dataset  = prepare_dataset(fold_test_df, LABEL2ID)

        # 7c) Tokenize + align labels (exactly as before) 
        tokenized_train = fold_train_dataset.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=False
        )
        tokenized_test = fold_test_dataset.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
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
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_macro_f1",
                greater_is_better=True,
                save_total_limit=1,
                logging_dir=os.path.join(output_dir, "logs", f"fold_{fold}"),
                report_to="none",
                seed=RAND_STATE
            )
        else:
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"fold_{fold}"),
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=4,   # smaller per-GPU batch size
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=4,   # to simulate batch size = 16
                num_train_epochs=EPOCHS,
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_macro_f1",
                greater_is_better=True,
                save_total_limit=1,
                logging_dir=os.path.join(output_dir, "logs", f"fold_{fold}"),
                report_to="none",
                seed=RAND_STATE,
                fp16=False,
                dataloader_pin_memory=False
            )

        # 10) Data collator (dynamic padding) 
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # 11) compute_metrics function (macro-F1) 

        def compute_metrics(eval_pred):
            """
            Works for both:
                • Plain classifier logits  → argmax along last dim
                • CRF Viterbi paths (list/2‑D ndarray) → use directly
            Returns:
                - eval_macro_f1
                - eval_token_accuracy
                - eval_identifier_accuracy
            """
            # ── 1. Unpack ────────────────────────────────────────────────────
            if isinstance(eval_pred, tuple):          # older HF (<4.38)
                preds, labels = eval_pred
            else:                                     # EvalPrediction obj
                preds  = eval_pred.predictions
                labels = eval_pred.label_ids

            # ── 2. Convert logits → label IDs if needed ─────────────────────
            #    * 3‑D tensor  : [B, T, C]  → argmax(C)
            #    * 2‑D tensor  : already IDs
            #    * list/obj‑nd : variable‑length decode paths
            if isinstance(preds, np.ndarray) and preds.ndim == 3:
                preds = np.argmax(preds, axis=-1)     # [B, T]
            elif isinstance(preds, list):
                preds = np.array(preds, dtype=object) # each row is a list

            # ── 3. Accumulate token & identifier stats ──────────────────────
            all_true, all_pred, id_correct_flags = [], [], []

            for pred_row, label_row in zip(preds, labels):
                ptr = 0
                example_correct = True

                for lbl in label_row:                 # iterate gold labels
                    if lbl == -100:                   # skip padding / specials
                        continue

                    # pick the corresponding prediction
                    if isinstance(pred_row, (list, np.ndarray)):
                        pred_lbl = pred_row[ptr]
                    else:                             # pred_row is scalar
                        pred_lbl = pred_row
                    ptr += 1

                    all_true.append(lbl)
                    all_pred.append(pred_lbl)
                    if pred_lbl != lbl:
                        example_correct = False

                id_correct_flags.append(example_correct)

            # ── 4. Metrics ──────────────────────────────────────────────────
            macro_f1  = f1_score(all_true, all_pred, average="macro")
            token_acc = accuracy_score(all_true, all_pred)
            id_acc    = float(sum(id_correct_flags)) / len(id_correct_flags)

            return {
                "eval_macro_f1":          macro_f1,
                "eval_token_accuracy":    token_acc,
                "eval_identifier_accuracy": id_acc,
            }

        # 12) Trainer for this fold (with EarlyStopping) 
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP)],
            compute_metrics=compute_metrics
        )
        # Avoid deprecation warning (explicitly set tokenizer on trainer)
        trainer.tokenizer = tokenizer

        # 13) Train this fold
        trainer.train()

        # 14) Evaluate on this fold’s held-out split
        preds_logits, labels, _ = trainer.predict(tokenized_test)
        preds = np.argmax(preds_logits, axis=-1)

        # Convert to (flattened) label strings for F1
        true_labels_list = [
            ID2LABEL[l]
            for sent_labels, sent_preds in zip(labels, preds)
            for (l, p) in zip(sent_labels, sent_preds)
            if l != -100
        ]
        pred_labels_list = [
            ID2LABEL[p]
            for sent_labels, sent_preds in zip(labels, preds)
            for (l, p) in zip(sent_labels, sent_preds)
            if l != -100
        ]

        fold_macro_f1 = f1_score(true_labels_list, pred_labels_list, average="macro")
        print(f"Fold {fold} Macro F1: {fold_macro_f1:.4f}")

        # 15) If this fold’s model is the best so far, save it
        if fold_macro_f1 > best_macro_f1:
            best_macro_f1 = fold_macro_f1
            best_model_dir = os.path.join(output_dir, "best_model")
            trainer.save_model(best_model_dir)
            model.config.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

        fold += 1

    # 16) After all folds, report best fold‐score & load best model for final evaluation
    print(f"\nBest fold model saved at: {best_model_dir}, Macro F1 = {best_macro_f1:.4f}")

    # 17) Final Evaluation on held-out val_df
    best_model = DistilBertCRFForTokenClassification.from_pretrained(best_model_dir)
    best_model.to(device)

    # Build a fresh set of TrainingArguments that never runs evaluation epochs:
    final_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "final_eval"),
        per_device_eval_batch_size=16,
        eval_strategy="no",
        save_strategy="no",
        logging_dir=os.path.join(output_dir, "logs", "final_eval"),
        report_to="none",
        seed=RAND_STATE
    )
    val_trainer = Trainer(
        model=best_model,
        args=final_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
        # ← note: no eval_dataset here, because we’ll call .predict(...) manually
    )

    start_time = time.perf_counter()
    val_preds_logits, val_labels, _ = val_trainer.predict(tokenized_val)
    end_time = time.perf_counter()

    val_preds = np.argmax(val_preds_logits, axis=-1)

    flat_true = [
        ID2LABEL[l]
        for sent_labels, sent_preds in zip(val_labels, val_preds)
        for (l, p) in zip(sent_labels, sent_preds)
        if l != -100
    ]
    flat_pred = [
        ID2LABEL[p]
        for sent_labels, sent_preds in zip(val_labels, val_preds)
        for (l, p) in zip(sent_labels, sent_preds)
        if l != -100
    ]

    # 18) Write hold-out predictions to CSV so that each row contains
    #     (tokens, true_tags, pred_tags) for sanity checking.
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
    
    # Report inference speed
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