# train_model.py

import os
import time
import random

import numpy as np
import pandas as pd
import torch

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
# Match test.py’s seed settings for reproducibility :contentReference[oaicite:0]{index=0}
RAND_STATE = 209
random.seed(RAND_STATE)
np.random.seed(RAND_STATE)
torch.manual_seed(RAND_STATE)
torch.cuda.manual_seed_all(RAND_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Hyperparameters / Config ===
K = 2                     # number of CV folds
HOLDOUT_RATIO = 0.15      # 15% held out for final evaluation
EPOCHS = 5               # number of epochs per fold
EARLY_STOP = 2            # patience for early stopping
LOW_FREQ_TAGS = {"CJ", "VM", "PRE", "V"}

# === Label List & Mappings (unchanged from your original) :contentReference[oaicite:1]{index=1} ===
LABEL_LIST = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL   = {i: label for label, i in LABEL2ID.items()}


def train_lm(script_dir: str):
    # 1) Paths
    input_path = os.path.join(script_dir, "input", "tagger_data.tsv")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2) Read the TSV & build “tokens” / “tags” columns :contentReference[oaicite:2]{index=2}
    df = pd.read_csv(input_path, sep="\t", dtype=str).dropna(subset=["SPLIT", "GRAMMAR_PATTERN"])
    df = df[df["SPLIT"].str.strip().astype(bool)]
    df["tokens"] = df["SPLIT"].apply(lambda x: x.strip().split())
    df["tags"]   = df["GRAMMAR_PATTERN"].apply(lambda x: x.strip().split())
    # Keep only rows where len(tokens) == len(tags)
    df = df[df.apply(lambda r: len(r["tokens"]) == len(r["tags"]), axis=1)]

    # 3) Initial Train/Val Split (15% hold-out) :contentReference[oaicite:3]{index=3}
    train_df, val_df = train_test_split(
        df,
        test_size=HOLDOUT_RATIO,
        random_state=RAND_STATE,
        stratify=df["CONTEXT"]
    )

    # 4) Upsample low-frequency tags **in the training set only** :contentReference[oaicite:4]{index=4}
    low_freq_df = train_df[train_df["tags"].apply(lambda tags: any(t in LOW_FREQ_TAGS for t in tags))]
    train_df_upsampled = pd.concat([train_df] + [low_freq_df] * 2, ignore_index=True)

    # 5) Tokenizer (uncased, matching test.py) 
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # 6) Prepare final hold-out “validation” Dataset :contentReference[oaicite:5]{index=5}
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

        # 7b) Build HuggingFace Datasets via prepare_dataset(...) :contentReference[oaicite:6]{index=6}
        fold_train_dataset = prepare_dataset(fold_train_df, LABEL2ID)
        fold_test_dataset  = prepare_dataset(fold_test_df, LABEL2ID)

        # 7c) Tokenize + align labels (exactly as before) :contentReference[oaicite:7]{index=7}
        tokenized_train = fold_train_dataset.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=False
        )
        tokenized_test = fold_test_dataset.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=False
        )

        # 8) Build fresh model + config for this fold :contentReference[oaicite:8]{index=8}
        config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        model = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-uncased",
            config=config
        )
        model.to(device)

        # 9) TrainingArguments (with early stopping) :contentReference[oaicite:9]{index=9}
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

        # 10) Data collator (dynamic padding) :contentReference[oaicite:10]{index=10}
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # 11) compute_metrics function (macro-F1) :contentReference[oaicite:11]{index=11}
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)

            true_preds = []
            true_labels = []
            for pred_row, label_row in zip(preds, labels):
                for p, l in zip(pred_row, label_row):
                    if l != -100:
                        true_preds.append(p)
                        true_labels.append(l)

            macro_f1 = f1_score(true_labels, true_preds, average="macro")
            return {"eval_macro_f1": macro_f1}

        # 12) Trainer for this fold (with EarlyStopping) :contentReference[oaicite:12]{index=12}
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
            tokenizer.save_pretrained(best_model_dir)

        fold += 1

    # 16) After all folds, report best fold‐score & load best model for final evaluation
    print(f"\nBest fold model saved at: {best_model_dir}, Macro F1 = {best_macro_f1:.4f}")

    # 17) Final Evaluation on held-out val_df
    best_model = DistilBertForTokenClassification.from_pretrained(best_model_dir)
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

    print("\nFinal Evaluation on Held-Out Set:")
    print(classification_report(flat_true, flat_pred))

    # Report inference speed
    total_tokens = sum(len(ex["tokens"]) for ex in val_dataset)
    total_examples = len(val_dataset)
    elapsed = end_time - start_time
    print(f"\nInference Time: {elapsed:.2f}s for {total_examples} identifiers ({total_tokens} tokens)")
    print(f"Tokens/sec: {total_tokens / elapsed:.2f}")
    print(f"Identifiers/sec: {total_examples / elapsed:.2f}")

    final_macro_f1 = f1_score(flat_true, flat_pred, average="macro")
    print(f"\nFinal Macro F1 on Held-Out Set: {final_macro_f1:.4f}")
