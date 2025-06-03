import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DistilBertTokenizerFast,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from src.lm_based_tagger.distilbert_preprocessing import prepare_dataset, tokenize_and_align_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Labels & Mappings ===
LABEL_LIST = ["CJ", "D", "DT", "N", "NM", "NPL", "P", "PRE", "V", "VM"]
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL   = {i: label for label, i in LABEL2ID.items()}

def train_lm(script_dir: str):
    input_path = os.path.join(script_dir, "input", "tagger_data.tsv")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Read TSV & build tokens/tags lists
    df = pd.read_csv(input_path, sep="\t", dtype=str).dropna(subset=["SPLIT", "GRAMMAR_PATTERN"])
    df = df[df["SPLIT"].str.strip().astype(bool)]
    df["tokens"] = df["SPLIT"].apply(lambda x: x.strip().split())
    df["tags"]   = df["GRAMMAR_PATTERN"].apply(lambda x: x.strip().split())
    df = df[df.apply(lambda r: len(r["tokens"]) == len(r["tags"]), axis=1)]

    # 2) Train/Test split (stratify by CONTEXT)
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["CONTEXT"]
    )

    # 3) Upsample low-frequency tags (in training set only)
    low_freq_tags = {"CJ", "VM", "PRE", "V"}
    low_freq_df = train_df[train_df["tags"].apply(lambda tags: any(t in low_freq_tags for t in tags))]
    train_df = pd.concat([train_df] + [low_freq_df] * 2, ignore_index=True)

    # 4) Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # 5) Convert each split into a HF Dataset via the shared prepare_dataset(...)
    train_dataset = prepare_dataset(train_df, LABEL2ID)
    test_dataset  = prepare_dataset(test_df, LABEL2ID)

    # 6) Tokenize + align labels
    tokenized_train = train_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=False
    )
    tokenized_test = test_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=False
    )

    # 7) Build config & model using uncased vocab
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

    if device == "cpu":
        # 8) Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_dir=os.path.join(output_dir, "logs"),
            report_to="none",
            seed=42
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=4,  # ↓ reduce to fit in VRAM
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # simulates batch size of 16
            num_train_epochs=10,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_macro_f1",  # or "eval_loss" if macro F1 isn't computed
            greater_is_better=True,  # set to False if using loss
            logging_dir=os.path.join(output_dir, "logs"),
            report_to="none",
            seed=42,
            fp16=False,
            dataloader_pin_memory=False,  # no benefit if no CUDA pinning
        )

    # 9) Collate Data
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 10) Macro‐F1 computation
    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score
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

    # 11) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 12) Train & save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
