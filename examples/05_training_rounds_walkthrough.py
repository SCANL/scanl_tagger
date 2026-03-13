from collections import Counter
from pathlib import Path
import math
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def summarize_contexts(frame):
    counts = Counter(frame["CONTEXT"])
    return ", ".join(f"{context}={counts[context]}" for context in sorted(counts))


def describe_dataset_row(dataset_row):
    return {
        "tokens_preview": dataset_row["tokens"][:12],
        "labels_preview": dataset_row["ner_tags"][:12],
        "total_tokens": len(dataset_row["tokens"]),
    }


def main():
    print("Example 5: how lm_based training proceeds from prepared rows to completed training rounds")
    print()

    try:
        from sklearn.model_selection import StratifiedKFold, train_test_split

        from src.lm_based_tagger.distilbert_preprocessing import prepare_dataset
        from src.lm_based_tagger.train_model import (
            HOLDOUT_RATIO,
            K,
            LABEL2ID,
            SPLIT_SEED,
            TRAIN_SEED,
            _build_position0_label_priors,
            _load_lm_training_dataframe,
            _prepare_training_frame,
        )
    except ImportError as exc:
        print("This example inspects the repository's actual LM training helpers.")
        print("Install the project dependencies first, then re-run this file.")
        print(f"Underlying error: {exc}")
        return

    try:
        full_df, source_names = _load_lm_training_dataframe(str(REPO_ROOT))
    except Exception as exc:
        print("Could not load the repository's LM training data.")
        print(f"Underlying error: {exc}")
        return

    print("Loaded sources:", source_names)
    print("Total raw examples:", len(full_df))
    print("Context distribution:", summarize_contexts(full_df))
    print()

    train_df, holdout_df = train_test_split(
        full_df,
        test_size=HOLDOUT_RATIO,
        random_state=SPLIT_SEED,
        stratify=full_df["CONTEXT"],
    )
    print(f"Holdout split: {len(train_df)} train / {len(holdout_df)} holdout")
    print("Holdout contexts:", summarize_contexts(holdout_df))
    print()

    priors = _build_position0_label_priors(train_df)
    print("Position-0 priors learned from the training slice:")
    print("Global priors:", len(priors["global"]))
    print("Context-specific priors:", sum(len(values) for values in priors["by_context"].values()))
    print()

    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SPLIT_SEED)
    chosen_epoch = None
    mock_best_fold = None
    mock_best_score = -1.0

    for fold_index, (fold_train_idx, fold_eval_idx) in enumerate(kf.split(train_df, train_df["CONTEXT"]), start=1):
        fold_train_df = train_df.iloc[fold_train_idx].reset_index(drop=True)
        fold_eval_df = train_df.iloc[fold_eval_idx].reset_index(drop=True)

        prepared_fold_train_df, verb_aug_count = _prepare_training_frame(
            fold_train_df,
            augmentation_seed=TRAIN_SEED + fold_index,
        )
        prepared_fold_dataset = prepare_dataset(prepared_fold_train_df, LABEL2ID)
        sample_preview = describe_dataset_row(prepared_fold_dataset[0])

        mock_macro_f1 = 0.80 + (0.02 * fold_index) + (verb_aug_count / max(len(prepared_fold_train_df), 1))
        mock_epoch = min(5.0, 2.0 + fold_index / 2.0)

        print(f"=== Fold {fold_index} ===")
        print(f"Base fold sizes: {len(fold_train_df)} train / {len(fold_eval_df)} eval")
        print(f"After in-fold resampling and augmentation: {len(prepared_fold_train_df)} train rows")
        print(f"Verb augmentation added: {verb_aug_count}")
        print("Prepared row preview:", sample_preview)
        print(
            "What the real trainer does next: tokenize, train DistilBERT+CRF, early-stop, then score this fold with Viterbi decoding."
        )
        print(f"Illustrative fold outcome: macro_f1~{mock_macro_f1:.4f}, best_epoch~{mock_epoch:.2f}")
        print()

        if mock_macro_f1 > mock_best_score:
            mock_best_score = mock_macro_f1
            mock_best_fold = fold_index
            chosen_epoch = mock_epoch

    full_train_prepared_df, final_verb_aug_count = _prepare_training_frame(
        train_df,
        augmentation_seed=TRAIN_SEED + K + 1,
    )
    final_dataset = prepare_dataset(full_train_prepared_df, LABEL2ID)

    print("=== Final retraining round ===")
    print(f"Chosen fold: {mock_best_fold}")
    print(f"Chosen epoch count: {chosen_epoch:.2f}")
    print(f"Final retrain rows after augmentation: {len(full_train_prepared_df)}")
    print(f"Final retrain verb augmentation added: {final_verb_aug_count}")
    print("Prepared final-training row preview:", describe_dataset_row(final_dataset[0]))
    print()

    estimated_warmup_steps = max(1, math.ceil(0.1 * (len(full_train_prepared_df) / 16) * chosen_epoch))
    print("What the real train_lm() function completes after this point:")
    print("- Retrains one final DistilBERT+CRF model on the full non-holdout split.")
    print("- Saves the best model, tokenizer, and config under output/best_model or the requested --model_dir.")
    print("- Runs holdout inference and writes output/holdout_predictions.csv.")
    print("- Writes holdout_report.txt with classification reports, selected features, chosen epoch count, and throughput.")
    print()
    print("Approximate final-training bookkeeping:")
    print(f"- Holdout identifiers: {len(holdout_df)}")
    print(f"- Final-training identifiers: {len(full_train_prepared_df)}")
    print(f"- Example warmup-steps estimate at batch size 16: {estimated_warmup_steps}")
    print()
    print("To run the real end-to-end training job, use:")
    print("python main --mode train --model_type lm_based")


if __name__ == "__main__":
    main()