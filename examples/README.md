# lm_based_tagger walkthrough

These examples are a staged path from "small sequence tagger" to "what this repository actually does."

The real lm_based_tagger is built around five ideas:

1. Each identifier is split into subtokens such as `get`, `employee`, `name`.
2. The code adds engineered helper tokens before those subtokens, such as context, type, language, and position markers.
3. A token-classification model predicts one grammar label per real identifier subtoken.
4. DistilBERT may split a token like `employee_name` into smaller pieces internally, so labels must be aligned back to the original token boundaries.
5. Training is not one pass over the data. The repository performs a holdout split, cross-validation rounds, final retraining, and then holdout evaluation.

## Recommended order

1. `python examples/01_build_lm_inputs.py`
2. `python examples/02_tiny_sequence_network.py`
3. `python examples/03_subword_alignment_demo.py`
4. `python examples/04_real_preprocessing_pipeline.py`
5. `python examples/05_training_rounds_walkthrough.py`

## What each example teaches

### 01_build_lm_inputs.py

Shows the exact input sequence shape used by the repository before DistilBERT sees anything.

You will see:

- the original identifier tokens
- the feature tokens added from metadata such as `CONTEXT`, `TYPE`, and `LANGUAGE`
- the `@pos_0`, `@pos_1`, `@pos_2` markers
- which label positions are ignored with `-100`

If you already understand feature engineering, start here.

### 02_tiny_sequence_network.py

Implements a very small neural sequence tagger in PyTorch:

- `Embedding` turns token ids into learned vectors
- `Linear` turns each vector into label scores, often called logits
- cross-entropy loss teaches the model to raise the correct label score
- ignored positions use `-100`, just like the repository

This is not DistilBERT. It is the smallest useful stepping stone toward it.

### 03_subword_alignment_demo.py

Shows why tokenization alignment exists at all.

The repository uses `DistilBertTokenizerFast(..., is_split_into_words=True)` and then uses `word_ids()` to map subword pieces back to the original token list. This example prints that mapping for one toy identifier.

This example may download `distilbert-base-uncased` the first time you run it.

### 04_real_preprocessing_pipeline.py

Calls the repository's actual helpers from [src/lm_based_tagger/distilbert_preprocessing.py](/home/wotterotter/scanl_tagger/src/lm_based_tagger/distilbert_preprocessing.py).

This is the closest match to the real training path in [src/lm_based_tagger/train_model.py](/home/wotterotter/scanl_tagger/src/lm_based_tagger/train_model.py).

It shows:

- `prepare_dataset(...)`
- `tokenize_and_align_labels(...)`
- how labels move from word level to subword-aware model inputs

### 05_training_rounds_walkthrough.py

Shows how the actual training job is organized after preprocessing has produced word-level rows.

It walks through:

- loading the real LM training sources
- the stratified holdout split
- conservative label-prior extraction for the first identifier token
- per-fold resampling and verb augmentation
- why cross-validation picks a training duration before final retraining
- which artifacts the real `train_lm(...)` call writes at the end

This example is intentionally lightweight. It explains the same control flow as the real trainer without launching a full DistilBERT training run.

## How this maps to the real model

Once these examples make sense, the real lm_based_tagger becomes much easier to read:

- preprocessing logic: [src/lm_based_tagger/distilbert_preprocessing.py](/home/wotterotter/scanl_tagger/src/lm_based_tagger/distilbert_preprocessing.py)
- training loop: [src/lm_based_tagger/train_model.py](/home/wotterotter/scanl_tagger/src/lm_based_tagger/train_model.py)
- inference wrapper: [src/lm_based_tagger/distilbert_tagger.py](/home/wotterotter/scanl_tagger/src/lm_based_tagger/distilbert_tagger.py)
- CRF model wrapper: [src/lm_based_tagger/distilbert_crf.py](/home/wotterotter/scanl_tagger/src/lm_based_tagger/distilbert_crf.py)

## Mental model for the real architecture

Read the model as:

`engineered token sequence -> DistilBERT contextual representations -> linear label scores -> optional CRF sequence cleanup`

The CRF layer does not replace DistilBERT. It sits on top of DistilBERT's per-token scores and prefers globally consistent tag sequences.

Read the training loop as:

`load raw rows -> prepare token sequences -> tokenize and align labels -> score CV folds -> choose training duration -> final retrain -> evaluate on holdout`