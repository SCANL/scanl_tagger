# SCALAR Part-of-speech tagger

THIS IS AN EXPERIMENTAL VERSION OF SCALAR

Install requirements via `pip install -r requirements.txt`

Run via `python3 main --mode run --model_type lm_based`

You can attempt to train it `python main --mode train --model_type lm_based` -- but I make no guarantees about how easily it will work at this stage

It still technically supports the old gradientboost model, too... but no guarantees as to how well it functions in this branch.

## Evaluation Results (Held-Out Set)

### Per-Class Metrics

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| CJ    | 0.88      | 0.88   | 0.88     | 8       |
| D     | 0.98      | 0.96   | 0.97     | 52      |
| DT    | 0.95      | 0.93   | 0.94     | 45      |
| N     | 0.94      | 0.94   | 0.94     | 418     |
| NM    | 0.91      | 0.93   | 0.92     | 440     |
| NPL   | 0.97      | 0.97   | 0.97     | 79      |
| P     | 0.94      | 0.92   | 0.93     | 79      |
| PRE   | 0.79      | 0.79   | 0.79     | 68      |
| V     | 0.89      | 0.84   | 0.86     | 110     |
| VM    | 0.79      | 0.85   | 0.81     | 13      |

### Aggregate Metrics

| Metric              | Score  |
|---------------------|--------|
| Accuracy            | 0.92   |
| Macro Avg F1        | 0.90   |
| Weighted Avg F1     | 0.92   |
| Total Examples      | 1312   |

### Inference Statistics

- **Inference Time:** 1.74s for 392 identifiers (3746 tokens)  
- **Tokens/sec:** 2157.78  
- **Identifiers/sec:** 225.80  

### Final Scores

- **Final Macro F1 on Held-Out Set:** 0.9032  
- **Final Token-level Accuracy:** 0.9223  
- **Final Identifier-level Accuracy:** 0.8291  
