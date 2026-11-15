import pandas as pd
import pytest

from src.lm_based_tagger import evaluation


def test_token_macro_f1_counts_each_tag_equally():
    true = [["V", "NM", "N"], ["N"]]
    pred = [["V", "N", "N"], ["N"]]
    # V: F1 1; NM: 0; N: tp 2, fp 1 -> 0.8
    assert evaluation.token_macro_f1(true, pred) == pytest.approx((1 + 0 + 0.8) / 3)
    assert evaluation.token_macro_f1(true, true) == 1.0


def test_real_code_summary_combines_real_subsets_only():
    preds = pd.DataFrame({
        "data_source": ["tagger_data", "tagger_data", "synthetic"],
        "subset": ["original_set", "headnoun_set", "synthetic"],
        "true_tags": ["V N", "N", "N"],
        "pred_tags": ["V N", "NM", "N"],
    })
    summary = evaluation.real_code_summary(preds).set_index("source")
    assert list(summary.index) == ["real", "headnoun_set", "original_set"]
    assert summary.loc["real", "identifiers"] == 2
    assert summary.loc["real", "identifier_accuracy"] == 0.5


def test_real_code_summary_keeps_new_subsets_out_of_headline():
    preds = pd.DataFrame({
        "data_source": ["tagger_data", "tagger_data"],
        "subset": ["original_set", "targeted_set"],
        "true_tags": ["V N", "N"],
        "pred_tags": ["V N", "NM"],
    })
    summary = evaluation.real_code_summary(preds).set_index("source")
    assert list(summary.index) == ["real", "real+new", "original_set", "targeted_set"]
    assert summary.loc["real", "identifier_accuracy"] == 1.0
    assert summary.loc["real+new", "identifiers"] == 2
    assert summary.loc["real+new", "identifier_accuracy"] == 0.5


def _write_run(reports, run, seed, rows, cv_rows=None):
    columns = ["tokens", "true_tags", "pred_tags", "context", "data_source", "subset"]
    pd.DataFrame(rows, columns=columns).to_csv(reports / f"{run}_seed{seed}_predictions.csv", index=False)
    if cv_rows is not None:
        pd.DataFrame(cv_rows, columns=columns).to_csv(reports / f"{run}_seed{seed}_cv_predictions.csv", index=False)


def test_load_run_reads_every_seed_and_both_splits(tmp_path):
    row = ("get name", "V N", "V N", "FUNCTION", "tagger_data", "original_set")
    _write_run(tmp_path, "tc", 7, [row], cv_rows=[row])
    _write_run(tmp_path, "tc", 42, [row])
    _write_run(tmp_path, "tc_more", 7, [row])   # another run whose name starts the same way
    seeds = evaluation.load_run("tc", str(tmp_path))
    assert sorted(seeds) == [7, 42]
    assert sorted(seeds[7]["eval_split"]) == ["cv", "holdout"]
    with pytest.raises(FileNotFoundError):
        evaluation.load_run("missing", str(tmp_path))


def test_compare_runs_rescores_and_uses_shared_identifiers(tmp_path):
    gold = tmp_path / "gold.tsv"
    pd.DataFrame({
        "SPLIT": ["get name", "max size", "buf out"],
        "CONTEXT": ["FUNCTION", "DECLARATION", "PARAMETER"],
        "GRAMMAR_PATTERN": ["V N", "NM N", "N NM"],   # buf out relabeled since run A
    }).to_csv(gold, sep="\t", index=False)

    a_rows = [
        ("get name", "V N", "V N", "FUNCTION", "tagger_data", "original_set"),
        ("max size", "NM N", "N N", "DECLARATION", "tagger_data", "original_set"),
        ("buf out", "NM N", "NM N", "PARAMETER", "tagger_data", "closed_category_set"),
        ("only in a", "N N", "N N", "PARAMETER", "tagger_data", "closed_category_set"),
        ("syn word", "NM N", "NM N", "FUNCTION", "synthetic", "synthetic"),
    ]
    b_rows = [
        ("get name", "V N", "V N", "FUNCTION", "tagger_data", "original_set"),
        ("max size", "NM N", "NM N", "DECLARATION", "tagger_data", "original_set"),
        ("buf out", "N NM", "N NM", "PARAMETER", "tagger_data", "closed_category_set"),
        ("syn word", "NM N", "NM N", "FUNCTION", "synthetic", "synthetic"),
    ]
    for seed in (7, 42):
        _write_run(tmp_path, "a", seed, a_rows)
        _write_run(tmp_path, "b", seed, b_rows)

    report = evaluation.compare_runs(["a", "b"], reports_dir=str(tmp_path), gold_path=str(gold), reps=200)
    assert "Identifiers scored in every seed of every run: 4" in report   # "only in a" dropped
    real_line = next(line for line in report.splitlines() if line.startswith("real "))
    # A: get name right, max size wrong, buf out wrong under current gold -> 33.33; B: all right
    assert "33.33" in real_line and "100.00" in real_line and "+66.67" in real_line
    assert any(line.startswith("synthetic") for line in report.splitlines())
    assert "Per-tag F1 on real identifiers" in report

    single = evaluation.compare_runs(["b"], reports_dir=str(tmp_path), gold_path=str(gold), reps=50)
    assert "diff" not in single


def test_run_folds_reads_reports(tmp_path):
    (tmp_path / "a_seed1.txt").write_text("x\nCV folds: 3, holdout ratio: 0.2\n")
    (tmp_path / "a_seed2.txt").write_text("CV folds: 5, holdout ratio: 0.2\n")
    (tmp_path / "ab_seed1.txt").write_text("CV folds: 7, holdout ratio: 0.2\n")
    assert evaluation.run_folds("a", str(tmp_path)) == {3, 5}
    assert evaluation.run_folds("missing", str(tmp_path)) == set()
