import random

import pandas as pd
import pytest

from scalar_tagger.calibration import (
    calibrate, confidence_fields, confidence_table, fit_isotonic, token_confidence,
)

MAP = {"method": "isotonic", "x": [0.2, 0.6, 0.9, 0.99], "y": [0.1, 0.5, 0.5, 0.95]}


def test_calibrate_interpolates_and_clips():
    assert calibrate(0.7, None) == 0.7
    assert calibrate(0.0, MAP) == 0.1 and calibrate(1.0, MAP) == 0.95   # clipped at both ends
    assert calibrate(0.4, MAP) == pytest.approx(0.3)
    assert calibrate(0.75, MAP) == pytest.approx(0.5)                     # flat segment
    assert calibrate(0.99, MAP) == 0.95


def test_calibration_never_reverses_order():
    values = [i / 1000 for i in range(1001)]
    mapped = [calibrate(v, MAP) for v in values]
    assert all(a <= b for a, b in zip(mapped, mapped[1:]))


def test_confidence_fields_without_calibration_are_the_raw_marginals():
    marginals = {"N": 0.8, "NM": 0.15, "V": 0.05}
    assert confidence_fields("N", marginals) == {"p": 0.8, "alt": {"tag": "NM", "p": 0.15}}


def test_calibrated_alt_shares_the_leftover_probability():
    marginals = {"N": 0.6, "NM": 0.3, "V": 0.1}
    fields = confidence_fields("N", marginals, MAP)
    assert fields["p"] == 0.5
    assert fields["alt"] == {"tag": "NM", "p": 0.375}   # 0.5 left over, NM had 3/4 of the raw 0.4
    # A raw marginal of exactly 1 leaves nothing to share; the runner-up gets all of 1 - p
    certain = confidence_fields("N", {"N": 1.0, "NM": 0.0}, MAP)
    assert certain == {"p": 0.95, "alt": {"tag": "NM", "p": 0.05}}
    # Viterbi's tag can have a lower marginal than the runner-up; p + alt.p still stays within 1
    odd = confidence_fields("N", {"N": 0.4, "NM": 0.6}, MAP)
    assert odd["p"] + odd["alt"]["p"] <= 1.0 + 1e-9


def test_fit_isotonic_matches_observed_accuracy():
    rng = random.Random(0)
    p = [rng.random() for _ in range(5000)]
    correct = [rng.random() < v * 0.8 for v in p]   # true accuracy is 0.8 p
    fitted = fit_isotonic(p, correct)
    assert fitted["method"] == "isotonic" and fitted["tokens"] == 5000
    assert fitted["x"] == sorted(set(fitted["x"]))
    assert all(a <= b for a, b in zip(fitted["y"], fitted["y"][1:]))
    assert 0 < fitted["y"][0] and fitted["y"][-1] < 1   # smoothing keeps it off exactly 0 and 1
    for v in (0.25, 0.5, 0.9):
        assert calibrate(v, fitted) == pytest.approx(0.8 * v, abs=0.08)


def test_token_confidence_reads_prediction_columns():
    frame = pd.DataFrame({
        "true_tags": ["V N", "N", "N"],
        "pred_tags": ["V NM", "N", "N"],
        "p": ["0.9900 0.6000", "0.9990", ""],   # the last row has no confidence and is skipped
    })
    assert token_confidence(frame) == ([0.99, 0.6, 0.999], [True, False, True])


def test_confidence_table_bins_and_flags():
    table = confidence_table([0.3, 0.85, 0.92, 0.999, 0.999], [False, True, False, True, True])
    lines = table.splitlines()
    assert lines[1].split()[:3] == ["under", "0.5", "1"]
    assert "5 tokens, 2 wrong tags." in table
    assert "Flagging tokens under 0.95 marks 60.0% of tokens and catches 2 of 2 errors." in table
    assert confidence_table([], []) == "(no confidence values)"


def test_fit_isotonic_interpolates_instead_of_stepping():
    # Two clean groups: plain isotonic would give every token in a group the same value
    p = [0.1 + 0.001 * i for i in range(100)] + [0.9 + 0.0005 * i for i in range(100)]
    correct = [False] * 100 + [True] * 100
    fitted = fit_isotonic(p, correct)
    assert fitted["y"] == [round(1 / 102, 4), round(101 / 102, 4)]
    assert calibrate(0.5, fitted) > calibrate(0.4, fitted)   # rises strictly between block centres
