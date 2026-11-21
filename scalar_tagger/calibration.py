"""
Per-token confidence: turning CRF marginals into the `p` and `alt` fields, calibrating `p`, and
measuring how often tags are right at each confidence level.

Calibration is an isotonic (monotone) map from the CRF's marginal probability of the returned tag
to the observed accuracy of tags with that probability, fitted on out-of-fold predictions for real
identifiers. The checkpoint config stores it as breakpoints, `{"x": [...], "y": [...]}`, applied by
linear interpolation and clipped at both ends, so serving needs no extra dependencies. The map never
changes a tag and never reverses the order of two tokens by `p` (tokens can tie where it is flat).
"""

from bisect import bisect_right
from typing import Dict, List, Optional, Sequence

# Bin edges for confidence tables: under 0.5, 0.5-0.8, ..., 0.995 and above.
CONFIDENCE_BINS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.995)


def calibrate(p: float, calibration: Optional[dict]) -> float:
    """Apply a stored isotonic map to a raw probability; without one, return `p` unchanged."""
    if not calibration:
        return p
    xs, ys = calibration["x"], calibration["y"]
    if p <= xs[0]:
        return ys[0]
    if p >= xs[-1]:
        return ys[-1]
    i = bisect_right(xs, p)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (p - x0) / (x1 - x0)


def confidence_fields(tag: str, marginals: Dict[str, float], calibration: Optional[dict] = None) -> dict:
    """
    The `p` and `alt` fields for one token, from its CRF marginals ({tag: probability}).

    `p` is the (calibrated, when the checkpoint has a map) probability of the returned tag. `alt` is
    the most probable other tag. With calibration, the probability left over after `p` (1 - p) is
    shared among the other tags in proportion to their raw marginals, so `p + alt.p` never exceeds 1.
    """
    alt_tag = max((t for t in marginals if t != tag), key=marginals.get)
    raw_p, raw_alt = float(marginals[tag]), float(marginals[alt_tag])
    p, alt_p = raw_p, raw_alt
    if calibration:
        p = calibrate(raw_p, calibration)
        rest = 1.0 - raw_p
        alt_p = (1.0 - p) * (raw_alt / rest if rest > 1e-12 else 1.0)
        alt_p = min(alt_p, 1.0 - p)
    return {"p": round(p, 4), "alt": {"tag": alt_tag, "p": round(alt_p, 4)}}


def fit_isotonic(p: Sequence[float], correct: Sequence[bool]) -> dict:
    """
    Fit the isotonic map from raw `p` to accuracy (training only; needs scikit-learn).

    Plain isotonic regression is a step function: it pools tokens into blocks and gives every
    token in a block the same value, which ties tokens the raw `p` had ranked, and a block with no
    errors (or no correct tags) gets exactly 1 (or 0). So each block's accuracy is smoothed with one
    pseudo-success and one pseudo-failure, (correct + 1) / (tokens + 2), made monotone again, and
    placed at the block's mean raw `p`; the map interpolates linearly between those points. It
    rises strictly between the first and last block centres, so it keeps the raw ranking there.
    """
    from sklearn.isotonic import IsotonicRegression

    xs = [float(v) for v in p]
    ys = [1.0 if c else 0.0 for c in correct]
    fitted = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(xs, ys).predict(xs)

    blocks: Dict[float, List[float]] = {}   # fitted value -> [tokens, correct, sum of raw p]
    for x, y, value in zip(xs, ys, fitted):
        block = blocks.setdefault(round(float(value), 12), [0, 0.0, 0.0])
        block[0] += 1
        block[1] += y
        block[2] += x
    ordered = [blocks[value] for value in sorted(blocks)]
    centres = [total_p / n for n, _, total_p in ordered]
    smoothed = [(k + 1) / (n + 2) for n, k, _ in ordered]
    # Smoothing can reverse two neighbouring blocks (a small block moves more); pool them again.
    smoothed = IsotonicRegression(increasing=True).fit(
        centres, smoothed, sample_weight=[n for n, _, _ in ordered]
    ).predict(centres)

    points: List[tuple] = []
    for x, y in zip(centres, smoothed):
        x, y = round(x, 6), round(float(y), 4)
        if points and points[-1][0] == x:
            points[-1] = (x, max(points[-1][1], y))
        else:
            points.append((x, y))
    return {
        "method": "isotonic",
        "x": [x for x, _ in points],
        "y": [y for _, y in points],
        "tokens": len(xs),
    }


def token_confidence(frame, p_column: str = "p"):
    """(p, correct) per token from a predictions frame with space-separated `p_column` values."""
    ps, correct = [], []
    for true, pred, pvals in zip(frame["true_tags"], frame["pred_tags"], frame[p_column]):
        if not isinstance(pvals, str) or not pvals.strip():
            continue
        values = [float(v) for v in pvals.split()]
        true_tags, pred_tags = true.split(), pred.split()
        if not (len(values) == len(true_tags) == len(pred_tags)):
            continue
        ps.extend(values)
        correct.extend(t == q for t, q in zip(true_tags, pred_tags))
    return ps, correct


def confidence_table(p: Sequence[float], correct: Sequence[bool], bins: Sequence[float] = CONFIDENCE_BINS) -> str:
    """How often tags are right at each confidence level, plus what flagging below each edge catches."""
    n = len(p)
    if n == 0:
        return "(no confidence values)"
    errors = sum(not c for c in correct)
    edges = [0.0, *bins, float("inf")]
    lines = [f"{'p':<14}{'tokens':>8}{'share':>8}{'accuracy':>10}{'mean p':>8}"]
    for lo, hi in zip(edges, edges[1:]):
        members = [(v, c) for v, c in zip(p, correct) if lo <= v < hi]
        label = f"under {hi:g}" if lo == 0.0 else (f"{lo:g} or higher" if hi == float("inf") else f"{lo:g} to {hi:g}")
        if not members:
            lines.append(f"{label:<14}{0:>8}{0:>8.1%}{'':>10}{'':>8}")
            continue
        acc = sum(c for _, c in members) / len(members)
        mean_p = sum(v for v, _ in members) / len(members)
        lines.append(f"{label:<14}{len(members):>8}{len(members) / n:>8.1%}{acc:>10.1%}{mean_p:>8.3f}")
    lines.append(f"{n} tokens, {errors} wrong tags.")
    for edge in bins[1:]:
        flagged = [c for v, c in zip(p, correct) if v < edge]
        caught = sum(not c for c in flagged)
        lines.append(f"Flagging tokens under {edge:g} marks {len(flagged) / n:.1%} of tokens "
                     f"and catches {caught} of {errors} errors.")
    return "\n".join(lines)
