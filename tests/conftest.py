import os
import sys

# Tests import `src` and `version` from the repository root, the same way `main` does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeTagger:
    """Tags every word N, and returns None (truncated) for rows over `max_words` words."""

    selected_features = ["context"]
    pattern_postprocessing = False

    def __init__(self, max_words=50, fail_on=None):
        self.max_words = max_words
        self.fail_on = fail_on
        self.calls = []

    def tag_identifiers(self, rows, batch_size=64):
        self.calls.append(rows)
        if self.fail_on and any(self.fail_on in row["tokens"] for row in rows):
            raise RuntimeError("model failure")
        return [None if len(row["tokens"]) > self.max_words else ["N"] * len(row["tokens"]) for row in rows]
