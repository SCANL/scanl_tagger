"""
Full float32 inference, whatever precision settings the surrounding process has chosen.

Training turns on TF32 matrix multiplication for speed, and a program hosting the tagger may do
the same. TF32 moves CRF marginals by up to ~0.002, which the calibration map can magnify, so
tags would stay the same but per-token `p` would depend on the caller's settings. Inference and
the confidence measurements that calibration is fitted on both run inside `full_float32()`, so the
map is fitted on the same numbers serving returns.
"""

import threading
from contextlib import contextmanager

import torch

# The precision settings are process-wide, so two threads (e.g. the HTTP server's request pool)
# must not interleave saving and restoring them: one would switch TF32 back on while the other is
# still running, and the last to finish would overwrite the host's setting. Re-entrant, so a
# nested block in the same thread doesn't deadlock.
_PRECISION_LOCK = threading.RLock()


@contextmanager
def full_float32(device_type: str = "cpu"):
    """
    Disable TF32 and autocast for the duration of the block, then restore the caller's settings.
    Blocks from different threads run one at a time.
    """
    with _PRECISION_LOCK:
        saved = (
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type=device_type, enabled=False):
                yield
        finally:
            torch.set_float32_matmul_precision(saved[0])
            torch.backends.cuda.matmul.allow_tf32 = saved[1]
            torch.backends.cudnn.allow_tf32 = saved[2]
