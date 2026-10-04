"""
Serve the tagging contract over stdin/stdout, one JSON message per line.

The parent process (for example the naming-analysis CLI) spawns the tagger and talks to it
through pipes, so there are no ports to manage and the tagger exits when its parent closes
stdin. Protocol:

    -> (startup)                   <- {"ready": true, "model": {...}}
    -> {"id": 1, "identifiers": [...]}   <- {"id": 1, "model": {...}, "results": [...]}
    -> {"id": 2, "command": "info"}      <- {"id": 2, "model": {...}}
    -> (stdin closed)              process exits with status 0

Only protocol messages are written to stdout. Everything else, including output from
libraries that print, goes to stderr (see `claim_stdout`).
"""

import json
import os
import sys

from src import contract


def claim_stdout():
    """
    Reserve the real stdout for protocol messages, and point file descriptor 1 at stderr.

    Redirecting the descriptor (not just `sys.stdout`) also catches output from C
    extensions and from libraries that hold their own reference to stdout.

    Returns:
        A binary file object that writes to the original stdout.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "wb", buffering=0)


def write_message(out, message: dict) -> None:
    out.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")
    out.flush()


def error_message(message_id, code: str, text: str) -> dict:
    return {"id": message_id, "error": {"code": code, "message": text}}


def handle_line(backend, line: bytes) -> dict | None:
    """Turn one input line into one response, or None for a blank line."""
    try:
        text = line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return error_message(None, contract.INVALID_JSON, "message is not valid UTF-8")
    if not text:
        return None

    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        return error_message(None, contract.INVALID_JSON, f"message is not valid JSON: {exc.msg}")

    command = message.get("command") if isinstance(message, dict) else None
    if command is None:
        return backend.tag_batch(message)
    if command == "info":
        return {"id": message.get("id"), "model": backend.model_info()}
    return error_message(message.get("id"), contract.UNKNOWN_COMMAND, f"unknown command: {command!r}")


def serve_stdio(load_backend, stdin=None, stdout=None) -> int:
    """
    Load the model, announce readiness, and answer one line at a time until stdin closes.

    Args:
        load_backend: a function returning a TaggingBackend. It is called after stdout is
            claimed, so anything it prints goes to stderr.
        stdin, stdout: binary streams; default to the process's stdin and a claimed stdout.

    Returns:
        The process exit status.
    """
    stdin = stdin if stdin is not None else sys.stdin.buffer
    out = stdout if stdout is not None else claim_stdout()

    try:
        backend = load_backend()
    except Exception as exc:
        print(f"Failed to load the model: {exc}", file=sys.stderr)
        write_message(out, {"ready": False, "error": {"code": "MODEL_LOAD_FAILED", "message": str(exc)}})
        return 1

    write_message(out, {"ready": True, "model": backend.model_info()})

    try:
        for line in stdin:
            try:
                response = handle_line(backend, line)
            except Exception as exc:  # never let one message end the session
                response = error_message(None, contract.INTERNAL_ERROR, str(exc))
            if response is not None:
                write_message(out, response)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    return 0
