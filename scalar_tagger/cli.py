"""
Command-line entry point, shared by the installed `scalar-tagger` command and the repository's
`main` script.

    scalar-tagger --version
    scalar-tagger serve --stdio          # JSON lines over stdin/stdout
    scalar-tagger serve --port 8080      # HTTP
    scalar-tagger train --features context

`train`, `run`, and `serve` can also be given as `--mode train` and so on; `run` and `serve` are
the same. Heavy modules (torch, transformers) are imported only by the mode that needs them.
"""

import argparse
import json
import os
import shlex
import sys

from scalar_tagger.lm_based_tagger.distilbert_preprocessing import AVAILABLE_FEATURES
from scalar_tagger.version import __version__

MODES = ("train", "run", "serve")
DEFAULT_LM_MODEL_DIR = os.path.join("output", "best_model")
DEFAULT_CONFIG_PATH = "serve.json"

# The release model, pinned to an exact Hugging Face commit so a fresh install tags with the
# model the reported metrics describe. Bump RELEASE_REVISION (a MINOR version change) when a
# new release model is published.
RELEASE_MODEL = "sourceslicer/scalar_lm_release_test"
RELEASE_REVISION = "41a3c953a6ec5612834a16b0da54d809531af2ee"


def get_version():
    """Return the current version of SCALAR."""
    return f"SCALAR tagger {__version__}"


def build_parser(discover_local: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scalar-tagger", description="SCALAR identifier tagger")
    parser.add_argument("-v", "--version", action="version", version=get_version(), help="print tagger application version")
    parser.add_argument("--local", action="store_true", help="Force use of a local model/tokenizer instead of the HuggingFace repo.")
    # Core run/train model arguments
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="'train' a model, or 'run'/'serve' the tagger ('run' and 'serve' are the same)")
    # Kept so existing `--model_type lm_based` invocations still work; the tree-based model was removed in 3.0.0.
    parser.add_argument("--model_type", choices=["lm_based"], default="lm_based", help=argparse.SUPPRESS)
    parser.add_argument("--input_path", type=str, help="Path to TSV file for training")
    parser.add_argument("--model_dir", type=str, help="Local model directory to save/load for lm_based runs")
    parser.add_argument("--revision", type=str,
                        help=f"Hugging Face revision (commit, branch, or tag) to load. Defaults to {RELEASE_REVISION[:12]} for {RELEASE_MODEL}")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH if discover_local else None,
                        help="Path to config JSON (used in run mode)")
    parser.add_argument(
        "--tagger-data",
        dest="use_tagger_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include input/tagger_data_new.tsv when training the lm_based model",
    )
    parser.add_argument(
        "--synthetic-data",
        dest="use_synthetic_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include input/synthetic_pos_data_full.csv when training the lm_based model",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=AVAILABLE_FEATURES,
        help="Feature tokens to include for lm_based training",
    )
    parser.add_argument(
        "--pattern-postprocessing",
        dest="pattern_postprocessing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override nominal-chunk repair postprocessing for lm_based train eval and inference",
    )

    # Run-specific options
    parser.add_argument("--device", type=str,
                        help="Device for inference: 'auto' (default; GPU if available), 'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--stdio", action="store_true",
                        help="Serve JSON lines over stdin/stdout instead of HTTP; all logging goes to stderr")
    parser.add_argument("--port", type=int, help="Port to bind server")
    parser.add_argument("--protocol", type=str, help="Protocol (http/https)")
    parser.add_argument("--word", type=str, help="Word used in config")
    parser.add_argument("--address", type=str, help="Server address")
    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    """Accept `scalar-tagger serve ...` as well as `scalar-tagger --mode serve ...`."""
    if argv and argv[0] in MODES:
        return ["--mode", argv[0], *argv[1:]]
    return argv


def load_runtime_config(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}

    with open(path) as handle:
        return json.load(handle)


def resolve_model(args, runtime_config: dict, resolve_path, discover_local: bool) -> tuple[str, bool, str | None]:
    """
    Pick the model to serve, as (model path or repo id, is local, Hugging Face revision).

    Order: --model_dir, then --local, then the config file, then output/best_model if
    `discover_local` and it exists, then the pinned release model.
    """
    config_model = runtime_config.get("model")
    config_local = bool(runtime_config.get("local", False))
    resolved_config_model = config_model
    if config_model and config_local:
        resolved_config_model = resolve_path(config_model)

    default_local_model_path = resolve_path(DEFAULT_LM_MODEL_DIR)
    revision = args.revision or runtime_config.get("revision")

    if args.model_dir:
        return resolve_path(args.model_dir), True, None
    if args.local:
        return (resolved_config_model if config_local and resolved_config_model else default_local_model_path), True, None
    if config_model:
        if not config_local and config_model == RELEASE_MODEL:
            revision = revision or RELEASE_REVISION
        return resolved_config_model, config_local, None if config_local else revision
    if discover_local and os.path.isdir(default_local_model_path):
        return default_local_model_path, True, None
    return RELEASE_MODEL, False, revision or RELEASE_REVISION


def main(argv: list[str] | None = None, base_dir: str | None = None, discover_local: bool = False) -> int:
    """
    Args:
        argv: arguments without the program name; defaults to sys.argv[1:].
        base_dir: directory that relative paths (config, model, training data) resolve
            against; defaults to the current directory.
        discover_local: also pick up serve.json and output/best_model from `base_dir` when no
            model is named. The repository's `main` script sets this; the installed command
            doesn't, so it serves the pinned release model unless told otherwise.
    """
    base_dir = base_dir or os.getcwd()
    parser = build_parser(discover_local)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalize_argv(raw_argv))

    def resolve_path(path: str | None) -> str | None:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)

    tagger_flag_explicit = any(flag in raw_argv for flag in {"--tagger-data", "--no-tagger-data"})
    synthetic_flag_explicit = any(flag in raw_argv for flag in {"--synthetic-data", "--no-synthetic-data"})

    if tagger_flag_explicit and not synthetic_flag_explicit:
        args.use_synthetic_data = False
    elif synthetic_flag_explicit and not tagger_flag_explicit:
        args.use_tagger_data = False

    if args.mode == "train":
        if not args.use_tagger_data and not args.use_synthetic_data:
            parser.error("At least one of --tagger-data or --synthetic-data must be enabled for training.")

        try:
            from scalar_tagger.lm_based_tagger.train_model import train_lm
        except ImportError as exc:
            parser.error(f"Training needs extra packages ({exc.name}). Install them with: pip install 'scalar-tagger[train]'")

        run_metadata = {
            "command": " ".join(
                shlex.quote(arg) for arg in getattr(sys, "orig_argv", sys.argv)
            ),
            "cli_options": dict(sorted(vars(args).items())),
        }

        train_lm(
            base_dir,
            use_tagger_data=args.use_tagger_data,
            use_synthetic_data=args.use_synthetic_data,
            selected_features=args.features,
            model_dir=resolve_path(args.model_dir),
            run_metadata=run_metadata,
        )
        return 0

    config_path = resolve_path(args.config_path)
    runtime_config = load_runtime_config(config_path)
    model_path, use_local_model, revision = resolve_model(args, runtime_config, resolve_path, discover_local)

    if args.stdio:
        from scalar_tagger.stdio_server import serve_stdio

        def load_backend():
            from scalar_tagger.tagging_backend import TaggingBackend

            postprocessing = args.pattern_postprocessing
            if postprocessing is None:
                postprocessing = runtime_config.get("pattern_postprocessing")
            return TaggingBackend(
                model_path,
                local=use_local_model,
                revision=revision,
                pattern_postprocessing=postprocessing,
                device=args.device or runtime_config.get("device"),
            )

        return serve_stdio(load_backend)

    from scalar_tagger.tag_identifier import start_server

    temp_config = {
        "script_dir": base_dir,
        "config_path": config_path,
        "model": model_path,
        "local": use_local_model,
        "revision": revision,
    }

    if args.pattern_postprocessing is not None:
        temp_config["pattern_postprocessing"] = args.pattern_postprocessing
    if args.port:
        temp_config["port"] = args.port
    if args.protocol:
        temp_config["protocol"] = args.protocol
    if args.address:
        temp_config["address"] = args.address
    if args.word:
        temp_config["words"] = resolve_path(args.word)
    if args.device:
        temp_config["device"] = args.device

    start_server(temp_config=temp_config)
    return 0


def run():
    """Console-script entry point for `scalar-tagger`."""
    sys.exit(main())
