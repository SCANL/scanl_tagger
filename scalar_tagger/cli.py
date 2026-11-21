"""
Command-line entry point, shared by the installed `scalar-tagger` command and the repository's
`main` script.

    scalar-tagger --version
    scalar-tagger serve --stdio          # JSON lines over stdin/stdout
    scalar-tagger serve --port 8080      # HTTP
    scalar-tagger train --features type context

`train`, `run`, and `serve` can also be given as `--mode train` and so on; `run` and `serve` are
the same. Heavy modules (torch, transformers) are imported only by the mode that needs them.
"""

import argparse
import json
import os
import sys

from scalar_tagger.lm_based_tagger.distilbert_preprocessing import AVAILABLE_FEATURES, DEFAULT_FEATURES
from scalar_tagger.version import __version__

MODES = ("train", "run", "serve")
DEFAULT_LM_MODEL_DIR = os.path.join("output", "best_model")
DEFAULT_CONFIG_PATH = "serve.json"

# The release model, pinned to an exact Hugging Face commit so a fresh install tags with the
# model the reported metrics describe. Bump RELEASE_REVISION (a MINOR version change) when a
# new release model is published; see RELEASE_MODEL.md.
RELEASE_MODEL = "sourceslicer/scalar_lm_v3"
RELEASE_REVISION = "d65d72568388965e3ce5a3bac8db7eafa475b457"


def get_version():
    """Return the current version of SCALAR."""
    return f"SCALAR tagger {__version__}"


def build_parser(discover_local: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scalar-tagger",
        description="SCALAR identifier tagger",
        epilog="Examples: `scalar-tagger serve --stdio`, `scalar-tagger serve --port 8080`, "
               "`python main --mode train --features type context --seed 7`.",
    )
    parser.add_argument("-v", "--version", action="version", version=get_version(), help="print tagger application version")
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="'train' a model, or 'run'/'serve' the tagger ('run' and 'serve' are the same)")
    # Kept so existing `--model_type lm_based` invocations still work; the tree-based model was removed in 3.0.0.
    parser.add_argument("--model_type", choices=["lm_based"], default="lm_based", help=argparse.SUPPRESS)

    # Model selection (run/serve)
    parser.add_argument("--local", action="store_true",
                        help="Serve the locally trained model in output/best_model (or the local model named in the config).")
    parser.add_argument("--model_dir", type=str, help="Local model directory to serve, or to save to when training.")
    revision_help = (f"Hugging Face revision (commit, branch, or tag) to load. Defaults to {RELEASE_REVISION[:12]} for {RELEASE_MODEL}."
                     if RELEASE_REVISION else "Hugging Face revision (commit, branch, or tag) to load.")
    parser.add_argument("--revision", type=str, help=revision_help)
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH if discover_local else None,
                        help="Path to the server config JSON (run/serve mode).")

    # Training options
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--features",
        nargs="+",
        choices=AVAILABLE_FEATURES,
        default=DEFAULT_FEATURES,
        metavar="FEATURE",
        help=f"Feature tokens to train with (train mode). Choices: {', '.join(AVAILABLE_FEATURES)}. "
             "Default: all of them except the opt-in lead_word.",
    )
    feature_group.add_argument(
        "--no-features",
        dest="features",
        action="store_const",
        const=[],
        help="Train without any feature tokens (train mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Training seed (train mode): weight init, batch order, augmentation. The holdout split "
             "is fixed, so runs with different seeds are comparable. Default: 209.",
    )
    synthetic_group = parser.add_mutually_exclusive_group()
    synthetic_group.add_argument(
        "--synthetic-data",
        metavar="PATH",
        help="Synthetic training file (train mode). Default: input/synthetic_pos_data_full.csv.",
    )
    synthetic_group.add_argument(
        "--no-synthetic-data",
        action="store_true",
        help="Train on the real identifiers only (train mode).",
    )
    parser.add_argument(
        "--marker-tokens",
        action="store_true",
        help="Add feature and position markers to the tokenizer as single tokens (train mode), "
             "instead of letting WordPiece split each into several subwords.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        metavar="K",
        help="Number of CV folds (train mode, default 5). Fewer folds are faster for screening "
             "experiments; compare only runs trained with the same fold count.",
    )
    parser.add_argument(
        "--run-name",
        metavar="NAME",
        help="Also save the report and predictions to output/reports/NAME_seed<N>.* (train mode), "
             "for comparing runs with `python -m scalar_tagger.lm_based_tagger.evaluation`.",
    )
    parser.add_argument(
        "--amplify",
        choices=["all", "real", "none"],
        help="Rows that get low-frequency upsampling and verb augmentation (train mode): "
             "'all' (default), 'real' for real identifiers only, or 'none'.",
    )

    # Run/serve options
    parser.add_argument("--device", type=str,
                        help="Device for inference: 'auto' (default; GPU if available), 'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--stdio", action="store_true",
                        help="Serve JSON lines over stdin/stdout instead of HTTP; all logging goes to stderr")
    parser.add_argument("--port", type=int, help="Port to bind server")
    parser.add_argument("--protocol", type=str, help="Protocol (http/https)")
    parser.add_argument("--word", type=str, help="Path to the word list (approved abbreviations)")
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
    resolved_config_model = resolve_path(config_model) if config_model and config_local else config_model

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


def train(args, parser, base_dir: str, resolve_path) -> int:
    try:
        from scalar_tagger.lm_based_tagger.train_model import train_lm
    except ImportError as exc:
        parser.error(f"Training needs extra packages ({exc.name}). Install them with: pip install 'scalar-tagger[train]'")

    # Pass only the options that were given, so train_lm's own defaults apply otherwise.
    train_args = {}
    if args.seed is not None:
        train_args["train_seed"] = args.seed
    if args.synthetic_data:
        train_args["synthetic_path"] = resolve_path(args.synthetic_data)
    if args.no_synthetic_data:
        train_args["use_synthetic_data"] = False
    if args.amplify is not None:
        train_args["amplify"] = args.amplify
    if args.run_name:
        train_args["run_name"] = args.run_name
    if args.marker_tokens:
        train_args["marker_tokens"] = True
    if args.model_dir:
        train_args["model_dir"] = resolve_path(args.model_dir)
    if args.folds is not None:
        if args.folds < 2:
            parser.error("--folds must be at least 2")
        train_args["folds"] = args.folds
    train_lm(base_dir, selected_features=args.features, **train_args)
    return 0


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
    args = parser.parse_args(normalize_argv(list(sys.argv[1:] if argv is None else argv)))

    def resolve_path(path: str | None) -> str | None:
        if not path:
            return None
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    if args.mode == "train":
        return train(args, parser, base_dir, resolve_path)

    config_path = resolve_path(args.config_path)
    runtime_config = load_runtime_config(config_path)
    model_path, use_local_model, revision = resolve_model(args, runtime_config, resolve_path, discover_local)

    if args.stdio:
        from scalar_tagger.stdio_server import serve_stdio

        def load_backend():
            from scalar_tagger.tagging_backend import TaggingBackend
            return TaggingBackend(
                model_path,
                local=use_local_model,
                revision=revision,
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
