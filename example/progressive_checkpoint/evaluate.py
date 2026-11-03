"""Evaluate selected EIS rows with a progressive-spectrum AutoREC configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "progressive_config.yaml"


def _build_parser() -> argparse.ArgumentParser:
    """Build the example command-line parser without importing the scientific stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--indices", default="0")
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_DIR / "outputs/evaluation")
    parser.add_argument("--run-id", default="progressive_example")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _autorec_command(args: argparse.Namespace) -> list[str]:
    """Translate example arguments into the supported ``autorec evaluate`` arguments."""
    command = [
        "evaluate",
        "--config",
        str(args.config),
        "--model",
        str(args.model),
        "--indices",
        args.indices,
        "--max-actions",
        str(args.max_actions),
        "--output-dir",
        str(args.output_dir),
        "--run-id",
        args.run_id,
    ]
    if args.verbose:
        command.append("--verbose")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Parse example options and delegate evaluation to AutoREC's public CLI."""
    args = _build_parser().parse_args(argv)
    from autorec.cli import main as autorec_main

    return autorec_main(_autorec_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
