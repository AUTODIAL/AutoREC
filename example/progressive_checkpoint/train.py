"""Run checkpointed AutoREC training with an example YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "progressive_config.yaml"


def _build_parser() -> argparse.ArgumentParser:
    """Build the example command-line parser without importing the scientific stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--additional-trials", type=int)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=EXAMPLE_DIR / "outputs/checkpoints"
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _autorec_command(args: argparse.Namespace) -> list[str]:
    """Translate example arguments into the supported ``autorec train`` arguments."""
    if args.resume is not None:
        if args.resume_latest:
            raise ValueError("--resume and --resume-latest cannot be combined.")
        source = ("--resume", args.resume)
    else:
        source = ("--config", args.config)

    command = [
        "train",
        source[0],
        str(source[1]),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    if args.resume_latest:
        command.append("--resume-latest")
    if args.additional_trials is not None:
        command.extend(("--additional-trials", str(args.additional_trials)))
    if args.output_dir is not None:
        command.extend(("--output-dir", str(args.output_dir)))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Parse example options and delegate training to AutoREC's public CLI."""
    args = _build_parser().parse_args(argv)
    from autorec.cli import main as autorec_main

    return autorec_main(_autorec_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
