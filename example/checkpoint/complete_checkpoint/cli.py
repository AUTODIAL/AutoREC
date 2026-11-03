"""Run checkpointed training through AutoREC's supported training command.

Use ``--resume`` for an explicit immutable checkpoint, or combine ``--config`` with
``--resume-latest`` to recover automatically toward cumulative ``agent.num_trials``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--additional-trials", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.resume_latest and args.config is None:
        raise ValueError("--resume-latest requires --config.")
    if args.resume_latest and args.additional_trials is not None:
        raise ValueError("--resume-latest cannot be combined with --additional-trials.")

    command = [
        "train",
        "--config" if args.config is not None else "--resume",
        str(args.config if args.config is not None else args.resume),
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

    from autorec.cli import main as autorec_main

    return autorec_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
