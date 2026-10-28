"""Run a finite number of open-ended checkpoint sessions in separate processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from checkpoint_runtime import deterministic_subprocess_environment


CHECKPOINT_SECTION = "checkpoint"


def _next_config(config_path: Path) -> Path:
    config = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    checkpoint_config = config[CHECKPOINT_SECTION]
    resume_from = checkpoint_config.get("resume_from")
    if resume_from is None:
        completed_before = 0
    else:
        manifest = json.loads((Path(resume_from) / "manifest.json").read_text())
        completed_before = manifest["completed_trials"]
    completed_after = completed_before + config["agent"]["num_trials"]
    return (
        Path(checkpoint_config["generated_config_dir"])
        / f"resume-from-trial-{completed_after:08d}.yaml"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(__file__).with_name("sequential_config.yaml"),
        help="Initial or generated full configuration file.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        required=True,
        help="Finite number of separate training sessions to run.",
    )
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be positive")

    os.environ.setdefault("AUTOREC_ROOT", str(Path(__file__).resolve().parents[2]))
    config_path = args.config.resolve()
    worker = Path(__file__).with_name("main_factory.py")
    environment = deterministic_subprocess_environment()

    for session in range(1, args.sessions + 1):
        print(
            f"\nStarting training process {session}/{args.sessions} with {config_path}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, str(worker), "--config", str(config_path)],
            env=environment,
            check=True,
        )
        config_path = _next_config(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Training completed but did not create the expected config: {config_path}"
            )

    print(f"\nCompleted {args.sessions} session(s).")
    print(f"Continue later with --config {config_path}")


if __name__ == "__main__":
    main()
