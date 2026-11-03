"""Run all controlled checkpoint-continuation scenarios and compare their final state."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .checkpoint_runtime import REPOSITORY_ROOT, deterministic_subprocess_environment
from .train_session import _read_and_validate


def scenario_commands(sessions: int, *, with_evaluation: bool) -> list[list[str]]:
    """Return ordered fresh-process commands for the complete comparison workflow."""
    commands = [
        [sys.executable, "-m", "example.checkpoint.continuous.run"],
        [
            sys.executable,
            "-m",
            "example.checkpoint.bounded.run",
            "--sessions",
            str(sessions),
        ],
        [
            sys.executable,
            "-m",
            "example.checkpoint.complete_checkpoint.run",
            "--sessions",
            str(sessions),
        ],
        [
            sys.executable,
            "-m",
            "example.checkpoint.model_only_restart.run",
            "--sessions",
            str(sessions),
        ],
        [sys.executable, "-m", "example.checkpoint.comparison.compare_scenarios"],
    ]
    if with_evaluation:
        commands[-1].append("--with-evaluation")
    return commands


def _validate_trial_horizons(sessions: int) -> None:
    """Require every scenario to represent the same nominal number of trials."""
    directory = Path(__file__).resolve().parent
    config_paths = {
        "continuous": directory / "continuous/config.yaml",
        "bounded": directory / "bounded/config.yaml",
        "complete_checkpoint": directory / "complete_checkpoint/config.yaml",
        "model_only_restart": directory / "model_only_restart/config.yaml",
    }
    trials = {
        name: _read_and_validate(path)[0]["agent"]["num_trials"]
        for name, path in config_paths.items()
    }
    expected = trials["continuous"]
    mismatches = {
        name: session_trials * sessions
        for name, session_trials in trials.items()
        if name != "continuous" and session_trials * sessions != expected
    }
    if mismatches:
        raise ValueError(
            f"scenario trial horizons do not match continuous={expected}: {mismatches}"
        )


def main() -> None:
    """Execute every scenario sequentially and stop at the first failed command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--with-evaluation", action="store_true")
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be positive")
    _validate_trial_horizons(args.sessions)

    environment = deterministic_subprocess_environment()
    environment.setdefault("AUTOREC_ROOT", str(REPOSITORY_ROOT))
    for command in scenario_commands(args.sessions, with_evaluation=args.with_evaluation):
        print(f"\nRunning: {' '.join(command)}", flush=True)
        subprocess.run(
            command,
            env=environment,
            cwd=REPOSITORY_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
