"""Run repeated fresh-process sessions carrying only the previous online model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ..checkpoint_runtime import REPOSITORY_ROOT, deterministic_subprocess_environment
from ..train_session import _read_and_validate


def session_command(
    config: Path,
    output_dir: Path,
    previous_model: Path | None,
    final_snapshot: Path | None,
) -> list[str]:
    """Return the fresh-process command for one model-only session."""
    command = [
        sys.executable,
        "-m",
        "example.checkpoint.model_only_restart.train_session",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    ]
    if previous_model is not None:
        command.extend(("--model", str(previous_model)))
    if final_snapshot is not None:
        command.extend(("--final-snapshot", str(final_snapshot)))
    return command


def main() -> None:
    """Run the configured number of model-only sessions in fresh processes."""
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=directory / "config.yaml")
    parser.add_argument("--sessions", type=int, default=10)
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be positive")

    config_path = args.config.resolve()
    config, checkpoint_config = _read_and_validate(config_path)
    if checkpoint_config.get("resume_from") is not None:
        parser.error("model-only restart must start fresh")

    output_root = Path(config["agent"]["save_dir"])
    checkpoint_root = Path(checkpoint_config["save_dir"])
    final_snapshot = checkpoint_root / "final"
    environment = deterministic_subprocess_environment()
    environment.setdefault("AUTOREC_ROOT", str(REPOSITORY_ROOT))
    previous_model = None

    for session in range(1, args.sessions + 1):
        session_output = output_root / "sessions" / f"session-{session:04d}"
        snapshot = final_snapshot if session == args.sessions else None
        subprocess.run(
            session_command(config_path, session_output, previous_model, snapshot),
            env=environment,
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        previous_model = session_output / "dqn_model.keras"
        if not previous_model.is_file():
            raise FileNotFoundError(f"Session did not create expected model: {previous_model}")

    summary = {
        "scenario": "model_only_restart",
        "sessions": args.sessions,
        "trials_per_session": config["agent"]["num_trials"],
        "nominal_total_trials": args.sessions * config["agent"]["num_trials"],
        "final_model": str(previous_model),
        "final_snapshot": str(final_snapshot),
        "restored_state": ["online_model"],
        "not_restored": [
            "target_model",
            "optimizer",
            "replay",
            "training_histories",
            "schedules_and_counters",
            "environments_and_caches",
            "random_state",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(destination)
    print(f"Completed {args.sessions} model-only session(s).")
    print(f"Final inspection snapshot: {final_snapshot}")


if __name__ == "__main__":
    main()
