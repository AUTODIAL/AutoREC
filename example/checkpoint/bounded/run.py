"""Run retained training sessions in one process and save a final comparison snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..checkpoint_runtime import configure_cache_environment
from ..train_session import _read_and_validate


def run_retained_sessions(agent, sessions: int, checkpoint_directory: Path) -> Path:
    """Run bounded same-process sessions and save one final checkpoint for comparison.

    The checkpoint is not used to continue between sessions. It is written only after all
    sessions so the retained in-memory state can be compared with the uninterrupted and
    complete-checkpoint scenarios.
    """
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
        raise ValueError("sessions must be a positive integer")
    if agent.completed_trials != 0:
        raise ValueError("bounded comparison must start from an untrained agent")

    session_trials = agent.num_trials
    agent.train(additional_trials=0)
    for session in range(1, sessions + 1):
        agent.train()
        expected_trials = session * session_trials
        if agent.completed_trials != expected_trials:
            raise RuntimeError(
                f"Session {session} stopped at trial {agent.completed_trials}; "
                f"expected {expected_trials}."
            )
        print(
            f"Completed bounded session {session}/{sessions} at cumulative trial "
            f"{agent.completed_trials}."
        )

    destination = checkpoint_directory / f"trial-{agent.completed_trials:08d}"
    agent.save_checkpoint(destination)
    print(f"Saved final comparison checkpoint: {destination}")
    return destination


def main() -> None:
    """Build a fresh agent and execute the bounded comparison scenario."""
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=directory / "config.yaml")
    parser.add_argument("--sessions", type=int, default=10)
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be positive")

    config, checkpoint_config = _read_and_validate(args.config.resolve())
    if checkpoint_config.get("resume_from") is not None:
        parser.error("bounded training must start fresh; checkpoint.resume_from must be null")

    configure_cache_environment()
    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

    from autorec.factory import pipeline_builder
    from autorec.utils import set_global_seed

    set_global_seed(config["agent"]["random_seed"], deterministic_ops=True)
    _, _, _, _, agent = pipeline_builder(config)
    run_retained_sessions(agent, args.sessions, Path(checkpoint_config["save_dir"]))


if __name__ == "__main__":
    main()
