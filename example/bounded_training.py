"""Run a finite number of retained training sessions in one Python process.

The configuration's ``agent.num_trials`` is the default length of each session. This example
does not save checkpoints or support restarting the Python process between calls.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--trials-per-session", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be positive")
    if args.trials_per_session is not None and args.trials_per_session < 1:
        parser.error("--trials-per-session must be positive")

    # Match the existing examples' native-library setup before scientific imports.
    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

    from autorec.factory import pipeline_builder
    from autorec.utils import set_global_seed

    set_global_seed(args.seed, deterministic_ops=True)
    _, _, _, _, agent = pipeline_builder(args.config)
    if args.trials_per_session is not None:
        agent.num_trials = args.trials_per_session

    agent.train(additional_trials=0)
    for session in range(1, args.sessions + 1):
        agent.train()
        print(
            f"Completed session {session}/{args.sessions} at cumulative trial "
            f"{agent.completed_trials}."
        )


if __name__ == "__main__":
    main()
