"""Complete a configured training horizon through bounded calls in one Python process.

The configuration's agent.num_trials is the total horizon. This example does not save
checkpoints or support restarting the Python process between calls.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials-per-session", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trials_per_session < 1:
        parser.error("--trials-per-session must be positive")

    # Match the existing examples' native-library setup before scientific imports.
    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

    from autorec.factory import pipeline_builder
    from autorec.utils import set_global_seed

    set_global_seed(args.seed, deterministic_ops=True)
    _, _, _, _, agent = pipeline_builder(args.config)

    while agent.completed_trials < agent.num_trials:
        remaining = agent.num_trials - agent.completed_trials
        agent.train(additional_trials=min(args.trials_per_session, remaining))
        print(f"Completed {agent.completed_trials}/{agent.num_trials} trials")


if __name__ == "__main__":
    main()
