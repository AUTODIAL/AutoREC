"""Run one bounded session, then resume it in another process with --resume.

Set deterministic CPU/thread/hash environment variables before starting Python, as
described in docs/checkpointing.md. The YAML num_trials is the full training horizon.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--additional-trials", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1)
    import tensorflow as tf
    from autorec.agent import DDQN_ECM
    from autorec.factory import config_reader, pipeline_builder
    from autorec.utils import set_global_seed

    tf.config.experimental.enable_op_determinism()
    if args.resume:
        agent = DDQN_ECM.from_checkpoint(args.resume, save_dir=args.output_dir)
    else:
        set_global_seed(42)
        config = config_reader(args.config)
        if args.output_dir is not None:
            config["agent"] = config.get("agent") or {}
            config["agent"]["save_dir"] = args.output_dir
        _, _, _, _, agent = pipeline_builder(config)
    agent.train(
        additional_trials=args.additional_trials,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Completed {agent.completed_trials}/{agent.num_trials} trials")


if __name__ == "__main__":
    main()
