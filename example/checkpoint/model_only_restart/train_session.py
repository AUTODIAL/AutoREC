"""Train one fresh agent session after optionally loading only an online model file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..checkpoint_runtime import configure_cache_environment
from ..train_session import _read_and_validate


def train_model_only_session(
    agent,
    previous_model: Path | None,
    final_snapshot: Path | None,
) -> Path:
    """Train one session without restoring any non-model checkpoint state."""
    if previous_model is not None:
        agent.load_model(previous_model)
    agent.train()
    model_path = Path(agent.model_name)
    if final_snapshot is not None:
        agent.save_checkpoint(final_snapshot)
    return model_path


def main() -> None:
    """Construct a fresh agent, optionally load a model, and run one session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--final-snapshot", type=Path)
    args = parser.parse_args()

    config, checkpoint_config = _read_and_validate(args.config.resolve())
    if checkpoint_config.get("resume_from") is not None:
        parser.error("model-only restart never accepts checkpoint.resume_from")
    config["agent"]["save_dir"] = str(args.output_dir.resolve())

    configure_cache_environment()
    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

    from autorec.factory import pipeline_builder
    from autorec.utils import set_global_seed

    # This is intentionally repeated in every process. Only model weights cross the boundary.
    set_global_seed(config["agent"]["random_seed"], deterministic_ops=True)
    _, _, _, _, agent = pipeline_builder(config)
    model_path = train_model_only_session(agent, args.model, args.final_snapshot)
    print(f"Saved model-only session output: {model_path}")


if __name__ == "__main__":
    main()
