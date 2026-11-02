"""Evaluation-mode helpers for the baseline calculation."""

from typing import Any


QUICK_SAMPLE_SIZE = 20
DEFAULT_RUN_MODE = "full"


def run_evaluation(
    agent: Any,
    run_mode: str,
    action_cap: int | None,
    dataset_size: int,
) -> tuple[Any, str]:
    """Evaluate a quick sample or every spectrum in the training dataset."""
    if run_mode == "quick":
        print(f"\nEvaluating {min(QUICK_SAMPLE_SIZE, dataset_size)} random spectra...")
        results = agent.eval_batch_eis(
            num_samples=QUICK_SAMPLE_SIZE,
            max_actions=action_cap,
            verbose=False,
            use_eval_env=False,
        )
        return results, "sample"

    if run_mode == "full":
        print(f"\nEvaluating all {dataset_size} spectra...")
        results = agent.eval_all_eis(
            max_actions=action_cap,
            verbose=False,
            use_eval_env=False,
        )
        return results, "all"

    raise ValueError(f"Invalid run mode: {run_mode}")
