"""Focused checks for reward-exploration hyperparameter-tuning workflows."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
REWARD_ROOT = REPOSITORY_ROOT / "example" / "reward_exploration"
REFERENCE_DATA = REPOSITORY_ROOT / "example" / "hyperparameter_tuning" / "data"


def test_reward_tuning_datasets_match_reference_splits():
    reward_data = REWARD_ROOT / "data"
    for name in ("training_dataset.pkl", "validation_dataset.pkl", "test_dataset.pkl"):
        assert (reward_data / name).read_bytes() == (REFERENCE_DATA / name).read_bytes()
