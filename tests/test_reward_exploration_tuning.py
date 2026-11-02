"""Focused checks for reward-exploration hyperparameter-tuning workflows."""

from pathlib import Path

import pandas as pd

from example.reward_exploration.tuning import stage, tuning_utils


REPOSITORY_ROOT = Path(__file__).parents[1]
REWARD_ROOT = REPOSITORY_ROOT / "example" / "reward_exploration"
REFERENCE_DATA = REPOSITORY_ROOT / "example" / "hyperparameter_tuning" / "data"


def test_reward_tuning_datasets_match_reference_splits():
    reward_data = REWARD_ROOT / "data"
    for name in ("training_dataset.pkl", "validation_dataset.pkl", "test_dataset.pkl"):
        assert (reward_data / name).read_bytes() == (REFERENCE_DATA / name).read_bytes()


def test_shared_reward_tuning_targets_scoring_and_slurm_script(tmp_path):
    assert tuning_utils.cumulative_trial_targets(10_000, 1_000) == list(
        range(1_000, 10_001, 1_000)
    )
    assert tuning_utils.promotion_trial_targets(10_000, 2_000, 2) == [2_000, 4_000, 8_000]
    assert tuning_utils.select_promoted_config_ids({1: 0.1, 2: 0.9, 3: 0.5}, 2) == {
        2,
        3,
    }

    results = pd.DataFrame({"found_solution": [True, False]})
    stage._write_evaluation_results(results, tmp_path, 1_000)
    assert pd.read_pickle(tmp_path / "eval_results_trial-00001000.pkl").equals(results)

    worker = tmp_path / "case" / "tuning_main_factory.py"
    worker.parent.mkdir()
    worker.touch()
    tuning_utils.write_job_script(
        tmp_path / "run",
        worker,
        tmp_path / "config.yaml",
        {"job_name": "reward_tuning"},
        script_args=("--target-trials", 1_000),
        archive_models=False,
    )
    script = (tmp_path / "run" / "submit.sh").read_text()
    assert "--target-trials 1000" in script
    assert f"WORK_DIR={worker.parent}" in script
    assert "MODULE_SAVELIST_NAME='basic'" in script
    assert "PYTHON_ENV_PATH=$HOME/eis_env" in script
    assert "test_dataset" not in script
