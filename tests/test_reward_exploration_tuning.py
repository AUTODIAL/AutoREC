"""Focused checks for reward-exploration hyperparameter-tuning workflows."""

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import yaml

from example.reward_exploration.tuning import stage, tuning_utils


CASES = (
    "baseline",
    "reward_calculated_on_simplified_circuits",
    "reward_calculated_with_fim_on_simplified_circuits",
    "reward_calculated_with_continuous_fim_on_simplified_circuits",
)


REPOSITORY_ROOT = Path(__file__).parents[1]
REWARD_ROOT = REPOSITORY_ROOT / "example" / "reward_exploration"
DATA_ROOT = REPOSITORY_ROOT / "example" / "data"
REFERENCE_DATA = REPOSITORY_ROOT / "example" / "hyperparameter_tuning" / "data"


def test_fixed_run_uses_local_full_spectrum_pipeline_config():
    local = yaml.safe_load((REWARD_ROOT / "pipeline_config.yaml").read_text())
    standard = yaml.safe_load(
        (REPOSITORY_ROOT / "configs_yaml" / "examples" / "pipeline_config.yaml").read_text()
    )

    assert local["dataprep"]["path"] == "../../data/training_dataset.pkl"
    assert local["agent"]["num_trials"] == 10_000
    assert "progressive_spectrum" not in local
    for case_name in CASES:
        assert (REWARD_ROOT / case_name / local["dataprep"]["path"]).resolve().is_file()

    comparable = deepcopy(local)
    comparable["dataprep"]["path"] = standard["dataprep"]["path"]
    assert comparable == standard


def test_fixed_run_entrypoints_use_local_pipeline_config():
    for case_name in CASES:
        case_dir = REWARD_ROOT / case_name
        runner = (case_dir / "main_factory.py").read_text()
        submit_job = (case_dir / "submit_job.sh").read_text()

        assert 'DEFAULT_CONFIG = EXPLORATION_ROOT / "pipeline_config.yaml"' in runner
        assert "default=str(DEFAULT_CONFIG)" in runner
        assert "AUTOREC_ROOT" not in runner
        assert "../pipeline_config.yaml" in submit_job
        assert "~/modules/AutoREC/configs_yaml" not in submit_job


def test_reward_tuning_datasets_match_reference_splits():
    for name in ("training_dataset.pkl", "validation_dataset.pkl", "test_dataset.pkl"):
        assert (DATA_ROOT / name).read_bytes() == (REFERENCE_DATA / name).read_bytes()


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


def test_case_tuning_configurations_and_entrypoints_are_isolated():
    reward_settings = {
        CASES[0]: {},
        CASES[1]: {},
        CASES[2]: {"identifiability_thresh": 1e-6},
        CASES[3]: {
            "identifiability_thresh": 1e-6,
            "penalty_sharpness": 2.0,
            "penalty_scale": 2.0,
        },
    }
    for case_name, fixed_reward_settings in reward_settings.items():
        case_dir = REWARD_ROOT / case_name
        config = yaml.safe_load((case_dir / "tuning_base_config.yaml").read_text())
        assert config["dataprep"]["training"]["path"] == "../../data/training_dataset.pkl"
        assert config["dataprep"]["eval"]["path"] == "../../data/validation_dataset.pkl"
        assert config["agent"]["num_trials"] == 1_000
        assert config["agent"]["start_jump"] == 4_000
        assert config["agent"]["beta_anneal_trials"] == 5_950
        for environment_name in ("training", "eval"):
            environment = config["environment"][environment_name]
            assert {
                name: environment[name] for name in fixed_reward_settings
            } == fixed_reward_settings

        search_space = yaml.safe_load((case_dir / "tuning_search_space.yaml").read_text())
        assert fixed_reward_settings.keys().isdisjoint(search_space)
        combined = "\n".join(
            (case_dir / name).read_text()
            for name in ("tuning_base_config.yaml", "tuning_main_factory.py")
        )
        assert "test_dataset" not in combined
        assert "battery_dataset" not in combined
        assert "experimental_dataset" not in combined
        assert (case_dir / "submit_job.sh").is_file()
        assert (case_dir / "tuning_runner_job.sh").is_file()

        worker = (case_dir / "tuning_main_factory.py").read_text()
        runner = (case_dir / "tuning_runner.py").read_text()
        assert "from reward_exploration.tuning.stage import main" in worker
        assert "from reward_exploration.tuning.runner import main" in runner
        assert "EXPLORATION_ROOT = CASE_DIR.parent" in worker
        assert "EXPLORATION_ROOT = CASE_DIR.parent" in runner
        assert "from tuning." not in worker
        assert "from tuning." not in runner

        if case_name != "baseline":
            assert "def prepare_experimental_runtime" in worker
            assert "from environment import EIS_ECM_Env" in worker
            assert "pipeline_builder_function=build_experimental_pipeline" in worker
            assert "prepare_runtime_function=prepare_experimental_runtime" in worker


def test_reward_tuning_worker_imports_from_standalone_copy(tmp_path):
    copied_root = tmp_path / "reward_exploration"
    shutil.copytree(REWARD_ROOT, copied_root)
    case_dir = copied_root / "baseline"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tuning_main_factory; print(tuning_main_factory.main.__module__)",
        ],
        cwd=case_dir,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "reward_exploration.tuning.stage"


def test_reward_tuning_documentation_uses_local_requirements():
    readme = (REWARD_ROOT / "README.md").read_text()

    assert "--config ../pipeline_config.yaml" in readme
    assert "example/reward_exploration/tuning/requirements.txt" in readme
    assert "local `tuning/` package" in readme
