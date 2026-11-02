"""Focused tests for the staged hyperparameter-tuning example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pandas as pd
import pytest
import yaml


EXAMPLE_DIR = Path(__file__).parents[1] / "example" / "hyperparameter_tuning"


def _load_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tuning_utils():
    return _load_module("hyperparameter_tuning_utils", "tuning_utils.py")


@pytest.fixture(scope="module")
def stage_worker():
    return _load_module("hyperparameter_stage_worker", "main_factory.py")


def test_fidelity_targets_and_promotions(tuning_utils):
    assert tuning_utils.cumulative_trial_targets(10_000, 1_000) == list(
        range(1_000, 10_001, 1_000)
    )
    assert tuning_utils.promotion_trial_targets(10_000, 2_000, 2) == [2_000, 4_000, 8_000]
    with pytest.raises(ValueError, match="divisible"):
        tuning_utils.cumulative_trial_targets(10_000, 3_000)
    with pytest.raises(ValueError, match="smaller"):
        tuning_utils.promotion_trial_targets(10_000, 10_000, 2)


def test_promotion_is_deterministic_and_keeps_the_best_half(tuning_utils):
    scores = {7: 0.5, 2: 0.9, 5: 0.5, 9: 0.1, 3: 0.8}
    assert tuning_utils.select_promoted_config_ids(scores, 2) == {2, 3, 5}
    with pytest.raises(ValueError, match="finite"):
        tuning_utils.select_promoted_config_ids({1: float("nan")}, 2)


def test_stage_worker_selects_latest_eligible_checkpoint(tmp_path, stage_worker):
    checkpoint_dir = tmp_path / "checkpoints"
    for trial in (1_000, 2_000, 4_000):
        (checkpoint_dir / f"trial-{trial:08d}").mkdir(parents=True)
    (checkpoint_dir / "trial-invalid").mkdir()

    assert stage_worker._latest_checkpoint(checkpoint_dir, 3_000).name == "trial-00002000"
    assert stage_worker._latest_checkpoint(checkpoint_dir, 500) is None


def test_evaluation_results_are_rung_specific_and_refresh_latest(tmp_path, stage_worker):
    results = pd.DataFrame({"found_solution": [True, False], "value": [1, 2]})
    stage_worker._write_evaluation_results(results, tmp_path, 1_000)

    rung_csv, rung_pickle = stage_worker._evaluation_paths(tmp_path, 1_000)
    assert pd.read_pickle(rung_pickle).equals(results)
    assert pd.read_csv(rung_csv).equals(results)
    assert pd.read_pickle(tmp_path / "eval_results.pkl").equals(results)
    assert pd.read_csv(tmp_path / "eval_results.csv").equals(results)
    assert not list(tmp_path.glob(".*.tmp"))


def test_worker_checkpoints_before_validation(tmp_path, stage_worker, monkeypatch):
    save_dir = tmp_path / "run"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {"agent": {"num_trials": 1_000, "random_seed": 42, "save_dir": str(save_dir)}}
        )
    )
    calls = []

    class FakeAgent:
        def __init__(self, completed_trials=0):
            self.completed_trials = completed_trials

        def train(self, *, additional_trials, checkpoint_dir):
            calls.append(("train", additional_trials))
            self.completed_trials += additional_trials
            (checkpoint_dir / f"trial-{self.completed_trials:08d}").mkdir(parents=True)

        def eval_all_eis(self, *, max_actions, verbose, use_eval_env):
            assert (save_dir / "checkpoints" / "trial-00001000").is_dir()
            calls.append(("evaluate", max_actions, verbose, use_eval_env))
            return pd.DataFrame({"found_solution": [False]})

    class FakeDDQN:
        @classmethod
        def from_checkpoint(cls, path, *, save_dir, num_trials):
            calls.append(("restore", path.name, num_trials))
            return FakeAgent(int(path.name.removeprefix("trial-")))

    runtime_module = types.ModuleType("autorec.runtime")
    runtime_module.configure_autorec_runtime = lambda **kwargs: None
    agent_module = types.ModuleType("autorec.agent")
    agent_module.DDQN_ECM = FakeDDQN
    factory_module = types.ModuleType("autorec.factory")
    factory_module.pipeline_builder = lambda path: (None, None, None, None, FakeAgent())
    utils_module = types.ModuleType("autorec.utils")
    utils_module.set_global_seed = lambda *args: calls.append(("seed", *args))
    monkeypatch.setitem(sys.modules, "autorec.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "autorec.agent", agent_module)
    monkeypatch.setitem(sys.modules, "autorec.factory", factory_module)
    monkeypatch.setitem(sys.modules, "autorec.utils", utils_module)
    monkeypatch.setattr(
        stage_worker,
        "_configure_checkpoint_tensorflow",
        lambda: calls.append(("configure_resume_runtime",)),
    )
    monkeypatch.setattr(stage_worker, "_set_evaluation_seed", lambda seed: None)

    stage_worker.run_stage(config_file, 1_000, 42)
    stage_worker.run_stage(config_file, 2_000, 42)

    assert calls == [
        ("seed", 42, True),
        ("train", 1_000),
        ("evaluate", 24, True, True),
        ("configure_resume_runtime",),
        ("restore", "trial-00001000", 1_000),
        ("train", 1_000),
        ("evaluate", 24, True, True),
    ]


def test_slurm_script_passes_stage_target_and_preserves_intermediate_models(
    tmp_path, tuning_utils, monkeypatch
):
    save_dir = tmp_path / "run"
    tuning_utils.write_job_script(
        save_dir,
        EXAMPLE_DIR / "main_factory.py",
        tmp_path / "config.yaml",
        {"job_name": "test"},
        script_args=("--target-trials", 1_000),
        archive_models=False,
    )
    script = (save_dir / "submit.sh").read_text()
    assert "--target-trials 1000" in script
    assert "CUDA_VISIBLE_DEVICES=-1" in script
    assert "TF_NUM_INTRAOP_THREADS=1" in script
    assert "tar -czf" not in script

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "sbatch":
            return types.SimpleNamespace(
                returncode=0, stdout="Submitted batch job 12345\n", stderr=""
            )
        return types.SimpleNamespace(returncode=0, stdout="12345|COMPLETED\n", stderr="")

    monkeypatch.setattr(tuning_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(tuning_utils.time, "sleep", lambda seconds: None)
    job_id = tuning_utils.write_job_script(
        save_dir,
        EXAMPLE_DIR / "main_factory.py",
        tmp_path / "config.yaml",
        {"job_name": "test"},
        submit=True,
    )
    final_stage_script = (save_dir / "submit.sh").read_text()
    tuning_utils.block_until_completed([job_id])

    assert job_id == "12345"
    assert "tar -czf models.tar.gz models && rm -rf models" in final_stage_script
    assert "tar -czf fit_plots.tar.gz fit_plots && rm -rf fit_plots" in final_stage_script
    assert commands[0][0] == "sbatch"
    assert commands[1][0] == "sacct"

    wrapper = (EXAMPLE_DIR / "runner_job.sh").read_text()
    assert "#SBATCH --time=7-00:00:00" in wrapper
    assert "#SBATCH --cpus-per-task=1" in wrapper
    assert "#SBATCH --mem-per-cpu=16G" in wrapper
    assert "#SBATCH --account=rrg-j3goals" in wrapper
    assert "scontrol show job" in wrapper
    assert "time python runner.py" in wrapper
    assert "NUM_INITIAL=5" in wrapper
    assert "MAX_TRIALS=10000" in wrapper
    assert "Slurm commands: direct from this compute node" in wrapper
    assert 'RUN_DIR=${SLURM_SUBMIT_DIR:-"$SCRIPT_DIR"}' in wrapper
    assert 'cd "$RUN_DIR"' in wrapper
    assert 'python runner.py "${RUNNER_ARGUMENTS[@]}"' in wrapper


def test_score_can_use_rung_specific_evaluation_file(tmp_path, tuning_utils):
    pd.DataFrame({"episodic_cumul_reward": [0.0, 12.0]}).to_pickle(
        tmp_path / "statistical_analysis.pkl"
    )
    rung_results = tmp_path / "eval_results_trial-00001000.pkl"
    pd.DataFrame({"found_solution": [False, False]}).to_pickle(rung_results)

    score, components = tuning_utils.compute_score(
        tmp_path,
        reward_sample_size=1,
        include_exact_rate=False,
        evaluation_results=rung_results,
    )

    assert score == pytest.approx(0.5)
    assert components == pytest.approx((1.0, 0.0, 0.0))


def test_base_config_uses_session_budget_and_separate_validation_data():
    config = yaml.safe_load((EXAMPLE_DIR / "base_config.yaml").read_text())
    assert config["agent"]["num_trials"] == 1_000
    assert config["agent"]["start_jump"] == 4_000
    assert config["agent"]["anneal_fraction"] is None
    assert config["agent"]["beta_anneal_trials"] == 5_950
    assert config["dataprep"]["training"]["path"].endswith("training_dataset.pkl")
    assert config["dataprep"]["eval"]["path"].endswith("validation_dataset.pkl")
    assert "test_dataset" not in (EXAMPLE_DIR / "base_config.yaml").read_text()
