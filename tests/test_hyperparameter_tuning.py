"""Focused tests for the full-budget hyperparameter-tuning example."""

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


def test_evaluation_results_are_published_atomically(tmp_path, stage_worker):
    results = pd.DataFrame({"found_solution": [True, False], "value": [1, 2]})
    stage_worker._write_evaluation_results(results, tmp_path)

    assert pd.read_pickle(tmp_path / "eval_results.pkl").equals(results)
    assert pd.read_csv(tmp_path / "eval_results.csv").equals(results)
    assert not list(tmp_path.glob(".*.tmp"))


def test_worker_trains_or_resumes_to_full_budget_before_validation(
    tmp_path, stage_worker, monkeypatch
):
    calls = []

    class FakeAgent:
        def __init__(self, completed_trials=0):
            self.completed_trials = completed_trials

        def train(self, *, additional_trials, checkpoint_dir, checkpoint_every):
            calls.append(("train", additional_trials, checkpoint_every))
            self.completed_trials += additional_trials
            (checkpoint_dir / f"trial-{self.completed_trials:08d}").mkdir(parents=True)

        def eval_all_eis(self, *, max_actions, verbose, use_eval_env):
            assert self.completed_trials == 10_000
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
    checkpoint_module = types.ModuleType("autorec.checkpoint")

    def plan_training_recovery(checkpoint_dir, target_trials):
        candidates = sorted(checkpoint_dir.glob("trial-*"))
        checkpoint = candidates[-1] if candidates else None
        completed = 0 if checkpoint is None else int(checkpoint.name.removeprefix("trial-"))
        return types.SimpleNamespace(
            checkpoint=checkpoint,
            completed_trials=completed,
            additional_trials=target_trials - completed,
        )

    checkpoint_module.plan_training_recovery = plan_training_recovery
    factory_module = types.ModuleType("autorec.factory")
    factory_module.pipeline_builder = lambda path: (None, None, None, None, FakeAgent())
    utils_module = types.ModuleType("autorec.utils")
    utils_module.set_global_seed = lambda *args: calls.append(("seed", *args))
    monkeypatch.setitem(sys.modules, "autorec.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "autorec.agent", agent_module)
    monkeypatch.setitem(sys.modules, "autorec.checkpoint", checkpoint_module)
    monkeypatch.setitem(sys.modules, "autorec.factory", factory_module)
    monkeypatch.setitem(sys.modules, "autorec.utils", utils_module)
    monkeypatch.setattr(
        stage_worker,
        "_configure_checkpoint_tensorflow",
        lambda: calls.append(("configure_resume_runtime",)),
    )
    monkeypatch.setattr(stage_worker, "_set_evaluation_seed", lambda seed: None)

    fresh_dir = tmp_path / "fresh"
    fresh_config = tmp_path / "fresh.yaml"
    fresh_config.write_text(
        yaml.safe_dump(
            {"agent": {"num_trials": 10_000, "random_seed": 42, "save_dir": str(fresh_dir)}}
        )
    )
    stage_worker.run_candidate(fresh_config, 42, 1_000)

    resumed_dir = tmp_path / "resumed"
    resumed_config = tmp_path / "resumed.yaml"
    resumed_config.write_text(
        yaml.safe_dump(
            {"agent": {"num_trials": 10_000, "random_seed": 42, "save_dir": str(resumed_dir)}}
        )
    )
    (resumed_dir / "checkpoints" / "trial-00003000").mkdir(parents=True)
    stage_worker.run_candidate(resumed_config, 42, 1_000)

    assert calls == [
        ("seed", 42, True),
        ("train", 10_000, 1_000),
        ("evaluate", 24, True, True),
        ("configure_resume_runtime",),
        ("restore", "trial-00003000", 10_000),
        ("train", 7_000, 1_000),
        ("evaluate", 24, True, True),
    ]


def test_runner_submits_only_incomplete_full_budget_candidates(
    tmp_path, tuning_utils, monkeypatch
):
    optuna = types.ModuleType("optuna")
    optuna_visualization = types.ModuleType("optuna.visualization")
    optuna_matplotlib = types.ModuleType("optuna.visualization.matplotlib")
    monkeypatch.setitem(sys.modules, "optuna", optuna)
    monkeypatch.setitem(sys.modules, "optuna.visualization", optuna_visualization)
    monkeypatch.setitem(sys.modules, "optuna.visualization.matplotlib", optuna_matplotlib)
    monkeypatch.setitem(sys.modules, "tuning_utils", tuning_utils)
    runner = _load_module("hyperparameter_tuning_runner", "runner.py")

    complete_dir = tmp_path / "complete"
    incomplete_dir = tmp_path / "incomplete"
    required_relative_paths = (
        Path("checkpoints/trial-00010000/manifest.json"),
        Path("statistical_analysis.pkl"),
        Path("eval_results.pkl"),
        Path("eval_results.csv"),
    )
    for relative_path in required_relative_paths:
        path = complete_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    submissions = []

    def write_job_script(save_dir, python_file, config_file, options, **kwargs):
        submissions.append((save_dir, options, kwargs))
        for relative_path in required_relative_paths:
            path = save_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return "12345"

    waited_for = []
    monkeypatch.setattr(runner, "write_job_script", write_job_script)
    monkeypatch.setattr(runner, "block_until_completed", waited_for.extend)
    candidates = [
        {"config_id": 1, "config_file": tmp_path / "1.yaml", "save_dir": complete_dir},
        {
            "config_id": 2,
            "config_file": tmp_path / "2.yaml",
            "save_dir": incomplete_dir,
        },
    ]

    runner._submit_candidates(
        candidates,
        target_trials=10_000,
        checkpoint_every=1_000,
        python_file=EXAMPLE_DIR / "main_factory.py",
        job_name_prefix="tuning",
        sbatch_options={"time": "72:00:00"},
    )

    assert len(submissions) == 1
    assert submissions[0][0] == incomplete_dir
    assert submissions[0][1]["job_name"] == "tuning_2"
    assert submissions[0][2]["script_args"] == ("--checkpoint-every", 1_000)
    assert submissions[0][2]["submit"] is True
    assert waited_for == ["12345"]


def test_slurm_script_runs_full_candidate_and_archives_outputs(
    tmp_path, tuning_utils, monkeypatch
):
    save_dir = tmp_path / "run"
    tuning_utils.write_job_script(
        save_dir,
        EXAMPLE_DIR / "main_factory.py",
        tmp_path / "config.yaml",
        {"job_name": "test"},
        script_args=("--checkpoint-every", 1_000),
    )
    script = (save_dir / "submit.sh").read_text()
    assert "--checkpoint-every 1000" in script
    assert "CUDA_VISIBLE_DEVICES=-1" in script
    assert "TF_NUM_INTRAOP_THREADS=1" in script
    assert f"WORK_DIR={EXAMPLE_DIR}" in script
    assert 'cd "$WORK_DIR"' in script
    assert "set -euo pipefail" in script
    assert script.index("set -euo pipefail") < script.index('time python "$SCRIPTS"')
    assert script.index('time python "$SCRIPTS"') < script.index("tar -czf models.tar.gz")
    assert "tar -czf models.tar.gz models && rm -rf models" in script
    assert "tar -czf fit_plots.tar.gz fit_plots && rm -rf fit_plots" in script
    assert "checkpoints" not in script[script.index("# ARCHIVE INTERMEDIATE OUTPUTS") :]

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "sbatch":
            return types.SimpleNamespace(
                returncode=0, stdout="Submitted batch job 12345\n", stderr=""
            )
        return types.SimpleNamespace(returncode=0, stdout="12345|COMPLETED|0:0\n", stderr="")

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


def test_slurm_monitor_recovers_from_temporary_accounting_outage(tuning_utils, monkeypatch):
    responses = iter(
        [
            types.SimpleNamespace(returncode=1, stdout="", stderr="Connection refused"),
            types.SimpleNamespace(returncode=0, stdout="12345|RUNNING\n", stderr=""),
            types.SimpleNamespace(returncode=0, stdout="12345|COMPLETED|0:0\n", stderr=""),
        ]
    )
    monkeypatch.setattr(
        tuning_utils.subprocess, "run", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(tuning_utils.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(tuning_utils.time, "monotonic", lambda: 0.0)

    assert tuning_utils.block_until_completed(
        ["12345"], poll_interval=1, accounting_timeout=10
    ) == {"12345": ("COMPLETED", "0:0")}


def test_score_combines_recent_training_reward_and_validation_success(tmp_path, tuning_utils):
    pd.DataFrame({"episodic_cumul_reward": [0.0, 12.0]}).to_pickle(
        tmp_path / "statistical_analysis.pkl"
    )
    pd.DataFrame({"found_solution": [False, False]}).to_pickle(tmp_path / "eval_results.pkl")

    score, components = tuning_utils.compute_score(
        tmp_path,
        reward_sample_size=1,
        include_exact_rate=False,
    )

    assert score == pytest.approx(0.5)
    assert components == pytest.approx((1.0, 0.0, 0.0))


def test_base_config_uses_full_budget_and_separate_validation_data():
    config = yaml.safe_load((EXAMPLE_DIR / "base_config.yaml").read_text())
    assert config["agent"]["num_trials"] == 10_000
    assert config["agent"]["start_jump"] == 4_000
    assert config["agent"]["anneal_fraction"] is None
    assert config["agent"]["beta_anneal_trials"] == 5_950
    assert config["dataprep"]["training"]["path"] == "./data/training_dataset.pkl"
    assert config["dataprep"]["eval"]["path"] == "./data/validation_dataset.pkl"
    assert "test_dataset" not in (EXAMPLE_DIR / "base_config.yaml").read_text()


def test_generated_config_preserves_working_directory_relative_data_paths(
    tmp_path, tuning_utils
):
    base_config = {
        "dataprep": {
            "training": {"path": "./data/training_dataset.pkl"},
            "eval": {"path": "./data/validation_dataset.pkl"},
        },
        "agent": {"num_trials": 1_000, "save_dir": "./"},
    }
    base_config_file = tmp_path / "base_config.yaml"
    base_config_file.write_text(yaml.safe_dump(base_config))

    handler = tuning_utils.ConfigurationHandler(
        base_config=base_config_file,
        configs_dir=tmp_path / "results" / "configs",
        search_space_bounds={},
        integer_variables=[],
    )
    _, generated_file, _ = handler.write_config(handler.base_config, tmp_path / "results")
    generated_config = yaml.safe_load(generated_file.read_text())
    assert generated_config["dataprep"]["training"]["path"] == ("./data/training_dataset.pkl")
    assert generated_config["dataprep"]["eval"]["path"] == ("./data/validation_dataset.pkl")
