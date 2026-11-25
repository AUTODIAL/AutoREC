"""Focused checks for reward-exploration hyperparameter-tuning workflows."""

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import types

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
        assert "set -euo pipefail" in submit_job
        assert 'RUN_DIR=${SLURM_SUBMIT_DIR:-"$SCRIPT_DIR"}' in submit_job
        assert 'cd "$RUN_DIR"' in submit_job


def test_reward_tuning_datasets_match_reference_splits():
    for name in ("training_dataset.pkl", "validation_dataset.pkl", "test_dataset.pkl"):
        assert (DATA_ROOT / name).read_bytes() == (REFERENCE_DATA / name).read_bytes()


def test_shared_reward_tuning_results_and_slurm_script(tmp_path):
    results = pd.DataFrame({"found_solution": [True, False]})
    stage._write_evaluation_results(results, tmp_path)
    assert pd.read_pickle(tmp_path / "eval_results.pkl").equals(results)
    assert pd.read_csv(tmp_path / "eval_results.csv").equals(results)
    assert not list(tmp_path.glob(".*.tmp"))

    worker = tmp_path / "case" / "tuning_main_factory.py"
    worker.parent.mkdir()
    worker.touch()
    tuning_utils.write_job_script(
        tmp_path / "run",
        worker,
        tmp_path / "config.yaml",
        {"job_name": "reward_tuning"},
        script_args=("--checkpoint-every", 1_000),
    )
    script = (tmp_path / "run" / "submit.sh").read_text()
    assert "--checkpoint-every 1000" in script
    assert f"WORK_DIR={worker.parent}" in script
    assert "MODULE_SAVELIST_NAME='basic'" in script
    assert "PYTHON_ENV_PATH=$HOME/eis_env" in script
    assert "set -euo pipefail" in script
    assert "tar -czf models.tar.gz models && rm -rf models" in script
    assert "tar -czf fit_plots.tar.gz fit_plots && rm -rf fit_plots" in script
    assert "test_dataset" not in script


def test_shared_reward_worker_configures_runtime_before_full_budget_recovery(
    tmp_path, monkeypatch
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

    def plan_training_recovery(checkpoint_dir, target_trials):
        candidates = sorted(checkpoint_dir.glob("trial-*"))
        checkpoint = candidates[-1] if candidates else None
        completed = 0 if checkpoint is None else int(checkpoint.name.removeprefix("trial-"))
        calls.append(("plan_recovery", completed))
        return types.SimpleNamespace(
            checkpoint=checkpoint,
            completed_trials=completed,
            additional_trials=target_trials - completed,
        )

    runtime_module = types.ModuleType("autorec.runtime")
    runtime_module.configure_autorec_runtime = lambda **kwargs: None
    agent_module = types.ModuleType("autorec.agent")
    agent_module.DDQN_ECM = FakeDDQN
    checkpoint_module = types.ModuleType("autorec.checkpoint")
    checkpoint_module.plan_training_recovery = plan_training_recovery
    factory_module = types.ModuleType("autorec.factory")
    factory_module.pipeline_builder = lambda path: None
    utils_module = types.ModuleType("autorec.utils")
    utils_module.set_global_seed = lambda *args: calls.append(("seed", *args))
    monkeypatch.setitem(sys.modules, "autorec.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "autorec.agent", agent_module)
    monkeypatch.setitem(sys.modules, "autorec.checkpoint", checkpoint_module)
    monkeypatch.setitem(sys.modules, "autorec.factory", factory_module)
    monkeypatch.setitem(sys.modules, "autorec.utils", utils_module)
    monkeypatch.setattr(
        stage,
        "_configure_deterministic_tensorflow",
        lambda: calls.append(("configure_deterministic_runtime",)),
    )
    monkeypatch.setattr(stage, "_set_evaluation_seed", lambda seed: None)

    def prepare_runtime():
        calls.append(("prepare_case_runtime",))

    def build_pipeline(config_file):
        calls.append(("build_case_pipeline", config_file.name))
        return None, None, None, None, FakeAgent()

    fresh_dir = tmp_path / "fresh"
    fresh_config = tmp_path / "fresh.yaml"
    fresh_config.write_text(
        yaml.safe_dump(
            {"agent": {"num_trials": 10_000, "random_seed": 42, "save_dir": str(fresh_dir)}}
        )
    )
    stage.run_candidate(fresh_config, 42, 1_000, build_pipeline, prepare_runtime)

    resumed_dir = tmp_path / "resumed"
    resumed_config = tmp_path / "resumed.yaml"
    resumed_config.write_text(
        yaml.safe_dump(
            {"agent": {"num_trials": 10_000, "random_seed": 42, "save_dir": str(resumed_dir)}}
        )
    )
    (resumed_dir / "checkpoints" / "trial-00003000").mkdir(parents=True)
    stage.run_candidate(resumed_config, 42, 1_000, build_pipeline, prepare_runtime)

    assert calls == [
        ("configure_deterministic_runtime",),
        ("prepare_case_runtime",),
        ("plan_recovery", 0),
        ("seed", 42, True),
        ("build_case_pipeline", "fresh.yaml"),
        ("train", 10_000, 1_000),
        ("plan_recovery", 10_000),
        ("evaluate", 24, True, True),
        ("configure_deterministic_runtime",),
        ("prepare_case_runtime",),
        ("plan_recovery", 3_000),
        ("restore", "trial-00003000", 10_000),
        ("train", 7_000, 1_000),
        ("plan_recovery", 10_000),
        ("evaluate", 24, True, True),
    ]


def test_case_tuning_configurations_and_entrypoints_are_isolated():
    shared_runner = (REWARD_ROOT / "tuning" / "runner.py").read_text()
    assert "--checkpoint-every" in shared_runner
    assert "--max-trials" not in shared_runner
    assert "promotion" not in shared_runner

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
        assert config["agent"]["num_trials"] == 10_000
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
        assert not (case_dir / "tuning_runner_job.sh").exists()

        worker = (case_dir / "tuning_main_factory.py").read_text()
        runner = (case_dir / "tuning_runner.py").read_text()
        assert "from reward_exploration.tuning.stage import main" in worker
        assert "from reward_exploration.tuning.runner import main" in runner
        assert "EXPLORATION_ROOT = CASE_DIR.parent" in worker
        assert "EXPLORATION_ROOT = CASE_DIR.parent" in runner
        assert "from tuning." not in worker
        assert "from tuning." not in runner
        assert "full-budget" in runner

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
    assert "full-budget tuning workflow" in readme
    assert "candidate_scores.csv" in readme
    assert "tuning_runner_job.sh" not in readme
