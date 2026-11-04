"""Workflow contracts for the progressive reward-comparison examples.

These tests intentionally avoid training or circuit fitting. They verify the explicit
progressive-spectrum configuration, shared exploration data paths, local reward modules,
and executable entry points.
"""

from copy import deepcopy
import importlib
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import yaml

from autorec.data_preparation import EISDataPrep
from autorec.environment import EIS_ECM_Env as BaseEISECMEnv
from autorec.factory.progressive_spectrum import progressive_spectrum_builder
from autorec.progressive_spectrum import (
    CarryBestStatePolicy,
    EqualPointWindowSelector,
    InterpolatingSpectrumEncoder,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "example" / "progressive_exploration"
REWARD_ROOT = REPOSITORY_ROOT / "example" / "reward_exploration"
DATA_ROOT = REPOSITORY_ROOT / "example" / "data"
CASES = (
    "baseline",
    "reward_calculated_on_simplified_circuits",
    "reward_calculated_with_fim_on_simplified_circuits",
    "reward_calculated_with_continuous_fim_on_simplified_circuits",
)
CUSTOM_CASES = CASES[1:]


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _without_progressive(config: dict) -> dict:
    config = deepcopy(config)
    config.pop("progressive_spectrum")
    return config


def test_full_run_configs_match_standard_settings_with_shared_exploration_data():
    progressive = _read_yaml(EXAMPLE_ROOT / "configs" / "pipeline_config.yaml")
    standard = _read_yaml(
        REPOSITORY_ROOT / "configs_yaml" / "examples" / "pipeline_config.yaml"
    )
    assert progressive["dataprep"]["frequency_npoints"] == 79
    assert progressive["dataprep"]["path"] == "../../data/training_dataset.pkl"
    assert progressive["agent"]["num_trials"] == 10_000
    comparable = _without_progressive(progressive)
    comparable["dataprep"]["path"] = standard["dataprep"]["path"]
    comparable["dataprep"]["frequency_npoints"] = standard["dataprep"]["frequency_npoints"]
    assert comparable == standard


def test_every_progressive_exploration_config_uses_same_components():
    configs = [_read_yaml(EXAMPLE_ROOT / "configs" / "pipeline_config.yaml")]
    configs.extend(
        _read_yaml(EXAMPLE_ROOT / case / "tuning_base_config.yaml") for case in CASES
    )
    progressive_configs = [config["progressive_spectrum"] for config in configs]
    assert all(config == progressive_configs[0] for config in progressive_configs)

    progressive = progressive_configs[0]
    assert progressive == {
        "enabled": True,
        "num_windows": 3,
        "window_selector": {
            "type": "EqualPointWindowSelector",
            "parameters": {},
        },
        "observation_encoder": {
            "type": "InterpolatingSpectrumEncoder",
            "parameters": {},
        },
        "spectrum_source": {
            "type": "PreparedSpectrumSource",
            "parameters": {},
        },
        "stage_transition_policy": {
            "type": "CarryBestStatePolicy",
            "parameters": {},
        },
    }


def test_tuning_configs_only_add_progressive_section():
    for case in CASES:
        progressive = _read_yaml(EXAMPLE_ROOT / case / "tuning_base_config.yaml")
        full_spectrum = _read_yaml(REWARD_ROOT / case / "tuning_base_config.yaml")
        assert progressive["dataprep"]["training"]["path"] == (
            "../../data/training_dataset.pkl"
        )
        assert progressive["dataprep"]["eval"]["path"] == ("../../data/validation_dataset.pkl")
        assert progressive["dataprep"]["training"]["frequency_npoints"] == 79
        assert progressive["dataprep"]["eval"]["frequency_npoints"] == 79
        comparable = _without_progressive(progressive)
        comparable["dataprep"]["training"]["frequency_npoints"] = full_spectrum["dataprep"][
            "training"
        ]["frequency_npoints"]
        comparable["dataprep"]["eval"]["frequency_npoints"] = full_spectrum["dataprep"][
            "eval"
        ]["frequency_npoints"]
        assert comparable == full_spectrum


def test_progressive_cases_bundle_matching_reward_environments_and_use_shared_data():
    for case in CUSTOM_CASES:
        progressive_module = importlib.import_module(
            f"example.progressive_exploration.{case}.environment"
        )
        assert issubclass(progressive_module.EIS_ECM_Env, BaseEISECMEnv)
        assert progressive_module.EIS_ECM_Env.__module__.startswith(
            "example.progressive_exploration."
        )
        assert (EXAMPLE_ROOT / case / "environment.py").read_text() == (
            REWARD_ROOT / case / "environment.py"
        ).read_text()

    for case in CUSTOM_CASES[1:]:
        assert (EXAMPLE_ROOT / case / "fim_utils.py").read_text() == (
            REWARD_ROOT / case / "fim_utils.py"
        ).read_text()

    for name in (
        "training_dataset.pkl",
        "validation_dataset.pkl",
        "test_dataset.pkl",
        "battery_dataset.pkl",
        "experimental_dataset.pkl",
    ):
        assert (DATA_ROOT / name).is_file()

    training_data = pd.read_pickle(DATA_ROOT / "training_dataset.pkl")
    assert len(training_data.iloc[0]["freq"]) == 79
    assert len(training_data.iloc[0]["flatten_Z"]) == 79 * 4


def test_local_reward_environment_imports_from_scratch_copy(tmp_path):
    case = "reward_calculated_on_simplified_circuits"
    copied_wrapper = tmp_path / "scratch" / "progressive_exploration" / case / "environment.py"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copyfile(EXAMPLE_ROOT / case / "environment.py", copied_wrapper)

    spec = importlib.util.spec_from_file_location(
        "scratch_progressive_environment", copied_wrapper
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert issubclass(module.EIS_ECM_Env, BaseEISECMEnv)
    assert "example.reward_exploration" not in copied_wrapper.read_text()


def test_full_run_entrypoints_select_local_progressive_configs():
    default_config = "../configs/pipeline_config.yaml"
    for case in CASES:
        case_dir = EXAMPLE_ROOT / case
        runner = (case_dir / "main_factory.py").read_text()
        assert 'DEFAULT_CONFIG = EXAMPLE_ROOT / "configs" / "pipeline_config.yaml"' in runner
        assert "default=str(DEFAULT_CONFIG)" in runner
        assert 'EVALUATION_DATA_DIR = EXAMPLE_ROOT.parent / "data"' in runner

        submit_job = (case_dir / "submit_job.sh").read_text()
        assert default_config in submit_job
        assert "#SBATCH --job-name=progressive_" in submit_job
        assert "set -euo pipefail" in submit_job
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f"export {variable}=1" in submit_job
        assert "export CUDA_VISIBLE_DEVICES=-1" in submit_job
        assert "export TF_CPP_MIN_LOG_LEVEL=3" in submit_job


def test_tuning_entrypoints_use_local_progressive_package():
    for case in CASES:
        case_dir = EXAMPLE_ROOT / case
        worker = (case_dir / "tuning_main_factory.py").read_text()
        runner = (case_dir / "tuning_runner.py").read_text()
        job = (case_dir / "tuning_runner_job.sh").read_text()
        assert "from progressive_exploration.tuning.stage import main" in worker
        assert "from progressive_exploration.tuning.runner import main" in runner
        assert "EXPLORATION_ROOT = CASE_DIR.parent" in worker
        assert "EXPLORATION_ROOT = CASE_DIR.parent" in runner
        assert "example.reward_exploration" not in worker
        assert "example.reward_exploration" not in runner
        assert "#SBATCH --job-name='progressive_" in job
        assert _read_yaml(case_dir / "tuning_search_space.yaml") == _read_yaml(
            REWARD_ROOT / case / "tuning_search_space.yaml"
        )

    for name in (
        "__init__.py",
        "requirements.txt",
        "runner.py",
        "stage.py",
        "tuning_utils.py",
    ):
        assert (EXAMPLE_ROOT / "tuning" / name).read_bytes() == (
            REWARD_ROOT / "tuning" / name
        ).read_bytes()


def test_progressive_tuning_worker_imports_from_standalone_copy(tmp_path):
    copied_root = tmp_path / "progressive_exploration"
    shutil.copytree(EXAMPLE_ROOT, copied_root)
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
    assert completed.stdout.strip() == "progressive_exploration.tuning.stage"


def test_progressive_documentation_uses_local_workflows_and_shared_requirements():
    parent_readme = (EXAMPLE_ROOT / "README.md").read_text()
    normalized_parent = " ".join(parent_readme.split())
    assert "cd example/progressive_exploration/baseline" in parent_readme
    assert "--config ../configs/pipeline_config.yaml" in parent_readme
    assert "example/reward_exploration/tuning/requirements.txt" in parent_readme
    assert (
        "Circuit fitting, reward calculation, and FIM calculation use all original"
        in normalized_parent
    )
    assert "example/reward_exploration/baseline/main_factory.py" not in parent_readme
    assert "configs_yaml/examples/pipeline_config.yaml" not in parent_readme

    for case in CASES:
        case_readme = (EXAMPLE_ROOT / case / "README.md").read_text()
        assert "python main_factory.py" in case_readme
        assert "../../reward_exploration/" not in case_readme

    example_index = (REPOSITORY_ROOT / "example" / "README.md").read_text()
    assert "[`progressive_exploration/`](progressive_exploration/)" in example_index


def test_progressive_config_resolves_registered_runtime_components():
    dataprep = object.__new__(EISDataPrep)
    dataprep.frequency_npoints = 79
    dataprep.eis_features = ["ImZ", "phi", "mag", "nphi"]
    config = _read_yaml(EXAMPLE_ROOT / "configs" / "pipeline_config.yaml")

    runtime = progressive_spectrum_builder(
        config["progressive_spectrum"],
        training_dataprep=dataprep,
    )

    assert runtime.enabled is True
    assert runtime.num_windows == 3
    assert isinstance(runtime.selector, EqualPointWindowSelector)
    assert isinstance(runtime.training_encoder, InterpolatingSpectrumEncoder)
    assert runtime.evaluation_encoder is runtime.training_encoder
    assert isinstance(runtime.transition_policy, CarryBestStatePolicy)
    assert runtime.spectrum_source.name == "prepared"
