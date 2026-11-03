"""Workflow contracts for the progressive reward-comparison examples.

These tests intentionally avoid training or circuit fitting. They verify that the new
examples differ from their full-spectrum counterparts only by one explicit, consistently
resolved progressive-spectrum configuration and that their executable documentation points
to the corresponding progressive entry points.
"""

from copy import deepcopy
import importlib
from pathlib import Path

import pandas as pd
import yaml

from autorec.data_preparation import EISDataPrep
from autorec.factory.progressive_spectrum import progressive_spectrum_builder
from autorec.progressive_spectrum import (
    CarryBestStatePolicy,
    EqualPointWindowSelector,
    InterpolatingSpectrumEncoder,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "example" / "progressive_exploration"
REWARD_ROOT = REPOSITORY_ROOT / "example" / "reward_exploration"
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


def test_full_run_configs_only_add_three_window_progression():
    config_pairs = (
        ("demo_pipeline_config.yaml", "demo_pipeline_config.yaml"),
        ("pipeline_config.yaml", "pipeline_config.yaml"),
    )
    for progressive_name, standard_name in config_pairs:
        progressive = _read_yaml(EXAMPLE_ROOT / "configs" / progressive_name)
        standard = _read_yaml(REPOSITORY_ROOT / "configs_yaml" / "examples" / standard_name)
        assert progressive["dataprep"]["frequency_npoints"] == 79
        comparable = _without_progressive(progressive)
        comparable["dataprep"]["frequency_npoints"] = standard["dataprep"]["frequency_npoints"]
        assert comparable == standard


def test_every_progressive_exploration_config_uses_same_components():
    configs = [
        _read_yaml(EXAMPLE_ROOT / "configs" / "demo_pipeline_config.yaml"),
        _read_yaml(EXAMPLE_ROOT / "configs" / "pipeline_config.yaml"),
    ]
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
            "../../reward_exploration/data/training_dataset.pkl"
        )
        assert progressive["dataprep"]["eval"]["path"] == (
            "../../reward_exploration/data/validation_dataset.pkl"
        )
        assert progressive["dataprep"]["training"]["frequency_npoints"] == 79
        assert progressive["dataprep"]["eval"]["frequency_npoints"] == 79
        comparable = _without_progressive(progressive)
        comparable["dataprep"]["training"]["path"] = full_spectrum["dataprep"]["training"][
            "path"
        ]
        comparable["dataprep"]["eval"]["path"] = full_spectrum["dataprep"]["eval"]["path"]
        comparable["dataprep"]["training"]["frequency_npoints"] = full_spectrum["dataprep"][
            "training"
        ]["frequency_npoints"]
        comparable["dataprep"]["eval"]["frequency_npoints"] = full_spectrum["dataprep"][
            "eval"
        ]["frequency_npoints"]
        assert comparable == full_spectrum


def test_progressive_cases_reuse_canonical_reward_environments_and_data():
    for case in CUSTOM_CASES:
        progressive_module = importlib.import_module(
            f"example.progressive_exploration.{case}.environment"
        )
        canonical_module = importlib.import_module(
            f"example.reward_exploration.{case}.environment"
        )
        assert progressive_module.EIS_ECM_Env is canonical_module.EIS_ECM_Env

    assert not (EXAMPLE_ROOT / "data").exists()
    for name in (
        "training_dataset.pkl",
        "validation_dataset.pkl",
        "test_dataset.pkl",
        "battery_dataset.pkl",
        "experimental_dataset.pkl",
    ):
        assert (REWARD_ROOT / "data" / name).is_file()

    training_data = pd.read_pickle(REWARD_ROOT / "data" / "training_dataset.pkl")
    assert len(training_data.iloc[0]["freq"]) == 79
    assert len(training_data.iloc[0]["flatten_Z"]) == 79 * 4


def test_full_run_entrypoints_select_local_progressive_configs():
    long_config = (
        "~/modules/AutoREC/example/progressive_exploration/configs/pipeline_config.yaml"
    )
    for case in CASES:
        case_dir = EXAMPLE_ROOT / case
        runner = (case_dir / "main_factory.py").read_text()
        assert (
            'DEFAULT_CONFIG = EXAMPLE_ROOT / "configs" / "demo_pipeline_config.yaml"' in runner
        )
        assert "default=str(DEFAULT_CONFIG)" in runner
        assert "configs_yaml/examples/demo_pipeline_config.yaml" not in runner

        submit_job = (case_dir / "submit_job.sh").read_text()
        assert long_config in submit_job
        assert "#SBATCH --job-name=progressive_" in submit_job


def test_tuning_entrypoints_reuse_checkpointed_reward_workflow():
    for case in CASES:
        case_dir = EXAMPLE_ROOT / case
        worker = (case_dir / "tuning_main_factory.py").read_text()
        runner = (case_dir / "tuning_runner.py").read_text()
        job = (case_dir / "tuning_runner_job.sh").read_text()
        assert "from example.reward_exploration.tuning.stage import main" in worker
        assert "from example.reward_exploration.tuning.runner import main" in runner
        assert "REPOSITORY_ROOT = CASE_DIR.parents[2]" in worker
        assert "REPOSITORY_ROOT = CASE_DIR.parents[2]" in runner
        assert "#SBATCH --job-name='progressive_" in job
        assert _read_yaml(case_dir / "tuning_search_space.yaml") == _read_yaml(
            REWARD_ROOT / case / "tuning_search_space.yaml"
        )

    copied_tuning = EXAMPLE_ROOT / "tuning"
    assert not list(copied_tuning.glob("*.py"))
    assert not (copied_tuning / "requirements.txt").exists()


def test_progressive_documentation_uses_local_workflows_and_shared_requirements():
    parent_readme = (EXAMPLE_ROOT / "README.md").read_text()
    normalized_parent = " ".join(parent_readme.split())
    assert "example/progressive_exploration/baseline/main_factory.py" in parent_readme
    assert "example/progressive_exploration/configs/pipeline_config.yaml" in parent_readme
    assert "example/reward_exploration/tuning/requirements.txt" in parent_readme
    assert (
        "Circuit fitting, reward calculation, and FIM calculation use all original"
        in normalized_parent
    )
    assert "example/reward_exploration/baseline/main_factory.py" not in parent_readme
    assert "configs_yaml/examples/pipeline_config.yaml" not in parent_readme

    for case in CASES:
        case_readme = (EXAMPLE_ROOT / case / "README.md").read_text()
        assert f"example/progressive_exploration/{case}/main_factory.py" in case_readme
        assert (
            "example/reward_exploration/" not in case_readme.split("```bash", maxsplit=1)[-1]
        )

    example_index = (REPOSITORY_ROOT / "example" / "README.md").read_text()
    assert "[`progressive_exploration/`](progressive_exploration/)" in example_index


def test_progressive_config_resolves_registered_runtime_components():
    dataprep = object.__new__(EISDataPrep)
    dataprep.frequency_npoints = 79
    dataprep.eis_features = ["ImZ", "phi", "mag", "nphi"]
    config = _read_yaml(EXAMPLE_ROOT / "configs" / "demo_pipeline_config.yaml")

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
