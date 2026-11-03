"""Configuration contracts for the progressive reward-comparison examples.

These tests intentionally avoid training or circuit fitting. They verify that the new
examples differ from their full-spectrum counterparts only by one explicit, consistently
resolved progressive-spectrum configuration.
"""

from copy import deepcopy
from pathlib import Path

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
        assert _without_progressive(progressive) == standard


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
        assert _without_progressive(progressive) == full_spectrum


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
