"""Tests for the open-ended checkpoint example configuration workflow."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


_SCRIPT = Path(__file__).parents[1] / "example" / "checkpoint" / "main_factory.py"
_SPEC = spec_from_file_location("checkpoint_example_main", _SCRIPT)
example = module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)

_RUNNER_SCRIPT = _SCRIPT.with_name("run_sequential.py")
_RUNNER_SPEC = spec_from_file_location("checkpoint_example_runner", _RUNNER_SCRIPT)
runner = module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(runner)

_COMPARE_SCRIPT = _SCRIPT.with_name("compare_checkpoints.py")
_COMPARE_SPEC = spec_from_file_location("checkpoint_example_compare", _COMPARE_SCRIPT)
comparison = module_from_spec(_COMPARE_SPEC)
_COMPARE_SPEC.loader.exec_module(comparison)


def config(tmp_path, *, num_trials=100, resume_from=None):
    return {
        "environment": {
            "elements": ["+", "/", "R"],
            "initial_state": ["+RR"],
            "seed": 42,
            "chromosome_HEAD_len": 2,
            "cache_enabled": True,
            "cache_capacity": 10,
            "cache_type": "lru",
        },
        "agent": {
            "save_dir": str(tmp_path / "results"),
            "num_trials": num_trials,
            "random_seed": 42,
            "action_cap": None,
            "start_jump": 400,
            "beta_anneal_trials": 595,
        },
        "checkpoint": {
            "save_dir": str(tmp_path / "checkpoints"),
            "generated_config_dir": str(tmp_path / "configs"),
            "resume_from": resume_from,
        },
    }


def write_config(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_configuration_uses_num_trials_as_current_session_length(tmp_path):
    value = config(tmp_path, num_trials=50)

    loaded, checkpoint_config = example._read_and_validate(write_config(tmp_path, value))

    assert loaded["agent"]["num_trials"] == 50
    assert checkpoint_config["resume_from"] is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_configuration_rejects_invalid_session_length(tmp_path, value):
    with pytest.raises(ValueError, match="agent.num_trials"):
        example._read_and_validate(write_config(tmp_path, config(tmp_path, num_trials=value)))


def test_next_configuration_preserves_settings_and_uses_exact_checkpoint(tmp_path):
    value = config(tmp_path)

    destination = example._write_next_config(value, 100)
    generated = yaml.safe_load(destination.read_text())

    assert destination.name == "resume-from-trial-00000100.yaml"
    assert generated["agent"] == value["agent"]
    assert generated["environment"] == value["environment"]
    assert generated["checkpoint"]["resume_from"] == str(
        tmp_path / "checkpoints" / "trial-00000100"
    )


def test_resumed_configuration_allows_only_session_and_output_changes(tmp_path):
    value = config(tmp_path, num_trials=25, resume_from="checkpoint")
    environment = SimpleNamespace(
        ELEMENTS=["+", "/", "R"],
        _all_start_states=["+RR"],
        seed=42,
        chromosome_HEAD_len=2,
        chromosome_TAIL_len=3,
        cache_enabled=True,
        cache_capacity=10,
        cache_type="lru",
    )
    agent = SimpleNamespace(
        training_env=environment,
        random_seed=42,
        action_cap=8,
        start_jump=400,
        beta_anneal_trials=595,
    )

    example._validate_resumed_configuration(value, agent)
    value["agent"]["start_jump"] = 401

    with pytest.raises(ValueError, match="agent.start_jump"):
        example._validate_resumed_configuration(value, agent)


def test_runner_follows_generated_config_for_open_ended_progress(tmp_path):
    value = config(tmp_path, num_trials=25)
    initial = write_config(tmp_path, value)
    assert runner._next_config(initial).name == "resume-from-trial-00000025.yaml"

    checkpoint = tmp_path / "checkpoints" / "trial-00000025"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text('{"completed_trials": 25}')
    value["checkpoint"]["resume_from"] = str(checkpoint)
    resumed = tmp_path / "resumed.yaml"
    resumed.write_text(yaml.safe_dump(value, sort_keys=False))

    assert runner._next_config(resumed).name == "resume-from-trial-00000050.yaml"


def test_checkpoint_comparison_excludes_only_session_and_output_configuration():
    first = {
        "agent": {"num_trials": 1000, "start_jump": 400, "beta_anneal_trials": 595},
        "output": {"save_dir": "continuous"},
    }
    second = {
        "agent": {"num_trials": 100, "start_jump": 400, "beta_anneal_trials": 595},
        "output": {"save_dir": "sequential"},
    }

    assert comparison._comparable_config(first) == comparison._comparable_config(second)
    second["agent"]["beta_anneal_trials"] = 594
    assert comparison._comparable_config(first) != comparison._comparable_config(second)
