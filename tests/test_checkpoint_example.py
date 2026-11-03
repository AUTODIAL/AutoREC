"""Tests for the open-ended checkpoint example configuration workflow."""

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


from example.checkpoint import checkpoint_runtime
from example.checkpoint import train_session as example
from example.checkpoint.comparison import compare_checkpoints as comparison
from example.checkpoint.complete_checkpoint import run as runner


_SCRIPT = Path(example.__file__)


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


def test_configuration_defines_repository_root_before_expansion(tmp_path, monkeypatch):
    value = config(tmp_path)
    value["agent"]["save_dir"] = "${AUTOREC_ROOT}/checkpoint-results"
    monkeypatch.delenv("AUTOREC_ROOT", raising=False)

    loaded, _ = example._read_and_validate(write_config(tmp_path, value))

    assert loaded["agent"]["save_dir"] == str(_SCRIPT.parents[2] / "checkpoint-results")


def test_checkpoint_runtime_matches_tensorflow_and_subprocess_configuration():
    calls = []
    tensorflow = SimpleNamespace(
        config=SimpleNamespace(
            experimental=SimpleNamespace(
                enable_op_determinism=lambda: calls.append("deterministic")
            ),
            threading=SimpleNamespace(
                set_inter_op_parallelism_threads=lambda value: calls.append(("inter", value)),
                set_intra_op_parallelism_threads=lambda value: calls.append(("intra", value)),
            ),
        )
    )

    checkpoint_runtime.configure_tensorflow_for_checkpoint(tensorflow)

    assert calls == ["deterministic", ("inter", 1), ("intra", 1)]
    environment = checkpoint_runtime.deterministic_subprocess_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["TF_NUM_INTRAOP_THREADS"] == "1"
    assert environment["TF_NUM_INTEROP_THREADS"] == "1"


def test_resume_normalizes_concise_optimizer_configuration_without_mutating_input(
    tmp_path,
):
    value = config(tmp_path)
    value["agent"]["optimizer"] = {"type": "Adam", "learning_rate": 0.0005}
    complete = {
        "type": "Adam",
        "learning_rate": 0.0005000000237487257,
        "beta_1": 0.9,
    }

    normalized = example._normalize_resumed_configuration(
        value,
        lambda optimizer_config: ("optimizer", optimizer_config),
        lambda optimizer: complete,
    )

    assert normalized["agent"]["optimizer"] == complete
    assert value["agent"]["optimizer"] == {"type": "Adam", "learning_rate": 0.0005}


def test_configuration_rejects_invalid_session_lengths(tmp_path):
    for index, value in enumerate((0, -1, True, 1.5, None)):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        with pytest.raises(ValueError, match="agent.num_trials"):
            example._read_and_validate(
                write_config(case_path, config(case_path, num_trials=value))
            )


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


def test_checkpoint_comparison_excludes_only_session_and_output_configuration(tmp_path):
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

    output = StringIO()
    comparison._print_verdict(
        True, "test comparison", ("values are identical",), stream=output
    )
    assert "VERDICT: PASS" in output.getvalue()
    assert "Comparison: test comparison" in output.getvalue()

    report = comparison._finalize_report(
        "test comparison",
        Path("continuous"),
        Path("sequential"),
        [{"name": "state", "passed": True, "detail": "Exact match."}],
    )
    destination = tmp_path / "reports" / "comparison.json"
    comparison._write_report(report, destination)
    assert yaml.safe_load(destination.read_text()) == report
