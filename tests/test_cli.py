"""Tests for CLI parsing, preprocess forwarding, and YAML config override plumbing.

Heavy preprocess dependencies are stubbed so these tests can verify argument forwarding without
processing EIS data. Train and evaluate execution belong in broader integration tests.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import Mock

import pytest

from autorec import cli
from autorec.checkpoint import RecoveryPlan


_CHECKPOINT_SCRIPT = Path(__file__).parents[1] / "example" / "checkpoint_training.py"
_CHECKPOINT_SPEC = spec_from_file_location("checkpoint_training_example", _CHECKPOINT_SCRIPT)
checkpoint_training = module_from_spec(_CHECKPOINT_SPEC)
_CHECKPOINT_SPEC.loader.exec_module(checkpoint_training)


def test_train_config_and_resume_are_exclusive():
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(
            ["train", "--config", "config.yaml", "--resume", "checkpoint"]
        )


@pytest.mark.parametrize("resume", [False, True])
def test_train_checkpoint_options_and_resume_without_reseeding(monkeypatch, resume):
    agent = Mock()
    seed = Mock()
    builder = Mock(return_value=(None, None, None, None, agent))
    loader = Mock(return_value=agent)
    monkeypatch.setattr(cli, "_configure_runtime", lambda args: None)
    monkeypatch.setattr(cli, "_read_config_with_overrides", lambda *args: {})
    monkeypatch.setattr("autorec.utils.set_global_seed", seed)
    monkeypatch.setattr("autorec.factory.pipeline_builder", builder)
    monkeypatch.setattr("autorec.agent.DDQN_ECM.from_checkpoint", loader)
    # Avoid changing this test process's global TF determinism setting.
    monkeypatch.setattr("tensorflow.config.experimental.enable_op_determinism", lambda: None)
    source = ["--resume", "checkpoint"] if resume else ["--config", "config.yaml"]
    args = cli._build_parser().parse_args(
        [
            "train",
            *source,
            "--additional-trials",
            "100",
            "--checkpoint-dir",
            "checkpoints",
            "--checkpoint-every",
            "25",
        ]
    )
    assert cli._run_train(args) == 0
    agent.train.assert_called_once_with(
        additional_trials=100, checkpoint_dir=Path("checkpoints"), checkpoint_every=25
    )
    if resume:
        seed.assert_not_called()
        builder.assert_not_called()
        loader.assert_called_once_with(Path("checkpoint"), save_dir=None)
    else:
        seed.assert_called_once_with(42, deterministic_ops=True)
        builder.assert_called_once()
        loader.assert_not_called()


def test_resume_latest_runs_only_remaining_cumulative_trials(monkeypatch):
    agent = Mock()
    seed = Mock()
    builder = Mock(return_value=(None, None, None, None, agent))
    loader = Mock(return_value=agent)
    config = {"environment": {"seed": 7}, "agent": {"num_trials": 10}}
    fingerprint = cli._recovery_config_fingerprint(config, seed=42, deterministic_ops=True)
    monkeypatch.setattr(cli, "_configure_runtime", lambda args: None)
    monkeypatch.setattr(cli, "_read_config_with_overrides", lambda *args: config)
    monkeypatch.setattr("autorec.utils.set_global_seed", seed)
    monkeypatch.setattr("autorec.factory.pipeline_builder", builder)
    monkeypatch.setattr("autorec.agent.DDQN_ECM.from_checkpoint", loader)
    monkeypatch.setattr("tensorflow.config.experimental.enable_op_determinism", lambda: None)

    plans = [
        RecoveryPlan(
            checkpoint=None,
            completed_trials=0,
            additional_trials=10,
            config_fingerprint=None,
        ),
        RecoveryPlan(
            checkpoint=Path("checkpoints/trial-00000004"),
            completed_trials=4,
            additional_trials=6,
            config_fingerprint=fingerprint,
        ),
        RecoveryPlan(
            checkpoint=Path("checkpoints/trial-00000010"),
            completed_trials=10,
            additional_trials=0,
            config_fingerprint=fingerprint,
        ),
    ]
    planner = Mock(side_effect=plans)
    monkeypatch.setattr("autorec.checkpoint.plan_training_recovery", planner)
    args = cli._build_parser().parse_args(
        [
            "train",
            "--config",
            "config.yaml",
            "--checkpoint-dir",
            "checkpoints",
            "--checkpoint-every",
            "2",
            "--resume-latest",
        ]
    )

    assert cli._run_train(args) == 0
    assert agent._checkpoint_config_fingerprint == fingerprint
    agent.train.assert_called_once_with(
        additional_trials=10, checkpoint_dir=Path("checkpoints"), checkpoint_every=2
    )
    seed.assert_called_once_with(42, deterministic_ops=True)

    agent.reset_mock()
    assert cli._run_train(args) == 0
    loader.assert_called_once_with(Path("checkpoints/trial-00000004"), save_dir=None)
    agent.train.assert_called_once_with(
        additional_trials=6, checkpoint_dir=Path("checkpoints"), checkpoint_every=2
    )

    agent.reset_mock()
    assert cli._run_train(args) == 0
    agent.train.assert_not_called()
    assert planner.call_count == 3

    planner.side_effect = None
    planner.return_value = RecoveryPlan(
        checkpoint=Path("checkpoints/trial-00000004"),
        completed_trials=4,
        additional_trials=6,
        config_fingerprint="different",
    )
    with pytest.raises(ValueError, match="different training configuration"):
        cli._run_train(args)
    agent.train.assert_not_called()


def test_recovery_fingerprint_allows_target_and_output_changes_only():
    base = {
        "dataprep": {"path": "data.pkl"},
        "environment": {"seed": 42},
        "agent": {"num_trials": 100, "save_dir": "first", "gamma": 0.9},
    }
    changed_session = {
        **base,
        "agent": {**base["agent"], "num_trials": 200, "save_dir": "second"},
    }
    changed_training = {
        **base,
        "agent": {**base["agent"], "gamma": 0.8},
    }
    changed_progressive = {
        **base,
        "progressive_spectrum": {"enabled": True, "num_windows": 3},
    }

    fingerprint = cli._recovery_config_fingerprint(base, seed=42, deterministic_ops=True)
    assert (
        cli._recovery_config_fingerprint(changed_session, seed=42, deterministic_ops=True)
        == fingerprint
    )
    assert (
        cli._recovery_config_fingerprint(changed_training, seed=42, deterministic_ops=True)
        != fingerprint
    )
    assert (
        cli._recovery_config_fingerprint(changed_progressive, seed=42, deterministic_ops=True)
        != fingerprint
    )
    assert (
        cli._recovery_config_fingerprint(base, seed=43, deterministic_ops=True) != fingerprint
    )


def test_checkpoint_training_script_forwards_manual_and_automatic_modes(monkeypatch):
    forwarded = Mock(return_value=0)
    monkeypatch.setattr("autorec.cli.main", forwarded)

    assert (
        checkpoint_training.main(
            [
                "--config",
                "config.yaml",
                "--resume-latest",
                "--checkpoint-dir",
                "snapshots",
                "--checkpoint-every",
                "10",
            ]
        )
        == 0
    )
    forwarded.assert_called_once_with(
        [
            "train",
            "--config",
            "config.yaml",
            "--checkpoint-dir",
            "snapshots",
            "--checkpoint-every",
            "10",
            "--resume-latest",
        ]
    )

    forwarded.reset_mock()
    assert (
        checkpoint_training.main(["--resume", "trial-00000100", "--additional-trials", "100"])
        == 0
    )
    forwarded.assert_called_once_with(
        [
            "train",
            "--resume",
            "trial-00000100",
            "--checkpoint-dir",
            "checkpoints",
            "--checkpoint-every",
            "100",
            "--additional-trials",
            "100",
        ]
    )


def test_parse_indices_handles_spaces_and_empty_items():
    assert cli._parse_indices("0, 4,,10") == [0, 4, 10]


def test_parse_indices_returns_none_for_missing_value():
    assert cli._parse_indices(None) is None


def test_build_parser_accepts_train_command():
    parser = cli._build_parser()

    args = parser.parse_args(["train", "--config", "config.yaml"])

    assert args.command == "train"
    assert args.config == Path("config.yaml")
    assert args.seed == 42
    assert args.threads == 1


def test_build_parser_accepts_infer_alias():
    parser = cli._build_parser()

    args = parser.parse_args(["infer", "--config", "config.yaml", "--indices", "1,2"])

    assert args.command == "infer"
    assert args.indices == "1,2"
    assert args.use_training_env is False


def test_run_preprocess_forwards_linkk_and_frequency_options(tmp_path, monkeypatch):
    input_path = tmp_path / "raw"
    input_path.mkdir()
    received = {}

    class RecordingDataPrep:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def load(self):
            received["loaded"] = True

    monkeypatch.setattr(cli, "_configure_runtime", lambda args: None)
    monkeypatch.setattr("autorec.data_preparation.EISDataPrep", RecordingDataPrep)
    args = cli._build_parser().parse_args(
        [
            "preprocess",
            "--input",
            str(input_path),
            "--perform-linkk-validation",
            "--tol-linkk",
            "0.02",
            "--frequency-bounds",
            "none",
            "1000",
            "--frequency-npoints",
            "80",
        ]
    )

    result = cli._run_preprocess(args)

    assert result == 0
    assert received == {
        "path": input_path,
        "mode": "process",
        "evaluation": False,
        "perform_linKK_validation": True,
        "tol_linKK": 0.02,
        "frequency_bounds": [None, 1_000.0],
        "frequency_npoints": 80,
        "eis_features": ["ImZ", "phi", "mag", "nphi"],
        "loaded": True,
    }


def test_read_config_with_overrides_updates_agent_save_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"

    def fake_config_reader(path):
        """Verify the CLI passes through the requested config path before overriding."""
        assert path == config_path
        return {"environment": {"dataset": "data.pkl"}, "agent": {"num_trials": 1}}

    monkeypatch.setattr("autorec.factory.config_reader", fake_config_reader)

    config = cli._read_config_with_overrides(config_path, output_dir)

    assert config["agent"]["save_dir"] == output_dir
    assert config["agent"]["num_trials"] == 1


def test_read_config_with_overrides_creates_agent_section(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        "autorec.factory.config_reader",
        lambda path: {"environment": {"dataset": "data.pkl"}},
    )

    config = cli._read_config_with_overrides(config_path, output_dir)

    assert config["agent"]["save_dir"] == output_dir
