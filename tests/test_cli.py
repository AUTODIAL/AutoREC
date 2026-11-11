"""Tests for CLI parsing, preprocess forwarding, and YAML config override plumbing.

Heavy preprocess dependencies are stubbed so these tests can verify argument forwarding without
processing EIS data. Train and evaluate execution belong in broader integration tests.
"""

from pathlib import Path

from autorec import cli


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
