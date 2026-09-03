"""Tests for CLI parsing and YAML config override plumbing.

The tests stop before running actual preprocess/train/evaluate commands; those paths import
heavy dependencies and belong in integration tests with more extensive stubbing.
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


def test_read_config_with_overrides_updates_agent_save_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"

    def fake_config_reader(path):
        """Verify the CLI passes through the requested config path before overriding."""
        assert path == config_path
        return {"environment": {"dataset_path": "data.pkl"}, "agent": {"num_trials": 1}}

    monkeypatch.setattr("autorec.factory.config_reader", fake_config_reader)

    config = cli._read_config_with_overrides(config_path, output_dir)

    assert config["agent"]["save_dir"] == output_dir
    assert config["agent"]["num_trials"] == 1


def test_read_config_with_overrides_creates_agent_section(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        "autorec.factory.config_reader",
        lambda path: {"environment": {"dataset_path": "data.pkl"}},
    )

    config = cli._read_config_with_overrides(config_path, output_dir)

    assert config["agent"]["save_dir"] == output_dir
