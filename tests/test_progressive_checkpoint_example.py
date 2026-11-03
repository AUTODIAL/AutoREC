"""Fast contract tests for the progressive-checkpoint example.

The example performs real fitting when executed by a user. These tests instead verify its
configuration semantics and public-CLI delegation without starting scientific runtimes.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import Mock

import yaml


EXAMPLE_DIR = Path(__file__).parents[1] / "example" / "progressive_checkpoint"


def _load_script(name: str):
    """Import one example script from its file path without making it a package."""
    spec = spec_from_file_location(
        f"progressive_checkpoint_{name}", EXAMPLE_DIR / f"{name}.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_configs_select_progressive_and_original_workflows():
    progressive = yaml.safe_load((EXAMPLE_DIR / "progressive_config.yaml").read_text())
    full_spectrum = yaml.safe_load((EXAMPLE_DIR / "full_spectrum_config.yaml").read_text())

    assert progressive["progressive_spectrum"]["enabled"] is True
    assert progressive["progressive_spectrum"]["num_windows"] == 4
    assert (
        progressive["progressive_spectrum"]["observation_encoder"]["type"]
        == "InterpolatingSpectrumEncoder"
    )
    assert "progressive_spectrum" not in full_spectrum
    assert progressive["environment"] == full_spectrum["environment"]
    assert progressive["agent"]["hidden_layers"] == full_spectrum["agent"]["hidden_layers"]


def test_training_example_delegates_automatic_and_explicit_resume(monkeypatch):
    training = _load_script("train")
    forwarded = Mock(return_value=0)
    monkeypatch.setattr("autorec.cli.main", forwarded)

    assert training.main(["--resume-latest", "--checkpoint-dir", "snapshots"]) == 0
    forwarded.assert_called_once_with(
        [
            "train",
            "--config",
            str(training.DEFAULT_CONFIG),
            "--checkpoint-dir",
            "snapshots",
            "--checkpoint-every",
            "1",
            "--resume-latest",
        ]
    )

    forwarded.reset_mock()
    assert training.main(["--resume", "trial-00000002", "--additional-trials", "1"]) == 0
    assert forwarded.call_args.args[0][:3] == ["train", "--resume", "trial-00000002"]
    assert forwarded.call_args.args[0][-2:] == ["--additional-trials", "1"]


def test_evaluation_example_delegates_to_public_cli(monkeypatch):
    evaluation = _load_script("evaluate")
    forwarded = Mock(return_value=0)
    monkeypatch.setattr("autorec.cli.main", forwarded)

    assert evaluation.main(["--model", "model.keras", "--indices", "0,2", "--verbose"]) == 0
    command = forwarded.call_args.args[0]
    assert command[:3] == ["evaluate", "--config", str(evaluation.DEFAULT_CONFIG)]
    assert command[3:7] == ["--model", "model.keras", "--indices", "0,2"]
    assert "--verbose" in command
