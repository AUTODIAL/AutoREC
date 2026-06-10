"""Tests for config-driven factory helpers.

The factory imports the real environment and agent modules at import time. These tests load
factory.py with lightweight stand-ins so path-resolution behavior can be tested without
constructing TensorFlow/AutoEIS-backed objects.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_factory(monkeypatch):
    """Import factory.py after replacing heavy environment/agent imports with stubs."""
    environment_module = types.ModuleType("autorec.environment")

    class Env:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.dataset = kwargs.get("dataset")

    environment_module.EIS_ECM_Env = Env
    monkeypatch.setitem(sys.modules, "autorec.environment", environment_module)

    agent_module = types.ModuleType("autorec.agent")

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    agent_module.DDQN_ECM = Agent
    monkeypatch.setitem(sys.modules, "autorec.agent", agent_module)

    spec = importlib.util.spec_from_file_location(
        "factory_under_test", Path("src/autorec/factory.py")
    )
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    return factory


def patch_dataset_loader(monkeypatch, factory):
    """Replace dataset loading with a recorder so tests only assert resolved paths."""
    loaded_paths = []

    def fake_load_dataset(path):
        loaded_paths.append(Path(path))
        return {"loaded_from": Path(path)}

    monkeypatch.setattr(factory, "_load_dataset", fake_load_dataset)
    return loaded_paths


def test_environment_builder_resolves_example_config_dataset_path(monkeypatch):
    factory = load_factory(monkeypatch)
    loaded_paths = patch_dataset_loader(monkeypatch, factory)

    config_path = Path("configs_yaml/examples/environment_config.yaml")

    env = factory.environment_builder(config_path)

    # Example configs are the one exception to the normal current-directory convention:
    # dataset_path is resolved relative to the YAML file.
    assert loaded_paths == [
        config_path.resolve().parent / "../../data/examples/training_dataset.pkl"
    ]
    assert env.kwargs["dataset"]["loaded_from"] == loaded_paths[0]


def test_environment_and_agent_builder_resolves_nested_example_config(monkeypatch):
    factory = load_factory(monkeypatch)
    loaded_paths = patch_dataset_loader(monkeypatch, factory)

    config_path = Path("configs_yaml/examples/demo_environment_agent_config.yaml")

    training_env, eval_env, agent = factory.environment_and_agent_builder(config_path)

    assert loaded_paths == [
        config_path.resolve().parent / "../../data/examples/training_dataset.pkl"
    ]
    assert eval_env is None
    assert agent.kwargs["training_env"] is training_env
    assert agent.kwargs["eval_env"] is None


def test_non_example_yaml_keeps_dataset_path_relative_to_cwd(tmp_path, monkeypatch):
    factory = load_factory(monkeypatch)
    loaded_paths = patch_dataset_loader(monkeypatch, factory)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset_path: data/examples/training_dataset.pkl\n")

    factory.environment_builder(config_path)

    # Non-example config files should preserve user-provided relative paths.
    assert loaded_paths == [Path("data/examples/training_dataset.pkl")]


def test_dict_config_is_not_mutated(monkeypatch):
    factory = load_factory(monkeypatch)
    loaded_paths = patch_dataset_loader(monkeypatch, factory)
    config = {"dataset_path": "data/examples/training_dataset.pkl", "seed": 42}

    env = factory.environment_builder(config)

    assert loaded_paths == [Path("data/examples/training_dataset.pkl")]
    # Builders pop dataset_path internally, so they must copy caller-owned dicts first.
    assert config == {"dataset_path": "data/examples/training_dataset.pkl", "seed": 42}
    assert env.kwargs["seed"] == 42


def test_missing_dataset_path_raises_clear_error(monkeypatch):
    factory = load_factory(monkeypatch)

    with pytest.raises(KeyError, match="dataset_path"):
        factory.environment_builder({"seed": 42})


def test_load_dataset_rejects_missing_file(monkeypatch, tmp_path):
    factory = load_factory(monkeypatch)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        factory._load_dataset(tmp_path / "missing.pkl")
