"""Tests for the config-driven AutoREC factory helpers.

The builders are imported from their public package and exercised with lightweight stand-ins
so these tests can verify configuration and pipeline wiring without constructing real
TensorFlow/AutoEIS-backed objects.
"""

import importlib
import types

import pandas as pd
import pytest


factory_package = importlib.import_module("autorec.factory")
agent_module = importlib.import_module("autorec.agent")
agent_factory = importlib.import_module("autorec.factory.agent")
dataprep_factory = importlib.import_module("autorec.factory.dataprep")
environment_factory = importlib.import_module("autorec.factory.environment")
pipeline_factory = importlib.import_module("autorec.factory.pipeline")


@pytest.fixture
def factory(monkeypatch):
    """Replace the split builders' heavyweight classes with recording stubs."""

    class DataPrep:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            return pd.DataFrame({"source": [self.kwargs["path"]]})

    class Env:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.dataset = kwargs.get("dataset")

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(dataprep_factory, "EISDataPrep", DataPrep)
    monkeypatch.setattr(environment_factory, "EIS_ECM_Env", Env)
    monkeypatch.setattr(agent_module, "DDQN_ECM", Agent)

    return types.SimpleNamespace(
        agent_builder=factory_package.agent_builder,
        config_reader=factory_package.config_reader,
        dataprep_builder=factory_package.dataprep_builder,
        environment_builder=factory_package.environment_builder,
        pipeline_builder=factory_package.pipeline_builder,
        _validate_pipeline_config=pipeline_factory._validate_pipeline_config,
    )


def test_factory_package_exports_builders_from_split_modules():
    assert factory_package.agent_builder is agent_factory.agent_builder
    assert factory_package.dataprep_builder is dataprep_factory.dataprep_builder
    assert factory_package.environment_builder is environment_factory.environment_builder
    assert factory_package.pipeline_builder is pipeline_factory.pipeline_builder


def test_config_reader_loads_yaml_and_expands_environment_variables(
    factory, monkeypatch, tmp_path
):
    monkeypatch.setenv("AUTOREC_TEST_DATA", "/tmp/example.pkl")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: ${AUTOREC_TEST_DATA}\nseed: 42\n")

    config = factory.config_reader(config_path)

    assert config == {"dataset": "/tmp/example.pkl", "seed": 42}


def test_config_reader_rejects_missing_file(factory, tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        factory.config_reader(tmp_path / "missing.yaml")


def test_config_reader_rejects_non_yaml_file(factory, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    with pytest.raises(ValueError, match="must be a YAML file"):
        factory.config_reader(config_path)


@pytest.mark.parametrize("contents", ["", "- first\n- second\n"])
def test_config_reader_rejects_yaml_without_top_level_mapping(factory, tmp_path, contents):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(contents)

    with pytest.raises(ValueError, match="must contain a dictionary"):
        factory.config_reader(config_path)


def test_dataprep_builder_loads_data_exports_output_and_preserves_config(factory, tmp_path):
    output_path = tmp_path / "processed.pkl"
    config = {
        "path": "raw",
        "mode": "process",
        "frequency_bounds": [10.0, 1_000.0],
        "frequency_npoints": 3,
        "output": output_path,
    }

    prep, dataset = factory.dataprep_builder(config)

    assert prep.kwargs == {
        "path": "raw",
        "mode": "process",
        "frequency_bounds": [10.0, 1_000.0],
        "frequency_npoints": 3,
    }
    pd.testing.assert_frame_equal(dataset, pd.DataFrame({"source": ["raw"]}))
    pd.testing.assert_frame_equal(pd.read_pickle(output_path), dataset)
    assert config == {
        "path": "raw",
        "mode": "process",
        "frequency_bounds": [10.0, 1_000.0],
        "frequency_npoints": 3,
        "output": output_path,
    }


def test_environment_builder_accepts_dataframe_and_preserves_config(factory):
    dataset = pd.DataFrame({"sample": [1]})
    config = {"dataset": dataset, "elements": ["+", "R", "P"], "seed": 42}

    env = factory.environment_builder(config)

    assert env.dataset is dataset
    assert env.kwargs["elements"] == ["+", "R", "P"]
    assert env.kwargs["seed"] == 42
    assert config["dataset"] is dataset


def test_environment_builder_loads_pickle_dataset_and_preserves_config(factory, tmp_path):
    dataset = pd.DataFrame({"sample": [1, 2]})
    dataset_path = tmp_path / "dataset.pkl"
    dataset.to_pickle(dataset_path)
    config = {"dataset": dataset_path, "seed": 7}

    env = factory.environment_builder(config)

    pd.testing.assert_frame_equal(env.dataset, dataset)
    assert env.kwargs["seed"] == 7
    assert config == {"dataset": dataset_path, "seed": 7}


def test_environment_builder_requires_dataset(factory):
    with pytest.raises(KeyError, match="include a 'dataset' key"):
        factory.environment_builder({"seed": 42})


def test_agent_builder_constructs_agent_without_mutating_config(factory):
    training_env = object()
    config = {"training_env": training_env, "num_trials": 5}

    agent = factory.agent_builder(config)

    assert agent.kwargs == config
    assert config == {"training_env": training_env, "num_trials": 5}


@pytest.mark.parametrize("builder_name", ["dataprep_builder", "environment_builder"])
def test_standalone_builders_reject_unsupported_config_types(factory, builder_name):
    with pytest.raises(TypeError, match="must be a str, Path, or dict"):
        getattr(factory, builder_name)(["not", "a", "config"])


def test_pipeline_builder_wires_single_dataprep_environment_and_agent(factory):
    config = {
        "dataprep": {"path": "training.pkl", "mode": "load"},
        "environment": {"seed": 42},
        "agent": {"num_trials": 5},
    }

    dataprep, dataprep_eval, env, env_eval, agent = factory.pipeline_builder(config)

    assert dataprep.kwargs == {"path": "training.pkl", "mode": "load"}
    assert dataprep_eval is dataprep
    assert env.dataset.equals(pd.DataFrame({"source": ["training.pkl"]}))
    assert env.kwargs["seed"] == 42
    assert env_eval is None
    assert agent.kwargs["training_env"] is env
    assert agent.kwargs["eval_env"] is None
    assert agent.kwargs["num_trials"] == 5


def test_pipeline_builder_wires_separate_training_and_eval_sections(factory):
    stale_dataset = pd.DataFrame({"stale": [True]})
    stale_env = object()
    config = {
        "dataprep": {
            "training": {"path": "training.pkl"},
            "eval": {"path": "eval.pkl"},
        },
        "environment": {
            "training": {"dataset": stale_dataset, "seed": 1},
            "eval": {"dataset": stale_dataset, "seed": 2},
        },
        "agent": {"training_env": stale_env, "eval_env": stale_env},
    }

    dataprep, dataprep_eval, env, env_eval, agent = factory.pipeline_builder(config)

    assert dataprep.kwargs == {"path": "training.pkl"}
    assert dataprep_eval.kwargs == {"path": "eval.pkl"}
    assert env.dataset.equals(pd.DataFrame({"source": ["training.pkl"]}))
    assert env_eval.dataset.equals(pd.DataFrame({"source": ["eval.pkl"]}))
    assert env.kwargs["seed"] == 1
    assert env_eval.kwargs["seed"] == 2
    assert agent.kwargs["training_env"] is env
    assert agent.kwargs["eval_env"] is env_eval


def test_pipeline_builder_uses_environment_dataset_without_dataprep(factory):
    dataset = pd.DataFrame({"sample": [1]})
    config = {
        "environment": {"dataset": dataset, "seed": 42},
        "agent": {"num_trials": 5},
    }

    dataprep, dataprep_eval, env, env_eval, agent = factory.pipeline_builder(config)

    assert dataprep is None
    assert dataprep_eval is None
    pd.testing.assert_frame_equal(env.dataset, dataset)
    assert config["environment"]["dataset"] is dataset
    assert env_eval is None
    assert agent.kwargs["training_env"] is env
    assert agent.kwargs["eval_env"] is None


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ([], TypeError, "must be a dictionary"),
        ({"agent": {}}, KeyError, "include an 'environment' section"),
        (
            {"dataprep": {"eval": {}}, "environment": {}, "agent": {}},
            KeyError,
            "must also include a 'training' subsection",
        ),
        (
            {"environment": {"eval": {"dataset": object()}}, "agent": {}},
            KeyError,
            "must also include a 'training' subsection",
        ),
        (
            {"environment": {"seed": 42}, "agent": {}},
            KeyError,
            "must include a 'dataset' key",
        ),
        (
            {"environment": {"dataset": object()}},
            KeyError,
            "include an 'agent' section",
        ),
    ],
)
def test_validate_pipeline_config_rejects_invalid_structure(
    factory, config, error_type, message
):
    with pytest.raises(error_type, match=message):
        factory._validate_pipeline_config(config)


def test_yaml_pipeline_forwards_generic_hidden_layer_configs(factory, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """dataprep:
  path: training.pkl
  mode: load
environment:
  seed: 42
agent:
  hidden_layers:
    - type: Dense
      units: 24
      activation: relu
    - type: Dropout
      rate: 0.2
"""
    )

    _, _, _, _, agent = factory.pipeline_builder(config_path)

    assert agent.kwargs["hidden_layers"] == [
        {"type": "Dense", "units": 24, "activation": "relu"},
        {"type": "Dropout", "rate": 0.2},
    ]


def test_pipeline_builder_does_not_mutate_generic_agent_config(factory):
    dataset = pd.DataFrame({"sample": [1]})
    hidden_layers = [{"type": "Dropout", "rate": 0.2}]
    config = {
        "environment": {"dataset": dataset},
        "agent": {"hidden_layers": hidden_layers},
    }

    _, _, _, _, agent = factory.pipeline_builder(config)

    assert agent.kwargs["hidden_layers"] == hidden_layers
    assert config["agent"] == {"hidden_layers": hidden_layers}
