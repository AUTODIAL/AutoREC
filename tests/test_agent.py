"""Unit tests for lightweight DDQN_ECM behavior.

Most tests stub model creation so they can verify constructor behavior without building real
Keras networks or running training. Focused architecture tests use small real Keras models.
"""

from pathlib import Path

import numpy as np
import pytest

from autorec import agent as agent_module


class DummyEnv:
    """Minimum environment shape needed by DDQN_ECM.__init__."""

    chromosome_HEAD_len = 2
    chromosome_TAIL_len = 3
    EIS_INPUT_SIZE = 4
    ELEMENTS_EXTENDED = ["+", "/", "R", "L", "P", "X"]
    ELEMENTS = ["+", "/", "R", "L", "P"]


class DummyModel:
    """Small model stand-in that supports the summary calls made by DDQN_ECM."""

    def summary(self, print_fn=None):
        lines = ["dummy model"]
        if print_fn is None:
            return None
        for line in lines:
            print_fn(line)
        return None


class DummyOptimizer:
    """Records optimizer construction without importing real optimizer behavior."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


def dense_signature(model):
    """Return the units and activations of a model's Dense layers."""
    return [
        (layer.units, layer.activation.__name__)
        for layer in model.layers
        if isinstance(layer, agent_module.layers.Dense)
    ]


@pytest.fixture
def light_agent(monkeypatch):
    """Return DDQN_ECM with expensive neural-network setup replaced by stubs."""
    monkeypatch.setattr(agent_module.tf.keras.optimizers, "Adam", DummyOptimizer)

    def fake_setup_model(self, model, hidden_layers):
        self.model_setup = (model, hidden_layers)
        self.model = DummyModel()
        self.target_model = DummyModel()

    monkeypatch.setattr(agent_module.DDQN_ECM, "_setup_model", fake_setup_model)
    return agent_module.DDQN_ECM


def test_agent_initializes_defaults_and_writes_summary(light_agent, tmp_path):
    agent = light_agent(
        training_env=DummyEnv(),
        save_dir=tmp_path,
        num_trials=10,
        initial_beta=0.2,
    )

    assert agent.training_env is agent.eval_env
    assert agent.get_active_env is agent.training_env
    assert agent.get_current_env_type() == "training"
    # Default action cap is derived from the active environment chromosome length.
    assert agent.action_cap == DummyEnv.chromosome_HEAD_len + DummyEnv.chromosome_TAIL_len + 3
    assert agent.start_jump == 4
    assert agent.anneal_fraction == pytest.approx(0.595)
    assert agent.prioritized_replay_beta == 0.2
    assert agent.hidden_layers == [[40, "relu"], [40, "relu"]]
    assert agent.model_setup == (None, [[40, "relu"], [40, "relu"]])
    assert agent.model_dir == tmp_path / "models"
    assert (tmp_path / "model_summary.txt").read_text() == "dummy model\n"


def test_agent_forwards_custom_model_configuration(light_agent, tmp_path):
    hidden_layers = [[32, "tanh"], [16, "relu"], [8, "elu"]]
    model_path = tmp_path / "existing.keras"

    agent = light_agent(
        training_env=DummyEnv(),
        save_dir=tmp_path,
        model=str(model_path),
        hidden_layers=hidden_layers,
    )

    assert agent.hidden_layers is hidden_layers
    assert agent.model_setup == (str(model_path), hidden_layers)


def test_agent_switches_between_training_and_eval_envs(light_agent, tmp_path):
    training_env = DummyEnv()
    eval_env = DummyEnv()
    agent = light_agent(training_env=training_env, eval_env=eval_env, save_dir=tmp_path)

    agent.switch_to_eval()
    assert agent.get_active_env is eval_env
    assert agent.get_current_env_type() == "evaluation"

    agent.switch_to_training()
    assert agent.get_active_env is training_env
    assert agent.get_current_env_type() == "training"


def test_agent_can_replace_envs(light_agent, tmp_path):
    agent = light_agent(training_env=DummyEnv(), save_dir=tmp_path)
    new_training_env = DummyEnv()
    new_eval_env = DummyEnv()

    agent.set_training_env(new_training_env, switch_to_it=True)
    assert agent.training_env is new_training_env
    assert agent.get_active_env is new_training_env

    agent.set_eval_env(new_eval_env, switch_to_it=True)
    assert agent.eval_env is new_eval_env
    assert agent.get_active_env is new_eval_env


def test_agent_rejects_unknown_optimizer(light_agent, tmp_path):
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        light_agent(
            training_env=DummyEnv(),
            save_dir=tmp_path,
            optimizer_type="not-an-optimizer",
        )


def test_agent_uses_custom_optimizer_object(light_agent, tmp_path):
    optimizer = object()

    agent = light_agent(
        training_env=DummyEnv(), save_dir=Path(tmp_path), optimizer_type=optimizer
    )

    assert agent.optimizer is optimizer


def test_agent_builds_requested_hidden_layers(tmp_path):
    hidden_layers = [[32, "tanh"], [16, "relu"], [8, "elu"]]

    agent = agent_module.DDQN_ECM(
        training_env=DummyEnv(),
        save_dir=tmp_path,
        hidden_layers=hidden_layers,
        optimizer_type=object(),
    )

    assert dense_signature(agent.model) == [
        (32, "tanh"),
        (16, "relu"),
        (8, "elu"),
        (16, "linear"),
    ]
    assert agent.model.input_shape == (None, 34)
    assert agent.model.output_shape == (None, 16)
    assert agent.target_model.input_shape == agent.model.input_shape
    assert agent.target_model.output_shape == agent.model.output_shape
    assert dense_signature(agent.target_model) == dense_signature(agent.model)
    for target_weight, model_weight in zip(
        agent.target_model.get_weights(), agent.model.get_weights(), strict=True
    ):
        np.testing.assert_array_equal(target_weight, model_weight)


def test_model_file_takes_precedence_over_hidden_layers(tmp_path):
    inputs = agent_module.layers.Input(shape=(34,))
    hidden = agent_module.layers.Dense(12, activation="sigmoid")(inputs)
    outputs = agent_module.layers.Dense(16, activation="linear")(hidden)
    saved_model = agent_module.tf.keras.Model(inputs=inputs, outputs=outputs)
    model_path = tmp_path / "saved.keras"
    saved_model.save(model_path)

    agent = agent_module.DDQN_ECM(
        training_env=DummyEnv(),
        save_dir=tmp_path / "loaded",
        model=str(model_path),
        hidden_layers=[[999, "relu"]],
        optimizer_type=object(),
    )

    assert agent.hidden_layers == [(12, "sigmoid")]
    assert dense_signature(agent.model) == [(12, "sigmoid"), (16, "linear")]
    assert agent.target_model.input_shape == agent.model.input_shape
    assert agent.target_model.output_shape == agent.model.output_shape
    assert dense_signature(agent.target_model) == dense_signature(agent.model)
    for target_weight, model_weight in zip(
        agent.target_model.get_weights(), agent.model.get_weights(), strict=True
    ):
        np.testing.assert_array_equal(target_weight, model_weight)
