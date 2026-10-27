"""Unit tests for lightweight DDQN_ECM behavior.

These tests intentionally stub model creation so they can verify constructor defaults and
environment switching without building real Keras networks or running training.
"""

from pathlib import Path

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


@pytest.fixture
def light_agent(monkeypatch):
    """Return DDQN_ECM with expensive neural-network setup replaced by stubs."""
    monkeypatch.setattr(agent_module.tf.keras.optimizers, "Adam", DummyOptimizer)

    def fake_setup_model(self):
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
    assert agent.model_dir == tmp_path / "models"
    assert (tmp_path / "model_summary.txt").read_text() == "dummy model\n"


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
