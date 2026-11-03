"""Unit tests for lightweight DDQN_ECM behavior.

Most tests stub model creation so they can verify constructor behavior without building real
Keras networks or running training. Focused architecture tests use small real Keras models.
"""

import numpy as np
import pandas as pd
import pytest

from autorec import agent as agent_module
from autorec.data_preparation import EISDataPrep
from autorec.progressive_spectrum import (
    CarryBestStatePolicy,
    EqualPointWindowSelector,
    InterpolatingSpectrumEncoder,
    ProgressiveSpectrumRuntimeConfig,
    StageTransitionDecision,
)


class DummyEnv:
    """Minimum environment shape needed by DDQN_ECM.__init__."""

    chromosome_HEAD_len = 2
    chromosome_TAIL_len = 3
    EIS_INPUT_SIZE = 4
    ELEMENTS_EXTENDED = ["+", "/", "R", "L", "P", "X"]
    ELEMENTS = ["+", "/", "R", "L", "P"]
    ACTIONS_LIST = list(range(16))

    def __init__(self, name="training"):
        self.name = name

    @property
    def metadata(self):
        return {"name": self.name}


class OneOperatorDummyEnv(DummyEnv):
    """Environment dimensions for an action space containing one operator."""

    ELEMENTS_EXTENDED = ["+", "R", "L", "P", "X"]
    ELEMENTS = ["+", "R", "L", "P"]
    ACTIONS_LIST = list(range(14))


class DummyModel:
    """Small model stand-in that supports the summary calls made by DDQN_ECM."""

    def summary(self, print_fn=None):
        lines = ["dummy model"]
        if print_fn is None:
            return None
        for line in lines:
            print_fn(line)
        return None


class ProgressiveEvaluationEnv:
    """Small prepared dataset container for progressive coordinator tests."""

    EIS_INPUT_SIZE = 12

    def __init__(self):
        frequency = np.geomspace(1e4, 1e-1, 6)
        log_frequency = np.log10(frequency)
        impedance = (10 + log_frequency**2) + 1j * (5 - log_frequency)
        dataprep = object.__new__(EISDataPrep)
        dataprep.frequency_npoints = 6
        dataprep.eis_features = ["ReZ", "ImZ"]
        self.dataprep = dataprep
        self.dataset = pd.DataFrame(
            {
                "freq": [frequency],
                "Z_true": [impedance],
                "flatten_Z": [dataprep.flatten_EIS(impedance, dataprep.eis_features)],
                "true_circuit": ["R0"],
            }
        )
        self.START_STATE = None
        self.current_spectrum_view_id = None


def dense_signature(model):
    """Return the units and activations of a model's Dense layers."""
    return [
        {
            "type": layer.__class__.__name__,
            "units": layer.units,
            "activation": layer.activation.__name__,
        }
        for layer in model.layers
        if isinstance(layer, agent_module.tf.keras.layers.Dense)
    ]


@pytest.fixture
def light_agent(monkeypatch):
    """Return DDQN_ECM with expensive neural-network setup replaced by stubs."""

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
    assert agent.beta_anneal_trials == pytest.approx(5.95)
    assert agent.prioritized_replay_beta == 0.2
    assert agent.gradient_steps == 1
    assert isinstance(agent.optimizer, agent_module.tf.keras.optimizers.Adam)
    assert float(agent.optimizer.learning_rate.numpy()) == pytest.approx(0.0005)
    assert agent.optimizer_config["type"] == "Adam"
    assert agent.optimizer_config["learning_rate"] == pytest.approx(0.0005)
    assert agent.hidden_layers == [
        {"type": "Dense", "units": 40, "activation": "relu"},
        {"type": "Dense", "units": 40, "activation": "relu"},
    ]
    assert agent.model_setup == (
        None,
        [
            {"type": "Dense", "units": 40, "activation": "relu"},
            {"type": "Dense", "units": 40, "activation": "relu"},
        ],
    )
    assert agent.model_name == tmp_path / "dqn_model.keras"
    assert agent.model_dir == tmp_path / "models"
    assert (tmp_path / "model_summary.txt").read_text() == "dummy model\n"


def test_absolute_beta_schedule_matches_legacy_horizon(light_agent, tmp_path):
    legacy = light_agent(
        training_env=DummyEnv(),
        save_dir=tmp_path / "legacy",
        num_trials=1000,
        start_jump=400,
        anneal_fraction=0.595,
    )
    absolute = light_agent(
        training_env=DummyEnv(),
        save_dir=tmp_path / "absolute",
        num_trials=100,
        start_jump=400,
        beta_anneal_trials=595,
    )

    for trial in range(1000):
        legacy._update_beta(trial)
        absolute._update_beta(trial)

    assert legacy.beta_list == absolute.beta_list
    assert legacy.prioritized_replay_beta == absolute.prioritized_replay_beta


def test_absolute_beta_schedule_rejects_invalid_durations(light_agent, tmp_path):
    for index, value in enumerate((0, -1, True, "595")):
        error = TypeError if value is True or isinstance(value, str) else ValueError
        with pytest.raises(error, match="beta_anneal_trials"):
            light_agent(
                training_env=DummyEnv(),
                save_dir=tmp_path / str(index),
                beta_anneal_trials=value,
            )


def test_absolute_and_fractional_beta_schedules_are_mutually_exclusive(light_agent, tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        light_agent(
            training_env=DummyEnv(),
            save_dir=tmp_path,
            anneal_fraction=0.595,
            beta_anneal_trials=595,
        )


def test_agent_forwards_custom_model_configuration(light_agent, tmp_path):
    hidden_layers = [
        {"type": "Dense", "units": 32, "activation": "tanh"},
        {"type": "Dense", "units": 16, "activation": "relu"},
        {"type": "Dense", "units": 8, "activation": "elu"},
    ]
    model_path = tmp_path / "existing.keras"

    agent = light_agent(
        training_env=DummyEnv(),
        save_dir=tmp_path,
        model=str(model_path),
        hidden_layers=hidden_layers,
        gradient_steps=3,
    )

    assert agent.hidden_layers is hidden_layers
    assert agent.gradient_steps == 3
    assert agent.model_setup == (str(model_path), hidden_layers)


def test_agent_rejects_invalid_gradient_steps(light_agent, tmp_path):
    cases = (
        (0, ValueError, "at least 1"),
        (-1, ValueError, "at least 1"),
        (1.5, TypeError, "must be an integer"),
        (True, TypeError, "must be an integer"),
    )
    for index, (gradient_steps, exception, message) in enumerate(cases):
        with pytest.raises(exception, match=message):
            light_agent(
                training_env=DummyEnv(),
                save_dir=tmp_path / str(index),
                gradient_steps=gradient_steps,
            )


def test_train_model_resamples_and_performs_requested_gradient_steps():
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.batch_size = 1
    agent.gamma = 0.99
    agent.prioritized_replay_beta = 0.4
    agent.prioritized_replay_eps = 1e-6

    inputs = agent_module.tf.keras.layers.Input(shape=(2,))
    outputs = agent_module.tf.keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros"
    )(inputs)
    agent.model = agent_module.tf.keras.Model(inputs=inputs, outputs=outputs)
    agent.target_model = agent_module.tf.keras.models.clone_model(agent.model)
    agent.target_model.set_weights(agent.model.get_weights())
    agent.optimizer = agent_module.tf.keras.optimizers.SGD(learning_rate=0.1)

    agent._active_env = type(
        "TrainingEnv",
        (),
        {
            "ACTIONS_LIST": pd.DataFrame([{"action_type": "R", "action_position": 0}]),
        },
    )()
    history = agent_module.CircularBuffer(
        capacity=1,
        dtypes=[
            ("EIS", "i4"),
            ("EIS_flatten", "f4", (1,)),
            ("encoded_state", "f4", (1,)),
            ("action_index", "i4"),
            ("encoded_new_state", "f4", (1,)),
            ("reward", "f4"),
            ("terminal_flag", "i1"),
            ("priority", "f4"),
        ],
    )
    history.add(
        EIS=0,
        EIS_flatten=np.array([1.0]),
        encoded_state=np.array([0.0]),
        action_index=0,
        encoded_new_state=np.array([0.0]),
        reward=1.0,
        terminal_flag=1,
        priority=1.0,
    )
    sample_count = 0

    def sample_experience(replay, beta):
        nonlocal sample_count
        sample_count += 1
        return replay.buffer[:1], np.array([0]), np.ones(1)

    agent._sample_experience = sample_experience

    loss = agent._train_model(gradient_steps=3, history=history)

    assert sample_count == 3
    assert agent.optimizer.iterations.numpy() == 3
    assert np.isfinite(loss)
    assert history["priority"][0] != 1.0


def test_replay_buffer_has_fixed_observation_and_view_identity_fields():
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.buffer_capacity = 3
    agent._active_env = type(
        "ReplayEnv",
        (),
        {
            "chromosome_len": 2,
            "ELEMENTS_EXTENDED": ["R", "P", "X"],
            "EIS_INPUT_SIZE": 4,
        },
    )()

    buffer = agent._create_replay_buffer()

    assert buffer.buffer.dtype["EIS_flatten"].shape == (4,)
    assert buffer.buffer.dtype["EIS_flatten"].base == np.dtype("f4")
    assert buffer.buffer.dtype["spectrum_view_id"].kind == "U"
    assert buffer.buffer.dtype["encoded_state"].shape == (6,)


def test_interaction_stores_exact_active_eis_observation_in_replay():
    class ReplayEnv:
        EIS_INPUT_SIZE = 3
        current_spectrum_view_id = "partial-view"
        ACTIONS_LIST = pd.DataFrame([{"action_type": "P", "action_position": 1}])

        def __init__(self):
            self.state = "+RR"
            self.encoded_state = np.array([1.0, 0.0])

        def step(self, action_type, action_position):
            self.state = "+PR"
            self.encoded_state = np.array([0.0, 1.0])
            return {
                "reward": 0.5,
                "validity": 1,
                "terminated": False,
                "coding": "+PR",
            }

    class RecordingBuffer:
        def __init__(self):
            self.buffer = np.zeros(1, dtype=[("priority", "f4")])
            self.size = 0
            self.added = None

        def __len__(self):
            return self.size

        def add(self, **kwargs):
            self.added = kwargs
            self.size += 1

    agent = object.__new__(agent_module.DDQN_ECM)
    agent._active_env = ReplayEnv()
    agent.eps_greedy = lambda flatten_Z, **kwargs: (1, "P")
    agent._check_and_train = lambda gradient_steps, buffer, iteration, memory: (None, memory)
    agent._check_and_update_target = lambda target_freq: target_freq + 1
    agent._detect_deadloop = lambda *args: ([], False)
    agent._action_index_maps = {}
    buffer = RecordingBuffer()
    partial_observation = np.array([0.2, 0.4, 0.8])

    agent._interact_update(
        action_num=0,
        flatten_Z=partial_observation,
        deadloop_flag=False,
        episode_history=[],
        deadloop_chain=[],
        episode=0,
        EIS_i=7,
        gradient_steps=1,
        buffer=buffer,
        iteration=0,
        target_freq=0,
        memory=0,
    )
    partial_observation[:] = -1

    np.testing.assert_array_equal(
        buffer.added["EIS_flatten"], np.array([0.2, 0.4, 0.8], dtype=np.float32)
    )
    assert buffer.added["spectrum_view_id"] == "partial-view"
    assert buffer.added["EIS"] == 7


def test_check_and_train_passes_numpy_buffer_without_dataframe_conversion():
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.NN_sleep = 10
    agent.train_frequency = 5
    buffer = agent_module.CircularBuffer(capacity=1, dtypes=[("priority", "f4")])
    buffer.add(priority=1.0)

    def fail_if_converted():
        raise AssertionError("training must not convert replay to a DataFrame")

    buffer.to_dataframe = fail_if_converted
    received = []

    def train_model(gradient_steps, replay):
        received.append((gradient_steps, replay))
        return 0.25

    agent._train_model = train_model

    loss, memory = agent._check_and_train(3, buffer, iteration=10, memory=4)

    assert received == [(3, buffer)]
    assert loss == pytest.approx(0.25)
    assert memory == 0


def test_default_hidden_layer_configs_are_not_shared(light_agent, tmp_path):
    first = light_agent(training_env=DummyEnv(), save_dir=tmp_path / "first")
    second = light_agent(training_env=DummyEnv(), save_dir=tmp_path / "second")

    first.hidden_layers[0]["units"] = 999

    assert second.hidden_layers[0]["units"] == 40


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


def test_progressive_evaluation_carries_best_state_and_runs_every_stage(monkeypatch):
    training_env = ProgressiveEvaluationEnv()
    eval_env = ProgressiveEvaluationEnv()
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.training_env = training_env
    agent.eval_env = eval_env
    agent._active_env = training_env
    agent.action_cap = 3
    calls = []
    best_states = ["+PR", "+LR", "+PP"]

    def fake_generate_ecm(**kwargs):
        stage_index = len(calls)
        calls.append(kwargs)
        active_env = agent._active_env
        active_env.START_STATE = kwargs["initial_state"] or "+RR"
        active_env.current_spectrum_view_id = f"view-{stage_index}"
        return {
            "ground_truth_circuit": "R0",
            "found_solution": stage_index == 0,
            "best_reward": 3.0 - stage_index,
            "best_state": best_states[stage_index],
        }

    monkeypatch.setattr(agent, "generate_ECM", fake_generate_ecm)

    result = agent.generate_progressive_ECM(
        EIS_i=0,
        num_windows=3,
        max_actions=2,
        verbose=False,
        use_eval_env=True,
    )

    assert agent.get_active_env is eval_env
    assert [call["initial_state"] for call in calls] == [None, "+PR", "+LR"]
    assert [stage["selected_point_count"] for stage in result["stage_results"]] == [
        2,
        4,
        6,
    ]
    assert [stage["continued_to_next_stage"] for stage in result["stage_results"]] == [
        True,
        True,
        False,
    ]
    assert result["completed_all_stages"] is True
    assert result["final_result"]["best_state"] == "+PP"


def test_progressive_evaluation_policy_can_reroute_and_stop(monkeypatch):
    env = ProgressiveEvaluationEnv()
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.training_env = env
    agent.eval_env = env
    agent._active_env = env
    agent.action_cap = 3
    calls = []

    def fake_generate_ecm(**kwargs):
        stage_index = len(calls)
        calls.append(kwargs)
        env.START_STATE = kwargs["initial_state"] or "+RR"
        env.current_spectrum_view_id = f"view-{stage_index}"
        return {
            "ground_truth_circuit": "R0",
            "found_solution": False,
            "best_reward": stage_index + 1.0,
            "best_state": ["+PR", "+LR"][stage_index],
        }

    class RerouteThenStopPolicy:
        def decide(self, outcome):
            if outcome.stage_index == 0:
                return StageTransitionDecision("+PP", True)
            return StageTransitionDecision(None, False)

    monkeypatch.setattr(agent, "generate_ECM", fake_generate_ecm)

    result = agent.generate_progressive_ECM(
        EIS_i=0,
        num_windows=3,
        verbose=False,
        transition_policy=RerouteThenStopPolicy(),
    )

    assert [call["initial_state"] for call in calls] == [None, "+PP"]
    assert len(result["stage_results"]) == 2
    assert result["completed_all_stages"] is False
    assert result["final_result"]["best_state"] == "+LR"


def test_progressive_evaluation_requires_dataprep_for_default_encoder():
    env = ProgressiveEvaluationEnv()
    env.dataprep = None
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.training_env = env
    agent.eval_env = env
    agent._active_env = env

    with pytest.raises(ValueError, match="must retain a loaded EISDataPrep"):
        agent.generate_progressive_ECM(EIS_i=0, num_windows=2, verbose=False)


def test_batch_evaluation_uses_runtime_and_can_disable_progressive_mode():
    env = ProgressiveEvaluationEnv()
    encoder = InterpolatingSpectrumEncoder(env.dataprep)
    runtime = ProgressiveSpectrumRuntimeConfig(
        enabled=True,
        num_windows=2,
        selector=EqualPointWindowSelector(2),
        training_encoder=encoder,
        evaluation_encoder=encoder,
        transition_policy=CarryBestStatePolicy(),
    )
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.training_env = env
    agent.eval_env = env
    agent._active_env = env
    agent.progressive_spectrum = runtime
    progressive_calls = []
    final_result = {
        "EIS_i": 0,
        "environment_type": "evaluation",
        "ground_truth_circuit": "R0",
        "found_solution": True,
        "best_reward": 2.0,
        "best_action_number": 1,
        "best_circuit": "R0",
        "best_coding": "R",
        "best_metrics": {"chi_square": 0.1},
        "best_param": {"R0": 1.0},
        "total_actions_taken": 2,
        "action_history": [],
    }

    def fake_progressive(**kwargs):
        progressive_calls.append(kwargs)
        return {
            "num_windows": 2,
            "completed_all_stages": True,
            "stage_results": [
                {"evaluation_result": {"total_actions_taken": 1}},
                {"evaluation_result": final_result},
            ],
            "final_result": final_result,
        }

    agent.generate_progressive_ECM = fake_progressive
    progressive_results = agent.eval_batch_eis(eis_indices=[0])

    assert progressive_calls[0]["encoder"] is encoder
    assert progressive_results.loc[0, "total_actions"] == 3
    assert progressive_results.loc[0, "progressive_completed_all_stages"]

    full_calls = []
    agent.generate_ECM = lambda **kwargs: full_calls.append(kwargs) or final_result
    full_results = agent.eval_batch_eis(eis_indices=[0], progressive=False)

    assert len(full_calls) == 1
    assert "progressive_num_windows" not in full_results


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
    with pytest.raises(ValueError, match="Unknown Keras optimizer type"):
        light_agent(
            training_env=DummyEnv(),
            save_dir=tmp_path,
            optimizer={"type": "NotAnOptimizer"},
        )


@pytest.mark.parametrize("optimizer", ["Adam", object()])
def test_agent_rejects_invalid_optimizer_inputs(light_agent, tmp_path, optimizer):
    with pytest.raises(TypeError, match="configuration dictionary or a Keras optimizer"):
        light_agent(training_env=DummyEnv(), save_dir=tmp_path, optimizer=optimizer)


def test_agent_builds_optimizer_from_config_without_mutating_it(light_agent, tmp_path):
    config = {"type": "SGD", "learning_rate": 0.01, "momentum": 0.9}

    agent = light_agent(training_env=DummyEnv(), save_dir=tmp_path, optimizer=config)

    assert isinstance(agent.optimizer, agent_module.tf.keras.optimizers.SGD)
    assert float(agent.optimizer.learning_rate.numpy()) == pytest.approx(0.01)
    assert float(agent.optimizer.momentum) == pytest.approx(0.9)
    assert agent.optimizer_config["type"] == "SGD"
    assert agent.optimizer_config["learning_rate"] == pytest.approx(0.01)
    assert agent.optimizer_config["momentum"] == pytest.approx(0.9)
    assert config == {"type": "SGD", "learning_rate": 0.01, "momentum": 0.9}


def test_agent_uses_custom_optimizer_object(light_agent, tmp_path):
    optimizer = agent_module.tf.keras.optimizers.SGD(learning_rate=0.01)

    agent = light_agent(training_env=DummyEnv(), save_dir=tmp_path, optimizer=optimizer)

    assert agent.optimizer is optimizer
    assert agent.optimizer_config["type"] == "SGD"
    assert agent.optimizer_config["learning_rate"] == pytest.approx(0.01)


def test_agent_builds_requested_hidden_layers(tmp_path):
    hidden_layers = [
        {"type": "Dense", "units": 32, "activation": "tanh"},
        {"type": "Dense", "units": 16, "activation": "relu"},
        {"type": "Dense", "units": 8, "activation": "elu"},
    ]
    agent = agent_module.DDQN_ECM(
        training_env=DummyEnv(),
        save_dir=tmp_path,
        hidden_layers=hidden_layers,
    )

    assert dense_signature(agent.model) == [
        {"type": "Dense", "units": 32, "activation": "tanh"},
        {"type": "Dense", "units": 16, "activation": "relu"},
        {"type": "Dense", "units": 8, "activation": "elu"},
        {"type": "Dense", "units": 16, "activation": "linear"},
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


def test_agent_output_size_matches_one_operator_action_list(tmp_path):
    env = OneOperatorDummyEnv()

    agent = agent_module.DDQN_ECM(
        training_env=env,
        save_dir=tmp_path,
        hidden_layers=[{"type": "Dense", "units": 8, "activation": "relu"}],
    )

    assert agent.model.input_shape == (None, 29)
    assert agent.model.output_shape == (None, len(env.ACTIONS_LIST))


def test_agent_metadata_records_configuration(light_agent, tmp_path):
    training_env = DummyEnv("training")
    eval_env = DummyEnv("evaluation")
    model_path = tmp_path / "input.keras"
    optimizer = {"type": "SGD", "learning_rate": 0.001, "momentum": 0.9}
    agent = light_agent(
        training_env=training_env,
        eval_env=eval_env,
        save_dir=tmp_path,
        save_start=3,
        save_frequency=20,
        action_cap=11,
        random_seed=7,
        episodes_trial=2,
        num_trials=100,
        gamma=0.8,
        continuous_deadloop=3,
        latent_deadloop=6,
        invalid_terminals=True,
        initial_epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        start_decay=4,
        bayesian=True,
        model=model_path,
        hidden_layers=[{"type": "Dense", "units": 12, "activation": "tanh"}],
        batch_size=32,
        train_frequency=2,
        update_target_frequency=50,
        NN_sleep=40,
        buffer_capacity=500,
        optimizer=optimizer,
        prioritized_replay_alpha=0.7,
        prioritized_replay_eps=1e-5,
        initial_beta=0.3,
        beta_jump=1.02,
        start_jump=10,
        anneal_fraction=0.4,
        final_beta=0.9,
    )

    metadata = agent.metadata
    optimizer_metadata = metadata.pop("optimizer")

    assert optimizer_metadata["type"] == "SGD"
    assert optimizer_metadata["learning_rate"] == pytest.approx(0.001)
    assert optimizer_metadata["momentum"] == pytest.approx(0.9)
    assert metadata == {
        "training_env": {"name": "training"},
        "eval_env": {"name": "evaluation"},
        "save_dir": str(tmp_path),
        "save_start": 3,
        "save_frequency": 20,
        "action_cap": 11,
        "random_seed": 7,
        "episodes_trial": 2,
        "num_trials": 100,
        "gamma": 0.8,
        "continuous_deadloop": 3,
        "latent_deadloop": 6,
        "invalid_terminals": True,
        "initial_epsilon": 0.9,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.99,
        "start_decay": 4,
        "bayesian": True,
        "model": str(model_path),
        "hidden_layers": [{"type": "Dense", "units": 12, "activation": "tanh"}],
        "batch_size": 32,
        "train_frequency": 2,
        "update_target_frequency": 50,
        "gradient_steps": 1,
        "NN_sleep": 40,
        "buffer_capacity": 500,
        "prioritized_replay_alpha": 0.7,
        "prioritized_replay_eps": 1e-5,
        "initial_beta": 0.3,
        "beta_jump": 1.02,
        "start_jump": 10,
        "anneal_fraction": 0.4,
        "beta_anneal_trials": 40.0,
        "final_beta": 0.9,
    }


def test_agent_builds_generic_hidden_layer_stack(tmp_path):
    hidden_layers = [
        {"type": "Dense", "units": 24, "activation": "relu"},
        {"type": "Dropout", "rate": 0.2},
        {"type": "LayerNormalization"},
    ]

    agent = agent_module.DDQN_ECM(
        training_env=DummyEnv(),
        save_dir=tmp_path,
        hidden_layers=hidden_layers,
    )

    assert [layer.__class__.__name__ for layer in agent.model.layers] == [
        "Dense",
        "Dropout",
        "LayerNormalization",
        "Dense",
    ]
    assert agent.model.layers[1].rate == pytest.approx(0.2)
    assert [layer.__class__.__name__ for layer in agent.target_model.layers] == [
        layer.__class__.__name__ for layer in agent.model.layers
    ]


def test_model_file_takes_precedence_over_hidden_layers(tmp_path):
    inputs = agent_module.tf.keras.layers.Input(shape=(34,))
    hidden = agent_module.tf.keras.layers.Dense(12, activation="sigmoid")(inputs)
    outputs = agent_module.tf.keras.layers.Dense(16, activation="linear")(hidden)
    saved_model = agent_module.tf.keras.Model(inputs=inputs, outputs=outputs)
    model_path = tmp_path / "saved.keras"
    saved_model.save(model_path)

    agent = agent_module.DDQN_ECM(
        training_env=DummyEnv(),
        save_dir=tmp_path / "loaded",
        model=str(model_path),
        hidden_layers=[{"type": "Dense", "units": 999, "activation": "relu"}],
    )

    assert len(agent.hidden_layers) == 1
    assert agent.hidden_layers[0]["type"] == "Dense"
    assert agent.hidden_layers[0]["units"] == 12
    assert agent.hidden_layers[0]["activation"] == "sigmoid"
    assert dense_signature(agent.model) == [
        {"type": "Dense", "units": 12, "activation": "sigmoid"},
        {"type": "Dense", "units": 16, "activation": "linear"},
    ]
    assert agent.target_model.input_shape == agent.model.input_shape
    assert agent.target_model.output_shape == agent.model.output_shape
    assert dense_signature(agent.target_model) == dense_signature(agent.model)
    for target_weight, model_weight in zip(
        agent.target_model.get_weights(), agent.model.get_weights(), strict=True
    ):
        np.testing.assert_array_equal(target_weight, model_weight)
