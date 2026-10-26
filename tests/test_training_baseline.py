"""Characterize training before adding persistent state or checkpoints.

The short CPU run uses the real agent, environment state/cache logic, replay, sampling,
and Adam updates. Circuit parsing/encoding, fitting, and model/plot output are replaced.
Exact repeatability here is a same-process baseline, not a claim about real scientific
execution or checkpoint recovery.
"""

import random
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from autorec import agent as agent_module
from autorec import environment as environment_module
from autorec.environment import EIS_ECM_Env
from autorec.neural_state import neural_state_dict, restore_neural_state
from autorec.optimized_data_structures.circular_buffer import CircularBuffer
from autorec.optimized_data_structures.circuit_cache import HistoryCache


@pytest.fixture(autouse=True)
def preserve_random_states():
    """Keep these tests' Python/NumPy draws out of other tests' random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


@pytest.fixture
def environment_factory(monkeypatch):
    """Use the same circuit-helper boundary as test_environment, without Julia setup."""

    class CircuitHelpers:
        @staticmethod
        def karva_to_tree(karva):
            return list(karva)

        @staticmethod
        def tree_to_circuit(tree):
            return ("R1-[R2,L1]", None)

    monkeypatch.setattr(environment_module, "ec", CircuitHelpers)
    monkeypatch.setattr(
        environment_module,
        "state_encode",
        lambda state, elements: np.eye(len(elements), dtype=np.float32)[
            [elements.index(char) for char in state]
        ].ravel(),
    )
    monkeypatch.setattr(
        environment_module,
        "action_validity",
        lambda state, action_type, action_position: int(state[action_position] != action_type),
    )
    monkeypatch.setattr(
        environment_module,
        "parse_state_to_circuit",
        lambda state: ("R1-[R2,L1]", len(state), state),
    )
    return make_env


def make_env(cache_type="lru"):
    dataset = pd.DataFrame(
        {
            "Z_true": [np.array([1 + 1j, 2 + 2j]), np.array([2 + 1j, 3 + 2j])],
            "freq": [np.array([1.0, 10.0])] * 2,
            "flatten_Z": [np.array([0.25, 0.5]), np.array([0.5, 0.75])],
            "true_circuit": ["R1-[P1,R2]"] * 2,
            "chi_thresh": [0.1] * 2,
            "r2_thresh": [0.9] * 2,
            "metadata": [{} for _ in range(2)],
        }
    )
    return EIS_ECM_Env(
        dataset=dataset,
        initial_state=["+/RPR", "+/RLR"],
        chromosome_HEAD_len=2,
        seed=7,
        cache_type=cache_type,
        cache_capacity=3,
    )


def controlled_fit(circuit, components_number):
    """Make cache misses consume randomness without invoking the scientific fitter."""
    return {
        "reward": float(np.random.uniform(0.125, 0.25)),
        "metrics": {"chi_square": 1.0, "r2_score": 0.5},
        "predicted_Z": np.array([1 + 1j, 2 + 2j]),
        "param": {},
        "terminated": False,
        "depth_penalty": 0,
    }


def array_state(value):
    """Compare numerical state bitwise, including dtype and shape."""
    value = np.asarray(value)
    return value.dtype, value.shape, value.tobytes()


def training_progress(state):
    return (
        state.iteration,
        state.target_freq,
        state.memory,
        state.success_count,
        state.completed_trials,
    )


def freeze_state(value):
    """Snapshot nested cache contents in order without calling cache lookup methods."""
    if isinstance(value, (np.ndarray, np.generic)):
        return array_state(value)
    if isinstance(value, dict):
        return tuple((key, freeze_state(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(freeze_state(item) for item in value)
    return value


def run_training(
    output_dir,
    monkeypatch,
    environment_factory,
    unrelated_numpy_seed,
    *,
    sessions=(None,),
    warmup=3,
    cache_type="lru",
    default_beta_schedule=False,
    roundtrip_cache=False,
    neural_mode=None,
    optimizer_config=None,
):
    random.seed(42)
    agent_module.tf.random.set_seed(42)
    # Environment construction must reseed NumPy independently of prior caller draws.
    np.random.seed(unrelated_numpy_seed)
    env = environment_factory(cache_type)
    initial_observation = (env.current_eis_index, env.state)
    trace = {"events": [], "resets": [], "transitions": [], "samples": [], "copies": []}
    trace["saves"] = []
    trace["plots"] = []
    buffers = []

    def capture_buffer(*args, **kwargs):
        buffer = CircularBuffer(*args, **kwargs)
        buffers.append(buffer)
        original_add = buffer.add

        def add(**transition):
            trace["transitions"].append(
                {key: value for key, value in transition.items() if "encoded" not in key}
            )
            original_add(**transition)
            trace["events"].append(("insert", len(trace["transitions"])))

        buffer.add = add
        return buffer

    with monkeypatch.context() as patch, agent_module.tf.device("/CPU:0"):
        patch.setattr(agent_module, "CircularBuffer", capture_buffer)
        patch.setattr(env, "_evaluate_circuit", controlled_fit)
        agent = agent_module.DDQN_ECM(
            training_env=env,
            save_dir=output_dir,
            num_trials=4,
            episodes_trial=2,
            action_cap=2,
            hidden_layers=[{"type": "Dense", "units": 4, "activation": "relu"}],
            batch_size=2,
            buffer_capacity=7,
            NN_sleep=warmup,
            train_frequency=3,
            update_target_frequency=5,
            gradient_steps=2,
            optimizer=optimizer_config,
            initial_epsilon=1.0,
            epsilon_min=0.125,
            epsilon_decay=0.5,
            start_decay=0,
            initial_beta=0.25,
            final_beta=0.75,
            start_jump=None if default_beta_schedule else 0,
            anneal_fraction=None if default_beta_schedule else 0.5,
        )
        assert agent._replay_buffer is None
        assert agent._training_state is None
        initial_weights = [array_state(value) for value in agent.model.weights]
        original_reset = env.reset
        original_train = agent._train_model
        original_sample = agent._sample_experience
        original_copy = agent.target_model.set_weights

        def reset(*args, **kwargs):
            # The next trial must see the previous trial's completed state, not a
            # snapshot assigned only when train() returns.
            steps = len(trace["transitions"])
            memory = steps if steps <= warmup else (steps - warmup - 1) % 3
            assert training_progress(agent._training_state) == (
                steps,
                steps % 5,
                memory,
                0,
                steps // 4,
            )
            result = original_reset(*args, **kwargs)
            trace["resets"].append(
                (len(trace["transitions"]), env.current_eis_index, env.state)
            )
            return result

        def train_model(gradient_steps, history):
            trace["events"].append(("train", len(trace["transitions"])))
            loss = original_train(gradient_steps, history)
            trace["last_priorities"] = array_state(history["priority"].to_numpy())
            return loss

        def sample(history, beta):
            result = original_sample(history, beta)
            trace["samples"].append((result[1].tolist(), array_state(result[2]), beta))
            return result

        def copy_target(weights):
            trace["events"].append(("target", len(trace["transitions"])))
            trace["copies"].append(int(agent.optimizer.iterations.numpy()))
            trace["last_target_weights"] = [array_state(value) for value in weights]
            original_copy(weights)

        patch.setattr(env, "reset", reset)
        patch.setattr(agent, "_train_model", train_model)
        patch.setattr(agent, "_sample_experience", sample)
        patch.setattr(agent.target_model, "set_weights", copy_target)
        patch.setattr(agent.model, "save", lambda path: trace["saves"].append(path.name))
        patch.setattr(
            agent,
            "_plot_training_metrics",
            lambda *args: trace["plots"].append(agent.completed_trials),
        )
        completed = 0
        for count in sessions:
            results = agent.train(additional_trials=count)
            completed = agent.num_trials if count is None else completed + count
            assert agent.completed_trials == completed
            assert len(buffers) == 1
            assert agent._replay_buffer is buffers[0]
            assert agent._training_state.at_trial_boundary
            assert len(results["success_rate"]) == completed * agent.episodes_trial
            if neural_mode is not None:
                neural_before = tuple(
                    tuple(array_state(variable) for variable in component.variables)
                    for component in (agent.model, agent.target_model, agent.optimizer)
                )
                rng_before = freeze_state(np.random.get_state()), random.getstate()
                was_built = agent.optimizer.built
                snapshot = neural_state_dict(agent.model, agent.target_model, agent.optimizer)
                if neural_mode == "restore":
                    agent.optimizer = restore_neural_state(
                        snapshot, agent.model, agent.target_model
                    )
                assert agent.optimizer.built == was_built
                assert (
                    tuple(
                        tuple(array_state(variable) for variable in component.variables)
                        for component in (agent.model, agent.target_model, agent.optimizer)
                    )
                    == neural_before
                )
                assert (freeze_state(np.random.get_state()), random.getstate()) == rng_before
            if roundtrip_cache:
                cache_before = freeze_state(vars(env.cache.cache))
                rng_before = freeze_state(np.random.get_state()), random.getstate()
                env.cache = HistoryCache.from_state_dict(env.cache.state_dict())
                assert freeze_state(vars(env.cache.cache)) == cache_before
                assert (freeze_state(np.random.get_state()), random.getstate()) == rng_before
            if completed < agent.num_trials:
                assert "dqn_model.keras" not in trace["saves"]
                assert trace["plots"] == []
                assert not (output_dir / "NN_loss.csv").exists()

    state = {
        "initial_observation": initial_observation,
        "initial_weights": initial_weights,
        "model": [array_state(value) for value in agent.model.variables],
        "target": [array_state(value) for value in agent.target_model.variables],
        "optimizer": [array_state(value) for value in agent.optimizer.variables],
        "replay": array_state(buffers[0].buffer),
        "numpy_rng": [array_state(value) for value in np.random.get_state()],
        "python_rng": random.getstate(),
        "cache_stats": env.get_cache_stats(),
        "cache": freeze_state(vars(env.cache.cache)),
        "environment": (
            env.current_eis_index,
            env.START_STATE,
            env.state,
            array_state(env.encoded_state),
        ),
        "progress": training_progress(agent._training_state),
        "trace": trace,
    }
    return agent, buffers[0], results, state


@pytest.mark.parametrize(
    (
        "sessions",
        "warmup",
        "cache_type",
        "default_beta_schedule",
        "roundtrip_cache",
        "neural_mode",
    ),
    [
        ((1, 1, 1, 1), 3, "lru", False, False, None),
        ((1, None), 5, "clock", True, False, None),
        ((0, 2, 0, 2), 3, "clock", True, False, None),
        ((4,), 5, "lru", False, False, None),
        ((1, 1, 1, 1), 3, "lru", False, True, None),
        ((1, 1, 1, 1), 5, "clock", True, True, None),
        ((0, 1, 1, 1, 1), 5, "lru", True, False, "export"),
        ((0, 1, 1, 1, 1), 5, "lru", True, False, "restore"),
    ],
)
def test_continuous_and_split_training_match_exactly(
    tmp_path,
    monkeypatch,
    environment_factory,
    sessions,
    warmup,
    cache_type,
    default_beta_schedule,
    roundtrip_cache,
    neural_mode,
):
    options = dict(
        warmup=warmup, cache_type=cache_type, default_beta_schedule=default_beta_schedule
    )
    if neural_mode is not None:
        options["optimizer_config"] = {
            "type": "Adam",
            "learning_rate": {
                "type": "ExponentialDecay",
                "initial_learning_rate": 0.0005,
                "decay_steps": 2,
                "decay_rate": 0.5,
            },
        }
    agent, buffer, results, state = run_training(
        tmp_path / "continuous", monkeypatch, environment_factory, 123, **options
    )
    other, other_buffer, other_results, other_state = run_training(
        tmp_path / "split",
        monkeypatch,
        environment_factory,
        999,
        sessions=sessions,
        roundtrip_cache=roundtrip_cache,
        neural_mode=neural_mode,
        **options,
    )
    assert state == other_state
    assert (buffer.capacity, buffer.size, buffer.position) == (
        other_buffer.capacity,
        other_buffer.size,
        other_buffer.position,
    )
    assert (
        agent.epsilon,
        agent.epsilon_list,
        agent.prioritized_replay_beta,
        agent.beta_list,
    ) == (other.epsilon, other.epsilon_list, other.prioritized_replay_beta, other.beta_list)
    assert agent.num_trials == other.num_trials == 4
    assert agent.completed_trials == other.completed_trials == 4
    assert state["trace"]["saves"] == ["dqn_model_trial_0.keras", "dqn_model.keras"]
    assert state["trace"]["plots"] == [4]
    for name in ("NN_loss", "success_rate", "terminal_states"):
        pd.testing.assert_frame_equal(results[name], other_results[name], check_exact=True)
        assert (tmp_path / "continuous" / f"{name}.csv").read_bytes() == (
            tmp_path / "split" / f"{name}.csv"
        ).read_bytes()
    assert results["statistical_analysis"] == other_results["statistical_analysis"]
    pd.testing.assert_frame_equal(
        agent.training_env.dataset, other.training_env.dataset, check_exact=True
    )


@pytest.fixture
def continuation_agent(tmp_path, monkeypatch, environment_factory):
    env = environment_factory()
    monkeypatch.setattr(env, "_evaluate_circuit", controlled_fit)
    agent = agent_module.DDQN_ECM(
        training_env=env,
        save_dir=tmp_path,
        num_trials=4,
        action_cap=1,
        hidden_layers=[{"type": "Dense", "units": 2, "activation": "relu"}],
        buffer_capacity=5,
        batch_size=2,
        NN_sleep=100,
        save_frequency=2,
    )
    monkeypatch.setattr(agent.model, "save", Mock())
    monkeypatch.setattr(agent, "_plot_training_metrics", Mock())
    return agent


@pytest.mark.parametrize(
    ("count", "error"),
    [(True, TypeError), (1.5, TypeError), ("1", TypeError), (-1, ValueError), (5, ValueError)],
)
def test_invalid_trial_count_does_not_initialize_or_consume_randomness(
    continuation_agent, count, error
):
    agent = continuation_agent
    rng = freeze_state(np.random.get_state()), random.getstate()
    with pytest.raises(error):
        agent.train(additional_trials=count)
    assert agent._training_state is None
    assert agent._replay_buffer is None
    assert not agent._continuation_mode
    assert agent.completed_trials == 0
    assert (freeze_state(np.random.get_state()), random.getstate()) == rng
    agent.model.save.assert_not_called()


def test_zero_steps_finish_remaining_and_completed_noop(continuation_agent):
    agent = continuation_agent
    rng = freeze_state(np.random.get_state()), random.getstate()
    empty = agent.train(additional_trials=0)
    assert agent.completed_trials == 0
    assert empty["success_rate"].empty
    assert agent._replay_buffer.size == 0
    assert (freeze_state(np.random.get_state()), random.getstate()) == rng
    agent.model.save.assert_not_called()
    agent._plot_training_metrics.assert_not_called()
    with pytest.raises(AttributeError):
        agent.completed_trials = 1

    first_state, first_buffer = agent._training_state, agent._replay_buffer
    partial = agent.train(additional_trials=2)
    assert agent.completed_trials == 2
    assert len(partial["success_rate"]) == 2
    assert not (agent.save_dir / "success_rate.csv").exists()
    assert agent.model.save.call_count == 1  # Original intermediate save at global t=0.
    agent._plot_training_metrics.assert_not_called()
    rng = freeze_state(np.random.get_state()), random.getstate()
    with pytest.raises(ValueError, match="exceed"):
        agent.train(additional_trials=3)
    unchanged = agent.train(additional_trials=0)
    assert unchanged["success_rate"] is partial["success_rate"]
    assert agent.completed_trials == 2
    assert agent.model.save.call_count == 1
    assert (freeze_state(np.random.get_state()), random.getstate()) == rng

    complete = agent.train()  # Continuation mode completes the remaining two trials.
    assert agent.completed_trials == 4
    assert agent._training_state is first_state
    assert agent._replay_buffer is first_buffer
    assert complete["success_rate"]["t"].tolist() == [0, 1, 2, 3]
    assert [call.args[0].name for call in agent.model.save.call_args_list] == [
        "dqn_model_trial_0.keras",
        "dqn_model_trial_2.keras",
        "dqn_model.keras",
    ]
    agent._plot_training_metrics.assert_called_once()
    assert (agent.save_dir / "success_rate.csv").exists()
    rng = freeze_state(np.random.get_state()), random.getstate()
    for count in (None, 0):
        repeated = agent.train(additional_trials=count)
        assert repeated["success_rate"] is complete["success_rate"]
    assert agent.model.save.call_count == 3
    agent._plot_training_metrics.assert_called_once()
    assert (freeze_state(np.random.get_state()), random.getstate()) == rng


@pytest.mark.parametrize(
    "name",
    [
        "num_trials",
        "start_decay",
        "epsilon_decay",
        "epsilon_min",
        "start_jump",
        "anneal_fraction",
        "initial_beta",
        "final_beta",
    ],
)
def test_continuation_rejects_changed_horizon_or_schedules(continuation_agent, name):
    agent = continuation_agent
    agent.train(additional_trials=1)
    progress = training_progress(agent._training_state)
    replay = array_state(agent._replay_buffer.buffer)
    rng = freeze_state(np.random.get_state()), random.getstate()
    setattr(agent, name, getattr(agent, name) + 1)
    with pytest.raises(ValueError, match="horizon or epsilon/beta schedules"):
        agent.train(additional_trials=1)
    assert training_progress(agent._training_state) == progress
    assert array_state(agent._replay_buffer.buffer) == replay
    assert (freeze_state(np.random.get_state()), random.getstate()) == rng


def test_continuation_rejects_unfinished_trial(continuation_agent, monkeypatch):
    agent = continuation_agent
    failed_step = Mock(side_effect=RuntimeError("injected step failure"))
    monkeypatch.setattr(agent.training_env, "step", failed_step)
    with pytest.raises(RuntimeError, match="injected step failure"):
        agent.train(additional_trials=1)
    assert agent.completed_trials == 0
    assert not agent._training_state.at_trial_boundary
    rng = freeze_state(np.random.get_state()), random.getstate()
    with pytest.raises(RuntimeError, match="unfinished trial"):
        agent.train()
    failed_step.assert_called_once()
    assert (freeze_state(np.random.get_state()), random.getstate()) == rng


def test_seeded_training_preserves_trial_update_order_and_is_exactly_repeatable(
    tmp_path, monkeypatch, environment_factory
):
    agent, buffer, results, state = run_training(
        tmp_path / "first", monkeypatch, environment_factory, 123
    )
    other, other_buffer, other_results, other_state = run_training(
        tmp_path / "second", monkeypatch, environment_factory, 999
    )
    assert agent._replay_buffer is buffer
    assert other._replay_buffer is other_buffer
    assert agent._training_state is not other._training_state
    assert training_progress(agent._training_state) == (16, 1, 0, 0, 4)
    assert training_progress(other._training_state) == (16, 1, 0, 0, 4)
    for name, value in results.items():
        assert getattr(agent._training_state, name) is value
        assert getattr(other._training_state, name) is other_results[name]
        assert value is not other_results[name]

    # Fixed semantic expectations supplement repeatability: two equally wrong runs
    # must not pass if warm-up, cadence, or ordering changes in a later refactor.
    trace = state["trace"]
    expected_events = []
    for step in range(1, 17):
        expected_events.append(("insert", step))
        if step in (4, 7, 10, 13, 16):
            expected_events.append(("train", step))
        if step in (5, 10, 15):
            expected_events.append(("target", step))
    assert trace["events"] == expected_events
    assert trace["copies"] == [2, 6, 8]  # Copy follows learning on shared step 10.
    assert agent.optimizer.iterations.numpy() == 10
    assert len(trace["samples"]) == 10  # Two real samples per training event.
    assert [row[0] for row in trace["resets"]] == [0, 4, 8, 12]
    assert results["success_rate"]["t"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert results["NN_loss"]["t"].tolist() == [0, 1, 2, 3, 3]
    assert np.isfinite(results["NN_loss"]["loss"]).all()
    assert len(results["statistical_analysis"]["episodic_cumul_reward"]) == 8
    assert agent.epsilon_list == [1.0, 1.0, 0.5, 0.25]
    assert agent.epsilon == 0.125
    assert agent.beta_list == [0.25, 0.25, 0.5, 0.75]
    assert agent.prioritized_replay_beta == 0.75
    assert [row[2] for row in trace["samples"]] == [0.25] * 4 + [0.5] * 2 + [0.75] * 4

    # A wrapped ring is sampled in physical slot order, not chronological order.
    assert (buffer.size, buffer.position) == (7, 2)
    physical_steps = [14, 15, 9, 10, 11, 12, 13]  # Zero-based transition indices.
    for field in ("EIS", "state", "new_state", "action_type", "action_position"):
        assert buffer.buffer[field].tolist() == [
            trace["transitions"][ii][field] for ii in physical_steps
        ]
    assert array_state(buffer.buffer["priority"]) == trace["last_priorities"]
    assert state["model"] != state["initial_weights"]
    assert state["target"] == trace["last_target_weights"]
    assert state["model"] != state["target"]  # Learning at step 16 leaves target lag.

    assert state == other_state
    assert (other_buffer.size, other_buffer.position) == (buffer.size, buffer.position)
    assert (other.epsilon, other.prioritized_replay_beta) == (
        agent.epsilon,
        agent.prioritized_replay_beta,
    )
    for name in ("NN_loss", "success_rate", "terminal_states"):
        pd.testing.assert_frame_equal(results[name], other_results[name], check_exact=True)
    assert results["statistical_analysis"] == other_results["statistical_analysis"]

    # Round-trip the actual training replay, then verify the next PER draw uses the same
    # physical slots, weights, and random stream. Restoring replay must not restore/reseed RNG.
    numpy_state = np.random.get_state()
    restored = CircularBuffer.from_state_dict(buffer.state_dict())
    assert freeze_state(np.random.get_state()) == freeze_state(numpy_state)
    original_sample = agent._sample_experience(
        buffer.to_dataframe(), agent.prioritized_replay_beta
    )
    after_sample = freeze_state(np.random.get_state())
    np.random.set_state(numpy_state)
    restored_sample = agent._sample_experience(
        restored.to_dataframe(), agent.prioritized_replay_beta
    )
    pd.testing.assert_frame_equal(original_sample[0], restored_sample[0], check_exact=True)
    assert array_state(original_sample[1]) == array_state(restored_sample[1])
    assert array_state(original_sample[2]) == array_state(restored_sample[2])
    assert freeze_state(np.random.get_state()) == after_sample


@pytest.mark.parametrize("successful_fit", [False, True])
def test_train_retains_state_but_replaces_it_on_each_call(
    tmp_path, monkeypatch, environment_factory, successful_fit
):
    env = environment_factory()

    def fit(circuit, components_number):
        result = controlled_fit(circuit, components_number)
        result["terminated"] = successful_fit
        return result

    monkeypatch.setattr(env, "_evaluate_circuit", fit)
    agent = agent_module.DDQN_ECM(
        training_env=env,
        save_dir=tmp_path,
        num_trials=1,
        episodes_trial=1,
        action_cap=2,
        hidden_layers=[{"type": "Dense", "units": 4, "activation": "relu"}],
        buffer_capacity=3,
        batch_size=2,
        NN_sleep=100,
    )
    monkeypatch.setattr(agent.model, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_plot_training_metrics", lambda *args: None)
    # Always make a valid mutation so both success and action-cap termination run.
    monkeypatch.setattr(
        agent, "eps_greedy", lambda _: (3, "L" if env.state[3] == "P" else "P")
    )

    assert agent._replay_buffer is None
    assert agent._training_state is None
    first_results = agent.train()
    first_state = agent._training_state
    first = agent._replay_buffer
    steps = 1 if successful_fit else 2
    assert isinstance(first, CircularBuffer)
    assert (first.capacity, first.size, first.position) == (3, steps, steps)
    expected_progress = (steps, steps, steps, int(successful_fit), 1)
    assert training_progress(first_state) == expected_progress
    assert first_results["success_rate"]["success_rate"].tolist() == [float(successful_fit)]
    # Preserve the known reporting bug: _run_episode does not propagate the new
    # terminal DataFrame. Correcting it belongs in a separate reporting change.
    assert first_results["terminal_states"].empty
    history_snapshots = {
        name: value.copy(deep=True)
        for name, value in first_results.items()
        if isinstance(value, pd.DataFrame)
    }
    reward_snapshot = {
        name: values.copy() for name, values in first_state.statistical_analysis.items()
    }
    first_contents = array_state(first.buffer)

    second_results = agent.train()
    second_state = agent._training_state
    assert second_state is not first_state
    assert training_progress(second_state) == expected_progress
    assert training_progress(first_state) == expected_progress
    for name, value in first_results.items():
        assert getattr(first_state, name) is value
        assert getattr(second_state, name) is second_results[name]
        assert second_results[name] is not value
    for name, snapshot in history_snapshots.items():
        pd.testing.assert_frame_equal(first_results[name], snapshot, check_exact=True)
    assert first_state.statistical_analysis == reward_snapshot
    assert len(second_state.statistical_analysis["episodic_cumul_reward"]) == 1
    # These histories were already agent-owned and must keep their legacy append behavior.
    assert agent.epsilon_list == [1.0, 1.0]
    assert agent.beta_list == [agent.initial_beta, agent.initial_beta]
    second = agent._replay_buffer
    assert isinstance(second, CircularBuffer)
    assert second is not first
    assert (second.capacity, second.size, second.position) == (3, steps, steps)
    assert second.buffer.dtype == first.buffer.dtype
    assert array_state(first.buffer) == first_contents
    assert agent.optimizer.iterations.numpy() == 0


def test_prioritized_sampling_has_known_indices_weights_and_rng_consumption():
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.batch_size = 2
    agent.prioritized_replay_alpha = 0.5
    history = pd.DataFrame({"priority": np.array([1, 4, 16, 64], dtype=np.float32)})
    np.random.seed(7)

    samples, indices, weights = agent._sample_experience(history, prioritized_replay_beta=1)

    # sqrt(priority) gives relative probabilities 1:2:4:8. Seed 7 selects slots 1,3;
    # normalized inverse-probability weights are therefore exactly 1 and 1/4.
    np.testing.assert_array_equal(indices, [1, 3])
    pd.testing.assert_frame_equal(samples, history.iloc[[1, 3]], check_exact=True)
    assert array_state(weights) == array_state(np.array([1.0, 0.25], dtype=np.float32))
    assert np.random.rand() == 0.4384092314408935


def test_ddqn_update_matches_hand_calculated_targets_loss_and_priorities():
    agent = object.__new__(agent_module.DDQN_ECM)
    agent.batch_size = 2
    agent.gamma = 0.5
    agent.prioritized_replay_alpha = 0.0
    agent.prioritized_replay_beta = 1.0
    agent.prioritized_replay_eps = 0.125
    agent._active_env = type(
        "ReplayEnv",
        (),
        {
            "dataset": pd.DataFrame({"flatten_Z": [np.zeros(1)]}),
            "ACTIONS_LIST": pd.DataFrame(
                {"action_type": ["R", "L"], "action_position": [0, 0]}
            ),
        },
    )()
    history = pd.DataFrame(
        {
            "EIS": [0, 0],
            "encoded_state": [np.zeros(1), np.zeros(1)],
            "encoded_new_state": [np.zeros(1), np.zeros(1)],
            "action_type": ["L", "R"],
            "action_position": [0, 0],
            "reward": [0.0, 1.0],
            "terminal_flag": [0, 1],
            "priority": np.ones(2, dtype=np.float32),
        }
    )
    with agent_module.tf.device("/CPU:0"):
        agent.model = agent_module.tf.keras.Sequential(
            [
                agent_module.tf.keras.layers.Input(shape=(2,)),
                agent_module.tf.keras.layers.Dense(2, kernel_initializer="zeros"),
            ]
        )
        agent.target_model = agent_module.tf.keras.models.clone_model(agent.model)
        agent.model.set_weights([np.zeros((2, 2)), np.array([1.0, 0.0])])
        agent.target_model.set_weights([np.zeros((2, 2)), np.array([2.0, 4.0])])
        agent.optimizer = agent_module.tf.keras.optimizers.SGD(learning_rate=0.25)
        np.random.seed(7)
        loss = agent._train_model(gradient_steps=1, history=history)

    # Online argmax is R, although the target's maximum is L: DDQN must use Q_target(R).
    # Targets are [0 + .5*2, 1] = [1,1]; selected predictions are [0,1].
    # Mean squared loss is .5, SGD changes only bias L to .25, and priorities use
    # POST-update residuals plus .125: [.75 + .125, 0 + .125].
    assert loss == np.float32(0.5)
    np.testing.assert_array_equal(agent.model.get_weights()[0], np.zeros((2, 2)))
    np.testing.assert_array_equal(agent.model.get_weights()[1], [1.0, 0.25])
    np.testing.assert_array_equal(agent.target_model.get_weights()[1], [2.0, 4.0])
    np.testing.assert_array_equal(history["priority"].to_numpy(), [0.875, 0.125])
    assert agent.optimizer.iterations.numpy() == 1


@pytest.mark.parametrize("cache_type", ["lru", "clock"])
@pytest.mark.parametrize("roundtrip_cache", [False, True])
def test_environment_cache_hit_reuses_reward_without_consuming_fit_randomness(
    monkeypatch, environment_factory, cache_type, roundtrip_cache
):
    env = environment_factory(cache_type)
    monkeypatch.setattr(env, "_evaluate_circuit", controlled_fit)
    env._update_state(new_state="+/RPR")
    reference_rng = np.random.RandomState()
    reference_rng.set_state(np.random.get_state())

    first = env.step(action_type="L", action_position=3)
    assert first["validity"] == 1
    assert first["reward"] == reference_rng.uniform(0.125, 0.25)
    if roundtrip_cache:
        env.cache = HistoryCache.from_state_dict(env.cache.state_dict())
    env._update_state(new_state="+/RPR")
    cached = env.step(action_type="L", action_position=3)
    assert cached["reward"] == first["reward"]
    assert env.get_cache_stats()["hits"] == 1
    assert env.get_cache_stats()["misses"] == 1

    # Repeating the mutation without resetting is invalid and must not fit either.
    invalid = env.step(action_type="L", action_position=3)
    assert invalid["validity"] == 0
    assert invalid["reward"] == -0.5
    assert env.get_cache_stats()["hits"] == 1
    assert env.get_cache_stats()["misses"] == 1
    assert np.random.rand() == reference_rng.rand()
