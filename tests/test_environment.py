"""Unit tests for EIS_ECM_Env setup and lightweight helper behavior.

The expensive circuit parsing/encoding paths are monkeypatched so these tests can focus on
state management, dataset wiring, and reward branches without fitting circuits.
"""

import numpy as np
import pandas as pd
import pytest

from autorec import environment


class FakeEC:
    """Small AutoEIS circuit helper stand-in used by environment reset/observation code."""

    @staticmethod
    def karva_to_tree(karva):
        return list(karva)

    @staticmethod
    def tree_to_circuit(tree):
        return ("R0", None)


def make_dataset(index=0):
    """Return the smallest processed dataset shape accepted by EIS_ECM_Env."""
    return pd.DataFrame(
        {
            "Z_true": [np.array([1 + 1j, 2 + 2j])],
            "freq": [np.array([1.0, 10.0])],
            "flatten_Z": [np.array([1.0, 2.0, 3.0, 4.0])],
            "true_circuit": ["R0"],
            "chi_thresh": [0.1],
            "r2_thresh": [0.9],
            "metadata": [{"path": "raw-data", "mode": "process"}],
        },
        index=[index],
    )


@pytest.fixture
def patched_environment(monkeypatch):
    """Patch AutoEIS-dependent helpers with deterministic local replacements."""
    monkeypatch.setattr(environment, "ec", FakeEC)
    monkeypatch.setattr(
        environment,
        "state_encode",
        lambda state, elements: np.array([elements.index(char) for char in state]),
    )
    return environment


def test_environment_requires_dataset(patched_environment):
    with pytest.raises(ValueError, match="dataset must be provided"):
        patched_environment.EIS_ECM_Env(dataset=None)


def test_environment_initializes_dimensions_and_actions(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    assert env.EIS_measurement_size == 2
    assert env.EIS_INPUT_SIZE == 4
    assert env.chromosome_HEAD_len == 1
    assert env.chromosome_TAIL_len == 2
    assert env.chromosome_len == 3
    assert list(env.ACTIONS_LIST.columns) == ["action_type", "action_position"]
    # With HEAD=1 and TAIL=2, actions include root operators and terminal tail elements.
    assert set(env.ACTIONS_LIST["action_type"]) == {"+", "/", "R", "L", "P"}


def test_environment_uses_configured_elements_for_actions(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        elements=["+", "/", "R", "P"],
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    assert env.ELEMENTS == ["+", "/", "R", "P"]
    assert env.ELEMENTS_EXTENDED == ["+", "/", "R", "P", "X"]
    root_actions = env.ACTIONS_LIST.loc[
        env.ACTIONS_LIST["action_position"] == 0, "action_type"
    ]
    tail_actions = env.ACTIONS_LIST.loc[env.ACTIONS_LIST["action_position"] > 0, "action_type"]
    assert set(root_actions) == {"+", "/"}
    assert set(tail_actions) == {"R", "P"}


def test_environment_rejects_invalid_elements_and_initial_state(patched_environment):
    with pytest.raises(ValueError, match="Invalid element 'C'"):
        patched_environment.EIS_ECM_Env(dataset=make_dataset(), elements=["+", "R", "C"])

    with pytest.raises(ValueError, match="initial_state"):
        patched_environment.EIS_ECM_Env(
            dataset=make_dataset(), elements=["+", "/", "R"], initial_state=["+LR"]
        )


@pytest.mark.parametrize(
    ("elements", "initial_state"),
    [(["+", "R"], ["+RR"]), (["/", "R"], ["/RR"])],
)
def test_environment_accepts_either_topology_operator(
    patched_environment, elements, initial_state
):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        elements=elements,
        initial_state=initial_state,
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    assert env._operator == [elements[0]]


def test_environment_requires_at_least_one_topology_operator(patched_environment):
    with pytest.raises(ValueError, match="at least one operator"):
        patched_environment.EIS_ECM_Env(
            dataset=make_dataset(), elements=["R", "P"], initial_state=["RRR"]
        )


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_environment_metadata_records_constructor_settings(patched_environment, cache_enabled):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(index=7),
        elements=["+", "R", "P"],
        initial_state=["+RR"],
        seed=7,
        chromosome_HEAD_len=1,
        cache_enabled=cache_enabled,
        cache_capacity=123,
        cache_type="clock",
    )

    assert env.metadata == {
        "dataset": {"path": "raw-data", "mode": "process"},
        "elements": ["+", "R", "P"],
        "initial_state": ["+RR"],
        "seed": 7,
        "chromosome_HEAD_len": 1,
        "cache_enabled": cache_enabled,
        "cache_capacity": 123,
        "cache_type": "clock",
    }


def test_reset_with_specific_eis_returns_observation(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    index, observation = env.reset(EIS_i=0)

    assert index == 0
    assert observation["state"] == "+RR"
    assert observation["circuit"] == "R0"
    np.testing.assert_array_equal(observation["Z_true"], np.array([1 + 1j, 2 + 2j]))


def test_update_state_rejects_ambiguous_inputs(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    with pytest.raises(ValueError, match="Cannot specify both"):
        env._update_state(action_type="R", action_position=1, new_state="+LR")

    with pytest.raises(ValueError, match="Must provide either"):
        env._update_state()


def test_calculate_reward_for_failed_and_moderate_fits(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    # These branches avoid the good-fit path, which depends on AutoEIS nested expressions.
    assert env._calculate_reward(0, 0, 0, 1.0, "R0", 1) == (-0.05, False, 0)
    assert env._calculate_reward(0.6, 0.7, 0.8, 1.0, "R0", 1) == (0.01, False, 0)
    assert env._calculate_reward(0.1, 0.2, 0.3, 1.0, "R0", 1) == (-0.01, False, 0)


def test_cache_helpers_return_none_when_cache_disabled(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    assert env.get_cache_stats() is None
    env.clear_cache()
