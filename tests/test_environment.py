"""Unit tests for EIS_ECM_Env setup and lightweight helper behavior.

The expensive circuit parsing/encoding paths are monkeypatched so these tests can focus on
state management, dataset wiring, and reward branches without fitting circuits.
"""

import numpy as np
import pandas as pd
import pytest

from autorec import environment
from autorec.data_preparation import EISDataPrep
from autorec.progressive_spectrum import (
    PREPARED_SPECTRUM_SOURCE,
    RETAINED_SPECTRUM_SOURCE,
    SpectrumView,
    SpectrumWindowSelection,
)


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


def make_progressive_dataset(index=0):
    """Return one row with distinct prepared and retained physical spectra."""
    dataset = make_dataset(index=index)
    dataset.at[index, "freq"] = np.array([1.0, 10.0, 100.0, 1_000.0])
    dataset.at[index, "Z_true"] = np.array([1 + 4j, 2 + 3j, 3 + 2j, 4 + 1j])
    dataset.at[index, "flatten_Z"] = np.linspace(0.0, 1.0, 8)
    dataset["source_freq"] = [np.array([1.0, 3.0, 10.0, 30.0, 100.0, 1_000.0])]
    dataset["source_Z"] = [np.array([1 + 6j, 2 + 5j, 3 + 4j, 4 + 3j, 5 + 2j, 6 + 1j])]
    return dataset


def make_spectrum_view(dataset, source_indices, observation, *, source=None):
    """Construct a progressive view tied to the first row of ``dataset``."""
    source = PREPARED_SPECTRUM_SOURCE if source is None else source
    source_frequency, source_impedance = source.read(dataset.iloc[0])
    source_indices = np.asarray(source_indices)
    selection = SpectrumWindowSelection(
        stage_index=0,
        included_window_indices=(0,),
        source_indices=source_indices,
        frequency=source_frequency[source_indices],
        impedance=source_impedance[source_indices],
        source=source,
    )
    return SpectrumView(
        selection=selection,
        frequency=selection.frequency,
        impedance=selection.impedance,
        observation=np.asarray(observation),
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


def test_environment_accepts_loaded_dataprep(patched_environment):
    dataset = make_dataset()
    dataprep = object.__new__(EISDataPrep)
    dataprep.dataset = dataset

    env = patched_environment.EIS_ECM_Env(
        dataprep=dataprep,
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    assert env.dataprep is dataprep
    assert env.dataset is dataset


def test_environment_initializes_dimensions_and_actions(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    assert env.EIS_measurement_size == 2
    assert env.EIS_INPUT_SIZE == 4
    assert env.dataprep is None
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


def test_reset_uses_raw_view_selection_and_fixed_observation(patched_environment):
    dataset = make_progressive_dataset()
    view = make_spectrum_view(dataset, [3, 2], np.linspace(0.0, 1.0, 8))
    env = patched_environment.EIS_ECM_Env(
        dataset=dataset,
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    _, observation = env.reset(EIS_i=0, spectrum_view=view)

    np.testing.assert_array_equal(observation["freq"], view.selection.frequency)
    np.testing.assert_array_equal(observation["Z_true"], view.selection.impedance)
    np.testing.assert_array_equal(observation["EIS_flatten"], view.observation)
    assert env.current_spectrum_view_id is not None


def test_retained_view_routes_all_selected_raw_points_to_fitting(patched_environment):
    dataset = make_progressive_dataset()
    view = make_spectrum_view(
        dataset,
        [5, 4, 3],
        np.linspace(0.0, 1.0, 8),
        source=RETAINED_SPECTRUM_SOURCE,
    )
    env = patched_environment.EIS_ECM_Env(
        dataset=dataset,
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    env.reset(EIS_i=0, spectrum_view=view)

    np.testing.assert_array_equal(env.freq, dataset.iloc[0]["source_freq"][[5, 4, 3]])
    np.testing.assert_array_equal(env.Z, dataset.iloc[0]["source_Z"][[5, 4, 3]])


def test_reset_accepts_and_validates_explicit_chromosome(patched_environment):
    env = patched_environment.EIS_ECM_Env(
        dataset=make_dataset(),
        initial_state=["+RR"],
        chromosome_HEAD_len=1,
        cache_enabled=False,
    )

    _, observation = env.reset(EIS_i=0, initial_state="+PR")
    assert observation["state"] == "+PR"
    with pytest.raises(ValueError, match="tail positions"):
        env.reset(EIS_i=0, initial_state="++R")


def test_spectrum_cache_identity_depends_on_raw_fitting_values(patched_environment):
    dataset = make_progressive_dataset()
    first = make_spectrum_view(dataset, [3, 2], np.zeros(8))
    same_physical_data = make_spectrum_view(dataset, [3, 2], np.ones(8))
    different_physical_data = make_spectrum_view(dataset, [3, 1], np.ones(8))

    assert patched_environment.EIS_ECM_Env._spectrum_view_id(
        first
    ) == patched_environment.EIS_ECM_Env._spectrum_view_id(same_physical_data)
    assert patched_environment.EIS_ECM_Env._spectrum_view_id(
        first
    ) != patched_environment.EIS_ECM_Env._spectrum_view_id(different_physical_data)


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
