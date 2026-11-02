"""Focused tests for the continuous simplified-circuit FIM environment."""

import numpy as np
import pytest

from autorec.environment import EIS_ECM_Env as BaseEISECMEnv

from . import environment as experiment_environment
from .environment import continuous_identifiability_penalty
from .fim_utils import compute_fim


def test_experimental_environment_inherits_from_autorec_environment():
    assert issubclass(experiment_environment.EIS_ECM_Env, BaseEISECMEnv)


def test_single_resistor_fim_has_one_normalized_identifiable_direction():
    frequency = np.array([1.0, 10.0, 100.0])
    impedance = np.full(frequency.shape, 5.0 + 0.0j)

    fim, eigenvalues, eigenvectors = compute_fim(
        "R0",
        frequency,
        impedance,
        np.array([5.0]),
    )

    assert fim.shape == (1, 1)
    assert eigenvectors.shape == (1, 1)
    np.testing.assert_allclose(eigenvalues, np.array([1.0]))


def test_continuous_penalty_matches_soft_unidentifiability_definition():
    eigenvalues = np.array([1.0, 1e-6, 1e-8])

    penalty = continuous_identifiability_penalty(
        eigenvalues,
        threshold=1e-6,
        sharpness=2.0,
        scale=2.0,
    )

    expected_contributions = 1.0 / (1.0 + (eigenvalues / 1e-6) ** 2)
    assert penalty == pytest.approx(2.0 * np.mean(expected_contributions))


def test_continuous_penalty_is_dimension_normalized_and_bounded():
    one_direction = continuous_identifiability_penalty(
        np.array([1e-6]),
        threshold=1e-6,
        sharpness=2.0,
        scale=2.0,
    )
    repeated_direction = continuous_identifiability_penalty(
        np.full(5, 1e-6),
        threshold=1e-6,
        sharpness=2.0,
        scale=2.0,
    )

    assert one_direction == pytest.approx(1.0)
    assert repeated_direction == pytest.approx(one_direction)
    assert 0.0 <= repeated_direction <= 2.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0.0, "sharpness": 2.0, "scale": 2.0}, "threshold"),
        ({"threshold": 1e-6, "sharpness": 0.0, "scale": 2.0}, "sharpness"),
        ({"threshold": 1e-6, "sharpness": 2.0, "scale": -1.0}, "scale"),
    ],
)
def test_continuous_penalty_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        continuous_identifiability_penalty(np.array([1.0]), **kwargs)


def test_fim_penalty_simplifies_redundant_series_resistors():
    env = object.__new__(experiment_environment.EIS_ECM_Env)
    env.identifiability_thresh = 1e-6
    env.penalty_sharpness = 2.0
    env.penalty_scale = 2.0
    env.freq = np.array([1.0, 10.0, 100.0])
    env.Z = np.full(env.freq.shape, 3.0 + 0.0j)

    penalty, simplified = env._fim_complexity_penalty(
        "R0-R1",
        {"R0": 1.0, "R1": 2.0},
    )

    assert simplified == "R0"
    assert penalty == pytest.approx(
        continuous_identifiability_penalty(
            np.array([1.0]),
            threshold=1e-6,
            sharpness=2.0,
            scale=2.0,
        )
    )


def test_fim_penalty_uses_simplified_circuit_and_parameters(monkeypatch):
    captured = {}

    def fake_simplify(circuit, parameters):
        captured["simplify"] = (circuit, parameters)
        return "simplified", {"R0": 0.5, "P1w": 3.0, "P1n": 0.8}

    def fake_compute_fim(circuit, frequency, impedance, parameters):
        captured["fim"] = (circuit, frequency, impedance, parameters)
        return np.eye(3), np.array([1e-6, 1e-3, 1.0]), np.eye(3)

    monkeypatch.setattr(experiment_environment, "simplify", fake_simplify)
    monkeypatch.setattr(experiment_environment, "compute_fim", fake_compute_fim)
    monkeypatch.setattr(
        experiment_environment.ae.parser,
        "get_parameter_labels",
        lambda circuit: ["R0", "P1w", "P1n"],
    )
    monkeypatch.setattr(
        experiment_environment.ae.parser,
        "find_ohmic_resistors",
        lambda circuit: ["R0"],
    )

    env = object.__new__(experiment_environment.EIS_ECM_Env)
    env.identifiability_thresh = 1e-6
    env.penalty_sharpness = 2.0
    env.penalty_scale = 2.0
    env.freq = np.array([1.0, 10.0])
    env.Z = np.array([100 + 1j, 50 + 2j])

    penalty, simplified = env._fim_complexity_penalty("raw", {"R0": 0.5})

    assert penalty == pytest.approx(
        continuous_identifiability_penalty(
            np.array([1e-6, 1e-3, 1.0]),
            threshold=1e-6,
            sharpness=2.0,
            scale=2.0,
        )
    )
    assert simplified == "simplified"
    assert captured["simplify"] == ("raw", {"R0": 0.5})
    assert captured["fim"][0] == "simplified"
    np.testing.assert_array_equal(captured["fim"][1], env.freq)
    np.testing.assert_array_equal(captured["fim"][2], env.Z)
    np.testing.assert_allclose(captured["fim"][3], np.array([1.5, 3.0, 0.8]))


def test_successful_reward_uses_fim_penalty_and_simplified_depth(monkeypatch):
    monkeypatch.setattr(
        experiment_environment.EIS_ECM_Env,
        "_fim_complexity_penalty",
        lambda self, circuit, parameters: (2, "simplified"),
    )
    monkeypatch.setattr(
        experiment_environment.ae.parser,
        "circuit_to_nested_expr",
        lambda circuit: [[[["R"]]]],
    )

    env = object.__new__(experiment_environment.EIS_ECM_Env)
    reward, depth_penalty = env._calculate_fim_reward(
        "raw",
        {"R0": 1.0},
        {
            "chi_square": 0.01,
            "r2_score": 0.9,
            "r2_mag": 0.9,
            "r2_phase": 0.9,
        },
    )

    assert reward == pytest.approx(0.8)
    assert depth_penalty == 0.2


def test_nonterminal_base_result_skips_fim_calculation(monkeypatch):
    base_result = {
        "reward": 0.01,
        "metrics": {"chi_square": 1.0},
        "predicted_Z": None,
        "param": {},
        "terminated": False,
        "depth_penalty": 0,
    }
    monkeypatch.setattr(
        BaseEISECMEnv,
        "_evaluate_circuit",
        lambda self, circuit, components_number: base_result,
    )

    env = object.__new__(experiment_environment.EIS_ECM_Env)
    assert env._evaluate_circuit("raw", 4) is base_result


def test_terminal_base_result_is_replaced_with_fim_reward(monkeypatch):
    base_result = {
        "reward": 2.5,
        "metrics": {"chi_square": 0.01},
        "predicted_Z": np.array([1 + 1j]),
        "param": {"R0": 1.0},
        "terminated": True,
        "depth_penalty": 0,
    }
    monkeypatch.setattr(
        BaseEISECMEnv,
        "_evaluate_circuit",
        lambda self, circuit, components_number: base_result.copy(),
    )
    monkeypatch.setattr(
        experiment_environment.EIS_ECM_Env,
        "_calculate_fim_reward",
        lambda self, circuit, parameters, metrics: (0.8, 0.2),
    )

    env = object.__new__(experiment_environment.EIS_ECM_Env)
    result = env._evaluate_circuit("raw", 4)

    assert result["reward"] == 0.8
    assert result["depth_penalty"] == 0.2
    assert result["terminated"] is True
    np.testing.assert_array_equal(result["predicted_Z"], np.array([1 + 1j]))
