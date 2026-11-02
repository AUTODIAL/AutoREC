"""Focused tests for the simplified-circuit complexity reward experiment."""

import pytest

from autorec.environment import EIS_ECM_Env as BaseEISECMEnv

from . import environment as experiment_environment


def test_experimental_environment_inherits_from_autorec_environment():
    assert issubclass(experiment_environment.EIS_ECM_Env, BaseEISECMEnv)


def test_good_fit_complexity_penalties_use_simplified_circuit(monkeypatch):
    calls = []

    def fake_simplify(circuit):
        calls.append(("simplify", circuit))
        return "simplified"

    def fake_get_component_types(circuit):
        calls.append(("components", circuit))
        return ["R", "R", "P", "L"]

    def fake_circuit_to_nested_expr(circuit):
        calls.append(("depth", circuit))
        return [[[["R"]]]]

    monkeypatch.setattr(experiment_environment, "simplify", fake_simplify)
    monkeypatch.setattr(
        experiment_environment.ae.parser,
        "get_component_types",
        fake_get_component_types,
    )
    monkeypatch.setattr(
        experiment_environment.ae.parser,
        "circuit_to_nested_expr",
        fake_circuit_to_nested_expr,
    )

    env = object.__new__(experiment_environment.EIS_ECM_Env)
    env.chi_threshold = 0.1

    reward, terminated, depth_penalty = env._calculate_reward(
        r2_score=0.9,
        r2_mag=0.9,
        r2_phase=0.9,
        chi_square=0.01,
        circuit="raw",
        components_count=99,
    )

    # Fit terms total 3.0; four simplified components cost 0.5 and depth costs 0.2.
    assert reward == pytest.approx(2.3)
    assert terminated is True
    assert depth_penalty == 0.2
    assert calls == [
        ("simplify", "raw"),
        ("components", "simplified"),
        ("depth", "simplified"),
    ]


def test_nonterminal_reward_does_not_need_circuit_simplification(monkeypatch):
    def fail_if_called(_circuit):
        raise AssertionError("simplify should only be needed for a successful fit")

    monkeypatch.setattr(experiment_environment, "simplify", fail_if_called)

    env = object.__new__(experiment_environment.EIS_ECM_Env)
    env.chi_threshold = 0.1

    assert env._calculate_reward(0.6, 0.7, 0.8, 1.0, "raw", 99) == (0.01, False, 0)
