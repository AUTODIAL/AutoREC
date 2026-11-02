"""Tests for quick and full evaluation selection."""

import pytest

from .evaluation import DEFAULT_RUN_MODE, QUICK_SAMPLE_SIZE, run_evaluation


class RecordingAgent:
    def __init__(self):
        self.calls = []

    def eval_batch_eis(self, **kwargs):
        self.calls.append(("batch", kwargs))
        return "sample results"

    def eval_all_eis(self, **kwargs):
        self.calls.append(("all", kwargs))
        return "all results"


def test_full_evaluation_is_the_default():
    assert DEFAULT_RUN_MODE == "full"


def test_full_mode_evaluates_every_training_spectrum():
    agent = RecordingAgent()

    results, suffix = run_evaluation(agent, "full", action_cap=24, dataset_size=123)

    assert results == "all results"
    assert suffix == "all"
    assert agent.calls == [
        (
            "all",
            {"max_actions": 24, "verbose": False, "use_eval_env": False},
        )
    ]


def test_quick_mode_evaluates_a_random_sample():
    agent = RecordingAgent()

    results, suffix = run_evaluation(agent, "quick", action_cap=None, dataset_size=8)

    assert results == "sample results"
    assert suffix == "sample"
    assert agent.calls == [
        (
            "batch",
            {
                "num_samples": QUICK_SAMPLE_SIZE,
                "max_actions": None,
                "verbose": False,
                "use_eval_env": False,
            },
        )
    ]


def test_unknown_evaluation_mode_is_rejected():
    with pytest.raises(ValueError, match="Invalid run mode"):
        run_evaluation(RecordingAgent(), "unknown", action_cap=24, dataset_size=10)
