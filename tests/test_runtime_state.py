"""Prepared environments and global random streams must round-trip without resets."""

import pickle
import random

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.ops import stateful_random_ops

from test_training_baseline import environment_factory as _environment_factory
from autorec.runtime_state import (
    environment_state_dict,
    restore_environment_state,
    restore_rng_state,
    rng_state_dict,
)


environment_factory = _environment_factory


@pytest.fixture(autouse=True)
def preserve_rng():
    saved = rng_state_dict()
    yield
    restore_rng_state(saved)


@pytest.mark.parametrize("generator", [False, True])
def test_rng_exact_next_draws_and_capture_is_inert(generator):
    random.seed(123)
    np.random.seed(456)
    random.gauss(0, 1)
    np.random.normal()  # Leave cached Gaussian samples in both streams.
    stateful_random_ops.global_generator = (
        tf.random.Generator.from_seed(789) if generator else None
    )
    state = rng_state_dict()
    assert pickle.dumps(state) == pickle.dumps(rng_state_dict())

    def draw():
        values = (random.random(), random.gauss(0, 1), np.random.normal(size=5).tobytes())
        if generator:
            values += (tf.random.get_global_generator().normal([5]).numpy().tobytes(),)
        return values

    expected = draw()
    restore_rng_state(state)
    assert draw() == expected


@pytest.mark.parametrize("cache_type", ["lru", "clock"])
@pytest.mark.parametrize("separate", [False, True])
def test_environment_roundtrip_preserves_graph_and_next_reset(
    environment_factory, cache_type, separate
):
    training = environment_factory(cache_type)
    evaluation = environment_factory(cache_type) if separate else training
    if separate:
        evaluation.dataset = training.dataset
    before = rng_state_dict()
    state = environment_state_dict(training, evaluation, evaluation)
    assert pickle.dumps(before) == pickle.dumps(rng_state_dict())
    restored, other, active = restore_environment_state(state)
    assert (restored is other) == (not separate)
    assert active is other
    assert restored.dataset is other.dataset
    assert restored.dataset is not training.dataset
    assert restored.dataset.iloc[0]["Z_true"] is not training.dataset.iloc[0]["Z_true"]
    assert pickle.dumps(before) == pickle.dumps(rng_state_dict())
    training.reset()
    expected = (training.state, training.current_eis_index, training.encoded_state.tobytes())
    restore_rng_state(before)
    restored.reset()
    assert (
        restored.state,
        restored.current_eis_index,
        restored.encoded_state.tobytes(),
    ) == expected


def test_invalid_rng_does_not_partially_restore():
    before = rng_state_dict()
    invalid = dict(before, numpy=("bad",))
    with pytest.raises((ValueError, IndexError)):
        restore_rng_state(invalid)
    assert pickle.dumps(before) == pickle.dumps(rng_state_dict())
