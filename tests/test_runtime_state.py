"""Prepared environments and global random streams must round-trip without resets."""

import pickle
import random

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.ops import stateful_random_ops

from test_training_baseline import environment_factory as _environment_factory
from autorec.environment import EIS_ECM_Env
from autorec.runtime_state import (
    environment_state_dict,
    environment_source_modules,
    restore_environment_state,
    restore_rng_state,
    rng_state_dict,
)


environment_factory = _environment_factory


class CheckpointEnvironment(EIS_ECM_Env):
    CHECKPOINT_FIELDS = frozenset({"reward_scale"})
    CHECKPOINT_SOURCE_MODULES = ("autorec.parser",)

    def __init__(self, *args, **kwargs):
        raise AssertionError("Checkpoint restoration must not call environment constructors.")


class UndeclaredCheckpointEnvironment(EIS_ECM_Env):
    pass


class MissingSourceDeclarationEnvironment(EIS_ECM_Env):
    CHECKPOINT_FIELDS = frozenset()


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


@pytest.mark.parametrize(
    ("cache_type", "separate"),
    [("lru", False), ("clock", True)],
    ids=["shared-lru", "distinct-clock"],
)
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


def test_distinct_environments_sharing_cache_are_rejected(environment_factory):
    training, evaluation = environment_factory(), environment_factory()
    evaluation.cache = training.cache
    with pytest.raises(ValueError, match="share a mutable circuit cache"):
        environment_state_dict(training, evaluation, training)


def test_disabled_cache_restores_without_enabling_it(environment_factory):
    env = environment_factory()
    env.cache_enabled, env.cache = False, None
    restored, evaluation, active = restore_environment_state(
        environment_state_dict(env, env, env)
    )
    assert restored is evaluation is active
    assert restored.cache is None
    assert restored.cache_enabled is False


def test_declared_environment_subclass_state_roundtrips_without_constructor(
    environment_factory,
):
    env = environment_factory()
    env.__class__ = CheckpointEnvironment
    env.reward_scale = 2.5

    restored, evaluation, active = restore_environment_state(
        environment_state_dict(env, env, env)
    )

    assert restored is evaluation is active
    assert type(restored) is CheckpointEnvironment
    assert restored.reward_scale == 2.5
    assert environment_source_modules(env) == ("autorec.parser", __name__)


def test_environment_subclass_requires_declared_complete_state(environment_factory):
    env = environment_factory()
    env.__class__ = UndeclaredCheckpointEnvironment
    with pytest.raises(ValueError, match="declare CHECKPOINT_FIELDS"):
        environment_state_dict(env, env, env)

    env.__class__ = CheckpointEnvironment
    env.reward_scale = 2.5
    env.undeclared = True
    with pytest.raises(ValueError, match="incomplete or undeclared"):
        environment_state_dict(env, env, env)

    del env.undeclared
    env.__class__ = MissingSourceDeclarationEnvironment
    del env.reward_scale
    with pytest.raises(ValueError, match="declare CHECKPOINT_SOURCE_MODULES"):
        environment_source_modules(env)
