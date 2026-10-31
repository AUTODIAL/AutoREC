"""Exact cache snapshots, subsequent eviction behavior, and malformed-state rejection."""

from copy import deepcopy
import pickle
import random

import numpy as np
import pytest

from autorec.optimized_data_structures.circuit_cache import ClockCache, HistoryCache, LRUCache


def frozen(value):
    """Compare containers in order, preserving array/scalar types and numerical bits."""
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        return type(value), array.dtype, array.shape, array.tobytes()
    if isinstance(value, dict):
        return type(value), tuple((frozen(key), frozen(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(frozen(item) for item in value)
    if isinstance(value, (float, complex)):
        return type(value), np.asarray(value).tobytes()
    return type(value), value


def payload():
    return {
        "reward": np.float32(-0.0),
        "metrics": {"chi_square": np.float64(0.125), "r2_score": 0.75},
        "predicted_Z": np.array([1 + 2j, 3 - 4j], dtype=np.complex128),
        "param": {"R0": np.float32(2.5), "nested": [np.array([1.0, -0.0])]},
        "good_fit": np.bool_(False),
        "depth_penalty": 0.0,
        "fit_bonus": 0.25,
        "fit_penalty": -0.5,
    }


@pytest.mark.parametrize("cache_class", [LRUCache, ClockCache])
@pytest.mark.parametrize("phase", ["empty", "evicted", "cleared"])
def test_roundtrip_preserves_state_and_future_operations(cache_class, phase):
    # Partial/full snapshots are exercised by the ownership and policy-specific tests.
    cache = cache_class(3)
    count = 0 if phase == "empty" else 4
    for ii in range(count):
        cache.put((str(ii),), payload())
    cache.get(("missing",))
    if count:
        cache.get((str(count - 1),))
    if phase == "cleared":
        cache.clear()
        assert (cache.hits, cache.misses) == (1, 1)  # clear() preserves statistics.

    before = frozen(vars(cache))
    snapshot = cache.state_dict()
    assert frozen(vars(cache)) == before
    # Exercise a trusted state-dictionary wire round trip, not whole-object pickling.
    restored = cache_class.from_state_dict(pickle.loads(pickle.dumps(snapshot)))
    assert frozen(vars(restored)) == before
    assert restored.get_stats() == cache.get_stats()

    for ii in range(6):
        key = (str(ii),)
        assert frozen(restored.get(key)) == frozen(cache.get(key))
        cache.put(key, payload())
        restored.put(key, payload())
        assert frozen(vars(restored)) == frozen(vars(cache))
    cache.clear()
    restored.clear()
    assert frozen(vars(restored)) == frozen(vars(cache))


def test_lru_restores_recency_after_reads_and_updates():
    cache = LRUCache(3)
    for key in ("a", "b", "c"):
        cache.put((key,), {})
    cache.get(("a",))
    cache.put(("b",), {"updated": True})
    assert list(cache.cache) == [("c",), ("a",), ("b",)]
    restored = LRUCache.from_state_dict(cache.state_dict())
    restored.put(("d",), {})
    assert list(restored.cache) == [("a",), ("b",), ("d",)]
    assert (restored.hits, restored.misses) == (1, 0)


def test_clock_restores_distinct_orders_bits_and_next_victim():
    cache = ClockCache(3)
    for key in ("a", "b", "c", "d"):
        cache.put((key,), {})
    cache.get(("b",))
    assert list(cache.cache) == [("b",), ("c",), ("d",)]
    assert cache.keys_list == [("d",), ("b",), ("c",)]
    assert cache.reference_bits == {("b",): True, ("c",): False, ("d",): True}
    assert cache.hand == 1
    restored = ClockCache.from_state_dict(cache.state_dict())
    restored.put(("e",), {})
    assert ("c",) not in restored.cache  # b gets its second chance; c is the victim.
    assert restored.keys_list == [("d",), ("b",), ("e",)]
    assert list(restored.cache) == [("b",), ("d",), ("e",)]
    assert restored.hand == 0
    assert restored.reference_bits == {("b",): False, ("d",): True, ("e",): True}


@pytest.mark.parametrize("cache_class", [LRUCache, ClockCache])
def test_copy_is_independent_and_preserves_shared_payloads(cache_class):
    cache = cache_class(3)
    shared = payload()
    cache.put(("a",), shared)
    cache.put(("b",), shared)
    snapshot = cache.state_dict()
    restored = cache_class.from_state_dict(snapshot)
    copied = snapshot["entries"][0][1]
    assert copied is snapshot["entries"][1][1]
    assert restored.cache[("a",)] is restored.cache[("b",)]
    assert copied is not shared
    assert restored.cache[("a",)] is not copied
    shared["predicted_Z"][0] = 10
    copied["predicted_Z"][0] = 20
    restored.cache[("a",)]["predicted_Z"][0] = 30
    assert shared["predicted_Z"][0] == 10
    assert copied["predicted_Z"][0] == 20
    assert restored.cache[("a",)]["predicted_Z"][0] == 30
    copied["param"]["nested"][0][0] = 99
    assert shared["param"]["nested"][0][0] == 1
    assert restored.cache[("a",)]["param"]["nested"][0][0] == 1


@pytest.mark.parametrize("cache_type", ["lru", "clock"])
def test_history_cache_preserves_full_and_failed_results_without_rng_use(
    cache_type, monkeypatch
):
    cache = HistoryCache(3, cache_type)
    cache.put("RPR", np.int64(2), "P", np.int64(1), **payload())
    failure = payload()
    failure.update(reward=-0.05, metrics=None, predicted_Z=None, param={}, good_fit=False)
    cache.put("RLR", 3, "L", 1, **failure)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    before = frozen(vars(cache.cache))
    snapshot = cache.state_dict()
    snapshot_before = frozen(snapshot)
    # Loading must not replay gets/puts, which would change metadata or eviction state.
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("Restoration must not call get() or put()")

        patch.setattr(type(cache.cache), "get", forbidden)
        patch.setattr(type(cache.cache), "put", forbidden)
        restored = HistoryCache.from_state_dict(snapshot)
    assert frozen(snapshot) == snapshot_before
    assert frozen(vars(cache.cache)) == before == frozen(vars(restored.cache))
    assert restored.cache_type == cache_type
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state
    assert frozen(restored.get("RPR", 2, "P", 1)) == frozen(cache.get("RPR", 2, "P", 1))
    assert frozen(restored.get("RLR", 3, "L", 1)) == frozen(failure)


@pytest.mark.parametrize("cache_class", [LRUCache, ClockCache])
def test_cache_payload_preserves_cpu_jax_scalars(cache_class):
    import jax.numpy as jnp

    scalar = jnp.asarray(0.125, dtype=jnp.float32)
    cache = cache_class(1)
    cache.put(("a",), {"metric": scalar})
    numpy_state, python_state = np.random.get_state(), random.getstate()
    snapshot = cache.state_dict()
    restored = cache_class.from_state_dict(pickle.loads(pickle.dumps(snapshot)))
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state
    value = restored.cache[("a",)]["metric"]
    assert type(value) is type(scalar)
    assert value.dtype == scalar.dtype
    assert value.shape == scalar.shape
    assert np.asarray(value).tobytes() == np.asarray(scalar).tobytes()


def test_invalid_common_state_is_rejected():
    cases = [
        ({"version": 2}, ValueError, "version"),
        ({"version": True}, TypeError, "integer"),
        ({"capacity": 0}, ValueError, "capacity"),
        ({"capacity": 1.5}, TypeError, "integer"),
        ({"hits": -1}, ValueError, "nonnegative"),
        ({"misses": -1}, ValueError, "nonnegative"),
        ({"misses": False}, TypeError, "integer"),
        ({"entries": {}}, TypeError, "ordered list"),
        ({"entries": [(("a",), {})] * 4}, ValueError, "exceeds capacity"),
        ({"entries": [(("a",), {}), (("a",), {})]}, ValueError, "duplicate"),
        ({"entries": [(("a",),)]}, TypeError, "entry"),
        ({"entries": [("a", {})]}, TypeError, "hashable tuples"),
        ({"entries": [(([1],), {})]}, TypeError, "hashable tuples"),
        ({"entries": [(("a",), None)]}, TypeError, "payloads"),
    ]
    # Both policies call the same common validator. Clock-specific metadata is tested below.
    for updates, error, message in cases:
        state = LRUCache(3).state_dict()
        state.update(deepcopy(updates))
        before = frozen(state)
        with pytest.raises(error, match=message):
            LRUCache.from_state_dict(state)
        assert frozen(state) == before


@pytest.mark.parametrize("cache_class", [LRUCache, ClockCache])
def test_missing_fields_are_rejected(cache_class):
    state = cache_class(3).state_dict()
    del state["entries"]
    with pytest.raises(ValueError, match="missing=.*entries"):
        cache_class.from_state_dict(state)


@pytest.mark.parametrize("cache_class", [LRUCache, ClockCache])
def test_wrong_policy_extra_fields_and_nondict_are_rejected(cache_class):
    state = cache_class(3).state_dict()
    with pytest.raises(TypeError, match="dictionary"):
        cache_class.from_state_dict(None)
    with pytest.raises(ValueError, match="unexpected=.*extra"):
        cache_class.from_state_dict({**state, "extra": 0})
    with pytest.raises(ValueError, match="policy mismatch"):
        cache_class.from_state_dict({**state, "cache_type": "unknown"})
    with pytest.raises(TypeError, match="string"):
        cache_class.from_state_dict({**state, "cache_type": np.array("lru")})


def test_invalid_clock_state_is_rejected():
    cases = [
        ({"hand": True}, TypeError, "integer"),
        ({"hand": -1}, ValueError, "hand"),
        ({"hand": 3}, ValueError, "hand"),
        ({"hand": 1}, ValueError, "until the cache is full"),
        ({"keys_list": ()}, TypeError, "list"),
        ({"keys_list": []}, ValueError, "exactly once"),
        ({"keys_list": [("a",), ("a",)]}, ValueError, "exactly once"),
        ({"reference_bits": []}, TypeError, "dictionary"),
        ({"reference_bits": {}}, ValueError, "match entry keys"),
        ({"reference_bits": {("a",): 1}}, TypeError, "booleans"),
    ]
    for updates, error, message in cases:
        cache = ClockCache(3)
        cache.put(("a",), {})
        state = cache.state_dict()
        state.update(updates)
        with pytest.raises(error, match=message):
            ClockCache.from_state_dict(state)


def test_history_cache_rejects_bad_keys_missing_payload_and_policy_mismatch():
    cache = HistoryCache(3)
    cache.put("RPR", 0, "P", 1, **payload())
    state = cache.state_dict()
    missing = deepcopy(state)
    del missing["entries"][0][1]["reward"]
    with pytest.raises(ValueError, match="missing fields.*reward"):
        HistoryCache.from_state_dict(missing)
    short_key = deepcopy(state)
    short_key["entries"] = [(("a",), payload())]
    with pytest.raises(ValueError, match="two components"):
        HistoryCache.from_state_dict(short_key)
    with pytest.raises(ValueError, match="cache_type"):
        HistoryCache.from_state_dict({**state, "cache_type": "other"})
    with pytest.raises(TypeError, match="string"):
        HistoryCache.from_state_dict({**state, "cache_type": ["lru"]})
    with pytest.raises(ValueError, match="missing cache_type"):
        HistoryCache.from_state_dict(
            {key: value for key, value in state.items() if key != "cache_type"}
        )
    cache.cache_type = "clock"
    with pytest.raises(ValueError, match="policy does not match"):
        cache.state_dict()
