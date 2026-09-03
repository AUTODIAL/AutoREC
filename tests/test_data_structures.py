"""Tests for small cache and buffer data structures used during training."""

import pytest

from autorec.optimized_data_structures.circular_buffer import CircularBuffer
from autorec.optimized_data_structures.circuit_cache import ClockCache, HistoryCache, LRUCache


def test_circular_buffer_tracks_size_and_overwrites_old_entries():
    buffer = CircularBuffer(capacity=2, dtypes=[("idx", "i4"), ("state", "U10")])

    buffer.add(idx=1, state="R")
    buffer.add(idx=2, state="L")
    buffer.add(idx=3, state="C")

    assert len(buffer) == 2
    assert list(buffer["idx"]) == [3, 2]
    assert list(buffer["state"]) == ["C", "L"]


def test_lru_cache_evicts_least_recently_used_key():
    cache = LRUCache(capacity=2)
    cache.put(("a", 1), {"reward": 1})
    cache.put(("b", 1), {"reward": 2})

    # Reading "a" makes "b" the least-recently-used entry.
    assert cache.get(("a", 1)) == {"reward": 1}
    cache.put(("c", 1), {"reward": 3})

    assert ("a", 1) in cache
    assert ("b", 1) not in cache
    assert ("c", 1) in cache
    assert cache.get_stats()["hits"] == 1


def test_clock_cache_tracks_hits_misses_and_capacity():
    cache = ClockCache(capacity=2)
    cache.put(("a", 1), {"reward": 1})
    cache.put(("b", 1), {"reward": 2})

    assert cache.get(("a", 1)) == {"reward": 1}
    assert cache.get(("missing", 1)) is None
    cache.put(("c", 1), {"reward": 3})

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 2
    assert stats["capacity"] == 2


def test_cache_capacity_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        LRUCache(capacity=0)

    with pytest.raises(ValueError, match="positive"):
        ClockCache(capacity=0)


def test_history_cache_uses_circuit_and_eis_index_key():
    cache = HistoryCache(capacity=2, cache_type="lru")

    # HistoryCache wraps the low-level caches and builds a key from circuit, EIS,
    # action type, and action position.
    cache.put(
        circuit_code="RPRR",
        eis_index=5,
        action_type="P",
        action_position=2,
        reward=0.75,
        metrics={"r2": 0.99},
        predicted_Z=[1 + 1j],
        param={"R1": 1.0},
        good_fit=True,
    )

    result = cache.get(circuit_code="RPRR", eis_index=5, action_type="P", action_position=2)

    assert result["reward"] == 0.75
    assert result["metrics"] == {"r2": 0.99}
    assert result["good_fit"] is True


def test_history_cache_rejects_unknown_cache_type():
    with pytest.raises(ValueError, match="cache_type"):
        HistoryCache(cache_type="unknown")
