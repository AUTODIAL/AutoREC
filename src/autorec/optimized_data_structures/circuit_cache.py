"""
Circuit Evaluation Cache Module

Provides efficient caching mechanisms to avoid re-computing expensive circuit parameter
fitting operations. Includes both Clock (Second-Chance) and LRU cache implementations with
configurable size limits.
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Tuple, Optional
import numpy as np


def _validate_cache_state(state: dict, cache_type: str) -> None:
    """Validate cache metadata without reading entries through the cache API."""
    if not isinstance(state, dict):
        raise TypeError("Cache state must be a dictionary.")
    required = {"version", "cache_type", "capacity", "entries", "hits", "misses"}
    if cache_type == "clock":
        required |= {"reference_bits", "keys_list", "hand"}
    if state.keys() != required:
        raise ValueError(
            f"Invalid cache state keys: missing={required - state.keys()}, "
            f"unexpected={state.keys() - required}."
        )
    for name in ("version", "capacity", "hits", "misses"):
        _validate_integer(state[name], name)
    if state["version"] != 1:
        raise ValueError(f"Unsupported cache state version: {state['version']}.")
    if not isinstance(state["cache_type"], str):
        raise TypeError("Cache cache_type must be a string.")
    if state["cache_type"] != cache_type:
        raise ValueError(f"Cache policy mismatch: expected {cache_type!r}.")
    if state["capacity"] < 1:
        raise ValueError("Cache capacity must be positive.")
    if state["hits"] < 0 or state["misses"] < 0:
        raise ValueError("Cache hits and misses must be nonnegative.")
    if not isinstance(state["entries"], list):
        raise TypeError("Cache entries must be an ordered list of (key, payload) pairs.")
    if len(state["entries"]) > state["capacity"]:
        raise ValueError("Cache entry count exceeds capacity.")
    keys = set()
    for entry in state["entries"]:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("Each cache entry must be a (key, payload) tuple.")
        key, payload = entry
        if not isinstance(key, tuple):
            raise TypeError("Cache keys must be hashable tuples.")
        try:
            hash(key)
        except TypeError as error:
            raise TypeError("Cache keys must be hashable tuples.") from error
        if key in keys:
            raise ValueError("Cache entries contain duplicate keys.")
        keys.add(key)
        if not isinstance(payload, dict):
            raise TypeError("Cache payloads must be dictionaries.")
    if cache_type == "clock":
        _validate_integer(state["hand"], "hand")
        if not 0 <= state["hand"] < state["capacity"]:
            raise ValueError("Clock hand must be between zero and capacity - 1.")
        if len(keys) < state["capacity"] and state["hand"] != 0:
            raise ValueError("Clock hand must be zero until the cache is full.")
        if not isinstance(state["keys_list"], list):
            raise TypeError("Clock keys_list must be a list.")
        if len(state["keys_list"]) != len(keys) or set(state["keys_list"]) != keys:
            raise ValueError("Clock keys_list must contain each entry key exactly once.")
        bits = state["reference_bits"]
        if not isinstance(bits, dict):
            raise TypeError("Clock reference_bits must be a dictionary.")
        if bits.keys() != keys:
            raise ValueError("Clock reference_bits keys must match entry keys.")
        if any(not isinstance(bit, (bool, np.bool_)) for bit in bits.values()):
            raise TypeError("Clock reference bits must be booleans.")


def _validate_integer(value, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Cache {name} must be an integer.")


def _cache_state(cache, cache_type: str) -> dict:
    state = {
        "version": 1,
        "cache_type": cache_type,
        "capacity": cache.capacity,
        "entries": list(cache.cache.items()),
        "hits": cache.hits,
        "misses": cache.misses,
    }
    if cache_type == "clock":
        state.update(
            reference_bits=cache.reference_bits, keys_list=cache.keys_list, hand=cache.hand
        )
    _validate_cache_state(state, cache_type)
    # One deep copy preserves shared payload objects across entries within the snapshot.
    return deepcopy(state)


class ClockCache:
    """
    Clock (Second-Chance) cache implementation.

    Uses a circular buffer with a "reference bit" to determine eviction. Items get a second
    chance before eviction, making it good for workloads with temporal locality (recently used
    items are likely to be used again).

    Advantages over LRU
    -------------------
    - O(1) insertion and lookup (LRU can be O(1) but with more overhead)
    - More efficient than LRU for certain access patterns
    - Better performance when items are accessed in bursts

    Time Complexity
    ---------------
    - Get: O(1) average, O(n) worst case (when all items have reference bit set)
    - Put: O(1) average, O(n) worst case
    """

    def __init__(self, capacity: int):
        """
        Initialize the clock cache.

        Parameters
        ----------
        capacity : int
            Maximum number of items in cache.
        """
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")

        self.capacity = capacity
        self.cache: Dict[Tuple, Dict[str, Any]] = {}
        self.reference_bits: Dict[Tuple, bool] = {}
        self.keys_list: list = []  # Circular buffer of keys
        self.hand: int = 0  # Clock hand position

        # Statistics
        self.hits = 0
        self.misses = 0

    def state_dict(self) -> dict:
        """Copy entries, counters, reference bits, ring order, and hand without touching them.

        Payloads are deep-copied without numerical conversion. Requires a paused writer
        and trusted, deepcopy-compatible data. No cache lookup, insertion, or RNG draw is
        performed. The version-1 dictionary is a component snapshot, not a checkpoint file.
        """
        return _cache_state(self, "clock")

    @classmethod
    def from_state_dict(cls, state: dict) -> "ClockCache":
        """Restore an independent cache after validating version, policy, and invariants."""
        _validate_cache_state(state, "clock")
        copied = deepcopy(state)
        restored = cls(capacity=int(copied["capacity"]))
        restored.cache = dict(copied["entries"])
        restored.reference_bits = copied["reference_bits"]
        restored.keys_list = copied["keys_list"]
        restored.hand = int(copied["hand"])
        restored.hits = int(copied["hits"])
        restored.misses = int(copied["misses"])
        return restored

    def get(self, key: Tuple) -> Optional[Dict[str, Any]]:
        """
        Retrieve item from cache.

        Parameters
        ----------
        key : tuple
            Cache key (typically (circuit_code, EIS_index)).

        Returns
        -------
        dict or None
            Cached value if found, None otherwise.
        """
        if key in self.cache:
            self.hits += 1
            self.reference_bits[key] = True  # Set reference bit (second chance)
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, key: Tuple, value: Dict[str, Any]) -> None:
        """
        Insert or update item in cache.

        Parameters
        ----------
        key : tuple
            Cache key.
        value : dict
            Value to cache.
        """
        # If key already exists, just update it
        if key in self.cache:
            self.cache[key] = value
            self.reference_bits[key] = True
            return

        # If cache is not full, just add
        if len(self.cache) < self.capacity:
            self.cache[key] = value
            self.reference_bits[key] = True
            self.keys_list.append(key)
            return

        # Cache is full - need to evict using clock algorithm
        while True:
            # Get key at current hand position
            evict_key = self.keys_list[self.hand]

            # Check reference bit
            if self.reference_bits[evict_key]:
                # Give second chance - clear bit and move hand
                self.reference_bits[evict_key] = False
                self.hand = (self.hand + 1) % self.capacity
            else:
                # Evict this item
                del self.cache[evict_key]
                del self.reference_bits[evict_key]

                # Replace with new item
                self.cache[key] = value
                self.reference_bits[key] = True
                self.keys_list[self.hand] = key

                # Move hand
                self.hand = (self.hand + 1) % self.capacity
                break

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.reference_bits.clear()
        self.keys_list.clear()
        self.hand = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Dictionary with hit rate, miss rate, and size info.
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "capacity": self.capacity,
            "utilization": len(self.cache) / self.capacity,
        }

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self.cache)

    def __contains__(self, key: Tuple) -> bool:
        """Check if key exists in cache."""
        return key in self.cache


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.

    Evicts the least recently used item when the cache is full. Uses OrderedDict for O(1)
    operations.

    Advantages
    ----------
    - Simple and well-understood eviction policy
    - Excellent for workloads with temporal locality
    - Predictable behavior

    Time Complexity
    ---------------
    - Get: O(1)
    - Put: O(1)
    """

    def __init__(self, capacity: int):
        """
        Initialize the LRU cache.

        Parameters
        ----------
        capacity : int
            Maximum number of items in cache.
        """
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")

        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0

    def state_dict(self) -> dict:
        """Copy entries in least-to-most-recent order, capacity, and hit/miss counters.

        Payloads are deep-copied without numerical conversion. Requires a paused writer
        and trusted, deepcopy-compatible data. No cache lookup, insertion, or RNG draw is
        performed. The version-1 dictionary is a component snapshot, not a checkpoint file.
        """
        return _cache_state(self, "lru")

    @classmethod
    def from_state_dict(cls, state: dict) -> "LRUCache":
        """Restore an independent cache without changing entry order or statistics."""
        _validate_cache_state(state, "lru")
        copied = deepcopy(state)
        restored = cls(capacity=int(copied["capacity"]))
        restored.cache = OrderedDict(copied["entries"])
        restored.hits = int(copied["hits"])
        restored.misses = int(copied["misses"])
        return restored

    def get(self, key: Tuple) -> Optional[Dict[str, Any]]:
        """
        Retrieve item from cache.

        Parameters
        ----------
        key : tuple
            Cache key (typically (circuit_code, EIS_index)).

        Returns
        -------
        dict or None
            Cached value if found, None otherwise.
        """
        if key in self.cache:
            self.hits += 1
            # Move to end (mark as recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, key: Tuple, value: Dict[str, Any]) -> None:
        """
        Insert or update item in cache.

        Parameters
        ----------
        key : tuple
            Cache key.
        value : dict
            Value to cache.
        """
        if key in self.cache:
            # Update existing item and move to end
            self.cache.move_to_end(key)
        else:
            # Check if cache is full
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)

        # Add new item (at end)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Dictionary with hit rate, miss rate, and size info.
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "capacity": self.capacity,
            "utilization": len(self.cache) / self.capacity,
        }

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self.cache)

    def __contains__(self, key: Tuple) -> bool:
        """Check if key exists in cache."""
        return key in self.cache


class HistoryCache:
    """
    High-level wrapper for caching circuit evaluation results.

    This class provides a simple interface for caching expensive circuit parameter fitting
    operations. It handles the creation of cache keys and provides convenience methods for
    common operations.

    Examples
    --------
        cache = HistoryCache(capacity=10000, cache_type='lru')

        # Try to get cached result
        result = cache.get(circuit_code='RPRPRR', eis_index=5)

        if result is None:
            # Compute result (expensive operation)
            result = fit_circuit_parameters(...)

            # Cache for future use
            cache.put(
                circuit_code='RPRPRR',
                eis_index=5,
                reward=0.85,
                metrics={...},
                predicted_Z=predicted_Z,
                param={...},
                good_fit=True
            )
    """

    def __init__(self, capacity: int = 20000, cache_type: str = "lru"):
        """
        Initialize the high-level circuit evaluation cache.

        Parameters
        ----------
        capacity : int
            Maximum number of circuit evaluation results to keep.
        cache_type : str
            Eviction policy to use. Supported values are ``"lru"`` and ``"clock"``.
        """
        if cache_type == "lru":
            self.cache = LRUCache(capacity)
        elif cache_type == "clock":
            self.cache = ClockCache(capacity)
        else:
            raise ValueError(f"Invalid cache_type: {cache_type}. Use 'lru' or 'clock'.")

        self.cache_type = cache_type

    def state_dict(self) -> dict:
        """Export the underlying cache with its policy, payloads, and eviction state.

        Uses the same flat version-1 format as the low-level caches. Keys and required
        circuit-result fields are validated too. See their methods for copying constraints.

        Examples
        --------
        >>> cache = HistoryCache(capacity=3, cache_type="clock")
        >>> restored = HistoryCache.from_state_dict(cache.state_dict())
        >>> restored.get_stats() == cache.get_stats()
        True
        """
        state = self.cache.state_dict()
        if state["cache_type"] != self.cache_type:
            raise ValueError("HistoryCache policy does not match its underlying cache.")
        self._validate_state(state)
        return state

    @classmethod
    def from_state_dict(cls, state: dict) -> "HistoryCache":
        """Restore the wrapper and its cache without inserting entries or recording hits."""
        cls._validate_state(state)
        cache_class = LRUCache if state["cache_type"] == "lru" else ClockCache
        cache = cache_class.from_state_dict(state)
        restored = cls(capacity=cache.capacity, cache_type=state["cache_type"])
        restored.cache = cache
        return restored

    @staticmethod
    def _validate_state(state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError("Cache state must be a dictionary.")
        if "cache_type" not in state:
            raise ValueError("HistoryCache state is missing cache_type.")
        if not isinstance(state["cache_type"], str):
            raise TypeError("Cache cache_type must be a string.")
        if state.get("cache_type") not in ("lru", "clock"):
            raise ValueError("HistoryCache state must specify cache_type 'lru' or 'clock'.")
        _validate_cache_state(state, state["cache_type"])
        fields = {
            "reward",
            "metrics",
            "predicted_Z",
            "param",
            "good_fit",
            "depth_penalty",
            "fit_bonus",
            "fit_penalty",
        }
        for key, payload in state["entries"]:
            if len(key) != 4:
                raise ValueError("HistoryCache keys must have four components.")
            if not fields <= payload.keys():
                raise ValueError(
                    f"HistoryCache payload is missing fields: {fields - payload.keys()}."
                )

    def _make_key(
        self, circuit_code: str, eis_index: int, action_type: str, action_position: int
    ) -> Tuple:
        """
        Create cache key from circuit code, EIS index, and action.

        Parameters
        ----------
        circuit_code : str
            The circuit coding string (GEP representation).
        eis_index : int
            Index of the EIS measurement in the dataset.
        action_type : str
            The element that was mutated (e.g., 'R', 'P', '+').
        action_position : int
            Position in the chromosome where mutation occurred.

        Returns
        -------
        tuple
            Tuple key for cache lookup.
        """
        return (circuit_code, eis_index, action_type, action_position)

    def get(
        self, circuit_code: str, eis_index: int, action_type: str, action_position: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached circuit evaluation result.

        Parameters
        ----------
        circuit_code : str
            The circuit coding string.
        eis_index : int
            Index of the EIS measurement.
        action_type : str
            The action element type.
        action_position : int
            The action position.

        Returns
        -------
        dict or None
            Dictionary with cached results or None if not found.
        """
        key = self._make_key(circuit_code, eis_index, action_type, action_position)
        return self.cache.get(key)

    def put(
        self,
        circuit_code: str,
        eis_index: int,
        action_type: str,
        action_position: int,
        reward: float,
        metrics: Dict[str, float],
        predicted_Z: np.ndarray,
        param: Dict[str, float],
        good_fit: bool,
        depth_penalty: float = 0.0,
        fit_bonus: float = 0.0,
        fit_penalty: float = 0.0,
    ) -> None:
        """
        Cache a circuit evaluation result.

        Parameters
        ----------
        circuit_code : str
            The circuit coding string.
        eis_index : int
            Index of the EIS measurement.
        action_type : str
            The action element type.
        action_position : int
            The action position.
        reward : float
            Computed reward value.
        metrics : dict
            Dictionary of evaluation metrics.
        predicted_Z : np.ndarray
            Predicted impedance array.
        param : dict
            Fitted circuit parameters.
        good_fit : bool
            Whether this was a good fit.
        depth_penalty : float
            Penalty for circuit depth.
        fit_bonus : float
            Bonus for fit quality.
        fit_penalty : float
            Penalty for fit issues.
        """
        key = self._make_key(circuit_code, eis_index, action_type, action_position)
        value = {
            "reward": reward,
            "metrics": metrics,
            "predicted_Z": predicted_Z,
            "param": param,
            "good_fit": good_fit,
            "depth_penalty": depth_penalty,
            "fit_bonus": fit_bonus,
            "fit_penalty": fit_penalty,
        }
        self.cache.put(key, value)

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = self.cache.get_stats()
        stats["cache_type"] = self.cache_type
        return stats

    def print_stats(self) -> None:
        """Print formatted cache statistics."""
        stats = self.get_stats()
        print(f"\n{'=' * 60}")
        print(f"Circuit Evaluation Cache Statistics ({stats['cache_type'].upper()})")
        print(f"{'=' * 60}")
        print(f"Capacity:     {stats['capacity']:,}")
        print(f"Size:         {stats['size']:,}")
        print(f"Utilization:  {stats['utilization']:.1%}")
        print(f"Hits:         {stats['hits']:,}")
        print(f"Misses:       {stats['misses']:,}")
        print(f"Hit Rate:     {stats['hit_rate']:.2%}")
        print(f"{'=' * 60}\n")

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self.cache)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Clock Cache...")
    clock_cache = ClockCache(capacity=3)

    # Add items
    clock_cache.put(("R-P-R", 0), {"reward": 0.5})
    clock_cache.put(("R-P-C", 0), {"reward": 0.6})
    clock_cache.put(("R-L-R", 0), {"reward": 0.7})

    print(f"Cache size: {len(clock_cache)}")
    print(f"Get ('R-P-R', 0): {clock_cache.get(('R-P-R', 0))}")

    # This should evict one item
    clock_cache.put(("R-C-R", 0), {"reward": 0.8})
    print(f"After adding 4th item, size: {len(clock_cache)}")
    print(f"Stats: {clock_cache.get_stats()}")

    print("\n" + "=" * 60)
    print("Testing LRU Cache...")
    lru_cache = LRUCache(capacity=3)

    lru_cache.put(("R-P-R", 0), {"reward": 0.5})
    lru_cache.put(("R-P-C", 0), {"reward": 0.6})
    lru_cache.put(("R-L-R", 0), {"reward": 0.7})

    print(f"Cache size: {len(lru_cache)}")
    print(f"Get ('R-P-R', 0): {lru_cache.get(('R-P-R', 0))}")

    lru_cache.put(("R-C-R", 0), {"reward": 0.8})
    print(f"After adding 4th item, size: {len(lru_cache)}")
    print(f"Stats: {lru_cache.get_stats()}")

    print("\n" + "=" * 60)
    print("Testing HistoryCache...")
    eval_cache = HistoryCache(capacity=3, cache_type="lru")

    eval_cache.put(
        circuit_code="RPRPRR",
        eis_index=0,
        reward=0.85,
        metrics={"r2_score": 0.95, "chi_square": 0.001},
        predicted_Z=np.array([1 + 2j, 3 + 4j]),
        param={"R0": 100, "P0-n": 0.8},
        good_fit=True,
    )

    result = eval_cache.get("RPRPRR", 0)
    print(f"Cached result: {result}")
    eval_cache.print_stats()
