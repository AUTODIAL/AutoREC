import numpy as np


class CircularBuffer:
    """
    Fixed-size circular buffer using NumPy structured array. O(1) insertion, replaces oldest
    data when full.
    """

    def __init__(self, capacity, dtypes):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of items to store.
        dtypes : list
            List of tuples like [('EIS', 'i4'), ('state', 'U50'), ...].
        """
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=dtypes)
        self.position = 0  # Where to write next
        self.size = 0  # How many items we currently have

    def add(self, **kwargs):
        """
        Add a single experience to the buffer.

        Examples
        --------
            buffer.add(EIS=0, state="R-R-R", reward=0.5, ...)
        """
        # Write data to current position
        for key, value in kwargs.items():
            self.buffer[self.position][key] = value

        # Move position forward (circular)
        self.position = (self.position + 1) % self.capacity

        # Update size (max = capacity)
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        """Return number of items currently in buffer"""
        return self.size

    def state_dict(self) -> dict:
        """Export an independent, versioned snapshot of the complete physical ring.

        The ``buffer`` entry is a structured NumPy array carrying its own dtype. It includes
        unused slots and can be saved with ``np.save(..., allow_pickle=False)``. Metadata
        entries are ``version``, ``capacity``, ``size``, and ``position``. Positive capacities
        and nonempty, object-free structured dtypes are supported.

        No slots are reordered and no random draws occur. Mutating the snapshot does not
        change this buffer. This exports replay only, not an agent checkpoint.

        Examples
        --------
        >>> replay = CircularBuffer(2, [("reward", "f4")])
        >>> replay.add(reward=0.5)
        >>> restored = CircularBuffer.from_state_dict(replay.state_dict())
        >>> restored.size, restored.position
        (1, 1)
        """
        state = {
            "version": 1,
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "buffer": self.buffer,
        }
        self._validate_state(state)
        for name in ("capacity", "size", "position"):
            state[name] = int(state[name])
        # A structured-array copy may omit padding bytes; copy the raw bytes instead.
        state["buffer"] = (
            np.frombuffer(self.buffer.tobytes(), dtype=np.uint8).copy().view(self.buffer.dtype)
        )
        return state

    @classmethod
    def from_state_dict(cls, state: dict) -> "CircularBuffer":
        """Validate a snapshot and restore a new buffer without modifying the input.

        The restored array is writable and shares no storage with the snapshot. Validation
        checks the version, exact keys, integer counters, ring invariants, shape, and dtype
        before allocation. Unsupported types raise ``TypeError``; invalid values raise
        ``ValueError``. Agent-specific dtype/configuration compatibility belongs to the
        caller. This method performs no file I/O and does not restore sampling RNG state.
        """
        cls._validate_state(state)
        data = state["buffer"]
        restored = cls(capacity=int(state["capacity"]), dtypes=data.dtype)
        restored.buffer.view(np.uint8)[:] = np.frombuffer(data.tobytes(), dtype=np.uint8)
        restored.size = int(state["size"])
        restored.position = int(state["position"])
        return restored

    @staticmethod
    def _validate_state(state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError("Replay state must be a dictionary.")
        required = {"version", "capacity", "size", "position", "buffer"}
        if state.keys() != required:
            raise ValueError(
                f"Invalid replay state keys: missing={required - state.keys()}, "
                f"unexpected={state.keys() - required}."
            )
        for name in ("version", "capacity", "size", "position"):
            value = state[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"Replay {name} must be an integer.")
        if state["version"] != 1:
            raise ValueError(f"Unsupported replay state version: {state['version']}.")
        capacity, size, position = state["capacity"], state["size"], state["position"]
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        if not 0 <= size <= capacity:
            raise ValueError("Replay size must be between zero and capacity.")
        if not 0 <= position < capacity:
            raise ValueError("Replay position must be between zero and capacity - 1.")
        if size < capacity and position != size:
            raise ValueError("Replay position must equal size until the buffer is full.")
        data = state["buffer"]
        if not isinstance(data, np.ndarray):
            raise TypeError("Replay buffer must be a NumPy array.")
        if data.shape != (capacity,):
            raise ValueError("Replay buffer shape must be (capacity,).")
        if not data.dtype.names or data.dtype.itemsize == 0:
            raise ValueError("Replay buffer must have a nonempty structured dtype.")
        if data.dtype.hasobject:
            raise TypeError("Replay buffer dtype must not contain Python objects.")

    # TODO: It is expensive. Shall be removed and use __getitem__ instead in future.
    def to_dataframe(self):
        """Convert buffer contents to pandas DataFrame, handling multi-dimensional fields."""
        import pandas as pd

        if self.size == 0:
            return pd.DataFrame()

        data = self.buffer[: self.size]
        df_dict = {}

        for name in data.dtype.names:
            if data[name].ndim == 1:
                # Simple 1D field (like state, reward, EIS)
                df_dict[name] = data[name]
            else:
                # Multi-dimensional field (like encoded_state)
                # Convert to list of arrays so each DataFrame cell contains one array
                df_dict[name] = list(data[name])

        return pd.DataFrame(df_dict)

    def __getitem__(self, key):
        """
        Allow indexing like buffer[0:100] or buffer['EIS']
        Returns view of the buffer data
        """
        if isinstance(key, str):
            # Access column: buffer['EIS']
            return self.buffer[key][: self.size]
        else:
            # Access rows: buffer[0:10]
            return self.buffer[key]
