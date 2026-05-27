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
