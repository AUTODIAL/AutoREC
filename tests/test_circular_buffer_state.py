"""Exact replay-state round trips, independent ownership, and invalid-state rejection."""

from io import BytesIO
import json
import random

import numpy as np
import pytest

from autorec.optimized_data_structures.circular_buffer import CircularBuffer


DTYPE = np.dtype(
    [("step", ">i4"), ("state", "U50"), ("encoded", "f4", (3,)), ("priority", "f4")]
)


def add_transition(buffer, ii):
    buffer.add(step=ii, state=f"R{ii}", encoded=[ii, -0.0, ii / 8], priority=(ii + 1) / 8)


def assert_same_buffer(first, second):
    assert (first.capacity, first.size, first.position) == (
        second.capacity,
        second.size,
        second.position,
    )
    assert first.buffer.dtype == second.buffer.dtype
    assert first.buffer.shape == second.buffer.shape
    assert first.buffer.tobytes() == second.buffer.tobytes()


@pytest.mark.parametrize(
    "insertions", [0, 1, 3, 5], ids=["empty", "partial", "full", "wrapped"]
)
def test_roundtrip_preserves_all_slots_and_future_writes(insertions):
    original = CircularBuffer(3, DTYPE)
    for ii in range(insertions):
        add_transition(original, ii)
    # Inactive slots are part of the snapshot too, even when they contain nonzero bits.
    original.buffer["priority"][original.size :] = -0.0
    original.buffer["encoded"][original.size :] = 9.0

    exported = original.state_dict()
    assert set(exported) == {"version", "capacity", "size", "position", "buffer"}
    assert exported["version"] == 1
    assert exported["buffer"].tobytes() == original.buffer.tobytes()

    # Exercise the proposed NumPy/JSON storage boundary without pickle or a checkpoint API.
    metadata = json.loads(
        json.dumps({key: value for key, value in exported.items() if key != "buffer"})
    )
    with BytesIO() as stream:
        np.save(stream, exported["buffer"], allow_pickle=False)
        stream.seek(0)
        metadata["buffer"] = np.load(stream, allow_pickle=False)
    restored = CircularBuffer.from_state_dict(metadata)
    assert_same_buffer(original, restored)

    for ii in range(insertions, insertions + 5):
        add_transition(original, ii)
        add_transition(restored, ii)
        # Priority updates must still address the same physical slots.
        original["priority"][0] = restored["priority"][0] = np.float32(0.0625)
        assert_same_buffer(original, restored)


def test_snapshot_and_restored_buffer_do_not_alias():
    original = CircularBuffer(3, DTYPE)
    add_transition(original, 0)
    snapshot = original.state_dict()
    restored = CircularBuffer.from_state_dict(snapshot)
    assert not np.shares_memory(snapshot["buffer"], original.buffer)
    assert not np.shares_memory(restored.buffer, snapshot["buffer"])
    assert not np.shares_memory(restored.buffer, original.buffer)
    original.buffer["encoded"][0, 0] = 10
    snapshot["buffer"]["encoded"][0, 1] = 20
    restored.buffer["encoded"][0, 2] = 30
    np.testing.assert_array_equal(original.buffer["encoded"][0], [10, -0.0, 0])
    np.testing.assert_array_equal(snapshot["buffer"]["encoded"][0], [0, 20, 0])
    np.testing.assert_array_equal(restored.buffer["encoded"][0], [0, -0.0, 30])


def test_padding_and_special_float_bits_are_preserved():
    padded = np.dtype(
        {"names": ["tag", "value"], "formats": ["u1", "f4"], "offsets": [0, 8], "itemsize": 16}
    )
    original = CircularBuffer(3, padded)
    original.buffer.view(np.uint8)[:] = np.arange(original.buffer.nbytes, dtype=np.uint8)
    # Negative zero, a NaN payload, and infinity must not be converted or normalized.
    original.buffer["value"].view(np.uint32)[:] = [0x80000000, 0x7FC00042, 0x7F800000]
    snapshot = original.state_dict()
    assert snapshot["buffer"].tobytes() == original.buffer.tobytes()
    assert_same_buffer(original, CircularBuffer.from_state_dict(snapshot))


def test_restore_accepts_readonly_strided_array_and_numpy_integer_metadata():
    original = CircularBuffer(3, DTYPE)
    add_transition(original, 7)
    snapshot = original.state_dict()
    storage = np.zeros(6, dtype=DTYPE)
    storage[::2] = snapshot["buffer"]
    snapshot["buffer"] = storage[::2]
    snapshot["buffer"].setflags(write=False)
    for name in ("version", "capacity", "size", "position"):
        snapshot[name] = np.int64(snapshot[name])
    restored = CircularBuffer.from_state_dict(snapshot)
    assert_same_buffer(original, restored)
    assert restored.buffer.flags.writeable
    assert not snapshot["buffer"].flags.writeable
    for name in ("capacity", "size", "position"):
        assert type(restored.state_dict()[name]) is int


def test_export_and_restore_leave_rng_streams_unchanged():
    original = CircularBuffer(3, DTYPE)
    add_transition(original, 0)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    CircularBuffer.from_state_dict(original.state_dict())
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


@pytest.mark.parametrize(
    ("name", "value"),
    [("version", True), ("capacity", 1.5), ("size", "1"), ("position", np.bool_(False))],
)
def test_noninteger_metadata_is_rejected(name, value):
    # All metadata fields use the same type check: cover each field/type without a product.
    state = CircularBuffer(3, DTYPE).state_dict()
    state[name] = value
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        CircularBuffer.from_state_dict(state)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"version": 2}, "version"),
        ({"capacity": 0}, "capacity"),
        ({"size": -1}, "size"),
        ({"size": 4}, "size"),
        ({"position": -1}, "position"),
        ({"position": 3}, "position"),
        ({"size": 1, "position": 0}, "position must equal size"),
        ({"buffer": np.zeros(2, dtype=DTYPE)}, "shape"),
        ({"buffer": np.zeros((3, 1), dtype=DTYPE)}, "shape"),
        ({"buffer": np.zeros(3)}, "structured dtype"),
        ({"buffer": np.zeros(3, dtype=[])}, "structured dtype"),
    ],
)
def test_invalid_metadata_or_array_is_rejected_without_modifying_input(updates, message):
    state = CircularBuffer(3, DTYPE).state_dict()
    state.update(updates)
    before = state["buffer"].tobytes()
    with pytest.raises(ValueError, match=message):
        CircularBuffer.from_state_dict(state)
    assert state["buffer"].tobytes() == before


def test_missing_keys_are_rejected():
    state = CircularBuffer(3, DTYPE).state_dict()
    del state["buffer"]
    with pytest.raises(ValueError, match="missing=.*buffer"):
        CircularBuffer.from_state_dict(state)


def test_extra_keys_and_wrong_payload_types_are_rejected():
    state = CircularBuffer(3, DTYPE).state_dict()
    with pytest.raises(ValueError, match="unexpected=.*extra"):
        CircularBuffer.from_state_dict({**state, "extra": 1})
    with pytest.raises(TypeError, match="dictionary"):
        CircularBuffer.from_state_dict(None)
    with pytest.raises(TypeError, match="NumPy array"):
        CircularBuffer.from_state_dict({**state, "buffer": state["buffer"].tolist()})


@pytest.mark.parametrize("dtype", [[("value", object)], [("nested", [("value", object)])]])
def test_object_fields_are_rejected_on_export_and_restore(dtype):
    unsupported = CircularBuffer(3, dtype)
    with pytest.raises(TypeError, match="Python objects"):
        unsupported.state_dict()
    state = CircularBuffer(3, DTYPE).state_dict()
    with pytest.raises(TypeError, match="Python objects"):
        CircularBuffer.from_state_dict({**state, "buffer": unsupported.buffer})
