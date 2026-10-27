"""Compare the final continuous and sequential checkpoints using exact equality."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import struct

import numpy as np
import pandas as pd


COMPONENTS = ("training.pkl", "environments.pkl", "neural.pkl")


def _difference(first, second, path="root"):
    if type(first) is not type(second):
        return f"{path}: types differ ({type(first).__name__} != {type(second).__name__})"
    if isinstance(first, np.ndarray):
        if first.dtype != second.dtype:
            return f"{path}: NumPy dtypes differ ({first.dtype} != {second.dtype})"
        if first.shape != second.shape:
            return f"{path}: NumPy shapes differ ({first.shape} != {second.shape})"
        if first.tobytes() != second.tobytes():
            return f"{path}: NumPy array bytes differ"
        return None
    if isinstance(first, np.generic):
        if first.dtype != second.dtype or first.tobytes() != second.tobytes():
            return f"{path}: NumPy scalar bytes differ"
        return None
    if isinstance(first, pd.DataFrame):
        if type(first.index) is not type(second.index) or not first.index.equals(second.index):
            return f"{path}: DataFrame indices differ"
        if type(first.columns) is not type(second.columns) or not first.columns.equals(
            second.columns
        ):
            return f"{path}: DataFrame columns differ"
        if tuple(map(str, first.dtypes)) != tuple(map(str, second.dtypes)):
            return f"{path}: DataFrame dtypes differ"
        for row in range(len(first)):
            for column in range(len(first.columns)):
                difference = _difference(
                    first.iat[row, column],
                    second.iat[row, column],
                    f"{path}.iat[{row}, {column}]",
                )
                if difference:
                    return difference
        return None
    if isinstance(first, dict):
        if tuple(first) != tuple(second):
            return f"{path}: dictionary keys/order differ"
        for key in first:
            difference = _difference(first[key], second[key], f"{path}[{key!r}]")
            if difference:
                return difference
        return None
    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            return f"{path}: sequence lengths differ ({len(first)} != {len(second)})"
        for index, (left, right) in enumerate(zip(first, second)):
            difference = _difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if isinstance(first, float):
        if struct.pack("!d", first) != struct.pack("!d", second):
            return f"{path}: float bytes differ ({first!r} != {second!r})"
        return None
    if isinstance(first, complex):
        if struct.pack("!dd", first.real, first.imag) != struct.pack(
            "!dd", second.real, second.imag
        ):
            return f"{path}: complex bytes differ ({first!r} != {second!r})"
        return None
    # Cache payloads can contain CPU arrays from installed numerical libraries.
    if hasattr(first, "__array__") and not isinstance(first, (str, bytes)):
        left, right = np.asarray(first), np.asarray(second)
        if left.dtype != right.dtype or left.shape != right.shape:
            return f"{path}: array-like dtype/shape differs"
        if left.tobytes() != right.tobytes():
            return f"{path}: array-like bytes differ"
        return None
    if first != second:
        return f"{path}: values differ ({first!r} != {second!r})"
    return None


def _load_json(directory: Path, filename: str):
    return json.loads((directory / filename).read_text())


def _load_pickle(directory: Path, filename: str):
    # Checkpoints are trusted local experiment output. Never unpickle untrusted files.
    with (directory / filename).open("rb") as stream:
        return pickle.load(stream)


def _comparable_config(config):
    """Exclude output location and mutable per-session length from training state."""
    comparable = deepcopy(config)
    comparable.pop("output")
    comparable["agent"].pop("num_trials")
    return comparable


def compare(first: Path, second: Path):
    for directory in (first, second):
        if not directory.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")

    first_manifest = _load_json(first, "manifest.json")
    second_manifest = _load_json(second, "manifest.json")
    for field in ("format_version", "completed_trials", "runtime"):
        difference = _difference(
            first_manifest[field], second_manifest[field], f"manifest.{field}"
        )
        if difference:
            raise AssertionError(difference)

    first_config = _comparable_config(_load_json(first, "config.json"))
    second_config = _comparable_config(_load_json(second, "config.json"))
    difference = _difference(first_config, second_config, "config")
    if difference:
        raise AssertionError(difference)

    for filename in COMPONENTS:
        difference = _difference(
            _load_pickle(first, filename), _load_pickle(second, filename), filename
        )
        if difference:
            raise AssertionError(difference)

    with (first / "replay.npy").open("rb") as stream:
        first_replay = np.load(stream, allow_pickle=False)
    with (second / "replay.npy").open("rb") as stream:
        second_replay = np.load(stream, allow_pickle=False)
    difference = _difference(first_replay, second_replay, "replay.npy")
    if difference:
        raise AssertionError(difference)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuous",
        type=Path,
        default=directory / "outputs/continuous/checkpoints/trial-00001000",
    )
    parser.add_argument(
        "--sequential",
        type=Path,
        default=directory / "outputs/sequential/checkpoints/trial-00001000",
    )
    args = parser.parse_args()
    compare(args.continuous.resolve(), args.sequential.resolve())
    print("Exact comparison passed: all persisted training state is identical.")
    print("Checkpoint output paths were intentionally excluded from comparison.")


if __name__ == "__main__":
    main()
