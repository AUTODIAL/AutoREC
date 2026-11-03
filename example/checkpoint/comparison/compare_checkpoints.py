"""Compare the final continuous and sequential checkpoints using exact equality."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import struct
import sys
import traceback

import numpy as np
import pandas as pd


def _print_verdict(passed, comparison, details=(), *, stream=None):
    """Print a prominent final comparison verdict."""
    if stream is None:
        stream = sys.stdout if passed else sys.stderr
    border = "=" * 72
    print(f"\n{border}", file=stream)
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}", file=stream)
    print(f"Comparison: {comparison}", file=stream)
    for detail in details:
        print(f"- {detail}", file=stream)
    print(border, file=stream, flush=True)


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


def _record_comparison(checks, name, first, second, path):
    """Append one exact comparison result without stopping later checks."""
    try:
        difference = _difference(first, second, path)
    except Exception as error:
        difference = f"{path}: {type(error).__name__}: {error}"
    checks.append(
        {
            "name": name,
            "passed": difference is None,
            "detail": "Exact match." if difference is None else difference,
        }
    )


def _record_load_failure(checks, name, error):
    checks.append(
        {
            "name": name,
            "passed": False,
            "detail": f"{type(error).__name__}: {error}",
        }
    )


def _finalize_report(comparison, first, second, checks):
    return {
        "report_version": 1,
        "comparison": comparison,
        "continuous_checkpoint": str(first),
        "sequential_checkpoint": str(second),
        "overall_passed": bool(checks) and all(check["passed"] for check in checks),
        "checks": checks,
    }


def _print_detailed_report(report):
    print("\nDetailed comparison report")
    print("-" * 72)
    for check in report["checks"]:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}")
        if not check["passed"]:
            print(f"       {check['detail']}")
    print("-" * 72, flush=True)


def _write_report(report, destination: Path):
    """Atomically write a machine-readable comparison report."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination)


def _comparable_config(config):
    """Exclude output location and mutable per-session length from training state."""
    comparable = deepcopy(config)
    comparable.pop("output")
    comparable["agent"].pop("num_trials")
    return comparable


def _checkpoint_report(first: Path, second: Path):
    """Compare checkpoint categories independently and return a structured report."""
    checks = []
    missing = [str(path) for path in (first, second) if not path.is_dir()]
    checks.append(
        {
            "name": "Checkpoint directories",
            "passed": not missing,
            "detail": "Both directories exist."
            if not missing
            else f"Missing checkpoint directories: {missing}",
        }
    )
    if missing:
        return _finalize_report("complete persisted checkpoint state", first, second, checks)

    try:
        manifests = (_load_json(first, "manifest.json"), _load_json(second, "manifest.json"))
    except Exception as error:
        _record_load_failure(checks, "Manifest", error)
    else:
        _record_comparison(
            checks,
            "Checkpoint format version",
            manifests[0]["format_version"],
            manifests[1]["format_version"],
            "manifest.format_version",
        )
        _record_comparison(
            checks,
            "Completed trial count",
            manifests[0]["completed_trials"],
            manifests[1]["completed_trials"],
            "manifest.completed_trials",
        )
        _record_comparison(
            checks,
            "Runtime, dependency versions, and AutoREC source",
            manifests[0]["runtime"],
            manifests[1]["runtime"],
            "manifest.runtime",
        )

    try:
        configs = (
            _comparable_config(_load_json(first, "config.json")),
            _comparable_config(_load_json(second, "config.json")),
        )
    except Exception as error:
        _record_load_failure(checks, "Training-critical configuration", error)
    else:
        _record_comparison(
            checks,
            "Training-critical configuration",
            configs[0],
            configs[1],
            "config",
        )

    try:
        training = (
            _load_pickle(first, "training.pkl"),
            _load_pickle(second, "training.pkl"),
        )
    except Exception as error:
        _record_load_failure(
            checks, "Training progress, schedules, replay metadata, and RNG", error
        )
    else:
        _record_comparison(
            checks,
            "Exploration and prioritized-replay schedule state",
            training[0]["attributes"],
            training[1]["attributes"],
            "training.attributes",
        )
        counter_names = (
            "iteration",
            "target_freq",
            "memory",
            "success_count",
            "completed_trials",
            "schedule_config",
            "at_trial_boundary",
        )
        history_names = (
            "NN_loss",
            "success_rate",
            "terminal_states",
            "statistical_analysis",
        )
        _record_comparison(
            checks,
            "Training counters and schedule progress",
            {name: training[0]["progress"][name] for name in counter_names},
            {name: training[1]["progress"][name] for name in counter_names},
            "training.progress.counters",
        )
        _record_comparison(
            checks,
            "Training histories and statistics",
            {name: training[0]["progress"][name] for name in history_names},
            {name: training[1]["progress"][name] for name in history_names},
            "training.progress.histories",
        )
        _record_comparison(
            checks,
            "Replay-buffer metadata and insertion position",
            training[0]["replay"],
            training[1]["replay"],
            "training.replay",
        )
        _record_comparison(
            checks,
            "Python, NumPy, and TensorFlow RNG state",
            training[0]["rng"],
            training[1]["rng"],
            "training.rng",
        )

    try:
        environments = (
            _load_pickle(first, "environments.pkl"),
            _load_pickle(second, "environments.pkl"),
        )
    except Exception as error:
        _record_load_failure(checks, "Environment and circuit-cache state", error)
    else:
        _record_comparison(
            checks,
            "Environment identity and active environment",
            {
                "version": environments[0]["version"],
                "active": environments[0]["active"],
                "count": len(environments[0]["environments"]),
            },
            {
                "version": environments[1]["version"],
                "active": environments[1]["active"],
                "count": len(environments[1]["environments"]),
            },
            "environments.identity",
        )
        _record_comparison(
            checks,
            "Prepared datasets and complete environment attributes",
            [item["attributes"] for item in environments[0]["environments"]],
            [item["attributes"] for item in environments[1]["environments"]],
            "environments.attributes",
        )
        _record_comparison(
            checks,
            "Circuit-cache entries, statistics, and eviction metadata",
            [item["cache"] for item in environments[0]["environments"]],
            [item["cache"] for item in environments[1]["environments"]],
            "environments.caches",
        )

    try:
        neural = (
            _load_pickle(first, "neural.pkl"),
            _load_pickle(second, "neural.pkl"),
        )
    except Exception as error:
        _record_load_failure(checks, "Neural-network and optimizer state", error)
    else:
        metadata_names = ("version", "tensorflow_version", "keras_version")
        _record_comparison(
            checks,
            "Neural-state format and framework versions",
            {name: neural[0][name] for name in metadata_names},
            {name: neural[1][name] for name in metadata_names},
            "neural.metadata",
        )
        _record_comparison(
            checks,
            "Online-network architecture and all variables",
            neural[0]["online"],
            neural[1]["online"],
            "neural.online",
        )
        _record_comparison(
            checks,
            "Target-network architecture and all variables",
            neural[0]["target"],
            neural[1]["target"],
            "neural.target",
        )
        _record_comparison(
            checks,
            "Optimizer configuration, iteration, and slot variables",
            neural[0]["optimizer"],
            neural[1]["optimizer"],
            "neural.optimizer",
        )

    try:
        with (first / "replay.npy").open("rb") as stream:
            first_replay = np.load(stream, allow_pickle=False)
        with (second / "replay.npy").open("rb") as stream:
            second_replay = np.load(stream, allow_pickle=False)
    except Exception as error:
        _record_load_failure(checks, "Complete physical replay-buffer contents", error)
    else:
        _record_comparison(
            checks,
            "Complete physical replay-buffer contents and ordering",
            first_replay,
            second_replay,
            "replay.npy",
        )

    return _finalize_report("complete persisted checkpoint state", first, second, checks)


def checkpoint_report(first: Path, second: Path):
    """Return a report even when an invalid component prevents normal comparison."""
    try:
        return _checkpoint_report(first, second)
    except Exception as error:
        checks = []
        _record_load_failure(checks, "Checkpoint report generation", error)
        return _finalize_report("complete persisted checkpoint state", first, second, checks)


def compare(first: Path, second: Path):
    """Retain the original raise-on-difference programmatic comparison API."""
    report = checkpoint_report(first, second)
    if not report["overall_passed"]:
        failure = next(check for check in report["checks"] if not check["passed"])
        raise AssertionError(f"{failure['name']}: {failure['detail']}")
    return report


def main():
    directory = Path(__file__).resolve().parent
    checkpoint_directory = directory.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuous",
        type=Path,
        default=checkpoint_directory / "outputs/continuous/checkpoints/trial-00001000",
    )
    parser.add_argument(
        "--sequential",
        type=Path,
        default=checkpoint_directory
        / "outputs/complete_checkpoint/checkpoints/trial-00001000",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=checkpoint_directory / "outputs/comparison_reports/checkpoint_comparison.json",
        help="JSON destination for the detailed comparison report.",
    )
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = checkpoint_report(args.continuous.resolve(), args.sequential.resolve())
    _write_report(report, report_path)
    _print_detailed_report(report)
    if not report["overall_passed"]:
        failure_count = sum(not check["passed"] for check in report["checks"])
        category = "category" if failure_count == 1 else "categories"
        _print_verdict(
            False,
            report["comparison"],
            (
                f"{failure_count} comparison {category} failed.",
                f"JSON report: {report_path}",
            ),
            stream=sys.stdout,
        )
        raise SystemExit(1)
    _print_verdict(
        True,
        "complete persisted checkpoint state",
        (
            "Continuous and sequential checkpoints are exactly identical.",
            "Output paths and per-session num_trials were intentionally excluded.",
            f"JSON report: {report_path}",
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        _print_verdict(
            False,
            "complete persisted checkpoint state",
            (f"{type(error).__name__}: {error}",),
        )
        raise SystemExit(1) from None
