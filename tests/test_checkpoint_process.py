"""Exact continuation across independent interpreters with a controlled circuit fitter."""

import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


def run_worker(directory, *, resume=None, horizon=12, count=12, cache="lru", save_every=None):
    command = [
        sys.executable,
        str(Path(__file__).with_name("checkpoint_worker.py")),
        str(directory),
        "--horizon",
        str(horizon),
        "--count",
        str(count),
        "--cache",
        cache,
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    if save_every is not None:
        command += ["--save-every", str(save_every)]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONHASHSEED="0",
    )
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=240
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    with (directory / "result.pkl").open("rb") as stream:
        return pickle.load(stream)


def compare_sessions(tmp_path, *, horizon, counts, cache):
    continuous = run_worker(
        tmp_path / "continuous", cache=cache, horizon=horizon, count=horizon
    )
    saving = run_worker(
        tmp_path / "saving", cache=cache, horizon=horizon, count=horizon, save_every=counts[0]
    )
    assert continuous == saving
    resume = None
    trace = []
    completed = 0
    for ii, count in enumerate(counts):
        directory = tmp_path / f"session-{ii}"
        current = run_worker(
            directory,
            resume=resume,
            cache=cache,
            horizon=horizon,
            count=count,
            save_every=counts[0],
        )
        trace += current["trace"]
        completed += count
        resume = directory / "periodic" / f"trial-{completed:08d}"
        assert resume.is_dir()
    assert continuous["state"] == current["state"]
    assert continuous["trace"] == trace


@pytest.mark.parametrize("cache", ["lru", "clock"])
def test_fresh_process_continuation(tmp_path, cache):
    compare_sessions(tmp_path, horizon=12, counts=[3, 4, 5], cache=cache)


@pytest.mark.skipif(
    os.environ.get("AUTOREC_LONG_CHECKPOINT_TEST") != "1",
    reason="Opt-in 1000-trial fresh-process acceptance test",
)
def test_1000_trials_vs_ten_fresh_process_sessions(tmp_path):
    compare_sessions(tmp_path, horizon=1000, counts=[100] * 10, cache="clock")
