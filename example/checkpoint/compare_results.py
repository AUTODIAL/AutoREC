"""Compare final weights and exact training-sample evaluations in fresh processes."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys

from compare_checkpoints import _difference, compare as compare_checkpoints


def _checkpoint_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _evaluate(worker, checkpoint, output, args, environment):
    command = [
        sys.executable,
        str(worker),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--sample-seed",
        str(args.sample_seed),
    ]
    if args.indices is not None:
        command += ["--indices", args.indices]
    else:
        command += ["--num-samples", str(args.num_samples)]
    if args.max_actions is not None:
        command += ["--max-actions", str(args.max_actions)]
    subprocess.run(command, env=environment, check=True)
    with (output / "evaluation.pkl").open("rb") as stream:
        return pickle.load(stream)


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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices", help="Comma-separated training dataset row positions.")
    selection.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=directory / "outputs/evaluation_comparison"
    )
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_actions is not None and args.max_actions < 1:
        parser.error("--max-actions must be positive")

    continuous = args.continuous.resolve()
    sequential = args.sequential.resolve()
    compare_checkpoints(continuous, sequential)
    print("Complete final checkpoint comparison passed.")
    checkpoint_digests = {
        continuous: _checkpoint_digest(continuous),
        sequential: _checkpoint_digest(sequential),
    }

    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONHASHSEED="0",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    output_dir = args.output_dir.resolve()
    worker = directory / "evaluate_checkpoint.py"
    first = _evaluate(worker, continuous, output_dir / "continuous", args, environment)
    second = _evaluate(worker, sequential, output_dir / "sequential", args, environment)
    for checkpoint, digest in checkpoint_digests.items():
        if _checkpoint_digest(checkpoint) != digest:
            raise RuntimeError(f"Evaluation modified checkpoint files: {checkpoint}")
    difference = _difference(first, second, "evaluation")
    if difference:
        raise AssertionError(difference)

    print("Exact online, target, and optimizer variable comparison passed.")
    print(f"Exact evaluation comparison passed for training rows {first['indices']}.")
    print(f"Human-readable evaluation CSV files are in {output_dir}.")


if __name__ == "__main__":
    main()
