"""Compare final weights and exact training-sample evaluations in fresh processes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

from checkpoint_runtime import deterministic_subprocess_environment
from compare_checkpoints import (
    _finalize_report,
    _print_detailed_report,
    _print_verdict,
    _record_comparison,
    _write_report,
    checkpoint_report,
)


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


def _fail_with_report(report, report_path, *details):
    """Write and print a failed report, then return a nonzero process status."""
    _write_report(report, report_path)
    _print_detailed_report(report)
    _print_verdict(
        False,
        report["comparison"],
        (*details, f"JSON report: {report_path}"),
        stream=sys.stdout,
    )
    raise SystemExit(1)


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
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report destination; defaults to OUTPUT_DIR/comparison_report.json.",
    )
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_actions is not None and args.max_actions < 1:
        parser.error("--max-actions must be positive")

    continuous = args.continuous.resolve()
    sequential = args.sequential.resolve()
    output_dir = args.output_dir.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_dir / "comparison_report.json"
    )
    print("\n[1/2] Comparing complete persisted checkpoint state...", flush=True)
    persisted = checkpoint_report(continuous, sequential)
    checks = [
        {**check, "name": f"Checkpoint / {check['name']}"} for check in persisted["checks"]
    ]
    if not persisted["overall_passed"]:
        report = _finalize_report(
            "checkpoint state and fresh-process evaluation results",
            continuous,
            sequential,
            checks,
        )
        _fail_with_report(
            report,
            report_path,
            "Checkpoint comparison failed; evaluation was not run.",
        )
    print("[1/2] PASS: checkpoint state is exactly identical.", flush=True)
    checkpoint_digests = {
        continuous: _checkpoint_digest(continuous),
        sequential: _checkpoint_digest(sequential),
    }

    environment = deterministic_subprocess_environment()
    worker = directory / "evaluate_checkpoint.py"
    print("\n[2/2] Evaluating identical training rows in fresh processes...", flush=True)
    try:
        first = _evaluate(worker, continuous, output_dir / "continuous", args, environment)
        second = _evaluate(worker, sequential, output_dir / "sequential", args, environment)
    except Exception as error:
        traceback.print_exc()
        checks.append(
            {
                "name": "Evaluation / Fresh-process execution",
                "passed": False,
                "detail": f"{type(error).__name__}: {error}",
            }
        )
        report = _finalize_report(
            "checkpoint state and fresh-process evaluation results",
            continuous,
            sequential,
            checks,
        )
        _fail_with_report(report, report_path, "Fresh-process evaluation failed.")

    current_digests = {
        continuous: _checkpoint_digest(continuous),
        sequential: _checkpoint_digest(sequential),
    }
    _record_comparison(
        checks,
        "Evaluation / Checkpoint files remained unchanged",
        checkpoint_digests,
        current_digests,
        "evaluation.checkpoint_digests",
    )
    _record_comparison(
        checks,
        "Evaluation / Selected training rows",
        first["indices"],
        second["indices"],
        "evaluation.indices",
    )
    _record_comparison(
        checks,
        "Evaluation / Action limit",
        first["max_actions"],
        second["max_actions"],
        "evaluation.max_actions",
    )
    _record_comparison(
        checks,
        "Evaluation / Online-network variables",
        first["online_variables"],
        second["online_variables"],
        "evaluation.online_variables",
    )
    _record_comparison(
        checks,
        "Evaluation / Target-network variables",
        first["target_variables"],
        second["target_variables"],
        "evaluation.target_variables",
    )
    _record_comparison(
        checks,
        "Evaluation / Optimizer variables",
        first["optimizer_variables"],
        second["optimizer_variables"],
        "evaluation.optimizer_variables",
    )
    _record_comparison(
        checks,
        "Evaluation / Greedy ECM results",
        first["results"],
        second["results"],
        "evaluation.results",
    )
    report = _finalize_report(
        "checkpoint state and fresh-process evaluation results",
        continuous,
        sequential,
        checks,
    )
    if not report["overall_passed"]:
        failure_count = sum(not check["passed"] for check in checks)
        category = "category" if failure_count == 1 else "categories"
        _fail_with_report(
            report,
            report_path,
            f"{failure_count} comparison {category} failed.",
        )

    _write_report(report, report_path)
    _print_detailed_report(report)
    print("[2/2] PASS: restored variables and evaluation results are exactly identical.")
    _print_verdict(
        True,
        "checkpoint state and fresh-process evaluation results",
        (
            "Complete persisted checkpoint state is exactly identical.",
            "Online, target, and optimizer variables are exactly identical.",
            f"Greedy evaluation results are exactly identical for rows {first['indices']}.",
            f"Human-readable CSV files: {output_dir}",
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
            "checkpoint state and fresh-process evaluation results",
            (f"{type(error).__name__}: {error}",),
        )
        raise SystemExit(1) from None
