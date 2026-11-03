"""Compare every checkpoint-continuation scenario with the uninterrupted reference."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import subprocess
import sys

from ..checkpoint_runtime import REPOSITORY_ROOT, deterministic_subprocess_environment
from .compare_checkpoints import _difference, _write_report, checkpoint_report
from .compare_results import _checkpoint_digest


MODEL_ONLY_REQUIRED_MATCHES = {
    "Checkpoint directories",
    "Checkpoint format version",
    "Runtime, dependency versions, and AutoREC source",
    "Training-critical configuration",
    "Environment identity and active environment",
    "Neural-state format and framework versions",
}
MODEL_ONLY_REQUIRED_DIFFERENCES = {
    "Completed trial count",
    "Training counters and schedule progress",
    "Training histories and statistics",
    "Replay-buffer metadata and insertion position",
    "Optimizer configuration, iteration, and slot variables",
    "Complete physical replay-buffer contents and ordering",
}


def assess_checkpoint_state(name: str, raw_report: dict, *, expect_exact: bool) -> dict:
    """Interpret one exact checkpoint report according to its scenario contract."""
    checks = {check["name"]: check for check in raw_report["checks"]}
    if expect_exact:
        passed = raw_report["overall_passed"]
        missing_matches = []
        missing_differences = []
    else:
        missing_matches = sorted(
            check_name
            for check_name in MODEL_ONLY_REQUIRED_MATCHES
            if check_name not in checks or not checks[check_name]["passed"]
        )
        missing_differences = sorted(
            check_name
            for check_name in MODEL_ONLY_REQUIRED_DIFFERENCES
            if check_name not in checks or checks[check_name]["passed"]
        )
        passed = not missing_matches and not missing_differences

    return {
        "name": name,
        "expected_relation": "exact_match" if expect_exact else "known_state_loss",
        "passed": passed,
        "observed_exact_match": raw_report["overall_passed"],
        "missing_required_matches": missing_matches,
        "missing_required_differences": missing_differences,
        "state_checks": raw_report["checks"],
        "evaluation": None,
    }


def _evaluate_checkpoint(checkpoint: Path, output: Path, args) -> dict:
    """Evaluate one trusted local checkpoint in a deterministic fresh process."""
    command = [
        sys.executable,
        "-m",
        "example.checkpoint.comparison.evaluate_checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--sample-seed",
        str(args.sample_seed),
    ]
    if args.indices is not None:
        command.extend(("--indices", args.indices))
    else:
        command.extend(("--num-samples", str(args.num_samples)))
    if args.max_actions is not None:
        command.extend(("--max-actions", str(args.max_actions)))
    subprocess.run(
        command,
        env=deterministic_subprocess_environment(),
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    with (output / "evaluation.pkl").open("rb") as stream:
        return pickle.load(stream)


def _evaluation_comparisons(reference: dict, candidate: dict) -> list[dict]:
    """Return named exact-match observations for comparable evaluation categories."""
    categories = {
        "Selected rows": "indices",
        "Action limit": "max_actions",
        "Online-network variables": "online_variables",
        "Target-network variables": "target_variables",
        "Optimizer variables": "optimizer_variables",
        "Greedy ECM results": "results",
    }
    comparisons = []
    for name, key in categories.items():
        difference = _difference(reference[key], candidate[key], f"evaluation.{key}")
        comparisons.append(
            {
                "name": name,
                "exact_match": difference is None,
                "detail": "Exact match." if difference is None else difference,
            }
        )
    return comparisons


def add_evaluations(report: dict, reference: Path, candidates: dict[str, Path], args) -> None:
    """Add optional fresh-process evaluation evidence without changing state expectations."""
    output_root = args.evaluation_output.resolve()
    paths = {"reference": reference, **candidates}
    before = {name: _checkpoint_digest(path) for name, path in paths.items()}
    reference_result = _evaluate_checkpoint(reference, output_root / "reference", args)

    by_name = {scenario["name"]: scenario for scenario in report["scenarios"]}
    for name, checkpoint in candidates.items():
        try:
            candidate_result = _evaluate_checkpoint(checkpoint, output_root / name, args)
            comparisons = _evaluation_comparisons(reference_result, candidate_result)
            exact_expected = by_name[name]["expected_relation"] == "exact_match"
            exact_satisfied = all(item["exact_match"] for item in comparisons)
            evaluation_passed = not exact_expected or exact_satisfied
            by_name[name]["evaluation"] = {
                "passed": evaluation_passed,
                "exact_match_expected": exact_expected,
                "comparisons": comparisons,
            }
            by_name[name]["passed"] &= evaluation_passed
        except Exception as error:
            by_name[name]["evaluation"] = {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
            by_name[name]["passed"] = False

    after = {name: _checkpoint_digest(path) for name, path in paths.items()}
    report["checkpoint_files_unchanged_by_evaluation"] = before == after
    if before != after:
        for scenario in report["scenarios"]:
            scenario["passed"] = False


def _print_report(report: dict) -> None:
    """Print concise per-scenario and overall verdicts."""
    print("\nCheckpoint continuation comparison")
    print("=" * 72)
    for scenario in report["scenarios"]:
        verdict = "PASS" if scenario["passed"] else "FAIL"
        print(f"[{verdict}] {scenario['name']}: {scenario['expected_relation']}")
        if scenario["missing_required_matches"]:
            print(f"  Missing required matches: {scenario['missing_required_matches']}")
        if scenario["missing_required_differences"]:
            print(
                f"  Missing required differences: {scenario['missing_required_differences']}"
            )
    print("-" * 72)
    print(f"OVERALL VERDICT: {'PASS' if report['overall_passed'] else 'FAIL'}")


def main() -> None:
    """Compare configured candidates and write one machine-readable report."""
    checkpoint_directory = Path(__file__).resolve().parents[1]
    default_trial = "trial-00001000"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=checkpoint_directory / "outputs/continuous/checkpoints" / default_trial,
    )
    parser.add_argument(
        "--bounded",
        type=Path,
        default=checkpoint_directory / "outputs/bounded/checkpoints" / default_trial,
    )
    parser.add_argument(
        "--complete-checkpoint",
        type=Path,
        default=checkpoint_directory
        / "outputs/complete_checkpoint/checkpoints"
        / default_trial,
    )
    parser.add_argument(
        "--model-only",
        type=Path,
        default=checkpoint_directory / "outputs/model_only_restart/checkpoints/final",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=checkpoint_directory / "outputs/comparison_reports/scenarios.json",
    )
    parser.add_argument("--with-evaluation", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices")
    selection.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=checkpoint_directory / "outputs/scenario_evaluations",
    )
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_actions is not None and args.max_actions < 1:
        parser.error("--max-actions must be positive")

    reference = args.reference.resolve()
    candidates = {
        "bounded": args.bounded.resolve(),
        "complete_checkpoint": args.complete_checkpoint.resolve(),
        "model_only_restart": args.model_only.resolve(),
    }
    scenarios = []
    for name, candidate in candidates.items():
        raw = checkpoint_report(reference, candidate)
        scenarios.append(
            assess_checkpoint_state(
                name,
                raw,
                expect_exact=name != "model_only_restart",
            )
        )

    report = {
        "report_version": 1,
        "comparison": "checkpoint continuation strategies",
        "reference_checkpoint": str(reference),
        "candidate_checkpoints": {name: str(path) for name, path in candidates.items()},
        "scenarios": scenarios,
    }
    if args.with_evaluation:
        add_evaluations(report, reference, candidates, args)
    report["overall_passed"] = all(scenario["passed"] for scenario in scenarios)
    _write_report(report, args.report.resolve())
    _print_report(report)
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
