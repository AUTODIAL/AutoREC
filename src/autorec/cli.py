"""Command line interface for AutoREC workflows."""

from __future__ import annotations

import argparse
import datetime as _datetime
from pathlib import Path
from typing import Sequence


def _configure_runtime(args: argparse.Namespace) -> None:
    """Configure AutoREC runtime settings before importing heavy dependencies."""
    from autorec.runtime import configure_autorec_runtime

    configure_autorec_runtime(
        thread_count=args.threads,
        warmup_autoeis=not args.skip_autoeis_warmup,
        suppress_tf_logs=not args.show_tf_logs,
    )


def _parse_indices(raw_indices: str | None) -> list[int] | None:
    """Parse a comma-separated CLI index list into integer row positions."""
    if raw_indices is None:
        return None
    indices = [item.strip() for item in raw_indices.split(",")]
    return [int(item) for item in indices if item]


def _read_config_with_overrides(config_path: Path, output_dir: Path | None = None) -> dict:
    """Read a YAML config and apply CLI overrides that should win over YAML values."""
    from autorec.factory import config_reader

    config = config_reader(config_path)
    if output_dir is not None:
        agent_config = config.get("agent") or {}
        config["agent"] = agent_config
        agent_config["save_dir"] = output_dir
    return config


def _run_preprocess(args: argparse.Namespace) -> int:
    """Run the data preprocessing or dataset validation CLI command."""
    _configure_runtime(args)

    from autorec.data_preparation import EISDataPrep

    prep = EISDataPrep(
        path=args.input,
        mode=args.mode,
        evaluation=args.evaluation,
        eis_features=args.eis_features,
    )
    prep.load()

    if args.summary:
        prep.get_summary()

    if args.output is not None:
        prep.save(args.output, file_type=args.output_format)

    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Run DDQN training from a YAML configuration."""
    _configure_runtime(args)

    from autorec.factory import environment_and_agent_builder
    from autorec.utils import set_global_seed

    set_global_seed(args.seed, deterministic_ops=not args.non_deterministic)

    config = _read_config_with_overrides(args.config, args.output_dir)
    _, _, agent = environment_and_agent_builder(config)
    agent.train()

    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation or inference for selected EIS rows."""
    _configure_runtime(args)

    from autorec.factory import environment_and_agent_builder
    from autorec.utils import save_evaluation_results, set_global_seed

    set_global_seed(args.seed, deterministic_ops=not args.non_deterministic)

    config = _read_config_with_overrides(args.config, args.output_dir)
    _, _, agent = environment_and_agent_builder(config)
    if args.model is not None:
        agent.load_model(args.model)

    results = agent.eval_batch_eis(
        eis_indices=_parse_indices(args.indices),
        max_actions=args.max_actions,
        verbose=args.verbose,
        use_eval_env=not args.use_training_env,
        gif_generation=args.gif_generation,
        all_rows=args.all_rows,
        num_samples=args.num_samples,
    )

    output_dir = args.output_dir or Path(config["agent"].get("save_dir", "."))
    if output_dir is not None:
        run_id = args.run_id or _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_evaluation_results(results, run_id, output_dir)

    return 0


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add runtime configuration options shared by all CLI subcommands."""
    parser.add_argument(
        "--threads",
        default=1,
        help="Thread count for BLAS/OpenMP/NumExpr runtime pools.",
    )
    parser.add_argument(
        "--skip-autoeis-warmup",
        action="store_true",
        help="Skip AutoEIS/Julia warmup during runtime setup.",
    )
    parser.add_argument(
        "--show-tf-logs",
        action="store_true",
        help="Do not suppress TensorFlow logs.",
    )


def _add_seed_args(parser: argparse.ArgumentParser) -> None:
    """Add reproducibility options shared by stochastic CLI subcommands."""
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Do not request deterministic TensorFlow operations.",
    )


def _add_output_dir_arg(parser: argparse.ArgumentParser) -> None:
    """Add the output directory override used by config-based commands."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for command artifacts; overrides agent.save_dir from YAML.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level AutoREC argument parser."""
    parser = argparse.ArgumentParser(
        prog="autorec",
        description="Run AutoREC data preparation, training, and inference workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Load or process EIS data and optionally save the validated dataset.",
    )
    preprocess.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Raw CSV folder for process mode, or processed CSV/pickle file for load mode.",
    )
    preprocess.add_argument(
        "--output",
        type=Path,
        help="Optional path for the validated processed dataset.",
    )
    preprocess.add_argument(
        "--mode",
        choices=("process", "load"),
        default="process",
        help="Whether to process raw CSV files or validate an existing dataset.",
    )
    preprocess.add_argument(
        "--output-format",
        choices=("pickle", "csv"),
        default="pickle",
        help="Output format used when --output is provided.",
    )
    preprocess.add_argument(
        "--evaluation",
        action="store_true",
        help="Require ground-truth circuit labels for evaluation datasets.",
    )
    preprocess.add_argument(
        "--eis-features",
        nargs="+",
        default=["ImZ", "phi", "mag", "nphi"],
        help="EIS features used to build flatten_Z.",
    )
    preprocess.add_argument(
        "--summary",
        action="store_true",
        help="Print a dataset summary after validation.",
    )
    _add_runtime_args(preprocess)
    preprocess.set_defaults(func=_run_preprocess)

    train = subparsers.add_parser(
        "train",
        help="Train an AutoREC DDQN agent from a YAML configuration.",
    )
    train.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML file containing environment and agent configuration.",
    )
    _add_output_dir_arg(train)
    _add_seed_args(train)
    _add_runtime_args(train)
    train.set_defaults(func=_run_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        aliases=["infer"],
        help="Load an optional model and evaluate/infer ECMs for dataset rows.",
    )
    evaluate.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML file containing environment and agent configuration.",
    )
    evaluate.add_argument(
        "--model",
        type=Path,
        help="Optional trained Keras model to load before evaluation.",
    )
    target = evaluate.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--indices",
        help="Comma-separated row positions to evaluate, for example '0,4,10'.",
    )
    target.add_argument(
        "--num-samples",
        type=int,
        help="Evaluate a random sample of this many rows.",
    )
    target.add_argument(
        "--all-rows",
        action="store_true",
        help="Evaluate every row in the selected dataset.",
    )
    evaluate.add_argument(
        "--max-actions",
        type=int,
        help="Maximum actions per EIS row. Defaults to the agent action cap.",
    )
    evaluate.add_argument(
        "--use-training-env",
        action="store_true",
        help="Evaluate against the training environment instead of the eval environment.",
    )
    evaluate.add_argument(
        "--gif-generation",
        action="store_true",
        help="Generate circuit evolution GIFs during evaluation.",
    )
    evaluate.add_argument(
        "--run-id",
        help="Run identifier used in saved evaluation filenames.",
    )
    evaluate.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation progress for each row.",
    )
    _add_output_dir_arg(evaluate)
    _add_seed_args(evaluate)
    _add_runtime_args(evaluate)
    evaluate.set_defaults(func=_run_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
