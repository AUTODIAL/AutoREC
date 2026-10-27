# Command-line interface

The `autorec` entry point configures runtime settings before importing the
scientific stack. Use `autorec <command> --help` as the definitive option list.

## Preprocess

Process a directory of raw three-column spectra:

```bash
autorec preprocess \
  --input tutorials/EIS_raw_demo \
  --output data/processed_training_data.pkl \
  --mode process \
  --frequency-npoints 80 \
  --summary
```

Use `--perform-linkk-validation` to enable lin-KK preprocessing and
`--evaluation` to require `true_circuit` in the processed result. See
{doc}`data preparation <data>` before choosing interpolation bounds.

Validate an existing processed pickle without rewriting it:

```bash
autorec preprocess \
  --input data/examples/training_dataset.pkl \
  --mode load \
  --evaluation \
  --summary
```

## Train

```bash
autorec train \
  --config configs_yaml/examples/demo_pipeline_config.yaml \
  --output-dir runs/demo_train \
  --seed 42
```

`--output-dir` overrides `agent.save_dir` from YAML. Training performs real RL
and circuit-fitting work; the demonstration's 50 trials are for workflow
inspection, not a performance claim.

## Evaluate or infer

`infer` is an alias for `evaluate`. Select exactly one of `--indices`,
`--num-samples`, or `--all-rows`:

```bash
autorec evaluate \
  --config configs_yaml/examples/demo_pipeline_config.yaml \
  --model models/examples/agent_trained.keras \
  --indices 0,4,10 \
  --max-actions 24 \
  --use-training-env \
  --output-dir results/evaluate_demo \
  --run-id evaluate_demo
```

The demo YAML defines no separate evaluation environment, so the agent falls
back to its training environment; `--use-training-env` makes that selection
explicit. The action limit is required because the demo has `action_cap: null`.
`--model` overrides the YAML model setting. Without `--run-id`, filenames
receive a timestamp.

`--gif-generation` creates additional visualization work and files. Start
without it when checking a new setup.

## Shared runtime controls

- `--threads` sets BLAS, OpenMP, and NumExpr thread-pool limits before imports.
- `--seed` seeds stochastic training/evaluation commands (default `42`).
- `--non-deterministic` stops requesting deterministic TensorFlow operations.
- `--skip-autoeis-warmup` avoids the explicit AutoEIS/Julia warmup, but a later
  fit can still initialize those dependencies.
- `--show-tf-logs` restores TensorFlow messages suppressed by default.

A seed is useful for comparison, but it is not a cross-platform reproducibility
guarantee. Record the environment, dependency versions, dataset, configuration,
and model with results.
