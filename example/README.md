# AutoREC examples

This directory contains progressively more advanced examples for constructing, training,
continuing, evaluating, and tuning AutoREC agents. This index describes only examples that
are tracked by Git.

Activate the AutoREC environment and install the package before running an example. Commands
for top-level scripts below assume the current directory is `example/`.

## Choosing an example

| Example | Purpose | Default or typical cost |
|---|---|---|
| [`main.py`](main.py) | Construct the data preparation, environment, and agent directly in Python | Quick mode trains for 50 trials |
| [`main_factory.py`](main_factory.py) | Build the same pipeline from YAML configuration | Defaults to the 50-trial demo configuration |
| [`checkpoint/`](checkpoint/) | Compare continuous, bounded, complete-checkpoint, and model-only continuation | Four controlled 1,000-trial scenarios; intentionally expensive |
| [`hyperparameter_tuning/`](hyperparameter_tuning/) | Run checkpointed multi-fidelity Optuna tuning on Slurm | Potentially expensive HPC workflow |

## Direct Python construction

[`main.py`](main.py) demonstrates the individual AutoREC APIs without a YAML configuration:

1. Load and validate an EIS dataset.
2. Construct `EIS_ECM_Env`.
3. Construct and train `DDQN_ECM`.
4. Evaluate a deterministic sample and save the results.

The script defaults to `RUN_MODE = "quick"`, which trains for 50 trials. Edit `RUN_MODE` in
the script to select its longer hard-coded workflow.

```bash
python main.py
```

## Configuration-driven training

[`main_factory.py`](main_factory.py) demonstrates the recommended configuration-driven
pipeline. With no arguments it uses
`configs_yaml/examples/demo_pipeline_config.yaml`, which trains for 50 trials:

```bash
python main_factory.py
```

Select another configuration explicitly when needed:

```bash
python main_factory.py \
  --config ../configs_yaml/examples/pipeline_config.yaml \
  --run-mode full
```

`--run-mode` controls the evaluation sample size; training length comes from
`agent.num_trials` in the selected YAML file. The standard `pipeline_config.yaml` is the
long-training configuration and currently specifies 10,000 trials.

## Checkpoint equivalence experiment

The [`checkpoint/`](checkpoint/) workflow compares a continuous 1,000-trial run with retained
same-process sessions, complete-checkpoint fresh processes, and deliberately incomplete
model-only restarts. It includes tools that compare complete state and run fresh-process agent
evaluation with exact equality checks where equality is expected.

See [`checkpoint/README.md`](checkpoint/README.md) for commands, output structure, comparison
categories, and reproducibility limitations.

## Multi-fidelity hyperparameter tuning

The [`hyperparameter_tuning/`](hyperparameter_tuning/) workflow uses Optuna, Slurm, and
complete checkpoints to avoid training every candidate for the maximum 10,000-trial budget.
Candidates are evaluated every 1,000 trials and synchronously pruned at configured promotion
budgets. It supports:

- direct orchestration from a login node;
- a low-resource Slurm job that runs the orchestrator on a compute node;
- direct child-job submission with restart-safe job tracking;
- separate terminal-candidate jobs that archive model and fit-plot directories;
- separate training, validation, and held-out test datasets;
- rung-level objective and promotion reports.

See [`hyperparameter_tuning/readme.md`](hyperparameter_tuning/readme.md) for configuration,
recovery behavior, cluster requirements, and submission commands. Its training, validation,
and held-out test datasets are tracked in `hyperparameter_tuning/data/`, so the complete
directory can be copied to shared HPC storage.

## Runtime expectations

Several workflows initialize AutoEIS/Julia, TensorFlow, and numerical thread pools. Their
runtime setup should occur before importing the scientific stack, as shown in the scripts.
Real training and circuit fitting can be substantially slower than CLI or unit-test checks.
Use demo configurations for smoke tests and the long configurations only for deliberate
training runs.
