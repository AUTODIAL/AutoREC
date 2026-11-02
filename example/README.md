# AutoREC examples

This directory contains progressively more advanced examples for constructing, training,
continuing, evaluating, tuning, and experimentally modifying AutoREC agents. This index
describes only examples that are tracked by Git.

Activate the AutoREC environment and install the package before running an example. Commands
for top-level scripts below assume the current directory is `example/`.

## Choosing an example

| Example | Purpose | Default or typical cost |
|---|---|---|
| [`main.py`](main.py) | Construct the data preparation, environment, and agent directly in Python | Quick mode trains for 50 trials |
| [`main_factory.py`](main_factory.py) | Build the same pipeline from YAML configuration | Defaults to the 50-trial demo configuration |
| [`bounded_training.py`](bounded_training.py) | Continue cumulative training for a fixed number of sessions in one Python process | Controlled by the selected configuration and CLI arguments |
| [`checkpoint_training.py`](checkpoint_training.py) | Save or resume complete training checkpoints across Python processes | One configurable session per invocation |
| [`checkpoint/`](checkpoint/) | Validate continuous versus fresh-process checkpoint-resumed equivalence | Full example compares 1,000 continuous trials with ten 100-trial sessions |
| [`hyperparameter_tuning/`](hyperparameter_tuning/) | Run checkpointed multi-fidelity Optuna tuning on Slurm | Potentially expensive HPC workflow |
| [`reward_exploration/`](reward_exploration/) | Compare the standard reward with simplified-circuit and FIM-based complexity penalties | Experimental workflows range from a 50-trial demo to 10,000-trial HPC runs |

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

## Bounded continuation in one process

[`bounded_training.py`](bounded_training.py) retains replay, histories, counters, schedules,
and neural state while running multiple sessions in one interpreter. It does not write a
complete checkpoint and cannot resume after the process exits.

```bash
python bounded_training.py \
  --config ../configs_yaml/examples/demo_pipeline_config.yaml \
  --sessions 3 \
  --trials-per-session 20
```

This is useful for testing cumulative continuation semantics without checkpoint I/O.

## General checkpoint training

[`checkpoint_training.py`](checkpoint_training.py) is a compact interface for one
checkpointed training session. To make one configured cumulative target automatically
recoverable after interruption, rerun this command as needed:

```bash
python checkpoint_training.py \
  --config ../configs_yaml/examples/demo_pipeline_config.yaml \
  --checkpoint-dir checkpoint_outputs \
  --checkpoint-every 10 \
  --resume-latest
```

A missing checkpoint starts fresh, a partial checkpoint runs only the remaining trials, and
an already reached target is a no-op. The explicit manual equivalent remains available:

```bash
python checkpoint_training.py \
  --resume checkpoint_outputs/trial-00000050 \
  --additional-trials 50 \
  --checkpoint-dir checkpoint_outputs
```

Checkpoint continuation requires the deterministic CPU/runtime conditions documented in
[`docs/checkpointing.md`](../docs/checkpointing.md).

## Checkpoint equivalence experiment

The [`checkpoint/`](checkpoint/) workflow compares a continuous 1,000-trial run with ten
fresh 100-trial processes. It includes tools that compare complete checkpoint state and run
fresh-process agent evaluation with exact equality checks.

See [`checkpoint/README.md`](checkpoint/README.md) for commands, output structure, comparison
categories, and reproducibility limitations.

## Multi-fidelity hyperparameter tuning

The [`hyperparameter_tuning/`](hyperparameter_tuning/) workflow uses Optuna, Slurm, and
complete checkpoints to avoid training every candidate for the maximum 10,000-trial budget.
Candidates are evaluated every 1,000 trials and synchronously pruned at configured promotion
budgets. It supports:

- direct orchestration from a login node;
- a low-resource Slurm job that runs the orchestrator on a compute node;
- SSH-routed child-job submission and monitoring;
- separate training, validation, and held-out test datasets;
- rung-level objective and promotion reports.

See [`hyperparameter_tuning/readme.md`](hyperparameter_tuning/readme.md) for configuration,
cluster requirements, and submission commands. Its dedicated dataset pickle files are local
and currently ignored by Git, so they must be provided separately in
`hyperparameter_tuning/data/`.

## Reward exploration

The [`reward_exploration/`](reward_exploration/) collection provides four isolated,
configuration-driven workflows:

- the standard AutoREC reward as a baseline;
- component-count and depth penalties calculated after circuit simplification;
- a discrete penalty based on weak Fisher information matrix (FIM) directions in the
  simplified circuit; and
- a bounded continuous penalty based on the same normalized FIM spectrum.

Use identical configurations, datasets, seeds, run modes, and execution environments when
comparing these methods. The included tests validate the reward calculations and workflow
mechanics, but they do not demonstrate that one reward produces better circuits or training
performance. See [`reward_exploration/README.md`](reward_exploration/README.md) for the
formulas, local commands, HPC workflow, and scientific validation caveats.

## Runtime expectations

Several workflows initialize AutoEIS/Julia, TensorFlow, and numerical thread pools. Their
runtime setup should occur before importing the scientific stack, as shown in the scripts.
Real training and circuit fitting can be substantially slower than CLI or unit-test checks.
Use demo configurations for smoke tests and the long configurations only for deliberate
training runs.
