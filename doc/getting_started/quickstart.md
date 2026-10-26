# Quick Start

This guide uses the dataset, trained model, and demonstration configuration
included in the repository. Complete the {doc}`installation` first, activate
`autorec_env`, and run each command from the repository root.

```{note}
The commands on this page perform real circuit fitting or reinforcement-learning
work. Their runtime depends on the machine and configuration. Sphinx displays
the examples without executing them during a documentation build.
```

## Evaluate the included model

The quickest complete workflow applies the example model to one row of the
included dataset:

```bash
autorec evaluate \
  --config configs_yaml/examples/demo_pipeline_config.yaml \
  --model models/examples/agent_trained.keras \
  --indices 0 \
  --max-actions 24 \
  --use-training-env \
  --output-dir runs/quickstart \
  --run-id quickstart
```

The demonstration configuration defines one dataset, so
`--use-training-env` makes that choice explicit. The command loads the model,
generates a candidate ECM, prints an evaluation summary, and writes
`eval_results_quickstart.csv` and `eval_results_quickstart.pkl` under
`runs/quickstart/`. Evaluation is stochastic; a single run is a functional
walkthrough rather than a performance benchmark.

## Prepare raw EIS data

Raw spectra are CSV files with exactly three columns: `freq`, `Z_real`, and
`Z_imag`. AutoREC recursively processes CSV files in the input directory. Run
the included small raw-data example with:

```bash
autorec preprocess \
  --input tutorials/EIS_raw_demo \
  --output processed_demo.pkl \
  --mode process \
  --summary
```

This command interpolates the spectra onto compatible frequency grids, creates
the flattened EIS features consumed by the agent, validates the resulting
dataset, and saves it as a pickle file. Use `--evaluation` when directory names
contain ground-truth circuit labels that should be required and validated.

For an already processed dataset, use `--mode load`. Optional interpolation and
lin-KK controls are listed by `autorec preprocess --help`. The
{doc}`data-preparation tutorial <../tutorials/data_preparation>` provides a
fuller walkthrough.

## Train from a YAML configuration

AutoREC's pipeline configuration contains `dataprep`, `environment`, and
`agent` sections. The included demonstration configuration runs 50 trials and
writes artifacts to the directory selected by `--output-dir`:

```bash
autorec train \
  --config configs_yaml/examples/demo_pipeline_config.yaml \
  --output-dir runs/quickstart-training
```

Training writes the final `dqn_model.keras` model, intermediate checkpoints,
metric CSV files, plots, and a model summary. Copy the demonstration YAML before
changing datasets or hyperparameters so the tracked example remains a stable
reference.

## Use the Python API

The factory API builds the same data-preparation, environment, and agent objects
from YAML. Configure the runtime before importing it:

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1, warmup_autoeis=True)

from autorec.factory import pipeline_builder

data_prep, data_prep_eval, env, env_eval, agent = pipeline_builder(
    "configs_yaml/examples/demo_pipeline_config.yaml"
)
agent.load_model("models/examples/agent_trained.keras")

results = agent.eval_batch_eis(
    eis_indices=[0],
    max_actions=24,
    use_eval_env=False,
    verbose=False,
)
print(results[["best_circuit", "best_reward", "found_solution"]])
```

`pipeline_builder` returns five objects: training data preparation, evaluation
data preparation, training environment, evaluation environment, and agent. When
the YAML does not define separate evaluation sections, the agent uses its
training environment as the evaluation fallback.

## Where to go next

- Browse the {doc}`notebook tutorials <../tutorials/index>` for detailed
  walkthroughs.
- Review the [example configurations](https://github.com/AUTODIAL/AutoREC/tree/main/configs_yaml/examples)
  before creating an experiment configuration.
- Use `autorec <command> --help` for the exact options accepted by
  `preprocess`, `train`, `evaluate`, and its `infer` alias.
