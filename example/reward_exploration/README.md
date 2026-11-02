# Reward exploration examples

These examples compare alternative reward complexity penalties while keeping the
AutoREC training and evaluation workflow as consistent as possible. Each method is
isolated in its own directory so that experimental reward definitions do not change the
package defaults.

## Experiments

| Example | Complexity term for a successful fit |
| --- | --- |
| [`baseline/`](baseline/) | Standard AutoREC component-count and depth penalties calculated from the proposed circuit. |
| [`reward_calculated_on_simplified_circuits/`](reward_calculated_on_simplified_circuits/) | Standard component-count and depth penalties calculated after structural circuit simplification. |
| [`reward_calculated_with_fim_on_simplified_circuits/`](reward_calculated_with_fim_on_simplified_circuits/) | Number of normalized Fisher information matrix (FIM) eigenvalues below an identifiability threshold, with depth calculated from the simplified circuit. |
| [`reward_calculated_with_continuous_fim_on_simplified_circuits/`](reward_calculated_with_continuous_fim_on_simplified_circuits/) | Bounded continuous penalty for weak normalized FIM directions, with depth calculated from the simplified circuit. |

The FIM examples use unweighted complex residuals and transform positive fitted
parameters into log space. See each example's README for its exact reward definition,
defaults, and assumptions.

## Running an example

Run commands from the AutoREC repository root. For a short workflow check, omit
`--config`; each runner then uses `demo_pipeline_config.yaml`, which currently trains for
50 trials. For example:

```bash
source ./autorec_env/bin/activate
python example/reward_exploration/baseline/main_factory.py --run-mode quick
```

For the full 10,000-trial configuration, provide the full pipeline configuration and a
dedicated output directory:

```bash
python example/reward_exploration/baseline/main_factory.py \
  --config configs_yaml/examples/pipeline_config.yaml \
  --run-mode full \
  --output-dir example/reward_exploration/baseline/calculation_output
```

Replace `baseline` in both paths with another experiment directory to run that method.
Use a different output directory for every experiment; an existing `dqn_model.keras` in
the selected directory causes the runner to load that model instead of training a new
one.

## Comparing methods

For a controlled comparison, use the same YAML configuration, datasets, random seed,
run mode, and software/hardware environment for every method. The runners set the global
seed to 42 before constructing the pipeline. The methods intentionally change the reward,
so their learned trajectories and final models are not expected to be numerically equal.

The included tests verify reward calculations, edge cases, and evaluation selection.
They do not establish that one reward produces better circuits or RL performance; that
requires replicated training runs and scientific analysis of the resulting models and
evaluation outputs.

## SLURM execution

Each directory includes a `submit_job.sh` intended to be submitted from that directory:

```bash
cd example/reward_exploration/baseline
sbatch submit_job.sh
```

Before submission, review the cluster-specific account, constraint, environment path,
resources, and repository location in the script. The scripts use the full pipeline
configuration, evaluate the full dataset, and write to `calculation_output/`. At the end
of a successful job, the `models/` and `fit_plots/` directories are archived as
`models.tar.gz` and `fit_plots.tar.gz`, then the original directories are removed to
reduce HPC file counts.
