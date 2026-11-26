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

Run commands from the selected case directory so the shared `../../data/` paths have the
same meaning in every exploration case. With no `--config`, the runner uses the local
10,000-trial full-spectrum configuration:

```bash
cd example/reward_exploration/baseline
python main_factory.py
```

To choose the configuration and output directory explicitly, run:

```bash
python main_factory.py \
  --config ../pipeline_config.yaml \
  --output-dir calculation_output
```

Replace `baseline` with another experiment directory to run that method. Use a different
output directory for every experiment; an existing `dqn_model.keras` in the selected
directory causes the runner to load that model instead of training a new one.

## Evaluation datasets and outputs

Every runner performs four complete evaluations after training or loading its model:

1. `training`: the training dataset selected by the YAML configuration;
2. `test`: `../data/test_dataset.pkl`;
3. `battery`: `../data/battery_dataset.pkl`; and
4. `experimental`: `../data/experimental_dataset.pkl`.

The three external datasets must be available in `example/data/` on the
machine running the example. Each external dataset is loaded into a fresh environment with
the same environment settings as the configured evaluation or training environment. This
prevents circuit-cache entries from being shared between datasets.

Each evaluation writes `eval_results_<dataset>.csv` and
`eval_results_<dataset>.pkl` under the selected output directory's `results/` folder.
Its fit images are moved into `results/fit_plots_<dataset>/`, preventing spectra with the
same row number in different datasets from overwriting one another. Repeating an evaluation
replaces the corresponding dataset-specific fit-plot directory.

On a later invocation, a dataset evaluation is skipped when its CSV and pickle results both
exist and its fit plots are available either as `fit_plots_<dataset>/` or the archived
`fit_plots_<dataset>.tar.gz`. If any of those outputs are incomplete, that dataset is
evaluated again. Delete one of the required outputs to force reevaluation of a dataset.

## Comparing methods

For a controlled comparison, use the same YAML configuration, datasets, random seed, and
software/hardware environment for every method. The runners set the global seed to 42 before
constructing the pipeline, reset all global random generators to the configured agent seed
before each dataset evaluation, and evaluate every spectrum in all four datasets. The methods
intentionally change the reward, so their learned trajectories and final models are not
expected to be numerically equal.

The included tests verify reward calculations and edge cases. They do not establish that
one reward produces better circuits or RL performance; that requires replicated training
runs and scientific analysis of the resulting models and evaluation outputs.

## SLURM execution

Each directory includes a `submit_job.sh` intended to be submitted from that directory:

```bash
cd example/reward_exploration/baseline
sbatch submit_job.sh
```

Before submission, review the cluster-specific account, constraint, environment path,
resources, and repository location in the script. The scripts use the full pipeline
configuration from `../pipeline_config.yaml`, evaluate all four datasets, and write to
`calculation_output/`. At the end of a successful job, `models/` is archived as
`models.tar.gz`. Each `results/fit_plots_<dataset>/` directory is archived beside itself as
`fit_plots_<dataset>.tar.gz`. The original directories are then removed to reduce HPC file
counts.

The job changes to `SLURM_SUBMIT_DIR` before invoking Python. Submit from the selected case
directory as shown above; this remains reliable on clusters that execute the copied batch
script from a Slurm spool directory.

## Hyperparameter tuning

Each experiment directory contains a full-budget tuning workflow matching
`example/hyperparameter_tuning`. Every candidate trains for the 10,000 trials configured in
its local `tuning_base_config.yaml`; there is no early pruning or promotion. Complete
checkpoints are saved every 1,000 trials by default, allowing a failed candidate job to
resume from its latest valid checkpoint. The shared orchestrator is included in this
folder's local `tuning/` package, so a copied `reward_exploration/` directory remains
self-contained.

The datasets have deliberately separate roles:

- `../data/training_dataset.pkl` supplies RL training interactions and learning-curve rewards;
- `../data/validation_dataset.pkl` supplies the evaluation success rate used in the tuning
  score; and
- `../data/test_dataset.pkl` is held back for assessment after hyperparameter selection.

Tuning does not evaluate the test, battery, or experimental datasets. Those evaluations
remain part of the separate full-run workflow so that model selection does not use held-out
results.

Install the additional orchestration dependencies once:

```bash
pip install -r example/reward_exploration/tuning/requirements.txt
```

Run the orchestrator from the selected case directory on a Slurm login node:

```bash
cd example/reward_exploration/baseline
python tuning_runner.py
```

`tuning_runner.py --help` lists the target directory, batch-size, iteration, and checkpoint
interval options. By default, it runs one seeded baseline and four other initial candidates,
then nine batches of five candidates, for 50 full-budget candidates in total. The runner
submits and monitors compute jobs and should not itself be submitted as a Slurm job.

The two Slurm scripts serve different purposes:

- `submit_job.sh` trains one selected configuration and evaluates all four datasets.
- `tuning_runner.py` orchestrates many full-budget candidates and evaluates only the
  validation data needed for hyperparameter scoring.

Tuning outputs are written under the case's `hyperparameter_tuning/` directory. The main
summary files are `candidate_scores.csv`, `optuna_trials.csv`,
`optimization_history.png`, and `param_importances.png`; candidate directories contain
configurations, checkpoints, and final validation results.

If a child job is interrupted, rerun the same `python tuning_runner.py` command after fixing
the external cause. Completed candidates are skipped, while incomplete candidates restore
their newest checkpoint and train only the remaining trials. The wrappers depend on the
shared `tuning/` package and sibling `example/data/` directory; preserve that relative
layout when copying the workflow to HPC storage.
