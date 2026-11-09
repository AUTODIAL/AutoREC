# Multi-fidelity hyperparameter tuning

This example tunes AutoREC with [Optuna](https://optuna.org/) and Slurm without training
every suggested configuration for the full 10,000-trial budget. Training is split into
checkpointed 1,000-trial sessions. Unpromising configurations are stopped early, while
promoted configurations resume from their most recent complete checkpoint.

The default policy is synchronous successive halving:

| Cumulative trials | Action |
|---:|---|
| 1,000 | Score every candidate; continue all candidates |
| 2,000 | Keep the best half |
| 3,000 | Score the survivors |
| 4,000 | Keep the best half |
| 5,000–7,000 | Score the survivors every 1,000 trials |
| 8,000 | Keep the best half |
| 9,000–10,000 | Score the survivors; 10,000 is the maximum budget |

The baseline hyperparameters from `base_config.yaml` are protected from pruning and always
run to 10,000 trials. Suggested configurations are ranked only against other candidates in
the same synchronous batch. Ties are resolved deterministically by Optuna trial number.

## Data separation and objective

The `data/` directory has three processed datasets:

- `training_dataset.pkl` is used for RL interactions and the learning-curve component.
- `validation_dataset.pkl` is used for success-rate and exact-circuit evaluation at every
  fidelity stage.
- `test_dataset.pkl` is reserved for final assessment after hyperparameter selection. The
  tuning scripts never load it.

The training and validation paths in `base_config.yaml` are relative to the directory from
which the calculation runs. Generated candidate configurations preserve those relative
paths. Each generated Slurm script explicitly changes to the directory containing
`main_factory.py` before starting Python, so copying the complete
`example/hyperparameter_tuning/` directory to any shared location also copies the data root
used by `./data/...`.

`tuning_utils.compute_score` combines the mean episodic cumulative reward over the most
recent 1,000 training episodes, validation success rate, and validation exact-circuit rate.
Because the validation set is consulted repeatedly, it must not be reported as an unbiased
test result.

## Requirements

Install the additional example dependencies:

```bash
pip install -r requirements.txt
```

The orchestrator assumes a Slurm cluster. Cluster-specific module and virtual-environment
settings are defined near the top of `tuning_utils.py`; adjust them for your system. Running
the orchestrator itself as a Slurm job additionally requires that its compute node can run
`sbatch` and `sacct` and can access storage shared with the child training jobs.

Checkpoint continuation currently has a strict deterministic CPU runtime contract. Each
Slurm job disables GPUs, fixes thread counts, and enables deterministic TensorFlow
operations. A checkpoint can only be restored with the same AutoREC source and supported
numerical-library/runtime metadata. See `docs/checkpointing.md` for the complete contract.

## Running the workflow directly on a login node

From this directory, run:

```bash
python runner.py \
  --target-dir ./hyperparameter_tuning \
  --base-config ./base_config.yaml \
  --search-space ./search_space.yaml \
  --num-initial 5 \
  --num-iterations 10 \
  --batch-size 5 \
  --max-trials 10000 \
  --session-trials 1000 \
  --grace-trials 2000 \
  --reduction-factor 2
```

`runner.py` executes `sbatch` and `sacct` directly on the node where it is running. This
supports both direct login-node execution and clusters that permit job submission from an
allocated compute node.

By default, a temporary `sacct` failure is retried for one hour. Adjust that behavior with
`--slurm-poll-seconds` and `--slurm-accounting-timeout`. The monitor accepts a job only when
Slurm reports both `COMPLETED` and exit code `0:0`.

## Running the orchestrator as a Slurm job

`runner_job.sh` requests one core, 16 GB of memory, and seven days for the mostly idle
orchestration process. Before submission, edit the user-configuration variables near the top
of the script. These include the result and configuration paths, Optuna batch sizes, and
multi-fidelity budgets. Then submit it from the login node without passing runner arguments
on the command line:

```bash
sbatch runner_job.sh
```

Submit from the `example/hyperparameter_tuning/` directory. Slurm runs a copied job script
from an internal spool directory, so the wrapper changes back to `$SLURM_SUBMIT_DIR` before
starting `runner.py`. Relative result and configuration paths are therefore resolved from
the directory where `sbatch runner_job.sh` was invoked.

Before starting a long run, verify these cluster-specific requirements:

1. The compute node can run `sbatch` and `sacct` directly.
2. The repository, generated configurations, job scripts, checkpoints, and result directory
   use absolute paths on storage visible at the same paths from the orchestrator and training
   compute nodes.
3. The `#SBATCH` directives in `runner_job.sh`, including the seven-day limit, account, and
   node constraint, are accepted by the cluster. They currently match the training-job
   template and should be edited for a different cluster.
4. `MODULE_SAVELIST_NAME` and `PYTHON_ENV_PATH` in `runner_job.sh` match the cluster. They are
   currently set to `default` and `$HOME/autorec_env`.

Cancelling the orchestrator job does not automatically cancel child jobs that it already
submitted. Their IDs and statuses are retained in `slurm_jobs.json`, and restarting the
orchestrator reattaches to jobs still recorded as submitted instead of duplicating them.

Run `python runner.py --help` for the complete argument reference. `max-trials` must be
divisible by `session-trials`. Every geometric promotion budget
(`grace-trials`, then repeated multiplication by `reduction-factor`) must coincide with a
session boundary. `agent.num_trials` in the base configuration must equal
`--session-trials`.

## Restarting an interrupted tuning run

Restart `runner.py` or resubmit `runner_job.sh` with the same target directory, base
configuration, search space, and budget settings. The workflow then:

1. skips stages whose checkpoint and both rung-specific evaluation files are complete;
2. reuses registered child jobs whose final Slurm result has not yet been observed;
3. resumes an incomplete stage from its newest eligible checkpoint; and
4. resubmits no job that Slurm has confirmed failed unless you explicitly pass
   `--retry-failed-jobs` (or set `RETRY_FAILED_JOBS=true` in `runner_job.sh`).

Inspect the failed job log and correct its cause before enabling a retry. The registry keeps
the prior attempt as well as the replacement job ID.

For a run started with an older copy of this example that did not create `slurm_jobs.json`,
first confirm that all previously submitted child jobs are terminal. The updated runner can
reuse their completed artifacts, but it cannot discover unregistered jobs that are still
running. For example:

```bash
squeue -j 12345,12346,12347 -o "%.18i %.20T %.10M %R"
sacct -X -j 12345,12346,12347 --format=JobIDRaw,State,ExitCode
```

Replace those illustrative IDs with the IDs from the earlier runner output. Do not run old
and updated orchestrators against the same target directory concurrently. These
orchestration changes do not alter AutoREC checkpoint contents or source hashes, so valid
checkpoints produced by the same AutoREC source/runtime remain usable.

## How continuation is kept reproducible

`main_factory.py` performs exactly one cumulative stage. For example:

```bash
python main_factory.py --config configs/config_0001.yaml --target-trials 3000
```

The worker finds the newest checkpoint at or below 3,000 trials, restores it, trains only
the missing trials, and writes `checkpoints/trial-00003000/`. It saves the checkpoint
*before* validation. Validation changes RNG, environment, and cache state, so the next stage
always restores the immutable pre-validation checkpoint rather than continuing from the
evaluated in-memory agent. This keeps validation from changing the later training path.

The 1,000-trial session length must not redefine schedules intended for the full experiment.
Consequently, `base_config.yaml` uses the explicit cumulative schedule values
`start_jump: 4000` and `beta_anneal_trials: 5950`; these match the former default schedule
for a 10,000-trial run. `anneal_fraction` remains `null` for legacy compatibility.

## Outputs

Each unique configuration has a stable ID:

```text
hyperparameter_tuning/
├── configs/
│   ├── config_0000.yaml
│   └── config_0001.yaml
├── run_0000/
│   ├── checkpoints/
│   │   ├── trial-00001000/
│   │   └── ...
│   ├── eval_results_trial-00001000.pkl
│   ├── eval_results_trial-00001000.csv
│   ├── eval_results.pkl
│   ├── eval_results.csv
│   ├── statistical_analysis.pkl
│   ├── models.tar.gz
│   ├── fit_plots.tar.gz
│   └── .archive_complete
├── slurm_jobs.json
├── optuna_trials.csv
├── rung_scores.csv
├── optimization_history.png
└── param_importances.png
```

The trial-specific evaluation files preserve the score used at each rung.
`eval_results.pkl` and `eval_results.csv` are refreshed as compatibility aliases for the
latest completed stage. `rung_scores.csv` records each objective component, promotion or
pruning status, Optuna trial number, configuration ID, and cumulative budget.

The worker is idempotent for a completed stage. If a checkpoint exists without evaluation
output, it performs validation without retraining. Slurm failures are reported as errors.

After a candidate is pruned or reaches 10,000 trials, the runner submits a separate,
low-resource archive job. Promoted candidates are not archived because the next training
stage may still need their ordinary output directories. The archive job validates temporary
`models.tar.gz` and `fit_plots.tar.gz` files before removing the corresponding directories,
and then writes `.archive_complete`. It is safe to rerun after interruption; ambiguous
pre-existing source/archive combinations fail rather than overwrite data. Checkpoints are
left unarchived so continuation and inspection remain direct.

`slurm_jobs.json` is written atomically and records training and archive attempts, job IDs,
script/log paths, terminal states, and exit codes. Keep this file with the result directory.

Early scores can be noisy, and successive halving can discard a configuration that would
have improved late in training. The grace budget and reduction factor therefore represent a
compute-versus-selection-risk tradeoff; the defaults are a starting policy, not a scientific
guarantee that the globally best 10,000-trial configuration will survive.
