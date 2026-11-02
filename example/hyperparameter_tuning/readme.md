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
  --target-dir ./hyperparameter_tuning_results \
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

The wrapper runs the equivalent of:

```bash
python runner.py \
  --target-dir "$TARGET_DIR" \
  --base-config "$BASE_CONFIG" \
  --search-space "$SEARCH_SPACE" \
  [remaining configured arguments]
```

Nested `sbatch` submissions and `sacct` status queries run directly on the allocated compute
node. This requires a cluster that permits Slurm client commands from compute nodes.

Before starting a long run, verify these cluster-specific requirements:

1. The compute node can run `sbatch` and `sacct` directly.
2. The repository, generated configurations, job scripts, checkpoints, and result directory
   use absolute paths on storage visible at the same paths from the orchestrator and training
   compute nodes.
3. The `#SBATCH` directives in `runner_job.sh`, including the seven-day limit, account, and
   node constraint, are accepted by the cluster. They currently match the training-job
   template and should be edited for a different cluster.
4. `MODULE_SAVELIST_NAME` and `PYTHON_ENV_PATH` match the cluster. They default to `default`
   and `$HOME/autorec_env`, and may be exported when submitting the wrapper job.

Cancelling the orchestrator job does not automatically cancel child jobs that it already
submitted; inspect and cancel those Slurm jobs separately if needed.

The arguments controlling candidate generation are:

- `--num-initial`: size of the first group, including the protected baseline.
- `--num-iterations`: total number of groups, including the initial group.
- `--batch-size`: number of suggestions in each later group.

The multi-fidelity arguments are:

- `--max-trials`: maximum cumulative budget for a promoted candidate.
- `--session-trials`: trials added before each checkpoint and score.
- `--grace-trials`: first cumulative budget at which pruning is allowed.
- `--reduction-factor`: at each promotion budget, retain
  `ceil(candidate_count / reduction_factor)` candidates.

`max-trials` must be divisible by `session-trials`. Every geometric promotion budget
(`grace-trials`, then repeated multiplication by `reduction-factor`) must coincide with a
session boundary. `agent.num_trials` in the base configuration must equal
`--session-trials`.

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
hyperparameter_tuning_results/
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
│   └── statistical_analysis.pkl
├── optuna_trials.csv
├── rung_scores.csv
├── optimization_history.png
└── param_importances.png
```

The trial-specific evaluation files preserve the score used at each rung.
`eval_results.pkl` and `eval_results.csv` are refreshed as compatibility aliases for the
latest completed stage. `rung_scores.csv` records each objective component, promotion or
pruning status, Optuna trial number, configuration ID, and cumulative budget.

The worker is idempotent for a stage whose checkpoint and both evaluation files already
exist. If a checkpoint exists but its evaluation output does not, the worker restores that
checkpoint and performs validation without retraining. Slurm failures are reported as
errors rather than treated as completed trials.

## Files

- `runner.py`: proposes candidates, submits each fidelity stage, performs synchronous
  promotion, reports final/pruned trials to Optuna, and exports summaries.
- `runner_job.sh`: long-running, low-resource Slurm wrapper for `runner.py`.
- `main_factory.py`: trains or restores one candidate to one cumulative trial target, then
  evaluates the validation dataset.
- `tuning_utils.py`: configuration handling, Slurm submission/monitoring, promotion helpers,
  and objective computation.
- `base_config.yaml`: training/validation pipeline and untuned baseline values.
- `search_space.yaml`: hyperparameters and Optuna bounds.

Early scores can be noisy, and successive halving can discard a configuration that would
have improved late in training. The grace budget and reduction factor therefore represent a
compute-versus-selection-risk tradeoff; the defaults are a starting policy, not a scientific
guarantee that the globally best 10,000-trial configuration will survive.
