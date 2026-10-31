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
the orchestrator itself as a Slurm job additionally requires passwordless, non-interactive
SSH from its compute node to a login node and a filesystem shared by both nodes.

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

When `--login-node` is omitted, `runner.py` executes `sbatch` and `sacct` directly. This
preserves the original login-node workflow.

## Running the orchestrator as a Slurm job

`runner_job.sh` requests one core, 16 GB of memory, and seven days for the mostly idle
orchestration process. Before submission, edit the user-configuration variables near the top
of the script. These include the login-node hostname, result and configuration paths, Optuna
batch sizes, and multi-fidelity budgets. Then submit it from the login node without passing
runner arguments on the command line:

```bash
sbatch runner_job.sh
```

The wrapper runs the equivalent of:

```bash
python runner.py \
  --login-node "$LOGIN_NODE" \
  --target-dir "$TARGET_DIR" \
  --base-config "$BASE_CONFIG" \
  --search-space "$SEARCH_SPACE" \
  [remaining configured arguments]
```

With `--login-node`, nested `sbatch` submissions and `sacct` status queries are routed as
non-interactive SSH commands. For example, submission is effectively performed as
`ssh login1 "sbatch /shared/path/to/submit.sh"`. Direct command execution remains the
default when the option is absent.

Before starting a long run, verify these cluster-specific requirements:

1. The compute node can run `ssh -o BatchMode=yes login1 true` without a password, host-key
   prompt, or multi-factor prompt.
2. The repository, generated configurations, job scripts, checkpoints, and result directory
   use absolute paths on storage visible at the same paths from the compute and login nodes.
3. A non-interactive SSH session on the login node can find `sbatch` and `sacct` in `PATH`.
4. The `#SBATCH` directives in `runner_job.sh`, including the seven-day limit, account, and
   node constraint, are accepted by the cluster. They currently match the training-job
   template and should be edited for a different cluster.
5. `MODULE_SAVELIST_NAME` and `PYTHON_ENV_PATH` match the cluster. They default to `default`
   and `$HOME/autorec_env`, and may be exported when submitting the wrapper job.

SSH uses batch mode and a 30-second connection timeout so authentication problems fail
clearly instead of leaving the orchestrator waiting for interactive input. Cancelling the
orchestrator job does not automatically cancel child jobs that it already submitted; inspect
and cancel those Slurm jobs separately if needed.

The arguments controlling candidate generation are:

- `--num-initial`: size of the first group, including the protected baseline.
- `--num-iterations`: total number of groups, including the initial group.
- `--batch-size`: number of suggestions in each later group.
- `--login-node`: optional SSH host for Slurm submission and monitoring when `runner.py`
  itself runs on a compute node.

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
