# Reward calculated on simplified circuits

The experimental `EIS_ECM_Env` subclasses `autorec.environment.EIS_ECM_Env` and overrides
only its reward-calculation hook. Circuit fitting, observations, cached identities, and
reported circuits remain unchanged. For a successful fit, the override first applies
`autorec.parser.simplify(circuit)` and uses that structurally simplified circuit to
calculate both the component-count penalty and the nesting-depth penalty.

From this directory, run the 10,000-trial exploration with:

```bash
python main_factory.py
```

The runner accepts `--config` and `--output-dir`. It evaluates every spectrum in the
training, test, battery, and experimental datasets described in the parent README. By
default, results are written to `demo_output/` beside the experimental runner so they do not
mix with the standard example's outputs.

For SLURM, review the account and resource directives in `submit_job.sh`, then submit from
this directory:

```bash
sbatch submit_job.sh
```

The runner and job use `../pipeline_config.yaml` (10,000 training trials), activate the
configured Python environment, perform all four evaluations, and write calculation artifacts
to `calculation_output/`. The SLURM log is written beside the submission script.

## Hyperparameter tuning

From this directory, run `python tuning_runner.py` directly or submit
`tuning_runner_job.sh`. Every candidate uses this directory's simplified-circuit reward
environment; only the common RL hyperparameters in `tuning_search_space.yaml` are varied.
Training uses `../../data/training_dataset.pkl`, while promotion scores use
`../../data/validation_dataset.pkl`. See the parent [reward exploration
README](../README.md#hyperparameter-tuning) for requirements, budgets, outputs, and the
separate role of `submit_job.sh`.
