# Baseline AutoREC calculation

This folder provides the baseline comparison for
`reward_calculated_on_simplified_circuits/`. It uses AutoREC's standard
`pipeline_builder` and `autorec.environment.EIS_ECM_Env`, so complexity penalties are
calculated from the raw circuit proposed by the agent.

The baseline and simplified-reward calculations use the same configuration, random seed,
training workflow, and evaluation behavior. Evaluation runs on every spectrum in the
training, test, battery, and experimental datasets described in the parent README.

From this directory, run the 10,000-trial exploration with:

```bash
python main_factory.py
```

To submit on SLURM, enter this directory and run:

```bash
sbatch submit_job.sh
```

The runner and SLURM job use `../pipeline_config.yaml`, activate the configured Python
environment, perform all four evaluations, and write outputs to `calculation_output/`.

## Hyperparameter tuning

From this directory, run `python tuning_runner.py` to orchestrate the baseline reward search
directly, or `sbatch tuning_runner_job.sh` to run the orchestrator as a low-resource Slurm
job. The tuning configuration uses `../../data/training_dataset.pkl` for training and
`../../data/validation_dataset.pkl` for promotion scores. It does not use held-out test,
battery, or experimental results. See the parent [reward exploration
README](../README.md#hyperparameter-tuning) for budgets, outputs, requirements, and the
difference from `submit_job.sh`.
