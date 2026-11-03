# FIM reward calculated on simplified circuits

This experiment inherits from AutoREC's standard `EIS_ECM_Env`. Circuit fitting,
observations, cache identity, and returned circuits retain the base behavior. For every
successful fit, the environment:

1. Simplifies the fitted circuit and its fitted parameter dictionary with
   `autorec.parser.simplify(circuit, parameters)`.
2. Computes the Fisher information matrix in log-parameter space from unweighted complex
   residuals.
3. Normalizes the FIM eigenvalues by the largest eigenvalue.
4. Uses the number of eigenvalues below `identifiability_thresh` as the complexity penalty.
5. Calculates the depth penalty from the same simplified circuit.

The default identifiability threshold is `1e-6`.

After training, the runner evaluates every spectrum in the training, test, battery, and
experimental datasets described in the parent README.

From the AutoREC repository root, run locally with:

```bash
source ./autorec_env/bin/activate
python \
  example/reward_exploration/reward_calculated_with_fim_on_simplified_circuits/main_factory.py \
  --config configs_yaml/examples/pipeline_config.yaml \
  --output-dir \
  example/reward_exploration/reward_calculated_with_fim_on_simplified_circuits/calculation_output
```

To submit on SLURM, enter this directory and run:

```bash
sbatch submit_job.sh
```

The SLURM job uses the same resources and pipeline configuration as the baseline and
simplified-component-penalty experiments, activates `$HOME/eis_env`, performs
all four evaluations, and writes outputs to `calculation_output/`.

## Hyperparameter tuning

From this directory, run `python tuning_runner.py` directly or submit
`tuning_runner_job.sh`. Every candidate uses the discrete-FIM reward environment. The
identifiability threshold remains fixed at `1e-6` in both training and validation and is not
part of the search space. Training uses `../../data/training_dataset.pkl`, while promotion
scores use `../../data/validation_dataset.pkl`. See the parent [reward exploration
README](../README.md#hyperparameter-tuning) for requirements, budgets, outputs, and the
separate role of `submit_job.sh`.
