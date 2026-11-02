# Continuous FIM reward calculated on simplified circuits

This isolated experiment inherits from AutoREC's standard `EIS_ECM_Env`. Circuit
fitting, observations, cache identity, and returned circuits retain the base behavior.
For every successful fit, the environment:

1. Simplifies the fitted circuit and fitted parameters with
   `autorec.parser.simplify(circuit, parameters)`.
2. Computes the Fisher information matrix (FIM) in log-parameter space from
   unweighted complex residuals.
3. Normalizes the FIM eigenvalues by the largest eigenvalue.
4. Calculates a continuous identifiability penalty from those eigenvalues.
5. Calculates the depth penalty from the same simplified circuit.

For normalized eigenvalues `lambda_i`, the complexity penalty is

```text
penalty = alpha * mean(1 / (1 + (lambda_i / tau) ** beta))
```

The defaults are `tau = 1e-6` (`identifiability_thresh`), `beta = 2`
(`penalty_sharpness`), and `alpha = 2` (`penalty_scale`). An eigenvalue equal to
`tau` contributes `0.5`, values far below `tau` approach `1`, and values far above
it approach `0`. Taking the mean makes the score independent of the number of fitted
parameters, while `alpha` bounds the overall penalty to `[0, 2]`. Fit-quality and
depth terms remain separate parts of the reward.

After training, the runner evaluates every spectrum in the training, test, battery, and
experimental datasets described in the parent README.

From the AutoREC repository root, run locally with:

```bash
source ~/modules/AutoREC/autorec_env/bin/activate
python \
  example/reward_exploration/reward_calculated_with_continuous_fim_on_simplified_circuits/main_factory.py \
  --config configs_yaml/examples/pipeline_config.yaml \
  --output-dir \
  example/reward_exploration/reward_calculated_with_continuous_fim_on_simplified_circuits/calculation_output
```

To submit on SLURM, enter this directory and run:

```bash
sbatch submit_job.sh
```

The submission script retains the corrected configuration location
`~/modules/AutoREC/configs_yaml/examples/pipeline_config.yaml`, performs all four
evaluations, and writes outputs to `calculation_output/`.

## Hyperparameter tuning

From this directory, run `python tuning_runner.py` directly or submit
`tuning_runner_job.sh`. Every candidate uses the continuous-FIM reward environment. The
threshold (`1e-6`), sharpness (`2.0`), and scale (`2.0`) remain fixed in training and
validation and are not part of the search space. Training uses
`../data/training_dataset.pkl`, while promotion scores use
`../data/validation_dataset.pkl`. See the parent [reward exploration
README](../README.md#hyperparameter-tuning) for requirements, budgets, outputs, and the
separate role of `submit_job.sh`.
