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

From the AutoREC repository root, run locally with:

```bash
source ./autorec_env/bin/activate
python \
  example/reward_exploration/reward_calculated_with_fim_on_simplified_circuits/main_factory.py \
  --config configs_yaml/examples/pipeline_config.yaml \
  --run-mode full \
  --output-dir \
  example/reward_exploration/reward_calculated_with_fim_on_simplified_circuits/calculation_output
```

To submit on SLURM, enter this directory and run:

```bash
sbatch submit_job.sh
```

The SLURM job uses the same resources and pipeline configuration as the baseline and
simplified-component-penalty experiments, activates `$HOME/eis_env`, performs
full-spectrum evaluation, and writes outputs to `calculation_output/`.
