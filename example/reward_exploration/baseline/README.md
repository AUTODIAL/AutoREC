# Baseline AutoREC calculation

This folder provides the baseline comparison for
`reward_calculated_on_simplified_circuits/`. It uses AutoREC's standard
`pipeline_builder` and `autorec.environment.EIS_ECM_Env`, so complexity penalties are
calculated from the raw circuit proposed by the agent.

The baseline and simplified-reward calculations use the same configuration, random seed,
training workflow, and evaluation behavior. In the default `full` mode, evaluation runs on
every spectrum in the training dataset.

From the AutoREC repository root, run locally with:

```bash
source ./autorec_env/bin/activate
python example/reward_exploration/baseline/main_factory.py \
  --config configs_yaml/examples/pipeline_config.yaml \
  --run-mode full \
  --output-dir example/reward_exploration/baseline/calculation_output
```

To submit on SLURM, enter this directory and run:

```bash
sbatch submit_job.sh
```

The SLURM job activates `$HOME/eis_env`, uses the repository's
`configs_yaml/examples/pipeline_config.yaml` (currently 10,000 training trials), performs
full-spectrum evaluation, and writes outputs to `calculation_output/`.
