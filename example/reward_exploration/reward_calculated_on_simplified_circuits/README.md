# Reward calculated on simplified circuits

The experimental `EIS_ECM_Env` subclasses `autorec.environment.EIS_ECM_Env` and overrides
only its reward-calculation hook. Circuit fitting, observations, cached identities, and
reported circuits remain unchanged. For a successful fit, the override first applies
`autorec.parser.simplify(circuit)` and uses that structurally simplified circuit to
calculate both the component-count penalty and the nesting-depth penalty.

From the AutoREC repository root, run:

```bash
source ./autorec_env/bin/activate
python example/reward_exploration/reward_calculated_on_simplified_circuits/main_factory.py
```

The runner accepts the same `--config` and `--run-mode` options as
`example/main_factory.py`. The default `full` mode evaluates every spectrum in the training
dataset; use `--run-mode quick` for a random sample of 20 spectra. The runner also accepts
`--output-dir`; by default, results are written to `demo_output/` beside the experimental
runner so they do not mix with the standard example's outputs.

For SLURM, review the account and resource directives in `submit_job.sh`, then submit from
this directory:

```bash
sbatch submit_job.sh
```

The job uses the repository's `configs_yaml/examples/pipeline_config.yaml` (currently 10,000
training trials), activates `$HOME/eis_env`, performs full-spectrum evaluation, and writes
calculation artifacts to `calculation_output/`. The SLURM log is written beside the
submission script.
