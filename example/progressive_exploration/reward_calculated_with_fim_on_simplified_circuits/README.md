# Progressive discrete-FIM reward

This case reuses the discrete-FIM reward environment from
[`../../reward_exploration/reward_calculated_with_fim_on_simplified_circuits/`](../../reward_exploration/reward_calculated_with_fim_on_simplified_circuits/).
After a successful fit, it simplifies the fitted circuit and parameters, builds an
unweighted complex-residual FIM in log-parameter space, normalizes its eigenvalues, and
counts values below `identifiability_thresh` as the complexity penalty. The default
threshold is `1e-6`; nesting depth comes from the same simplified circuit.

The FIM is calculated from all original measurements in the active cumulative window, not
from the interpolated neural-network observation.

Run the 50-trial demo from the repository root:

```bash
source ./autorec_env/bin/activate
python \
  example/progressive_exploration/reward_calculated_with_fim_on_simplified_circuits/main_factory.py
```

Use `submit_job.sh` for the long Slurm calculation. Use `python tuning_runner.py` or
`tuning_runner_job.sh` for checkpointed multi-fidelity tuning. The FIM threshold remains
fixed and is not in `tuning_search_space.yaml`.

See the parent [progressive reward exploration README](../README.md) for configuration,
outputs, checkpoint constraints, and scientific comparison caveats.
