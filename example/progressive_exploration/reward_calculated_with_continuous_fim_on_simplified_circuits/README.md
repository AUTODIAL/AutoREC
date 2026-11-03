# Progressive continuous-FIM reward

This case reuses the continuous-FIM reward environment from
[`../../reward_exploration/reward_calculated_with_continuous_fim_on_simplified_circuits/`](../../reward_exploration/reward_calculated_with_continuous_fim_on_simplified_circuits/).
After a successful fit, it simplifies the fitted circuit and parameters, builds an
unweighted complex-residual FIM in log-parameter space, and applies this bounded penalty to
the normalized eigenvalues `lambda_i`:

```text
penalty = alpha * mean(1 / (1 + (lambda_i / tau) ** beta))
```

The defaults are `tau = 1e-6` (`identifiability_thresh`), `beta = 2`
(`penalty_sharpness`), and `alpha = 2` (`penalty_scale`). Nesting depth comes from the same
simplified circuit. The FIM uses all original measurements in the active cumulative window,
not the interpolated neural-network observation.

Run the 50-trial demo from the repository root:

```bash
source ./autorec_env/bin/activate
python \
  example/progressive_exploration/reward_calculated_with_continuous_fim_on_simplified_circuits/main_factory.py
```

Use `submit_job.sh` for the long Slurm calculation. Use `python tuning_runner.py` or
`tuning_runner_job.sh` for checkpointed multi-fidelity tuning. The threshold, sharpness,
and scale remain fixed and are not in `tuning_search_space.yaml`.

See the parent [progressive reward exploration README](../README.md) for configuration,
outputs, checkpoint constraints, and scientific comparison caveats.
