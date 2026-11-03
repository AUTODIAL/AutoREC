# Progressive simplified-circuit reward

This case reuses the reward environment from
[`../../reward_exploration/reward_calculated_on_simplified_circuits/`](../../reward_exploration/reward_calculated_on_simplified_circuits/).
After a successful fit, it structurally simplifies the proposed circuit and calculates
component-count and nesting-depth penalties from that simplified circuit. Spectrum
selection and fixed-size observation encoding are independent of this reward override.

Run the 50-trial demo from the repository root:

```bash
source ./autorec_env/bin/activate
python \
  example/progressive_exploration/reward_calculated_on_simplified_circuits/main_factory.py
```

The runner accepts `--config` and `--output-dir`. For HPC execution, submit
`submit_job.sh` from this directory. Run `python tuning_runner.py` or submit
`tuning_runner_job.sh` for checkpointed multi-fidelity tuning; only the RL parameters in
`tuning_search_space.yaml` vary.

See the parent [progressive reward exploration README](../README.md) for the shared
three-window workflow, physical fitting-data boundary, outputs, and comparison caveats.
