# Progressive simplified-circuit reward

This case includes a local simplified-circuit reward environment. After a successful fit,
it structurally simplifies the proposed circuit and calculates component-count and
nesting-depth penalties from that simplified circuit. Spectrum selection and fixed-size
observation encoding are independent of this reward override.

Run the 10,000-trial exploration from this directory:

```bash
python main_factory.py
```

The runner accepts `--config` and `--output-dir`. For HPC execution, submit
`submit_job.sh` from this directory. Run `python tuning_runner.py` or submit
`tuning_runner_job.sh` for checkpointed multi-fidelity tuning; only the RL parameters in
`tuning_search_space.yaml` vary.

See the parent [progressive reward exploration README](../README.md) for the shared
three-window workflow, physical fitting-data boundary, outputs, and comparison caveats.
