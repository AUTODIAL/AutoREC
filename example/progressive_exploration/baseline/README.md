# Progressive baseline reward

This case uses AutoREC's standard `EIS_ECM_Env` reward while presenting every spectrum in
three cumulative frequency windows. Here, "baseline" means the unmodified reward under the
progressive workflow.

Run the 10,000-trial exploration from this directory:

```bash
python main_factory.py
```

This uses `../pipeline_config.yaml` by default. Alternatively, submit
`submit_job.sh` from this directory. Use `python tuning_runner.py` (or
`sbatch tuning_runner_job.sh`) for checkpointed multi-fidelity tuning. Training and
validation use the corresponding pickles in the shared `../../data/` directory.

See the parent [progressive reward exploration README](../README.md) for the exact window,
observation, fitting-data, output, checkpoint, and comparison contracts.
