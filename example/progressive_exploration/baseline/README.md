# Progressive baseline reward

This case uses AutoREC's standard `EIS_ECM_Env` reward while presenting every spectrum in
three cumulative frequency windows. Here, "baseline" means the unmodified reward under the
progressive workflow; the full-spectrum baseline remains in
[`../../reward_exploration/baseline/`](../../reward_exploration/baseline/).

From the repository root, run the 50-trial demo:

```bash
source ./autorec_env/bin/activate
python example/progressive_exploration/baseline/main_factory.py
```

Select `example/progressive_exploration/configs/pipeline_config.yaml` through `--config`
for the long calculation, or submit `submit_job.sh` from this directory. Use
`python tuning_runner.py` (or
`sbatch tuning_runner_job.sh`) for checkpointed multi-fidelity tuning. Training uses the
shared reward-exploration training pickle, while promotion scores use its validation
pickle.

See the parent [progressive reward exploration README](../README.md) for the exact window,
observation, fitting-data, output, checkpoint, and comparison contracts.
