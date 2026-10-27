# Open-ended checkpoint training and equivalence experiment

This example supports two related workflows:

- Train for one configured session, inspect the cumulative results, and decide later whether
  to continue from the generated checkpoint configuration.
- Compare one continuous 1,000-trial process with ten fresh 100-trial processes.

`agent.num_trials` is the number of trials performed by the current invocation. There is no
configured total horizon. The sequential configuration therefore contains `num_trials: 100`,
while the continuous validation configuration contains `num_trials: 1000`.

Both configurations use the same absolute cumulative schedules:

```yaml
start_jump: 400
beta_anneal_trials: 595
```

Those settings reproduce the beta schedule formerly obtained from a 1,000-trial run with
`anneal_fraction: 0.595`, without requiring training to stop at trial 1,000.

## Inspect and decide after each session

From the repository root, run one 100-trial session:

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=-1 \
autorec_env/bin/python example/checkpoint/main_factory.py \
  --config example/checkpoint/sequential_config.yaml
```

The process saves cumulative results, an immutable complete checkpoint, and a possible next
configuration:

```text
example/checkpoint/outputs/sequential/checkpoints/trial-00000100/
example/checkpoint/outputs/sequential/configs/resume-from-trial-00000100.yaml
```

Inspect or evaluate the checkpoint. If more training is useful, run the generated full YAML:

```bash
autorec_env/bin/python example/checkpoint/main_factory.py \
  --config example/checkpoint/outputs/sequential/configs/resume-from-trial-00000100.yaml
```

That invocation restores trial 100, performs the configured 100 additional trials, and
produces `trial-00000200` plus another optional continuation YAML. You can change only
`agent.num_trials`, output paths, and checkpoint paths between sessions. Changes to model,
optimizer, environment, replay, or schedule settings are rejected or remain checkpoint-owned.

Evaluate in a separate process so assessment cannot modify the resumable training state:

```bash
autorec_env/bin/python example/checkpoint/evaluate_checkpoint.py \
  --checkpoint example/checkpoint/outputs/sequential/checkpoints/trial-00000100 \
  --output example/checkpoint/outputs/sequential/evaluation-trial-00000100
```

Checkpoint directories are immutable, so this evaluation does not affect a later decision to
resume the same trial-100 checkpoint.

## Automate a finite number of sessions

The open-ended runner requires an explicit finite count and starts each session only after the
previous child process succeeds:

```bash
autorec_env/bin/python example/checkpoint/run_sequential.py --sessions 10
```

To run three more sessions from a previously generated configuration:

```bash
autorec_env/bin/python example/checkpoint/run_sequential.py \
  --config example/checkpoint/outputs/sequential/configs/resume-from-trial-00000400.yaml \
  --sessions 3
```

The runner never searches for a mutable “latest” checkpoint. Each generated YAML identifies
the exact immutable checkpoint it will restore.

## Run the 1,000-trial comparison

Use clean output paths; existing checkpoint destinations are never overwritten.

```bash
autorec_env/bin/python example/checkpoint/run_continuous.py
autorec_env/bin/python example/checkpoint/run_sequential.py --sessions 10
autorec_env/bin/python example/checkpoint/compare_checkpoints.py
autorec_env/bin/python example/checkpoint/compare_results.py
```

The expected final checkpoints are:

```text
example/checkpoint/outputs/continuous/checkpoints/trial-00001000/
example/checkpoint/outputs/sequential/checkpoints/trial-00001000/
```

`compare_checkpoints.py` compares exact values, ordered containers, NumPy dtypes, shapes, and
raw numerical bytes for counters, histories, RNG streams, environments, caches, online and
target networks, optimizer, and replay. Output paths and the deliberately different current
session lengths are excluded; schedule configuration is not excluded.

`compare_results.py` first performs that complete checkpoint comparison. It then restores
both agents in separate fresh processes, evaluates the same training rows with greedy actions,
and compares the complete evaluation payloads exactly. It also verifies that evaluation did
not modify either checkpoint directory.

Choose a different deterministic sample or exact rows when useful:

```bash
autorec_env/bin/python example/checkpoint/compare_results.py \
  --num-samples 20 --sample-seed 42 --max-actions 12

autorec_env/bin/python example/checkpoint/compare_results.py --indices 0,5,12,20
```

The pickle outputs are used for exact comparison; CSV files under
`outputs/evaluation_comparison/` are intended for inspection. Checkpoints and evaluation
pickles must be treated as trusted local experiment output. Exact reproducibility remains
qualified to the deterministic CPU/software conditions documented in
[`docs/checkpointing.md`](../../docs/checkpointing.md); real AutoEIS/Julia/SciPy fitting can
have additional native-runtime limitations.
