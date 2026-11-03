# Complete-checkpoint continuation and equivalence

This directory currently contains two controlled scenarios:

- `continuous/`: one uninterrupted 1,000-trial reference process;
- `complete_checkpoint/`: ten fresh 100-trial processes that restore complete checkpoints.

Shared session, runtime, evaluation, and comparison code remains at this directory level or
under `comparison/`. Run commands from the repository root with the AutoREC environment active.

Both scenarios use the same absolute cumulative schedules:

```yaml
start_jump: 400
beta_anneal_trials: 595
```

These settings reproduce the beta trajectory formerly obtained from a 1,000-trial run with
`anneal_fraction: 0.595` without making the session length a final training horizon.

## Inspect one complete-checkpoint session

Run one fresh-process 100-trial session:

```bash
python -m example.checkpoint.complete_checkpoint.run \
  --config example/checkpoint/complete_checkpoint/config.yaml \
  --sessions 1
```

The process writes an immutable complete checkpoint and a possible next configuration:

```text
example/checkpoint/outputs/complete_checkpoint/checkpoints/trial-00000100/
example/checkpoint/outputs/complete_checkpoint/configs/resume-from-trial-00000100.yaml
```

Continue later from that generated configuration:

```bash
python -m example.checkpoint.complete_checkpoint.run \
  --config example/checkpoint/outputs/complete_checkpoint/configs/resume-from-trial-00000100.yaml \
  --sessions 1
```

The configuration may change only the next session length, output paths, and checkpoint paths.
Model, optimizer, environment, replay, and schedule settings remain checkpoint-owned.

The smaller public CLI wrapper is also available when a comparison configuration is not needed:

```bash
python -m example.checkpoint.complete_checkpoint.cli \
  --config configs_yaml/examples/demo_pipeline_config.yaml \
  --checkpoint-dir checkpoint_outputs \
  --checkpoint-every 10 \
  --resume-latest
```

## Run the controlled equivalence experiment

Use clean output paths because checkpoint destinations are immutable:

```bash
python -m example.checkpoint.continuous.run
python -m example.checkpoint.complete_checkpoint.run --sessions 10
```

The expected final checkpoints are:

```text
example/checkpoint/outputs/continuous/checkpoints/trial-00001000/
example/checkpoint/outputs/complete_checkpoint/checkpoints/trial-00001000/
```

Choose the comparison appropriate for the desired validation:

```bash
# Faster exact persisted-state comparison
python -m example.checkpoint.comparison.compare_checkpoints

# Persisted state plus fresh-process greedy evaluation
python -m example.checkpoint.comparison.compare_results
```

The checkpoint comparison covers counters, histories, schedules, random streams,
environments, caches, online and target networks, optimizer state, and replay. The evaluation
comparison additionally evaluates identical rows in separate processes and verifies that the
immutable checkpoint files did not change.

Both tools print named per-category verdicts, finish with an overall verdict, write an atomic
JSON report, and return nonzero when an expected exact comparison fails. Default reports are:

```text
example/checkpoint/outputs/comparison_reports/checkpoint_comparison.json
example/checkpoint/outputs/evaluation_comparison/comparison_report.json
```

The 1,000-trial workflow is deliberately expensive and is not part of routine unit tests.
Exact equivalence remains qualified to the deterministic CPU/software conditions documented
in [`docs/checkpointing.md`](../../docs/checkpointing.md); real AutoEIS/Julia/SciPy fitting can
have additional native-runtime limitations.
