# Open-ended checkpoint training and equivalence experiment

This example supports two related workflows:

- Train for one configured session, inspect the cumulative results, and decide later whether
  to continue from the generated checkpoint configuration.
- Compare one continuous 1,000-trial process with ten fresh 100-trial processes.

All commands below assume that the desired virtual environment is active and the current
working directory is `example/checkpoint`.

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

Run one 100-trial session. The wrapper starts the training worker in a fresh process with the
deterministic environment used by the equivalence experiment:

```bash
python run_sequential.py --config sequential_config.yaml --sessions 1
```

The process saves cumulative results, an immutable complete checkpoint, and a possible next
configuration:

```text
outputs/sequential/checkpoints/trial-00000100/
outputs/sequential/configs/resume-from-trial-00000100.yaml
```

Inspect or evaluate the checkpoint. If more training is useful, run the generated full YAML:

```bash
python run_sequential.py \
  --config outputs/sequential/configs/resume-from-trial-00000100.yaml \
  --sessions 1
```

That invocation restores trial 100, performs the configured 100 additional trials, and
produces `trial-00000200` plus another optional continuation YAML. You can change only
`agent.num_trials`, output paths, and checkpoint paths between sessions. Changes to model,
optimizer, environment, replay, or schedule settings are rejected or remain checkpoint-owned.

Evaluate in a separate process so assessment cannot modify the resumable training state:

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=-1 \
python evaluate_checkpoint.py \
  --checkpoint outputs/sequential/checkpoints/trial-00000100 \
  --output outputs/sequential/evaluation-trial-00000100
```

Checkpoint directories are immutable, so this evaluation does not affect a later decision to
resume the same trial-100 checkpoint.

## Automate a finite number of sessions

The open-ended runner requires an explicit finite count and starts each session only after the
previous child process succeeds:

```bash
python run_sequential.py --sessions 10
```

To run three more sessions from a previously generated configuration:

```bash
python run_sequential.py \
  --config outputs/sequential/configs/resume-from-trial-00000400.yaml \
  --sessions 3
```

This equivalence runner deliberately uses explicit immutable checkpoint paths recorded in
generated YAML. For automatic crash recovery toward a cumulative target, use
`autorec train --config ... --checkpoint-dir ... --checkpoint-every ... --resume-latest`
or the general `example/checkpoint_training.py` script described in `example/README.md`.

## Run the 1,000-trial comparison

Use clean output paths; existing checkpoint destinations are never overwritten.

```bash
python run_continuous.py
python run_sequential.py --sessions 10
```

The expected final checkpoints are:

```text
outputs/continuous/checkpoints/trial-00001000/
outputs/sequential/checkpoints/trial-00001000/
```

Choose the comparison appropriate for the desired validation:

| Script | What it checks | Relative cost |
| --- | --- | --- |
| `python compare_checkpoints.py` | Exact persisted counters, histories, RNG streams, environments, caches, online and target networks, optimizer, and replay | Fast; does not run agent evaluation |
| `python compare_results.py` | Everything checked by `compare_checkpoints.py`, plus fresh-process restoration and exact greedy evaluation on identical training rows | Slower; performs circuit evaluations |

`compare_checkpoints.py` compares exact values, ordered containers, NumPy dtypes, shapes, and
raw numerical bytes. Output paths and the deliberately different current session lengths are
excluded; schedule configuration is not excluded.

Because `compare_results.py` begins with the complete checkpoint comparison, it is unnecessary
to run both scripts unless a quick checkpoint-only verdict is useful first. The results script
also verifies that evaluation does not modify either checkpoint directory. Both scripts finish
with a prominent `VERDICT: PASS` or `VERDICT: FAIL` summary and return a nonzero exit status on
failure.

Before the overall verdict, each script prints a checklist for the individual state categories.
On failure, the category includes the first exact mismatch path. The same information is written
atomically as JSON for later inspection or automated processing:

```text
outputs/comparison_reports/checkpoint_comparison.json
outputs/evaluation_comparison/comparison_report.json
```

Use `--report PATH` with either script to select another report destination. The JSON contains
the two checkpoint paths, each named check and its detail, and the overall pass/fail result.

Choose a different deterministic sample or exact rows when useful:

```bash
python compare_results.py \
  --num-samples 20 --sample-seed 42 --max-actions 12

python compare_results.py --indices 0,5,12,20
```

The pickle outputs are used for exact comparison; CSV files under
`outputs/evaluation_comparison/` are intended for inspection. Checkpoints and evaluation
pickles must be treated as trusted local experiment output. Exact reproducibility remains
qualified to the deterministic CPU/software conditions documented in
[`docs/checkpointing.md`](../../docs/checkpointing.md); real AutoEIS/Julia/SciPy fitting can
have additional native-runtime limitations.
