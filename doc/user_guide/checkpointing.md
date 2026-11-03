# Training checkpoints

AutoREC can save complete training state at completed-trial boundaries and
resume it in a fresh Python process. This differs from `save_model()`, which
exports only the Keras model and cannot resume training exactly.

## Open-ended sessions

`num_trials` is the work requested for the current session. The checkpoint's
read-only `completed_trials` value is cumulative, so training can continue
without declaring a final trial count in advance. For example, an agent
constructed with `num_trials=100` reaches cumulative trial 100 after one
session and trial 200 after the next restored session.

Use absolute cumulative-trial schedule settings for open-ended runs. In
particular, prefer `beta_anneal_trials` to the legacy `anneal_fraction`, whose
duration is resolved from the initial `num_trials`. Do not change resolved
epsilon or beta schedules after training begins.

## Python workflow

Configure the same deterministic CPU runtime at the start of every process,
before importing TensorFlow or other numerical libraries. For exact-comparison
runs, establish the process environment before starting Python:

```bash
export CUDA_VISIBLE_DEVICES=-1
export PYTHONHASHSEED=0
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

The initial process can then create the configured agent and save a session-end
checkpoint:

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1)

from autorec.factory import pipeline_builder
from autorec.utils import set_global_seed

set_global_seed(42, deterministic_ops=True)  # Initial process only
_, _, _, _, agent = pipeline_builder("config.yaml")
agent.train(checkpoint_dir="checkpoints")
assert agent.completed_trials == 100
```

In the next process, configure TensorFlow identically and restore the complete
checkpoint. Do not reseed after restoration:

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1)

import tensorflow as tf

from autorec.agent import DDQN_ECM

tf.config.experimental.enable_op_determinism()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

agent = DDQN_ECM.from_checkpoint(
    "checkpoints/trial-00000100",
    num_trials=100,
)
agent.train(checkpoint_dir="checkpoints")
assert agent.completed_trials == 200
```

`additional_trials=N` overrides the stored session length for one `train()`
call. `checkpoint_every=N` saves at cumulative multiples of `N` and at the
session endpoint. Manual snapshots use `agent.save_checkpoint(path)`.

If results must be inspected before deciding whether to continue, evaluate a
checkpoint in a separate process. Do not evaluate or otherwise consume random
numbers in the training process between retained sessions. Later resume the
same immutable checkpoint that was evaluated.

## CLI workflow

For a known cumulative target, automatic recovery lets the same command be rerun after an
interruption. Here `agent.num_trials` in the YAML is the cumulative target:

```bash
autorec train --config config.yaml \
  --checkpoint-dir checkpoints \
  --checkpoint-every 10 \
  --resume-latest
```

No checkpoint starts a fresh run. A latest checkpoint at trial 300 with a target of 1,000
runs 700 more trials; a checkpoint at 1,000 is a no-op. Invalid, incompatible, mixed-series,
or above-target checkpoints fail rather than silently restarting. Rerun the same command
after cancellation or machine failure.

The explicit workflow remains available for open-ended sessions:

```bash
# Initial configured session
autorec train --config config.yaml --checkpoint-dir checkpoints

# Continue for 100 trials using the complete saved state
autorec train \
  --resume checkpoints/trial-00000100 \
  --additional-trials 100 \
  --checkpoint-dir checkpoints \
  --checkpoint-every 100
```

`--resume-latest` requires `--config` and `--checkpoint-dir`; it cannot be combined with
`--resume` or `--additional-trials`. It fingerprints training settings and seed policy while
allowing the cumulative target and output directory to change. `--config` and `--resume`
are mutually exclusive. `--output-dir` may relocate
future outputs, but architecture, optimizer, data, and training schedules are
checkpoint-owned and cannot be changed during an exact resume.

The same automatic option is exposed by the Python entry script:

```bash
python -m example.checkpoint.complete_checkpoint.cli --config config.yaml \
  --checkpoint-dir checkpoints --checkpoint-every 10 --resume-latest
```

## What is preserved

A complete checkpoint stores:

- online and target networks, including nontrainable variables;
- optimizer configuration, iterations, slots, and supported schedules;
- the physical replay allocation, ordering, priorities, and insertion state;
- cumulative counters, exploration state, histories, and statistics;
- prepared training/evaluation environments and their active identity;
- cache entries, hit/miss counts, and LRU or Clock eviction metadata; and
- supported Python, NumPy, TensorFlow, and layer-local random state.

Checkpoint directories are immutable. AutoREC writes a temporary sibling,
records file sizes and SHA-256 hashes in the manifest, and atomically renames
the completed directory. Load checkpoints only from trusted sources: integrity
hashes detect damage but do not make pickle payloads safe to open.

A checkpoint has the following layout:

```text
checkpoints/trial-00000100/
├── manifest.json       # Version, trial count, runtime/source identity, hashes
├── config.json         # Effective agent configuration and output paths
├── training.pkl        # Progress, histories, replay metadata, and RNG state
├── environments.pkl    # Prepared data, active environments, and caches
├── neural.pkl          # Online/target networks and optimizer state
└── replay.npy          # Complete physical replay allocation
```

The loader validates the complete file list, sizes, hashes, format version,
runtime and source fingerprint before restoration. It intentionally rejects
best-effort loading when exact continuation cannot be supported.

## Exact-equivalence conditions

Bitwise continuation is supported only under the validated deterministic CPU
conditions: identical AutoREC source, dependency/runtime fingerprint,
threading and TensorFlow determinism settings, supported built-in
network/optimizer components, and no mutation or evaluation between training
boundaries. An environment subclass must explicitly declare all additional
instance fields through `CHECKPOINT_FIELDS` and all behavior-affecting source
modules through `CHECKPOINT_SOURCE_MODULES`; the detailed contract is in the
complete checkpoint documentation linked below. GPU execution, changed native
libraries or hardware, other arbitrary custom components, and nondeterministic
real fitting require separate qualification. Checkpoints are supported at
completed-trial boundaries, not in the middle of a trial.

For the complete format, compatibility checks, limitations, and validation
evidence, read `docs/checkpointing.md` in the repository. The
`example/checkpoint/` workflow compares one continuous run with sequential
fresh-process sessions and exports per-category JSON comparison reports.

From the repository root, the complete validation workflow is:

```bash
python -m example.checkpoint.continuous.run
python -m example.checkpoint.complete_checkpoint.run --sessions 10
python -m example.checkpoint.comparison.compare_checkpoints
python -m example.checkpoint.comparison.compare_results
```

`compare_checkpoints.py` is the faster persisted-state comparison. It checks
counters, histories, schedules, random streams, environments, caches, online
and target networks, optimizer state, and the replay buffer. Its default report
is `outputs/comparison_reports/checkpoint_comparison.json`.

`compare_results.py` includes those checks and additionally restores both
agents in fresh processes, evaluates identical training rows using greedy
actions, compares the resulting ECM records exactly, and verifies that
evaluation did not modify either checkpoint. Its default report is
`outputs/evaluation_comparison/comparison_report.json`. Both commands print
named per-check verdicts, finish with a prominent overall verdict, and return a
nonzero status when comparison fails.
