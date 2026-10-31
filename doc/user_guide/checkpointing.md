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
before importing TensorFlow or other numerical libraries. The initial process
can then create the configured agent and save a session-end checkpoint:

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

## CLI workflow

The same workflow is available through the CLI:

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

`--config` and `--resume` are mutually exclusive. `--output-dir` may relocate
future outputs, but architecture, optimizer, data, and training schedules are
checkpoint-owned and cannot be changed during an exact resume.

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

## Exact-equivalence conditions

Bitwise continuation is supported only under the validated deterministic CPU
conditions: identical AutoREC source, dependency/runtime fingerprint,
threading and TensorFlow determinism settings, supported built-in
network/optimizer components, and no mutation or evaluation between training
boundaries. GPU execution, changed native libraries or hardware, arbitrary
custom components, and nondeterministic real fitting require separate
qualification. Checkpoints are supported at completed-trial boundaries, not
in the middle of a trial.

For the complete format, compatibility checks, limitations, and validation
evidence, read `docs/checkpointing.md` in the repository. The
`example/checkpoint/` workflow compares one continuous run with sequential
fresh-process sessions and exports per-category JSON comparison reports.
