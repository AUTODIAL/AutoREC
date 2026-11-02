# Training checkpoints

> The documentation website also provides a concise
> [checkpointing workflow](../doc/user_guide/checkpointing.md). This file is the
> detailed format, compatibility, and validation reference.

AutoREC can save complete training state at completed-trial boundaries and resume in a
new Python process. Normally, `num_trials` is the amount of work requested for the current
session and the checkpoint owns cumulative `completed_trials`. The opt-in `--resume-latest`
workflow instead interprets `num_trials` as a cumulative target. Do not change
training-critical settings or reseed after restoration.

## Python workflow

Start each process with the same deterministic CPU setup, before importing numerical
libraries. For example:

```bash
export CUDA_VISIBLE_DEVICES=-1
export PYTHONHASHSEED=0
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Create the initial agent using the existing factory. For schedules that must not depend on
session length, use absolute cumulative-trial settings such as `beta_anneal_trials`.

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1)
from autorec.factory import pipeline_builder
from autorec.utils import set_global_seed

set_global_seed(42, deterministic_ops=True)  # Initial run only
_, _, _, _, agent = pipeline_builder("config.yaml")  # num_trials: 100
agent.train(checkpoint_dir="checkpoints")
assert agent.completed_trials == 100
```

In a fresh process, configure the runtime and deterministic operations, then load:

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1)
import tensorflow as tf
from autorec.agent import DDQN_ECM

tf.config.experimental.enable_op_determinism()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
agent = DDQN_ECM.from_checkpoint("checkpoints/trial-00000100", num_trials=100)
agent.train(checkpoint_dir="checkpoints")
assert agent.completed_trials == 200
```

At this point the user can assess the cumulative results and either stop or repeat with
`trial-00000200`. Calling `agent.train()` on a restored agent advances by its current
`num_trials`; `additional_trials=N` overrides that count for one invocation. There is no
cumulative ceiling. A zero-trial call remains a no-op.

The beta scheduler previously accepted only `anneal_fraction`, whose effective duration is
`num_trials * anneal_fraction`. That legacy option remains supported. Open-ended workflows
should instead set the mutually exclusive absolute duration, for example:

```yaml
agent:
  num_trials: 100
  start_jump: 400
  beta_anneal_trials: 595
```

This produces the same beta trajectory through cumulative trial 1,000 as the legacy
`num_trials: 1000`, `start_jump: 400`, `anneal_fraction: 0.595` configuration. Beta remains
at its final value if training continues beyond that point.

Manual snapshots use `agent.save_checkpoint("checkpoints/my-boundary")`. To save before
the first trial, first call `agent.train(additional_trials=0)`. Zero-trial calls do not
automatically save. `checkpoint_dir` alone saves at the session end; `checkpoint_every=N`
also saves at cumulative multiples of N. Automatic names use eight digits, for example
`trial-00000100`. A session ending between multiples still gets a final snapshot.

Without checkpoint arguments, `agent.train()` and existing model exports retain their
previous behavior. `save_model()` and `load_model()` remain model-only APIs and cannot
resume complete training. Every non-empty retained session exports the current cumulative
model, histories and plots. See [the runnable open-ended example](../example/checkpoint/).
That example also compares continuous and resumed checkpoints exactly. Its faster checkpoint
comparison and slower fresh-process evaluation comparison print per-category results, finish
with an overall verdict, and export machine-readable JSON reports.

## CLI workflow

Use the same startup environment variables as above. The commands below run in separate
processes. Automatic recovery is appropriate when the target is known and the same command
should survive scheduler cancellation or machine failure:

```bash
# Ordinary training remains supported.
autorec train --config config.yaml

# Train toward cumulative agent.num_trials, saving every 10 completed trials.
autorec train --config config.yaml --checkpoint-dir checkpoints \
  --checkpoint-every 10 --resume-latest

# After an interruption, rerun exactly the same command.
autorec train --config config.yaml --checkpoint-dir checkpoints \
  --checkpoint-every 10 --resume-latest
```

If no checkpoint exists, automatic recovery starts from the configured seed. If the latest
checkpoint is trial 300 and `agent.num_trials` is 1,000, it restores trial 300 and requests
700 additional trials. Trial 1,000 produces a successful no-op. A malformed, corrupt,
incompatible, mixed-series, or above-target checkpoint fails instead of silently starting
over. `--resume-latest` requires `--config` and `--checkpoint-dir` and cannot be combined
with `--resume` or `--additional-trials`.

The explicit manual workflow remains available for open-ended training:

```bash
# Start a configured session.
autorec train --config config.yaml --checkpoint-dir checkpoints

# Perform the next 100 trials using the saved configuration and prepared data.
autorec train --resume checkpoints/trial-00000100 --additional-trials 100 \
  --checkpoint-dir checkpoints --checkpoint-every 100

# Use the stored session length, optionally relocating result/model outputs.
autorec train --resume checkpoints/trial-00000200 --output-dir resumed-results \
  --checkpoint-dir checkpoints
```

`--config` and `--resume` are mutually exclusive. Manual resume does not call `set_global_seed`;
`--seed` applies only to initial training. `--non-deterministic` must agree with the saved
runtime, but exact reproducibility requires deterministic operations. `--output-dir`
relocates exported results and the final model; it does not change training configuration.
Checkpoint paths and frequencies can change freely between sessions. The Python API also
allows `from_checkpoint(..., num_trials=N)` to change the next session length; the CLI can
use `--additional-trials N` for the same per-invocation override.

Automatic recovery records a checkpoint-series ID and a fingerprint of the configured
training configuration and initial seed policy. It permits changes to `agent.num_trials`
(to extend a cumulative target) and `agent.save_dir`, but rejects other configuration
changes. It scans immutable `trial-NNNNNNNN` directories; it does not maintain a mutable
`latest` pointer. The same option is available through `example/checkpoint_training.py`.
Extending a target does not recompute schedules already owned by the checkpoint; configure
absolute cumulative schedule values when later extension is possible.

## Contents and persistence

```text
checkpoints/trial-00000100/
├── manifest.json       # Format, series/config identity, trial count, runtime, file hashes/sizes
├── config.json         # Effective agent configuration and output paths
├── training.pkl        # Counters, histories, epsilon/beta, replay metadata, RNG state
├── environments.pkl    # Prepared datasets, current observations, caches, environment identities
├── neural.pkl          # Architectures, independent networks, all variables, optimizer state
└── replay.npy          # Entire physical replay array, including unused slots; no object pickle
```

The payloads use existing component APIs. Neural state contains trainable and nontrainable
variables, target weights independent of online weights, layer-local random counters,
optimizer slots/iteration, mutable learning rate and supported schedule configuration.
Replay preserves capacity, size, physical order, insertion position and priorities. Cache
state preserves values, counters, LRU order or Clock ring/reference bits/hand. Histories
include loss, success and reward statistics, terminal history as currently maintained by
training, and epsilon/beta history. The existing terminal-history reporting defect is not
changed by checkpointing.

Prepared DataFrames include nested numerical arrays and metadata. Shared training/evaluation
environment identity, active environment choice, and shared prepared-dataset references
are retained. Environment reconstruction does not parse circuits, sample, reset or seed.
Neural construction and validation finish before restoring random streams.

Python and NumPy global streams include their cached Gaussian samples. An already existing
TensorFlow `Generator` is captured without creating one during saving. Keras layer-local
generators are captured with network variables. TensorFlow's legacy `tf.random.*` operation
counters cannot be restored through a supported API. Arbitrary stochastic custom layers,
external generators, callbacks and concurrently executing random operations are outside
the contract. Current AutoREC action/replay/data/fitting initialization draws use NumPy's
captured global stream; supported neural training does not use legacy TF random operations.

Each checkpoint is immutable. A temporary sibling directory is written and fsynced, with
the manifest last, then renamed into place. Existing destinations raise `FileExistsError`.
A failed write leaves previous checkpoints intact; a killed process can leave an unused
temporary directory. Filesystem durability relies on local filesystem fsync/rename semantics;
concurrent writers to the same destination and network-filesystem crash guarantees are not
supported. No automatic pruning or mutable `latest` pointer is provided.

## Compatibility and safety

**Load trusted checkpoints only.** Pickle can execute code. SHA-256 checksums detect damaged
or incomplete files; they do not authenticate a checkpoint. The loader checks the complete
file list, sizes, hashes, format, runtime and source before unpickling. Component validation
then rejects incompatible arrays/configuration and incomplete state. Restoration creates
a new agent; it never partially overwrites an existing one. On failure, the supported
caller RNG streams are restored. Output directories can have been created before a late
filesystem/restoration error.

Format version 6 requires identical AutoREC Python source and recorded Python/numerical
library versions, OS/platform, threading settings, TensorFlow determinism and backend
settings. This is deliberately conservative: there is no migration or best-effort loading.
Source edits, including implementation updates between stages, invalidate old checkpoints.
Output relocation and session-length changes are supported; changing architecture,
optimizer, data, or schedule hyperparameters is not a resume operation. Runtime matching
does not prove identical
hardware or native-library builds: use the same hardware and software environment as well.

Checkpoint support currently requires CPU execution and an ordinary `DDQN_ECM` agent.
The standard `EIS_ECM_Env` and explicitly declared subclasses are supported. A subclass
must be importable by the same module and qualified name in the resumed process and must
define both declarations directly on the class:

```python
class ExperimentalEnvironment(EIS_ECM_Env):
    CHECKPOINT_FIELDS = frozenset({"reward_scale"})
    CHECKPOINT_SOURCE_MODULES = ("my_experiment.reward_utils",)
```

`CHECKPOINT_FIELDS` must name every instance attribute added by the subclass; standard
environment fields and `cache` must not be repeated. The environment class's module is
fingerprinted automatically. `CHECKPOINT_SOURCE_MODULES` adds importable modules whose
implementation affects transitions, rewards, or other future training behavior. Both
declarations are required even when empty. Restoration creates the exact class without
calling its constructor, then restores its declared fields. Missing, extra, changed, or
undeclared state is rejected, and changed declared source fails compatibility validation
before checkpoint payloads are unpickled. External resources and mutable module-level state
are not captured by these declarations and remain unsupported.

Networks must be uncompiled built-in Keras Sequential models with Dense,
BatchNormalization, Dropout, Activation or Flatten, using float32/float64. Optimizers are
Adam (including AMSGrad) or SGD, with constant learning rate, ExponentialDecay,
PiecewiseConstantDecay or CosineDecay. Custom layers/optimizers, Functional/compiled models,
mixed precision/loss scaling, EMA, gradient accumulation, LoRA/quantization and stateful
custom schedules are unsupported. Ordinary training retains its existing support for
configurations outside this checkpoint subset.

Only completed trials are checkpoint boundaries. An interruption during a trial discards
work since the last saved boundary; reload that checkpoint. Mid-episode gradients, temporary
episode histories, and fitting processes are not snapshotted. Do not evaluate the live
training agent, mutate data/caches/history, change settings, or consume its RNG streams before
continuing it. For inspect-and-decide workflows, load the immutable checkpoint in a separate
evaluation process and later resume the original checkpoint. Shared mutable caches between
distinct environments and arbitrary external object aliasing are outside the supported
configuration.

## Validation and limits

The routine fresh-process test compares 8 continuous trials with fresh-process sessions of
3 and 5 trials using the more stateful Clock cache. The split sessions perform periodic
saves, and every worker verifies that its final save is state-inert. The test compares exact
numerical bytes and ordered state for networks, optimizer, replay, cache, environments, RNG
streams, schedules, counters and histories, plus transition and replay-sampling traces.
Separate component and same-process integration tests cover both LRU and Clock behavior. The
expensive 1,000-trial test additionally uses a continuous periodic-saving control alongside
ten fresh 100-trial processes. It uses sparse optimizer updates to keep runtime practical
while crossing warm-up, target updates, replay wrapping and schedule boundaries.

A separate fresh-process acceptance test simulates termination immediately after publishing
a periodic checkpoint, reruns `--resume-latest`, and compares the final state and combined
interaction/replay trace exactly against uninterrupted training.

Both use a controlled, RNG-consuming fitter and real RL/TensorFlow updates. Circuit helper
calls and model/plot output are replaced. These tests establish checkpoint equivalence
under that controlled CPU configuration; they do **not** establish bitwise reproducibility
of real AutoEIS/Julia/SciPy fitting, different machines, GPUs, nondeterministic kernels or
arbitrary native runtimes. Real-fitting qualification remains a separate validation task.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
autorec_env/bin/python -m pytest -q tests/test_checkpoint_process.py

AUTOREC_LONG_CHECKPOINT_TEST=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg autorec_env/bin/python -m pytest -q \
tests/test_checkpoint_process.py -k 1000
```
