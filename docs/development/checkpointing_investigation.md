# AutoREC Checkpointing — Investigation and Implementation Plan

> **Historical design notice (2026-10-27):** Phase 2 superseded this report's fixed-horizon
> proposal with open-ended sessions. `num_trials` now means the current session length, and
> `beta_anneal_trials` defines an absolute cumulative-trial beta schedule. See
> [`docs/checkpointing.md`](../checkpointing.md) for the implemented behavior.

**Phase 1: investigation and design only**

**Status:** implementation reference proposed for review. The APIs, checkpoint format, and
code changes described below have not been implemented. Saving this report does not
authorize Phase 2 implementation.

## Executive assessment

AutoREC can reasonably support exact training continuation through **complete checkpoints
at completed-trial boundaries**. Its current model-saving functionality does not provide
this capability.

The principal requirements are to:

- Retain the replay buffer, counters, and histories that currently live inside `train()`.
- Restore the online network, target network, and separate optimizer independently.
- Preserve complete RNG states, including supported Keras layer seed variables.
- Preserve circuit-evaluation caches, including eviction order.
- Separate a session's trial budget from the experiment's fixed training horizon.
- Validate equivalence in fresh processes, including a control proving that checkpoint
  creation itself does not change training.

Exact continuation is **not yet established empirically for AutoREC's complete scientific
training pipeline**. The investigation identified a practical design and verified several
dependency-level behaviors, but the integration and real-fitting validation remain Phase 2
work.

### Investigation basis

The assessed checkout was:

```text
a71a0bfb0a9f9052d823d38fcbd03be82e3b6bfe
```

The investigation covered relevant package implementations, factories, CLI, examples,
notebook sources, configuration files, tests, CI, and installed dependency code. Existing
untracked examples and outputs were left untouched.

No repository files were modified during the Phase 1 investigation. Small dependency probes
used temporary files outside the repository. No full AutoREC training-equivalence
experiment was performed. This documentation file was subsequently created at the user's
explicit request; no implementation changes accompany it.

References below identify the source locations examined during Phase 1. Line numbers refer
to that assessed checkout and may move during implementation. Repository-relative links
make this report usable outside the original workspace.

## 1. Existing Training Architecture

### 1.1 Entry points and construction

Training can begin through:

- Direct construction of `EIS_ECM_Env` and `DDQN_ECM`.
- `pipeline_builder(...)`.
- The `autorec train` CLI.

The factory builds prepared datasets, environments, and then the agent. It supports
separate training and evaluation environments. If no evaluation environment is supplied,
the agent aliases the training environment.

See:

- [pipeline_builder — src/autorec/factory/pipeline.py:18](../../src/autorec/factory/pipeline.py#L18)
- [agent_builder — src/autorec/factory/agent.py:9](../../src/autorec/factory/agent.py#L9)
- [DDQN_ECM environment assignment — src/autorec/agent.py:207](../../src/autorec/agent.py#L207)
- [CLI training — src/autorec/cli.py:81](../../src/autorec/cli.py#L81)

`EIS_ECM_Env.__init__`:

1. Retains the prepared dataset.
2. Stores the ordered circuit elements and initial states.
3. Seeds NumPy's process-global RNG.
4. Computes chromosome dimensions and the action table.
5. Creates an LRU or Clock circuit-evaluation cache.
6. Calls `reset()`, immediately consuming randomness.

See [src/autorec/environment.py:166](../../src/autorec/environment.py#L166).

`DDQN_ECM.__init__`:

1. Sets the TensorFlow seed.
2. Stores and resolves hyperparameters.
3. Creates a separate Keras optimizer.
4. Builds the online model and initializes a target model.
5. Initializes epsilon, beta, and their histories.
6. Creates output directories and writes a model summary.

See [src/autorec/agent.py:215](../../src/autorec/agent.py#L215).

The default model has two Dense layers with 40 ReLU units and a linear action-value output.
The model factory accepts other built-in Keras layer classes, so checkpointing cannot
assume that all persistent model state consists of Dense weights.

See [src/autorec/factory/model.py:13](../../src/autorec/factory/model.py#L13).

### 1.2 Training state and lifecycle

`DDQN_ECM.train()` currently initializes these variables locally on every invocation:

- Replay buffer.
- `iteration`.
- `target_freq`.
- `memory`.
- `success_count`.
- Loss, success-rate, terminal-state, and reward histories.

See [src/autorec/agent.py:889](../../src/autorec/agent.py#L889).

The lifecycle is:

```text
Prepared data + environment configuration + initialized agent
                             |
                             v
Trial t: reset environment, select starting state and EIS
                             |
                             v
Run episodes_trial episodes
  - Select action
  - Apply environment transition
       - Cache hit: reuse fitted result
       - Cache miss: randomized circuit fitting
  - Insert replay transition
  - Perform scheduled replay/optimizer updates
  - Perform scheduled target-network copy
                             |
                             v
Accumulate episode histories and statistics
                             |
                             v
Optional existing model export
                             |
                             v
Update epsilon and beta
                             |
                             v
Recommended complete-checkpoint boundary
```

Important existing semantics:

- The environment resets once per **trial**, not once per episode.
- With `episodes_trial > 1`, subsequent episodes continue from the environment state left
  by the previous episode.
- A transition enters replay before the training-frequency check.
- The training-frequency check uses `iteration` before its increment.
- The first optimizer update occurs when the incoming `iteration == NN_sleep`.
- `gradient_steps` controls repeated replay sampling and optimizer updates per training
  event.
- Target-network synchronization has a separate interaction counter.

See:

- [Trial and episode loop — src/autorec/agent.py:947](../../src/autorec/agent.py#L947)
- [Episode execution — src/autorec/agent.py:1030](../../src/autorec/agent.py#L1030)
- [Interaction ordering — src/autorec/agent.py:1139](../../src/autorec/agent.py#L1139)
- [Training and target schedules — src/autorec/agent.py:1206](../../src/autorec/agent.py#L1206)

Checkpointing must preserve this ordering rather than refactor it into superficially
equivalent formulas.

### 1.3 Replay and optimization

Replay is a fixed-capacity NumPy structured circular buffer. Its DataFrame conversion
exposes **physical array order**, not chronological order after wraparound.

Prioritized sampling uses NumPy's global RNG and samples without replacement. Training
updates sampled priorities, then copies them back into the physical buffer slots.

See:

- [CircularBuffer — src/autorec/optimized_data_structures/circular_buffer.py:10](../../src/autorec/optimized_data_structures/circular_buffer.py#L10)
- [Physical-order conversion — circular_buffer.py:47](../../src/autorec/optimized_data_structures/circular_buffer.py#L47)
- [Prioritized sampling — src/autorec/agent.py:616](../../src/autorec/agent.py#L616)
- [Gradient updates — src/autorec/agent.py:870](../../src/autorec/agent.py#L870)
- [Priority writeback — src/autorec/agent.py:1213](../../src/autorec/agent.py#L1213)

The optimizer belongs to the agent as `self.optimizer`. The custom training loop uses
`GradientTape` and `apply_gradients()`; the optimizer is not attached through
`model.compile()`.

### 1.4 Environment fitting and caching

On a valid action, the environment updates its circuit state, computes a cache key, and
either reuses a result or fits circuit parameters.

The cache key contains:

```text
(circuit_code, eis_index, action_type, action_position)
```

Cached values include rewards, fitted parameters, predicted impedance, metrics,
termination status, and penalties.

See [src/autorec/environment.py:721](../../src/autorec/environment.py#L721) and
[circuit_cache.py:332](../../src/autorec/optimized_data_structures/circuit_cache.py#L332).

Circuit fitting performs repeated randomized initial guesses. Therefore, the cache changes
both future rewards and future random-number consumption.

### 1.5 Existing save/load behavior

`save_model()` writes only the online Keras model.

`load_model()` loads that model and constructs a new target network with the online weights.
It does not recover the original target lag, optimizer, replay, counters, RNG state, or
histories.

See [src/autorec/agent.py:496](../../src/autorec/agent.py#L496) and
[src/autorec/agent.py:535](../../src/autorec/agent.py#L535).

At the end of `train()`, AutoREC writes:

- Final online model.
- `success_rate.csv`.
- `NN_loss.csv`.
- `terminal_states.csv`.
- `statistical_analysis.pkl`.
- Training plots.

See [src/autorec/agent.py:995](../../src/autorec/agent.py#L995).

Existing `.keras` files remain suitable for inference or model initialization. They cannot
provide exact continuation because the missing training state is unrecoverable.

The existing untracked
[sequential-training example:115](../../example/sequential_training/main_factory.py#L115)
also loads an existing final model and skips training; it does not implement checkpoint
continuation. That path existed in the investigated workspace but may not be present in a
clean checkout until the user separately decides to track it.

## 2. Complete Checkpoint State Inventory

The classifications used below are:

- **E — Essential:** affects subsequent training.
- **H — History/output:** required for identical accumulated results and reporting.
- **R — Reconstructible:** recoverable exactly from saved configuration or state.
- **T — Temporary:** unnecessary at a completed-trial boundary.

“Required” includes information stored in the checkpoint configuration, not only mutable
state. Some proposed persistent fields currently exist only as local variables.

### 2.1 Models, optimizer, replay, and schedules

| State / variable | File and location | Purpose | Checkpoint required? | Proposed storage / restoration |
|---|---|---|---|---|
| Online architecture, layer order, trainability, dtype policy, build configuration | [agent.py:436](../../src/autorec/agent.py#L436); [factory/model.py:60](../../src/autorec/factory/model.py#L60) | Defines forward computation and variable interpretation | **E** | Full Keras architecture serialization; `hidden_layers` alone is insufficient |
| Online trainable variables | [agent.py:870](../../src/autorec/agent.py#L870) | Learned Q-function | **E** | TensorFlow checkpoint with explicit variable inventory |
| Online nontrainable variables and layer-local RNG state | Model construction and Keras layer state | Normalization statistics, seed counters, other persistent layer values | **E**, where present | Include complete `model.variables`, not only `model.weights` |
| Target architecture and all target variables | [agent.py:451](../../src/autorec/agent.py#L451); [agent.py:1228](../../src/autorec/agent.py#L1228) | Delayed DDQN targets | **E** | Restore independently; never replace with online weights during resume |
| Optimizer class and effective configuration | [agent.py:272](../../src/autorec/agent.py#L272) | Defines update rule | **E** | Reuse existing optimizer configuration helpers |
| Optimizer iterations, moments, momentum, mutable LR, optional EMA/accumulators | [agent.py:875](../../src/autorec/agent.py#L875) | Determines future updates | **E** | Save complete optimizer variable set |
| Optimizer built/unbuilt status | Optimizer construction and first `apply_gradients()` | Distinguishes pre-warm-up checkpoints from initialized slot state | **E/R** | Record status; create slots only as needed without applying dummy gradients |
| LR schedule configuration and state | [factory/optimizer.py:46](../../src/autorec/factory/optimizer.py#L46) | Determines LR from optimizer progress | **E/R** | Save configuration and optimizer iteration; include additional state if a supported schedule has any |
| Gradient-scaling state | No explicit scaler in current loop | Would affect mixed-precision updates | Not present by default | Reject unvalidated wrappers initially; add complete state support before claiming compatibility |
| Replay records: EIS reference, states, encodings, action, reward, terminal flag, priority | [agent.py:901](../../src/autorec/agent.py#L901); [agent.py:1140](../../src/autorec/agent.py#L1140) | Supplies learning samples | **E** | Lossless structured NumPy array with original dtype and field shapes |
| Replay `capacity`, `size`, `position` | [circular_buffer.py:19](../../src/autorec/optimized_data_structures/circular_buffer.py#L19) | Controls active entries and overwrite position | **E** | Explicit scalars with bounds validation |
| Replay physical ordering | [circular_buffer.py:54](../../src/autorec/optimized_data_structures/circular_buffer.py#L54) | Maps sampled indices to transitions and priority updates | **E** | Restore physical slots directly; do not reorder chronologically |
| PER alpha and priority epsilon | [agent.py:254](../../src/autorec/agent.py#L254) | Controls sampling and updated priorities | **E** | Effective configuration |
| Current `prioritized_replay_beta` | [agent.py:312](../../src/autorec/agent.py#L312); [agent.py:1317](../../src/autorec/agent.py#L1317) | Current importance-sampling correction | **E** | Exact scalar |
| Current `epsilon` | [agent.py:238](../../src/autorec/agent.py#L238); [agent.py:1309](../../src/autorec/agent.py#L1309) | Current exploration probability | **E** | Exact scalar; do not reconstruct repeated multiplication using exponentiation |
| Exploration configuration | [agent.py:235](../../src/autorec/agent.py#L235) | Future epsilon behavior | **E** | Initial epsilon, minimum, decay, and start threshold |
| Beta configuration and fixed horizon | [agent.py:263](../../src/autorec/agent.py#L263); [agent.py:1321](../../src/autorec/agent.py#L1321) | Future beta behavior | **E** | Original `num_trials`, resolved `start_jump`, `anneal_fraction`, initial/final beta, effective annealing duration |

Omitting any essential row above changes the next prediction, sample, update, or schedule.
Merely restoring model weights cannot compensate for these differences. Replay has no
separate priority tree or independent sampling generator in the current implementation;
the priorities reside in the structured array and sampling uses the global NumPy stream.

### 2.2 Progress, configuration, and randomness

| State / variable | File and location | Purpose | Checkpoint required? | Proposed storage / restoration |
|---|---|---|---|---|
| Completed trials / next global trial index | Currently local `t` at [agent.py:947](../../src/autorec/agent.py#L947) | Schedule timing and history labels | **E** | Add persistent `completed_trials`; next zero-based trial index equals this count |
| `iteration` | [agent.py:937](../../src/autorec/agent.py#L937); [agent.py:1190](../../src/autorec/agent.py#L1190) | Global interaction count and warm-up | **E** | Exact integer |
| `memory` | [agent.py:939](../../src/autorec/agent.py#L939); [agent.py:1206](../../src/autorec/agent.py#L1206) | Online-update frequency phase | **E** | Exact integer; preserve current warm-up/reset behavior |
| `target_freq` | [agent.py:938](../../src/autorec/agent.py#L938); [agent.py:1228](../../src/autorec/agent.py#L1228) | Target synchronization phase | **E/R** | Store explicitly rather than depend on a derived formula |
| `success_count` | [agent.py:940](../../src/autorec/agent.py#L940); [agent.py:1067](../../src/autorec/agent.py#L1067) | Cumulative success-rate reporting | **H** | Exact integer; deduplicated terminal records cannot reconstruct it reliably |
| Remaining training hyperparameters | [agent.py:219](../../src/autorec/agent.py#L219) | Actions, episodes, discount, deadloop logic, update cadence | **E** | Preserve action cap, episodes per trial, gamma, deadloop thresholds, invalid-terminal flag, batch size, frequencies, gradient steps, warm-up, and capacities |
| Python RNG state | [utils.py:93](../../src/autorec/utils.py#L93); Keras initialization | Python-random consumers and initialization | **E** | `random.getstate()` / `setstate()`, including Gaussian cache |
| NumPy legacy global RNG state | [environment.py:311](../../src/autorec/environment.py#L311); [agent.py:627](../../src/autorec/agent.py#L627), [agent.py:726](../../src/autorec/agent.py#L726) | Start states, EIS sampling, exploration, replay, pandas sampling, fitting | **E** | Full `np.random.get_state()` tuple, including cached Gaussian state |
| Keras layer-owned seed generators | Keras model variables | Stateful random streams associated with layers | **E**, where present | Explicit variable references in tensor checkpoint |
| Other TensorFlow/Keras generators | Framework execution | Additional supported random streams | **E**, if used | Explicit adapters; legacy operation counters are not covered by saving a seed |
| Local NumPy generators / JAX PRNG keys | None found in the first-party training path | Potential independent random streams | Not currently present | Add explicit support before permitting extensions that use them |
| Seeds and inactive compatibility fields | [agent.py:215](../../src/autorec/agent.py#L215), [agent.py:254](../../src/autorec/agent.py#L254) | Provenance; `beta_jump` and `bayesian` are inactive in current training calculations | Configuration | Preserve values, but do not substitute seeds for live RNG state |
| Runtime configuration | [runtime.py:15](../../src/autorec/runtime.py#L15); [utils.py:74](../../src/autorec/utils.py#L74) | Numerical execution conditions | **E as compatibility constraints** | Record and validate versions, device, precision, deterministic settings, and relevant thread/backend options |

No separate cumulative episode counter is currently used by the algorithm. At a
completed-trial boundary, it can be derived from completed trials and `episodes_trial` if
needed for reporting. The existing local episode index must not be replaced with a global
index inside deadloop logic.

Losing the interaction or cadence counters shifts warm-up, optimizer updates, and target
copies even if the model and replay are restored correctly. Losing RNG state shifts the
stochastic trajectory even if the original seeds are known.

### 2.3 Environment, caches, histories, and temporary state

| State / variable | File and location | Purpose | Checkpoint required? | Proposed storage / restoration |
|---|---|---|---|---|
| Prepared training dataset | [environment.py:146](../../src/autorec/environment.py#L146); [agent.py:827](../../src/autorec/agent.py#L827) | Replay stores EIS references rather than complete spectra | **E** | Exact DataFrame snapshot: index, order, arrays, dtypes, thresholds, labels, metadata |
| Environment configuration | [environment.py:166](../../src/autorec/environment.py#L166) | Representation, reset choices, action mapping, caching | **E** | Ordered elements, initial states, head length, seed, cache settings |
| Action table, chromosome dimensions, start-state list, operator/terminal lists | [environment.py:185](../../src/autorec/environment.py#L185), [environment.py:222](../../src/autorec/environment.py#L222) | Determines encoding and action-index meanings | **R** | Rebuild from identical code/configuration; verify saved dimensions and ordered fingerprints |
| `state`, `START_STATE`, `current_eis_index`, `encoded_state` | [environment.py:286](../../src/autorec/environment.py#L286) | Current observation | **R/T for next trial** | Save a compact boundary snapshot for complete object restoration; next trial still performs its normal reset |
| Current spectrum references and thresholds | [environment.py:399](../../src/autorec/environment.py#L399) | Current fit/reward inputs | **R** | Reattach `Z`, `freq`, `Z_norm_flatten`, label, and thresholds to saved dataset row without random reset |
| Circuit-cache keys and result payloads | [environment.py:729](../../src/autorec/environment.py#L729); [circuit_cache.py:426](../../src/autorec/optimized_data_structures/circuit_cache.py#L426) | Preserves prior fits and avoids RNG-consuming refits | **E** | All keys and payloads, including failure results, metrics, parameters, predictions, termination and penalties |
| LRU order and capacity | [circuit_cache.py:194](../../src/autorec/optimized_data_structures/circuit_cache.py#L194) | Determines future eviction | **E** | Ordered entries; restore without cache reads that update recency |
| Clock cache, reference bits, key list, hand, capacity | [circuit_cache.py:46](../../src/autorec/optimized_data_structures/circuit_cache.py#L46) | Determines second-chance eviction | **E** | Explicit snapshot of all fields |
| Cache hit/miss counters | Both cache implementations | Reporting | **H** | Exact counters |
| Training/evaluation environment aliases and dataset sharing | [agent.py:208](../../src/autorec/agent.py#L208) | Shared evaluation can mutate training cache | **E/H** | Explicit identity markers; store shared objects once |
| Separate evaluation environment and cache | [agent.py:328](../../src/autorec/agent.py#L328) | Reproduces subsequent evaluation state | **H/E if interleaved** | Separate snapshot when present |
| `NN_loss`, `success_rate`, `terminal_states` | [agent.py:896](../../src/autorec/agent.py#L896), [agent.py:932](../../src/autorec/agent.py#L932) | Accumulated results | **H** | Lossless DataFrames with indices, columns, and dtypes |
| Reward histories and rolling means | [agent.py:941](../../src/autorec/agent.py#L941); [agent.py:1348](../../src/autorec/agent.py#L1348) | Full reporting and next 50-episode rolling mean | **H** | Preserve complete lists and scalar types |
| `epsilon_list`, `beta_list` | [agent.py:315](../../src/autorec/agent.py#L315) | Schedule histories | **H** | Continue appending; retain existing pre-update recording convention |
| Output paths and model-export policy | [agent.py:290](../../src/autorec/agent.py#L290); [agent.py:987](../../src/autorec/agent.py#L987) | Artifact naming and cadence | **H** | Save policy; permit explicit output-root relocation |
| Episode locals: action index, deadloop chain, partial rewards/losses, episode history | [agent.py:1031](../../src/autorec/agent.py#L1031) | Controls unfinished episode | **T** | Omit at completed-trial boundaries |
| Sampled batches, gradients, tape, Q-values, fitter locals | [agent.py:792](../../src/autorec/agent.py#L792); [utils.py:638](../../src/autorec/utils.py#L638) | In-flight computation | **T** | Checkpoint only after completion |
| Compiled functions, plots, timers | TensorFlow/JAX runtime and reporting | Execution caches and presentation | **T/diagnostic** | Recreate; elapsed time and file timestamps are outside numerical equivalence |

Scalar **types** matter. Fitting uses JAX operations, and cache/history payloads can contain
framework scalar arrays. Converting all values to ordinary JSON numbers could change later
arithmetic or reporting precision.

The inspected first-party module globals include default configuration dictionaries,
AutoEIS/Julia handles, runtime environment settings, and data-preparation constants. Resolved
configuration and source/runtime identity cover these dependencies. Live library handles,
loggers, and compiled caches should be reconstructed rather than serialized. No additional
first-party mutable global training accumulator was found.

## 3. Reproducibility Assessment

### 3.1 Current implementation cannot resume exactly

Repeated calls to `train()` produce a hybrid state:

- Model, optimizer, epsilon, beta, and environment cache remain.
- Replay, counters, and most histories restart.
- Trial indices restart at zero.

This is neither a clean restart nor cumulative continuation.

Loading a model into a new agent loses still more state, including target-network lag and
the optimizer trajectory.

### 3.2 Total-trial dependence must be explicit

Defaults are resolved as:

```text
start_jump = 0.4 × num_trials
anneal_fraction ≈ 0.595
annealing duration = num_trials × anneal_fraction
```

The beta update subsequently uses `self.num_trials`.

For a 1,000-trial experiment, the default `start_jump` is 400. For a 100-trial experiment,
it is 40. Consequently, configuring ten independent 100-trial sessions cannot reproduce a
configuration intended for 1,000 trials.

Explicitly fixing `start_jump` alone is insufficient because the denominator still depends
on `num_trials`.

**Recommended rule:** initialize both scenarios with the same 1,000-trial horizon. A session
budget of 100 limits execution only; it must not change the horizon or resolved schedules.
Changing the horizon later cannot undo beta-dependent updates already performed.

### 3.3 Circuit caching is essential state

The fitter generates an initial guess and additional guesses during repeated attempts:

- [src/autorec/utils.py:652](../../src/autorec/utils.py#L652)
- [src/autorec/utils.py:677](../../src/autorec/utils.py#L677)

The installed AutoEIS implementation uses SciPy `loguniform.rvs()` and NumPy random draws.
A focused probe confirmed that restoring NumPy's legacy RNG reproduced those SciPy draws
exactly. The dependency implementation inspected was
`autoeis/utils.py:326–355` in the installed AutoEIS 0.0.44 package.

However, restoring RNG state without restoring the cache is insufficient. Different cache
misses cause different numbers of fits and random draws. Different eviction order
eventually causes the same problem.

### 3.4 Framework serialization needs explicit coverage

Dependency probes used installed TensorFlow 2.21.0, Keras 3.12.1, and NumPy 2.2.6.

| Probe | Observed result |
|---|---|
| TensorFlow checkpoint containing model and Adam | Model weights and optimizer variables restored exactly |
| Same checkpoint after advancing a Dropout seed generator | Saved seed `[17, 1]` restored as `[17, 0]` |
| `assert_consumed()` for that restore | Passed despite omitted seed state |
| `.keras` export of the uncompiled model | Did not contain the agent's external optimizer; also reset the Dropout seed state |
| Explicit inclusion of complete `model.variables` | Restored the seed state exactly |
| Architecture serialization plus checkpoint write | Preserved Python and NumPy RNG states in the probe |

Keras documents stateful `SeedGenerator` objects and their role in producing successive
random values. See [Keras SeedGenerator documentation](https://keras.io/api/random/seed_generator/).
The installed Keras `layers/layer.py:617–668` also distinguishes `variables`, which include
seed state, from `weights`, which exclude it.

These observations justify an explicit variable inventory. They do not prove that every
built-in or custom Keras layer is checkpoint-safe. The seed counter in the probe was
deliberately advanced to test serialization coverage; this does not imply that AutoREC's
current default training calls activate Dropout.

The first probe emitted an ignored TensorFlow cleanup exception during interpreter shutdown
after its checks. The follow-up completed without that diagnostic. Neither probe was a
full AutoREC training or scientific validation run.

### 3.5 Seeds are not RNG snapshots

Environment construction reseeds NumPy and consumes draws through `reset()`. Agent
construction seeds TensorFlow and initializes models. Keras initialization also consumes
Python randomness.

Restoration must therefore:

1. Construct all required objects.
2. Restore data, model, optimizer, replay, cache, progress, and histories.
3. Finish validation and any initialization.
4. Restore saved RNG states last.
5. Begin the next trial without another seed/reset operation beyond the normal trial reset.

Legacy TensorFlow random operations have internal operation counters. Saving a TensorFlow
global generator does not automatically recover every stream associated with
`tf.random.set_seed()`. See
[TensorFlow seed semantics](https://www.tensorflow.org/api_docs/python/tf/random/set_seed).

The default Dense training path contains no explicit TensorFlow random draws after
initialization. Initial exact-checkpoint support should therefore cover validated
configurations and reject custom untracked randomness.

Support should initially include the default Dense architecture and explicitly validated
built-in layer/optimizer combinations. Ordinary training should remain available for
broader configurations; an unsupported exact-checkpoint request should fail clearly.

### 3.6 Evaluation changes training state

Evaluation is not neutral:

- `eps_greedy()` consumes a NumPy draw even when epsilon is zero.
- Evaluation resets environments.
- Evaluation can run randomized fitting.
- Shared training/evaluation environments share cache mutations.

See [src/autorec/agent.py:726](../../src/autorec/agent.py#L726),
[src/autorec/agent.py:1492](../../src/autorec/agent.py#L1492), and
[src/autorec/agent.py:1728](../../src/autorec/agent.py#L1728).

For the initial exactness contract, perform evaluation after training finishes or in a
separate process loaded from an artifact. Evaluating between resumed sessions is equivalent
only if the continuous reference performs the same evaluations at the same points.

### 3.7 Execution constraints

Code changes can preserve state and execution order. They cannot establish bitwise equality
across arbitrary hardware and numerical environments.

Exactness should require:

- Identical AutoREC implementation and compatible checkpoint schema.
- Identical relevant dependency versions, including Python, TensorFlow/Keras,
  NumPy/SciPy, JAX/jaxlib, AutoEIS, Julia, and EquivalentCircuits.
- Matching hardware, device placement, and precision.
- Matching deterministic-operation, thread, and backend/compiler settings.
- Identical prepared data.
- No concurrent consumers of shared RNG state.
- No untracked external state in custom components.

TensorFlow's determinism guidance likewise requires matching hardware and software
conditions. See
[TensorFlow determinism documentation](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).

Repository-specific concerns include:

- `PYTHONHASHSEED` is set inside `set_global_seed()`, after interpreter startup; this cannot
  retroactively fix the interpreter's hash seed.
- Native thread variables are assigned in several places, sometimes after scientific
  imports. `environment.py` also assigns them unconditionally.
- TensorFlow determinism does not establish deterministic JAX, SciPy, or Julia behavior.
- Raw CSV discovery uses unsorted `rglob()` at
  [data_preparation.py:508](../../src/autorec/data_preparation.py#L508); reprocessing can
  change dataset order.
- Training mixes index-label and positional dataset access. Exact mode should initially
  require an index matching `0..len(dataset)-1`, rather than silently relabeling data.
- Fitting exceptions are converted into fallback rewards at
  [environment.py:513](../../src/autorec/environment.py#L513). Differences in numerical or
  runtime failures can therefore become trajectory differences.

No persistent data-loader iterator, first-party training worker pool, or JAX PRNG-key stream
was found in the training path. Optuna and scheduler state in hyperparameter tuning belong
to the outer experiment orchestration, not an individual agent checkpoint.

The code-fixable requirements are state retention, exact serialization, counter/schedule
continuation, safe restoration, validation, and neutral checkpoint writes. Matching numeric
execution conditions and avoiding concurrent external RNG consumers are execution
constraints. Real-fitting equality across fresh processes remains a validation question.

### 3.8 Supported checkpoint boundary

Checkpoint after:

1. All episodes for a trial.
2. All optimizer and target updates.
3. All history/statistics updates.
4. Existing scheduled model export, where applicable.
5. Epsilon and beta updates.
6. Incrementing `completed_trials`.

A checkpoint with `completed_trials = 100` means trials `0…99` are complete. The next trial
index is `100`.

Defer episode-boundary and mid-trial resumption. They require additional loop phase,
partial histories, environment state, deadloop state, and partial reward/loss accumulators.
Mid-fit or mid-gradient resumption would be substantially more complex.

Following a hard interruption, restart from the last committed checkpoint and replay
subsequent work. Do not save partially modified state from an arbitrary exception handler.
A future graceful-stop request can be honored at the next completed-trial boundary without
introducing mid-trial serialization.

### 3.9 Existing implementation issues

| Finding | Effect | Recommended treatment |
|---|---|---|
| `_record_terminal_state()` creates a new DataFrame, but `_run_episode()` does not return it to `train()` | Terminal history is lost | Separate small reporting fix |
| Agent `metadata` omits `gradient_steps` | Incomplete reconstruction metadata | Add field; use a dedicated complete checkpoint schema |
| `save_start` / `save_frequency` allow `None` in annotations but are used arithmetically | Some apparent configurations fail | Validate new checkpoint options separately |
| Latent-deadloop condition compares local episode index with `NN_sleep` | Suspicious existing behavior | Preserve during checkpoint work; investigate separately |
| Replay state strings use `U50` | Long chromosomes can truncate | Validate supported dimensions; avoid silently changing replay format |
| Replay sampling uses `replace=False` | Insufficient warm-up/capacity can fail | Validate compatible batch/buffer settings |

Relevant locations:

- [Terminal-state propagation — src/autorec/agent.py:1067](../../src/autorec/agent.py#L1067)
- [Metadata — src/autorec/agent.py:1868](../../src/autorec/agent.py#L1868)
- [Deadloop condition — src/autorec/agent.py:1266](../../src/autorec/agent.py#L1266)

Also preserve the current model-call semantics: `_train_model()` does not pass
`training=True`. Enabling Dropout or changing normalization behavior would be an algorithm
change, not checkpoint implementation.

## 4. Proposed Checkpoint File/Folder Structure

### 4.1 Recommended layout

Use an immutable directory per checkpoint:

```text
run/
├── models/                         # Existing inference/model exports
├── checkpoints/
│   ├── latest.json                 # Optional pointer to committed checkpoint
│   └── trial_00000100/
│       ├── manifest.json
│       ├── config.json
│       ├── models.json
│       ├── training_state.pkl
│       ├── environments.pkl
│       ├── replay.npy
│       └── tensors/
│           ├── state.index
│           └── state.data-00000-of-00001
└── ...                             # Existing final results and plots
```

| Component | Contents |
|---|---|
| `manifest.json` | Format version, boundary phase, completed trials, component sizes/checksums, source/package identity, runtime fingerprint, variable inventory |
| `config.json` | Effective agent/environment configuration, fixed horizon, resolved schedules, optimizer configuration, output policy |
| `models.json` | Full online/target architecture and build information |
| `training_state.pkl` | Progress, epsilon/beta, histories, Python/NumPy RNG states, supported additional RNG metadata |
| `environments.pkl` | Prepared datasets, environment snapshots, cache payloads/eviction metadata, alias markers |
| `replay.npy` | Exact structured replay allocation |
| `tensors/state.*` | Online/target variables, optimizer state, explicit layer RNG variables, supported framework generators |

This uses existing dependencies and separates inspectable metadata from large numerical
state. A directory fits TensorFlow's multiple-file checkpoint representation and avoids
adding an archive/unpacking layer. A single-file container can be considered later if
distribution requirements justify it.

### 4.2 Serialization choices

Use pickle for **explicit state dictionaries**, not whole agent/environment objects. This
follows existing repository conventions while preserving DataFrames, nested arrays, and
scalar types.

Load `replay.npy` with `allow_pickle=False`.

Use TensorFlow checkpoint serialization for the custom optimizer loop, with explicit
references covering complete model-variable collections. Architecture must be stored
separately because TensorFlow checkpoints primarily preserve variable state. See
[TensorFlow checkpoint documentation](https://www.tensorflow.org/guide/checkpoint).

Prefer `Checkpoint.write()` / `Checkpoint.read()` for the tensor component to avoid
introducing a save-number counter into training state.

The tensor root should contain the online model, target model, optimizer, explicit variable
references covering both models' `variables` collections, and any additional supported
generators. Validate these against the manifest inventory; `assert_consumed()` alone is
insufficient. Use stable logical identities and saved model structure rather than relying
on fresh process-global autogenerated names.

### 4.3 Atomic writes

1. Capture state synchronously at a supported boundary.
2. Write into a temporary sibling directory on the same filesystem.
3. Flush component files.
4. Compute component checksums.
5. Write the manifest last.
6. Validate completeness.
7. Rename to a previously nonexistent final directory.
8. Atomically replace `latest.json` afterward.

Never overwrite a valid checkpoint directory in place. A failure must leave the previous
checkpoint usable.

Reject duplicate final names unless an existing identical checkpoint is explicitly
verified. Temporary names should use filesystem/OS facilities rather than training RNG
streams.

Use filesystem synchronization where supported. Document that crash durability on network
storage requires validation of that filesystem's behavior.

The save path must not perform extra predictions, fitting, cache accesses, or optimizer
updates. Capture and verify the supported RNG states around serialization; restore them
on save failure if any supported serialization path can consume randomness. The save-only
equivalence control is the ultimate validation of observational neutrality.

### 4.4 Integrity and compatibility

Validate before returning a restored agent:

- Schema version and supported boundary.
- Required components, sizes, and hashes.
- Architecture and variable inventory.
- Optimizer configuration and slot coverage.
- Replay dtype, capacity, size, insertion position, and references.
- Cache structure, key sets, order, and Clock hand.
- Dataset content, index, row order, and representation.
- Progress/history consistency.
- Runtime compatibility.

Checkpoint hashes detect corruption; they do not authenticate malicious files.

Pickle checkpoints must be treated as **trusted-input artifacts**. Avoid claiming
cross-version or untrusted portability. Start with one explicit schema version and reject
unsupported versions; add migrations only when there is an actual supported migration to
test.

Strict resume should reject changes to:

- Data or action mapping.
- Network architecture.
- Optimizer or schedules.
- Training hyperparameters.
- Fixed horizon.
- Unsupported software/hardware execution conditions.

Allow explicit output-root relocation and changes to checkpoint frequency. Fine-tuning
with changed settings remains a separate model-loading workflow. Original input-model and
dataset paths are provenance, not prerequisites for a self-contained checkpoint restore.

Restore into a new agent. On failure, do not leave an existing agent partially modified,
and restore the caller's prior RNG states if reconstruction changed them. The loader must
not silently restart training or accept partially restored state.

## 5. Proposed API and User Workflow

The following API is proposed; it does not exist yet.

```python
agent.train(
    *,
    additional_trials=None,
    checkpoint_dir=None,
    checkpoint_frequency=None,
)

agent.save_checkpoint(path)

agent = DDQN_ECM.from_checkpoint(
    path,
    *,
    save_dir=None,
)
```

Expose `completed_trials` as a read-only progress property.

### 5.1 Cumulative semantics

- `num_trials` remains the experiment horizon and schedule input.
- `additional_trials=100` advances exactly 100 further trials.
- Reject exceeding the fixed horizon in the initial release.
- Checkpoint frequency is measured in completed global trials.
- Automatic checkpointing saves a session endpoint even if it does not fall on the interval.
- Partial sessions return cumulative histories.
- Final model/results/plots are produced when the horizon is reached.
- `save_model()` and `load_model()` retain their existing roles.

A manual checkpoint is valid only at an initialized or completed-trial boundary. During a
running trial, saving must be rejected or deferred to the next supported boundary. Initial
state creation must not consume additional randomness relative to ordinary training.

### 5.2 Backward compatibility

A fresh `agent.train()` must preserve today's full-run behavior.

Repeated legacy no-argument calls currently reset local trackers. Silently making them
cumulative would change observable behavior. Recommended policy:

- Preserve legacy behavior unless continuation is explicitly selected.
- Enter continuation mode through `additional_trials`, checkpoint options, or
  `from_checkpoint`.
- In continuation mode, no-argument `train()` completes the remaining horizon.

This distinction needs documentation and regression tests. It is a review decision listed
in Section 8, not an implemented behavior.

### 5.3 Normal training

```python
_, _, _, _, agent = pipeline_builder(config)
results = agent.train()
```

### 5.4 First session

The configuration must specify `agent.num_trials: 1000`.

```python
_, _, _, _, agent = pipeline_builder(config)

results = agent.train(
    additional_trials=100,
    checkpoint_dir="runs/experiment/checkpoints",
    checkpoint_frequency=100,
)

assert agent.completed_trials == 100
```

### 5.5 Resume in a new process

Apply the documented runtime setup before constructing scientific components. The
checkpoint loader restores RNG states; the user should not reseed afterward.

```python
agent = DDQN_ECM.from_checkpoint(
    "runs/experiment/checkpoints/trial_00000100"
)

results = agent.train(
    additional_trials=100,
    checkpoint_dir="runs/experiment/checkpoints",
    checkpoint_frequency=100,
)

assert agent.completed_trials == 200
```

### 5.6 Multiple sessions

```python
checkpoint = None

for ii in range(10):
    if checkpoint is None:
        _, _, _, _, agent = pipeline_builder(config)
    else:
        agent = DDQN_ECM.from_checkpoint(checkpoint)

    agent.train(additional_trials=100)

    checkpoint = (
        "runs/experiment/checkpoints/"
        f"trial_{agent.completed_trials:08d}"
    )
    agent.save_checkpoint(checkpoint)
```

This illustrates the sequence. The acceptance test must launch fresh processes between
sessions.

### 5.7 CLI

Add:

```text
--additional-trials
--checkpoint-dir
--checkpoint-frequency
--resume
```

Examples:

```bash
autorec train --config experiment.yaml \
  --additional-trials 100 \
  --checkpoint-dir runs/experiment/checkpoints \
  --checkpoint-frequency 100
```

```bash
autorec train \
  --resume runs/experiment/checkpoints/trial_00000100 \
  --additional-trials 100 \
  --checkpoint-dir runs/experiment/checkpoints \
  --checkpoint-frequency 100
```

Make `--config` and `--resume` mutually exclusive initially. Resume should use the
checkpoint's authoritative configuration and prepared data, without silently applying a
new seed or training overrides. Runtime setup may still be required before loading, but it
must match the saved execution contract.

## 6. File-by-File Modification Plan

Priorities:

- **P0:** required correctness foundation.
- **P1:** complete user-facing feature and documentation.
- **P2:** optional later extension.

| File | Current functionality | Proposed changes | Justification | Priority |
|---|---|---|---|---|
| [src/autorec/agent.py:889](../../src/autorec/agent.py#L889) — `train()` | Local replay, counters, histories, full-run finalization | Persistent training-state record, bounded continuation, boundary hook, cumulative results | Core continuation support | P0 |
| [src/autorec/agent.py:1309](../../src/autorec/agent.py#L1309) — epsilon/beta updates | Uses local trial indices and total horizon | Global indices and frozen resolved horizon; preserve arithmetic | Prevent schedule drift | P0 |
| [src/autorec/agent.py:376](../../src/autorec/agent.py#L376) — model setup | Initializes/clones models or loads model-only artifact | Internal restoration path with independent target recovery and controlled side effects | Avoid target loss and post-restore initialization | P0 |
| `src/autorec/checkpoint.py` — new | No complete checkpoint layer | Schema, serialization, validation, RNG handling, atomic publication, reconstruction orchestration | Keep file-format logic separate from RL updates | P0 |
| [src/autorec/optimized_data_structures/circular_buffer.py:10](../../src/autorec/optimized_data_structures/circular_buffer.py#L10) | Structured circular replay | Small explicit state export/restore methods | Preserve physical layout and pointer | P0 |
| [src/autorec/optimized_data_structures/circuit_cache.py:34](../../src/autorec/optimized_data_structures/circuit_cache.py#L34) | Clock/LRU caching | State export/restore and invariant checks for both policies | Preserve fitting and RNG trajectory | P0 |
| [src/autorec/environment.py:68](../../src/autorec/environment.py#L68) | Seeds, resets, stores data/cache | Explicit environment snapshots and validated restoration | Restore state without extra random reset | P0 |
| [src/autorec/agent.py:1868](../../src/autorec/agent.py#L1868) — metadata | Configuration summary omits `gradient_steps` | Add omitted field; keep dedicated checkpoint schema complete | Reproducible configuration | P0 |
| [src/autorec/factory/optimizer.py:66](../../src/autorec/factory/optimizer.py#L66) | Optimizer/schedule configuration helpers | Reuse; extend only if supported round trips expose gaps | Avoid duplicate factory logic | P0 validation |
| [src/autorec/factory/model.py:96](../../src/autorec/factory/model.py#L96) | Layer construction/configuration | Reuse normal construction; use full Keras serialization for checkpoint architecture | Preserve complete model definition | P0 validation |
| [src/autorec/runtime.py:15](../../src/autorec/runtime.py#L15), [src/autorec/utils.py:74](../../src/autorec/utils.py#L74) | Runtime configuration and seeding | Separate compatibility validation from reseeding; document startup requirements | Avoid resetting restored RNG streams | P0/P1 |
| [src/autorec/cli.py:81](../../src/autorec/cli.py#L81) | Fresh config-based training | Resume and session/checkpoint arguments | Existing CLI integration | P1 |
| [src/autorec/agent.py:1067](../../src/autorec/agent.py#L1067) — terminal history | Updated DataFrame lost in local scope | Return and propagate it | Separate reporting-correctness fix | P1, separate patch |
| `tests/test_checkpoint.py` — new | No checkpoint format tests | State round trips, integrity, compatibility, failure atomicity | Serialization correctness | P0 |
| `tests/test_checkpoint_equivalence.py` — new | No complete equivalence tests | Fresh-process continuous/save-only/resumed comparisons | Main acceptance criterion | P0 |
| Existing agent/cache/factory/runtime/CLI tests | Focused component coverage | Counters, boundaries, RNG coverage, legacy behavior | Regression protection | P0/P1 |
| [README.md:137](../../README.md#L137), [README.md:490](../../README.md#L490) | Training and CLI guidance | Checkpoint API, fixed horizon, compatibility, limitations | User-facing documentation | P1 |
| [tutorials/02_training_walkthrough.ipynb](../../tutorials/02_training_walkthrough.ipynb) | Model-saving walkthrough uses checkpoint terminology | Distinguish model exports from full checkpoints; add continuation example | Correct expectations | P1 |
| New checkpoint example and example configuration | No dedicated working example | Fresh run, checkpoint save, resume, repeated sessions | Demonstrable workflow | P1 |
| [.github/workflows/ci.yml](../../.github/workflows/ci.yml), [pyproject.toml](../../pyproject.toml) | Regular pytest execution | Slow-test markers and deterministic test jobs | Maintainable validation | P1 |

The existing pipeline builder can remain unchanged if session options stay on `train()` and
the CLI. Existing factory APIs should not be renamed or broadly refactored.

No algorithm changes are required in replay sampling, fitting, reward calculation, or
circuit simplification for the initial scope. The checkpoint state refactoring should be
validated against ordinary training before adding serialization.

The terminal-history correction intentionally changes previously lost output and should
be reviewed as such. Other changes should preserve existing fresh-run numerical behavior;
strict checkpoint validation may reject unsupported configurations without restricting
ordinary non-checkpoint training.

Existing untracked user files should remain untouched unless explicitly included in the
approved implementation scope.

## 7. Testing and Validation Plan

### 7.1 Component tests

Exact round trips must cover:

- Online and target models with deliberately different state.
- Nontrainable variables and advanced layer RNG counters.
- Adam moments, optimizer iteration, SGD momentum.
- Constant LR and an iteration-dependent schedule.
- Optimizer state before and after the first update.
- Empty, partial, full, and wrapped replay.
- Replay physical ordering, insertion pointer, and priorities.
- LRU order and the next eviction.
- Clock bits, hand, key order, and the next eviction.
- Cache payloads containing complex arrays, framework scalars, and failed-fit results.
- Shared and separate environments/datasets.
- Histories, DataFrame dtypes, and scalar types.
- Python and NumPy RNG states, including cached Gaussian values.
- Every additional supported framework generator.

RNG tests should compare saved states and the next sequence of draws after restoration.

### 7.2 Primary equivalence test

Run three scenarios with the same fixed 1,000-trial configuration:

| Run | Execution |
|---|---|
| A | 1,000 continuous trials without full checkpoint creation |
| B | 1,000 continuous trials, writing checkpoints every 100 trials without loading them |
| C | Ten 100-trial sessions, each resumed session in a fresh process |

Comparisons:

- **A versus B:** checkpoint creation is observationally neutral.
- **B versus C:** restoration preserves continuation.
- **A versus C:** the requested complete equivalence.

Keep existing model-export cadence identical in all runs. Initialize each independent
reference experiment with the same setup and seeds. Do not construct multiple reference
agents sequentially in one shared random stream and assume their initial states match.

At every shared boundary, compare:

1. All online and target variables.
2. Optimizer variables and effective LR.
3. Complete physical replay array, dtype, size, pointer, and priorities.
4. Epsilon, beta, and progress counters.
5. Complete histories.
6. Caches, eviction metadata, and statistics.
7. RNG states.
8. Environment boundary state.
9. Frozen configuration and schedules.

Also collect a test-only trajectory trace:

- EIS and initial-state selections.
- Actions, rewards, termination flags.
- Cache hits and misses.
- Replay sample indices and importance weights.
- Losses and optimizer-update events.
- Target synchronization events.

Tracing must not introduce extra sampling, fitting, predictions, or cache reads.

For bitwise validation, compare array shape, dtype, and raw bytes. Use exact DataFrame
comparison with dtype checking. Approximate tolerances are inappropriate. Ordinary
numerical equality alone can overlook signed-zero or NaN-payload differences.

Exclude wall time, timestamps, output paths, and archive headers from numerical
equivalence. Compare decoded training state rather than requiring checkpoint files or
plots to have identical container metadata.

### 7.3 Routine CI test

Use **60 trials versus six 10-trial sessions**, plus the save-only control.

Use:

- Real Keras networks.
- Real optimizer updates.
- Real replay sampling.
- Real environment reset/action/cache behavior.
- A controlled substitute for expensive fitting that consumes randomness reproducibly.

Select parameters that exercise:

- Warm-up completion.
- Replay wraparound.
- Training and target events across split boundaries.
- Epsilon and beta changes.
- LR schedule transitions.
- Multiple gradient steps.
- The 50-episode smoothing window.
- Terminal and nonterminal outcomes.

Assert that these events actually occurred. A test that never trains a network is
insufficient. Include a split while target and online weights differ, rather than testing
only boundaries that happen to synchronize them.

### 7.4 Scientific validation

Run the requested **1,000 versus ten × 100** comparison as a slower validation job.

Use two levels:

1. **Controlled fitter:** establishes checkpoint/RL integration behavior.
2. **Real fitter on a fixed small dataset:** establishes behavior of the complete scientific
   path under a pinned execution environment.

Passing the first must not be presented as proof of the second. Qualify CPU execution
first, and add GPU qualification only after dedicated exact-equivalence tests pass.

The existing tests include
[real optimizer updates — tests/test_agent.py:156](../../tests/test_agent.py#L156) and
[buffer/cache checks — tests/test_data_structures.py:9](../../tests/test_data_structures.py#L9),
but no complete fresh-process training-equivalence test.

### 7.5 Failure and backward-compatibility tests

Cover:

- Missing or corrupt components.
- Truncated tensor files and invalid pickle/NumPy payloads.
- Unsupported schema versions.
- Architecture, data, optimizer, horizon, and runtime mismatches.
- Invalid replay/cache/counter invariants.
- Checkpoint requests during incomplete trials.
- Model-only files passed as training checkpoints.
- Duplicate checkpoint names.
- Interrupted writes at each publication phase.
- Invalid `latest.json`.
- Failed restores preserving caller RNG and existing objects.
- Output-root relocation.
- Checkpoints before warm-up and at final completion.
- Multiple episodes per trial.
- Legacy no-checkpoint behavior, including repeated no-argument calls.
- No additional model initialization or random reset after restoration.

### 7.6 Documentation and examples

Documentation must explain:

- What full checkpoints preserve.
- Why model files are insufficient.
- How cumulative trial counts work.
- Why the horizon must be fixed from initialization.
- Saving and resuming through Python and CLI.
- Trusted-input requirements.
- Compatibility policy.
- Exactness requirements and unsupported configurations.
- Recovery from interruption.
- Why evaluation between sessions can alter training.

Add a dedicated example showing normal training, periodic checkpointing, one resume, and
repeated sessions. Keep the example independent of the user's existing untracked
experimental files.

The training tutorial's current model-save descriptions need correction. The evaluation
tutorial may need a short clarification that loading a model is appropriate for inference
but not exact training recovery. The data-preparation tutorial should emphasize retaining
the prepared dataset used for an experiment; no preprocessing algorithm change is needed.

## 8. Risks, Tradeoffs, and Open Questions

| Decision | Preferred approach | Tradeoff |
|---|---|---|
| Checkpoint boundary | Completed trial | Small implementation; interrupted work after the last checkpoint is replayed |
| Schedule horizon | Fixed at initial construction | Exact comparison is unambiguous; extension beyond the horizon requires separate semantics |
| Data storage | Embed prepared datasets | Larger checkpoints; avoids mutable paths and preprocessing drift |
| Serialization | TensorFlow + NumPy + explicit pickle state + JSON | Minimal dependencies; trusted inputs and restricted compatibility |
| Generic Keras support | Validated configuration set | Honest exactness scope; unsupported custom state fails clearly |
| RNG ownership | Preserve existing global streams initially | Minimal trajectory change; concurrent use requires restrictions |
| Legacy repeated training | Preserve until continuation explicitly selected | More API explanation; avoids silent behavior change |
| Terminal-history defect | Separate reporting fix | Changes previously lost output, but isolates it from checkpoint semantics |
| Retention | No automatic deletion initially | Higher disk usage; recovery points are not silently removed |
| GPU support | Qualify after dedicated equivalence testing | CPU validation is simpler; broader device support takes additional evidence |

### 8.1 Decisions requiring review

The most consequential design decisions are:

1. Whether to approve the fixed-horizon rule for the first release.
2. Whether trusted pickle state is acceptable.
3. Which model/optimizer configurations must be supported initially.
4. Whether to include the terminal-history correction as a separate accompanying patch.
5. Whether preserving repeated legacy `train()` behavior is required, or whether a
   documented cumulative behavior change is acceptable.

### 8.2 Unresolved technical risks

- Real-fitting bitwise behavior across fresh JAX/Julia processes has not been demonstrated.
- Less common layers and optimizer wrappers may contain state beyond the validated variable
  inventory.
- Different numerical failures can become fallback rewards and diverging trajectories.
- Network-filesystem durability must be assessed separately from local atomic rename.
- Exact arbitrary TensorFlow legacy RNG restoration is not covered by ordinary seed
  snapshots.
- Raw-data reprocessing and dataset reordering must not occur during resume.

These are validation gates or support limitations, not reasons to expand the initial design
into a general checkpoint framework. Moving all randomness to new local generators would
be a broader behavioral change and should not be bundled into the first checkpoint release.

## 9. Recommended Implementation Sequence

1. **Approve the exactness contract.** Establish supported trial boundary, fixed horizon,
   cumulative counts, runtime constraints, trusted-input format, and supported component set.

2. **Build a deterministic baseline harness.** Record the current action/update order,
   warm-up timing, target cadence, schedules, and no-checkpoint behavior before structural
   changes.

3. **Persist cross-trial state.** Move replay, counters, and histories into one agent-owned
   record while preserving arithmetic and random-call order.

4. **Add bounded continuation.** Implement `additional_trials`, `completed_trials`, fixed
   schedules, and partial-session behavior. Verify same-process splitting first.

5. **Add buffer/cache state methods.** Validate exact replay layout and both eviction
   policies, including subsequent sampling/insertion/eviction behavior.

6. **Implement model, optimizer, and RNG coverage.** Include complete model variables and
   seed counters. Test checkpoints before and after optimizer slot creation.

7. **Implement the versioned checkpoint format.** Add effective configuration, prepared
   data, environment snapshots, histories, manifest checks, and atomic publication.

8. **Implement fresh-process reconstruction.** Restore into a new agent, validate
   completeness, and install saved RNG state last. Fail without partial recovery.

9. **Pass three-way equivalence testing.** First run the smaller controlled test, then
   1,000 continuous trials versus ten 100-trial sessions, including the save-only control.

10. **Expose the feature through CLI and documentation.** Update README and training
    tutorial; add a focused checkpoint example and configuration.

11. **Apply the terminal-history correction separately, if approved.** Validate it as a
    reporting change so it does not obscure trajectory-preservation testing.

12. **Run real-fitting qualification.** Execute the larger exact comparison in the pinned
    scientific environment. Report actual equality results and only claim exact continuation
    for configurations that pass.

Phase 2 should begin with the state-preservation and equivalence harness work, before
committing to broad model compatibility or convenience features. Implementation remains
subject to the user's review and authorization.
