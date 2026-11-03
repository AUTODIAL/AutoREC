# Configuration guide

AutoREC's factory accepts a YAML file or Python dictionary with `dataprep`,
`environment`, and `agent` sections, plus an optional `progressive_spectrum`
section. Section keys are passed as constructor arguments to the corresponding
AutoREC class, with a few factory-specific rules described below.

Start from `configs_yaml/examples/demo_pipeline_config.yaml` in the repository
and keep experiment-specific copies outside the tracked example.

## Pipeline structure

```yaml
dataprep:
  path: ${AUTOREC_ROOT}/data/examples/training_dataset.pkl
  mode: load
  evaluation: false

environment:
  elements: ["+", "/", R, L, P]
  initial_state: [+RRRRRR, ++RRRRRR, +++RRRRRR]
  seed: 42
  chromosome_HEAD_len: 10
  cache_enabled: true
  cache_capacity: 20000
  cache_type: lru

agent:
  save_dir: runs/experiment
  action_cap: 24
  random_seed: 42
  num_trials: 50
  hidden_layers:
    - type: Dense
      units: 40
      activation: relu
  optimizer:
    type: Adam
    learning_rate: 0.0005
```

`${AUTOREC_ROOT}` is expanded when YAML is read. AutoREC defines it as the
repository root only when the variable is not already set, so an existing
environment value takes precedence.

## Training and evaluation sections

`dataprep` is optional. If it is absent, every configured environment must
provide a `dataset` path or DataFrame. When `dataprep` is present, its resulting
DataFrame replaces `environment.dataset`.

Separate datasets and environments use `training` and `eval` subsections:

```yaml
dataprep:
  training:
    path: data/train.pkl
    mode: load
  eval:
    path: data/eval.pkl
    mode: load
    evaluation: true

environment:
  training:
    elements: ["+", "/", R, L, P]
    initial_state: [+RRRRRR]
    chromosome_HEAD_len: 10
  eval:
    elements: ["+", "/", R, L, P]
    initial_state: [+RRRRRR]
    chromosome_HEAD_len: 10

agent:
  save_dir: runs/experiment
  action_cap: 24
```

An `eval` subsection requires a `training` subsection. Without a `dataprep.eval`
section, the prepared training dataset is copied for evaluation. Without an
`environment.eval` section, no separate evaluation environment is built; the
agent falls back to its training environment internally. With the
single-environment demonstration configuration, `--use-training-env` makes that
selection explicit but does not select a different object.

Do not specify `training_env` or `eval_env` in `agent`; the pipeline factory
supplies both.

## Data-preparation options

The `dataprep` keys mirror
{class}`~autorec.data_preparation.EISDataPrep`. Its special `output` key tells
the factory to save the prepared DataFrame as a pickle. See {doc}`data <data>`
for schema and interpolation rules.

## Environment options

The environment section mirrors {class}`~autorec.environment.EIS_ECM_Env`.
The most consequential compatibility settings are:

- `elements`: ordered circuit symbols controlling the encoding and actions;
- `chromosome_HEAD_len`: circuit chromosome size;
- the dataset's `flatten_Z` length;
- cache settings (`cache_enabled`, `cache_capacity`, and `cache_type`).

Use the same model-shaping settings for training and later model evaluation.
`cache_type` accepts `lru` or `clock`; disabling the cache can substantially
increase repeated circuit-fitting work.

## Agent and neural-network options

The agent section mirrors {class}`~autorec.agent.DDQN_ECM`. Notable groups are:

- output and episode limits: `save_dir`, `save_start`, `save_frequency`, and
  `action_cap`;
- DDQN training: `num_trials`, `gamma`, `batch_size`, `train_frequency`,
  `gradient_steps`, `update_target_frequency`, `NN_sleep`, and
  `buffer_capacity`;
- exploration and replay: epsilon settings, deadloop settings, and prioritized
  replay alpha/beta settings;
- model construction: `hidden_layers` and `optimizer`.

`hidden_layers` uses built-in `tf.keras.layers` class names, with remaining
keys passed to the layer constructor. AutoREC supplies the input and linear
Q-value output layers. Optimizer and learning-rate schedule `type` values are
case-sensitive built-in Keras class names:

```yaml
optimizer:
  type: Adam
  learning_rate:
    type: ExponentialDecay
    initial_learning_rate: 0.0005
    decay_steps: 1000
    decay_rate: 0.96
    staircase: true
```

Keep `gradient_steps: 1` as a baseline. Larger values perform additional fresh
replay sampling and optimizer updates at each training event, increasing cost
and changing training behavior.

For retained or manually checkpoint-resumed training, `num_trials` is the number of
trials in the current session; `completed_trials` is cumulative. With the explicit CLI
`--resume-latest` option, `num_trials` is instead the cumulative target. Open-ended
workflows should configure absolute schedule timing with `start_jump` and
`beta_anneal_trials`. The legacy `anneal_fraction` remains supported when
`beta_anneal_trials` is omitted, but its duration is resolved from the initial
`num_trials`; the two beta-duration options are mutually exclusive. See
{doc}`checkpointing <checkpointing>` before changing session lengths or relying
on exact continuation.

If `action_cap` is `null`, evaluation must receive an explicit `max_actions`
argument or CLI `--max-actions` value.

## Progressive-spectrum options

Progressive execution is opt-in. Omitting `progressive_spectrum`, setting
`enabled: false`, or using `num_windows: 1` without explicitly enabling it
preserves the ordinary full-spectrum training and evaluation path. When enabled,
the same runtime is used automatically by `agent.train()` and batch evaluation:

```yaml
progressive_spectrum:
  enabled: true
  num_windows: 4
  window_selector:
    type: EqualPointWindowSelector
    parameters: {}
  observation_encoder:
    type: InterpolatingSpectrumEncoder
    parameters: {}
  spectrum_source:
    type: PreparedSpectrumSource
    parameters: {}
  stage_transition_policy:
    type: CarryBestStatePolicy
    parameters: {}
```

`EqualPointWindowSelector` divides points in their source order and exposes
cumulative windows: stage 0 uses window 0, stage 1 uses windows 0 and 1, and so
on. It expects the source spectrum to be ordered from high to low frequency.
`PreparedSpectrumSource` reads the prepared `freq` and `Z_true` columns.
`RetainedSpectrumSource` instead reads `source_freq` and `source_Z`, which are
available when data preparation retained the original measurements.

The encoder always produces the feature dimension expected by the network while
the environment fits every physical point selected for the current stage.
`InterpolatingSpectrumEncoder` maps selected points to the configured prepared
frequency grid. `RawSpectrumEncoder` preserves selected measurements, pads short
observations, and deterministically downsamples only the observation when needed:

```yaml
observation_encoder:
  type: RawSpectrumEncoder
  parameters:
    padding_value: -1.0
    normalizer:
      type: SentinelSafeMinMaxNormalizer
      parameters: {}
    downsampler:
      type: UniformIndexDownsampler
      parameters: {}
```

Padding and downsampling never remove points from physical ECM fitting. The raw
encoder's sentinel-safe normalizer keeps valid features in `[0, 1]`, so `-1`
remains an unambiguous padding value. `LegacyEISFeatureNormalizer` is available
for controlled comparisons with existing feature conventions, subject to its
documented sentinel constraints.

`CarryBestStatePolicy` starts the next stage from the best state found in the
current stage and continues until every configured stage is complete. Component
types are registry names rather than import paths. Custom Python components can
be supplied through direct APIs, but complete checkpoint restoration currently
supports only the registered built-ins. See the
{doc}`progressive API reference <../api/progressive_spectrum>` and the runnable
[`example/progressive_checkpoint/`](https://github.com/AUTODIAL/AutoREC/tree/main/example/progressive_checkpoint).

## Configuration validation

Before a long run, build the pipeline and inspect the selected data and model
shapes in a small, controlled test. YAML parsing and factory construction catch
structural errors but do not establish scientific suitability. Progressive
windowing changes the sequence of fitting datasets and generally requires
training or fine-tuning for that observation distribution. The definitive
argument lists are in the {doc}`API reference <../api/index>`.
