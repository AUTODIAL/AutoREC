# Configuration guide

AutoREC's factory accepts a YAML file or Python dictionary with `dataprep`,
`environment`, and `agent` sections. Section keys are passed as constructor
arguments to the corresponding AutoREC class, with a few factory-specific rules
described below.

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

If `action_cap` is `null`, evaluation must receive an explicit `max_actions`
argument or CLI `--max-actions` value.

## Configuration validation

Before a long run, build the pipeline and inspect the selected data and model
shapes in a small, controlled test. YAML parsing and factory construction catch
structural errors but do not establish scientific suitability. The definitive
argument lists are in the {doc}`API reference <../api/index>`.
