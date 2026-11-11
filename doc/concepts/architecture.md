# Core concepts and architecture

AutoREC frames equivalent-circuit-model (ECM) generation as a sequential
decision problem. A prepared electrochemical impedance spectroscopy (EIS)
spectrum is combined with a circuit state, a reinforcement-learning agent
chooses a circuit mutation, and the environment fits and scores the resulting
circuit.

## Workflow at a glance

1. {class}`~autorec.data_preparation.EISDataPrep` loads or prepares EIS data,
   creates a fixed-length feature vector, and supplies fit-quality thresholds.
2. {class}`~autorec.environment.EIS_ECM_Env` selects a spectrum, represents an
   ECM as a gene-expression-programming chromosome, and defines the available
   mutations.
3. {class}`~autorec.agent.DDQN_ECM` selects mutations with a Double Deep
   Q-Network (DDQN), stores transitions in prioritized replay, and updates its
   main network from replay mini-batches.
4. During evaluation, the agent records the best circuit visited, its fitted
   parameters and metrics, and the action history.

The {func}`~autorec.factory.pipeline.pipeline_builder` factory connects these
components from one YAML configuration. Direct construction remains useful for
research that replaces or extends an individual component.

## Circuit representation

AutoREC represents topology with chromosome strings. The environment maps the
chromosome to a circuit understood by AutoEIS before fitting it to the active
spectrum. The supported terminal symbols are `R` (resistor), `L` (inductor),
`C` (capacitor), and `P` (constant-phase element); `+` and `/` represent series
and parallel composition.

The configured `elements` list determines both the one-hot state encoding and
part of the action space. A valid list must include `R` and at least one of `+`
or `/`. Its order is part of a saved model's compatibility contract.

## State, actions, and rewards

The neural-network input concatenates the encoded chromosome with the prepared
`flatten_Z` features. An action changes one chromosome position to one of the
configured symbols. Consequently, the following choices affect model input or
output dimensions:

- the frequency-grid length and ordered `eis_features` used for `flatten_Z`;
- `chromosome_HEAD_len`, which also determines the chromosome tail length;
- the ordered `elements` list and resulting action list.

The environment fits valid candidate circuits and derives its reward from fit
quality and circuit complexity. Dataset-specific `chi_thresh` and `r2_thresh`
values determine when a fit is accepted. Repeated fitted circuits can be cached
with either the `lru` or `clock` policy to avoid repeating equivalent fitting
work.

## Training and evaluation lifecycle

Training uses epsilon-greedy exploration, a main network, a target network, and
prioritized experience replay. `train_frequency` controls how often a training
event occurs; `gradient_steps` controls how many independently sampled replay
mini-batches are updated at each event. The target network is synchronized at
`update_target_frequency` intervals.

Evaluation is still stochastic unless all relevant runtime behavior is
controlled. A random seed helps comparison, but identical results are not
guaranteed across arbitrary hardware, TensorFlow/JAX versions, or external
fitting behavior. Report the configuration, package versions, seed, dataset,
and model together with scientific results.

```{important}
A `.keras` file stores the neural-network model. It does not store the replay
buffer, target-network state, schedules, counters, environment caches, or
random-number-generator state needed to resume an interrupted training run
exactly.
```

## Next steps

- Prepare compatible inputs with the {doc}`data guide <../user_guide/data>`.
- Define a pipeline with the {doc}`configuration guide
  <../user_guide/configuration>`.
- Understand saved artifacts in {doc}`models and results
  <../user_guide/models_and_results>`.
- Consult the {doc}`API reference <../api/index>` when extending components.
