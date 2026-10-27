# Models and results

AutoREC writes model exports, training diagnostics, and evaluation records to
the agent's `save_dir`. The CLI `--output-dir` option overrides that YAML value.
Creating an agent immediately creates the output directory, a `models/`
subdirectory, and `model_summary.txt`.

## Model compatibility

`DDQN_ECM.save_model()` writes a Keras model (`.keras` recommended; `.h5`
supported for legacy use). Loading clones the target network from the loaded
main model and derives hidden-layer configuration from it.

The active environment must produce the input and action dimensions expected by
the model. Preserve at least:

- the ordered environment `elements` list;
- `chromosome_HEAD_len` and the derived chromosome encoding;
- the prepared `flatten_Z` length and feature order;
- compatible TensorFlow/Keras and custom-layer requirements.

Shape incompatibilities normally fail during loading or first use. Matching
shapes alone do not establish that a model is scientifically appropriate for a
new dataset.

```{warning}
A `.keras` export is not a complete training checkpoint. It does not preserve
the replay buffer, optimizer progress as used by the running agent, target
network, exploration/replay schedules, counters, caches, histories, or random
state required for exact continuation.
```

## Training outputs

After `agent.train()` completes, `save_dir` contains:

| Path | Content |
| --- | --- |
| `dqn_model.keras` | Final main-network export |
| `models/dqn_model_trial_<n>.keras` | Periodic main-network exports |
| `model_summary.txt` | Keras architecture summary |
| `success_rate.csv` | Cumulative training success-rate history |
| `NN_loss.csv` | Recorded neural-network losses |
| `terminal_states.csv` | Successful terminal states and fit details |
| `statistical_analysis.pkl` | Per-episode cumulative and smoothed rewards |
| `epsilon_beta.png` | Exploration/replay schedule plot |
| `reward_plot.png` | Training reward plot |
| `loss.png` | Neural-network loss plot |

Periodic exports begin at `save_start` and repeat every `save_frequency` trials.
They are model snapshots, not resumable training-state checkpoints.

## Evaluation outputs

`eval_batch_eis()` returns one row per evaluated spectrum with identifiers,
environment type, ground truth, success flag, best reward/action/circuit/coding,
fit metrics and parameters, total actions, and action history. The CLI saves:

```text
eval_results_<run-id>.csv
eval_results_<run-id>.pkl
```

Use CSV for quick human inspection. Use the pickle as the lossless Python form
because metrics, parameters, and histories can contain nested objects or arrays.
Only unpickle trusted files.

Fit plots and optional circuit-evolution GIFs may also be written during
evaluation. Their exact presence depends on the selected evaluation options and
whether a valid fitted circuit is available.

## Reporting results

Keep each experiment in its own output directory and store its exact YAML,
dataset identity, AutoREC/dependency versions, seed, and invocation alongside
the artifacts. Do not infer general model accuracy from the included demo model
or a single stochastic run. Define success and aggregation choices explicitly
when reporting `found_solution`, rewards, or exact-circuit matches.
