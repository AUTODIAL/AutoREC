# Progressive reward exploration

This collection applies four reward definitions to the same three-stage
progressive-spectrum workflow. The example code does not change AutoREC's package defaults.

## Experiments

| Example | Complexity term after a successful fit |
| --- | --- |
| [`baseline/`](baseline/) | Standard component-count and depth penalties on the proposed circuit. |
| [`reward_calculated_on_simplified_circuits/`](reward_calculated_on_simplified_circuits/) | Standard penalties after structural circuit simplification. |
| [`reward_calculated_with_fim_on_simplified_circuits/`](reward_calculated_with_fim_on_simplified_circuits/) | Number of weak normalized Fisher information matrix (FIM) directions after simplification. |
| [`reward_calculated_with_continuous_fim_on_simplified_circuits/`](reward_calculated_with_continuous_fim_on_simplified_circuits/) | Bounded continuous penalty for weak normalized FIM directions after simplification. |

The reward implementations and configurations are included in this directory. EIS pickle
files are shared through the sibling `example/data/` directory. See each case README for
the exact reward definition.

## Common progressive workflow

Every configuration in this directory explicitly selects:

- `EqualPointWindowSelector` with `num_windows: 3`;
- `InterpolatingSpectrumEncoder` for a fixed 79-point neural-network observation;
- `PreparedSpectrumSource` for the measured arrays stored in the shared pickles; and
- `CarryBestStatePolicy` to initialize each later stage from the preceding best ECM.

Window 0 contains the highest-frequency third of the measurements. Stage 1 uses window 0,
stage 2 uses windows 0 and 1, and stage 3 uses all three windows. The encoder interpolates
only the neural-network observation to its fixed dimension. Circuit fitting, reward
calculation, and FIM calculation use all original measured `freq` and `Z_true` values in
the active cumulative window. The final stage therefore fits all 79 measurements.

The shared legacy pickles do not contain preparation metadata, so these YAML files state
`frequency_npoints: 79` explicitly. That value declares the observation capacity; it does
not replace or resample the physical fitting arrays.

## Single-run training and evaluation

Run commands from the selected case directory so the shared `../../data/` paths have the
same meaning in every exploration case. With no `--config`, the runner uses the local
10,000-trial exploration configuration:

```bash
cd example/progressive_exploration/baseline
python main_factory.py
```

To choose the configuration and output directory explicitly, run:

```bash
python main_factory.py \
  --config ../configs/pipeline_config.yaml \
  --output-dir calculation_output
```

Replace `baseline` with another case directory to change the reward. Each runner trains or
loads one model, then evaluates the configured training dataset and the shared test,
battery, and experimental datasets. It writes dataset-specific CSV and pickle results
under `<output-dir>/results/`, along with fit plots. Progressive result rows include
`progressive_num_windows`, `progressive_completed_all_stages`, and
`progressive_stage_results` in addition to the final-stage ECM fields.

Use a new output directory for every method and configuration. If `dqn_model.keras`
already exists, the runner loads it instead of training. Reusing a full-spectrum model for
a progressive experiment is not a controlled comparison, even though the observation
dimension is unchanged.

## Slurm execution

Each case contains `submit_job.sh`, which selects the same progressive configuration.
Review its account, constraint, environment path, resources, and repository location before
submission:

```bash
cd example/progressive_exploration/baseline
sbatch submit_job.sh
```

The scripts write to `calculation_output/` and archive model and fit-plot directories after
a successful run to reduce HPC file counts.

For scratch execution, preserve the relative layout between `progressive_exploration/` and
its sibling `data/` directory. The single-run reward modules are local to the exploration
folder and do not import `example.reward_exploration`. The job scripts explicitly limit
NumExpr and the other native thread pools to one thread and exit immediately if Python
fails.

## Checkpointed multi-fidelity tuning

The existing checkpoint-safe orchestrator from `example/reward_exploration/tuning/` is
reused by the tuning entry points. Candidates train in successive cumulative trial rungs,
and promoted candidates resume their complete agent, optimizer, replay, RNG, environment,
and progressive runtime state.

Install the optional orchestration dependencies once:

```bash
pip install -r example/reward_exploration/tuning/requirements.txt
```

Then run or submit a case-local orchestrator:

```bash
cd example/progressive_exploration/baseline
python tuning_runner.py
# or
sbatch tuning_runner_job.sh
```

`tuning_base_config.yaml` uses the shared training data for RL interactions and validation
data for promotion scores. Test, battery, and experimental data remain held out
from tuning. `tuning_search_space.yaml` matches the corresponding full-spectrum reward
case. Outputs are written under the case's `hyperparameter_tuning/` directory, including
immutable checkpoints and rung-specific validation results.

Progressive settings are part of checkpoint compatibility. Resume a candidate only with
the same reward environment, progressive components, data preparation, and deterministic
runtime settings. Start a separate checkpoint series when any of those change.

## Comparisons and limitations

The four cases in this directory use the same progressive components, data, seeds, and RL
hyperparameters, isolating the configured reward change. A comparison against
`reward_exploration` additionally changes the data-presentation schedule: a three-window
episode can allow up to three times as many stage-level action opportunities and fitting
calls as one full-spectrum episode. Equal training trials therefore do not imply equal
compute or equal numbers of circuit evaluations.

Downstream comparisons should report both training budget and actual action/fitting cost,
use replicated seeds, and evaluate the same held-out spectra. The included tests validate
configuration, data routing, checkpoint restoration, and reward mechanics; they do not
establish that a reward or spectrum schedule improves ECM quality or RL performance.
