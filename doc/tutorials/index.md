# Tutorials

The tutorials follow the core AutoREC workflow from preparing EIS spectra to
training and evaluating a DDQN agent. Each page renders the existing notebook
from the repository with its saved outputs. The website build never starts a
kernel or reruns preprocessing, fitting, training, or evaluation.

## Run the notebooks locally

Complete the {doc}`../getting_started/installation`, activate `autorec_env`,
and start Jupyter with `tutorials/` as the working directory:

```bash
conda activate autorec_env
cd tutorials
jupyter lab
```

The notebooks use paths such as `../data/`, `../models/`, and
`../configs_yaml/`, which resolve correctly from that directory. Run the cells
in order. Runtime configuration must remain ahead of imports that initialize
AutoEIS/JuliaCall, TensorFlow, or JAX.

```{important}
Rendered outputs are previously saved examples. They demonstrate the workflow
and expected output shapes, but they are not regenerated or scientifically
revalidated during a documentation build. Results from stochastic training and
evaluation can differ when you rerun a notebook.
```

## 1. Data preparation

{doc}`data_preparation` covers loading a processed dataset, processing raw CSV
spectra, inspecting the data contract, and plotting spectra.

- **Inputs:** `../data/examples/training_dataset.pkl` and `EIS_raw_demo/`.
- **Generated files:** `demo_output/processed_training_data.pkl` when the save
  cell is run.
- **Substantial work:** raw processing performs interpolation, optional lin-KK
  preprocessing, threshold fitting, and plotting.

## 2. Agent training

{doc}`agent_training` constructs the environment and DDQN agent directly and
through `pipeline_builder`, then demonstrates training and model persistence.

- **Inputs:** `../data/examples/training_dataset.pkl` and
  `../configs_yaml/examples/demo_pipeline_config.yaml`.
- **Generated files:** models, checkpoints, summaries, metrics, and plots under
  `demo_output/`.
- **Substantial work:** both `agent.train()` cells run reinforcement-learning
  workloads; the configuration-based cell starts a second training run.

## 3. Agent evaluation

{doc}`agent_evaluation` loads the included trained model, generates ECMs,
examines action histories and fit metrics, and saves batch results.

- **Inputs:** `../data/examples/training_dataset.pkl` and
  `../models/examples/agent_trained.keras`.
- **Generated files:** fit plots, circuit diagrams, a circuit-evolution GIF,
  and CSV/pickle results under `demo_output/`.
- **Substantial work:** single-spectrum generation performs circuit fitting and
  visualization; batch evaluation repeats fitting over 20 spectra by default.

```{toctree}
:hidden:
:maxdepth: 1

data_preparation
agent_training
agent_evaluation
```
