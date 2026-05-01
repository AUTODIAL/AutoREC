# AutoREC: Reinforcement Learning for Automated ECM Generation from EIS Data

<p align="center">
  <img src="Logo.png" alt="AutoREC logo" width="200">
</p>

AutoREC is an open-source Python package for developing reinforcement learning
(RL) agents that automatically generate equivalent circuit models (ECMs) from
electrochemical impedance spectroscopy (EIS) data.

It formulates ECM construction as a Markov Decision Process and trains a Double
Deep Q-Network (DDQN) agent to mutate circuit representations, fit candidate
models, and learn which actions produce high-quality ECMs. The package includes
data preparation, training, evaluation, caching, prioritized replay, dead-loop
mitigation, and YAML-based configuration utilities.

In the AutoREC paper, the trained agent achieved over 99.6% success on synthetic
datasets and generalized to unseen experimental EIS data from several
electrochemical systems.


>  AutoREC is now published in arxiv. You can find the paper [here](https://doi.org/10.48550/arXiv.2604.27266). If you find AutoREC useful, please consider citing it in your work.
> 
> A. Jaberi, Y. Kurniawan, et al., AutoREC: A software platform for developing reinforcement learning agents for equivalent circuit model generation from electrochemical impedance spectroscopy data

## Key Features

- DDQN-based ECM generation for EIS spectra
- Gene expression programming (GEP) circuit representation
- Prioritized experience replay for efficient learning
- Dead-loop mitigation strategy to efficiently explore a complex action space
  for circuit generation
- Cached circuit evaluations to reduce repeated fitting cost
- Data preparation tools for processed datasets and raw CSV files
- YAML-based configuration for reproducible training and evaluation
- Jupyter tutorials for data preparation, training, and evaluation

## Installation

### Prerequisites

- Conda, through Miniconda or Anaconda
- Python 3.10

AutoREC is packaged with a `pyproject.toml` file. The repository also provides
`environment.yml`, `requirements.txt`, and `setup_env.sh` as convenience entry
points that install from the package metadata.

### Option 1: Setup Script Using `environment.yml`

```bash
bash setup_env.sh
conda activate autorec_env
```

### Option 2: Manual Editable Install

```bash
conda create -n autorec_env python=3.10 -y
conda activate autorec_env
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

In VS Code or Jupyter, select the kernel named `Python (autorec_env)`.

### Installation Check

```bash
python -m pip check
python -c "import autorec, autoeis, lcapy, tensorflow as tf; print('AutoREC environment ready')"
```

Note: the first time you import the agent, Julia will be automatically set up
and configured if it is not already available in the current environment.

### Fixing Segmentation Faults or Kernel Crashes

AutoREC uses AutoEIS/JuliaCall together with TensorFlow/JAX. In some terminal
or notebook sessions, Python can crash if TensorFlow, JAX, or other native
scientific libraries initialize before Julia.

For standalone scripts, add this at the very beginning of the file, before
importing NumPy, TensorFlow, JAX, Matplotlib, AutoEIS, or other AutoREC modules:

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1, warmup_autoeis=True)
```

For notebooks, run the same lines in the first cell before other imports. If
you do not want to add this setup cell to every notebook, install the AutoREC
kernel guard once in the active environment:

```bash
conda activate autorec_env
python scripts/install_autorec_kernel_guard.py
```

Then restart the notebook kernel.

The guard is installed only into the active Python environment's
`site-packages/sitecustomize.py`; it does not modify AutoREC source code or the
notebooks.

## Quick Start

Complete step-by-step tutorials are provided in the `tutorials` folder. This
section gives a quick overview of the main workflow.

### 1. Prepare Data

Load an existing processed dataset:

```python
from autorec.data_preparation import EISDataPrep

data_prep = EISDataPrep(path="PATH/training_dataset.pkl", mode='load')
dataset = data_prep.load()
```

Or process raw CSV files:

```python
from autorec.data_preparation import EISDataPrep

# CSV files should contain freq, Z_real, and Z_imag columns.
data_prep = EISDataPrep(
    path="PATH/EIS_raw",
    mode="process")

dataset = data_prep.load()
data_prep.save("processed_training_data.pkl")
```

### 2. Train an Agent

Using the YAML configuration:

```python
from autorec.factory import environment_and_agent_builder

env, eval_env, agent = environment_and_agent_builder(
    "PATH/demo_environment_agent_config.yaml"
)

agent.train()
agent.save_model("PATH/final_model.keras")
```

Using classes directly:

```python
from autorec.agent import DDQN_ECM
from autorec.environment import EIS_ECM_Env
from autorec.utils import set_global_seed

set_global_seed(42, deterministic_ops=True)

env = EIS_ECM_Env(dataset=dataset, seed=42, cache_enabled=True)

agent = DDQN_ECM(
    training_env=env,
    num_trials=1000,
    save_dir="demo_output",
    random_seed=42,
)

agent.train()
agent.save_model("PATH/final_model.keras")
```

### 3. Evaluate an Agent

```python
eval_results = agent.eval_batch_eis(
    eis_indices=eval_indices,
    max_actions=action_cap,
    verbose=False  # Set to True to see progress for each EIS
)

mean_reward = eval_results['best_reward'].mean()
success_rate = eval_results['found_solution'].astype(int).sum() / len(eval_results) * 100

print(f"Success rate: {success_rate:.2%}")
print(f"Mean reward: {mean_reward:.4f}")
```

## Tutorials

The `tutorials` folder contains notebook walkthroughs for:

- Preparing EIS data
- Training an AutoREC agent
- Evaluating generated ECMs

Start there if you are using the package for the first time.

## Configuration

YAML Configuration examples are available in `default_configs/`. The package also
ships default configuration files under `src/autorec/default_configs/`.

Environment options:

```yaml
dataset_path: "data/training_dataset.pkl"
seed: 42
chromosome_HEAD_len: 10
cache_enabled: true
cache_capacity: 20000
cache_type: lru
```

Agent options:

```yaml
num_trials: 1000
save_dir: "results/demo"
save_frequency: 500
gamma: 0.99
learning_rate: 0.0005
batch_size: 150
buffer_capacity: 15000
initial_epsilon: 1.0
epsilon_min: 0.1
epsilon_decay: 0.99968
prioritized_replay_alpha: 0.6
initial_beta: 0.4
final_beta: 0.7
```

Training recommendations:

- Start with the provided default configuration.
- Increase `num_trials` for diverse datasets or more complex ECM families.
- Reduce `num_trials` for small demonstrations or narrowly defined circuit sets.
- Keep caching enabled unless memory use becomes a problem.

## How AutoREC Works

AutoREC represents ECM topologies as GEP chromosome strings. For example,
`+/RPRRRR` represents an R in series with (P parallel to R).

Circuit symbols:

- `+`: series connection
- `/`: parallel connection
- `R`: resistor
- `L`: inductor
- `C`: Capacitor
- `P`: constant phase element

Training loop:

1. The environment selects an EIS spectrum and initializes a circuit state.
2. The DDQN agent chooses an action that mutates one chromosome position.
3. The environment converts the chromosome to a circuit and fits it to the EIS data.
4. Rewards are computed from fit quality metrics such as R2 and chi-square as well as circuit complexity.
5. The agent stores the transition in prioritized replay and updates its policy.
6. The process continues until a valid solution is found or the action budget is reached.

The trained agent can then propose ECM topologies for new EIS spectra and report
the best circuit, fitted parameters, metrics, and action history.

## Project Structure

```text
AutoREC/
|-- pyproject.toml                  # Package metadata and dependencies
|-- environment.yml                 # Conda environment that installs this package
|-- requirements.txt                # Pip entry point that installs from pyproject.toml
|-- setup_env.sh                    # Convenience conda setup script
|-- README.md
|-- data/                           # Example processed datasets
|-- default_configs/                # Example YAML configurations
|-- example/                        # Example Python entry points
|-- scripts/
|   `-- install_autorec_kernel_guard.py
|-- src/
|   `-- autorec/
|       |-- agent.py                # DDQN agent and evaluation helpers
|       |-- data_preparation.py     # Data loading, processing, and validation
|       |-- environment.py          # RL environment for ECM generation
|       |-- factory.py              # Builders for config-driven workflows
|       |-- utils.py                # Shared utilities
|       |-- runtime.py              # helper for run time fix 
|       |-- default_configs/        # Package-level default configs
|       `-- optimized_data_structures/
|-- trained_agent/                  # Example trained model
`-- tutorials/                      # Step-by-step notebook tutorials
```


<p align="center">
  <img src="Design.png" alt="AutoREC design overview" width="550">
</p>

## Development Notes

The codebase includes support for saving models and training outputs. More
efficient training and circuit representation are under development for public use.

## Citation

If you use AutoREC, please cite the accompanying AutoREC paper. You can find the paper [here](https://doi.org/10.48550/arXiv.2604.27266).

> AutoREC: A software platform for developing reinforcement learning agents for equivalent circuit model generation from electrochemical impedance spectroscopy data. 

## License

This project is distributed under the MIT license.
