# AutoREC: Reinforcement Learning for Automated ECM Generation

<p align="center">
  <img src="Logo.png" alt="AutoREC logo" width="200">
</p>

[![CI](https://github.com/AUTODIAL/AutoREC/actions/workflows/ci.yml/badge.svg)](https://github.com/AUTODIAL/AutoREC/actions/workflows/ci.yml)
[![Documentation](https://github.com/AUTODIAL/AutoREC/actions/workflows/docs.yml/badge.svg)](https://github.com/AUTODIAL/AutoREC/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AutoREC is an open-source Python package for developing reinforcement-learning
agents that automatically generate equivalent circuit models (ECMs) from
electrochemical impedance spectroscopy (EIS) data. It combines EIS data
preparation, a circuit-building environment, and a configurable Double Deep
Q-Network (DDQN) agent in Python and YAML-driven workflows.

Full documentation: **<https://autodiallab.ca/AutoREC/>**

## Features

- DDQN-based ECM generation with prioritized experience replay
- Gene-expression-programming circuit representation and mutation actions
- Raw EIS preprocessing, interpolation, feature construction, and validation
- Configurable training and evaluation through YAML, Python, or the CLI
- Cached circuit evaluations and dead-loop mitigation
- Jupyter tutorials, source-derived API reference, and Docker support

## Installation

AutoREC supports Python 3.10 through 3.12. The recommended source installation
uses the supplied Conda environment:

```bash
git clone https://github.com/AUTODIAL/AutoREC.git
cd AutoREC
bash setup_env.sh
conda activate autorec_env
autorec --help
```

The scientific stack includes TensorFlow, JAX, AutoEIS, and Julia-backed
components. The `autorec` CLI configures their runtime in the required order. If
you use the Python API directly, call `configure_autorec_runtime()` before other
scientific imports. See the
[installation guide](https://autodiallab.ca/AutoREC/getting_started/installation.html)
for manual setup, Jupyter configuration, and crash recovery.

## Quick start

Evaluate the included demonstration model on one row of the included dataset:

```bash
autorec evaluate \
  --config configs_yaml/examples/demo_pipeline_config.yaml \
  --model models/examples/agent_trained.keras \
  --indices 0 \
  --max-actions 24 \
  --use-training-env \
  --output-dir runs/quickstart \
  --run-id quickstart
```

This performs real circuit fitting and writes CSV and pickle results under
`runs/quickstart/`. Evaluation is stochastic; the demonstration is a functional
walkthrough rather than a performance benchmark.

Continue with:

- [Quick Start](https://autodiallab.ca/AutoREC/getting_started/quickstart.html)
- [Tutorials](https://autodiallab.ca/AutoREC/tutorials/index.html)
- [Data preparation](https://autodiallab.ca/AutoREC/user_guide/data.html)
- [Configuration](https://autodiallab.ca/AutoREC/user_guide/configuration.html)
- [Command-line interface](https://autodiallab.ca/AutoREC/user_guide/cli.html)
- [API reference](https://autodiallab.ca/AutoREC/api/index.html)

## Citation

If AutoREC contributes to published work, cite the current preprint:

> Ali Jaberi, Yonatan Kurniawan, et al. *AutoREC: A reinforcement learning
> platform for equivalent circuit model generation*. arXiv:2604.27266 (2026).

See the [paper](https://doi.org/10.48550/arXiv.2604.27266) and
[citation guide](https://autodiallab.ca/AutoREC/project/citation.html) for the
current metadata and BibTeX entry.

## Contributing

Bug reports, documentation improvements, tests, and focused code contributions
are welcome. Read the
[contribution guide](https://autodiallab.ca/AutoREC/development/contributing.html)
and use the repository's issue and pull-request templates.

## License

AutoREC is distributed under the [MIT License](LICENSE).
