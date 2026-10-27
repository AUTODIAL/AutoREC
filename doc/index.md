# AutoREC

AutoREC is an open-source Python package for developing reinforcement learning
agents that generate equivalent circuit models (ECMs) from electrochemical
impedance spectroscopy (EIS) data.

The workflow combines data preparation, a circuit-building environment, and a
Double Deep Q-Network (DDQN) agent. Users can configure experiments with Python
or YAML, train agents, and evaluate the circuits they propose for EIS spectra.

## Getting started

Start with the installation and quick-start guides. The repository's notebooks
and example configurations provide detailed walkthroughs and reference inputs:

- [Installation](getting_started/installation.md)
- [Quick Start](getting_started/quickstart.md)
- [Jupyter tutorials](tutorials/index.md)
- [Core concepts](concepts/architecture.md)
- [Configuration guide](user_guide/configuration.md)
- [API reference](api/index.rst)

## Use and extend AutoREC

- Prepare inputs with the [data guide](user_guide/data.md).
- Run workflows through the [command-line interface](user_guide/cli.md).
- Interpret artifacts with [models and results](user_guide/models_and_results.md).
- Diagnose common setup and usage problems in
  [troubleshooting](user_guide/troubleshooting.md).
- Develop AutoREC with the [contribution guide](development/contributing.md).

## Project

- [Source code](https://github.com/AUTODIAL/AutoREC)
- [Report an issue](https://github.com/AUTODIAL/AutoREC/issues)
- [Citation and paper](project/citation.md)
- [MIT license](https://github.com/AUTODIAL/AutoREC/blob/main/LICENSE)

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting started

getting_started/installation
getting_started/quickstart
tutorials/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: User guide

concepts/architecture
user_guide/data
user_guide/configuration
user_guide/cli
user_guide/models_and_results
user_guide/docker
user_guide/advanced_examples
user_guide/troubleshooting
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference

api/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

development/contributing
development/documentation
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

project/citation
Source code <https://github.com/AUTODIAL/AutoREC>
Report an issue <https://github.com/AUTODIAL/AutoREC/issues>
License <https://github.com/AUTODIAL/AutoREC/blob/main/LICENSE>
```
