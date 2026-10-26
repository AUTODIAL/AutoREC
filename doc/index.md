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
- [Example configurations](https://github.com/AUTODIAL/AutoREC/tree/main/configs_yaml/examples)

## Project

- [Source code](https://github.com/AUTODIAL/AutoREC)
- [Report an issue](https://github.com/AUTODIAL/AutoREC/issues)
- [AutoREC paper](https://arxiv.org/abs/2604.27266)
- [MIT license](https://github.com/AUTODIAL/AutoREC/blob/main/LICENSE)

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting started

getting_started/installation
getting_started/quickstart
tutorials/index
Example configurations <https://github.com/AUTODIAL/AutoREC/tree/main/configs_yaml/examples>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

development/documentation
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

Source code <https://github.com/AUTODIAL/AutoREC>
Report an issue <https://github.com/AUTODIAL/AutoREC/issues>
Paper <https://arxiv.org/abs/2604.27266>
License <https://github.com/AUTODIAL/AutoREC/blob/main/LICENSE>
```
