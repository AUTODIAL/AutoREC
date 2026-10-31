# Contributing

Contributions to source, tests, examples, and documentation are welcome. Open an
[issue](https://github.com/AUTODIAL/AutoREC/issues) before a large or
behavior-changing proposal so maintainers can discuss scope and scientific
expectations.

## Development setup

AutoREC supports Python 3.10 through 3.12. From a clone, create and activate an
isolated environment, then install the development extra:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

`make setup-dev` runs both commands. For notebook work, install
`.[dev,notebook]` or run `make setup-notebook-dev`. Scientific dependencies are
substantial and AutoEIS can initialize Julia-backed components; follow the
{doc}`installation guide <../getting_started/installation>` for the supported
environment workflow.

## Make a focused change

- Branch from the current default branch and keep unrelated changes out of the
  pull request.
- Add or update tests for behavior changes. Avoid expensive scientific work in
  routine unit tests unless it is deliberately marked and justified.
- Update configuration, tutorials, narrative guides, and docstrings when a
  public workflow changes.
- Do not commit generated runs, caches, local environments, or credentials.
- Explain numerical assumptions and the limits of any reproducibility or
  performance claim.

Public classes and functions should have accurate NumPy-style docstrings. Keep
user-facing import paths and backward compatibility in mind; API changes need
explicit maintainer review.

## Validate locally

Run formatting and lint hooks:

```bash
make lint
```

Run tests in the environment where AutoREC is installed:

```bash
python -m pytest
```

A unit-test pass does not validate a scientific training result. When a change
affects fitting, preprocessing, rewards, actions, training, or reproducibility,
describe the additional controlled validation performed and its limits.

Build documentation changes with the isolated process in
{doc}`building the documentation <documentation>`. Notebook pages render saved
outputs without execution, so review code/output consistency separately.

## Repository layout

The main implementation lives under `src/autorec/`. Configuration-driven
builders are in `src/autorec/factory/`; tracked pipeline configurations are in
`configs_yaml/examples/`; notebooks and their small raw-data inputs are in
`tutorials/`; and additional scripts are under `example/`. The `data/` and
`models/` directories contain tracked walkthrough inputs, while `runs/` and
`results/` are intended for generated artifacts. Sphinx sources are under
`doc/`, separate from the scientific runtime.

## Pull request expectations

Use the repository pull-request template. Summarize the motivation, link the
issue, identify new dependencies, and report exact checks run. Include
configuration and reproducibility implications where relevant. GitHub Actions
runs pre-commit, unit tests, and the strict documentation build on pull
requests; contributors should still run the relevant checks before pushing.

By submitting a contribution, you agree that it is distributed under the
repository's [MIT license](https://github.com/AUTODIAL/AutoREC/blob/main/LICENSE).
