# Installation

These instructions install AutoREC from a source checkout. The package metadata
supports Python 3.10 through 3.12; the provided Conda environment uses Python
3.10, which is the recommended version for the pinned scientific stack.

The installation includes TensorFlow, JAX, AutoEIS, JuliaCall, and circuit-fitting
dependencies. Use a dedicated environment rather than adding AutoREC to an
unrelated scientific environment.

## Recommended Conda setup

Install Miniconda or another Conda distribution, then clone AutoREC and run its
setup script:

```bash
git clone https://github.com/AUTODIAL/AutoREC.git
cd AutoREC
bash setup_env.sh
conda activate autorec_env
```

The script creates `autorec_env` from `environment.yml` and installs AutoREC in
editable mode with its notebook dependencies. If an environment with that name
already exists, the script asks before replacing it.

## Manual editable install

Use this option on Windows or when you want to create the environment yourself:

```bash
conda create -n autorec_env python=3.10 -y
conda activate autorec_env
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

Run these commands from the repository root. The runtime requirements are
defined in `pyproject.toml`; `requirements.txt` delegates to the same package
metadata so there is one dependency definition to maintain.

## Verify the installation

Check the resolved dependencies and the lightweight command-line entry point:

```bash
python -m pip check
autorec --help
```

This verifies package installation without starting Julia, TensorFlow, or JAX.
The first scientific AutoREC run may take longer while JuliaCall and AutoEIS
prepare their runtime environment.

## Initialize the scientific runtime safely

AutoREC combines AutoEIS/JuliaCall with TensorFlow and JAX. In a standalone
Python script, configure the runtime before importing NumPy, TensorFlow, JAX,
Matplotlib, AutoEIS, or AutoREC's scientific modules:

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1, warmup_autoeis=True)

from autorec.factory import pipeline_builder
```

The helper creates user-writable Julia and Matplotlib cache directories, limits
native thread pools, and warms up AutoEIS before TensorFlow/JAX initialization.
The `autorec` command performs this setup automatically for preprocessing,
training, evaluation, and inference.

```{warning}
Import order matters. If a process crashes at the native-library level, restart
it and put `configure_autorec_runtime(...)` before all scientific imports.
Changing the order later in the same Python process is insufficient because the
libraries have already initialized.
```

## Jupyter and VS Code

Select the Python interpreter from `autorec_env`. In a notebook, make runtime
configuration the first cell:

```python
from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1, warmup_autoeis=True)
```

Alternatively, install the repository's environment-local startup guard:

```bash
conda activate autorec_env
python scripts/install_autorec_kernel_guard.py
```

Restart the kernel afterward. The installer writes `sitecustomize.py` only in
the active environment's `site-packages`; if an unrelated file already exists,
it creates an `.autorec-backup` before installing the guard.

## Next step

Continue with the {doc}`quickstart` to evaluate the included model, preprocess
data, and start a small configuration-driven training run.
