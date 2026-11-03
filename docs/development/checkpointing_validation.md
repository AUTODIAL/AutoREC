# Checkpoint implementation validation

Stages 7–10 were implemented incrementally on 2026-10-26. Each stage has a separate
commit. The user guide is [checkpointing.md](../checkpointing.md); the earlier investigation
report is unchanged. This record describes executed tests, not a claim of real-fitting or
GPU reproducibility.

Tests used `autorec_env/bin/python`, with this environment prefix:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
```

Append the Python commands below to that prefix. The long test additionally set
`AUTOREC_LONG_CHECKPOINT_TEST=1`.

| Stage/check | Command after environment prefix | Result |
|---|---|---|
| 7: RNG/environment/cache | `autorec_env/bin/python -m pytest -q -p no:cacheprovider tests/test_runtime_state.py tests/test_circuit_cache_state.py` | 78 passed |
| 8: storage/integrity | `autorec_env/bin/python -m pytest -q -p no:cacheprovider tests/test_checkpoint.py tests/test_runtime_state.py` | 15 passed; one ArviZ future warning |
| 9: fresh processes and reconstruction | `autorec_env/bin/python -m pytest -q -p no:cacheprovider tests/test_checkpoint.py tests/test_checkpoint_process.py` | Both process comparisons passed; one unit assertion initially failed because it compared pickle streams rather than state values |
| 9: corrected value comparison | `autorec_env/bin/python -m pytest -q -p no:cacheprovider tests/test_checkpoint.py tests/test_runtime_state.py` | 17 passed |
| 10: public API/CLI/three-way process comparisons | `autorec_env/bin/python -m pytest -q -p no:cacheprovider tests/test_checkpoint.py tests/test_cli.py tests/test_checkpoint_process.py` | 28 passed, one opt-in long test skipped |
| 10: long acceptance test | `autorec_env/bin/python -m pytest -q -p no:cacheprovider tests/test_checkpoint_process.py -k 1000` | 1 passed, 2 deselected; 177.51 seconds |
| 10: final regressions, process tests run separately above | `autorec_env/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_checkpoint_process.py` | 365 passed; 10 existing Keras/NumPy deprecation warnings; 77.97 seconds |

Ruff checks/formatting and `git diff --check` passed for the changed files. Both
`autorec_env/bin/python -m autorec.cli train --help` and
`autorec_env/bin/autorec train --help` succeeded. Commit hooks
also run whitespace, file-size and Ruff checks.

The process comparisons use real replay insertion/sampling, neural gradients, scheduled
Adam updates, target updates, cache logic and cumulative histories. The regular comparison
covers both LRU and Clock using 12 trials versus sessions of 3, 4 and 5. The long comparison
uses Clock and 1,000 trials versus ten separate processes performing 100 trials each.
Both compare continuous training without periodic saving, continuous training with periodic
saving, and repeated restoration. Final numerical state and transition/sampling traces
are compared exactly, including dtype, shape and raw bytes for numerical arrays. Sparse
optimizer updates keep the 1,000-trial test practical.

Circuit parsing/encoding/fitting and model/plot output are controlled substitutes in the
process comparisons. The fitter deliberately consumes NumPy randomness so cache behavior
affects the trajectory. Exact scientific fitting on the real AutoEIS/Julia/SciPy stack,
GPU execution and different hardware remain unqualified. No approximate comparisons are
used to claim checkpoint equivalence.
