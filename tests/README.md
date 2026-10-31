# Running the tests

From the repository root, use the repository environment and deterministic CPU settings:

```bash
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 MPLBACKEND=Agg
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

For routine edits, run the unit tests:

```bash
autorec_env/bin/python -m pytest -q tests -m "not integration"
```

Run the complete regular suite before merging:

```bash
autorec_env/bin/python -m pytest -q tests
```

Plain `pytest` and the existing CI command still include integration tests. The marker
only provides an explicit way to select a faster subset; nothing is deselected by default.
To run just the complete-training and fresh-process checks:

```bash
autorec_env/bin/python -m pytest -q tests -m integration
```

The 1,000-trial acceptance test remains opt-in:

```bash
AUTOREC_LONG_CHECKPOINT_TEST=1 autorec_env/bin/python -m pytest -q \
  tests/test_checkpoint_process.py -k 1000
```

Use `--durations=20` with any command to see where time is spent. The fresh-process tests
start several independent Python interpreters, each importing the scientific stack. Their
runtime is much larger than most validation tests, despite having few pytest cases.

## Coverage organization

- Component tests cover replay ownership/order/metadata, both cache eviction policies,
  RNG state, environment identity, neural variables and optimizer state, and rejection of
  malformed or unsupported checkpoints.
- Shared validators use representative field/type combinations rather than every Cartesian
  product. Equivalent malformed-input cases run as a table inside one test instead of
  appearing as dozens of separate collected items. Policy-specific checks remain separate.
  Both optimizer classes retain unbuilt, built and trained coverage; every supported
  learning-rate schedule and AMSGrad still has a trained-state round trip and exact
  subsequent-update comparison.
- Same-process integration tests retain repeated single-trial sessions, continuation across
  warm-up, default/custom schedules, zero-length sessions, and fixed expectations for
  replay insertion, learning and target-update timing.
- The routine fresh-process integration test compares continuous training with restoration
  across one process boundary using the more stateful Clock policy. The split sessions perform
  periodic saves, and every worker separately checks that its final save is state-inert. The
  test uses exact final states and transition/sampling traces. Component and same-process
  integration tests retain separate coverage of both LRU and Clock behavior. The opt-in
  1,000-trial test additionally retains a continuous periodic-saving control.

The suite deliberately no longer runs every optimizer configuration in every slot phase,
every invalid value against every metadata field, or every cache policy against the same
common validation matrix. This reduces duplication while preserving focused checks for
the distinct behaviors. It is not exhaustive combination coverage.

Integration tests use a controlled circuit fitter and real RL/neural updates. They do not
qualify real scientific fitting or GPU reproducibility; see
[the checkpoint guide](../docs/checkpointing.md) for the execution contract.
