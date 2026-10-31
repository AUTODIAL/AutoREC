# Troubleshooting and FAQ

## Evaluation fails when switching environments

The demo configuration defines only one environment. Pass
`--use-training-env` on the CLI or `use_eval_env=False` to `eval_batch_eis()`.
For a genuinely held-out evaluation, define matching `training` and `eval`
subsections as shown in the {doc}`configuration guide <configuration>`.

## Evaluation reports that an integer is required

If `agent.action_cap` is `null`, supply `--max-actions` on the CLI or
`max_actions` through Python. The included demo configuration intentionally
leaves the value unset.

## A processed CSV fails array validation

Processed rows contain NumPy/complex arrays. Plain `pandas.read_csv()` returns
their serialized text rather than reconstructing those arrays. Use pickle for
lossless processed datasets, and only load trusted pickle files. Raw input CSVs
are a different format with scalar `freq`, `Z_real`, and `Z_imag` columns.

## Interpolation says extrapolation is not possible

Explicit frequency bounds must lie inside every spectrum after optional lin-KK
preprocessing. Choose the common overlap of all spectra, or leave a bound as
`null` to use each spectrum's own endpoint. If spectra have different lengths,
set `frequency_npoints` explicitly.

## A loaded model has incompatible shapes

Use the same ordered `elements`, `chromosome_HEAD_len`, EIS feature order, and
frequency-grid length used during training. These settings determine input and
output dimensions. See {doc}`models and results <models_and_results>`.

## Output files appear before training starts

Constructing `DDQN_ECM` creates `save_dir`, `save_dir/models`, and
`model_summary.txt`. Use a dedicated output directory for each experiment.

## The first run initializes or downloads Julia components

AutoEIS uses Julia-backed functionality. AutoREC's normal runtime setup performs
a warmup unless `--skip-autoeis-warmup` is passed. Skipping the explicit warmup
does not prevent a later circuit fit from initializing the dependency. Follow
the {doc}`installation guide <../getting_started/installation>` and use the
kernel guard when running notebooks.

## A run is slower than expected

Circuit fitting dominates many workflows. Start with one row and a small action
limit, retain the circuit cache, avoid GIF generation, and confirm the workflow
before scaling. Increasing `gradient_steps`, sample count, trial count, or the
action budget increases work. Performance depends on hardware, software
versions, circuit families, and cache behavior; the documentation does not
promise a fixed speedup.

## Does setting a seed guarantee identical results?

No. The CLI requests deterministic TensorFlow operations by default and seeds
the supported random generators, but external fitting, numerical libraries,
parallelism, hardware, and dependency versions can still affect results. A
`.keras` model is also not a complete resumable training checkpoint.

## Checkpoint loading reports a runtime or source mismatch

Exact resumption deliberately requires the same AutoREC source, Python and
numerical-library versions, CPU/backend selection, TensorFlow determinism, and
thread settings used when the checkpoint was created. Configure the process
before importing numerical libraries and do not edit AutoREC between sessions.
Use the same environment for every sequential run. A checkpoint created before
an AutoREC update is not migrated or loaded on a best-effort basis; restart the
experiment or return to the recorded compatible revision.

If a sequential example fails only after its first process, compare the process
environment and TensorFlow inter/intra-op settings rather than reseeding. The
checkpoint restores random state and must not be followed by `set_global_seed`.
See {doc}`checkpointing <checkpointing>` for the supported runtime setup.

## Checkpoint saving says the destination already exists

Checkpoint directories are immutable and AutoREC never overwrites an existing
destination. Resume from the existing valid checkpoint, select a new output
root, or remove an unwanted run only after confirming it is not needed. The
example runner records the exact previous checkpoint in each generated resume
configuration instead of relying on a mutable `latest` pointer.

## Where should I report a problem?

Search the [issue tracker](https://github.com/AUTODIAL/AutoREC/issues), then use
the relevant bug, installation, question, or documentation template. Include a
minimal reproducer, operating system, Python version, AutoREC revision, command
or configuration, and complete traceback. Remove private data and credentials.
