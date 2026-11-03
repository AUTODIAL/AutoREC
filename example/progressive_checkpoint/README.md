# Progressive-spectrum training with checkpoints

This example uses AutoREC's public YAML factory and CLI to combine cumulative
progressive-spectrum ECM construction with complete training checkpoints. It is
an executable workflow, not a performance benchmark: even the small settings
perform real circuit fitting and may take time.

Run all commands from the repository root after activating `autorec_env` and
installing AutoREC.

## Progressive training and automatic recovery

`progressive_config.yaml` divides each prepared spectrum into four cumulative
high-to-low-frequency windows. At every stage, the neural-network observation
is encoded to its fixed input size, physical fitting uses every original point
selected for that stage, and `CarryBestStatePolicy` supplies the next stage's
starting ECM.

```bash
python example/progressive_checkpoint/train.py \
  --config example/progressive_checkpoint/progressive_config.yaml \
  --checkpoint-dir example/progressive_checkpoint/outputs/checkpoints \
  --checkpoint-every 1 \
  --resume-latest
```

Rerun the same command after an interruption. `--resume-latest` restores the
latest compatible checkpoint and trains only to the cumulative `agent.num_trials`
target. To extend that target, update only `agent.num_trials` and rerun the
command. Progressive settings are part of the recovery fingerprint; changing
them requires a new checkpoint series.

The explicit immutable-checkpoint form is also available:

```bash
python example/progressive_checkpoint/train.py \
  --resume example/progressive_checkpoint/outputs/checkpoints/trial-00000002 \
  --additional-trials 1 \
  --checkpoint-dir example/progressive_checkpoint/outputs/checkpoints
```

## Progressive evaluation

Evaluation automatically uses the enabled progressive runtime from the YAML:

```bash
python example/progressive_checkpoint/evaluate.py \
  --config example/progressive_checkpoint/progressive_config.yaml \
  --model example/progressive_checkpoint/outputs/progressive/dqn_model.keras \
  --indices 0
```

The saved evaluation table includes the final-stage ECM fields plus
`progressive_num_windows`, `progressive_completed_all_stages`, and the detailed
`progressive_stage_results`.

## Recovering original full-spectrum behavior

The progressive feature is optional. `full_spectrum_config.yaml` intentionally
omits `progressive_spectrum`, so the same entry points execute the original
full-spectrum training and evaluation path:

```bash
python example/progressive_checkpoint/train.py \
  --config example/progressive_checkpoint/full_spectrum_config.yaml \
  --checkpoint-dir example/progressive_checkpoint/outputs/full-checkpoints

python example/progressive_checkpoint/evaluate.py \
  --config example/progressive_checkpoint/full_spectrum_config.yaml \
  --model example/progressive_checkpoint/outputs/full_spectrum/dqn_model.keras \
  --indices 0
```

An omitted section, `enabled: false`, or `num_windows: 1` without an explicit
`enabled: true` all leave progressive execution disabled. This is configuration
selection, not a separate legacy implementation.

## Configuration choices and outputs

- `num_windows` controls the number of cumulative frequency windows.
- `window_selector`, `observation_encoder`, `spectrum_source`, and
  `stage_transition_policy` select registered modular components.
- `PreparedSpectrumSource` uses the prepared `freq` and `Z_true` arrays.
  `RetainedSpectrumSource` uses retained pre-interpolation measurements when the
  dataset provides `source_freq` and `source_Z`.
- `InterpolatingSpectrumEncoder` preserves the existing progressive observation
  behavior. `RawSpectrumEncoder` is the non-interpolating alternative and
  accepts registered `normalizer` and `downsampler` parameter blocks.

Training writes the Keras model and histories under `agent.save_dir`, including
`progressive_stage_history.pkl` for enabled progressive runs. Complete immutable
checkpoints are written beneath `--checkpoint-dir`. Evaluation results are saved
under the wrapper's `--output-dir`.

Exact checkpoint continuation has stricter runtime and component requirements
than ordinary training. Built-in progressive components are checkpoint-safe;
arbitrary custom components are not automatically serializable. See
[`../../doc/user_guide/checkpointing.md`](../../doc/user_guide/checkpointing.md)
for the supported contract. These commands exercise software paths only and do
not establish scientific accuracy or comparative RL performance.
