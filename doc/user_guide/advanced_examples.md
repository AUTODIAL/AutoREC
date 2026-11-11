# Advanced examples

The repository contains small Python entry points in `example/` in addition to
the maintained {doc}`notebook tutorials <../tutorials/index>`. Read each script
and configuration before running it: examples perform real fitting or training
and may write substantial output.

## Direct and factory-based pipelines

- [`example/main.py`](https://github.com/AUTODIAL/AutoREC/blob/main/example/main.py)
  constructs components directly.
- [`example/main_factory.py`](https://github.com/AUTODIAL/AutoREC/blob/main/example/main_factory.py)
  builds the configured pipeline through the factory.

These are development examples rather than stable command-line interfaces. For
new automation, prefer the documented {doc}`CLI <cli>` or the public
{doc}`factory API <../api/factory>`.

## Hyperparameter tuning on Slurm

[`example/hyperparameter_tuning/`](https://github.com/AUTODIAL/AutoREC/tree/main/example/hyperparameter_tuning)
contains an Optuna workflow that generates configurations, submits Slurm jobs,
waits for batches, and scores their outputs. It has extra requirements in its
own `requirements.txt` and cluster-specific job-script assumptions in
`tuning_utils.py`.

Treat this workflow as a template requiring review for your scheduler,
partitions, environment modules, filesystem, resource limits, and accounting.
Run its smallest dry/controlled case before scheduling a sweep. Hyperparameter
search can consume substantial compute and does not by itself validate
generalization; preserve held-out evaluation data and report the search space
and selection procedure.

The example's own
[`readme.md`](https://github.com/AUTODIAL/AutoREC/blob/main/example/hyperparameter_tuning/readme.md)
documents its arguments and output layout.

## Example configurations and artifacts

Tracked configurations under
[`configs_yaml/examples/`](https://github.com/AUTODIAL/AutoREC/tree/main/configs_yaml/examples)
are starting points. The repository also includes one small processed dataset
and one trained model for functional walkthroughs. They are not benchmark
datasets or universal pretrained models.

Keep generated runs outside tracked example directories, record the exact
configuration used, and review {doc}`models and results <models_and_results>`
before comparing experiments.
