"""Run one simplified-circuit reward hyperparameter-tuning fidelity stage."""

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CASE_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from example.reward_exploration.tuning.stage import main


def prepare_experimental_runtime():
    """Import the custom environment after native runtimes are configured."""
    import environment  # noqa: F401


def build_experimental_pipeline(config):
    """Build the standard pipeline with this case's reward environment."""
    import autorec.factory.environment as environment_factory
    from autorec.factory import pipeline_builder

    from environment import EIS_ECM_Env

    standard_environment_class = environment_factory.EIS_ECM_Env
    environment_factory.EIS_ECM_Env = EIS_ECM_Env
    try:
        return pipeline_builder(config)
    finally:
        environment_factory.EIS_ECM_Env = standard_environment_class


if __name__ == "__main__":
    main(
        pipeline_builder_function=build_experimental_pipeline,
        prepare_runtime_function=prepare_experimental_runtime,
        default_config=CASE_DIR / "tuning_base_config.yaml",
    )
