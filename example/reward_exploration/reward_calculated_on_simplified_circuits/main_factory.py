# %%
"""
Train AutoREC while calculating complexity penalties on simplified circuits.

This script follows ``example/main_factory.py``, but injects the experimental
environment from this directory into the standard configuration-driven pipeline.

Workflow:
---------
1. Read YAML configuration (either from CLI or hardcoded path)
2. Build environment and agent from configuration
3. Train agent for the configured number of trials
4. Save trained model
5. Evaluate on either a quick sample or every EIS measurement
6. Save evaluation results
"""

import datetime
from pathlib import Path
import argparse
from pprint import pprint

from autorec.runtime import configure_autorec_runtime

# Configure JuliaCall/AutoEIS, Matplotlib, TensorFlow, and BLAS/OpenMP before
# importing scientific Python libraries. This avoids terminal-only native
# crashes that can happen when TensorFlow/JAX load before Julia.
configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

from autorec.utils import save_evaluation_results, set_global_seed

# Set the global random seed before constructing environments or agents.
# This ensures reproducibility across numpy, tensorflow, and python random.
set_global_seed(42, True)

import autorec.factory.environment as environment_factory
from autorec.factory import AUTOREC_ROOT, config_reader, pipeline_builder

from environment import EIS_ECM_Env
from evaluation import DEFAULT_RUN_MODE, run_evaluation


def build_experimental_pipeline(config):
    """Build the standard pipeline with this experiment's environment class."""
    standard_environment_class = environment_factory.EIS_ECM_Env
    environment_factory.EIS_ECM_Env = EIS_ECM_Env
    try:
        return pipeline_builder(config)
    finally:
        environment_factory.EIS_ECM_Env = standard_environment_class


# THREADING CONFIGURATION
# ============================================================================
# Runtime/threading configuration is applied at the top of the file before
# importing numpy, TensorFlow, JAX, Matplotlib, or AutoEIS.

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# All environment and agent hyperparameters come from this YAML file.
parser = argparse.ArgumentParser(
    description="Train AutoREC with complexity penalties based on simplified circuits"
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default=str(AUTOREC_ROOT / "configs_yaml" / "examples" / "demo_pipeline_config.yaml"),
    help="Path to the YAML configuration file for the environment and agent.",
)
parser.add_argument(
    "--run-mode",
    "-r",
    type=str,
    default=DEFAULT_RUN_MODE,
    choices=["quick", "full"],
    help="Evaluation mode: 'quick' for 20 random spectra, 'full' for all spectra.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "demo_output",
    help="Directory for this experiment's models, plots, checkpoints, and results.",
)
args = parser.parse_args()

CONFIG_FILE = Path(args.config)
configs = config_reader(CONFIG_FILE)
configs["agent"]["save_dir"] = args.output_dir
print(f"Loaded configuration from {CONFIG_FILE}:")
pprint(configs, sort_dicts=False)

# Training length is controlled by the YAML file's agent.num_trials value.
RUN_MODE = args.run_mode

# ============================================================================
# 2. ENVIRONMENT AND AGENT SETUP
# ============================================================================
# Build the RL-EIS environment and DDQN agent from YAML configuration. Only the
# environment class differs from the standard AutoREC pipeline.
dataprep, _, env, eval_env, agent = build_experimental_pipeline(configs)
dataset = env.dataset

# action_cap and num_trials are either read from YAML so we can access them here.
action_cap = agent.action_cap
num_trials = agent.num_trials

print("The action cap and number of trials are:", action_cap, num_trials)

print(f"Loaded training dataset with {len(dataset)} EIS measurements")
if eval_env is not None:
    print(f"Loaded evaluation dataset with {len(eval_env.dataset)} EIS measurements")

# ============================================================================
# 3. TRAINING OR LOADING MODEL
# ============================================================================
# Check if a trained model already exists.
# If yes: load it and skip training (useful for evaluation only).
# If no: train a new model from scratch.

# Directory structure for outputs.
SAVE_DIR = agent.save_dir  # Base directory for this agent's outputs
MODEL_SAVE_DIR = SAVE_DIR / Path("models")  # Trained neural networks
RESULTS_SAVE_DIR = SAVE_DIR / Path("results")  # Evaluation DataFrames

final_model_file = SAVE_DIR / "dqn_model.keras"
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if final_model_file.exists():
    # Model found - load it for evaluation.
    print(f"Loading existing model from {final_model_file}...")
    agent.load_model(final_model_file)
else:
    # No model found - train a new one.
    print(f"No existing model found. Training new model for {num_trials} trials...")

    # Train the agent.
    training_results = agent.train()

    # Save the trained model.
    model_filename = f"ddqn_model_{run_id}_trials{num_trials}.keras"
    model_path = MODEL_SAVE_DIR / model_filename

    try:
        agent.save_model(model_path)
        print("✓ Model saved successfully")
    except Exception as e:
        print(f"⚠ Error saving model: {e}")

# ============================================================================
# 4. EVALUATION
# ============================================================================
# Test the trained agent on either a random sample or the complete training dataset.

try:
    evaluation_results, evaluation_suffix = run_evaluation(
        agent=agent,
        run_mode=RUN_MODE,
        action_cap=action_cap,
        dataset_size=len(dataset),
    )

    # Save results for analysis.
    save_evaluation_results(
        evaluation_results,
        f"{run_id}_{evaluation_suffix}",
        RESULTS_SAVE_DIR,
    )

except Exception as e:
    print(f"\n⚠ Error during evaluation: {e}")
    import traceback

    traceback.print_exc()
    raise
# %%
