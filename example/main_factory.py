# %%
"""
Script for using AutoREC with a YAML configuration file.
This script follows the same workflow as main.py, but reads the environment and
agent configuration from a YAML file instead of specifying it directly in code.

Workflow:
---------
1. Read YAML configuration
2. Build environment and agent from configuration
3. Train agent for the configured number of trials
4. Save trained model
5. Evaluate on random sample of EIS measurements
6. Save evaluation results
"""

import datetime
from pathlib import Path

from autorec.runtime import configure_autorec_runtime

# Configure JuliaCall/AutoEIS, Matplotlib, TensorFlow, and BLAS/OpenMP before
# importing scientific Python libraries. This avoids terminal-only native
# crashes that can happen when TensorFlow/JAX load before Julia.
configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

import numpy as np

from autorec.utils import save_evaluation_results, set_global_seed

# Set the global random seed before constructing environments or agents.
# This ensures reproducibility across numpy, tensorflow, and python random.
set_global_seed(42, True)

from autorec.factory import _yaml_reader, environment_and_agent_builder


# THREADING CONFIGURATION
# ============================================================================
# Runtime/threading configuration is applied at the top of the file before
# importing numpy, TensorFlow, JAX, Matplotlib, or AutoEIS.

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# All environment and agent hyperparameters come from this YAML file.
CONFIG_FILE = "../default_configs/demo_environment_agent_config.yaml"
config = _yaml_reader(CONFIG_FILE)

# Evaluation settings for this example script.
# Training length is controlled by the YAML file's agent.num_trials value.
RUN_MODE = "quick"
# RUN_MODE = "full"

if RUN_MODE == "quick":
    EVAL_SAMPLE_SIZE = 20      # Evaluate on 20 random EIS
elif RUN_MODE == "full":
    EVAL_SAMPLE_SIZE = 100     # More comprehensive evaluation
else:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}. Use 'quick' or 'full'")

# Directory structure for outputs is read from the YAML agent configuration.
SAVE_DIR = Path(config["agent"]["save_dir"])
MODEL_SAVE_DIR = SAVE_DIR / Path("models")       # Trained neural networks
RESULTS_SAVE_DIR = SAVE_DIR / Path("results")    # Evaluation DataFrames

# ============================================================================
# 2. ENVIRONMENT AND AGENT SETUP
# ============================================================================
# Build the RL-EIS environment and DDQN agent from YAML configuration.
env, eval_env, agent = environment_and_agent_builder(CONFIG_FILE)
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
# Test the trained agent on a random sample of EIS measurements.
# This gives us metrics on how well the agent discovers circuits.

try:
    # Select random sample of EIS row positions to evaluate.
    # We don't evaluate all EIS because it takes a long time.
    eval_indices = np.random.choice(
        len(dataset),
        size=min(EVAL_SAMPLE_SIZE, len(dataset)),
        replace=False,
    ).tolist()

    print(f"\nEvaluating {len(eval_indices)} random EIS samples...")

    # Run evaluation batch on the training environment used by this example.
    eval_sample_results = agent.eval_batch_eis(
        eis_indices=eval_indices,
        max_actions=action_cap,    # Maximum mutations per EIS
        verbose=False,             # Don't print details for each EIS
    )

    # Save results for analysis.
    save_evaluation_results(eval_sample_results, f"{run_id}_sample", RESULTS_SAVE_DIR)

except Exception as e:
    print(f"\n⚠ Error during sample evaluation: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# OPTIONAL: FULL DATASET EVALUATION
# ============================================================================
# This is slow but useful for final model assessment.


# eval_all_results = agent.eval_all_eis(
#     max_actions=action_cap,
#     verbose=False,
#     use_eval_env=False,
# )


# save_evaluation_results(eval_all_results, f"{run_id}_all", RESULTS_SAVE_DIR)
# %%
