# %%
"""
Direct instantiation script for AutoREC.
This script demonstrates training a DDQN agent without using configuration files.
For production runs, consider using main_factory.py with YAML configs instead.

Workflow:
---------
1. Load and prepare EIS dataset
2. Create environment with circuit evaluation and caching
3. Create DDQN agent with neural network and replay buffer
4. Train agent for specified number of trials
5. Save trained model
6. Evaluate on random sample of EIS measurements
7. Save evaluation results
"""

import datetime

from autorec.runtime import configure_autorec_runtime

# Configure JuliaCall/AutoEIS, Matplotlib, TensorFlow, and BLAS/OpenMP before
# importing scientific Python libraries. This avoids terminal-only native
# crashes that can happen when TensorFlow/JAX load before Julia.
configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)

import numpy as np
from pathlib import Path

from autorec.utils import save_evaluation_results, set_global_seed

# Set the global random seed before constructing environments or agents.
# This ensures reproducibility across numpy, tensorflow, and python random.
set_global_seed(42, True)

from autorec.data_preparation import EISDataPrep
from autorec.environment import EIS_ECM_Env
from autorec.agent import DDQN_ECM


# THREADING CONFIGURATION
# ============================================================================
# Runtime/threading configuration is applied at the top of the file before
# importing numpy, TensorFlow, JAX, Matplotlib, or AutoEIS.

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# Choose between quick test run or full training
RUN_MODE = "quick"
# RUN_MODE = "full"

# Configure based on run mode
if RUN_MODE == "quick":
    NUM_TRIALS = 50           # Short training for testing
    EVAL_SAMPLE_SIZE = 20      # Evaluate on 20 random EIS
elif RUN_MODE == "full":
    NUM_TRIALS = 8000          # Full training run
    EVAL_SAMPLE_SIZE = 100     # More comprehensive evaluation
else:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}. Use 'quick' or 'full'")

# Directory structure for outputs
SAVE_DIR = Path("example_outputs")   # Base directory for this run
MODEL_SAVE_DIR = SAVE_DIR / Path("models")        # Trained neural networks
RESULTS_SAVE_DIR = SAVE_DIR / Path("results")    # Evaluation DataFrames


# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
# Load and prepare the EIS dataset that contains impedance measurements that the agent will learn to fit

data_prepper = EISDataPrep(
    path="../data/training_dataset.pkl",  # Pre-processed pickle file
    evaluation=True,             # Ground truth circuits available for validation
)

# Load the dataset into memory
dataset = data_prepper.load()

# Display dataset statistics
validation_summary = data_prepper.get_summary()

print(validation_summary)

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================
# Create the RL-EIS environment for AutoREC

env = EIS_ECM_Env(
    dataset=dataset,           # The EIS measurements to fit
    seed=42,                    # Random seed for reproducibility
    chromosome_HEAD_len=10,     # Length of GEP head (affects circuit complexity)
    cache_enabled=True          # Enable LRU cache (highly recommended)
)

# Calculate action space size
# action_cap prevents episodes from running indefinitely. You can set it to a fixed number or calculate based on environment parameters.
action_cap = env.chromosome_HEAD_len + env.chromosome_TAIL_len + 3

# ============================================================================
# 3. AGENT SETUP
# ============================================================================
# Create the DDQN agent
agent = DDQN_ECM(
    env,                        # The environment to interact with
    action_cap=action_cap,      # Maximum actions per episode
    num_trials=NUM_TRIALS,      # Total training episodes
    save_dir=SAVE_DIR,          # Where to save checkpoints and models
    save_frequency=100          # Save checkpoint every 100 trials
)

# ============================================================================
# 4. TRAINING OR LOADING MODEL
# ============================================================================
# Check if a trained model already exists
# If yes: load it and skip training (useful for evaluation only)
# If no: train a new model from scratch

final_model_file = SAVE_DIR / "dqn_model.keras"
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if final_model_file.exists():
    # Model found - load it for evaluation
    print(f"Loading existing model from {final_model_file}...")
    agent.load_model(final_model_file)
else:
    # No model found - train a new one
    print(f"No existing model found. Training new model for {NUM_TRIALS} trials...")
    
    # Train the agent
    training_results = agent.train()

    # Save the trained model
    model_filename = f"ddqn_model_{run_id}_trials{NUM_TRIALS}.keras"
    model_path = MODEL_SAVE_DIR / model_filename

    try:
        agent.save_model(model_path)
        print(f"✓ Model saved successfully")
    except Exception as e:
        print(f"⚠ Error saving model: {e}")

# ============================================================================
# 5. EVALUATION
# ============================================================================
# Test the trained agent on a random sample of EIS measurements
# This gives us metrics on how well the agent discovers circuits

try:
    # Select random sample of EIS to evaluate
    # We don't evaluate all EIS because it takes a long time
    eval_indices = np.random.choice(
        len(dataset), 
        size=min(EVAL_SAMPLE_SIZE, len(dataset)), 
        replace=False
    ).tolist()

    print(f"\nEvaluating {len(eval_indices)} random EIS samples...")

    # Run evaluation batch
    eval_sample_results = agent.eval_batch_eis(
        eis_indices=eval_indices, 
        max_actions=action_cap,    # Maximum mutations per EIS
        verbose=False              # Don't print details for each EIS
    )

    # Save results for analysis
    save_evaluation_results(eval_sample_results, f"{run_id}_sample", RESULTS_SAVE_DIR)

except Exception as e:
    print(f"\n⚠ Error during sample evaluation: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# OPTIONAL: FULL DATASET EVALUATION
# ============================================================================
# Uncomment to evaluate on the entire dataset (slow but comprehensive)
# Useful for final model assessment

# eval_all_results = agent.eval_all_eis(
#     max_actions=action_cap,
#     verbose=False,
#     use_eval_env=False,
# )

# %%
