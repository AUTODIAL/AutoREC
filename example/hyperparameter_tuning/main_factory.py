"""Training using the factory to build the environment and agent."""

import pandas as pd
import argparse
import traceback

from autorec.runtime import configure_autorec_runtime

configure_autorec_runtime(thread_count=1, warmup_autoeis=True, suppress_tf_logs=True)
from autorec.utils import set_global_seed

set_global_seed(42, True)
from autorec.factory import environment_and_agent_builder


print("#" * 55)
print("TRAINING RL AGENT FOR GENERATING ECM FROM EIS DATA")
print("#" * 55, "\n")

# Read configuration from the command line argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="./base_config.yaml",
    help="Path to the YAML configuration file.",
)
args = parser.parse_args()
CONFIG_FILE = args.config
print("Configuration file:", CONFIG_FILE, "\n")

# MAIN CALCULATION - TRAIN THE AGENT
print("MAIN CALCULATION - TRAIN THE AGENT")
# Create environment and agent. We assume the data are already prepared, so we don't need
# to call EISDataPrep.
env, _, agent = environment_and_agent_builder(CONFIG_FILE)
final_model_file = agent.save_dir / "dqn_model.keras"
if final_model_file.exists():
    print(f"Loading existing model from {final_model_file}...")
    agent.load_model(final_model_file)
else:
    print("No existing model found. Training new model...")
    training_results = agent.train()


# EVALUATION STEP
print("\nEVALUATION STEP")
RESULTS_SAVE_DIR = agent.save_dir
print("Evaluation results will be saved in:", RESULTS_SAVE_DIR, "\n")

csv_path = RESULTS_SAVE_DIR / "eval_results.csv"
pkl_path = RESULTS_SAVE_DIR / "eval_results.pkl"
if csv_path.exists() and pkl_path.exists():
    print("Loading existing evaluation results...")
    eval_results = pd.read_pickle(pkl_path)
    print(f"✓ Loaded existing evaluation results from: {pkl_path}")
else:
    try:
        print("No existing evaluation results found. Performing evaluation...")
        eval_results = agent.eval_all_eis(max_actions=24, verbose=True, use_eval_env=False)
        # Save evaluation results
        print("Saving evaluation results...")
        eval_results.to_csv(csv_path, index=False)
        eval_results.to_pickle(pkl_path)
        print(f"✓ Saved evaluation CSV to: {csv_path}")
        print(f"✓ Saved evaluation pickle to: {pkl_path}")

    except Exception as e:
        print(f"⚠ Error during sample evaluation: {e}")
        traceback.print_exc()
