"""Generate configurations for initial hyperparameter search. We will use the results to
perform Bayesian optimization later.
To better cover the hyperparameter space for this initial search, we will use latin
hypercube sampling to generate configurations.
"""

from pathlib import Path
import argparse
import yaml
import pprint
from glob import glob
import re
from tqdm import tqdm
import numpy as np
from scipy.stats import qmc

from hyperparameter_tuning_utils import search_space_bounds_array, sample_to_config

seed = 42
np.random.seed(seed)


WORK_DIR = Path(__file__).parent
RESULTS_DIR = WORK_DIR / "results"

# Command line arguments
parser = argparse.ArgumentParser("Generate hyperparameter configurations")
parser.add_argument(
    "--nsamples", "-n", type=int, default=100, help="Number of configurations to generate"
)
parser.add_argument(
    "--configs-dir",
    "-c",
    type=str,
    default=str(WORK_DIR / "configs"),
    help="Directory to save generated configurations",
)
args = parser.parse_args()

CONFIGS_DIR = Path(args.configs_dir)
CONFIGS_DIR.mkdir(exist_ok=True, parents=True)

# Read base configuration
print("Reading base configuration...")
with open(WORK_DIR / "default_configs" / "environment_agent_config_long.yaml", "r") as f:
    base_config = yaml.safe_load(f)
pprint.pprint(base_config, sort_dicts=False)


#########################################################################################
# Latin hypercube sampling in hyperparameter space
#########################################################################################
print(f"Generating {args.nsamples} Latin Hypercube Samples...")
sampler = qmc.LatinHypercube(d=len(search_space_bounds_array), seed=seed)
unit_samples = sampler.random(args.nsamples)
# Scale to bounds
samples = qmc.scale(unit_samples, *(search_space_bounds_array.T))
# Convert samples to configuration dictionaries
generated_configs_list = [sample_to_config(row, base_config) for row in samples]
print(f"Generated {len(generated_configs_list)} configurations.")


#########################################################################################
# Write generated configurations to files, avoiding duplicates
#########################################################################################
# Collect all configurations that have been generated
print("Collecting existing configuration files...")
config_file_pattern = str(CONFIGS_DIR / "config_[0-9][0-9][0-9][0-9][0-9][0-9].yaml")
all_config_files = sorted(glob(config_file_pattern))
print(f"Found {len(all_config_files)} existing configuration files.")
# Load existing configurations
existing_configs = {}
for config_file in tqdm(all_config_files):
    # Get the config number
    match = re.search(r"config_(\d+)\.yaml", Path(config_file).name)
    config_number = int(match.group(1))
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        existing_configs[config_number] = config
existing_configs_list = [dd for dd in existing_configs.values()]

# Check if the generated configurations already exist
print("Checking for existing configurations...")
new_configs_list = []
for gen_config in tqdm(generated_configs_list):
    config1 = gen_config.copy()
    config1["agent"]["save_dir"] = None  # ignore directory when comparing
    for existing_config in existing_configs_list:
        config2 = existing_config.copy()
        config2["agent"]["save_dir"] = None  # ignore directory when comparing
        if config1 == config2:
            break
    else:
        new_configs_list.append(gen_config)
print(
    f"Generated {len(new_configs_list)} new configurations "
    "after removing existing ones."
)

# Write new configurations to files
print("Writing new configurations to files...")
# Get the config number
# There might be some missing values in the existing configs
existing_config_numbers = existing_configs.keys()
missing_config_numbers = []
if len(existing_config_numbers) > 0:
    for num in range(max(existing_config_numbers)):
        if num not in existing_config_numbers:
            missing_config_numbers.append(num)
# Write new configurations to files
for new_config in new_configs_list:
    if len(missing_config_numbers) == 0:
        try:
            counter
        except NameError:
            if len(existing_config_numbers) == 0:
                counter = 0
            else:
                counter = max(existing_config_numbers) + 1
        config_number = counter
        counter += 1
    else:
        config_number = missing_config_numbers.pop(0)
    # Update the directory
    new_config["agent"]["save_dir"] = str(RESULTS_DIR / f"run_{config_number:06d}")
    # Write to file
    config_file = CONFIGS_DIR / f"config_{config_number:06d}.yaml"
    print(f"Writing configuration to {config_file.name}...")
    # with open(config_file, "w") as f:
    #     yaml.dump(new_config, f, sort_keys=False)
