"""Run continuous-FIM reward multi-fidelity hyperparameter tuning."""

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CASE_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from example.reward_exploration.tuning.runner import main


if __name__ == "__main__":
    main(CASE_DIR)
