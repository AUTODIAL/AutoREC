"""Run baseline reward multi-fidelity hyperparameter tuning."""

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent
EXPLORATION_ROOT = CASE_DIR.parent
sys.path.insert(0, str(EXPLORATION_ROOT.parent))

from reward_exploration.tuning.runner import main


if __name__ == "__main__":
    main(CASE_DIR)
