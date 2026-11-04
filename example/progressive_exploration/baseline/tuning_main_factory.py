"""Run one baseline reward hyperparameter-tuning fidelity stage."""

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent
EXPLORATION_ROOT = CASE_DIR.parent
sys.path.insert(0, str(EXPLORATION_ROOT.parent))

from progressive_exploration.tuning.stage import main


if __name__ == "__main__":
    main(default_config=CASE_DIR / "tuning_base_config.yaml")
