"""Run simplified-circuit reward multi-fidelity hyperparameter tuning."""

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent))

from tuning.runner import main


if __name__ == "__main__":
    main(CASE_DIR)
