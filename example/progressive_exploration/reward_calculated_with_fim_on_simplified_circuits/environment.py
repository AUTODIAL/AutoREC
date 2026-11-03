"""Re-export the canonical discrete-FIM reward environment."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from example.reward_exploration.reward_calculated_with_fim_on_simplified_circuits.environment import (
    EIS_ECM_Env,
)


__all__ = ["EIS_ECM_Env"]
