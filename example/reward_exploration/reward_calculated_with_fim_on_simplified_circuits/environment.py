"""AutoREC environment with a simplified-circuit FIM complexity penalty."""

from typing import Any

import autoeis as ae
import numpy as np

from autorec.environment import EIS_ECM_Env as BaseEISECMEnv
from autorec.parser import simplify

if __package__:
    from .fim_utils import compute_fim
else:
    from fim_utils import compute_fim


def _find_max_depth(nested: Any) -> int:
    """Return the maximum list-nesting depth."""
    if isinstance(nested, list):
        return 1 + max(_find_max_depth(item) for item in nested)
    return 0


class EIS_ECM_Env(BaseEISECMEnv):
    """Replace the standard component penalty with simplified-circuit FIM rank loss."""

    CHECKPOINT_FIELDS = frozenset({"identifiability_thresh"})

    def __init__(
        self,
        *args,
        identifiability_thresh: float = 1e-6,
        **kwargs,
    ):
        self.identifiability_thresh = identifiability_thresh
        super().__init__(*args, **kwargs)

    def _evaluate_circuit(self, circuit: str, components_number: int) -> dict:
        """Fit with the base environment, then replace a successful fit's reward."""
        result = super()._evaluate_circuit(circuit, components_number)
        if not result["terminated"]:
            return result

        try:
            reward, depth_penalty = self._calculate_fim_reward(
                circuit,
                result["param"],
                result["metrics"],
            )
            result["reward"] = reward
            result["depth_penalty"] = depth_penalty
            return result
        except Exception as exc:
            print(f"Error in FIM reward calculation: {exc}")
            return {
                "reward": -0.05,
                "metrics": None,
                "predicted_Z": None,
                "param": {},
                "terminated": False,
                "depth_penalty": 0,
            }

    def _calculate_fim_reward(
        self,
        circuit: str,
        parameters: dict[str, float],
        metrics: dict[str, float],
    ) -> tuple[float, float]:
        """Calculate a successful-fit reward using simplified-circuit FIM sensitivity."""
        components_penalty, simplified_circuit = self._fim_complexity_penalty(
            circuit,
            parameters,
        )

        nested_expr = ae.parser.circuit_to_nested_expr(simplified_circuit)
        max_depth = _find_max_depth(nested_expr) - 1
        depth_penalty = 0.2 if max_depth >= 3 else 0.0

        r2_average = (metrics["r2_score"] + metrics["r2_mag"] + metrics["r2_phase"]) / 3
        chi_term = np.log10(max(1 / metrics["chi_square"], 1e-10))
        r2_term = np.log10(max(1 / (1 - r2_average), 1e-10))
        reward = chi_term + r2_term - components_penalty - depth_penalty
        return float(reward), depth_penalty

    def _fim_complexity_penalty(
        self,
        circuit: str,
        parameters: dict[str, float],
    ) -> tuple[int, str]:
        """Count unidentifiable FIM directions after simplifying circuit and parameters."""
        simplified_circuit, simplified_parameters = simplify(circuit, parameters)
        parameter_names = ae.parser.get_parameter_labels(simplified_circuit)
        parameters_array = np.array([simplified_parameters[name] for name in parameter_names])

        ohmic_resistors = ae.parser.find_ohmic_resistors(simplified_circuit)
        if ohmic_resistors:
            resistor = ohmic_resistors[0]
            resistor_idx = parameter_names.index(resistor)
            resistor_scale = 0.01 * np.max(self.Z.real)
            if simplified_parameters[resistor] <= resistor_scale:
                parameters_array[resistor_idx] += resistor_scale

        _, eigenvalues, _ = compute_fim(
            simplified_circuit,
            self.freq,
            self.Z,
            parameters_array,
        )
        unidentifiable_count = int(np.sum(eigenvalues < self.identifiability_thresh))
        return unidentifiable_count, simplified_circuit

    @property
    def metadata(self) -> dict:
        """Include the FIM identifiability threshold in environment metadata."""
        metadata = super().metadata
        metadata["identifiability_thresh"] = self.identifiability_thresh
        return metadata
