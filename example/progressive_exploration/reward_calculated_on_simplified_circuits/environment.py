"""Experimental AutoREC environment with simplified-circuit complexity penalties."""

import autoeis as ae

from autorec.environment import EIS_ECM_Env as BaseEISECMEnv
from autorec.parser import simplify


class EIS_ECM_Env(BaseEISECMEnv):
    """Calculate successful-fit complexity penalties from a simplified circuit.

    All state transitions, fitting, observations, caching, and returned results retain the
    standard :class:`autorec.environment.EIS_ECM_Env` behavior. Only the circuit and
    component count supplied to the base reward calculation are replaced after a successful
    fit.
    """

    CHECKPOINT_FIELDS = frozenset()
    CHECKPOINT_SOURCE_MODULES = ()

    def _calculate_reward(
        self,
        r2_score: float,
        r2_mag: float,
        r2_phase: float,
        chi_square: float,
        circuit: str,
        components_count: int,
    ) -> tuple[float, bool, float]:
        """Use the simplified circuit for successful-fit complexity penalties."""
        if r2_score != 0 and chi_square < self.chi_threshold:
            circuit = simplify(circuit)
            components_count = len(ae.parser.get_component_types(circuit))

        return super()._calculate_reward(
            r2_score,
            r2_mag,
            r2_phase,
            chi_square,
            circuit,
            components_count,
        )
