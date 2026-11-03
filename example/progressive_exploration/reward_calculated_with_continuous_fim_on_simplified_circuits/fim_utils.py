"""Fisher information calculation used by the FIM reward experiment."""

import autoeis as ae
import numdifftools as nd
import numpy as np


def _transform_parameters(circuit: str, parameters: np.ndarray) -> np.ndarray:
    """Transform positive parameters to log space while leaving CPE exponents linear."""
    parameter_names = ae.parser.get_parameter_labels(circuit)
    transformed = np.asarray(parameters, dtype=float).copy()
    for ii, name in enumerate(parameter_names):
        if not (name.startswith("P") and name.endswith("n")):
            transformed[ii] = np.log(transformed[ii])
    return transformed


def _inverse_transform(circuit: str, transformed: np.ndarray) -> np.ndarray:
    """Map log-transformed parameters back to the circuit's physical parameter space."""
    parameter_names = ae.parser.get_parameter_labels(circuit)
    parameters = np.asarray(transformed, dtype=float).copy()
    for ii, name in enumerate(parameter_names):
        if not (name.startswith("P") and name.endswith("n")):
            parameters[ii] = np.exp(parameters[ii])
    return parameters


def compute_fim(
    circuit: str,
    frequency: np.ndarray,
    impedance: np.ndarray,
    parameters: np.ndarray,
    method: str = "UW",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the FIM and normalized eigenpairs using unweighted complex residuals."""
    if method != "UW":
        raise ValueError("This experiment implements the downloaded environment's UW FIM.")

    circuit_fn = ae.utils.generate_circuit_fn(circuit, jit=True)
    impedance_scale = np.max(np.abs(impedance))

    def residual(transformed):
        predicted = circuit_fn(frequency, _inverse_transform(circuit, transformed))
        return (
            np.hstack((predicted.real - impedance.real, predicted.imag - impedance.imag))
            / impedance_scale
        )

    transformed = _transform_parameters(circuit, parameters)
    jacobian = nd.Jacobian(residual)(transformed)
    fim = jacobian.T @ jacobian
    fim_norm = np.linalg.norm(fim)
    if not np.isfinite(fim_norm) or fim_norm == 0.0:
        raise ValueError("Cannot normalize a zero or non-finite Fisher Information Matrix.")

    eigenvalues, eigenvectors = np.linalg.eigh(fim / fim_norm)
    largest_eigenvalue = float(np.max(eigenvalues))
    if largest_eigenvalue <= 0.0:
        raise ValueError("Fisher Information Matrix has no positive eigenvalues.")
    eigenvalues = eigenvalues / largest_eigenvalue

    if verbose:
        print("Eigenvalues of FIM (normalized):", eigenvalues)
    return fim, eigenvalues, eigenvectors
