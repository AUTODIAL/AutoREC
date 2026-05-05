import numpy as np
from scipy.interpolate import CubicSpline
import autoeis as ae
import impedance.validation
from impedance.validation import linKK
from utils import chi_obj_func


impedance.validation.circuit_elements["np"] = np  # Bug in impedance.validation


def interpolate_eis(freq, Z, freq_new):
    """
    Interpolate the EIS impedance data.

    The main idea is that given a list of new frequency values that we want to
    interpolate, we interpolate the impedance data for those new frequency values.

    Parameters
    ----------
    freq: np.array
        Original frequency data.
    Z: np.array
        Original impedance data.
    freq_new: np.array
        New array of frequencies that we want.

    Returns
    -------
    New, interpolated impedance values

    Usage
    -----
    >>> freq_new = np.logspace(*(np.log10(freq[[0, -1]])), npoints, endpoint=True)
    >>> Z_new = interpolate_eis(freq, Z, freq_new)
    """
    idx_sort = np.argsort(freq)
    idx_sort_new = np.argsort(freq_new)
    real_new = CubicSpline(freq[idx_sort], Z[idx_sort].real)(freq_new[idx_sort_new])
    imag_new = CubicSpline(freq[idx_sort], Z[idx_sort].imag)(freq_new[idx_sort_new])
    # Revert the sorting
    inv_idx_sort_new = np.argsort(idx_sort_new)
    return real_new[inv_idx_sort_new] + 1j * imag_new[inv_idx_sort_new]


def find_both_threshold(
    experiment_freq, experiment_Z, Upper_bound_decrease=0.997, lower_bound_increase=2.5
):
    # tol_linKK= 5e-1
    for _ in range(3):
        linKK_kwargs = {"c": 0.5, "max_M": 100, "fit_type": "complex", "add_cap": True}
        linKK_silent = ae.utils.suppress_output_legacy(
            linKK,
        )
        M, mu, Z_linKK, res_real, res_imag = linKK_silent(
            experiment_freq, experiment_Z, **linKK_kwargs
        )

    r2_kk = ae.metrics.r2_score(experiment_Z, Z_linKK)
    r2_thresh_kk = r2_kk * Upper_bound_decrease

    chi_kk = chi_obj_func(experiment_Z, Z_linKK)
    chi_thresh_kk = lower_bound_increase * chi_kk

    return r2_thresh_kk, chi_thresh_kk


def find_chi_threshold(experiment_freq, experiment_Z, lower_bound_increase=2.5):
    # tol_linKK= 5e-1
    for _ in range(3):
        linKK_kwargs = {"c": 0.5, "max_M": 100, "fit_type": "complex", "add_cap": True}
        linKK_silent = ae.utils.suppress_output_legacy(
            linKK,
        )
        M, mu, Z_linKK, res_real, res_imag = linKK_silent(
            experiment_freq, experiment_Z, **linKK_kwargs
        )

    chi_kk = chi_obj_func(experiment_Z, Z_linKK)
    chi_thresh_kk = lower_bound_increase * chi_kk

    return chi_thresh_kk


def find_r2_threshold(experiment_freq, experiment_Z, Upper_bound_decrease=0.997):
    # tol_linKK= 5e-1
    for _ in range(3):
        linKK_kwargs = {"c": 0.5, "max_M": 100, "fit_type": "complex", "add_cap": True}
        linKK_silent = ae.utils.suppress_output_legacy(
            linKK,
        )
        M, mu, Z_linKK, res_real, res_imag = linKK_silent(
            experiment_freq, experiment_Z, **linKK_kwargs
        )

    r2_kk = ae.metrics.r2_score(experiment_Z, Z_linKK)
    r2_thresh_kk = r2_kk * Upper_bound_decrease

    return r2_thresh_kk


def normalize_EIS(experiment_Z):
    """
    Normalize EIS data using min-max normalization.
    Returns normalized impedance, angles, magnitude, and scaled magnitude.
    """

    def get_norm(value):
        """Generic min-max normalization: scales to [0, 1] range"""
        value_shifted_by_offset = value - np.min(value)
        return value_shifted_by_offset / np.max(value_shifted_by_offset)

    real_norm = get_norm(np.real(experiment_Z))
    imag_norm = get_norm(np.imag(experiment_Z))
    Z_norm = real_norm + 1j * imag_norm

    angles = np.angle(experiment_Z)
    angles_norm = get_norm(angles)

    mag = np.abs(experiment_Z)
    mag_norm = get_norm(mag)

    mag_scaled = np.log10(np.abs(experiment_Z)) / max(np.log10(np.abs(experiment_Z)))

    return Z_norm, angles_norm, mag_norm, mag_scaled


def flatten_EIS(experiment_Z, quatities=["ImZ", "phi", "mag", "nphi"]):
    """
    Flatten normalized EIS data into 1D array.
    Multiple stacking options available (currently using imag, angles, mag_scaled, -angles).
    """
    Z, phi, _, mag = normalize_EIS(experiment_Z)
    ReZ = np.real(Z)
    ImZ = np.imag(Z)
    all_values = {
        "ReZ": ReZ,
        "ImZ": ImZ,
        "phi": phi,
        "mag": mag,
        "nReZ": -ReZ,
        "nImZ": -ImZ,
        "nphi": -phi,
        "nmag": -mag,
    }
    flatten_Z = np.concatenate([all_values[q] for q in quatities])
    return flatten_Z
