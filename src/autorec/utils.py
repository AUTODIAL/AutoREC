from collections.abc import Callable, Iterable, Mapping
import contextlib
import glob
import io
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import autoeis as ae
from autoeis.utils import parse_initial_guess, generate_initial_guess, generate_circuit_fn

import arviz as az
from impedance.validation import linKK
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import pandas as pd
import jax.numpy as jnp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import tensorflow as tf


log = logging.getLogger(__name__)

class DivergenceError(Exception):
    """Raised when circuit parameter fitting fails to converge."""

    pass

ec = ae.core.ec


def save_evaluation_results(
    eval_df: pd.DataFrame,
    run_id: str,
    results_save_dir: str | os.PathLike = "results",
) -> tuple[Path, Path]:
    """
    Save evaluation results in both CSV and pickle formats.

    CSV is human-readable and useful for quick analysis. Pickle preserves
    complex data types such as circuits and arrays.

    Args:
        eval_df: Evaluation results DataFrame.
        run_id: Unique identifier for this evaluation run.
        results_save_dir: Directory where result files should be saved.

    Returns:
        Tuple of ``(csv_path, pkl_path)`` for the saved files.
    """
    results_save_dir = Path(results_save_dir)
    results_save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_save_dir / f"eval_results_{run_id}.csv"
    eval_df.to_csv(csv_path, index=False)
    print(f"✓ Saved evaluation CSV to: {csv_path}")

    pkl_path = results_save_dir / f"eval_results_{run_id}.pkl"
    eval_df.to_pickle(pkl_path)
    print(f"✓ Saved evaluation pickle to: {pkl_path}")

    return csv_path, pkl_path


def set_global_seed(seed: int = 42, deterministic_ops: bool = True):
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed: Random seed value
        deterministic_ops: If True, enables deterministic operations in TensorFlow
                          (may reduce performance but ensures reproducibility)
    
    Notes:
        - Must be called BEFORE importing any modules that use randomness
        - Must be called BEFORE any random operations occur
        - Some operations (especially GPU) may still have non-deterministic behavior
    """
    import os
    import random
    import numpy as np
    import tensorflow as tf
    
    # 1. Set Python's built-in random seed
    random.seed(seed)
    
    # 2. Set NumPy random seed
    np.random.seed(seed)
    
    # 3. Set TensorFlow seeds
    tf.random.set_seed(seed)
    
    # 4. Set environment variables for hash seed (affects Python's hash randomization)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 5. Configure TensorFlow for deterministic operations
    if deterministic_ops:
        # Enable deterministic ops (TF 2.x)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        # For older TensorFlow versions, also set:
        # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        try:
            # Enable op determinism (TF 2.9+)
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            print("Warning: tf.config.experimental.enable_op_determinism() not available")
            print("Using TF_DETERMINISTIC_OPS environment variable instead")
    
    # 6. Limit TensorFlow parallelism for reproducibility
    # This reduces non-determinism from parallel execution
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    print(f"Global seed set to {seed}")
    print(f"Deterministic operations: {deterministic_ops}")

def chi_obj_func(Z, Z_pred):
    """Computes ECM error based on residual-based χ2."""
    residual = (Z_pred.real - Z.real) ** 2 + (Z_pred.imag - Z.imag) ** 2
    weight = 1 / (Z.real**2 + Z.imag**2)
    return (residual * weight).mean()


def _is_nonblank_image(path: Path) -> bool:
    """Return True when a saved circuit image contains visible non-white content."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image)
    if pixels.size == 0:
        return False

    return bool(np.any(pixels < 245))


def draw_circuit_png(circuit: str, output_path: Path, dpi: int) -> None:
    """Draw an AutoEIS circuit string with lcapy and save a validated PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    plt.close("all")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from lcapy import CPE as P
            from lcapy import C, L, R
    except ImportError as exc:
        raise RuntimeError(
            "Could not import lcapy while drawing the circuit. In the active "
            "notebook kernel, run: import lcapy"
        ) from exc

    lcapy_expr = circuit.replace("[", "(").replace("]", ")")
    lcapy_expr = lcapy_expr.replace(",", "|").replace("-", "+")
    lcapy_expr = re.sub(r"([A-Z])(\d+)", r'\1("\1\2")', lcapy_expr)
    try:
        circuit_network = eval(
            lcapy_expr,
            {"__builtins__": {}},
            {"C": C, "L": L, "P": P, "R": R},
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            circuit_network.draw(filename=str(output_path), style="american", dpi=dpi)
    except Exception as exc:
        raise RuntimeError(
            f"lcapy failed to draw circuit {circuit!r} "
            f"(converted expression: {lcapy_expr!r})."
        ) from exc

    if not output_path.exists() or not _is_nonblank_image(output_path):
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"lcapy produced a blank circuit diagram for: {circuit}")

    plt.close("all")


def plot_eval_fit(
    EIS_i,
    Z_true,
    predicted_Z,
    good_fit,
    metrics,
    save_dir,
    circuit=None,
):
    """Save a fit plot for a single evaluated EIS, optionally with circuit diagram."""
    save_dir = Path(save_dir)
    (save_dir / "fit_plots").mkdir(exist_ok=True)

    if not good_fit or predicted_Z is None:
        return

    circuit_img = None
    if circuit:
        from PIL import Image
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_circuit_path = Path(temp_file.name)
        try:
            draw_circuit_png(circuit, temp_circuit_path, dpi=150)
            circuit_img = Image.open(temp_circuit_path).copy()
        finally:
            temp_circuit_path.unlink(missing_ok=True)

    if circuit_img is not None:
        fig = plt.figure(figsize=(12, 5.2))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.2, 1],
            height_ratios=[1, 0.24],
            hspace=0.32,
            wspace=0.28,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_metrics = fig.add_subplot(gs[1, 0])
        ax_circuit = fig.add_subplot(gs[:, 1])
    else:
        fig = plt.figure(figsize=(7, 4.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.24], hspace=0.32)
        ax = fig.add_subplot(gs[0, 0])
        ax_metrics = fig.add_subplot(gs[1, 0])
        ax_circuit = None

    ax.scatter(predicted_Z.real, -predicted_Z.imag, label="predicted", alpha=0.7)
    ax.scatter(Z_true.real, -Z_true.imag, label="True", alpha=0.7)
    ax.legend(loc="upper left", fontsize="medium")
    ax.set_xlabel("Re(Z)")
    ax.set_ylabel("-Im(Z)")
    ax.set_title(f"Nyquist plot - EIS {EIS_i+1}")

    metrics_text = (
        f"Reached terminal state: {good_fit}\n"
        f"Final r² score: {metrics.get('r2_score', 'N/A'):.10f}\n"
        f"Final χ²: {metrics.get('chi_square', 'N/A'):.10f}"
    )

    ax_metrics.axis("off")
    ax_metrics.text(
        0.02,
        0.95,
        metrics_text,
        transform=ax_metrics.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if ax_circuit is not None:
        ax_circuit.imshow(circuit_img)
        ax_circuit.axis("off")
        ax_circuit.set_title(f"Circuit: {circuit}")

    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.12)
    plt.savefig(
        save_dir / "fit_plots" / f"fit_EIS_i_{EIS_i+1}.png",
        bbox_inches="tight",
    )
    plt.close()


def state_encode(state, ELEMENTS_extended):
    """
    One-hot encode a GEP chromosome state for neural-network input.

    The expressed coding region is detected from the circuit tree. Any
    non-coding chromosome positions are replaced with ``X`` before one-hot
    encoding so the returned vector has a fixed length for a given chromosome
    configuration.

    Parameters
    ----------
    state : str
        GEP chromosome string using ``/`` for parallel connections.
    ELEMENTS_extended : list[str]
        Ordered list of allowed symbols, including ``X`` for non-coding
        positions.

    Returns
    -------
    np.ndarray
        Flattened one-hot encoded state vector.
    """
    ec = ae.core.ec
    karva = state.replace('/', '-')
    tree = ec.karva_to_tree(karva)
    conding_length = len(tree)
    len_non_coding = len(state) - conding_length

    coding = state[0:conding_length]
    refined_state = coding+'X'*len_non_coding
    refined_state_index = [ELEMENTS_extended.index(char) for char in refined_state]
    encoded_string = tf.one_hot(refined_state_index, depth=len(ELEMENTS_extended))
    # bipolar_encoded_string = 2 * encoded_string - 1
    flattened_encoded_string = tf.reshape(encoded_string, [-1]).numpy()

    return flattened_encoded_string

def parse_state_to_circuit(state: str) -> tuple[str, int, str]:
    """
    Parse chromosome state into circuit representation.

    Args:
        state: Chromosome string (e.g., '+RPRRRR')

    Returns:
        tuple: (circuit, coding_length, coding)
            - circuit: Human-readable circuit notation (e.g., 'R0-p(R1,P2)')
            - coding_length: Length of expressed (coding) region
            - coding: The coding portion of the chromosome
    """
    state_karva = state.replace("/", "-")
    tree = ec.karva_to_tree(state_karva)
    circuit = ec.tree_to_circuit(tree)[0]
    coding_length = len(tree)
    coding = state[0:coding_length]
    return circuit, coding_length, coding

def karva_to_circuit(karva: str):
    """
    Convert a karva representation of a circuit into its corresponding circuit structure 

    Parameters:
    - karva (str): The karva representation of the circuit: karva is the same as GEP, with '/' replaced by '-'.

    Returns:
    - circuit: The circuit structure obtained from the karva representation.
    """
    ec = ae.core.ec
    tree = ec.karva_to_tree(karva)
    circuit = ec.tree_to_circuit(tree)[0]
    # function = ae.utils.generate_circuit_fn(circuit)
    return circuit 


def validity_check(circuit: str):
    """
    Check the validity of the circuit based on:
    1. Non-empty, no duplication, and contains valid elements.
    2. Must contain at least one ohmic resistor.
    3. Must contain at least one parallel route.
    4. Optionally check for circuit depth (commented out).

    Parameters:
    circuit (str): an string with a CDC representation of the circuit.

    Returns:
    validity (bool): True if the circuit is valid, False otherwise.
    """
    validity = ae.parser.validate_circuit(circuit)  # First check: not empty/duplication, contains valid element
    
    # Second check: circuits without an ohmic resistance
    resistors = ae.parser.find_ohmic_resistors(circuit)      
    if not resistors:
        validity = False

    # Third check: circuits with series-only components
    contains_parallel_route = "[" in circuit
    if not contains_parallel_route:
        validity = False

    # Fourth check: circuits depth
    # nested_expr = ae.parser.circuit_to_nested_expr(circuit)
    # def find_max_depth(nested_list):
    #     if isinstance(nested_list, list):
    #         return 1 + max(find_max_depth(item) for item in nested_list)
    #     else:
    #         return 0
    # max_depth = find_max_depth(nested_expr) - 1
    # if max_depth >= 3:
    #     validity = False

    return validity

def action_validity(state, action_type, action_position):
    """
    Check whether replacing one chromosome position creates a valid action.

    An action is considered invalid if it changes a non-coding position, leaves
    the chromosome unchanged, or produces a circuit that fails the circuit
    validity checks.

    Parameters
    ----------
    state : str
        Current GEP chromosome state.
    action_type : str
        Symbol to place at ``action_position``.
    action_position : int
        Chromosome index to mutate.

    Returns
    -------
    int
        ``1`` if the action is valid, otherwise ``0``.
    """
    new_state = state[:action_position] + f"{action_type}" + state[action_position+1:]
    karva = new_state.replace('/', '-')
    tree = ec.karva_to_tree(karva)
    conding_length = len(tree)
    # coding = state[0:conding_length]
    circuit = karva_to_circuit(karva)
    validity = validity_check(circuit)
    if  action_position > conding_length-1:
        validity = False 
    if state == new_state:
        validity = False 

    if validity == False:
        validity_state = 0
    else:
        validity_state = 1
    
    return validity_state

def get_parameter_bounds(circuit: str) -> tuple:
    """Returns a 2-element tuple of lower and upper bounds, to be used in
    SciPy's ``least_squares``.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    tuple
        A 2-element tuple of lower and upper bounds for the circuit parameters.
    """
    bounds_dict = {
        "R": (0.0, 1e9),
        "C": (0.0, 10.0),
        "Pw": (0.0, 1e9),
        "Pn": (0.0, 1.0),
        "L": (0.0, 5.0),
    }
    types = ae.parser.get_parameter_types(circuit)
    bounds = [bounds_dict[type_] for type_ in types]
    bounds = tuple(zip(*bounds))
    return bounds

def fit_circuit_parameters_NEW(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Mapping[str, float] | Iterable[float] = None,
    max_iters: int = 50,
    min_iters: int = 25,
    bounds: Iterable[tuple] = None,
    max_nfev: int = None,
    ftol: float = 1e-15,
    xtol: float = 1e-15,
    tol_chi_squared: float = 1e-2,
    method: str = "log-B",
    verbose: bool = False,
) -> dict[str, float]:
    """Fit circuit parameters to impedance data.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Impedance data.
    p0 : Mapping[str, float] | Iterable[float], optional
        Initial guess for the circuit parameters. Default is None.
    max_iters : int, optional
        Maximum number of fitting attempts. Default is 50.
    min_iters : int, optional
        Minimum number of fitting attempts before early stopping. Default is 25.
        If ``min_iters`` is reached AND circuit fitter converges, the fitting
        process stops.
    bounds : Iterable[tuple], optional
        List of two tuples, each containing the lower and upper bounds,
        respectively, for the circuit parameters. Default is None. The order
        of the values should match the order of the circuit parameters as
        returned by ``parser.get_parameter_labels``.
    max_nfev : int, optional
        Maximum number of function evaluations for the circuit fitter.
        Default is None. See ``scipy.optimize.least_squares`` for details.
    ftol : float, optional
        Relative tolerance for termination by cost-function change. Default is
        1e-15. See ``scipy.optimize.least_squares`` for details.
    xtol : float, optional
        Relative tolerance for termination by parameter change. Default is
        1e-15. See ``scipy.optimize.least_squares`` for details.
    tol_chi_squared : float, optional
        Tolerance for the chi-squared error. This only gets triggered if
        ``min_iters`` is set. A good chi-squared value is 1e-3 or smaller.
        Default is 1e-2.
    method : str, optional
        Objective function to use for fitting. Choose from ``"UW"``, ``"X2"``,
        ``"PW"``, ``"B"``, ``"log-B"``, and ``"log-BW"``:

          * ``"UW"``: unweighted real and imaginary residuals.
          * ``"X2"``: residuals weighted by impedance magnitude.
          * ``"PW"``: real and imaginary residuals weighted separately.
          * ``"B"``: magnitude and phase residuals.
          * ``"log-B"``: log-magnitude and phase residuals.
          * ``"log-BW"``: weighted log-magnitude and phase residuals.

        Default is ``"log-B"``.

    verbose : bool, optional
        If True, prints the fitting results. Default is False.

    Returns
    -------
    tuple
        ``(parameters, chi_square, r2_score, r2_mag, r2_phase)`` where
        ``parameters`` is a dictionary mapping parameter names to fitted values.

    Notes
    -----
    This function uses SciPy's ``least_squares`` to fit the circuit parameters.
    """
    def obj_UW(p):
        """Computes ECM error based on the Nyquist plot."""
        Z_pred = fn(freq, p)
        res = jnp.hstack((Z_pred.real - Z.real, Z_pred.imag - Z.imag))
        return res
    
    def obj_X2(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual_real = (Z_pred.real - Z.real)
        residual_imag = (Z_pred.imag - Z.imag)
        weight = 1 / np.sqrt(Z.real**2 + Z.imag**2)
        res = jnp.hstack((residual_real*weight, residual_imag*weight))
        return res

    def obj_PW(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual_real = (Z_pred.real - Z.real)
        residual_imag = (Z_pred.imag - Z.imag)
        weight_real = 1 / Z.real
        weight_imag = 1 / Z.imag
        res = jnp.hstack((residual_real*weight_real, residual_imag*weight_imag))
        return res
    
    def obj_B(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        # res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        return res
    
    def obj_log_B(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((jnp.log10(mag) - jnp.log10(mag_gt), phase - phase_gt))
        # res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        return res
    
    def obj_log_BW(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((jnp.log10(mag / mag_gt)/jnp.log10(mag_gt), (phase - phase_gt)/phase_gt))
        # res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        return res
    
    def obj_chi_squared(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual = (Z_pred.real - Z.real) ** 2 + (Z_pred.imag - Z.imag) ** 2
        weight = 1 / (Z.real**2 + Z.imag**2)
        return residual * weight

    def obj_phase_chi(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)

        residual = (Z_pred.real - Z.real) ** 2 + (Z_pred.imag - Z.imag) ** 2
        weight = 1 / (Z.real**2 + Z.imag**2)
    
        phase = jnp.angle(Z_pred)
        res = jnp.hstack(((phase - phase_gt)/phase_gt, residual * weight))
        return res
    
    def obj_mag(p):
        """Computes ECM error based on the magnitude of impedance deviation."""
        Z_pred = fn(freq, p)
        res = jnp.abs(Z - Z_pred)
        return res

    msg = f"Invalid method: {method}. Use 'chi-squared', 'nyquist', 'bode', or 'magnitude'."
    assert method in ["UW", "X2", "PW", "B", "log-B", "log-BW"], msg
    assert len(freq) == len(Z), "Length of frequency and impedance data must match."

    fn = generate_circuit_fn(circuit, jit=True)
    obj = {
        "UW": obj_UW,
        "X2": obj_X2,
        "PW": obj_PW,
        "B": obj_B,
        "log-B" : obj_log_B,
        "log-BW": obj_log_BW

    }[method]

    mag_gt = jnp.abs(Z)
    phase_gt = jnp.angle(Z)

    # Sanitize initial guess
    p0 = parse_initial_guess(p0) if p0 is not None else generate_initial_guess(circuit)
    num_params = ae.parser.count_parameters(circuit)
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."

    # Assemble kwargs for curve_fit
    bounds = get_parameter_bounds(circuit) if bounds is None else bounds
    kwargs = {"x0": p0, "bounds": bounds, "max_nfev": max_nfev, "ftol": ftol, "xtol": xtol}

    # Ensure p0 is not out-of-bounds
    if p0 is not None:
        for i, (lower, upper) in enumerate(zip(*bounds)):
            p0[i] = np.clip(p0[i], lower, upper)

    # Fit circuit parameters by brute force
    min_iters = max_iters if min_iters is None else min_iters
    err_min = np.inf

    for i in range(max_iters):
        res = least_squares(obj, verbose=verbose, **kwargs)
        if (err := norm(obj(res.x))) < err_min:
            err_min = err
            p0 = res.x
        converged = (X2 := obj_chi_squared(res.x).mean()) < tol_chi_squared
        if i + 1 >= min_iters and converged:
            break
        kwargs["x0"] = generate_initial_guess(circuit)

    r2_score = ae.metrics.r2_score(Z, fn(freq, p0))
    r2_mag = ae.metrics.r2_score(jnp.abs(Z), jnp.abs(fn(freq, p0)))
    r2_phase = ae.metrics.r2_score(jnp.angle(Z), jnp.angle(fn(freq, p0)))
    X2 = obj_chi_squared(p0).mean()
    log.info(
        f"Converged in {i+1} iterations with "
        f"X^2 = {X2:.3e}, R^2 (|Z|) = {r2_mag:.4f}, R^2 (phase) = {r2_phase:.4f}"
    )

    if err_min == np.inf:
        raise DivergenceError(
            "Failed to fit the circuit parameters. Try increasing 'iters' or "
            "'maxfev', or narrow down the search by providing 'bounds'."
        )

    variables = ae.parser.get_parameter_labels(circuit)
    return dict(zip(variables, p0)), X2, r2_score, r2_mag, r2_phase


def plot_bar(data, title = None, filename = None, color = 'red'):
    """
    Save a labeled bar plot for a mapping of category values.

    Parameters
    ----------
    data : Mapping
        Category labels mapped to numeric values.
    title : str, optional
        Plot title. Currently retained for API compatibility.
    xlabel : str, optional
        X-axis label. Currently retained for API compatibility.
    ylabel : str, optional
        Y-axis label. Currently retained for API compatibility.
    filename : str or Path, optional
        Output filename stem. The function appends ``.png``.
    color : str, optional
        Bar color. Default is ``"red"``.
    """
    plt.figure(figsize=(8, 5))
    bars = plt.bar(data.keys(), data.values(), color=color)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,  # x coordinate: center of the bar
            yval,                            # y coordinate: top of the bar
            f'{yval:.2f}',                       # text: the value of the bar
            ha='center',                     # horizontal alignment: center
            va='bottom'                      # vertical alignment: bottom (just above the bar)
        )

    # Labels and title
    plt.xlabel("Categories")
    plt.ylabel("Success rate %")
    plt.title("Bar Plot of Given Data ({title})" if title else "Bar Plot of Given Data")
    plt.ylim(0,100)

    # Rotate x labels for better visibility
    plt.xticks(rotation=45, ha="right")

    # Show the plot
    if filename is not None:
        plt.savefig(f'{filename}.png')
    # plt.clf()


def create_circuit_evolution_visualization(
    # Input data
    freq,
    Z_true,
    circuits,
    circuit_predictions,  # List of Z_sim for each circuit
    # Computed metrics
    # avg_errors,
    r2_mag_list,
    r2_phase_list,
    r2_scores,
    chi2_scores,
    passed_thresholds,
    # Configuration parameters with defaults
    error_threshold=100,
    eis_id=1,
    output_dir=None,
    frame_dpi=150,
    gif_frame_duration=2500,
    gif_final_frame_duration=5000,
    cleanup_temp_files=True,
    save_dir = None
):
    """
    Create circuit evolution visualization frames and GIF for evaluation results.

    Parameters:
    -----------
    freq : array
        Frequency data
    Z_true : array
        True impedance data
    circuits : list
        List of circuit strings showing evolution
    circuit_predictions : list
        List of predicted impedance arrays (Z_sim) for each circuit
    r2_mag_list : list
        R-squared values for impedance magnitude at each circuit stage
    r2_phase_list : list
        R-squared values for impedance phase at each circuit stage
    r2_scores : list
        Overall R-squared scores for each circuit
    chi2_scores : list
        Chi-squared scores for each circuit
    passed_thresholds : list
        Boolean list indicating if each circuit passed threshold
    error_threshold : float
        Error threshold in Ohms
    eis_id : int
        EIS dataset identifier for title
    output_dir : str
        Directory for output files. Defaults to 'save_dir/generated_circuits'
    frame_dpi : int
        DPI for saved frames
    gif_frame_duration : int
        Duration of each frame in milliseconds
    gif_final_frame_duration : int
        Duration of final frame in milliseconds
    cleanup_temp_files : bool
        Whether to cleanup temporary files after GIF creation
    save_dir : str or Path, optional
        Base directory used when ``output_dir`` is not provided
    """
    from PIL import Image
    from matplotlib.patches import Rectangle

    # plt.rcParams['font.family'] = ''
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["sans-serif"]
    ae.visualization.set_plot_style()

    print("Generating circuit evolution GIF...")

    # Create output directories
    if output_dir is None:
        output_dir = save_dir / "generated_circuits"
    else:
        output_dir = Path(output_dir)
    (output_dir / "final_circuits").mkdir(exist_ok=True, parents=True)

    # Clean up old frames
    for f in glob.glob(str(output_dir / "frame_*.png")):
        os.remove(f)
    for f in glob.glob(str(output_dir / "temp_circuit_*.png")):
        os.remove(f)

    # Generate frames for each circuit
    for i, circuit in enumerate(circuits):
        print(f"  Processing circuit {i+1}/{len(circuits)}: {circuit}")

        # Get precomputed data for this circuit
        Z_sim = circuit_predictions[i]
        # avg_error = avg_errors[i]
        r2_mag = r2_mag_list[i]
        r2_phase = r2_phase_list[i]
        r2 = r2_scores[i]
        chi2 = chi2_scores[i]
        passed_threshold = passed_thresholds[i]

        # Step 1: Draw circuit diagram separately and save it
        temp_circuit_path = output_dir / f"temp_circuit_{i}.png"
        draw_circuit_png(circuit, temp_circuit_path, frame_dpi)

        # Step 2: Create combined figure with custom layout
        fig = plt.figure(figsize=(16, 8.5))

        # Add main title for the entire figure
        fig.suptitle(
            f"Nyquist Plot & Circuit Generation for EIS: {eis_id}",
            fontsize=16,
            fontweight="bold",
            # family='',
            y=0.97,
        )

        # Create grid: left side for Nyquist plot, right side split for circuit (top) and metrics (bottom)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.5, 1],
            height_ratios=[1, 1],
            left=0.05,
            right=0.95,
            top=0.92,
            bottom=0.05,
            hspace=0.3,
            wspace=0.25,
        )

        # Left: Nyquist plot (spans both rows)
        ax_nyquist = fig.add_subplot(gs[:, 0])
        ax_nyquist.plot(
            Z_true.real,
            -Z_true.imag,
            marker="o",
            linestyle="None",
            markersize=6,
            label="True EIS",
        )
        ax_nyquist.plot(
            Z_sim.real,
            -Z_sim.imag,
            marker="o",
            linestyle="-",
            markersize=4,
            label="Predicted EIS",
        )
        ax_nyquist.set_title(
            "Impedance Fit",
            fontsize=14,
            fontweight="bold",
            # family=''
        )
        ax_nyquist.legend(
            fontsize=12,
            loc="best",
            #   prop={'family': ''}
        )
        ax_nyquist.grid(True, alpha=0.3)

        ax_nyquist.set_xlabel(
            ax_nyquist.get_xlabel(),
            #   family=''
        )
        ax_nyquist.set_ylabel(
            ax_nyquist.get_ylabel(),
            #   family=''
        )
        for label in ax_nyquist.get_xticklabels() + ax_nyquist.get_yticklabels():
            # label.set_fontfamily('')/
            pass

        # Top-right: Circuit diagram
        ax_circuit = fig.add_subplot(gs[0, 1])
        circuit_img = Image.open(output_dir / f"temp_circuit_{i}.png")
        ax_circuit.imshow(circuit_img)
        ax_circuit.axis("off")
        stage_title = f"Stage {i+1}: {circuit}"
        if i == len(circuits) - 1:  # Last frame
            stage_title = f"Final Circuit: {circuit}"
        ax_circuit.set_title(
            stage_title,
            fontsize=13,
            fontweight="bold",
            pad=5,
            # family=''
        )

        # Bottom-right: Metrics text
        ax_metrics = fig.add_subplot(gs[1, 1])
        ax_metrics.axis("off")
        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_ylim(0, 1)

        # Format metrics with color-coding for pass/fail
        pass_color = "green" if passed_threshold else "red"
        pass_text = "PASS" if passed_threshold else "FAIL"

        # Create metrics text
        metrics_lines = [
            "Performance Metrics:",
            "",
            f"Status          : {pass_text}",
            "",
            f"R² Score        : {r2:.10f}",
            f"χ² (Chi-squared): {chi2:.10f}",
            f"R² Magnitude    : {r2_mag:.10f}",
            f"R² Phase        : {r2_phase:.10f}",
        ]

        # Draw each line individually for better control
        y_start = 0.85
        line_height = 0.10

        for idx, line in enumerate(metrics_lines):
            y_pos = y_start - (idx * line_height)

            # Color the status line
            if "Status:" in line:
                # Draw "Status: " in black
                ax_metrics.text(
                    0.15,
                    y_pos,
                    "Status: ",
                    fontsize=11,
                    va="top",
                    # family='',
                    fontweight="bold",
                )
                # Draw PASS/FAIL in color
                ax_metrics.text(
                    0.35,
                    y_pos,
                    pass_text,
                    fontsize=11,
                    va="top",
                    # family='',
                    fontweight="bold",
                    color=pass_color,
                )
            elif line == "Performance Metrics:":
                ax_metrics.text(
                    0.15,
                    y_pos,
                    line,
                    fontsize=12,
                    va="top",
                    fontweight="bold",
                    # family=''
                )
            elif line:  # Non-empty lines
                ax_metrics.text(
                    0.15,
                    y_pos,
                    line,
                    fontsize=11,
                    va="top",
                    # family=''
                )

        # Add subtle background box with visible border
        rect = Rectangle(
            (0.1, 0.05),
            0.8,
            0.9,
            linewidth=2,
            edgecolor="steelblue",
            facecolor="aliceblue",
            alpha=0.3,
            transform=ax_metrics.transAxes,
        )
        ax_metrics.add_patch(rect)

        # Save frame
        frame_path = output_dir / f"frame_{i:03d}.png"
        plt.savefig(
            frame_path,
            dpi=frame_dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.1,
            edgecolor="none",
        )

        # Save final frame to special folder
        if i == len(circuits) - 1:
            final_path = (
                output_dir / "final_circuits" / f"final_circuit_EIS_{eis_id}.png"
            )
            plt.savefig(
                final_path,
                dpi=frame_dpi,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.1,
                edgecolor="none",
            )
            print(f"    ✓ Final frame saved to: {final_path}")

        plt.close("all")

        print(
            f"    ✓ Frame saved (R²: {r2:.4f}, χ²: {chi2:.4f}, passed: {passed_threshold})"
        )

    print("\nCreating GIF...")

    # Create GIF with longer last frame
    frames = [
        Image.open(f) for f in sorted(glob.glob(str(output_dir / "frame_*.png")))
    ]
    durations = [gif_frame_duration] * len(frames)
    durations[-1] = gif_final_frame_duration  # Hold last frame longer

    gif_path = output_dir / f"circuit_evolution_EIS_{eis_id}.gif"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0
    )

    # Cleanup temp files if requested
    if cleanup_temp_files:
        for f in glob.glob(str(output_dir / "temp_circuit_*.png")):
            os.remove(f)
        for f in glob.glob(str(output_dir / "frame_*.png")):
            os.remove(f)
        print("✓ Cleaned up temporary files")

    print(f"✓ Saved {gif_path} ({len(frames)} frames)")
    print(f"✓ Final frame held for {durations[-1]/1000}s")



# Add this new method to the DDQN_ECM class (before create_circuit_evolution_visualization)

def prepare_and_generate_circuit_gif(
    _active_env,
    action_history,
    best_result,
    EIS_i,
    Z_true,
    save_dir
):
    """
    Extract circuit progression from action history and generate circuit evolution GIF.
    All metrics are taken directly from action_history (no recalculation).

    Parameters:
    -----------
    _active_env : EIS_ECM_Env
        Environment containing frequency data and circuit-fitting context.
    action_history : list
        List of dictionaries containing action history from evaluation
    best_result : dict
        Dictionary containing the best result information. Retained for API
        compatibility with evaluation callers.
    EIS_i : int
        EIS dataset index
    Z_true : array
        True impedance data
    save_dir : str or Path
        Base directory where generated visualizations are saved
    """
    # Extract circuit progression from action history
    circuit_progression = []
    circuit_predictions = []
    # avg_errors = []
    r2_mag_list = []
    r2_phase_list = []
    r2_scores = []
    chi2_scores = []
    passed_thresholds = []

    last_circuit = None

    print(f"\nPreparing circuit evolution data for EIS {EIS_i}...")

    for step in action_history:
        if step["validity"] and step["circuit"] is not None:
            # Only add if circuit changed
            if step["circuit"] != last_circuit:
                circuit = step["circuit"]
                circuit_progression.append(circuit)
                last_circuit = circuit

                # Extract metrics directly from action history (no calculation!)
                metrics = step.get("metrics", {})
                good_fit = step["good_fit"]

                # Generate Z_sim for plotting only
                try:
                    params = ae.utils.fit_circuit_parameters(
                        circuit, _active_env.freq, Z_true
                    )
                    circuit_fn = ae.utils.generate_circuit_fn(circuit)
                    Z_sim = circuit_fn(_active_env.freq, list(params.values()))

                    # Extract pre-calculated metrics from action_history
                    r2 = metrics.get("r2_score", 0.0)
                    chi2 = metrics.get("chi_square", 0.0)
                    r2_mag = metrics.get("r2_mag", 0.0)
                    r2_phase = metrics.get("r2_phase", 0.0)
                    # avg_error = metrics.get('avg_error', np.mean(np.abs(Z_true - Z_sim)))

                    # Store all data
                    circuit_predictions.append(Z_sim)
                    # avg_errors.append(avg_error)
                    r2_mag_list.append(r2_mag)
                    r2_phase_list.append(r2_phase)
                    r2_scores.append(r2)
                    chi2_scores.append(chi2)
                    passed_thresholds.append(
                        good_fit
                    )  # Use good_fit from action_history!

                except Exception as e:
                    print(f"    ⚠ Warning: Could not fit circuit {circuit}: {e}")
                    continue

    # Make sure we have at least the final circuit
    if not circuit_progression and best_result["circuit"] is not None:
        circuit = best_result["circuit"]
        circuit_progression = [circuit]

        # Get metrics from best_result
        try:
            params = ae.utils.fit_circuit_parameters(
                circuit, _active_env.freq, Z_true
            )
            circuit_fn = ae.utils.generate_circuit_fn(circuit)
            Z_sim = circuit_fn(_active_env.freq, list(params.values()))

            # Extract from best_result
            metrics = best_result.get("metrics", {})
            r2 = metrics.get("r2_score", 0.0)
            chi2 = metrics.get("chi_square", 0.0)

            # TODO add these into the gif
            r2_mag = metrics.get("r2_mag", 0.0)
            r2_phase = metrics.get("r2_phase", 0.0)
            # avg_error = metrics.get('avg_error', np.mean(np.abs(Z_true - Z_sim)))

            good_fit = best_result["found_solution"]

            circuit_predictions.append(Z_sim)
            # avg_errors.append(avg_error)
            r2_mag_list.append(r2_mag)
            r2_phase_list.append(r2_phase)
            r2_scores.append(r2)
            chi2_scores.append(chi2)
            passed_thresholds.append(good_fit)

        except Exception as e:
            print(f"    ⚠ Warning: Could not fit final circuit: {e}")

    # Generate the visualization if we have valid data
    if circuit_progression and len(circuit_progression) == len(circuit_predictions):
        try:
            create_circuit_evolution_visualization(
                freq=_active_env.freq,
                Z_true=Z_true,
                circuits=circuit_progression,
                circuit_predictions=circuit_predictions,
                # avg_errors=avg_errors,
                r2_scores=r2_scores,
                chi2_scores=chi2_scores,
                r2_mag_list=r2_mag_list,
                r2_phase_list=r2_phase_list,
                passed_thresholds=passed_thresholds,
                eis_id=EIS_i,
                output_dir=save_dir / "generated_circuits",
                frame_dpi=150,
                gif_frame_duration=2500,
                gif_final_frame_duration=5000,
                cleanup_temp_files=True,
                save_dir=save_dir
            )
            print(f"✓ Circuit evolution GIF generated for EIS {EIS_i}")
        except Exception as e:
            print(f"⚠ Warning: Could not generate GIF: {e}")
    else:
        print(
            f"⚠ Warning: Insufficient data for GIF generation (circuits: {len(circuit_progression)}, predictions: {len(circuit_predictions)})"
        )
