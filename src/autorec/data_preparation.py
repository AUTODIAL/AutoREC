from pathlib import Path
import pickle
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


import autoeis as ae
import impedance.validation
from impedance.validation import linKK
from autorec.utils import chi_obj_func

impedance.validation.circuit_elements["np"] = np

# Constant definition for threshold calculations
# TODO: Should I put these in init, so that they can be changed if someone wants to change them
DEFAULT_UPPER_BOUND_DECREASE = 0.997
DEFAULT_LOWER_BOUND_INCREASE = 2.5


class EISDataPrep:
    """
    EIS data preparation class for loading, processing, and validating datasets containing EIS
    (Electrochemical Impedance Spectroscopy) data required for training the RL agent using
    DDDN_ECM.

    Can either:
    1. Load raw CSV files from a folder and process them
       Process includes calculating the necessary values needed for EIS such as thresholds and
       flattening the frequency and Z values.
    2. Load and validate existing processed pickle/CSV files

    Expectations
    ------------
        The final dataset must have the following 6 columns with an optional column that's
        required only for evaluation.

        - sub_id: Identifier tracking which subfolder/file the data came from
        - true_circuit: Circuit representation as string (from folder name)
        - freq: Frequency values (array)
        - Z_true: Complex impedance values (array)
        - flatten_Z: Flattened EIS representation (array)
        - chi_thresh: Chi-square threshold value
        - r2_thresh: R-squared threshold value

        We expect each EIS to have the same number of values in their freq, Z_imag, Z_real
        columns.
        Eg. EIS 1: will have 80 data points and EIS n should have the same number of points as
        EIS 1.

    File Requirements
    -----------------
    1. Pickle files
        Pickle files will be loaded directly. The expectation will be that they have all the
        data, unless the process flag is on.
    2. CSV files
        Three columns should be provided: freq, Z_imag, Z_real with these exact column names.
        The CSV files should all be in a single folder.

        Optional:
        - The folder name will be used as the string for column: true_circuit, such that all
          the csv files in a folder are expected to have the same ECM. The name used for the
          circuit will be the direct parent of the file. So please be mindful of folder
          substructures. If you would like more descriptive folder names, or make
          subcategorizations, then the name of the folder will be split on the '_' character.
        Eg. if two folder paths exist:
        Raw_EIS/simple_circuits/RRR_1 and Raw_EIS/simple_circuits/RRR
        All the CSV files in both folders, will be classified as having their true_circuit as
        "RRR"

        - If the true_circuit isn't available, an arbitrary name can be used and the dataset
          won't be prepared for evaluation to help with specific and detailed tracking, the
          name of the file is used as the label for sub-id

    """

    REQUIRED_COLUMNS = [
        "sub_id",
        "freq",
        "Z_true",
        "flatten_Z",
        "chi_thresh",
        "r2_thresh",
        "metadata",
    ]

    EVAL_REQUIRED_COLUMNS = [
        "sub_id",
        "true_circuit",
        "freq",
        "Z_true",
        "flatten_Z",
        "chi_thresh",
        "r2_thresh",
    ]

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "load",
        evaluation: bool = False,
        # For lin-KK preprocessing
        perform_linKK_validation: bool = False,
        tol_linKK: float = 5e-2,
        # For frequency-based interpolation
        frequency_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        frequency_npoints: Optional[int] = None,
        # EIS features to include in the flatten_Z representation
        eis_features: Optional[list] = ["ImZ", "phi", "mag", "nphi"],
    ):
        """
        Initialize EISDataPrep, please ensure the provided data fits the expectations below.

        Parameters
        ----------
        path : str or Path
            Path to either:
            - A folder containing raw CSV files (mode='process')
            - A processed data file .pkl or .csv (mode='load')
        mode : str
            'process' - Load, process and validate raw CSV files from folder
            'load' - Load and validate existing processed file
        evaluation : bool
            If True, ground truth (true_circuit from folder names) is required for evaluation.
            In 'process' mode: enforces that folder structure provides circuit names
            In 'load' mode: validates that true_circuit column exists and is valid
        perform_linKK_validation : bool
            If True, perform lin-KK pre-processing on the EIS data to clean it up before
            further processing. This may change the length of the EIS data. The check on
            ``frequency_bounds`` and ``frequency_npoints`` will be done based on the
            lin-KK-validated data.
        tol_linKK : float
            Tolerance for linKK fitting when calculating thresholds (default: 5e-2)
        frequency_bounds : Tuple[Optional[float], Optional[float]]
            Lower and upper frequency bounds used for frequency interpolation. If either bound
            is None, it will use the min/max frequency from the lin-KK-validated data.
            The interpolated frequency grid is then generated between these bounds using
            ``frequency_npoints`` points.
        frequency_npoints : Optional[int]
            Number of points for frequency interpolation. If None, it will use the number of
            points in the provided frequency data, assuming all data has the same number of
            points. If inconsistent, it will raise an error.
        eis_features : list
            List of EIS features to include in the flatten_Z representation.
            Options include:

            * "ReZ": Real part of the normalized impedance.
            * "ImZ": Imaginary part of the normalized impedance.
            * "phi": Phase angle of the normalized impedance.
            * "mag": Magnitude of the normalized impedance.
            * "n<feature>": The negative counterpart of any of the above features
              (e.g., "nphi" for negative phase angle).

        Raises
        ------
        ValueError
            If path is None or mode is invalid
        FileNotFoundError
            If the specified path doesn't exist
        """

        if path is None:
            raise ValueError("EISDataPrep: Path to the dataset file cannot be None")
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"EISDataPrep: Path not found: {self.path}")

        if mode not in ["process", "load"]:
            raise ValueError(
                f"EISDataPrep: Invalid mode: '{mode}'. Must be 'process' or 'load'"
            )

        # For lin-KK pre-processing
        self.perform_linKK_validation = perform_linKK_validation
        self.tol_linKK = tol_linKK

        # For frequency interpolation
        self._validate_frequency_bounds(frequency_bounds)
        self._validate_frequency_npoints(frequency_npoints)
        self.frequency_bounds = frequency_bounds
        self.frequency_npoints = frequency_npoints

        avail_eis_features = ["ReZ", "ImZ", "phi", "mag", "nReZ", "nImZ", "nphi", "nmag"]
        if isinstance(eis_features, str):
            eis_features = [eis_features]
        for feature in eis_features:
            if feature not in avail_eis_features:
                raise ValueError(
                    f"Invalid EIS feature: '{feature}'. "
                    f"Available features: {avail_eis_features}"
                )
        self.eis_features = eis_features

        self.mode = mode
        self.evaluation = evaluation
        self.dataset = None
        self._validation_errors = []
        self._validation_warnings = []

        if mode == "load":
            # Determine file type if in load mode
            if not self.path.is_file():
                raise ValueError(f"In 'load' mode, path must be a file, got: {self.path}")

            suffix = self.path.suffix.lower()
            if suffix in [".pkl", ".pickle"]:
                self.file_type = "pickle"
            elif suffix == ".csv":
                self.file_type = "csv"
            else:
                raise ValueError(
                    f"Cannot auto-detect file type from extension '{suffix}'. "
                    f"Supported extensions: .pkl, .pickle, .csv"
                )
        else:
            if not self.path.is_dir():
                raise ValueError(f"In 'process' mode, path must be a folder, got: {self.path}")
            self.file_type = None

    def _validate_frequency_bounds(self, frequency_bounds):
        """
        Validate the frequency bounds parameters for interpolation.
        """
        if np.ndim(frequency_bounds) != 1 or len(frequency_bounds) != 2:
            raise ValueError(
                "frequency_bounds must be a tuple/list of two elements: "
                "(lower_bound, upper_bound)"
            )

    def _validate_frequency_npoints(self, frequency_npoints):
        """
        Validate the frequency_npoints parameter for interpolation.
        """
        if frequency_npoints is not None:
            if not isinstance(frequency_npoints, int) or frequency_npoints <= 2:
                raise ValueError("frequency_npoints must be an integer greater than 2.")

    def calculate_thresholds(
        self,
        freq: np.ndarray,
        Z_true: np.ndarray,
        upper_bound_decrease: float = DEFAULT_UPPER_BOUND_DECREASE,
        lower_bound_increase: float = DEFAULT_LOWER_BOUND_INCREASE,
    ) -> tuple:
        """
        Calculate chi_thresh and r2_thresh by fitting the circuit.
        Only runs in 'process' mode.

        Parameters
        ----------
        freq : np.ndarray
            Frequency array
        Z_true : np.ndarray
            Complex impedance array
        upper_bound_decrease : float
            Multiplier for r2_thresh (should be < 1.0)
        lower_bound_increase : float
            Multiplier for chi_thresh (should be > 1.0)

        Returns
        -------
        tuple
            (chi_thresh, r2_thresh) values
        """
        # Explicit check for process mode only
        if self.mode != "process":
            raise RuntimeError("calculate_thresholds should only be called in 'process' mode")

        for _ in range(3):
            linKK_kwargs = {"c": 0.4, "max_M": 100, "fit_type": "complex", "add_cap": True}
            linKK_silent = ae.utils.suppress_output_legacy(
                linKK,
            )
            _, _, Z_linKK, _, _ = linKK_silent(freq, Z_true, **linKK_kwargs)
        chi_kk = chi_obj_func(Z_true, Z_linKK)
        r2_kk = ae.metrics.r2_score(Z_true, Z_linKK)

        chi_thresh = chi_kk * lower_bound_increase
        r2_thresh = r2_kk * upper_bound_decrease

        return chi_thresh, r2_thresh

    @staticmethod
    def interpolate_EIS(freq, Z, freq_new):
        """
        Interpolate the EIS impedance data.

        Given a list of new frequency values, we interpolate the current impedance data for
        those new frequency values.

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
        New interpolated impedance values

        Usage
        -----
        >>> freq_new = np.logspace(*(np.log10(freq[[0, -1]])), npoints, endpoint=True)
        >>> Z_new = interpolate_eis(freq, Z, freq_new)
        """
        idx_sort = np.argsort(freq)
        idx_sort_new = np.argsort(freq_new)
        real_new = CubicSpline(freq[idx_sort], Z[idx_sort].real)(
            freq_new[idx_sort_new], extrapolate=False
        )
        imag_new = CubicSpline(freq[idx_sort], Z[idx_sort].imag)(
            freq_new[idx_sort_new], extrapolate=False
        )
        # Revert the sorting
        inv_idx_sort_new = np.argsort(idx_sort_new)
        return real_new[inv_idx_sort_new] + 1j * imag_new[inv_idx_sort_new]

    @staticmethod
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

    @staticmethod
    def flatten_EIS(experiment_Z, eis_features=["ImZ", "phi", "mag", "nphi"]):
        """
        Normalize each feature of EIS data and concatenate (flatten) the selected features.
        """
        Z, phi, _, mag = EISDataPrep.normalize_EIS(experiment_Z)
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
        flatten_Z = np.concatenate([all_values[f] for f in eis_features])
        return flatten_Z

    def process_single_EIS(
        self,
        freq: np.array,
        Z: np.array,
        csv_path: Path,
        base_path: Path,
    ) -> Optional[dict]:
        """
        Process a single EIS data, including:
        * Frequency interpolation to ensure all data of the same length
        * Computing EIS representation (flatten_Z)

        Parameters
        ----------
        freq : np.array
            Frequency values of the EIS data
        Z : np.array
            Complex impedance values of the EIS data
        csv_path : Path
            Path to the CSV file (used for sub_id and circuit extraction)
        base_path : Path
            Base path of the data directory

        Returns
        -------
        dict or None
            Dictionary with processed data, or None if error occurs
        """
        try:
            # Frequency interpolation
            lower_bound, upper_bound = self.frequency_bounds
            if lower_bound is None:
                lower_bound = np.min(freq)
            if upper_bound is None:
                upper_bound = np.max(freq)
            freq_new = np.logspace(
                np.log10(lower_bound),
                np.log10(upper_bound),
                self.frequency_npoints,
                endpoint=True,
            )
            Z_new = self.interpolate_EIS(freq, Z, freq_new)

            # Normalize and flatten
            flatten_Z = self.flatten_EIS(Z_new, self.eis_features)

            # Get circuit from folder name (not filename)
            circuit_string = self.get_circuit_from_folder(csv_path, base_path)

            # Create unique sub_id for tracking
            sub_id = self.create_sub_id(csv_path, base_path)

            # Calculate thresholds (only in process mode)
            # print(f"Processing {sub_id} (circuit: {circuit_string})...")
            chi_thresh, r2_thresh = self.calculate_thresholds(freq, Z)

            return {
                "sub_id": sub_id,
                "true_circuit": circuit_string,
                "freq": freq_new,
                "Z_true": Z_new,
                "flatten_Z": flatten_Z,
                "chi_thresh": chi_thresh,
                "r2_thresh": r2_thresh,
                "metadata": self.metadata,
            }

        except Exception as e:
            print(f"EISDataPrep: Error reading {csv_path}: {e}")
            return None

    def get_circuit_from_folder(self, csv_path: Path, base_path: Path) -> str:
        """
        Extract circuit string from the immediate parent folder name.

        Parameters
        ----------
        csv_path : Path
            Full path to the CSV file
        base_path : Path
            Base path of the data directory

        Returns
        -------
        str
            Circuit string from folder name
        """
        parent_folder = csv_path.parent  # Get the immediate parent folder

        # If the parent is the base path itself, use the filename
        if parent_folder == base_path:
            # No subfolder, extract from filename as fallback
            stem = csv_path.stem
            if "_" in stem:
                circuit_string = stem.rsplit("_", 1)[0]
            else:
                circuit_string = stem
        else:
            # Use the folder name as the circuit string
            circuit_string = parent_folder.name

        return circuit_string

    def create_sub_id(self, csv_path: Path, base_path: Path) -> str:
        """
        Create a unique sub_id for tracking which file/folder data came from.

        Format: "relative/path/to/file.csv"

        Parameters
        ----------
        csv_path : Path
            Full path to the CSV file
        base_path : Path
            Base path of the data directory

        Returns
        -------
        str
            Unique identifier for this data source
        """
        relative_path = csv_path.relative_to(base_path)
        return str(relative_path)

    def process_raw_data(self) -> pd.DataFrame:
        """
        Load and process all CSV files from the data folder recursively.
        Handles subfolders by extracting all CSVs and tracking with sub_id.

        Returns
        -------
        pd.DataFrame
            Processed dataset with all required columns
        """
        all_data = []
        self._preprocess_trace = {}

        # Recursively get all CSV files from all subfolders
        csv_files = list(self.path.rglob("*.csv"))
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in folder: {self.path}")
        self._preprocess_trace["csv_files"] = [str(f) for f in csv_files]
        print(f"Found {len(csv_files)} CSV files (including subfolders)")

        # Group files by folder for reporting
        folders = set(f.parent for f in csv_files)
        print(f"Across {len(folders)} folders/subfolders")

        # Load all the csv_files
        raw_eis_data = []
        for csv_file in csv_files:
            try:
                data = pd.read_csv(csv_file)
                raw_eis_data.append(data)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        self._preprocess_trace["raw_eis_data"] = raw_eis_data

        # First, we do lin-KK preprocessing to clean up the data.
        # **Note:** This operation can change the length of EIS data.
        linKK_preprocessed_data = []
        for eis_data in raw_eis_data:
            if self.perform_linKK_validation:
                freq, Z_true = self._perform_linKK_preprocessing(eis_data)
            else:
                freq = eis_data["freq"].values
                Z_true = (eis_data["Z_real"] + 1j * eis_data["Z_imag"]).values
            linKK_preprocessed_data.append((freq, Z_true))
        self._preprocess_trace["linKK_preprocessed_data"] = {
            "freq": [freq.tolist() for freq, _ in linKK_preprocessed_data],
            "Z_true": [Z.tolist() for _, Z in linKK_preprocessed_data],
        }

        # Validate the interpolation parameters based on the lin-KK preprocessed data
        self._validate_interpolation_parameters_after_linKK(linKK_preprocessed_data)

        # Process each EIS data
        # TODO: Add a progress bar
        for ii, (freq, Z_true) in enumerate(linKK_preprocessed_data):
            data_dict = self.process_single_EIS(freq, Z_true, csv_files[ii], self.path)
            if data_dict is not None:
                all_data.append(data_dict)

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        print(f"\nProcessed {len(df)} samples successfully")

        # Report unique circuits found
        if "true_circuit" in df.columns:
            unique_circuits = df["true_circuit"].nunique()
            print(f"Found {unique_circuits} unique circuit types")
            print(f"Circuits: {sorted(df['true_circuit'].unique())}")

        return df

    def _perform_linKK_preprocessing(
        self, eis_data: pd.DataFrame
    ) -> Tuple[np.array, np.array]:
        """Perform lin-KK preprocessing on the raw EIS data.

        Notes
        -----
        This calculation may change the length of the EIS data, so it should be done before
        any length validation.
        """
        # Extract frequency and impedance
        freq = eis_data["freq"].values
        Z_true = (eis_data["Z_real"] + 1j * eis_data["Z_imag"]).values

        # lin-KK pre-processing
        freq, Z_true = ae.utils.preprocess_impedance_data(
            freq, Z_true, tol_linKK=self.tol_linKK
        )
        return freq, Z_true

    def _validate_interpolation_parameters_after_linKK(self, linKK_preprocessed_data: list):
        """Validate the values of the interpolation parameters (frequency_bounds and
        frequency_npoints) based on the lin-KK preprocessed data.
        """
        # Validate frequency_npoints
        data_npoints = []
        for freq, _ in linKK_preprocessed_data:
            data_npoints.append(len(freq))
        if self.frequency_npoints is None:
            if len(set(data_npoints)) > 1:
                raise ValueError(
                    "Inconsistent frequency lengths across lin-KK preprocessed data. "
                    "Please specify 'frequency_npoints' for interpolation."
                )
            else:
                self.frequency_npoints = data_npoints[0]

        # Validate frequency_bounds
        for freq, _ in linKK_preprocessed_data:
            lower_bound, upper_bound = self.frequency_bounds
            msg = (
                "Cannot do extrapolation. "
                "Some frequency bounds fall outside the data range. "
                "Refer to the lin-KK-preprocessed data in "
                "self._preprocess_trace['linKK_preprocessed_data'] for min/max frequencies."
            )
            if lower_bound is not None and lower_bound < np.min(freq):
                raise ValueError(msg)
            if upper_bound is not None and upper_bound > np.max(freq):
                raise ValueError(msg)

    # ==================== LOADING EXISTING FILES ====================
    def _load_data(self) -> pd.DataFrame:
        """
        Internal method to load data based on file type.

        Returns
        -------
        pd.DataFrame
            Loaded pandas DataFrame
        """
        if self.file_type in ["pkl", "pickle"]:
            with open(self.path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, pd.DataFrame):
                raise ValueError(
                    f"EISDataPrep: Pickle file does not contain a pandas DataFrame. "
                    f"Found type: {type(data)}"
                )

            # Update metadata
            if "metadata" in data.columns:
                # Update attributes from metadata
                print("Metadata found in pickle file. Updating attributes accordingly.")
                metadata = data["metadata"].iloc[0]
                self.perform_linKK_validation = metadata.get("perform_linKK_validation", False)
                self.tol_linKK = metadata.get("tol_linKK", 5e-2)
                self.frequency_bounds = metadata.get("frequency_bounds", (None, None))
                self.frequency_npoints = metadata.get("frequency_npoints", None)
                self.eis_features = metadata.get("eis_features", ["ImZ", "phi", "mag", "nphi"])
            else:
                # If metadata is missing, then just leave it as None
                print("Warning: Pickle file does not contain metadata.")
                data["metadata"] = [None] * len(data)

            return data

        if self.file_type == "csv":
            # Load CSV - may need special handling for array columns
            return pd.read_csv(self.path)

        raise ValueError(f"Unsupported file type: {self.file_type}")

    # ==================== VALIDATION ====================
    def validate(self) -> bool:
        """
        Validate the loaded dataset structure.
        Checks required columns, data types, and circuit validity.

        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        self._validation_errors = []
        self._validation_warnings = []

        if self.dataset is None:
            self._validation_errors.append("Dataset is None. Load or process data first.")
            return False

        if not isinstance(self.dataset, pd.DataFrame):
            self._validation_errors.append(
                f"Dataset must be a pandas DataFrame, got {type(self.dataset)}"
            )
            return False

        # Check if DataFrame is empty
        if self.dataset.empty:
            self._validation_errors.append("Dataset is empty (0 rows)")
            return False

        # Determine which columns are required based on evaluation mode
        required_cols = (
            self.EVAL_REQUIRED_COLUMNS if self.evaluation else self.REQUIRED_COLUMNS
        )

        # Check for required columns
        missing_columns = []
        for col in required_cols:
            if col not in self.dataset.columns:
                missing_columns.append(col)

        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            if self.evaluation:
                error_msg += " (evaluation mode requires ground truth)"
            self._validation_errors.append(error_msg)

        # Check for extra columns (as warning only)
        extra_columns = [col for col in self.dataset.columns if col not in required_cols]
        if extra_columns:
            self._validation_warnings.append(
                f"Dataset contains extra columns (will be ignored): {', '.join(extra_columns)}"
            )

        # Validate column data types and structure (if all required columns present)
        if not missing_columns:
            self._validate_column_types()
            self._validate_circuits()  # New: validate circuit strings

        return len(self._validation_errors) == 0

    def _validate_circuits(self):
        """
        Validate circuit strings using ae.parser.validate_circuit().
        Adds warnings for invalid circuit formats.
        """
        if "true_circuit" not in self.dataset.columns:
            return

        invalid_circuits = []
        circuit_errors = {}

        # Check each unique circuit
        unique_circuits = self.dataset["true_circuit"].unique()

        for circuit in unique_circuits:
            try:
                # Try to validate the circuit
                ae.parser.validate_circuit(circuit)
            except Exception as e:
                invalid_circuits.append(circuit)
                circuit_errors[circuit] = str(e)

        if invalid_circuits:
            self._validation_warnings.append(
                f"Invalid circuit formats found: {len(invalid_circuits)} circuits"
            )
            for circuit in invalid_circuits[:5]:  # Show first 5
                error_msg = circuit_errors.get(circuit, "Unknown error")
                self._validation_warnings.append(f"  - '{circuit}': {error_msg}")
            if len(invalid_circuits) > 5:
                self._validation_warnings.append(
                    f"  ... and {len(invalid_circuits) - 5} more invalid circuits"
                )

    def _validate_column_types(self):
        """
        Validate that each column contains the expected data type/structure.
        """

        def check_array_column(col_name):
            """Check if column contains array-like data."""
            if col_name in self.dataset.columns:
                sample_val = self.dataset.iloc[0][col_name]
                if not isinstance(sample_val, (np.ndarray, list)):
                    self._validation_errors.append(
                        f"Column '{col_name}' should contain arrays, "
                        f"found {type(sample_val)} in row 0"
                    )

        def check_string_column(col_name):
            """Check if column contains string data."""
            if col_name in self.dataset.columns:
                if not pd.api.types.is_string_dtype(self.dataset[col_name]):
                    self._validation_warnings.append(
                        f"Column '{col_name}' should contain string values"
                    )

        def check_numeric_column(col_name):
            """Check if column contains numeric data (including numpy/JAX scalars)."""
            if col_name in self.dataset.columns:
                # Check if values are numeric-like (including numpy/JAX types)
                sample_val = self.dataset.iloc[0][col_name]
                try:
                    # Try to convert to float - if it works, it's numeric enough
                    float(sample_val)
                except (ValueError, TypeError):
                    self._validation_errors.append(
                        f"Column '{col_name}' should contain numeric values, "
                        f"found {type(sample_val)} in row 0"
                    )

        # Validate array columns
        for col in ["Z_true", "freq", "flatten_Z"]:
            check_array_column(col)

        # Validate string columns
        for col in ["true_circuit", "sub_id"]:
            check_string_column(col)

        # Validate numeric columns
        for col in ["chi_thresh", "r2_thresh"]:
            check_numeric_column(col)

    # TODO: double check if it works fine
    def _validate_EIS_len(self):
        """
        Validate that all EIS samples have the same number of data points.

        Checks two things:
        1. Within each sample: freq and Z_true have matching lengths
        2. Across all samples: all have the same number of data points
        """
        if self.dataset is None or len(self.dataset) == 0:
            return

        # Check columns exist
        array_cols = ["freq", "Z_true", "flatten_Z"]
        missing = [col for col in array_cols if col not in self.dataset.columns]
        if missing:
            return  # Will be caught by other validation

        # First: Check within-sample consistency (freq and Z_true must match for each sample)
        within_sample_errors = []
        for idx, row in self.dataset.iterrows():
            freq_len = len(row["freq"])
            Z_len = len(row["Z_true"])

            if freq_len != Z_len:
                sub_id = row.get("sub_id", f"row {idx}")
                within_sample_errors.append(
                    f"{sub_id}: freq has {freq_len} points but Z_true has {Z_len} points"
                )

        if within_sample_errors:
            self._validation_errors.append(
                f"Found {len(within_sample_errors)} samples where freq and Z_true lengths "
                "don't match. Each sample must have equal freq and impedance points."
            )
            for error in within_sample_errors[:3]:
                self._validation_errors.append(f"  - {error}")
            if len(within_sample_errors) > 3:
                self._validation_errors.append(
                    f"  ... and {len(within_sample_errors) - 3} more mismatches"
                )
            return  # Don't proceed to cross-sample validation if within-sample is broken

        # Second: Check cross-sample consistency (all samples have same dimensions)
        first_freq_len = len(self.dataset.iloc[0]["freq"])
        first_flatten_len = len(self.dataset.iloc[0]["flatten_Z"])

        # Track inconsistencies across samples
        freq_mismatches = []
        flatten_mismatches = []

        # Check all samples (skip first since it's the reference)
        for idx, row in self.dataset.iloc[1:].iterrows():
            current_freq_len = len(row["freq"])
            current_flatten_len = len(row["flatten_Z"])

            # Check freq length (Z_true will match freq due to within-sample check)
            if current_freq_len != first_freq_len:
                sub_id = row.get("sub_id", f"row {idx}")
                freq_mismatches.append(
                    f"{sub_id}: {current_freq_len} points (expected {first_freq_len})"
                )

            # Check flatten_Z length
            if current_flatten_len != first_flatten_len:
                sub_id = row.get("sub_id", f"row {idx}")
                flatten_mismatches.append(
                    f"{sub_id}: {current_flatten_len} points (expected {first_flatten_len})"
                )

        # Report errors for inconsistent dimensions across samples
        if freq_mismatches:
            self._validation_errors.append(
                f"Inconsistent array lengths across {len(freq_mismatches)} samples. "
                f"All samples must have the same number of data points (freq/Z_true)."
            )
            for mismatch in freq_mismatches[:3]:
                self._validation_errors.append(f"  - {mismatch}")
            if len(freq_mismatches) > 3:
                self._validation_errors.append(
                    f"  ... and {len(freq_mismatches) - 3} more mismatches"
                )

        # Report warnings for flatten_Z (should be consistent but derived)
        if flatten_mismatches:
            self._validation_warnings.append(
                f"Inconsistent flatten_Z lengths across {len(flatten_mismatches)} samples"
            )
            for mismatch in flatten_mismatches[:3]:
                self._validation_warnings.append(f"  - {mismatch}")
            if len(flatten_mismatches) > 3:
                self._validation_warnings.append(
                    f"  ... and {len(flatten_mismatches) - 3} more mismatches"
                )

    def _format_validation_errors(self) -> str:
        """
        Format validation errors into a readable error message.

        Returns
        -------
        str
            Formatted error message
        """
        error_msg = f"\n{'=' * 60}\n"
        error_msg += "EISDataPrep VALIDATION FAILED\n"
        error_msg += f"{'=' * 60}\n\n"
        error_msg += f"Path: {self.path}\n"
        error_msg += f"Mode: {self.mode}\n"
        error_msg += f"Evaluation Mode: {self.evaluation}\n\n"

        if self.dataset is not None:
            error_msg += f"Found Columns: {list(self.dataset.columns)}\n\n"

        error_msg += "Errors:\n"
        for i, error in enumerate(self._validation_errors, 1):
            error_msg += f"  {i}. {error}\n"

        required_cols = (
            self.EVAL_REQUIRED_COLUMNS if self.evaluation else self.REQUIRED_COLUMNS
        )

        error_msg += f"\n{'=' * 60}\n"
        error_msg += "Required Columns:\n"
        for col in required_cols:
            status = "✓" if self.dataset is not None and col in self.dataset.columns else "✗"
            error_msg += f"  {status} {col}\n"

        return error_msg + f"{'=' * 60}\n"

    def load(self) -> pd.DataFrame:
        """
        Load/process data and validate.

        Returns
        -------
        pd.DataFrame
            Validated dataset

        Raises
        ------
        ValueError
            If validation fails
        """
        try:
            if self.mode == "process":
                self.dataset = self.process_raw_data()
            else:
                self.dataset = self._load_data()
        except Exception as e:
            raise ValueError(f"Failed to load/process data: {str(e)}")

        # Validate before dropping any columns
        is_valid = self.validate()

        if not is_valid:
            error_message = self._format_validation_errors()
            raise ValueError(error_message)

        # Show warnings if any
        if self._validation_warnings:
            print("\nValidation Warnings:")
            for warning in self._validation_warnings:
                print(f"  ⚠ {warning}")

        print(f"\n✓ Validation passed ({len(self.dataset)} samples)")

        return self.dataset

    # ==================== UTILITY METHODS ====================
    def get_summary(self) -> None:
        """Print summary of the loaded/processed dataset."""
        print("=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        print(f"Mode: {self.mode}")
        print(f"Evaluation Mode: {self.evaluation}")
        print(f"Path: {self.path}")
        print(f"Total samples: {len(self.dataset) if self.dataset is not None else 0}")

        if self.dataset is not None:
            print(f"\nColumns: {list(self.dataset.columns)}")
            print(f"\nData types:\n{self.dataset.dtypes}")
            print(f"\nShape: {self.dataset.shape}")

            # Show circuit distribution
            if "true_circuit" in self.dataset.columns:
                circuit_counts = self.dataset["true_circuit"].value_counts()
                print("\nCircuit distribution:")
                print(circuit_counts)

            # Show sub_id info
            if "sub_id" in self.dataset.columns:
                print(f"\nUnique files/subfolders: {self.dataset['sub_id'].nunique()}")

            if len(self.dataset) > 0:
                print(f"\nSample flatten_Z length: {len(self.dataset.iloc[0]['flatten_Z'])}")
                print(f"Sample freq length: {len(self.dataset.iloc[0]['freq'])}")

            print("\n" + "=" * 80)
            print("FIRST 5 ROWS (metadata only)")
            print("=" * 80)
            # Show just the metadata columns for clarity
            metadata_cols = [
                col
                for col in ["true_circuit", "sub_id", "chi_thresh", "r2_thresh"]
                if col in self.dataset.columns
            ]
            print(self.dataset[metadata_cols].head())

    def get_validation_summary(self) -> dict:
        """
        Get a summary of the validation results.

        Returns
        -------
        dict
            Dictionary containing validation status, errors, and warnings
        """
        return {
            "is_valid": len(self._validation_errors) == 0,
            "errors": self._validation_errors.copy(),
            "warnings": self._validation_warnings.copy(),
            "path": str(self.path),
            "mode": self.mode,
            "evaluation": self.evaluation,
            "num_rows": len(self.dataset) if self.dataset is not None else 0,
            "columns_found": list(self.dataset.columns) if self.dataset is not None else [],
            "required_columns": (
                self.EVAL_REQUIRED_COLUMNS if self.evaluation else self.REQUIRED_COLUMNS
            ).copy(),
        }

    def save(self, output_path: Union[str, Path], file_type: str = "pickle"):
        """
        Save the processed dataset.

        Parameters
        ----------
        output_path : str or Path
            Path where to save the file
        file_type : str
            'pickle' or 'csv'
        """
        if self.dataset is None:
            raise ValueError("No dataset to save. Load or process data first.")

        output_path = Path(output_path)

        if file_type == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(self.dataset, f)
            print(f"Dataset saved to {output_path}")
        elif file_type == "csv":
            self.dataset.to_csv(output_path, index=False)
            print(f"Dataset saved to {output_path}")
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

    @property
    def metadata(self) -> dict:
        """Get metadata about the data preparation process."""
        return {
            "path": str(self.path),
            "mode": self.mode,
            "evaluation": self.evaluation,
            "perform_linKK_validation": self.perform_linKK_validation,
            "tol_linKK": self.tol_linKK,
            "frequency_bounds": self.frequency_bounds,
            "frequency_npoints": self.frequency_npoints,
            "eis_features": self.eis_features,
        }


# ==================== USAGE EXAMPLES ====================
if __name__ == "__main__":
    # Example 1: Process raw CSV files from a folder (with subfolders)
    print("=" * 80)
    print("EXAMPLE 1: Processing raw CSV files (recursive)")
    print("=" * 80)

    prep1 = EISDataPrep(
        path="./EIS_raw/",
        mode="process",
        evaluation=False,  # Training mode
    )

    dataset1 = prep1.load()
    prep1.get_summary()

    # Save the processed data
    prep1.save("processed_training_data.pkl")

    # print("\n" + "="*80)
    # print("EXAMPLE 2: Processing for evaluation (requires ground truth)")
    # print("="*80)

    # prep2 = EISDataPrep(
    #     path="/path/to/eval/folder",
    #     mode='process',
    #     evaluation=True  # Evaluation mode - enforces circuit validation
    # )

    # dataset2 = prep2.load()
    # prep2.get_summary()

    # print("\n" + "="*80)
    # print("EXAMPLE 3: Loading existing processed file")
    # print("="*80)

    # prep3 = EISDataPrep(
    #     path="processed_training_data.pkl",
    #     mode='load',
    #     evaluation=False
    # )

    # dataset3 = prep3.load()
    # prep3.get_summary()

    # Get validation details
    validation_summary = prep1.get_validation_summary()
    print("\nValidation Summary:")
    print(f"Valid: {validation_summary['is_valid']}")
    print(f"Errors: {validation_summary['errors']}")
    print(f"Warnings: {validation_summary['warnings']}")
