import os

# Limit numpy/scipy threading BEFORE importing them
thread_count = "4"
thread_count = "1"
os.environ["OMP_NUM_THREADS"] = thread_count
os.environ["MKL_NUM_THREADS"] = thread_count
os.environ["OPENBLAS_NUM_THREADS"] = thread_count
os.environ["NUMEXPR_NUM_THREADS"] = thread_count


import autoeis as ae

import numpy as np
import pandas as pd


from autorec.utils import (
    state_encode,
    fit_circuit_parameters_NEW,
    parse_state_to_circuit,
    action_validity,
)
from autorec.optimized_data_structures.circuit_cache import HistoryCache

ec = ae.core.ec


class EIS_ECM_Env:
    """
    Environment for AutoREC: EIS Equivalent Circuit Modeling with Reinforcement Learning.

    This class implements a reinforcement learning environment for discovering equivalent
    circuit models that fit EIS data. The agent learns to construct circuit topologies by
    mutating a gene expression programming (GEP) chromosome representation.

    Key Concepts
    ------------
    - GEP Chromosome: A string representation of circuit topology (e.g., '+RPRRRR')
    - Head: First part of chromosome that can contain any element (+, /, R, L, P, C)
    - Tail: Second part that can only contain terminal elements (R, L, P, C)
    - Actions: Mutations that change one position in the chromosome to a different element
    - Episode: One complete attempt to find a circuit that fits a given EIS measurement

    Workflow
    --------
    1. Environment is reset with a random starting circuit and EIS measurement
    2. Agent takes actions to mutate the circuit chromosome
    3. Each mutation is evaluated by fitting circuit parameters to the EIS data
    4. Rewards are given based on fit quality (R², chi-square test)
    5. Episode terminates when a good fit is found or agent gives up

    Attributes
    ----------
    _N_MAX : int
        Maximum number of child nodes per circuit element (binary tree = 2)
    _ELEMENTS : list
        Available circuit elements: ['+', '/', 'R', 'L', 'P']
        '+' = series connection, '/' = parallel connection
        'R' = resistor, 'L' = inductor, 'P' = constant phase element
    ELEMENTS_EXTENDED : list
        Elements plus 'X' for encoding (used in state representation)
    """

    # Constant declaration
    _N_MAX = 2
    ELEMENTS = ["+", "/", "R", "L", "P"]
    # ELEMENTS = ['+', '/', 'R', 'L', 'P', 'C']
    ELEMENTS_EXTENDED = ELEMENTS + ["X"]

    def __init__(
        self,
        dataset: pd.DataFrame = None,
        initial_state: list[str] = ["+RRRRRR", "++RRRRRR", "+++RRRRRR"],
        seed: int = 42,
        chromosome_HEAD_len: int = 10,
        # Cache configuration
        cache_enabled: bool = True,
        cache_capacity: int = 20000,
        cache_type: str = "lru",  # 'lru' or 'clock',
    ):
        """
        Initialize the EIS circuit modeling environment.

        This sets up the reinforcement learning environment with a dataset of EIS
        measurements, configures the chromosome (circuit representation) parameters.

        Parameters
        ----------
        dataset : pd.DataFrame, required
            A DataFrame where each row contains:
            - 'Z_true': Complex impedance measurements (numpy array)
            - 'freq': Frequency points for measurements (numpy array)
            - 'flatten_Z': Flattened real/imaginary impedance values
            - 'true_circuit': The actual circuit that generated the data (for reference)
            - 'chi_thresh': Chi-square threshold for determining good fit
            - 'r2_thresh': R² threshold for determining good fit

        seed : int, default=42
            Random seed for reproducibility. Controls:
            - Which starting circuit is chosen
            - Which EIS measurement is sampled during reset

        chromosome_HEAD_len : int, default=10
            Length of the chromosome "head" region. This determines:
            - How complex circuits can be (longer head = more complex circuits possible)
            - Total chromosome length via formula: total = head + tail
            - Tail length is calculated as: head * (N_MAX - 1) + 1
            Example: head=10 gives tail=11, total length=21

        cache_enabled : bool, default=True
            Whether to enable caching of circuit evaluations. When enabled:
            - Previously evaluated circuits are stored in memory
            - Repeated evaluations return cached results instantly
            - Dramatically speeds up training (10-100x faster)
            Disable only for debugging or if memory is constrained

        cache_capacity : int, default=10000
            Maximum number of circuit evaluations to store in cache.
            Higher values:
            - Use more memory (~1-5 KB per cached result)
            - Improve hit rate for long training runs
            Typical values: 5000-50000 depending on available RAM

        cache_type : str, default='lru'
            Cache eviction policy. Options:
            - 'lru': Least Recently Used (recommended for most cases)
            - 'clock': Clock/second-chance algorithm (more efficient for certain patterns)

        Raises
        ------
        ValueError
            If dataset is None (a dataset must always be provided)
        """
        if dataset is None:
            raise ValueError(
                "EIS_ECM_ENV: A dataset must be provided to create the environment. "
                "Please use the EISDataPrep class to you have not already created "
                "the dataset with data_preparation.py"
            )
        self.dataset = dataset
        self.seed = seed

        np.random.seed(self.seed)

        self._all_start_states = initial_state
        self.EIS_measurement_size = len(dataset.loc[0, "Z_true"])
        self.EIS_INPUT_SIZE = len(dataset.loc[0, "flatten_Z"])
        self.chromosome_HEAD_len = chromosome_HEAD_len
        self.chromosome_TAIL_len = self.chromosome_HEAD_len * (self._N_MAX - 1) + 1
        self.chromosome_len = self.chromosome_HEAD_len + self.chromosome_TAIL_len

        # Initialize cache
        self.cache_enabled = cache_enabled
        if self.cache_enabled:
            self.cache = HistoryCache(capacity=cache_capacity, cache_type=cache_type)
            print(
                f"✓ Circuit evaluation cache enabled: {cache_type.upper()} "
                f"with capacity {cache_capacity:,}"
            )
        else:
            self.cache = None
            print("Circuit evaluation cache disabled")

        def non_coding_to_R(state, HEAD, TAIL):  # Removed the ec from here.
            """
            Replace non-coding region of chromosome with resistors.

            In GEP, not all positions in the chromosome are "expressed" (used in the circuit).
            The expressed portion depends on the tree structure created by operators (+, /).
            This function replaces the unused "non-coding" tail with 'R' (resistors) to
            standardize the representation.
            """
            karva = state.replace("/", "-")
            tree = ec.karva_to_tree(karva)

            coding_length = len(tree)
            non_coding_length = HEAD + TAIL - coding_length
            replacing_element = "R" * non_coding_length
            return state[0:coding_length] + replacing_element  # The state

        self.start_state_list = [
            non_coding_to_R(rnd_state, self.chromosome_HEAD_len, self.chromosome_TAIL_len)
            for rnd_state in self._all_start_states
        ]
        self.ACTIONS_LIST = self.all_actions_list()

        # Track current EIS index
        self.current_eis_index = None
        self.encoded_state = None
        self.reset()

    def all_actions_list(self):
        """
        Generate a complete list of all valid actions the agent can take.

        An "action" is defined as changing one position in the chromosome to a specific
        element. Not all combinations are valid due to GEP structural rules:

        Rules
        -----
        1. Position 0 (root) must be an operator ('+' or '/') - it connects the circuit
        2. Head positions (0 to HEAD-1) can be any element: +, /, R, L, P
        3. Tail positions (HEAD to end) can only be terminals: R, L, P

        This enforces proper tree structure where operators appear before terminals.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'action_type': The element to place ('+', '/', 'R', 'L', 'P')
            - 'action_position': The chromosome position to modify (0 to length-1)
        """
        all_actions = pd.DataFrame(columns=["action_type", "action_position"])

        for pos in range(self.chromosome_HEAD_len + self.chromosome_TAIL_len):
            if pos < self.chromosome_HEAD_len:
                for item in self.ELEMENTS:
                    if pos == 0 and (item not in ["+", "/"]):
                        continue
                    else:
                        all_actions = pd.concat(
                            [
                                all_actions,
                                pd.DataFrame(
                                    {"action_type": f"{item}", "action_position": [pos]}
                                ),
                            ],
                            ignore_index=True,
                        )
            else:
                for item in self.ELEMENTS[2:]:
                    all_actions = pd.concat(
                        [
                            all_actions,
                            pd.DataFrame({"action_type": f"{item}", "action_position": [pos]}),
                        ],
                        ignore_index=True,
                    )

        all_actions = all_actions.reset_index(drop=True)

        return all_actions

    def reset(self, EIS_i=None):
        """
        Reset the environment to start a new episode.

        This method:
        1. Selects a random starting circuit from predefined templates
        2. Samples a new EIS measurement to fit (or uses specified one)
        3. Returns the initial observation

        This is called at the beginning of each training episode. Each episode represents one
        attempt to discover a circuit that fits a particular EIS measurement.

        Parameters
        ----------
        EIS_i : int or None, default=None
            If provided, uses the EIS measurement at this index in the dataset
            If None, randomly samples an EIS measurement from the dataset

        Returns
        -------
        tuple[int or pd.Index, dict]
            Tuple containing:
            - index: The index/row number of the selected EIS measurement.
            - observation: Initial state observation (see ``_get_obs()`` for details).
        """
        start_state = self.start_state_list[np.random.randint(0, len(self.start_state_list))]
        self.START_STATE = start_state

        # self.state: str = start_state
        # self.encoded_state = state_encode(self.state, self.ELEMENTS_EXTENDED)
        self._update_state(new_state=start_state)

        index = self._set_EIS(EIS_i)

        observation = self._get_obs()

        return index, observation

    def _update_state(
        self, action_type: str = None, action_position: int = None, new_state: str = None
    ):
        """
        Update the circuit state and get the encoded state.

        If called during reset, it gives the encoded state of the START_STATE.

        Examples
        --------
        # For mutations (during episodes):
        self._update_state(action_type='R', action_position=3)

        # For complete replacement (during reset):
        self._update_state(new_state='START_STATE)

        Parameters
        ----------
        action_type : str, optional
            The element to place: '+', '/', 'R', 'L', or 'P'
        action_position : int, optional
            The chromosome position to modify (0 to chromosome_length-1)
        new_state : str, optional
            Complete new chromosome string to set

        Raises
        ------
        ValueError
            If neither mutation parameters nor new_state is provided,
            or if both are provided simultaneously
        """
        if new_state is not None:
            # Complete replacement mode
            if action_type is not None or action_position is not None:
                raise ValueError(
                    "Cannot specify both new_state and mutation parameters "
                    "(action_type/action_position)"
                )
            self.state = new_state
        elif action_type is not None and action_position is not None:
            # Mutation mode
            self.state = (
                self.state[:action_position] + action_type + self.state[action_position + 1 :]
            )
        else:
            raise ValueError(
                "Must provide either new_state OR both action_type and action_position"
            )

        # Always update encoded state after any modification
        self.encoded_state = state_encode(self.state, self.ELEMENTS_EXTENDED)

    def _set_EIS(self, EIS_i):
        """
        Load an EIS measurement from the dataset for the current episode.

        Each time that the env is reset a new EIS will be sampled from the data set
        This method extracts all necessary data for one EIS measurement:
        - Complex impedance values (what we're trying to fit)
        - Frequency points (where measurements were taken)
        - Ground truth circuit (for evaluation/comparison)
        - Fit quality thresholds (when to consider fit "good enough")
        - Normalized data (for neural network input)

        Parameters
        ----------
        EIS_i : int or None
            If int: Load the EIS measurement at this specific row index
            If None: Randomly sample an EIS measurement from the dataset

        Returns
        -------
        int or pd.Index
            The index of the loaded EIS measurement
        """
        if EIS_i is None:
            # Random sampling returns a DataFrame
            sample = self.dataset.sample(n=1)
            self.current_eis_index = sample.index[0]
            row = sample.iloc[0]
        else:
            # iloc with integer returns a Series - access values directly
            row = self.dataset.iloc[EIS_i]
            self.current_eis_index = EIS_i

        self.Z = row["Z_true"]
        self.freq = row["freq"]
        self.Z_norm_flatten = row["flatten_Z"]
        self.true_circuit = row["true_circuit"]
        self.chi_threshold = row["chi_thresh"]
        self.r2_threshold = row["r2_thresh"]
        return self.current_eis_index

    def _get_obs(self):
        """
        Get the current observation of the environment state.

        This method packages all the information the agent needs to make decisions:
        - Current circuit representation (chromosome)
        - EIS measurement to fit (impedance data)
        - Parsed circuit in human-readable form

        The observation is what the agent's neural network receives as input.

        Returns
        -------
        dict
            Dictionary containing:

            'state' : str
                The current chromosome string (e.g., '+RPRRRR')
                This is the raw genetic representation of the circuit

            'encoded_state' : np.ndarray
                One-hot encoded version of the state for neural network input
                Shape: (chromosome_length, num_elements)
                Each position becomes a vector like [0,0,1,0,0,0] indicating which element

            'EIS_flatten' : np.ndarray
                The target EIS measurement to fit, flattened to 1D
                Contains normalized [real_part, imaginary_part] impedance values
                Shape: (2 * num_frequency_points,)

            'circuit' : str
                Human-readable circuit notation (e.g., 'R0-p(R1,P2)')
                This is what the chromosome represents when converted to an actual circuit
                Useful for visualization and understanding
        """
        state_karva = self.state.replace("/", "-")
        tree = ec.karva_to_tree(state_karva)
        return {
            "state": self.state,
            "encoded_state": self.encoded_state,
            "EIS_flatten": self.Z_norm_flatten,
            "circuit": ec.tree_to_circuit(tree)[0],
            "Z_true": self.Z,
            "freq": self.freq,
        }

    # Extract the circuit fitting and reward calculation to separate methods
    def _evaluate_circuit(self, circuit: str, components_number: int) -> dict:
        """
        Evaluate circuit by fitting to EIS data and calculating reward.

        Returns
        -------
        dict
            Dictionary with keys: reward, metrics, predicted_Z, param, terminated, and
            depth_penalty.
        """
        try:
            # Fit circuit parameters to EIS data
            param, chi_square, r2_score, r2_mag, r2_phase = fit_circuit_parameters_NEW(
                circuit,
                self.freq,
                self.Z,
                min_iters=10,
                max_iters=30,
                method="log-B",
                tol_chi_squared=self.chi_threshold,
            )

            # Predict impedance with fitted parameters
            param_list = [values for key, values in param.items()]
            np_param = np.array(param_list)
            predicted_Z = ae.utils.eval_circuit(circuit, self.freq, np_param)

            # Store metrics
            metrics = {
                "chi_square": chi_square,
                "r2_score": r2_score,
                "r2_mag": r2_mag,
                "r2_phase": r2_phase,
            }

            # Calculate reward and determine if terminated
            reward, terminated, depth_penalty = self._calculate_reward(
                r2_score, r2_mag, r2_phase, chi_square, circuit, components_number
            )

            return {
                "reward": reward,
                "metrics": metrics,
                "predicted_Z": predicted_Z,
                "param": param,
                "terminated": terminated,
                "depth_penalty": depth_penalty,
            }

        except Exception as e:
            # Catch ALL exceptions from fitting or reward calculation
            print(f"Error in circuit evaluation: {e}")
            return {
                "reward": -0.05,
                "metrics": None,
                "predicted_Z": None,
                "param": {},
                "terminated": False,
                "depth_penalty": 0,
            }

    def _calculate_reward(
        self,
        r2_score: float,
        r2_mag: float,
        r2_phase: float,
        chi_square: float,
        circuit: str,
        components_count: int,
    ) -> tuple[float, bool, float]:
        """
        Calculate reward based on circuit fit quality.

        Returns
        -------
        tuple[float, bool, float]
            Tuple of ``(reward, terminated, depth_penalty)``.
        """
        depth_penalty = 0

        # Complete fitting failure
        if r2_score == 0:
            return -0.05, False, depth_penalty

        # GOOD FIT!
        if chi_square < self.chi_threshold:
            # Component penalty
            components_penalty_list = [0, 0, 0, 0, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
            components_penalty = (
                2 if components_count >= 10 else components_penalty_list[components_count]
            )

            # Depth penalty
            nested_expr = ae.parser.circuit_to_nested_expr(circuit)

            def _find_max_depth(nested_list) -> int:
                """Recursively find maximum nesting depth."""
                if isinstance(nested_list, list):
                    return 1 + max(_find_max_depth(item) for item in nested_list)
                return 0

            max_depth = _find_max_depth(nested_expr) - 1
            if max_depth >= 3:
                depth_penalty = 0.2

            # Calculate reward
            r2_average = (r2_score + r2_mag + r2_phase) / 3
            chi_term = np.log10(max(1 / chi_square, 1e-10))
            r2_term = np.log10(max(1 / (1 - r2_average), 1e-10))
            reward = chi_term + r2_term - components_penalty - depth_penalty

            return reward, True, depth_penalty  # Terminated!

        # Moderate fit
        if r2_score > 0.5 and r2_mag > 0.5 and r2_phase > 0.5:
            return 0.01, False, depth_penalty

        # Poor fit
        if r2_score > 0 and r2_mag > 0 and r2_phase > 0:
            return -0.01, False, depth_penalty

        # Very poor fit
        return -0.02, False, depth_penalty

    def step(self, action_type, action_position):
        """
        Execute one action in the environment and return the results.

        This is the core of the RL environment. It:
        1. Validates if the action is legal
        2. If valid, mutates the circuit chromosome
        3. Fits circuit parameters to the EIS data (expensive!)
        4. Calculates reward based on fit quality
        5. Determines if episode should terminate (good fit found)

        The step function uses caching to dramatically speed up evaluations. If the exact
        same circuit has been evaluated for this EIS measurement before, it returns the
        cached result instantly (~1000x faster).

        Parameters
        ----------
        action_type : str
            The element to place: '+', '/', 'R', 'L', or 'P'
            '+' = series connection
            '/' = parallel connection
            'R' = resistor
            'L' = inductor
            'P' = constant phase element (CPE)

        action_position : int
            Which chromosome position to modify (0 to chromosome_length-1)

        Returns
        -------
        dict
            Dictionary containing:

            'observation' : dict
                Current state after taking the action (see _get_obs())

            'info' : dict
                Additional info (same as observation)

            'reward' : float
                Reward for this action. Scale varies:
                - Invalid action: -0.5 (penalty for breaking rules)
                - Failed fit: -0.05 to -0.02 (circuit couldn't be evaluated)
                - Poor fit: -0.01 to 0.01 (circuit doesn't match data well)
                - Good fit: ~0.5 to 5.0 (circuit matches data, episode terminates)

            'terminated' : bool
                True if good fit found (chi² < threshold), False otherwise
                When True, episode ends - agent succeeded!

            'validity' : int
                1 if action was valid, 0 if invalid (broke GEP rules)

            'coding' : str
                The "expressed" part of chromosome (excludes non-coding tail)
                This is what actually forms the circuit

            'param' : dict
                Fitted circuit parameters (e.g., {'R0': 100, 'P1-n': 0.85})
                Empty dict if fitting failed

            'circuit' : str
                Human-readable circuit notation (e.g., 'R0-p(R1,CPE2)')

            'predicted_Z' : np.ndarray or None
                Impedance predicted by fitted circuit
                None if fitting failed

            'metrics' : dict or None
                Fit quality metrics:
                - 'r2_score': Overall R² (coefficient of determination)
                - 'chi_square': Chi-square statistic
                - 'r2_mag': R² for impedance magnitude
                - 'r2_phase': R² for impedance phase
                None if fitting failed

            'depth_penalty' : float
                Penalty for circuit complexity (deeper = more penalty)

            'fit_bonus' : float
                Bonus for exceptional fit quality (currently unused, always 0)

            'fit_penalty' : float
                Penalty for fit issues (currently unused, always 0)

        Reward Engineering
        ------------------
        The reward function encourages:
        1. Valid actions (no penalty)
        2. Circuits that fit the data well (high R², low chi²)
        3. Simple circuits over complex ones (fewer components, less depth)

        Reward formula for good fits:
            reward = log10(1/chi²) + log10(1/(1-R²)) - component_penalty - depth_penalty

        Where:
        - log10(1/chi²): Larger for better fits (chi² closer to 0)
        - log10(1/(1-R²)): Larger for better fits (R² closer to 1)
        - component_penalty: Increases with circuit size (0 to 2.0)
        - depth_penalty: 0.2 if circuit depth ≥ 3, else 0
        """
        # Initialize return values
        terminated = False
        param = None
        predicted_Z = None
        metrics = None
        depth_penalty = 0
        reward = 0

        # Check if action is valid according to GEP rules
        validity = action_validity(self.state, action_type, action_position)

        if validity == 0:  # Invalid action - penalize and return
            reward = -0.5
            circuit, _, coding = parse_state_to_circuit(self.state)
            # Return complete step result
            return {
                "observation": self._get_obs(),
                # "info": self._get_info(),
                "reward": reward,
                "terminated": terminated,
                "validity": validity,
                "coding": coding,
                "param": {} if param is None else param,
                "circuit": circuit,
                "predicted_Z": predicted_Z,
                "metrics": metrics,
                "depth_penalty": depth_penalty,
                "fit_bonus": 0,
                "fit_penalty": 0,
            }

        # Valid action - apply mutation to chromosome
        self._update_state(action_type=action_type, action_position=action_position)
        circuit, _, coding = parse_state_to_circuit(self.state)

        # Count circuit components for complexity penalty
        components = ae.parser.get_component_types(circuit)
        components_number = len(components)

        # CACHE CHECK: Try to retrieve previously computed result
        cached_result = (
            self.cache.get(
                circuit_code=coding,
                eis_index=self.current_eis_index,
                action_type=action_type,
                action_position=action_position,
            )
            if (self.cache_enabled and self.current_eis_index is not None)
            else None
        )

        # Use cached result or compute new one
        if cached_result is not None:
            # Normalize cached result format to match _evaluate_circuit output
            result = {
                "reward": cached_result["reward"],
                "metrics": cached_result["metrics"],
                "predicted_Z": cached_result["predicted_Z"],
                "param": cached_result["param"],
                "terminated": cached_result["good_fit"],  # <-- Map good_fit to terminated
                "depth_penalty": cached_result["depth_penalty"],
            }
        else:
            result = self._evaluate_circuit(circuit, components_number)

            # Store in cache
            if self.cache_enabled and self.current_eis_index is not None:
                self.cache.put(
                    circuit_code=coding,
                    eis_index=self.current_eis_index,
                    action_type=action_type,
                    action_position=action_position,
                    reward=result["reward"],
                    metrics=result["metrics"],
                    predicted_Z=result["predicted_Z"],
                    param=result["param"],
                    good_fit=result["terminated"],  # <-- Map terminated to good_fit for cache
                    depth_penalty=result["depth_penalty"],
                    fit_bonus=0.0,
                    fit_penalty=0.0,
                )

        # Extract values for return
        reward = result["reward"]
        metrics = result["metrics"]
        predicted_Z = result["predicted_Z"]
        param = result["param"]
        terminated = result["terminated"]
        depth_penalty = result["depth_penalty"]

        # Return complete step result
        return {
            "observation": self._get_obs(),
            # "info": self._get_info(),
            "reward": reward,
            "terminated": terminated,
            "validity": validity,
            "coding": coding,
            "param": {} if param is None else param,
            "circuit": circuit,
            "predicted_Z": predicted_Z,
            "metrics": metrics,
            "depth_penalty": depth_penalty,
            "fit_bonus": 0,
            "fit_penalty": 0,
        }

    # Functions for seeing the usefulness of the cache, for later evaluation of the two
    # different caches
    # LRU vs clock cache
    def get_cache_stats(self):
        """
        Get detailed statistics about cache performance.

        This helps monitor how well the cache is working and whether it should be tuned
        (capacity increased/decreased, different eviction policy, etc.).

        Returns
        -------
        dict or None
            If cache is enabled, returns dictionary with:
            - 'hits': Number of times cached result was used (fast!)
            - 'misses': Number of times new computation was needed (slow)
            - 'hit_rate': Fraction of lookups that hit cache (0.0 to 1.0)
            - 'size': Current number of cached results
            - 'capacity': Maximum cache capacity
            - 'utilization': Fraction of cache capacity used (0.0 to 1.0)
            - 'cache_type': 'lru' or 'clock'

            If cache is disabled, returns None

        Interpretation
        --------------
        - High hit rate (>50%): Cache is working well, major speedup
        - Low hit rate (<20%): Consider increasing capacity or different policy
        - High utilization (>80%): Cache is being fully used
        - Low utilization (<20%): Cache capacity could be reduced to save memory
        """
        if self.cache_enabled:
            return self.cache.get_stats()
        else:
            return None

    # TODO: For dev puprpose only. Will be removed in the future
    def print_cache_stats(self):
        """
        Print formatted cache performance statistics to console.

        This is a convenience method that displays cache statistics in a readable format.
        Useful for quick checks during training.

        Call this periodically during training to monitor cache effectiveness. If hit rate is
        very low, training is slower than it could be.
        """
        if self.cache_enabled:
            self.cache.print_stats()
        else:
            print("Cache is disabled")

    def clear_cache(self):
        """
        Clear all cached circuit evaluations.

        This resets the cache to empty state. Useful when:
        - Starting a new training run with different hyperparameters
        - Testing specific scenarios from scratch
        - Freeing memory if needed
        - Debugging cache-related issues

        After clearing, hit/miss statistics are reset to zero.
        """
        if self.cache_enabled:
            self.cache.clear()
            print("Cache cleared")
        else:
            print("Cache is disabled")
