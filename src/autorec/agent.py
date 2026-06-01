from typing import Optional, Dict, Union, Any
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import autoeis as ae
from autorec.environment import EIS_ECM_Env
from autorec.utils import (
    karva_to_circuit,
    validity_check,
    plot_eval_fit,
    prepare_and_generate_circuit_gif,
)
from autorec.optimized_data_structures.circular_buffer import CircularBuffer


class DDQN_ECM:
    """
    Double Deep Q-Network agent for EIS equivalent-circuit discovery.

    The agent interacts with an ``EIS_ECM_Env`` environment by mutating GEP chromosome strings,
    stores transitions in a prioritized replay buffer, and trains a neural network to select
    circuit mutations that eventually prodcue ECM. The class also provides helpers for
    saving/loading Keras models and evaluating trained agents on one or more EIS measurements.
    """

    def __init__(
        self,
        training_env: EIS_ECM_Env,
        eval_env: Optional[EIS_ECM_Env] = None,
        save_dir: Path = Path("./"),
        save_start: Optional[int] = 0,
        save_frequency: Optional[int] = 500,
        action_cap: int | None = None,
        random_seed: int = 42,
        # RL Parameters
        episodes_trial: int = 1,
        num_trials: int = 8000,
        gamma: float = 0.99,
        continuous_deadloop: int = 2,
        latent_deadloop: int = 4,
        invalid_terminals: bool = False,
        # Dynamic variable
        initial_epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.99968,
        start_decay: int = 10,
        bayesian: bool = False,
        # NN hyperparameters
        learning_rate: float = 0.0005,
        batch_size: int = 150,
        train_frequency: int = 5,
        update_target_frequency: int = 1000,
        NN_sleep: int = 1000,
        buffer_capacity: int = 15000,
        optimizer_type: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        # Prioritized replay parameters
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_eps: float = 1e-6,
        initial_beta: float = 0.4,
        beta_jump: float = 1.06,
        start_jump: Optional[float] = None,
        anneal_fraction: Optional[float] = None,
        final_beta: float = 0.7,
    ):
        """
        Initialize the reinforcement learning agent and its training configuration.

        This constructor sets up the training environment, neural network training parameters,
        replay buffer configuration, and exploration strategy used by the agent. The agent
        follows a Double Deep Q-Network (DDQN) framework with prioritized experience replay
        and epsilon-greedy exploration.

        Parameters
        ----------
        training_env : EIS_ECM_Env
            Environment used for training the agent.

        eval_env : Optional[EIS_ECM_Env], optional
            Separate environment used for evaluation during training.

        save_dir : Path, optional
            Directory where models, logs, and figures will be saved.

        save_start : Optional[int], optional
            Trial index after which model checkpoints begin to be saved.

        save_frequency : Optional[int], optional
            Frequency (in trials) for saving model checkpoints.

        action_cap : int | None, optional
            Maximum number of actions allowed before the start state is revisited.

        random_seed : int, optional
            Random seed for reproducibility.

        episodes_trial : int, optional
            Number of episodes executed per trial.

        num_trials : int, optional
            Total number of trials for training.

        gamma : float, optional
            Discount factor controlling the importance of future rewards.

        continuous_deadloop : int, optional
            Maximum number of repeated identical state-action in a row before
            detecting a continuous deadloop.

        latent_deadloop : int, optional
            Maximum number of the same visited state so far before detecting a latent deadloop.

        invalid_terminals : bool, optional
            If True, invalid actions terminate the episode. Otherwise, a kickout mechanism is
            used to prevent deadloops (Not efficient).

        initial_epsilon : float, optional
            Initial exploration rate for epsilon-greedy action selection.

        epsilon_min : float, optional
            Minimum exploration rate.

        epsilon_decay : float, optional
            Multiplicative decay factor applied to epsilon.

        start_decay : int, optional
            Trial number after which epsilon decay begins.

        bayesian : bool, optional
            If True, a Bayesian posterior-based bonus is added to the reward. This option is
            currently retained for compatibility and is not active in the reward calculation.

        learning_rate : float, optional
            Learning rate used by the neural network optimizer.

        batch_size : int, optional
            Mini-batch size used during neural network training.

        train_frequency : int, optional
            Frequency (in environment steps) for training the main network.

        update_target_frequency : int, optional
            Frequency (in steps) for updating the target network.

        NN_sleep : int, optional
            Number of initial transitions collected before training begins.

        buffer_capacity : int, optional
            Maximum number of experiences stored in the replay buffer.

        optimizer_type : Union[str, tf.keras.optimizers.Optimizer], optional
            Optimizer used for training the neural network.

        prioritized_replay_alpha : float, optional
            Exponent controlling how strongly sampling prioritizes large TD errors.

        prioritized_replay_eps : float, optional
            Small constant added to priorities to avoid zero probability sampling.

        initial_beta : float, optional
            Initial importance-sampling correction factor.

        beta_jump : float, optional
            Multiplicative factor used to increase beta during training.

        start_jump : Optional[float], optional
            Trial index when beta annealing begins.

        anneal_fraction : Optional[float], optional
            Fraction of training during which beta is annealed. If None, computed as
            (3/5 * num_trials) / num_trials - 0.005

        final_beta : float, optional
            Final importance-sampling correction factor.
        """

        # Store environments
        self.training_env = training_env
        self.eval_env = (
            eval_env if eval_env is not None else training_env
        )  # Default to training_env
        self._active_env = self.training_env  # Start with training environment active

        # Store circuit chromosome parameters as private (read-only via properties)
        self.random_seed = random_seed
        tf.random.set_seed(42)

        # Store regular parameters
        self.action_cap = (
            action_cap
            if action_cap is not None
            else (
                self._active_env.chromosome_HEAD_len + self._active_env.chromosome_TAIL_len + 3
            )
        )
        self.episodes_trial = episodes_trial
        self.num_trials = num_trials
        self.gamma = gamma

        self.continieous_deadloop = continuous_deadloop
        self.latent_deadloop = latent_deadloop
        # self.convergence_check: bool = convergence_check
        self.invalid_terminals = invalid_terminals

        self.initial_epsilon = initial_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = initial_epsilon  # ADDED IN
        self.start_decay = start_decay
        # self.decay_fraction: float = decay_fraction

        # NN hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.update_target_frequency = update_target_frequency
        self.NN_sleep = NN_sleep  # The number of initial data that should be gathered before the first training (should be larger than batch_size)
        self.buffer_capacity = buffer_capacity
        self.optimizer_type = optimizer_type

        # Prioritized replay parameters
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_eps = prioritized_replay_eps
        self.initial_beta = initial_beta
        self.beta_jump = beta_jump
        self.start_jump = start_jump
        self.anneal_fraction = anneal_fraction
        self.final_beta = final_beta

        # Compute dynamic hyperparameter values if not explicitly set
        self.start_jump = (2 / 5) * num_trials if self.start_jump is None else self.start_jump

        self.anneal_fraction = (
            (3 / 5 * num_trials) / num_trials - 0.005
            if self.anneal_fraction is None
            else self.anneal_fraction
        )

        # Create optimizer
        if isinstance(self.optimizer_type, str):
            if self.optimizer_type == "adam":
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        else:  # User passed optimizer object directly
            self.optimizer = self.optimizer_type

        self.bayesian = bayesian
        # self.adaptive_reward = adaptive_reward

        self._save_start = save_start
        self._save_frequency = save_frequency

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.save_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

        self._setup_model()

        self.model.summary()
        summary_file = self.save_dir / "model_summary.txt"
        with open(summary_file, "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

        print(f"Model summary saved to {summary_file}")

        self.prioritized_replay_beta = self.initial_beta

        # lists for visualizations
        self.epsilon_list = []
        self.beta_list = []

    @property
    def get_active_env(self) -> EIS_ECM_Env:
        """Get the currently active environment."""
        return self._active_env

    def switch_to_training(self) -> None:
        """Switch to training environment."""
        self._active_env = self.training_env
        print("✓ Switched to training environment")

    def switch_to_eval(self) -> None:
        """Switch to evaluation environment."""
        self._active_env = self.eval_env
        # print("✓ Switched to evaluation environment")

    def set_training_env(
        self, new_training_env: EIS_ECM_Env, switch_to_it: bool = False
    ) -> None:
        """
        Update the training environment.

        Args:
            new_training_env: New environment to use for training switch_to_it: If True,
            immediately switch active_env to this new environment
        """
        self.training_env = new_training_env
        if switch_to_it:
            self.switch_to_training()
        print("✓ Training environment updated")

    def set_eval_env(self, new_eval_env: EIS_ECM_Env, switch_to_it: bool = False) -> None:
        """
        Update the evaluation environment.

        Args:
            new_eval_env: New environment to use for evaluation
            switch_to_it: If True, immediately switch active_env to this new environment
        """
        self.eval_env = new_eval_env
        if switch_to_it:
            self.switch_to_eval()
        print("✓ Evaluation environment updated")

    def get_current_env_type(self) -> str:
        """Return which environment is currently active."""
        if self.get_active_env is self.training_env:
            return "training"
        elif self.get_active_env is self.eval_env:
            return "evaluation"
        else:
            return "unknown"

    def _setup_model(self) -> None:
        """
        Create and initialize the neural network architecture for the DDQN agent.

        This method builds two identical neural networks:
        1. Main model: Used for selecting actions and gets trained
        2. Target model: Used for calculating Q-value targets, updated less frequently
        The target model starts with identical weights to the main model but diverges during
        training as only the main model is updated frequently.

        Network Architecture:
        ---------------------
        Input Layer:
            - Size: (chromosome_length x num_elements) + EIS_data_length
            - Combines: One-hot encoded circuit state + normalized EIS measurements

        Hidden Layer 1:
            - 40 neurons with ReLU activation
            - Learns basic patterns in state-action relationships

        Hidden Layer 2:
            - 40 neurons with ReLU activation
            - Learns higher-level representations

        Output Layer:
            - Size: Number of possible actions
            - Linear activation (outputs raw Q-values)
            - Each output represents Q(state, action_i)
            - Example: For HEAD=5, TAIL=6: ~40 possible actions

        Sets:
        -----
        self.model : tf.keras.Model
            The main neural network that will be trained
            Used to select actions during training

        self.target_model : tf.keras.Model
            A copy of the main model with frozen weights
            Used to compute target Q-values for training
            Updated periodically (every update_target_frequency steps)
        """
        HEAD_len, TAIL_len = (
            self._active_env.chromosome_HEAD_len,
            self._active_env.chromosome_TAIL_len,
        )
        elems_extended_len, elems_len = (
            len(self._active_env.ELEMENTS_EXTENDED),
            len(self._active_env.ELEMENTS),
        )
        eis_input_len = self._active_env.EIS_INPUT_SZE

        state_shape = (HEAD_len + TAIL_len) * elems_extended_len + eis_input_len
        n_actions = 2 + (HEAD_len - 1) * elems_len + (TAIL_len) * (elems_len - 2)
        inputs = layers.Input(shape=(state_shape,))
        layer1 = layers.Dense(40, activation="relu")(inputs)
        layer2 = layers.Dense(40, activation="relu")(layer1)
        outputs = layers.Dense(n_actions, activation="linear")(layer2)
        print("input shape: ", inputs.shape)
        print("output shape: ", outputs.shape)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filepath: str | Path, save_format: str = "keras") -> None:
        """
        Save the trained neural network model to disk for later use.

        Parameters:
        -----------
        filepath : str or Path
            Location to save the model file

            Examples:
            - 'my_model.keras' (saves in current directory)
            - 'models/experiment_1/model.keras' (creates nested directories)
            - Path('results/best_model.h5') (using pathlib)

        save_format : str, default='keras'
            File format for saving the model

            OPTIONS:
            - 'keras' (recommended):
            - 'h5' (legacy)

        Returns:
        --------
        None
            Prints confirmation message upon successful save

        File Extension Handling:
        ------------------------
        The method automatically adds the correct extension:
        - save_format='keras' → ensures '.keras' extension
        - save_format='h5' → ensures '.h5' extension

        If you specify 'model' as filepath with save_format='keras', it will be saved as
        'model.keras' automatically.

        Note:
        -----
        Only the MAIN model is saved (not the target model).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if save_format == "keras" and not str(filepath).endswith(".keras"):
            filepath = filepath.with_suffix(".keras")
        elif save_format == "h5" and not str(filepath).endswith(".h5"):
            filepath = filepath.with_suffix(".h5")

        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str | Path) -> None:
        """
        Load a previously trained neural network model from disk. Also, Target model is
        automatically created as a copy of the loaded model.

        If resuming training, you'll need to separately restore:
        - Replay buffer: history = pd.read_pickle('replay_buffer.pkl')
        - Epsilon value: agent.epsilon = saved_epsilon
        - Trial counter: start_trial = saved_trial

        Parameters:
        -----------
        filepath : str or Path
            Location of the saved model file

        Returns:
        --------
        None (Prints confirmation message upon successful load)

        Note:
        -----
        If you're loading a model trained with different environment parameters (different
        chromosome_HEAD_len, different elements, etc.), the model architecture may not match
        and loading will fail. Ensure the environment configuration matches what was used
        during training.

        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = tf.keras.models.load_model(filepath)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        print(f"✓ Model loaded from: {filepath}")

    def scheduler(self, trial: int, schedule_timesteps: float) -> float:
        """
        Calculate the importance sampling weight exponent (beta) for prioritized replay (Refer
        to the PER original paper by Google DeepMind).

        This scheduler gradually increases beta from initial_beta to final_beta over training,
        starting with less bias correction (focusing on learning from important samples) and
        ending with full bias correction (ensuring unbiased gradient updates).

        Parameters:
        -----------
        trial : int
            Current training trial/episode number (0 to num_trials)

        schedule_timesteps : float
            Total number of trials over which to increase beta
            Controls how quickly beta increases

        Returns:
        --------
        float
            The beta value to use for this trial
            Range: [initial_beta, final_beta]
            Increases linearly with trial number
        """
        fraction = min(trial / schedule_timesteps, 1.0)
        value = self.initial_beta + fraction * (self.final_beta - self.initial_beta)
        return value

    def _sample_experience(
        self, history: pd.DataFrame, prioritized_replay_beta: float
    ) -> tuple:
        """
        Sample a batch of experiences from the replay buffer using prioritized sampling (Refer
        to the PER original paper by Google DeepMind)..

        Instead of sampling uniformly (all experiences equally likely), this method samples
        experiences proportional to their priority. Experiences with higher priority (larger
        TD error = more surprising) are sampled more frequently because the agent can learn
        more from them.

        Parameters:
        -----------
        history : pd.DataFrame
            The replay buffer containing all stored experiences
            Must have a 'priority' column with priority values for each experience

        prioritized_replay_beta : float
            Importance sampling exponent
            Higher values = more bias correction
            Should increase over training (use scheduler() method)

        Returns:
        --------
        tuple: (samples, indices, weights)
            samples : pd.DataFrame
                Subset of history containing batch_size sampled experiences
                These are the experiences that will be used for model training

            indices : np.ndarray
                Array of integer indices indicating which rows were sampled
                Used to update priorities after training (based on new TD errors)
                Shape: (batch_size,)

            weights : np.ndarray
                Importance sampling weights for each sampled experience
                Multiply these with TD errors during gradient computation
                Normalized so max(weights) = 1.0
                Shape: (batch_size,)
        """
        priority_powered_alpha = history["priority"].values ** self.prioritized_replay_alpha
        total_priority = priority_powered_alpha.sum()

        # Use the already calculated priority_powered_alpha values
        probabilities = priority_powered_alpha / total_priority

        # Ensure probabilities sum to exactly 1.0 to avoid numerical errors
        # NOTE: Changed from original no OOP code so we need to keep an eye on it
        # (add this comment)
        probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(
            len(history), size=self.batch_size, p=probabilities, replace=False
        )
        samples = history.iloc[indices]

        # NOTE: In case you need to calculate the weights in the future
        weights = (len(history) * probabilities[indices]) ** (-prioritized_replay_beta)
        weights /= weights.max()

        return samples, indices, weights

    def invalid_actions(self, conding_length: int) -> list:
        """
        Identify all actions that would create invalid circuit configurations or repetitions.

        Parameters:
        -----------
        conding_length : int
            Length of the "coding" part of the chromosome
            This is the portion that actually forms the circuit tree
            Actions beyond this position modify unused tail positions and are invalid

            Example: If chromosome is '+RPRRRR' and tree only uses 4 positions,
                    conding_length = 4, so positions 4-6 are non-coding

        Returns:
        --------
        list of int
            Indices of invalid actions in the ACTIONS_LIST DataFrame
            These correspond to row numbers in self._active_env.ACTIONS_LIST
        """
        invalid_indices = []
        for idx, action_type, action_position in self._active_env.ACTIONS_LIST.itertuples(
            index=True
        ):
            new_state = (
                self._active_env.state[:action_position]
                + f"{action_type}"
                + self._active_env.state[action_position + 1 :]
            )
            new_state_karva = new_state.replace("/", "-")
            circuit = karva_to_circuit(new_state_karva)

            validity = validity_check(circuit)
            if action_position > conding_length - 1:
                validity = False
            if self._active_env.state == new_state:
                validity = False

            if validity is False:
                invalid_indices.append(idx)

        return invalid_indices

    def eps_greedy(self, flatten_Z, uniform=True, non_uniform_ratio=0.5):
        """
        Select an action using epsilon-greedy strategy with configurable exploration.

        Epsilon-greedy is a fundamental exploration-exploitation tradeoff strategy:
        - Higher ε: explore (random action)
        - Lower ε: exploit (best known action)

        This implementation offers two modes:
        1. Uniform: Consider all actions (valid + invalid) equally
        2. Non-uniform: Bias selection toward valid or invalid actions (This will be used for
           deadloop detection algorithm)

        The epsilon value typically decays over training: starting high (explore a lot) and
        decreasing toward epsilon_min (exploit learned knowledge).

        Parameters:
        -----------
        flatten_Z : np.ndarray
            Flattened, normalized EIS impedance data for the current episode (referred to as
            the combined EIS representation in the paper)

        uniform : bool, default=True
            Action selection strategy:

            True (Uniform mode):
                - Exploration: Random action from all actions (valid + invalid)
                - Exploitation: Best action from all actions (may be invalid)
                - Faster (no validity checking)

            False (Non-uniform mode):
                - Exploration: Biased sampling (invalid vs valid actions)
                - Exploitation: Best action from only valid actions
                - Slower (checks validity for every action)

        Returns:
        --------
        tuple: (action_position, action_type)
            action_position : int
                Which chromosome position to modify (0 to chromosome_length-1)

            action_type : str
                Which element to place ('+', '/', 'R', 'L', 'P')
        """
        ec = ae.core.ec
        ACTION_LIST = self._active_env.ACTIONS_LIST
        if np.random.rand() <= self.epsilon:  # Explore
            if uniform is True:
                # Option 1: just random selection amonge all valid/invalid actions
                rnd_number = np.random.randint(0, len(ACTION_LIST))
                chosen_action = ACTION_LIST.iloc[rnd_number]
                action_position = int(chosen_action["action_position"])
                action_type = str(chosen_action["action_type"])
            else:
                karva = self._active_env.state.replace("/", "-")
                tree = ec.karva_to_tree(karva)
                conding_length = len(tree)
                invalid_indices = self.invalid_actions(conding_length)
                rnd_valid_invalid = np.random.rand()
                if rnd_valid_invalid <= non_uniform_ratio:
                    invalids = ACTION_LIST.loc[invalid_indices]
                    chosen_action = invalids.sample()
                else:
                    valids = ACTION_LIST.drop(invalid_indices)
                    chosen_action = valids.sample()

                action_position = chosen_action.iloc[0]["action_position"]
                action_type = chosen_action.iloc[0]["action_type"]

        else:  # Exploit
            encoded_state = np.array(self._active_env.encoded_state)
            NN_state = np.concatenate((flatten_Z, encoded_state))
            q_values = self.model.predict(NN_state[np.newaxis], verbose=0)

            if uniform is True:
                # Option 1: just select the best action amonge all valid/invalid actions
                chosen_action = ACTION_LIST.iloc[int(np.argmax(q_values[0]))]
            else:
                # Option 2: select the best action amonge only valid actions
                karva = self._active_env.state.replace("/", "-")
                tree = ec.karva_to_tree(karva)
                conding_length = len(tree)
                invalid_indices = self.invalid_actions(conding_length)
                q_values[0][invalid_indices] = -np.inf
                chosen_action = ACTION_LIST.iloc[int(np.argmax(q_values[0]))]

            action_position = int(chosen_action["action_position"])
            action_type = str(chosen_action["action_type"])

        return action_position, action_type

    def _train_model(
        self,
        history,
    ):
        """
        Perform one training step using DDQN with prioritized experience replay.

        Args:
            history: DataFrame containing experience replay buffer

        Returns:
            Mean loss for this training step
        """
        samples, sample_indices, weights = self._sample_experience(
            history, self.prioritized_replay_beta
        )
        all_states = []
        all_next_state = []
        all_rewards = []
        all_flags = []
        all_actions = []

        # Select only the 8 columns needed
        # samples_subset = samples[
        #     [
        #         "EIS",
        #         "state",
        #         "action_type",
        #         "action_position",
        #         "new_state",
        #         "reward",
        #         "terminal_flag",
        #         "priority",
        #     ]
        # ]
        samples_subset = samples[
            [
                "EIS",
                "encoded_state",
                "action_type",
                "action_position",
                "encoded_new_state",
                "reward",
                "terminal_flag",
                "priority",
            ]
        ]

        # for (
        #     sample_EIS_i,
        #     sample_state,
        #     sample_action_type,
        #     sample_action_position,
        #     sample_next_state,
        #     sample_reward,
        #     sample_flag,
        #     sample_priority,
        # ) in samples_subset.itertuples(index=False):
        for (
            sample_EIS_i,
            encoded_state,
            sample_action_type,
            sample_action_position,
            encoded_next_state,
            sample_reward,
            sample_flag,
            sample_priority,
        ) in samples_subset.itertuples(index=False):
            sample_action = (sample_action_type, sample_action_position)
            sample_flatten_Z = self._active_env.dataset.iloc[sample_EIS_i]["flatten_Z"]

            # Use pre-encoded states from buffer
            NN_state = np.concatenate((sample_flatten_Z, encoded_state))
            NN_next_state = np.concatenate((sample_flatten_Z, encoded_next_state))

            all_states.append(NN_state)
            all_next_state.append(NN_next_state)
            all_rewards.append(sample_reward)
            all_flags.append(sample_flag)
            all_actions.append(sample_action)

        # NOTE: This part is for the target Q-value
        # Use the main model for finding the next best action
        next_q_values_main = self.model.predict(np.array(all_next_state), verbose=0)
        next_actions = np.argmax(next_q_values_main, axis=1)

        # Use the target model for predicting the Q-values of the best next action
        next_q_values = self.target_model.predict(np.array(all_next_state), verbose=0)
        max_next_q_values = next_q_values[np.arange(self.batch_size), next_actions]

        # Calculate target Q-values
        target_q_values = (
            np.array(all_rewards) + (1 - np.array(all_flags)) * self.gamma * max_next_q_values
        )

        # NOTE: This part is for Predicted Q-value and loss calculation (Predicted Q-value is
        # also know as Selected Q-value as it is the Q-value for the selected/sampled action
        # of that state)
        # find action indices (labels)
        action_indices = []
        counter = 0
        ACTIONS_LIST = self._active_env.ACTIONS_LIST
        for action in all_actions:
            indices = ACTIONS_LIST.loc[
                (ACTIONS_LIST["action_type"] == action[0])
                & (ACTIONS_LIST["action_position"] == action[1])
            ].index
            action_indices.append([counter, indices[0]])
            counter += 1

        with tf.GradientTape() as tape:
            q_values = self.model(np.array(all_states))
            selected_q_values = tf.gather_nd(q_values, action_indices)
            loss = weights * tf.square(target_q_values - selected_q_values)
            loss_mean = tf.reduce_mean(loss)

        grads = tape.gradient(loss_mean, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Calculate the loss and then calculate/update the priority for PER
        q_values = self.model(np.array(all_states))
        selected_q_values = tf.gather_nd(q_values, action_indices)

        td_errors = np.abs(target_q_values - selected_q_values.numpy())
        priority = (td_errors + self.prioritized_replay_eps).astype(np.float32)
        history.loc[sample_indices, "priority"] = priority

        return loss_mean.numpy()

    def train(self):
        """Main training loop across all trials."""

        self.switch_to_training()
        print(f"Training on: {self.training_env.dataset.shape[0]} EIS samples")

        # Tracking variables
        NN_loss = pd.DataFrame()
        success_rate = pd.DataFrame()
        initial_time = time.time()

        # Initialize CircularBuffer for efficient experience replay
        buffer = CircularBuffer(
            capacity=self.buffer_capacity,
            dtypes=[
                # Standard for NumPy: (name, dtype, shape) for multi-dimensional fields
                ("EIS", "i4"),
                ("state", "U50"),
                (
                    "encoded_state",
                    "f4",
                    (
                        self._active_env.chromosome_len
                        * len(self._active_env.ELEMENTS_EXTENDED),
                    ),
                ),  # ADDED THIS
                ("action_type", "U10"),
                ("action_position", "i4"),
                ("new_state", "U50"),
                (
                    "encoded_new_state",
                    "f4",
                    (
                        self._active_env.chromosome_len
                        * len(self._active_env.ELEMENTS_EXTENDED),
                    ),
                ),  # ADDED THIS
                ("reward", "f4"),
                ("terminal_flag", "i1"),
                ("priority", "f4"),
            ],
        )

        terminal_states = pd.DataFrame(
            columns=["EIS", "state", "coding", "circuit", "episode", "action"]
        )

        # Training state trackers
        iteration = 0
        target_freq = 0
        memory = 0
        success_count = 0
        statistical_analysis = {
            "episodic_cumul_reward": [],
            "episodic_mean_reward": [],
        }  # Learning curve tracking

        # Main trial loop
        for t in range(self.num_trials):
            EIS_i, _ = self._active_env.reset()
            flatten_Z = self._active_env.Z_norm_flatten
            ground_truth = self._active_env.dataset["true_circuit"][EIS_i]
            print("Trial: ", t, f"for EIS_i: {EIS_i} (EIS of {ground_truth})")

            # Run all episodes for this trial
            for e in range(
                self.episodes_trial
            ):  # NOTE: For dev purpose only as episodes_trial=1 for now
                episode_result = self._run_episode(
                    EIS_i,
                    t,
                    e,
                    flatten_Z,
                    buffer,
                    terminal_states,
                    iteration,
                    target_freq,
                    memory,
                    success_count,
                )

                # Update counters from episode
                iteration = episode_result["iteration"]
                target_freq = episode_result["target_freq"]
                memory = episode_result["memory"]
                success_count = episode_result["success_count"]

                # Track episode statistics
                NN_loss = self._update_loss_tracking(NN_loss, episode_result["losses"], t)
                success_rate = self._update_success_rate_tracking(
                    success_rate, success_count, iteration, t
                )
                statistical_analysis = self._update_episode_statistics(
                    statistical_analysis, episode_result["total_reward"]
                )

            # intermediate saving
            if t >= self._save_start and (t - self._save_start) % self._save_frequency == 0:
                self.model.save(self.model_dir / f"dqn_model_trial_{t}.keras")
                print(f"Intermediate model saved at trial {t}")

            # Post-trial updates
            self._update_epsilon(t)
            self._update_beta(t)

        print("RL loop is completed")
        final_time = time.time()
        print(f"Training took {(final_time - initial_time) / 60:.2f} minutes")

        self.model.save(self.save_dir / "dqn_model.keras")
        success_rate.to_csv(self.save_dir / "success_rate.csv")
        NN_loss.to_csv(self.save_dir / "NN_loss.csv")
        terminal_states.to_csv(self.save_dir / "terminal_states.csv")
        pd.DataFrame(statistical_analysis).to_pickle(
            self.save_dir / "statistical_analysis.pkl"
        )

        self._plot_training_metrics(NN_loss, statistical_analysis)

        return {
            "NN_loss": NN_loss,
            "success_rate": success_rate,
            "terminal_states": terminal_states,
            "statistical_analysis": statistical_analysis,
        }

    def _run_episode(
        self,
        EIS_i,
        trial,
        episode,
        flatten_Z,
        buffer,
        terminal_states,
        iteration,
        target_freq,
        memory,
        success_count,
    ):
        """Execute a single episode with multiple actions."""
        episode_history = []
        total_reward = 0
        deadloop_chain = []
        deadloop_flag = False
        losses = []

        for a in range(self.action_cap):
            # Execute single action step
            step_result = self._interact_update(
                a,
                flatten_Z,
                deadloop_flag,
                episode_history,
                deadloop_chain,
                episode,
                EIS_i,
                buffer,
                iteration,
                target_freq,
                memory,
            )

            # Update episode tracking
            episode_history.append(step_result["history_entry"])
            total_reward += step_result["reward"]
            deadloop_chain = step_result["deadloop_chain"]
            deadloop_flag = step_result["deadloop_flag"]
            iteration = step_result["iteration"]
            target_freq = step_result["target_freq"]
            memory = step_result["memory"]

            if step_result["loss"] is not None:
                losses.append({"loss": step_result["loss"], "t": trial})

            # Check for terminal conditions
            if step_result["good_fit"]:
                success_count += 1
                terminal_states = self._record_terminal_state(
                    terminal_states, EIS_i, episode, a, step_result["env_output"]
                )  # For dev purpose only
                break

            if self.invalid_terminals and step_result["validity"] == 0:
                break

        return {
            "iteration": iteration,
            "target_freq": target_freq,
            "memory": memory,
            "success_count": success_count,
            "total_reward": total_reward,
            "losses": losses,
        }

    def _interact_update(
        self,
        action_num,
        flatten_Z,
        deadloop_flag,
        episode_history,
        deadloop_chain,
        episode,
        EIS_i,
        buffer,
        iteration,
        target_freq,
        memory,
    ):
        """
        This function performs one complete RL interaction step inside an episode. It handles:

        - action selection,
        - environment transition,
        - replay-buffer insertion,
        - model training and target-network update,
        - deadloop detection,

        and

            returns all updated tracking values needed by the caller.
        """
        # Action selection with deadloop handling
        if not deadloop_flag:
            action_position, action_type = self.eps_greedy(flatten_Z)
        else:
            action_position, action_type = self.eps_greedy(
                flatten_Z, uniform=False, non_uniform_ratio=0
            )
            deadloop_flag = False

        prev_state = self._active_env.state
        prev_encoded_state = self._active_env.encoded_state.copy()

        # Execute action in environment
        env_output = self._active_env.step(
            action_type=action_type, action_position=action_position
        )

        reward = env_output["reward"]
        validity = env_output["validity"]
        good_fit = env_output["terminated"]

        terminal_flag = 1 if good_fit else 0

        new_encoded_state = self._active_env.encoded_state.copy()

        # Add experience to buffer with priority
        priority = buffer.buffer["priority"][: buffer.size].max() if len(buffer) > 0 else 1.0

        buffer.add(
            EIS=EIS_i,
            state=prev_state,
            encoded_state=prev_encoded_state,  # NEW: Store encoded state
            action_type=action_type,
            action_position=action_position,
            new_state=self._active_env.state,
            encoded_new_state=new_encoded_state,  # NEW: Store encoded next state
            reward=reward,
            terminal_flag=terminal_flag,
            priority=priority,
        )

        # # Check and perform training
        # NOTE: not moved to episode training
        loss, memory = self._check_and_train(buffer, iteration, memory)

        # # Update target network if needed
        # NOTE: not Moved to episode training
        target_freq = self._check_and_update_target(target_freq)

        # Detect deadloops
        # NOTE: not moved to episode training because deadloop detection needs a lot of
        # information from the current action
        new_deadloop_chain, new_deadloop_flag = self._detect_deadloop(
            validity,
            action_num,
            action_type,
            action_position,
            episode_history,
            deadloop_chain,
            episode,
        )

        # Create history entry for this action
        history_entry = {
            "action": action_num,
            "state": prev_state,
            "new_state": self._active_env.state,
            "new_state_coding": env_output["coding"],
            "action_type": action_type,
            "action_position": action_position,
            "reward": reward,
            "total_reward": 0,  # Will be updated by caller
            "terminal_flag": terminal_flag,
        }

        # Update iteration counter
        iteration += 1

        return {
            "history_entry": history_entry,
            "reward": reward,
            "validity": validity,
            "good_fit": good_fit,
            "env_output": env_output,
            "deadloop_chain": new_deadloop_chain,
            "deadloop_flag": new_deadloop_flag,
            "iteration": iteration,
            "target_freq": target_freq,
            "memory": memory,
            "loss": loss,
        }

    def _check_and_train(self, buffer, iteration, memory):
        """Check if training is needed and perform training step."""
        loss = None
        run_training = (iteration == self.NN_sleep) or (
            iteration > self.NN_sleep and memory + 1 == self.train_frequency
        )

        if run_training:
            history_buffer_df = buffer.to_dataframe()
            loss = self._train_model(history_buffer_df)
            memory = 0
            # Update buffer priorities
            for idx in range(len(history_buffer_df)):
                if idx < buffer.size:
                    buffer.buffer["priority"][idx] = history_buffer_df.iloc[idx]["priority"]

        else:
            # Update memory counter
            memory += 1

        return loss, memory

    def _check_and_update_target(self, target_freq):
        """Update target network if frequency threshold is met."""
        if target_freq + 1 == self.update_target_frequency:
            self.target_model.set_weights(self.model.get_weights())
            return 0
        else:
            # Update target frequency counter
            return target_freq + 1

    def _detect_deadloop(
        self,
        validity,
        action_num,
        action_type,
        action_position,
        episode_history,
        deadloop_chain,
        episode,
    ):
        """Detect if agent is stuck in a deadloop pattern."""
        # Check for consecutive invalid actions
        if validity == 0 and action_num > 0:
            prev_action = episode_history[action_num - 1]
            if (
                prev_action["action_type"] == action_type
                and prev_action["action_position"] == action_position
            ):
                deadloop_chain.append(action_num)
            else:
                deadloop_chain = []
        else:
            deadloop_chain = []

        # Check for state revisitation (latent deadloop)
        current_state = self._active_env.state
        state_count = sum(1 for h in episode_history if h["new_state"] == current_state)

        # Trigger deadloop flag if thresholds exceeded
        deadloop_flag = len(deadloop_chain) >= self.continieous_deadloop or (
            state_count >= self.latent_deadloop and episode > self.NN_sleep
        )

        return deadloop_chain, deadloop_flag

    def _record_terminal_state(self, terminal_states, EIS_i, episode, action, env_output):
        """Record a successful terminal state if it's new for each EIS."""
        state = self._active_env.state

        # Only record if this state hasn't been seen for this EIS
        if (
            len(
                terminal_states[
                    (terminal_states["state"] == state) & (terminal_states["EIS"] == EIS_i)
                ]
            )
            == 0
        ):
            terminal_states = pd.concat(
                [
                    terminal_states,
                    pd.DataFrame(
                        {
                            "EIS": [EIS_i],
                            "state": [state],
                            "coding": env_output["coding"],
                            "circuit": env_output["circuit"],
                            "reward": [env_output["reward"]],
                            "episode": [episode],
                            "action": [action],
                            "r2_score": env_output["metrics"]["r2_score"],
                            "chi_square": env_output["metrics"]["chi_square"],
                            "Parameters": [env_output["param"]],
                        }
                    ),
                ],
                axis=0,
                ignore_index=True,
            )

        return terminal_states

    def _update_epsilon(self, trial):
        """Apply epsilon decay schedule."""
        self.epsilon_list.append(self.epsilon)
        if trial > self.start_decay and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def _update_beta(self, trial):
        """Apply beta annealing schedule for prioritized replay."""
        self.beta_list.append(self.prioritized_replay_beta)
        if trial > self.start_jump and self.prioritized_replay_beta < 1:
            self.prioritized_replay_beta = self.scheduler(
                trial - self.start_jump,
                self.num_trials * self.anneal_fraction,
            )
            if self.prioritized_replay_beta > 1:
                self.prioritized_replay_beta = 1

    def _update_loss_tracking(self, NN_loss, losses, trial):
        """Add new losses to tracking DataFrame."""
        if losses:
            loss_df = pd.DataFrame(losses)
            NN_loss = pd.concat([NN_loss, loss_df], axis=0, ignore_index=True)
        return NN_loss

    def _update_success_rate_tracking(self, success_rate, success_count, iteration, trial):
        """Calculate and record current success rate."""
        success_fraction = success_count / iteration if iteration > 0 else 0
        success_rate = pd.concat(
            [
                success_rate,
                pd.DataFrame({"t": [trial], "success_rate": [success_fraction]}),
            ],
            axis=0,
            ignore_index=True,
        )
        return success_rate

    def _update_episode_statistics(self, statistical_analysis, total_reward):
        """Update running statistics for episode performance."""
        statistical_analysis["episodic_cumul_reward"].append(total_reward)

        # Calculate mean reward over the last 50 episodes for a smoothed learning curve
        mean_reward = np.mean(statistical_analysis["episodic_cumul_reward"][-50:])
        statistical_analysis["episodic_mean_reward"].append(mean_reward)

        return statistical_analysis

    def _plot_training_metrics(self, NN_loss, statistical_analysis):
        """Generate training visualization plots"""
        self._use_file_plot_backend()

        # 1. Epsilon and Beta over trials
        _ = plt.figure(figsize=(8, 5))
        plt.plot(self.epsilon_list, label="epsilon", color="blue", linestyle="-")
        plt.plot(self.beta_list, label="beta", color="red", linestyle="-")
        plt.title("Dynamic variables")
        plt.xlabel("trial")
        plt.ylabel("beta/epsilon")
        plt.legend()
        plot_path = self.save_dir / "epsilon_beta.png"
        plt.savefig(plot_path)
        self._display_plot_if_notebook(plot_path)
        plt.close()

        # 2. Episodic rewards
        _ = plt.figure(figsize=(8, 5))
        plt.plot(
            [i for i in range(len(statistical_analysis["episodic_cumul_reward"]))],
            statistical_analysis["episodic_cumul_reward"],
            label="Cumulative Reward",
        )
        plt.plot(
            [i for i in range(len(statistical_analysis["episodic_mean_reward"]))],
            statistical_analysis["episodic_mean_reward"],
            label="Mean Reward (50-trial window)",
        )
        plt.xlabel("Trial")
        plt.ylabel("Reward")
        plt.title("Training Rewards Over Time")
        plt.legend()
        plot_path = self.save_dir / "reward_plot.png"
        plt.savefig(plot_path)
        self._display_plot_if_notebook(plot_path)
        plt.close()

        # 3. Neural network loss
        _ = plt.figure(figsize=(8, 5))
        if "loss" in NN_loss:
            plt.plot(NN_loss["loss"])
        plt.xlabel("Training steps")
        plt.ylabel("NN Loss")
        plt.title("Neural Network Loss During Training")
        plot_path = self.save_dir / "loss.png"
        plt.savefig(plot_path)
        self._display_plot_if_notebook(plot_path)
        plt.close()

    def _use_file_plot_backend(self):
        """Use a non-GUI backend for training plots that are only saved to disk."""
        try:
            if plt.get_backend().lower() == "macosx":
                plt.switch_backend("Agg")
        except Exception:
            # Plot generation should not fail because an interactive backend is unavailable.
            plt.switch_backend("Agg")

    def _display_plot_if_notebook(self, plot_path):
        """Display saved training PNGs in notebook frontends."""
        try:
            from IPython import get_ipython
            from IPython.display import Image, display

            if get_ipython() is not None:
                display(Image(filename=str(plot_path)))
        except Exception:
            pass

    def generate_ECM(
        self,
        EIS_i: Optional[int] = None,
        flatten_Z: Optional[np.ndarray] = None,
        # TODO: Should fix the unkown ground_truth_state during dataprep
        # ground_truth_state: Optional[Any] = None,
        ground_truth_circuit: Optional[str] = None,
        max_actions: Optional[int] = None,
        verbose: bool = True,
        get_fit_plots: bool = True,
        gif_generation: bool = True,
        use_eval_env: bool = True,  # ADD THIS
    ) -> Dict[str, Any]:
        """
        Evaluate the trained agent on a single EIS measurement.

        Can be called either with:
        - EIS_i: to evaluate on a dataset entry (fetches flatten_Z and ground truth
          automatically)
        - flatten_Z: to evaluate on provided impedance data (ground truth optional)

        Args:
            EIS_i: Index of the EIS measurement in dataset (if None, must provide flatten_Z)
            flatten_Z: Flattened impedance data (if None, must provide EIS_i)
            ground_truth_circuit: Optional ground truth circuit for comparison
            max_actions: Maximum number of actions to try (default: self.action_cap)
            verbose: Whether to print progress
            get_fit_plots: Whether to save fit plots for successful evaluations
            gif_generation: Whether to generate a circuit-evolution GIF
            use_eval_env: Whether to run on the evaluation environment instead
                of the training environment

        Returns:
            Dictionary containing evaluation results
        """
        # Switch to appropriate environment
        if use_eval_env:
            self.switch_to_eval()
        else:
            self.switch_to_training()

        # Validate inputs - need either EIS_i or flatten_Z
        # TODO: Right now the option for using flatten_Z is not working (Have to find a smart
        # way of handling different input size)
        if EIS_i is None and flatten_Z is None:
            raise ValueError("Must provide either EIS_i or flatten_Z")
        if EIS_i is not None and flatten_Z is not None:
            raise ValueError("Cannot provide both EIS_i and flatten_Z - choose one")

        if max_actions is None:
            max_actions = self.action_cap

        # If using dataset index, fetch the data
        if EIS_i is not None:
            EIS_i, _ = self._active_env.reset(EIS_i)
        else:
            EIS_i, _ = self._active_env.reset()  # Reset to valid state

        Z_true = self._active_env.Z
        flatten_Z = self._active_env.Z_norm_flatten
        ground_truth_circuit = self._active_env.true_circuit

        # Store original epsilon and set to 0 for greedy evaluation
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        # Track evaluation metrics
        action_history = []
        best_result = {
            "found_solution": False,
            "reward": -np.inf,
            "action_number": -1,
            "circuit": None,
            "coding": None,
            "metrics": None,
            "param": None,
            "state": None,
        }

        if verbose:
            print(f"\n{'=' * 80}")
            # if ground_truth_state is not None:
            #     print(f"Evaluating EIS {EIS_i}: Ground Truth = {ground_truth_state}")
            # else:
            #     print(f"Evaluating EIS {EIS_i} (no ground truth provided)")
            if ground_truth_circuit is not None:
                print(f"True Circuit: {ground_truth_circuit}")
            print(f"{'=' * 80}")

        for a in range(max_actions):
            # Get greedy action (epsilon=0)
            action_position, action_type = self.eps_greedy(flatten_Z)
            prev_state = self._active_env.state

            # Take action in environment
            env_step_out = self._active_env.step(
                action_type=action_type, action_position=action_position
            )

            reward = env_step_out["reward"]
            validity = env_step_out["validity"]
            good_fit = env_step_out["terminated"]
            circuit = env_step_out["circuit"]
            coding = env_step_out["coding"]
            metrics = env_step_out["metrics"]
            param = env_step_out["param"]
            predicted_Z = env_step_out["predicted_Z"]

            # Store action history
            action_history.append(
                {
                    "action": a,
                    "action_type": action_type,
                    "action_position": action_position,
                    "prev_state": prev_state,
                    "new_state": self._active_env.state,
                    "coding": coding,
                    "circuit": circuit,
                    "reward": reward,
                    "validity": validity,
                    "good_fit": good_fit,
                    "metrics": metrics,
                }
            )

            # Update best result if this is better
            if reward > best_result["reward"]:
                best_result = {
                    "found_solution": good_fit,
                    "reward": reward,
                    "action_number": a,
                    "circuit": circuit,
                    "coding": coding,
                    "metrics": metrics,
                    "param": param,
                    "state": self._active_env.state,
                }

            if verbose and good_fit:
                print(f"\n✓ Found good fit at action {a}!")
                print(f"  Circuit: {circuit}")
                print(f"  Coding: {coding}")
                print(f"  Reward: {reward:.4f}")
                if metrics:
                    print("  Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"    {key}: {value:.6f}")

            # Break if good fit found
            # NOTE: fixed, changed the return values for if a good fit
            if good_fit:
                best_result = {
                    "found_solution": good_fit,
                    "reward": reward,
                    "action_number": a,
                    "circuit": circuit,
                    "coding": coding,
                    "metrics": metrics,
                    "param": param,
                    "state": self._active_env.state,
                }
                break

        # Restore original epsilon
        self.epsilon = original_epsilon

        # Prepare evaluation summary
        eval_result = {
            "EIS_i": EIS_i,
            "environment_type": self.get_current_env_type(),
            # 'ground_truth_state': ground_truth_state,
            "ground_truth_circuit": ground_truth_circuit,
            "found_solution": best_result["found_solution"],
            "best_reward": best_result["reward"],
            "best_action_number": best_result["action_number"],
            "best_circuit": best_result["circuit"],
            "best_coding": best_result["coding"],
            "best_metrics": best_result["metrics"],
            "best_param": best_result["param"],
            "best_state": best_result["state"],
            "total_actions_taken": len(action_history),
            "action_history": action_history,
        }

        if verbose:
            print(f"\n{'=' * 80}")
            print("Evaluation Summary:")
            print(f"  Solution Found: {best_result['found_solution']}")
            print(f"  Best Reward: {best_result['reward']:.4f}")
            print(f"  Actions Taken: {len(action_history)}/{max_actions}")
            if best_result["circuit"]:
                print(f"  Best Circuit: {best_result['circuit']}")
            print(f"{'=' * 80}\n")

        if get_fit_plots:
            plot_eval_fit(
                EIS_i,
                Z_true,
                predicted_Z,
                good_fit,
                metrics,
                self.save_dir,
                best_result["circuit"],
            )

        # Generate circuit evolution GIF if requested
        if gif_generation and EIS_i is not None and Z_true is not None:
            prepare_and_generate_circuit_gif(
                _active_env=self._active_env,
                action_history=action_history,
                best_result=best_result,
                EIS_i=EIS_i,
                Z_true=Z_true,
                save_dir=self.save_dir,
            )

        return eval_result

    def eval_batch_eis(
        self,
        eis_indices: Optional[list[int]] = None,
        max_actions: Optional[int] = None,
        verbose: bool = False,
        use_eval_env: bool = True,
        gif_generation: bool = False,
        all_rows: bool = False,
        num_samples: Optional[int] = None,  # NEW: Quick check with random samples
    ) -> pd.DataFrame:
        """
        Evaluate the trained agent on a batch of EIS measurements.

        Common usage patterns:

        1. Evaluate specific EIS rows by row position:
           ``agent.eval_batch_eis(eis_indices=[0, 5, 12], max_actions=24)``

        2. Evaluate a random subset from the selected environment:
           ``agent.eval_batch_eis(num_samples=100, max_actions=24)``

        3. Evaluate every EIS in the selected environment:
           ``agent.eval_batch_eis(all_rows=True, max_actions=24)``

        4. Use the training environment instead of the evaluation environment:
           ``agent.eval_batch_eis(eis_indices=[0, 1, 2], use_eval_env=False)``

        If ``use_eval_env=True`` the agent switches to its evaluation
        environment before running. If ``use_eval_env=False`` it evaluates on
        the training environment. Row positions are selected from the active
        environment's dataset.

        Args:
            eis_indices: List of EIS row positions to evaluate. Ignored if
                ``all_rows=True`` or ``num_samples`` is set.
            max_actions: Maximum number of actions per EIS (default: self.action_cap)
            verbose: Whether to print detailed progress for each EIS
            use_eval_env: Whether to use the evaluation environment. If False,
                use the training environment.
            gif_generation: Generate a circuit-evolution GIF for each EIS.
            all_rows: If True, evaluate all EIS rows in the active
                environment's dataset. Overrides ``eis_indices``.
            num_samples: If set, randomly sample this many EIS indices for
                quick evaluation. Overrides ``eis_indices``.

        Returns:
            DataFrame containing evaluation results for all EIS
        """
        # Switch to appropriate environment
        if use_eval_env:
            self.switch_to_eval()
        else:
            self.switch_to_training()

        dataset = self._active_env.dataset
        env_type = self.get_current_env_type()
        print(f"⚠ Evaluation is being run on the {env_type} dataset.")

        # Determine which EIS to evaluate
        if num_samples is not None:
            # Quick check mode: random sample
            all_indices = list(range(len(dataset)))
            sample_size = min(num_samples, len(all_indices))
            eis_indices = np.random.choice(
                all_indices, size=sample_size, replace=False
            ).tolist()
            print(f"\n{'=' * 80}")
            print(f"Quick Check: Evaluating {sample_size} random EIS from {env_type} dataset")
            print(f"{'=' * 80}\n")
        elif all_rows:
            # Evaluate all EIS in dataset
            eis_indices = list(range(len(dataset)))
            print(f"\n{'=' * 80}")
            print(f"Evaluating ALL EIS in {env_type} dataset ({len(eis_indices)} total)")
            print(f"{'=' * 80}\n")
        else:
            # Use provided eis_indices
            if eis_indices is None:
                raise ValueError(
                    "Must provide eis_indices, set all_rows=True, or set num_samples"
                )
            print(f"\n{'=' * 80}")
            print(f"Batch Evaluation: {len(eis_indices)} EIS measurements")
            print(f"{'=' * 80}\n")

        results = []
        start_time = time.time()

        for idx, eis_i in enumerate(tqdm(eis_indices, desc="Evaluating EIS")):
            try:
                result = self.generate_ECM(
                    EIS_i=eis_i,
                    max_actions=max_actions,
                    verbose=verbose,
                    use_eval_env=use_eval_env,
                    gif_generation=gif_generation,
                )
                results.append(
                    {
                        "EIS_i": result["EIS_i"],
                        "environment_type": result["environment_type"],
                        # 'ground_truth_state': result['ground_truth_state'],
                        "ground_truth_circuit": result["ground_truth_circuit"],
                        "found_solution": result["found_solution"],
                        "best_reward": result["best_reward"],
                        "best_action_number": result["best_action_number"],
                        "best_circuit": result["best_circuit"],
                        "best_coding": result["best_coding"],
                        "best_metrics": result["best_metrics"],
                        "best_param": result["best_param"],
                        "total_actions": result["total_actions_taken"],
                        "action_history": result["action_history"],
                    }
                )
            except Exception as e:
                print(f"\n⚠ Error evaluating EIS {eis_i}: {e}")
                results.append(
                    {
                        "EIS_i": eis_i,
                        # 'ground_truth_state': None,
                        "ground_truth_circuit": None,
                        "found_solution": False,
                        "best_reward": -np.inf,
                        "best_action_number": -1,
                        "best_circuit": None,
                        "best_coding": None,
                        "best_metrics": None,
                        "best_param": None,
                        "total_actions": 0,
                        "action_history": [],
                    }
                )

        eval_time = time.time() - start_time
        results_df = pd.DataFrame(results)

        # Print summary statistics
        print(f"\n{'=' * 80}")
        print("Batch Evaluation Complete")
        print(f"{'=' * 80}")
        print(f"Total EIS evaluated: {len(results_df)}")
        print(
            f"Solutions found: {results_df['found_solution'].sum()} "
            f"({results_df['found_solution'].mean() * 100:.1f}%)"
        )
        print(f"Average best reward: {results_df['best_reward'].mean():.4f}")
        if results_df["found_solution"].sum() > 0:
            print(
                "Average actions to solution: "
                f"{results_df[results_df['found_solution']]['best_action_number'].mean():.1f}"
            )
        print(
            f"Evaluation time: {eval_time:.1f} seconds "
            f"({eval_time / len(results_df):.2f} s/EIS)"
        )
        print(f"{'=' * 80}\n")

        return results_df

    def eval_all_eis(
        self,
        max_actions: Optional[int] = None,
        verbose: bool = False,
        use_eval_env: bool = True,
        gif_generation: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate the trained agent on every EIS row in the selected dataset.

        This is a convenience wrapper around ``eval_batch_eis(all_rows=True)``. Use
        ``use_eval_env=True`` to evaluate all rows in the evaluation environment, or
        ``use_eval_env=False`` to evaluate all rows in the training environment.

        Args:
            max_actions: Maximum number of actions per EIS (default: self.action_cap).
            verbose: Whether to print detailed progress for each EIS.
            use_eval_env: Whether to use the evaluation environment. If False,
                use the training environment.
            gif_generation: Generate a circuit-evolution GIF for each EIS.

        Returns:
            DataFrame containing evaluation results for all EIS rows.
        """
        return self.eval_batch_eis(
            max_actions=max_actions,
            verbose=verbose,
            use_eval_env=use_eval_env,
            gif_generation=gif_generation,
            all_rows=True,
        )
