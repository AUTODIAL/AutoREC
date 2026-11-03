"""Composable spectrum selection, encoding, and transitions for progressive ECMs.

The progressive-spectrum workflow is separated into three replaceable operations:

1. A :class:`SpectrumWindowSelector` chooses the original EIS points available at a stage.
2. A :class:`SpectrumObservationEncoder` constructs a fixed-size neural-network observation.
   :class:`InterpolatingSpectrumEncoder` is the default implementation.
3. A :class:`StageTransitionPolicy` decides which ECM state starts the next stage and whether
   progression should continue.

The module deliberately has no dependency on the RL environment or agent. It prepares
immutable spectrum views and defines environment-independent transition decisions; applying
those decisions remains the responsibility of the orchestration layer.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol

import numpy as np

from autorec.data_preparation import EISDataPrep


def _readonly_copy(values, *, dtype=None) -> np.ndarray:
    """Return an owned NumPy array whose contents cannot be modified."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_spectrum(frequency, impedance) -> tuple[np.ndarray, np.ndarray]:
    """Validate and copy a source spectrum without changing its order."""
    frequency = np.asarray(frequency, dtype=float)
    impedance = np.asarray(impedance, dtype=complex)

    if frequency.ndim != 1 or impedance.ndim != 1:
        raise ValueError("frequency and impedance must both be one-dimensional arrays")
    if len(frequency) == 0:
        raise ValueError("frequency and impedance must contain at least one point")
    if len(frequency) != len(impedance):
        raise ValueError("frequency and impedance must contain the same number of points")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("frequency values must be finite and strictly positive")
    if len(np.unique(frequency)) != len(frequency):
        raise ValueError("frequency values must be unique")
    if not np.all(np.isfinite(impedance.real)) or not np.all(np.isfinite(impedance.imag)):
        raise ValueError("impedance values must be finite")

    return np.array(frequency, copy=True), np.array(impedance, copy=True)


class SpectrumSource(Protocol):
    """Interface for resolving a physical spectrum from one dataset row."""

    name: str

    def read(self, row) -> tuple[np.ndarray, np.ndarray]:
        """Return validated frequency and impedance arrays from ``row``."""
        ...


@dataclass(frozen=True)
class DataFrameSpectrumSource:
    """Resolve a physical spectrum from named columns in a dataset row.

    Parameters
    ----------
    name : str
        Stable source name used in runtime metadata.
    frequency_column : str
        Dataset column containing one-dimensional frequency arrays.
    impedance_column : str
        Dataset column containing aligned complex impedance arrays.
    """

    name: str
    frequency_column: str
    impedance_column: str

    def __post_init__(self):
        for attribute in ("name", "frequency_column", "impedance_column"):
            value = getattr(self, attribute)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{attribute} must be a non-empty string")

    def read(self, row) -> tuple[np.ndarray, np.ndarray]:
        """Read and validate this source from a pandas-like dataset row.

        Parameters
        ----------
        row : Mapping or pandas.Series
            Dataset row containing ``frequency_column`` and ``impedance_column``.

        Returns
        -------
        tuple of numpy.ndarray
            Owned one-dimensional frequency and complex-impedance arrays. Their source order
            is preserved.

        Raises
        ------
        ValueError
            If either configured column is absent or the arrays are empty, misaligned,
            nonfinite, nonpositive in frequency, or contain duplicate frequencies.
        """
        missing = [
            column
            for column in (self.frequency_column, self.impedance_column)
            if column not in row
        ]
        if missing:
            raise ValueError(
                f"spectrum source {self.name!r} requires dataset columns: "
                + ", ".join(missing)
            )
        return _validate_spectrum(row[self.frequency_column], row[self.impedance_column])


PREPARED_SPECTRUM_SOURCE = DataFrameSpectrumSource(
    name="prepared",
    frequency_column="freq",
    impedance_column="Z_true",
)
RETAINED_SPECTRUM_SOURCE = DataFrameSpectrumSource(
    name="retained",
    frequency_column="source_freq",
    impedance_column="source_Z",
)


@dataclass(frozen=True)
class SpectrumWindowSelection:
    """Source EIS points included in one progressive stage.

    A selection describes source data before fixed-size interpolation.  Arrays are copied
    when the object is created and exposed as read-only arrays so that later processing
    cannot accidentally modify the original spectrum or the recorded selection.

    Attributes
    ----------
    stage_index : int
        Zero-based progressive-stage index that produced the selection.
    included_window_indices : tuple of int
        Window indices included at this stage.  A cumulative selector normally returns
        ``(0,)`` for stage 0, ``(0, 1)`` for stage 1, and so on.
    source_indices : numpy.ndarray
        One-dimensional integer indices into the input frequency and impedance arrays, in
        the order chosen by the selector.
    frequency : numpy.ndarray
        Read-only one-dimensional array of selected frequencies in hertz.
    impedance : numpy.ndarray
        Read-only one-dimensional complex array of selected impedance values, aligned with
        ``frequency``.
    source : SpectrumSource
        Dataset source that supplied the complete spectrum from which this selection was made.
    """

    stage_index: int
    included_window_indices: tuple[int, ...]
    source_indices: np.ndarray
    frequency: np.ndarray
    impedance: np.ndarray
    source: SpectrumSource = PREPARED_SPECTRUM_SOURCE

    def __post_init__(self):
        object.__setattr__(
            self, "source_indices", _readonly_copy(self.source_indices, dtype=int)
        )
        object.__setattr__(self, "frequency", _readonly_copy(self.frequency, dtype=float))
        object.__setattr__(self, "impedance", _readonly_copy(self.impedance, dtype=complex))


@dataclass(frozen=True)
class SpectrumView:
    """A selected spectrum and its fixed-size encoded representation.

    Arrays are copied when the view is created and exposed as read-only arrays. Circuit
    fitting uses the source points in ``selection``. The ``frequency`` and ``impedance``
    arrays record the values represented in the neural-network observation; they are not
    fitting targets and need not have the same length as the physical selection.

    Attributes
    ----------
    selection : SpectrumWindowSelection
        Raw source points used for circuit fitting and stage metadata from the window
        selector.
    frequency : numpy.ndarray
        One-dimensional frequency values represented by the encoder. Interpolating encoders
        normally provide their complete fixed-size grid, while padding encoders may provide
        only the valid, unpadded values.
    impedance : numpy.ndarray
        One-dimensional complex impedance values aligned with ``frequency`` and used for
        observation construction.
    observation : numpy.ndarray
        Flattened, normalized EIS features produced by
        ``EISDataPrep.flatten_EIS``. Its length is
        ``frequency_npoints * len(eis_features)``.
    observation_valid_mask : numpy.ndarray or None
        Optional one-dimensional boolean mask over the encoder's fixed point positions. This
        metadata is not appended to ``observation``. When provided, its number of true values
        must equal the number of values recorded by ``frequency`` and ``impedance``.
    observation_source_indices : numpy.ndarray or None
        Optional indices into the original source spectrum for observation points that directly
        preserve source measurements. Interpolated points have no corresponding source indices.
    """

    selection: SpectrumWindowSelection
    frequency: np.ndarray
    impedance: np.ndarray
    observation: np.ndarray
    observation_valid_mask: np.ndarray | None = None
    observation_source_indices: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "frequency", _readonly_copy(self.frequency, dtype=float))
        object.__setattr__(self, "impedance", _readonly_copy(self.impedance, dtype=complex))
        object.__setattr__(self, "observation", _readonly_copy(self.observation))

        if self.frequency.ndim != 1 or self.impedance.ndim != 1:
            raise ValueError("encoded frequency and impedance must be one-dimensional")
        if len(self.frequency) == 0 or len(self.frequency) != len(self.impedance):
            raise ValueError(
                "encoded frequency and impedance must contain the same nonzero number of points"
            )
        if self.observation.ndim != 1 or len(self.observation) == 0:
            raise ValueError("observation must be a nonempty one-dimensional array")

        if self.observation_valid_mask is not None:
            mask = _readonly_copy(self.observation_valid_mask, dtype=bool)
            if mask.ndim != 1:
                raise ValueError("observation_valid_mask must be one-dimensional")
            if np.count_nonzero(mask) != len(self.frequency):
                raise ValueError(
                    "observation_valid_mask must identify each encoded frequency exactly once"
                )
            object.__setattr__(self, "observation_valid_mask", mask)

        if self.observation_source_indices is not None:
            source_indices = _readonly_copy(self.observation_source_indices, dtype=int)
            if source_indices.ndim != 1 or len(source_indices) != len(self.frequency):
                raise ValueError(
                    "observation_source_indices must align with encoded frequency and impedance"
                )
            if not np.all(np.isin(source_indices, self.selection.source_indices)):
                raise ValueError(
                    "observation_source_indices must refer to points in the physical selection"
                )
            object.__setattr__(self, "observation_source_indices", source_indices)


class SpectrumObservationEncoder(Protocol):
    """Interface for replaceable fixed-size spectrum observation encoders.

    Encoders may interpolate, pad, or downsample their neural-network representation, but they
    must retain the complete physical fitting data in ``SpectrumView.selection``.
    """

    output_npoints: int
    eis_features: tuple[str, ...]
    observation_size: int

    def encode(self, selection: SpectrumWindowSelection) -> SpectrumView:
        """Encode one physical spectrum selection as a fixed-size view."""
        ...


class SpectrumFeatureNormalizer(Protocol):
    """Interface for converting impedance values into a feature-major matrix.

    Implementations receive all physical points available in the current stage. They do not
    pad or downsample; :class:`RawSpectrumEncoder` performs those operations afterward.
    """

    def normalize(
        self,
        impedance: np.ndarray,
        eis_features: tuple[str, ...],
    ) -> np.ndarray:
        """Normalize one stage's physical impedance measurements.

        Parameters
        ----------
        impedance : numpy.ndarray
            One-dimensional complex measurements from the complete physical stage selection.
        eis_features : tuple of str
            Ordered feature names defining the rows of the result.

        Returns
        -------
        numpy.ndarray
            Finite values with shape ``(len(eis_features), len(impedance))``.
        """
        ...


class SpectrumDownsampler(Protocol):
    """Interface for selecting original points represented in a fixed-size observation.

    Downsamplers affect only neural-network input. The complete physical selection remains in
    :attr:`SpectrumView.selection` for ECM fitting and reward evaluation.
    """

    def select_indices(
        self,
        frequency: np.ndarray,
        impedance: np.ndarray,
        capacity: int,
    ) -> np.ndarray:
        """Choose original measurements for the fixed-capacity observation.

        Parameters
        ----------
        frequency, impedance : numpy.ndarray
            Aligned one-dimensional arrays for every point in the current physical selection.
        capacity : int
            Maximum number of point positions accepted by the neural-network observation.

        Returns
        -------
        numpy.ndarray
            Unique, increasing integer indices into ``frequency`` and ``impedance``.
        """
        ...


@dataclass(frozen=True)
class UniformIndexDownsampler:
    """Select deterministic, uniformly spaced source indices while preserving endpoints.

    Spectra at or below ``capacity`` retain every point in their existing order. Larger spectra
    select exactly ``capacity`` original measurements, including the first and last points.
    """

    def select_indices(
        self,
        frequency: np.ndarray,
        impedance: np.ndarray,
        capacity: int,
    ) -> np.ndarray:
        """Return all indices or uniformly spaced endpoint-preserving indices.

        Parameters
        ----------
        frequency, impedance : numpy.ndarray
            Aligned source measurements. Values are validated but never interpolated.
        capacity : int
            Positive maximum number of returned positions.

        Returns
        -------
        numpy.ndarray
            ``arange(npoints)`` when the data fit, otherwise exactly ``capacity`` uniformly
            spaced indices including zero and ``npoints - 1``.
        """
        frequency, impedance = _validate_spectrum(frequency, impedance)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        selected_count = min(len(frequency), capacity)
        return np.linspace(0, len(frequency) - 1, selected_count, dtype=int)


@dataclass(frozen=True)
class SentinelSafeMinMaxNormalizer:
    """Normalize every configured raw EIS feature independently to ``[0, 1]``.

    Negative features are constructed from the literal negative physical quantity before
    normalization. Constant features map to zero. This keeps ``-1`` available exclusively for
    padding without changing the legacy
    :meth:`~autorec.data_preparation.EISDataPrep.flatten_EIS` implementation.
    """

    def normalize(
        self,
        impedance: np.ndarray,
        eis_features: tuple[str, ...],
    ) -> np.ndarray:
        """Normalize each requested feature over the supplied stage.

        Parameters
        ----------
        impedance : numpy.ndarray
            Complete one-dimensional physical impedance selection for the current stage.
        eis_features : tuple of str
            Ordered feature names supported by
            :meth:`~autorec.data_preparation.EISDataPrep.flatten_EIS`.

        Returns
        -------
        numpy.ndarray
            Feature-major values in ``[0, 1]`` with shape ``(features, points)``. A constant
            feature is represented by zeros.

        Raises
        ------
        ValueError
            If measurements are empty, nonfinite, not one-dimensional, or a feature name is
            unsupported.
        """
        impedance = np.asarray(impedance, dtype=complex)
        if impedance.ndim != 1 or len(impedance) == 0:
            raise ValueError("impedance must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(impedance.real)) or not np.all(np.isfinite(impedance.imag)):
            raise ValueError("impedance values must be finite")

        magnitude = np.abs(impedance)
        values = {
            "ReZ": impedance.real,
            "ImZ": impedance.imag,
            "phi": np.angle(impedance),
            "mag": magnitude,
            "nReZ": -impedance.real,
            "nImZ": -impedance.imag,
            "nphi": -np.angle(impedance),
            "nmag": -magnitude,
        }
        unknown_features = [feature for feature in eis_features if feature not in values]
        if unknown_features:
            raise ValueError(f"unsupported EIS features: {unknown_features}")

        normalized_features = []
        for feature in eis_features:
            feature_values = np.asarray(values[feature], dtype=float)
            minimum = np.min(feature_values)
            value_range = np.max(feature_values) - minimum
            normalized = (
                np.zeros_like(feature_values)
                if value_range == 0
                else (feature_values - minimum) / value_range
            )
            normalized_features.append(np.clip(normalized, 0.0, 1.0))
        return np.stack(normalized_features)


@dataclass(frozen=True)
class LegacyEISFeatureNormalizer:
    """Reproduce the existing ``EISDataPrep.flatten_EIS`` feature representation.

    This compatibility normalizer may produce valid negative values, including ``-1``, and
    therefore does not make the padding sentinel unambiguous. It is intended for reproducing
    existing full-capacity observations, not as the default raw-padding representation.
    """

    def normalize(
        self,
        impedance: np.ndarray,
        eis_features: tuple[str, ...],
    ) -> np.ndarray:
        """Return the legacy flattened values reshaped into feature-major form.

        Parameters
        ----------
        impedance : numpy.ndarray
            One-dimensional physical impedance selection for the current stage.
        eis_features : tuple of str
            Ordered feature names passed unchanged to
            :meth:`~autorec.data_preparation.EISDataPrep.flatten_EIS`.

        Returns
        -------
        numpy.ndarray
            Legacy values with shape ``(features, points)``. Unlike
            :class:`SentinelSafeMinMaxNormalizer`, valid values are not guaranteed to exclude
            the ``-1`` padding sentinel.
        """
        impedance = np.asarray(impedance, dtype=complex)
        flattened = EISDataPrep.flatten_EIS(impedance, list(eis_features))
        return np.asarray(flattened).reshape(len(eis_features), len(impedance))


class RawSpectrumEncoder:
    """Normalize, select, and pad original spectrum points without interpolation.

    Normalization is calculated over every physical point available in the current stage before
    the downsampler selects the values represented in the neural-network observation. The
    complete physical selection remains available for ECM fitting.

    Parameters
    ----------
    dataprep : EISDataPrep
        Supplies the fixed point capacity and ordered EIS feature set.
    normalizer : SpectrumFeatureNormalizer, optional
        Feature normalization strategy. Defaults to
        :class:`SentinelSafeMinMaxNormalizer`.
    downsampler : SpectrumDownsampler, optional
        Original-point selection strategy. Defaults to
        :class:`UniformIndexDownsampler`.
    padding_value : float, default=-1.0
        Reserved value written to every feature at each missing point position.

    Attributes
    ----------
    dataprep : EISDataPrep
        Data-preparation instance defining observation capacity and feature order.
    output_npoints : int
        Maximum number of measurement positions represented in the observation.
    eis_features : tuple of str
        Ordered features occupying contiguous blocks in the flattened observation.
    observation_size : int
        Fixed neural-network input length, ``output_npoints * len(eis_features)``.
    normalizer : SpectrumFeatureNormalizer
        Resolved feature-normalization component.
    downsampler : SpectrumDownsampler
        Resolved original-point selection component.
    padding_value : float
        Universal missing-position sentinel. Currently fixed at ``-1.0``.
    """

    def __init__(
        self,
        dataprep: EISDataPrep,
        *,
        normalizer: SpectrumFeatureNormalizer | None = None,
        downsampler: SpectrumDownsampler | None = None,
        padding_value: float = -1.0,
    ):
        if not isinstance(dataprep, EISDataPrep):
            raise TypeError("dataprep must be an EISDataPrep instance")
        if (
            isinstance(dataprep.frequency_npoints, bool)
            or not isinstance(dataprep.frequency_npoints, int)
            or dataprep.frequency_npoints <= 2
        ):
            raise ValueError("dataprep.frequency_npoints must be an integer greater than two")
        if not dataprep.eis_features:
            raise ValueError("dataprep.eis_features must contain at least one feature")
        if not np.isscalar(padding_value) or not np.isfinite(padding_value):
            raise ValueError("padding_value must be a finite scalar")
        if float(padding_value) != -1.0:
            raise ValueError("padding_value must be -1.0")

        self.dataprep = dataprep
        self.output_npoints = dataprep.frequency_npoints
        self.eis_features = tuple(dataprep.eis_features)
        self.observation_size = self.output_npoints * len(self.eis_features)
        self.normalizer = SentinelSafeMinMaxNormalizer() if normalizer is None else normalizer
        self.downsampler = UniformIndexDownsampler() if downsampler is None else downsampler
        self.padding_value = float(padding_value)

    def encode(self, selection: SpectrumWindowSelection) -> SpectrumView:
        """Encode original points with stage-local normalization and trailing padding.

        Parameters
        ----------
        selection : SpectrumWindowSelection
            Complete physical stage selection. All of its impedance values contribute to
            normalization and remain available for ECM fitting.

        Returns
        -------
        SpectrumView
            Fixed-size, feature-major observation plus provenance for the original points
            represented in it. Padding is trailing and is not included in ``frequency`` or
            ``impedance``.

        Raises
        ------
        TypeError
            If ``selection`` is not a :class:`SpectrumWindowSelection`.
        ValueError
            If a component returns an invalid shape, index set, or nonfinite values.
        """
        if not isinstance(selection, SpectrumWindowSelection):
            raise TypeError("selection must be a SpectrumWindowSelection")
        npoints = len(selection.frequency)

        normalized = np.asarray(
            self.normalizer.normalize(selection.impedance, self.eis_features),
            dtype=float,
        )
        expected_shape = (len(self.eis_features), npoints)
        if normalized.shape != expected_shape:
            raise ValueError(
                "normalizer output shape does not match configured features and selected points: "
                f"expected {expected_shape}, got {normalized.shape}"
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("normalizer produced non-finite observation values")

        observation_indices = np.asarray(
            self.downsampler.select_indices(
                selection.frequency,
                selection.impedance,
                self.output_npoints,
            )
        )
        expected_count = min(npoints, self.output_npoints)
        if observation_indices.shape != (expected_count,):
            raise ValueError(
                "downsampler must return one-dimensional indices matching the available "
                f"observation capacity: expected {expected_count}, got "
                f"{observation_indices.shape}"
            )
        if not np.issubdtype(observation_indices.dtype, np.integer):
            raise ValueError("downsampler indices must be integers")
        if (
            np.any(observation_indices < 0)
            or np.any(observation_indices >= npoints)
            or len(np.unique(observation_indices)) != len(observation_indices)
            or np.any(np.diff(observation_indices) <= 0)
        ):
            raise ValueError(
                "downsampler indices must be unique, in range, and preserve source order"
            )

        selected_features = normalized[:, observation_indices]
        selected_count = len(observation_indices)

        feature_matrix = np.full(
            (len(self.eis_features), self.output_npoints),
            self.padding_value,
            dtype=float,
        )
        feature_matrix[:, :selected_count] = selected_features
        observation = feature_matrix.reshape(-1)
        valid_mask = np.zeros(self.output_npoints, dtype=bool)
        valid_mask[:selected_count] = True

        return SpectrumView(
            selection=selection,
            frequency=selection.frequency[observation_indices],
            impedance=selection.impedance[observation_indices],
            observation=observation,
            observation_valid_mask=valid_mask,
            observation_source_indices=selection.source_indices[observation_indices],
        )


@dataclass(frozen=True)
class ProgressiveStageOutcome:
    """Information available to a transition policy after one completed stage.

    The outcome contains only orchestration-level values shared by training and evaluation.
    Method-specific information can be exposed through ``metadata`` without making the
    policy interface depend on an RL environment or agent implementation.

    Parameters
    ----------
    stage_index : int
        Zero-based index of the stage that just completed.
    num_stages : int
        Total number of configured progressive stages.
    initial_state : str
        Chromosome used to start the completed stage.
    best_state : str
        Preferred chromosome produced by the stage. AutoREC supplies a satisfactory
        terminal chromosome when one was found; otherwise it supplies the highest-reward
        chromosome.
    best_reward : float
        Reward associated with ``best_state``.
    found_solution : bool
        Whether the completed stage found a satisfactory ECM.
    metadata : mapping, optional
        Read-only-by-convention method-specific context. The default policy does not inspect
        it. Training and evaluation may expose different keys.

    Attributes
    ----------
    stage_index, num_stages, initial_state, best_state, best_reward, found_solution, metadata
        Values supplied at construction.
    """

    stage_index: int
    num_stages: int
    initial_state: str
    best_state: str
    best_reward: float
    found_solution: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.stage_index, bool) or not isinstance(self.stage_index, int):
            raise TypeError("stage_index must be an integer")
        if isinstance(self.num_stages, bool) or not isinstance(self.num_stages, int):
            raise TypeError("num_stages must be an integer")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be greater than zero")
        if not 0 <= self.stage_index < self.num_stages:
            raise ValueError("stage_index must identify a configured stage")
        if not isinstance(self.initial_state, str) or not self.initial_state:
            raise ValueError("initial_state must be a non-empty chromosome string")
        if not isinstance(self.best_state, str) or not self.best_state:
            raise ValueError("best_state must be a non-empty chromosome string")
        if not isinstance(self.found_solution, bool):
            raise TypeError("found_solution must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@dataclass(frozen=True)
class StageTransitionDecision:
    """A policy decision controlling the transition to the next spectrum stage.

    Parameters
    ----------
    next_state : str or None
        Chromosome to use when the next stage starts. It is required when
        ``continue_to_next_stage`` is true and may be omitted when progression stops.
    continue_to_next_stage : bool
        Whether the orchestrator should run the next configured stage.

    Attributes
    ----------
    next_state : str or None
        Selected next-stage chromosome.
    continue_to_next_stage : bool
        Selected continuation behavior.
    """

    next_state: str | None
    continue_to_next_stage: bool

    def __post_init__(self):
        if not isinstance(self.continue_to_next_stage, bool):
            raise TypeError("continue_to_next_stage must be a boolean")
        if self.next_state is not None and (
            not isinstance(self.next_state, str) or not self.next_state
        ):
            raise ValueError("next_state must be None or a non-empty chromosome string")
        if self.continue_to_next_stage and self.next_state is None:
            raise ValueError("next_state is required when progression continues")


class StageTransitionPolicy(Protocol):
    """Interface for replaceable progressive-stage transition strategies."""

    def decide(self, outcome: ProgressiveStageOutcome) -> StageTransitionDecision:
        """Choose the next chromosome and whether another stage should run.

        Parameters
        ----------
        outcome : ProgressiveStageOutcome
            Result and context from the stage that just completed.

        Returns
        -------
        StageTransitionDecision
            Next-state and continuation decision.
        """
        ...


@dataclass(frozen=True)
class CarryBestStatePolicy:
    """Continue through every configured stage using the preceding best ECM state.

    This is AutoREC's compatibility policy. The orchestrator defines ``best_state`` as the
    satisfactory terminal state when a solution was found and otherwise as the
    highest-reward state. Consequently, this policy preserves the progressive behavior that
    preceded the transition interface and introduces no new scientific selection rule.
    """

    def decide(self, outcome: ProgressiveStageOutcome) -> StageTransitionDecision:
        """Carry ``outcome.best_state`` into the next configured stage."""
        return StageTransitionDecision(
            next_state=outcome.best_state,
            continue_to_next_stage=True,
        )


def decide_stage_transition(
    policy: StageTransitionPolicy,
    outcome: ProgressiveStageOutcome,
) -> StageTransitionDecision:
    """Execute and validate a transition policy decision.

    Parameters
    ----------
    policy : StageTransitionPolicy
        Policy to evaluate.
    outcome : ProgressiveStageOutcome
        Completed-stage context supplied to the policy.

    Returns
    -------
    StageTransitionDecision
        Validated policy decision.

    Raises
    ------
    TypeError
        If the policy does not return :class:`StageTransitionDecision`.
    """
    decision = policy.decide(outcome)
    if not isinstance(decision, StageTransitionDecision):
        raise TypeError("transition policy must return a StageTransitionDecision")
    return decision


class SpectrumWindowSelector(Protocol):
    """Interface for replaceable progressive-spectrum selection strategies.

    Implementations decide how source points are grouped and which groups are available at
    each stage.  They must return a :class:`SpectrumWindowSelection`; the encoder does not
    depend on a particular grouping method.
    """

    @property
    def num_windows(self) -> int:
        """Total number of selectable windows and progressive stages."""
        ...

    def select(
        self, frequency: np.ndarray, impedance: np.ndarray, stage_index: int
    ) -> SpectrumWindowSelection:
        """Select the source spectrum available at a progressive stage.

        Parameters
        ----------
        frequency : numpy.ndarray
            One-dimensional source frequency array.
        impedance : numpy.ndarray
            One-dimensional complex impedance array aligned with ``frequency``.
        stage_index : int
            Zero-based stage to select.

        Returns
        -------
        SpectrumWindowSelection
            Selected source points and the associated stage metadata.
        """
        ...


@dataclass(frozen=True)
class EqualPointWindowSelector:
    """Split EIS points into cumulative, approximately equal frequency windows.

    For multiple windows, points are first ordered from highest to lowest frequency. Window 0
    therefore contains the highest-frequency points, and stage ``i`` contains windows ``0``
    through ``i``. ``numpy.array_split`` assigns any remainder to the earlier,
    higher-frequency windows. A single window preserves source order so a complete prepared
    spectrum can pass through a :class:`SpectrumView` without changing its representation.

    Parameters
    ----------
    num_windows : int
        Total number of windows.  This is also the number of valid stages.  It must be
        positive and cannot exceed the number of source spectrum points supplied to
        :meth:`select`.

    Attributes
    ----------
    num_windows : int
        Configured total number of windows.

    Notes
    -----
    This selector defines an initial exploration strategy, not a scientific requirement.
    Alternative selectors can implement :class:`SpectrumWindowSelector` without changing
    the encoder.

    The encoder requires at least two points in a selection.  Consequently, a window count
    that leaves only one point in window 0 cannot be encoded at stage 0.

    Raises
    ------
    TypeError
        If ``num_windows`` is not an integer.
    ValueError
        If ``num_windows`` is not positive.
    """

    num_windows: int

    def __post_init__(self):
        if isinstance(self.num_windows, bool) or not isinstance(self.num_windows, int):
            raise TypeError("num_windows must be an integer")
        if self.num_windows <= 0:
            raise ValueError("num_windows must be greater than zero")

    def select(
        self, frequency: np.ndarray, impedance: np.ndarray, stage_index: int
    ) -> SpectrumWindowSelection:
        """Select cumulative equal-point windows through ``stage_index``.

        Parameters
        ----------
        frequency : numpy.ndarray
            One-dimensional source frequencies.  Values must be finite, unique, and
            strictly positive.  Input order does not matter.
        impedance : numpy.ndarray
            One-dimensional finite complex impedance values aligned with ``frequency``.
        stage_index : int
            Zero-based final window to include.  For example, stage 0 includes window 0,
            while stage 2 includes windows 0, 1, and 2.

        Returns
        -------
        SpectrumWindowSelection
            A read-only selection ordered from highest to lowest frequency.

        Raises
        ------
        TypeError
            If ``stage_index`` is not an integer.
        ValueError
            If the arrays are invalid or misaligned, ``num_windows`` exceeds the number of
            points, or ``stage_index`` is outside ``[0, num_windows)``.
        """
        frequency, impedance = _validate_spectrum(frequency, impedance)

        if self.num_windows > len(frequency):
            raise ValueError("num_windows cannot exceed the number of spectrum points")
        if isinstance(stage_index, bool) or not isinstance(stage_index, int):
            raise TypeError("stage_index must be an integer")
        if not 0 <= stage_index < self.num_windows:
            raise ValueError(f"stage_index must be between 0 and {self.num_windows - 1}")

        # With a single window, preserve the prepared spectrum exactly so routing a full
        # spectrum through SpectrumView is an identity operation. Multiple-window selection
        # still defines window 0 by the highest frequencies, independent of source order.
        if self.num_windows == 1:
            descending_indices = np.arange(len(frequency))
        else:
            descending_indices = np.argsort(-frequency, kind="stable")
        windows = np.array_split(descending_indices, self.num_windows)
        selected_indices = np.concatenate(windows[: stage_index + 1])

        return SpectrumWindowSelection(
            stage_index=stage_index,
            included_window_indices=tuple(range(stage_index + 1)),
            source_indices=selected_indices,
            frequency=frequency[selected_indices],
            impedance=impedance[selected_indices],
        )


class InterpolatingSpectrumEncoder:
    """Interpolate a selection and flatten it using ``EISDataPrep`` settings.

    ``frequency_npoints`` fixes the number of interpolated points, and ``eis_features`` fixes
    the feature order.  Both are read from the supplied data-preparation instance so every
    stage has the same observation dimension as its prepared dataset.

    Partial selections are interpolated onto a descending logarithmic grid bounded by the
    selected frequencies.  The encoder does not extrapolate.  When a selection already has
    ``frequency_npoints`` points, its frequency grid is preserved exactly; this keeps the
    complete-spectrum representation unchanged, including for an irregular prepared grid.

    Parameters
    ----------
    dataprep : EISDataPrep
        Data-preparation instance defining interpolation point count and EIS feature order.
        Object construction is performed in Python; this instance is not expected to be
        represented directly in YAML.

    Attributes
    ----------
    dataprep : EISDataPrep
        Data-preparation instance used for interpolation and flattening.
    output_npoints : int
        Fixed number of frequency and impedance points in every encoded view.  Copied from
        ``dataprep.frequency_npoints`` during initialization.
    eis_features : tuple of str
        Ordered EIS features used to construct the observation.  Copied from
        ``dataprep.eis_features`` during initialization.
    observation_size : int
        Expected flattened observation length, equal to
        ``output_npoints * len(eis_features)``.

    Raises
    ------
    TypeError
        If ``dataprep`` is not an
        :class:`~autorec.data_preparation.EISDataPrep` instance.
    ValueError
        If ``frequency_npoints`` is not an integer greater than two or no EIS features are
        configured.
    """

    def __init__(self, dataprep: EISDataPrep):
        if not isinstance(dataprep, EISDataPrep):
            raise TypeError("dataprep must be an EISDataPrep instance")
        if (
            isinstance(dataprep.frequency_npoints, bool)
            or not isinstance(dataprep.frequency_npoints, int)
            or dataprep.frequency_npoints <= 2
        ):
            raise ValueError("dataprep.frequency_npoints must be an integer greater than two")
        if not dataprep.eis_features:
            raise ValueError("dataprep.eis_features must contain at least one feature")

        self.dataprep = dataprep
        self.output_npoints = dataprep.frequency_npoints
        self.eis_features = tuple(dataprep.eis_features)
        self.observation_size = self.output_npoints * len(self.eis_features)

    def encode(self, selection: SpectrumWindowSelection) -> SpectrumView:
        """Interpolate and flatten one selected progressive spectrum.

        Parameters
        ----------
        selection : SpectrumWindowSelection
            Source points and stage metadata produced by a compatible window selector.  At
            least two selected frequency points are required for cubic interpolation.

        Returns
        -------
        SpectrumView
            Read-only interpolated frequency and impedance arrays together with their
            fixed-size flattened observation.

        Raises
        ------
        TypeError
            If ``selection`` is not a :class:`SpectrumWindowSelection`.
        ValueError
            If fewer than two points are selected, the encoded length differs from
            ``observation_size``, or interpolation or flattening produces non-finite values.
        """
        if not isinstance(selection, SpectrumWindowSelection):
            raise TypeError("selection must be a SpectrumWindowSelection")
        if len(selection.frequency) < 2:
            raise ValueError("at least two selected frequency points are required")

        if len(selection.frequency) == self.output_npoints:
            # Preserve an already fixed-size full spectrum exactly, including an irregular
            # prepared frequency grid. No interpolation is needed on the identity grid.
            frequency = np.array(selection.frequency, copy=True)
            impedance = np.array(selection.impedance, copy=True)
        else:
            frequency = np.geomspace(
                np.max(selection.frequency),
                np.min(selection.frequency),
                self.output_npoints,
            )
            impedance = self.dataprep.interpolate_EIS(
                selection.frequency, selection.impedance, frequency
            )
        observation = self.dataprep.flatten_EIS(impedance, list(self.eis_features))

        if len(observation) != self.observation_size:
            raise ValueError(
                "encoded observation size does not match the EISDataPrep settings: "
                f"expected {self.observation_size}, got {len(observation)}"
            )
        if not np.all(np.isfinite(impedance.real)) or not np.all(np.isfinite(impedance.imag)):
            raise ValueError("interpolation produced non-finite impedance values")
        if not np.all(np.isfinite(observation)):
            raise ValueError("flattening produced non-finite observation values")

        preserves_source_points = len(selection.frequency) == self.output_npoints
        return SpectrumView(
            selection=selection,
            frequency=frequency,
            impedance=impedance,
            observation=observation,
            observation_valid_mask=np.ones(self.output_npoints, dtype=bool),
            observation_source_indices=(
                selection.source_indices if preserves_source_points else None
            ),
        )


def build_spectrum_view(
    frequency: np.ndarray,
    impedance: np.ndarray,
    stage_index: int,
    selector: SpectrumWindowSelector,
    encoder: SpectrumObservationEncoder,
    spectrum_source: SpectrumSource = PREPARED_SPECTRUM_SOURCE,
) -> SpectrumView:
    """Select and encode one progressive stage using replaceable components.

    This convenience function composes the selector and encoder interfaces without coupling
    either component to an RL environment.

    Parameters
    ----------
    frequency : numpy.ndarray
        One-dimensional source frequency array.
    impedance : numpy.ndarray
        One-dimensional complex impedance array aligned with ``frequency``.
    stage_index : int
        Zero-based progressive stage passed to ``selector``.
    selector : SpectrumWindowSelector
        Strategy used to choose the source points available at this stage.
    encoder : SpectrumObservationEncoder
        Component used to construct the fixed-size neural-network observation.
    spectrum_source : SpectrumSource
        Dataset source associated with ``frequency`` and ``impedance``. Defaults to the legacy
        prepared ``freq`` and ``Z_true`` columns.

    Returns
    -------
    SpectrumView
        Fixed-size spectrum view and observation for the requested stage.

    Raises
    ------
    TypeError
        If an argument violates the selected component's type requirements.
    ValueError
        If selection or encoding validation fails.
    """
    selection = selector.select(frequency, impedance, stage_index)
    return encoder.encode(replace(selection, source=spectrum_source))


@dataclass(frozen=True)
class ProgressiveSpectrumRuntimeConfig:
    """Resolved progressive-spectrum components attached to an AutoREC agent.

    YAML configuration is converted into this runtime object by the factory after data
    preparation and environment construction. Keeping instantiated Python components here
    avoids attempting to serialize ``EISDataPrep`` or encoder instances in YAML.

    Parameters
    ----------
    enabled : bool
        Whether configured training and batch evaluation should use spectrum stages.
    num_windows : int
        Number of cumulative spectrum windows.
    selector : SpectrumWindowSelector or None
        Resolved source-window selector. Required when enabled.
    training_encoder : SpectrumObservationEncoder or None
        Encoder bound to the training environment's data preparation. Required when enabled.
    evaluation_encoder : SpectrumObservationEncoder or None
        Encoder bound to the evaluation environment's data preparation. If there is no
        distinct evaluation environment, this is the same object as ``training_encoder``.
    transition_policy : StageTransitionPolicy or None
        Resolved stage-transition policy. Required when enabled.
    spectrum_source : SpectrumSource
        Physical dataset source used for progressive selection and fitting.
    """

    enabled: bool
    num_windows: int
    selector: SpectrumWindowSelector | None = None
    training_encoder: SpectrumObservationEncoder | None = None
    evaluation_encoder: SpectrumObservationEncoder | None = None
    transition_policy: StageTransitionPolicy | None = None
    spectrum_source: SpectrumSource = PREPARED_SPECTRUM_SOURCE

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if isinstance(self.num_windows, bool) or not isinstance(self.num_windows, int):
            raise TypeError("num_windows must be an integer")
        if self.num_windows <= 0:
            raise ValueError("num_windows must be greater than zero")
        if self.enabled and any(
            component is None
            for component in (
                self.selector,
                self.training_encoder,
                self.evaluation_encoder,
                self.transition_policy,
            )
        ):
            raise ValueError(
                "enabled progressive configuration requires all runtime components"
            )

    def encoder_for(self, use_eval_env: bool) -> SpectrumObservationEncoder:
        """Return the encoder associated with the selected agent environment."""
        encoder = self.evaluation_encoder if use_eval_env else self.training_encoder
        if encoder is None:
            raise ValueError("progressive spectrum encoding is not enabled")
        return encoder

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a serializable summary of the resolved configuration."""
        return {
            "enabled": self.enabled,
            "num_windows": self.num_windows,
            "window_selector": (
                type(self.selector).__name__ if self.selector is not None else None
            ),
            "observation_encoder": (
                type(self.training_encoder).__name__
                if self.training_encoder is not None
                else None
            ),
            "stage_transition_policy": (
                type(self.transition_policy).__name__
                if self.transition_policy is not None
                else None
            ),
            "spectrum_source": self.spectrum_source.name,
        }


def _component_state(component, supported: dict[type, tuple[str, dict]]) -> dict:
    """Return a JSON-compatible state for one supported built-in component."""
    try:
        component_type, parameters = supported[type(component)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported checkpoint component: {type(component).__name__}"
        ) from exc
    return {"type": component_type, "parameters": parameters}


def _encoder_state(encoder: SpectrumObservationEncoder) -> dict:
    """Return the checkpoint specification for a built-in observation encoder."""
    if type(encoder) is InterpolatingSpectrumEncoder:
        return {"type": "InterpolatingSpectrumEncoder", "parameters": {}}
    if type(encoder) is RawSpectrumEncoder:
        normalizer = _component_state(
            encoder.normalizer,
            {
                SentinelSafeMinMaxNormalizer: ("SentinelSafeMinMaxNormalizer", {}),
                LegacyEISFeatureNormalizer: ("LegacyEISFeatureNormalizer", {}),
            },
        )
        downsampler = _component_state(
            encoder.downsampler,
            {UniformIndexDownsampler: ("UniformIndexDownsampler", {})},
        )
        return {
            "type": "RawSpectrumEncoder",
            "parameters": {
                "padding_value": encoder.padding_value,
                "normalizer": normalizer,
                "downsampler": downsampler,
            },
        }
    raise ValueError(f"unsupported checkpoint encoder: {type(encoder).__name__}")


def progressive_runtime_state_dict(
    config: ProgressiveSpectrumRuntimeConfig | None,
    *,
    training_dataprep: EISDataPrep | None = None,
    evaluation_dataprep: EISDataPrep | None = None,
) -> dict | None:
    """Serialize a resolved built-in progressive runtime without pickling components.

    The returned mapping is JSON-compatible and records component types and constructor
    parameters. Encoder ``EISDataPrep`` objects remain environment-owned and are rebound when
    the checkpoint is restored. Unsupported custom components fail explicitly so a checkpoint
    cannot silently restore different scientific behavior.
    """
    if config is None:
        return None
    if not isinstance(config, ProgressiveSpectrumRuntimeConfig):
        raise TypeError("config must be a ProgressiveSpectrumRuntimeConfig or None")
    state = {
        "version": 1,
        "enabled": config.enabled,
        "num_windows": config.num_windows,
        "selector": None,
        "training_encoder": None,
        "evaluation_encoder": None,
        "transition_policy": None,
        "spectrum_source": None,
    }
    if not config.enabled:
        return state
    if getattr(config.training_encoder, "dataprep", None) is not training_dataprep:
        raise ValueError("training encoder is not bound to the training environment dataprep")
    if getattr(config.evaluation_encoder, "dataprep", None) is not evaluation_dataprep:
        raise ValueError(
            "evaluation encoder is not bound to the evaluation environment dataprep"
        )
    state.update(
        selector=_component_state(
            config.selector,
            {
                EqualPointWindowSelector: (
                    "EqualPointWindowSelector",
                    {"num_windows": config.selector.num_windows},
                )
            },
        ),
        training_encoder=_encoder_state(config.training_encoder),
        evaluation_encoder=_encoder_state(config.evaluation_encoder),
        transition_policy=_component_state(
            config.transition_policy,
            {CarryBestStatePolicy: ("CarryBestStatePolicy", {})},
        ),
        spectrum_source=_component_state(
            config.spectrum_source,
            {
                DataFrameSpectrumSource: (
                    "DataFrameSpectrumSource",
                    {
                        "name": config.spectrum_source.name,
                        "frequency_column": config.spectrum_source.frequency_column,
                        "impedance_column": config.spectrum_source.impedance_column,
                    },
                )
            },
        ),
    )
    return state


def _validate_component_state(state: dict, name: str) -> tuple[str, dict]:
    """Validate and unpack one declarative checkpoint component."""
    if not isinstance(state, dict) or set(state) != {"type", "parameters"}:
        raise ValueError(f"invalid progressive {name} checkpoint state")
    if not isinstance(state["type"], str) or not isinstance(state["parameters"], dict):
        raise ValueError(f"invalid progressive {name} checkpoint state")
    return state["type"], state["parameters"]


def _restore_encoder(state: dict, dataprep: EISDataPrep) -> SpectrumObservationEncoder:
    """Rebind a built-in encoder specification to restored data preparation."""
    encoder_type, parameters = _validate_component_state(state, "encoder")
    if encoder_type == "InterpolatingSpectrumEncoder" and not parameters:
        return InterpolatingSpectrumEncoder(dataprep)
    if encoder_type != "RawSpectrumEncoder" or set(parameters) != {
        "padding_value",
        "normalizer",
        "downsampler",
    }:
        raise ValueError(f"unsupported progressive encoder checkpoint type: {encoder_type}")
    normalizer_type, normalizer_parameters = _validate_component_state(
        parameters["normalizer"], "normalizer"
    )
    downsampler_type, downsampler_parameters = _validate_component_state(
        parameters["downsampler"], "downsampler"
    )
    normalizers = {
        "SentinelSafeMinMaxNormalizer": SentinelSafeMinMaxNormalizer,
        "LegacyEISFeatureNormalizer": LegacyEISFeatureNormalizer,
    }
    if normalizer_type not in normalizers or normalizer_parameters:
        raise ValueError(f"unsupported progressive normalizer: {normalizer_type}")
    if downsampler_type != "UniformIndexDownsampler" or downsampler_parameters:
        raise ValueError(f"unsupported progressive downsampler: {downsampler_type}")
    return RawSpectrumEncoder(
        dataprep,
        normalizer=normalizers[normalizer_type](),
        downsampler=UniformIndexDownsampler(),
        padding_value=parameters["padding_value"],
    )


def restore_progressive_runtime_state(
    state: dict | None,
    *,
    training_dataprep: EISDataPrep | None,
    evaluation_dataprep: EISDataPrep | None,
) -> ProgressiveSpectrumRuntimeConfig | None:
    """Restore declarative progressive configuration against restored environments."""
    if state is None:
        return None
    required = {
        "version",
        "enabled",
        "num_windows",
        "selector",
        "training_encoder",
        "evaluation_encoder",
        "transition_policy",
        "spectrum_source",
    }
    if not isinstance(state, dict) or set(state) != required or state["version"] != 1:
        raise ValueError("invalid progressive runtime checkpoint schema/version")
    if not state["enabled"]:
        if any(
            state[name] is not None
            for name in required - {"version", "enabled", "num_windows"}
        ):
            raise ValueError("disabled progressive runtime must not contain components")
        return ProgressiveSpectrumRuntimeConfig(False, state["num_windows"])
    if training_dataprep is None or evaluation_dataprep is None:
        raise ValueError("enabled progressive checkpoint requires environment dataprep")

    selector_type, selector_parameters = _validate_component_state(
        state["selector"], "selector"
    )
    if selector_type != "EqualPointWindowSelector" or set(selector_parameters) != {
        "num_windows"
    }:
        raise ValueError(f"unsupported progressive selector: {selector_type}")
    selector = EqualPointWindowSelector(**selector_parameters)
    if selector.num_windows != state["num_windows"]:
        raise ValueError("progressive selector window count does not match runtime")

    policy_type, policy_parameters = _validate_component_state(
        state["transition_policy"], "transition policy"
    )
    if policy_type != "CarryBestStatePolicy" or policy_parameters:
        raise ValueError(f"unsupported progressive transition policy: {policy_type}")
    source_type, source_parameters = _validate_component_state(
        state["spectrum_source"], "spectrum source"
    )
    if source_type != "DataFrameSpectrumSource" or set(source_parameters) != {
        "name",
        "frequency_column",
        "impedance_column",
    }:
        raise ValueError(f"unsupported progressive spectrum source: {source_type}")

    training_encoder = _restore_encoder(state["training_encoder"], training_dataprep)
    evaluation_encoder = (
        training_encoder
        if evaluation_dataprep is training_dataprep
        and state["evaluation_encoder"] == state["training_encoder"]
        else _restore_encoder(state["evaluation_encoder"], evaluation_dataprep)
    )
    return ProgressiveSpectrumRuntimeConfig(
        enabled=True,
        num_windows=state["num_windows"],
        selector=selector,
        training_encoder=training_encoder,
        evaluation_encoder=evaluation_encoder,
        transition_policy=CarryBestStatePolicy(),
        spectrum_source=DataFrameSpectrumSource(**source_parameters),
    )
