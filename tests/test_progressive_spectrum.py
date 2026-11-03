"""Unit tests for environment-independent progressive-spectrum components.

This module owns component-level contracts: selection, source provenance, fixed-size encoding,
normalization, downsampling, and transition decisions. Environment fitting, agent orchestration,
factory wiring, and runnable scripts are intentionally covered in their respective test modules
instead of being repeated here.
"""

import numpy as np
import pytest

from autorec.data_preparation import EISDataPrep
from autorec.progressive_spectrum import (
    CarryBestStatePolicy,
    DataFrameSpectrumSource,
    EqualPointWindowSelector,
    InterpolatingSpectrumEncoder,
    LegacyEISFeatureNormalizer,
    PREPARED_SPECTRUM_SOURCE,
    RETAINED_SPECTRUM_SOURCE,
    ProgressiveStageOutcome,
    ProgressiveSpectrumRuntimeConfig,
    RawSpectrumEncoder,
    SentinelSafeMinMaxNormalizer,
    SpectrumView,
    SpectrumWindowSelection,
    StageTransitionDecision,
    UniformIndexDownsampler,
    build_spectrum_view,
    decide_stage_transition,
    progressive_runtime_state_dict,
    restore_progressive_runtime_state,
)


def make_dataprep(tmp_path, *, npoints=10, features=None):
    return EISDataPrep(
        tmp_path,
        mode="process",
        frequency_npoints=npoints,
        eis_features=features or ["ReZ", "ImZ"],
    )


def make_spectrum(npoints=10):
    frequency = np.geomspace(1e5, 1e-2, npoints)
    log_frequency = np.log10(frequency)
    impedance = (20.0 + 0.3 * log_frequency + 0.1 * log_frequency**2) + 1j * (
        5.0 - 0.4 * log_frequency + 0.05 * log_frequency**2
    )
    return frequency, impedance


def test_progressive_runtime_checkpoint_state_rebinds_builtin_components(tmp_path):
    dataprep = make_dataprep(tmp_path, npoints=6)
    encoder = RawSpectrumEncoder(dataprep)
    runtime = ProgressiveSpectrumRuntimeConfig(
        enabled=True,
        num_windows=3,
        selector=EqualPointWindowSelector(3),
        training_encoder=encoder,
        evaluation_encoder=encoder,
        transition_policy=CarryBestStatePolicy(),
        spectrum_source=RETAINED_SPECTRUM_SOURCE,
    )

    state = progressive_runtime_state_dict(
        runtime,
        training_dataprep=dataprep,
        evaluation_dataprep=dataprep,
    )
    restored = restore_progressive_runtime_state(
        state,
        training_dataprep=dataprep,
        evaluation_dataprep=dataprep,
    )

    assert restored.metadata == runtime.metadata
    assert isinstance(restored.training_encoder, RawSpectrumEncoder)
    assert restored.training_encoder is restored.evaluation_encoder
    assert restored.training_encoder.dataprep is dataprep
    assert isinstance(restored.training_encoder.normalizer, SentinelSafeMinMaxNormalizer)
    assert isinstance(restored.training_encoder.downsampler, UniformIndexDownsampler)


def test_progressive_runtime_checkpoint_rejects_custom_components(tmp_path):
    dataprep = make_dataprep(tmp_path, npoints=6)

    class CustomPolicy:
        def decide(self, outcome):
            return StageTransitionDecision(outcome.best_state, True)

    runtime = ProgressiveSpectrumRuntimeConfig(
        enabled=True,
        num_windows=2,
        selector=EqualPointWindowSelector(2),
        training_encoder=InterpolatingSpectrumEncoder(dataprep),
        evaluation_encoder=InterpolatingSpectrumEncoder(dataprep),
        transition_policy=CustomPolicy(),
    )

    with pytest.raises(ValueError, match="unsupported checkpoint component"):
        progressive_runtime_state_dict(
            runtime,
            training_dataprep=dataprep,
            evaluation_dataprep=dataprep,
        )


# Window selection, physical sources, and one-window compatibility


def test_equal_point_windows_are_numbered_high_to_low_and_cumulative():
    frequency = np.array([1.0, 1_000.0, 0.1, 10.0, 100.0, 0.01, 0.001])
    impedance = frequency + 2j * frequency
    selector = EqualPointWindowSelector(num_windows=3)

    stage_0 = selector.select(frequency, impedance, stage_index=0)
    stage_1 = selector.select(frequency, impedance, stage_index=1)
    stage_2 = selector.select(frequency, impedance, stage_index=2)

    np.testing.assert_array_equal(stage_0.frequency, [1_000.0, 100.0, 10.0])
    np.testing.assert_array_equal(stage_1.frequency, [1_000.0, 100.0, 10.0, 1.0, 0.1])
    np.testing.assert_array_equal(
        stage_2.frequency, [1_000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    )
    assert stage_0.included_window_indices == (0,)
    assert stage_1.included_window_indices == (0, 1)
    assert stage_2.included_window_indices == (0, 1, 2)


def test_selection_is_independent_of_source_order():
    descending_frequency, descending_impedance = make_spectrum()
    ascending_frequency = descending_frequency[::-1]
    ascending_impedance = descending_impedance[::-1]
    selector = EqualPointWindowSelector(num_windows=3)

    from_descending = selector.select(
        descending_frequency, descending_impedance, stage_index=1
    )
    from_ascending = selector.select(ascending_frequency, ascending_impedance, stage_index=1)

    np.testing.assert_array_equal(from_ascending.frequency, from_descending.frequency)
    np.testing.assert_array_equal(from_ascending.impedance, from_descending.impedance)


def test_dataframe_spectrum_sources_resolve_explicit_dataset_columns():
    row = {
        "freq": np.array([1.0, 10.0]),
        "Z_true": np.array([1 + 2j, 2 + 1j]),
        "source_freq": np.array([1.0, 3.0, 10.0]),
        "source_Z": np.array([1 + 3j, 2 + 2j, 3 + 1j]),
    }

    prepared_frequency, prepared_impedance = PREPARED_SPECTRUM_SOURCE.read(row)
    retained_frequency, retained_impedance = RETAINED_SPECTRUM_SOURCE.read(row)

    np.testing.assert_array_equal(prepared_frequency, row["freq"])
    np.testing.assert_array_equal(prepared_impedance, row["Z_true"])
    np.testing.assert_array_equal(retained_frequency, row["source_freq"])
    np.testing.assert_array_equal(retained_impedance, row["source_Z"])


def test_dataframe_spectrum_source_requires_its_configured_columns():
    source = DataFrameSpectrumSource("custom", "measurement_frequency", "measurement_Z")

    with pytest.raises(
        ValueError,
        match="requires dataset columns: measurement_frequency, measurement_Z",
    ):
        source.read({"freq": np.array([1.0]), "Z_true": np.array([1 + 1j])})


def test_single_window_view_exactly_preserves_full_spectrum(tmp_path):
    frequency = np.array([1.0, 25.0, 120.0, 600.0, 1_000.0, 8.0])
    impedance = np.array([7 + 12j, 10 + 8j, 13 + 5j, 15 + 3j, 18 + 2j, 8 + 9j])
    dataprep = make_dataprep(tmp_path, npoints=6, features=["ReZ", "ImZ"])

    view = build_spectrum_view(
        frequency,
        impedance,
        stage_index=0,
        selector=EqualPointWindowSelector(num_windows=1),
        encoder=InterpolatingSpectrumEncoder(dataprep),
    )

    np.testing.assert_array_equal(view.selection.source_indices, np.arange(6))
    np.testing.assert_array_equal(view.frequency, frequency)
    np.testing.assert_array_equal(view.impedance, impedance)
    np.testing.assert_array_equal(
        view.observation,
        dataprep.flatten_EIS(impedance, dataprep.eis_features),
    )


# Stage-transition contracts


def test_default_transition_policy_carries_best_state():
    outcome = ProgressiveStageOutcome(
        stage_index=0,
        num_stages=3,
        initial_state="+RR",
        best_state="+PR",
        best_reward=1.5,
        found_solution=False,
        metadata={"mode": "evaluation"},
    )

    decision = decide_stage_transition(CarryBestStatePolicy(), outcome)

    assert decision == StageTransitionDecision(
        next_state="+PR",
        continue_to_next_stage=True,
    )


def test_transition_decision_requires_state_when_continuing():
    with pytest.raises(ValueError, match="next_state is required"):
        StageTransitionDecision(next_state=None, continue_to_next_stage=True)


def test_transition_policy_must_return_decision():
    outcome = ProgressiveStageOutcome(
        stage_index=0,
        num_stages=2,
        initial_state="+RR",
        best_state="+PR",
        best_reward=1.5,
        found_solution=True,
        metadata={},
    )

    class InvalidPolicy:
        def decide(self, outcome):
            return {"next_state": outcome.best_state}

    with pytest.raises(TypeError, match="must return a StageTransitionDecision"):
        decide_stage_transition(InvalidPolicy(), outcome)


# Selection and view validation


def test_selection_copies_inputs_and_exposes_read_only_arrays():
    frequency, impedance = make_spectrum()
    original_frequency = frequency.copy()
    original_impedance = impedance.copy()

    selection = EqualPointWindowSelector(num_windows=2).select(
        frequency, impedance, stage_index=0
    )
    frequency[0] = 123.0
    impedance[0] = 456.0j

    np.testing.assert_array_equal(selection.frequency, original_frequency[:5])
    np.testing.assert_array_equal(selection.impedance, original_impedance[:5])
    with pytest.raises(ValueError, match="read-only"):
        selection.frequency[0] = 1.0


@pytest.mark.parametrize("num_windows", [0, -1])
def test_selector_rejects_nonpositive_window_count(num_windows):
    with pytest.raises(ValueError, match="greater than zero"):
        EqualPointWindowSelector(num_windows=num_windows)


@pytest.mark.parametrize("num_windows", [1.5, True])
def test_selector_rejects_noninteger_window_count(num_windows):
    with pytest.raises(TypeError, match="must be an integer"):
        EqualPointWindowSelector(num_windows=num_windows)


def test_selector_validates_window_count_stage_and_spectrum():
    frequency, impedance = make_spectrum(npoints=4)
    selector = EqualPointWindowSelector(num_windows=2)

    with pytest.raises(ValueError, match="cannot exceed"):
        EqualPointWindowSelector(num_windows=5).select(frequency, impedance, stage_index=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        selector.select(frequency, impedance, stage_index=2)
    with pytest.raises(ValueError, match="same number"):
        selector.select(frequency, impedance[:-1], stage_index=0)
    with pytest.raises(ValueError, match="must be unique"):
        selector.select(np.array([10.0, 10.0]), impedance[:2], stage_index=0)


def test_encoder_uses_dataprep_settings_for_fixed_size_at_every_stage(tmp_path):
    frequency, impedance = make_spectrum(npoints=10)
    dataprep = make_dataprep(tmp_path, npoints=10, features=["ReZ", "ImZ", "phi", "mag"])
    selector = EqualPointWindowSelector(num_windows=3)
    encoder = InterpolatingSpectrumEncoder(dataprep)

    views = [
        build_spectrum_view(
            frequency, impedance, stage_index, selector=selector, encoder=encoder
        )
        for stage_index in range(selector.num_windows)
    ]

    assert encoder.output_npoints == 10
    assert encoder.eis_features == ("ReZ", "ImZ", "phi", "mag")
    assert encoder.observation_size == 40
    assert [len(view.selection.frequency) for view in views] == [4, 7, 10]
    assert all(len(view.frequency) == 10 for view in views)
    assert all(len(view.impedance) == 10 for view in views)
    assert all(len(view.observation) == 40 for view in views)
    assert all(np.all(np.isfinite(view.observation)) for view in views)
    assert all(np.all(view.observation_valid_mask) for view in views)
    assert all(not view.observation_valid_mask.flags.writeable for view in views)
    assert views[0].observation_source_indices is None
    np.testing.assert_array_equal(
        views[-1].observation_source_indices,
        views[-1].selection.source_indices,
    )

    for view in views:
        assert view.frequency[0] == view.selection.frequency[0]
        assert view.frequency[-1] == view.selection.frequency[-1]
        assert np.all(np.diff(view.frequency) < 0)


def test_build_spectrum_view_accepts_structural_encoder_interface():
    frequency, impedance = make_spectrum(npoints=4)

    class DirectEncoder:
        output_npoints = 2
        eis_features = ("ReZ", "ImZ")
        observation_size = 4

        def encode(self, selection):
            return SpectrumView(
                selection=selection,
                frequency=selection.frequency,
                impedance=selection.impedance,
                observation=np.arange(self.observation_size, dtype=float),
                observation_valid_mask=np.ones(self.output_npoints, dtype=bool),
                observation_source_indices=selection.source_indices,
            )

    view = build_spectrum_view(
        frequency,
        impedance,
        stage_index=0,
        selector=EqualPointWindowSelector(num_windows=2),
        encoder=DirectEncoder(),
    )

    np.testing.assert_array_equal(view.frequency, view.selection.frequency)
    np.testing.assert_array_equal(
        view.observation_source_indices,
        view.selection.source_indices,
    )
    assert len(view.observation) == 4


def test_build_spectrum_view_records_physical_dataset_source(tmp_path):
    frequency, impedance = make_spectrum(npoints=6)
    dataprep = make_dataprep(tmp_path, npoints=6)

    view = build_spectrum_view(
        frequency,
        impedance,
        stage_index=0,
        selector=EqualPointWindowSelector(num_windows=2),
        encoder=InterpolatingSpectrumEncoder(dataprep),
        spectrum_source=RETAINED_SPECTRUM_SOURCE,
    )

    assert view.selection.source is RETAINED_SPECTRUM_SOURCE


def test_spectrum_view_validates_optional_observation_provenance():
    selection = SpectrumWindowSelection(
        stage_index=0,
        included_window_indices=(0,),
        source_indices=np.array([3, 2]),
        frequency=np.array([1_000.0, 100.0]),
        impedance=np.array([4 + 1j, 3 + 2j]),
    )

    with pytest.raises(ValueError, match="identify each encoded frequency"):
        SpectrumView(
            selection=selection,
            frequency=selection.frequency,
            impedance=selection.impedance,
            observation=np.arange(8, dtype=float),
            observation_valid_mask=np.array([True, False, False, False]),
        )

    with pytest.raises(ValueError, match="points in the physical selection"):
        SpectrumView(
            selection=selection,
            frequency=selection.frequency,
            impedance=selection.impedance,
            observation=np.arange(8, dtype=float),
            observation_valid_mask=np.array([True, True, False, False]),
            observation_source_indices=np.array([3, 1]),
        )


# Raw encoding, normalization, padding, and downsampling


def test_raw_encoder_normalizes_valid_points_then_pads_every_feature(tmp_path):
    features = ["ReZ", "ImZ", "phi", "mag", "nReZ", "nImZ", "nphi", "nmag"]
    dataprep = make_dataprep(tmp_path, npoints=5, features=features)
    frequency, impedance = make_spectrum(npoints=3)
    selection = EqualPointWindowSelector(num_windows=1).select(
        frequency, impedance, stage_index=0
    )

    view = RawSpectrumEncoder(dataprep).encode(selection)
    feature_matrix = view.observation.reshape(len(features), 5)

    assert feature_matrix.shape == (len(features), 5)
    assert np.all((feature_matrix[:, :3] >= 0.0) & (feature_matrix[:, :3] <= 1.0))
    np.testing.assert_array_equal(feature_matrix[:, 3:], -1.0)
    np.testing.assert_array_equal(
        view.observation_valid_mask, [True, True, True, False, False]
    )
    np.testing.assert_array_equal(view.frequency, selection.frequency)
    np.testing.assert_array_equal(view.impedance, selection.impedance)
    np.testing.assert_array_equal(view.observation_source_indices, selection.source_indices)


def test_raw_encoder_handles_single_constant_point_without_nonfinite_values(tmp_path):
    dataprep = make_dataprep(
        tmp_path,
        npoints=4,
        features=["ReZ", "ImZ", "phi", "mag", "nReZ", "nImZ", "nphi", "nmag"],
    )
    selection = EqualPointWindowSelector(num_windows=1).select(
        np.array([100.0]),
        np.array([5 + 2j]),
        stage_index=0,
    )

    view = RawSpectrumEncoder(dataprep).encode(selection)
    feature_matrix = view.observation.reshape(8, 4)

    np.testing.assert_array_equal(feature_matrix[:, 0], 0.0)
    np.testing.assert_array_equal(feature_matrix[:, 1:], -1.0)
    assert np.all(np.isfinite(view.observation))


def test_raw_encoder_exact_capacity_uses_original_values_without_padding(tmp_path):
    dataprep = make_dataprep(tmp_path, npoints=4, features=["ReZ", "ImZ"])
    frequency, impedance = make_spectrum(npoints=4)
    selection = EqualPointWindowSelector(num_windows=1).select(
        frequency, impedance, stage_index=0
    )

    view = RawSpectrumEncoder(dataprep).encode(selection)
    expected = SentinelSafeMinMaxNormalizer().normalize(
        impedance, tuple(dataprep.eis_features)
    )

    np.testing.assert_array_equal(view.frequency, frequency)
    np.testing.assert_array_equal(view.impedance, impedance)
    np.testing.assert_allclose(view.observation, expected.reshape(-1))
    assert np.all(view.observation_valid_mask)


def test_raw_encoder_legacy_normalizer_recovers_existing_full_observation(tmp_path):
    features = ["ImZ", "phi", "mag", "nphi"]
    dataprep = make_dataprep(tmp_path, npoints=4, features=features)
    frequency, impedance = make_spectrum(npoints=4)
    selection = EqualPointWindowSelector(num_windows=1).select(
        frequency, impedance, stage_index=0
    )

    view = RawSpectrumEncoder(
        dataprep,
        normalizer=LegacyEISFeatureNormalizer(),
    ).encode(selection)

    np.testing.assert_allclose(
        view.observation,
        dataprep.flatten_EIS(impedance, features),
    )


def test_uniform_downsampler_preserves_points_and_downsamples_deterministically():
    frequency, impedance = make_spectrum(npoints=4)
    downsampler = UniformIndexDownsampler()

    np.testing.assert_array_equal(
        downsampler.select_indices(frequency, impedance, capacity=6),
        np.arange(4),
    )
    np.testing.assert_array_equal(
        downsampler.select_indices(frequency, impedance, capacity=4),
        np.arange(4),
    )
    larger_frequency, larger_impedance = make_spectrum(npoints=10)
    first = downsampler.select_indices(larger_frequency, larger_impedance, capacity=4)
    second = downsampler.select_indices(larger_frequency, larger_impedance, capacity=4)

    np.testing.assert_array_equal(first, [0, 3, 6, 9])
    np.testing.assert_array_equal(second, first)


def test_raw_encoder_downsamples_observation_after_full_stage_normalization(tmp_path):
    dataprep = make_dataprep(tmp_path, npoints=3, features=["ReZ", "ImZ"])
    frequency, impedance = make_spectrum(npoints=7)
    selection = EqualPointWindowSelector(num_windows=1).select(
        frequency, impedance, stage_index=0
    )

    view = RawSpectrumEncoder(dataprep).encode(selection)
    expected_indices = np.array([0, 3, 6])
    full_normalized = SentinelSafeMinMaxNormalizer().normalize(
        impedance, tuple(dataprep.eis_features)
    )

    np.testing.assert_array_equal(view.frequency, frequency[expected_indices])
    np.testing.assert_array_equal(view.impedance, impedance[expected_indices])
    np.testing.assert_array_equal(
        view.observation_source_indices,
        selection.source_indices[expected_indices],
    )
    np.testing.assert_allclose(
        view.observation,
        full_normalized[:, expected_indices].reshape(-1),
    )
    assert np.all(view.observation_valid_mask)
    assert len(view.selection.frequency) == 7


def test_raw_encoder_preserves_falsey_modular_components(tmp_path):
    dataprep = make_dataprep(tmp_path, npoints=3, features=["ReZ", "ImZ"])
    frequency, impedance = make_spectrum(npoints=5)
    selection = EqualPointWindowSelector(num_windows=1).select(
        frequency, impedance, stage_index=0
    )

    class FalseyNormalizer:
        def __bool__(self):
            return False

        def normalize(self, impedance, eis_features):
            return np.full((len(eis_features), len(impedance)), 0.25)

    class FalseyDownsampler:
        def __bool__(self):
            return False

        def select_indices(self, frequency, impedance, capacity):
            assert capacity == 3
            return np.array([0, 1, len(frequency) - 1])

    normalizer = FalseyNormalizer()
    downsampler = FalseyDownsampler()
    encoder = RawSpectrumEncoder(
        dataprep,
        normalizer=normalizer,
        downsampler=downsampler,
    )
    view = encoder.encode(selection)

    assert encoder.normalizer is normalizer
    assert encoder.downsampler is downsampler
    np.testing.assert_array_equal(view.frequency, frequency[[0, 1, 4]])
    np.testing.assert_array_equal(view.observation, np.full(6, 0.25))


# Interpolation identity and error boundaries


def test_final_stage_preserves_an_irregular_prepared_frequency_grid(tmp_path):
    frequency = np.array([1_000.0, 600.0, 120.0, 25.0, 8.0, 1.0])
    impedance = np.array([18 + 2j, 15 + 3j, 13 + 5j, 10 + 8j, 8 + 9j, 7 + 12j])
    dataprep = make_dataprep(tmp_path, npoints=6, features=["ReZ", "ImZ"])
    selector = EqualPointWindowSelector(num_windows=2)
    encoder = InterpolatingSpectrumEncoder(dataprep)

    view = build_spectrum_view(
        frequency, impedance, stage_index=1, selector=selector, encoder=encoder
    )

    np.testing.assert_array_equal(view.frequency, frequency)
    np.testing.assert_allclose(view.impedance, impedance)
    np.testing.assert_allclose(
        view.observation,
        dataprep.flatten_EIS(impedance, dataprep.eis_features),
    )


def test_encoder_requires_dataprep_with_fixed_point_count(tmp_path):
    dataprep = make_dataprep(tmp_path, npoints=10)
    dataprep.frequency_npoints = None

    with pytest.raises(ValueError, match="frequency_npoints"):
        InterpolatingSpectrumEncoder(dataprep)


def test_encoder_rejects_stage_with_only_one_selected_point(tmp_path):
    dataprep = make_dataprep(tmp_path, npoints=4)
    selection = EqualPointWindowSelector(num_windows=4).select(
        np.array([1_000.0, 100.0, 10.0, 1.0]),
        np.array([4 + 1j, 3 + 2j, 2 + 3j, 1 + 4j]),
        stage_index=0,
    )

    with pytest.raises(ValueError, match="at least two"):
        InterpolatingSpectrumEncoder(dataprep).encode(selection)
