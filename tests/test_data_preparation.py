"""Tests for EISDataPrep data-shaping, loading, saving, and raw CSV processing.

The expensive Lin-KK threshold calculation is monkeypatched in process-mode tests so the
suite verifies AutoREC's file/data plumbing without running circuit-fitting routines.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from autorec.data_preparation import EISDataPrep


def make_processed_dataset(metadata=None):
    """Return a minimal processed dataset that satisfies load-mode validation."""
    dataset = pd.DataFrame(
        {
            "sub_id": ["sample.csv"],
            "freq": [np.array([1.0, 10.0, 100.0])],
            "Z_true": [np.array([1 + 1j, 2 + 2j, 3 + 4j])],
            "flatten_Z": [np.array([0.0, 0.5, 1.0])],
            "chi_thresh": [0.01],
            "r2_thresh": [0.95],
        }
    )
    if metadata is not None:
        dataset["metadata"] = [metadata]
    return dataset


def write_raw_csv(path, freq, *, z_real=None, z_imag=None):
    """Write a raw spectrum, with linear defaults for both impedance components."""
    path.parent.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(freq, dtype=float)
    z_real = freq if z_real is None else np.asarray(z_real, dtype=float)
    z_imag = 2 * freq if z_imag is None else np.asarray(z_imag, dtype=float)
    pd.DataFrame(
        {
            "freq": freq,
            "Z_real": z_real,
            "Z_imag": z_imag,
        }
    ).to_csv(path, index=False)


def test_normalize_eis_returns_expected_min_max_scaled_values():
    z = np.array([1 + 10j, 2 + 20j, 3 + 40j])

    z_norm, angles_norm, mag_norm, mag_scaled = EISDataPrep.normalize_EIS(z)

    np.testing.assert_allclose(z_norm.real, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(z_norm.imag, np.array([0.0, 1 / 3, 1.0]))
    np.testing.assert_allclose(
        angles_norm, (np.angle(z) - np.angle(z).min()) / np.ptp(np.angle(z))
    )
    np.testing.assert_allclose(mag_norm, (np.abs(z) - np.abs(z).min()) / np.ptp(np.abs(z)))
    np.testing.assert_allclose(mag_scaled, np.log10(np.abs(z)) / max(np.log10(np.abs(z))))


def test_flatten_eis_concatenates_selected_normalized_features():
    z = np.array([1 + 10j, 2 + 20j, 3 + 40j])
    z_norm, phi, _, mag = EISDataPrep.normalize_EIS(z)

    flattened = EISDataPrep.flatten_EIS(z, eis_features=["ReZ", "ImZ", "nphi", "mag"])

    expected = np.concatenate([z_norm.real, z_norm.imag, -phi, mag])
    np.testing.assert_allclose(flattened, expected)


def test_interpolate_eis_handles_unsorted_input_and_output_frequencies():
    freq = np.array([10.0, 1.0, 5.0, 20.0])
    z = freq + 2j * freq
    freq_new = np.array([8.0, 3.0, 15.0])

    interpolated = EISDataPrep.interpolate_EIS(freq, z, freq_new)

    np.testing.assert_allclose(interpolated.real, freq_new)
    np.testing.assert_allclose(interpolated.imag, 2 * freq_new)


@pytest.mark.parametrize("frequency_bounds", [(10.0, 1_000.0), [10.0, 1_000.0]])
def test_process_mode_accepts_single_frequency_bounds_pair(tmp_path, frequency_bounds):
    raw_path = tmp_path / "raw"
    raw_path.mkdir()

    prep = EISDataPrep(
        raw_path,
        mode="process",
        perform_linKK_validation=True,
        tol_linKK=0.01,
        frequency_bounds=frequency_bounds,
        frequency_npoints=7,
    )

    assert prep.perform_linKK_validation is True
    assert prep.tol_linKK == 0.01
    assert prep.frequency_bounds == frequency_bounds
    assert prep.frequency_npoints == 7


@pytest.mark.parametrize(
    "frequency_bounds",
    [
        (1.0, 10.0, 100.0),
        [(1.0, 10.0), (10.0, 100.0)],
    ],
)
def test_process_mode_rejects_frequency_bounds_with_wrong_shape(tmp_path, frequency_bounds):
    raw_path = tmp_path / "raw"
    raw_path.mkdir()

    with pytest.raises(ValueError, match="frequency_bounds must be"):
        EISDataPrep(raw_path, mode="process", frequency_bounds=frequency_bounds)


def test_get_circuit_from_folder_uses_parent_folder_name(tmp_path):
    base_path = tmp_path / "raw"
    csv_path = base_path / "R0-[R1,P2]" / "sample.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.touch()
    prep = EISDataPrep(base_path, mode="process")

    assert prep.get_circuit_from_folder(csv_path, base_path) == "R0-[R1,P2]"


def test_get_circuit_from_folder_falls_back_to_filename_stem(tmp_path):
    base_path = tmp_path / "raw"
    base_path.mkdir()
    csv_path = base_path / "R0-R1_003.csv"
    csv_path.touch()
    prep = EISDataPrep(base_path, mode="process")

    assert prep.get_circuit_from_folder(csv_path, base_path) == "R0-R1"


def test_create_sub_id_returns_path_relative_to_base(tmp_path):
    base_path = tmp_path / "raw"
    csv_path = base_path / "R0" / "sample.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.touch()
    prep = EISDataPrep(base_path, mode="process")

    assert prep.create_sub_id(csv_path, base_path) == "R0/sample.csv"


def test_load_mode_reads_pickle_metadata_and_updates_attributes(tmp_path):
    metadata = {
        "path": "raw/source",
        "mode": "process",
        "evaluation": True,
        "perform_linKK_validation": True,
        "tol_linKK": 0.01,
        "frequency_bounds": (10.0, 1_000.0),
        "frequency_npoints": 3,
        "eis_features": ["ReZ", "ImZ"],
    }
    dataset = make_processed_dataset(metadata=metadata)
    pkl_path = tmp_path / "dataset.pkl"
    dataset.to_pickle(pkl_path)

    prep = EISDataPrep(pkl_path, mode="load")
    loaded = prep.load()

    pd.testing.assert_frame_equal(loaded, dataset)
    assert prep.perform_linKK_validation is True
    assert prep.tol_linKK == 0.01
    assert prep.frequency_bounds == (10.0, 1_000.0)
    assert prep.frequency_npoints == 3
    assert prep.eis_features == ["ReZ", "ImZ"]


def test_load_mode_rejects_pickle_that_is_not_dataframe(tmp_path):
    pkl_path = tmp_path / "not_dataframe.pkl"
    with open(pkl_path, "wb") as file:
        pickle.dump({"not": "a dataframe"}, file)

    prep = EISDataPrep(pkl_path, mode="load")

    with pytest.raises(ValueError, match="Pickle file does not contain a pandas DataFrame"):
        prep.load()


def test_load_mode_reads_csv_with_pandas(tmp_path):
    csv_path = tmp_path / "simple.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_path, index=False)
    prep = EISDataPrep(csv_path, mode="load")

    loaded = prep._load_data()

    pd.testing.assert_frame_equal(loaded, pd.DataFrame({"a": [1], "b": [2]}))


def test_save_writes_pickle_and_csv(tmp_path):
    dataset = make_processed_dataset()
    source_path = tmp_path / "source.pkl"
    dataset.to_pickle(source_path)
    prep = EISDataPrep(source_path, mode="load")
    prep.dataset = dataset

    pickle_path = tmp_path / "saved.pkl"
    csv_path = tmp_path / "saved.csv"
    prep.save(pickle_path, file_type="pickle")
    prep.save(csv_path, file_type="csv")

    pd.testing.assert_frame_equal(pd.read_pickle(pickle_path), dataset)
    assert list(pd.read_csv(csv_path).columns) == list(dataset.columns)


def test_save_requires_loaded_dataset(tmp_path):
    source_path = tmp_path / "source.pkl"
    make_processed_dataset().to_pickle(source_path)
    prep = EISDataPrep(source_path, mode="load")

    with pytest.raises(ValueError, match="No dataset to save"):
        prep.save(tmp_path / "out.pkl")


def test_process_raw_data_rejects_folder_without_csv_files(tmp_path):
    base_path = tmp_path / "raw"
    base_path.mkdir()
    prep = EISDataPrep(base_path, mode="process")

    with pytest.raises(ValueError, match="No CSV files found"):
        prep.load()


def test_interpolation_validation_defaults_to_consistent_processed_length(tmp_path):
    base_path = tmp_path / "raw"
    write_raw_csv(base_path / "R0" / "first.csv", [1.0, 10.0, 100.0, 1_000.0])
    write_raw_csv(base_path / "R1" / "second.csv", [1.0, 10.0, 100.0, 1_000.0])
    prep = EISDataPrep(base_path, mode="process")
    processed_data = [
        (np.array([1.0, 10.0, 100.0]), np.ones(3, dtype=complex)),
        (np.array([2.0, 20.0, 200.0]), np.ones(3, dtype=complex)),
    ]

    prep._validate_interpolation_parameters_after_linKK(processed_data)

    assert prep.frequency_npoints == 3


def test_interpolation_validation_requires_npoints_for_inconsistent_processed_lengths(
    tmp_path,
):
    base_path = tmp_path / "raw"
    write_raw_csv(base_path / "R0" / "first.csv", [1.0, 10.0, 100.0])
    write_raw_csv(base_path / "R1" / "second.csv", [1.0, 10.0, 100.0, 1_000.0])
    prep = EISDataPrep(base_path, mode="process")
    processed_data = [
        (np.array([1.0, 10.0, 100.0]), np.ones(3, dtype=complex)),
        (np.array([1.0, 10.0, 100.0, 1_000.0]), np.ones(4, dtype=complex)),
    ]

    with pytest.raises(ValueError, match="specify 'frequency_npoints'"):
        prep._validate_interpolation_parameters_after_linKK(processed_data)


@pytest.mark.parametrize(
    ("frequency_bounds", "processed_frequencies"),
    [
        ((5.0, None), ([1.0, 10.0, 100.0], [10.0, 100.0, 1_000.0])),
        ((None, 2_000.0), ([1.0, 100.0, 3_000.0], [1.0, 100.0, 1_000.0])),
    ],
)
def test_interpolation_validation_checks_shared_bounds_against_every_spectrum(
    tmp_path, frequency_bounds, processed_frequencies
):
    base_path = tmp_path / "raw"
    base_path.mkdir()
    prep = EISDataPrep(
        base_path,
        mode="process",
        frequency_bounds=frequency_bounds,
        frequency_npoints=3,
    )
    processed_data = [
        (np.array(freq), np.ones(3, dtype=complex)) for freq in processed_frequencies
    ]

    with pytest.raises(ValueError, match="Cannot do extrapolation"):
        prep._validate_interpolation_parameters_after_linKK(processed_data)


def test_process_raw_csv_without_linkk_interpolates_dataset(tmp_path, monkeypatch):
    base_path = tmp_path / "raw"
    csv_path = base_path / "R0-[R1,P2]" / "sample.csv"
    raw_freq = np.array([1.0, 10.0, 100.0, 1_000.0])
    raw_impedance = raw_freq + 1j * raw_freq**2
    write_raw_csv(csv_path, raw_freq, z_imag=raw_impedance.imag)

    monkeypatch.setattr(
        "autorec.data_preparation.ae.utils.preprocess_impedance_data",
        lambda *args, **kwargs: pytest.fail("Lin-KK preprocessing should be disabled"),
    )
    monkeypatch.setattr(
        EISDataPrep, "calculate_thresholds", lambda self, freq, z: (0.01, 0.95)
    )
    monkeypatch.setattr(
        "autorec.data_preparation.ae.parser.validate_circuit", lambda circuit: True
    )

    prep = EISDataPrep(
        base_path,
        mode="process",
        frequency_bounds=(10.0, 1_000.0),
        frequency_npoints=3,
        eis_features=["ReZ", "ImZ"],
    )
    dataset = prep.load()

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["sub_id"] == "R0-[R1,P2]/sample.csv"
    assert row["true_circuit"] == "R0-[R1,P2]"
    expected_freq = np.array([10.0, 100.0, 1_000.0])
    np.testing.assert_allclose(row["freq"], expected_freq)
    np.testing.assert_allclose(row["Z_true"], expected_freq + 1j * expected_freq**2)
    assert len(row["flatten_Z"]) == 2 * len(expected_freq)
    assert row["chi_thresh"] == 0.01
    assert row["r2_thresh"] == 0.95
    assert row["metadata"] == {
        "path": str(base_path),
        "mode": "process",
        "evaluation": False,
        "perform_linKK_validation": False,
        "tol_linKK": 0.05,
        "frequency_bounds": (10.0, 1_000.0),
        "frequency_npoints": 3,
        "eis_features": ["ReZ", "ImZ"],
    }


def test_process_raw_csv_with_linkk_preprocesses_before_interpolation(tmp_path, monkeypatch):
    base_path = tmp_path / "raw"
    csv_path = base_path / "R0-[R1,P2]" / "sample.csv"
    raw_freq = np.array([1.0, 10.0, 100.0, 1_000.0])
    raw_impedance = raw_freq + 1j * raw_freq**2
    write_raw_csv(csv_path, raw_freq, z_imag=raw_impedance.imag)
    preprocess_calls = []

    def fake_preprocess(freq, impedance, tol_linKK):
        preprocess_calls.append((freq.copy(), impedance.copy(), tol_linKK))
        return freq[1:], impedance[1:]

    monkeypatch.setattr(
        "autorec.data_preparation.ae.utils.preprocess_impedance_data", fake_preprocess
    )
    monkeypatch.setattr(
        EISDataPrep, "calculate_thresholds", lambda self, freq, z: (0.01, 0.95)
    )
    monkeypatch.setattr(
        "autorec.data_preparation.ae.parser.validate_circuit", lambda circuit: True
    )

    prep = EISDataPrep(
        base_path,
        mode="process",
        perform_linKK_validation=True,
        tol_linKK=0.02,
        frequency_bounds=(10.0, 1_000.0),
        frequency_npoints=3,
        eis_features=["ReZ", "ImZ"],
    )
    dataset = prep.load()

    row = dataset.iloc[0]
    expected_freq = np.array([10.0, 100.0, 1_000.0])
    np.testing.assert_allclose(row["freq"], expected_freq)
    np.testing.assert_allclose(row["Z_true"], expected_freq + 1j * expected_freq**2)
    assert len(preprocess_calls) == 1
    np.testing.assert_allclose(preprocess_calls[0][0], raw_freq)
    np.testing.assert_allclose(preprocess_calls[0][1], raw_impedance)
    assert preprocess_calls[0][2] == 0.02
    assert row["metadata"]["perform_linKK_validation"] is True
    assert row["metadata"]["tol_linKK"] == 0.02
