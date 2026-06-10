"""Tests for EISDataPrep data-shaping, loading, saving, and raw CSV processing.

The expensive Lin-KK threshold calculation is monkeypatched in process-mode tests so the
suite verifies AutoREC's file/data plumbing without running circuit-fitting routines.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from autorec.data_preparation import EISDataPrep


def make_processed_dataset():
    """Return a minimal processed dataset that satisfies load-mode validation."""
    return pd.DataFrame(
        {
            "sub_id": ["sample.csv"],
            "freq": [np.array([1.0, 10.0, 100.0])],
            "Z_true": [np.array([1 + 1j, 2 + 2j, 3 + 4j])],
            "flatten_Z": [np.array([0.0, 0.5, 1.0])],
            "chi_thresh": [0.01],
            "r2_thresh": [0.95],
        }
    )


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


def test_load_mode_reads_pickle_dataframe(tmp_path):
    dataset = make_processed_dataset()
    pkl_path = tmp_path / "dataset.pkl"
    dataset.to_pickle(pkl_path)

    prep = EISDataPrep(pkl_path, mode="load")
    loaded = prep.load()

    pd.testing.assert_frame_equal(loaded, dataset)


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


def test_process_raw_csv_builds_processed_dataset(tmp_path, monkeypatch):
    base_path = tmp_path / "raw"
    csv_path = base_path / "R0-[R1,P2]" / "sample.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "freq": [1.0, 10.0, 100.0],
            "Z_real": [1.0, 2.0, 3.0],
            "Z_imag": [1.0, 2.0, 4.0],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        EISDataPrep, "calculate_thresholds", lambda self, freq, z: (0.01, 0.95)
    )
    monkeypatch.setattr(
        "autorec.data_preparation.ae.parser.validate_circuit", lambda circuit: True
    )

    prep = EISDataPrep(base_path, mode="process")
    dataset = prep.load()

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["sub_id"] == "R0-[R1,P2]/sample.csv"
    assert row["true_circuit"] == "R0-[R1,P2]"
    np.testing.assert_allclose(row["freq"], np.array([1.0, 10.0, 100.0]))
    np.testing.assert_allclose(row["Z_true"], np.array([1 + 1j, 2 + 2j, 3 + 4j]))
    assert row["chi_thresh"] == 0.01
    assert row["r2_thresh"] == 0.95
