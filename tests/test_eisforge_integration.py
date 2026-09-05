"""Consumer contract tests for raw datasets exported by EISForge."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# AutoREC imports native scientific dependencies at module import time.  Keep
# their caches outside the repository until the shared test setup on upstream's
# unit-test branch is merged.
_TEST_CACHE = Path(tempfile.gettempdir()) / "autorec_eisforge_contract_test"
os.environ.setdefault("XDG_CACHE_HOME", str(_TEST_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_TEST_CACHE / "matplotlib"))
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(_TEST_CACHE / "julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(_TEST_CACHE / "julia_depot"))
os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from autorec.data_preparation import EISDataPrep


def _skip_threshold_fitting(monkeypatch):
    """Keep file-contract tests independent from Lin-KK circuit fitting."""
    monkeypatch.setattr(
        EISDataPrep,
        "calculate_thresholds",
        lambda self, freq, impedance: (0.01, 0.95),
    )
    monkeypatch.setattr(
        "autorec.data_preparation.ae.parser.validate_circuit",
        lambda circuit: True,
    )


def test_eis_feature_generator_is_materialized_once():
    """One-shot feature iterables retain their values after validation."""
    features = (feature for feature in ["ImZ", "phi"])

    assert EISDataPrep._validate_eis_features(features) == ["ImZ", "phi"]


def test_none_eis_features_uses_defaults():
    """An omitted feature selection resolves to AutoREC's standard features."""
    assert EISDataPrep._validate_eis_features(None) == list(
        EISDataPrep.DEFAULT_EIS_FEATURES
    )


def test_eisforge_output_is_consumed_without_reading_summary_csv(
    tmp_path, monkeypatch, capsys
):
    """An EISForge output root is a valid AutoREC process-mode input."""
    # Recreate the small EISForge output contract under pytest's temporary
    # directory so no generated data is committed to the repository.
    (tmp_path / ".eisforge-output").write_text("Managed by EISForge\n", encoding="utf-8")
    circuit_dir = tmp_path / "R1-[P2,R3]"
    circuit_dir.mkdir()
    pd.DataFrame(
        {
            "freq": [10000.0, 1000.0, 100.0, 10.0, 1.0],
            "Z_real": [100.0, 100.0, 100.0011, 100.3607, 155.7142],
            "Z_imag": [0.00, -0.151037772, -1.48644847, -11.7323132, -39.3695186],
        }
    ).to_csv(circuit_dir / "sample_0000.csv", index=False)
    pd.DataFrame({"summary_only": [True]}).to_csv(
        tmp_path / "balanced_final_params.csv", index=False
    )
    _skip_threshold_fitting(monkeypatch)

    dataset = EISDataPrep(tmp_path, mode="process", evaluation=True).load()

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["sub_id"] == "R1-[P2,R3]/sample_0000.csv"
    assert row["true_circuit"] == "R1-[P2,R3]"
    np.testing.assert_array_equal(row["freq"], [10000.0, 1000.0, 100.0, 10.0, 1.0])
    np.testing.assert_allclose(
        row["Z_true"],
        [
            100.0 + 0.0j,
            100.0 - 0.151037772j,
            100.0011 - 1.48644847j,
            100.3607 - 11.7323132j,
            155.7142 - 39.3695186j,
        ],
    )

    # The consolidated EISForge summary is intentionally present in the
    # fixture.  AutoREC must not attempt to parse it as a spectrum.
    output = capsys.readouterr().out
    assert "Found 1 CSV files" in output
    assert "Error reading" not in output


def test_eisforge_dataframe_is_prepared_without_intermediate_files(monkeypatch):
    """AutoREC accepts the in-memory result returned by EISForge generation."""
    frequency = np.array([100.0, 10.0, 1.0])
    final_impedance = np.array([10.0 - 1.0j, 11.0 - 2.0j, 12.0 - 3.0j])
    eisforge_rows = pd.DataFrame({"relabel_ecm": ["R1-C2"], "final_Z": [final_impedance]})
    _skip_threshold_fitting(monkeypatch)

    prep = EISDataPrep.from_eisforge(eisforge_rows, frequency, evaluation=True)

    assert len(prep.dataset) == 1
    row = prep.dataset.iloc[0]
    assert row["sub_id"] == "R1-C2/sample_0000"
    assert row["true_circuit"] == "R1-C2"
    np.testing.assert_array_equal(row["freq"], frequency)
    np.testing.assert_array_equal(row["Z_true"], final_impedance)


def test_unmarked_directory_keeps_generic_csv_discovery(tmp_path, monkeypatch):
    """Non-EISForge users retain AutoREC's existing all-CSV discovery behavior."""
    circuit_dir = tmp_path / "R1-C2"
    circuit_dir.mkdir()
    pd.DataFrame(
        {
            "freq": [100.0, 10.0, 1.0],
            "Z_real": [10.0, 11.0, 12.0],
            "Z_imag": [-1.0, -2.0, -3.0],
        }
    ).to_csv(circuit_dir / "measurement.csv", index=False)
    _skip_threshold_fitting(monkeypatch)

    dataset = EISDataPrep(tmp_path, mode="process", evaluation=True).load()

    assert len(dataset) == 1
    assert dataset.iloc[0]["sub_id"] == "R1-C2/measurement.csv"
