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


def test_eisforge_sample_csv_is_consumed_from_disk(tmp_path, monkeypatch):
    """AutoREC accepts the numeric sample CSV exported by EISForge."""
    circuit_dir = tmp_path / "R1-[P2,R3]"
    circuit_dir.mkdir()
    pd.DataFrame(
        {
            "freq": [10000.0, 1000.0, 100.0, 10.0, 1.0],
            "Z_real": [100.0, 100.0, 100.0011, 100.3607, 155.7142],
            "Z_imag": [0.00, -0.151037772, -1.48644847, -11.7323132, -39.3695186],
        }
    ).to_csv(circuit_dir / "sample_0000.csv", index=False)
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


def test_invalid_eisforge_sample_reports_missing_columns(tmp_path, capsys):
    """A malformed exported sample reports the broken disk contract."""
    circuit_dir = tmp_path / "R1-C2"
    circuit_dir.mkdir()
    pd.DataFrame({"freq": [100.0], "Z_real": [10.0]}).to_csv(
        circuit_dir / "sample_0000.csv", index=False
    )

    prep = EISDataPrep(tmp_path, mode="process", evaluation=True)
    with np.testing.assert_raises(ValueError):
        prep.load()

    output = capsys.readouterr().out
    assert "Missing required columns ['Z_imag']" in output
