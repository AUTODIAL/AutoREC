"""Tests for lightweight utilities that do not run circuit fitting or plotting workflows."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from autorec import utils


class FakeEC:
    """Small circuit parser stand-in for state encoding and chromosome parsing tests."""

    @staticmethod
    def karva_to_tree(karva):
        return list(karva[:2])

    @staticmethod
    def tree_to_circuit(tree):
        return ("R0-[R1,P2]", None)


def test_save_evaluation_results_writes_csv_and_pickle(tmp_path):
    df = pd.DataFrame({"eis_index": [0, 1], "reward": [0.5, 0.75]})

    csv_path, pkl_path = utils.save_evaluation_results(df, "run-1", tmp_path)

    assert csv_path == tmp_path / "eval_results_run-1.csv"
    assert pkl_path == tmp_path / "eval_results_run-1.pkl"
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), df)
    pd.testing.assert_frame_equal(pd.read_pickle(pkl_path), df)


def test_chi_obj_func_computes_weighted_complex_residual():
    z = np.array([1 + 1j, 2 + 0j])
    z_pred = np.array([2 + 1j, 1 + 0j])

    assert utils.chi_obj_func(z, z_pred) == pytest.approx(0.375)


def test_is_nonblank_image_detects_blank_and_nonblank_images(tmp_path):
    blank = tmp_path / "blank.png"
    nonblank = tmp_path / "nonblank.png"
    Image.new("RGB", (4, 4), "white").save(blank)
    Image.new("RGB", (4, 4), "black").save(nonblank)

    assert utils._is_nonblank_image(blank) is False
    assert utils._is_nonblank_image(nonblank) is True


def test_state_encode_marks_non_coding_region(monkeypatch):
    monkeypatch.setattr(utils.ae.core, "ec", FakeEC)
    elements = ["+", "/", "R", "L", "P", "X"]

    encoded = utils.state_encode("+RRR", elements)

    # FakeEC exposes only the first two chromosome positions as coding; the rest become X.
    expected = np.eye(len(elements))[[0, 2, 5, 5]].reshape(-1)
    np.testing.assert_array_equal(encoded, expected)


def test_parse_state_to_circuit_uses_coding_region(monkeypatch):
    monkeypatch.setattr(utils, "ec", FakeEC)

    circuit, coding_length, coding = utils.parse_state_to_circuit("+RRR")

    assert circuit == "R0-[R1,P2]"
    assert coding_length == 2
    assert coding == "+R"


def test_action_validity_returns_zero_for_unchanged_state(monkeypatch):
    monkeypatch.setattr(utils, "ec", FakeEC)
    monkeypatch.setattr(utils, "karva_to_circuit", lambda karva: "R0-[R1,P2]")
    monkeypatch.setattr(utils, "validity_check", lambda circuit: True)

    assert utils.action_validity("+RRR", "R", 1) == 0


def test_action_validity_returns_zero_for_non_coding_position(monkeypatch):
    monkeypatch.setattr(utils, "ec", FakeEC)
    monkeypatch.setattr(utils, "karva_to_circuit", lambda karva: "R0-[R1,P2]")
    monkeypatch.setattr(utils, "validity_check", lambda circuit: True)

    # Position 3 is outside FakeEC's two-position coding region.
    assert utils.action_validity("+RRR", "L", 3) == 0


def test_action_validity_returns_one_for_valid_mutation(monkeypatch):
    monkeypatch.setattr(utils, "ec", FakeEC)
    monkeypatch.setattr(utils, "karva_to_circuit", lambda karva: "R0-[R1,P2]")
    monkeypatch.setattr(utils, "validity_check", lambda circuit: True)

    assert utils.action_validity("+RRR", "L", 1) == 1


def test_validity_check_requires_valid_circuit_resistor_and_parallel_route(monkeypatch):
    # validity_check delegates syntax/resistor checks to AutoEIS parser helpers.
    parser = SimpleNamespace(
        validate_circuit=lambda circuit: True,
        find_ohmic_resistors=lambda circuit: ["R0"] if "R0" in circuit else [],
    )
    monkeypatch.setattr(utils.ae, "parser", parser)

    assert utils.validity_check("R0-[R1,P2]") is True
    assert utils.validity_check("C0-[P1,P2]") is False
    assert utils.validity_check("R0-R1") is False


def test_get_parameter_bounds_maps_autoeis_parameter_types(monkeypatch):
    parser = SimpleNamespace(get_parameter_types=lambda circuit: ["R", "Pw", "Pn", "C"])
    monkeypatch.setattr(utils.ae, "parser", parser)

    lower, upper = utils.get_parameter_bounds("R0-P1-C2")

    assert lower == (0.0, 0.0, 0.0, 0.0)
    assert upper == (1e9, 1e9, 1.0, 10.0)
