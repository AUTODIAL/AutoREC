"""Tests for pure parser helpers and local simplification utilities."""

import pytest

from autorec import parser


def test_direct_and_reciprocal_sum():
    assert parser.direct_sum(1, 2, 3) == 6
    assert parser.reciprocal_sum(2, 2) == 1


@pytest.mark.parametrize(
    ("component_type", "connection", "values", "expected"),
    [
        ("R", "series", (1, 2), 3),
        ("R", "parallel", (2, 2), 1),
        ("C", "series", (2, 2), 1),
        ("C", "parallel", (1, 2), 3),
        ("L", "series", (1, 2), 3),
        ("L", "parallel", (2, 2), 1),
    ],
)
def test_combine_components(component_type, connection, values, expected):
    assert (
        parser.combine_components(
            *values, component_type=component_type, connection=connection
        )
        == expected
    )


def test_attach_values_to_structure_handles_resistors_and_cpe_values():
    structure = ["s", "R1", "P2"]
    parameters = {"R1": 10.0, "P2w": 0.5, "P2n": 0.8}

    # CPE/P elements carry both weight and exponent parameters.
    assert parser._attach_values_to_structure(structure, parameters) == [
        {"s": None},
        {"R1": 10.0},
        {"P2": {"w": 0.5, "n": 0.8}},
    ]


def test_get_structure_and_values_from_simplified_structure():
    structure = [{"s": None}, {"R1": 3.0}, {"C2": 4.0}]

    assert parser._get_structure_only(structure) == ["s", "R1", "C2"]
    assert parser._get_values_only(structure) == {"R1": 3.0, "C2": 4.0}


def test_simplify_p_replaces_low_exponent_with_resistor():
    circuit, params = parser._simplify_P("R1-P2", {"R1": 10.0, "P2w": 0.25, "P2n": 0.05})

    # Near-zero CPE exponents behave like resistors.
    assert circuit == "R1-R2"
    assert params == {"R1": 10.0, "R2": 4.0}


def test_simplify_p_replaces_high_exponent_with_capacitor():
    circuit, params = parser._simplify_P("R1-P2", {"R1": 10.0, "P2w": 0.25, "P2n": 0.95})

    # Near-one CPE exponents behave like capacitors.
    assert circuit == "R1-C2"
    assert params == {"R1": 10.0, "C2": 0.25}
