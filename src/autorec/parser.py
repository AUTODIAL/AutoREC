"""Contain some functions to simplify circuit AND recompute the component values of the
simplified circuit. It is a modification to `autoeis.parser.simplify`.
"""

from copy import deepcopy
import re
from typing import Tuple, List, Union, Dict, Optional
import numpy as np
import autoeis as ae


def direct_sum(*values):
    return np.sum(values)


def reciprocal_sum(*values):
    return 1.0 / np.sum(1.0 / np.array(values))


def combine_R(*values, connection):
    """Combine R values based on the connection type (series or parallel)."""
    if connection == "series":
        return direct_sum(*values)
    elif connection == "parallel":
        return reciprocal_sum(*values)


def combine_C(*values, connection):
    """Combine C values based on the connection type (series or parallel)."""
    if connection == "series":
        return reciprocal_sum(*values)
    elif connection == "parallel":
        return direct_sum(*values)


def combine_L(*values, connection):
    """Combine L values based on the connection type (series or parallel)."""
    return combine_R(*values, connection=connection)


def combine_components(*values, component_type, connection):
    """Combine component values based on the component type and connection type."""
    if component_type == "R":
        return combine_R(*values, connection=connection)
    elif component_type == "C":
        return combine_C(*values, connection=connection)
    elif component_type == "L":
        return combine_L(*values, connection=connection)


def _simplify_P(
    circuit: str,
    params: Dict[str, float],
    Pn_low: Optional[float] = 0.1,
    Pn_high: Optional[float] = 0.9,
) -> Tuple[str, Dict[str, float]]:
    """
    Simplify the circuit by replacing P with either R or C based on Pn value.

    If `Pn < Pn_low`, then P is replaced with R. If `Pn > Pn_high`, then P is replaced with C.

    Parameters
    ----------
    circuit : str
        CDC string representation of the circuit to be simplified.
    params : dict
        Dictionary of parameters for the circuit, where keys are component names and values
        are the corresponding component values.
    Pn_low : float, optional
        Threshold for replacing P with R, i.e., replace P with R when Pn < Pn_low.
    Pn_high : float, optional
        Threshold for replacing P with C, i.e., replace P with C when Pn > Pn_high.

    Returns
    -------
    tuple
        The simplified circuit (str) and the updated parameters (dict).
    """
    circuit = deepcopy(circuit)
    # Replace P with R or C
    params_copy = {}
    for name, value in params.items():
        # Parameter name that needs attention is "PXn", where "X" indicates the index of the
        # component.
        match = re.match(r"P(\d+)n", name)
        if match:
            index = int(match.group(1))
            if value < Pn_low:
                # Replace P with R
                circuit = re.sub(f"P{index}", f"R{index}", circuit)
                params_copy[f"R{index}"] = 1 / params[f"P{index}w"]
                if f"P{index}w" in params_copy:
                    del params_copy[f"P{index}w"]
            elif value > Pn_high:
                # Replace P with C
                circuit = re.sub(f"P{index}", f"C{index}", circuit)
                params_copy[f"C{index}"] = params[f"P{index}w"]
                if f"P{index}w" in params_copy:
                    del params_copy[f"P{index}w"]
            else:
                params_copy[name] = value
        else:
            params_copy[name] = value
    return circuit, params_copy


def _attach_values_to_structure(
    structure: Union[List, str], parameters: Dict[str, float]
) -> List:
    """
    Given the structure representation of the circuit (output of `ae._parser_to_structure`),
    attach the component values to the structure.

    Parameters
    ----------
    structure : list or str
        The structure representation of the circuit in the form of nested lists of strings.
    parameters : dict
        Dictionary of parameters for the circuit, where keys are component names and values
        are the corresponding component values.

    Returns
    -------
    list
        The structure representation of the circuit with component values attached, in the
        form of nested lists of dictionaries, where each dictionary has the component name as
        the key and the component value as the value.
    """
    structure = deepcopy(structure)
    parameters = deepcopy(parameters)
    if isinstance(structure, list):
        connection = structure[0]  # "s" or "p"
        return [{connection: None}] + [
            _attach_values_to_structure(item, parameters) for item in structure[1:]
        ]
    # structure is a component string like "R1", "C5", "P4"
    comp = structure
    if comp.startswith("P"):
        return {comp: {"w": parameters.get(f"{comp}w"), "n": parameters.get(f"{comp}n")}}
    else:
        return {comp: parameters.get(comp)}


def _simplify_structure_with_values(
    struct: List[Union[Dict[str, float], List]],
) -> Union[List, Dict[str, float]]:
    """
    Recursively simplifies the nested list structure and compute the equivalent component
    values in a single pass

    Parameters
    ----------
    struct : list or dict
        The structure representation of the circuit with component values attached, in the
        form of nested lists of dictionaries, where each dictionary has the component name as
        the key and the component value as the value.

    Returns
    -------
    list or dict
        The simplified structure representation of the circuit with component values attached,
        in the form of nested lists of dictionaries, where each dictionary has the component
        name as the key and the component value as the value. If the structure can be fully
        simplified to a single component, then a dictionary is returned instead of a list.
    """
    # Base case: An element ({string: float}) cannot be simplified further.
    if isinstance(struct, dict):
        return struct

    op = struct[0]
    components = struct[1:]

    # Recursively simplify children first (post-order traversal).
    simplified_components = [_simplify_structure_with_values(c) for c in components]

    # --- Apply Simplification Logic ---
    final_components = []
    seen_types = set()
    simplifiable_types = {"R", "C", "L"}

    if list(op)[0] == "s":  # Series simplification
        for comp in simplified_components:
            comp_type = ""
            if isinstance(comp, dict):
                comp_type = list(comp)[0][0]
            else:
                if list(comp[0])[0] == "p":  # It's a parallel block
                    comp_type = "p"

            if comp_type in simplifiable_types:
                if comp_type not in seen_types:
                    seen_types.add(comp_type)
                    final_components.append(comp)
                else:
                    # Update the component value
                    for ii, cc in enumerate(final_components):
                        cc_name = list(cc)[0]
                        if comp_type in cc_name:
                            val1 = final_components[ii][cc_name]
                            val2 = list(comp.values())[0]
                            final_components[ii][cc_name] = combine_components(
                                val1, val2, component_type=comp_type, connection="series"
                            )
                            break
            else:  # P elements or parallel blocks
                final_components.append(comp)

    elif list(op)[0] == "p":  # Parallel simplification
        for comp in simplified_components:
            is_atomic = isinstance(comp, dict) and list(comp)[0] != "p"
            if is_atomic:
                comp_type = list(comp)[0][0]
                if comp_type in simplifiable_types:
                    if comp_type not in seen_types:
                        seen_types.add(comp_type)
                        final_components.append(comp)
                    else:
                        # Update the component value
                        for ii, cc in enumerate(final_components):
                            cc_name = list(cc)[0]
                            if comp_type in cc_name:
                                val1 = final_components[ii][cc_name]
                                val2 = list(comp.values())[0]
                                final_components[ii][cc_name] = combine_components(
                                    val1, val2, component_type=comp_type, connection="parallel"
                                )
                                break
                else:  # P element
                    final_components.append(comp)
            else:  # Complex branch
                final_components.append(comp)

    # If only one component remains after simplification, unwrap it.
    if len(final_components) == 1:
        return final_components[0]

    return [op] + final_components


def _get_structure_only(struct: Union[List, Dict[str, float]]) -> Union[List, str]:
    """
    Given the structure representation of the circuit with component values attached, extract
    the structure only. The output will the the output of `ae.parser._parse_to_structure`.
    """
    if isinstance(struct, dict):
        return next(iter(struct))
    elif isinstance(struct, str):
        return struct
    elif isinstance(struct, list):
        result = [_get_structure_only(c) for c in struct]
        if len(result) == 1:
            return result[0]
        return result
    else:
        raise TypeError(f"Unsupported type: {type(struct)}")


def _get_values_only(struct: Union[List, Dict[str, float]]) -> Dict[str, float]:
    """
    Given the structure representation of the circuit with component values attached, extract
    the values only. The output will be a flat dictionary of component names and their
    corresponding values.
    """
    out = {}
    if isinstance(struct, list):
        for item in struct:
            out.update(_get_values_only(item))
    elif isinstance(struct, dict):
        for k, v in struct.items():
            if v is None:
                continue
            if isinstance(v, dict):
                for subk, subv in v.items():
                    out[f"{k}{subk}"] = subv
            else:
                out[k] = v
    return out


def _move_ohmic_resistors_to_the_beginning(
    circuit: str, parameters: Optional[Dict[str, float]] = None
) -> Union[str, Tuple[str, Dict[str, float]]]:
    """
    Move the the ohmic resistors to the beginning of the circuit, which is how many usually
    present the ohmic resistors in their ECM.

    Parameters
    ----------
    circuit : str
        CDC string representation of the circuit to be rearranged.
    parameters : dict, optional
        Dictionary of parameters for the circuit, where keys are component names and values
        are the corresponding component values. If provided, the parameters will be rearranged
        accordingly and returned as the second element of the output tuple.

    Returns
    -------
    str or tuple
        If `parameters` is not provided, returns the rearranged circuit as a string. If
        `parameters` is provided, returns a tuple containing the rearranged circuit as a
        string and the rearranged parameters as a dictionary.
    """
    ohmic_resistors = ae.parser.find_ohmic_resistors(circuit)
    struct = ae.parser._parse_to_structure(circuit)
    struct = [comp for comp in struct if comp not in ohmic_resistors]
    # We assume that the first element is always "s"
    for R in ohmic_resistors[::-1]:
        struct.insert(1, R)
    if parameters is not None:
        # Update the parameters
        new_parameters = {name: parameters[name] for name in ohmic_resistors}
        for name, val in parameters.items():
            if name not in ohmic_resistors:
                new_parameters[name] = val
        return ae.parser._stringify_structure(struct), new_parameters
    else:
        return ae.parser._stringify_structure(struct)


def simplify(
    circuit: str,
    parameters: Optional[Dict[str, float]] = None,
    Pn_low: Optional[float] = 0.1,
    Pn_high: Optional[float] = 0.9,
) -> Union[str, Tuple[str, Dict[str, float]]]:
    """
    Simplify the circuit by applying the following steps:
    1. If `parameters` is provided, simplify P elements based on their Pn values and update
       the circuit and parameters accordingly.
    2. Simplify the circuit structure by combining series and parallel components and
       computing the equivalent component values.
    3. Move the ohmic resistors to the beginning of the circuit.

    Parameters
    ----------
    circuit : str
        CDC string representation of the circuit to be simplified.
    parameters : dict, optional
        Dictionary of parameters for the circuit, where keys are component names and values
        are the corresponding component values. If provided, the parameters will be updated
        accordingly.
    Pn_low : float, optional
        Threshold for replacing P with R, i.e., replace P with R when Pn < Pn_low. Only used
        if `parameters` is provided.
    Pn_high : float, optional
        Threshold for replacing P with C, i.e., replace P with C when Pn > Pn_high. Only
        used if `parameters` is provided.

    Returns
    -------
    str or tuple
        If `parameters` is not provided, returns the simplified circuit as a string. If
        `parameters` is provided, returns a tuple containing the simplified circuit as a
        string and the updated parameters as a dictionary.
    """
    circuit = deepcopy(circuit)
    if parameters is None:
        # Just simplify series and parallel connections
        circuit = ae.parser.simplify(circuit)
    else:
        parameters = deepcopy(parameters)
        # Simplify P
        circuit, parameters = _simplify_P(circuit, parameters, Pn_low, Pn_high)
        # Simplify series and parallel connections and compute equivalent values
        structure = ae.parser._parse_to_structure(circuit)
        struct_with_vals = _attach_values_to_structure(structure, parameters)
        struct_with_vals = _simplify_structure_with_values(struct_with_vals)
        structure = _get_structure_only(struct_with_vals)
        circuit = ae.parser._stringify_structure(structure)
        parameters = _get_values_only(struct_with_vals)
    # Move ohmic resistors to the beginning
    if len(ae.parser.get_component_labels(circuit)) == 1:
        # There is only 1 element
        if parameters is None:
            return circuit
        else:
            return circuit, parameters
    else:
        return _move_ohmic_resistors_to_the_beginning(circuit, parameters)


if __name__ == "__main__":
    from pprint import pprint

    circuit = "[C1,P2-[C3,P4]]-[R5,[P6,R7,R8]]-L9-R10-[L11,L12]-R13"
    print("Original Circuit:")
    print(circuit)
    parameters = {
        name: val
        for name, val in zip(
            ae.parser.get_parameter_labels(circuit),
            ae.utils.generate_initial_guess(circuit),
        )
    }
    parameters["P2n"] = 0.95
    parameters["P4n"] = 0.95
    parameters["P6n"] = 0.01
    print("\nOriginal Parameters:")
    pprint(parameters)

    # Test starts here
    circuit2, parameters2 = _simplify_P(circuit, parameters)
    structure = ae.parser._parse_to_structure(circuit2)
    struct = _attach_values_to_structure(structure, parameters2)
    print("\nStructure with values attached:")
    pprint(struct)

    struct_with_vals = _simplify_structure_with_values(struct)
    print("\nSimplified structure with values attached:")
    pprint(struct_with_vals)
    print("\nExtracted structure only:")
    pprint(_get_structure_only(struct_with_vals))
    print("\nExtracted values only:")
    pprint(_get_values_only(struct_with_vals))
    print("\nStringified simplified structure:")
    pprint(_move_ohmic_resistors_to_the_beginning(circuit, parameters))
    print("\nFinal Simplified Circuit without parameters:")
    pprint(simplify(circuit))
    print("\nFinal Simplified Circuit when parameters are provided:")
    pprint(simplify(circuit, parameters))
