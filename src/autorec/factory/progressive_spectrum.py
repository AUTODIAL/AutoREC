"""Build progressive-spectrum runtime components from YAML-compatible mappings."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from autorec.progressive_spectrum import (
    CarryBestStatePolicy,
    DataFrameSpectrumSource,
    EqualPointWindowSelector,
    InterpolatingSpectrumEncoder,
    LegacyEISFeatureNormalizer,
    ProgressiveSpectrumRuntimeConfig,
    RawSpectrumEncoder,
    SentinelSafeMinMaxNormalizer,
    UniformIndexDownsampler,
)


WINDOW_SELECTOR_TYPES = {"EqualPointWindowSelector": EqualPointWindowSelector}
OBSERVATION_ENCODER_TYPES = {
    "InterpolatingSpectrumEncoder": InterpolatingSpectrumEncoder,
    "RawSpectrumEncoder": RawSpectrumEncoder,
}
FEATURE_NORMALIZER_TYPES = {
    "SentinelSafeMinMaxNormalizer": SentinelSafeMinMaxNormalizer,
    "LegacyEISFeatureNormalizer": LegacyEISFeatureNormalizer,
}
DOWNSAMPLER_TYPES = {"UniformIndexDownsampler": UniformIndexDownsampler}
SPECTRUM_SOURCE_TYPES = {
    "PreparedSpectrumSource": partial(
        DataFrameSpectrumSource,
        name="prepared",
        frequency_column="freq",
        impedance_column="Z_true",
    ),
    "RetainedSpectrumSource": partial(
        DataFrameSpectrumSource,
        name="retained",
        frequency_column="source_freq",
        impedance_column="source_Z",
    ),
    "DataFrameSpectrumSource": DataFrameSpectrumSource,
}
TRANSITION_POLICY_TYPES = {"CarryBestStatePolicy": CarryBestStatePolicy}

_ALLOWED_KEYS = {
    "enabled",
    "num_windows",
    "window_selector",
    "observation_encoder",
    "spectrum_source",
    "stage_transition_policy",
}


def _component_config(
    config: Mapping[str, Any], key: str, default_type: str
) -> tuple[str, dict[str, Any]]:
    """Return a validated component type and copied parameter mapping."""
    component = config.get(key, {})
    if component is None:
        component = {}
    if not isinstance(component, Mapping):
        raise TypeError(f"progressive_spectrum.{key} must be a mapping")
    unknown = set(component) - {"type", "parameters"}
    if unknown:
        raise KeyError(f"Unknown progressive_spectrum.{key} keys: {sorted(unknown)}")
    component_type = component.get("type", default_type)
    if not isinstance(component_type, str) or not component_type:
        raise ValueError(f"progressive_spectrum.{key}.type must be a non-empty string")
    parameters = component.get("parameters", {})
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise TypeError(f"progressive_spectrum.{key}.parameters must be a mapping")
    return component_type, dict(parameters)


def _registered_component(registry, component_type: str, config_key: str):
    """Resolve a registered component class with a configuration-focused error."""
    try:
        return registry[component_type]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown progressive_spectrum.{config_key} type {component_type!r}. "
            f"Available types: {choices}"
        ) from exc


def _resolve_encoder_dependencies(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve optional normalizer and downsampler mappings in encoder parameters."""
    resolved = dict(parameters)
    nested_components = (
        ("normalizer", FEATURE_NORMALIZER_TYPES, "SentinelSafeMinMaxNormalizer"),
        ("downsampler", DOWNSAMPLER_TYPES, "UniformIndexDownsampler"),
    )
    for key, registry, default_type in nested_components:
        if key not in resolved:
            continue
        component_type, component_parameters = _component_config(resolved, key, default_type)
        component_class = _registered_component(
            registry,
            component_type,
            f"observation_encoder.parameters.{key}",
        )
        resolved[key] = component_class(**component_parameters)
    return resolved


def progressive_spectrum_builder(
    args: Mapping[str, Any], *, training_dataprep, evaluation_dataprep=None
) -> ProgressiveSpectrumRuntimeConfig:
    """Resolve YAML configuration into validated progressive runtime components.

    ``enabled`` defaults to true only when ``num_windows`` exceeds one. Thus an omitted
    section, an explicitly disabled section, or ``num_windows: 1`` retains the legacy
    full-spectrum computation unless ``enabled: true`` is requested.
    """
    if not isinstance(args, Mapping):
        raise TypeError("progressive_spectrum must be a mapping")
    config = dict(args)
    unknown = set(config) - _ALLOWED_KEYS
    if unknown:
        raise KeyError(f"Unknown progressive_spectrum keys: {sorted(unknown)}")

    num_windows = config.get("num_windows", 1)
    if isinstance(num_windows, bool) or not isinstance(num_windows, int):
        raise TypeError("progressive_spectrum.num_windows must be an integer")
    if num_windows <= 0:
        raise ValueError("progressive_spectrum.num_windows must be greater than zero")
    enabled = config.get("enabled", num_windows > 1)
    if not isinstance(enabled, bool):
        raise TypeError("progressive_spectrum.enabled must be a boolean")

    selector_type, selector_parameters = _component_config(
        config, "window_selector", "EqualPointWindowSelector"
    )
    encoder_type, encoder_parameters = _component_config(
        config, "observation_encoder", "InterpolatingSpectrumEncoder"
    )
    policy_type, policy_parameters = _component_config(
        config, "stage_transition_policy", "CarryBestStatePolicy"
    )
    source_type, source_parameters = _component_config(
        config, "spectrum_source", "PreparedSpectrumSource"
    )

    if not enabled:
        return ProgressiveSpectrumRuntimeConfig(False, num_windows)
    if training_dataprep is None:
        raise ValueError(
            "enabled progressive_spectrum configuration requires a training EISDataPrep "
            "instance; configure the pipeline dataprep section"
        )
    if evaluation_dataprep is None:
        evaluation_dataprep = training_dataprep

    configured_windows = selector_parameters.pop("num_windows", num_windows)
    if configured_windows != num_windows:
        raise ValueError(
            "window_selector.parameters.num_windows must match "
            "progressive_spectrum.num_windows"
        )
    selector_class = _registered_component(
        WINDOW_SELECTOR_TYPES, selector_type, "window_selector"
    )
    encoder_class = _registered_component(
        OBSERVATION_ENCODER_TYPES, encoder_type, "observation_encoder"
    )
    policy_class = _registered_component(
        TRANSITION_POLICY_TYPES, policy_type, "stage_transition_policy"
    )
    source_factory = _registered_component(
        SPECTRUM_SOURCE_TYPES, source_type, "spectrum_source"
    )

    selector = selector_class(num_windows=num_windows, **selector_parameters)
    training_encoder = encoder_class(
        training_dataprep, **_resolve_encoder_dependencies(encoder_parameters)
    )
    evaluation_encoder = (
        training_encoder
        if evaluation_dataprep is training_dataprep
        else encoder_class(
            evaluation_dataprep,
            **_resolve_encoder_dependencies(encoder_parameters),
        )
    )
    return ProgressiveSpectrumRuntimeConfig(
        enabled=True,
        num_windows=num_windows,
        selector=selector,
        training_encoder=training_encoder,
        evaluation_encoder=evaluation_encoder,
        transition_policy=policy_class(**policy_parameters),
        spectrum_source=source_factory(**source_parameters),
    )
