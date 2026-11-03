Progressive-spectrum components
===============================

Progressive execution keeps physical fitting data separate from the fixed-size
neural-network observation. A spectrum source reads one dataset row, a selector
chooses cumulative physical points, an encoder creates the observation, and a
transition policy selects the ECM state used by the next stage.

Data and component interfaces
-----------------------------

.. autoapiclass:: autorec.progressive_spectrum.DataFrameSpectrumSource
   :members: read

.. autoapiclass:: autorec.progressive_spectrum.SpectrumWindowSelection

.. autoapiclass:: autorec.progressive_spectrum.SpectrumView

.. autoapiclass:: autorec.progressive_spectrum.SpectrumSource
   :members: read

.. autoapiclass:: autorec.progressive_spectrum.SpectrumObservationEncoder
   :members: encode

.. autoapiclass:: autorec.progressive_spectrum.SpectrumFeatureNormalizer
   :members: normalize

.. autoapiclass:: autorec.progressive_spectrum.SpectrumDownsampler
   :members: select_indices

.. autoapiclass:: autorec.progressive_spectrum.SpectrumWindowSelector
   :members: select

.. autoapiclass:: autorec.progressive_spectrum.StageTransitionPolicy
   :members: decide

Built-in implementations
------------------------

.. autoapiclass:: autorec.progressive_spectrum.EqualPointWindowSelector
   :members: select

.. autoapiclass:: autorec.progressive_spectrum.InterpolatingSpectrumEncoder
   :members: encode

.. autoapiclass:: autorec.progressive_spectrum.RawSpectrumEncoder
   :members: encode

.. autoapiclass:: autorec.progressive_spectrum.SentinelSafeMinMaxNormalizer
   :members: normalize

.. autoapiclass:: autorec.progressive_spectrum.LegacyEISFeatureNormalizer
   :members: normalize

.. autoapiclass:: autorec.progressive_spectrum.UniformIndexDownsampler
   :members: select_indices

Stage transitions and runtime configuration
-------------------------------------------

.. autoapiclass:: autorec.progressive_spectrum.ProgressiveStageOutcome

.. autoapiclass:: autorec.progressive_spectrum.StageTransitionDecision

.. autoapiclass:: autorec.progressive_spectrum.CarryBestStatePolicy
   :members: decide

.. autoapiclass:: autorec.progressive_spectrum.ProgressiveSpectrumRuntimeConfig
   :members: encoder_for, metadata

.. autoapifunction:: autorec.progressive_spectrum.build_spectrum_view

.. autoapifunction:: autorec.progressive_spectrum.progressive_runtime_state_dict

.. autoapifunction:: autorec.progressive_spectrum.restore_progressive_runtime_state
