# Data preparation and dataset contract

AutoREC accepts raw EIS spectra for preprocessing and processed pandas
DataFrames for training or evaluation. Use the
{class}`~autorec.data_preparation.EISDataPrep` class directly or the
{doc}`command-line interface <cli>`.

## Raw CSV input

In `mode="process"`, `path` must be a directory. AutoREC recursively discovers
`*.csv` files, each with these exact columns:

| Column | Meaning |
| --- | --- |
| `freq` | Frequency values |
| `Z_real` | Real impedance component |
| `Z_imag` | Imaginary impedance component |

Each file is one spectrum. The file's relative path becomes `sub_id`. Its
immediate parent directory becomes `true_circuit`; when the file is directly in
the input directory, AutoREC uses the filename stem, dropping the final
underscore-delimited suffix when present.

```text
raw_eis/
├── R1-C1/
│   ├── sample_0.csv
│   └── sample_1.csv
└── R1-P1/
    └── sample_0.csv
```

The directory labels are validated as AutoEIS circuit strings only as warnings.
Use meaningful circuit labels when preparing evaluation data and inspect all
validation warnings before interpreting metrics.

## Frequency alignment and features

All model inputs must have consistent dimensions. Processing optionally applies
lin-KK preprocessing first, then interpolates each spectrum on a logarithmic
frequency grid. Lin-KK preprocessing can change the number of points.

- If `frequency_npoints` is omitted, every preprocessed spectrum must already
  have the same length.
- Explicit `frequency_bounds` must lie within every preprocessed spectrum;
  AutoREC does not extrapolate.
- A `None` lower or upper bound uses that spectrum's own minimum or maximum.
  For a truly shared grid, specify bounds common to every spectrum.

`flatten_Z` concatenates the selected features in their configured order. The
available names are `ReZ`, `ImZ`, `phi`, `mag`, `nReZ`, `nImZ`, `nphi`, and
`nmag`; the default is `[ImZ, phi, mag, nphi]`. Changing the number or order of
features changes the neural-network input contract.

Threshold calculation performs circuit-fitting work and may be expensive. A
documentation build never runs it.

## Processed DataFrame schema

For training (`evaluation=False`), validation requires:

| Column | Expected content |
| --- | --- |
| `sub_id` | Source identifier string |
| `freq` | One-dimensional frequency array |
| `Z_true` | Complex impedance array matching `freq` |
| `flatten_Z` | Fixed-length feature array |
| `chi_thresh` | Numeric fit threshold |
| `r2_thresh` | Numeric fit threshold |
| `metadata` | Preprocessing metadata |

For evaluation (`evaluation=True`), `true_circuit` is required and `metadata`
is not required. Extra columns produce warnings rather than errors. Invalid
`true_circuit` strings also produce warnings, so a successful load does not by
itself prove that every label is scientifically meaningful.

Validation checks representative types and required columns. It does not prove
data provenance, units, physical validity, or that a dataset is appropriate for
a particular experiment.

## Save and reload processed data

Prefer pickle for processed datasets because array and complex-valued columns
retain their Python/NumPy representation:

```bash
autorec preprocess \
  --input tutorials/EIS_raw_demo \
  --output data/processed_training_data.pkl \
  --mode process \
  --frequency-npoints 80 \
  --summary
```

Load and validate it later with:

```bash
autorec preprocess \
  --input data/processed_training_data.pkl \
  --mode load \
  --summary
```

Although load/save supports `.csv`, `pandas.read_csv()` does not reconstruct
serialized array and complex columns automatically. Treat CSV output as a
human-readable interchange or inspection format unless you implement and
verify an explicit conversion back to the processed schema.

Pickle files can execute code while loading. Only load datasets from trusted
sources.

## Reproducible preparation records

Keep the original raw files, preprocessing YAML, AutoREC version, feature order,
frequency bounds and point count, and whether lin-KK preprocessing was enabled.
The metadata embedded in a processed pickle records preprocessing options, but
it is not a substitute for source-data provenance.

See the {doc}`data-preparation notebook <../tutorials/data_preparation>` for a
walkthrough and the {doc}`data-preparation API <../api/data_preparation>` for
method signatures.
