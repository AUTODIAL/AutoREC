"""Sphinx configuration for the independently built AutoREC documentation."""

from pathlib import Path

project = "AutoREC"
author = "AutoREC developers"
copyright = "2026, AutoREC developers"
release = "development"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_parser",
    "nbsphinx_link",
    "sphinx_copybutton",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

autoapi_dirs = [str(Path(__file__).resolve().parent.parent / "src" / "autorec")]
autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
autoapi_options = ["members"]
autoapi_member_order = "bysource"
autoapi_python_class_content = "init"
autoapi_python_use_implicit_namespaces = True

napoleon_use_param = False
napoleon_use_rtype = False

nitpick_ignore_regex = [
    (
        "py:class",
        r"(?:numpy\.ndarray|os\.PathLike|pandas\.DataFrame|pathlib\.Path|tensorflow\..*)",
    ),
]
nitpick_ignore = [
    # These documented aliases are Python data objects; Sphinx treats annotations as classes.
    ("py:class", "LayerConfig"),
    ("py:class", "OptimizerConfig"),
]

nbsphinx_execute = "never"
nbsphinx_link_target_root = ".."
nbsphinx_prolog = r"""
{% set notebook = env.metadata[env.docname].get("nbsphinx-link-target") %}
.. note::

   This page shows outputs saved in the source notebook; the documentation build
   does not execute its code. To run it yourself, start Jupyter from the
   repository's ``tutorials/`` directory in ``autorec_env``.

   .. raw:: html

      <a href="{{ env.docname.rsplit("/", 1)[-1] }}.ipynb" download>Download this notebook</a>
      or <a href="https://github.com/AUTODIAL/AutoREC/blob/main/{{ notebook }}">view its source on GitHub</a>.
"""

html_theme = "furo"
html_title = "AutoREC"
html_logo = "../Logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#245b64",
        "color-brand-content": "#245b64",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8bd5d5",
        "color-brand-content": "#8bd5d5",
    },
}


def _prepare_api_docstring(app, what, name, obj, options, lines):
    """Normalize known source-docstring issues before Napoleon parses them."""
    object_name = (what, name.rsplit(".", 1)[-1])
    replacements = {
        ("class", "EIS_ECM_Env"): {
            "cache_capacity : int, default=10000": "cache_capacity : int, default=20000",
            "Dramatically speeds up training (10-100x faster)": (
                "Avoids repeated circuit fitting when a cached result is available"
            ),
        },
        ("method", "step"): {
            "cached result instantly (~1000x faster).": "cached result without refitting.",
            "Good fit: ~0.5 to 5.0 (circuit matches data, episode terminates)": (
                "Good fit: value calculated by the reward formula below; the episode terminates"
            ),
            "log10(1/(1-R²))": "log10(1/(1-R²_avg))",
        },
        ("class", "DDQN_ECM"): {
            "model checkpoints begin to be saved": "model exports begin to be saved",
            "Frequency (in trials) for saving model checkpoints": (
                "Frequency (in trials) for saving model exports"
            ),
        },
        ("function", "set_global_seed"): {
            "Must be called BEFORE importing any modules that use randomness": (
                "Call before creating environments, models, or agents"
            ),
            "Must be called BEFORE any random operations occur": (
                "Call before any random operations occur"
            ),
        },
    }
    for index, line in enumerate(lines):
        for old, new in replacements.get(object_name, {}).items():
            lines[index] = lines[index].replace(old, new)

    if object_name == ("method", "step"):
        start = next(
            (index for index, line in enumerate(lines) if line.strip() == "'info' : dict"),
            None,
        )
        if start is not None:
            end = next(
                index
                for index in range(start + 1, len(lines))
                if lines[index].strip() == "'reward' : float"
            )
            del lines[start:end]

    if object_name == ("method", "generate_ECM"):
        start = lines.index("Can be called either with:")
        end = lines.index("Parameters")
        lines[start:end] = [
            "Use ``EIS_i`` to select a row from the active environment's dataset.",
            "The ``flatten_Z`` and ``ground_truth_circuit`` parameters are currently retained",
            "for compatibility but are replaced with values from the active environment.",
            "",
        ]
        replacements = {
            "Flattened impedance data (if None, must provide ``EIS_i``).": (
                "Currently replaced by data from the active environment."
            ),
            "Optional ground truth circuit for comparison.": (
                "Currently replaced by the active environment's circuit label."
            ),
            "Maximum number of actions to try (default: ``self.action_cap``).": (
                "Maximum number of actions to try. If omitted, ``self.action_cap`` must be set."
            ),
        }
        for index, line in enumerate(lines):
            for old, new in replacements.items():
                lines[index] = lines[index].replace(old, new)


def setup(app):
    """Register documentation-only source preprocessing."""
    app.connect("autodoc-process-docstring", _prepare_api_docstring, priority=400)
