"""Sphinx configuration for the independently built AutoREC documentation."""

project = "AutoREC"
author = "AutoREC developers"
copyright = "2026, AutoREC developers"
release = "development"

extensions = ["myst_parser", "nbsphinx_link", "sphinx_copybutton"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

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
