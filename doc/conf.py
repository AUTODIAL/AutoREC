"""Sphinx configuration for the independently built AutoREC documentation."""

project = "AutoREC"
author = "AutoREC developers"
copyright = "2026, AutoREC developers"
release = "development"

extensions = ["myst_parser", "sphinx_copybutton"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

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
