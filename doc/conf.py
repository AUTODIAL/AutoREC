"""Sphinx configuration for the independently built AutoREC documentation."""

project = "AutoREC"
author = "AutoREC developers"
copyright = "2026, AutoREC developers"
release = "development"

extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "AutoREC documentation"
