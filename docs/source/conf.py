# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
from datetime import datetime
import os
import sys

import nums

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "NumS"
copyright = (  # pylint: disable=redefined-builtin
    str(datetime.now().year) + ", The NumS Team"
)
author = "The NumS Team"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_panels",
    "myst_parser",
    "sphinx_external_toc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

version = str(nums.__version__)

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_title = ""

html_logo = "_static/logo.svg"

html_favicon = "_static/favicon.ico"

html_theme_options = {
    "repository_url": "https://github.com/nums-project/nums",
    "use_repository_button": True,
    "use_issues_button": True,
}

html_css_files = [
    "css/custom.css",
]

# Source files supported by Sphinx.
source_suffix = [".rst", ".md"]
