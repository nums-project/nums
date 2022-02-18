# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "NumS"
copyright = "2022, The NumS Team"  # pylint: disable=redefined-builtin
author = "The NumS Team"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_panels",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/logo.svg"

html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nums-project/nums",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/kanyewest",
            "icon": "fab fa-twitter-square",
            "type": "fontawesome",
        },
    ],
    "favicons": [
        {"rel": "icon", "sizes": "16x16", "href": "icon.svg"},
        {
            "rel": "icon",
            "sizes": "32x32",
            "href": "icon.svg",
        },
        {
            "rel": "apple-touch-icon",
            "sizes": "180x180",
            "href": "apple-touch-icon-180x180.png",
        },
    ],
}

http_favicon = "_static/logo.jpeg"

# Source files supported by Sphinx.
source_suffix = [".rst", ".md"]

autosummary_generate = True

# A way to automatically generate API documentation upon push to GitHub.
# https://github.com/readthedocs/readthedocs.org/issues/1139
def run_apidoc(_):
    ignore_paths = []

    argv = ["-f", "-T", "-e", "-M", "-o", "source/generated", ".."] + ignore_paths

    try:
        # Sphinx 1.7+
        from sphinx.ext import apidoc

        apidoc.main(argv)
    except ImportError:
        # Sphinx 1.6 (and earlier)
        from sphinx import apidoc

        argv.insert(0, apidoc.__file__)
        apidoc.main(argv)


def setup(app):
    app.connect("builder-inited", run_apidoc)
