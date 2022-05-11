# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information

project = "Entity Gym"
copyright = "2021-2022, Clemens Winter"
author = "Clemens Winter"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.inheritance_diagram",
    "autoapi.sphinx",
]
# extensions = ["sphinx.ext.autodoc", "sphinx.ext.inheritance_diagram", "autoapi.sphinx"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

autoapi_modules = {
    "entity_gym": {
        "prune": True,
    }
}

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

autosummary_generate = True

autosummary_ignore_module_all = False
autosummary_imported_members = False
