# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '../src')))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'deepFRI2'
copyright = '2024, Patryk Orzechowski, Tomasz Kościółek'
author = 'Patryk Orzechowski, Tomasz Kościółek'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', # Core library for html generation from docstrings
    'sphinx.ext.viewcode', # Add links to highlighted source code
    'sphinx.ext.napoleon', # Support for NumPy and Google style docstrings
    'sphinx.ext.doctest', # Test snippets in the documentation¶
    'sphinx.ext.coverage', # Collect doc coverage stats
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.githubpages', # Creates .nojekyll
    'sphinx.ext.viewcode', # Add links to highlighted source code
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {'navigation_depth': 2}