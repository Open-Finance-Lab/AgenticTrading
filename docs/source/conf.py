# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path Setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath(''))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FinAgent Orchestration Documentation'
copyright = '2025, FinAgent Orchestration'
author = 'FinAgent Orchestration'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "nbsphinx",
    "nbsphinx_link",
]

templates_path = ['_templates']

master_doc = "index"

exclude_patterns = []

numfig = True
math_numfig = True

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "Open-Finance-Lab", # Username
    "github_repo": "FinAgent-Orchestration", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
}

latex_elements = {
    'preamble': r'''
    \usepackage{amsmath}
    \usepackage{braket}
    \usepackage{algorithm}
    \usepackage{algorithmic}
    '''
}
