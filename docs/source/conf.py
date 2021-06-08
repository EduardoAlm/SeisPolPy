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

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = 'SeisPolPy'
copyright = '2021, Eduardo Rodrigues de Almeida'
author = 'Eduardo Rodrigues de Almeida'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_github_changelog'
]

sphinx_github_changelog_token = os.environ.get('changelog_token')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_mock_imports = [ 'adjoint', 'forw_op', 'diags']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_logo = "img/logo_no_back.png"

github_url = "https://github.com/EduardoAlm/SeisPolPy"

html_theme_options = {
    'logo_only': True,
    'style_external_links': True,
    'style_nav_header_background': "#000000",
    'vcs_pageview_mode': 'blob',
}

html_context = {
    'display_github': True,    
    'github_user': 'EduardoAlm',
    'github_repo': 'SeisPolPy',
    'github_version': '0.0.1' 
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/default.css',
]

