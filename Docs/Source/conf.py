import os
import sys
from unittest.mock import MagicMock
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'Torsion Angular Bin Strings'
copyright = '2025, Jessica Braun'
author = 'Jessica Braun'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Mock heavy imports for documentation generation --------------------------
# This is to avoid importing large libraries that are not needed for documentation generation.
MOCK_MODULES = [
    'mdtraj'
]

for mod in MOCK_MODULES:
    sys.modules[mod] = MagicMock()