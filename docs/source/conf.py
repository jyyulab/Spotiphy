import os
import sys
sys.path.insert(0, os.path.abspath('../../spotiphy'))
sys.path.insert(0, os.path.abspath('../../'))

project = 'Spotiphy'
copyright = '2023, Ziqian Zheng, Jiyuan Yang'
author = 'Ziqian Zheng, Jiyuan Yang'
release = '0.1.2'

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinxcontrib.napoleon',
]
templates_path = ['_templates']
exclude_patterns = []
numpydoc_show_class_members = False

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sphinx = False
