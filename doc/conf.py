# -*- coding: utf-8 -*-
#
# FeatureMapper documentation build configuration file, created by
# sphinx-quickstart on Thu May 15 23:32:51 2014.
#

import os, sys

sys.path.append(os.path.abspath('.'))

from .builder.shared_conf import *

paths = ['../param/', '../imagen/', '../dataviews/', '.', '..']
add_paths(paths)

# General information about the project.
project = u'FeatureMapper'
copyright = u'2014, IOAM'
ioam_project = 'featuremapper'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.1'
# The full version, including alpha/beta/rc tags.
release = '0.1a'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'test_data', 'reference_data', 'nbpublisher',
                    'builder']


# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = project


# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'


# -- Options for LaTeX output ---------------------------------------------


# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', project+'.tex', project+ ' Documentation',
   u'IOAM', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', ioam_project, project + ' Documentation',
     [u'IOAM'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', project, project + ' Documentation',
   u'IOAM', project, 'One line description of project.',
   'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None,
                       'http://ioam.github.io/param/': None,
                       'http://ioam.github.io/imagen/': None,
                       'http://ioam.github.io/dataviews/': None,
                       'http://ipython.org/ipython-doc/2/' : None}

from builder.paramdoc import param_formatter
from nbpublisher import nbbuild

def setup(app):
    app.connect('autodoc-process-docstring', param_formatter)
    nbbuild.setup(app)
