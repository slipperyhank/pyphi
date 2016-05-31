#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyPhi documentation build configuration file, created by
# sphinx-quickstart on Fri Jan 17 11:15:55 2014.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
from pyphi import __version__, __author__


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinxcontrib.napoleon'
]

mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'PyPhi'
copyright = '2014 ' + __author__

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = 'v' + __version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = Fale

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# Substitutions for math elements to make docstrings more readable
rst_prolog = """
.. |big_phi| replace:: :math:`\\Phi`
.. |big_phi_max| replace:: :math:`\\Phi^{\\textrm{max}}`
.. |small_phi| replace:: :math:`\\varphi`
.. |small_phi_max| replace:: :math:`\\varphi^{\\textrm{max}}`
.. |A| replace:: :math:`A`
.. |B| replace:: :math:`B`
.. |C| replace:: :math:`C`
.. |D| replace:: :math:`D`
.. |E| replace:: :math:`E`
.. |F| replace:: :math:`F`
.. |AB| replace:: :math:`AB`
.. |CD| replace:: :math:`CD`
.. |DE| replace:: :math:`DE`
.. |FG| replace:: :math:`FG`
.. |ABC| replace:: :math:`ABC`
.. |CDE| replace:: :math:`CDE`
.. |DEF| replace:: :math:`DEF`
.. |n0| replace:: :math:`n_{0}`
.. |n1| replace:: :math:`n_{1}`
.. |n2| replace:: :math:`n_{2}`
.. |n0, n1| replace:: :math:`n_{0}n_{1}`
.. |n0, n2| replace:: :math:`n_{0}n_{2}`
.. |n1, n2| replace:: :math:`n_{1}n_{2}`
.. |n0, n1, n2| replace:: :math:`n_{0}n_{1}n_{2}`
.. |(AB / DE) x ([] / C)| replace:: :math:`\\frac{AB}{DE} \\times \\frac{\left[\\;\\right]}{C}`
.. |(A / CD) x ([] / E)| replace:: :math:`\\frac{A}{CD} \\times \\frac{\left[\\;\\right]}{E}`
.. |([] / C) x (A / D)| replace:: :math:`\\frac{\\left[\\;\\right]}{C} \\times \\frac{A}{D}`
.. |small_phi = 1/6| replace:: :math:`\\varphi = \\frac{1}{6}`
.. |small_phi = 1/10| replace:: :math:`\\varphi = \\frac{1}{10}`
.. |t_{-1}| replace:: :math:`t_{-1}`
.. |t_0| replace:: :math:`t_0`
.. |1,0,0| replace:: :math:`\{1,0,0\}`
.. |0,1,0| replace:: :math:`\{0,1,0\}`
.. |0,0,1| replace:: :math:`\{0,0,1\}`
.. |N_0 = 0, N_1 = 0, N_2 = 1| replace:: :math:`\{N_0 = 0, N_1 = 0, N_2 = 1\}`
.. |ith| replace:: :math:`i^{\\textrm{th}}`
.. |jth| replace:: :math:`j^{\\textrm{th}}`
.. |r| replace:: :math:`r`
.. |n| replace:: :math:`n`
.. |N| replace:: :math:`N`
.. |n x n| replace:: :math:`N \\times N`
.. |2^n x 2^n| replace:: :math:`2^N \\times 2^N`
.. |i| replace:: :math:`i`
.. |j| replace:: :math:`j`
.. |i,jth| replace:: :math:`i,j^{\\textrm{th}}`
.. |k| replace:: :math:`k`
.. |past| replace:: :const:`DIRECTIONS[PAST]`
.. |future| replace:: :const:`DIRECTIONS[FUTURE]`
.. |PRECISION| replace:: :const:`constants.PRECISION`
.. |EPSILON| replace:: :const:`constants.EPSILON`
.. |CM[i][j] = 1| replace:: :math:`CM_{i,j} = 1`
.. |compute| replace:: :mod:`~pyphi.compute`
.. |compute.concept| replace:: :mod:`~pyphi.compute.big_phi`
.. |compute.big_phi| replace:: :mod:`~pyphi.compute.concept`
.. |compute.distance| replace:: :mod:`~pyphi.compute.distance`
.. |models.big_phi| replace:: :mod:`~pyphi.models.big_phi`
.. |models.concept| replace:: :mod:`~pyphi.models.concept`
.. |models.cuts| replace:: :mod:`~pyphi.models.cuts`
.. |network| replace:: :mod:`~pyphi.network`
.. |subsystem| replace:: :mod:`~pyphi.subsystem`
.. |macro| replace:: :mod:`~pyphi.macro`
.. |convert| replace:: :mod:`~pyphi.convert`
.. |examples| replace:: :mod:`~pyphi.examples`
.. |node| replace:: :mod:`~pyphi.node`
.. |concept_caching| replace:: :mod:`~pyphi.concept_caching`
.. |memory| replace:: :mod:`~pyphi.memory`
.. |db| replace:: :mod:`~pyphi.db`
.. |utils| replace:: :mod:`~pyphi.utils`
.. |validate| replace:: :mod:`~pyphi.validate`
.. |config| replace:: :mod:`~pyphi.config`
.. |Subsystem| replace:: :class:`~pyphi.subsystem.Subsystem`
.. |MacroSubsystem| replace:: :class:`~pyphi.macro.MacroSubsystem`
.. |Network| replace:: :class:`~pyphi.network.Network`
.. |BigMip| replace:: :class:`~pyphi.models.big_phi.BigMip`
.. |Concept| replace:: :class:`~pyphi.models.concept.Concept`
.. |Constellation| replace:: :class:`~pyphi.models.concept.Constellation`
.. |Cut| replace:: :class:`~pyphi.models.cuts.Cut`
.. |Mip| replace:: :class:`~pyphi.models.concept.Mip`
.. |Mice| replace:: :class:`~pyphi.models.concept.Mice`
.. |Node| replace:: :class:`~pyphi.node.Node`
.. |CoarseGrain| replace:: :class:`~pyphi.macro.CoarseGrain`
.. |Blackbox| replace:: :class:`~pyphi.macro.Blackbox`
.. |find_mip| replace:: :meth:`~pyphi.subsystem.Subsystem.find_mip`
.. |find_mice| replace:: :meth:`~pyphi.subsystem.Subsystem.find_mice`
.. |loli_index2state| replace:: :class:`~pyphi.convert.loli_index2state`
.. |holi_index2state| replace:: :class:`~pyphi.convert.holi_index2state`
.. |big_phi > 0| replace:: :math:`\\Phi > 0`
"""

# -- Options for Napoleon (docstring format extension) --------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    # Add any paths that contain custom themes here, relative to this
    # directory.
    html_theme_path = ['_themes'] + [sphinx_rtd_theme.get_html_theme_path()]

# (Optional) Logo. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
html_logo = "phi.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

html_sidebars = {'**': ['localtoc.html']}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = version + " documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'PyPhidoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'PyPhi.tex', 'PyPhi Documentation',
   'Will Mayner', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'pyphi', 'PyPhi Documentation',
     ['Will Mayner'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'PyPhi', 'PyPhi Documentation',
   'Will Mayner', 'PyPhi',
   'A Python module for computing integrated information',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False

# -- Options for autodoc output ------------------------------------------
autoclass_content = 'both'
autodoc_member_order = 'bysource'
