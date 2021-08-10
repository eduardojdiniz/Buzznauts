#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "Buzznauts: NMA-DL 2021 elated-buzzards project"
# Long description will go up on the pypi page
long_description = """

Buzznauts
=========
Buzznauts is the repository for the NMA-DL 2021 elated-buzzards pod project

It contains software implementations of the Algonauts Challenge 2021

License
=======
``Buzznauts`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2021--, Amir Asiaee, Davide Momi, Eda Bozdemir, Eduardo Diniz,
Nicolás Nieto, and Robert Woodry.
"""

NAME = "Buzznauts"
MAINTAINER = "Eduardo Diniz"
MAINTAINER_EMAIL = "edd32m@pitt.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/eduardojdiniz/Buzznauts"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Amir Asiaee, Davide Momi, Eda Bozdemir, Eduardo Diniz, Nicolás Nieto,\
    and Robert Woodry"
AUTHOR_EMAIL = "amir.asiaeetaheri@gmail.com, momi.davide89@gmail.com, \
    edabozdemir@gmail.com, edd32@pitt.edu, nnieto@sinc.unl.edu.ar, \
    rwoodry@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'Buzznauts': [pjoin('data', '*')]}
REQUIRES = ["duecredit", "numpy"]
PYTHON_REQUIRES = ">= 3.7"
