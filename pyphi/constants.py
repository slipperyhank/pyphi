#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Package-wide constants.
"""

from enum import Enum
import pickle

import joblib

from . import config


class Direction(Enum):
    """Constants that parametrize cause and effect methods.

    Accessed using ``Direction.PAST`` and ``Direction.FUTURE``.
    """
    PAST = 0
    FUTURE = 1
    BIDIRECTIONAL = 2


#: The threshold below which we consider differences in phi values to be zero.
EPSILON = 10 ** - config.PRECISION

#: Label for the filesystem cache backend.
FILESYSTEM = 'fs'

#: Label for the MongoDB cache backed.
DATABASE = 'db'

#: The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: The joblib Memory object for persistent caching without a database.
joblib_memory = joblib.Memory(cachedir=config.FS_CACHE_DIRECTORY,
                              verbose=config.FS_CACHE_VERBOSITY)

#: Earth Movers Distance
EMD = 'EMD'

#: Kullback-Leibler Divergence
KLD = 'KLD'

#: L1 distance
L1 = 'L1'

# Difference in Entropy distance
ENT = 'ENT'

#: All available measures
MEASURES = [EMD, KLD, L1, ENT]

# Bipartitions
BIPARTITION = 'BIPARTITION'

# "Wedge" Tripartitions
WEDGE = 'WEDGE'

# Approximation: Full mechanism cut
FULL = 'FULL'

# All available sets of partitions
PARTITIONS = [BIPARTITION, WEDGE, FULL]

# Break ties to largest purview
LARGEST = 'LARGEST'

# Break ties to smallest purview
SMALLEST = 'SMALLEST'

# All available tie breaking options
TIES = [LARGEST, SMALLEST]
