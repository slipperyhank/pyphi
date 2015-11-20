#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# extrinsic.py

"""
Methods for computing extrinsic existence of subsystems and mechanisms.
"""

import logging
import numpy as np
from .compute import complexes

# Create a logger for this module.
log = logging.getLogger(__name__)


def existence(network, state):
    """Return a list of all subsystems of a network which exist
       extrinsically
    """
    L = [(c.subsystem, c.phi) for c in complexes(network, state)]
    n = len(L)
    subsystems = [t[0] for t in L]
    phis = [t[1] for t in L]
    ind = np.zeros(n)
    D = hamming_matrix(subsystems)
    for i in range(n):
        if (np.min(phis[np.where(D[i] == 1)[0]] - phis[i]) > 0):
            ind[i] = 1
    return [subsystems[i] for i in range(n) if ind[i] == 1]


def hamming_matrix(s):
    """Return a matrix of hamming distances between a list of subsystems
    Args:
        s (list[subsystem]): A list of subsystems to create the distance matrix
    """
    n = len(s)
    ind = [set(s[i].node_indices) for i in range(n)]
    D = np.array([[len(ind[i].union(ind[j]).difference(ind[i].intersection(ind[j])))
                   for i in range(n)] for j in range(n)])
    return D
