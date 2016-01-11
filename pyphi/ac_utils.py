#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ac_utils.py

"""
Functions used by more than one actual causation module or class, or that might be of
external use.
"""
import numpy as np
from .constants import EPSILON
from .utils import directed_bipartition
# Utils
# ============================================================================
def ap_phi_abs_eq(x, y):
    """Compare the abs of two ap_phi values up to |PRECISION|."""
    # This is used to find the mip.
    return abs(abs(x) - abs(y)) <= EPSILON   

def ap_phi_eq(x, y):
    """Compare two ap_phi values up to |PRECISION|. Different sign means they are different. """
    # This is used to find the mice. We want to chose the most positive one.
    return abs(x - y) <= EPSILON   
        
def directed_bipartitions_existing_connections(a, connectivity):
    """Get all bipartitions that cut actual connections in order to evaluate 
        set of causes and effect consisting of different elements.
    
    Returns:
        ACP: actual connection partitions    
    """
    
    """       [((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)), ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,))]
    """
    ACP = [];
    for partition in directed_bipartition(a)[1:-1]:
        rows = connectivity[np.array(partition[0])]
        vals = [i[np.array(partition[1])] for i in rows]
        if np.sum(vals) > 0:
            ACP.append(partition)

    return ACP