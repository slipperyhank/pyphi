#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ac_utils.py

"""
Functions used by more than one actual causation module or class, or that might be of
external use.
"""
import numpy as np
from .constants import EPSILON

# Utils
# ============================================================================
def ap_diff_abs_eq(x, y):
    """Compare the abs of two ap_diff values up to |PRECISION|."""
    # This is used to find the mip.
    return abs(abs(x) - abs(y)) <= EPSILON   

def ap_diff_eq(x, y):
    """Compare two ap_diff values up to |PRECISION|. Different sign means they are different. """
    # This is used to find the mice. We want to chose the most positive one.
    return abs(x - y) <= EPSILON   
        