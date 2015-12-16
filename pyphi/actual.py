#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# actual.py

"""
Methods for computing actual causation of subsystems and mechanisms.
"""

import logging
import numpy as np

from . import config, convert, validate
from .subsystem import Subsystem
from .compute import complexes
from .config import PRECISION
from .network import list_future_purview, list_past_purview
from .utils import powerset
from .constants import DIRECTIONS, FUTURE, PAST, EPSILON

from .ac_models import AcMip, AcMice
from .ac_utils import ap_diff_abs_eq

from itertools import chain
from collections import defaultdict
# Create a logger for this module.
log = logging.getLogger(__name__)

# Utils
# ============================================================================
def state_probability(repertoire, fixed_nodes, state):
    """ The dimensions of the repertoire that correspond to the fixed nodes are
        collapsed onto their state. All other dimension should be singular already
        (repertoire size and fixed_nodes need to match), and thus should receive 0
        as the conditioning index. 
        A single probability is returned.
    """
    #Todo: throw error if repertoire size doesn't fit fixed_nodes
    conditioning_indices = [0] * len(state)
    for i in fixed_nodes:
        conditioning_indices[i] = state[i]

    return repertoire[tuple(conditioning_indices)]

def _null_acmip(second_state, direction, mechanism, purview):
    # TODO Use properties here to infer mechanism and purview from
    # partition yet access them with .mechanism and .partition
    return AcMip(second_state=second_state,
               direction=direction,
               mechanism=mechanism,
               purview=purview,
               partition=None,
               unpartitioned_ap=None,
               partitioned_ap=None,
               ap_diff=0.0)

def nice_ac_composition(acmices):
    if acmices:
        if acmices[0].direction == DIRECTIONS[PAST]:
            dir_arrow = '<--'
        elif acmices[0].direction == DIRECTIONS[FUTURE]:
            dir_arrow = '-->'
        else:
            validate.acmices.direction(direction)
        acmices_list = [[acmice.ap_diff, acmice.mechanism, dir_arrow, acmice.purview] for acmice in acmices]
        return acmices_list
    else:
        return None

# ============================================================================
# Single purview
# ============================================================================
def find_ac_mip(subsystem, second_state, direction, mechanism, purview, norm=True, allow_neg=False):
    """ Return the cause coef mip minimum information partition for a mechanism 
        over a cause purview. 
        Returns: 
            ap_diff_min:    the min. difference of the actual probabilities of 
                            the unpartitioned cause and its MIP
        Todo: also return cut etc. ?
    """

    repertoire = Subsystem._get_repertoire(subsystem, direction)

    ap_diff_min = float('inf')
    # Calculate the unpartitioned acp to compare against the partitioned ones
    unpartitioned_repertoire = repertoire(mechanism, purview)
    ap = state_probability(unpartitioned_repertoire, purview, second_state)

    # Loop over possible MIP bipartitions
    for part0, part1 in Subsystem._mip_bipartition(mechanism, purview):
        # Find the distance between the unpartitioned repertoire and
        # the product of the repertoires of the two parts, e.g.
        #   D( p(ABC/ABC) || p(AC/C) * p(B/AB) )
        part1rep = repertoire(part0.mechanism, part0.purview)
        part2rep = repertoire(part1.mechanism, part1.purview)
        partitioned_repertoire = part1rep * part2rep
        partitioned_ap = state_probability(partitioned_repertoire, purview, second_state)

        ap_diff = ap - partitioned_ap
        
        # First check for 0
        # Default: don't count contrary causes and effects
        if ap_diff_abs_eq(ap_diff, 0) or (ap_diff < 0 and not allow_neg):
            return AcMip(second_state=second_state,
                       direction=direction,
                       mechanism=mechanism,
                       purview=purview,
                       partition=(part0, part1),
                       unpartitioned_ap=ap,
                       partitioned_ap=partitioned_ap,
                       ap_diff=0.0)
        
        # Then take closest to 0
        if (abs(ap_diff_min) - abs(ap_diff)) > EPSILON:
            ap_diff_min = ap_diff
            if norm:
                norm_factor = ap
            else:
                norm_factor = 1.0
            acmip = AcMip(second_state=second_state,
                      direction=direction,
                      mechanism=mechanism,
                      purview=purview,
                      partition=(part0, part1),
                      unpartitioned_ap=ap,
                      partitioned_ap=partitioned_ap,
                      ap_diff=round(ap_diff/norm_factor, PRECISION))
    
    return acmip

# ============================================================================
# Average over purviews
# ============================================================================
def find_ac_mice(subsystem, second_state, direction, mechanism, purviews=False, norm=True, allow_neg=False):
    """Return the maximally irreducible cause or effect coefficient for a mechanism.

    Args:
        direction (str): The temporal direction, specifying cause or
            effect.
        mechanism (tuple(int)): The mechanism to be tested for
            irreducibility.

    Keyword Args:
        purviews (tuple(int)): Optionally restrict the possible purviews
            to a subset of the subsystem. This may be useful for _e.g._
            finding only concepts that are "about" a certain subset of
            nodes.
    Returns:
        ac_mice: The maximally-irreducible actual cause or effect.

    .. note::
        Strictly speaking, the AC_MICE is a pair of coefficients: the actual 
        cause and actual effect of a mechanism. Here, we
        return only information corresponding to one direction, |past| or
        |future|, i.e., we return an actual cause or actual effect coefficient, 
        not the pair of them.
    """
    if purviews is False:
        if direction == DIRECTIONS[PAST]:
            purviews = list_past_purview(subsystem.network, mechanism)
        elif direction == DIRECTIONS[FUTURE]:
            purviews = list_future_purview(subsystem.network, mechanism)
        else:
            validate.direction(direction)
        # Filter out purviews that aren't in the subsystem and convert to
        # nodes.
        purviews = [purview for purview in purviews if
                    set(purview).issubset(subsystem.node_indices)]

    # Filter out trivially reducible purviews.
    def not_trivially_reducible(purview):
        if direction == DIRECTIONS[PAST]:
            return subsystem._fully_connected(purview, mechanism)
        elif direction == DIRECTIONS[FUTURE]:
            return subsystem._fully_connected(mechanism, purview)
    purviews = tuple(filter(not_trivially_reducible, purviews))

    # Find the maximal MIP over the remaining purviews.
    if not purviews:
        maximal_acmip = _null_acmip(second_state, direction, mechanism, None)
    else:
        #This max should be most positive
        try:
            maximal_acmip = max(list(filter(None, [find_ac_mip(subsystem, second_state, direction, mechanism, purview, norm, allow_neg) for
                          purview in purviews])))
        except: #Todo: put right error? if max is empty? 
            # If there are only reducible purviews, take largest 
            maximal_acmip = _null_acmip(second_state, direction, mechanism, None)

    # Identify the relevant connections for the AcMICE.
    if not ap_diff_abs_eq(maximal_acmip.ap_diff, 0):
        relevant_connections = \
            subsystem._connections_relevant_for_mice(maximal_acmip)
    else:
        relevant_connections = None
    
    # Construct the corresponding AcMICE.
    acmice = AcMice(maximal_acmip, relevant_connections)
    
    return acmice

# ============================================================================
# Average over mechanisms
# ============================================================================        
def directed_ac_composition(subsystem, second_state, direction, mechanisms=False, purviews=False, norm=True, allow_neg=False):
    """Set of all AcMice of the specified direction"""
    if mechanisms is False:
        mechanisms = powerset(subsystem.node_indices)
    acmices = [find_ac_mice(subsystem, second_state, direction, mechanism, purviews=purviews, norm=norm, allow_neg=allow_neg)
                for mechanism in mechanisms]
    # Filter out falsy acmices, i.e. those with effectively zero ac_diff.
    return tuple(filter(None, acmices))
