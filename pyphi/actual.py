#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# actual.py

"""
Methods for computing actual causation of subsystems and mechanisms.

Bidirectional analysis of a transition:
    Note: during the init the subsystem should always come with the past state. 
    This is because the subsystem should take the past state as background states in both directions. 
    Then in the "past" case, the subsystem state should be swapped, to condition on the current state.
    All functions that take subsystems assume that the subsystem has already been prepared in this way.
    "past": evaluate cause-repertoires given current state
            background: past, condition on: current, actual state: past
    "future": evaluate effect-repertoires given past state
            background: past, condition on: past, actual state: current

To do this with the minimal effort, the subsystem state and actual state have to be swapped in the "past" case,
after the subsystem is conditioned on the background condition.

# Todo: check that transition between past and current state is possible for every function
"""

import logging
import numpy as np

from . import config, convert, validate
from .subsystem import Subsystem
from .compute import complexes
from .network import list_future_purview, list_past_purview, Network
from .utils import powerset, directed_bipartition, cut_mechanism_indices
from .constants import DIRECTIONS, FUTURE, PAST, EPSILON
from .models import Cut, _bigmip_attributes

from .ac_models import AcMip, AcMice, AcBigMip, _acbigmip_attributes
from .ac_utils import ap_phi_abs_eq, directed_bipartitions_existing_connections

from itertools import chain
from collections import defaultdict
from pprint import pprint
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Create a logger for this module.
log = logging.getLogger(__name__)

# Utils
# ============================================================================
def make_ac_subsystem(network, past_state, current_state, direction=False, subset=False):
    """ To have the right background, the init state for the subsystem should always be the past_state.
        In past direction after subsystem is created the actual state and the system state need to be swapped. """
    
    if not subset:
        subset = range(network.size)
    
    subsystem = Subsystem(network, past_state, subset)

    if direction == DIRECTIONS[PAST]:
        actual_state = past_state
        subsystem.state = current_state
    else: 
        actual_state = current_state

    return subsystem, actual_state

def state_probability(repertoire, fixed_nodes, cond_state):
    """ The dimensions of the repertoire that correspond to the fixed nodes are
        collapsed onto their state. All other dimension should be singular already
        (repertoire size and fixed_nodes need to match), and thus should receive 0
        as the conditioning index. 
        A single probability is returned.
    """
    #Todo: throw error if repertoire size doesn't fit fixed_nodes
    conditioning_indices = [0] * len(cond_state)
    for i in fixed_nodes:
        conditioning_indices[i] = cond_state[i]

    return repertoire[tuple(conditioning_indices)]

def nice_ac_composition(acmices):
    if acmices:
        if acmices[0].direction == DIRECTIONS[PAST]:
            dir_arrow = '<--'
        elif acmices[0].direction == DIRECTIONS[FUTURE]:
            dir_arrow = '-->'
        else:
            validate.acmices.direction(direction)
        acmices_list = [["{0:.4f}".format(round(acmice.ap_phi,4)), acmice.mechanism, dir_arrow, acmice.purview] for acmice in acmices]
        return acmices_list
    else:
        return None

def multiple_states_nice_ac_composition(net, transitions, node_indices, mechanisms=False, purviews=False, norm=True, allow_neg=False):
    """nice composition for multiple pairs of states
    Args: as above
        transitions (list(2 state tuples)):    The first is past the second current.
                                                For 'past' current belongs to subsystem and past is the second state.
                                                Vice versa for "future"
    """
    for transition in transitions:
        c_subs = Subsystem(net, transition[1], node_indices)
        acp_mice = directed_ac_constellation(c_subs, transition[0], 'past', mechanisms, purviews, norm, allow_neg)

        p_subs = Subsystem(net, transition[0], node_indices)
        aep_mice = directed_ac_constellation(p_subs, transition[1], 'future', mechanisms, purviews, norm, allow_neg)
        print('#####################################')
        print(transition)
        print('- cause coefs ----------------------')
        pprint(nice_ac_composition(acp_mice))
        print('- effect coefs ----------------------')
        pprint(nice_ac_composition(aep_mice))
        print('---------------------------')    

# ============================================================================
# Single purview
# ============================================================================
def _null_acmip(actual_state, direction, mechanism, purview):
    # TODO Use properties here to infer mechanism and purview from
    # partition yet access them with .mechanism and .partition
    return AcMip(actual_state=actual_state,
               direction=direction,
               mechanism=mechanism,
               purview=purview,
               partition=None,
               unpartitioned_ap=None,
               partitioned_ap=None,
               ap_phi=0.0)

def find_ac_mip(subsystem, actual_state, direction, mechanism, purview, norm=True, allow_neg=False):
    """ Return the cause coef mip minimum information partition for a mechanism 
        over a cause purview. 
        Returns: 
            ap_phi_min:    the min. difference of the actual probabilities of 
                            the unpartitioned cause and its MIP
        Todo: also return cut etc. ?
    """

    repertoire = Subsystem._get_repertoire(subsystem, direction)
    
    ap_phi_min = float('inf')
    # Calculate the unpartitioned acp to compare against the partitioned ones
    unpartitioned_repertoire = repertoire(mechanism, purview)
    ap = state_probability(unpartitioned_repertoire, purview, actual_state)

    # Loop over possible MIP bipartitions
    for part0, part1 in Subsystem._mip_bipartition(mechanism, purview):
        # Find the distance between the unpartitioned repertoire and
        # the product of the repertoires of the two parts, e.g.
        #   D( p(ABC/ABC) || p(AC/C) * p(B/AB) )
        part1rep = repertoire(part0.mechanism, part0.purview)
        part2rep = repertoire(part1.mechanism, part1.purview)
        partitioned_repertoire = part1rep * part2rep
        partitioned_ap = state_probability(partitioned_repertoire, purview, actual_state)

        ap_phi = ap - partitioned_ap
        
        # First check for 0
        # Default: don't count contrary causes and effects
        if ap_phi_abs_eq(ap_phi, 0) or (ap_phi < 0 and not allow_neg):
            return AcMip(actual_state=actual_state,
                       direction=direction,
                       mechanism=mechanism,
                       purview=purview,
                       partition=(part0, part1),
                       unpartitioned_ap=ap,
                       partitioned_ap=partitioned_ap,
                       ap_phi=0.0)
        
        # Then take closest to 0
        if (abs(ap_phi_min) - abs(ap_phi)) > EPSILON:
            ap_phi_min = ap_phi
            if norm:
                uc_cr = repertoire((), purview)
                norm_factor = state_probability(uc_cr, purview, actual_state)
                #print(norm_factor)
            else:
                norm_factor = 1.0
            acmip = AcMip(actual_state=actual_state,
                      direction=direction,
                      mechanism=mechanism,
                      purview=purview,
                      partition=(part0, part1),
                      unpartitioned_ap=ap,
                      partitioned_ap=partitioned_ap,
                      ap_phi=ap_phi/norm_factor)
    
    return acmip

# ============================================================================
# Average over purviews
# ============================================================================
def find_ac_mice(subsystem, actual_state, direction, mechanism, purviews=False, norm=True, allow_neg=False):
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
        maximal_acmip = _null_acmip(actual_state, direction, mechanism, None)
    else:
        #This max should be most positive
        try:
            MIP_list = [find_ac_mip(subsystem, actual_state, direction, mechanism, purview, norm, allow_neg) for
                          purview in purviews]
            #print([mip.ap_phi for mip in MIP_list])
            maximal_acmip = max(list(filter(None, MIP_list)))
        except: #Todo: put right error? if max is empty? 
            # If there are only reducible purviews, take largest 
            maximal_acmip = _null_acmip(actual_state, direction, mechanism, None)

    # Identify the relevant connections for the AcMICE.
    if not ap_phi_abs_eq(maximal_acmip.ap_phi, 0):
        relevant_connections = \
            subsystem._connections_relevant_for_mice(maximal_acmip)
    else:
        relevant_connections = None
    
    # Construct the corresponding AcMICE.
    acmice = AcMice(maximal_acmip, relevant_connections)
    
    return acmice

# ============================================================================
# Average over mechanisms - constellations
# ============================================================================        
def directed_ac_constellation(subsystem, actual_state, direction, mechanisms=False, purviews=False, norm=True, allow_neg=False):
    """Set of all AcMice of the specified direction, similar to "sequential_constellation" in compute.py"""
    if mechanisms is False:
        mechanisms = powerset(subsystem.node_indices)
    acmices = [find_ac_mice(subsystem, actual_state, direction, mechanism, purviews=purviews, norm=norm, allow_neg=allow_neg)
                for mechanism in mechanisms]
    # Filter out falsy acmices, i.e. those with effectively zero ac_diff.
    return tuple(filter(None, acmices))

# ============================================================================
# AcBigMips and System cuts
# ============================================================================        
def ac_constellation_distance(C1, C2):
    """Return the distance between two constellations. Here that is just the difference in sum(ap_phis)

    Args:
        C1 (tuple(Concept)): The first constellation.
        C2 (tuple(Concept)): The second constellation.

    Returns:
        distance (``float``): The distance between the two constellations in
            concept-space.
    """
    return sum([acmice.ap_phi for acmice in C1]) - sum([acmice.ap_phi for acmice in C2])


def _null_acbigmip(subsystem, actual_state, direction):
    """Returns an ac |BigMip| with zero |big_ap_phi| and empty constellations."""

    return AcBigMip(subsystem=subsystem, actual_state=actual_state, direction=direction, cut_subsystem=subsystem, ap_phi=0.0,
                  unpartitioned_constellation=(), partitioned_constellation=())

#TODO: single node BigMip

def _evaluate_cut(uncut_subsystem, cut, unpartitioned_constellation, actual_state, direction):
    """Find the |AcBigMip| for a given cut. This is unidirectional for a transition."""
    log.debug("Evaluating cut {}...".format(cut))

    cut_subsystem = Subsystem(uncut_subsystem.network,
                              uncut_subsystem.state,
                              uncut_subsystem.node_indices,
                              cut=cut,
                              mice_cache=uncut_subsystem._mice_cache)
    if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms = set([c.mechanism for c in unpartitioned_constellation])
    else:
        mechanisms = set([c.mechanism for c in unpartitioned_constellation] +
                         list(cut_mechanism_indices(uncut_subsystem, cut)))


    partitioned_constellation = directed_ac_constellation(cut_subsystem, actual_state, direction, mechanisms)
    log.debug("Finished evaluating cut {}.".format(cut))

    ap_phi = ac_constellation_distance(unpartitioned_constellation,
                                 partitioned_constellation)
    return AcBigMip(
        ap_phi=ap_phi,
        actual_state=actual_state,
        direction=direction, 
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=partitioned_constellation,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem) 

def find_big_ac_mip(subsystem, cuts, unpartitioned_constellation, actual_state, direction,
                         min_ac_mip):
    """Find the minimal cut for a subsystem by sequential loop over all cuts,
        holding only two ``BigMip``s in memory at once."""
    for i, cut in enumerate(cuts):
        new_ac_mip = _evaluate_cut(subsystem, cut, unpartitioned_constellation, actual_state, direction)
        log.debug("Finished {} of {} cuts.".format(
            i + 1, len(cuts)))
        if new_ac_mip < min_ac_mip:
            min_ac_mip = new_ac_mip
        # Short-circuit as soon as we find a MIP with effectively 0 phi.
        if not min_ac_mip:
            break
    return min_ac_mip    

def big_ac_mip(subsystem, actual_state, direction=False, weak_connectivity=True):
    """Return the minimal information partition of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.
        weak_connectivity:  To be able to evaluate transitions in open systems, 
                            weakly connected sets of causes and effects.
    Returns:
        big_mip (|BigMip|): A nested structure containing all the data from the
            intermediate calculations. The top level contains the basic MIP
            information for the given subsystem.
    TODO: no time measured yet.        
    """
    # if direction == DIRECTIONS[PAST]:
    #         purviews = list_past_purview(subsystem.network, mechanism)
    #     elif direction == DIRECTIONS[FUTURE]:
    #         purviews = list_future_purview(subsystem.network, mechanism)
    #     else:
    #         validate.direction(direction)

    if not direction:
        direction = 'future'

    log.info("Calculating big-ac-phi data for {}...".format(subsystem))

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty; or
    #   - ToDo: an elementary mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null MIP.
    if not subsystem:
        log.info('Subsystem {} is empty; returning null MIP '
                 'immediately.'.format(subsystem))

    # Get the connectivity of just the subsystem nodes.
    submatrix_indices = np.ix_(subsystem.node_indices, subsystem.node_indices)
    cm = subsystem.network.connectivity_matrix[submatrix_indices]
    # Get the number of strongly connected components.
    if weak_connectivity:
        num_components, _ = connected_components(csr_matrix(cm),
                                             connection='weak')
    else:
        num_components, _ = connected_components(csr_matrix(cm),
                                             connection='strong')

    if num_components > 1:
        log.info('{} is not strongly/weakly connected; returning null MIP '
                 'immediately.'.format(subsystem))

    # =========================================================================
    # Todo: add approximation as an option for weak_connectivity
    if config.CUT_ONE_APPROXIMATION:
        bipartitions = directed_bipartition_of_one(subsystem.node_indices)
    elif weak_connectivity:
        # use full connectivity matrix so that it fits with the node_indices
        bipartitions = directed_bipartitions_existing_connections(subsystem.node_indices, subsystem.network.connectivity_matrix)
    else:
        # The first and last bipartitions are the null cut (trivial
        # bipartition), so skip them.
        bipartitions = directed_bipartition(subsystem.node_indices)[1:-1]

   
    cuts = [Cut(bipartition[0], bipartition[1])
            for bipartition in bipartitions]
    
    #import pdb; pdb.set_trace()
    log.debug("Finding unpartitioned directed_ac_constellation...")   
    unpartitioned_constellation = directed_ac_constellation(subsystem, actual_state, direction)

    log.debug("Found unpartitioned directed_ac_constellation.")
    if not unpartitioned_constellation:
        # Short-circuit if there are no concepts in the unpartitioned
        # directed_ac_constellation.
        result = _null_acbigmip(subsystem, actual_state, direction)
    else:
        min_ac_mip = _null_acbigmip(subsystem, actual_state, direction)
        min_ac_mip.ap_phi = float('inf')
        min_ac_mip = find_big_ac_mip(subsystem, cuts, unpartitioned_constellation, actual_state, direction,
                            min_ac_mip)
        result = min_ac_mip

    log.info("Finished calculating big-ac-phi data for {}.".format(subsystem))
    log.debug("RESULT: \n" + str(result))
    return result

# ============================================================================
# Complexes
# ============================================================================        
# TODO: Fix this to test whether the transition is possible
def subsystems(network, past_state, current_state, direction):
    """Return a generator of all **possible** subsystems of a network.

    Does not return subsystems that are in an impossible transitions."""
    subsystem_list = [];
    for subset in powerset(network.node_indices):
        subsystem_list.append(make_ac_subsystem(network, past_state, current_state, direction, subset))
        # try: 
        #     yield Subsystem(network, state, subset)    
#        except validate.TransitionImpossibleError:
#            pass
    
    # First is empty set
    return subsystem_list[1:]

def unidirectional_ac_complexes(network, past_state, current_state, direction, weak_connectivity):
    """Return a generator for all irreducible ac_complexes of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    return tuple(filter(None, (big_ac_mip(subsystem, actual_state, direction, weak_connectivity) for subsystem, actual_state in
                               subsystems(network, past_state, current_state, direction))))

def unidirectional_main_ac_complex(network, past_state, current_state, direction=False, weak_connectivity=True):
    """Return the main complex of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    log.info("Calculating main ac_complex...")
    result = unidirectional_ac_complexes(network, past_state, current_state, direction, weak_connectivity)
    if result:
        result = max(result)
    else:
        empty_subsystem = Subsystem(network, past_state, ())
        
        if direction == DIRECTIONS[PAST]:
            actual_state = past_state
            empty_subsystem.state = current_state
        else: 
            actual_state = current_state
        result = _null_acbigmip(empty_subsystem, actual_state, direction)

    log.info("Finished calculating main ac_complex.")
    log.debug("RESULT: \n" + str(result))
    return result