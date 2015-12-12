#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# actual.py

"""
Methods for computing actual causation of subsystems and mechanisms.
"""

import logging
import numpy as np

from .subsystem import Subsystem
from .compute import complexes
from . import convert
from .network import list_future_purview, list_past_purview
from .utils import powerset
from .config import PRECISION
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



# ============================================================================
# ONE SIDED FUNCTIONS (EITHER CAUSE OR EFFECT)
# ============================================================================  
# Single set functions
# ============================================================================
def cause_probability(subsystem, mechanism, purview, past_state, norm=True):
    """Return the probability value of the actual past state from a cause 
       repertoire
    """
    cr = Subsystem.cause_repertoire(subsystem, mechanism, purview)
    ac_probability = float(state_probability(cr, purview, past_state))
    if norm:
        uc_cr = Subsystem.unconstrained_cause_repertoire(subsystem, purview)
        ac_norm = float(state_probability(uc_cr, purview, past_state))
        #ac_probability = ac_probability/ac_norm
        ac_probability = (ac_probability - ac_norm)/ac_probability #Todo: this is inf if state is not possible!
        #ac_probability = (ac_probability - ac_norm)/ac_norm #Todo: this is inf if state is not possible!

    return ac_probability

def effect_probability(subsystem, mechanism, purview, future_state, norm=True):
    """Return the probability value of the actual future state from an effect
       repertoire
    """
    er = Subsystem.effect_repertoire(subsystem, mechanism, purview)
    ac_probability = float(state_probability(er, purview, future_state))
    if norm:
        uc_er = Subsystem.unconstrained_effect_repertoire(subsystem, purview)
        ac_norm = float(state_probability(uc_er, purview, future_state))
        #ac_probability = ac_probability/ac_norm
        ac_probability = (ac_probability - ac_norm)/ac_probability
        #ac_probability = (ac_probability - ac_norm)/ac_norm #Todo: this is inf if state is not possible!

    
    return ac_probability

# All purview functions (given a mechanism all possible purviews)
# ============================================================================
def cause_coefs_all_purviews(subsystem, mechanism, past_state):
    """Return list of probabilities of the actual past state from all 
       irreducible cause repertoires of the mechanism
    """        
    purviews = list_past_purview(subsystem.network, mechanism)

    # List of all irreducible purviews
    all_cause_mips = [Subsystem.mip_past(subsystem, mechanism, purview) for purview in purviews]
    irreducible_cause_mips = [cause_mip for cause_mip in all_cause_mips if cause_mip.phi > 0]

    coefficients = [cause_probability(subsystem, mechanism, icm.purview, past_state) for icm in irreducible_cause_mips]
    #output = [[coefficients[i], irreducible_cause_mips[i].purview, irreducible_cause_mips[i].phi] for i in range(len(coefficients))]

    if not coefficients:
        return None
    else:
        output = {
            'effect': mechanism,
            'cause': [irreducible_cause_mips[i].purview for i in range(len(irreducible_cause_mips))],
            'phi_cause': [irreducible_cause_mips[i].phi for i in range(len(irreducible_cause_mips))],
            'coef_cause': [coefficients[i] for i in range(len(coefficients))]
        }
        return output
 
def effect_coefs_all_purviews(subsystem, mechanism, future_state):
    """Return list of probabilities of the actual future state from all 
       irreducible effect repertoires of the mechanism
    """        
    purviews = list_future_purview(subsystem.network, mechanism)

    # List of all irreducible purviews
    all_effect_mips = [Subsystem.mip_future(subsystem, mechanism, purview) for purview in purviews]
    irreducible_effect_mips = [effect_mip for effect_mip in all_effect_mips if effect_mip.phi > 0]

    coefficients = [effect_probability(subsystem, mechanism, iem.purview, future_state) for iem in irreducible_effect_mips]
    #output = [[coefficients[i], irreducible_effect_mips[i].purview, irreducible_effect_mips[i].phi] for i in range(len(coefficients))]
    
    if not coefficients:
        return None
    else:
        output = {
            'cause': mechanism,
            'effect': [irreducible_effect_mips[i].purview for i in range(len(irreducible_effect_mips))],
            'phi_effect': [irreducible_effect_mips[i].phi for i in range(len(irreducible_effect_mips))],
            'coef_effect': [coefficients[i] for i in range(len(coefficients))]
        }
        return output


# ============================================================================
# TWO WAY FUNCTIONS (BOTH CAUSE AND EFFECT)
# ============================================================================  
# Single set functions
# ============================================================================
def cause_effect_probability(subsystem_cause, subsystem_effect, cause_set, effect_set):
    """Return the probability values of the actual past/current state from the cause/effect
       repertoires over the cause/effect set of elements
       args: subsystem_cause: subsystem with past state
             subsystem_effect: subsystem with current state
             cause_set:  set of elements that is mechanism on the cause side (past)
                         and purview on the effect side (current)
             effect_set: set of elements that is purview on the cause side (past)
                         and mechanism on the effect side (current)
    """
    cause_ac_probability = cause_probability(subsystem_effect, effect_set, cause_set, subsystem_cause.state)
    effect_ac_probability = effect_probability(subsystem_cause, cause_set, effect_set, subsystem_effect.state)

    return (cause_ac_probability, effect_ac_probability)


# All sets functions
# ============================================================================
def effect_list_of_dicts(subsystem_cause, subsystem_effect, cause_set=False, effect_set=False):
    """Return the probability values of the actual past/current state from the cause/effect
       repertoires of all subsets of the cause/effect set of elements
       args: subsystem_cause: subsystem with past state
             subsystem_effect: subsystem with current state
             cause_set:  set of elements whose powerset is the mechanism on the cause side (past)
                         and purview on the effect side (current)
             effect_set: set of elements whose powerset is the purview on the cause side (past)
                         and mechanism on the effect side (current)
    """
    if cause_set is False:
        cause_set = subsystem_cause.node_indices
    if effect_set is False:
        effect_set = subsystem_effect.node_indices    

    # powerset of mechanisms (made tuple because I don't understand the iterable)
    powerset_effects = tuple(filter(None, (powerset(effect_set))))
    # list of effects, first get all their cause_coefficients
    effects_list = list(filter(None, [cause_coefs_all_purviews(subsystem_effect, p_set, subsystem_cause.state) 
                                for p_set in powerset_effects]))

    # then add coef_effect and phi_effect of each effect for the set of all its potential causes
    for e in effects_list:
        e['coef_effect'] = [effect_probability(subsystem_cause, c, e['effect'], subsystem_effect.state) for c in e['cause']]
        e['phi_effect'] = [Subsystem.phi_mip_future(subsystem_cause, c, e['effect']) for c in e['cause']]
        # make effect list as long as cause
        e['effect'] = [e['effect'] for c in e['cause']]

    return effects_list

def cause_effect_list(subsystem_cause, subsystem_effect, cause_set=False, effect_set=False):
    """Return a big list of all cause-effect pairs and their coefficients and phi
    ordered: 'cause' 'effect' 'coef_cause' 'coef_effect' 'phi_cause' 'phi_effect'
    """
    effects_dict_list = cause_effect_coefs(subsystem_cause, subsystem_effect, cause_set, effect_set)
    # expand effect to list with the same number of elements as the cause list
    big_list = []
    for e in effects_dict_list:
        # make row of values for every cause_set (basically swap row column dimensions) 
        big_list.append(list([e['cause'][i], e['effect'][i], e['coef_cause'][i], 
                            e['coef_effect'][i], e['phi_cause'][i], e['phi_effect'][i]] 
                            for i in range(len(e['cause']))))

        #[(e[key][i]) for key in e.keys()])    
    big_list = list(chain.from_iterable(big_list))   

    return big_list

def cause_effect_dict(subsystem_cause, subsystem_effect, cause_set=False, effect_set=False):
    """Return a big dictionary of all cause-effect pairs and their coefficients and phi
    """
    effects_dict_list = effect_list_of_dicts(subsystem_cause, subsystem_effect, cause_set, effect_set)
    # expand effect to list with the same number of elements as the cause list
    # for each key append values
    one_dict = defaultdict(list)
    for e in effects_dict_list:
       for key,val in e.items():
          one_dict[key].append(val)

    for key in one_dict.keys():
        one_dict[key] = list(chain.from_iterable(one_dict[key])) 

    return dict(one_dict)      
    #pprint(dict(one_dict))

# Get max values
# ============================================================================
def max_cause_coef(subsystem_cause, subsystem_effect, cause_set=False, effect_set=False):
    """Return cause, effect combination with max cause_coef
    """
    # Todo: option to output the most negative one if there are only negative values?
    one_dict = cause_effect_dict(subsystem_cause, subsystem_effect, cause_set, effect_set)
    if one_dict:
        max_cause_coef = max(one_dict['coef_cause'])
        # make dict with list of all max
        # Todo: instead of == put < EPSILON
        indices = [i for i in range(len(one_dict['coef_cause'])) if one_dict['coef_cause'][i] == max_cause_coef]
        max_dict = {}
        for k in one_dict.keys():
            max_dict[k] = [one_dict[k][i] for i in indices]
        
        return max_dict
    else:
        return None    

def max_effect_coef(subsystem_cause, subsystem_effect, cause_set=False, effect_set=False):
    """Return cause, effect combination with max effect_coef
    """
    # Todo: option to output the most negative one if there are only negative values?
    one_dict = cause_effect_dict(subsystem_cause, subsystem_effect, cause_set, effect_set)
    if one_dict:
        max_effect_coef = max(one_dict['coef_effect'])
        # make dict with list of all max
        # Todo: instead of == put < EPSILON
        indices = [i for i in range(len(one_dict['coef_effect'])) if one_dict['coef_effect'][i] == max_effect_coef]
        max_dict = {}
        for k in one_dict.keys():
            max_dict[k] = [one_dict[k][i] for i in indices]
        
        return max_dict
    else:
        return None

def paired_cause_effect_coef(cause_effect_dict):
    """Reorder cause_effect dict with keys: 'cause' 'effect' 'coef_cause' 'coef_effect' 'phi_cause' 'phi_effect'
       into cause-effect pairs
    """
    if cause_effect_dict:
        cause_effect_list = []
        for i in range(len(cause_effect_dict['cause'])):
            cause_effect_list.append([(cause_effect_dict['cause'][i], cause_effect_dict['effect'][i]), 
                                      (cause_effect_dict['coef_cause'][i], cause_effect_dict['coef_effect'][i]),
                                      (cause_effect_dict['phi_cause'][i], cause_effect_dict['phi_effect'][i]) ])
        return cause_effect_list
    else:
        return None

    