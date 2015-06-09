#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# macro.py
"""
Methods for coarse-graining systems to different levels of spatial analysis.
"""

import numpy as np
import itertools
import logging
from . import constants, validate, compute, convert, utils
from .network import Network

# Create a logger for this module.
log = logging.getLogger(__name__)

# Load precomputed partition lists.
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
_NUM_PRECOMPUTED_PARTITION_LISTS = 10
_partition_lists = [
    np.load(os.path.join(_ROOT, 'data', 'partition_lists', str(i) + '.npy'))
    for i in range(_NUM_PRECOMPUTED_PARTITION_LISTS)
]


class MacroNetwork:
    """A coarse-grained network of nodes.

    See the 'macro' example in the documentation for more information.

    Attributes:
        network (Network): The network object of the macro-system.
        phi (float): The |big_phi| of the network's main complex.
        micro_network (Network): The network object of the corresponding micro
            system.
        micro_phi (float): The |big_phi| of the main complex of the
            corresponding micro-system.
        partition (list): The partition which defines macro-elements in terms
            of micro-elements.
        grouping (list(list)): The correspondence between micro-states and
            macro-states.
        emergence (float): The difference between the |big_phi| of the macro-
            and the micro-system.
    """
    def __init__(self, macro_network, macro_phi, micro_network, micro_phi,
                 partition, grouping):
        self.network = macro_network
        self.phi = macro_phi
        self.micro_network = micro_network
        self.micro_phi = micro_phi
        self.partition = partition
        self.grouping = grouping
        self.emergence = self.phi - self.micro_phi


def _partitions_list(N):
    """Return a list of partitions of the |N| binary nodes.

    Args:
        N (int): The number of nodes under consideration.

    Returns:
        partition_list (``list``): A list of lists, where each inner list is
        the set of micro-elements corresponding to a macro-element.

    Example:
        >>> from pyphi.macro import _partitions_list
        >>> _partitions_list(3)
        [[[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]], [[0], [1], [2]]]
    """
    if N < (_NUM_PRECOMPUTED_PARTITION_LISTS):
        return list(_partition_lists[N])
    else:
        raise ValueError('Partition lists not yet available for system with %d \
                         nodes or more' % (_NUM_PRECOMPUTED_PARTITION_LISTS))


def list_all_partitions(network):
    """Return a list of all possible coarse grains of a network.

    Args:
        network (Network): The physical system to act as the 'micro' level.

    Returns:
        partitions (``list(list))``): A list of possible partitions. Each
            element of the list is a list of micro-elements which correspong to
            macro-elements.
    """
    partitions = _partitions_list(network.size)
    if network.size > 0:
        partitions[-1] = [list(range(network.size))]
    return tuple(tuple(tuple(part)
                       for part in partition)
                 for partition in partitions)


def list_all_groupings(partition):
    """Return a list of all possible groupings of states, for a particular
    coarse graining (partition) of a network.

    Args:
        network (Network): The physical system on the micro level.
        partitions (list(list)): The partition of micro-elements into macro
            elements.

    Returns:
        groupings (``list(list(list(list)))``): A list of all possible
            correspondences between micro-states and macro-states for the
            partition.
    """
    if not all([len(part) > 0 for part in partition]):
        raise ValueError('Each part of the partition must have at least one '
                         'element.')
    micro_state_groupings = [_partitions_list(len(part) + 1) if len(part) > 1
                             else [[[0], [1]]] for part in partition]
    groupings = [list(grouping) for grouping in
                 itertools.product(*micro_state_groupings) if
                 np.all(np.array([len(element) < 3 for element in grouping]))]
    return tuple(tuple(tuple(tuple(state) for state in states) for states in group) for group in groupings)



def make_macro_network(network, mapping):
    """Create the macro-network for a given mapping from micro to macro-states.

    Returns ``None`` if the macro TPM does not satisfy the conditional
    independence assumption.

    Args:
        micro_tpm (nd.array): TPM of the micro-system.
        mapping (nd.array): Mapping from micro-states to macro-states.

    Returns:
        macro_network (``Network``): Network of the macro-system, or ``None``.
    """
    num_macro_nodes = int(np.log2(max(mapping) + 1))
    macro_tpm = utils.make_macro_tpm(network.tpm, mapping)
    macro_current_state = convert.loli_index2state(
        mapping[convert.state2loli_index(network.current_state)].astype(int),
        num_macro_nodes)
    if validate.conditionally_independent(macro_tpm):
        return Network(macro_tpm, macro_current_state)
    else:
        return None


def emergence(network):
    """Check for emergence of a macro-system into a macro-system.

    Checks all possible partitions and groupings of the micro-system to find
    the spatial scale with maximum integrated information.

    Args:
        network (Network): The network of the micro-system under investigation.

    Returns:
        macro_network (``MacroNetwork``): The maximal coarse-graining of the
            micro-system.
    """
    micro_phi = compute.main_complex(network).phi
    partitions = list_all_partitions(network)
    max_phi = float('-inf')
    for partition in partitions:
        groupings = list_all_groupings(partition)
        for grouping in groupings:
            mapping = utils.make_mapping(partition, grouping)
            macro_network = make_macro_network(network, mapping)
            if macro_network:
                main_complex = compute.main_complex(macro_network)
                if (main_complex.phi - max_phi) > constants.EPSILON:
                    max_phi = main_complex.phi
                    max_partition = partition
                    max_grouping = grouping
                    max_network = macro_network
    return MacroNetwork(macro_network=max_network,
                        macro_phi=max_phi,
                        micro_network=network,
                        micro_phi=micro_phi,
                        partition=max_partition,
                        grouping=max_grouping)
