#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# validate.py

"""
Methods for validating common types of input.
"""

import numpy as np

from . import constants, convert, config
from .constants import EPSILON


class StateUnreachableError(ValueError):
    """Raised when the state of a network cannot be reached from any past
    state."""

    def __init__(self, state, message):
        self.state = state
        self.message = message

    def __str__(self):
        return self.message


def direction(direction):
    if direction not in constants.DIRECTIONS:
        raise ValueError("Direction must be either 'past' or 'future'.")
    return True


def tpm(tpm):
    """Validate a TPM."""
    see_tpm_docs = ('See documentation for pyphi.Network for more information '
                    'TPM formats.')
    # Cast to np.array.
    tpm = np.array(tpm)
    # Get the number of nodes from the state-by-node TPM.
    N = tpm.shape[-1]
    if tpm.ndim == 2:
        if not ((tpm.shape[0] == 2**N and tpm.shape[1] == N) or
                (tpm.shape[0] == tpm.shape[1])):
            raise ValueError(
                'Invalid shape for 2-D TPM: {}\nFor a state-by-node TPM, '
                'there must be ' '2^N rows and N columns, where N is the '
                'number of nodes. State-by-state TPM must be square. '
                '{}'.format(tpm.shape, see_tpm_docs))
        if (tpm.shape[0] == tpm.shape[1]
                and not conditionally_independent(tpm)):
            raise ValueError('TPM is not conditionally independent. See the '
                             'conditional independence example in the '
                             'documentation for more information.')
    elif tpm.ndim == (N + 1):
        if not (tpm.shape == tuple([2] * N + [N])):
            raise ValueError(
                'Invalid shape for N-D state-by-node TPM: {}\nThe shape '
                'should be {} for {} nodes.'.format(
                    tpm.shape, ([2] * N) + [N], N, see_tpm_docs))
    else:
        raise ValueError(
            'Invalid state-by-node TPM: TPM must be in either 2-D or N-D '
            'form. {}'.format(see_tpm_docs))
    return True


def conditionally_independent(tpm):
    tpm = np.array(tpm)
    there_and_back_again = convert.state_by_node2state_by_state(
        convert.state_by_state2state_by_node(tpm))
    return np.all((tpm - there_and_back_again) < EPSILON)


def connectivity_matrix(cm):
    # Special case for empty matrices.
    if cm.size == 0:
        return True
    if (cm.ndim != 2):
        raise ValueError("Connectivity matrix must be 2-dimensional.")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("Connectivity matrix must be square.")
    if not np.all(np.logical_or(cm == 1, cm == 0)):
        raise ValueError("Connectivity matrix must contain only binary "
                         "values.")
    return True


# TODO test
def perturb_vector(pv, size):
    """Validate a network's pertubation vector."""
    if pv.size != size:
        raise ValueError("Perturbation vector must have one element per node.")
    if np.any(pv > 1) or np.any(pv < 0):
        raise ValueError("Perturbation vector elements must be probabilities, "
                         "between 0 and 1.")
    return True


def network(n):
    """Validate a network's TPM, connectivity matrix, and perturbation
    vector."""
    tpm(n.tpm)
    connectivity_matrix(n.connectivity_matrix)
    perturb_vector(n.perturb_vector, n.size)
    if n.connectivity_matrix.shape[0] != n.size:
        raise ValueError("Connectivity matrix must be NxN, where N is the "
                         "number of nodes in the network.")
    return True


def node_states(state):
    """Check that a state contains only zeros and ones."""
    if not all([n in (0, 1) for n in state]):
        raise ValueError(
            'Invalid state: states must consist of only zeros and ones.')


def state_length(state, size):
    if len(state) != size:
        raise ValueError('Invalid state: there must be one entry per '
                         'node in the network; this state has {} entries, but '
                         'there are {} nodes.'.format(len(state), size))
    return True


def state_reachable(subsystem):
    """Return whether a state can be reached according to the given network's
    TPM.

    If ``constrained_nodes`` is provided, then nodes not in
    `constrained_nodes`` will be left free (their state will not considered
    restricted by the TPM). Otherwise, any nodes without inputs will be left
    free."""
    # If there is a row `r` in the TPM such that all entries of `r - state` are
    # between -1 and 1, then the given state has a nonzero probability of being
    # reached from some state.
    # First we take the submatrix of the conditioned TPM that corresponds to
    # the nodes that are actually in the subsystem...
    tpm = subsystem.tpm[..., subsystem.node_indices]
    # Then we do the subtraction and test.
    test = tpm - np.array(subsystem.state)[list(subsystem.node_indices)]
    if not np.any(np.logical_and(-1 < test, test < 1).all(-1)):
        raise StateUnreachableError(
            subsystem.state, 'This state cannot be reached according to the '
                             'given TPM.')


def subsystem(s):
    """Validate a subsystem's state."""
    state_length(s.state, s.network.size)
    node_states(s.state)
    if config.VALIDATE_SUBSYSTEM_STATES:
        state_reachable(s)
    return True
