#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# node.py
"""
Represents a node in a subsystem. Each node has a unique index, its position
in the subsystem's list of nodes.
"""

import functools
import numpy as np
from . import utils


# TODO extend to nonbinary nodes
# TODO? refactor to use purely indexes for nodes
@functools.total_ordering
class Node:

    """A node in a subsystem.

    Attributes:
        index (int):
            The node's index in the subsystem's list of nodes.
        subsystem (Subsystem):
            The subsystem the node belongs to.
        label (str):
            An optional label for the node.
        inputs (list(Node)):
            A list of nodes that have connections to this node.
        past_tpm (np.ndarray):
            The TPM for this node, conditioned on the past state of the
            boundary nodes, whose states are fixed. ``this_node.past_tpm[0]``
            and ``this_node.past_tpm[1]`` gives the probability tables that
            this node is off and on, respectively, indexed by subsystem state,
            **after marginalizing-out nodes that don't connect to this node**.
        current_tpm (np.ndarray):
            Same as ``past_tpm``, but conditioned on the current state of the
            boundary nodes.

    Examples:
        In a 3-node subsystem, ``self.tpm[0][(0, 0, 1)]`` gives the
        probability that this node is off at |t_1| if the state of the subsystem
        is |N_0 = 0, N_1 = 0, N_2 = 1| at |t_0|.

       """

    def __init__(self, index, subsystem, label=None):
        # This node's index in the subsystem's list of nodes.
        self.index = index
        # This node's parent subsystem.
        self.subsystem = subsystem
        # Label for display.
        self.label = label
        # State of this node.
        self.state = self.subsystem.current_state[self.index]
        # Get indices of the inputs.
        self._input_indices = utils.get_inputs_from_cm(
            self.index, subsystem.micro_connectivity_matrix)
        self._output_indices = utils.get_outputs_from_cm(
            self.index, subsystem.micro_connectivity_matrix)
        # Generate the node's TPMs.
        # TODO update this description
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For the past and current state, get the part of the subsystem's TPM
        # that gives just the state of this node. This part is still indexed by
        # network state, but its last dimension will be gone, since now there's
        # just a single scalar value (this node's state) rather than a
        # state-vector for all the network nodes.
        tpm_on = self.subsystem.tpm[..., self.index]
        # Get the TPMs that give the probability of the node being off, rather
        # than on.
        tpm_off = 1 - tpm_on
        # Marginalize-out non-input subsystem nodes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO extend to nonbinary nodes
        # Marginalize out non-input nodes that are in the subsystem, since
        # the external nodes have already been dealt with as boundary
        # conditions in the subsystem's TPMs.
        for i in self.subsystem.subsystem_indices:
            if i not in self._input_indices:
                tpm_on = tpm_on.sum(i, keepdims=True) / 2
                tpm_off = tpm_off.sum(i, keepdims=True) / 2

        # Combine the on- and off-TPMs.
        self.tpm = np.array([tpm_off, tpm_on])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Make the TPM immutable (for hashing).
        self.tpm.flags.writeable = False

        # Only compute the hash once.
        self._hash = hash((self.index, self.subsystem))

        # Deferred properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ``inputs``, ``outputs``, and ``marbl`` must be properties because at
        # the time of node creation, the subsystem doesn't have a list of Node
        # objects yet, only a size (and thus a range of node indices). So, we
        # defer construction until the properties are needed.
        self._inputs = None
        self._outputs = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def inputs(self):
        """The set of nodes with connections to this node."""
        if self._inputs is not None:
            return self._inputs
        else:
            self._inputs = [node for node in self.subsystem.nodes if
                            node.index in self._input_indices]
            return self._inputs

    @property
    def outputs(self):
        """The set of nodes this node has connections to."""
        if self._outputs is not None:
            return self._outputs
        else:
            self._outputs = [node for node in self.subsystem.nodes if
                             node.index in self._output_indices]
            return self._outputs

    def __repr__(self):
        return (self.label if self.label is not None
                else 'n' + str(self.index))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same subsystem and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (self.index == other.index and self.subsystem == other.subsystem)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        return self._hash

    # TODO do we need more than the index?
    def json_dict(self):
        return self.index
