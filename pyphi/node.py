#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import numpy as np
from marbl import Marbl
from . import utils, json
from .constants import DIRECTIONS, PAST, FUTURE


# TODO extend to nonbinary nodes
# TODO? refactor to use purely indexes for nodes
@functools.total_ordering
class Node:

    """A node in a context.

    Attributes:
        network (network):
            The network the node belongs to.
        index (int):
            The node's index in the network's list of nodes.
        context (context):
            The context the node belongs to.
        label (str):
            An optional label for the node.
        inputs (list(Node)):
            A list of nodes that have connections to this node.
        past_tpm (np.ndarray):
            The TPM for this node, conditioned on the past state of the
            boundary nodes, whose states are fixed. ``this_node.past_tpm[0]``
            and ``this_node.past_tpm[1]`` gives the probability tables that
            this node is off and on, respectively, indexed by context state,
            **after marginalizing-out nodes that don't connect to this node**.
        current_tpm (np.ndarray):
            Same as ``past_tpm``, but conditioned on the current state of the
            boundary nodes.

    Examples:
        In a 3-node context, ``self.past_tpm[0][(0, 1, 0)]`` gives the
        probability that this node is off at |t_0| if the state of the network
        is |0,1,0| at |t_{-1}|.

        Similarly, ``self.current_tpm[0][(0, 1, 0)]`` gives the probability
        that this node is off at |t_1| if the state of the network is |0,1,0|
        at |t_0|.
    """

    def __init__(self, network, index, context, label=None):
        # This node's parent network.
        self.network = network
        # This node's index in the network's list of nodes.
        self.index = index
        # This node's parent context.
        self.context = context
        # Label for display.
        self.label = label
        # State of this node.
        self.state = self.network.current_state[self.index]
        # Get indices of the inputs.
        self._input_indices = utils.get_inputs_from_cm(
            self.index, context.connectivity_matrix)
        self._output_indices = utils.get_outputs_from_cm(
            self.index, context.connectivity_matrix)
        # Generate the node's TPMs.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For the past and current state, get the part of the context's TPM
        # that gives just the state of this node. This part is still indexed by
        # network state, but its last dimension will be gone, since now there's
        # just a single scalar value (this node's state) rather than a
        # state-vector for all the network nodes.
        past_tpm_on = self.context.past_tpm[..., self.index]
        current_tpm_on = self.context.current_tpm[..., self.index]
        # Get the TPMs that give the probability of the node being off, rather
        # than on.
        past_tpm_off = 1 - past_tpm_on
        current_tpm_off = 1 - current_tpm_on
        # Marginalize-out non-input context nodes and get dimension labels.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This list will hold the indices of the nodes that correspond to
        # non-singleton dimensions of this node's on-TPM. It maps any context
        # node index to the corresponding dimension of this node's TPM with
        # singleton dimensions removed. We need this for creating this node's
        # Marbl.
        self._dimension_labels = []
        # This is the counter that will provide the actual labels.
        current_non_singleton_dim_index = 0
        # Iterate over all the nodes in the network, since we need to keep
        # track of all singleton dimensions.
        for i in range(self.network.size):

            # Input nodes that are within the context will correspond to a
            # dimension in this node's squeezed TPM, so we map it to the index
            # of the corresponding dimension and increment the corresponding
            # index for the next one.
            if i in self._input_indices and i in self.context.node_indices:
                self._dimension_labels.append(current_non_singleton_dim_index)
                current_non_singleton_dim_index += 1
            # Boundary nodes and non-input nodes have already been conditioned
            # and marginalized-out, so their dimension in the TPM will be a
            # singleton and will be squeezed out when creating a Marbl. So, we
            # don't give them a dimension label.
            else:
                self._dimension_labels.append(None)

            # TODO extend to nonbinary nodes
            # Marginalize out non-input nodes that are in the context, since
            # the external nodes have already been dealt with as boundary
            # conditions in the context's TPMs.
            if i not in self._input_indices and i in self.context.node_indices:
                past_tpm_on = past_tpm_on.sum(i, keepdims=True) / 2
                past_tpm_off = past_tpm_off.sum(i, keepdims=True) / 2
                current_tpm_on = current_tpm_on.sum(i, keepdims=True) / 2
                current_tpm_off = current_tpm_off.sum(i, keepdims=True) / 2

        # Combine the on- and off-TPMs.
        self.past_tpm = np.array([past_tpm_off, past_tpm_on])
        self.current_tpm = np.array([current_tpm_off, current_tpm_on])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Make the TPM immutable (for hashing).
        self.past_tpm.flags.writeable = False
        self.current_tpm.flags.writeable = False

        # Only compute the hash once.
        self._hash = hash((self.index, self.context))

        # Deferred properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ``inputs``, ``outputs``, and ``marbl`` must be properties because at
        # the time of node creation, the context doesn't have a list of Node
        # objects yet, only a size (and thus a range of node indices). So, we
        # defer construction until the properties are needed.
        self._inputs = None
        self._outputs = None
        self._past_marbl = None
        self._current_marbl = None
        self._raw_past_marbl = None
        self._raw_current_marbl = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_marbl(self, direction, normalize=True):
        """Generate a Marbl for this node, using either the past or current
        TPM."""
        if direction == DIRECTIONS[PAST]:
            tpm_name = 'past_tpm'
        if direction == DIRECTIONS[FUTURE]:
            tpm_name = 'current_tpm'
        # We take only the part of the TPM giving the probability the node
        # is on.
        # TODO extend to nonbinary nodes
        augmented_child_tpms = [
            [child._dimension_labels[self.index],
             getattr(child, tpm_name)[1].squeeze()] for child in self.outputs
        ]
        marbl = Marbl(getattr(self, tpm_name)[1], augmented_child_tpms,
                      normalize=normalize)
        return marbl

    @property
    def inputs(self):
        """The set of nodes with connections to this node."""
        if self._inputs is not None:
            return self._inputs
        else:
            self._inputs = [node for node in self.context.nodes if
                            node.index in self._input_indices]
            return self._inputs

    @property
    def outputs(self):
        """The set of nodes this node has connections to."""
        if self._outputs is not None:
            return self._outputs
        else:
            self._outputs = [node for node in self.context.nodes if
                             node.index in self._output_indices]
            return self._outputs

    @property
    def past_marbl(self):
        """The normalized representation of this node's Markov blanket,
        conditioned on the fixed state of boundary-condition nodes in the
        previous timestep."""
        if self._past_marbl is not None:
            return self._past_marbl
        else:
            self._past_marbl = self.get_marbl(DIRECTIONS[PAST])
            return self._past_marbl

    @property
    def current_marbl(self):
        """The normalized representation of this node's Markov blanket,
        conditioned on the fixed state of boundary-condition nodes in the
        current timestep."""
        if self._current_marbl is not None:
            return self._current_marbl
        else:
            self._current_marbl = self.get_marbl(DIRECTIONS[FUTURE])
            return self._current_marbl

    @property
    def raw_past_marbl(self):
        """The un-normalized representation of this node's Markov blanket,
        conditioned on the fixed state of boundary-condition nodes in the
        previous timestep."""
        if self._past_marbl is not None:
            return self._past_marbl
        else:
            self._raw_past_marbl = self.get_marbl(DIRECTIONS[PAST],
                                                  normalize=False)
            return self._raw_past_marbl

    @property
    def raw_current_marbl(self):
        """The un-normalized representation of this node's Markov blanket,
        conditioned on the fixed state of boundary-condition nodes in the
        current timestep."""
        if self._raw_current_marbl is not None:
            return self._raw_current_marbl
        else:
            self._raw_current_marbl = self.get_marbl(DIRECTIONS[FUTURE],
                                                     normalize=False)
            return self._raw_current_marbl

    def __repr__(self):
        return (self.label if self.label is not None
                else 'n' + str(self.index))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same context and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (self.index == other.index and self.context == other.context)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        return self._hash

    # TODO do we need more than the index?
    def json_dict(self):
        return self.index
