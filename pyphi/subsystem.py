#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem.py

"""Represents a candidate system for |small_phi| and |big_phi| evaluation."""

import itertools

import numpy as np

from . import cache, config, utils, validate
from .config import PRECISION
from .constants import DIRECTIONS, FUTURE, PAST
from .jsonify import jsonify
from .models import Concept, Cut, Mice, Mip, _null_mip, Part, Bipartition
from .network import irreducible_purviews
from .node import generate_nodes


class Subsystem:
    # TODO! go through docs and make sure to say when things can be None
    # TODO: make subsystem attributes immutable
    """A set of nodes in a network.

    Args:
        network (Network): The network the subsystem belongs to.
        state (tuple[int]): The state of the network.
        nodes (tuple[int] or tuple[str]): The nodes of the network which are in
            this subsystem. Nodes can be specified either as indices or as
            labels if the |Network| was passed ``node_labels``.

    Keyword Args:
        cut (Cut): The unidirectional |Cut| to apply to this subsystem.

    Attributes:
        network (Network): The network the subsystem belongs to.
        tpm (np.array): The TPM conditioned on the state of the external nodes.
        cm (np.array): The connectivity matrix after applying the cut.
        nodes (list[Node]): A list of nodes in the subsystem.
        node_indices (tuple[int]): The indices of the nodes in the subsystem.
        cut (Cut): The cut that has been applied to this subsystem.
        cut_matrix (np.array): A matrix of connections which have been severed
            by the cut.
        null_cut (Cut): The cut object representing no cut.
        perturb_vector (np.array): The vector of perturbation probabilities for
            each node.
    """

    def __init__(self, network, state, nodes, cut=None,
                 mice_cache=None, repertoire_cache=None):
        # The network this subsystem belongs to.
        self.network = network

        # Remove duplicates, sort, and ensure native Python `int`s
        # (for JSON serialization).
        self.node_indices = network.parse_node_indices(nodes)

        validate.state_length(state, self.network.size)

        # The state of the network.
        self._state = tuple(state)

        # Get the external node indices.
        # TODO: don't expose this as an attribute?
        self.external_indices = tuple(
            set(network.node_indices) - set(self.node_indices))

        # The TPM conditioned on the state of the external nodes.
        self.tpm = utils.condition_tpm(
            self.network.tpm, self.external_indices, self.state)

        # The null cut (that leaves the system intact)
        self.null_cut = Cut((), self.cut_indices)

        # The unidirectional cut applied for phi evaluation
        self.cut = cut if cut is not None else self.null_cut

        # The matrix of connections which are severed due to the cut
        # Note: this matrix is N x N, where N is the number of elements in
        # the subsystem, *not* the number of elements in the network.
        # TODO: save/memoize on the cut so we just say self.cut.matrix()?
        self.cut_matrix = self.cut.cut_matrix()

        # The network's connectivity matrix with cut applied
        self.cm = utils.apply_cut(cut, network.cm)

        # The perturbation probabilities for each node in the network
        self.perturb_vector = network.perturb_vector

        # Only compute hash once.
        self._hash = hash((self.network, self.node_indices, self.state,
                           self.cut))

        # Reusable cache for core causes & effects
        self._mice_cache = cache.MiceCache(self, mice_cache)

        # Cause & effect repertoire cache
        # TODO: if repertoire caches are never reused, there's no reason to
        # have an accesible object-level cache. Just use a simple memoizer
        self._repertoire_cache = repertoire_cache or cache.DictCache()

        self.nodes = generate_nodes(self, labels=True)

        validate.subsystem(self)

    @property
    def state(self):
        """tuple[int]: The state of the Network this Subsystem belongs to."""
        return self._state

    @state.setter
    def state(self, state):
        # Cast state to a tuple so it can be hashed and properly used as
        # np.array indices.
        self._state = tuple(state)
        # Validate.
        validate.subsystem(self)

    @property
    def proper_state(self):
        """tuple[int]): The state of the subsystem.

        ``proper_state[i]`` gives the state of the |ith| node **in the
        subsystem**. Note that this is **not** the state of ``nodes[i]``.
        """
        return utils.state_of(self.node_indices, self.state)

    @proper_state.setter
    def proper_state(self, proper_state):
        # Update the network's state.
        self.state = tuple(proper_state[self.node_indices.index(n)]
                           if n in self.node_indices else self.state[n]
                           for n in self.network.node_indices)
        # Validate.
        validate.subsystem(self)

    @property
    def connectivity_matrix(self):
        """np.ndarray: Alias for ``Subsystem.cm``."""
        return self.cm

    @connectivity_matrix.setter
    def connectivity_matrix(self, cm):
        self.cm = cm

    @property
    def size(self):
        """int: The number of nodes in the subsystem."""
        return len(self.node_indices)

    @property
    def is_cut(self):
        """boolean: True if this Subsystem has a cut applied to it."""
        return self.cut != self.null_cut

    @property
    def cut_indices(self):
        """tuple[int]: The nodes of this subsystem cut for |big_phi|
        computations.

        This was added to support ``MacroSubsystem``, which cuts indices other
        than ``node_indices``.
        """
        return self.node_indices

    @property
    def tpm_indices(self):
        """tuple[int]: The indices of nodes in the tpm."""
        return tuple(range(self.tpm.shape[-1]))

    def repertoire_cache_info(self):
        """Report repertoire cache statistics."""
        return self._repertoire_cache.info()

    def __repr__(self):
        """Return a representation of this Subsystem."""
        return "Subsystem(" + repr(self.nodes) + ")"

    def __str__(self):
        """Return this Subsystem as a string."""
        return repr(self)

    def __eq__(self, other):
        """Return whether this Subsystem is equal to the other object.

        Two Subsystems are equal if their sets of nodes, networks, and cuts are
        equal.
        """
        return (set(self.node_indices) == set(other.node_indices)
                and self.state == other.state
                and self.network == other.network
                and self.cut == other.cut)

    def __bool__(self):
        """Return false if the Subsystem has no nodes, true otherwise."""
        return bool(self.nodes)

    def __ne__(self, other):
        """Return whether this Subsystem is not equal to the other object."""
        return not self.__eq__(other)

    def __ge__(self, other):
        """Return whether this Subsystem >= the other object."""
        return len(self.nodes) >= len(other.nodes)

    def __le__(self, other):
        """Return whether this Subsystem <= the other object."""
        return len(self.nodes) <= len(other.nodes)

    def __gt__(self, other):
        """Return whether this Subsystem > the other object."""
        return len(self.nodes) > len(other.nodes)

    def __lt__(self, other):
        """Return whether this Subsystem < the other object."""
        return len(self.nodes) < len(other.nodes)

    def __len__(self):
        """Return the number of nodes in this Subsystem."""
        return len(self.node_indices)

    def __hash__(self):
        """Return the hash value of this Subsystem."""
        return self._hash

    def to_json(self):
        """Return this Subsystem as a JSON object."""
        return {
            'node_indices': jsonify(self.node_indices),
            'cut': jsonify(self.cut),
        }

    def apply_cut(self, cut):
        """Return a cut version of this |Subsystem|.

        Args:
            cut (|Cut|): The cut to apply to this |Subsystem|.

        Returns:
            |Subsystem|
        """
        return Subsystem(self.network, self.state, self.node_indices,
                         cut=cut, mice_cache=self._mice_cache)

    def indices2nodes(self, indices):
        """Return nodes for these indices.

        Args:
            indices (tuple[int]): The indices in question.

        Returns:
            tuple[Node]: The |Node| objects corresponding to these indices.

        Raises:
            ValueError: If requested indices are not in the subsystem.
        """
        if not indices:
            return ()

        if set(indices) - set(self.node_indices):
            raise ValueError(
                "`indices` must be a subset of the Subsystem's indices.")

        return tuple(n for n in self.nodes if n.index in indices)

    def indices2labels(self, indices):
        """Returns the node labels for these indices."""
        return tuple(n.label for n in self.indices2nodes(indices))

    @cache.method('_repertoire_cache', DIRECTIONS[PAST])
    def cause_repertoire(self, mechanism, purview):
        """Return the cause repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                cause repertoire.
            purview (tuple[int]): The purview over which to calculate the
                cause repertoire.

        Returns:
            ``np.ndarray``: The cause repertoire of the mechanism over the purview.

        .. note::
            The returned repertoire is a distribution over the nodes in the
            purview, not the whole network. This is because we never actually
            need to compare proper cause/effect repertoires, which are
            distributions over the whole network; we need only compare the
            purview-repertoires with each other, since cut vs. whole
            comparisons are only ever done over the same purview.
        """
        # If the purview is empty, the distribution is empty; return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])

        # If the mechanism is empty, nothing is specified about the past state
        # of the purview -- return the purview's maximum entropy distribution.
        max_entropy_dist = utils.max_entropy_distribution(
            purview, len(self.tpm_indices),
            tuple(self.perturb_vector[i] for i in purview))
        if not mechanism:
            return max_entropy_dist

        # Preallocate the mechanism's conditional joint distribution.
        # TODO extend to nonbinary nodes
        cjd = np.ones([2 if i in purview else 1 for i in self.tpm_indices])

        # Loop over all nodes in this mechanism, successively taking the
        # product (with expansion/broadcasting of singleton dimensions) of each
        # individual node's TPM (conditioned on that node's state) in order to
        # get the conditional joint distribution for the whole mechanism
        # (conditioned on the whole mechanism's state).
        for mechanism_node in self.indices2nodes(mechanism):
            # TODO extend to nonbinary nodes
            # We're conditioning on this node's state, so take the probability
            # table for the node being in that state.
            conditioned_tpm = mechanism_node.tpm[mechanism_node.state]

            # Marginalize-out all nodes which connect to this node but which
            # are not in the purview:
            non_purview_inputs = (set(mechanism_node.input_indices) -
                                  set(purview))
            for index in non_purview_inputs:
                conditioned_tpm = utils.marginalize_out(
                    index, conditioned_tpm, self.perturb_vector[index])

            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product (with singleton
            # broadcasting, which spreads the singleton probabilities in the
            # collapsed dimensions out along the whole distribution in the
            # appropriate way.
            cjd *= conditioned_tpm

        # If the perturbation vector is not maximum entropy, weight the
        # probabilities before normalization.
        if not np.all(self.perturb_vector == 0.5):
            cjd *= max_entropy_dist

        return utils.normalize(cjd)

    @cache.method('_repertoire_cache', DIRECTIONS[FUTURE])
    def effect_repertoire(self, mechanism, purview):
        """Return the effect repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                effect repertoire.
            purview (tuple[int]): The purview over which to calculate the
                effect repertoire.

        Returns:
            ``np.ndarray``: The effect repertoire of the mechanism over the
                purview.

        .. note::
            The returned repertoire is a distribution over the nodes in the
            purview, not the whole network. This is because we never actually
            need to compare proper cause/effect repertoires, which are
            distributions over the whole network; we need only compare the
            purview-repertoires with each other, since cut vs. whole
            comparisons are only ever done over the same purview.
        """
        purview_nodes = self.indices2nodes(purview)
        mechanism_nodes = self.indices2nodes(mechanism)

        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])

        # Preallocate the purview's joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(
            [1] * len(self.tpm_indices) +
            [2 if i in purview else 1 for i in self.tpm_indices])

        # Loop over all nodes in the purview, successively taking the product
        # (with 'expansion'/'broadcasting' of singleton dimensions) of each
        # individual node's TPM in order to get the joint distribution for the
        # whole purview.
        for purview_node in purview_nodes:
            # Unlike in calculating the cause repertoire, here the TPM is not
            # conditioned yet. `tpm` is an array with twice as many dimensions
            # as the network has nodes. For example, in a network with three
            # nodes {n0, n1, n2}, the CPT for node n1 would have shape
            # (2,2,2,1,2,1). The CPT for the node being off would be given by
            # `tpm[:,:,:,0,0,0]`, and the CPT for the node being on would be
            # given by `tpm[:,:,:,0,1,0]`. The second half of the shape is for
            # indexing based on the current node's state, and the first half of
            # the shape is the CPT indexed by network state, so that the
            # overall CPT can be broadcast over the `accumulated_cjd` and then
            # later conditioned by indexing.
            # TODO extend to nonbinary nodes

            # Rotate the dimensions so the first dimension is the last (the
            # first dimension corresponds to the state of the node)
            tpm = purview_node.tpm
            tpm = tpm.transpose(list(range(tpm.ndim))[1:] + [0])

            # Expand the dimensions so the TPM can be indexed as described
            first_half_shape = list(tpm.shape[:-1])
            second_half_shape = [1] * len(self.tpm_indices)
            second_half_shape[purview_node.index] = 2
            tpm = tpm.reshape(first_half_shape + second_half_shape)

            # Marginalize-out non-mechanism purview inputs.
            non_mechanism_inputs = (set(purview_node.input_indices) -
                                    set(mechanism))
            for index in non_mechanism_inputs:
                tpm = utils.marginalize_out(index, tpm,
                                            self.perturb_vector[index])

            # Incorporate this node's CPT into the future_nodes' conditional
            # joint distribution (with singleton broadcasting).
            accumulated_cjd = accumulated_cjd * tpm

        # Collect all mechanism nodes which input to purview nodes; condition
        # on the state of these nodes by collapsing the CJD onto those states.
        mechanism_inputs = [node.index for node in mechanism_nodes
                            if set(node.output_indices) & set(purview)]
        accumulated_cjd = utils.condition_tpm(
            accumulated_cjd, mechanism_inputs, self.state)

        # The distribution still has twice as many dimensions as the network
        # has nodes, with the first half of the shape now all singleton
        # dimensions, so we reshape to eliminate those singleton dimensions
        # (the second half of the shape may also contain singleton dimensions,
        # depending on how many nodes are in the purview).
        accumulated_cjd = accumulated_cjd.reshape(
            accumulated_cjd.shape[len(self.tpm_indices):])

        return accumulated_cjd

    def _repertoire(self, direction, mechanism, purview):
        """Return the cause or effect repertoire based on a direction.

        Args:
            direction (str): One of 'past' or 'future'.
            mechanism (tuple[int]): The mechanism for which to calculate the
                repertoire.
            purview (tuple[int]): The purview over which to calculate the
                repertoire.

        Returns:
            np.ndarray: The cause or effect repertoire of the mechanism over
                the purview.
        """
        if direction == DIRECTIONS[PAST]:
            return self.cause_repertoire(mechanism, purview)
        elif direction == DIRECTIONS[FUTURE]:
            return self.effect_repertoire(mechanism, purview)

    def _unconstrained_repertoire(self, direction, purview):
        """Return the unconstrained cause/effect repertoire over a purview."""
        return self._repertoire(direction, (), purview)

    def unconstrained_cause_repertoire(self, purview):
        """Return the unconstrained cause repertoire for a purview.

        This is just the cause repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire(DIRECTIONS[PAST], purview)

    def unconstrained_effect_repertoire(self, purview):
        """Return the unconstrained effect repertoire for a purview.

        This is just the effect repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire(DIRECTIONS[FUTURE], purview)

    def partitioned_repertoire(self, direction, partition):
        """Compute the repertoire of a partitioned mechanism and purview."""
        part1rep = self._repertoire(direction, partition[0].mechanism,
                                    partition[0].purview)
        part2rep = self._repertoire(direction, partition[1].mechanism,
                                    partition[1].purview)

        return part1rep * part2rep

    def expand_repertoire(self, direction, repertoire, new_purview=None):
        """Expand a partial repertoire over a purview to a distribution over a
        new state space.

        Args:
            direction (str): Either |past| or |future|.
            repertoire (np.ndarray): A repertoire.

        Keyword Args:
            new_purview (tuple[int]): The purview to expand the repertoire
                over. Defaults to the entire subsystem.

        Returns:
            np.ndarray: The expanded repertoire.
        """
        purview = utils.purview(repertoire)

        if new_purview is None:
            new_purview = self.node_indices  # full subsystem

        if not set(purview).issubset(new_purview):
            raise ValueError("Expanded purview must contain original purview.")

        # Get the unconstrained repertoire over the other nodes in the network.
        non_purview_indices = tuple(set(new_purview) - set(purview))
        uc = self._unconstrained_repertoire(direction, non_purview_indices)
        # Multiply the given repertoire by the unconstrained one to get a
        # distribution over all the nodes in the network.
        expanded_repertoire = repertoire * uc

        return utils.normalize(expanded_repertoire)

    def expand_cause_repertoire(self, repertoire, new_purview=None):
        """Expand a partial cause repertoire over a purview to a distribution
        over the entire subsystem's state space.
        """
        return self.expand_repertoire(DIRECTIONS[PAST], repertoire,
                                      new_purview)

    def expand_effect_repertoire(self, repertoire, new_purview=None):
        """Expand a partial effect repertoire over a purview to a distribution
        over the entire subsystem's state space.
        """
        return self.expand_repertoire(DIRECTIONS[FUTURE], repertoire,
                                      new_purview)

    def cause_info(self, mechanism, purview):
        """Return the cause information for a mechanism over a purview."""
        return emd(DIRECTIONS[PAST],
                   self.cause_repertoire(mechanism, purview),
                   self.unconstrained_cause_repertoire(purview))

    def effect_info(self, mechanism, purview):
        """Return the effect information for a mechanism over a purview."""
        return emd(DIRECTIONS[FUTURE],
                   self.effect_repertoire(mechanism, purview),
                   self.unconstrained_effect_repertoire(purview))

    def cause_effect_info(self, mechanism, purview):
        """Return the cause-effect information for a mechanism over a purview.

        This is the minimum of the cause and effect information.
        """
        return min(self.cause_info(mechanism, purview),
                   self.effect_info(mechanism, purview))

    # MIP methods
    # =========================================================================

    def find_mip(self, direction, mechanism, purview):
        """Return the minimum information partition for a mechanism over a
        purview.

        Args:
            direction (str): Either |past| or |future|.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.

        Returns:
            |Mip|: The mininum-information partition in one temporal direction.
        """
        # We default to the null MIP (the MIP of a reducible mechanism)
        mip = _null_mip(direction, mechanism, purview)

        if not purview:
            return mip

        phi_min = float('inf')
        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones
        unpartitioned_repertoire = self._repertoire(direction, mechanism,
                                                    purview)

        def _mip(phi, partition, partitioned_repertoire):
            # Prototype of MIP with already known data
            # TODO: Use properties here to infer mechanism and purview from
            # partition yet access them with `.mechanism` and `.purview`.
            return Mip(phi=phi,
                       direction=direction,
                       mechanism=mechanism,
                       purview=purview,
                       partition=partition,
                       unpartitioned_repertoire=unpartitioned_repertoire,
                       partitioned_repertoire=partitioned_repertoire,
                       subsystem=self)

        # State is unreachable - return 0 instead of giving nonsense results
        if (direction == DIRECTIONS[PAST] and
                np.all(unpartitioned_repertoire == 0)):
            return _mip(0, None, None)

        # Loop over possible MIP bipartitions
        for partition in mip_bipartitions(mechanism, purview):
            partitioned_repertoire = self.partitioned_repertoire(direction,
                                                                 partition)

            if config.L1_DISTANCE_APPROXIMATION:
                phi = utils.l1(unpartitioned_repertoire,
                               partitioned_repertoire)
                phi = round(phi, PRECISION)
            else:
                phi = emd(direction, unpartitioned_repertoire,
                          partitioned_repertoire)

            # Return immediately if mechanism is reducible.
            if phi == 0:
                return _mip(0.0, partition, partitioned_repertoire)

            # Update MIP if it's more minimal.
            if phi < phi_min:
                phi_min = phi
                mip = _mip(phi, partition, partitioned_repertoire)

        # Recompute distance for minimal MIP using the EMD
        if config.L1_DISTANCE_APPROXIMATION:
            phi = emd(direction, mip.unpartitioned_repertoire,
                      mip.partitioned_repertoire)
            mip = _mip(phi, mip.partition, mip.partitioned_repertoire)

        return mip

    def mip_past(self, mechanism, purview):
        """Return the past minimum information partition.

        Alias for |find_mip| with ``direction`` set to |past|.
        """
        return self.find_mip(DIRECTIONS[PAST], mechanism, purview)

    def mip_future(self, mechanism, purview):
        """Return the future minimum information partition.

        Alias for |find_mip| with ``direction`` set to |future|.
        """
        return self.find_mip(DIRECTIONS[FUTURE], mechanism, purview)

    def phi_mip_past(self, mechanism, purview):
        """Return the |small_phi| of the past minimum information partition.

        This is the distance between the unpartitioned cause repertoire and the
        MIP cause repertoire.
        """
        mip = self.mip_past(mechanism, purview)
        return mip.phi if mip else 0

    def phi_mip_future(self, mechanism, purview):
        """Return the |small_phi| of the future minimum information partition.

        This is the distance between the unpartitioned effect repertoire and
        the MIP cause repertoire.
        """
        mip = self.mip_future(mechanism, purview)
        return mip.phi if mip else 0

    def phi(self, mechanism, purview):
        """Return the |small_phi| of a mechanism over a purview."""
        return min(self.phi_mip_past(mechanism, purview),
                   self.phi_mip_future(mechanism, purview))

    # Phi_max methods
    # =========================================================================

    def _potential_purviews(self, direction, mechanism, purviews=False):
        """Return all purviews that could belong to the core cause/effect.

        Filters out trivially-reducible purviews.

        Args:
            direction ('str'): Either |past| or |future|.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Args:
            purviews (tuple[int]): Optional subset of purviews of interest.
        """
        if purviews is False:
            purviews = self.network._potential_purviews(direction, mechanism)
            # Filter out purviews that aren't in the subsystem
            purviews = [purview for purview in purviews
                        if set(purview).issubset(self.node_indices)]

        # Purviews are already filtered in network._potential_purviews
        # over the full network connectivity matrix. However, since the cm
        # is cut/smaller we check again here.
        return irreducible_purviews(self.cm, direction, mechanism, purviews)

    @cache.method('_mice_cache')
    def find_mice(self, direction, mechanism, purviews=False):
        """Return the maximally irreducible cause or effect for a mechanism.

        Args:
            direction (str): The temporal direction (|past| or |future|)
                specifying cause or effect.
            mechanism (tuple[int]): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple[int]): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.

        Returns:
            |Mice|: The maximally-irreducible cause or effect.

        .. note::
            Strictly speaking, the MICE is a pair of repertoires: the core
            cause repertoire and core effect repertoire of a mechanism, which
            are maximally different than the unconstrained cause/effect
            repertoires (*i.e.*, those that maximize |small_phi|). Here, we
            return only information corresponding to one direction, |past| or
            |future|, i.e., we return a core cause or core effect, not the pair
            of them.
        """
        purviews = self._potential_purviews(direction, mechanism, purviews)

        if not purviews:
            max_mip = _null_mip(direction, mechanism, ())
        else:
            max_mip = max(self.find_mip(direction, mechanism, purview)
                          for purview in purviews)

        return Mice(max_mip)

    def core_cause(self, mechanism, purviews=False):
        """Return the core cause repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|.
        """
        return self.find_mice('past', mechanism, purviews=purviews)

    def core_effect(self, mechanism, purviews=False):
        """Return the core effect repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|.
        """
        return self.find_mice('future', mechanism, purviews=purviews)

    def phi_max(self, mechanism):
        """Return the |small_phi_max| of a mechanism.

        This is the maximum of |small_phi| taken over all possible purviews.
        """
        return min(self.core_cause(mechanism).phi,
                   self.core_effect(mechanism).phi)

    # Big Phi methods
    # =========================================================================

    # TODO add `concept-space` section to the docs:
    @property
    def null_concept(self):
        """Return the null concept of this subsystem.

        The null concept is a point in concept space identified with
        the unconstrained cause and effect repertoire of this subsystem.
        """
        # Unconstrained cause repertoire.
        cause_repertoire = self.cause_repertoire((), ())
        # Unconstrained effect repertoire.
        effect_repertoire = self.effect_repertoire((), ())

        # Null cause.
        cause = Mice(_null_mip(DIRECTIONS[PAST], (), (), cause_repertoire))
        # Null effect.
        effect = Mice(_null_mip(DIRECTIONS[FUTURE], (), (), effect_repertoire))

        # All together now...
        return Concept(mechanism=(), phi=0, cause=cause, effect=effect,
                       subsystem=self)

    def concept(self, mechanism, purviews=False, past_purviews=False,
                future_purviews=False):
        """Calculate a concept.

        See :func:`pyphi.compute.concept` for more information.
        """
        # Calculate the maximally irreducible cause repertoire.
        cause = self.core_cause(mechanism,
                                purviews=(past_purviews or purviews))
        # Calculate the maximally irreducible effect repertoire.
        effect = self.core_effect(mechanism,
                                  purviews=(future_purviews or purviews))
        # Get the minimal phi between them.
        phi = min(cause.phi, effect.phi)
        # NOTE: Make sure to expand the repertoires to the size of the
        # subsystem when calculating concept distance. For now, they must
        # remain un-expanded so the concept doesn't depend on the subsystem.
        return Concept(mechanism=mechanism, phi=phi, cause=cause,
                       effect=effect, subsystem=self)


def mip_bipartitions(mechanism, purview):
    """Return all |small_phi| bipartitions of a mechanism over a purview.

    Excludes all bipartitions where one half is entirely empty, e.g::

         A    []                     A    []
        --- X -- is not valid,  but --- X --- is.
         B    []                    []     B

    Args:
        mechanism (tuple[int]): The mechanism to partition
        purview (tuple[int]): The purview to partition

    Returns:
        list[|Bipartition|]: Where each partition is

        ::

            bipart[0].mechanism   bipart[1].mechanism
            ------------------- X -------------------
            bipart[0].purview     bipart[1].purview

    Example:
        >>> mechanism = (0,)
        >>> purview = (2, 3)
        >>> for partition in mip_bipartitions(mechanism, purview):
        ...     print(partition, "\\n")  # doctest: +NORMALIZE_WHITESPACE
        []   0
        -- X -
        2    3
        <BLANKLINE>
        []   0
        -- X -
        3    2
        <BLANKLINE>
        []    0
        --- X --
        2,3   []
    """
    numerators = utils.bipartition(mechanism)
    denominators = utils.directed_bipartition(purview)

    return [Bipartition(Part(n[0], d[0]), Part(n[1], d[1]))
            for (n, d) in itertools.product(numerators, denominators)
            if len(n[0]) + len(d[0]) > 0 and len(n[1]) + len(d[1]) > 0]


def effect_emd(d1, d2):
    """Compute the EMD between two effect repertoires.

    Billy's synopsis: Because the nodes are independent, the EMD between
    effect repertoires is equal to the sum of the EMDs between the marginal
    distributions of each node, and the EMD between marginal distribution for a
    node is the absolute difference in the probabilities that the node is off.

    Args:
        d1 (np.ndarray): The first repertoire.
        d2 (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``d1`` and ``d2``.
    """
    return sum(np.abs(utils.marginal_zero(d1, i) - utils.marginal_zero(d2, i))
               for i in range(d1.ndim))


def emd(direction, d1, d2):
    """Compute the EMD between two repertoires for a given direction.

    The full EMD computation is used for cause repertoires. A fast analytic
    solution is used for effect repertoires.

    Args:
        direction (str): Either |past| or |future|.
        d1 (np.ndarray): The first repertoire.
        d2 (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``d1`` and ``d2``, rounded to |PRECISION|.
    """

    if direction == DIRECTIONS[PAST]:
        func = utils.hamming_emd
    elif direction == DIRECTIONS[FUTURE]:
        func = effect_emd

    return round(func(d1, d2), PRECISION)
