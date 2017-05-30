#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""
Functions used by more than one PyPhi module or class, or that might be of
external use.
"""

import hashlib
import itertools
import logging
import os
from itertools import chain, combinations

import numpy as np

from pyemd import emd
from scipy.misc import comb
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from . import constants, convert
from .cache import cache

# Create a logger for this module.
log = logging.getLogger(__name__)


def state_of(nodes, network_state):
    """Return the state-tuple of the given nodes."""
    return tuple(network_state[n] for n in nodes) if nodes else ()


def all_states(n):
    """Return all binary states for a system.

    Args:
        n (int): The number of elements in the system.

    Yields:
        tuple[int]: The next state of an ``n``-element system, in LOLI order.
    """
    if n == 0:
        return

    for state in itertools.product((0, 1), repeat=n):
        yield state[::-1]  # Convert to LOLI-ordering


# Methods for converting the time scale of the tpm
# ================================================

def sparse(matrix, threshold=0.1):
    return np.sum(matrix > 0) / matrix.size > threshold


def sparse_time(tpm, time_scale):
    sparse_tpm = csc_matrix(tpm)
    return (sparse_tpm ** time_scale).toarray()


def dense_time(tpm, time_scale):
    return np.linalg.matrix_power(tpm, time_scale)


def run_tpm(tpm, time_scale):
    """Iterate a TPM by the specified number of time steps.

    Args:
        tpm (np.ndarray): A state-by-node tpm.
        time_scale (int): The number of steps to run the tpm.

    Returns:
        np.ndarray
    """
    sbs_tpm = convert.state_by_node2state_by_state(tpm)
    if sparse(tpm):
        tpm = sparse_time(sbs_tpm, time_scale)
    else:
        tpm = dense_time(sbs_tpm, time_scale)
    return convert.state_by_state2state_by_node(tpm)


def run_cm(cm, time_scale):
    """Iterate a connectivity matrix the specified number of steps.

    Args:
        cm (np.ndarray): A |N x N| connectivity matrix
        time_scale (int): The number of steps to run.

    Returns:
        np.ndarray
    """
    cm = np.linalg.matrix_power(cm, time_scale)
    # Round non-unitary values back to 1
    cm[cm > 1] = 1
    return cm


# TPM and Connectivity Matrix utils
# ============================================================================

def state_by_state(tpm):
    """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
    ``False``."""
    return tpm.ndim == 2 and tpm.shape[0] == tpm.shape[1]


def condition_tpm(tpm, fixed_nodes, state):
    """Return a TPM conditioned on the given fixed node indices, whose states
    are fixed according to the given state-tuple.

    The dimensions of the new TPM that correspond to the fixed nodes are
    collapsed onto their state, making those dimensions singletons suitable for
    broadcasting. The number of dimensions of the conditioned TPM will be the
    same as the unconditioned TPM.
    """
    conditioning_indices = [[slice(None)]] * len(state)
    for i in fixed_nodes:
        # Preserve singleton dimensions with `np.newaxis`
        conditioning_indices[i] = [state[i], np.newaxis]
    # Flatten the indices.
    conditioning_indices = list(chain.from_iterable(conditioning_indices))
    # Obtain the actual conditioned TPM by indexing with the conditioning
    # indices.
    return tpm[conditioning_indices]


def expand_tpm(tpm):
    """Broadcast a state-by-node TPM so that singleton dimensions are expanded
    over the full network."""
    uc = np.ones([2] * (tpm.ndim - 1) + [tpm.shape[-1]])
    return tpm * uc


def fully_connected(cm, nodes1, nodes2):
    """Test connectivity of one set of nodes to another.

    Args:
        cm (``np.ndarrray``): The connectivity matrix
        nodes1 (tuple[int]): The nodes whose outputs to ``nodes2`` will be
            tested.
        nodes2 (tuple[int]): The nodes whose inputs from ``nodes1`` will
            be tested.

    Returns:
        bool: Returns ``True`` if all elements in ``nodes1`` output to some
        element in ``nodes2`` AND all elements in ``nodes2`` have an input from
        some element in ``nodes1``. Otherwise return ``False``. Return ``True``
        if either set of nodes is empty.
    """
    if not nodes1 or not nodes2:
        return True

    cm = cm[np.ix_(nodes1, nodes2)]

    # Do all nodes have at least one connection?
    return cm.sum(0).all() and cm.sum(1).all()


def apply_boundary_conditions_to_cm(external_indices, cm):
    """Return a connectivity matrix with all connections to or from external
    nodes removed.
    """
    cm = cm.copy()
    for i in external_indices:
        # Zero-out row
        cm[i] = 0
        # Zero-out column
        cm[:, i] = 0
    return cm


def get_inputs_from_cm(index, cm):
    """Return a tuple of node indices that have connections to the node with
    the given index.
    """
    return tuple(i for i in range(cm.shape[0]) if cm[i][index])


def get_outputs_from_cm(index, cm):
    """Return a tuple of node indices that the node with the given index has
    connections to.
    """
    return tuple(i for i in range(cm.shape[0]) if cm[index][i])


def causally_significant_nodes(cm):
    """Return a tuple of all nodes indices in the connectivity matrix which
    are causally significant (have inputs and outputs)."""
    inputs = cm.sum(0)
    outputs = cm.sum(1)
    nodes_with_inputs_and_outputs = np.logical_and(inputs > 0, outputs > 0)
    return tuple(np.where(nodes_with_inputs_and_outputs)[0])


def np_hash(a):
    """Return a hash of a NumPy array."""
    if a is None:
        return hash(None)
    # Ensure that hashes are equal whatever the ordering in memory (C or
    # Fortran)
    a = np.ascontiguousarray(a)
    # Compute the digest and return a decimal int
    return int(hashlib.sha1(a.view(a.dtype)).hexdigest(), 16)


def phi_eq(x, y):
    """Compare two phi values up to |PRECISION|."""
    return abs(x - y) <= constants.EPSILON


def normalize(a):
    """Normalize a distribution.

    Args:
        a (np.ndarray): The array to normalize.

    Returns:
        np.ndarray: ``a`` normalized so that the sum of its entries is 1.
    """
    sum_a = a.sum()
    if sum_a == 0:
        return a
    return a / sum_a


# see http://stackoverflow.com/questions/16003217
def combs(a, r):
    """NumPy implementation of itertools.combinations.

    Return successive |r|-length combinations of elements in the array ``a``.

    Args:
        a (np.ndarray): The array from which to get combinations.
        r (int): The length of the combinations.

    Returns:
        np.ndarray: An array of combinations.
    """
    # Special-case for 0-length combinations
    if r == 0:
        return np.asarray([])

    a = np.asarray(a)
    data_type = a.dtype if r == 0 else np.dtype([('', a.dtype)] * r)
    b = np.fromiter(combinations(a, r), data_type)
    return b.view(a.dtype).reshape(-1, r)


# see http://stackoverflow.com/questions/16003217/
def comb_indices(n, k):
    """|N-D| version of itertools.combinations.

    Args:
        a (np.ndarray): The array from which to get combinations.
        k (int): The desired length of the combinations.

    Returns:
        np.ndarray: Indices that give the |k|-combinations of |n| elements.

    Example:
        >>> n, k = 3, 2
        >>> data = np.arange(6).reshape(2, 3)
        >>> data[:, comb_indices(n, k)]
        array([[[0, 1],
                [0, 2],
                [1, 2]],
        <BLANKLINE>
               [[3, 4],
                [3, 5],
                [4, 5]]])
    """
    # Count the number of combinations for preallocation
    count = comb(n, k, exact=True)
    # Get numpy iterable from ``itertools.combinations``
    indices = np.fromiter(
        chain.from_iterable(combinations(range(n), k)),
        int,
        count=(count * k))
    # Reshape output into the array of combination indicies
    return indices.reshape(-1, k)


# TODO? implement this with numpy
def powerset(iterable):
    """Return the power set of an iterable (see `itertools recipes
    <http://docs.python.org/2/library/itertools.html#recipes>`_).

    Args:
        iterable (Iterable): The iterable from which to generate the power set.

    Returns:
        generator: An chained generator over the power set.

    Example:
        >>> ps = powerset(np.arange(2))
        >>> print(list(ps))
        [(), (0,), (1,), (0, 1)]
    """
    return chain.from_iterable(combinations(iterable, r)
                               for r in range(len(iterable) + 1))


def uniform_distribution(number_of_nodes):
    """
    Return the uniform distribution for a set of binary nodes, indexed by state
    (so there is one dimension per node, the size of which is the number of
    possible states for that node).

    Args:
        nodes (np.ndarray): A set of indices of binary nodes.

    Returns:
        np.ndarray: The uniform distribution over the set of nodes.
    """
    # The size of the state space for binary nodes is 2^(number of nodes).
    number_of_states = 2 ** number_of_nodes
    # Generate the maximum entropy distribution
    # TODO extend to nonbinary nodes
    return (np.ones(number_of_states) /
            number_of_states).reshape([2] * number_of_nodes)


def marginalize_out(indices, tpm):
    """
    Marginalize out a node from a TPM.

    Args:
        indices (list[int]): The indices of nodes to be marginalized out.
        tpm (np.ndarray): The TPM to marginalize the node out of.

    Returns:
        np.ndarray: A TPM with the same number of dimensions, with the nodes
        marginalized out.
    """
    return tpm.sum(tuple(indices), keepdims=True) / (
        np.array(tpm.shape)[list(indices)].prod())


def marginal_zero(repertoire, node_index):
    """Return the marginal probability that the node is off."""
    index = [slice(None) for i in range(repertoire.ndim)]
    index[node_index] = 0

    return repertoire[index].sum()


def marginal(repertoire, node_index):
    """Get the marginal distribution for a node."""
    index = tuple(i for i in range(repertoire.ndim) if i != node_index)

    return repertoire.sum(index, keepdims=True)


def independent(repertoire):
    """Check whether the repertoire is independent."""
    marginals = [marginal(repertoire, i) for i in range(repertoire.ndim)]

    # TODO: is there a way to do without an explicit iteration?
    joint = marginals[0]
    for m in marginals[1:]:
        joint = joint * m

    # TODO: should we round here?
    #repertoire = repertoire.round(config.PRECISION)
    #joint = joint.round(config.PRECISION)

    return np.array_equal(repertoire, joint)


def purview(repertoire):
    """The purview of the repertoire.

    Args:
        repertoire (np.ndarray): A repertoire

    Returns:
        tuple[int]: The purview that the repertoire was computed over.
    """
    if repertoire is None:
        return None

    return tuple(np.where(np.array(repertoire.shape) == 2)[0])


def purview_size(repertoire):
    """Return the size of the purview of the repertoire.

    Args:
        repertoire (np.ndarray): A repertoire

    Returns:
        int: The size of purview that the repertoire was computed over.
    """
    return len(purview(repertoire))


def repertoire_shape(purview, N):
    """Return the shape a repertoire.

    Args:
        purview (tuple[int]): The purview over which the repertoire is
            computed.
        N (int): The number of elements in the system.

    Returns:
        list[int]: The shape of the repertoire. Purview nodes have two
        dimensions and non-purview nodes are collapsed to a unitary dimension.

    Example:
        >>> purview = (0, 2)
        >>> N = 3
        >>> repertoire_shape(purview, N)
        [2, 1, 2]
    """
    # TODO: extend to non-binary nodes
    return [2 if i in purview else 1 for i in range(N)]


@cache(cache={}, maxmem=None)
def max_entropy_distribution(node_indices, number_of_nodes):
    """Return the maximum entropy distribution over a set of nodes.

    This is different from the network's uniform distribution because nodes
    outside ``node_indices`` are fixed and treated as if they have only 1
    state.

    Args:
        node_indices (tuple[int]): The set of node indices over which to take
            the distribution.
        number_of_nodes (int): The total number of nodes in the network.

    Returns:
        np.ndarray: The maximum entropy distribution over the set of nodes.
    """
    distribution = np.ones(repertoire_shape(node_indices, number_of_nodes))

    return distribution / distribution.size


# TODO extend to binary nodes
def hamming_emd(d1, d2):
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node).

    Singleton dimensions are sqeezed out.
    """
    d1, d2 = d1.squeeze(), d2.squeeze()
    N = d1.ndim

    # Compute EMD using the Hamming distance between states as the
    # transportation cost function.
    return emd(d1.ravel(), d2.ravel(), _hamming_matrix(N))


def l1(d1, d2):
    """Return the L1 distance between two distributions.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.

    Returns:
        float: The sum of absolute differences of ``d1`` and ``d2``.
    """
    return np.absolute(d1 - d2).sum()


def kld(d1, d2):
    """Return the Kullback-Leibler Divergence (KLD) between two distributions.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.

    Returns:
        float: The KLD of ``d1`` from ``d2``.
    """
    d1, d2 = d1.squeeze().ravel(), d2.squeeze().ravel()
    return entropy(d1, d2, 2.0)


def ent(d1, d2):
    """Return the absolute difference in entropy between the two distributions.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.

    Returns:
        float: |H(d1) - H(d2)|.
    """
    d1, d2 = d1.squeeze().ravel(), d2.squeeze().ravel()
    return abs(entropy(d1, 2.0) - entropy(d2, 2.0))


def bipartition(a):
    """Return a list of bipartitions for a sequence.

    Args:
        a (Iterable): The iterable to partition.

    Returns:
        list[tuple[tuple]]: A list of tuples containing each of the two
        partitions.

    Example:
        >>> bipartition((1,2,3))
        [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,))]
    """
    return [(tuple(a[i] for i in part0_idx), tuple(a[j] for j in part1_idx))
            for part0_idx, part1_idx in bipartition_indices(len(a))]


# TODO? [optimization] optimize this to use indices rather than nodes
# TODO? are native lists really slower
def directed_bipartition(a):
    """Return a list of directed bipartitions for a sequence.

    Args:
        a (Iterable): The iterable to partition.

    Returns:
        list[tuple[tuple]]: A list of tuples containing each of the two
        partitions.

    Example:
        >>> directed_bipartition((1, 2, 3))  # doctest: +NORMALIZE_WHITESPACE
        [((), (1, 2, 3)),
         ((1,), (2, 3)),
         ((2,), (1, 3)),
         ((1, 2), (3,)),
         ((3,), (1, 2)),
         ((1, 3), (2,)),
         ((2, 3), (1,)),
         ((1, 2, 3), ())]
    """
    return [(tuple(a[i] for i in part0_idx), tuple(a[j] for j in part1_idx))
            for part0_idx, part1_idx in directed_bipartition_indices(len(a))]


def directed_bipartition_of_one(a):
    """Return a list of directed bipartitions for a sequence where each
    bipartitions includes a set of size 1.

    Args:
        a (Iterable): The iterable to partition.

    Returns:
        list[tuple[tuple]]: A list of tuples containing each of the two
        partitions.

    Example:
        >>> directed_bipartition_of_one((1,2,3))  # doctest: +NORMALIZE_WHITESPACE
        [((1,), (2, 3)),
         ((2,), (1, 3)),
         ((1, 2), (3,)),
         ((3,), (1, 2)),
         ((1, 3), (2,)),
         ((2, 3), (1,))]
    """
    return [partition for partition in directed_bipartition(a)
            if len(partition[0]) == 1 or len(partition[1]) == 1]


@cache(cache={}, maxmem=None)
def directed_bipartition_indices(N):
    """Return indices for directed bipartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        list: A list of tuples containing the indices for each of the two
        partitions.

    Example:
        >>> N = 3
        >>> directed_bipartition_indices(N)  # doctest: +NORMALIZE_WHITESPACE
        [((), (0, 1, 2)),
         ((0,), (1, 2)),
         ((1,), (0, 2)),
         ((0, 1), (2,)),
         ((2,), (0, 1)),
         ((0, 2), (1,)),
         ((1, 2), (0,)),
         ((0, 1, 2), ())]
    """
    indices = bipartition_indices(N)
    return indices + [idx[::-1] for idx in indices[::-1]]


@cache(cache={}, maxmem=None)
def bipartition_indices(N):
    """Return indices for undirected bipartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        list: A list of tuples containing the indices for each of the two
        partitions.

    Example:
        >>> N = 3
        >>> bipartition_indices(N)
        [((), (0, 1, 2)), ((0,), (1, 2)), ((1,), (0, 2)), ((0, 1), (2,))]
    """
    result = []
    if N <= 0:
        return result

    for i in range(2 ** (N-1)):
        part = [[], []]
        for n in range(N):
            bit = (i >> n) & 1
            part[bit].append(n)
        result.append((tuple(part[1]), tuple(part[0])))
    return result


@cache(cache={}, maxmem=None)
def directed_tripartition_indices(N):
    """Return indices for directed tripartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        list[tuple]: A list of tuples containing the indices for each partition.

    Example:
        >>> N = 1
        >>> directed_tripartition_indices(N)
        [((0,), (), ()), ((), (0,), ()), ((), (), (0,))]
    """

    result = []
    if N <= 0:
        return result

    base = [0, 1, 2]
    for key in itertools.product(base, repeat=N):
        part = [[], [], []]
        for i, location in enumerate(key):
            part[location].append(i)

        result.append(tuple(tuple(p) for p in part))

    return result


def directed_tripartition(seq):
    """Generator over all directed tripartitions of a sequence.

    Args:
        seq (Iterable): a sequence.

    Yields:
        tuple[tuple]: A tripartition of ``seq``.

    Example:
        >>> seq = (2, 5)
        >>> list(directed_tripartition(seq))  # doctest: +NORMALIZE_WHITESPACE
        [((2, 5), (), ()),
         ((2,), (5,), ()),
         ((2,), (), (5,)),
         ((5,), (2,), ()),
         ((), (2, 5), ()),
         ((), (2,), (5,)),
         ((5,), (), (2,)),
         ((), (5,), (2,)),
         ((), (), (2, 5))]
    """
    for a, b, c in directed_tripartition_indices(len(seq)):
        yield (tuple(seq[i] for i in a),
               tuple(seq[j] for j in b),
               tuple(seq[k] for k in c))


# Internal helper methods
# =============================================================================

def load_data(dir, num):
    """Load numpy data from the data directory.

    The files should stored in ``data/{dir}`` and named
    ``0.npy, 1.npy, ... {num - 1}.npy``.

    Returns:
        list: A list of loaded data, such that ``list[i]`` contains the the
        contents of ``i.npy``.
    """

    root = os.path.abspath(os.path.dirname(__file__))

    def get_path(i):
        return os.path.join(root, 'data', dir, str(i) + '.npy')

    return [np.load(get_path(i)) for i in range(num)]


# Load precomputed hamming matrices.
_NUM_PRECOMPUTED_HAMMING_MATRICES = 10
_hamming_matrices = load_data('hamming_matrices',
                              _NUM_PRECOMPUTED_HAMMING_MATRICES)


# TODO extend to nonbinary nodes
def _hamming_matrix(N):
    """Return a matrix of Hamming distances for the possible states of |N|
    binary nodes.

    Args:
        N (int): The number of nodes under consideration

    Returns:
        np.ndarray: A |2^N x 2^N| matrix where the |ith| element is the Hamming
        distance between state |i| and state |j|.

    Example:
        >>> _hamming_matrix(2)
        array([[ 0.,  1.,  1.,  2.],
               [ 1.,  0.,  2.,  1.],
               [ 1.,  2.,  0.,  1.],
               [ 2.,  1.,  1.,  0.]])
    """
    if N < _NUM_PRECOMPUTED_HAMMING_MATRICES:
        return _hamming_matrices[N]
    else:
        return _compute_hamming_matrix(N)


@constants.joblib_memory.cache
def _compute_hamming_matrix(N):
    """
    Compute and store a Hamming matrix for |N| nodes.

    Hamming matrices have the following sizes:

    n   MBs
    ==  ===
    9   2
    10  8
    11  32
    12  128
    13  512

    Given these sizes and the fact that large matrices are needed infrequently,
    we store computed matrices using the Joblib filesystem cache instead of
    adding computed matrices to the ``_hamming_matrices`` global and clogging
    up memory.

    This function is only called when N > _NUM_PRECOMPUTED_HAMMING_MATRICES.
    Don't call this function directly; use ``utils._hamming_matrix`` instead.
    """
    possible_states = np.array(list(all_states((N))))
    return cdist(possible_states, possible_states, 'hamming') * N


# TODO: better name?
def relevant_connections(n, _from, to):
    """Construct a connectivity matrix.

    Args:
        n (int): The dimensions of the matrix
        _from (tuple[int]): Nodes with outgoing connections to ``to``
        to (tuple[int]): Nodes with incoming connections from ``_from``

    Returns:
        np.ndarray: An |n x n| connectivity matrix with the |i,jth| entry set
        to ``1`` if |i| is in ``_from`` and |j| is in ``to``.
    """
    cm = np.zeros((n, n))

    # Don't try and index with empty arrays. Older versions of NumPy
    # (at least up to 1.9.3) break with empty array indices.
    if not _from or not to:
        return cm

    cm[np.ix_(_from, to)] = 1
    return cm


def block_cm(cm):
    """Return whether ``cm`` can be arranged as a block connectivity matrix.

    If so, the corresponding mechanism/purview is trivially reducible.
    Technically, only square matrices are "block diagonal", but the notion of
    connectivity carries over.

    We test for block connectivity by trying to grow a block of nodes such
    that:

    * 'source' nodes only input to nodes in the block
    * 'sink' nodes only receive inputs from source nodes in the block

    For example, the following connectivity matrix represents connections from
    ``nodes1 = A, B, C`` to ``nodes2 = D, E, F, G`` (without loss of
    generality—note that ``nodes1`` and ``nodes2`` may share elements)::

         D  E  F  G
      A [1, 1, 0, 0]
      B [1, 1, 0, 0]
      C [0, 0, 1, 1]

    Since nodes |AB| only connect to nodes |DE|, and node |C| only connects to
    nodes |FG|, the subgraph is reducible; the cut ::

      AB   C
      -- X --
      DE   FG

    does not change the structure of the graph.
    """
    if np.any(cm.sum(1) == 0):
        return True
    if np.all(cm.sum(1) == 1):
        return True

    outputs = list(range(cm.shape[1]))

    # CM helpers:
    def outputs_of(nodes):
        # All nodes that `nodes` connect to (output to)
        return np.where(cm[nodes, :].sum(0))[0]

    def inputs_to(nodes):
        # All nodes which connect to (input to) `nodes`
        return np.where(cm[:, nodes].sum(1))[0]

    # Start: source node with most outputs
    sources = [np.argmax(cm.sum(1))]
    sinks = outputs_of(sources)
    sink_inputs = inputs_to(sinks)

    while True:
        if np.all(sink_inputs == sources):
            # sources exclusively connect to sinks.
            # There are no other nodes which connect sink nodes,
            # hence set(sources) + set(sinks) form a component
            # which is not connected to the rest of the graph
            return True

        # Recompute sources, sinks, and sink_inputs
        sources = sink_inputs
        sinks = outputs_of(sources)
        sink_inputs = inputs_to(sinks)

        # Considering all output nodes?
        if np.all(sinks == outputs):
            return False


# TODO: simplify the conditional validation here and in block_cm
# TODO: combine with fully_connected
def block_reducible(cm, nodes1, nodes2):
    """Return whether connections from ``nodes1`` to ``nodes2`` are reducible.

    Args:
        cm (np.ndarray): The network's connectivity matrix.
        nodes1 (tuple[int]): Source nodes
        nodes2 (tuple[int]): Sink nodes
    """
    if not nodes1 or not nodes2:
        return True  # trivially

    cm = cm[np.ix_(nodes1, nodes2)]

    # Validate the connectivity matrix.
    if not cm.sum(0).all() or not cm.sum(1).all():
        return True
    if len(nodes1) > 1 and len(nodes2) > 1:
        return block_cm(cm)
    return False


def _connected(cm, nodes, connection):
    """Test connectivity for the connectivity matrix."""
    if nodes is not None:
        cm = cm[np.ix_(nodes, nodes)]

    num_components, _ = connected_components(cm, connection=connection)
    return num_components < 2


def strongly_connected(cm, nodes=None):
    """Return whether the connectivity matrix is strongly connected.

    Args:
        cm (np.ndarray): A square connectivity matrix.

    Keyword Args:
        nodes (tuple[int]): An optional subset of node indices to test strong
            connectivity over.
    """
    return _connected(cm, nodes, 'strong')


def weakly_connected(cm, nodes=None):
    """Return whether the connectivity matrix is weakly connected.

    Args:
        cm (np.ndarray): A square connectivity matrix.

    Keyword Args:
        nodes (tuple[int]): An optional subset of node indices to test weak
            connectivity over.
    """
    return _connected(cm, nodes, 'weak')


# Custom printing methods
# =============================================================================


def print_repertoire(r):
    """Print a vertical, human-readable cause/effect repertoire."""
    r = np.squeeze(r)
    print('\n', '-' * 80)
    for i in range(r.size):
        strindex = bin(i)[2:].zfill(r.ndim)
        index = tuple(map(int, list(strindex)))
        print('\n', strindex, '\t', r[index])
    print('\n', '-' * 80, '\n')


def print_repertoire_horiz(r):
    """Print a horizontal, human-readable cause/effect repertoire."""
    r = np.squeeze(r)
    colwidth = 11
    print('\n' + '-' * 70 + '\n')
    index_labels = [bin(i)[2:].zfill(r.ndim) for i in range(r.size)]
    indices = [tuple(map(int, list(s))) for s in index_labels]
    print('     p:  ', '|'.join('{0:.3f}'.format(r[index]).center(colwidth) for
                                index in indices))
    print('         ', '|'.join(' ' * colwidth for index in indices))
    print(' state:  ', '|'.join(label.center(colwidth) for label in
                                index_labels))
    print('\n' + '-' * 70 + '\n')
