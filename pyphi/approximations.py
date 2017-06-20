# Module which contains all different approximations

from pyphi.models import Part, Bipartition, KPartition, Tripartition
from pyphi import utils, config, validate
from itertools import product, permutations
import numpy as np
from pyphi.constants import Direction
past, future = Direction.PAST, Direction.FUTURE


def full_cut(mechanism, purview):
    # The complete partition that cuts all mechanism elements from all purview
    # elements. Essentially returning cause or effect information.
    #
    # Args:
    #   mechanism (tuple(int)): set of mechanism elemment indices
    #   purview (tuple(int)): set of purview element indices
    #
    # Return:
    #   list(kPartition): Partitions to be evaluated by the approximation
    part0 = Part(mechanism, ())
    part1 = Part((), purview)
    yield Bipartition(part0, part1)


def one_mechanism(mechanism, purview):
    return [Bipartition(Part((mechanism[i],), ()),
                        Part(mechanism[:i] + mechanism[(i + 1):], purview))
            for i in range(len(mechanism))]


def one_slice(mechanism, purview):
    if len(mechanism) == 2:
        for j in range(len(purview)):
            yield Bipartition(Part((mechanism[0],), (purview[j],)),
                              Part((mechanism[1],),
                                   purview[:j] + purview[(j + 1):]))
    else:
        for i, j in product(range(len(mechanism)), range(len(purview))):
            yield Bipartition(Part((mechanism[i],), (purview[j],)),
                              Part(mechanism[:i] + mechanism[(i + 1):],
                                   purview[:j] + purview[(j + 1):]))


def bipartitions(mechanism, purview):
    # All possible bipartitions of the mechanism such that each purview element
    # is assigned to exactly one part (no wedge).
    yield Bipartition(Part(mechanism, ()), Part((), purview))
    numerators = utils.bipartition(mechanism)
    denominators = utils.directed_bipartition(purview)
    for n, d in product(numerators, denominators):
        if (n[0] and n[1]):
            yield Bipartition(Part(n[0], d[0]), Part(n[1], d[1]))


def intensity_based(mechanism, purview, cm, direction):
    # Consider a set of cuts based on 'intensities' (number of inputs or
    # outputs) of purview and mechanism elements. Based on an assumption that
    # elements and mechanisms are indistinguishable aside from their
    # connectivity.
    def non_empty_powerset(collection):
        for subset in utils.powerset(collection):
            if subset:
                yield subset
    if direction == past:
        purview_intensities = np.sum(cm[np.ix_(purview, mechanism)], 1)
    elif direction == future:
        purview_intensities = np.sum(cm[np.ix_(mechanism, purview)], 0)
    for purview_intensity in non_empty_powerset(set(purview_intensities)):
        purview_index = [np.where(purview_intensities == k)[0][0]
                         for k in purview_intensity]
        cut_purview = tuple(purview[index] for index in purview_index)
        uncut_purview = tuple(index for index in purview
                              if index not in cut_purview)
        if direction == past:
            mechanism_intensities = np.sum(cm[np.ix_(cut_purview,
                                                     mechanism)], 0)
        elif direction == future:
            mechanism_intensities = np.sum(cm[np.ix_(mechanism,
                                                     cut_purview)], 1)
        for mechanism_intensity in non_empty_powerset(set(mechanism_intensities[mechanism_intensities > 0])):
            mechanism_index = [np.where(mechanism_intensities == k)[0][0]
                               for k in mechanism_intensity]
            cut_mechanism = tuple(mechanism[index] for index in mechanism_index)
            uncut_mechanism = tuple(index for index in mechanism
                                    if index not in cut_mechanism)
            yield Bipartition(Part(cut_mechanism, cut_purview),
                              Part(uncut_mechanism, uncut_purview))


def all_mechanism(mechanism, purview, cm, direction):
    # Try all possible bipartitions of the mechanism, and then the one purview
    # partition that minimizes cut connections.
    if direction == past:
        cm = np.transpose(cm)
    mechanism_partitions = utils.bipartition(mechanism)[1:]
    for partition in mechanism_partitions:
        cut_purview = tuple(index for index in purview
                            if (sum(cm[partition[1], index]) >
                                sum(cm[partition[0], index])))
        uncut_purview = tuple(element for element in purview
                              if element not in cut_purview)
        yield Bipartition(Part(partition[0], uncut_purview),
                          Part(partition[1], cut_purview))


def prewedge(mechanism, purview, subsystem, direction):
    # Test for information in each purview element to determine if wedge is
    # desired
    if direction == past:
        info = subsystem.cause_info
    elif direction == future:
        info = subsystem.effect_info
    else:
        validate.direction(direction)
    wedged = tuple(element for element in purview
                   if info(mechanism, (element,)) == 0)
    if len(wedged) == len(purview):
        yield Bipartition(Part(mechanism, ()), Part((), purview))
    else:
        cm = subsystem.connectivity_matrix
        new_purview = tuple(element for element in purview
                            if element not in wedged)
        partitions = get_partitions(mechanism, new_purview, cm, direction)
        for partition in partitions:
            mechanism_parts = [part[0] for part in partition] + [()]
            purview_parts = [part[1] for part in partition] + [wedged]
            yield KPartition(
                *(Part(mechanism_parts[i], purview_parts[i])
                  for i in range(len(mechanism_parts))))


def get_partitions(mechanism, purview, cm, direction):
    # Get all partitions corresponding to the PARTITION_TYPE
    if config.PARTITION_TYPE == 'IIT3':
        partitions = mip_bipartitions(mechanism, purview)
    elif config.PARTITION_TYPE == 'BI_W':
        partitions = wedge_partitions(mechanism, purview)
    elif config.PARTITION_TYPE == 'ALL':
        partitions = all_partitions(mechanism, purview)
    elif config.PARTITION_TYPE == 'FULL':
        partitions = full_cut(mechanism, purview)
    elif config.PARTITION_TYPE == 'ONE_MECHANISM':
        partitions = one_mechanism(mechanism, purview)
    elif config.PARTITION_TYPE == 'ONE_SLICE':
        partitions = one_slice(mechanism, purview)
    elif config.PARTITION_TYPE == 'BI':
        partitions = bipartitions(mechanism, purview)
    elif config.PARTITION_TYPE == 'INTENSITY':
        partitions = intensity_based(mechanism, purview, cm, direction)
    elif config.PARTITION_TYPE == 'ALL_MECHANISM':
        partitions = all_mechanism(mechanism, purview, cm, direction)
    else:
        partitions = None
    return partitions


def all_partitions(m, p):
    yield KPartition(Part(m, ()), Part((), p))
    m = list(m)
    p = list(p)
    mechanism_partitions = partitions(m)
    for mechanism_partition in mechanism_partitions:
        if len(mechanism_partition) > 1:
            mechanism_partition.append([])
            n_mechanism_parts = len(mechanism_partition)
            max_purview_partition = min(len(p), n_mechanism_parts)
            for n_purview_parts in range(1, max_purview_partition + 1):
                purview_partitions = k_partitions(p, n_purview_parts)
                n_empty = n_mechanism_parts - n_purview_parts
                for purview_partition in purview_partitions:
                    purview_partition = [tuple(_list)
                                         for _list in purview_partition]
                    # Extend with empty tuples so purview partition has same size
                    # as mechanism purview
                    purview_partition.extend([() for j in range(n_empty)])
                    # Unique permutations to avoid duplicates empties
                    for permutation in set(permutations(purview_partition)):
                        yield KPartition(
                            *(Part(tuple(mechanism_partition[i]), tuple(permutation[i]))
                              for i in range(n_mechanism_parts)))


def k_partitions(collection, k):
    # Algorithm for generating k-partitions of a collection
    # codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions
    def visit(n, a):
        ps = [[] for i in range(k)]
        for j in range(n):
            ps[a[j + 1]].append(collection[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - a, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
    if k == 1:
        return ([[[item for item in collection]]])
    else:
        n = len(collection)
        a = [0] * (n + 1)
        for j in range(1, k + 1):
            a[n - k + j] = j - 1
        return f(k, n, 0, n, a)


def wedge_partitions(mechanism, purview):
    """Return an iterator over all wedge partitions.

    These are partitions which strictly split the mechanism and allow a subset
    of the purview to be split into a third partition, eg::

        A    B   []
        -- X - X --
        B    C   D

    See ``pyphi.config.PARTITION_TYPE`` for more information.

    Args:
        mechanism (tuple[int]): A mechanism.
        purview (tuple[int]): A purview.

    Yields:
        Tripartition: all unique tripartitions of this mechanism and purview.
    """

    numerators = utils.bipartition(mechanism)
    denominators = utils.directed_tripartition(purview)

    yielded = set()

    for n, d in product(numerators, denominators):
        if ((n[0] or d[0]) and (n[1] or d[1]) and
            ((n[0] and n[1]) or not d[0] or not d[1])):

            # Normalize order of parts to remove duplicates.
            tripart = Tripartition(*sorted((
                Part(n[0], d[0]),
                Part(n[1], d[1]),
                Part((),   d[2]))))

            def nonempty(part):
                return part.mechanism or part.purview

            # Check if the tripartition can be transformed into a causally
            # equivalent partition by combing two of its parts; eg.
            # A/[] x B/[] x []/CD is equivalent to AB/[] x []/CD so we don't
            # include it.
            def compressible(tripart):
                pairs = [
                    (tripart[0], tripart[1]),
                    (tripart[0], tripart[2]),
                    (tripart[1], tripart[2])]

                for x, y in pairs:
                    if (nonempty(x) and nonempty(y) and
                        (x.mechanism + y.mechanism == () or
                         x.purview + y.purview == ())):
                        return True

            if not compressible(tripart) and tripart not in yielded:
                yielded.add(tripart)
                yield tripart


def mip_bipartitions(mechanism, purview):
    """Return an generator of all |small_phi| bipartitions of a mechanism over
    a purview.

    Excludes all bipartitions where one half is entirely empty, e.g::

         A    []
        --- X --
         B    []

    is not valid, but ::

        A    []
        -- X --
        []   B

    is.

    Args:
        mechanism (tuple[int]): The mechanism to partition
        purview (tuple[int]): The purview to partition

    Yields:
        Bipartition: Where each bipartition is

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

    for n, d in product(numerators, denominators):
        if (n[0] or d[0]) and (n[1] or d[1]):
            yield Bipartition(Part(n[0], d[0]), Part(n[1], d[1]))


def partitions(collection):
    # all possible partitions
    # stackoverflow.com/questions/19368375/set-partitions-in-python
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller

