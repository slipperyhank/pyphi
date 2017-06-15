# Module which contains all different approximations

from pyphi.models import Part, Bipartition
from pyphi import utils
from itertools import product
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
