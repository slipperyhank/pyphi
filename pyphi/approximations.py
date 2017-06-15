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
        cut_purview_index = tuple(np.where(purview_intensities == k)[0][0]
                                  for k in purview_intensity)
        uncut_purview_index = tuple(index for index in purview
                                    if index not in cut_purview_index)
        if direction == past:
            mechanism_intensities = np.sum(cm[np.ix_(cut_purview_index,
                                                     mechanism)], 0)
        elif direction == future:
            mechanism_intensities = np.sum(cm[np.ix_(mechanism,
                                                     cut_purview_index)], 1)
        for mechanism_intensity in non_empty_powerset(set(mechanism_intensities[mechanism_intensities > 0])):
            cut_mechanism_index = tuple(np.where(mechanism_intensities == k)[0][0]
                                        for k in mechanism_intensity)
            uncut_mechanism_index = tuple(index for index in mechanism
                                          if index not in cut_mechanism_index)
            yield Bipartition(Part(cut_mechanism_index, cut_purview_index),
                              Part(uncut_mechanism_index, uncut_purview_index))
