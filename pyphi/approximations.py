# Module which contains all different approximations

from pyphi.models import Part, Bipartition
from pyphi import utils
from itertools import product


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
