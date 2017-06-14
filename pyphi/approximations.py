# Module which contains all different approximations

from pyphi.models import Part, Bipartition


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
    return [Bipartition(part0, part1)]


def one_mechanism(mechanism, purview):
    return [Bipartition(Part((mechanism[i],), ()),
                        Part(mechanism[:i] + mechanism[(i + 1):], purview))
            for i in range(len(mechanism))]


def one_slice(mechanism, purview):
    if len(mechanism) == 2:
        return [Bipartition(Part((mechanism[0],), (purview[j],)),
                            Part((mechanism[1],),
                                 purview[:j] + purview[(j + 1):]))
                for j in range(len(purview))]
    return [Bipartition(Part((mechanism[i],), (purview[j],)),
                        Part(mechanism[:i] + mechanism[(i + 1):],
                             purview[:j] + purview[(j + 1):]))
            for i in range(len(mechanism))
            for j in range(len(purview))]
