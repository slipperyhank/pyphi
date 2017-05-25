# Module which contains all different approximations

from pyphi.models import Part, Bipartition


def full_cut(mechanism, purview):
    # A complete partition
    part0 = Part(mechanism, ())
    part1 = Part((), purview)
    return [Bipartition(part0, part1)]
