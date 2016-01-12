#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ac_models.py

"""
Containers for AcMip and AcMice.
"""

from collections import Iterable, namedtuple

import numpy as np

from . import utils
from .jsonify import jsonify
from .models import _numpy_aware_eq, make_repr, indent, fmt_constellation, fmt_partition
from .ac_utils import ap_phi_eq

# TODO use properties to avoid data duplication

# Ac_diff-ordering methods
# =============================================================================

# Compare ac_diff
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Todo: check how that works out also with abs values
def _ap_phi_eq(self, other):
    try:
        return ap_phi_eq(self.ap_phi, other.ap_phi)
    except AttributeError:
        return False


def _ap_phi_lt(self, other):
    try:
        if not ap_phi_eq(self.ap_phi, other.ap_phi):
            return self.ap_phi < other.ap_phi
        return False
    except AttributeError:
        return False


def _ap_phi_gt(self, other):
    try:
        if not ap_phi_eq(self.ap_phi, other.ap_phi):
            return self.ap_phi > other.ap_phi
        return False
    except AttributeError:
        return False


def _ap_phi_le(self, other):
    return _ap_phi_lt(self, other) or _ap_phi_eq(self, other)


def _ap_phi_ge(self, other):
    return _ap_phi_gt(self, other) or _ap_phi_eq(self, other)


# First compare ap_phi, then mechanism size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _ap_phi_then_mechanism_size_lt(self, other):
    if _ap_phi_eq(self, other):
        return (len(self.mechanism) < len(other.mechanism)
                if hasattr(other, 'mechanism') else False)
    else:
        return _ap_phi_lt(self, other)


def _ap_phi_then_mechanism_size_gt(self, other):
    return (not _ap_phi_then_mechanism_size_lt(self, other) and
            not self == other)


def _ap_phi_then_mechanism_size_le(self, other):
    return (_ap_phi_then_mechanism_size_lt(self, other) or
            _ap_phi_eq(self, other))


def _ap_phi_then_mechanism_size_ge(self, other):
    return (_ap_phi_then_mechanism_size_gt(self, other) or
            _ap_phi_eq(self, other))


# Equality helpers
# =============================================================================
def _general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'ap_phi'``, it is compared up to |PRECISION|. All
    other attributes are compared with :func:`_numpy_aware_eq`.

    If an attribute is called ``'mechanism'`` or ``'purview'``, it is compared
    using set equality."""
    try:
        for attr in attributes:
            _a, _b = getattr(a, attr), getattr(b, attr)
            if attr == 'ap_phi':
                if not ap_phi_eq(_a, _b):
                    return False
            elif (attr == 'mechanism' or attr == 'purview'):
                if _a is None or _b is None and not _a == _b:
                    return False
                # Don't use `set` because hashes may be different (contexts are
                # included in node hashes); we want to use Node.__eq__.
                elif not (all(n in _b for n in _a) and len(_a) == len(_b)):
                    return False
            else:
                if not _numpy_aware_eq(_a, _b):
                    return False
        return True
    except AttributeError:
        return False

# =============================================================================
# Todo: Why do we even need this?
# Todo: add second state
_acmip_attributes = ['ap_phi','second_state','direction', 'mechanism', 'purview', 'partition',
                   'unpartitioned_ap', 'partitioned_ap']
_acmip_attributes_for_eq = ['ap_phi', 'direction', 'mechanism',
                          'unpartitioned_ap']


class AcMip(namedtuple('AcMip', _acmip_attributes)):

    """A minimum information partition for ac_coef calculation.

    MIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``ap_phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        ap_phi (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned actual probability.
        second_state (tuple(int)): 
            actual state of system in specified direction (past, future)
        direction (str):
            The temporal direction specifiying whether this AcMIP should be
            calculated with cause or effect repertoires.
        mechanism (tuple(int)):
            The mechanism over which to evaluate the AcMIP.
        purview (tuple(int)):
            The purview over which the unpartitioned actual probability differs the
            least from the actual probability of the partition.
        partition (tuple(Part, Part)):
            The partition that makes the least difference to the mechanism's
            repertoire.
        unpartitioned_ap (float):
            The actual probability of the unpartitioned mechanism.
        partitioned_ap (float):
            The actual probability of the partitioned mechanism from the product of
            the repertoires of each part of the partition.
    """
    __slots__ = ()

    def __eq__(self, other):
        # We don't count the partition and partitioned repertoire in checking
        # for MIP equality, since these are lost during normalization. We also
        # don't count the mechanism and purview, since these may be different
        # depending on the order in which purviews were evaluated.
        # TODO!!! clarify the reason for that
        # We do however check whether the size of the mechanism or purview is
        # the same, since that matters (for the exclusion principle).
        #TODO: include 2nd state here?
        if not self.purview or not other.purview:
            return (_general_eq(self, other, _acmip_attributes_for_eq) and
                    len(self.mechanism) == len(other.mechanism))
        else:
            return (_general_eq(self, other, _acmip_attributes_for_eq) and
                    len(self.mechanism) == len(other.mechanism) and
                    len(self.purview) == len(other.purview))

    def __bool__(self):
        """An AcMip is truthy if it is not reducible; i.e. if it has a significant
        amount of |ap_phi|."""
        return not ap_phi_eq(self.ap_phi, 0)

    # def __hash__(self):
    #     return hash((self.ap_phi, self.second_state, self.direction, self.mechanism, self.purview,
    #                  utils.np_hash(self.unpartitioned_ap)))

    def to_json(self):
        d = self.__dict__
        return d

    def __repr__(self):
        return make_repr(self, _acmip_attributes)

    def __str__(self):
        return "Mip\n" + indent(fmt_ac_mip(self))

    # Order by ap_phi value, then by mechanism size
    __lt__ = _ap_phi_then_mechanism_size_lt
    __gt__ = _ap_phi_then_mechanism_size_gt
    __le__ = _ap_phi_then_mechanism_size_le
    __ge__ = _ap_phi_then_mechanism_size_ge


# # =============================================================================

class AcMice:

    """A maximally irreducible actual cause or effect (i.e., "actual cause” or “actual
    effect”).

    relevant_connections (np.array):
        An ``N x N`` matrix, where ``N`` is the number of nodes in this
        corresponding subsystem, that identifies connections that “matter” to
        this AcMICE.

        ``direction == 'past'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            cause purview and node ``j`` is in the mechanism (and ``0``
            otherwise).

        ``direction == 'future'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            mechanism and node ``j`` is in the effect purview (and ``0``
            otherwise).

    AcMICEs may be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, ``phi`` values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared (exclusion
    principle).
    """

    def __init__(self, acmip, relevant_connections=None):
        self._acmip = acmip
        self._relevant_connections = relevant_connections

    @property
    def ap_phi(self):
        """
        ``float`` -- The difference between the mechanism's unpartitioned and
        partitioned actual probabilities.
        """
        return self._acmip.ap_phi

    @property
    def second_state(self):
        """
        ``tuple`` -- The actual state of system in specified direction (past, future).
        """
        return self._acmip.second_state    

    @property
    def direction(self):
        """
        ``str`` -- Either 'past' or 'future'. If 'past' ('future'), this
        represents a maximally irreducible cause (effect).
        """
        return self._acmip.direction

    @property
    def mechanism(self):
        """
        ``list(int)`` -- The mechanism for which the AcMICE is evaluated.
        """
        return self._acmip.mechanism

    @property
    def purview(self):
        """
        ``list(int)`` -- The purview over which this mechanism's |ap_phi|
        is maximal.
        """
        return self._acmip.purview

    @property
    def ap_phi(self):
        """
        ``np.ndarray`` -- The unpartitioned repertoire of the mechanism over
        the purview.
        """
        return self._acmip.ap_phi

    @property
    def acmip(self):
        """
        ``AcMip`` -- The minimum information partition for this mechanism.
        """
        return self._acmip

    def __repr__(self):
        return make_repr(self, ['acmip'])

    def __str__(self):
        return "AcMice\n" + indent(fmt_ac_mip(self.acmip))

    def __eq__(self, other):
        return self.acmip == other.acmip

    def __hash__(self):
        return hash(('AcMice', self._acmip))

    def __bool__(self):
        """An AcMice is truthy if it is not reducible; i.e. if it has a
        significant amount of |ap_phi|."""
        return not utils.phi_eq(self._acmip.ap_phi, 0)    

    def to_json(self):
        return {'acmip': self._acmip}

    # Order by ap_phi value, then by mechanism size
    __lt__ = _ap_phi_then_mechanism_size_lt
    __gt__ = _ap_phi_then_mechanism_size_gt
    __le__ = _ap_phi_then_mechanism_size_le
    __ge__ = _ap_phi_then_mechanism_size_ge


# =============================================================================

_acbigmip_attributes = ['ap_phi', 'second_state', 'unpartitioned_constellation',
                      'partitioned_constellation', 'subsystem',
                      'cut_subsystem']


class AcBigMip:

    """A minimum information partition for |big_ap_phi| calculation.

    BigMIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``ac_diff`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).
    Todo: Check if we do the same, i.e. take the bigger system, or take the smaller?

    Attributes:
        ap_phi (float): The |big_ap_phi| value for the subsystem when taken against
            this MIP, *i.e.* the difference between the unpartitioned
            constellation and this MIP's partitioned constellation.
        unpartitioned_constellation (tuple(Concept)): The constellation of the
            whole subsystem.
        partitioned_constellation (tuple(Concept)): The constellation when the
            subsystem is cut.
        subsystem (Subsystem): The subsystem this MIP was calculated for.
        cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
    """

    def __init__(self, ap_phi=None, second_state=None, unpartitioned_constellation=None,
                 partitioned_constellation=None, subsystem=None,
                 cut_subsystem=None):
        self.ap_phi = ap_phi
        self.second_state = second_state
        self.unpartitioned_constellation = unpartitioned_constellation
        self.partitioned_constellation = partitioned_constellation
        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem
        
    def __repr__(self):
        return make_repr(self, _acbigmip_attributes)

    def __str__(self):
        return "\nAcBigMip\n======\n" + fmt_ac_big_mip(self)

    @property
    def cut(self):
        """The unidirectional cut that makes the least difference to the
        subsystem."""
        return self.cut_subsystem.cut

    def __eq__(self, other):
        return _general_eq(self, other, _acbigmip_attributes)

    def __bool__(self):
        """A BigMip is truthy if it is not reducible; i.e. if it has a
        significant amount of |big_ap_phi|."""
        return not ap_phi_eq(self.ap_phi, 0)

    def __hash__(self):
        return hash((self.ap_phi, self.second_state, self.unpartitioned_constellation,
                     self.partitioned_constellation, self.subsystem,
                     self.cut_subsystem))

    # First compare ap_phi (same comparison as ap_phi), then subsystem size
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __lt__(self, other):
        if _ap_phi_eq(self, other):
            if len(self.subsystem) == len(other.subsystem):
                return False
            else:
                return len(self.subsystem) < len(other.subsystem)
        else:
            return _ap_phi_lt(self, other)

    def __gt__(self, other):
        if _ap_phi_eq(self, other):
            if len(self.subsystem) == len(other.subsystem):
                return False
            else:
                return len(self.subsystem) > len(other.subsystem)
        else:
            return _ap_phi_gt(self, other)

    def __le__(self, other):
        return (self.__lt__(other) or _ap_phi_eq(self, other))

    def __ge__(self, other):
        return (self.__gt__(other) or _ap_phi_eq(self, other))

# Formatting functions for __str__ and __repr__
# TODO: probably move this to utils.py, or maybe fmt.py??

def fmt_ac_mip(acmip, verbose=True):
    """Helper function to format a nice Mip string"""

    if acmip is False or acmip is None:  # mips can be Falsy
        return ""

    mechanism = "mechanism: {}\t".format(acmip.mechanism) if verbose else ""
    direction = "direction: {}\n".format(acmip.direction) if verbose else ""
    return (
        "ap_phi: {acmip.ap_phi}\t"
        "{mechanism}"
        "purview: {acmip.purview}\t"
        "{direction}"
        "partition:\n{partition}\n"
        "unpartitioned_ap:\t{unpart_rep}\t"
        "partitioned_ap:\t{part_rep}\n").format(
            mechanism=mechanism,
            direction=direction,
            acmip=acmip,
            partition=indent(fmt_partition(acmip.partition)),
            unpart_rep=indent(acmip.unpartitioned_ap),
            part_rep=indent(acmip.partitioned_ap))

def fmt_ac_big_mip(ac_big_mip):
    """Format a AcBigMip"""
    return (
        "ap_phi: {ac_big_mip.ap_phi}\n"
        "second_state: {ac_big_mip.second_state}\n"
        "subsystem: {ac_big_mip.subsystem}\n"
        "cut: {ac_big_mip.cut}\n"
        "unpartitioned_constellation: {unpart_const}"
        "partitioned_constellation: {part_const}".format(
            ac_big_mip=ac_big_mip,
            unpart_const=fmt_constellation(ac_big_mip.unpartitioned_constellation),
            part_const=fmt_constellation(ac_big_mip.partitioned_constellation)))