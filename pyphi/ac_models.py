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
from .models import _numpy_aware_eq
from .ac_utils import ap_diff_eq

# TODO use properties to avoid data duplication

# Ac_diff-ordering methods
# =============================================================================

# Compare ac_diff
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Todo: check how that works out also with abs values
def _ap_diff_eq(self, other):
    try:
        return ap_diff_eq(self.ap_diff, other.ap_diff)
    except AttributeError:
        return False


def _ap_diff_lt(self, other):
    try:
        if not ap_diff_eq(self.ap_diff, other.ap_diff):
            return abs(self.ap_diff) < abs(other.ap_diff)
        return False
    except AttributeError:
        return False


def _ap_diff_gt(self, other):
    try:
        if not ap_diff_eq(self.ap_diff, other.ap_diff):
            return abs(self.ap_diff) > abs(other.ap_diff)
        return False
    except AttributeError:
        return False


def _ap_diff_le(self, other):
    return _ap_diff_lt(self, other) or _ap_diff_eq(self, other)


def _ap_diff_ge(self, other):
    return _ap_diff_gt(self, other) or _ap_diff_eq(self, other)


# First compare ap_diff, then mechanism size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _ap_diff_then_mechanism_size_lt(self, other):
    if _ap_diff_eq(self, other):
        return (len(self.mechanism) < len(other.mechanism)
                if hasattr(other, 'mechanism') else False)
    else:
        return _ap_diff_lt(self, other)


def _ap_diff_then_mechanism_size_gt(self, other):
    return (not _ap_diff_then_mechanism_size_lt(self, other) and
            not self == other)


def _ap_diff_then_mechanism_size_le(self, other):
    return (_ap_diff_then_mechanism_size_lt(self, other) or
            _ap_diff_eq(self, other))


def _ap_diff_then_mechanism_size_ge(self, other):
    return (_ap_diff_then_mechanism_size_gt(self, other) or
            _ap_diff_eq(self, other))


# Equality helpers
# =============================================================================
def _general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'ap_diff'``, it is compared up to |PRECISION|. All
    other attributes are compared with :func:`_numpy_aware_eq`.

    If an attribute is called ``'mechanism'`` or ``'purview'``, it is compared
    using set equality."""
    try:
        for attr in attributes:
            _a, _b = getattr(a, attr), getattr(b, attr)
            if attr == 'ap_diff':
                if not ap_diff_eq(_a, _b):
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
_acmip_attributes = ['ap_diff','second_state','direction', 'mechanism', 'purview', 'partition',
                   'unpartitioned_ap', 'partitioned_ap']
_acmip_attributes_for_eq = ['ap_diff', 'direction', 'mechanism',
                          'unpartitioned_ap']


class AcMip(namedtuple('AcMip', _acmip_attributes)):

    """A minimum information partition for ac_coef calculation.

    MIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``ap_diff`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        ap_diff (float):
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
        amount of |ap_diff|."""
        return not ap_diff_eq(self.ap_diff, 0)

    # def __hash__(self):
    #     return hash((self.ap_diff, self.second_state, self.direction, self.mechanism, self.purview,
    #                  utils.np_hash(self.unpartitioned_ap)))

    def to_json(self):
        d = self.__dict__
        return d

    # Order by ap_diff value, then by mechanism size
    __lt__ = _ap_diff_then_mechanism_size_lt
    __gt__ = _ap_diff_then_mechanism_size_gt
    __le__ = _ap_diff_then_mechanism_size_le
    __ge__ = _ap_diff_then_mechanism_size_ge


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
    def ap_diff(self):
        """
        ``float`` -- The difference between the mechanism's unpartitioned and
        partitioned actual probabilities.
        """
        return self._acmip.ap_diff

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
        ``list(int)`` -- The purview over which this mechanism's |ap_diff|
        is maximal.
        """
        return self._acmip.purview

    @property
    def ap_diff(self):
        """
        ``np.ndarray`` -- The unpartitioned repertoire of the mechanism over
        the purview.
        """
        return self._acmip.ap_diff

    @property
    def acmip(self):
        """
        ``AcMip`` -- The minimum information partition for this mechanism.
        """
        return self._acmip

    def __str__(self):
        return "AcMice(" + str(self._acmip) + ")"

    def __repr__(self):
        return "AcMice(" + repr(self._acmip) + ")"

    def __eq__(self, other):
        return self.acmip == other.acmip

    def __hash__(self):
        return hash(('AcMice', self._acmip))

    def __bool__(self):
        """An AcMice is truthy if it is not reducible; i.e. if it has a
        significant amount of |ap_diff|."""
        return not utils.phi_eq(self._acmip.ap_diff, 0)    

    def to_json(self):
        return {'acmip': self._acmip}

    # Order by ap_diff value, then by mechanism size
    __lt__ = _ap_diff_then_mechanism_size_lt
    __gt__ = _ap_diff_then_mechanism_size_gt
    __le__ = _ap_diff_then_mechanism_size_le
    __ge__ = _ap_diff_then_mechanism_size_ge


# # =============================================================================

# _concept_attributes = ['phi', 'mechanism', 'cause', 'effect', 'subsystem',
#                        'normalized']


# # TODO: make mechanism a property
# # TODO: make phi a property
# class Concept:

#     """A star in concept-space.

#     The ``phi`` attribute is the |small_phi_max| value. ``cause`` and
#     ``effect`` are the MICE objects for the past and future, respectively.

#     Concepts may be compared with the built-in Python comparison operators
#     (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
#     are equal up to |PRECISION|, the size of the mechanism is compared.

#     Attributes:
#         phi (float):
#             The size of the concept. This is the minimum of the |small_phi|
#             values of the concept's core cause and core effect.
#         mechanism (tuple(int)):
#             The mechanism that the concept consists of.
#         cause (|Mice|):
#             The |Mice| representing the core cause of this concept.
#         effect (|Mice|):
#             The |Mice| representing the core effect of this concept.
#         subsystem (Subsystem):
#             This concept's parent subsystem.
#         time (float):
#             The number of seconds it took to calculate.
#     """

#     def __init__(self, phi=None, mechanism=None, cause=None, effect=None,
#                  subsystem=None, normalized=False):
#         self.phi = phi
#         self.mechanism = mechanism
#         self.cause = cause
#         self.effect = effect
#         self.subsystem = subsystem
#         self.normalized = normalized
#         self.time = None

#     def __repr__(self):
#         return 'Concept(' + ', '.join(attr + '=' + str(getattr(self, attr)) for
#                                       attr in _concept_attributes) + ')'

#     def __str__(self):
#         return self.__repr__()

#     @property
#     def location(self):
#         """
#         ``tuple(np.ndarray)`` -- The concept's location in concept space. The
#         two elements of the tuple are the cause and effect repertoires.
#         """
#         if self.cause and self.effect:
#             return (self.cause.repertoire, self.effect.repertoire)
#         else:
#             return (self.cause, self.effect)

#     def __eq__(self, other):
#         # TODO: this was required by the nodes->indices refactoring
#         # since subsystem is an optional arg, and mechanism is now
#         # passed with indices instead of Node objects. Is this needed?
#         # This should only matter when comparing nodes from different
#         # subsystems so checking subsystem/state equality may be sufficient.
#         if self.subsystem is not None:
#             state_eq = ([n.state for n in self.subsystem.indices2nodes(self.mechanism)] ==
#                         [n.state for n in other.subsystem.indices2nodes(other.mechanism)])
#         else:
#             state_eq = True  # maybe?

#         # TODO: handle cause and effect purviews when they are None
#         # TODO: don't use phi_eq now that all phi values should be rounded
#         # (check that)??
#         return (self.phi == other.phi
#                 and self.mechanism == other.mechanism
#                 and state_eq
#                 and self.cause.purview == other.cause.purview
#                 and self.effect.purview == other.effect.purview
#                 and self.eq_repertoires(other))

#     def __hash__(self):
#         # TODO: test and handle for nodes->indices conversion
#         return hash((self.phi,
#                      tuple(n.index for n in self.mechanism),
#                      tuple(n.state for n in self.mechanism),
#                      tuple(n.index for n in self.cause.purview),
#                      tuple(n.index for n in self.effect.purview),
#                      utils.np_hash(self.cause.repertoire),
#                      utils.np_hash(self.effect.repertoire)))

#     def __bool__(self):
#         """A concept is truthy if it is not reducible; i.e. if it has a
#         significant amount of |big_phi|."""
#         return not utils.phi_eq(self.phi, 0)

#     def eq_repertoires(self, other):
#         """Return whether this concept has the same cause and effect
#         repertoires as another.

#         .. warning::
#             This only checks if the cause and effect repertoires are equal as
#             arrays; mechanisms, purviews, or even the nodes that node indices
#             refer to, might be different.
#         """
#         return (
#             np.array_equal(self.cause.repertoire, other.cause.repertoire) and
#             np.array_equal(self.effect.repertoire, other.effect.repertoire))

#     def emd_eq(self, other):
#         """Return whether this concept is equal to another in the context of an
#         EMD calculation."""
#         return self.mechanism == other.mechanism and self.eq_repertoires(other)

#     # TODO Rename to expanded_cause_repertoire, etc
#     def expand_cause_repertoire(self, new_purview=None):
#         """Expands a cause repertoire to be a distribution over an entire
#         network."""
#         return self.subsystem.expand_cause_repertoire(self.cause.purview,
#                                                       self.cause.repertoire,
#                                                       new_purview)

#     def expand_effect_repertoire(self, new_purview=None):
#         """Expands an effect repertoire to be a distribution over an entire
#         network."""
#         return self.subsystem.expand_effect_repertoire(self.effect.purview,
#                                                        self.effect.repertoire,
#                                                        new_purview)

#     def expand_partitioned_cause_repertoire(self):
#         """Expands a partitioned cause repertoire to be a distribution over an
#         entire network."""
#         return self.subsystem.expand_cause_repertoire(
#             self.cause.purview,
#             self.cause.mip.partitioned_repertoire)

#     def expand_partitioned_effect_repertoire(self):
#         """Expands a partitioned effect repertoire to be a distribution over an
#         entire network."""
#         return self.subsystem.expand_effect_repertoire(
#             self.effect.purview,
#             self.effect.mip.partitioned_repertoire)

#     def to_json(self):
#         d = jsonify(self.__dict__)
#         # Attach the expanded repertoires to the jsonified MICEs.
#         d['cause']['repertoire'] = self.expand_cause_repertoire().flatten()
#         d['effect']['repertoire'] = self.expand_effect_repertoire().flatten()
#         d['cause']['partitioned_repertoire'] = \
#             self.expand_partitioned_cause_repertoire().flatten()
#         d['effect']['partitioned_repertoire'] = \
#             self.expand_partitioned_effect_repertoire().flatten()
#         return d

#     # Order by phi value, then by mechanism size
#     __lt__ = _phi_then_mechanism_size_lt
#     __gt__ = _phi_then_mechanism_size_gt
#     __le__ = _phi_then_mechanism_size_le
#     __ge__ = _phi_then_mechanism_size_ge


# # =============================================================================

# _bigmip_attributes = ['phi', 'unpartitioned_constellation',
#                       'partitioned_constellation', 'subsystem',
#                       'cut_subsystem']


# class BigMip:

#     """A minimum information partition for |big_phi| calculation.

#     BigMIPs may be compared with the built-in Python comparison operators
#     (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
#     are equal up to |PRECISION|, the size of the mechanism is compared
#     (exclusion principle).

#     Attributes:
#         phi (float): The |big_phi| value for the subsystem when taken against
#             this MIP, *i.e.* the difference between the unpartitioned
#             constellation and this MIP's partitioned constellation.
#         unpartitioned_constellation (tuple(Concept)): The constellation of the
#             whole subsystem.
#         partitioned_constellation (tuple(Concept)): The constellation when the
#             subsystem is cut.
#         subsystem (Subsystem): The subsystem this MIP was calculated for.
#         cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
#         time (float): The number of seconds it took to calculate.
#         small_phi_time (float): The number of seconds it took to calculate the
#             unpartitioned constellation.
#     """

#     def __init__(self, phi=None, unpartitioned_constellation=None,
#                  partitioned_constellation=None, subsystem=None,
#                  cut_subsystem=None):
#         self.phi = phi
#         self.unpartitioned_constellation = unpartitioned_constellation
#         self.partitioned_constellation = partitioned_constellation
#         self.subsystem = subsystem
#         self.cut_subsystem = cut_subsystem
#         self.time = None
#         self.small_phi_time = None

#     def __repr__(self):
#         return 'BigMip(' + ', '.join(attr + '=' + str(getattr(self, attr)) for
#                                      attr in _bigmip_attributes) + ')'

#     def __str__(self):
#         return self.__repr__()

#     @property
#     def cut(self):
#         """The unidirectional cut that makes the least difference to the
#         subsystem."""
#         return self.cut_subsystem.cut

#     def __eq__(self, other):
#         return _general_eq(self, other, _bigmip_attributes)

#     def __bool__(self):
#         """A BigMip is truthy if it is not reducible; i.e. if it has a
#         significant amount of |big_phi|."""
#         return not utils.phi_eq(self.phi, 0)

#     def __hash__(self):
#         return hash((self.phi, self.unpartitioned_constellation,
#                      self.partitioned_constellation, self.subsystem,
#                      self.cut_subsystem))

#     # First compare phi, then subsystem size
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#     def __lt__(self, other):
#         if _phi_eq(self, other):
#             if len(self.subsystem) == len(other.subsystem):
#                 return False
#             else:
#                 return len(self.subsystem) < len(other.subsystem)
#         else:
#             return _phi_lt(self, other)

#     def __gt__(self, other):
#         if _phi_eq(self, other):
#             if len(self.subsystem) == len(other.subsystem):
#                 return False
#             else:
#                 return len(self.subsystem) > len(other.subsystem)
#         else:
#             return _phi_gt(self, other)

#     def __le__(self, other):
#         return (self.__lt__(other) or _phi_eq(self, other))

#     def __ge__(self, other):
#         return (self.__gt__(other) or _phi_eq(self, other))
