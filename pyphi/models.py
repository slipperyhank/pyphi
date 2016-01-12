#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models.py

"""
Containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import Iterable, namedtuple

import numpy as np

from . import utils, config
from .jsonify import jsonify


# TODO use properties to avoid data duplication

def make_repr(self, attrs):
    """Construct a repr string.
    If `config.READABLE_REPRS` is True, this function calls out
    to the object's __str__ method. Although this breaks the convention
    that __repr__ should return a string which can reconstruct the object,
    readable reprs are invaluable since the Python interpreter calls
    `repr` to represent all objects in the shell. Since PyPhi is often
    used in the interpreter we want to have meaningful and useful
    representations.
    Args:
        self (obj): The object in question
        attrs (iterable(str)): Attributes to include in the repr
    Returns:
        (str): the `repr`esentation of the object
    """
    # TODO: change this to a closure so we can do
    # __repr__ = make_repr(attrs) ???

    if config.READABLE_REPRS:
        return self.__str__()

    return "{}({})".format(
        self.__class__.__name__,
        ", ".join(attr + '=' + repr(getattr(self, attr)) for attr in attrs))


class Cut(namedtuple('Cut', ['severed', 'intact'])):

    """Represents a unidirectional cut.

    Attributes:
        severed (tuple(int)):
            Connections from this group of nodes to those in ``intact`` are
            severed.
        intact (tuple(int)):
            Connections to this group of nodes from those in ``severed`` are
            severed.
    """
    # This allows accessing the namedtuple's ``__dict__``; see
    # https://docs.python.org/3.3/reference/datamodel.html#notes-on-using-slots
    __slots__ = ()

    def __repr__(self):
        return make_repr(self, ['severed', 'intact'])

    def __str__(self):
        return "Cut {self.severed} --//--> {self.intact}".format(self=self)


class Part(namedtuple('Part', ['mechanism', 'purview'])):

    """Represents one part of a bipartition.

    Attributes:
        mechanism (tuple(int)):
            The nodes in the mechanism for this part.
        purview (tuple(int)):
            The nodes in the mechanism for this part.

    Example:
        When calculating |small_phi| of a 3-node subsystem, we partition the
        system in the following way::

            mechanism:   A C        B
                        -----  X  -----
              purview:    B        A C

        This class represents one term in the above product.
    """
    __slots__ = ()
    pass


# Phi-ordering methods
# =============================================================================

# Compare phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _phi_eq(self, other):
    try:
        return utils.phi_eq(self.phi, other.phi)
    except AttributeError:
        return False


def _phi_lt(self, other):
    try:
        if not utils.phi_eq(self.phi, other.phi):
            return self.phi < other.phi
        return False
    except AttributeError:
        return False


def _phi_gt(self, other):
    try:
        if not utils.phi_eq(self.phi, other.phi):
            return self.phi > other.phi
        return False
    except AttributeError:
        return False


def _phi_le(self, other):
    return _phi_lt(self, other) or _phi_eq(self, other)


def _phi_ge(self, other):
    return _phi_gt(self, other) or _phi_eq(self, other)


# First compare phi, then mechanism size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _phi_then_mechanism_size_lt(self, other):
    if _phi_eq(self, other):
        return (len(self.mechanism) < len(other.mechanism)
                if hasattr(other, 'mechanism') else False)
    else:
        return _phi_lt(self, other)


def _phi_then_mechanism_size_gt(self, other):
    return (not _phi_then_mechanism_size_lt(self, other) and
            not self == other)


def _phi_then_mechanism_size_le(self, other):
    return (_phi_then_mechanism_size_lt(self, other) or
            _phi_eq(self, other))


def _phi_then_mechanism_size_ge(self, other):
    return (_phi_then_mechanism_size_gt(self, other) or
            _phi_eq(self, other))


# Equality helpers
# =============================================================================

# TODO use builtin numpy methods here
def _numpy_aware_eq(a, b):
    """Return whether two objects are equal via recursion, using
    :func:`numpy.array_equal` for comparing numpy arays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if ((isinstance(a, Iterable) and isinstance(b, Iterable))
            and not isinstance(a, str) and not isinstance(b, str)):
        if len(a) != len(b):
            return False
        return all(_numpy_aware_eq(x, y) for x, y in zip(a, b))
    return a == b


def _general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'phi'``, it is compared up to |PRECISION|. All
    other attributes are compared with :func:`_numpy_aware_eq`.

    If an attribute is called ``'mechanism'`` or ``'purview'``, it is compared
    using set equality."""
    try:
        for attr in attributes:
            _a, _b = getattr(a, attr), getattr(b, attr)
            if attr == 'phi':
                if not utils.phi_eq(_a, _b):
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

_mip_attributes = ['phi', 'direction', 'mechanism', 'purview', 'partition',
                   'unpartitioned_repertoire', 'partitioned_repertoire']
_mip_attributes_for_eq = ['phi', 'direction', 'mechanism',
                          'unpartitioned_repertoire']


class Mip(namedtuple('Mip', _mip_attributes)):

    """A minimum information partition for |small_phi| calculation.

    MIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        phi (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned repertoires.
        direction (str):
            The temporal direction specifiying whether this MIP should be
            calculated with cause or effect repertoires.
        mechanism (tuple(int)):
            The mechanism over which to evaluate the MIP.
        purview (tuple(int)):
            The purview over which the unpartitioned repertoire differs the
            least from the partitioned repertoire.
        partition (tuple(Part, Part)):
            The partition that makes the least difference to the mechanism's
            repertoire.
        unpartitioned_repertoire (np.ndarray):
            The unpartitioned repertoire of the mechanism.
        partitioned_repertoire (np.ndarray):
            The partitioned repertoire of the mechanism. This is the product of
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
        if not self.purview or not other.purview:
            return (_general_eq(self, other, _mip_attributes_for_eq) and
                    len(self.mechanism) == len(other.mechanism))
        else:
            return (_general_eq(self, other, _mip_attributes_for_eq) and
                    len(self.mechanism) == len(other.mechanism) and
                    len(self.purview) == len(other.purview))

    def __bool__(self):
        """A Mip is truthy if it is not reducible; i.e. if it has a significant
        amount of |small_phi|."""
        return not utils.phi_eq(self.phi, 0)

    def __hash__(self):
        return hash((self.phi, self.direction, self.mechanism, self.purview,
                     utils.np_hash(self.unpartitioned_repertoire)))

    def to_json(self):
        d = self.__dict__
        # Flatten the repertoires.
        d['partitioned_repertoire'] = self.partitioned_repertoire.flatten()
        d['unpartitioned_repertoire'] = self.unpartitioned_repertoire.flatten()
        return d

    def __repr__(self):
        return make_repr(self, _mip_attributes)

    def __str__(self):
        return "Mip\n" + indent(fmt_mip(self))

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================


class Mice:

    """A maximally irreducible cause or effect (i.e., “core cause” or “core
    effect”).

    relevant_connections (np.array):
        An ``N x N`` matrix, where ``N`` is the number of nodes in this
        corresponding subsystem, that identifies connections that “matter” to
        this MICE.

        ``direction == 'past'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            cause purview and node ``j`` is in the mechanism (and ``0``
            otherwise).

        ``direction == 'future'``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            mechanism and node ``j`` is in the effect purview (and ``0``
            otherwise).

    MICEs may be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, ``phi`` values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared (exclusion
    principle).
    """

    def __init__(self, mip, relevant_connections=None):
        self._mip = mip
        self._relevant_connections = relevant_connections

    @property
    def phi(self):
        """
        ``float`` -- The difference between the mechanism's unpartitioned and
        partitioned repertoires.
        """
        return self._mip.phi

    @property
    def direction(self):
        """
        ``str`` -- Either 'past' or 'future'. If 'past' ('future'), this
        represents a maximally irreducible cause (effect).
        """
        return self._mip.direction

    @property
    def mechanism(self):
        """
        ``list(int)`` -- The mechanism for which the MICE is evaluated.
        """
        return self._mip.mechanism

    @property
    def purview(self):
        """
        ``list(int)`` -- The purview over which this mechanism's |small_phi|
        is maximal.
        """
        return self._mip.purview

    @property
    def repertoire(self):
        """
        ``np.ndarray`` -- The unpartitioned repertoire of the mechanism over
        the purview.
        """
        return self._mip.unpartitioned_repertoire

    @property
    def mip(self):
        """
        ``Mip`` -- The minimum information partition for this mechanism.
        """
        return self._mip

    def __repr__(self):
        return make_repr(self, ['mip'])

    def __str__(self):
        return "Mice\n" + indent(fmt_mip(self.mip))

    def __eq__(self, other):
        return self.mip == other.mip

    def __hash__(self):
        return hash(('Mice', self._mip))

    def to_json(self):
        return {'mip': self._mip}

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


# =============================================================================

_concept_attributes = ['phi', 'mechanism', 'cause', 'effect', 'subsystem',
                       'normalized']


# TODO: make mechanism a property
# TODO: make phi a property
class Concept:

    """A star in concept-space.

    The ``phi`` attribute is the |small_phi_max| value. ``cause`` and
    ``effect`` are the MICE objects for the past and future, respectively.

    Concepts may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared.

    Attributes:
        phi (float):
            The size of the concept. This is the minimum of the |small_phi|
            values of the concept's core cause and core effect.
        mechanism (tuple(int)):
            The mechanism that the concept consists of.
        cause (|Mice|):
            The |Mice| representing the core cause of this concept.
        effect (|Mice|):
            The |Mice| representing the core effect of this concept.
        subsystem (Subsystem):
            This concept's parent subsystem.
        time (float):
            The number of seconds it took to calculate.
    """

    def __init__(self, phi=None, mechanism=None, cause=None, effect=None,
                 subsystem=None, normalized=False):
        self.phi = phi
        self.mechanism = mechanism
        self.cause = cause
        self.effect = effect
        self.subsystem = subsystem
        self.normalized = normalized
        self.time = None

    def __repr__(self):
        return make_repr(self, _concept_attributes)

    def __str__(self):
        return "Concept\n""-------\n" + fmt_concept(self)


    @property
    def location(self):
        """
        ``tuple(np.ndarray)`` -- The concept's location in concept space. The
        two elements of the tuple are the cause and effect repertoires.
        """
        if self.cause and self.effect:
            return (self.cause.repertoire, self.effect.repertoire)
        else:
            return (self.cause, self.effect)

    def __eq__(self, other):
        # TODO: this was required by the nodes->indices refactoring
        # since subsystem is an optional arg, and mechanism is now
        # passed with indices instead of Node objects. Is this needed?
        # This should only matter when comparing nodes from different
        # subsystems so checking subsystem/state equality may be sufficient.
        if self.subsystem is not None:
            state_eq = ([n.state for n in self.subsystem.indices2nodes(self.mechanism)] ==
                        [n.state for n in other.subsystem.indices2nodes(other.mechanism)])
        else:
            state_eq = True  # maybe?

        # TODO: handle cause and effect purviews when they are None
        # TODO: don't use phi_eq now that all phi values should be rounded
        # (check that)??
        return (self.phi == other.phi
                and self.mechanism == other.mechanism
                and state_eq
                and self.cause.purview == other.cause.purview
                and self.effect.purview == other.effect.purview
                and self.eq_repertoires(other))

    def __hash__(self):
        # TODO: test and handle for nodes->indices conversion
        return hash((self.phi,
                     tuple(n.index for n in self.mechanism),
                     tuple(n.state for n in self.mechanism),
                     tuple(n.index for n in self.cause.purview),
                     tuple(n.index for n in self.effect.purview),
                     utils.np_hash(self.cause.repertoire),
                     utils.np_hash(self.effect.repertoire)))

    def __bool__(self):
        """A concept is truthy if it is not reducible; i.e. if it has a
        significant amount of |big_phi|."""
        return not utils.phi_eq(self.phi, 0)

    def eq_repertoires(self, other):
        """Return whether this concept has the same cause and effect
        repertoires as another.

        .. warning::
            This only checks if the cause and effect repertoires are equal as
            arrays; mechanisms, purviews, or even the nodes that node indices
            refer to, might be different.
        """
        return (
            np.array_equal(self.cause.repertoire, other.cause.repertoire) and
            np.array_equal(self.effect.repertoire, other.effect.repertoire))

    def emd_eq(self, other):
        """Return whether this concept is equal to another in the context of an
        EMD calculation."""
        return self.mechanism == other.mechanism and self.eq_repertoires(other)

    # TODO Rename to expanded_cause_repertoire, etc
    def expand_cause_repertoire(self, new_purview=None):
        """Expands a cause repertoire to be a distribution over an entire
        network."""
        return self.subsystem.expand_cause_repertoire(self.cause.purview,
                                                      self.cause.repertoire,
                                                      new_purview)

    def expand_effect_repertoire(self, new_purview=None):
        """Expands an effect repertoire to be a distribution over an entire
        network."""
        return self.subsystem.expand_effect_repertoire(self.effect.purview,
                                                       self.effect.repertoire,
                                                       new_purview)

    def expand_partitioned_cause_repertoire(self):
        """Expands a partitioned cause repertoire to be a distribution over an
        entire network."""
        return self.subsystem.expand_cause_repertoire(
            self.cause.purview,
            self.cause.mip.partitioned_repertoire)

    def expand_partitioned_effect_repertoire(self):
        """Expands a partitioned effect repertoire to be a distribution over an
        entire network."""
        return self.subsystem.expand_effect_repertoire(
            self.effect.purview,
            self.effect.mip.partitioned_repertoire)

    def to_json(self):
        d = jsonify(self.__dict__)
        # Attach the expanded repertoires to the jsonified MICEs.
        d['cause']['repertoire'] = self.expand_cause_repertoire().flatten()
        d['effect']['repertoire'] = self.expand_effect_repertoire().flatten()
        d['cause']['partitioned_repertoire'] = \
            self.expand_partitioned_cause_repertoire().flatten()
        d['effect']['partitioned_repertoire'] = \
            self.expand_partitioned_effect_repertoire().flatten()
        return d

    # Order by phi value, then by mechanism size
    __lt__ = _phi_then_mechanism_size_lt
    __gt__ = _phi_then_mechanism_size_gt
    __le__ = _phi_then_mechanism_size_le
    __ge__ = _phi_then_mechanism_size_ge


class Constellation(tuple):
    """A constellation of concepts.
    This is a wrapper around a tuple to provide a nice string
    representation and place to put constellation methods. Previously,
    constellations were represented as `tuple(Concept)`; this usage still
    works in all functions.
    """

    def __repr__(self):
        if config.READABLE_REPRS:
            return self.__str__()
        return "Constellation({})".format(super(Constellation, self).__repr__())

    def __str__(self):
        return "\nConstellation\n*************" + fmt_constellation(self)
        
# =============================================================================

_bigmip_attributes = ['phi', 'unpartitioned_constellation',
                      'partitioned_constellation', 'subsystem',
                      'cut_subsystem']


class BigMip:

    """A minimum information partition for |big_phi| calculation.

    BigMIPs may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the mechanism is compared
    (exclusion principle).

    Attributes:
        phi (float): The |big_phi| value for the subsystem when taken against
            this MIP, *i.e.* the difference between the unpartitioned
            constellation and this MIP's partitioned constellation.
        unpartitioned_constellation (tuple(Concept)): The constellation of the
            whole subsystem.
        partitioned_constellation (tuple(Concept)): The constellation when the
            subsystem is cut.
        subsystem (Subsystem): The subsystem this MIP was calculated for.
        cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
        time (float): The number of seconds it took to calculate.
        small_phi_time (float): The number of seconds it took to calculate the
            unpartitioned constellation.
    """

    def __init__(self, phi=None, unpartitioned_constellation=None,
                 partitioned_constellation=None, subsystem=None,
                 cut_subsystem=None):
        self.phi = phi
        self.unpartitioned_constellation = unpartitioned_constellation
        self.partitioned_constellation = partitioned_constellation
        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem
        self.time = None
        self.small_phi_time = None

    def __repr__(self):
        return make_repr(self, _bigmip_attributes)

    def __str__(self):
        return "\nBigMip\n======\n" + fmt_big_mip(self)

    @property
    def cut(self):
        """The unidirectional cut that makes the least difference to the
        subsystem."""
        return self.cut_subsystem.cut

    def __eq__(self, other):
        return _general_eq(self, other, _bigmip_attributes)

    def __bool__(self):
        """A BigMip is truthy if it is not reducible; i.e. if it has a
        significant amount of |big_phi|."""
        return not utils.phi_eq(self.phi, 0)

    def __hash__(self):
        return hash((self.phi, self.unpartitioned_constellation,
                     self.partitioned_constellation, self.subsystem,
                     self.cut_subsystem))

    # First compare phi, then subsystem size
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __lt__(self, other):
        if _phi_eq(self, other):
            if len(self.subsystem) == len(other.subsystem):
                return False
            else:
                return len(self.subsystem) < len(other.subsystem)
        else:
            return _phi_lt(self, other)

    def __gt__(self, other):
        if _phi_eq(self, other):
            if len(self.subsystem) == len(other.subsystem):
                return False
            else:
                return len(self.subsystem) > len(other.subsystem)
        else:
            return _phi_gt(self, other)

    def __le__(self, other):
        return (self.__lt__(other) or _phi_eq(self, other))

    def __ge__(self, other):
        return (self.__gt__(other) or _phi_eq(self, other))

# Formatting functions for __str__ and __repr__
# TODO: probably move this to utils.py, or maybe fmt.py??


def indent(lines, amount=2, chr=' '):
    """Indent a string.
    Prepends whitespace to every line in the passed string. (Lines are
    separated by ``\n``)
    Args:
        lines (str): The string to indent.
    Keyword Args:
        amount (int): The number of columns to indent by.
        chr (char): The character to to use as the indentation.
    Returns:
        str: The indented string.
    """
    lines = str(lines)
    padding = amount * chr
    return padding + ('\n' + padding).join(lines.split('\n'))


def fmt_constellation(c):
    """Format a constellation."""
    if not c:
        return "()\n"
    return "\n\n" + "\n".join(map(lambda x: indent(x), c)) + "\n"


def fmt_partition(partition):
    """Format a partition
    Args:
        partition (tuple(Part, Part)): The partition in question.
    Returns:
        str: A string representation that looks like
            0,1   []
            --- X ---
             2    0,1
    """
    if not partition:
        return ""

    part0, part1 = partition
    node_repr = lambda x: ','.join(map(str, x)) if x else '[]'
    numer0, denom0 = node_repr(part0.mechanism), node_repr(part0.purview)
    numer1, denom1 = node_repr(part1.mechanism), node_repr(part1.purview)

    width0 = max(len(numer0), len(denom0))
    width1 = max(len(numer1), len(denom1))

    return ("{numer0:^{width0}}   {numer1:^{width1}}\n"
                        "{div0} X {div1}\n"
            "{denom0:^{width0}}   {denom1:^{width1}}").format(
                numer0=numer0, denom0=denom0, width0=width0, div0='-' * width0,
                numer1=numer1, denom1=denom1, width1=width1, div1='-' * width1)


def fmt_concept(concept):
    """Format a Concept string"""
    return (
        "phi: {concept.phi}\n"
        "mechanism: {concept.mechanism}\n"
        "cause: {cause}\n"
        "effect: {effect}\n".format(
            concept=concept,
            cause=("\n" + indent(fmt_mip(concept.cause.mip, verbose=False))
                   if concept.cause else ""),
            effect=("\n" + indent(fmt_mip(concept.effect.mip, verbose=False))
                    if concept.effect else "")))


def fmt_mip(mip, verbose=True):
    """Helper function to format a nice Mip string"""

    if mip is False or mip is None:  # mips can be Falsy
        return ""

    mechanism = "mechanism: {}\n".format(mip.mechanism) if verbose else ""
    direction = "direction: {}\n".format(mip.direction) if verbose else ""
    return (
        "phi: {mip.phi}\n"
        "{mechanism}"
        "purview: {mip.purview}\n"
        "partition:\n{partition}\n"
        "{direction}"
        "unpartitioned_repertoire:\n{unpart_rep}\n"
        "partitioned_repertoire:\n{part_rep}").format(
            mechanism=mechanism,
            direction=direction,
            mip=mip,
            partition=indent(fmt_partition(mip.partition)),
            unpart_rep=indent(mip.unpartitioned_repertoire.flatten(order = 'F')),
            part_rep=indent(mip.partitioned_repertoire.flatten(order = 'F')))


def fmt_big_mip(big_mip):
    """Format a BigMip"""
    return (
        "phi: {big_mip.phi}\n"
        "subsystem: {big_mip.subsystem}\n"
        "cut: {big_mip.cut}\n"
        "unpartitioned_constellation: {unpart_const}"
        "partitioned_constellation: {part_const}".format(
            big_mip=big_mip,
            unpart_const=fmt_constellation(big_mip.unpartitioned_constellation),
            part_const=fmt_constellation(big_mip.partitioned_constellation)))