#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON
~~~~

PyPhi- and NumPy-aware JSON codec.
"""

import numpy as np
import json as _json


class JSONEncoder(_json.JSONEncoder):

    """
    An extension of the built-in JSONEncoder that can handle native PyPhi
    objects as well as NumPy arrays.

    Uses the ``json_dict`` method for PyPhi objects, and NumPy's ``tolist``
    function for arrays.
    """

    def default(self, obj):
        # If we have an array, convert it to a list.
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Otherwise, use the ``json_dict`` function if available.
        try:
            return obj.json_dict()
        # If not, let the super handle it.
        except AttributeError:
            return super().default(obj)

    def encode(self, obj):
        # Force ``default`` to be called even for already-serializable native
        # Python types.
        return super().encode(self.default(obj))


def dumps(obj):
    """Serialize ``obj`` to a JSON formatted ``str``."""
    # Use our encoder and compact separators.
    return _json.dumps(obj, cls=JSONEncoder, separators=(',', ':'))


class JSONDecoder(_json.JSONDecoder):

    """
    An extension of the built-in JSONDecoder that can handle native PyPhi
    objects as well as NumPy arrays.
    """
    pass


def loads(s):
    """Deserialize ``s`` (a ``str`` instance containing a JSON document) to a
    PyPhi object."""
    pass
