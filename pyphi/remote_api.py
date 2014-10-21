#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remote API
~~~~~~~~~~

Exposes PyPhi functions to a JSON-RPC remote API.
"""

from jsonrpc import dispatcher


@dispatcher.add_method
def big_mip(subsystem):
    return {}
