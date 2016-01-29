#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_concept_caching.py

import pytest
import numpy as np

from pyphi import compute, constants, Network, Subsystem
import pyphi.concept_caching as cc


# Unit tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_normalized_mechanism():
    # TODO test_normalized_mechanism
    pass


def test_different_states(big):
    all_off = [0] * 5
    all_on = [1] * 5
    s1 = Subsystem(big, all_off, range(2, 5))
    s2 = Subsystem(big, all_on, range(2, 5))
    a = s1.nodes[2:]
    b = s2.nodes[2:]
    x = cc.NormalizedMechanism(a, s1)
    y = cc.NormalizedMechanism(b, s2)
    assert x != y


def test_normalize_purview_and_repertoire(big_subsys_all):
    purview = (big_subsys_all.nodes[0],
               big_subsys_all.nodes[3],
               big_subsys_all.nodes[4])
    repertoire = np.arange(8).reshape(2, 1, 1, 2, 2)
    normalized_indices = {0: 1, 3: 2, 4: 0}
    unnormalized_indices = {v: k for k, v in normalized_indices.items()}

    normalized_purview = (0, 1, 2)
    normalized_repertoire = np.arange(8).reshape(2, 2, 2).transpose((2, 0, 1))

    result_purview, result_repertoire = \
        cc._normalize_purview_and_repertoire(purview, repertoire,
                                             normalized_indices)

    assert result_purview == normalized_purview
    assert np.array_equal(result_repertoire, normalized_repertoire)

    result_purview, result_repertoire = \
        cc._unnormalize_purview_and_repertoire(normalized_purview,
                                               normalized_repertoire,
                                               unnormalized_indices,
                                               big_subsys_all)

    assert result_purview == purview
    assert np.array_equal(result_repertoire, repertoire)


# TODO implement
def test_unnormalize_mice():
    pass


# TODO implement
def test_unnormalize_concept():
    pass


# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def check_concept_caching(net, states, flushcache):
    flushcache()

    # Build the networks for each state.
    networks = {
        s: Network(net.tpm, net.connectivity_matrix) for s in states
    }

    # Empty the cache.
    flushcache()

    # Get the complexes for each state with no concept caching.
    constants.CACHE_CONCEPTS = False
    no_caching_results = []
    for s in states:
        no_caching_results.append(list(compute.complexes(networks[s], s)))

    # Empty the cache.
    flushcache()

    # Get the complexes for each state with concept caching.
    constants.CACHE_CONCEPTS = True
    caching_results = []
    for s in states:
        caching_results.append(list(compute.complexes(networks[s], s)))

    assert caching_results == no_caching_results


# End-to-end tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This test sometimes fails
@pytest.mark.xfail
def test_standard(s, flushcache):
    check_concept_caching(s.network, [(s.state)], flushcache)


@pytest.mark.xfail
def test_noised(s_noised, flushcache):
    check_concept_caching(s_noised.network, [(s_noised.state)], flushcache)


@pytest.mark.slow
def test_big(big_subsys_all, flushcache):
    check_concept_caching(big_subsys_all.network, [(big_subsys_all.state)],
                          flushcache)


@pytest.mark.veryslow
def test_rule152(rule152, flushcache):
    states = [
        (0, 1, 0, 0, 0),
        (1, 1, 1, 1, 1),
        (1, 1, 0, 1, 0),
        (0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0)
    ]
    check_concept_caching(rule152, states, flushcache)
