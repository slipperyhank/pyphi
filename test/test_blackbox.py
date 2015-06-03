#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyphi

tpm = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1]
])

cm = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
])

current_state = (1, 0, 0, 0)

network = pyphi.Network(tpm, current_state, connectivity_matrix=cm)

# Null blackboxing
subsystem1 = pyphi.Subsystem(range(network.size), network)

# Only temporal blackbox
subsystem2 = pyphi.Subsystem(range(network.size), network, time_scale=3)

# Spatio-temporal blackbox
hidden_indices = np.array([1, 3])
time_scale = 2
subsystem3 = pyphi.Subsystem(range(network.size), network, hidden_indices=hidden_indices, time_scale=time_scale)

# Single blackbox
hidden_indices = np.array([1, 2, 3])
time_scale = 4
subsystem4 = pyphi.Subsystem(range(network.size), network, hidden_indices=hidden_indices, time_scale=time_scale)


def test_blackbox_tpm():
    answer1 = tpm
    answer2 = tpm[:, [2, 3, 0, 1]]
    answer3 = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    answer4 = np.array([[0], [1]])
    assert np.array_equal(subsystem1.tpm.reshape([16]+[4], order='F'), answer1)
    assert np.array_equal(subsystem2.tpm.reshape([16]+[4], order='F'), answer2)
    assert np.array_equal(
        subsystem3.tpm.reshape([4]+[2], order='F'),
        answer3)
    assert np.array_equal(
        subsystem4.tpm.reshape([2]+[1], order='F'),
        answer4)


def test_blackbox_cm():
    answer1 = cm
    answer2 = cm[:, [2, 3, 0, 1]]
    answer3 = np.array([
        [0, 1],
        [1, 0]
    ])
    answer4 = np.array([
        [1]
    ])
    assert np.array_equal(subsystem1.connectivity_matrix, answer1)
    assert np.array_equal(subsystem2.connectivity_matrix, answer2)
    assert np.array_equal(subsystem3.connectivity_matrix, answer3)
    assert np.array_equal(subsystem4.connectivity_matrix, answer4)
