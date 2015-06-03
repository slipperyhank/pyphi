#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pyphi.utils import sparse_time, dense_time
from pyphi.convert import state_by_node2state_by_state as sbn2sbs


tpm_noise = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25]
])

tpm_copy = sbn2sbs(np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
]))

tpm_copy2 = sbn2sbs(np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1]
]))

tpm_copy3 = sbn2sbs(np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
]))

tpm_huge = np.array([[1 if i == j+1 else 0
                      for i in range(1000)]
                     for j in range(1000)])
tpm_huge[999, 0] = 1

# Tests for purely temporal blackboxing
# =====================================


def test_sparse_blackbox():
    assert np.array_equal(sparse_time(tpm_huge, 1001), tpm_huge)


def test_dense_blackbox():
    assert np.array_equal(dense_time(tpm_noise, 2), tpm_noise)
    assert np.array_equal(dense_time(tpm_noise, 3), tpm_noise)


def test_cycle_blackbox():
    assert np.array_equal(sparse_time(tpm_copy, 2), tpm_copy2)
    assert np.array_equal(sparse_time(tpm_copy, 3), tpm_copy3)
    assert np.array_equal(dense_time(tpm_copy, 2), tpm_copy2)
    assert np.array_equal(dense_time(tpm_copy, 3), tpm_copy3)
