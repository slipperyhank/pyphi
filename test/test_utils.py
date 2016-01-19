#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyphi
from pyphi import utils, constants, models


sub = pyphi.examples.basic_subsystem()
mip = pyphi.compute.big_mip(sub)
cut = pyphi.models.Cut((0,), (1, 2))
s_cut = pyphi.subsystem.Subsystem(sub.network, sub.state, sub.node_indices,
                                  cut=cut, mice_cache=sub._mice_cache)


def test_apply_cut():
    cm = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    cut = models.Cut(severed=(0, 3), intact=(1, 2))
    cut_cm = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0]
    ])
    assert np.array_equal(utils.apply_cut(cut, cm), cut_cm)


def test_fully_connected():
    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert not utils.fully_connected(cm, (0,), (0, 1, 2))
    assert not utils.fully_connected(cm, (2,), (2,))
    assert not utils.fully_connected(cm, (0, 1), (1, 2))
    assert utils.fully_connected(cm, (0, 1), (0, 2))
    assert utils.fully_connected(cm, (1, 2), (1, 2))
    assert utils.fully_connected(cm, (0, 1, 2), (0, 1, 2))


def test_phi_eq():
    phi = 0.5
    close_enough = phi - constants.EPSILON/2
    not_quite = phi - constants.EPSILON*2
    assert utils.phi_eq(phi, close_enough)
    assert not utils.phi_eq(phi, not_quite)
    assert not utils.phi_eq(phi, (phi - phi))


def test_marginalize_out(s):
    marginalized_distribution = utils.marginalize_out(s.nodes[0].index,
                                                      s.network.tpm)
    assert np.array_equal(marginalized_distribution,
                          np.array([[[[0.,  0.,  0.5],
                                      [1.,  1.,  0.5]],
                                     [[1.,  0.,  0.5],
                                      [1.,  1.,  0.5]]]]))


def test_purview_max_entropy_distribution():
    max_ent = utils.max_entropy_distribution((0, 1), 3)
    assert max_ent.shape == (2, 2, 1)
    assert np.array_equal(max_ent,
                          (np.ones(4) / 4).reshape((2, 2, 1)))
    assert max_ent[0][1][0] == 0.25


def test_combs_for_1D_input():
    n, k = 3, 2
    data = np.arange(n)
    assert np.array_equal(utils.combs(data, k),
                          np.asarray([[0, 1],
                                      [0, 2],
                                      [1, 2]]))


def test_combs_r_is_0():
    n, k = 3, 0
    data = np.arange(n)
    assert np.array_equal(utils.combs(data, k), np.asarray([]))


def test_comb_indices():
    n, k = 3, 2
    data = np.arange(6).reshape(2, 3)
    assert np.array_equal(data[:, utils.comb_indices(n, k)],
                          np.asarray([[[0, 1],
                                       [0, 2],
                                       [1, 2]],
                                      [[3, 4],
                                       [3, 5],
                                       [4, 5]]]))


def test_powerset():
    a = np.arange(2)
    assert list(utils.powerset(a)) == [(), (0,), (1,), (0, 1)]


def test_hamming_matrix():
    H = utils._hamming_matrix(3)
    answer = np.array([[0.,  1.,  1.,  2.,  1.,  2.,  2.,  3.],
                       [1.,  0.,  2.,  1.,  2.,  1.,  3.,  2.],
                       [1.,  2.,  0.,  1.,  2.,  3.,  1.,  2.],
                       [2.,  1.,  1.,  0.,  3.,  2.,  2.,  1.],
                       [1.,  2.,  2.,  3.,  0.,  1.,  1.,  2.],
                       [2.,  1.,  3.,  2.,  1.,  0.,  2.,  1.],
                       [2.,  3.,  1.,  2.,  1.,  2.,  0.,  1.],
                       [3.,  2.,  2.,  1.,  2.,  1.,  1.,  0.]])
    assert (H == answer).all()


def test_directed_bipartition():
    answer = [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)),
              ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,)), ((1, 2, 3), ())]
    assert answer == utils.directed_bipartition((1, 2, 3))
    # Test with empty input
    assert [] == utils.directed_bipartition(())


def test_emd_same_distributions():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    assert utils.hamming_emd(a, b) == 0.0


def test_uniform_distribution():
    assert np.array_equal(utils.uniform_distribution(3),
                          (np.ones(8)/8).reshape([2]*3))


def test_block_cm():
    cm1 = np.array([
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    cm3 = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1]
    ])
    cm4 = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0]
    ])
    cm5 = np.array([
        [1, 1],
        [0, 1]
    ])
    assert not utils.block_cm(cm1)
    assert utils.block_cm(cm2)
    assert utils.block_cm(cm3)
    assert not utils.block_cm(cm4)
    assert not utils.block_cm(cm5)


def test_block_reducible():
    cm1 = np.array([
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    cm2 = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    cm3 = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1]
    ])
    cm4 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert not utils.block_reducible(cm1, tuple(range(cm1.shape[0] - 1)),
                                     tuple(range(cm1.shape[1])))
    assert utils.block_reducible(cm2, (0, 1, 2), (0, 1, 2))
    assert utils.block_reducible(cm3, (0, 1), (0, 1, 2, 3, 4))
    assert not utils.block_reducible(cm4, (0, 1), (1, 2))


def test_get_inputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert utils.get_inputs_from_cm(0, cm) == (1,)
    assert utils.get_inputs_from_cm(1, cm) == (0, 1)
    assert utils.get_inputs_from_cm(2, cm) == (1,)


def test_get_outputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert utils.get_outputs_from_cm(0, cm) == (1,)
    assert utils.get_outputs_from_cm(1, cm) == (0, 1, 2)
    assert utils.get_outputs_from_cm(2, cm) == tuple()


def test_submatrix():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert np.array_equal(utils.submatrix(cm, (0,), (0, 1)),
                          np.array([[0, 1]]))
    assert np.array_equal(utils.submatrix(cm, (0, 1), (1, 2)),
                          np.array([[1, 0], [1, 1]]))
    assert np.array_equal(utils.submatrix(cm, (0, 1, 2), (0, 1, 2)), cm)


def test_relevant_connections():
    cm = utils.relevant_connections(2, (0, 1), (1,))
    assert np.array_equal(cm, [
        [0, 1],
        [0, 1],
    ])
    cm = utils.relevant_connections(3, (0, 1), (0, 2))
    assert np.array_equal(cm, [
        [1, 0, 1],
        [1, 0, 1],
        [0, 0, 0],
    ])


def test_strongly_connected():
    cm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])
    assert utils.strongly_connected(cm)

    cm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    assert not utils.strongly_connected(cm)

    cm = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    assert utils.strongly_connected(cm, (0, 1))
