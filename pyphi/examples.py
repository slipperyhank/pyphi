#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# examples.py
"""
Example networks and subsystems to go along with the documentation.
"""

import numpy as np
from pyphi.convert import loli_index2state
from .network import Network
from .subsystem import Subsystem


def basic_network():
    """A simple 3-node network with roughly two bits of |big_phi|.

    Diagram::

               +~~~~~~~~+
          +~~~~|   A    |<~~~~+
          |    |  (OR)  +~~~+ |
          |    +~~~~~~~~+   | |
          |                 | |
          |                 v |
        +~+~~~~~~+      +~~~~~+~+
        |   B    |<~~~~~+   C   |
        | (COPY) +~~~~~>| (XOR) |
        +~~~~~~~~+      +~~~~~~~+

    TPM:

    +--------------+---------------+
    |  Past state  | Current state |
    +--------------+---------------+
    |   A, B, C    |    A, B, C    |
    +==============+===============+
    |   0, 0, 0    |    0, 0, 0    |
    +--------------+---------------+
    |   1, 0, 0    |    0, 0, 1    |
    +--------------+---------------+
    |   0, 1, 0    |    1, 0, 1    |
    +--------------+---------------+
    |   1, 1, 0    |    1, 0, 0    |
    +--------------+---------------+
    |   0, 0, 1    |    1, 1, 0    |
    +--------------+---------------+
    |   1, 0, 1    |    1, 1, 1    |
    +--------------+---------------+
    |   0, 1, 1    |    1, 1, 1    |
    +--------------+---------------+
    |   1, 1, 1    |    1, 1, 0    |
    +--------------+---------------+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 0 | 1 |
    +---+---+---+---+
    | B | 1 | 0 | 1 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+

    .. note::
        |CM[i][j] = 1| means that node |i| is connected to node |j|.
    """
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ])

    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    current_state = (1, 0, 0)
    past_state = (1, 1, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def basic_subsystem():
    """A subsystem containing all the nodes of the
    :func:`pyphi.examples.basic_network`."""
    net = basic_network()
    return Subsystem(range(net.size), net)


def residue_network():
    """The network for the residue example.

    Current and past state are all nodes off.

    Diagram::

                +~~~~~~~+         +~~~~~~~+
                |   A   |         |   B   |
            +~~>| (AND) |         | (AND) |<~~+
            |   +~~~~~~~+         +~~~~~~~+   |
            |        ^               ^        |
            |        |               |        |
            |        +~~~~~+   +~~~~~+        |
            |              |   |              |
        +~~~+~~~+        +~+~~~+~+        +~~~+~~~+
        |   C   |        |   D   |        |   E   |
        |       |        |       |        |       |
        +~~~~~~~+        +~~~~~~~+        +~~~~~~~+

    Connectivity matrix:

    +---+---+---+---+---+---+
    | . | A | B | C | D | E |
    +---+---+---+---+---+---+
    | A | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | B | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | C | 1 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | D | 1 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | E | 0 | 1 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    """
    tpm = np.array([
        [int(s) for s in bin(x)[2:].zfill(5)[::-1]] for x in range(32)
    ])
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) == 2), 0] = 1
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) == 2), 1] = 1
    tpm[np.where(np.sum(tpm[0:, 2:4], 1) < 2), 0] = 0
    tpm[np.where(np.sum(tpm[0:, 3:5], 1) < 2), 1] = 0

    cm = np.zeros((5, 5))
    cm[2:4, 0] = 1
    cm[3:, 1] = 1

    current_state = (0, 0, 0, 0, 0)
    past_state = (0, 0, 0, 0, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def residue_subsystem():
    """The subsystem containing all the nodes of the
    :func:`pyphi.examples.residue_network`."""
    net = residue_network()
    return Subsystem(range(net.size), net)


def xor_network():
    """A fully connected system of three XOR gates. In the state ``(0, 0, 0)``,
    none of the elementary mechanisms exist.

    Diagram::

        +~~~~~~~+       +~~~~~~~+
        |   A   +<~~~~~>|   B   |
        | (XOR) |       | (XOR) |
        +~~~~~~~+       +~~~~~~~+
            ^               ^
            |   +~~~~~~~+   |
            +~~>|   C   |<~~+
                | (XOR) |
                +~~~~~~~+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 1 | 1 |
    +---+---+---+---+
    | B | 1 | 0 | 1 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+
    """
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ])

    cm = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    current_state = (0, 0, 0)
    past_state = (0, 0, 0)

    return Network(tpm, current_state, past_state, connectivity_matrix=cm)


def xor_subsystem():
    """The subsystem containing all the nodes of the
    :func:`pyphi.examples.xor_network`."""
    net = xor_network()
    return Subsystem(range(net.size), net)


def cond_depend_tpm():
    """A system of two general logic gates A and B such if they are in the same
    state they stay the same, but if they are in different states, they flip
    with probability 50%.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  |<~~~~~~~>|  B  |
        +~~~~~+         +~~~~~+

    TPM:

    +------+------+------+------+------+
    |      |(0, 0)|(1, 0)|(0, 1)|(1, 1)|
    +------+------+------+------+------+
    |(0, 0)| 1.0  | 0.0  | 0.0  | 0.0  |
    +------+------+------+------+------+
    |(1, 0)| 0.0  | 0.5  | 0.5  | 0.0  |
    +------+------+------+------+------+
    |(0, 1)| 0.0  | 0.5  | 0.5  | 0.0  |
    +------+------+------+------+------+
    |(1, 1)| 0.0  | 0.0  | 0.0  | 1.0  |
    +------+------+------+------+------+

    Connectivity matrix:

    +---+---+---+
    | . | A | B |
    +---+---+---+
    | A | 0 | 1 |
    +---+---+---+
    | B | 1 | 0 |
    +---+---+---+
    """

    tpm = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return tpm


def cond_independ_tpm():
    """A system of three general logic gates A, B and C such that if A and B
    are in the same state then they stay the same. If they are in different
    states, they flip if C is ''ON and stay the same if C is OFF. Node C is ON
    50% of the time, independent of the previous state.

    Diagram::

        +~~~~~+         +~~~~~+
        |  A  |<~~~~~~~>|  B  |
        +~~~~~+         +~~~~~+
           ^               ^
           |    +~~~~~+    |
           +~~~~+  C  +~~~~+
                +~~~~~+

    TPM:

    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |         |(0, 0, 0)|(1, 0, 0)|(0, 1, 0)|(1, 1, 0)|(0, 0, 1)|(1, 0, 1)|(0, 1, 1)|(1, 1, 1)|
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 0, 0)|   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 0, 0)|   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 1, 0)|   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 1, 0)|   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 0, 1)|   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 0, 1)|   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(0, 1, 1)|   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+
    |(1, 1, 1)|   0.0   |   0.0   |   0.0   |   0.5   |   0.0   |   0.0   |   0.0   |   0.5   |
    +---------+---------+---------+---------+---------+---------+---------+---------+---------+

    Connectivity matrix:

    +---+---+---+---+
    | . | A | B | C |
    +---+---+---+---+
    | A | 0 | 1 | 0 |
    +---+---+---+---+
    | B | 1 | 0 | 0 |
    +---+---+---+---+
    | C | 1 | 1 | 0 |
    +---+---+---+---+
    """

    tpm = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
    ])

    return tpm


def propagation_delay_network():
   """ A version of the primary example from the IIT 3.0 paper
   with deterministic COPY gates on each connection. These
   copy gates essentially function as propagation delays on the
   signal between OR, AND and XOR gates from the original system.

   The current and past states of the network are also selected
   to  mimic the corresponding states from the IIT 3.0 paper.


   Diagram::
                                        +----------+
                  +---------------------+ C (COPY) +<---------------------+
                  V                     +----------+                      |
       +----------+--+                                                 +--+----------+
       |             |                  +----------+                   |             |
       |   A (OR)    +----------------->+ B (COPY) +------------------>+   D (XOR)   |
       |             |                  +----------+                   |             |
       +--+-------+--+                                                 +--+-------+--+
          |       ^                                                       ^       |
          |       |                                                       |       |
          |       |   +----------+                        +----------+    |       |
          |       +---+ H (COPY) +<-------+      +------->+ F (COPY) +----+       |
          |           +----------+        |      |        +----------+            |
          |                               |      |                                |
          |                            +--+------+--+                             |
          |           +----------+     |            |     +----------+            |
          +---------->+ I (COPY) +---->|  G (AND)   |<----+ E (COPY) +<-----------+
                      +----------+     |            |     +----------+
                                       +----------- +

   Connectivity matrix:

    +---+---+---+---+---+---+---+---+---+---+
    | . | A | B | C | D | E | F | G | H | I |
    +---+---+---+---+---+---+---+---+---+---+
    | A | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
    +---+---+---+---+---+---+---+---+---+---+
    | B | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | C | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | D | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | E | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | F | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | G | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | H | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+
    | I | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
    +---+---+---+---+---+---+---+---+---+---+

   States:

        In the IIT 3.0 paper example, the past state of the system has only
        the XOR gate on. For the propagation delay network, this corresponds
        to a state of (0, 0, 0, 1, 0, 0, 0, 0, 0).

        The current state of the IIT 3.0 example has only the OR gate on. By
        advancing the propagation delay system two time steps, the current
        state (1, 0, 0, 0, 0, 0, 0, 0, 0) is achieved, with corresponding
        past state (0, 0, 1, 0, 1, 0, 0, 0, 0).
    """
   num_nodes = 9
   num_states = 2**num_nodes

   tpm = np.zeros((num_states, num_nodes))

   for past_state_index in range(num_states):
       past_state = loli_index2state(past_state_index, num_nodes)
       current_state = [0 for i in range(num_nodes)]
       if (past_state[2] == 1 or past_state[7] == 1):
           current_state[0] = 1
       if (past_state[0] == 1):
           current_state[1] = 1
           current_state[8] = 1
       if (past_state[3] == 1):
           current_state[2] = 1
           current_state[4] = 1
       if (past_state[1] == 1 ^ past_state[5] == 1):
           current_state[3] = 1
       if (past_state[4] == 1 and past_state[8] == 1):
           current_state[6] = 1
       if (past_state[6] == 1):
           current_state[5] = 1
           current_state[7] = 1
       tpm[past_state_index, :] = current_state

   cm = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0]])

   current_state = (1, 0, 0, 0, 0, 0, 0, 0, 0)
   past_state = (0, 0, 1, 0, 1, 0, 0, 0, 0)

   return Network(tpm, current_state, past_state, connectivity_matrix=cm)





