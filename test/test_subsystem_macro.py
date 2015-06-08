#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyphi

tpm = np.zeros((16, 4)) + 0.3

tpm[12:,0:2] = 1
tpm[3, 3:5] = 1
tpm[7, 3:5] = 1
tpm[11, 3:5] = 1


cm = np.array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [1, 1, 0, 0],
               [1, 1, 0, 0]])

cs = (1, 1, 1, 1)

macro_network = pyphi.Network(tpm, cs, connectivity_matrix=cm)

