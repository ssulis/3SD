#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FUNCTION RELATED TO DETECTION TESTS APPLIED ON PERIODOGRAMS (OR STANDARDIZED PERIODOGRAMS)

@author: ssulis
@author contact: sophia.sulis@lam.fr
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#%***************************************************************************#
# Detection tests involved in Algorithm 1 (3SD procedure)
# freq = frequency grid
# pstand = standardized periodogram (len(pstand) = len(freq))
# Ttype = Test to be applied to pstand(freq)
# Outputs: test value and corresponding pstand's frequency
#*****************************************************************************#
    
def compute_test(freq, pstand, Ttype = 'Max test'):
    
    test = np.nan # initialize: test's value
    
    # Classical test of the highest periodogram value
    if Ttype == 'Max test':
        test  = np.max(pstand)
        wtest =  np.where(pstand == np.max(pstand))[0][0]
        ftest = freq[wtest]

    
    # You can implement here other detection tests
    # [...]    
    
    
    return ftest, test