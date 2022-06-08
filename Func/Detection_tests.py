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
#*****************************************************************************#
'''
# Inputs:
# freq = frequency grid
# pstand = standardized periodogram (len(pstand) = len(freq))
# Ttype = Test to be applied to pstand(freq)
# args  = arguments needed to run test named "Ttype" 
# Outputs: test value and corresponding pstand's frequency
'''
    
def compute_test(freq, pstand, Ttype = 'Max test', flim=None, args = None):
    
    test = np.nan # initialize: test's value
    if flim is not None: 
        freq_test, pstand_test = np.copy(freq[freq<flim]), np.copy(pstand[freq<flim])
    else:
        freq_test, pstand_test = np.copy(freq), np.copy(pstand)

    
    
    # Classical test of the highest periodogram value
    if Ttype == 'Max test':
        test  = np.max(pstand_test)
        wtest =  np.where(pstand == test)[0][0]
        ftest = freq[wtest]
  

    # Test of the Nc^th largest periodogram value ("TC" in the paper)
    if Ttype == 'TC test':
        NC = args # NC^th largest periodogram value
        test = np.sort(pstand_test)[-NC]
        wtest =  np.where(pstand == test)[0][0]
        ftest = freq[wtest]
        

    
    # You can implement here other detection tests
    # [...]    
    
    
    return ftest, test
