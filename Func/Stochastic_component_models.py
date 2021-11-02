#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FUNCTION RELATED TO THE STOCHASTIC NOISE SOURCE COMPONENT

@author: ssulis
@author contact: sophia.sulis@lam.fr

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# My functions
from AR_estimation import *

import warnings
warnings.filterwarnings("ignore")

#%***************************************************************************#
# Generate an estimate of the stochastic noise component based on model Mn_model
# and parameters theta_n
# time = dates of the observations or NTS time sampling
# nseries = number of time series to be generated
# Output: nseries>=1 synthetic time series following model Mn_model
#*****************************************************************************#
    
def Mn_generate(time, Mn_model, theta_n, nseries=1):

    N = len(time) # number of data points
    
    # If Mn_model is chosen as an autoregressive process
    if Mn_model == 'AR':         
        if nseries == 1: 
            x_Mn = ARgenerator(theta_n, N)
        else:
            x_Mn = np.zeros((nseries, N))
            for il in range(nseries): x_Mn[il] = ARgenerator(theta_n, N)
    
    # You can implement here other parametric noise
    # [...]
        
    return x_Mn

#%***************************************************************************#
# Estimate the parameters of the stochastic noise component based on model Mn_model
# time = dates of the observations or NTS time sampling
# data = observations or NTS 
# Output: estimated parameters of model Mn_model
#*****************************************************************************#

def Mn_estimate(time, data, Mn_model):

    N  = len(time) # number of data points
    
    hat2_theta_n = [] # initialize: parameters to be estimated
    
    # If Mn_model is chosen as an autoregressive process
    if Mn_model == 'AR':
        
        pmax = 10   # max order: defaut is 10
        criterion = 'FPE' # AR estimation criterion: defaut is 'FPE' 
        hat2_theta_n = AR_estimation(data, [pmax,criterion])

    # You can implement here other parametric noise
    # [...]
    
    return hat2_theta_n