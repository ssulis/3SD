#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

ALGORITHM 1 = SEMI-SUPERVIZED STANDARDIZED DETECTION (3SD) PROCEDURE

@author: ssulis
@author contact: sophia.sulis@lam.fr
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# My functions
from Periodograms import *
from Stochastic_component_models import *
from Nuisance_component_models import *
from Detection_tests import *
from Algorithm_1_3SD import *

import warnings
warnings.filterwarnings("ignore")

#%***************************************************************************#
# Algorithm 1: 3SD procedure
# see Sec.2 of Sulis et al., 2021, submitted to A&A
#*****************************************************************************#
    

def Algorithm1_3SD(x, 
                   Ptype = 'Lomb-Scargle',
                   Ttype = 'Max test', 
                   freq_grid=None,
                   TL = None,
                   Mn = None,
                   Md = None,
                   c  = None,
                   check = False):
    
    # check the input parameters ==============================================
    time, rv, rv_err = x
    N                = len(time) # number of data points

    if Ptype not in ['FFT','Lomb-Scargle','GLS']: 
        raise ValueError('Input parameter "Ptype" should be FFT, Lomb-Scargle or GLS. If different, please add it with a new function')

    if Ttype not in ['Max test']: 
        raise ValueError('Input parameter "Ttype" should be Max test. If different, please add it with a new function')

    if freq_grid != None and len(freq_grid)<N:
        raise ValueError('Problem with the input frequency grid with length < input data')
        
    if TL is not None:
        L = len(TL) # number of NTS avalaible
        if L ==1 and len(TL) < N:
            raise ValueError('Problem with the input NTS with length < input data')
        if L >1 :
            for il in range(L):
                if len(TL[il]) < N:
                    raise ValueError('Problem with the input NTS with length < input data')

    if Mn is not None:
        Mn_model, theta_n = Mn
        # To BE IMPLEMENTED
        # try:
        #     x_Mn = Mn_generate(time, Mn_model, theta_n, nseries=1)
        # except:
        #     raise ValueError('Problem with the input Mn model: impossible to generate the synthetic dataset.')
        
    if Md is not None:
        Md_model, theta_d_ini, theta_d_bounds = Md
        try:
            x_Md = Md_estimate(x, Md_model, c, theta_d_ini, theta_d_bounds)
        except:
            raise ValueError('Problem with the input Md model: impossible to generate the synthetic dataset. ')

    if c is not None and len(c) != N:
        raise ValueError('Problem with the length of the ancillary time series.')
   
    # Core function ===========================================================
    res = np.copy(rv)
    
    # l.1 - if there is a nuisance component in data
    if Md is not None:
        
        # l.2 - Estimate the parameters of model Md on x (and c if c)
        hat_theta_d, delta_d = Md_estimate(x, Md_model, c, theta_d_ini, theta_d_bounds )

        # l.3 - Compute the data residuals
        model = Md_generate(time, Md_model, hat_theta_d, c)
        res  = res - model
        
    
    # l.4 - Compute the periodogram
    fnum, numerator = periodogram(time*24*3600, res, Ptype=Ptype, freq_grid = freq_grid)
    
    # l.5 to 9 - If there is an NTS or Mn is not None
    if TL is not None or Mn is not None: 
        _, denominator = averag_periodo(time*24*3600, TL, len(fnum), Ptype=Ptype, freq_grid = freq_grid)
        
    # To be tested with Example 2
    # if TL is None and Mn is not None: 
    #     hat_TL = Mn_generate(time, Mn_model, theta_n, nseries=L)
    #     _, denominator = averag_periodo(time, hat_TL, Ptype=Ptype, freq_grid = freq_grid)
        
    # if TL is None and Mn is None:
    #     hat_sig2 = np.var(res)
    #     _, denominator = np.zeros(N) + hat_sig2
    
    # l.10 - Compute the standardized periodogram
    pstand = numerator / denominator

    # l.11 - Apply the detection test
    ftest, test = compute_test(fnum, pstand, Ttype)
        
    # check
    if check:
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(time, rv,'k.',label='data')
        if Md is not None:     plt.plot(time, model,'r',label='model Md')
        # plt.plot(time,c,'m.')
        plt.ylabel('RV [m/s]')
        plt.legend()
        plt.subplot(212)
        plt.plot(time, res,'k.')
        plt.xlabel('time')
        plt.ylabel('residuals for numerator [m/s]')
    
        plt.figure()
        plt.subplot(211)
        plt.loglog(fnum, numerator,'c',label='numerator')
        plt.loglog(fnum, denominator,'b',label='denominator')
        plt.ylabel('periodograms')
        plt.legend()    
        plt.subplot(212)
        plt.semilogx(fnum, numerator,'c',label='un-standardized periodogram')
        plt.semilogx(fnum, pstand,'k',label='standardized periodogram')
        plt.xlabel('frequencies')
        plt.ylabel('un-standardized vs. standardized periodogram')
        plt.legend()    

    return ftest, test, hat_theta_d, delta_d