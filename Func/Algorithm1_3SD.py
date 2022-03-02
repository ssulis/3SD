#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

ALGORITHM 1 = SEMI-SUPERVISED STANDARDIZED DETECTION (3SD) PROCEDURE

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

    if freq_grid is not None and len(freq_grid)<N:
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
        Mn_model, theta_n_ini, theta_n_bounds, L = Mn
        try:
            xnew = time*24*3600, rv, rv_err
            x_Mn = Mn_estimate(xnew, Mn_model, theta_n_ini, theta_n_bounds)
        except:
            raise ValueError('Problem with the input Mn model: impossible to generate the synthetic dataset.')
        
    if Md is not None:
        Md_model, theta_d_ini, theta_d_bounds = Md
        try:
            x_Md = Md_estimate(x, Md_model, c, theta_d_ini, theta_d_bounds)
        except:
            raise ValueError('Problem with the input Md model: impossible to generate the synthetic dataset. ')
    else:
        hat_theta_d, delta_d = None, None
        
    if c is not None and len(c) != N:
        raise ValueError('Problem with the length of the ancillary time series.')
   
    # Core function ===========================================================
    res = np.copy(rv)
    
    # l.1 - if there is a nuisance component d in data
    if Md is not None:
        
        # l.2 - Estimate the parameters of model Md on x (and c if c)
        hat_theta_d, delta_d = Md_estimate(x, Md_model, c, theta_d_ini, theta_d_bounds )

        # l.3 - Compute the data residuals
        model = Md_generate(time, Md_model, hat_theta_d, c)
        model_tot = np.copy(model)
        res  = res - model
    
    # l.4 - Compute the periodogram
    fnum, numerator = periodogram(time*24*3600, res, Ptype=Ptype, freq_grid = freq_grid)
    
    # l.5 to 10 - If there is a stochastic colored noise n
    # if NTS
    if TL is not None: 
        # l.17 - compute data for denominator
        _, denominator = averag_periodo(time*24*3600, TL, len(fnum), Ptype=Ptype, freq_grid = freq_grid)
        
    # if no NTS
    hat_theta_n = None
    if Mn is not None: 
        # l.9 - Estimate the parameters of model Mn on the residuals of the rv
        xres = time*24*3600, res, rv_err
        hat_theta_n    = Mn_estimate(xres, Mn_model, theta_n_ini, theta_n_bounds )
        # l.10 - compute data for denominator
        hat_TL         = Mn_generate(time, Mn_model, hat_theta_n, nseries=L)
        model_tot += hat_TL[0]
        
        _, denominator = averag_periodo(time*24*3600, hat_TL, len(fnum), Ptype=Ptype, freq_grid = freq_grid)
        
    # l.11 to 13 - If there is a stochastic white noise n    
    if TL is None and Mn is None:
        hat_sig2 = np.var(res)
        denominator = np.zeros(len(numerator)) + hat_sig2
    
    # l.15 - Compute the standardized periodogram
    pstand = numerator / denominator

    # l.16 - Apply the detection test
    ftest, test = compute_test(fnum[fnum>0][:10], pstand[fnum>0][:10], Ttype)
        
    # check
    if check:
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(time, rv,'k.',label='data')
        if Md is not None:     plt.plot(time, model_tot,'r',label='model noises')
        # plt.plot(time,c,'m.')
        plt.ylabel('RV [m/s]')
        plt.legend()
        plt.subplot(212)
        plt.plot(time, res,'k.')
        plt.xlabel('time')
        plt.ylabel('residuals for numerator [m/s]')
    
        # for visibility plot periodograms without the first frequencies
        w = np.where(numerator>1e-10) # you can comment this line

        fig2 = plt.figure()        
        plt.subplot(211)
        plt.loglog(fnum[w], numerator[w],'c',label='numerator')
        plt.loglog(fnum[w], denominator[w],'b',label='denominator')
        plt.ylabel('periodograms')
        plt.legend()    
        plt.subplot(212)
        plt.loglog(fnum[w], numerator[w],'c',label='un-standardized periodogram')
        plt.plot(fnum[w], pstand[w],'k',label='standardized periodogram')
        plt.xlabel('frequencies')
        plt.ylabel('periodograms')
        plt.legend()    

    if check:
        return ftest, test, hat_theta_d, delta_d, hat_theta_n, fig, fig2
    else:
        return ftest, test, hat_theta_d, delta_d, hat_theta_n
        
