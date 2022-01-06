#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALGORITHM 2 = EVALUATION OF P-VALUES AND CREDIBLE INTERVALS RELATED TO THE 3SD PROCEDURE

@author: ssulis
@author contact: sophia.sulis@lam.fr
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tqdm import *
  
from statsmodels.distributions.empirical_distribution import ECDF

# My functions
from Periodograms import *
from Stochastic_component_models import *
from Nuisance_component_models import *
from Detection_tests import *
from Algorithm1_3SD import *
from Result_files import *

import warnings
warnings.filterwarnings("ignore")

#%%***************************************************************************#
# Function which randomly select values x in interval +/- x_err
# [lim1, lim2] are the boundaries 
#*****************************************************************************#
def rselect(x, x_err, priors = 'uniform', lim1 = np.nan, lim2 = np.nan):
    
    if priors not in ["uniform", "normal"]: 
        raise ValueError('You need to indicate "priors" with ["uniform", "normal"] or implement a new one.')
    
    born1 = x - x_err
    born2 = x + x_err
    if lim1 != np.nan: 
        if x - x_err < lim1: born1 = 0
    if lim2 != np.nan: 
        if x + x_err > lim2: born2 = 100
        
    if priors == "uniform": x_ij = random.uniform(born1, born2)
    if priors == "normal":   x_ij = random.normal(loc=x, scale=x_err)
    
    return x_ij

#%%***************************************************************************#
# Algorithm 2: Evaluate the significance interval (p-values) of the 3SD procedure
# see Sec.3 of Sulis et al., 2021, submitted to A&A
#*****************************************************************************#
    

def Algorithm2_pvalue(x,
               Ptype = 'Lomb-Scargle',
               Ttype = 'Max test', 
               freq_grid=None,
               B=100, b=1000,
               Mn=None, theta_n=None,L=1,
               sig2=None, delta_sig2=None,
               Md = None, theta_d=None, delta_d=None, priors = 'uniform',
               c  = None,
               save_path = 'Outputs/',
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
        
    if B<1 or b<1:
        raise ValueError('The bootstrap size (B,b) should be >1')

    if Mn is not None:
        Mn_model, time_NTS, grid_NTS, theta_n_ini, theta_n_bounds = Mn
        try:
            x_Mn = Mn_generate(time_NTS, Mn_model, theta_n, nseries=1)
        except:
            raise ValueError('Problem with the input Mn model: impossible to generate the synthetic dataset')
        
    if sig2 is not None:
        if sig2<0: raise ValueError('WGN variance (sig2) should be > 0')
        if delta_sig2 is None: raise ValueError('You need to indicate a confidence interval for the WGN variance')
    if sig2 is None and delta_sig2 is not None: raise ValueError('Parameters delta_d should be None')

        
    if Md is not None:
        Md_model, theta_d_ini, theta_d_bounds = Md
        try:
            x_Md = Md_generate(time, Md_model, theta_d_ini, c) 
            x_Md = Md_generate(time, Md_model, theta_d, c) 
        except:
            raise ValueError('Problem with the Md model: impossible to generate the synthetic dataset. ')
        if theta_d is None or delta_d is None : raise ValueError('You need to indicate the estimated parameters of Md (+ their confidence intervals)')

    if c is not None:
        if len(c) != N: raise ValueError('Problem with the length of the ancillary time series.')

   
    # Core function ===========================================================

    # l.1 - First bootstrap loop, aim: generate p-values distribution ==============
    for i in range(B): # loop
        
        print ('\nAlgo2: run %d / %d'%(i,B))
        
        # l.2 - If there is a stochastic colored noise n # ===============================
        if Mn is not None:
            
             # l.3 - Generate L noise time series T^{i} with model Mn(theta_n)
             x_Mn = Mn_generate(time_NTS, Mn_model, theta_n, nseries=1)

             # l.4 - Fit T^{i} and generate new estimates hat_theta_n
             x_NTS = time_NTS, x_Mn, rv_err
             hat_theta_n = Mn_estimate(x_NTS, Mn_model, theta_n_ini, theta_n_bounds)
             # print("hat_theta_n=",hat_theta_n)
             
        # l.5 - Second bootstrap loop, aim: generate 1 p-value  # =================
        ftest_ij, test_ij  = np.zeros(b), np.zeros(b) # initialize test outputs arrays

        pbar = tqdm(total=b) # Init pbar

        for j in range(b): # loop
            
            pbar.update(n=1) # Increments counter
            
            y_ij = np.zeros(N) # initialize training series
                 
            # l.6 - If there is a stochastic colored noise n # ===========================
            if Mn is not None: 
                 
                  # l.7 - Generate (L+1) noise training series T^{i,j} with model Mn(hat_theta_n)
                  x_Mn = Mn_generate(time_NTS, Mn_model, hat_theta_n, nseries=L+1)
                 
                  # l.8- Take one T^{i,j} and implement vector y_ij
                  # the NTS has been generated with regular sampling grid --> take the sampling grid of the observations                 
                  if len(x_Mn[0])!=N: x_Mn = x_Mn[:, grid_NTS==1]
                  y_ij += x_Mn[0]                
                  
                  TL_ij = x_Mn[1:]
            
            # l.9 - If there is a stochastic white noise n # ===============================================
            else: 

                # l.10 - Randomly select the WGN variance in interval
                sig2_ij = rselect(sig2, delta_sig2)
                
                # l.11 - Generate a synthetic WGN and implement vector y_ij
                y_ij = np.random.normal(0, sig2_ij, N)
                
                TL_ij = None

            # l.12 - If there is a nuisance component in data # ===============
            if Md is not None:
                 
                  # l.13 - Randomly select hat2_theta_d in interval
                  theta_d_ij = np.zeros(len(theta_d))
                  for k in range(len(theta_d)):
                      theta_d_ij[k] = rselect(theta_d[k], delta_d[k], priors)
                    
                  # l.14 - Generate hat_d with model Md and new parameters hat2_theta_d
                  #      - then, implement vector y_il
                  x_Md = Md_generate(time, Md_model, theta_d_ij, c)
                  #  If c, Generate a synthetic indicators series cij
                  if c is not None: 
                      c_ij = generate_synthetic_c(time, c, Md_model, theta_d_ij)
                  else:
                      c_ij = None

                  # l.15 - Add it to the synthetic time series under test
                  y_ij += x_Md
                   
            # # l.16 - Compute the test statistics with procedure 3DS ===========
            x_ij = time, y_ij, rv_err 
            
            output = Algorithm1_3SD(x_ij, Ptype=Ptype, Ttype=Ttype, freq_grid=freq_grid, 
                                    TL = TL_ij, Mn = None, Md = Md, c = c_ij, check=False)        
            
            ftest_ij[j] = output[0]
            test_ij[j]  = output[1]
            
        pbar.close(); del pbar 
        
        # l.17 - Compute the test's CDF using test_ij =========================
        ecdf     = ECDF(test_ij) 
        test_t   = ecdf.x[1:]
        cdf_test =  ecdf.y[1:]

        # l.18 - Compute the p-value estimate =====================================
        pvalue =  1.0-cdf_test

        # If savefile is True, save the test_ij[i,:] generated random variables
        save_individual_runs(save_path, test_ij, ftest_ij, test_t, pvalue)

    # l.19 - Compute the mean of the p-value estimates ============================
    test_ij, ftest_ij, test_t, pvalue = read_individual_runs(save_path, b)
    t_mean =  np.mean(test_t,0)
    pvalue_mean = np.mean(pvalue,0)

    # Save final results in a file
    inputs = [x, Ptype, Ttype, freq_grid, B, b,  L, Mn, 
              theta_n, sig2, delta_sig2, Md, theta_d, delta_d, c]
        
    save_results_Algo2(save_path, inputs, t_mean, pvalue_mean)
        
            
    return t_mean, pvalue_mean
               
               