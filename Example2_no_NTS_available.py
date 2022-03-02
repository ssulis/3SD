#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

reference paper: Sulis et al. submitted to A&A
                 Semi-supervised standardized detection of extrasolar planets


This code shows how to implement the Algorithms 1 and 2 of the paper with an example.

Example 2:
    
Input datasets: synthetic RV dataset described in Sec. 5.5 of the paper

Studied configuration:
    - NO null training sample (NTS) of the stochastic noise avalaible (TL == emptyspace)
    - NEED for a parametric model of the stochastic noise source to be estimated on the dataset (M_n != emptyspace)
    - an ancillary data for stellar activity is available (c = logR'HK)
                                                           
Credits: S. Sulis, D. Mary , L. Bigot, and M. Deleuil
@author contact: sophia.sulis@lam.fr

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from astropy import stats
from astropy.stats import LombScargle
from astropy.io import fits

# My functions
sys.path.append("Func/")

from Periodograms import *
from Stochastic_component_models import *
from Nuisance_component_models import *
from Detection_tests import *
from AR_estimation import *
from Result_files import *

from Algorithm1_3SD import *
from Algorithm2_pvalue import *

#%%***************************************************************************#
# Load input RV time series
# These data have been used in Sec.5 of Sulis et al. (2021), submitted to A&A
#*****************************************************************************#

filename = 'Files/synthetic_rv_data.txt'
N = sum(1 for line in open(filename))-1
time, rv, rv_err, logRHK = [np.zeros(N) for _ in range(4) ]
fileR = open(filename, "r");
header = fileR.readline().split(',');
for i in range(N):
    adump       = fileR.readline().split()
    time[i]     = float(adump[0]) # days
    rv[i]       = float(adump[1])
    rv_err[i]   = float(adump[2])
    logRHK[i]   = float(adump[3])
fileR.close()

print('The input RV time series spans %f days and contains %d points'%((time[-1]-time[0]), N))

plt.figure(figsize=(10,9))
plt.subplot(211)
plt.errorbar(time,rv,yerr=rv_err,fmt='k.',label='data')
plt.ylabel('RV [m/s]')
plt.subplot(212)
plt.plot(time,logRHK,'k.')
plt.ylabel("log. R'HK ")
plt.xlabel("time [days] ")
plt.savefig('Outputs/Example2/Fig1_Input_dataset.png')
       
#%%***************************************************************************#
# Evaluate the test statistic on this dataset: procedure 3SD 
#*****************************************************************************#

# DEFINE INPUTS

x     = [time, rv, rv_err]          # the serie under test
Ptype = 'FFT'              # The selected periodogram type, defaut: 'Lomb-Scargle'  
Ttype = 'Max test'                  # The selected detection test to be applied to the periodogra, defaut: 'Max test'
freq_grid = None                    # the considered set of frequencies, defaut:None

# If TL not ∅ 
TL    = None      # optional: the null training sample sampled as the observed RV, defaut: None

# If Mn not ∅ 
model_n        = 'Harvey'      # Chosen parametric model for stochastic noise, defaut: None
theta_n_ini =   [30, 210, 0.5, 2.5] # input parameters of the model 'Harvey': params = [beta, a_harvey, b_harvey,  WGN, index_harvey]
a_min, a_max = 0,100
b_min, b_max = 100,500
WGN_min, WGN_max = 0.5, 10
index_min, index_max  =1,4
theta_n_bound = [a_min, a_min, a_max, b_min, b_max,  WGN_min, WGN_max,  index_min, index_max] # parameter bounds
L  = 5                                          # number of synthetic time series to be generated for the averaged periodogram, best is is L >>1
Mn = [model_n, theta_n_ini, theta_n_bound, L]  # optional: parameters of the stochastic signal, defaut: None

# If Md not ∅ 
model_d       = 'Model_Md_Example2'                    # Chosen parametric model for activity noise: Md = beta*logR'HK
beta_ini = 0.8                                         # Input parameters for model Md
theta_d_ini   = [beta_ini]                             
beta_min, beta_max = -5,5                              # Bound parameters for model Md
theta_d_bound = [beta_min, beta_max]                   
Md            = [model_d, theta_d_ini, theta_d_bound]  # optional: model of the nuisance signal, defaut: None

# If Md not ∅ if Md(c)
c = np.copy(logRHK)                  # optional: activity indicators time series, defaut: None

# RUN 3SD PROCEDURE
check = True
output_Algo1 = Algorithm1_3SD(x, 
                              Ptype = Ptype, 
                              Ttype = Ttype, 
                              freq_grid=freq_grid,
                              TL = TL,
                              Mn = Mn,
                              Md = Md,
                              c  = c,
                              check = check)

if check:
    freq_test, test_value, hat_theta_d, delta_d, hat_theta_n, sig2, fig, fig2 = output_Algo1
    fig.savefig('Outputs/Example2/Fig2_3SD_procedure.png')
    fig2.savefig('Outputs/Example2/Fig3_3SD_procedure.png')
else:
    freq_test, test_value, hat_theta_d, delta_d, hat_theta_n, sig2 = output_Algo1

print('Results from Algorithm 1 for time series under test is t = %.2f'%test_value)
print('It correspond to frequency f = %.2f muHz(period = %.2f days) '%(freq_test*1e6, 1.0/freq_test/3600/24))

if model_d:
    print("\nEstimated Md parameters fitted on input dataset:")
    print("Model: %s: "%model_d)
    print("Estimated parameters:",hat_theta_d)
    print("Perturbation interval:", delta_d)

if model_n:
    print("\nEstimated Mn parameters fitted on input dataset:")
    print("Model: %s: "%model_n)
    print("Estimated parameters:",hat_theta_n)
          

#%%***************************************************************************#
# Evaluate the p-value of Algorithm 1  (Algorithm 2)
#*****************************************************************************#
"""
this loop can take several hours to compute (for B=1 and b=1000 on classical laptop ~ 25-30 min)
we advice to test it first for low (B,b) values 
we also advise to save iteratively the results in file (see 'Outputs/') to avoid
loosing all calculations in case of technical problems
"""

# DEFINE INPUTS
B, b = 100, 1000                   # Number of Monte Carlo simulations, typical values are 100 and 1000, respectively

# If Mn not ∅ :
theta_n = hat_theta_n            # optional: estimated parameters, defaut=None
time_NTS = np.copy(time) # because there is no NTS in this example Mn is fitted on the observed rv
grid_NTS = np.arange(len(time))*0+1 # all one because no NTS
Mn = [model_n, time_NTS, grid_NTS, hat_theta_n, theta_n_bound] 

# If Mn = ∅ :
sig2       = None               # estimated variance of WGN, defaut: None 
delta_sig2 = None               # perturbation interval for hat_sig2, defaut: None

# If Md = ∅ :
priors = 'uniform' # Priors to randomly select the theta_d +/- delta_d parameters, defaut: 'uniform'

# Path to save the results
save_path = 'Outputs/Example2/Algorithm2/'
if not os.path.exists(save_path):os.makedirs(save_path)

# RUN 3SD PROCEDURE
outputs_Algo2  = Algorithm2_pvalue(x, Ptype, Ttype,
                            freq_grid=freq_grid,
                            B=B, b=b,
                            Mn = Mn, theta_n=hat_theta_n, L=L,
                            sig2=sig2, delta_sig2=delta_sig2,
                            Md = Md, theta_d=hat_theta_d, delta_d=delta_d, priors=priors,
                            c  = c,
                            save_path = save_path,
                            check = False)

t_mean, pvalue_mean = outputs_Algo2

#%%***************************************************************************#
# Plot the P-values estimates obtained from Algorithm 2
#*****************************************************************************#

from Result_files import *

test_ij, ftest_ij, test_t, pvalue = read_individual_runs(save_path, b)
t_mean, pvalue_mean = read_results_Algo2(save_path)

plt.figure()

plt.loglog(test_t[0], pvalue[0],'0.75', label='P-value from Algorithm 2')
for i in range(1,B): plt.loglog(test_t[i], pvalue[i],'0.75')
plt.loglog(t_mean, pvalue_mean,'k-', label='mean P-value from Algorithm 2')

tmp = np.abs(t_mean-test_value)
wtmp = np.where(tmp==min(tmp))[0][0]
plt.plot(test_value, pvalue_mean[wtmp],'ro', label="Observed test's value")
plt.plot([test_value, test_value], [min(pvalue_mean),pvalue_mean[wtmp]],'r--')
plt.plot([min(t_mean),test_value], [pvalue_mean[wtmp],pvalue_mean[wtmp]],'r--')
plt.legend()

plt.xlabel(r'$t$')
plt.ylabel(r'$P$-value')
# plt.xlim([min(t_mean), max(t_mean)])    
plt.ylim([1e-3,1.1])    

plt.savefig('Outputs/Example2/Fig4_Algorithm2_outputs.png')

