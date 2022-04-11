#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

reference paper: Sulis et al. submitted to A&A
                 Semi-supervised standardized detection of extrasolar planets


This code shows how to implement the Algorithms 1 and 2 of the paper with an example.

Example 1:
    
Input datasets: synthetic RV dataset described in Sec. 5.3 of the paper

Studied configuration:
    - a null training sample (NTS) of the stochastic noise is avalaible (TL != emptyspace)
    - no need for a parametric model of the stochastic noise source to be estimated on the dataset (Mn == emptyspace in Algo.1)
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
plt.subplot(311)
plt.errorbar(time,rv,yerr=rv_err,fmt='k.')
plt.ylabel('RV [m/s]')
plt.subplot(312)
plt.plot(time,logRHK,'k.')
plt.ylabel("log. R'HK ")


#%%***************************************************************************#
# Load the NTS series for granulation noise
# Here, the NTS corresponds to RV solar data generated with 3D simulations
# see Sulis, S., Mary, D., & Bigot, L. 2020, A&A, 635, A146 for details
#*****************************************************************************#
 
filename_NTS = "Files/3DMHD_solar_velocities.txt"

Nnts = sum(1 for line in open(filename_NTS))-1 # number of data point in the raw NTS file

fileR = open(filename_NTS, "r");

# Evaluate L, the number of NTS series avalaible
header = fileR.readline().split(',')
L = len(header)-2
print('There is L=%d NTS time series available'%L)

time_NTS = np.zeros(Nnts) 
rv_NTS   = np.zeros((L,Nnts))
grid_NTS = np.arange(Nnts)*0
for i in range(Nnts):
    adump       = fileR.readline().split()
    time_NTS[i]     = float(adump[0]) # regularly sampled dates [days]
    for il in range(L): rv_NTS[il,i] = float(adump[1+il])
    grid_NTS[i] = int(adump[-1]) # 1 if on the grid of the observed RV, 0 otherwise 
fileR.close()

# we sample the NTS time series as the observations (check where grid_NTS is "1")
time_NTS_obs = time_NTS[grid_NTS==1]
rv_NTS_obs   = rv_NTS[:,grid_NTS==1]

plt.subplot(313)
for il in range(L): plt.plot(time_NTS_obs/24/3600,rv_NTS_obs[il],'.')
plt.xlabel('time [days]')
plt.ylabel("NTS [m/s]")
plt.savefig('Outputs/Example1/Fig1_Input_dataset.png')

#%%***************************************************************************#
# Evaluate the test statistic on this dataset: procedure 3SD 
#*****************************************************************************#

# DEFINE INPUTS

x     = [time, rv, rv_err]          # the serie under test
Ptype = 'FFT'              # The selected periodogram type, defaut: 'Lomb-Scargle'  
Ttype = 'Max test'                  # The selected detection test to be applied to the periodogra, defaut: 'Max test'
freq_grid = None                    # the considered set of frequencies, defaut:None

# If TL not ∅ 
TL    = np.copy(rv_NTS_obs)         # optional: the null training sample sampled as the observed RV, defaut: None

# If Mn not ∅ 
Mn    = None                        # optional: stochastic noise component --> not considered in Example1, defaut: None

# If Md not ∅ 
model_d       = 'Model_Md_Example1'                    # Chosen parametric model for activity noise
theta_d_ini   = [0.17, -0.023, 0.14, -0.93]            # Input parameters for model Md
theta_d_bound = [-5.8, 5.8, -10, 10, -10, 10, -10, 10] # Bound parameters for model Md
Md            = [model_d, theta_d_ini, theta_d_bound]  # optional: model of the nuisance signal, defaut: None

# If Md not ∅ if Md(c)
c = np.copy(logRHK)                  # optional: activity indicators time series, defaut: None

# cut-ioff frequency for test Ttype
flim = 56e-6 # in muHz, defaut is None

# Send any additional arguments ?
argss = None # defaut is None

# RUN 3SD PROCEDURE
check = True
output_Algo1 = Algorithm1_3SD(x, 
                              Ptype = Ptype, 
                              Ttype = Ttype, 
                              flim = flim,
                              freq_grid=freq_grid,                           
                              TL = TL,
                              Mn = Mn,
                              Md = Md,
                              c  = c,
                              add_args = argss,
                              check = check)

if check:
    freq_test, test_value, hat_theta_d, delta_d, hat_theta_n, sig2, fig, fig2 = output_Algo1
    fig.savefig('Outputs/Example1/Fig2_3SD_procedure.png')
    fig2.savefig('Outputs/Example1/Fig3_3SD_procedure.png')
else:
    freq_test, test_value, hat_theta_d, delta_d, hat_theta_n, sig2 = output_Algo1


print('n\Results from Algorithm 1 for time series under test is t = %.2f'%test_value)
print('It correspond to frequency f = %.2f muHz (period = %.2f days) '%(freq_test*1e6, 1.0/freq_test/3600/24))

if model_d:
    print("\nEstimated Md parameters fitted on input dataset:")
    print("Model: %s: "%model_d)
    print("Estimated parameters:",hat_theta_d)
    print("Perturbation interval:", delta_d)
          

#%%===========================================================================#
# To compute Algorithm2 (= estimation of p-values of the 3SD procedure) with an NTS (TL):
# --> we need a parametric noise model to be able to generate L>5 NTS series with
# parametric bootstrap (see Sec.5 of the paper). 
#=============================================================================#

# We fit an autoregressive (AR) process to the NTS (see Sec. 5.3)
fileNTS = 'Files/Params_AR_MHDfit.dat' # file where we save the fitted AR process 

if os.path.isfile(fileNTS) is False: # generate the file just ones, to save time
    
    # fit AR
    pmax      = 10
    criterion = 'FPE'
    argfunc   = [pmax,criterion] 
    rvf       = rv_NTS[:L].flatten()
    rvf      -= mean(rvf)
    hat_theta_n = AR_estimation(rvf, argfunc)
    
    # save the file
    fileW = open(fileNTS, "w")
    for i in range(len(hat_theta_n[0])): fileW.write('%f\n' %(hat_theta_n[0][i])) 
    fileW.write('%f\n' %(hat_theta_n[1])) 
    fileW.close()
    
    print("\nEstimated Mn parameters fitted on TL:")
    print("Estimated AR order: %d"%len(hat_theta_n[0]))
    print("Estimated AR coefs:",hat_theta_n[0])
    print("Estimated AR std: %.2f\n"%hat_theta_n[1])
    

else: # to save time, we can load the results
    nparams = sum(1 for line in open(fileNTS))
    hat_theta_n = [np.zeros(nparams-1),0]
    fileR = open(fileNTS, "r");
    for i in range(nparams-1):
        lignes = fileR.readline(); 
        adump  = lignes.split(); 
        hat_theta_n[0][i] = float(adump[0]) # the AR coefficient
    lignes = fileR.readline(); adump  = lignes.split(); hat_theta_n[1]= float(adump[0]) # the AR variance
    fileR.close()

print("\nEstimated Mn parameters fitted on TL:")
print("Estimated AR order: %d"%len(hat_theta_n[0]))
print("Estimated AR coefs:",hat_theta_n[0])
print("Estimated AR std: %.2f\n"%hat_theta_n[1])
    
# check 
freq, PL  = averag_periodo(time_NTS*24*3600, rv_NTS, len(time_NTS), Ptype='FFT')
DSP_AR_th = AR_DSPth(hat_theta_n, len(time_NTS)) 

fig=plt.figure()
plt.loglog(freq*1e6,PL,'k',label='Averaged periodogram of NTS series')
plt.loglog(freq*1e6,DSP_AR_th,'r',label='Theoretical DSP of the AR estimated process')
plt.xlabel('Frequency [muHz]')
plt.ylabel('Periodogram [m2/s]')
plt.xlim([min(freq)*1e6, max(freq)*1e6])
plt.ylim([0.03,20])
plt.legend()
plt.savefig('Outputs/Example1/Fig4_Parametric_model_for_NTS.png')

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

x     = [time, rv, rv_err]          # the serie under test
Ptype = 'FFT'              # The selected periodogram type, defaut: 'Lomb-Scargle'  
Ttype = 'Max test'                  # The selected detection test to be applied to the periodogra, defaut: 'Max test'
freq_grid = None                     # the considered set of frequencies, defaut: None

B, b = 20,1000                   # Number of Monte Carlo simulations, typical values are 100 and 1000, respectively

# If Mn not ∅ :
model_n = 'AR'                   # optional: Chosen parametric model for NTS, defaut=None
theta_n = hat_theta_n            # optional: estimated parameters, defaut=None
L       = L                      # optional: number of series to be generated, defaut=1
theta_n_ini = None
theta_n_bounds = None
Mn = [model_n, time_NTS, grid_NTS, theta_n_ini, theta_n_bounds] 

# If Mn = ∅ :
sig2       = None               # estimated variance of WGN, defaut: None 
delta_sig2 = None               # perturbation interval for hat_sig2, defaut: None

# If Md not ∅ 
model_d       = 'Model_Md_Example1'                    # Chosen parametric model for activity noise
theta_d_ini   = [0.17, -0.023, 0.14, -0.93]            # Input parameters for model Md
theta_d_bound = [-5.8, 5.8, -10, 10, -10, 10, -10, 10] # Bound parameters for model Md
Md            = [model_d, theta_d_ini, theta_d_bound]  # optional: model of the nuisance signal, defaut: None

theta_d   = hat_theta_d        # Estimated parameters for model Md fitted on x (x=RV series under test)
delta_d  = delta_d             # perturbation interval for hat_theta_d

priors = 'uniform' # Priors to randomly select the theta_d +/- delta_d parameters, defaut: 'uniform'

# If Md not ∅ if Md(c)
c = np.copy(logRHK)     # optional: activity indicators time series, defaut: None

# Path to save the results
save_path = 'Outputs/Example1/Algorithm2/'
if not os.path.exists(save_path):os.makedirs(save_path)

# from Algorithm2_pvalue import *

# RUN 3SD PROCEDURE
outputs_Algo2  = Algorithm2_pvalue(x, Ptype, Ttype,
                            flim = flim,
                            freq_grid=freq_grid,
                            B=B, b=b,
                            Mn = Mn, theta_n=theta_n, L=L,
                            sig2=sig2, delta_sig2=delta_sig2,
                            Md = Md, theta_d=theta_d, delta_d=delta_d, priors=priors,
                            c  = c,
                            add_args = argss,
                            save_path = save_path,
                            check = True)

t_mean, pvalue_mean = outputs_Algo2

#%%***************************************************************************#
# Plot the P-values estimates obtained from Algorithm 2
#*****************************************************************************#
# from Result_files import *

test_ij, ftest_ij, test_t, pvalue = read_individual_runs(save_path, b)
t_mean, pvalue_mean = read_results_Algo2(save_path)

plt.figure()

plt.loglog(test_t[0], pvalue[0],'0.75', label='P-value from Algorithm 2')
if B>1:
    for i in range(1,B): plt.loglog(test_t[i], pvalue[i],'0.75')
plt.loglog(t_mean, pvalue_mean,'k-', label='mean P-value from Algorithm 2')

tmp = np.abs(t_mean-test_value)
wtmp = np.where(tmp==min(tmp))[0][0]
plt.plot(test_value, pvalue_mean[wtmp],'ro', label="Observed: t = %.2f, pval(t)=%.2f"%(test_value, pvalue_mean[wtmp]))
plt.plot([test_value, test_value], [min(pvalue_mean),pvalue_mean[wtmp]],'r--')
plt.plot([min(t_mean),test_value], [pvalue_mean[wtmp],pvalue_mean[wtmp]],'r--')
plt.legend()

plt.xlabel(r'$t$')
plt.ylabel(r'$P$-value')
# plt.xlim([min(t_mean), max(t_mean)])    
plt.xlim([min(t_mean), None])    
plt.ylim([1e-3,1.1])    

plt.savefig('Outputs/Example1/Fig5_Algorithm2_outputs.png')

#%%***************************************************************************#
# Check: 
# Verify if the test's distribution is close to a uniform distribution
# if this is not the case: see discussion in paper
#*****************************************************************************#

plt.figure()
plt.hist(ftest_ij.flatten(),bins=500) 
plt.ylabel('Count') 
plt.xlabel(r'Frequencies [$\mu$Hz]')
plt.savefig('Outputs/Example1/Fig6_Test_distribution.png')

