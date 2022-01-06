#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FUNCTION RELATED TO THE STOCHASTIC NOISE SOURCE COMPONENT "n" in Eq.(1) of the paper
@author: ssulis
@author contact: sophia.sulis@lam.fr

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import lmfit
from lmfit import minimize, Parameters, fit_report 

# My functions
from AR_estimation import *
from Periodograms import *

import warnings
warnings.filterwarnings("ignore")


#%%***************************************************************************#
# Generate an estimate of the stochastic noise component based on model Mn_model
# and parameters theta_n
# time = dates of the observations or NTS time sampling
# nseries = number of time series to be generated
# Output: nseries>=1 synthetic time series following model Mn_model
#*****************************************************************************#
    
def Mn_generate(time, Mn_model, theta_n,  nseries=1, delmoy='y'):
    
    N  = len(time)       # number of data points

    # If Mn_model is chosen as an autoregressive process
    # (=model taken as example in Example1)    
    if Mn_model == 'AR':         
        if nseries == 1: 
            x_Mn = ARgenerator(theta_n, N)
        else:
            x_Mn = np.zeros((nseries, N))
            for il in range(nseries): x_Mn[il] = ARgenerator(theta_n, N)
    
    # If Mn_model is chosen as an Harvey profile
    # (=model taken as example in Example2)    
    if Mn_model == 'Harvey':      
        if nseries == 1: 
            x_Mn = Harvey_generator(time,theta_n, N, delmoy=delmoy)
        else:
            x_Mn = np.zeros((nseries, N))
            for il in range(nseries): x_Mn[il] = Harvey_generator(time,theta_n, N, delmoy=delmoy)
 
    # You can implement here other parametric noise models
    # [...]
        
    return x_Mn


#%%***************************************************************************#
# Estimate the parameters of the stochastic noise component based on model Mn_model
# x = dates of the observations or NTS time sampling if TL, data or data residuals, errors on data points 
# Mn_model = chosen model for the stochastic noise component 
# params_ini, par_bounds = initial parameters and bounds
# Output: Estimated parameters 
#*****************************************************************************#

def Mn_estimate(x, Mn_model, params_ini, par_bounds ):

    time, rv, rv_err = x # input data
    N  = len(time)       # number of data points
    
    hat_theta_n = []     # initilize: estimated parameters
    
    # If Mn_model is chosen as an autoregressive process
    # (=model taken as example in Example1)    
    if Mn_model == 'AR':
        
        pmax = 10   # max order: defaut is 10
        criterion = 'FPE' # AR estimation criterion: defaut is 'FPE' 
        hat_theta_n = AR_estimation(rv, [pmax,criterion])

    # If Mn_model is chosen as an Harvey profile
    # (=model taken as example in Example2)    
    if Mn_model == 'Harvey':      
        
        param_ini = set_dict_params_Mn_Harvey(params_ini, par_bounds)
        out = minimize(residuals_Mn_Harvey, param_ini, args = (time*24*3600, rv))        
        # print(out.message)
        # print(lmfit.fit_report(out.params))

        a_harvey = out.params.get('a_harvey').value
        b_harvey = out.params.get('b_harvey').value 
        index_harvey = out.params.get('index_harvey').value          
        WGN = out.params.get('WGN').value          
        hat_theta_n = [a_harvey, b_harvey,WGN, index_harvey]             

    # You can implement here other parametric noise models
    # [...]
        
    return hat_theta_n

#%%***************************************************************************#
#  Function related to the Harvey profile
#*****************************************************************************#

# set dictionary of parameters
def set_dict_params_Mn_Harvey(par_input,par_bounds):
    
    a_harvey, b_harvey, WGN, index_harvey= par_input
    a_min, a_min, a_max, b_min, b_max,  WGN_min, WGN_max,  index_min, index_max = par_bounds
    param_output = Parameters()
    param_output.add('a_harvey', value=a_harvey, min=a_min, max=a_max)
    param_output.add('b_harvey', value=b_harvey, min=b_min, max=b_max)
    param_output.add('WGN', value=WGN, min=WGN_min, max=WGN_max)
    param_output.add('index_harvey', value=index_harvey, min=index_min, max=index_max)
    
    return  param_output

# model Harvey (DSP)
def harveyfunc(params,freq):
    a, b, sig2, index = params
    zeta = 2*np.sqrt(2)/np.pi
    model = (zeta*a**2/b) / (1+(freq/b)**index) + sig2
    return model

# residuals to be used in lmfit    
def residuals_Mn_Harvey(param_tofit, t, y):
    
    a_harvey, b_harvey, WGN, index_harvey = param_tofit
    a = param_tofit.get('a_harvey').value
    b = param_tofit.get('b_harvey').value      
    w = param_tofit.get('WGN').value 
    i = param_tofit.get('index_harvey').value     
    params = [a,b,w,i]

    # fit of the Harvey profile is made on the periodogram
    f, periodo   = periodogram(t, y, Ptype='FFT', freq_grid=None, nfac=1)
    fpos_muHz = f[f>0]*1e6
    model = harveyfunc(params,fpos_muHz)
    err   = periodo[f>0]-model
    return err

# Generate syntehtic time series with Harvey function
def Harvey_generator(time,theta_n, N,  delmoy='y'):

    # Create Harvey DSP
    serie_fake = np.zeros(N)+1
    treg =  np.linspace(time[0], time[-1],N)
    f, _   = periodogram(treg, serie_fake, Ptype='FFT', freq_grid=None, nfac=1)
    fpos_muHz  = f[f>=0]*1e6
 
    target_DSP             = np.zeros(len(f))
    target_DSP[:len(fpos_muHz)] = harveyfunc(theta_n, fpos_muHz)[::-1]
    if np.mod(len(f),2) == 0: target_DSP[len(fpos_muHz):] = harveyfunc(theta_n, fpos_muHz) # cas pair
    if np.mod(len(f),2) == 1: target_DSP[len(fpos_muHz)-1:] = harveyfunc(theta_n, fpos_muHz) # cas impair
    
    # square root of the target PSD
    G=np.sqrt(target_DSP)#*(1/np.sqrt(dt))     
    G[np.isinf(G) == True] = np.amax(G[np.isinf(G) == False])
    
    #  scaled white gaussian noise, PSD~constant
    n=len(target_DSP)
    sigma0 = 1
    huc = sigma0*np.random.normal(0,1,n) 
    
    # FFT
    TFh = np.fft.fftshift((np.fft.fft(huc)*G))
    
    # Imaginary part (about 1e-18 or so) 
    h   = np.fft.ifft(TFh)
    
    # come back to temporal domain and take the real part
    ysynth=np.real(h)
    
    # remove mean (optional)
    if delmoy=='y': ysynth-=np.mean(ysynth)
    
    return ysynth[:len(time)]




