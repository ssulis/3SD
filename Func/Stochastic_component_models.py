#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FUNCTION RELATED TO THE STOCHASTIC NOISE SOURCE COMPONENT "n" in Eq.(1) of the paper

Implemented models are: 
    - autoregressive noises
    - Harvey functions
    
@author: ssulis
@author contact: sophia.sulis@lam.fr

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from math import *
from numpy.linalg import inv

import lmfit
from lmfit import minimize, Parameters, fit_report 

# My functions
from Periodograms import *

import warnings
warnings.filterwarnings("ignore")


#%%***************************************************************************#
# Generate an estimate of the stochastic noise component based on model Mn_model
#*****************************************************************************#
'''
# Inputs
# and parameters theta_n
# time = dates of the observations or NTS time sampling
# nseries = number of time series to be generated
# Output: nseries>=1 synthetic time series following model Mn_model
'''
 
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
#*****************************************************************************#
'''
# Inputs
# x = dates of the observations or NTS time sampling if TL, data or data residuals, errors on data points 
# Mn_model = chosen model for the stochastic noise component 
# params_ini, par_bounds = initial parameters and bounds
# Output: Estimated parameters 
'''
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
#  Functions related to autoregressive noise processes
#*****************************************************************************#


#******************************
# Generation AR noise process
#******************************
'''
# INPUTS: 
# coefs:   Is the array with coefficients, e.g. a=array([a1,a2]).
# sigma:   The white noise (zero-mean normal in this case) standard deviation.
# n:       Number of points to generate.
# OUTPUTS : AR time series + Measured standard deviation 
'''
def ARgenerator(pfunc,n,burnin=0):

  coefs,sigma2 = pfunc
  sigma = sigma2**0.5
  
  if(burnin==0): burnin=100*len(coefs)       # Burn-in elements!
  w=np.random.normal(0,sigma,n+burnin)
  AR=array([])
  s=0.0
  for i in range(n+burnin):
      if(i<len(coefs)):
        AR=append(AR,w[i])
      else:
        s=0.0
        for j in range(len(coefs)):
            s=s+coefs[j]*AR[i-j-1]
        AR=append(AR,s+w[i])
  
  return AR[burnin:]


#******************************
# Theoretical PSD of AR noise
#******************************
'''
# INPUTS: 
# coefs:     Is the array with coefficients.
# sigma:    The white noise (zero-mean normal in this case) standard deviation.
# n:        Number of points to generate.
# OUTPUT : Theoretical PSD of AR noise
'''

def AR_DSPth( pfunc, N):
    
    coefs,sigma2 = pfunc
    sigma = sigma2**0.5

    o = len(coefs)
    filtre = np.zeros(N)
    filtre[0] = 1.
    for i in range(1, o+1):
        filtre[i] = -coefs[i-1]
    DSP_th = np.fft.fftshift(1.0*(abs(np.fft.fft(filtre)))**2);
    DSP_th = sigma**2/DSP_th
    return DSP_th
    
#******************************
# Estimation of AR order, coefficients and  estimated pred. error var.
#******************************
'''
# Inputs : Input time series (X), cut-off order (max) and AR critetion
# Outputs : coefficients (alpha) and estimated pred. error var. (Varp) 
'''

def AR_estimation(X, argfunc):

    pmax,criter = argfunc

    N = len(X);
    X = X - mean(X)		# remove mean
    
    # Order estimation
    crit = np.zeros(pmax)
    for p in range(1,pmax+1):        
        Varp, alp = Prediction_error_power_estimate(X, p)  
        if criter != 'CAT':          
            crit[p-1] = ARcriterion(criter, Varp, N, p)
        else:
            crit[p-1] = ARcriterion(criter, Varp, N, p,X)
                
    order = np.where(crit == min(crit))[0]
    order = np.min(order)+1
    
    # Coefficients and  estimated pred. error var.
    Varp, alp = Prediction_error_power_estimate(X, order)
    alpha = -np.array(alp.T)[0]
    
    return [alpha,Varp**0.5]

#******************************
# Criterion to chose the best order
#******************************

'''
# Implemented options are FPR, CAT, AIC and RIS
'''
   
def ARcriterion(criter, Varp, N, p, X=[]):
    
    if criter =='FPE':
        return Varp * ( (N+p+1)/(N-p-1) )

    if criter =='CAT':
        som=0
        for j in range(1,p):
            Varp_j, _ = Prediction_error_power_estimate(X, p)  
            som+= (N-j)/(N*Varp_j)
        return 1.0/N * som - (N-p)/(N*Varp)

    if criter =='AIC':
        return np.log(Varp) + 2*(p+1)/N

    if criter =='RIS':
        return Varp * ( 1 + (p+1)/N*np.log(N) )

#******************************
# Estimated pred. error var.
#******************************

def Prediction_error_power_estimate(X, p):
        
    N = len(X)
    X = X - mean(X)		# remove mean
    
    # Autocovariance matrix (cf. p.244 of Akaike, 1969)    
    Cxx = [0 for i in range(p+1)]             
    for il in range(0,p+1):
        somme = 0.0
        for i in range(1,N-il):
            somme += X[i+il] * X[i]
        Cxx[il]  = [1.0/N * somme]   
        
    Cxx_matrix =    np.mat(Cxx)

    # Matrix's index. Note: Cxx(-i)=Cxx(i) 
    ind_Matrix = [[0 for i in range(p)] for j in range(p)]
    for ligne in range(0,p):              # line (2 : p), col(:)
        i=ligne;       
        for col in range(0,p):            # line (i), col(col)
            ind_Matrix[ligne][col] = np.abs(i) 
            i -= 1

	# Fill matrix M_Cxx
    M_Cxx = np.zeros((p,p))
    for j in range(p):
        M_Cxx[j] = np.mat([ Cxx[ind_Matrix[i][j]][0] for i in range(p)])
    M_Cxx = np.mat(M_Cxx)
    
    # Evaluate coefficients alpha_i with inverse of M_Cxx
    alpha =  inv(M_Cxx)*Cxx[1:]
    #print (alpha)
    
    model = np.zeros(N)
    model[:p] = X[:p]
    
    m = np.arange(0,p)  # i1 = 1:p
    for nn in range(p,N):
        i2 = sort(nn - m)
        model[nn] = np.sum(np.array(alpha.T) * X[i2])

    Var_p = 1.0/N/4 * np.sum((X[p:N] + model[p:N])**2)
   
    return Var_p, -alpha


#%%***************************************************************************#
#  Function related to Harvey functions
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

# Harvey function (model the noise power spectral density with Harvey profiles)
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

# Generate synthetic time series with Harvey functions
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




