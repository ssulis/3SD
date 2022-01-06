#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FUNCTIONS RELATED TO THE NUISANCE NOISE COMPONENT "d" of Eq.(1)

@author: ssulis
@author contact: sophia.sulis@lam.fr
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import lmfit
from lmfit import minimize, Parameters, fit_report 

import warnings
warnings.filterwarnings("ignore")


#%%***************************************************************************#
# Generate an estimate of the nuisance noise component based on model Md_model
# and parameters theta_d
# time = dates of the observations or NTS time sampling
# Md_model = chosen model for the nuisance noise component 
# IC = ancillary time series involved in model Md_model (e.g., logR'HK indexes)
# Output: One synthetic time series following model Md_model
#*****************************************************************************#

def Md_generate(time, Md_model, theta_d, IC):

    N = len(time) # number of data points
    x_Md = np.zeros(N) # initialize: output time series
    
    # If Md_model is chosen as beta*IC + 3rd order polynomial function of time
    # (=model taken as example in Example1)
    if Md_model == 'Model_Md_Example1':
        x_Md = Md_Example1(theta_d, time, IC)

    # If Md_model is chosen as beta*IC 
    # (=model taken as example in Example2)
    if Md_model == 'Model_Md_Example2':
        x_Md = Md_Example2(theta_d, time, IC)
        
    return x_Md

#%%***************************************************************************#
# Estimate the parameters of the nuisance noise component model (Md_model)
# x = dates of the observations, data, errors on data points
# Md_model = chosen model for the nuisance noise component 
# IC = ancillary time series involved in model Md_model (e.g., logR'HK indexes)
# params_ini, par_bounds = initial parameters and bounds
# Output: Estimated parameters and confidence intervals
#*****************************************************************************#

def Md_estimate(x, Md_model, IC, params_ini, par_bounds ):

    time, rv, rv_err = x # data
    N  = len(time)       # number of data points
    
    theta_d = []         # initilize: estimated parameters
    
    # If Md_model is chosen as beta*IC + 3rd order polynomial function of time
    # (=model taken as example in Example1)    
    if Md_model ==  'Model_Md_Example1':               
        
        param_ini = set_dict_params_Md_Example1(params_ini, par_bounds)
        out = minimize(residuals_Md_Example1, param_ini, args = (time, IC, rv, rv_err))        
        # print(out.message)
        # print(out.params)

        beta_fit = out.params.get('beta').value  
        a_fit = out.params.get('a').value
        b_fit = out.params.get('b').value 
        c_fit = out.params.get('c').value  
        theta_d = [beta_fit,a_fit,b_fit,c_fit]               
        
        beta_fit_err = out.params.get('beta').stderr
        a_fit_err    = out.params.get('a').stderr
        b_fit_err    = out.params.get('b').stderr
        c_fit_err    = out.params.get('c').stderr        
        delta_d = [beta_fit_err, a_fit_err, b_fit_err, c_fit_err]

    # If Md_model is chosen as beta*IC + 3rd order polynomial function of time
    # (=model taken as example in Example1)    
    if Md_model ==  'Model_Md_Example2':               
        
        param_ini = set_dict_params_Md_Example2(params_ini, par_bounds)
        out = minimize(residuals_Md_Example2, param_ini, args = (time, IC, rv, rv_err))        
        # print(out.message)
        # print(out.params)

        beta_fit = out.params.get('beta').value  
        theta_d = [beta_fit]               
        
        beta_fit_err = out.params.get('beta').stderr      
        delta_d = [beta_fit_err]
        
    # You can implement here other parametric noise
    # [...]
        
    return theta_d, delta_d

#%%***************************************************************************#
#  Functions related to model "Md_Example1"
#*****************************************************************************#

# set dictionary of parameters
def set_dict_params_Md_Example1(par_input,par_bounds):
    
    beta, a, b, c = par_input
    beta_min, beta_max, a_min, a_max, b_min, b_max, c_min, c_max = par_bounds
    param_output = Parameters()
    param_output.add('beta', value=beta, min=beta_min, max=beta_max)
    param_output.add('a', value=a, min=a_min, max=a_max)
    param_output.add('b', value=b, min=b_min, max=b_max)
    param_output.add('c', value=c, min=c_min, max=c_max)
    
    return  param_output

# model "Md_Example1" 
def Md_Example1(params, t, IC):
    beta, a, b, c = params
    model = beta*IC + a*t**2 + b*t + c
    return model

# residuals to be used in lmfit    
def residuals_Md_Example1(param_tofit, t, IC, y, yerr):
    
    beta, a, b, c = param_tofit
    beta = param_tofit.get('beta').value
    a = param_tofit.get('a').value
    b = param_tofit.get('b').value      
    c = param_tofit.get('c').value      
    params = [beta, a, b, c]
    
    model = Md_Example1(params,t,IC)
    err    = (y-model)/yerr
    return err

#%%***************************************************************************#
# Generation of synthetic ancillary data time series
#*****************************************************************************#
def generate_synthetic_c(time, c, model, theta_d):
    
    N = len(time)   
    c_ij = np.zeros(N)
    
    if model == 'Model_Md_Example1' or model == 'Model_Md_Example2':
        std_c = np.std(c-theta_d[0]*c) # this nuisance model involved c as c*beta 
        c_ij = np.random.normal(0, std_c, N)
        
    return c_ij

#%%***************************************************************************#
#  Functions related to model "Md_Example2"
#*****************************************************************************#

# set dictionary of parameters
def set_dict_params_Md_Example2(par_input,par_bounds):
    beta = par_input[0]
    beta_min, beta_max = par_bounds
    param_output = Parameters()
    # print(beta,beta_min, beta_max)
    param_output.add('beta', value=beta, min=beta_min, max=beta_max)
    return  param_output

# model "Md_Example2" 
def Md_Example2(beta, t, IC):
    model = beta*IC
    return model

# residuals to be used in lmfit    
def residuals_Md_Example2(param_tofit, t, IC, y, yerr):
    
    beta = param_tofit.get('beta').value    
    model = Md_Example2(beta,t,IC)
    err    = (y-model)/yerr
    return err




#%%***************************************************************************#
#  Functions for other noise models
#*****************************************************************************#

# You can implement here any other parametric noise functions you want following
# the set of functions decribed above
# [...]
 