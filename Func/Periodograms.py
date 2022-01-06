#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

FUNCTIONS RELATED TO PERIODOGRAMS

@author: ssulis
@author contact: sophia.sulis@lam.fr
"""

#import pyfits
from numpy import *
import numpy as np

from astropy.stats import LombScargle
import matplotlib.pyplot as plt

from PyAstronomy.pyTiming import pyPeriod

#%% ===========================================================================
# Compute the periodogram of a time series y
# t = date of observations
# y = input time series
# Ptype = periodogram type: 'FFT', 'Lomb-Scargle', 'GLS' --> other type can be added
# freq_grid = frequency grid to compute the periodogram (defaut: None). units = [t]^-1
# nfac = factor to oversample the frequency grid (defaut: 1)
# Outputs: frequency and periodogram
# =============================================================================

def periodogram(t, y, Ptype='FFT', freq_grid=None, nfac=1):
    
    # Schuster's periodogram based on FFT
    if Ptype == 'FFT': 
        dt = np.min(t[1:]-t[:-1])
        freq  = np.linspace(-1./2/dt, 1./2/dt, len(t)) 
        y -= np.mean(y)
        periodo = 1.0/len(y) * np.abs(np.fft.fftshift(np.fft.fft(y)))**2 

    # Lomb-Scargle periodogram based on astropy library
    if Ptype == 'Lomb-Scargle':
        y -= np.mean(y)
        if freq_grid is None:
            ls  = LombScargle(t, y)
            freq, periodo  = ls.autopower(nyquist_factor=nfac,normalization='psd')
        else:
            freq = np.copy(freq_grid)
            periodo = LombScargle(t, y).power(freq_grid)

    # Generalized Lomb-Scargle periodogram based on PyAstronomy library
    if Ptype =='GLS':
        if freq_grid is None:
            clp = pyPeriod.Gls((t,y), norm='ZK',freq=freq_grid)
        else:
            clp = pyPeriod.Gls((t,y), norm='ZK',freq=freq_grid)
        freq, periodo = clp.freq, clp.power
       
    # You can implement here other periodogram functions
    # [...]
                 
    return freq, periodo  

#%% ===========================================================================
# Compute the averaged periodogram of a set of L training series yl
# t = date of observations
# yl = set of L training series yl (size = [L,len(t)])
# Ptype = periodogram type: 'FFT', 'Lomb-Scargle', 'GLS' --> other type can be added
# freq_grid = frequency grid to compute the periodogram (defaut: None). units = [t]^-1
# nfac = factor to oversample the frequency grid (defaut: 1)
# Outputs: frequency and averaged periodogram
# =============================================================================

def averag_periodo(time, yl, n, Ptype='FFT', freq_grid=None, nfac=1):    

    L = len(yl)
    
    p = np.zeros((L,n))
    for il in range(L):
        f, p[il] = periodogram(time, yl[il], Ptype=Ptype, freq_grid = freq_grid, nfac=nfac)
    PL = np.mean(p,0)
    return f, PL


