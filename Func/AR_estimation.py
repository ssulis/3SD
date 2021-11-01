#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
FONCTIONS RELATED TO AUTOREGRESSIVE PROCESS
- AR NOISE GENERATION
- EVALUATION OF THEORETICAL Power Spectral Density (PSD)
- ESTIMATION OF AR ORDER AND PARAMETERS
- CERIA FOR AR ORDER SELECTION

'''

from numpy import *
import numpy as np

from math import *

from numpy.linalg import inv

#%%***************************************************************************#
# Generation AR noise process
# INPUTS: 
# coefs:   Is the array with coefficients, e.g. a=array([a1,a2]).
# sigma:   The white noise (zero-mean normal in this case) standard deviation.
# n:       Number of points to generate.
# OUTPUT : AR time series + Measured standard deviation

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
  
  #sdev = sqrt(var(w[burnin:]))      
  return AR[burnin:]


#%%***************************************************************************#
# Theoretical PSD of AR noise
# INPUTS: 
# coefs:     Is the array with coefficients.
# sigma:    The white noise (zero-mean normal in this case) standard deviation.
# n:        Number of points to generate.
# OUTPUT : Theoretical PSD of AR noise

def AR_DSPth( pfunc, N):
    
    coefs,sigma2 = pfunc
    sigma        = sigma2**0.5

    o = len(coefs)
    filtre = np.zeros(N)
    filtre[0] = 1.
    for i in range(1, o+1):
        filtre[i] = -coefs[i-1]
    DSP_th = np.fft.fftshift(1.0*(abs(np.fft.fft(filtre)))**2);
    DSP_th = sigma2**0.5/DSP_th;
    return DSP_th
    
#%%***************************************************************************#
# Estimation of AR order, coefficients and  estimated pred. error var.
# Inputs : Input time series (X), cut-off order (max) and AR critetion
# Outputs : Best order, estimated pred. error var. (Varp) and coefs (alpha)

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
    
    #print '\ncrit=',crit
    #print 'order=',order
    
    # Coefficients and  estimated pred. error var.
    Varp, alp = Prediction_error_power_estimate(X, order)
    alpha = -np.array(alp.T)[0]
    
    return [alpha,Varp]

#%%***************************************************************************#
# Criterion to chose the best order
    
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

#%%***************************************************************************#
# Evaluation of  estimated pred. error var.

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
    #print (Cxx=', Cxx_matrix)

    # Matrix's index. Note: Cxx(-i)=Cxx(i) 
    ind_Matrix = [[0 for i in range(p)] for j in range(p)]
    for ligne in range(0,p):                  # line (2 : p), col(:)
        i=ligne;       
        for col in range(0,p):            # line (i), col(col)
            ind_Matrix[ligne][col] = np.abs(i) 
            i -= 1
            #print ('\nind_Matrix=',ind_Matrix)

	# Fill matrix M_Cxx
    M_Cxx = np.zeros((p,p))
    for j in range(p):
        M_Cxx[j] = np.mat([ Cxx[ind_Matrix[i][j]][0] for i in range(p)])
    M_Cxx = np.mat(M_Cxx)
    #print ('\nM_Cxx=',M_Cxx)
    
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

