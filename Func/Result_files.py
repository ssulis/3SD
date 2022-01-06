#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FUNCTIONS RELATED TO SAVE/READ FILES INVOLVED IN ALGORITHM 2

@author: ssulis
@author contact: sophia.sulis@lam.fr

"""

import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#%%***************************************************************************#
# Save the intermediate results of Algorithm 2:
# tests, corresponding test's frequencies,  test levels, p-values
#*****************************************************************************#
    
def save_individual_runs(save_path, test_ij, ftest_ij, t, pvalue):
    
    ii = 0
    filesave =  save_path + "outputs_intermediate_b="+str(ii)+".txt" 
    while os.path.isfile(filesave) is True:
        ii+=1; 
        filesave = save_path + "outputs_intermediate_b="+str(ii)+".txt" 
                         
    b = len(test_ij)
    
    fileW = open(filesave, "w");
    fileW.write('# Test output:\n')
    for j in range(b): fileW.write("%f " %(test_ij[j]))
    fileW.write("\n#\n# Corresponding test's frequencies [muHz]:\n")
    for j in range(b): fileW.write("%f " %(ftest_ij[j]*1e6))
    fileW.write("\n#\n# test levels:\n")
    for j in range(len(t)): fileW.write("%f " %(t[j]))
    fileW.write("\n#\n# p-values:\n")
    for j in range(len(pvalue)): fileW.write("%f " %(pvalue[j]))
    fileW.close()
    
    return

#%%***************************************************************************#
# Read the intermediate results of Algorithm 2:
# OUTPUTS: tests, corresponding test's frequencies, test levels, p-values
#*****************************************************************************#
    
def read_individual_runs(save_path,b):
    

    nfiles = len(glob.glob1(save_path,"outputs_intermediate_b=*.txt"))
    
    test_ij, ftest_ij, t, pvalue = [np.zeros((nfiles, b)) for _ in range(4)]
    counter=0
    for k in range(nfiles):
        filename = save_path + "outputs_intermediate_b="+str(k)+".txt" 
                  
        if os.path.isfile(filename):
            # nlig = sum(1 for line in open(filename))
            fileR = open(filename, "r")
            
            fileR.readline()
            ligne = fileR.readline()
            adump = ligne.split()
            test_ij[counter]   = [float(adump[m]) for m in range(len(adump))]

            fileR.readline(); fileR.readline();
            ligne = fileR.readline()
            adump = ligne.split()
            ftest_ij[counter]   = [float(adump[m]) for m in range(len(adump))]
                    
            fileR.readline(); fileR.readline();
            ligne = fileR.readline()
            adump = ligne.split()
            t[counter]   = [float(adump[m]) for m in range(len(adump))]
            
            fileR.readline(); fileR.readline();
            ligne = fileR.readline()
            adump = ligne.split()
            pvalue[counter]   = [float(adump[m]) for m in range(len(adump))]
            counter+=1
            fileR.close()
   
    return test_ij[:counter], ftest_ij[:counter], t[:counter], pvalue[:counter]


#%%***************************************************************************#
# Save the final results of Algorithm 2+
# inputs +  test_mean, pvalue_mean
#*****************************************************************************#

        
def save_results_Algo2(save_path, inputs, t_mean, pvalue_mean):
        
    x, Ptype, Ttype, freq_grid, B, b, L, Mn, theta_n, sig2, delta_sig2, Md, theta_d, delta_d, c = inputs

    filesave =  save_path + "Algorithm2_final_outputs.txt" 
    fileW = open(filesave, "w");

    fileW.write('# Final results of Algorithm 2:\n')
    fileW.write('#\n')
    
    fileW.write('# Inputs parameters ================\n')
    fileW.write('#\n')
    
    fileW.write('# Length of input RV dataset: %d \n'%len(x[0]))
    fileW.write('# Periodogram type: %s \n'%Ptype)
    fileW.write('# Test name: %s \n'%Ttype)
    
    text = 'yes' if freq_grid != None else None
    fileW.write('# Does a frequency grid was provided? %s \n'%text)
    
    fileW.write('# MC size B: %d \n'%B)
    fileW.write('# MC size b: %d \n'%b)
    
    
    if Mn is None: 
        fileW.write('# Does a model Mn was provided ? No\n')
        fileW.write('# Mn input parameters: N/A\n')    
    else:
        Mn_model = Mn[0]
        fileW.write('# Does a model Mn was provided ? Yes, model = %s\n'%Mn_model)
        fileW.write('# Mn input parameters: ')
        tn  = str(theta_n)
        tn = tn.replace('\n',''); tn = tn.replace(' ',''); tn = tn.replace(',',', ')
        fileW.write(tn)
        fileW.write('\n')
        
    if sig2 is not None:
        fileW.write('# Does a sig2 value was provided? Yes (value = %f)\n'%(sig2))
    else:
        fileW.write('# Does a sig2 value was provided? No\n')
           
    if Md is None: 
        fileW.write('# Does a model Md was provided ? No\n')
        fileW.write('# Md input parameters: N/A\n')    
    else:
        Md_model, theta_d_ini, theta_d_bounds = Md
        fileW.write('# Does a model Md was provided ? Yes, model = %s\n'%Md_model)
        fileW.write('# Md input parameters: [')
        for item in theta_d_ini: fileW.write("%f," %item)
        fileW.write(']\n')
        text = 'yes' if c is not None else None
        fileW.write('# Does an ancillary dataset was provided? %s\n'%text)

    
    fileW.write('#\n')           
    fileW.write('# Results ================\n')
    fileW.write('#\n')           

    fileW.write("# test levels:\n")
    for j in range(len(t_mean)): fileW.write("%f " %(t_mean[j]))
    fileW.write("\n#\n# P-value:\n")
    for j in range(len(pvalue_mean)): fileW.write("%f " %(pvalue_mean[j]))

    fileW.close()     
    
    return

#%%***************************************************************************#
# Read the final results of Algorithm 2+
# Outputs:  gam_mean, pvalue_mean
#*****************************************************************************#

        
def read_results_Algo2(save_path):
        

    filename =  save_path + "Algorithm2_final_outputs.txt" 
                  
    if os.path.isfile(filename):
        
        fileR = open(filename, "r");
        
        [fileR.readline() for _ in range(20)]
        ligne = fileR.readline()
        adump = ligne.split()
        t_mean  = [float(adump[m]) for m in range(len(adump))]

        fileR.readline(); fileR.readline();
        ligne = fileR.readline()
        adump = ligne.split()
        pvalue_mean   = [float(adump[m]) for m in range(len(adump))]
                
        fileR.close()
   
    return t_mean, pvalue_mean

          


