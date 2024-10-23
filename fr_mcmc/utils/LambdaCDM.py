"""
Functions related to LCDM model
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000
#%%

def E_LCDM(z, Omega_m):
    '''
    Calculation of the normalized Hubble parameter, independent
    of the Hubble constant H0.
    '''
    Omega_lambda = 1 - Omega_m
    E = np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)
    return E

def H_LCDM(z, Omega_m, H_0):
    '''
    Calculation of the Hubble parameter.
    Here we neclect the radiation (it holds 
    that \Omega_r + \Omega_m + \Omega_L = 1).
    '''
    H = H_0 * E_LCDM(z, Omega_m)
    return H

def H_LCDM_rad(z, Omega_m, H_0):
    '''
    Calculation of the Hubble parameter. Here it holds that
    \Omega_r + \Omega_m + \Omega_L = 1 
    '''
    Omega_r = 4.18343*10**(-5) / (H_0/100)**2
    Omega_lambda = 1 - Omega_m - Omega_r

    if isinstance(z, (np.ndarray, list)):
        H = H_0 * np.sqrt(Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + Omega_lambda)
    else:
        H = H_0 * (Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + Omega_lambda)**(1/2)

    return H