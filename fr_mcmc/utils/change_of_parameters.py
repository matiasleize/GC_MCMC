'''
Change of parameters used in the numeric integrarion
'''

import numpy as np

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from constants import LAMBDA, L


def F_H(H, params, model):     

    #Fixed exponents
    r = 3
    s = 5
    ################

    lamb, L, beta, L_bar = params # L and L_bar have to be in units of H0^{-1}

    if model == 'GILA':
        lamb = 0; p = 3; q = 1
        FH = H**2 - beta * H**(2*r) * L_bar**(2*(r-1)) * np.exp(-beta*(L_bar*H)**(2*s)) #\
                  #+ lamb * H**(2*p) * L**(2*(p-1))     * np.exp(lamb*(L*H)**(2*q))

    elif model == 'BETA':
        lamb = 0; p = 1; q = 2; r = 1
        FH = H**2 - beta * H**2                        * np.exp(-beta*(L_bar*H)**(2*s)) #\
                  #+ lamb * H**(2*p) * L**(2*(p-1))     * np.exp(lamb*(L*H)**(2*q))

    return FH


def F_H_prime(H, params, model):

    #Fixed exponents
    r = 3
    s = 5
    ################

    lamb, L, beta, L_bar = params # L and L_bar have to be in units of H0^{-1}
   
    if model == 'GILA':
        lamb = 0; p = 3; q = 1
        aux = beta * np.exp(-beta*(L_bar*H)**(2*s)) * (L_bar*H)**(2*(r-1)) * (-r + s * beta * (L_bar*H)**(2*s)) #+\
              #lamb * np.exp(lamb*(L*H)**(2*q))      * (L*H)**(2*(p-1))     * (p  + q * lamb * (L*H)**(2*q))

    elif model == 'BETA':
        lamb = 0; p = 1; q = 2; r = 1
        aux = beta * np.exp(-beta*(L_bar*H)**(2*s))                        * (-1 + s * beta * (L_bar*H)**(2*s)) #+\
              #lamb * np.exp(lamb*(L*H)**(2*q))      * (L*H)**(2*(p-1)) * (p  + q * lamb * (L*H)**(2*q))

    FH_prime = 2 * H * (1 + aux) 
    return FH_prime

def omega_luisa_to_CDM(beta, L_bar, H0, Omega_m_luisa, model):
    factor = F_H(H0, [LAMBDA, L/H0, beta, L_bar/H0], model) / H0**2
    omega_cdm = Omega_m_luisa * factor
    return omega_cdm

def omega_CDM_to_luisa(beta, L_bar, H0, Omega_lcdm, model):
    factor_inv = H0**2 / F_H(H0, [LAMBDA, L/H0, beta, L_bar/H0], model) 
    omega_luisa = Omega_lcdm * factor_inv
    return omega_luisa

#%%
if __name__ == '__main__':
    pass