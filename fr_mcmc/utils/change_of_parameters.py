'''
Change of parameters used in the numeric integrarion
'''

import numpy as np

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from constants import LAMBDA, L

#GILA model
def F_H(H, params, model):     
    lamb, L, beta, L_bar = params # L and L_bar have to be in units of H0^{-1}
    if model == 'GILA':
        FH = H**2 - H**6 * L_bar**4 * beta * np.exp(-beta*(L_bar*H)**10) \
                  + H**8 * L**6     * lamb * np.exp(lamb*(L*H)**2)
    elif model == 'BETA':
        FH = H**2 - H**2        * beta * np.exp(-beta*(L_bar*H)**8) \
                  + H**8 * L**6 * lamb * np.exp(lamb*(L*H)**4)

    return FH

def F_H_prime(H, params, model):
    lamb, L, beta, L_bar = params # L and L_bar have to be in units of H0^{-1}

    if model == 'GILA':
        aux = np.exp(-beta*(L_bar*H)**10) * beta * (L_bar*H)**4 * (-3 + 5 * beta * (L_bar*H)**10) +\
              np.exp(lamb*(L*H)**2)       * lamb * (L*H)**6     * (4 + lamb*(L*H)**2)
    if model == 'BETA':
        aux =     np.exp(-beta*(L_bar*H)**8) * beta             * (-1 + 4 * beta * (L_bar*H)**8) +\
              2 * np.exp(lamb*(L*H)**4)      * lamb * (L*H)**6  * (2 + lamb*(L*H)**4)

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