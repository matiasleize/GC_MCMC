'''
Change of parameters used in the numeric integrarion
'''

from scipy.constants import c as c_luz # meters/seconds
import numpy as np
c_luz_km = c_luz/1000

# Parameters order: omega_m, b, H_0, n
def F_H(H, params):     
    lamb, L, beta, L_bar = params
    #FH = H**2 #Caso LCDM
    #FH = H**2 + H**8 * (lamb * L**6 * np.exp(lamb*(L*H)**4) - beta * L_bar**6 * np.exp(-beta*(L_bar*H)**2))
    FH = H**2 + H**8 * lamb * L**6 * np.exp(lamb*(L*H)**4) -  H**2 * beta * np.exp(-beta*(L_bar*H)**8)
    return FH

def omega_luisa_to_CDM(beta, L_bar, H0, omega_m_luisa):
    factor = F_H(H0, [0, 1e-27/H0, beta, L_bar/H0]) / H0**2
    omega_cdm = omega_m_luisa * factor
    return omega_cdm

def omega_CDM_to_luisa(beta, L_bar, H0, omega_lcdm):
    factor_inv = H0**2 / F_H(H0, [0, 1e-27/H0, beta, L_bar/H0]) 
    omega_luisa = omega_lcdm * factor_inv
    return omega_luisa

#%%
if __name__ == '__main__':
    # Hu-Sawicki
    omega_m_true = 0.24
    b_true = 2
    H_0=73.48

    aux = c_luz_km**2 * omega_m_true / (7800 * (8315)**2 * (1-omega_m_true)) 
    print(aux)