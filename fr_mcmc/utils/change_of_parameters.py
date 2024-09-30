'''
Change of parameters used in the numeric integrarion
'''

from scipy.constants import c as c_luz # meters/seconds
import numpy as np
c_luz_km = c_luz/1000

# Parameters order: omega_m, b, H_0, n

#GILA model
def F_H(H, params, model='GILA'):     
    lamb, L, beta, L_bar = params
    if model == 'GILA':
        FH = H**2 - H**6 * L_bar**4 * beta * np.exp(-beta*(L_bar*H)**10) \
                + H**8 * L**6     * lamb * np.exp(lamb*(L*H)**2)
    elif model == 'BETA':
        FH = H**2 - H**2        * beta * np.exp(-beta*(L_bar*H)**8) \
        + H**8 * L**6 * lamb * np.exp(lamb*(L*H)**4)

    return FH

def F_H_prime(H, params, model='GILA'):
    lamb, L, beta, L_bar = params

    if model == 'GILA':
        aux = np.exp(-beta*(L_bar*H)**10) * beta * (L_bar*H)**4 * (-3 + 5 * beta * (L_bar*H)**10) +\
                np.exp(lamb*(L*H)**2) * lamb * (L*H)**6 * (4 + lamb*(L*H)**2)
    if model == 'BETA':
        aux = np.exp(-beta*(L_bar*H)**8)  * beta                * (-1 + 4 * beta * (L_bar*H)**8) +\
            2 * np.exp(lamb*(L*H)**4) * lamb * (L*H)**6 * (2 + lamb*(L*H)**4)

    FH_prime = 2 * H * (1 + aux) 
    return FH_prime

def omega_luisa_to_CDM(beta, L_bar, H0, Omega_m_luisa):
    factor = F_H(H0, [0, 1e-27/H0, beta, L_bar/H0]) / H0**2
    omega_cdm = Omega_m_luisa * factor
    return omega_cdm

def omega_CDM_to_luisa(beta, L_bar, H0, Omega_lcdm):
    factor_inv = H0**2 / F_H(H0, [0, 1e-27/H0, beta, L_bar/H0]) 
    omega_luisa = Omega_lcdm * factor_inv
    return omega_luisa

#%%
if __name__ == '__main__':
    # Hu-Sawicki
    omega_m_true = 0.24
    b_true = 2
    H_0=73.48

    aux = c_luz_km**2 * omega_m_true / (7800 * (8315)**2 * (1-omega_m_true)) 
    print(aux)