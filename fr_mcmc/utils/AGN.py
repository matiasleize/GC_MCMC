"""
Functions related to AGN data.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz

from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000;

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
#from int import Hubble_th
from solve_sys import Hubble_th
from LambdaCDM import H_LCDM

#Parameters order: omega_m,b,H_0,n

def zs_2_logDlH0(INT,z_data):
    DlH0 = (c_light_km * (1 + z_data)) * INT #km/sec
    return np.log10(DlH0) #log(km/sec)

### Nuisance AGN
def Hs_2_logDl(zs,Hs,z_data):
    INT = cumtrapz(Hs**(-1), zs, initial=0)

    Dl = (c_light_km * (1 + zs)) * INT #Mpc
    output = interp1d(zs,Dl)
    return np.log10(output(z_data)) #log(Mpc)

def Hs_2_logDlH0(zs,Hs,z_data):
    INT = cumtrapz(Hs**(-1), zs, initial=0)

    Dl = (c_light_km * (1 + zs)) * H0 * INT #Mpc
    output = interp1d(zs,Dl)
    return np.log10(output(z_data)) #log(Mpc)



def chi2_AGN_nuisance(teo, data, errors_cuad):
    chi2 = np.sum( ((data-teo)**2/errors_cuad) + np.log(2*np.pi*errors_cuad))
    return chi2

def params_to_chi2_AGN_nuisance(theta, fixed_params, dataset_AGN, n=1,
                                num_z_points=int(10**6), model='LCDM'
                                ,less_z=False,all_analytic=False):
    '''
    Given the model parameters, it returns the statistics chi squared
    for the AGN data.
    '''
    #Here we define the parameters
    if model == 'LCDM':
        if isinstance(theta,float):
            #print(theta)
            omega_m = theta
            [beta, gamma, delta, H_0] = fixed_params
            zs_model = np.linspace(0,10,10**5)
            Hs_model = H_LCDM(zs_model,omega_m,H_0)


        else:
            if len(theta) == 4:
                [omega_m, beta, gamma, delta] = theta #This beta is different from the other
                H_0 = fixed_params
                zs_model = np.linspace(0,10,10**5)
                Hs_model = H_LCDM(zs_model,omega_m,H_0)


    else:
        if len(theta) == 5:
            [omega_m, b, beta, gamma, delta] = theta #This beta is different from the other
            H_0 = fixed_params
        elif len(theta) == 4:
            [omega_m, b] = theta
            [beta, gamma, delta, H_0] = fixed_params #This beta is different from the other

        physical_params = [omega_m,b,H_0]
        zs_model, Hs_model = Hubble_th(physical_params, n=n, model=model,
                                    z_min=0, z_max=10, num_z_points=num_z_points,
                                    all_analytic=all_analytic)


    #Import data
    z_data_unmasked, logFuv_unmasked, eFuv_unmasked, logFx_unmasked, eFx_unmasked  = dataset_AGN

    if less_z == True:
        mask = z_data_unmasked<1.5
        z_data = z_data_unmasked[mask]
        logFuv = logFuv_unmasked[mask]
        eFuv = eFuv_unmasked[mask]
        logFx = logFx_unmasked[mask]
        eFx = eFx_unmasked[mask]
    else:
        z_data = z_data_unmasked
        logFuv = logFuv_unmasked
        eFuv = eFuv_unmasked
        logFx = logFx_unmasked
        eFx = eFx_unmasked


    Dl_teo = Hs_2_logDl(zs_model,Hs_model,z_data) #Mpc
    Dl_teo_cm = Dl_teo - np.log10(3.24) + 25
    psi = beta + gamma * logFuv + 2 * (gamma-1) * (Dl_teo_cm + 0.5 * np.log10(4*np.pi))

    si_2 = eFx**2 + gamma**2 * eFuv**2 + np.exp(2*np.log(delta)) #Squared errors

    chi2_AGN = chi2_AGN_nuisance(psi, logFx, si_2)

    return chi2_AGN

#%%
if __name__ == '__main__':
    #Data AGN
    from data import read_data_AGN
    from matplotlib import pyplot as plt
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    data_agn = read_data_AGN('table3.dat')

    beta_true =  6.8 #7.735
    gamma_true = 0.648
    delta_true = 0.235
    H0_true =  70


    for j,beta_true in enumerate(np.linspace(7.9,8.5,20)):
        fixed_params = [beta_true, gamma_true, delta_true, H0_true]
        omegas = np.linspace(0,1,50)
        chi_2 = np.zeros(50)
        for i,omega_m in enumerate(omegas):
            chi_2[i] = params_to_chi2_AGN_nuisance(omega_m, fixed_params,
                                data_agn, model='LCDM')
        plt.figure()
        plt.plot(omegas, chi_2)
        plt.grid()
        plt.title(r'$\beta$ = {}'.format(beta_true))
        plt.ylabel('$\chi^{2}$')
        plt.xlabel('$\Omega_{m}$')
        plt.close()
