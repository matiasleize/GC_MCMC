"""
Definition of the log likelihood distribution and the chi_square in terms of
the parameters of the model and the datasets which are use. 
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir

os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from change_of_parameters import omega_CDM_to_luisa, omega_luisa_to_CDM
from solve_sys import Hubble_th
from supernovae import aparent_magnitude_th, chi2_supernovae
from BAO import r_drag, Hs_to_Ds, Ds_to_obs_final
from AGN import zs_2_logDlH0

def chi2_sin_cov(teo, data, errors_cuad):
    '''
    Calculate chi square assuming no correlation.

    teo (array): Theoretical prediction of the model.
    data (array): Observational data to compare with the model.
    errors_cuad (array): The square of the errors of the data.

    '''

    chi2 = np.sum((data-teo)**2/errors_cuad)
    return chi2

def all_parameters(theta, fixed_params, index):
    '''
    Auxiliary function that reads and organizes fixed and variable parameters into one 
    list according to the index criteria.

    theta: object with variable parameters.
    fixed_params: object with fixed parameters.
    index (int): indicate if the parameters are fixed or variable.
    
    '''

    if index == 5:
        [Mabs, L_bar, b, H_0, omega_m] = theta
        _ = fixed_params

    if index == 4:
        [Mabs, L_bar, b, H_0] = theta
        omega_m = fixed_params

    if index == 41:
        [Mabs, b, H_0, omega_m] = theta
        L_bar = fixed_params

    elif index == 31:
        [L_bar, b, H_0] = theta
        Mabs, omega_m = fixed_params

    elif index == 32:
        [Mabs, L_bar, H_0] = theta
        b, omega_m = fixed_params

    elif index == 33:
        [Mabs, L_bar, b] = theta
        H_0, omega_m = fixed_params

    elif index == 34:
        [Mabs, b, H_0] = theta
        L_bar, omega_m = fixed_params

    elif index == 21:
        [L_bar, b] = theta
        Mabs, H_0, omega_m = fixed_params

    elif index == 22:
        [L_bar, H_0] = theta
        Mabs, b, omega_m = fixed_params

    elif index == 23:
        [Mabs, L_bar] = theta
        b, H_0, omega_m = fixed_params

    elif index == 1:
        L_bar = theta
        Mabs, b, H_0, omega_m = fixed_params

    return [Mabs, L_bar, b, H_0, omega_m]


def params_to_chi2(theta, fixed_params, index=0,
                   dataset_SN_plus_shoes=None, dataset_SN_plus=None,
                   dataset_SN=None, dataset_CC=None,
                   dataset_BAO=None, dataset_DESI=None, dataset_AGN=None, H0_Riess=False,
                   num_z_points=int(10**5), model='HS',n=1,
                   nuisance_2 = False, enlarged_errors=False,
                   all_analytic=False):
    '''
    Given the free parameters of the model, return chi square for the data.
    
    theta: object with variable parameters.
    fixed_params: object with fixed parameters.
    index (int): indicate if the parameters are fixed or variable.

    dataset_SN:
    dataset_CC:
    dataset_BAO: This data goes up to z=7.4 aproximately. Don't integrate with z less than that!
    dataset_DESI:
    dataset_AGN:
    H0_Riess:

    num_z_points:
    model (str): cosmological model ('LCDM', 'HS', 'EXP').
    n (int): (1, 2)
    nuisance_2 (bool):
    enlarged_errors (bool):
    all_analytic (bool):
    '''

    chi2_SN = 0
    chi2_CC = 0
    chi2_BAO = 0
    chi2_DESI = 0
    chi2_AGN = 0
    chi2_H0 =  0

    #[Mabs, L_bar, b, H_0, omega_m] = all_parameters(theta, fixed_params, index)
    #omega_m_luisa = omega_CDM_to_luisa(b,L_bar,H_0,omega_m)

    [Mabs, L_bar, b, H_0, omega_m_luisa] = all_parameters(theta, fixed_params, index)
    omega_m_luisa = 0.9999 + 10**(-5) * omega_m_luisa
    omega_m = omega_luisa_to_CDM(b,L_bar,H_0,omega_m_luisa)
    #print(omega_m_luisa)

    physical_params = [L_bar,b,H_0,omega_m_luisa]
    zs_model, Hs_model = Hubble_th(physical_params, n=n, model=model,
                                z_min=0, z_max=10, num_z_points=num_z_points,
                                all_analytic=all_analytic)

    if (dataset_CC != None or dataset_BAO != None or dataset_DESI != None or dataset_AGN != None):
        Hs_interpolado = interp1d(zs_model, Hs_model)

    #if (dataset_SN != None or dataset_BAO != None or dataset_AGN != None):
    #    int_inv_Hs = cumtrapz(Hs_model**(-1), zs_model, initial=0)
    #    int_inv_Hs_interpolado = interp1d(zs_model, int_inv_Hs)

    if (dataset_SN_plus_shoes != None or dataset_SN_plus != None or
        dataset_SN != None or dataset_BAO != None or dataset_AGN != None):
        int_inv_Hs = cumtrapz(Hs_model**(-1), zs_model, initial=0)
        int_inv_Hs_interpol = interp1d(zs_model, int_inv_Hs)

    if dataset_SN_plus_shoes != None:
        zhd, zhel, mb, mu_shoes, Cinv, is_cal = dataset_SN_plus_shoes #Import the data
        muobs = mb - Mabs
        muth_num = aparent_magnitude_th(int_inv_Hs_interpol, zhd, zhel) #Numeric prediction of mu
        muth = muth_num*(-is_cal + 1) + mu_shoes*(is_cal) #Merge num predicion with mu_shoes
        chi2_SN = chi2_supernovae(muth, muobs, Cinv)

    if dataset_SN != None:
        zcmb, zhel, Cinv, mb = dataset_SN #Import the data
        muth = aparent_magnitude_th(int_inv_Hs_interpol, zcmb, zhel)
        muobs =  mb - Mabs
        chi2_SN = chi2_supernovae(muth, muobs, Cinv)

    if dataset_CC != None:
        z_data, H_data, dH = dataset_CC #Import the data
        H_teo = Hs_interpolado(z_data)
        chi2_CC = chi2_sin_cov(H_teo, H_data, dH**2)

    if dataset_BAO != None:
        num_datasets=5
        chies_BAO = np.zeros(num_datasets)
        for i in range(num_datasets): # For each datatype
            (z_data_BAO, valores_data, errores_data_cuad,wb_fid) = dataset_BAO[i]
            if i==0: #Da entry
                rd = r_drag(omega_m,H_0,wb_fid) # rd calculation
                distancias_teoricas = Hs_to_Ds(Hs_interpolado, int_inv_Hs_interpol, z_data_BAO, i)
                output_th = Ds_to_obs_final(zs_model, distancias_teoricas, rd, i)
            else: #If not..
                distancias_teoricas = Hs_to_Ds(Hs_interpolado, int_inv_Hs_interpol, z_data_BAO, i)
                output_th = np.zeros(len(z_data_BAO))
                for j in range(len(z_data_BAO)): # For each datatype
                     rd = r_drag(omega_m,H_0,wb_fid[j]) #rd calculation
                     output_th[j] = Ds_to_obs_final(zs_model,distancias_teoricas[j],rd,i)
            #Chi square calculation for each datatype (i)
            chies_BAO[i] = chi2_sin_cov(output_th,valores_data,errores_data_cuad)


        if np.isnan(sum(chies_BAO))==True:
            print('There are some errors!')
            print(omega_m,H_0,rd)

        chi2_BAO = np.sum(chies_BAO)

    if dataset_DESI != None:
        num_datasets=5
        chies_DESI = np.zeros(num_datasets)

        (set_1, set_2) = dataset_DESI
        z_eff_1, data_dm_rd, data_dh_rd, Cinv, wb_fid_1 = set_1 
        z_eff_2, data_dv_rd, errors_dv_rd, wb_fid_2 = set_2

    
        #index: 1 (DH)
        #index: 2 (DM)
        #index: 3 (DV)

        #DM_DH

        output_th_dh = np.zeros(len(z_eff_1))
        output_th_dm = np.zeros(len(z_eff_1))

        for j in range(len(output_th_dh)): # For each datatype
            rd = r_drag(omega_m, H_0, wb_fid_1) #rd calculation

            aux = Hs_to_Ds(Hs_interpolado, int_inv_Hs_interpol, z_eff_1, 1)
            output_th_dh[j] = Ds_to_obs_final(zs_model, aux, rd, 1)

            aux = Hs_to_Ds(Hs_interpolado, int_inv_Hs_interpol, z_eff_1, 2)
            output_th_dm[j] = Ds_to_obs_final(zs_model, aux, rd, 2)

        #DV
        output_th_dv = np.zeros(len(z_eff_2))
        for j in range(len(z_eff_2)): # For each datatype
            rd = r_drag(omega_m, H_0, wb_fid_2) #rd calculation
            aux = Hs_to_Ds(Hs_interpolado, int_inv_Hs_interpol, z_eff_2, 3)
            output_th_dv[j] = Ds_to_obs_final(zs_model, aux, rd, 3)
            
        #Chi square calculation for each datatype (i)

        # ARREGLAR ESTO: Es algo nuevo, no puedo separar el chi2 de Dh y Dm por la covarianza, como lo escribo??
        #chies_DESI_1 = chi2_sin_cov(output_th_dv,data_dv_rd,errors_dv_rd) 
        
        chies_DESI_2 = chi2_supernovae(muth, muobs, Cinv)
        chies_DESI = chies_DESI_1 + chies_DESI_2
        return chies_DESI

    if dataset_AGN != None:
        z_data, logFuv, eFuv, logFx, eFx  = dataset_AGN #Import the data

        if nuisance_2 == True: #Deprecated
            beta = 8.513
            ebeta = 0.437
            gamma = 0.622
            egamma = 0.014
        elif enlarged_errors == True:
            beta = 7.735
            ebeta = 0.6
            gamma = 0.648
            egamma = 0.007
        else: #Standard case
            beta = 7.735
            ebeta = 0.244
            gamma = 0.648
            egamma = 0.007

        DlH0_teo = zs_2_logDlH0(int_inv_Hs_interpol(z_data)*H_0,z_data)
        DlH0_obs =  np.log10(3.24) - 25 + (logFx - gamma * logFuv - beta) / (2*gamma - 2)

        df_dgamma =  (-logFx+beta+logFuv) / (2*(gamma-1)**2)
        eDlH0_cuad = (eFx**2 + gamma**2 * eFuv**2 + ebeta**2)/ (2*gamma - 2)**2 + (df_dgamma)**2 * egamma**2 #Square of the errors

        chi2_AGN = chi2_sin_cov(DlH0_teo, DlH0_obs, eDlH0_cuad)

    if H0_Riess == True:
        chi2_H0 = ((Hs_model[0]-73.48)/1.66)**2
    #print(chi2_SN + chi2_CC)
    return chi2_SN + chi2_CC + chi2_AGN + chi2_BAO + chi2_H0

def log_likelihood(*args, **kargs):  
    '''
    Return the log likelihood in terms of the chi square.
    '''
    return -0.5 * params_to_chi2(*args, **kargs)

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
    from data import read_data_pantheon_plus_shoes, read_data_pantheon_plus, \
                    read_data_pantheon, read_data_chronometers, read_data_BAO, read_data_AGN
    
    # Pantheon plus + SH0ES
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    ds_SN_plus_shoes = read_data_pantheon_plus_shoes('Pantheon+SH0ES.dat',
                                    'Pantheon+SH0ES_STAT+SYS.cov')

    # Pantheon plus
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    ds_SN_plus = read_data_pantheon_plus('Pantheon+SH0ES.dat',
                                'covmat_pantheon_plus_only.npz')

    # Pantheon
    os.chdir(path_git+'/fr_mcmc/source/Pantheon/')
    ds_SN = read_data_pantheon('lcparam_full_long_zhel.txt')

    # Cosmic Chronometers
    os.chdir(path_git+'/fr_mcmc/source/CC/')
    ds_CC = read_data_chronometers('chronometers_data.txt')

    # BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO/')
    ds_BAO = []
    files_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                    'BAO_data_dv.txt','BAO_data_H.txt']
    for i in range(5):
        aux = read_data_BAO(files_BAO[i])
        ds_BAO.append(aux)

    # AGN
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    ds_AGN = read_data_AGN('table3.dat')



    #%%
    chi2 = params_to_chi2([-19.37, 0.9, 70], 1.0, index=32,
                    dataset_SN_plus_shoes = ds_SN_plus_shoes,
                    dataset_SN_plus = ds_SN_plus,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    dataset_AGN = ds_AGN,
                    H0_Riess = True,
                    model = 'LCDM'
                    )
    print(chi2)