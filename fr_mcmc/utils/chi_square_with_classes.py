"""
Definition of the log likelihood distribution and the chi_square in terms of
the parameters of the model and the datasets which are use. 
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir

os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from change_of_parameters import omega_CDM_to_luisa, omega_luisa_to_CDM
from solve_sys import Hubble_th
from supernovae import aparent_magnitude_th, chi2_supernovae
from BAO import r_drag, Hs_to_Ds, Ds_to_obs_final
from AGN import zs_2_logDlH0

class data_part():
    def __init__(self):
        pass
    def BAO(ds_BAO):
        pass
    def CC(ds_CC):
        pass
    def Pantheon(ds_Pantheon, cov_mat):
        pass
    def Pantheon_Plus(ds_Pantheon_Plus, cov_mat):
        pass
    def PPS(ds_PPS, cov_mat):
        pass

class theo_part:
    def __init__(self,z,hubble,title):
        self.z = z
        self.hubble = hubble
    
    def hubble_parameter(self):
        return self.z, self.hubble
    
    def int_inv_Hs_interpol(self):
        zs_model, Hs_model = theo_part.hubble_parameter()
        Hs_model = self.hubble
        int_inv_Hs = cumtrapz(Hs_model**(-1), zs_model, initial=0)
        int_inv_Hs_interpol = interp1d(zs_model, int_inv_Hs)
        return int_inv_Hs_interpol
    
    def luminosity_distance_sn(self,zhel,zcmb):
        dc_int =  c_light_km * theo_part.int_inv_Hs_interpol(zcmb)
        d_L = (1 + zhel) * dc_int
        return d_L
    
    def distance_modulus_sn(self,*args):
        '''
        Given an interpolate function of 1/H and arrays for zcmb and zhel,
        this function returns the theoretical expression for the distance modulus (mu)
        muth = 25 + 5 * log_{10}(d_L),
        where d_L = (c/H_0) (1+z) int(dz'/E(z')).
        '''
        d_L = theo_part.luminosity_distance(*args)
        muth = 25.0 + 5.0 * np.log10(d_L)
        return muth
    
    def DA(self):
        pass
    
    def DM(self):
        pass

    def DH(self):
        pass


class calc_chi_squared:
    def __init__(self,teo,data,errors):
        self.teo = teo
        self.data = data
        self.errors = errors

    def without_cov(self):
        '''
        Calculate chi square assuming no correlation.

        teo (array): Theoretical prediction of the model.
        data (array): Observational data to compare with the model.
        errors_cuad (array): The square of the errors of the data.

        '''

        errors_cuad = self.errors

        chi2 = np.sum((self.data-self.teo)**2/errors_cuad)
        return chi2

    def with_cov(self):
        '''This function estimates the value of the statistic chi squared
        for the Supernovae data.'''

        C_inv = self.errors
        muth = self.teo
        muobs = self.data

        deltamu = muth - muobs #row vector
        transp = np.transpose(deltamu) #column vector
        aux = np.dot(C_inv,transp) #column vector
        chi2 = np.dot(deltamu,aux) #scalar
        return chi2


class total_chi_squared:
    def __init__(self,chi_sq_array):
        self.chi_sq_array = chi_sq_array
    
    def sum_all(self):
        np.sum(self.chi_sq_array)
