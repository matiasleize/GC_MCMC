"""
Calculate the derivate parameters. IMPORTANT: It Doesn't work for all the indeces yet.
"""

from numba import jit
import numpy as np

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
from config import cfg as config
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
#from solve_sys import Hubble_th
from change_of_parameters import omega_CDM_to_luisa
from constants import OMEGA_R_0

@jit
def derived_parameters(sampler,discard, thin, model):
	'''Convert LCDM, BETA or GILA chains into physical chains (for Omega_m and H_0 parameters).'''

	flat_samples = sampler.get_chain(discard=discard, flat=True, thin=thin)
	len_chain = flat_samples.shape[0]
	new_samples = np.full_like(flat_samples,1)

	if model == 'LCDM':
		for i in range(len_chain):
			if len(flat_samples[0,:])==3:
				M_abs = flat_samples[i,0]
				H_0 = flat_samples[i,1]
				omega_m = flat_samples[i,2]

				h = H_0/100
				Omega_r_lcdm = OMEGA_R_0 / h**2
				Omega_m_lcdm = omega_m / h**2
				Omega_Lambda_lcdm = 1 - Omega_m_lcdm - Omega_r_lcdm			

				new_samples[i,0] = H_0
				new_samples[i,1] = Omega_m_lcdm
				new_samples[i,2] = Omega_Lambda_lcdm


	elif model == 'GILA' or model == 'BETA':
		L_bar = config.FIXED_PARAMS # Fixed parameters
		for i in range(len_chain):
			if len(flat_samples[0,:])==4:
				M_abs = flat_samples[i,0]
				beta = flat_samples[i,1]
				H_0 = flat_samples[i,2]
				omega_m = flat_samples[i,3]

				h = H_0/100
				Omega_r_lcdm = OMEGA_R_0 / h**2
				Omega_m_lcdm = omega_m / h**2
				Omega_m_GILA = omega_CDM_to_luisa(beta,L_bar,H_0,Omega_m_lcdm,model)
				Omega_Lambda_lcdm = 1 - Omega_m_lcdm - Omega_r_lcdm			

				new_samples[i,0] = H_0
				new_samples[i,1] = Omega_m_lcdm
				new_samples[i,2] = Omega_m_GILA
				new_samples[i,3] = Omega_Lambda_lcdm

	return new_samples
