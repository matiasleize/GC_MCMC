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
from change_of_parameters import omega_luisa_to_CDM



@jit
def derived_parameters(sampler,discard, thin,model='EXP'):
	'''Convert LCDM chains into physical chains (for Omega_m and H_0 parameters).'''

	L_bar = config.FIXED_PARAMS # Fixed parameters


	flat_samples = sampler.get_chain(discard=discard, flat=True, thin=thin)
	len_chain=flat_samples.shape[0]
	new_samples = np.full_like(flat_samples,1)
	for i in range(len_chain):
		if len(flat_samples[0,:])==4:
			beta = flat_samples[i,1]
			H_0 = flat_samples[i,2]
			omega_m_luisa = flat_samples[i,3]
			aux = 0.9999 + 10**(-5) * omega_m_luisa
			omega_m_lcdm = omega_luisa_to_CDM(beta,L_bar,H_0,aux)
			new_samples[i,3] = omega_m_lcdm
	return new_samples
