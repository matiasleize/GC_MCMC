# Initial setup:
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#%load_ext autoreload
#%autoreload 1

#For fancy plots
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# import libraries:
import sys, os
import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from change_of_parameters import omega_CDM_to_luisa

sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'../..')))
#path_git = git.Repo('.', search_parent_directories=True).working_tree_dir


from getdist import plots
import getdist
from getdist import plots, MCSamples, loadMCSamples
getdist.chains.print_load_details = False
import scipy
import numpy as np

from matplotlib import pyplot as plt
from scipy.linalg import sqrtm

from seaborn import heatmap
import pandas as pd
import emcee

discard=0
thin=1
root_dic = '/home/mleize/Documents/Repos/GILA-output/LCDM/'

# Get the samples:
samples_lcdm_1 = emcee.backends.HDFBackend(root_dic + 'sample_LCDM_PPS_CC_3params/sample_LCDM_PPS_CC_3params.h5')
samples_lcdm_2 = emcee.backends.HDFBackend(root_dic + 'sample_LCDM_PPS_CC_BAO_4params/sample_LCDM_PPS_CC_BAO_4params.h5')
samples_lcdm_3 = emcee.backends.HDFBackend(root_dic + 'sample_LCDM_PPS_CC_DESI_4params/sample_LCDM_PPS_CC_DESI_4params.h5')

#nwalkers, ndim = reader_lcdm.shape #Number of walkers and parameters
flat_samples_1 = samples_lcdm_1.get_chain(discard=discard, flat=True, thin=thin)
flat_samples_2 = samples_lcdm_2.get_chain(discard=discard, flat=True, thin=thin)
flat_samples_3 = samples_lcdm_3.get_chain(discard=discard, flat=True, thin=thin)

omega_r = 2.47e-5 
L_bar = 0.90
names_LCDM = ['M_{{abs}}','H_0','\omega_m','\Omega_{{m}}^{{LCDM}}','\Omega_{{\\Lambda}}^{{LCDM}}']
labels_LCDM = names_LCDM

#LCDM MODEL
M_abs = flat_samples_1[:,0]
omega_m = flat_samples_1[:,1]
H0 = flat_samples_1[:,2]

Omega_r_lcdm = omega_r / (H0/100)**2
Omega_m_lcdm = omega_m / (H0/100)**2
Omega_L_lcdm = 1 - Omega_m_lcdm - Omega_r_lcdm

chains_lcdm_1 = np.zeros((len(M_abs),5))
chains_lcdm_1[:,0] = M_abs
chains_lcdm_1[:,1] = H0
chains_lcdm_1[:,2] = omega_m
chains_lcdm_1[:,3] = Omega_m_lcdm
chains_lcdm_1[:,4] = Omega_L_lcdm


samples1 = MCSamples(samples=chains_lcdm_1, names=names_LCDM, labels=names_LCDM)
samples1 = samples1.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
			settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
			'smooth_scale_1D':0.3})


#LCDM MODEL
M_abs = flat_samples_2[:,0]
bao_param = flat_samples_2[:,1]
omega_m = flat_samples_2[:,2]
H0 = flat_samples_2[:,3]

Omega_r_lcdm = omega_r / (H0/100)**2
Omega_m_lcdm = omega_m / (H0/100)**2
Omega_L_lcdm = 1 - Omega_m_lcdm - Omega_r_lcdm

chains_lcdm_2 = np.zeros((len(M_abs),5))
chains_lcdm_2[:,0] = M_abs
chains_lcdm_2[:,1] = H0
chains_lcdm_2[:,2] = omega_m
chains_lcdm_2[:,3] = Omega_m_lcdm
chains_lcdm_2[:,4] = Omega_L_lcdm

samples2 = MCSamples(samples=chains_lcdm_2, names=names_LCDM, labels=names_LCDM)
samples2 = samples2.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
			settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
			'smooth_scale_1D':0.3})

#LCDM MODEL
M_abs = flat_samples_3[:,0]
bao_param = flat_samples_3[:,1]
omega_m = flat_samples_3[:,2]
H0 = flat_samples_3[:,3]

Omega_r_lcdm = omega_r / (H0/100)**2
Omega_m_lcdm = omega_m / (H0/100)**2
Omega_L_lcdm = 1 - Omega_m_lcdm - Omega_r_lcdm

chains_lcdm_3 = np.zeros((len(M_abs),5))
chains_lcdm_3[:,0] = M_abs
chains_lcdm_3[:,1] = H0
chains_lcdm_3[:,2] = omega_m
chains_lcdm_3[:,3] = Omega_m_lcdm
chains_lcdm_3[:,4] = Omega_L_lcdm

samples3 = MCSamples(samples=chains_lcdm_3, names=names_LCDM, labels=names_LCDM)
samples3 = samples3.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
			settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
			'smooth_scale_1D':0.3})


g = plots.get_subplot_plotter()
g.settings.legend_fontsize = 18
g.settings.axes_fontsize = 15
g.settings.axes_labelsize = 18
g.triangle_plot([samples1, samples2, samples3],
				filled=True, params=names_LCDM,
				#contour_colors=color,
				contour_lws=1,
				legend_labels=['CC+PPS','CC+PPS+BAO','CC+PPS+DESI'])
g.export('/home/mleize/Documents/Repos/GILA_MCMC/notebooks/figures/triangle_plot_lcdm.pdf')