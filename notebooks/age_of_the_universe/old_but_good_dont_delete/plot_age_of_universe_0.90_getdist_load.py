
#For fancy plots
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

#Import standard libraries
import numpy as np
import time
import seaborn as sns
import pandas as pd

from scipy.interpolate import interp1d
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps, solve_ivp
from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000 #km/s

import getdist
getdist.chains.print_load_details = False

# import repo functions:
import sys, os
import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_data = os.path.join(os.path.dirname(path_git), 'GILA-output', 'paper')
path_figures = os.path.join(path_git, 'notebooks', 'figures')
os.chdir(os.path.join(path_git, 'fr_mcmc', 'utils'))
#from supernovae import aparent_magnitude_th, chi2_supernovae
#from constants import OMEGA_R_0, LAMBDA, L, KAPPA


import numpy as np

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)

os.chdir(path_git)
os.sys.path.append('./fr_mcmc/utils/')

#from LambdaCDM import H_LCDM

from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000; #kilometers/seconds
#Parameters order: Mabs,omega_m,b,H_0,n

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000

#Fix params
omega_r = 2.47e-5 
L_bar = 0.90
#M_abs = -19.3
#M_abs = -19.321 #GILA

Gyr_to_second = int(3.1536e16)
Mpc_to_km = int(3.0857e19)
inv_Hub_to_Gyr = Mpc_to_km/Gyr_to_second

z = np.linspace(0, 1000, int(10e3))
aou_threshold = 12.7

#GILA MODEL
#r_min = 3
#s_min = 1

#r = 3; s = 1 #Does not work :(
#H0_values = np.linspace(60,80,25)[::-1] 
#beta_values = np.linspace(0.2,8,25)

#r = 8; s = 1 #Does not work :(
#H0_values = np.linspace(60,80,25)[::-1] 
#beta_values = np.linspace(1,80,25)

#r = 3; s = 2 #It works, not with AoU
#H0_values = np.linspace(64,74,25)[::-1] 
#beta_values = np.linspace(0.8,4,25)

#r = 3; s = 4 #It works, not with AoU
#H0_values = np.linspace(64,78,25)[::-1] 
#beta_values = np.linspace(0.3,8,25)

r = 3; s = 5 #It worked, even with AoU!
H0_values = np.linspace(67,76,50)[::-1] 
beta_values = np.linspace(0.1,7,50)
Mabs_values = np.linspace(-19.5,-19.1,10)

#r = 3; s = 6 #It worked, even with AoU!
#H0_values = np.linspace(64,78,25)[::-1] 
#beta_values = np.linspace(0.3,4,25)



import numpy as np
import getdist
import getdist.plots as gdplt
import matplotlib.pyplot as plt

# Example: Define a 3D probability grid (replace with your actual posterior)
#Nx, Ny, Nz = 50, 50, 50  # Grid size
#x_vals = np.linspace(67,76,50)[::-1]  #np.linspace(-3, 3, Nx)
#y_vals = np.linspace(0.1,7,50) #np.linspace(-3, 3, Ny)
#z_vals = np.linspace(-3, 3, Nz)

# Example posterior: A 3D Gaussian distribution
#X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
#p3d = np.exp(-0.5 * (X**2 + Y**2 + Z**2))  # Gaussian posterior

x_vals = H0_values 
y_vals = beta_values 
z_vals = Mabs_values

matrix_gila_chi2_1 = np.load(os.path.join(path_data, 'matrix_gila_chi2_1.npy'))
matrix_gila_chi2_2 = np.load(os.path.join(path_data, 'matrix_gila_chi2_2.npy'))
matrix_gila_chi2_3 = np.load(os.path.join(path_data, 'matrix_gila_chi2_3.npy'))
matrix_gila_chi2_4 = np.load(os.path.join(path_data, 'matrix_gila_chi2_4.npy'))


p3d = np.exp(matrix_gila_chi2_1)
print(p3d)

# Normalize the distribution
p3d /= np.sum(p3d)
print(p3d.shape)
# Convert 3D posterior into samples
flattened_probs = p3d.flatten()
grid_points = np.array(np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')).reshape(3, -1).T

# Sample points from the distribution
num_samples = 10000  # Number of samples you want
sample_indices = np.random.choice(len(flattened_probs), size=num_samples, p=flattened_probs)
samples = grid_points[sample_indices]

# Convert to GetDist format
gd_samples = getdist.mcsamples.MCSamples(samples=samples, names=['x', 'y', 'z'], labels=['X', 'Y', 'Z'])

# Plot using GetDist
g = gdplt.get_subplot_plotter()
g.triangle_plot([gd_samples], filled=True)
plt.show()





