'''
Run MCMC analyses and calculations of the physical parameters of the models.

Parameter order in this file: Mabs,L_bar,b,H_0,n
'''

import numpy as np
import emcee
import yaml
from scipy.optimize import minimize, basinhopping

import os
import git

# Get the root directory of the Git repository
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

# Add necessary paths to sys.path
os.sys.path.extend([
    os.path.join(path_git, 'fr_mcmc'),
    os.path.join(path_git, 'fr_mcmc', 'plotting')
])
# Get the root directory of the Git repository
from utils.sampling import MCMC_sampler
from utils.data import (
    read_data_pantheon_plus_shoes, read_data_pantheon_plus, read_data_pantheon,
    read_data_chronometers, read_data_BAO, read_data_DESI, read_data_BAO_full, read_data_AGN
)
from utils.chi_square import log_likelihood
from utils.derived_parameters import derived_parameters

from config import cfg as config
import analysis

# Change to the mcmc directory
os.chdir(os.path.join(path_git, 'fr_mcmc', 'mcmc'))

def run():
    output_dir = config.OUTPUT_DIR
    model = config.MODEL
    fixed_params = config.FIXED_PARAMS # Fixed parameters
    index = config.LOG_LIKELIHOOD_INDEX
    num_params = int(str(index)[0])
    all_analytic = config.ALL_ANALYTIC

    witness_file = f'witness_{config.WITNESS_NUM}.txt'
    
    bnds = config.BOUNDS
    if model == 'LCDM':
        [omega_m_min, omega_m_max] = config.OMEGA_M_PRIOR
        [H0_min, H0_max] = config.H0_PRIOR

    elif model in ['GILA', 'BETA']:
        [L_bar_min, L_bar_max] = config.L_BAR_PRIOR
        [b_min, b_max] = config.B_PRIOR
        [H0_min, H0_max] = config.H0_PRIOR


    if config.USE_BAO or config.USE_DESI or config.USE_BAO_FULL:
        [bao_param_min, bao_param_max] = config.BAO_PARAM_PRIOR

    if config.USE_SN or config.USE_PPLUS or config.USE_PPLUS_SHOES:
        [M_min, M_max] = config.M_PRIOR

    #%% Import cosmological data
    path_data = os.path.join(path_git, 'fr_mcmc', 'source')
    
    datasets = []

    # Pantheon Plus + Shoes
    if config.USE_PPLUS_SHOES == True:
        os.chdir(os.path.join(path_data, 'Pantheon_plus_shoes'))

        ds_SN_plus_shoes = read_data_pantheon_plus_shoes('Pantheon+SH0ES.dat',
                                    'Pantheon+SH0ES_STAT+SYS.cov')
        datasets.append('_PPS')
    else:
        ds_SN_plus_shoes = None

    # Pantheon Plus
    if config.USE_PPLUS == True:
        os.chdir(os.path.join(path_data, 'Pantheon_plus_shoes'))

        ds_SN_plus = read_data_pantheon_plus('Pantheon+SH0ES.dat',
                                'covmat_pantheon_plus_only.npz')        
        datasets.append('_PP')
    else:
        ds_SN_plus = None

    # Supernovae type IA
    if config.USE_SN == True:
        os.chdir(os.path.join(path_data, 'Pantheon'))

        ds_SN = read_data_pantheon('lcparam_full_long_zhel.txt')
        datasets.append('_SN')
    else:
        ds_SN = None

    # Cosmic Chronometers
    if config.USE_CC == True:
        os.chdir(os.path.join(path_data, 'CC'))

        ds_CC = read_data_chronometers('chronometers_data.txt')
        datasets.append('_CC')
    else:
        ds_CC = None

    # BAO
    if config.USE_BAO == True:    
        os.chdir(os.path.join(path_data, 'BAO'))

        ds_BAO = []
        files_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                        'BAO_data_dv.txt','BAO_data_H.txt']
        for i in range(5):
            aux = read_data_BAO(files_BAO[i])
            ds_BAO.append(aux)
        datasets.append('_BAO')
    else:
        ds_BAO = None

    # DESI
    if config.USE_DESI == True:    
        os.chdir(os.path.join(path_data, 'DESI'))

        ds_DESI = read_data_DESI('DESI_data_dm_dh.txt','DESI_data_dv.txt')
        datasets.append('_DESI')
    else:
        ds_DESI = None

    # BAO full
    if config.USE_BAO_FULL == True:    
        os.chdir(os.path.join(path_data, 'BAO_full'))
        ds_BAO_full = read_data_BAO_full('BAO_full_1.csv','BAO_full_2.csv')
        datasets.append('_BAO_full')
    else:
        ds_BAO_full = None

    # AGN
    if config.USE_AGN == True:
        os.chdir(os.path.join(path_data, 'AGN'))
        ds_AGN = read_data_AGN('table3.dat')
        datasets.append('_AGN')
    else:
        ds_AGN = None

    # Riess H0
    if config.USE_H0 == True:
        H0_Riess = config.USE_H0
        datasets.append('_H0')
    else:
        H0_Riess = False

    datasets = str(''.join(datasets))

    # Define the log-likelihood distribution
    ll = lambda theta: log_likelihood(theta, fixed_params, 
                                        index=index,
                                        dataset_SN_plus_shoes = ds_SN_plus_shoes,
                                        dataset_SN_plus = ds_SN_plus,
                                        dataset_SN = ds_SN,
                                        dataset_CC = ds_CC,
                                        dataset_BAO = ds_BAO,
                                        dataset_DESI = ds_DESI,
                                        dataset_BAO_full = ds_BAO_full,
                                        dataset_AGN = ds_AGN,
                                        H0_Riess = H0_Riess,
                                        model = model,
                                        all_analytic = all_analytic
                                        )

    nll = lambda theta: -ll(theta) # negative log likelihood

    # Define the prior distribution
    def log_prior(theta, model):
        if model == 'LCDM':
            if index == 4:
                M, bao_param, omega_m, H0 = theta
                if (M_min < M < M_max and bao_param_min < bao_param < bao_param_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 31:
                M, omega_m, H0 = theta
                if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 32:
                bao_param, omega_m, H0 = theta
                if (bao_param_min < bao_param < bao_param_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                    return 0.0

        elif model in ['GILA', 'BETA']:
            if index == 5:
                M, bao_param, L_bar, b, H0 = theta
                if (M_min < M < M_max and bao_param_min < bao_param < bao_param_max and L_bar_min < L_bar < L_bar_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 41:
                M, bao_param, b, H0 = theta
                if (M_min < M < M_max and bao_param_min < bao_param < bao_param_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 42:
                M, L_bar, b, H0 = theta
                if (M_min < M < M_max and L_bar_min < L_bar < L_bar_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 43:
                bao_param, L_bar, b, H0 = theta
                if (bao_param_min < bao_param < bao_param_max and L_bar_min < L_bar < L_bar_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 31:
                M, b, H0 = theta
                if (M_min < M < M_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 32:
                bao_param, b, H0 = theta
                if (bao_param_min < bao_param < bao_param_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 2:
                b, H0 = theta
                if (b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
        return -np.inf
    
    # Define the posterior distribution
    def log_probability(theta):
        lp = log_prior(theta, model)
        if not np.isfinite(lp): # Maybe this condition is not necessary..
            return -np.inf
        return lp + ll(theta)

    filename = f'sample_{model}{datasets}_{num_params}params'

    #output_directory = os.path.join(path_global, output_dir, filename) #No se porque no anda..
    output_directory = path_global + output_dir + filename

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    filename_ml = f'maximun_likelihood_{model}{datasets}_{num_params}params'
    
    # If exist, import mean values of the free parameters. If not, calculate, save and load calculation.
    os.chdir(output_directory)
    if (os.path.exists(filename_ml + '.npz') == True):
        with np.load(filename_ml + '.npz') as data:
            sol = data['sol']
    else:
        print('Calculating maximum likelihood parameters ..')
        initial = np.array(config.GUEST)
        soln = minimize(nll, initial, options = {'eps': 0.01}, bounds = bnds)
        
        #minimizer_kwargs= dict(method='L-BFGS-B', bounds = bnds)
        #soln = basinhopping(nll, initial, minimizer_kwargs=minimizer_kwargs)
        np.savez(filename_ml, sol=soln.x)
        
        #np.savez(filename_ml, sol=initial) #Use ansatz as minimun
        with np.load(filename_ml + '.npz') as data:
            sol = data['sol']
    print(f'Maximun likelihood corresponds to the parameters: {sol}')

    # Define initial values of each chain using the minimun 
    # values of the chi-squared.
    pos = sol * (1 +  0.01 * np.random.randn(config.NUM_WALKERS, num_params))
    filename_h5 = filename + '.h5'

    MCMC_sampler(log_probability,pos, 
                filename = filename_h5,
                witness_file = witness_file,
                witness_freq = config.WITNESS_FREQ,
                max_samples = config.MAX_SAMPLES,
                save_path = output_directory)

    # If it corresponds, derive physical parameters

    if index in [41, 35]:
        os.chdir(output_directory)
 
        textfile_witness = open(witness_file,'a')
        textfile_witness.write('\n Initializing derivation of parameters..')
        textfile_witness.close()

        reader = emcee.backends.HDFBackend(filename_h5)
        nwalkers, ndim = reader.shape #Number of walkers and parameters

        # Hardcode definition of burnin and thin
        samples = reader.get_chain()
        burnin= int(0.2*len(samples[:,0])) # Burnin 20% 
        thin = 1

        samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

        textfile_witness = open(witness_file,'a')
        textfile_witness.write('\n Number of effective steps: {}'.format(len(samples))) 
        textfile_witness.write(('\n Estimated time: {} min'.format(len(samples)/60)))
        textfile_witness.close()

        #new_samples = derived_parameters(reader,discard=burnin,thin=thin,model=model)
        #np.savez(filename+'_deriv', new_samples=new_samples)

        textfile_witness = open(witness_file,'a')
        textfile_witness.write('\n Done!')
        textfile_witness.close()

        # Print the output
        #with np.load(filename+'_deriv.npz') as data:
        #    ns = data['new_samples']        

    # Plot the results
    analysis.run(filename)


if __name__ == "__main__":
    run()
