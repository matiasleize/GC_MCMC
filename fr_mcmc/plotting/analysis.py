'''
In this file we make the plots of the Markov chains, the corner plots of the free parameters of the model and report the confidence intervals.
All this information will be stored at the output directory (called '/results/').
'''


import numpy as np; #np.random.seed(42)
import emcee
from matplotlib import pyplot as plt
import yaml

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)

os.chdir(path_git); os.sys.path.append('./fr_mcmc/')
from utils.plotter import Plotter
from config import cfg as config

os.chdir(path_git + '/fr_mcmc/plotting/')

def parameters_labels(index):
    if index == 5:
        return ['$M_{abs}$', '$\L$', r'$\beta$', '$H_{0}$', r'$\omega_m$']
    if index == 4:
        return ['$M_{abs}$', '$\L$', r'$\beta$', '$H_{0}$']
    if index == 41:
        return ['$M_{abs}$', r'$\beta$', '$H_{0}$', r'$\omega_m$']
    elif index == 31:
        return ['$\L$', r'$\beta$', '$H_{0}$']
    elif index == 32:
        return ['$M_{abs}$', '$\L$', '$H_{0}$']
    elif index == 33:
        return ['$M_{abs}$', '$\L$', r'$\beta$']
    elif index == 34:
        return ['$M_{abs}$', r'$\beta$', '$H_{0}$']
    elif index == 35:
        return ['$M_{abs}$', '$H_{0}$', r'$\omega_m$']
    elif index == 21:
        return ['$\L$', r'$\beta$']
    elif index == 22:
        return ['$\L$', '$H_{0}$']
    elif index == 23:
        return ['$M_{abs}$', '$\L$']
    elif index == 1:
        return ['$\L$'] #list or str?

def run(filename):
    model = config.MODEL
    output_dir = config.OUTPUT_DIR
    index = config.LOG_LIKELIHOOD_INDEX

    output_path = path_datos_global + output_dir + filename
    os.chdir(output_path)

    parameters_label = parameters_labels(config.LOG_LIKELIHOOD_INDEX)
    #if model == 'LCDM':
    reader = emcee.backends.HDFBackend(filename + '.h5')
    samples = reader.get_chain()
    #if index == 41:
        #aux =0.9999 * samples[:,:,3]/samples[:,:,3] + samples[:,:,3] * 10**(-5)
        #samples[:,:,3] = aux
        #print(samples[:,:,3])
        #print(samples[:,0,0]) #1) Num of step
        #print(samples[0,:,0]) #2) Num of walker
        #print(samples[0,0,:]) #3) Num of parameter
    burnin= burnin=int(0.2*len(samples[:,0])); thin=1
    analisis = Plotter(reader, parameters_label, 'Title')

    results_dir = '/results'
    if not os.path.exists(output_path + results_dir):
            os.mkdir(output_path + results_dir)
 
    analisis.plot_contours(discard=burnin, thin=thin)
    plt.savefig(output_path + results_dir + '/cornerplot.png')
    plt.close()
    
    analisis.plot_chains()
    plt.savefig(output_path + results_dir + '/chains.png')
    plt.close()

    if index == 41:
        with np.load(filename + '_deriv.npz') as data:
            ns = data['new_samples']
        parameters_label_derived = [r'$H_{0}$', r'$\Omega_m^{LCDM}$', r'$\Omega_m^{GILA-\beta}$', r'$\Omega_{\Lambda}^{LCDM}$']
        analisis = Plotter(ns, parameters_label_derived, '')
        burnin = 0 # already has the burnin
        thin = 1

        results_dir = '/results_derivs'
        if not os.path.exists(output_path + results_dir):
                os.mkdir(output_path + results_dir)
    
        analisis.plot_contours(discard=burnin, thin=thin)
        plt.savefig(output_path + results_dir + '/cornerplot.png')
        plt.close()
        
        analisis.plot_chains_derivs()
        plt.savefig(output_path + results_dir + '/chains.png')
        plt.close()

    if index == 35:
        with np.load(filename + '_deriv.npz') as data:
            ns = data['new_samples']
        parameters_label_derived = ['$H_{0}$', r'$\Omega_m^{LCDM}$', r'$\Omega_{\Lambda}^{LCDM}$']
        analisis = Plotter(ns, parameters_label_derived, '')
        burnin = 0 # already has the burnin
        thin = 1

        results_dir = '/results_derivs'
        if not os.path.exists(output_path + results_dir):
                os.mkdir(output_path + results_dir)
    
        analisis.plot_contours(discard=burnin, thin=thin)
        plt.savefig(output_path + results_dir + '/cornerplot.png')
        plt.close()
        
        analisis.plot_chains_derivs()
        plt.savefig(output_path + results_dir + '/chains.png')
        plt.close()

    analisis.report_intervals(discard=burnin, thin=thin, save_path = output_path + results_dir)
    textfile_witness = open(output_path + results_dir + '/metadata.dat','w')
    textfile_witness.write('{}'.format(config))

if __name__ == "__main__":
    run('sample_LCDM_SN_2params')


