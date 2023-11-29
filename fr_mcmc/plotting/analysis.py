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
        return ['$M_{abs}$', '$\L$', r'$\beta$', '$H_{0}$', r'$\Omega_m$']
    if index == 4:
        return ['$M_{abs}$', '$\L$', r'$\beta$', '$H_{0}$']
    if index == 41:
        return ['$M_{abs}$', r'$\beta$', '$H_{0}$', r'$\Omega_m$']
    elif index == 31:
        return ['$\L$', r'$\beta$', '$H_{0}$']
    elif index == 32:
        return ['$M_{abs}$', '$\L$', '$H_{0}$']
    elif index == 33:
        return ['$M_{abs}$', '$\L$', r'$\beta$']
    elif index == 34:
        return ['$M_{abs}$', r'$\beta$', '$H_{0}$']
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
    if index == 41:
        aux =0.9999 * samples[:,:,3]/samples[:,:,3] + samples[:,:,3] * 10**(-5)
        samples[:,:,3] = aux
        #print(samples[:,:,3])
        #print(samples[:,0,0]) #1) Num de paso
        #print(samples[0,:,0]) #2) Num de caminante
        #print(samples[0,0,:]) #3) Num de parametro
    burnin= burnin=int(0.2*len(samples[:,0])); thin=1
    analisis = Plotter(reader, parameters_label, 'Titulo')

    results_dir = '/results'
    if not os.path.exists(output_path + results_dir):
            os.mkdir(output_path + results_dir)
 
    analisis.graficar_contornos(discard=burnin, thin=thin)
    plt.savefig(output_path + results_dir + '/cornerplot.png')
    plt.close()
    analisis.graficar_cadenas()


    if index == 41:
        with np.load(filename + '_deriv.npz') as data:
            ns = data['new_samples']
        analisis = Plotter(ns, parameters_label, '')
        burnin = 0 # already has the burnin
        thin = 1

        results_dir = '/results_derivs'
        if not os.path.exists(output_path + results_dir):
                os.mkdir(output_path + results_dir)
    
        analisis.graficar_contornos(discard=burnin, thin=thin)
        plt.savefig(output_path + results_dir + '/cornerplot.png')
        plt.close()
        analisis.graficar_cadenas_derivs()
        
    plt.savefig(output_path + results_dir + '/chains.png')
    plt.close()


    analisis.reportar_intervalos(discard=burnin, thin=thin, save_path = output_path + results_dir)
    textfile_witness = open(output_path + results_dir + '/metadata.dat','w')
    textfile_witness.write('{}'.format(config))

if __name__ == "__main__":
    run('sample_LCDM_SN_2params')


